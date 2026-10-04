#include "server-disk-cache.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <unistd.h> // environ

#define DSK_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, "disk_cache", __VA_ARGS__)
#define DSK_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, "disk_cache", __VA_ARGS__)

namespace fs = std::filesystem;

static constexpr uint32_t DISK_CACHE_VERSION = 1;
static constexpr uint32_t AUX_MAGIC          = 0x58554144; // "DAUX"

static int64_t now_s() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string random_id() {
    static std::mt19937_64 rng(std::random_device{}());
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) rng());
    return buf;
}

static uint64_t file_size_or_0(const std::string & p) {
    std::error_code ec;
    const auto n = fs::file_size(p, ec);
    return ec ? 0 : (uint64_t) n;
}

// estimated seconds to re-process a prompt of n tokens from scratch: linear MLP/recurrent part + quadratic
// attention part, fitted to s1 Flash-Next brain prefills (27k -> 55 s, 139k -> 468 s); only the order matters
static double prefill_seconds(size_t n) {
    const double x = (double) n;
    return x / 520.0 + 1.0e-8 * x * x;
}

// how much keeping an entry is worth per byte: re-prefill time it saves, per GiB, decayed by idle time and
// boosted by reuse. Long chats save more time per byte (attention grows faster than the state), so they stay.
static double entry_value(const server_disk_cache_entry & e, int64_t t_now) {
    const double gib      = std::max<double>(1e-3, e.bytes / (1024.0 * 1024.0 * 1024.0));
    const double idle_d   = std::max<double>(0.0, (t_now - e.t_used) / 86400.0);
    const double recency  = 1.0 / (1.0 + idle_d);
    const double reuse    = 1.0 + 0.5 * std::min<int32_t>(e.hits, 4);
    return prefill_seconds(e.tokens.size()) / gib * recency * reuse;
}

// little-endian POD stream helpers for the .aux file
template <typename T> static void put(std::ofstream & f, const T & v) {
    f.write(reinterpret_cast<const char *>(&v), sizeof(T));
}
static void put_vec(std::ofstream & f, const std::vector<uint8_t> & v) {
    put<uint64_t>(f, v.size());
    if (!v.empty()) {
        f.write(reinterpret_cast<const char *>(v.data()), v.size());
    }
}
template <typename T> static bool get(std::ifstream & f, T & v) {
    return (bool) f.read(reinterpret_cast<char *>(&v), sizeof(T));
}
static bool get_vec(std::ifstream & f, std::vector<uint8_t> & v) {
    uint64_t n = 0;
    if (!get(f, n) || n > (1ull << 33)) { // 8 GiB: far above one recurrent state, below a corrupt length
        return false;
    }
    v.resize(n);
    return n == 0 || (bool) f.read(reinterpret_cast<char *>(v.data()), n);
}

std::string server_disk_cache_fingerprint(const common_params & params, int32_t n_ctx) {
    std::error_code ec;
    std::string exe_id;
    {
        const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
        if (!ec) {
            const auto sz = fs::file_size(exe, ec);
            const auto mt = fs::last_write_time(exe, ec).time_since_epoch().count();
            exe_id = std::to_string(sz) + ":" + std::to_string(mt);
        }
    }
    // LLAMA_* / GGML_* environment knobs (QSA, KQ_VIS, int8, ...) select graph and state variants under one binary
    std::vector<std::string> knobs;
    for (char ** ep = environ; ep && *ep; ep++) {
        const std::string kv = *ep;
        if (kv.rfind("LLAMA_", 0) == 0 || kv.rfind("GGML_", 0) == 0) {
            if (kv.rfind("LLAMA_ARG_", 0) != 0) {
                knobs.push_back(kv);
            }
        }
    }
    // the command line (rope scaling, QSA ratio, ... change the stored K/V) minus the disk cache's own options
    {
        std::ifstream fc("/proc/self/cmdline", std::ios::binary);
        std::string arg;
        bool skip_value = false;
        while (std::getline(fc, arg, '\0')) {
            if (skip_value) {
                skip_value = false;
                continue;
            }
            if (arg.rfind("--cache-disk", 0) == 0) {
                skip_value = arg.find('=') == std::string::npos;
                continue;
            }
            knobs.push_back("argv:" + arg);
        }
    }
    std::sort(knobs.begin(), knobs.end());
    std::string env_id;
    for (const auto & kv : knobs) {
        env_id += kv + ";";
    }
    const std::string s = string_format("v%u|%s|%llu|%s|%llu|ctx%d|k%d|v%d|dk%d|dv%d|%s|%s|%s",
        DISK_CACHE_VERSION,
        params.model.path.c_str(), (unsigned long long) file_size_or_0(params.model.path),
        params.speculative.draft.mparams.path.c_str(), (unsigned long long) file_size_or_0(params.speculative.draft.mparams.path),
        n_ctx, (int) params.cache_type_k, (int) params.cache_type_v,
        (int) params.speculative.draft.cache_type_k, (int) params.speculative.draft.cache_type_v,
        params.mmproj.path.c_str(), exe_id.c_str(), env_id.c_str());
    return string_format("%016llx", (unsigned long long) std::hash<std::string>{}(s)) + " " + s;
}

server_disk_cache::server_disk_cache(std::string dir, uint64_t quota_bytes, int32_t min_tokens, int32_t ttl_days,
                                     std::string fingerprint, bool has_mtmd)
    : dir(std::move(dir)), quota(quota_bytes), n_min_tokens(min_tokens), ttl_days(ttl_days),
      fingerprint(std::move(fingerprint)), has_mtmd(has_mtmd) {}

std::string server_disk_cache::path(const std::string & id, const char * ext) const {
    return (fs::path(dir) / (id + ext)).string();
}

void server_disk_cache::remove_files(const std::string & id) const {
    std::error_code ec;
    for (const char * ext : { ".meta", ".tok", ".tgt", ".dft", ".aux", ".meta.tmp", ".tok.tmp", ".aux.tmp" }) {
        fs::remove(path(id, ext), ec);
    }
}

bool server_disk_cache::write_meta(const server_disk_cache_entry & e) const {
    const json j = {
        { "version",     DISK_CACHE_VERSION },
        { "fingerprint", fingerprint        },
        { "n_tokens",    e.tokens.size()    },
        { "bytes",       e.bytes            },
        { "t_saved",     e.t_saved          },
        { "t_used",      e.t_used           },
        { "hits",        e.hits             },
        { "has_dft",     e.has_dft          },
    };
    const std::string tmp = path(e.id, ".meta.tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << j.dump(1);
        if (!f) {
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, path(e.id, ".meta"), ec);
    return !ec;
}

uint64_t server_disk_cache::total_bytes() const {
    uint64_t res = 0;
    for (const auto & e : entries) {
        res += e.bytes;
    }
    return res;
}

void server_disk_cache::erase(size_t idx) {
    remove_files(entries[idx].id);
    entries.erase(entries.begin() + idx);
}

void server_disk_cache::init() {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        DSK_WRN("cannot create %s: %s, disk cache disabled\n", dir.c_str(), ec.message().c_str());
        quota = 0;
        return;
    }

    std::set<std::string> ids;
    for (const auto & de : fs::directory_iterator(dir, ec)) {
        const std::string name = de.path().filename().string();
        const auto dot = name.find('.');
        if (dot == 16) {
            ids.insert(name.substr(0, 16));
        }
    }

    size_t n_dropped = 0;
    for (const auto & id : ids) {
        server_disk_cache_entry e;
        e.id = id;
        bool ok = false;
        try {
            std::ifstream fm(path(id, ".meta"));
            if (fm) {
                const std::string text((std::istreambuf_iterator<char>(fm)), std::istreambuf_iterator<char>());
                const json j = json::parse(text);
                if (j.value("version", 0u) == DISK_CACHE_VERSION && j.value("fingerprint", std::string()) == fingerprint) {
                    e.bytes   = j.at("bytes").get<uint64_t>();
                    e.t_saved = j.at("t_saved").get<int64_t>();
                    e.t_used  = j.at("t_used").get<int64_t>();
                    e.hits    = j.at("hits").get<int32_t>();
                    e.has_dft = j.at("has_dft").get<bool>();

                    std::ifstream ft(path(id, ".tok"), std::ios::binary);
                    std::vector<char> raw((std::istreambuf_iterator<char>(ft)), std::istreambuf_iterator<char>());
                    if (!raw.empty() && raw.size() % sizeof(llama_token) == 0) {
                        llama_tokens packed(raw.size() / sizeof(llama_token));
                        memcpy(packed.data(), raw.data(), raw.size());
                        e.tokens = server_tokens::deserialize(packed, has_mtmd);
                        ok = e.tokens.size() == j.at("n_tokens").get<size_t>() &&
                             file_size_or_0(path(id, ".tgt")) > 0 &&
                             (!e.has_dft || file_size_or_0(path(id, ".dft")) > 0);
                    }
                }
            }
        } catch (const std::exception & err) {
            DSK_WRN("entry %s unreadable: %s\n", id.c_str(), err.what());
            ok = false;
        }
        if (ok) {
            entries.push_back(std::move(e));
        } else {
            remove_files(id);
            n_dropped++;
        }
    }

    enforce("");

    DSK_INF("dir %s: %zu entries, %.2f GiB (quota %.0f GiB, min %d tokens, ttl %d days), %zu stale/foreign dropped\n",
            dir.c_str(), entries.size(), total_bytes() / (1024.0 * 1024.0 * 1024.0), quota / (1024.0 * 1024.0 * 1024.0),
            n_min_tokens, ttl_days, n_dropped);
}

void server_disk_cache::enforce(const std::string & protect) {
    const int64_t t_now = now_s();

    if (ttl_days > 0) {
        for (size_t i = 0; i < entries.size();) {
            if (entries[i].id != protect && t_now - entries[i].t_used > (int64_t) ttl_days * 86400) {
                DSK_INF("expired %s (%zu tokens, idle %.1f days)\n", entries[i].id.c_str(), entries[i].tokens.size(),
                        (t_now - entries[i].t_used) / 86400.0);
                erase(i);
                n_evict++;
            } else {
                i++;
            }
        }
    }

    while (total_bytes() > quota && !entries.empty()) {
        size_t victim = SIZE_MAX;
        double v_min  = 0.0;
        for (size_t i = 0; i < entries.size(); i++) {
            if (entries[i].id == protect) {
                continue;
            }
            const double v = entry_value(entries[i], t_now);
            if (victim == SIZE_MAX || v < v_min) {
                victim = i;
                v_min  = v;
            }
        }
        if (victim == SIZE_MAX) {
            break;
        }
        DSK_INF("quota: evicting %s (%zu tokens, %.2f GiB, value %.1f s/GiB)\n", entries[victim].id.c_str(),
                entries[victim].tokens.size(), entries[victim].bytes / (1024.0 * 1024.0 * 1024.0), v_min);
        erase(victim);
        n_evict++;
    }
}

bool server_disk_cache::save(llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, int32_t id_slot,
                             const server_prompt & prompt) {
    const size_t n = prompt.tokens.size();
    if (quota == 0 || n < (size_t) n_min_tokens) {
        return false;
    }

    // already stored: exactly this prompt (a longer entry does not count: its recurrent state is at its own end,
    // so this prefix could only come back through a checkpoint)
    for (auto & e : entries) {
        if (e.tokens.size() == n && e.tokens.get_common_prefix(prompt.tokens) == n) {
            e.t_used = now_s();
            write_meta(e);
            return true;
        }
    }

    for (const auto & ckpt : prompt.checkpoints) {
        if ((ckpt.flags_tgt | ckpt.flags_dft) & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
            DSK_WRN("%s", "a checkpoint lives on the device, not saving\n");
            return false;
        }
    }

    const int64_t t_start = ggml_time_us();

    server_disk_cache_entry e;
    e.id      = random_id();
    e.has_dft = ctx_dft != nullptr;

    bool ok = true;
    try {
        // tokens
        {
            const std::vector<char> packed = prompt.tokens.serialize();
            std::ofstream f(path(e.id, ".tok"), std::ios::binary | std::ios::trunc);
            f.write(packed.data(), packed.size());
            ok = ok && (bool) f;
        }
        // checkpoints + speculative state
        if (ok) {
            std::ofstream f(path(e.id, ".aux"), std::ios::binary | std::ios::trunc);
            put<uint32_t>(f, AUX_MAGIC);
            put<uint32_t>(f, (uint32_t) prompt.checkpoints.size());
            for (const auto & c : prompt.checkpoints) {
                put<int64_t>(f, c.n_tokens);
                put<int32_t>(f, c.pos_min);
                put<int32_t>(f, c.pos_max);
                put<uint32_t>(f, c.flags_tgt);
                put<uint32_t>(f, c.flags_dft);
                put_vec(f, c.data_tgt);
                put_vec(f, c.data_dft);
                put_vec(f, c.data_spec);
            }
            std::vector<uint8_t> spec_state;
            if (spec) {
                common_speculative_get_state(spec, id_slot, spec_state);
            }
            put_vec(f, spec_state);
            ok = ok && (bool) f;
        }
    } catch (const std::exception & err) {
        DSK_WRN("cannot serialize the prompt: %s\n", err.what());
        ok = false;
    }

    // the KV / recurrent state streams to the file without a host-sized buffer
    ok = ok && llama_state_seq_save_file(ctx_tgt, path(e.id, ".tgt").c_str(), id_slot, nullptr, 0) > 0;
    if (ok && ctx_dft) {
        ok = llama_state_seq_save_file(ctx_dft, path(e.id, ".dft").c_str(), id_slot, nullptr, 0) > 0;
    }

    if (ok) {
        e.tokens  = prompt.tokens.clone();
        e.t_saved = e.t_used = now_s();
        for (const char * ext : { ".tok", ".tgt", ".dft", ".aux" }) {
            e.bytes += file_size_or_0(path(e.id, ext));
        }
        ok = write_meta(e);
    }

    if (!ok) {
        DSK_WRN("failed to save %zu tokens, removing the partial entry\n", n);
        remove_files(e.id);
        return false;
    }

    // older saves of the same conversation that this one extends are obsolete
    for (size_t i = 0; i < entries.size();) {
        if (entries[i].tokens.size() < n && prompt.tokens.get_common_prefix(entries[i].tokens) == entries[i].tokens.size()) {
            erase(i);
        } else {
            i++;
        }
    }

    const double ms = (ggml_time_us() - t_start) / 1000.0;
    DSK_INF("saved %s: %zu tokens, %zu checkpoints, %.1f MiB in %.0f ms (%.2f GB/s)\n", e.id.c_str(), n,
            prompt.checkpoints.size(), e.bytes / (1024.0 * 1024.0), ms, e.bytes / std::max(1e-3, ms) / 1e6);

    const std::string id = e.id;
    entries.push_back(std::move(e));
    n_saves++;

    enforce(id);

    return true;
}

int server_disk_cache::find(const server_tokens & tokens, size_t lcp_min, size_t n_ctx_slot, size_t * lcp_out) const {
    int    best     = -1;
    size_t lcp_best = lcp_min;
    for (size_t i = 0; i < entries.size(); i++) {
        if (entries[i].tokens.size() > n_ctx_slot) {
            continue;
        }
        const size_t lcp = entries[i].tokens.get_common_prefix(tokens);
        if (lcp > lcp_best) {
            lcp_best = lcp;
            best     = (int) i;
        }
    }
    if (lcp_out) {
        *lcp_out = best >= 0 ? lcp_best : 0;
    }
    return best;
}

bool server_disk_cache::load(int idx, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec,
                             int32_t id_slot, server_prompt & prompt) {
    auto & e = entries[idx];

    const int64_t t_start = ggml_time_us();

    std::list<common_prompt_checkpoint> checkpoints;
    std::vector<uint8_t> spec_state;
    try {
        std::ifstream f(path(e.id, ".aux"), std::ios::binary);
        uint32_t magic = 0, n_ckpt = 0;
        bool ok = get(f, magic) && magic == AUX_MAGIC && get(f, n_ckpt);
        for (uint32_t i = 0; ok && i < n_ckpt; i++) {
            common_prompt_checkpoint c;
            int64_t  n_tokens = 0;
            int32_t  pos_min = 0, pos_max = 0;
            uint32_t ft = 0, fd = 0;
            ok = get(f, n_tokens) && get(f, pos_min) && get(f, pos_max) && get(f, ft) && get(f, fd) &&
                 get_vec(f, c.data_tgt) && get_vec(f, c.data_dft) && get_vec(f, c.data_spec);
            c.n_tokens  = n_tokens;
            c.pos_min   = pos_min;
            c.pos_max   = pos_max;
            c.flags_tgt = ft;
            c.flags_dft = fd;
            checkpoints.push_back(std::move(c));
        }
        ok = ok && get_vec(f, spec_state);
        if (!ok) {
            throw std::runtime_error("truncated or corrupt");
        }
    } catch (const std::exception & err) {
        DSK_WRN("entry %s: unreadable .aux (%s), dropping it\n", e.id.c_str(), err.what());
        erase(idx);
        return false;
    }

    // a non-null token buffer: with tokens_out == nullptr llama_state_seq_load_file only probes the token count
    llama_token no_tokens = 0;
    size_t      n_tok     = 0;
    llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
    bool ok = llama_state_seq_load_file(ctx_tgt, path(e.id, ".tgt").c_str(), id_slot, &no_tokens, 0, &n_tok) > 0;
    if (ok && ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        ok = e.has_dft && llama_state_seq_load_file(ctx_dft, path(e.id, ".dft").c_str(), id_slot, &no_tokens, 0, &n_tok) > 0;
    }
    if (!ok) {
        DSK_WRN("entry %s: state load failed, dropping it\n", e.id.c_str());
        erase(idx);
        return false;
    }

    if (spec && !spec_state.empty()) {
        common_speculative_set_state(spec, id_slot, spec_state);
    }

    prompt.tokens      = e.tokens.clone();
    prompt.checkpoints = std::move(checkpoints);

    e.t_used = now_s();
    e.hits++;
    write_meta(e);
    n_loads++;

    const double ms = (ggml_time_us() - t_start) / 1000.0;
    DSK_INF("loaded %s: %zu tokens, %zu checkpoints, %.1f MiB in %.0f ms (%.2f GB/s, hit %d)\n", e.id.c_str(),
            prompt.tokens.size(), prompt.checkpoints.size(), e.bytes / (1024.0 * 1024.0), ms,
            e.bytes / std::max(1e-3, ms) / 1e6, e.hits);

    return true;
}
