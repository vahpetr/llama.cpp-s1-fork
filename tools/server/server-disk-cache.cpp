#include "server-disk-cache.h"

#include "log.h"

#include "ggml-backend.h"
#include "../../src/llama-ext.h" // llama_state_seq_write_cb / read_cb [TAG_DISK_DELTA]

#include <algorithm>
#include <chrono>
#include <cinttypes>
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

static constexpr uint32_t DISK_CACHE_VERSION = 4; // 2: deep checkpoint refs in .aux, 3: block-deduplicated state streams, 4: .aux checkpoints by key
static constexpr uint32_t AUX_MAGIC          = 0x58554144; // "DAUX"
static constexpr uint32_t CKPT_MAGIC         = 0x324b4344; // "DCK2"
static constexpr uint32_t MAP_MAGIC          = 0x3150414d; // "MAP1"
static constexpr size_t   BLOB_BYTES         = 4ull << 20; // block size of the state streams (aligned within each tensor)

static int64_t now_s() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// 2 x 64-bit hash, 4 lanes for throughput (memory-bound on s1); content keys of the state blocks [TAG_DISK_DELTA]
static std::string hash_key(const uint8_t * p, size_t n) {
    uint64_t a = 0x9E3779B97F4A7C15ull, b = 0xC2B2AE3D27D4EB4Full, c = 0x165667B19E3779F9ull, d = 0x27D4EB2F165667C5ull;
    const auto mix = [](uint64_t h, uint64_t w, uint64_t k) {
        h = (h ^ w) * k;
        return h ^ (h >> 31);
    };
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        uint64_t w[4];
        memcpy(w, p + i, 32);
        a = mix(a, w[0], 0x9FB21C651E98DF25ull);
        b = mix(b, w[1], 0xD6E8FEB86659FD93ull);
        c = mix(c, w[2], 0xFF51AFD7ED558CCDull);
        d = mix(d, w[3], 0xC4CEB9FE1A85EC53ull);
    }
    for (; i < n; i++) {
        a = mix(a, p[i], 0x9FB21C651E98DF25ull);
    }
    uint64_t h1 = mix(mix(a, c, 0xFF51AFD7ED558CCDull), n, 0x9FB21C651E98DF25ull);
    uint64_t h2 = mix(mix(b, d, 0xC4CEB9FE1A85EC53ull), a ^ n, 0xD6E8FEB86659FD93ull);
    return string_format("%016llx%016llx-%zu", (unsigned long long) h1, (unsigned long long) h2, n);
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
static bool get_vec(std::ifstream & f, std::vector<uint8_t> & v);
static bool skip_vec(std::ifstream & f) {
    uint64_t n = 0;
    if (!get(f, n) || n > (1ull << 33)) {
        return false;
    }
    f.seekg((std::streamoff) n, std::ios::cur);
    return (bool) f;
}
static void put_str(std::ofstream & f, const std::string & v) {
    put<uint32_t>(f, (uint32_t) v.size());
    f.write(v.data(), v.size());
}
static bool get_str(std::ifstream & f, std::string & v) {
    uint32_t n = 0;
    if (!get(f, n) || n > 4096) {
        return false;
    }
    v.resize(n);
    return n == 0 || (bool) f.read(v.data(), n);
}

// .aux layout: magic, n_ckpt, n_ckpt x {n_tokens, pos_min, pos_max, flags_tgt, flags_dft, tgt, dft, spec},
// spec_state, n_deep, n_deep x {n_tokens, kind, key}. with_data = false seeks over the state blobs.
static bool read_aux(const std::string & path, bool with_data, std::list<common_prompt_checkpoint> * checkpoints,
                     std::vector<uint8_t> * spec_state, std::vector<server_deep_ref> & deep,
                     std::vector<std::string> & ckpt_keys) {
    std::ifstream f(path, std::ios::binary);
    uint32_t magic = 0, n_ckpt = 0;
    bool ok = get(f, magic) && magic == AUX_MAGIC && get(f, n_ckpt) && n_ckpt < 4096;
    for (uint32_t i = 0; ok && i < n_ckpt; i++) {
        common_prompt_checkpoint c;
        int64_t  n_tokens = 0;
        int32_t  pos_min = 0, pos_max = 0;
        uint32_t ft = 0, fd = 0;
        std::string key;
        ok = get(f, n_tokens) && get(f, pos_min) && get(f, pos_max) && get(f, ft) && get(f, fd) && get_str(f, key);
        if (ok) {
            ckpt_keys.push_back(key);
        }
        c.n_tokens  = n_tokens;
        c.pos_min   = pos_min;
        c.pos_max   = pos_max;
        c.flags_tgt = ft;
        c.flags_dft = fd;
        if (ok && checkpoints) {
            checkpoints->push_back(std::move(c));
        }
    }
    if (with_data && spec_state) {
        ok = ok && get_vec(f, *spec_state);
    } else {
        ok = ok && skip_vec(f);
    }
    uint32_t n_deep = 0;
    ok = ok && get(f, n_deep) && n_deep < 65536;
    for (uint32_t i = 0; ok && i < n_deep; i++) {
        server_deep_ref r;
        ok = get(f, r.n_tokens) && get(f, r.kind) && get_str(f, r.key);
        if (ok) {
            deep.push_back(std::move(r));
        }
    }
    return ok;
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
                                     std::string fingerprint, bool has_mtmd, float ckpt_ratio, int32_t ckpt_min_step,
                                     int32_t ckpt_max)
    : dir(std::move(dir)), quota(quota_bytes), n_min_tokens(min_tokens), ttl_days(ttl_days),
      fingerprint(std::move(fingerprint)), has_mtmd(has_mtmd), ckpt_ratio(ckpt_ratio),
      ckpt_min_step(std::max(1, ckpt_min_step)), ckpt_max(std::max(2, ckpt_max)) {}

std::string server_disk_cache::ckpt_path(const std::string & key) const {
    return (fs::path(dir) / "ckpt" / (key + ".ck")).string();
}

std::string server_disk_cache::path(const std::string & id, const char * ext) const {
    return (fs::path(dir) / (id + ext)).string();
}

void server_disk_cache::remove_files(const std::string & id) const {
    std::error_code ec;
    for (const char * ext : { ".meta", ".tok", ".tgt", ".dft", ".aux", ".meta.tmp", ".tok.tmp", ".aux.tmp", ".tgt.tmp", ".dft.tmp" }) {
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
    uint64_t res = blob_bytes;
    for (const auto & e : entries) {
        res += e.own_bytes; // .tok, .aux and the maps (which hold the small fields inline)
    }
    for (const auto & it : ckpts) {
        res += it.second.bytes;
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
                             (!e.has_dft || file_size_or_0(path(id, ".dft")) > 0) &&
                             read_aux(path(id, ".aux"), false, nullptr, nullptr, e.deep, e.ckpt_keys) &&
                             read_map_keys(path(id, ".tgt"), e.blobs) &&
                             (!e.has_dft || read_map_keys(path(id, ".dft"), e.blobs));
                        for (const char * ext : { ".tok", ".aux", ".tgt", ".dft" }) {
                            e.own_bytes += file_size_or_0(path(id, ext));
                        }
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

    // block store: every complete block file (a crash leaves only .tmp files); unreferenced ones go in enforce()
    fs::create_directories(fs::path(dir) / "blk", ec);
    for (const auto & de : fs::directory_iterator(fs::path(dir) / "blk", ec)) {
        const std::string name = de.path().filename().string();
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".blk") == 0) {
            const uint64_t n = file_size_or_0(de.path().string());
            blobs[name.substr(0, name.size() - 4)] = n;
            blob_bytes += n;
        } else {
            fs::remove(de.path(), ec);
        }
    }
    {
        // a block whose size does not match its key is a torn write
        for (auto it = blobs.begin(); it != blobs.end();) {
            const auto dash = it->first.rfind('-');
            if (dash == std::string::npos || std::to_string(it->second) != it->first.substr(dash + 1)) {
                fs::remove(blob_path(it->first), ec);
                blob_bytes -= it->second;
                it = blobs.erase(it);
            } else {
                ++it;
            }
        }
    }

    // entries whose blocks are gone (a crash between writes, a manual cleanup) cannot be loaded
    for (size_t i = 0; i < entries.size();) {
        const bool complete = std::all_of(entries[i].blobs.begin(), entries[i].blobs.end(),
                [&](const std::string & k) { return blobs.count(k) > 0; });
        if (complete) {
            i++;
        } else {
            erase(i);
            n_dropped++;
        }
    }

    // deep checkpoint store: keep the files with this fingerprint
    fs::create_directories(fs::path(dir) / "ckpt", ec);
    size_t n_ck_dropped = 0;
    const std::string fp_short = fingerprint.substr(0, 16);
    for (const auto & de : fs::directory_iterator(fs::path(dir) / "ckpt", ec)) {
        const std::string name = de.path().filename().string();
        bool ok = false;
        server_deep_ckpt_meta m;
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".ck") == 0) {
            std::ifstream f(de.path(), std::ios::binary);
            uint32_t magic = 0;
            std::string fp;
            ok = get(f, magic) && magic == CKPT_MAGIC && get_str(f, fp) && fp == fp_short && get_str(f, m.key) &&
                 get(f, m.n_tokens) && get(f, m.pos_min) && get(f, m.pos_max) && m.key + ".ck" == name;
        }
        if (ok) {
            m.bytes  = file_size_or_0(de.path().string());
            std::error_code ec_t;
            const auto mt = fs::last_write_time(de.path(), ec_t);
            m.t_used = ec_t ? now_s() : std::chrono::duration_cast<std::chrono::seconds>(
                           (mt - fs::file_time_type::clock::now()) + std::chrono::system_clock::now().time_since_epoch()).count();
            ckpts[m.key] = m;
        } else {
            fs::remove(de.path(), ec);
            n_ck_dropped++;
        }
    }

    enforce("");

    DSK_INF("dir %s: %zu entries + %zu deep checkpoints, %.2f GiB (quota %.0f GiB, min %d tokens, ttl %d days, "
            "deep ratio %.2f min step %d max %d), %zu + %zu stale/foreign dropped\n",
            dir.c_str(), entries.size(), ckpts.size(), total_bytes() / (1024.0 * 1024.0 * 1024.0),
            quota / (1024.0 * 1024.0 * 1024.0), n_min_tokens, ttl_days, ckpt_ratio, ckpt_min_step, ckpt_max,
            n_dropped, n_ck_dropped);
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

    deep_gc();
    blob_gc();

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
        deep_gc(); // the evicted entry's deep checkpoints and blocks may now be unreferenced
        blob_gc();
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

    uint64_t n_ckpt_written = 0;

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
            // the in-RAM checkpoints go to the content-addressed store (a later save of the same conversation, a
            // branch or a copy writes them once), the .aux keeps their keys
            std::vector<size_t> ns;
            for (const auto & c : prompt.checkpoints) {
                ns.push_back((size_t) c.n_tokens);
            }
            std::vector<size_t> ns_sorted = ns;
            std::sort(ns_sorted.begin(), ns_sorted.end());
            ns_sorted.erase(std::unique(ns_sorted.begin(), ns_sorted.end()), ns_sorted.end());
            const std::vector<std::string> keys_sorted = prompt.tokens.prefix_keys(ns_sorted);
            put<uint32_t>(f, AUX_MAGIC);
            put<uint32_t>(f, (uint32_t) prompt.checkpoints.size());
            for (const auto & c : prompt.checkpoints) {
                const auto at = std::lower_bound(ns_sorted.begin(), ns_sorted.end(), (size_t) c.n_tokens) - ns_sorted.begin();
                const std::string & key = keys_sorted[at];
                bool written = false;
                if (!ckpt_store(key, c, &written)) {
                    throw std::runtime_error("cannot store a checkpoint");
                }
                n_ckpt_written += written ? file_size_or_0(ckpt_path(key)) : 0;
                e.ckpt_keys.push_back(key);
                put<int64_t>(f, c.n_tokens);
                put<int32_t>(f, c.pos_min);
                put<int32_t>(f, c.pos_max);
                put<uint32_t>(f, c.flags_tgt);
                put<uint32_t>(f, c.flags_dft);
                put_str(f, key);
            }
            std::vector<uint8_t> spec_state;
            if (spec) {
                common_speculative_get_state(spec, id_slot, spec_state);
            }
            put_vec(f, spec_state);
            put<uint32_t>(f, (uint32_t) prompt.deep.size());
            for (const auto & r : prompt.deep) {
                put<int64_t>(f, r.n_tokens);
                put<uint8_t>(f, r.kind);
                put_str(f, r.key);
            }
            ok = ok && (bool) f;
        }
    } catch (const std::exception & err) {
        DSK_WRN("cannot serialize the prompt: %s\n", err.what());
        ok = false;
    }

    // [TAG_DISK_DELTA] the KV / recurrent state streams through 4 MiB content blocks: a block already on disk (the
    //   unchanged part of a conversation saved before, a branch, a copy) is referenced, only new ones are written
    uint64_t n_logical = 0, n_written = 0;
    ok = ok && save_stream(ctx_tgt, id_slot, path(e.id, ".tgt"), e.blobs, n_logical, n_written);
    if (ok && ctx_dft) {
        ok = save_stream(ctx_dft, id_slot, path(e.id, ".dft"), e.blobs, n_logical, n_written);
    }

    if (ok) {
        e.tokens  = prompt.tokens.clone();
        e.deep    = prompt.deep;
        e.t_saved = e.t_used = now_s();
        e.bytes   = n_logical + file_size_or_0(path(e.id, ".tok")) + file_size_or_0(path(e.id, ".aux"));
        for (const auto & c : prompt.checkpoints) {
            e.bytes += c.size();
        }
        for (const char * ext : { ".tok", ".aux", ".tgt", ".dft" }) {
            e.own_bytes += file_size_or_0(path(e.id, ext));
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
    DSK_INF("saved %s: %zu tokens, %zu checkpoints + %zu deep, %.1f MiB state, %.1f MiB written (%.1f %%) in %.0f ms\n",
            e.id.c_str(), n, prompt.checkpoints.size(), prompt.deep.size(), n_logical / (1024.0 * 1024.0),
            (n_written + n_ckpt_written + file_size_or_0(path(e.id, ".aux"))) / (1024.0 * 1024.0),
            100.0 * (n_written + n_ckpt_written + file_size_or_0(path(e.id, ".aux"))) / std::max<double>(1.0, e.bytes), ms);

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
    std::vector<server_deep_ref> deep;
    try {
        std::vector<std::string> keys;
        if (!read_aux(path(e.id, ".aux"), true, &checkpoints, &spec_state, deep, keys) || keys.size() != checkpoints.size()) {
            throw std::runtime_error("truncated or corrupt");
        }
        size_t i = 0;
        for (auto & c : checkpoints) {
            common_prompt_checkpoint full;
            if (!deep_get(keys[i++], full) || full.n_tokens != c.n_tokens) {
                throw std::runtime_error("a checkpoint is missing");
            }
            c = std::move(full);
        }
    } catch (const std::exception & err) {
        DSK_WRN("entry %s: unreadable .aux (%s), dropping it\n", e.id.c_str(), err.what());
        erase(idx);
        return false;
    }

    llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
    bool ok = load_stream(ctx_tgt, id_slot, path(e.id, ".tgt"));
    if (ok && ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        ok = e.has_dft && load_stream(ctx_dft, id_slot, path(e.id, ".dft"));
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
    prompt.deep.clear();
    for (auto & r : deep) {
        if (ckpts.count(r.key)) { // a collected checkpoint is just a gap in the schedule
            prompt.deep.push_back(std::move(r));
        }
    }

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

//
// deep checkpoints [TAG_DEEP_CKPT]
//

// depth schedule: walking up from the anchor, a checkpoint is kept when it lies at least
// max(min_step, ratio x its distance to the end) above the previous kept one; the anchor (the shallowest one) and
// branch points are always kept; above ckpt_max the oldest depth / branch ones go first. Walking up (not down from
// the end) keeps the decision for old checkpoints stable while the conversation grows: they are thinned only as
// their distance to the end grows, so the gaps stay ~ratio x distance (ratio 0.125: a branch replays <= ~0.2 x its
// distance to the end; ~20-24 checkpoints for 1M tokens).
void server_disk_cache::thin(std::vector<server_deep_ref> & refs, int64_t n_end) const {
    std::sort(refs.begin(), refs.end(), [](const auto & a, const auto & b) { return a.n_tokens < b.n_tokens; });
    // one ref per position, the most important kind wins
    for (size_t i = 1; i < refs.size();) {
        if (refs[i].n_tokens == refs[i - 1].n_tokens) {
            if (refs[i].kind > refs[i - 1].kind) {
                refs[i - 1] = refs[i];
            }
            refs.erase(refs.begin() + i);
        } else {
            i++;
        }
    }
    if (refs.empty()) {
        return;
    }

    std::vector<server_deep_ref> kept;
    for (size_t i = 0; i < refs.size(); i++) {
        const auto & r = refs[i];
        const int64_t gap_min = std::max<int64_t>(ckpt_min_step, (int64_t) (ckpt_ratio * (double) (n_end - r.n_tokens)));
        if (i == 0 || r.kind != server_deep_ref::DEEP_KIND_DEPTH || r.n_tokens - kept.back().n_tokens >= gap_min) {
            kept.push_back(r);
        }
    }

    for (uint8_t kind : { server_deep_ref::DEEP_KIND_DEPTH, server_deep_ref::DEEP_KIND_BRANCH }) {
        for (size_t i = 1; (int32_t) kept.size() > ckpt_max && i < kept.size();) {
            if (kept[i].kind == kind) {
                kept.erase(kept.begin() + i);
            } else {
                i++;
            }
        }
    }

    refs = std::move(kept);
}

// write c under key unless a file with that key exists (content-addressed: the same prefix gives the same state)
bool server_disk_cache::ckpt_store(const std::string & key, const common_prompt_checkpoint & c, bool * written) {
    if (written) {
        *written = false;
    }
    const int64_t t_now = now_s();
    auto it = ckpts.find(key);
    if (it != ckpts.end()) {
        it->second.t_used = t_now;
        return true;
    }
    const std::string fin = ckpt_path(key);
    const std::string tmp = fin + ".tmp";
    bool ok = false;
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        put<uint32_t>(f, CKPT_MAGIC);
        put_str(f, fingerprint.substr(0, 16));
        put_str(f, key);
        put<int64_t>(f, c.n_tokens);
        put<int32_t>(f, c.pos_min);
        put<int32_t>(f, c.pos_max);
        put<uint32_t>(f, c.flags_tgt);
        put<uint32_t>(f, c.flags_dft);
        put_vec(f, c.data_tgt);
        put_vec(f, c.data_dft);
        put_vec(f, c.data_spec);
        ok = (bool) f;
    }
    std::error_code ec;
    if (ok) {
        fs::rename(tmp, fin, ec);
        ok = !ec;
    }
    if (!ok) {
        fs::remove(tmp, ec);
        DSK_WRN("cannot write checkpoint at %" PRId64 "\n", c.n_tokens);
        return false;
    }
    server_deep_ckpt_meta m;
    m.key      = key;
    m.n_tokens = c.n_tokens;
    m.pos_min  = c.pos_min;
    m.pos_max  = c.pos_max;
    m.bytes    = file_size_or_0(fin);
    m.t_used   = t_now;
    ckpts.emplace(key, m);
    if (written) {
        *written = true;
    }
    return true;
}

void server_disk_cache::deep_put(server_prompt & prompt, const common_prompt_checkpoint & c, uint8_t kind) {
    // below min-step a re-process is cheaper than 113 MiB of disk (a short system prompt, a tiny chat)
    if (quota == 0 || c.data_tgt.empty() || c.n_tokens < ckpt_min_step || c.n_tokens > (int64_t) prompt.tokens.size() ||
        ((c.flags_tgt | c.flags_dft) & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        return;
    }

    // refs whose file was collected (e.g. carried by a RAM prompt-cache state) are gaps, not schedule members
    prompt.deep.erase(std::remove_if(prompt.deep.begin(), prompt.deep.end(),
                [&](const auto & r) { return ckpts.count(r.key) == 0; }), prompt.deep.end());

    std::string key;
    try {
        key = prompt.tokens.prefix_keys({ (size_t) c.n_tokens })[0];
    } catch (const std::exception & err) {
        DSK_WRN("deep checkpoint key: %s\n", err.what());
        return;
    }

    bool need_enforce = false;

    // would the schedule keep it? then it is worth the write (~113 MiB on the prod model)
    {
        std::vector<server_deep_ref> trial = prompt.deep;
        trial.push_back({ c.n_tokens, key, kind });
        thin(trial, (int64_t) prompt.tokens.size());
        const bool kept = std::any_of(trial.begin(), trial.end(), [&](const auto & r) { return r.key == key; });
        if (!kept) {
            return;
        }
    }

    bool written = false;
    if (!ckpt_store(key, c, &written)) {
        return;
    }
    need_enforce = written;
    if (written) {
        DSK_INF("deep checkpoint at %" PRId64 " (%s) stored\n", c.n_tokens,
                kind == server_deep_ref::DEEP_KIND_BRANCH ? "branch" : kind == server_deep_ref::DEEP_KIND_ANCHOR ? "anchor" : "depth");
    }

    bool found = false;
    for (auto & r : prompt.deep) {
        if (r.n_tokens == c.n_tokens) {
            r.key  = key;
            r.kind = std::max(r.kind, kind);
            found  = true;
        }
    }
    if (!found) {
        prompt.deep.push_back({ c.n_tokens, key, kind });
    }

    const size_t n_before = prompt.deep.size();
    thin(prompt.deep, (int64_t) prompt.tokens.size());
    if (prompt.deep.size() != n_before) {
        DSK_INF("deep schedule: %zu -> %zu checkpoints for %zu tokens\n", n_before, prompt.deep.size(), prompt.tokens.size());
    }

    if (need_enforce && total_bytes() > quota) {
        enforce("");
    }
}

void server_disk_cache::deep_adopt(server_prompt & prompt, const server_deep_ckpt_meta & m) {
    for (const auto & r : prompt.deep) {
        if (r.key == m.key) {
            return;
        }
    }
    prompt.deep.push_back({ m.n_tokens, m.key, server_deep_ref::DEEP_KIND_DEPTH });
    thin(prompt.deep, (int64_t) prompt.tokens.size());
}

std::vector<server_deep_ckpt_meta> server_disk_cache::deep_candidates(const server_tokens & tokens, int64_t n_max) const {
    std::vector<server_deep_ckpt_meta> cand;
    for (const auto & it : ckpts) {
        if (it.second.n_tokens <= n_max && it.second.n_tokens <= (int64_t) tokens.size()) {
            cand.push_back(it.second);
        }
    }
    if (cand.empty()) {
        return cand;
    }
    std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.n_tokens < b.n_tokens; });
    std::vector<size_t> ns;
    for (const auto & c : cand) {
        if (ns.empty() || ns.back() != (size_t) c.n_tokens) {
            ns.push_back((size_t) c.n_tokens);
        }
    }
    std::vector<std::string> keys;
    try {
        keys = tokens.prefix_keys(ns);
    } catch (const std::exception &) {
        return {};
    }
    const std::set<std::string> want(keys.begin(), keys.end());
    std::vector<server_deep_ckpt_meta> res;
    for (auto it = cand.rbegin(); it != cand.rend(); ++it) {
        if (want.count(it->key)) {
            res.push_back(*it);
        }
    }
    return res;
}

bool server_disk_cache::deep_get(const std::string & key, common_prompt_checkpoint & out) {
    auto it = ckpts.find(key);
    if (it == ckpts.end()) {
        return false;
    }
    try {
        std::ifstream f(ckpt_path(key), std::ios::binary);
        uint32_t magic = 0;
        std::string fp, k;
        int64_t  n_tokens = 0;
        int32_t  pos_min = 0, pos_max = 0;
        uint32_t ft = 0, fd = 0;
        const bool ok = get(f, magic) && magic == CKPT_MAGIC && get_str(f, fp) && get_str(f, k) && k == key &&
                        get(f, n_tokens) && get(f, pos_min) && get(f, pos_max) && get(f, ft) && get(f, fd) &&
                        get_vec(f, out.data_tgt) && get_vec(f, out.data_dft) && get_vec(f, out.data_spec);
        if (!ok) {
            throw std::runtime_error("truncated or corrupt");
        }
        out.n_tokens  = n_tokens;
        out.pos_min   = pos_min;
        out.pos_max   = pos_max;
        out.flags_tgt = ft;
        out.flags_dft = fd;
        out.id_task   = -1;
    } catch (const std::exception & err) {
        DSK_WRN("deep checkpoint %s unreadable (%s), dropping it\n", key.c_str(), err.what());
        std::error_code ec;
        fs::remove(ckpt_path(key), ec);
        ckpts.erase(it);
        return false;
    }
    it->second.t_used = now_s();
    it->second.hits++;
    std::error_code ec;
    fs::last_write_time(ckpt_path(key), fs::file_time_type::clock::now(), ec); // t_used survives a restart
    return true;
}

void server_disk_cache::deep_gc() {
    std::set<std::string> referenced;
    if (live_refs) {
        referenced = live_refs();
    }
    for (const auto & e : entries) {
        for (const auto & r : e.deep) {
            referenced.insert(r.key);
        }
        referenced.insert(e.ckpt_keys.begin(), e.ckpt_keys.end());
    }
    for (auto it = ckpts.begin(); it != ckpts.end();) {
        if (!referenced.count(it->first)) {
            std::error_code ec;
            fs::remove(ckpt_path(it->first), ec);
            it = ckpts.erase(it);
        } else {
            ++it;
        }
    }
}

//
// block-deduplicated state streams [TAG_DISK_DELTA]
//
// <id>.tgt / <id>.dft = MAP1 magic, then segments: u8 kind (0 = inline bytes: u64 n + bytes; 1 = block: key) until
// kind 255. The concatenation is exactly the llama_state_seq_write stream, so loading is format-agnostic; only the
// cutting of tensor ranges into blocks aligned to BLOB_BYTES within each tensor makes the unchanged part of a growing
// conversation produce the same blocks again.

std::string server_disk_cache::blob_path(const std::string & key) const {
    return (fs::path(dir) / "blk" / (key + ".blk")).string();
}

bool server_disk_cache::save_stream(llama_context * ctx, int32_t id_slot, const std::string & map_path,
                                    std::vector<std::string> & keys, uint64_t & n_logical, uint64_t & n_written) {
    const std::string tmp = map_path + ".tmp";
    std::ofstream fm(tmp, std::ios::binary | std::ios::trunc);
    put<uint32_t>(fm, MAP_MAGIC);

    std::vector<uint8_t> inl;
    std::vector<uint8_t> buf;
    bool ok = true;

    const auto flush_inline = [&]() {
        if (inl.empty()) {
            return;
        }
        put<uint8_t>(fm, 0);
        put<uint64_t>(fm, inl.size());
        fm.write(reinterpret_cast<const char *>(inl.data()), inl.size());
        n_written += inl.size();
        inl.clear();
    };

    llama_state_seq_writer w;
    w.write = [&](const void * src, size_t size) {
        const auto * p = static_cast<const uint8_t *>(src);
        inl.insert(inl.end(), p, p + size);
        n_logical += size;
    };
    w.write_tensor = [&](ggml_tensor * tensor, size_t offset, size_t size) {
        if (size < (64u << 10)) {
            // small ranges (e.g. the per-row ranges of a transposed V cache) stay inline: no file per few KiB
            const size_t at = inl.size();
            inl.resize(at + size);
            ggml_backend_tensor_get(tensor, inl.data() + at, offset, size);
            n_logical += size;
            return;
        }
        flush_inline();
        n_logical += size;
        size_t pos = offset;
        const size_t end = offset + size;
        while (ok && pos < end) {
            // block boundaries are fixed offsets of the tensor, so the same rows give the same blocks every save
            const size_t blk_end = std::min(end, (pos / BLOB_BYTES + 1) * BLOB_BYTES);
            const size_t n = blk_end - pos;
            buf.resize(n);
            ggml_backend_tensor_get(tensor, buf.data(), pos, n);
            const std::string key = hash_key(buf.data(), n);
            if (blobs.count(key) == 0) {
                const std::string fin = blob_path(key);
                const std::string ftmp = fin + ".tmp";
                {
                    std::ofstream fb(ftmp, std::ios::binary | std::ios::trunc);
                    fb.write(reinterpret_cast<const char *>(buf.data()), n);
                    ok = (bool) fb;
                }
                std::error_code ec;
                if (ok) {
                    fs::rename(ftmp, fin, ec);
                    ok = !ec;
                }
                if (!ok) {
                    fs::remove(ftmp, ec);
                    break;
                }
                blobs[key] = n;
                blob_bytes += n;
                n_written += n;
            }
            put<uint8_t>(fm, 1);
            put_str(fm, key);
            keys.push_back(key);
            pos = blk_end;
        }
    };

    ok = llama_state_seq_write_cb(ctx, id_slot, 0, w) > 0 && ok;
    flush_inline();
    put<uint8_t>(fm, 255);
    fm.close();
    ok = ok && (bool) fm;

    std::error_code ec;
    if (ok) {
        fs::rename(tmp, map_path, ec);
        ok = !ec;
    }
    if (!ok) {
        fs::remove(tmp, ec);
    }
    return ok;
}

bool server_disk_cache::read_map_keys(const std::string & map_path, std::vector<std::string> & keys) {
    std::ifstream fm(map_path, std::ios::binary);
    uint32_t magic = 0;
    if (!get(fm, magic) || magic != MAP_MAGIC) {
        return false;
    }
    for (;;) {
        uint8_t kind = 0;
        if (!get(fm, kind)) {
            return false;
        }
        if (kind == 255) {
            return true;
        }
        if (kind == 0) {
            uint64_t n = 0;
            if (!get(fm, n)) {
                return false;
            }
            fm.seekg((std::streamoff) n, std::ios::cur);
        } else if (kind == 1) {
            std::string key;
            if (!get_str(fm, key)) {
                return false;
            }
            keys.push_back(std::move(key));
        } else {
            return false;
        }
    }
}

bool server_disk_cache::load_stream(llama_context * ctx, int32_t id_slot, const std::string & map_path) {
    std::ifstream fm(map_path, std::ios::binary);
    uint32_t magic = 0;
    if (!get(fm, magic) || magic != MAP_MAGIC) {
        return false;
    }

    // the current segment: inline bytes come from the map itself, a block from its (verified) buffer
    uint64_t seg_left = 0;
    bool     seg_inline = false;
    std::vector<uint8_t> blk;
    size_t   blk_pos = 0;
    bool done = false;
    bool corrupt = false;

    const auto next_segment = [&]() -> bool {
        uint8_t kind = 0;
        if (!get(fm, kind)) {
            return false;
        }
        if (kind == 255) {
            done = true;
            return false;
        }
        if (kind == 0) {
            seg_inline = true;
            return (bool) get(fm, seg_left);
        }
        std::string key;
        if (kind != 1 || !get_str(fm, key)) {
            return false;
        }
        const auto it = blobs.find(key);
        if (it == blobs.end()) {
            DSK_WRN("block %s is missing\n", key.c_str());
            return false;
        }
        // a block is shared by every entry with this prefix: check it against its key, a torn write after a host
        // hang must fail the load (and go), not restore wrong state
        blk.resize(it->second);
        {
            std::ifstream fb(blob_path(key), std::ios::binary);
            if (!fb.read(reinterpret_cast<char *>(blk.data()), blk.size()) || hash_key(blk.data(), blk.size()) != key) {
                DSK_WRN("block %s is corrupt, dropping it\n", key.c_str());
                std::error_code ec;
                fs::remove(blob_path(key), ec);
                blob_bytes -= it->second;
                blobs.erase(it);
                corrupt = true;
                return false;
            }
        }
        seg_inline = false;
        seg_left   = blk.size();
        blk_pos    = 0;
        return true;
    };

    llama_state_seq_reader r;
    r.read = [&](void * dst, size_t size) -> bool {
        auto * out = static_cast<char *>(dst);
        while (size > 0) {
            if (seg_left == 0 && !next_segment()) {
                return false;
            }
            const size_t n = (size_t) std::min<uint64_t>(seg_left, size);
            if (seg_inline) {
                if (!fm.read(out, n)) {
                    return false;
                }
            } else {
                memcpy(out, blk.data() + blk_pos, n);
                blk_pos += n;
            }
            out      += n;
            size     -= n;
            seg_left -= n;
        }
        return true;
    };

    const bool ok = llama_state_seq_read_cb(ctx, id_slot, 0, r) > 0 && !corrupt;
    GGML_UNUSED(done);
    return ok;
}

void server_disk_cache::blob_gc() {
    std::set<std::string> referenced;
    for (const auto & e : entries) {
        referenced.insert(e.blobs.begin(), e.blobs.end());
    }
    for (auto it = blobs.begin(); it != blobs.end();) {
        if (!referenced.count(it->first)) {
            std::error_code ec;
            fs::remove(blob_path(it->first), ec);
            blob_bytes -= it->second;
            it = blobs.erase(it);
        } else {
            ++it;
        }
    }
}
