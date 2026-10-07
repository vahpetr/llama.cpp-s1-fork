#include "server-disk-cache.h"

#include "log.h"

#include "ggml-backend.h"
#include "../../src/llama-ext.h" // llama_state_seq_write_cb / read_cb [TAG_DISK_DELTA]

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <unistd.h> // environ
#include <sys/stat.h>

#define DSK_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, "disk_cache", __VA_ARGS__)
#define DSK_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, "disk_cache", __VA_ARGS__)
#define DSK_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, "disk_cache", __VA_ARGS__)

namespace fs = std::filesystem;

static constexpr uint32_t DISK_CACHE_VERSION = 7; // 7: envelope (magic, version, descriptor hash, length) at the head of .tgt / .dft maps and .ck // 6: user-message starts at the end of .aux (optional: a 5 entry loads with them unknown) // 5: sealed (checksummed) .tok / .aux / .ck and map segments // 2: deep checkpoint refs in .aux, 3: block-deduplicated state streams, 4: .aux checkpoints by key
static constexpr uint32_t AUX_MAGIC          = 0x58554144; // "DAUX"
static constexpr uint32_t CKPT_MAGIC         = 0x334b4344; // "DCK3" (format 5 / 6: no envelope)
static constexpr uint32_t CKPT_ENV_MAGIC     = 0x344b4344; // "DCK4": fingerprint, envelope version, descriptor hash
static constexpr uint32_t MAP_MAGIC          = 0x3250414d; // "MAP2" (format 5 / 6: no envelope)
static constexpr uint32_t MAP_ENV_MAGIC      = 0x3350414d; // "MAP3": envelope version, descriptor hash, stream length
static constexpr uint32_t ENVELOPE_VERSION   = 1;
// [TAG_STATE_DESCRIPTOR] bump on any change of what a restored checkpoint means (its flags, the draft-prefix retention,
//   the replay rules after a deep rewind) that keeps the bytes the same
static constexpr uint32_t SERVER_CKPT_SEMANTICS_REV = 1;
static constexpr uint32_t SEAL_MAGIC         = 0x4c414553; // "SEAL"
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
template <typename T> static void put(std::ostream & f, const T & v) {
    f.write(reinterpret_cast<const char *>(&v), sizeof(T));
}
static void put_vec(std::ostream & f, const std::vector<uint8_t> & v) {
    put<uint64_t>(f, v.size());
    if (!v.empty()) {
        f.write(reinterpret_cast<const char *>(v.data()), v.size());
    }
}
template <typename T> static bool get(std::istream & f, T & v) {
    return (bool) f.read(reinterpret_cast<char *>(&v), sizeof(T));
}
static bool get_vec(std::istream & f, std::vector<uint8_t> & v);
static bool skip_vec(std::istream & f) {
    uint64_t n = 0;
    if (!get(f, n) || n > (1ull << 33)) {
        return false;
    }
    f.seekg((std::streamoff) n, std::ios::cur);
    return (bool) f;
}
static void put_str(std::ostream & f, const std::string & v) {
    put<uint32_t>(f, (uint32_t) v.size());
    f.write(v.data(), v.size());
}
static bool get_str(std::istream & f, std::string & v) {
    uint32_t n = 0;
    if (!get(f, n) || n > 4096) {
        return false;
    }
    v.resize(n);
    return n == 0 || (bool) f.read(v.data(), n);
}

// sealed file = body | hash_key(body) | u32 key length | SEAL: a torn or corrupted file never parses as valid
// (the content keys of the blocks only name their bytes; these files carry their own check)
static bool write_sealed(const std::string & path, const std::string & body) {
    const std::string key = hash_key(reinterpret_cast<const uint8_t *>(body.data()), body.size());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(body.data(), body.size());
    f.write(key.data(), key.size());
    put<uint32_t>(f, (uint32_t) key.size());
    put<uint32_t>(f, SEAL_MAGIC);
    return (bool) f;
}

// opens path for reading: SEALED_BAD if it does not exist, SEALED_IO if it exists but cannot be opened
static sealed_status open_for_read(const std::string & path, std::ifstream & f) {
    f.open(path, std::ios::binary);
    if (f.is_open()) {
        return SEALED_OK;
    }
    std::error_code ec;
    return !fs::exists(path, ec) && !ec ? SEALED_BAD : SEALED_IO;
}

// a file that must be non-empty: SEALED_BAD if it is missing or empty, SEALED_IO if it cannot be inspected now
static sealed_status nonempty_file(const std::string & path) {
    std::error_code ec;
    const auto n = fs::file_size(path, ec);
    if (!ec) {
        return n > 0 ? SEALED_OK : SEALED_BAD;
    }
    std::error_code ec_e;
    return !fs::exists(path, ec_e) && !ec_e ? SEALED_BAD : SEALED_IO;
}

// [TAG_DISK_READ_STATUS] (Hermes gate Major 1) a directory listing that failed (opening it or mid-way) is not an
//   empty directory: false, and the caller must not act on the partial view. missing_ok: a missing directory is empty
static bool list_dir(const fs::path & p, std::vector<fs::directory_entry> & out, bool missing_ok, std::string * err = nullptr) {
    out.clear();
    std::error_code ec;
    fs::directory_iterator it(p, ec);
    for (const fs::directory_iterator end; !ec && it != end; it.increment(ec)) {
        out.push_back(*it);
    }
    if (ec && missing_ok && ec == std::errc::no_such_file_or_directory && out.empty()) {
        return true;
    }
    if (ec && err) {
        *err = p.string() + ": " + ec.message();
    }
    return !ec;
}

static sealed_status read_sealed(const std::string & path, std::string & body) {
    std::ifstream f;
    if (const auto st = open_for_read(path, f); st != SEALED_OK) {
        return st;
    }
    std::string all;
    try {
        f.seekg(0, std::ios::end);
        const std::streamoff n = f.tellg();
        f.seekg(0, std::ios::beg);
        if (n < 0 || !f) {
            return SEALED_IO;
        }
        all.resize((size_t) n);
        if (n > 0 && !f.read(all.data(), n)) {
            return SEALED_IO;
        }
    } catch (const std::exception &) { // allocation
        return SEALED_IO;
    }
    if (all.size() < 8) {
        return SEALED_BAD;
    }
    uint32_t klen = 0, magic = 0;
    memcpy(&klen,  all.data() + all.size() - 8, 4);
    memcpy(&magic, all.data() + all.size() - 4, 4);
    if (magic != SEAL_MAGIC || klen == 0 || klen > 256 || all.size() < 8 + (size_t) klen) {
        return SEALED_BAD;
    }
    const size_t n_body = all.size() - 8 - klen;
    const std::string key = all.substr(n_body, klen);
    if (hash_key(reinterpret_cast<const uint8_t *>(all.data()), n_body) != key) {
        return SEALED_BAD;
    }
    all.resize(n_body);
    body = std::move(all);
    return SEALED_OK;
}

// .aux layout: magic, n_ckpt, n_ckpt x {n_tokens, pos_min, pos_max, flags_tgt, flags_dft, tgt, dft, spec},
// spec_state, n_deep, n_deep x {n_tokens, kind, key}, [v6: n_user, n_user x pos]. with_data = false seeks over the
// state blobs. [TAG_TURN_BOUNDARY] the user-message starts are optional: a v5 body ends before them (unknown = empty)
static sealed_status read_aux(const std::string & path, bool with_data, std::list<common_prompt_checkpoint> * checkpoints,
                     std::vector<uint8_t> * spec_state, std::vector<server_deep_ref> & deep,
                     std::vector<std::string> & ckpt_keys, std::vector<int32_t> * user_starts = nullptr) {
    std::string body;
    if (const auto st = read_sealed(path, body); st != SEALED_OK) {
        return st;
    }
    std::istringstream f(body);
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
    if (ok && f.peek() != std::char_traits<char>::eof()) {
        uint32_t n_user = 0;
        ok = get(f, n_user) && n_user < (1u << 20);
        for (uint32_t i = 0; ok && i < n_user; i++) {
            int32_t p = 0;
            ok = get(f, p);
            if (ok && user_starts) {
                user_starts->push_back(p);
            }
        }
    }
    return ok ? SEALED_OK : SEALED_BAD;
}

static bool get_vec(std::istream & f, std::vector<uint8_t> & v) {
    uint64_t n = 0;
    if (!get(f, n) || n > (1ull << 33)) { // 8 GiB: far above one recurrent state, below a corrupt length
        return false;
    }
    v.resize(n);
    return n == 0 || (bool) f.read(reinterpret_cast<char *>(v.data()), n);
}

std::string server_disk_cache_fingerprint(const common_params & params, int32_t n_ctx) {
    std::error_code ec;
    // the code that computes and lays out the state: the content of the loaded llama / ggml / mtmd
    // libraries (builds are deterministic, so a rebuild with unchanged sources keeps the cache). The
    // server library (HTTP, webui) is left out - a webui or server-side change does not change the
    // state; the disk cache's own format is DISK_CACHE_VERSION. Static builds fall back to the binary.
    std::string exe_id;
    {
        // path -> inode from /proc/self/maps ("addr perms offset dev inode path")
        std::map<std::string, unsigned long long> libs;
        bool ok = true;
        std::ifstream maps("/proc/self/maps");
        std::string line;
        while (std::getline(maps, line)) {
            const auto at = line.find('/');
            if (at == std::string::npos) {
                continue;
            }
            std::string p = line.substr(at);
            const std::string base = fs::path(p).filename().string();
            const bool ours = base.rfind("libllama", 0) == 0 || base.rfind("libggml", 0) == 0 || base.rfind("libmtmd", 0) == 0;
            if (!ours || base.rfind("libllama-server", 0) == 0) {
                continue;
            }
            if (p.size() > 10 && p.compare(p.size() - 10, 10, " (deleted)") == 0) {
                ok = false; // the loaded file was replaced on disk: its bytes cannot be hashed
                continue;
            }
            std::istringstream fields(line.substr(0, at));
            std::string addr, perms, offset, dev;
            unsigned long long inode = 0;
            fields >> addr >> perms >> offset >> dev >> inode;
            libs[p] = inode;
        }
        for (const auto & it : libs) {
            struct stat st {};
            std::ifstream f(it.first, std::ios::binary);
            const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            if (!f.good() && !f.eof()) {
                ok = false;
            }
            if (stat(it.first.c_str(), &st) != 0 || (unsigned long long) st.st_ino != it.second || bytes.empty()) {
                ok = false; // not the file that is mapped
            }
            exe_id += fs::path(it.first).filename().string() + "=" +
                      hash_key(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()) + ";";
        }
        if (libs.empty()) {
            const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
            if (!ec) {
                const auto sz = fs::file_size(exe, ec);
                const auto mt = fs::last_write_time(exe, ec).time_since_epoch().count();
                exe_id = std::to_string(sz) + ":" + std::to_string(mt);
            }
        }
        if (!ok) {
            exe_id = "unverifiable:" + random_id(); // matches nothing: the cache is not reused
        }
    }
#ifdef SERVER_STATE_SRC_HASH
    // the server's own state-handling sources (checkpoints, restore, serialization), hashed at build time
    exe_id += std::string("|server-src=") + SERVER_STATE_SRC_HASH;
#endif
    // LLAMA_* / GGML_* environment knobs (QSA, KQ_VIS, int8, ...) select graph and state variants under one binary
    std::vector<std::string> knobs;
    for (char ** ep = environ; ep && *ep; ep++) {
        const std::string kv = *ep;
        if (kv.rfind("LLAMA_", 0) == 0 || kv.rfind("GGML_", 0) == 0) {
            if (kv.rfind("LLAMA_ARG_CACHE_DISK", 0) != 0) { // only the disk cache's own options are neutral
                knobs.push_back(kv);
            }
        }
    }
    std::sort(knobs.begin(), knobs.end()); // environment order is arbitrary; the argv below keeps its order
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
    std::string env_id;
    for (const auto & kv : knobs) {
        env_id += kv + ";";
    }
    // a file's identity: size + mtime (a re-converted model at the same path and size has a new mtime)
    const auto file_id = [](const std::string & p) {
        std::error_code ec_f;
        const auto mt = fs::last_write_time(p, ec_f);
        return string_format("%llu@%lld", (unsigned long long) file_size_or_0(p),
                             ec_f ? 0ll : (long long) mt.time_since_epoch().count());
    };
    // every shard of a split GGUF (<name>-00001-of-0000N.gguf)
    const auto model_id = [&](const std::string & p) {
        std::string id = file_id(p);
        const auto at = p.rfind("-00001-of-");
        if (at != std::string::npos && p.size() >= at + 15) {
            const int n = atoi(p.substr(at + 10, 5).c_str());
            for (int i = 2; i <= n && i < 100000; i++) {
                const std::string shard = p.substr(0, at) + string_format("-%05d-of-", i) + p.substr(at + 10);
                id += "|" + file_id(shard);
            }
        }
        return id;
    };
    const std::string s = string_format("v%u|%s|%s|%s|%s|ctx%d|k%d|v%d|dk%d|dv%d|%s|%s|%s|%s",
        DISK_CACHE_VERSION,
        params.model.path.c_str(), model_id(params.model.path).c_str(),
        params.speculative.draft.mparams.path.c_str(), model_id(params.speculative.draft.mparams.path).c_str(),
        n_ctx, (int) params.cache_type_k, (int) params.cache_type_v,
        (int) params.speculative.draft.cache_type_k, (int) params.speculative.draft.cache_type_v,
        params.mmproj.path.c_str(), file_id(params.mmproj.path).c_str(), exe_id.c_str(), env_id.c_str());
    return string_format("%016llx", (unsigned long long) std::hash<std::string>{}(s)) + " " + s;
}

server_disk_cache::server_disk_cache(std::string dir, uint64_t quota_bytes, int32_t min_tokens, int32_t ttl_days,
                                     std::string fingerprint, bool has_mtmd, float ckpt_ratio, int32_t ckpt_min_step,
                                     int32_t ckpt_max)
    : root(std::move(dir)), quota(quota_bytes), n_min_tokens(min_tokens), ttl_days(ttl_days),
      fingerprint(std::move(fingerprint)), has_mtmd(has_mtmd), ckpt_ratio(ckpt_ratio),
      ckpt_min_step(std::max(1, ckpt_min_step)), ckpt_max(std::max(2, ckpt_max)) {
    this->dir = (fs::path(root) / ("ns-" + this->fingerprint.substr(0, 16))).string();
}

static int64_t mtime_s(const fs::path & p) {
    std::error_code ec;
    const auto mt = fs::last_write_time(p, ec);
    return ec ? 0 : std::chrono::duration_cast<std::chrono::seconds>(
        (mt - fs::file_time_type::clock::now()) + std::chrono::system_clock::now().time_since_epoch()).count();
}

// the fingerprint recorded in an entry's .meta, empty if unreadable
static std::string meta_fingerprint(const std::string & path) {
    try {
        std::ifstream fm(path);
        if (!fm) {
            return "";
        }
        const std::string text((std::istreambuf_iterator<char>(fm)), std::istreambuf_iterator<char>());
        return json::parse(text).value("fingerprint", std::string());
    } catch (const std::exception &) {
        return "";
    }
}

// the fingerprint hash16 in a .ck header (DCK3 and DCK4 alike), empty if unreadable
static std::string ckpt_fingerprint(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    uint32_t magic = 0;
    std::string fp;
    if (get(f, magic) && (magic == CKPT_MAGIC || magic == CKPT_ENV_MAGIC) && get_str(f, fp)) {
        return fp;
    }
    return "";
}

static const char * const ENTRY_EXTS[] = { ".meta", ".tok", ".tgt", ".dft", ".aux" };

std::string server_disk_cache::ckpt_path(const std::string & key) const {
    const auto it = ckpts.find(key);
    if (it != ckpts.end() && !it->second.src_dir.empty()) {
        return (fs::path(it->second.src_dir) / (key + ".ck")).string(); // [TAG_DISK_ADOPT] read-only source
    }
    return (fs::path(dir) / "ckpt" / (key + ".ck")).string();
}

std::string server_disk_cache::path(const std::string & id, const char * ext) const {
    const auto it = src_dir_of.find(id);
    return (fs::path(it != src_dir_of.end() ? it->second : dir) / (id + ext)).string();
}

void server_disk_cache::remove_files(const std::string & id) const {
    std::error_code ec;
    for (const char * ext : { ".meta", ".tok", ".tgt", ".dft", ".aux", ".meta.tmp", ".tok.tmp", ".aux.tmp", ".tgt.tmp", ".dft.tmp" }) {
        fs::remove(path(id, ext), ec);
    }
}

bool server_disk_cache::write_meta(const server_disk_cache_entry & e) const {
    json j = {
        { "version",     e.version ? e.version : DISK_CACHE_VERSION },
        { "fingerprint", fingerprint        },
        { "descriptor",  desc_h             },
        { "n_tokens",    e.tokens.size()    },
        { "bytes",       e.bytes            },
        { "t_saved",     e.t_saved          },
        { "t_used",      e.t_used           },
        { "hits",        e.hits             },
        { "has_dft",     e.has_dft          },
    };
    if (!e.src.empty()) {
        return false; // [TAG_DISK_ADOPT] an adopt source is never written
    }
    if (!e.adopted_from.empty()) {
        j["adopted_from"] = e.adopted_from;
    }
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
    // [TAG_DISK_NAMESPACE] the own quota: the own namespace + the blocks its entries reference
    uint64_t res = 0;
    std::set<std::string> refs;
    for (const auto & e : entries) {
        if (!e.src.empty()) {
            continue;
        }
        res += e.own_bytes; // .tok, .aux and the maps (which hold the small fields inline)
        refs.insert(e.blobs.begin(), e.blobs.end());
    }
    for (const auto & k : refs) {
        const auto it = blobs.find(k);
        res += it != blobs.end() ? it->second : 0;
    }
    std::set<std::string> ck_refs;
    if (gc_hold) {
        // [TAG_DISK_READ_STATUS] held for the entries skipped at start: an unreferenced checkpoint is not collected
        //   this run, so it must not push the quota into evicting entries that can be read
        for (const auto & e : entries) {
            ck_refs.insert(e.ckpt_keys.begin(), e.ckpt_keys.end());
            for (const auto & r : e.deep) {
                ck_refs.insert(r.key);
            }
        }
    }
    for (const auto & it : ckpts) {
        res += !gc_hold || ck_refs.count(it.first) ? it.second.bytes : 0;
    }
    return res;
}

void server_disk_cache::erase(size_t idx) {
    if (!entries[idx].src.empty()) {
        // [TAG_DISK_ADOPT] an adopt source's entry only loses its eligibility; its files stay
        src_dir_of.erase(entries[idx].id);
        entries.erase(entries.begin() + idx);
        return;
    }
    remove_files(entries[idx].id);
    entries.erase(entries.begin() + idx);
}

// [TAG_DISK_READ_STATUS] (Hermes gate Major 1) SEALED_IO: a file of the entry cannot be read now (open / read /
//   allocation error) - inconclusive, its files stay; SEALED_BAD: verified incomplete, damaged or another fingerprint
sealed_status server_disk_cache::read_entry(const std::string & id, const std::string & fp, server_disk_cache_entry & e) const {
    e.id = id;
    try {
        std::ifstream fm;
        if (const auto st = open_for_read(path(id, ".meta"), fm); st != SEALED_OK) {
            return st;
        }
        const std::string text((std::istreambuf_iterator<char>(fm)), std::istreambuf_iterator<char>());
        if (fm.bad()) {
            return SEALED_IO;
        }
        const json j = json::parse(text);
        const uint32_t ver = j.value("version", 0u);
        if (ver < 5 || ver > DISK_CACHE_VERSION || j.value("fingerprint", std::string()) != fp) {
            return SEALED_BAD;
        }
        e.version = ver;
        e.adopted_from = j.value("adopted_from", std::string());
        e.bytes   = j.at("bytes").get<uint64_t>();
        e.t_saved = j.at("t_saved").get<int64_t>();
        e.t_used  = j.at("t_used").get<int64_t>();
        e.hits    = j.at("hits").get<int32_t>();
        e.has_dft = j.at("has_dft").get<bool>();

        std::string raw;
        if (const auto st = read_sealed(path(id, ".tok"), raw); st != SEALED_OK) {
            return st;
        }
        if (raw.empty() || raw.size() % sizeof(llama_token) != 0) {
            return SEALED_BAD;
        }
        llama_tokens packed(raw.size() / sizeof(llama_token));
        memcpy(packed.data(), raw.data(), raw.size());
        e.tokens = server_tokens::deserialize(packed, has_mtmd);
        if (e.tokens.size() != j.at("n_tokens").get<size_t>()) {
            return SEALED_BAD;
        }
        for (const char * ext : { ".tgt", ".dft" }) {
            if (ext[1] == 'd' && !e.has_dft) {
                continue;
            }
            if (const auto st = nonempty_file(path(id, ext)); st != SEALED_OK) {
                return st;
            }
        }
        if (const auto st = read_aux(path(id, ".aux"), false, nullptr, nullptr, e.deep, e.ckpt_keys); st != SEALED_OK) {
            return st;
        }
        for (const char * ext : { ".tgt", ".dft" }) {
            if (ext[1] == 'd' && !e.has_dft) {
                continue;
            }
            sealed_status st = SEALED_BAD;
            if (!read_map_keys(path(id, ext), e.blobs, &st)) {
                return st;
            }
        }
        for (const char * ext : { ".tok", ".aux", ".tgt", ".dft" }) {
            e.own_bytes += file_size_or_0(path(id, ext));
        }
        return SEALED_OK;
    } catch (const std::bad_alloc & err) {
        DSK_WRN("entry %s not readable now: %s\n", id.c_str(), err.what());
        return SEALED_IO;
    } catch (const std::exception & err) {
        DSK_WRN("entry %s unreadable: %s\n", id.c_str(), err.what());
        return SEALED_BAD;
    }
}

void server_disk_cache::init() {
    std::error_code ec;
    fs::create_directories(fs::path(dir) / "ckpt", ec);
    if (ec) {
        DSK_WRN("cannot create %s: %s, disk cache disabled\n", dir.c_str(), ec.message().c_str());
        quota = 0;
        return;
    }
    const std::string key = fingerprint.substr(0, 16);
    gc_hold = false;

    // [TAG_DISK_NAMESPACE] flat entries / checkpoints of this very fingerprint move into the own namespace (rename,
    //   atomic per file; the .meta last, so an interrupted move leaves an incomplete flat entry that is skipped);
    //   flat files of another fingerprint stay where they are (legacy namespace: a rollback still finds them)
    {
        std::vector<fs::directory_entry> ls;
        std::set<std::string> flat;
        list_dir(root, ls, true); // a failed listing only postpones the move
        for (const auto & de : ls) {
            std::error_code ec_t;
            const std::string name = de.path().filename().string();
            if (de.is_regular_file(ec_t) && name.find('.') == 16) {
                flat.insert(name.substr(0, 16));
            }
        }
        for (const auto & id : flat) {
            if (meta_fingerprint((fs::path(root) / (id + ".meta")).string()) == fingerprint) {
                for (const char * ext : { ".tok", ".tgt", ".dft", ".aux", ".meta" }) {
                    std::error_code ec_m;
                    if (fs::exists(fs::path(root) / (id + ext), ec_m)) {
                        fs::rename(fs::path(root) / (id + ext), fs::path(dir) / (id + ext), ec_m);
                    }
                }
                DSK_INF("moved flat entry %s into namespace %s\n", id.c_str(), key.c_str());
            }
        }
        list_dir(fs::path(root) / "ckpt", ls, true);
        for (const auto & de : ls) {
            if (ckpt_fingerprint(de.path()) == key) {
                std::error_code ec_m;
                fs::rename(de.path(), fs::path(dir) / "ckpt" / de.path().filename(), ec_m);
            }
        }
        ec.clear();
    }

    // [TAG_DISK_READ_STATUS] (Hermes gate Major 1) every listing the start acts on is taken before anything is removed;
    //   a failed one disables the cache for this run with all files untouched - an empty view of blk/ would otherwise
    //   drop every entry as "blocks gone", an empty ckpt/ every checkpoint-bearing entry at its first load
    std::vector<fs::directory_entry> ls_own, ls_blk, ls_ck;
    {
        std::string err;
        std::error_code ec_b;
        fs::create_directories(fs::path(root) / "blk", ec_b);
        if (ec_b) {
            err = (fs::path(root) / "blk").string() + ": " + ec_b.message();
        }
        if (ec_b || !list_dir(dir, ls_own, false, &err) || !list_dir(fs::path(root) / "blk", ls_blk, false, &err) ||
                !list_dir(fs::path(dir) / "ckpt", ls_ck, false, &err)) {
            DSK_WRN("cannot list %s: disk cache disabled for this run, no file touched\n", err.c_str());
            quota = 0;
            entries.clear();
            ckpts.clear();
            blobs.clear();
            blob_bytes = 0;
            foreign.clear();
            adopted.clear();
            src_dir_of.clear();
            refresh_status();
            return;
        }
    }

    std::set<std::string> ids;
    for (const auto & de : ls_own) {
        const std::string name = de.path().filename().string();
        const auto dot = name.find('.');
        if (dot == 16) {
            ids.insert(name.substr(0, 16));
        }
    }

    size_t n_dropped = 0;
    size_t n_skipped = 0;
    for (const auto & id : ids) {
        server_disk_cache_entry e;
        const sealed_status st = read_entry(id, fingerprint, e);
        if (st == SEALED_OK) {
            entries.push_back(std::move(e));
        } else if (st == SEALED_IO) {
            DSK_WRN("entry %s not readable now (I/O error): kept on disk, skipped for this run\n", id.c_str());
            n_skipped++;
        } else {
            remove_files(id);
            n_dropped++;
        }
    }

    // block store: every complete block file (a crash leaves only .tmp files); unreferenced ones go in enforce()
    std::set<std::string> io_keys; // blocks / checkpoints present but not inspectable now: kept, not indexed
    for (const auto & de : ls_blk) {
        const std::string name = de.path().filename().string();
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".blk") == 0) {
            std::error_code ec_s;
            const uint64_t n = fs::file_size(de.path(), ec_s);
            if (ec_s) {
                io_keys.insert(name.substr(0, name.size() - 4));
                continue;
            }
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

    // deep checkpoint store: keep the files with this fingerprint
    size_t n_ck_dropped = 0;
    const std::string fp_short = fingerprint.substr(0, 16);
    for (const auto & de : ls_ck) {
        const std::string name = de.path().filename().string();
        bool ok = false;
        server_deep_ckpt_meta m;
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".ck") == 0) {
            std::ifstream f;
            if (open_for_read(de.path().string(), f) == SEALED_IO) {
                io_keys.insert(name.substr(0, name.size() - 3));
                continue;
            }
            uint32_t magic = 0;
            std::string fp;
            uint32_t env_ver = 0;
            std::string dh;
            ok = get(f, magic) && magic == CKPT_ENV_MAGIC && get_str(f, fp) && fp == fp_short && get(f, env_ver) &&
                 get_str(f, dh) && dh == desc_h && get_str(f, m.key) &&
                 get(f, m.n_tokens) && get(f, m.pos_min) && get(f, m.pos_max) && m.key + ".ck" == name;
            if (!ok && f.bad()) {
                io_keys.insert(name.substr(0, name.size() - 3));
                continue;
            }
            // the header is indexed without reading the 113 MiB body (its seal is checked on use): bound it, and the
            // key carries the token count, so a corrupt n_tokens cannot reach prefix_keys
            const auto dash = m.key.rfind('-');
            ok = ok && m.n_tokens > 0 && m.n_tokens < (1ll << 31) && m.pos_min >= 0 && m.pos_max >= m.pos_min &&
                 dash != std::string::npos && m.key.substr(dash + 1) == std::to_string(m.n_tokens);
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

    // entries whose blocks are gone (a crash between writes, a manual cleanup) cannot be loaded; one that references
    //   a block or checkpoint not inspectable now is skipped for this run with its files kept
    for (size_t i = 0; i < entries.size();) {
        const auto & e = entries[i];
        const auto io_ref = [&](const std::string & k) { return io_keys.count(k) > 0; };
        const bool io = std::any_of(e.blobs.begin(), e.blobs.end(), io_ref) ||
                        std::any_of(e.ckpt_keys.begin(), e.ckpt_keys.end(), io_ref) ||
                        std::any_of(e.deep.begin(), e.deep.end(), [&](const server_deep_ref & r) { return io_ref(r.key); });
        const bool complete = std::all_of(e.blobs.begin(), e.blobs.end(),
                [&](const std::string & k) { return blobs.count(k) > 0; });
        if (io) {
            DSK_WRN("entry %s references a file not readable now (I/O error): kept on disk, skipped for this run\n",
                    e.id.c_str());
            entries.erase(entries.begin() + i);
            n_skipped++;
        } else if (complete) {
            i++;
        } else {
            erase(i);
            n_dropped++;
        }
    }
    if (n_skipped > 0 || !io_keys.empty()) {
        gc_hold = true;
        DSK_WRN("%zu entries and %zu blocks / checkpoints skipped on I/O errors: no block or checkpoint is collected "
                "this run\n", n_skipped, io_keys.size());
    }

    open_adopt_sources();
    scan_foreign();

    enforce("");

    DSK_INF("own key %s (descriptor %s), %zu foreign namespaces\n", key.c_str(), desc_h.c_str(), foreign.size());
    for (const auto & ns : foreign) {
        // [TAG_DISK_ADOPT] what an operator passes to --cache-disk-adopt
        DSK_INF("foreign namespace %s%s: %zu entries, %.2f GiB, idle %.1f days\n", ns.key.c_str(),
                ns.legacy ? " (legacy)" : "", ns.n_entries, ns.bytes / (1024.0 * 1024.0 * 1024.0),
                (now_s() - ns.t_newest) / 86400.0);
    }
    DSK_INF("dir %s: %zu entries + %zu deep checkpoints, %.2f GiB (quota %.0f GiB, min %d tokens, ttl %d days, "
            "deep ratio %.2f min step %d max %d), %zu + %zu stale/foreign dropped\n",
            dir.c_str(), entries.size(), ckpts.size(), total_bytes() / (1024.0 * 1024.0 * 1024.0),
            quota / (1024.0 * 1024.0 * 1024.0), n_min_tokens, ttl_days, ckpt_ratio, ckpt_min_step, ckpt_max,
            n_dropped, n_ck_dropped);

    refresh_status();
}

void server_disk_cache::enforce(const std::string & protect) {
    const int64_t t_now = now_s();

    if (ttl_days > 0) {
        for (size_t i = 0; i < entries.size();) {
            if (entries[i].src.empty() && entries[i].id != protect && entries[i].id != protect_id && t_now - entries[i].t_used > (int64_t) ttl_days * 86400) {
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
            if (entries[i].id == protect || entries[i].id == protect_id || !entries[i].src.empty()) {
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

    refresh_status();
}

bool server_disk_cache::save(llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, int32_t id_slot,
                             const server_prompt & prompt) {
    const size_t n = prompt.tokens.size();
    if (quota == 0 || n < (size_t) n_min_tokens) {
        return false;
    }
    if (prompt.noncanonical) { // [TAG_NONCANONICAL]
        DSK_INF("not saving %zu tokens: the state was shifted (not the state of this prefix)\n", n);
        return false;
    }
    if (prompt.partial) { // [TAG_PARTIAL_PROMPT]
        DSK_INF("not saving %zu tokens: a partial prompt (legacy /slots restore, draft side incomplete)\n", n);
        return false;
    }

    // already stored: exactly this prompt (a longer entry does not count: its recurrent state is at its own end,
    // so this prefix could only come back through a checkpoint)
    for (auto & e : entries) {
        if (e.src.empty() && e.tokens.size() == n && e.tokens.get_common_prefix(prompt.tokens) == n) {
            e.t_used = now_s();
            write_meta(e);
            return true;
        }
    }

    for (const auto & ckpt : prompt.checkpoints) {
        // [TAG_NONSPLIT_REPLACE] a checkpoint must lie inside this prompt (its key is a prefix of these tokens)
        if (ckpt.n_tokens < 0 || ckpt.n_tokens > (int64_t) n) {
            DSK_WRN("a checkpoint at %" PRId64 " tokens lies outside the %zu-token prompt, not saving\n", ckpt.n_tokens, n);
            return false;
        }
        if ((ckpt.flags_tgt | ckpt.flags_dft) & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
            DSK_WRN("%s", "a checkpoint lives on the device, not saving\n");
            return false;
        }
    }

    const int64_t t_start = ggml_time_us();

    server_disk_cache_entry e;
    e.id      = random_id();
    e.version = DISK_CACHE_VERSION;
    if (slot_adopted.count(id_slot)) {
        e.adopted_from = slot_adopted[id_slot]; // [TAG_DISK_ADOPT] provenance of a resave after an adopted load
    }
    e.has_dft = ctx_dft != nullptr;

    uint64_t n_ckpt_written = 0;

    bool ok = true;
    try {
        // tokens
        {
            const std::vector<char> packed = prompt.tokens.serialize();
            ok = ok && write_sealed(path(e.id, ".tok"), std::string(packed.begin(), packed.end()));
        }
        // checkpoints + speculative state
        if (ok) {
            std::ostringstream f;
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
            // [TAG_TURN_BOUNDARY] v6 trailer: the user-message starts inside this prompt
            std::vector<int32_t> users;
            for (const int32_t p : prompt.user_starts) {
                if (p >= 0 && p < (int32_t) n) {
                    users.push_back(p);
                }
            }
            put<uint32_t>(f, (uint32_t) users.size());
            for (const int32_t p : users) {
                put<int32_t>(f, p);
            }
            ok = ok && (bool) f && write_sealed(path(e.id, ".aux"), f.str());
        }
    } catch (const std::exception & err) {
        DSK_WRN("cannot serialize the prompt: %s\n", err.what());
        ok = false;
    }

    // [TAG_DISK_DELTA] the KV / recurrent state streams through 4 MiB content blocks: a block already on disk (the
    //   unchanged part of a conversation saved before, a branch, a copy) is referenced, only new ones are written
    uint64_t n_logical = 0, n_written = 0;
    ok = ok && save_stream(ctx_tgt, id_slot, path(e.id, ".tgt"), desc_tgt_h, e.blobs, n_logical, n_written);
    if (ok && ctx_dft) {
        ok = save_stream(ctx_dft, id_slot, path(e.id, ".dft"), desc_dft_h, e.blobs, n_logical, n_written);
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
        // [TAG_DISK_GC] the aborted save is a transaction boundary: blocks and checkpoints it wrote that no
        //   registered entry or live slot references go now (never inside erase(): during a prefix replacement the
        //   new entry's files exist before it is registered)
        deep_gc();
        blob_gc();
        refresh_status();
        return false;
    }

    // older saves of the same conversation that this one extends are obsolete
    for (size_t i = 0; i < entries.size();) {
        if (entries[i].tokens.size() < n && entries[i].id != protect_id &&
            prompt.tokens.get_common_prefix(entries[i].tokens) == entries[i].tokens.size()) {
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
    bytes_written += n_written + n_ckpt_written + file_size_or_0(path(e.id, ".aux")) + file_size_or_0(path(e.id, ".tok"));
    save_us       += (uint64_t) (ggml_time_us() - t_start);
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
        // [TAG_DISK_ADOPT] the own namespace wins a tie with an adopt source
        if (lcp > lcp_best || (lcp == lcp_best && best >= 0 && entries[i].src.empty() && !entries[best].src.empty())) {
            lcp_best = lcp;
            best     = (int) i;
        }
    }
    if (lcp_out) {
        *lcp_out = best >= 0 ? lcp_best : 0;
    }
    return best;
}

int server_disk_cache::find_id(const std::string & id) const {
    for (size_t i = 0; i < entries.size(); i++) {
        if (entries[i].id == id) {
            return (int) i;
        }
    }
    return -1;
}

void server_disk_cache::drop_invalid(int idx) {
    const std::string src = entries[idx].src;
    for (auto & a : adopted) {
        a.dropped += a.key == src;
    }
    erase(idx);
    n_load_failed++;
    deep_gc(); // [TAG_DISK_GC] the dropped entry's checkpoints and blocks may now be unreferenced
    blob_gc();
}

server_disk_load_result server_disk_cache::verify_stream(const std::string & map_path, const std::string & env_h,
                                                         bool legacy_ok) {
    std::ifstream fm;
    const sealed_status st_open = open_for_read(map_path, fm);
    if (st_open != SEALED_OK) {
        return st_open == SEALED_IO ? SERVER_DISK_LOAD_OTHER : SERVER_DISK_LOAD_INVALID;
    }
    uint32_t magic = 0;
    uint64_t n_total = 0;
    if (!get(fm, magic) || (magic != MAP_ENV_MAGIC && !(legacy_ok && magic == MAP_MAGIC))) {
        return fm.bad() ? SERVER_DISK_LOAD_OTHER : SERVER_DISK_LOAD_INVALID;
    }
    if (magic == MAP_ENV_MAGIC) {
        uint32_t ver = 0;
        std::string h;
        if (!get(fm, ver) || ver != ENVELOPE_VERSION || !get_str(fm, h) || h != env_h || !get(fm, n_total) || n_total == 0) {
            return fm.bad() ? SERVER_DISK_LOAD_OTHER : SERVER_DISK_LOAD_INVALID;
        }
    }
    uint64_t n_seen = 0;
    std::vector<uint8_t> buf;
    try {
        while (true) {
            uint8_t kind = 0;
            if (!get(fm, kind)) {
                break;
            }
            if (kind == 255) {
                return n_total == 0 || n_seen == n_total ? SERVER_DISK_LOAD_OK : SERVER_DISK_LOAD_INVALID;
            }
            if (kind == 0) {
                uint64_t n = 0;
                std::string h;
                if (!get(fm, n) || n > (1ull << 32)) {
                    break;
                }
                buf.resize(n);
                if (!fm.read(reinterpret_cast<char *>(buf.data()), n) || !get_str(fm, h)) {
                    break;
                }
                if (hash_key(buf.data(), n) != h) {
                    DSK_WRN("%s: inline segment corrupt\n", map_path.c_str());
                    return SERVER_DISK_LOAD_INVALID;
                }
                n_seen += n;
                continue;
            }
            std::string key;
            if (kind != 1 || !get_str(fm, key)) {
                break;
            }
            const auto it = blobs.find(key);
            if (it == blobs.end()) {
                DSK_WRN("block %s is missing\n", key.c_str());
                return SERVER_DISK_LOAD_INVALID;
            }
            buf.resize(it->second);
            std::ifstream fb;
            const sealed_status st = open_for_read(blob_path(key), fb);
            const bool got = st == SEALED_OK && (bool) fb.read(reinterpret_cast<char *>(buf.data()), buf.size());
            if (st == SEALED_IO || (st == SEALED_OK && !got && fb.bad())) {
                DSK_WRN("block %s: read error, keeping it\n", key.c_str());
                return SERVER_DISK_LOAD_OTHER;
            }
            if (!got || hash_key(buf.data(), buf.size()) != key) {
                DSK_WRN("block %s is corrupt, dropping it\n", key.c_str());
                std::error_code ec;
                fs::remove(blob_path(key), ec);
                blob_bytes -= it->second;
                blobs.erase(it);
                return SERVER_DISK_LOAD_INVALID;
            }
            n_seen += buf.size();
        }
    } catch (const std::bad_alloc &) {
        return SERVER_DISK_LOAD_OTHER;
    }
    // the map ended before its end marker
    return fm.bad() ? SERVER_DISK_LOAD_OTHER : SERVER_DISK_LOAD_INVALID;
}

server_disk_load_result server_disk_cache::verify(int idx) {
    auto & e = entries[idx];
    const std::string id = e.id;

    server_disk_load_result res = SERVER_DISK_LOAD_OK;
    {
        std::list<common_prompt_checkpoint> checkpoints;
        std::vector<uint8_t> spec_state;
        std::vector<server_deep_ref> deep;
        std::vector<std::string> keys;
        sealed_status st = SEALED_OK;
        try {
            st = read_aux(path(id, ".aux"), true, &checkpoints, &spec_state, deep, keys);
        } catch (const std::bad_alloc &) {
            st = SEALED_IO;
        }
        if (st == SEALED_IO) {
            res = SERVER_DISK_LOAD_OTHER;
        } else if (st != SEALED_OK || keys.size() != checkpoints.size()) {
            res = SERVER_DISK_LOAD_INVALID;
        }
        // the prompt checkpoints load() restores: present in the index and sealed intact
        for (size_t i = 0; res == SERVER_DISK_LOAD_OK && i < keys.size(); i++) {
            const auto it = ckpts.find(keys[i]);
            if (it == ckpts.end()) {
                res = SERVER_DISK_LOAD_INVALID;
                break;
            }
            std::string body;
            const sealed_status sc = read_sealed(ckpt_path(keys[i]), body);
            if (sc != SEALED_OK) {
                res = sc == SEALED_IO ? SERVER_DISK_LOAD_OTHER : SERVER_DISK_LOAD_INVALID;
            }
        }
    }
    if (res == SERVER_DISK_LOAD_OK) {
        res = verify_stream(path(id, ".tgt"), desc_tgt_h, !e.src.empty());
    }
    if (res == SERVER_DISK_LOAD_OK && e.has_dft) {
        res = verify_stream(path(id, ".dft"), desc_dft_h, !e.src.empty());
    }

    if (res == SERVER_DISK_LOAD_INVALID) {
        DSK_WRN("entry %s: verification failed (invalid), dropping it\n", id.c_str());
        drop_invalid(idx);
    } else if (res == SERVER_DISK_LOAD_OTHER) {
        DSK_WRN("entry %s: verification inconclusive, keeping it\n", id.c_str());
        n_load_other++;
    }
    refresh_status();
    return res;
}

server_disk_load_result server_disk_cache::load(int idx, llama_context * ctx_tgt, llama_context * ctx_dft,
                                                common_speculative * spec, int32_t id_slot, server_prompt & prompt) {
    auto & e = entries[idx];

    const int64_t t_start = ggml_time_us();

    std::list<common_prompt_checkpoint> checkpoints;
    std::vector<uint8_t> spec_state;
    std::vector<server_deep_ref> deep;
    std::vector<int32_t> user_starts;
    // [TAG_DISK_READ_STATUS] only verified damage (aux_invalid) drops the entry; an I/O or allocation failure while
    //   reading the .aux or a checkpoint file is inconclusive: the entry and its files stay
    struct aux_invalid : std::runtime_error {
        using std::runtime_error::runtime_error;
    };
    try {
        std::vector<std::string> keys;
        const sealed_status st = read_aux(path(e.id, ".aux"), true, &checkpoints, &spec_state, deep, keys, &user_starts);
        if (st == SEALED_IO) {
            throw std::runtime_error("read error");
        }
        if (st != SEALED_OK || keys.size() != checkpoints.size()) {
            throw aux_invalid("truncated or corrupt");
        }
        size_t i = 0;
        for (auto & c : checkpoints) {
            common_prompt_checkpoint full;
            bool io_error = false;
            if (!deep_get(keys[i++], full, &io_error) || full.n_tokens != c.n_tokens) {
                if (io_error) {
                    throw std::runtime_error("a checkpoint read error");
                }
                throw aux_invalid("a checkpoint is missing");
            }
            c = std::move(full);
        }
    } catch (const aux_invalid & err) {
        DSK_WRN("entry %s: unreadable .aux (%s), dropping it\n", e.id.c_str(), err.what());
        erase(idx);
        n_load_failed++;
        deep_gc(); // [TAG_DISK_GC] the dropped entry's checkpoints and blocks may now be unreferenced
        blob_gc();
        refresh_status();
        return SERVER_DISK_LOAD_INVALID;
    } catch (const std::exception & err) {
        DSK_WRN("entry %s: .aux or checkpoint not readable (%s), keeping it\n", e.id.c_str(), err.what());
        n_load_other++;
        refresh_status();
        return SERVER_DISK_LOAD_OTHER;
    }

    llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
    server_disk_load_result res = load_stream(ctx_tgt, id_slot, path(e.id, ".tgt"), desc_tgt_h, !e.src.empty());
    if (res == SERVER_DISK_LOAD_OK && ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        res = e.has_dft ? load_stream(ctx_dft, id_slot, path(e.id, ".dft"), desc_dft_h, !e.src.empty()) : SERVER_DISK_LOAD_INVALID;
    }
    if (res != SERVER_DISK_LOAD_OK) {
        // [TAG_DISK_LOAD_RESULT] the caller clears the slot; only an invalid entry goes
        if (res == SERVER_DISK_LOAD_INVALID) {
            DSK_WRN("entry %s: state load failed (invalid), dropping it%s\n", e.id.c_str(),
                    e.src.empty() ? "" : " from the adopt sources (its files stay)");
            drop_invalid(idx);
        } else if (res == SERVER_DISK_LOAD_CAPACITY) {
            DSK_WRN("entry %s: state load failed (no room in the KV cache), keeping it\n", e.id.c_str());
            n_load_capacity++;
        } else {
            DSK_WRN("entry %s: state load failed (inconclusive), keeping it\n", e.id.c_str());
            n_load_other++;
        }
        refresh_status();
        return res;
    }

    if (spec && !spec_state.empty()) {
        common_speculative_set_state(spec, id_slot, spec_state);
    }

    prompt.tokens      = e.tokens.clone();
    prompt.checkpoints = std::move(checkpoints);
    prompt.user_starts = std::move(user_starts); // [TAG_TURN_BOUNDARY]
    prompt.deep.clear();
    for (auto & r : deep) {
        if (ckpts.count(r.key)) { // a collected checkpoint is just a gap in the schedule
            prompt.deep.push_back(std::move(r));
        }
    }

    e.t_used = now_s();
    e.hits++;
    write_meta(e); // a no-op for an adopt source
    n_loads++;
    slot_adopted.erase(id_slot);
    if (!e.src.empty()) {
        slot_adopted[id_slot] = e.src;
        for (auto & a : adopted) {
            a.hits += a.key == e.src;
        }
        DSK_INF("adopted load: entry %s from source %s (format %u)\n", e.id.c_str(), e.src.c_str(), e.version);
    }
    bytes_loaded += e.bytes;
    load_us      += (uint64_t) (ggml_time_us() - t_start);
    refresh_status();

    const double ms = (ggml_time_us() - t_start) / 1000.0;
    DSK_INF("loaded %s: %zu tokens, %zu checkpoints, %.1f MiB in %.0f ms (%.2f GB/s, hit %d)\n", e.id.c_str(),
            prompt.tokens.size(), prompt.checkpoints.size(), e.bytes / (1024.0 * 1024.0), ms,
            e.bytes / std::max(1e-3, ms) / 1e6, e.hits);

    return SERVER_DISK_LOAD_OK;
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
    if (it != ckpts.end() && it->second.src_dir.empty()) {
        it->second.t_used = t_now;
        return true;
    }
    if (it != ckpts.end()) {
        ckpts.erase(it); // [TAG_DISK_ADOPT] the source's file stays; the own namespace gets its own copy
    }
    const std::string fin = ckpt_path(key);
    const std::string tmp = fin + ".tmp";
    bool ok = false;
    {
        std::ostringstream f;
        put<uint32_t>(f, CKPT_ENV_MAGIC);
        put_str(f, fingerprint.substr(0, 16));
        put<uint32_t>(f, ENVELOPE_VERSION);
        put_str(f, desc_h);
        put_str(f, key);
        put<int64_t>(f, c.n_tokens);
        put<int32_t>(f, c.pos_min);
        put<int32_t>(f, c.pos_max);
        put<uint32_t>(f, c.flags_tgt);
        put<uint32_t>(f, c.flags_dft);
        put_vec(f, c.data_tgt);
        put_vec(f, c.data_dft);
        put_vec(f, c.data_spec);
        ok = (bool) f && write_sealed(tmp, f.str());
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
    bytes_written += m.bytes;
    if (written) {
        *written = true;
    }
    return true;
}

void server_disk_cache::deep_put(server_prompt & prompt, const common_prompt_checkpoint & c, uint8_t kind) {
    if (prompt.noncanonical || prompt.partial) { // [TAG_NONCANONICAL] [TAG_PARTIAL_PROMPT]
        return;
    }
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

bool server_disk_cache::deep_get(const std::string & key, common_prompt_checkpoint & out, bool * io_error) {
    auto it = ckpts.find(key);
    if (it == ckpts.end()) {
        return false;
    }
    try {
        std::string body;
        const sealed_status st = read_sealed(ckpt_path(key), body);
        if (st == SEALED_IO) {
            // [TAG_DISK_READ_STATUS] inconclusive: the checkpoint and its file stay
            DSK_WRN("deep checkpoint %s: read error, keeping it\n", key.c_str());
            if (io_error) {
                *io_error = true;
            }
            return false;
        }
        if (st != SEALED_OK) {
            throw std::runtime_error("checksum mismatch or truncated");
        }
        std::istringstream f(body);
        uint32_t magic = 0;
        std::string fp, k;
        int64_t  n_tokens = 0;
        int32_t  pos_min = 0, pos_max = 0;
        uint32_t ft = 0, fd = 0;
        uint32_t env_ver = 0;
        std::string dh;
        // [TAG_DISK_ADOPT] an adopt source's format 5 / 6 checkpoint has no envelope: accepted on the operator's word
        const bool from_src = !it->second.src_dir.empty();
        bool ok = get(f, magic) && (magic == CKPT_ENV_MAGIC || (from_src && magic == CKPT_MAGIC)) && get_str(f, fp);
        if (ok && magic == CKPT_ENV_MAGIC) {
            ok = get(f, env_ver) && get_str(f, dh);
        } else {
            env_ver = ENVELOPE_VERSION;
            dh      = desc_h;
        }
        // [TAG_STATE_DESCRIPTOR] checked before anything reaches the contexts: a checkpoint of another state layout
        //   is a miss (it would otherwise abort in common_prompt_checkpoint::load_tgt on a size mismatch)
        if (ok && (env_ver != ENVELOPE_VERSION || dh != desc_h)) {
            throw std::runtime_error("state descriptor mismatch (" + dh + ", expected " + desc_h + ")");
        }
        ok = ok && get_str(f, k) && k == key &&
             get(f, n_tokens) && get(f, pos_min) && get(f, pos_max) && get(f, ft) && get(f, fd) &&
             get_vec(f, out.data_tgt) && get_vec(f, out.data_dft) && get_vec(f, out.data_spec) &&
             f.peek() == std::char_traits<char>::eof();
        if (!ok) {
            throw std::runtime_error("truncated or corrupt");
        }
        out.n_tokens  = n_tokens;
        out.pos_min   = pos_min;
        out.pos_max   = pos_max;
        out.flags_tgt = ft;
        out.flags_dft = fd;
        out.id_task   = -1;
    } catch (const std::bad_alloc &) {
        // [TAG_DISK_READ_STATUS] inconclusive: the checkpoint and its file stay
        DSK_WRN("deep checkpoint %s: out of memory reading it, keeping it\n", key.c_str());
        if (io_error) {
            *io_error = true;
        }
        return false;
    } catch (const std::exception & err) {
        DSK_WRN("deep checkpoint %s unreadable (%s), dropping it\n", key.c_str(), err.what());
        std::error_code ec;
        if (it->second.src_dir.empty()) {
            fs::remove(ckpt_path(key), ec);
        }
        ckpts.erase(it);
        return false;
    }
    it->second.t_used = now_s();
    it->second.hits++;
    std::error_code ec;
    if (it->second.src_dir.empty()) {
        fs::last_write_time(ckpt_path(key), fs::file_time_type::clock::now(), ec); // t_used survives a restart
    }
    return true;
}

void server_disk_cache::deep_gc() {
    if (gc_hold) {
        return; // [TAG_DISK_READ_STATUS] an entry skipped at start may reference any of them
    }
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
        if (!referenced.count(it->first) && it->second.src_dir.empty()) {
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
    return (fs::path(root) / "blk" / (key + ".blk")).string();
}

bool server_disk_cache::save_stream(llama_context * ctx, int32_t id_slot, const std::string & map_path,
                                    const std::string & env_h, std::vector<std::string> & keys, uint64_t & n_logical,
                                    uint64_t & n_written) {
    const std::string tmp = map_path + ".tmp";
    std::ofstream fm(tmp, std::ios::binary | std::ios::trunc);
    // [TAG_STATE_DESCRIPTOR] envelope: magic, version, descriptor hash, stream length (patched at the end)
    put<uint32_t>(fm, MAP_ENV_MAGIC);
    put<uint32_t>(fm, ENVELOPE_VERSION);
    put_str(fm, env_h);
    const auto len_at = fm.tellp();
    put<uint64_t>(fm, 0);
    const uint64_t n_logical0 = n_logical;

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
        put_str(fm, hash_key(inl.data(), inl.size())); // cell metadata, types: checked on load like the blocks
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

    const size_t n_stream = llama_state_seq_write_cb(ctx, id_slot, 0, w);
    ok = n_stream > 0 && n_stream == n_logical - n_logical0 && ok;
    flush_inline();
    put<uint8_t>(fm, 255);
    fm.seekp(len_at);
    put<uint64_t>(fm, (uint64_t) n_stream);
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

bool server_disk_cache::read_map_keys(const std::string & map_path, std::vector<std::string> & keys, sealed_status * st) {
    // st: SEALED_IO when the map could not be read now (open / read error), SEALED_BAD when it is missing or malformed
    std::ifstream fm;
    const sealed_status st_open = open_for_read(map_path, fm);
    const auto fail = [&](sealed_status res) {
        if (st) {
            *st = res;
        }
        return false;
    };
    if (st_open != SEALED_OK) {
        return fail(st_open);
    }
    const auto bad = [&]() { return fail(fm.bad() ? SEALED_IO : SEALED_BAD); };
    uint32_t magic = 0;
    if (!get(fm, magic) || (magic != MAP_MAGIC && magic != MAP_ENV_MAGIC)) {
        return bad();
    }
    if (magic == MAP_ENV_MAGIC) {
        uint32_t ver = 0;
        std::string h;
        uint64_t n = 0;
        if (!get(fm, ver) || !get_str(fm, h) || !get(fm, n)) {
            return bad();
        }
    }
    for (;;) {
        uint8_t kind = 0;
        if (!get(fm, kind)) {
            return bad();
        }
        if (kind == 255) {
            if (st) {
                *st = SEALED_OK;
            }
            return true;
        }
        if (kind == 0) {
            uint64_t n = 0;
            std::string h;
            if (!get(fm, n)) {
                return bad();
            }
            fm.seekg((std::streamoff) n, std::ios::cur);
            if (!get_str(fm, h)) {
                return bad();
            }
        } else if (kind == 1) {
            std::string key;
            if (!get_str(fm, key)) {
                return bad();
            }
            keys.push_back(std::move(key));
        } else {
            return bad();
        }
    }
}

server_disk_load_result server_disk_cache::load_stream(llama_context * ctx, int32_t id_slot, const std::string & map_path,
                                                       const std::string & env_h, bool legacy_ok) {
    std::ifstream fm;
    if (open_for_read(map_path, fm) == SEALED_IO) {
        // [TAG_DISK_READ_STATUS] the map exists but cannot be opened: inconclusive, the entry stays
        DSK_WRN("%s: cannot open the state map\n", map_path.c_str());
        return SERVER_DISK_LOAD_OTHER;
    }
    uint32_t magic = 0;
    if (!get(fm, magic) || (magic != MAP_ENV_MAGIC && !(legacy_ok && magic == MAP_MAGIC))) {
        DSK_WRN("%s: no state envelope\n", map_path.c_str());
        return SERVER_DISK_LOAD_INVALID;
    }
    // [TAG_STATE_DESCRIPTOR] the stream must describe this context's state, checked before the context is touched
    uint64_t n_total = 0;
    if (magic == MAP_ENV_MAGIC) {
        uint32_t ver = 0;
        std::string h;
        if (!get(fm, ver) || ver != ENVELOPE_VERSION || !get_str(fm, h) || h != env_h || !get(fm, n_total) || n_total == 0) {
            DSK_WRN("%s: state envelope mismatch (descriptor %s, expected %s)\n", map_path.c_str(), h.c_str(), env_h.c_str());
            return SERVER_DISK_LOAD_INVALID;
        }
    }
    uint64_t n_consumed = 0;

    // the current segment: inline bytes come from the map itself, a block from its (verified) buffer
    uint64_t seg_left = 0;
    bool     seg_inline = false;
    std::vector<uint8_t> blk;
    size_t   blk_pos = 0;
    bool done = false;
    bool corrupt = false;
    bool io_error = false; // [TAG_DISK_READ_STATUS] a read error (not verified damage): inconclusive, nothing dropped

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
            // an inline segment is read whole and checked against the hash that follows it
            uint64_t n = 0;
            std::string h;
            if (!get(fm, n) || n > (1ull << 32)) {
                return false;
            }
            blk.resize(n);
            if (!fm.read(reinterpret_cast<char *>(blk.data()), n) || !get_str(fm, h) || hash_key(blk.data(), n) != h) {
                DSK_WRN("%s: inline segment corrupt\n", map_path.c_str());
                corrupt = true;
                return false;
            }
            seg_inline = false; // served from blk like a block
            seg_left   = n;
            blk_pos    = 0;
            return true;
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
            std::ifstream fb;
            const sealed_status st = open_for_read(blob_path(key), fb);
            const bool got = st == SEALED_OK && (bool) fb.read(reinterpret_cast<char *>(blk.data()), blk.size());
            // [TAG_DISK_READ_STATUS] an open / read error is inconclusive and keeps the block; a missing file, a short
            //   file (end of file before the indexed size) or a hash mismatch is verified damage
            if (st == SEALED_IO || (st == SEALED_OK && !got && fb.bad())) {
                DSK_WRN("block %s: read error, keeping it\n", key.c_str());
                io_error = true;
                return false;
            }
            if (!got || hash_key(blk.data(), blk.size()) != key) {
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

    bool read_failed = false; // the stream itself (map, inline segment, block) could not supply the bytes

    llama_state_seq_reader r;
    r.read = [&](void * dst, size_t size) -> bool {
        auto * out = static_cast<char *>(dst);
        while (size > 0) {
            if (seg_left == 0 && !next_segment()) {
                read_failed = true;
                return false;
            }
            const size_t n = (size_t) std::min<uint64_t>(seg_left, size);
            if (seg_inline) {
                if (!fm.read(out, n)) {
                    read_failed = true;
                    return false;
                }
            } else {
                memcpy(out, blk.data() + blk_pos, n);
                blk_pos += n;
            }
            out      += n;
            size     -= n;
            seg_left -= n;
            n_consumed += n;
        }
        return true;
    };

    // [TAG_DISK_LOAD_RESULT] [TAG_STATE_READ_STATUS] the library returns why the read failed: NO_SPACE ("failed to
    //   find ... available cells", KV and recurrent memory alike) is a capacity failure, INVALID a stream that
    //   contradicts this context; no log hook, so nothing process-global is touched
    llama_state_read_status st = LLAMA_STATE_READ_OK;
    const bool ok = llama_state_seq_read_cb(ctx, id_slot, 0, r, &st) > 0;

    if (ok && !corrupt) {
        // full consumption: nothing of the stream is left unread (an unread suffix is state this context dropped)
        const bool rest_empty = seg_left == 0 && !next_segment() && done && !corrupt;
        if (!rest_empty && (io_error || fm.bad())) {
            return SERVER_DISK_LOAD_OTHER; // [TAG_DISK_READ_STATUS]
        }
        if (!rest_empty || (n_total > 0 && n_consumed != n_total)) {
            DSK_WRN("%s: stream not fully consumed (%" PRIu64 " of %" PRIu64 " bytes)\n", map_path.c_str(), n_consumed, n_total);
            return SERVER_DISK_LOAD_INVALID;
        }
        return SERVER_DISK_LOAD_OK;
    }
    if (io_error || fm.bad()) {
        return SERVER_DISK_LOAD_OTHER;
    }
    if (corrupt || read_failed) {
        return SERVER_DISK_LOAD_INVALID;
    }
    return st == LLAMA_STATE_READ_NO_SPACE ? SERVER_DISK_LOAD_CAPACITY
         : st == LLAMA_STATE_READ_INVALID  ? SERVER_DISK_LOAD_INVALID
                                           : SERVER_DISK_LOAD_OTHER;
}

void server_disk_cache::blob_gc() {
    if (gc_hold) {
        return; // [TAG_DISK_READ_STATUS] an entry skipped at start may reference any of them
    }
    std::set<std::string> referenced;
    for (const auto & e : entries) {
        referenced.insert(e.blobs.begin(), e.blobs.end());
    }
    // [TAG_DISK_NAMESPACE] blocks of every other namespace on disk (incl. other live processes) stay: rescan them
    //   only when something would be deleted; a failed rescan deletes nothing
    if (std::any_of(blobs.begin(), blobs.end(), [&](const auto & b) { return !referenced.count(b.first); })) {
        if (!scan_foreign()) {
            return;
        }
        for (const auto & ns : foreign) {
            referenced.insert(ns.blocks.begin(), ns.blocks.end());
        }
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

void server_disk_cache::revalidate(const std::string & fp) {
    if (fp == fingerprint) {
        return;
    }
    // [TAG_DISK_NAMESPACE] a mismatch is a miss, never a delete: open the new fingerprint's namespace, the old one
    //   stays as a foreign namespace (TTL / cap)
    DSK_WRN("the model / context / flags changed since the cache was opened: switching namespace %s -> %s\n",
            fingerprint.substr(0, 16).c_str(), fp.substr(0, 16).c_str());
    fingerprint = fp;
    dir = (fs::path(root) / ("ns-" + fp.substr(0, 16))).string();
    entries.clear();
    ckpts.clear();
    blobs.clear();
    blob_bytes = 0;
    init();
}

void server_disk_cache::refresh_status() {
    json list = json::array();
    for (const auto & e : entries) {
        list.push_back({
            { "id",          e.id                 },
            { "n_tokens",    e.tokens.size()      },
            { "state_bytes", e.bytes              },
            { "own_bytes",   e.own_bytes          },
            { "n_deep",      e.deep.size()        },
            { "t_saved",     e.t_saved            },
            { "t_used",      e.t_used             },
            { "hits",        e.hits               },
            { "version",     e.version            },
            { "source",      e.src                },
            { "adopted_from", e.adopted_from      },
            { "value",       entry_value(e, now_s()) },
        });
    }
    json fl = json::array();
    for (const auto & ns : foreign) {
        fl.push_back({
            { "key",       ns.key       },
            { "legacy",    ns.legacy    },
            { "n_entries", ns.n_entries },
            { "bytes",     ns.bytes     },
            { "age_s",     now_s() - ns.t_newest },
        });
    }
    json al = json::array();
    for (const auto & a : adopted) {
        al.push_back({ { "key", a.key }, { "legacy", a.legacy }, { "n_entries", a.n_entries }, { "hits", a.hits },
                       { "dropped", a.dropped } });
    }
    json j = {
        { "enabled",        quota > 0           },
        { "adopt",          al                  },
        { "key",            fingerprint.substr(0, 16) },
        { "descriptor",     desc_h              },
        { "foreign",        fl                  },
        { "dir",            dir                 },
        { "quota_bytes",    quota               },
        { "bytes_on_disk",  total_bytes()       },
        { "min_tokens",     n_min_tokens        },
        { "ttl_days",       ttl_days            },
        { "ckpt_ratio",     ckpt_ratio          },
        { "ckpt_max",       ckpt_max            },
        { "n_entries",      entries.size()      },
        { "n_deep_ckpts",   ckpts.size()        },
        { "n_blocks",       blobs.size()        },
        { "block_bytes",    blob_bytes          },
        { "saves",          n_saves             },
        { "loads",          n_loads             },
        { "load_failures",  n_load_failed       },
        { "load_failures_capacity", n_load_capacity },
        { "load_failures_other",    n_load_other    },
        { "deep_rewinds",   n_deep_rewinds      },
        { "evictions",      n_evict             },
        { "bytes_written",  bytes_written       },
        { "bytes_loaded",   bytes_loaded        },
        { "save_seconds",   save_us / 1e6       },
        { "load_seconds",   load_us / 1e6       },
        { "entries",        list                },
    };
    std::lock_guard<std::mutex> lock(status_mtx);
    status_snap = std::move(j);
}

json server_disk_cache::status() const {
    std::lock_guard<std::mutex> lock(status_mtx);
    return status_snap;
}

void server_disk_cache::set_descriptor(llama_context * ctx_tgt, llama_context * ctx_dft, const std::string & spec_types) {
    // the speculative carry: implementation names + the MTP pending-hidden schema (magic "MPTH", one f32 row of n_embd)
    const std::string srv = string_format("|carry=%s/mtp-pending-h:4854504d-f32|tokens=%u|mtmd=%d|ckpt=%u",
                                          spec_types.c_str(), SERVER_TOKENS_STATE_VERSION, MTMD_SERIALIZATION_VERSION,
                                          SERVER_CKPT_SEMANTICS_REV);
    const auto h16 = [](const std::string & s) {
        return hash_key(reinterpret_cast<const uint8_t *>(s.data()), s.size()).substr(0, 16);
    };
    const std::string tgt = "tgt:" + (ctx_tgt ? llama_state_seq_descriptor(ctx_tgt) : std::string("none")) + srv;
    const std::string dft = "dft:" + (ctx_dft ? llama_state_seq_descriptor(ctx_dft) : std::string("none")) + srv;
    desc_tgt_h = h16(tgt);
    desc_dft_h = h16(dft);
    desc_h     = h16(tgt + "\n" + dft);
    DSK_INF("state descriptor %s (target %s, draft %s)\n", desc_h.c_str(), desc_tgt_h.c_str(), desc_dft_h.c_str());
    DSK_DBG("target descriptor: %s\n", tgt.c_str());
    DSK_DBG("draft descriptor: %s\n", dft.c_str());
    refresh_status();
}

// [TAG_DISK_NAMESPACE] every other namespace: ns-<key>/ directories and the flat legacy top level (grouped by the
//   fingerprint in its .meta files); TTL (newest file older than ttl_days) and cap (all together <= quota / 2, oldest
//   first) remove whole namespaces; the blocks they reference are kept by blob_gc
bool server_disk_cache::scan_foreign() {
    const std::string own = fingerprint.substr(0, 16);
    std::map<std::string, foreign_ns> found;
    bool complete = true;
    const auto add_maps = [&complete](foreign_ns & ns, const fs::path & base, const std::string & id) {
        std::vector<std::string> keys;
        for (const char * ext : { ".tgt", ".dft" }) {
            keys.clear();
            sealed_status st = SEALED_BAD;
            if (read_map_keys((base / (id + ext)).string(), keys, &st)) {
                ns.blocks.insert(keys.begin(), keys.end());
            } else if (st == SEALED_IO) {
                ns.incomplete = true; // its blocks are not known
                complete = false;
            }
        }
    };
    // [TAG_DISK_READ_STATUS] (Hermes gate Major 1) a failed listing keeps the previous view and removes nothing;
    //   a namespace whose own listing failed is incomplete: never removed, and its blocks are not known
    std::vector<fs::directory_entry> ls;
    std::string err;
    if (!list_dir(root, ls, false, &err)) {
        DSK_WRN("cannot list %s: foreign namespaces not rescanned, nothing removed\n", err.c_str());
        return false;
    }
    for (const auto & de : ls) {
        std::error_code ec_t;
        const std::string name = de.path().filename().string();
        if (de.is_directory(ec_t) && name.rfind("ns-", 0) == 0 && name.size() == 19 && name.substr(3) != own) {
            foreign_ns & ns = found["ns:" + name.substr(3)];
            ns.key  = name.substr(3);
            ns.path = de.path().string();
            ns.t_newest = mtime_s(de.path()); // a namespace just created by another process is not idle
            std::error_code ec_r;
            fs::recursive_directory_iterator it(de.path(), ec_r);
            for (const fs::recursive_directory_iterator end; !ec_r && it != end; it.increment(ec_r)) {
                const auto & f = *it;
                std::error_code ec_f;
                if (!f.is_regular_file(ec_f)) {
                    continue;
                }
                const std::string fn = f.path().filename().string();
                ns.bytes   += file_size_or_0(f.path().string());
                ns.t_newest = std::max(ns.t_newest, mtime_s(f.path()));
                if (f.path().parent_path() == de.path() && fn.size() == 21 && fn.compare(16, 5, ".meta") == 0) {
                    ns.n_entries++;
                    add_maps(ns, de.path(), fn.substr(0, 16));
                }
            }
            if (ec_r) {
                DSK_WRN("cannot list %s: %s (namespace kept)\n", ns.path.c_str(), ec_r.message().c_str());
                ns.incomplete = true;
                complete = false;
            }
        } else if (de.is_regular_file(ec_t) && name.size() == 21 && name.compare(16, 5, ".meta") == 0) {
            const std::string fp = meta_fingerprint(de.path().string());
            const std::string k  = fp.size() >= 16 ? fp.substr(0, 16) : std::string("unknown");
            foreign_ns & ns = found["legacy:" + k];
            ns.key    = k;
            ns.path   = root;
            ns.legacy = true;
            ns.n_entries++;
            const std::string id = name.substr(0, 16);
            ns.ids.push_back(id);
            for (const char * ext : ENTRY_EXTS) {
                const fs::path p = fs::path(root) / (id + ext);
                ns.bytes   += file_size_or_0(p.string());
                ns.t_newest = std::max(ns.t_newest, mtime_s(p));
            }
            add_maps(ns, fs::path(root), id);
        }
    }
    if (!list_dir(fs::path(root) / "ckpt", ls, true, &err)) {
        DSK_WRN("cannot list %s: legacy namespaces kept\n", err.c_str());
        for (auto & it : found) {
            it.second.incomplete = it.second.incomplete || it.second.legacy;
        }
        ls.clear();
    }
    for (const auto & de : ls) {
        const std::string k = ckpt_fingerprint(de.path());
        auto it = found.find("legacy:" + k);
        if (it != found.end()) {
            it->second.bytes   += file_size_or_0(de.path().string());
            it->second.t_newest = std::max(it->second.t_newest, mtime_s(de.path()));
        }
    }

    foreign.clear();
    for (auto & it : found) {
        foreign.push_back(std::move(it.second));
    }
    std::sort(foreign.begin(), foreign.end(), [](const foreign_ns & a, const foreign_ns & b) { return a.t_newest < b.t_newest; });

    const int64_t t_now = now_s();
    uint64_t total = 0;
    for (const auto & ns : foreign) {
        total += ns.bytes;
    }
    for (size_t i = 0; i < foreign.size();) {
        const foreign_ns & ns = foreign[i];
        const bool expired = ttl_days > 0 && t_now - ns.t_newest > (int64_t) ttl_days * 86400;
        const bool over    = total > quota / 2; // oldest first (sorted)
        if ((expired || over) && !foreign_protected(ns) && !ns.incomplete) {
            DSK_INF("removing foreign namespace %s%s (%zu entries, %.2f GiB, idle %.1f days): %s\n", ns.key.c_str(),
                    ns.legacy ? " (legacy)" : "", ns.n_entries, ns.bytes / (1024.0 * 1024.0 * 1024.0),
                    (t_now - ns.t_newest) / 86400.0, expired ? "ttl" : "cap");
            total -= ns.bytes;
            remove_foreign(ns);
            foreign.erase(foreign.begin() + i);
        } else {
            i++;
        }
    }
    return complete;
}

bool server_disk_cache::foreign_protected(const foreign_ns & ns) const {
    // [TAG_DISK_ADOPT] an eligible adopt source and its checkpoints survive TTL / cap
    return std::any_of(adopted.begin(), adopted.end(), [&](const adopt_src & a) { return a.key == ns.key; });
}

// the structural part of a fingerprint: model / draft / mmproj paths + size@mtime of every file, n_ctx, KV types
//   (the fields after the format version, before the binary / environment / argv)
static std::string fingerprint_structure(const std::string & fp) {
    const auto sp = fp.find(' ');
    const std::vector<std::string> f = string_split<std::string>(sp == std::string::npos ? fp : fp.substr(sp + 1), '|');
    std::string res;
    for (size_t i = 1; i < f.size() && i <= 11; i++) {
        res += f[i] + "|";
    }
    return f.size() > 11 ? res : std::string();
}

void server_disk_cache::open_adopt_sources() {
    adopted.clear();
    src_dir_of.clear();
    const std::string own_struct = fingerprint_structure(fingerprint);
    for (const auto & key : adopt_keys) {
        if (key.empty() || key == fingerprint.substr(0, 16)) {
            continue;
        }
        adopt_src a;
        a.key = key;
        std::error_code ec;
        fs::path base = fs::path(root) / ("ns-" + key);
        fs::path ckdir = base / "ckpt";
        std::vector<std::string> ids;
        if (!fs::is_directory(base, ec)) {
            base   = root; // the legacy (flat) namespace of a format <= 6 build
            ckdir  = fs::path(root) / "ckpt";
            a.legacy = true;
        }
        // [TAG_DISK_READ_STATUS] (Hermes gate Major 1) both listings first: a source that cannot be listed now is
        //   not opened this run (its files are never written either way)
        std::vector<fs::directory_entry> ls_base, ls_ck;
        std::string err;
        if (!list_dir(base, ls_base, false, &err) || !list_dir(ckdir, ls_ck, true, &err)) {
            DSK_WRN("adopt %s: cannot list %s, not opened this run\n", key.c_str(), err.c_str());
            adopted.push_back(a); // still protected from the foreign TTL / cap
            continue;
        }
        for (const auto & de : ls_base) {
            std::error_code ec_t;
            const std::string name = de.path().filename().string();
            if (de.is_regular_file(ec_t) && name.size() == 21 && name.compare(16, 5, ".meta") == 0) {
                const std::string fp = meta_fingerprint(de.path().string());
                if (fp.rfind(key, 0) == 0) {
                    a.fingerprint = fp;
                    ids.push_back(name.substr(0, 16));
                }
            }
        }
        if (ids.empty()) {
            DSK_WRN("adopt %s: no entries of that key in %s\n", key.c_str(), root.c_str());
            continue;
        }
        if (fingerprint_structure(a.fingerprint).empty() || fingerprint_structure(a.fingerprint) != own_struct) {
            DSK_WRN("adopt %s refused: its structural part (model / draft / mmproj files, n_ctx, KV types) differs: %s\n",
                    key.c_str(), fingerprint_structure(a.fingerprint).c_str());
            continue;
        }
        for (const auto & id : ids) {
            if (src_dir_of.count(id) || std::any_of(entries.begin(), entries.end(), [&](const auto & o) { return o.id == id; })) {
                continue;
            }
            src_dir_of[id] = base.string();
            server_disk_cache_entry e;
            const bool ok = read_entry(id, a.fingerprint, e) == SEALED_OK &&
                std::all_of(e.blobs.begin(), e.blobs.end(), [&](const std::string & k) { return blobs.count(k) > 0; });
            if (ok) {
                e.src = key;
                entries.push_back(std::move(e));
                a.n_entries++;
            } else {
                src_dir_of.erase(id);
            }
        }
        size_t n_ck = 0;
        for (const auto & de : ls_ck) {
            std::ifstream f(de.path(), std::ios::binary);
            uint32_t magic = 0, env_ver = 0;
            std::string fp, dh;
            server_deep_ckpt_meta m;
            bool ok = get(f, magic) && (magic == CKPT_MAGIC || magic == CKPT_ENV_MAGIC) && get_str(f, fp) && fp == key;
            if (ok && magic == CKPT_ENV_MAGIC) {
                ok = get(f, env_ver) && get_str(f, dh);
            }
            ok = ok && get_str(f, m.key) && get(f, m.n_tokens) && get(f, m.pos_min) && get(f, m.pos_max) &&
                 m.key + ".ck" == de.path().filename().string() && m.n_tokens > 0 && !ckpts.count(m.key);
            if (ok) {
                m.bytes   = file_size_or_0(de.path().string());
                m.t_used  = mtime_s(de.path());
                m.src_dir = ckdir.string();
                ckpts[m.key] = m;
                n_ck++;
            }
        }
        DSK_INF("adopt source %s%s: %zu entries + %zu deep checkpoints (read-only, fingerprint %s)\n", key.c_str(),
                a.legacy ? " (legacy)" : "", a.n_entries, n_ck, a.fingerprint.substr(0, 16).c_str());
        adopted.push_back(a);
    }
}

void server_disk_cache::remove_foreign(const foreign_ns & ns) {
    std::error_code ec;
    if (!ns.legacy) {
        fs::remove_all(ns.path, ec);
        return;
    }
    for (const auto & id : ns.ids) {
        for (const char * ext : ENTRY_EXTS) {
            fs::remove(fs::path(root) / (id + ext), ec);
        }
    }
    std::vector<fs::directory_entry> ls;
    list_dir(fs::path(root) / "ckpt", ls, true); // a failed listing leaves the checkpoints (TTL / cap retries)
    for (const auto & de : ls) {
        if (ckpt_fingerprint(de.path()) == ns.key) {
            std::error_code ec_r;
            fs::remove(de.path(), ec_r);
        }
    }
}
