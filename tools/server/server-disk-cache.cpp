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
#include <type_traits>
#include <unistd.h> // environ
#include <sys/stat.h>
#include <sys/file.h> // flock
#include <fcntl.h>

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
// a block key ends in "-<bytes>" (the startup check drops a file whose size does not match): its size, 0 if malformed
static uint64_t blob_key_bytes(const std::string & key) {
    const auto dash = key.rfind('-');
    if (dash == std::string::npos || dash + 1 >= key.size()) {
        return 0;
    }
    uint64_t n = 0;
    for (size_t i = dash + 1; i < key.size(); i++) {
        if (key[i] < '0' || key[i] > '9') {
            return 0;
        }
        n = n * 10 + (uint64_t) (key[i] - '0');
    }
    return n;
}

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
// [TAG_STATE_DESCRIPTOR] (Codex gate r2 Minor 2 / D1) the envelope (magic, version, descriptor hash, stream length)
//   and the .ck header are written field by field in fixed little-endian order, whatever the host; on a
//   little-endian host the bytes equal the earlier native writes (format unchanged)
template <typename T> static void put_le(std::ostream & f, T v) {
    static_assert(std::is_unsigned<T>::value, "unsigned fields only");
    uint8_t b[sizeof(T)];
    for (size_t i = 0; i < sizeof(T); i++) {
        b[i] = (uint8_t) (v >> (8 * i));
    }
    f.write(reinterpret_cast<const char *>(b), sizeof(T));
}
template <typename T> static bool get_le(std::istream & f, T & v) {
    static_assert(std::is_unsigned<T>::value, "unsigned fields only");
    uint8_t b[sizeof(T)];
    if (!f.read(reinterpret_cast<char *>(b), sizeof(T))) {
        return false;
    }
    v = 0;
    for (size_t i = 0; i < sizeof(T); i++) {
        v |= (T) b[i] << (8 * i);
    }
    return true;
}
static void put_str_le(std::ostream & f, const std::string & v) {
    put_le<uint32_t>(f, (uint32_t) v.size());
    f.write(v.data(), v.size());
}
static bool get_str_le(std::istream & f, std::string & v) {
    uint32_t n = 0;
    if (!get_le(f, n) || n > 4096) {
        return false;
    }
    v.resize(n);
    return n == 0 || (bool) f.read(v.data(), n);
}

// (Codex gate r2 Minor 2) after a container's terminator / trailer: SEALED_OK at end of file, SEALED_BAD when bytes
//   follow, SEALED_IO when the stream failed (a read error is not damage)
static sealed_status at_eof(std::istream & f) {
    if (f.bad()) {
        return SEALED_IO;
    }
    const auto c = f.peek();
    if (f.bad()) {
        return SEALED_IO;
    }
    return c == std::char_traits<char>::eof() ? SEALED_OK : SEALED_BAD;
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

// [TAG_DISK_READ_STATUS] (Codex gate r3 Major 3) the rest of an open stream through checked reads: SEALED_IO on a read
//   error (EIO, allocation) - libstdc++'s filebuf throws ios_base::failure from underflow(), which istream::read turns
//   into badbit (an istreambuf_iterator would let it escape and be classified as a format error)
static sealed_status read_text(std::istream & f, std::string & out) {
    out.clear();
    try {
        std::vector<char> buf(1 << 16);
        while (true) {
            f.read(buf.data(), (std::streamsize) buf.size());
            const std::streamsize n = f.gcount();
            if (n > 0) {
                out.append(buf.data(), (size_t) n);
            }
            if (!f) {
                break;
            }
        }
        return f.bad() || !f.eof() ? SEALED_IO : SEALED_OK;
    } catch (const std::ios_base::failure &) {
        return SEALED_IO;
    } catch (const std::bad_alloc &) {
        return SEALED_IO;
    }
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
        ok = get(f, n_tokens) && get(f, pos_min) && get(f, pos_max) && get(f, ft) && get(f, fd) && get_str(f, key) &&
             // (Codex gate r2 Blocker 1) a stored checkpoint is host data: it must never select the on-device path
             !((ft | fd) & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
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
    // (Codex gate r2 Minor 2) nothing may follow the trailer (the body is in memory: no read error here)
    return ok && at_eof(f) == SEALED_OK ? SEALED_OK : SEALED_BAD;
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
            std::string bytes;
            if (!f.is_open() || read_text(f, bytes) != SEALED_OK) {
                ok = false; // (Codex gate r3 Major 3) checked reads: a read error is no exception here
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
    // (Codex gate r3 Major 4) "v<N>L": the paths are length-delimited ("<len>:<path>"), so the structural fields
    //   parse unambiguously even with a '|' in a path (server_disk_cache_fingerprint_structure)
    const auto lp = [](const std::string & p) { return std::to_string(p.size()) + ":" + p; };
    const std::string s = string_format("v%uL|%s|%s|%s|%s|ctx%d|k%d|v%d|dk%d|dv%d|%s|%s|%s|%s",
        DISK_CACHE_VERSION,
        lp(params.model.path).c_str(), model_id(params.model.path).c_str(),
        lp(params.speculative.draft.mparams.path).c_str(), model_id(params.speculative.draft.mparams.path).c_str(),
        n_ctx, (int) params.cache_type_k, (int) params.cache_type_v,
        (int) params.speculative.draft.cache_type_k, (int) params.speculative.draft.cache_type_v,
        lp(params.mmproj.path).c_str(), file_id(params.mmproj.path).c_str(), exe_id.c_str(), env_id.c_str());
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
// io_error (Codex gate r2 Major 2): the file exists but could not be read - its identity is unknown, not damaged
static std::string meta_fingerprint(const std::string & path, bool * io_error = nullptr) {
    try {
        std::ifstream fm(path);
        if (!fm) {
            std::error_code ec;
            if (io_error && (fs::exists(path, ec) || ec)) {
                *io_error = true;
            }
            return "";
        }
        std::string text;
        if (read_text(fm, text) != SEALED_OK) { // (Codex gate r3 Major 3) a read error is I/O, not damage
            if (io_error) {
                *io_error = true;
            }
            return "";
        }
        return json::parse(text).value("fingerprint", std::string());
    } catch (const std::ios_base::failure &) {
        if (io_error) {
            *io_error = true;
        }
        return "";
    } catch (const std::bad_alloc &) {
        if (io_error) {
            *io_error = true;
        }
        return "";
    } catch (const std::exception &) {
        return "";
    }
}

// the fingerprint hash16 in a .ck header (DCK3 and DCK4 alike), empty if unreadable
static std::string ckpt_fingerprint(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    uint32_t magic = 0;
    std::string fp;
    if (get_le(f, magic) && (magic == CKPT_MAGIC || magic == CKPT_ENV_MAGIC) && get_str_le(f, fp)) {
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
    // (Codex gate r4 Major) every physical file of the own store is charged, referenced or not (a copy of an adopted /
    //   legacy block kept after its entry went is own storage until blob_gc() reclaims it); a legacy-store original
    //   an own entry references is charged here too (it is kept for that entry); an adopt source's files are its
    //   namespace's bytes (foreign)
    uint64_t res = 0;
    std::set<std::string> refs;
    for (const auto & e : entries) {
        if (!e.src.empty()) {
            continue;
        }
        res += e.own_bytes; // .tok, .aux and the maps (which hold the small fields inline)
        refs.insert(e.blobs.begin(), e.blobs.end());
    }
    for (const auto & k : blk_own) {
        // [TAG_DISK_READ_STATUS] held for the entries skipped at start: an unreferenced block is not collected this
        //   run, so it must not push the quota into evicting entries that can be read (as the checkpoints below)
        if (!gc_hold || refs.count(k)) {
            res += blob_key_bytes(k);
        }
    }
    for (const auto & k : refs) {
        res += blk_legacy.count(k) ? blob_key_bytes(k) : 0;
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
        // (Codex gate r2 Major 5) an adopt source's checkpoint is foreign storage: it never presses own eviction
        if (!it.second.src_dir.empty()) {
            continue;
        }
        res += !gc_hold || ck_refs.count(it.first) ? it.second.bytes : 0;
    }
    return res;
}

// (Codex gate r2 Major 5) the storage split reported by GET /disk-cache: own (= total_bytes, the own quota's
//   charge), foreign (foreign / legacy namespace files incl. adopted checkpoints + the blocks only their maps
//   reference: the foreign cap's charge, quota / 2), shared (legacy-store blocks an own entry references:
//   inside own), held (gc_hold: checkpoints / blocks nothing known references, kept this run, in neither budget)
json server_disk_cache::storage_split() const {
    std::set<std::string> own_refs, foreign_refs;
    for (const auto & e : entries) {
        if (e.src.empty()) {
            own_refs.insert(e.blobs.begin(), e.blobs.end());
        }
    }
    for (const auto & ns : foreign) {
        foreign_refs.insert(ns.blocks.begin(), ns.blocks.end());
    }
    // (Codex gate r4 Major) per physical file, each once: the own store's files are own (held: unreferenced under
    //   gc_hold); a legacy-store file is own when an own entry references it ("shared": inside own), foreign when only
    //   foreign maps do ("foreign_blocks"), held when nothing known does under gc_hold; a namespace's own <ns>/blk is
    //   part of that namespace's bytes (Codex gate r3 Minor 1)
    uint64_t shared = 0, foreign_blk = 0, held = 0, own_blk = 0, legacy_blk = 0;
    for (const auto & k : blk_own) {
        own_blk += blob_key_bytes(k);
        held    += gc_hold && !own_refs.count(k) ? blob_key_bytes(k) : 0;
    }
    for (const auto & k : blk_legacy) {
        const bool o = own_refs.count(k) > 0, f = foreign_refs.count(k) > 0;
        const uint64_t n = blob_key_bytes(k);
        legacy_blk  += n;
        shared      += o ? n : 0;
        foreign_blk += !o && f ? n : 0;
        held        += gc_hold && !o && !f ? n : 0;
    }
    if (gc_hold) {
        std::set<std::string> ck_refs;
        for (const auto & e : entries) {
            ck_refs.insert(e.ckpt_keys.begin(), e.ckpt_keys.end());
            for (const auto & r : e.deep) {
                ck_refs.insert(r.key);
            }
        }
        for (const auto & it : ckpts) {
            held += it.second.src_dir.empty() && !ck_refs.count(it.first) ? it.second.bytes : 0;
        }
    }
    uint64_t foreign_files = 0;
    for (const auto & ns : foreign) {
        foreign_files += ns.bytes;
    }
    return {
        { "own",            total_bytes()               },
        { "foreign",        foreign_files + foreign_blk },
        { "foreign_blocks", foreign_blk                 },
        { "shared",         shared                      },
        { "held",           held                        },
        { "own_blocks",     own_blk                     }, // physical: <ns>/blk
        { "legacy_blocks",  legacy_blk                  }, // physical: <root>/blk
    };
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
        std::string text;
        if (const auto st = read_text(fm, text); st != SEALED_OK) { // (Codex gate r3 Major 3) checked reads
            return st;
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
    } catch (const std::ios_base::failure & err) {
        // (Codex gate r3 Major 3) a stream-buffer read error: inconclusive, the files stay (JSON / format errors below)
        DSK_WRN("entry %s not readable now (I/O): %s\n", id.c_str(), err.what());
        return SEALED_IO;
    } catch (const std::exception & err) {
        DSK_WRN("entry %s unreadable: %s\n", id.c_str(), err.what());
        return SEALED_BAD;
    }
}

void server_disk_cache::init() {
    std::error_code ec;
    // [TAG_DISK_LOCK] (Codex gate r2 Major 4) one writer per cache root: the scan, the collection and the block
    //   publication assume no other process changes the tree. A second server on the same root runs without the
    //   disk cache (shared writers are not supported); the lock goes with the process (flock, released at exit)
    if (lock_fd == -2) {
        return; // refused at the first init: disabled for the whole run (a re-init after a sleep does not retry)
    }
    if (lock_fd < 0) {
        fs::create_directories(root, ec);
        const std::string lp = (fs::path(root) / ".lock").string();
        const int fd = ec ? -1 : open(lp.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB) != 0) {
            const int err = errno;
            DSK_WRN("%s: %s - disk cache disabled for this run (one server per cache directory)\n", lp.c_str(),
                    fd < 0 ? (ec ? ec.message().c_str() : strerror(err)) :
                    err == EWOULDBLOCK ? "held by another process" : strerror(err));
            if (fd >= 0) {
                close(fd);
            }
            quota   = 0;
            lock_fd = -2;
            refresh_status();
            return;
        }
        lock_fd = fd;
    }
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
    std::vector<fs::directory_entry> ls_own, ls_blk, ls_oblk, ls_ck;
    {
        std::string err;
        std::error_code ec_b;
        // [TAG_DISK_BLK_NS] (Codex gate r2 Major 6) new blocks go to the own namespace's store <ns>/blk; the shared
        //   <root>/blk is the legacy store (format <= 6 entries): indexed and referenced, never written
        fs::create_directories(fs::path(dir) / "blk", ec_b);
        if (ec_b) {
            err = (fs::path(dir) / "blk").string() + ": " + ec_b.message();
        }
        std::error_code ec_l;
        const bool has_legacy_blk = fs::is_directory(fs::path(root) / "blk", ec_l);
        if (ec_l == std::errc::no_such_file_or_directory) {
            ec_l.clear(); // no legacy store
        } else if (ec_l) {
            err = (fs::path(root) / "blk").string() + ": " + ec_l.message();
        }
        if (ec_b || ec_l || !list_dir(dir, ls_own, false, &err) ||
                (has_legacy_blk && !list_dir(fs::path(root) / "blk", ls_blk, false, &err)) ||
                !list_dir(fs::path(dir) / "blk", ls_oblk, false, &err) ||
                !list_dir(fs::path(dir) / "ckpt", ls_ck, false, &err)) {
            DSK_WRN("cannot list %s: disk cache disabled for this run, no file touched\n", err.c_str());
            quota = 0;
            entries.clear();
            ckpts.clear();
            blobs.clear();
            blk_own.clear();
            blk_legacy.clear();
            blk_src.clear();
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
    blk_own.clear();
    blk_legacy.clear();
    blk_src.clear();
    // (Codex gate r4 Major) both stores are inventoried: a block present in both is served from the own store, and
    //   its legacy original stays counted (and collectable) as a file of the legacy store
    for (const auto * ls : { &ls_blk, &ls_oblk }) {
        const bool legacy = ls == &ls_blk;
        for (const auto & de : *ls) {
            const std::string name = de.path().filename().string();
            if (name.size() > 4 && name.compare(name.size() - 4, 4, ".blk") == 0) {
                std::error_code ec_s;
                const uint64_t n = fs::file_size(de.path(), ec_s);
                const std::string key = name.substr(0, name.size() - 4);
                if (ec_s) {
                    io_keys.insert(key);
                    continue;
                }
                const auto dash = key.rfind('-');
                if (dash == std::string::npos || std::to_string(n) != key.substr(dash + 1)) {
                    fs::remove(de.path(), ec); // a block whose size does not match its key is a torn write
                    continue;
                }
                (legacy ? blk_legacy : blk_own).insert(key);
                if (!blobs.count(key)) {
                    blobs[key] = n;
                    blob_bytes += n;
                }
            } else if (!legacy) {
                fs::remove(de.path(), ec); // an own .tmp left by a crash; the legacy store is not written
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
            ok = get_le(f, magic) && magic == CKPT_ENV_MAGIC && get_str_le(f, fp) && fp == fp_short && get_le(f, env_ver) &&
                 get_str_le(f, dh) && dh == desc_h && get_str(f, m.key) &&
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

    // (Codex gate r3 Major 1, belt-and-braces) the adopt sources' block stores are indexed before the own entries'
    //   missing-block check: an own entry published by an earlier build that still references an adopted block is
    //   kept while its source is adopted (since r3 every published entry owns its blocks)
    open_adopt_sources();

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
    if (spec && common_speculative_n_stateful(spec) > 1) {
        // (Codex gate r3 Major 2) the carry container holds one implementation's state: with several stateful ones
        //   a save would persist an incomplete state - refused until a multi-state container exists
        static bool logged = false;
        if (!logged) {
            logged = true;
            DSK_WRN("not saving: %d stateful speculative implementations (one carry per entry is supported), the disk "
                    "cache does not persist this run\n", common_speculative_n_stateful(spec));
        }
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
    e.adopted_from = prompt.adopted_from; // [TAG_DISK_ADOPT] provenance of a resave after an adopted load
    e.has_dft = ctx_dft != nullptr;

    uint64_t n_ckpt_written = 0;
    std::vector<server_deep_ref> deep; // (Codex gate r3 Major 5) the committed deep refs: the .aux and e.deep hold the same set

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
            // [TAG_DISK_ADOPT] (Codex gate r2 Major 7) a deep checkpoint still indexed from an adopt source is copied
            //   into the own namespace before the entry is published, so the branch history survives without the
            //   adopt key. (Codex gate r3 Major 5) a copy that fails inconclusively (a read error, a failed write)
            //   aborts the publication - the slot is not saved now, a later eviction retries; only a source verified
            //   missing / damaged (deep_get dropped it from the index) is left out: an explicitly reported gap
            for (const auto & r : prompt.deep) {
                const auto ck = ckpts.find(r.key);
                if (ck != ckpts.end() && !ck->second.src_dir.empty()) {
                    common_prompt_checkpoint c;
                    bool written = false;
                    bool io      = false;
                    if (!deep_get(r.key, c, &io)) {
                        if (io || ckpts.count(r.key)) {
                            throw std::runtime_error("deep checkpoint " + r.key + ": its adopt source is not readable "
                                                     "now, not publishing the entry (retried at the next eviction)");
                        }
                        DSK_WRN("deep checkpoint %s: its adopt source is missing or damaged, left out of the entry "
                                "(a gap in the checkpoint schedule)\n", r.key.c_str());
                        continue;
                    }
                    if (!ckpt_store(r.key, c, &written)) {
                        throw std::runtime_error("deep checkpoint " + r.key + ": cannot copy it into the own "
                                                 "namespace, not publishing the entry (retried at the next eviction)");
                    }
                    n_ckpt_written += written ? file_size_or_0(ckpt_path(r.key)) : 0;
                    n_deep_copied++;
                }
                deep.push_back(r);
            }
            put<uint32_t>(f, (uint32_t) deep.size());
            for (const auto & r : deep) {
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

    // [TAG_DISK_BLK_NS] (Codex gate r3 Major 1) an own entry depends on no other store: every block it references
    //   that is served from an adopt source's <ns>/blk or the legacy <root>/blk (the unchanged prefix of an adopted
    //   conversation) is copied into the own store before the .meta commits - otherwise a restart without the adopt
    //   key (or with it: the own entries are checked before the sources are indexed) drops the entry as "blocks
    //   gone", and a v47 rollback collecting the legacy block breaks it. One-time per conversation; a failed copy
    //   aborts the publication (the source block stays served as before)
    uint64_t n_blk_copied = 0, n_blk_copied_bytes = 0;
    if (ok) {
        std::set<std::string> seen;
        for (const auto & k : e.blobs) {
            if (!seen.insert(k).second || blk_own.count(k)) {
                continue;
            }
            const std::string src = blob_path(k);
            const std::string fin = (fs::path(dir) / "blk" / (k + ".blk")).string();
            const std::string tmp = fin + ".tmp";
            std::error_code ec;
            fs::copy_file(src, tmp, fs::copy_options::overwrite_existing, ec);
            if (!ec && file_size_or_0(tmp) != blobs[k]) {
                ec = std::make_error_code(std::errc::io_error); // a short copy
            }
            if (!ec) {
                fs::rename(tmp, fin, ec);
            }
            if (ec) {
                std::error_code ec_r;
                fs::remove(tmp, ec_r);
                DSK_WRN("block %s: cannot copy it from %s into the own store (%s), not publishing the entry\n",
                        k.c_str(), src.c_str(), ec.message().c_str());
                ok = false;
                break;
            }
            blk_own.insert(k); // served from the own copy from now on; the original stays inventoried in its store
            n_blk_copied++;
            n_blk_copied_bytes += blobs[k];
        }
        if (n_blk_copied > 0) {
            DSK_INF("entry %s: %" PRIu64 " blocks (%.1f MiB) copied from adopted / legacy stores into the own store\n",
                    e.id.c_str(), n_blk_copied, n_blk_copied_bytes / (1024.0 * 1024.0));
            n_written += n_blk_copied_bytes;
            n_blk_copied_total += n_blk_copied;
        }
    }

    if (ok) {
        e.tokens  = prompt.tokens.clone();
        e.deep    = deep; // (Codex gate r3 Major 5) the refs the .aux holds
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
            e.id.c_str(), n, prompt.checkpoints.size(), e.deep.size(), n_logical / (1024.0 * 1024.0),
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
    if (!get_le(fm, magic) || (magic != MAP_ENV_MAGIC && !(legacy_ok && magic == MAP_MAGIC))) {
        return fm.bad() ? SERVER_DISK_LOAD_OTHER : SERVER_DISK_LOAD_INVALID;
    }
    if (magic == MAP_ENV_MAGIC) {
        uint32_t ver = 0;
        std::string h;
        if (!get_le(fm, ver) || ver != ENVELOPE_VERSION || !get_str_le(fm, h) || h != env_h || !get_le(fm, n_total) || n_total == 0) {
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
                const sealed_status e = at_eof(fm);
                if (e == SEALED_IO) {
                    return SERVER_DISK_LOAD_OTHER;
                }
                return e == SEALED_OK && (n_total == 0 || n_seen == n_total) ? SERVER_DISK_LOAD_OK : SERVER_DISK_LOAD_INVALID;
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
                blob_drop_copy(key);
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

    if (spec && !common_speculative_set_state(spec, id_slot, spec_state)) {
        // (Codex gate r2 Major 1) a half-restored state (streams in, carry not) is worse than a miss: the entry is
        //   invalid (an adopt source's files stay), the caller clears the slot. (Codex gate r3 Major 2) an EMPTY carry
        //   is restorable only for stateless implementations: MTP / eagle3 require theirs
        common_speculative_reset_state(spec, id_slot);
        DSK_WRN("entry %s: speculative carry not restorable, dropping it%s\n", e.id.c_str(),
                e.src.empty() ? "" : " from the adopt sources (its files stay)");
        drop_invalid(idx);
        refresh_status();
        return SERVER_DISK_LOAD_INVALID;
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
    prompt.adopted_from = e.src;
    if (!e.src.empty()) {
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
    // [TAG_DISK_ADOPT] an adopt source's checkpoint: the source's file stays, the own namespace gets its own copy;
    //   (Codex gate r3 Major 5) the source's index record stays until the copy commits - a failed write leaves the
    //   intact source checkpoint indexed
    const std::string fin = (fs::path(dir) / "ckpt" / (key + ".ck")).string();
    const std::string tmp = fin + ".tmp";
    bool ok = false;
    {
        std::ostringstream f;
        put_le<uint32_t>(f, CKPT_ENV_MAGIC);
        put_str_le(f, fingerprint.substr(0, 16));
        put_le<uint32_t>(f, ENVELOPE_VERSION);
        put_str_le(f, desc_h);
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
    ckpts[key] = m; // replaces an adopt source's record (src_dir) only now that the own copy is committed
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
        bool ok = get_le(f, magic) && (magic == CKPT_ENV_MAGIC || (from_src && magic == CKPT_MAGIC)) && get_str_le(f, fp);
        if (ok && magic == CKPT_ENV_MAGIC) {
            ok = get_le(f, env_ver) && get_str_le(f, dh);
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
        if ((ft | fd) & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
            // (Codex gate r2 Blocker 1) a stored checkpoint is host data: it must never select the on-device path
            throw std::runtime_error("on-device state flags");
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

void server_disk_cache::deep_drop(const std::string & key, const char * why) {
    auto it = ckpts.find(key);
    if (it == ckpts.end()) {
        return;
    }
    DSK_WRN("deep checkpoint %s: %s, dropping it\n", key.c_str(), why);
    if (it->second.src_dir.empty()) {
        std::error_code ec;
        fs::remove(ckpt_path(key), ec);
    }
    ckpts.erase(it);
    n_ckpt_dropped++;
    refresh_status();
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

// the own store or the legacy store (whose blocks an unreferenced / damaged state may remove); never an adopt
//   source's store
bool server_disk_cache::blob_writable(const std::string & key) const {
    return blk_own.count(key) || blk_legacy.count(key) || !blk_src.count(key);
}

// the read location: own > legacy > an adopt source; a key in none of them is a new own block
std::string server_disk_cache::blob_path(const std::string & key) const {
    if (!blk_own.count(key)) {
        if (blk_legacy.count(key)) {
            return (fs::path(root) / "blk" / (key + ".blk")).string();
        }
        const auto it = blk_src.find(key);
        if (it != blk_src.end()) {
            return (fs::path(it->second) / (key + ".blk")).string();
        }
    }
    return (fs::path(dir) / "blk" / (key + ".blk")).string();
}

// a damaged read-location copy: removed when writable (own / legacy), forgotten when an adopt source's; the key stays
//   indexed while another store still holds it (the next read verifies that copy)
void server_disk_cache::blob_drop_copy(const std::string & key) {
    std::error_code ec;
    if (blob_writable(key)) {
        fs::remove(blob_path(key), ec);
    }
    if (!blk_own.erase(key) && !blk_legacy.erase(key)) {
        blk_src.erase(key);
    }
    if (!blk_own.count(key) && !blk_legacy.count(key) && !blk_src.count(key)) {
        const auto it = blobs.find(key);
        if (it != blobs.end()) {
            blob_bytes -= it->second;
            blobs.erase(it);
        }
    }
}

bool server_disk_cache::save_stream(llama_context * ctx, int32_t id_slot, const std::string & map_path,
                                    const std::string & env_h, std::vector<std::string> & keys, uint64_t & n_logical,
                                    uint64_t & n_written) {
    const std::string tmp = map_path + ".tmp";
    std::ofstream fm(tmp, std::ios::binary | std::ios::trunc);
    // [TAG_STATE_DESCRIPTOR] envelope: magic, version, descriptor hash, stream length (patched at the end)
    put_le<uint32_t>(fm, MAP_ENV_MAGIC);
    put_le<uint32_t>(fm, ENVELOPE_VERSION);
    put_str_le(fm, env_h);
    const auto len_at = fm.tellp();
    put_le<uint64_t>(fm, 0);
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
                blk_own.insert(key); // written to the own store
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
    put_le<uint64_t>(fm, (uint64_t) n_stream);
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
    if (!get_le(fm, magic) || (magic != MAP_MAGIC && magic != MAP_ENV_MAGIC)) {
        return bad();
    }
    if (magic == MAP_ENV_MAGIC) {
        uint32_t ver = 0;
        std::string h;
        uint64_t n = 0;
        if (!get_le(fm, ver) || !get_str_le(fm, h) || !get_le(fm, n)) {
            return bad();
        }
    }
    for (;;) {
        uint8_t kind = 0;
        if (!get(fm, kind)) {
            return bad();
        }
        if (kind == 255) {
            const sealed_status e = at_eof(fm);
            if (e != SEALED_OK) {
                return fail(e);
            }
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
    if (!get_le(fm, magic) || (magic != MAP_ENV_MAGIC && !(legacy_ok && magic == MAP_MAGIC))) {
        // [TAG_DISK_READ_STATUS] (Codex gate r2 Major 2) a read error is inconclusive, only a read that worked is damage
        if (fm.bad()) {
            DSK_WRN("%s: read error at the state envelope, keeping it\n", map_path.c_str());
            return SERVER_DISK_LOAD_OTHER;
        }
        DSK_WRN("%s: no state envelope\n", map_path.c_str());
        return SERVER_DISK_LOAD_INVALID;
    }
    // [TAG_STATE_DESCRIPTOR] the stream must describe this context's state, checked before the context is touched
    uint64_t n_total = 0;
    if (magic == MAP_ENV_MAGIC) {
        uint32_t ver = 0;
        std::string h;
        if (!get_le(fm, ver) || ver != ENVELOPE_VERSION || !get_str_le(fm, h) || h != env_h || !get_le(fm, n_total) || n_total == 0) {
            if (fm.bad()) {
                DSK_WRN("%s: read error at the state envelope, keeping it\n", map_path.c_str());
                return SERVER_DISK_LOAD_OTHER;
            }
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
            const sealed_status e = at_eof(fm);
            if (e == SEALED_IO) {
                io_error = true;
            } else if (e == SEALED_BAD) {
                DSK_WRN("%s: bytes after the map terminator\n", map_path.c_str());
                corrupt = true;
            }
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
                blob_drop_copy(key);
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
        const bool rest_empty = seg_left == 0 && !next_segment() && done && !corrupt && !io_error;
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
    std::set<std::string> referenced, own_referenced, elsewhere;
    for (const auto & e : entries) {
        referenced.insert(e.blobs.begin(), e.blobs.end());
        if (e.src.empty()) {
            own_referenced.insert(e.blobs.begin(), e.blobs.end());
        }
    }
    // (Codex gate r4 Major) an own-store copy no own entry needs goes when another store still holds the key (an
    //   adopted / legacy original: the read location falls back to it) - foreign references never pin a redundant own
    //   copy; no rescan needed for that
    std::error_code ec;
    for (auto it = blk_own.begin(); it != blk_own.end();) {
        if (!own_referenced.count(*it) && (blk_legacy.count(*it) || blk_src.count(*it))) {
            fs::remove((fs::path(dir) / "blk" / (*it + ".blk")).string(), ec);
            it = blk_own.erase(it);
        } else {
            ++it;
        }
    }
    // [TAG_DISK_NAMESPACE] blocks of every other namespace on disk (incl. other live processes) stay: rescan them
    //   only when something would be deleted; a failed rescan deletes nothing
    std::set<std::string> entry_refs; // the blocks the loaded entries (own + adopted) reference, set when rescanned
    if (std::any_of(blobs.begin(), blobs.end(), [&](const auto & b) { return !referenced.count(b.first); })) {
        if (!scan_foreign()) {
            return;
        }
        entry_refs = referenced;
        for (const auto & ns : foreign) {
            referenced.insert(ns.blocks.begin(), ns.blocks.end());
            elsewhere.insert(ns.stored.begin(), ns.stored.end());
        }
    }
    for (auto it = blobs.begin(); it != blobs.end();) {
        // an own copy only foreign maps reference, held in that namespace's own <ns>/blk too (not adopted this run,
        //   e.g. a restart without the adopt key): redundant, reclaimed; the sole physical copy of a block a foreign
        //   map references stays (charged to own)
        const bool own_only_foreign = referenced.count(it->first) && blk_own.count(it->first) &&
                                      !entry_refs.count(it->first) && elsewhere.count(it->first);
        if (own_only_foreign) {
            blk_own.erase(it->first);
            fs::remove((fs::path(dir) / "blk" / (it->first + ".blk")).string(), ec);
            if (!blk_legacy.count(it->first) && !blk_src.count(it->first)) {
                blob_bytes -= it->second;
                it = blobs.erase(it);
                continue;
            }
        }
        if (!referenced.count(it->first)) {
            // every physical copy: the own one, a legacy one nothing on disk references; an adopt source's is
            //   only forgotten (never written)
            if (blk_own.erase(it->first)) {
                fs::remove((fs::path(dir) / "blk" / (it->first + ".blk")).string(), ec);
            }
            if (blk_legacy.erase(it->first)) {
                fs::remove((fs::path(root) / "blk" / (it->first + ".blk")).string(), ec);
            }
            blk_src.erase(it->first);
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
    blk_own.clear();
    blk_legacy.clear();
    blk_src.clear();
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
            { "incomplete", ns.incomplete }, // never collected this scan (listing / identity unknown)
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
        { "storage",        storage_split()     },
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
        { "ckpt_dropped",   n_ckpt_dropped      },
        { "deep_copied",    n_deep_copied       },
        { "blk_copied",     n_blk_copied_total  },
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
    // the speculative carry: implementation names + each stateful implementation's schema, tag and row width
    //   (common_speculative_state_schema, Codex gate r2 Major 1)
    const std::string srv = string_format("|carry=%s|tokens=%u|mtmd=%d|ckpt=%u",
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
    bool any_unidentified = false;
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
                } else if (f.path().parent_path() == de.path() / "blk" && fn.size() > 4 &&
                           fn.compare(fn.size() - 4, 4, ".blk") == 0) {
                    ns.stored.insert(fn.substr(0, fn.size() - 4));
                }
            }
            if (ec_r) {
                DSK_WRN("cannot list %s: %s (namespace kept)\n", ns.path.c_str(), ec_r.message().c_str());
                ns.incomplete = true;
                complete = false;
            }
        } else if (de.is_regular_file(ec_t) && name.size() == 21 && name.compare(16, 5, ".meta") == 0) {
            bool unreadable = false;
            const std::string fp = meta_fingerprint(de.path().string(), &unreadable);
            const std::string k  = unreadable ? std::string("unidentified") :
                                   fp.size() >= 16 ? fp.substr(0, 16) : std::string("unknown");
            foreign_ns & ns = found["legacy:" + k];
            ns.key    = k;
            ns.path   = root;
            ns.legacy = true;
            if (unreadable) {
                // [TAG_DISK_READ_STATUS] (Codex gate r2 Major 2) a legacy entry whose .meta cannot be read now has an
                //   unknown identity: it may belong to any legacy fingerprint (an adopt source among them), so no legacy
                //   namespace is collected while it stays unidentified (marked below), and it is never collected itself
                ns.incomplete = true;
                any_unidentified = true;
                if (!warned_unidentified) {
                    warned_unidentified = true;
                    DSK_WRN("legacy entry %s: .meta not readable, identity unknown - legacy files are kept while it is "
                            "unidentified (logged once)\n", name.substr(0, 16).c_str());
                }
            }
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
    if (any_unidentified) {
        for (auto & it : found) {
            it.second.incomplete = it.second.incomplete || it.second.legacy;
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
    // (Codex gate r2 Major 5) the foreign cap counts the namespaces' files AND the blocks only foreign / legacy maps
    //   reference (each once); a block an own entry references is own storage
    std::set<std::string> own_refs;
    for (const auto & e : entries) {
        if (e.src.empty()) {
            own_refs.insert(e.blobs.begin(), e.blobs.end());
        }
    }
    std::map<std::string, int> foreign_only; // block -> foreign namespaces referencing it
    uint64_t total = 0;
    for (const auto & ns : foreign) {
        total += ns.bytes;
        for (const auto & b : ns.blocks) {
            // a block in the shared legacy store (one in a namespace's own blk/ is part of its ns.bytes)
            //   (Codex gate r4 Major) by the legacy inventory: an original shadowed by an own copy stays counted
            if (!own_refs.count(b) && blk_legacy.count(b) && foreign_only[b]++ == 0) {
                total += blob_key_bytes(b);
            }
        }
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
            for (const auto & b : ns.blocks) {
                auto fo = foreign_only.find(b);
                if (fo != foreign_only.end() && --fo->second == 0) {
                    total -= std::min<uint64_t>(total, blob_key_bytes(b));
                }
            }
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

// [TAG_DISK_ADOPT] (Codex gate r2 Major 3, r3 Major 4) the structural part of a fingerprint (model / draft / mmproj
//   paths with the size@mtime of every file - each shard of a split GGUF -, n_ctx, KV types), parsed into named fields
//   and re-encoded length-delimited with arrays of shard identities: "name=<len>:<value>;" / "name=[<n>]<len>:<v>...;".
//   Two source formats ("v|model|id[|shard id...]|draft|id[|shard id...]|ctxN|kN|vN|dkN|dvN|mmproj|id|binary|env"):
//   - "v<N>L" (written since r3): every path is length-delimited ("<len>:<path>"), so a '|' in a path is unambiguous;
//   - "v<N>" (legacy, v47): positional. A '|' inside a path shifts every later field, so the parse is strict: every
//     field must match its grammar INCLUDING the binary identity after the mmproj id ("<lib>=<hash>;..." /
//     "<size>:<mtime>" / "unverifiable:<id>"), and the rest (environment knobs, argv) must hold no '|'-delimited file
//     identity or binary identity - otherwise the boundaries cannot be established and "" is returned (adoption
//     refused). The number of shard identities after a path follows from its "-00001-of-NNNNN" suffix.
std::string server_disk_cache_fingerprint_structure(const std::string & fp, std::string * why) {
    const auto fail = [&](const std::string & w) {
        if (why) {
            *why = w;
        }
        return std::string();
    };
    const auto sp = fp.find(' ');
    const std::string t = sp == std::string::npos ? fp : fp.substr(sp + 1);
    size_t pos = 0;
    // the next '|'-delimited token; false at the end of the text
    const auto token = [&](std::string & out) -> bool {
        if (pos > t.size()) {
            return false;
        }
        const auto e = t.find('|', pos);
        out = t.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
        pos = e == std::string::npos ? t.size() + 1 : e + 1;
        return true;
    };
    const auto all_of = [](const std::string & v, size_t a, size_t b, const char * set) {
        return a < b && v.find_first_not_of(set, a) >= b;
    };
    const auto is_file_id = [&](const std::string & v) { // "<size>@<mtime>"
        const auto at = v.find('@');
        if (at == std::string::npos || !all_of(v, 0, at, "0123456789")) {
            return false;
        }
        const size_t m = at + 1 < v.size() && v[at + 1] == '-' ? at + 2 : at + 1;
        return all_of(v, m, v.size(), "0123456789");
    };
    const auto is_exe_id = [&](const std::string & v) { // "<lib>=<hash>;..." | "<size>:<mtime>" | "unverifiable:<id>"
        static const char * const NAME = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._+-";
        static const char * const HASH = "0123456789abcdefABCDEF-";
        if (v.rfind("unverifiable:", 0) == 0) {
            return all_of(v, 13, v.size(), "0123456789abcdef");
        }
        const auto c = v.find(':');
        if (c != std::string::npos && v.find('=') == std::string::npos) {
            const size_t m = c + 1 < v.size() && v[c + 1] == '-' ? c + 2 : c + 1;
            return all_of(v, 0, c, "0123456789") && all_of(v, m, v.size(), "0123456789");
        }
        size_t a = 0;
        while (a < v.size()) {
            const auto eq = v.find('=', a);
            const auto sc = v.find(';', a);
            if (eq == std::string::npos || sc == std::string::npos || sc < eq || !all_of(v, a, eq, NAME) ||
                !all_of(v, eq + 1, sc, HASH)) {
                return false;
            }
            a = sc + 1;
        }
        return !v.empty();
    };
    const auto field = [](const std::string & name, const std::string & v) {
        return name + "=" + std::to_string(v.size()) + ":" + v + ";";
    };

    std::string ver;
    if (!token(ver) || ver.size() < 2 || ver[0] != 'v') {
        return fail("no version field");
    }
    const bool ld = ver.back() == 'L';
    if (!all_of(ver, 1, ver.size() - (ld ? 1 : 0), "0123456789")) {
        return fail("version field does not parse");
    }
    // a path: length-delimited "<len>:<path>|" (ld) or the next token (legacy)
    const auto read_path = [&](std::string & out) -> bool {
        if (!ld) {
            return token(out);
        }
        const auto c = t.find(':', pos);
        if (c == std::string::npos || c - pos > 9 || !all_of(t, pos, c, "0123456789")) {
            return false;
        }
        const size_t n = std::stoul(t.substr(pos, c - pos));
        if (c + 1 + n >= t.size() || t[c + 1 + n] != '|') {
            return false;
        }
        out = t.substr(c + 1, n);
        pos = c + 2 + n;
        return true;
    };
    std::string res;
    // a path and the identities of its files: 1 + (N - 1) shards for "<name>-00001-of-0000N.gguf"
    const auto path_files = [&](const char * name) -> bool {
        std::string p;
        if (!read_path(p)) {
            return false;
        }
        size_t n = 1;
        const auto at = p.rfind("-00001-of-");
        if (at != std::string::npos && p.size() >= at + 15) {
            const int ns = atoi(p.substr(at + 10, 5).c_str());
            n = (size_t) std::max(1, std::min(ns, 99999));
        }
        std::string arr = std::string(name) + "_files=[" + std::to_string(n) + "]";
        for (size_t k = 0; k < n; k++) {
            std::string id;
            if (!token(id) || !is_file_id(id)) {
                return false;
            }
            arr += std::to_string(id.size()) + ":" + id;
        }
        res += field(name, p) + arr + ";";
        return true;
    };
    if (!path_files("model")) {
        return fail("model path / shard identities do not parse (a '|' in the path or a shard count mismatch)");
    }
    if (!path_files("draft")) {
        return fail("draft path / shard identities do not parse (a '|' in the path or a shard count mismatch)");
    }
    for (const char * name : { "ctx", "k", "v", "dk", "dv" }) {
        const std::string pre = name;
        std::string v;
        if (!token(v) || v.rfind(pre, 0) != 0 || v.size() == pre.size() ||
            v.find_first_not_of("-0123456789", pre.size()) != std::string::npos) {
            return fail("field " + pre + " does not parse");
        }
        res += field(name, v.substr(pre.size()));
    }
    std::string mp, mid;
    if (!read_path(mp) || !token(mid) || !is_file_id(mid)) {
        return fail("mmproj path / identity does not parse");
    }
    // (Codex gate r3 Major 4) the binary identity closes the structural part: it anchors the field boundaries
    std::string exe;
    if (!token(exe) || !is_exe_id(exe)) {
        return fail("the binary identity after the mmproj does not parse (a '|' in a path shifts the fields)");
    }
    if (!ld) {
        // legacy: what follows (server-src, environment knobs, argv) must not look like a shifted structural field
        std::string rest;
        while (token(rest)) {
            if (is_file_id(rest) || (rest.find('=') != std::string::npos && rest.rfind("server-src=", 0) != 0 &&
                                     rest.find("argv:") == std::string::npos && is_exe_id(rest))) {
                return fail("ambiguous field boundaries (a '|' inside a path): '" + rest + "' after the binary identity");
            }
        }
    }
    res += field("mmproj", mp) + field("mmproj_file", mid);
    return res;
}

void server_disk_cache::open_adopt_sources() {
    adopted.clear();
    src_dir_of.clear();
    std::string own_why;
    const std::string own_struct = server_disk_cache_fingerprint_structure(fingerprint, &own_why);
    if (own_struct.empty() && !adopt_keys.empty()) {
        DSK_WRN("adopt: this server's own fingerprint does not parse (%s), no source is adopted\n", own_why.c_str());
    }
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
        // [TAG_DISK_BLK_NS] a namespaced source keeps its blocks in <ns>/blk: indexed read-only (own / legacy first)
        if (!a.legacy) {
            std::vector<fs::directory_entry> ls_sb;
            std::error_code ec_d;
            if (fs::is_directory(base / "blk", ec_d) && list_dir(base / "blk", ls_sb, false, &err)) {
                for (const auto & de : ls_sb) {
                    const std::string name = de.path().filename().string();
                    std::error_code ec_s;
                    const uint64_t n = fs::file_size(de.path(), ec_s);
                    if (ec_s || name.size() <= 4 || name.compare(name.size() - 4, 4, ".blk") != 0) {
                        continue;
                    }
                    const std::string k = name.substr(0, name.size() - 4);
                    // (Codex gate r4 Major) inventoried even behind an own / legacy copy: a reclaimed own copy
                    //   falls back to it
                    blk_src.emplace(k, (base / "blk").string());
                    if (!blobs.count(k)) {
                        blobs[k] = n;
                        blob_bytes += n;
                    }
                }
            }
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
        std::string why;
        const std::string src_struct = server_disk_cache_fingerprint_structure(a.fingerprint, &why);
        if (src_struct.empty() || own_struct.empty()) {
            DSK_WRN("adopt %s refused: ambiguous structural identity (%s)\n", key.c_str(),
                    src_struct.empty() ? why.c_str() : "own fingerprint");
            continue;
        }
        if (src_struct != own_struct) {
            DSK_WRN("adopt %s refused: its structural part (model / draft / mmproj files, n_ctx, KV types) differs: %s\n",
                    key.c_str(), src_struct.c_str());
            continue;
        }
        DSK_INF("adopt %s: structure %s\n", key.c_str(), src_struct.c_str());
        for (const auto & id : ids) {
            if (src_dir_of.count(id) || std::any_of(entries.begin(), entries.end(), [&](const auto & o) { return o.id == id; })) {
                continue;
            }
            src_dir_of[id] = base.string();
            server_disk_cache_entry e;
            const sealed_status st = read_entry(id, a.fingerprint, e);
            const bool complete = st == SEALED_OK &&
                std::all_of(e.blobs.begin(), e.blobs.end(), [&](const std::string & k) { return blobs.count(k) > 0; });
            if (complete) {
                e.src = key;
                entries.push_back(std::move(e));
                a.n_entries++;
            } else {
                // (Codex gate r2 Minor 1) an entry-specific warning and the dropped counter; its files stay either way
                DSK_WRN("adopt %s: entry %s not adopted (%s)\n", key.c_str(), id.c_str(),
                        st == SEALED_IO ? "temporarily unreadable" : st == SEALED_OK ? "incomplete: a block is missing" :
                        "damaged");
                a.dropped++;
                src_dir_of.erase(id);
            }
        }
        size_t n_ck = 0;
        for (const auto & de : ls_ck) {
            std::ifstream f(de.path(), std::ios::binary);
            uint32_t magic = 0, env_ver = 0;
            std::string fp, dh;
            server_deep_ckpt_meta m;
            bool ok = get_le(f, magic) && (magic == CKPT_MAGIC || magic == CKPT_ENV_MAGIC) && get_str_le(f, fp) && fp == key;
            if (ok && magic == CKPT_ENV_MAGIC) {
                ok = get_le(f, env_ver) && get_str_le(f, dh);
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
