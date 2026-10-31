#include "server-disk-cache.h"

#include "log.h"

#include "ggml-backend.h"
#include "../../src/llama-ext.h" // llama_state_seq_write_cb / read_cb [TAG_DISK_DELTA]

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <random>
#include <atomic>
#include <set>
#include <thread>
#include <sstream>
#include <type_traits>
#include <unistd.h> // environ
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/file.h> // flock
#include <fcntl.h>

#define DSK_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, "disk_cache", __VA_ARGS__)
#define DSK_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, "disk_cache", __VA_ARGS__)
#define DSK_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, "disk_cache", __VA_ARGS__)

namespace fs = std::filesystem;

static constexpr uint32_t DISK_CACHE_VERSION = 8; // 8: envelope v2 - full-coverage descriptor hashes in .tgt / .dft / .ck envelopes, descriptor_full in .meta // 7: envelope (magic, version, descriptor hash, length) at the head of .tgt / .dft maps and .ck // 6: user-message starts at the end of .aux (optional: a 5 entry loads with them unknown) // 5: sealed (checksummed) .tok / .aux / .ck and map segments // 2: deep checkpoint refs in .aux, 3: block-deduplicated state streams, 4: .aux checkpoints by key
static constexpr uint32_t AUX_MAGIC          = 0x58554144; // "DAUX"
static constexpr uint32_t CKPT_MAGIC         = 0x334b4344; // "DCK3" (format 5 / 6: no envelope)
static constexpr uint32_t CKPT_ENV_MAGIC     = 0x344b4344; // "DCK4": fingerprint, envelope version, descriptor hash
static constexpr uint32_t MAP_MAGIC          = 0x3250414d; // "MAP2" (format 5 / 6: no envelope)
static constexpr uint32_t MAP_ENV_MAGIC      = 0x3350414d; // "MAP3": envelope version, descriptor hash, stream length
static constexpr uint32_t ENVELOPE_VERSION   = 2; // 2: full-coverage descriptor hashes (1: weak, format 7)
// [TAG_STATE_DESCRIPTOR] bump on any change of what a restored checkpoint means (its flags, the draft-prefix retention,
//   the replay rules after a deep rewind) that keeps the bytes the same
static constexpr uint32_t SERVER_CKPT_SEMANTICS_REV = 1;
static constexpr uint32_t SEAL_MAGIC         = 0x4c414553; // "SEAL"
static constexpr size_t   BLOB_BYTES         = 4ull << 20; // block size of the state streams (cut from the start of each tensor range)

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

// [TAG_CELL_EXT_PROVENANCE] a per-seq data blob (u32 magic, i32 seq id, stream) of a PARTIAL_ONLY qwen4exp trunk
//   checkpoint opens its stream with the decoy header {1, 0, LLAMA_STATE_IXY_MARK} when a build with the indexer x/y
//   fix wrote it (llama-ext.h)
static const size_t CKPT_IXY_HEAD = 20;
static bool ckpt_blob_ixy_marked(const uint8_t * d, size_t n) {
    uint32_t w[3] = {};
    if (n < CKPT_IXY_HEAD) {
        return false;
    }
    memcpy(w, d + 8, sizeof(w));
    return w[0] == 1 && w[1] == 0 && w[2] == LLAMA_STATE_IXY_MARK;
}
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

// [TAG_CACHE_COMPAT2] ---------------------------------------------------------------------------------------------
// (Codex compat2 C) an envelope-less stream / checkpoint (MAP2 / DCK3) is read only when it arrives through adopt AND
//   its payload format is one that had no envelope (5 / 6): an adopted entry of a format with envelopes that carries
//   an envelope-less stream is damaged or forged, never a legacy entry
static bool legacy_format(uint32_t version) {
    return version == 5 || version == 6;
}

static bool legacy_payload_ok(const server_disk_cache_entry & e) {
    return !e.src.empty() && legacy_format(e.version);
}

// [TAG_CACHE_COMPAT2] (Codex compat2 D) an envelope is accepted at v2 with the full-coverage hash, or at v1 with the
//   weak hash only where the rule allows it (an adopted format-7 entry)
template <typename R>
static bool envelope_ok(const R & rule, uint32_t ver, const std::string & h) {
    return (ver == 2 && !rule.h2.empty() && h == rule.h2) || (ver == 1 && rule.v1_ok && h == rule.h1);
}

// the payload format a fingerprint was written with: its "v<N>" field (0 when it does not parse)
static uint32_t fingerprint_format(const std::string & fp) {
    const auto sp = fp.find(' ');
    size_t at = sp == std::string::npos ? 0 : sp + 1;
    if (at >= fp.size() || fp[at] != 'v') {
        return 0;
    }
    uint32_t v = 0;
    for (at++; at < fp.size() && isdigit((unsigned char) fp[at]) && v < 100000; at++) {
        v = v * 10 + (fp[at] - '0');
    }
    return v;
}
// hash_key's first 16 hex digits cover only lanes a / c (bytes 0-7 and 16-23 of every 32-byte block): fold both
//   halves so every input byte counts
static std::string h16_of(const std::string & s) {
    const std::string k = hash_key(reinterpret_cast<const uint8_t *>(s.data()), s.size());
    const unsigned long long h1 = std::stoull(k.substr(0, 16), nullptr, 16);
    const unsigned long long h2 = std::stoull(k.substr(16, 16), nullptr, 16);
    return string_format("%016llx", h1 ^ (h2 * 0x9E3779B97F4A7C15ull));
}

// [TAG_CACHE_COMPAT2] the parsed-param digest (Codex compat2 A): the semantic parsed configuration, normalized
//   (field = value, fixed order, by meaning - not by spelling): the same configuration reached through argv or the
//   LLAMA_ARG_* environment gives the same string. The exact-qualified exclusions are listed in the header
static std::string canon_cpu(const common_cpu_params & c) {
    return string_format("thr=%d", c.n_threads);
}

static std::string canon_devices(const std::vector<ggml_backend_dev_t> & devs) {
    std::string r = "[";
    for (auto * d : devs) {
        r += d ? std::string(ggml_backend_dev_name(d)) + ";" : std::string("null;");
    }
    return r + "]";
}

static std::string canon_buft(const std::vector<llama_model_tensor_buft_override> & v) {
    std::string r = "[";
    for (const auto & o : v) {
        if (o.pattern == nullptr) {
            continue; // the terminating sentinel
        }
        r += string_format("%zu:%s=%s;", strlen(o.pattern), o.pattern, o.buft ? ggml_backend_buft_name(o.buft) : "null");
    }
    return r + "]";
}

std::string server_cache_param_canon(const common_params & p) {
    // ctx-checkpoints / checkpoint-min-step set the prefill ubatch boundaries (the 4 + n_ubatch / 4 splits before a
    //   checkpoint), not only which states are kept (astra compat2 r2)
    std::string r = string_format("nb=%d;ub=%d;npar=%d;ctxcp=%d;cms=%d;numa=%d;%s;b%s;dev=%s;ngl=%d;main=%d;split=%d;ts=[",
                                  p.n_batch, p.n_ubatch, p.n_parallel, p.n_ctx_checkpoints, p.checkpoint_min_step,
                                  (int) p.numa, canon_cpu(p.cpuparams).c_str(),
                                  canon_cpu(p.cpuparams_batch).c_str(), canon_devices(p.devices).c_str(), p.n_gpu_layers,
                                  p.main_gpu, (int) p.split_mode);
    for (size_t i = 0; i < llama_max_devices() && i < 128; i++) {
        r += string_format("%a;", p.tensor_split[i]);
    }
    r += string_format("];buft=%s;nokvo=%d;noopo=%d;noxb=%d;nohost=%d;fa=%d;k=%d;v=%d;swa_full=%d;unified=%d;qsa=%d;kvo=[",
                       canon_buft(p.tensor_buft_overrides).c_str(), (int) p.no_kv_offload, (int) p.no_op_offload,
                       (int) p.no_extra_bufts, (int) p.no_host, (int) p.flash_attn_type, (int) p.cache_type_k,
                       (int) p.cache_type_v, (int) p.swa_full, (int) p.kv_unified, p.qsa_ratio);
    for (const auto & o : p.kv_overrides) {
        if (o.key[0] == 0) {
            continue; // the terminating sentinel
        }
        std::string v;
        switch (o.tag) {
            case LLAMA_KV_OVERRIDE_TYPE_INT:   v = std::to_string(o.val_i64); break;
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT: v = string_format("%a", o.val_f64); break;
            case LLAMA_KV_OVERRIDE_TYPE_BOOL:  v = o.val_bool ? "true" : "false"; break;
            case LLAMA_KV_OVERRIDE_TYPE_STR:   v = std::string(o.val_str, strnlen(o.val_str, sizeof(o.val_str))); break;
        }
        const std::string k(o.key, strnlen(o.key, sizeof(o.key)));
        r += string_format("%zu:%s=%d:%zu:%s;", k.size(), k.c_str(), (int) o.tag, v.size(), v.c_str());
    }
    const auto & sp = p.speculative;
    const auto & d  = sp.draft;
    r += "];spec{types=";
    for (auto t : sp.types) {
        r += std::to_string((int) t) + ",";
    }
    r += string_format(";synth=%a;rates=", sp.synth_len);
    for (double x : sp.synth_rates) {
        r += string_format("%a,", x);
    }
    r += string_format(";nmax=%d;nmin=%d;adapt=%d/%a/%a/%a/%a/%d;psplit=%a;pmin=%a;bs=%d;ngl=%d;k=%d;v=%d;%s;b%s;dev=%s;"
                       "buft=%s;ngram_mod=%d/%d/%d;ngram_simple=%u/%u/%u;ngram_k=%u/%u/%u;ngram_k4v=%u/%u/%u}",
                       d.n_max, d.n_min, (int) d.n_max_adaptive, d.n_max_adaptive_t_tok, d.n_max_adaptive_t_dec,
                       d.n_max_adaptive_k_tok, d.n_max_adaptive_k_dec, (int) d.n_max_adaptive_measure, d.p_split,
                       d.p_min, (int) d.backend_sampling, d.n_gpu_layers, (int) d.cache_type_k, (int) d.cache_type_v,
                       canon_cpu(d.cpuparams).c_str(), canon_cpu(d.cpuparams_batch).c_str(),
                       canon_devices(d.devices).c_str(), canon_buft(d.tensor_buft_overrides).c_str(),
                       sp.ngram_mod.n_match, sp.ngram_mod.n_max, sp.ngram_mod.n_min,
                       sp.ngram_simple.size_n, sp.ngram_simple.size_m, sp.ngram_simple.min_hits,
                       sp.ngram_map_k.size_n, sp.ngram_map_k.size_m, sp.ngram_map_k.min_hits,
                       sp.ngram_map_k4v.size_n, sp.ngram_map_k4v.size_m, sp.ngram_map_k4v.min_hits);
    return r;
}

// [TAG_CACHE_COMPAT2] the effective placement of a loaded model (Codex compat2 astra r2): fit-params rewrites the
//   GPU-layer count, tensor split and buffer overrides after parsing, so the requested values do not identify it -
//   the loaded model reports what was applied (llama_model_placement_desc)
std::string server_cache_model_placement(const llama_model * m) {
    if (m == nullptr) {
        return "none";
    }
    const int32_t n = llama_model_placement_desc(m, nullptr, 0);
    if (n < 0) {
        return "unknown";
    }
    std::string s((size_t) n + 1, '\0');
    llama_model_placement_desc(m, s.data(), s.size());
    s.resize((size_t) n);
    return h16_of(s);
}

std::string server_cache_placement_identity(const common_params & p, const llama_context * ctx_tgt,
                                            const llama_context * ctx_dft) {
    return h16_of(string_format("numa=%d|t=%s|d=%s", (int) p.numa,
                                server_cache_model_placement(ctx_tgt ? llama_get_model(ctx_tgt) : nullptr).c_str(),
                                server_cache_model_placement(ctx_dft ? llama_get_model(ctx_dft) : nullptr).c_str()));
}

std::string server_cache_state_desc(const llama_context * ctx_tgt, const llama_context * ctx_dft,
                                    const std::string & carry_schema) {
    return h16_of("tgt:" + (ctx_tgt ? llama_state_seq_descriptor(ctx_tgt) : std::string("none")) + "\ndft:" +
                  (ctx_dft ? llama_state_seq_descriptor(ctx_dft) : std::string("none")) + "\ncarry:" + carry_schema);
}

std::string server_cache_ctx_canon(const llama_context * ctx) {
    if (ctx == nullptr) {
        return "none";
    }
    llama_context * c = const_cast<llama_context *>(ctx);
    std::string r = string_format("nb=%u;ub=%u;seq=%u;thr=%d/%d;place=%s;qsa_meta=[", llama_n_batch(ctx),
                                  llama_n_ubatch(ctx), llama_n_seq_max(ctx), llama_n_threads(c),
                                  llama_n_threads_batch(c), server_cache_model_placement(llama_get_model(ctx)).c_str());
    // QSA selection budget / policy as loaded (GGUF metadata; --override-kv is in the parsed part): indexer_* keys
    const llama_model * m = llama_get_model(ctx);
    char kb[256], vb[1024];
    for (int32_t i = 0; m && i < llama_model_meta_count(m); i++) {
        if (llama_model_meta_key_by_index(m, i, kb, sizeof(kb)) < 0) {
            continue;
        }
        const std::string k = kb;
        if (k.find("indexer") == std::string::npos && k.find("qsa") == std::string::npos) {
            continue;
        }
        const int n = llama_model_meta_val_str_by_index(m, i, vb, sizeof(vb));
        r += string_format("%zu:%s=%d:%s;", k.size(), k.c_str(), n, n >= 0 ? vb : "");
    }
    return r + "]";
}

std::string server_cache_param_digest(const common_params & params, const llama_context * ctx_tgt,
                                      const llama_context * ctx_dft) {
    return h16_of("p:" + server_cache_param_canon(params) + "|t:" + server_cache_ctx_canon(ctx_tgt) + "|d:" +
                  server_cache_ctx_canon(ctx_dft));
}

// the env registry, seeded from spec series D "Appendix: Registry seed" (Codex 2026-10-07). Only the exact class is
//   listed name by name: everything else - the registered semantic / unclear names AND every unregistered LLAMA_* /
//   GGML_* name - is semantic (unknown => strict). The inventory comes from the readers (helpers such as the Vulkan
//   backend's env lookups and the LLAMA_ARG_* aliases of common/arg.cpp), not from a literal grep of getenv
static const char * const CACHE_ENV_EXACT[] = {
    // speed / diagnostics / control by contract
    "LLAMA_TRACE", "LLAMA_BATCH_DEBUG", "LLAMA_GRAPH_INPUT_DEBUG", "LLAMA_GRAPH_RESULT_DEBUG", "LLAMA_KV_CACHE_DEBUG",
    "LLAMA_GRAPH_REUSE_DISABLE", "LLAMA_RESERVE_RUNTIME_SHAPE", "LLAMA_SAMPLER_FASTPATH", "LLAMA_DSV4_COMPRESS_DEBUG",
    "LLAMA_KQ_VIS_PROF", "LLAMA_QSA_DUMP_IDS", "LLAMA_QSA_CENSUS_CHECK", "LLAMA_QSA_COMPACT_BIAS_CHECK",
    "LLAMA_QSA_TAIL_ASSERT", "LLAMA_PLE_DUMP", "LLAMA_PLE_HOIST", "LLAMA_PLE_PREFETCH", "LLAMA_PLE_GATHER_SORT",
    "LLAMA_SERVER_SLOTS_DEBUG", "LLAMA_SERVER_SLOTS_N_DIFF", "LLAMA_SERVER_DEBUG_FAKE_TIMING", "LLAMA_SERVER_CHILD_MODE",
    "LLAMA_SERVER_ROUTER_PORT", "LLAMA_APP_CMD", "LLAMA_CACHE",
    "LLAMA_GTT_ADMISSION_MIB", "LLAMA_GTT_ADMISSION_MAX_WAIT_S", "LLAMA_GTT_SLOPE_B_PER_CELL",
    "LLAMA_GTT_STAGING_B_PER_CELL", "LLAMA_GTT_STAGING_STEP_MIB", "LLAMA_GTT_SLACK_PCT", "LLAMA_GTT_ONCE_MIB",
    "LLAMA_GTT_HOLES_FULL_CHUNK", "LLAMA_GTT_USED_PATH",
    "LLAMA_LEGACY_UNMARKED", // [TAG_CELL_EXT_PROVENANCE] which stored states load, not what a loaded one computes
    "LLAMA_DISK_ASYNC_TEST", // [TAG_DISK_ASYNC_SAVE] test hooks, honoured only with --cache-disk-test
    "LLAMA_SPEC_OVERLAP", // [TAG_T2_SEQ_RM_COUNTED] host-phase diet of the speculative round: output-exact (class A), so
                          //   OVERLAP=1 keeps the namespace key (the v68 store is reused); GGML_VK_F32ACC* stay semantic
    "LLAMA_GDN_REPLAY", // [TAG_GDN_REPLAY] GDN commit by replay: the state bytes and outputs are the planes path's (EXACT,
                        //   docs/verification/2026-10-31-gdn-replay-kproof.md), so =1 keeps the namespace key
    // LLAMA_ARG_* aliases (common/arg.cpp): classified one by one by their parsed meaning - only the control plane
    "LLAMA_ARG_HOST", "LLAMA_ARG_PORT", "LLAMA_API_KEY",
    // Vulkan speed / diagnostics only by contract
    "GGML_VK_PREFER_HOST_MEMORY", "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "GGML_VK_ALLOW_SYSMEM_FALLBACK",
    "GGML_VK_ENABLE_MEMORY_PRIORITY", "GGML_VK_FORCE_MAX_ALLOCATION_SIZE", "GGML_VK_FORCE_MAX_BUFFER_SIZE",
    "GGML_VK_SUBALLOCATION_BLOCK_SIZE", "GGML_VK_MAX_NODES_PER_SUBMIT", "GGML_VK_DISABLE_ASYNC",
    "GGML_VK_ALLOW_GRAPHICS_QUEUE", "GGML_VK_ASYNC_USE_TRANSFER_QUEUE", "GGML_VK_SERIALIZE_SUBMISSIONS",
    "GGML_VK_DISABLE_DESCRIPTOR_REUSE", "GGML_VK_FREE_WAIT_IDLE", "GGML_VK_ASYNC_READ_COPYBUF", "GGML_VK_ASYNC_READ_UMA",
    "GGML_VK_QSA_SCRATCH_PRESIZE", "GGML_VK_QSA_SCRATCH_PRESIZE_FA", "GGML_VK_DEBUG_MARKERS", "GGML_VK_PERF_LOGGER",
    "GGML_VK_PERF_LOGGER_CONCURRENT", "GGML_VK_PERF_LOGGER_FREQUENCY", "GGML_VK_SYNC_LOGGER", "GGML_VK_MEMORY_LOGGER",
    "GGML_VK_PIPELINE_STATS", "GGML_VK_OPT_WINDOW", "GGML_VK_FLOPS_SEED_LEGACY", "GGML_VK_SUBMIT_LOG",
    "GGML_VK_FUSION_DIAG", "GGML_VK_FUSION_COUNT", "GGML_VK_FUSION_LIST", "GGML_VULKAN_SKIP_CHECKS",
    "GGML_VULKAN_OUTPUT_TENSOR", "GGML_ALLOC_GROW_SLACK", "GGML_SCHED_PROFILE", "GGML_SCHED_DEBUG",
    "GGML_SCHED_DEBUG_REALLOC", "GGML_META_DEBUG", "GGML_NO_BACKTRACE", "GGML_BACKTRACE_LLDB", "GGML_GET_ROWS_SORTED",
    "GGML_GET_ROWS_STATS", "GGML_GET_ROWS_THREADS",
};

server_cache_env_class server_cache_env_classify(const std::string & name) {
    if (name.rfind("LLAMA_ARG_", 0) == 0) {
        // the disk cache's own options (neutral since v1) and every other option alias: common_params_parse reads it
        //   into common_params like its argv form, so the parsed-param digest (server_cache_param_canon) carries its
        //   semantic meaning - argv and env aliases of one option give one key
        return SERVER_CACHE_ENV_EXACT;
    }
    for (const char * e : CACHE_ENV_EXACT) {
        if (name == e) {
            return SERVER_CACHE_ENV_EXACT;
        }
    }
    return SERVER_CACHE_ENV_SEMANTIC;
}

std::string server_cache_env_digest(const std::vector<std::string> & env) {
    std::vector<std::string> sem;
    for (const auto & kv : env) {
        const auto eq = kv.find('=');
        if (server_cache_env_classify(kv.substr(0, eq)) == SERVER_CACHE_ENV_SEMANTIC) {
            sem.push_back(kv);
        }
    }
    if (sem.empty()) {
        return "-";
    }
    std::sort(sem.begin(), sem.end());
    std::string s;
    for (const auto & kv : sem) {
        s += std::to_string(kv.size()) + ":" + kv + ";";
    }
    return h16_of(s);
}

std::vector<std::string> server_cache_environ() {
    std::vector<std::string> out;
    for (char ** ep = environ; ep && *ep; ep++) {
        const std::string kv = *ep;
        if (kv.rfind("LLAMA_", 0) == 0 || kv.rfind("GGML_", 0) == 0) {
            out.push_back(kv);
        }
    }
    return out;
}

std::string server_cache_file_identity(const std::string & p) {
    if (p.empty()) {
        return "none";
    }
    std::error_code ec;
    const uint64_t n = fs::file_size(p, ec);
    std::ifstream f(p, std::ios::binary);
    if (ec || !f.is_open()) {
        return "absent";
    }
    // 4 samples of 1 MiB at fixed relative offsets (head, 1/3, 2/3, tail) + the size: 4 MiB per shard
    constexpr uint64_t W = 1ull << 20;
    constexpr int      K = 4;
    const uint64_t len = std::min(W, n);
    std::string buf = string_format("n=%llu;", (unsigned long long) n);
    std::vector<char> tmp(len);
    for (int i = 0; i < K; i++) {
        const uint64_t off = n <= W ? 0 : i == K - 1 ? n - W : (n - W) / (K - 1) * i;
        f.seekg((std::streamoff) off);
        f.read(tmp.data(), (std::streamsize) len);
        if (!f || (uint64_t) f.gcount() != len) {
            return "absent";
        }
        buf.append(tmp.data(), len);
        if (n <= W) {
            break;
        }
    }
    return h16_of(buf);
}

std::string server_disk_cache_model_identity(const common_params & params) {
    // every shard of a split GGUF (<name>-00001-of-0000N.gguf)
    const auto shards = [](const std::string & p) {
        std::string id = server_cache_file_identity(p);
        const auto at = p.rfind("-00001-of-");
        if (at != std::string::npos && p.size() >= at + 15) {
            const int n = atoi(p.substr(at + 10, 5).c_str());
            for (int i = 2; i <= n && i < 100000; i++) {
                id += "," + server_cache_file_identity(p.substr(0, at) + string_format("-%05d-of-", i) + p.substr(at + 10));
            }
        }
        return id;
    };
    return "m:" + shards(params.model.path) + "|d:" + shards(params.speculative.draft.mparams.path) +
           "|p:" + shards(params.mmproj.path);
}

std::string server_disk_cache_fingerprint_v2(const std::string & cheap, const server_cache_key_parts & p) {
    const std::string body = string_format("c2|model=%s|desc=%s|meas=%s|depth=%u|env=%s|param=%s", p.model_id.c_str(),
                                           p.desc_h.c_str(), p.measure_h.c_str(), p.depth_version, p.env_digest.c_str(),
                                           p.param_digest.c_str());
    const auto sp = cheap.find(' ');
    const std::string cheap_body = sp == std::string::npos ? cheap : cheap.substr(sp + 1);
    return h16_of(body) + " " + cheap_body + "|compat2=" + std::to_string(body.size()) + ":" + body;
}

std::string server_disk_cache_compat2_body(const std::string & fp) {
    const auto at = fp.rfind("|compat2=");
    if (at == std::string::npos) {
        return "";
    }
    const size_t a = at + 9;
    const auto c = fp.find(':', a);
    if (c == std::string::npos || c == a || c - a > 9 || fp.find_first_not_of("0123456789", a) != c) {
        return "";
    }
    const size_t n = std::stoul(fp.substr(a, c - a));
    return fp.size() == c + 1 + n ? fp.substr(c + 1) : "";
}

bool server_disk_cache_same_identity(const std::string & a, const std::string & b) {
    const std::string ba = server_disk_cache_compat2_body(a);
    const std::string bb = server_disk_cache_compat2_body(b);
    if (ba.empty() || bb.empty()) {
        return a == b;
    }
    return a.size() >= 16 && b.size() >= 16 && a.compare(0, 16, b, 0, 16) == 0 && ba == bb;
}

// [TAG_CACHE_COMPAT2] one /proc/self/maps line (astra compat2 r2 mapped-library identity): the mapping's device +
//   inode must be those of the path as stat() sees it now - a file replaced or unlinked after it was mapped would
//   otherwise lend the new file's identity to the old loaded code
int server_cache_maps_lib(const std::string & line, std::string & entry) {
    const auto at = line.find('/');
    if (at == std::string::npos) {
        return 0;
    }
    std::string p = line.substr(at);
    const std::string del = " (deleted)";
    const bool deleted = p.size() > del.size() && p.compare(p.size() - del.size(), del.size(), del) == 0;
    if (deleted) {
        p.resize(p.size() - del.size());
    }
    const std::string base = fs::path(p).filename().string();
    if (base.find(".so") == std::string::npos || base.rfind("libllama", 0) == 0 || base.rfind("libggml", 0) == 0 ||
        base.rfind("libmtmd", 0) == 0) {
        return 0;
    }
    if (deleted) {
        return -1;
    }
    // "addr perms offset MM:mm inode   path": device major:minor (hex) and inode (decimal) of the mapped file
    unsigned int maj = 0, mnr = 0;
    unsigned long long ino = 0;
    char addr[128], perms[16], off[32];
    if (sscanf(line.c_str(), "%127s %15s %31s %x:%x %llu", addr, perms, off, &maj, &mnr, &ino) != 6) {
        return -1;
    }
    struct stat st {};
    if (stat(p.c_str(), &st) != 0 || (unsigned long long) st.st_ino != ino || major(st.st_dev) != maj ||
        minor(st.st_dev) != mnr) {
        return -1;
    }
    entry = string_format("%s:%lld:%lld.%09ld:%llu", p.c_str(), (long long) st.st_size, (long long) st.st_mtim.tv_sec,
                          (long) st.st_mtim.tv_nsec, (unsigned long long) st.st_ino);
    return 1;
}

std::string server_cache_runtime_identity() {
    // GPU devices: name, description, PCI id (ggml's device description carries no driver version - the Vulkan
    //   backend's description is the device name only; the driver is covered by its mapped libraries below)
    std::string gpu;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        gpu += string_format("%s/%s/%s;", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                             props.device_id ? props.device_id : "-");
    }
    // backend features (the CPU backend's ISA list: AVX2, AVX512, ... and any other backend's feature list)
    std::string feat;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        auto get = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        feat += std::string(ggml_backend_reg_name(reg)) + "{";
        for (const ggml_backend_feature * f = get ? get(reg) : nullptr; f && f->name; f++) {
            feat += string_format("%s=%s,", f->name, f->value ? f->value : "");
        }
        feat += "}";
    }
    // the driver / runtime: every other mapped shared object (Vulkan loader + ICD, libdrm, LLVM, libc / libm /
    //   libstdc++, ...) by path + size + mtime + inode - an upgrade or a swap changes it. libllama / libggml* /
    //   libmtmd are content-hashed in the cheap identity already
    std::set<std::string> libs;
    bool verified = true;
    {
        std::ifstream maps("/proc/self/maps");
        verified = maps.good();
        std::string line;
        while (std::getline(maps, line)) {
            std::string entry;
            const int r = server_cache_maps_lib(line, entry);
            if (r < 0) {
                verified = false;
            } else if (r > 0) {
                libs.insert(entry);
            }
        }
    }
    if (!verified) {
        return SERVER_CACHE_RUNTIME_UNVERIFIED;
    }
    std::string lib;
    for (const auto & l : libs) {
        lib += l + ";";
    }
    // the driver's own environment (not LLAMA_* / GGML_*): RADV / Mesa / ACO / AMD / Vulkan loader / ROCm / CUDA /
    //   OpenMP switches can change the compiled kernels
    std::vector<std::string> denv;
    static const char * const DRIVER_ENV[] = { "RADV_", "MESA_", "ACO_", "AMD_", "VK_", "NIR_", "HSA_", "HIP_", "ROCR_",
                                               "ROC_", "CUDA_", "NVIDIA_", "__GL", "OMP_", "GOMP_", "KMP_", "ZES_",
                                               "SYCL_", "ONEAPI_", "DRI_", "LD_PRELOAD", "LD_LIBRARY_PATH" };
    for (char ** ep = environ; ep && *ep; ep++) {
        const std::string kv = *ep;
        for (const char * pre : DRIVER_ENV) {
            if (kv.rfind(pre, 0) == 0) {
                denv.push_back(kv);
                break;
            }
        }
    }
    std::sort(denv.begin(), denv.end());
    std::string env;
    for (const auto & kv : denv) {
        env += std::to_string(kv.size()) + ":" + kv + ";";
    }
    return "gpu=" + (gpu.empty() ? std::string("none") : gpu) + "|feat=" + feat + "|libs=" + h16_of(lib) +
           "|denv=" + h16_of(env);
}

std::string server_cache_memo_key(const std::string & cheap, const std::string & runtime_id,
                                  const std::string & placement_id) {
    if (runtime_id == SERVER_CACHE_RUNTIME_UNVERIFIED) {
        return ""; // fail-closed: no memo is read or published for an unverifiable runtime
    }
    return h16_of(string_format("cheap=%s|probe=%u|runtime=%s|place=%s", cheap.c_str(), DISK_CACHE_PROBE_REVISION,
                                runtime_id.c_str(), placement_id.c_str()));
}

static fs::path memo_path(const std::string & root, const std::string & memo_key) {
    return fs::path(root) / "compat-memo" / (memo_key + ".txt");
}

// [TAG_CACHE_COMPAT2] (Codex compat2 C memo race) one record per file, validated whole on read:
//   "compat-memo/1 key=<memo key> meas=<hash16> sum=<hash16 of the text before ' sum='>\n" and nothing else
static std::string memo_record(const std::string & memo_key, const std::string & measure_h) {
    const std::string head = "compat-memo/1 key=" + memo_key + " meas=" + measure_h;
    return head + " sum=" + h16_of(head) + "\n";
}

static bool is_hash16(const std::string & v) {
    return v.size() == 16 && v.find_first_not_of("0123456789abcdef") == std::string::npos;
}

bool server_cache_memo_get(const std::string & root, const std::string & memo_key, std::string & measure_h) {
    std::ifstream f(memo_path(root, memo_key), std::ios::binary);
    if (!f.is_open()) {
        return false;
    }
    std::string text;
    char buf[512];
    while (f.read(buf, sizeof(buf)) || f.gcount() > 0) {
        text.append(buf, (size_t) f.gcount());
        if (text.size() > 4096) {
            return false;
        }
    }
    if (f.bad()) {
        return false;
    }
    const std::string pre = "compat-memo/1 key=" + memo_key + " meas=";
    if (text.rfind(pre, 0) != 0 || text.size() < pre.size() + 16) {
        return false;
    }
    const std::string v = text.substr(pre.size(), 16);
    if (!is_hash16(v) || text != memo_record(memo_key, v)) {
        return false; // truncated, torn, foreign key or a corrupt checksum
    }
    measure_h = v;
    return true;
}

bool server_cache_memo_put(const std::string & root, const std::string & memo_key, const std::string & measure_h) {
    if (!is_hash16(measure_h)) {
        return false; // only a real measurement is ever published
    }
    std::error_code ec;
    const fs::path p = memo_path(root, memo_key);
    fs::create_directories(p.parent_path(), ec);
    if (ec) {
        return false;
    }
    // a unique temporary per writer: two publishers never share an inode; rename is the last step
    static std::atomic<uint64_t> n_put{0};
    const fs::path tmp = p.string() + ".tmp." + std::to_string((long long) getpid()) + "." +
                         std::to_string(n_put.fetch_add(1)) + "." +
                         std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    const std::string rec = memo_record(memo_key, measure_h);
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        return false;
    }
    bool ok = true;
    for (size_t off = 0; ok && off < rec.size();) {
        const ssize_t n = write(fd, rec.data() + off, rec.size() - off);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        ok = n > 0;
        off += ok ? (size_t) n : 0;
    }
    ok = ok && fsync(fd) == 0;
    ok = (close(fd) == 0) && ok;
    if (ok) {
        fs::rename(tmp, p, ec);
        ok = !ec;
    }
    if (!ok) {
        fs::remove(tmp, ec);
        return false;
    }
    const int dfd = open(p.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd >= 0) {
        fsync(dfd); // the rename's durability; best effort (a lost rename is a re-measurement, never a wrong key)
        close(dfd);
    }
    return true;
}

std::string server_cache_measure_component(const std::string & measure_h) {
    // fail-closed: a failed measurement gets a namespace no other start can reach (random), so nothing written under
    //   it is ever reused and no compatibility claim is made
    return measure_h.empty() ? "unmeasured-" + random_id() : measure_h;
}

std::string server_cache_measure_hash(const std::string & bytes) {
    return h16_of(bytes);
}

bool server_disk_cache_desc_ok(bool own, uint32_t version, const std::string & desc_full, const std::string & desc_key) {
    const bool need_full = own || version >= 8;
    return (!need_full && desc_full.empty()) || desc_full == desc_key;
}

server_disk_cache::stream_rule server_disk_cache::rule_for(const server_disk_cache_entry & e, bool tgt, bool tgt_only) const {
    stream_rule r;
    // the writer's full descriptor must be ours, else nothing passes. Mandatory (absent / empty => miss) for an own
    //   entry and for any format >= 8 entry (astra compat2 r2); optional only for an adopted pre-8 entry, which has none.
    //   [TAG_ADOPT_TARGET_ONLY] an adopt source's target-only entry: its descriptor_full names its own draft, which
    //   differs by design - still mandatory (present) at format >= 8, each stream's own full-coverage hash decides
    const bool desc_pass = (tgt_only && !e.src.empty()) ? (e.version < 8 || !e.desc_full.empty())
                                                       : server_disk_cache_desc_ok(e.src.empty(), e.version, e.desc_full, desc_key);
    if (!desc_pass) {
        DSK_WRN("entry %s: written under descriptor %s, this context is %s\n", e.id.c_str(), e.desc_full.c_str(),
                desc_key.c_str());
        return r;
    }
    r.h2        = tgt ? desc_tgt_full : desc_dft_full;
    r.h1        = tgt ? desc_tgt_h : desc_dft_h;
    r.v1_ok     = !e.src.empty() && e.version == 7;
    r.legacy_ok = legacy_payload_ok(e);
    return r;
}

void server_disk_cache::set_fingerprint(const std::string & fp) {
    fingerprint = fp;
    dir = (fs::path(root) / ("ns-" + fp.substr(0, 16))).string();
}

server_disk_cache::server_disk_cache(std::string dir, uint64_t quota_bytes, int32_t min_tokens, int32_t ttl_days,
                                     std::string fingerprint, bool has_mtmd, float ckpt_ratio, int32_t ckpt_min_step,
                                     int32_t ckpt_max)
    : root(std::move(dir)), quota(quota_bytes), n_min_tokens(min_tokens), ttl_days(ttl_days),
      fingerprint(std::move(fingerprint)), has_mtmd(has_mtmd), ckpt_ratio(ckpt_ratio),
      ckpt_min_step(std::max(1, ckpt_min_step)), ckpt_max(std::max(2, ckpt_max)) {
    this->dir = (fs::path(root) / ("ns-" + this->fingerprint.substr(0, 16))).string();
    // [TAG_CELL_EXT_PROVENANCE] LLAMA_LEGACY_UNMARKED=0: no pre-fix (unmarked) state or checkpoint is ever vouched for -
    //   every one is refused (a cache miss); default 1: vouched for when its stored token stream proves it text-only
    const char * lu = getenv("LLAMA_LEGACY_UNMARKED");
    legacy_unmarked = !(lu && atoi(lu) == 0);
}

// [TAG_DISK_ASYNC_SAVE] (review r7 5) the test hooks: one variable LLAMA_DISK_ASYNC_TEST=key=value,... read here and only
//   with --cache-disk-test (enabled = false: the variable is ignored with a warning - a leaked env never changes a prod
//   cache). Keys:
//     delay_ms=N             each background block task sleeps N ms first
//     threads=N              worker count
//     fail_maps=N            the first N map writes throw (a worker-side failure)
//     fail_meta=N            the first N async commits fail their .meta (a commit-side failure)
//     fail_job=N             the N-th submitted async save (1-based) fails at its end
//     finalize_delay_ms=N    each save's end (maps) lags its block tasks by N ms
//     finalize_delay_jobK_ms=N  the same for the K-th submitted save only (overrides the above)
//     hold_write_jobK_ms=N   the K-th submitted save's first block write holds N ms while marked "being written"
//     fail_enqueue=N         the N-th block task enqueue (1-based, all saves) throws std::bad_alloc
//     fail_finalize=N        the N-th finalizer task enqueue (a save whose blocks were all done - or inline only - at
//                            the end of its copy; 1-based) throws std::bad_alloc
//     max_jobs=N             saves in flight before a new save runs synchronously (default 2)
//     check_pins=1           after each commit compare the cached pin set with a fresh derivation (pin_check_failures)
//     quota_mib=N            the quota in MiB instead of --cache-disk-gib
//     incr_verify=1          [TAG_DISK_INCR] with --cache-disk-incr: copy the blocks the incremental save would reuse and
//                            compare their keys with the record (incr_verify_mismatch in GET /disk-cache, error log)
//     fail_incr_alloc=N      (sc3fix 2) the N-th reused block's record allocation throws std::bad_alloc
//     fail_incr_rec=N        (sc3fix 3) the N-th slot record built at a commit throws std::bad_alloc
void server_disk_cache::set_test_hooks(bool enabled) {
    const char * v = getenv("LLAMA_DISK_ASYNC_TEST");
    if (!v || !*v) {
        return;
    }
    if (!enabled) {
        DSK_WRN("LLAMA_DISK_ASYNC_TEST=%s ignored: test hooks need --cache-disk-test\n", v);
        return;
    }
    for (const auto & kv : string_split<std::string>(v, ',')) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        const std::string k = kv.substr(0, eq);
        const int val = atoi(kv.c_str() + eq + 1);
        if (k == "delay_ms") {
            hooks.delay_ms = std::max(0, val);
        } else if (k == "threads") {
            hooks.threads = std::max(0, val);
        } else if (k == "fail_maps") {
            hooks.fail_maps = std::max(0, val);
        } else if (k == "fail_meta") {
            test_fail_meta = std::max(0, val);
        } else if (k == "fail_job") {
            hooks.fail_job = val;
        } else if (k == "finalize_delay_ms") {
            hooks.finalize_delay_ms = std::max(0, val);
        } else if (int job = 0, n_used = 0;
                   sscanf(k.c_str(), "finalize_delay_job%d_ms%n", &job, &n_used) == 1 && n_used == (int) k.size()) {
            hooks.finalize_delay_job[job] = std::max(0, val);
        } else if (int job = 0, n_used = 0;
                   sscanf(k.c_str(), "hold_write_job%d_ms%n", &job, &n_used) == 1 && n_used == (int) k.size()) {
            hooks.hold_write_job[job] = std::max(0, val);
        } else if (k == "fail_enqueue") {
            hooks.fail_enqueue = val;
        } else if (k == "fail_finalize") {
            hooks.fail_finalize = val;
        } else if (k == "check_pins") {
            hooks.check_pins = val != 0;
        } else if (k == "max_jobs") {
            hooks.max_jobs = std::max(1, val);
        } else if (k == "incr_verify") { // [TAG_DISK_INCR] reuse nothing, compare the recorded keys with the device's
            hooks.incr_verify = val != 0;
        } else if (k == "fail_incr_alloc") { // (sc3fix 2)
            hooks.fail_incr_alloc = val;
        } else if (k == "fail_incr_rec") { // (sc3fix 3)
            hooks.fail_incr_rec = val;
        } else if (k == "quota_mib") {
            quota = (uint64_t) std::max(1, val) << 20;
        } else {
            DSK_WRN("LLAMA_DISK_ASYNC_TEST: unknown key %s\n", k.c_str());
        }
    }
    DSK_WRN("test hooks active (--cache-disk-test): %s\n", v);
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
        { "draft_cover", e.draft_cover      }, // [TAG_PARTIAL_PROMPT]
    };
    if (!e.src.empty()) {
        return false; // [TAG_DISK_ADOPT] an adopt source is never written
    }
    if (e.src.empty() || !e.desc_full.empty()) {
        j["descriptor_full"] = e.src.empty() ? desc_key : e.desc_full; // [TAG_CACHE_COMPAT2]
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
        erase_entry_at(idx);
        return;
    }
    remove_files(entries[idx].id);
    erase_entry_at(idx);
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
        if (ver < 5 || ver > DISK_CACHE_VERSION || !server_disk_cache_same_identity(j.value("fingerprint", std::string()), fp)) {
            return SEALED_BAD;
        }
        e.version = ver;
        e.adopted_from = j.value("adopted_from", std::string());
        e.desc_full    = j.value("descriptor_full", std::string());
        e.bytes   = j.at("bytes").get<uint64_t>();
        e.t_saved = j.at("t_saved").get<int64_t>();
        e.t_used  = j.at("t_used").get<int64_t>();
        e.hits    = j.at("hits").get<int32_t>();
        e.has_dft = j.at("has_dft").get<bool>();
        // [TAG_PARTIAL_PROMPT] older .meta files have no draft_cover: their writers kept C1 (no partial prompt saved),
        //   except v50rc4, which published adopted target-only states (own namespace, adopted_from set) - unknown cover,
        //   taken as none
        e.draft_cover = j.contains("draft_cover") ? j.at("draft_cover").get<int64_t>()
                      : (fp == fingerprint && !e.adopted_from.empty() && e.has_dft) ? 0
                      : (int64_t) j.at("n_tokens").get<size_t>();

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
            clear_entries();
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
            push_entry(std::move(e));
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
                 get_str_le(f, dh) && env_ver == ENVELOPE_VERSION && dh == desc_key && get_str(f, m.key) &&
                 get(f, m.n_tokens) && get(f, m.pos_min) && get(f, m.pos_max) && m.key + ".ck" == name;
            if (ok) {
                // [TAG_CELL_EXT_PROVENANCE] the first 20 bytes of the target blob: flags_tgt, flags_dft, u64 length, blob
                uint32_t ft = 0, fd = 0;
                uint64_t n_tgt = 0;
                uint8_t  head[CKPT_IXY_HEAD] = {};
                m.ixy_marked = get(f, ft) && get(f, fd) && get(f, n_tgt) && n_tgt >= sizeof(head) &&
                               (bool) f.read(reinterpret_cast<char *>(head), sizeof(head)) &&
                               ckpt_blob_ixy_marked(head, sizeof(head));
            }
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
            erase_entry_at(i);
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
            "deep ratio %.2f min step %d max %d semantic %d), %zu + %zu stale/foreign dropped, legacy unmarked states: %s\n",
            dir.c_str(), entries.size(), ckpts.size(), total_bytes() / (1024.0 * 1024.0 * 1024.0),
            quota / (1024.0 * 1024.0 * 1024.0), n_min_tokens, ttl_days, ckpt_ratio, ckpt_min_step, ckpt_max, ckpt_sem_max,
            n_dropped, n_ck_dropped,
            legacy_unmarked ? "accepted with a text-only proof (LLAMA_LEGACY_UNMARKED=1)" : "refused (LLAMA_LEGACY_UNMARKED=0)");

    refresh_status();
}

void server_disk_cache::enforce(const std::string & protect, bool refresh) {
    const int64_t t_now = now_s();
    // [TAG_DISK_ASYNC_SAVE] the pinned entries (the in-flight model before pinned_set()), once per pass: TTL and eviction
    //   never remove one, so the set does not change inside this function
    const std::set<std::string> pinned = pinned_set(); // (review r12 6) a copy: TTL / eviction erase() bump the generation

    if (ttl_days > 0) {
        for (size_t i = 0; i < entries.size();) {
            if (entries[i].src.empty() && entries[i].id != protect && entries[i].id != protect_id &&
                !pinned.count(entries[i].id) && t_now - entries[i].t_used > (int64_t) ttl_days * 86400) {
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

    // [TAG_DISK_ASYNC_SAVE] the saves in flight are charged (inflight_charge(): pending, capped by what this victim
    //   search can free), so the quota holds while they write; TTL and the collections above always run
    const uint64_t charge = inflight_charge(pinned, protect);
    while (total_bytes() + charge > quota && !entries.empty()) {
        size_t victim = SIZE_MAX;
        double v_min  = 0.0;
        for (size_t i = 0; i < entries.size(); i++) {
            // [TAG_DISK_ASYNC_SAVE] a pinned entry stays: the save that extends it may still fail (the conversation
            //   would be gone) and its own files are already netted out of the charge; with only pinned / protected
            //   entries left the loop stops
            if (entries[i].id == protect || entries[i].id == protect_id || !entries[i].src.empty() ||
                pinned.count(entries[i].id)) {
                continue;
            }
            const double v = entry_value(entries[i], t_now);
            if (victim == SIZE_MAX || v < v_min) {
                victim = i;
                v_min  = v;
            }
        }
        if (victim == SIZE_MAX) {
            break; // only pinned / protected entries left
        }
        DSK_INF("quota: evicting %s (%zu tokens, %.2f GiB, value %.1f s/GiB)\n", entries[victim].id.c_str(),
                entries[victim].tokens.size(), entries[victim].bytes / (1024.0 * 1024.0 * 1024.0), v_min);
        erase(victim);
        n_evict++;
        deep_gc(); // the evicted entry's deep checkpoints and blocks may now be unreferenced
        blob_gc();
    }

    if (refresh) {
        refresh_status();
    }
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
    if (prompt.partial) { // [TAG_PARTIAL_PROMPT] C1: a partial prompt (legacy /slots restore) is never saved
        n_save_partial++;
        DSK_INF("not saving %zu tokens: a partial prompt (legacy /slots restore, draft side incomplete)\n", n);
        refresh_status();
        return false;
    }
    // [TAG_CACHE_COMPAT2] (astra compat2 r2) a slot that came from an adopted entry is resaved under this namespace
    //   (full descriptor) only because its state was fully restored here (load() sets adopted_from after streams +
    //   carry) AND its source is an adopt key approved for this run (--cache-disk-adopt). A target-only adopted load
    //   ([TAG_ADOPT_TARGET_ONLY]) restored the target stream + reset the draft side: its target state is canonical,
    //   saved draft-partial ([TAG_DRAFT_PARTIAL])
    if (!prompt.adopted_from.empty() &&
        std::none_of(adopted.begin(), adopted.end(), [&](const adopt_src & a) { return a.key == prompt.adopted_from; })) {
        DSK_WRN("not saving %zu tokens: adopted from %s, which is not an approved adopt source of this run\n", n,
                prompt.adopted_from.c_str());
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
    if (pending_holds(prompt.tokens)) { // [TAG_DISK_ASYNC_SAVE] a save of exactly these tokens is in flight
        return true;
    }

    // [TAG_DISK_ASYNC_SAVE] (review r10 4, r11 1-2) backpressure, decided before any file of this save is written: with
    //   two saves in flight (fixed; test hook max_jobs) this one runs synchronously (bounded by one save, as v63; the
    //   entry is registered before any enforce() runs and the saves in flight keep their pins) - the loop never waits for an in-flight
    //   save's remaining hash / write time, and no commit / quota pass runs while this save's files are unregistered.
    //   Quota overshoot is bounded by the new blocks of max_jobs saves in flight
    bool use_async = as != nullptr;
    if (as) {
        // (review r12 3) finished saves are committed first (nothing of this save is on disk yet), so a done-but-uncommitted
        //   job does not turn this save synchronous; their failures are reported by the next poll()
        poll_into(deferred_lost);
        const int max_jobs = hooks.max_jobs > 0 ? hooks.max_jobs : 2;
        use_async = (int) n_pending() < max_jobs;
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
    e.desc_full = desc_key; // [TAG_CACHE_COMPAT2] in memory as in its .meta: an own entry is checked against it
    e.adopted_from = prompt.adopted_from; // [TAG_DISK_ADOPT] provenance of a resave after an adopted load
    e.has_dft = ctx_dft != nullptr;
    // [TAG_DRAFT_PARTIAL] the prompt rows the draft history holds: all of them, or n - draft_gap for a draft-partial
    //   prompt (a partial one never reaches this point, C1)
    e.draft_cover = prompt.draft_cover_rows();

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
            // [TAG_CACHE_COMPAT2] adopted pre-8 checkpoints (foreign_unverified) are left out: never stamped
            // [TAG_SLOT_FILE_CKPT_NO_DEEP] (astra pkg2 Important, merge review M1) so are checkpoints restored from a
            //   /slots file (id_task -2): stored here they would enter the shared checkpoint store, come back from a
            //   load as id_task -1 and be found by rewinds / cache_pin like deep checkpoints - RAM use only
            std::vector<const common_prompt_checkpoint *> own;
            size_t n_slot_file = 0;
            for (const auto & c : prompt.checkpoints) {
                if (c.id_task == SERVER_SLOT_FILE_CKPT_ID_TASK) {
                    n_slot_file++;
                } else if (!c.foreign_unverified) {
                    own.push_back(&c);
                }
            }
            if (own.size() + n_slot_file != prompt.checkpoints.size()) {
                DSK_INF("%zu adopted pre-8 checkpoint(s) not written under the full descriptor (read-only at their "
                        "source)\n", prompt.checkpoints.size() - own.size() - n_slot_file);
            }
            if (n_slot_file > 0) {
                DSK_INF("%zu checkpoint(s) restored from a slot file not written (RAM use only)\n", n_slot_file);
            }
            std::vector<size_t> ns;
            for (const auto * c : own) {
                ns.push_back((size_t) c->n_tokens);
            }
            std::vector<size_t> ns_sorted = ns;
            std::sort(ns_sorted.begin(), ns_sorted.end());
            ns_sorted.erase(std::unique(ns_sorted.begin(), ns_sorted.end()), ns_sorted.end());
            const std::vector<std::string> keys_sorted = prompt.tokens.prefix_keys(ns_sorted);
            put<uint32_t>(f, AUX_MAGIC);
            put<uint32_t>(f, (uint32_t) own.size());
            for (const auto * cp : own) {
                const auto & c = *cp;
                const auto at = std::lower_bound(ns_sorted.begin(), ns_sorted.end(), (size_t) c.n_tokens) - ns_sorted.begin();
                const std::string & key = keys_sorted[at];
                bool written = false;
                // [TAG_DRAFT_PARTIAL] a draft-partial prompt's checkpoint draft data would carry the gap into the shared,
                //   content-addressed store (another conversation restoring it would get a draft without its first
                //   rows): stored draft-less, which a restore turns into a draft reset up to it (draft-partial again)
                if (prompt.draft_partial() && !c.data_dft.empty()) {
                    common_prompt_checkpoint t = c;
                    t.data_dft.clear();
                    if (!ckpt_store(key, t, &written)) {
                        throw std::runtime_error("cannot store a checkpoint");
                    }
                } else if (!ckpt_store(key, c, &written)) {
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
                if (ck != ckpts.end() && !ck->second.src_dir.empty() && ck->second.src_version < 8) {
                    // [TAG_CACHE_COMPAT2] (astra compat2 r2) a pre-8 source's checkpoint was never restored here: no
                    //   copy under the full descriptor - the ref stays, read-only at its source (a gap without it)
                } else if (ck != ckpts.end() && !ck->second.src_dir.empty()) {
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

    e.deep = deep; // (Codex gate r3 Major 5) the refs the .aux holds (published with the entry only)
    uint64_t ckpt_bytes = 0;
    for (const auto & c : prompt.checkpoints) {
        ckpt_bytes += c.foreign_unverified ? 0 : c.size();
    }

    // [TAG_DISK_ASYNC_SAVE] --cache-disk-async: the streams are copied off the device here, hashed and written by the
    //   workers; poll() commits the entry (save_commit) once its files are on disk
    if (ok && use_async) {
        return save_async_begin(ctx_tgt, ctx_dft, id_slot, e, prompt, prompt.checkpoints.size(), ckpt_bytes,
                                n_ckpt_written, t_start);
    }

    if (ok && as) { // (review r12 4) counted when the synchronous fallback starts writing
        n_save_backpressure++;
        if (!warned_backpressure) {
            warned_backpressure = true;
            DSK_WRN("%zu saves in flight: this save runs synchronously (counted as save_backpressure; logged once)\n",
                    n_pending());
        }
    }

    // [TAG_DISK_DELTA] the KV / recurrent state streams through 4 MiB content blocks: a block already on disk (the
    //   unchanged part of a conversation saved before, a branch, a copy) is referenced, only new ones are written
    uint64_t n_logical = 0, n_written = 0;
    try {
        ok = ok && save_stream(ctx_tgt, id_slot, path(e.id, ".tgt"), desc_tgt_full, e.blobs, n_logical, n_written);
        if (ok && ctx_dft) {
            ok = save_stream(ctx_dft, id_slot, path(e.id, ".dft"), desc_dft_full, e.blobs, n_logical, n_written);
        }
    } catch (const std::exception & err) {
        if (!as) {
            throw; // v63
        }
        // (review r13 2) [TAG_DISK_ASYNC_SAVE] the synchronous fallback fails like a write error: claims released,
        //   partial files removed by save_commit
        DSK_WRN("synchronous save failed: %s\n", err.what());
        ok = false;
    }
    release_sync_claims();

    return save_commit(e, ok, prompt.tokens, prompt.checkpoints.size(), ckpt_bytes, n_logical, n_written, n_ckpt_written,
                       t_start, 0, 0, 0);
}

// (review r14 7) the one prefix predicate of the in-flight model: a is a prefix of b (strict: a proper prefix)
static bool tokens_prefix_of(const server_tokens & a, const server_tokens & b, bool strict) {
    return (strict ? a.size() < b.size() : a.size() <= b.size()) && b.get_common_prefix(a) == a.size();
}

bool server_disk_cache::save_commit(server_disk_cache_entry & e, bool ok, const server_tokens & tokens, size_t n_ckpts,
                                    uint64_t ckpt_bytes, uint64_t n_logical, uint64_t n_written, uint64_t n_ckpt_written,
                                    int64_t t_start, uint64_t blocking_us, int64_t t_done, int64_t t_captured) {
    const size_t n = tokens.size();

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
        e.tokens  = tokens.clone();
        e.t_saved = e.t_used = now_s();
        e.bytes   = n_logical + file_size_or_0(path(e.id, ".tok")) + file_size_or_0(path(e.id, ".aux")) + ckpt_bytes;
        for (const char * ext : { ".tok", ".aux", ".tgt", ".dft" }) {
            e.own_bytes += file_size_or_0(path(e.id, ext));
        }
        if (blocking_us != 0 && test_fail_meta > 0) { // test hook fail_meta
            test_fail_meta--;
            DSK_WRN("%s", "injected .meta write failure (test hook fail_meta)\n");
            ok = false;
        } else {
            ok = write_meta(e);
        }
    }

    if (!ok) {
        DSK_WRN("failed to save %zu tokens, removing the partial entry\n", n);
        remove_files(e.id);
        if (blocking_us != 0) {
            // [TAG_DISK_ASYNC_SAVE] (review r3 I3, m7) commit_job collects and refreshes once, after it released the
            //   job's block claims (a collection here would still see them as referenced)
            n_saves_async_failed++;
            return false;
        }
        // [TAG_DISK_GC] the aborted save is a transaction boundary: blocks and checkpoints it wrote that no
        //   registered entry or live slot references go now (never inside erase(): during a prefix replacement the
        //   new entry's files exist before it is registered)
        deep_gc();
        blob_gc();
        check_pins();
        refresh_status();
        return false;
    }

    // [TAG_DISK_ASYNC_SAVE] (review r13 1) a synchronous save is newer than every save in flight: those it extends are
    //   obsolete now - marked here (they stop pinning, so the pass below replaces their prefixes) and dropped at their
    //   commit. An async commit marks nothing (commits go in submission order: a later, shorter save is newer)
    if (blocking_us == 0) {
        mark_superseded(tokens);
    }

    // older saves of the same conversation that this one extends are obsolete - except one another save in flight also
    //   extends ([TAG_DISK_ASYNC_SAVE] (review r9 10) pinned: if that save fails, the prefix is still its conversation's)
    const std::set<std::string> pinned = pinned_set(); // (review r12 6) a copy: erase() bumps the generation
    for (size_t i = 0; i < entries.size();) {
        if (entries[i].id != protect_id && !pinned.count(entries[i].id) && tokens_prefix_of(entries[i].tokens, tokens, true)) {
            erase(i);
        } else {
            i++;
        }
    }

    const double ms = (ggml_time_us() - t_start) / 1000.0;
    const uint64_t n_out = n_written + n_ckpt_written + file_size_or_0(path(e.id, ".aux"));
    if (blocking_us == 0) {
        DSK_INF("saved %s: %zu tokens, %zu checkpoints + %zu deep, %.1f MiB state, %.1f MiB written (%.1f %%) in %.0f ms\n",
                e.id.c_str(), n, n_ckpts, e.deep.size(), n_logical / (1024.0 * 1024.0), n_out / (1024.0 * 1024.0),
                100.0 * n_out / std::max<double>(1.0, e.bytes), ms);
    } else {
        // [TAG_DISK_ASYNC_SAVE] blocking = the server loop's part (device copy + .tok / .aux / checkpoints)
        DSK_INF("saved %s: %zu tokens, %zu checkpoints + %zu deep, %.1f MiB state, %.1f MiB written (%.1f %%) in %.0f ms "
                "(blocking %.0f ms, async; committed after %.0f ms)\n",
                e.id.c_str(), n, n_ckpts, e.deep.size(), n_logical / (1024.0 * 1024.0), n_out / (1024.0 * 1024.0),
                100.0 * n_out / std::max<double>(1.0, e.bytes), (t_done - t_start) / 1000.0, blocking_us / 1000.0, ms);
    }

    const std::string id = e.id;
    bytes_written += n_written + n_ckpt_written + file_size_or_0(path(e.id, ".aux")) + file_size_or_0(path(e.id, ".tok"));
    // [TAG_DISK_ASYNC_SAVE] save_us = the server loop's part (all of a synchronous save)
    const uint64_t total_us = (uint64_t) (ggml_time_us() - t_start);
    save_us += blocking_us == 0 ? total_us : blocking_us;
    if (blocking_us == 0) { // (review r11 5) an async save's gauges are set by commit_job (its commit included)
        last_save_blocking_us = total_us;
        last_save_total_us    = total_us;
    }
    last_save_bytes       = e.bytes;
    last_save_written     = n_written + n_ckpt_written + file_size_or_0(path(e.id, ".aux")) + file_size_or_0(path(e.id, ".tok"));
    if (e.draft_cover < (int64_t) n) {
        n_save_draft_partial++;
        DSK_INF("saved %s draft-partial: draft_cover %" PRId64 " of %zu tokens (the draft lacks the first %" PRId64 " rows)\n",
                e.id.c_str(), e.draft_cover, n, (int64_t) n - e.draft_cover);
    }
    push_entry(std::move(e));
    n_saves++;
    if (blocking_us != 0) { // [TAG_DISK_ASYNC_SAVE]
        n_saves_async++;
        save_async_us += (uint64_t) std::max<int64_t>(0, t_done - t_captured);
    }

    enforce(id, blocking_us == 0); // (review r11 4) an async commit refreshes once, in commit_job, after its timing
    check_pins();

    return true;
}

// (review r11 6, r12 5) test hook check_pins: after every commit (both paths) the cached pin set must equal a fresh
//   derivation
void server_disk_cache::check_pins() {
    if (as && hooks.check_pins && pinned_set() != pinned_set_fresh()) {
        n_pin_check_failures++;
        DSK_WRN("%s", "check_pins: the cached pin set differs from a fresh derivation\n");
    }
}

//
// asynchronous save [TAG_DISK_ASYNC_SAVE] (--cache-disk-async N MiB, v65 64.1 = SC2)
//
// The server loop keeps only what needs the device: save() writes .tok / .aux / checkpoints as before, then streams the
// target / draft state through llama_state_seq_write_cb and copies every 4 MiB block off the device into a host
// buffer (ggml_backend_tensor_get, synchronous: the slot can be overwritten as soon as save() returns). The copied
// blocks go to worker threads that hash them (the content key), write the ones the own store does not hold
// (tmp + rename, as the synchronous path) and drop the rest; the last one writes the .tgt / .dft maps. poll() on the
// server thread commits finished jobs in submission order (save_commit: .meta, registration, the prefixes it extends,
// quota) - the same files, bytes and entry as the synchronous path. A job whose copy or write fails is committed as a
// failure (its files removed, the blocks only it wrote unlinked). Staging: copied blocks not yet hashed / written are
// held up to N MiB; beyond that the copy waits (blocking time, save_wait_us). In flight, a job's block keys are
// "pending" (blob_gc keeps them; the claim and the collection share one mutex) and its checkpoint keys are kept by
// deep_gc; a lookup that would find its conversation (await_pending) and a namespace switch / sleep / exit drain first.

static int file_present(const fs::path & p); // below, with blob_gc()

struct server_disk_save_job {
    struct blk {
        std::string key;   // set by a worker (an incremental reuse: set at the copy, no task)
        uint64_t    n     = 0;
        bool        fresh = false; // this job wrote the file
        std::string expect; // [TAG_DISK_INCR] incr_verify hook: the key the reuse would have taken (compared by the worker)
    };
    struct seg {
        blk *                blk_ref = nullptr; // a block, else inline bytes
        std::vector<uint8_t> inl;
    };
    struct stream {
        std::string      map_path;
        std::string      env_h;
        std::vector<seg> segs;
        uint64_t         n_stream = 0;
    };

    server_disk_cache_entry e;
    server_tokens           tokens;
    std::string             blk_dir;
    size_t                  n_ckpts        = 0;
    uint64_t                ckpt_bytes     = 0;
    uint64_t                n_ckpt_written = 0;
    uint64_t                n_logical      = 0;
    int64_t                 t_start        = 0;
    // own entries its commit replaces (id, own bytes): pinned while it is in flight, their own bytes netted out of its
    //   quota charge
    uint64_t                seq = 0; // submission number (test hooks fail_job / finalize_delay_jobK_ms)
    int64_t                 t_captured     = 0;
    int64_t                 t_done         = 0;
    uint64_t                blocking_us    = 0;
    int32_t                 id_slot        = -1;
    std::vector<stream>     streams;
    std::deque<blk>         blocks; // stable addresses: the server thread appends while workers fill earlier ones
    std::atomic<uint64_t>   n_written{0};
    std::atomic<bool>       ok{true};
    // under server_disk_async::mtx
    int  n_open   = 0;     // block tasks queued and not finished
    uint64_t inl_held = 0; // (review r10 1) staging held by its inline segments (capacity), released at its end
    bool captured = false; // every block queued (or the capture failed)
    bool done     = false; // maps written (or failed): poll() may commit
    std::atomic<bool> write_held{false}; // test hook hold_write_jobK_ms: the first block write was held
    // (review r13 1) server thread: a synchronous save that extends this one committed while it was in flight - it no
    //   longer pins anything and its commit drops it
    bool superseded = false;
    // server thread: save() returned false for it (the copy failed, logged there): its commit reports no loss again
    bool copy_failed = false;
    // [TAG_DISK_INCR] (server thread) the clock value ticked before this job's copy, and per stream (.tgt, .dft) its KV
    //   ranges (tensor, offset) -> (size, the blocks cut from it in order): the slot's record once it commits
    uint64_t incr_since = 0;
    std::vector<std::map<std::pair<const ggml_tensor *, size_t>, std::pair<uint64_t, std::vector<const blk *>>>> incr_ranges;
};

// [TAG_DISK_INCR] the last committed asynchronous save of a slot: what an incremental copy may take instead of the device
struct server_disk_incr_rec {
    uint64_t    since = 0;  // clock value ticked before that save's copy
    std::string blk_dir;    // its block store
    // per stream: KV range (tensor, offset) -> (range size, the keys of its FULL blocks in order)
    std::vector<std::map<std::pair<const ggml_tensor *, size_t>, std::pair<uint64_t, std::vector<std::string>>>> ranges;
};

struct server_disk_async {
    struct task {
        server_disk_save_job *      job = nullptr;
        server_disk_save_job::blk * b   = nullptr; // nullptr: finalize the job (maps)
        // the copied block (a BLOB_BYTES buffer from the pool, n_buf bytes used)
        std::unique_ptr<uint8_t[]>  buf;
        size_t                      n_buf = 0;
    };

    uint64_t budget   = 0;
    server_disk_cache::test_hooks_t hooks; // set_test_hooks() (delay_ms, fail_job, finalize delays)
    std::atomic<int> fail_maps{0}; // test hook fail_maps: map writes left to fail (counts down across workers)
    std::function<void()> on_job_done; // a job is done (worker thread): wakes the server loop

    std::mutex              mtx;
    std::condition_variable cv_task; // workers: a task or stop
    std::condition_variable cv_room; // the copy: staging room
    std::condition_variable cv_done; // drain: a job done; a claim: a block write finished
    std::deque<task>        tasks;
    uint64_t                held = 0;
    bool                    stop = false;
    std::vector<std::thread> threads;

    std::map<std::string, int> pending; // block keys referenced by jobs in flight (blob_gc keeps them)

    std::atomic<uint64_t> incr_verified{0}, incr_mismatch{0}; // [TAG_DISK_INCR] incr_verify hook (workers)

    // block buffers kept for reuse (under mtx): a fresh 4 MiB allocation is a new mapping whose page faults cost as
    //   much as the copy itself; at most POOL_MAX are kept (128 MiB resident between saves)
    static constexpr size_t POOL_MAX = 32;
    std::vector<std::unique_ptr<uint8_t[]>> pool;

    uint64_t n_submitted = 0;  // jobs submitted (server thread)
    uint64_t n_enqueued  = 0;  // block tasks enqueued (test hook fail_enqueue; under mtx)
    uint64_t n_finalize_enqueued = 0; // finalizer tasks queued at a capture's end (test hook fail_finalize; under mtx)

    std::set<std::string>      writing; // keys a worker is writing now

    std::deque<std::unique_ptr<server_disk_save_job>> jobs; // submission order; the deque itself: server thread only
};

static bool disk_async_write_map(const server_disk_save_job::stream & st, std::atomic<uint64_t> & n_written) {
    const std::string tmp = st.map_path + ".tmp";
    std::ofstream fm(tmp, std::ios::binary | std::ios::trunc);
    put_le<uint32_t>(fm, MAP_ENV_MAGIC);
    put_le<uint32_t>(fm, ENVELOPE_VERSION);
    put_str_le(fm, st.env_h);
    put_le<uint64_t>(fm, st.n_stream);
    for (const auto & sg : st.segs) {
        if (sg.blk_ref) {
            put<uint8_t>(fm, 1);
            put_str(fm, sg.blk_ref->key);
        } else {
            put<uint8_t>(fm, 0);
            put<uint64_t>(fm, sg.inl.size());
            fm.write(reinterpret_cast<const char *>(sg.inl.data()), sg.inl.size());
            put_str(fm, hash_key(sg.inl.data(), sg.inl.size()));
            n_written += sg.inl.size();
        }
    }
    put<uint8_t>(fm, 255);
    fm.close();
    std::error_code ec;
    bool ok = (bool) fm;
    if (ok) {
        fs::rename(tmp, st.map_path, ec);
        ok = !ec;
    }
    if (!ok) {
        fs::remove(tmp, ec);
    }
    return ok;
}

// marks the job done whatever happens (a map write that throws fails the job); never throws
static void disk_async_finalize(server_disk_async & as, server_disk_save_job & job) {
    { // test hooks finalize_delay_ms / finalize_delay_jobK_ms: a job's end lags its block tasks
        const auto it = as.hooks.finalize_delay_job.find((int) job.seq + 1);
        const int ms = it != as.hooks.finalize_delay_job.end() ? it->second : as.hooks.finalize_delay_ms;
        if (ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        }
    }
    try {
        if (job.ok && (int64_t) job.seq + 1 == as.hooks.fail_job) {
            DSK_WRN("%s", "asynchronous save: injected failure (test hook fail_job)\n");
            job.ok = false;
        }
        if (job.ok) {
            for (const auto & st : job.streams) {
                if (as.fail_maps.load() > 0 && as.fail_maps.fetch_sub(1) > 0) { // test hook fail_maps
                    throw std::runtime_error("injected map write failure (test hook fail_maps)");
                }
                if (!disk_async_write_map(st, job.n_written)) {
                    job.ok = false;
                    break;
                }
            }
        }
    } catch (const std::exception & err) {
        DSK_WRN("asynchronous save: map write failed: %s\n", err.what());
        job.ok = false;
    }
    for (auto & st : job.streams) { // the inline bytes are in the maps (or the job failed): their staging goes back
        for (auto & sg : st.segs) {
            std::vector<uint8_t>().swap(sg.inl);
        }
    }
    {
        std::lock_guard<std::mutex> lock(as.mtx);
        as.held -= job.inl_held;
        job.inl_held = 0;
        as.cv_room.notify_all();
        job.t_done = ggml_time_us();
        job.done   = true;
        as.cv_done.notify_all();
    }
    if (as.on_job_done) {
        as.on_job_done(); // (review C1) wake the server loop: an idle server commits now, not at the next request
    }
}

// the key a worker is writing: released on every path (review C3)
struct disk_async_writing_guard {
    server_disk_async & as;
    const std::string * key = nullptr;
    ~disk_async_writing_guard() {
        if (key) {
            std::lock_guard<std::mutex> lock(as.mtx);
            as.writing.erase(*key);
            as.cv_done.notify_all();
        }
    }
};

// hash / claim / write one copied block; the staging and job accounting is released exactly once, after the work, and
//   a failure only fails the job (review C2)
// (review r13 6) one block file write: a writer-unique temp file renamed into place (concurrent writers of the same
//   key write the same bytes; the rename is atomic); false and no temp left on an error
static bool disk_write_block_file(const fs::path & fin, const uint8_t * p, size_t n) {
    static std::atomic<uint64_t> tmp_seq{0};
    const std::string ftmp = fin.string() + ".tmp" + std::to_string(tmp_seq++);
    bool wok = false;
    {
        std::ofstream fb(ftmp, std::ios::binary | std::ios::trunc);
        fb.write(reinterpret_cast<const char *>(p), (std::streamsize) n);
        wok = (bool) fb;
    }
    std::error_code ec;
    if (wok) {
        fs::rename(ftmp, fin, ec);
        wok = !ec;
    }
    if (!wok) {
        fs::remove(ftmp, ec);
    }
    return wok;
}

// (review r13 6) a claim's release (as.mtx held): the key stops being pending with its last claim
static void disk_async_pending_release(server_disk_async & as, const std::string & key, int n) {
    auto it = as.pending.find(key);
    if (it != as.pending.end() && (it->second -= n) <= 0) {
        as.pending.erase(it);
    }
}

// (review r12 1) a worker's block store: under as.mtx wait while another writer holds THIS key (one block), claim it
//   (pending[key]++: blob_gc() keeps it), and if the own file is absent mark it being written; write it outside the
//   mutex; drop the mark. The commit releases the claim. fresh = this call wrote the file; false on a write error.
//   hold_ms: test hook hold_write_jobK_ms (held while marked)
static bool disk_async_store_block(server_disk_async & as, const fs::path & fin, const std::string & key,
                                   const uint8_t * p, size_t n, bool & fresh, int hold_ms) {
    fresh = false;
    disk_async_writing_guard guard{ as };
    bool present = false;
    {
        std::unique_lock<std::mutex> lock(as.mtx);
        as.cv_done.wait(lock, [&] { return !as.writing.count(key); });
        as.pending[key]++;
        // under the mutex blob_gc() holds while it collects: a file seen here stays until the claim is released
        present = file_present(fin) > 0;
        if (!present) {
            as.writing.insert(key);
            guard.key = &key;
        }
    }
    if (present) {
        return true;
    }
    if (hold_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(hold_ms));
    }
    if (!disk_write_block_file(fin, p, n)) {
        return false;
    }
    fresh = true;
    return true;
}

static void disk_async_block(server_disk_async & as, server_disk_async::task & t) {
    server_disk_save_job & job = *t.job;
    try {
        if (as.hooks.delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(as.hooks.delay_ms));
        }
        if (job.ok) {
            auto & b = *t.b;
            b.n   = t.n_buf;
            b.key = hash_key(t.buf.get(), t.n_buf);
            if (!b.expect.empty()) { // [TAG_DISK_INCR] incr_verify hook: the reuse would have produced another stream
                as.incr_verified++;
                if (b.expect != b.key) {
                    as.incr_mismatch++;
                    DSK_WRN("incremental save verify: a block the incremental copy would reuse changed on the device "
                            "(recorded %s, device %s) - the record missed a modification\n", b.expect.c_str(), b.key.c_str());
                }
            }
            const fs::path fin = fs::path(job.blk_dir) / (b.key + ".blk");
            bool fresh = false;
            int hold_ms = 0; // test hook hold_write_jobK_ms
            if (const auto ih = as.hooks.hold_write_job.find((int) job.seq + 1); ih != as.hooks.hold_write_job.end() &&
                !job.write_held.exchange(true)) {
                hold_ms = ih->second;
            }
            if (!disk_async_store_block(as, fin, b.key, t.buf.get(), t.n_buf, fresh, hold_ms)) {
                job.ok = false;
            } else if (fresh) {
                b.fresh = true;
                job.n_written += b.n;
            }
        }
    } catch (const std::exception & err) {
        DSK_WRN("asynchronous save: block failed: %s\n", err.what());
        job.ok = false;
    }
    bool fin = false;
    {
        std::lock_guard<std::mutex> lock(as.mtx);
        if (as.pool.size() < server_disk_async::POOL_MAX) {
            as.pool.push_back(std::move(t.buf)); // capacity reserved: no allocation
        }
        t.buf.reset();
        as.held -= BLOB_BYTES; // (review r10 1) charged at allocation capacity
        fin = --job.n_open == 0 && job.captured;
        as.cv_room.notify_all();
    }
    if (fin) {
        disk_async_finalize(as, job);
    }
}

static void disk_async_worker(server_disk_async * as) {
    for (;;) {
        server_disk_async::task t;
        {
            std::unique_lock<std::mutex> lock(as->mtx);
            as->cv_task.wait(lock, [&] { return as->stop || !as->tasks.empty(); });
            if (as->tasks.empty()) {
                return; // stopping, nothing left
            }
            t = std::move(as->tasks.front());
            as->tasks.pop_front();
        }
        // both handlers release their accounting themselves and do not throw
        if (t.b == nullptr) {
            disk_async_finalize(*as, *t.job);
        } else {
            disk_async_block(*as, t);
        }
    }
}

server_disk_cache::~server_disk_cache() {
    if (!as) {
        return;
    }
    // the server drains (commits) in save_all_slots(); here the workers only finish their tasks: a job not committed
    //   by now has no .meta and is dropped as incomplete at the next start (live_refs may point into a gone server)
    {
        std::lock_guard<std::mutex> lock(as->mtx);
        as->stop = true;
    }
    as->cv_task.notify_all();
    for (auto & t : as->threads) {
        t.join();
    }
}

void server_disk_cache::set_async(uint64_t budget_bytes) {
    if (budget_bytes == 0) {
        as.reset();
        DSK_INF("%s", "synchronous save (--cache-disk-async 0)\n");
        return;
    }
    as = std::make_unique<server_disk_async>();
    as->budget = budget_bytes;
    if (incr) {
        DSK_INF("incremental save (--cache-disk-incr): KV blocks unmodified since the slot's last save are not copied%s\n",
                hooks.incr_verify ? " (incr_verify: copied and compared)" : "");
    }
    as->hooks     = hooks; // test hooks (set_test_hooks), copied once
    as->fail_maps = hooks.fail_maps;
    as->pool.reserve(server_disk_async::POOL_MAX);
    as->on_job_done = [this]() {
        if (on_async_done) {
            on_async_done();
        }
    };
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int n_threads = (int) std::min(4u, std::max(1u, hw / 4));
    if (hooks.threads > 0) { // test hook
        n_threads = hooks.threads;
    }
    for (int i = 0; i < n_threads; i++) {
        as->threads.emplace_back(disk_async_worker, as.get());
    }
    DSK_INF("asynchronous save: %d worker threads, %.0f MiB staging%s\n", n_threads, budget_bytes / (1024.0 * 1024.0),
            as->hooks.delay_ms > 0 ? string_format(" (test delay %d ms per block)", as->hooks.delay_ms).c_str() : "");
}

size_t server_disk_cache::n_pending() const {
    return as ? as->jobs.size() : 0;
}

bool server_disk_cache::pending_holds(const server_tokens & tokens) const {
    if (!as) {
        return false;
    }
    for (const auto & j : as->jobs) {
        // (review r14 3) a superseded save is dropped at its commit: it holds nothing
        if (!j->superseded && j->tokens.size() == tokens.size() && tokens_prefix_of(tokens, j->tokens, false)) {
            return true;
        }
    }
    return false;
}

// (review r3 I5, r4 6) every commit entry point reports the background saves that failed into the caller's sink (one
//   record per job: its entry id, token count and the slot it was copied from)

void server_disk_cache::poll_into(std::vector<server_disk_lost> & lost) {
    if (!as) {
        return;
    }
    if (&lost != &deferred_lost && !deferred_lost.empty()) { // committed at the top of a save() (review r12 3)
        lost.insert(lost.end(), deferred_lost.begin(), deferred_lost.end());
        deferred_lost.clear();
    }
    while (!as->jobs.empty()) {
        {
            std::lock_guard<std::mutex> lock(as->mtx);
            if (!as->jobs.front()->done) {
                return;
            }
        }
        std::unique_ptr<server_disk_save_job> job = std::move(as->jobs.front());
        jobs_pop_front();
        commit_job(*job, lost);
    }
}

std::vector<server_disk_lost> server_disk_cache::poll() {
    std::vector<server_disk_lost> lost;
    poll_into(lost);
    return lost;
}

std::vector<server_disk_lost> server_disk_cache::drain() {
    std::vector<server_disk_lost> lost;
    if (!as) {
        return lost;
    }
    if (!as->jobs.empty()) {
        const int64_t t0 = ggml_time_us();
        const size_t  n  = as->jobs.size();
        {
            std::unique_lock<std::mutex> lock(as->mtx);
            as->cv_done.wait(lock, [&] {
                return std::all_of(as->jobs.begin(), as->jobs.end(), [](const auto & j) { return j->done; });
            });
        }
        DSK_INF("asynchronous save: waited %.0f ms for %zu save(s) in flight\n", (ggml_time_us() - t0) / 1000.0, n);
    }
    poll_into(lost);
    return lost;
}

std::vector<server_disk_lost> server_disk_cache::await_pending(const server_tokens & tokens, size_t lcp_min) {
    std::vector<server_disk_lost> lost;
    if (!as) {
        return lost;
    }
    poll_into(lost);
    // (review I5) wait for the newest matching job and the ones before it (commits go in order); later jobs go on
    size_t n_wait = 0;
    for (size_t i = 0; i < as->jobs.size(); i++) {
        // (review r14 2) a superseded save commits nothing a lookup could find (its extension is registered)
        if (!as->jobs[i]->superseded && as->jobs[i]->tokens.get_common_prefix(tokens) > lcp_min) {
            n_wait = i + 1;
        }
    }
    if (n_wait > 0) {
        DSK_INF("a save of this conversation (%zu tokens) is in flight: waiting for it\n",
                as->jobs[n_wait - 1]->tokens.size());
        {
            std::unique_lock<std::mutex> lock(as->mtx);
            as->cv_done.wait(lock, [&] {
                return std::all_of(as->jobs.begin(), as->jobs.begin() + n_wait, [](const auto & j) { return j->done; });
            });
        }
        poll_into(lost);
    }
    return lost;
}

// [TAG_DISK_ASYNC_SAVE] (review r11 6) every change of the entries or of the jobs in flight goes through these: the
//   derived pin set (pinned_set()) is cached per generation
void server_disk_cache::push_entry(server_disk_cache_entry && e) {
    entries.push_back(std::move(e));
    ++gen;
}

void server_disk_cache::erase_entry_at(size_t idx) {
    entries.erase(entries.begin() + idx);
    ++gen;
}

void server_disk_cache::clear_entries() {
    entries.clear();
    ++gen;
}

void server_disk_cache::jobs_push(std::unique_ptr<server_disk_save_job> && job) {
    as->jobs.push_back(std::move(job));
    ++gen;
}

void server_disk_cache::jobs_pop_front() {
    as->jobs.pop_front();
    ++gen;
}

// [TAG_DISK_ASYNC_SAVE] the in-flight model (design after review r9):
//   pinned  = an own entry whose tokens are a proper prefix of a save in flight (that save replaces it at its commit),
//             derived from the jobs and the entries whenever needed - no per-job bookkeeping. A pinned entry is neither
//             evicted, TTL-expired nor replaced by another save's commit until the save that extends it commits or fails
//   pending = the bytes the saves in flight wrote so far, minus the OWN files (.tok / .aux / maps) of the pinned entries
//             (each once): their blocks stay referenced by the continuation, so their own files are what the
//             replacement frees (their unshared checkpoints are not netted: errs toward evicting earlier), at least 0
//   charge  = min(pending, the bytes the victim search could free) - what enforce() / deep_put() add to the disk total
std::set<std::string> server_disk_cache::pinned_set_fresh() const {
    std::set<std::string> out;
    if (!as) {
        return out;
    }
    for (const auto & e : entries) {
        if (!e.src.empty()) {
            continue;
        }
        for (const auto & j : as->jobs) {
            if (!j->superseded && tokens_prefix_of(e.tokens, j->tokens, true)) {
                out.insert(e.id);
                break;
            }
        }
    }
    return out;
}

// (review r10 5, r11 8) cached per generation (push_entry / erase_entry_at / clear_entries / jobs_push / jobs_pop_front)
const std::set<std::string> & server_disk_cache::pinned_set() const {
    if (pin_gen != gen) {
        pin_cache = pinned_set_fresh();
        pin_gen   = gen;
    }
    return pin_cache;
}

// (review r13 1) a committed synchronous save that strictly extends saves in flight supersedes them (server thread)
void server_disk_cache::mark_superseded(const server_tokens & tokens) {
    if (!as) {
        return;
    }
    for (auto & j : as->jobs) {
        if (!j->superseded && tokens_prefix_of(j->tokens, tokens, true)) {
            j->superseded = true;
            ++gen; // its pins go
            DSK_INF("asynchronous save of %s (%zu tokens) superseded by a synchronous save of %zu tokens\n",
                    j->e.id.c_str(), j->tokens.size(), tokens.size());
        }
    }
}

uint64_t server_disk_cache::pending_bytes(const std::set<std::string> & pinned) const {
    if (!as) {
        return 0;
    }
    uint64_t written = 0, freed = 0;
    for (const auto & j : as->jobs) {
        written += j->n_written.load();
    }
    for (const auto & e : entries) {
        if (pinned.count(e.id)) {
            freed += e.own_bytes;
        }
    }
    return written > freed ? written - freed : 0;
}

uint64_t server_disk_cache::pending_bytes() const {
    return pending_bytes(pinned_set());
}

// (review r9 1) the charge is capped by what eviction can free (the logical size of the own entries that are neither
//   pinned nor protected), so the loop's target stays reachable and the behaviour is continuous: a save larger than
//   that cannot empty the cache before it commits (its commit's own enforce() charges it exactly). Warned once per
//   episode (when the cap starts to bind)
uint64_t server_disk_cache::inflight_charge(const std::set<std::string> & pinned, const std::string & protect) {
    const uint64_t pending = pending_bytes(pinned);
    uint64_t evictable = 0;
    for (const auto & e : entries) {
        if (e.src.empty() && e.id != protect && e.id != protect_id && !pinned.count(e.id)) {
            evictable += e.bytes;
        }
    }
    const bool capped = pending > evictable;
    if (capped && !charge_capped) {
        DSK_WRN("saves in flight (%.1f MiB) exceed what eviction can free (%.1f MiB): charged %.1f MiB until they commit\n",
                pending / (1024.0 * 1024.0), evictable / (1024.0 * 1024.0), evictable / (1024.0 * 1024.0));
    }
    charge_capped = capped;
    return capped ? evictable : pending;
}

bool server_disk_cache::save_async_begin(llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
                                         server_disk_cache_entry & e, const server_prompt & prompt, size_t n_ckpts,
                                         uint64_t ckpt_bytes, uint64_t n_ckpt_written, int64_t t_start) {
    uint64_t copy_us = 0, wait_us = 0;
    uint64_t job_skipped = 0; // [TAG_DISK_INCR] bytes this save took from the slot's last save instead of the device

    auto job_up = std::make_unique<server_disk_save_job>();
    server_disk_save_job & job = *job_up;
    job.e              = std::move(e);
    job.tokens         = prompt.tokens.clone();
    job.blk_dir        = (fs::path(dir) / "blk").string();
    job.n_ckpts        = n_ckpts;
    job.ckpt_bytes     = ckpt_bytes;
    job.n_ckpt_written = n_ckpt_written;
    job.t_start        = t_start;
    job.id_slot        = id_slot;
    job.seq = as->n_submitted++; // (the entries it pins are derived from its tokens: pinned_set())
    jobs_push(std::move(job_up));

    // (review r10 1) every staged byte counts against the budget at allocation capacity: a block buffer as BLOB_BYTES,
    //   the inline segments (cell metadata, small tensor ranges, e.g. a transposed V cache) by the capacity they grow
    //   to, held until the job's end. Admission on that capacity; a single save larger than the budget proceeds once
    //   nothing else is staged (its own inline bytes excepted)
    const auto reserve = [&](uint64_t cap) {
        const int64_t t_wait = ggml_time_us();
        std::unique_lock<std::mutex> lock(as->mtx);
        as->cv_room.wait(lock, [&] { return as->held <= job.inl_held || as->held + cap <= as->budget; });
        as->held += cap;
        wait_us += (uint64_t) (ggml_time_us() - t_wait);
    };
    const auto unreserve = [&](uint64_t cap) {
        std::lock_guard<std::mutex> lock(as->mtx);
        as->held -= cap;
        as->cv_room.notify_all();
    };

    const auto capture = [&](llama_context * ctx, const std::string & map_path, const std::string & env_h) {
        job.streams.emplace_back();
        auto & st = job.streams.back();
        st.map_path = map_path;
        st.env_h    = env_h;
        const uint64_t n_logical0 = job.n_logical;
        std::vector<uint8_t> inl;
        uint64_t inl_reserved = 0; // (review r12 7) staging reserved for the local buffer (not yet moved into a segment)
        const auto grow_inline = [&](size_t add) { // reserve staging before the inline buffer grows
            if (inl.size() + add <= inl.capacity()) {
                return;
            }
            const size_t cap = std::max({ inl.size() + add, 2 * inl.capacity(), (size_t) (64u << 10) });
            const uint64_t delta = cap - inl.capacity();
            reserve(delta);
            try {
                inl.reserve(cap);
            } catch (...) {
                unreserve(delta);
                throw;
            }
            std::lock_guard<std::mutex> lock(as->mtx);
            job.inl_held += delta;
            inl_reserved += delta;
        };
        const auto flush_inline = [&]() {
            if (!inl.empty()) {
                st.segs.emplace_back();
                st.segs.back().inl = std::move(inl); // keeps its capacity: still counted in job.inl_held
                inl = std::vector<uint8_t>();
                inl_reserved = 0; // released with the job's segments at its end
            }
        };
        llama_state_seq_writer w;
        w.write = [&](const void * src, size_t size) {
            grow_inline(size);
            const auto * p = static_cast<const uint8_t *>(src);
            inl.insert(inl.end(), p, p + size);
            job.n_logical += size;
        };
        // [TAG_DISK_INCR] the blocks of a large range from byte `from` (relative, a multiple of BLOB_BYTES) on; the
        //   blocks before it were appended by the caller. expect (incr_verify hook): the recorded keys of the blocks the
        //   reuse skipped, compared by the workers. Returns the blocks it appended (for the slot's next record)
        const auto copy_blocks = [&](ggml_tensor * tensor, size_t offset, size_t size, size_t from,
                                     const std::vector<std::string> * expect, std::vector<const server_disk_save_job::blk *> * out) {
            size_t pos = offset + from;
            const size_t end = offset + size;
            while (pos < end) {
                // [TAG_DISK_DELTA_REL] blocks are cut at fixed offsets from the START OF THE RANGE (v63: of the tensor),
                //   here and in the synchronous fallback beside it (async 0 keeps the v68 tensor-aligned cut): a
                //   sequence's rows are one contiguous cell range in position order, so the same prefix gives the
                //   same blocks wherever its cells sit - with several slots in a unified pool (or a load placed at
                //   another free range) the tensor-aligned cut shifted every block and a re-save rewrote the whole
                //   state. For a range at cell 0 (one slot) both cuts are the same. Only the cut
                //   moves: the stream (the concatenation) is unchanged, the loader is cut-agnostic
                const size_t blk_end = std::min(end, offset + ((pos - offset) / BLOB_BYTES + 1) * BLOB_BYTES);
                const size_t n = blk_end - pos;
                server_disk_async::task t;
                t.job = &job;
                reserve(BLOB_BYTES);
                try {
                    {
                        std::lock_guard<std::mutex> lock(as->mtx);
                        if (!as->pool.empty()) {
                            t.buf = std::move(as->pool.back());
                            as->pool.pop_back();
                        }
                    }
                    if (!t.buf) {
                        t.buf.reset(new uint8_t[BLOB_BYTES]); // n <= BLOB_BYTES: one block of a range
                    }
                    t.n_buf = n;
                    const int64_t t_copy = ggml_time_us();
                    ggml_backend_tensor_get(tensor, t.buf.get(), pos, n);
                    copy_us += (uint64_t) (ggml_time_us() - t_copy);
                    job.blocks.emplace_back();
                    t.b = &job.blocks.back();
                    if (expect) { // [TAG_DISK_INCR]
                        const size_t kb = (pos - offset) / BLOB_BYTES;
                        if (kb < expect->size()) {
                            t.b->expect = (*expect)[kb];
                        }
                    }
                    if (out) {
                        out->push_back(t.b);
                    }
                    st.segs.emplace_back();
                    st.segs.back().blk_ref = t.b;
                    job.e.blobs.push_back(""); // filled at commit (key order = map order)
                    // (review r10 2) queue insertion and n_open move together under the mutex: an allocation failure
                    //   leaves no phantom task (the staging goes back below, the job fails, commits / drain go on)
                    std::lock_guard<std::mutex> lock(as->mtx);
                    if (as->hooks.fail_enqueue > 0 && (int64_t) ++as->n_enqueued == as->hooks.fail_enqueue) {
                        throw std::bad_alloc(); // test hook fail_enqueue
                    }
                    as->tasks.push_back(std::move(t));
                    job.n_open++;
                } catch (...) {
                    unreserve(BLOB_BYTES);
                    throw;
                }
                as->cv_task.notify_one();
                pos = blk_end;
            }
        };
        const auto copy_inline = [&](ggml_tensor * tensor, size_t offset, size_t size) {
            grow_inline(size);
            const size_t at = inl.size();
            inl.resize(at + size);
            const int64_t t_copy = ggml_time_us();
            ggml_backend_tensor_get(tensor, inl.data() + at, offset, size);
            copy_us += (uint64_t) (ggml_time_us() - t_copy);
            job.n_logical += size;
        };
        w.write_tensor = [&](ggml_tensor * tensor, size_t offset, size_t size) {
            if (size < (64u << 10)) { // inline, as save_stream
                copy_inline(tensor, offset, size);
                return;
            }
            flush_inline();
            job.n_logical += size;
            copy_blocks(tensor, offset, size, 0, nullptr, nullptr);
        };
        // [TAG_DISK_INCR] --cache-disk-incr: a KV range's leading full blocks whose cells were not modified since the
        //   slot's last committed save of the same range (same tensor, same offset = same cut) take that save's keys
        //   (claimed like a worker's present block: the file is in the own store and stays while this job is in
        //   flight); the device copy starts at the first block that is new, partial or modified. Same bytes, same
        //   blocks, same map as the full copy
        const size_t si = job.streams.size() - 1;
        const server_disk_incr_rec * rec = nullptr;
        if (incr) {
            const auto it = incr_recs.find(id_slot);
            if (it != incr_recs.end() && it->second && it->second->blk_dir == job.blk_dir && si < it->second->ranges.size()) {
                rec = it->second.get();
            }
            w.since = rec ? rec->since : 0;
            job.incr_ranges.resize(si + 1);
            w.write_tensor_kv = [&](ggml_tensor * tensor, size_t offset, size_t size, size_t clean) {
                if (size < (64u << 10)) {
                    copy_inline(tensor, offset, size);
                    return;
                }
                flush_inline();
                job.n_logical += size;
                auto & mine = job.incr_ranges[si][{ tensor, offset }];
                mine.first = size;
                const std::vector<std::string> * keys = nullptr;
                size_t n_reuse = 0;
                if (rec) {
                    const auto ir = rec->ranges[si].find({ tensor, offset });
                    if (ir != rec->ranges[si].end()) {
                        keys = &ir->second.second;
                        const uint64_t lim = std::min<uint64_t>({ (uint64_t) clean, ir->second.first, (uint64_t) size });
                        n_reuse = std::min<size_t>((size_t) (lim / BLOB_BYTES), keys->size());
                    }
                }
                if (n_reuse > 0 && hooks.incr_verify) { // copy everything, the workers compare the reusable blocks' keys
                    incr_blocks_verify_planned += n_reuse;
                    const std::vector<std::string> expect(keys->begin(), keys->begin() + n_reuse);
                    copy_blocks(tensor, offset, size, 0, &expect, &mine.second);
                    return;
                }
                size_t k = 0;
                for (; k < n_reuse; ++k) {
                    const std::string & key = (*keys)[k];
                    {
                        std::lock_guard<std::mutex> lock(as->mtx);
                        // a key being written by another job, or gone from the own store: copy from here on
                        if (as->writing.count(key) || file_present(fs::path(job.blk_dir) / (key + ".blk")) <= 0) {
                            break;
                        }
                        as->pending[key]++; // released at the commit with the job's other keys
                    }
                    if (hooks.fail_incr_alloc > 0 && (int64_t) ++n_incr_alloc_hook == hooks.fail_incr_alloc) {
                        throw std::bad_alloc(); // test hook fail_incr_alloc: the block record's allocation
                    }
                    job.blocks.emplace_back();
                    auto & b = job.blocks.back();
                    b.key = key;
                    b.n   = BLOB_BYTES;
                    mine.second.push_back(&b);
                    st.segs.emplace_back();
                    st.segs.back().blk_ref = &b;
                    job.e.blobs.push_back("");
                }
                incr_blocks_reused += k;
                incr_bytes_skipped += (uint64_t) k * BLOB_BYTES;
                job_skipped        += (uint64_t) k * BLOB_BYTES;
                copy_blocks(tensor, offset, size, k * BLOB_BYTES, nullptr, &mine.second);
            };
        }
        try {
            st.n_stream = llama_state_seq_write_cb(ctx, id_slot, 0, w);
            flush_inline();
        } catch (...) {
            // (review r11 7) the local inline buffer dies here: its staging share goes back now, not at the job's end
            const uint64_t cap = inl_reserved; // exactly what was reserved for it (reserve() may allocate more)
            {
                std::lock_guard<std::mutex> lock(as->mtx);
                job.inl_held -= cap;
            }
            unreserve(cap);
            throw;
        }
        return st.n_stream > 0 && st.n_stream == job.n_logical - n_logical0;
    };

    bool ok = false;
    if (incr) {
        job.incr_since = llama_state_mod_clock_tick(); // [TAG_DISK_INCR] cells modified from here on are newer than this copy
    }
    try {
        ok = capture(ctx_tgt, path(job.e.id, ".tgt"), desc_tgt_full);
        if (ok && ctx_dft) {
            ok = capture(ctx_dft, path(job.e.id, ".dft"), desc_dft_full);
        }
    } catch (const std::exception & err) {
        DSK_WRN("cannot copy the state: %s\n", err.what());
        ok = false;
    }
    if (!ok) {
        job.ok = false;
    }

    job.t_captured = ggml_time_us();
    save_copy_us += copy_us;
    save_wait_us += wait_us;
    bytes_copied += job.n_logical;
    bool fin_here = false; // (pkg9-fix 4) the finalizer task could not be queued: the job ends on this thread
    {
        std::lock_guard<std::mutex> lock(as->mtx);
        job.captured = true;
        if (job.n_open == 0) {
            // (pkg9-fix 4) an allocation failure of the finalizer task fails the job, which then ends here without
            //   another allocation: nothing waits for a job that no task would finish (drain / exit cannot hang)
            try {
                if (as->hooks.fail_finalize > 0 && (int64_t) ++as->n_finalize_enqueued == as->hooks.fail_finalize) {
                    throw std::bad_alloc(); // test hook fail_finalize
                }
                as->tasks.push_back({ &job, nullptr, {} });
                as->cv_task.notify_one();
            } catch (...) {
                job.ok   = false;
                fin_here = true;
            }
        }
    }
    if (fin_here) {
        DSK_WRN("%s", "cannot queue the end of an asynchronous save: the save fails\n");
        ok = false;
        disk_async_finalize(*as, job); // !job.ok: only releases its staging and marks it done; never throws
    }
    job.copy_failed = !ok;
    DSK_INF("saving %s: %zu tokens, %.1f MiB copied off the device in %.0f ms (copy %.0f ms, staging wait "
            "%.0f ms), hashing / writing in the background%s%s\n", job.e.id.c_str(), job.tokens.size(),
            (job.n_logical - job_skipped) / (1024.0 * 1024.0), (job.t_captured - t_start) / 1000.0, copy_us / 1000.0,
            wait_us / 1000.0, incr ? string_format(" (incremental: %.1f MiB of %.1f MiB taken from the slot's last save)",
            job_skipped / (1024.0 * 1024.0), job.n_logical / (1024.0 * 1024.0)).c_str() : "",
            ok ? "" : " - the copy FAILED, the entry is dropped");
    refresh_status(); // (review r4 5) the counters save() bumped, the copy timings and the save in flight
    // (review r10 3, r11 3) the blocking part ends here: everything save() did on the server loop (the copy with its
    //   staging wait - the loop is stalled then too - and the refresh); no wait for other saves happens on this path
    job.blocking_us = (uint64_t) std::max<int64_t>(1, ggml_time_us() - t_start);
    return ok;
}

void server_disk_cache::commit_job(server_disk_save_job & job, std::vector<server_disk_lost> & lost) {
    const int64_t t_commit = ggml_time_us(); // (review r10 3) the whole commit is server-loop time
    bool ok = job.ok;
    // keys this commit adds to the index / the own inventory: taken back if save_commit fails (review r2 C2)
    std::set<std::string> added_blob, added_own;
    if (ok) {
        // the keys in map order, and the blocks this job wrote (or found in the own store) indexed as own - before
        //   save_commit: its own-store copy loop must see them as own
        std::lock_guard<std::mutex> lock(as->mtx);
        GGML_ASSERT(job.e.blobs.size() == job.blocks.size());
        size_t i = 0;
        for (const auto & b : job.blocks) {
            job.e.blobs[i++] = b.key;
            if (!blobs.count(b.key)) {
                blobs[b.key] = b.n;
                blob_bytes  += b.n;
                added_blob.insert(b.key);
            }
            if (blk_own.insert(b.key).second) { // the own file exists: written by this job or seen at its claim
                added_own.insert(b.key);
            }
        }
    }
    const uint64_t n_written = job.n_written.load();
    const std::string id     = job.e.id; // save_commit moves the entry
    // (review r13 1, r14 1) superseded (marked by the synchronous save that extends it): dropped only while an own
    //   registered entry still holds its tokens (extends or equals them - that extension may have been evicted,
    //   dropped after a failed load or cleared since; then the save commits normally). A dropped save: its own files
    //   go (its physical writes are counted), its claims are released as for a success (the rest is collected), no
    //   timing gauge; (review r14 4) a failed one is not a lost conversation either - logged with its failure, counted
    //   as superseded, not as failed
    bool superseded = false;
    if (job.superseded) {
        for (const auto & en : entries) {
            if (en.src.empty() && tokens_prefix_of(job.tokens, en.tokens, false)) {
                superseded = true;
                break;
            }
        }
        if (!superseded) {
            DSK_INF("asynchronous save of %s (%zu tokens): the save that superseded it is no longer registered, committing it\n",
                    id.c_str(), job.tokens.size());
        }
    }
    if (superseded) {
        bytes_written += n_written + job.n_ckpt_written + file_size_or_0(path(id, ".aux")) + file_size_or_0(path(id, ".tok"));
        remove_files(id);
        n_saves_superseded++;
        DSK_INF("asynchronous save of %s (%zu tokens) dropped at its commit: superseded by a registered extension%s\n",
                id.c_str(), job.tokens.size(), ok ? "" : " (its background write had failed; the conversation is on disk in the extension)");
    } else {
        // (review I7) save_commit counts saves_async / saves_async_failed by its own result
        ok = save_commit(job.e, ok, job.tokens, job.n_ckpts, job.ckpt_bytes, job.n_logical, n_written,
                         job.n_ckpt_written, job.t_start, job.blocking_us, job.t_done, job.t_captured);
    }
    if (!ok && !superseded && !job.copy_failed) {
        // (review I9, r9 4) the save reported "on disk" when it was queued: the caller reports the loss (once, with when);
        //   (pkg9-fix 4) a failed copy was reported by save_async_begin already (save() returned false)
        lost.push_back({ job.id_slot, id, job.tokens.size() });
    }
    if (incr) { // [TAG_DISK_INCR] the slot's record: this save's full blocks, if it committed (its keys stay claimed by its entry)
        if (ok && !superseded && job.incr_since != 0 && !job.incr_ranges.empty()) {
            if (hooks.fail_incr_rec > 0 && (int64_t) ++n_incr_rec_hook == hooks.fail_incr_rec) {
                throw std::bad_alloc(); // test hook fail_incr_rec: the record's allocation
            }
            auto rec = std::make_shared<server_disk_incr_rec>();
            rec->since   = job.incr_since;
            rec->blk_dir = job.blk_dir;
            rec->ranges.resize(job.incr_ranges.size());
            for (size_t si = 0; si < job.incr_ranges.size(); ++si) {
                for (const auto & [rk, rv] : job.incr_ranges[si]) {
                    auto & out = rec->ranges[si][rk];
                    out.first = rv.first;
                    for (const auto * b : rv.second) {
                        if (b->n != BLOB_BYTES || b->key.empty()) {
                            break; // the full blocks only (a range's last block may be partial)
                        }
                        out.second.push_back(b->key);
                    }
                }
            }
            incr_recs[job.id_slot] = std::move(rec);
        } else if (!superseded) {
            incr_recs.erase(job.id_slot);
        }
    }
    // (review I8) per key: this job's references, and whether it wrote the file; a failed job's fresh file goes when
    //   no other job in flight and no entry references it
    std::map<std::string, std::pair<int, bool>> refs;
    for (const auto & b : job.blocks) {
        if (!b.key.empty()) {
            auto & r = refs[b.key];
            r.first++;
            r.second = r.second || b.fresh;
        }
    }
    // (review r6 1, r7 1) this job's keys an OWN registered entry references (an adopted entry reads its blocks from
    //   its source store, never from the own dir unless blk_own lists the key - so its references do not keep an own copy)
    std::set<std::string> entry_refs;
    if (!ok) {
        for (const auto & e : entries) {
            if (!e.src.empty()) {
                continue;
            }
            for (const auto & k : e.blobs) {
                if (refs.count(k)) {
                    entry_refs.insert(k);
                }
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(as->mtx);
        for (const auto & [key, r] : refs) {
            auto it = as->pending.find(key);
            if (it == as->pending.end()) {
                continue;
            }
            if (!ok && it->second == r.first) {
                // (review r2 C2) the records this failed commit added go again (no other job in flight claims the key)
                if (added_own.count(key)) {
                    blk_own.erase(key);
                }
                const auto ib = blobs.find(key);
                if (added_blob.count(key) && ib != blobs.end()) {
                    blob_bytes -= ib->second;
                    blobs.erase(ib);
                }
                // (review r3 I4, r6 1) the own-store file goes whenever this job is its last claimant, it is not (or no
                //   longer) inventoried as own and no registered entry references it - whoever wrote it (a block
                //   written by an earlier failed job that this one had claimed would otherwise stay on disk unindexed,
                //   uncharged and invisible to blob_gc() until a restart). A legacy / adopt record of the key stays (that
                //   copy is the read location again); an own record not added here stays for blob_gc()
                if (!blk_own.count(key) && !entry_refs.count(key)) {
                    std::error_code ec;
                    fs::remove(fs::path(job.blk_dir) / (key + ".blk"), ec);
                }
            }
            disk_async_pending_release(*as, key, r.first);
        }
    }
    if (!ok || superseded) {
        // [TAG_DISK_GC] (review r3 I3) the aborted / superseded save's checkpoints / blocks, collected after its claims went
        deep_gc();
        blob_gc();
    }
    // (review r10 3) timings as implemented: the commit (index insertion, save_commit, claim cleanup / GC, this refresh
    //   excepted) is server-loop time - save_us and the last save's blocking part = its capture + its commit; its total
    //   runs from save() to the end of the commit (the FIFO wait for earlier saves included)
    const uint64_t commit_us = (uint64_t) (ggml_time_us() - t_commit);
    save_us += commit_us;
    if (!superseded) {
        last_save_commit_us = commit_us;
    }
    if (ok && !superseded) {
        last_save_blocking_us = job.blocking_us + commit_us;
        last_save_total_us    = (uint64_t) std::max<int64_t>(0, ggml_time_us() - job.t_start);
        DSK_DBG("committed %s in %.1f ms\n", id.c_str(), commit_us / 1000.0);
    }
    check_pins(); // (review r13 3) every path: failed, superseded, committed
    refresh_status(); // the snapshot carries the commit and its timing
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

// [TAG_ADOPT_TARGET_ONLY] the envelope of a stream map: 1 = read (ver, h set), 0 = no envelope (legacy map) or not
//   readable - the strict path decides then
static int read_stream_env(const std::string & map_path, uint32_t & ver, std::string & h) {
    std::ifstream fm(map_path, std::ios::binary);
    uint32_t magic = 0;
    if (!fm.is_open() || !get_le(fm, magic) || magic != MAP_ENV_MAGIC || !get_le(fm, ver) || !get_str_le(fm, h)) {
        return 0;
    }
    return 1;
}

bool server_disk_cache::draft_kv_src(const server_disk_cache_entry & e) const {
    return !e.src.empty() && std::any_of(adopted.begin(), adopted.end(),
                                         [&](const adopt_src & a) { return a.key == e.src && a.draft_kv_differs; });
}

bool server_disk_cache::target_only(const server_disk_cache_entry & e) const {
    if (e.src.empty() || !e.has_dft) {
        return false; // own namespace: both streams must match (strict)
    }
    // [TAG_CACHE_COMPAT2] each stream envelope is judged by its own rule (v2 full-coverage hash; v1 only for an adopted
    //   format-7 entry); the writer's combined descriptor_full names its draft too, so it is not compared here: the
    //   target stream's own full-coverage hash binds the target
    uint32_t tv = 0, dv = 0;
    std::string th, dh;
    if (!read_stream_env(path(e.id, ".tgt"), tv, th) || !envelope_ok(rule_for(e, true, true), tv, th)) {
        return false;
    }
    if (!read_stream_env(path(e.id, ".dft"), dv, dh)) {
        return false;
    }
    return !envelope_ok(rule_for(e, false, true), dv, dh);
}

server_disk_load_result server_disk_cache::verify_stream(const std::string & map_path, const stream_rule & rule) {
    const bool legacy_ok = rule.legacy_ok;
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
        if (!get_le(fm, ver) || !get_str_le(fm, h) || !envelope_ok(rule, ver, h) || !get_le(fm, n_total) || n_total == 0) {
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
    const bool tgt_only = target_only(e); // [TAG_ADOPT_TARGET_ONLY] what load() reads: .aux + .tgt
    if (!tgt_only && e.has_dft && draft_kv_src(e)) {
        res = SERVER_DISK_LOAD_INVALID; // [TAG_ADOPT_TARGET_ONLY] its draft stream never loads (another draft KV type)
    }
    if (res == SERVER_DISK_LOAD_OK) {
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
        for (size_t i = 0; !tgt_only && res == SERVER_DISK_LOAD_OK && i < keys.size(); i++) {
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
        res = verify_stream(path(id, ".tgt"), rule_for(e, true, tgt_only));
    }
    if (res == SERVER_DISK_LOAD_OK && e.has_dft && !tgt_only) {
        res = verify_stream(path(id, ".dft"), rule_for(e, false));
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
    // [TAG_ADOPT_TARGET_ONLY] an adopt source's entry whose target stream matches this target while its draft stream
    //   (format / descriptor) does not: the target stream loads, the draft sequence and the carry are reset (series C,
    //   [TAG_SPEC_RESET]); the prompt is draft-partial ([TAG_DRAFT_PARTIAL]), not partial: its target state is canonical.
    //   The prompt checkpoints and deep refs (their envelope names the source's combined descriptor, draft included)
    //   load target-only: deep_get learns that descriptor from this entry's checkpoints (same save as the verified
    //   target stream) and drops their draft data. Own-namespace entries keep the strict rule (both streams).
    const bool tgt_only = ctx_dft != nullptr && target_only(e);
    size_t keys_tgt_only = 0;
    if (!tgt_only && ctx_dft && e.has_dft && draft_kv_src(e)) {
        // a source adopted for target-only loads (its draft KV types differ) whose entry does not qualify (a target
        //   descriptor mismatch, an envelope-less legacy stream): its draft stream must never reach this draft context
        DSK_WRN("entry %s: its source's draft KV types differ and its target stream does not load alone, dropping it "
                "from the adopt sources (its files stay)\n", e.id.c_str());
        drop_invalid(idx);
        refresh_status();
        return SERVER_DISK_LOAD_INVALID;
    }
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
        if (tgt_only) {
            keys_tgt_only = keys.size();
        }
        size_t i = 0;
        for (auto it = checkpoints.begin(); it != checkpoints.end();) {
            common_prompt_checkpoint full;
            bool io_error = false;
            if (!deep_get(keys[i++], full, &io_error, tgt_only) || full.n_tokens != it->n_tokens) {
                if (io_error) {
                    throw std::runtime_error("a checkpoint read error");
                }
                if (tgt_only) {
                    // [TAG_ADOPT_TARGET_ONLY] a source checkpoint that does not load target-only (another descriptor)
                    //   is left out, as before the target-only checkpoint load: the entry stays eligible
                    it = checkpoints.erase(it);
                    keys_tgt_only--;
                    continue;
                }
                throw aux_invalid("a checkpoint is missing");
            }
            *it = std::move(full);
            ++it;
        }
    } catch (const aux_invalid & err) {
        DSK_WRN("entry %s: unreadable .aux (%s), dropping it\n", e.id.c_str(), err.what());
        drop_invalid(idx); // (R92) an adopted entry counts in its source's `dropped`
        refresh_status();
        return SERVER_DISK_LOAD_INVALID;
    } catch (const std::exception & err) {
        DSK_WRN("entry %s: .aux or checkpoint not readable (%s), keeping it\n", e.id.c_str(), err.what());
        n_load_other++;
        refresh_status();
        return SERVER_DISK_LOAD_OTHER;
    }

    // [TAG_CELL_EXT_PROVENANCE] a pre-fix (unmarked) state loads only when this entry's own stored token stream (.tok,
    //   read into e.tokens at init) proves the conversation text-only - no media chunk anywhere in it
    //   (LLAMA_LEGACY_UNMARKED=0: never); the scope clears the flag on every exit, an exception included
    const bool legacy_text = legacy_unmarked && !e.tokens.has_media();
    llama_state_legacy_accept_scope legacy(ctx_tgt, ctx_dft, legacy_text);

    llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
    server_disk_load_result res = load_stream(ctx_tgt, id_slot, path(e.id, ".tgt"), rule_for(e, true, tgt_only));
    if (res == SERVER_DISK_LOAD_OK && ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        res = tgt_only ? SERVER_DISK_LOAD_OK : e.has_dft ? load_stream(ctx_dft, id_slot, path(e.id, ".dft"), rule_for(e, false)) : SERVER_DISK_LOAD_INVALID;
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

    if (tgt_only) {
        // [TAG_ADOPT_TARGET_ONLY] the series-C reset ([TAG_SPEC_RESET]): the draft sequence was removed above (it stays
        //   empty), the speculative carry of this sequence is reset; the stored carry is not restored
        if (spec) {
            common_speculative_reset_state(spec, id_slot);
        }
    } else if (spec && !common_speculative_set_state(spec, id_slot, spec_state)) {
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

    // [TAG_CACHE_COMPAT2] (astra compat2 r2) the .aux checkpoints of an adopted pre-8 entry were checked against the
    //   weak descriptor hash only and are not restored by this load: never re-stamped into own storage
    if (!e.src.empty() && e.version < 8) {
        for (auto & c : checkpoints) {
            c.foreign_unverified = true;
        }
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
    // [TAG_DRAFT_PARTIAL] the draft history lacks the first rows: none of the loaded prefix after a target-only load
    //   (the draft sequence stays empty; a sparse MTP draft runs dense while depth > n_kv, 86661f3c3), the first
    //   n_tokens - draft_cover of an entry saved draft-partial. The target state is canonical: the prompt is reused as
    //   a prefix, saved (draft_cover = the rows the draft holds then) and deep-put like any other; `partial` (series C,
    //   the legacy /slots restore) is not set
    const bool short_cover = ctx_dft != nullptr && e.has_dft && !tgt_only && e.draft_cover < (int64_t) e.tokens.size();
    if (short_cover) {
        n_load_partial_cover++;
        DSK_WRN("entry %s: draft_cover %" PRId64 " < %zu tokens, the prompt is draft-partial (the draft lacks the first "
                "%" PRId64 " rows; the target state is reused and saved)\n", e.id.c_str(), e.draft_cover, e.tokens.size(),
                (int64_t) e.tokens.size() - std::max<int64_t>(0, e.draft_cover));
    }
    prompt.partial   = false;
    prompt.draft_gap = tgt_only ? (int32_t) e.tokens.size()
                     : short_cover ? (int32_t) ((int64_t) e.tokens.size() - std::max<int64_t>(0, e.draft_cover)) : 0;
    n_load_draft_partial += prompt.draft_partial();
    if (!e.src.empty()) {
        for (auto & a : adopted) {
            a.hits        += a.key == e.src;
            a.target_only += a.key == e.src && tgt_only;
        }
        if (tgt_only) {
            n_adopted_target_only++;
            DSK_INF("adopted load (target only; draft descriptor mismatch — draft reset): entry %s from source %s "
                    "(format %u), %zu prompt checkpoints loaded target-only, the prompt is draft-partial (gap %d)\n",
                    e.id.c_str(), e.src.c_str(), e.version, keys_tgt_only, prompt.draft_gap);
        } else {
            DSK_INF("adopted load: entry %s from source %s (format %u)\n", e.id.c_str(), e.src.c_str(), e.version);
        }
    }
    bytes_loaded += e.bytes;
    load_us      += (uint64_t) (ggml_time_us() - t_start);
    last_load_us    = (uint64_t) (ggml_time_us() - t_start); // [TAG_DISK_ASYNC_SAVE] /metrics
    last_load_bytes = e.bytes;
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
            if (server_deep_ref::rank(refs[i].kind) > server_deep_ref::rank(refs[i - 1].kind)) {
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

    // [TAG_SEMANTIC_CKPT] the newest ckpt_sem_max message-boundary refs stay on top of the schedule; an older one becomes
    //   a depth candidate (kept only where the ratio rule wants a checkpoint). 0: every semantic ref is a depth one
    {
        int32_t n_sem = 0;
        for (auto it = refs.rbegin(); it != refs.rend(); ++it) {
            if (it->kind == server_deep_ref::DEEP_KIND_SEMANTIC && ++n_sem > ckpt_sem_max) {
                it->kind = server_deep_ref::DEEP_KIND_DEPTH;
            }
        }
    }

    std::vector<server_deep_ref> kept;
    for (size_t i = 0; i < refs.size(); i++) {
        const auto & r = refs[i];
        const int64_t gap_min = std::max<int64_t>(ckpt_min_step, (int64_t) (ckpt_ratio * (double) (n_end - r.n_tokens)));
        if (i == 0 || r.kind != server_deep_ref::DEEP_KIND_DEPTH || r.n_tokens - kept.back().n_tokens >= gap_min) {
            kept.push_back(r);
        }
    }

    // ckpt_max bounds the schedule (depth, branch, anchor); the semantic refs (<= ckpt_sem_max) come on top of it
    const auto n_sched = [&]() {
        return (int32_t) std::count_if(kept.begin(), kept.end(),
                [](const auto & r) { return r.kind != server_deep_ref::DEEP_KIND_SEMANTIC; });
    };
    for (uint8_t kind : { server_deep_ref::DEEP_KIND_DEPTH, server_deep_ref::DEEP_KIND_BRANCH }) {
        for (size_t i = 1; n_sched() > ckpt_max && i < kept.size();) {
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
    // [TAG_CELL_EXT_PROVENANCE] the same prefix key, but a stored state from a build before the indexer x/y fix (its
    //   recurrent rows may come from inference on zeroed image x/y): replaced by this one, never linked
    const bool c_marked = ckpt_blob_ixy_marked(c.data_tgt.data(), c.data_tgt.size());
    if (it != ckpts.end() && it->second.src_dir.empty() && (it->second.ixy_marked || !c_marked)) {
        it->second.t_used = t_now;
        return true;
    }
    const bool replacing_unmarked = it != ckpts.end() && it->second.src_dir.empty();
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
        put_str_le(f, desc_key);
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
    m.ixy_marked = c_marked;
    if (replacing_unmarked) {
        n_ckpt_unmarked_replaced++; // counted once the replacement is committed
        DSK_INF("checkpoint %s: stored before the indexer x/y fix, replaced\n", key.c_str());
    }
    ckpts[key] = m; // replaces an adopt source's record (src_dir) only now that the own copy is committed
    bytes_written += m.bytes;
    if (written) {
        *written = true;
    }
    return true;
}

void server_disk_cache::deep_put(server_prompt & prompt, const common_prompt_checkpoint & c, uint8_t kind) {
    if (prompt.noncanonical || prompt.partial) { // [TAG_NONCANONICAL] [TAG_PARTIAL_PROMPT] C1
        return;
    }
    if (c.foreign_unverified) { // [TAG_CACHE_COMPAT2] adopted pre-8 bytes: no full-descriptor stamp
        return;
    }
    if (prompt.draft_partial() && !c.data_dft.empty()) { // [TAG_DRAFT_PARTIAL] see save(): stored draft-less
        common_prompt_checkpoint t = c;
        t.data_dft.clear();
        deep_put(prompt, t, kind);
        return;
    }
    // below min-step a re-process is cheaper than 113 MiB of disk (a short system prompt, a tiny chat); [TAG_CACHE_PIN] a
    //   pin is the client's explicit choice
    const bool pin = kind == server_deep_ref::DEEP_KIND_PIN;
    if (quota == 0 || c.data_tgt.empty() || (!pin && c.n_tokens < ckpt_min_step) || c.n_tokens < 1 ||
        c.n_tokens > (int64_t) prompt.tokens.size() ||
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
        DSK_INF("deep checkpoint at %" PRId64 " (%s) stored\n", c.n_tokens, server_deep_ref::name(kind));
        if (kind == server_deep_ref::DEEP_KIND_SEMANTIC) {
            n_deep_semantic++;
        }
    }
    // [TAG_CACHE_PIN] (review m2) counted only when the ref becomes a pin anew (a client re-sending the same pin is not)
    const bool was_pin = pin && std::any_of(prompt.deep.begin(), prompt.deep.end(), [&](const auto & r) {
        return r.n_tokens == c.n_tokens && r.kind == server_deep_ref::DEEP_KIND_PIN;
    });
    if (pin) {
        if (!was_pin) {
            n_deep_pins++;
        }
        // [TAG_CACHE_PIN] one pin per conversation: an older pin becomes a depth candidate
        for (auto & r : prompt.deep) {
            if (r.kind == server_deep_ref::DEEP_KIND_PIN && r.n_tokens != c.n_tokens) {
                r.kind = server_deep_ref::DEEP_KIND_DEPTH;
            }
        }
    }

    bool found = false;
    for (auto & r : prompt.deep) {
        if (r.n_tokens == c.n_tokens) {
            r.key  = key;
            if (server_deep_ref::rank(kind) > server_deep_ref::rank(r.kind)) {
                r.kind = kind;
            }
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

    // (review r4 2) as enforce(): the in-flight charge; (review r6 3) enforce() always runs its TTL / collections - its
    //   victim search finds nothing when only pinned / protected entries are left
    if (need_enforce && total_bytes() + inflight_charge(pinned_set(), "") > quota) {
        enforce("");
    }
    if (written || pin) { // (merge review m1) /disk-cache shows the new deep checkpoint / pin counters at once
        refresh_status();
    }
}

void server_disk_cache::deep_adopt(server_prompt & prompt, const server_deep_ckpt_meta & m, uint8_t kind) {
    if (kind == server_deep_ref::DEEP_KIND_PIN) { // [TAG_CACHE_PIN] see deep_put
        if (!std::any_of(prompt.deep.begin(), prompt.deep.end(), [&](const auto & r) {
                return r.key == m.key && r.kind == server_deep_ref::DEEP_KIND_PIN; })) {
            n_deep_pins++; // (review m2) only a new pin
            refresh_status(); // (merge review m1)
        }
        for (auto & r : prompt.deep) {
            if (r.kind == server_deep_ref::DEEP_KIND_PIN && r.key != m.key) {
                r.kind = server_deep_ref::DEEP_KIND_DEPTH;
            }
        }
    }
    for (auto & r : prompt.deep) {
        if (r.key == m.key) {
            if (server_deep_ref::rank(kind) > server_deep_ref::rank(r.kind)) {
                r.kind = kind;
            }
            return;
        }
    }
    prompt.deep.push_back({ m.n_tokens, m.key, kind });
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

bool server_disk_cache::deep_get(const std::string & key, common_prompt_checkpoint & out, bool * io_error, bool learn) {
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
        // [TAG_DISK_ADOPT] an adopt source's format 5 / 6 checkpoint has no envelope: accepted on the operator's word,
        //   only from a source whose format had no envelope (Codex compat2 C)
        const bool from_src = !it->second.src_dir.empty();
        const bool legacy_ok = from_src && legacy_format(it->second.src_version);
        bool ok = get_le(f, magic) && (magic == CKPT_ENV_MAGIC || (legacy_ok && magic == CKPT_MAGIC)) && get_str_le(f, fp);
        if (ok && magic == CKPT_ENV_MAGIC) {
            ok = get_le(f, env_ver) && get_str_le(f, dh);
        } else {
            env_ver = ENVELOPE_VERSION; // an eligible format 5 / 6 checkpoint (legacy_ok): no envelope to compare
            dh      = desc_key;
        }
        // [TAG_STATE_DESCRIPTOR] checked before anything reaches the contexts: a checkpoint of another state layout
        //   is a miss (it would otherwise abort in common_prompt_checkpoint::load_tgt on a size mismatch)
        // [TAG_CACHE_COMPAT2] v2 full hash; a v1 (weak) envelope only from an adopted format-7 source
        const bool v1_src = from_src && it->second.src_version == 7;
        const bool env_ok = (env_ver == ENVELOPE_VERSION && dh == desc_key) || (env_ver == 1 && v1_src && dh == desc_h);
        // [TAG_ADOPT_TARGET_ONLY] except an adopt source's checkpoint that carries the source's learned target-
        //   compatible descriptor (or teaches it: `learn`, a target-only load of an entry whose target stream matched,
        //   written in the same save): its target state loads, its draft data is dropped below. The learned value
        //   names the envelope version (a v2 hash, or a v1 one of a format-7 source)
        bool tgt_only = false;
        if (ok && !env_ok && from_src && magic == CKPT_ENV_MAGIC && (env_ver == ENVELOPE_VERSION || (env_ver == 1 && v1_src))) {
            const std::string vdh = "v" + std::to_string(env_ver) + ":" + dh;
            // (sol review) keyed by the owning source, not its directory: legacy sources share <root>/ckpt
            auto lt = tgt_compat_dh.find(it->second.src_key);
            if (lt == tgt_compat_dh.end() && learn) {
                lt = tgt_compat_dh.emplace(it->second.src_key, vdh).first;
                DSK_INF("adopt source %s (%s): combined descriptor %s is target-compatible (learned from a target-only load)\n",
                        it->second.src_key.c_str(), it->second.src_dir.c_str(), vdh.c_str());
            }
            tgt_only = lt != tgt_compat_dh.end() && lt->second == vdh;
        }
        if (ok && !env_ok && !tgt_only) {
            throw std::runtime_error("state descriptor mismatch (v" + std::to_string(env_ver) + " " + dh + ", expected v" +
                                     std::to_string(ENVELOPE_VERSION) + " " + desc_key + ")");
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
        if (tgt_only) {
            // [TAG_ADOPT_TARGET_ONLY] the draft data has the source's draft layout: dropped (a draft-less checkpoint, as
            //   [TAG_CKPT_NO_DFT] writes them); the carry is the target's (its schema is part of the target descriptor)
            out.data_dft.clear();
            n_ckpt_target_only++;
        }
        out.foreign_unverified = from_src && it->second.src_version < 8; // [TAG_CACHE_COMPAT2]
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
        count_src_ckpt_drop(it->second); // (R92) e.g. a state descriptor mismatch of an adopted checkpoint
        ckpts.erase(it);
        refresh_status();
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

void server_disk_cache::count_src_ckpt_drop(const server_deep_ckpt_meta & m) {
    if (m.src_dir.empty()) {
        return; // an own checkpoint
    }
    // (sol review) the owner only, once: legacy sources share <root>/ckpt, a src_dir match would charge all of them
    for (auto & a : adopted) {
        if (a.key == m.src_key) {
            a.dropped++;
            a.dropped_ckpts++;
            return;
        }
    }
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
    count_src_ckpt_drop(it->second); // (R92)
    ckpts.erase(it);
    n_ckpt_dropped++;
    refresh_status();
}

bool server_disk_cache::ckpt_legacy_text_ok(const std::string & key, int64_t n_tokens) const {
    // [TAG_CELL_EXT_PROVENANCE] see the declaration
    if (!legacy_unmarked) {
        return false;
    }
    for (const auto & e : entries) {
        const bool refs = std::any_of(e.deep.begin(), e.deep.end(), [&](const auto & r) { return r.key == key; }) ||
                          std::find(e.ckpt_keys.begin(), e.ckpt_keys.end(), key) != e.ckpt_keys.end();
        if (refs && n_tokens > 0 && (size_t) n_tokens <= e.tokens.size() && !e.tokens.has_media((size_t) n_tokens)) {
            return true;
        }
    }
    return false;
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
    if (as) { // [TAG_DISK_ASYNC_SAVE] the checkpoints of the saves in flight (stored before their copy)
        for (const auto & j : as->jobs) {
            for (const auto & r : j->e.deep) {
                referenced.insert(r.key);
            }
            referenced.insert(j->e.ckpt_keys.begin(), j->e.ckpt_keys.end());
        }
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
// cutting of tensor ranges into blocks of BLOB_BYTES (--cache-disk-async > 0: from the start of each range; async 0:
// aligned within the tensor, as v68) makes the unchanged part of a growing conversation produce the same blocks again.

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

// (review r12 1) the synchronous fallback's block claims go once its streams are written (the entry is registered
//   right after, before any collection of the success path)
void server_disk_cache::release_sync_claims() {
    if (!as) {
        sync_claims.clear();
        return;
    }
    std::lock_guard<std::mutex> lock(as->mtx);
    for (const auto & k : sync_claims) {
        disk_async_pending_release(*as, k, 1);
    }
    sync_claims.clear();
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
        bytes_copied += size; // [TAG_DISK_ASYNC_SAVE] /metrics: the stream bytes, as the async path counts them
    };
    w.write_tensor = [&](ggml_tensor * tensor, size_t offset, size_t size) {
        if (size < (64u << 10)) {
            // small ranges (e.g. the per-row ranges of a transposed V cache) stay inline: no file per few KiB
            const size_t at = inl.size();
            inl.resize(at + size);
            const int64_t t_copy = ggml_time_us(); // [TAG_DISK_ASYNC_SAVE] /metrics
            ggml_backend_tensor_get(tensor, inl.data() + at, offset, size);
            save_copy_us += (uint64_t) (ggml_time_us() - t_copy);
            bytes_copied += size;
            n_logical += size;
            return;
        }
        flush_inline();
        n_logical += size;
        size_t pos = offset;
        const size_t end = offset + size;
        while (ok && pos < end) {
            // [TAG_DISK_DELTA_REL] with --cache-disk-async > 0 (as) block boundaries are fixed offsets from the start of
            //   the range, as the asynchronous path cuts them (the fallback shares its blocks), so the same rows give the
            //   same blocks wherever the sequence's cells sit. With --cache-disk-async 0 the v68 cut (aligned within the
            //   tensor): the synchronous path writes v68's maps and blocks byte for byte. Either cut is a dedup choice
            //   only - the stream (the concatenation) is the same and every loader (v68's included) is cut-agnostic
            const size_t blk_end = as ? std::min(end, offset + ((pos - offset) / BLOB_BYTES + 1) * BLOB_BYTES)
                                      : std::min(end, (pos / BLOB_BYTES + 1) * BLOB_BYTES);
            const size_t n = blk_end - pos;
            buf.resize(n);
            const int64_t t_copy = ggml_time_us(); // [TAG_DISK_ASYNC_SAVE] /metrics
            ggml_backend_tensor_get(tensor, buf.data(), pos, n);
            save_copy_us += (uint64_t) (ggml_time_us() - t_copy);
            bytes_copied += n;
            const std::string key = hash_key(buf.data(), n);
            if (as) {
                // [TAG_DISK_ASYNC_SAVE] (review r12 1, r13 2 / 7) the synchronous fallback beside saves in flight, on the
                //   workers' claims and never waiting for them: the claim is taken and recorded in one step (recorded
                //   claims are released after the streams, an exception included); a block a worker is writing NOW is
                //   written here too (same bytes, writer-unique temp, atomic rename) - the entry never references a
                //   file that is not there yet; any other absent block is marked being written (a worker meeting it
                //   waits for this write, on its own thread), written, unmarked
                const fs::path fin = fs::path(dir) / "blk" / (key + ".blk");
                if (sync_claims.size() == sync_claims.capacity()) { // (review r14 6) geometric: the push_back below cannot throw
                    sync_claims.reserve(std::max<size_t>(256, 2 * sync_claims.capacity()));
                }
                disk_async_writing_guard guard{ *as };
                bool present = false, dup = false;
                {
                    std::lock_guard<std::mutex> lock(as->mtx);
                    as->pending[key]++;
                    sync_claims.push_back(key);
                    // own files are removed on this thread only: an own record (a block this save or an earlier
                    //   commit already holds) needs no stat and is never rewritten
                    if (blk_own.count(key)) {
                        present = true;
                    } else if (as->writing.count(key)) {
                        dup = true;
                    } else {
                        present = file_present(fin) > 0;
                        if (!present) {
                            as->writing.insert(key);
                            guard.key = &key;
                        }
                    }
                }
                // (review r14 5) indexed before the write (blobs + blk_own): a file written here is never uncharged or
                //   invisible to blob_gc(), whatever throws after it; a failed write takes back exactly the records it added
                const bool new_blob = !blobs.count(key);
                if (new_blob) {
                    blobs[key] = n;
                    blob_bytes += n;
                }
                const bool new_own = blk_own.insert(key).second;
                if (!present) {
                    if (!disk_write_block_file(fin, buf.data(), n)) {
                        if (new_own) {
                            blk_own.erase(key);
                        }
                        if (new_blob) {
                            blobs.erase(key);
                            blob_bytes -= n;
                        }
                        ok = false;
                        break;
                    }
                    n_written += n;
                    if (dup) {
                        n_bytes_written_twice += n;
                    }
                }
            } else if (blobs.count(key) == 0) {
                if (!disk_write_block_file(blob_path(key), buf.data(), n)) {
                    ok = false;
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
                                                       const stream_rule & rule) {
    const bool legacy_ok = rule.legacy_ok;
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
        if (!get_le(fm, ver) || !get_str_le(fm, h) || !envelope_ok(rule, ver, h) || !get_le(fm, n_total) || n_total == 0) {
            if (fm.bad()) {
                DSK_WRN("%s: read error at the state envelope, keeping it\n", map_path.c_str());
                return SERVER_DISK_LOAD_OTHER;
            }
            DSK_WRN("%s: state envelope mismatch (v%u descriptor %s, expected v%u %s%s)\n", map_path.c_str(), ver, h.c_str(),
                    ENVELOPE_VERSION, rule.h2.c_str(), rule.v1_ok ? " or an adopted v1" : "");
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

bool server_disk_cache::gc_unlink(const std::string & path) {
    std::error_code ec;
    fs::remove(path, ec); // an absent file: false without an error
    if (!ec || ec == std::errc::no_such_file_or_directory) {
        return true;
    }
    if (!warned_gc_unlink) {
        warned_gc_unlink = true;
        DSK_WRN("block collection: cannot remove %s (%s): kept inventoried and charged, retried at the next "
                "collection (further failures this run are not logged)\n", path.c_str(), ec.message().c_str());
    }
    return false;
}

// (Codex gate r5 Major 2) a physical presence check: 1 present, 0 confirmed absent, -1 unknown (an I/O error)
static int file_present(const fs::path & p) {
    std::error_code ec;
    const bool r = fs::is_regular_file(p, ec);
    if (r) {
        return 1;
    }
    return !ec || ec == std::errc::no_such_file_or_directory ? 0 : -1;
}

void server_disk_cache::blob_gc() {
    if (gc_hold) {
        return; // [TAG_DISK_READ_STATUS] an entry skipped at start may reference any of them
    }
    std::set<std::string> referenced, own_referenced;
    std::map<std::string, std::string> elsewhere; // key -> a foreign namespace's <ns>/blk holding it
    for (const auto & e : entries) {
        referenced.insert(e.blobs.begin(), e.blobs.end());
        if (e.src.empty()) {
            own_referenced.insert(e.blobs.begin(), e.blobs.end());
        }
    }
    // [TAG_DISK_ASYNC_SAVE] the blocks of the saves in flight are referenced; a worker's claim (pending + presence
    //   check) waits while this collection runs
    std::unique_lock<std::mutex> lock_async;
    if (as) {
        lock_async = std::unique_lock<std::mutex>(as->mtx);
        for (const auto & p : as->pending) {
            referenced.insert(p.first);
            own_referenced.insert(p.first);
        }
    }
    // (Codex gate r5 Major 2) the alternative an own copy would fall back to must exist now: a stale record (the
    //   file went behind this process) is forgotten (confirmed absent) and the own copy stays
    const auto alt_present = [&](const std::string & k) {
        if (blk_legacy.count(k)) {
            const int p = file_present(fs::path(root) / "blk" / (k + ".blk"));
            if (p > 0) {
                return true;
            }
            if (p == 0) {
                blk_legacy.erase(k);
            }
        }
        const auto s = blk_src.find(k);
        if (s != blk_src.end()) {
            const int p = file_present(fs::path(s->second) / (k + ".blk"));
            if (p > 0) {
                return true;
            }
            if (p == 0) {
                blk_src.erase(s);
            }
        }
        return false;
    };
    // (Codex gate r4 Major) an own-store copy no own entry needs goes when another store still holds the key (an
    //   adopted / legacy original: the read location falls back to it) - foreign references never pin a redundant own
    //   copy; no rescan needed for that. (Codex gate r5 Major 1) the record goes only with the file
    for (auto it = blk_own.begin(); it != blk_own.end();) {
        if (!own_referenced.count(*it) && alt_present(*it) &&
            gc_unlink((fs::path(dir) / "blk" / (*it + ".blk")).string())) {
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
            for (const auto & k : ns.stored) {
                elsewhere.emplace(k, (fs::path(ns.path) / "blk").string());
            }
        }
    }
    for (auto it = blobs.begin(); it != blobs.end();) {
        // an own copy only foreign maps reference, held in that namespace's own <ns>/blk too (not adopted this run,
        //   e.g. a restart without the adopt key): redundant, reclaimed; the sole physical copy of a block a foreign
        //   map references stays (charged to own)
        const auto el = elsewhere.find(it->first);
        const bool own_only_foreign = referenced.count(it->first) && blk_own.count(it->first) &&
                                      !entry_refs.count(it->first) && el != elsewhere.end() &&
                                      file_present(fs::path(el->second) / (it->first + ".blk")) > 0;
        if (own_only_foreign && gc_unlink((fs::path(dir) / "blk" / (it->first + ".blk")).string())) {
            blk_own.erase(it->first);
            if (!blk_legacy.count(it->first) && !blk_src.count(it->first)) {
                blob_bytes -= it->second;
                it = blobs.erase(it);
                continue;
            }
        }
        if (!referenced.count(it->first)) {
            // every physical copy: the own one, a legacy one nothing on disk references; an adopt source's is
            //   only forgotten (never written). (Codex gate r5 Major 1) a copy that cannot be removed stays
            //   inventoried and charged (the key with it), retried at the next collection
            bool gone = true;
            if (blk_own.count(it->first)) {
                if (gc_unlink((fs::path(dir) / "blk" / (it->first + ".blk")).string())) {
                    blk_own.erase(it->first);
                } else {
                    gone = false;
                }
            }
            if (blk_legacy.count(it->first)) {
                if (gc_unlink((fs::path(root) / "blk" / (it->first + ".blk")).string())) {
                    blk_legacy.erase(it->first);
                } else {
                    gone = false;
                }
            }
            if (!gone) {
                ++it;
                continue;
            }
            blk_src.erase(it->first);
            blob_bytes -= it->second;
            it = blobs.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<server_disk_lost> server_disk_cache::revalidate(const std::string & fp) {
    std::vector<server_disk_lost> lost = drain(); // [TAG_DISK_ASYNC_SAVE] the saves in flight belong to the current namespace
    if (server_disk_cache_same_identity(fp, fingerprint)) {
        fingerprint = fp; // [TAG_CACHE_COMPAT2] same namespace key: only the cheap part may differ
        return lost;
    }
    // [TAG_DISK_NAMESPACE] a mismatch is a miss, never a delete: open the new fingerprint's namespace, the old one
    //   stays as a foreign namespace (TTL / cap)
    DSK_WRN("the model / context / flags changed since the cache was opened: switching namespace %s -> %s\n",
            fingerprint.substr(0, 16).c_str(), fp.substr(0, 16).c_str());
    fingerprint = fp;
    dir = (fs::path(root) / ("ns-" + fp.substr(0, 16))).string();
    clear_entries();
    ckpts.clear();
    blobs.clear();
    blk_own.clear();
    blk_legacy.clear();
    blk_src.clear();
    blob_bytes = 0;
    init();
    return lost;
}

void server_disk_cache::refresh_status() {
    size_t n_claims = 0; // (sc3fix) server thread, never under as->mtx here
    if (as) {
        std::lock_guard<std::mutex> lock(as->mtx);
        n_claims = as->pending.size();
    }
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
            { "draft_cover", e.draft_cover        }, // [TAG_PARTIAL_PROMPT]
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
                       { "dropped", a.dropped }, { "dropped_ckpts", a.dropped_ckpts },
                       { "adopted_target_only", a.target_only },
                       { "draft_kv_differs", a.draft_kv_differs } });
    }
    json j = {
        { "enabled",        quota > 0           },
        { "adopt",          al                  },
        { "key",            fingerprint.substr(0, 16) },
        { "descriptor",     desc_h              },
        { "descriptor_full", desc_key           }, // [TAG_CACHE_COMPAT2] envelope v2
        { "foreign",        fl                  },
        { "dir",            dir                 },
        { "quota_bytes",    quota               },
        { "bytes_on_disk",  total_bytes()       },
        { "storage",        storage_split()     },
        { "min_tokens",     n_min_tokens        },
        { "ttl_days",       ttl_days            },
        { "ckpt_ratio",     ckpt_ratio          },
        { "ckpt_max",       ckpt_max            },
        { "ckpt_semantic",  ckpt_sem_max        },
        { "deep_semantic",  n_deep_semantic     },
        { "deep_pins",      n_deep_pins         },
        { "n_entries",      entries.size()      },
        { "n_deep_ckpts",   ckpts.size()        },
        { "n_blocks",       blobs.size()        },
        { "block_bytes",    blob_bytes          },
        { "saves",          n_saves             },
        { "loads",          n_loads             },
        { "load_failures",  n_load_failed       },
        { "load_failures_capacity", n_load_capacity },
        { "load_failures_other",    n_load_other    },
        { "adopted_target_only",    n_adopted_target_only },
        { "saves_refused_partial",  n_save_partial  },
        { "loads_partial_cover",    n_load_partial_cover },
        { "loads_draft_partial",    n_load_draft_partial }, // [TAG_DRAFT_PARTIAL]
        { "saves_draft_partial",    n_save_draft_partial },
        { "ckpt_target_only",       n_ckpt_target_only   }, // [TAG_ADOPT_TARGET_ONLY]
        { "n_entries_partial_cover", (size_t) std::count_if(entries.begin(), entries.end(), [](const server_disk_cache_entry & e) {
                                        return e.has_dft && e.draft_cover < (int64_t) e.tokens.size(); }) },
        { "deep_rewinds",   n_deep_rewinds      },
        { "ckpt_dropped",   n_ckpt_dropped      },
        { "ckpt_unmarked_replaced", n_ckpt_unmarked_replaced }, // [TAG_CELL_EXT_PROVENANCE]
        { "legacy_unmarked",        legacy_unmarked          }, // [TAG_CELL_EXT_PROVENANCE] LLAMA_LEGACY_UNMARKED
        { "deep_copied",    n_deep_copied       },
        { "blk_copied",     n_blk_copied_total  },
        { "evictions",      n_evict             },
        { "bytes_written",  bytes_written       },
        { "bytes_loaded",   bytes_loaded        },
        { "save_seconds",   save_us / 1e6       },
        { "load_seconds",   load_us / 1e6       },
        { "save_async",     as ? as->budget : 0 }, // [TAG_DISK_ASYNC_SAVE] staging bytes, 0 = synchronous
        { "saves_pending",  n_pending()         },
        { "bytes_pending",  pending_bytes()     }, // quota charge of the saves in flight: written minus what they replace
        { "save_backpressure", n_save_backpressure }, // [TAG_DISK_ASYNC_SAVE] saves run synchronously: two saves in flight (fixed)
        { "last_save_commit_seconds", last_save_commit_us / 1e6 }, // the commit part of the last async save
        { "pin_check_failures", n_pin_check_failures }, // test hook check_pins
        { "saves_superseded",   n_saves_superseded   }, // async saves dropped: a synchronous save extending them committed first
        { "incr_blocks_reused", incr_blocks_reused   }, // [TAG_DISK_INCR] blocks taken from the slot's last save, not the device
        { "incr_bytes_skipped", incr_bytes_skipped   },
        { "incr_verify_planned", incr_blocks_verify_planned },
        { "incr_verify_checked", as ? as->incr_verified.load() : 0 },
        { "incr_verify_mismatch", as ? as->incr_mismatch.load() : 0 },
        { "bytes_device_read", bytes_copied - incr_bytes_skipped }, // (sc3fix 5) bytes_copied minus the reused blocks
        { "async_claims",   n_claims            }, // (sc3fix 2-4) block keys claimed by saves in flight (0 when settled)
        { "bytes_written_twice", n_bytes_written_twice }, // the synchronous fallback rewrote a block a worker was writing
        { "saves_async",    n_saves_async       },
        { "saves_async_failed", n_saves_async_failed },
        { "save_async_seconds", save_async_us / 1e6 },
        { "save_copy_seconds",  save_copy_us / 1e6  },
        { "save_wait_seconds",  save_wait_us / 1e6  },
        { "bytes_copied",       bytes_copied        },
        { "last_save_blocking_seconds", last_save_blocking_us / 1e6 },
        { "last_save_seconds",          last_save_total_us / 1e6    },
        { "last_save_written",          last_save_written           },
        { "last_load_seconds",          last_load_us / 1e6          },
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
    desc_key   = h16_of(tgt + "\n" + dft); // [TAG_CACHE_COMPAT2] every byte counts: key v2 + envelope v2
    desc_tgt_full = h16_of(tgt);
    desc_dft_full = h16_of(dft);
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
std::string server_disk_cache_fingerprint_structure(const std::string & fp, std::string * why, bool draft_kv) {
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
        if (draft_kv || (pre != "dk" && pre != "dv")) {
            res += field(name, v.substr(pre.size()));
        }
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
        // [TAG_DISK_BLK_NS] a namespaced source keeps its blocks in <ns>/blk: indexed read-only (own / legacy first).
        //   (Codex gate r5 Major 2) collected here, published to the inventory only once the source is accepted: a
        //   refused / empty source is not protected from the foreign TTL / cap, so its blocks are never an alternative
        std::map<std::string, uint64_t> src_blk;
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
                    src_blk.emplace(k, n);
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
            // [TAG_ADOPT_TARGET_ONLY] only the draft KV types differ: the target stream may still match this target's
            //   descriptor (checked per entry at load), the draft stream never does (its descriptor names its KV types)
            const std::string src_tgt_struct = server_disk_cache_fingerprint_structure(a.fingerprint, nullptr, false);
            if (src_tgt_struct.empty() || src_tgt_struct != server_disk_cache_fingerprint_structure(fingerprint, nullptr, false)) {
                DSK_WRN("adopt %s refused: its structural part (model / draft / mmproj files, per-slot n_ctx, KV types) differs: %s\n",
                        key.c_str(), src_struct.c_str());
                continue;
            }
            a.draft_kv_differs = true;
            DSK_INF("adopt %s: only the draft KV types differ, its entries load target-only (draft reset)\n", key.c_str());
        }
        DSK_INF("adopt %s: structure %s\n", key.c_str(), src_struct.c_str());
        for (const auto & [k, n] : src_blk) {
            // (Codex gate r4 Major) inventoried even behind an own / legacy copy: a reclaimed own copy falls back to it
            blk_src.emplace(k, (base / "blk").string());
            if (!blobs.count(k)) {
                blobs[k] = n;
                blob_bytes += n;
            }
        }
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
                push_entry(std::move(e));
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
        a.version = fingerprint_format(a.fingerprint);
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
                m.src_version = a.version;
                m.src_key = key; // (R92) its header names this source (fp == key above)
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
