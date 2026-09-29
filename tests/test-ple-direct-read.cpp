// [TAG_PLE_DIRECT] ctest: the direct-read gather (src/llama-ple-direct.cpp) returns, for every id, exactly the bytes the
// mmap gather reads -- on a synthetic GGUF with Q5_0 tables of the PLE row width (160 -> 110 B rows) at three different
// file-offset misalignments, the last one ending exactly at EOF (O_DIRECT short read).
//
// For every reader configuration -- mode 1 (O_DIRECT, falls back to buffered where the file system refuses it) with and
// without the RWF_NOWAIT page-cache probe, mode 2 (buffered pread) -- x row cache {none, 3 rows, 1000 rows, larger
// than the table} x reader threads {1, 4, 16}, it gathers random and adversarial id sets (first / last rows, rows
// straddling 512 B and 4 KiB boundaries, all-equal, runs of adjacent rows (merged reads), descending, heavy duplicates,
// a Zipf stream of decode-sized gathers that cycles the LRU, one prefill-sized gather) and compares the output byte for
// byte with the mapping. After every sequence the row cache's hash index and LRU list are verified.
// Also: the mmap fallback (every read fails -> the rows come from the mapping, identical), two threads gathering
// concurrently through one reader, and two negative controls (a flipped output byte, a reader opened one byte off the
// table start) that the comparison must catch.
//
// usage: test-ple-direct-read [dir]   (dir for the ~27 MB scratch file; default: $LLAMA_PLE_TEST_DIR or the cwd --
//                                     /tmp is often tmpfs, which refuses O_DIRECT and would test the fallback only)

#include "../src/llama-ple-direct.h"
#include "ple-direct-gguf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int g_fail = 0;
static uint64_t g_checks = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); g_fail++; } } while (0)

struct table_ref {
    std::string    name;
    ple_gguf_info  info;
    const uint8_t * rows = nullptr;   // row 0 in the mapping
};

// first mismatching id index, or -1
static int64_t compare(const table_ref & t, const std::vector<int32_t> & ids, const std::vector<uint8_t> & out) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (memcmp(out.data() + i * t.info.rs, t.rows + (size_t) ids[i] * t.info.rs, t.info.rs) != 0) {
            return (int64_t) i;
        }
    }
    return -1;
}

// ids whose 110 B row crosses a multiple of `boundary` in the file
static std::vector<int32_t> straddlers(const table_ref & t, size_t boundary, size_t max_n) {
    std::vector<int32_t> r;
    for (int64_t id = 0; id < t.info.rows && r.size() < max_n; ++id) {
        const size_t o = t.info.offs + (size_t) id * t.info.rs;
        if (o / boundary != (o + t.info.rs - 1) / boundary) {
            r.push_back((int32_t) id);
        }
    }
    return r;
}

static std::vector<std::vector<int32_t>> id_sets(const table_ref & t, std::mt19937_64 & rng, bool big) {
    const int32_t R = (int32_t) t.info.rows;
    std::uniform_int_distribution<int32_t> u(0, R - 1);
    std::vector<std::vector<int32_t>> s;
    s.push_back({ 0 });
    s.push_back({ R - 1 });
    s.push_back({ 0, 1, R - 2, R - 1, 0, R - 1 });
    { std::vector<int32_t> v(16); for (auto & x : v) x = u(rng); s.push_back(v); }              // one decode token
    { std::vector<int32_t> v(96); for (auto & x : v) x = u(rng); s.push_back(v); }              // a 6-token verify
    { std::vector<int32_t> v(64, u(rng)); s.push_back(v); }                                      // all equal
    { std::vector<int32_t> v; int32_t b = u(rng) % std::max(1, R - 1200); for (int32_t k = 0; k < 1200; ++k) v.push_back(b + k); s.push_back(v); } // adjacent: merged reads
    { std::vector<int32_t> v; for (int32_t k = 0; k < 300; ++k) v.push_back(R - 1 - k * (R / 301)); s.push_back(v); } // descending
    { std::vector<int32_t> v(500); std::vector<int32_t> pool(7); for (auto & x : pool) x = u(rng); for (auto & x : v) x = pool[rng() % 7]; s.push_back(v); } // duplicates
    s.push_back(straddlers(t, 512, 200));
    s.push_back(straddlers(t, 4096, 200));
    { std::vector<int32_t> v; for (int32_t k = 0; k < 64 && k < R; ++k) v.push_back(R - 1 - k); s.push_back(v); }   // the rows at EOF
    if (big) { std::vector<int32_t> v(8192); for (auto & x : v) x = u(rng); s.push_back(v); }    // a prefill-sized gather
    return s;
}

// Zipf(1.1) over a random permutation of the rows: a decode-like stream with a hot head, 16..96 ids per gather
static std::vector<std::vector<int32_t>> zipf_stream(const table_ref & t, std::mt19937_64 & rng, int n_gathers) {
    const int64_t n_items = std::min<int64_t>(t.info.rows, 20000);
    std::vector<double> cdf(n_items);
    double acc = 0.0;
    for (int64_t k = 0; k < n_items; ++k) { acc += 1.0 / pow((double) (k + 1), 1.1); cdf[k] = acc; }
    std::vector<int32_t> perm(n_items);
    std::uniform_int_distribution<int32_t> u(0, (int32_t) t.info.rows - 1);
    for (auto & x : perm) x = u(rng);
    std::uniform_real_distribution<double> ur(0.0, acc);
    std::vector<std::vector<int32_t>> s;
    for (int g = 0; g < n_gathers; ++g) {
        std::vector<int32_t> v(16 * (1 + rng() % 6));
        for (auto & x : v) { x = perm[std::lower_bound(cdf.begin(), cdf.end(), ur(rng)) - cdf.begin()]; }
        s.push_back(v);
    }
    return s;
}

static void run_sets(llama_ple_direct_reader & rd, const table_ref & t, const std::vector<std::vector<int32_t>> & sets,
                     const uint8_t * mmap_rows, const char * what) {
    std::vector<uint8_t> out;
    for (size_t k = 0; k < sets.size(); ++k) {
        const auto & ids = sets[k];
        out.assign(ids.size() * t.info.rs, 0xA5);
        rd.gather(ids.data(), (int64_t) ids.size(), out.data(), mmap_rows);
        const int64_t bad = compare(t, ids, out);
        g_checks += ids.size();
        CHECK(bad < 0, "%s %s: set %zu (%zu ids): id[%" PRId64 "] = row %d differs from the mapping (%s)", t.name.c_str(), what, k,
              ids.size(), bad, bad >= 0 ? ids[bad] : -1, rd.describe().c_str());
        if (bad >= 0) {
            return;
        }
    }
    const std::string sc = rd.self_check();
    CHECK(sc.empty(), "%s %s: row cache inconsistent: %s", t.name.c_str(), what, sc.c_str());
}

int main(int argc, char ** argv) {
    std::string dir = argc > 1 ? argv[1] : (getenv("LLAMA_PLE_TEST_DIR") ? getenv("LLAMA_PLE_TEST_DIR") : ".");
    const std::string path = dir + "/test-ple-direct-read." + std::to_string(getpid()) + ".gguf";

    // pad tensors shift the tables to different offsets mod 512 / 4096; table_eof is last and its size is a multiple of
    // the 32 B GGUF alignment (16 rows x 110 B = 1760 = 55 x 32), so its last row ends at EOF
    const std::vector<ple_gguf_tensor> ts = {
        { "pad0",      GGML_TYPE_F32,  7,   1      },
        { "table_a",   GGML_TYPE_Q5_0, 160, 20011  },
        { "pad1",      GGML_TYPE_F32,  93,  1      },
        { "table_b",   GGML_TYPE_Q5_0, 160, 3001   },
        { "pad2",      GGML_TYPE_F32,  517, 1      },
        { "table_eof", GGML_TYPE_Q5_0, 160, 200016 },
    };
    if (!ple_gguf_write(path, ts, 0x5eed)) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return 1;
    }
    const int fd = open(path.c_str(), O_RDONLY);
    fsync(fd); // the pages just written must be clean for POSIX_FADV_DONTNEED to drop them below
    struct stat sb;
    fstat(fd, &sb);
    const size_t fsize = (size_t) sb.st_size;
    const uint8_t * map = (const uint8_t *) mmap(nullptr, fsize, PROT_READ, MAP_SHARED, fd, 0);
    if (fd < 0 || map == MAP_FAILED) {
        fprintf(stderr, "cannot map %s\n", path.c_str());
        unlink(path.c_str());
        return 1;
    }

    std::vector<table_ref> tabs;
    for (const char * n : { "table_a", "table_b", "table_eof" }) {
        table_ref t;
        t.name = n;
        if (!ple_gguf_locate(path, n, t.info) || t.info.rs != 110) {
            fprintf(stderr, "cannot locate %s\n", n);
            return 1;
        }
        t.rows = map + t.info.offs;
        tabs.push_back(t);
        printf("%-9s: %7" PRId64 " rows x %zu B at file offset %zu (mod 512 = %zu, mod 4096 = %zu)\n", n, t.info.rows, t.info.rs,
               t.info.offs, t.info.offs % 512, t.info.offs % 4096);
    }
    CHECK(tabs[2].info.offs + (size_t) tabs[2].info.rows * tabs[2].info.rs == fsize, "table_eof does not end at EOF");

    // the synthetic generator is what we compare against only through the mapping; make sure the mapping has the bytes
    {
        std::vector<uint8_t> want(4096);
        ple_gguf_fill(0x5eed, 5, 0, want.data(), want.size());
        CHECK(memcmp(want.data(), tabs[2].rows, want.size()) == 0, "mapping of table_eof does not hold the generated bytes");
    }

    std::mt19937_64 rng(42);
    int n_readers = 0;
    std::string modes_seen;
    for (int mode : { 1, 2 }) {
        for (int nowait : { 0, 1 }) {
            if (mode == 2 && nowait) {
                continue;
            }
            for (int64_t cache_rows : { (int64_t) 0, (int64_t) 3, (int64_t) 1000, (int64_t) 1 << 22 }) {
                for (int th : { 1, 4, 16 }) {
                    for (size_t ti = 0; ti < tabs.size(); ++ti) {
                        const auto & t = tabs[ti];
                        // cold file cache for every reader: the page-cache probe then meets both resident rows (those
                        // the comparison has touched through the mapping since) and cold ones that go to the device
                        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                        llama_ple_direct_params p;
                        p.mode        = mode;
                        p.nowait      = nowait != 0;
                        p.cache_bytes = (size_t) cache_rows * t.info.rs;
                        p.n_threads   = th;
                        std::string err;
                        auto rd = llama_ple_direct_reader::open(fd, t.info.offs, t.info.rs, t.info.rows, p, &err);
                        CHECK(rd != nullptr, "open failed: %s", err.c_str());
                        if (!rd) {
                            continue;
                        }
                        n_readers++;
                        if (n_readers == 1 || (mode == 2 && cache_rows == 0 && th == 1 && ti == 0)) {
                            printf("reader: %s\n", rd->describe().c_str());
                        }
                        if (modes_seen.find(rd->describe().substr(0, 6)) == std::string::npos) {
                            modes_seen += rd->describe().substr(0, 6) + " ";
                        }
                        char what[96];
                        snprintf(what, sizeof(what), "mode %d nowait %d cache %" PRId64 " rows threads %d", mode, nowait, cache_rows, th);
                        const bool big = th == 16;
                        run_sets(*rd, t, id_sets(t, rng, big), nullptr, what);
                        run_sets(*rd, t, zipf_stream(t, rng, cache_rows == 1000 ? 400 : 60), nullptr, what);
                        run_sets(*rd, t, id_sets(t, rng, false), nullptr, what); // again, now partly cached
                        const auto st = rd->stats();
                        CHECK(st.fallback_rows == 0, "%s %s: %" PRIu64 " fallback rows without a failure", t.name.c_str(), what, st.fallback_rows);
                        if (cache_rows == 1000) {
                            CHECK(st.cache_hits > 0 && st.evictions > 0, "%s %s: the LRU was not exercised (hits %" PRIu64 ", evictions %" PRIu64 ")",
                                  t.name.c_str(), what, st.cache_hits, st.evictions);
                        }
                    }
                }
            }
        }
    }
    printf("readers: %d configurations; effective modes: %s\n", n_readers, modes_seen.c_str());
    if (modes_seen.find("mode 1") == std::string::npos) {
        printf("NOTE: O_DIRECT was refused on %s -- mode 1 fell back to buffered reads; run from a disk-backed directory to cover it\n", dir.c_str());
    }

    // mmap fallback: every read fails, the rows come from the mapping (and then from the cache)
    for (int mode : { 1, 2 }) {
        const auto & t = tabs[2];
        llama_ple_direct_params p;
        p.mode = mode; p.cache_bytes = 500 * t.info.rs; p.n_threads = 4; p.fail_reads_for_test = 1;
        std::string err;
        auto rd = llama_ple_direct_reader::open(fd, t.info.offs, t.info.rs, t.info.rows, p, &err);
        CHECK(rd != nullptr, "open failed: %s", err.c_str());
        if (rd) {
            run_sets(*rd, t, id_sets(t, rng, true), t.rows, "fallback");
            run_sets(*rd, t, zipf_stream(t, rng, 100), t.rows, "fallback");
            const auto st = rd->stats();
            CHECK(st.fallback_rows > 0 && st.dev_rows == st.fallback_rows, "fallback: %" PRIu64 " fallback rows of %" PRIu64 " device rows",
                  st.fallback_rows, st.dev_rows);
            printf("fallback (mode %d): %s\n", mode, rd->stats_line().c_str());
        }
    }

    // two threads through one reader
    {
        const auto & t = tabs[0];
        llama_ple_direct_params p;
        p.mode = 1; p.cache_bytes = 300 * t.info.rs; p.n_threads = 8; p.nowait = true;
        auto rd = llama_ple_direct_reader::open(fd, t.info.offs, t.info.rs, t.info.rows, p, nullptr);
        CHECK(rd != nullptr, "open failed");
        if (rd) {
            std::atomic<int> bad{0};
            auto body = [&](uint64_t seed) {
                std::mt19937_64 r(seed);
                auto s = zipf_stream(t, r, 300);
                std::vector<uint8_t> out;
                for (const auto & ids : s) {
                    out.assign(ids.size() * t.info.rs, 0);
                    rd->gather(ids.data(), (int64_t) ids.size(), out.data(), nullptr);
                    if (compare(t, ids, out) >= 0) { bad++; }
                }
            };
            std::thread a(body, 1), b(body, 2);
            a.join(); b.join();
            CHECK(bad == 0, "concurrent gathers: %d mismatching gathers", bad.load());
            CHECK(rd->self_check().empty(), "concurrent gathers: row cache inconsistent: %s", rd->self_check().c_str());
            g_checks += 600;
        }
    }

    // negative controls: the comparison must see a single flipped byte, and a reader one byte off the table start
    {
        const auto & t = tabs[1];
        llama_ple_direct_params p;
        p.mode = 1; p.n_threads = 4;
        auto rd = llama_ple_direct_reader::open(fd, t.info.offs, t.info.rs, t.info.rows, p, nullptr);
        std::vector<int32_t> ids = id_sets(t, rng, false)[4];
        std::vector<uint8_t> out(ids.size() * t.info.rs);
        rd->gather(ids.data(), (int64_t) ids.size(), out.data(), nullptr);
        CHECK(compare(t, ids, out) < 0, "control gather differs");
        out[out.size() / 2] ^= 0x10;
        CHECK(compare(t, ids, out) >= 0, "negative control 1: a flipped byte went unnoticed");

        auto off = llama_ple_direct_reader::open(fd, t.info.offs + 1, t.info.rs, t.info.rows - 1, p, nullptr);
        std::vector<uint8_t> out2(ids.size() * t.info.rs);
        for (auto & x : ids) { x = std::min<int32_t>(x, (int32_t) t.info.rows - 2); }
        off->gather(ids.data(), (int64_t) ids.size(), out2.data(), nullptr);
        CHECK(compare(t, ids, out2) >= 0, "negative control 2: a reader one byte off the table start went unnoticed");
        printf("negative controls: detected\n");
    }

    munmap(const_cast<uint8_t *>(map), fsize);
    close(fd);
    unlink(path.c_str());

    printf("%s: %" PRIu64 " rows compared, %d failures\n", g_fail ? "FAILED" : "OK", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
