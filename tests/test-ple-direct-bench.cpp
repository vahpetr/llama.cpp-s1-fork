// [TAG_PLE_DIRECT] microbench of the PLE row gather (not a ctest): today's mmap gather (with / without the madvise
// WILLNEED prefetch) vs the direct-read reader (src/llama-ple-direct.cpp), on a real id stream, under whatever page cache
// the process is allowed -- plus an exact LRU simulation of row cache vs page cache budgets.
//
//   test-ple-direct-bench make --out FILE [--rows 320001536] [--pad-bytes 525608960]
//       a synthetic GGUF with the prod PLE geometry: per_layer_token_embd.weight Q5_0 [160, rows] (110 B rows) behind a
//       pad tensor, so the table starts at the prod file offset mod 4096 (3872) by default; deterministic bytes.
//   test-ple-direct-bench run --gguf FILE --ids STREAM --mode mmap|prefetch|direct|buffered
//       [--tensor per_layer_token_embd.weight] [--heads 16] [--skip-tokens N]
//       [--phases K:tokens[:lead_us],...  |  --tokens-per-gather K=3 --max-tokens N --lead-us L]
//       [--cache-mib M=1024] [--threads T=16] [--nowait 0|1 (1)] [--drop-cache] [--dequant] [--label S]
//       walks the stream phase by phase in ONE process (so the page cache / row cache a prefill phase leaves behind is
//       what the decode phase after it meets), timing every gather of K tokens x heads ids (decode: K = 1..6; prefill:
//       K = 2048); prefetch mode sleeps lead_us between the WILLNEED advice and the gather (the GPU work the prod split
//       waits for) and times the gather only; the direct reader's time is all exposed (it runs in set_input). One JSON
//       line per phase:
//       us/gather p50/p90/p99/max, us/token, major/minor faults, bytes read from storage (/proc/self/io), the table's
//       page-cache residency before/after (mincore), RSS, and the reader's row-cache / page-cache / device split.
//       mmap = the mapping with MADV_RANDOM (llama_mmap's lazy range), rows copied one by one (the CPU get_rows' reads);
//       prefetch = the same after posix_madvise(WILLNEED) of the sorted, coalesced pages (qwen4exp_ple_prefetch).
//   test-ple-direct-bench sim --ids STREAM [--table-offset 536633120] [--row-bytes 110] [--heads 16]
//       [--row-mib 64,128,256,512,1024,2048] [--page-mib 512,1024,2048,4000,8000,13400,33570] [--tail-tokens 20000]
//       exact LRU hit ratios (stack distances) of a row cache and of a 4 KiB page cache over the stream, whole stream
//       and the last --tail-tokens tokens (= decode after the preceding context has been read).
//
// STREAM: int32 little-endian row ids in the graph's layout (idx[token*heads + head]) -- scripts/bench/probe/
// ple_idstream.py in piagent writes them from real text; a LLAMA_PLE_DUMP text file ("<n_tokens> <id> ...") also works.
//
// Constrained page cache (the 1M-context host state, ~12-25 % of the table resident): run it in a memory-capped cgroup,
//   systemd-run --user --scope -p MemoryHigh=4G -p MemoryMax=5G -p MemorySwapMax=0 test-ple-direct-bench run ...
//   docker run --memory 4g --memory-swap 4g ... test-ple-direct-bench run ...      (the same cap as memory.max)
// Page cache is charged to the cgroup that faults it in, so the cap bounds the table's resident share -- but only for
// pages not already resident under another cgroup: sync, then --drop-cache (POSIX_FADV_DONTNEED on the table range)
// so the run starts cold. The direct reader's row cache is anonymous memory inside the same cap; its O_DIRECT reads use
// no page cache at all. All modes therefore get the same memory budget.

#include "../src/llama-ple-direct.h"
#include "ple-direct-gguf.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

static const char * PLE_NAME = "per_layer_token_embd.weight";

static std::map<std::string, std::string> parse_args(int argc, char ** argv, int first) {
    std::map<std::string, std::string> a;
    for (int i = first; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) { fprintf(stderr, "unexpected argument %s\n", k.c_str()); exit(2); }
        k = k.substr(2);
        if (k == "drop-cache" || k == "dequant") { a[k] = "1"; continue; }
        if (i + 1 >= argc) { fprintf(stderr, "--%s needs a value\n", k.c_str()); exit(2); }
        a[k] = argv[++i];
    }
    return a;
}

static std::string arg(const std::map<std::string, std::string> & a, const char * k, const char * def) {
    auto it = a.find(k);
    return it == a.end() ? def : it->second;
}

static std::vector<double> parse_list(const std::string & s) {
    std::vector<double> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t e = s.find(',', p);
        if (e == std::string::npos) e = s.size();
        v.push_back(atof(s.substr(p, e - p).c_str()));
        p = e + 1;
    }
    return v;
}

static std::vector<int32_t> read_ids(const std::string & path) {
    std::vector<int32_t> ids;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    int c = fgetc(f);
    ungetc(c, f);
    if (c >= '0' && c <= '9') {
        // LLAMA_PLE_DUMP text: "<n_tokens> <id0> <id1> ..." per ubatch
        char * line = nullptr; size_t cap = 0;
        while (getline(&line, &cap, f) > 0) {
            char * p = line; char * e;
            strtol(p, &e, 10); p = e;
            for (;;) { long v = strtol(p, &e, 10); if (e == p) break; ids.push_back((int32_t) v); p = e; }
        }
        free(line);
    } else {
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        ids.resize((size_t) n / 4);
        if (fread(ids.data(), 4, ids.size(), f) != ids.size()) { fprintf(stderr, "short read %s\n", path.c_str()); exit(1); }
    }
    fclose(f);
    return ids;
}

struct io_counters { uint64_t read_bytes = 0, rchar = 0; };
static io_counters read_io() {
    io_counters c;
    FILE * f = fopen("/proc/self/io", "r");
    if (!f) return c;
    char k[64]; unsigned long long v;
    while (fscanf(f, "%63s %llu", k, &v) == 2) {
        if (!strcmp(k, "read_bytes:")) c.read_bytes = v;
        if (!strcmp(k, "rchar:")) c.rchar = v;
    }
    fclose(f);
    return c;
}

static long rss_kib() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long v = -1;
    while (fgets(line, sizeof(line), f)) { if (!strncmp(line, "VmRSS:", 6)) { v = atol(line + 6); break; } }
    fclose(f);
    return v;
}

// share of the table's 4 KiB pages in the page cache (mincore over a private read-only mapping of the range)
static double resident_pct(int fd, size_t offs, size_t len) {
    const size_t ps = 4096;
    const size_t a0 = offs & ~(ps - 1);
    const size_t n  = offs + len - a0;
    void * m = mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, (off_t) a0);
    if (m == MAP_FAILED) return -1.0;
    std::vector<unsigned char> vec((n + ps - 1) / ps);
    double pct = -1.0;
    if (mincore(m, n, vec.data()) == 0) {
        size_t r = 0; for (unsigned char b : vec) r += b & 1;
        pct = 100.0 * (double) r / (double) vec.size();
    }
    munmap(m, n);
    return pct;
}

static int cmd_make(const std::map<std::string, std::string> & a) {
    const std::string out = arg(a, "out", "");
    if (out.empty()) { fprintf(stderr, "make: --out FILE\n"); return 2; }
    const int64_t rows = atoll(arg(a, "rows", "320001536").c_str());
    // prod: the table's data starts at file offset 536 633 120 (mod 4096 = 3872); a pad tensor reproduces that shift
    const int64_t pad  = atoll(arg(a, "pad-bytes", "525608960").c_str());
    std::vector<ple_gguf_tensor> ts;
    if (pad > 0) ts.push_back({ "pad", GGML_TYPE_I8, pad, 1 });
    ts.push_back({ PLE_NAME, GGML_TYPE_Q5_0, 160, rows });
    const auto t0 = std::chrono::steady_clock::now();
    if (!ple_gguf_write(out, ts, 0x91e)) { fprintf(stderr, "write failed\n"); return 1; }
    ple_gguf_info info;
    ple_gguf_locate(out, PLE_NAME, info);
    printf("{\"make\": \"%s\", \"rows\": %" PRId64 ", \"row_bytes\": %zu, \"offset\": %zu, \"offset_mod_4096\": %zu, \"seconds\": %.1f}\n",
           out.c_str(), info.rows, info.rs, info.offs, info.offs % 4096,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}

// one phase of a run: consecutive tokens of the stream gathered K tokens at a time
struct run_phase {
    int     K       = 3;      // tokens per gather (16*K ids)
    int64_t tokens  = 0;      // tokens in this phase (0 = the rest of the stream)
    int64_t lead_us = 0;      // prefetch mode: sleep between the WILLNEED advice and the gather (the GPU time the mid-graph
                              // split waits for in prod: ~0.8 ms per decode round, ~72 ms per 2048-token ubatch at 213k)
};

static std::vector<run_phase> parse_phases(const std::string & s) {
    // "K:tokens[:lead_us],K:tokens[:lead_us],..."
    std::vector<run_phase> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t e = s.find(',', p);
        if (e == std::string::npos) e = s.size();
        const std::string one = s.substr(p, e - p);
        run_phase ph;
        long long k = 0, t = 0, l = 0;
        const int n = sscanf(one.c_str(), "%lld:%lld:%lld", &k, &t, &l);
        if (n < 1 || k <= 0) { fprintf(stderr, "bad phase '%s' (K:tokens[:lead_us])\n", one.c_str()); exit(2); }
        ph.K = (int) k; ph.tokens = n >= 2 ? t : 0; ph.lead_us = n >= 3 ? l : 0;
        v.push_back(ph);
        p = e + 1;
    }
    return v;
}

static int cmd_run(const std::map<std::string, std::string> & a) {
    const std::string gguf = arg(a, "gguf", ""), ids_path = arg(a, "ids", ""), mode = arg(a, "mode", "");
    if (gguf.empty() || ids_path.empty() || mode.empty()) { fprintf(stderr, "run: --gguf FILE --ids STREAM --mode mmap|prefetch|direct|buffered\n"); return 2; }
    ple_gguf_info info;
    if (!ple_gguf_locate(gguf, arg(a, "tensor", PLE_NAME), info)) { fprintf(stderr, "no tensor %s in %s\n", arg(a, "tensor", PLE_NAME).c_str(), gguf.c_str()); return 1; }
    const int heads = atoi(arg(a, "heads", "16").c_str());
    const std::vector<int32_t> all = read_ids(ids_path);
    const int64_t n_tok_all = (int64_t) all.size() / heads;
    const int64_t skip = std::min<int64_t>(atoll(arg(a, "skip-tokens", "0").c_str()), n_tok_all);
    std::vector<run_phase> phases;
    if (a.count("phases")) {
        phases = parse_phases(arg(a, "phases", ""));
    } else {
        run_phase ph;
        ph.K = atoi(arg(a, "tokens-per-gather", "3").c_str());
        ph.tokens = atoll(arg(a, "max-tokens", "0").c_str());
        ph.lead_us = atoll(arg(a, "lead-us", "0").c_str());
        phases.push_back(ph);
    }
    for (int64_t i = skip * heads; i < n_tok_all * heads; ++i) {
        if (all[i] < 0 || all[i] >= info.rows) { fprintf(stderr, "id %d out of range (table %" PRId64 " rows)\n", all[i], info.rows); return 1; }
    }

    const int fd = open(gguf.c_str(), O_RDONLY);
    struct stat sb; fstat(fd, &sb);
    const size_t tbytes = (size_t) info.rows * info.rs;
    if (a.count("drop-cache")) {
        posix_fadvise(fd, (off_t) info.offs, (off_t) tbytes, POSIX_FADV_DONTNEED);
    }

    // the mapping: every mode has it (the direct reader's fallback source), only mmap/prefetch read through it
    const uint8_t * map = (const uint8_t *) mmap(nullptr, (size_t) sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { fprintf(stderr, "mmap failed\n"); return 1; }
    const uint8_t * rows = map + info.offs;
    {
        const size_t ps = 4096;
        const uintptr_t b = (uintptr_t) rows & ~(uintptr_t) (ps - 1);
        const uintptr_t e = ((uintptr_t) rows + tbytes + ps - 1) & ~(uintptr_t) (ps - 1);
        posix_madvise((void *) b, e - b, POSIX_MADV_RANDOM);
    }

    std::unique_ptr<llama_ple_direct_reader> rd;
    if (mode == "direct" || mode == "buffered") {
        llama_ple_direct_params p;
        p.mode        = mode == "direct" ? 1 : 2;
        p.cache_bytes = (size_t) atoll(arg(a, "cache-mib", "1024").c_str()) << 20;
        p.n_threads   = atoi(arg(a, "threads", "16").c_str());
        p.nowait      = atoi(arg(a, "nowait", "1").c_str()) != 0;
        std::string err;
        rd = llama_ple_direct_reader::open(fd, info.offs, info.rs, info.rows, p, &err);
        if (!rd) { fprintf(stderr, "reader: %s\n", err.c_str()); return 1; }
        fprintf(stderr, "reader: %s\n", rd->describe().c_str());
    } else if (mode != "mmap" && mode != "prefetch") {
        fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 2;
    }
    const bool dequant = a.count("dequant") > 0;
    const auto * tt = ggml_get_type_traits(info.type);
    const int64_t ne0 = (int64_t) (info.rs / ggml_type_size(info.type)) * ggml_blck_size(info.type);
    uint64_t sink = 0;

    int64_t t_next = skip;
    for (size_t pi = 0; pi < phases.size(); ++pi) {
        const run_phase & ph = phases[pi];
        const int K = ph.K;
        const int64_t n_tok = ph.tokens > 0 ? std::min<int64_t>(ph.tokens, n_tok_all - t_next) : n_tok_all - t_next;
        const int64_t per = (int64_t) K * heads;
        std::vector<uint8_t> stage((size_t) per * info.rs);
        std::vector<float>   deq((size_t) per * ne0);
        std::vector<double>  us, us_adv;
        const double res0 = resident_pct(fd, info.offs, tbytes);
        const llama_ple_direct_stats st0 = rd ? rd->stats() : llama_ple_direct_stats();
        struct rusage r0, r1; getrusage(RUSAGE_SELF, &r0);
        const io_counters io0 = read_io();
        const auto T0 = std::chrono::steady_clock::now();
        std::vector<std::pair<uintptr_t, uintptr_t>> r;
        for (int64_t t = 0; t + K <= n_tok; t += K) {
            const int32_t * ids = all.data() + (t_next + t) * heads;
            if (mode == "prefetch") {
                // qwen4exp_ple_prefetch: page-aligned absolute ranges, sorted, coalesced, WILLNEED -- timed apart (set_input)
                const auto a0 = std::chrono::steady_clock::now();
                r.clear();
                for (int64_t i = 0; i < per; ++i) {
                    const uintptr_t b = (uintptr_t) (rows + (size_t) ids[i] * info.rs);
                    r.emplace_back(b & ~(uintptr_t) 4095, (b + info.rs + 4095) & ~(uintptr_t) 4095);
                }
                std::sort(r.begin(), r.end());
                uintptr_t b = r[0].first, e = r[0].second;
                for (size_t i = 1; i <= r.size(); ++i) {
                    if (i < r.size() && r[i].first <= e) { e = std::max(e, r[i].second); continue; }
                    posix_madvise((void *) b, e - b, POSIX_MADV_WILLNEED);
                    if (i < r.size()) { b = r[i].first; e = r[i].second; }
                }
                us_adv.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a0).count());
                if (ph.lead_us > 0) {
                    usleep((useconds_t) ph.lead_us);
                }
            }
            const auto t0 = std::chrono::steady_clock::now();
            if (rd) {
                rd->gather(ids, per, stage.data(), rows);
            } else {
                for (int64_t i = 0; i < per; ++i) {
                    memcpy(stage.data() + (size_t) i * info.rs, rows + (size_t) ids[i] * info.rs, info.rs);
                }
            }
            if (dequant && tt->to_float) {
                for (int64_t i = 0; i < per; ++i) {
                    tt->to_float(stage.data() + (size_t) i * info.rs, deq.data() + (size_t) i * ne0, ne0);
                }
                sink += (uint64_t) (deq[0] != 0.0f);
            }
            sink += stage[(size_t) (t % per) * info.rs];
            us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
        getrusage(RUSAGE_SELF, &r1);
        const io_counters io1 = read_io();
        const double res1 = resident_pct(fd, info.offs, tbytes);
        t_next += n_tok;

        std::vector<double> s = us;
        std::sort(s.begin(), s.end());
        auto pct = [&](double q) { return s.empty() ? 0.0 : s[std::min(s.size() - 1, (size_t) (q * (double) (s.size() - 1)))]; };
        double sum = 0; for (double x : us) sum += x;
        double sum_adv = 0; for (double x : us_adv) sum_adv += x;
        const int64_t n_g = (int64_t) us.size();
        printf("{\"label\": \"%s\", \"mode\": \"%s\", \"phase\": %zu, \"tokens_per_gather\": %d, \"lead_us\": %" PRId64 ", \"gathers\": %" PRId64 ", \"tokens\": %" PRId64 ", "
               "\"us_per_gather\": {\"mean\": %.1f, \"p50\": %.1f, \"p90\": %.1f, \"p99\": %.1f, \"max\": %.1f}, \"us_per_token\": %.2f, "
               "\"advise_us_per_gather\": %.1f, \"wall_s\": %.2f, \"majflt\": %ld, \"majflt_per_token\": %.3f, \"minflt\": %ld, "
               "\"storage_read_mib\": %.1f, \"resident_pct_before\": %.2f, \"resident_pct_after\": %.2f, \"rss_mib\": %.0f, \"dequant\": %d",
               arg(a, "label", "").c_str(), mode.c_str(), pi, K, ph.lead_us, n_g, n_g * K, n_g ? sum / n_g : 0.0, pct(0.5), pct(0.9), pct(0.99),
               s.empty() ? 0.0 : s.back(), n_g ? sum / (double) (n_g * K) : 0.0, n_g ? sum_adv / n_g : 0.0, wall,
               r1.ru_majflt - r0.ru_majflt, n_g ? (double) (r1.ru_majflt - r0.ru_majflt) / (double) (n_g * K) : 0.0,
               r1.ru_minflt - r0.ru_minflt, (io1.read_bytes - io0.read_bytes) / 1048576.0, res0, res1, rss_kib() / 1024.0, dequant ? 1 : 0);
        if (rd) {
            const auto st = rd->stats();
            const uint64_t un = st.unique - st0.unique;
            const double u = un ? 100.0 / (double) un : 0.0;
            printf(", \"distinct_ids\": %" PRIu64 ", \"row_cache_pct\": %.2f, \"page_cache_pct\": %.2f, \"device_pct\": %.2f, \"device_rows_per_token\": %.3f, "
                   "\"device_reads\": %" PRIu64 ", \"device_mib\": %.1f, \"fallback_rows\": %" PRIu64 ", \"evictions\": %" PRIu64,
                   un, (st.cache_hits - st0.cache_hits) * u, (st.pc_hits - st0.pc_hits) * u, (st.dev_rows - st0.dev_rows) * u,
                   n_g ? (double) (st.dev_rows - st0.dev_rows) / (double) (n_g * K) : 0.0, st.dev_reads - st0.dev_reads,
                   (st.dev_bytes - st0.dev_bytes) / 1048576.0, st.fallback_rows - st0.fallback_rows, st.evictions - st0.evictions);
        }
        printf(", \"sink\": %" PRIu64 "}\n", sink & 1);
        fflush(stdout);
    }
    rd.reset();
    munmap(const_cast<uint8_t *>(map), (size_t) sb.st_size);
    close(fd);
    return 0;
}

// Fenwick tree over access times: 1 at the last access time of every key
struct fenwick {
    std::vector<int32_t> t;
    explicit fenwick(size_t n) : t(n + 1, 0) {}
    void add(size_t i, int32_t v) { for (++i; i < t.size(); i += i & (0 - i)) t[i] += v; }
    int64_t sum(size_t i) const { int64_t s = 0; for (++i; i > 0; i -= i & (0 - i)) s += t[i]; return s; } // [0, i]
};

// LRU stack distance of every access (distinct keys since the previous access of the same key; -1 = first access)
static std::vector<int64_t> stack_distances(const std::vector<uint64_t> & keys) {
    const size_t n = keys.size();
    std::vector<uint64_t> u = keys;
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    std::vector<int64_t> last(u.size(), -1), d(n);
    fenwick fw(n);
    for (size_t t = 0; t < n; ++t) {
        const size_t k = (size_t) (std::lower_bound(u.begin(), u.end(), keys[t]) - u.begin());
        if (last[k] >= 0) {
            d[t] = fw.sum(t) - fw.sum((size_t) last[k]);  // keys whose latest access lies in (last, t)
            fw.add((size_t) last[k], -1);
        } else {
            d[t] = -1;
        }
        fw.add(t, 1);
        last[k] = (int64_t) t;
    }
    return d;
}

static int cmd_sim(const std::map<std::string, std::string> & a) {
    const std::string ids_path = arg(a, "ids", "");
    if (ids_path.empty()) { fprintf(stderr, "sim: --ids STREAM\n"); return 2; }
    const std::vector<int32_t> ids = read_ids(ids_path);
    const int     heads = atoi(arg(a, "heads", "16").c_str());
    const size_t  rs    = (size_t) atoll(arg(a, "row-bytes", "110").c_str());
    const size_t  offs  = (size_t) atoll(arg(a, "table-offset", "536633120").c_str());
    const auto row_mib  = parse_list(arg(a, "row-mib", "64,128,256,512,1024,2048,4096"));
    const auto page_mib = parse_list(arg(a, "page-mib", "512,1024,2048,4000,8000,13400,33570"));
    const int64_t n_tok = (int64_t) ids.size() / heads;
    const int64_t tail  = std::min<int64_t>(atoll(arg(a, "tail-tokens", "20000").c_str()), n_tok);
    const size_t  n     = (size_t) n_tok * heads;
    const size_t  tail0 = (size_t) (n_tok - tail) * heads;

    // rows
    std::vector<uint64_t> rk(ids.begin(), ids.begin() + n);
    const std::vector<int64_t> rd = stack_distances(rk);
    // pages: a row touches one or two 4 KiB pages; it is a page-cache hit only when all of them are
    std::vector<uint64_t> pk; pk.reserve(n + n / 30);
    std::vector<size_t>   first(n + 1);
    for (size_t i = 0; i < n; ++i) {
        first[i] = pk.size();
        const size_t o = offs + (size_t) ids[i] * rs;
        pk.push_back(o >> 12);
        if ((o >> 12) != ((o + rs - 1) >> 12)) pk.push_back((o + rs - 1) >> 12);
    }
    first[n] = pk.size();
    const std::vector<int64_t> pd = stack_distances(pk);
    std::vector<int64_t> prow(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t m = 0;
        for (size_t j = first[i]; j < first[i + 1]; ++j) { if (pd[j] < 0) { m = -1; break; } m = std::max(m, pd[j]); }
        prow[i] = m;
    }
    std::vector<uint64_t> ur(rk); std::sort(ur.begin(), ur.end()); ur.erase(std::unique(ur.begin(), ur.end()), ur.end());
    std::vector<uint64_t> up(pk); std::sort(up.begin(), up.end()); up.erase(std::unique(up.begin(), up.end()), up.end());

    auto hit = [&](const std::vector<int64_t> & d, int64_t cap, size_t from) {
        size_t h = 0; for (size_t i = from; i < n; ++i) h += (d[i] >= 0 && d[i] < cap); return 100.0 * (double) h / (double) (n - from);
    };
    printf("{\"sim\": \"%s\", \"tokens\": %" PRId64 ", \"accesses\": %zu, \"distinct_rows\": %zu, \"distinct_rows_mib\": %.1f, "
           "\"distinct_pages\": %zu, \"distinct_pages_mib\": %.1f, \"tail_tokens\": %" PRId64 ",\n \"row_cache\": [",
           ids_path.c_str(), n_tok, n, ur.size(), ur.size() * rs / 1048576.0, up.size(), up.size() * 4096 / 1048576.0, tail);
    for (size_t k = 0; k < row_mib.size(); ++k) {
        const int64_t cap = (int64_t) (row_mib[k] * 1048576.0 / (double) rs);
        printf("%s{\"mib\": %g, \"rows\": %" PRId64 ", \"hit_pct\": %.2f, \"tail_hit_pct\": %.2f}", k ? ", " : "", row_mib[k], cap,
               hit(rd, cap, 0), hit(rd, cap, tail0));
    }
    printf("],\n \"page_cache\": [");
    for (size_t k = 0; k < page_mib.size(); ++k) {
        const int64_t cap = (int64_t) (page_mib[k] * 256.0);
        printf("%s{\"mib\": %g, \"pages\": %" PRId64 ", \"table_pct\": %.1f, \"hit_pct\": %.2f, \"tail_hit_pct\": %.2f}", k ? ", " : "", page_mib[k], cap,
               100.0 * page_mib[k] / 33570.0, hit(prow, cap, 0), hit(prow, cap, tail0));
    }
    printf("]}\n");
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s make|run|sim [--options]  (see the header of tests/test-ple-direct-bench.cpp)\n", argv[0]);
        return 2;
    }
    const std::string cmd = argv[1];
    const auto a = parse_args(argc, argv, 2);
    if (cmd == "make") return cmd_make(a);
    if (cmd == "run")  return cmd_run(a);
    if (cmd == "sim")  return cmd_sim(a);
    fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
