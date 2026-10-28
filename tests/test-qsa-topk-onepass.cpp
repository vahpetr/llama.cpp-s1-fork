// [TAG_QSA_TOPK_ONEPASS] (plan 2.7.2 #2) the single-pass QSA indexer selection, knob off vs on, the ordered ids bit for bit,
// on a GPU. One child per arm (GGML_VK_QSA_TOPK_ONEPASS is read once at device init):
//   off:    GGML_VK_QSA_TOPK_ONEPASS=0                    (v2; set explicitly: unset = the device default, ON on gfx1151)
//   on:     =1 with the default gates                     (what a deployment with the knob runs; backoff off in both on arms,
//           so the path of every case is the gate's, not the history's)
//   on_all: =1 + GGML_VK_QSA_TOPK_ONEPASS_MIN_KV=1 + _MIN_CELLS=1 (every admissible shape on the onepass kernel)
// Each child runs the same case list on synthetic score planes - the metadata form (the deployed path; nokeys when the
// device has it) and the plane form (keyed), depths 2 051 .. 1 048 576 cells, 1 / 6 / 64 x 2 streams / 128 rows, k 2051 and
// 2048, modes: distinct (4-cell blocks = duplicate keys), fine ties (ReLU zeros + 1024 levels), census (dead blocks, +1e9
// tail boost), mixed (odd rows all equal: forced fallback next to onepass rows), equal / coarse (overflow: every row falls
// back), sparse (fewer visible cells than k: the -inf tie group), permuted cell->block maps - and checks every row against a
// host reference of the documented contract (keys > T ascending, then the lowest-id ties ascending), every one of the R
// copies of the op against the first, and dumps the ids. Timing: median ms per op over `reps` graphs of R ops.
// PASS: every arm exact on every case; the on / on_all dumps byte-identical to off; off dispatched no onepass selection, on
// did for every case at or above the default gate, on_all for every admissible case; the fallback counters show the forced
// cases falling back (and the parent prints them per case).
// Negative control: a -DGGML_VULKAN_QSA_TOPK_ONEPASS_NEGCTL=ON build (the onepass select takes the highest-id ties) must keep
// the off arm exact and make >= 90 % of the on_all arm's onepass cases FAIL (and no other case); it prints NEGATIVE
// CONTROL PASS.
// Not a ctest: needs a Vulkan device with the v2 selection; without one it prints SKIP and exits 0.
// usage: test-qsa-topk-onepass [reps]                 (parent: the three arms, compare, timing table)
//        test-qsa-topk-onepass --child <dir> [reps]   (one arm, the env decides)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

enum { M_DISTINCT, M_FINE, M_CENSUS, M_MIXED, M_EQUAL, M_COARSE, M_SPARSE, M_COUNT };
static const char * mode_name(int m) {
    static const char * n[M_COUNT] = { "distinct", "fine", "census", "mixed", "equal", "coarse", "sparse" };
    return n[m];
}

struct op_case {
    int64_t n_kv, n_tps, n_stream;
    int     width, mode;
    bool    plane;   // the plane form (ggml_qsa_top_k over an f16 mask) instead of the metadata form
    bool    perm;    // random cell->block map instead of cell i -> block i/4
    std::string desc() const {
        char b[200];
        snprintf(b, sizeof(b), "n_kv=%" PRId64 " rows=%" PRId64 "x%" PRId64 " k=%d mode=%s form=%s map=%s", n_kv, n_tps, n_stream, width,
                 mode_name(mode), plane ? "plane" : "vis", perm ? "perm" : "blk4");
        return b;
    }
};

static std::vector<op_case> make_cases() {
    std::vector<op_case> v;
    // the brief's shapes: k 2051 (indexer_top_k 2048 + r - 1), 32k .. 1M cells, decode / verify / prefill-chunk rows
    for (int64_t n : { 32768, 65536, 131072, 262144, 524288, 1048576 }) {
        for (int64_t r : { 1, 6, 128 }) {
            v.push_back({ n, r, 1, 2051, M_DISTINCT, false, false });
        }
    }
    v.push_back({ 65536,   128, 1, 2048, M_DISTINCT, false, false });
    v.push_back({ 1048576, 128, 1, 2048, M_DISTINCT, false, false });
    for (int64_t n : { 65536, 262144, 1048576 }) {
        for (int64_t r : { 6, 128 }) {
            v.push_back({ n, r, 1, 2051, M_FINE,   false, false });
            v.push_back({ n, r, 1, 2051, M_CENSUS, false, false });
        }
    }
    v.push_back({ 262144,  128, 1, 2051, M_MIXED,  false, false });
    v.push_back({ 1048576, 128, 1, 2051, M_MIXED,  false, false });
    v.push_back({ 65536,   6,   1, 2051, M_EQUAL,  false, false });
    v.push_back({ 65536,   128, 1, 2051, M_EQUAL,  false, false });
    v.push_back({ 262144,  128, 1, 2051, M_COARSE, false, false });
    v.push_back({ 131072,  6,   1, 2051, M_SPARSE, false, false });
    v.push_back({ 262144,  128, 1, 2051, M_DISTINCT, false, true });
    v.push_back({ 1048576, 6,   1, 2051, M_CENSUS,   false, true });
    v.push_back({ 131072,  64,  2, 2051, M_DISTINCT, false, false });
    v.push_back({ 131072,  64,  2, 2051, M_MIXED,    false, false });
    // the plane form (keyed passes; the mask plane is n_kv x rows f16)
    v.push_back({ 65536,  128, 1, 2051, M_DISTINCT, true, false });
    v.push_back({ 262144, 6,   1, 2051, M_DISTINCT, true, false });
    v.push_back({ 262144, 128, 1, 2051, M_CENSUS,   true, false });
    v.push_back({ 131072, 128, 1, 2051, M_MIXED,    true, true  });
    // small depths (onepass only in on_all): k == n_kv, the sample = every cell, ragged tiles
    for (int64_t n : { 2051, 4096, 8192, 9000, 20000 }) {
        for (int64_t r : { 1, 6 }) {
            v.push_back({ n, r, 1, 2051, n == 2051 ? M_DISTINCT : M_FINE, false, false });
        }
        v.push_back({ n, 6, 1, 2051, M_DISTINCT, true, false });
    }
    // the gate's neighbourhood (default MIN_KV / MIN_CELLS)
    for (int64_t n : { 49152, 98304, 196608 }) {
        for (int64_t r : { 1, 6, 128 }) {
            v.push_back({ n, r, 1, 2051, M_DISTINCT, false, false });
        }
    }
    // a whole 2048-row ubatch in one dispatch (LLAMA_QSA_CHUNK off; ~0.6 GiB of planes)
    v.push_back({ 262144, 2048, 1, 2051, M_DISTINCT, false, false });
    return v;
}

static ggml_backend_t gpu_backend() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); }
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

static uint32_t key_of(float x) {
    uint32_t y;
    memcpy(&y, &x, sizeof(y));
    if (y == 0x80000000u) { y = 0u; }
    return (y & 0x80000000u) ? ~y : (y | 0x80000000u);
}

// the case's inputs on the host: scores [nb, n_tps, n_stream], cell_blk [n_kv, n_stream], the metadata (cells [n_kv, 4,
// n_stream], toks [4, n_tps, n_stream]) and the visibility it implies (the plane form's mask is 0 / -inf from it)
struct op_inputs {
    int64_t nb = 0;
    std::vector<float>   score;
    std::vector<int32_t> cb, cells, toks;
    std::vector<uint8_t> vis;   // [n_tps*n_stream][n_kv]
};

static op_inputs make_inputs(const op_case & c) {
    op_inputs in;
    const int64_t n = c.n_kv, nr = c.n_tps*c.n_stream;
    in.nb = (n + 3)/4;
    std::mt19937 rng(0x0b1e55edu ^ (uint32_t) (n*131 + c.n_tps*7919 + c.n_stream*31 + c.width*17 + c.mode*1009 + c.plane*3 + c.perm*5));
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const int64_t used = n - n/64;   // the padded tail of the pool: empty cells
    in.cb.resize(n*c.n_stream);
    in.cells.assign(n*4*c.n_stream, 0);
    in.toks.resize(4*nr);
    in.vis.assign(nr*n, 0);
    for (int64_t s = 0; s < c.n_stream; s++) {
        int32_t * pos = in.cells.data() + s*4*n;
        int32_t * sm  = pos + n;
        for (int64_t j = 0; j < n; j++) {
            in.cb[s*n + j] = c.perm ? (int32_t) (rng() % (uint32_t) in.nb) : (int32_t) (j/4);
            const bool empty = j >= used;
            pos[j] = empty ? -1 : (int32_t) j;
            // sparse: a cell belongs to the queries' sequence with probability ~ (k/2) / n, else to sequence 1
            sm[j]  = empty ? 0 : (c.mode == M_SPARSE && u01(rng) >= 0.5f*c.width/(float) n) ? 2 : 1;
            pos[2*n + j] = INT32_MAX;
            pos[3*n + j] = INT32_MAX;
        }
        for (int64_t t = 0; t < c.n_tps; t++) {
            int32_t * tk = in.toks.data() + 4*(s*c.n_tps + t);
            tk[0] = (int32_t) (used - c.n_tps + t);   // the chunk's last positions: causal
            tk[1] = 1;
            tk[2] = INT32_MAX;
            tk[3] = INT32_MAX;
            uint8_t * vr = in.vis.data() + (s*c.n_tps + t)*n;
            for (int64_t j = 0; j < n; j++) {
                vr[j] = (sm[j] & tk[1]) != 0 && pos[j] <= tk[0];
            }
        }
    }
    in.score.resize(in.nb*nr);
    for (int64_t row = 0; row < nr; row++) {
        float * r = in.score.data() + row*in.nb;
        const bool eq_row = c.mode == M_EQUAL || (c.mode == M_MIXED && (row & 1));
        for (int64_t b = 0; b < in.nb; b++) {
            const float x = u01(rng);
            float v = 8.0f*x;
            if (c.mode == M_FINE)   { v = x < 0.6f ? 0.0f : std::floor(1024.0f*u01(rng))/256.0f; }
            if (c.mode == M_COARSE) { v = std::floor(16.0f*x)/4.0f; }
            if (c.mode == M_CENSUS) { v = u01(rng) < 0.05f ? -INFINITY : 4.0f*x; }
            if (eq_row)             { v = 0.5f; }
            r[b] = v;
        }
        if (c.mode == M_CENSUS) {
            // the census's +1e9 boost on the block of the row's own (last visible) cell
            const int64_t t = row % c.n_tps, s = row / c.n_tps;
            const int64_t own = in.toks[4*(s*c.n_tps + t)];
            r[in.cb[s*n + own]] = 1e9f;
        }
    }
    return in;
}

// the contract: T = the k-th largest key, every key > T ascending, then the lowest-id keys == T ascending, until k
static void reference_row(const op_inputs & in, const op_case & c, int64_t row, std::vector<uint32_t> & keys,
                          std::vector<uint32_t> & tmp, std::vector<int32_t> & out) {
    const int64_t n = c.n_kv, s = row / c.n_tps;
    const float   * sc = in.score.data() + row*in.nb;
    const uint8_t * vr = in.vis.data() + row*n;
    keys.resize(n);
    for (int64_t j = 0; j < n; j++) {
        keys[j] = vr[j] ? key_of(sc[in.cb[s*n + j]]) : 0x007FFFFFu;   // visible: score + 0, else -inf
    }
    tmp = keys;
    std::nth_element(tmp.begin(), tmp.begin() + (c.width - 1), tmp.end(), std::greater<uint32_t>());
    const uint32_t T = tmp[c.width - 1];
    out.clear();
    for (int64_t j = 0; j < n; j++) {
        if (keys[j] > T) { out.push_back((int32_t) j); }
    }
    for (int64_t j = 0; j < n && (int64_t) out.size() < c.width; j++) {
        if (keys[j] == T) { out.push_back((int32_t) j); }
    }
}

struct op_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> ops;
    ~op_graph() {
        if (buf) { ggml_backend_buffer_free(buf); }
        if (ctx) { ggml_free(ctx); }
    }
};

static bool build(op_graph & g, ggml_backend_t be, const op_case & c, const op_inputs & in, int R, size_t idx) {
    ggml_init_params ip = { (size_t) (16 + R)*ggml_tensor_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    ggml_tensor * score = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, in.nb, c.n_tps, c.n_stream);
    ggml_tensor * cb    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, c.n_kv, c.n_stream);
    ggml_tensor * cells = nullptr, * toks = nullptr, * mask = nullptr;
    if (c.plane) {
        mask = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, c.n_kv, c.n_tps, 1, c.n_stream);
    } else {
        cells = ggml_new_tensor_3d(g.ctx, GGML_TYPE_I32, c.n_kv, 4, c.n_stream);
        toks  = ggml_new_tensor_3d(g.ctx, GGML_TYPE_I32, 4, c.n_tps, c.n_stream);
    }
    for (int r = 0; r < R; r++) {
        ggml_tensor * op = c.plane ? ggml_qsa_top_k(g.ctx, score, cb, mask, c.width)
                                   : ggml_qsa_top_k_vis(g.ctx, score, cb, cells, toks, c.width, 0, true);
        ggml_format_name(op, "opcase%zu_%d", idx, r);
        g.ops.push_back(op);
    }
    if (!ggml_backend_supports_op(be, g.ops[0])) {
        return false;
    }
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    GGML_ASSERT(g.buf);
    ggml_backend_tensor_set(score, in.score.data(), 0, ggml_nbytes(score));
    ggml_backend_tensor_set(cb, in.cb.data(), 0, ggml_nbytes(cb));
    if (c.plane) {
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ni = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> m(in.vis.size());
        for (size_t i = 0; i < m.size(); i++) { m[i] = in.vis[i] ? z : ni; }
        ggml_backend_tensor_set(mask, m.data(), 0, ggml_nbytes(mask));
    } else {
        ggml_backend_tensor_set(cells, in.cells.data(), 0, ggml_nbytes(cells));
        ggml_backend_tensor_set(toks, in.toks.data(), 0, ggml_nbytes(toks));
    }
    // a recognisable junk value, so a row the kernel skips cannot pass
    for (ggml_tensor * op : g.ops) {
        std::vector<int32_t> j(ggml_nelements(op), -7);
        ggml_backend_tensor_set(op, j.data(), 0, ggml_nbytes(op));
    }
    return true;
}

static bool compute(ggml_backend_t be, const std::vector<ggml_tensor *> & nodes) {
    const size_t size = 16 + nodes.size();
    ggml_init_params ip = { ggml_graph_overhead_custom(size, false) + 1024, nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, size, false);
    for (ggml_tensor * n : nodes) { ggml_build_forward_expand(gf, n); }
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    ggml_free(gctx);
    return ok;
}

static int child(const std::string & dir, int reps) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    ggml_backend_load_all();
    {
        ggml_backend_t probe = gpu_backend();
        if (!probe) { printf("[op] SKIP no GPU\n"); return 0; }
        ggml_backend_free(probe);
    }
    const auto cases = make_cases();
    // GGML_TEST_ONEPASS_R=<n>: copies of the op per graph (default 4; 16 = the 16 chunk selections of a 2048-row ubatch)
    const int R = getenv("GGML_TEST_ONEPASS_R") ? std::max(1, atoi(getenv("GGML_TEST_ONEPASS_R"))) : 4;
    std::vector<uint32_t> keys, tmp;
    std::vector<int32_t> ref;
    // GGML_TEST_ONEPASS_CASES=i,j,... runs those cases only (timing probes; the parent's PASS needs every case)
    std::vector<bool> run(cases.size(), true);
    if (const char * sel = getenv("GGML_TEST_ONEPASS_CASES")) {
        std::fill(run.begin(), run.end(), false);
        for (const char * q = sel; *q; ) {
            const size_t i = (size_t) strtoul(q, (char **) &q, 10);
            if (i < run.size()) { run[i] = true; }
            if (*q == ',') { q++; } else if (*q) { break; }
        }
    }
    for (size_t i = 0; i < cases.size(); ++i) {
        if (!run[i]) { continue; }
        const op_case & c = cases[i];
        printf("[op] begin %zu\n", i);   // the parent attributes the counter lines from here on to this case
        const op_inputs in = make_inputs(c);
        // a backend per case: its free prints the onepass counters ("(final)"), so the parent sees per-case deltas
        ggml_backend_t be = gpu_backend();
        GGML_ASSERT(be);
        {
            op_graph g;
            if (!build(g, be, c, in, R, i)) {
                printf("[op] case %zu %s unsupported\n", i, c.desc().c_str());
                ggml_backend_free(be);
                continue;
            }
            GGML_ASSERT(compute(be, g.ops));   // also the warm-up (pipeline compile)
            const int64_t nr = c.n_tps*c.n_stream;
            std::vector<int32_t> o0((size_t) (c.width*nr)), o((size_t) (c.width*nr));
            ggml_backend_tensor_get(g.ops[0], o0.data(), 0, o0.size()*sizeof(int32_t));
            size_t copies_differ = 0;
            for (int r = 1; r < R; r++) {
                ggml_backend_tensor_get(g.ops[r], o.data(), 0, o.size()*sizeof(int32_t));
                copies_differ += o != o0;
            }
            size_t bad_rows = 0;
            int64_t first_bad = -1;
            for (int64_t row = 0; row < nr; row++) {
                reference_row(in, c, row, keys, tmp, ref);
                if (memcmp(ref.data(), o0.data() + row*c.width, c.width*sizeof(int32_t)) != 0) {
                    bad_rows++;
                    if (first_bad < 0) { first_bad = row; }
                }
            }
            // warm-up until 150 ms of this graph ran (the GPU clock ramps after a lighter case or the process start)
            for (auto tw = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - tw < std::chrono::milliseconds(150); ) {
                GGML_ASSERT(compute(be, g.ops));
            }
            std::vector<double> t;
            for (int rep = 0; rep < reps; rep++) {
                const auto t0 = std::chrono::steady_clock::now();
                GGML_ASSERT(compute(be, g.ops));
                t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()/R);
            }
            std::sort(t.begin(), t.end());
            char path[512];
            snprintf(path, sizeof(path), "%s/case%03zu.ids", dir.c_str(), i);
            FILE * f = fopen(path, "wb");
            GGML_ASSERT(f);
            fwrite(o0.data(), sizeof(int32_t), o0.size(), f);
            fclose(f);
            printf("[op] case %zu %s exact=%d bad_rows=%zu first_bad=%" PRId64 " copies_differ=%zu ms_median=%.4f ms_min=%.4f reps=%d\n",
                   i, c.desc().c_str(), (int) (bad_rows == 0 && copies_differ == 0), bad_rows, first_bad, copies_differ,
                   t.empty() ? 0.0 : t[t.size()/2], t.empty() ? 0.0 : t.front(), reps);
        }
        ggml_backend_free(be);
        printf("[op] end %zu\n", i);
    }
    return 0;
}

struct case_view {
    bool        seen = false, exact = false, unsupported = false;
    std::string path;              // the QSA_TOP_K dispatch path of the case (first line)
    double      ms = 0.0;
    uint64_t    rows = 0, fb = 0, under = 0, over = 0;   // onepass counters, this case's delta
    double      cand = 0.0, cand_sum = 0.0;
};

struct arm_view {
    std::vector<case_view> c;
    bool rc0 = false, negctl = false, skip = false;
    std::string proof;
    size_t n_onepass = 0;
};

static arm_view run_arm(const char * self, const std::string & env, const std::string & dir, int reps, size_t n_cases) {
    arm_view a;
    a.c.resize(n_cases);
    const std::string cmd = "rm -rf '" + dir + "' && mkdir -p '" + dir + "' && " + env + " GGML_VK_DISPATCH_LOG=1 '" + self +
                            "' --child '" + dir + "' " + std::to_string(reps) + " 2>&1";
    FILE * p = popen(cmd.c_str(), "r");
    GGML_ASSERT(p);
    char buf[4096];
    size_t cur = 0;
    uint64_t prev_rows = 0, prev_u = 0, prev_o = 0;
    double prev_cand_tot = 0.0;
    while (fgets(buf, sizeof(buf), p)) {
        const std::string l(buf);
        size_t idx;
        if (l.find("[op] SKIP") != std::string::npos) { a.skip = true; }
        if (l.find("qsa_topk: onepass ") != std::string::npos) { a.proof = l; }
        if (l.find("NEGATIVE-CONTROL BUILD: the onepass") != std::string::npos) { a.negctl = true; }
        if (sscanf(l.c_str(), "[op] begin %zu", &idx) == 1 && idx < n_cases) {
            cur = idx;
        }
        if (sscanf(l.c_str(), "[op] case %zu", &idx) == 1 && idx < n_cases) {
            case_view & v = a.c[idx];
            v.seen = true;
            v.unsupported = l.find(" unsupported") != std::string::npos;
            v.exact = l.find(" exact=1 ") != std::string::npos;
            const size_t pm = l.find("ms_median=");
            if (pm != std::string::npos) { v.ms = atof(l.c_str() + pm + 10); }
        }
        if (l.rfind("[vk-dispatch] op=QSA_TOP_K ", 0) == 0) {
            const size_t pp = l.find(" path="), pn = l.find(" node=opcase");
            if (pp != std::string::npos && pn != std::string::npos) {
                const size_t ci = (size_t) atoll(l.c_str() + pn + 12);
                if (ci < n_cases && a.c[ci].path.empty()) {
                    a.c[ci].path = l.substr(pp + 6, l.find(' ', pp + 6) - pp - 6);
                    a.n_onepass += a.c[ci].path.rfind("onepass", 0) == 0;
                }
            }
        }
        // ggml_vulkan: qsa_topk onepass: rows N, fallback F (x %: under U, over O), candidates/row C (final)
        unsigned long long rows, fb, u, o;
        double pct, cand;
        const size_t ps = l.find("qsa_topk onepass: rows ");
        if (ps != std::string::npos &&
            sscanf(l.c_str() + ps, "qsa_topk onepass: rows %llu, fallback %llu (%lf %%: under %llu, over %llu), candidates/row %lf",
                   &rows, &fb, &pct, &u, &o, &cand) == 6) {
            const double cand_tot = cand*(double) (rows - u - o);
            case_view & v = a.c[cur];
            v.rows     += rows - prev_rows;
            v.under    += u - prev_u;
            v.over     += o - prev_o;
            v.fb        = v.under + v.over;
            v.cand_sum += cand_tot - prev_cand_tot;
            const uint64_t done = v.rows - v.fb;
            v.cand      = done ? v.cand_sum/(double) done : 0.0;
            prev_rows = rows; prev_u = u; prev_o = o; prev_cand_tot = cand_tot;
        }
    }
    a.rc0 = pclose(p) == 0;
    return a;
}

static std::vector<char> slurp(const std::string & f) {
    std::vector<char> v;
    FILE * p = fopen(f.c_str(), "rb");
    if (!p) { return v; }
    char b[1 << 16];
    size_t n;
    while ((n = fread(b, 1, sizeof(b), p)) > 0) { v.insert(v.end(), b, b + n); }
    fclose(p);
    return v;
}

int main(int argc, char ** argv) {
    if (argc > 2 && strcmp(argv[1], "--child") == 0) {
        return child(argv[2], argc > 3 ? atoi(argv[3]) : 9);
    }
    const int reps = argc > 1 ? atoi(argv[1]) : 9;
    const char * tmpd = getenv("TMPDIR");
    const std::string base = std::string(tmpd && *tmpd ? tmpd : "/tmp") + "/test-qsa-topk-onepass-" + std::to_string((long long) getpid());
    const auto cases = make_cases();
    const char * arms[3][2] = {
        { "off",    "GGML_VK_QSA_TOPK_ONEPASS=0" },
        { "on",     "GGML_VK_QSA_TOPK_ONEPASS=1 GGML_VK_QSA_TOPK_ONEPASS_BACKOFF=0" },
        { "on_all", "GGML_VK_QSA_TOPK_ONEPASS=1 GGML_VK_QSA_TOPK_ONEPASS_MIN_KV=1 GGML_VK_QSA_TOPK_ONEPASS_MIN_CELLS=1 GGML_VK_QSA_TOPK_ONEPASS_BACKOFF=0" },
    };
    std::vector<arm_view> v;
    for (const auto & arm : arms) {
        v.push_back(run_arm(argv[0], arm[1], base + "-" + arm[0], reps, cases.size()));
        if (v.back().skip) { printf("test-qsa-topk-onepass: SKIP (no GPU)\n"); return 0; }
        printf("[op-arm] %s rc0=%d onepass_cases=%zu proof: %s", arm[0], (int) v.back().rc0, v.back().n_onepass,
               v.back().proof.empty() ? "(none)\n" : v.back().proof.c_str());
    }

    bool ok = true;
    size_t n_same = 0, n_diff = 0, n_missing = 0;
    std::vector<bool> differs(cases.size(), false);
    for (size_t i = 0; i < cases.size(); ++i) {
        char name[32];
        snprintf(name, sizeof(name), "/case%03zu.ids", i);
        const auto d0 = slurp(base + "-off" + name);
        for (int a = 1; a < 3; ++a) {
            const auto d = slurp(base + "-" + arms[a][0] + name);
            if (d0.empty() || d.empty()) { n_missing++; continue; }
            if (d == d0) { n_same++; } else { n_diff++; differs[i] = true; }
        }
    }
    for (const auto & arm : arms) {
        const int rc = system(("rm -rf '" + base + "-" + arm[0] + "'").c_str());
        GGML_UNUSED(rc);
    }

    printf("%-4s %-62s %-15s %9s %9s %9s %7s %8s %8s %9s\n", "case", "shape", "path(on_all)", "v2_ms", "on_ms", "on_all_ms", "x(all)",
           "fb_on", "fb_all", "cand/row");
    for (size_t i = 0; i < cases.size(); ++i) {
        const case_view & a = v[0].c[i], & b = v[1].c[i], & c = v[2].c[i];
        printf("%-4zu %-62s %-15s %9.4f %9.4f %9.4f %7.2f %8.4f %8.4f %9.1f%s%s\n", i, cases[i].desc().c_str(),
               c.path.empty() ? "-" : c.path.c_str(), a.ms, b.ms, c.ms, c.ms > 0 ? a.ms/c.ms : 0.0,
               b.rows ? (double) b.fb/(double) b.rows : 0.0, c.rows ? (double) c.fb/(double) c.rows : 0.0, c.cand,
               (a.exact && b.exact && c.exact) ? "" : "  NOT-EXACT", differs[i] ? "  DUMP-DIFFERS" : "");
    }
    printf("[op-dump] %zu comparisons: %zu identical, %zu differ, %zu missing\n", n_same + n_diff + n_missing, n_same, n_diff, n_missing);

    const bool negctl = v[1].negctl || v[2].negctl;
    if (negctl) {
        // the control holds when the off arm is exact everywhere, and the on_all arm fails on >= 90 % of its onepass cases
        // and on no other case
        size_t n_op = 0, n_op_fail = 0, n_other_fail = 0;
        bool off_ok = v[0].rc0;
        for (size_t i = 0; i < cases.size(); ++i) {
            off_ok = off_ok && v[0].c[i].seen && (v[0].c[i].exact || v[0].c[i].unsupported);
            const bool op = v[2].c[i].path.rfind("onepass", 0) == 0;
            const bool fail = !v[2].c[i].exact;
            // a onepass case whose rows all fell back runs v2 for every row: it must stay exact
            const bool all_fb = v[2].c[i].rows > 0 && v[2].c[i].fb == v[2].c[i].rows;
            if (op && !all_fb) { n_op++; n_op_fail += fail; } else { n_other_fail += fail; }
        }
        const bool cok = off_ok && n_op > 0 && n_op_fail*10 >= n_op*9 && n_other_fail == 0;
        printf("test-qsa-topk-onepass: NEGATIVE CONTROL %s (highest-id ties: %zu/%zu onepass cases FAIL, %zu other cases FAIL, "
               "off arm exact %d)\n", cok ? "PASS" : "FAIL", n_op_fail, n_op, n_other_fail, (int) off_ok);
        return cok ? 0 : 1;
    }

    // every arm exact on every case
    for (size_t a = 0; a < 3; ++a) {
        ok = ok && v[a].rc0;
        for (size_t i = 0; i < cases.size(); ++i) {
            const case_view & c = v[a].c[i];
            if (!c.seen || (!c.exact && !c.unsupported)) { ok = false; printf("  %s case %zu not exact / not seen\n", arms[a][0], i); }
        }
    }
    ok = ok && n_diff == 0 && n_missing == 0 && n_same == 2*cases.size();
    // the knob decides: off never onepass; on_all onepass wherever admissible (k <= 4096 <= n_kv, expected candidates fit);
    // on = on_all at n_kv >= the default gate. Forced cases fall back on every row (equal / coarse / sparse) or on the odd rows.
    size_t n_fb_forced = 0, n_fb_forced_ok = 0;
    for (size_t i = 0; i < cases.size(); ++i) {
        const op_case & c = cases[i];
        const bool op0 = v[0].c[i].path.rfind("onepass", 0) == 0;
        const bool op2 = v[2].c[i].path.rfind("onepass", 0) == 0;
        if (op0) { ok = false; printf("  off arm dispatched onepass on case %zu\n", i); }
        if (!op2 && c.width <= c.n_kv) { ok = false; printf("  on_all arm did not dispatch onepass on case %zu (%s)\n", i, v[2].c[i].path.c_str()); }
        if (op2 && (c.mode == M_EQUAL || c.mode == M_COARSE || c.mode == M_SPARSE)) {
            n_fb_forced++;
            n_fb_forced_ok += v[2].c[i].rows > 0 && v[2].c[i].fb == v[2].c[i].rows;
        }
        if (op2 && c.mode == M_MIXED) {
            n_fb_forced++;
            // the odd rows (all equal) fall back, the even ones need not
            n_fb_forced_ok += v[2].c[i].rows > 0 && v[2].c[i].fb*2 >= v[2].c[i].rows && v[2].c[i].fb < v[2].c[i].rows;
        }
    }
    ok = ok && n_fb_forced > 0 && n_fb_forced_ok == n_fb_forced && v[1].n_onepass > 0 && v[0].n_onepass == 0;
    printf("test-qsa-topk-onepass: %s (%zu cases x 3 arms exact vs the contract reference; on / on_all dumps byte-identical to off "
           "%zu/%zu; onepass cases on %zu, on_all %zu; forced-fallback cases behaving %zu/%zu)\n", ok ? "PASS" : "FAIL", cases.size(),
           n_same, 2*cases.size(), v[1].n_onepass, v[2].n_onepass, n_fb_forced_ok, n_fb_forced);
    return ok ? 0 : 1;
}
