// [TAG_QSA_SCORE_KREUSE] (Program v8.1 A2b, contrib 0241) GGML_OP_QSA_SCORE with the pooled keys reused across the query
// tiles of a chunk (the kr twins: workgroups numbered key tile major), knob off vs on, bit for bit, on a GPU. One child per
// arm (GGML_VK_QSA_SCORE_KREUSE is read once at device init):
//   off: GGML_VK_QSA_SCORE_KREUSE=0 (the A2a dispatch order)    on: =1 (the default)
//   lim4: =1 with GGML_VK_QSA_SCORE_KREUSE_MAX_WG=4 (test hook): planes of more than 4 workgroups take the fail-closed
//         fallback to the A2a dispatch, the rest the kr twin
// Each child runs the same case list - the three forms (pooled keys, store + remap, store + remap + f16 mirror), decode /
// verify widths 1..9 and chunk sizes 33..512 queries (ragged N tails: 100, 127, 227 queries = partial last N tiles), ragged
// and small M (17 / 33 rows: the s tile), K 128 (aligned), 72 and 192 (unaligned), two streams, 8 heads, bias none / the
// QSA_BIAS constants, f16-denormal and large keys - and writes every score plane to a file.
// PASS: the on and lim4 arms' planes are byte-identical to the off arm's (and the supported / unsupported pattern is the
// same), the off arm dispatched no kr pipeline, the on arm did (the dispatch log's path=..._kr, matched to the case by the
// node name), widths 1..2 were refused (mat-vec, not QSA_SCORE), widths 3..9 took kr only at M <= 32 (width 9 at M 17 must:
// two N tiles of the s tile), lim4 kept kr on some cases and fell back on others (M 65536 x 128 queries must fall back),
// no output kept the junk the planes start with.
// Negative control: in a -DGGML_VULKAN_QSA_KREUSE_NEGCTL=ON build (the kr twins sum the TK steps of each K block in reverse;
// the proof line says NEGATIVE-CONTROL BUILD) the same run must find every kr case different and every other case equal,
// and prints NEGATIVE CONTROL PASS.
// Not a ctest: needs a Vulkan coopmat1 device whose QSA_SCORE runs; without one it prints SKIP and exits 0.
// usage: test-qsa-score-kreuse                      (parent: runs the arms, compares)
//        test-qsa-score-kreuse --child <file>       (one arm, the env decides)
//        test-qsa-score-kreuse --perf [reps [M ...]] (one arm: ms per QSA_SCORE call at the deployed prefill chunk shape,
//                                                   M = pooled blocks = depth / 4; default 16384 .. 262144)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

struct kr_case {
    int64_t K, M, Q, n_b;   // Q = queries: the product has n_head*Q columns
    int     n_head;
    int     form;           // 0 pooled keys (MUL_MAT form), 1 store + remap, 2 store + remap + f16 mirror
    int     bias;           // 0 none, 1 the QSA_BIAS constants (+0, 1e9, -inf)
    float   scale;          // the keys x scale (1e-5: f16 denormals in the mirror)

    std::string desc() const {
        char b[160];
        snprintf(b, sizeof(b), "K=%" PRId64 " M=%" PRId64 " Q=%" PRId64 " n_b=%" PRId64 " H=%d form=%s bias=%d scale=%g", K, M, Q, n_b,
                 n_head, form == 0 ? "pooled" : (form == 1 ? "rows" : "rows_f16a"), bias, scale);
        return b;
    }
};

static std::vector<kr_case> make_cases() {
    std::vector<kr_case> v;
    const int64_t widths[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 64, 100, 127, 128, 227, 256 };
    for (int form = 0; form < 3; ++form) {
        for (int64_t M : { 17, 1000, 4133, 16384 + 37 }) {
            for (int64_t Q : widths) {
                v.push_back({ 128, M, Q, 1, 4, form, 1, 1.0f });
            }
        }
        for (int64_t Q : { 9, 100, 128 }) {
            v.push_back({ 128, 65536, Q, 1, 4, form, 1, 1.0f });
        }
        v.push_back({ 128, 33, 512, 1, 4, form, 1, 1.0f });
        for (int64_t Q : { 9, 64, 128 }) {
            v.push_back({ 72, 777, Q, 1, 4, form, 1, 1.0f });
        }
        v.push_back({ 192, 1000, 128, 1, 4, form, 1, 1.0f });
        for (int64_t Q : { 37, 128 }) {
            v.push_back({ 128, 4133, Q, 2, 4, form, 1, 1.0f });
        }
        v.push_back({ 128, 1000, 64, 1, 8, form, 1, 1.0f });
        v.push_back({ 128, 1000, 128, 1, 4, form, 0, 1.0f });
    }
    v.push_back({ 128, 1000, 128, 1, 4, 2, 1, 1e-5f });
    v.push_back({ 128, 1000, 128, 1, 4, 2, 1, 1e3f });
    return v;
}

static uint64_t fnv(const void * p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    for (size_t i = 0; i < n; ++i) { h = (h ^ ((const uint8_t *) p)[i]) * 0x100000001b3ull; }
    return h;
}

static ggml_backend_t gpu_backend() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); }
    return dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
}

// the case's tensors and op (and, for the f16 form, the mirror's set_rows); data as test-backend-ops' QSA_SCORE cases
struct kr_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * a = nullptr, * ids = nullptr, * b = nullptr, * bi = nullptr, * a16 = nullptr, * all = nullptr, * fill16 = nullptr;
    std::vector<ggml_tensor *> ops;

    ~kr_graph() {
        if (buf) { ggml_backend_buffer_free(buf); }
        if (ctx) { ggml_free(ctx); }
    }
};

static int64_t n_rows_of(const kr_case & c) { return c.n_b*(c.M + c.M/8 + 1); }

static bool build(kr_graph & g, ggml_backend_t be, const kr_case & c, int n_ops, const char * name) {
    ggml_init_params ip = { (size_t) (64 + 4*n_ops)*ggml_tensor_overhead(), nullptr, true };
    g.ctx = ggml_init(ip);
    g.b = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, c.K, c.n_head*c.Q, c.n_b);
    const bool rows = c.form != 0;
    g.a = rows ? ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, c.K, n_rows_of(c)) : ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, c.K, c.M, c.n_b);
    if (rows) { g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, c.M, c.n_b); }
    if (c.bias) { g.bi = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, c.M, c.Q, c.n_b); }
    if (c.form == 2) {
        g.a16 = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, c.K, n_rows_of(c));
        g.all = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, n_rows_of(c));
        g.fill16 = ggml_set_rows(g.ctx, g.a16, g.a, g.all);
    }
    for (int i = 0; i < n_ops; ++i) {
        ggml_tensor * op = ggml_qsa_score(g.ctx, g.a, g.ids, g.b, g.bi, c.n_head);
        if (c.form == 2) { ggml_qsa_score_set_a_f16(op, g.a16); }
        ggml_format_name(op, "%s_%d", name, i);
        g.ops.push_back(op);
    }
    if (!ggml_backend_supports_op(be, g.ops[0])) {
        return false;
    }
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    GGML_ASSERT(g.buf);

    std::mt19937 rng(0x5eed1234u ^ (uint32_t) (c.K*31 + c.M*131 + c.Q*17 + c.n_b*7 + c.n_head*1009 + c.form*3 + c.bias*5));
    std::normal_distribution<float> nrm(0.0f, 1.0f);
    auto unit_rows = [&](ggml_tensor * t, float sc) {
        const int64_t n = ggml_nelements(t);
        std::vector<float> v(n);
        for (int64_t r = 0; r < n/c.K; r++) {
            double ss = 0.0;
            for (int64_t k = 0; k < c.K; k++) { v[r*c.K + k] = nrm(rng); ss += (double) v[r*c.K + k]*v[r*c.K + k]; }
            const float f = (ss > 0.0 ? (float) (1.0/std::sqrt(ss)) : 1.0f)*sc;
            for (int64_t k = 0; k < c.K; k++) { v[r*c.K + k] *= f; }
        }
        ggml_backend_tensor_set(t, v.data(), 0, n*sizeof(float));
    };
    unit_rows(g.a, c.scale);
    unit_rows(g.b, 1.0f);
    if (g.ids) {
        // a near-identity remap with 1 % of the blocks moved, per stream
        const int64_t rps = c.M + c.M/8 + 1;
        std::vector<int32_t> v(c.M*c.n_b);
        for (int64_t s = 0; s < c.n_b; s++) {
            for (int64_t m = 0; m < c.M; m++) {
                const int64_t r = (rng() % 100 == 0) ? (int64_t) (rng() % (uint32_t) rps) : m;
                v[s*c.M + m] = (int32_t) (s*rps + r);
            }
        }
        ggml_backend_tensor_set(g.ids, v.data(), 0, v.size()*sizeof(int32_t));
    }
    if (g.bi) {
        std::vector<float> v(ggml_nelements(g.bi));
        for (auto & x : v) { const uint32_t k = rng() % 8; x = k == 0 ? -INFINITY : (k == 1 ? 1e9f : 0.0f); }
        ggml_backend_tensor_set(g.bi, v.data(), 0, v.size()*sizeof(float));
    }
    if (g.all) {
        std::vector<int32_t> v(ggml_nelements(g.all));
        for (size_t i = 0; i < v.size(); i++) { v[i] = (int32_t) i; }
        ggml_backend_tensor_set(g.all, v.data(), 0, v.size()*sizeof(int32_t));
        std::vector<ggml_fp16_t> j(ggml_nelements(g.a16), ggml_fp32_to_fp16(-7.0f));
        ggml_backend_tensor_set(g.a16, j.data(), 0, j.size()*sizeof(ggml_fp16_t));
    }
    // the outputs start as a recognisable junk value, so an output the kernel skips cannot pass as computed
    for (ggml_tensor * op : g.ops) {
        std::vector<float> j(ggml_nelements(op), -3.25f);
        ggml_backend_tensor_set(op, j.data(), 0, j.size()*sizeof(float));
    }
    return true;
}

static bool compute(ggml_context * ctx, ggml_backend_t be, const std::vector<ggml_tensor *> & nodes) {
    GGML_UNUSED(ctx);
    const size_t size = 64 + nodes.size();
    ggml_init_params ip = { ggml_graph_overhead_custom(size, false) + 1024, nullptr, true };
    ggml_context * gctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, size, false);
    for (ggml_tensor * n : nodes) { ggml_build_forward_expand(gf, n); }
    const bool ok = ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    ggml_free(gctx);
    return ok;
}

static int child(const char * path) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    ggml_backend_load_all();
    ggml_backend_t be = gpu_backend();
    if (!be) { printf("[kr] SKIP no GPU\n"); return 0; }
    FILE * out = fopen(path, "wb");
    GGML_ASSERT(out);
    const auto cases = make_cases();
    for (size_t i = 0; i < cases.size(); ++i) {
        const kr_case & c = cases[i];
        kr_graph g;
        char name[32]; snprintf(name, sizeof(name), "krcase%zu", i);
        if (!build(g, be, c, 1, name)) {
            printf("[kr] case %zu %s unsupported\n", i, c.desc().c_str());
            continue;
        }
        if (g.fill16) { GGML_ASSERT(compute(g.ctx, be, { g.fill16 })); }
        GGML_ASSERT(compute(g.ctx, be, g.ops));
        std::vector<float> o(ggml_nelements(g.ops[0]));
        ggml_backend_tensor_get(g.ops[0], o.data(), 0, o.size()*sizeof(float));
        size_t n_junk = 0, n_pos = 0;
        for (float x : o) { n_junk += x == -3.25f; n_pos += x > 0.0f; }
        fwrite(o.data(), sizeof(float), o.size(), out);
        printf("[kr] case %zu %s fnv=%016" PRIx64 " junk=%zu pos=%zu\n", i, c.desc().c_str(), fnv(o.data(), o.size()*sizeof(float)),
               n_junk, n_pos);
    }
    fclose(out);
    ggml_backend_free(be);
    return 0;
}

// ms per QSA_SCORE call at the deployed prefill shape: K 128, 4 heads, one 128-query chunk (N 512), store + remap + the f16
// mirror, the QSA_BIAS constants; 16 calls per graph (the 16 chunks of a 2048 ubatch), the mirror filled once outside
static int perf(int reps, std::vector<int64_t> Ms) {
    ggml_backend_load_all();
    ggml_backend_t be = gpu_backend();
    if (!be) { printf("[kr-perf] SKIP no GPU\n"); return 0; }
    if (Ms.empty()) {
        Ms = { 16384, 32768, 49152, 65536, 98304, 131072, 196608, 262144 };
    }
    for (int64_t M : Ms) {
        for (int form : { 2, 1 }) {
            const kr_case c = { 128, M, 128, 1, 4, form, 1, 1.0f };
            kr_graph g;
            if (!build(g, be, c, 16, "krperf")) { printf("[kr-perf] %s unsupported\n", c.desc().c_str()); continue; }
            if (g.fill16) { GGML_ASSERT(compute(g.ctx, be, { g.fill16 })); }
            GGML_ASSERT(compute(g.ctx, be, g.ops));   // warm-up (pipeline compile)
            std::vector<double> t;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                GGML_ASSERT(compute(g.ctx, be, g.ops));
                t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count()/16.0);
            }
            std::sort(t.begin(), t.end());
            printf("[kr-perf] %s ms_per_call median=%.4f min=%.4f max=%.4f reps=%d\n", c.desc().c_str(), t[t.size()/2], t.front(),
                   t.back(), reps);
        }
    }
    ggml_backend_free(be);
    return 0;
}

static std::vector<std::string> run_child(const char * self, const std::string & env, const std::string & file) {
    const std::string cmd = env + " '" + self + "' --child '" + file + "' 2>&1";
    std::vector<std::string> lines;
    FILE * p = popen(cmd.c_str(), "r");
    GGML_ASSERT(p);
    char buf[1024];
    while (fgets(buf, sizeof(buf), p)) { lines.emplace_back(buf); }
    const int rc = pclose(p);
    lines.push_back("rc=" + std::to_string(rc) + "\n");
    return lines;
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

struct arm_view {
    std::vector<std::string> cases;          // the [kr] case lines, in order
    std::map<size_t, std::string> path;      // case -> the QSA_SCORE dispatch path
    size_t n_kr = 0;
    bool   rc0  = false;
};

static arm_view view(const std::vector<std::string> & lines) {
    arm_view a;
    for (const auto & l : lines) {
        if (l.rfind("[kr] case ", 0) == 0) {
            a.cases.push_back(l);
        } else if (l.rfind("[vk-dispatch] op=QSA_SCORE ", 0) == 0) {
            const size_t pp = l.find(" path="), pn = l.find(" node=krcase");
            if (pp != std::string::npos && pn != std::string::npos) {
                const std::string path = l.substr(pp + 6, l.find(' ', pp + 6) - pp - 6);
                const size_t idx = (size_t) atoll(l.c_str() + pn + 12);
                a.path[idx] = path;
                a.n_kr += path.size() > 3 && path.compare(path.size() - 3, 3, "_kr") == 0;
            }
        } else if (l == "rc=0\n") {
            a.rc0 = true;
        }
    }
    return a;
}

int main(int argc, char ** argv) {
    if (argc > 2 && strcmp(argv[1], "--child") == 0) {
        return child(argv[2]);
    }
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        std::vector<int64_t> Ms;
        for (int i = 3; i < argc; ++i) { Ms.push_back(atoll(argv[i])); }
        return perf(argc > 2 ? atoi(argv[2]) : 7, Ms);
    }
    const char * tmp = getenv("TMPDIR");
    const std::string dir = tmp && *tmp ? tmp : "/tmp";
    const std::string tag = std::to_string((long long) getpid());
    // off / on / on with a 4-workgroup limit (the test hook: planes of more than 4 workgroups fall back to the A2a dispatch)
    const char * arms[3][2] = { { "off", "GGML_VK_QSA_SCORE_KREUSE=0" }, { "on", "GGML_VK_QSA_SCORE_KREUSE=1" },
                                { "lim4", "GGML_VK_QSA_SCORE_KREUSE=1 GGML_VK_QSA_SCORE_KREUSE_MAX_WG=4" } };
    std::vector<arm_view> v;
    std::vector<std::vector<char>> planes;
    bool negctl = false;
    for (const auto & arm : arms) {
        const std::string file = dir + "/test-qsa-score-kreuse-" + tag + "-" + arm[0] + ".bin";
        const auto lines = run_child(argv[0], std::string("GGML_VK_DISPATCH_LOG=1 ") + arm[1], file);
        if (!lines.empty() && lines[0].find("SKIP") != std::string::npos) { printf("test-qsa-score-kreuse: SKIP (no GPU)\n"); return 0; }
        v.push_back(view(lines));
        planes.push_back(slurp(file));
        remove(file.c_str());
        for (const auto & l : lines) {
            negctl = negctl || l.find("NEGATIVE-CONTROL BUILD") != std::string::npos;
            if (strcmp(arm[0], "on") == 0 && (l.rfind("[kr]", 0) == 0 || l.find("qsa_score_kreuse") != std::string::npos)) {
                fputs(l.c_str(), stdout);
            }
        }
    }

    const auto cases = make_cases();
    auto is_kr = [&](const arm_view & a, size_t i) {
        const auto it = a.path.find(i);
        return it != a.path.end() && it->second.size() > 3 && it->second.compare(it->second.size() - 3, 3, "_kr") == 0;
    };
    bool ok = true;
    for (const auto & a : v) { ok = ok && a.rc0 && a.cases.size() == cases.size(); }

    if (negctl) {
        // [negative control] a -DGGML_VULKAN_QSA_KREUSE_NEGCTL=ON build: the kr twins sum the TK steps of each K block in
        // reverse. The control holds when every kr case of the on arm differs from the off arm and no other case does
        size_t n_kr = 0, n_kr_diff = 0, n_other_diff = 0;
        for (size_t i = 0; i < cases.size() && i < v[1].cases.size() && i < v[0].cases.size(); ++i) {
            const bool diff = v[1].cases[i] != v[0].cases[i];
            if (is_kr(v[1], i)) { n_kr++; n_kr_diff += diff; } else { n_other_diff += diff; }
        }
        ok = ok && v[0].n_kr == 0 && n_kr > 0 && n_kr_diff == n_kr && n_other_diff == 0;
        printf("test-qsa-score-kreuse: NEGATIVE CONTROL %s (build with the kr summation order reversed: %zu/%zu kr cases differ, "
               "%zu other cases differ)\n", ok ? "PASS" : "FAIL", n_kr_diff, n_kr, n_other_diff);
        return ok ? 0 : 1;
    }

    size_t n_sup = 0, n_kr_cases = 0, n_lim_kr = 0, n_lim_fallback = 0;
    for (size_t i = 0; i < v[0].cases.size(); ++i) { n_sup += v[0].cases[i].find(" unsupported") == std::string::npos; }
    for (size_t a = 0; a < v.size(); ++a) {
        const bool same_lines = v[a].cases == v[0].cases;
        const bool same_bytes = planes[a] == planes[0] && !planes[0].empty();
        printf("[kr-arm] %s rc0=%d cases=%zu identical_lines=%d identical_bytes=%d (%zu B) kr_dispatches=%zu\n", arms[a][0], (int) v[a].rc0,
               v[a].cases.size(), (int) same_lines, (int) same_bytes, planes[a].size(), v[a].n_kr);
        ok = ok && same_lines && same_bytes;
        if (!same_lines) {
            for (size_t i = 0; i < std::min(v[a].cases.size(), v[0].cases.size()); ++i) {
                if (v[a].cases[i] != v[0].cases[i]) { printf("  first difference:\n    off: %s    %s: %s", v[0].cases[i].c_str(), arms[a][0], v[a].cases[i].c_str()); break; }
            }
        }
    }
    // the knob decides: none off, kr on; widths 1..2 never reach QSA_SCORE; no junk left
    ok = ok && v[0].n_kr == 0 && v[1].n_kr > 0;
    for (size_t i = 0; i < cases.size() && i < v[1].cases.size(); ++i) {
        const kr_case & c = cases[i];
        const bool unsup = v[1].cases[i].find(" unsupported") != std::string::npos;
        const bool kr    = is_kr(v[1], i);
        n_kr_cases += kr;
        if (c.Q <= 2 && c.n_head == 4 && !unsup) { printf("  width %" PRId64 " reached QSA_SCORE: %s", c.Q, v[1].cases[i].c_str()); ok = false; }
        if (!unsup && v[1].cases[i].find(" junk=0 ") == std::string::npos) { printf("  junk left: %s", v[1].cases[i].c_str()); ok = false; }
        if (!unsup && v[1].path.count(i) == 0) { printf("  no dispatch line for case %zu\n", i); ok = false; }
        // the widths 3..9 rule (ggml_vk_qsa_score_plan): one N tile - so the A2a dispatch - for M > 32; width 9 at M <= 32 is
        // two N tiles of the s tile (BN 32) and takes kr
        if (c.n_head == 4 && c.Q >= 3 && c.Q <= 9 && c.M > 32 && kr) { printf("  width %" PRId64 " at M %" PRId64 " took kr\n", c.Q, c.M); ok = false; }
        if (c.n_head == 4 && c.Q == 9 && c.M == 17 && c.K == 128 && !kr) { printf("  width 9 at M 17 (2 s-tile N tiles) did not take kr: case %zu\n", i); ok = false; }
        // the 4-workgroup limit: kr only where the plane needs <= 4 workgroups, the rest on the A2a dispatch (same bits, above)
        if (kr) {
            if (is_kr(v[2], i)) { n_lim_kr++; } else { n_lim_fallback++; }
        } else if (is_kr(v[2], i)) { printf("  lim4 took kr where on did not: case %zu\n", i); ok = false; }
        if (c.M == 65536 && c.Q == 128 && (!kr || is_kr(v[2], i))) { printf("  M 65536 x 128 queries: on kr=%d, lim4 kr=%d\n", (int) kr, (int) is_kr(v[2], i)); ok = false; }
    }
    ok = ok && n_lim_kr > 0 && n_lim_fallback > 0;
    printf("test-qsa-score-kreuse: %s (%zu cases, %zu on the GEMM path, %zu of them on the kr twin in the on arm; with the "
           "4-workgroup limit %zu stay kr and %zu fall back to the A2a dispatch; off / on / lim4 byte-identical planes)\n",
           ok ? "PASS" : "FAIL", cases.size(), n_sup, n_kr_cases, n_lim_kr, n_lim_fallback);
    return ok ? 0 : 1;
}
