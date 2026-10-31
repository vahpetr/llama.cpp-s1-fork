// [TAG_GDN_REPLAY] commit by replay of the gated delta net states (LLAMA_GDN_REPLAY, ggml_gated_delta_net_set_replay)
//
//   test-gdn-replay ops            - op level, on every backend: the replayed states and the window outputs against
//                                    the snapshot planes path of the same backend, byte for byte (memcmp); CPU vs the
//                                    other backends as NMSE
//   test-gdn-replay perf           - us per verification-shaped op on the GPU backends: K = 1, planes, replay
//   test-gdn-replay <common args>  - llama level with a real qwen4exp model (-m ...): two contexts, LLAMA_GDN_REPLAY=0
//                                    and =1, decode windows with a rollback depth -> seq_rm r -> decode ...; logits and
//                                    llama_state_seq_get_data bytes identical at every step, seq_cp, save -> load round
//                                    trips (both directions), for n_seq_max 1 and 5
//
// Exit code 0 = all identical.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------------
// op level
// ---------------------------------------------------------------------------------------------------------------------

namespace {

constexpr int64_t S  = 128; // head size
constexpr int64_t HK = 16;  // key heads
constexpr int64_t HV = 48;  // value heads
constexpr int64_t W  = 5;   // stash rows per half (n_rs_seq)
constexpr int64_t ROW = S * HK + S * HV + HV + HV;

struct window_data {
    std::vector<float> q, k, v, g, b; // [S, HK, n, ns], [S, HV, n, ns], [1, HV, n, ns]
};

static void fill_window(std::mt19937 & rng, window_data & w, int64_t n, int64_t ns) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f), ug(-2.0f, -0.01f), ub(0.05f, 0.95f);
    auto l2 = [&](std::vector<float> & x) {
        for (size_t i = 0; i < x.size(); i += S) {
            double s = 0;
            for (int64_t j = 0; j < S; ++j) { s += (double) x[i + j] * x[i + j]; }
            const float r = 1.0f / std::sqrt((float) s + 1e-6f);
            for (int64_t j = 0; j < S; ++j) { x[i + j] *= r; }
        }
    };
    w.q.resize(S * HK * n * ns); w.k.resize(S * HK * n * ns); w.v.resize(S * HV * n * ns);
    w.g.resize(HV * n * ns); w.b.resize(HV * n * ns);
    for (auto & x : w.q) { x = u(rng); }
    for (auto & x : w.k) { x = u(rng); }
    for (auto & x : w.v) { x = u(rng); }
    for (auto & x : w.g) { x = ug(rng); }
    for (auto & x : w.b) { x = ub(rng); }
    l2(w.q); l2(w.k);
}

struct op_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ~op_graph() {
        if (buf) { ggml_backend_buffer_free(buf); }
        if (ctx) { ggml_free(ctx); }
    }
};

static void set_f(ggml_tensor * t, const std::vector<float> & x) {
    GGML_ASSERT(ggml_nbytes(t) == x.size() * sizeof(float));
    ggml_backend_tensor_set(t, x.data(), 0, ggml_nbytes(t));
}

static std::vector<float> get_f(const ggml_tensor * t, size_t off = 0, size_t n = 0) {
    if (n == 0) { n = ggml_nelements(t); }
    std::vector<float> x(n);
    ggml_backend_tensor_get(t, x.data(), off * sizeof(float), n * sizeof(float));
    return x;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += (double) (a[i] - b[i]) * (a[i] - b[i]); den += (double) a[i] * a[i]; }
    return den > 0 ? num / den : num;
}

// results of one case on one backend, for the cross-backend comparison
struct case_out {
    std::vector<float> attn1, s1, s2, attn3, s3;
};

// one case: ns sequences, a window of n tokens with n_defer d, then per sequence a replay of p_s of the d kept tokens
//   (commit-only graph, and a second window of m tokens on top)
static bool run_case(ggml_backend_t be, int64_t ns, int64_t n, int64_t d, int64_t p, int64_t m, uint32_t seed, case_out & out) {
    std::mt19937 rng(seed);
    window_data w1, w2;
    fill_window(rng, w1, n, ns);
    fill_window(rng, w2, m, ns);
    std::vector<float> s0(S * S * HV * ns);
    {
        std::uniform_real_distribution<float> u(-0.1f, 0.1f);
        for (auto & x : s0) { x = u(rng); }
    }
    std::vector<int64_t> ps(ns), slot(ns), half(ns);
    for (int64_t s = 0; s < ns; ++s) {
        ps[s]   = (p + s) % (d + 1);
        slot[s] = (s * 3 + 1) % ns;
        half[s] = s % 2;
    }

    op_graph og;
    ggml_init_params ip = { 256 * ggml_tensor_overhead() + 8 * ggml_graph_overhead(), nullptr, true };
    og.ctx = ggml_init(ip);
    ggml_context * ctx = og.ctx;

    auto mk_win = [&](int64_t nt, ggml_tensor ** q, ggml_tensor ** k, ggml_tensor ** v, ggml_tensor ** g, ggml_tensor ** b) {
        *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, nt, ns);
        *k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, nt, ns);
        *v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HV, nt, ns);
        *g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, nt, ns);
        *b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, nt, ns);
    };
    ggml_tensor * q1, * k1, * v1, * g1, * b1, * q2, * k2, * v2, * g2, * b2, * qd, * kd, * vd, * gd, * bd;
    mk_win(n, &q1, &k1, &v1, &g1, &b1);
    mk_win(m, &q2, &k2, &v2, &g2, &b2);
    mk_win(1, &qd, &kd, &vd, &gd, &bd); // the commit graph's unused window

    ggml_tensor * st0   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, HV, ns);
    ggml_tensor * st_p  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, HV, ns); // reference start of window 2
    ggml_tensor * stash = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ROW, W, 2 * ns);
    ggml_tensor * S1    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S * S * HV, ns, 1);
    ggml_tensor * S2    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S * S * HV, ns, 1);
    ggml_tensor * S3    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S * S * HV, ns, 1);
    ggml_tensor * c1    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, ns);
    ggml_tensor * c2    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, ns);
    ggml_tensor * c3    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, ns);

    // A: the planes path, K = n (plane j = the state j tokens back); A3: window 2 from the reference start, K = 1
    ggml_tensor * A  = ggml_gated_delta_net(ctx, q1, k1, v1, g1, b1, st0, n);
    ggml_tensor * A3 = ggml_gated_delta_net(ctx, q2, k2, v2, g2, b2, st_p, 1);
    // B1: the deferred window; B2: commit only from S1; B3: replay + window 2 from S1
    auto as_state = [&](ggml_tensor * t) { return ggml_reshape_4d(ctx, t, S, S, HV, ns); };
    ggml_tensor * B1 = ggml_gated_delta_net_ext(ctx, q1, k1, v1, g1, b1, st0, S1);
    ggml_gated_delta_net_set_replay(B1, stash, c1);
    ggml_tensor * B2 = ggml_gated_delta_net_ext(ctx, qd, kd, vd, gd, bd, as_state(S1), S2);
    ggml_gated_delta_net_set_replay(B2, stash, c2);
    ggml_tensor * B3 = ggml_gated_delta_net_ext(ctx, q2, k2, v2, g2, b2, as_state(S1), S3);
    ggml_gated_delta_net_set_replay(B3, stash, c3);

    og.buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!og.buf) {
        fprintf(stderr, "ops: allocation failed on %s\n", ggml_backend_name(be));
        return false;
    }
    ggml_backend_buffer_clear(og.buf, 0);
    set_f(q1, w1.q); set_f(k1, w1.k); set_f(v1, w1.v); set_f(g1, w1.g); set_f(b1, w1.b);
    set_f(q2, w2.q); set_f(k2, w2.k); set_f(v2, w2.v); set_f(g2, w2.g); set_f(b2, w2.b);
    set_f(st0, s0);
    {
        std::vector<int32_t> a(4 * ns), b(4 * ns), c(4 * ns);
        for (int64_t s = 0; s < ns; ++s) {
            const int32_t sl = (int32_t) slot[s], h = (int32_t) half[s];
            a[4*s+0] = 0;              a[4*s+1] = h;     a[4*s+2] = (int32_t) d; a[4*s+3] = sl;
            b[4*s+0] = (int32_t) ps[s]; b[4*s+1] = 1 - h; b[4*s+2] = -1;         b[4*s+3] = sl;
            c[4*s+0] = (int32_t) ps[s]; c[4*s+1] = 1 - h; c[4*s+2] = 0;          c[4*s+3] = sl;
        }
        ggml_backend_tensor_set(c1, a.data(), 0, a.size() * sizeof(int32_t));
        ggml_backend_tensor_set(c2, b.data(), 0, b.size() * sizeof(int32_t));
        ggml_backend_tensor_set(c3, c.data(), 0, c.size() * sizeof(int32_t));
    }

    auto compute = [&](ggml_tensor * t) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t);
        return ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS;
    };
    if (!compute(A) || !compute(B1) || !compute(B2)) {
        fprintf(stderr, "ops: compute failed on %s\n", ggml_backend_name(be));
        return false;
    }

    const size_t SD = S * S * HV;          // one state
    const size_t NA = S * HV * n * ns;     // attention of window 1
    auto plane = [&](int64_t j, int64_t s) -> std::vector<float> {
        if (j == n) {
            return std::vector<float>(s0.begin() + s * SD, s0.begin() + (s + 1) * SD);
        }
        return get_f(A, NA + (size_t) j * SD * ns + s * SD, SD);
    };

    bool ok = true;
    auto check = [&](const char * what, const std::vector<float> & x, const std::vector<float> & y) {
        if (x.size() != y.size() || memcmp(x.data(), y.data(), x.size() * sizeof(float)) != 0) {
            size_t nd = 0;
            for (size_t i = 0; i < std::min(x.size(), y.size()); ++i) { nd += memcmp(&x[i], &y[i], sizeof(float)) != 0; }
            fprintf(stderr, "ops: %s ns=%lld n=%lld d=%lld p=%lld m=%lld: %s differs (%zu of %zu floats)\n", ggml_backend_name(be),
                    (long long) ns, (long long) n, (long long) d, (long long) p, (long long) m, what, nd, x.size());
            ok = false;
        }
    };

    out.attn1 = get_f(B1, 0, NA);
    check("window attn (B1 vs planes)", out.attn1, get_f(A, 0, NA));
    out.s1 = get_f(S1);
    out.s2 = get_f(S2);
    std::vector<float> ref_start(SD * ns);
    for (int64_t s = 0; s < ns; ++s) {
        const auto pd = plane(d, s);
        const auto pp = plane(d - ps[s], s);
        check("deferred state (B1 vs plane d)", std::vector<float>(out.s1.begin() + s * SD, out.s1.begin() + (s + 1) * SD), pd);
        check("committed state (B2 vs plane d-p)", std::vector<float>(out.s2.begin() + s * SD, out.s2.begin() + (s + 1) * SD), pp);
        std::copy(pp.begin(), pp.end(), ref_start.begin() + s * SD);
    }

    // window 2: replay inside the next window vs the planes path from plane d - p
    set_f(st_p, ref_start);
    if (!compute(A3) || !compute(B3)) {
        fprintf(stderr, "ops: compute failed on %s\n", ggml_backend_name(be));
        return false;
    }
    const size_t NA3 = S * HV * m * ns;
    out.attn3 = get_f(B3, 0, NA3);
    out.s3    = get_f(S3);
    check("replay+window attn (B3 vs planes)", out.attn3, get_f(A3, 0, NA3));
    check("replay+window state (B3 vs planes)", out.s3, get_f(A3, NA3, SD * ns));
    return ok;
}

// a chain of rounds on one backend, as a verification sequence: window r (n_r tokens, n_defer D_r = min(n_r - 1, W)) replays
//   the p_{r-1} tokens kept by the previous one and defers its own tail; per sequence its own p (distinct prefixes),
//   halves alternate. Oracle = the planes path of the same backend (K = n_r) from the reference start, memcmp: window
//   attn, deferred state = plane D_r; the next reference start = plane D_r - p_r. Then a commit-only op (n_next = 0)
static bool run_chain(ggml_backend_t be, int64_t ns, const std::vector<int64_t> & wins, const std::vector<int64_t> & keep,
                      int64_t p_final, uint32_t seed) {
    std::mt19937 rng(seed);
    const size_t SD = S * S * HV;
    std::vector<float> ref(SD * ns);
    {
        std::uniform_real_distribution<float> u(-0.1f, 0.1f);
        for (auto & x : ref) { x = u(rng); }
    }
    std::vector<int64_t> pk(ns, 0), half(ns), slot(ns);
    for (int64_t s = 0; s < ns; ++s) { half[s] = s % 2; slot[s] = (s * 3 + 1) % ns; }
    std::vector<float> live = ref; // the replay path's cell row (deferred), host copy between rounds
    bool ok = true;
    const size_t R = wins.size();
    for (size_t r = 0; r <= R; ++r) {
        const bool fin = r == R;
        const int64_t n = fin ? 1 : wins[r];
        const int64_t d = fin ? -1 : std::min<int64_t>(n - 1, W);
        window_data w;
        fill_window(rng, w, n, ns);

        op_graph og;
        ggml_init_params ip = { 64 * ggml_tensor_overhead() + 4 * ggml_graph_overhead(), nullptr, true };
        og.ctx = ggml_init(ip);
        ggml_context * ctx = og.ctx;
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, n, ns);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, n, ns);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HV, n, ns);
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, n, ns);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, n, ns);
        ggml_tensor * st_ref  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, HV, ns);
        ggml_tensor * st_live = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, HV, ns);
        ggml_tensor * dst     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, SD, ns, 1);
        static thread_local std::vector<float> stash_host; // the stash survives the rounds (host copy)
        ggml_tensor * stash = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ROW, W, 2 * ns);
        ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, ns);
        ggml_tensor * A = ggml_gated_delta_net(ctx, q, k, v, g, b, st_ref, fin ? 1 : n);
        ggml_tensor * B = ggml_gated_delta_net_ext(ctx, q, k, v, g, b, st_live, dst);
        ggml_gated_delta_net_set_replay(B, stash, c);
        og.buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_buffer_clear(og.buf, 0);
        if (r == 0) {
            stash_host.assign(ggml_nelements(stash), 0.0f);
        }
        set_f(stash, stash_host);
        set_f(q, w.q); set_f(k, w.k); set_f(v, w.v); set_f(g, w.g); set_f(b, w.b);
        set_f(st_ref, ref); set_f(st_live, live);
        std::vector<int32_t> cv(4 * ns);
        for (int64_t s = 0; s < ns; ++s) {
            cv[4*s+0] = (int32_t) pk[s]; cv[4*s+1] = (int32_t) half[s]; cv[4*s+2] = (int32_t) d; cv[4*s+3] = (int32_t) slot[s];
        }
        ggml_backend_tensor_set(c, cv.data(), 0, cv.size() * sizeof(int32_t));
        for (ggml_tensor * t : { A, B }) {
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, t);
            if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { return false; }
        }
        const std::vector<float> out_live = get_f(dst);
        stash_host = get_f(stash);
        auto fail = [&](const char * what) {
            fprintf(stderr, "chain: %s ns=%lld round %zu (n=%lld d=%lld): %s differs\n", ggml_backend_name(be), (long long) ns, r,
                    (long long) n, (long long) d, what);
            ok = false;
        };
        if (fin) {
            // commit only: the state after the kept prefix = the reference start
            if (memcmp(out_live.data(), ref.data(), ref.size() * sizeof(float)) != 0) { fail("committed state (n_next 0)"); }
            break;
        }
        const size_t NA = S * HV * n * ns;
        const auto a_ref = get_f(A, 0, NA), a_rep = get_f(B, 0, NA);
        if (memcmp(a_ref.data(), a_rep.data(), NA * sizeof(float)) != 0) { fail("window attn"); }
        std::vector<float> next_ref(SD * ns);
        for (int64_t s = 0; s < ns; ++s) {
            auto plane = [&](int64_t j) { return get_f(A, NA + (size_t) j * SD * ns + s * SD, SD); };
            const auto pd = plane(d);
            if (memcmp(pd.data(), out_live.data() + s * SD, SD * sizeof(float)) != 0) { fail("deferred state vs plane d"); }
            const int64_t want = r + 1 < R ? keep[r] : p_final;
            const int64_t p = s == 0 ? std::min<int64_t>(want, d) : (want + s) % (d + 1); // distinct per sequence
            pk[s] = p;
            const auto pp = plane(d - p);
            std::copy(pp.begin(), pp.end(), next_ref.begin() + s * SD);
            if (d > 0) { half[s] = 1 - half[s]; }
        }
        ref  = next_ref;
        live = out_live;
    }
    return ok;
}

static int run_ops() {
    std::vector<ggml_backend_t> backends;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU ||
            ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
            if (be) {
                backends.push_back(be);
            }
        }
    }
    int n_cases = 0, n_fail = 0;
    double worst_nmse = 0;
    uint32_t seed = 1;
    for (int64_t ns : { 1, 5 }) {
        for (int64_t n = 1; n <= 6; ++n) {
            for (int64_t d = 0; d <= std::min<int64_t>(n, W); ++d) {
                for (int64_t p = 0; p <= d; ++p) {
                    const int64_t m = 1 + (n + p) % 4;
                    std::vector<case_out> outs(backends.size());
                    for (size_t bi = 0; bi < backends.size(); ++bi) {
                        n_cases++;
                        if (!run_case(backends[bi], ns, n, d, p, m, seed, outs[bi])) {
                            n_fail++;
                        }
                    }
                    for (size_t bi = 1; bi < backends.size(); ++bi) {
                        for (auto pr : { std::make_pair(&outs[0].attn1, &outs[bi].attn1), std::make_pair(&outs[0].s2, &outs[bi].s2),
                                         std::make_pair(&outs[0].attn3, &outs[bi].attn3), std::make_pair(&outs[0].s3, &outs[bi].s3) }) {
                            worst_nmse = std::max(worst_nmse, nmse(*pr.first, *pr.second));
                        }
                    }
                    seed++;
                }
            }
        }
    }
    // the chains: the (next window, kept prefix) pairs (1,5) (6,1) (7,5) inside, (0,1) / (0,5) at the end; >= 3
    //   deferring rounds per chain (both halves), 1 and 5 sequences, sequence 0 keeps exactly the listed prefix
    int n_chain = 0, n_chain_fail = 0;
    for (auto * be : backends) {
        for (int64_t ns : { 1, 5 }) {
            for (int64_t pf : { 1, 5 }) {
                n_chain++;
                if (!run_chain(be, ns, { 6, 1, 6, 6, 7, 6 }, { 5, 0, 1, 5, 3 }, pf, 1000u + (uint32_t) (ns * 10 + pf))) {
                    n_chain_fail++;
                }
            }
        }
    }
    printf("chain: %d chains (6 windows + commit, both halves), %d failed memcmp\n", n_chain, n_chain_fail);
    n_fail += n_chain_fail;

    printf("ops: %d case-runs on %zu backends (", n_cases, backends.size());
    for (size_t bi = 0; bi < backends.size(); ++bi) { printf("%s%s", bi ? ", " : "", ggml_backend_name(backends[bi])); }
    printf("): %d failed memcmp; worst NMSE vs %s = %.3e (limit 1e-6)\n", n_fail, backends.empty() ? "-" : ggml_backend_name(backends[0]), worst_nmse);
    for (auto * be : backends) { ggml_backend_free(be); }
    return (n_fail == 0 && worst_nmse <= 1e-6) ? 0 : 1;
}

// perf: one verification-shaped op per backend, us per op (median of 5 x 100): K = 1, the planes path K = n (direct
//   store into n planes), and the replay (n_prefix p, the window, n_defer n - 1)
static int run_perf() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) {
            continue;
        }
        for (int64_t ns : { 1, 5 }) {
            for (int64_t n : { 1, 5, 512 }) {
                const int64_t p  = n == 1 ? 0 : 3;
                const int64_t K  = std::min<int64_t>(n, 6);
                const int64_t dd = std::min<int64_t>(n - 1, W);
                op_graph og;
                ggml_init_params ip = { 256 * ggml_tensor_overhead() + 4 * ggml_graph_overhead(), nullptr, true };
                og.ctx = ggml_init(ip);
                ggml_context * ctx = og.ctx;
                ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, n, ns);
                ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, n, ns);
                ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HV, n, ns);
                ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, n, ns);
                ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, n, ns);
                ggml_tensor * st = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, HV, ns);
                ggml_tensor * planes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S * S * HV, ns, K);
                ggml_tensor * S1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S * S * HV, ns, 1);
                ggml_tensor * stash = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ROW, W, 2 * ns);
                ggml_tensor * c = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, ns);
                // 36 nodes per graph (the layers of a step): us per op = graph time / 36
                constexpr int NL = 36;
                ggml_cgraph * gfs[3];
                for (int o = 0; o < 3; ++o) {
                    gfs[o] = ggml_new_graph(ctx);
                    for (int l = 0; l < NL; ++l) {
                        ggml_tensor * op = ggml_gated_delta_net_ext(ctx, q, k, v, g, b, st, o == 1 ? planes : S1);
                        if (o == 2) {
                            ggml_gated_delta_net_set_replay(op, stash, c);
                        }
                        ggml_build_forward_expand(gfs[o], op);
                    }
                }
                og.buf = ggml_backend_alloc_ctx_tensors(ctx, be);
                ggml_backend_buffer_clear(og.buf, 0);
                std::vector<int32_t> cv(4 * ns);
                for (int64_t s = 0; s < ns; ++s) { cv[4*s] = (int32_t) p; cv[4*s+1] = 0; cv[4*s+2] = (int32_t) dd; cv[4*s+3] = (int32_t) s; }
                ggml_backend_tensor_set(c, cv.data(), 0, cv.size() * sizeof(int32_t));
                double us[3];
                for (int o = 0; o < 3; ++o) {
                    std::vector<double> t;
                    ggml_backend_graph_compute(be, gfs[o]);
                    for (int rep = 0; rep < 5; ++rep) {
                        const int64_t t0 = ggml_time_us();
                        for (int r = 0; r < 20; ++r) { ggml_backend_graph_compute(be, gfs[o]); }
                        t.push_back((ggml_time_us() - t0) / (20.0 * NL));
                    }
                    std::sort(t.begin(), t.end());
                    us[o] = t[2];
                }
                printf("perf %s ns=%lld n=%lld: K=1 %.1f us, planes K=%lld %.1f us, replay p=%lld d=%lld %.1f us (median us per op, 36 ops per graph)\n",
                        ggml_backend_name(be), (long long) ns, (long long) n, us[0], (long long) K, us[1], (long long) p, (long long) dd, us[2]);
            }
        }
        ggml_backend_free(be);
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// llama level
// ---------------------------------------------------------------------------------------------------------------------

static bool g_saw_replay = false;

static void log_cb(ggml_log_level level, const char * text, void * ud) {
    if (strstr(text, "gdn replay:") != nullptr) {
        g_saw_replay = true;
    }
    if (level >= GGML_LOG_LEVEL_WARN || ud != nullptr) {
        fputs(text, stderr);
    }
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_seq, bool replay) {
    setenv("LLAMA_GDN_REPLAY", replay ? "1" : "0", 1);
    g_saw_replay = false;
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq;
    cparams.n_rs_seq   = 5;
    cparams.kv_unified = true;
    cparams.n_ctx      = std::max<uint32_t>(cparams.n_ctx, 2048);
    cparams.n_batch    = std::max<uint32_t>(cparams.n_batch, 512);
    cparams.n_ubatch   = std::max<uint32_t>(cparams.n_ubatch, 512);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx && g_saw_replay != replay) {
        fprintf(stderr, "model: LLAMA_GDN_REPLAY=%d but the context %s the replay\n", replay, g_saw_replay ? "uses" : "does not use");
        llama_free(ctx);
        return nullptr;
    }
    return ctx;
}

static std::vector<uint8_t> seq_state(llama_context * ctx, llama_seq_id s) {
    std::vector<uint8_t> buf(llama_state_seq_get_size(ctx, s));
    const size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), s);
    buf.resize(n);
    return buf;
}

static std::vector<uint8_t> full_state(llama_context * ctx) {
    std::vector<uint8_t> buf(llama_state_get_size(ctx));
    const size_t n = llama_state_get_data(ctx, buf.data(), buf.size());
    buf.resize(n);
    return buf;
}

static uint64_t fnv(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h = (h ^ ((const uint8_t *) p)[i]) * 1099511628211ULL; }
    return h;
}

// decode one window per sequence (n tokens each, all outputs), depth n - 1; returns the logits of every output
static bool decode_window_n(llama_context * ctx, const std::vector<llama_seq_id> & seqs, std::vector<llama_pos> & pos,
                          const std::vector<uint32_t> & ns, uint32_t depth, int n_vocab, uint32_t round, std::vector<float> & logits) {
    uint32_t total = 0;
    for (llama_seq_id s : seqs) { total += ns[s]; }
    llama_batch batch = llama_batch_init((int32_t) total, 0, 1);
    for (llama_seq_id s : seqs) {
        for (uint32_t j = 0; j < ns[s]; ++j) {
            const llama_pos p = pos[s] + (llama_pos) j;
            const llama_token t = (llama_token) ((7919u * (uint32_t) p + 104729u * (uint32_t) s + 13u * round + 1) % (uint32_t) n_vocab);
            common_batch_add(batch, t, p, { s }, true);
        }
    }
    llama_set_rs_rollback_depth(ctx, depth);
    const bool ok = llama_decode(ctx, batch) == 0;
    logits.clear();
    if (ok) {
        for (int i = 0; i < batch.n_tokens; ++i) {
            const float * l = llama_get_logits_ith(ctx, i);
            logits.insert(logits.end(), l, l + n_vocab);
        }
    }
    llama_batch_free(batch);
    return ok;
}

static bool decode_window(llama_context * ctx, const std::vector<llama_seq_id> & seqs, std::vector<llama_pos> & pos,
                          uint32_t n, uint32_t depth, int n_vocab, uint32_t round, std::vector<float> & logits) {
    return decode_window_n(ctx, seqs, pos, std::vector<uint32_t>(pos.size(), n), depth, n_vocab, round, logits);
}

static int run_model(const common_params & params, llama_model * model) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    int n_fail = 0, n_checks = 0;

    for (uint32_t n_seq : { 1u, 5u }) {
        llama_context * off = make_ctx(params, model, n_seq, false);
        llama_context * on  = make_ctx(params, model, n_seq, true);
        if (!off || !on) {
            fprintf(stderr, "model: context init failed\n");
            return 1;
        }
        std::vector<llama_seq_id> seqs;
        for (uint32_t s = 0; s < n_seq; ++s) { seqs.push_back((llama_seq_id) s); }
        std::vector<llama_pos> pos(n_seq, 0);
        std::vector<float> lo, ln;

        auto expect = [&](bool c, const char * what, uint32_t round) {
            n_checks++;
            if (!c) {
                n_fail++;
                fprintf(stderr, "model: np%u round %u: %s\n", n_seq, round, what);
            }
        };

        // a prompt (no rollback depth), then 40 speculative-like rounds
        {
            std::vector<llama_pos> p0 = pos;
            expect(decode_window(off, seqs, pos, 37, 0, n_vocab, 0, lo) && decode_window(on, seqs, p0, 37, 0, n_vocab, 0, ln), "prompt decode", 0);
            expect(lo.size() == ln.size() && memcmp(lo.data(), ln.data(), lo.size() * sizeof(float)) == 0, "prompt logits", 0);
            for (auto & p : pos) { p += 37; }
        }
        uint64_t h_state = 1469598103934665603ULL;
        bool done_cp = false;
        std::vector<uint8_t> ckpt;     // a checkpoint of sequence 1 (np5), restored later while the others stay deferred
        llama_pos ckpt_pos = 0;
        for (uint32_t round = 1; round <= 40; ++round) {
            // windows of n tokens, depth n - 1; at np5 every other round unequal lengths per sequence (depth = the
            //   longest - 1: several ubatches, per-ubatch rollback limits)
            const uint32_t n = 1 + round % 6;
            std::vector<uint32_t> ns(n_seq, n);
            uint32_t depth = n - 1;
            if (n_seq > 1 && round % 2 == 1) {
                depth = 0;
                for (llama_seq_id s : seqs) { ns[s] = 1 + (round + 2 * (uint32_t) s) % 6; depth = std::max(depth, ns[s] - 1); }
            }
            std::vector<llama_pos> p0 = pos;
            expect(decode_window_n(off, seqs, p0, ns, depth, n_vocab, round, lo) &&
                   decode_window_n(on,  seqs, p0, ns, depth, n_vocab, round, ln), "window decode", round);
            expect(lo.size() == ln.size() && memcmp(lo.data(), ln.data(), lo.size() * sizeof(float)) == 0, "window logits", round);
            // decode -> save -> rollback: a state write (ON: committed into snapshots) must leave the rollback possible
            if (round % 4 == 1) {
                for (llama_seq_id s : seqs) {
                    const auto a = seq_state(off, s), b = seq_state(on, s);
                    expect(!a.empty() && a == b, "state bytes before the rollback", round);
                }
            }
            for (llama_seq_id s : seqs) {
                const uint32_t rb = (round / 2 + 3 * (uint32_t) s) % ns[s]; // 0 .. n - 1 tokens rolled back
                if (rb > 0) {
                    const bool r_off = llama_memory_seq_rm(llama_get_memory(off), s, pos[s] + (llama_pos) (ns[s] - rb), -1);
                    const bool r_on  = llama_memory_seq_rm(llama_get_memory(on),  s, pos[s] + (llama_pos) (ns[s] - rb), -1);
                    expect(r_off == r_on, "seq_rm rollback: same answer from both", round);
                    expect(r_off || n_seq > 1, "seq_rm rollback accepted (np1)", round);
                    pos[s] += (llama_pos) (r_off ? ns[s] - rb : ns[s]);
                } else {
                    pos[s] += (llama_pos) ns[s];
                }
            }
            // checkpoint fallback (np5): take sequence 1 at round 30, restore it at round 33 after its window while the
            //   other sequences keep deferred tokens; both contexts get the same bytes
            if (n_seq > 1 && round == 30) {
                ckpt = seq_state(on, 1);
                ckpt_pos = pos[1];
                expect(!ckpt.empty() && ckpt == seq_state(off, 1), "checkpoint bytes", round);
            }
            if (n_seq > 1 && round == 33 && !ckpt.empty()) {
                for (llama_context * c : { off, on }) {
                    expect(llama_state_seq_set_data(c, ckpt.data(), ckpt.size(), 1) == ckpt.size(), "checkpoint restore", round);
                }
                pos[1] = ckpt_pos;
            }
            // states: every third round per sequence (commits ON), every seventh round the whole context; the
            //   other rounds leave the replay to the next window
            if (round % 3 == 0) {
                for (llama_seq_id s : seqs) {
                    const auto a = seq_state(off, s), b = seq_state(on, s);
                    expect(!a.empty() && a == b, "llama_state_seq_get_data bytes", round);
                    h_state = h_state * 31 + fnv(a.data(), a.size());
                }
            }
            if (round % 7 == 0) {
                const auto a = full_state(off), b = full_state(on);
                expect(!a.empty() && a == b, "llama_state_get_data bytes", round);
            }
            // seq_cp of a sequence with deferred tokens (np5): 0 -> n_seq - 1, then both go on. In a round where
            //   sequence 0 rolled nothing back: a seq_cp drops a pending rollback of the planes path (rs_idx of the
            //   destination is 0), so with one the two paths would copy different states
            if (n_seq > 1 && !done_cp && round >= 20 && n > 1 && (round / 2) % n == 0) {
                done_cp = true;
                const llama_seq_id d = (llama_seq_id) n_seq - 1;
                for (llama_context * c : { off, on }) {
                    llama_memory_seq_rm(llama_get_memory(c), d, -1, -1);
                    llama_memory_seq_cp(llama_get_memory(c), 0, d, -1, -1);
                }
                pos[d] = pos[0];
                const auto a = seq_state(off, d), b = seq_state(on, d);
                expect(!a.empty() && a == b, "state after seq_cp", round);
            }
        }

        // save -> load round trips: ON bytes into a fresh OFF context and OFF bytes into a fresh ON context, then the
        //   same next window everywhere
        {
            const llama_seq_id s = 0;
            const auto a = seq_state(off, s), b = seq_state(on, s);
            expect(a == b, "state before the round trip", 99);
            llama_context * off2 = make_ctx(params, model, n_seq, false);
            llama_context * on2  = make_ctx(params, model, n_seq, true);
            expect(off2 && on2, "round-trip contexts", 99);
            if (off2 && on2) {
                expect(llama_state_seq_set_data(off2, b.data(), b.size(), s) == b.size(), "load ON bytes into OFF", 99);
                expect(llama_state_seq_set_data(on2,  a.data(), a.size(), s) == a.size(), "load OFF bytes into ON", 99);
                std::vector<llama_seq_id> one = { s };
                std::vector<float> l1, l2, l3, l4;
                std::vector<llama_pos> pa = pos, pb = pos, pc = pos, pd = pos;
                expect(decode_window(off,  one, pa, 4, 3, n_vocab, 100, l1) && decode_window(on,  one, pb, 4, 3, n_vocab, 100, l2) &&
                       decode_window(off2, one, pc, 4, 3, n_vocab, 100, l3) && decode_window(on2, one, pd, 4, 3, n_vocab, 100, l4),
                       "round-trip decode", 100);
                // OFF == ON on the live contexts and on the reloaded ones; live == reloaded only at np1: at np5 the
                //   live unified KV also holds the other sequences (other cells, other attention chunking)
                expect(l1 == l2, "round-trip logits OFF == ON (live)", 100);
                expect(l3 == l4, "round-trip logits ON->OFF == OFF->ON (reloaded)", 100);
                expect(n_seq > 1 || l1 == l3, "round-trip logits live == reloaded (np1)", 100);
                for (llama_context * c : { off, on, off2, on2 }) {
                    llama_memory_seq_rm(llama_get_memory(c), s, pos[s] + 2, -1);
                }
                const auto x = seq_state(off, s);
                expect(x == seq_state(on, s) && x == seq_state(off2, s) && x == seq_state(on2, s), "round-trip states after a rollback", 100);
                h_state = h_state * 31 + fnv(x.data(), x.size());
            }
            llama_free(off2);
            llama_free(on2);
        }
        // seq_cp of a sequence with deferred tokens (astra's reproducer: defer A 4 rows -> cp A -> B -> roll A back 1 ->
        //   save B): ON commits A first, so B and A read the same committed row and A's rollback is refused (OFF
        //   accepts it and moves the shared cell's position - a planes-path quirk, not compared). Then seq_keep(A)
        // rollback -> seq_cp (v70.1 item 2): A defers 4 tokens and rolls 2 back (R / P rollback pending), then A -> B. ON
        //   materialises S, R and P of the rolled-back position before sharing: B == A on ON, A == A on OFF (the planes
        //   path gives B its plane 0 = the pre-rollback state - not compared). B is then dropped in both
        if (n_seq > 1) {
            const llama_seq_id A = 1, B = 3;
            std::vector<llama_seq_id> one = { A };
            std::vector<llama_pos> pa = pos;
            expect(decode_window(off, one, pa, 5, 4, n_vocab, 190, lo) && decode_window(on, one, pa, 5, 4, n_vocab, 190, ln) && lo == ln,
                   "rb-cp: window of A", 190);
            pos[A] += 3;
            const bool r_off = llama_memory_seq_rm(llama_get_memory(off), A, pos[A], -1);
            const bool r_on  = llama_memory_seq_rm(llama_get_memory(on),  A, pos[A], -1);
            expect(r_off && r_on, "rb-cp: rollback of A by 2", 190);
            for (llama_context * c : { off, on }) {
                llama_memory_seq_rm(llama_get_memory(c), B, -1, -1);
                llama_memory_seq_cp(llama_get_memory(c), A, B, -1, -1);
            }
            const auto a_off = seq_state(off, A), a_on = seq_state(on, A), b_on = seq_state(on, B);
            expect(!a_off.empty() && a_on == a_off, "rb-cp: A bytes (ON == OFF)", 190);
            expect(b_on == a_on, "rb-cp: B bytes == A bytes (ON: S, R, P of the rolled-back position)", 190);
            // A and the copy go on with the same tokens: same logits (ON)
            {
                std::vector<float> la, lb;
                llama_batch ba = llama_batch_init(3, 0, 1), bb = llama_batch_init(3, 0, 1);
                for (int j = 0; j < 3; ++j) {
                    common_batch_add(ba, (llama_token) (1000 + 7 * j) % n_vocab, pos[A] + j, { A }, true);
                    common_batch_add(bb, (llama_token) (1000 + 7 * j) % n_vocab, pos[A] + j, { B }, true);
                }
                llama_set_rs_rollback_depth(on, 2);
                bool ok = llama_decode(on, ba) == 0;
                for (int i = 0; ok && i < 3; ++i) { const float * l = llama_get_logits_ith(on, i); la.insert(la.end(), l, l + n_vocab); }
                llama_set_rs_rollback_depth(on, 2);
                ok = ok && llama_decode(on, bb) == 0;
                for (int i = 0; ok && i < 3; ++i) { const float * l = llama_get_logits_ith(on, i); lb.insert(lb.end(), l, l + n_vocab); }
                // info only: B's new tokens sit in other KV cells than A's (attention order), the states above are the check
                printf("model: rb-cp: A and B logits on the same tokens %s\n", ok && la == lb ? "identical" : "differ (KV cell layout)");
                llama_batch_free(ba); llama_batch_free(bb);
                // OFF follows A's tokens too, so the two contexts stay in lockstep for A
                llama_batch bo = llama_batch_init(3, 0, 1);
                for (int j = 0; j < 3; ++j) { common_batch_add(bo, (llama_token) (1000 + 7 * j) % n_vocab, pos[A] + j, { A }, true); }
                llama_set_rs_rollback_depth(off, 2);
                std::vector<float> lo2;
                ok = llama_decode(off, bo) == 0;
                for (int i = 0; ok && i < 3; ++i) { const float * l = llama_get_logits_ith(off, i); lo2.insert(lo2.end(), l, l + n_vocab); }
                expect(ok && lo2 == la, "rb-cp: A on OFF == A on ON", 191);
                llama_batch_free(bo);
                pos[A] += 3;
            }
            for (llama_context * c : { off, on }) {
                llama_memory_seq_rm(llama_get_memory(c), B, -1, -1);
            }
        }

        if (n_seq > 1) {
            const llama_seq_id A = 2, B = 4;
            std::vector<llama_seq_id> one = { A };
            std::vector<llama_pos> pa = pos;
            expect(decode_window(off, one, pa, 5, 4, n_vocab, 200, lo) && decode_window(on, one, pa, 5, 4, n_vocab, 200, ln) && lo == ln,
                   "cp: window of A", 200);
            pos[A] += 5;
            for (llama_context * c : { off, on }) {
                llama_memory_seq_rm(llama_get_memory(c), B, -1, -1);
                llama_memory_seq_cp(llama_get_memory(c), A, B, -1, -1);
            }
            pos[B] = pos[A];
            const auto b_on = seq_state(on, B);
            expect(!b_on.empty() && b_on == seq_state(off, B), "cp: B bytes", 200);
            expect(seq_state(on, A) == seq_state(off, A), "cp: A bytes", 200);
            expect(!llama_memory_seq_rm(llama_get_memory(on), A, pos[A] - 1, -1), "cp: ON refuses rolling back the copied A", 200);
            expect(seq_state(on, B) == b_on, "cp: B unchanged after the refused rollback", 200);
            // both go on from the copy (OFF untouched by the refused step), then keep only A
            std::vector<llama_seq_id> two = { A, B };
            std::vector<llama_pos> pb = pos;
            expect(decode_window(off, two, pb, 3, 2, n_vocab, 201, lo) && decode_window(on, two, pb, 3, 2, n_vocab, 201, ln) && lo == ln,
                   "cp: A and B go on", 201);
            pos[A] += 3; pos[B] += 3;
            for (llama_context * c : { off, on }) {
                llama_memory_seq_keep(llama_get_memory(c), A);
            }
            expect(seq_state(on, A) == seq_state(off, A), "seq_keep: A bytes", 201);
            std::vector<llama_pos> pc = pos;
            expect(decode_window(off, one, pc, 4, 3, n_vocab, 202, lo) && decode_window(on, one, pc, 4, 3, n_vocab, 202, ln) && lo == ln,
                   "seq_keep: next window of A", 202);
        }

        printf("model: np%u: %d checks, %d failed, state hash %016llx\n", n_seq, n_checks, n_fail, (unsigned long long) h_state);
        llama_free(off);
        llama_free(on);
    }
    return n_fail == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    if (argc >= 2 && strcmp(argv[1], "perf") == 0) {
        ggml_backend_load_all();
        return run_perf();
    }
    if (argc >= 2 && strcmp(argv[1], "ops") == 0) {
        ggml_backend_load_all();
        const int rc = run_ops();
        printf("test-gdn-replay ops: %s\n", rc == 0 ? "PASS" : "FAIL");
        return rc;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_log_set(log_cb, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (model == nullptr) {
        fprintf(stderr, "failed to load the model\n");
        return 1;
    }
    const int rc = run_model(params, model);
    llama_model_free(model);
    printf("test-gdn-replay model: %s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
