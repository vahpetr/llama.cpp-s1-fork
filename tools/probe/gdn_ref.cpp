// gdn_ref — f64 reference for the gated delta rule vs every llama.cpp formulation.
//
// Task 2.3 (spec-perf-ladder-2). Compares, on random inputs of the production shape class:
//   REF      : f64 sequential gated delta rule, exactly the semantics of
//              ggml_compute_forward_gated_delta_net_one_chunk (ggml/src/ggml-cpu/ops.cpp)
//   fused-T  : ggml_gated_delta_net with n_tokens = T in ONE call
//   fused-1x : ggml_gated_delta_net with n_tokens = 1, stepped T times (state fed forward)
//   chunk    : the unfused decomposition, a verbatim copy of
//              llm_build_delta_net_base::build_delta_net_chunking (src/models/delta-net-base.cpp)
//   ar       : the unfused decomposition, a verbatim copy of
//              llm_build_delta_net_base::build_delta_net_autoregressive (T == 1 only), stepped T times
// on any backend (CPU / Vulkan). Reports max abs error of the attention output AND of the final state.
//
// build (dev container):
//   scripts/bench/llama-dev.sh bash -c 'cd /src/.wt-p1 && \
//     g++ -std=c++17 -O1 -I ggml/include tools/probe/gdn_ref.cpp \
//     -o build-dev/bin/gdn_ref -L build-dev/bin -lggml -lggml-base \
//     -Wl,-rpath,/src/.wt-p1/build-dev/bin'
//
// usage: gdn_ref [--backend cpu|vk] [-S 128] [--hk 2] [--hv 4] [-T 5] [--seq 1] [--kda] [--seed 1234]
//        gdn_ref [--backend ...] --mulmat [-K 4096] [-N 512]   batch-invariance of mul_mat (n=1 vs n>1)
//        gdn_ref [--backend ...] --ops                         batch-invariance of flash_attn_ext / ssm_conv
//
// This file is NOT wired into CMake on purpose (tests are OFF in build-dev); build it with the line above.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

struct Shape {
    int64_t S  = 128;
    int64_t Hk = 2;
    int64_t Hv = 4;
    int64_t T  = 5;
    int64_t B  = 1;
    bool    kda = false;
};

// ---------------------------------------------------------------------------------------------
// f64 reference: exactly ggml_compute_forward_gated_delta_net_one_chunk, in double.
//   state layout (both in and out): element (i,j) of S at i + S*(j + S*(h + Hv*b));  S[i][j]
//   out layout: j + S*(h + Hv*(t + T*b))
// ---------------------------------------------------------------------------------------------
static void ref_gdn(const Shape & sh,
                    const std::vector<float> & q, const std::vector<float> & k,
                    const std::vector<float> & v, const std::vector<float> & g,
                    const std::vector<float> & b, const std::vector<float> & s0,
                    std::vector<double> & out, std::vector<double> & sfin) {
    const int64_t S = sh.S, Hk = sh.Hk, Hv = sh.Hv, T = sh.T, B = sh.B;
    const int64_t GS = sh.kda ? S : 1;
    const double scale = 1.0 / std::sqrt((double) S);

    out.assign((size_t)(S*Hv*T*B), 0.0);
    sfin.assign((size_t)(S*S*Hv*B), 0.0);

    std::vector<double> St((size_t)(S*S));
    std::vector<double> delta((size_t) S);

    for (int64_t ib = 0; ib < B; ++ib) {
        for (int64_t h = 0; h < Hv; ++h) {
            const int64_t hk = h % Hk;                       // [TAG_GGML_GDN_BCAST] tiled
            for (int64_t j = 0; j < S; ++j)
                for (int64_t i = 0; i < S; ++i)
                    St[(size_t)(i*S + j)] = s0[(size_t)(i + S*(j + S*(h + Hv*ib)))];

            for (int64_t t = 0; t < T; ++t) {
                const float * qd = &q[(size_t)(S*(hk + Hk*(t + T*ib)))];
                const float * kd = &k[(size_t)(S*(hk + Hk*(t + T*ib)))];
                const float * vd = &v[(size_t)(S*(h  + Hv*(t + T*ib)))];
                const float * gd = &g[(size_t)(GS*(h + Hv*(t + T*ib)))];
                const double  bt =  b[(size_t)(   (h + Hv*(t + T*ib)))];

                if (sh.kda) {
                    for (int64_t i = 0; i < S; ++i) {
                        const double e = std::exp((double) gd[i]);
                        for (int64_t j = 0; j < S; ++j) St[(size_t)(i*S + j)] *= e;
                    }
                } else {
                    const double e = std::exp((double) gd[0]);
                    for (size_t x = 0; x < St.size(); ++x) St[x] *= e;
                }

                for (int64_t j = 0; j < S; ++j) {
                    double sum = 0.0;
                    for (int64_t i = 0; i < S; ++i) sum += St[(size_t)(i*S + j)] * (double) kd[i];
                    delta[(size_t) j] = ((double) vd[j] - sum) * bt;
                }
                for (int64_t i = 0; i < S; ++i)
                    for (int64_t j = 0; j < S; ++j)
                        St[(size_t)(i*S + j)] += (double) kd[i] * delta[(size_t) j];

                for (int64_t j = 0; j < S; ++j) {
                    double sum = 0.0;
                    for (int64_t i = 0; i < S; ++i) sum += St[(size_t)(i*S + j)] * (double) qd[i];
                    out[(size_t)(j + S*(h + Hv*(t + T*ib)))] = sum * scale;
                }
            }
            for (int64_t j = 0; j < S; ++j)
                for (int64_t i = 0; i < S; ++i)
                    sfin[(size_t)(i + S*(j + S*(h + Hv*ib)))] = St[(size_t)(i*S + j)];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// verbatim copy of llm_build_delta_net_base::build_delta_net_chunking (src/models/delta-net-base.cpp),
// with ctx0 as a parameter and cb() removed.  DO NOT EDIT except to track that file.
// ---------------------------------------------------------------------------------------------
static ggml_tensor * get_slice_2d(ggml_context * ctx0, ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3], t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

static std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
        ggml_context * ctx0,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * g, ggml_tensor * b, ggml_tensor * s) {
    const int64_t S_k      = q->ne[0];
    const int64_t H_k      = q->ne[1];
    const int64_t n_tokens = q->ne[2];
    const int64_t n_seqs   = q->ne[3];

    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    const bool kda = (g->ne[0] == S_k && g->ne[1] == H_k);

    const float scale = 1.0f / sqrtf(S_k);

    q = ggml_scale(ctx0, q, scale);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);
    g = ggml_permute(ctx0, g, 0, 2, 1, 3);
    b = ggml_permute(ctx0, b, 0, 2, 1, 3);

    const int CS = kda ? 16 : 64;

    const int pad = (CS - n_tokens % CS) % CS;
    const int n_chunks = (n_tokens + pad) / CS;

    q = ggml_pad(ctx0, q, 0, pad, 0, 0);
    k = ggml_pad(ctx0, k, 0, pad, 0, 0);
    v = ggml_pad(ctx0, v, 0, pad, 0, 0);
    g = ggml_pad(ctx0, g, 0, pad, 0, 0);
    b = ggml_pad(ctx0, b, 0, pad, 0, 0);

    ggml_tensor * v_b = ggml_mul(ctx0, v, b);
    ggml_tensor * k_b = ggml_mul(ctx0, k, b);

    q   = ggml_reshape_4d(ctx0, q,   S_k, CS, n_chunks, H_k * n_seqs);
    k   = ggml_reshape_4d(ctx0, k,   S_k, CS, n_chunks, H_k * n_seqs);
    k_b = ggml_reshape_4d(ctx0, k_b, S_k, CS, n_chunks, H_v * n_seqs);
    v   = ggml_reshape_4d(ctx0, v,   S_v, CS, n_chunks, H_v * n_seqs);
    v_b = ggml_reshape_4d(ctx0, v_b, S_v, CS, n_chunks, H_v * n_seqs);

    g = ggml_reshape_4d(ctx0, g, g->ne[0], CS, n_chunks, H_v * n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        CS, n_chunks, H_v * n_seqs);

    ggml_tensor * g_cs = ggml_cumsum(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, g)));

    ggml_tensor * kb = nullptr;
    ggml_tensor * kq = nullptr;
    if (kda) {
        const int64_t CHB = n_chunks * H_k * n_seqs;
        ggml_tensor * g_cs_i = ggml_reshape_4d(ctx0, g_cs, CS, 1, S_k, CHB);
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, S_k, CHB);
        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, S_k, CHB);
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        decay_mask = ggml_cont_4d(ctx0, ggml_permute(ctx0, decay_mask, 2, 1, 0, 3), S_k, CS, CS, CHB);
        ggml_tensor * k_b_i = ggml_reshape_4d(ctx0, k_b, S_k, CS,  1, CHB);
        ggml_tensor * k_j   = ggml_reshape_4d(ctx0, k,   S_k,  1, CS, CHB);
        ggml_tensor * q_i   = ggml_reshape_4d(ctx0, q,   S_k, CS,  1, CHB);
        ggml_tensor * decay_k_b_i = ggml_mul(ctx0, decay_mask, k_b_i);
        ggml_tensor * decay_q_i   = ggml_mul(ctx0, decay_mask, q_i);
        kb = ggml_mul_mat(ctx0, decay_k_b_i, k_j);
        kq = ggml_mul_mat(ctx0, decay_q_i,   k_j);
        kb = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kb, CS, CS, n_chunks, H_v * n_seqs)));
        kq = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_4d(ctx0, kq, CS, CS, n_chunks, H_v * n_seqs)));
    } else {
        ggml_tensor * g_cs_i = g_cs;
        ggml_tensor * g_cs_j = ggml_reshape_4d(ctx0, g_cs, 1, CS, n_chunks, H_v * n_seqs);
        g_cs_j = ggml_repeat_4d(ctx0, g_cs_j, CS, CS, n_chunks, H_v * n_seqs);
        ggml_tensor * decay_mask;
        decay_mask = ggml_sub(ctx0, g_cs_j, g_cs_i);
        decay_mask = ggml_tri(ctx0, decay_mask, GGML_TRI_TYPE_LOWER_DIAG);
        decay_mask = ggml_exp(ctx0, decay_mask);
        kb = ggml_mul_mat(ctx0, k,  k_b);
        kb = ggml_mul    (ctx0, kb, decay_mask);
        kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_mul(ctx0, kq, decay_mask);
    }

    kq = ggml_tri(ctx0, kq, GGML_TRI_TYPE_LOWER_DIAG);

    ggml_tensor * attn;
    attn = ggml_tri(ctx0, kb, GGML_TRI_TYPE_LOWER);

    ggml_tensor * identity;
    identity = ggml_view_1d(ctx0, attn, CS, 0);
    identity = ggml_fill   (ctx0, identity, 1.0f);
    identity = ggml_diag   (ctx0, identity);

    ggml_tensor * lhs = ggml_add(ctx0, attn, identity);

    attn = ggml_neg(ctx0, attn);

    ggml_tensor * lin_solve = ggml_solve_tri(ctx0, lhs, attn, true, true, false);
    attn = ggml_add(ctx0, lin_solve, identity);

    v = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v_b)), attn);

    ggml_tensor * g_exp = ggml_exp(ctx0, g_cs);

    k_b = ggml_cont(ctx0, ggml_transpose(ctx0, k_b));

    ggml_tensor * kbg = ggml_mul(ctx0, k_b, g_exp);

    ggml_tensor * k_cd = ggml_mul_mat(ctx0, kbg, attn);

    ggml_tensor * g_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, g_exp));
    ggml_tensor * q_g_exp = ggml_mul(ctx0, q, g_exp_t);

    ggml_tensor * g_last = ggml_view_4d(ctx0, g_cs, 1, g_cs->ne[1], g_cs->ne[2], g_cs->ne[3],
            g_cs->nb[1], g_cs->nb[2], g_cs->nb[3], ggml_row_size(g_cs->type, g_cs->ne[0] - 1));

    g_last = ggml_cont(ctx0, g_last);

    ggml_tensor * g_last_exp_t = ggml_transpose(ctx0, ggml_exp(ctx0, g_last));

    ggml_tensor * g_diff = ggml_neg(ctx0, ggml_sub(ctx0, g_cs, g_last));

    ggml_tensor * g_diff_exp_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_exp(ctx0, g_diff)));

    ggml_tensor * kg = ggml_mul(ctx0, k, g_diff_exp_t);

    ggml_tensor * kg_t = ggml_cont(ctx0, ggml_transpose(ctx0, kg));

    s = ggml_reshape_4d(ctx0, s, S_v, S_v, 1, H_v * n_seqs);

    ggml_tensor * v_t = ggml_cont(ctx0, ggml_transpose(ctx0, v));

    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        ggml_tensor * ch_k_cd    = get_slice_2d(ctx0, k_cd,    chunk);
        ggml_tensor * ch_v_t     = get_slice_2d(ctx0, v_t,     chunk);
        ggml_tensor * ch_kq      = get_slice_2d(ctx0, kq,      chunk);
        ggml_tensor * ch_q_g_exp = get_slice_2d(ctx0, q_g_exp, chunk);
        ggml_tensor * ch_kg_t    = get_slice_2d(ctx0, kg_t,    chunk);

        ggml_tensor * v_t_p   = ggml_mul_mat(ctx0, ch_k_cd, s);
        ggml_tensor * v_t_new = ggml_sub(ctx0, ch_v_t, v_t_p);
        ggml_tensor * v_attn  = ggml_mul_mat(ctx0, v_t_new, ch_kq);
        ggml_tensor * attn_inter = ggml_mul_mat(ctx0, s, ch_q_g_exp);
        ggml_tensor * o_ch = ggml_add(ctx0, attn_inter, v_attn);

        v = ggml_set_inplace(ctx0, v, o_ch, v->nb[1], v->nb[2], v->nb[3], chunk * v->nb[2]);

        ggml_tensor * kgv = ggml_mul_mat(ctx0, ch_kg_t, v_t_new);
        ggml_tensor * ch_g_last_exp_t = get_slice_2d(ctx0, g_last_exp_t, chunk);
        s = ggml_mul(ctx0, s, ch_g_last_exp_t);
        s = ggml_add(ctx0, s, kgv);
    }

    ggml_tensor * o = ggml_view_4d(ctx0, v, S_v, n_tokens, H_v, n_seqs,
            ggml_row_size(v->type, S_v),
            ggml_row_size(v->type, S_v * CS * n_chunks),
            ggml_row_size(v->type, S_v * CS * n_chunks * H_v), 0);
    o = ggml_permute  (ctx0, o, 0, 2, 1, 3);
    s = ggml_reshape_4d(ctx0, s, S_v, S_v, H_v, n_seqs);

    return {o, s};
}

// verbatim copy of build_delta_net_autoregressive
static std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
        ggml_context * ctx0,
        ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * g, ggml_tensor * b, ggml_tensor * s) {
    const int64_t S_k    = q->ne[0];
    const int64_t H_k    = q->ne[1];
    const int64_t n_seqs = q->ne[3];
    const int64_t S_v = v->ne[0];
    const int64_t H_v = v->ne[1];
    (void) S_v; (void) H_k;

    const float scale = 1.0f / sqrtf(S_k);
    q = ggml_scale(ctx0, q, scale);
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    g = ggml_reshape_4d(ctx0, g, 1, g->ne[0], H_v, n_seqs);
    b = ggml_reshape_4d(ctx0, b, 1,        1, H_v, n_seqs);

    g = ggml_exp(ctx0, g);
    s = ggml_mul(ctx0, s, g);

    ggml_tensor * sk;
    sk = ggml_mul     (ctx0, s, k);
    sk = ggml_sum_rows(ctx0, sk);

    ggml_tensor * d;
    d = ggml_sub(ctx0, v, ggml_transpose(ctx0, sk));
    d = ggml_mul(ctx0, d, b);

    ggml_tensor * d_t = ggml_transpose(ctx0, d);

    ggml_tensor * kd;
    k  = ggml_repeat(ctx0, k, s);
    kd = ggml_mul   (ctx0, k, d_t);

    s = ggml_add(ctx0, s, kd);

    ggml_tensor * s_q = ggml_mul     (ctx0, s, q);
    ggml_tensor * o   = ggml_sum_rows(ctx0, s_q);

    o = ggml_permute  (ctx0, o, 2, 0, 1, 3);

    return {o, s};
}

// ---------------------------------------------------------------------------------------------

struct Inputs {
    std::vector<float> q, k, v, g, b, s0;
};

static double maxabs_diff(const float * a, const std::vector<double> & r, size_t n, double * rel) {
    double m = 0.0, mrel = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - r[i]);
        if (d > m) m = d;
        const double den = std::fabs(r[i]) > 1e-6 ? std::fabs(r[i]) : 1e-6;
        if (d/den > mrel) mrel = d/den;
    }
    if (rel) *rel = mrel;
    return m;
}

static ggml_tensor * mk(ggml_context * ctx, int64_t a, int64_t b_, int64_t c, int64_t d, const char * name) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, a, b_, c, d);
    ggml_set_name(t, name);
    ggml_set_input(t);
    return t;
}

enum Mode { MODE_FUSED, MODE_CHUNK, MODE_AR };

// run one T-token step on `backend`; s_io is the state in op layout (updated in place)
static void run_once(ggml_backend_t backend, const Shape & sh, int64_t T, Mode mode,
                     const Inputs & in, int64_t t_off,
                     std::vector<float> & s_io, float * out_attn) {
    const int64_t S = sh.S, Hk = sh.Hk, Hv = sh.Hv, B = sh.B;
    const int64_t GS = sh.kda ? S : 1;

    static std::vector<uint8_t> membuf;
    const size_t msz = ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(8192, false);
    membuf.resize(msz);
    ggml_init_params ip = { msz, membuf.data(), true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = mk(ctx, S,  Hk, T, B, "q");
    ggml_tensor * k = mk(ctx, S,  Hk, T, B, "k");
    ggml_tensor * v = mk(ctx, S,  Hv, T, B, "v");
    ggml_tensor * g = mk(ctx, GS, Hv, T, B, "g");
    ggml_tensor * b = mk(ctx, 1,  Hv, T, B, "beta");
    ggml_tensor * s = mk(ctx, S,  S,  Hv, B, "state");

    ggml_tensor * o_out = nullptr;
    ggml_tensor * s_out = nullptr;
    ggml_tensor * fused = nullptr;
    if (mode == MODE_FUSED) {
        fused = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 1);
    } else {
        // src/models/qwen35.cpp:441 -- the unfused paths need q/k explicitly repeated to H_v heads
        ggml_tensor * qr = q, * kr = k;
        if (Hk != Hv) {
            qr = ggml_repeat_4d(ctx, q, S, Hv, T, B);
            kr = ggml_repeat_4d(ctx, k, S, Hv, T, B);
        }
        auto pr = (mode == MODE_CHUNK)
            ? build_delta_net_chunking(ctx, qr, kr, v, g, b, s)
            : build_delta_net_autoregressive(ctx, qr, kr, v, g, b, s);
        o_out = ggml_cont(ctx, pr.first);   // [S, Hv, T, B]
        s_out = ggml_cont(ctx, pr.second);  // [S, S, Hv, B]
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    if (mode == MODE_FUSED) {
        ggml_build_forward_expand(gf, fused);
    } else {
        ggml_build_forward_expand(gf, o_out);
        ggml_build_forward_expand(gf, s_out);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); exit(1); }

    const size_t tokoff_q = (size_t)(S*Hk*t_off);
    const size_t tokoff_v = (size_t)(S*Hv*t_off);
    const size_t tokoff_g = (size_t)(GS*Hv*t_off);
    const size_t tokoff_b = (size_t)(Hv*t_off);
    ggml_backend_tensor_set(q, in.q.data() + tokoff_q, 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, in.k.data() + tokoff_q, 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, in.v.data() + tokoff_v, 0, ggml_nbytes(v));
    ggml_backend_tensor_set(g, in.g.data() + tokoff_g, 0, ggml_nbytes(g));
    ggml_backend_tensor_set(b, in.b.data() + tokoff_b, 0, ggml_nbytes(b));
    ggml_backend_tensor_set(s, s_io.data(),            0, ggml_nbytes(s));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); exit(1);
    }

    const size_t attn_elems  = (size_t)(S*Hv*T*B);
    const size_t state_elems = (size_t)(S*S*Hv*B);
    if (mode == MODE_FUSED) {
        ggml_backend_tensor_get(fused, out_attn, 0, attn_elems*sizeof(float));
        ggml_backend_tensor_get(fused, s_io.data(), attn_elems*sizeof(float), state_elems*sizeof(float));
    } else {
        ggml_backend_tensor_get(o_out, out_attn,   0, attn_elems*sizeof(float));
        ggml_backend_tensor_get(s_out, s_io.data(), 0, state_elems*sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// -------------------------------------------------------------------------------------------
// --mulmat: is mul_mat batch-invariant?  Compare column 0 of W @ X for n = 1 vs n = N columns,
// with identical column-0 data.  This is the n==1 (vec_dot) vs n>1 (GEMM) kernel switch.
// -------------------------------------------------------------------------------------------
static void run_mulmat(ggml_backend_t backend, ggml_type wt, int64_t K, int64_t N, int64_t n,
                       const void * wdata, size_t wbytes, const float * xdata, float * out) {
    const size_t msz = ggml_tensor_overhead()*64 + ggml_graph_overhead();
    std::vector<uint8_t> membuf(msz);
    ggml_init_params ip = { msz, membuf.data(), true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * W = ggml_new_tensor_2d(ctx, wt, K, N);
    ggml_tensor * X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
    ggml_tensor * Y = ggml_mul_mat(ctx, W, X);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(W, wdata, 0, wbytes);
    ggml_backend_tensor_set(X, xdata, 0, (size_t)(K*n)*sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "mm compute failed\n"); exit(1); }
    ggml_backend_tensor_get(Y, out, 0, (size_t) N*sizeof(float));  // column 0 only
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static int mulmat_main(ggml_backend_t backend, int64_t K, int64_t N, int64_t nmax, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> wf((size_t)(K*N));
    for (auto & x : wf) x = 0.1f*nd(rng);
    std::vector<float> X((size_t)(K*nmax));
    for (auto & x : X) x = nd(rng);

    const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
    for (ggml_type wt : types) {
        std::vector<uint8_t> wq((size_t) ggml_row_size(wt, K) * N);
        if (wt == GGML_TYPE_F32) memcpy(wq.data(), wf.data(), wq.size());
        else ggml_quantize_chunk(wt, wf.data(), wq.data(), 0, N, K, nullptr);
        std::vector<float> y1((size_t) N), yn((size_t) N);
        run_mulmat(backend, wt, K, N, 1,    wq.data(), wq.size(), X.data(), y1.data());
        double prev = -1.0;
        for (int64_t n = 2; n <= nmax; ++n) {
            run_mulmat(backend, wt, K, N, n, wq.data(), wq.size(), X.data(), yn.data());
            double m = 0.0, mrel = 0.0;
            for (int64_t i = 0; i < N; ++i) {
                const double d = std::fabs((double) yn[i] - (double) y1[i]);
                if (d > m) m = d;
                const double den = std::fabs((double) y1[i]) > 1e-6 ? std::fabs((double) y1[i]) : 1e-6;
                if (d/den > mrel) mrel = d/den;
            }
            printf("%-8s K=%lld N=%lld  n=1 vs n=%lld : col0 maxabs %.3e  maxrel %.3e%s\n",
                   ggml_type_name(wt), (long long) K, (long long) N, (long long) n, m, mrel,
                   (prev >= 0.0 && m == prev) ? "   (== previous n)" : "");
            prev = m;
        }
    }
    return 0;
}


// -------------------------------------------------------------------------------------------
// --ops: batch-invariance of the two remaining non-mul_mat suspects in the qwen35 graph:
//   ggml_flash_attn_ext (f16 KV cache)  and  ggml_ssm_conv (f32 conv1d weight).
// Compare TOKEN 0 of the output for n_tokens = 1 vs n_tokens = n; token 0 sees identical inputs.
// -------------------------------------------------------------------------------------------
static void cmp0(const char * tag, int64_t n, const std::vector<float> & a, const std::vector<float> & b, double * prev) {
    double m = 0.0, mrel = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > m) m = d;
        const double den = std::fabs((double) a[i]) > 1e-6 ? std::fabs((double) a[i]) : 1e-6;
        if (d/den > mrel) mrel = d/den;
    }
    printf("%-16s n=1 vs n=%lld : tok0 maxabs %.3e  maxrel %.3e%s\n", tag, (long long) n, m, mrel,
           (*prev >= 0.0 && m == *prev) ? "   (== previous n)" : "");
    *prev = m;
}

static int ops_main(ggml_backend_t backend, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // ---- flash_attn_ext: D=128, H=32, H_kv=8, n_kv=128 ----
    const int64_t D = 128, H = 32, HKV = 8, NKV = 128, NQMAX = 6;
    std::vector<float> qf((size_t)(D*NQMAX*H));
    std::vector<uint16_t> kh((size_t)(D*NKV*HKV)), vh((size_t)(D*NKV*HKV));
    for (auto & x : qf) x = nd(rng);
    { std::vector<float> t((size_t)(D*NKV*HKV));
      for (auto & x : t) x = nd(rng); ggml_fp32_to_fp16_row(t.data(), (ggml_fp16_t *) kh.data(), t.size());
      for (auto & x : t) x = nd(rng); ggml_fp32_to_fp16_row(t.data(), (ggml_fp16_t *) vh.data(), t.size()); }

    // causal: the batched run attends over KV = prompt + all nq tail tokens (future ones masked),
    // the single run over KV = prompt + this token only.  Token 0 is mathematically identical in both.
    const int64_t NBASE = 32;
    auto run_fa = [&](int64_t nq, std::vector<float> & out) {
        const int64_t NKVC = NBASE + nq;
        const int64_t nqpad = ((nq + 63)/64)*64;
        const size_t msz = ggml_tensor_overhead()*64 + ggml_graph_overhead();
        std::vector<uint8_t> mb(msz);
        ggml_init_params ip = { msz, mb.data(), true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq,  H,   1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKVC, HKV, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKVC, HKV, 1);
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, NKVC, nqpad, 1, 1);
        ggml_tensor * y = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) D), 0.0f, 0.0f);
        if (!getenv("FA_PREC_F32")) { /* graph default precision */ } else ggml_flash_attn_ext_set_prec(y, GGML_PREC_F32);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        // q is [D, nq, H, 1]: token t of head h at (h*nq + t)*D -> build per-nq layout from qf (head-major source)
        std::vector<float> qq((size_t)(D*nq*H));
        for (int64_t h = 0; h < H; ++h)
            for (int64_t t = 0; t < nq; ++t)
                memcpy(&qq[(size_t)((h*nq + t)*D)], &qf[(size_t)((h*NQMAX + t)*D)], (size_t) D*sizeof(float));
        std::vector<uint16_t> mz((size_t)(NKVC*nqpad), 0);
        { const float ninf = -INFINITY; ggml_fp16_t nih; ggml_fp32_to_fp16_row(&ninf, &nih, 1);
          for (int64_t i = 0; i < nqpad; ++i)
            for (int64_t j = 0; j < NKVC; ++j)
                if (j > NBASE + std::min(i, nq-1)) mz[(size_t)(i*NKVC + j)] = *(const uint16_t *)&nih; }
        // K/V rows are head-major [D, NKVC, HKV]; take the first NKVC of the NKV-row source per head
        std::vector<uint16_t> kk((size_t)(D*NKVC*HKV)), vv((size_t)(D*NKVC*HKV));
        for (int64_t h = 0; h < HKV; ++h) {
            memcpy(&kk[(size_t)(h*NKVC*D)], &kh[(size_t)(h*NKV*D)], (size_t)(NKVC*D)*sizeof(uint16_t));
            memcpy(&vv[(size_t)(h*NKVC*D)], &vh[(size_t)(h*NKV*D)], (size_t)(NKVC*D)*sizeof(uint16_t));
        }
        ggml_backend_tensor_set(q, qq.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k, kk.data(), 0, ggml_nbytes(k));
        ggml_backend_tensor_set(v, vv.data(), 0, ggml_nbytes(v));
        ggml_backend_tensor_set(m, mz.data(), 0, ggml_nbytes(m));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "fa failed\n"); exit(1); }
        out.assign((size_t)(D*H), 0.0f);           // dst [D, H, nq, 1] -> token 0 is the first D*H floats
        ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));
        ggml_backend_buffer_free(buf); ggml_free(ctx);
    };
    { std::vector<float> y1, yn; run_fa(1, y1); double prev = -1.0;
      for (int64_t n = 2; n <= NQMAX; ++n) { run_fa(n, yn); cmp0("flash_attn_ext", n, y1, yn, &prev); } }

    // ---- ssm_conv: d_conv=4, d_inner=8192 ----
    const int64_t DC = 4, DI = 8192, NTMAX = 6;
    std::vector<float> sxf((size_t)((DC - 1 + NTMAX)*DI)), cf((size_t)(DC*DI));
    for (auto & x : sxf) x = nd(rng);
    for (auto & x : cf)  x = nd(rng);
    auto run_conv = [&](int64_t nt, std::vector<float> & out) {
        const size_t msz = ggml_tensor_overhead()*64 + ggml_graph_overhead();
        std::vector<uint8_t> mb(msz);
        ggml_init_params ip = { msz, mb.data(), true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * sx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, DC - 1 + nt, DI, 1);
        ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, DC, DI);
        ggml_tensor * y  = ggml_ssm_conv(ctx, sx, c);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<float> sxx((size_t)((DC - 1 + nt)*DI));
        for (int64_t d = 0; d < DI; ++d)
            memcpy(&sxx[(size_t)(d*(DC - 1 + nt))], &sxf[(size_t)(d*(DC - 1 + NTMAX))], (size_t)(DC - 1 + nt)*sizeof(float));
        ggml_backend_tensor_set(sx, sxx.data(), 0, ggml_nbytes(sx));
        ggml_backend_tensor_set(c,  cf.data(),  0, ggml_nbytes(c));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "conv failed\n"); exit(1); }
        out.assign((size_t) DI, 0.0f);             // dst [DI, nt, 1] -> token 0 is the first DI floats
        ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));
        ggml_backend_buffer_free(buf); ggml_free(ctx);
    };
    { std::vector<float> y1, yn; run_conv(1, y1); double prev = -1.0;
      for (int64_t n = 2; n <= NTMAX; ++n) { run_conv(n, yn); cmp0("ssm_conv", n, y1, yn, &prev); } }
    return 0;
}

int main(int argc, char ** argv) {
    Shape sh;
    std::string bk = "cpu";
    unsigned seed = 1234;
    bool do_mulmat = false;
    bool do_ops = false;
    int64_t mmK = 4096, mmN = 512;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]() { return atoll(argv[++i]); };
        if      (a == "--backend") bk = argv[++i];
        else if (a == "-S")   sh.S  = nx();
        else if (a == "--hk") sh.Hk = nx();
        else if (a == "--hv") sh.Hv = nx();
        else if (a == "-T")   sh.T  = nx();
        else if (a == "--seq") sh.B = nx();
        else if (a == "--kda") sh.kda = true;
        else if (a == "--seed") seed = (unsigned) nx();
        else if (a == "--mulmat") do_mulmat = true;
        else if (a == "--ops") do_ops = true;
        else if (a == "-K") mmK = nx();
        else if (a == "-N") mmN = nx();
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    const int64_t S = sh.S, Hk = sh.Hk, Hv = sh.Hv, T = sh.T, B = sh.B;
    const int64_t GS = sh.kda ? S : 1;

    std::mt19937 rng(seed);
    std::normal_distribution<float>       nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    Inputs in;
    in.q.resize((size_t)(S*Hk*T*B)); in.k.resize(in.q.size());
    in.v.resize((size_t)(S*Hv*T*B));
    in.g.resize((size_t)(GS*Hv*T*B));
    in.b.resize((size_t)(Hv*T*B));
    in.s0.resize((size_t)(S*S*Hv*B));

    for (auto & x : in.v)  x = nd(rng);
    for (auto & x : in.s0) x = 0.5f*nd(rng);
    // q,k: L2-normalised per (head, token), exactly like build_gdn_l2_norm feeds them
    for (auto * vec : { &in.q, &in.k }) {
        for (size_t r = 0; r < vec->size(); r += (size_t) S) {
            double n2 = 0.0;
            for (int64_t i = 0; i < S; ++i) { (*vec)[r+i] = nd(rng); n2 += (double)(*vec)[r+i]*(*vec)[r+i]; }
            const float inv = (float)(1.0/std::sqrt(n2 + 1e-6));
            for (int64_t i = 0; i < S; ++i) (*vec)[r+i] *= inv;
        }
    }
    // gate: log-decay in [-4, 0) like -A.exp()*softplus(...)
    for (auto & x : in.g) x = -std::exp(-2.0f + 2.0f*ud(rng));
    // beta = sigmoid(N(0,1))
    for (auto & x : in.b) x = 1.0f/(1.0f + std::exp(-nd(rng)));

    std::vector<double> ref_out, ref_state;
    ref_gdn(sh, in.q, in.k, in.v, in.g, in.b, in.s0, ref_out, ref_state);

    ggml_backend_t backend = nullptr;
    if (bk == "cpu") {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
            backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) break;
        }
    }
    if (!backend) { fprintf(stderr, "no backend %s\n", bk.c_str()); return 1; }
    if (do_ops)    { int r = ops_main(backend, seed); ggml_backend_free(backend); return r; }
    if (do_mulmat) { int r = mulmat_main(backend, mmK, mmN, 6, seed); ggml_backend_free(backend); return r; }
    printf("backend: %s | S=%lld Hk=%lld Hv=%lld T=%lld B=%lld kda=%d seed=%u\n",
           ggml_backend_name(backend), (long long) S, (long long) Hk, (long long) Hv,
           (long long) T, (long long) B, (int) sh.kda, seed);

    const size_t attn_elems  = (size_t)(S*Hv*T*B);
    const size_t state_elems = (size_t)(S*S*Hv*B);

    struct Row { const char * name; double eo, ro, es, rs; };
    std::vector<Row> rows;

    auto eval = [&](const char * name, Mode mode, bool stepped) {
        std::vector<float> out(attn_elems, 0.0f);
        std::vector<float> st = in.s0;
        if (stepped) {
            std::vector<float> o1((size_t)(S*Hv*B));
            for (int64_t t = 0; t < T; ++t) {
                run_once(backend, sh, 1, mode, in, t, st, o1.data());
                for (int64_t ib = 0; ib < B; ++ib)
                    memcpy(&out[(size_t)(S*Hv*(t + T*ib))], &o1[(size_t)(S*Hv*ib)], (size_t)(S*Hv)*sizeof(float));
            }
        } else {
            run_once(backend, sh, T, mode, in, 0, st, out.data());
        }
        double ro = 0.0, rs = 0.0;
        const double eo = maxabs_diff(out.data(), ref_out,   attn_elems,  &ro);
        const double es = maxabs_diff(st.data(),  ref_state, state_elems, &rs);
        rows.push_back({name, eo, ro, es, rs});
        printf("%-14s | %-12.3e %-12.3e | %-12.3e %-12.3e\n", name, eo, ro, es, rs);
        fflush(stdout);
    };

    printf("%-14s | %-12s %-12s | %-12s %-12s\n", "path", "out maxabs", "out maxrel", "state maxabs", "state maxrel");
    eval("fused  T-in-1 ", MODE_FUSED, false);
    eval("fused  1x-step", MODE_FUSED, true);
    eval("chunk  T-in-1 ", MODE_CHUNK, false);
    eval("chunk  1x-step", MODE_CHUNK, true);
    if (!sh.kda) eval("unfsd  ar 1x  ", MODE_AR,    true);


    ggml_backend_free(backend);
    return 0;
}
