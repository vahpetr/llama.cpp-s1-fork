// [TAG_HC_MIX_TAIL] graph-order evidence for the Vulkan fusions, through the REAL scheduler path.
//
// test-backend-ops computes a graph with ggml_backend_compare_graph_backend, which never runs the backend's
// graph_optimize, so neither the node order ggml_vk_graph_optimize produces nor its carve-outs are exercised
// there. This program builds a qwen4exp-shaped graph in the model's own expansion order - two layers of
// build_hc_mix (+ the early forward-expand of the mixed stream), a block, build_hc_combine (inject expanded
// first), the ffn mix and combine, plus the QSA pooled-key mean (get_rows -> r CONT slices -> ADD chain ->
// SCALE(1/r) -> RMS_NORM) and, in layer 0, the GDN conv input (concat(history, x^T) -> ssm_conv -> silu, the
// history tail copied from x^T after the conv, [TAG_CONV_INPUT_SPLIT]) - and computes it with
// ggml_backend_sched (GPU + CPU), i.e. with graph_optimize. The conv history row is hashed too (conv_state=).
//
// Output (stdout): one line per run
//   [order-test] nt=<n> rounds=<k> out=0x<fnv1a of the final stream> pool=0x<fnv1a of the pool output>
// With GGML_VK_FUSION_COUNT=1 the backend adds its per-graph line (pattern counts + the dispatch signature
// sig=, which hashes the pipeline name and workgroup counts of every dispatch in order). The same source
// builds against any fork revision, so two binaries (e.g. v59 and a candidate) can be compared line by line:
// same sig = same dispatch sequence; same out/pool = same bits.
//
// usage: test-vk-graph-order [n_tokens=64] [rounds=1] [b4|v59]   (rounds > 1 re-runs graph_optimize on the permuted
//        graph in one process, which changes the fusions in every revision incl. v59 - a pre-existing effect; keep 1)
// Pinned v59 reference: tests/fixtures/vk-graph-order-v59.txt (+ the producing command), checked by
// tests/test-vk-graph-order.sh.
// Exits 0 without output if no GPU device is present (not a ctest: the CPU ctest container has no GPU).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const int64_t N_EMBD  = 2560;
static const int64_t HC      = 4;
static const int64_t RANK    = 320;
static const int64_t IDX_DIM = 128;
static const int64_t POOL_R  = 4;
static const int64_t N_ROWS  = 512;   // pooled blocks
static const int     N_LAYER = 2;
static const int64_t CONV_CH = 2048;  // GDN conv channels (qwen4exp: 10240), kernel 4

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const uint8_t *) p)[i]) * 0x100000001b3ull;
    }
    return h;
}

struct layer_w {
    ggml_tensor * norm[2], * down[2], * up[2], * inject[2];
    ggml_tensor * g_attn, * w_attn, * g_ffn, * w_ffn1, * w_ffn2;
    ggml_tensor * w_qkv, * conv_w, * w_out;   // the GDN-shaped layer 0
};

struct model_t {
    layer_w       l[N_LAYER];
    ggml_tensor * k_all;       // [IDX_DIM, n_cells] f32
    ggml_tensor * conv_store;  // [3*CONV_CH] f32: the conv history row (read, then the tail written back)
};

// build_conv_state_at + the GDN conv (n_rs_seq == 0, prefill): concat(state, x^T) -> ssm_conv -> silu and the
// history tail written back. Two forms, chosen by the third argument:
//   v59  the tail is a view of the concat, expanded inside build_conv_state_at, i.e. BEFORE the conv (a1b6c324e)
//   b4   [TAG_CONV_INPUT_SPLIT] the tail is transpose(view of x), expanded AFTER the conv (qwen4exp at stage8 with
//        GGML_VK_FUSE_CONCAT_SSM_CONV on)
// Both must write the same history bytes (conv_state=) and give the same output.
static bool g_v59_form = false;
static ggml_tensor * gdn_conv(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * store, ggml_tensor * x, ggml_tensor * w) {
    const int64_t nt = x->ne[1];
    // the history is read (build_rs's gather) before the tail overwrites it
    ggml_tensor * state = ggml_reshape_3d(ctx, ggml_cont(ctx, store), 3, CONV_CH, 1);
    ggml_build_forward_expand(gf, state);
    ggml_tensor * x_t = ggml_transpose(ctx, x);
    ggml_tensor * ci  = ggml_concat(ctx, state, x_t, 0);
    ggml_tensor * dst = ggml_view_1d(ctx, store, 3*CONV_CH, 0);
    if (g_v59_form) {
        ggml_tensor * tail = ggml_view_3d(ctx, ci, 3, CONV_CH, 1, ci->nb[1], ci->nb[2], ggml_row_size(ci->type, ci->ne[0] - 3));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, tail), dst));
        return ggml_silu(ctx, ggml_ssm_conv(ctx, ci, w));
    }
    ggml_tensor * y = ggml_silu(ctx, ggml_ssm_conv(ctx, ci, w));
    if (nt >= 3) {
        ggml_build_forward_expand(gf, y);
        ggml_tensor * tail = ggml_transpose(ctx, ggml_view_3d(ctx, x, CONV_CH, 3, 1, x->nb[1], x->nb[2], (nt - 3)*x->nb[1]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, tail), dst));
    } else {
        ggml_tensor * tail = ggml_view_3d(ctx, ci, 3, CONV_CH, 1, ci->nb[1], ci->nb[2], ggml_row_size(ci->type, ci->ne[0] - 3));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, tail), dst));
    }
    return y;
}

// build_hc_mix, node for node (src/models/qwen4exp.cpp)
static ggml_tensor * hc_mix(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w_norm, ggml_tensor * w_down,
                            ggml_tensor * w_up, ggml_tensor * w_inject, ggml_tensor ** inject) {
    const int64_t nt = x->ne[2];
    ggml_tensor * xn = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w_norm);
    xn = ggml_reshape_2d(ctx, xn, HC*N_EMBD, nt);
    ggml_tensor * lo = ggml_mul_mat(ctx, w_down, xn);
    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f/(float) HC));
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));
    ggml_tensor * gated = ggml_mul(ctx, xn, gate);
    gated = ggml_reshape_3d(ctx, gated, N_EMBD, HC, nt);
    ggml_tensor * mixed = ggml_view_2d(ctx, gated, N_EMBD, nt, ggml_row_size(gated->type, N_EMBD)*HC, 0);
    for (int64_t c = 1; c < HC; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx, gated, N_EMBD, nt, ggml_row_size(gated->type, N_EMBD)*HC,
                                       ggml_row_size(gated->type, N_EMBD)*c);
        mixed = ggml_add(ctx, mixed, s);
    }
    mixed = ggml_scale(ctx, mixed, 1.0f/(float) HC);
    *inject = ggml_mul_mat(ctx, w_inject, xn);
    return mixed;
}

// build_hc_combine, node for node
static ggml_tensor * hc_combine(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * residual, ggml_tensor * block_out,
                                ggml_tensor * inject) {
    const int64_t nt = residual->ne[2];
    ggml_build_forward_expand(gf, inject);
    ggml_tensor * w = ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f/(float) HC));
    w = ggml_scale(ctx, w, 2.0f);
    w = ggml_reshape_3d(ctx, w, 1, HC, nt);
    ggml_tensor * b = ggml_reshape_3d(ctx, block_out, N_EMBD, 1, nt);
    b = ggml_repeat_4d(ctx, b, N_EMBD, HC, nt, 1);
    return ggml_add(ctx, residual, ggml_mul(ctx, b, w));
}

// build_qsa_pool's mean (n_stream = 1): r CONT slices, ADD chain, SCALE(1/r), then the rms_norm
static ggml_tensor * pool_mean(ggml_context * ctx, ggml_tensor * k_all, ggml_tensor * cells) {
    ggml_tensor * members = ggml_get_rows(ctx, k_all, cells);
    members = ggml_reshape_4d(ctx, members, IDX_DIM, POOL_R, N_ROWS, 1);
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < POOL_R; ++i) {
        ggml_tensor * slice = ggml_cont(ctx, ggml_view_3d(ctx, members, IDX_DIM, N_ROWS, 1,
                                                           members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx, pooled, 1.0f/(float) POOL_R);
    pooled = ggml_reshape_3d(ctx, pooled, IDX_DIM, N_ROWS, 1);
    return ggml_rms_norm(ctx, pooled, 1e-6f);
}

static void fill(ggml_tensor * t, uint32_t seed, float scale) {
    std::vector<float> v(ggml_nelements(t));
    uint32_t s = seed * 2654435761u + 12345u;
    for (auto & e : v) {
        s = s * 1664525u + 1013904223u;
        e = scale * (((s >> 8) & 0xffff) / 32768.0f - 1.0f);
        if ((s & 63) == 0) {
            e = (s & 64) ? -0.0f : 0.0f;   // exact zeros of both signs
        }
    }
    ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(float));
}

int main(int argc, char ** argv) {
    const int64_t nt     = argc > 1 ? atoll(argv[1]) : 64;
    const int     rounds = argc > 2 ? atoi(argv[2]) : 1;
    g_v59_form           = argc > 3 && strcmp(argv[3], "v59") == 0;

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev == nullptr) {
        gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);   // UMA parts (780M, Strix Halo)
    }
    if (gpu_dev == nullptr) {
        fprintf(stderr, "no GPU device - skipped\n");
        return 0;
    }
    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (gpu == nullptr || cpu == nullptr) {
        fprintf(stderr, "backend init failed\n");
        return 1;
    }

    // weights in a GPU buffer
    ggml_init_params wp = { ggml_tensor_overhead()*96, nullptr, true };
    ggml_context * ctx_w = ggml_init(wp);
    model_t m;
    for (int il = 0; il < N_LAYER; ++il) {
        layer_w & L = m.l[il];
        for (int k = 0; k < 2; ++k) {
            L.norm[k]   = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, N_EMBD, HC);
            L.down[k]   = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, HC*N_EMBD, RANK);
            L.up[k]     = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, RANK, HC*N_EMBD);
            L.inject[k] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, HC*N_EMBD, HC);
        }
        L.g_attn = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, N_EMBD);
        L.w_attn = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, N_EMBD, N_EMBD);
        L.g_ffn  = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, N_EMBD);
        L.w_ffn1 = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, N_EMBD, 1024);
        L.w_ffn2 = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 1024, N_EMBD);
        L.w_qkv  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, N_EMBD, CONV_CH);
        L.conv_w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 4, CONV_CH);
        L.w_out  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, CONV_CH, N_EMBD);
    }
    m.conv_store = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 3*CONV_CH);
    const int64_t n_cells = POOL_R*N_ROWS;
    m.k_all = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, IDX_DIM, n_cells);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, gpu);
    {
        uint32_t seed = 1;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx_w); t; t = ggml_get_next_tensor(ctx_w, t)) {
            const float sc = t->ne[1] == HC && t->ne[0] == N_EMBD ? 0.5f : 2.0f/sqrtf((float) t->ne[0]);
            fill(t, seed++, sc);
        }
    }

    // the graph, in the model's expansion order
    const size_t graph_size = 8192;
    ggml_init_params gp = { ggml_tensor_overhead()*graph_size + ggml_graph_overhead_custom(graph_size, false), nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, HC, nt);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * cells = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_cells);
    ggml_set_name(cells, "cells");
    ggml_set_input(cells);

    ggml_tensor * res  = x;
    ggml_tensor * pool = nullptr;
    for (int il = 0; il < N_LAYER; ++il) {
        layer_w & L = m.l[il];
        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = hc_mix(ctx, res, L.norm[0], L.down[0], L.up[0], L.inject[0], &inject);
        ggml_build_forward_expand(gf, cur);
        if (il == 1) {
            // the QSA layer: its indexer pools the keys before the attention reads them
            pool = pool_mean(ctx, m.k_all, cells);
            ggml_build_forward_expand(gf, pool);
        }
        if (il == 0) {
            // the GDN layer: qkv projection -> conv over the history | x -> out projection
            ggml_tensor * qkv = ggml_mul_mat(ctx, L.w_qkv, ggml_mul(ctx, ggml_rms_norm(ctx, cur, 1e-6f), L.g_attn));
            qkv = ggml_reshape_3d(ctx, qkv, CONV_CH, nt, 1);
            ggml_tensor * conv = gdn_conv(ctx, gf, m.conv_store, qkv, L.conv_w);
            cur = ggml_mul_mat(ctx, L.w_out, ggml_reshape_2d(ctx, conv, CONV_CH, nt));
        } else {
            cur = ggml_mul_mat(ctx, L.w_attn, ggml_mul(ctx, ggml_rms_norm(ctx, cur, 1e-6f), L.g_attn));
        }
        res = hc_combine(ctx, gf, res, cur, inject);

        cur = hc_mix(ctx, res, L.norm[1], L.down[1], L.up[1], L.inject[1], &inject);
        cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, 1e-6f), L.g_ffn);
        cur = ggml_mul_mat(ctx, L.w_ffn2, ggml_silu(ctx, ggml_mul_mat(ctx, L.w_ffn1, cur)));
        res = hc_combine(ctx, gf, res, cur, inject);
    }
    ggml_set_output(res);
    ggml_set_output(pool);
    ggml_build_forward_expand(gf, res);

    ggml_backend_t backends[2] = { gpu, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, graph_size, false, true);

    uint64_t h_out = 0, h_pool = 0, h_conv = 0;
    for (int r = 0; r < rounds; ++r) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }
        fill(x, 1000, 1.0f);
        fill(m.conv_store, 2000, 1.0f);
        {
            std::vector<int32_t> c(n_cells);
            uint32_t s = 7;
            for (auto & e : c) {
                s = s * 1664525u + 1013904223u;
                e = (int32_t) ((s >> 8) % n_cells);
            }
            ggml_backend_tensor_set(cells, c.data(), 0, c.size()*sizeof(int32_t));
        }
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed\n");
            return 1;
        }
        std::vector<uint8_t> o(ggml_nbytes(res)), p(ggml_nbytes(pool)), cs(ggml_nbytes(m.conv_store));
        ggml_backend_tensor_get(res,  o.data(), 0, o.size());
        ggml_backend_tensor_get(pool, p.data(), 0, p.size());
        ggml_backend_tensor_get(m.conv_store, cs.data(), 0, cs.size());
        const uint64_t ho = fnv1a(o.data(), o.size()), hp = fnv1a(p.data(), p.size());
        h_conv = fnv1a(cs.data(), cs.size());
        if (r > 0 && (ho != h_out || hp != h_pool)) {
            fprintf(stderr, "run-to-run mismatch\n");
            return 1;
        }
        h_out = ho;
        h_pool = hp;
    }
    printf("[order-test] form=%s nt=%" PRId64 " rounds=%d nodes=%d out=0x%016" PRIx64 " pool=0x%016" PRIx64 " conv_state=0x%016" PRIx64 "\n",
           g_v59_form ? "v59" : "b4", nt, rounds, ggml_graph_n_nodes(gf), h_out, h_pool, h_conv);

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return 0;
}
