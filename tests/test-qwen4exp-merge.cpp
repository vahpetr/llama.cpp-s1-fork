// [TAG_MERGE_ROWS] exactness evidence for the qwen4exp load-time projection merges (upstream #31028), through the REAL
// scheduler path (ggml_backend_sched: GPU + CPU, so ggml-vulkan's graph_optimize and fusions run).
//
// One qwen4exp-shaped block per weight type is built twice from the same weight bytes:
//   sep  the separate tensors (each its own allocation) and their consumers, node for node as qwen4exp builds them
//        without the merge: hc down -> SCALE -> SILU, hc inject -> the combine chain (SCALE SIGMOID SCALE RESHAPE MUL ADD),
//        qkv -> concat(conv history, x^T) -> ssm_conv -> silu + the history tail write-back (build_conv_state_at with
//        one tail), z -> sigmoid * rms_norm(o) * w, beta -> sigmoid,
//        alpha -> ADD(dt) -> SOFTPLUS -> MUL(a)
//   mrg  the weights as llama_model_loader::merge_rows lays them out (one container, the halves as leaf views of it) and
//        the graph qwen4exp builds with LLAMA_QWEN4EXP_MERGE on: one MUL_MAT over the container cut into views when
//        nt <= max_n (default 8), else the two MUL_MATs over the views
// and every output is compared bytewise (fnv1a + memcmp); each arm also runs twice and must repeat itself bytewise.
// Exit code 1 on any difference.
//
// usage: test-qwen4exp-merge [max_n=8] [nt list=1,2,3,4,5,6,7,8,9,16,64,512]
//   max_n > 8 is the negative control: the merged GEMM is then used above the mat-vec range, where the tile / split_k
//   choice depends on the row count (expected to differ for some types; a difference there is reported, not failed:
//   the run fails only for nt <= 8 or for nt > max_n).
// Output: "[merge-test] pair=<hc|ba|qkvz> type=<t> nt=<n> path=<merged|views> out=<name> equal=<0|1> sep=0x.. mrg=0x.."
// and one summary line "[merge-test] SUMMARY cases=<n> equal=<n> differ=<n> fail=<n>".
// Exits 0 without output if no GPU device is present.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int64_t N_EMBD  = 2560;
static const int64_t HC      = 4;
static const int64_t HC_DIM  = HC*N_EMBD;
static const int64_t RANK    = 320;
static const int64_t N_QKV   = 10240;   // key_dim * 2 + value_dim
static const int64_t N_V     = 48;      // value heads
static const int64_t HEAD_V  = 128;
static const int64_t N_Z     = N_V*HEAD_V;

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const uint8_t *) p)[i]) * 0x100000001b3ull;
    }
    return h;
}

static std::vector<float> rnd(size_t n, uint32_t seed, float scale) {
    std::vector<float> v(n);
    uint32_t s = seed * 2654435761u + 12345u;
    for (auto & e : v) {
        s = s * 1664525u + 1013904223u;
        e = scale * (((s >> 8) & 0xffff) / 32768.0f - 1.0f);
    }
    return v;
}

// quantized bytes of a [k, m] weight (rows of k), deterministic
static std::vector<uint8_t> qbytes(ggml_type type, int64_t k, int64_t m, uint32_t seed) {
    const std::vector<float> f = rnd((size_t) (k*m), seed, 2.0f/sqrtf((float) k));
    std::vector<uint8_t> q(ggml_row_size(type, k)*m);
    ggml_quantize_chunk(type, f.data(), q.data(), 0, m, k, nullptr);
    return q;
}

static void set_f32(ggml_tensor * t, uint32_t seed, float scale) {
    const std::vector<float> v = rnd((size_t) ggml_nelements(t), seed, scale);
    ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(float));
}

struct pair_w {
    ggml_tensor * a = nullptr;      // sep: own allocation
    ggml_tensor * b = nullptr;
    ggml_tensor * m = nullptr;      // mrg: container
    ggml_tensor * ma = nullptr;     // mrg: leaf views
    ggml_tensor * mb = nullptr;
};

// the separate pair + the merged layout of the same bytes, as merge_rows makes it
static void make_pair(ggml_context * ctx, ggml_type type, int64_t k, int64_t n_a, int64_t n_b, pair_w & p) {
    p.a  = ggml_new_tensor_2d(ctx, type, k, n_a);
    p.b  = ggml_new_tensor_2d(ctx, type, k, n_b);
    p.m  = ggml_new_tensor_2d(ctx, type, k, n_a + n_b);
    p.mb = ggml_new_tensor_2d(ctx, type, k, n_b);
    p.mb->view_src  = p.m;
    p.mb->view_offs = ggml_row_size(type, k)*n_a;
    p.ma = ggml_new_tensor_2d(ctx, type, k, n_a);
    p.ma->view_src  = p.m;
    p.ma->view_offs = 0;
}

static void upload_pair(const pair_w & p, uint32_t seed) {
    const std::vector<uint8_t> qa = qbytes(p.a->type, p.a->ne[0], p.a->ne[1], seed);
    const std::vector<uint8_t> qb = qbytes(p.b->type, p.b->ne[0], p.b->ne[1], seed + 1);
    ggml_backend_tensor_set(p.a,  qa.data(), 0, qa.size());
    ggml_backend_tensor_set(p.b,  qb.data(), 0, qb.size());
    ggml_backend_tensor_set(p.ma, qa.data(), 0, qa.size());
    ggml_backend_tensor_set(p.mb, qb.data(), 0, qb.size());
}

struct outs {
    std::vector<std::pair<std::string, ggml_tensor *>> list;
    void add(const char * name, ggml_tensor * t) { ggml_set_output(t); list.emplace_back(name, t); }
};

static int g_cases = 0, g_equal = 0, g_differ = 0, g_fail = 0;

int main(int argc, char ** argv) {
    const int64_t max_n = argc > 1 ? atoll(argv[1]) : 8;
    std::vector<int64_t> nts = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 64, 512 };
    if (argc > 2) {
        nts.clear();
        std::string l = argv[2];
        size_t pos = 0;
        while (pos <= l.size()) {
            const size_t e = std::min(l.find(',', pos), l.size());
            if (e > pos) {
                nts.push_back(atoll(l.substr(pos, e - pos).c_str()));
            }
            pos = e + 1;
        }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev == nullptr) {
        gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
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

    // the type sets of the two production GGUFs: hands IQ3_S-hc8-o8 (hc Q8_0, beta/alpha BF16, qkv/gate Q6_K / Q5_K /
    // Q4_K / IQ4_XS where both match) and the brain i1-Q4_K_M (hc Q4_K, beta/alpha Q4_K, qkv/gate Q4_K), plus Q6_K hc
    struct cfg { const char * pair; ggml_type type; };
    const cfg cfgs[] = {
        { "hc",   GGML_TYPE_Q8_0 }, { "hc",   GGML_TYPE_Q4_K }, { "hc",   GGML_TYPE_Q6_K },
        { "ba",   GGML_TYPE_BF16 }, { "ba",   GGML_TYPE_Q4_K },
        { "qkvz", GGML_TYPE_Q6_K }, { "qkvz", GGML_TYPE_Q5_K }, { "qkvz", GGML_TYPE_Q4_K }, { "qkvz", GGML_TYPE_IQ4_XS },
    };

    for (const cfg & c : cfgs) {
        const std::string pair = c.pair;
        ggml_init_params wp = { ggml_tensor_overhead()*32, nullptr, true };
        ggml_context * ctx_w = ggml_init(wp);
        pair_w P;
        ggml_tensor * up = nullptr, * dt = nullptr, * sa = nullptr, * conv_w = nullptr, * norm_w = nullptr, * conv_store = nullptr;
        int64_t k = 0, n_a = 0, n_b = 0;
        if (pair == "hc")   { k = HC_DIM; n_a = RANK;  n_b = HC; }
        if (pair == "ba")   { k = N_EMBD; n_a = N_V;   n_b = N_V; }
        if (pair == "qkvz") { k = N_EMBD; n_a = N_QKV; n_b = N_Z; }
        make_pair(ctx_w, c.type, k, n_a, n_b, P);
        if (pair == "hc") {
            up = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, RANK, HC_DIM);
        }
        if (pair == "ba") {
            dt = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, N_V);
            sa = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, N_V);
        }
        if (pair == "qkvz") {
            conv_w = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 4, N_QKV);
            norm_w = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, HEAD_V);
            conv_store = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, 3*N_QKV);
        }
        ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, gpu);
        if (buf_w == nullptr) {
            fprintf(stderr, "weight alloc failed\n");
            return 1;
        }
        upload_pair(P, 17);
        if (up)     { set_f32(up, 3, 0.05f); }
        if (dt)     { set_f32(dt, 4, 1.0f); }
        if (sa)     { set_f32(sa, 5, 1.0f); }
        if (conv_w) { set_f32(conv_w, 6, 0.5f); }
        if (norm_w) { set_f32(norm_w, 7, 1.0f); }

        for (const int64_t nt : nts) {
            std::vector<std::vector<uint8_t>> res[2];
            std::vector<std::string> names;
            bool merged_path = false;
            for (int arm = 0; arm < 2; ++arm) {
                const size_t graph_size = 1024;
                ggml_init_params gp = { ggml_tensor_overhead()*graph_size + ggml_graph_overhead_custom(graph_size, false), nullptr, true };
                ggml_context * ctx = ggml_init(gp);
                ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);
                outs O;
                const bool use_m = arm == 1 && nt <= max_n;
                merged_path = nt <= max_n;
                ggml_tensor * wa = arm == 0 ? P.a : P.ma;
                ggml_tensor * wb = arm == 0 ? P.b : P.mb;

                std::vector<ggml_tensor *> inputs;
                auto input = [&](ggml_tensor * t) { ggml_set_input(t); inputs.push_back(t); return t; };

                if (pair == "hc") {
                    // build_hc_mix's projections + build_hc_combine (residual [n_embd, hc, nt], block [n_embd, nt])
                    ggml_tensor * xn  = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HC_DIM, nt));
                    ggml_tensor * res_in = input(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, HC, nt));
                    ggml_tensor * blk = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, nt));
                    ggml_tensor * lo = nullptr, * inj = nullptr;
                    if (use_m) {
                        ggml_tensor * li = ggml_mul_mat(ctx, P.m, xn);
                        lo  = ggml_view_2d(ctx, li, RANK, nt, li->nb[1], 0);
                        inj = ggml_view_2d(ctx, li, HC,   nt, li->nb[1], ggml_row_size(li->type, RANK));
                    } else {
                        lo = ggml_mul_mat(ctx, wa, xn);
                    }
                    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f/(float) HC));
                    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, up, lo));
                    O.add("hc_gate", gate);
                    ggml_build_forward_expand(gf, gate);
                    if (!inj) {
                        inj = ggml_mul_mat(ctx, wb, xn);
                    }
                    // build_hc_combine
                    ggml_build_forward_expand(gf, inj);
                    ggml_tensor * w = ggml_sigmoid(ctx, ggml_scale(ctx, inj, 1.0f/(float) HC));
                    w = ggml_scale(ctx, w, 2.0f);
                    w = ggml_reshape_3d(ctx, w, 1, HC, nt);
                    ggml_tensor * b = ggml_reshape_3d(ctx, blk, N_EMBD, 1, nt);
                    b = ggml_repeat_4d(ctx, b, N_EMBD, HC, nt, 1);
                    ggml_tensor * out = ggml_add(ctx, res_in, ggml_mul(ctx, b, w));
                    O.add("hc_combine", out);
                    ggml_build_forward_expand(gf, out);
                } else if (pair == "ba") {
                    ggml_tensor * cur = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, nt));
                    ggml_tensor * ba = use_m ? ggml_mul_mat(ctx, P.m, cur) : nullptr;
                    ggml_tensor * beta = ba ?
                        ggml_view_4d(ctx, ba, 1, N_V, nt, 1, ggml_element_size(ba), ba->nb[1], ba->nb[1]*nt, 0) :
                        ggml_reshape_4d(ctx, ggml_mul_mat(ctx, wa, cur), 1, N_V, nt, 1);
                    beta = ggml_sigmoid(ctx, beta);
                    ggml_tensor * alpha = ba ?
                        ggml_view_3d(ctx, ba, N_V, nt, 1, ba->nb[1], ba->nb[1]*nt, ggml_row_size(ba->type, N_V)) :
                        ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wb, cur), N_V, nt, 1);
                    ggml_tensor * g = ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, dt)), sa);
                    // the GDN consumes both; here two outputs, gate first as in the model's src order
                    O.add("gate", g);
                    O.add("beta_sigmoid", beta);
                    ggml_build_forward_expand(gf, g);
                    ggml_build_forward_expand(gf, beta);
                } else {
                    ggml_tensor * cur   = input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, nt));
                    ggml_tensor * o     = input(ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_V, N_V, nt, 1));
                    ggml_tensor * qkv = nullptr, * z = nullptr;
                    if (use_m) {
                        ggml_tensor * qkvz = ggml_mul_mat(ctx, P.m, cur);
                        qkv = ggml_view_3d(ctx, qkvz, N_QKV, nt, 1, qkvz->nb[1], qkvz->nb[1]*nt, 0);
                        z   = ggml_view_2d(ctx, qkvz, N_Z, nt, qkvz->nb[1], ggml_row_size(qkvz->type, N_QKV));
                    } else {
                        qkv = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, wa, cur), N_QKV, nt, 1);
                        z   = ggml_mul_mat(ctx, wb, cur);
                    }
                    // build_conv_state_at + the conv + silu as qwen4exp builds them with n_rs_seq == 0 (one tail): the
                    // history row is gathered first; from 3 tokens on the tail is taken from x^T and copied AFTER the conv
                    // ([TAG_CONV_INPUT_SPLIT], keeps x alive across the fused conv), below that from the concat
                    ggml_tensor * state = ggml_reshape_3d(ctx, ggml_cont(ctx, conv_store), 3, N_QKV, 1);
                    ggml_build_forward_expand(gf, state);
                    ggml_tensor * ci   = ggml_concat(ctx, state, ggml_transpose(ctx, qkv), 0);
                    ggml_tensor * dst  = ggml_view_1d(ctx, conv_store, 3*N_QKV, 0);
                    ggml_tensor * conv = nullptr;
                    if (nt >= 3) {
                        conv = ggml_silu(ctx, ggml_ssm_conv(ctx, ci, conv_w));
                        ggml_build_forward_expand(gf, conv);
                        ggml_tensor * tail = ggml_transpose(ctx, ggml_view_3d(ctx, qkv, N_QKV, 3, 1, qkv->nb[1], qkv->nb[2], (nt - 3)*qkv->nb[1]));
                        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, tail), dst));
                    } else {
                        ggml_tensor * tail = ggml_view_3d(ctx, ci, 3, N_QKV, 1, ci->nb[1], ci->nb[2], ggml_row_size(ci->type, ci->ne[0] - 3));
                        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, tail), dst));
                        conv = ggml_silu(ctx, ggml_ssm_conv(ctx, ci, conv_w));
                        ggml_build_forward_expand(gf, conv);
                    }
                    O.add("conv_silu", conv);
                    // build_norm_gated(output, ssm_norm, z_2d): rms_norm * w, sigmoid(z), mul
                    ggml_tensor * z2 = ggml_is_contiguous(z) ?
                        ggml_reshape_4d(ctx, z, HEAD_V, N_V, nt, 1) :
                        ggml_view_4d(ctx, z, HEAD_V, N_V, nt, 1, ggml_row_size(z->type, HEAD_V), z->nb[1], z->nb[1]*nt, 0);
                    ggml_tensor * nrm = ggml_mul(ctx, ggml_rms_norm(ctx, o, 1e-6f), norm_w);
                    ggml_tensor * gated = ggml_mul(ctx, nrm, ggml_sigmoid(ctx, z2));
                    O.add("norm_gated", gated);
                    ggml_build_forward_expand(gf, gated);
                }

                ggml_backend_t backends[2] = { gpu, cpu };
                ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, graph_size, false, true);
                if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                    fprintf(stderr, "alloc failed\n");
                    return 1;
                }
                std::vector<std::vector<uint8_t>> rep[2];
                for (int r = 0; r < 2; ++r) {
                    uint32_t seed = 1000;
                    for (ggml_tensor * t : inputs) {
                        set_f32(t, seed++, 1.0f);
                    }
                    if (conv_store) {
                        set_f32(conv_store, 8, 1.0f);
                    }
                    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "compute failed\n");
                        return 1;
                    }
                    names.clear();
                    for (auto & [name, t] : O.list) {
                        std::vector<uint8_t> v(ggml_nbytes(t));
                        ggml_backend_tensor_get(t, v.data(), 0, v.size());
                        rep[r].push_back(std::move(v));
                        names.push_back(name);
                    }
                    if (conv_store) {
                        std::vector<uint8_t> v(ggml_nbytes(conv_store));
                        ggml_backend_tensor_get(conv_store, v.data(), 0, v.size());
                        rep[r].push_back(std::move(v));
                        names.push_back("conv_state");
                    }
                }
                // run-to-run: the same graph twice must give the same bytes (a hazard would show here first)
                for (size_t i = 0; i < rep[0].size(); ++i) {
                    if (rep[0][i] != rep[1][i]) {
                        printf("[merge-test] pair=%s type=%s nt=%" PRId64 " arm=%s out=%s RUN-TO-RUN MISMATCH\n",
                               c.pair, ggml_type_name(c.type), nt, arm == 0 ? "sep" : "mrg", names[i].c_str());
                        g_fail++;
                    }
                }
                res[arm] = std::move(rep[0]);
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
            }
            for (size_t i = 0; i < names.size(); ++i) {
                const bool eq = res[0][i].size() == res[1][i].size() &&
                                memcmp(res[0][i].data(), res[1][i].data(), res[0][i].size()) == 0;
                const bool must = nt <= 8 || nt > max_n;
                g_cases++;
                if (eq) {
                    g_equal++;
                } else {
                    g_differ++;
                    if (must) {
                        g_fail++;
                    }
                }
                printf("[merge-test] pair=%s type=%s nt=%" PRId64 " path=%s out=%s equal=%d sep=0x%016" PRIx64 " mrg=0x%016" PRIx64 "%s\n",
                       c.pair, ggml_type_name(c.type), nt, merged_path ? "merged" : "views", names[i].c_str(), eq ? 1 : 0,
                       fnv1a(res[0][i].data(), res[0][i].size()), fnv1a(res[1][i].data(), res[1][i].size()),
                       (!eq && !must) ? " (negctl range)" : "");
                fflush(stdout);
            }
        }
        ggml_backend_buffer_free(buf_w);
        ggml_free(ctx_w);
    }
    printf("[merge-test] SUMMARY max_n=%" PRId64 " cases=%d equal=%d differ=%d fail=%d\n", max_n, g_cases, g_equal, g_differ, g_fail);

    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return g_fail == 0 ? 0 : 1;
}
