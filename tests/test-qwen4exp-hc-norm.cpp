// [TAG_UP28896] qwen4exp grouped norms: RMS_NORM and MUL adjacent (upstream #28896 pattern).
//
// A. op level, CPU backend: the old form (rms_norm -> reshape_2d -> mul by the [hc_dim] gamma, and the PLE form with
//    reshape_3d around it) against the new one (rms_norm -> mul by the same gamma viewed as [n_embd, hc]). In the new
//    graph the two nodes are adjacent, so the CPU backend runs its fused RMS_NORM+MUL kernel; the outputs must be
//    bit-identical (the fused kernel computes (x * scale) * w like rms_norm's x * scale followed by the mul).
// B. model level, CPU: a random qwen4exp model (the test-llama-archs fixture: 2 layers, hc = 4, MoE); an eval
//    callback records the node order of the target graph. Every RMS_NORM whose output feeds a MUL must be followed
//    by that MUL directly - no RMS_NORM -> RESHAPE -> MUL chain is left (the old build_hc_mix had one per mixer).
//    Prints an FNV-1a hash of all logits so a build of the old graph can be compared (bit identity end to end).

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

static void fill(ggml_tensor * t, uint32_t seed, float mean, float sd) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(mean, sd);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) {
        d[i] = dis(gen);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// A. op level

static void op_case(int64_t n_embd, int64_t hc, int64_t nt, bool ple_form, int n_threads) {
    const int64_t hc_dim = n_embd * hc;
    const float   eps    = 1e-6f;

    ggml_init_params ip = { /*mem_size*/ (size_t) (8 * hc_dim * nt + 4 * hc_dim) * sizeof(float) + 64 * ggml_tensor_overhead() + 2 * ggml_graph_overhead(),
                            /*mem_buffer*/ nullptr, /*no_alloc*/ false };
    ggml_context * ctx = ggml_init(ip);

    // PLE form: the input arrives as [hc_dim, nt] and is viewed as [n_embd, hc, nt]; hc_mix: x is [n_embd, hc, nt]
    ggml_tensor * x_in = ple_form ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, nt)
                                  : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, nt);
    ggml_tensor * w1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hc_dim);         // the gamma as stored
    ggml_tensor * w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, hc);     // the same bytes as loaded now
    fill(x_in, 1 + (uint32_t) nt, 0.0f, 3.0f);
    fill(w1, 2, 1.0f, 0.25f); // folded (1 + w)
    memcpy(w2->data, w1->data, ggml_nbytes(w1));

    ggml_tensor * x3 = ple_form ? ggml_reshape_3d(ctx, x_in, n_embd, hc, nt) : x_in;

    // old
    ggml_tensor * o = ggml_rms_norm(ctx, x3, eps);
    o = ggml_reshape_2d(ctx, o, hc_dim, nt);
    o = ggml_mul(ctx, o, w1);
    if (ple_form) {
        o = ggml_reshape_3d(ctx, o, n_embd, hc, nt);
    }
    // new
    ggml_tensor * rn = ggml_rms_norm(ctx, x3, eps);
    ggml_tensor * n  = ggml_mul(ctx, rn, w2);
    if (!ple_form) {
        n = ggml_reshape_2d(ctx, n, hc_dim, nt);
    }

    ggml_cgraph * g_old = ggml_new_graph(ctx);
    ggml_build_forward_expand(g_old, o);
    ggml_cgraph * g_new = ggml_new_graph(ctx);
    ggml_build_forward_expand(g_new, n);

    // the new graph must have the MUL right after the RMS_NORM (that is what lets a backend fuse them)
    bool adjacent = false;
    for (int i = 0; i + 1 < ggml_graph_n_nodes(g_new); ++i) {
        if (ggml_graph_node(g_new, i) == rn && ggml_graph_node(g_new, i + 1)->op == GGML_OP_MUL &&
            ggml_graph_node(g_new, i + 1)->src[0] == rn) {
            adjacent = true;
        }
    }

    GGML_ASSERT(ggml_graph_compute_with_ctx(ctx, g_old, n_threads) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_graph_compute_with_ctx(ctx, g_new, n_threads) == GGML_STATUS_SUCCESS);

    GGML_ASSERT(ggml_nelements(o) == ggml_nelements(n) && ggml_is_contiguous(o) && ggml_is_contiguous(n));
    size_t nd = 0;
    for (int64_t i = 0; i < ggml_nelements(o); ++i) {
        nd += memcmp((const float *) o->data + i, (const float *) n->data + i, sizeof(float)) != 0;
    }
    const bool ok = adjacent && nd == 0;
    printf("A %-6s n_embd %4" PRId64 " hc %" PRId64 " nt %3" PRId64 " threads %d: adjacent %d, %zu of %" PRId64 " differ  %s\n",
           ple_form ? "ple" : "hc_mix", n_embd, hc, nt, n_threads, (int) adjacent, nd, ggml_nelements(o), ok ? "ok" : "FAIL");
    if (!ok) {
        n_fail++;
    }
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------------------------------------------
// B. model level

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// the qwen4exp MoE fixture of test-llama-archs (get_gguf_ctx), trimmed to the keys the arch reads
static gguf_context_ptr make_qwen4exp_meta() {
    const llm_arch arch = LLM_ARCH_QWEN4EXP;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());

    const uint32_t n_vocab = 128, n_ctx = 128, n_embd = 256, n_head = 2, n_ff = 384, n_layer = 2;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,     uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,    n_ctx/8);

    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,      uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK,   uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,   std::vector<uint32_t>(n_layer, 4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE, uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,     std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        n_ff);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);
    ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,         uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_COUNT,                      uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,                 uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,               uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,                uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,                1.0f);
    ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,                 uint32_t(1));

    ms.add_kv(LLM_KV_SSM_INNER_SIZE,     uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,    uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,     uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,    uint32_t(2));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,    "no_vocab");
    return ret;
}

struct node_rec {
    std::vector<const ggml_tensor *> nodes; // in evaluation order (all nodes, views included)
};

static bool record_nodes(struct ggml_tensor * t, bool ask, void * ud) {
    if (ask) {
        ((node_rec *) ud)->nodes.push_back(t);
    }
    return false; // never ask for data: the splits run whole, so the backend can still fuse
}

static bool silent_progress(float, void *) { return true; }

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static void model_case() {
    gguf_context_ptr meta = make_qwen4exp_meta();

    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;
    size_t seed = 28896;
    llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, &seed, mp));
    GGML_ASSERT(model && "model init failed");

    node_rec rec;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = 128;
    cp.n_batch           = 64;
    cp.n_ubatch          = 64;
    cp.n_threads         = 4;
    cp.n_threads_batch   = 4;
    cp.cb_eval           = record_nodes;
    cp.cb_eval_user_data = &rec;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(ctx && "context failed");

    std::mt19937 gen(3);
    std::uniform_int_distribution<llama_token> dis(0, 127);
    const int n_tok = 24;
    llama_batch b = llama_batch_init(n_tok, 0, 1);
    for (int i = 0; i < n_tok; ++i) {
        common_batch_add(b, dis(gen), i, { 0 }, true);
    }
    GGML_ASSERT(llama_decode(ctx.get(), b) == 0);
    llama_batch_free(b);

    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n_tok; ++i) {
        const uint8_t * p = (const uint8_t *) llama_get_logits_ith(ctx.get(), i);
        for (size_t k = 0; k < 128 * sizeof(float); ++k) {
            h = (h ^ p[k]) * 1099511628211ULL;
        }
    }

    // RMS_NORM -> MUL adjacency over the recorded order
    int n_rms = 0, n_adj = 0, n_split = 0;
    const auto & nd = rec.nodes;
    for (size_t i = 0; i < nd.size(); ++i) {
        if (nd[i]->op != GGML_OP_RMS_NORM) {
            continue;
        }
        n_rms++;
        if (i + 1 < nd.size() && nd[i + 1]->op == GGML_OP_MUL && (nd[i + 1]->src[0] == nd[i] || nd[i + 1]->src[1] == nd[i])) {
            n_adj++;
            continue;
        }
        // the old grouped-norm form: the MUL reads a RESHAPE of the RMS_NORM
        for (size_t j = i + 1; j < nd.size(); ++j) {
            const ggml_tensor * m = nd[j];
            if (m->op == GGML_OP_MUL && m->src[0]->op == GGML_OP_RESHAPE && m->src[0]->src[0] == nd[i]) {
                n_split++;
                break;
            }
        }
    }
    // 2 mixers per layer + the head mixer, at least
    const int n_want = 2 * 2 + 1;
    const bool ok = n_split == 0 && n_adj >= n_want;
    printf("B qwen4exp random model, %zu nodes: RMS_NORM %d, followed by their MUL %d (want >= %d), split by a RESHAPE %d  %s\n",
           nd.size(), n_rms, n_adj, n_want, n_split, ok ? "ok" : "FAIL");
    printf("B logits fnv1a %016" PRIx64 " (%d tokens x 128)\n", h, n_tok);
    if (!ok) {
        n_fail++;
    }
}

int main() {
    llama_log_set(quiet_log, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    for (bool ple : { false, true }) {
        for (int64_t nt : { 1, 6, 64 }) {
            op_case(2560, 4, nt, ple, 4);
        }
        op_case(256, 4, 7, ple, 1);
    }

    model_case();

    llama_backend_free();

    if (n_fail) {
        printf("test-qwen4exp-hc-norm: %d failures\n", n_fail);
        return 1;
    }
    printf("test-qwen4exp-hc-norm: OK\n");
    return 0;
}
