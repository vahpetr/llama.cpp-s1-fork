#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <tuple>
#include <map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

// vocabulary-pruned LM head (d2t): scatter the n_draft_vocab logits of `logits_k` into a full-vocabulary tensor
// filled with -INF, so every consumer keeps seeing n_vocab logits (same mechanism as eagle3 / dflash). A head that
// is not the pruned output.weight (e.g. a dedicated nextn shared head) is returned untouched.
static ggml_tensor * qwen4exp_scatter_pruned_logits(ggml_context * ctx0, const llama_model & model, const ggml_tensor * head_w, ggml_tensor * logits_k) {
    if (!model.d2t || head_w != model.output) {
        return logits_k;
    }
    const int64_t n_draft_vocab = logits_k->ne[0];
    const int64_t n_outputs_cur = logits_k->ne[1];
    const int64_t n_vocab_full  = (int64_t) model.vocab.n_tokens();
    GGML_ASSERT(model.d2t->type == GGML_TYPE_I64 && model.d2t->ne[0] == n_draft_vocab);
    ggml_tensor * full = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab_full, n_outputs_cur), -INFINITY);
    ggml_tensor * cur  = ggml_set_rows(ctx0, full,
            ggml_reshape_3d(ctx0, logits_k,  1,             n_draft_vocab, n_outputs_cur),
            ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
    return ggml_reshape_2d(ctx0, cur, n_vocab_full, n_outputs_cur);
}


// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    // NextN/MTP: an extra decoder block appended past the trunk. Read this first, since
    // n_layer() == n_layer_all - n_layer_nextn feeds every per-layer array below.
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

// [TAG_MERGE_ROWS] upstream #31028 (2280f0fd1) merges the projection pairs that read the same input in the converter:
// qkv | z (wqkv | wqkv_gate), beta | alpha (ssm_beta | ssm_alpha) and hc down | inject (attn and ffn), so one matmul
// computes both. Our GGUFs keep the separate tensors, so the pairs are merged AT LOAD instead (llama_model_loader::
// merge_rows: one [K, M_a + M_b] container, the file tensors become row views of it - no re-quantisation, no new GGUF,
// the same bytes in the same buffer type). Only a pair of equal type and width merges (mixed-type quant mixes keep the
// pair separate, see the load log line). A converter-merged GGUF (upstream names) is loaded as it is.
//
// The graph takes the merged matmul only while the backend computes it as a matrix-vector product - ubatches of at
// most LLAMA_QWEN4EXP_MERGE_MAX_N tokens (default 8 = ggml-vulkan's mul_mat_vec_max_cols, i.e. decode, the MTP draft
// and verify batches): a mat-vec row is the same dot product whatever the row count of the matrix (the pipeline and the
// per-row reduction do not depend on M on AMD: ggml_vk_get_dequantize_mul_mat_vec / ggml_vk_should_use_mmvq take M
// for NVIDIA / Intel workgroup heuristics only), so the merged outputs are bit-identical to the separate ones. Larger
// ubatches keep the two matmuls over the views - the very kernels of the separate tensors (the GEMM tile, split_k and
// the G1 tile table depend on M). LLAMA_QWEN4EXP_MERGE_MAX_N=<n> raises the bound (a negative control for the exactness
// tests: above 8 the merged GEMM is not guaranteed identical).
//
// Every consumer keeps its node: the merged output is cut into views (no cont, no reshape), and at n_tokens >= 2 those
// views are row-strided. The Vulkan fusions that read them (scale_silu, hc_combine, add_softplus_mul) and the plain
// SCALE accept a row-strided source for that (strided pipeline variants; the contiguous pipelines are unchanged).
//
// LLAMA_QWEN4EXP_MERGE: unset / empty / 1 = all three pairs, 0 = none (the load and every graph are those of the
// unmerged build), or a comma list of hc, ba, qkvz. Read once, at the first model load.
enum {
    QWEN4EXP_MERGE_HC   = 1,
    QWEN4EXP_MERGE_BA   = 2,
    QWEN4EXP_MERGE_QKVZ = 4,
    QWEN4EXP_MERGE_ALL  = QWEN4EXP_MERGE_HC | QWEN4EXP_MERGE_BA | QWEN4EXP_MERGE_QKVZ,
};

static int qwen4exp_merge_mask() {
    static const int mask = [] {
        const char * s = getenv("LLAMA_QWEN4EXP_MERGE");
        if (s == nullptr || *s == '\0' || strcmp(s, "1") == 0) {
            return (int) QWEN4EXP_MERGE_ALL;
        }
        if (strcmp(s, "0") == 0) {
            return 0;
        }
        int m = 0;
        std::string list = s;
        size_t pos = 0;
        while (pos <= list.size()) {
            const size_t end = std::min(list.find(',', pos), list.size());
            const std::string item = list.substr(pos, end - pos);
            if      (item == "hc")   { m |= QWEN4EXP_MERGE_HC; }
            else if (item == "ba")   { m |= QWEN4EXP_MERGE_BA; }
            else if (item == "qkvz") { m |= QWEN4EXP_MERGE_QKVZ; }
            else if (!item.empty())  { LLAMA_LOG_WARN("LLAMA_QWEN4EXP_MERGE: unknown item '%s' ignored\n", item.c_str()); }
            pos = end + 1;
        }
        return m;
    }();
    return mask;
}

static int64_t qwen4exp_merge_max_n() {
    static const int64_t n = [] {
        const char * s = getenv("LLAMA_QWEN4EXP_MERGE_MAX_N");
        return (s != nullptr && *s != '\0') ? (int64_t) atoll(s) : (int64_t) 8;
    }();
    return n;
}

// the merged matmul for this ubatch: a converter-merged file has no separate halves, so it always takes it. A load-time
// merge is skipped while a LoRA adapter is attached: build_lora_mm looks the adapter up by the weight tensor, and no
// adapter references the container, so the merged matmul would drop the LoRA of both halves
static bool qwen4exp_use_merged(const ggml_tensor * merged, const ggml_tensor * a, int64_t n_tokens, const llama_adapter_loras * loras) {
    if (merged == nullptr) {
        return false;
    }
    if (a == nullptr) {
        return true;
    }
    if (loras != nullptr && !loras->empty()) {
        return false;
    }
    return n_tokens <= qwen4exp_merge_max_n();
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // A detached draft head ships as its own GGUF: the nextn metadata and the trailing block
    // are there, the trunk is not. Probe a trunk tensor to spot that, and the MTP block to spot
    // the reverse -- nextn metadata kept but the head stripped -- exactly as deepseek2 does.
    // Either way the block index is unchanged: the head is always blk.<n_layer>.
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool mtp_only   = hparams.n_layer_nextn > 0 && ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr;
    const bool trunk_only = hparams.n_layer_nextn > 0 && ml.get_weight(mtp_probe.c_str())          == nullptr;

    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it.
    // a detached head carries no trunk mixer, and the checkpoint's own head mixer is unindexed,
    // so exports land it on these names instead of the per-block ones -- see below
    // [TAG_UP28896] the mixer / PLE norm gammas load as [n_embd, hc] (same bytes as the stored [hc_dim]) so the
    // grouped norm multiplies them without a graph reshape and RMS_NORM + MUL can fuse (upstream #28896)
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { n_embd, hc }, trunk_flags | TENSOR_ALLOW_RESHAPE);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    // A detached MTP head may ship a vocabulary-pruned LM head (FR-Spec / VocabTrim style): `output.weight` keeps
    // only n_draft_vocab rows and an I64 tensor `d2t` [n_draft_vocab] maps row -> full vocabulary id; the graph
    // scatters the reduced logits into a full-size -INF tensor (same mechanism as eagle3 / dflash), so samplers
    // and callers see n_vocab logits as before.
    int64_t n_draft_vocab = n_vocab;
    if (const ggml_tensor * d2t_meta = ml.get_tensor_meta("d2t")) {
        n_draft_vocab = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_draft_vocab }, 0);
        LLAMA_LOG_INFO("%s: QWEN4EXP head vocab pruned via d2t: %lld of %lld rows\n", __func__, (long long) n_draft_vocab, (long long) n_vocab);
    }
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_draft_vocab }, TENSOR_NOT_REQUIRED);
    if (d2t) {
        // the pruned head must be a real output.weight with exactly n_draft_vocab rows: a tied (token_embd) head or a
        // dedicated nextn shared head would be full-vocabulary and the scatter would assert
        if (!output || output->ne[1] != n_draft_vocab) {
            throw std::runtime_error("QWEN4EXP: d2t requires an output.weight with d2t->ne[0] rows (tied or missing head is not supported)");
        }
        if (ml.get_tensor_meta(("blk." + std::to_string(n_layer) + ".nextn.shared_head_head.weight").c_str())) {
            throw std::runtime_error("QWEN4EXP: d2t with a dedicated nextn.shared_head_head is not supported (prune output.weight instead)");
        }
    }
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target; n_rows is padded, so read it back.
    // the PLE layer is in the trunk, so a detached head describes no table and carries none
    if (hparams.ple_n_heads > 0 && !mtp_only) {
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        const auto & ple_w = ml.require_weight(ple_name.c_str());
        const int64_t ple_rows = ple_w.tensor->ne[1];

        // sanity check
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if ((int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h] > ple_rows) {
                throw std::runtime_error(format("PLE head %u range exceeds the %" PRId64 " table rows", h, ple_rows));
            }
        }
        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    // MTP tensors sit in the trailing blocks; skip them entirely unless a draft head was asked for
    int mtp_flags = trunk_only ? TENSOR_NOT_REQUIRED : 0;
    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    // [TAG_MERGE_ROWS] pairs tried / merged per kind (index = the QWEN4EXP_MERGE_* bit)
    std::array<int, QWEN4EXP_MERGE_ALL + 1> n_merge_try  = {};
    std::array<int, QWEN4EXP_MERGE_ALL + 1> n_merge_done = {};

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        // the MTP block is structurally a trunk block: is_recr()/is_ple() are both false past
        // the trunk, so it takes the full-attention + MoE path below with no special casing
        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp   ? hparams.n_ff_exp   : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // [TAG_MERGE_ROWS] matmuls that read the same input: the converter-merged [a | b] weight if the file has one,
        // else the separate tensors, merged at load when LLAMA_QWEN4EXP_MERGE selects the pair. `between` creates the
        // tensors that sit between a and b in the file order (kept, so the tensor list and its order are unchanged).
        auto create_merged = [&](llm_tensor t_merged, llm_tensor t_a, llm_tensor t_b, int64_t n_in, int64_t n_a, int64_t n_b,
                int bit, ggml_tensor *& merged, ggml_tensor *& a, ggml_tensor *& b, const std::function<void()> & between) {
            if (flags & TENSOR_SKIP) {
                const int skip = TENSOR_NOT_REQUIRED | TENSOR_SKIP;
                create_tensor(tn(t_merged, "weight", il), { n_in, n_a + n_b }, skip | TENSOR_SKIP_IF_VIRTUAL);
                create_tensor(tn(t_a,      "weight", il), { n_in, n_a },       skip);
                between();
                create_tensor(tn(t_b,      "weight", il), { n_in, n_b },       skip);
                return;
            }
            merged = create_tensor(tn(t_merged, "weight", il), { n_in, n_a + n_b }, flags | TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (merged) {
                between();
                return;
            }
            a = create_tensor(tn(t_a, "weight", il), { n_in, n_a }, flags);
            between();
            b = create_tensor(tn(t_b, "weight", il), { n_in, n_b }, flags);
            if (a && b && (qwen4exp_merge_mask() & bit)) {
                n_merge_try[bit]++;
                merged = ml.merge_rows(a, b, tn(t_merged, "weight", il).str());
                if (merged) {
                    n_merge_done[bit]++;
                }
            }
        };
        const auto nothing = [] {};

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        create_merged(LLM_TENSOR_HC_ATTN_DOWN_INJECT, LLM_TENSOR_HC_ATTN_DOWN, LLM_TENSOR_HC_ATTN_INJECT, hc_dim, hc_lr, hc,
                QWEN4EXP_MERGE_HC, layer.hc_attn_down_inject, layer.hc_attn_down, layer.hc_attn_inject, [&] {
            layer.hc_attn_up = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP, "weight", il), { hc_lr, hc_dim }, flags);
        });
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
        create_merged(LLM_TENSOR_HC_FFN_DOWN_INJECT, LLM_TENSOR_HC_FFN_DOWN, LLM_TENSOR_HC_FFN_INJECT, hc_dim, hc_lr, hc,
                QWEN4EXP_MERGE_HC, layer.hc_ffn_down_inject, layer.hc_ffn_down, layer.hc_ffn_inject, [&] {
            layer.hc_ffn_up = create_tensor(tn(LLM_TENSOR_HC_FFN_UP, "weight", il), { hc_lr, hc_dim }, flags);
        });

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            create_merged(LLM_TENSOR_ATTN_QKV_GATE, LLM_TENSOR_ATTN_QKV, LLM_TENSOR_ATTN_GATE, n_embd, key_dim * 2 + value_dim, value_dim,
                    QWEN4EXP_MERGE_QKVZ, layer.wqkv_z, layer.wqkv, layer.wqkv_gate, nothing);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            create_merged(LLM_TENSOR_SSM_BETA_ALPHA_CAT, LLM_TENSOR_SSM_BETA, LLM_TENSOR_SSM_ALPHA, n_embd, n_v_heads, n_v_heads,
                    QWEN4EXP_MERGE_BA, layer.ssm_beta_alpha_cat, layer.ssm_beta, layer.ssm_alpha, nothing);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { n_embd, hc }, flags | TENSOR_ALLOW_RESHAPE);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        // NextN/MTP head. enorm/hnorm gate the two inputs; eh_proj is the checkpoint's
        // fc_embedding and fc_hidden fused side by side, so one matmul over
        // concat(e, h) computes fc_embedding@e + fc_hidden@h.
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        // the head's own output mixer, mirroring the trunk's hc_head_*: it collapses the
        // hc streams and stands in for the output norm, of which qwen4exp has none.
        // it is unindexed in the checkpoint (mtp.hyper_connection_mixer.*), so a detached-head
        // export -- which has no trunk mixer to collide with -- writes it to the model-level
        // hc_head_* names instead. accept either spelling.
        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { n_embd, hc }, flags | TENSOR_NOT_REQUIRED | TENSOR_ALLOW_RESHAPE);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags | TENSOR_NOT_REQUIRED);

        // eh_proj is required whenever the head is loaded at all, so it tells the two apart
        if (layer.nextn.eh_proj && !layer.nextn.hc_head_norm) {
            if (!hc_head_norm) {
                throw std::runtime_error(format("qwen4exp MTP head %d has no output mixer: "
                        "neither blk.%d.nextn.hc_head_norm nor output_hc_norm is present", il, il));
            }
            layer.nextn.hc_head_norm = hc_head_norm;
            layer.nextn.hc_head_down = hc_head_down;
            layer.nextn.hc_head_up   = hc_head_up;
        }

        // qwen4exp sets mtp_use_dedicated_embeddings=false, so these are absent and the
        // head falls back to the trunk's embedding table and LM head
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
    }

    // [TAG_MERGE_ROWS] proof line (WARN, so it shows at the default -lv like the other proof lines): merged / tried per
    // kind (a pair is tried when both halves load and the knob selects it; it merges when the halves share type, width
    // and buffer type)
    {
        const char * env = getenv("LLAMA_QWEN4EXP_MERGE");
        LLAMA_LOG_WARN("%s: qwen4exp merge_rows: hc %d/%d, ba %d/%d, qkvz %d/%d (LLAMA_QWEN4EXP_MERGE=%s, max_n %lld)\n", __func__,
                n_merge_done[QWEN4EXP_MERGE_HC],   n_merge_try[QWEN4EXP_MERGE_HC],
                n_merge_done[QWEN4EXP_MERGE_BA],   n_merge_try[QWEN4EXP_MERGE_BA],
                n_merge_done[QWEN4EXP_MERGE_QKVZ], n_merge_try[QWEN4EXP_MERGE_QKVZ],
                env ? env : "unset", (long long) qwen4exp_merge_max_n());
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor *  w_down_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [n_embd, hc] gamma
    // the converter folded each gamma to (1 + w)
    // [TAG_UP28896] the MUL right after the RMS_NORM (no reshape between) so they fuse; same products
    ggml_tensor * xn = ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps), w_norm);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    // [TAG_MERGE_ROWS] one matmul for the hc_down rows and the hc_inject rows, cut into two views; the SCALE / SILU and
    // the combine chain read the views as they read the two matmul outputs (row-strided at nt >= 2)
    ggml_tensor * lo  = nullptr;
    ggml_tensor * inj = nullptr;
    if (inject && qwen4exp_use_merged(w_down_inject, w_down, nt, loras)) {
        const int64_t n_lo = w_down_inject->ne[1] - hc;
        ggml_tensor * li = build_lora_mm(w_down_inject, xn);
        cb(li, "hc_down_inject", il);
        lo  = ggml_view_2d(ctx0, li, n_lo, nt, li->nb[1], 0);
        inj = ggml_view_2d(ctx0, li, hc,   nt, li->nb[1], ggml_row_size(li->type, n_lo));
    } else {
        lo = build_lora_mm(w_down, xn);
    }
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // collapse the streams by their mean
    // [TAG_HC_MIX_NO_CONT]
    // no ggml_cont on the first stream: ggml_add already allocates a contiguous destination
    // (ggml_dup_tensor of src0), and every add consumer here takes a strided src0 -- the CPU
    // kernel only needs nb00 == sizeof(float), the Vulkan add carries nb00..nb03 in its push
    // constants, and ggml_vk_fuse_multi_add explicitly allows "different strides". The values are
    // unchanged: a f32->f32 cont is a bit copy, and the sum still runs as (((s0+s1)+s2)+s3) with
    // the same operands in the same order.
    ggml_tensor * mixed = ggml_view_2d(ctx0, gated, n_embd, nt,
            ggml_row_size(gated->type, n_embd) * hc, 0);
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx0, mixed, s);
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = inj ? inj : build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // Emit the injection projection before the chain below. Without this the depth-first
    // expansion of the add walks into it from the scale and drops its nodes in the middle of
    // the REPEAT/SCALE/SIGMOID/SCALE/MUL/ADD sequence, which stops the backend from matching
    // the chain as one fused op. Pure reordering: no node and no value changes.
    ggml_build_forward_expand(gf, inject);

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

// [TAG_PLE_HOIST]
// The PLE gather (ggml_get_rows over per_layer_token_embd) is the only node of the graph that the
// backend scheduler assigns to the CPU: the table is a CPU-resident weight (-ot
// per_layer_token_embd=CPU, buffer usage WEIGHTS -> cause "1.wgt0"), and ggml_vk_get_op_batch_size()
// returns 0 for GGML_OP_GET_ROWS, so ggml_backend_vk_device_offload_op() can never claim it. Built
// inside the layer loop it lands *between* two Vulkan runs, and the scheduler emits three splits:
//   Vulkan[inp_embd + blk.0] / CPU[ple gather] / Vulkan[blk.1 .. blk.47]
// The middle split has n_inputs == 0 (both srcs are already CPU-side), which trips the un-guarded
// drain at the top of ggml_backend_sched_compute_splits() (ggml/src/ggml-backend.cpp):
//   if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id)
//       ggml_backend_synchronize(sched->backends[prev_backend_id]);
// i.e. a full mid-graph GPU drain -- plus the ggml_vk_graph_cleanup() that
// ggml_backend_vk_synchronize() runs -- once per ubatch.
//
// Emitting the gather as the FIRST node of the graph makes it split 0, where prev_backend_id is
// still -1, so the drain cannot fire; the split list becomes CPU[ple gather] / Vulkan[everything],
// and the single remaining Vulkan split sees the whole graph's flops when it sizes its submit
// batches (ggml_backend_vk_graph_compute's flops_per_submit).
//
// Why the values cannot move: this changes the *order in which nodes are appended to the graph*,
// nothing else. Same ggml_get_rows, same table tensor, same index tensor, same reshape, same
// consumers (build_lora_mm(ple_key/ple_value, emb)); no operand, no operand order, no dtype, no
// epsilon and no accumulation order anywhere in the DAG is touched. ggml executes a DAG, so a
// topological reordering that keeps every edge is value-identical by construction. The row ids are
// unaffected too: llm_graph_input_ple::set_input() reads only the ubatch and the KV cells (which
// apply_ubatch() filled before any set_input runs) and writes only its own tensor, so it is
// independent of where in the input list it sits and of when the node that consumes it is built.
// LLAMA_PLE_HOIST=1 enables the hoist (default off, see below); unset/0 = the in-loop placement.
// [TAG_CONV_INPUT_SPLIT] the model side of the Vulkan concat_ssm_conv fusion, under the SAME knob
// GGML_VK_FUSE_CONCAT_SSM_CONV (default on; =0 = the v59 graph node for node: the tails view the concat and are
// expanded inside build_conv_state_at). Parsed as ggml-vulkan parses it (absent or empty = on, else atoi != 0).
static bool qwen4exp_conv_input_split() {
    static const bool enabled = [] {
        const char * s = getenv("GGML_VK_FUSE_CONCAT_SSM_CONV");
        return s == nullptr || *s == '\0' || atoi(s) != 0;
    }();
    return enabled;
}

static bool qwen4exp_ple_hoist_enabled() {
    static const bool enabled = [] {
        // default OFF since 2026-09-08: the v20 A/B measured the hoist at -3.5 % prefill (483.5 -> 466.5 t/s at 8k,
        // 404 -> 393 at 32k) and decode unchanged. With the gather in the middle of the graph the scheduler overlaps the
        // CPU get_rows with blk.0's GPU work; hoisted to the first node it runs while the GPU is idle, and that costs more
        // than the mid-graph drain it removes. LLAMA_PLE_HOIST=1 re-enables it for an A/B; the right design is the gather
        // as a graph input computed ahead of the previous ubatch (G25).
        const char * s = getenv("LLAMA_PLE_HOIST");
        return s != nullptr && atoi(s) != 0;
    }();

    return enabled;
}

// [TAG_QSA_BYPASS_EXHAUSTIVE] (M1.1, triage row T12)
//
// The indexer budget is width = min(n_kv, indexer_top_k + r - 1). When that clamps to n_kv the
// selection is provably exhaustive - ggml_top_k(expanded, n_kv) returns every column, so QSA is
// mathematically the dense attention, computed through ~40 extra nodes per layer, a second KV
// write, a top-k over the whole window and a rebuilt KQ mask. Below the budget the two arms must
// therefore be byte-identical, and on the v19-qsa prototype they were not (T1) - while short-context
// decode also fell 60.4 -> 50.4 t/s for a selection that selects everything.
//
// With this on, an exhaustive layer still writes its indexer K cache (so the history stays complete
// for the first ubatch that IS above budget) but calls the dense build_attn(). That makes the
// below-budget identity a property of the graph rather than of the arithmetic, and it removes the
// short-context regression.
//
// Default ON. LLAMA_QSA_BYPASS_EXHAUSTIVE=0 keeps the full QSA graph at every depth, which is what
// the M0/M1 divergence diagnosis has to run with: the bypass may ship, the diagnosis must be proven
// without it (gate M1).
static bool qwen4exp_qsa_bypass_exhaustive() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_BYPASS_EXHAUSTIVE");
        return s == nullptr || atoi(s) != 0;
    }();

    return enabled;
}

// [TAG_QSA_MIN_KV] (M5.0) LLAMA_QSA_MIN_KV=N > 0 widens [TAG_QSA_BYPASS_EXHAUSTIVE] into a depth threshold: a
// ubatch whose sequence depth (M5.0b: its deepest sequence's pos + 1, qwen4exp_qsa_depth; M5.0 compared the pool's
// indexer n_kv) is below N runs the dense build_attn(), exactly as an exhaustive one does - the indexer K write
// (cache_only) stays, so the history is complete for the first ubatch at depth >= N; no selection, no QSA mask, no
// pooled-store plan (the store's plan is stamp-based and stateless across bypassed ubatches: every block written
// meanwhile has no valid stamp and is pooled by the first plan above N, llama-memory-hybrid-idx.cpp qsa_store_plan).
// The decision is per ubatch over its depth including the ubatch's own tokens: a prefill ubatch that straddles N
// runs QSA whole. Independent of LLAMA_QSA_BYPASS_EXHAUSTIVE.
// Read once, logged at the first graph build (the reserve). Default 0 = off, today's graph node for node.
static int64_t qwen4exp_qsa_min_kv() {
    static const int64_t min_kv = [] {
        const char * s = getenv("LLAMA_QSA_MIN_KV");
        const long long v = s ? atoll(s) : 0;
        if (v > 0) {
            LLAMA_LOG_INFO("qwen4exp_qsa_min_kv: LLAMA_QSA_MIN_KV=%lld: QSA layers run dense attention while the ubatch's sequence depth < %lld\n", v, v);
        }
        return (int64_t) (v > 0 ? v : 0);
    }();

    return min_kv;
}

// [TAG_QSA_MIN_KV] (M5.0b) the depth LLAMA_QSA_MIN_KV compares is the SEQUENCE's, not the pool's: under
// --kv-unified the indexer n_kv is the pool's high-water cell, so one deep resident sequence would put every other
// sequence "above N". The depth of a ubatch is that of its deepest sequence, max over its tokens of pos + 1 = the
// sequence's n_past + its tokens in this ubatch (ubatch.pos[0..n_tokens) is the sequential row, the temporal one
// under M-RoPE). A ubatch that mixes a sequence above N with ones below stays QSA (one graph; each row selects
// among its own visible cells). A full (reserve) context has no real positions: it keeps the pool n_kv, so the
// reserve is today's worst case.
static int64_t qwen4exp_qsa_depth(const llama_memory_hybrid_idx_context * mctx_hyb, const llama_ubatch & ubatch, int64_t n_kv_idx) {
    if (mctx_hyb->is_full()) {
        return n_kv_idx;
    }

    llama_pos p_max = -1;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        p_max = std::max(p_max, ubatch.pos[i]);
    }

    return (int64_t) p_max + 1;
}

// the dense-path decision of a QSA layer: the exhaustive bypass ([TAG_QSA_BYPASS_EXHAUSTIVE], over the pool n_kv
// the budget width is taken over) or the ubatch's sequence depth below LLAMA_QSA_MIN_KV ([TAG_QSA_MIN_KV]);
// shared by build_layer_attn and llm_graph_input_qsa::can_reuse
// [TAG_QSA_WINDOW_MISALIGNED] or the sequence depth past the indexer cell window: positions beyond n_kv cells, which
// only a memory without its prefix has (the MTP draft after a legacy /slots restore or a refused draft rewind drops
// its sequence, and the catch-up writes positions >= the restored depth into cells 0..T). No block can cover such a
// cell (pb >= n_blocks); the census excludes it (routed to an id scored -inf, which also keeps another sequence of a
// unified pool safe) and asserts that no sparse query belongs to its sequence - this term: the dense attention over
// the partial window is what LLAMA_MTP_QSA=0 runs there. A complete history keeps one cell per position 0..depth-1, so
// n_kv >= depth (the reserve: qwen4exp_qsa_depth returns n_kv) and this term never changes a working graph.
// [TAG_MULTISLOT] (stage 5.1 item 3) LLAMA_QSA_HOMOGENEOUS=1 (the multi-slot qualification mode, read once): the pool
// high-water is not sequence coverage - under --kv-unified a deep resident B keeps n_kv above a partial A's depth, so
// `depth > n_kv` misses A. Then the dense term is also taken when ANY sequence of the ubatch lacks a complete indexer
// history (llama_memory_hybrid_idx_context::qsa_history_complete: min 0, no hole, up to its depth; [qsa-history]).
// Off (unset / not exactly 1): today's decision bit for bit - an M-RoPE image (repeated + skipped positions) would
// read as partial and route every image conversation of the hands (5 slots, mmproj) dense.
// [TAG_MULTISLOT] round 2 (astra r1 Major 2): the same coverage term also under llama_qsa_history_dense()
// (LLAMA_QSA_HISTORY_DENSE, default on until an M-RoPE image is decoded): a partial draft history (target-only restore,
// rewind hole) at depth <= n_kv ran sparse over a nonexhaustive selection - R85 says dense until a full recompute.
// LLAMA_QSA_HISTORY_DENSE=0 without LLAMA_QSA_HOMOGENEOUS=1: the a25b663c1 decision bit for bit.
static bool qwen4exp_qsa_homogeneous() {
    static const bool on = [] {
        const char * s = getenv("LLAMA_QSA_HOMOGENEOUS");
        return s != nullptr && strcmp(s, "1") == 0;
    }();

    return on;
}

// [TAG_MULTISLOT] (stage 5.1 item 4b) the routing-class guard of LLAMA_QSA_HOMOGENEOUS=1: the dense decision above is
// one per UBATCH (its deepest sequence), also under per-stream KV, so a ubatch mixing a sequence below
// LLAMA_QSA_MIN_KV with one at/above it would route the shallow one through the deep one's topology. The server keeps
// such sequences in separate decode calls (homogeneous routing groups); this guard proves it on the executed ubatches
// (build and reuse): per sequence its depth = max pos + 1 over ITS rows, class = depth >= N; all classes must agree.
// One sequence whose rows straddle N is valid (one depth); row min/max positions are deliberately not compared.
// Reserve (full) contexts carry no real positions and are skipped. Strict mode only (the knob); off = no code runs.
// Proof: [qsa-routing] checks=... multi_seq=... below=... above=... mixed=... every 1024 checks and at exit.
struct qwen4exp_qsa_routing_stats {
    std::atomic<uint64_t> checks{0};
    std::atomic<uint64_t> multi_seq{0};
    std::atomic<uint64_t> below{0};
    std::atomic<uint64_t> above{0};
    std::atomic<uint64_t> mixed{0};
    // LLAMA_QSA_ROUTING_COST=1: every validation pass (not deduplicated), its rows and wall time - a separate line
    std::atomic<uint64_t> cost_calls{0};
    std::atomic<uint64_t> cost_rows{0};
    std::atomic<uint64_t> cost_ns{0};

    void print(const char * when) const {
        // fprintf: also safe from the static destructor at exit, after the logger may be gone
        fprintf(stderr, "[qsa-routing] checks=%" PRIu64 " multi_seq=%" PRIu64 " below=%" PRIu64 " above=%" PRIu64 " mixed=%" PRIu64 " (%s)\n",
                checks.load(), multi_seq.load(), below.load(), above.load(), mixed.load(), when);
    }

    ~qwen4exp_qsa_routing_stats() {
        if (checks.load() > 0) {
            print("exit");
        }
        if (cost_calls.load() > 0) {
            fprintf(stderr, "[qsa-routing-cost] calls=%" PRIu64 " rows=%" PRIu64 " ns=%" PRIu64 " ns_per_call=%.0f ns_per_row=%.1f (exit)\n",
                    cost_calls.load(), cost_rows.load(), cost_ns.load(), (double) cost_ns.load() / cost_calls.load(),
                    (double) cost_ns.load() / std::max<uint64_t>(1, cost_rows.load()));
        }
    }
};

static qwen4exp_qsa_routing_stats & qwen4exp_qsa_routing() {
    static qwen4exp_qsa_routing_stats s;
    return s;
}

static void qwen4exp_qsa_routing_check(const llama_memory_hybrid_idx_context * mctx_hyb, const llama_ubatch & ubatch) {
    if (!qwen4exp_qsa_homogeneous() || mctx_hyb->is_full() || ubatch.n_tokens == 0) {
        return;
    }

    // (r3, astra r2 Major 1) every invocation is validated - no early return on a remembered ubatch: a key over a few
    // fields (pos pointer, n_tokens, first/last pos, first seq id) let a reused batch buffer whose INTERIOR sequence
    // changed depth skip the check. The full pass is O(n_tokens x n_seq_id) per call (cost: LLAMA_QSA_ROUTING_COST=1).
    // Only the proof counters are deduplicated ("one check per ubatch", not per QSA layer): by a hash over the
    // context, n_tokens and every (seq_id, pos) row - a collision can only under-count, never skip validation.
    static const bool cost = [] {
        const char * s = getenv("LLAMA_QSA_ROUTING_COST");
        return s != nullptr && strcmp(s, "1") == 0;
    }();
    const auto t0 = cost ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

    const int64_t n_min = qwen4exp_qsa_min_kv();

    uint64_t h = 1469598103934665603ull; // a word-wise FNV-style mix over (context, n_tokens, every (seq_id, pos) row)
    const auto mix = [&h](uint64_t v) {
        h = (h ^ v) * 1099511628211ull;
        h ^= h >> 29;
    };
    mix((uint64_t) (uintptr_t) mctx_hyb);
    mix(ubatch.n_tokens);

    std::vector<std::pair<llama_seq_id, llama_pos>> seqs; // (seq, depth)
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        mix((uint64_t) (uint32_t) ubatch.pos[i]);
        mix((uint64_t) (uint32_t) ubatch.n_seq_id[i]);
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id s = ubatch.seq_id[i][j];
            mix((uint64_t) (uint32_t) s);
            auto it = std::find_if(seqs.begin(), seqs.end(), [s](const auto & e) { return e.first == s; });
            if (it == seqs.end()) {
                seqs.emplace_back(s, ubatch.pos[i] + 1);
            } else {
                it->second = std::max(it->second, ubatch.pos[i] + 1);
            }
        }
    }

    bool any_below = false;
    bool any_above = false;
    for (const auto & e : seqs) {
        if ((int64_t) e.second >= n_min) {
            any_above = true;
        } else {
            any_below = true;
        }
    }

    auto & st = qwen4exp_qsa_routing();
    if (cost) {
        st.cost_calls++;
        st.cost_rows += ubatch.n_tokens;
        st.cost_ns   += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    if (any_below && any_above) {
        st.checks++;
        st.mixed++;
        std::string desc;
        for (const auto & e : seqs) {
            desc += format(" seq %d depth %d (%s)", (int) e.first, (int) e.second, (int64_t) e.second >= n_min ? "above" : "below");
        }
        st.print("MIXED");
        GGML_ABORT("[qsa-routing] mixed routing classes in one ubatch (N = %" PRId64 "):%s", n_min, desc.c_str());
    }
    thread_local uint64_t last = 0;
    if (h == last) {
        return; // the same ubatch again (another QSA layer of its build, a failed reuse): counted once
    }
    last = h;
    const uint64_t k = st.checks.fetch_add(1) + 1;
    if (seqs.size() > 1) {
        st.multi_seq++;
    }
    if (any_above) {
        st.above++;
    } else {
        st.below++;
    }
    if (k % 1024 == 0) {
        st.print("periodic");
    }
}

static bool qwen4exp_qsa_dense(const llama_memory_hybrid_idx_context * mctx_hyb, const llama_ubatch & ubatch,
                               int64_t n_kv_idx, int64_t width) {
    qwen4exp_qsa_routing_check(mctx_hyb, ubatch);

    const int64_t depth = qwen4exp_qsa_depth(mctx_hyb, ubatch, n_kv_idx);
    return depth > n_kv_idx
        || (qwen4exp_qsa_bypass_exhaustive() && width == n_kv_idx) || depth < qwen4exp_qsa_min_kv()
        || ((qwen4exp_qsa_homogeneous() || llama_qsa_history_dense()) && !mctx_hyb->qsa_history_complete(ubatch));
}

// [TAG_MTP_QSA] the selection budget (indexer_top_k) of layer il. TEST ONLY: LLAMA_MTP_QSA_TOP_K=N > 0 replaces it for
// the MTP block (il >= n_layer) - a large N forces the draft selection exhaustive (width == n_kv), a small one gives a
// tiny model a real ratio. The trunk's layers always take the GGUF value; unset = the GGUF value everywhere.
static uint32_t qwen4exp_qsa_top_k(const llama_hparams & hparams, int il) {
    static const long long mtp_top_k = [] {
        const char * s = getenv("LLAMA_MTP_QSA_TOP_K");
        const long long v = s ? atoll(s) : 0;
        if (v > 0) {
            LLAMA_LOG_WARN("qwen4exp_qsa_top_k: LLAMA_MTP_QSA_TOP_K=%lld: TEST ONLY - the MTP block's indexer_top_k\n", v);
        }
        return v > 0 ? v : 0;
    }();

    if (mtp_top_k > 0 && il >= (int) hparams.n_layer()) {
        return (uint32_t) mtp_top_k;
    }

    return hparams.indexer_top_k;
}

// [TAG_MTP_QSA] LLAMA_MTP_QSA_MAX_UBATCH=N (default 8): the MTP block runs the QSA selection only for ubatches of <= N
// tokens (the draft decode steps); a larger ubatch (catch-up / prompt-size process()) takes the dense build_attn() plus the
// indexer K write (cache_only), so the MTP context's load-time reserve (n_ubatch tokens) stays today's dense graph instead
// of the trunk's QSA prefill reservation. 0 = no limit (slice 1: every ubatch through QSA). The trunk is never affected.
static int64_t qwen4exp_mtp_qsa_max_ubatch() {
    static const int64_t v = [] {
        const char * s = getenv("LLAMA_MTP_QSA_MAX_UBATCH");
        const long long n = s ? atoll(s) : 8;
        if (s) {
            LLAMA_LOG_WARN("qwen4exp_mtp_qsa_max_ubatch: LLAMA_MTP_QSA_MAX_UBATCH=%s: MTP ubatches %s run dense\n", s,
                    n > 0 ? "above it" : "never (0 = no limit)");
        }
        return (int64_t) (n > 0 ? n : 0);
    }();
    return v;
}

// [TAG_QSA_FUSED_SELECT] GGML_VK_QSA_FUSED_SELECT=1 emits the indexer selection as ONE node
// (GGML_OP_QSA_TOP_K) instead of the six that the Vulkan topk_qsa pattern matches and elides.
//
// The six exist for the matcher, not for any backend: ggml_vk_topk_qsa already computes
// gather(score, cell_blk) + mask + top-k in a single shader. What they cost is ggml-alloc, which
// has to reserve every intermediate whether or not the backend reads it - at the deployed shape
// (n_kv 196608, n_tps 2050, 12 QSA layers) the permuted score, the gathered [n_tps x n_kv]
// expansion, its transpose back, the f32 mask cast and the sum are 384 MiB + 4 x 1537 MiB of
// compute-buffer surface per layer, all of it write-once/read-once. The op materializes the top-k
// ids and nothing else.
//
// It also makes M2.0's hazard impossible rather than vetoed. The fused dispatch reads three
// buffers at the TOP_K node that ggml-alloc believes died six nodes earlier, so
// ggml_vk_can_fuse_topk_qsa has to test each of them against the destination and un-fuse the layer
// when the allocator happened to reuse one. With one node all three are its own srcs: ggml-alloc
// cannot place a destination over a live source, so there is nothing to veto and nothing to lose
// the fusion over.
//
// The knob is named for the backend behaviour it selects, like the other QSA A/B knobs, even
// though it is read here on the graph side - the graph is where the node count is decided.
// Default OFF (QSA plan section 3.3: one A/B knob per change, off until its own gate passes); with it
// off this builder emits exactly the graph that shipped, node for node.
static bool qwen4exp_qsa_fused_select() {
    static const bool enabled = [] {
        const char * s = getenv("GGML_VK_QSA_FUSED_SELECT");
        return s != nullptr && atoi(s) != 0;
    }();

    return enabled;
}

// [TAG_QSA_CHUNK] (M4.2) LLAMA_QSA_CHUNK=C > 0 runs the query-dependent half of a QSA layer -
// indexer score, selection, scan mask and attention - one chunk of at most C queries at a time,
// inside the attention, and concatenates the chunk outputs along the token axis. The transformer
// ubatch is not split (that would regroup the recurrent layers' rollback).
//
// What it is for: at ub 2048 the per-layer query planes ([n_blocks] and [n_kv] x n_tps: the score
// GEMM, its relu, the head sums, the scan's fill and mask add, the fused select's scratch) are the
// compute-buffer watermark. Built chunk after chunk, ggml-alloc frees chunk c's planes before chunk
// c+1's are placed, so the watermark holds one chunk's planes plus the running output.
//
// Shape: n_chunks = ceil(n_tps / C) chunks, balanced to differ by at most one query (2050 at C 256
// -> 7 x 228 + 2 x 227), so no chunk is small enough to reach the backends' small-N kernels
// (mul_mat_vec at N <= 8, FA split_k). The scan/gather branch is decided ONCE for the ubatch, with
// the unchunked rule, and only a scan ubatch is chunked; a gather ubatch (n_tps <= n_kv/(4*width))
// is small by construction and keeps today's graph. n_tps <= C likewise keeps today's graph, node
// for node - so decode and every short batch are untouched. The chunks stay scan-only ([TAG_QSA_PATH]).
//
// Values: 0 = off, else a chunk of at least 64 queries (smaller values are raised to 64: every chunk
// costs ~40 graph nodes per QSA layer, and below that the chunks reach the small-N kernels anyway).
// Default 0 = off (QSA plan section 3.3: one A/B knob per change, off until its own gate passes).
static int64_t qwen4exp_qsa_chunk() {
    static const int64_t chunk = [] {
        const char * s = getenv("LLAMA_QSA_CHUNK");
        const long long v = s ? atoll(s) : 0;
        if (v > 0 && v < 64) {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_CHUNK=%lld raised to 64\n", __func__, v);
        }
        const int64_t c = (int64_t) (v <= 0 ? 0 : std::max<long long>(v, 64));
        // [TAG_QSA_RELU_INPLACE] (R80.1 v49, B5) the proof line, printed once at the first QSA graph (the reserve at load)
        if (c > 0) {
            LLAMA_LOG_WARN("%s: qsa_chunk: on (C=%lld, LLAMA_QSA_CHUNK=%s): scan ubatches with more than C queries run the "
                           "query-dependent half of each QSA layer chunk by chunk\n", __func__, (long long) c, s);
        } else {
            LLAMA_LOG_WARN("%s: qsa_chunk: off (LLAMA_QSA_CHUNK %s)\n", __func__, s ? s : "unset");
        }
        return c;
    }();

    return chunk;
}

// [TAG_QSA_RELU_INPLACE] (R80.1 v49, QSA depth plan B5 step 2) LLAMA_QSA_RELU_INPLACE=1 (default 0): the indexer
// score's ReLU (build_qsa_select) runs in place on the score GEMM's output (ggml_relu_inplace) instead of into a new
// [n_blocks, n_idx_h, n_tps, n_stream] f32 plane. Exact: the same unary kernel computes max(x, 0) element by element
// from the same input, only into the input's own memory; nothing else reads the pre-ReLU score (the head sums read
// the ReLU output). What it saves is the allocator's second plane where it does not already reuse the GEMM output.
// Off = today's graph node for node.
static bool qwen4exp_qsa_relu_inplace() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_RELU_INPLACE");
        const bool on = s != nullptr && atoi(s) != 0;
        LLAMA_LOG_WARN("%s: qsa_relu_inplace: %s (LLAMA_QSA_RELU_INPLACE %s)\n", __func__, on ? "on" : "off",
                       s ? s : "unset");
        return on;
    }();

    return enabled;
}

// [TAG_QSA_SCORE_EPILOGUE] (Program v8.1 A1, contrib 0235) GGML_VK_QSA_SCORE_EPILOGUE (default 1 = on; 0 = off): the
// indexer score of build_qsa_select is ONE node, GGML_OP_QSA_SCORE - the score GEMM (MUL_MAT of the pooled keys or
// MUL_MAT_ROWS of the store) with the ReLU, the ordered head sum ((s0 + s1) + s2) + s3 and the per-block bias computed in
// its store epilogue - instead of MUL_MAT[_ROWS] + RELU + CONT + ADD/MULTI_ADD + bias ADD. EXACT: the product is the
// same kernel / tile / K order, and the epilogue repeats the chain's arithmetic element by element (see ggml.h), so the
// summed plane and the selection's input are the same bytes; what goes is the [n_blocks, 4 x queries] per-head plane
// (written by the GEMM, rewritten by the ReLU, read by the CONT and the adds) and four dispatches per chunk.
// Used only where the backend of the first device takes the node (Vulkan: the coopmat1 matmul path of the product;
// decode / short-verify widths that run the mat-vec, and any other backend that refuses it, keep the chain).
// LLAMA_QSA_DUMP_SCORE=1 (the oracle's score capture) keeps the bias as its own ADD after the node, so the captured
// "indexer_score-<il>" is still the pre-bias summed plane - the same epilogue minus its last add ([vk-fusion-count]
// qsa_score_epilogue only). The production path (capture off) folds the bias: qsa_score_epilogue + qsa_score_epilogue_bias,
// node "indexer_score_biased-<il>". The capture pass and the ids pass thus differ by that one f32 add of {+0, 1e9, -inf}.
// The knob is named for the backend behaviour it selects, like the other QSA A/B knobs, but read on the graph side;
// off = exactly the v59 graph, node for node (the new pipelines are built and never dispatched).
static bool qwen4exp_qsa_score_epilogue() {
    static const bool enabled = [] {
        const char * s = getenv("GGML_VK_QSA_SCORE_EPILOGUE");
        const bool on = s == nullptr || atoi(s) != 0;
        LLAMA_LOG_WARN("%s: qsa_score_epilogue: %s (GGML_VK_QSA_SCORE_EPILOGUE %s)\n", __func__, on ? "on" : "off",
                       s ? s : "unset");
        return on;
    }();

    return enabled;
}

// [TAG_QSA_SCORE_EPILOGUE] the oracle's score capture (llama-context.cpp, LLAMA_QSA_DUMP_SCORE) reads "indexer_score-<il>"
static bool qwen4exp_qsa_dump_score() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_DUMP_SCORE");
        return s != nullptr && atoi(s) != 0;
    }();

    return enabled;
}

// [TAG_QSA_CHUNK_PAD] (M4.3) LLAMA_QSA_CHUNK_PAD=1: [TAG_QSA_CHUNK] cuts the ubatch into full chunks of C queries
// and pads the LAST one to C (its padding rows repeat the ubatch's last query), so every chunk's score GEMM and
// selection run the same column count - gate M4 compares that against the balanced split (default 0, today's
// graph node for node). The padding queries' ids are cut right after the selection (a view of the chunk's real
// columns, see build_qsa_chunked): they never reach the scan mask, the attention, the output or the
// LLAMA_QSA_DUMP_IDS hash (the cut view carries the "indexer_top_k-<il>" name). There is no gather in the chunk
// loop, and the pooled-key store's dirty list is planned on the host from the cells, before any query is scored.
// Only a chunked ubatch (n_tps > C) is affected.
static bool qwen4exp_qsa_chunk_pad() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_CHUNK_PAD");
        const bool on = s != nullptr && atoi(s) != 0;
        if (on) {
            LLAMA_LOG_INFO("qwen4exp_qsa_chunk_pad: LLAMA_QSA_CHUNK_PAD=1: the last query chunk is padded to the chunk size\n");
        }
        return on;
    }();

    return enabled;
}

// [TAG_QSA_GATHER_Q8] (M4.1) LLAMA_QSA_GATHER_Q8=1 makes the QSA gathers type-preserving
// (ggml_get_rows_type, fork-only): build_qsa_gather gathers k_sel/v_sel in the cache's own type -
// q8_0 on the deployed profile, f16 on an f16 cache - and hands them to flash attention as they
// are, instead of expanding every selected row to f32 (ggml_get_rows) for build_attn_mha to cast
// back to f16. The attention mask gathers of the gather branch and of [TAG_QSA_CHUNK]'s chunks
// likewise stay f16 instead of going f16 -> f32 -> f16 through a cast.
//
// Exactness: every gathered row is the source row byte for byte (a q8_0 row is whole blocks, an
// f16 row whole halves), so the selection, the mask values and the K/V bytes flash attention sees
// are what the dense path would see; the kernel flash attention then runs is the q8_0 K/V kernel
// the dense path runs (same rounding class) instead of the f16 kernel over rounded values. On an
// f16 cache the knob changes no value at all (f16 -> f32 -> f16 is exact).
//
// Any other cache type keeps today's f32 gather. Default OFF (QSA plan section 3.3: one A/B knob
// per change, off until its own gate passes); with it off every builder emits today's graph, node
// for node.
static bool qwen4exp_qsa_gather_q8() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_GATHER_Q8");
        return s != nullptr && atoi(s) != 0;
    }();

    return enabled;
}

// the cache types [TAG_QSA_GATHER_Q8] gathers as-is
static bool qwen4exp_qsa_gather_keeps(ggml_type type) {
    return qwen4exp_qsa_gather_q8() && (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_F16);
}

// [TAG_QSA_PATH] (M4.3) LLAMA_QSA_PATH=auto|gather|scan forces the attention branch of every NON-chunked
// QSA layer, for gate M4 to measure gather against scan on a grid; `auto` (the default, and any unset or
// unknown value) is the rule below, byte for byte. Read once, logged at the first graph build (the reserve).
//
// gather is supported for 1 <= n_tps*n_stream <= QWEN4EXP_QSA_GATHER_MAX_TPS query windows in all (the gather
// sizes K/V by width*n_q, n_q = n_tps*n_stream, whether the KV cache is unified or not), at any n_kv (the
// gathered windows are width cells per query, whatever n_kv is) and needs flash attention (build_qsa_gather
// hands flash attention one query per window). The hard limits behind the range: the n_tps*n_stream windows
// ride the stream axis of flash attention, which the Vulkan FA dispatches as workgroups_z = neq3
// (ggml-vulkan.cpp ggml_vk_flash_attn, `workgroups_z = (uint32_t)neq3`) under the maxComputeWorkGroupCount[2]
// assert of ggml_vk_dispatch_pipeline; and the gathered K and V are width*n_tps*n_stream whole cells each, a
// compute-buffer surface linear in n_q (width 2051, 2 x 256 q8_0 cells: 2 x 68.1 MiB at 64 windows, 2 x 136.2
// MiB at 128). 128 covers the gate's grid (the 7-25-row band, -ub <= 64) and the fused-op probe graph that
// llama_context::resolve_fused_ops builds at 16 tokens per sequence whatever -ub is (80 at --parallel 5, 128 at 8,
// unified or one stream per sequence alike; graph_reserve rounds n_tokens up to a multiple of n_seqs, so -ub 128
// at --parallel 5 is 130 and asserts);
// anything above it asserts rather than silently building a multi-GiB gather (a -ub 2048 prefill would be 2 x 2.1
// GiB). scan has no range limit (it is what auto builds above the rule).
//
// The [TAG_QSA_CHUNK] loop stays scan-only: it exists to bound the scan's [n_kv x n_tps] planes of prefill-sized
// ubatches, which are far outside gather's range, and its decision is `n_tps > chunk && !qwen4exp_qsa_use_gather`,
// so a forced gather ubatch is never chunked and a forced scan ubatch is chunked exactly as an auto one is.
enum qwen4exp_qsa_path_mode {
    QWEN4EXP_QSA_PATH_AUTO,
    QWEN4EXP_QSA_PATH_GATHER,
    QWEN4EXP_QSA_PATH_SCAN,
};

static constexpr int64_t QWEN4EXP_QSA_GATHER_MAX_TPS = 128;

static qwen4exp_qsa_path_mode qwen4exp_qsa_path() {
    static const qwen4exp_qsa_path_mode mode = [] {
        const char * s = getenv("LLAMA_QSA_PATH");
        if (s == nullptr || *s == '\0') {
            return QWEN4EXP_QSA_PATH_AUTO;
        }
        qwen4exp_qsa_path_mode m = QWEN4EXP_QSA_PATH_AUTO;
        if (strcmp(s, "gather") == 0) {
            m = QWEN4EXP_QSA_PATH_GATHER;
        } else if (strcmp(s, "scan") == 0) {
            m = QWEN4EXP_QSA_PATH_SCAN;
        } else if (strcmp(s, "auto") != 0) {
            LLAMA_LOG_WARN("qwen4exp_qsa_path: LLAMA_QSA_PATH=%s unknown, using auto\n", s);
        }
        if (m == QWEN4EXP_QSA_PATH_GATHER) {
            LLAMA_LOG_INFO("qwen4exp_qsa_path: LLAMA_QSA_PATH=%s: the non-chunked QSA attention branch is forced to gather"
                    " (flash attention, n_tps*n_stream <= QWEN4EXP_QSA_GATHER_MAX_TPS = %lld)\n", s,
                    (long long) QWEN4EXP_QSA_GATHER_MAX_TPS);
        } else {
            LLAMA_LOG_INFO("qwen4exp_qsa_path: LLAMA_QSA_PATH=%s: the non-chunked QSA attention branch is %s\n", s,
                    m == QWEN4EXP_QSA_PATH_SCAN ? "forced to scan" : "the 4*n_tps*width < n_kv rule");
        }
        return m;
    }();

    return mode;
}

// [TAG_QSA_GATHER_MAX_WINDOWS] (M5.0 item 5) LLAMA_QSA_GATHER_MAX_WINDOWS=N > 0 re-prices the auto rule: gather iff
// flash attention, the ubatch is above the budget (width < n_kv) and n_tps*n_stream <= N query windows - whatever
// n_kv is (gate M4's routing grid, 2026-09-24: gather beats scan up to 64 windows at 107k). N > 128 is clamped to
// QWEN4EXP_QSA_GATHER_MAX_TPS with a warning. LLAMA_QSA_PATH=gather|scan still overrides it. Read once, logged at
// the first QSA graph build. Default 0 = `4*n_tps*width < n_kv`, today's rule.
static int64_t qwen4exp_qsa_gather_max_windows();

// build_attn_qsa's branch rule, shared with the [TAG_QSA_CHUNK] decision so the two cannot drift.
// flash attention is what keeps the values as rows, so the gather needs it.
static bool qwen4exp_qsa_use_gather(bool flash_attn, int64_t n_tps, int64_t n_stream, int64_t width, int64_t n_kv) {
    switch (qwen4exp_qsa_path()) {
        case QWEN4EXP_QSA_PATH_GATHER:
            GGML_ASSERT(flash_attn && "LLAMA_QSA_PATH=gather needs flash attention");
            if (n_tps < 1 || n_tps*n_stream > QWEN4EXP_QSA_GATHER_MAX_TPS) {
                GGML_ABORT("LLAMA_QSA_PATH=gather supports 1 <= n_tps*n_stream <= %lld query windows, got %lld x %lld"
                        " (the reserve rounds -ub up to a multiple of --parallel)",
                        (long long) QWEN4EXP_QSA_GATHER_MAX_TPS, (long long) n_tps, (long long) n_stream);
            }
            return true;
        case QWEN4EXP_QSA_PATH_SCAN:
            return false;
        case QWEN4EXP_QSA_PATH_AUTO:
            break;
    }
    if (const int64_t max_windows = qwen4exp_qsa_gather_max_windows(); max_windows > 0) {
        return flash_attn && width < n_kv && n_tps*n_stream <= max_windows;
    }
    return flash_attn && 4*n_tps*width < n_kv;
}

static int64_t qwen4exp_qsa_gather_max_windows() {
    static const int64_t max_windows = [] {
        const char * s = getenv("LLAMA_QSA_GATHER_MAX_WINDOWS");
        long long v = s ? atoll(s) : 0;
        if (v > QWEN4EXP_QSA_GATHER_MAX_TPS) {
            LLAMA_LOG_WARN("qwen4exp_qsa_gather_max_windows: LLAMA_QSA_GATHER_MAX_WINDOWS=%lld clamped to QWEN4EXP_QSA_GATHER_MAX_TPS = %lld\n",
                    v, (long long) QWEN4EXP_QSA_GATHER_MAX_TPS);
            v = QWEN4EXP_QSA_GATHER_MAX_TPS;
        }
        if (v > 0) {
            LLAMA_LOG_INFO("qwen4exp_qsa_gather_max_windows: LLAMA_QSA_GATHER_MAX_WINDOWS=%lld: auto routes a QSA ubatch above the budget"
                    " to gather iff n_tps*n_stream <= %lld (flash attention)\n", v, v);
        }
        return (int64_t) (v > 0 ? v : 0);
    }();

    return max_windows;
}

// [TAG_QSA_INDEXED_FA] (M6) LLAMA_QSA_PREFILL=scan|indexed: the SCAN branch at n_tps > 8 is the masked scan (default, node
// for node as before) or FA over each query's own QSA_TOP_K ids (src[5]: no fill/set_rows/add planes, no zeros, no K/V
// staging). Read once, logged at the first graph build (the reserve); unknown values fall back to scan with a warning.
// The decision is made once per ubatch from its token count (spec §3.4): every chunk of a chunked ubatch, the
// [TAG_QSA_CHUNK_PAD] tail included, takes the same attention. It acts only where the scan is taken today: gather
// ([TAG_QSA_GATHER_MAX_WINDOWS], LLAMA_QSA_PATH=gather) and dense ([TAG_QSA_MIN_KV], exhaustive bypass) are untouched.
static bool qwen4exp_qsa_prefill_indexed() {
    static const bool indexed = [] {
        const char * s = getenv("LLAMA_QSA_PREFILL");
        if (s == nullptr || *s == '\0') { return false; }
        const bool on = strcmp(s, "indexed") == 0;
        if (!on && strcmp(s, "scan") != 0) {
            LLAMA_LOG_WARN("qwen4exp_qsa_prefill: LLAMA_QSA_PREFILL=%s unknown, using scan\n", s);
        }
        LLAMA_LOG_INFO("qwen4exp_qsa_prefill: LLAMA_QSA_PREFILL=%s: the scan branch at n_tps > 8 is %s\n", s,
                on ? "indexed flash attention (src[5] = QSA_TOP_K ids)" : "the masked scan");
        return on;
    }();
    return indexed;
}

// [TAG_QSA_INDEXED_SORT] (frozen-M6 isolation) LLAMA_QSA_INDEXED_SORT=1 (default 0 = node for
// node as before): the indexed flash attention reads each row's QSA_TOP_K ids SORTED ASCENDING, i.e. walks the selected
// cells in KV order as the scan does, so an indexed-vs-scan difference that the order explains disappears with it (the
// radix select emits winners, then ties, each run ascending). The ids are < 2^24, so cast to F32, argsort ascending,
// get_rows, cast to I32 is exact and keeps the set. Acts only where the indexed branch is taken (LLAMA_QSA_PREFILL=indexed).
// Strict: unset / "" / "0" off, "1" on, anything else WARN + off. Read with LLAMA_QSA_PREFILL at the first QSA layer (the
// reserve); logged once and only when on: `qsa_indexed_sort: on (LLAMA_QSA_INDEXED_SORT=1)`.
static bool qwen4exp_qsa_indexed_sort() {
    static const bool on = [] {
        const char * s = getenv("LLAMA_QSA_INDEXED_SORT");
        if (s == nullptr || *s == '\0' || strcmp(s, "0") == 0) { return false; }
        if (strcmp(s, "1") != 0) {
            LLAMA_LOG_WARN("qwen4exp_qsa_indexed_sort: LLAMA_QSA_INDEXED_SORT=%s is neither 0 nor 1: off\n", s);
            return false;
        }
        LLAMA_LOG_INFO("qsa_indexed_sort: on (LLAMA_QSA_INDEXED_SORT=1): the indexed flash attention reads each row's ids sorted "
                "ascending%s\n", qwen4exp_qsa_prefill_indexed() ? "" : " - but LLAMA_QSA_PREFILL is not indexed, so no graph uses it");
        return true;
    }();
    return on;
}

// [TAG_QSA_INDEXED_SORT] each row of ids [W, n, 1, ns] I32 sorted ascending, as a fresh contiguous I32 tensor
static ggml_tensor * qwen4exp_qsa_sort_ids(ggml_context * ctx, ggml_tensor * ids) {
    const int64_t W = ids->ne[0];
    const int64_t R = ids->ne[1]*ids->ne[2]*ids->ne[3];

    ggml_tensor * f    = ggml_reshape_2d(ctx, ggml_cast(ctx, ids, GGML_TYPE_F32), W, R);    // exact: ids < 2^24
    ggml_tensor * perm = ggml_argsort(ctx, f, GGML_SORT_ORDER_ASC);                          // [W, R] I32
    ggml_tensor * val  = ggml_get_rows(ctx, ggml_reshape_3d(ctx, f, 1, W, R), perm);         // [1, W, R] F32
    return ggml_cast(ctx, ggml_reshape_4d(ctx, val, W, ids->ne[1], ids->ne[2], ids->ne[3]), GGML_TYPE_I32);
}

// [TAG_QSA_VIS] (A4 T3/T4) LLAMA_QSA_VIS=1 (default 0 = node for node as before): in a ubatch of the KQ_VIS metadata class
// (LLAMA_KQ_VIS=1 and >= LLAMA_KQ_VIS_MIN_TOKENS tokens, where the dense layers already run FA over the metadata) that takes
// the QSA path, every QSA consumer of the KQ mask reads the 16 B/cell + 16 B/token metadata instead of the [n_kv x n_tps] f16
// plane: the selection (ggml_qsa_top_k_vis, the fused select), the indexed flash attention (ggml_flash_attn_ext_set_vis on the
// src[5] node) and the gather's window mask (ggml_kq_mask_ids). The KQ_MASK node then has no consumer and is never computed
// or allocated (-4 096 B/cell at ub 2048, plus the chunk's [n_kv x C] f16 rows, -512 B/cell at C 256). Every value read is
// bit-identical to the plane's, so the ids, the attention and the outputs equal knob off (EXACT). The decode class (< 64
// tokens) keeps its small device plane. Needs LLAMA_QSA_PREFILL=indexed (the scan reads "selected AND visible" from a
// plane) and GGML_VK_QSA_FUSED_SELECT=1 (the unfused chain adds the cast plane) - else WARN once, off; a graph without the
// metadata inputs (LLAMA_KQ_VIS=0, n_seq_max > 32) or a backend without the indexed+vis FA keeps the plane. Strict parse:
// unset / "" / "0" off, "1" on, anything else WARN + off. Read at the first QSA layer (the reserve); proof line (WARN, so it
// shows at the server's default verbosity) `qsa_vis: on (LLAMA_QSA_VIS=1) ...`.
static bool qwen4exp_qsa_vis() {
    static const bool on = [] {
        const char * s = getenv("LLAMA_QSA_VIS");
        if (s == nullptr || *s == '\0' || strcmp(s, "0") == 0) { return false; }
        if (strcmp(s, "1") != 0) {
            LLAMA_LOG_WARN("qwen4exp_qsa_vis: LLAMA_QSA_VIS=%s is neither 0 nor 1: off\n", s);
            return false;
        }
        if (!qwen4exp_qsa_prefill_indexed()) {
            LLAMA_LOG_WARN("qwen4exp_qsa_vis: LLAMA_QSA_VIS=1 needs LLAMA_QSA_PREFILL=indexed (the scan reads a plane): off\n");
            return false;
        }
        if (!qwen4exp_qsa_fused_select()) {
            LLAMA_LOG_WARN("qwen4exp_qsa_vis: LLAMA_QSA_VIS=1 needs GGML_VK_QSA_FUSED_SELECT=1 (the unfused select casts a plane): off\n");
            return false;
        }
        LLAMA_LOG_WARN("qsa_vis: on (LLAMA_QSA_VIS=1): in the KQ_VIS metadata class the QSA selection, the indexed flash "
                "attention and the gather mask read the cell/token metadata - no [n_kv x n_tps] KQ mask plane\n");
        return true;
    }();
    return on;
}

// [TAG_QSA_WHOLE_BLOCKS] + [TAG_QSA_INDEXED_FA] the block-level selection pads a row with repeats of its last id
// (ggml_qsa_top_k_blocks); the indexed flash attention would count such a cell twice (the scan's set_rows writes the same
// 0 twice, a gathered window masks the repeat itself). So every slot j > 0 whose id equals id[j - 1] becomes -1, a dead
// column (ggml.h: never read, never counted). Selected ids are distinct, so only padding changes; a cell-form row is
// returned with the same values. ids < 2^24: the f32 round trip is exact. keep = step(|id[j] - id[j-1]|), slot 0 kept;
// id' = (id + 1) * keep - 1.
static ggml_tensor * qwen4exp_qsa_mask_repeats(ggml_context * ctx, ggml_tensor * ids) {
    const int64_t W = ids->ne[0];
    const int64_t R = ids->ne[1]*ids->ne[2]*ids->ne[3];

    ggml_tensor * f    = ggml_reshape_2d(ctx, ggml_cast(ctx, ids, GGML_TYPE_F32), W, R);
    ggml_tensor * cur  = ggml_view_2d(ctx, f, W - 1, R, f->nb[1], f->nb[0]);
    ggml_tensor * prv  = ggml_view_2d(ctx, f, W - 1, R, f->nb[1], 0);
    ggml_tensor * keep = ggml_step(ctx, ggml_abs(ctx, ggml_sub(ctx, cur, prv)));                                  // [W - 1, R]
    ggml_tensor * one  = ggml_scale_bias(ctx, ggml_cont(ctx, ggml_view_2d(ctx, f, 1, R, f->nb[1], 0)), 0.0f, 1.0f); // [1, R]
    keep = ggml_concat(ctx, one, keep, 0);                                                                         // [W, R]
    ggml_tensor * out  = ggml_scale_bias(ctx, ggml_mul(ctx, ggml_scale_bias(ctx, f, 1.0f, 1.0f), keep), 1.0f, -1.0f);
    return ggml_cast(ctx, ggml_reshape_4d(ctx, out, W, ids->ne[1], ids->ne[2], ids->ne[3]), GGML_TYPE_I32);
}

// the knob refuses rather than let the scheduler put the node on the CPU (spec §6): ask the first backend about a small one.
// No scheduler (an unknown backend) is a refusal. The answer is cached per (device, K/V types, heads, head dims), so the
// stand-in node is built once, at the reserve, not per QSA layer per graph.
// [TAG_QSA_VIS] vis: the same node over the visibility metadata (ggml_flash_attn_ext_vis, causal) instead of an f16 mask.
static bool qwen4exp_qsa_indexed_supported(ggml_backend_sched_t sched, ggml_type type_k, ggml_type type_v,
        int64_t d_k, int64_t d_v, int64_t n_head, int64_t n_head_kv, bool vis = false) {
    if (sched == nullptr) { return false; }
    ggml_backend_t backend = ggml_backend_sched_get_backend(sched, 0);
    using key_t = std::tuple<ggml_backend_dev_t, int, int, int64_t, int64_t, int64_t, int64_t, bool>;
    static std::mutex mtx;
    static std::map<key_t, bool> answers;
    const key_t key { ggml_backend_get_device(backend), (int) type_k, (int) type_v, d_k, d_v, n_head, n_head_kv, vis };
    std::lock_guard<std::mutex> lock(mtx);
    if (auto it = answers.find(key); it != answers.end()) { return it->second; }
    ggml_init_params ip = { /*.mem_size =*/ 24*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d_k, 9, n_head, 1);
    ggml_tensor * k_t = ggml_new_tensor_4d(ctx, type_k, d_k, 256, n_head_kv, 1);
    ggml_tensor * v_t = ggml_new_tensor_4d(ctx, type_v, d_v, 256, n_head_kv, 1);
    ggml_tensor * fa = vis
        ? ggml_flash_attn_ext_vis(ctx, q_t, k_t, v_t, ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 256, 4, 1),
                ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, 9, 1), 1.0f, 0.0f, 0.0f, true, false, 0, 0)
        : ggml_flash_attn_ext(ctx, q_t, k_t, v_t, ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, 9, 1, 1), 1.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_flash_attn_ext_set_kv_ids(fa, ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 64, 9, 1, 1));
    bool ok = ggml_backend_supports_op(backend, fa);
    if (vis && ok) {
        // [TAG_QSA_VIS] the other two metadata consumers: the selection (causal, cell form) and the gather's window mask -
        // a backend that refused either would run it elsewhere instead of keeping the plane
        ggml_tensor * cells = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 256, 4, 1);
        ggml_tensor * toks  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, 9, 1);
        ggml_tensor * sel   = ggml_qsa_top_k_vis(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 9, 1),
                ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 256, 1), cells, toks, 11, 0, true);
        ggml_tensor * msk   = ggml_kq_mask_ids(ctx, cells, toks, ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 11, 9, 1, 1),
                GGML_TYPE_F16, true, false, 0, 0);
        ok = ggml_backend_supports_op(backend, sel) && ggml_backend_supports_op(backend, msk);
    }
    ggml_free(ctx);
    answers.emplace(key, ok);
    return ok;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    // [TAG_PLE_HOIST] before anything else touches gf: the CPU-resident PLE gather has to be the
    // graph's first node for the scheduler to put its split first. One gather per PLE layer, in
    // layer order, so the node set and the tensor names are exactly what the in-loop build produced.
    //
    // SINGLE PLE LAYER ASSUMED. The loader rejects anything else: qwen4exp.ple.layers must name
    // exactly one layer or model loading throws ("only one PLE layer is supported", the n_ple != 1
    // check at the top of this file, ~line 111), because hparams carries one set of PLE hash
    // constants and cannot represent two modules. That is what bounds the cost of hoisting: each
    // hoisted gather stays live from the head of the graph to its consuming layer instead of being
    // built and consumed inside the layer loop, which for the deployed model is one extra live
    // [n_embd, ple_n_heads*n_tokens] tensor (~21 MiB at -ub 2048). With k PLE layers it would be
    // k of them, i.e. ~21 MiB *per* extra layer, not a fixed +21 MiB - so the assertion below is
    // not just a tidy-up: a future multi-PLE model must revisit this trade before it hoists.
    if (qwen4exp_ple_hoist_enabled()) {
        const auto * mctx_ple = static_cast<const llama_memory_hybrid_idx_context *>(mctx);

        int n_ple_hoisted = 0;

        for (int il = 0; il < (int) n_layer; ++il) {
            if (!hparams.is_ple(il)) {
                continue;
            }

            ggml_tensor * emb = build_ple_gather(mctx_ple, il);
            ggml_build_forward_expand(gf, emb);
            ple_gathered[il] = emb;

            ++n_ple_hoisted;
        }

        GGML_ASSERT(n_ple_hoisted <= 1 &&
                "PLE hoisting assumes a single PLE layer (enforced by the loader's n_ple != 1 check); "
                "peak liveness grows per hoisted gather, re-measure before allowing more");
    }

    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), mctx_hyb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                model.layers[il].hc_attn_down_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // an unmasked MTP export needs a hidden row for every token, so in that case the
        // gather is deferred until after t_h_nextn is taken below
        const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

        if (il == n_layer - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                model.layers[il].hc_ffn_down_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // The MTP head consumes the wide residual, before the head mixer collapses it. Export the
    // combine result itself rather than a reshape of it: a pure view gets no backend assignment
    // from the scheduler, and the readback in llama_context looks one up. It is contiguous, so
    // [n_embd, hc, rows] already has the [n_embd_out, rows] layout the reader expects, and it
    // carries exactly the right rows either way -- gathered above when masked, ungathered when not.
    if (cparams.embeddings_nextn) {
        cb(res_hc, "h_nextn", -1);
        res->t_h_nextn = res_hc;

        // deferred from the last layer: collapse to the output rows now that the export is taken
        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cur = qwen4exp_scatter_pruned_logits(ctx0, model, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// [TAG_MTP_EHPROJ_FLAT]
// The MTP draft head applies nextn.eh_proj (q4_K, 5120 x 2560) to the [5120, hc, n_tokens] concat
// of the broadcast token embedding and the wide hidden stream. ggml_mul_mat treats dims 2/3 as a
// batch, so the Vulkan backend sees dst->ne[1] == hc == 4 with src1->ne[2] == n_tokens, and the
// mat-vec gate (ggml-vulkan.cpp ggml_vk_mul_mat:
//     dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2]*src1->ne[3] == 1)
// with mul_mat_vec_max_cols == 8) fails as soon as n_tokens > 1. So it runs the general GEMM with
// a degenerate N of 4 over n_tokens batch groups, and every batch group re-reads and
// re-dequantises the whole 7.03 MiB weight: at n_tokens = 2048 that is 14.1 GiB of nominal A
// traffic for 215 GFLOP of useful work. Measured 145.2 ms per ubatch at 1479 GFLOPS/s against
// 16772 GFLOPS/s for the neighbouring dense MUL_MAT of the same dtype -- 514.6 ms of the 19.3 s
// pp8k prefill (2.7 %), ~11x below the dense-GEMM rate.
// See docs/research/2026-09-07-m3-followups.md section 2 for the full derivation.
//
// The concat is contiguous -- both sources are (e_norm is a ggml_repeat_4d result, h_norm a
// reshape of a ggml_mul) and ggml_concat allocates a fresh contiguous dst -- so the hc axis and
// the token axis can be merged into a single N by a pure view:
//     [5120, hc, n_tokens] -> [5120, hc*n_tokens]
// The element order is unchanged. ggml is row-major over (ne0, ne1, ne2) for a contiguous tensor,
// so element (i0, c, t) sits at i0 + 5120*(c + hc*t) in both forms; the GEMM's column j = c + hc*t
// is exactly the (c, t) column of the 3D form, and reshaping the [2560, hc*n_tokens] result back
// to [2560, hc, n_tokens] restores bit-for-bit the layout every consumer already expects
// (build_hc_mix's rms_norm + reshape_2d, the inp_out_ids get_rows below, the t_h_nextn export).
// Same weight, same k = 5120 reduction, same operand order, same per-tensor scale broadcast --
// the weight is simply read once instead of n_tokens times.
//
// NOT BIT-EXACT BY CONSTRUCTION. ggml_vk_mul_mat_q_f16 selects among the _s/_m/_l tile variants
// and the aligned/unaligned pipelines by shape, so the merged N can land on a different BK and
// reassociate the k-loop. Same kernel family, same dtypes, same operands -- but the summation
// order is not guaranteed, so this has to clear the census oracle
// (suite.py run --tracks speed,exact) before it ships, per the standing speed/quality ruling.
//
// Decode is deliberately left alone. The draft head runs with n_tokens of a handful there; today
// that is a GEMM with N = 4 over n_tokens batch groups, and the flat form would be either a
// mat-vec with hc*n_tokens <= 8 columns (n_tokens <= 2) or a GEMM with N = hc*n_tokens
// (n_tokens >= 3). Analytically the flat form is not worse -- it still reads the 7 MiB weight once
// instead of n_tokens times -- but it crosses a kernel-family boundary at n_tokens == 2 and cannot
// be settled without a GPU run, and decode is the latency-critical path. So the reshape is gated
// on n_tokens >= LLAMA_MTP_EHPROJ_FLAT_MIN_TOKENS (default 32): every decode/verify ubatch keeps
// today's graph node-for-node and value-for-value, and every prefill ubatch (n_ubatch is 512..2048
// in prod) takes the flat path. Set the threshold to 1 to A/B the decode side once the oracle has
// cleared the prefill side.
//
// LLAMA_MTP_EHPROJ_FLAT=0 restores today's graph everywhere.
static bool qwen4exp_mtp_ehproj_flat_enabled(int64_t n_tokens) {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_MTP_EHPROJ_FLAT");
        return s == nullptr || atoi(s) != 0;
    }();

    static const int64_t min_tokens = [] {
        const char * s = getenv("LLAMA_MTP_EHPROJ_FLAT_MIN_TOKENS");
        const int v = s == nullptr ? 32 : atoi(s);
        return (int64_t) (v < 1 ? 1 : v);
    }();

    return enabled && n_tokens >= min_tokens;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for qwen4exp.
//
// The head folds the next token's embedding into the trunk's wide hyper-connection residual,
// runs one trunk-style block over it, and collapses the result with its own mixer before
// reusing the trunk's LM head. The wide post-block residual is exported as t_h_nextn so the
// speculative driver can feed it straight back in for the next draft step.
//
// v1 simplification: the block attends densely. The trunk's QSA only prunes context past a
// 2048-token budget, so dense is a numerical superset; drafts are verified by the target
// either way. The indexer tensors are still loaded so the GGUF stays complete.
// [TAG_MTP_QSA] LLAMA_MTP_QSA=1 (default off): the block runs the trunk's QSA path with its own indexer
// (blk.<n_layer>.indexer.*), as SGLang's Qwen4ExpForCausalLMMTP does with index_share_for_mtp_iteration unset.
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    // resolved at load time to either blk.%d.nextn.hc_head_* or the model-level hc_head_*
    GGML_ASSERT(layer.nextn.hc_head_norm && "MTP block missing its output mixer");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    // [TAG_MTP_CATCHUP_NOATTN] LLAMA_MTP_CATCHUP_ATTN=0 (cparams.mtp_catchup_attn = false): a ubatch that requests no
    // output row and no unmasked hidden export - the speculative catch-up of prefill and verification rows, whose
    // pending_h / verify rows come from the TARGET's embeddings - needs only the draft block's K/V cache writes. Build
    // the embedding / hidden mixing, K proj + norm + RoPE, V proj, the cache's rotations and cpy_k/cpy_v; skip Q, the
    // gate, the attention, wo, the hc combines, the FFN and the head. Exact: none of the skipped nodes feeds a stored
    // byte (K/V depend on cur = build_hc_mix(eh_proj(...)) and the positions only), and a zero-output ubatch extracts
    // nothing. Mixed-output ubatches and the reserve graph (it requests outputs) keep the full graph; graph reuse keys
    // on n_outputs, so the two shapes never alias.
    const bool kv_only = !cparams.mtp_catchup_attn && n_outputs == 0 &&
            (!cparams.embeddings_nextn || cparams.embeddings_nextn_masked) && !cparams.embeddings;

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = kv_only ? nullptr : build_inp_out_ids();

    // [TAG_MTP_QSA] LLAMA_MTP_QSA=1: the context's memory is a llama_memory_hybrid_idx without recurrent layers
    // (llama-model.cpp create_memory) - the block runs the trunk's QSA attention; otherwise today's dense graph
    const auto * mctx_hyb = dynamic_cast<const llama_memory_hybrid_idx_context *>(mctx);

    llm_graph_input_attn_kv * inp_attn = mctx_hyb ? build_attn_inp_kv_hyb() : build_attn_inp_kv();

    // grouped RMSNorm over the wide stream: normalise each hc stream, then scale the flattened
    // [hc_dim] vector with the head's gamma, exactly as build_hc_mix does
    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    // the token embedding is shared across the streams, so broadcast it to hc copies
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // eh_proj holds fc_embedding and fc_hidden side by side, so this one matmul is
    // fc_embedding @ e_norm + fc_hidden @ h_norm, applied to each stream independently.
    // Keeping the streams distinct here is the point of the hyper-connection residual:
    // pooling them before the projection would throw that away.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    // [TAG_MTP_EHPROJ_FLAT] see the comment above graph_mtp: merging the hc axis into N turns a
    // GEMM with N = 4 over n_tokens batch groups into one GEMM with N = hc*n_tokens, so the q4_K
    // weight is read and dequantised once per graph instead of once per token. Prefill only --
    // NOT bit-exact (tile-variant selection may reassociate), oracle-gated.
    ggml_tensor * res_hc = nullptr;

    if (qwen4exp_mtp_ehproj_flat_enabled(n_tokens)) {
        GGML_ASSERT(ggml_is_contiguous(concat) && "MTP eh_proj flat path needs a contiguous concat");

        ggml_tensor * concat_flat = ggml_reshape_2d(ctx0, concat, 2*n_embd, hc*n_tokens);
        cb(concat_flat, "mtp_concat_flat", il);

        ggml_tensor * res_flat = build_lora_mm(layer.nextn.eh_proj, concat_flat, layer.nextn.eh_proj_s);
        cb(res_flat, "mtp_eh_proj_flat", il);

        res_hc = ggml_reshape_3d(ctx0, res_flat, n_embd, hc, n_tokens);
    } else {
        res_hc = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    }

    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, layer.hc_attn_down_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    if (mctx_hyb && kv_only) {
        // [TAG_MTP_QSA] + [TAG_MTP_CATCHUP_NOATTN] the sparse draft's zero-output catch-up: the cache writes of
        // build_layer_attn and nothing else - the indexer K (build_qsa_top_k cache_only: index_k_proj(cur) -> cpy_k, the
        // same node the exhaustive / MAX_UBATCH / sparse branches emit; Codex B2), then K (wk, k norm, IMRoPE) and V (wv)
        // through build_attn_store_kv (the store half of build_attn / build_attn_qsa: same rotations, same cpy_k/cpy_v).
        // Exact: every stored byte (attention K/V cells, indexer K cells) is a function of cur and the positions only.
        // The pooled-key store is not touched here; the census marks these cells dirty (0158) and the next sparse plan
        // pools them, as after an exhaustive ubatch.
        const int64_t n_embd_head = hparams.n_embd_head_v();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

        const uint32_t qsa_ratio = llama_hparams_qsa_ratio(hparams, cparams.qsa_ratio, (uint32_t) il);
        if (mctx_hyb->get_idx() != nullptr && qsa_ratio > 0) {
            build_qsa_top_k(mctx_hyb, cur, inp_pos, inp_attn->get_kq_mask(), sections, il, /*cache_only=*/ true,
                    nullptr, nullptr, nullptr, /*cache_only_forced=*/ true);
        }

        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        cb(Vcur, "Vcur", il);

        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);

        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // the cache writes are the graph's only roots; no t_logits / t_h_nextn
        build_attn_store_kv(inp_attn, Kcur, Vcur, il);
        return;
    }

    if (mctx_hyb) {
        // [TAG_MTP_QSA] the trunk's full-attention layer, QSA included: the indexer reads the block's normed input
        // `cur` (the HC attention mixer's output, the same tensor q/k/v are projected from), exactly as the trunk's
        // layers do and as SGLang's Qwen4ExpAttentionDecoderLayer (is_nextn) feeds `hidden_states` after
        // _prepare_qwen4_exp_attn (attn_hyper_connection.mix) to _compute_qsa_topk_indices (qwen4_exp.py 1388-1419,
        // 1564-1606, 1643-1646). Below LLAMA_QSA_MIN_KV / at an exhaustive budget it is the dense build_attn() plus the
        // indexer K write (cache_only), so the draft's indexer history is complete for the first sparse ubatch.
        cur = build_layer_attn(inp_attn, mctx_hyb, cur, inp_pos, sections, il);
        cb(cur, "mtp_attn_out", il);
    } else {
    // ---- dense attention, mirroring the trunk's full-attention branch ----
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur = nullptr;
    ggml_tensor * gate = nullptr;

    if (!kv_only) {
        ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur_full, "mtp_Qcur_full", il);

        Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "mtp_Qcur_normed", il);

        gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
        gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
        cb(gate, "mtp_gate", il);
    }

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    // IMRoPE, same convention and freq_base as the trunk
    if (!kv_only) {
        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
    }
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    if (!kv_only) {
        cb(Qcur, "mtp_Qcur", il);
    }
    cb(Kcur, "mtp_Kcur", il);

    if (kv_only) {
        // [TAG_MTP_CATCHUP_NOATTN] the cache writes are the graph's only roots; no t_logits / t_h_nextn
        build_attn_store_kv(inp_attn, Kcur, Vcur, il);
        return;
    }

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cb(cur, "mtp_attn_gated", il);

    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);
    } // [TAG_MTP_QSA]

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    // ---- MoE, identical to the trunk's build_layer_ffn ----
    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, layer.hc_ffn_down_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // The next draft step re-enters here, so export the wide stream before it is collapsed.
    // As in the trunk, export the combine result rather than a reshape view of it.
    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    // the head's own mixer collapses the streams and doubles as the output norm
    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // deliberately no res->t_embd: it would be n_embd wide while the context sizes its
    // embedding buffer by n_embd_out (the wide stream). The driver reads t_h_nextn instead.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN4EXP MTP: missing LM head (nextn.shared_head_head or model.output)");

    cur = build_lora_mm(head_w, cur, head_s);
    cur = qwen4exp_scatter_pruned_logits(ctx0, model, head_w, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [TAG_MERGE_ROWS] qkv and z read the same input: one matmul, then the qkv rows and the z rows as views (z_2d is
    // then taken with a view, see build_layer_attn_linear)
    const auto & layer = model.layers[il];
    if (qwen4exp_use_merged(layer.wqkv_z, layer.wqkv, ubatch.n_tokens, loras) && !layer.wqkv_s && !layer.wqkv_gate_s) {
        ggml_tensor * qkvz = build_lora_mm(layer.wqkv_z, input);
        cb(qkvz, "linear_attn_qkvz", il);

        const int64_t n_z   = hparams.ssm_d_state * hparams.ssm_dt_rank;
        const int64_t n_qkv = qkvz->ne[0] - n_z;

        ggml_tensor * qkv_mixed = ggml_view_3d(ctx0, qkvz, n_qkv, n_seq_tokens, n_seqs, qkvz->nb[1], qkvz->nb[1] * n_seq_tokens, 0);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = ggml_view_2d(ctx0, qkvz, n_z, n_seq_tokens * n_seqs, qkvz->nb[1], ggml_row_size(qkvz->type, n_qkv));
        cb(z, "z", il);

        return { qkv_mixed, z };
    }

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias, bool cache_only = false) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias), cache_only(cache_only) {}
    virtual ~llm_graph_input_qsa() = default;

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);

        // [TAG_QSA_BYPASS_EXHAUSTIVE] cache-only: nothing but the indexer K write is in the graph,
        // so skip the O(n_kv) host-side block census entirely
        if (cache_only) {
            return;
        }

        // [TAG_QSA_POOLED_CACHE] the census feeds the store's dirty list and remap instead of blk_cells/blk_pos
        if (pooled_cache) {
            mctx->set_input_qsa_store(cell_blk, bias, qsa_blk, qsa_qry, dirty_cells, dirty_pos, dirty_dst, remap, ubatch, ratio, blk_bias);
            return;
        }

        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, qsa_blk, qsa_qry, ubatch, ratio, blk_bias);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_stream = mctx->get_n_stream();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;

        bool res = true;

        res &= params.ubatch.n_tokens % n_stream == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;

        // the dense decision depends on n_kv and (LLAMA_QSA_MIN_KV) on the ubatch's sequence depth, so a graph is
        // never reused across it: not a dense graph once the window has grown past the budget or the sequence
        // reached LLAMA_QSA_MIN_KV, and not a QSA graph for a ubatch whose sequences are all below it
        // [TAG_MULTISLOT] incl. the sequence coverage: a coverage change at an unchanged shape rejects the reuse
        const bool dense = qwen4exp_qsa_dense(mctx, params.ubatch, n_kv,
                std::min<int64_t>(n_kv, (int64_t) bypass_top_k + (int64_t) ratio - 1));

        if (cache_only) {
            // [TAG_MTP_QSA] a forced cache-only input (the MTP catch-up / MAX_UBATCH) is one whatever the depth
            res &= dense || cache_only_forced;

            return res;
        }

        res &= !dense;

        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;

        // [TAG_QSA_COMPACT_BIAS] a compact graph never pairs with a plane graph
        res &= compact == (blk_bias && mctx->qsa_compact_bias());

        if (compact) {
            res &= qsa_blk->ne[1] == n_blocks && qsa_blk->ne[2] == n_stream;
            res &= qsa_qry->ne[1] == params.ubatch.n_tokens/n_stream && qsa_qry->ne[2] == n_stream;
        } else {
            res &= bias->ne[0]  == (blk_bias ? n_blocks : n_kv);
            res &= bias->ne[1]  == params.ubatch.n_tokens/n_stream;
        }

        if (pooled_cache) {
            // [TAG_QSA_POOLED_CACHE] the dirty list's capacity is part of the topology: plan this ubatch
            // (the plan is cached, so the build and set_input see the same one). cap == n_blocks is the
            // full pool, which has no remap.
            //
            // Contract: qsa_store_plan is a const-context call that MUTATES the memory's (mutable) store - its
            // pass 2 commits slot assignments (new blocks take free slots, vanished blocks free theirs, dirty
            // slots drop to stamp 0) here, during can_reuse, before it is known whether this graph is reused
            // or rebuilt. That is safe because (1) the plan is cached per ubatch (i_cur, ratio, n_kv, blk_bias),
            // so the build and set_input_qsa_store reuse this very plan instead of planning again; (2) no stamp
            // becomes valid here - the new stamps stay pending until set_input_qsa_store arms them and
            // qsa_store_publish commits them after the graph ran; (3) a plan whose graph never carries the store
            // update (a cache-only rebuild) is dropped unarmed, and its dirty slots are already at stamp 0, so
            // the next plan simply pools them again.
            res &= remap == nullptr || remap->ne[0] == n_blocks*n_stream;

            if (res && params.ubatch.n_tokens % n_stream == 0) {
                const int64_t cap = mctx->qsa_store_plan(params.ubatch, ratio, n_kv, blk_bias);

                res &= dirty_cells->ne[0] == (int64_t) ratio*cap;
                res &= dirty_cells->ne[1] == n_stream;
                res &= (remap == nullptr) == (cap == n_blocks);

                if (remap != nullptr) {
                    res &= dirty_dst->ne[0] == cap*n_stream;
                } else {
                    // the full pool's store view starts at the ubatch's first cache stream
                    res &= mctx->qsa_store_s0() == full_s0;
                }
            }

            return res;
        }

        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]; nullptr if compact

    // [TAG_QSA_COMPACT_BIAS] LLAMA_QSA_COMPACT_BIAS=1 (and blk_bias): bias is nullptr, and GGML_OP_QSA_BIAS builds each
    // use of it on the device from these two tables (llama_memory_hybrid_idx_context::set_input_qsa)
    bool          compact   = false;
    ggml_tensor * qsa_blk   = nullptr;   // I32 [2, n_blocks, n_stream]           (key, vis) per block
    ggml_tensor * qsa_qry   = nullptr;   // I32 [2, n_tokens/n_stream, n_stream]  (tail, bit) per query

    // the per-block bias of the whole ubatch (rows == nullptr) or of the query rows `rows` ([np, n_stream] I32):
    // exactly ggml_get_rows(bias, rows) of the plane, bit for bit
    ggml_tensor * bias_rows(ggml_context * ctx, ggml_tensor * rows) const {
        if (compact) {
            return ggml_qsa_bias(ctx, qsa_blk, qsa_qry, rows);
        }
        return rows ? ggml_get_rows(ctx, bias, rows) : bias;
    }

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    // [TAG_QSA_BYPASS_EXHAUSTIVE] only k_idxs is live; cell_blk/blk_cells/blk_pos/bias are nullptr
    const bool cache_only;

    // [TAG_MTP_QSA] cache_only by the graph's own rule (MTP kv-only catch-up, LLAMA_MTP_QSA_MAX_UBATCH), which depends
    // on n_tokens / n_outputs only - both already key the reuse - not on the exhaustive test
    bool cache_only_forced = false;

    // hparams.indexer_top_k, copied so can_reuse() can re-evaluate the bypass condition
    uint32_t bypass_top_k = 0;

    // [TAG_QSA_POOLED_CACHE] LLAMA_QSA_POOLED_CACHE=1: blk_cells/blk_pos are nullptr, the four below are live
    bool pooled_cache = false;

    ggml_tensor * dirty_cells = nullptr;  // I32 [ratio*cap, n_stream]
    ggml_tensor * dirty_pos   = nullptr;  // I32 [4*cap*n_stream]
    ggml_tensor * dirty_dst   = nullptr;  // I32 [cap*n_stream]
    ggml_tensor * remap       = nullptr;  // I32 [n_blocks*n_stream]; nullptr for the full pool (cap == n_blocks)

    uint32_t full_s0 = 0;                 // the full pool's first cache stream
};

// The block pooling chain: members of n_rows blocks per stream (cells: I32 [r*n_rows, n_stream], indices
// into each stream's own cells of k_all) -> mean over the r members -> rms_norm -> rope at the block's
// positions (pos: I32 [4*n_rows*n_stream], sec-major). Returns [idx_dim, n_rows, n_stream] f32. Every op is
// row-local, so a block's row does not depend on which other rows are pooled with it: the recompute (every
// block) and [TAG_QSA_POOLED_CACHE]'s dirty-block write run exactly this chain.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_pool(
        ggml_tensor * k_all,
        ggml_tensor * cells,
        ggml_tensor * pos,
        int64_t       n_rows,
        int64_t       n_stream,
        int *         sections,
        const char *  name_pooled,
        int           il) {
    const int64_t idx_dim = hparams.indexer_head_size;
    const int64_t r       = cells->ne[0]/n_rows;

    // gathers per stream: cells row s indexes stream s's own cells
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, cells);
    members = ggml_reshape_4d(ctx0, members, idx_dim, r, n_rows, n_stream);

    // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_3d(ctx0, members, idx_dim, n_rows, n_stream,
                        members->nb[2], members->nb[3], i*members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
    cb(pooled, name_pooled, il);

    // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_rows*n_stream, 1);
    pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

    // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_rows*n_stream);
    pooled = ggml_rope_multi(ctx0, pooled, pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_rows, n_stream);

    return pooled;
}

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il,
        bool                                    cache_only,
        qsa_prep *                              prep,
        ggml_tensor *                           vis_cells,
        ggml_tensor *                           vis_toks,
        bool                                    cache_only_forced) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    GGML_ASSERT(!cache_only_forced || cache_only);

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    // [TAG_QSA_RATIO_OVERRIDE] the context override wins over the GGUF value
    const int64_t r        = llama_hparams_qsa_ratio(hparams, cparams.qsa_ratio, (uint32_t) il);
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    const int64_t n_blocks = (n_kv + r - 1)/r;

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_stream = mctx_hyb->get_n_stream();
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask and non-causal keeps future cells, so both opt out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tps && kq_mask->ne[3] == n_stream &&
        cparams.causal_attn && !hparams.use_alibi;

    // [TAG_QSA_POOLED_CACHE] LLAMA_QSA_POOLED_CACHE=1: the memory holds a persistent store of pooled keys for
    // every QSA layer (llama_memory_hybrid_idx); this graph pools only the blocks the plan found dirty
    const llama_qsa_pooled_store * store = mctx_hyb->get_qsa_store();
    const bool pooled_cache = store != nullptr;

    if (pooled_cache) {
        GGML_ASSERT((uint32_t) r == store->ratio && store->layers.count(il) == 1 &&
                "qsa: the pooled-key cache holds every QSA layer at the one ratio it was built for");
    }

    // [TAG_QSA_COMPACT_BIAS] the per-block bias as two tables + GGML_OP_QSA_BIAS (knob read by the memory)
    const bool compact = blk_bias && mctx_hyb->qsa_compact_bias();

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;

        // every layer sharing a ratio also shares the exhaustive/not decision (it is a function of
        // n_kv, r and the ubatch's sequence depth alone), so a cached input set can never be the wrong kind
        GGML_ASSERT(inp->cache_only == cache_only);
        GGML_ASSERT(inp->cache_only_forced == cache_only_forced);
        GGML_ASSERT(cache_only || inp->compact == compact);
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias, cache_only);
        qsa->cache_only_forced = cache_only_forced;

        qsa->k_idxs        = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->bypass_top_k  = qwen4exp_qsa_top_k(hparams, il);

        // [TAG_QSA_BYPASS_EXHAUSTIVE] cache-only needs the K write and nothing else: skipping the
        // four layout inputs also drops their device buffers and their host-side fill
        if (!cache_only && pooled_cache) {
            // [TAG_QSA_POOLED_CACHE] the pooling inputs are the dirty list, sized by this ubatch's plan
            const int64_t cap = mctx_hyb->qsa_store_plan(ubatch, (uint32_t) r, n_kv, blk_bias);

            qsa->pooled_cache = true;

            // [TAG_QSA_INPUTS_INC] cell_blk, remap and the block table are views of the store's device mirrors, written
            // in place by set_input (only what changed); a compact graph only, the condition set_input_qsa_store tests
            const bool mir = store->mir_on && compact;

            if (mir) {
                using store_t = llama_qsa_pooled_store;

                GGML_ASSERT(n_kv*n_stream <= ggml_nelements(store->mir_t[store_t::MIR_CELL_BLK]) &&
                            2*n_blocks*n_stream <= ggml_nelements(store->mir_t[store_t::MIR_BTAB]));

                qsa->cell_blk = ggml_view_2d(ctx0, store->mir_t[store_t::MIR_CELL_BLK], n_kv, n_stream,
                        n_kv*sizeof(int32_t), 0);

                qsa->compact = true;
                qsa->qsa_blk = ggml_view_3d(ctx0, store->mir_t[store_t::MIR_BTAB], 2, n_blocks, n_stream,
                        2*sizeof(int32_t), 2*n_blocks*sizeof(int32_t), 0);
                qsa->qsa_qry = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, 2, n_tps, n_stream);

                ggml_set_name (qsa->qsa_qry, "qsa_qry");
                ggml_set_input(qsa->qsa_qry);
            } else {
                qsa->cell_blk    = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
                if (compact) {
                    build_qsa_compact_inputs(qsa.get(), n_blocks, n_tps, n_stream);
                } else {
                    qsa->bias    = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);
                }
            }
            qsa->dirty_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*cap, n_stream);
            qsa->dirty_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*cap*n_stream);

            ggml_set_name(qsa->cell_blk,    "qsa_cell_blk");
            if (!compact) {
                ggml_set_name(qsa->bias,    "qsa_bias");
            }
            ggml_set_name(qsa->dirty_cells, "qsa_dirty_cells");
            ggml_set_name(qsa->dirty_pos,   "qsa_dirty_pos");

            if (!mir) {
                ggml_set_input(qsa->cell_blk);
            }
            if (!compact) {
                ggml_set_input(qsa->bias);
            }
            ggml_set_input(qsa->dirty_cells);
            ggml_set_input(qsa->dirty_pos);

            if (cap < n_blocks) {
                qsa->dirty_dst = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cap*n_stream);
                if (mir) {
                    GGML_ASSERT(n_blocks*n_stream <= ggml_nelements(store->mir_t[llama_qsa_pooled_store::MIR_REMAP]));
                    qsa->remap = ggml_view_1d(ctx0, store->mir_t[llama_qsa_pooled_store::MIR_REMAP], n_blocks*n_stream, 0);
                } else {
                    qsa->remap = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_blocks*n_stream);
                }

                ggml_set_name (qsa->dirty_dst, "qsa_dirty_dst");
                ggml_set_name (qsa->remap,     "qsa_remap");
                ggml_set_input(qsa->dirty_dst);
                if (!mir) {
                    ggml_set_input(qsa->remap);
                }
            } else {
                // the full pool (a cold rebuild, and every reserve graph): the recompute's leaves exactly
                qsa->full_s0 = mctx_hyb->qsa_store_s0();
            }
        } else if (!cache_only) {
            qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
            qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
            qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
            if (compact) {
                build_qsa_compact_inputs(qsa.get(), n_blocks, n_tps, n_stream);
            } else {
                qsa->bias  = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);
            }

            ggml_set_input(qsa->cell_blk);
            ggml_set_input(qsa->blk_cells);
            ggml_set_input(qsa->blk_pos);
            if (!compact) {
                ggml_set_input(qsa->bias);
            }
        }

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // [TAG_QSA_BYPASS_EXHAUSTIVE] the selection would name every cell; the caller runs dense
    // attention instead, and the indexer cache above stays complete for the batches that follow
    if (cache_only) {
        return nullptr;
    }

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_stream].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_stream, k_all->nb[2], k_all->nb[3], 0);

    ggml_tensor * pooled = nullptr;

    // [TAG_QSA_POOL_REMAP] LLAMA_QSA_POOL_REMAP=indexed (default): instead of `pooled`, the store and the remap the
    // score reads it through (a dirty-list graph only; the full pool has no remap and keeps `pooled`)
    ggml_tensor * pool_store = nullptr;
    ggml_tensor * pool_remap = nullptr;
    ggml_tensor * pool_store16 = nullptr; // [TAG_QSA_POOLED_F16_MIRROR]

    if (pooled_cache) {
        // [TAG_QSA_POOLED_CACHE] (M3) one update per layer, before the selection (and every [TAG_QSA_CHUNK]
        // chunk of it) reads it, in the order: (i) the raw indexer write above, (ii) the pooling of the dirty
        // blocks from the stored raw rows - the same chain the recompute runs, over the dirty list instead of
        // every block, (iii) the set_rows into the persistent store (a cache-side tensor, like cpy_k's), (iv)
        // one get_rows remap of the stable slots into the compact block order, which depends on (iii).
        // Unchanged valid blocks cost that one gathered row and no pooling, norm or rope.
        //
        // The FULL pool (cap == n_blocks: the plan found about every block dirty - a cold rebuild - and every
        // reserve graph) has no (iv): its dirty list is every compact id in order, i.e. the recompute's own
        // blk_cells/blk_pos, so its output IS the recompute's `pooled` and is used directly. The plan re-lays the
        // stream's slots in compact order (slot = block id), so (iii) is one ggml_cpy into the store's first
        // n_blocks rows of each stream, expanded right after the pooling. That graph is the recompute's graph
        // with the same leaves plus one CPY node, whose output is a view of the store (no compute-buffer bytes).
        const bool    full_pool = inp->remap == nullptr;
        const int64_t cap       = inp->dirty_cells->ne[0]/r;

        GGML_ASSERT(full_pool == (cap == n_blocks));

        ggml_tensor * store_t = store->layers.at(il);

        ggml_tensor * fresh = build_qsa_pool(k_all, inp->dirty_cells, inp->dirty_pos, cap, n_stream, sections,
                full_pool ? "indexer_k_pooled" : "indexer_k_dirty_pooled", il);

        if (full_pool) {
            ggml_tensor * dst = ggml_view_3d(ctx0, store_t, idx_dim, n_blocks, n_stream,
                    store_t->nb[1], store_t->nb[1]*store->n_rows, store_t->nb[1]*store->n_rows*inp->full_s0);

            ggml_tensor * st = ggml_cpy(ctx0, fresh, dst);
            ggml_format_name(st, "indexer_k_store-%d", il);

            // [TAG_QSA_POOLED_F16_MIRROR] the same rows into the f16 mirror (the full pool's score reads the f32 copy)
            if (store->f16_mirror) {
                ggml_tensor * store16 = store->layers_f16.at(il);
                ggml_tensor * dst16   = ggml_view_3d(ctx0, store16, idx_dim, n_blocks, n_stream,
                        store16->nb[1], store16->nb[1]*store->n_rows, store16->nb[1]*store->n_rows*inp->full_s0);
                ggml_tensor * st16 = ggml_cpy(ctx0, fresh, dst16);
                ggml_format_name(st16, "indexer_k_store16-%d", il);
                ggml_build_forward_expand(gf, st16);
            }

            if (n_stream == 1) {
                // the selection reads the rows it just stored (f32 copies: the same bits), so the chain is pulled
                // into the graph where the recompute's is - at its first consumer - and `fresh` dies at the copy.
                // The read is a real tensor (ggml_cont of the store view), not the view itself: it is the full-pool
                // twin of the remap get_rows output of every run-time graph with cap < n_blocks - same shape, same
                // lifetime across the [TAG_QSA_CHUNK] chunks - so the reserve (always the full pool) holds every
                // tensor a run-time graph holds, with pooling planes at least as large (M3 fix round 1: with the
                // bare view the reserve lacked this plane and the buffer regrew inside the graph span).
                pooled = ggml_cont(ctx0, st);
            } else {
                // the store view skips the other streams' rows between batches: keep the contiguous rows for the
                // GEMM and expand the copy on its own
                ggml_build_forward_expand(gf, st);
                pooled = fresh;
            }
        } else {
            cb(fresh, "indexer_k_dirty", il);

            ggml_tensor * st = ggml_set_rows(ctx0, store_t,
                    ggml_reshape_2d(ctx0, fresh, idx_dim, cap*n_stream), inp->dirty_dst);
            ggml_format_name(st, "indexer_k_store-%d", il);

            // [TAG_QSA_POOLED_F16_MIRROR] the same rows, from the same values, into the f16 mirror - expanded here, so it
            // is written whether or not this graph's score reads it (a mat-vec width reads the f32 store)
            ggml_tensor * st16 = nullptr;
            if (store->f16_mirror) {
                st16 = ggml_set_rows(ctx0, store->layers_f16.at(il),
                        ggml_reshape_2d(ctx0, fresh, idx_dim, cap*n_stream), inp->dirty_dst);
                ggml_format_name(st16, "indexer_k_store16-%d", il);
                ggml_build_forward_expand(gf, st16);
            }

            if (llama_qsa_pool_remap_indexed()) {
                // [TAG_QSA_POOL_REMAP] no copy: (iv) is folded into the score, which reads row remap[b] of the store
                // (after (iii), through st) for block b - ggml_mul_mat_rows, bit-identical to get_rows + mul_mat. The
                // remap input is shared by every layer of this ratio; the store is the layer's own.
                pool_store = st;
                pool_remap = ggml_reshape_2d(ctx0, inp->remap, n_blocks, n_stream);
                pool_store16 = st16;
            } else {
                pooled = ggml_get_rows(ctx0, st, inp->remap);
                pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
            }
        }
    } else {
        pooled = build_qsa_pool(k_all, inp->blk_cells, inp->blk_pos, n_blocks, n_stream, sections,
                "indexer_k_pooled", il);
    }
    if (pooled) {
        cb(pooled, "indexer_k", il);
    }

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
    q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q), idx_dim, n_idx_h*n_tps, n_stream);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) qwen4exp_qsa_top_k(hparams, il) + r - 1);

    // [TAG_QSA_CHUNK] the query-independent half is done; the caller runs the selection one query
    // chunk at a time, next to each chunk's attention (build_qsa_chunked)
    if (prep) {
        prep->inp      = inp;
        prep->pooled   = pooled;
        prep->pool_store = pool_store;
        prep->pool_remap = pool_remap;
        prep->pool_store16 = pool_store16;
        prep->q        = q;
        prep->kq_mask  = kq_mask;
        prep->width    = width;
        prep->blk_bias = blk_bias;
        prep->vis_cells = vis_cells; // [TAG_QSA_VIS]
        prep->vis_toks  = vis_toks;
        return nullptr;
    }

    ggml_tensor * bias = inp->bias_rows(ctx0, nullptr);
    if (inp->compact) {
        // [TAG_QSA_COMPACT_BIAS] place the QSA_BIAS node before the score chain, where the plane (a leaf) used to be
        // available: expanded later, as src[1] of the bias ADD, it would sit between the head-sum ADDs and that ADD
        // and split the chain the Vulkan multi_add fusion folds (the chunked path keeps its old get_rows position)
        ggml_build_forward_expand(gf, bias);
    }

    return build_qsa_select(inp, pooled, q, bias, kq_mask, width, blk_bias, il, pool_store, pool_remap, vis_cells, vis_toks,
            pool_store16);
}

// [TAG_QSA_COMPACT_BIAS] the two table inputs that replace the [n_blocks, n_tps, n_stream] f32 plane
void llama_model_qwen4exp::graph::build_qsa_compact_inputs(
        llm_graph_input_qsa * qsa,
        int64_t               n_blocks,
        int64_t               n_tps,
        int64_t               n_stream) {
    qsa->compact = true;
    qsa->qsa_blk = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, 2, n_blocks, n_stream);
    qsa->qsa_qry = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, 2, n_tps,    n_stream);

    ggml_set_name(qsa->qsa_blk, "qsa_blk");
    ggml_set_name(qsa->qsa_qry, "qsa_qry");

    ggml_set_input(qsa->qsa_blk);
    ggml_set_input(qsa->qsa_qry);
}

// The query-dependent half of the indexer: score the queries in q against the pooled keys and
// select `width` cells per query. q is [idx_dim, n_idx_h*n_tps, n_stream]; bias and kq_mask carry
// the same n_tps queries (the whole ubatch, or one [TAG_QSA_CHUNK] chunk of it).
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_select(
        llm_graph_input_qsa * inp,
        ggml_tensor *         pooled,
        ggml_tensor *         q,
        ggml_tensor *         bias,
        ggml_tensor *         kq_mask,
        int64_t               width,
        bool                  blk_bias,
        int                   il,
        ggml_tensor *         pool_store,
        ggml_tensor *         pool_remap,
        ggml_tensor *         vis_cells,
        ggml_tensor *         vis_toks,
        ggml_tensor *         pool_store16) {
    GGML_ASSERT((pooled != nullptr) != (pool_store != nullptr && pool_remap != nullptr));

    // [TAG_QSA_DUMP_SCORE] (pkg2 review, Major) LLAMA_QSA_DUMP_SCORE=1: the score capture (llama-context.cpp
    // dump_score_operand) reads the pooled-key operand when the observed "indexer_score-<il>" node has run. In the unfused
    // chain the operand's last consumer is the earlier matmul, so ggml-alloc may hand its storage to the CONT / ADD nodes
    // in between. Keep the operand (and the remap rows) alive for the whole graph while the capture is on - debug only,
    // the graph and its allocation are unchanged with the knob off. Knob on: the flag also lands on the external cache
    // tensor (pool_store, and its view_src chain) for good - harmless, KEEP_ALIVE only matters for gallocr-owned tensors -
    // and is set even when the caller installed its own eval callback (which ignores the dump): some compute-buffer bytes.
    if (qwen4exp_qsa_dump_score()) {
        ggml_set_keep_alive(pooled ? pooled : pool_store);
        if (pool_remap != nullptr) {
            ggml_set_keep_alive(pool_remap);
        }
    }

    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t n_blocks = pooled ? pooled->ne[1] : pool_remap->ne[0];
    const int64_t n_kv     = inp->cell_blk->ne[0];
    const int64_t n_tps    = q->ne[1]/n_idx_h;
    const int64_t n_stream = q->ne[2];

    ggml_tensor * score  = nullptr;
    bool          biased = false;

    // [TAG_QSA_SCORE_EPILOGUE] the whole chain below as one node, where the backend takes it (bias folded unless the score
    // capture needs the pre-bias plane)
    if (qwen4exp_qsa_score_epilogue()) {
        const bool fold_bias = blk_bias && !qwen4exp_qsa_dump_score();
        ggml_tensor * fused = ggml_qsa_score(ctx0, pooled ? pooled : pool_store, pooled ? nullptr : pool_remap, q,
                fold_bias ? bias : nullptr, (int) n_idx_h);
        // [TAG_QSA_POOLED_F16_MIRROR] the store's f16 mirror, a read hint: the coopmat1 product reads it instead of
        // converting the f32 store (same f16 values); the result is the same node's either way
        if (pool_store16 != nullptr && !pooled) {
            ggml_qsa_score_set_a_f16(fused, pool_store16);
        }
        if (sched != nullptr && ggml_backend_supports_op(ggml_backend_sched_get_backend(sched, 0), fused)) {
            score  = fused;
            biased = fold_bias;
        }
    }

    if (score == nullptr) {
        // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
        // [TAG_QSA_POOL_REMAP] the same product read straight from the store: [n_blocks, n_idx_h*n_tps, n_stream] either way
        score = pooled ? ggml_mul_mat(ctx0, pooled, q) : ggml_mul_mat_rows(ctx0, pool_store, pool_remap, q);
        score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
        score = qwen4exp_qsa_relu_inplace() ? ggml_relu_inplace(ctx0, score) : ggml_relu(ctx0, score); // [TAG_QSA_RELU_INPLACE]

        // the heads sit side by side on ne[1] and there are few of them, so summing slices beats a
        // transpose that would carry the whole block by token surface twice over
        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, n_tps, n_stream,
                    score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }

        score = summed;
    }

    if (biased) {
        // [TAG_QSA_SCORE_EPILOGUE] the biased plane: not the "indexer_score-<il>" the score capture hashes
        cb(score, "indexer_score_biased", il);
    } else {
        cb(score, "indexer_score", il);

        // one value per block, so it is cheaper to bias here than after the cells are expanded
        if (blk_bias) {
            score = ggml_add(ctx0, score, bias);
        }
    }

    // [TAG_QSA_WHOLE_BLOCKS] LLAMA_QSA_WHOLE_BLOCKS=1: the reference's block-level selection, always as the one node
    // (fused select or not): the indexer_top_k/r best eligible blocks whole plus the boosted tail, ascending, the row
    // padded with a repeat of its last id - so a query attends to 2048 + t cells, t = (q+1) mod r, where the cell
    // form attends to 2051 (D1), and a block is never split at a tie (D2). The block form needs the per-block bias
    // (a census-shaped score); the per-cell bias path (alibi / non-causal) keeps the cell form. The Vulkan shader
    // binds an f16 mask; an f32 mask (no flash attention) runs the op on the CPU.
    // [TAG_QSA_VIS] vis_cells/vis_toks: the same selections over the metadata (ggml_qsa_top_k_vis), no mask plane
    // (without the per-block bias - alibi, non-causal - the selection keeps the plane, and qsa_use_vis' attention side is
    // independent of it)
    const bool vis = vis_cells != nullptr && vis_toks != nullptr && blk_bias;

    if (blk_bias && llama_qsa_whole_blocks()) {
        GGML_ASSERT((vis || kq_mask->ne[2] == 1) && qwen4exp_qsa_top_k(hparams, il) % inp->ratio == 0);
        ggml_tensor * top_k = vis
            ? ggml_qsa_top_k_vis(ctx0, score, inp->cell_blk, vis_cells, vis_toks, (int) width,
                    (int) (qwen4exp_qsa_top_k(hparams, il) / inp->ratio), cparams.causal_attn)
            : ggml_qsa_top_k_blocks(ctx0, score, inp->cell_blk, kq_mask, (int) width,
                    (int) (qwen4exp_qsa_top_k(hparams, il) / inp->ratio));
        cb(top_k, "indexer_top_k", il);

        return top_k;
    }
    if (!blk_bias && llama_qsa_whole_blocks()) {
        static std::atomic<bool> warned { false };
        if (!warned.exchange(true)) {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_WHOLE_BLOCKS=1 needs the per-block bias (causal, non-alibi, a full mask): "
                           "this graph keeps the cell selection; warned once\n", __func__);
        }
    }

    // [TAG_QSA_FUSED_SELECT] the single-node form. Gated on exactly what the Vulkan topk_qsa
    // pattern needs (blk_bias, i.e. the per-block bias and a matching mask, and an f16 mask - the
    // shader's binding 3 is float16_t and the pattern likewise only matches through ggml_cast), so
    // no configuration that fuses today falls off the fused path, and none that does not gets a
    // node the backend would have to run somewhere else.
    if (blk_bias && (vis || (kq_mask->type == GGML_TYPE_F16 && kq_mask->ne[2] == 1)) && qwen4exp_qsa_fused_select()) {
        // one node, [width, n_tps, 1, n_stream] i32 - already the shape build_attn_qsa wants, so
        // neither the ggml_cont nor the reshape below is needed either.
        // Note for the node-diff probes: this arm has no "indexer_score_tokens-<il>" node, because
        // the surface that name refers to is the one the op exists not to materialize. The
        // selection keeps the name "indexer_top_k-<il>", so a diff of two runs on the SAME side of
        // the knob still pairs it. It does NOT pair ACROSS the knob: fa_rb_nodediff's key is
        // name|op|type|ne (the node-diff tool's record key) and the op differs -
        // RESHAPE below, QSA_TOP_K here. Cross-arm equality is established by the corpus census and
        // by qsa_fusion_replay's REPLAY_QSA_OP_SELFTEST, not by the node diff.
        ggml_tensor * top_k = vis
            ? ggml_qsa_top_k_vis(ctx0, score, inp->cell_blk, vis_cells, vis_toks, (int) width, 0, cparams.causal_attn)
            : ggml_qsa_top_k(ctx0, score, inp->cell_blk, kq_mask, (int) width);
        cb(top_k, "indexer_top_k", il);

        return top_k;
    }
    GGML_ASSERT(!vis); // qwen4exp_qsa_vis requires the fused select

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary
    ggml_tensor * expanded = ggml_get_rows(ctx0,
            ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
    expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

    if (blk_bias) {
        // flash attention keeps the mask in f16; the scores are f32
        ggml_tensor * mask = kq_mask->type == GGML_TYPE_F32 ? kq_mask : ggml_cast(ctx0, kq_mask, GGML_TYPE_F32);
        expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_tps, n_stream));
    } else {
        expanded = ggml_add(ctx0, expanded, bias);
    }
    cb(expanded, "indexer_score_tokens", il);

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_stream], matching the KQ mask.
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il,
        const qsa_prep *          prep,
        bool                      vis) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    // [TAG_QSA_CHUNK] the caller has already decided that this ubatch takes the scan branch
    if (prep) {
        ggml_tensor * cur = build_qsa_chunked(inp, q_cur, *prep, kq_scale, il);
        cb(cur, "kqv_out", il);

        if (inp->self_v_rot) {
            cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
        }

        return cur;
    }

    const int64_t width = top_k->ne[0];
    const int64_t n_tps = top_k->ne[1];
    const int64_t n_kv  = mctx_cur->get_n_kv();

    // the two costs meet at 2*n_tps*width == n_kv; the margin keeps the windows small enough
    // for the compute buffer of a decode graph. flash attention is what keeps the values as rows
    ggml_tensor * cur = nullptr;
    if (qwen4exp_qsa_use_gather(cparams.flash_attn, n_tps, top_k->ne[3], width, n_kv)) {
        cur = build_qsa_gather(inp, q_cur, top_k, kq_scale, il, vis);
    } else if (vis) {
        // [TAG_QSA_VIS] qsa_use_vis() includes the indexed rule: indexed flash attention over the metadata
        cur = build_qsa_indexed(inp, q_cur, top_k, kq_scale, il, nullptr, inp->self_kq_vis_toks);
    } else if (qsa_use_indexed(inp, n_tps, il)) {
        cur = build_qsa_indexed(inp, q_cur, top_k, kq_scale, il); // [TAG_QSA_INDEXED_FA] the implementation, not the branch
    } else {
        cur = build_qsa_scan(inp, q_cur, top_k, kq_scale, il);
    }
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

// The window stays whole and the mask hides every cell the selection leaves out.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_scan(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il,
        ggml_tensor *             kq_mask_chunk,
        ggml_tensor *             zeros_chunk) {
    const auto * mctx_cur = inp->mctx;

    // [TAG_QSA_CHUNK] a query chunk brings its own rows of the mask and a view of its layer's zeros
    ggml_tensor * kq_mask = kq_mask_chunk ? kq_mask_chunk : inp->get_kq_mask();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = zeros_chunk;
    if (!zeros) {
        zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx0, zeros, 0.0f);
    }

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    return build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, kq_scale, il);
}

bool llama_model_qwen4exp::graph::qsa_use_indexed(llm_graph_input_attn_kv * inp, int64_t n_tps, int il, bool vis) {
    if (!qwen4exp_qsa_prefill_indexed() || !cparams.flash_attn || n_tps <= 8) { return false; } // decode/verify: never touched
    // the cache's types and the layer's shape, no K/V views: the check builds nothing in ctx0
    const ggml_type type_k = inp->mctx->type_k(), type_v = inp->mctx->type_v();
    if (qwen4exp_qsa_indexed_supported(sched, type_k, type_v, hparams.n_embd_head_k(il), hparams.n_embd_head_v(il),
                hparams.n_head(il), hparams.n_head_kv(il), vis)) {
        return true;
    }
    if (vis) {
        static std::atomic<bool> warned_vis{false};
        if (!warned_vis.exchange(true)) {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_VIS=1: the first backend does not support indexed flash attention over the metadata"
                    " for K/V %s/%s - the QSA layers keep the KQ mask plane\n", __func__, ggml_type_name(type_k), ggml_type_name(type_v));
        }
        return false;
    }
    static std::atomic<bool> warned{false}; // graphs may be built concurrently (several contexts): once, race-free
    if (!warned.exchange(true)) {
        LLAMA_LOG_WARN("%s: LLAMA_QSA_PREFILL=indexed: the first backend does not support indexed flash attention for K/V %s/%s"
                " - using the scan\n", __func__, ggml_type_name(type_k), ggml_type_name(type_v));
    }
    return false;
}

// [TAG_QSA_VIS] see qwen4exp_qsa_vis(): this ubatch's QSA consumers read the metadata (decided once per ubatch, from its
// token count, like the KQ_VIS class and the indexed rule)
bool llama_model_qwen4exp::graph::qsa_use_vis(llm_graph_input_attn_kv * inp, int64_t n_tps, int il) {
    if (!qwen4exp_qsa_vis() || !inp->kq_vis_fa || inp->self_kq_vis_cells == nullptr || inp->self_kq_vis_toks == nullptr) {
        return false;
    }
    return qsa_use_indexed(inp, n_tps, il, /*vis =*/ true);
}

// [TAG_QSA_INDEXED_FA] build_qsa_scan's attention without the scan: flash attention reads K, V and the mask through
// the ids, so each query touches only its W cells. Same live cells and logits (spec §2); E1 class.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_indexed(llm_graph_input_attn_kv * inp, ggml_tensor * q_cur,
        ggml_tensor * top_k, float kq_scale, int il, ggml_tensor * kq_mask_chunk, ggml_tensor * vis_toks) {
    if (llama_qsa_whole_blocks() && top_k->ne[0] > 1) {
        // [TAG_QSA_WHOLE_BLOCKS] the padding repeats become dead columns (qwen4exp_qsa_mask_repeats)
        top_k = qwen4exp_qsa_mask_repeats(ctx0, top_k);
        cb(top_k, "indexer_top_k_dedup", il);
    }
    if (qwen4exp_qsa_indexed_sort()) {
        top_k = qwen4exp_qsa_sort_ids(ctx0, top_k); // [TAG_QSA_INDEXED_SORT]
        cb(top_k, "indexer_top_k_sorted", il);
    }
    if (vis_toks) {
        // [TAG_QSA_VIS] the mask of column c is the metadata's value at cell ids[c]: no plane, no chunk rows
        return build_attn_mha(q_cur, inp->mctx->get_k(ctx0, il), inp->mctx->get_v(ctx0, il), nullptr, nullptr, nullptr, nullptr,
                kq_scale, il, top_k, inp->self_kq_vis_cells, vis_toks);
    }
    ggml_tensor * kq_mask = kq_mask_chunk ? kq_mask_chunk : inp->get_kq_mask();
    return build_attn_mha(q_cur, inp->mctx->get_k(ctx0, il), inp->mctx->get_v(ctx0, il), nullptr, kq_mask, nullptr, nullptr,
            kq_scale, il, top_k);
}

// [TAG_QSA_CHUNK] the scan branch, one query chunk at a time. For each chunk, in program order:
// the indexer score and selection of its queries (build_qsa_select over a view of the indexer q and
// the chunk's rows of the bias and the mask, gathered on the device), then its scan attention
// (build_qsa_scan over the same mask rows and a view of q_cur), then the running output grows by its
// rows (a left-deep concat: the copy of the output so far is the price of keeping one chunk's planes
// live). build_attn_mha expands each chunk's attention and the concat is expanded right after it, so
// chunk c's nodes all precede chunk c+1's and ggml-alloc can free chunk c's planes before it places
// chunk c+1's. Live across chunks: the pooled keys, the indexer q, the shared zeros, the running
// output - and the whole-ubatch mask and bias uploads, which the dense layers keep anyway.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_chunked(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        const qsa_prep &          prep,
        float                     kq_scale,
        int                       il) {
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t n_stream = prep.q->ne[2];
    const int64_t n_tps    = prep.q->ne[1]/n_idx_h;
    const int64_t width    = prep.width;

    const int64_t n_chunks = (n_tps + prep.chunk - 1)/prep.chunk;
    const int64_t n_short  = n_tps/n_chunks;      // every chunk has n_short or n_short + 1 queries
    const int64_t n_long   = n_tps%n_chunks;      // ... and the first n_long have n_short + 1
    const bool    pad      = qwen4exp_qsa_chunk_pad();
    const int64_t n_max    = pad ? prep.chunk : n_short + (n_long > 0 ? 1 : 0);

    GGML_ASSERT(n_chunks > 1 && n_max <= prep.chunk);
    GGML_ASSERT(q_cur->ne[2] == n_tps*n_stream);

    ggml_tensor * kq_mask = prep.kq_mask;   // [n_kv, n_tps, 1, n_stream]

    // [TAG_QSA_VIS] the ubatch reads the metadata (qsa_use_vis, decided in build_layer_attn): no chunk mask rows, the token
    // records of the chunk's rows instead; the per-block bias is required (without it the chunk keeps the plane rows)
    const bool vis = prep.vis_cells != nullptr && prep.vis_toks != nullptr && prep.blk_bias;

    // [TAG_QSA_INDEXED_FA] decided once for the ubatch (n_tps > C >= 64), so a CHUNK_PAD tail of <= 8 queries is indexed too
    // ([TAG_QSA_VIS] qsa_use_vis includes the indexed rule over the metadata)
    const bool indexed = vis || qsa_use_indexed(inp, n_tps, il);

    // one zero source per layer, sized for the longest chunk; each chunk reads a view of it (the scan only)
    ggml_tensor * zeros = nullptr;
    if (!indexed) {
        zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, width, n_max, n_stream);
        zeros = ggml_fill(ctx0, zeros, 0.0f);
    }

    ggml_tensor * out = nullptr;

    int64_t t0 = 0;
    for (int64_t c = 0; c < n_chunks; ++c) {
        // [TAG_QSA_CHUNK_PAD] full chunks, the last one holding the rest; np = the queries it is scored as
        const int64_t nc = pad ? std::min(prep.chunk, n_tps - t0) : n_short + (c < n_long ? 1 : 0);
        const int64_t np = pad ? prep.chunk : nc;

        // q is a device node, so a view of it is free. The mask and the bias are graph INPUTS, and
        // ggml_backend_sched puts a view of an input on the input's (CPU) backend and uploads it as a
        // split input of its own: nine [n_kv, nc] uploads per layer, all placed at the start of their
        // split, which breaks the one-chunk-live lifetime and multiplies the splits (measured: the
        // pp reserve graph went from 1 to 8 device splits). So their chunk rows are GATHERED on the device
        // from the whole-ubatch copy the scheduler uploads anyway: get_rows of rows t0 .. t0+nc of
        // every stream. f16 -> f32 -> f16 is exact, so the chunk mask holds exactly the input's values.
        ggml_tensor * rows = ggml_arange(ctx0, (float) t0, (float) (t0 + np), 1.0f); // [np]
        if (np > nc) {
            // [TAG_QSA_CHUNK_PAD] the padding rows repeat the last query t0 + nc - 1 = n_tps - 1
            rows = ggml_clamp(ctx0, rows, (float) t0, (float) (n_tps - 1));
        }
        if (n_stream > 1) {
            rows = ggml_repeat_4d(ctx0, rows, np, n_stream, 1, 1);
        }
        rows = ggml_cast(ctx0, rows, GGML_TYPE_I32); // [np, n_stream]

        ggml_tensor * q_idx = nullptr;
        if (np > nc) {
            // [TAG_QSA_CHUNK_PAD] a view would run past the ubatch: gather the rows (each query's n_idx_h heads)
            q_idx = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, prep.q, prep.q->ne[0]*n_idx_h, n_tps, n_stream), rows);
            q_idx = ggml_reshape_3d(ctx0, q_idx, prep.q->ne[0], n_idx_h*np, n_stream);
        } else {
            q_idx = ggml_view_3d(ctx0, prep.q, prep.q->ne[0], n_idx_h*nc, n_stream,
                    prep.q->nb[1], prep.q->nb[2], t0*n_idx_h*prep.q->nb[1]);
        }
        if (!ggml_is_contiguous(q_idx)) {
            // more than one stream interleaves the rows; the Vulkan GEMM wants contiguous batches
            q_idx = ggml_cont(ctx0, q_idx);
        }

        // [n_blocks or n_kv, np, n_stream] f32: get_rows of the plane, or [TAG_QSA_COMPACT_BIAS] QSA_BIAS over rows
        ggml_tensor * bias_c = prep.inp->bias_rows(ctx0, rows);

        // [n_kv, np, 1, n_stream], back in the mask's own type
        ggml_tensor * mask_c = nullptr;
        // [TAG_QSA_VIS] [4, np, n_stream] I32: the chunk's token records (the CHUNK_PAD rows repeat the last query's), for
        // the selection and - cut to the nc real rows - the indexed flash attention
        ggml_tensor * toks_c = nullptr;
        if (vis) {
            toks_c = ggml_get_rows(ctx0, prep.vis_toks, rows);
            cb(toks_c, "qsa_vis_toks_c", il);
        } else if (kq_mask->type == GGML_TYPE_F16 && qwen4exp_qsa_gather_keeps(kq_mask->type)) {
            // [TAG_QSA_GATHER_Q8] gathered as f16 rows, no f32 transient and no cast
            mask_c = ggml_get_rows_type(ctx0, kq_mask, ggml_reshape_3d(ctx0, rows, np, 1, n_stream), kq_mask->type);
        } else {
            mask_c = ggml_get_rows(ctx0, kq_mask, ggml_reshape_3d(ctx0, rows, np, 1, n_stream));
            if (kq_mask->type != GGML_TYPE_F32) {
                mask_c = ggml_cast(ctx0, mask_c, kq_mask->type);
            }
        }

        ggml_tensor * top_k = build_qsa_select(prep.inp, prep.pooled, q_idx, bias_c, mask_c, width, prep.blk_bias, il,
                prep.pool_store, prep.pool_remap, vis ? prep.vis_cells : nullptr, toks_c, prep.pool_store16);

        if (np > nc) {
            // [TAG_QSA_CHUNK_PAD] THE CUT: from here on only the nc real queries exist - the ids and the mask rows
            // the scan reads are views of their first nc columns, so the padding ids reach no mask, no attention
            // and no output. The padded selection is renamed so LLAMA_QSA_DUMP_IDS hashes the cut ids instead.
            ggml_format_name(top_k, "indexer_top_k_pad-%d", il);
            top_k = ggml_view_4d(ctx0, top_k, width, nc, 1, n_stream, top_k->nb[1], top_k->nb[2], top_k->nb[3], 0);
            if (vis) {
                // [TAG_QSA_VIS] the attention's token records: the first nc rows of each stream
                toks_c = ggml_view_3d(ctx0, toks_c, 4, nc, n_stream, toks_c->nb[1], toks_c->nb[2], 0);
                if (!ggml_is_contiguous(toks_c)) {
                    toks_c = ggml_cont(ctx0, toks_c);
                }
            } else {
                mask_c = ggml_view_4d(ctx0, mask_c, mask_c->ne[0], nc, 1, n_stream, mask_c->nb[1], mask_c->nb[2], mask_c->nb[3], 0);
            }
            if (!ggml_is_contiguous(top_k)) {
                // more than one stream: the real columns of each stream are apart. The mask cont is an
                // [n_kv, nc, 1, n_stream] f16 transient next to the padded rows. Measured (M4.3 fix round 1, reserve
                // replay, --parallel 5 without --kv-unified, ub 2048, span 196 608: 2 x 205 -> 256 + 154 padded per
                // stream): the knob costs +379.29 MiB there (2378.59 -> 2757.88), this cont ([196608, 154, 5] f16 =
                // 288.8 MiB) being most of it. With a unified cache (n_stream = 1, deployed) both are views: +3.09 MiB.
                top_k  = ggml_cont(ctx0, top_k);
                if (mask_c) {
                    mask_c = ggml_cont(ctx0, mask_c);
                }
            }
            cb(top_k, "indexer_top_k", il);
        }

        // the attention queries of the chunk, laid out as build_attn_mha splits them into streams
        ggml_tensor * q_att = nullptr;
        if (n_stream == 1) {
            q_att = ggml_view_3d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], nc,
                    q_cur->nb[1], q_cur->nb[2], t0*q_cur->nb[2]);
        } else {
            q_att = ggml_view_4d(ctx0, q_cur, q_cur->ne[0], q_cur->ne[1], nc, n_stream,
                    q_cur->nb[1], q_cur->nb[2], n_tps*q_cur->nb[2], t0*q_cur->nb[2]);
            q_att = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q_att), q_cur->ne[0], q_cur->ne[1], nc*n_stream);
        }

        // [n_embd_head_v*n_head, nc*n_stream], stream-major
        ggml_tensor * o = nullptr;
        if (indexed) {
            // the ids are this chunk's own QSA_TOP_K output (or the CHUNK_PAD cut: a view at offset 0, or its cont)
            o = build_qsa_indexed(inp, q_att, top_k, kq_scale, il, mask_c, toks_c); // toks_c: [TAG_QSA_VIS], else NULL
        } else {
            ggml_tensor * zeros_c = ggml_view_4d(ctx0, zeros, 1, width, nc, n_stream,
                    zeros->nb[1], zeros->nb[2], zeros->nb[3], 0);
            o = build_qsa_scan(inp, q_att, top_k, kq_scale, il, mask_c, zeros_c);
        }
        o = ggml_reshape_3d(ctx0, o, o->ne[0], nc, n_stream);

        out = out ? ggml_concat(ctx0, out, o, 1) : o;
        ggml_build_forward_expand(gf, out);

        t0 += nc;
    }

    GGML_ASSERT(t0 == n_tps);

    return ggml_reshape_2d(ctx0, out, out->ne[0], n_tps*n_stream);
}

// Key and value traffic follows the budget instead of the whole cache. The queries ride the
// stream axis of the attention, each carrying the window its own selection named.
ggml_tensor * llama_model_qwen4exp::graph::build_qsa_gather(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il,
        bool                      vis) {
    const auto * mctx_cur = inp->mctx;

    const int64_t width    = top_k->ne[0];
    const int64_t n_tps    = top_k->ne[1];
    const int64_t n_stream = top_k->ne[3];
    const int64_t n_q      = n_tps*n_stream;

    ggml_tensor * k_all = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v_all = mctx_cur->get_v(ctx0, il);

    const int64_t n_kv = k_all->ne[2];

    // a cell holds its heads back to back, so one row of the gather is one whole cell
    ggml_tensor * k_cells = ggml_view_3d(ctx0, k_all, k_all->ne[0]*k_all->ne[1], n_kv, n_stream,
            k_all->nb[2], k_all->nb[3], 0);
    ggml_tensor * v_cells = ggml_view_3d(ctx0, v_all, v_all->ne[0]*v_all->ne[1], n_kv, n_stream,
            v_all->nb[2], v_all->nb[3], 0);

    // a cell index names a cell of its own stream, and the ubatch lays the queries of a stream
    // out contiguously, so the flat index of a window is the index of its query
    ggml_tensor * idx_stream = ggml_reshape_2d(ctx0, top_k, width*n_tps, n_stream);

    // [TAG_QSA_GATHER_Q8] in the cache's own type: a q8_0 cell row is n_head_kv*head_dim/32 whole
    // blocks (16 at 2 x 256), so the reshape below splits it on a block boundary and the tensor
    // stays row-major per cell (nb1 = one head's row, nb2 = one cell); build_attn_mha's permute then
    // gives flash attention the same [head_dim, cells, heads] stride pattern the dense cache view
    // has, with the queries on ne[3], and its f32 -> f16 cast does not fire
    ggml_tensor * k_sel = qwen4exp_qsa_gather_keeps(k_cells->type)
        ? ggml_get_rows_type(ctx0, k_cells, idx_stream, k_cells->type)
        : ggml_get_rows     (ctx0, k_cells, idx_stream);
    ggml_tensor * v_sel = qwen4exp_qsa_gather_keeps(v_cells->type)
        ? ggml_get_rows_type(ctx0, v_cells, idx_stream, v_cells->type)
        : ggml_get_rows     (ctx0, v_cells, idx_stream);

    k_sel = ggml_reshape_4d(ctx0, k_sel, k_all->ne[0], k_all->ne[1], width, n_q);
    v_sel = ggml_reshape_4d(ctx0, v_sel, v_all->ne[0], v_all->ne[1], width, n_q);
    cb(k_sel, "qsa_k_sel", il);
    cb(v_sel, "qsa_v_sel", il);

    // gathering the attention mask at the selected cells leaves the same values the scan path
    // would put there, so the window carries the reach of its query
    ggml_tensor * idx_query = ggml_reshape_3d(ctx0, top_k, width, n_q, 1);

    ggml_tensor * mask = nullptr;
    ggml_tensor * kq_mask = vis ? nullptr : inp->get_kq_mask();
    if (vis) {
        // [TAG_QSA_VIS] the same f16 values straight from the metadata at the ids: [width, 1, 1, n_q], no plane
        mask = ggml_kq_mask_ids(ctx0, inp->self_kq_vis_cells, inp->self_kq_vis_toks, top_k, GGML_TYPE_F16,
                cparams.causal_attn, hparams.use_alibi, (int32_t) LLAMA_SWA_TYPE_NONE, 0);
    } else if (kq_mask->type == GGML_TYPE_F16 && qwen4exp_qsa_gather_keeps(kq_mask->type)) {
        GGML_ASSERT(kq_mask->nb[3] == kq_mask->nb[1]*n_tps);

        ggml_tensor * mask_cells = ggml_view_3d(ctx0, kq_mask, 1, n_kv, n_q,
                kq_mask->nb[0], kq_mask->nb[1], 0);

        // [TAG_QSA_GATHER_Q8] f16 cells gathered as f16: flash attention's mask type, no cast
        mask = ggml_get_rows_type(ctx0, mask_cells, idx_query, kq_mask->type);
        mask = ggml_reshape_4d(ctx0, mask, width, 1, 1, n_q);
    } else {
        GGML_ASSERT(kq_mask->nb[3] == kq_mask->nb[1]*n_tps);

        ggml_tensor * mask_cells = ggml_view_3d(ctx0, kq_mask, 1, n_kv, n_q,
                kq_mask->nb[0], kq_mask->nb[1], 0);

        mask = ggml_get_rows(ctx0, mask_cells, idx_query);
        mask = ggml_cast(ctx0, ggml_reshape_4d(ctx0, mask, width, 1, 1, n_q), GGML_TYPE_F16);
    }

    // [TAG_QSA_WHOLE_BLOCKS] the block-level selection pads a row with a repeat of its last id
    // (ggml_qsa_top_k_blocks). The scan's set_rows writes the same 0 twice, but a gathered window would
    // count the cell twice, so mask every slot j > 0 whose id equals id[j - 1]: selected ids are distinct,
    // so only padding repeats (and a cell-form row, distinct ids in any order, is left alone).
    // pad = 1 - step(|id[j] - id[j-1]|), as a bias step*BIG - BIG = 0 or -BIG, -BIG -> -inf in f16.
    if (llama_qsa_whole_blocks() && width > 1) {
        const float big = 1e30f;

        ggml_tensor * ids_f = ggml_cast(ctx0, idx_query, GGML_TYPE_F32); // [width, n_q, 1]; ids < 2^24, exact
        ggml_tensor * cur_i = ggml_view_2d(ctx0, ids_f, width - 1, n_q, ids_f->nb[1], ids_f->nb[0]);
        ggml_tensor * prv_i = ggml_view_2d(ctx0, ids_f, width - 1, n_q, ids_f->nb[1], 0);

        ggml_tensor * pad = ggml_step(ctx0, ggml_abs(ctx0, ggml_sub(ctx0, cur_i, prv_i)));  // [width - 1, n_q]
        pad = ggml_scale_bias(ctx0, pad, big, -big);

        // slot 0 is never padding
        ggml_tensor * slot0 = ggml_scale(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, ids_f, 1, n_q, ids_f->nb[1], 0)), 0.0f);
        pad = ggml_concat(ctx0, slot0, pad, 0);                                               // [width, n_q]

        mask = ggml_add(ctx0, mask, ggml_cast(ctx0, ggml_reshape_4d(ctx0, pad, width, 1, 1, n_q), GGML_TYPE_F16));
    }
    cb(mask, "qsa_mask_sel", il);

    return build_attn_mha(q_cur, k_sel, v_sel, nullptr, mask, nullptr, nullptr, kq_scale, il);
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    // [TAG_QSA_RATIO_OVERRIDE] --qsa-ratio 0 forces dense even for a GGUF that carries ratios;
    // --qsa-ratio N > 0 forces sparse wherever the indexer cache exists. get_idx() is itself
    // allocated from the same effective ratios (llama-model.cpp [TAG_QSA_DEAD_IDX_CACHE]).
    const uint32_t qsa_ratio = llama_hparams_qsa_ratio(hparams, cparams.qsa_ratio, (uint32_t) il);

    const bool qsa = mctx_hyb->get_idx() != nullptr && qsa_ratio > 0;

    // [TAG_QSA_BYPASS_EXHAUSTIVE] width == n_kv means ggml_top_k() returns every column: the
    // selection is exhaustive and QSA is the dense attention with ~40 extra nodes per layer.
    // Keep the indexer K write (cache_only), run the dense build_attn().
    bool qsa_exhaustive = false;

    if (qsa) {
        // [TAG_QSA_INDEXED_FA] read and log LLAMA_QSA_PREFILL at the first QSA layer (the reserve), whatever branch it takes
        qwen4exp_qsa_prefill_indexed();
        qwen4exp_qsa_indexed_sort(); // [TAG_QSA_INDEXED_SORT] and its sort knob, after it
        if (qwen4exp_qsa_vis() && inp->self_kq_vis_cells == nullptr) { // [TAG_QSA_VIS] and its knob
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                LLAMA_LOG_WARN("%s: LLAMA_QSA_VIS=1 without the KQ_VIS metadata inputs (LLAMA_KQ_VIS=1, n_seq_max <= 32): "
                        "the QSA layers keep the KQ mask plane\n", __func__);
            }
        }
    }

    // evaluated for every QSA layer: [TAG_QSA_WINDOW_MISALIGNED] holds whatever the bypass / LLAMA_QSA_MIN_KV knobs
    if (qsa) {
        // same expression as build_qsa_top_k()'s `width`, over the same n_kv
        const int64_t n_kv_idx = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width    = std::min<int64_t>(n_kv_idx, (int64_t) qwen4exp_qsa_top_k(hparams, il) + (int64_t) qsa_ratio - 1);

        // [TAG_QSA_MIN_KV] also while the ubatch's sequence depth is below LLAMA_QSA_MIN_KV
        qsa_exhaustive = qwen4exp_qsa_dense(mctx_hyb, ubatch, n_kv_idx, width);

        // the first QSA graph a process builds is the load-time reserve (full memory, n_kv = --kv-graph-span or the
        // cache): below LLAMA_QSA_MIN_KV it reserves the dense graph, and the first QSA ubatch regrows the buffer
        static bool warned = false;
        if (!warned) {
            warned = true;
            if (n_kv_idx < qwen4exp_qsa_min_kv()) {
                LLAMA_LOG_WARN("%s: LLAMA_QSA_MIN_KV=%lld > the reserve's indexer n_kv %lld (--kv-graph-span or the cache size):"
                        " the load-time reserve is the DENSE graph; the first QSA ubatch (n_kv >= %lld) regrows the compute buffer\n",
                        __func__, (long long) qwen4exp_qsa_min_kv(), (long long) n_kv_idx, (long long) qwen4exp_qsa_min_kv());
            }
        }
    }

    // [TAG_MTP_QSA] the MTP block: a ubatch above LLAMA_MTP_QSA_MAX_UBATCH (catch-up / prompt-size process(), and the
    // load-time reserve) is dense + the indexer K write, so every draft row still lands in the indexer cache
    // With LLAMA_MTP_CATCHUP_ATTN=0 the zero-output catch-up never gets here (graph_mtp's kv-only branch writes the
    // indexer K cache-only itself); what remains of this rule: the load-time reserve (it requests outputs, so the MTP
    // compute buffer stays the dense one), ubatches with outputs above N (a prompt-size process() that wants logits),
    // and every catch-up while LLAMA_MTP_CATCHUP_ATTN=1 / unset.
    bool mtp_forced = false;
    if (qsa && il >= (int) hparams.n_layer() && qwen4exp_mtp_qsa_max_ubatch() > 0 && n_tokens > qwen4exp_mtp_qsa_max_ubatch()) {
        mtp_forced     = true;
        qsa_exhaustive = true;
    }

    // [TAG_QSA_CHUNK] chunk a scan ubatch that is longer than the chunk; the branch is decided here,
    // once for the ubatch, with build_attn_qsa's own rule over the same n_tps, width and n_kv
    qsa_prep prep;
    bool     chunked = false;

    if (qsa && !qsa_exhaustive && qwen4exp_qsa_chunk() > 0) {
        const int64_t n_stream = mctx_hyb->get_n_stream();
        const int64_t n_tps    = n_tokens/n_stream;
        const int64_t n_kv_idx = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width    = std::min<int64_t>(n_kv_idx, (int64_t) qwen4exp_qsa_top_k(hparams, il) + (int64_t) qsa_ratio - 1);

        chunked = n_tps > qwen4exp_qsa_chunk() &&
            !qwen4exp_qsa_use_gather(cparams.flash_attn, n_tps, n_stream, width, inp->mctx->get_n_kv());
        prep.chunk = qwen4exp_qsa_chunk();
    }

    // [TAG_QSA_VIS] (A4) the ubatch's QSA consumers read the metadata, no [n_kv x n_tps] plane (qwen4exp_qsa_vis)
    const bool qsa_vis = qsa && !qsa_exhaustive && qsa_use_vis(inp, n_tokens/mctx_hyb->get_n_stream(), il);

    ggml_tensor * top_k = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il, qsa_exhaustive,
            chunked ? &prep : nullptr,
            qsa_vis ? inp->self_kq_vis_cells : nullptr, qsa_vis ? inp->self_kq_vis_toks : nullptr, mtp_forced) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (chunked) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, nullptr, kq_scale, il, &prep);
    } else if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il, nullptr, qsa_vis);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    // [TAG_MERGE_ROWS] beta and alpha read the same input: one matmul, then the beta rows and the alpha rows as views
    const auto & layer = model.layers[il];
    ggml_tensor * ba = nullptr;
    if (qwen4exp_use_merged(layer.ssm_beta_alpha_cat, layer.ssm_beta, ubatch.n_tokens, loras) && !layer.ssm_beta_s && !layer.ssm_alpha_s) {
        ba = build_lora_mm(layer.ssm_beta_alpha_cat, cur);
        cb(ba, "beta_alpha", il);
    }

    ggml_tensor * beta = nullptr;
    if (ba) {
        beta = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                ggml_element_size(ba), ba->nb[1], ba->nb[1] * n_seq_tokens, 0);
    } else {
        beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
        beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = nullptr;
    if (ba) {
        alpha = ggml_view_3d(ctx0, ba, num_v_heads, n_seq_tokens, n_seqs,
                ba->nb[1], ba->nb[1] * n_seq_tokens, ggml_row_size(ba->type, num_v_heads));
    } else {
        alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    }
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    std::vector<ggml_tensor *> conv_tails; // [TAG_CONV_INPUT_SPLIT] expanded after the conv, see build_conv_state_at
    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il, &conv_tails);

    // [TAG_CONV_INPUT_SPLIT] with the tails deferred, the conv (+ silu) and then the tails are expanded HERE, where the v59
    // graph expanded the concat-view tails - before build_rs's state nodes - so the rest of the graph keeps the v59
    // expansion order (B4 fix round 2: expanding them after build_rs moved the state rows ahead of the qkv projection and
    // reshuffled the backend's reorder sets). Knob off (no deferred tails): the v59 code path, unchanged.
    ggml_tensor * conv_output_proper = nullptr;
    ggml_tensor * conv_output_silu   = nullptr;
    if (!conv_tails.empty()) {
        conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);

        conv_output_silu = ggml_silu(ctx0, conv_output_proper);
        cb(conv_output_silu, "conv_output_silu", il);

        ggml_build_forward_expand(gf, conv_output_silu);
        for (ggml_tensor * t : conv_tails) {
            ggml_build_forward_expand(gf, t);
        }
    }

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    if (conv_tails.empty()) {
        conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);

        conv_output_silu = ggml_silu(ctx0, conv_output_proper);
        cb(conv_output_silu, "conv_output_silu", il);
    }

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // [TAG_MERGE_ROWS] z is a row view of the merged qkvz output when build_qkvz merged: take it with a view
    ggml_tensor * z_2d = ggml_is_contiguous(z) ?
        ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs) :
        ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
                ggml_row_size(z->type, head_v_dim), z->nb[1], z->nb[1] * n_seq_tokens, 0);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    ggml_tensor * moe_out =
        build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            nullptr, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        return rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * params.ubatch.n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

// [TAG_PLE_PREFETCH]
// Ask the kernel to start reading the PLE rows this ubatch is about to gather.
//
// per_layer_token_embd lives in the mmap'd GGUF and is loaded with TENSOR_READ_LAZY, so
// llama_mmap marks its byte range POSIX_MADV_RANDOM and leaves it out of MAP_POPULATE
// (src/llama-model-loader.cpp lazy_tensor_ranges -> src/llama-mmap.cpp). There is therefore no
// readahead of any kind on this path: the first touch is the CPU ggml_get_rows inside the graph,
// which faults the rows in one at a time, synchronously, with the GPU idle behind the CPU split.
// The row indices are already known here, one ubatch ahead of that gather.
//
// posix_madvise(POSIX_MADV_WILLNEED) is purely advisory: it starts asynchronous page-ins and
// changes no byte of the mapping, no protection, and no value any op reads. The gather reads the
// same file bytes either way, so the graph output is bit-identical with the prefetch on or off.
// A failure is therefore never fatal, but it must not be silent: the first non-zero return is
// logged once (see below). LLAMA_PLE_PREFETCH=0 turns it off for an A/B.
//
// ALIGNMENT (the reason this was a no-op before): madvise(2) rejects a non-page-aligned `start`
// with EINVAL. The address handed to it is `t->data + row*nb[1]` == mmap_base + tensor_file_offset
// + row*nb[1]; the mmap base is page-aligned but the GGUF tensor offset is only 32 B aligned
// (per_layer_token_embd.weight sits at file offset 536633120, 536633120 % 4096 == 3872), so
// rounding the *tensor-relative* offset left every advised address 3872 B into a page and every
// call returned EINVAL. The ranges below are therefore rounded as ABSOLUTE addresses.
static void qwen4exp_ple_prefetch(const ggml_tensor * t, const std::vector<int32_t> & rows) {
#if defined(__linux__)
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_PLE_PREFETCH");
        return s == nullptr || atoi(s) != 0;
    }();

    if (!enabled || rows.empty() || t == nullptr || t->data == nullptr || t->buffer == nullptr) {
        return;
    }

    // only a host mapping has a page cache behind it; a device buffer must be left alone
    if (!ggml_backend_buffer_is_host(t->buffer)) {
        return;
    }

    const long ps_signed = sysconf(_SC_PAGESIZE);
    if (ps_signed <= 0) {
        return;
    }

    const size_t    ps     = (size_t) ps_signed;
    const size_t    row_sz = t->nb[1];
    const size_t    n_rows = (size_t) t->ne[1];
    const uintptr_t base   = (uintptr_t) t->data;
    const uintptr_t mask   = ~(uintptr_t) (ps - 1);

    // page-aligned ABSOLUTE address range of every row, sorted, then coalesced where the ranges
    // touch. Rounding out of the tensor stays inside the same mapping: the tensor lies strictly
    // inside the mapped file, and the page that contains a valid mapped address is itself mapped
    // in full, so rounding a valid address down (or the end address up by < 4 KiB) can only name
    // pages of the same VMA -- llama_mmap maps the whole file and mmap rounds the length up to a
    // page, so even the last row's tail page exists. The advice can never name an unmapped page.
    std::vector<std::pair<uintptr_t, uintptr_t>> r;
    r.reserve(rows.size());

    for (const int32_t row : rows) {
        if (row < 0 || (size_t) row >= n_rows) {
            continue;
        }

        const uintptr_t a_beg = base + ((size_t) row)*row_sz;
        const uintptr_t a_end = a_beg + row_sz;

        r.emplace_back(a_beg & mask, (a_end + ps - 1) & mask);
    }

    if (r.empty()) {
        return;
    }

    std::sort(r.begin(), r.end());

    // a failure means the whole prefetch is dead (as it was while the rounding was relative);
    // report the first one once so it cannot go unnoticed again, then stay quiet
    static std::atomic<bool> warned{false};

    auto advise = [&](uintptr_t b, uintptr_t e) {
        const int rc = posix_madvise((void *) b, (size_t) (e - b), POSIX_MADV_WILLNEED);

        if (rc != 0 && !warned.exchange(true)) {
            LLAMA_LOG_WARN("%s: posix_madvise(%p, %zu, WILLNEED) failed: %s (%d) -- PLE prefetch inactive\n",
                    __func__, (void *) b, (size_t) (e - b), strerror(rc), rc);
        }
    };

    uintptr_t beg = r[0].first;
    uintptr_t end = r[0].second;

    for (size_t i = 1; i < r.size(); ++i) {
        if (r[i].first <= end) {
            end = std::max(end, r[i].second);
            continue;
        }

        advise(beg, end);

        beg = r[i].first;
        end = r[i].second;
    }

    advise(beg, end);
#else
    GGML_UNUSED(t);
    GGML_UNUSED(rows);
#endif
}

// [TAG_PLE_ROWID_DUMP]
// Observation only, for the off-line row-id skew census (M4). With LLAMA_PLE_DUMP=<path> set,
// append one line per set_input() call:
//   "<n_tokens> <id0> <id1> ... <id_{n_heads*n_tokens-1}>"
// the ids in exactly the layout the graph input holds them, head-minor: idx[i*n_heads + h], so a
// reader recovers n_heads as (count-1)/n_tokens and the head of an id as (position % n_heads).
// Consumed by an external row-id skew analysis (unique fraction, hot-row and 4 KiB-page
// coverage) -- no GPU and no server change needed to answer whether a packed hot-row cache pays.
//
// Cost when LLAMA_PLE_DUMP is unset: one load of a function-local static and a null test per
// ubatch. No allocation, no formatting, no syscall, no lock. The function reads only the id vector
// that has already been computed and writes only to its own file; it feeds nothing back into the
// graph, so a dumping run and a non-dumping run compute identical values.
//
// The FILE * is deliberately never fclose()d: it is a function-local static opened once and written
// until the process exits, which is the whole intended lifetime of a census run. Every line is
// fflush()ed, so a killed server still leaves a complete file. (review 0023 - noted, not changed.)
static void qwen4exp_ple_rowid_dump(const std::vector<int32_t> & idx, int64_t n_tokens) {
    static FILE * const out = [] () -> FILE * {
        const char * path = getenv("LLAMA_PLE_DUMP");

        if (path == nullptr || path[0] == '\0') {
            return nullptr;
        }

        FILE * h = fopen(path, "a");
        if (h == nullptr) {
            LLAMA_LOG_WARN("%s: cannot open LLAMA_PLE_DUMP=%s for append, PLE row-id dump disabled\n",
                    __func__, path);
        }

        return h;
    }();

    if (out == nullptr) {
        return;
    }

    // one fwrite per ubatch, under a lock: several llama_contexts can share the process
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    std::string line;
    line.reserve(12*(idx.size() + 1));

    char buf[24];

    // snprintf returns int: negative on an encoding error and >= sizeof(buf) on truncation, either
    // of which becomes a huge size_t if cast blind. Both formats fit buf comfortably, so this only
    // has to be safe, not clever: clamp to [0, sizeof(buf)-1]. (review 0023 nit)
    auto appended = [&](int n) -> size_t {
        if (n < 0) {
            return 0;
        }
        return std::min((size_t) n, sizeof(buf) - 1);
    };

    line.append(buf, appended(snprintf(buf, sizeof(buf), "%" PRId64, n_tokens)));

    for (const int32_t row : idx) {
        line.append(buf, appended(snprintf(buf, sizeof(buf), " %d", row)));
    }

    line += '\n';

    fwrite(line.data(), 1, line.size(), out);
    fflush(out);
}

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    // an image arrives as an embd batch, so ubatch->token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch->token ? ubatch->token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    std::vector<int32_t> idx(n_heads * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        std::vector<int64_t> ctx(n_gram);
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }

    // [TAG_PLE_ROWID_DUMP] off by default; a no-op unless LLAMA_PLE_DUMP names a file
    qwen4exp_ple_rowid_dump(idx, n_tokens);

    // start the page-ins before the upload, so they overlap the rest of set_inputs and the
    // graph nodes that run ahead of blk.1's gather
    qwen4exp_ple_prefetch(pmodel.per_layer_tok_embd, idx);

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il,
        std::vector<ggml_tensor *> * deferred_tails) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * x_t        = ggml_transpose(ctx0, x);
    ggml_tensor * conv_input = ggml_concat(ctx0, state, x_t, 0);

    // keep the last state_cols columns for the next ubatch
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const auto   mem_size = mctx_cur->get_size();

    // [TAG_CONV_INPUT_SPLIT] the tails are the last state_cols (+ slot) columns of the concat. When the ubatch
    // has at least state_cols + K_write - 1 tokens they all lie in its x half, so they are taken from x^T
    // directly: the same floats (the concat is a bit copy), and the concat is left with the conv as its only
    // consumer, which lets the Vulkan backend read the conv window from state | x without writing the concat
    // (concat_ssm_conv). Shorter ubatches (decode, short verify) keep the concat views, as before.
    // x is [channels, n_seq_t, n_seqs] (tokens on ne[1]); a tail of x^T is the transpose of a token-range view of
    // x (ggml_view_3d always gives nb[0] = the element size, so a view OF x^T cannot carry its token stride).
    const int64_t n_seq_t   = x->ne[1];
    const int64_t k_tails   = cparams.n_rs_seq == 0 ? 1 :
        std::min<int64_t>(ubatch.n_seq_tokens, (int64_t) ubatch.rs_rollback_depth + 1);
    // the x-tail views below assume x has unit element stride in dim 0 (the conv itself reads x through the concat, so a
    // dim-0-strided x would only corrupt the history): otherwise keep the concat-view tails
    // only for a caller that defers the tails past its conv (the GDN layer): the PLE layer keeps its concat (per-tap views
    // read it), so x-tails there would only move the PLE CONCAT away from its v59 position (review-B4c I2)
    const bool    tails_x   = deferred_tails != nullptr && qwen4exp_conv_input_split() && n_seq_t >= state_cols + k_tails - 1 &&
                              x->nb[0] == ggml_type_size(x->type) && !ggml_is_quantized(x->type);
    // the [state_cols, channels, n_seqs] tail `back` columns before the end of the concat (back = state_cols + slot)
    auto const & tail_view = [&](int64_t back) -> ggml_tensor * {
        if (tails_x) {
            return ggml_transpose(ctx0, ggml_view_3d(ctx0, x,
                    channels, state_cols, n_seqs,
                    x->nb[1], x->nb[2],
                    (n_seq_t - back) * x->nb[1]));
        }
        return ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, conv_input->ne[0] - back));
    };
    // A caller that reads the concat with a conv takes the x-tail copies and expands them AFTER the conv:
    // that keeps x alive across the conv, so ggml-alloc cannot hand the conv's output the block x frees -
    // which would make the conv read x while it is being overwritten (the backend's overlap veto would
    // then undo concat_ssm_conv). Pure ordering: the history row is read by build_rs's gather before.
    // B4 fix round 4 (gfx1151 interim run4: vetoed:concat_ssm_conv_silu in some wide graphs): the history rows are a
    // GET_ROWS result in the compute buffer whose only consumer is the CONCAT, so ggml-alloc freed them right after the
    // concat and the conv output could take their block - while the fused conv still reads them (overlap veto). Keep
    // them alive for the graph (3 x channels floats per sequence and layer) whenever the concat is fused-capable.
    if (tails_x && deferred_tails) {
        ggml_set_keep_alive(rows);
    }
    auto const & emit_tail = [&](ggml_tensor * cpy) {
        if (tails_x && deferred_tails) {
            deferred_tails->push_back(cpy);
        } else {
            ggml_build_forward_expand(gf, cpy);
        }
    };

    if (cparams.n_rs_seq == 0) {
        // no rollback capacity: a single slot-0 write, unchanged from before [TAG_RECURRENT_ROLLBACK_SPLITS]
        ggml_tensor * tail = tail_view(state_cols);

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                kv_head * row_size);

        emit_tail(ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
    } else {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // slot 0 is the final state, slot s the state s tokens back, as llm_build_delta_net_base::build_conv_state
        //   writes them; the rollback reads slot d after set_rs_idx(d). Only the K_write planes written here are
        //   valid, which is what the memory records as this sequence's rollback depth after the ubatch.
        // The strided view is copied straight into the destination plane (as build_conv_state does): a
        //   ggml_cont() per plane would materialise K extra state_cols*channels*n_seqs temporaries in every
        //   recurrent layer of every graph, which is pure per-decode overhead.
        const int64_t K_write = std::min<int64_t>(ubatch.n_seq_tokens, (int64_t) ubatch.rs_rollback_depth + 1);

        for (int64_t slot = 0; slot < K_write; ++slot) {
            ggml_tensor * tail = tail_view(state_cols + slot);

            ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                    state_cols * channels, n_seqs,
                    conv_states_all->nb[1],
                    ((size_t) slot * mem_size + kv_head) * row_size);

            emit_tail(ggml_cpy(ctx0, tail, dst));
        }
    }

    return conv_input;
}

// [TAG_PLE_HOIST] the gather, verbatim from build_ple. Called either from the graph ctor (hoisted,
// the default) or from build_ple itself (LLAMA_PLE_HOIST=0). Identical either way -- the only
// difference is the position of these two nodes in gf.
ggml_tensor * llama_model_qwen4exp::graph::build_ple_gather(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        int                  il) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
    ggml_set_input(ple_inp->rows);
    ggml_tensor * rows = ple_inp->rows;
    res->add_input(std::move(ple_inp));

    // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
    ggml_tensor * emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    cb(emb, "ple_embd", il);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    // [TAG_PLE_HOIST] normally the gather is already built and expanded at the head of the graph;
    // with LLAMA_PLE_HOIST=0 the map is empty and it is built right here, as it was before.
    const auto it = ple_gathered.find(il);
    ggml_tensor * emb = it != ple_gathered.end() ? it->second : build_ple_gather(mctx_hyb, il);

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a [n_embd, hc] weight ([TAG_UP28896]: RMS_NORM + MUL adjacent)
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        return ggml_mul(ctx0, ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps), w);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
