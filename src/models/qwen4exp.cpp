#include "models.h"
#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
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
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { hc_dim }, trunk_flags);
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

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

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
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { hc_dim }, flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { hc_dim }, flags);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { hc_dim }, flags);
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
        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { hc_dim }, flags | TENSOR_NOT_REQUIRED);
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
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: reduce over one stream, then scale all streams with the [hc_dim] gamma
    // the converter folded each gamma to (1 + w)
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    xn = ggml_mul(ctx0, xn, w_norm);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
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
        *inject = build_lora_mm(w_inject, xn);
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
        return (int64_t) (v <= 0 ? 0 : std::max<long long>(v, 64));
    }();

    return chunk;
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
    return flash_attn && 4*n_tps*width < n_kv;
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
            nullptr, nullptr, -1);

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
// TODO: wire up QSA here for long-context draft fidelity.
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

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

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
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    // ---- dense attention, mirroring the trunk's full-attention branch ----
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    // IMRoPE, same convention and freq_base as the trunk
    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "mtp_Qcur", il);
    cb(Kcur, "mtp_Kcur", il);

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
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
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
            nullptr, nullptr, -1);
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
            mctx->set_input_qsa_store(cell_blk, bias, dirty_cells, dirty_pos, dirty_dst, remap, ubatch, ratio, blk_bias);
            return;
        }

        mctx->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
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

        if (cache_only) {
            // the bypass decision itself depends on n_kv, so a graph built for the exhaustive
            // regime must not be reused once the window has grown past the budget
            res &= (int64_t) std::min<int64_t>(n_kv, (int64_t) bypass_top_k + (int64_t) ratio - 1) == n_kv;

            return res;
        }

        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        res &= bias->ne[0]      == (blk_bias ? n_blocks : n_kv);
        res &= bias->ne[1]      == params.ubatch.n_tokens/n_stream;

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
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;

    // [TAG_QSA_BYPASS_EXHAUSTIVE] only k_idxs is live; cell_blk/blk_cells/blk_pos/bias are nullptr
    const bool cache_only;

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
        qsa_prep *                              prep) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

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

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;

        // every layer sharing a ratio also shares the exhaustive/not decision (it is a function of
        // n_kv and r alone), so a cached input set can never be the wrong kind
        GGML_ASSERT(inp->cache_only == cache_only);
    } else {
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias, cache_only);

        qsa->k_idxs        = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        qsa->bypass_top_k  = hparams.indexer_top_k;

        // [TAG_QSA_BYPASS_EXHAUSTIVE] cache-only needs the K write and nothing else: skipping the
        // four layout inputs also drops their device buffers and their host-side fill
        if (!cache_only && pooled_cache) {
            // [TAG_QSA_POOLED_CACHE] the pooling inputs are the dirty list, sized by this ubatch's plan
            const int64_t cap = mctx_hyb->qsa_store_plan(ubatch, (uint32_t) r, n_kv, blk_bias);

            qsa->pooled_cache = true;

            qsa->cell_blk    = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
            qsa->bias        = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);
            qsa->dirty_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*cap, n_stream);
            qsa->dirty_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*cap*n_stream);

            ggml_set_name(qsa->cell_blk,    "qsa_cell_blk");
            ggml_set_name(qsa->bias,        "qsa_bias");
            ggml_set_name(qsa->dirty_cells, "qsa_dirty_cells");
            ggml_set_name(qsa->dirty_pos,   "qsa_dirty_pos");

            ggml_set_input(qsa->cell_blk);
            ggml_set_input(qsa->bias);
            ggml_set_input(qsa->dirty_cells);
            ggml_set_input(qsa->dirty_pos);

            if (cap < n_blocks) {
                qsa->dirty_dst = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cap*n_stream);
                qsa->remap     = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_blocks*n_stream);

                ggml_set_name (qsa->dirty_dst, "qsa_dirty_dst");
                ggml_set_name (qsa->remap,     "qsa_remap");
                ggml_set_input(qsa->dirty_dst);
                ggml_set_input(qsa->remap);
            } else {
                // the full pool (a cold rebuild, and every reserve graph): the recompute's leaves exactly
                qsa->full_s0 = mctx_hyb->qsa_store_s0();
            }
        } else if (!cache_only) {
            qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
            qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
            qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
            qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

            ggml_set_input(qsa->cell_blk);
            ggml_set_input(qsa->blk_cells);
            ggml_set_input(qsa->blk_pos);
            ggml_set_input(qsa->bias);
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

            pooled = ggml_get_rows(ctx0, st, inp->remap);
            pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, n_stream);
        }
    } else {
        pooled = build_qsa_pool(k_all, inp->blk_cells, inp->blk_pos, n_blocks, n_stream, sections,
                "indexer_k_pooled", il);
    }
    cb(pooled, "indexer_k", il);

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
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    // [TAG_QSA_CHUNK] the query-independent half is done; the caller runs the selection one query
    // chunk at a time, next to each chunk's attention (build_qsa_chunked)
    if (prep) {
        prep->inp      = inp;
        prep->pooled   = pooled;
        prep->q        = q;
        prep->kq_mask  = kq_mask;
        prep->width    = width;
        prep->blk_bias = blk_bias;
        return nullptr;
    }

    return build_qsa_select(inp, pooled, q, inp->bias, kq_mask, width, blk_bias, il);
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
        int                   il) {
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t n_blocks = pooled->ne[1];
    const int64_t n_kv     = inp->cell_blk->ne[0];
    const int64_t n_tps    = q->ne[1]/n_idx_h;
    const int64_t n_stream = q->ne[2];

    // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
    ggml_tensor * score = ggml_mul_mat(ctx0, pooled, q);
    score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_tps, n_stream);
    score = ggml_relu(ctx0, score);

    // the heads sit side by side on ne[1] and there are few of them, so summing slices beats a
    // transpose that would carry the whole block by token surface twice over
    ggml_tensor * summed = nullptr;
    for (int64_t h = 0; h < n_idx_h; ++h) {
        ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, n_tps, n_stream,
                score->nb[2], score->nb[3], h*score->nb[1]);
        summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
    }

    score = summed;
    cb(score, "indexer_score", il);

    // one value per block, so it is cheaper to bias here than after the cells are expanded
    if (blk_bias) {
        score = ggml_add(ctx0, score, bias);
    }

    // [TAG_QSA_FUSED_SELECT] the single-node form. Gated on exactly what the Vulkan topk_qsa
    // pattern needs (blk_bias, i.e. the per-block bias and a matching mask, and an f16 mask - the
    // shader's binding 3 is float16_t and the pattern likewise only matches through ggml_cast), so
    // no configuration that fuses today falls off the fused path, and none that does not gets a
    // node the backend would have to run somewhere else.
    if (blk_bias && kq_mask->type == GGML_TYPE_F16 && kq_mask->ne[2] == 1 && qwen4exp_qsa_fused_select()) {
        // one node, [width, n_tps, 1, n_stream] i32 - already the shape build_attn_qsa wants, so
        // neither the ggml_cont nor the reshape below is needed either.
        // Note for the node-diff probes: this arm has no "indexer_score_tokens-<il>" node, because
        // the surface that name refers to is the one the op exists not to materialize. The
        // selection keeps the name "indexer_top_k-<il>", so a diff of two runs on the SAME side of
        // the knob still pairs it. It does NOT pair ACROSS the knob: fa_rb_nodediff's key is
        // name|op|type|ne (scripts/bench/probe/fa_rb_nodediff.cpp:281-290) and the op differs -
        // RESHAPE below, QSA_TOP_K here. Cross-arm equality is established by the corpus census and
        // by qsa_fusion_replay's REPLAY_QSA_OP_SELFTEST, not by the node diff.
        ggml_tensor * top_k = ggml_qsa_top_k(ctx0, score, inp->cell_blk, kq_mask, (int) width);
        cb(top_k, "indexer_top_k", il);

        return top_k;
    }

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
        const qsa_prep *          prep) {
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
    ggml_tensor * cur = qwen4exp_qsa_use_gather(cparams.flash_attn, n_tps, top_k->ne[3], width, n_kv)
        ? build_qsa_gather(inp, q_cur, top_k, kq_scale, il)
        : build_qsa_scan  (inp, q_cur, top_k, kq_scale, il);
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
    ggml_tensor * bias    = prep.inp->bias; // [n_blocks or n_kv, n_tps, n_stream]

    // one zero source per layer, sized for the longest chunk; each chunk reads a view of it
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, width, n_max, n_stream);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

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

        // [n_blocks or n_kv, np, n_stream] f32
        ggml_tensor * bias_c = ggml_get_rows(ctx0, bias, rows);

        // [n_kv, np, 1, n_stream], back in the mask's own type
        ggml_tensor * mask_c = nullptr;
        if (kq_mask->type == GGML_TYPE_F16 && qwen4exp_qsa_gather_keeps(kq_mask->type)) {
            // [TAG_QSA_GATHER_Q8] gathered as f16 rows, no f32 transient and no cast
            mask_c = ggml_get_rows_type(ctx0, kq_mask, ggml_reshape_3d(ctx0, rows, np, 1, n_stream), kq_mask->type);
        } else {
            mask_c = ggml_get_rows(ctx0, kq_mask, ggml_reshape_3d(ctx0, rows, np, 1, n_stream));
            if (kq_mask->type != GGML_TYPE_F32) {
                mask_c = ggml_cast(ctx0, mask_c, kq_mask->type);
            }
        }

        ggml_tensor * top_k = build_qsa_select(prep.inp, prep.pooled, q_idx, bias_c, mask_c, width, prep.blk_bias, il);

        if (np > nc) {
            // [TAG_QSA_CHUNK_PAD] THE CUT: from here on only the nc real queries exist - the ids and the mask rows
            // the scan reads are views of their first nc columns, so the padding ids reach no mask, no attention
            // and no output. The padded selection is renamed so LLAMA_QSA_DUMP_IDS hashes the cut ids instead.
            ggml_format_name(top_k, "indexer_top_k_pad-%d", il);
            top_k = ggml_view_4d(ctx0, top_k, width, nc, 1, n_stream, top_k->nb[1], top_k->nb[2], top_k->nb[3], 0);
            mask_c = ggml_view_4d(ctx0, mask_c, mask_c->ne[0], nc, 1, n_stream, mask_c->nb[1], mask_c->nb[2], mask_c->nb[3], 0);
            if (!ggml_is_contiguous(top_k)) {
                // more than one stream: the real columns of each stream are apart. The mask cont is an
                // [n_kv, nc, 1, n_stream] f16 transient next to the padded rows. Measured (M4.3 fix round 1, reserve
                // replay, --parallel 5 without --kv-unified, ub 2048, span 196 608: 2 x 205 -> 256 + 154 padded per
                // stream): the knob costs +379.29 MiB there (2378.59 -> 2757.88), this cont ([196608, 154, 5] f16 =
                // 288.8 MiB) being most of it. With a unified cache (n_stream = 1, deployed) both are views: +3.09 MiB.
                top_k  = ggml_cont(ctx0, top_k);
                mask_c = ggml_cont(ctx0, mask_c);
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

        ggml_tensor * zeros_c = ggml_view_4d(ctx0, zeros, 1, width, nc, n_stream,
                zeros->nb[1], zeros->nb[2], zeros->nb[3], 0);

        // [n_embd_head_v*n_head, nc*n_stream], stream-major
        ggml_tensor * o = build_qsa_scan(inp, q_att, top_k, kq_scale, il, mask_c, zeros_c);
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
        int                       il) {
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
    ggml_tensor * kq_mask = inp->get_kq_mask();

    GGML_ASSERT(kq_mask->nb[3] == kq_mask->nb[1]*n_tps);

    ggml_tensor * mask_cells = ggml_view_3d(ctx0, kq_mask, 1, n_kv, n_q,
            kq_mask->nb[0], kq_mask->nb[1], 0);

    ggml_tensor * idx_query = ggml_reshape_3d(ctx0, top_k, width, n_q, 1);

    ggml_tensor * mask = nullptr;
    if (kq_mask->type == GGML_TYPE_F16 && qwen4exp_qsa_gather_keeps(kq_mask->type)) {
        // [TAG_QSA_GATHER_Q8] f16 cells gathered as f16: flash attention's mask type, no cast
        mask = ggml_get_rows_type(ctx0, mask_cells, idx_query, kq_mask->type);
        mask = ggml_reshape_4d(ctx0, mask, width, 1, 1, n_q);
    } else {
        mask = ggml_get_rows(ctx0, mask_cells, idx_query);
        mask = ggml_cast(ctx0, ggml_reshape_4d(ctx0, mask, width, 1, 1, n_q), GGML_TYPE_F16);
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

    if (qsa && qwen4exp_qsa_bypass_exhaustive()) {
        // same expression as build_qsa_top_k()'s `width`, over the same n_kv
        const int64_t n_kv_idx = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width    = std::min<int64_t>(n_kv_idx, (int64_t) hparams.indexer_top_k + (int64_t) qsa_ratio - 1);

        qsa_exhaustive = width == n_kv_idx;
    }

    // [TAG_QSA_CHUNK] chunk a scan ubatch that is longer than the chunk; the branch is decided here,
    // once for the ubatch, with build_attn_qsa's own rule over the same n_tps, width and n_kv
    qsa_prep prep;
    bool     chunked = false;

    if (qsa && !qsa_exhaustive && qwen4exp_qsa_chunk() > 0) {
        const int64_t n_stream = mctx_hyb->get_n_stream();
        const int64_t n_tps    = n_tokens/n_stream;
        const int64_t n_kv_idx = mctx_hyb->get_idx()->get_n_kv();
        const int64_t width    = std::min<int64_t>(n_kv_idx, (int64_t) hparams.indexer_top_k + (int64_t) qsa_ratio - 1);

        chunked = n_tps > qwen4exp_qsa_chunk() &&
            !qwen4exp_qsa_use_gather(cparams.flash_attn, n_tps, n_stream, width, inp->mctx->get_n_kv());
        prep.chunk = qwen4exp_qsa_chunk();
    }

    ggml_tensor * top_k = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il, qsa_exhaustive,
            chunked ? &prep : nullptr) : nullptr;

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
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
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

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
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

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

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

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

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
// Consumed by scripts/bench/probe/ple_rowid_skew.py (unique fraction, hot-row and 4 KiB-page
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
        int                  il) {
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

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // keep the last state_cols columns for the next ubatch
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const auto   mem_size = mctx_cur->get_size();

    if (cparams.n_rs_seq == 0) {
        // no rollback capacity: a single slot-0 write, unchanged from before [TAG_RECURRENT_ROLLBACK_SPLITS]
        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, conv_input->ne[0] - state_cols));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                kv_head * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, tail), dst));
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
            ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                    state_cols, channels, n_seqs,
                    conv_input->nb[1], conv_input->nb[2],
                    ggml_row_size(conv_input->type, conv_input->ne[0] - state_cols - slot));

            ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                    state_cols * channels, n_seqs,
                    conv_states_all->nb[1],
                    ((size_t) slot * mem_size + kv_head) * row_size);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
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

    // both norms group over one hc stream, with a weight over the whole hc*n_embd layout
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
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
