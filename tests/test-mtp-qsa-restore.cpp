// [TAG_QSA_WINDOW_MISALIGNED] the sparse MTP draft after a legacy /slots restore (v50 gate parity-exh crash).
//
// The legacy restore (server-context.cpp [TAG_PARTIAL_PROMPT]) loads the target state only: it drops the whole draft
// sequence and the drafter's carry and never rebuilds the draft KV for the restored prefix. The kv-only catch-up then
// writes the new rows at positions >= the restored depth L into cells 0..T-1, so a QSA census over the draft's cell
// window finds cells whose block index (pos - phase)/r >= n_blocks = ceil(n_kv/r) and aborts on
// GGML_ASSERT((!blk_bias || !oor) ...) (llama-memory-hybrid-idx.cpp) at the first draft step. The fix
// (qwen4exp_qsa_dense: depth > n_kv) runs such a ubatch dense.
//
// A CPU qwen4exp with random weights + 1 MTP block (the test-mtp-catchup fixture), LLAMA_MTP_QSA=1 (+ the prod QSA env
// from tests/CMakeLists.txt), kv-only catch-up (LLAMA_MTP_CATCHUP_ATTN=0):
//  restore:    target at depth L = 640, the draft sequence + carry dropped as the server does, a catch-up of T = 32
//              rows, then 12 draft steps (each followed by a 1-token target decode + catch-up) -> no abort, and the
//              draft graphs run no top-k selection (dense; draft cell window 256 < depth ~ 690);
//  no restore: the same without the drop -> the draft steps run the QSA selection (TOP_K / QSA_TOP_K in their graphs);
//  target exact: both scenarios' target logits (every row) are byte-identical to a child run of this binary with
//              LLAMA_MTP_QSA=0 (dense draft memory).
//  unified:    (Codex v50 review, Major 2) the depth guard only covers the ubatch's own sequence, the census walks every
//              cell of the stream. Two sequences in one --kv-unified pool: A restored at depth L (draft dropped, T
//              catch-up rows at positions >= L, i.e. block index >= n_blocks of the draft window), then B (a short
//              prompt, depth < the window, so its draft takes the sparse path) drafts N_STEPS times while A's high cells
//              are present -> no abort (before the census exclusion: GGML_ASSERT "cell position runs past the cell
//              window"), B's draft graphs run the QSA selection (one top-k node per graph more than A's last draft,
//              which runs dense), target logits == the LLAMA_MTP_QSA=0 child.
//  descriptor: (series D1) children of this binary print llama_state_seq_descriptor of the target and the MTP draft
//              context under LLAMA_MTP_QSA=0 / =1 / =1 + LLAMA_MTP_QSA_PHASE=1: the draft descriptors are pairwise
//              different (dense kv{...} vs hybrid_idx{recr=none,attn,idx(K-only q8_0 indexer)[,qsa_phase=1]}), the
//              target descriptor is the same in all three - so a dense-draft disk entry is a draft-descriptor mismatch
//              for the sparse context (target-only adoption), and vice versa.
//  census:     the same pool layout straight into llama_qsa_census_full (blk_bias): every out-of-window cell of A is
//              routed to the second unused id n_bid + 1 (block table vis 0 = -inf for every query), no compact block
//              holds an A cell, cs.oor_seqs == {A}.
//
// usage: test-mtp-qsa-restore [--ref]   (--ref: the child mode, writes the target logits to stdout)

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-kv-cells.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

static const uint32_t N_VOCAB = 128;

static const llama_pos L_RESTORED = 640; // the restored target depth (> the 256-cell draft window)
static const int       T_CATCHUP  = 32;  // rows caught up after the restore
static const int       N_STEPS    = 12;  // draft steps

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-1f);

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

// qwen4exp: the test-mtp-catchup fixture (hc 4, QSA ratios 4, indexer top_k 8, MoE 2 experts, 1 NextN/MTP block)
static gguf_context_ptr make_meta(uint32_t n_ctx_train) {
    const llm_arch arch = LLM_ARCH_QWEN4EXP;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());

    const uint32_t n_embd = 256, n_head = 2, n_ff = 384, n_layer_trunk = 2, n_nextn = 1;
    const uint32_t n_layer_all = n_layer_trunk + n_nextn;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              n_ctx_train);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer_all);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,        n_nextn);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,     uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,     std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,      uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK,   uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,   std::vector<uint32_t>(n_layer_all, 4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        n_ff);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);
    ms.add_kv(LLM_KV_EXPERT_COUNT,                uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,           uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,         uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,          uint32_t(2)); // sigmoid
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,        1.0f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,             uint32_t(2));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");
    return ret;
}

// the ops of the draft context's graphs (scheduler eval callback, ask phase, while active)
struct op_census {
    bool active = false;
    std::map<std::string, int> n;
};

static bool census_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * c = (op_census *) ud;
    if (ask && c->active) {
        c->n[ggml_op_desc(t)]++;
    }
    return false;
}

struct mtp_env {
    llama_model_ptr   model;
    llama_context_ptr ctx_tgt;
    llama_context_ptr ctx_dft;
    common_params_speculative params;
    common_speculative * spec = nullptr;
    op_census census;

    ~mtp_env() { common_speculative_free(spec); }
};

static bool silent_progress(float, void *) { return true; }

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN || strstr(text, "[qsa-history]") != nullptr || // [TAG_MULTISLOT] the proof lines
        strstr(text, "qsa_history_dense") != nullptr) {
        fputs(text, stderr);
    }
}

static void init_env(mtp_env & env, gguf_context * meta, uint32_t n_seq = 1, bool unified = true) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    mp.load_mtp          = true;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;

    size_t seed = 31;
    env.model.reset(llama_model_init_from_user(meta, set_tensor_data, &seed, mp));
    GGML_ASSERT(env.model && "model init failed");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = unified ? 1024 : 1024*n_seq; // [TAG_MULTISLOT] per-stream KV: 1024 cells per stream
    cp.n_batch         = 128;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = n_seq;
    cp.kv_unified      = n_seq > 1 && unified;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    env.ctx_tgt.reset(llama_init_from_model(env.model.get(), cp));
    GGML_ASSERT(env.ctx_tgt && "target context failed");

    llama_context_params cd = cp;
    cd.ctx_type          = LLAMA_CONTEXT_TYPE_MTP;
    cd.n_rs_seq          = 0;
    cd.type_k            = GGML_TYPE_F16;
    cd.type_v            = GGML_TYPE_F16;
    cd.cb_eval           = census_cb;
    cd.cb_eval_user_data = &env.census;
    common_set_env("LLAMA_MTP_CATCHUP_ATTN", "0");   // the prod kv-only catch-up
    env.ctx_dft.reset(llama_init_from_model(env.model.get(), cd));
    common_set_env("LLAMA_MTP_CATCHUP_ATTN", "");
    GGML_ASSERT(env.ctx_dft && "MTP context failed");

    env.params.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    env.params.draft.n_max   = 4;
    env.params.draft.n_min   = 0;
    env.params.draft.p_min   = 0.0f;
    env.params.draft.ctx_tgt = env.ctx_tgt.get();
    env.params.draft.ctx_dft = env.ctx_dft.get();
    env.spec = common_speculative_init(env.params, n_seq);
    GGML_ASSERT(env.spec && "draft-mtp init failed");
}

static std::vector<llama_token> tokens(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<llama_token> dis(0, N_VOCAB - 1);
    std::vector<llama_token> t(n);
    for (auto & x : t) {
        x = dis(gen);
    }
    return t;
}

// target decode (every row an output) + the MTP catch-up; appends the target logits to `out`
static void feed(mtp_env & env, const std::vector<llama_token> & toks, llama_pos pos0, std::vector<float> & out,
                 llama_seq_id seq = 0) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], pos0 + (llama_pos) i, { seq }, true);
    }
    GGML_ASSERT(llama_decode(env.ctx_tgt.get(), b) == 0);
    for (size_t i = 0; i < toks.size(); ++i) {
        const float * l = llama_get_logits_ith(env.ctx_tgt.get(), (int32_t) i);
        out.insert(out.end(), l, l + N_VOCAB);
    }
    GGML_ASSERT(common_speculative_process(env.spec, b));
    llama_synchronize(env.ctx_dft.get());
    llama_batch_free(b);
}

static int argmax(const float * l) {
    return (int) (std::max_element(l, l + N_VOCAB) - l);
}

struct run_result {
    std::vector<float> logits;    // the target logits, every row of every decode
    int  n_sel      = 0;          // QSA selection top-k nodes over the draft steps' graphs (beyond the MoE router)
    int  n_graphs   = 0;          // draft graphs computed over the steps
    int  n_drafted  = 0;          // drafted tokens over the steps
    std::vector<llama_token> dtoks;   // [TAG_MULTISLOT] the draft proposals
    std::vector<float>       dlogits; // the last draft logits per step
};

// fill the target to L (+ catch-up), optionally drop the draft sequence + carry (the legacy restore), catch up T rows,
// then N_STEPS rounds of: draft -> 1-token target decode + catch-up (0 accepted, no rollback needed)
static run_result run_scenario(gguf_context * meta, bool restore) {
    run_result res;
    mtp_env E;
    init_env(E, meta);
    common_speculative_begin(E.spec, 0, {});

    llama_pos pos = 0;
    for (uint32_t c = 0; pos < L_RESTORED; ++c) {
        const auto toks = tokens(std::min<llama_pos>(128, L_RESTORED - pos), 100 + c);
        feed(E, toks, pos, res.logits);
        pos += (llama_pos) toks.size();
    }

    if (restore) {
        // server-context.cpp [TAG_PARTIAL_PROMPT]: the snapshot holds the target state only
        llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), 0, -1, -1);
        common_speculative_reset_state(E.spec, 0);
    }

    feed(E, tokens(T_CATCHUP, 7), pos, res.logits);
    pos += T_CATCHUP;

    llama_token id_last = (llama_token) argmax(res.logits.data() + res.logits.size() - N_VOCAB);

    for (int s = 0; s < N_STEPS; ++s) {
        std::vector<llama_token> drafted;
        auto & dp = common_speculative_get_draft_params(E.spec, 0);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.n_past   = pos;
        dp.id_last  = id_last;
        dp.prompt   = nullptr;
        dp.result   = &drafted;
        E.census.active = true;
        common_speculative_draft(E.spec);
        llama_synchronize(E.ctx_dft.get());
        E.census.active = false;
        if (!drafted.empty()) {
            const float * l = llama_get_logits_ith(E.ctx_dft.get(), -1);
            res.dlogits.insert(res.dlogits.end(), l, l + N_VOCAB);
        }
        res.dtoks.insert(res.dtoks.end(), drafted.begin(), drafted.end());
        llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), 0, pos, -1);
        res.n_drafted += (int) drafted.size();

        feed(E, { id_last }, pos, res.logits);
        common_speculative_accept(E.spec, 0, 0);
        pos    += 1;
        id_last = (llama_token) argmax(res.logits.data() + res.logits.size() - N_VOCAB);
    }

    // the MoE router is one TOP_K per draft graph (one attention layer: one FLASH_ATTN_EXT per graph); the QSA
    // selection adds its own top-k nodes on top of it
    res.n_graphs = E.census.n["FLASH_ATTN_EXT"];
    res.n_sel    = E.census.n["TOP_K"] + E.census.n["QSA_TOP_K"] - res.n_graphs;
    return res;
}

// one draft step of sequence `seq` at n_past `pos` (census active), the draft rows rolled back, then the 1-token target
// decode + catch-up of id_last; returns the drafted count
static int draft_step(mtp_env & E, llama_seq_id seq, llama_pos & pos, llama_token & id_last, std::vector<float> & logits,
                      std::vector<llama_token> * dtoks = nullptr, std::vector<float> * dlogits = nullptr) {
    std::vector<llama_token> drafted;
    auto & dp = common_speculative_get_draft_params(E.spec, seq);
    dp.drafting = true;
    dp.n_max    = -1;
    dp.n_past   = pos;
    dp.id_last  = id_last;
    dp.prompt   = nullptr;
    dp.result   = &drafted;
    E.census.active = true;
    common_speculative_draft(E.spec);
    llama_synchronize(E.ctx_dft.get());
    E.census.active = false;
    if (dtoks) { // [TAG_MULTISLOT] the proposals + the last draft logits
        dtoks->insert(dtoks->end(), drafted.begin(), drafted.end());
        const float * l = drafted.empty() ? nullptr : llama_get_logits_ith(E.ctx_dft.get(), -1);
        if (l != nullptr) {
            dlogits->insert(dlogits->end(), l, l + N_VOCAB);
        }
    }
    llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), seq, pos, -1);

    feed(E, { id_last }, pos, logits, seq);
    common_speculative_accept(E.spec, seq, 0);
    pos    += 1;
    id_last = (llama_token) argmax(logits.data() + logits.size() - N_VOCAB);
    return (int) drafted.size();
}

static const llama_pos B_PROMPT = 96; // B's prompt: depth < the 256-cell draft window -> B's draft takes the sparse path

struct unified_result {
    std::vector<float> logits;    // the target logits of A and B, in decode order
    int n_topk_b    = 0;          // top-k nodes (TOP_K + QSA_TOP_K) over B's draft graphs
    int n_graphs_b  = 0;
    int n_drafted_b = 0;
    int n_topk_a    = 0;          // over A's final draft step (depth > the window: dense)
    int n_graphs_a  = 0;
};

// two sequences in one unified pool: A (seq 0) restored at L (draft dropped, T catch-up rows at positions >= L), then
// B (seq 1) prompted and drafting N_STEPS times next to A's out-of-window draft cells, then one draft step of A
static unified_result run_unified(gguf_context * meta) {
    unified_result res;
    mtp_env E;
    init_env(E, meta, 2);
    common_speculative_begin(E.spec, 0, {});
    common_speculative_begin(E.spec, 1, {});

    llama_pos pos_a = 0;
    for (uint32_t c = 0; pos_a < L_RESTORED; ++c) {
        const auto toks = tokens(std::min<llama_pos>(128, L_RESTORED - pos_a), 100 + c);
        feed(E, toks, pos_a, res.logits, 0);
        pos_a += (llama_pos) toks.size();
    }

    // A: the legacy restore drops the draft sequence + carry; the catch-up writes positions >= L into the freed cells
    llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), 0, -1, -1);
    common_speculative_reset_state(E.spec, 0);
    feed(E, tokens(T_CATCHUP, 7), pos_a, res.logits, 0);
    pos_a += T_CATCHUP;
    llama_token id_a = (llama_token) argmax(res.logits.data() + res.logits.size() - N_VOCAB);

    // B: a short prompt in the same pool
    llama_pos pos_b = 0;
    feed(E, tokens(B_PROMPT, 55), pos_b, res.logits, 1);
    pos_b += B_PROMPT;
    llama_token id_b = (llama_token) argmax(res.logits.data() + res.logits.size() - N_VOCAB);

    E.census.n.clear();
    for (int s = 0; s < N_STEPS; ++s) {
        res.n_drafted_b += draft_step(E, 1, pos_b, id_b, res.logits);
    }
    res.n_graphs_b = E.census.n["FLASH_ATTN_EXT"];
    res.n_topk_b   = E.census.n["TOP_K"] + E.census.n["QSA_TOP_K"];

    E.census.n.clear();
    draft_step(E, 0, pos_a, id_a, res.logits);
    res.n_graphs_a = E.census.n["FLASH_ATTN_EXT"];
    res.n_topk_a   = E.census.n["TOP_K"] + E.census.n["QSA_TOP_K"];
    return res;
}

// [TAG_MULTISLOT] stage 5.1 item 3, partial-history dense fallback by sequence coverage. A (seq 0, or 1 = reversed ids)
// holds a target history of depth A_DEPTH and a PARTIAL draft history: the target-only restore (draft sequence + carry
// dropped, the catch-up at positions >= A_DEPTH) or an interior hole (draft positions 40..47 removed). Beside it B, deep
// and complete (L_RESTORED), keeps the unified pool's high-water above A's depth - `depth > n_kv` misses A there.
// solo: the same A in a one-sequence context. b_active: B drafts + decodes between A's steps.
static const llama_pos A_DEPTH = 96;

struct partial_result {
    std::vector<float>       logits_a;   // A's target logits
    std::vector<llama_token> dtoks_a;    // A's draft proposals
    std::vector<float>       dlogits_a;  // A's last draft logits per step
    std::vector<uint8_t>     carry_a;
    int n_sel_a    = 0;                  // QSA selection nodes over A's draft graphs: QSA_BIAS (compact bias, the
                                         //   ctest env) + QSA_TOP_K - an n_seq 2 graph has one more TOP_K whatever the
                                         //   route, so the router-subtracted TOP_K count of run_scenario does not apply
    int n_graphs_a = 0;
};

static partial_result run_partial(gguf_context * meta, bool solo, bool b_active, bool reversed, bool hole) {
    partial_result res;
    mtp_env E;
    init_env(E, meta, solo ? 1 : 2);
    const llama_seq_id A = solo || !reversed ? 0 : 1;
    const llama_seq_id B = 1 - A;
    std::vector<float> scratch;
    llama_pos   pos_b = 0;
    llama_token id_b  = 0;
    if (!solo) {
        common_speculative_begin(E.spec, B, {});
        for (uint32_t c = 0; pos_b < L_RESTORED; ++c) {
            const auto toks = tokens(std::min<llama_pos>(128, L_RESTORED - pos_b), 300 + c);
            feed(E, toks, pos_b, scratch, B);
            pos_b += (llama_pos) toks.size();
        }
        id_b = (llama_token) argmax(scratch.data() + scratch.size() - N_VOCAB);
    }
    common_speculative_begin(E.spec, A, {});
    feed(E, tokens(A_DEPTH, 55), 0, res.logits_a, A);
    llama_pos pos_a = A_DEPTH;
    if (hole) {
        llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), A, 40, 48);
    } else {
        llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), A, -1, -1);
        common_speculative_reset_state(E.spec, A);
    }
    feed(E, tokens(T_CATCHUP, 7), pos_a, res.logits_a, A);
    pos_a += T_CATCHUP;
    llama_token id_a = (llama_token) argmax(res.logits_a.data() + res.logits_a.size() - N_VOCAB);

    for (int s = 0; s < N_STEPS; ++s) {
        if (b_active) {
            draft_step(E, B, pos_b, id_b, scratch);
        }
        const int g0 = E.census.n["FLASH_ATTN_EXT"], t0 = E.census.n["QSA_BIAS"] + E.census.n["QSA_TOP_K"];
        draft_step(E, A, pos_a, id_a, res.logits_a, &res.dtoks_a, &res.dlogits_a);
        res.n_graphs_a += E.census.n["FLASH_ATTN_EXT"] - g0;
        res.n_sel_a    += E.census.n["QSA_BIAS"] + E.census.n["QSA_TOP_K"] - t0;
    }
    GGML_ASSERT(common_speculative_get_state(E.spec, A, res.carry_a));
    return res;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return 1e9f;
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

// [TAG_MULTISLOT] round 2: the coverage term is on under LLAMA_QSA_HOMOGENEOUS=1 or LLAMA_QSA_HISTORY_DENSE (unset = the
// default on: this binary decodes no image; 1 = on; 0 = off)
static bool history_on() {
    const char * h  = getenv("LLAMA_QSA_HOMOGENEOUS");
    const char * hd = getenv("LLAMA_QSA_HISTORY_DENSE");
    return (h != nullptr && strcmp(h, "1") == 0) || hd == nullptr || strcmp(hd, "0") != 0;
}

static void partial_history(gguf_context * meta, const std::function<void(bool, const std::string &)> & check) {
    const bool   on = history_on();
    const float  TOL = 1e-3f;
    if (!getenv("LLAMA_QSA_COMPACT_BIAS") || atoi(getenv("LLAMA_QSA_COMPACT_BIAS")) == 0) {
        printf("partial history: skipped (the route census needs LLAMA_QSA_COMPACT_BIAS=1, see tests/CMakeLists.txt)\n");
        return;
    }
    for (bool hole : { false, true }) {
        const partial_result solo = run_partial(meta, true, false, false, hole);
        const std::string kind = hole ? "interior hole" : "target-only restore";
        printf("partial %-19s solo:          %d draft graphs, QSA selection nodes %d\n", kind.c_str(), solo.n_graphs_a, solo.n_sel_a);
        if (on) {
            check(solo.n_graphs_a >= N_STEPS && solo.n_sel_a == 0, "partial " + kind + " solo: A's draft runs dense (incomplete history)");
        }
        for (int v = 0; v < 3; ++v) {
            const bool b_active = v != 1, reversed = v == 2;
            const partial_result m = run_partial(meta, false, b_active, reversed, hole);
            const std::string what = "partial " + kind + (b_active ? ", B active" : ", B inactive") + (reversed ? ", reversed ids" : "");
            printf("%-58s %d draft graphs, QSA selection nodes %d\n", (what + ":").c_str(), m.n_graphs_a, m.n_sel_a);
            if (!on) {
                // knob off: today's decision node for node - the pool high-water hides A's partial history (the gap)
                check(m.n_sel_a > 0, what + ": knob off keeps today's routing (A sparse beside deep B)");
                continue;
            }
            check(m.n_graphs_a >= N_STEPS && m.n_sel_a == 0, what + ": A's draft runs dense beside deep B");
            const float dt = max_abs_diff(m.logits_a, solo.logits_a), dd = max_abs_diff(m.dlogits_a, solo.dlogits_a);
            std::vector<float> cm((m.carry_a.size() - 4)/4), cs((solo.carry_a.size() - 4)/4);
            std::memcpy(cm.data(), m.carry_a.data() + 4, cm.size()*4);
            std::memcpy(cs.data(), solo.carry_a.data() + 4, cs.size()*4);
            const float dc = max_abs_diff(cm, cs);
            check(dt <= TOL, what + ": A target logits == solo (max |d| " + std::to_string(dt) + ")");
            check(m.dtoks_a == solo.dtoks_a && dd <= TOL, what + ": A draft proposals + logits == solo (max |d| " + std::to_string(dd) + ")");
            check(dc <= TOL, what + ": A carry == solo (max |d| " + std::to_string(dc) + ")");
        }
    }
}

// ---- [TAG_MULTISLOT] round 2 (Codex astra r1, Majors 1 + 2) -----------------------------------------------------------

template <class T> static void put_vec(const std::vector<T> & v) {
    const uint64_t n = v.size();
    fwrite(&n, sizeof(n), 1, stdout);
    fwrite(v.data(), sizeof(T), v.size(), stdout);
}

template <class T> static bool get_vec(FILE * f, std::vector<T> & v) {
    uint64_t n = 0;
    if (fread(&n, sizeof(n), 1, f) != 1 || n >= (1u << 26)) {
        return false;
    }
    v.resize(n);
    return fread(v.data(), sizeof(T), n, f) == n;
}

template <class T> static bool same_bytes(const std::vector<T> & a, const std::vector<T> & b) {
    return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(T)) == 0;
}

static std::string self_exe() {
#ifdef _WIN32
    return "";
#else
    char exe[4096];
    const ssize_t ne = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    GGML_ASSERT(ne > 0);
    exe[ne] = '\0';
    return exe;
#endif
}

// one sequence of a simultaneous run
struct seq_track {
    llama_seq_id id   = 0;
    llama_pos    pos  = 0;
    llama_token  last = 0;
    std::vector<float>       logits;  // its target logits
    std::vector<llama_token> dtoks;   // its draft proposals
    std::vector<float>       dlogits; // its last draft logits per round
};

// one joint round: A and B draft in ONE draft call (each draft position is one decode whose batch holds both rows),
// the draft rows rolled back, then ONE target decode with both id_last rows + the joint catch-up, 0 accepted
static void joint_step(mtp_env & E, seq_track & a, seq_track & b) {
    seq_track * t[2] = { a.id < b.id ? &a : &b, a.id < b.id ? &b : &a }; // the draft batch adds rows in seq id order
    std::vector<llama_token> drafted[2];
    for (int k = 0; k < 2; ++k) {
        auto & dp = common_speculative_get_draft_params(E.spec, t[k]->id);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.n_past   = t[k]->pos;
        dp.id_last  = t[k]->last;
        dp.prompt   = nullptr;
        dp.result   = &drafted[k];
    }
    E.census.active = true;
    common_speculative_draft(E.spec);
    llama_synchronize(E.ctx_dft.get());
    E.census.active = false;
    GGML_ASSERT(!drafted[0].empty() && drafted[0].size() == drafted[1].size()); // p_min 0: both run n_max positions
    for (int k = 0; k < 2; ++k) {
        t[k]->dtoks.insert(t[k]->dtoks.end(), drafted[k].begin(), drafted[k].end());
        const float * l = llama_get_logits_ith(E.ctx_dft.get(), k); // the last draft decode: row k = the k-th id
        t[k]->dlogits.insert(t[k]->dlogits.end(), l, l + N_VOCAB);
        llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), t[k]->id, t[k]->pos, -1);
    }
    llama_batch bt = llama_batch_init(2, 0, 1);
    for (int k = 0; k < 2; ++k) {
        common_batch_add(bt, t[k]->last, t[k]->pos, { t[k]->id }, true);
    }
    GGML_ASSERT(llama_decode(E.ctx_tgt.get(), bt) == 0);
    for (int k = 0; k < 2; ++k) {
        const float * l = llama_get_logits_ith(E.ctx_tgt.get(), k);
        t[k]->logits.insert(t[k]->logits.end(), l, l + N_VOCAB);
    }
    GGML_ASSERT(common_speculative_process(E.spec, bt));
    llama_synchronize(E.ctx_dft.get());
    llama_batch_free(bt);
    for (int k = 0; k < 2; ++k) {
        common_speculative_accept(E.spec, t[k]->id, 0);
        t[k]->pos += 1;
        t[k]->last = (llama_token) argmax(t[k]->logits.data() + t[k]->logits.size() - N_VOCAB);
    }
}

// B's fill: complete history to L_RESTORED (seeds 300 + c, as run_partial)
static void fill_b(mtp_env & E, seq_track & b) {
    common_speculative_begin(E.spec, b.id, {});
    for (uint32_t c = 0; b.pos < L_RESTORED; ++c) {
        const auto toks = tokens(std::min<llama_pos>(128, L_RESTORED - b.pos), 300 + c);
        feed(E, toks, b.pos, b.logits, b.id);
        b.pos += (llama_pos) toks.size();
    }
    b.last = (llama_token) argmax(b.logits.data() + b.logits.size() - N_VOCAB);
}

// A's fill + partial draft history (target-only restore or interior hole) + catch-up (as run_partial)
static void fill_a_partial(mtp_env & E, seq_track & a, bool hole) {
    common_speculative_begin(E.spec, a.id, {});
    feed(E, tokens(A_DEPTH, 55), 0, a.logits, a.id);
    a.pos = A_DEPTH;
    if (hole) {
        llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), a.id, 40, 48);
    } else {
        llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), a.id, -1, -1);
        common_speculative_reset_state(E.spec, a.id);
    }
    feed(E, tokens(T_CATCHUP, 7), a.pos, a.logits, a.id);
    a.pos += T_CATCHUP;
    a.last = (llama_token) argmax(a.logits.data() + a.logits.size() - N_VOCAB);
}

struct simul_result {
    seq_track a, b;
    int n_graphs = 0; // draft graphs (FLASH_ATTN_EXT) over the joint rounds
    int n_sel    = 0; // QSA selection nodes (QSA_BIAS + QSA_TOP_K) in them
};

// per-stream KV (kv_unified = false, n_stream = n_seq > 1): B deep + complete, A partial, then N_STEPS joint rounds
static simul_result run_simul(gguf_context * meta, uint32_t n_seq, llama_seq_id id_a, llama_seq_id id_b, bool hole) {
    simul_result r;
    mtp_env E;
    init_env(E, meta, n_seq, /*unified =*/ false);
    r.a.id = id_a;
    r.b.id = id_b;
    fill_b(E, r.b);
    fill_a_partial(E, r.a, hole);
    r.a.logits.clear(); r.b.logits.clear(); // compare the joint rounds only
    E.census.n.clear();
    for (int s = 0; s < N_STEPS; ++s) {
        joint_step(E, r.a, r.b);
    }
    r.n_graphs = E.census.n["FLASH_ATTN_EXT"];
    r.n_sel    = E.census.n["QSA_BIAS"] + E.census.n["QSA_TOP_K"];
    return r;
}

// the solo references of the joint rounds: A (partial) or B (complete) alone in a one-sequence context, same schedule
static seq_track run_solo_ref(gguf_context * meta, bool is_a, bool hole, int * n_sel = nullptr) {
    mtp_env E;
    init_env(E, meta, 1);
    seq_track t;
    if (is_a) {
        fill_a_partial(E, t, hole);
    } else {
        fill_b(E, t);
    }
    t.logits.clear();
    E.census.n.clear();
    for (int s = 0; s < N_STEPS; ++s) {
        draft_step(E, 0, t.pos, t.last, t.logits, &t.dtoks, &t.dlogits);
    }
    if (n_sel) {
        *n_sel = E.census.n["QSA_BIAS"] + E.census.n["QSA_TOP_K"];
    }
    return t;
}

struct simul_cfg { uint32_t n_seq; llama_seq_id a, b; bool hole; };
static const simul_cfg SIMUL_CFGS[6] = { {2, 0, 1, false}, {2, 0, 1, true}, {2, 1, 0, false}, {2, 1, 0, true},
                                         {4, 1, 2, false}, {4, 1, 2, true} };

// child --simul-ref: the joint runs (A's then B's logits, dtoks, dlogits per config)
static void simul_ref_child(gguf_context * meta) {
    for (const simul_cfg & c : SIMUL_CFGS) {
        const simul_result r = run_simul(meta, c.n_seq, c.a, c.b, c.hole);
        for (const seq_track * t : { &r.a, &r.b }) {
            put_vec(t->logits);
            put_vec(t->dtoks);
            put_vec(t->dlogits);
        }
    }
}

// child --solo-ref: the solo references (A restore, A hole, B), each as logits, dtoks, dlogits
static void solo_ref_child(gguf_context * meta) {
    for (int k = 0; k < 3; ++k) {
        const seq_track t = run_solo_ref(meta, k < 2, k == 1);
        put_vec(t.logits);
        put_vec(t.dtoks);
        put_vec(t.dlogits);
    }
}

// one draft position (dp.n_max 1: exactly one draft decode) of seq 0 at n_past `pos`, rolled back; returns the graph
// reuse delta (llama_perf_context n_reused: the graph cache's hit or the previous-graph reuse, both via can_reuse) and
// the QSA selection nodes of that graph
static void draft_once(mtp_env & E, llama_pos pos, llama_token id_last, int & reused, int & n_sel) {
    std::vector<llama_token> drafted;
    auto & dp = common_speculative_get_draft_params(E.spec, 0);
    dp.drafting = true;
    dp.n_max    = 1;
    dp.n_past   = pos;
    dp.id_last  = id_last;
    dp.prompt   = nullptr;
    dp.result   = &drafted;
    const int r0 = llama_perf_context(E.ctx_dft.get()).n_reused;
    E.census.n.clear();
    E.census.active = true;
    common_speculative_draft(E.spec);
    llama_synchronize(E.ctx_dft.get());
    E.census.active = false;
    GGML_ASSERT(drafted.size() == 1 && E.census.n["FLASH_ATTN_EXT"] == 1);
    reused = llama_perf_context(E.ctx_dft.get()).n_reused - r0;
    n_sel  = E.census.n["QSA_BIAS"] + E.census.n["QSA_TOP_K"];
    llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), 0, pos, -1);
}

// child --trans: the can_reuse transitions at an unchanged shape (one draft row, draft cell window 256 throughout).
// Prints "TRANS" + (reused, n_sel) of: ctl0 ctl1 (complete, sparse) | hole0 hole1 (complete -> partial: interior
// hole 40..47) | rew0 rew1 (partial -> complete: the rewind seq_rm [40, -1) removes the hole; the draft row at n_past
// 40 completes 0..40). The *0 step is the transition, the *1 step the same state again.
static void trans_child(gguf_context * meta) {
    mtp_env E;
    init_env(E, meta, 1);
    common_speculative_begin(E.spec, 0, {});
    std::vector<float> lg;
    feed(E, tokens(A_DEPTH, 55), 0, lg, 0);
    const llama_token last = (llama_token) argmax(lg.data() + lg.size() - N_VOCAB);
    int v[12];
    draft_once(E, A_DEPTH, last, v[0], v[1]); // the first draft graph of the context (built)
    draft_once(E, A_DEPTH, last, v[0], v[1]);
    draft_once(E, A_DEPTH, last, v[2], v[3]);
    llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), 0, 40, 48);
    draft_once(E, A_DEPTH, last, v[4], v[5]);
    draft_once(E, A_DEPTH, last, v[6], v[7]);
    llama_memory_seq_rm(llama_get_memory(E.ctx_dft.get()), 0, 40, -1);
    draft_once(E, 40, last, v[8], v[9]);
    draft_once(E, 40, last, v[10], v[11]);
    printf("TRANS");
    for (int x : v) {
        printf(" %d", x);
    }
    printf("\n");
}

// child --knob: the complete single-sequence history (run_scenario, no restore): QSA selection count, draft proposals,
// draft logits, target logits - the knob on/off byte-identity check
static void knob_child(gguf_context * meta) {
    const run_result r = run_scenario(meta, false);
    std::vector<int32_t> sel = { r.n_sel };
    put_vec(sel);
    put_vec(r.dtoks);
    put_vec(r.dlogits);
    put_vec(r.logits);
}

// child --mm (knob unset): an M-RoPE embedding batch (an mtmd image chunk: token == nullptr, 4 position rows) in a
// throwaway context flips the default off for the process; afterwards the partial solo restore routes as a25b663c1
static void mm_child(gguf_context * meta) {
    printf("MM before=%d\n", (int) llama_qsa_history_dense());
    fflush(stdout);
    {
        mtp_env E;
        init_env(E, meta, 1);
        const int32_t n = 4, n_embd = llama_model_n_embd_inp(E.model.get());
        llama_batch b = llama_batch_init(n, n_embd, 1);
        std::vector<llama_pos> pos4(4*n);
        for (int32_t i = 0; i < n; ++i) {
            pos4[i] = 0; pos4[n + i] = i / 2; pos4[2*n + i] = i % 2; pos4[3*n + i] = 0; // t, h, w, - (one 2x2 image)
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i == n - 1;
        }
        std::fill(b.embd, b.embd + (size_t) n*n_embd, 0.01f);
        b.n_tokens = n;
        llama_pos * pos_own = b.pos;
        b.pos = pos4.data();
        const int rc = llama_decode(E.ctx_tgt.get(), b);
        b.pos = pos_own;
        llama_batch_free(b);
        printf("MM decode=%d after=%d\n", rc, (int) llama_qsa_history_dense());
    }
    int n_sel = 0;
    run_solo_ref(meta, true, false, &n_sel);
    printf("MM partial_sel=%d\n", n_sel);
}

static void round2(gguf_context * meta, const std::function<void(bool, const std::string &)> & check) {
    const std::string exe = self_exe();
    if (exe.empty()) {
        printf("round 2: skipped (no re-exec on _WIN32)\n");
        return;
    }
    if (!getenv("LLAMA_QSA_COMPACT_BIAS") || atoi(getenv("LLAMA_QSA_COMPACT_BIAS")) == 0) {
        printf("round 2: skipped (the route census needs LLAMA_QSA_COMPACT_BIAS=1)\n");
        return;
    }
    const bool on = history_on();

    // (a) the dense reference: the same schedules in a LLAMA_MTP_QSA=0 child (plain dense draft KV, the target
    // unchanged - see the target-exact checks)
    std::vector<float> rl[3], rdl[3];
    std::vector<llama_token> rdt[3];
    {
        FILE * p = popen(("LLAMA_MTP_QSA=0 '" + exe + "' --solo-ref").c_str(), "r");
        GGML_ASSERT(p != nullptr);
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            ok = ok && get_vec(p, rl[k]) && get_vec(p, rdt[k]) && get_vec(p, rdl[k]);
        }
        check(pclose(p) == 0 && ok, "round 2: dense reference child (LLAMA_MTP_QSA=0 --solo-ref) exited 0");
    }

    // (b) Major 2: solo, knob unset - the target-only restore (draft positions 96..127 only, depth 128 <= the
    // 256-cell window: the old `depth > n_kv` term misses it) runs dense and drafts exactly what the dense draft
    // memory drafts. Byte equality: the QSA layer's dense branch over the same F16 cells is the LLAMA_MTP_QSA=0 graph.
    for (int k = 0; k < 2; ++k) {
        int n_sel = 0;
        const seq_track t = run_solo_ref(meta, true, k == 1, &n_sel);
        const std::string kind = k == 1 ? "interior hole" : "target-only restore";
        printf("round 2 solo %-19s: QSA selection nodes %d, %zu proposals\n", kind.c_str(), n_sel, t.dtoks.size());
        check(same_bytes(t.logits, rl[k]), "solo " + kind + ": target logits == the LLAMA_MTP_QSA=0 run (bytes)");
        if (on) {
            check(n_sel == 0, "solo " + kind + ": history fallback ON -> the draft runs dense");
            check(t.dtoks == rdt[k] && same_bytes(t.dlogits, rdl[k]),
                  "solo " + kind + ": draft proposals + logits == the dense draft memory (bytes)");
        } else {
            check(n_sel > 0, "solo " + kind + ": LLAMA_QSA_HISTORY_DENSE=0 keeps the a25b663c1 routing (sparse)");
        }
    }

    // (c) Major 1.1: per-stream KV (n_stream = n_seq > 1), A partial + B deep complete in the SAME draft ubatch.
    // Two references: the same joint schedule with the dense draft memory (LLAMA_MTP_QSA=0 child, the same 2-row
    // batches: the same arithmetic path -> BYTE equality), and the solo runs (TOLERANCE only: the CPU mul_mat picks
    // its kernel / accumulation order by the src1 row count, so a 2-row decode rounds differently from a 1-row one -
    // visible already in the target, whose route the knob does not touch: ~2e-8)
    std::vector<float> sl[6][2], sdl[6][2];
    std::vector<llama_token> sdt[6][2];
    {
        FILE * p = popen(("LLAMA_MTP_QSA=0 '" + exe + "' --simul-ref").c_str(), "r");
        GGML_ASSERT(p != nullptr);
        bool ok = true;
        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < 2; ++j) {
                ok = ok && get_vec(p, sl[k][j]) && get_vec(p, sdt[k][j]) && get_vec(p, sdl[k][j]);
            }
        }
        check(pclose(p) == 0 && ok, "round 2: joint dense reference child (LLAMA_MTP_QSA=0 --simul-ref) exited 0");
    }
    const float TOL = 1e-5f;
    for (int ci = 0; ci < 6; ++ci) {
        const simul_cfg c = SIMUL_CFGS[ci];
        const bool hole = c.hole;
        {
            const simul_result r = run_simul(meta, c.n_seq, c.a, c.b, hole);
            const std::string what = "simul np" + std::to_string(c.n_seq) + " A=" + std::to_string(c.a) + " B=" +
                    std::to_string(c.b) + (hole ? " hole" : " restore");
            printf("%-34s %d draft graphs, QSA selection nodes %d, max |d| vs solo: A tgt %g dft %g, B tgt %g dft %g, proposals A %s B %s\n",
                   (what + ":").c_str(), r.n_graphs, r.n_sel, max_abs_diff(r.a.logits, rl[hole ? 1 : 0]),
                   max_abs_diff(r.a.dlogits, rdl[hole ? 1 : 0]), max_abs_diff(r.b.logits, rl[2]), max_abs_diff(r.b.dlogits, rdl[2]),
                   r.a.dtoks == rdt[hole ? 1 : 0] ? "==" : "!=", r.b.dtoks == rdt[2] ? "==" : "!=");
            // one decode per draft position with both rows in one ubatch: as many graphs as a solo run's
            check(r.n_graphs == N_STEPS*(int) (r.a.dtoks.size()/N_STEPS), what + ": A and B share every draft ubatch");
            // the target: unaffected by the draft route
            check(same_bytes(r.a.logits, sl[ci][0]) && same_bytes(r.b.logits, sl[ci][1]),
                  what + ": A and B target logits == the joint dense-draft run (bytes)");
            check(max_abs_diff(r.a.logits, rl[hole ? 1 : 0]) <= TOL && max_abs_diff(r.b.logits, rl[2]) <= TOL,
                  what + ": A and B target logits == solo (tolerance 1e-5: 2-row vs 1-row rounding)");
            if (!on) {
                check(r.n_sel > 0, what + ": knob off: today's routing (the stream high-water hides A, sparse)");
                continue;
            }
            // routing is per ubatch: A's partial history routes the WHOLE ubatch dense, B's rows included
            check(r.n_sel == 0, what + ": the shared ubatch runs dense (A partial -> one dense topology for A and B)");
            check(r.a.dtoks == sdt[ci][0] && same_bytes(r.a.dlogits, sdl[ci][0]),
                  what + ": A's draft proposals + logits == the joint dense-draft run (bytes)");
            check(r.b.dtoks == sdt[ci][1] && same_bytes(r.b.dlogits, sdl[ci][1]),
                  what + ": B's draft proposals + logits == the joint dense-draft run (bytes)");
            check(r.a.dtoks == rdt[hole ? 1 : 0] && max_abs_diff(r.a.dlogits, rdl[hole ? 1 : 0]) <= TOL,
                  what + ": A's draft proposals == A solo dense, logits within 1e-5");
            check(r.b.dtoks == rdt[2] && max_abs_diff(r.b.dlogits, rdl[2]) <= TOL,
                  what + ": B's draft proposals == B solo dense, logits within 1e-5");
        }
    }

    // (d) Major 1.2: can_reuse at an unchanged shape, knob on vs a LLAMA_QSA_HISTORY_DENSE=0 child (the only
    // difference): a coverage change must miss where the old routing reuses
    auto trans = [&](const std::string & env, int (&v)[12]) {
        FILE * p = popen((env + " '" + exe + "' --trans").c_str(), "r");
        GGML_ASSERT(p != nullptr);
        char line[512] = {0};
        bool got = false;
        while (fgets(line, sizeof(line), p)) {
            if (sscanf(line, "TRANS %d %d %d %d %d %d %d %d %d %d %d %d", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5],
                       &v[6], &v[7], &v[8], &v[9], &v[10], &v[11]) == 12) {
                got = true;
            }
        }
        return pclose(p) == 0 && got;
    };
    int t_on[12], t_off[12];
    const bool ok_on  = trans("env -u LLAMA_QSA_HOMOGENEOUS -u LLAMA_QSA_HISTORY_DENSE", t_on);
    const bool ok_off = trans("env -u LLAMA_QSA_HOMOGENEOUS LLAMA_QSA_HISTORY_DENSE=0", t_off);
    check(ok_on && ok_off, "transitions: both children (knob default / =0) ran");
    if (ok_on && ok_off) {
        const char * gc = getenv("LLAMA_GRAPH_CACHE");
        const bool   k1 = gc == nullptr || atoi(gc) <= 0; // previous-graph reuse: only the last graph is a candidate
        printf("transitions (reused, sel) on : ctl %d,%d %d,%d hole %d,%d %d,%d rewind %d,%d %d,%d\n", t_on[0], t_on[1],
               t_on[2], t_on[3], t_on[4], t_on[5], t_on[6], t_on[7], t_on[8], t_on[9], t_on[10], t_on[11]);
        printf("transitions (reused, sel) off: ctl %d,%d %d,%d hole %d,%d %d,%d rewind %d,%d %d,%d\n", t_off[0], t_off[1],
               t_off[2], t_off[3], t_off[4], t_off[5], t_off[6], t_off[7], t_off[8], t_off[9], t_off[10], t_off[11]);
        check(t_on[2] == 1 && t_on[3] > 0 && t_off[2] == 1 && t_off[3] > 0,
              "transitions: control - the complete sparse draft graph is reused at this shape");
        check(t_off[4] == 1 && t_off[5] > 0, "transitions: knob off, complete -> partial (hole) reuses the sparse graph (old routing)");
        check(t_on[4] == 0 && t_on[5] == 0, "transitions: knob on, complete -> partial (hole) MISSES and runs dense");
        check(t_on[6] == 1 && t_on[7] == 0, "transitions: the partial state again reuses its dense graph");
        check(t_off[8] == 1 && t_off[9] > 0, "transitions: knob off, the rewound complete state reuses the sparse graph");
        if (k1) {
            check(t_on[8] == 0 && t_on[9] > 0, "transitions: knob on, partial -> complete (rewind) MISSES the dense graph, runs sparse");
        } else {
            check(t_on[9] > 0, "transitions: knob on, partial -> complete (rewind) runs sparse (K > 1: the cached sparse entry may hit)");
        }
        check(t_on[10] == 1 && t_on[11] > 0, "transitions: the complete state again reuses its sparse graph");
    }

    // (f) Major 2 provenance: the default switches off after an M-RoPE image (knob unset)
    {
        FILE * p = popen(("env -u LLAMA_QSA_HOMOGENEOUS -u LLAMA_QSA_HISTORY_DENSE '" + exe + "' --mm 2>&1").c_str(), "r");
        GGML_ASSERT(p != nullptr);
        std::string out;
        char buf[4096];
        size_t nr;
        while ((nr = fread(buf, 1, sizeof(buf), p)) > 0) {
            out.append(buf, nr);
        }
        const int rc = pclose(p);
        int before = -1, dec = -1, after = -1, sel = -1;
        const size_t a = out.find("MM before="), b = out.find("MM decode="), c = out.find("MM partial_sel=");
        if (a != std::string::npos) sscanf(out.c_str() + a, "MM before=%d", &before);
        if (b != std::string::npos) sscanf(out.c_str() + b, "MM decode=%d after=%d", &dec, &after);
        if (c != std::string::npos) sscanf(out.c_str() + c, "MM partial_sel=%d", &sel);
        const size_t f = out.find("\nqsa_history_dense = off");
        printf("mm: before %d, image decode rc %d, after %d, partial solo selection nodes %d; %s", before, dec, after, sel,
               f == std::string::npos ? "no flip line\n" : out.substr(f + 1, out.find('\n', f + 1) - f).c_str());
        check(rc == 0 && before == 1 && dec == 0 && after == 0 && sel > 0 && f != std::string::npos,
              "mm: knob unset - on until an M-RoPE image is decoded, then off (logged once), the partial history routes as a25b663c1");
    }

    // (e) Major 1.4: single complete sequence - the knob changes nothing: unset / HISTORY_DENSE=0 / =1 /
    // HOMOGENEOUS=1 children give the same selection count, proposals, draft logits and target logits (bytes)
    const char * envs[4] = { "env -u LLAMA_QSA_HOMOGENEOUS -u LLAMA_QSA_HISTORY_DENSE",
                             "env -u LLAMA_QSA_HOMOGENEOUS LLAMA_QSA_HISTORY_DENSE=0",
                             "env -u LLAMA_QSA_HOMOGENEOUS LLAMA_QSA_HISTORY_DENSE=1",
                             "env LLAMA_QSA_HOMOGENEOUS=1 LLAMA_QSA_HISTORY_DENSE=0" };
    std::vector<int32_t> ks[4];
    std::vector<llama_token> kt[4];
    std::vector<float> kd[4], kl[4];
    for (int k = 0; k < 4; ++k) {
        FILE * p = popen((std::string(envs[k]) + " '" + exe + "' --knob").c_str(), "r");
        GGML_ASSERT(p != nullptr);
        const bool ok = get_vec(p, ks[k]) && get_vec(p, kt[k]) && get_vec(p, kd[k]) && get_vec(p, kl[k]);
        check(pclose(p) == 0 && ok && ks[k][0] > 0, std::string("knob child ") + envs[k] + ": ran, draft sparse");
    }
    for (int k = 1; k < 4; ++k) {
        check(ks[k] == ks[0] && kt[k] == kt[0] && same_bytes(kd[k], kd[0]) && same_bytes(kl[k], kl[0]),
              std::string("knob identity, complete single sequence: ") + envs[k] + " == unset (selection, proposals, draft + target logits)");
    }
}

// the unified pool layout straight into the census: A (seq 0) holds cells 0..T-1 at positions L..L+T-1 (past the
// window), B (seq 1) cells T..T+B_PROMPT-1 at positions 0..B_PROMPT-1; n_kv 256, r 4
static void census_unit(const std::function<void(bool, const std::string &)> & check) {
    const int64_t n_kv = 256, r = 4, n_blocks = (n_kv + r - 1)/r;

    llama_kv_cells cells;
    cells.resize((uint32_t) n_kv);
    for (int i = 0; i < T_CATCHUP; ++i) {
        cells.pos_set((uint32_t) i, L_RESTORED + i);
        cells.seq_add((uint32_t) i, 0);
    }
    for (int i = 0; i < B_PROMPT; ++i) {
        cells.pos_set((uint32_t) (T_CATCHUP + i), i);
        cells.seq_add((uint32_t) (T_CATCHUP + i), 1);
    }

    std::vector<int32_t> cell_blk(n_kv), blk_cells(r*n_blocks), blk_pos(4*n_blocks, 0), btab(2*n_blocks);
    llama_qsa_census_stream cs;
    llama_qsa_census_flags  fl;
    llama_qsa_census_full(cells, false, n_kv, n_blocks, r, /*blk_bias =*/ true, cell_blk.data(), blk_cells.data(),
                          blk_pos.data(), n_blocks, cs, &fl, 0);
    llama_qsa_fill_blk_tab(cells, cs, n_blocks, 0, n_blocks, /*tail_fix =*/ true, btab.data());

    const int32_t kill = cs.n_bid + 1;
    bool a_killed = kill < n_blocks;
    for (int i = 0; i < T_CATCHUP; ++i) {
        a_killed = a_killed && cell_blk[i] == kill;
    }
    bool a_pooled = false;
    for (int32_t b = 0; b < cs.n_bid; ++b) {
        a_pooled = a_pooled || cells.seq_has((uint32_t) cs.bid_cell[b], 0);
    }

    printf("census:     n_bid %d of %lld blocks, oor %d, A cells -> id %d (vis %d), oor_seqs %zu\n", cs.n_bid,
           (long long) n_blocks, (int) fl.oor, cell_blk[0], kill < n_blocks ? btab[2*kill + 1] : -1, cs.oor_seqs.size());
    check(fl.oor,                                           "census: A's catch-up cells are past the window (oor)");
    check(a_killed && btab[2*kill + 1] == 0,                "census: every A cell -> the second unused id, scored -inf (vis 0)");
    check(!a_pooled,                                        "census: no compact block holds an A cell");
    check(cs.n_bid == B_PROMPT/r,                           "census: B's blocks pooled as without A");
    check(cs.oor_seqs.size() == 1 && cs.oor_seqs[0] == 0,   "census: oor_seqs == {A}");
}

int main(int argc, char ** argv) {
    const bool ref  = argc > 1 && strcmp(argv[1], "--ref") == 0;
    const bool desc = argc > 1 && strcmp(argv[1], "--desc") == 0;
    const std::string mode = argc > 1 ? argv[1] : ""; // [TAG_MULTISLOT] round 2 child modes

    llama_log_set(quiet_log, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    gguf_context_ptr meta = make_meta(4096);

    if (desc) {
        // the child: the two descriptors, one per line
        mtp_env E;
        init_env(E, meta.get());
        printf("T %s\nD %s\n", llama_state_seq_descriptor(E.ctx_tgt.get()).c_str(),
               llama_state_seq_descriptor(E.ctx_dft.get()).c_str());
        fflush(stdout);
        common_speculative_free(E.spec); // [TAG_MULTISLOT] round 2: before the contexts (was a use-after-free at exit)
        E.spec = nullptr;
        E.ctx_dft.reset(); E.ctx_tgt.reset();
        llama_backend_free();
        return 0;
    }

    if (mode == "--mm") {
        mm_child(meta.get());
        fflush(stdout);
        llama_backend_free();
        return 0;
    }
    if (mode == "--solo-ref" || mode == "--simul-ref" || mode == "--trans" || mode == "--knob") {
        if (mode == "--solo-ref") {
            solo_ref_child(meta.get());
        } else if (mode == "--simul-ref") {
            simul_ref_child(meta.get());
        } else if (mode == "--trans") {
            trans_child(meta.get());
        } else {
            knob_child(meta.get());
        }
        fflush(stdout);
        llama_backend_free();
        return 0;
    }

    if (ref) {
        // the child: the scenarios' target logits, raw f32, restore first, unified last
        for (bool restore : { true, false }) {
            const run_result r = run_scenario(meta.get(), restore);
            const uint64_t n = r.logits.size();
            fwrite(&n, sizeof(n), 1, stdout);
            fwrite(r.logits.data(), sizeof(float), r.logits.size(), stdout);
        }
        {
            const unified_result u = run_unified(meta.get());
            const uint64_t n = u.logits.size();
            fwrite(&n, sizeof(n), 1, stdout);
            fwrite(u.logits.data(), sizeof(float), u.logits.size(), stdout);
        }
        fflush(stdout);
        llama_backend_free();
        return 0;
    }

    const char * mq = getenv("LLAMA_MTP_QSA");
    if (mq == nullptr || atoi(mq) == 0) {
        printf("test-mtp-qsa-restore: needs LLAMA_MTP_QSA=1 (see tests/CMakeLists.txt)\n");
        return 1;
    }

    int n_fail = 0;
    auto check = [&](bool ok, const std::string & what) {
        printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
        n_fail += ok ? 0 : 1;
    };

    printf("test-mtp-qsa-restore: L %d, catch-up %d, %d draft steps, LLAMA_MTP_QSA_PHASE=%s\n", (int) L_RESTORED,
           T_CATCHUP, N_STEPS, getenv("LLAMA_MTP_QSA_PHASE") ? getenv("LLAMA_MTP_QSA_PHASE") : "(unset)");

    const run_result rr = run_scenario(meta.get(), true);
    printf("restore:    %d drafted tokens, %d draft graphs, QSA selection nodes in them %d\n", rr.n_drafted, rr.n_graphs, rr.n_sel);
    check(rr.n_drafted > 0 && rr.n_graphs >= N_STEPS, "restore: the draft steps ran (no abort)");
    check(rr.n_sel == 0,     "restore: the draft runs dense (depth > the draft's cell window)");

    partial_history(meta.get(), check); // [TAG_MULTISLOT]
    round2(meta.get(), check);          // [TAG_MULTISLOT] round 2

    const run_result rn = run_scenario(meta.get(), false);
    printf("no restore: %d drafted tokens, %d draft graphs, QSA selection nodes in them %d\n", rn.n_drafted, rn.n_graphs, rn.n_sel);
    check(rn.n_drafted > 0,  "no restore: the draft steps ran");
    check(rn.n_sel > 0,      "no restore: the draft runs the QSA selection (sparse)");

    const unified_result ru = run_unified(meta.get());
    // the per-graph top-k count without selection differs from the one-sequence context (2 vs 1 here), so the two
    // sequences are compared: B's sparse draft graphs carry exactly one more top-k node (the QSA selection) than A's
    printf("unified:    B %d drafted tokens, %d draft graphs, top-k nodes %d; A %d draft graphs, top-k nodes %d\n",
           ru.n_drafted_b, ru.n_graphs_b, ru.n_topk_b, ru.n_graphs_a, ru.n_topk_a);
    check(ru.n_drafted_b > 0 && ru.n_graphs_b >= N_STEPS, "unified: B's draft steps ran next to A's out-of-window cells (no abort)");
    check(ru.n_graphs_a > 0 && ru.n_topk_a % ru.n_graphs_a == 0 && ru.n_topk_b % ru.n_graphs_b == 0 &&
          ru.n_topk_b/ru.n_graphs_b == ru.n_topk_a/ru.n_graphs_a + 1,
          "unified: B's draft runs the QSA selection (sparse), A's runs dense (depth > the window)");

    census_unit(check);

    // the reference target logits: this binary with LLAMA_MTP_QSA=0 (dense draft memory)
#ifdef _WIN32
    printf("  skip target-exact check (no re-exec on _WIN32)\n");
#else
    {
        char exe[4096];
        const ssize_t ne = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
        GGML_ASSERT(ne > 0);
        exe[ne] = '\0';
        const std::string cmd = std::string("LLAMA_MTP_QSA=0 '") + exe + "' --ref";
        FILE * p = popen(cmd.c_str(), "r");
        GGML_ASSERT(p != nullptr);
        std::vector<float> ref_logits[3];
        for (auto & v : ref_logits) {
            uint64_t n = 0;
            if (fread(&n, sizeof(n), 1, p) == 1 && n < (1u << 26)) {
                v.resize(n);
                if (fread(v.data(), sizeof(float), n, p) != n) {
                    v.clear();
                }
            }
        }
        const int rc = pclose(p);
        check(rc == 0, "reference child (LLAMA_MTP_QSA=0) exited 0");

        // [series D1] the draft descriptor tells the dense draft memory from the sparse one (and the phase)
        std::string dt[3], dd[3];
        const char * envs[3] = { "LLAMA_MTP_QSA=0 LLAMA_MTP_QSA_PHASE=0", "LLAMA_MTP_QSA=1 LLAMA_MTP_QSA_PHASE=0",
                                 "LLAMA_MTP_QSA=1 LLAMA_MTP_QSA_PHASE=1" };
        for (int k = 0; k < 3; ++k) {
            FILE * q = popen((std::string(envs[k]) + " '" + exe + "' --desc").c_str(), "r");
            GGML_ASSERT(q != nullptr);
            std::string out;
            char buf[4096];
            size_t nr;
            while ((nr = fread(buf, 1, sizeof(buf), q)) > 0) {
                out.append(buf, nr);
            }
            pclose(q);
            const size_t t0 = out.find("T "), d0 = out.find("\nD ");
            if (t0 != std::string::npos && d0 != std::string::npos) {
                dt[k] = out.substr(t0 + 2, d0 - t0 - 2);
                dd[k] = out.substr(d0 + 3, out.find('\n', d0 + 1) - d0 - 3);
            }
            const size_t m = dd[k].find(";mem=");
            printf("descriptor: %-38s draft mem=%s\n", envs[k], m == std::string::npos ? "?" : dd[k].substr(m + 5).c_str());
        }
        check(!dt[0].empty() && dt[0] == dt[1] && dt[1] == dt[2], "descriptor: target descriptor equal for MTP_QSA 0 / 1 / phase 1");
        check(!dd[0].empty() && dd[0] != dd[1] && dd[1] != dd[2] && dd[0] != dd[2],
              "descriptor: draft descriptor differs between dense, sparse and sparse phase 1");
        auto same = [](const std::vector<float> & a, const std::vector<float> & b) {
            return !a.empty() && a.size() == b.size() && memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
        };
        check(same(rr.logits, ref_logits[0]), "restore: target logits (" + std::to_string(rr.logits.size()/N_VOCAB) +
              " rows) == the LLAMA_MTP_QSA=0 run");
        check(same(rn.logits, ref_logits[1]), "no restore: target logits (" + std::to_string(rn.logits.size()/N_VOCAB) +
              " rows) == the LLAMA_MTP_QSA=0 run");
        check(same(ru.logits, ref_logits[2]), "unified: target logits (" + std::to_string(ru.logits.size()/N_VOCAB) +
              " rows) == the LLAMA_MTP_QSA=0 run");
    }
#endif

    llama_backend_free();

    printf("%s\n", n_fail == 0 ? "test-mtp-qsa-restore: OK" : "test-mtp-qsa-restore: FAIL");
    return n_fail == 0 ? 0 : 1;
}
