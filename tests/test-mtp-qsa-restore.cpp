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

#include <algorithm>
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
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static void init_env(mtp_env & env, gguf_context * meta, uint32_t n_seq = 1) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    mp.load_mtp          = true;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;

    size_t seed = 31;
    env.model.reset(llama_model_init_from_user(meta, set_tensor_data, &seed, mp));
    GGML_ASSERT(env.model && "model init failed");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 1024;
    cp.n_batch         = 128;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = n_seq;
    cp.kv_unified      = n_seq > 1;
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
static int draft_step(mtp_env & E, llama_seq_id seq, llama_pos & pos, llama_token & id_last, std::vector<float> & logits) {
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
    const bool ref = argc > 1 && strcmp(argv[1], "--ref") == 0;

    llama_log_set(quiet_log, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    gguf_context_ptr meta = make_meta(4096);

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
