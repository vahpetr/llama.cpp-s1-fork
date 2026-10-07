// [TAG_MTP_CATCHUP_NOATTN] Attention-free MTP draft catch-up (R80.4, stage-2 item 7) is exact.
//
// A CPU qwen4exp with random weights (2 trunk layers, the 2nd full attention + QSA indexer, 1 MTP block, hc = 4, MoE)
// drives the draft-mtp speculative implementation through the server's call order. Two instances built from the same
// weights replay identical target batches: A with LLAMA_MTP_CATCHUP_ATTN=1 (today's full draft graph), B with
// LLAMA_MTP_CATCHUP_ATTN=0 (a zero-output catch-up ubatch stores K/V only). Codex B3:
//  1. after EVERY catch-up (sizes 1, 31/32/33 around the LLAMA_MTP_EHPROJ_FLAT threshold, a full ubatch, a full +
//     final partial ubatch, all but the first at a non-zero start) the complete draft host state stream
//     (llama_state_seq_get_data of ctx_dft, flags 0) and the speculative carry (common_speculative_get_state) are
//     byte-identical, and so are the target's logits;
//  2. then verification rounds: draft tokens + the last draft logits, the target's verification logits (every row),
//     the acceptance decision (target argmax), the per-round checkpoint restore + rollback of the rejected suffix +
//     replay, and the state streams after each step are identical;
//  3. the op census of one catch-up ubatch, counted from the scheduler's eval callback on separate instances: the new
//     graph keeps the cache writes (SET_ROWS = cpy_k + cpy_v) and has no attention op (FLASH_ATTN_EXT / SOFT_MAX), no
//     MoE (MUL_MAT_ID); the old one has them;
//  4. a timing line (CPU, indication only): the catch-up per ubatch, old vs new, over a deeper fill.
// Variants: draft KV f16 (no rotation), q8_0 (K/V Hadamard rotations in the store path), f16 + backend draft sampling.

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

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static const uint32_t N_VOCAB = 128;

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

// qwen4exp: the test-llama-archs fixture (hc 4, low rank 8, QSA ratios 4, MoE 2 experts) + 1 NextN/MTP block
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

// op census of the draft context's computed nodes (scheduler eval callback, ask phase only)
struct op_census {
    bool active = false;
    std::map<std::string, int> n;
    int total = 0;
};

static bool census_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * c = (op_census *) ud;
    if (ask && c->active) {
        c->n[ggml_op_desc(t)]++;
        c->total++;
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
    if (level >= GGML_LOG_LEVEL_WARN || strstr(text, "LLAMA_MTP_CATCHUP_ATTN") != nullptr) {
        fputs(text, stderr);
    }
}

struct env_cfg {
    const char * catchup_attn;   // the knob for ctx_dft
    ggml_type    type_kv_dft;
    uint32_t     n_ctx;
    bool         census;
    bool         backend_sampling = false;
};

static void init_env(mtp_env & env, gguf_context * meta, const env_cfg & cfg) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    mp.load_mtp          = true;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;

    size_t seed = 31;
    env.model.reset(llama_model_init_from_user(meta, set_tensor_data, &seed, mp));
    GGML_ASSERT(env.model && "model init failed");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = cfg.n_ctx;
    cp.n_batch         = 128;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = 1;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    env.ctx_tgt.reset(llama_init_from_model(env.model.get(), cp));
    GGML_ASSERT(env.ctx_tgt && "target context failed");

    llama_context_params cd = cp;
    cd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cd.n_rs_seq = 0;
    cd.type_k   = cfg.type_kv_dft;
    cd.type_v   = cfg.type_kv_dft;
    if (cfg.census) {
        cd.cb_eval           = census_cb;
        cd.cb_eval_user_data = &env.census;
    }
    setenv("LLAMA_MTP_CATCHUP_ATTN", cfg.catchup_attn, 1);
    env.ctx_dft.reset(llama_init_from_model(env.model.get(), cd));
    unsetenv("LLAMA_MTP_CATCHUP_ATTN");
    GGML_ASSERT(env.ctx_dft && "MTP context failed");

    env.params.types                  = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    env.params.draft.n_max            = 4;
    env.params.draft.n_min            = 0;
    env.params.draft.p_min            = 0.0f;
    env.params.draft.backend_sampling = cfg.backend_sampling;
    env.params.draft.ctx_tgt          = env.ctx_tgt.get();
    env.params.draft.ctx_dft          = env.ctx_dft.get();
    env.spec = common_speculative_init(env.params, 1);
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

// target decode (every row an output: the full continuation logits) + the MTP catch-up; returns the target logits
static std::vector<float> feed(mtp_env & env, const std::vector<llama_token> & toks, llama_pos pos0, double * t_catchup_ms = nullptr) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], pos0 + (llama_pos) i, { 0 }, true);
    }
    GGML_ASSERT(llama_decode(env.ctx_tgt.get(), b) == 0);
    std::vector<float> logits((size_t) toks.size() * N_VOCAB);
    for (size_t i = 0; i < toks.size(); ++i) {
        const float * l = llama_get_logits_ith(env.ctx_tgt.get(), (int32_t) i);
        std::memcpy(logits.data() + i * N_VOCAB, l, N_VOCAB * sizeof(float));
    }
    env.census.active = true;
    const auto t0 = std::chrono::steady_clock::now();
    GGML_ASSERT(common_speculative_process(env.spec, b));
    llama_synchronize(env.ctx_dft.get());
    const auto t1 = std::chrono::steady_clock::now();
    env.census.active = false;
    if (t_catchup_ms) {
        *t_catchup_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    llama_batch_free(b);
    return logits;
}

struct snap {
    std::vector<uint8_t> dft;   // draft host state stream, seq 0, flags 0
    std::vector<uint8_t> carry; // common_speculative_get_state
};

static snap snapshot(mtp_env & env) {
    snap s;
    s.dft.resize(llama_state_seq_get_size_ext(env.ctx_dft.get(), 0, 0));
    const size_t n = llama_state_seq_get_data_ext(env.ctx_dft.get(), s.dft.data(), s.dft.size(), 0, 0);
    GGML_ASSERT(n == s.dft.size());
    GGML_ASSERT(common_speculative_get_state(env.spec, 0, s.carry));
    return s;
}

struct draft_out {
    std::vector<llama_token> toks;
    std::vector<float>       logits;
};

static draft_out draft(mtp_env & env, llama_pos n_past, llama_token id_last) {
    draft_out out;
    auto & dp = common_speculative_get_draft_params(env.spec, 0);
    dp.drafting = true;
    dp.n_max    = -1;
    dp.n_past   = n_past;
    dp.id_last  = id_last;
    dp.prompt   = nullptr;
    dp.result   = &out.toks;
    common_speculative_draft(env.spec);
    // with backend sampling the raw draft logits may not reach the host: then the tokens alone are compared
    const float * l = env.params.draft.backend_sampling ? nullptr : llama_get_logits_ith(env.ctx_dft.get(), -1);
    if (l != nullptr) {
        out.logits.assign(l, l + N_VOCAB);
    }
    llama_memory_seq_rm(llama_get_memory(env.ctx_dft.get()), 0, n_past, -1);
    return out;
}

static int n_fail = 0;
static int n_ok   = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        printf("FAIL %s\n", what.c_str());
        n_fail++;
    } else {
        n_ok++;
    }
}

template <typename T>
static bool eq(const std::vector<T> & a, const std::vector<T> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

static void cmp_snap(mtp_env & a, mtp_env & b, const std::string & what) {
    const snap sa = snapshot(a);
    const snap sb = snapshot(b);
    check(eq(sa.dft, sb.dft),     what + ": draft state stream (" + std::to_string(sa.dft.size()) + " B)");
    check(eq(sa.carry, sb.carry), what + ": speculative carry (" + std::to_string(sa.carry.size()) + " B)");
}

static int argmax(const float * l) {
    return (int) (std::max_element(l, l + N_VOCAB) - l);
}

// one paired replay: catch-ups of every size class, then verification rounds with rejection, checkpoint restore and replay
static void run_pair(gguf_context * meta, ggml_type type_kv, bool backend_sampling) {
    const std::string tn_s = std::string(ggml_type_name(type_kv)) + (backend_sampling ? "+bs" : "");
    const char * tn = tn_s.c_str();
    mtp_env A, B;
    init_env(A, meta, { "1", type_kv, 512, false, backend_sampling });
    init_env(B, meta, { "0", type_kv, 512, false, backend_sampling });

    common_speculative_begin(A.spec, 0, {});
    common_speculative_begin(B.spec, 0, {});

    // 1 | 31 32 33 (the eh_proj flat threshold) | 64 (a full ubatch) | 101 (64 + a final partial 37) | 7
    const std::vector<int> sizes = { 1, 31, 32, 33, 64, 101, 7 };
    llama_pos pos = 0;
    int n_catchups = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto toks = tokens(sizes[i], 100 + (uint32_t) i);
        const auto la = feed(A, toks, pos);
        const auto lb = feed(B, toks, pos);
        const std::string what = std::string(tn) + " catch-up " + std::to_string(sizes[i]) + " @" + std::to_string(pos);
        check(eq(la, lb), what + ": target logits");
        cmp_snap(A, B, what);
        pos += sizes[i];
        n_catchups++;
    }

    // verification rounds (server order): draft -> per-round checkpoint -> verify [id_last, draft] (catch-up) ->
    // acceptance (0..4 rows) -> on rejection restore the checkpoint + carrier, roll back, replay the accepted rows
    llama_token id_last = 5;
    int n_rounds = 0, n_reject = 0, n_acc_total = 0;
    for (int r = 0; r < 5; ++r) {
        const std::string what = std::string(tn) + " round " + std::to_string(r) + " @" + std::to_string(pos);
        const draft_out da = draft(A, pos, id_last);
        const draft_out db = draft(B, pos, id_last);
        check(da.toks == db.toks && eq(da.logits, db.logits), what + ": draft tokens + logits");

        common_prompt_checkpoint ca, cb;
        const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
        mtp_env * envs[2] = { &A, &B };
        common_prompt_checkpoint * cks[2] = { &ca, &cb };
        for (int k = 0; k < 2; ++k) {
            auto * mem = llama_get_memory(envs[k]->ctx_tgt.get());
            cks[k]->update_pos(pos, llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0));
            cks[k]->update_tgt(envs[k]->ctx_tgt.get(), 0, flags);
            cks[k]->data_spec.clear();
            GGML_ASSERT(common_speculative_get_state(envs[k]->spec, 0, cks[k]->data_spec));
        }

        std::vector<llama_token> verify = { id_last };
        verify.insert(verify.end(), da.toks.begin(), da.toks.end());
        GGML_ASSERT(!da.toks.empty());
        const auto va = feed(A, verify, pos);
        const auto vb = feed(B, verify, pos);
        check(eq(va, vb), what + ": verification logits (" + std::to_string(verify.size()) + " rows)");
        cmp_snap(A, B, what + " after the verification catch-up");
        n_catchups++;

        // acceptance: the target's argmax along the draft
        auto accept_n = [&](const std::vector<float> & v) {
            size_t n = 0;
            while (n < da.toks.size() && argmax(v.data() + n * N_VOCAB) == da.toks[n]) {
                n++;
            }
            return n;
        };
        const size_t n_arg  = accept_n(va);
        const size_t n_argb = accept_n(vb);
        check(n_arg == n_argb, what + ": acceptance decision (target argmax)");
        // random weights never agree with the draft, so the rounds force 0, 1, 2, 3, 4 accepted rows (the last one,
        // with a full draft, accepts everything: no rollback); the bonus token is the target's argmax at that row
        const size_t n_acc = std::min((size_t) r, da.toks.size());
        const llama_token x = argmax(va.data() + n_acc * N_VOCAB);
        n_acc_total += (int) n_acc;

        if (n_acc < da.toks.size()) {
            n_reject++;
            const std::vector<llama_token> committed(verify.begin(), verify.begin() + n_acc + 1);
            for (int k = 0; k < 2; ++k) {
                cks[k]->load_tgt(envs[k]->ctx_tgt.get(), 0, flags);
                cks[k]->load_dft(envs[k]->ctx_dft.get(), 0, flags);
                common_memory mem;
                mem.init(envs[k]->ctx_tgt.get(), envs[k]->ctx_dft.get());
                mem.seq_rm(0, cks[k]->pos_max + 1, -1);
                common_speculative_set_state(envs[k]->spec, 0, cks[k]->data_spec);
            }
            cmp_snap(A, B, what + " after the checkpoint restore");
            const auto ra = feed(A, committed, pos);
            const auto rb = feed(B, committed, pos);
            check(eq(ra, rb), what + ": replay logits");
            cmp_snap(A, B, what + " after the replay catch-up");
            n_catchups++;
        }
        common_speculative_accept(A.spec, 0, (uint16_t) n_acc);
        common_speculative_accept(B.spec, 0, (uint16_t) n_acc);
        cmp_snap(A, B, what + " after accept");

        pos    += (llama_pos) n_acc + 1;
        id_last = x;
        n_rounds++;
    }
    const draft_out fa = draft(A, pos, id_last);
    const draft_out fb = draft(B, pos, id_last);
    check(fa.toks == fb.toks && eq(fa.logits, fb.logits), std::string(tn) + " final draft tokens + logits");

    printf("draft KV %-7s: %d catch-ups (sizes 1,31,32,33,64,64+37,7 + %d verify rounds, %d rolled back, %d drafted tokens accepted): "
           "draft state + carry + target logits identical old vs new: %s\n",
           tn, n_catchups, n_rounds, n_reject, n_acc_total, n_fail == 0 ? "yes" : "NO");
}

// the op census of one catch-up ubatch of 64 rows at pos 64 (after a first ubatch)
static std::map<std::string, int> census(gguf_context * meta, ggml_type type_kv, const char * knob, int * total) {
    mtp_env E;
    init_env(E, meta, { knob, type_kv, 512, true });
    common_speculative_begin(E.spec, 0, {});
    feed(E, tokens(64, 7), 0);
    E.census.n.clear();
    E.census.total = 0;
    feed(E, tokens(64, 8), 64);
    *total = E.census.total;
    return E.census.n;
}

static void op_table(gguf_context * meta, ggml_type type_kv) {
    int t_old = 0, t_new = 0;
    auto o = census(meta, type_kv, "1", &t_old);
    auto n = census(meta, type_kv, "0", &t_new);
    std::map<std::string, int> all = o;
    for (auto & kv : n) {
        all[kv.first] += 0;
    }
    printf("op census, one 64-row catch-up ubatch, draft KV %s: op | old (ATTN=1) | new (ATTN=0)\n", ggml_type_name(type_kv));
    for (auto & kv : all) {
        printf("  %-16s %5d %5d\n", kv.first.c_str(), o[kv.first], n[kv.first]);
    }
    printf("  %-16s %5d %5d\n", "TOTAL", t_old, t_new);
    const std::string tn = ggml_type_name(type_kv);
    check(n["SET_ROWS"] == 2 && o["SET_ROWS"] == 2,      tn + " census: K + V cache writes in both graphs");
    check(n["FLASH_ATTN_EXT"] == 0 && n["SOFT_MAX"] == 0, tn + " census: no attention op in the new graph");
    check(o["FLASH_ATTN_EXT"] + o["SOFT_MAX"] > 0,        tn + " census: the old graph attends");
    check(n["MUL_MAT_ID"] == 0 && o["MUL_MAT_ID"] > 0,    tn + " census: no MoE in the new graph");
    check(n["ROPE"] == 1 && o["ROPE"] == 2,                tn + " census: K rope only (old: Q + K)");
    if (type_kv != GGML_TYPE_F16) {
        // the cache rotation (llama_mul_mat_hadamard): old = Q, K, V and the output's V inverse; new = K, V
        check(n["MUL_MAT"] < o["MUL_MAT"], tn + " census: fewer matmuls");
    }
}

static void timing(gguf_context * meta) {
    const int n_ub = 32; // 32 ubatches of 64 rows = 2048 cells
    double ms[2] = { 0, 0 };
    const char * knobs[2] = { "1", "0" };
    for (int k = 0; k < 2; ++k) {
        mtp_env E;
        init_env(E, meta, { knobs[k], GGML_TYPE_F16, 4096, false });
        common_speculative_begin(E.spec, 0, {});
        for (int u = 0; u < n_ub; ++u) {
            feed(E, tokens(64, 1000 + u), u * 64, &ms[k]);
        }
    }
    printf("timing (CPU, 4 threads, tiny model, indication only): MTP catch-up per 64-row ubatch over a 0..2048 fill: "
           "old %.3f ms, new %.3f ms (%.1f %% less)\n", ms[0] / n_ub, ms[1] / n_ub, 100.0 * (1.0 - ms[1] / ms[0]));
}

int main() {
    llama_log_set(quiet_log, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    gguf_context_ptr meta = make_meta(4096);

    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        run_pair(meta.get(), t, false);
        op_table(meta.get(), t);
    }
    run_pair(meta.get(), GGML_TYPE_F16, true); // backend draft sampling: the samplers see no zero-output graph now

    timing(meta.get());

    llama_backend_free();

    if (n_fail) {
        printf("test-mtp-catchup: %d failures (%d checks ok)\n", n_fail, n_ok);
        return 1;
    }
    printf("test-mtp-catchup: OK (%d checks)\n", n_ok);
    return 0;
}
