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
//
// [TAG_MTP_QSA] sparse MTP draft (0159, stage 3): the ctests test-mtp-catchup-qsa / -qsa-mu0 run this binary with
// LLAMA_MTP_QSA=1 (+ the prod QSA env, see tests/CMakeLists.txt). The draft memory is then a llama_memory_hybrid_idx in
// recr_none mode, whose state stream carries the attention K/V AND the indexer K (q8_0) whatever the flags, so every
// draft-state comparison above also covers the indexer cache (Codex B2: the kv-only catch-up writes the indexer K
// cache-only). The op census then expects three cache writes (K, V, indexer K) in the new graph, and the draft step
// of the paired replay runs the real QSA selection (TOP_K in its census; top_k 8, ratio 4 => width 11 < n_kv).

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"
#include "server-context.h" // [TAG_MULTISLOT] item 4a: the routing-class planner

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

static const uint32_t N_VOCAB = 128;

// [TAG_MTP_QSA] set by the ctest variants; read once (the library caches it the same way)
static bool mtp_qsa_mode() {
    const char * s = getenv("LLAMA_MTP_QSA");
    return s != nullptr && atoi(s) != 0;
}

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

// [TAG_MULTISLOT] set in a disposable negative-case process: the first draft graph node reports itself, so the parent
// can require the [mtp-contiguity] diagnostic BEFORE any draft decode
static bool g_neg_child = false;

static bool census_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * c = (op_census *) ud;
    if (ask && g_neg_child) {
        fprintf(stderr, "NEG-CHILD: draft decode reached\n");
        g_neg_child = false;
    }
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
    if (level >= GGML_LOG_LEVEL_WARN || strstr(text, "LLAMA_MTP_CATCHUP_ATTN") != nullptr || strstr(text, "mtp_qsa") != nullptr) {
        fputs(text, stderr);
    }
}

struct env_cfg {
    const char * catchup_attn;   // the knob for ctx_dft
    ggml_type    type_kv_dft;
    uint32_t     n_ctx;
    bool         census;
    bool         backend_sampling = false;
    float        p_min            = 0.0f;
    int          n_min            = 0;
    uint32_t     n_seq            = 1;   // [TAG_MULTISLOT] n_seq_max of both contexts + the speculative n_seq
};

static void init_env(mtp_env & env, gguf_context * meta, const env_cfg & cfg) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    mp.load_mtp          = true;
    // TEST_MTP_GPU=1: the first GPU (else integrated GPU) device with every layer offloaded (a Vulkan / GPU build check), default the CPU
    static const bool use_gpu = getenv("TEST_MTP_GPU") != nullptr && atoi(getenv("TEST_MTP_GPU")) != 0;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(use_gpu ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    if (use_gpu && devs[0] == nullptr) {
        devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); // an integrated GPU (e.g. the 780M)
    }
    GGML_ASSERT(devs[0] != nullptr && "TEST_MTP_GPU=1 but no GPU device");
    mp.devices = devs;
    if (use_gpu) {
        mp.n_gpu_layers = 999;
    }

    size_t seed = 31;
    env.model.reset(llama_model_init_from_user(meta, set_tensor_data, &seed, mp));
    GGML_ASSERT(env.model && "model init failed");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = cfg.n_ctx;
    cp.n_batch         = 128;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = cfg.n_seq;
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
    common_set_env("LLAMA_MTP_CATCHUP_ATTN", cfg.catchup_attn);   // portable (_putenv_s on _WIN32; sol r1)
    env.ctx_dft.reset(llama_init_from_model(env.model.get(), cd));
    common_set_env("LLAMA_MTP_CATCHUP_ATTN", "");                 // empty = unset
    GGML_ASSERT(env.ctx_dft && "MTP context failed");

    env.params.types                  = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    env.params.draft.n_max            = 4;
    env.params.draft.n_min            = cfg.n_min;
    env.params.draft.p_min            = cfg.p_min;
    env.params.draft.backend_sampling = cfg.backend_sampling;
    env.params.draft.ctx_tgt          = env.ctx_tgt.get();
    env.params.draft.ctx_dft          = env.ctx_dft.get();
    env.spec = common_speculative_init(env.params, cfg.n_seq);
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

static draft_out draft(mtp_env & env, llama_pos n_past, llama_token id_last, llama_seq_id seq = 0) {
    draft_out out;
    auto & dp = common_speculative_get_draft_params(env.spec, seq);
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
    llama_memory_seq_rm(llama_get_memory(env.ctx_dft.get()), seq, n_past, -1);
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

    const size_t n_state = snapshot(A).dft.size();
    printf("%sdraft KV %-7s (draft state %zu B): %d catch-ups (sizes 1,31,32,33,64,64+37,7 + %d verify rounds, %d rolled back, %d drafted tokens accepted): "
           "draft state + carry + target logits identical old vs new: %s\n",
           mtp_qsa_mode() ? "[sparse MTP] " : "", tn, n_state, n_catchups, n_rounds, n_reject, n_acc_total, n_fail == 0 ? "yes" : "NO");
}

// the op census of one catch-up ubatch of 64 rows at pos 64 (after a first ubatch)
static std::map<std::string, int> census(gguf_context * meta, ggml_type type_kv, const char * knob, int * total,
        std::map<std::string, int> * draft_ops = nullptr) {
    mtp_env E;
    init_env(E, meta, { knob, type_kv, 512, true });
    common_speculative_begin(E.spec, 0, {});
    feed(E, tokens(64, 7), 0);
    E.census.n.clear();
    E.census.total = 0;
    feed(E, tokens(64, 8), 64);
    *total = E.census.total;
    auto res = E.census.n;
    if (draft_ops) {
        // [TAG_MTP_QSA] one draft step at depth 128 (the selection's width 11 < n_kv): the ops its graphs ran
        E.census.n.clear();
        E.census.active = true;
        draft(E, 128, 5);
        E.census.active = false;
        *draft_ops = E.census.n;
    }
    return res;
}

static void op_table(gguf_context * meta, ggml_type type_kv) {
    int t_old = 0, t_new = 0;
    std::map<std::string, int> d;
    auto o = census(meta, type_kv, "1", &t_old);
    auto n = census(meta, type_kv, "0", &t_new, &d);
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
    if (mtp_qsa_mode()) {
        // [TAG_MTP_QSA] + the indexer K (cache_only); the old graph may add the pooled-store writes (MAX_UBATCH=0)
        check(n["SET_ROWS"] == 3 && o["SET_ROWS"] >= 3, tn + " census: K + V + indexer K cache writes in both graphs");
        check(d["TOP_K"] > 0,                            tn + " census: the draft step runs the QSA selection (TOP_K)");
        printf("  draft step (sparse MTP): TOP_K %d, SET_ROWS %d\n", d["TOP_K"], d["SET_ROWS"]);
    } else {
    check(n["SET_ROWS"] == 2 && o["SET_ROWS"] == 2,      tn + " census: K + V cache writes in both graphs");
    }
    check(n["FLASH_ATTN_EXT"] == 0 && n["SOFT_MAX"] == 0, tn + " census: no attention op in the new graph");
    check(o["FLASH_ATTN_EXT"] + o["SOFT_MAX"] > 0,        tn + " census: the old graph attends");
    check(n["MUL_MAT_ID"] == 0 && o["MUL_MAT_ID"] > 0,    tn + " census: no MoE in the new graph");
    // sparse MTP, old graph: the indexer's pooled keys / query may add ropes (MAX_UBATCH=0)
    check(n["ROPE"] == 1 && (mtp_qsa_mode() ? o["ROPE"] >= 2 : o["ROPE"] == 2), tn + " census: K rope only (old: Q + K)");
    if (type_kv != GGML_TYPE_F16) {
        // the cache rotation (llama_mul_mat_hadamard): old = Q, K, V and the output's V inverse; new = K, V
        check(n["MUL_MAT"] < o["MUL_MAT"], tn + " census: fewer matmuls");
    }
}

static void timing(gguf_context * meta) {
    const int n_ub = 32; // 32 ubatches of 64 rows = 2048 cells
    double ms[2] = { 0, 0 };
    int reused[2] = { 0, 0 };
    const char * knobs[2] = { "1", "0" };
    for (int k = 0; k < 2; ++k) {
        mtp_env E;
        init_env(E, meta, { knobs[k], GGML_TYPE_F16, 4096, false });
        common_speculative_begin(E.spec, 0, {});
        for (int u = 0; u < n_ub; ++u) {
            feed(E, tokens(64, 1000 + u), u * 64, &ms[k]);
        }
        reused[k] = llama_perf_context(E.ctx_dft.get()).n_reused;
    }
    // [TAG_MTP_QSA] the kv-only catch-up graph is reused across ubatches in both modes (the sparse memory's forced
    // cache-only indexer input does not veto the reuse once the depth passes the selection width)
    printf("graph reuse, draft context, %d catch-up ubatches: old %d, new %d\n", n_ub, reused[0], reused[1]);
    // the draft KV's n_kv grows in steps of 256 cells: one rebuild per step (8 over the fill) plus the first graph
    check(reused[1] >= n_ub - 2048/256 - 1, "new catch-up graphs reused (all but one per 256-cell n_kv step)");
    printf("timing (CPU, 4 threads, tiny model, indication only): MTP catch-up per 64-row ubatch over a 0..2048 fill: "
           "old %.3f ms, new %.3f ms (%.1f %% less)\n", ms[0] / n_ub, ms[1] / n_ub, 100.0 * (1.0 - ms[1] / ms[0]));
}

// [TAG_MTP_DRAFT_KV_TYPE] (v51) a greedy speculative generation loop in the server's call order: prefill, then rounds
// of draft (n tokens) -> per-round checkpoint -> verify [id_last, draft] -> accept the target's argmax prefix -> on a
// rejection restore + roll back + replay the committed rows; n = 0 is a plain one-token decode (the true-zero round of
// --spec-draft-policy tput: no draft call, process() still runs). Every committed token is recorded with the target
// logits row it was taken from, so two runs compare per committed token whatever their round structure.
struct loop_result {
    std::vector<llama_token> committed;
    std::vector<float>       logits;   // N_VOCAB per committed token
    std::vector<int>         width;    // the target batch width each committed token's logits row came from
    std::vector<uint8_t>     tgt_state; // the target's full state stream at the end
    std::vector<std::vector<llama_token>> drafts; // the proposals, per round with a draft call
    std::vector<std::vector<float>>       dlogits; // the last draft step's logits, per round with a draft call
    int n_rounds = 0, n_drafted = 0, n_accepted = 0, n_zero = 0;
    std::map<int, int> hist; // chosen draft length -> rounds
};

// returns the draft length for the next round (0 = no draft call); observe() gets the round's outcome
struct loop_policy {
    virtual ~loop_policy() = default;
    virtual int  decide() { return 4; }
    virtual void observe(int /*n_req*/, int /*n_proposed*/, int /*n_acc*/) {}
};

static loop_result run_loop(mtp_env & env, loop_policy & pol, size_t n_commit) {
    loop_result res;
    common_speculative_begin(env.spec, 0, {});
    const auto prompt = tokens(40, 4242);
    const auto lp = feed(env, prompt, 0);
    llama_pos   pos     = (llama_pos) prompt.size();
    llama_token id_last = argmax(lp.data() + (prompt.size() - 1) * N_VOCAB);
    res.committed.push_back(id_last);
    res.logits.insert(res.logits.end(), lp.end() - N_VOCAB, lp.end());
    res.width.push_back((int) prompt.size());

    const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    while (res.committed.size() < n_commit) {
        const int n_req = pol.decide();
        res.hist[n_req]++;
        res.n_rounds++;
        if (n_req == 0) {
            const auto v = feed(env, { id_last }, pos);
            id_last = argmax(v.data());
            res.committed.push_back(id_last);
            res.logits.insert(res.logits.end(), v.begin(), v.end());
            res.width.push_back(1);
            pos += 1;
            res.n_zero++;
            continue;
        }
        draft_out d;
        {
            auto & dp = common_speculative_get_draft_params(env.spec, 0);
            dp.drafting = true;
            dp.n_max    = n_req;
            dp.n_past   = pos;
            dp.id_last  = id_last;
            dp.prompt   = nullptr;
            dp.result   = &d.toks;
            common_speculative_draft(env.spec);
            dp.drafting = false;
            const float * l = llama_get_logits_ith(env.ctx_dft.get(), -1);
            res.dlogits.emplace_back(l, l + N_VOCAB);
            llama_memory_seq_rm(llama_get_memory(env.ctx_dft.get()), 0, pos, -1);
        }
        res.drafts.push_back(d.toks);
        res.n_drafted += (int) d.toks.size();
        if (d.toks.empty()) {
            pol.observe(n_req, 0, 0);
            const auto v = feed(env, { id_last }, pos);
            id_last = argmax(v.data());
            res.committed.push_back(id_last);
            res.logits.insert(res.logits.end(), v.begin(), v.end());
            res.width.push_back(1);
            pos += 1;
            continue;
        }
        common_prompt_checkpoint ck;
        auto * mem = llama_get_memory(env.ctx_tgt.get());
        ck.update_pos(pos, llama_memory_seq_pos_min(mem, 0), llama_memory_seq_pos_max(mem, 0));
        ck.update_tgt(env.ctx_tgt.get(), 0, flags);
        GGML_ASSERT(common_speculative_get_state(env.spec, 0, ck.data_spec));

        std::vector<llama_token> verify = { id_last };
        verify.insert(verify.end(), d.toks.begin(), d.toks.end());
        const auto v = feed(env, verify, pos);
        size_t n_acc = 0;
        while (n_acc < d.toks.size() && argmax(v.data() + n_acc * N_VOCAB) == d.toks[n_acc]) {
            n_acc++;
        }
        for (size_t i = 0; i <= n_acc && res.committed.size() < n_commit + 8; ++i) {
            res.committed.push_back(argmax(v.data() + i * N_VOCAB));
            res.logits.insert(res.logits.end(), v.begin() + i * N_VOCAB, v.begin() + (i + 1) * N_VOCAB);
            res.width.push_back((int) verify.size());
        }
        res.n_accepted += (int) n_acc;
        pol.observe(n_req, (int) d.toks.size(), (int) n_acc);
        if (n_acc < d.toks.size()) {
            const std::vector<llama_token> committed(verify.begin(), verify.begin() + n_acc + 1);
            ck.load_tgt(env.ctx_tgt.get(), 0, flags);
            common_memory cm;
            cm.init(env.ctx_tgt.get(), env.ctx_dft.get());
            cm.seq_rm(0, ck.pos_max + 1, -1);
            common_speculative_set_state(env.spec, 0, ck.data_spec);
            feed(env, committed, pos);
        }
        common_speculative_accept(env.spec, 0, (uint16_t) n_acc);
        pos    += (llama_pos) n_acc + 1;
        id_last = res.committed.back();
    }
    // compare the first n_commit tokens only (a round may overshoot)
    res.committed.resize(n_commit);
    res.logits.resize(n_commit * N_VOCAB);
    res.width.resize(n_commit);
    res.tgt_state.resize(llama_state_seq_get_size_ext(env.ctx_tgt.get(), 0, 0));
    GGML_ASSERT(llama_state_seq_get_data_ext(env.ctx_tgt.get(), res.tgt_state.data(), res.tgt_state.size(), 0, 0) == res.tgt_state.size());
    return res;
}

// first index where two committed streams differ (token or logits row), or -1
// same_class: compare the logits rows only where both came from the same width class (1 = a plain decode, or >= 2)
static int first_diff(const loop_result & a, const loop_result & b, bool same_class = false) {
    for (size_t i = 0; i < std::min(a.committed.size(), b.committed.size()); ++i) {
        const bool cmp_row = !same_class || (a.width[i] == 1) == (b.width[i] == 1);
        if (a.committed[i] != b.committed[i] || (cmp_row &&
                memcmp(a.logits.data() + i * N_VOCAB, b.logits.data() + i * N_VOCAB, N_VOCAB * sizeof(float)) != 0)) {
            return (int) i;
        }
    }
    return a.committed.size() == b.committed.size() ? -1 : (int) std::min(a.committed.size(), b.committed.size());
}

// [TAG_MTP_DRAFT_KV_TYPE] draft KV f16 vs q8_0 (--spec-draft-type-k/-v): the target side is exact (committed tokens, their
// logits rows, the target state stream), the draft state layout (descriptor) differs, the proposals may differ
static void draft_kv_type_pair(gguf_context * meta) {
    mtp_env A, B;
    init_env(A, meta, { "", GGML_TYPE_F16,  512, false });
    init_env(B, meta, { "", GGML_TYPE_Q8_0, 512, false });
    check(llama_state_seq_descriptor(A.ctx_tgt.get()) == llama_state_seq_descriptor(B.ctx_tgt.get()),
          "draft KV type: the target descriptor is unchanged");
    check(llama_state_seq_descriptor(A.ctx_dft.get()) != llama_state_seq_descriptor(B.ctx_dft.get()),
          "draft KV type: the draft descriptor (state layout, disk-cache compat key) differs f16 vs q8_0");
    loop_policy fixed;
    const auto ra = run_loop(A, fixed, 64);
    const auto rb = run_loop(B, fixed, 64);
    const int fd = first_diff(ra, rb);
    check(fd < 0, "draft KV f16 vs q8_0: committed tokens + their target logits identical (first diff at " + std::to_string(fd) + ")");
    check(eq(ra.tgt_state, rb.tgt_state), "draft KV f16 vs q8_0: target state stream identical");
    int same = 0, total = 0;
    float dmax = 0.0f;
    for (size_t r = 0; r < std::min(ra.dlogits.size(), rb.dlogits.size()); ++r) {
        for (size_t i = 0; i < N_VOCAB; ++i) {
            dmax = std::max(dmax, std::fabs(ra.dlogits[r][i] - rb.dlogits[r][i]));
        }
    }
    for (size_t r = 0; r < std::min(ra.drafts.size(), rb.drafts.size()); ++r) {
        for (size_t i = 0; i < std::min(ra.drafts[r].size(), rb.drafts[r].size()); ++i) {
            same += ra.drafts[r][i] == rb.drafts[r][i];
            total++;
        }
    }
    printf("%sdraft KV f16 vs q8_0, greedy loop n=4, 64 committed tokens: target identical %s; acceptance f16 %d/%d, q8_0 %d/%d; "
           "proposal agreement %d/%d, last-step draft logits max |diff| %.3g\n", mtp_qsa_mode() ? "[sparse MTP] " : "",
           fd < 0 && eq(ra.tgt_state, rb.tgt_state) ? "yes" : "NO",
           ra.n_accepted, ra.n_drafted, rb.n_accepted, rb.n_drafted, same, total, (double) dmax);
}

// [TAG_SPEC_DRAFT_POLICY] the server's tput policy on a fixed cost model: the per-position acceptance EMA (alpha 1/32,
// optimistic start, an empty draft or a short draft = a failure at the stop position) -> common_spec_tput_best_n ->
// common_spec_cooldown, exactly as server_context_impl::spec_tput_n / spec_acc_update use them
struct tput_policy : loop_policy {
    common_spec_tput_cost c;
    common_spec_cooldown  cd;
    std::vector<float>    ema;
    std::vector<uint32_t> nobs;
    int n_max = 4;
    tput_policy() {
        c.d0 = 1.0; c.d1 = 2.2; c.v0 = 25.0; c.v1 = 1.5; c.c0 = 0.8; c.c1 = 0.2; c.host = 1.2;
    }
    int decide() override {
        return std::min(cd.decide(common_spec_tput_best_n(common_spec_tput_p_eff(ema, nobs), n_max, c)), n_max);
    }
    void observe(int n_req, int n_proposed, int n_acc) override {
        if (ema.empty()) {
            ema.assign(n_max, 1.0f);
            nobs.assign(n_max, 0);
        }
        const int n_obs = n_proposed < n_req ? n_proposed + 1 : n_proposed;
        for (int i = 0; i < n_obs && i <= n_acc && i < (int) ema.size(); ++i) {
            ema[i] += (1.0f/32.0f) * ((i < n_acc ? 1.0f : 0.0f) - ema[i]);
            nobs[i]++;
        }
        cd.observe(n_proposed, n_acc);
    }
};

static std::string hist_str(const std::map<int, int> & h) {
    std::string s;
    for (const auto & kv : h) {
        s += (s.empty() ? "" : " ") + std::to_string(kv.first) + ":" + std::to_string(kv.second);
    }
    return s;
}

// [TAG_SPEC_DRAFT_POLICY] policy off (n = 4 every round) vs tput (true-zero rounds, probes, other widths): the same
// committed tokens with the same target logits rows, compared per committed token (the round structure and so the
// verify widths differ: a composition class); then tput with a q8_0 draft KV
static void draft_policy_e2e(gguf_context * meta) {
    const size_t n_commit = 96;
    mtp_env A, B, C;
    init_env(A, meta, { "", GGML_TYPE_F16,  512, false });
    init_env(B, meta, { "", GGML_TYPE_F16,  512, false });
    init_env(C, meta, { "", GGML_TYPE_Q8_0, 512, false });
    loop_policy off;
    tput_policy tb, tc;
    const auto ra = run_loop(A, off, n_commit);
    const auto rb = run_loop(B, tb,  n_commit);
    const auto rc = run_loop(C, tc,  n_commit);
    // a plain one-token decode is its own class on the CPU (the n = 1 output matmul path; measured below): the rows are
    // compared bitwise within a class, the committed tokens and the target state everywhere
    const int fb = first_diff(ra, rb, true);
    const int fc = first_diff(ra, rc, true);
    check(fb < 0, "policy off vs tput: committed tokens + target logits identical (first diff at " + std::to_string(fb) + ")");
    check(eq(ra.tgt_state, rb.tgt_state), "policy off vs tput: target state stream identical");
    check(first_diff(rb, rc) < 0, "tput f16 vs tput q8_0 draft KV: committed tokens + target logits identical (same widths)");
    check(fc < 0, "policy off vs tput + q8_0 draft KV: committed tokens + target logits identical (first diff at " + std::to_string(fc) + ")");
    check(eq(ra.tgt_state, rc.tgt_state), "policy off vs tput + q8_0 draft KV: target state stream identical");
    check(rb.n_zero > 0 && rb.hist.count(1) > 0, "tput: true-zero rounds and probes happened");
    // the width class on this backend: fixed n = 0..3 vs n = 4 (verify widths 1..4 vs 5): token mismatches, max |dlogit|
    const auto dist = [](const loop_result & x, const loop_result & y, int & n_tok) {
        float m = 0.0f;
        n_tok = 0;
        for (size_t i = 0; i < x.committed.size(); ++i) {
            n_tok += x.committed[i] != y.committed[i];
            for (size_t j = 0; j < N_VOCAB; ++j) {
                m = std::max(m, std::fabs(x.logits[i*N_VOCAB + j] - y.logits[i*N_VOCAB + j]));
            }
        }
        return m;
    };
    {
        int n_tok = 0;
        const float m = dist(ra, rb, n_tok);
        printf("  off vs tput f16: %d committed tokens differ, max |dlogit| %.3g\n", n_tok, (double) m);
    }
    struct fixed_n : loop_policy { int n; int decide() override { return n; } };
    for (int n = 0; n < 4; ++n) {
        mtp_env E;
        init_env(E, meta, { "", GGML_TYPE_F16, 512, false });
        fixed_n f; f.n = n;
        const auto rf = run_loop(E, f, n_commit);
        int n_tok = 0;
        const float m = dist(ra, rf, n_tok);
        check(n_tok == 0 && eq(ra.tgt_state, rf.tgt_state) && first_diff(ra, rf, true) < 0,
              "verify width " + std::to_string(n + 1) + " vs 5: tokens, target state and same-class rows identical");
        printf("  width class: verify width %d vs 5: first diff at %d, %d tokens differ, max |dlogit| %.3g, target state %s\n",
               n + 1, first_diff(ra, rf), n_tok, (double) m, eq(ra.tgt_state, rf.tgt_state) ? "identical" : "differs");
    }
    const auto line = [](const char * name, const loop_result & r) {
        printf("  %-14s rounds %3d (zero %3d), drafted %3d, accepted %3d, n* distribution %s\n", name, r.n_rounds, r.n_zero,
               r.n_drafted, r.n_accepted, hist_str(r.hist).c_str());
    };
    printf("%sdraft policy e2e, %zu committed tokens: target identical off vs tput %s, off vs tput+q8_0 %s\n",
           mtp_qsa_mode() ? "[sparse MTP] " : "", n_commit, fb < 0 && eq(ra.tgt_state, rb.tgt_state) ? "yes" : "NO",
           fc < 0 && eq(ra.tgt_state, rc.tgt_state) ? "yes" : "NO");
    line("off (n=4)", ra);
    line("tput f16", rb);
    line("tput q8_0", rc);
}

// [TAG_SPEC_DRAFT_POLICY] (v51 r1, Codex D Minor) the MTP drafter reports the positions it sampled (n_attempted, a p_min
// stop included) and the tokens it drafted before the n_min rule (n_kept): the server prices the draft call by the
// attempted length and observes a stop at its real position
static void draft_attempt_report(gguf_context * meta) {
    // keep: v51 r2 (Hermes) a tput cooldown probe (keep_short) is never cleared by the n_min rule
    struct tc { float p_min; int n_min; int n_max; bool keep; int att; int kept; size_t out; const char * what; };
    for (const tc & t : { tc{ 0.0f,  0, 4, false, 4, 4, 4, "full draft: attempted 4, kept 4" },
                          tc{ 1.01f, 0, 4, false, 1, 0, 0, "p_min stop at position 0: attempted 1, kept 0" },
                          tc{ 0.0f,  3, 2, false, 2, 2, 0, "n_min 3 clears a 2-token draft: kept 2 reported" },
                          tc{ 0.0f,  3, 2, true,  2, 2, 2, "keep_short probe: n_min 3 keeps the 2-token draft" },
                          tc{ 0.0f,  3, 3, true,  3, 3, 3, "keep_short probe at n_min: 3 drafted, 3 kept" } }) {
        mtp_env e;
        env_cfg cfg = { "", GGML_TYPE_F16, 512, false };
        cfg.p_min = t.p_min;
        cfg.n_min = t.n_min;
        init_env(e, meta, cfg);
        const auto prompt = tokens(24, 7);
        const auto v = feed(e, prompt, 0);
        llama_token id_last = 0;
        for (size_t j = 1; j < N_VOCAB; ++j) {
            if (v[(prompt.size() - 1)*N_VOCAB + j] > v[(prompt.size() - 1)*N_VOCAB + id_last]) {
                id_last = (llama_token) j;
            }
        }
        llama_tokens out;
        auto & dp = common_speculative_get_draft_params(e.spec, 0);
        dp.drafting = true;
        dp.n_max    = t.n_max;
        dp.n_past   = (llama_pos) prompt.size();
        dp.id_last  = id_last;
        dp.prompt   = nullptr;
        dp.result   = &out;
        dp.keep_short = t.keep;
        common_speculative_draft(e.spec);
        check(dp.n_attempted == t.att && dp.n_kept == t.kept && out.size() == t.out,
              std::string(t.what) + " (got attempted " + std::to_string(dp.n_attempted) + ", kept " +
              std::to_string(dp.n_kept) + ", result " + std::to_string(out.size()) + ")");
    }
}

// [TAG_MULTISLOT] stage 5.1 item 1: the MTP catch-up of a multi-sequence batch ([0,0,1,1], reversed ids, one-row runs,
// runs sliced over ubatches, a restored carry, rejection + replay) gives each sequence the carry, the draft logits and
// the draft proposals of the same runs fed one sequence per batch (same context shape; tolerance for the target's
// batch-size-dependent rounding, a row mix-up is O(0.1)); then the [mtp-contiguity] proof line with multi_seq > 0
struct seq_run {
    llama_seq_id             seq;
    std::vector<llama_token> toks;
    llama_pos                pos0;
};

// one target decode of the runs in order (every row an output) + the catch-up; returns the target logits per run
static std::vector<std::vector<float>> feed_runs(mtp_env & env, const std::vector<seq_run> & runs) {
    int32_t n = 0;
    for (const auto & r : runs) {
        n += (int32_t) r.toks.size();
    }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (const auto & r : runs) {
        for (size_t i = 0; i < r.toks.size(); ++i) {
            common_batch_add(b, r.toks[i], r.pos0 + (llama_pos) i, { r.seq }, true);
        }
    }
    GGML_ASSERT(llama_decode(env.ctx_tgt.get(), b) == 0);
    std::vector<std::vector<float>> out;
    int32_t row = 0;
    for (const auto & r : runs) {
        out.emplace_back(r.toks.size() * N_VOCAB);
        for (size_t i = 0; i < r.toks.size(); ++i, ++row) {
            std::memcpy(out.back().data() + i * N_VOCAB, llama_get_logits_ith(env.ctx_tgt.get(), row), N_VOCAB * sizeof(float));
        }
    }
    GGML_ASSERT(common_speculative_process(env.spec, b));
    llama_batch_free(b);
    return out;
}

static float max_abs_diff(const float * a, const float * b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

static void multi_seq_catchup(gguf_context * meta) {
    const float TOL = 1e-3f;
    mtp_env M, S; // M: the runs of one step in ONE batch; S: the same runs, one sequence per batch
    env_cfg cfg = { "", GGML_TYPE_F16, 1024, false };
    cfg.n_seq = 2;
    init_env(M, meta, cfg);
    init_env(S, meta, cfg);

    llama_pos pos[2] = { 0, 0 };
    uint32_t  seed   = 100;
    std::vector<llama_token> last[2];
    float worst = 0.0f;
    bool  bytes_same = true;

    const auto compare = [&](const std::string & what) {
        for (llama_seq_id s = 0; s < 2; ++s) {
            std::vector<uint8_t> cm, cs;
            GGML_ASSERT(common_speculative_get_state(M.spec, s, cm) && common_speculative_get_state(S.spec, s, cs));
            const size_t nf = (cm.size() - sizeof(uint32_t)) / sizeof(float);
            const float dc = cm.size() == cs.size() ? max_abs_diff((const float *) (cm.data() + 4), (const float *) (cs.data() + 4), nf) : 1e9f;
            check(dc <= TOL, what + ": seq " + std::to_string(s) + " carry (max |d| " + std::to_string(dc) + ")");
            bytes_same = bytes_same && eq(cm, cs);
            worst = std::max(worst, dc);
            const llama_token id_last = last[s].back();
            const auto dm = draft(M, pos[s], id_last, s);
            const auto ds = draft(S, pos[s], id_last, s);
            check(dm.toks == ds.toks, what + ": seq " + std::to_string(s) + " draft proposals");
            const float dl = dm.logits.size() == ds.logits.size() && !dm.logits.empty()
                ? max_abs_diff(dm.logits.data(), ds.logits.data(), N_VOCAB) : 1e9f;
            check(dl <= TOL, what + ": seq " + std::to_string(s) + " draft logits (max |d| " + std::to_string(dl) + ")");
            worst = std::max(worst, dl);
        }
    };
    // one step: M gets every run in one batch, S each run alone; target logits compared per run
    const auto step = [&](const std::string & what, const std::vector<std::pair<llama_seq_id, size_t>> & shape) {
        std::vector<seq_run> runs;
        for (const auto & [s, n] : shape) {
            runs.push_back({ s, tokens(n, seed++), pos[s] });
            pos[s] += (llama_pos) n;
            last[s] = runs.back().toks;
        }
        const auto lm = feed_runs(M, runs);
        for (size_t k = 0; k < runs.size(); ++k) {
            const auto ls = feed_runs(S, { runs[k] });
            const float d = max_abs_diff(lm[k].data(), ls[0].data(), lm[k].size());
            check(d <= TOL, what + ": target logits of run " + std::to_string(k) + " (max |d| " + std::to_string(d) + ")");
        }
        compare(what);
    };

    step("[0,0,1,1] prompt runs",     { { 0, 20 }, { 1, 13 } });
    step("reversed ids [1,1,0,0]",    { { 1, 5 },  { 0, 7 } });
    step("one-row runs [0,1]",        { { 0, 1 },  { 1, 1 } });
    step("one-row runs [1,0]",        { { 1, 1 },  { 0, 1 } });
    step("runs sliced over ubatches", { { 0, 50 }, { 1, 45 } });

    // rejection + replay with a restored carry: per-sequence checkpoints (target state + carry, the server's round
    // order), verify-like runs of 4, restore both (the recurrent target cannot seq_rm partially), replay different
    // accepted counts
    {
        const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
        mtp_env * envs[2] = { &M, &S };
        common_prompt_checkpoint ck[2][2];
        const llama_pos p0[2] = { pos[0], pos[1] };
        std::vector<llama_token> last0[2] = { last[0], last[1] };
        for (int e = 0; e < 2; ++e) {
            auto * mem = llama_get_memory(envs[e]->ctx_tgt.get());
            for (llama_seq_id q = 0; q < 2; ++q) {
                ck[e][q].update_pos(p0[q], llama_memory_seq_pos_min(mem, q), llama_memory_seq_pos_max(mem, q));
                ck[e][q].update_tgt(envs[e]->ctx_tgt.get(), q, flags);
                ck[e][q].data_spec.clear();
                GGML_ASSERT(common_speculative_get_state(envs[e]->spec, q, ck[e][q].data_spec));
            }
        }
        step("verify runs [1x4,0x4]", { { 1, 4 }, { 0, 4 } });
        for (int e = 0; e < 2; ++e) {
            common_memory mem;
            mem.init(envs[e]->ctx_tgt.get(), envs[e]->ctx_dft.get());
            for (llama_seq_id q = 0; q < 2; ++q) {
                ck[e][q].load_tgt(envs[e]->ctx_tgt.get(), q, flags);
                ck[e][q].load_dft(envs[e]->ctx_dft.get(), q, flags);
                mem.seq_rm(q, ck[e][q].pos_max + 1, -1);
                common_speculative_set_state(envs[e]->spec, q, ck[e][q].data_spec);
            }
        }
        pos[0] = p0[0]; pos[1] = p0[1];
        last[0] = last0[0]; last[1] = last0[1];
        compare("after the restore of both carries");
        seed -= 2; // replay the same draws: the accepted prefixes of the rejected runs + their bonus rows
        step("replay [1x2,0x3]", { { 1, 2 }, { 0, 3 } });
    }
    printf("multi-seq catch-up: %s vs one sequence per batch (max |d| %.2e, carry bytes %s)\n",
           n_fail == 0 ? "matches" : "DIFFERS", (double) worst, bytes_same ? "identical" : "within tolerance");
    // the proof line ([mtp-contiguity] ... multi_seq > 0) is printed by M's destructor
    common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
    common_speculative_free(M.spec);
    M.spec = nullptr;
    common_log_set_verbosity_thold(-1);
}

// [TAG_MULTISLOT] disposable-process negative cases: each child feeds one malformed batch straight to the catch-up
// (no target decode: the check must fire first) and must abort with the [mtp-contiguity] diagnostic and no draft node
static int neg_child(gguf_context * meta, const std::string & which) {
    mtp_env e;
    env_cfg cfg = { "", GGML_TYPE_F16, 512, true };
    cfg.n_seq = 2;
    init_env(e, meta, cfg);
    std::vector<std::pair<llama_seq_id, llama_pos>> rows;
    if (which == "interleave") {
        rows = { { 0, 0 }, { 1, 0 }, { 0, 1 } };           // [0,1,0]
    } else if (which == "badid") {
        rows = { { 0, 0 }, { 2, 0 } };                     // id 2 with n_seq 2
    } else if (which == "negid") {
        rows = { { -1, 0 } };
    } else if (which == "hole") {
        rows = { { 0, 5 }, { 0, 7 } };                     // position hole inside a run
    } else if (which == "multiseq") {
        rows = { { 0, 0 } };                               // one row owned by seq 0 AND seq 1 (n_seq_id 2), below
    } else {
        return 2;
    }
    llama_batch b = llama_batch_init((int32_t) rows.size(), 0, 2);
    for (const auto & [s, p] : rows) {
        if (which == "multiseq") {
            common_batch_add(b, 1, p, { s, s + 1 }, true); // [TAG_MULTISLOT] round 2 (astra r1): n_seq_id != 1
        } else {
            common_batch_add(b, 1, p, { s }, true);
        }
    }
    g_neg_child = true;
    e.census.active = true;
    common_speculative_process(e.spec, b);
    fprintf(stderr, "NEG-CHILD: no abort\n");
    llama_batch_free(b);
    return 0;
}

static void neg_cases(const char * self) {
#ifdef _WIN32
    (void) self;
    printf("mtp-contiguity negative cases: skipped on _WIN32\n");
#else
    for (const char * which : { "interleave", "badid", "negid", "hole", "multiseq" }) {
        const std::string cmd = std::string("'") + self + "' --neg " + which + " 2>&1";
        FILE * f = popen(cmd.c_str(), "r");
        GGML_ASSERT(f != nullptr);
        std::string out;
        char buf[512];
        while (fgets(buf, sizeof(buf), f)) {
            out += buf;
        }
        const int st = pclose(f);
        const bool died = st != 0;
        const size_t pd = out.find("[mtp-contiguity] violation");
        const bool decoded = out.find("NEG-CHILD: draft decode reached") != std::string::npos;
        check(died && pd != std::string::npos && !decoded,
              std::string("negative case '") + which + "': aborts with the diagnostic before draft decode");
        if (pd != std::string::npos) {
            printf("  neg %-10s -> %s", which, out.substr(pd, out.find('\n', pd) - pd + 1).c_str());
        } else {
            printf("  neg %-10s -> NO DIAGNOSTIC (status %d)\n", which, st);
        }
    }
#endif
}

// [TAG_MULTISLOT] (stage 5.1 item 4) homogeneous routing-class scheduling under per-stream KV (no --kv-unified), run
// in a child process with LLAMA_QSA_HOMOGENEOUS=1 LLAMA_QSA_MIN_KV=64 + the QSA env (ctest). Each round's runs are
// planned by the server's planner (server_qsa_route_groups, N = 64, MAX_UBATCH = 8) and executed like the server's
// ranges: one target decode + MTP catch-up per group; drafts per group with only the members' drafting flags. The
// model guard ([qsa-routing]) aborts on any mixed-class ubatch, so completing proves mixed=0; the parent parses the
// exit proof line and asserts multi_seq > 0, below > 0, above > 0. M = grouped, S = each run alone (solo reference).
static const int64_t HOMOG_N = 64;

struct homog_round_out {
    std::vector<std::vector<float>> logits; // per run
    int n_calls = 0;
    double ms = 0.0;
};

static std::vector<std::vector<int32_t>> homog_plan(mtp_env & env, const std::vector<seq_run> & runs, int64_t max_ub = 8) {
    std::vector<server_qsa_route_run> infos;
    auto * mem = llama_get_memory(env.ctx_tgt.get());
    int32_t row = 0;
    for (const auto & r : runs) {
        const llama_pos c_max = llama_memory_seq_pos_max(mem, r.seq);
        const bool partial = r.pos0 != c_max + 1; // as the server (pos_min is the recurrent part's latest position)
        infos.push_back({ r.seq, (int32_t) r.toks.size(), r.pos0, r.pos0 + (llama_pos) r.toks.size() - 1, partial, row });
        row += (int32_t) r.toks.size();
    }
    auto groups = server_qsa_route_groups(infos, HOMOG_N, max_ub);
    std::string d;
    for (const auto & g : groups) {
        d += " {";
        for (const int32_t k : g) {
            d += string_format(" %d:%d-%d%s", infos[k].id_slot, (int) infos[k].p_lo, (int) infos[k].p_hi, infos[k].partial ? "(partial)" : "");
        }
        d += " }";
    }
    printf("  plan:%s\n", d.c_str());
    return groups;
}

static homog_round_out homog_exec(mtp_env & env, const std::vector<seq_run> & runs, bool grouped, int64_t max_ub = 8) {
    homog_round_out out;
    out.logits.resize(runs.size());
    std::vector<std::vector<int32_t>> groups;
    if (grouped) {
        groups = homog_plan(env, runs, max_ub);
    } else {
        for (size_t k = 0; k < runs.size(); ++k) {
            groups.push_back({ (int32_t) k });
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto & g : groups) {
        std::vector<seq_run> part;
        for (const int32_t k : g) {
            part.push_back(runs[k]);
        }
        const auto l = feed_runs(env, part);
        for (size_t j = 0; j < g.size(); ++j) {
            out.logits[g[j]] = l[j];
        }
        out.n_calls++;
    }
    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

// drafts of several sequences: grouped like the server (one common_speculative_draft per group, only its members'
// drafting flags), or one sequence per call (solo)
static std::map<llama_seq_id, std::vector<llama_token>> homog_drafts(mtp_env & env,
        const std::vector<std::tuple<llama_seq_id, llama_pos, llama_token>> & req, bool grouped, int * n_calls,
        std::vector<std::vector<float>> * dlogits = nullptr) {
    std::map<llama_seq_id, std::vector<llama_token>> res;
    const int n_max = env.params.draft.n_max;
    std::vector<server_qsa_route_run> infos;
    for (size_t k = 0; k < req.size(); ++k) {
        const auto & [s, n_past, id] = req[k];
        res[s] = {};
        infos.push_back({ s, 1, n_past, n_past + n_max - 1, false, (int32_t) k });
    }
    std::vector<std::vector<int32_t>> groups;
    if (grouped) {
        groups = server_qsa_route_groups(infos, HOMOG_N, 8);
    } else {
        for (size_t k = 0; k < req.size(); ++k) {
            groups.push_back({ (int32_t) k });
        }
    }
    for (const auto & g : groups) {
        for (const int32_t k : g) {
            const auto & [s, n_past, id] = req[k];
            auto & dp = common_speculative_get_draft_params(env.spec, s);
            dp.drafting = true;
            dp.n_max    = -1;
            dp.n_past   = n_past;
            dp.id_last  = id;
            dp.prompt   = nullptr;
            dp.result   = &res[s];
        }
        common_speculative_draft(env.spec);
        (*n_calls)++;
        if (dlogits != nullptr) { // the last draft output row of this group's call (empty: no draft output)
            const float * l = llama_get_logits_ith(env.ctx_dft.get(), -1);
            dlogits->emplace_back(l != nullptr ? std::vector<float>(l, l + N_VOCAB) : std::vector<float>());
        }
    }
    for (const auto & [s, n_past, id] : req) {
        llama_memory_seq_rm(llama_get_memory(env.ctx_dft.get()), s, n_past, -1);
    }
    return res;
}

static void homog_case(gguf_context * meta, uint32_t n_seq) {
    const float TOL = 1e-3f;
    // (r3, astra r2 Major 3) the exactness standard: G runs the SAME grouped schedule as M (identical prefill / history
    // construction, ranges and draft groups) with the graph cache K = 8 + CHECK = 3; M is K = 0. M vs G must be BYTE
    // equal: target logits, draft logits, proposals, carry. M vs S (solo) is R81 composition (different row counts per
    // decode round differently on CPU): tolerance + equal proposals, the first divergence printed
    mtp_env M, S, G;
    env_cfg cfg = { "", GGML_TYPE_F16, 1024 * n_seq, false };
    cfg.n_seq = n_seq;
    common_set_env("LLAMA_GRAPH_CACHE", "0");
    common_set_env("LLAMA_GRAPH_CACHE_CHECK", "0");
    init_env(M, meta, cfg);
    init_env(S, meta, cfg);
    common_set_env("LLAMA_GRAPH_CACHE", "8");
    common_set_env("LLAMA_GRAPH_CACHE_CHECK", "3");
    init_env(G, meta, cfg);
    common_set_env("LLAMA_GRAPH_CACHE", "");
    common_set_env("LLAMA_GRAPH_CACHE_CHECK", "");
    GGML_ASSERT(!llama_context_default_params().kv_unified && "per-stream KV expected");
    int n_gc_byte = 0, n_gc_cmp = 0, n_solo_div = 0;

    std::vector<llama_pos> pos(n_seq, 0);
    std::vector<llama_token> last(n_seq, 0);
    uint32_t seed = 500;
    int n_bytes_same = 0, n_runs = 0, n_mixed_rounds = 0;
    float worst = 0.0f;
    double ms_mixed = 0.0, ms_single = 0.0;
    int n_ms_mixed = 0, n_ms_single = 0;

    // one round: shape = (seq, n rows); expect = expected decode calls of the grouped execution
    const auto round = [&](const std::string & what, const std::vector<std::pair<llama_seq_id, size_t>> & shape, int expect) {
        std::vector<seq_run> runs;
        for (const auto & [s, n] : shape) {
            runs.push_back({ s, tokens(n, seed++), pos[s] });
            pos[s] += (llama_pos) n;
            last[s] = runs.back().toks.back();
        }
        const auto om = homog_exec(M, runs, true);
        const auto og = homog_exec(G, runs, true);
        const auto os = homog_exec(S, runs, false);
        check(og.n_calls == om.n_calls, what + ": K=8 schedule == K=0 schedule");
        check(om.n_calls == expect, what + ": " + std::to_string(om.n_calls) + " decode calls (expected " + std::to_string(expect) + ")");
        if (om.n_calls > 1) {
            n_mixed_rounds++;
        }
        if (shape.size() == 2 && shape[0].second == 1 && shape[1].second == 1) { // two single rows: the cost probe
            (om.n_calls > 1 ? ms_mixed : ms_single) += om.ms;
            (om.n_calls > 1 ? n_ms_mixed : n_ms_single)++;
        }
        for (size_t k = 0; k < runs.size(); ++k) {
            // exactness: identical grouping + history, graph cache K = 0 vs K = 8 + CHECK = 3 -> byte equal
            check(eq(om.logits[k], og.logits[k]), what + ": run of seq " + std::to_string(runs[k].seq) + " target logits K=0 == K=8 (bytes)");
            n_gc_byte += eq(om.logits[k], og.logits[k]) ? 1 : 0;
            n_gc_cmp++;
            // R81 composition vs solo: tolerance is kept ONLY here - the grouped call decodes a different number of rows
            // per ubatch than the solo one (CPU mul_mat accumulates a 2-row decode differently from a 1-row one)
            const float d = max_abs_diff(om.logits[k].data(), os.logits[k].data(), om.logits[k].size());
            if (!eq(om.logits[k], os.logits[k])) {
                size_t i = 0;
                while (i < om.logits[k].size() && std::memcmp(&om.logits[k][i], &os.logits[k][i], sizeof(float)) == 0) {
                    ++i;
                }
                printf("  composition: %s seq %d first divergence vs solo at row %zu (pos %d) vocab %zu: %.9g vs %.9g\n",
                       what.c_str(), (int) runs[k].seq, i / N_VOCAB, (int) (runs[k].pos0 + (llama_pos) (i / N_VOCAB)),
                       i % N_VOCAB, (double) om.logits[k][i], (double) os.logits[k][i]);
                n_solo_div++;
            }
            check(d <= TOL, what + ": run of seq " + std::to_string(runs[k].seq) + " logits vs solo (max |d| " + std::to_string(d) + ")");
            worst = std::max(worst, d);
            n_bytes_same += eq(om.logits[k], os.logits[k]) ? 1 : 0;
            n_runs++;
        }
        std::vector<std::tuple<llama_seq_id, llama_pos, llama_token>> req;
        for (const auto & [s, n] : shape) {
            req.emplace_back(s, pos[s], last[s]); // the next position, as the server (n_past = n_tokens)
        }
        int nc_m = 0, nc_s = 0, nc_g = 0;
        std::vector<std::vector<float>> dl_m, dl_g;
        const auto dm = homog_drafts(M, req, true, &nc_m, &dl_m);
        const auto dg = homog_drafts(G, req, true, &nc_g, &dl_g);
        const auto ds = homog_drafts(S, req, false, &nc_s);
        check(nc_g == nc_m && dl_m == dl_g, what + ": draft logits K=0 == K=8 (bytes, per draft group)");
        for (const auto & [s, toks] : dm) {
            check(toks == dg.at(s), what + ": seq " + std::to_string(s) + " draft proposals K=0 == K=8");
            check(toks == ds.at(s), what + ": seq " + std::to_string(s) + " draft proposals vs solo");
            std::vector<uint8_t> cm, cg;
            GGML_ASSERT(common_speculative_get_state(M.spec, s, cm) && common_speculative_get_state(G.spec, s, cg));
            check(cm == cg, what + ": seq " + std::to_string(s) + " carry K=0 == K=8 (bytes)");
        }
    };

    const bool np4 = n_seq >= 4;
    // unequal prompt lengths; seq 2's prompt straddles N (a singleton range)
    if (np4) {
        round("prefill unequal lengths A",     { { 0, 61 }, { 3, 30 } }, 1);  // n_batch 128: two prefill rounds
        round("prefill unequal lengths B",     { { 1, 63 }, { 2, 64 } }, 2);
        round("depths 62/64/65/31 (ids 0,3 | 1,2)", { { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } }, 2);
        round("verify blocks, seq 0 crosses N",  { { 0, 4 }, { 1, 4 }, { 2, 4 }, { 3, 4 } }, 3);
        round("same class above + below",       { { 0, 1 }, { 1, 1 }, { 2, 1 }, { 3, 1 } }, 2);
        round("reversed ids, above",            { { 2, 2 }, { 1, 2 }, { 0, 2 } }, 1);
    } else {
        round("prefill unequal lengths",        { { 0, 61 }, { 1, 63 } }, 1);
        round("depths 62/64 (mixed)",           { { 0, 1 }, { 1, 1 } }, 2);
        round("verify blocks, seq 0 crosses N", { { 0, 4 }, { 1, 4 } }, 2);
        round("same class above",               { { 0, 1 }, { 1, 1 } }, 1);
        round("reversed ids, above",            { { 1, 2 }, { 0, 2 } }, 1);
    }

    // rejection + replay: checkpoint every sequence (target + carry), verify runs, restore, replay shorter runs
    {
        const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
        mtp_env * envs[3] = { &M, &S, &G };
        std::vector<common_prompt_checkpoint> ck[3];
        const std::vector<llama_pos> p0 = pos;
        const std::vector<llama_token> last0 = last;
        for (int e = 0; e < 3; ++e) {
            ck[e].resize(n_seq);
            auto * mem = llama_get_memory(envs[e]->ctx_tgt.get());
            for (llama_seq_id q = 0; q < (llama_seq_id) n_seq; ++q) {
                ck[e][q].update_pos(p0[q], llama_memory_seq_pos_min(mem, q), llama_memory_seq_pos_max(mem, q));
                ck[e][q].update_tgt(envs[e]->ctx_tgt.get(), q, flags);
                GGML_ASSERT(common_speculative_get_state(envs[e]->spec, q, ck[e][q].data_spec));
            }
        }
        std::vector<std::pair<llama_seq_id, size_t>> vshape, rshape;
        for (llama_seq_id q = 0; q < (llama_seq_id) n_seq; ++q) {
            vshape.push_back({ q, 4 });
            rshape.push_back({ q, (size_t) (1 + q % 3) });
        }
        round("verify before rejection", vshape, np4 ? 3 : 1); // np4: above 4+4 | 4 (MAX_UBATCH 8) | below
        for (int e = 0; e < 3; ++e) {
            common_memory mem;
            mem.init(envs[e]->ctx_tgt.get(), envs[e]->ctx_dft.get());
            for (llama_seq_id q = 0; q < (llama_seq_id) n_seq; ++q) {
                ck[e][q].load_tgt(envs[e]->ctx_tgt.get(), q, flags);
                ck[e][q].load_dft(envs[e]->ctx_dft.get(), q, flags);
                mem.seq_rm(q, ck[e][q].pos_max + 1, -1);
                common_speculative_set_state(envs[e]->spec, q, ck[e][q].data_spec);
            }
        }
        pos = p0;
        last = last0;
        round("replay after rejection", rshape, np4 ? 2 : 1);
    }

    // cancellation: seq 1 released (both memories cleared), a new short prompt (below) beside an above neighbour
    for (mtp_env * e : { &M, &S, &G }) {
        common_memory mem;
        mem.init(e->ctx_tgt.get(), e->ctx_dft.get());
        mem.seq_rm(1, -1, -1);
    }
    pos[1] = 0;
    round("cancel seq 1 + new prompt beside seq 0", { { 1, 10 }, { 0, 1 } }, 2);
    round("new prompt continues beside seq 0",      { { 1, 1 },  { 0, 1 } }, 2);

    const auto pm = llama_perf_context(M.ctx_tgt.get());
    const auto pg = llama_perf_context(G.ctx_tgt.get());
    const auto pgd = llama_perf_context(G.ctx_dft.get());
    check(n_gc_byte == n_gc_cmp && n_gc_cmp > 0, "exactness: every run byte-equal K=0 vs K=8 + CHECK=3 under identical grouping");
    check(pg.n_reused > 0, "K=8: positive target graph reuse");
    printf("homogeneous np%u exactness: %d/%d runs byte-equal K=0 vs K=8+CHECK=3 (target n_reused K=0 %d, K=8 %d, draft K=8 %d); %d runs diverge from solo (composition)\n",
           n_seq, n_gc_byte, n_gc_cmp, pm.n_reused, pg.n_reused, pgd.n_reused, n_solo_div);
    printf("homogeneous np%u: %d runs, %d byte-identical to solo, max |d| %.2e, mixed rounds %d; two-row round %.2f ms mixed (2 calls) vs %.2f ms same-class (1 call)\n",
           n_seq, n_runs, n_bytes_same, (double) worst, n_mixed_rounds,
           n_ms_mixed ? ms_mixed / n_ms_mixed : 0.0, n_ms_single ? ms_single / n_ms_single : 0.0);
}

// (r3, astra r2 Minor) the target-range row cap: MAX_UBATCH (8) applies to target ranges only with an attention
// catch-up or LLAMA_QSA_HOMOGENEOUS_TARGET_CAP=1. With the attention-free catch-up (LLAMA_MTP_CATCHUP_ATTN=0, prod) two
// same-class 6-row verify runs share ONE target call (+ one catch-up); vs solo: tolerance + equal proposals (R81
// composition: a 12-row decode vs two 6-row ones rounds differently on CPU)
static void homog_wide_case(gguf_context * meta) {
    check(server_qsa_route_target_cap(8, nullptr, "0") == 0, "target cap: attention-free catch-up -> uncapped");
    check(server_qsa_route_target_cap(8, "1", "0") == 8, "target cap: LLAMA_QSA_HOMOGENEOUS_TARGET_CAP=1 -> 8");
    check(server_qsa_route_target_cap(8, nullptr, nullptr) == 8, "target cap: catch-up attention unset (default 1) -> 8");
    check(server_qsa_route_target_cap(8, "0", "1") == 8, "target cap: LLAMA_MTP_CATCHUP_ATTN=1 -> 8");
    const std::vector<server_qsa_route_run> two6 = { { 0, 6, 70, 75, false, 0 }, { 1, 6, 70, 75, false, 6 } };
    check(server_qsa_route_groups(two6, HOMOG_N, 0).size() == 1, "planner: two same-class 6-row runs, cap 0 -> one group");
    check(server_qsa_route_groups(two6, HOMOG_N, 8).size() == 2, "planner: two same-class 6-row runs, cap 8 -> two groups");

    const float TOL = 1e-3f;
    mtp_env M, S;
    env_cfg cfg = { "0", GGML_TYPE_F16, 2048, false };
    cfg.n_seq = 2;
    init_env(M, meta, cfg);
    init_env(S, meta, cfg);
    const int64_t cap = server_qsa_route_target_cap(8, nullptr, "0");
    for (mtp_env * e : { &M, &S }) {
        feed_runs(*e, { { 0, tokens(70, 901), 0 } });
        feed_runs(*e, { { 1, tokens(70, 902), 0 } });
    }
    const std::vector<seq_run> runs = { { 0, tokens(6, 903), 70 }, { 1, tokens(6, 904), 70 } };
    const auto om = homog_exec(M, runs, true, cap);
    const auto os = homog_exec(S, runs, false);
    check(om.n_calls == 1, "attention-free catch-up: two same-class 6-row verify runs share one target call (" + std::to_string(om.n_calls) + ")");
    for (size_t k = 0; k < runs.size(); ++k) {
        const float d = max_abs_diff(om.logits[k].data(), os.logits[k].data(), om.logits[k].size());
        check(d <= TOL, "12-row call: seq " + std::to_string(k) + " logits vs solo (max |d| " + std::to_string(d) + ")");
    }
    std::vector<std::tuple<llama_seq_id, llama_pos, llama_token>> req = { { 0, 76, runs[0].toks.back() }, { 1, 76, runs[1].toks.back() } };
    int nc_m = 0, nc_s = 0;
    const auto dm = homog_drafts(M, req, true, &nc_m);
    const auto ds = homog_drafts(S, req, false, &nc_s);
    for (const auto & [q, toks] : dm) {
        check(toks == ds.at(q), "12-row call: seq " + std::to_string(q) + " draft proposals vs solo");
    }
    printf("homogeneous wide: two 6-row runs in %d target call(s), cap %lld\n", om.n_calls, (long long) cap);
}

// the guard's negative case (disposable child): seq 0 at depth 63 and seq 1 at depth 64 in ONE target ubatch (two
// streams) must abort with the [qsa-routing] diagnostic
static int homog_neg_child(gguf_context * meta) {
    mtp_env E;
    env_cfg cfg = { "", GGML_TYPE_F16, 2048, false };
    cfg.n_seq = 2;
    init_env(E, meta, cfg);
    feed_runs(E, { { 0, tokens(62, 1), 0 } });
    feed_runs(E, { { 1, tokens(63, 2), 0 } });
    feed_runs(E, { { 0, tokens(1, 3), 62 }, { 1, tokens(1, 4), 63 } }); // must abort
    printf("NEG-CHILD: mixed ubatch decoded\n");
    return 0;
}

// (r3, astra r2 Major 1) the guard's same-key negative case (disposable child): ONE llama_batch buffer is decoded
// twice with an unchanged first/last row (seq 0 @ 80, seq 2 @ 80), n_tokens and first sequence - the r2 dedup key -
// while the INTERIOR sequence's depth changes: call 1 seq 1 @ 80 (all above N = 64), then a checkpoint rewind of all
// three and call 2 with seq 1 @ 61 (depth 62, below). Target decodes only (no catch-up in between), so the guard's
// previous invocation is call 1's ubatch. The r2 guard returned early on the remembered key; r3 must abort
static int homog_neg_reuse_child(gguf_context * meta) {
    mtp_env E;
    env_cfg cfg = { "", GGML_TYPE_F16, 3 * 1024, false };
    cfg.n_seq = 3;
    init_env(E, meta, cfg);
    const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    auto * mem = llama_get_memory(E.ctx_tgt.get());
    common_prompt_checkpoint ck[3];
    const auto take = [&](llama_seq_id q, llama_pos n_past) {
        ck[q].update_pos(n_past, llama_memory_seq_pos_min(mem, q), llama_memory_seq_pos_max(mem, q));
        ck[q].update_tgt(E.ctx_tgt.get(), q, flags);
    };
    feed_runs(E, { { 0, tokens(80, 11), 0 } });
    feed_runs(E, { { 2, tokens(80, 12), 0 } });
    feed_runs(E, { { 1, tokens(61, 13), 0 } });
    take(1, 61);
    feed_runs(E, { { 1, tokens(19, 14), 61 } });
    take(0, 80);
    take(2, 80);

    llama_batch b = llama_batch_init(3, 0, 1);
    for (llama_seq_id q = 0; q < 3; ++q) {
        common_batch_add(b, (llama_token) (7 + q), 80, { q }, true);
    }
    GGML_ASSERT(llama_decode(E.ctx_tgt.get(), b) == 0); // homogeneous (above)
    printf("NEG-REUSE: call 1 (seq 0/1/2 @ 80) decoded\n");
    fflush(stdout);
    for (llama_seq_id q = 0; q < 3; ++q) {
        GGML_ASSERT(ck[q].load_tgt(E.ctx_tgt.get(), q, flags));
        GGML_ASSERT(llama_memory_seq_rm(mem, q, ck[q].pos_max + 1, -1));
    }
    b.pos[1] = 61;   // the interior sequence only: same buffer, same n_tokens, first/last row and first seq id
    b.token[1] = 9;
    GGML_ASSERT(llama_decode(E.ctx_tgt.get(), b) == 0); // must abort
    printf("NEG-CHILD: mixed ubatch decoded\n");
    llama_batch_free(b);
    return 0;
}

static int homog_parent(const char * self, uint32_t n_seq) {
#ifdef _WIN32
    (void) self; (void) n_seq;
    printf("homogeneous routing: skipped on _WIN32\n");
    return 0;
#else
    const std::string cmd = std::string("'") + self + "' --homog-child " + std::to_string(n_seq) + " 2>&1";
    FILE * f = popen(cmd.c_str(), "r");
    GGML_ASSERT(f != nullptr);
    std::string out;
    char buf[1024];
    while (fgets(buf, sizeof(buf), f)) {
        out += buf;
        fputs(buf, stdout);
    }
    const int st = pclose(f);
    check(st == 0, "homogeneous child exits 0 (no [qsa-routing] abort)");
    const size_t p = out.rfind("[qsa-routing] checks=");
    unsigned long long c = 0, ms = 0, b = 0, a = 0, x = 1;
    const bool parsed = p != std::string::npos &&
        sscanf(out.c_str() + p, "[qsa-routing] checks=%llu multi_seq=%llu below=%llu above=%llu mixed=%llu", &c, &ms, &b, &a, &x) == 5;
    check(parsed, "the [qsa-routing] exit proof line is printed");
    check(ms > 0, "multi-sequence ubatches (n_stream > 1) were checked");
    check(b > 0 && a > 0, "both routing classes were executed");
    check(x == 0, "mixed=0");
    {
        const std::string cmd_neg = std::string("'") + self + "' --homog-neg 2 2>&1";
        FILE * fn = popen(cmd_neg.c_str(), "r");
        GGML_ASSERT(fn != nullptr);
        std::string neg;
        while (fgets(buf, sizeof(buf), fn)) {
            neg += buf;
        }
        const int stn = pclose(fn);
        const size_t pd = neg.find("[qsa-routing] mixed routing classes");
        check(stn != 0 && pd != std::string::npos && neg.find("NEG-CHILD: mixed ubatch decoded") == std::string::npos,
              "negative: a below + above ubatch aborts with the [qsa-routing] diagnostic");
        if (pd != std::string::npos) {
            printf("  neg mixed -> %s", neg.substr(pd, neg.find('\n', pd) - pd + 1).c_str());
        }
    }
    {
        // (r3) the same-key reuse case: call 1 must decode, call 2 must abort with the diagnostic
        const std::string cmd_neg = std::string("'") + self + "' --homog-neg-reuse 3 2>&1";
        FILE * fn = popen(cmd_neg.c_str(), "r");
        GGML_ASSERT(fn != nullptr);
        std::string neg;
        while (fgets(buf, sizeof(buf), fn)) {
            neg += buf;
        }
        const int stn = pclose(fn);
        const size_t pd = neg.find("[qsa-routing] mixed routing classes");
        check(stn != 0 && neg.find("NEG-REUSE: call 1") != std::string::npos && pd != std::string::npos &&
              neg.find("NEG-CHILD: mixed ubatch decoded") == std::string::npos,
              "negative (same dedup key, interior sequence changed): call 1 decodes, call 2 aborts with the [qsa-routing] diagnostic");
        if (pd != std::string::npos) {
            printf("  neg reuse -> %s", neg.substr(pd, neg.find('\n', pd) - pd + 1).c_str());
        } else {
            printf("  neg reuse child output (status %d):\n%s", stn, neg.c_str());
        }
    }
    printf("test-mtp-catchup --homog %u: %s (%d checks ok)\n", n_seq, n_fail ? "FAILED" : "OK", n_ok);
    return n_fail ? 1 : 0;
#endif
}

// [TAG_MULTISLOT] (stage 5.1 item 5b, astra r2 Major 2) the operation-start gate of LLAMA_GTT_CONCURRENT_RESERVE_MIB
// (server_admission_op_check) against a fake LLAMA_GTT_USED_PATH file: N = 1 / 2, simultaneous starts, defer and
// recovery, unreadable usage, the default reserve 0, and a scheduler round in which the active neighbour keeps
// decoding (a deferred draft is a plain decode, a deferred restore waits) - no abort, no rejection
static int adm_op_case() {
    const std::string path = "test-mtp-catchup-adm-op.gtt_used";
    const auto write_used = [&](double mib) {
        FILE * f = fopen(path.c_str(), "w");
        GGML_ASSERT(f != nullptr);
        fprintf(f, "%llu\n", (unsigned long long) (mib * 1024.0 * 1024.0));
        fclose(f);
    };
    const double bar = 1000.0, reserve = 200.0;

    write_used(900.0);
    // default reserve 0 / admission off / N = 1 (no other slot processing): start, nothing evaluated
    auto d = server_admission_op_check(true, bar, 0.0, true, path);
    check(d.start && !d.gated, "reserve 0 (default): start, gate not evaluated");
    d = server_admission_op_check(false, bar, reserve, true, path);
    check(d.start && !d.gated, "admission off: start, gate not evaluated");
    d = server_admission_op_check(true, bar, reserve, false, path);
    check(d.start && !d.gated, "N = 1 / no concurrent slot: start even at used 900 + reserve 200 > bar 1000");

    // N = 2, another slot processing: used + reserve vs the bar
    d = server_admission_op_check(true, bar, reserve, true, path);
    check(!d.start && d.gated && d.projected_mib == 1100.0, "N = 2: used 900 + reserve 200 > bar 1000 -> defer (projected 1100)");
    write_used(800.0);
    d = server_admission_op_check(true, bar, reserve, true, path);
    check(d.start && d.gated, "N = 2: used 800 + reserve 200 == bar -> start (equality admits)");
    write_used(800.5);
    d = server_admission_op_check(true, bar, reserve, true, path);
    check(!d.start, "N = 2: used 800.5 + reserve 200 > bar -> defer");
    check(server_admission_op_check(true, bar, reserve, true, path + ".missing").start, "unreadable usage: start (no abort)");

    // a scheduler round, 2 slots (A generating = the active neighbour, B starting a restore), then pressure
    // drops (recovery). Per round: every generating slot asks DRAFT, the starting slot asks RESTORE_*.
    struct sim_slot { bool processing; int n_plain = 0, n_draft = 0, n_restore_wait = 0; bool restored = false; };
    std::vector<sim_slot> sl = { { true }, { false } };
    uint64_t deferred[SERVER_ADM_OP_COUNT] = { 0 };
    const auto concurrent = [&](size_t me) {
        bool c = false;
        for (size_t o = 0; o < sl.size(); ++o) {
            c |= o != me && sl[o].processing;
        }
        return c;
    };
    const auto round = [&](double used) {
        write_used(used);
        for (size_t k = 0; k < sl.size(); ++k) {
            if (sl[k].processing) { // generating: draft or plain decode - always one step
                const auto g = server_admission_op_check(true, bar, reserve, concurrent(k), path);
                (g.start ? sl[k].n_draft : sl[k].n_plain)++;
                deferred[SERVER_ADM_OP_DRAFT] += g.start ? 0 : 1;
            } else if (!sl[k].restored) { // simultaneous: a restore start in the same round
                const auto g = server_admission_op_check(true, bar, reserve, concurrent(k), path);
                if (g.start) {
                    sl[k].restored   = true;
                    sl[k].processing = true;
                } else {
                    sl[k].n_restore_wait++;
                    deferred[SERVER_ADM_OP_RESTORE_DISK]++;
                }
            }
        }
    };
    // a waiting task is not "processing": A (the active neighbour) is not gated by B's pending restore
    round(950.0); // pressure: A drafts on (no concurrent slot for A), B's restore waits
    round(950.0);
    check(sl[0].n_draft == 2 && sl[0].n_plain == 0, "pressure, B waiting: the active neighbour is untouched (2 drafted steps)");
    check(sl[1].n_restore_wait == 2 && !sl[1].restored, "pressure: the restore start waits (deferred twice, not rejected)");
    round(700.0); // recovery: A drafts, B restores (simultaneous starts in one round)
    check(sl[0].n_draft == 3 && sl[1].restored, "recovery: the deferred restore starts beside the drafting neighbour");
    round(950.0); // both processing under pressure: both drafts deferred -> plain decodes, both still advance
    check(sl[0].n_plain == 1 && sl[1].n_plain == 1, "both processing, pressure: both drafts deferred, both decode plain");
    round(700.0);
    check(sl[0].n_draft == 4 && sl[1].n_draft == 1, "both processing, under the bar: both draft again");
    check(deferred[SERVER_ADM_OP_DRAFT] == 2 && deferred[SERVER_ADM_OP_RESTORE_DISK] == 2, "deferrals counted per kind");
    check(strcmp(server_adm_op_name(SERVER_ADM_OP_RESTORE_SLOT), "restore_slot") == 0, "kind names");
    std::remove(path.c_str());
    printf("test-mtp-catchup --adm-op: %s (%d checks ok)\n", n_fail ? "FAILED" : "OK", n_ok);
    return n_fail ? 1 : 0;
}

int main(int argc, char ** argv) {
    llama_log_set(quiet_log, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    if (argc >= 2 && strcmp(argv[1], "--adm-op") == 0) {
        return adm_op_case(); // [TAG_MULTISLOT] item 5b, no model
    }

    gguf_context_ptr meta = make_meta(4096);

    if (argc >= 3 && strcmp(argv[1], "--neg") == 0) {
        return neg_child(meta.get(), argv[2]); // [TAG_MULTISLOT] disposable negative-case process
    }
    if (argc >= 3 && strcmp(argv[1], "--homog-child") == 0) {
        homog_case(meta.get(), (uint32_t) atoi(argv[2])); // [TAG_MULTISLOT] item 4
        if (atoi(argv[2]) == 2) {
            homog_wide_case(meta.get()); // (r3) the relaxed target cap
        }
        llama_backend_free();
        return n_fail ? 1 : 0;
    }
    if (argc >= 3 && strcmp(argv[1], "--homog-neg") == 0) {
        return homog_neg_child(meta.get());
    }
    if (argc >= 3 && strcmp(argv[1], "--homog-neg-reuse") == 0) {
        return homog_neg_reuse_child(meta.get());
    }
    if (argc >= 3 && strcmp(argv[1], "--homog") == 0) {
        char self[4096] = {};
#ifndef _WIN32
        const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
        if (n <= 0) {
            snprintf(self, sizeof(self), "%s", argv[0]);
        }
#endif
        return homog_parent(self, (uint32_t) atoi(argv[2]));
    }

    printf("mode: %s%s\n", mtp_qsa_mode() ? "sparse MTP draft (LLAMA_MTP_QSA=1)" : "dense MTP draft",
           getenv("TEST_MTP_GPU") && atoi(getenv("TEST_MTP_GPU")) ? ", GPU device (TEST_MTP_GPU=1)" : "");

    for (ggml_type t : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        run_pair(meta.get(), t, false);
        op_table(meta.get(), t);
    }
    run_pair(meta.get(), GGML_TYPE_F16, true); // backend draft sampling: the samplers see no zero-output graph now

    draft_kv_type_pair(meta.get());
    draft_policy_e2e(meta.get());
    draft_attempt_report(meta.get());

    multi_seq_catchup(meta.get());
    {
        // the running binary itself (ctest may pass a relative argv[0])
        char self[4096] = {};
#ifndef _WIN32
        const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
        if (n <= 0) {
            snprintf(self, sizeof(self), "%s", argc > 0 ? argv[0] : "test-mtp-catchup");
        }
#endif
        neg_cases(self);
    }

    timing(meta.get());

    llama_backend_free();

    if (n_fail) {
        printf("test-mtp-catchup: %d failures (%d checks ok)\n", n_fail, n_ok);
        return 1;
    }
    printf("test-mtp-catchup: OK (%d checks)\n", n_ok);
    return 0;
}
