// [TAG_UP28333] The MTP draft carrier (pending_h) at the start of a sequence (upstream #28333).
//
// A CPU model with random weights (qwen35 trunk of 2 layers + 1 MTP block, like test-llama-archs) drives the
// draft-mtp speculative implementation through the server's call order: target decode of the prompt ->
// common_speculative_process() (the MTP catch-up decode, which pairs x_0 with the carrier) -> draft.
//
//  1. fresh instance: prompt B -> draft tokens + the last draft decode's logits = reference.
//  2. one instance:   prompt U (unrelated), then B, then B again -> both B runs must equal the reference bit for
//     bit. Before the fix the carrier of B's x_0 was U's last hidden row, then B's own last row.
//  3. restore ([TAG_MTP_CKPT_PENDING_H], contrib 0126): after prompt P, a carrier saved with get_state and put back
//     with set_state over a foreign one must give the same continuation draft as the uninterrupted run; the
//     foreign carrier left in place must give a different one (the carrier is used, not zeroed, at pos > 0).

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
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

// qwen35: 2 trunk layers (layer 1 full attention) + 1 NextN/MTP block
static gguf_context_ptr make_meta() {
    const llm_arch arch = LLM_ARCH_QWEN35;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());

    const uint32_t n_embd = 256, n_head = 2, n_ff = 384, n_layer_trunk = 2, n_nextn = 1;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              uint32_t(256));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer_trunk + n_nextn);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,        n_nextn);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,     uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,     std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,             uint32_t(2));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");
    return ret;
}

struct mtp_env {
    llama_model_ptr   model;
    llama_context_ptr ctx_tgt;
    llama_context_ptr ctx_dft;
    common_params_speculative params;
    common_speculative * spec = nullptr;

    ~mtp_env() { common_speculative_free(spec); }
};

static bool silent_progress(float, void *) { return true; }

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

static void init_env(mtp_env & env, gguf_context * meta) {
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    mp.load_mtp          = true;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;

    size_t seed = 29;
    env.model.reset(llama_model_init_from_user(meta, set_tensor_data, &seed, mp));
    GGML_ASSERT(env.model && "model init failed");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 256;
    cp.n_batch         = 64;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = 1;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;
    env.ctx_tgt.reset(llama_init_from_model(env.model.get(), cp));
    GGML_ASSERT(env.ctx_tgt && "target context failed");

    llama_context_params cd = cp;
    cd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cd.n_rs_seq = 0;
    env.ctx_dft.reset(llama_init_from_model(env.model.get(), cd));
    GGML_ASSERT(env.ctx_dft && "MTP context failed");

    env.params.types                  = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    env.params.draft.n_max            = 4;
    env.params.draft.n_min            = 0;
    env.params.draft.p_min            = 0.0f;
    env.params.draft.backend_sampling = false; // full logits on the host, compared below
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

// target decode + MTP catch-up of toks at positions pos0.., seq 0
static void feed(mtp_env & env, const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], pos0 + (llama_pos) i, { 0 }, i + 1 == toks.size());
    }
    GGML_ASSERT(llama_decode(env.ctx_tgt.get(), b) == 0);
    GGML_ASSERT(common_speculative_process(env.spec, b));
    llama_batch_free(b);
}

static void new_request(mtp_env & env, const std::vector<llama_token> & prompt) {
    llama_memory_clear(llama_get_memory(env.ctx_tgt.get()), true);
    llama_memory_clear(llama_get_memory(env.ctx_dft.get()), true);
    common_speculative_begin(env.spec, 0, prompt);
    feed(env, prompt, 0);
}

struct draft_out {
    std::vector<llama_token> toks;
    std::vector<float>       logits; // the last draft decode's row
};

// one draft after n_past tokens, then drop its draft-KV cells again (the server's rollback of a rejected draft)
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
    const float * l = llama_get_logits_ith(env.ctx_dft.get(), -1);
    GGML_ASSERT(l != nullptr);
    out.logits.assign(l, l + N_VOCAB);
    llama_memory_seq_rm(llama_get_memory(env.ctx_dft.get()), 0, n_past, -1);
    return out;
}

static int n_fail = 0;

static bool same(const draft_out & a, const draft_out & b) {
    return a.toks == b.toks && a.logits.size() == b.logits.size() &&
           memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0;
}

static void expect(bool ok, const char * what, const draft_out & a, const draft_out & b) {
    size_t nd = 0;
    for (size_t i = 0; i < a.logits.size() && i < b.logits.size(); ++i) {
        nd += memcmp(&a.logits[i], &b.logits[i], sizeof(float)) != 0;
    }
    printf("%-58s %s  (draft %zu vs %zu tokens, %zu/%u logits differ)\n", what, ok ? "ok" : "FAIL",
           a.toks.size(), b.toks.size(), nd, N_VOCAB);
    if (!ok) {
        n_fail++;
    }
}

int main() {
    llama_log_set(quiet_log, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    gguf_context_ptr meta = make_meta();

    const std::vector<llama_token> P_B = tokens(40, 1);
    const std::vector<llama_token> P_U = tokens(29, 2);
    const std::vector<llama_token> P_C = tokens(9,  3); // continuation of B for the restore case
    const llama_token id_b = 7, id_c = 11;

    // 1. reference: B on a fresh instance (carrier = the ctor's zeros)
    draft_out ref;
    {
        mtp_env env;
        init_env(env, meta.get());
        new_request(env, P_B);
        ref = draft(env, (llama_pos) P_B.size(), id_b);
    }
    if (ref.toks.empty()) {
        printf("FAIL: the reference draft is empty\n");
        return 1;
    }

    // 2. U, then B twice on one instance
    {
        mtp_env env;
        init_env(env, meta.get());
        new_request(env, P_U);
        (void) draft(env, (llama_pos) P_U.size(), id_b);
        new_request(env, P_B);
        const draft_out b1 = draft(env, (llama_pos) P_B.size(), id_b);
        new_request(env, P_B);
        const draft_out b2 = draft(env, (llama_pos) P_B.size(), id_b);
        expect(same(ref, b1), "B after an unrelated request == B on a fresh instance", ref, b1);
        expect(same(ref, b2), "B after B == B on a fresh instance", ref, b2);
    }

    // 3. restore of a checkpointed carrier (pos > 0 keeps it)
    {
        mtp_env env;
        init_env(env, meta.get());

        std::vector<uint8_t> blob_u;
        new_request(env, P_U);
        GGML_ASSERT(common_speculative_get_state(env.spec, 0, blob_u));

        const llama_pos n_b = (llama_pos) P_B.size();

        // uninterrupted: B, C, draft
        new_request(env, P_B);
        feed(env, P_C, n_b);
        const draft_out cont = draft(env, n_b + (llama_pos) P_C.size(), id_c);

        // checkpoint after B, a foreign carrier, the restore, C, draft
        std::vector<uint8_t> blob_b;
        new_request(env, P_B);
        GGML_ASSERT(common_speculative_get_state(env.spec, 0, blob_b));
        common_speculative_set_state(env.spec, 0, blob_u);
        common_speculative_set_state(env.spec, 0, blob_b);
        feed(env, P_C, n_b);
        const draft_out restored = draft(env, n_b + (llama_pos) P_C.size(), id_c);

        // the foreign carrier left in place: the continuation pairs x_{n_b} with it
        new_request(env, P_B);
        common_speculative_set_state(env.spec, 0, blob_u);
        feed(env, P_C, n_b);
        const draft_out foreign = draft(env, n_b + (llama_pos) P_C.size(), id_c);

        expect(same(cont, restored), "restored carrier at pos > 0 == uninterrupted run", cont, restored);
        expect(!same(cont, foreign), "foreign carrier at pos > 0 != uninterrupted run (control)", cont, foreign);
    }

    llama_backend_free();

    if (n_fail) {
        printf("test-mtp-carrier: %d failures\n", n_fail);
        return 1;
    }
    printf("test-mtp-carrier: OK\n");
    return 0;
}
