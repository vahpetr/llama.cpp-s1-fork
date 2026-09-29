// [TAG_UP29638] Draft verification stops accepting after an EOG (upstream #29638), at both fork call sites:
// common_sampler_sample_and_accept_n() and the server's replay helper server_sample_and_accept_replay().
//
// A random CPU model (llama arch, 2 layers) with a 128-token SPM vocab. The greedy continuation t0 t1 .. t5 of a
// prompt is found first; the draft is [t1 t2 t3 t4] (every token matches the target), the verify batch is
// [t0 t1 t2 t3 t4]. The EOG is chosen after the fact with tokenizer.ggml.eos_token_id (the weights do not depend
// on it, so the chain is the same - checked):
//   no EOG in the draft                       -> t1 t2 t3 t4 t5  (all accepted + the bonus token)
//   EOG = t2 (draft index 1, middle)          -> t1 t2           (up to and including the EOG only)
//   EOG = t4 (the last draft token)           -> t1 t2 t3 t4 t5  (a trailing EOG is still accepted)
//   replay helper, 1 committed, EOG = t3      -> t1 t2 t3
//   replay helper, all committed, EOG = t4    -> t1 t2 t3 t4 t5  (committed tokens are re-emitted, not re-verified)

#include "common.h"
#include "log.h"
#include "sampling.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include "server-context.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
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
    std::normal_distribution<float> dis(0.0f, 2.0e-1f);

    // norm gains around 1 (not 0) so the chain depends on more than the last token's embedding
    const float base = strstr(tensor->name, "norm") != nullptr ? 1.0f : 0.0f;

    const int64_t ne = ggml_nelements(tensor);
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    std::vector<float> tmp(ne);
    for (int64_t i = 0; i < ne; i++) {
        tmp[i] = base + dis(gen);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

static gguf_context_ptr make_meta(llama_token eos) {
    const llm_arch arch = LLM_ARCH_LLAMA;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              uint32_t(128));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            uint32_t(128));
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 uint32_t(2));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         uint32_t(256));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);

    // SPM vocab: 0 <unk>, 1 <s> (control), the rest plain pieces; no text that matches a built-in EOG string
    std::vector<std::string> toks(N_VOCAB);
    std::vector<float>       scores(N_VOCAB);
    std::vector<int32_t>     types(N_VOCAB, LLAMA_TOKEN_TYPE_NORMAL);
    for (uint32_t i = 0; i < N_VOCAB; ++i) {
        toks[i]   = "p" + std::to_string(i);
        scores[i] = -(float) i;
    }
    toks[0] = "<unk>"; types[0] = LLAMA_TOKEN_TYPE_UNKNOWN;
    toks[1] = "<s>";   types[1] = LLAMA_TOKEN_TYPE_CONTROL;
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,      "llama");
    ms.add_kv(LLM_KV_TOKENIZER_LIST,       toks);
    ms.add_kv(LLM_KV_TOKENIZER_SCORES,     scores);
    gguf_set_arr_data(ret.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    ms.add_kv(LLM_KV_TOKENIZER_BOS_ID,     uint32_t(1));
    ms.add_kv(LLM_KV_TOKENIZER_EOS_ID,     uint32_t(eos));
    return ret;
}

static bool silent_progress(float, void *) { return true; }

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

struct env {
    llama_model_ptr   model;
    llama_context_ptr ctx;
};

static void init_env(env & e, llama_token eos) {
    gguf_context_ptr meta = make_meta(eos);
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;
    size_t seed = 29638;
    e.model.reset(llama_model_init_from_user(meta.get(), set_tensor_data, &seed, mp));
    GGML_ASSERT(e.model && "model init failed");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 128;
    cp.n_batch         = 64;
    cp.n_ubatch        = 64;
    cp.n_threads       = 2;
    cp.n_threads_batch = 2;
    e.ctx.reset(llama_init_from_model(e.model.get(), cp));
    GGML_ASSERT(e.ctx && "context failed");
}

static llama_token argmax(const float * l) {
    return (llama_token) (std::max_element(l, l + N_VOCAB) - l);
}

// prompt, then greedy t0 .. t5 one token at a time
static std::vector<llama_token> greedy_chain(env & e, const std::vector<llama_token> & prompt) {
    llama_memory_clear(llama_get_memory(e.ctx.get()), true);
    llama_batch b = llama_batch_init(64, 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) {
        common_batch_add(b, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
    }
    GGML_ASSERT(llama_decode(e.ctx.get(), b) == 0);
    std::vector<llama_token> chain;
    chain.push_back(argmax(llama_get_logits_ith(e.ctx.get(), -1)));
    for (int k = 0; k < 5; ++k) {
        common_batch_clear(b);
        common_batch_add(b, chain.back(), (llama_pos) (prompt.size() + chain.size() - 1), { 0 }, true);
        GGML_ASSERT(llama_decode(e.ctx.get(), b) == 0);
        chain.push_back(argmax(llama_get_logits_ith(e.ctx.get(), -1)));
    }
    llama_batch_free(b);
    return chain;
}

// the verify batch [t0 t1 t2 t3 t4] after the prompt; logits for all five rows
static void verify_batch(env & e, const std::vector<llama_token> & prompt, const std::vector<llama_token> & chain) {
    llama_memory_clear(llama_get_memory(e.ctx.get()), true);
    llama_batch b = llama_batch_init(64, 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) {
        common_batch_add(b, prompt[i], (llama_pos) i, { 0 }, false);
    }
    GGML_ASSERT(llama_decode(e.ctx.get(), b) == 0);
    common_batch_clear(b);
    for (int k = 0; k < 5; ++k) {
        common_batch_add(b, chain[k], (llama_pos) (prompt.size() + k), { 0 }, true);
    }
    GGML_ASSERT(llama_decode(e.ctx.get(), b) == 0);
    llama_batch_free(b);
}

static int n_fail = 0;

static std::string str(const std::vector<llama_token> & v) {
    std::string s;
    for (auto t : v) {
        s += (s.empty() ? "" : " ") + std::to_string(t);
    }
    return s;
}

static void run_case(const char * what, const std::vector<llama_token> & prompt, const std::vector<llama_token> & chain,
                     llama_token eos, size_t n_committed, bool replay, size_t n_expect) {
    env e;
    init_env(e, eos);
    GGML_ASSERT(greedy_chain(e, prompt) == chain && "the chain must not depend on the EOS id");
    verify_batch(e, prompt, chain);

    common_params_sampling sp;
    sp.temp = 0.0f; // greedy
    common_sampler_ptr smpl(common_sampler_init(e.model.get(), sp));

    const std::vector<int>         idxs  = { 0, 1, 2, 3, 4 };
    const std::vector<llama_token> draft = { chain[1], chain[2], chain[3], chain[4] };
    std::vector<llama_token> got;
    if (replay) {
        int n_mismatch = 0;
        got = server_sample_and_accept_replay(smpl.get(), e.ctx.get(), idxs, draft, n_committed, false, n_mismatch);
    } else {
        got = common_sampler_sample_and_accept_n(smpl.get(), e.ctx.get(), idxs, draft);
    }
    const std::vector<llama_token> want(chain.begin() + 1, chain.begin() + 1 + n_expect);
    const bool ok = got == want;
    printf("%-52s eos %3d: got [%s], want [%s] %s\n", what, eos, str(got).c_str(), str(want).c_str(), ok ? "ok" : "FAIL");
    if (!ok) {
        n_fail++;
    }
}

int main() {
    llama_log_set(quiet_log, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    const llama_token eos_none = 0; // <unk>: never produced by the chains used below (checked)

    // find a prompt whose chain has a first occurrence of t2 at draft index 1, of t3 at index 2 and of t4 at index 3
    std::vector<llama_token> prompt, chain;
    {
        env e;
        init_env(e, eos_none);
        std::mt19937 gen(1);
        std::uniform_int_distribution<llama_token> dis(2, N_VOCAB - 1);
        for (int attempt = 0; attempt < 200 && chain.empty(); ++attempt) {
            std::vector<llama_token> p(12);
            for (auto & t : p) {
                t = dis(gen);
            }
            std::vector<llama_token> c = greedy_chain(e, p);
            const bool distinct = c[1] != c[2] && c[3] != c[1] && c[3] != c[2] &&
                                  c[4] != c[1] && c[4] != c[2] && c[4] != c[3];
            const bool no_unk   = std::find(c.begin(), c.end(), eos_none) == c.end();
            if (distinct && no_unk) {
                prompt = p;
                chain  = c;
            }
        }
    }
    if (chain.empty()) {
        env e;
        init_env(e, eos_none);
        printf("chain of the prompt 2..13: [%s]\n", str(greedy_chain(e, {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13})).c_str());
        printf("FAIL: no prompt with a distinct greedy chain in 200 attempts\n");
        return 1;
    }
    printf("prompt [%s] -> greedy chain t0..t5 [%s]\n", str(prompt).c_str(), str(chain).c_str());

    run_case("accept_n: no EOG in the draft",                 prompt, chain, eos_none, 0, false, 5);
    run_case("accept_n: EOG in the middle (draft index 1)",   prompt, chain, chain[2], 0, false, 2);
    run_case("accept_n: EOG as the last draft token",         prompt, chain, chain[4], 0, false, 5);
    run_case("replay: 1 committed, no EOG",                   prompt, chain, eos_none, 1, true,  5);
    run_case("replay: 1 committed, EOG in the tail (index 2)", prompt, chain, chain[3], 1, true,  3);
    run_case("replay: all committed, EOG last",               prompt, chain, chain[4], 4, true,  5);

    llama_backend_free();

    if (n_fail) {
        printf("test-draft-accept-eog: %d failures\n", n_fail);
        return 1;
    }
    printf("test-draft-accept-eog: OK\n");
    return 0;
}
