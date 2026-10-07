// [TAG_GRAPH_CACHE] LLAMA_GRAPH_CACHE end to end on the synthetic qwen4exp model (+ 1 MTP block) of test-mtp-catchup:
// the same verify/draft sequence - verify widths cycling 1..6 over >= 48 rounds, an MTP draft and catch-up every
// round - runs three times in one process: cache off (LLAMA_GRAPH_CACHE=0, today), K=8, and K=8 with
// LLAMA_GRAPH_CACHE_CHECK=1 (every hit rebuilt and compared, abort on mismatch). Required: target logits, draft
// tokens + logits and the [qsa-ids] selection lines (LLAMA_QSA_DUMP_IDS=1, ctest env) bit-identical across the three;
// the K=8 runs hit the cache; kv_graph_span = 256 with GGML_ALLOC_GROW_SLACK=0 makes the compute buffers regrow once
// n_kv passes 256, which must show as invalidated snapshots (graph-cache inval > 0) with still-identical results.

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cinttypes>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
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
        for (auto & x : tmp) {
            x = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (auto & x : tmp) {
            x = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// the qwen4exp fixture of test-mtp-catchup (hc 4, low rank 8, QSA ratio 4, top-k 8, MoE 2 experts) + 1 MTP block
static gguf_context_ptr make_meta(uint32_t n_ctx_train) {
    const llm_arch arch = LLM_ARCH_QWEN4EXP;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_embd = 256, n_head = 2, n_ff = 384, n_layer_all = 3, n_embd_head = n_embd / n_head;
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              n_ctx_train);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer_all);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,        uint32_t(1));
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


// [TAG_GRAPH_CACHE] a synthetic second device for the split-input tests (--split): "FAKE" computes with an inner CPU
// backend, but its buffer type is not host and it supports only its own buffer type, while the CPU device does not
// support FAKE's. Layers are offloaded to it; the inputs stay on the CPU, so the scheduler splits the graph and the
// FAKE splits read inputs (inp_pos, masks, ...) only through backend copies - the Vulkan + CPU topology, on the CPU.
namespace fake {
static ggml_backend_t g_cpu = nullptr; // the inner compute backend
static ggml_backend_buffer_type g_buft;
static ggml_backend_device g_dev;
static ggml_backend_reg g_reg;

static void   buf_free(ggml_backend_buffer_t b) { ggml_backend_buffer_free((ggml_backend_buffer_t) b->context); }
static void * buf_base(ggml_backend_buffer_t b) { return ggml_backend_buffer_get_base((ggml_backend_buffer_t) b->context); }
static void   buf_memset(ggml_backend_buffer_t, ggml_tensor * t, uint8_t v, size_t o, size_t n) { memset((char *) t->data + o, v, n); }
static void   buf_set(ggml_backend_buffer_t, ggml_tensor * t, const void * d, size_t o, size_t n) { memcpy((char *) t->data + o, d, n); }
static void   buf_get(ggml_backend_buffer_t, const ggml_tensor * t, void * d, size_t o, size_t n) { memcpy(d, (const char *) t->data + o, n); }
static bool   buf_cpy(ggml_backend_buffer_t, const ggml_tensor *, ggml_tensor *) { return false; }
static void   buf_clear(ggml_backend_buffer_t b, uint8_t v) { memset(buf_base(b), v, b->size); }

static const char * buft_name(ggml_backend_buffer_type_t) { return "FAKE"; }
static ggml_backend_buffer_t buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t inner = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (inner == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_i iface = {
        buf_free, buf_base, nullptr, buf_memset, buf_set, buf_get, nullptr, nullptr, buf_cpy, buf_clear, nullptr,
    };
    return ggml_backend_buffer_init(buft, iface, inner, size);
}
static size_t buft_align(ggml_backend_buffer_type_t) { return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type()); }
static bool   buft_is_host(ggml_backend_buffer_type_t) { return false; }

static const char * be_name(ggml_backend_t) { return "FAKE"; }
static void be_free(ggml_backend_t b) { delete b; }
static ggml_status be_compute(ggml_backend_t, ggml_cgraph * gf) { return ggml_backend_graph_compute(g_cpu, gf); }

static const char * dev_name(ggml_backend_dev_t) { return "FAKE"; }
static void dev_memory(ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = (size_t) 16 << 30; }
static enum ggml_backend_dev_type dev_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void dev_props(ggml_backend_dev_t d, ggml_backend_dev_props * p) {
    memset(p, 0, sizeof(*p));
    p->name = p->description = "FAKE";
    p->type = dev_type(d);
    dev_memory(d, &p->memory_free, &p->memory_total);
}
static ggml_backend_t dev_init(ggml_backend_dev_t d, const char *) {
    static ggml_guid guid = { 0xfa, 0x4e };
    ggml_backend_i iface = {};
    iface.get_name      = be_name;
    iface.free          = be_free;
    iface.graph_compute = be_compute;
    return new ggml_backend { &guid, iface, d, nullptr };
}
static ggml_backend_buffer_type_t dev_buft(ggml_backend_dev_t) { return &g_buft; }
static bool dev_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return ggml_backend_dev_supports_op(ggml_backend_get_device(g_cpu), op);
}
static bool dev_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t buft) { return buft == &g_buft; }
static bool dev_offload_op(ggml_backend_dev_t, const ggml_tensor *) { return false; }

static const char * reg_name(ggml_backend_reg_t) { return "FAKE"; }
static size_t reg_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t reg_dev(ggml_backend_reg_t, size_t) { return &g_dev; }
static void * reg_proc(ggml_backend_reg_t, const char *) { return nullptr; }

static ggml_backend_dev_t init() {
    g_cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(g_cpu, 4);
    g_buft = { { buft_name, buft_alloc, buft_align, nullptr, nullptr, buft_is_host }, &g_dev, nullptr };
    g_reg  = { GGML_BACKEND_API_VERSION, { reg_name, reg_count, reg_dev, reg_proc }, nullptr };
    ggml_backend_device_i di = {};
    di.get_name        = dev_name;
    di.get_description = dev_name;
    di.get_memory      = dev_memory;
    di.get_type        = dev_type;
    di.get_props       = dev_props;
    di.init_backend    = dev_init;
    di.get_buffer_type = dev_buft;
    di.supports_op     = dev_supports_op;
    di.supports_buft   = dev_supports_buft;
    di.offload_op      = dev_offload_op;
    g_dev = { di, &g_reg, nullptr };
    return &g_dev;
}
} // namespace fake

static ggml_backend_dev_t g_split_dev = nullptr; // --split: offload to the FAKE device

static bool silent_progress(float, void *) { return true; }

static std::string g_log; // every "graph-cache:" line of the library

static void capture_log(ggml_log_level level, const char * text, void *) {
    if (strncmp(text, "graph-cache:", 12) == 0 || strstr(text, "max_nodes = ") != nullptr || strstr(text, "reallocating") != nullptr || strstr(text, "graph cache: K") != nullptr) {
        g_log += text;
        if (getenv("TEST_GRAPH_CACHE_VERBOSE")) {
            fputs(text, stdout);
        }
    }
    if (level >= GGML_LOG_LEVEL_WARN && strstr(text, "kqvis-prof") == nullptr) {
        fputs(text, stdout);
    }
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

struct run_out {
    std::vector<float>       logits; // target logits of every verify row + the draft logits, in order
    std::vector<llama_token> drafts;
    std::vector<std::string> qsa_ids; // [qsa-ids] lines without the graph counter
    int64_t hit[2] = {0, 0}, miss[2] = {0, 0}, check[2] = {0, 0}, inval[2] = {0, 0}; // [0] main, [1] mtp
    int64_t evict[2] = {0, 0}, clear[2] = {0, 0}, check2[2] = {0, 0}, check_sched[2] = {0, 0}, park_fail[2] = {0, 0}, bypass[2] = {0, 0}, check2_state_only[2] = {0, 0};
};

static run_out run(gguf_context * meta, const char * k, const char * check, const std::string & stderr_path, int fail_take = -1,
                   bool backend_sampling = false) {
    run_out out;
    ggml_backend_sched_snapshot_fail_inject(fail_take); // -1 = off; n = the take after n successful ones fails
    common_set_env("LLAMA_GRAPH_CACHE",       k);
    common_set_env("LLAMA_GRAPH_CACHE_CHECK", check);
    g_log.clear();
    fflush(stderr);
    { std::ofstream truncate(stderr_path); }
    GGML_ASSERT(freopen(stderr_path.c_str(), "a", stderr) != nullptr); // append: a crash backtrace must not overwrite
    setvbuf(stderr, nullptr, _IOLBF, 0); // line-buffered: an abort keeps its message
    {
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = silent_progress;
        mp.load_mtp          = true;
        ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
        if (getenv("TEST_GRAPH_CACHE_GPU") != nullptr) {
            devs[0] = nullptr; // all devices (Vulkan run)
        }
        if (g_split_dev != nullptr) {
            devs[0] = g_split_dev; // --split: layers on FAKE, inputs on the CPU
        }
        mp.devices = devs[0] ? devs : nullptr;
        size_t seed = 31;
        llama_model_ptr model(llama_model_init_from_user(meta, set_tensor_data, &seed, mp));
        GGML_ASSERT(model);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 1024;
        cp.n_batch         = 128;
        cp.n_ubatch        = 64;
        cp.n_seq_max       = 1;
        cp.n_threads       = 4;
        cp.n_threads_batch = 4;
        cp.kv_graph_span   = 256; // worst-case reserve up to n_kv 256: past it the compute buffers regrow
        llama_context_ptr ctx_tgt(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(ctx_tgt);
        llama_context_params cd = cp;
        cd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cd.n_rs_seq = 0;
        llama_context_ptr ctx_dft(llama_init_from_model(model.get(), cd));
        GGML_ASSERT(ctx_dft);

        common_params_speculative params;
        params.types                  = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.draft.n_max            = 4;
        params.draft.n_min            = 0;
        params.draft.p_min            = 0.0f;
        params.draft.backend_sampling = backend_sampling; // prod default true: the draft context then bypasses the graph cache
        params.draft.ctx_tgt          = ctx_tgt.get();
        params.draft.ctx_dft          = ctx_dft.get();
        common_speculative * spec = common_speculative_init(params, 1);
        GGML_ASSERT(spec);

        auto feed = [&](const std::vector<llama_token> & toks, llama_pos pos0) {
            llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(b, toks[i], pos0 + (llama_pos) i, { 0 }, true);
            }
            GGML_ASSERT(llama_decode(ctx_tgt.get(), b) == 0);
            for (size_t i = 0; i < toks.size(); ++i) {
                const float * l = llama_get_logits_ith(ctx_tgt.get(), (int32_t) i);
                out.logits.insert(out.logits.end(), l, l + N_VOCAB);
            }
            GGML_ASSERT(common_speculative_process(spec, b));
            llama_synchronize(ctx_dft.get());
            llama_batch_free(b);
        };

        // prompt (two ubatches), then rounds: an MTP draft of up to 4 tokens, then a verify batch of width 1..6
        llama_pos n_past = 0;
        const auto prompt = tokens(200, 7);
        feed(std::vector<llama_token>(prompt.begin(), prompt.begin() + 100), 0); // n_batch 128
        feed(std::vector<llama_token>(prompt.begin() + 100, prompt.end()), 100);
        n_past += (llama_pos) prompt.size();
        llama_token id_last = prompt.back();
        const int n_rounds = 60;
        for (int r = 0; r < n_rounds; ++r) {
            if (r == 40) {
                // a full ubatch at n_kv > kv_graph_span: the compute buffers regrow (GGML_ALLOC_GROW_SLACK=0), so the
                // snapshots of the decode graphs cached before are stale from here on
                const auto pp = tokens(64, 999);
                feed(pp, n_past);
                n_past += (llama_pos) pp.size();
                id_last = pp.back();
            }
            if (r == 20) {
                // the target's scheduler is replaced at the next decode (a reserve, as after an adapter change): the
                // cache is cleared and CHECK must remake its scratch scheduler for the new one
                llama_set_causal_attn(ctx_tgt.get(), false);
                llama_set_causal_attn(ctx_tgt.get(), true);
            }
            std::vector<llama_token> drafted;
            auto & dp = common_speculative_get_draft_params(spec, 0);
            dp.drafting = true;
            dp.n_max    = -1;
            dp.n_past   = n_past;
            dp.id_last  = id_last;
            dp.prompt   = nullptr;
            dp.result   = &drafted;
            common_speculative_draft(spec);
            const float * l = llama_get_logits_ith(ctx_dft.get(), -1);
            if (l != nullptr) {
                out.logits.insert(out.logits.end(), l, l + N_VOCAB);
            }
            out.drafts.insert(out.drafts.end(), drafted.begin(), drafted.end());
            out.drafts.push_back(-1);
            llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), 0, n_past, -1);

            const int width = 1 + r % 6;
            const auto v = tokens((size_t) width, 1000 + (uint32_t) r);
            feed(v, n_past);
            n_past += width;
            id_last = v.back();
        }
        common_speculative_free(spec);
    } // contexts destroyed: their "graph-cache:" totals are logged
    fflush(stderr);
    GGML_ASSERT(freopen("/dev/tty", "w", stderr) != nullptr || freopen("/dev/null", "w", stderr) != nullptr);

    std::ifstream f(stderr_path);
    std::string line;
    const std::regex g_field(" g=[0-9]+");
    while (std::getline(f, line)) {
        if (line.rfind("[qsa-ids]", 0) == 0) {
            out.qsa_ids.push_back(std::regex_replace(line, g_field, ""));
        }
    }
    std::istringstream ls(g_log);
    while (std::getline(ls, line)) {
        char ctx[16] = {0};
        int kk = 0;
        long long h = 0, m = 0, c = 0, inv = 0, ev = 0, cl = 0, c2 = 0, cs = 0, pf = 0, bp = 0;
        if (sscanf(line.c_str(), "graph-cache: ctx=%15s K=%d hit=%lld miss=%lld check=%lld inval=%lld evict=%lld clear=%lld check2=%lld check_sched=%lld park_fail=%lld bypass=%lld",
                   ctx, &kk, &h, &m, &c, &inv, &ev, &cl, &c2, &cs, &pf, &bp) == 12) {
            const int i = strcmp(ctx, "mtp") == 0 ? 1 : 0;
            out.hit[i] += h; out.miss[i] += m; out.check[i] += c; out.inval[i] += inv;
            out.evict[i] += ev; out.clear[i] += cl; out.check2[i] += c2; out.check_sched[i] += cs; out.park_fail[i] += pf; out.bypass[i] += bp;
            if (const char * so = strstr(line.c_str(), " check2_state_only=")) {
                out.check2_state_only[i] += atoll(so + strlen(" check2_state_only="));
            }
        }
    }
    return out;
}

static int n_fail = 0;

static void expect(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    n_fail += ok ? 0 : 1;
}

template <typename T>
static bool same(const std::vector<T> & a, const std::vector<T> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// [TAG_GRAPH_CACHE] memory transitions on the target context (QSA env of ctest, recurrent rollback depth per verify
// batch): rollback (seq_rm of the last batch's tail), state save -> more rounds -> state load, shrink (clear + 20
// tokens) -> regrow (a 100-token batch), clear. A unified-cache purge / hole refill is not reachable through the API
// for this hybrid M-RoPE model (position overwrite rejected, recurrent seq_rm only at the tail). Every row's logits are recorded; cached graphs must be reused
// across the transitions only where they stay valid.
static run_out run_transitions(gguf_context * meta, const char * k, const char * check, const std::string & stderr_path) {
    run_out out;
    common_set_env("LLAMA_GRAPH_CACHE",       k);
    common_set_env("LLAMA_GRAPH_CACHE_CHECK", check);
    g_log.clear();
    fflush(stderr);
    { std::ofstream truncate(stderr_path); }
    GGML_ASSERT(freopen(stderr_path.c_str(), "a", stderr) != nullptr);
    setvbuf(stderr, nullptr, _IOLBF, 0);
    {
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = silent_progress;
        mp.load_mtp          = true;
        ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
        mp.devices = devs;
        size_t seed = 31;
        llama_model_ptr model(llama_model_init_from_user(meta, set_tensor_data, &seed, mp));
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 1024; cp.n_batch = 128; cp.n_ubatch = 64; cp.n_seq_max = 1; cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.kv_graph_span = 256;
        cp.n_rs_seq      = 4; // recurrent rollback (a verify batch of up to 4 tokens)
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(ctx);
        llama_memory_t mem = llama_get_memory(ctx.get());
        llama_pos n_past = 0;
        uint32_t tag = 0;
        auto feed = [&](const std::vector<llama_token> & toks, llama_pos pos0) {
            llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
            for (size_t i = 0; i < toks.size(); ++i) {
                common_batch_add(b, toks[i], pos0 + (llama_pos) i, { 0 }, true);
            }
            llama_set_rs_rollback_depth(ctx.get(), toks.size() <= 4 ? (uint32_t) toks.size() : 0);
            GGML_ASSERT(llama_decode(ctx.get(), b) == 0);
            for (size_t i = 0; i < toks.size(); ++i) {
                const float * l = llama_get_logits_ith(ctx.get(), (int32_t) i);
                out.logits.insert(out.logits.end(), l, l + N_VOCAB);
            }
            llama_batch_free(b);
        };
        auto rounds = [&](int n) {
            for (int r = 0; r < n; ++r) {
                const int w = 1 + r % 4;
                feed(tokens((size_t) w, 2000 + tag++), n_past);
                n_past += w;
            }
        };
        feed(tokens(120, 77), 0); n_past = 120;
        rounds(12);
        // rollback: drop 3 of the last verify batch's 4 tokens (recurrent state rolled back), then continue
        GGML_ASSERT(llama_memory_seq_rm(mem, 0, n_past - 3, -1)); n_past -= 3;
        rounds(8);
        // state save -> diverge -> state load
        std::vector<uint8_t> state(llama_state_get_size(ctx.get()));
        GGML_ASSERT(llama_state_get_data(ctx.get(), state.data(), state.size()) == state.size());
        const llama_pos n_saved = n_past;
        rounds(8);
        GGML_ASSERT(llama_state_set_data(ctx.get(), state.data(), state.size()) == state.size());
        n_past = n_saved;
        rounds(8);
        // shrink -> regrow (a hybrid recurrent sequence cannot be cut in the middle: clear, 20 tokens, then a
        // 100-token batch that grows n_kv again)
        llama_memory_clear(mem, true);
        feed(tokens(20, 3500), 0); n_past = 20;
        rounds(8);
        feed(tokens(100, 4000), n_past); n_past += 100;
        rounds(8);
        // clear
        llama_memory_clear(mem, true); n_past = 0;
        feed(tokens(90, 5000), 0); n_past = 90;
        rounds(12);
    }
    fflush(stderr);
    GGML_ASSERT(freopen("/dev/tty", "w", stderr) != nullptr || freopen("/dev/null", "w", stderr) != nullptr);
    std::istringstream ls(g_log);
    std::string line;
    while (std::getline(ls, line)) {
        char cx[16] = {0};
        int kk = 0;
        long long h = 0, m = 0, c = 0, inv = 0, ev = 0, cl = 0, c2 = 0;
        if (sscanf(line.c_str(), "graph-cache: ctx=%15s K=%d hit=%lld miss=%lld check=%lld inval=%lld evict=%lld clear=%lld check2=%lld",
                   cx, &kk, &h, &m, &c, &inv, &ev, &cl, &c2) == 9) {
            out.hit[0] += h; out.miss[0] += m; out.check[0] += c; out.check2[0] += c2;
        }
    }
    return out;
}

// [TAG_GRAPH_CACHE] memory of the cache at the brain's node budget: n_ubatch 2048 gives this qwen4exp context
// graph_max_nodes = max(2048*40, 32*n_tensors) = 81920, as the prod brain (n_ubatch 2048, 1224 tensors). Decode
// widths 1..9 (each a new graph: a miss) and report VmSize / VmRSS after each; with K=8 every miss from the second
// on parks one snapshot (+ keeps one llm_graph_result).
static void proc_mem(int64_t & vsz_kib, int64_t & rss_kib) {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmSize:", 0) == 0) { vsz_kib = atoll(line.c_str() + 7); }
        if (line.rfind("VmRSS:",  0) == 0) { rss_kib = atoll(line.c_str() + 6); }
    }
}

static int mem_report(gguf_context * meta, const char * k, int64_t & per_vsz_kib, int64_t & per_rss_kib) {
    common_set_env("LLAMA_GRAPH_CACHE", k);
    common_set_env("LLAMA_GRAPH_CACHE_CHECK", "0");
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    mp.load_mtp          = true;
    ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    mp.devices = devs;
    size_t seed = 31;
    llama_model_ptr model(llama_model_init_from_user(meta, set_tensor_data, &seed, mp));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 2048; cp.n_seq_max = 1; cp.n_threads = 4; cp.n_threads_batch = 4;
    g_log.clear();
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(ctx);
    unsigned long long max_nodes = 0;
    const char * mn = strstr(g_log.c_str(), "max_nodes = ");
    if (mn) { max_nodes = strtoull(mn + 12, nullptr, 10); }
    llama_pos n_past = 0;
    int64_t v0 = 0, r0 = 0, v = 0, r = 0;
    printf("mem K=%s graph_max_nodes=%llu\n", k, max_nodes);
    for (int w = 1; w <= 9; ++w) {
        const auto t = tokens((size_t) w, 500 + (uint32_t) w);
        llama_batch b = llama_batch_init(w, 0, 1);
        for (int i = 0; i < w; ++i) {
            common_batch_add(b, t[i], n_past + i, { 0 }, i == w - 1);
        }
        GGML_ASSERT(llama_decode(ctx.get(), b) == 0);
        llama_batch_free(b);
        n_past += w;
        proc_mem(v, r);
        if (w == 1) { v0 = v; r0 = r; }
        printf("mem K=%s width %d: VmSize %lld MiB, VmRSS %lld MiB\n", k, w, (long long) v / 1024, (long long) r / 1024);
    }
    per_vsz_kib = (v - v0) / 8; // 8 more graphs after the first
    per_rss_kib = (r - r0) / 8;
    printf("mem K=%s: per extra graph VmSize %+.1f MiB, VmRSS %+.1f MiB\n", k, per_vsz_kib / 1024.0, per_rss_kib / 1024.0);
    return (int) max_nodes;
}

// a CHECK=1 run with one induced corruption of a cached entry (LLAMA_GRAPH_CACHE_CHECK_INJECT) in a child process:
// it must abort with the matching graph-cache check message
static bool aborts_with(gguf_context * meta, const char * inject, const std::string & path, const char * msg,
                        const char * check = "1", const char * mode = "plain") {
    fflush(stdout);
    GGML_UNUSED(meta);
    // fork + exec (a plain fork of this multi-threaded process can deadlock): the child is this binary in inject mode
    const pid_t pid = fork();
    if (pid == 0) {
        execl("/proc/self/exe", "test-graph-cache", "--inject", inject, path.c_str(), check, mode, (char *) nullptr);
        _exit(127);
    }
    int st = 0;
    waitpid(pid, &st, 0);
    std::ifstream f(path);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const bool found = text.find(msg) != std::string::npos;
    printf("inject %s: %s, message %s\n", inject, WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT ? "SIGABRT" : "no abort",
           found ? "found" : "missing");
    return WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT && found;
}

int main(int argc, char ** argv) {
    llama_log_set(capture_log, nullptr);
    common_set_env("GGML_ALLOC_GROW_SLACK", "0");
    const std::string dir = argc == 2 ? argv[1] : ".";
    gguf_context_ptr meta = make_meta(4096);
    if (argc >= 2 && strcmp(argv[1], "--mem") == 0) {
        int64_t v0, r0, v8, r8;
        mem_report(meta.get(), "0", v0, r0);
        const int max_nodes = mem_report(meta.get(), "8", v8, r8);
        const double sv = (v8 - v0) / 1024.0, sr = (r8 - r0) / 1024.0;
        printf("mem: graph_max_nodes %d; per cached graph (K=8 minus K=0): VmSize %+.1f MiB, VmRSS %+.1f MiB\n", max_nodes, sv, sr);
        const bool ok = max_nodes == 2048*40 && sv < 256.0;
        printf("test-graph-cache --mem: %s (virtual reservation per cached graph < 256 MiB)\n", ok ? "OK" : "FAILED");
        return ok ? 0 : 1;
    }
    if (argc == 6 && strcmp(argv[1], "--inject") == 0) {
        // --inject <seam> <stderr> <CHECK level> <plain|split>
        common_set_env("LLAMA_GRAPH_CACHE_CHECK_INJECT", argv[2]);
        common_set_env("GGML_NO_BACKTRACE", "1");
        if (strcmp(argv[5], "split") == 0) {
            g_split_dev = fake::init();
        }
        run(meta.get(), "8", argv[4], argv[3]);
        return 0; // not aborted: the check missed the corruption
    }
    if (argc == 3 && strcmp(argv[1], "--check3") == 0) {
        // [TAG_GRAPH_CACHE] CHECK=3: the fresh plan prepares its inputs by a second set_inputs (one-shot inputs
        // copied), both plans' inputs pre-filled with different sentinels
        const std::string cdir = argv[2];
        const run_out c_off = run(meta.get(), "0", "0", cdir + "/test-graph-cache-check3-off.stderr");
        const run_out c_ch3 = run(meta.get(), "8", "3", cdir + "/test-graph-cache-check3.stderr");
        const run_out t_off = run_transitions(meta.get(), "0", "0", cdir + "/test-graph-cache-check3-trans-off.stderr");
        const run_out t_ch3 = run_transitions(meta.get(), "8", "3", cdir + "/test-graph-cache-check3-trans.stderr");
        printf("check3: main hit=%" PRId64 " check2=%" PRId64 "; mtp hit=%" PRId64 " check2=%" PRId64 "; transitions hit=%" PRId64 " check2=%" PRId64 "\n",
               c_ch3.hit[0], c_ch3.check2[0], c_ch3.hit[1], c_ch3.check2[1], t_ch3.hit[0], t_ch3.check2[0]);
        expect(same(c_off.logits, c_ch3.logits) && same(c_off.drafts, c_ch3.drafts) && c_off.qsa_ids == c_ch3.qsa_ids,
               "CHECK=3: logits, drafts and [qsa-ids] identical to cache off");
        expect(c_ch3.hit[0] > 0 && c_ch3.check2[0] == c_ch3.hit[0] && c_ch3.check2[1] == c_ch3.hit[1],
               "CHECK=3: every hit executed fresh from independently prepared inputs, outputs + state bytes equal");
        expect(same(t_off.logits, t_ch3.logits) && t_ch3.hit[0] > 0 && t_ch3.check2[0] == t_ch3.hit[0],
               "CHECK=3 transitions (recurrent rollback: the one-shot s_copy copied): identical, green on every hit");
        expect(aborts_with(meta.get(), "stale", cdir + "/test-graph-cache-check3-stale.stderr",
                           "differs between the cached and a fresh plan", "3"),
               "CHECK=3 aborts on an induced partial input refresh");
        expect(!aborts_with(meta.get(), "stale", cdir + "/test-graph-cache-check2-stale.stderr",
                            "differs between the cached and a fresh plan", "2"),
               "CHECK=2 does not see the same partial refresh (common-mode inputs; the reason for CHECK=3)");
        printf("test-graph-cache --check3: %s\n", n_fail == 0 ? "OK" : "FAILED");
        return n_fail == 0 ? 0 : 1;
    }
    if (argc == 3 && strcmp(argv[1], "--kvonly") == 0) {
        // [TAG_GRAPH_CACHE] state-only graphs: LLAMA_MTP_CATCHUP_ATTN=0 makes the zero-output MTP catch-up ubatches
        // KV-only (the cache writes are the graph's only roots); CHECK=2 compares their state bytes
        const std::string kdir = argv[2];
        common_set_env("LLAMA_MTP_CATCHUP_ATTN", "0");
        const run_out k_off = run(meta.get(), "0", "0", kdir + "/test-graph-cache-kvonly-off.stderr");
        const run_out k_ch2 = run(meta.get(), "8", "2", kdir + "/test-graph-cache-kvonly-check2.stderr");
        printf("kvonly: %zu logits; CHECK=2 main hit=%" PRId64 " check2=%" PRId64 "; mtp hit=%" PRId64 " check2=%" PRId64 " state-only=%" PRId64 "\n",
               k_off.logits.size(), k_ch2.hit[0], k_ch2.check2[0], k_ch2.hit[1], k_ch2.check2[1], k_ch2.check2_state_only[1]);
        expect(!k_off.logits.empty() && same(k_off.logits, k_ch2.logits) && same(k_off.drafts, k_ch2.drafts) && k_off.qsa_ids == k_ch2.qsa_ids,
               "kvonly: K=8 + CHECK=2 identical to cache off");
        expect(k_ch2.check2[0] == k_ch2.hit[0] && k_ch2.check2[1] == k_ch2.hit[1] && k_ch2.check2_state_only[1] > 0,
               "kvonly: CHECK=2 on every hit, KV-only catch-up hits compared by their state (state-only > 0)");
        printf("test-graph-cache --kvonly: %s\n", n_fail == 0 ? "OK" : "FAILED");
        return n_fail == 0 ? 0 : 1;
    }
    if (argc == 3 && strcmp(argv[1], "--split") == 0) {
        // [TAG_GRAPH_CACHE] split-only inputs: the layers on FAKE, the inputs on the CPU (see namespace fake)
        const std::string sdir = argv[2];
        g_split_dev = fake::init();
        const run_out s_off = run(meta.get(), "0", "0", sdir + "/test-graph-cache-split-off.stderr");
        const run_out s_ch2 = run(meta.get(), "8", "2", sdir + "/test-graph-cache-split-check2.stderr");
        printf("split: %zu logits; CHECK=2 main hit=%" PRId64 " check2=%" PRId64 "; mtp hit=%" PRId64 " check2=%" PRId64 "\n",
               s_off.logits.size(), s_ch2.hit[0], s_ch2.check2[0], s_ch2.hit[1], s_ch2.check2[1]);
        expect(!s_off.logits.empty() && same(s_off.logits, s_ch2.logits) && same(s_off.drafts, s_ch2.drafts) && s_off.qsa_ids == s_ch2.qsa_ids,
               "split (FAKE + CPU): K=8 + CHECK=2 identical to cache off");
        expect(s_ch2.hit[0] > 0 && s_ch2.check2[0] == s_ch2.hit[0] && s_ch2.check2[1] == s_ch2.hit[1],
               "split: CHECK=2 green on every hit (inputs consumed only through backend copies are transferred)");
        expect(aborts_with(meta.get(), "split-input", sdir + "/test-graph-cache-split-inject.stderr",
                           "differs between the cached and a fresh plan", "2", "split"),
               "split: CHECK=2 aborts when an input consumed only through a backend copy is poisoned in the fresh plan");
        printf("test-graph-cache --split: %s\n", n_fail == 0 ? "OK" : "FAILED");
        return n_fail == 0 ? 0 : 1;
    }

    const run_out off = run(meta.get(), "0", "0", dir + "/test-graph-cache-off.stderr");
    const run_out on  = run(meta.get(), "8", "0", dir + "/test-graph-cache-on.stderr");
    const run_out chk = run(meta.get(), "8", "1", dir + "/test-graph-cache-check.stderr");
    const run_out ch2 = run(meta.get(), "8", "2", dir + "/test-graph-cache-check2.stderr");

    if (const char * dump = getenv("TEST_GRAPH_CACHE_DUMP")) {
        // the cache-off run, for a byte comparison with another build (K = 0 must equal the parent commit)
        FILE * f = fopen(dump, "wb");
        GGML_ASSERT(f);
        fwrite(off.logits.data(), sizeof(float), off.logits.size(), f);
        fwrite(off.drafts.data(), sizeof(llama_token), off.drafts.size(), f);
        for (const auto & l : off.qsa_ids) {
            fprintf(f, "%s\n", l.c_str());
        }
        fclose(f);
    }

    const run_out pfl = run(meta.get(), "8", "1", dir + "/test-graph-cache-parkfail.stderr", 10);
    // the production draft configuration: backend sampling ON in the draft (MTP) context
    const run_out obs = run(meta.get(), "0", "0", dir + "/test-graph-cache-bs-off.stderr", -1, true);
    const run_out cbs = run(meta.get(), "8", "1", dir + "/test-graph-cache-bs-check.stderr", -1, true);
    for (const run_out * r : { &off, &on, &chk, &ch2, &pfl, &obs, &cbs }) {
        printf("run: %zu logits, %zu draft tokens, %zu [qsa-ids] lines; main hit=%" PRId64 " miss=%" PRId64 " check=%" PRId64
               " inval=%" PRId64 "; mtp hit=%" PRId64 " miss=%" PRId64 " check=%" PRId64 " inval=%" PRId64 "\n",
               r->logits.size(), r->drafts.size(), r->qsa_ids.size(), r->hit[0], r->miss[0], r->check[0], r->inval[0],
               r->hit[1], r->miss[1], r->check[1], r->inval[1]);
    }

    expect(!off.qsa_ids.empty(), "the QSA selection ran ([qsa-ids] lines present; LLAMA_QSA_DUMP_IDS=1 + the QSA env)");
    expect(same(off.logits, on.logits),  "logits: cache off == K=8");
    expect(same(off.logits, chk.logits), "logits: cache off == K=8 + CHECK");
    expect(same(off.drafts, on.drafts) && same(off.drafts, chk.drafts), "draft tokens identical");
    expect(off.qsa_ids == on.qsa_ids && off.qsa_ids == chk.qsa_ids, "[qsa-ids] identical");
    expect(off.hit[0] == 0 && off.miss[0] == 0, "cache off: no graph-cache activity");
    expect(on.hit[0] > on.miss[0] && on.hit[1] > 0, "K=8: the target hits more than it misses, the MTP context hits");
    expect(chk.check[0] == chk.hit[0] && chk.check[1] == chk.hit[1] && chk.hit[0] > 0, "CHECK: every hit rebuilt and compared");
    expect(pfl.park_fail[0] + pfl.park_fail[1] == 1 && same(off.logits, pfl.logits) && same(off.drafts, pfl.drafts) &&
           pfl.check[0] == pfl.hit[0] && pfl.hit[0] > 0,
           "a snapshot allocation failure drops the resident graph cleanly: identical, CHECK green (park_fail == 1)");
    expect(chk.check_sched[0] >= 2 && chk.clear[0] >= 1,
           "CHECK: the scratch scheduler was remade after the target's scheduler was replaced (check_sched >= 2)");
    expect(same(off.logits, ch2.logits) && same(off.drafts, ch2.drafts) && off.qsa_ids == ch2.qsa_ids,
           "CHECK=2: logits, drafts and [qsa-ids] identical to cache off");
    expect(ch2.check2[0] == ch2.hit[0] && ch2.check2[1] == ch2.hit[1] && ch2.check[0] == ch2.hit[0] && ch2.hit[0] > 0,
           "CHECK=2: every hit also executed fresh from the same state, outputs + state bytes equal");
    expect(on.inval[0] + on.inval[1] > 0, "regrow past kv_graph_span invalidated snapshots (inval > 0)");

    const run_out t_off = run_transitions(meta.get(), "0", "0", dir + "/test-graph-cache-trans-off.stderr");
    const run_out t_chk = run_transitions(meta.get(), "8", "1", dir + "/test-graph-cache-trans-check.stderr");
    printf("transitions: cache off == K=8 + CHECK=1: %s (%" PRId64 " hits)\n", same(t_off.logits, t_chk.logits) ? "yes" : "NO", t_chk.hit[0]);
    fflush(stdout);
    const run_out t_ch2 = run_transitions(meta.get(), "8", "2", dir + "/test-graph-cache-trans-check2.stderr");
    printf("transitions: %zu logits; K=8+CHECK hit=%" PRId64 " miss=%" PRId64 " check=%" PRId64 "; CHECK=2 hit=%" PRId64 " check2=%" PRId64 "\n",
           t_off.logits.size(), t_chk.hit[0], t_chk.miss[0], t_chk.check[0], t_ch2.hit[0], t_ch2.check2[0]);
    expect(same(t_off.logits, t_chk.logits) && same(t_off.logits, t_ch2.logits),
           "transitions (rollback, state load, shrink -> regrow, clear): logits cache off == K=8 + CHECK=1 == CHECK=2");
    expect(t_chk.hit[0] > 0 && t_chk.check[0] == t_chk.hit[0] && t_ch2.check2[0] == t_ch2.hit[0] && t_ch2.hit[0] > 0,
           "transitions: the cache hits and CHECK (1 and 2) is green on every hit");

    printf("backend sampling: mtp bypass=%" PRId64 " clear=%" PRId64 "; main hit=%" PRId64 " check=%" PRId64 "\n",
           cbs.bypass[1], cbs.clear[1], cbs.hit[0], cbs.check[0]);
    expect(same(obs.logits, cbs.logits) && same(obs.drafts, cbs.drafts) && obs.qsa_ids == cbs.qsa_ids,
           "draft backend sampling ON (prod): K=8 + CHECK identical to cache off");
    expect(cbs.hit[1] == 0 && cbs.miss[1] == 0 && cbs.bypass[1] > 0,
           "draft backend sampling ON: the MTP context bypasses the cache (no hit/miss, bypass > 0)");
    expect(cbs.hit[0] > cbs.miss[0] && cbs.check[0] == cbs.hit[0] && cbs.bypass[0] == 0,
           "draft backend sampling ON: the target context still hits, CHECK on every hit, no bypass");

    expect(aborts_with(meta.get(), "ptr",   dir + "/test-graph-cache-inject-ptr.stderr",   "allocation differs from the manifest"),
           "CHECK aborts on an induced compute-pointer corruption");
    expect(aborts_with(meta.get(), "view",  dir + "/test-graph-cache-inject-view.stderr",  "view inconsistent"),
           "CHECK aborts on an induced view corruption");
    // the CPU-only plan has one split without inputs: the seam routes a phantom copy into split 0 (n_inputs
    // differs); with split inputs (Vulkan + CPU) it feeds input 0's copy from another source ("copy routing differs")
    expect(aborts_with(meta.get(), "route", dir + "/test-graph-cache-inject-route.stderr", "the cached split plan differs from a fresh build"),
           "CHECK aborts on an induced copy-routing mismatch");

    printf("test-graph-cache: %s\n", n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
