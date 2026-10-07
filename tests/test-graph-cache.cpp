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
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

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

static bool silent_progress(float, void *) { return true; }

static std::string g_log; // every "graph-cache:" line of the library

static void capture_log(ggml_log_level level, const char * text, void *) {
    if (strncmp(text, "graph-cache:", 12) == 0 || strstr(text, "reallocating") != nullptr || strstr(text, "graph cache: K") != nullptr) {
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
    int64_t evict[2] = {0, 0}, clear[2] = {0, 0}, check2[2] = {0, 0}, check_sched[2] = {0, 0}, park_fail[2] = {0, 0}, bypass[2] = {0, 0};
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

// a CHECK=1 run with one induced corruption of a cached entry (LLAMA_GRAPH_CACHE_CHECK_INJECT) in a child process:
// it must abort with the matching graph-cache check message
static bool aborts_with(gguf_context * meta, const char * inject, const std::string & path, const char * msg) {
    fflush(stdout);
    GGML_UNUSED(meta);
    // fork + exec (a plain fork of this multi-threaded process can deadlock): the child is this binary in inject mode
    const pid_t pid = fork();
    if (pid == 0) {
        execl("/proc/self/exe", "test-graph-cache", "--inject", inject, path.c_str(), (char *) nullptr);
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
    if (argc == 4 && strcmp(argv[1], "--inject") == 0) {
        common_set_env("LLAMA_GRAPH_CACHE_CHECK_INJECT", argv[2]);
        common_set_env("GGML_NO_BACKTRACE", "1");
        run(meta.get(), "8", "1", argv[3]);
        return 0; // not aborted: the check missed the corruption
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
