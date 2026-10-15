// [TAG_QSA_POOLED_F16_MIRROR] (A2a, contrib 0240) the f16 mirror of the QSA pooled-key store on a GPU, through the model
// graph: the test-mtp-qsa-restore fixture (qwen4exp, 2 trunk layers + 1 MTP block, QSA ratio 4, here 4 indexer heads) on the
// first GPU device with the prod QSA env (pooled cache, indexed remap, compact bias, chunk 128, vis) + LLAMA_MTP_QSA=1, so
// BOTH stores (the target's and the MTP draft memory's) carry a mirror. One child per arm (the knobs are read once):
//   off: LLAMA_QSA_POOLED_F16_MIRROR=0      on: =1 + LLAMA_QSA_POOLED_F16_MIRROR_CHECK=1 (every graph: mirror == f16(f32))
// Scenario: a cold fill of 384 rows in 128-row target ubatches (the first plan pools every block: the FULL-POOL cpy
// writer), then 18 rounds of draft -> target decode of 1..6 rows + MTP catch-up of the same rows (widths >= 3 take the
// GEMM path that reads the mirror, 1-2 the f32 mat-vec; the DIRTY set_rows writer), then the target sequence state is
// saved / cleared / restored and the draft sequence dropped (the legacy restore), then 12 more rounds.
// PASS: every target logits row and every draft logits row bit-identical between the arms; the on arm's stores report
// f16_mirror = 1 (target and draft), a full-pool plan (stats.n_full > 0) and dirty plans; CHECK passed (no abort).
// Not a ctest: needs a GPU whose QSA_SCORE takes the f16 store (Vulkan coopmat1); without one it prints SKIP and exits 0.
// usage: test-qsa-f16-mirror            (parent: runs both children, compares)
//        test-qsa-f16-mirror --child    (one arm, the env decides; prints the [mirror] lines)

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   uint32_t(4)); // the brain's 4 heads: the score GEMM sums 4
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

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR || strstr(text, "qsa_pooled_f16_mirror") != nullptr || strstr(text, "[qsa-f16-mirror]") != nullptr) {
        fputs(text, stderr);
    }
}

static uint64_t fnv(const void * p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    for (size_t i = 0; i < n; ++i) { h = (h ^ ((const uint8_t *) p)[i]) * 0x100000001b3ull; }
    return h;
}

static const llama_qsa_pooled_store * store_of(llama_context * ctx) {
    const auto * mem = dynamic_cast<const llama_memory_hybrid_idx *>(llama_get_memory(ctx));
    return mem ? mem->get_qsa_store() : nullptr;
}

static int child() {
    llama_log_set(quiet_log, nullptr);
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gpu) { gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); }
    if (!gpu) { printf("[mirror] SKIP no GPU\n"); return 0; }

    gguf_context_ptr meta = make_meta(4096);
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    mp.load_mtp          = true;
    mp.n_gpu_layers      = 999;
    ggml_backend_dev_t devs[2] = { gpu, nullptr };
    mp.devices = devs;
    size_t seed = 31;
    llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, &seed, mp));
    GGML_ASSERT(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 128; cp.n_ubatch = 128; cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr tgt(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(tgt);
    llama_context_params cd = cp;
    cd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cd.type_k = GGML_TYPE_F16; cd.type_v = GGML_TYPE_F16;
    llama_context_ptr dft(llama_init_from_model(model.get(), cd));
    GGML_ASSERT(dft);

    common_params_speculative params;
    params.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.draft.n_max   = 4;
    params.draft.n_min   = 0;
    params.draft.p_min   = 0.0f;
    params.draft.ctx_tgt = tgt.get();
    params.draft.ctx_dft = dft.get();
    common_speculative * spec = common_speculative_init(params, 1);
    GGML_ASSERT(spec);
    common_speculative_begin(spec, 0, {});

    std::mt19937 rng(77);
    llama_pos pos = 0;
    llama_token id_last = 0;
    int step = 0;

    auto feed = [&](int n, const char * what) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            common_batch_add(b, (llama_token) (rng() % N_VOCAB), pos + i, { 0 }, true);
        }
        GGML_ASSERT(llama_decode(tgt.get(), b) == 0);
        uint64_t h = 0xcbf29ce484222325ull;
        for (int i = 0; i < n; ++i) { h = fnv(llama_get_logits_ith(tgt.get(), i), N_VOCAB*sizeof(float), h); }
        const float * last = llama_get_logits_ith(tgt.get(), n - 1);
        id_last = (llama_token) (std::max_element(last, last + N_VOCAB) - last);
        GGML_ASSERT(common_speculative_process(spec, b));
        llama_synchronize(dft.get());
        printf("[mirror] %d %s n=%d pos=%d tgt=%016" PRIx64 "\n", step++, what, n, (int) pos, h);
        llama_batch_free(b);
        pos += n;
    };
    auto rounds = [&](int n) {
        for (int r = 0; r < n; ++r) {
            std::vector<llama_token> drafted;
            auto & dp = common_speculative_get_draft_params(spec, 0);
            dp.drafting = true; dp.n_max = -1; dp.n_past = pos; dp.id_last = id_last; dp.prompt = nullptr; dp.result = &drafted;
            common_speculative_draft(spec);
            llama_synchronize(dft.get());
            uint64_t hd = 0;
            if (!drafted.empty()) {
                hd = fnv(llama_get_logits_ith(dft.get(), -1), N_VOCAB*sizeof(float));
                hd = fnv(drafted.data(), drafted.size()*sizeof(llama_token), hd);
            }
            printf("[mirror] %d draft n=%zu dft=%016" PRIx64 "\n", step++, drafted.size(), hd);
            llama_memory_seq_rm(llama_get_memory(dft.get()), 0, pos, -1);
            feed(1 + r % 6, "round");
            common_speculative_accept(spec, 0, 0);
        }
    };

    for (int c = 0; c < 3; ++c) { feed(128, "fill"); }
    rounds(18);

    // the target state saved / cleared / restored; the draft sequence dropped as the server's legacy restore does
    const size_t sz = llama_state_seq_get_size(tgt.get(), 0);
    std::vector<uint8_t> st(sz);
    const size_t got = llama_state_seq_get_data(tgt.get(), st.data(), st.size(), 0);
    llama_memory_clear(llama_get_memory(tgt.get()), true);
    GGML_ASSERT(llama_state_seq_set_data(tgt.get(), st.data(), got, 0) == got);
    llama_memory_seq_rm(llama_get_memory(dft.get()), 0, -1, -1);
    common_speculative_reset_state(spec, 0);
    printf("[mirror] %d restore bytes=%zu\n", step++, got);
    rounds(12);

    const std::pair<const char *, llama_context *> ctxs[2] = { { "tgt", tgt.get() }, { "dft", dft.get() } };
    for (const auto & [name, c] : ctxs) {
        const auto * s = store_of(c);
        printf("[mirror-store] %s present=%d f16_mirror=%d layers=%zu n_plan=%" PRIu64 " n_full=%" PRIu64 " n_dirty=%" PRIu64 "\n", name,
               s != nullptr, s ? (int) s->f16_mirror : 0, s ? s->layers_f16.size() : 0, s ? s->stats.n_plan : 0,
               s ? s->stats.n_full : 0, s ? s->stats.n_dirty : 0);
    }
    common_speculative_free(spec);
    return 0;
}

static std::vector<std::string> run_child(const char * self, const char * env) {
    const std::string cmd = std::string(env) + " '" + self + "' --child";
    std::vector<std::string> lines;
    FILE * p = popen(cmd.c_str(), "r");
    GGML_ASSERT(p);
    char buf[512];
    while (fgets(buf, sizeof(buf), p)) { lines.emplace_back(buf); }
    const int rc = pclose(p);
    lines.push_back("rc=" + std::to_string(rc) + "\n");
    return lines;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {
        return child();
    }
    const char * QSA_ENV = "LLAMA_MTP_QSA=1 LLAMA_QSA_POOLED_CACHE=1 LLAMA_QSA_COMPACT_BIAS=1 LLAMA_QSA_CENSUS_INC=1 LLAMA_QSA_INPUTS_INC=1 "
                           "LLAMA_QSA_POOL_REMAP=indexed LLAMA_QSA_CHUNK=128 LLAMA_KQ_VIS=1 LLAMA_QSA_VIS=1 LLAMA_QSA_PREFILL=indexed "
                           "GGML_VK_QSA_FUSED_SELECT=1 LLAMA_QSA_RELU_INPLACE=1 LLAMA_MTP_CATCHUP_ATTN=0";
    const auto off = run_child(argv[0], (std::string(QSA_ENV) + " LLAMA_QSA_POOLED_F16_MIRROR=0").c_str());
    const auto on  = run_child(argv[0], (std::string(QSA_ENV) + " LLAMA_QSA_POOLED_F16_MIRROR=1 LLAMA_QSA_POOLED_F16_MIRROR_CHECK=1").c_str());
    for (const auto & l : on) { fputs(l.c_str(), stdout); }
    if (!on.empty() && on[0].find("SKIP") != std::string::npos) { printf("test-qsa-f16-mirror: SKIP (no GPU)\n"); return 0; }

    bool ok = off.back() == "rc=0\n" && on.back() == "rc=0\n";
    size_t n_cmp = 0, n_diff = 0;
    std::vector<std::string> a, b;
    for (const auto & l : off) { if (l.rfind("[mirror] ", 0) == 0) { a.push_back(l); } }
    for (const auto & l : on)  { if (l.rfind("[mirror] ", 0) == 0) { b.push_back(l); } }
    ok = ok && a.size() == b.size() && !a.empty();
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        n_cmp++;
        if (a[i] != b[i]) { if (n_diff++ == 0) { printf("first difference:\n  off: %s  on:  %s", a[i].c_str(), b[i].c_str()); } }
    }
    ok = ok && n_diff == 0;
    // the on arm really mirrored both stores, ran a full-pool plan and dirty plans
    bool tgt_ok = false, dft_ok = false;
    for (const auto & l : on) {
        uint64_t n_plan = 0, n_full = 0, n_dirty = 0; int present = 0, mir = 0; size_t layers = 0; char name[8] = {0};
        if (sscanf(l.c_str(), "[mirror-store] %7s present=%d f16_mirror=%d layers=%zu n_plan=%" SCNu64 " n_full=%" SCNu64 " n_dirty=%" SCNu64,
                   name, &present, &mir, &layers, &n_plan, &n_full, &n_dirty) == 7) {
            const bool good = present && mir && layers > 0 && n_full > 0 && n_dirty > 0;
            if (strcmp(name, "tgt") == 0) { tgt_ok = good; } else { dft_ok = good; }
        }
    }
    ok = ok && tgt_ok && dft_ok;
    printf("test-qsa-f16-mirror: %s (%zu lines compared, %zu differ; target store mirrored+full+dirty %d, draft store %d)\n",
           ok ? "PASS" : "FAIL", n_cmp, n_diff, (int) tgt_ok, (int) dft_ok);
    return ok ? 0 : 1;
}
