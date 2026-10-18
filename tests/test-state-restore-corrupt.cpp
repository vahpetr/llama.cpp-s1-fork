// [TAG_RESTORE_HARDEN] a damaged per-sequence state (truncated slot file / blob, overwritten header words, garbage
// tensor data) must be an error - never an abort, never an allocation sized by a damaged count - and a failed restore
// must leave the target sequence empty in every cache while the other sequences keep their state byte for byte.
//
// CPU qwen4exp fixture of test-state-portable (hybrid-idx target: attention KV + recurrent + QSA indexer K; MTP
// draft: a plain KV cache, or attention + indexer under LLAMA_MTP_QSA=1), np2 per-stream. Seq 0 and seq 1 hold
// different prompts; seq 1's state is saved (target: seq data + slot file, draft: seq data, ON_DEVICE target
// snapshot) and then restored into seq 1 in damaged forms:
//  TR  every truncation length in the first 512 bytes, then a stride over the rest, and size - 1: the restore
//      returns 0, seq 1 is empty in the target (attention, recurrent, indexer) / the draft, seq 0's state is unchanged;
//  CW  each of the first 1024 header / metadata words overwritten with 0xffffffff, 0x7fffffff, 0x00100000 (1M cells)
//      and n_ctx + 1: the restore returns 0 (then the same post-conditions) or the full length (a word the reader
//      cannot validate, e.g. a position or a K value); the process' peak RSS stays within 256 MiB of the start;
//  NAN the tensor region filled with 0xff (upstream #27530 test): same as CW;
//  FILE truncated and count-damaged slot files (llama_state_seq_load_file): 0, same post-conditions;
//  DEV  the ON_DEVICE snapshot (in-process checkpoints): truncated / overwritten headers return 0 (no abort);
//  OK  after all of it the intact blobs restore, re-save byte-identical, and seq 0 / seq 1 continue with the logits
//      of a fresh context that never saw a damaged restore;
//  P   (#30514) PARTIAL_ONLY: the dense draft (plain KV cache) writes no cells (equal size for 40 and 60 tokens);
//      the sparse draft (hybrid-idx recr_none) keeps its cells ([TAG_MTP_QSA]); the target trunk = recurrent only;
//  ROT (#28498) the state descriptor names the KV rotation: q8_0 KV with / without LLAMA_ATTN_ROT_DISABLE differ.

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

static const uint32_t N_VOCAB   = 128;
static const uint32_t N_CTX_SEQ = 512;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-1f);
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (auto & x : tmp) { x = dis(gen); }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (auto & x : tmp) { x = ggml_fp32_to_fp16(dis(gen)); }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// the test-mtp-catchup qwen4exp fixture (hc 4, QSA ratios 4, MoE, 1 MTP block) - as test-state-portable
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
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,          uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,        1.0f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,             uint32_t(2));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");
    return ret;
}

static int n_fail = 0;
static int n_ok   = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        printf("FAIL %s\n", what.c_str());
        fflush(stdout);
        n_fail++;
    } else {
        n_ok++;
    }
}

template <typename T>
static bool eq(const std::vector<T> & a, const std::vector<T> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

static bool silent_progress(float, void *) { return true; }

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN && getenv("TEST_STATE_CORRUPT_VERBOSE")) {
        fputs(text, stderr);
    }
}

static long peak_rss_kib() {
    struct rusage ru = {};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

static llama_model_ptr g_model;

struct env {
    llama_context_ptr ctx_tgt;
    llama_context_ptr ctx_dft;
    common_params_speculative params;
    common_speculative * spec = nullptr;
    ~env() { common_speculative_free(spec); }
};

static std::unique_ptr<env> make_env() {
    auto e = std::make_unique<env>();
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = N_CTX_SEQ * 2;
    cp.n_batch         = 128;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = 2;
    cp.kv_unified      = false;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    e->ctx_tgt.reset(llama_init_from_model(g_model.get(), cp));
    GGML_ASSERT(e->ctx_tgt && "target context failed");
    llama_context_params cd = cp;
    cd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cd.n_rs_seq = 0;
    e->ctx_dft.reset(llama_init_from_model(g_model.get(), cd));
    GGML_ASSERT(e->ctx_dft && "MTP context failed");
    e->params.types           = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    e->params.draft.n_max     = 4;
    e->params.draft.ctx_tgt   = e->ctx_tgt.get();
    e->params.draft.ctx_dft   = e->ctx_dft.get();
    e->spec = common_speculative_init(e->params, 2);
    GGML_ASSERT(e->spec && "draft-mtp init failed");
    return e;
}

static std::vector<llama_token> tokens(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<llama_token> dis(0, N_VOCAB - 1);
    std::vector<llama_token> t(n);
    for (auto & x : t) { x = dis(gen); }
    return t;
}

// target decode of seq (every row an output) + the MTP catch-up; returns the target logits
static std::vector<float> feed(env & e, llama_seq_id seq, const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], pos0 + (llama_pos) i, { seq }, true);
    }
    GGML_ASSERT(llama_decode(e.ctx_tgt.get(), b) == 0);
    std::vector<float> logits(toks.size() * N_VOCAB);
    for (size_t i = 0; i < toks.size(); ++i) {
        memcpy(logits.data() + i * N_VOCAB, llama_get_logits_ith(e.ctx_tgt.get(), (int32_t) i), N_VOCAB * sizeof(float));
    }
    GGML_ASSERT(common_speculative_process(e.spec, b));
    llama_batch_free(b);
    return logits;
}

static std::vector<uint8_t> seq_data(llama_context * ctx, llama_seq_id seq, llama_state_seq_flags flags = 0) {
    std::vector<uint8_t> d(llama_state_seq_get_size_ext(ctx, seq, flags));
    GGML_ASSERT(llama_state_seq_get_data_ext(ctx, d.data(), d.size(), seq, flags) == d.size());
    return d;
}

static std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static void write_file(const std::string & path, const std::vector<uint8_t> & d) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write((const char *) d.data(), (std::streamsize) d.size());
}

struct fixture {
    std::unique_ptr<env> e;
    std::vector<uint8_t> tgt0, dft0;   // seq 0's state, must survive every failed restore into seq 1
    std::vector<uint8_t> tgt1, dft1;   // seq 1's saved state (the restore source)
    std::vector<uint8_t> file1;        // seq 1's slot file
    std::vector<llama_token> p1;
    std::string path;
    int n_bad = 0;                     // damaged restores that failed (returned 0)
    int n_accepted = 0;                // damaged restores the reader could not tell from an intact one
};

static bool seq_empty(llama_context * ctx, llama_seq_id seq) {
    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq) == -1;
}

// one damaged restore into seq 1 of ctx (target or draft); returns true when it was refused
static bool try_restore(fixture & fx, bool target, const std::vector<uint8_t> & blob, const std::string & what) {
    llama_context * ctx = target ? fx.e->ctx_tgt.get() : fx.e->ctx_dft.get();
    llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);
    const size_t n = llama_state_seq_set_data_ext(ctx, blob.data(), blob.size(), 1, 0);
    if (n == 0) {
        fx.n_bad++;
        check(seq_empty(ctx, 1), what + ": refused restore leaves seq 1 empty (pos " +
                std::to_string(llama_memory_seq_pos_min(llama_get_memory(ctx), 1)) + ".." +
                std::to_string(llama_memory_seq_pos_max(llama_get_memory(ctx), 1)) + ")");
    } else {
        fx.n_accepted++;
        check(n == blob.size(), what + ": an accepted restore consumed the whole blob");
        llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);
    }
    check(eq(seq_data(ctx, 0), target ? fx.tgt0 : fx.dft0), what + ": seq 0 state unchanged");
    return n == 0;
}

static void run_mutations(fixture & fx, bool target) {
    const std::vector<uint8_t> & src = target ? fx.tgt1 : fx.dft1;
    const char * tn = target ? "target" : "draft";
    const long rss0 = peak_rss_kib();

    // TR: truncations
    int n_tr = 0, n_tr_bad = 0;
    for (size_t L = 0; L < src.size(); L = L < 512 ? L + 1 : L + std::max<size_t>(1, src.size() / 97)) {
        n_tr++;
        n_tr_bad += try_restore(fx, target, std::vector<uint8_t>(src.begin(), src.begin() + L),
                std::string(tn) + " TR len " + std::to_string(L)) ? 1 : 0;
    }
    n_tr++;
    n_tr_bad += try_restore(fx, target, std::vector<uint8_t>(src.begin(), src.end() - 1), std::string(tn) + " TR size-1") ? 1 : 0;
    check(n_tr_bad == n_tr, std::string(tn) + " TR: every truncated blob refused");
    printf("%s TR: %d truncations, %d refused\n", tn, n_tr, n_tr_bad);

    // CW: overwritten words in the header / metadata region
    const uint32_t vals[4] = { 0xffffffffu, 0x7fffffffu, 0x00100000u, N_CTX_SEQ + 1 };
    int n_cw = 0, n_cw_bad = 0;
    const size_t n_words = std::min<size_t>(src.size() / 4, 1024);
    for (size_t w = 0; w < n_words; ++w) {
        for (uint32_t v : vals) {
            std::vector<uint8_t> b = src;
            memcpy(b.data() + 4 * w, &v, 4);
            if (eq(b, src)) {
                continue;
            }
            n_cw++;
            n_cw_bad += try_restore(fx, target, b, std::string(tn) + " CW word " + std::to_string(w) + " = " + std::to_string(v)) ? 1 : 0;
        }
    }
    printf("%s CW: %d damaged blobs, %d refused, %d accepted (unvalidated words)\n", tn, n_cw, n_cw_bad, n_cw - n_cw_bad);
    check(n_cw_bad > 0, std::string(tn) + " CW: damaged headers are refused");

    // NAN: the tensor region overwritten (upstream #27530: 0xff from 4096 to 3/4 of the blob)
    if (src.size() >= 3 * 4096) {
        std::vector<uint8_t> b = src;
        std::fill(b.begin() + 4096, b.end() - b.size() / 4, 0xff);
        try_restore(fx, target, b, std::string(tn) + " NAN");
    }

    const long drss = peak_rss_kib() - rss0;
    printf("%s: peak RSS growth over the damaged restores %ld KiB\n", tn, drss);
    check(drss < 256 * 1024, std::string(tn) + ": no allocation sized by a damaged count (peak RSS growth < 256 MiB)");
}

int main() {
    llama_log_set(log_cb, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    gguf_context_ptr meta = make_meta(4096);
    {
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = silent_progress;
        mp.load_mtp          = true;
        ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
        mp.devices = devs;
        size_t seed = 31;
        g_model.reset(llama_model_init_from_user(meta.get(), set_tensor_data, &seed, mp));
        GGML_ASSERT(g_model && "model init failed");
    }
    const char * mode = getenv("LLAMA_MTP_QSA");
    const bool sparse = mode && atoi(mode);
    printf("mode: %s MTP draft\n", sparse ? "sparse (LLAMA_MTP_QSA=1)" : "dense");

    char tmpl[] = "/tmp/test-state-restore-corrupt-XXXXXX";
    const char * dir = mkdtemp(tmpl);
    GGML_ASSERT(dir && "mkdtemp failed");

    const auto P0 = tokens(40, 11);
    const auto P1 = tokens(60, 12);

    fixture fx;
    fx.e = make_env();
    feed(*fx.e, 0, P0, 0);
    feed(*fx.e, 1, P1, 0);
    fx.tgt0 = seq_data(fx.e->ctx_tgt.get(), 0);
    fx.dft0 = seq_data(fx.e->ctx_dft.get(), 0);
    fx.tgt1 = seq_data(fx.e->ctx_tgt.get(), 1);
    fx.dft1 = seq_data(fx.e->ctx_dft.get(), 1);
    fx.p1   = P1;
    fx.path = std::string(dir) + "/seq1.bin";
    GGML_ASSERT(llama_state_seq_save_file(fx.e->ctx_tgt.get(), fx.path.c_str(), 1, P1.data(), P1.size()) > 0);
    fx.file1 = read_file(fx.path);
    printf("blobs: target %zu B, draft %zu B, slot file %zu B\n", fx.tgt1.size(), fx.dft1.size(), fx.file1.size());
    for (int t = 0; t < 2; ++t) {
        const auto & b = t ? fx.dft1 : fx.tgt1;
        printf("%s blob words 0..11:", t ? "draft" : "target");
        for (size_t w = 0; w < 12 && 4 * w + 4 <= b.size(); ++w) {
            uint32_t v = 0;
            memcpy(&v, b.data() + 4 * w, 4);
            printf(" %u", v);
        }
        printf("\n");
    }

    // P (#30514): PARTIAL_ONLY sizes for 40 vs 60 tokens
    {
        const llama_state_seq_flags fp = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
        const size_t t0 = llama_state_seq_get_size_ext(fx.e->ctx_tgt.get(), 0, fp);
        const size_t t1 = llama_state_seq_get_size_ext(fx.e->ctx_tgt.get(), 1, fp);
        const size_t d0 = llama_state_seq_get_size_ext(fx.e->ctx_dft.get(), 0, fp);
        const size_t d1 = llama_state_seq_get_size_ext(fx.e->ctx_dft.get(), 1, fp);
        printf("P: PARTIAL_ONLY target %zu / %zu B, draft %zu / %zu B (40 / 60 tokens)\n", t0, t1, d0, d1);
        check(t0 == t1, "P: the target trunk partial state (recurrent only) does not grow with the tokens");
        if (sparse) {
            check(d1 > d0, "P: the sparse draft (hybrid-idx recr_none) keeps its KV cells under PARTIAL_ONLY");
        } else {
            check(d0 == d1 && d0 < fx.dft0.size(), "P: the dense draft (plain KV cache) writes no cells under PARTIAL_ONLY");
        }
    }

    run_mutations(fx, true);
    run_mutations(fx, false);

    // FILE: truncated / count-damaged slot files
    {
        int n = 0, n_bad = 0;
        std::vector<llama_token> toks(P1.size() + 8);
        auto try_file = [&](const std::vector<uint8_t> & d, const std::string & what) {
            write_file(fx.path + ".bad", d);
            llama_memory_seq_rm(llama_get_memory(fx.e->ctx_tgt.get()), 1, -1, -1);
            size_t n_tok = 0;
            const size_t r = llama_state_seq_load_file(fx.e->ctx_tgt.get(), (fx.path + ".bad").c_str(), 1, toks.data(), toks.size(), &n_tok);
            n++;
            if (r == 0) {
                n_bad++;
                check(seq_empty(fx.e->ctx_tgt.get(), 1), what + ": refused load leaves seq 1 empty");
            }
            llama_memory_seq_rm(llama_get_memory(fx.e->ctx_tgt.get()), 1, -1, -1);
            check(eq(seq_data(fx.e->ctx_tgt.get(), 0), fx.tgt0), what + ": seq 0 state unchanged");
        };
        for (size_t L : { (size_t) 0, (size_t) 4, (size_t) 11, (size_t) 12, (size_t) 100, fx.file1.size() / 2, fx.file1.size() - 1 }) {
            try_file(std::vector<uint8_t>(fx.file1.begin(), fx.file1.begin() + L), "FILE len " + std::to_string(L));
        }
        // the token count, then the first words of the state
        for (size_t off : { (size_t) 8, (size_t) 12 + 4 * P1.size(), (size_t) 16 + 4 * P1.size(), (size_t) 20 + 4 * P1.size() }) {
            for (uint32_t v : { 0xffffffffu, 0x00100000u }) {
                std::vector<uint8_t> d = fx.file1;
                memcpy(d.data() + off, &v, 4);
                try_file(d, "FILE word @" + std::to_string(off) + " = " + std::to_string(v));
            }
        }
        check(n_bad == n, "FILE: every damaged slot file refused");
        printf("FILE: %d damaged slot files, %d refused\n", n, n_bad);
    }

    // DEV: ON_DEVICE snapshot (in-process checkpoint form), damaged headers
    {
        const llama_state_seq_flags fd = LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
        const std::vector<uint8_t> dev = seq_data(fx.e->ctx_tgt.get(), 1, fd);
        int n = 0, n_bad = 0;
        auto try_dev = [&](const std::vector<uint8_t> & b, const std::string & what) {
            llama_memory_seq_rm(llama_get_memory(fx.e->ctx_tgt.get()), 1, -1, -1);
            const size_t r = llama_state_seq_set_data_ext(fx.e->ctx_tgt.get(), b.data(), b.size(), 1, fd);
            n++;
            if (r == 0) {
                n_bad++;
                check(seq_empty(fx.e->ctx_tgt.get(), 1), what + ": refused restore leaves seq 1 empty");
            }
            llama_memory_seq_rm(llama_get_memory(fx.e->ctx_tgt.get()), 1, -1, -1);
            check(eq(seq_data(fx.e->ctx_tgt.get(), 0), fx.tgt0), what + ": seq 0 state unchanged");
        };
        for (size_t L : { (size_t) 0, (size_t) 3, (size_t) 7, (size_t) 8, (size_t) 16, dev.size() / 2, dev.size() - 1 }) {
            try_dev(std::vector<uint8_t>(dev.begin(), dev.begin() + L), "DEV len " + std::to_string(L));
        }
        for (size_t w : { (size_t) 0, (size_t) 1, (size_t) 2, (size_t) 3 }) {
            std::vector<uint8_t> b = dev;
            const uint32_t v = 0x7fffffffu;
            memcpy(b.data() + 4 * w, &v, 4);
            try_dev(b, "DEV word " + std::to_string(w));
        }
        check(n_bad == n, "DEV: every damaged ON_DEVICE snapshot refused");
        printf("DEV: %d damaged snapshots, %d refused\n", n, n_bad);
    }

    // OK: the intact states restore and continue exactly like a context that never saw a damaged restore
    {
        llama_memory_seq_rm(llama_get_memory(fx.e->ctx_tgt.get()), 1, -1, -1);
        llama_memory_seq_rm(llama_get_memory(fx.e->ctx_dft.get()), 1, -1, -1);
        check(llama_state_seq_set_data_ext(fx.e->ctx_tgt.get(), fx.tgt1.data(), fx.tgt1.size(), 1, 0) == fx.tgt1.size(), "OK: target restore");
        check(llama_state_seq_set_data_ext(fx.e->ctx_dft.get(), fx.dft1.data(), fx.dft1.size(), 1, 0) == fx.dft1.size(), "OK: draft restore");
        check(eq(seq_data(fx.e->ctx_tgt.get(), 1), fx.tgt1), "OK: target re-save byte-identical");
        check(eq(seq_data(fx.e->ctx_dft.get(), 1), fx.dft1), "OK: draft re-save byte-identical");

        auto ref = make_env();
        feed(*ref, 0, P0, 0);
        feed(*ref, 1, P1, 0);
        const auto cont = tokens(5, 13);
        for (llama_seq_id s = 0; s < 2; ++s) {
            const llama_pos p = (llama_pos) (s ? P1.size() : P0.size());
            const auto a = feed(*fx.e, s, cont, p);
            const auto b = feed(*ref,  s, cont, p);
            check(eq(a, b), "OK: seq " + std::to_string(s) + " continuation logits byte-equal to a fresh context");
        }
    }

    printf("damaged restores: %d refused, %d accepted (unvalidated words)\n", fx.n_bad, fx.n_accepted);

    // ROT (#28498): the descriptor separates rotated from unrotated q8_0 KV
    {
        auto make_q8 = [&](bool disable) {
            setenv("LLAMA_ATTN_ROT_DISABLE", disable ? "1" : "0", 1);
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = N_CTX_SEQ;
            cp.n_batch = 64;
            cp.n_ubatch = 64;
            cp.type_k = GGML_TYPE_Q8_0;
            cp.type_v = GGML_TYPE_Q8_0;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            return llama_context_ptr(llama_init_from_model(g_model.get(), cp));
        };
        auto a = make_q8(false);
        auto b = make_q8(true);
        unsetenv("LLAMA_ATTN_ROT_DISABLE");
        if (!a || !b) {
            printf("ROT: SKIP (no q8_0 KV context on this build)\n");
        } else {
            const std::string da = llama_state_seq_descriptor(a.get());
            const std::string db = llama_state_seq_descriptor(b.get());
            const auto rot = [](const std::string & d) {
                const size_t i = d.find("rot_k=");
                return i == std::string::npos ? std::string("?") : d.substr(i, 15);
            };
            printf("ROT: q8_0 KV descriptor %s, with LLAMA_ATTN_ROT_DISABLE=1 %s\n", rot(da).c_str(), rot(db).c_str());
            check(da.find("rot_k=1,rot_v=1") != std::string::npos, "ROT: q8_0 KV is rotated (descriptor rot_k=1,rot_v=1)");
            check(db.find("rot_k=0,rot_v=0") != std::string::npos, "ROT: LLAMA_ATTN_ROT_DISABLE=1 prints rot_k=0,rot_v=0");
            check(da != db, "ROT: the descriptors (disk-cache key part) differ");
        }
    }

    std::remove(fx.path.c_str());
    std::remove((fx.path + ".bad").c_str());
    rmdir(dir);

    g_model.reset();
    llama_backend_free();

    printf("%s: %d/%d checks\n", n_fail ? "FAIL" : "PASS", n_ok, n_ok + n_fail);
    return n_fail ? 1 : 0;
}
