// [TAG_CELL_EXT_RESTORE] a per-sequence restore must give the indexer KV cache the cell ext (M-RoPE x / y + token) the
// live cells hold.
//
// The qwen4exp indexer cache runs with rope NONE (hparams_idx: its keys are rotated after pooling), so its n_pos is 1:
//  - with PLE it serializes the ext, but llama_kv_cache::state_read_meta rebuilt it through a 1-position ubatch and
//    left x = y = 0 (S0.1 oracle C3: the restore gap idx[s0].meta(ext=12), 2 x 65 536 fields);
//  - without PLE (the MTP draft GGUF) it serializes no ext at all, and the restore left x = y = tok = 0.
// x / y are an image token's 2D position: the QSA cell rank / visibility / sec_pos and the pooled-key stamp read them.
//
// CPU qwen4exp fixture of test-state-portable (hybrid-idx target, 1 MTP block), twice: as built (no PLE) and with a
// PLE layer (written to a temporary GGUF, the user-metadata loader cannot provide the PLE table). Per fixture:
//  T  target: text, a 3x4 "image" embedding chunk (M-RoPE positions as mtmd lays them out), text; seq save file ->
//     restore into a fresh context: every cell's {pos, ext} of the attention AND the indexer cache equals the live
//     one, a re-save is byte-identical to the live save, and a continuation (text + another image chunk) gives
//     byte-identical logits to the in-memory continuation;
//  D  MTP draft memory (LLAMA_MTP_QSA=1 only, otherwise it is a plain KV cache): seq data with flags 0 into a fresh
//     context, and the speculative rollback checkpoint form (PARTIAL_ONLY, same context after seq_rm): the attention
//     and indexer cells' {pos, ext} equal the live ones.
//
// usage: test-cell-ext-restore [dump_dir]   (dump_dir: keeps the T save files live-<fixture>.bin / rest-<fixture>.bin)

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
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

static const uint32_t N_VOCAB = 128;
static const uint32_t N_EMBD  = 256;
static const uint32_t N_CTX   = 512;

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

// the test-mtp-catchup / test-state-portable qwen4exp fixture (hc 4, QSA ratios 4, MoE, 1 MTP block)
static void add_meta(llama_model_saver & ms, uint32_t n_ctx_train) {
    const llm_arch arch = LLM_ARCH_QWEN4EXP;
    const uint32_t n_embd = N_EMBD, n_head = 2, n_ff = 384, n_layer_trunk = 2, n_nextn = 1;
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
}

// PLE on layer 0 (a recurrent layer, as build_ple needs): 2-grams + 3-grams x 2 heads = 4 heads x 64 = n_embd
static const uint32_t PLE_NGRAM = 3, PLE_HPN = 2, PLE_KERNEL = 4, PLE_HEAD_DIM = 64, PLE_HEAD_VOCAB = 64;

static void add_ple_meta(llama_model_saver & ms) {
    const uint32_t n_heads = (PLE_NGRAM - 1) * PLE_HPN;
    std::vector<uint64_t> offs, sizes;
    for (uint32_t h = 0; h < n_heads; ++h) {
        offs.push_back((uint64_t) h * PLE_HEAD_VOCAB);
        sizes.push_back(PLE_HEAD_VOCAB);
    }
    ms.add_kv(LLM_KV_PLE_LAYERS,                 std::vector<uint32_t>({0}));
    ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,             PLE_NGRAM);
    ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,        PLE_HPN);
    ms.add_kv(LLM_KV_PLE_CONV_KERNEL,            PLE_KERNEL);
    ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,           uint32_t(1));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, PLE_HEAD_DIM);
    ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,      std::vector<uint64_t>({1000003ull, 999983ull, 1000033ull}));
    ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,           offs);
    ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,       sizes);
}

static int n_fail = 0;
static int n_ok   = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        printf("FAIL %s\n", what.c_str());
        n_fail++;
    } else {
        printf("ok   %s\n", what.c_str());
        n_ok++;
    }
}

template <typename T>
static bool eq(const std::vector<T> & a, const std::vector<T> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

static bool silent_progress(float, void *) { return true; }

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN && getenv("TEST_CELL_EXT_VERBOSE")) {
        fputs(text, stderr);
    }
}

static llama_model_params model_params() {
    static ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    mp.load_mtp          = true;
    mp.devices           = devs;
    return mp;
}

// the no-PLE model from user metadata; with ple, its tensors + the PLE layer go through a temporary GGUF
static llama_model_ptr make_model(bool ple, const std::string & tmp_path) {
    gguf_context_ptr meta(gguf_init_empty());
    {
        llama_model_saver ms(LLM_ARCH_QWEN4EXP, meta.get());
        add_meta(ms, 4096);
    }
    size_t seed = 31;
    llama_model_ptr base(llama_model_init_from_user(meta.get(), set_tensor_data, &seed, model_params()));
    GGML_ASSERT(base && "fixture model");
    if (!ple) {
        return base;
    }

    gguf_context_ptr out(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN4EXP, out.get());
    add_meta(ms, 4096);
    add_ple_meta(ms);
    for (const auto & [name, t] : base->tensors_by_name) {
        if (gguf_find_tensor(out.get(), ggml_get_name(t)) < 0) {
            ms.add_tensor(t);
        }
    }

    const uint32_t n_heads = (PLE_NGRAM - 1) * PLE_HPN;
    const int64_t  hc      = 4;
    ggml_init_params ip = { /*.mem_size =*/ 16u << 20, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context_ptr ctx(ggml_init(ip));
    const LLM_TN tn(LLM_ARCH_QWEN4EXP);
    auto add = [&](const std::string & name, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(t, name.c_str());
        std::mt19937 gen(std::hash<std::string>()(name));
        std::normal_distribution<float> dis(0.0f, 1.0e-1f);
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) { d[i] = dis(gen); }
        ms.add_tensor(t);
    };
    add(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str(), PLE_HEAD_DIM, n_heads * PLE_HEAD_VOCAB);
    add(tn(LLM_TENSOR_PLE_KEY,        "weight", 0).str(), N_EMBD, hc * N_EMBD);
    add(tn(LLM_TENSOR_PLE_VALUE,      "weight", 0).str(), N_EMBD, N_EMBD);
    add(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", 0).str(), N_EMBD, hc);
    add(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", 0).str(), N_EMBD, hc);
    add(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", 0).str(), N_EMBD, hc);
    add(tn(LLM_TENSOR_PLE_CONV1D,     "weight", 0).str(), PLE_KERNEL, hc * N_EMBD);
    ms.save(tmp_path);

    llama_model_ptr m(llama_model_load_from_file(tmp_path.c_str(), model_params()));
    GGML_ASSERT(m && "PLE fixture model");
    return m;
}

static llama_context_ptr make_ctx(llama_model * model, bool mtp) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = N_CTX;
    cp.n_batch         = 128;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = 1;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    if (mtp) {
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_rs_seq = 0;
    }
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    GGML_ASSERT(ctx && "context");
    return ctx;
}

static std::vector<llama_token> tokens(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<llama_token> dis(2, N_VOCAB - 1);
    std::vector<llama_token> t(n);
    for (auto & x : t) { x = dis(gen); }
    return t;
}

// the logits of every row of a decode
static std::vector<float> all_logits(llama_context * ctx, int32_t n) {
    std::vector<float> l((size_t) n * N_VOCAB);
    for (int32_t i = 0; i < n; ++i) {
        memcpy(l.data() + (size_t) i * N_VOCAB, llama_get_logits_ith(ctx, i), N_VOCAB * sizeof(float));
    }
    return l;
}

static std::vector<float> decode_text(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], pos0 + (llama_pos) i, { 0 }, true);
    }
    GGML_ASSERT(llama_decode(ctx, b) == 0);
    auto l = all_logits(ctx, b.n_tokens);
    llama_batch_free(b);
    return l;
}

// an h x w "image" at temporal position p0, as mtmd lays out the M-RoPE positions: section 0 = p0, 1 = y, 2 = x;
// returns the logits, the next text position is p0 + max(h, w)
static std::vector<float> decode_image(llama_context * ctx, int h, int w, llama_pos p0, uint32_t seed) {
    const int n = h * w;
    llama_batch b = llama_batch_init(n, N_EMBD, 1);
    // llama_batch_init sizes pos for one section; M-RoPE embeddings take 4
    free(b.pos);
    b.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * n);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    for (int i = 0; i < n; ++i) {
        for (uint32_t k = 0; k < N_EMBD; ++k) { b.embd[(size_t) i * N_EMBD + k] = dis(gen); }
        b.pos[i]         = p0;
        b.pos[i + n]     = p0 + i / w;
        b.pos[i + 2 * n] = p0 + i % w;
        b.pos[i + 3 * n] = 0;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = true;
    }
    b.n_tokens = n;
    GGML_ASSERT(llama_decode(ctx, b) == 0);
    auto l = all_logits(ctx, n);
    llama_batch_free(b);
    return l;
}

// {cell, pos, x, y, tok} of every occupied cell, in cell order
static std::vector<int32_t> cells_of(const llama_kv_cache * kv) {
    std::vector<int32_t> r;
    const auto & cells = kv->get_cells(0);
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_empty(i)) {
            continue;
        }
        const auto & e = cells.ext_get(i);
        r.insert(r.end(), { (int32_t) i, cells.pos_get(i), e.x, e.y, e.tok });
    }
    return r;
}

static llama_memory_hybrid_idx * hyb(llama_context * ctx) {
    return dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx));
}

static int n_xy_nontrivial(const std::vector<int32_t> & c) {
    int n = 0;
    for (size_t i = 0; i + 4 < c.size(); i += 5) {
        n += c[i + 2] != c[i + 1] || c[i + 3] != c[i + 1];
    }
    return n;
}

static int n_differing(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    if (a.size() != b.size()) {
        return -1;
    }
    int n = 0;
    for (size_t i = 0; i + 4 < a.size(); i += 5) {
        n += memcmp(&a[i], &b[i], 5 * sizeof(int32_t)) != 0;
    }
    return n;
}

static std::vector<uint8_t> read_file(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static size_t n_bytes_differing(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    if (a.size() != b.size()) {
        return (size_t) -1;
    }
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        n += a[i] != b[i];
    }
    return n;
}

// T: target save file -> fresh context restore
static void test_target(llama_model * model, const std::string & tag, const std::string & dir) {
    const auto t0 = tokens(40, 11);
    const auto t1 = tokens(9,  12);
    const auto t2 = tokens(5,  13);
    const auto t3 = tokens(4,  14);

    auto live = make_ctx(model, false);
    decode_text(live.get(), t0, 0);
    decode_image(live.get(), 3, 4, 40, 21);
    decode_text(live.get(), t1, 44);
    llama_pos n_past = 53;

    std::vector<llama_token> prompt = t0;
    prompt.insert(prompt.end(), 12, 0); // the image's 12 positions in the token list (as the server keeps them)
    prompt.insert(prompt.end(), t1.begin(), t1.end());

    const std::string f_live = dir + "/live-" + tag + ".bin";
    const std::string f_rest = dir + "/rest-" + tag + ".bin";
    GGML_ASSERT(llama_state_seq_save_file(live.get(), f_live.c_str(), 0, prompt.data(), prompt.size()) > 0);

    auto * h_live = hyb(live.get());
    GGML_ASSERT(h_live && h_live->get_mem_idx());
    const auto attn_live = cells_of(h_live->get_mem_attn());
    const auto idx_live  = cells_of(h_live->get_mem_idx());
    check(n_xy_nontrivial(idx_live) == 11, tag + " T: the live indexer cells hold the image's 2D positions (x/y != pos on 11 of its 12 cells)");

    // the in-memory continuation
    std::vector<float> cont_mem = decode_text(live.get(), t2, n_past);
    {
        auto l = decode_image(live.get(), 2, 2, n_past + 5, 22);
        cont_mem.insert(cont_mem.end(), l.begin(), l.end());
        l = decode_text(live.get(), t3, n_past + 7);
        cont_mem.insert(cont_mem.end(), l.begin(), l.end());
    }

    auto rest = make_ctx(model, false);
    std::vector<llama_token> toks(prompt.size() + 8);
    size_t n_tok = 0;
    GGML_ASSERT(llama_state_seq_load_file(rest.get(), f_live.c_str(), 0, toks.data(), toks.size(), &n_tok) > 0);
    GGML_ASSERT(n_tok == prompt.size());

    auto * h_rest = hyb(rest.get());
    const auto attn_rest = cells_of(h_rest->get_mem_attn());
    const auto idx_rest  = cells_of(h_rest->get_mem_idx());
    const int d_attn = n_differing(attn_live, attn_rest);
    const int d_idx  = n_differing(idx_live,  idx_rest);
    printf("info %s T: cells differing after restore: attention %d, indexer %d (of %zu)\n", tag.c_str(), d_attn, d_idx, idx_live.size() / 5);
    check(d_attn == 0, tag + " T: attention cells {pos, x, y, tok} restored byte-equal");
    check(d_idx  == 0, tag + " T: indexer cells {pos, x, y, tok} restored byte-equal");

    GGML_ASSERT(llama_state_seq_save_file(rest.get(), f_rest.c_str(), 0, prompt.data(), prompt.size()) > 0);
    const size_t d_file = n_bytes_differing(read_file(f_live), read_file(f_rest));
    printf("info %s T: restore gap (live save vs restored re-save): %zu differing bytes\n", tag.c_str(), d_file);
    check(d_file == 0, tag + " T: restored re-save == live save (0 differing bytes)");

    std::vector<float> cont_rest = decode_text(rest.get(), t2, n_past);
    {
        auto l = decode_image(rest.get(), 2, 2, n_past + 5, 22);
        cont_rest.insert(cont_rest.end(), l.begin(), l.end());
        l = decode_text(rest.get(), t3, n_past + 7);
        cont_rest.insert(cont_rest.end(), l.begin(), l.end());
    }
    check(eq(cont_mem, cont_rest), tag + " T: continuation logits (text + image + text, every row) == in-memory");
    check(eq(cells_of(h_live->get_mem_idx()), cells_of(h_rest->get_mem_idx())), tag + " T: indexer cells equal after the continuation");

    if (dir.rfind("/tmp", 0) == 0 && getenv("TEST_CELL_EXT_DUMP") == nullptr) {
        unlink(f_live.c_str());
        unlink(f_rest.c_str());
    }
}

static std::vector<uint8_t> seq_data(llama_context * ctx, llama_state_seq_flags flags) {
    std::vector<uint8_t> d(llama_state_seq_get_size_ext(ctx, 0, flags));
    GGML_ASSERT(llama_state_seq_get_data_ext(ctx, d.data(), d.size(), 0, flags) == d.size());
    return d;
}

// D: the MTP draft memory, fed by the speculative catch-up
static void test_draft(llama_model * model, const std::string & tag) {
    auto tgt = make_ctx(model, false);
    auto dft = make_ctx(model, true);
    auto * h = hyb(dft.get());
    if (!h || !h->get_mem_idx()) {
        printf("info %s D: the draft memory has no indexer (LLAMA_MTP_QSA unset) - skipped\n", tag.c_str());
        return;
    }

    common_params_speculative params;
    params.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.draft.n_max   = 4;
    params.draft.ctx_tgt = tgt.get();
    params.draft.ctx_dft = dft.get();
    common_speculative * spec = common_speculative_init(params, 1);
    GGML_ASSERT(spec);

    const auto t0 = tokens(96, 31);
    llama_batch b = llama_batch_init((int32_t) t0.size(), 0, 1);
    for (size_t i = 0; i < t0.size(); ++i) {
        common_batch_add(b, t0[i], (llama_pos) i, { 0 }, true);
    }
    GGML_ASSERT(llama_decode(tgt.get(), b) == 0);
    GGML_ASSERT(common_speculative_process(spec, b));
    llama_batch_free(b);

    const auto attn_live = cells_of(h->get_mem_attn());
    const auto idx_live  = cells_of(h->get_mem_idx());
    check(!idx_live.empty(), tag + " D: the draft indexer holds cells");

    // disk / RAM prompt cache form: flags 0 into a fresh draft context
    {
        const auto data = seq_data(dft.get(), 0);
        auto fresh = make_ctx(model, true);
        GGML_ASSERT(llama_state_seq_set_data_ext(fresh.get(), data.data(), data.size(), 0, 0) == data.size());
        auto * hf = hyb(fresh.get());
        const int d_attn = n_differing(attn_live, cells_of(hf->get_mem_attn()));
        const int d_idx  = n_differing(idx_live,  cells_of(hf->get_mem_idx()));
        printf("info %s D: flags 0 restore, cells differing: attention %d, indexer %d (of %zu)\n", tag.c_str(), d_attn, d_idx, idx_live.size() / 5);
        check(d_attn == 0 && d_idx == 0, tag + " D: flags 0 restore - draft attention + indexer cells byte-equal");
    }

    // speculative rollback checkpoint form: PARTIAL_ONLY, the same context after dropping the sequence
    {
        const auto data = seq_data(dft.get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        llama_memory_seq_rm(llama_get_memory(dft.get()), 0, -1, -1);
        GGML_ASSERT(llama_state_seq_set_data_ext(dft.get(), data.data(), data.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == data.size());
        const int d_attn = n_differing(attn_live, cells_of(h->get_mem_attn()));
        const int d_idx  = n_differing(idx_live,  cells_of(h->get_mem_idx()));
        printf("info %s D: PARTIAL_ONLY restore, cells differing: attention %d, indexer %d\n", tag.c_str(), d_attn, d_idx);
        check(d_attn == 0 && d_idx == 0, tag + " D: PARTIAL_ONLY (rollback checkpoint) restore - draft cells byte-equal");
    }

    common_speculative_free(spec);
}

int main(int argc, char ** argv) {
    llama_log_set(log_cb, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    std::string dir = argc > 1 ? argv[1] : "/tmp";
    const std::string gguf_path = dir + "/test-cell-ext-restore-ple-" + std::to_string(getpid()) + ".gguf";

    for (const bool ple : { false, true }) {
        const std::string tag = ple ? "ple" : "plain";
        llama_model_ptr model = make_model(ple, gguf_path);
        if (ple) {
            check(model->hparams.ple_n_heads > 0, "ple: the fixture has a PLE layer");
            unlink(gguf_path.c_str());
        }
        test_target(model.get(), tag, dir);
        test_draft (model.get(), tag);
    }

    llama_backend_free();
    printf("%s: %d ok, %d failed\n", n_fail ? "FAIL" : "PASS", n_ok, n_fail);
    return n_fail ? 1 : 0;
}
