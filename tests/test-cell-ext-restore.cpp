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
//  T-old: the image save in the shape a pre-fix build wrote (no "IXY1" provenance mark before the indexer section,
//     with PLE the indexer x / y zeroed) is rejected: the load fails, the rejection counter grows, nothing is left;
//  TX: a text-only save in the pre-fix shape is accepted and healed: indexer cells == live, re-save == the live
//     (marked) save, continuation logits == in-memory;
//  D  MTP draft memory (LLAMA_MTP_QSA=1 only, otherwise it is a plain KV cache): text + an image (the MTP catch-up
//     skips embedding batches, so the draft holds no image cell); seq data with flags 0 into a fresh context, and the
//     speculative rollback checkpoint form (PARTIAL_ONLY, same context after seq_rm): the attention and indexer cells'
//     {pos, ext} equal the live ones, and the next draft step's logits equal those before the rollback;
//  CK: a target context checkpoint (PARTIAL_ONLY, recurrent only) opens with the decoy header {1, 0, IXY1} that the v59
//     recurrent reader (unchanged since) refuses at the first cell record, and restores exact; the pre-fix
//     shape (no mark) is refused when the live sequence holds an image (checkpoint counter +1, sequence dropped from
//     all three caches), loads and replays exact for a text-only sequence;
//  T-rm / T-hole: an unmarked trunk state must be an intact history (cells at positions 0..n-1): an image conversation
//     with its image prefix removed by seq_rm before the save, and a text one with a hole, are refused; marked they load;
//     the checkpoint triplet must be exact ({1, 5, IXY1} is refused as an invalid stream);
//  provenance: an unmarked state loads only while the caller vouches for it (llama_state_legacy_unmarked_accept, the
//     server's proof from its stored token stream); the cell checks are extra guards. TX / CK-text / D-old also run
//     without the proof (refused); CP: the seq_cp(0, 1, 0, 40) destination of an image conversation is refused;
//  D-old: a pre-fix draft stream (recr_none branch, no mark) loads with cells == live;
//  cost: the added per-restore copy over 1 048 576 cells (info line).
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
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-io.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
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
// spec: also run the MTP catch-up over the batch, as the server's mtmd decode callback does
static std::vector<float> decode_image(llama_context * ctx, int h, int w, llama_pos p0, uint32_t seed,
                                       common_speculative * spec = nullptr) {
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
    if (spec) {
        GGML_ASSERT(common_speculative_process(spec, b));
    }
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

// [TAG_CELL_EXT_PROVENANCE] a save file rewritten into the shape a pre-fix build wrote: the "IXY1" mark in front of the
// indexer section removed and, when the indexer serializes the cell ext, its x / y zeroed (what a pre-fix restore left)
static bool prefix_shape(std::vector<uint8_t> & b, uint32_t n_cells, bool idx_ext) {
    size_t o = std::string::npos;
    for (size_t p = 0; p + 12 <= b.size(); ++p) {
        uint32_t w[3];
        memcpy(w, b.data() + p, sizeof(w));
        if (w[0] == 0x31595849u && w[1] == 1 && w[2] == n_cells) {
            o = p; // the last match: the indexer section is the state's last
        }
    }
    if (o == std::string::npos) {
        return false;
    }
    b.erase(b.begin() + o, b.begin() + o + 4);
    if (idx_ext) {
        for (uint32_t c = 0; c < n_cells; ++c) {
            memset(b.data() + o + 8 + (size_t) c * 24 + 8, 0, 8); // record {pos, n_seq_id, x, y, tok, seq_id}: x, y
        }
    }
    return true;
}

static bool write_prefix_shape(const std::string & src, const std::string & dst, uint32_t n_cells, bool idx_ext) {
    auto b = read_file(src);
    if (!prefix_shape(b, n_cells, idx_ext)) {
        return false;
    }
    std::ofstream f(dst, std::ios::binary);
    f.write((const char *) b.data(), b.size());
    return true;
}

static std::vector<uint8_t> seq_data(llama_context * ctx, llama_state_seq_flags flags);

// a plain in-memory llama_io_read_i (the library's own readers are not exported)
struct buf_reader : llama_io_read_i {
    buf_reader(const uint8_t * p, size_t n) : p(p), n(n) {}
    void read(void * dst, size_t size) override {
        if (size > n - off) {
            throw std::runtime_error("buf_reader: past the end");
        }
        memcpy(dst, p + off, size);
        off += size;
    }
    void read_tensor(ggml_tensor * t, size_t offset, size_t size) override {
        std::vector<uint8_t> tmp(size);
        read(tmp.data(), size);
        ggml_backend_tensor_set(t, tmp.data(), offset, size);
    }
    size_t n_bytes() override { return off; }
    const uint8_t * p;
    size_t n, off = 0;
};

// all three caches of the hybrid memory hold nothing of seq 0
static bool seq_gone(llama_context * ctx) {
    auto * h = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx));
    return h->get_mem_attn()->seq_pos_max(0) == -1 && h->get_mem_recr()->seq_pos_max(0) == -1 &&
           h->get_mem_idx()->seq_pos_max(0) == -1;
}

// T-rm / T-hole: an unmarked (pre-fix) trunk state is accepted only as an intact history: an image conversation whose
// image prefix was removed by seq_rm(seq, 0, k) before the save (no image cell left, but its effect in the later rows),
// and a text conversation with a hole, are refused (counter +1, nothing left); the same states MARKED load
static void test_history(llama_model * model, const std::string & tag, const std::string & dir, bool hole) {
    const std::string tc = tag + (hole ? " T-hole" : " T-rm");
    auto live = make_ctx(model, false);
    std::vector<llama_token> prompt;
    auto * h = hyb(live.get());
    if (hole) {
        const auto t = tokens(60, 71);
        decode_text(live.get(), t, 0);
        prompt = t;
        GGML_ASSERT(h->get_mem_attn()->seq_rm(0, 10, 20) && h->get_mem_idx()->seq_rm(0, 10, 20));
    } else {
        const auto t0 = tokens(40, 72);
        const auto t1 = tokens(9,  73);
        decode_text(live.get(), t0, 0);
        decode_image(live.get(), 3, 4, 40, 74);
        decode_text(live.get(), t1, 44);
        prompt = t0;
        prompt.insert(prompt.end(), 12, 0);
        prompt.insert(prompt.end(), t1.begin(), t1.end());
        // the image prefix removed from the KV caches (the recurrent rows keep its effect)
        GGML_ASSERT(h->get_mem_attn()->seq_rm(0, 0, 44) && h->get_mem_idx()->seq_rm(0, 0, 44));
    }
    const auto attn_left = cells_of(h->get_mem_attn());
    printf("info %s: %zu cells left, %d with x/y != pos\n", tc.c_str(), attn_left.size() / 5, n_xy_nontrivial(attn_left));

    const std::string f_live = dir + "/hist-live-" + tag + ".bin";
    const std::string f_old  = dir + "/hist-old-"  + tag + ".bin";
    GGML_ASSERT(llama_state_seq_save_file(live.get(), f_live.c_str(), 0, prompt.data(), prompt.size()) > 0);
    std::vector<llama_token> toks(prompt.size() + 8);
    size_t n_tok = 0;
    {
        auto m = make_ctx(model, false);
        check(llama_state_seq_load_file(m.get(), f_live.c_str(), 0, toks.data(), toks.size(), &n_tok) > 0,
                tc + ": the same state MARKED (written by this build) loads");
    }
    check(write_prefix_shape(f_live, f_old, (uint32_t) (attn_left.size() / 5), h->get_mem_idx()->has_cell_ext()),
            tc + ": provenance mark located and stripped (pre-fix shape)");
    auto old = make_ctx(model, false);
    llama_state_legacy_unmarked_accept(old.get(), true); // even if vouched for: the intact-history guard refuses it
    const uint64_t r0 = llama_memory_hybrid_idx::n_xy_rejects();
    const size_t nr = llama_state_seq_load_file(old.get(), f_old.c_str(), 0, toks.data(), toks.size(), &n_tok);
    printf("info %s: pre-fix shape %s, rejections %llu -> %llu\n", tc.c_str(), nr ? "accepted" : "refused",
            (unsigned long long) r0, (unsigned long long) llama_memory_hybrid_idx::n_xy_rejects());
    check(nr == 0 && llama_memory_hybrid_idx::n_xy_rejects() == r0 + 1, tc + ": the pre-fix shape is refused (not an intact history, counter +1)");
    check(seq_gone(old.get()), tc + ": nothing of it is left in attention, recurrent or indexer");
    unlink(f_live.c_str());
    unlink(f_old.c_str());
}

// CP: astra's counterexample to proving history from cells: a unified pool, seq 0 = text + image + text, then a partial
// seq_cp(0, 1, 0, 40) - attention / indexer copy cells 0..39 (text only, contiguous) while the recurrent copy takes seq
// 0's whole state (the image included). An unmarked save of seq 1 passes both cell guards; without the caller's proof
// (a memory-API path never has one) it is refused
static void test_seq_cp(llama_model * model, const std::string & tag, const std::string & dir) {
    const std::string tc = tag + " CP";
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = N_CTX; cp.n_batch = 128; cp.n_ubatch = 64; cp.n_seq_max = 2; cp.kv_unified = true;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    GGML_ASSERT(ctx);
    const auto t0 = tokens(40, 81);
    const auto t1 = tokens(9,  82);
    decode_text(ctx.get(), t0, 0);
    decode_image(ctx.get(), 3, 4, 40, 83);
    decode_text(ctx.get(), t1, 44);
    llama_memory_seq_cp(llama_get_memory(ctx.get()), 0, 1, 0, 40);

    auto * h = hyb(ctx.get());
    std::vector<int32_t> c1;
    {
        const auto & cells = h->get_mem_attn()->get_cells(1);
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && cells.seq_has(i, 1)) {
                c1.push_back(cells.pos_get(i));
            }
        }
    }
    printf("info %s: seq 1 holds %zu attention cells, recurrent pos_max %d, intact %d, image cells %u\n", tc.c_str(), c1.size(),
            (int) h->get_mem_recr()->seq_pos_max(1), (int) h->get_mem_attn()->seq_history_intact(1, (llama_pos) INT32_MAX),
            h->get_mem_attn()->n_cells_2d_seq(1, (llama_pos) INT32_MAX));

    const std::string f_live = dir + "/cp-live-" + tag + ".bin";
    const std::string f_old  = dir + "/cp-old-"  + tag + ".bin";
    std::vector<llama_token> prompt(t0.begin(), t0.end());
    GGML_ASSERT(llama_state_seq_save_file(ctx.get(), f_live.c_str(), 1, prompt.data(), prompt.size()) > 0);
    check(write_prefix_shape(f_live, f_old, (uint32_t) c1.size(), h->get_mem_idx()->has_cell_ext()),
            tc + ": provenance mark located and stripped (pre-fix shape)");
    auto old = make_ctx(model, false);
    std::vector<llama_token> toks(prompt.size() + 8);
    size_t n_tok = 0;
    const uint64_t r0 = llama_memory_hybrid_idx::n_xy_rejects();
    const size_t nr = llama_state_seq_load_file(old.get(), f_old.c_str(), 0, toks.data(), toks.size(), &n_tok);
    printf("info %s: unmarked seq_cp destination %s\n", tc.c_str(), nr ? "accepted" : "refused");
    check(nr == 0 && llama_memory_hybrid_idx::n_xy_rejects() == r0 + 1 && seq_gone(old.get()),
            tc + ": the unmarked seq_cp(0, 40) destination (cells intact + text-only) is refused without the proof");
    unlink(f_live.c_str());
    unlink(f_old.c_str());
}

// CK: a target context checkpoint (PARTIAL_ONLY: the recurrent rows only) is marked; restored exact (marked); an
// unmarked (pre-fix) one is refused when the live sequence holds an image (counter +1, the sequence dropped - the server
// then re-processes the prompt), and loads as before for a text-only sequence
static void test_target_ckpt(llama_model * model, const std::string & tag, bool image) {
    const std::string tc = tag + (image ? " CK-image" : " CK-text");
    const auto t0 = tokens(40, 61);
    const auto t1 = tokens(9,  62);
    const auto t2 = tokens(5,  63);

    auto live = make_ctx(model, false);
    decode_text(live.get(), t0, 0);
    llama_pos n_past = 40;
    if (image) {
        decode_image(live.get(), 3, 4, 40, 64);
        n_past = 44;
    }
    decode_text(live.get(), t1, n_past);
    n_past += (llama_pos) t1.size();

    const auto ck = seq_data(live.get(), LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    uint32_t w[3] = {};
    if (ck.size() > 20) {
        memcpy(w, ck.data() + 8, sizeof(w));
    }
    check(w[0] == 1 && w[1] == 0 && w[2] == LLAMA_STATE_IXY_MARK, tc + ": the checkpoint stream opens with the decoy header {1, 0, IXY1}");

    const auto cont = decode_text(live.get(), t2, n_past);
    llama_memory_t mem = llama_get_memory(live.get());

    // marked: back to the checkpoint, the replay is exact
    GGML_ASSERT(llama_state_seq_set_data_ext(live.get(), ck.data(), ck.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == ck.size());
    GGML_ASSERT(llama_memory_seq_rm(mem, 0, n_past, -1));
    check(eq(decode_text(live.get(), t2, n_past), cont), tc + ": marked checkpoint restore + replay == the first run");

    // text: an image AFTER the checkpoint's prefix in the live sequence (the server trims it only after the load) does not
    // make a text-only pre-fix checkpoint refused
    if (!image) {
        decode_image(live.get(), 2, 2, n_past + (llama_pos) t2.size(), 65);
    }

    // unmarked (the shape a pre-fix build stored)
    std::vector<uint8_t> old = ck;
    old.erase(old.begin() + 8, old.begin() + 20);
    const uint64_t r0 = llama_memory_hybrid_idx::n_xy_ckpt_rejects();
    llama_state_legacy_unmarked_accept(live.get(), true); // the server's proof (entry with a media-free prefix)
    const size_t nr = llama_state_seq_set_data_ext(live.get(), old.data(), old.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    printf("info %s: unmarked checkpoint %s, checkpoint rejections %llu -> %llu\n", tc.c_str(), nr ? "accepted" : "refused",
            (unsigned long long) r0, (unsigned long long) llama_memory_hybrid_idx::n_xy_ckpt_rejects());
    if (image) {
        check(nr == 0 && llama_memory_hybrid_idx::n_xy_ckpt_rejects() == r0 + 1, tc + ": an unmarked checkpoint of an image sequence is refused (counter +1)");
        check(seq_gone(live.get()), tc + ": the sequence is dropped from attention, recurrent and indexer (the server re-processes)");
    } else {
        check(nr == old.size() && llama_memory_hybrid_idx::n_xy_ckpt_rejects() == r0, tc + ": an unmarked checkpoint of a text-only prefix loads (an image after it in the live sequence)");
        GGML_ASSERT(llama_memory_seq_rm(mem, 0, n_past, -1));
        check(eq(decode_text(live.get(), t2, n_past), cont), tc + ": its replay == the first run");

        // only the exact decoy triplet {1, 0, IXY1} is the mark: {1, 5, IXY1} reads as a legacy record whose n_seq_id is
        // not 0 - refused by the recurrent reader, no provenance rejection counted
        std::vector<uint8_t> bad = ck;
        const uint32_t five = 5;
        memcpy(bad.data() + 12, &five, sizeof(five));
        const uint64_t r1 = llama_memory_hybrid_idx::n_xy_ckpt_rejects();
        const size_t nb = llama_state_seq_set_data_ext(live.get(), bad.data(), bad.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        check(nb == 0 && llama_memory_hybrid_idx::n_xy_ckpt_rejects() == r1, tc + ": {1, 5, IXY1} is not the mark (refused as an invalid stream)");
    }
    llama_state_legacy_unmarked_accept(live.get(), false);
    if (!image) {
        // without the proof the same text checkpoint is refused (counter +1)
        const uint64_t r2 = llama_memory_hybrid_idx::n_xy_ckpt_rejects();
        const size_t n2 = llama_state_seq_set_data_ext(live.get(), old.data(), old.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        check(n2 == 0 && llama_memory_hybrid_idx::n_xy_ckpt_rejects() == r2 + 1, tc + ": the unmarked text checkpoint WITHOUT the proof is refused (counter +1)");
    }

    // the OLD reader: v59 (a1b6c324e) hands a PARTIAL_ONLY trunk stream straight to llama_memory_recurrent::state_read
    // (its llama-memory-hybrid-idx.cpp:933), and that reader and llama_batch_allocr::ubatch_reserve are unchanged since
    // - so the same call here is v59's path: the decoy header must be refused at the first cell record, cheaply
    {
        buf_reader io(ck.data() + 8, ck.size() - 8);
        bool threw = false;
        const auto t_start = std::chrono::steady_clock::now();
        try {
            hyb(live.get())->get_mem_recr()->state_read(io, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        } catch (const std::exception &) {
            threw = true;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        printf("info %s: old (v59) recurrent reader on this checkpoint: %s after %zu bytes, status %d, %.3f ms\n", tc.c_str(),
                threw ? "refused" : "ACCEPTED", io.n_bytes(), io.status, ms);
        check(threw && io.status == LLAMA_STATE_READ_INVALID && io.n_bytes() <= 12,
                tc + ": the v59 recurrent reader refuses the checkpoint at its first cell record (INVALID, <= 12 bytes read)");
    }
}

// TX: a text-only entry written by a pre-fix build stays accepted and is healed
static void test_prefix_text(llama_model * model, const std::string & tag, const std::string & dir) {
    const auto t0 = tokens(70, 51);
    const auto t1 = tokens(6,  52);

    auto live = make_ctx(model, false);
    decode_text(live.get(), t0, 0);
    const std::string f_live = dir + "/tx-live-" + tag + ".bin";
    const std::string f_old  = dir + "/tx-old-"  + tag + ".bin";
    const std::string f_re   = dir + "/tx-re-"   + tag + ".bin";
    GGML_ASSERT(llama_state_seq_save_file(live.get(), f_live.c_str(), 0, t0.data(), t0.size()) > 0);
    auto * h_live = hyb(live.get());
    const auto idx_live = cells_of(h_live->get_mem_idx());
    const auto cont_mem = decode_text(live.get(), t1, 70);

    check(write_prefix_shape(f_live, f_old, (uint32_t) (idx_live.size() / 5), h_live->get_mem_idx()->has_cell_ext()),
            tag + " TX: provenance mark located and stripped (pre-fix shape)");
    std::vector<llama_token> toks(t0.size() + 8);
    size_t n_tok = 0;
    {
        // without the caller's proof (llama_state_legacy_unmarked_accept) an unmarked state is refused
        auto old0 = make_ctx(model, false);
        const uint64_t r = llama_memory_hybrid_idx::n_xy_rejects();
        const size_t n0 = llama_state_seq_load_file(old0.get(), f_old.c_str(), 0, toks.data(), toks.size(), &n_tok);
        check(n0 == 0 && llama_memory_hybrid_idx::n_xy_rejects() == r + 1 && seq_gone(old0.get()),
                tag + " TX: a pre-fix text entry WITHOUT the caller's proof is refused (counter +1)");
    }
    {
        // the server's scope clears the flag on every exit: an exception thrown inside the load (e.g. a bad_alloc in a
        // state reader) leaves both contexts unvouched, and a later restore of the unmarked state is refused again
        auto c1 = make_ctx(model, false);
        auto c2 = make_ctx(model, true);
        bool inside = false;
        try {
            llama_state_legacy_accept_scope scope(c1.get(), c2.get(), true);
            inside = llama_state_legacy_unmarked_accepted(c1.get());
            throw std::bad_alloc();
        } catch (const std::bad_alloc &) {
        }
        check(inside && !llama_state_legacy_unmarked_accepted(c1.get()) && !llama_state_legacy_unmarked_accepted(c2.get()),
                tag + " TX: an exception inside the accept scope clears the flag on both contexts");
        const uint64_t r = llama_memory_hybrid_idx::n_xy_rejects();
        check(llama_state_seq_load_file(c1.get(), f_old.c_str(), 0, toks.data(), toks.size(), &n_tok) == 0 &&
              llama_memory_hybrid_idx::n_xy_rejects() == r + 1, tag + " TX: after that exception the unmarked state is refused again");
    }
    auto old = make_ctx(model, false);
    const uint64_t r0 = llama_memory_hybrid_idx::n_xy_rejects();
    size_t nr = 0;
    {
        llama_state_legacy_accept_scope scope(old.get(), nullptr, true); // the server's proof: its stored token stream has no media
        nr = llama_state_seq_load_file(old.get(), f_old.c_str(), 0, toks.data(), toks.size(), &n_tok);
    }
    check(nr > 0 && llama_memory_hybrid_idx::n_xy_rejects() == r0, tag + " TX: a pre-fix text entry WITH the proof is accepted (no rejection)");
    if (nr > 0) {
        const int d = n_differing(idx_live, cells_of(hyb(old.get())->get_mem_idx()));
        printf("info %s TX: pre-fix text entry, indexer cells differing after restore: %d\n", tag.c_str(), d);
        check(d == 0, tag + " TX: healed - indexer cells == live");
        GGML_ASSERT(llama_state_seq_save_file(old.get(), f_re.c_str(), 0, t0.data(), t0.size()) > 0);
        check(n_bytes_differing(read_file(f_live), read_file(f_re)) == 0, tag + " TX: its re-save == the live (marked) save");
        check(eq(decode_text(old.get(), t1, 70), cont_mem), tag + " TX: continuation logits == in-memory");
    }
    unlink(f_live.c_str());
    unlink(f_old.c_str());
    unlink(f_re.c_str());
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

    // [TAG_CELL_EXT_PROVENANCE] the same image entry as a pre-fix build wrote it (no provenance mark; with PLE the
    // indexer x = y = 0 a restore left behind): its K/V may carry inference on zeroed x/y - rejected, a cache miss
    {
        const std::string f_old = dir + "/old-" + tag + ".bin";
        check(write_prefix_shape(f_live, f_old, (uint32_t) (idx_live.size() / 5), h_live->get_mem_idx()->has_cell_ext()),
                tag + " T-old: provenance mark located and stripped (pre-fix shape)");
        auto old = make_ctx(model, false);
        llama_state_legacy_unmarked_accept(old.get(), true); // even if vouched for: the image-cell guard refuses it
        const uint64_t r0 = llama_memory_hybrid_idx::n_xy_rejects();
        n_tok = 0;
        const size_t nr = llama_state_seq_load_file(old.get(), f_old.c_str(), 0, toks.data(), toks.size(), &n_tok);
        printf("info %s T-old: pre-fix image entry: load %s, rejections %llu -> %llu\n", tag.c_str(), nr ? "accepted" : "refused",
                (unsigned long long) r0, (unsigned long long) llama_memory_hybrid_idx::n_xy_rejects());
        check(nr == 0 && llama_memory_hybrid_idx::n_xy_rejects() == r0 + 1, tag + " T-old: a pre-fix image entry is rejected (load fails, counter +1)");
        check(seq_gone(old.get()), tag + " T-old: nothing of it is left in attention, recurrent or indexer");
        unlink(f_old.c_str());
    }

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
    decode_image(tgt.get(), 3, 4, 96, 41, spec); // an image (mtmd decode + the server's MTP catch-up call), next text at 100
    const llama_pos n_past = 100;

    const auto attn_live = cells_of(h->get_mem_attn());
    const auto idx_live  = cells_of(h->get_mem_idx());
    {
        const int n_nt = n_xy_nontrivial(idx_live), n_nt_attn = n_xy_nontrivial(attn_live);
        printf("info %s D: draft cells with x/y != pos: indexer %d, attention %d (of %zu)\n", tag.c_str(), n_nt, n_nt_attn, idx_live.size() / 5);
        if (getenv("TEST_CELL_EXT_VERBOSE")) {
            for (size_t i = 0; i + 4 < idx_live.size(); i += 5) {
                if (idx_live[i + 1] >= 90) printf("  cell %d pos %d x %d y %d tok %d\n", idx_live[i], idx_live[i + 1], idx_live[i + 2], idx_live[i + 3], idx_live[i + 4]);
            }
        }
        // the MTP catch-up skips embedding batches (common/speculative.cpp: "TODO: how to make it work with vision
        // tokens?"), so the draft memory never holds image cells - its rows are text (x = y = pos) + the token
        check(n_nt == 0 && n_nt_attn == 0, tag + " D: the draft memory holds no image cells (the MTP catch-up skips embedding batches)");
    }

    // one draft from (n_past, id_last) with the given carry; returns the draft logits, the drafted cells dropped again
    std::vector<uint8_t> carry;
    GGML_ASSERT(common_speculative_get_state(spec, 0, carry));
    auto draft = [&]() {
        GGML_ASSERT(common_speculative_set_state(spec, 0, carry));
        std::vector<llama_token> out;
        auto & dp = common_speculative_get_draft_params(spec, 0);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.n_past   = n_past;
        dp.id_last  = t0.back();
        dp.prompt   = nullptr;
        dp.result   = &out;
        common_speculative_draft(spec);
        const float * l = llama_get_logits_ith(dft.get(), -1);
        std::vector<float> r = l ? std::vector<float>(l, l + N_VOCAB) : std::vector<float>();
        llama_memory_seq_rm(llama_get_memory(dft.get()), 0, n_past, -1);
        return r;
    };
    const auto logits_live = draft();
    check(!logits_live.empty() && eq(cells_of(h->get_mem_idx()), idx_live), tag + " D: a draft step leaves the draft cells as they were");

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

    // D-old: the draft stream as a pre-fix build wrote it (no mark; with PLE indexer x / y zeroed) - the draft holds no
    // image cell, so it loads and heals
    {
        auto data = seq_data(dft.get(), 0);
        const bool ok = prefix_shape(data, (uint32_t) (idx_live.size() / 5), h->get_mem_idx()->has_cell_ext());
        check(ok, tag + " D-old: provenance mark located and stripped in the draft stream");
        {
            auto f0 = make_ctx(model, true);
            const uint64_t r = llama_memory_hybrid_idx::n_xy_rejects();
            check(llama_state_seq_set_data_ext(f0.get(), data.data(), data.size(), 0, 0) == 0 &&
                  llama_memory_hybrid_idx::n_xy_rejects() == r + 1, tag + " D-old: WITHOUT the proof the pre-fix draft stream is refused");
        }
        auto fresh = make_ctx(model, true);
        llama_state_legacy_unmarked_accept(fresh.get(), true);
        const uint64_t r0 = llama_memory_hybrid_idx::n_xy_rejects();
        const size_t nr = ok ? llama_state_seq_set_data_ext(fresh.get(), data.data(), data.size(), 0, 0) : 0;
        auto * hf = hyb(fresh.get());
        const int d_attn = n_differing(attn_live, cells_of(hf->get_mem_attn()));
        const int d_idx  = n_differing(idx_live,  cells_of(hf->get_mem_idx()));
        printf("info %s D-old: pre-fix draft stream %s, cells differing: attention %d, indexer %d\n", tag.c_str(),
                nr == data.size() ? "accepted" : "refused", d_attn, d_idx);
        check(nr == data.size() && llama_memory_hybrid_idx::n_xy_rejects() == r0 && d_attn == 0 && d_idx == 0,
                tag + " D-old: a pre-fix draft stream (recr_none branch) loads, cells == live");
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
        check(eq(draft(), logits_live), tag + " D: PARTIAL_ONLY restore - the next draft step's logits == before the rollback");
    }

    common_speculative_free(spec);
}

// the restore's added cost at the prod size: llama_kv_cache::ext_copy_from's per-cell body over 1 048 576 cells
static void bench_ext_copy() {
    const uint32_t n = 1u << 20;
    llama_kv_cells src, dst;
    src.resize(n);
    dst.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        src.pos_set(i, (llama_pos) i); src.seq_add(i, 0); src.ext_set(i, { (llama_pos) i, (llama_pos) i, (llama_token) (i % 977) });
        dst.pos_set(i, (llama_pos) i); dst.seq_add(i, 0);
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < n; ++i) {
        if (!src.is_empty(i) && !dst.is_empty(i)) {
            llama_kv_cell_ext e = src.ext_get(i);
            e.tok = dst.ext_get(i).tok;
            dst.ext_set(i, e);
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("info cost: indexer ext copy over %u cells: %.2f ms per restore\n", n, ms);
    check(dst.ext_get(n - 1).x == (llama_pos) (n - 1), "cost bench: the copy ran");
}

int main(int argc, char ** argv) {
    llama_log_set(log_cb, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    // a private scratch directory per run: the ctest variants (plain / -qsa) run in parallel and use the same file names
    std::string dir;
    bool own_dir = false;
    if (argc > 1) {
        dir = argv[1];
    } else {
        char tmpl[] = "/tmp/test-cell-ext-restore-XXXXXX";
        GGML_ASSERT(mkdtemp(tmpl) != nullptr);
        dir = tmpl;
        own_dir = true;
    }
    const std::string gguf_path = dir + "/test-cell-ext-restore-ple-" + std::to_string(getpid()) + ".gguf";

    for (const bool ple : { false, true }) {
        const std::string tag = ple ? "ple" : "plain";
        llama_model_ptr model = make_model(ple, gguf_path);
        if (ple) {
            check(model->hparams.ple_n_heads > 0, "ple: the fixture has a PLE layer");
            unlink(gguf_path.c_str());
        }
        test_target(model.get(), tag, dir);
        test_prefix_text(model.get(), tag, dir);
        test_target_ckpt(model.get(), tag, true);
        test_history(model.get(), tag, dir, false);
        test_history(model.get(), tag, dir, true);
        test_seq_cp(model.get(), tag, dir);
        test_target_ckpt(model.get(), tag, false);
        test_draft (model.get(), tag);
    }

    bench_ext_copy();

    if (own_dir && getenv("TEST_CELL_EXT_DUMP") == nullptr) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }

    llama_backend_free();
    printf("%s: %d ok, %d failed\n", n_fail ? "FAIL" : "PASS", n_ok, n_fail);
    return n_fail ? 1 : 0;
}
