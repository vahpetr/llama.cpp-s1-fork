// [TAG_DISK_INCR] incremental sequence-state capture (--cache-disk-incr).
//
// The KV caches report, per K / V range of a sequence-state stream, how many leading bytes belong to cells not modified
// since a llama_state_mod_clock_tick() value (llama_state_seq_writer::write_tensor_kv). A writer that takes those
// bytes - cut into whole blocks from the range start, as the server's 4 MiB blocks - from the previous capture of the
// same (tensor, offset) range and reads only the rest from the device must produce the stream of a full capture.
// On the CPU qwen4exp fixture of test-state-portable (hybrid-idx target: attention KV + recurrent + QSA indexer K; MTP
// draft: attention KV + indexer K), target and draft, np1 and np2 unified:
//  1. append (prefill continues): composite == full, blocks reused;
//  2. a rollback (seq_rm of the tail, other tokens decoded at those positions): composite == full, fewer reused;
//  3. the sequence reloaded from a state blob (state_read writes every cell): nothing reused, composite == full;
//  4. np2 unified: seq 1 decoding next to seq 0 does not dirty seq 0's rows (reuse continues), composite == full;
//  5. negative control: a writer that ignores the hint (reuses every recorded block) differs from the full capture
//     after the rollback - the test can fail;
//  6. llama_kv_cells stamps: pos_set / seq_rm / pos_add + reset_shift (the K-shift) / mod_mark_all / reset.

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
#include "../src/llama-kv-cells.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

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

// the test-mtp-catchup qwen4exp fixture (hc 4, QSA ratios 4, MoE, 1 MTP block)
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
    printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        n_fail++;
    } else {
        n_ok++;
    }
}

static bool silent_progress(float, void *) { return true; }

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN && getenv("TEST_STATE_INCR_VERBOSE")) {
        fputs(text, stderr);
    }
}

static llama_model_ptr g_model;

struct env {
    llama_context_ptr ctx_tgt;
    llama_context_ptr ctx_dft;
    common_params_speculative params;
    common_speculative * spec = nullptr;
    ~env() { common_speculative_free(spec); }
};

static std::unique_ptr<env> make_env(uint32_t np) {
    auto e = std::make_unique<env>();
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = N_CTX_SEQ;
    cp.n_batch         = 128;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = np;
    cp.kv_unified      = true;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.n_rs_seq        = 16; // the speculative rollback window of the recurrent state (rm_tail after a drafted batch)
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
    e->spec = common_speculative_init(e->params, np);
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

// target decode of seq (every row an output) + the MTP catch-up (the draft memory follows the target's)
static void feed(env & e, llama_seq_id seq, const std::vector<llama_token> & toks, llama_pos pos0, uint32_t rs_depth = 0) {
    if (toks.size() > 64) { // n_batch 128: in chunks
        for (size_t i = 0; i < toks.size(); i += 64) {
            const std::vector<llama_token> part(toks.begin() + i, toks.begin() + std::min(toks.size(), i + 64));
            feed(e, seq, part, pos0 + (llama_pos) i, 0);
        }
        return;
    }
    llama_set_rs_rollback_depth(e.ctx_tgt.get(), rs_depth);
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], pos0 + (llama_pos) i, { seq }, true);
    }
    GGML_ASSERT(llama_decode(e.ctx_tgt.get(), b) == 0);
    GGML_ASSERT(common_speculative_process(e.spec, b));
    llama_batch_free(b);
}

static void rm_tail(env & e, llama_seq_id seq, llama_pos p0) {
    GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(e.ctx_tgt.get()), seq, p0, -1));
    GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(e.ctx_dft.get()), seq, p0, -1));
}

using range_key = std::pair<const ggml_tensor *, size_t>;

// one context's capture record: (tensor, offset) -> the range's bytes at the capture
struct record {
    uint64_t since = 0;
    std::map<range_key, std::vector<uint8_t>> ranges;
};

struct capture_out {
    std::vector<uint8_t> stream;
    size_t n_reused = 0;   // bytes taken from the record
    size_t n_kv     = 0;   // KV range bytes in the stream
    size_t n_hinted = 0;   // KV ranges reported through write_tensor_kv
};

// BLK: the block size of the reuse cut (the server's is 4 MiB; the fixture's ranges are KiB)
static const size_t BLK = 1024;

// mode 0: full (plain writer); 1: incremental (reuse min(clean, recorded) whole blocks); 2: negative control (reuse every
//   recorded whole block, ignoring clean)
static capture_out capture(llama_context * ctx, llama_seq_id seq, const record * prev, record * next, int mode) {
    capture_out out;
    const uint64_t since_next = llama_state_mod_clock_tick();
    llama_state_seq_writer w;
    w.write = [&](const void * src, size_t size) {
        const auto * p = static_cast<const uint8_t *>(src);
        out.stream.insert(out.stream.end(), p, p + size);
    };
    w.write_tensor = [&](ggml_tensor * t, size_t off, size_t size) {
        const size_t at = out.stream.size();
        out.stream.resize(at + size);
        ggml_backend_tensor_get(t, out.stream.data() + at, off, size);
    };
    if (mode != 0) {
        w.since = prev ? prev->since : 0;
        w.write_tensor_kv = [&](ggml_tensor * t, size_t off, size_t size, size_t clean) {
            out.n_hinted++;
            out.n_kv += size;
            const size_t at = out.stream.size();
            out.stream.resize(at + size);
            size_t reuse = 0;
            if (prev) {
                const auto it = prev->ranges.find({ t, off });
                if (it != prev->ranges.end()) {
                    const size_t lim = std::min({ mode == 2 ? size : clean, it->second.size(), size });
                    reuse = lim / BLK * BLK;
                    memcpy(out.stream.data() + at, it->second.data(), reuse);
                }
            }
            if (reuse < size) {
                ggml_backend_tensor_get(t, out.stream.data() + at + reuse, off + reuse, size - reuse);
            }
            out.n_reused += reuse;
            if (next) {
                next->ranges[{ t, off }].assign(out.stream.begin() + at, out.stream.begin() + at + size);
            }
        };
    }
    const size_t n = llama_state_seq_write_cb(ctx, seq, 0, w);
    GGML_ASSERT(n == out.stream.size());
    if (next) {
        next->since = since_next;
    }
    return out;
}

static std::vector<uint8_t> full_data(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> d(llama_state_seq_get_size_ext(ctx, seq, 0));
    GGML_ASSERT(llama_state_seq_get_data_ext(ctx, d.data(), d.size(), seq, 0) == d.size());
    return d;
}

// one save step on both contexts: the composite against the full capture; records advanced; returns reused bytes
static size_t step(env & e, llama_seq_id seq, record (&rec)[2], const std::string & what, bool expect_reuse,
                   bool expect_none = false) {
    size_t reused = 0;
    for (int c = 0; c < 2; ++c) {
        llama_context * ctx = c ? e.ctx_dft.get() : e.ctx_tgt.get();
        const char * nm = c ? "draft" : "target";
        // the reference: the plain streamed capture = the bytes the server stores (llama_state_seq_get_data_ext is
        //   another framing of the same sections; printed for information)
        const capture_out plain = capture(ctx, seq, nullptr, nullptr, 0);
        const std::vector<uint8_t> & full = plain.stream;
        if (getenv("TEST_STATE_INCR_VERBOSE")) {
            printf("info %s: %s write_cb %zu B, get_data_ext %zu B\n", what.c_str(), nm, full.size(), full_data(ctx, seq).size());
        }
        record nx;
        const capture_out inc = capture(ctx, seq, rec[c].since ? &rec[c] : nullptr, &nx, 1);
        check(inc.stream == full, what + ": " + nm + " incremental composite == full (" + std::to_string(full.size()) +
              " B, " + std::to_string(inc.n_reused) + " B reused of " + std::to_string(inc.n_kv) + " KV B, " +
              std::to_string(inc.n_hinted) + " KV ranges)");
        check(inc.n_hinted > 0, what + ": " + nm + " the KV ranges arrive through write_tensor_kv (also without a record)");
        if (expect_reuse) {
            check(inc.n_reused > 0, what + ": " + nm + " reuses blocks");
        }
        if (expect_none) {
            check(inc.n_reused == 0, what + ": " + nm + " reuses nothing");
        }
        reused += inc.n_reused;
        rec[c] = std::move(nx);
    }
    return reused;
}

static void test_cells() {
    // the header's clock directly (the cells below live in this binary; llama_state_mod_clock_tick ticks the library's,
    //   the same object in a static build)
    const auto tick = [] { return llama_kv_mod_clock().fetch_add(1, std::memory_order_relaxed) + 1; };
    llama_kv_cells cells;
    cells.resize(16);
    const uint64_t s0 = tick();
    check(cells.n_unmod_prefix(0, 16, s0) == 16, "cells: after resize, every cell older than a later tick");
    for (uint32_t i = 0; i < 8; ++i) {
        cells.pos_set(i, (llama_pos) i);
        cells.seq_add(i, 0);
    }
    check(cells.n_unmod_prefix(0, 8, s0) == 0, "cells: pos_set after the tick -> modified");
    const uint64_t s1 = tick();
    check(cells.n_unmod_prefix(0, 8, s1) == 8, "cells: nothing since the next tick -> 8 clean");
    check(cells.n_unmod_prefix(0, 8, 0) == 0, "cells: since 0 = no incremental capture -> 0");
    cells.seq_rm(5, 0);
    check(cells.n_unmod_prefix(0, 8, s1) == 5, "cells: seq_rm of cell 5 -> clean prefix 5");
    const uint64_t s2 = tick();
    cells.pos_set(5, 5);
    cells.seq_add(5, 0);
    const uint64_t s3 = tick();
    check(cells.n_unmod_prefix(0, 8, s2) == 5 && cells.n_unmod_prefix(0, 8, s3) == 8, "cells: re-set cell 5 dirty after s2, clean after s3");
    cells.pos_add(2, 3); // the K-shift: positions now, rows at update() -> reset_shift
    const uint64_t s4 = tick();
    check(cells.n_unmod_prefix(0, 8, s3) == 2, "cells: pos_add -> modified");
    cells.reset_shift(); // the K-shift graph has rewritten the shifted rows
    check(cells.n_unmod_prefix(0, 8, s4) == 2, "cells: reset_shift after the tick -> the shifted cell modified again");
    const uint64_t s5 = tick();
    check(cells.n_unmod_prefix(0, 8, s5) == 8, "cells: clean again after the next tick");
    cells.mod_mark_all();
    check(cells.n_unmod_prefix(0, 8, s5) == 0, "cells: mod_mark_all (stream copy) -> all modified");
    const uint64_t s6 = tick();
    cells.reset();
    check(cells.n_unmod_prefix(0, 8, s6) == 0, "cells: reset -> all modified");
}

int main() {
    llama_log_set(log_cb, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    test_cells();

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
    printf("mode: %s MTP draft\n", mode && atoi(mode) ? "sparse (LLAMA_MTP_QSA=1)" : "dense");

    // np1: append, append, rollback + other tokens, reload
    {
        auto e = make_env(1);
        record rec[2];
        const auto P1 = tokens(70, 7), P2 = tokens(60, 8), P3 = tokens(12, 9);
        feed(*e, 0, P1, 0);
        step(*e, 0, rec, "np1 first save", false, true);
        feed(*e, 0, P2, 70);
        step(*e, 0, rec, "np1 append 60", true);
        // a drafted batch of 12 saved, then 8 of them rejected (rolled back) and 8 other tokens decoded there: the
        //   rewritten cells lie inside the recorded ranges
        feed(*e, 0, P3, 130, (uint32_t) P3.size());
        step(*e, 0, rec, "np1 append 12 (drafted)", true);
        // the negative control needs the record of the state before the rollback
        record neg_rec[2] = { rec[0], rec[1] };
        rm_tail(*e, 0, 134);
        feed(*e, 0, tokens(8, 10), 134);
        for (int c = 0; c < 2; ++c) {
            llama_context * ctx = c ? e->ctx_dft.get() : e->ctx_tgt.get();
            const capture_out neg = capture(ctx, 0, &neg_rec[c], nullptr, 2);
            check(neg.stream != capture(ctx, 0, nullptr, nullptr, 0).stream, std::string("negative control: ") + (c ? "draft" : "target") +
                  " ignoring the hint after a rollback gives another stream (" + std::to_string(neg.n_reused) + " B reused)");
        }
        step(*e, 0, rec, "np1 rollback to 134 + 8 other tokens", true);
        feed(*e, 0, P2, 142);
        step(*e, 0, rec, "np1 append after the rollback", true);

        // reload: the state of another sequence history put into seq 0 (state_read writes every cell)
        auto o = make_env(1);
        feed(*o, 0, tokens(150, 11), 0);
        const std::vector<uint8_t> tgt = full_data(o->ctx_tgt.get(), 0), dft = full_data(o->ctx_dft.get(), 0);
        rm_tail(*e, 0, -1);
        GGML_ASSERT(llama_state_seq_set_data_ext(e->ctx_tgt.get(), tgt.data(), tgt.size(), 0, 0) > 0);
        GGML_ASSERT(llama_state_seq_set_data_ext(e->ctx_dft.get(), dft.data(), dft.size(), 0, 0) > 0);
        step(*e, 0, rec, "np1 after a reload", false, true);
        feed(*e, 0, P2, 150);
        step(*e, 0, rec, "np1 append after the reload", true);
    }

    // np2 unified: seq 1 decodes next to seq 0 between seq 0's saves
    {
        auto e = make_env(2);
        record rec[2];
        feed(*e, 0, tokens(66, 21), 0);
        feed(*e, 1, tokens(40, 22), 0);
        step(*e, 0, rec, "np2 seq 0 first save", false, true);
        feed(*e, 1, tokens(30, 23), 40);
        feed(*e, 0, tokens(50, 24), 66);
        feed(*e, 1, tokens(20, 25), 70);
        step(*e, 0, rec, "np2 seq 0 append next to seq 1", true);
        record rec1[2];
        step(*e, 1, rec1, "np2 seq 1 save", false, true);
    }

    printf("%d ok, %d failed\n", n_ok, n_fail);
    g_model.reset();
    llama_backend_free();
    return n_fail == 0 ? 0 : 1;
}
