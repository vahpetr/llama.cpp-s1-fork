#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-ext.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <stdexcept>

//
// [TAG_QSA_POOLED_CACHE] llama_qsa_pooled_store
//

// LLAMA_QSA_POOLED_CACHE=1: read once per memory, so two contexts of one process may differ (the replay probe
// compares them side by side); a server has one memory and so one value for its lifetime
static bool llama_qsa_pooled_cache_env() {
    const char * s = getenv("LLAMA_QSA_POOLED_CACHE");
    return s != nullptr && atoi(s) != 0;
}

// [TAG_QSA_WHOLE_BLOCKS] LLAMA_QSA_WHOLE_BLOCKS (see the header)
int llama_qsa_whole_blocks_parse(const char * s) {
    if (s == nullptr || s[0] == '\0' || strcmp(s, "0") == 0) {
        return 0;
    }
    if (strcmp(s, "1") == 0) {
        return 1;
    }
    return -1;
}

bool llama_qsa_whole_blocks() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_WHOLE_BLOCKS");
        const int    v = llama_qsa_whole_blocks_parse(s);
        if (v < 0) {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_WHOLE_BLOCKS=%s is neither 0 nor 1: using 0\n", __func__, s);
        }
        if (v == 1) {
            LLAMA_LOG_INFO("%s: qsa_select = whole_blocks (LLAMA_QSA_WHOLE_BLOCKS=1): per query the indexer_top_k/r best "
                           "whole blocks + the (q+1) mod r tail cells, as the reference (ggml_qsa_top_k_blocks; padding = a "
                           "repeat of the last id, masked in the gathered window)\n", __func__);
        } else {
            LLAMA_LOG_INFO("%s: qsa_select = cells (LLAMA_QSA_WHOLE_BLOCKS=%s): per query the top indexer_top_k + r - 1 cells\n",
                           __func__, s != nullptr ? s : "unset");
        }
        return v == 1;
    }();

    return enabled;
}

// [TAG_MTP_QSA_PHASE] LLAMA_MTP_QSA_PHASE (see the header)
bool llama_mtp_qsa_phase_env() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_MTP_QSA_PHASE");
        return s != nullptr && atoi(s) != 0;
    }();
    return enabled;
}

int llama_qsa_pool_remap_parse(const char * s) {
    if (s == nullptr || s[0] == '\0' || strcmp(s, "indexed") == 0) {
        return 1;
    }
    if (strcmp(s, "copy") == 0) {
        return 0;
    }
    return -1;
}

bool llama_qsa_pool_remap_indexed() {
    static const bool indexed = [] {
        const char * s = getenv("LLAMA_QSA_POOL_REMAP");
        const int    v = llama_qsa_pool_remap_parse(s);
        if (v < 0) {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_POOL_REMAP=%s is neither copy nor indexed: using indexed\n", __func__, s);
        }
        // WARN: the proof line must reach the server log (INFO does not)
        if (v != 0) {
            LLAMA_LOG_WARN("%s: qsa_pool_remap = indexed (LLAMA_QSA_POOL_REMAP=%s): the indexer score reads the pooled-key "
                           "store through the remap (GGML_OP_MUL_MAT_ROWS), no per-layer copy of the pool\n", __func__,
                           s != nullptr ? s : "unset");
        } else {
            LLAMA_LOG_WARN("%s: qsa_pool_remap = copy (LLAMA_QSA_POOL_REMAP=copy): every QSA layer copies the pool through "
                           "the remap (get_rows) and scores the copy (mul_mat)\n", __func__);
        }
        return v != 0;
    }();

    return indexed;
}

// [TAG_QSA_CENSUS_INC] LLAMA_QSA_CENSUS_INC / LLAMA_QSA_CENSUS_CHECK (see the header)
int llama_qsa_census_inc_parse(const char * s) {
    if (s == nullptr || s[0] == '\0' || strcmp(s, "0") == 0) {
        return 0;
    }
    if (strcmp(s, "1") == 0) {
        return 1;
    }
    return -1;
}

static bool llama_qsa_census_check_env() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_CENSUS_CHECK");
        const int    v = llama_qsa_census_inc_parse(s);
        if (v < 0) {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_CENSUS_CHECK=%s is neither 0 nor 1: using 0\n", __func__, s);
        }
        return v == 1;
    }();

    return enabled;
}

bool llama_qsa_census_inc() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_CENSUS_INC");
        const int    v = llama_qsa_census_inc_parse(s);
        if (v < 0) {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_CENSUS_INC=%s is neither 0 nor 1: using 0\n", __func__, s);
        }
        // WARN: the proof line must reach the server log (INFO does not)
        if (v == 1) {
            LLAMA_LOG_WARN("%s: qsa_census = incremental (LLAMA_QSA_CENSUS_INC=1): the block census / stamps / slot plan "
                           "re-group only the cells past the lowest touched position (persistent per-stream state)%s\n",
                           __func__, llama_qsa_census_check_env() ?
                           "; LLAMA_QSA_CENSUS_CHECK=1: every plan also runs the full path and aborts on the first "
                           "differing element" : "");
        } else {
            LLAMA_LOG_WARN("%s: qsa_census = full (LLAMA_QSA_CENSUS_INC=%s): every plan takes the full census of every "
                           "cell%s\n", __func__, s != nullptr ? s : "unset",
                           llama_qsa_census_check_env() ? "; LLAMA_QSA_CENSUS_CHECK=1 ignored (needs INC=1)" : "");
        }
        return v == 1;
    }();

    return enabled;
}

// [TAG_MULTISLOT] round 2: see the header
static std::atomic<bool> g_qsa_mm_pos_seen{false};

void llama_qsa_note_mm_positions() {
    if (!g_qsa_mm_pos_seen.exchange(true)) {
        const char * reason = nullptr;
        const bool   on     = llama_qsa_history_dense(&reason);
        LLAMA_LOG_INFO("qsa_history_dense = %s (%s)\n", on ? "on" : "off", reason);
    }
}

bool llama_qsa_history_dense(const char ** reason) {
    static const int env = [] {
        const char * s = getenv("LLAMA_QSA_HISTORY_DENSE");
        return s == nullptr ? -1 : strcmp(s, "1") == 0 ? 1 : strcmp(s, "0") == 0 ? 0 : -1;
    }();
    if (env >= 0) {
        if (reason) {
            *reason = env ? "LLAMA_QSA_HISTORY_DENSE=1" : "LLAMA_QSA_HISTORY_DENSE=0";
        }
        return env == 1;
    }
    const bool mm = g_qsa_mm_pos_seen.load(std::memory_order_relaxed);
    if (reason) {
        *reason = mm ? "default; M-RoPE image positions decoded in this process, sticky off"
                     : "default; no M-RoPE image positions decoded";
    }
    return !mm;
}

bool llama_qsa_census_check() {
    return llama_qsa_census_inc() && llama_qsa_census_check_env();
}

// [TAG_QSA_INPUTS_INC] LLAMA_QSA_INPUTS_INC (see the header); the store decides whether its prerequisites hold
static bool llama_qsa_inputs_inc_env() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_INPUTS_INC");
        const int    v = llama_qsa_census_inc_parse(s);
        if (v < 0) {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_INPUTS_INC=%s is neither 0 nor 1: using 0\n", __func__, s);
        }
        return v == 1;
    }();

    return enabled;
}

bool llama_qsa_inputs_inc() {
    return llama_qsa_inputs_inc_env();
}

// [TAG_QSA_COMPACT_BIAS] LLAMA_QSA_COMPACT_BIAS=1: read once per memory, like LLAMA_QSA_POOLED_CACHE, so the replay
// probe can run both arms in one process
static bool llama_qsa_compact_bias_env() {
    const char * s = getenv("LLAMA_QSA_COMPACT_BIAS");
    return s != nullptr && atoi(s) != 0;
}

// [TAG_QSA_COMPACT_BIAS] LLAMA_QSA_COMPACT_BIAS_CHECK=1 (diagnostic, only with the knob on): every ubatch also fills
// the old plane with the unchanged qsa_fill_bias and compares it bitwise with the tables expanded by the op's formula.
// Cost: a host vector of the old plane's full size, [n_blocks, n_tokens] f32 = 4*U/r B per pool cell (≈ 0.8 GiB at
// n_kv 393216, ub 2048), allocated per set_input, plus the old O(n_blocks*U) fill - i.e. the memory the knob saves,
// in plain host RAM. LLAMA_QSA_TAIL_ASSERT=1 with the knob on runs this check too: the tail assert lives in the old fill.
static bool llama_qsa_compact_check_env() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_COMPACT_BIAS_CHECK");
        return s != nullptr && atoi(s) != 0;
    }();

    return enabled;
}

void llama_qsa_pooled_store::reset_stream(uint32_t s) {
    GGML_ASSERT(s < strm.size());

    auto & st = strm[s];

    std::fill(st.slot_of_key.begin(), st.slot_of_key.end(), -1);
    std::fill(st.slot_key   .begin(), st.slot_key   .end(), -1);
    std::fill(st.slot_stamp .begin(), st.slot_stamp .end(),  0);
    std::fill(st.slot_seen  .begin(), st.slot_seen  .end(),  0);

    st.free_slots.resize(n_slots);
    for (uint32_t i = 0; i < n_slots; ++i) {
        st.free_slots[i] = (int32_t) (n_slots - 1 - i);
    }

    // a stamp of this stream still waiting for its graph must not revive an entry
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                [s](const pending_t & p) { return p.s == s; }), pending.end());

    // [TAG_QSA_CENSUS_INC] the slots the census state refers to are gone
    inc_mark_all(s);

    stats.n_reset++;
}

void llama_qsa_pooled_store::reset_all() {
    for (uint32_t s = 0; s < n_stream; ++s) {
        reset_stream(s);
    }
}

void llama_qsa_pooled_store::on_write(const llama_kv_cache::slot_info & sinfo) {
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & gen = strm[sinfo.strm[s]].gen;

        for (const auto idx : sinfo.idxs[s]) {
            gen[idx]++;
        }
    }
}

// [TAG_QSA_CENSUS_INC] the marks
void llama_qsa_pooled_store::inc_mark_all(uint32_t s) {
    if (!inc) {
        return;
    }

    GGML_ASSERT(s < strm.size());

    auto & c = strm[s].cen;

    c.inc_all    = true;
    c.valid      = false;
    c.inc_lo_pos = INT32_MAX;
    c.written.clear();
}

void llama_qsa_pooled_store::inc_mark_all_streams() {
    for (uint32_t s = 0; s < (uint32_t) strm.size(); ++s) {
        inc_mark_all(s);
    }
}

void llama_qsa_pooled_store::inc_mark_pos(uint32_t s, llama_pos p0) {
    if (!inc) {
        return;
    }

    GGML_ASSERT(s < strm.size());

    auto & c = strm[s].cen;

    c.inc_lo_pos = std::min<llama_pos>(c.inc_lo_pos, std::max<llama_pos>(p0, 0));
}

void llama_qsa_pooled_store::inc_mark_write_pre(uint32_t s, const llama_kv_cells & cells, const std::vector<uint32_t> & idxs) {
    if (!inc) {
        return;
    }

    for (const uint32_t idx : idxs) {
        if (!cells.is_empty(idx)) {
            // llama_kv_cache::apply_ubatch removes the old cell and purges positions of its sequence (seq_rm inside
            // the cache, past the wrappers): the census state cannot follow that cheaply
            inc_mark_all(s);
            return;
        }
    }
}

void llama_qsa_pooled_store::inc_mark_write_post(uint32_t s, const llama_kv_cells & cells, const std::vector<uint32_t> & idxs) {
    if (!inc) {
        return;
    }

    GGML_ASSERT(s < strm.size());

    auto & c = strm[s].cen;

    if (c.inc_all) {
        return;
    }

    // a long run of writes with no plan in between (dense prefill below LLAMA_QSA_MIN_KV) is cheaper as a full census
    if (c.written.size() + idxs.size() > kv_size/4 + 64) {
        inc_mark_all(s);
        return;
    }

    for (const uint32_t idx : idxs) {
        c.written.push_back((int32_t) idx);
        c.inc_lo_pos = std::min<llama_pos>(c.inc_lo_pos, cells.pos_get(idx));
    }
}

void llama_qsa_pooled_store::publish(bool ok) {
    if (ok && pending_armed) {
        for (const auto & p : pending) {
            // the slot may have been reset (and reassigned) since the plan; then its stamp stays invalid
            strm[p.s].slot_stamp[p.slot] = p.stamp;
        }

        stats.n_publish++;
    } else if (!ok) {
        // a graph that failed or was aborted may have written any part of the planned rows
        for (const uint32_t s : pending_streams) {
            reset_stream(s);
        }

        stats.n_fail++;
    }
    // ok && !armed: the plan was made (can_reuse) but the graph that ran has no store update;
    // its dirty slots were left at stamp 0 by the plan, so dropping the stamps is enough

    pending.clear();
    pending_streams.clear();
    pending_armed = false;
}

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx,
                  int32_t   qsa_ratio_override,
                ggml_type   type_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // [TAG_IDX_CACHE_K_ONLY]
        // the indexer stores and reads keys only (build_qsa_top_k: cpy_k at the write, get_k at the
        // mean-pool); nothing on that path ever calls cpy_v or get_v, so a V plane is pure waste --
        // 2x the K bytes here, because n_embd_head_v is left at the attention value length.
        // llama_kv_cache picks K-only storage off hparams.is_mla(); set the pair of MLA head sizes
        // the same way llama_kv_cache_dsv4 does (dsv4_make_k_only, src/llama-kv-cache-dsv4.cpp).
        // Note: this changes the byte layout of the indexer section of a full (non-PARTIAL_ONLY)
        // sequence state, so slot-save files written by an older build cannot be restored.
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        LLAMA_LOG_INFO("%s: creating indexer KV cache (K only), size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_idx == GGML_TYPE_COUNT ? type_k : type_idx, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    // [TAG_MTP_QSA] no recurrent layer at all: the attention + indexer caches carry the whole state
    recr_none = true;
    for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
        if (filter_recr && filter_recr(il)) {
            recr_none = false;
            break;
        }
    }

    // [TAG_MTP_QSA_PHASE] the MTP draft memory - an indexer cache that holds no trunk layer (only il >= n_layer, the
    // trailing MTP block) - takes the reference block phase under LLAMA_MTP_QSA_PHASE=1; every other memory keeps 0
    if (mem_idx && llama_mtp_qsa_phase_env()) {
        bool any_idx   = false;
        bool trunk_idx = false;
        for (uint32_t il = 0; il < model.hparams.n_layer_all; ++il) {
            if (filter_idx && filter_idx(il)) {
                any_idx    = true;
                trunk_idx |= il < model.hparams.n_layer();
            }
        }
        if (any_idx && !trunk_idx) {
            qsa_phase_v = 1;
            LLAMA_LOG_WARN("%s: qsa_phase = 1 (LLAMA_MTP_QSA_PHASE=1): MTP draft memory, reference block phase - bucket "
                           "(pos-1)/r, block start pb*r+1, tail start (q/r)*r+1; the carrier row at pos 0 is unpooled and never selected\n",
                           __func__);
        }
    }

    // [TAG_QSA_WHOLE_BLOCKS] read (and prove) the selection form at load
    if (mem_idx) {
        (void) llama_qsa_whole_blocks();
    }

    // [TAG_QSA_COMPACT_BIAS] the query table carries 1 << seq_id in 32 bits
    if (mem_idx && llama_qsa_compact_bias_env()) {
        if (n_seq_max <= 32) {
            qsa_compact = true;
            LLAMA_LOG_INFO("%s: LLAMA_QSA_COMPACT_BIAS=1: the QSA per-block bias is built on the device from "
                           "per-block/per-query tables (GGML_OP_QSA_BIAS)%s\n", __func__,
                           llama_qsa_compact_check_env() ? ", LLAMA_QSA_COMPACT_BIAS_CHECK=1" : "");
        } else {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_COMPACT_BIAS=1 ignored: n_seq_max = %u > 32, the plane is kept\n",
                           __func__, n_seq_max);
        }
    }

    if (!mem_idx || !llama_qsa_pooled_cache_env()) {
        return;
    }

    // [TAG_QSA_POOLED_CACHE] the main QSA layers only: the layers the indexer cache holds that run with a
    // positive effective ratio. v1 serves one ratio; a GGUF mixing ratios keeps the recompute everywhere.
    const auto & hp = model.hparams;

    std::vector<int32_t> ils;
    uint32_t ratio = 0;
    bool     mixed = false;

    // [TAG_MTP_QSA] n_layer_all: the MTP context's indexer holds the trailing block; filter_idx picks the context's layers
    for (uint32_t il = 0; il < hp.n_layer_all; ++il) {
        if (!filter_idx(il)) {
            continue;
        }

        const uint32_t r = llama_hparams_qsa_ratio(hp, qsa_ratio_override, il);
        if (r == 0) {
            continue;
        }

        mixed |= ratio != 0 && r != ratio;
        ratio  = r;

        ils.push_back((int32_t) il);
    }

    if (ils.empty() || mixed) {
        LLAMA_LOG_WARN("%s: LLAMA_QSA_POOLED_CACHE=1 ignored: %s\n", __func__,
                ils.empty() ? "no QSA layer" : "the QSA layers mix compress ratios");
        return;
    }

    auto st = std::make_unique<llama_qsa_pooled_store>();

    st->ratio    = ratio;
    st->idx_dim  = hp.indexer_head_size;
    st->kv_size  = mem_idx->get_size();
    st->n_stream = mem_idx->get_n_stream();
    st->n_slots  = (st->kv_size + ratio - 1)/ratio;
    st->n_rows   = st->n_slots + 1;
    st->phase    = qsa_phase_v;   // [TAG_MTP_QSA_PHASE]

    // one context per buffer type, as llama_kv_cache places its layers
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr> ctx_map;

    for (const int32_t il : ils) {
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }

        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*ils.size()*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context for the QSA pooled-key cache");
            }

            it = ctx_map.emplace(buft, ggml_context_ptr(ctx)).first;
        }

        ggml_tensor * t = ggml_new_tensor_2d(it->second.get(), GGML_TYPE_F32, st->idx_dim, (int64_t) st->n_rows*st->n_stream);
        ggml_format_name(t, "cache_qsa_pooled_l%d", il);

        st->layers[il] = t;
    }

    size_t total = 0;

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams_idx.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer, as llama_kv_cache does
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for the QSA pooled-key cache");
        }

        // unused rows are read by the score GEMM before the bias drops them: keep them finite
        ggml_backend_buffer_clear(buf, 0);

        total += ggml_backend_buffer_get_size(buf);

        LLAMA_LOG_INFO("%s: %10s QSA pooled-key cache buffer size = %8.2f MiB\n", __func__,
                ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        st->ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    st->strm.resize(st->n_stream);
    for (auto & ss : st->strm) {
        ss.gen        .assign(st->kv_size, 0);
        ss.slot_of_key.assign(st->kv_size, -1);
        ss.slot_key   .assign(st->n_slots, -1);
        ss.slot_stamp .assign(st->n_slots,  0);
        ss.slot_seen  .assign(st->n_slots,  0);
    }

    // [TAG_QSA_CENSUS_INC] read (and prove) the census mode at load
    st->inc = llama_qsa_census_inc();

    // [TAG_QSA_INPUTS_INC] the device-resident input mirrors, next to the first QSA layer's store
    if (llama_qsa_inputs_inc_env()) {
        if (!st->inc || !qsa_compact) {
            LLAMA_LOG_WARN("%s: qsa_inputs = full (LLAMA_QSA_INPUTS_INC=1 ignored: needs %s)\n", __func__,
                    !st->inc ? "LLAMA_QSA_CENSUS_INC=1" : "LLAMA_QSA_COMPACT_BIAS=1 (and n_seq_max <= 32)");
        } else {
            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
            if (offload) {
                buft = ggml_backend_dev_buffer_type(model.dev_layer(ils[0]));
            }

            ggml_init_params params = {
                /*.mem_size   =*/ size_t(4u*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context for the QSA input mirrors");
            }

            const int64_t ns = st->n_stream;

            st->mir_t[llama_qsa_pooled_store::MIR_CELL_BLK] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) st->kv_size*ns);
            st->mir_t[llama_qsa_pooled_store::MIR_REMAP]    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) st->n_slots*ns);
            st->mir_t[llama_qsa_pooled_store::MIR_BTAB]     = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2*(int64_t) st->n_slots*ns);
            ggml_set_name(st->mir_t[llama_qsa_pooled_store::MIR_CELL_BLK], "qsa_mir_cell_blk");
            ggml_set_name(st->mir_t[llama_qsa_pooled_store::MIR_REMAP],    "qsa_mir_remap");
            ggml_set_name(st->mir_t[llama_qsa_pooled_store::MIR_BTAB],     "qsa_mir_blk");

            ggml_backend_buffer_t buf;
            if (hparams_idx.no_alloc) {
                buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0);
                for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf;
                }
            } else {
                buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            }
            if (!buf) {
                ggml_free(ctx);
                throw std::runtime_error("failed to allocate buffer for the QSA input mirrors");
            }

            // never read before a full write; zero for a deterministic reserve graph
            ggml_backend_buffer_clear(buf, 0);

            st->mir_on = true;
            st->mir_ctxs_bufs.emplace_back(ggml_context_ptr(ctx), buf);

            // WARN: the proof line must reach the server log (INFO does not)
            LLAMA_LOG_WARN("%s: qsa_inputs = incremental (LLAMA_QSA_INPUTS_INC=1): cell_blk / remap / block table are "
                           "device-resident mirrors (%s, %.2f MiB), set_input writes the changed runs only%s\n", __func__,
                           ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0,
                           llama_qsa_census_check() ? "; LLAMA_QSA_CENSUS_CHECK=1: every set_input downloads and compares them" : "");
        }
    } else {
        LLAMA_LOG_WARN("%s: qsa_inputs = full (LLAMA_QSA_INPUTS_INC unset/0): the QSA layout inputs are filled and uploaded "
                       "whole per graph\n", __func__);
    }

    qsa_store = std::move(st);
    qsa_store->reset_all();
    qsa_store->stats.n_reset = 0;

    LLAMA_LOG_INFO("%s: LLAMA_QSA_POOLED_CACHE=1: %zu layers, ratio %u, %u block slots + 1 spare row x %u streams, "
            "idx_dim %u, f32: %zu bytes\n", __func__, ils.size(), ratio, qsa_store->n_slots, qsa_store->n_stream,
            qsa_store->idx_dim, total);

    // [TAG_QSA_POOL_REMAP] read (and prove) how the score reads the store, at load
    (void) llama_qsa_pool_remap_indexed();
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + rs_rollback_depth) tokens of each seq must stay in the same ubatch,
                //   so that every rollback plane the graph writes for it stays valid
                const uint32_t rs_depth = balloc.get_rs_rollback_depth();
                ubatch = balloc.split_equal(n_ubatch, !unified, rs_depth > 0 ? rs_depth + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

// [TAG_KV_GRAPH_SPAN] forward the worst-case-reserve cap to every sub-cache
void llama_memory_hybrid_idx::set_n_kv_reserve_max(uint32_t n_kv_max) {
    llama_memory_hybrid::set_n_kv_reserve_max(n_kv_max);

    if (get_mem_idx()) {
        get_mem_idx()->set_n_kv_reserve_max(n_kv_max);
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }

    // [TAG_QSA_POOLED_CACHE] invalidate every stream: data == true zeroes the raw rows behind the stamps
    qsa_store_invalidate(-1);
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // [TAG_MTP_QSA] no recurrent layer: nothing can refuse a partial removal, as in the plain KV cache this replaces
    if (recr_none) {
        // the layerless recurrent cache only tracks the sequence's last position; drop it whole, so the next
        // find_slot does not see a position gap (it holds no state to keep). The indexer + census marks follow below.
        get_mem_recr()->seq_rm(seq_id, -1, -1);
    } else if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    // [TAG_QSA_CENSUS_INC] the positions >= p0 of seq_id's stream (every stream for seq_id < 0) may have lost cells
    if (qsa_store) {
        if (seq_id < 0) {
            for (uint32_t s = 0; s < qsa_store->n_stream; ++s) {
                qsa_store->inc_mark_pos(s, p0);
            }
        } else {
            qsa_store->inc_mark_pos(qsa_stream_of(seq_id), p0);
        }
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

llama_pos llama_memory_hybrid_idx::seq_pos_min(llama_seq_id seq_id) const {
    return recr_none ? get_mem_attn()->seq_pos_min(seq_id) : llama_memory_hybrid::seq_pos_min(seq_id);
}

llama_pos llama_memory_hybrid_idx::seq_pos_max(llama_seq_id seq_id) const {
    return recr_none ? get_mem_attn()->seq_pos_max(seq_id) : llama_memory_hybrid::seq_pos_max(seq_id);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }

    // [TAG_QSA_POOLED_CACHE] within a stream seq_cp is metadata only: a regrouped block changes its member
    // list, which the stamp carries. Across streams the next update() copies the whole raw buffer of the
    // source stream over the destination (llama_kv_cache::update): its derived entries go wholesale.
    if (qsa_store && qsa_stream_of(seq_id_src) != qsa_stream_of(seq_id_dst)) {
        qsa_store->reset_stream(qsa_stream_of(seq_id_dst));
    }

    // [TAG_QSA_CENSUS_INC] a second sequence on the cells: the full census (reset_stream above marks the cross-stream case)
    if (qsa_store) {
        qsa_store->inc_mark_all(qsa_stream_of(seq_id_dst));
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }

    // [TAG_QSA_CENSUS_INC] every stream may lose sequences
    if (qsa_store) {
        qsa_store->inc_mark_all_streams();
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }

    // [TAG_QSA_CENSUS_INC] positions move (and cells may empty)
    if (qsa_store) {
        if (seq_id < 0) {
            qsa_store->inc_mark_all_streams();
        } else {
            qsa_store->inc_mark_all(qsa_stream_of(seq_id));
        }
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }

    // [TAG_QSA_CENSUS_INC] positions move
    if (qsa_store) {
        if (seq_id < 0) {
            qsa_store->inc_mark_all_streams();
        } else {
            qsa_store->inc_mark_all(qsa_stream_of(seq_id));
        }
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    // [TAG_QSA_POOLED_CACHE] the persistent store counts as memory, like the caches it derives from
    if (qsa_store) {
        for (const auto & [ctx, buf] : qsa_store->ctxs_bufs) {
            mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
        }
        for (const auto & [ctx, buf] : qsa_store->mir_ctxs_bufs) {
            mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return mb;
}

// [TAG_CELL_EXT_PROVENANCE] written in front of the indexer section by a build whose indexer cells hold the true M-RoPE
// x / y at save time (the 0239 restore fix: before it, a restore zeroed them and later inference ran on that). An
// indexer section starts with its stream count (<= LLAMA_MAX_SEQ), so an unmarked (older) section cannot read as it.
static const uint32_t LLAMA_IDX_STATE_XY_MARK = LLAMA_STATE_IXY_MARK; // "IXY1"

static std::atomic<uint64_t> g_idx_xy_rejects{0};
static std::atomic<uint64_t> g_idx_xy_ckpt_rejects{0};

uint64_t llama_memory_hybrid_idx::n_xy_rejects() {
    return g_idx_xy_rejects.load();
}

uint64_t llama_memory_hybrid_idx::n_xy_ckpt_rejects() {
    return g_idx_xy_ckpt_rejects.load();
}

namespace {
// [TAG_CELL_EXT_PROVENANCE] a reader that hands out the already-read u32 words first, then the wrapped stream
class llama_io_read_unread : public llama_io_read_i {
public:
    llama_io_read_unread(llama_io_read_i & in, const uint32_t * words, size_t n_words) : in(in), n_avail(n_words * sizeof(uint32_t)) {
        memcpy(head, words, n_avail);
    }

    void read(void * dst, size_t size) override {
        auto * out = static_cast<uint8_t *>(dst);
        while (size > 0 && n_head < n_avail) {
            *out++ = reinterpret_cast<const uint8_t *>(head)[n_head++];
            size--;
        }
        if (size > 0) {
            in.read(out, size);
        }
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        GGML_ASSERT(n_head == n_avail); // tensors never come first in a section
        in.read_tensor(tensor, offset, size);
    }

    size_t n_bytes() override {
        return in.n_bytes();
    }

private:
    llama_io_read_i & in;
    uint32_t          head[3] = {};
    size_t            n_avail = 0;
    size_t            n_head  = 0;
};
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // [TAG_MTP_QSA] the MTP draft memory: the draft KV's own semantics (llama_kv_cache writes its cells whatever the
    // PARTIAL_ONLY flag says), and the indexer section follows it the same way; no recurrent section
    if (recr_none) {
        get_mem_attn()->state_write(io, seq_id, flags);
        if (mem_idx) {
            io.write(&LLAMA_IDX_STATE_XY_MARK, sizeof(LLAMA_IDX_STATE_XY_MARK));
            mem_idx->state_write(io, seq_id, flags);
        }
        return;
    }

    // [TAG_CELL_EXT_PROVENANCE] a PARTIAL_ONLY trunk stream (the target's context checkpoints: recurrent only) carries the
    // mark too, in front (its end cannot be probed in a stream), as a decoy recurrent header {cell_count 1, pos 0,
    // n_seq_id = mark} that an older reader refuses at once: v59 a1b6c324e src/llama-memory-hybrid-idx.cpp:933 hands a
    // PARTIAL_ONLY stream to llama_memory_recurrent::state_read, which reads cell_count = 1 (llama-memory-recurrent.cpp:881),
    // reserves a 1-cell ubatch (:1034) and fails the first record's `n_seq_id != 0` check (:1043, INVALID) - no large
    // allocation (a bare mark word would be read as 828 987 465 cells, ~27 GB). The real recurrent section follows
    if (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
        const uint32_t decoy[3] = { 1, 0, LLAMA_IDX_STATE_XY_MARK };
        io.write(decoy, sizeof(decoy));
    }

    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            io.write(&LLAMA_IDX_STATE_XY_MARK, sizeof(LLAMA_IDX_STATE_XY_MARK));
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // [TAG_QSA_POOLED_CACHE] the pooled store is not part of a slot-save file (state_write is unchanged): a
    // restore writes raw rows behind the stamps, so the stream's derived entries go wholesale, before the
    // read - and so on every path out of it, PARTIAL_ONLY and failure included (state_drop repeats it)
    qsa_store_invalidate(seq_id);

    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        // [TAG_MTP_QSA] mirrors state_write's recr_none branch: attention (whatever PARTIAL_ONLY says), then the indexer
        if (recr_none) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
            if (mem_idx) {
                idx_state_read(io, seq_id, flags, sinfos_attn);
            }
            return;
        }

        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
            get_mem_recr()->state_read(io, seq_id, flags);
        } else {
            recr_state_read_partial(io, seq_id, flags);
        }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                idx_state_read(io, seq_id, flags, sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::recr_state_read_partial(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // [TAG_CELL_EXT_PROVENANCE] a context checkpoint (PARTIAL_ONLY: recurrent only, the live attention / indexer cells
    // stay). One written before the fix (no mark) may hold recurrent rows computed on zeroed indexer x / y after an
    // image restore - with an image in the live sequence it is refused (the caller drops it and re-processes the
    // prompt); without one it is text-only state and loads as before
    // the stream opens either with the decoy header {1, 0, mark} or with a pre-fix recurrent section: {cell_count, then
    // per cell pos, n_seq_id (always 0 in a per-seq stream)} - only a cell_count of 1 is followed by those two words
    uint32_t w[3] = {};
    size_t   n_w  = 1;
    io.read(&w[0], sizeof(w[0]));
    if (w[0] == 1) {
        io.read(&w[1], 2 * sizeof(uint32_t));
        n_w = 3;
        if (w[2] == LLAMA_IDX_STATE_XY_MARK) {
            get_mem_recr()->state_read(io, seq_id, flags);
            return;
        }
    }

    // pre-fix: only image cells within the checkpoint's own prefix count (its recurrent cell's position; the live
    // sequence may run past it until the caller trims it)
    const llama_pos p_max = w[0] == 1 ? (llama_pos) w[1] : (llama_pos) INT32_MAX;
    const uint32_t n_img = get_mem_attn()->n_cells_2d_seq(seq_id, p_max);
    if (n_img > 0) {
        const uint64_t n = ++g_idx_xy_ckpt_rejects;
        LLAMA_LOG_WARN("%s: context checkpoint saved before the indexer x/y fix, %u image cells in the sequence: rejected "
                "(checkpoint rejections so far: %" PRIu64 ")\n", __func__, n_img, n);
        io.set_status(LLAMA_STATE_READ_INVALID);
        throw std::runtime_error("pre-fix context checkpoint of a sequence with image cells");
    }

    llama_io_read_unread io_head(io, w, n_w);
    try {
        get_mem_recr()->state_read(io_head, seq_id, flags);
    } catch (...) {
        io.set_status(io_head.status);
        throw;
    }
    io.set_status(io_head.status);
}

void llama_memory_hybrid_idx::idx_state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags,
        const llama_kv_cache::slot_info_vec_t & sinfos_attn) {
    // [TAG_CELL_EXT_PROVENANCE] the attention cells are restored at this point (exactly, x / y included)
    uint32_t head = 0;
    io.read(&head, sizeof(head));

    const bool marked = head == LLAMA_IDX_STATE_XY_MARK;

    if (!marked) {
        // saved by a build before the fix: if it holds image cells, its indexer x / y may have been zeroed by a restore
        // before further inference was run and saved on top, so the K/V and recurrent rows can carry that - not
        // repairable here. Text-only states are unaffected (x = y = pos, no consumer tells 0 from pos) and are healed.
        // Conservative: a pre-fix image state that never went through a restore is correct but refused too; text cells
        // cannot look like image ones (pos_add keeps the ext, but an M-RoPE cache has get_can_shift() false, so no
        // K-shift / context shift / cache reuse ever moves them)
        const uint32_t n_img = get_mem_attn()->n_cells_2d(sinfos_attn);
        if (n_img > 0) {
            const uint64_t n = ++g_idx_xy_rejects;
            LLAMA_LOG_WARN("%s: state saved before the indexer x/y fix holds %u image cells: rejected as a cache miss "
                    "(rejections so far: %" PRIu64 ")\n", __func__, n_img, n);
            io.set_status(LLAMA_STATE_READ_INVALID);
            throw std::runtime_error("pre-fix sequence state with image cells");
        }
    }

    mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn, marked ? nullptr : &head);
    idx_ext_restore(sinfos_attn);
}

void llama_memory_hybrid_idx::idx_ext_restore(const llama_kv_cache::slot_info_vec_t & sinfos_attn) {
    // [TAG_CELL_EXT_RESTORE] the indexer's own restore cannot be trusted for x / y:
    //  - an indexer state without the cell ext (no PLE: the indexer's rope NONE gives has_cell_ext() false, e.g. the MTP
    //    draft memory) restored x = y = tok = 0;
    //  - an indexer state with the ext (PLE) written by a build before this fix after a restore of its own already holds
    //    x = y = 0 on the restored rows, and would keep them through every re-save.
    // The live indexer cells hold the ext apply_ubatch wrote from the same ubatches as the attention cells, the attention
    // cache serializes and restores x / y exactly, and the indexer took its layout - so x / y always come from there
    // (old cache entries heal on the first restore; no cache-key change). The token is kept when the indexer serialized
    // it. An attention cache without the ext (no M-RoPE, no PLE) has nothing better: the indexer keeps what it restored.
    if (!get_mem_attn()->has_cell_ext()) {
        return;
    }

    const int64_t t0 = ggml_time_us();
    const uint32_t n = mem_idx->ext_copy_from(*get_mem_attn(), sinfos_attn, /*keep_tok =*/ mem_idx->has_cell_ext());
    LLAMA_LOG_DEBUG("%s: indexer cell ext of %u cells from the attention cells in %.3f ms\n", __func__, n,
            (ggml_time_us() - t0) / 1000.0);
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // [TAG_QSA_POOLED_CACHE] a half-read restore left raw rows the stamps know nothing about
    qsa_store_invalidate(seq_id);

    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

llama_qsa_pooled_store * llama_memory_hybrid_idx::get_qsa_store() const {
    return qsa_store.get();
}

uint32_t llama_memory_hybrid_idx::qsa_stream_of(llama_seq_id seq_id) const {
    // llama_kv_cache maps every sequence to stream 0 when unified, else sequence s to stream s
    return mem_idx == nullptr || mem_idx->get_n_stream() == 1 ? 0 : (uint32_t) seq_id;
}

void llama_memory_hybrid_idx::qsa_store_invalidate(llama_seq_id seq_id) {
    if (!qsa_store) {
        return;
    }

    if (seq_id < 0) {
        qsa_store->reset_all();
    } else {
        qsa_store->reset_stream(qsa_stream_of(seq_id));
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    full(true),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    // [TAG_KV_GRAPH_SPAN] init_full(), not the llama_kv_cache_context constructor: the attention side
    // goes through llama_kv_cache::init_full() (llama_memory_hybrid_context above), which applies the
    // n_kv_reserve_max cap. Constructing this one directly would default n_kv_span to 0 == "the whole
    // cache", so with the cap set the indexer would report a wider span than the attention cache and
    // qwen4exp's "the indexer cache must track the attention cache cell for cell" assertion would abort
    // graph construction. set_n_kv_reserve_max is forwarded to mem_idx and the padding rule is the same
    // on both caches, so the two spans stay equal with the cap on and with it off.
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_full()) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)),
    is_update(true) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    // [TAG_QSA_POOLED_CACHE] the write generations need the cells each ubatch writes
    sinfos_idx_copy(mem->get_qsa_store() != nullptr ? sinfos_idx : slot_info_vec_t()),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    qsa_plan.valid = false;
    hist_valid     = false;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    auto * st = get_qsa_store();

    // [TAG_QSA_CENSUS_INC] the cells of a write (per slot-info stream), for the marks around it
    auto write_cells = [&](uint32_t s) -> const llama_kv_cells & {
        const auto & sinfo = sinfos_idx_copy[i_cur];
        const auto & ub    = get_ubatch();
        return mem->get_mem_idx()->get_cells(ub.seq_id[s*sinfo.size()][0]);
    };

    const bool write = st != nullptr && st->inc && ctx_idx && i_cur < sinfos_idx_copy.size();

    if (write) {
        // an overwrite of a used cell (and the purge it triggers inside the cache) forces a full census
        const auto & sinfo = sinfos_idx_copy[i_cur];
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            st->inc_mark_write_pre(sinfo.strm[s], write_cells(s), sinfo.idxs[s]);
        }
    }

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // [TAG_QSA_POOLED_CACHE] the committed write path: this ubatch's raw indexer keys go into these cells
    // (cpy_k in build_qsa_top_k), so every block holding one of them must be pooled again. Batch planning
    // (llama_kv_cache::prepare applies and restores cell metadata around find_slot) never gets here.
    if (st != nullptr && i_cur < sinfos_idx_copy.size()) {
        st->on_write(sinfos_idx_copy[i_cur]);
    }

    if (write) {
        const auto & sinfo = sinfos_idx_copy[i_cur];
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            st->inc_mark_write_post(sinfo.strm[s], write_cells(s), sinfo.idxs[s]);
        }
    }

    // [TAG_QSA_CENSUS_INC] a K-shift / stream copy update: the full census
    if (st != nullptr && is_update) {
        st->inc_mark_all_streams();
    }

    qsa_plan.valid = false;
    hist_valid     = false;

    return res;
}

bool llama_memory_hybrid_idx_context::qsa_history_complete(const llama_ubatch & ubatch) const {
    if (full || mem == nullptr || mem->get_mem_idx() == nullptr || ubatch.n_tokens == 0) {
        return true;
    }
    if (hist_valid) {
        return hist_complete;
    }

    // per sequence of the ubatch: its depth = max pos + 1 over its rows (the sequential row of ubatch.pos)
    std::vector<std::pair<llama_seq_id, llama_pos>> seqs;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id s = ubatch.seq_id[i][j];
            auto it = std::find_if(seqs.begin(), seqs.end(), [s](const auto & e) { return e.first == s; });
            if (it == seqs.end()) {
                seqs.emplace_back(s, ubatch.pos[i] + 1);
            } else {
                it->second = std::max(it->second, ubatch.pos[i] + 1);
            }
        }
    }

    hist_complete = true;
    for (const auto & [s, depth] : seqs) {
        const llama_kv_cells & cells = mem->get_mem_idx()->get_cells(s);
        const llama_pos p_min = cells.seq_pos_min(s);
        const llama_pos p_max = cells.seq_pos_max(s);
        const uint32_t  n_pos = cells.seq_pos_count(s);
        if (p_min == 0 && (int64_t) n_pos == (int64_t) p_max + 1 && p_max >= depth - 1) {
            continue;
        }
        hist_complete = false;
        // the proof: the first 16 partial ubatches of the process, then every 256th
        static std::atomic<uint64_t> n_partial{0};
        const uint64_t k = n_partial.fetch_add(1);
        if (k < 16 || k % 256 == 0) {
            LLAMA_LOG_INFO("[qsa-history] ctx=%s seq=%d depth=%d positions=%u min=%d max=%d complete=0 route=dense reason=partial (#%" PRIu64 ")\n",
                    mem->is_recr_none() ? "dft" : "tgt", (int) s, (int) depth, n_pos, (int) p_min, (int) p_max, k + 1);
        }
        break;
    }
    hist_valid = true;

    return hist_complete;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

llama_qsa_pooled_store * llama_memory_hybrid_idx_context::get_qsa_store() const {
    return mem != nullptr ? mem->get_qsa_store() : nullptr;
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

// [TAG_QSA_TAIL_FIX] (M1.5, triage row T3)
//
// Only a COMPLETE group of r cells is given a block id (`grp_slots[g] != slots_full -> continue`
// below), so the up-to-(r-1) newest cells of a stream - the incomplete final block, which at decode
// contains the query's OWN key - have no block of their own. They are routed to `dead_bid`, and in
// the live per-block bias path every block index >= n_bid is scored -INFINITY, while the `1e9f`
// tail boost only ever reaches ALLOCATED complete blocks. Above the budget (width < n_kv) those
// cells therefore lose the top-k to any finite cell and the query cannot attend to its own key.
// (Below the budget width == n_kv, top_k returns every column and the bias cannot change the
// selection - so this fix does not touch the below-budget identity oracle.)
//
// The fix routes them to the same spare id but scores it +1e9f, which is what the reference does
// with the tail: always visible. Two facts make the spare id safe:
//   * if any cell is unpooled then n_bid < n_blocks, so `dead_bid == n_bid` is a genuinely unused
//     block id. Proof: n_bid complete groups occupy n_bid*r distinct cells, so
//     n_bid*r + 1 <= n_kv <= n_blocks*r, hence n_bid < n_blocks. When there is no unpooled cell,
//     dead_bid is reached by EMPTY cells only and must stay -inf-ish; the guard below keeps it so.
//   * empty, foreign and future cells also land on dead_bid, and the caller adds the attention
//     mask afterwards, which is -INFINITY for exactly those. 1e9f + (-inf) = -inf, never a nan.
//
// Default ON. LLAMA_QSA_TAIL_FIX=0 restores the -inf tail for an A/B against the pre-fix numbers.
static bool qsa_tail_fix_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_TAIL_FIX");
        return s == nullptr || atoi(s) != 0;
    }();

    return enabled;
}

// LLAMA_QSA_TAIL_ASSERT=1 turns on an O(n_kv) per-token re-check of the property the fix exists
// for: every non-empty cell this query can see whose in-block index is at or past the tail start
// must carry a finite, boosted bias. Off by default - at 213k cells x 2048 tokens it is the graph.
static bool qsa_tail_assert_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_TAIL_ASSERT");
        return s != nullptr && atoi(s) != 0;
    }();

    return enabled;
}

bool llama_memory_hybrid_idx_context::qsa_compact_bias() const {
    return mem != nullptr && mem->qsa_compact_bias();
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * qsa_blk,
        ggml_tensor * qsa_qry,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(mem != nullptr && mem->get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);

    // [TAG_QSA_COMPACT_BIAS] exactly one of the plane and the tables
    const bool compact = bias == nullptr;
    GGML_ASSERT(compact == (qsa_blk != nullptr) && (qsa_blk != nullptr) == (qsa_qry != nullptr));
    GGML_ASSERT(!compact || (blk_bias && qsa_compact_bias()));

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = compact ? nullptr : (float *) bias->data;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    qsa_census_stream cs;

    std::vector<float> check_plane;

    for (int64_t s = 0; s < n_ns; ++s) {
        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        qsa_census   (s, ubatch, n_kv, n_ns, n_blocks, r, blk_bias, cur_cell_blk, cur_blk_cells, dst_blk_pos, cs);

        if (compact) {
            GGML_ASSERT(ggml_backend_buffer_is_host(qsa_blk->buffer) && ggml_backend_buffer_is_host(qsa_qry->buffer));
            GGML_ASSERT(qsa_blk->ne[1] == n_blocks && qsa_blk->ne[2] == n_ns && qsa_qry->ne[1] == n_tokens/n_ns);

            qsa_fill_bias_compact(s, ubatch, n_ns, n_blocks, r, (int32_t *) qsa_blk->data, (int32_t *) qsa_qry->data, cs);

            if (llama_qsa_compact_check_env() || qsa_tail_assert_enabled()) {
                qsa_compact_check(s, ubatch, n_kv, n_ns, n_blocks, r, cur_cell_blk,
                        (const int32_t *) qsa_blk->data, (const int32_t *) qsa_qry->data, cs, check_plane);
            }
        } else {
            qsa_fill_bias(s, ubatch, n_kv, n_ns, n_blocks, r, blk_bias, cur_cell_blk, dst_bias, cs);
        }
    }
}

// the block census of stream s: which complete group of r cells is which compact block id, the cells of
// each block (cur_blk_cells), the block of each cell (cur_cell_blk) and the rope positions (dst_blk_pos,
// all streams, sec-major). The caller zero-fills dst_blk_pos once for all streams.
void llama_qsa_census_full(
        const llama_kv_cells & cells,
        bool is_pos_2d,
        int64_t n_kv,
        int64_t n_blocks,
        int64_t r,
        bool blk_bias,
        int32_t * cur_cell_blk,
        int32_t * cur_blk_cells,
        int32_t * blk_pos,
        int64_t pos_stride,
        llama_qsa_census_stream & cs,
        llama_qsa_census_flags * flags,
        int64_t phase) {
    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    std::vector<int32_t> & blk_of = cs.blk_of;
    std::vector<int32_t> & order  = cs.order;
    std::vector<int32_t> & rank   = cs.rank;

    blk_of.resize(n_kv);

    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> & bid_idx  = cs.bid_idx;
    std::vector<int32_t> & bid_cell = cs.bid_cell;

    std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

    bid_idx  .clear();
    bid_cell .clear();
    bid_slot0.clear();

    int n_seq_present = 0;

    for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
        if (cells.seq_pos_min(sq) >= 0) {
            n_seq_present++;
        }
    }

    const bool one_seq = n_seq_present <= 1;

    // [TAG_QSA_WINDOW_MISALIGNED] a cell past the cell window (block index >= n_blocks): no block can cover it.
    // A complete history keeps every position below the window, so this stays false; a memory without its prefix
    // (the MTP draft after a legacy /slots restore or a dropped draft, its catch-up rows written at positions >= the
    // restored depth) has such cells, and in a unified pool they sit next to other sequences' cells. They are
    // excluded from selection below (routed to an id both bias fills score -inf), never pooled.
    bool oor = false;

    cs.oor_seqs.clear();

    bool dup = false;

    bool & ranked = cs.ranked;
    ranked = false;

    auto group_cells = [&]() {
        // -1 means no usable block: an incomplete or short group cannot be pooled
        std::fill(blk_of.begin(),   blk_of.end(),   -1);
        std::fill(cell_grp.begin(), cell_grp.end(), -1);
        std::fill(grp_head.begin(), grp_head.end(), -1);

        grp_next .clear();
        grp_first.clear();
        grp_slot0.clear();
        grp_slots.clear();
        grp_bid  .clear();

        oor = false;
        dup = false;

        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                continue;
            }

            const int64_t idx = (ranked ? rank[j] : cells.pos_get(j)) - phase;

            // [TAG_MTP_QSA_PHASE] the MTP carrier row (index -1 with phase 1) never joins a block
            if (idx < 0) {
                continue;
            }

            const int64_t pb  = idx/r;

            if (pb >= n_blocks) {
                oor = true;
                continue;
            }

            int32_t g = -1;

            for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                if (one_seq || cells.seq_set((uint32_t) grp_first[c]) == cells.seq_set((uint32_t) j)) {
                    g = c;
                    break;
                }
            }

            if (g < 0) {
                g = (int32_t) grp_first.size();

                grp_next .push_back(grp_head[pb]);
                grp_first.push_back((int32_t) j);
                grp_slot0.push_back(-1);
                grp_slots.push_back(0);
                grp_bid  .push_back(-1);

                grp_head[pb] = g;
            }

            const uint64_t bit = uint64_t(1) << (idx%r);

            dup |= (grp_slots[g] & bit) != 0;

            cell_grp[j]   = g;
            grp_slots[g] |= bit;

            if (idx%r == 0) {
                grp_slot0[g] = (int32_t) j;
            }
        }
    };

    group_cells();

    // mrope repeats one position across an image, so rank cells instead of using the position
    if (dup && is_pos_2d && one_seq) {
        order.clear();
        order.reserve(n_kv);

        for (int64_t j = 0; j < n_kv; ++j) {
            if (!cells.is_empty(j)) {
                order.push_back((int32_t) j);
            }
        }

        // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
        std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
            const llama_pos pa = cells.pos_get(a);
            const llama_pos pb = cells.pos_get(b);

            if (pa != pb) {
                return pa < pb;
            }

            const auto & ea = cells.ext_get(a);

            return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
        });

        rank.assign(n_kv, -1);

        for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
            rank[order[k]] = (int32_t) k;
        }

        ranked = true;

        group_cells();
    }

    int32_t n_bid = 0;

    for (int64_t pb = 0; pb < n_blocks; ++pb) {
        for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
            if (grp_slots[g] != slots_full) {
                continue;
            }

            grp_bid[g] = n_bid++;

            bid_idx  .push_back((int32_t) (pb*r));
            bid_cell .push_back(grp_first[g]);
            bid_slot0.push_back(grp_slot0[g]);
        }
    }

    GGML_ASSERT(n_bid <= n_blocks);

    cs.n_bid = n_bid;

    for (int32_t b = 0; b < n_bid; ++b) {
        // [TAG_MTP_QSA_PHASE] the rope position of the block's first member: its index + the phase
        const int32_t p0 = bid_idx[b] + (int32_t) phase;

        int32_t sec_pos[4] = { p0, p0, p0, p0 };

        if (ranked) {
            const int32_t   c = bid_slot0[b];
            const llama_pos p = cells.pos_get(c);
            const auto &    e = cells.ext_get(c);

            sec_pos[0] = p;
            sec_pos[1] = e.y;
            sec_pos[2] = e.x;
            sec_pos[3] = p;
        }

        for (int64_t sec = 0; sec < 4; ++sec) {
            blk_pos[sec*pos_stride + b] = sec_pos[sec];
        }
    }

    // unpooled cells point at a dead block. [TAG_QSA_TAIL_FIX]: when that id is a spare one
    // (which is implied by the existence of any unpooled cell, see the proof at the top of the
    // file) the per-block bias below boosts it instead of killing it, so the incomplete tail -
    // including the query's own key at decode - stays selectable above the budget.
    const int32_t dead_bid = n_bid < n_blocks ? n_bid : n_blocks - 1;

    for (int64_t j = 0; j < n_kv; ++j) {
        const int32_t g = cell_grp[j];

        blk_of[j] = g < 0 ? -1 : grp_bid[g];

        if (blk_of[j] >= 0) {
            const int64_t idx = (ranked ? rank[j] : cells.pos_get(j)) - phase;

            cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
        }

        cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
    }

    // [TAG_QSA_WHOLE_BLOCKS] D6 (docs/verification/2026-10-03-indexer-reference-notes.md): the spare id above is
    // boosted for EVERY query, so every unpooled visible cell is selected outside the block budget - right for the
    // tail (the newest incomplete group of a sequence), wrong for a FRAGMENT: an incomplete group in an older bucket
    // (a block straddling a seq_cp fork point, a position hole), which the reference never forms because it groups
    // each query's own visible list. In the whole-block form the budget is the reference's (n_sel blocks + the
    // tail), so a fragment - an unpooled cell that is the tail of NONE of its sequences - goes to a second unused
    // id, which both bias fills score -inf (b > n_bid). A cell that is the tail of one of its sequences and a
    // fragment for another keeps the boost (a per-block bias cannot tell the two queries apart; <= r - 1 cells per
    // such group); so does every fragment when no second unused id exists (n_bid + 1 == n_blocks: the cache is full
    // to the last block). With one sequence and no position holes there is no fragment and nothing changes.
    if (llama_qsa_whole_blocks() && n_bid < n_blocks) {
        const int64_t newest_rank = ranked ? (int64_t) order.size() - 1 - phase : -1;

        int64_t n_unrouted = 0;

        for (int64_t j = 0; j < n_kv; ++j) {
            if (blk_of[j] >= 0 || cells.is_empty(j)) {
                continue;
            }

            const int64_t idx = (ranked ? rank[j] : cells.pos_get(j)) - phase;

            bool tail_of_some = false;
            if (ranked) {
                tail_of_some = llama_qsa_idx_bucket(idx, r) == llama_qsa_idx_bucket(newest_rank, r);
            } else {
                const auto & set = cells.seq_set((uint32_t) j);
                for (int32_t sq = 0; sq < LLAMA_MAX_SEQ && !tail_of_some; ++sq) {
                    tail_of_some = set.test(sq) &&
                        llama_qsa_idx_bucket(idx, r) == llama_qsa_idx_bucket((int64_t) cells.seq_pos_max(sq) - phase, r);
                }
            }

            if (tail_of_some) {
                continue;
            }

            if (n_bid + 1 < n_blocks) {
                cur_cell_blk[j] = n_bid + 1;
            } else {
                n_unrouted++;
            }
        }

        if (n_unrouted > 0) {
            static std::atomic<bool> warned { false };
            if (!warned.exchange(true)) {
                LLAMA_LOG_WARN("%s: LLAMA_QSA_WHOLE_BLOCKS=1: %" PRId64 " fragment cell(s) keep the tail boost: no second "
                               "unused block id (n_bid %d + 1 == n_blocks %" PRId64 "); warned once\n",
                               __func__, n_unrouted, n_bid, n_blocks);
            }
        }
    }

    // [TAG_MTP_QSA_PHASE] the carrier (index < 0) is no entry of the reference: in both selection forms it goes to the
    // second unused id, which both bias fills score -inf (the spare id would boost it); none left: it keeps the boost
    if (phase > 0 && n_bid + 1 < n_blocks) {
        for (int64_t j = 0; j < n_kv; ++j) {
            if (!cells.is_empty(j) && (ranked ? rank[j] : cells.pos_get(j)) < phase) {
                cur_cell_blk[j] = n_bid + 1;
            }
        }
    }

    // [TAG_QSA_WINDOW_MISALIGNED] the out-of-window cells (block index >= n_blocks; only reached where this census
    // used to abort with "qsa: cell position runs past the cell window", so every census without such a cell is
    // unchanged). Under the per-block bias they are excluded from selection: routed to the second unused id n_bid + 1,
    // which both bias fills (qsa_fill_bias, the compact tables) score -inf for every query - after the fragment and the
    // carrier routing above, so neither can boost them. A query of a FOREIGN sequence never sees them anyway (the
    // attention mask the caller adds is -inf for foreign cells); a query of the OWNING sequence has depth > n_kv and
    // runs the dense layer (qwen4exp_qsa_dense), which the fills assert (cs.oor_seqs). Without a second unused id
    // (n_bid + 1 == n_blocks) the cells keep the spare id: for a foreign query the attention mask drops them, exactly
    // as it drops the empty / foreign cells that already share that id ([TAG_QSA_TAIL_FIX]); an owning query is the
    // asserted impossible case either way.
    if (oor) {
        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j) || blk_of[j] >= 0) {
                continue;
            }

            const int64_t idx = (ranked ? rank[j] : cells.pos_get(j)) - phase;

            if (idx < 0 || idx/r < n_blocks) {
                continue;
            }

            if (blk_bias && n_bid + 1 < n_blocks) {
                cur_cell_blk[j] = n_bid + 1;
            }

            const auto & set = cells.seq_set((uint32_t) j);
            for (int32_t sq = 0; sq < LLAMA_MAX_SEQ; ++sq) {
                if (set.test(sq) && std::find(cs.oor_seqs.begin(), cs.oor_seqs.end(), sq) == cs.oor_seqs.end()) {
                    cs.oor_seqs.push_back(sq);
                }
            }
        }
    }

    if (flags) {
        flags->one_seq = one_seq;
        flags->dup     = dup;
        flags->oor     = oor;
    }
}

// [TAG_QSA_WINDOW_MISALIGNED] a sparse query of a sequence that owns an out-of-window cell would lose that cell (or,
// without a second unused id, see it boosted outside its block): impossible, its depth is past the window
static void qsa_assert_not_oor_seq(const llama_qsa_census_stream & cs, llama_seq_id seq_id) {
    if (!cs.oor_seqs.empty()) {
        GGML_ASSERT(std::find(cs.oor_seqs.begin(), cs.oor_seqs.end(), (int32_t) seq_id) == cs.oor_seqs.end() &&
                    "qsa: sparse query of a sequence with a cell past the cell window (the layer must run dense)");
    }
}

// the block census of stream s of the ubatch (dst_blk_pos: all streams, sec-major; the caller zero-fills it once)
void llama_memory_hybrid_idx_context::qsa_census(
        int64_t s,
        const llama_ubatch * ubatch,
        int64_t n_kv,
        int64_t n_ns,
        int64_t n_blocks,
        int64_t r,
        bool blk_bias,
        int32_t * cur_cell_blk,
        int32_t * cur_blk_cells,
        int32_t * dst_blk_pos,
        qsa_census_stream & cs) const {
    const int64_t n_tps = ubatch->n_tokens/n_ns;          // tokens per stream

    // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
    const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
    const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

    llama_qsa_census_full(cells, ubatch->is_pos_2d(), n_kv, n_blocks, r, blk_bias, cur_cell_blk, cur_blk_cells,
                          dst_blk_pos + s*n_blocks, n_blocks*n_ns, cs, nullptr, mem->qsa_phase());
}

// [TAG_QSA_COMPACT_BIAS] the query side of the bias, shared by qsa_fill_bias and qsa_fill_bias_compact so both
// fills see the same integers: the query's position - or, for a ranked (M-RoPE image) census, its rank among the
// stream's cells, found by a binary search over `order` - and the start of its incomplete tail block
// [TAG_MTP_QSA_PHASE] minus the memory's phase: the query's census index
static int64_t qsa_query_pos(const llama_ubatch * ubatch, int64_t i, bool ranked, const std::vector<int32_t> & order,
                             const llama_kv_cells & cells, int64_t phase) {
    const int64_t n_tokens = ubatch->n_tokens;

    int64_t q = ubatch->pos[i];

    if (ranked) {
        const llama_pos qt = ubatch->pos[i];
        const llama_pos qy = ubatch->pos[i + n_tokens];
        const llama_pos qx = ubatch->pos[i + n_tokens*2];

        int64_t lo = 0;
        int64_t hi = (int64_t) order.size();

        while (lo < hi) {
            const int64_t   mid = (lo + hi)/2;
            const int32_t   c   = order[mid];
            const llama_pos pc  = cells.pos_get(c);

            if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        q = lo - 1;
    }

    return q - phase;
}

static inline int64_t qsa_tail_start(int64_t q, int64_t r) {
    return llama_qsa_tail_start(q, r);
}

// the per-token bias of stream s from its census
void llama_memory_hybrid_idx_context::qsa_fill_bias(
        int64_t s,
        const llama_ubatch * ubatch,
        int64_t n_kv,
        int64_t n_ns,
        int64_t n_blocks,
        int64_t r,
        bool blk_bias,
        const int32_t * cur_cell_blk,
        float * dst_bias,
        const qsa_census_stream & cs) const {
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_tps    = n_tokens/n_ns;          // tokens per stream

    const bool tail_fix    = qsa_tail_fix_enabled();     // [TAG_QSA_TAIL_FIX]
    const bool tail_assert = qsa_tail_assert_enabled();

    const int64_t phase    = mem->qsa_phase();           // [TAG_MTP_QSA_PHASE]

    const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
    const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

    const int32_t n_bid  = cs.n_bid;
    const bool    ranked = cs.ranked;

    const std::vector<int32_t> & bid_idx  = cs.bid_idx;
    const std::vector<int32_t> & bid_cell = cs.bid_cell;
    const std::vector<int32_t> & order    = cs.order;
    const std::vector<int32_t> & rank     = cs.rank;
    const std::vector<int32_t> & blk_of   = cs.blk_of;

    const int32_t dead_bid = n_bid < n_blocks ? n_bid : n_blocks - 1;

    // n_bid == n_blocks => no unpooled non-empty cell exists => dead_bid aliases a REAL block
    // and must keep that block's own bias
    const bool tail_bid_spare = n_bid < n_blocks;

    for (int64_t ii = 0; ii < n_tps; ++ii) {
        const int64_t      i      = s*n_tps + ii;
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        if (blk_bias) {
            qsa_assert_not_oor_seq(cs, seq_id);
        }

        const int64_t q = qsa_query_pos(ubatch, i, ranked, order, cells, phase);

        // the tail is an incomplete block and is always visible, as in the reference
        const int64_t tail_start = qsa_tail_start(q, r);

        if (blk_bias) {
            // a block sits wholly inside or outside the tail, so one value covers it
            // the caller adds the attention mask, which drops empty, foreign and future cells
            float * cur_blk_bias = dst_bias + i*n_blocks;

            for (int64_t b = 0; b < n_blocks; ++b) {
                if (b >= n_bid) {
                    // [TAG_QSA_TAIL_FIX] (T3) the spare id every unpooled cell points at: the
                    // incomplete tail block. Boost it - the attention mask the caller adds
                    // still drops the empty / foreign / future cells that share the id.
                    cur_blk_bias[b] = (tail_fix && tail_bid_spare && b == dead_bid) ? 1e9f : -INFINITY;
                    continue;
                }

                if (!cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                    cur_blk_bias[b] = -INFINITY;
                    continue;
                }

                // finite, so it can never meet a -inf and produce a nan
                cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;
            }

            if (tail_assert) {
                // every visible cell at or past the tail start must be boosted
                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j) || !cells.seq_has(j, seq_id)) {
                        continue;
                    }

                    const int64_t idx = (ranked ? rank[j] : cells.pos_get(j)) - phase;

                    if (idx < tail_start || idx > q) {
                        continue;
                    }

                    GGML_ASSERT(cur_blk_bias[cur_cell_blk[j]] >= 1e9f &&
                            "qsa: visible tail cell is not selectable [TAG_QSA_TAIL_FIX]");
                }
            }

            continue;
        }

        float * cur_bias = dst_bias + i*n_kv;

        for (int64_t j = 0; j < n_kv; ++j) {
            float v = -INFINITY;

            if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                const int64_t idx = (ranked ? rank[j] : cells.pos_get(j)) - phase;

                if (idx <= q) {
                    // finite, so it can never meet a -inf and produce a nan
                    // [TAG_QSA_TAIL_FIX] this per-cell path already boosts the tail whether or
                    // not the cell has a block, so T3 never applied here - only to the per-block
                    // path above, which is the live one for a causal, non-alibi run
                    v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                }
            }

            cur_bias[j] = v;
        }
    }
}

void llama_qsa_fill_blk_tab(const llama_kv_cells & cells, const llama_qsa_census_stream & cs, int64_t n_blocks, int64_t b0,
                            int64_t b1, bool tail_fix, int32_t * blk) {
    const int32_t n_bid = cs.n_bid;

    GGML_ASSERT(n_bid <= n_blocks && 0 <= b0 && b0 <= b1 && b1 <= n_blocks);

    const bool tail_bid_spare = n_bid < n_blocks;

    for (int64_t b = b0; b < std::min<int64_t>(b1, n_bid); ++b) {
        const auto & set = cells.seq_set((uint32_t) cs.bid_cell[b]);

        uint32_t vis = 0;
        for (uint32_t k = 0; k < 32; ++k) {
            vis |= set.test(k) ? 1u << k : 0u;
        }

        blk[2*b + 0] = cs.bid_idx[b];
        blk[2*b + 1] = (int32_t) vis;
    }

    for (int64_t b = std::max<int64_t>(b0, n_bid); b < b1; ++b) {
        const bool boost = tail_fix && tail_bid_spare && b == n_bid;

        blk[2*b + 0] = INT32_MAX;
        blk[2*b + 1] = (int32_t) (boost ? 0xFFFFFFFFu : 0u);
    }
}

// [TAG_QSA_COMPACT_BIAS] the tables that GGML_OP_QSA_BIAS turns back into qsa_fill_bias's per-block plane.
// Per block b of stream s (dst_blk [2, n_blocks, n_ns]):
//   b <  n_bid: key = bid_idx[b], vis = the seq bitmask of bid_cell[b] (the cell the old fill tests)
//   b >= n_bid: key = INT32_MAX,  vis = ~0 for the [TAG_QSA_TAIL_FIX] spare id (tail_fix && n_bid < n_blocks &&
//               b == n_bid: 1e9 for every query, as before), 0 for every other id (-inf, as before)
// Per query i (dst_qry [2, n_tps, n_ns]): tail = qsa_tail_start(qsa_query_pos(...)) - the same integers the old fill
// compares - and bit = 1 << seq_id[i][0]. The op writes (vis & bit) == 0 ? -inf : (key >= tail ? 1e9f : 0.0f).
// When n_bid == n_blocks there is no spare id: dead_bid aliases a real block, which keeps its own entry here too.
void llama_memory_hybrid_idx_context::qsa_fill_bias_compact(
        int64_t s,
        const llama_ubatch * ubatch,
        int64_t n_ns,
        int64_t n_blocks,
        int64_t r,
        int32_t * dst_blk,
        int32_t * dst_qry,
        const qsa_census_stream & cs) const {
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_tps    = n_tokens/n_ns;

    const bool tail_fix = qsa_tail_fix_enabled();     // [TAG_QSA_TAIL_FIX]

    const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
    const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

    const int32_t n_bid = cs.n_bid;

    GGML_ASSERT(n_bid <= n_blocks);

    // [TAG_QSA_INPUTS_INC] dst_blk == nullptr: the block table lives in the device mirror, only the query side is filled
    if (dst_blk != nullptr) {
        llama_qsa_fill_blk_tab(cells, cs, n_blocks, 0, n_blocks, tail_fix, dst_blk + 2*s*n_blocks);
    }

    for (int64_t ii = 0; ii < n_tps; ++ii) {
        const int64_t      i      = s*n_tps + ii;
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        GGML_ASSERT(seq_id >= 0 && seq_id < 32);

        qsa_assert_not_oor_seq(cs, seq_id);

        const int64_t tail = qsa_tail_start(qsa_query_pos(ubatch, i, cs.ranked, cs.order, cells, mem->qsa_phase()), r);

        GGML_ASSERT(tail >= 0 && tail < INT32_MAX);

        dst_qry[2*i + 0] = (int32_t) tail;
        dst_qry[2*i + 1] = (int32_t) (1u << seq_id);
    }
}

void llama_memory_hybrid_idx_context::qsa_compact_check(
        int64_t s,
        const llama_ubatch * ubatch,
        int64_t n_kv,
        int64_t n_ns,
        int64_t n_blocks,
        int64_t r,
        const int32_t * cur_cell_blk,
        const int32_t * blk,
        const int32_t * qry,
        const qsa_census_stream & cs,
        std::vector<float> & plane) const {
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_tps    = n_tokens/n_ns;

    // the old plane, [n_blocks, n_tokens] as the input tensor was; qsa_fill_bias writes stream s's rows
    plane.resize((size_t) n_blocks*n_tokens);
    qsa_fill_bias(s, ubatch, n_kv, n_ns, n_blocks, r, /*blk_bias =*/ true, cur_cell_blk, plane.data(), cs);

    const int32_t * be = blk + 2*s*n_blocks;

    for (int64_t ii = 0; ii < n_tps; ++ii) {
        const int64_t  i    = s*n_tps + ii;
        const int32_t  tail = qry[2*i + 0];
        const uint32_t bit  = (uint32_t) qry[2*i + 1];

        for (int64_t b = 0; b < n_blocks; ++b) {
            const int32_t  key = be[2*b + 0];
            const uint32_t vis = (uint32_t) be[2*b + 1];

            // the op's formula (ggml_compute_forward_qsa_bias, qsa_bias.comp)
            const float v_new = (vis & bit) == 0 ? -INFINITY : (key >= tail ? 1e9f : 0.0f);
            const float v_old = plane[(size_t) i*n_blocks + b];

            if (memcmp(&v_new, &v_old, sizeof(float)) != 0) {
                GGML_ABORT("[qsa-compact-check] MISMATCH s=%lld i=%lld b=%lld old=%g new=%g (n_bid=%d n_blocks=%lld "
                           "key=%d vis=0x%08x tail=%d bit=0x%08x)", (long long) s, (long long) i, (long long) b,
                           (double) v_old, (double) v_new, cs.n_bid, (long long) n_blocks, key, vis, tail, bit);
            }
        }
    }

    // one count per ubatch: its last stream closes it
    if (s == n_ns - 1) {
        static std::atomic<uint64_t> n_checked { 0 };

        const uint64_t n = ++n_checked;
        if (n % 64 == 0 || n == 1) {
            LLAMA_LOG_WARN("[qsa-compact-check] %llu ubatches identical\n", (unsigned long long) n);
        }
    }
}

//
// [TAG_QSA_POOLED_CACHE] the pooled-key cache: plan, inputs, publication
//

// the block stamp: every value folded in order by a multiply-add (cheap: n_blocks x ~20 values per graph),
// then one splitmix64 finaliser per block
static inline uint64_t qsa_fold(uint64_t h, uint64_t v) {
    return (h ^ v) * 0x100000001b3ull + 0x9e3779b97f4a7c15ull;
}

static inline uint64_t qsa_mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

// the dirty-list capacity per stream: at least `need` (the dirty blocks of the fullest stream, plus one entry
// for the spare row when that stream has a spare id), bucketed so that a decode graph keeps one shape across
// steps (graph reuse), and never more than a cold rebuild of every block needs: need <= n_blocks always
// (dirty <= n_bid, and the spare entry exists only when n_bid < n_blocks), so the cold-rebuild graph pools
// exactly the n_blocks rows the recompute pools
static int64_t qsa_store_cap(int64_t need, int64_t n_blocks) {
    int64_t cap = 8;
    while (cap < need) {
        cap *= 2;
    }
    return std::min<int64_t>(cap, n_blocks);
}

// [TAG_QSA_CENSUS_INC] the per-stream views of a full-path plan: its own arrays
static void llama_qsa_plan_views_own(llama_qsa_plan & pl) {
    const int64_t n_ns     = pl.n_ns;
    const int64_t n_kv     = pl.n_kv;
    const int64_t n_blocks = pl.n_blocks;
    const int64_t r        = pl.ratio;

    pl.v_cell_blk .resize(n_ns);
    pl.v_blk_cells.resize(n_ns);
    pl.v_remap    .resize(n_ns);
    pl.v_blk_pos  .resize(n_ns);
    pl.v_cs       .resize(n_ns);

    pl.v_pos_stride = n_blocks*n_ns;

    for (int64_t s = 0; s < n_ns; ++s) {
        pl.v_cell_blk [s] = pl.cell_blk .data() + s*n_kv;
        pl.v_blk_cells[s] = pl.blk_cells.data() + s*(r*n_blocks);
        pl.v_remap    [s] = pl.remap    .data() + s*n_blocks;
        pl.v_blk_pos  [s] = pl.blk_pos  .data() + s*n_blocks;
        pl.v_cs       [s] = &pl.cs[s];
    }
}

// today's plan (the body of qsa_store_plan before [TAG_QSA_CENSUS_INC]), unchanged but for its inputs: the cells and
// cache stream of each ubatch stream come in `in`, the census is llama_qsa_census_full
int64_t llama_qsa_plan_full(llama_qsa_pooled_store & st_ref, const llama_qsa_plan_in & in, llama_qsa_plan & pl,
                            std::vector<std::vector<uint64_t>> * stamps_out) {
    llama_qsa_pooled_store * st = &st_ref;

    const uint32_t ratio    = st->ratio;
    const int64_t  r        = ratio;
    const int64_t  n_kv     = in.n_kv;
    const int64_t  n_blocks = (n_kv + r - 1)/r;
    const bool     blk_bias = in.blk_bias;

    const int64_t n_ns     = in.n_ns;

    pl.valid    = false;
    pl.inc      = false;
    pl.ratio    = ratio;
    pl.n_kv     = n_kv;
    pl.n_blocks = n_blocks;
    pl.n_ns     = n_ns;
    pl.blk_bias = blk_bias;

    pl.cell_blk .resize(n_kv*n_ns);
    pl.blk_cells.resize(r*n_blocks*n_ns);
    pl.blk_pos  .assign(4*n_blocks*n_ns, 0);
    pl.cs       .resize(n_ns);
    pl.remap    .resize(n_blocks*n_ns);

    std::vector<uint32_t>              strm_of(n_ns);   // cache stream of each ubatch stream
    std::vector<std::vector<uint64_t>> stamps(n_ns);    // [n_bid] per stream
    std::vector<std::vector<int32_t>>  slot_of(n_ns);   // [n_bid] the slot the block owns already, -1

    size_t   need        = 1;
    uint32_t n_dirty_all = 0;
    uint32_t n_bid_all   = 0;

    // pass 1, read-only: the census, the identity of every complete block, and how many would be dirty
    for (int64_t s = 0; s < n_ns; ++s) {
        int32_t * cur_cell_blk  = pl.cell_blk .data() + s*n_kv;
        int32_t * cur_blk_cells = pl.blk_cells.data() + s*(r*n_blocks);

        auto & cs = pl.cs[s];

        const auto &   cells = *in.cells[s];
        const uint32_t g     = in.g[s];

        llama_qsa_census_full(cells, in.is_pos_2d, n_kv, n_blocks, r, blk_bias, cur_cell_blk, cur_blk_cells,
                              pl.blk_pos.data() + s*n_blocks, n_blocks*n_ns, cs, nullptr, st->phase);

        GGML_ASSERT(g < st->n_stream);
        GGML_ASSERT((s == 0 || g == strm_of[0] + s) && "qsa: the ubatch streams are one contiguous range (get_k)");

        strm_of[s] = g;

        const auto & ss = st->strm[g];

        const int32_t n_bid = cs.n_bid;

        stamps [s].resize(n_bid);
        slot_of[s].assign(n_bid, -1);

        size_t n_dirty = 0;

        for (int32_t b = 0; b < n_bid; ++b) {
            const int32_t * mem_cells = cur_blk_cells + b*r;

            uint64_t h = qsa_fold(0x51534150u /* "QSAP" */, ((uint64_t) g << 32) | ((uint64_t) r << 1) | (cs.ranked ? 1 : 0));

            for (int64_t k = 0; k < r; ++k) {
                const int32_t c = mem_cells[k];
                const auto &  e = cells.ext_get(c);

                h = qsa_fold(h, ((uint64_t) (uint32_t) c << 32) | ss.gen[c]);
                h = qsa_fold(h, (uint32_t) cells.pos_get(c));
                h = qsa_fold(h, ((uint64_t) (uint32_t) e.x << 32) | (uint32_t) e.y);
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                h = qsa_fold(h, (uint32_t) pl.blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b]);
            }

            h = qsa_mix(h);
            h = h == 0 ? 1 : h;   // 0 is "invalid"

            stamps[s][b] = h;

            // the block's key cell owns a slot already?
            const int32_t key  = mem_cells[0];
            const int32_t slot = ss.slot_of_key[key];

            if (slot >= 0 && ss.slot_key[slot] == key) {
                slot_of[s][b] = slot;
            }

            n_dirty += slot_of[s][b] < 0 || ss.slot_stamp[slot_of[s][b]] != h;
        }

        need = std::max(need, n_dirty + (n_bid < n_blocks ? 1 : 0));

        n_dirty_all += (uint32_t) n_dirty;
        n_bid_all   += (uint32_t) n_bid;
    }

    const int64_t cap = qsa_store_cap((int64_t) need, n_blocks);
    GGML_ASSERT(cap >= (int64_t) need);

    // cap == n_blocks: the FULL pool. The dirty list becomes every compact block id in order - exactly the
    // recompute's blk_cells/blk_pos, padding rows included - so the graph pools what the recompute pools and
    // uses it directly as the selection's keys. The stream's slots are re-laid in compact order (slot = block
    // id), so the store write is one contiguous copy with no index input. This is the shape of a cold rebuild
    // and of every reserve graph: the recompute's graph plus one CPY node.
    const bool full_pool = cap == n_blocks;

    pl.cap      = cap;
    pl.full     = full_pool;
    pl.s0       = strm_of[0];

    const uint32_t serial = ++st->serial;

    // per stream: (slot, block id) of every dirty block
    std::vector<std::vector<std::pair<int32_t, int32_t>>> dirty(n_ns);

    // pass 2: commit the slot assignment, collect the dirty list and the pending stamps
    for (int64_t s = 0; s < n_ns; ++s) {
        const uint32_t g     = strm_of[s];
        const int32_t  n_bid = pl.cs[s].n_bid;

        auto & ss = st->strm[g];

        const int32_t * cur_blk_cells = pl.blk_cells.data() + s*(r*n_blocks);

        if (full_pool) {
            // every slot of the stream is re-laid: block b owns slot b, the rest are free
            for (uint32_t slot = 0; slot < st->n_slots; ++slot) {
                if (ss.slot_key[slot] >= 0) {
                    ss.slot_of_key[ss.slot_key[slot]] = -1;
                }
                ss.slot_key[slot]   = -1;
                ss.slot_stamp[slot] = 0;
            }

            ss.free_slots.clear();
            for (int32_t slot = (int32_t) st->n_slots - 1; slot >= n_bid; --slot) {
                ss.free_slots.push_back(slot);
            }

            for (int32_t b = 0; b < n_bid; ++b) {
                const int32_t key = cur_blk_cells[b*r];

                ss.slot_key[b]      = key;
                ss.slot_of_key[key] = b;
                ss.slot_seen[b]     = serial;

                pl.remap[s*n_blocks + b] = (int32_t) (g*st->n_rows + b);

                dirty[s].emplace_back(b, b);
                st->pending.push_back({ g, b, stamps[s][b] });
            }
        } else {
            for (int32_t b = 0; b < n_bid; ++b) {
                if (slot_of[s][b] >= 0) {
                    ss.slot_seen[slot_of[s][b]] = serial;
                }
            }

            // the slots whose block is gone are free again
            for (uint32_t slot = 0; slot < st->n_slots; ++slot) {
                const int32_t key = ss.slot_key[slot];

                if (key >= 0 && ss.slot_seen[slot] != serial) {
                    ss.slot_of_key[key] = -1;
                    ss.slot_key[slot]   = -1;
                    ss.slot_stamp[slot] = 0;
                    ss.free_slots.push_back((int32_t) slot);
                }
            }

            // new blocks take a free slot; a block whose stamp differs is dirty
            for (int32_t b = 0; b < n_bid; ++b) {
                int32_t slot = slot_of[s][b];

                if (slot < 0) {
                    GGML_ASSERT(!ss.free_slots.empty() && "qsa: more complete blocks than block slots");

                    slot = ss.free_slots.back();
                    ss.free_slots.pop_back();

                    const int32_t key = cur_blk_cells[b*r];

                    ss.slot_key[slot]   = key;
                    ss.slot_of_key[key] = slot;
                    ss.slot_stamp[slot] = 0;
                    ss.slot_seen[slot]  = serial;
                }

                pl.remap[s*n_blocks + b] = (int32_t) (g*st->n_rows + slot);

                if (ss.slot_stamp[slot] != stamps[s][b]) {
                    // not valid until the graph that pools it has run
                    ss.slot_stamp[slot] = 0;

                    dirty[s].emplace_back(slot, b);

                    st->pending.push_back({ g, slot, stamps[s][b] });
                }
            }
        }

        // every compact id past the complete blocks reads the spare row, which this graph rewrites
        for (int64_t b = n_bid; b < n_blocks; ++b) {
            pl.remap[s*n_blocks + b] = (int32_t) (g*st->n_rows + st->n_slots);
        }

        st->pending_streams.push_back(g);
    }

    // the dirty list; the padding entries recompute the spare row: r copies of cell 0 at position 0, which is
    // what the recompute gives every compact id >= n_bid (blk_cells and blk_pos are zero-filled there). A stream
    // with a spare id always gets at least one padding entry (need above), so its spare row is this graph's.
    // (The full pool has no spare-row write: it reads its own rows, and its padding rows land in free slots.)
    pl.dirty_cells.assign(r*cap*n_ns, 0);
    pl.dirty_pos  .assign(4*cap*n_ns, 0);
    pl.dirty_dst  .resize(cap*n_ns);

    for (int64_t s = 0; s < n_ns; ++s) {
        const uint32_t g = strm_of[s];

        const int32_t * cur_blk_cells = pl.blk_cells.data() + s*(r*n_blocks);

        for (int64_t e = 0; e < cap; ++e) {
            if (e < (int64_t) dirty[s].size()) {
                const int32_t slot = dirty[s][e].first;
                const int32_t b    = dirty[s][e].second;

                for (int64_t k = 0; k < r; ++k) {
                    pl.dirty_cells[s*(r*cap) + e*r + k] = cur_blk_cells[b*r + k];
                }

                for (int64_t sec = 0; sec < 4; ++sec) {
                    pl.dirty_pos[sec*(cap*n_ns) + s*cap + e] = pl.blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b];
                }

                pl.dirty_dst[s*cap + e] = (int32_t) (g*st->n_rows + slot);
            } else {
                pl.dirty_dst[s*cap + e] = (int32_t) (g*st->n_rows + st->n_slots);
            }
        }
    }

    pl.valid  = true;
    pl.serial = serial;

    llama_qsa_plan_views_own(pl);

    if (stamps_out) {
        *stamps_out = std::move(stamps);
    }

    st->stats.n_plan++;
    st->stats.n_full     += full_pool;
    st->stats.n_dirty    += full_pool ? n_bid_all : n_dirty_all;
    st->stats.n_valid    += full_pool ? 0 : n_bid_all - n_dirty_all;
    st->stats.last_dirty  = full_pool ? n_bid_all : n_dirty_all;
    st->stats.last_cap    = (uint32_t) cap;
    st->stats.last_n_bid  = n_bid_all;

    return cap;
}

//
// [TAG_QSA_CENSUS_INC] the incremental census + plan
//

// the stamp of block b: the full path's pass-1 formula, value for value
static uint64_t qsa_block_stamp(const llama_kv_cells & cells, const std::vector<uint32_t> & gen, uint32_t g, int64_t r,
                                bool ranked, const int32_t * mem_cells, const int32_t * blk_pos, int64_t pos_stride, int64_t b) {
    uint64_t h = qsa_fold(0x51534150u /* "QSAP" */, ((uint64_t) g << 32) | ((uint64_t) r << 1) | (ranked ? 1 : 0));

    for (int64_t k = 0; k < r; ++k) {
        const int32_t c = mem_cells[k];
        const auto &  e = cells.ext_get(c);

        h = qsa_fold(h, ((uint64_t) (uint32_t) c << 32) | gen[c]);
        h = qsa_fold(h, (uint32_t) cells.pos_get(c));
        h = qsa_fold(h, ((uint64_t) (uint32_t) e.x << 32) | (uint32_t) e.y);
    }

    for (int64_t sec = 0; sec < 4; ++sec) {
        h = qsa_fold(h, (uint32_t) blk_pos[sec*pos_stride + b]);
    }

    h = qsa_mix(h);
    return h == 0 ? 1 : h;   // 0 is "invalid"
}

static bool qsa_one_seq(const llama_kv_cells & cells) {
    int n_seq_present = 0;

    for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
        if (cells.seq_pos_min(sq) >= 0) {
            n_seq_present++;
        }
    }

    return n_seq_present <= 1;
}

// per ubatch stream, what pass 1 leaves for pass 2
struct qsa_inc_stream_work {
    bool    fast      = false;
    int32_t region_b0 = 0;                          // fast: the first re-grouped block id (all ids below are kept)
    std::vector<int32_t> slot_of;                   // the slot block region_b0 + k owns already, -1
    std::vector<std::pair<int32_t, int32_t>> old_region; // fast: (slot, key cell) of the last census's blocks >= region_b0
    std::vector<int32_t> carry_dirty;               // fast: the prefix blocks (< region_b0) whose stamp is not committed
};

// the fast census of one stream: false (nothing changed) unless the census state can be extended exactly - one sequence,
// no ranked / dup / oor census, no n_kv shrink, no full-census mark, a re-grouped region not larger than ~1/8 of the
// blocks. On true, C holds the census of the current cells, as llama_qsa_census_full would write it, with blk_pos at
// stride n_slots; ws.region_b0 / old_region describe what changed for the slot plan.
// returns LLAMA_QSA_CEN_FAST or the reason of the fallback (llama_qsa_pooled_store::stats_t::n_cen_fb)
static int llama_qsa_census_inc(llama_qsa_pooled_store & st, uint32_t g, const llama_kv_cells & cells, int64_t n_kv,
                                 int64_t n_blocks, int64_t r, qsa_inc_stream_work & ws) {
    auto & C = st.strm[g].cen;

    if (C.inc_all || !C.valid) {
        return LLAMA_QSA_CEN_FB_MARK;
    }
    if (!C.inc_ok) {
        return LLAMA_QSA_CEN_FB_SHAPE;
    }
    if (n_kv < C.n_kv) {
        return LLAMA_QSA_CEN_FB_SHRINK;
    }

    if (!qsa_one_seq(cells)) {
        return LLAMA_QSA_CEN_FB_SHAPE;
    }

    const int32_t n_bid_old = C.cs.n_bid;

    // [TAG_MTP_QSA_PHASE] every bucket below is one of census indices (position - phase)
    const int64_t phase = st.phase;

    const int64_t lo_pb = std::min<int64_t>(C.tail_pb, C.inc_lo_pos == INT32_MAX ? INT64_MAX :
                                                       llama_qsa_idx_bucket((int64_t) C.inc_lo_pos - phase, r));

    // block ids ascend with the position bucket (one group per bucket with one sequence)
    const int32_t region_b0 = (int32_t) (std::lower_bound(C.cs.bid_idx.begin(), C.cs.bid_idx.begin() + n_bid_old,
                (int32_t) std::min<int64_t>(lo_pb*r, INT32_MAX)) - C.cs.bid_idx.begin());

    if (n_bid_old - region_b0 > std::max<int32_t>(64, n_bid_old/8)) {
        return LLAMA_QSA_CEN_FB_REGION;
    }

    // the candidate cells: the cells of the re-grouped blocks, the unpooled cells, the written cells, the cells n_kv adds
    if (C.seen.size() != st.kv_size) {
        C.seen.assign(st.kv_size, 0);
        C.seen_epoch = 0;
    }
    if (++C.seen_epoch == 0) {
        std::fill(C.seen.begin(), C.seen.end(), 0);
        C.seen_epoch = 1;
    }
    const uint32_t ep = C.seen_epoch;

    std::vector<int32_t> cand;
    cand.reserve((size_t) (n_bid_old - region_b0)*r + C.unpooled.size() + C.written.size() + (size_t) (n_kv - C.n_kv));

    auto add = [&](int32_t j) {
        if (j < 0 || j >= n_kv || C.seen[j] == ep) {
            return;
        }
        C.seen[j] = ep;
        cand.push_back(j);
    };

    for (int64_t b = region_b0; b < n_bid_old; ++b) {
        for (int64_t k = 0; k < r; ++k) {
            add(C.blk_cells[b*r + k]);
        }
    }
    for (const int32_t j : C.unpooled) {
        add(j);
    }
    for (const int32_t j : C.written) {
        add(j);
    }
    for (int64_t j = C.n_kv; j < n_kv; ++j) {
        add((int32_t) j);
    }

    std::vector<std::pair<llama_pos, int32_t>> region;   // (census index, cell) of the non-empty candidates
    std::vector<int32_t>                       emptied;  // the empty candidates
    std::vector<int32_t>                       carrier;  // [TAG_MTP_QSA_PHASE] the candidates at index < 0 (no block)

    region.reserve(cand.size());

    for (const int32_t j : cand) {
        if (cells.is_empty(j)) {
            emptied.push_back(j);
            continue;
        }

        const llama_pos p  = cells.pos_get(j) - (llama_pos) phase;

        if (p < 0) {
            carrier.push_back(j);
            continue;
        }

        const int64_t   pb = p/r;

        if (pb < lo_pb || pb >= n_blocks) {
            // below the region: a mutator left no mark (cannot happen with the marks in place);
            // past the window: oor, the full census handles (excludes) it [TAG_QSA_WINDOW_MISALIGNED]
            return LLAMA_QSA_CEN_FB_SHAPE;
        }

        region.emplace_back(p, j);
    }

    std::sort(region.begin(), region.end());

    // group the region by position bucket; a repeated position is a dup census -> the full path
    struct grp_t {
        int64_t  pb;
        uint64_t mask;
        int32_t  first;     // the lowest cell index (the full census scans j ascending)
        size_t   i0, i1;    // [i0, i1) in region
    };

    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    std::vector<grp_t> grps;

    for (size_t i = 0; i < region.size(); ) {
        const int64_t pb = region[i].first/r;

        grp_t gr = { pb, 0, INT32_MAX, i, i };

        for (; i < region.size() && region[i].first/r == pb; ++i) {
            const uint64_t bit = uint64_t(1) << (region[i].first % r);
            if (gr.mask & bit) {
                return LLAMA_QSA_CEN_FB_SHAPE;
            }
            gr.mask |= bit;
            gr.first = std::min(gr.first, region[i].second);
        }

        gr.i1 = i;
        grps.push_back(gr);
    }

    //
    // commit: no failure past this point
    //

    const uint32_t n_rows  = st.n_rows;
    const int64_t  stride  = st.n_slots;

    ws.fast      = true;
    ws.region_b0 = region_b0;
    ws.old_region.clear();

    for (int64_t b = region_b0; b < n_bid_old; ++b) {
        ws.old_region.emplace_back(C.remap[b] - (int32_t) (g*n_rows), C.blk_cells[b*r]);
    }

    auto & cs = C.cs;

    cs.blk_of.resize(n_kv, -1);
    for (const int32_t j : cand) {
        cs.blk_of[j] = -1;
    }

    cs.bid_idx .resize(region_b0);
    cs.bid_cell.resize(region_b0);

    for (const auto & gr : grps) {
        if (gr.mask != slots_full) {
            continue;
        }

        const int32_t b = (int32_t) cs.bid_idx.size();

        cs.bid_idx .push_back((int32_t) (gr.pb*r));
        cs.bid_cell.push_back(gr.first);

        for (size_t i = gr.i0; i < gr.i1; ++i) {
            cs.blk_of[region[i].second] = b;
        }
    }

    const int32_t n_bid_new = (int32_t) cs.bid_idx.size();
    const int32_t max_bid   = std::max(n_bid_old, n_bid_new);

    GGML_ASSERT(n_bid_new <= n_blocks);

    // blk_cells: zero the re-grouped and vanished ids, write the new blocks
    C.blk_cells.resize(r*n_blocks, 0);
    std::fill(C.blk_cells.begin() + r*region_b0, C.blk_cells.begin() + r*max_bid, 0);

    // blk_pos: stride n_slots, so an n_kv growth moves nothing
    for (int64_t sec = 0; sec < 4; ++sec) {
        std::fill(C.blk_pos.begin() + sec*stride + region_b0, C.blk_pos.begin() + sec*stride + max_bid, 0);
    }

    std::vector<int32_t> unpooled = carrier;

    for (const auto & [p, j] : region) {
        const int32_t b = cs.blk_of[j];

        if (b >= 0) {
            C.blk_cells[b*r + (p % r)] = j;
        } else {
            unpooled.push_back(j);
        }
    }

    for (int32_t b = region_b0; b < n_bid_new; ++b) {
        for (int64_t sec = 0; sec < 4; ++sec) {
            C.blk_pos[sec*stride + b] = cs.bid_idx[b] + (int32_t) phase;
        }
    }

    // cell_blk: the candidates; every unpooled / empty cell when the dead id moves
    const int32_t dead_old = n_bid_old < C.n_blocks ? n_bid_old : (int32_t) C.n_blocks - 1;
    const int32_t dead_new = n_bid_new < n_blocks   ? n_bid_new : (int32_t) n_blocks   - 1;

    C.cell_blk.resize(n_kv);

    for (const int32_t j : cand) {
        C.cell_blk[j] = cs.blk_of[j] >= 0 ? cs.blk_of[j] : dead_new;
    }

    for (const int32_t j : emptied) {
        C.empty.push_back(j);
    }

    if (dead_new != dead_old) {
        // compact the empty list (written cells leave it lazily) and move every empty cell to the new dead id
        if (++C.seen_epoch == 0) {
            std::fill(C.seen.begin(), C.seen.end(), 0);
            C.seen_epoch = 1;
        }
        const uint32_t ep2 = C.seen_epoch;

        size_t n = 0;
        for (size_t i = 0; i < C.empty.size(); ++i) {
            const int32_t j = C.empty[i];
            if (C.seen[j] == ep2 || !cells.is_empty(j)) {
                continue;
            }
            C.seen[j] = ep2;
            C.empty[n++] = j;
            C.cell_blk[j] = dead_new;
        }
        C.empty.resize(n);
    }

    // [TAG_QSA_INPUTS_INC] the cell_blk entries written above (the fragment routing below rewrites only unpooled cells,
    // which are candidates): the candidates, and the empty list when the dead id moved
    if (st.mir_on && !C.mir_cb_all) {
        C.mir_cb.insert(C.mir_cb.end(), cand.begin(), cand.end());
        if (dead_new != dead_old) {
            C.mir_cb.insert(C.mir_cb.end(), C.empty.begin(), C.empty.end());
        }
        if (C.mir_cb.size() > std::max<size_t>(1024, st.kv_size/8)) {
            C.mir_cb_all = true;
            C.mir_cb.clear();
        }
    }
    if (st.mir_on) {
        // block ids below the region keep bid_idx, bid_cell, remap and their side of n_bid
        C.mir_rm_lo = std::min<int64_t>(C.mir_rm_lo, region_b0);
        C.mir_bt_lo = std::min<int64_t>(C.mir_bt_lo, region_b0);
    }

    // [TAG_QSA_WHOLE_BLOCKS] the fragment routing of the full census, over the unpooled cells (all in the region)
    if (llama_qsa_whole_blocks() && n_bid_new < n_blocks) {
        int64_t n_unrouted = 0;

        for (const int32_t j : unpooled) {
            const int64_t idx = (int64_t) cells.pos_get(j) - phase;

            bool tail_of_some = false;
            const auto & set = cells.seq_set((uint32_t) j);
            for (int32_t sq = 0; sq < LLAMA_MAX_SEQ && !tail_of_some; ++sq) {
                tail_of_some = set.test(sq) &&
                    llama_qsa_idx_bucket(idx, r) == llama_qsa_idx_bucket((int64_t) cells.seq_pos_max(sq) - phase, r);
            }

            if (tail_of_some) {
                continue;
            }

            if (n_bid_new + 1 < n_blocks) {
                C.cell_blk[j] = n_bid_new + 1;
            } else {
                n_unrouted++;
            }
        }

        if (n_unrouted > 0) {
            static std::atomic<bool> warned { false };
            if (!warned.exchange(true)) {
                LLAMA_LOG_WARN("%s: LLAMA_QSA_WHOLE_BLOCKS=1: %" PRId64 " fragment cell(s) keep the tail boost: no second "
                               "unused block id (n_bid %d + 1 == n_blocks %" PRId64 "); warned once\n",
                               __func__, n_unrouted, n_bid_new, n_blocks);
            }
        }
    }

    // [TAG_MTP_QSA_PHASE] the carrier's kill id, as the full census routes it (every carrier cell is a candidate)
    if (n_bid_new + 1 < n_blocks) {
        for (const int32_t j : carrier) {
            C.cell_blk[j] = n_bid_new + 1;
        }
    }

    // the stamps of the new blocks
    const auto & gen = st.strm[g].gen;

    C.stamps.resize(region_b0);
    for (int32_t b = region_b0; b < n_bid_new; ++b) {
        C.stamps.push_back(qsa_block_stamp(cells, gen, g, r, false, C.blk_cells.data() + b*r, C.blk_pos.data(), stride, b));
    }

    int64_t tail_pb = INT64_MAX;
    for (const int32_t j : unpooled) {
        // [TAG_MTP_QSA_PHASE] the carrier (index < 0) is unpooled for good: it bounds no region
        if ((int64_t) cells.pos_get(j) >= phase) {
            tail_pb = std::min<int64_t>(tail_pb, ((int64_t) cells.pos_get(j) - phase)/r);
        }
    }
    if (tail_pb == INT64_MAX) {
        tail_pb = n_bid_new > 0 ? cs.bid_idx[n_bid_new - 1]/r + 1 : 0;
    }

    cs.n_bid  = n_bid_new;
    cs.ranked = false;

    C.unpooled   = std::move(unpooled);
    C.tail_pb    = tail_pb;
    C.n_kv       = n_kv;
    C.n_blocks   = n_blocks;
    C.inc_lo_pos = INT32_MAX;
    C.written.clear();

    return LLAMA_QSA_CEN_FAST;
}

int64_t llama_qsa_plan_inc(llama_qsa_pooled_store & st, const llama_qsa_plan_in & in, llama_qsa_plan & pl) {
    GGML_ASSERT(st.inc);

    const uint32_t ratio    = st.ratio;
    const int64_t  r        = ratio;
    const int64_t  n_kv     = in.n_kv;
    const int64_t  n_blocks = (n_kv + r - 1)/r;
    const bool     blk_bias = in.blk_bias;
    const int64_t  n_ns     = in.n_ns;
    const int64_t  stride   = st.n_slots;   // blk_pos stride of the census state

    GGML_ASSERT(n_blocks <= (int64_t) st.n_slots);

    pl.valid    = false;
    pl.inc      = true;
    pl.ratio    = ratio;
    pl.n_kv     = n_kv;
    pl.n_blocks = n_blocks;
    pl.n_ns     = n_ns;
    pl.blk_bias = blk_bias;

    pl.cell_blk .clear();
    pl.blk_cells.clear();
    pl.blk_pos  .clear();
    pl.cs       .clear();
    pl.remap    .clear();

    std::vector<uint32_t>            strm_of(n_ns);
    std::vector<qsa_inc_stream_work> work(n_ns);

    size_t   need        = 1;
    uint32_t n_dirty_all = 0;
    uint32_t n_bid_all   = 0;
    bool     all_fast    = true;

    // pass 1: the census (fast or full) into the stream's census state, the stamps, the slots blocks own, the dirty count
    for (int64_t s = 0; s < n_ns; ++s) {
        const auto &   cells = *in.cells[s];
        const uint32_t g     = in.g[s];

        GGML_ASSERT(g < st.n_stream);
        GGML_ASSERT((s == 0 || g == strm_of[0] + s) && "qsa: the ubatch streams are one contiguous range (get_k)");

        strm_of[s] = g;

        auto & ss = st.strm[g];
        auto & C  = ss.cen;
        auto & ws = work[s];

        const int why = llama_qsa_census_inc(st, g, cells, n_kv, n_blocks, r, ws);

        if (why == LLAMA_QSA_CEN_FAST) {
            st.stats.n_cen_inc++;
        } else {
            all_fast = false;
            st.stats.n_cen_full++;
            st.stats.n_cen_fb[why]++;

            ws = qsa_inc_stream_work();

            C.cell_blk .resize(n_kv);
            C.blk_cells.resize(r*n_blocks);
            C.blk_pos  .assign(4*stride, 0);

            llama_qsa_census_flags fl;
            llama_qsa_census_full(cells, in.is_pos_2d, n_kv, n_blocks, r, blk_bias, C.cell_blk.data(), C.blk_cells.data(),
                                  C.blk_pos.data(), stride, C.cs, &fl, st.phase);

            C.n_kv     = n_kv;
            C.n_blocks = n_blocks;
            C.inc_ok   = fl.one_seq && !C.cs.ranked && !fl.dup && !fl.oor;

            C.unpooled.clear();
            C.empty   .clear();
            C.tail_pb = 0;

            if (C.inc_ok) {
                int64_t tail_pb = INT64_MAX;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j)) {
                        C.empty.push_back((int32_t) j);
                    } else if (C.cs.blk_of[j] < 0) {
                        C.unpooled.push_back((int32_t) j);
                        // [TAG_MTP_QSA_PHASE] the carrier (index < 0) is unpooled for good: it bounds no region
                        if ((int64_t) cells.pos_get(j) >= st.phase) {
                            tail_pb = std::min<int64_t>(tail_pb, ((int64_t) cells.pos_get(j) - st.phase)/r);
                        }
                    }
                }

                if (tail_pb == INT64_MAX) {
                    tail_pb = C.cs.n_bid > 0 ? C.cs.bid_idx[C.cs.n_bid - 1]/r + 1 : 0;
                }

                C.tail_pb = tail_pb;
            }

            C.stamps.resize(C.cs.n_bid);
            for (int32_t b = 0; b < C.cs.n_bid; ++b) {
                C.stamps[b] = qsa_block_stamp(cells, ss.gen, g, r, C.cs.ranked, C.blk_cells.data() + b*r, C.blk_pos.data(),
                                              stride, b);
            }

            C.inc_all    = false;
            C.inc_lo_pos = INT32_MAX;
            C.written.clear();
            C.carry  .clear();

            // [TAG_QSA_INPUTS_INC] every array may have changed
            if (st.mir_on) {
                C.mir_cb_all = true;
                C.mir_cb.clear();
                C.mir_rm_lo  = 0;
                C.mir_bt_lo  = 0;
            }
        }

        const int32_t n_bid = C.cs.n_bid;

        // the slots the (re-grouped) blocks own already, and the dirty count
        size_t n_dirty = 0;

        ws.slot_of.assign(n_bid - ws.region_b0, -1);

        for (int32_t b = ws.region_b0; b < n_bid; ++b) {
            const int32_t key  = C.blk_cells[(int64_t) b*r];
            const int32_t slot = ss.slot_of_key[key];

            if (slot >= 0 && ss.slot_key[slot] == key) {
                ws.slot_of[b - ws.region_b0] = slot;
            }

            n_dirty += ws.slot_of[b - ws.region_b0] < 0 || ss.slot_stamp[ws.slot_of[b - ws.region_b0]] != C.stamps[b];
        }

        if (ws.fast) {
            // a kept block is dirty only if the last plan left its stamp uncommitted
            for (const int32_t b : C.carry) {
                if (b >= ws.region_b0) {
                    break;
                }

                const int32_t slot = C.remap[b] - (int32_t) (g*st.n_rows);

                if (ss.slot_stamp[slot] != C.stamps[b]) {
                    ws.carry_dirty.push_back(b);
                }
            }

            n_dirty += ws.carry_dirty.size();
        }

        need = std::max(need, n_dirty + (n_bid < n_blocks ? 1 : 0));

        n_dirty_all += (uint32_t) n_dirty;
        n_bid_all   += (uint32_t) n_bid;
    }

    const int64_t cap = qsa_store_cap((int64_t) need, n_blocks);
    GGML_ASSERT(cap >= (int64_t) need);

    const bool full_pool = cap == n_blocks;

    pl.cap  = cap;
    pl.full = full_pool;
    pl.s0   = strm_of[0];

    const uint32_t serial = ++st.serial;

    std::vector<std::vector<std::pair<int32_t, int32_t>>> dirty(n_ns);

    // pass 2: the slot assignment, the dirty list, the pending stamps (the full path's order)
    for (int64_t s = 0; s < n_ns; ++s) {
        const uint32_t g = strm_of[s];

        auto & ss = st.strm[g];
        auto & C  = ss.cen;
        auto & ws = work[s];

        const int32_t n_bid = C.cs.n_bid;
        const int32_t row0  = (int32_t) (g*st.n_rows);

        C.remap.resize(n_blocks);

        // [TAG_QSA_INPUTS_INC] the full pool renumbers every slot; the fast path changes ids >= region_b0 (recorded by the
        // census), the full census every id (recorded above)
        if (full_pool && st.mir_on) {
            C.mir_rm_lo = 0;
        }

        if (full_pool) {
            for (uint32_t slot = 0; slot < st.n_slots; ++slot) {
                if (ss.slot_key[slot] >= 0) {
                    ss.slot_of_key[ss.slot_key[slot]] = -1;
                }
                ss.slot_key[slot]   = -1;
                ss.slot_stamp[slot] = 0;
            }

            ss.free_slots.clear();
            for (int32_t slot = (int32_t) st.n_slots - 1; slot >= n_bid; --slot) {
                ss.free_slots.push_back(slot);
            }

            for (int32_t b = 0; b < n_bid; ++b) {
                const int32_t key = C.blk_cells[(int64_t) b*r];

                ss.slot_key[b]      = key;
                ss.slot_of_key[key] = b;
                ss.slot_seen[b]     = serial;

                C.remap[b] = row0 + b;

                dirty[s].emplace_back(b, b);
                st.pending.push_back({ g, b, C.stamps[b] });
            }
        } else if (!ws.fast) {
            // the full path's pass 2
            for (int32_t b = 0; b < n_bid; ++b) {
                if (ws.slot_of[b] >= 0) {
                    ss.slot_seen[ws.slot_of[b]] = serial;
                }
            }

            for (uint32_t slot = 0; slot < st.n_slots; ++slot) {
                const int32_t key = ss.slot_key[slot];

                if (key >= 0 && ss.slot_seen[slot] != serial) {
                    ss.slot_of_key[key] = -1;
                    ss.slot_key[slot]   = -1;
                    ss.slot_stamp[slot] = 0;
                    ss.free_slots.push_back((int32_t) slot);
                }
            }

            for (int32_t b = 0; b < n_bid; ++b) {
                int32_t slot = ws.slot_of[b];

                if (slot < 0) {
                    GGML_ASSERT(!ss.free_slots.empty() && "qsa: more complete blocks than block slots");

                    slot = ss.free_slots.back();
                    ss.free_slots.pop_back();

                    const int32_t key = C.blk_cells[(int64_t) b*r];

                    ss.slot_key[slot]   = key;
                    ss.slot_of_key[key] = slot;
                    ss.slot_stamp[slot] = 0;
                    ss.slot_seen[slot]  = serial;
                }

                C.remap[b] = row0 + slot;

                if (ss.slot_stamp[slot] != C.stamps[b]) {
                    ss.slot_stamp[slot] = 0;

                    dirty[s].emplace_back(slot, b);

                    st.pending.push_back({ g, slot, C.stamps[b] });
                }
            }
        } else {
            // fast: only the re-grouped region changes owners. The slots owned before this plan are exactly the last
            // census's blocks (a reset forces the full census), so the vanished ones are the old region blocks whose key
            // cell no new block has - freed in ascending slot order, as the full path's scan frees them
            std::vector<int32_t> kept;
            for (const int32_t slot : ws.slot_of) {
                if (slot >= 0) {
                    kept.push_back(slot);
                }
            }
            std::sort(kept.begin(), kept.end());

            std::vector<int32_t> vanished;
            for (const auto & [slot, key] : ws.old_region) {
                GGML_ASSERT(slot >= 0 && ss.slot_key[slot] == key && "qsa: census state out of step with the slots");

                if (!std::binary_search(kept.begin(), kept.end(), slot)) {
                    vanished.push_back(slot);
                }
            }
            std::sort(vanished.begin(), vanished.end());

            for (const int32_t slot : vanished) {
                ss.slot_of_key[ss.slot_key[slot]] = -1;
                ss.slot_key[slot]   = -1;
                ss.slot_stamp[slot] = 0;
                ss.free_slots.push_back(slot);
            }

            // the kept blocks with an uncommitted stamp (ids ascending, all below the region)
            for (const int32_t b : ws.carry_dirty) {
                const int32_t slot = C.remap[b] - row0;

                ss.slot_stamp[slot] = 0;

                dirty[s].emplace_back(slot, b);

                st.pending.push_back({ g, slot, C.stamps[b] });
            }

            for (int32_t b = ws.region_b0; b < n_bid; ++b) {
                int32_t slot = ws.slot_of[b - ws.region_b0];

                if (slot < 0) {
                    GGML_ASSERT(!ss.free_slots.empty() && "qsa: more complete blocks than block slots");

                    slot = ss.free_slots.back();
                    ss.free_slots.pop_back();

                    const int32_t key = C.blk_cells[(int64_t) b*r];

                    ss.slot_key[slot]   = key;
                    ss.slot_of_key[key] = slot;
                    ss.slot_stamp[slot] = 0;
                    ss.slot_seen[slot]  = serial;
                }

                C.remap[b] = row0 + slot;

                if (ss.slot_stamp[slot] != C.stamps[b]) {
                    ss.slot_stamp[slot] = 0;

                    dirty[s].emplace_back(slot, b);

                    st.pending.push_back({ g, slot, C.stamps[b] });
                }
            }
        }

        for (int64_t b = n_bid; b < n_blocks; ++b) {
            C.remap[b] = row0 + (int32_t) st.n_slots;
        }

        C.carry.clear();
        for (const auto & d : dirty[s]) {
            C.carry.push_back(d.second);
        }

        C.valid = true;

        st.pending_streams.push_back(g);
    }

    // the dirty list (the full path's fill)
    pl.dirty_cells.assign(r*cap*n_ns, 0);
    pl.dirty_pos  .assign(4*cap*n_ns, 0);
    pl.dirty_dst  .resize(cap*n_ns);

    for (int64_t s = 0; s < n_ns; ++s) {
        const uint32_t g = strm_of[s];
        const auto &   C = st.strm[g].cen;

        for (int64_t e = 0; e < cap; ++e) {
            if (e < (int64_t) dirty[s].size()) {
                const int32_t slot = dirty[s][e].first;
                const int32_t b    = dirty[s][e].second;

                for (int64_t k = 0; k < r; ++k) {
                    pl.dirty_cells[s*(r*cap) + e*r + k] = C.blk_cells[b*r + k];
                }

                for (int64_t sec = 0; sec < 4; ++sec) {
                    pl.dirty_pos[sec*(cap*n_ns) + s*cap + e] = C.blk_pos[sec*stride + b];
                }

                pl.dirty_dst[s*cap + e] = (int32_t) (g*st.n_rows + slot);
            } else {
                pl.dirty_dst[s*cap + e] = (int32_t) (g*st.n_rows + st.n_slots);
            }
        }
    }

    // the views: the stream's census state
    pl.v_cell_blk .resize(n_ns);
    pl.v_blk_cells.resize(n_ns);
    pl.v_remap    .resize(n_ns);
    pl.v_blk_pos  .resize(n_ns);
    pl.v_cs       .resize(n_ns);

    pl.v_pos_stride = stride;

    for (int64_t s = 0; s < n_ns; ++s) {
        const auto & C = st.strm[strm_of[s]].cen;

        pl.v_cell_blk [s] = C.cell_blk .data();
        pl.v_blk_cells[s] = C.blk_cells.data();
        pl.v_remap    [s] = C.remap    .data();
        pl.v_blk_pos  [s] = C.blk_pos  .data();
        pl.v_cs       [s] = &C.cs;
    }

    pl.valid  = true;
    pl.serial = serial;

    st.stats.n_plan++;
    st.stats.n_plan_inc  += all_fast;
    st.stats.n_full      += full_pool;
    st.stats.n_dirty     += full_pool ? n_bid_all : n_dirty_all;
    st.stats.n_valid     += full_pool ? 0 : n_bid_all - n_dirty_all;
    st.stats.last_dirty   = full_pool ? n_bid_all : n_dirty_all;
    st.stats.last_cap     = (uint32_t) cap;
    st.stats.last_n_bid   = n_bid_all;

    return cap;
}

// [TAG_QSA_INPUTS_INC] see the header
void llama_qsa_mirror_update(llama_qsa_pooled_store & st, const llama_qsa_plan & pl,
                             const std::vector<const llama_kv_cells *> & cells, bool tail_fix, bool want_remap,
                             bool want_btab, llama_qsa_mirror_writer & w) {
    using store_t = llama_qsa_pooled_store;

    GGML_ASSERT(st.mir_on && st.inc && pl.valid && pl.inc && pl.serial == st.serial);
    GGML_ASSERT((int64_t) cells.size() == pl.n_ns);

    const int64_t n_ns     = pl.n_ns;
    const int64_t n_kv     = pl.n_kv;
    const int64_t n_blocks = pl.n_blocks;

    // runs closer than this many elements are written as one
    const int64_t gap = 32;

    auto write_full = [&](int arr, int64_t off, const int32_t * src, int64_t n) {
        w.write(arr, off, src, n);
        st.stats.mir_bytes += (uint64_t) n*sizeof(int32_t);
        st.stats.mir_runs  += 1;
        st.stats.mir_full  += 1;
    };

    auto write_run = [&](int arr, int64_t off, const int32_t * src, int64_t n) {
        w.write(arr, off, src, n);
        st.stats.mir_bytes += (uint64_t) n*sizeof(int32_t);
        st.stats.mir_runs  += 1;
    };

    auto layout_ok = [&](int arr, int64_t n) {
        const auto & L = st.mir_layout[arr];
        return L.valid && L.s0 == pl.s0 && L.n_ns == n_ns && (n_ns == 1 ? n >= L.n : n == L.n);
    };

    auto set_layout = [&](int arr, int64_t n) {
        auto & L = st.mir_layout[arr];
        L.valid = true;
        L.s0    = pl.s0;
        L.n_ns  = n_ns;
        L.n     = n;
    };

    // cell_blk: every graph reads it
    {
        const bool ok = layout_ok(store_t::MIR_CELL_BLK, n_kv);

        for (int64_t s = 0; s < n_ns; ++s) {
            auto & C = st.strm[pl.s0 + s].cen;

            GGML_ASSERT((int64_t) C.cell_blk.size() >= n_kv);

            if (!ok || C.mir_cb_all) {
                write_full(store_t::MIR_CELL_BLK, s*n_kv, C.cell_blk.data(), n_kv);
            } else {
                st.stats.mir_inc++;

                auto & v = C.mir_cb;
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());

                size_t i = 0;
                while (i < v.size() && v[i] < n_kv) {
                    const int64_t lo = v[i];
                    int64_t       hi = lo + 1;
                    for (++i; i < v.size() && v[i] < n_kv && v[i] <= hi + gap; ++i) {
                        hi = v[i] + 1;
                    }
                    write_run(store_t::MIR_CELL_BLK, s*n_kv + lo, C.cell_blk.data() + lo, hi - lo);
                }
            }

            C.mir_cb_all = false;
            C.mir_cb.clear();
        }

        set_layout(store_t::MIR_CELL_BLK, n_kv);
    }

    // remap: only a dirty-list graph has one
    if (want_remap) {
        GGML_ASSERT(!pl.full);

        const bool ok = layout_ok(store_t::MIR_REMAP, n_blocks);

        for (int64_t s = 0; s < n_ns; ++s) {
            auto & C = st.strm[pl.s0 + s].cen;

            GGML_ASSERT((int64_t) C.remap.size() >= n_blocks);

            if (!ok || C.mir_rm_lo <= 0) {
                write_full(store_t::MIR_REMAP, s*n_blocks, C.remap.data(), n_blocks);
            } else {
                st.stats.mir_inc++;

                const int64_t lo = std::min<int64_t>(C.mir_rm_lo, n_blocks);
                if (lo < n_blocks) {
                    write_run(store_t::MIR_REMAP, s*n_blocks + lo, C.remap.data() + lo, n_blocks - lo);
                }
            }

            C.mir_rm_lo = INT64_MAX;
        }

        set_layout(store_t::MIR_REMAP, n_blocks);
    }

    // the compact bias block table: refilled from the census from the lowest changed id up, then written
    if (want_btab) {
        const bool ok = layout_ok(store_t::MIR_BTAB, 2*n_blocks);

        for (int64_t s = 0; s < n_ns; ++s) {
            auto & C = st.strm[pl.s0 + s].cen;

            const int64_t lo = std::min<int64_t>(std::max<int64_t>(C.mir_bt_lo, 0), n_blocks);

            C.btab.resize(2*n_blocks);
            llama_qsa_fill_blk_tab(*cells[s], C.cs, n_blocks, lo, n_blocks, tail_fix, C.btab.data());

            if (!ok || lo == 0) {
                write_full(store_t::MIR_BTAB, s*2*n_blocks, C.btab.data(), 2*n_blocks);
            } else {
                st.stats.mir_inc++;

                if (lo < n_blocks) {
                    write_run(store_t::MIR_BTAB, s*2*n_blocks + 2*lo, C.btab.data() + 2*lo, 2*(n_blocks - lo));
                }
            }

            C.mir_bt_lo = INT64_MAX;
        }

        set_layout(store_t::MIR_BTAB, 2*n_blocks);
    }
}

void llama_qsa_store_save(const llama_qsa_pooled_store & st, llama_qsa_store_snapshot & snap) {
    snap.strm            = st.strm;
    snap.serial          = st.serial;
    snap.pending         = st.pending;
    snap.pending_streams = st.pending_streams;
    snap.pending_armed   = st.pending_armed;
    snap.stats           = st.stats;
}

void llama_qsa_store_restore(llama_qsa_pooled_store & st, const llama_qsa_store_snapshot & snap) {
    st.strm            = snap.strm;
    st.serial          = snap.serial;
    st.pending         = snap.pending;
    st.pending_streams = snap.pending_streams;
    st.pending_armed   = snap.pending_armed;
    st.stats           = snap.stats;
}

template <typename T>
static bool qsa_cmp_arr(std::string & out, const char * what, int64_t s, const T * a, const T * b, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        if (memcmp(&a[i], &b[i], sizeof(T)) != 0) {
            out = std::string(what) + " s=" + std::to_string(s) + " i=" + std::to_string(i) +
                  " full=" + std::to_string(a[i]) + " inc=" + std::to_string(b[i]);
            return false;
        }
    }
    return true;
}

template <typename T>
static bool qsa_cmp_vec(std::string & out, const char * what, int64_t s, const std::vector<T> & a, const std::vector<T> & b) {
    if (a.size() != b.size()) {
        out = std::string(what) + " s=" + std::to_string(s) + " size full=" + std::to_string(a.size()) +
              " inc=" + std::to_string(b.size());
        return false;
    }
    return qsa_cmp_arr(out, what, s, a.data(), b.data(), (int64_t) a.size());
}

std::string llama_qsa_plan_compare(const llama_qsa_plan & ref, const std::vector<std::vector<uint64_t>> & ref_stamps,
                                   const llama_qsa_pooled_store & st_ref, const llama_qsa_plan & inc,
                                   const llama_qsa_pooled_store & st_inc, uint32_t ratio,
                                   const std::vector<std::vector<uint64_t>> * inc_stamps) {
    std::string out;

    const int64_t r = ratio;

#define QSA_CMP_SCALAR(name, a, b) \
    if ((a) != (b)) { return std::string(name) + " full=" + std::to_string(a) + " inc=" + std::to_string(b); }

    QSA_CMP_SCALAR("valid",    ref.valid,    inc.valid);
    QSA_CMP_SCALAR("cap",      ref.cap,      inc.cap);
    QSA_CMP_SCALAR("full",     ref.full,     inc.full);
    QSA_CMP_SCALAR("s0",       ref.s0,       inc.s0);
    QSA_CMP_SCALAR("n_kv",     ref.n_kv,     inc.n_kv);
    QSA_CMP_SCALAR("n_blocks", ref.n_blocks, inc.n_blocks);
    QSA_CMP_SCALAR("n_ns",     ref.n_ns,     inc.n_ns);

    const int64_t n_kv     = ref.n_kv;
    const int64_t n_blocks = ref.n_blocks;

    for (int64_t s = 0; s < ref.n_ns; ++s) {
        if (!qsa_cmp_arr(out, "cell_blk",  s, ref.v_cell_blk [s], inc.v_cell_blk [s], n_kv))       { return out; }
        if (!qsa_cmp_arr(out, "blk_cells", s, ref.v_blk_cells[s], inc.v_blk_cells[s], r*n_blocks)) { return out; }
        for (int64_t sec = 0; sec < 4; ++sec) {
            if (!qsa_cmp_arr(out, sec == 0 ? "blk_pos[0]" : sec == 1 ? "blk_pos[1]" : sec == 2 ? "blk_pos[2]" : "blk_pos[3]", s,
                             ref.v_blk_pos[s] + sec*ref.v_pos_stride, inc.v_blk_pos[s] + sec*inc.v_pos_stride, n_blocks)) {
                return out;
            }
        }
        if (!qsa_cmp_arr(out, "remap", s, ref.v_remap[s], inc.v_remap[s], n_blocks)) { return out; }

        const auto & a = *ref.v_cs[s];
        const auto & b = *inc.v_cs[s];

        QSA_CMP_SCALAR("cs.n_bid",  a.n_bid,  b.n_bid);
        QSA_CMP_SCALAR("cs.ranked", a.ranked, b.ranked);
        if (!qsa_cmp_arr(out, "cs.bid_idx",  s, a.bid_idx .data(), b.bid_idx .data(), a.n_bid)) { return out; }
        if (!qsa_cmp_arr(out, "cs.bid_cell", s, a.bid_cell.data(), b.bid_cell.data(), a.n_bid)) { return out; }
        if ((int64_t) b.blk_of.size() < n_kv) {
            return "cs.blk_of s=" + std::to_string(s) + " inc size " + std::to_string(b.blk_of.size());
        }
        if (!qsa_cmp_arr(out, "cs.blk_of", s, a.blk_of.data(), b.blk_of.data(), n_kv)) { return out; }
        if (a.ranked) {
            if (!qsa_cmp_vec(out, "cs.order", s, a.order, b.order)) { return out; }
            if (!qsa_cmp_vec(out, "cs.rank",  s, a.rank,  b.rank))  { return out; }
        }

        // every block's stamp
        const uint32_t g = ref.s0 + (uint32_t) s;
        if (!qsa_cmp_vec(out, "stamps", s, ref_stamps[s], inc.inc ? st_inc.strm[g].cen.stamps : (*inc_stamps)[s])) { return out; }
    }

    if (!qsa_cmp_vec(out, "dirty_cells", -1, ref.dirty_cells, inc.dirty_cells)) { return out; }
    if (!qsa_cmp_vec(out, "dirty_pos",   -1, ref.dirty_pos,   inc.dirty_pos))   { return out; }
    if (!qsa_cmp_vec(out, "dirty_dst",   -1, ref.dirty_dst,   inc.dirty_dst))   { return out; }

    // the store: the planned streams' slot state, the pending stamps
    if (!qsa_cmp_vec(out, "pending_streams", -1, st_ref.pending_streams, st_inc.pending_streams)) { return out; }

    for (const uint32_t g : st_ref.pending_streams) {
        const auto & a = st_ref.strm[g];
        const auto & b = st_inc.strm[g];

        if (!qsa_cmp_vec(out, "slot_key",    g, a.slot_key,    b.slot_key))    { return out; }
        if (!qsa_cmp_vec(out, "slot_of_key", g, a.slot_of_key, b.slot_of_key)) { return out; }
        if (!qsa_cmp_vec(out, "slot_stamp",  g, a.slot_stamp,  b.slot_stamp))  { return out; }
        if (!qsa_cmp_vec(out, "free_slots",  g, a.free_slots,  b.free_slots))  { return out; }
    }

    QSA_CMP_SCALAR("pending.size", st_ref.pending.size(), st_inc.pending.size());
    for (size_t i = 0; i < st_ref.pending.size(); ++i) {
        const auto & a = st_ref.pending[i];
        const auto & b = st_inc.pending[i];
        if (a.s != b.s || a.slot != b.slot || a.stamp != b.stamp) {
            return "pending i=" + std::to_string(i) + " full=(" + std::to_string(a.s) + "," + std::to_string(a.slot) + "," +
                   std::to_string(a.stamp) + ") inc=(" + std::to_string(b.s) + "," + std::to_string(b.slot) + "," +
                   std::to_string(b.stamp) + ")";
        }
    }

    QSA_CMP_SCALAR("serial", st_ref.serial, st_inc.serial);

#undef QSA_CMP_SCALAR

    return "";
}

std::string llama_qsa_census_state_check(const llama_qsa_pooled_store & st, uint32_t g, const llama_kv_cells & cells) {
    const auto & C = st.strm[g].cen;

    if (!C.valid || !C.inc_ok) {
        return "";
    }

    const int64_t r = st.ratio;

    std::vector<uint8_t> in_unpooled(C.n_kv, 0);
    std::vector<uint8_t> in_empty   (C.n_kv, 0);

    for (const int32_t j : C.unpooled) {
        if (j < 0 || j >= C.n_kv || in_unpooled[j]) {
            return "census state g=" + std::to_string(g) + ": unpooled list holds cell " + std::to_string(j) + " twice / out of range";
        }
        in_unpooled[j] = 1;
    }
    for (const int32_t j : C.empty) {
        if (j >= 0 && j < C.n_kv) {
            in_empty[j] = 1;
        }
    }

    for (int64_t j = 0; j < C.n_kv; ++j) {
        if (cells.is_empty(j)) {
            if (!in_empty[j]) {
                return "census state g=" + std::to_string(g) + ": empty cell " + std::to_string(j) + " not in the empty list";
            }
            if (in_unpooled[j]) {
                return "census state g=" + std::to_string(g) + ": empty cell " + std::to_string(j) + " in the unpooled list";
            }
            continue;
        }

        const bool unp = C.cs.blk_of[j] < 0;

        if (unp != (in_unpooled[j] != 0)) {
            return "census state g=" + std::to_string(g) + ": cell " + std::to_string(j) + " unpooled=" + std::to_string(unp) +
                   " but list=" + std::to_string(in_unpooled[j]);
        }
        if (unp && (int64_t) cells.pos_get(j) >= st.phase && ((int64_t) cells.pos_get(j) - st.phase)/r < C.tail_pb) {
            return "census state g=" + std::to_string(g) + ": unpooled cell " + std::to_string(j) + " below tail_pb " +
                   std::to_string(C.tail_pb);
        }
    }

    return "";
}

int64_t llama_memory_hybrid_idx_context::qsa_store_plan(const llama_ubatch & ubatch, uint32_t ratio, int64_t n_kv, bool blk_bias) const {
    llama_qsa_pooled_store * st = get_qsa_store();
    GGML_ASSERT(st != nullptr && ratio == st->ratio);

    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;

    if (full) {
        // the reserve graphs: every block may be dirty at once (the first selection after the exhaustive
        // bypass, a restore, a shift), so size for the cold rebuild: the full pool
        return n_blocks;
    }

    auto & pl = qsa_plan;

    if (pl.valid && pl.i_ubatch == i_cur && pl.ratio == ratio && pl.n_kv == n_kv && pl.blk_bias == blk_bias) {
        return pl.cap;
    }

    const int64_t t_start_us = ggml_time_us();

    // a plan whose graph never reported back (an exception, an optimizer path) must not publish later
    if (!st->pending.empty() || !st->pending_streams.empty()) {
        st->publish(st->pending_armed ? false : true);
    }

    const int64_t n_ns     = get_n_stream();
    const int64_t n_tokens = ubatch.n_tokens;
    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps    = n_tokens/n_ns;

    llama_qsa_plan_in in;
    in.n_ns      = n_ns;
    in.is_pos_2d = ubatch.is_pos_2d();
    in.n_kv      = n_kv;
    in.blk_bias  = blk_bias;
    in.cells.resize(n_ns);
    in.g    .resize(n_ns);

    for (int64_t s = 0; s < n_ns; ++s) {
        const llama_seq_id seq_of_stream = ubatch.seq_id[s*n_tps][0];

        in.cells[s] = &mem->get_mem_idx()->get_cells(seq_of_stream);
        in.g    [s] = mem->qsa_stream_of(seq_of_stream);
    }

    pl.valid = false;

    if (!st->inc) {
        llama_qsa_plan_full(*st, in, pl);
    } else if (!llama_qsa_census_check()) {
        llama_qsa_plan_inc(*st, in, pl);
    } else {
        // [TAG_QSA_CENSUS_INC] LLAMA_QSA_CENSUS_CHECK=1: the full path on the store as it is, then the incremental path on
        // the same starting state; the incremental result is kept
        llama_qsa_store_snapshot before;
        llama_qsa_store_save(*st, before);

        llama_qsa_plan                     ref;
        std::vector<std::vector<uint64_t>> ref_stamps;
        llama_qsa_plan_full(*st, in, ref, &ref_stamps);

        // the full path's store, for the comparison (moved out: only the planned streams' slot state is compared)
        llama_qsa_store_snapshot after_ref;
        llama_qsa_store_save(*st, after_ref);
        llama_qsa_store_restore(*st, before);

        llama_qsa_plan_inc(*st, in, pl);

        llama_qsa_pooled_store st_ref_view;
        st_ref_view.ratio   = st->ratio;
        st_ref_view.n_slots = st->n_slots;
        st_ref_view.n_rows  = st->n_rows;
        st_ref_view.phase   = st->phase;
        llama_qsa_store_restore(st_ref_view, after_ref);

        std::string diff = llama_qsa_plan_compare(ref, ref_stamps, st_ref_view, pl, *st, ratio);
        for (int64_t s = 0; s < n_ns && diff.empty(); ++s) {
            diff = llama_qsa_census_state_check(*st, in.g[s], *in.cells[s]);
        }
        if (!diff.empty()) {
            GGML_ABORT("[qsa-census-check] MISMATCH %s (n_kv=%lld n_ns=%lld plan %llu)", diff.c_str(), (long long) n_kv,
                       (long long) n_ns, (unsigned long long) st->stats.n_plan);
        }

        static std::atomic<uint64_t> n_checked { 0 };

        const uint64_t n = ++n_checked;
        if (n % 16 == 0 || n <= 4) {
            LLAMA_LOG_WARN("[qsa-census-check] %llu plans identical (fast-path stream censuses %llu, full %llu)\n",
                           (unsigned long long) n, (unsigned long long) st->stats.n_cen_inc,
                           (unsigned long long) st->stats.n_cen_full);
        }
    }

    pl.i_ubatch = i_cur;

    st->stats.t_plan_us += ggml_time_us() - t_start_us;

    return pl.cap;
}

uint32_t llama_memory_hybrid_idx_context::qsa_store_s0() const {
    // the reserve graphs are built over every stream from 0
    return full || !qsa_plan.valid ? 0 : qsa_plan.s0;
}

void llama_memory_hybrid_idx_context::set_input_qsa_store(
        ggml_tensor * cell_blk,
        ggml_tensor * bias,
        ggml_tensor * qsa_blk,
        ggml_tensor * qsa_qry,
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_dst,
        ggml_tensor * remap,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    llama_qsa_pooled_store * st = get_qsa_store();
    GGML_ASSERT(st != nullptr);

    const int64_t n_kv = cell_blk->ne[0];

    const int64_t cap = qsa_store_plan(*ubatch, ratio, n_kv, blk_bias);

    // [TAG_QSA_INPUTS_INC] the input time (the plan above is cached: planned in can_reuse / the build)
    const int64_t t_start_us = ggml_time_us();

    const auto & pl = qsa_plan;

    GGML_ASSERT(pl.valid);
    GGML_ASSERT(cell_blk->ne[1]    == pl.n_ns);
    GGML_ASSERT(dirty_cells->ne[0] == (int64_t) ratio*cap && dirty_cells->ne[1] == pl.n_ns);
    GGML_ASSERT(dirty_pos->ne[0]   == 4*cap*pl.n_ns);
    // the full pool (cap == n_blocks) reads its own output and copies it to the slots in compact order:
    // no store row index, no remap
    GGML_ASSERT(pl.full == (cap == pl.n_blocks));
    GGML_ASSERT((dirty_dst == nullptr) == pl.full && (remap == nullptr) == pl.full);

    GGML_ASSERT(ggml_backend_buffer_is_host(dirty_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(dirty_pos->buffer));

    // [TAG_QSA_CENSUS_INC] the census arrays are read through the per-stream views (the plan's own arrays, or the store's
    // census state, which this plan wrote and no later plan has touched yet)
    GGML_ASSERT(!pl.inc || pl.serial == st->serial);
    GGML_ASSERT(ggml_nbytes(cell_blk) == (size_t) (n_kv*pl.n_ns)*sizeof(int32_t));

    // [TAG_QSA_INPUTS_INC] cell_blk / remap / qsa_blk are views of the device mirrors: write what changed (a compact
    // graph; the graph build makes the same choice, llm_graph_input_qsa)
    if (st->mir_on && bias == nullptr) {
        set_input_qsa_mirror(cell_blk, qsa_blk, qsa_qry, dirty_cells, dirty_pos, dirty_dst, remap, ubatch, ratio);
        st->stats.t_inputs_us += ggml_time_us() - t_start_us;
        return;
    }

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    for (int64_t s = 0; s < pl.n_ns; ++s) {
        memcpy((int32_t *) cell_blk->data + s*n_kv, pl.v_cell_blk[s], n_kv*sizeof(int32_t));
    }
    memcpy(dirty_cells->data, pl.dirty_cells.data(), ggml_nbytes(dirty_cells));
    memcpy(dirty_pos  ->data, pl.dirty_pos  .data(), ggml_nbytes(dirty_pos));

    if (!pl.full) {
        GGML_ASSERT(dirty_dst->ne[0] == cap*pl.n_ns && remap->ne[0] == pl.n_blocks*pl.n_ns);
        GGML_ASSERT(ggml_backend_buffer_is_host(dirty_dst->buffer) && ggml_backend_buffer_is_host(remap->buffer));

        memcpy(dirty_dst->data, pl.dirty_dst.data(), ggml_nbytes(dirty_dst));
        for (int64_t s = 0; s < pl.n_ns; ++s) {
            memcpy((int32_t *) remap->data + s*pl.n_blocks, pl.v_remap[s], pl.n_blocks*sizeof(int32_t));
        }
    }

    // [TAG_QSA_COMPACT_BIAS] exactly one of the plane and the tables
    const bool compact = bias == nullptr;
    GGML_ASSERT(compact == (qsa_blk != nullptr) && (qsa_blk != nullptr) == (qsa_qry != nullptr));
    GGML_ASSERT(!compact || (blk_bias && qsa_compact_bias()));

    if (compact) {
        GGML_ASSERT(ggml_backend_buffer_is_host(qsa_blk->buffer) && ggml_backend_buffer_is_host(qsa_qry->buffer));
        GGML_ASSERT(qsa_blk->ne[1] == pl.n_blocks && qsa_blk->ne[2] == pl.n_ns && qsa_qry->ne[1] == ubatch->n_tokens/pl.n_ns);

        std::vector<float> check_plane;

        for (int64_t s = 0; s < pl.n_ns; ++s) {
            qsa_fill_bias_compact(s, ubatch, pl.n_ns, pl.n_blocks, ratio,
                    (int32_t *) qsa_blk->data, (int32_t *) qsa_qry->data, *pl.v_cs[s]);

            if (llama_qsa_compact_check_env() || qsa_tail_assert_enabled()) {
                qsa_compact_check(s, ubatch, n_kv, pl.n_ns, pl.n_blocks, ratio, pl.v_cell_blk[s],
                        (const int32_t *) qsa_blk->data, (const int32_t *) qsa_qry->data, *pl.v_cs[s], check_plane);
            }
        }
    } else {
        for (int64_t s = 0; s < pl.n_ns; ++s) {
            qsa_fill_bias(s, ubatch, n_kv, pl.n_ns, pl.n_blocks, ratio, blk_bias,
                    pl.v_cell_blk[s], (float *) bias->data, *pl.v_cs[s]);
        }
    }

    // this graph carries the store update: its stamps may be published once it has run
    st->pending_armed = true;

    st->stats.t_inputs_us += ggml_time_us() - t_start_us;
}

namespace {
// [TAG_QSA_INPUTS_INC] the memory's writer: straight into the mirror tensors (memcpy on a host-visible / UMA buffer)
struct qsa_mirror_writer_dev : llama_qsa_mirror_writer {
    const llama_qsa_pooled_store & st;
    explicit qsa_mirror_writer_dev(const llama_qsa_pooled_store & st) : st(st) {}
    void write(int arr, int64_t off, const int32_t * src, int64_t n) override {
        ggml_backend_tensor_set(st.mir_t[arr], src, (size_t) off*sizeof(int32_t), (size_t) n*sizeof(int32_t));
    }
};
}

void llama_memory_hybrid_idx_context::set_input_qsa_mirror(
        ggml_tensor * cell_blk,
        ggml_tensor * qsa_blk,
        ggml_tensor * qsa_qry,
        ggml_tensor * dirty_cells,
        ggml_tensor * dirty_pos,
        ggml_tensor * dirty_dst,
        ggml_tensor * remap,
        const llama_ubatch * ubatch,
        uint32_t ratio) const {
    using store_t = llama_qsa_pooled_store;

    llama_qsa_pooled_store * st = get_qsa_store();

    const auto & pl = qsa_plan;

    const int64_t n_kv     = pl.n_kv;
    const int64_t n_blocks = pl.n_blocks;
    const int64_t n_ns     = pl.n_ns;

    GGML_ASSERT(pl.inc && pl.blk_bias && qsa_compact_bias());
    GGML_ASSERT(qsa_blk != nullptr && qsa_qry != nullptr);
    GGML_ASSERT(cell_blk->view_src == st->mir_t[store_t::MIR_CELL_BLK] && cell_blk->view_offs == 0 &&
                ggml_is_contiguous(cell_blk) && cell_blk->ne[0] == n_kv && cell_blk->ne[1] == n_ns);
    GGML_ASSERT(qsa_blk->view_src == st->mir_t[store_t::MIR_BTAB] && qsa_blk->view_offs == 0 &&
                ggml_is_contiguous(qsa_blk) && qsa_blk->ne[1] == n_blocks && qsa_blk->ne[2] == n_ns);
    GGML_ASSERT(remap == nullptr || (remap->view_src == st->mir_t[store_t::MIR_REMAP] && remap->view_offs == 0 &&
                ggml_nelements(remap) == n_blocks*n_ns));
    GGML_ASSERT(ggml_backend_buffer_is_host(qsa_qry->buffer));
    GGML_ASSERT(qsa_qry->ne[1] == ubatch->n_tokens/n_ns);

    // the dirty list: O(cap), ordinary inputs
    memcpy(dirty_cells->data, pl.dirty_cells.data(), ggml_nbytes(dirty_cells));
    memcpy(dirty_pos  ->data, pl.dirty_pos  .data(), ggml_nbytes(dirty_pos));
    if (!pl.full) {
        GGML_ASSERT(dirty_dst != nullptr && remap != nullptr && ggml_backend_buffer_is_host(dirty_dst->buffer));
        GGML_ASSERT(dirty_dst->ne[0] == pl.cap*n_ns);
        memcpy(dirty_dst->data, pl.dirty_dst.data(), ggml_nbytes(dirty_dst));
    }

    // the query side of the compact bias: O(n_tokens), an ordinary input
    const int64_t n_tps = ubatch->n_tokens/n_ns;

    std::vector<const llama_kv_cells *> cv(n_ns);
    for (int64_t s = 0; s < n_ns; ++s) {
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];

        cv[s] = &mem->get_mem_idx()->get_cells(seq_of_stream);

        GGML_ASSERT(mem->qsa_stream_of(seq_of_stream) == pl.s0 + s);

        qsa_fill_bias_compact(s, ubatch, n_ns, n_blocks, ratio, nullptr, (int32_t *) qsa_qry->data, *pl.v_cs[s]);
    }

    // the mirrors: the runs the census changed since each was last written
    qsa_mirror_writer_dev w(*st);
    llama_qsa_mirror_update(*st, pl, cv, qsa_tail_fix_enabled(), !pl.full, /*want_btab =*/ true, w);

    // LLAMA_QSA_CENSUS_CHECK=1: what the device holds now == what the full fill uploads
    if (llama_qsa_census_check()) {
        std::vector<int32_t> dev;
        std::vector<int32_t> ref;

        auto cmp = [&](int arr, const char * what, int64_t s, int64_t n, const int32_t * want) {
            dev.resize(n);
            ggml_backend_tensor_get(st->mir_t[arr], dev.data(), (size_t) (s*n)*sizeof(int32_t), (size_t) n*sizeof(int32_t));
            for (int64_t i = 0; i < n; ++i) {
                if (dev[i] != want[i]) {
                    GGML_ABORT("[qsa-inputs-check] MISMATCH %s stream %lld [%lld]: device %d, full fill %d (n_kv=%lld "
                               "n_blocks=%lld n_ns=%lld plan %llu)", what, (long long) s, (long long) i, dev[i], want[i],
                               (long long) n_kv, (long long) n_blocks, (long long) n_ns,
                               (unsigned long long) st->stats.n_plan);
                }
            }
        };

        for (int64_t s = 0; s < n_ns; ++s) {
            cmp(store_t::MIR_CELL_BLK, "cell_blk", s, n_kv, pl.v_cell_blk[s]);
            if (!pl.full) {
                cmp(store_t::MIR_REMAP, "remap", s, n_blocks, pl.v_remap[s]);
            }
            ref.assign(2*n_blocks, 0);
            llama_qsa_fill_blk_tab(*cv[s], *pl.v_cs[s], n_blocks, 0, n_blocks, qsa_tail_fix_enabled(), ref.data());
            cmp(store_t::MIR_BTAB, "qsa_blk", s, 2*n_blocks, ref.data());
        }

        static std::atomic<uint64_t> n_checked { 0 };

        const uint64_t n = ++n_checked;
        if (n % 16 == 0 || n <= 4) {
            LLAMA_LOG_WARN("[qsa-inputs-check] %llu input sets identical (mirror writes: %llu KiB, %llu runs, %llu whole "
                           "arrays, %llu incremental)\n", (unsigned long long) n,
                           (unsigned long long) (st->stats.mir_bytes/1024), (unsigned long long) st->stats.mir_runs,
                           (unsigned long long) st->stats.mir_full, (unsigned long long) st->stats.mir_inc);
        }
    }

    // this graph carries the store update: its stamps may be published once it has run
    st->pending_armed = true;
}

void llama_memory_hybrid_idx_context::qsa_store_publish(bool ok) const {
    llama_qsa_pooled_store * st = get_qsa_store();

    if (st == nullptr || (st->pending.empty() && st->pending_streams.empty())) {
        return;
    }

    st->publish(ok);
}

std::string llama_memory_hybrid_idx::state_descriptor() const {
    // [TAG_MTP_QSA] the MTP draft memory (no recurrent layer): attention then indexer, both written whatever
    // PARTIAL_ONLY says (state_write's recr_none branch), so neither is "(full-only)" and there is no recr section
    if (recr_none) {
        // [TAG_MTP_QSA_PHASE] the phase changes what the indexer cells mean to the census: its own key (absent at 0)
        return "hybrid_idx{recr=none,attn=" + get_mem_attn()->state_descriptor() + ",idx=" +
               (mem_idx ? mem_idx->state_descriptor() : std::string("none")) +
               (qsa_phase_v != 0 ? ",qsa_phase=" + std::to_string(qsa_phase_v) : std::string()) + "}";
    }

    // [TAG_HYBRID_IDX_STATE] attention, recurrent, then the K-only indexer (full streams only)
    return "hybrid_idx{" + llama_memory_hybrid::state_descriptor() + ",idx(full-only)=" +
           (mem_idx ? mem_idx->state_descriptor() : std::string("none")) + "}";
}
