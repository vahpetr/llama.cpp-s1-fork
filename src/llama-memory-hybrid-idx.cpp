#include "llama-memory-hybrid-idx.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cassert>
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
                  int32_t   qsa_ratio_override) :
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
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
    if (!mem_idx || !llama_qsa_pooled_cache_env()) {
        return;
    }

    // [TAG_QSA_POOLED_CACHE] the main QSA layers only: the layers the indexer cache holds that run with a
    // positive effective ratio. v1 serves one ratio; a GGUF mixing ratios keeps the recompute everywhere.
    const auto & hp = model.hparams;

    std::vector<int32_t> ils;
    uint32_t ratio = 0;
    bool     mixed = false;

    for (uint32_t il = 0; il < hp.n_layer(); ++il) {
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

    qsa_store = std::move(st);
    qsa_store->reset_all();
    qsa_store->stats.n_reset = 0;

    LLAMA_LOG_INFO("%s: LLAMA_QSA_POOLED_CACHE=1: %zu layers, ratio %u, %u block slots + 1 spare row x %u streams, "
            "idx_dim %u, f32: %zu bytes\n", __func__, ils.size(), ratio, qsa_store->n_slots, qsa_store->n_stream,
            qsa_store->idx_dim, total);
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
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
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
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
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
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
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
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
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
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    return res;
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

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
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
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    const bool tail_fix    = qsa_tail_fix_enabled();     // [TAG_QSA_TAIL_FIX]
    const bool tail_assert = qsa_tail_assert_enabled();

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    float   * dst_bias      = (float   *) bias->data;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = mem->get_mem_idx()->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

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

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

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

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
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
        if (dup && ubatch->is_pos_2d() && one_seq) {
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

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

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

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

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
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells point at a dead block. [TAG_QSA_TAIL_FIX]: when that id is a spare one
        // (which is implied by the existence of any unpooled cell, see the proof at the top of the
        // file) the per-block bias below boosts it instead of killing it, so the incomplete tail -
        // including the query's own key at decode - stays selectable above the budget.
        const int32_t dead_bid = n_bid < n_blocks ? n_bid : n_blocks - 1;

        // n_bid == n_blocks => no unpooled non-empty cell exists => dead_bid aliases a REAL block
        // and must keep that block's own bias
        const bool tail_bid_spare = n_bid < n_blocks;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

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

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

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

                        const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

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
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

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
}
