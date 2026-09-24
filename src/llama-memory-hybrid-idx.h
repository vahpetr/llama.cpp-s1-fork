#pragma once

#include "llama-memory-hybrid.h"

#include <map>
#include <memory>
#include <vector>

//
// [TAG_QSA_POOLED_CACHE] (M3 v1) the pooled-key cache
//
// LLAMA_QSA_POOLED_CACHE=1 (read when the memory is created, default 0 = off) keeps the indexer's
// pooled block keys - the values `indexer_k` has after pooling, norm and rope - in a persistent
// per-layer device tensor, so a graph pools only the blocks whose inputs changed ("dirty") instead of
// every block of the window. Derived state: the raw indexer cache stays the source of truth, and every
// entry can be rebuilt from it.
//
//   storage  F32 [idx_dim, n_rows*n_stream] per QSA layer, n_rows = n_slots + 1, n_slots = ceil(kv_size/r).
//            Row `slot` of stream s is a STABLE block slot: a block (identified by its key cell, the cell
//            at in-block index 0) keeps its slot for as long as it exists, whatever compact id the census
//            gives it. Row n_slots of each stream is the SPARE row: it holds what the recompute gives every
//            compact id >= n_bid (r copies of cell 0 at position 0), rewritten by every graph.
//   validity a block's slot is valid iff its committed stamp equals
//            hash(stream, ordered member cells, each member's raw-row write generation, each member's
//            position + ext, the block's 4 rope positions, ranked). Generations are bumped only on the
//            committed write path (llama_memory_hybrid_idx_context::apply), never by batch planning.
//   publish  stamps of the blocks a graph pools are committed only after that graph was submitted
//            successfully; a failed graph invalidates its streams.
struct llama_qsa_pooled_store {
    uint32_t ratio    = 0;
    uint32_t idx_dim  = 0;
    uint32_t kv_size  = 0;
    uint32_t n_stream = 0;
    uint32_t n_slots  = 0;   // ceil(kv_size/ratio)
    uint32_t n_rows   = 0;   // n_slots + 1 (the spare row)

    // il -> F32 [idx_dim, n_rows*n_stream]
    std::map<int32_t, ggml_tensor *> layers;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    struct stream_state {
        std::vector<uint32_t> gen;          // [kv_size] raw-row write generation per cell
        std::vector<int32_t>  slot_of_key;  // [kv_size] slot owned by the block whose key cell this is, -1
        std::vector<int32_t>  slot_key;     // [n_slots] key cell of the owning block, -1 = free
        std::vector<uint64_t> slot_stamp;   // [n_slots] committed stamp, 0 = invalid
        std::vector<uint32_t> slot_seen;    // [n_slots] census serial that last saw the owner
        std::vector<int32_t>  free_slots;   // stack, pops the lowest free slot first after a reset
    };

    std::vector<stream_state> strm;

    uint32_t serial = 0;

    // stamps waiting for their graph
    struct pending_t {
        uint32_t s;
        int32_t  slot;
        uint64_t stamp;
    };

    std::vector<pending_t> pending;
    std::vector<uint32_t>  pending_streams;  // the streams of the planned ubatch
    bool                   pending_armed = false; // the graph's store inputs were set

    // counters (read by tests and the debug log)
    struct stats_t {
        uint64_t n_plan        = 0;
        uint64_t n_full        = 0;  // plans that pooled every block (cold rebuilds)
        uint64_t n_dirty       = 0;  // pooled blocks, summed over plans
        uint64_t n_valid       = 0;  // blocks served from the store, summed over plans
        uint64_t n_reset       = 0;  // stream invalidations
        uint64_t n_publish     = 0;
        uint64_t n_fail        = 0;
        uint64_t t_plan_us     = 0;
        uint32_t last_dirty    = 0;  // of the last plan, all streams
        uint32_t last_cap      = 0;
        uint32_t last_n_bid    = 0;
    } stats;

    void reset_stream(uint32_t s);      // invalidate stream s: free every slot, no stamp valid
    void reset_all();
    void on_write(const struct llama_kv_cache::slot_info & sinfo); // bump the generations of the written cells
    void publish(bool ok);              // commit or drop the pending stamps
};

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
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
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx,
                            /* [TAG_QSA_RATIO_OVERRIDE] llama_cparams::qsa_ratio, for the pooled-key cache */
                  int32_t   qsa_ratio_override = -1);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void set_n_kv_reserve_max(uint32_t n_kv_max) override; // [TAG_KV_GRAPH_SPAN]

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // [TAG_QSA_POOLED_CACHE] nullptr unless LLAMA_QSA_POOLED_CACHE=1 and the model runs QSA
    llama_qsa_pooled_store * get_qsa_store() const;

    // the cache stream a sequence lives in
    uint32_t qsa_stream_of(llama_seq_id seq_id) const;

private:
    // [TAG_QSA_POOLED_CACHE] drop the derived entries of seq_id's stream (every stream if seq_id < 0)
    void qsa_store_invalidate(llama_seq_id seq_id);

    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // [TAG_QSA_POOLED_CACHE] mutable: the graph inputs are planned and filled through const contexts
    mutable std::unique_ptr<llama_qsa_pooled_store> qsa_store;
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias) const;

    // [TAG_QSA_POOLED_CACHE] the pooled-key cache (nullptr = off)
    llama_qsa_pooled_store * get_qsa_store() const;

    // a full context (init_full): the reserve graphs, whose ubatch carries no real positions
    bool is_full() const { return full; }

    // plan this ubatch's store update: the block census, the stamps, the dirty list and the remap.
    // Returns the per-stream capacity of the dirty list, the shape the graph must be built with.
    // A full (reserve) context has no census: it returns the cold-rebuild worst case, n_blocks.
    // Cached per ubatch, so can_reuse, the graph build and set_input see one plan.
    int64_t qsa_store_plan(const llama_ubatch & ubatch, uint32_t ratio, int64_t n_kv, bool blk_bias) const;

    //   cell_blk/bias as in set_input_qsa
    //   dirty_cells I32 [ratio*cap, ns]  member cells of each block to pool (per stream, like blk_cells)
    //   dirty_pos   I32 [4*cap*ns]       its rope positions (the layout of blk_pos)
    //   dirty_dst   I32 [cap*ns]         its store row (stream offset included)
    //   remap       I32 [n_blocks*ns]    the store row of each compact block id (stream offset included)
    // dirty_dst and remap are nullptr for the FULL pool (cap == n_blocks): its dirty list is every compact id in
    // order (= blk_cells/blk_pos), the graph reads the pooled rows directly and copies them to slots 0..n_blocks-1
    void set_input_qsa_store(ggml_tensor * cell_blk, ggml_tensor * bias,
                             ggml_tensor * dirty_cells, ggml_tensor * dirty_pos, ggml_tensor * dirty_dst,
                             ggml_tensor * remap, const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;

    // the cache stream of the planned ubatch's first stream (the full pool's store view starts there)
    uint32_t qsa_store_s0() const;

    // after the graph: commit the planned stamps (ok) or invalidate the ubatch's streams
    void qsa_store_publish(bool ok) const;

    // what the per-token bias fill needs from the block census of one stream
    struct qsa_census_stream {
        int32_t n_bid  = 0;
        bool    ranked = false;
        std::vector<int32_t> bid_idx;
        std::vector<int32_t> bid_cell;
        std::vector<int32_t> order;
        std::vector<int32_t> rank;
        std::vector<int32_t> blk_of;
    };

private:
    void qsa_census(int64_t s, const llama_ubatch * ubatch, int64_t n_kv, int64_t n_ns, int64_t n_blocks, int64_t r,
                    bool blk_bias, int32_t * cur_cell_blk, int32_t * cur_blk_cells, int32_t * dst_blk_pos,
                    qsa_census_stream & cs) const;

    void qsa_fill_bias(int64_t s, const llama_ubatch * ubatch, int64_t n_kv, int64_t n_ns, int64_t n_blocks, int64_t r,
                       bool blk_bias, const int32_t * cur_cell_blk, float * dst_bias, const qsa_census_stream & cs) const;

    // [TAG_QSA_POOLED_CACHE] the cached plan of the current ubatch
    struct qsa_plan_t {
        bool     valid    = false;
        size_t   i_ubatch = 0;
        uint32_t ratio    = 0;
        int64_t  n_kv     = 0;
        int64_t  n_blocks = 0;
        int64_t  n_ns     = 0;
        bool     blk_bias = false;
        int64_t  cap      = 0;
        bool     full     = false;   // cap == n_blocks: every block pooled, slots in compact order
        uint32_t s0       = 0;       // the cache stream of ubatch stream 0

        std::vector<int32_t> cell_blk;
        std::vector<int32_t> blk_cells;
        std::vector<int32_t> blk_pos;
        std::vector<qsa_census_stream> cs;

        std::vector<int32_t> dirty_cells;
        std::vector<int32_t> dirty_pos;
        std::vector<int32_t> dirty_dst;
        std::vector<int32_t> remap;
    };

    mutable qsa_plan_t qsa_plan;

    // a full context builds the reserve graphs: no ubatch, no census
    const bool full = false;

    // the indexer slot infos, kept for the write generations (pooled-key cache only)
    slot_info_vec_t sinfos_idx_copy;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
