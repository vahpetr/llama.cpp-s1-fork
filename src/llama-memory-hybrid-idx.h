#pragma once

#include "llama-memory-hybrid.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

//
// [TAG_QSA_WHOLE_BLOCKS] LLAMA_QSA_WHOLE_BLOCKS=1 (default 0): the QSA selection in the reference's block-level form -
// the indexer_top_k/r best whole blocks plus each query's (q+1) mod r tail cells (HF Qwen4ExpTextQSAIndexer, D1/D2 of
// docs/verification/2026-10-03-indexer-reference-notes.md) - instead of the top indexer_top_k + r - 1 cells.
// MODEL-OUTPUT knob: off is today's graph node for node.
//
// the parser: 0 for unset / "" / "0", 1 for "1", -1 for anything else (WARN, treated as 0)
int  llama_qsa_whole_blocks_parse(const char * s);
// the knob, read once per process; the first call logs the proof line `qsa_select = whole_blocks` / `qsa_select = cells`
// (the memory calls it when it creates the indexer cache, so the line is in the load log)
bool llama_qsa_whole_blocks();

//
// [TAG_QSA_POOL_REMAP] LLAMA_QSA_POOL_REMAP=copy|indexed (default indexed) - only with LLAMA_QSA_POOLED_CACHE=1 (the
// remap exists only there). copy: every QSA layer of a graph with a dirty list copies the whole pool into the compact
// block order (get_rows of the store through the remap, n_blocks x idx_dim f32 read + written per layer) and scores
// that copy (mul_mat). indexed: the score reads the store through the remap itself (GGML_OP_MUL_MAT_ROWS), the copy is
// never made. EXACT: the backends compute MUL_MAT_ROWS with the plain MUL_MAT's kernel and reduction order, so scores,
// selections and logits are bit-identical; the knob is an A/B and rollback switch, not a model-output knob.
//
// the parser: 1 = indexed for unset / "" / "indexed", 0 = copy for "copy", -1 for anything else (WARN, treated as 1)
int  llama_qsa_pool_remap_parse(const char * s);
// the knob, read once per process; the first call logs the proof line `qsa_pool_remap = indexed` / `= copy` at WARN
// (the memory calls it when it creates the pooled-key cache, so the line is in the load log)
bool llama_qsa_pool_remap_indexed();

//
// [TAG_QSA_CENSUS_INC] LLAMA_QSA_CENSUS_INC=1 (default 0) - only with LLAMA_QSA_POOLED_CACHE=1 (the plan exists only
// there). The block census + stamps + slot plan of every graph keep a persistent per-stream state in the pooled store
// and re-group only the cells at or past the lowest position touched since the last plan (appends, MTP rollback); a
// stream whose state cannot be extended (more than one sequence, an M-RoPE ranked or duplicated census, a cell past the
// window, an n_kv shrink, or a mark that forces it: seq_cp / seq_keep / seq_add / seq_div / clear / state_read /
// state_drop / invalidate / K-shift or stream-copy update / overwrite of a used cell) gets the full census, as before.
// EXACT: every plan output (cell_blk, blk_cells, blk_pos, remap, dirty_*, cap, the census stream, the stamps, the slot
// assignment and free-slot order) is byte-identical to the full path's; LLAMA_QSA_CENSUS_CHECK=1 runs both per plan and
// aborts on the first differing element. Off is today's code path.
//
// the parser: 0 for unset / "" / "0", 1 for "1", -1 for anything else (WARN, treated as 0)
int  llama_qsa_census_inc_parse(const char * s);
// the knobs, read once per process; the first call logs the proof line `qsa_census = incremental (LLAMA_QSA_CENSUS_INC=1)`
// / `qsa_census = full (...)` at WARN (the memory calls it when it creates the pooled-key cache, so the line is in the
// load log). CHECK is honoured only with INC on.
bool llama_qsa_census_inc();

// [TAG_MULTISLOT] round 2 (astra r1 Major 2): the complete-history dense fallback of the QSA routing (R85 "dense until
// a full recompute") for TEXT sequences, independent of LLAMA_QSA_HOMOGENEOUS. LLAMA_QSA_HISTORY_DENSE (read once):
// 1 = on, 0 = off (today's routing bit for bit), unset / other = the default: on until this process has decoded an
// M-RoPE image (an embedding batch of a model with n_pos_per_embd > 1 - llama_qsa_note_mm_positions, sticky), then
// off - an image's repeated + skipped sequential positions read as an incomplete history and would route every image
// conversation dense. qwen4exp_qsa_dense takes the coverage term when this or LLAMA_QSA_HOMOGENEOUS=1 holds.
// `reason` (optional) receives a static string for the start-log proof line.
bool llama_qsa_history_dense(const char ** reason = nullptr);
void llama_qsa_note_mm_positions();
bool llama_qsa_census_check();

//
// [TAG_QSA_INPUTS_INC] LLAMA_QSA_INPUTS_INC=1 (default 0) - only with LLAMA_QSA_POOLED_CACHE=1, LLAMA_QSA_CENSUS_INC=1 and
// LLAMA_QSA_COMPACT_BIAS=1 (WARN + off otherwise). The three O(n_kv) QSA graph inputs - cell_blk [n_kv, ns], remap
// [n_blocks*ns] and the compact bias block table qsa_blk [2, n_blocks, ns] - stop being per-graph host inputs (memcpy /
// fill of every element + the scheduler's upload of the whole tensor): they are views of persistent device-resident
// mirrors owned by the pooled store, and set_input writes only the runs the incremental census changed since that
// mirror was last written (the block table is also FILLED incrementally from a persistent host copy). A full census, a
// layout change (other streams, n_kv change with > 1 stream) or a hint overflow writes the whole array. qsa_qry and the
// dirty list stay ordinary inputs (O(n_tokens) / O(cap)). The mirrors are written in place, so the context synchronizes
// its scheduler before set_inputs (the previous graph must be done reading them). EXACT: the mirrors hold the bytes
// the full fill would upload; LLAMA_QSA_CENSUS_CHECK=1 also downloads them after every set_input and aborts on the
// first differing element. Off is today's code path.
//
// the knob, read once per process; the first call logs the proof line `qsa_inputs = incremental (LLAMA_QSA_INPUTS_INC=1)`
// / `qsa_inputs = full (...)` at WARN (the memory calls it when it creates the pooled-key cache)
bool llama_qsa_inputs_inc();

// what the per-token bias fill needs from the block census of one stream
struct llama_qsa_census_stream {
    int32_t n_bid  = 0;
    bool    ranked = false;
    std::vector<int32_t> bid_idx;
    std::vector<int32_t> bid_cell;
    std::vector<int32_t> order;
    std::vector<int32_t> rank;
    std::vector<int32_t> blk_of;
    // [TAG_QSA_WINDOW_MISALIGNED] the sequences that own a cell past the cell window (block index >= n_blocks); empty
    // unless such a cell exists. The census excludes those cells; a query of one of these sequences cannot be served by
    // the sparse path (its depth is past the window, so the layer runs dense) and the bias fills assert it.
    std::vector<int32_t> oor_seqs;
};

// [TAG_QSA_CENSUS_INC] why a stream census took the full path (stats n_cen_fb, the kqvis-prof line)
enum {
    LLAMA_QSA_CEN_FAST      = 0,
    LLAMA_QSA_CEN_FB_MARK   = 1, // a full-census mark (seq_cp/keep/add/div, clear, state_read/drop, reset, update, overwrite,
                                 // a write run past kv_size/4) or the first plan
    LLAMA_QSA_CEN_FB_SHAPE  = 2, // more than one sequence, ranked / dup census, a cell past the window
    LLAMA_QSA_CEN_FB_SHRINK = 3, // n_kv shrank
    LLAMA_QSA_CEN_FB_REGION = 4, // the re-grouped region is > max(64, n_bid/8) blocks
    LLAMA_QSA_CEN_FB_COUNT  = 5,
};

// properties of a census the incremental path needs (out-parameter of llama_qsa_census_full)
struct llama_qsa_census_flags {
    bool one_seq = false;
    bool dup     = false;
    bool oor     = false;
};

// the full block census of one stream (the body of llama_memory_hybrid_idx_context::qsa_census): which complete group
// of r cells is which compact block id, the cells of each block (blk_cells [r*n_blocks]), the block of each cell
// (cell_blk [n_kv]) and the rope positions: blk_pos[sec*pos_stride + b] (the caller zero-fills them)
void llama_qsa_census_full(const llama_kv_cells & cells, bool is_pos_2d, int64_t n_kv, int64_t n_blocks, int64_t r,
                           bool blk_bias, int32_t * cell_blk, int32_t * blk_cells, int32_t * blk_pos, int64_t pos_stride,
                           llama_qsa_census_stream & cs, llama_qsa_census_flags * flags = nullptr, int64_t phase = 0);

// [TAG_MTP_QSA_PHASE] LLAMA_MTP_QSA_PHASE=1 (default 0, read once per process; honoured only in the MTP draft memory,
// i.e. an indexer cache that holds no trunk layer): the reference block phase for the sparse MTP draft (M1 of
// docs/verification/2026-10-07-mtp-indexer-parity.md). The draft stores e_k = (h_{k-1}, x_k) at position k plus a carrier
// row at 0; SGLang stores e_k at k - 1. With phase 1 every census works on idx = pos - 1: bucket (pos - 1)/r, block rope
// start pb*r + 1, tail start (q/r)*r + 1 - the reference's 4-blocks and tail. The carrier (idx -1) has no block, ever,
// and is handled as the reference's absent entry: in both selection forms it goes to the second unused block id, which
// the bias scores -inf (when no second unused id exists - the cache full to its last block - it keeps the tail boost of
// the unpooled cells). The dense paths (below LLAMA_QSA_MIN_KV, the catch-up) still attend it, as today.
// Phase 0 is today's code, byte for byte.
bool llama_mtp_qsa_phase_env();

// [TAG_MTP_QSA_PHASE] the census index of a cell or query is its position (rank, for a ranked census) minus the phase;
// its bucket is qsa_idx_bucket (-1 for the carrier's index -1, which has no block). The tail of a query at census
// index qi starts at llama_qsa_tail_start(qi, r) (qi = -1 -> 0); every block key (bid_idx) is a census index too.
static inline int64_t llama_qsa_idx_bucket(int64_t idx, int64_t r) { return idx < 0 ? -1 : idx/r; }
static inline int64_t llama_qsa_tail_start(int64_t qi, int64_t r)  { return (qi + 1)/r*r; }

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
    int32_t  phase    = 0;   // [TAG_MTP_QSA_PHASE] 0 or 1, the memory's census phase

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

        // [TAG_QSA_CENSUS_INC] the census of this stream's last plan (inc mode only) and the marks since then
        struct census_t {
            bool    valid    = false;  // the arrays hold the census + plan of this stream's last plan
            bool    inc_ok   = false;  // ... and the fast path may extend it: one seq, not ranked, no dup, no oor
            int64_t n_kv     = 0;
            int64_t n_blocks = 0;

            std::vector<int32_t>  cell_blk;   // [n_kv]
            std::vector<int32_t>  blk_cells;  // [r*n_blocks]
            std::vector<int32_t>  blk_pos;    // [4*n_blocks], sec-major
            std::vector<int32_t>  remap;      // [n_blocks] store row (stream offset included)
            std::vector<uint64_t> stamps;     // [n_bid]
            llama_qsa_census_stream cs;

            std::vector<int32_t> unpooled;    // the non-empty cells without a block (all at pb >= tail_pb)
            std::vector<int32_t> empty;       // the empty cells < n_kv
            int64_t              tail_pb = 0; // every non-empty cell below tail_pb*r is in a complete block
            std::vector<int32_t> carry;       // block ids of the last plan's dirty list (their stamps may be unpublished)

            // marks since the last plan
            bool                 inc_all    = true;       // the next plan of this stream takes the full census
            llama_pos            inc_lo_pos = INT32_MAX;  // the lowest position touched (written, removed)
            std::vector<int32_t> written;                 // cells written

            // scratch: per-cell dedup epoch
            std::vector<uint32_t> seen;
            uint32_t              seen_epoch = 0;

            // [TAG_QSA_INPUTS_INC] what changed since each device mirror last received this stream (store.mir_on only):
            // cell_blk: every listed cell (all = the whole array); remap / block table: every block id >= lo (0 = all,
            // INT64_MAX = none). btab = the persistent host block table [2*n_blocks] the mirror is written from
            bool                 mir_cb_all = true;
            std::vector<int32_t> mir_cb;
            int64_t              mir_rm_lo  = 0;
            int64_t              mir_bt_lo  = 0;
            std::vector<int32_t> btab;
        } cen;
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
        uint64_t n_plan_inc    = 0;  // [TAG_QSA_CENSUS_INC] plans of which every stream took the fast path
        uint64_t n_cen_inc     = 0;  // [TAG_QSA_CENSUS_INC] stream censuses: fast path / full
        uint64_t n_cen_full    = 0;
        uint64_t n_cen_fb[LLAMA_QSA_CEN_FB_COUNT] = {}; // full censuses by reason (index 0 unused)
        uint32_t last_dirty    = 0;  // of the last plan, all streams
        uint32_t last_cap      = 0;
        uint32_t last_n_bid    = 0;
        uint64_t t_inputs_us   = 0;  // [TAG_QSA_INPUTS_INC] time in set_input_qsa_store (both modes)
        uint64_t mir_bytes     = 0;  // [TAG_QSA_INPUTS_INC] bytes written to the device mirrors
        uint64_t mir_runs      = 0;  //   runs written (one tensor_set each)
        uint64_t mir_full      = 0;  //   (array, stream) writes of the whole array
        uint64_t mir_inc       = 0;  //   (array, stream) incremental writes (possibly empty)
    } stats;

    // [TAG_QSA_INPUTS_INC] the device-resident input mirrors (mir_on only): I32, per array the layout it was last written
    // with - ubatch stream s of the array at offset s*n (n = n_kv / n_blocks / 2*n_blocks), streams s0 .. s0+n_ns-1
    enum { MIR_CELL_BLK = 0, MIR_REMAP = 1, MIR_BTAB = 2, MIR_COUNT = 3 };
    bool mir_on = false;
    ggml_tensor * mir_t[MIR_COUNT] = { nullptr, nullptr, nullptr };   // [kv_size*n_stream], [n_slots*n_stream], [2*n_slots*n_stream]
    struct mir_layout_t {
        bool     valid = false;
        uint32_t s0    = 0;
        int64_t  n_ns  = 0;
        int64_t  n     = 0;
    } mir_layout[MIR_COUNT];
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> mir_ctxs_bufs;

    void reset_stream(uint32_t s);      // invalidate stream s: free every slot, no stamp valid
    void reset_all();
    void on_write(const struct llama_kv_cache::slot_info & sinfo); // bump the generations of the written cells
    void publish(bool ok);              // commit or drop the pending stamps

    // [TAG_QSA_CENSUS_INC] the marks every mutator of the indexer cells leaves (no-ops unless `inc`)
    bool inc = false;
    void inc_mark_all(uint32_t s);                       // the next census of stream s is a full one
    void inc_mark_all_streams();
    void inc_mark_pos(uint32_t s, llama_pos p0);         // positions >= p0 were removed (seq_rm)
    // before a write: a written cell that is in use (an overwrite, which may purge other cells) forces a full census
    void inc_mark_write_pre (uint32_t s, const llama_kv_cells & cells, const std::vector<uint32_t> & idxs);
    // after a write: the written cells and their new positions
    void inc_mark_write_post(uint32_t s, const llama_kv_cells & cells, const std::vector<uint32_t> & idxs);
};

// [TAG_QSA_CENSUS_INC] the plan of one ubatch (the census, the stamps, the slot assignment, the dirty list, the remap)
struct llama_qsa_plan {
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
    bool     inc      = false;   // the views point into the store's per-stream census (inc mode)
    uint32_t serial   = 0;       // the store serial of this plan

    // the full path's own arrays (all ubatch streams; inc mode leaves them empty)
    std::vector<int32_t> cell_blk;
    std::vector<int32_t> blk_cells;
    std::vector<int32_t> blk_pos;
    std::vector<llama_qsa_census_stream> cs;
    std::vector<int32_t> remap;

    std::vector<int32_t> dirty_cells;
    std::vector<int32_t> dirty_pos;
    std::vector<int32_t> dirty_dst;

    // per ubatch stream: where the census arrays are (set by both paths; read by the inputs and the check)
    std::vector<const int32_t *>                 v_cell_blk;   // [n_kv]
    std::vector<const int32_t *>                 v_blk_cells;  // [r*n_blocks]
    std::vector<const int32_t *>                 v_remap;      // [n_blocks]
    std::vector<const int32_t *>                 v_blk_pos;    // value (sec, b) = v_blk_pos[s][sec*v_pos_stride + b]
    int64_t                                      v_pos_stride = 0;
    std::vector<const llama_qsa_census_stream *> v_cs;
};

// [TAG_QSA_CENSUS_INC] the inputs of a plan: per ubatch stream its cells and its cache stream
struct llama_qsa_plan_in {
    int64_t n_ns      = 0;
    bool    is_pos_2d = false;
    int64_t n_kv      = 0;
    bool    blk_bias  = false;
    std::vector<const llama_kv_cells *> cells;  // [n_ns]
    std::vector<uint32_t>               g;      // [n_ns] cache stream
};

// the full path (today's plan, unchanged); stamps_out (optional) receives every block's stamp per ubatch stream
int64_t llama_qsa_plan_full(llama_qsa_pooled_store & st, const llama_qsa_plan_in & in, llama_qsa_plan & pl,
                            std::vector<std::vector<uint64_t>> * stamps_out = nullptr);
// the incremental path (st.inc must be set): per stream the fast census when its marks allow it, else the full census
int64_t llama_qsa_plan_inc(llama_qsa_pooled_store & st, const llama_qsa_plan_in & in, llama_qsa_plan & pl);

// the store state a plan changes, for the check (and the tests): copy out / back
struct llama_qsa_store_snapshot {
    std::vector<llama_qsa_pooled_store::stream_state> strm;
    uint32_t serial = 0;
    std::vector<llama_qsa_pooled_store::pending_t> pending;
    std::vector<uint32_t> pending_streams;
    bool pending_armed = false;
    llama_qsa_pooled_store::stats_t stats;
};
void llama_qsa_store_save   (const llama_qsa_pooled_store & st, llama_qsa_store_snapshot & snap);
void llama_qsa_store_restore(llama_qsa_pooled_store & st, const llama_qsa_store_snapshot & snap);

// the census state of cache stream g against its invariants after a plan (check mode, tests): the unpooled list is exactly
// the non-empty cells without a block, every empty cell < n_kv is in the empty list, no non-empty cell below tail_pb*r is
// unpooled; "" when they hold (inc_ok streams only)
std::string llama_qsa_census_state_check(const llama_qsa_pooled_store & st, uint32_t g, const llama_kv_cells & cells);

// the first difference between a full-path plan (+ its stamps and its store) and an incremental one (+ its store), as a
// readable string; "" when byte-identical. Compares cap/full/s0, per stream cell_blk, blk_cells, blk_pos, remap, the census
// stream (n_bid, ranked, bid_idx, bid_cell, blk_of, order/rank when ranked), dirty_cells/pos/dst, every block's stamp, and
// of the planned cache streams slot_key, slot_of_key, slot_stamp, free_slots, plus pending (in order) and pending_streams.
// (slot_seen is the full path's scratch and is not compared.)
std::string llama_qsa_plan_compare(const llama_qsa_plan & ref, const std::vector<std::vector<uint64_t>> & ref_stamps,
                                   const llama_qsa_pooled_store & st_ref, const llama_qsa_plan & inc,
                                   const llama_qsa_pooled_store & st_inc, uint32_t r,
                                   const std::vector<std::vector<uint64_t>> * inc_stamps = nullptr); // for a full-path `inc`

// [TAG_QSA_COMPACT_BIAS] the per-block half of the compact bias tables of one stream: entries [b0, b1) of dst_blk
// ([2, n_blocks]), from the census (qsa_fill_bias_compact's block loop; tail_fix = [TAG_QSA_TAIL_FIX])
void llama_qsa_fill_blk_tab(const llama_kv_cells & cells, const llama_qsa_census_stream & cs, int64_t n_blocks, int64_t b0,
                            int64_t b1, bool tail_fix, int32_t * dst_blk);

// [TAG_QSA_INPUTS_INC] where the mirror writes go (the device in the memory, host arrays in the tests)
struct llama_qsa_mirror_writer {
    virtual ~llama_qsa_mirror_writer() = default;
    // write n I32 elements of src at element offset off of mirror array `arr` (llama_qsa_pooled_store::MIR_*)
    virtual void write(int arr, int64_t off, const int32_t * src, int64_t n) = 0;
};

// [TAG_QSA_INPUTS_INC] bring the mirrors up to the plan `pl` (made by llama_qsa_plan_inc on `st`): per ubatch stream the
// changed runs of cell_blk, of remap (want_remap: a dirty-list graph) and of the block table (want_btab, refilled from
// cells[s] incrementally), or the whole array on a full census / a layout change / an overflow; consumes the hints
void llama_qsa_mirror_update(llama_qsa_pooled_store & st, const llama_qsa_plan & pl,
                             const std::vector<const llama_kv_cells *> & cells, bool tail_fix, bool want_remap,
                             bool want_btab, llama_qsa_mirror_writer & w);

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
                  int32_t   qsa_ratio_override = -1,
                            /* [TAG_MTP_QSA] the indexer cache's K type; GGML_TYPE_COUNT = type_k */
                ggml_type   type_idx = GGML_TYPE_COUNT);

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

    // [TAG_MTP_QSA] without recurrent layers the (layerless) recurrent cache holds no state: positions are the attention's
    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    std::string state_descriptor() const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // [TAG_QSA_POOLED_CACHE] nullptr unless LLAMA_QSA_POOLED_CACHE=1 and the model runs QSA
    llama_qsa_pooled_store * get_qsa_store() const;

    // the cache stream a sequence lives in
    uint32_t qsa_stream_of(llama_seq_id seq_id) const;

    // [TAG_MULTISLOT] the MTP draft memory (recr_none): the [qsa-history] proof's ctx label
    bool is_recr_none() const { return recr_none; }

    // [TAG_QSA_COMPACT_BIAS] LLAMA_QSA_COMPACT_BIAS=1 (read at creation, honoured only with an indexer and
    // n_seq_max <= 32): the per-block bias travels as two small tables instead of the [n_blocks, n_tps] plane
    bool qsa_compact_bias() const { return qsa_compact; }

    // [TAG_MTP_QSA_PHASE] the census phase of this memory: 1 for the MTP draft memory under LLAMA_MTP_QSA_PHASE=1, else 0
    int32_t qsa_phase() const { return qsa_phase_v; }

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

    // [TAG_QSA_COMPACT_BIAS]
    bool qsa_compact = false;

    // [TAG_MTP_QSA] filter_recr accepts no layer (the MTP draft context): the memory behaves as the plain attention KV
    // cache it replaces - partial seq_rm, positions and state (PARTIAL_ONLY ignored, as llama_kv_cache does) come from
    // the attention + indexer caches; the layerless recurrent cache is skipped
    bool recr_none = false;

    // [TAG_MTP_QSA_PHASE]
    int32_t qsa_phase_v = 0;
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
    // [TAG_QSA_COMPACT_BIAS] compact (qsa_compact_bias() and blk_bias): bias is nullptr and the two tables are set
    //   qsa_blk   I32 [2, n_blocks, ns]    per block (key = start index, INT32_MAX past n_bid; vis = seq bitmask)
    //   qsa_qry   I32 [2, n_tokens/ns, ns] per query (tail = tail start; bit = 1 << seq_id)
    // GGML_OP_QSA_BIAS turns them back into the per-block bias, bit for bit; otherwise both are nullptr
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, ggml_tensor * qsa_blk, ggml_tensor * qsa_qry,
                       const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;

    // [TAG_QSA_COMPACT_BIAS] the memory's knob (false for a context without a memory)
    bool qsa_compact_bias() const;

    // [TAG_QSA_POOLED_CACHE] the pooled-key cache (nullptr = off)
    llama_qsa_pooled_store * get_qsa_store() const;

    // a full context (init_full): the reserve graphs, whose ubatch carries no real positions
    bool is_full() const { return full; }

    // [TAG_MULTISLOT] stage 5.1 item 3: true when every sequence of the ubatch has a complete indexer history - its
    // positions 0..depth-1 all present (min 0, distinct count = max + 1, max >= depth - 1, depth = its max pos + 1 in
    // this ubatch). Sequence-owned counts, never the pool high-water. Evaluated after apply() (the ubatch's own cells
    // are in), cached per ubatch; the first partial sequence is logged as [qsa-history]. A full (reserve) context or
    // no indexer: true (the reserve keeps today's worst case and inspects no fabricated metadata).
    bool qsa_history_complete(const llama_ubatch & ubatch) const;

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
    //   qsa_blk/qsa_qry as in set_input_qsa ([TAG_QSA_COMPACT_BIAS]; bias is then nullptr)
    void set_input_qsa_store(ggml_tensor * cell_blk, ggml_tensor * bias, ggml_tensor * qsa_blk, ggml_tensor * qsa_qry,
                             ggml_tensor * dirty_cells, ggml_tensor * dirty_pos, ggml_tensor * dirty_dst,
                             ggml_tensor * remap, const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const;

    // [TAG_QSA_INPUTS_INC] set_input_qsa_store with the device mirrors (store mir_on): cell_blk / qsa_blk / remap are
    // views of llama_qsa_pooled_store::mir_t, written in place; qsa_qry and the dirty list are ordinary inputs
    void set_input_qsa_mirror(ggml_tensor * cell_blk, ggml_tensor * qsa_blk, ggml_tensor * qsa_qry,
                              ggml_tensor * dirty_cells, ggml_tensor * dirty_pos, ggml_tensor * dirty_dst,
                              ggml_tensor * remap, const llama_ubatch * ubatch, uint32_t ratio) const;

    // the cache stream of the planned ubatch's first stream (the full pool's store view starts there)
    uint32_t qsa_store_s0() const;

    // after the graph: commit the planned stamps (ok) or invalidate the ubatch's streams
    void qsa_store_publish(bool ok) const;

    // what the per-token bias fill needs from the block census of one stream
    using qsa_census_stream = llama_qsa_census_stream;

private:
    void qsa_census(int64_t s, const llama_ubatch * ubatch, int64_t n_kv, int64_t n_ns, int64_t n_blocks, int64_t r,
                    bool blk_bias, int32_t * cur_cell_blk, int32_t * cur_blk_cells, int32_t * dst_blk_pos,
                    qsa_census_stream & cs) const;

    void qsa_fill_bias(int64_t s, const llama_ubatch * ubatch, int64_t n_kv, int64_t n_ns, int64_t n_blocks, int64_t r,
                       bool blk_bias, const int32_t * cur_cell_blk, float * dst_bias, const qsa_census_stream & cs) const;

    // [TAG_QSA_COMPACT_BIAS] the tables of stream s (dst_blk: [2, n_blocks, n_ns], dst_qry: [2, n_tps, n_ns]), from the
    // same census and the same per-query helpers as qsa_fill_bias's per-block branch
    void qsa_fill_bias_compact(int64_t s, const llama_ubatch * ubatch, int64_t n_ns, int64_t n_blocks, int64_t r,
                               int32_t * dst_blk, int32_t * dst_qry, const qsa_census_stream & cs) const;

    // [TAG_QSA_COMPACT_BIAS] LLAMA_QSA_COMPACT_BIAS_CHECK=1: fill the old plane with the unchanged qsa_fill_bias, expand
    // the tables with the op's formula and GGML_ABORT on the first bit difference
    // on stream s; `plane` is the caller's [n_blocks, n_tokens] scratch (the old plane's own size)
    void qsa_compact_check(int64_t s, const llama_ubatch * ubatch, int64_t n_kv, int64_t n_ns, int64_t n_blocks,
                           int64_t r, const int32_t * cur_cell_blk, const int32_t * blk, const int32_t * qry,
                           const qsa_census_stream & cs, std::vector<float> & plane) const;

    // [TAG_QSA_POOLED_CACHE] the cached plan of the current ubatch
    using qsa_plan_t = llama_qsa_plan;

    mutable qsa_plan_t qsa_plan;

    // [TAG_MULTISLOT] qsa_history_complete() of the current ubatch (reset by apply / next)
    mutable bool hist_valid    = false;
    mutable bool hist_complete = true;

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

    // [TAG_QSA_CENSUS_INC] an update context (K-shift / stream copies): its apply() forces a full census
    bool is_update = false;
};
