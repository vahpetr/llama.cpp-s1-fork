#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <functional>
#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    std::string state_descriptor() const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;
    // valid rollback planes written by the last ubatch for each seq
    std::vector<uint32_t> rs_valid_depth;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);
    void set_rs_valid_depth(llama_seq_id seq_id, uint32_t depth);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // [TAG_GDN_REPLAY] commit by replay of the gated delta net states (LLAMA_GDN_REPLAY=1, default off; see
    //   ggml_gated_delta_net_set_replay): S keeps one row per cell instead of 1 + n_rs_seq planes. A graph with a
    //   rollback depth commits the state after its first n_seq_tokens - D tokens (D = min(depth, n_seq_tokens - 1),
    //   the depth the planes would allow) and keeps the inputs of the last D tokens in the stash slot of the sequence
    //   (st_l, slot = seq id, two halves of gdn_w rows); the next graph replays the replay_n kept ones first. seq_rm
    //   drops tokens from the replay. Every state_write needs replay_n == 0: llama_context commits first (gdn_commit).
    //   R and P keep their planes
    bool gdn_replay = false;
    uint32_t gdn_w = 0; // stash rows per half (n_rs_seq)
    std::vector<uint32_t> replay_n;    // per seq
    std::vector<uint32_t> replay_half; // per seq: the stash half holding its rows
    std::vector<int32_t>  gdn_ctl_cur; // the current ubatch's control, I32 [4, n_seqs] (apply())

    // any deferred state to commit for seq_id (-1: any sequence)
    bool gdn_pending(llama_seq_id seq_id) const;
    // replays the kept tokens of seq_id (-1: of every sequence). in_place: into the cell's S row, then replay_n = 0
    //   (no rollback into those tokens afterwards). Otherwise into a snapshot row that state_write writes instead of
    //   the cell's row (gdn_snap): the live state stays deferred and a later seq_rm can still drop kept tokens.
    //   get_backend: the backend that computes on a buffer type. Returns false on a failure
    bool gdn_commit(llama_seq_id seq_id, bool in_place, const std::function<ggml_backend_t(ggml_backend_buffer_type_t)> & get_backend);
    // drops the snapshots (the next commit / decode; they live until then because an ON_DEVICE writer copies them
    //   when it is destroyed, after state_write)
    void gdn_release();
    // set by llama_context: an in-place commit of one sequence (seq_cp commits its source first)
    std::function<void(llama_seq_id)> gdn_commit_hook;
    // set by llama_context: waits for the running graphs (seq_cp reads the R / P planes of a pending rollback)
    std::function<void()> gdn_sync_hook;
    // snapshot of a cell's committed S per layer (gdn_commit without in_place), and the buffers that hold them
    std::map<uint32_t, std::vector<ggml_tensor *>> gdn_snap;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> gdn_snap_bufs;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    std::vector<ggml_tensor *> st_l; // the replay stash, F32 [row, gdn_w, 2*n_seq_max] (gdn_replay)
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges,
            const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges_s) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    // [TAG_GDN_REPLAY] the commit by replay (llama_memory_recurrent::gdn_replay)
    bool          get_gdn_replay() const;
    ggml_tensor * get_st_l(int32_t il) const;
    int32_t       s_copy_s(int i) const;           // the S row of a cell: no rollback planes
    void          set_gdn_ctl(int32_t * ctl) const; // I32 [4, n_seqs] of the current ubatch

    int32_t s_copy(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
