#include "llama-memory-recurrent.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-ext.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                ggml_type   type_r,
                ggml_type   type_s,
                     bool   offload,
                 uint32_t   mem_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter) : hparams(model.hparams), n_seq_max(n_seq_max) {
    const int32_t n_layer = hparams.n_layer();

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);
    rs_valid_depth.assign(n_seq_max, 0);

    // [TAG_GDN_REPLAY] the gated delta net layers of qwen4exp (S = S_v x S_v per value head, scalar gate)
    {
        const char * env = getenv("LLAMA_GDN_REPLAY");
        const bool want = env != nullptr && atoi(env) != 0;
        gdn_replay = want && n_rs_seq > 0 && model.arch == LLM_ARCH_QWEN4EXP && type_s == GGML_TYPE_F32 &&
                     hparams.n_embd_s() == hparams.ssm_d_state * hparams.ssm_d_state * hparams.ssm_dt_rank;
        if (want && !gdn_replay && n_rs_seq > 0) {
            LLAMA_LOG_WARN("%s: LLAMA_GDN_REPLAY=1 ignored: not a qwen4exp gated delta net cache\n", __func__);
        }
        gdn_w = gdn_replay ? n_rs_seq : 0;
        replay_n.assign(n_seq_max, 0);
        replay_half.assign(n_seq_max, 0);
    }
    // the stash row of a token: k [S_k*H_k], v [S_v*H_v], g [H_v], beta [H_v]
    const int64_t gdn_row = (int64_t) hparams.ssm_d_state * hparams.ssm_n_group +
                            (int64_t) hparams.ssm_d_state * hparams.ssm_dt_rank + 2 * (int64_t) hparams.ssm_dt_rank;

    cells.clear();
    cells.resize(mem_size);

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                // r and s per layer, plus the separate PLE conv row where the model has one
                /*.mem_size   =*/ size_t((hparams.ple_conv_state() > 0 ? 3u : 2u)*n_layer*ggml_tensor_overhead() +
                                           (gdn_replay ? n_layer*ggml_tensor_overhead() : 0)), // + the replay stash
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    r_l.resize(n_layer);
    s_l.resize(n_layer);
    p_l.resize(n_layer);
    st_l.resize(n_layer);

    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(i);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, i, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for rs cache");
        }

        const uint32_t n_rows = mem_size * (1 + n_rs_seq);
        ggml_tensor * r = ggml_new_tensor_2d(ctx, type_r, hparams.n_embd_r(), n_rows);
        // [TAG_GDN_REPLAY] S without rollback planes, the inputs of the deferred tokens in the stash instead
        ggml_tensor * s = ggml_new_tensor_2d(ctx, type_s, hparams.n_embd_s(), gdn_replay ? mem_size : n_rows);
        ggml_format_name(r, "cache_r_l%d", i);
        ggml_format_name(s, "cache_s_l%d", i);
        r_l[i] = r;
        s_l[i] = s;
        if (gdn_replay) {
            st_l[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, gdn_row, gdn_w, 2 * n_seq_max);
            ggml_format_name(st_l[i], "cache_gdn_stash_l%d", i);
        }

        // the PLE history needs its own row: Meta must mirror it while the delta-net conv state next door stays split
        if (hparams.ple_conv_state() > 0 && hparams.is_ple(i)) {
            ggml_tensor * p = ggml_new_tensor_2d(ctx, type_r, hparams.ple_conv_state(), n_rows);
            ggml_format_name(p, "cache_ple_r_l%d", i);
            p_l[i] = p;
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for rs cache");
        }
        ggml_backend_buffer_clear(buf, 0);
        LLAMA_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_r = size_r_bytes();
        const size_t memory_size_s = size_s_bytes();
        const size_t memory_size_p = size_p_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, S (%s): %7.2f MiB, P (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_r + memory_size_s + memory_size_p) / (1024.0f * 1024.0f), mem_size, n_layer, n_seq_max, n_rs_seq,
                ggml_type_name(type_r), (float)memory_size_r / (1024.0f * 1024.0f),
                ggml_type_name(type_s), (float)memory_size_s / (1024.0f * 1024.0f),
                ggml_type_name(type_r), (float)memory_size_p / (1024.0f * 1024.0f));
    }

    if (gdn_replay) {
        size_t st_bytes = 0;
        for (const auto * st : st_l) {
            st_bytes += st ? ggml_nbytes(st) : 0;
        }
        LLAMA_LOG_INFO("%s: gdn replay: S without rollback planes, stash %u x 2 x %u rows = %.2f MiB (LLAMA_GDN_REPLAY)\n",
                __func__, n_seq_max, gdn_w, st_bytes / (1024.0f * 1024.0f));
    }
}

void llama_memory_recurrent::clear(bool data) {
    for (int32_t i = 0; i < (int32_t) size; ++i) {
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
        cells[i].tail = -1;
    }

    head = 0;
    used = 0;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);
    std::fill(rs_valid_depth.begin(), rs_valid_depth.end(), 0);
    std::fill(replay_n.begin(), replay_n.end(), 0);
    std::fill(replay_half.begin(), replay_half.end(), 0);
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id >= this->n_seq_max) {
        LLAMA_LOG_ERROR("%s: invalid seq_id (%d) - larger than n_seq_max (%d)\n", __func__, seq_id, this->n_seq_max);
        return false;
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        set_rs_idx(seq_id, 0);
        set_rs_valid_depth(seq_id, 0);
        if (seq_id < 0) {
            std::fill(replay_n.begin(), replay_n.end(), 0);
        } else if ((uint32_t) seq_id < replay_n.size()) {
            replay_n[seq_id] = 0;
        }
    }

    // models like Mamba or RWKV can't have a state partially erased at the end
    // of the sequence because their state isn't preserved for previous tokens
    if (seq_id >= (int64_t) size) {
        // could be fatal
        return false;
    }
    if (0 <= seq_id) {
        int32_t & tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];

            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);
                // pending rollback is single-use
                const bool pending = rs_idx[seq_id] != 0;
                // planes deeper than rs_valid_depth were not written by the last ubatch of this seq (they still
                //   hold snapshots of an older one), so refuse instead of restoring a stale state - the caller
                //   then falls back to its checkpoint/replay path
                // [TAG_GDN_REPLAY] the deferred S state can only drop the tokens it still keeps as inputs
                const bool replay_ok = !gdn_replay || (rollback <= (llama_pos) replay_n[seq_id] && cell.seq_id.size() == 1);
                if (!pending && replay_ok && rollback >= 1 && rollback <= (llama_pos) rs_valid_depth[seq_id]) {
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    if (gdn_replay) {
                        replay_n[seq_id] -= (uint32_t) rollback;
                    }
                    cell.pos = p0 - 1;
                    return true;
                }
                return false;
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
                set_rs_idx(seq_id, 0);
                set_rs_valid_depth(seq_id, 0);
                if ((uint32_t) seq_id < replay_n.size()) {
                    replay_n[seq_id] = 0;
                }
            }
        }
    } else {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            //printf("[DEBUG] inside `llama_memory_recurrent::seq_rm`: `seq_id` is negative, so returning false\n");
            return false;
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id_dst < size && (uint32_t) seq_id_src < size) {
        set_rs_idx(seq_id_dst, 0);
        set_rs_valid_depth(seq_id_dst, 0);

        // [TAG_GDN_REPLAY] the destination shares the source's cell: the source is committed first (in place), so a
        //   shared cell never holds deferred tokens (a later rollback of either sequence is refused, as a deferred
        //   state of one sequence must not be replayed into a row another one reads)
        if (gdn_replay && replay_n[seq_id_src] > 0) {
            GGML_ASSERT(gdn_commit_hook && "gdn replay: seq_cp of a deferred state needs the context's commit");
            gdn_commit_hook(seq_id_src);
            GGML_ASSERT(replay_n[seq_id_src] == 0);
        }
        if (gdn_replay) {
            replay_n[seq_id_dst] = 0;
        }

        auto & tail_src = cells[seq_id_src];
        auto & tail_dst = cells[seq_id_dst];
        if (tail_dst.tail >= 0) {
            // clear destination seq_id if it wasn't empty
            auto & cell_dst = cells[tail_dst.tail];

            cell_dst.seq_id.erase(seq_id_dst);
            tail_dst.tail = -1;
            if (cell_dst.seq_id.empty()) {
                cell_dst.pos = -1;
                cell_dst.src = -1;
                used -= 1;
            }
        }
        if (tail_src.tail >= 0) {
            auto & cell_src = cells[tail_src.tail];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst.tail = tail_src.tail;
        }
    }
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < size; ++i) {
        if ((llama_seq_id) i != seq_id) {
            cells[i].tail = -1;
            set_rs_idx(i, 0);
            set_rs_valid_depth(i, 0);
            if (i < replay_n.size()) {
                replay_n[i] = 0; // [TAG_GDN_REPLAY]
            }
        }

        if (!cells[i].has_seq_id(seq_id)) {
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) size) {
        const int32_t tail_id = cells[seq_id].tail;
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0) {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
        return;
    }

    assert(n_seq_max == rs_idx.size());

    GGML_ASSERT((uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(idx <= n_rs_seq);

    rs_idx[seq_id] = idx;
}

void llama_memory_recurrent::set_rs_valid_depth(llama_seq_id seq_id, uint32_t depth) {
    if (seq_id < 0) {
        std::fill(rs_valid_depth.begin(), rs_valid_depth.end(), 0);
        return;
    }

    assert(n_seq_max == rs_valid_depth.size());

    GGML_ASSERT((uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(depth <= n_rs_seq);

    rs_valid_depth[seq_id] = depth;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ret[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

bool llama_memory_recurrent::gdn_pending(llama_seq_id seq_id) const {
    for (size_t i = 0; i < replay_n.size(); ++i) {
        if (replay_n[i] > 0 && (seq_id < 0 || (llama_seq_id) i == seq_id)) {
            return true;
        }
    }
    return false;
}

void llama_memory_recurrent::gdn_release() {
    gdn_snap.clear();
    gdn_snap_bufs.clear();
}

// [TAG_GDN_REPLAY] a graph of its own, on the backend of each layer's buffer: per layer one gated delta net op in commit
//   mode (n_defer < 0) that replays the kept rows on the cell's S row - the same pipeline and arithmetic as the replay
//   at the start of the next decode graph would have run - into the row itself (in_place) or into a snapshot row
bool llama_memory_recurrent::gdn_commit(llama_seq_id seq_id, bool in_place, const std::function<ggml_backend_t(ggml_backend_buffer_type_t)> & get_backend) {
    gdn_release();
    if (!gdn_replay) {
        return true;
    }

    const int64_t S_v = hparams.ssm_d_state;
    const int64_t H_v = hparams.ssm_dt_rank;
    const int64_t H_k = hparams.ssm_n_group;

    for (llama_seq_id seq = 0; seq < (llama_seq_id) replay_n.size(); ++seq) {
        if (replay_n[seq] == 0 || (seq_id >= 0 && seq != seq_id)) {
            continue;
        }
        const int32_t tail = (uint32_t) seq < size ? cells[seq].tail : -1;
        if (tail < 0) {
            replay_n[seq] = 0;
            continue;
        }
        // seq_cp commits its source and a decode moves a sequence to a cell of its own: deferred = unshared
        GGML_ASSERT(cells[tail].seq_id.size() == 1 && "gdn replay: deferred tokens in a shared cell");
        const uint32_t row = cells[tail].src >= 0 ? (uint32_t) cells[tail].src : (uint32_t) tail;

        // one graph per buffer type (the layers of one device)
        std::map<ggml_backend_buffer_type_t, std::vector<int>> groups;
        for (int il = 0; il < (int) s_l.size(); ++il) {
            if (s_l[il] != nullptr) {
                groups[ggml_backend_buffer_get_type(s_l[il]->buffer)].push_back(il);
            }
        }
        std::vector<ggml_tensor *> snap(s_l.size(), nullptr);
        for (const auto & [buft, layers] : groups) {
            ggml_backend_t backend = get_backend(buft);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: no backend for %s\n", __func__, ggml_backend_buft_name(buft));
                gdn_release();
                return false;
            }
            const size_t n_tensors = 16 * layers.size() + 4;
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors * ggml_tensor_overhead() + ggml_graph_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr ctx { ggml_init(params) };
            ggml_cgraph * gf = ggml_new_graph(ctx.get());
            ggml_tensor * ctl = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 4, 1);
            // the window inputs are not read in commit mode
            ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H_k, 1, 1);
            ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H_k, 1, 1);
            ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S_v, H_v, 1, 1);
            ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H_v, 1, 1);
            ggml_tensor * b = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H_v, 1, 1);
            for (int il : layers) {
                ggml_tensor * sl = s_l[il];
                ggml_tensor * state = ggml_view_4d(ctx.get(), sl, S_v, S_v, H_v, 1,
                        S_v * ggml_element_size(sl), S_v * S_v * ggml_element_size(sl), sl->nb[1], (size_t) row * sl->nb[1]);
                ggml_tensor * dst = in_place ?
                    ggml_view_3d(ctx.get(), sl, sl->ne[0], 1, 1, sl->nb[1], sl->nb[1], (size_t) row * sl->nb[1]) :
                    ggml_new_tensor_3d(ctx.get(), sl->type, sl->ne[0], 1, 1);
                if (!in_place) {
                    snap[il] = dst;
                }
                ggml_tensor * gdn = ggml_gated_delta_net_ext(ctx.get(), q, k, v, g, b, state, dst);
                ggml_gated_delta_net_set_replay(gdn, st_l[il], ctl);
                ggml_build_forward_expand(gf, gdn);
            }
            ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft) };
            if (!buf) {
                LLAMA_LOG_ERROR("%s: failed to allocate the commit graph\n", __func__);
                gdn_release();
                return false;
            }
            ggml_backend_buffer_clear(buf.get(), 0);
            const int32_t c[4] = { (int32_t) replay_n[seq], (int32_t) replay_half[seq], -1, (int32_t) seq };
            ggml_backend_tensor_set(ctl, c, 0, sizeof(c));
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute the commit graph\n", __func__);
                gdn_release();
                return false;
            }
            if (!in_place) {
                gdn_snap_bufs.emplace_back(std::move(ctx), std::move(buf));
            }
        }

        if (in_place) {
            replay_n[seq] = 0;
        } else {
            gdn_snap[row] = std::move(snap);
        }
    }

    return true;
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + rs_rollback_depth) tokens of each seq must stay in the same ubatch,
                //   so that every rollback plane the graph writes for it stays valid
                const uint32_t rs_depth = balloc.get_rs_rollback_depth();
                ubatch = balloc.split_equal(n_ubatch, true, rs_depth > 0 ? rs_depth + 1 : 0);
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

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    auto org_cells = cells;
    auto org_used = used;
    auto org_head = head;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    // restore the original state
    cells = std::move(org_cells);
    used = org_used;
    head = org_head;

    return success;
}

bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head > used + 2*n_seqs) {
        head = 0;
    }

    // For recurrent state architectures (like Mamba or RWKV),
    // each cache cell can store the state for a whole sequence.
    // A slot should be always be contiguous.

    // can only process batches with an equal number of new tokens in each sequence
    GGML_ASSERT(ubatch.equal_seqs());

    int32_t min = size - 1;
    int32_t max = 0;

    // everything should fit if all seq_ids are smaller than the max
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens; // first token of sequence set s
        const uint32_t n_seq_id = ubatch.n_seq_id[i];

        for (uint32_t j = 0; j < n_seq_id; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];

            if (seq_id < 0 || (uint32_t) seq_id >= size) {
                // too big seq_id
                // TODO: would it be possible to resize the cache instead?
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                return false;
            }
            if (j > 0) {
                auto & seq = cells[seq_id];
                if (seq.tail >= 0) {
                    auto & cell = cells[seq.tail];
                    // clear cells from seq_ids that become shared
                    // (should not normally happen, but let's handle it anyway)
                    cell.seq_id.erase(seq_id);
                    seq.tail = -1;
                    if (cell.seq_id.empty()) {
                        cell.pos = -1;
                        cell.src = -1;
                        used -= 1;
                    }
                }
            }
        }
    }

#ifndef NDEBUG
    {
        std::vector<int32_t> tails_verif;
        tails_verif.assign(size, -1);
        for (uint32_t i = 0; i < size; ++i) {
            auto & cell = cells[i];
            for (llama_seq_id seq_id : cell.seq_id) {
                if (tails_verif[seq_id] != -1) {
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tails_verif[seq_id]);
                }
                tails_verif[seq_id] = i;
            }
        }
        for (uint32_t i = 0; i < size; ++i) {
            if (tails_verif[i] != cells[i].tail) {
                LLAMA_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, i, cells[i].tail, tails_verif[i]);
            }
        }
    }
#endif

    // find next empty cell
    uint32_t next_empty_cell = head;

    for (uint32_t i = 0; i < size; ++i) {
        if (next_empty_cell >= size) { next_empty_cell -= size; }
        auto & cell = cells[next_empty_cell];
        if (cell.is_empty()) { break; }
        next_empty_cell += 1;
    }

    // find usable cell range
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        auto & seq_meta = cells[seq_id];
        bool has_cell = false;
        if (seq_meta.tail >= 0) {
            auto & cell = cells[seq_meta.tail];
            GGML_ASSERT(cell.has_seq_id(seq_id));
            // does this seq_id "own" the cell?
            if (cell.seq_id.size() == 1) { has_cell = true; }
        }
        if (!has_cell) {
            auto & empty_cell = cells[next_empty_cell];
            GGML_ASSERT(empty_cell.is_empty());
            // copy old tail into the empty cell
            if (seq_meta.tail >= 0) {
                auto & orig_cell = cells[seq_meta.tail];
                empty_cell.pos = orig_cell.pos;
                empty_cell.src = orig_cell.src;
                orig_cell.seq_id.erase(seq_id);
                empty_cell.seq_id.insert(seq_id); // will be overwritten
                GGML_ASSERT(!orig_cell.is_empty()); // has at least one remaining seq_id
            }
            seq_meta.tail = next_empty_cell;
            // find next empty cell
            if (s + 1 < n_seqs) {
                for (uint32_t j = 0; j < size; ++j) {
                    next_empty_cell += 1;
                    if (next_empty_cell >= size) { next_empty_cell -= size; }
                    auto & cell = cells[next_empty_cell];
                    if (cell.is_empty()) { break; }
                }
            }
        }
        if (min > seq_meta.tail) { min = seq_meta.tail; }
        if (max < seq_meta.tail) { max = seq_meta.tail; }
    }

    // gather and re-order
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const int32_t dst_id = s + min;
        const int32_t src_id = cells[ubatch.seq_id[i][0]].tail;
        if (dst_id != src_id) {
            auto & dst_cell = cells[dst_id];
            auto & src_cell = cells[src_id];

            std::swap(dst_cell.pos, src_cell.pos);
            std::swap(dst_cell.src, src_cell.src);
            std::swap(dst_cell.seq_id, src_cell.seq_id);

            // swap tails
            for (uint32_t j = 0; j < size; ++j) {
                int32_t & tail = cells[j].tail;
                if (tail == src_id) {
                    tail = dst_id;
                } else if (tail == dst_id) {
                    tail = src_id;
                }
            }
        }
    }

    // update the pos of the used seqs
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;
        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];
        const int32_t cell_id = s + min;
        auto & cell = cells[cell_id];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            // What should happen when the pos backtracks or skips a value?
            // Clearing the state mid-batch would require special-casing which isn't done.
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, ubatch.seq_id[i][0], n_seq_tokens);
        }
        cell.pos = last_pos;
        cell.seq_id.clear();
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            const llama_seq_id seq_id = ubatch.seq_id[i][j];
            cell.seq_id.insert(seq_id);
            cells[seq_id].tail = cell_id;
        }
    }

    // Find first cell without src refs, to use as the zero-ed state
    {
        // TODO: bake-in src refcounts in the cell metadata
        std::vector<int32_t> refcounts(size, 0);
        for (size_t i = 0; i < size; ++i) {
            const int32_t src = cells[i].src;
            if (src >= 0) {
                refcounts[src] += 1;
            }
        }

        rs_z = -1;
        for (int i = min; i <= max; ++i) {
            if (refcounts[i] == 0) {
                rs_z = i;
                break;
            }
        }

        for (int i = min; i <= max; ++i) {
            if (cells[i].src < 0) {
                GGML_ASSERT(rs_z >= 0);
                cells[i].src0 = rs_z;
            } else {
                // Stage the source ids for all used cells to allow correct seq_* behavior
                // and still make these values available when setting the inputs
                cells[i].src0 = cells[i].src;
            }
            cells[i].src = i; // avoid moving or clearing twice
        }
    }

    // allow getting the range of used cells, from head to head + n
    head = min;
    n    = max - min + 1;
    used = std::count_if(cells.begin(), cells.end(),
        [](const mem_cell & cell){ return !cell.is_empty(); });

    // sanity check
    return n >= n_seqs;
}

bool llama_memory_recurrent::get_can_shift() const {
    // shifting the pos is trivial for recurrent models
    return true;
}

size_t llama_memory_recurrent::total_size() const {
    size_t size = 0;
    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_memory_recurrent::size_r_bytes() const {
    size_t size_r_bytes = 0;

    for (const auto & r : r_l) {
        if (r != nullptr) {
            size_r_bytes += ggml_nbytes(r);
        }
    }

    return size_r_bytes;
}

size_t llama_memory_recurrent::size_s_bytes() const {
    size_t size_s_bytes = 0;

    for (const auto & s : s_l) {
        if (s != nullptr) {
            size_s_bytes += ggml_nbytes(s);
        }
    }

    return size_s_bytes;
}

size_t llama_memory_recurrent::size_p_bytes() const {
    size_t size_p_bytes = 0;

    for (const auto & p : p_l) {
        if (p != nullptr) {
            size_p_bytes += ggml_nbytes(p);
        }
    }

    return size_p_bytes;
}

void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data; // logical source row ranges
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_s;    // [TAG_GDN_REPLAY] S rows: no rollback planes
    uint32_t cell_count = 0;

    // [TAG_GDN_REPLAY] a deferred S state is not a state yet: llama_context commits it first (gdn_commit), into a
    //   snapshot row that is written instead of the cell's
    for (size_t i = 0; i < replay_n.size(); ++i) {
        if (replay_n[i] == 0 || (seq_id != -1 && (llama_seq_id) i != seq_id)) {
            continue;
        }
        const int32_t tail = i < size ? cells[i].tail : -1;
        const uint32_t row = tail < 0 ? 0 : (cells[tail].src >= 0 ? (uint32_t) cells[tail].src : (uint32_t) tail);
        GGML_ASSERT(tail >= 0 && gdn_snap.count(row) && "gdn replay: state_write before the commit");
    }

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            const uint32_t cell_id = rs_idx_cur * size + (cell.src >= 0 ? cell.src : (int32_t) i);
            if (cell_ranges_data.empty() || cell_ranges_data.back().second != cell_id) {
                cell_ranges_data.emplace_back(cell_id, cell_id + 1);
            } else {
                cell_ranges_data.back().second++;
            }
            const uint32_t cell_id_s = gdn_replay ? cell_id - rs_idx_cur * size : cell_id;
            if (cell_ranges_s.empty() || cell_ranges_s.back().second != cell_id_s) {
                cell_ranges_s.emplace_back(cell_id_s, cell_id_s + 1);
            } else {
                cell_ranges_s.back().second++;
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        // throw instead of aborting: state_seq_get_size/get_data catch this and return 0
        throw std::runtime_error("cannot save/load multiple ranges of cells to/from device memory");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : cell_ranges_data) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges_data, cell_ranges_s);
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    // save the head of the restored cells - could be needed to clear the state
    // the head is valid only when state_read_meta() succeeded
    const bool meta_read = state_read_meta(io, cell_count, seq_id);
    const uint32_t cell_head = head;

    res = res && meta_read;

    try {
        res = res && state_read_data(io, cell_count);
    } catch (...) {
        res = false;
    }

    if (!res) {
        state_clear(seq_id, cell_head, meta_read ? cell_count : 0);
        throw std::runtime_error("failed to restore kv cache");
    }

    if (n_rs_seq != 0) {
        set_rs_idx(seq_id, 0);
        set_rs_valid_depth(seq_id, 0);
    }
    // [TAG_GDN_REPLAY] a restored S state is committed
    if (seq_id < 0) {
        std::fill(replay_n.begin(), replay_n.end(), 0);
    } else if ((uint32_t) seq_id < replay_n.size()) {
        replay_n[seq_id] = 0;
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges,
        const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges_s) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : cell_ranges) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * r_size_row;
            io.write_tensor(r_l[il], range.first * r_size_row, buf_size);
        }

        // the PLE conv history is a second recurrent row, so it has to travel with the first
        if (p_l[il] != nullptr) {
            const uint64_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            io.write(&p_size_row, sizeof(p_size_row));

            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                io.write_tensor(p_l[il], range.first * p_size_row, range_size * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write row size of S tensor
            const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            io.write(&s_size_row, sizeof(s_size_row));

            // Write each logical cell row range. With pending recurrent rollback,
            // the logical current state may live in a rollback snapshot plane ([TAG_GDN_REPLAY] S has none).
            for (const auto & range : cell_ranges_s) {
                if (!gdn_snap.empty()) {
                    // [TAG_GDN_REPLAY] a committed snapshot replaces its cell's row
                    for (uint32_t r = range.first; r < range.second; ++r) {
                        const auto it = gdn_snap.find(r);
                        if (it != gdn_snap.end()) {
                            io.write_tensor(it->second[il], 0, s_size_row);
                        } else {
                            io.write_tensor(s_l[il], r * s_size_row, s_size_row);
                        }
                    }
                    continue;
                }
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * s_size_row;
                io.write_tensor(s_l[il], range.first * s_size_row, buf_size);
            }
        }
    } else {
        // When S tensor is transposed, we also need the element size and get the element ranges from each row
        const uint32_t mem_size = size;
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Write S tensor type
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            io.write(&s_type_i, sizeof(s_type_i));

            // Write element size
            const uint32_t s_size_el = ggml_type_size(s_l[il]->type);
            io.write(&s_size_el, sizeof(s_size_el));

            // Write GQA embedding size
            io.write(&n_embd_s, sizeof(n_embd_s));

            // For each row, we get the element values of each logical cell
            for (uint32_t j = 0; j < n_embd_s; ++j) {
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * mem_size) * s_size_el;
                    const size_t buf_size = range_size * s_size_el;
                    io.write_tensor(s_l[il], src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence
        // checked before anything is allocated by cell_count (the ubatch below): a damaged count must not OOM
        if (cell_count > size) {
            io.set_status(LLAMA_STATE_READ_INVALID);
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 0) {
                io.set_status(LLAMA_STATE_READ_INVALID);
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (pos < 0 || pos == std::numeric_limits<llama_pos>::max()) { // [TAG_RESTORE_HARDEN] see llama_kv_cache
                io.set_status(LLAMA_STATE_READ_INVALID);
                LLAMA_LOG_ERROR("%s: invalid recurrent cell position %d\n", __func__, pos);
                return false;
            }

            ubatch.pos[i] = pos;
        }
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0] = &dest_seq_id;

        if (!find_slot(ubatch)) {
            io.set_status(LLAMA_STATE_READ_NO_SPACE);
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // kv.head should be our first cell, kv.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
        // Assume that this is one contiguous block of cells
        // [TAG_RESTORE_HARDEN] the stream decides cell_count and the positions: a damaged one is an invalid stream, not an abort
        if (head + cell_count > size ||
                cells[head].pos != ubatch.pos[0] ||
                cells[head + cell_count - 1].pos != ubatch.pos[cell_count - 1] ||
                !cells[head].has_seq_id(dest_seq_id) ||
                !cells[head + cell_count - 1].has_seq_id(dest_seq_id)) {
            io.set_status(LLAMA_STATE_READ_INVALID);
            LLAMA_LOG_ERROR("%s: restored cells do not match the stream (cell_count = %u)\n", __func__, cell_count);
            return false;
        }
    } else {
        // whole KV cache restore

        if (cell_count > size) {
            io.set_status(LLAMA_STATE_READ_INVALID);
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            auto & cell = cells[i];

            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cell.pos = pos;

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= this->n_seq_max) {
                    io.set_status(LLAMA_STATE_READ_INVALID);
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, this->n_seq_max);
                    return false;
                }

                cell.seq_id.insert(seq_id);

                int32_t & tail = cells[seq_id].tail;
                if (tail != -1) {
                    io.set_status(LLAMA_STATE_READ_INVALID);
                    LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                    return false;
                }
                tail = i;
            }
        }

        head = 0;
        used = cell_count;
    }

    for (uint32_t i = 0; i < cell_count; ++i) {
        uint32_t cell_id = head + i;
        // make sure the recurrent states will keep their restored state
        cells[cell_id].src = cell_id;
    }

    return true;
}

bool llama_memory_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        io.set_status(LLAMA_STATE_READ_INVALID);
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        io.set_status(LLAMA_STATE_READ_INVALID);
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) s_trans) {
        io.set_status(LLAMA_STATE_READ_INVALID);
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            io.set_status(LLAMA_STATE_READ_INVALID);
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            io.set_status(LLAMA_STATE_READ_INVALID);
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // Read and set the keys for the whole cell range
            io.read_tensor(r_l[il], head * r_size_row, cell_count * r_size_row);
        }

        if (p_l[il] != nullptr) {
            uint64_t p_size_row_ref;
            io.read(&p_size_row_ref, sizeof(p_size_row_ref));
            const size_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            if (p_size_row != p_size_row_ref) {
                io.set_status(LLAMA_STATE_READ_INVALID);
                LLAMA_LOG_ERROR("%s: mismatched ple row size (%zu != %zu, layer %d)\n", __func__, p_size_row, (size_t) p_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                io.read_tensor(p_l[il], head * p_size_row, cell_count * p_size_row);
            }
        }
    }

    if (!s_trans) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;

            if (s_type_i != s_type_i_ref) {
                io.set_status(LLAMA_STATE_READ_INVALID);
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t s_size_row_ref;
            io.read(&s_size_row_ref, sizeof(s_size_row_ref));
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            if (s_size_row != s_size_row_ref) {
                io.set_status(LLAMA_STATE_READ_INVALID);
                LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the values for the whole cell range
                io.read_tensor(s_l[il], head * s_size_row, cell_count * s_size_row);
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (uint32_t il = 0; il < n_layer; ++il) {
            // skip null layers
            if (s_l[il] == nullptr) continue;

            const uint32_t n_embd_s = hparams.n_embd_s();

            // Read type of value
            int32_t s_type_i_ref;
            io.read(&s_type_i_ref, sizeof(s_type_i_ref));
            const int32_t s_type_i = (int32_t)s_l[il]->type;
            if (s_type_i != s_type_i_ref) {
                io.set_status(LLAMA_STATE_READ_INVALID);
                LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t s_size_el_ref;
            io.read(&s_size_el_ref, sizeof(s_size_el_ref));
            const size_t s_size_el = ggml_type_size(s_l[il]->type);
            if (s_size_el != s_size_el_ref) {
                io.set_status(LLAMA_STATE_READ_INVALID);
                LLAMA_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, s_size_el, (size_t) s_size_el_ref, il);
                return false;
            }

            // Read state embedding size
            uint32_t n_embd_s_ref;
            io.read(&n_embd_s_ref, sizeof(n_embd_s_ref));
            if (n_embd_s != n_embd_s_ref) {
                io.set_status(LLAMA_STATE_READ_INVALID);
                LLAMA_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, n_embd_s, n_embd_s_ref, il);
                return false;
            }

            if (cell_count) {
                // For each row in the transposed matrix, read the values for the whole cell range
                for (uint32_t j = 0; j < n_embd_s; ++j) {
                    const size_t dst_offset = (head + j * size) * s_size_el;
                    io.read_tensor(s_l[il], dst_offset, cell_count * s_size_el);
                }
            }
        }
    }

    return true;
}

// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
// the transposed s layout is not handled - state_read_data() rejects it before any write
void llama_memory_recurrent::state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count) {
    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    if (seq_id == -1) {
        clear(true);
        return;
    }

    seq_rm(seq_id, -1, -1);

    if (cell_count == 0) {
        return;
    }

    const uint32_t n_layer = hparams.n_layer();

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (r_l[il] != nullptr) {
            const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
            llama_clear_tensor_data(r_l[il], cell_head * r_size_row, cell_count * r_size_row);
        }

        if (s_l[il] != nullptr) {
            const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
            llama_clear_tensor_data(s_l[il], cell_head * s_size_row, cell_count * s_size_row);
        }

        if (p_l[il] != nullptr) {
            const size_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            llama_clear_tensor_data(p_l[il], cell_head * p_size_row, cell_count * p_size_row);
        }
    }
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update
    if (ubatches.empty()) {
        // recurrent cache never performs updates
        assert(status == LLAMA_MEMORY_STATUS_NO_UPDATE);

        return true;
    }

    const auto & ubatch = ubatches[i_next];

    mem->find_slot(ubatch);

    // [TAG_RECURRENT_ROLLBACK_SPLITS]
    // this ubatch writes K_write = min(n_seq_tokens, rs_rollback_depth + 1) planes for each of its sequences
    //   (plane 0 = the final state), so K_write - 1 is the deepest rollback a later seq_rm may accept
    const uint32_t valid_depth = std::min(ubatch.rs_rollback_depth, ubatch.n_seq_tokens - 1);
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        mem->set_rs_valid_depth(ubatch.seq_id_unq[s], valid_depth);
    }

    // [TAG_GDN_REPLAY] the graph's control per sequence {n_prefix, read_half, n_defer, slot}: it replays the kept tokens,
    //   then keeps the inputs of the last valid_depth tokens (the rollback the planes would allow) instead of their
    //   states. A token shared by several sequences defers nothing (its cell has no single stash slot)
    if (mem->gdn_replay && ubatch.rs_rollback_depth == 0) {
        // the graph runs the planes path with K = 1 (no replay): llama_context committed its sequences first
        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
            for (int32_t j = 0; j < ubatch.n_seq_id[s * ubatch.n_seq_tokens]; ++j) {
                GGML_ASSERT(mem->replay_n[ubatch.seq_id[s * ubatch.n_seq_tokens][j]] == 0 && "gdn replay: depth-0 graph over a deferred state");
            }
        }
    } else if (mem->gdn_replay) {
        mem->gdn_ctl_cur.assign(4 * ubatch.n_seqs, 0);
        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
            const uint32_t i = s * ubatch.n_seq_tokens;
            const llama_seq_id seq = ubatch.seq_id[i][0];
            const uint32_t n_defer = ubatch.n_seq_id[i] == 1 ? std::min(valid_depth, mem->gdn_w) : 0;
            int32_t * c = mem->gdn_ctl_cur.data() + 4 * s;
            c[0] = (int32_t) mem->replay_n[seq];
            c[1] = (int32_t) mem->replay_half[seq];
            c[2] = (int32_t) n_defer;
            c[3] = (int32_t) seq;
            for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
                mem->replay_n[ubatch.seq_id[i][j]] = n_defer;
            }
            if (n_defer > 0) {
                mem->replay_half[seq] = 1 - mem->replay_half[seq];
            }
        }
    }

    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_memory_recurrent_context::get_n_rs() const {
    return is_full ? mem->size : mem->n;
}

uint32_t llama_memory_recurrent_context::get_head() const {
    return is_full ? 0 : mem->head;
}

int32_t llama_memory_recurrent_context::get_rs_z() const {
    return is_full ? 0 : mem->rs_z;
}

uint32_t llama_memory_recurrent_context::get_size() const {
    return mem->size;
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_p_l(int32_t il) const {
    return mem->p_l[il];
}

bool llama_memory_recurrent_context::get_gdn_replay() const {
    return mem->gdn_replay;
}

ggml_tensor * llama_memory_recurrent_context::get_st_l(int32_t il) const {
    return mem->st_l[il];
}

int32_t llama_memory_recurrent_context::s_copy_s(int i) const {
    return mem->cells[i + mem->head].src0;
}

void llama_memory_recurrent_context::set_gdn_ctl(int32_t * ctl) const {
    GGML_ASSERT(mem->gdn_ctl_cur.size() == 4 * get_ubatch().n_seqs);
    memcpy(ctl, mem->gdn_ctl_cur.data(), mem->gdn_ctl_cur.size() * sizeof(int32_t));
}

int32_t llama_memory_recurrent_context::s_copy(int i) const {
    const uint32_t cell_idx = i + mem->head;
    const int32_t  src0     = mem->cells[cell_idx].src0;

    if (mem->n_rs_seq == 0) {
        return src0;
    }

    uint32_t idx = 0;
    if (!mem->cells[cell_idx].seq_id.empty()) {
        const llama_seq_id seq = *mem->cells[cell_idx].seq_id.begin();
        if (seq >= 0 && (size_t) seq < mem->rs_idx.size()) {
            idx = mem->rs_idx[seq];
            // reset rollback idx
            mem->rs_idx[seq] = 0;
        }
    }
    return (int32_t)(idx * mem->size) + src0;
}

std::string llama_memory_recurrent::state_descriptor() const {
    // logical current row per sequence: R (n_embd_r), the PLE conv history P (ple_conv_state) when present, S (n_embd_s)
    // [TAG_MULTISLOT] the per-seq stream holds one logical row per sequence and no cell index: the row pool size
    //   (max(1, n_seq_max)) and n_seq_max print the single-slot values (1), like llama_kv_cache's descriptor
    std::string res = format("recr{size=%u,n_seq_max=%u,n_embd_r=%u,n_embd_s=%u,ple=%u,layers=[", 1u, 1u,
                                    hparams.n_embd_r(), hparams.n_embd_s(), hparams.ple_conv_state());
    for (size_t il = 0; il < r_l.size(); il++) {
        const auto td = [](const ggml_tensor * t) {
            return t ? format("%s/b%lld/s%zu", ggml_type_name(t->type), (long long) ggml_blck_size(t->type),
                                     ggml_type_size(t->type)) : std::string("none");
        };
        res += format("%zu:r=%s:s=%s:p=%s;", il, td(r_l[il]).c_str(), td(s_l[il]).c_str(),
                             td(il < p_l.size() ? p_l[il] : nullptr).c_str());
    }
    return res + "]}";
}
