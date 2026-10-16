// [TAG_QSA_CENSUS_INC] FROZEN reference of the QSA block census and the pooled-key store plan as of prod-v11-a4r @ 5669f1de7
// (src/llama-memory-hybrid-idx.cpp: llama_memory_hybrid_idx_context::qsa_census, qsa_fold / qsa_mix / qsa_store_cap,
// qsa_store_plan, llama_qsa_pooled_store::reset_stream / reset_all / on_write / publish), copied verbatim except for the
// inputs: the cells and the cache stream of each ubatch stream are parameters instead of the memory + ubatch, the
// LLAMA_QSA_WHOLE_BLOCKS knob is a parameter, the plan cache / pre-publish / timing are the caller's.
// tests/test-qsa-census-inc.cpp compares the incremental (LLAMA_QSA_CENSUS_INC=1) and the refactored full path against it.
// DO NOT EDIT to follow later changes of the production code: it is the oracle.
#pragma once

#include "ggml.h"

#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace qsa_census_ref {

struct census_stream {
    int32_t n_bid  = 0;
    bool    ranked = false;
    std::vector<int32_t> bid_idx;
    std::vector<int32_t> bid_cell;
    std::vector<int32_t> order;
    std::vector<int32_t> rank;
    std::vector<int32_t> blk_of;
};

struct store {
    uint32_t ratio    = 0;
    uint32_t kv_size  = 0;
    uint32_t n_stream = 0;
    uint32_t n_slots  = 0;
    uint32_t n_rows   = 0;

    struct stream_state {
        std::vector<uint32_t> gen;
        std::vector<int32_t>  slot_of_key;
        std::vector<int32_t>  slot_key;
        std::vector<uint64_t> slot_stamp;
        std::vector<uint32_t> slot_seen;
        std::vector<int32_t>  free_slots;
    };

    std::vector<stream_state> strm;

    uint32_t serial = 0;

    struct pending_t {
        uint32_t s;
        int32_t  slot;
        uint64_t stamp;
    };

    std::vector<pending_t> pending;
    std::vector<uint32_t>  pending_streams;
    bool                   pending_armed = false;

    struct stats_t {
        uint64_t n_plan        = 0;
        uint64_t n_full        = 0;
        uint64_t n_dirty       = 0;
        uint64_t n_valid       = 0;
        uint64_t n_reset       = 0;
        uint64_t n_publish     = 0;
        uint64_t n_fail        = 0;
        uint64_t t_plan_us     = 0;
        uint32_t last_dirty    = 0;
        uint32_t last_cap      = 0;
        uint32_t last_n_bid    = 0;
    } stats;

    void reset_stream(uint32_t s);
    void reset_all();
    void on_write(const llama_kv_cache::slot_info & sinfo);
    void publish(bool ok);

    void init(uint32_t ratio_, uint32_t kv_size_, uint32_t n_stream_) {
        ratio    = ratio_;
        kv_size  = kv_size_;
        n_stream = n_stream_;
        n_slots  = (kv_size + ratio - 1)/ratio;
        n_rows   = n_slots + 1;
        strm.resize(n_stream);
        for (auto & ss : strm) {
            ss.gen        .assign(kv_size, 0);
            ss.slot_of_key.assign(kv_size, -1);
            ss.slot_key   .assign(n_slots, -1);
            ss.slot_stamp .assign(n_slots,  0);
            ss.slot_seen  .assign(n_slots,  0);
        }
        reset_all();
        stats.n_reset = 0;
    }
};

struct plan_t {
    bool     valid    = false;
    uint32_t ratio    = 0;
    int64_t  n_kv     = 0;
    int64_t  n_blocks = 0;
    int64_t  n_ns     = 0;
    bool     blk_bias = false;
    int64_t  cap      = 0;
    bool     full     = false;
    uint32_t s0       = 0;

    std::vector<int32_t> cell_blk;
    std::vector<int32_t> blk_cells;
    std::vector<int32_t> blk_pos;
    std::vector<census_stream> cs;

    std::vector<int32_t> dirty_cells;
    std::vector<int32_t> dirty_pos;
    std::vector<int32_t> dirty_dst;
    std::vector<int32_t> remap;

    std::vector<std::vector<uint64_t>> stamps; // [n_ns][n_bid]
};

inline void store::reset_stream(uint32_t s) {
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

inline void store::reset_all() {
    for (uint32_t s = 0; s < n_stream; ++s) {
        reset_stream(s);
    }
}

inline void store::on_write(const llama_kv_cache::slot_info & sinfo) {
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & gen = strm[sinfo.strm[s]].gen;

        for (const auto idx : sinfo.idxs[s]) {
            gen[idx]++;
        }
    }
}

inline void store::publish(bool ok) {
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


inline void census(
        const llama_kv_cells & cells,
        bool is_pos_2d,
        bool whole_blocks,
        int64_t s,
        int64_t n_tps_unused,
        int64_t n_kv,
        int64_t n_ns,
        int64_t n_blocks,
        int64_t r,
        bool blk_bias,
        int32_t * cur_cell_blk,
        int32_t * cur_blk_cells,
        int32_t * dst_blk_pos,
        census_stream & cs) {
    (void) n_tps_unused;

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

    // a cell no block covers needs its own -inf, which a per-block bias cannot carry
    // every cache path keeps the position below the cell window, so this stays false
    bool oor = false;

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

    cs.n_bid = n_bid;

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

    for (int64_t j = 0; j < n_kv; ++j) {
        const int32_t g = cell_grp[j];

        blk_of[j] = g < 0 ? -1 : grp_bid[g];

        if (blk_of[j] >= 0) {
            const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

            cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
        }

        cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
    }

    // [TAG_QSA_WHOLE_BLOCKS] D6 (the indexer reference notes): the spare id above is
    // boosted for EVERY query, so every unpooled visible cell is selected outside the block budget - right for the
    // tail (the newest incomplete group of a sequence), wrong for a FRAGMENT: an incomplete group in an older bucket
    // (a block straddling a seq_cp fork point, a position hole), which the reference never forms because it groups
    // each query's own visible list. In the whole-block form the budget is the reference's (n_sel blocks + the
    // tail), so a fragment - an unpooled cell that is the tail of NONE of its sequences - goes to a second unused
    // id, which both bias fills score -inf (b > n_bid). A cell that is the tail of one of its sequences and a
    // fragment for another keeps the boost (a per-block bias cannot tell the two queries apart; <= r - 1 cells per
    // such group); so does every fragment when no second unused id exists (n_bid + 1 == n_blocks: the cache is full
    // to the last block). With one sequence and no position holes there is no fragment and nothing changes.
    if (whole_blocks && n_bid < n_blocks) {
        const int64_t newest_rank = ranked ? (int64_t) order.size() - 1 : -1;

        int64_t n_unrouted = 0;

        for (int64_t j = 0; j < n_kv; ++j) {
            if (blk_of[j] >= 0 || cells.is_empty(j)) {
                continue;
            }

            const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

            bool tail_of_some = false;
            if (ranked) {
                tail_of_some = idx/r == newest_rank/r;
            } else {
                const auto & set = cells.seq_set((uint32_t) j);
                for (int32_t sq = 0; sq < LLAMA_MAX_SEQ && !tail_of_some; ++sq) {
                    tail_of_some = set.test(sq) && idx/r == (int64_t) cells.seq_pos_max(sq)/r;
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
                fprintf(stderr, "%s: LLAMA_QSA_WHOLE_BLOCKS=1: %" PRId64 " fragment cell(s) keep the tail boost: no second "
                               "unused block id (n_bid %d + 1 == n_blocks %" PRId64 "); warned once\n",
                               __func__, n_unrouted, n_bid, n_blocks);
            }
        }
    }
}


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
static inline int64_t qsa_store_cap(int64_t need, int64_t n_blocks) {
    int64_t cap = 8;
    while (cap < need) {
        cap *= 2;
    }
    return std::min<int64_t>(cap, n_blocks);
}

// the plan of one ubatch (qsa_store_plan without the per-ubatch cache, the pre-publish and the timing, which the caller does)
inline int64_t plan(store * st, const std::vector<const llama_kv_cells *> & cells_of, const std::vector<uint32_t> & g_of,
                    bool is_pos_2d, bool whole_blocks, uint32_t ratio, int64_t n_kv, bool blk_bias, plan_t & pl) {
    GGML_ASSERT(st != nullptr && ratio == st->ratio);

    const int64_t r        = ratio;
    const int64_t n_blocks = (n_kv + r - 1)/r;

    const int64_t n_ns     = (int64_t) cells_of.size();
    const int64_t n_tps    = 1;

    pl.valid    = false;
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

        census(*cells_of[s], is_pos_2d, whole_blocks, s, n_tps, n_kv, n_ns, n_blocks, r, blk_bias, cur_cell_blk, cur_blk_cells, pl.blk_pos.data(), cs);

        const uint32_t     g             = g_of[s];
        const auto &       cells         = *cells_of[s];

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

    pl.valid = true;

    pl.stamps = stamps;

    st->stats.n_plan++;
    st->stats.n_full     += full_pool;
    st->stats.n_dirty    += full_pool ? n_bid_all : n_dirty_all;
    st->stats.n_valid    += full_pool ? 0 : n_bid_all - n_dirty_all;
    st->stats.last_dirty  = full_pool ? n_bid_all : n_dirty_all;
    st->stats.last_cap    = (uint32_t) cap;
    st->stats.last_n_bid  = n_bid_all;

    return cap;
}


} // namespace qsa_census_ref
