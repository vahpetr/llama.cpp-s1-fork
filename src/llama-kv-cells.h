#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstring>
#include <map>
#include <set>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

struct llama_kv_cell_ext {
    // 2D spatial positions, typically used for M-RoPE
    llama_pos x = 0;
    llama_pos y = 0;

    // when tok = LLAMA_TOKEN_NULL when the cell is produced by embedding input (i.e. multimodal)
    // use case: n-gram embeddings hash
    llama_token tok = LLAMA_TOKEN_NULL;

    // return true if the current 2D spatial position is greater than other
    bool is_2d_gt(llama_pos ox, llama_pos oy) const {
        return (y > oy) || (y == oy && x > ox);
    }

    void reset() {
        static_assert(std::is_trivially_copyable_v<llama_kv_cell_ext>);

        *this = llama_kv_cell_ext{};
    }
};

// index of the lowest set bit, and of the highest; the word is never zero here
#if defined(_MSC_VER)
static inline uint32_t llama_kv_ctz64(uint64_t x) {
    unsigned long r;
    _BitScanForward64(&r, x);
    return (uint32_t) r;
}

static inline uint32_t llama_kv_clz64(uint64_t x) {
    unsigned long r;
    _BitScanReverse64(&r, x);
    return (uint32_t) r;
}
#else
static inline uint32_t llama_kv_ctz64(uint64_t x) {
    return (uint32_t) __builtin_ctzll(x);
}

static inline uint32_t llama_kv_clz64(uint64_t x) {
    return (uint32_t) (63 - __builtin_clzll(x));
}
#endif

// the cells tracked are a large fraction of the cache, so a bitmap is both smaller and faster
// to walk than a tree node per index
class llama_kv_idx_set {
public:
    void resize(uint32_t n) {
        bits.assign((n + 63)/64, 0);
        n_set = 0;
    }

    void clear() {
        std::fill(bits.begin(), bits.end(), 0);
        n_set = 0;
    }

    void insert(uint32_t i) {
        uint64_t & w = bits[i/64];
        const uint64_t b = 1ull << (i%64);

        n_set += (w & b) == 0;
        w |= b;
    }

    void erase(uint32_t i) {
        uint64_t & w = bits[i/64];
        const uint64_t b = 1ull << (i%64);

        n_set -= (w & b) != 0;
        w &= ~b;
    }

    bool contains(uint32_t i) const {
        return (bits[i/64] >> (i%64)) & 1;
    }

    uint32_t size()  const { return n_set; }
    bool     empty() const { return n_set == 0; }

    uint32_t first() const {
        for (size_t w = 0; w < bits.size(); ++w) {
            if (bits[w]) {
                return 64*w + llama_kv_ctz64(bits[w]);
            }
        }

        return 0;
    }

    uint32_t last() const {
        for (size_t w = bits.size(); w-- > 0; ) {
            if (bits[w]) {
                return 64*w + llama_kv_clz64(bits[w]);
            }
        }

        return 0;
    }

    // ascending order, as the set it replaces guaranteed
    template <class F>
    void for_each(F && f) const {
        for (size_t w = 0; w < bits.size(); ++w) {
            uint64_t m = bits[w];

            while (m) {
                f((uint32_t) (64*w + llama_kv_ctz64(m)));
                m &= m - 1;
            }
        }
    }

private:
    std::vector<uint64_t> bits;

    uint32_t n_set = 0;
};

// meta information about KV cells that can be part of multiple sequences at the same time
// [TAG_DISK_INCR] the cell modification clock: every llama_kv_cells mutator stamps the cells it touches with the current
//   value; a sequence-state capture ticks it first (llama_state_mod_clock_tick) and records the value, and a later
//   capture asks for the cells whose stamp is still below that value = not modified (metadata or rows) since the
//   earlier capture began. One process-wide clock so a value is comparable across caches and contexts (a cache created
//   after a capture is all-modified for it). Relaxed: the cells are mutated and captured on one thread
inline std::atomic<uint64_t> & llama_kv_mod_clock() {
    static std::atomic<uint64_t> clock{1};
    return clock;
}

// TODO: add unit tests
class llama_kv_cells {
public:
    using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;

    void reset() {
        vis_mark_all(); // [TAG_KQ_VIS_MIRROR]
        for (uint32_t i = 0; i < pos.size(); ++i) {
            pos[i]   = -1;
            ext[i].reset();
            shift[i] =  0;
            seq[i].reset();
            seq_lo[i] = 0;
        }

        has_shift = false;

        used.clear();

        for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
            seq_pos[s].clear();
        }
    }

    void reset_shift() {
        has_shift = false;

        for (uint32_t i = 0; i < shift.size(); ++i) {
            if (shift[i] != 0) {
                mod_mark(i); // [TAG_DISK_INCR] reset_shift follows the K-shift graph: the shifted rows changed now
            }
            shift[i] = 0;
        }
    }

    uint32_t size() const {
        return pos.size();
    }

    void resize(uint32_t n) {
        pos.resize(n);
        ext.resize(n);
        shift.resize(n);
        seq.resize(n);
        seq_lo.resize(n);
        used.resize(n);
        mod.resize(n);

        reset();
    }

    bool is_empty(uint32_t i) const {
        assert(i < pos.size());
        assert((pos[i] < 0 && pos[i] == -1) || pos[i] >= 0);

        return pos[i] == -1;
    }

    uint32_t get_used() const {
        return used.size();
    }

    // the index of the first cell that is used
    // return 0 if no cells are used
    uint32_t used_min() const {
        return used.empty() ? 0 : used.first();
    }

    // the index of the last cell that is used + 1
    // return 0 if no cells are used
    uint32_t used_max_p1() const {
        return used.empty() ? 0 : used.last() + 1;
    }

    bool get_has_shift() const {
        return has_shift;
    }

    // move cell isrc to idst (used during defrag)
    //void mv(uint32_t isrc, uint32_t idst) {
    //    assert(isrc < pos.size());
    //    assert(idst < pos.size());

    //    assert(pos[idst] == -1);
    //    assert(pos[isrc] != -1);

    //    pos  [idst] = pos  [isrc];
    //    shift[idst] = shift[isrc];
    //    seq  [idst] = seq  [isrc];

    //    pos  [isrc] = -1;
    //    shift[isrc] =  0;
    //    seq  [isrc].reset();

    //    used.erase (isrc);
    //    used.insert(idst);
    //}

    // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
    llama_kv_cells cp(uint32_t i, uint32_t n) const {
        assert(i + n <= pos.size());

        llama_kv_cells res;

        res.resize(n);

        for (uint32_t j = 0; j < n; ++j) {
            const auto idx = i + j;

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];
            res.seq_lo[j] = seq_lo[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    llama_kv_cells cp(const std::vector<uint32_t> & idxs) const {
        llama_kv_cells res;

        res.resize(idxs.size());

        for (uint32_t j = 0; j < idxs.size(); ++j) {
            const auto idx = idxs[j];

            res.pos[j] = pos[idx];
            res.ext[j] = ext[idx];
            res.seq[j] = seq[idx];
            res.seq_lo[j] = seq_lo[idx];

            assert(shift[idx] == 0);
        }

        return res;
    }

    // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
    void set(uint32_t i, const llama_kv_cells & other) {
        assert(i + other.pos.size() <= pos.size());

        vis_mark_range(i, i + (uint32_t) other.pos.size()); // [TAG_KQ_VIS_MIRROR]

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = i + j;

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(i + j);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(i + j);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(i + j);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];
            seq_lo[idx] = other.seq_lo[j];

            if (pos[idx] != -1) {
                seq_pos_add(i + j);
            }

            assert(shift[idx] == 0);
        }
    }

    // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
    void set(const std::vector<uint32_t> & idxs, const llama_kv_cells & other) {
        assert(idxs.size() == other.pos.size());

        for (uint32_t j = 0; j < other.pos.size(); ++j) {
            const auto idx = idxs[j];

            vis_mark(idx); // [TAG_KQ_VIS_MIRROR]

            if (pos[idx] == -1 && other.pos[j] != -1) {
                used.insert(idx);
            }

            if (pos[idx] != -1 && other.pos[j] == -1) {
                used.erase(idx);
            }

            if (pos[idx] != -1) {
                seq_pos_rm(idx);
            }

            pos[idx] = other.pos[j];
            ext[idx] = other.ext[j];
            seq[idx] = other.seq[j];
            seq_lo[idx] = other.seq_lo[j];

            if (pos[idx] != -1) {
                seq_pos_add(idx);
            }

            assert(shift[idx] == 0);
        }
    }

    // clear a non-empty cell
    void rm(uint32_t i) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        vis_mark(i); // [TAG_KQ_VIS_MIRROR]

        seq_pos_rm(i);
        seq[i].reset();
        seq_lo[i] = 0;

        pos[i] = -1;
        ext[i].reset();
        shift[i] = 0;

        used.erase(i);
    }

    // note: call only if the cell has seq_id
    // return true if the cell becomes empty
    bool seq_rm(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(seq[i].test(seq_id));
        assert(pos[i] != -1);
        assert(seq_id >= 0);

        vis_mark(i); // [TAG_KQ_VIS_MIRROR]
        seq[i].reset(seq_id);
        seq_lo[i] &= ~seq_lo_bit(seq_id);
        seq_pos_dec(seq_id, pos[i]);

        if (seq[i].none()) {
            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        return false;
    }

    // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
    bool seq_keep(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());

        vis_mark(i); // [TAG_KQ_VIS_MIRROR] (conservative: also when nothing changes - after a seq_keep the next update is a
                     // near-full partial write; correct, not on the prod path, C2 review M7)

        if (seq[i].test(seq_id)) {
            seq_pos_rm(i);
            seq[i].reset();
            seq_lo[i] = 0;

            seq[i].set(seq_id);
            seq_lo[i] |= seq_lo_bit(seq_id);
            seq_pos_inc(seq_id, pos[i]);

            return false;
        }

        if (seq[i].any()) {
            seq_pos_rm(i);
            seq[i].reset();
            seq_lo[i] = 0;

            pos[i] = -1;
            ext[i].reset();
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        assert(pos[i] == -1);

        return false;
    }

    // number of different sequences in the cell
    int seq_count(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return seq[i].count();
    }

    // two cells with the same set are visible to exactly the same sequences
    const seq_set_t & seq_set(uint32_t i) const {
        assert(i < pos.size());

        return seq[i];
    }

    // check if the cell contains seq_id
    bool seq_has(uint32_t i, llama_seq_id seq_id) const {
        assert(i < pos.size());
        assert(seq_id >= 0);

        return seq[i].test(seq_id);
    }

    // [TAG_KQ_VIS_FILL] sequence ids 0..31 of cell i as a bit mask (bit s = seq_has(i, s)), O(1)
    uint32_t seq_mask32(uint32_t i) const {
        assert(i < pos.size());
        assert(seq_lo[i] == seq_lo_of_bitset(i) && "seq_lo out of step with seq: a seq[] write without its seq_lo update");

        return seq_lo[i];
    }

    // [TAG_KQ_VIS_FILL] the visibility metadata of cells [0, n) as four planes (llama_kv_cells_fill_kq_vis):
    // c_pos = pos (-1 = empty), c_seq = seq_mask32, c_x / c_y = ext.x / ext.y; an empty cell gives {-1, 0, 0, 0}.
    // One pass, no branch per cell (the empty test is a select mask), no bitset access.
    // MASK_EMPTY = false exists only for test-kq-vis-fill-equiv's negative control (empty cells keep their stale ext).
    template <bool MASK_EMPTY = true>
    void get_vis(uint32_t n, int32_t * __restrict c_pos, int32_t * __restrict c_seq,
                             int32_t * __restrict c_x,   int32_t * __restrict c_y) const {
        assert(n <= pos.size());

#ifndef NDEBUG
        // rebase guard: every seq[] write must have its seq_lo twin (checked against the bitset itself)
        for (uint32_t j = 0; j < n; ++j) {
            assert(seq_lo[j] == seq_lo_of_bitset(j) && "seq_lo out of step with seq: a seq[] write without its seq_lo update");
        }
#endif

        const llama_pos         * __restrict p  = pos.data();
        const uint32_t          * __restrict sl = seq_lo.data();
        const llama_kv_cell_ext * __restrict e  = ext.data();

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos pj   = p[j];
            const uint32_t  keep = !MASK_EMPTY || pj != -1 ? ~0u : 0u;

            c_pos[j] = pj;
            c_seq[j] = (int32_t) (sl[j] & keep);
            c_x[j]   = (int32_t) ((uint32_t) e[j].x & keep);
            c_y[j]   = (int32_t) ((uint32_t) e[j].y & keep);
        }
    }

    // [TAG_KQ_VIS_MIRROR] get_vis restricted to cells [lo, hi): the same per-cell expressions, plane j - lo
    void get_vis_range(uint32_t lo, uint32_t hi, int32_t * __restrict c_pos, int32_t * __restrict c_seq,
                       int32_t * __restrict c_x, int32_t * __restrict c_y) const {
        assert(lo <= hi && hi <= pos.size());

        const llama_pos         * __restrict p  = pos.data();
        const uint32_t          * __restrict sl = seq_lo.data();
        const llama_kv_cell_ext * __restrict e  = ext.data();

        for (uint32_t j = lo; j < hi; ++j) {
            const llama_pos pj   = p[j];
            const uint32_t  keep = pj != -1 ? ~0u : 0u;

            c_pos[j - lo] = pj;
            c_seq[j - lo] = (int32_t) (sl[j] & keep);
            c_x[j - lo]   = (int32_t) ((uint32_t) e[j].x & keep);
            c_y[j - lo]   = (int32_t) ((uint32_t) e[j].y & keep);
        }
    }

    // [TAG_KQ_VIS_MIRROR] per-consumer dirty ranges of the visibility metadata (pos / seq_lo / ext): every mutator marks
    // the cells it touches; a device mirror registers a slot and takes (reads + clears) its range before an update.
    // A fresh slot, reset() and resize() mark everything. Up to 4 consumers (cells shared between caches).
    // a slot is reclaimed by vis_unregister (the consumer's destructor): at most 4 consumers ALIVE at a time (pkg2 review, Minor)
    // not synchronised (vis_used / n_vis / vis_dirty are plain mutable members): creating / destroying the caches that share
    // one llama_kv_cells must be serialised by the caller, as every other use of shared cells already is (pkgfix review, Minor 2)
    int vis_register() const {
        for (int k = 0; k < (int) vis_dirty.size(); ++k) {
            if (!(vis_used & (1u << k))) {
                vis_used |= 1u << k;
                vis_dirty[k] = vis_range_t();
                n_vis = std::max(n_vis, k + 1);
                return k;
            }
        }
        GGML_ABORT("KQ visibility mirror: more than 4 live consumers of one llama_kv_cells");
    }

    void vis_unregister(int slot) const {
        GGML_ASSERT(slot >= 0 && slot < (int) vis_dirty.size() && (vis_used & (1u << slot)));
        vis_used &= ~(1u << slot);
    }

    void vis_take(int slot, uint32_t & lo, uint32_t & hi, bool & all) const {
        assert(slot >= 0 && slot < n_vis && (vis_used & (1u << slot)));
        lo  = vis_dirty[slot].lo;
        hi  = vis_dirty[slot].hi;
        all = vis_dirty[slot].all;
        vis_dirty[slot] = vis_range_t{ UINT32_MAX, 0, false };
    }

    // gather the token ids of the cells in `seqs` with position in [p0, p1)
    // the callback receives (seq_id, pos, token) for every such (cell, seq) pair
    // note: used by n-gram input embeddings to recover the tokens preceding a ubatch
    template<typename F>
    void for_each_token_in(const std::bitset<LLAMA_MAX_SEQ> & seqs, llama_pos p0, llama_pos p1, F && f) const {
        used.for_each([&](uint32_t i) {
            if (pos[i] < p0 || pos[i] >= p1) {
                return;
            }

            const auto m = seq[i] & seqs;

            // a cell carries a handful of sequences at most, so stop once they are all seen
            // instead of walking the whole LLAMA_MAX_SEQ width
            size_t left = m.count();

            for (llama_seq_id s = 0; left > 0 && s < (llama_seq_id) LLAMA_MAX_SEQ; ++s) {
                if (m.test(s)) {
                    f(s, pos[i], ext[i].tok);
                    --left;
                }
            }
        });
    }

    // note: call only if the cell is not empty and the seq_id is not in the cell
    void seq_add(uint32_t i, llama_seq_id seq_id) {
        assert(i < pos.size());
        assert(pos[i] != -1);
        assert(!seq[i].test(seq_id));

        vis_mark(i); // [TAG_KQ_VIS_MIRROR]
        seq[i].set(seq_id);
        seq_lo[i] |= seq_lo_bit(seq_id);
        seq_pos_inc(seq_id, pos[i]);
    }

    // return the sequence id of this cell
    // note: call only for cells with exactly one sequence
    llama_seq_id seq_get(uint32_t i) const {
        assert(seq[i].count() == 1);

        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                return s;
            }
        }

        return -1;
    }

    // the minimum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_min(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        assert(seq_pos[seq_id].begin()->second > 0);

        return seq_pos[seq_id].begin()->first;
    }

    // [TAG_MULTISLOT] the number of distinct positions of sequence seq_id currently present (0 if absent): a complete
    // history 0..max has max + 1 of them (the [qsa-history] sequence-coverage query; O(1), sequence-owned)
    uint32_t seq_pos_count(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        return (uint32_t) seq_pos[seq_id].size();
    }

    // [TAG_T2_SEQ_RM_COUNTED] (T2 step 1, LLAMA_SPEC_OVERLAP=1) n = the number of cells of seq_id whose position lies
    //   in [p0, p1), read from seq_pos (every such cell holds exactly one count of its position). Returns false when
    //   more than max_keys distinct positions lie in the range: the caller then scans as before (walking a long
    //   map would cost more than the scan it replaces). O(max_keys + log n).
    bool seq_pos_count_in(llama_seq_id seq_id, llama_pos p0, llama_pos p1, uint32_t max_keys, uint32_t & n) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        n = 0;
        uint32_t n_keys = 0;
        for (auto it = seq_pos[seq_id].lower_bound(p0); it != seq_pos[seq_id].end() && it->first < p1; ++it) {
            if (++n_keys > max_keys) {
                return false;
            }
            n += (uint32_t) it->second;
        }

        return true;
    }

    // [TAG_T2_SEQ_RM_COUNTED] remove seq_id from its n cells with a position in [p0, p1) (n from seq_pos_count_in):
    //   exactly the cells the ascending full scan of llama_kv_cache::seq_rm visits, found from the top of the used
    //   range downwards and stopped once all n are done. A decoding sequence's newest cells sit at the top, so the
    //   tail of a speculative round costs O(gap + n) instead of O(size()). Every per-cell effect of seq_rm(i) (seq
    //   bit, seq_lo, seq_pos count, used set, vis_dirty min/max) does not depend on the visiting order. Returns the
    //   lowest index of a cell that became empty (size() if none) = the cell the ascending scan reports first.
    uint32_t seq_rm_counted(llama_seq_id seq_id, llama_pos p0, llama_pos p1, uint32_t n) {
        assert(seq_id >= 0);
        assert(p0 >= 0);

        uint32_t new_head = size();

        for (uint32_t i = n > 0 ? used_max_p1() : 0; i-- > 0 && n > 0; ) {
            if (!pos_in(i, p0, p1) || !seq_has(i, seq_id)) {
                continue;
            }
            --n;
            if (seq_rm(i, seq_id)) {
                new_head = i; // descending: the last one set is the lowest
            }
        }

        assert(n == 0);

        return new_head;
    }

    // the maximum position of sequence seq_id currently present in any of the cells
    // return -1 if the sequence is not present
    llama_pos seq_pos_max(llama_seq_id seq_id) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        if (seq_pos[seq_id].empty()) {
            return -1;
        }

        assert(seq_pos[seq_id].rbegin()->second > 0);

        return seq_pos[seq_id].rbegin()->first;
    }

    // note: call only if the cell is not empty
    llama_pos pos_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return pos[i];
    }

    const llama_kv_cell_ext & ext_get(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return ext[i];
    }

    // note: call only if the cell is not empty
    llama_pos get_shift(uint32_t i) const {
        assert(i < pos.size());
        assert(pos[i] != -1);

        return shift[i];
    }

    // check if a cell is not empty and its position is within [p0, p1)
    bool pos_in(uint32_t i, llama_pos p0, llama_pos p1) const {
        assert(i < pos.size());

        return pos[i] >= p0 && pos[i] < p1;
    }

    // set the position of an empty cell
    // does not modify "has_shift"
    // note: call only if the cell is empty
    void pos_set(uint32_t i, llama_pos p) {
        assert(i < pos.size());
        assert(pos[i] == -1);
        assert(seq[i].none());

        vis_mark(i); // [TAG_KQ_VIS_MIRROR]
        pos[i] = p;

        used.insert(i);
    }

    void ext_set(uint32_t i, llama_kv_cell_ext p) {
        assert(i < ext.size());
        vis_mark(i); // [TAG_KQ_VIS_MIRROR]
        ext[i] = p;
    }

    // pos[i] = pos[i] + d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    bool pos_add(uint32_t i, llama_pos d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        vis_mark(i); // [TAG_KQ_VIS_MIRROR]
        seq_pos_rm(i);

        pos[i]   += d;
        shift[i] += d;

        has_shift = true;

        if (pos[i] < 0) {
            seq[i].reset();
            seq_lo[i] = 0;
            pos[i] = -1;
            shift[i] = 0;

            used.erase(i);

            return true;
        }

        seq_pos_add(i);

        return false;
    }

    // pos[i] = pos[i] / d
    // sets "has_shift" to true
    // note: call only if the cell is not empty
    void pos_div(uint32_t i, int d) {
        assert(i < pos.size());
        assert(pos[i] != -1);

        const llama_pos p_old = pos[i];

        vis_mark(i); // [TAG_KQ_VIS_MIRROR]

        seq_pos_rm(i);

        pos[i]   /= d;
        shift[i] += p_old - pos[i];

        seq_pos_add(i);

        has_shift = true;
    }

    // [TAG_DISK_INCR] the rows of every cell changed now (data written outside the mutators, e.g. a stream copy)
    void mod_mark_all() const {
        mod_floor = llama_kv_mod_clock().load(std::memory_order_relaxed);
    }

    // [TAG_DISK_INCR] the number of leading cells of [i0, i1) not modified since the clock value `since` (0 = none)
    uint32_t n_unmod_prefix(uint32_t i0, uint32_t i1, uint64_t since) const {
        if (since == 0 || mod_floor >= since) {
            return 0;
        }
        uint32_t i = i0;
        while (i < i1 && i < mod.size() && mod[i] < since) {
            ++i;
        }
        return i - i0;
    }

private:
    // [TAG_DISK_INCR] per-cell modification stamps (llama_kv_mod_clock); a cell is modified at max(mod[i], mod_floor)
    mutable std::vector<uint64_t> mod;
    mutable uint64_t mod_floor = 0;

    void mod_mark(uint32_t i) const {
        if (i < mod.size()) {
            mod[i] = llama_kv_mod_clock().load(std::memory_order_relaxed);
        }
    }

    // [TAG_KQ_VIS_MIRROR]
    struct vis_range_t { uint32_t lo = UINT32_MAX; uint32_t hi = 0; bool all = true; };
    mutable std::array<vis_range_t, 4> vis_dirty;
    mutable int n_vis = 0;        // slots [0, n_vis) are marked (a free slot is marked harmlessly; registering resets it)
    mutable uint32_t vis_used = 0; // live consumers

    void vis_mark(uint32_t i) const {
        mod_mark(i); // [TAG_DISK_INCR] every mutator marks here (or in the two below)
        for (int k = 0; k < n_vis; ++k) {
            vis_dirty[k].lo = std::min(vis_dirty[k].lo, i);
            vis_dirty[k].hi = std::max(vis_dirty[k].hi, i + 1);
        }
    }
    void vis_mark_range(uint32_t i0, uint32_t i1) const {
        if (i0 >= i1) {
            return;
        }
        for (uint32_t i = i0; i < i1; ++i) {
            mod_mark(i); // [TAG_DISK_INCR]
        }
        for (int k = 0; k < n_vis; ++k) {
            vis_dirty[k].lo = std::min(vis_dirty[k].lo, i0);
            vis_dirty[k].hi = std::max(vis_dirty[k].hi, i1);
        }
    }
    void vis_mark_all() const {
        mod_mark_all(); // [TAG_DISK_INCR]
        for (int k = 0; k < n_vis; ++k) {
            vis_dirty[k].all = true;
        }
    }

    bool has_shift = false;

    // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
    llama_kv_idx_set used;

    std::vector<llama_pos> pos;

    // stores extra info per cell
    std::vector<llama_kv_cell_ext> ext;

    // this array accumulates any applied shifts to the pos array since the last reset_shift() call
    // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
    //
    //   cells.pos_add(x, shift_x);
    //   cells.pos_div(y, shift_y);
    //   ...
    //
    //   if (cells.has_shift()) {
    //      for (int i = 0; i < n; ++i) {
    //          auto shift_i = cells.get_shift(i);
    //          ...
    //      }
    //      cells.reset_shift();
    //   }
    //
    std::vector<llama_pos> shift;

    // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
    std::vector<seq_set_t> seq;

    // [TAG_KQ_VIS_FILL] seq_lo[i] = the low 32 bits of seq[i] (sequence ids 0..31), updated next to every write of
    // seq[i] above, so the visibility metadata fill reads a cell's seq mask as one word (not 32 bitset tests, and
    // without relying on the std::bitset object layout)
    std::vector<uint32_t> seq_lo;

    static uint32_t seq_lo_bit(llama_seq_id s) {
        return s >= 0 && s < 32 ? 1u << s : 0u;
    }

    // the low 32 bits of seq[i] from the bitset itself (mask + to_ullong, no layout reliance): the debug check of seq_lo
    uint32_t seq_lo_of_bitset(uint32_t i) const {
        return (uint32_t) (seq[i] & seq_set_t(0xFFFFFFFFull)).to_ullong();
    }

    // the set seq_pos[s][p] tells us how many times the position p is currently present for sequence s
    // if the position p is not present, seq_pos[s][p] is not set
    // this way seq_pos[s].begin() and seq_pos[s].rbegin() give us the min/max positions currently in the cache
    //
    // note that we cannot a use an std::set because in some cases a position can occur more than once for the same seq:
    //  - during performing a cache reuse via (rm + add)
    //  - some vision models have input embeddings with repeating positions
    //
    std::map<llama_pos, int> seq_pos[LLAMA_MAX_SEQ];

    // helper functions for updating `seq_pos`, once cell at a time:

    void seq_pos_dec(llama_seq_id s, llama_pos p) {
        auto it = seq_pos[s].find(p);
        assert(it != seq_pos[s].end());

        if (--it->second == 0) {
            seq_pos[s].erase(it);
        }
    }

    void seq_pos_inc(llama_seq_id s, llama_pos p) {
        seq_pos[s][p]++;
    }

    // remove cell i
    void seq_pos_rm(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_dec(s, pos[i]);
            }
        }
    }

    // add cell i
    void seq_pos_add(uint32_t i) {
        for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
            if (seq[i].test(s)) {
                seq_pos_inc(s, pos[i]);
            }
        }
    }
};

using llama_kv_cells_vec = std::vector<llama_kv_cells>;
