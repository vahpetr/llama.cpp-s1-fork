// [TAG_QSA_CENSUS_INC] randomized equivalence test of the incremental QSA census + store plan (LLAMA_QSA_CENSUS_INC=1):
//   inc : llama_qsa_plan_inc  (src/llama-memory-hybrid-idx.cpp) on its own llama_qsa_pooled_store, fed the mutator marks
//   full: llama_qsa_plan_full (the refactored full path)        on its own llama_qsa_pooled_store
//   ref : qsa_census_ref::plan (tests/qsa-census-ref.h, the FROZEN pre-change code) on its own store
// All three plan the same llama_kv_cells after the same random mutator sequence and are compared after EVERY plan, element
// for element (llama_qsa_plan_compare): cell_blk, blk_cells, blk_pos, remap, the census stream, dirty_cells/pos/dst, cap,
// every block's stamp, slot_key / slot_of_key / slot_stamp / free_slots of the planned streams, the pending stamps in
// order, the serial; plus the inc census state's invariants (llama_qsa_census_state_check).
// Mutators, as llama_memory_hybrid_idx and its context apply them (the marks are the same calls the wrappers make):
//   writes of ubatches of 1..20 tokens (decode / verify / chunked prefill), MTP rollback seq_rm(p > accepted), partial seq_rm
//   (middle of the tail, across block boundaries, holes), cell reuse after seq_rm, overwrite of a used cell (SWA-style, the
//   pre-write mark), seq_cp within a stream and across streams, seq_keep, seq_add / seq_div shifts, clear, state_read restore
//   (invalidate + cells replaced), FAILED / partial state_read (invalidate + half the cells, then state_drop), runs of writes
//   with no plan in between (dense below LLAMA_QSA_MIN_KV, then QSA again), M-RoPE image positions (repeated positions with
//   2-D ext -> ranked census), 1-4 streams, 1-3 sequences in a unified stream, n_kv paddings 8/16/32 (bucket / stride
//   changes), replans of one ubatch at a larger n_kv, every publish outcome (armed ok, unarmed ok, failed, none).
// Negative controls (each in a child process): drop one mutator's mark -> the comparison (or an internal assert) must fail.
// Bench mode: µs per plan, full vs incremental, on a synthetic single-sequence decode with MTP rollbacks.
//
// usage: test-qsa-census-inc [n_seq = 10000] [seed = 42] [neg | bench [n_cells ...]]

#include "ggml.h"

#include "qsa-census-ref.h"

#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

struct rng_t {
    std::mt19937_64 g;
    explicit rng_t(uint64_t seed) : g(seed) {}
    uint64_t u(uint64_t n) { return n == 0 ? 0 : std::uniform_int_distribution<uint64_t>(0, n - 1)(g); }
    int64_t  r(int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(g); }
    bool     p(double x) { return std::uniform_real_distribution<double>(0.0, 1.0)(g) < x; }
};

// which mark a negative control drops
enum drop_t { DROP_NONE, DROP_RM, DROP_WRITE, DROP_PREWRITE, DROP_CP, DROP_KEEP, DROP_ADD, DROP_RESET, DROP_COUNT };
static const char * drop_name[DROP_COUNT] = { "none", "seq_rm", "write", "overwrite", "seq_cp", "seq_keep", "seq_add/div", "invalidate" };

// the seq_cp mark is defence in depth: a same-stream seq_cp puts a second sequence on the cells, which the fast path's
// live one-sequence check already refuses, and a cross-stream one resets the stream - so dropping it cannot be detected
static bool drop_redundant(int d) { return d == DROP_CP; }

static drop_t g_drop = DROP_NONE;

// [TAG_QSA_INPUTS_INC] negative control of the input mirrors: drop the change hints before every mirror update
static bool g_drop_mir = false;

// [TAG_QSA_INPUTS_INC] the device mirrors as host arrays (poisoned, so a missing write shows)
struct mirror_sim : llama_qsa_mirror_writer {
    std::vector<int32_t> a[llama_qsa_pooled_store::MIR_COUNT];
    uint64_t bytes = 0;
    void init(uint32_t kv_size, uint32_t n_slots, uint32_t n_stream) {
        a[llama_qsa_pooled_store::MIR_CELL_BLK].assign((size_t) kv_size*n_stream, 0x5A5A5A5A);
        a[llama_qsa_pooled_store::MIR_REMAP]   .assign((size_t) n_slots*n_stream, 0x5A5A5A5A);
        a[llama_qsa_pooled_store::MIR_BTAB]    .assign((size_t) 2*n_slots*n_stream, 0x5A5A5A5A);
    }
    void write(int arr, int64_t off, const int32_t * src, int64_t n) override {
        if (off < 0 || n < 0 || (size_t) (off + n) > a[arr].size()) {
            fprintf(stderr, "mirror write out of range: arr %d off %lld n %lld size %zu\n", arr, (long long) off,
                    (long long) n, a[arr].size());
            abort();
        }
        memcpy(a[arr].data() + off, src, (size_t) n*sizeof(int32_t));
        bytes += (uint64_t) n*sizeof(int32_t);
    }
};

// the mirrors after an update vs the full fill of plan `pl` (cell_blk / remap views, the block table from the census);
// "" when identical
static std::string mirror_compare(const mirror_sim & m, const llama_qsa_plan & pl,
                                  const std::vector<const llama_kv_cells *> & cells, bool tail_fix, bool with_remap) {
    const int64_t n_kv = pl.n_kv, n_blocks = pl.n_blocks;
    std::vector<int32_t> ref;
    char buf[256];
    for (int64_t s = 0; s < pl.n_ns; ++s) {
        for (int64_t j = 0; j < n_kv; ++j) {
            if (m.a[llama_qsa_pooled_store::MIR_CELL_BLK][s*n_kv + j] != pl.v_cell_blk[s][j]) {
                snprintf(buf, sizeof(buf), "mirror cell_blk s %lld [%lld]: %d vs full %d", (long long) s, (long long) j,
                         m.a[llama_qsa_pooled_store::MIR_CELL_BLK][s*n_kv + j], pl.v_cell_blk[s][j]);
                return buf;
            }
        }
        if (with_remap) {
            for (int64_t b = 0; b < n_blocks; ++b) {
                if (m.a[llama_qsa_pooled_store::MIR_REMAP][s*n_blocks + b] != pl.v_remap[s][b]) {
                    snprintf(buf, sizeof(buf), "mirror remap s %lld [%lld]: %d vs full %d", (long long) s, (long long) b,
                             m.a[llama_qsa_pooled_store::MIR_REMAP][s*n_blocks + b], pl.v_remap[s][b]);
                    return buf;
                }
            }
        }
        ref.assign(2*n_blocks, 0);
        llama_qsa_fill_blk_tab(*cells[s], *pl.v_cs[s], n_blocks, 0, n_blocks, tail_fix, ref.data());
        for (int64_t i = 0; i < 2*n_blocks; ++i) {
            if (m.a[llama_qsa_pooled_store::MIR_BTAB][s*2*n_blocks + i] != ref[i]) {
                snprintf(buf, sizeof(buf), "mirror qsa_blk s %lld [%lld]: %d vs full %d", (long long) s, (long long) i,
                         m.a[llama_qsa_pooled_store::MIR_BTAB][s*2*n_blocks + i], ref[i]);
                return buf;
            }
        }
    }
    return "";
}

static void store_init(llama_qsa_pooled_store & st, uint32_t r, uint32_t kv_size, uint32_t n_stream, bool inc) {
    st.ratio    = r;
    st.idx_dim  = 1;
    st.kv_size  = kv_size;
    st.n_stream = n_stream;
    st.n_slots  = (kv_size + r - 1)/r;
    st.n_rows   = st.n_slots + 1;
    st.strm.assign(n_stream, llama_qsa_pooled_store::stream_state());
    for (auto & ss : st.strm) {
        ss.gen        .assign(kv_size, 0);
        ss.slot_of_key.assign(kv_size, -1);
        ss.slot_key   .assign(st.n_slots, -1);
        ss.slot_stamp .assign(st.n_slots,  0);
        ss.slot_seen  .assign(st.n_slots,  0);
    }
    st.inc = inc;
    st.mir_on = inc;   // [TAG_QSA_INPUTS_INC] the inc store also keeps the mirror hints (no device tensors here)
    st.reset_all();
    st.stats = llama_qsa_pooled_store::stats_t();
}

// the frozen reference plan + store, as a llama_qsa_plan / llama_qsa_pooled_store for llama_qsa_plan_compare
static void ref_to_plan(const qsa_census_ref::plan_t & a, llama_qsa_plan & pl) {
    pl = llama_qsa_plan();
    pl.valid = a.valid; pl.ratio = a.ratio; pl.n_kv = a.n_kv; pl.n_blocks = a.n_blocks; pl.n_ns = a.n_ns;
    pl.blk_bias = a.blk_bias; pl.cap = a.cap; pl.full = a.full; pl.s0 = a.s0;
    pl.cell_blk = a.cell_blk; pl.blk_cells = a.blk_cells; pl.blk_pos = a.blk_pos; pl.remap = a.remap;
    pl.dirty_cells = a.dirty_cells; pl.dirty_pos = a.dirty_pos; pl.dirty_dst = a.dirty_dst;
    pl.cs.resize(a.cs.size());
    for (size_t s = 0; s < a.cs.size(); ++s) {
        pl.cs[s].n_bid = a.cs[s].n_bid; pl.cs[s].ranked = a.cs[s].ranked; pl.cs[s].bid_idx = a.cs[s].bid_idx;
        pl.cs[s].bid_cell = a.cs[s].bid_cell; pl.cs[s].order = a.cs[s].order; pl.cs[s].rank = a.cs[s].rank;
        pl.cs[s].blk_of = a.cs[s].blk_of;
    }
    const int64_t n_ns = pl.n_ns, n_kv = pl.n_kv, n_blocks = pl.n_blocks, r = pl.ratio;
    pl.v_cell_blk.resize(n_ns); pl.v_blk_cells.resize(n_ns); pl.v_remap.resize(n_ns); pl.v_blk_pos.resize(n_ns); pl.v_cs.resize(n_ns);
    pl.v_pos_stride = n_blocks*n_ns;
    for (int64_t s = 0; s < n_ns; ++s) {
        pl.v_cell_blk[s]  = pl.cell_blk.data()  + s*n_kv;
        pl.v_blk_cells[s] = pl.blk_cells.data() + s*r*n_blocks;
        pl.v_remap[s]     = pl.remap.data()     + s*n_blocks;
        pl.v_blk_pos[s]   = pl.blk_pos.data()   + s*n_blocks;
        pl.v_cs[s]        = &pl.cs[s];
    }
}

static void ref_to_store(const qsa_census_ref::store & a, llama_qsa_pooled_store & st) {
    st.ratio = a.ratio; st.kv_size = a.kv_size; st.n_stream = a.n_stream; st.n_slots = a.n_slots; st.n_rows = a.n_rows;
    st.strm.assign(a.n_stream, llama_qsa_pooled_store::stream_state());
    for (uint32_t g = 0; g < a.n_stream; ++g) {
        st.strm[g].gen = a.strm[g].gen; st.strm[g].slot_of_key = a.strm[g].slot_of_key; st.strm[g].slot_key = a.strm[g].slot_key;
        st.strm[g].slot_stamp = a.strm[g].slot_stamp; st.strm[g].slot_seen = a.strm[g].slot_seen;
        st.strm[g].free_slots = a.strm[g].free_slots;
    }
    st.serial = a.serial;
    st.pending.clear();
    for (const auto & p : a.pending) {
        st.pending.push_back({ p.s, p.slot, p.stamp });
    }
    st.pending_streams = a.pending_streams;
    st.pending_armed   = a.pending_armed;
}

// one fuzzed world: the cells of every stream + the three stores
struct world {
    rng_t &  rng;
    uint32_t r, kv_size, n_stream, pad;
    bool     unified;
    int      n_seq;          // sequences in use (unified: all in stream 0; else seq s in stream s)
    bool     whole_blocks;

    std::vector<llama_kv_cells> cells;   // [n_stream]
    std::vector<uint32_t>       head;    // [n_stream]
    std::vector<llama_kv_cells> saved;   // a state_write snapshot per stream
    std::vector<bool>           has_saved;

    llama_qsa_pooled_store st_inc, st_full;
    qsa_census_ref::store  st_ref;

    bool last_2d = false;

    uint64_t n_plans = 0, n_cmp_fail = 0, n_fast_partial = 0; // n_fast_partial: fast stream censuses in non-full-pool plans
    std::string first_fail;

    // [TAG_QSA_INPUTS_INC]
    mirror_sim mir;
    bool       tail_fix = true;
    uint64_t   n_mir = 0, n_mir_skip = 0;

    world(rng_t & rng_, uint32_t r_, uint32_t kv_, uint32_t ns_, uint32_t pad_, int n_seq_, bool wb) :
        rng(rng_), r(r_), kv_size(kv_), n_stream(ns_), pad(pad_), unified(ns_ == 1), n_seq(n_seq_), whole_blocks(wb) {
        cells.resize(n_stream);
        saved.resize(n_stream);
        has_saved.assign(n_stream, false);
        head.assign(n_stream, 0);
        for (auto & c : cells) {
            c.resize(kv_size);
        }
        store_init(st_inc,  r, kv_size, n_stream, true);
        store_init(st_full, r, kv_size, n_stream, false);
        st_ref.init(r, kv_size, n_stream);
        mir.init(kv_size, st_inc.n_slots, n_stream);
        tail_fix = rng.p(0.7);
    }

    uint32_t strm_of(int seq) const { return unified ? 0 : (uint32_t) seq; }

    // --- the write path (llama_memory_hybrid_idx_context::apply) ---
    void write(int seq, int n_tok, bool image, bool overwrite) {
        const uint32_t s = strm_of(seq);
        auto & c = cells[s];

        // the cells: empty ones from head, wrapping (find_slot, non-contiguous); an overwrite takes one used cell of `seq`
        std::vector<uint32_t> idxs;
        for (uint32_t k = 0; k < kv_size && (int) idxs.size() < n_tok; ++k) {
            const uint32_t i = (head[s] + k) % kv_size;
            if (c.is_empty(i)) {
                idxs.push_back(i);
            }
        }
        if (overwrite) {
            std::vector<uint32_t> used;
            for (uint32_t i = 0; i < kv_size; ++i) {
                if (!c.is_empty(i) && c.seq_count(i) == 1 && c.seq_has(i, seq)) {
                    used.push_back(i);
                }
            }
            if (used.empty()) {
                return;
            }
            if (!idxs.empty()) {
                idxs.pop_back();
            }
            idxs.push_back(used[rng.u(used.size())]);
        }
        if (idxs.empty()) {
            return;
        }

        llama_pos p = c.seq_pos_max(seq) + 1;
        if (p + (llama_pos) idxs.size() + 4 >= (llama_pos) kv_size) {
            return; // keep positions inside the window (oor is exercised by the shrink paths)
        }

        if (g_drop != DROP_PREWRITE) {
            st_inc.inc_mark_write_pre(s, c, idxs);
        }

        // M-RoPE image: groups of tokens share one position, different (x, y)
        const int img_w = image ? (int) rng.r(2, 4) : 1;

        for (size_t k = 0; k < idxs.size(); ++k) {
            const uint32_t i = idxs[k];
            if (!c.is_empty(i)) {
                // apply_ubatch's overwrite: rm, then the purge of the older positions of that sequence
                const llama_pos old = c.pos_get(i);
                c.rm(i);
                const llama_pos mn = c.seq_pos_min(seq);
                for (uint32_t j = 0; mn >= 0 && mn <= old && j < kv_size; ++j) {
                    if (!c.is_empty(j) && c.seq_has(j, seq) && c.pos_get(j) >= mn && c.pos_get(j) <= old) {
                        c.seq_rm(j, seq);
                    }
                }
            }
            c.pos_set(i, p);
            if (image) {
                llama_kv_cell_ext e;
                e.x = (llama_pos) (k % img_w);
                e.y = (llama_pos) ((k / img_w) % 3);
                c.ext_set(i, e);
                if ((k + 1) % (img_w*2) == 0) {
                    p++;
                }
            } else {
                p++;
            }
            c.seq_add(i, seq);
            if (unified && n_seq > 1 && rng.p(0.03)) {
                c.seq_add(i, (seq + 1) % n_seq); // a shared cell
            }
        }
        head[s] = (idxs.back() + 1) % kv_size;
        last_2d = image;

        llama_kv_cache::slot_info si;
        si.s0 = s; si.s1 = s;
        si.strm = { (decltype(si.strm)::value_type) s };
        si.idxs = { idxs };

        st_inc.on_write(si);
        st_full.on_write(si);
        st_ref.on_write(si);

        if (g_drop != DROP_WRITE) {
            st_inc.inc_mark_write_post(s, c, idxs);
        }
    }

    // --- llama_kv_cache::seq_rm + the wrapper's mark ---
    void seq_rm(int seq, llama_pos p0, llama_pos p1) {
        const llama_pos q0 = p0 < 0 ? 0 : p0;
        const llama_pos q1 = p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1;
        for (uint32_t s = 0; s < n_stream; ++s) {
            if (seq >= 0 && s != strm_of(seq)) {
                continue;
            }
            auto & c = cells[s];
            uint32_t new_head = kv_size;
            for (uint32_t i = 0; i < kv_size; ++i) {
                if (!c.pos_in(i, q0, q1)) {
                    continue;
                }
                if (seq >= 0) {
                    if (c.seq_has(i, seq) && c.seq_rm(i, seq)) {
                        new_head = std::min(new_head, i);
                    }
                } else {
                    c.rm(i);
                    new_head = std::min(new_head, i);
                }
            }
            if (new_head != kv_size && new_head < head[s]) {
                head[s] = new_head;
            }
            if (g_drop != DROP_RM) {
                st_inc.inc_mark_pos(s, p0);
            }
        }
    }

    void invalidate(uint32_t s) {
        st_inc.reset_stream(s);
        st_full.reset_stream(s);
        st_ref.reset_stream(s);
        if (g_drop == DROP_RESET) {
            // the negative control: the census state survives a reset (as if reset_stream left no mark)
            st_inc.strm[s].cen.inc_all = false;
            st_inc.strm[s].cen.valid   = true;
        }
    }

    void seq_cp(int src, int dst, llama_pos p0, llama_pos p1) {
        const uint32_t ss = strm_of(src), sd = strm_of(dst);
        if (ss == sd) {
            auto & c = cells[ss];
            for (uint32_t i = 0; i < kv_size; ++i) {
                if (c.seq_has(i, src) && c.pos_in(i, p0 < 0 ? 0 : p0, p1 < 0 ? std::numeric_limits<llama_pos>::max() : p1)) {
                    c.seq_add(i, dst);
                }
            }
        } else {
            // llama_kv_cache: the whole destination stream becomes a copy of the source sequence
            auto & cd = cells[sd];
            cd.reset();
            for (uint32_t i = 0; i < kv_size; ++i) {
                if (cells[ss].seq_has(i, src)) {
                    cd.pos_set(i, cells[ss].pos_get(i));
                    cd.ext_set(i, cells[ss].ext_get(i));
                    cd.seq_add(i, dst);
                }
            }
            head[sd] = 0;
            invalidate(sd); // the wrapper's reset_stream
        }
        if (g_drop != DROP_CP) {
            st_inc.inc_mark_all(sd);
        }
    }

    void seq_keep(int seq) {
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & c = cells[s];
            for (uint32_t i = 0; i < kv_size; ++i) {
                if (!c.is_empty(i)) {
                    c.seq_keep(i, seq);
                }
            }
        }
        if (g_drop != DROP_KEEP) {
            st_inc.inc_mark_all_streams();
        }
    }

    // seq_add (d != 0) / seq_div (div > 1) over [p0, inf)
    void seq_add(int seq, llama_pos p0, llama_pos d, int div) {
        auto & c = cells[strm_of(seq)];
        for (uint32_t i = 0; i < kv_size; ++i) {
            if (c.seq_has(i, seq) && c.pos_in(i, p0, std::numeric_limits<llama_pos>::max())) {
                if (c.seq_count(i) == 1) {
                    if (div > 1) {
                        c.pos_div(i, div);
                    } else {
                        c.pos_add(i, d);
                    }
                }
            }
        }
        if (g_drop != DROP_ADD) {
            st_inc.inc_mark_all(strm_of(seq));
        }
    }

    void clear() {
        for (uint32_t s = 0; s < n_stream; ++s) {
            cells[s].reset();
            head[s] = 0;
        }
        st_inc.reset_all();
        st_full.reset_all();
        st_ref.reset_all();
    }

    // state_read: the wrapper invalidates first, then the cells are replaced (ok) or half-replaced and dropped (failed)
    void state_read(uint32_t s, bool fail) {
        if (!has_saved[s]) {
            return;
        }
        invalidate(s);
        if (!fail) {
            cells[s] = saved[s];
        } else {
            // half the restore landed, then state_drop: invalidate again and remove the sequences of the stream
            for (uint32_t i = 0; i < kv_size / 2; ++i) {
                if (!saved[s].is_empty(i) && cells[s].is_empty(i)) {
                    cells[s].pos_set(i, saved[s].pos_get(i));
                    cells[s].seq_add(i, unified ? 0 : (llama_seq_id) s);
                }
            }
            invalidate(s);
            cells[s].reset();
        }
        head[s] = 0;
    }

    int64_t n_kv_of(uint32_t s0, uint32_t s1) const {
        int64_t res = 0;
        for (uint32_t s = s0; s <= s1; ++s) {
            res = std::max<int64_t>(res, std::min<int64_t>(kv_size, std::max<int64_t>(pad, GGML_PAD(cells[s].used_max_p1(), pad))));
        }
        return res;
    }

    void publish_all(bool ok, bool armed) {
        for (auto * st : { &st_inc, &st_full }) {
            st->pending_armed = armed;
            st->publish(ok);
        }
        st_ref.pending_armed = armed;
        st_ref.publish(ok);
    }

    // one plan of streams [s0, s1] at n_kv (+ extra), compared three ways; false on the first mismatch
    bool plan(uint32_t s0, uint32_t s1, int64_t extra) {
        int64_t n_kv = std::min<int64_t>(kv_size, n_kv_of(s0, s1) + extra);

        // blk_bias = true asserts on oor: plan with false then (both sides handle oor the same way)
        bool oor = false;
        for (uint32_t s = s0; s <= s1; ++s) {
            for (uint32_t j = 0; j < n_kv; ++j) {
                if (!cells[s].is_empty(j) && cells[s].pos_get(j)/(int64_t) r >= (n_kv + r - 1)/r) {
                    oor = true;
                }
            }
        }

        // the member's pre-publish of a plan whose graph never reported back
        for (auto * st : { &st_inc, &st_full }) {
            if (!st->pending.empty() || !st->pending_streams.empty()) {
                st->publish(st->pending_armed ? false : true);
            }
        }
        if (!st_ref.pending.empty() || !st_ref.pending_streams.empty()) {
            st_ref.publish(st_ref.pending_armed ? false : true);
        }

        llama_qsa_plan_in in;
        in.n_ns = s1 - s0 + 1;
        in.is_pos_2d = last_2d;
        in.n_kv = n_kv;
        in.blk_bias = !oor;
        std::vector<const llama_kv_cells *> cv;
        std::vector<uint32_t>               gv;
        for (uint32_t s = s0; s <= s1; ++s) {
            cv.push_back(&cells[s]);
            gv.push_back(s);
        }
        in.cells = cv;
        in.g     = gv;

        qsa_census_ref::plan_t rp;
        qsa_census_ref::plan(&st_ref, cv, gv, last_2d, whole_blocks, r, n_kv, !oor, rp);

        llama_qsa_plan pf, pi;
        std::vector<std::vector<uint64_t>> stamps_full;
        llama_qsa_plan_full(st_full, in, pf, &stamps_full);
        const uint64_t fast0 = st_inc.stats.n_cen_inc;
        llama_qsa_plan_inc (st_inc,  in, pi);
        if (!pi.full) {
            n_fast_partial += st_inc.stats.n_cen_inc - fast0;
        }

        llama_qsa_plan         rpl;
        llama_qsa_pooled_store rst;
        ref_to_plan(rp, rpl);
        ref_to_store(st_ref, rst);

        n_plans++;

        std::string d = llama_qsa_plan_compare(rpl, rp.stamps, rst, pi, st_inc, r);
        if (d.empty()) {
            d = llama_qsa_plan_compare(rpl, rp.stamps, rst, pf, st_full, r, &stamps_full);
            if (!d.empty()) {
                d = "[full path vs frozen ref] " + d;
            }
        } else {
            d = "[inc vs frozen ref] " + d;
        }
        for (uint32_t s = s0; s <= s1 && d.empty(); ++s) {
            d = llama_qsa_census_state_check(st_inc, s, cells[s]);
        }

        // [TAG_QSA_INPUTS_INC] the set_input of a compact graph (blk_bias) - or not: a graph rebuilt cache-only, or a
        // plan replaced before its graph ran, leaves the hints to the next update
        if (d.empty() && !oor && rng.p(0.85)) {
            if (g_drop_mir) {
                for (uint32_t s = s0; s <= s1; ++s) {
                    auto & C = st_inc.strm[s].cen;
                    C.mir_cb.clear();
                    C.mir_rm_lo = INT64_MAX;
                    C.mir_bt_lo = INT64_MAX;
                }
            }
            llama_qsa_mirror_update(st_inc, pi, cv, tail_fix, !pi.full, true, mir);
            n_mir++;
            d = mirror_compare(mir, pi, cv, tail_fix, !pi.full);
            if (!d.empty()) {
                d = "[inputs mirror vs full fill] " + d;
            }
        } else if (d.empty()) {
            n_mir_skip++;
        }
        if (!d.empty()) {
            n_cmp_fail++;
            if (first_fail.empty()) {
                first_fail = d;
            }
            return false;
        }
        return true;
    }
};

struct totals_t {
    uint64_t seqs = 0, plans = 0, fails = 0, ops = 0;
    uint64_t mir = 0, mir_skip = 0, mir_bytes = 0, mir_full = 0, mir_inc = 0, mir_runs = 0;
    uint64_t cen_inc = 0, cen_full = 0, fb[LLAMA_QSA_CEN_FB_COUNT] = {};
    uint64_t full_pool = 0, fast_partial = 0;
    std::string first_fail;
};

// one random sequence of mutators and plans
static bool run_sequence(rng_t & rng, bool whole_blocks, totals_t & tot) {
    const uint32_t rs[4]   = { 2, 4, 4, 8 };
    const uint32_t kvs[4]  = { 128, 512, 1024, 2048 };
    const uint32_t pads[3] = { 8, 16, 32 };

    const uint32_t r        = rs[rng.u(4)];
    const uint32_t kv_size  = kvs[rng.u(4)];
    const uint32_t n_stream = rng.p(0.6) ? 1 : (uint32_t) rng.r(2, 4);
    const int      n_seq    = n_stream == 1 ? (rng.p(0.7) ? 1 : (int) rng.r(2, 3)) : (int) n_stream;
    const uint32_t pad      = pads[rng.u(3)];

    world w(rng, r, kv_size, n_stream, pad, n_seq, whole_blocks);

    const int  n_ops     = (int) rng.r(100, 400);
    const bool images    = rng.p(0.1);
    // dense phase: no plan while true (below LLAMA_QSA_MIN_KV), toggled now and then
    bool dense = rng.p(0.2);

    for (int op = 0; op < n_ops; ++op) {
        tot.ops++;

        const int seq = (int) rng.u(n_seq);
        const uint32_t s = w.strm_of(seq);
        const double x = rng.u(1000) / 1000.0;

        if (x < 0.40) {
            // decode / verify / chunk
            const int n = rng.p(0.5) ? (int) rng.r(1, 6) : (int) rng.r(1, 20);
            w.write(seq, n, images && rng.p(0.3), false);
        } else if (x < 0.55) {
            // MTP rollback: drop the last 1..6 positions
            const llama_pos pm = w.cells[s].seq_pos_max(seq);
            if (pm >= 0) {
                w.seq_rm(seq, std::max<llama_pos>(0, pm - (llama_pos) rng.r(0, 6)), -1);
            }
        } else if (x < 0.60) {
            // partial seq_rm: middle of the tail / across block boundaries / a hole
            const llama_pos pm = w.cells[s].seq_pos_max(seq);
            if (pm > 4) {
                const llama_pos a = rng.p(0.5) ? std::max<llama_pos>(0, pm - (llama_pos) rng.r(1, 3*r)) : (llama_pos) rng.u(pm);
                const llama_pos b = a + (llama_pos) rng.r(1, 2*r + 1);
                w.seq_rm(rng.p(0.1) ? -1 : seq, a, b);
            }
        } else if (x < 0.62) {
            w.write(seq, (int) rng.r(1, 3), false, true); // overwrite of a used cell
        } else if (x < 0.64 && n_seq > 1) {
            const int dst = (seq + 1) % n_seq;
            if (rng.p(0.5)) {
                w.seq_rm(dst, -1, -1);
            }
            w.seq_cp(seq, dst, -1, rng.p(0.5) ? -1 : (llama_pos) rng.u(64));
        } else if (x < 0.65 && n_seq > 1) {
            w.seq_keep(seq);
        } else if (x < 0.66) {
            const llama_pos pm = w.cells[s].seq_pos_max(seq);
            if (pm > 8) {
                if (rng.p(0.8)) {
                    w.seq_add(seq, (llama_pos) rng.u(pm), -(llama_pos) rng.r(1, 8), 1);
                } else {
                    w.seq_add(seq, (llama_pos) rng.u(pm), 0, 2);
                }
            }
        } else if (x < 0.67) {
            w.clear();
        } else if (x < 0.69) {
            w.saved[s] = w.cells[s];
            w.has_saved[s] = true;
        } else if (x < 0.71) {
            w.state_read(s, rng.p(0.3));
        } else if (x < 0.72) {
            dense = !dense;
        } else if (x < 0.74) {
            // restart a sequence near the end of the window
            if (w.cells[s].seq_pos_max(seq) > (llama_pos) kv_size*3/4) {
                w.seq_rm(seq, -1, -1);
            }
        } else if (!dense) {
            // plan (+ publish)
            uint32_t s0 = 0, s1 = 0;
            if (n_stream > 1) {
                s0 = (uint32_t) rng.u(n_stream);
                s1 = s0 + (uint32_t) rng.u(n_stream - s0);
            }
            if (!w.plan(s0, s1, 0)) {
                break;
            }
            if (rng.p(0.05) && !w.plan(s0, s1, pad)) { // a replan of the ubatch at a larger n_kv
                break;
            }
            const double y = rng.u(100) / 100.0;
            if (y < 0.85) {
                w.publish_all(true, true);
            } else if (y < 0.90) {
                w.publish_all(true, false);
            } else if (y < 0.93) {
                w.publish_all(false, true);
            } // else: no publish, the next plan's pre-publish handles it
        }
    }

    tot.seqs++;
    tot.plans += w.n_plans;
    tot.fails += w.n_cmp_fail;
    tot.cen_inc  += w.st_inc.stats.n_cen_inc;
    tot.cen_full += w.st_inc.stats.n_cen_full;
    tot.full_pool += w.st_inc.stats.n_full;
    tot.fast_partial += w.n_fast_partial;
    tot.mir       += w.n_mir;
    tot.mir_skip  += w.n_mir_skip;
    tot.mir_bytes += w.st_inc.stats.mir_bytes;
    tot.mir_full  += w.st_inc.stats.mir_full;
    tot.mir_inc   += w.st_inc.stats.mir_inc;
    tot.mir_runs  += w.st_inc.stats.mir_runs;
    for (int k = 0; k < LLAMA_QSA_CEN_FB_COUNT; ++k) {
        tot.fb[k] += w.st_inc.stats.n_cen_fb[k];
    }
    if (!w.first_fail.empty() && tot.first_fail.empty()) {
        tot.first_fail = w.first_fail;
    }
    return w.n_cmp_fail == 0;
}

static int run_fuzz(uint64_t n_seq, uint64_t seed, bool quiet) {
    rng_t rng(seed);
    totals_t tot;
    const bool wb = llama_qsa_whole_blocks();
    for (uint64_t i = 0; i < n_seq; ++i) {
        if (!run_sequence(rng, wb, tot)) {
            if (!quiet) {
                fprintf(stderr, "FAIL: sequence %" PRIu64 " (seed %" PRIu64 "): %s\n", i, seed, tot.first_fail.c_str());
            }
            return 1;
        }
    }
    if (!quiet) {
        printf("fuzz: %" PRIu64 "/%" PRIu64 " sequences identical, %" PRIu64 " plans x 3-way compare, %" PRIu64 " ops; "
               "stream censuses fast %" PRIu64 " / full %" PRIu64 " (mark %" PRIu64 ", shape %" PRIu64 ", shrink %" PRIu64
               ", region %" PRIu64 "), full-pool plans %" PRIu64 ", fast censuses in partial-pool plans %" PRIu64
               ", whole_blocks=%d\n",
               tot.seqs, n_seq, tot.plans, tot.ops, tot.cen_inc, tot.cen_full, tot.fb[1], tot.fb[2], tot.fb[3], tot.fb[4],
               tot.full_pool, tot.fast_partial, wb ? 1 : 0);
        printf("inputs: %" PRIu64 " mirror updates compared with the full fill (%" PRIu64 " plans left without one), "
               "%" PRIu64 " whole-array / %" PRIu64 " incremental (array, stream) writes, %" PRIu64 " runs, %" PRIu64 " KiB\n",
               tot.mir, tot.mir_skip, tot.mir_full, tot.mir_inc, tot.mir_runs, tot.mir_bytes/1024);
    }
    if (tot.mir_inc == 0 || tot.mir_full == 0) {
        fprintf(stderr, "FAIL: the fuzz did not exercise both mirror write paths\n");
        return 1;
    }
    if (tot.cen_inc == 0 || tot.cen_full == 0 || tot.fast_partial == 0) {
        fprintf(stderr, "FAIL: the fuzz did not exercise both census paths\n");
        return 1;
    }
    return 0;
}

// the negative controls: a child per dropped mark; it must fail (mismatch exit or abort)
static int run_neg(uint64_t n_seq, uint64_t seed) {
#ifdef _WIN32
    (void) n_seq; (void) seed;
    printf("neg: skipped on Windows\n");
    return 0;
#else
    int bad = 0;
    // d == DROP_COUNT: [TAG_QSA_INPUTS_INC] the input mirrors' change hints
    for (int d = DROP_RM; d <= DROP_COUNT; ++d) {
        fflush(stdout);
        const pid_t pid = fork();
        if (pid == 0) {
            // silence the child's abort message
            if (freopen("/dev/null", "w", stderr) == nullptr) { _exit(3); }
            if (d == DROP_COUNT) {
                g_drop_mir = true;
            } else {
                g_drop = (drop_t) d;
            }
            _exit(run_fuzz(n_seq, seed, true) == 0 ? 0 : 1);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        const bool exited = WIFEXITED(status);
        const int  code   = exited ? WEXITSTATUS(status) : -1;
        const bool caught = !exited || code != 0;
        printf("neg: drop the %-11s mark -> %s (%s)%s\n", d == DROP_COUNT ? "inputs-hint" : drop_name[d],
               caught ? "DETECTED" : "NOT DETECTED", exited ? (code == 0 ? "exit 0" : "mismatch") : "abort",
               d < DROP_COUNT && drop_redundant(d) ? " [redundant by design: the one-sequence check / the stream reset cover it]" : "");
        bad += (caught || (d < DROP_COUNT && drop_redundant(d))) ? 0 : 1;
    }
    return bad == 0 ? 0 : 1;
#endif
}

// the knob parser
static int run_parser() {
    int bad = 0;
    bad += llama_qsa_census_inc_parse(nullptr) != 0;
    bad += llama_qsa_census_inc_parse("")      != 0;
    bad += llama_qsa_census_inc_parse("0")     != 0;
    bad += llama_qsa_census_inc_parse("1")     != 1;
    bad += llama_qsa_census_inc_parse("2")     != -1;
    bad += llama_qsa_census_inc_parse("on")    != -1;
    printf("parser: %s\n", bad == 0 ? "OK" : "FAIL");
    return bad == 0 ? 0 : 1;
}

// µs per plan, full vs incremental: one sequence filled to n_cells - 64, then decode rounds of a verify of 1..6 tokens and
// an MTP rollback of 0..5 of them, published like a live decode
static void run_bench(int64_t n_cells, uint64_t seed) {
    const uint32_t r = 4;
    rng_t rng(seed);

    for (int mode = 0; mode < 2; ++mode) {
        llama_kv_cells cells;
        cells.resize((uint32_t) n_cells);
        llama_qsa_pooled_store st;
        store_init(st, r, (uint32_t) n_cells, 1, mode == 1);

        const int64_t fill = n_cells - 4096;
        for (int64_t i = 0; i < fill; ++i) {
            cells.pos_set((uint32_t) i, (llama_pos) i);
            cells.seq_add((uint32_t) i, 0);
        }

        llama_qsa_plan_in in;
        in.n_ns = 1; in.is_pos_2d = false; in.blk_bias = true;
        in.cells = { &cells }; in.g = { 0 };

        auto n_kv_now = [&]() { return std::min<int64_t>(n_cells, GGML_PAD((int64_t) cells.used_max_p1(), 256)); };

        // warm: one full plan, published
        llama_qsa_plan pl;
        in.n_kv = n_kv_now();
        if (mode == 0) { llama_qsa_plan_full(st, in, pl); } else { llama_qsa_plan_inc(st, in, pl); }
        st.pending_armed = true; st.publish(true);

        // [TAG_QSA_INPUTS_INC]
        std::vector<int32_t> full_cb(n_cells), full_rm(n_cells/r + 1), full_bt(2*(n_cells/r + 1));
        mirror_sim msim;
        msim.init((uint32_t) n_cells, st.n_slots, 1);
        if (mode == 1) {
            llama_qsa_mirror_update(st, pl, { &cells }, true, !pl.full, true, msim);
        }
        double   in_us    = 0;
        uint64_t in_bytes = 0;

        llama_pos pos = (llama_pos) fill;
        uint32_t  head = (uint32_t) fill;
        const int n_round = 200;
        double t_us = 0;
        int n_plans = 0;
        rng_t rr(seed);
        for (int round = 0; round < n_round; ++round) {
            const int n = (int) rr.r(1, 6);
            std::vector<uint32_t> idxs;
            for (int k = 0; k < n; ++k) {
                while (!cells.is_empty(head)) { head++; }
                idxs.push_back(head++);
            }
            st.inc_mark_write_pre(0, cells, idxs);
            for (int k = 0; k < n; ++k) {
                cells.pos_set(idxs[k], pos + k);
                cells.seq_add(idxs[k], 0);
            }
            llama_kv_cache::slot_info si; si.s0 = 0; si.s1 = 0; si.strm = { 0 }; si.idxs = { idxs };
            st.on_write(si);
            st.inc_mark_write_post(0, cells, idxs);

            in.n_kv = n_kv_now();
            const auto t0 = std::chrono::steady_clock::now();
            if (mode == 0) { llama_qsa_plan_full(st, in, pl); } else { llama_qsa_plan_inc(st, in, pl); }
            const auto t1 = std::chrono::steady_clock::now();
            t_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
            n_plans++;

            // [TAG_QSA_INPUTS_INC] the host side of the inputs: full = memcpy of cell_blk + remap + the whole block table
            // (today's set_input_qsa_store), inc = the mirror update (sorted runs, partial table fill) into host arrays
            {
                const auto u0 = std::chrono::steady_clock::now();
                if (mode == 0) {
                    memcpy(full_cb.data(), pl.v_cell_blk[0], pl.n_kv*sizeof(int32_t));
                    if (!pl.full) { memcpy(full_rm.data(), pl.v_remap[0], pl.n_blocks*sizeof(int32_t)); }
                    llama_qsa_fill_blk_tab(cells, *pl.v_cs[0], pl.n_blocks, 0, pl.n_blocks, true, full_bt.data());
                    in_bytes += (pl.n_kv + (pl.full ? 0 : pl.n_blocks) + 2*pl.n_blocks)*sizeof(int32_t);
                } else {
                    const uint64_t b0 = st.stats.mir_bytes;
                    llama_qsa_mirror_update(st, pl, { &cells }, true, !pl.full, true, msim);
                    in_bytes += st.stats.mir_bytes - b0;
                }
                const auto u1 = std::chrono::steady_clock::now();
                in_us += std::chrono::duration<double, std::micro>(u1 - u0).count();
            }
            st.pending_armed = true; st.publish(true);

            // rollback: accept a of n
            const int a = (int) rr.r(1, n);
            for (int k = a; k < n; ++k) {
                cells.seq_rm(idxs[k], 0);
            }
            head = idxs[a < n ? a : n - 1] + (a < n ? 0 : 1);
            st.inc_mark_pos(0, pos + a);
            pos += a;
        }
        printf("bench: n_cells %7" PRId64 " r %u  %-4s  %8.1f us/plan  (%d plans; fast stream censuses %" PRIu64 ")"
               "  inputs host %8.1f us/graph, %10.1f bytes/graph written\n",
               n_cells, r, mode == 0 ? "full" : "inc", t_us/n_plans, n_plans, st.stats.n_cen_inc, in_us/n_plans,
               (double) in_bytes/n_plans);
    }
}

} // namespace

int main(int argc, char ** argv) {
    uint64_t n_seq = argc > 1 ? strtoull(argv[1], nullptr, 10) : 10000;
    uint64_t seed  = argc > 2 ? strtoull(argv[2], nullptr, 10) : 42;
    const std::string mode = argc > 3 ? argv[3] : "";

    if (mode == "bench") {
        std::vector<int64_t> sizes;
        for (int i = 4; i < argc; ++i) { sizes.push_back(strtoll(argv[i], nullptr, 10)); }
        if (sizes.empty()) { sizes = { 262144, 524288 }; }
        for (const int64_t n : sizes) { run_bench(n, seed); }
        return 0;
    }

    int rc = run_parser();
    rc |= run_fuzz(n_seq, seed, false);
    if (mode == "neg" || mode.empty()) {
        // the negative controls run a shorter fuzz each
        rc |= run_neg(std::max<uint64_t>(200, n_seq/10), seed);
    }
    printf("%s\n", rc == 0 ? "OK" : "FAIL");
    return rc;
}
