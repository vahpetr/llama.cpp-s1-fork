// [TAG_MTP_DRAFT_WINDOW] CPU tests of the MTP draft attention window (LLAMA_MTP_DRAFT_WINDOW=N, LLAMA_MTP_DRAFT_SINK=S):
// in the draft context a cell is visible to a query token iff today's rule holds AND (cell_pos < S || cell_pos > pos - N).
//
//  1. predicate identity over random layouts - unified multi-sequence pools (up to 8 sequences interleaved in chunks, as
//     find_slot places concurrent slots) and per-stream caches, holes, seqless and shared-prefix cells, seq_rm rollbacks
//     followed by in-flight draft cells placed into the freed holes (cell order != position order), 1-D and M-RoPE
//     positions, decode (one token per sequence), catch-up (<= 6) and prefill ubatches, windows / sinks straddling the
//     positions (N in {1 .. span + 1, 65536}, cells exactly at pos - N and pos - N + 1, at S - 1 and S) - every mask form
//     the draft graph can run equals the reference
//         ref(c, t) = base(c, t) if (c.pos < S || c.pos > t.pos - N) else -inf
//     where base = today's host fill (window off) and the window term is evaluated here from the cell positions:
//       a. the host plane with the band tracking (llama_kv_cache::set_input_kq_mask of the draft input)
//       b. the host plane with today's row tracking (win_band off: is_masked_swa + sinks alone)
//       c. GGML_OP_KQ_MASK over the metadata records + ggml_kq_mask_set_sink (the bridge class, CPU backend)
//       d. FA over the metadata (ggml_flash_attn_ext_vis + ggml_flash_attn_ext_set_vis_sink) vs FA over the ref plane
//          and vs FA over the op's plane, f32 outputs bit for bit (layouts under the FA budget)
//     negative controls: on layouts with a cell exactly at the window edge (resp. at the sink edge) the fill with N + 1
//     (resp. S + 1) must differ from ref - the comparison sees one-position errors.
//  2. window off = the base commit, byte for byte: the current host fill with the window off (every SWA type, random
//     n_swa, causal / not, ALiBi, F16 / F32; no sink, no band) against the frozen fill of prod-v11-vis-fill @ 2a8090bd4
//     (tests/kq-mask-fill-base-ref.h); and the sink setters with 0 leave a node's op params as the base API builds them.
//  3. window beyond every distance (N > every pos): the windowed plane equals the dense one (8k-census argument: below
//     the window the draft sees exactly today's cells).
//  4. the LLAMA_KQ_VIS_PROF tile census (llama_kv_cells_window_census) is an upper bound of the exact live tiles of the
//     ref plane, and exact for causal 1-D ubatches of one token per sequence (the draft's decode steps).
//  5. llama_hparams::is_masked_swa with sinks on a hand table.
//
//  6. --device NAME (e.g. Vulkan0): the same identity on that backend - per layout GGML_OP_KQ_MASK (window + sinks) vs the
//     ref plane, and FA over the ref plane = FA over the metadata = the bridge (device KQ_MASK plane + plane FA), f32 bit
//     for bit on the device - and, with --perf, the skip probe at the MTP draft's shapes (24 q / 2 kv heads, D 256, f16
//     K/V; one sequence whose cells are in position order, the query tokens last; n_tps 1 = a draft step, 6 = the
//     catch-up): per n_kv (--perf-nkv, default 32768,213248,393216) the device time of the bridge FA (the draft's decode
//     class) and of the metadata FA, dense vs window 65536 / 16384 (sink 4), median of --perf-reps, outputs of each window
//     run identical across the three forms. The window arms read K/V only for the 64-cell tiles holding a sink or a
//     window cell: the time ratio is the measured skip.
//
// usage: test-mtp-draft-window [n_layouts = 3000] [seed = 7] [n_threads = 4] [fa_budget = 49152]
//                              [--device NAME] [--dev-layouts N (300)] [--perf] [--perf-reps N (15)] [--perf-nkv a,b,c]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"

#include "kq-mask-fill-base-ref.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

std::mt19937_64 g_rng;

int rnd(int lo, int hi) { // inclusive
    return std::uniform_int_distribution<int>(lo, hi)(g_rng);
}
double rnd01() {
    return std::uniform_real_distribution<double>(0.0, 1.0)(g_rng);
}

struct stats {
    uint64_t layouts = 0, elems = 0, unified = 0, per_stream = 0, multi_seq_ub = 0, is_2d = 0, non_causal = 0, alibi = 0;
    uint64_t decode = 0, catchup = 0, prefill = 0, rollbacks = 0, shared = 0;
    uint64_t win_masked = 0;       // elements visible in base, masked by the window
    uint64_t sink_saved = 0;       // elements outside the window but visible through a sink
    uint64_t edge_layouts = 0, edge_detected = 0, sink_edge_layouts = 0, sink_edge_detected = 0;
    uint64_t fa_runs = 0, fa_elems = 0;
    uint64_t base_runs = 0, base_elems = 0, beyond_runs = 0;
    uint64_t census_runs = 0, census_exact_runs = 0, census_tiles = 0, census_live = 0, census_exact_live = 0;
    uint64_t fail = 0;
};
stats st;

#define CHECK(cond, ...) do { if (!(cond)) { st.fail++; if (st.fail <= 20) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

struct layout {
    bool unified = true;
    bool is_2d   = false;
    bool causal  = true;
    bool alibi   = false;
    bool f32     = false;
    int  n_seq   = 1;
    int64_t n_kv = 0;
    int64_t n_stream = 1; // ubatch stream blocks
    int64_t n_tps = 0;
    int     mode  = 0;    // 0 decode, 1 catch-up, 2 prefill
    uint32_t N = 0, S = 0;
    llama_pos max_pos = 0;

    std::vector<llama_kv_cells> v_cells;
    std::vector<uint32_t>       seq_to_stream;

    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_data;
    std::vector<llama_seq_id *> seq_ptr;
    std::vector<llama_seq_id>   seq_unq;
    std::vector<int32_t>        seq_idx;
    std::vector<int8_t>         output;
    llama_ubatch                ub{};
};

// place cell (pos p, seqs) into a free cell of `cells`; `order` = the scan order of free cells
bool place(llama_kv_cells & cells, uint32_t & cursor, llama_pos p, const std::vector<int> & seqs, bool from_start,
           int x = 0, int y = 0) {
    const uint32_t n = cells.size();
    uint32_t j = from_start ? 0 : cursor;
    for (uint32_t k = 0; k < n; ++k, j = (j + 1) % n) {
        if (cells.is_empty(j)) {
            cells.pos_set(j, p);
            for (int s : seqs) {
                cells.seq_add(j, s);
            }
            llama_kv_cell_ext e; e.x = x; e.y = y;
            cells.ext_set(j, e);
            if (!from_start) {
                cursor = (j + 1) % n;
            }
            return true;
        }
    }
    return false;
}

void build(layout & L, bool count) {
    L.unified = rnd01() < 0.7;
    L.n_seq   = rnd01() < 0.35 ? 1 : rnd(2, rnd01() < 0.8 ? 5 : 8);
    L.is_2d   = rnd01() < 0.15;
    L.causal  = rnd01() < 0.85;
    L.alibi   = rnd01() < 0.08;
    L.f32     = rnd01() < 0.15;
    L.mode    = rnd01() < 0.45 ? 0 : rnd01() < 0.6 ? 1 : 2;

    // history per sequence
    const double r = rnd01();
    const int H = r < 0.35 ? rnd(1, 48) : r < 0.8 ? rnd(16, 700) : rnd(300, 3000);
    std::vector<int> hist(L.n_seq);
    for (int s = 0; s < L.n_seq; ++s) {
        hist[s] = rnd01() < 0.1 ? 0 : rnd(std::max(1, H/3), H);
    }

    // the ubatch shape (tokens per sequence)
    const int n_tok_seq = L.mode == 0 ? 1 : L.mode == 1 ? rnd(2, 6) : rnd(7, H < 64 ? 64 : 256);

    // caches: unified = one stream for all; else one stream per sequence
    const int n_cache_streams = L.unified ? 1 : L.n_seq;
    L.seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
    for (int s = 0; s < L.n_seq; ++s) {
        L.seq_to_stream[s] = L.unified ? 0 : (uint32_t) s;
    }
    int64_t cap = 0;
    for (int cs = 0; cs < n_cache_streams; ++cs) {
        int64_t need = 0;
        for (int s = 0; s < L.n_seq; ++s) {
            if ((int) L.seq_to_stream[s] == cs) need += hist[s] + n_tok_seq + 8;
        }
        cap = std::max(cap, need);
    }
    cap += rnd(0, 64);
    L.v_cells.assign(n_cache_streams, llama_kv_cells());
    for (auto & c : L.v_cells) c.resize((uint32_t) std::max<int64_t>(cap, 8));

    // shared prefix: seqs 0 and 1 share their first P positions (one cell, two bits) - only on a common stream
    int shared_P = 0;
    if (L.unified && L.n_seq >= 2 && rnd01() < 0.25) {
        shared_P = rnd(1, std::max(1, std::min(hist[0], hist[1])));
        st.shared += count;
    }

    // interleaved chunks, round robin over the sequences (concurrent slots), positions 0 .. hist - 1
    std::vector<uint32_t> cursor(n_cache_streams, 0);
    std::vector<int> next(L.n_seq, 0);
    bool any = true;
    while (any) {
        any = false;
        for (int s = 0; s < L.n_seq; ++s) {
            const int chunk = rnd01() < 0.5 ? rnd(1, 8) : rnd(1, 96);
            for (int k = 0; k < chunk && next[s] < hist[s]; ++k) {
                const llama_pos p = next[s]++;
                if (s == 1 && p < shared_P) {
                    continue; // lives in seq 0's cells
                }
                std::vector<int> seqs = { s };
                if (s == 0 && p < shared_P) seqs.push_back(1);
                auto & cells = L.v_cells[L.seq_to_stream[s]];
                GGML_ASSERT(place(cells, cursor[L.seq_to_stream[s]], p, seqs, false, rnd(0, 2), rnd(0, 2)));
                any = true;
            }
        }
    }
    // a few seqless used cells and holes (removed cells of a finished slot)
    for (auto & cells : L.v_cells) {
        for (uint32_t j = 0; j < cells.size(); ++j) {
            if (!cells.is_empty(j)) {
                const double q = rnd01();
                if (q < 0.03) {
                    // drop this cell's sequences: a hole (keep shared-prefix cells to keep the prefix whole)
                    bool multi = false;
                    for (int s = 0; s < L.n_seq; ++s) multi |= cells.seq_has(j, s) && s > 0 && cells.seq_has(j, 0);
                    if (!multi) cells.rm(j);
                }
            } else if (rnd01() < 0.02) {
                cells.pos_set(j, rnd(0, H + 8)); // used, no sequence
            }
        }
    }

    // rollback in flight: per sequence, seq_rm of positions >= R (a rejected draft suffix), and the next tokens then
    // land in the freed holes (from the cache start: cell order no longer follows positions)
    std::vector<llama_pos> start(L.n_seq);
    for (int s = 0; s < L.n_seq; ++s) {
        auto & cells = L.v_cells[L.seq_to_stream[s]];
        llama_pos pmax = cells.seq_pos_max(s);
        start[s] = pmax + 1;
        if (pmax >= 0 && rnd01() < 0.35) {
            const llama_pos R = std::max<llama_pos>(0, pmax - rnd(0, 6));
            for (uint32_t j = 0; j < cells.size(); ++j) {
                if (!cells.is_empty(j) && cells.seq_has(j, s) && cells.pos_get(j) >= R) {
                    cells.seq_rm(j, s);
                }
            }
            start[s] = cells.seq_pos_max(s) + 1;
            st.rollbacks += count;
        }
    }

    // the ubatch: n_tok_seq consecutive positions per sequence from start[s]; every token gets its own cell (the ubatch
    // is stored before set_input); in-flight draft cells = the decode-step tokens themselves after a rollback
    std::vector<int> useq;
    for (int s = 0; s < L.n_seq; ++s) useq.push_back(s);
    std::shuffle(useq.begin(), useq.end(), g_rng);
    int n_useq = L.mode == 2 ? 1 : rnd(1, (int) useq.size());
    useq.resize(n_useq);

    // stream blocks: unified = one block with every token; per-stream = one block per sequence, equal sizes
    L.n_stream = L.unified ? 1 : n_useq;
    L.n_tps    = L.unified ? (int64_t) n_useq*n_tok_seq : n_tok_seq;
    const uint32_t n_tokens = (uint32_t) (L.n_tps*L.n_stream);
    const uint32_t n_pos    = L.is_2d ? 4 : 1;
    L.pos.assign((size_t) n_tokens*n_pos, 0);
    L.n_seq_id.assign(n_tokens, 1);
    L.seq_data.assign((size_t) n_tokens, 0);
    L.seq_ptr.assign(n_tokens, nullptr);
    L.output.assign(n_tokens, 1);
    L.seq_idx.assign(LLAMA_MAX_SEQ, -1);
    L.seq_unq.clear();

    std::vector<std::pair<int, int>> order; // (seq, k)
    if (L.unified) {
        // decode steps of concurrent slots: one token per sequence, interleaved; prefill/catch-up: grouped
        if (L.mode == 0) {
            for (int k = 0; k < n_tok_seq; ++k) for (int s : useq) order.push_back({ s, k });
        } else {
            for (int s : useq) for (int k = 0; k < n_tok_seq; ++k) order.push_back({ s, k });
        }
    } else {
        for (int s : useq) for (int k = 0; k < n_tok_seq; ++k) order.push_back({ s, k });
    }
    GGML_ASSERT(order.size() == n_tokens);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const int s = order[i].first;
        const int k = order[i].second;
        const llama_pos p = L.is_2d && rnd01() < 0.3 && k > 0 ? start[s] + k - 1 : start[s] + k; // 2-D: some equal positions
        int x = 0, y = 0;
        L.pos[i] = p;
        if (L.is_2d) {
            x = rnd(0, 2); y = rnd(0, 2);
            L.pos[i + n_tokens]   = y;
            L.pos[i + n_tokens*2] = x;
            L.pos[i + n_tokens*3] = 0;
        }
        L.seq_data[i] = s;
        L.seq_ptr[i]  = &L.seq_data[i];
        if (L.seq_idx[s] < 0) { L.seq_idx[s] = (int32_t) L.seq_unq.size(); L.seq_unq.push_back(s); }
        auto & cells = L.v_cells[L.seq_to_stream[s]];
        uint32_t cur = 0;
        const bool from_start = rnd01() < 0.5; // into the rollback holes / wherever find_slot finds room
        uint32_t c0 = from_start ? 0 : (uint32_t) rnd(0, (int) cells.size() - 1);
        GGML_ASSERT(place(cells, c0, p, { s }, from_start, x, y) || place(cells, cur, p, { s }, true, x, y));
        L.max_pos = std::max(L.max_pos, p);
    }
    if (n_useq > 1 && count) st.multi_seq_ub++;

    // n_kv: the used span padded like the cache (256) when it fits, else the whole cache
    int64_t used = 0;
    for (auto & cells : L.v_cells) {
        used = std::max<int64_t>(used, cells.used_max_p1());
        for (uint32_t j = 0; j < cells.size(); ++j) if (!cells.is_empty(j)) L.max_pos = std::max(L.max_pos, cells.pos_get(j));
    }
    int64_t minsize = INT64_MAX;
    for (auto & cells : L.v_cells) minsize = std::min<int64_t>(minsize, cells.size());
    L.n_kv = std::min<int64_t>(minsize, rnd01() < 0.5 ? GGML_PAD(used, 256) : used);
    L.n_kv = std::max<int64_t>(L.n_kv, 1);

    // window and sinks: around the positions so the edge and the sink edge are hit
    const int span = (int) L.max_pos + 1;
    const double w = rnd01();
    L.N = w < 0.15 ? (uint32_t) rnd(1, 3) : w < 0.55 ? (uint32_t) rnd(1, std::max(1, span)) : w < 0.8 ? (uint32_t) rnd(1, std::max(1, n_tok_seq + 8))
        : w < 0.92 ? (uint32_t) (span + rnd(0, 2)) : 65536u;
    const double sk = rnd01();
    L.S = sk < 0.2 ? 0u : sk < 0.6 ? 4u : sk < 0.8 ? (uint32_t) rnd(1, 8) : (uint32_t) rnd(0, std::max(1, span));

    L.ub = llama_ubatch{};
    L.ub.b_equal_seqs = L.n_stream > 1;
    L.ub.n_tokens     = n_tokens;
    L.ub.n_seq_tokens = (uint32_t) L.n_tps;
    L.ub.n_seqs       = (uint32_t) L.n_stream;
    L.ub.n_seqs_unq   = (uint32_t) L.seq_unq.size();
    L.ub.n_pos        = n_pos;
    L.ub.pos          = L.pos.data();
    L.ub.n_seq_id     = L.n_seq_id.data();
    L.ub.seq_id       = L.seq_ptr.data();
    L.ub.seq_id_unq   = L.seq_unq.data();
    L.ub.seq_idx      = L.seq_idx.data();
    L.ub.output       = L.output.data();

    if (!count) {
        return;
    }
    st.layouts++;
    (L.unified ? st.unified : st.per_stream)++;
    if (L.is_2d)   st.is_2d++;
    if (!L.causal) st.non_causal++;
    if (L.alibi)   st.alibi++;
    (L.mode == 0 ? st.decode : L.mode == 1 ? st.catchup : st.prefill)++;
}

size_t esz(const layout & L) { return L.f32 ? 4 : 2; }

std::vector<uint8_t> fill_plane(const layout & L, const llama_hparams & hp, uint32_t n_swa, llama_swa_type swa_type, uint32_t n_sink,
                                bool band, bool f32) {
    std::vector<uint8_t> out((size_t) (L.n_kv*L.n_tps*L.n_stream)*(f32 ? 4 : 2));
    llama_kv_cells_fill_kq_mask(out.data(), f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, L.n_kv, L.n_stream, &L.ub, L.causal, hp,
            n_swa, swa_type, L.v_cells, L.seq_to_stream, n_sink, band);
    return out;
}

bool is_neg_inf(const uint8_t * e, bool f32) {
    if (f32) { float v; memcpy(&v, e, 4); return std::isinf(v) && v < 0; }
    uint16_t h; memcpy(&h, e, 2); return h == 0xFC00;
}

void set_neg_inf(uint8_t * e, bool f32) {
    if (f32) { const float v = -INFINITY; memcpy(e, &v, 4); }
    else { const ggml_fp16_t h = ggml_fp32_to_fp16(-INFINITY); memcpy(e, &h, 2); }
}

// ref = base AND the window term, evaluated here from the positions of the cells each element reads
std::vector<uint8_t> make_ref(const layout & L, const std::vector<uint8_t> & base, uint32_t N, uint32_t S, bool count) {
    std::vector<uint8_t> ref = base;
    const size_t es = esz(L);
    for (int64_t i = 0; i < L.n_tps*L.n_stream; ++i) {
        const llama_seq_id s = L.ub.seq_id[i][0];
        const auto & cells = L.v_cells[L.seq_to_stream[s]];
        const int64_t p1 = L.ub.pos[i];
        for (int64_t j = 0; j < L.n_kv; ++j) {
            uint8_t * e = ref.data() + (size_t) (i*L.n_kv + j)*es;
            if (is_neg_inf(e, L.f32)) continue;
            const int64_t p0 = cells.pos_get((uint32_t) j); // not empty: base kept it
            const bool in_window = p0 > p1 - (int64_t) N;
            const bool sink      = p0 < (int64_t) S;
            if (!in_window && !sink) {
                set_neg_inf(e, L.f32);
                if (count) st.win_masked++;
            } else if (!in_window && count) {
                st.sink_saved++;
            }
        }
    }
    return ref;
}

uint64_t n_diff(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, size_t es) {
    GGML_ASSERT(a.size() == b.size());
    uint64_t n = 0;
    for (size_t i = 0; i < a.size(); i += es) n += memcmp(&a[i], &b[i], es) != 0;
    return n;
}

void fill_meta(const layout & L, std::vector<int32_t> & cells_rec, std::vector<int32_t> & toks_rec) {
    cells_rec.assign((size_t) (L.n_kv*4*L.n_stream), 0);
    toks_rec .assign((size_t) (4*L.n_tps*L.n_stream), 0);
    GGML_ASSERT(llama_kv_cells_fill_kq_vis(cells_rec.data(), toks_rec.data(), L.n_kv, L.n_stream, &L.ub, L.v_cells, L.seq_to_stream));
}

std::vector<uint8_t> run_kq_mask_op(const layout & L, const std::vector<int32_t> & cells_rec, const std::vector<int32_t> & toks_rec,
                                    int32_t swa_type, uint32_t N, uint32_t S, int n_threads) {
    const size_t bytes = cells_rec.size()*4 + toks_rec.size()*4 + (size_t) (L.n_kv*L.n_tps*L.n_stream)*esz(L);
    ggml_init_params ip = { bytes + 16*ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 20), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, L.n_kv, 4, L.n_stream);
    ggml_tensor * tt = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, L.n_tps, L.n_stream);
    memcpy(tc->data, cells_rec.data(), ggml_nbytes(tc));
    memcpy(tt->data, toks_rec.data(),  ggml_nbytes(tt));
    ggml_tensor * out = ggml_kq_mask(ctx, tc, tt, L.f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, L.causal, L.alibi, swa_type, (int32_t) N);
    if (S > 0) ggml_kq_mask_set_sink(out, (int32_t) S);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    GGML_ASSERT(ggml_graph_compute_with_ctx(ctx, gf, n_threads) == GGML_STATUS_SUCCESS);
    std::vector<uint8_t> res(ggml_nbytes(out));
    memcpy(res.data(), out->data, res.size());
    ggml_free(ctx);
    return res;
}

// FA over the ref plane, over the metadata with the window, and over the KQ_MASK op's plane (the bridge), CPU backend
void run_fa(const layout & L, const std::vector<uint8_t> & ref_f16, const std::vector<int32_t> & cells_rec,
            const std::vector<int32_t> & toks_rec, int n_threads) {
    const int64_t D = 32, n_head = 4, n_head_kv = 2;
    const int64_t n_q  = D*L.n_tps*n_head*L.n_stream;
    const int64_t n_kvel = D*L.n_kv*n_head_kv*L.n_stream;
    const size_t bytes = (size_t) n_q*4*4 + (size_t) n_kvel*2*2 + ref_f16.size()*2 + cells_rec.size()*4 + toks_rec.size()*4;
    ggml_init_params ip = { bytes + 64*ggml_tensor_overhead() + ggml_graph_overhead() + (4 << 20), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, L.n_tps, n_head,    L.n_stream);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, L.n_kv,  n_head_kv, L.n_stream);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, L.n_kv,  n_head_kv, L.n_stream);
    ggml_tensor * m  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, L.n_kv, L.n_tps, 1, L.n_stream);
    ggml_tensor * tc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, L.n_kv, 4, L.n_stream);
    ggml_tensor * tt = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, L.n_tps, L.n_stream);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int64_t i = 0; i < n_q; ++i) ((float *) q->data)[i] = nd(g_rng);
    for (ggml_tensor * t : { k, v }) {
        for (int64_t i = 0; i < n_kvel; ++i) ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(nd(g_rng));
    }
    memcpy(m->data,  ref_f16.data(),   ggml_nbytes(m));
    memcpy(tc->data, cells_rec.data(), ggml_nbytes(tc));
    memcpy(tt->data, toks_rec.data(),  ggml_nbytes(tt));
    const float scale = 1.0f/sqrtf((float) D);
    const float max_bias = L.alibi ? 8.0f : 0.0f;

    ggml_tensor * o_ref = ggml_flash_attn_ext(ctx, q, k, v, m, scale, max_bias, 0.0f);
    ggml_flash_attn_ext_set_prec(o_ref, GGML_PREC_F32);
    ggml_tensor * o_vis = ggml_flash_attn_ext_vis(ctx, q, k, v, tc, tt, scale, max_bias, 0.0f, L.causal, L.alibi, 1, (int32_t) L.N);
    if (L.S > 0) ggml_flash_attn_ext_set_vis_sink(o_vis, (int32_t) L.S);
    ggml_flash_attn_ext_set_prec(o_vis, GGML_PREC_F32);
    ggml_tensor * m_op = ggml_kq_mask(ctx, tc, tt, GGML_TYPE_F16, L.causal, L.alibi, 1, (int32_t) L.N);
    if (L.S > 0) ggml_kq_mask_set_sink(m_op, (int32_t) L.S);
    ggml_tensor * o_bridge = ggml_flash_attn_ext(ctx, q, k, v, m_op, scale, max_bias, 0.0f);
    ggml_flash_attn_ext_set_prec(o_bridge, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o_ref);
    ggml_build_forward_expand(gf, o_vis);
    ggml_build_forward_expand(gf, o_bridge);
    GGML_ASSERT(ggml_graph_compute_with_ctx(ctx, gf, n_threads) == GGML_STATUS_SUCCESS);

    const size_t nb = ggml_nbytes(o_ref);
    CHECK(memcmp(o_ref->data, o_vis->data,    nb) == 0, "layout %" PRIu64 ": FA over the metadata (window N %u sink %u) != FA over the ref plane", st.layouts, L.N, L.S);
    CHECK(memcmp(o_ref->data, o_bridge->data, nb) == 0, "layout %" PRIu64 ": FA over the KQ_MASK op plane != FA over the ref plane", st.layouts);
    st.fa_runs++;
    st.fa_elems += nb/4;
    ggml_free(ctx);
}

// exact live tiles of the ref plane per cache stream the ubatch touches, vs the census
void check_census(const layout & L, const std::vector<uint8_t> & ref) {
    const uint32_t tile = 64;
    int64_t n_tiles = 0, n_live = 0;
    llama_kv_cells_window_census(L.n_kv, &L.ub, L.N, L.S, tile, L.v_cells, L.seq_to_stream, &n_tiles, &n_live);

    std::map<uint32_t, std::vector<bool>> live; // cache stream -> tiles
    std::set<uint32_t> touched;
    for (int64_t i = 0; i < L.n_tps*L.n_stream; ++i) {
        const uint32_t cs = L.seq_to_stream[L.ub.seq_id[i][0]];
        touched.insert(cs);
        auto & lv = live[cs];
        const int64_t n = std::min<int64_t>(L.n_kv, L.v_cells[cs].size());
        lv.resize((size_t) ((n + tile - 1)/tile), false);
        for (int64_t j = 0; j < n; ++j) {
            if (!is_neg_inf(ref.data() + (size_t) (i*L.n_kv + j)*esz(L), L.f32)) lv[(size_t) (j/tile)] = true;
        }
    }
    int64_t exp_tiles = 0, exact_live = 0;
    for (uint32_t cs : touched) {
        exp_tiles += (int64_t) live[cs].size();
        for (bool b : live[cs]) exact_live += b ? 1 : 0;
    }
    CHECK(n_tiles == exp_tiles, "layout %" PRIu64 ": census tiles %" PRId64 " != %" PRId64, st.layouts, n_tiles, exp_tiles);
    CHECK(n_live >= exact_live, "layout %" PRIu64 ": census live %" PRId64 " < exact live %" PRId64 " (not an upper bound)", st.layouts, n_live, exact_live);
    st.census_runs++;
    st.census_tiles += n_tiles;
    st.census_live  += n_live;
    st.census_exact_live += exact_live;
    if (L.mode == 0 && !L.is_2d) {
        CHECK(n_live == exact_live, "layout %" PRIu64 ": census live %" PRId64 " != exact live %" PRId64 " on a 1-D decode ubatch", st.layouts, n_live, exact_live);
        st.census_exact_runs++;
    }
}

void run_layout(int n_threads, int64_t fa_budget) {
    layout L;
    build(L, true);
    llama_hparams hp{};
    hp.use_alibi = L.alibi;

    const size_t es = esz(L);
    const std::vector<uint8_t> base = fill_plane(L, hp, 0, LLAMA_SWA_TYPE_NONE, 0, false, L.f32);
    const std::vector<uint8_t> ref  = make_ref(L, base, L.N, L.S, true);
    st.elems += base.size()/es;

    // 1a / 1b: the host plane, band tracking and today's tracking
    const std::vector<uint8_t> win_band = fill_plane(L, hp, L.N, LLAMA_SWA_TYPE_STANDARD, L.S, true,  L.f32);
    const std::vector<uint8_t> win_std  = fill_plane(L, hp, L.N, LLAMA_SWA_TYPE_STANDARD, L.S, false, L.f32);
    uint64_t d = n_diff(win_band, ref, es);
    CHECK(d == 0, "layout %" PRIu64 ": host fill (band) differs from ref in %" PRIu64 " elements (N %u S %u mode %d unified %d n_seq %d 2d %d causal %d alibi %d)",
          st.layouts, d, L.N, L.S, L.mode, L.unified, L.n_seq, L.is_2d, L.causal, L.alibi);
    d = n_diff(win_std, ref, es);
    CHECK(d == 0, "layout %" PRIu64 ": host fill (standard tracking) differs from ref in %" PRIu64 " elements", st.layouts, d);

    // 1c: GGML_OP_KQ_MASK over the metadata
    std::vector<int32_t> cells_rec, toks_rec;
    fill_meta(L, cells_rec, toks_rec);
    const std::vector<uint8_t> op = run_kq_mask_op(L, cells_rec, toks_rec, 1, L.N, L.S, n_threads);
    d = n_diff(op, ref, es);
    CHECK(d == 0, "layout %" PRIu64 ": GGML_OP_KQ_MASK (window) differs from ref in %" PRIu64 " elements", st.layouts, d);

    // negative controls: a cell exactly at the window edge / at the sink edge must make N + 1 / S + 1 visible
    bool edge = false, sink_edge = false;
    for (int64_t i = 0; i < L.n_tps*L.n_stream && !(edge && sink_edge); ++i) {
        const auto & cells = L.v_cells[L.seq_to_stream[L.ub.seq_id[i][0]]];
        const int64_t p1 = L.ub.pos[i];
        for (int64_t j = 0; j < L.n_kv; ++j) {
            if (is_neg_inf(base.data() + (size_t) (i*L.n_kv + j)*es, L.f32)) continue;
            const int64_t p0 = cells.pos_get((uint32_t) j);
            if (p0 == p1 - (int64_t) L.N && p0 >= (int64_t) L.S) edge = true;                       // masked at N, visible at N + 1
            if (p0 == (int64_t) L.S && p1 - p0 >= (int64_t) L.N + 0) sink_edge = true;              // masked at S, a sink at S + 1
        }
    }
    if (edge) {
        st.edge_layouts++;
        const std::vector<uint8_t> w1 = fill_plane(L, hp, L.N + 1, LLAMA_SWA_TYPE_STANDARD, L.S, true, L.f32);
        if (n_diff(w1, ref, es) > 0) st.edge_detected++;
    }
    if (sink_edge) {
        st.sink_edge_layouts++;
        const std::vector<uint8_t> s1 = fill_plane(L, hp, L.N, LLAMA_SWA_TYPE_STANDARD, L.S + 1, true, L.f32);
        if (n_diff(s1, ref, es) > 0) st.sink_edge_detected++;
    }

    // 1d: FA (f16 planes only; FA reads an f16 plane)
    if (!L.f32 && L.n_kv*L.n_tps*L.n_stream <= fa_budget) {
        run_fa(L, ref, cells_rec, toks_rec, n_threads);
    }

    // 3: window beyond every distance = dense
    if ((int64_t) L.N > (int64_t) L.max_pos) {
        d = n_diff(win_band, base, es);
        CHECK(d == 0, "layout %" PRIu64 ": N %u > every position but the windowed plane differs from the dense one in %" PRIu64, st.layouts, L.N, d);
        st.beyond_runs++;
    }

    // 4: tile census (causal layouts, the draft's)
    if (L.causal) {
        check_census(L, ref);
    }
}

// 2: window off = the base commit
void run_base_identity(int n_threads) {
    layout L;
    build(L, false);
    llama_hparams hp{};
    hp.use_alibi = L.alibi;
    const int swa_type = rnd01() < 0.4 ? 0 : rnd(1, 3);
    const uint32_t n_swa = swa_type == 0 ? 0 : (uint32_t) (rnd01() < 0.5 ? rnd(1, 64) : rnd(1, (int) L.max_pos + 2));
    for (bool f32 : { false, true }) {
        std::vector<uint8_t> cur((size_t) (L.n_kv*L.n_tps*L.n_stream)*(f32 ? 4 : 2));
        std::vector<uint8_t> old(cur.size());
        // the call the unchanged callers make (defaults: no sink, no band)
        llama_kv_cells_fill_kq_mask(cur.data(), f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, L.n_kv, L.n_stream, &L.ub, L.causal, hp,
                n_swa, (llama_swa_type) swa_type, L.v_cells, L.seq_to_stream);
        llama_kv_cells_fill_kq_mask_base(old.data(), f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, L.n_kv, L.n_stream, &L.ub, L.causal, hp,
                n_swa, (llama_swa_type) swa_type, L.v_cells, L.seq_to_stream);
        CHECK(cur == old, "base identity: the window-off fill differs from 2a8090bd4's (swa %d n_swa %u f32 %d)", swa_type, n_swa, f32);
        // the draft input's own call with the window off (win_n_swa 0) is the same call
        st.base_elems += cur.size()/(f32 ? 4 : 2);
    }
    st.base_runs++;
    GGML_UNUSED(n_threads);
}

int32_t op_param(const ggml_tensor * t, int i) { // (the ggml accessor is internal: ggml-impl.h)
    return ((const int32_t *) t->op_params)[i];
}

void run_op_param_identity() {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + (1 << 16), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * tc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 256, 4, 1);
    ggml_tensor * tt = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, 8, 1);
    ggml_tensor * q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 32, 8, 4, 1);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 32, 256, 2, 1);
    for (int swa : { 0, 1 }) {
        ggml_tensor * a = ggml_kq_mask(ctx, tc, tt, GGML_TYPE_F16, true, false, swa, swa ? 100 : 0);
        ggml_tensor * b = ggml_kq_mask(ctx, tc, tt, GGML_TYPE_F16, true, false, swa, swa ? 100 : 0);
        if (swa) ggml_kq_mask_set_sink(b, 0);
        CHECK(memcmp(a->op_params, b->op_params, GGML_MAX_OP_PARAMS) == 0, "op params: ggml_kq_mask_set_sink(0) changed the node");
        ggml_tensor * fa = ggml_flash_attn_ext_vis(ctx, q, k, k, tc, tt, 0.125f, 0.0f, 0.0f, true, false, swa, swa ? 100 : 0);
        ggml_tensor * fb = ggml_flash_attn_ext_vis(ctx, q, k, k, tc, tt, 0.125f, 0.0f, 0.0f, true, false, swa, swa ? 100 : 0);
        if (swa) ggml_flash_attn_ext_set_vis_sink(fb, 0);
        CHECK(memcmp(fa->op_params, fb->op_params, GGML_MAX_OP_PARAMS) == 0, "op params: ggml_flash_attn_ext_set_vis_sink(0) changed the node");
        CHECK(op_param(fa, GGML_FA_VIS_PARAM + 5) == 0 && op_param(a, 4) == 0, "op params: sink slot not 0 by default");
    }
    ggml_tensor * c = ggml_kq_mask(ctx, tc, tt, GGML_TYPE_F16, true, false, 1, 100);
    ggml_kq_mask_set_sink(c, 4);
    CHECK(op_param(c, 4) == 4 && op_param(c, 3) == 100, "op params: sink not stored in slot 4");
    ggml_free(ctx);
}

void run_swa_table() {
    // N 4, S 2, query at 10: 0, 1 sinks; 2..6 outside the window (10 - p0 >= 4); 7..10 inside
    for (int p0 = 0; p0 <= 10; ++p0) {
        const bool masked = llama_hparams::is_masked_swa(4, LLAMA_SWA_TYPE_STANDARD, p0, 10, 2);
        const bool want   = p0 >= 2 && p0 <= 6;
        CHECK(masked == want, "is_masked_swa(4, STANDARD, %d, 10, sink 2) = %d, want %d", p0, masked, want);
        const bool m0 = llama_hparams::is_masked_swa(4, LLAMA_SWA_TYPE_STANDARD, p0, 10);
        CHECK(m0 == (p0 <= 6), "is_masked_swa(4, STANDARD, %d, 10) (no sink) = %d", p0, m0);
    }
    // a sink never unmasks for the other window types (the parameter is ignored there)
    CHECK(llama_hparams::is_masked_swa(4, LLAMA_SWA_TYPE_CHUNKED, 0, 9, 8) == llama_hparams::is_masked_swa(4, LLAMA_SWA_TYPE_CHUNKED, 0, 9),
          "is_masked_swa: sinks leak into the chunked window");
}

// ---------------------------------------------------------------------------------------------
// device mode

ggml_backend_t g_dev = nullptr;

struct dev_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    dev_graph() {
        ggml_init_params ip = { 64*ggml_tensor_overhead() + 4*ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        GGML_ASSERT(ctx);
    }
    ~dev_graph() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
    void alloc() {
        buf = ggml_backend_alloc_ctx_tensors(ctx, g_dev);
        GGML_ASSERT(buf && "device allocation failed");
    }
};

std::vector<uint8_t> dev_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> r(ggml_nbytes(t));
    ggml_backend_tensor_get(t, r.data(), 0, r.size());
    return r;
}

double dev_compute_ms(ggml_cgraph * gf) {
    const auto t0 = std::chrono::steady_clock::now();
    GGML_ASSERT(ggml_backend_graph_compute(g_dev, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(g_dev);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct dev_stats {
    uint64_t op_runs = 0, op_diff = 0, fa_runs = 0, fa_vis_diff = 0, fa_bridge_diff = 0, unsupported = 0;
} g_ds;

// one FA problem on the device: q/k/v random, the ref plane uploaded, the metadata uploaded; three graphs
struct dev_fa {
    dev_graph g;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * m = nullptr, * tc = nullptr, * tt = nullptr;
    ggml_tensor * m_op = nullptr, * o_plane = nullptr, * o_vis = nullptr, * o_bridge = nullptr;
    ggml_cgraph * gf_op = nullptr, * gf_plane = nullptr, * gf_vis = nullptr, * gf_bridge = nullptr;
    bool supported = true;

    void build(int64_t n_kv, int64_t n_tps, int64_t n_stream, int64_t D, int64_t n_head, int64_t n_head_kv, bool causal, bool alibi,
               int swa_type, uint32_t N, uint32_t S, bool f32_plane, const std::vector<uint8_t> & plane_f16,
               const std::vector<int32_t> & cells_rec, const std::vector<int32_t> & toks_rec) {
        q  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, D, n_tps, n_head,    n_stream);
        k  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, D, n_kv,  n_head_kv, n_stream);
        v  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, D, n_kv,  n_head_kv, n_stream);
        m  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, n_kv, n_tps, 1, n_stream);
        tc = ggml_new_tensor_3d(g.ctx, GGML_TYPE_I32, n_kv, 4, n_stream);
        tt = ggml_new_tensor_3d(g.ctx, GGML_TYPE_I32, 4, n_tps, n_stream);
        const float scale = 1.0f/sqrtf((float) D);
        const float max_bias = alibi ? 8.0f : 0.0f;
        m_op = ggml_kq_mask(g.ctx, tc, tt, f32_plane ? GGML_TYPE_F32 : GGML_TYPE_F16, causal, alibi, swa_type, (int32_t) N);
        if (S > 0) ggml_kq_mask_set_sink(m_op, (int32_t) S);
        ggml_tensor * m_br = m_op;
        if (f32_plane) { // the bridge FA reads an f16 plane: a second node
            m_br = ggml_kq_mask(g.ctx, tc, tt, GGML_TYPE_F16, causal, alibi, swa_type, (int32_t) N);
            if (S > 0) ggml_kq_mask_set_sink(m_br, (int32_t) S);
        }
        o_plane = ggml_flash_attn_ext(g.ctx, q, k, v, m, scale, max_bias, 0.0f);
        ggml_flash_attn_ext_set_prec(o_plane, GGML_PREC_F32);
        o_vis = ggml_flash_attn_ext_vis(g.ctx, q, k, v, tc, tt, scale, max_bias, 0.0f, causal, alibi, swa_type, (int32_t) N);
        if (S > 0) ggml_flash_attn_ext_set_vis_sink(o_vis, (int32_t) S);
        ggml_flash_attn_ext_set_prec(o_vis, GGML_PREC_F32);
        o_bridge = ggml_flash_attn_ext(g.ctx, q, k, v, m_br, scale, max_bias, 0.0f);
        ggml_flash_attn_ext_set_prec(o_bridge, GGML_PREC_F32);
        supported = ggml_backend_supports_op(g_dev, m_op) && ggml_backend_supports_op(g_dev, o_plane) &&
                    ggml_backend_supports_op(g_dev, o_vis) && ggml_backend_supports_op(g_dev, o_bridge);
        if (!supported) {
            return;
        }
        gf_op     = ggml_new_graph(g.ctx); ggml_build_forward_expand(gf_op,     m_op);
        gf_plane  = ggml_new_graph(g.ctx); ggml_build_forward_expand(gf_plane,  o_plane);
        gf_vis    = ggml_new_graph(g.ctx); ggml_build_forward_expand(gf_vis,    o_vis);
        gf_bridge = ggml_new_graph(g.ctx); ggml_build_forward_expand(gf_bridge, o_bridge);
        g.alloc();
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> qf((size_t) ggml_nelements(q));
        for (auto & x : qf) x = nd(g_rng);
        ggml_backend_tensor_set(q, qf.data(), 0, ggml_nbytes(q));
        std::vector<ggml_fp16_t> kf((size_t) ggml_nelements(k));
        for (ggml_tensor * t : { k, v }) {
            for (auto & x : kf) x = ggml_fp32_to_fp16(nd(g_rng));
            ggml_backend_tensor_set(t, kf.data(), 0, ggml_nbytes(t));
        }
        ggml_backend_tensor_set(m,  plane_f16.data(), 0, ggml_nbytes(m));
        ggml_backend_tensor_set(tc, cells_rec.data(), 0, ggml_nbytes(tc));
        ggml_backend_tensor_set(tt, toks_rec.data(),  0, ggml_nbytes(tt));
    }
};

// the identity on the device over random layouts (as run_layout's 1c / 1d)
void run_dev_layout() {
    layout L;
    build(L, true);
    llama_hparams hp{};
    hp.use_alibi = L.alibi;
    const std::vector<uint8_t> base = fill_plane(L, hp, 0, LLAMA_SWA_TYPE_NONE, 0, false, L.f32);
    const std::vector<uint8_t> ref  = make_ref(L, base, L.N, L.S, false);
    std::vector<uint8_t> ref16 = ref;
    if (L.f32) { // the FA plane is f16: the same predicate in f16
        const std::vector<uint8_t> b16 = fill_plane(L, hp, 0, LLAMA_SWA_TYPE_NONE, 0, false, false);
        layout & Lm = L; Lm.f32 = false; ref16 = make_ref(Lm, b16, L.N, L.S, false); Lm.f32 = true;
    }
    std::vector<int32_t> cells_rec, toks_rec;
    fill_meta(L, cells_rec, toks_rec);
    dev_fa f;
    f.build(L.n_kv, L.n_tps, L.n_stream, 64, 4, 2, L.causal, L.alibi, 1, L.N, L.S, L.f32, ref16, cells_rec, toks_rec);
    if (!f.supported) {
        g_ds.unsupported++;
        return;
    }
    dev_compute_ms(f.gf_op);
    const uint64_t d = n_diff(dev_bytes(f.m_op), ref, esz(L));
    CHECK(d == 0, "device layout %" PRIu64 ": GGML_OP_KQ_MASK (window N %u S %u) differs from ref in %" PRIu64 " elements", st.layouts, L.N, L.S, d);
    g_ds.op_runs++;
    g_ds.op_diff += d;
    dev_compute_ms(f.gf_plane);
    dev_compute_ms(f.gf_vis);
    dev_compute_ms(f.gf_bridge);
    const std::vector<uint8_t> a = dev_bytes(f.o_plane), b = dev_bytes(f.o_vis), c = dev_bytes(f.o_bridge);
    const uint64_t dv = n_diff(a, b, 4), db = n_diff(a, c, 4);
    CHECK(dv == 0 && db == 0, "device layout %" PRIu64 ": FA plane vs vis %" PRIu64 " / vs bridge %" PRIu64 " differing outputs", st.layouts, dv, db);
    g_ds.fa_runs++;
    g_ds.fa_vis_diff += dv;
    g_ds.fa_bridge_diff += db;
}

// the skip probe at the draft's shapes
void run_dev_perf(const std::vector<int64_t> & nkvs, int reps) {
    printf("  skip probe on %s (24/2 heads, D 256, f16 K/V, one sequence in position order, sink 4; ms = median of %d):\n",
           ggml_backend_name(g_dev), reps);
    printf("    %8s %5s %7s | %10s %10s | %8s %8s | %s\n", "n_kv", "n_tps", "window", "bridge ms", "vis ms", "bridge x", "vis x", "identical");
    for (int64_t n_kv : nkvs) {
        for (int64_t n_tps : { (int64_t) 1, (int64_t) 6 }) {
            double dense_br = 0, dense_vis = 0;
            for (uint32_t N : { 0u, 65536u, 16384u }) {
                const uint32_t S = N ? 4u : 0u;
                // cells 0..n_kv-1 at positions 0..n_kv-1 of sequence 0; the tokens are the last n_tps cells
                layout L;
                L.n_kv = n_kv; L.n_tps = n_tps; L.n_stream = 1; L.causal = true; L.N = N; L.S = S;
                L.seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
                L.v_cells.assign(1, llama_kv_cells());
                L.v_cells[0].resize((uint32_t) n_kv);
                for (int64_t j = 0; j < n_kv; ++j) { L.v_cells[0].pos_set((uint32_t) j, (llama_pos) j); L.v_cells[0].seq_add((uint32_t) j, 0); }
                L.pos.resize((size_t) n_tps); L.n_seq_id.assign((size_t) n_tps, 1); L.seq_data.assign((size_t) n_tps, 0);
                L.seq_ptr.resize((size_t) n_tps); L.output.assign((size_t) n_tps, 1); L.seq_idx.assign(LLAMA_MAX_SEQ, -1);
                L.seq_unq = { 0 }; L.seq_idx[0] = 0;
                for (int64_t i = 0; i < n_tps; ++i) { L.pos[(size_t) i] = (llama_pos) (n_kv - n_tps + i); L.seq_ptr[(size_t) i] = &L.seq_data[(size_t) i]; }
                L.ub = llama_ubatch{};
                L.ub.n_tokens = (uint32_t) n_tps; L.ub.n_seq_tokens = (uint32_t) n_tps; L.ub.n_seqs = 1; L.ub.n_seqs_unq = 1; L.ub.n_pos = 1;
                L.ub.pos = L.pos.data(); L.ub.n_seq_id = L.n_seq_id.data(); L.ub.seq_id = L.seq_ptr.data();
                L.ub.seq_id_unq = L.seq_unq.data(); L.ub.seq_idx = L.seq_idx.data(); L.ub.output = L.output.data();
                llama_hparams hp{};
                const std::vector<uint8_t> base = fill_plane(L, hp, 0, LLAMA_SWA_TYPE_NONE, 0, false, false);
                const std::vector<uint8_t> ref  = N ? make_ref(L, base, N, S, false) : base;
                std::vector<int32_t> cells_rec, toks_rec;
                fill_meta(L, cells_rec, toks_rec);
                dev_fa f;
                f.build(n_kv, n_tps, 1, 256, 24, 2, true, false, N ? 1 : 0, N, S, false, ref, cells_rec, toks_rec);
                if (!f.supported) {
                    printf("    %8" PRId64 " %5" PRId64 " %7u | unsupported\n", n_kv, n_tps, N);
                    continue;
                }
                std::vector<double> tb, tv;
                for (int r = 0; r < reps + 2; ++r) { // 2 warm-ups
                    const double b = dev_compute_ms(f.gf_bridge);
                    const double v = dev_compute_ms(f.gf_vis);
                    if (r >= 2) { tb.push_back(b); tv.push_back(v); }
                }
                std::sort(tb.begin(), tb.end());
                std::sort(tv.begin(), tv.end());
                const double mb = tb[tb.size()/2], mv = tv[tv.size()/2];
                if (N == 0) { dense_br = mb; dense_vis = mv; }
                dev_compute_ms(f.gf_plane);
                const std::vector<uint8_t> a = dev_bytes(f.o_plane), b = dev_bytes(f.o_vis), c = dev_bytes(f.o_bridge);
                const bool same = n_diff(a, b, 4) == 0 && n_diff(a, c, 4) == 0;
                CHECK(same, "skip probe n_kv %" PRId64 " n_tps %" PRId64 " window %u: plane / vis / bridge outputs differ", n_kv, n_tps, N);
                printf("    %8" PRId64 " %5" PRId64 " %7u | %10.3f %10.3f | %8.2f %8.2f | %s\n", n_kv, n_tps, N, mb, mv,
                       N ? dense_br/mb : 1.0, N ? dense_vis/mv : 1.0, same ? "yes" : "NO");
                fflush(stdout);
            }
        }
    }
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> pos_args;
    const char * dev_name = nullptr;
    bool perf = false;
    int dev_layouts = 300, perf_reps = 15;
    std::vector<int64_t> perf_nkv = { 32768, 213248, 393216 };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) { dev_name = argv[++i]; }
        else if (a == "--perf") { perf = true; }
        else if (a == "--dev-layouts" && i + 1 < argc) { dev_layouts = atoi(argv[++i]); }
        else if (a == "--perf-reps" && i + 1 < argc) { perf_reps = std::max(1, atoi(argv[++i])); }
        else if (a == "--perf-nkv" && i + 1 < argc) {
            perf_nkv.clear();
            std::string v = argv[++i];
            for (size_t p0 = 0; p0 <= v.size();) {
                const size_t p1 = v.find(',', p0);
                perf_nkv.push_back(atoll(v.substr(p0, p1 == std::string::npos ? std::string::npos : p1 - p0).c_str()));
                if (p1 == std::string::npos) break;
                p0 = p1 + 1;
            }
        } else { pos_args.push_back(a); }
    }
    const int      n_layouts = pos_args.size() > 0 ? atoi(pos_args[0].c_str()) : 3000;
    const uint64_t seed      = pos_args.size() > 1 ? strtoull(pos_args[1].c_str(), nullptr, 10) : 7;
    const int      n_threads = pos_args.size() > 2 ? atoi(pos_args[2].c_str()) : 4;
    const int64_t  fa_budget = pos_args.size() > 3 ? atoll(pos_args[3].c_str()) : 49152;
    g_rng.seed(seed);

    ggml_cpu_init();

    run_swa_table();
    run_op_param_identity();
    for (int i = 0; i < n_layouts; ++i) {
        run_layout(n_threads, fa_budget);
        if (i % 3 == 0) {
            run_base_identity(n_threads);
        }
    }

    printf("test-mtp-draft-window: %" PRIu64 " layouts (seed %" PRIu64 "): unified %" PRIu64 ", per-stream %" PRIu64 ", multi-seq ubatches %" PRIu64
           ", decode %" PRIu64 " / catch-up %" PRIu64 " / prefill %" PRIu64 ", M-RoPE %" PRIu64 ", non-causal %" PRIu64 ", ALiBi %" PRIu64
           ", rollbacks %" PRIu64 ", shared prefixes %" PRIu64 "\n",
           st.layouts, seed, st.unified, st.per_stream, st.multi_seq_ub, st.decode, st.catchup, st.prefill, st.is_2d, st.non_causal,
           st.alibi, st.rollbacks, st.shared);
    printf("  1. identity vs ref over %" PRIu64 " elements (%" PRIu64 " masked by the window, %" PRIu64 " kept by a sink): host fill band + "
           "standard, GGML_OP_KQ_MASK; FA plane = metadata = bridge on %" PRIu64 " layouts (%" PRIu64 " f32 outputs)\n",
           st.elems, st.win_masked, st.sink_saved, st.fa_runs, st.fa_elems);
    printf("     negative controls: window edge %" PRIu64 "/%" PRIu64 " detected (N+1), sink edge %" PRIu64 "/%" PRIu64 " detected (S+1)\n",
           st.edge_detected, st.edge_layouts, st.sink_edge_detected, st.sink_edge_layouts);
    printf("  2. window off = 2a8090bd4: %" PRIu64 " layouts x F16/F32 (%" PRIu64 " elements), op params of unset/zero sinks identical\n",
           st.base_runs, st.base_elems);
    printf("  3. window beyond every position = dense: %" PRIu64 " layouts\n", st.beyond_runs);
    printf("  4. tile census: %" PRIu64 " layouts, %" PRIu64 " tiles, census live %" PRIu64 " >= exact live %" PRIu64 " (exact on %" PRIu64 " 1-D decode layouts)\n",
           st.census_runs, st.census_tiles, st.census_live, st.census_exact_live, st.census_exact_runs);

    CHECK(st.edge_layouts == 0 || st.edge_detected == st.edge_layouts, "negative control: %" PRIu64 " of %" PRIu64 " window-edge layouts undetected",
          st.edge_layouts - st.edge_detected, st.edge_layouts);
    CHECK(st.sink_edge_layouts == 0 || st.sink_edge_detected == st.sink_edge_layouts, "negative control: sink edge undetected");
    if (n_layouts >= 1000) {
        CHECK(st.edge_layouts >= 50 && st.sink_edge_layouts >= 20 && st.win_masked > 0 && st.sink_saved > 0 && st.fa_runs >= 100 &&
              st.multi_seq_ub >= 100 && st.rollbacks >= 100 && st.beyond_runs >= 50,
              "coverage too thin: edge %" PRIu64 " sink-edge %" PRIu64 " fa %" PRIu64 " multi-seq %" PRIu64 " rollbacks %" PRIu64 " beyond %" PRIu64,
              st.edge_layouts, st.sink_edge_layouts, st.fa_runs, st.multi_seq_ub, st.rollbacks, st.beyond_runs);
    }

    if (dev_name != nullptr) {
        ggml_backend_load_all();
        g_dev = ggml_backend_init_by_name(dev_name, nullptr);
        if (!g_dev) {
            fprintf(stderr, "test-mtp-draft-window: no backend '%s'\n", dev_name);
            return 1;
        }
        const uint64_t fail0 = st.fail;
        for (int i = 0; i < dev_layouts; ++i) {
            run_dev_layout();
        }
        printf("  6. DEVICE %s: %" PRIu64 " layouts, GGML_OP_KQ_MASK vs ref DIFFERING: %" PRIu64 "; FA plane vs vis DIFFERING BITS: %" PRIu64
               ", plane vs bridge DIFFERING BITS: %" PRIu64 " (unsupported %" PRIu64 ")\n", ggml_backend_name(g_dev), g_ds.op_runs,
               g_ds.op_diff, g_ds.fa_vis_diff, g_ds.fa_bridge_diff, g_ds.unsupported);
        CHECK(g_ds.op_runs > 0 && g_ds.unsupported == 0, "device: %" PRIu64 " layouts unsupported", g_ds.unsupported);
        if (perf) {
            run_dev_perf(perf_nkv, perf_reps);
        }
        printf("  device checks: %s\n", st.fail == fail0 ? "PASS" : "FAIL");
        ggml_backend_free(g_dev);
    }

    if (st.fail) {
        printf("test-mtp-draft-window: FAILED (%" PRIu64 " checks)\n", st.fail);
        return 1;
    }
    printf("test-mtp-draft-window: OK\n");
    return 0;
}
