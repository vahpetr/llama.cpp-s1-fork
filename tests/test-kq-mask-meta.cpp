// [TAG_KQ_VIS] differential test: GGML_OP_KQ_MASK over the metadata records (llama_kv_cells_fill_kq_vis)
// against the unchanged host mask fill (llama_kv_cells_fill_kq_mask = the body of
// llama_kv_cache::set_input_kq_mask / set_input_kq_mask_impl), bit for bit, on the CPU backend.
//
// usage: test-kq-mask-meta [n_random_layouts = 10000] [seed = 42] [n_threads = 4]
//
// Every layout builds synthetic llama_kv_cells (holes, seqless used cells, multi-seq cells, seq ids
// >= 32 in cells, random positions up to ~2^20, M-RoPE ext), a synthetic llama_ubatch (single- and
// multi-sequence, unified and per-stream blocks, 1-D and M-RoPE positions) and random attention
// parameters (causal / non-causal, SWA none/standard/chunked/symmetric, ALiBi, F16/F32 mask).
// A negative control flips one seq bit in the cell records and requires the comparison to fail.

#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

struct layout_desc {
    // forced properties (-1 = random)
    int64_t n_kv       = -1;
    int64_t n_tps      = -1;
    int     n_stream   = -1; // ubatch stream blocks
    int     unified    = -1;
    int     n_seq      = -1; // sequences that exist
    int     n_useq     = -1; // sequences in the ubatch
    int     is_2d      = -1;
    int     causal     = -1;
    int     swa_type   = -1;
    int     alibi      = -1;
    int     f32        = -1;
    int     pos_base   = -1; // -1 random, else value
    int     decode     = -1; // 1 = consecutive positions right after each seq's cells
    double  hole       = -1.0;
    const char * name  = "random";
};

struct stats_t {
    uint64_t layouts = 0, elements = 0, diff = 0, kept = 0, dropped = 0, alibi_vals = 0;
    uint64_t by_ns1 = 0, by_nsN = 0, unified = 0, multi_seq_ub = 0, is_2d = 0, non_causal = 0, alibi = 0, f32 = 0;
    uint64_t swa[4] = {0, 0, 0, 0};
    uint64_t cells_empty = 0, cells_seqless = 0, cells_multibit = 0, cells_seq_ge32 = 0, cells_used = 0;
    uint64_t ties_2d = 0; // (cell, token) pairs with p0 == p1 under 2-D positions and a seq match
    int64_t  max_n_kv = 0, max_n_tps = 0, max_pos = 0;
    uint64_t ctrl_runs = 0, ctrl_detected = 0;
    uint64_t precond_runs = 0, precond_rejected = 0;
};

struct built_layout {
    int64_t n_kv, n_tps, n_stream;
    bool    is_2d, causal, alibi, f32;
    int     swa_type;
    uint32_t n_swa;

    std::vector<llama_kv_cells> v_cells;
    std::vector<uint32_t>       seq_to_stream;

    // ubatch storage
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_data;  // 4 per token
    std::vector<llama_seq_id *> seq_ptr;
    std::vector<llama_seq_id>   seq_unq;
    std::vector<int32_t>        seq_idx;
    std::vector<int8_t>         output;
    llama_ubatch                ub{};
};

int rnd_int(std::mt19937_64 & rng, int lo, int hi) { // inclusive
    return std::uniform_int_distribution<int>(lo, hi)(rng);
}
double rnd01(std::mt19937_64 & rng) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

void build(built_layout & L, const layout_desc & d, std::mt19937_64 & rng, stats_t & st) {
    const int n_seq = d.n_seq >= 0 ? d.n_seq : rnd_int(rng, 1, rnd01(rng) < 0.2 ? 32 : 8);

    const bool unified = d.unified >= 0 ? d.unified != 0 : rnd01(rng) < 0.6;

    // seq -> cache stream: unified = all 0; otherwise a random map onto 1..n_seq streams (the
    // non-unified cache uses the identity, which is one of the maps drawn here)
    int n_cache_streams = 1;
    L.seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
    if (!unified) {
        if (rnd01(rng) < 0.5) {
            n_cache_streams = n_seq;
            for (int s = 0; s < n_seq; ++s) L.seq_to_stream[s] = s;
        } else {
            n_cache_streams = rnd_int(rng, 1, n_seq);
            for (int s = 0; s < n_seq; ++s) L.seq_to_stream[s] = rnd_int(rng, 0, n_cache_streams - 1);
        }
    }

    // ubatch sequences and stream blocks
    std::vector<int> seqs(n_seq);
    for (int s = 0; s < n_seq; ++s) seqs[s] = s;
    std::shuffle(seqs.begin(), seqs.end(), rng);

    // blocks: each block is one cache stream; its tokens come from seqs of that stream
    std::map<uint32_t, std::vector<int>> stream_seqs;
    for (int s : seqs) stream_seqs[L.seq_to_stream[s]].push_back(s);
    std::vector<uint32_t> streams;
    for (auto & kv : stream_seqs) streams.push_back(kv.first);
    std::shuffle(streams.begin(), streams.end(), rng);

    int n_stream = unified ? 1 : (d.n_stream >= 0 ? std::min<int>(d.n_stream, (int) streams.size()) : rnd_int(rng, 1, std::min<int>(4, (int) streams.size())));
    if (unified) n_stream = 1;
    streams.resize(n_stream);

    int64_t n_tps = d.n_tps;
    if (n_tps < 0) {
        const double r = rnd01(rng);
        n_tps = r < 0.30 ? rnd_int(rng, 1, 6) : r < 0.80 ? rnd_int(rng, 1, 64) : r < 0.97 ? rnd_int(rng, 64, 512) : rnd_int(rng, 512, 2048);
        if (n_stream > 1 && n_tps > 256) n_tps = rnd_int(rng, 1, 256);
    }

    int64_t n_kv = d.n_kv;
    if (n_kv < 0) {
        const double r = rnd01(rng);
        n_kv = r < 0.3 ? rnd_int(rng, 1, 256) : r < 0.8 ? rnd_int(rng, 1, 1024) : rnd_int(rng, 1024, 4096);
        if (n_tps > 512 && n_kv > 2048) n_kv = rnd_int(rng, 256, 2048) ;
        if (rnd01(rng) < 0.3) n_kv = ((n_kv + 255)/256)*256; // the cache pads to 256
    }

    L.n_kv = n_kv; L.n_tps = n_tps; L.n_stream = n_stream;
    L.is_2d  = d.is_2d  >= 0 ? d.is_2d  != 0 : rnd01(rng) < 0.25;
    L.causal = d.causal >= 0 ? d.causal != 0 : rnd01(rng) < 0.8;
    L.alibi  = d.alibi  >= 0 ? d.alibi  != 0 : rnd01(rng) < 0.1;
    L.f32    = d.f32    >= 0 ? d.f32    != 0 : rnd01(rng) < 0.15;
    L.swa_type = d.swa_type >= 0 ? d.swa_type : (rnd01(rng) < 0.55 ? 0 : rnd_int(rng, 1, 3));
    L.n_swa  = L.swa_type == 0 ? 0 : (uint32_t) (rnd01(rng) < 0.5 ? rnd_int(rng, 1, 64) : rnd_int(rng, 64, 4096));

    const int max_pos_base = (1 << 20) - 20000;
    const int pos_base = d.pos_base >= 0 ? d.pos_base : (rnd01(rng) < 0.5 ? rnd_int(rng, 0, 64) : rnd_int(rng, 0, max_pos_base));
    const int span = rnd01(rng) < 0.3 ? rnd_int(rng, 1, 48) : (int) std::max<int64_t>(8, rnd01(rng) < 0.5 ? n_kv : 4*n_kv);
    const double hole = d.hole >= 0 ? d.hole : (rnd01(rng) < 0.3 ? 0.0 : rnd01(rng) < 0.5 ? 0.05 : 0.5);

    // cells
    L.v_cells.clear();
    L.v_cells.resize(n_cache_streams);
    for (auto & cells : L.v_cells) cells.resize((uint32_t) (n_kv + rnd_int(rng, 0, 300)));

    // ubatch tokens
    const uint32_t n_tokens = (uint32_t) (n_tps*n_stream);
    const uint32_t n_pos    = L.is_2d ? 4 : 1;
    L.pos.assign((size_t) n_tokens*n_pos, 0);
    L.n_seq_id.assign(n_tokens, 1);
    L.seq_data.assign((size_t) n_tokens*4, 0);
    L.seq_ptr.assign(n_tokens, nullptr);
    L.output.assign(n_tokens, 1);
    L.seq_idx.assign(LLAMA_MAX_SEQ, -1);
    L.seq_unq.clear();

    const bool decode = d.decode >= 0 ? d.decode != 0 : rnd01(rng) < 0.3;
    std::map<int, llama_pos> seq_next; // per seq: next position (decode mode) or window start
    std::map<int, std::pair<int,int>> seq_img; // per seq: 2-D grid cursor

    int n_useq_total = 0;
    for (int64_t s = 0; s < n_stream; ++s) {
        auto & cand = stream_seqs[streams[s]];
        int n_useq = d.n_useq >= 0 ? std::min<int>(d.n_useq, (int) cand.size()) : rnd_int(rng, 1, std::min<int>((int) cand.size(), rnd01(rng) < 0.5 ? 1 : 6));
        n_useq = std::max(1, n_useq);
        std::vector<int> useq(cand.begin(), cand.begin() + n_useq);
        n_useq_total += n_useq;

        const int mode = rnd_int(rng, 0, 3); // 0: grouped consecutive, 1: interleaved, 2: random order, 3: image block runs
        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = s*n_tps + ii;
            int sq;
            if (mode == 0) sq = useq[(size_t) (ii*n_useq/n_tps)];
            else sq = useq[(size_t) rnd_int(rng, 0, n_useq - 1)];

            if (seq_next.find(sq) == seq_next.end()) {
                seq_next[sq] = decode ? pos_base + span : pos_base + rnd_int(rng, 0, span);
                seq_img[sq] = {0, 0};
            }
            llama_pos p;
            int x = 0, y = 0;
            if (mode == 2) {
                p = pos_base + rnd_int(rng, 0, span + 64);
                x = rnd_int(rng, 0, 3); y = rnd_int(rng, 0, 3);
            } else if (mode == 3 && L.is_2d) {
                // an image block: equal temporal position, raster (x, y); a new block now and then
                auto & g = seq_img[sq];
                p = seq_next[sq];
                x = g.first; y = g.second;
                if (++g.first >= 4) { g.first = 0; if (++g.second >= 3) { g.second = 0; seq_next[sq] += 1; } }
            } else {
                p = seq_next[sq]++;
                if (L.is_2d) { x = rnd_int(rng, 0, 2); y = rnd_int(rng, 0, 2); }
            }
            L.pos[i] = p;
            if (L.is_2d) {
                L.pos[i + n_tokens]   = y;
                L.pos[i + n_tokens*2] = x;
                L.pos[i + n_tokens*3] = rnd_int(rng, 0, 9);
            }
            st.max_pos = std::max<int64_t>(st.max_pos, p);

            L.seq_data[i*4 + 0] = sq;
            if (rnd01(rng) < 0.05) { // extra seq ids (only seq_id[i][0] is read by the fill)
                L.n_seq_id[i] = 2;
                L.seq_data[i*4 + 1] = rnd_int(rng, 0, 31);
            }
            L.seq_ptr[i] = &L.seq_data[i*4];

            if (L.seq_idx[sq] < 0) { L.seq_idx[sq] = (int32_t) L.seq_unq.size(); L.seq_unq.push_back(sq); }

            // the token's own cell (the ubatch is stored before set_input runs), most of the time
            auto & cells = L.v_cells[L.seq_to_stream[sq]];
            if (rnd01(rng) < 0.85) {
                for (int tries = 0; tries < 8; ++tries) {
                    const uint32_t j = (uint32_t) rnd_int(rng, 0, (int) n_kv - 1);
                    if (cells.is_empty(j)) {
                        cells.pos_set(j, p);
                        cells.seq_add(j, sq);
                        llama_kv_cell_ext e; e.x = x; e.y = y;
                        cells.ext_set(j, e);
                        break;
                    }
                }
            }
        }
    }
    if (n_useq_total > 1) st.multi_seq_ub++;

    // the other cells
    for (auto & cells : L.v_cells) {
        for (uint32_t j = 0; j < cells.size(); ++j) {
            if (!cells.is_empty(j)) continue;
            if (rnd01(rng) < hole) continue;
            llama_pos p;
            // near the ubatch positions most of the time, so p0 == p1 ties and the +32 window are hit
            if (rnd01(rng) < 0.5 && !L.pos.empty()) {
                p = std::max<llama_pos>(0, L.pos[(size_t) rnd_int(rng, 0, (int) n_tokens - 1)] + rnd_int(rng, -40, 8));
            } else {
                p = pos_base + rnd_int(rng, 0, span);
            }
            cells.pos_set(j, p);
            llama_kv_cell_ext e; e.x = rnd_int(rng, 0, 3); e.y = rnd_int(rng, 0, 3);
            cells.ext_set(j, e);
            const double r = rnd01(rng);
            if (r < 0.03) {
                continue; // used, no sequence
            }
            cells.seq_add(j, rnd_int(rng, 0, n_seq - 1));
            if (r < 0.25) { // shared prefix: more bits
                const int extra = rnd_int(rng, 1, 4);
                for (int k = 0; k < extra; ++k) {
                    const int sq = rnd_int(rng, 0, n_seq - 1);
                    if (!cells.seq_has(j, sq)) cells.seq_add(j, sq);
                }
            }
            if (r > 0.95) { // a seq id >= 32 (truncated by the records; never in the ubatch)
                const int sq = rnd_int(rng, 32, LLAMA_MAX_SEQ - 1);
                if (!cells.seq_has(j, sq)) cells.seq_add(j, sq);
            }
        }
        // census over the first n_kv cells (the ones the mask covers)
        for (uint32_t j = 0; j < (uint32_t) n_kv; ++j) {
            if (cells.is_empty(j)) { st.cells_empty++; continue; }
            st.cells_used++;
            int nb = 0; bool ge32 = false;
            for (int k = 0; k < LLAMA_MAX_SEQ; ++k) if (cells.seq_has(j, k)) { nb++; ge32 |= k >= 32; }
            if (nb == 0) st.cells_seqless++;
            if (nb > 1)  st.cells_multibit++;
            if (ge32)    st.cells_seq_ge32++;
        }
    }

    L.ub = llama_ubatch{};
    L.ub.b_equal_seqs = n_stream > 1;
    L.ub.n_tokens     = n_tokens;
    L.ub.n_seq_tokens = (uint32_t) n_tps;
    L.ub.n_seqs       = (uint32_t) n_stream;
    L.ub.n_seqs_unq   = (uint32_t) L.seq_unq.size();
    L.ub.n_pos        = n_pos;
    L.ub.token        = nullptr;
    L.ub.embd         = nullptr;
    L.ub.pos          = L.pos.data();
    L.ub.n_seq_id     = L.n_seq_id.data();
    L.ub.seq_id       = L.seq_ptr.data();
    L.ub.seq_id_unq   = L.seq_unq.data();
    L.ub.seq_idx      = L.seq_idx.data();
    L.ub.output       = L.output.data();

    st.max_n_kv  = std::max(st.max_n_kv, n_kv);
    st.max_n_tps = std::max(st.max_n_tps, n_tps);
    (n_stream == 1 ? st.by_ns1 : st.by_nsN)++;
    if (unified)    st.unified++;
    if (L.is_2d)    st.is_2d++;
    if (!L.causal)  st.non_causal++;
    if (L.alibi)    st.alibi++;
    if (L.f32)      st.f32++;
    st.swa[L.swa_type]++;
}

// run GGML_OP_KQ_MASK on the CPU backend; returns the mask bytes
std::vector<uint8_t> run_op(const built_layout & L, const std::vector<int32_t> & cells_rec, const std::vector<int32_t> & toks_rec, int n_threads) {
    const size_t esz   = L.f32 ? 4 : 2;
    const size_t bytes = cells_rec.size()*4 + toks_rec.size()*4 + (size_t) (L.n_kv*L.n_tps*L.n_stream)*esz;

    ggml_init_params ip = { bytes + 16*ggml_tensor_overhead() + ggml_graph_overhead() + (1 << 20), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    GGML_ASSERT(ctx);

    ggml_tensor * tc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, L.n_kv,  L.n_stream);
    ggml_tensor * tt = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, L.n_tps, L.n_stream);
    memcpy(tc->data, cells_rec.data(), ggml_nbytes(tc));
    memcpy(tt->data, toks_rec.data(),  ggml_nbytes(tt));

    ggml_tensor * out = ggml_kq_mask(ctx, tc, tt, L.f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, L.causal, L.alibi, L.swa_type, (int32_t) L.n_swa);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> res(ggml_nbytes(out));
    memcpy(res.data(), out->data, res.size());
    ggml_free(ctx);
    return res;
}

uint64_t count_diff(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, bool f32) {
    GGML_ASSERT(a.size() == b.size());
    uint64_t n = 0;
    const size_t esz = f32 ? 4 : 2;
    for (size_t i = 0; i < a.size(); i += esz) {
        n += memcmp(&a[i], &b[i], esz) != 0;
    }
    return n;
}

// returns the number of differing elements for one layout (and runs the negative control if asked)
uint64_t run_layout(const layout_desc & d, std::mt19937_64 & rng, stats_t & st, int n_threads, bool control) {
    built_layout L;
    build(L, d, rng, st);

    llama_hparams hp{};
    hp.use_alibi = L.alibi;

    const size_t esz = L.f32 ? 4 : 2;
    std::vector<uint8_t> ref((size_t) (L.n_kv*L.n_tps*L.n_stream)*esz, 0xAB);
    llama_kv_cells_fill_kq_mask(ref.data(), L.f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, L.n_kv, L.n_stream, &L.ub, L.causal,
            hp, L.n_swa, (llama_swa_type) L.swa_type, L.v_cells, L.seq_to_stream);

    std::vector<int32_t> cells_rec((size_t) (4*L.n_kv*L.n_stream), 0x5A5A5A5A);
    std::vector<int32_t> toks_rec ((size_t) (4*L.n_tps*L.n_stream), 0x5A5A5A5A);
    const bool ok = llama_kv_cells_fill_kq_vis(cells_rec.data(), toks_rec.data(), L.n_kv, L.n_stream, &L.ub, L.v_cells, L.seq_to_stream);
    if (!ok) {
        fprintf(stderr, "FAIL: %s: llama_kv_cells_fill_kq_vis rejected an expressible layout\n", d.name);
        return UINT64_MAX;
    }

    const std::vector<uint8_t> got = run_op(L, cells_rec, toks_rec, n_threads);
    const uint64_t nd = count_diff(ref, got, L.f32);

    st.layouts++;
    st.elements += (uint64_t) (L.n_kv*L.n_tps*L.n_stream);
    for (size_t i = 0; i < ref.size(); i += esz) {
        float v;
        if (L.f32) { memcpy(&v, &ref[i], 4); } else { ggml_fp16_t h; memcpy(&h, &ref[i], 2); v = ggml_fp16_to_fp32(h); }
        if (v == -INFINITY) st.dropped++; else { st.kept++; if (v != 0.0f) st.alibi_vals++; }
    }
    if (L.is_2d) {
        for (int64_t s = 0; s < L.n_stream; ++s) for (int64_t t = 0; t < L.n_tps; ++t) {
            const int32_t * tk = &toks_rec[4*(s*L.n_tps + t)];
            for (int64_t j = 0; j < L.n_kv; ++j) {
                const int32_t * c = &cells_rec[4*(s*L.n_kv + j)];
                if (((uint32_t) c[1] & (uint32_t) tk[1]) && c[0] == tk[0]) st.ties_2d++;
            }
        }
    }

    if (nd != 0) {
        fprintf(stderr, "MISMATCH: %s: %" PRIu64 " elements differ (n_kv %" PRId64 " n_tps %" PRId64 " n_stream %" PRId64
                " is_2d %d causal %d swa %d/%u alibi %d f32 %d)\n", d.name, nd, L.n_kv, L.n_tps, L.n_stream,
                L.is_2d, L.causal, L.swa_type, L.n_swa, L.alibi, L.f32);
        // first differing element
        for (size_t i = 0; i < ref.size(); i += esz) {
            if (memcmp(&ref[i], &got[i], esz) != 0) {
                const size_t e = i/esz;
                fprintf(stderr, "  first at s %zu t %zu j %zu\n", (size_t) (e/(L.n_kv*L.n_tps)), (size_t) ((e/L.n_kv)%L.n_tps), (size_t) (e%L.n_kv));
                break;
            }
        }
    }

    if (control) {
        // negative control: clear the token's bit in the record of a kept cell -> the op must now differ
        for (int tries = 0; tries < 64; ++tries) {
            const int64_t s = rnd_int(rng, 0, (int) L.n_stream - 1);
            const int64_t t = rnd_int(rng, 0, (int) L.n_tps - 1);
            const int64_t j = rnd_int(rng, 0, (int) L.n_kv - 1);
            const size_t e = (size_t) ((s*L.n_tps + t)*L.n_kv + j);
            float v;
            if (L.f32) { memcpy(&v, &ref[e*4], 4); } else { ggml_fp16_t h; memcpy(&h, &ref[e*2], 2); v = ggml_fp16_to_fp32(h); }
            if (v == -INFINITY) continue;
            std::vector<int32_t> mut = cells_rec;
            mut[4*(s*L.n_kv + j) + 1] &= ~toks_rec[4*(s*L.n_tps + t) + 1];
            const std::vector<uint8_t> got2 = run_op(L, mut, toks_rec, n_threads);
            st.ctrl_runs++;
            st.ctrl_detected += count_diff(ref, got2, L.f32) != 0;
            break;
        }
    }

    return nd;
}

} // namespace

int main(int argc, char ** argv) {
    const int      n_random  = argc > 1 ? atoi(argv[1]) : 10000;
    const uint64_t seed      = argc > 2 ? strtoull(argv[2], nullptr, 10) : 42;
    const int      n_threads = argc > 3 ? atoi(argv[3]) : 4;

    std::mt19937_64 rng(seed);
    stats_t st;
    uint64_t total_diff = 0;
    int failures = 0;

    auto run = [&](const layout_desc & d, bool control) {
        const uint64_t nd = run_layout(d, rng, st, n_threads, control);
        if (nd == UINT64_MAX) { failures++; return; }
        total_diff += nd;
        failures += nd != 0;
    };

    // named cases (spec §2.5)
    std::vector<layout_desc> named;
    { layout_desc d; d.name = "decode 5 slots x 1 token, unified";     d.unified = 1; d.n_seq = 5; d.n_useq = 5; d.n_tps = 5;  d.decode = 1; named.push_back(d); }
    { layout_desc d; d.name = "verify 5 x 5, unified";                d.unified = 1; d.n_seq = 5; d.n_useq = 5; d.n_tps = 25; d.decode = 1; named.push_back(d); }
    { layout_desc d; d.name = "decode 5 streams x 1 token";           d.unified = 0; d.n_seq = 5; d.n_stream = 5; d.n_useq = 1; d.n_tps = 1; d.decode = 1; named.push_back(d); }
    { layout_desc d; d.name = "3 streams x 64 tokens";                d.unified = 0; d.n_seq = 3; d.n_stream = 3; d.n_useq = 1; d.n_tps = 64; named.push_back(d); }
    { layout_desc d; d.name = "prefill 2048 x n_kv 4096";             d.unified = 1; d.n_seq = 1; d.n_tps = 2048; d.n_kv = 4096; d.is_2d = 0; d.swa_type = 0; d.causal = 1; d.alibi = 0; d.f32 = 0; named.push_back(d); }
    { layout_desc d; d.name = "prefill 2048 x 4096 M-RoPE image";     d.unified = 1; d.n_seq = 1; d.n_tps = 2048; d.n_kv = 4096; d.is_2d = 1; d.causal = 1; named.push_back(d); }
    { layout_desc d; d.name = "31 seqs multi-seq ubatch";             d.unified = 1; d.n_seq = 31; d.n_useq = 31; d.n_tps = 228; named.push_back(d); }
    { layout_desc d; d.name = "32 seqs (bit 31)";                     d.unified = 1; d.n_seq = 32; d.n_useq = 32; d.n_tps = 256; named.push_back(d); }
    { layout_desc d; d.name = "n_kv 65536, 6 rows, pos ~2^20";        d.unified = 1; d.n_seq = 5; d.n_tps = 6; d.n_kv = 65536; d.pos_base = (1 << 20) - 70000; named.push_back(d); }
    { layout_desc d; d.name = "n_kv 262144, 2 rows, pos ~2^20, 50% holes"; d.unified = 1; d.n_seq = 2; d.n_tps = 2; d.n_kv = 262144; d.pos_base = (1 << 20) - 270000; d.hole = 0.5; named.push_back(d); }
    { layout_desc d; d.name = "n_kv 256, no holes";                   d.n_kv = 256; d.hole = 0.0; named.push_back(d); }
    for (int swa = 1; swa <= 3; ++swa) { layout_desc d; d.name = "swa sweep"; d.swa_type = swa; d.n_tps = 64; named.push_back(d); }
    { layout_desc d; d.name = "alibi f32";                            d.alibi = 1; d.f32 = 1; named.push_back(d); }
    { layout_desc d; d.name = "alibi f16 pos ~2^20";                  d.alibi = 1; d.f32 = 0; d.pos_base = (1 << 20) - 20000; named.push_back(d); }
    { layout_desc d; d.name = "non-causal";                           d.causal = 0; named.push_back(d); }

    for (const auto & d : named) {
        run(d, true);
    }
    const uint64_t named_layouts = st.layouts;

    for (int k = 0; k < n_random; ++k) {
        layout_desc d;
        run(d, k < 500);
    }

    // precondition: a seq id >= 32 in the ubatch must be rejected (fall back to the plane)
    {
        built_layout L;
        stats_t dummy;
        layout_desc d; d.unified = 1; d.n_seq = 1; d.n_tps = 4;
        build(L, d, rng, dummy);
        for (int64_t i = 0; i < L.n_tps; ++i) L.seq_data[i*4] = 40;
        std::vector<int32_t> c((size_t) (4*L.n_kv*L.n_stream)), t((size_t) (4*L.n_tps*L.n_stream));
        st.precond_runs++;
        st.precond_rejected += !llama_kv_cells_fill_kq_vis(c.data(), t.data(), L.n_kv, L.n_stream, &L.ub, L.v_cells, L.seq_to_stream);
    }

    printf("test-kq-mask-meta: seed %" PRIu64 ", %" PRIu64 " layouts (%" PRIu64 " named + %d random), %" PRIu64 " mask elements compared\n",
            seed, st.layouts, named_layouts, n_random, st.elements);
    printf("  elements: kept %" PRIu64 " (of which ALiBi non-zero %" PRIu64 "), dropped %" PRIu64 "\n", st.kept, st.alibi_vals, st.dropped);
    printf("  layouts: n_stream=1 %" PRIu64 ", n_stream>1 %" PRIu64 ", unified %" PRIu64 ", multi-seq ubatch %" PRIu64 ", M-RoPE %" PRIu64
           ", non-causal %" PRIu64 ", ALiBi %" PRIu64 ", F32 mask %" PRIu64 "\n",
            st.by_ns1, st.by_nsN, st.unified, st.multi_seq_ub, st.is_2d, st.non_causal, st.alibi, st.f32);
    printf("  SWA: none %" PRIu64 ", standard %" PRIu64 ", chunked %" PRIu64 ", symmetric %" PRIu64 "\n", st.swa[0], st.swa[1], st.swa[2], st.swa[3]);
    printf("  cells (within n_kv): used %" PRIu64 ", empty/holes %" PRIu64 ", seqless %" PRIu64 ", multi-seq %" PRIu64 ", with seq >= 32 %" PRIu64 "\n",
            st.cells_used, st.cells_empty, st.cells_seqless, st.cells_multibit, st.cells_seq_ge32);
    printf("  M-RoPE p0 == p1 same-seq pairs: %" PRIu64 "; max n_kv %" PRId64 ", max n_tps %" PRId64 ", max pos %" PRId64 "\n",
            st.ties_2d, st.max_n_kv, st.max_n_tps, st.max_pos);
    printf("  negative control (one seq bit cleared): %" PRIu64 "/%" PRIu64 " detected\n", st.ctrl_detected, st.ctrl_runs);
    printf("  precondition (ubatch seq id >= 32 rejected): %" PRIu64 "/%" PRIu64 "\n", st.precond_rejected, st.precond_runs);
    printf("  DIFFERING ELEMENTS: %" PRIu64 " (layouts with a difference: %d)\n", total_diff, failures);

    const bool pass = total_diff == 0 && failures == 0 && st.ctrl_detected == st.ctrl_runs && st.ctrl_runs > 0 &&
                      st.precond_rejected == st.precond_runs;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
