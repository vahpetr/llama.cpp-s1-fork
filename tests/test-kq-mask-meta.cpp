// [TAG_KQ_VIS] differential test: GGML_OP_KQ_MASK over the metadata records (llama_kv_cells_fill_kq_vis)
// against the unchanged host mask fill (llama_kv_cells_fill_kq_mask = the body of
// llama_kv_cache::set_input_kq_mask / set_input_kq_mask_impl), bit for bit, on the CPU backend.
//
// usage: test-kq-mask-meta [n_random_layouts = 10000] [seed = 42] [n_threads = 4] [fa_budget = 65536]
//                          [--device NAME] [--perf] [--perf-reps N] [--perf-nkv a,b,c] [--perf-kv q8_0|f16]
//   --device NAME: also run, per layout, GGML_OP_KQ_MASK on that backend against the host fill, and
//     FLASH_ATTN_EXT over the plane vs over the metadata on that backend (both f32 outputs compared
//     bit for bit: the identity check of the device shaders). NAME as ggml_backend_dev_by_name
//     spells it (e.g. Vulkan0, CPU).
//   --perf (with --device): time FA plane (+ its mask-opt pass) vs FA vis (+ its metadata pass) at the
//     prod shapes (24 q heads / 2 kv heads / D 256): prefill n_tps 2048 and decode n_tps 1, 5 (5 slots),
//     6 (verify) at each n_kv of --perf-nkv (default 32768,107520,213248), and check their outputs
//     are identical. Only the perf part runs when n_random_layouts is 0.
//   fa_budget: layouts with n_kv*n_tps*n_stream <= fa_budget also run FLASH_ATTN_EXT over the plane and
//   over the metadata (ggml_flash_attn_ext_vis) on the CPU and compare the f32 outputs bit for bit
//
// Every layout builds synthetic llama_kv_cells (holes, seqless used cells, multi-seq cells, seq ids
// >= 32 in cells, random positions up to ~2^20, M-RoPE ext), a synthetic llama_ubatch (single- and
// multi-sequence, unified and per-stream blocks, 1-D and M-RoPE positions) and random attention
// parameters (causal / non-causal, SWA none/standard/chunked/symmetric, ALiBi, F16/F32 mask).
// A negative control flips one seq bit in the cell records and requires the comparison to fail.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
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
    uint64_t fa_runs = 0, fa_diff = 0, fa_elems = 0, fa_model_shape = 0, fa_q8 = 0, fa_alibi = 0, fa_ctrl_runs = 0, fa_ctrl_detected = 0;
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

    ggml_tensor * tc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, L.n_kv, 4, L.n_stream);
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

struct fa_shape {
    int64_t   D, n_head, n_head_kv;
    ggml_type kv_type;
    float     max_bias;
    float     softcap;
};

// FA over the materialized f16 plane vs FA over the metadata (ggml_flash_attn_ext_vis), both on the
// CPU backend, same Q/K/V. Returns {plane output bytes, vis output bytes}.
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> run_fa(const built_layout & L, const std::vector<uint8_t> & plane_f16,
        const std::vector<int32_t> & cells_rec, const std::vector<int32_t> & toks_rec, const fa_shape & fs,
        std::mt19937_64 & rng, int n_threads) {
    const int64_t n_q  = fs.D*L.n_tps*fs.n_head*L.n_stream;
    const int64_t n_kv = fs.D*L.n_kv*fs.n_head_kv*L.n_stream;

    const size_t bytes = (size_t) n_q*4*3 + (size_t) n_kv*4*2 + plane_f16.size() + cells_rec.size()*4 + toks_rec.size()*4;
    ggml_init_params ip = { bytes + 64*ggml_tensor_overhead() + ggml_graph_overhead() + (4 << 20), nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    GGML_ASSERT(ctx);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, fs.D, L.n_tps, fs.n_head,    L.n_stream);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, fs.kv_type,    fs.D, L.n_kv,  fs.n_head_kv, L.n_stream);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, fs.kv_type,    fs.D, L.n_kv,  fs.n_head_kv, L.n_stream);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, L.n_kv, L.n_tps, 1, L.n_stream);
    ggml_tensor * tc = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, L.n_kv, 4, L.n_stream);
    ggml_tensor * tt = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, L.n_tps, L.n_stream);

    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int64_t i = 0; i < n_q; ++i) ((float *) q->data)[i] = nd(rng);
    std::vector<float> tmp((size_t) n_kv);
    for (ggml_tensor * t : { k, v }) {
        for (auto & x : tmp) x = nd(rng);
        ggml_quantize_chunk(fs.kv_type, tmp.data(), t->data, 0, n_kv/fs.D, fs.D, nullptr);
    }
    memcpy(m->data,  plane_f16.data(), ggml_nbytes(m));
    memcpy(tc->data, cells_rec.data(), ggml_nbytes(tc));
    memcpy(tt->data, toks_rec.data(),  ggml_nbytes(tt));

    const float scale = 1.0f/sqrtf((float) fs.D);

    ggml_tensor * o_plane = ggml_flash_attn_ext(ctx, q, k, v, m, scale, fs.max_bias, fs.softcap);
    ggml_flash_attn_ext_set_prec(o_plane, GGML_PREC_F32);
    ggml_tensor * o_vis = ggml_flash_attn_ext_vis(ctx, q, k, v, tc, tt, scale, fs.max_bias, fs.softcap,
            L.causal, L.alibi, L.swa_type, (int32_t) L.n_swa);
    ggml_flash_attn_ext_set_prec(o_vis, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o_plane);
    ggml_build_forward_expand(gf, o_vis);
    GGML_ASSERT(ggml_graph_compute_with_ctx(ctx, gf, n_threads) == GGML_STATUS_SUCCESS);

    std::pair<std::vector<uint8_t>, std::vector<uint8_t>> res;
    res.first .resize(ggml_nbytes(o_plane));
    res.second.resize(ggml_nbytes(o_vis));
    memcpy(res.first .data(), o_plane->data, res.first .size());
    memcpy(res.second.data(), o_vis->data,   res.second.size());
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
int64_t fa_budget = 1 << 16; // max n_kv*n_tps*n_stream of a layout that also runs FA (0 = off)
bool nd_fa_fail = false;

// ---------------------------------------------------------------------------------------------
// [TAG_KQ_VIS] device mode: the same checks on a ggml backend (the identity binary)

ggml_backend_t g_dev = nullptr;

struct dev_stats {
    uint64_t op_runs = 0, op_elems = 0, op_diff = 0, op_diff_zero_sign = 0, op_diff_non_alibi_layouts = 0;
    uint64_t fa_runs = 0, fa_elems = 0, fa_diff = 0, fa_unsupported = 0;
};
dev_stats g_ds;

struct dev_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ~dev_graph() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
    void init() {
        ggml_init_params ip = { 64*ggml_tensor_overhead() + 4*ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        GGML_ASSERT(ctx);
    }
    void alloc(ggml_backend_t be) {
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        GGML_ASSERT(buf && "device allocation failed");
    }
};

std::vector<uint8_t> tensor_bytes(const ggml_tensor * t) {
    std::vector<uint8_t> r(ggml_nbytes(t));
    ggml_backend_tensor_get(t, r.data(), 0, r.size());
    return r;
}

void compute(ggml_backend_t be, ggml_cgraph * gf) {
    GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(be);
}

// GGML_OP_KQ_MASK on the device
std::vector<uint8_t> dev_run_op(const built_layout & L, const std::vector<int32_t> & cells_rec, const std::vector<int32_t> & toks_rec) {
    dev_graph g; g.init();
    ggml_tensor * tc = ggml_new_tensor_3d(g.ctx, GGML_TYPE_I32, L.n_kv, 4, L.n_stream);
    ggml_tensor * tt = ggml_new_tensor_3d(g.ctx, GGML_TYPE_I32, 4, L.n_tps, L.n_stream);
    ggml_tensor * out = ggml_kq_mask(g.ctx, tc, tt, L.f32 ? GGML_TYPE_F32 : GGML_TYPE_F16, L.causal, L.alibi, L.swa_type, (int32_t) L.n_swa);
    if (!ggml_backend_supports_op(g_dev, out)) {
        return {};
    }
    g.alloc(g_dev);
    ggml_backend_tensor_set(tc, cells_rec.data(), 0, ggml_nbytes(tc));
    ggml_backend_tensor_set(tt, toks_rec.data(),  0, ggml_nbytes(tt));
    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, out);
    compute(g_dev, gf);
    return tensor_bytes(out);
}

// Q/K/V/plane/metadata of one FA problem on the device, plus the two graphs
struct dev_fa {
    dev_graph g;
    ggml_tensor * q = nullptr, * k = nullptr, * v = nullptr, * m = nullptr, * tc = nullptr, * tt = nullptr;
    ggml_tensor * o_plane = nullptr, * o_vis = nullptr;
    ggml_cgraph * gf_plane = nullptr, * gf_vis = nullptr;
    bool supported = true;

    // kv_rows_fill: fill K/V with random rows in chunks (no full f32 copy of a 213k-cell cache)
    void build(int64_t n_kv, int64_t n_tps, int64_t n_stream, const fa_shape & fs, bool causal, bool alibi, int swa_type, int n_swa,
               const std::vector<uint8_t> & plane_f16, const std::vector<int32_t> & cells_rec, const std::vector<int32_t> & toks_rec,
               std::mt19937_64 & rng) {
        g.init();
        q  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, fs.D, n_tps, fs.n_head,    n_stream);
        k  = ggml_new_tensor_4d(g.ctx, fs.kv_type,    fs.D, n_kv,  fs.n_head_kv, n_stream);
        v  = ggml_new_tensor_4d(g.ctx, fs.kv_type,    fs.D, n_kv,  fs.n_head_kv, n_stream);
        m  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, n_kv, n_tps, 1, n_stream);
        tc = ggml_new_tensor_3d(g.ctx, GGML_TYPE_I32, n_kv, 4, n_stream);
        tt = ggml_new_tensor_3d(g.ctx, GGML_TYPE_I32, 4, n_tps, n_stream);

        const float scale = 1.0f/sqrtf((float) fs.D);
        o_plane = ggml_flash_attn_ext(g.ctx, q, k, v, m, scale, fs.max_bias, fs.softcap);
        ggml_flash_attn_ext_set_prec(o_plane, GGML_PREC_F32);
        o_vis = ggml_flash_attn_ext_vis(g.ctx, q, k, v, tc, tt, scale, fs.max_bias, fs.softcap, causal, alibi, swa_type, n_swa);
        ggml_flash_attn_ext_set_prec(o_vis, GGML_PREC_F32);

        supported = ggml_backend_supports_op(g_dev, o_plane) && ggml_backend_supports_op(g_dev, o_vis);
        if (!supported) {
            return;
        }
        g.alloc(g_dev);

        std::normal_distribution<float> nd(0.0f, 1.0f);
        {
            std::vector<float> qh((size_t) ggml_nelements(q));
            for (auto & x : qh) x = nd(rng);
            ggml_backend_tensor_set(q, qh.data(), 0, ggml_nbytes(q));
        }
        const int64_t rows      = n_kv*fs.n_head_kv*n_stream;
        const int64_t chunk     = 4096;
        const size_t  row_bytes = ggml_row_size(fs.kv_type, fs.D);
        std::vector<float>   tmp((size_t) (chunk*fs.D));
        std::vector<uint8_t> qbuf((size_t) chunk*row_bytes);
        for (ggml_tensor * t : { k, v }) {
            for (int64_t r0 = 0; r0 < rows; r0 += chunk) {
                const int64_t nr = std::min(chunk, rows - r0);
                for (int64_t i = 0; i < nr*fs.D; ++i) tmp[i] = nd(rng);
                ggml_quantize_chunk(fs.kv_type, tmp.data(), qbuf.data(), 0, nr, fs.D, nullptr);
                ggml_backend_tensor_set(t, qbuf.data(), (size_t) r0*row_bytes, (size_t) nr*row_bytes);
            }
        }
        ggml_backend_tensor_set(m,  plane_f16.data(), 0, ggml_nbytes(m));
        ggml_backend_tensor_set(tc, cells_rec.data(), 0, ggml_nbytes(tc));
        ggml_backend_tensor_set(tt, toks_rec.data(),  0, ggml_nbytes(tt));

        gf_plane = ggml_new_graph(g.ctx);
        ggml_build_forward_expand(gf_plane, o_plane);
        gf_vis = ggml_new_graph(g.ctx);
        ggml_build_forward_expand(gf_vis, o_vis);
    }
};

void dev_check_layout(const built_layout & L, const std::vector<uint8_t> & ref, const std::vector<int32_t> & cells_rec,
                      const std::vector<int32_t> & toks_rec, const llama_hparams & hp, std::mt19937_64 & rng, const char * name) {
    // the op
    const std::vector<uint8_t> got = dev_run_op(L, cells_rec, toks_rec);
    if (!got.empty()) {
        const uint64_t nd = count_diff(ref, got, L.f32);
        g_ds.op_runs++;
        g_ds.op_elems += ref.size()/(L.f32 ? 4 : 2);
        g_ds.op_diff  += nd;
        if (nd) {
            // classify: a sign-of-zero difference (+0 in the reference, -0 on the device) vs anything else
            uint64_t n_zero_sign = 0;
            const size_t esz = L.f32 ? 4 : 2;
            size_t first = SIZE_MAX;
            for (size_t i = 0; i < ref.size(); i += esz) {
                if (memcmp(&ref[i], &got[i], esz) == 0) continue;
                if (first == SIZE_MAX) first = i/esz;
                float a, b;
                if (L.f32) { memcpy(&a, &ref[i], 4); memcpy(&b, &got[i], 4); }
                else { ggml_fp16_t ha, hb; memcpy(&ha, &ref[i], 2); memcpy(&hb, &got[i], 2); a = ggml_fp16_to_fp32(ha); b = ggml_fp16_to_fp32(hb); }
                n_zero_sign += (a == 0.0f && b == 0.0f);
            }
            g_ds.op_diff_zero_sign += n_zero_sign;
            fprintf(stderr, "DEVICE KQ_MASK MISMATCH: %s: %" PRIu64 " elements (%" PRIu64 " only the sign of a zero) "
                    "(n_kv %" PRId64 " n_tps %" PRId64 " n_stream %" PRId64 " causal %d swa %d/%u alibi %d 2d %d f32 %d; first at %zu)\n",
                    name, nd, n_zero_sign, L.n_kv, L.n_tps, L.n_stream, L.causal, L.swa_type, L.n_swa, L.alibi, L.is_2d, L.f32, first);
            if (!L.alibi) {
                g_ds.op_diff_non_alibi_layouts++;
            }
        }
    }

    // FA plane vs FA vis on the device
    if (fa_budget > 0 && L.n_kv*L.n_tps*L.n_stream <= 4*fa_budget) {
        std::vector<uint8_t> plane((size_t) (L.n_kv*L.n_tps*L.n_stream)*2);
        llama_kv_cells_fill_kq_mask(plane.data(), GGML_TYPE_F16, L.n_kv, L.n_stream, &L.ub, L.causal,
                hp, L.n_swa, (llama_swa_type) L.swa_type, L.v_cells, L.seq_to_stream);
        fa_shape fs;
        fs.D = 256; fs.n_head = 24; fs.n_head_kv = 2;
        fs.kv_type  = rnd01(rng) < 0.5 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        fs.max_bias = L.alibi ? 8.0f : 0.0f;
        fs.softcap  = 0.0f;
        dev_fa f;
        f.build(L.n_kv, L.n_tps, L.n_stream, fs, L.causal, L.alibi, L.swa_type, (int) L.n_swa, plane, cells_rec, toks_rec, rng);
        if (!f.supported) {
            g_ds.fa_unsupported++;
            return;
        }
        compute(g_dev, f.gf_plane);
        compute(g_dev, f.gf_vis);
        const uint64_t nd = count_diff(tensor_bytes(f.o_plane), tensor_bytes(f.o_vis), true);
        g_ds.fa_runs++;
        g_ds.fa_elems += ggml_nelements(f.o_plane);
        g_ds.fa_diff  += nd;
        if (nd) {
            fprintf(stderr, "DEVICE FA MISMATCH: %s: %" PRIu64 " outputs (n_kv %" PRId64 " n_tps %" PRId64 " n_stream %" PRId64 " kv %s causal %d swa %d alibi %d 2d %d)\n",
                    name, nd, L.n_kv, L.n_tps, L.n_stream, ggml_type_name(fs.kv_type), L.causal, L.swa_type, L.alibi, L.is_2d);
        }
    }
}

// perf: a prod-like layout - n_seq sequences sharing a unified pool of n_kv cells (round robin), the
// ubatch = the next n_tps tokens (prefill: one sequence; decode: one token per sequence; verify: 6
// tokens of one sequence)
void build_perf_layout(built_layout & L, int64_t n_kv, int64_t n_tps, int n_seq, bool one_seq_ubatch) {
    L.n_kv = n_kv; L.n_tps = n_tps; L.n_stream = 1;
    L.is_2d = false; L.causal = true; L.alibi = false; L.f32 = false; L.swa_type = 0; L.n_swa = 0;
    L.seq_to_stream.assign(LLAMA_MAX_SEQ, 0);
    L.v_cells.clear();
    L.v_cells.resize(1);
    auto & cells = L.v_cells[0];
    cells.resize((uint32_t) n_kv);
    std::vector<llama_pos> next(n_seq, 0);
    // the cache holds n_kv - n_tps cells, then the ubatch's own cells
    for (int64_t j = 0; j < n_kv - n_tps; ++j) {
        const int sq = (int) (j % n_seq);
        cells.pos_set((uint32_t) j, next[sq]++);
        cells.seq_add((uint32_t) j, sq);
    }
    L.pos.assign((size_t) n_tps, 0);
    L.n_seq_id.assign((size_t) n_tps, 1);
    L.seq_data.assign((size_t) n_tps*4, 0);
    L.seq_ptr.assign((size_t) n_tps, nullptr);
    L.output.assign((size_t) n_tps, 1);
    L.seq_idx.assign(LLAMA_MAX_SEQ, -1);
    L.seq_unq.clear();
    for (int64_t i = 0; i < n_tps; ++i) {
        const int sq = one_seq_ubatch ? 0 : (int) (i % n_seq);
        const llama_pos p = next[sq]++;
        const uint32_t j = (uint32_t) (n_kv - n_tps + i);
        cells.pos_set(j, p);
        cells.seq_add(j, sq);
        L.pos[i] = p;
        L.seq_data[i*4] = sq;
        L.seq_ptr[i] = &L.seq_data[i*4];
        if (L.seq_idx[sq] < 0) { L.seq_idx[sq] = (int32_t) L.seq_unq.size(); L.seq_unq.push_back(sq); }
    }
    L.ub = llama_ubatch{};
    L.ub.n_tokens = (uint32_t) n_tps; L.ub.n_seq_tokens = (uint32_t) n_tps; L.ub.n_seqs = 1;
    L.ub.n_seqs_unq = (uint32_t) L.seq_unq.size(); L.ub.n_pos = 1;
    L.ub.pos = L.pos.data(); L.ub.n_seq_id = L.n_seq_id.data(); L.ub.seq_id = L.seq_ptr.data();
    L.ub.seq_id_unq = L.seq_unq.data(); L.ub.seq_idx = L.seq_idx.data(); L.ub.output = L.output.data();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size()/2];
}

int run_perf(const std::vector<int64_t> & nkvs, ggml_type kv_type, int reps) {
    struct shape { const char * name; int64_t n_tps; int n_seq; bool one_seq; };
    const shape shapes[] = {
        { "prefill 2048",       2048, 5, true  },
        { "decode 1 (1 slot)",  1,    1, true  },
        { "decode 5 (5 slots)", 5,    5, false },
        { "verify 6 (1 slot)",  6,    5, true  },
    };
    int bad = 0;
    printf("perf: FA 24 q heads / 2 kv heads / D 256, K/V %s, median of %d interleaved runs (plane = FA + mask-opt over the f16 plane; vis = FA + metadata pass)\n",
           ggml_type_name(kv_type), reps);
    printf("  %-20s %8s %12s %12s %8s %10s\n", "shape", "n_kv", "plane ms", "vis ms", "delta", "diff bits");
    std::mt19937_64 rng(7);
    for (int64_t n_kv : nkvs) {
        for (const auto & sh : shapes) {
            built_layout L;
            build_perf_layout(L, n_kv, sh.n_tps, sh.n_seq, sh.one_seq);
            llama_hparams hp{};
            std::vector<uint8_t> plane((size_t) (L.n_kv*L.n_tps)*2);
            llama_kv_cells_fill_kq_mask(plane.data(), GGML_TYPE_F16, L.n_kv, 1, &L.ub, true, hp, 0, LLAMA_SWA_TYPE_NONE, L.v_cells, L.seq_to_stream);
            std::vector<int32_t> cells_rec((size_t) (4*L.n_kv)), toks_rec((size_t) (4*L.n_tps));
            GGML_ASSERT(llama_kv_cells_fill_kq_vis(cells_rec.data(), toks_rec.data(), L.n_kv, 1, &L.ub, L.v_cells, L.seq_to_stream));

            fa_shape fs; fs.D = 256; fs.n_head = 24; fs.n_head_kv = 2; fs.kv_type = kv_type; fs.max_bias = 0.0f; fs.softcap = 0.0f;
            dev_fa f;
            f.build(L.n_kv, L.n_tps, 1, fs, true, false, 0, 0, plane, cells_rec, toks_rec, rng);
            if (!f.supported) {
                printf("  %-20s %8" PRId64 "  not supported\n", sh.name, n_kv);
                continue;
            }
            // warm up (pipeline compiles, preallocations)
            for (int w = 0; w < 2; ++w) { compute(g_dev, f.gf_plane); compute(g_dev, f.gf_vis); }
            const uint64_t nd = count_diff(tensor_bytes(f.o_plane), tensor_bytes(f.o_vis), true);
            std::vector<double> tp, tv;
            for (int r = 0; r < reps; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                compute(g_dev, f.gf_plane);
                auto t1 = std::chrono::steady_clock::now();
                compute(g_dev, f.gf_vis);
                auto t2 = std::chrono::steady_clock::now();
                tp.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                tv.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
            const double mp = median(tp), mv = median(tv);
            printf("  %-20s %8" PRId64 " %12.3f %12.3f %+7.2f%% %10" PRIu64 "\n", sh.name, n_kv, mp, mv, 100.0*(mv - mp)/mp, nd);
            fflush(stdout);
            bad += nd != 0;
        }
    }
    printf("perf identity: %s\n", bad ? "FAIL (differing outputs above)" : "PASS (0 differing bits in every shape)");
    return bad;
}

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

    if (g_dev) {
        dev_check_layout(L, ref, cells_rec, toks_rec, hp, rng, d.name);
    }

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
                const int32_t * c = &cells_rec[4*s*L.n_kv + j]; // planes: c[0] pos, c[n_kv] seqmask
                if (((uint32_t) c[L.n_kv] & (uint32_t) tk[1]) && c[0] == tk[0]) st.ties_2d++;
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

    // flash attention: plane vs metadata, bit for bit (a budgeted subset of the layouts)
    if (fa_budget > 0 && L.n_kv*L.n_tps*L.n_stream <= fa_budget) {
        // the f16 plane FA reads (the layout itself may have asked for an f32 mask)
        std::vector<uint8_t> plane((size_t) (L.n_kv*L.n_tps*L.n_stream)*2);
        llama_kv_cells_fill_kq_mask(plane.data(), GGML_TYPE_F16, L.n_kv, L.n_stream, &L.ub, L.causal,
                hp, L.n_swa, (llama_swa_type) L.swa_type, L.v_cells, L.seq_to_stream);

        fa_shape fs;
        const bool model = L.n_kv*L.n_tps*L.n_stream <= fa_budget/8 || rnd01(rng) < 0.3;
        if (model) { fs.D = 256; fs.n_head = 24; fs.n_head_kv = 2; } // qwen4exp target and MTP draft
        else       { fs.D = 64;  fs.n_head = 4;  fs.n_head_kv = 2; }
        fs.kv_type  = rnd01(rng) < 0.5 ? GGML_TYPE_F16 : GGML_TYPE_Q8_0; // draft f16, target q8_0
        fs.max_bias = L.alibi ? 8.0f : 0.0f;
        fs.softcap  = rnd01(rng) < 0.1 ? 30.0f : 0.0f;

        auto outs = run_fa(L, plane, cells_rec, toks_rec, fs, rng, n_threads);
        const uint64_t fd = count_diff(outs.first, outs.second, true);
        st.fa_runs++;
        st.fa_elems += outs.first.size()/4;
        st.fa_diff  += fd;
        st.fa_model_shape += model;
        st.fa_q8    += fs.kv_type == GGML_TYPE_Q8_0;
        st.fa_alibi += L.alibi;
        if (fd != 0) {
            fprintf(stderr, "FA MISMATCH: %s: %" PRIu64 " outputs differ (n_kv %" PRId64 " n_tps %" PRId64 " n_stream %" PRId64 " D %" PRId64 ")\n",
                    d.name, fd, L.n_kv, L.n_tps, L.n_stream, fs.D);
            nd_fa_fail = true;
        }

        // negative control for the FA path: drop one kept cell for one token
        if (control && st.fa_ctrl_runs < 64) {
            for (int tries = 0; tries < 64; ++tries) {
                const int64_t s = rnd_int(rng, 0, (int) L.n_stream - 1);
                const int64_t t = rnd_int(rng, 0, (int) L.n_tps - 1);
                const int64_t j = rnd_int(rng, 0, (int) L.n_kv - 1);
                ggml_fp16_t h; memcpy(&h, &plane[(size_t) ((s*L.n_tps + t)*L.n_kv + j)*2], 2);
                if (ggml_fp16_to_fp32(h) == -INFINITY) continue;
                std::vector<int32_t> mut = cells_rec;
                mut[4*s*L.n_kv + L.n_kv + j] &= ~toks_rec[4*(s*L.n_tps + t) + 1];
                std::mt19937_64 rng2(1234); // same Q/K/V as a fresh run with this seed
                std::mt19937_64 rng3(1234);
                auto a = run_fa(L, plane, cells_rec, toks_rec, fs, rng2, n_threads);
                auto b = run_fa(L, plane, mut,       toks_rec, fs, rng3, n_threads);
                st.fa_ctrl_runs++;
                st.fa_ctrl_detected += count_diff(a.second, b.second, true) != 0;
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
            mut[4*s*L.n_kv + L.n_kv + j] &= ~toks_rec[4*(s*L.n_tps + t) + 1];
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
    std::vector<const char *> pos_args;
    const char * dev_name = nullptr;
    bool perf = false;
    int perf_reps = 20;
    std::vector<int64_t> perf_nkv = { 32768, 107520, 213248 };
    ggml_type perf_kv = GGML_TYPE_Q8_0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) { dev_name = argv[++i]; }
        else if (a == "--perf") { perf = true; }
        else if (a == "--perf-reps" && i + 1 < argc) { perf_reps = atoi(argv[++i]); }
        else if (a == "--perf-kv" && i + 1 < argc) { perf_kv = std::string(argv[++i]) == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q8_0; }
        else if (a == "--perf-nkv" && i + 1 < argc) {
            perf_nkv.clear();
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) perf_nkv.push_back(atoll(tok.c_str()));
        }
        else { pos_args.push_back(argv[i]); }
    }
    const int      n_random  = pos_args.size() > 0 ? atoi(pos_args[0]) : 10000;
    const uint64_t seed      = pos_args.size() > 1 ? strtoull(pos_args[1], nullptr, 10) : 42;
    const int      n_threads = pos_args.size() > 2 ? atoi(pos_args[2]) : 4;
    fa_budget                = pos_args.size() > 3 ? atoll(pos_args[3]) : fa_budget;

    if (dev_name) {
        ggml_backend_load_all();
        g_dev = ggml_backend_init_by_name(dev_name, nullptr);
        if (!g_dev) {
            fprintf(stderr, "no backend device named %s\n", dev_name);
            return 2;
        }
        printf("device: %s\n", ggml_backend_name(g_dev));
    }
    int perf_bad = 0;
    if (perf) {
        GGML_ASSERT(g_dev && "--perf needs --device");
        perf_bad = run_perf(perf_nkv, perf_kv, perf_reps);
        if (n_random == 0) {
            ggml_backend_free(g_dev);
            return perf_bad ? 1 : 0;
        }
    }

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
    printf("  FLASH_ATTN_EXT plane vs vis (CPU): %" PRIu64 " layouts (%" PRIu64 " at D 256 / 24 heads / 2 kv heads, %" PRIu64 " q8_0 K/V, %" PRIu64 " ALiBi), %" PRIu64 " outputs, DIFFERING OUTPUTS: %" PRIu64 "\n",
            st.fa_runs, st.fa_model_shape, st.fa_q8, st.fa_alibi, st.fa_elems, st.fa_diff);
    printf("  FA negative control (one seq bit cleared): %" PRIu64 "/%" PRIu64 " detected\n", st.fa_ctrl_detected, st.fa_ctrl_runs);

    if (g_dev) {
        printf("  DEVICE %s: KQ_MASK %" PRIu64 " layouts, %" PRIu64 " elements, DIFFERING: %" PRIu64 " (sign of zero only: %" PRIu64 "; differing non-ALiBi layouts: %" PRIu64 ")\n",
                ggml_backend_name(g_dev), g_ds.op_runs, g_ds.op_elems, g_ds.op_diff, g_ds.op_diff_zero_sign, g_ds.op_diff_non_alibi_layouts);
        printf("  DEVICE %s: FLASH_ATTN_EXT plane vs vis %" PRIu64 " layouts (D 256, 24/2 heads), %" PRIu64 " outputs, DIFFERING BITS: %" PRIu64 " (unsupported: %" PRIu64 ")\n",
                ggml_backend_name(g_dev), g_ds.fa_runs, g_ds.fa_elems, g_ds.fa_diff, g_ds.fa_unsupported);
    }
    const bool dev_pass = !g_dev || (g_ds.op_diff == 0 && g_ds.fa_diff == 0 && g_ds.op_runs > 0 && g_ds.fa_runs > 0 && perf_bad == 0);
    const bool pass = dev_pass && total_diff == 0 && failures == 0 && st.ctrl_detected == st.ctrl_runs && st.ctrl_runs > 0 &&
                      st.precond_rejected == st.precond_runs && st.fa_diff == 0 && !nd_fa_fail &&
                      (fa_budget == 0 || (st.fa_runs > 0 && st.fa_ctrl_detected == st.fa_ctrl_runs));
    printf("%s\n", pass ? "PASS" : "FAIL");
    if (g_dev) {
        ggml_backend_free(g_dev);
    }
    return pass ? 0 : 1;
}
