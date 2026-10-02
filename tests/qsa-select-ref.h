// [TAG_QSA_WHOLE_BLOCKS] census-shaped inputs for the qwen4exp QSA selection and a port of the reference selection
// (HF transformers Qwen4ExpTextQSAIndexer.forward, models/qwen4_exp/modeling_qwen4_exp.py @ 0a896aa4, :725-760).
// Shared by tests/test-qsa-select.cpp (the CPU op against the reference) and tests/test-backend-ops.cpp (the
// backends against the CPU op on the same inputs). Header-only, no ggml dependency.
//
// One synthetic census per stream, built the way llama_memory_hybrid_idx_context::qsa_census +
// qsa_fill_bias build theirs for ONE sequence:
//   - positions 0..P-1 sit in randomly permuted cells, P = n_kv - e_s with e_s = (3 s) mod 7 empty cells
//     (stream 0 is a FULL cache: every cell visible to its last query, no masked cell to pad with);
//   - complete groups of r positions are the blocks b = p / r (n_bid = P / r of them); the newest incomplete
//     group and the empty cells go to the dead id n_bid (or n_blocks - 1 when n_bid == n_blocks);
//   - per query row q: mask 0 for the cells at positions <= q, -inf else; the per-block bias is +1e9 for a
//     block whose first position is >= the tail start (q+1)/r*r, +1e9 for the spare dead id, -inf for every
//     other id >= n_bid, 0 else; score = raw + bias (f32, as the graph's ggml_add).
//   - raw block scores: `distinct` = a per-row permutation of 0, 0.25, 0.5, ... (no ties), else the ReLU
//     shape - 35 % exactly 0, the rest on a coarse 0.25 grid (ties everywhere, also at the cut).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

struct qsa_census_case {
    int64_t n_kv     = 0;
    int64_t n_blocks = 0;
    int64_t n_tps    = 0;
    int64_t n_stream = 0;
    int64_t r        = 4;

    std::vector<float>   raw;      // [n_blocks, n_tps, n_stream] unbiased block scores (the reference's scores)
    std::vector<float>   score;    // [n_blocks, n_tps, n_stream] raw + the census bias (the op's input)
    std::vector<int32_t> cell_blk; // [n_kv, n_stream]
    std::vector<float>   mask;     // [n_kv, n_tps, n_stream] 0 / -inf
    std::vector<int32_t> cell_pos; // [n_kv, n_stream] the cell's position, -1 = empty
    std::vector<int32_t> pos_cell; // [n_kv, n_stream] the cell holding position p, -1 = none
    std::vector<int32_t> qpos;     // [n_tps, n_stream] the query position of each row
};

static inline qsa_census_case qsa_census_make(int64_t n_kv, int64_t n_tps, int64_t n_stream, uint32_t seed,
                                              bool distinct, int64_t r = 4) {
    qsa_census_case c;
    c.n_kv     = n_kv;
    c.r        = r;
    c.n_blocks = (n_kv + r - 1)/r;
    c.n_tps    = n_tps;
    c.n_stream = n_stream;

    const int64_t n_blocks = c.n_blocks;

    c.raw     .assign(n_blocks*n_tps*n_stream, 0.0f);
    c.score   .assign(n_blocks*n_tps*n_stream, 0.0f);
    c.cell_blk.assign(n_kv*n_stream, 0);
    c.mask    .assign(n_kv*n_tps*n_stream, -INFINITY);
    c.cell_pos.assign(n_kv*n_stream, -1);
    c.pos_cell.assign(n_kv*n_stream, -1);
    c.qpos    .assign(n_tps*n_stream, 0);

    std::mt19937 rng(seed);

    for (int64_t s = 0; s < n_stream; ++s) {
        const int64_t e = (3*s) % 7;
        const int64_t P = n_kv - e;

        std::vector<int32_t> perm(n_kv);
        for (int64_t j = 0; j < n_kv; ++j) {
            perm[j] = (int32_t) j;
        }
        std::shuffle(perm.begin(), perm.end(), rng);

        for (int64_t p = 0; p < P; ++p) {
            c.cell_pos[s*n_kv + perm[p]] = (int32_t) p;
            c.pos_cell[s*n_kv + p]       = perm[p];
        }

        const int64_t n_bid    = P/r;
        const bool    spare    = n_bid < n_blocks;
        const int64_t dead_bid = spare ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t p = c.cell_pos[s*n_kv + j];
            c.cell_blk[s*n_kv + j] = (int32_t) ((p >= 0 && p/r < n_bid) ? p/r : dead_bid);
        }

        for (int64_t t = 0; t < n_tps; ++t) {
            // row 0: the newest token; row 1: a short window (fewer complete blocks than any budget); then a mix
            // of the last 64 positions (all four tail residues) and anywhere
            int64_t q;
            if (t == 0) {
                q = P - 1;
            } else if (t == 1) {
                q = std::min<int64_t>(P - 1, 37);
            } else if (t % 2 == 0) {
                q = P - 1 - (int64_t) (rng() % (uint32_t) std::min<int64_t>(P, 64));
            } else {
                q = (int64_t) (rng() % (uint32_t) P);
            }
            c.qpos[s*n_tps + t] = (int32_t) q;

            const int64_t tail_start = (q + 1)/r*r;

            float * mk = c.mask.data() + (s*n_tps + t)*n_kv;
            for (int64_t j = 0; j < n_kv; ++j) {
                const int32_t p = c.cell_pos[s*n_kv + j];
                mk[j] = (p >= 0 && p <= q) ? 0.0f : -INFINITY;
            }

            float * rw = c.raw  .data() + (s*n_tps + t)*n_blocks;
            float * sc = c.score.data() + (s*n_tps + t)*n_blocks;

            if (distinct) {
                std::vector<int32_t> v(n_blocks);
                for (int64_t b = 0; b < n_blocks; ++b) {
                    v[b] = (int32_t) b;
                }
                std::shuffle(v.begin(), v.end(), rng);
                for (int64_t b = 0; b < n_blocks; ++b) {
                    rw[b] = 0.25f*(float) v[b];
                }
            } else {
                for (int64_t b = 0; b < n_blocks; ++b) {
                    rw[b] = (rng() % 100) < 35 ? 0.0f : 0.25f*(float) (rng() % 64);
                }
            }

            for (int64_t b = 0; b < n_blocks; ++b) {
                float bias;
                if (b < n_bid) {
                    bias = b*r >= tail_start ? 1e9f : 0.0f;
                } else {
                    bias = (spare && b == n_bid) ? 1e9f : -INFINITY;
                }
                sc[b] = rw[b] + bias;
            }
        }
    }

    return c;
}

// The reference selection for row (t, s): HF's per-query loop over the visible tokens in kv (= position) order.
// n_complete = |visible| / r blocks of r consecutive visible tokens, scored by the block's raw score, the
// top min(block_topk, n_complete) of them (equal scores to the lower block index: torch.topk leaves the order
// unspecified), expanded to their tokens, plus the |visible| mod r tail tokens. Returned as CELL ids, sorted.
static inline std::vector<int32_t> qsa_ref_select(const qsa_census_case & c, int64_t t, int64_t s, int64_t block_topk) {
    const int64_t r = c.r;
    const int64_t q = c.qpos[s*c.n_tps + t];

    // visible = positions 0..q, all present (q < P)
    const int64_t n_visible  = q + 1;
    const int64_t n_complete = n_visible/r;

    const float * rw = c.raw.data() + (s*c.n_tps + t)*c.n_blocks;

    std::vector<int32_t> blocks(n_complete);
    for (int64_t k = 0; k < n_complete; ++k) {
        blocks[k] = (int32_t) k;
    }
    const int64_t n_take = std::min(block_topk, n_complete);
    std::partial_sort(blocks.begin(), blocks.begin() + n_take, blocks.end(), [rw](int32_t a, int32_t b) {
        return rw[a] > rw[b] || (rw[a] == rw[b] && a < b);
    });

    std::vector<int32_t> out;
    out.reserve(n_take*r + r);
    for (int64_t i = 0; i < n_take; ++i) {
        for (int64_t m = 0; m < r; ++m) {
            out.push_back(c.pos_cell[s*c.n_kv + blocks[i]*r + m]);
        }
    }
    for (int64_t p = n_complete*r; p <= q; ++p) {
        out.push_back(c.pos_cell[s*c.n_kv + p]);
    }
    std::sort(out.begin(), out.end());

    return out;
}
