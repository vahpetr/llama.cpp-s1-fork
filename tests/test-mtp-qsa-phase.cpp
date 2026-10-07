// [TAG_MTP_QSA_PHASE] the sparse MTP draft's QSA block phase against the reference (M1 of
// docs/verification/2026-10-07-mtp-indexer-parity.md in the piagent repo), on the CPU.
//
// Entries e_k = (h_{k-1}, x_k), k >= 1. The fork's draft memory stores e_k at position k and a carrier row c at position 0;
// SGLang's draft-extend keeps the target positions with the tokens shifted left, so it stores e_k at k - 1 and has no c.
// The reference selection (HF Qwen4ExpTextQSAIndexer.forward :725-760, the semantics SGLang's qwen_sparse_attn_backend
// runs per draft query) groups each query's visible list - in kv order - into whole blocks of r from its start, keeps
// the top min(512, n_complete) blocks by score and adds the n_visible mod r tail entries.
//
// Per case: the fork's cells (positions 0..P in permuted cells, empty padding), the REAL census (llama_qsa_census_full,
// incl. the [TAG_QSA_WHOLE_BLOCKS] fragment routing), the REAL compact block table (llama_qsa_fill_blk_tab) and query tail
// (llama_qsa_tail_start), the op's bias formula, the REAL whole-block selection op (ggml_qsa_top_k_blocks, n_sel 512) -
// against a C++ port of the reference loop over the same synthetic per-entry scores (a block's score = the f32 sum of
// its members' values in entry order: the pooled-key proxy; distinct values, so no ties).
//   phase 1: LLAMA_QSA_WHOLE_BLOCKS=1 (the reference's form): selected entries == the reference's; the default cell form
//            (width 2051): the reference plus only the inherited D1 cells of its 513th-ranked block. The carrier is never
//            selected (the reference has no such entry). Every block's rope position == its first member's (pb*r + 1).
//   phase 0: selected entries == the same loop over the fork's own list [c, e_1, .., e_q] (blocks {c,e1,e2,e3},
//            {e4..e7}, ..) and != the reference: the documented one-entry phase shift (the diagnosis).
// Depths >= LLAMA_QSA_MIN_KV (65 536) with every tail residue n_visible mod 4 = 0..3.
//
// usage: test-mtp-qsa-phase [seed = 7]

#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-kv-cells.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static constexpr int64_t R     = 4;
static constexpr int     N_SEL = 2048/R;   // indexer_top_k / r

// one GGML_OP_QSA_TOP_K node on the CPU: the whole-block form (n_sel > 0) or the cell form (n_sel == 0, prod default)
static std::vector<int32_t> run_op(int64_t n_kv, int64_t n_blocks, int64_t n_tps, const std::vector<float> & score,
                                   const std::vector<int32_t> & cell_blk, const std::vector<float> & mask, int width,
                                   int n_sel) {
    const size_t data = score.size()*4 + cell_blk.size()*4 + mask.size()*4 + (size_t) width*n_tps*4;
    const size_t work = (size_t) 2*n_kv*4*16 + (16u << 20);

    ggml_init_params ip = { data + work + (4u << 20), nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * sc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, n_tps, 1);
    ggml_tensor * cb = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, 1);
    ggml_tensor * mk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tps, 1, 1);

    memcpy(sc->data, score.data(),    score.size()*sizeof(float));
    memcpy(cb->data, cell_blk.data(), cell_blk.size()*sizeof(int32_t));
    memcpy(mk->data, mask.data(),     mask.size()*sizeof(float));

    ggml_tensor * out = n_sel > 0 ? ggml_qsa_top_k_blocks(ctx, sc, cb, mk, width, n_sel)
                                  : ggml_qsa_top_k(ctx, sc, cb, mk, width);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    std::vector<int32_t> res((size_t) width*n_tps);
    memcpy(res.data(), out->data, res.size()*sizeof(int32_t));

    ggml_free(ctx);

    return res;
}

// the reference loop over one query's visible list `vis` (entry ids in kv order): whole blocks of R from the start,
// scored by the f32 sum of their members' values, the top min(N_SEL, n_complete) (equal scores: the lower block), plus
// the tail; the selected entry ids, sorted; next (optional): the entries of the (N_SEL+1)-th ranked block (D1's cells)
static std::vector<int32_t> ref_select(const std::vector<int32_t> & vis, const float * v,
                                       std::vector<int32_t> * next = nullptr) {
    const int64_t n_complete = (int64_t) vis.size()/R;

    std::vector<float> score(n_complete);
    for (int64_t b = 0; b < n_complete; ++b) {
        float s = 0.0f;
        for (int64_t m = 0; m < R; ++m) {
            s += v[vis[b*R + m]];
        }
        score[b] = s;
    }

    std::vector<int32_t> blocks(n_complete);
    std::iota(blocks.begin(), blocks.end(), 0);

    const int64_t n_take = std::min<int64_t>(N_SEL, n_complete);
    const int64_t n_part = std::min<int64_t>(N_SEL + 1, n_complete);
    std::partial_sort(blocks.begin(), blocks.begin() + n_part, blocks.end(), [&](int32_t a, int32_t b) {
        return score[a] > score[b] || (score[a] == score[b] && a < b);
    });

    if (next) {
        next->clear();
        if (n_complete > N_SEL) {
            for (int64_t m = 0; m < R; ++m) {
                next->push_back(vis[blocks[N_SEL]*R + m]);
            }
        }
    }

    std::vector<int32_t> out;
    for (int64_t i = 0; i < n_take; ++i) {
        for (int64_t m = 0; m < R; ++m) {
            out.push_back(vis[blocks[i]*R + m]);
        }
    }
    for (int64_t k = n_complete*R; k < (int64_t) vis.size(); ++k) {
        out.push_back(vis[k]);
    }
    std::sort(out.begin(), out.end());

    return out;
}

struct row_result {
    bool eq_ref    = false; // whole blocks: fork == reference; cells (default): nothing missing, extras only the
                            // inherited D1 cells of the 513th-ranked reference block
    bool eq_shift  = false; // fork == the loop over the fork's own list (the phase-0 grouping)
    int64_t n_diff = 0;     // |fork xor reference| (entries)
};

// one case: P entries (fork positions 0..P), queries at fork positions qs, the given phase
static std::vector<row_result> run_case(int64_t P, const std::vector<int64_t> & qs, int32_t phase, uint32_t seed) {
    const int64_t n_kv     = (P + 1 + 3*R + 255)/256*256;   // >= 2 spare ids: the fragment kill has its own id
    const int64_t n_blocks = (n_kv + R - 1)/R;
    const int64_t n_tps    = (int64_t) qs.size();

    std::mt19937 rng(seed);

    // the fork's cells: positions 0..P (0 = the carrier) in permuted cells, one sequence
    std::vector<int32_t> perm(n_kv);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    llama_kv_cells cells;
    cells.resize((uint32_t) n_kv);

    std::vector<int32_t> pos_cell(P + 1);
    for (int64_t p = 0; p <= P; ++p) {
        cells.pos_set((uint32_t) perm[p], (llama_pos) p);
        cells.seq_add((uint32_t) perm[p], 0);
        pos_cell[p] = perm[p];
    }

    // the real census at this phase
    std::vector<int32_t> cell_blk(n_kv), blk_cells(R*n_blocks), blk_pos(4*n_blocks, 0);
    llama_qsa_census_stream cs;
    llama_qsa_census_full(cells, false, n_kv, n_blocks, R, true, cell_blk.data(), blk_cells.data(), blk_pos.data(),
                          n_blocks, cs, nullptr, phase);

    // the block rope position is the first member's position (the reference: rope at the block's first token)
    for (int32_t b = 0; b < cs.n_bid; ++b) {
        const llama_pos p0 = cells.pos_get(blk_cells[b*R + 0]);
        for (int64_t sec = 0; sec < 4; ++sec) {
            CHECK(blk_pos[sec*n_blocks + b] == p0, "phase %d P %lld: block %d rope pos %d != first member pos %d", phase,
                  (long long) P, b, blk_pos[sec*n_blocks + b], p0);
        }
        CHECK(cs.bid_idx[b] + phase == p0 && (p0 - phase) % R == 0, "phase %d: block %d key %d / first pos %d",
              phase, b, cs.bid_idx[b], p0);
    }

    // the compact block table, as the graph gets it
    std::vector<int32_t> blk(2*n_blocks);
    llama_qsa_fill_blk_tab(cells, cs, n_blocks, 0, n_blocks, /*tail_fix =*/ true, blk.data());

    // per-entry values per row (entry id = fork position; 0 = the carrier)
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<float> val((size_t) n_tps*(P + 1));
    for (auto & x : val) {
        x = U(rng);
    }

    std::vector<float> score((size_t) n_blocks*n_tps), mask((size_t) n_kv*n_tps);

    for (int64_t t = 0; t < n_tps; ++t) {
        const int64_t q    = qs[t];
        const float * v    = val.data() + t*(P + 1);
        const int64_t tail = llama_qsa_tail_start(q - phase, R);

        for (int64_t j = 0; j < n_kv; ++j) {
            mask[t*n_kv + j] = (!cells.is_empty(j) && cells.pos_get(j) <= q) ? 0.0f : -INFINITY;
        }

        for (int64_t b = 0; b < n_blocks; ++b) {
            float raw = 0.0f;
            if (b < cs.n_bid) {
                for (int64_t m = 0; m < R; ++m) {
                    raw += v[cells.pos_get(blk_cells[b*R + m])];
                }
            }
            const int32_t  key = blk[2*b + 0];
            const uint32_t vis = (uint32_t) blk[2*b + 1];
            // the op's formula (ggml_compute_forward_qsa_bias), bit = 1 << seq 0
            const float bias = (vis & 1u) == 0 ? -INFINITY : (key >= tail ? 1e9f : 0.0f);
            score[t*n_blocks + b] = raw + bias;
        }
    }

    const bool wb    = llama_qsa_whole_blocks();
    const int  width = (int) std::min<int64_t>(n_kv, 2048 + R - 1);
    const std::vector<int32_t> sel = run_op(n_kv, n_blocks, n_tps, score, cell_blk, mask, width, wb ? N_SEL : 0);

    std::vector<row_result> res(n_tps);

    for (int64_t t = 0; t < n_tps; ++t) {
        const int64_t   q = qs[t];
        const float   * v = val.data() + t*(P + 1);
        const int32_t * w = sel.data() + t*width;

        std::vector<int32_t> fork;
        if (wb) {
            int64_t nv = width;
            for (int64_t j = 1; j < width; ++j) {
                if (w[j] == w[j - 1]) {
                    nv = j;
                    break;
                }
            }
            for (int64_t j = 0; j < nv; ++j) {
                CHECK(!cells.is_empty(w[j]) && cells.pos_get(w[j]) <= q, "row %lld: selected an invisible cell", (long long) t);
                fork.push_back(cells.pos_get(w[j]));   // the entry id
            }
        } else {
            // the cell form: the visible part of the window
            for (int64_t j = 0; j < width; ++j) {
                if (std::isfinite(mask[t*n_kv + w[j]])) {
                    fork.push_back(cells.pos_get(w[j]));
                }
            }
        }
        std::sort(fork.begin(), fork.end());
        CHECK(std::adjacent_find(fork.begin(), fork.end()) == fork.end(), "row %lld: repeated entry", (long long) t);

        // the reference: e_1..e_q (no carrier); the phase-0 grouping: c, e_1..e_q
        std::vector<int32_t> vis_ref(q), vis_fork(q + 1);
        std::iota(vis_ref .begin(), vis_ref .end(), 1);
        std::iota(vis_fork.begin(), vis_fork.end(), 0);

        std::vector<int32_t> next_ref, next_shift;
        std::vector<int32_t> ref   = ref_select(vis_ref,  v, &next_ref);
        std::vector<int32_t> shift = ref_select(vis_fork, v, &next_shift);

        std::vector<int32_t> x, extra, missing;
        std::set_symmetric_difference(fork.begin(), fork.end(), ref.begin(), ref.end(), std::back_inserter(x));
        std::set_difference(fork.begin(), fork.end(), ref.begin(), ref.end(), std::back_inserter(extra));
        std::set_difference(ref.begin(), ref.end(), fork.begin(), fork.end(), std::back_inserter(missing));

        res[t].n_diff = (int64_t) x.size();

        if (phase != 0) {
            CHECK(fork.empty() || fork[0] != 0, "phase 1, depth %lld: the carrier was selected", (long long) q);
        }

        if (wb) {
            res[t].eq_ref   = fork == ref;
            res[t].eq_shift = fork == shift;
        } else {
            // the cell form (D1): extras may only be cells of the next-ranked block; nothing of the reference is missing
            bool ok = missing.empty();
            for (const int32_t e : extra) {
                ok &= std::find(next_ref.begin(), next_ref.end(), e) != next_ref.end();
            }
            res[t].eq_ref = ok;

            std::vector<int32_t> ms, xs;
            std::set_difference(shift.begin(), shift.end(), fork.begin(), fork.end(), std::back_inserter(ms));
            std::set_difference(fork.begin(), fork.end(), shift.begin(), shift.end(), std::back_inserter(xs));
            bool oks = ms.empty();
            for (const int32_t e : xs) {
                oks &= std::find(next_shift.begin(), next_shift.end(), e) != next_shift.end();
            }
            res[t].eq_shift = oks;
        }
    }

    return res;
}

int main(int argc, char ** argv) {
    const uint32_t seed = argc > 1 ? (uint32_t) strtoul(argv[1], nullptr, 10) : 7;

    const bool wb = llama_qsa_whole_blocks();

    printf("test-mtp-qsa-phase: LLAMA_QSA_WHOLE_BLOCKS=%d (%s), r %lld, n_sel %d, seed %u\n", wb ? 1 : 0,
           wb ? "whole blocks: == means equal" : "cells, width 2051: == means equal up to D1", (long long) R, N_SEL, seed);

    // (P, query depths): the query at fork position q sees q reference entries, residue q mod 4
    struct case_t { int64_t P; std::vector<int64_t> qs; };
    const std::vector<case_t> cases = {
        { 65536 + 7,  { 65536, 65537, 65538, 65539, 65540, 65541, 65542, 65543 } },
        { 131072 + 9, { 131072, 131073, 131074, 131075, 100001, 100002, 100003, 100004, 131081 } },
        { 213000,     { 212996, 212997, 212998, 212999, 213000 } },
    };

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const auto & c = cases[ci];

        const std::vector<row_result> r1 = run_case(c.P, c.qs, 1, seed + (uint32_t) ci);
        const std::vector<row_result> r0 = run_case(c.P, c.qs, 0, seed + (uint32_t) ci);

        for (size_t t = 0; t < c.qs.size(); ++t) {
            const int64_t q = c.qs[t];
            printf("  depth %7lld (n_visible mod 4 = %lld): phase 1 %s the reference (|xor| %lld); phase 0 %s the "
                   "reference (|xor| %lld), %s the fork-list grouping\n",
                   (long long) q, (long long) (q % R),
                   r1[t].eq_ref ? "==" : "!=", (long long) r1[t].n_diff,
                   r0[t].n_diff == 0 ? "==" : "!=", (long long) r0[t].n_diff,
                   r0[t].eq_shift ? "==" : "!=");

            CHECK(r1[t].eq_ref, "phase 1, depth %lld: the selection differs from the reference", (long long) q);
            CHECK(r0[t].eq_shift, "phase 0, depth %lld: the selection is not the fork-list grouping", (long long) q);
            CHECK(r0[t].n_diff > 0, "phase 0, depth %lld: equal to the reference (the shift is not shown)", (long long) q);
        }
    }

    printf("%s\n", g_fail == 0 ? "OK" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
