// [TAG_QSA_WHOLE_BLOCKS] The qwen4exp QSA selection (GGML_OP_QSA_TOP_K) on the CPU against the reference.
//
// 1. LLAMA_QSA_WHOLE_BLOCKS parser: unset / "" / "0" -> 0, "1" -> 1, anything else -> -1. [TAG_QSA_POOL_REMAP] the
//    LLAMA_QSA_POOL_REMAP parser: unset / "" / "indexed" -> 1, "copy" -> 0, anything else -> -1.
// 2. Census-shaped inputs (tests/qsa-select-ref.h: one sequence, permuted cells, the +1e9 tail boost, the spare
//    dead id, full and partly empty caches, rows at every tail residue and with fewer complete blocks than the
//    budget), both selection forms, against a port of HF Qwen4ExpTextQSAIndexer (512 whole blocks + the tail):
//    - whole blocks (ggml_qsa_top_k_blocks): the selected set EQUALS the reference, |set| = 4 min(512, n_complete)
//      + t, the ids ascending, the padding a repeat of the last id - with distinct scores and with ReLU-style ties
//      (the tie order is the reference's unspecified one, made deterministic: lower block first);
//    - cells (ggml_qsa_top_k, width 2051, today's prod): with distinct scores the visible selection is the reference
//      PLUS the 3 - t lowest-index cells of the 513th-ranked block when n_complete > 512 (D1, confirmed by
//      execution), and equal to it otherwise; with ties the differing cells are counted (D2) and reported.
// 3. The test-backend-ops QSA_TOP_K vectors (qsa_select_init_tensors: blocked layout, ties 0..4): how many cells per
//    row the whole-block form selects differently from the cell form (reported, not asserted beyond consistency).
#include "ggml.h"
#include "ggml-cpu.h"

#include "qsa-select-ref.h"

#include "../src/llama-memory-hybrid-idx.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>

static int g_fail = 0;

#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

// run one QSA_TOP_K node on the CPU; n_sel == 0 = the cell form
static std::vector<int32_t> run_op(int64_t n_kv, int64_t n_blocks, int64_t n_tps, int64_t n_stream,
                                   const std::vector<float> & score, const std::vector<int32_t> & cell_blk,
                                   const std::vector<float> & mask, bool mask_f16, int width, int n_sel) {
    const size_t data = score.size()*4 + cell_blk.size()*4 + mask.size()*4 + (size_t) width*n_tps*n_stream*4;
    const size_t work = (size_t) 2*n_kv*4*16 + (16u << 20);

    ggml_init_params ip = { data + work + (4u << 20), nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * sc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, n_tps, n_stream);
    ggml_tensor * cb = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_stream);
    ggml_tensor * mk = ggml_new_tensor_4d(ctx, mask_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv, n_tps, 1, n_stream);

    memcpy(sc->data, score.data(), score.size()*sizeof(float));
    memcpy(cb->data, cell_blk.data(), cell_blk.size()*sizeof(int32_t));
    if (mask_f16) {
        ggml_fp16_t * d = (ggml_fp16_t *) mk->data;
        for (size_t i = 0; i < mask.size(); ++i) {
            d[i] = ggml_fp32_to_fp16(mask[i]);
        }
    } else {
        memcpy(mk->data, mask.data(), mask.size()*sizeof(float));
    }

    ggml_tensor * out = n_sel > 0 ? ggml_qsa_top_k_blocks(ctx, sc, cb, mk, width, n_sel)
                                  : ggml_qsa_top_k(ctx, sc, cb, mk, width);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 8);

    std::vector<int32_t> res((size_t) width*n_tps*n_stream);
    memcpy(res.data(), out->data, res.size()*sizeof(int32_t));

    ggml_free(ctx);

    return res;
}

// the valid part of a whole-block row: the ids before the first repeat
static int64_t whole_valid_len(const int32_t * row, int64_t width) {
    for (int64_t j = 1; j < width; ++j) {
        if (row[j] == row[j - 1]) {
            return j;
        }
    }
    return width;
}

static void test_parser() {
    CHECK(llama_qsa_whole_blocks_parse(nullptr) == 0, "parse(null)");
    CHECK(llama_qsa_whole_blocks_parse("")      == 0, "parse(\"\")");
    CHECK(llama_qsa_whole_blocks_parse("0")     == 0, "parse(0)");
    CHECK(llama_qsa_whole_blocks_parse("1")     == 1, "parse(1)");
    for (const char * bad : { "2", "true", "yes", " 1", "1 ", "01", "-1", "on" }) {
        CHECK(llama_qsa_whole_blocks_parse(bad) == -1, "parse(\"%s\") must be invalid", bad);
    }
    // [TAG_QSA_POOL_REMAP] LLAMA_QSA_POOL_REMAP: unset / "" / "indexed" -> 1 (indexed), "copy" -> 0, else -1
    CHECK(llama_qsa_pool_remap_parse(nullptr)   == 1, "remap parse(null)");
    CHECK(llama_qsa_pool_remap_parse("")        == 1, "remap parse(\"\")");
    CHECK(llama_qsa_pool_remap_parse("indexed") == 1, "remap parse(indexed)");
    CHECK(llama_qsa_pool_remap_parse("copy")    == 0, "remap parse(copy)");
    for (const char * bad : { "0", "1", "Copy", "INDEXED", " copy", "copy ", "index", "gather", "on" }) {
        CHECK(llama_qsa_pool_remap_parse(bad) == -1, "remap parse(\"%s\") must be invalid", bad);
    }
    printf("parser: ok\n");
}

struct census_stats {
    int64_t rows = 0;
    int64_t rows_whole_eq = 0;   // whole == reference
    int64_t rows_cells_eq = 0;   // cells (visible part) == reference
    int64_t cells_extra   = 0;   // cell form: visible cells the reference does not select
    int64_t cells_missing = 0;   // cell form: reference cells it does not select
    int64_t extra_by_t[4]   = {0, 0, 0, 0};
    int64_t rows_by_t[4]    = {0, 0, 0, 0};
};

static void test_census(int64_t n_kv, int64_t n_tps, int64_t n_stream, uint32_t seed, bool distinct, bool mask_f16,
                        census_stats & st) {
    const qsa_census_case c = qsa_census_make(n_kv, n_tps, n_stream, seed, distinct);

    const int64_t r      = c.r;
    const int     width  = (int) std::min<int64_t>(n_kv, 2048 + r - 1);
    const int     n_sel  = 2048/(int) r;

    const std::vector<int32_t> whole = run_op(n_kv, c.n_blocks, n_tps, n_stream, c.score, c.cell_blk, c.mask, mask_f16, width, n_sel);
    const std::vector<int32_t> cells = run_op(n_kv, c.n_blocks, n_tps, n_stream, c.score, c.cell_blk, c.mask, mask_f16, width, 0);

    for (int64_t s = 0; s < n_stream; ++s) {
        for (int64_t t = 0; t < n_tps; ++t) {
            const int64_t row = s*n_tps + t;
            const int64_t q   = c.qpos[row];
            const int64_t nc  = (q + 1)/r;
            const int64_t tr  = (q + 1)%r;

            const std::vector<int32_t> ref = qsa_ref_select(c, t, s, n_sel);

            // whole blocks
            const int32_t * w  = whole.data() + row*width;
            const int64_t   nv = whole_valid_len(w, width);
            std::vector<int32_t> wv(w, w + nv);

            CHECK(std::is_sorted(wv.begin(), wv.end()) && std::adjacent_find(wv.begin(), wv.end()) == wv.end(),
                  "whole row %lld: valid ids not strictly ascending", (long long) row);
            for (int64_t j = nv; j < width; ++j) {
                CHECK(w[j] == w[nv - 1], "whole row %lld: padding slot %lld = %d, not the last id %d",
                      (long long) row, (long long) j, w[j], w[nv - 1]);
            }
            CHECK((int64_t) wv.size() == r*std::min<int64_t>(n_sel, nc) + tr,
                  "whole row %lld (q %lld): %zu ids, expected 4 min(512, %lld) + %lld", (long long) row, (long long) q,
                  wv.size(), (long long) nc, (long long) tr);
            const bool weq = wv == ref;
            CHECK(weq, "whole row %lld (q %lld, t %lld): differs from the reference (%zu vs %zu ids)",
                  (long long) row, (long long) q, (long long) tr, wv.size(), ref.size());

            // cells: the visible part against the reference
            const int32_t * cr = cells.data() + row*width;
            const float   * mk = c.mask.data() + row*n_kv;
            std::vector<int32_t> cv;
            for (int64_t j = 0; j < width; ++j) {
                if (std::isfinite(mk[cr[j]])) {
                    cv.push_back(cr[j]);
                }
            }
            std::sort(cv.begin(), cv.end());
            CHECK(std::adjacent_find(cv.begin(), cv.end()) == cv.end(), "cells row %lld: repeated id", (long long) row);
            CHECK((int64_t) cv.size() == std::min<int64_t>(q + 1, width),
                  "cells row %lld: %zu visible ids, expected min(%lld, %d)", (long long) row, cv.size(), (long long) (q + 1), width);

            std::vector<int32_t> extra, missing;
            std::set_difference(cv.begin(), cv.end(), ref.begin(), ref.end(), std::back_inserter(extra));
            std::set_difference(ref.begin(), ref.end(), cv.begin(), cv.end(), std::back_inserter(missing));

            if (distinct) {
                // D1: the 3 - t lowest-index cells of the 513th-ranked block, nothing else
                std::vector<int32_t> expect;
                if (nc > n_sel) {
                    const float * rw = c.raw.data() + row*c.n_blocks;
                    std::vector<int32_t> blocks(nc);
                    for (int64_t k = 0; k < nc; ++k) {
                        blocks[k] = (int32_t) k;
                    }
                    std::nth_element(blocks.begin(), blocks.begin() + n_sel, blocks.end(), [rw](int32_t a, int32_t b) {
                        return rw[a] > rw[b] || (rw[a] == rw[b] && a < b);
                    });
                    const int32_t b513 = blocks[n_sel];
                    std::vector<int32_t> members;
                    for (int64_t m = 0; m < r; ++m) {
                        members.push_back(c.pos_cell[s*n_kv + b513*r + m]);
                    }
                    std::sort(members.begin(), members.end());
                    expect.assign(members.begin(), members.begin() + (r - 1 - tr));
                }
                CHECK(missing.empty() && extra == expect,
                      "cells row %lld (q %lld, t %lld, n_complete %lld): %zu extra / %zu missing, expected %zu extra (D1)",
                      (long long) row, (long long) q, (long long) tr, (long long) nc, extra.size(), missing.size(), expect.size());
            }

            st.rows++;
            st.rows_whole_eq += weq ? 1 : 0;
            st.rows_cells_eq += (extra.empty() && missing.empty()) ? 1 : 0;
            st.cells_extra   += (int64_t) extra.size();
            st.cells_missing += (int64_t) missing.size();
            if (nc > n_sel) {
                st.extra_by_t[tr] += (int64_t) extra.size();
                st.rows_by_t[tr]++;
            }
        }
    }
}

// test-backend-ops' qsa_select_init_tensors, blocked layout (cell i in block i/r), ties 0..4
static void make_backend_ops_vectors(int64_t n_blocks, int64_t n_kv, int64_t n_tps, int ties, int width,
                                     std::vector<float> & score, std::vector<int32_t> & cell_blk, std::vector<float> & mask) {
    const int64_t r = (n_kv + n_blocks - 1)/n_blocks;
    cell_blk.resize(n_kv);
    for (int64_t i = 0; i < n_kv; ++i) {
        cell_blk[i] = (int32_t) std::min(i/r, n_blocks - 1);
    }
    mask.resize(n_kv*n_tps);
    for (int64_t row = 0; row < n_tps; ++row) {
        const int64_t cut = std::max<int64_t>(1, width - 1 - (row % 4)*(width/8));
        for (int64_t i = 0; i < n_kv; ++i) {
            mask[row*n_kv + i] = ties == 4 ? (i < cut ? 0.0f : -INFINITY) : ties ? 0.0f : 0.25f*(float) (i % r);
        }
    }
    score.resize(n_blocks*n_tps);
    for (int64_t row = 0; row < n_tps; ++row) {
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t v = (b + 7919*row) % n_blocks;
            score[row*n_blocks + b] = ties == 2 ? 0.0f : (float) (ties == 3 ? v/8 : v);
        }
    }
}

static void test_backend_ops_vectors() {
    printf("\ntest-backend-ops QSA_TOP_K vectors (blocked, f16 mask, width 2051, n_sel 512): cells per row that the\n"
           "whole-block form attends differently from the cell form (|cells \\ whole| + |whole \\ cells| over the selected\n"
           "VISIBLE cells; padding and masked fillers attend to nothing). Not census-shaped (no tail boost): ties 0 = distinct\n"
           "cells via 0.25 (i %% r) in-block mask offsets, 1 = one score per block, 2 = all equal, 3 = 8 blocks per score,\n"
           "4 = a -inf mask past a per-row cutoff below the width\n");
    printf("%-26s %6s %12s %12s %10s\n", "shape", "ties", "rows differ", "cells diff", "per row");
    for (const auto & sh : { std::array<int64_t, 3>{ 1024, 4096, 6 }, std::array<int64_t, 3>{ 4096, 16384, 6 } }) {
        for (int ties = 0; ties <= 4; ++ties) {
            const int64_t n_blocks = sh[0], n_kv = sh[1], n_tps = sh[2];
            const int width = 2051;
            std::vector<float> score, mask;
            std::vector<int32_t> cell_blk;
            make_backend_ops_vectors(n_blocks, n_kv, n_tps, ties, width, score, cell_blk, mask);

            const std::vector<int32_t> whole = run_op(n_kv, n_blocks, n_tps, 1, score, cell_blk, mask, true, width, 512);
            const std::vector<int32_t> cells = run_op(n_kv, n_blocks, n_tps, 1, score, cell_blk, mask, true, width, 0);

            int64_t rows_differ = 0, diff = 0;
            for (int64_t row = 0; row < n_tps; ++row) {
                const int32_t * w  = whole.data() + row*width;
                const int64_t   nv = whole_valid_len(w, width);
                CHECK(std::is_sorted(w, w + nv), "backend-ops vector ties %d row %lld: not ascending", ties, (long long) row);
                // the ATTENDED cells: selected and visible (a masked filler of the cell form attends to nothing)
                const float * mk = mask.data() + row*n_kv;
                std::vector<int32_t> wv, cv;
                for (int64_t j = 0; j < nv; ++j) {
                    if (std::isfinite(mk[w[j]])) {
                        wv.push_back(w[j]);
                    }
                }
                for (int64_t j = 0; j < width; ++j) {
                    const int32_t id = cells[row*width + j];
                    if (std::isfinite(mk[id])) {
                        cv.push_back(id);
                    }
                }
                std::sort(cv.begin(), cv.end());
                std::vector<int32_t> sd;
                std::set_symmetric_difference(cv.begin(), cv.end(), wv.begin(), wv.end(), std::back_inserter(sd));
                rows_differ += sd.empty() ? 0 : 1;
                diff += (int64_t) sd.size();
            }
            char name[64];
            snprintf(name, sizeof(name), "%lld/%lld/%lld", (long long) n_blocks, (long long) n_kv, (long long) n_tps);
            printf("%-26s %6d %8lld/%-3lld %12lld %10.2f\n", name, ties, (long long) rows_differ, (long long) n_tps,
                   (long long) diff, (double) diff/(double) n_tps);
        }
    }
}

int main() {
    test_parser();

    struct cfg { int64_t n_kv, n_tps, n_stream; uint32_t seed; bool distinct, f16; };
    const cfg cfgs[] = {
        {  2600, 8, 1, 11, true,  true  },
        {  4096, 8, 1,  1, true,  true  },
        {  4096, 8, 2,  2, false, true  },
        { 16384, 6, 1,  3, true,  true  },
        { 16384, 6, 1,  4, false, true  },
        { 16384, 6, 3,  8, false, false },
        { 65536, 4, 2,  5, true,  true  },
        { 65536, 4, 1,  6, false, true  },
    };

    census_stats sd, st;
    for (const cfg & k : cfgs) {
        test_census(k.n_kv, k.n_tps, k.n_stream, k.seed, k.distinct, k.f16, k.distinct ? sd : st);
    }

    printf("\ncensus vectors vs the HF reference (512 whole blocks + tail):\n");
    printf("  distinct scores: %lld rows; whole == ref %lld; cells == ref %lld; cells extra %lld / missing %lld\n",
           (long long) sd.rows, (long long) sd.rows_whole_eq, (long long) sd.rows_cells_eq,
           (long long) sd.cells_extra, (long long) sd.cells_missing);
    printf("    cells extra per row above the budget by tail residue t: t0 %lld/%lld  t1 %lld/%lld  t2 %lld/%lld  t3 %lld/%lld (D1 predicts 3, 2, 1, 0)\n",
           (long long) sd.extra_by_t[0], (long long) sd.rows_by_t[0], (long long) sd.extra_by_t[1], (long long) sd.rows_by_t[1],
           (long long) sd.extra_by_t[2], (long long) sd.rows_by_t[2], (long long) sd.extra_by_t[3], (long long) sd.rows_by_t[3]);
    printf("  ReLU ties:       %lld rows; whole == ref %lld; cells == ref %lld; cells extra %lld / missing %lld (D1 + D2)\n",
           (long long) st.rows, (long long) st.rows_whole_eq, (long long) st.rows_cells_eq,
           (long long) st.cells_extra, (long long) st.cells_missing);

    CHECK(sd.rows_whole_eq == sd.rows && st.rows_whole_eq == st.rows, "whole-block form != reference on some row");

    test_backend_ops_vectors();

    printf("\n%s (%d failure(s))\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
