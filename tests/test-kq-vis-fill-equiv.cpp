// [TAG_KQ_VIS_FILL] randomized equivalence test of the visibility metadata fill (P3-F1):
//   new: llama_kv_cells_fill_kq_vis  - one branch-free pass over the cells, seq mask = the cells' low-32 shadow word
//   ref: llama_kv_cells_fill_kq_vis_ref (kq-vis-fill-ref.h) - the pre-P3-F1 body, 32 seq_has tests per cell
// compared byte for byte (cell records, token records and the return value) after random sequences of every
// llama_kv_cells mutator: apply-style overwrites (rm + pos_set + ext_set + seq_add), seq_add / seq_rm / seq_keep per cell
// and per stream (the llama_kv_cache seq_* loops), rm, pos_add (shifts, incl. below 0 = the cell empties with its ext
// left in place), pos_div, cp / set by range and by index list (checkpoint save / restore, also across streams),
// whole-stream copies, reset, resize, reset_shift. 1..4 streams, seq ids over [0, LLAMA_MAX_SEQ) (ids >= 32 are
// invisible to the records), holes, seqless used cells, multi-seq cells, M-RoPE ext; ubatches unified and per-stream
// blocks, 1-D and 2-D positions, and rejected ones (a seq id >= 32, a block mixing streams). Every check also compares
// seq_mask32(i) against the 32 seq_has tests for EVERY cell (empty ones included), so a stale shadow word is caught
// before a later pos_set could expose it.
// Negative controls: (1) one flipped seq bit must be detected; (2) a mutant fill (get_vis<false>: no empty-cell mask) runs
// through the same comparison and must be detected, only in the x / y planes of empty cells (the stale ext a pos_add below
// 0 leaves behind), at least once.
// Rebase tripwire: the write statements of seq[...] and of its shadow seq_lo[...] in src/llama-kv-cells.h (comments and
// string literals stripped) must be equally many - on an upstream sync, every seq[ write needs its seq_lo twin.
// Debug builds (NDEBUG undefined: asserts, incl. the seq_lo check in get_vis / seq_mask32) cap the default workload to
// 100 rounds without the prod-sized pools; pass "full" as the third argument for the Release workload.
//
// usage: test-kq-vis-fill-equiv [n_rounds = 300 (Debug: 100)] [seed = 42] [full]

#include "ggml.h"

#include "kq-vis-fill-ref.h"

#include "../src/llama-batch.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#ifndef LLAMA_KV_CELLS_H_PATH
#define LLAMA_KV_CELLS_H_PATH ""
#endif

namespace {

struct rng_t {
    std::mt19937_64 g;

    explicit rng_t(uint64_t seed) : g(seed) {}

    // uniform in [0, n)
    uint64_t u(uint64_t n) { return n == 0 ? 0 : std::uniform_int_distribution<uint64_t>(0, n - 1)(g); }
    int64_t  r(int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(g); }
    bool     p(double x) { return std::uniform_real_distribution<double>(0.0, 1.0)(g) < x; }
};

struct stats_t {
    uint64_t ops = 0, checks = 0, cells_cmp = 0, bytes_cmp = 0, rejected = 0, is_2d = 0, multi_block = 0;
    uint64_t cells_empty = 0, cells_seqless = 0, cells_multi = 0, cells_hi_only = 0, emptied_by_shift = 0;
    uint64_t op_count[16] = {};
    uint64_t diff = 0, shadow_diff = 0, ctrl_runs = 0, ctrl_detected = 0;
    uint64_t mut_runs = 0, mut_detected = 0, mut_words = 0, mut_misplaced = 0;
};

// mostly low ids (visible), some in [8, 32), some >= 32 (invisible to the records)
llama_seq_id rand_seq(rng_t & rng, int n_seq_hint) {
    if (rng.p(0.8)) {
        return (llama_seq_id) rng.u(n_seq_hint);
    }
    if (rng.p(0.5)) {
        return (llama_seq_id) rng.u(32);
    }
    return (llama_seq_id) (32 + rng.u(LLAMA_MAX_SEQ - 32));
}

uint32_t ref_mask32(const llama_kv_cells & cells, uint32_t i) {
    uint32_t m = 0;
    for (llama_seq_id k = 0; k < 32; ++k) {
        m |= cells.seq_has(i, k) ? (1u << k) : 0u;
    }
    return m;
}

// a used cell with at least one seq (seq_keep's precondition for a used cell)
bool cell_has_any_seq(const llama_kv_cells & cells, uint32_t i) {
    return !cells.is_empty(i) && cells.seq_count(i) > 0;
}

llama_seq_id pick_seq_of(const llama_kv_cells & cells, uint32_t i, rng_t & rng) {
    std::vector<llama_seq_id> have;
    for (llama_seq_id k = 0; k < LLAMA_MAX_SEQ; ++k) {
        if (cells.seq_has(i, k)) {
            have.push_back(k);
        }
    }
    return have.empty() ? -1 : have[rng.u(have.size())];
}

llama_kv_cell_ext rand_ext(rng_t & rng) {
    llama_kv_cell_ext e;
    e.x   = (llama_pos) rng.r(-5, 5000);
    e.y   = (llama_pos) rng.r(-5, 5000);
    e.tok = (llama_token) rng.r(-1, 150000);
    return e;
}

// the llama_kv_cache::apply_ubatch cell write: overwrite (rm if used), position, optional ext, seqs
void op_apply(llama_kv_cells & cells, uint32_t i, rng_t & rng, int n_seq_hint) {
    if (!cells.is_empty(i)) {
        cells.rm(i);
    }
    cells.pos_set(i, (llama_pos) (rng.p(0.9) ? rng.u(1u << 20) : rng.u(64)));
    if (rng.p(0.3)) {
        cells.ext_set(i, rand_ext(rng));
    }
    if (rng.p(0.04)) {
        return; // a seqless used cell
    }
    const int n_add = rng.p(0.8) ? 1 : 1 + (int) rng.u(4);
    for (int a = 0; a < n_add; ++a) {
        const llama_seq_id s = rand_seq(rng, n_seq_hint);
        if (!cells.seq_has(i, s)) {
            cells.seq_add(i, s);
        }
    }
}

std::vector<uint32_t> rand_idxs(rng_t & rng, uint32_t size, uint32_t n) {
    std::vector<uint32_t> all(size);
    std::iota(all.begin(), all.end(), 0u);
    for (uint32_t k = 0; k < n; ++k) {
        std::swap(all[k], all[k + rng.u(size - k)]);
    }
    all.resize(n);
    return all;
}

void random_op(std::vector<llama_kv_cells> & v, rng_t & rng, int n_seq_hint, bool large, stats_t & st) {
    const uint32_t ns   = (uint32_t) v.size();
    const uint32_t s    = (uint32_t) rng.u(ns);
    auto &         c    = v[s];
    const uint32_t size = c.size();
    const uint32_t i    = (uint32_t) rng.u(size);

    // weights: the per-cell ops dominate, the structural ones are rare
    static const int w[] = { 30, 8, 8, 4, 6, 6, 3, 4, 4, 2, 2, 1, 1, 2, 1, 2 };
    int total = 0;
    for (int x : w) total += x;
    int pick = (int) rng.u(total), op = 0;
    while (pick >= w[op]) { pick -= w[op]; ++op; }
    if (large && (op == 11 || op == 12)) {
        op = 0; // keep the prod-sized pools filled
    }
    st.op_count[op]++;
    st.ops++;

    switch (op) {
        case 0: { // apply-style overwrite of a short run of cells (a ubatch)
            const uint32_t n = std::min<uint32_t>(size - i, 1 + (uint32_t) rng.u(rng.p(0.8) ? 6 : 300));
            for (uint32_t k = 0; k < n; ++k) {
                op_apply(c, i + k, rng, n_seq_hint);
            }
        } break;
        case 1: { // seq_add to a used cell (seq_cp within a stream)
            if (!c.is_empty(i)) {
                const llama_seq_id q = rand_seq(rng, n_seq_hint);
                if (!c.seq_has(i, q)) {
                    c.seq_add(i, q);
                }
            }
        } break;
        case 2: { // seq_rm of one of the cell's seqs
            if (!c.is_empty(i)) {
                const llama_seq_id q = pick_seq_of(c, i, rng);
                if (q >= 0) {
                    c.seq_rm(i, q);
                }
            }
        } break;
        case 3: { // seq_keep (skip used seqless cells: seq_keep's precondition)
            if (c.is_empty(i) || cell_has_any_seq(c, i)) {
                c.seq_keep(i, rng.p(0.5) && cell_has_any_seq(c, i) ? pick_seq_of(c, i, rng) : rand_seq(rng, n_seq_hint));
            }
        } break;
        case 4: { // rm
            if (!c.is_empty(i)) {
                c.rm(i);
            }
        } break;
        case 5: { // pos_add, sometimes below 0 (the cell empties, its ext is left as is)
            if (!c.is_empty(i)) {
                const llama_pos p = c.pos_get(i);
                const llama_pos d = rng.p(0.3) ? (llama_pos) -(p + 1 + (llama_pos) rng.u(8)) : (llama_pos) rng.r(-64, 1000);
                st.emptied_by_shift += c.pos_add(i, d);
            }
        } break;
        case 6: { // pos_div
            if (!c.is_empty(i)) {
                c.pos_div(i, 1 + (int) rng.u(4));
            }
        } break;
        case 7: { // cp(range) -> mutate -> set(range): a checkpoint restore over the same cells
            c.reset_shift();
            const uint32_t n = std::min<uint32_t>(size - i, 1 + (uint32_t) rng.u(64));
            const llama_kv_cells saved = c.cp(i, n);
            for (int k = 0; k < 8; ++k) {
                const uint32_t j = i + (uint32_t) rng.u(n);
                op_apply(c, j, rng, n_seq_hint);
            }
            c.reset_shift();
            c.set(i, saved);
        } break;
        case 8: { // cp(idxs) -> set(other idxs): restore into different cells (also over used ones)
            c.reset_shift();
            const uint32_t n = std::min<uint32_t>(size, 1 + (uint32_t) rng.u(32));
            const auto src = rand_idxs(rng, size, n);
            const auto dst = rng.p(0.5) ? src : rand_idxs(rng, size, n);
            const llama_kv_cells saved = c.cp(src);
            c.set(dst, saved);
        } break;
        case 9: { // cp(range) of another stream -> set(range) here
            if (ns > 1) {
                const uint32_t s2 = (s + 1 + (uint32_t) rng.u(ns - 1)) % ns;
                v[s2].reset_shift();
                c.reset_shift();
                const uint32_t n = std::min<uint32_t>(size - i, 1 + (uint32_t) rng.u(128));
                c.set(i, v[s2].cp(i, n));
            }
        } break;
        case 10: { // whole-stream copy (the object copy the state paths and tests rely on)
            if (ns > 1) {
                const uint32_t s2 = (s + 1 + (uint32_t) rng.u(ns - 1)) % ns;
                v[s2] = c;
            } else {
                const llama_kv_cells tmp = c;
                c = tmp;
            }
        } break;
        case 11: { // reset (seq_rm(-1) / clear)
            c.reset();
        } break;
        case 12: { // resize all streams (reinit), then refill a few runs
            const uint32_t n = 256 + (uint32_t) rng.u(4096);
            for (auto & cc : v) {
                cc.resize(n);
            }
            for (int k = 0; k < 16; ++k) {
                const uint32_t ss = (uint32_t) rng.u(ns);
                const uint32_t j  = (uint32_t) rng.u(n);
                const uint32_t m  = std::min<uint32_t>(n - j, 1 + (uint32_t) rng.u(200));
                for (uint32_t q = 0; q < m; ++q) {
                    op_apply(v[ss], j + q, rng, n_seq_hint);
                }
            }
        } break;
        case 13: { // llama_kv_cache::seq_rm(seq, p0, p1) over the stream
            const llama_seq_id q  = rand_seq(rng, n_seq_hint);
            const llama_pos    p0 = (llama_pos) rng.u(1u << 20);
            const llama_pos    p1 = rng.p(0.5) ? std::numeric_limits<llama_pos>::max() : p0 + (llama_pos) rng.u(1u << 18);
            for (uint32_t j = 0; j < size; ++j) {
                if (c.pos_in(j, p0, p1) && c.seq_has(j, q)) {
                    c.seq_rm(j, q);
                }
            }
        } break;
        case 14: { // llama_kv_cache::seq_keep(seq) over the stream
            const llama_seq_id q = rand_seq(rng, n_seq_hint);
            for (uint32_t j = 0; j < size; ++j) {
                if (c.is_empty(j) || cell_has_any_seq(c, j)) {
                    c.seq_keep(j, q);
                }
            }
        } break;
        case 15: { // llama_kv_cache::seq_cp (same stream) / seq_add (shift) over the stream
            const llama_seq_id q  = rand_seq(rng, n_seq_hint);
            const llama_seq_id q2 = rand_seq(rng, n_seq_hint);
            const bool shift = rng.p(0.4);
            const llama_pos d = (llama_pos) rng.r(-2000, 2000);
            for (uint32_t j = 0; j < size; ++j) {
                if (c.is_empty(j) || !c.seq_has(j, q)) {
                    continue;
                }
                if (shift) {
                    st.emptied_by_shift += c.pos_add(j, d);
                } else if (!c.seq_has(j, q2)) {
                    c.seq_add(j, q2);
                }
            }
        } break;
        default: break;
    }
}

// negative control 2: the real fill, then every stream block's cell planes rewritten by get_vis<false> (no empty-cell mask:
// an empty cell keeps its stale ext in the x / y planes)
bool fill_mutant_nomask(int32_t * cells_data, int32_t * toks_data, int64_t n_kv, int64_t n_stream, const llama_ubatch * ubatch,
        const std::vector<llama_kv_cells> & v_cells, const std::vector<uint32_t> & seq_to_stream) {
    const bool ok = llama_kv_cells_fill_kq_vis(cells_data, toks_data, n_kv, n_stream, ubatch, v_cells, seq_to_stream);
    if (!ok) {
        return ok;
    }
    const int64_t n_tps = ubatch->n_tokens/n_stream;
    for (int64_t s = 0; s < n_stream; ++s) {
        int32_t * c = cells_data + 4*s*n_kv;
        v_cells[seq_to_stream[ubatch->seq_id[s*n_tps][0]]].get_vis<false>((uint32_t) n_kv, c, c + n_kv, c + 2*n_kv, c + 3*n_kv);
    }
    return ok;
}

// rebase tripwire: count the write statements of seq[...] and seq_lo[...] in llama-kv-cells.h; 0 = equal (and > 0)
int tripwire() {
    std::string path = LLAMA_KV_CELLS_H_PATH;
    if (path.empty()) {
        path = __FILE__;
        path = path.substr(0, path.find_last_of("/\\") + 1) + "../src/llama-kv-cells.h";
    }
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "tripwire: cannot open %s\n", path.c_str());
        return 1;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string src = ss.str();

    // strip // and /* */ comments and string literals
    std::string code;
    for (size_t i = 0; i < src.size(); ) {
        if (src.compare(i, 2, "//") == 0) {
            const size_t e = src.find('\n', i);
            i = e == std::string::npos ? src.size() : e;
        } else if (src.compare(i, 2, "/*") == 0) {
            const size_t e = src.find("*/", i + 2);
            i = e == std::string::npos ? src.size() : e + 2;
        } else if (src[i] == '"') {
            size_t j = i + 1;
            while (j < src.size() && src[j] != '"') {
                j += src[j] == '\\' ? 2 : 1;
            }
            code += "\"\"";
            i = j + 1;
        } else {
            code += src[i++];
        }
    }

    // a write: assignment (incl. compound) to an element, or a mutating bitset call on it
    const std::regex re_seq   (R"(\bseq\s*\[[^\]]*\]\s*(?:[&|^]?=(?!=)|\.\s*(?:set|reset|flip)\s*\())");
    const std::regex re_seq_lo(R"(\bseq_lo\s*\[[^\]]*\]\s*[&|^]?=(?!=))");
    const auto count = [&](const std::regex & re) {
        return (int) std::distance(std::sregex_iterator(code.begin(), code.end(), re), std::sregex_iterator());
    };
    const int n_seq = count(re_seq), n_lo = count(re_seq_lo);
    const bool ok = n_seq > 0 && n_seq == n_lo;
    printf("  rebase tripwire (%s): %d seq[...] write statements, %d seq_lo[...] write statements -> %s\n", path.c_str(), n_seq, n_lo,
            ok ? "paired" : "MISMATCH: on upstream sync, pair every seq[ write in llama-kv-cells.h with a seq_lo update");
    return ok ? 0 : 1;
}

struct ubatch_buf {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_data;
    std::vector<llama_seq_id *> seq_ptr;
    llama_ubatch                ub{};
};

// n_blk stream blocks of n_tps tokens; per block the tokens' seqs map to one stream (unless a rejection is injected)
void make_ubatch(ubatch_buf & b, rng_t & rng, int64_t n_blk, int64_t n_tps, const std::vector<uint32_t> & seq_to_stream,
        uint32_t n_streams, bool is_2d, int n_seq_hint) {
    const int64_t n_tokens = n_blk*n_tps;
    const uint32_t n_pos   = is_2d ? 4 : 1;

    b.pos.assign(n_tokens*n_pos, 0);
    b.n_seq_id.assign(n_tokens, 1);
    b.seq_data.assign(n_tokens, 0);
    b.seq_ptr.assign(n_tokens, nullptr);

    for (int64_t blk = 0; blk < n_blk; ++blk) {
        const uint32_t st = (uint32_t) rng.u(n_streams);
        std::vector<llama_seq_id> cand;
        for (llama_seq_id q = 0; q < 32; ++q) {
            if (seq_to_stream[q] == st && q < std::max(n_seq_hint, 1) + 4) {
                cand.push_back(q);
            }
        }
        if (cand.empty()) {
            cand.push_back(0);
        }
        for (int64_t k = 0; k < n_tps; ++k) {
            b.seq_data[blk*n_tps + k] = cand[rng.u(cand.size())];
        }
    }
    // rare rejections: a seq id >= 32, or a token of another stream inside a block
    if (rng.p(0.05)) {
        b.seq_data[rng.u(n_tokens)] = (llama_seq_id) (32 + rng.u(LLAMA_MAX_SEQ - 32));
    } else if (rng.p(0.05) && n_streams > 1) {
        b.seq_data[rng.u(n_tokens)] = (llama_seq_id) rng.u(32);
    }
    for (int64_t k = 0; k < n_tokens; ++k) {
        b.seq_ptr[k] = &b.seq_data[k];
        for (uint32_t d = 0; d < n_pos; ++d) {
            b.pos[k + d*n_tokens] = (llama_pos) (d == 0 ? rng.u(1u << 20) : rng.u(5000));
        }
    }

    b.ub = llama_ubatch{};
    b.ub.n_tokens     = (uint32_t) n_tokens;
    b.ub.n_seq_tokens = (uint32_t) n_tps;
    b.ub.n_seqs       = (uint32_t) n_blk;
    b.ub.n_pos        = n_pos;
    b.ub.pos          = b.pos.data();
    b.ub.n_seq_id     = b.n_seq_id.data();
    b.ub.seq_id       = b.seq_ptr.data();
}

// one comparison: random ubatch + n_kv, same garbage in both destinations, byte-for-byte compare
void check(const std::vector<llama_kv_cells> & v, rng_t & rng, const std::vector<uint32_t> & seq_to_stream,
        int n_seq_hint, stats_t & st) {
    const uint32_t ns   = (uint32_t) v.size();
    const uint32_t size = v[0].size();

    // the shadow word of every cell of every stream
    for (uint32_t s = 0; s < ns; ++s) {
        for (uint32_t i = 0; i < size; ++i) {
            const uint32_t ref = ref_mask32(v[s], i);
            if (v[s].seq_mask32(i) != ref) {
                if (st.shadow_diff < 10) {
                    fprintf(stderr, "shadow mismatch: stream %u cell %u seq_mask32 %08x ref %08x\n", s, i, v[s].seq_mask32(i), ref);
                }
                st.shadow_diff++;
            }
            if (v[s].is_empty(i)) {
                st.cells_empty++;
            } else {
                const int cnt = v[s].seq_count(i);
                st.cells_seqless += cnt == 0;
                st.cells_multi   += cnt > 1;
                st.cells_hi_only += cnt > 0 && ref == 0;
            }
        }
    }

    const bool    is_2d = rng.p(0.3);
    const int64_t n_blk = ns == 1 ? 1 : 1 + (int64_t) rng.u(ns + 1);
    const int64_t n_tps = rng.p(0.6) ? 1 + (int64_t) rng.u(6) : 1 + (int64_t) rng.u(300);
    int64_t n_kv;
    if (rng.p(0.3)) {
        n_kv = size;
    } else if (rng.p(0.5)) {
        n_kv = std::max<int64_t>(1, (int64_t) (rng.u(size + 1)/256)*256);
    } else {
        n_kv = 1 + (int64_t) rng.u(size);
    }

    ubatch_buf b;
    make_ubatch(b, rng, n_blk, n_tps, seq_to_stream, ns, is_2d, n_seq_hint);

    const size_t n_cells_i32 = (size_t) 4*n_kv*n_blk;
    const size_t n_toks_i32  = (size_t) 4*b.ub.n_tokens;
    std::vector<int32_t> c_new(n_cells_i32), c_ref(n_cells_i32), t_new(n_toks_i32), t_ref(n_toks_i32);
    for (auto & x : c_ref) x = (int32_t) rng.g();
    for (auto & x : t_ref) x = (int32_t) rng.g();
    c_new = c_ref;
    t_new = t_ref;
    std::vector<int32_t> c_mut = c_ref, t_mut = t_ref;

    const bool ok_new = llama_kv_cells_fill_kq_vis    (c_new.data(), t_new.data(), n_kv, n_blk, &b.ub, v, seq_to_stream);
    const bool ok_ref = llama_kv_cells_fill_kq_vis_ref(c_ref.data(), t_ref.data(), n_kv, n_blk, &b.ub, v, seq_to_stream);

    st.checks++;
    st.rejected    += !ok_ref;
    st.is_2d       += is_2d;
    st.multi_block += n_blk > 1;
    st.cells_cmp   += (uint64_t) n_kv*n_blk;
    st.bytes_cmp   += (n_cells_i32 + n_toks_i32)*sizeof(int32_t);

    if (ok_new != ok_ref ||
            memcmp(c_new.data(), c_ref.data(), n_cells_i32*sizeof(int32_t)) != 0 ||
            memcmp(t_new.data(), t_ref.data(), n_toks_i32*sizeof(int32_t)) != 0) {
        if (st.diff < 10) {
            size_t first = SIZE_MAX;
            for (size_t k = 0; k < n_cells_i32 && first == SIZE_MAX; ++k) {
                if (c_new[k] != c_ref[k]) first = k;
            }
            fprintf(stderr, "fill mismatch: ok %d/%d n_kv %" PRId64 " n_blk %" PRId64 " n_tps %" PRId64 " 2d %d first cell word %zu\n",
                    ok_new, ok_ref, n_kv, n_blk, n_tps, (int) is_2d, first);
        }
        st.diff++;
    }

    // negative control 2 through the same comparison: the mutant must differ only in the x / y planes of empty cells
    if (ok_ref) {
        const bool ok_mut = fill_mutant_nomask(c_mut.data(), t_mut.data(), n_kv, n_blk, &b.ub, v, seq_to_stream);
        st.mut_runs++;
        bool detected = ok_mut != ok_ref || memcmp(t_mut.data(), t_ref.data(), n_toks_i32*sizeof(int32_t)) != 0;
        for (size_t k = 0; k < n_cells_i32; ++k) {
            if (c_mut[k] == c_ref[k]) {
                continue;
            }
            detected = true;
            st.mut_words++;
            const int64_t blk = (int64_t) k/(4*n_kv), plane = ((int64_t) k/n_kv)%4, j = (int64_t) k%n_kv;
            const bool empty_xy = (plane == 2 || plane == 3) && c_ref[(size_t) (4*blk*n_kv + j)] == -1;
            st.mut_misplaced += !empty_xy;
        }
        st.mut_detected += detected;
    }
}

// negative control: one flipped visible seq bit in a used cell must make the two fills differ
void control(rng_t & rng, stats_t & st) {
    std::vector<llama_kv_cells> a(1), bcells(1);
    a[0].resize(1024);
    for (uint32_t i = 0; i < 1024; ++i) {
        op_apply(a[0], i, rng, 4);
    }
    bcells[0] = a[0];
    uint32_t i = (uint32_t) rng.u(1024);
    llama_seq_id q = (llama_seq_id) rng.u(32);
    while (bcells[0].seq_has(i, q)) {
        q = (llama_seq_id) rng.u(32);
    }
    bcells[0].seq_add(i, q);

    std::vector<uint32_t> seq_to_stream(LLAMA_MAX_SEQ, 0);
    ubatch_buf b;
    make_ubatch(b, rng, 1, 1, seq_to_stream, 1, false, 4);
    b.seq_data[0] = 0;

    std::vector<int32_t> c1(4*1024, 7), c2(4*1024, 7), t1(4, 7), t2(4, 7);
    llama_kv_cells_fill_kq_vis    (c1.data(), t1.data(), 1024, 1, &b.ub, bcells, seq_to_stream);
    llama_kv_cells_fill_kq_vis_ref(c2.data(), t2.data(), 1024, 1, &b.ub, a,      seq_to_stream);
    st.ctrl_runs++;
    st.ctrl_detected += memcmp(c1.data(), c2.data(), c1.size()*sizeof(int32_t)) != 0;
}

} // namespace

int main(int argc, char ** argv) {
#ifdef NDEBUG
    const bool capped = argc > 3 && std::string(argv[3]) == "capped";
#else
    const bool capped = !(argc > 3 && std::string(argv[3]) == "full");  // Debug: asserts + no optimization
#endif
    const int      n_rounds = argc > 1 ? atoi(argv[1]) : (capped ? 100 : 300);
    const uint64_t seed     = argc > 2 ? strtoull(argv[2], nullptr, 10) : 42;

    rng_t   rng(seed);
    stats_t st;

    for (int round = 0; round < n_rounds; ++round) {
        const uint32_t ns   = 1 + (uint32_t) rng.u(4);
        // mostly small pools (many ops per cell), some large ones (the prod-sized planes)
        const uint32_t size = round % 100 == 99 && !capped ? 131072 + (uint32_t) rng.u(262144) : 16 + (uint32_t) rng.u(rng.p(0.7) ? 512 : 8192);
        const int n_seq_hint = 1 + (int) rng.u(rng.p(0.7) ? 5 : 32);

        std::vector<uint32_t> seq_to_stream(LLAMA_MAX_SEQ);
        for (uint32_t q = 0; q < LLAMA_MAX_SEQ; ++q) {
            seq_to_stream[q] = ns == 1 ? 0 : (rng.p(0.7) ? q % ns : (uint32_t) rng.u(ns));
        }

        std::vector<llama_kv_cells> v(ns);
        for (auto & c : v) {
            c.resize(size);
        }
        // start from a filled pool: contiguous runs per seq, like a warm cache
        for (uint32_t s = 0; s < ns; ++s) {
            const uint32_t n_fill = (uint32_t) (size*(0.3 + 0.7*rng.p(0.5)));
            for (uint32_t j = 0; j < n_fill; ++j) {
                op_apply(v[s], j, rng, n_seq_hint);
            }
        }

        const int n_steps = size > 65536 ? 4 : 24;
        const int n_ops   = size > 65536 ? 200 : 40;
        for (int step = 0; step < n_steps; ++step) {
            for (int k = 0; k < n_ops; ++k) {
                random_op(v, rng, n_seq_hint, size > 65536, st);
            }
            check(v, rng, seq_to_stream, n_seq_hint, st);
        }
        if (round % 25 == 0) {
            control(rng, st);
        }
    }

    printf("kq-vis-fill-equiv: seed %" PRIu64 ", %d rounds%s, %" PRIu64 " ops, %" PRIu64 " checks, %" PRIu64 " cell records "
           "(%.1f MiB) compared\n", seed, n_rounds, capped ? " (capped: no prod-sized pools)" : "", st.ops, st.checks, st.cells_cmp,
           st.bytes_cmp/(1024.0*1024.0));
    printf("  coverage: cells empty %" PRIu64 ", seqless used %" PRIu64 ", multi-seq %" PRIu64 ", only seq ids >= 32 %" PRIu64
           ", emptied by a shift (ext kept) %" PRIu64 "; ubatches 2-D %" PRIu64 ", multi-block %" PRIu64 ", rejected %" PRIu64 "\n",
           st.cells_empty, st.cells_seqless, st.cells_multi, st.cells_hi_only, st.emptied_by_shift, st.is_2d, st.multi_block,
           st.rejected);
    printf("  ops:");
    for (int k = 0; k < 16; ++k) {
        printf(" %" PRIu64, st.op_count[k]);
    }
    printf("\n");
    printf("  differing fills %" PRIu64 ", shadow mismatches %" PRIu64 ", negative control (flipped seq bit) %" PRIu64 "/%" PRIu64 " detected\n",
           st.diff, st.shadow_diff, st.ctrl_detected, st.ctrl_runs);
    printf("  mutant fill (no empty-cell mask): detected in %" PRIu64 "/%" PRIu64 " checks, %" PRIu64 " differing words, "
           "%" PRIu64 " outside the x / y planes of empty cells\n", st.mut_detected, st.mut_runs, st.mut_words, st.mut_misplaced);
    const int trip = tripwire();

    const bool pass = st.diff == 0 && st.shadow_diff == 0 && st.ctrl_detected == st.ctrl_runs && st.ctrl_runs > 0 &&
        st.rejected > 0 && st.cells_seqless > 0 && st.cells_hi_only > 0 && st.is_2d > 0 && st.emptied_by_shift > 0 &&
        st.mut_detected > 0 && st.mut_misplaced == 0 && trip == 0;
    printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
