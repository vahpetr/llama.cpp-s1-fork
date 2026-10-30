// [TAG_T2_SEQ_RM_COUNTED] differential test: llama_kv_cells::seq_rm_counted (+ seq_pos_count_in), the T2 step-1
// replacement of the per-sequence full scan in llama_kv_cache::seq_rm, against that full scan (copied below verbatim
// from llama-kv-cache.cpp), on randomized cell layouts: holes, multi-sequence cells, sequence ids >= 32, repeated
// positions of one sequence, bounded and open ranges, op sequences that evolve the state. After every op the whole
// observable state must be equal: per cell pos / seq set / seq_lo-backed mask, used count / min / max, per sequence
// seq_pos min / max / count, the returned head candidate and the KQ-vis dirty range of a registered mirror slot.
// A negative control (a mutant that removes one cell too few) must be caught.
//
// usage: test-kv-seq-rm-counted [n_layouts = 2000] [seed = 42]
//        test-kv-seq-rm-counted --perf [reps = 200]   (1M cells, decode-round tails at depths 32k / 305k / 678k)

#include "../src/llama-kv-cells.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

// the per-sequence branch of llama_kv_cache::seq_rm before T2 (the reference)
static uint32_t seq_rm_scan(llama_kv_cells & cells, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    return new_head;
}

// the T2 path as llama_kv_cache::seq_rm runs it (max_keys = the cache's constant, or a test value)
static uint32_t seq_rm_t2(llama_kv_cells & cells, llama_seq_id seq_id, llama_pos p0, llama_pos p1, uint32_t max_keys,
                          bool mutant, bool & counted) {
    uint32_t n = 0;
    counted = cells.seq_pos_count_in(seq_id, p0, p1, max_keys, n);
    if (!counted) {
        return seq_rm_scan(cells, seq_id, p0, p1);
    }
    if (mutant && n > 0) {
        n -= 1; // negative control: one cell too few
    }
    return cells.seq_rm_counted(seq_id, p0, p1, n);
}

static int n_vis_slot_a = -1;
static int n_vis_slot_b = -1;

static std::string state_diff(const llama_kv_cells & a, const llama_kv_cells & b) {
    if (a.size() != b.size()) {
        return "size";
    }
    for (uint32_t i = 0; i < a.size(); ++i) {
        if (a.is_empty(i) != b.is_empty(i)) {
            return "empty@" + std::to_string(i);
        }
        if (a.is_empty(i)) {
            continue;
        }
        if (a.pos_get(i) != b.pos_get(i)) {
            return "pos@" + std::to_string(i);
        }
        if (a.seq_set(i) != b.seq_set(i)) {
            return "seq@" + std::to_string(i);
        }
        if (a.seq_mask32(i) != b.seq_mask32(i)) {
            return "seq_lo@" + std::to_string(i);
        }
    }
    if (a.get_used() != b.get_used() || a.used_min() != b.used_min() || a.used_max_p1() != b.used_max_p1()) {
        return "used";
    }
    for (llama_seq_id s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (a.seq_pos_min(s) != b.seq_pos_min(s) || a.seq_pos_max(s) != b.seq_pos_max(s) ||
            a.seq_pos_count(s) != b.seq_pos_count(s)) {
            return "seq_pos@" + std::to_string(s);
        }
    }
    uint32_t alo, ahi, blo, bhi;
    bool aall, ball;
    a.vis_take(n_vis_slot_a, alo, ahi, aall);
    b.vis_take(n_vis_slot_b, blo, bhi, ball);
    if (alo != blo || ahi != bhi || aall != ball) {
        return "vis_dirty";
    }
    return "";
}

static void fill_random(llama_kv_cells & c, std::mt19937 & rng, uint32_t n, llama_pos p_max) {
    c.resize(n);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double occ = 0.2 + 0.75*u(rng);
    const bool   contiguous = u(rng) < 0.5; // a decoding layout: cell i holds position i of seq 0
    for (uint32_t i = 0; i < n; ++i) {
        if (u(rng) > occ) {
            continue; // hole
        }
        const llama_pos p = contiguous ? (llama_pos) i : (llama_pos) (rng() % (uint32_t) (p_max + 1));
        c.pos_set(i, p);
        const int ns = contiguous ? 1 : 1 + (int) (rng() % 3);
        for (int k = 0; k < ns; ++k) {
            llama_seq_id s = u(rng) < 0.9 ? (llama_seq_id) (rng() % 6) : (llama_seq_id) (32 + rng() % 8);
            if (contiguous) {
                s = 0;
            }
            if (!c.seq_has(i, s)) {
                c.seq_add(i, s);
            }
        }
    }
}

static int run_fuzz(int n_layouts, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    uint64_t n_ops = 0, n_counted = 0, n_fallback = 0, n_mutant_caught = 0, n_mutant_ops = 0;

    for (int l = 0; l < n_layouts; ++l) {
        static const uint32_t sizes[4] = { 16, 100, 1000, 4096 };
        const uint32_t n     = sizes[rng() % 4];
        const llama_pos pmax = (llama_pos) (u(rng) < 0.5 ? n : 4*n);

        llama_kv_cells a;
        fill_random(a, rng, n, pmax);
        llama_kv_cells b = a;   // independent copy (vectors, maps, bitsets)
        llama_kv_cells m = a;   // mutant
        n_vis_slot_a = a.vis_register();
        n_vis_slot_b = b.vis_register();
        const int n_vis_slot_m = m.vis_register();
        {
            uint32_t lo, hi; bool all;
            a.vis_take(n_vis_slot_a, lo, hi, all);
            b.vis_take(n_vis_slot_b, lo, hi, all);
            m.vis_take(n_vis_slot_m, lo, hi, all);
        }

        bool mutant_diverged = false;

        for (int op = 0; op < 24; ++op) {
            const llama_seq_id s = u(rng) < 0.85 ? (llama_seq_id) (rng() % 6) : (llama_seq_id) (32 + rng() % 8);
            llama_pos p0 = (llama_pos) (rng() % (uint32_t) (pmax + 2));
            llama_pos p1;
            const double r = u(rng);
            if (r < 0.5) {
                p1 = std::numeric_limits<llama_pos>::max();   // seq_rm(seq, p0, -1): the decode-round tail
                if (u(rng) < 0.5 && a.seq_pos_max(s) >= 0) {
                    p0 = std::max(0, a.seq_pos_max(s) - (llama_pos) (rng() % 7)); // a speculative tail (<= 6 positions)
                }
            } else if (r < 0.8) {
                p1 = p0 + 1 + (llama_pos) (rng() % 8);
            } else {
                p1 = p0 + (llama_pos) (rng() % (uint32_t) (pmax + 1));
            }
            const uint32_t max_keys = u(rng) < 0.8 ? (1u << 30) : (uint32_t) (rng() % 4);

            const uint32_t ha = seq_rm_scan(a, s, p0, p1);
            bool counted = false;
            const uint32_t hb = seq_rm_t2(b, s, p0, p1, max_keys, false, counted);
            bool counted_m = false;
            const uint32_t hm = seq_rm_t2(m, s, p0, p1, max_keys, true, counted_m);
            n_ops++;
            n_counted  += counted ? 1 : 0;
            n_fallback += counted ? 0 : 1;

            const std::string d = state_diff(a, b);
            if (ha != hb || !d.empty()) {
                fprintf(stderr, "FAIL layout %d op %d: seq %d [%d, %d) max_keys %u counted %d: head %u vs %u, diff '%s'\n",
                        l, op, s, p0, p1, max_keys, counted ? 1 : 0, ha, hb, d.c_str());
                return 1;
            }
            if (!mutant_diverged) {
                uint32_t lo, hi; bool all;
                // compare the mutant against the reference without consuming a's dirty range twice
                bool same = hm == ha && m.get_used() == a.get_used();
                for (llama_seq_id q = 0; q < LLAMA_MAX_SEQ && same; ++q) {
                    same = m.seq_pos_count(q) == a.seq_pos_count(q);
                }
                m.vis_take(n_vis_slot_m, lo, hi, all);
                n_mutant_ops += counted_m ? 1 : 0;
                if (!same) {
                    mutant_diverged = true;
                    n_mutant_caught++;
                }
            }

            // re-grow: put a few cells back so later ops have something to remove
            if (u(rng) < 0.5) {
                for (int k = 0; k < 4; ++k) {
                    const uint32_t i = rng() % n;
                    if (a.is_empty(i)) {
                        const llama_pos p = (llama_pos) (rng() % (uint32_t) (pmax + 1));
                        a.pos_set(i, p); a.seq_add(i, s);
                        b.pos_set(i, p); b.seq_add(i, s);
                        if (m.is_empty(i)) { m.pos_set(i, p); m.seq_add(i, s); }
                    }
                }
                const std::string d2 = state_diff(a, b);
                if (!d2.empty()) {
                    fprintf(stderr, "FAIL layout %d op %d: diverged after re-grow: '%s'\n", l, op, d2.c_str());
                    return 1;
                }
            }
        }
    }

    printf("seq_rm counted vs full scan: %" PRIu64 " ops identical (%" PRIu64 " counted, %" PRIu64 " fallback) over %d layouts\n",
           n_ops, n_counted, n_fallback, n_layouts);
    printf("negative control (one cell too few): caught in %" PRIu64 " of %d layouts (%" PRIu64 " counted mutant ops)\n",
           n_mutant_caught, n_layouts, n_mutant_ops);
    if (n_mutant_caught == 0) {
        fprintf(stderr, "FAIL: the negative control was never caught\n");
        return 1;
    }
    if (n_counted == 0 || n_fallback == 0) {
        fprintf(stderr, "FAIL: a path was never exercised (counted %" PRIu64 ", fallback %" PRIu64 ")\n", n_counted, n_fallback);
        return 1;
    }
    printf("PASS\n");
    return 0;
}

// one speculative round on a decoding sequence: remove the rejected tail [d - k, inf) then write it back (the next verify)
static int run_perf(int reps) {
    const uint32_t n = 1u << 20;
    for (const llama_pos d : { 32768, 305279, 678257 }) {
        llama_kv_cells c;
        c.resize(n);
        for (llama_pos p = 0; p < d; ++p) {
            c.pos_set((uint32_t) p, p);
            c.seq_add((uint32_t) p, 0);
        }
        double t_scan = 0.0, t_cnt = 0.0, t_scan0 = 0.0, t_cnt0 = 0.0;
        for (int r = 0; r < reps; ++r) {
            for (int arm = 0; arm < 2; ++arm) {
                const llama_pos p0 = d - 3;
                auto t0 = std::chrono::steady_clock::now();
                if (arm == 0) {
                    seq_rm_scan(c, 0, p0, std::numeric_limits<llama_pos>::max());
                } else {
                    uint32_t k = 0;
                    if (c.seq_pos_count_in(0, p0, std::numeric_limits<llama_pos>::max(), 256, k)) {
                        c.seq_rm_counted(0, p0, std::numeric_limits<llama_pos>::max(), k);
                    }
                }
                auto t1 = std::chrono::steady_clock::now();
                // nothing to remove (a fully accepted round)
                if (arm == 0) {
                    seq_rm_scan(c, 0, d + 10, std::numeric_limits<llama_pos>::max());
                } else {
                    uint32_t k = 0;
                    if (c.seq_pos_count_in(0, d + 10, std::numeric_limits<llama_pos>::max(), 256, k)) {
                        c.seq_rm_counted(0, d + 10, std::numeric_limits<llama_pos>::max(), k);
                    }
                }
                auto t2 = std::chrono::steady_clock::now();
                (arm == 0 ? t_scan : t_cnt)   += std::chrono::duration<double, std::micro>(t1 - t0).count();
                (arm == 0 ? t_scan0 : t_cnt0) += std::chrono::duration<double, std::micro>(t2 - t1).count();
                for (llama_pos p = p0; p < d; ++p) {
                    c.pos_set((uint32_t) p, p);
                    c.seq_add((uint32_t) p, 0);
                }
            }
        }
        printf("perf 1M cells, depth %7d: tail of 3  scan %8.1f us  counted %6.2f us | nothing to remove  scan %8.1f us  counted %6.2f us\n",
               d, t_scan/reps, t_cnt/reps, t_scan0/reps, t_cnt0/reps);
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        return run_perf(argc > 2 ? atoi(argv[2]) : 200);
    }
    const int      n_layouts = argc > 1 ? atoi(argv[1]) : 2000;
    const uint32_t seed      = argc > 2 ? (uint32_t) atoi(argv[2]) : 42;
    return run_fuzz(n_layouts, seed);
}
