// [TAG_SPEC_DRAFT_POLICY] --spec-draft-policy tput (v51): the throughput-max draft length as a pure function
// (acceptance p_i, round costs -> n*) and the cooldown state machine around true-zero rounds.

#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int n_fail = 0;
static int n_ok   = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        printf("FAIL %s\n", what.c_str());
        n_fail++;
    } else {
        n_ok++;
    }
}

// brute force reference: enumerate the outcomes of a round (a accepted of n, probability P_a*(1 - p_{a+1}) or P_n),
// each with its committed tokens and wall time (+ the restore/replay cycle when the rollback n - a needs a restore);
// argmax of E[tokens]/E[time] with ties to the shorter draft
static double ref_rate(const std::vector<float> & p, int n, const common_spec_tput_cost & c) {
    const auto pj = [&](int j) { return j - 1 < (int) p.size() ? (double) p[j - 1] : 1.0; };
    const double m    = n + 1.0;
    const double base = (n ? c.d0 + c.d1*n : 0.0) + c.v0 + c.v1*m + c.c0 + c.c1*m + c.host;
    double e_tok = 0.0, e_t = 0.0;
    for (int a = 0; a <= n; ++a) {
        double pr = 1.0;
        for (int j = 1; j <= a; ++j) {
            pr *= pj(j);
        }
        if (a < n) {
            pr *= 1.0 - pj(a + 1);
        }
        const bool rb = a < n && n - a > c.n_rb_free;
        e_tok += pr*(a + 1 + (rb ? 1 : 0));
        e_t   += pr*(base + (rb ? c.r0 + c.r1*(a + 2) : 0.0));
    }
    return e_tok / e_t;
}

static int ref_best(const std::vector<float> & p, int n_max, const common_spec_tput_cost & c) {
    int best = 0;
    double best_r = -1.0;
    for (int n = 0; n <= n_max; ++n) {
        const double r = ref_rate(p, n, c);
        if (r > best_r + 1e-12) {
            best_r = r;
            best   = n;
        }
    }
    return best;
}

static void test_rule() {
    // s1-like costs at depth (ms): draft call 1.0 + 2.2/token, verify 25 + 1.5/row, catch-up 0.8 + 0.2/row, host 1.2
    common_spec_tput_cost c;
    c.d0 = 1.0; c.d1 = 2.2; c.v0 = 25.0; c.v1 = 1.5; c.c0 = 0.8; c.c1 = 0.2; c.host = 1.2;

    check(common_spec_tput_best_n({ 0.95f, 0.95f, 0.95f, 0.95f, 0.95f }, 5, c) == 5, "easy text drafts n_max");
    check(common_spec_tput_best_n({ 0.0f, 0.0f, 0.0f }, 3, c) == 0, "hopeless draft -> true zero");
    check(common_spec_tput_best_n({ 0.02f, 0.5f, 0.5f }, 3, c) == 0, "p1 = 0.02 does not pay the draft call -> 0");
    check(common_spec_tput_best_n({ 0.9f, 0.1f, 0.9f, 0.9f }, 4, c) == 1, "a cliff at position 2 -> 1");
    check(common_spec_tput_best_n({ 0.9f, 0.9f }, 5, c) == 5, "missing p entries count as 1");
    check(common_spec_tput_best_n({ 0.9f, 0.9f }, 0, c) == 0, "n_max 0 -> 0");
    check(common_spec_tput_best_n({}, 3, c) == 3, "no estimate yet (empty EMA = optimistic) -> n_max");

    // break-even of a one-token draft: (1 + p) / (T0 + d0 + d1 + v1 + c1) = 1 / T0, T0 = v0 + v1 + c0 + c1 + host
    const double T0 = c.v0 + c.v1 + c.c0 + c.c1 + c.host;
    const double p_be = (T0 + c.d0 + c.d1 + c.v1 + c.c1) / T0 - 1.0;
    check(common_spec_tput_best_n({ (float) (p_be - 0.01) }, 1, c) == 0, "just below the 1-token break-even -> 0");
    check(common_spec_tput_best_n({ (float) (p_be + 0.01) }, 1, c) == 1, "just above the 1-token break-even -> 1");

    double rate = 0.0;
    const int n = common_spec_tput_best_n({ 0.8f, 0.7f, 0.6f }, 3, c, &rate);
    check(rate > 0.0 && std::isfinite(rate), "rate_out is set");
    printf("rule: p = 0.8/0.7/0.6 -> n* = %d (%.4f tok/ms), 1-token break-even p = %.3f\n", n, rate, p_be);

    // a sweep against the brute-force reference
    int n_cmp = 0, hist[6] = { 0 };
    for (int a = 0; a <= 20; ++a) {
        for (int b = 0; b <= 10; ++b) {
            for (double v0 : { 5.0, 25.0, 60.0 }) {
                const std::vector<float> p = { a / 20.0f, b / 10.0f, 0.7f, 0.3f, 0.9f };
                common_spec_tput_cost cc = c;
                cc.v0 = v0;
                const int got = common_spec_tput_best_n(p, 5, cc);
                check(got == ref_best(p, 5, cc), "sweep vs reference");
                hist[got]++;
                n_cmp++;
            }
        }
    }
    printf("rule sweep: %d cases match the brute force; n* distribution 0:%d 1:%d 2:%d 3:%d 4:%d 5:%d\n",
           n_cmp, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
}

// (v51 r1, C Major) the complete verification/restore/replay cycle: with a FULL target every rejection restores the
// checkpoint and replays a+2 rows in the next pass; that cost (and the replay's extra token) moves n*
static void test_rollback_cycle() {
    common_spec_tput_cost c;
    c.d0 = 1.0; c.d1 = 2.2; c.v0 = 25.0; c.v1 = 1.5; c.c0 = 0.8; c.c1 = 0.2; c.host = 1.2;
    const std::vector<float> p = { 0.6f, 0.6f, 0.6f, 0.6f, 0.6f };

    const int n_free = common_spec_tput_best_n(p, 5, c); // no restore (partial seq_rm)
    common_spec_tput_cost f = c;
    f.n_rb_free = 0;                                      // FULL: every rejection restores
    f.r0 = 30.0; f.r1 = 1.7;                              // restore 2 ms + a replay pass of a+2 rows
    const int n_full = common_spec_tput_best_n(p, 5, f);
    printf("rollback cycle: p = 0.6 x5, no restore n* = %d, FULL restore+replay n* = %d\n", n_free, n_full);
    check(n_full < n_free, "the rollback cycle cost lowers n* at low P_i");
    check(n_full == ref_best(p, 5, f), "FULL rollback: n* matches the outcome enumeration");

    // the term is exact against the enumeration: rate(n) for every n, FULL and RS (restore only beyond n_rs_seq = 2)
    for (int free : { 0, 2, 1 << 30 }) {
        common_spec_tput_cost g = f;
        g.n_rb_free = free;
        for (int n = 0; n <= 5; ++n) {
            double gain = 0.0, cost = 0.0;
            common_spec_tput_eval(p, n, g, gain, cost);
            check(std::fabs(gain/cost - ref_rate(p, n, g)) < 1e-12, "eval == enumeration (free " + std::to_string(free) +
                  ", n " + std::to_string(n) + ")");
        }
    }
    // a restore-free rollback shallower than n_rs_seq costs nothing extra: RS with free >= n_max == no restore
    common_spec_tput_cost rs = f;
    rs.n_rb_free = 5;
    check(common_spec_tput_best_n(p, 5, rs) == n_free, "RS rollback within n_rs_seq: no cycle cost");
    // a cheap rollback (replay pass ~ free) with the replay's extra token does not shorten the draft
    common_spec_tput_cost cheap = f;
    cheap.r0 = 0.0; cheap.r1 = 0.0;
    check(common_spec_tput_best_n(p, 5, cheap) >= n_free, "a free replay (extra token) never shortens the draft");
    // sweep vs the enumeration, FULL
    for (int a = 0; a <= 10; ++a) {
        for (double r0 : { 5.0, 30.0, 80.0 }) {
            const std::vector<float> q = { a / 10.0f, 0.8f, 0.5f, 0.9f };
            common_spec_tput_cost h = f;
            h.r0 = r0;
            check(common_spec_tput_best_n(q, 4, h) == ref_best(q, 4, h), "FULL sweep vs enumeration");
        }
    }
}

// (v51 r1, D Major) the fits are per KV-depth bucket: shallow rounds never price a deep slot
static void test_depth_buckets() {
    check(common_spec_tput_bucket(0) == 0 && common_spec_tput_bucket(8191) == 0, "bucket 0 below 8k");
    check(common_spec_tput_bucket(8192) == 1 && common_spec_tput_bucket(16383) == 1, "bucket 1 = [8k, 16k)");
    check(common_spec_tput_bucket(107000) == 4, "107k -> bucket 4");
    check(common_spec_tput_bucket(524288) == 7 && common_spec_tput_bucket(524287) == 6, "524k -> bucket 7");
    check(common_spec_tput_bucket(1048576) == 8, "1M -> bucket 8");
    check(common_spec_tput_bucket(1 << 30) == COMMON_SPEC_TPUT_N_BUCKETS - 1, "deeper -> the last bucket");

    common_spec_tput_model m;
    // 200 shallow rounds (2k): verify 10 + 1/row
    for (int r = 0; r < 200; ++r) {
        m.at(2048).verify.add(1 + r % 5, 10.0 + 1.0*(1 + r % 5));
    }
    double a = 0.0, b = 0.0;
    check(!m.at(524288).verify.get(a, b, 0.0), "a deep bucket without its own samples is not warm (prior path)");
    // 20 deep rounds (520k): verify 40 + 3/row; 200 more shallow rounds afterwards do not move it
    for (int r = 0; r < 20; ++r) {
        m.at(520000).verify.add(1 + r % 5, 40.0 + 3.0*(1 + r % 5));
    }
    for (int r = 0; r < 200; ++r) {
        m.at(4096).verify.add(1 + r % 5, 10.0 + 1.0*(1 + r % 5));
    }
    check(m.at(524000).verify.get(a, b, 0.0) && std::fabs(a - 40.0) < 1e-6 && std::fabs(b - 3.0) < 1e-6,
          "the deep bucket's fit is its own rounds only (a " + std::to_string(a) + ", b " + std::to_string(b) + ")");
    check(m.at(100).verify.get(a, b, 0.0) && std::fabs(a - 10.0) < 1e-6 && std::fabs(b - 1.0) < 1e-6,
          "the shallow bucket's fit is untouched by deep rounds");
}

// (v51 r1, D Major) the checkpoint save/restore is charged to drafting candidates only, not as a common intercept
static void test_ckpt_cost() {
    common_spec_tput_cost c;
    c.d0 = 1.0; c.d1 = 2.2; c.v0 = 25.0; c.v1 = 1.5; c.c0 = 0.8; c.c1 = 0.2; c.host = 1.2;
    common_spec_tput_cost k = c;
    k.k0 = 6.0; k.k1 = 0.5;
    double g0 = 0.0, t0 = 0.0, g1 = 0.0, t1 = 0.0;
    common_spec_tput_eval({ 0.5f }, 0, c, g0, t0);
    common_spec_tput_eval({ 0.5f }, 0, k, g1, t1);
    check(t0 == t1, "n = 0 pays no checkpoint");
    for (int n = 1; n <= 4; ++n) {
        common_spec_tput_eval({ 0.5f, 0.5f, 0.5f, 0.5f }, n, c, g0, t0);
        common_spec_tput_eval({ 0.5f, 0.5f, 0.5f, 0.5f }, n, k, g1, t1);
        check(std::fabs(t1 - t0 - (6.0 + 0.5*n)) < 1e-12, "n > 0 pays k0 + k1*n");
    }
    // the same 6.5 ms as a common host intercept (the old pooling) keeps drafting; charged to n > 0 it flips to 0
    const std::vector<float> p = { 0.3f, 0.5f, 0.5f };
    common_spec_tput_cost pooled = c;
    pooled.host += 6.5;
    const int n_pooled = common_spec_tput_best_n(p, 3, pooled);
    const int n_sep    = common_spec_tput_best_n(p, 3, k);
    printf("ckpt cost: p = 0.3/0.5/0.5, pooled into host n* = %d, charged to n > 0 n* = %d\n", n_pooled, n_sep);
    check(n_pooled > 0 && n_sep == 0, "a checkpoint cost the draft cannot pay -> true zero");
}

// (v51 r1, D Minors) timing hygiene: non-finite / negative samples are rejected, the warm-up is 8 samples, a cost model
// with a non-finite or non-positive candidate cost is invalid (-1: the caller's prior path)
static void test_timing_hygiene() {
    common_spec_ew_line l;
    double a = 0.0, b = 0.0;
    for (int k = 0; k < 7; ++k) {
        l.add(1 + k % 3, 10.0 + (1 + k % 3));
    }
    check(!l.get(a, b, 0.0), "7 samples: not warm");
    l.add(NAN, 5.0);
    l.add(2.0, INFINITY);
    l.add(2.0, -1.0);
    l.add(2.0, NAN);
    check(l.n_rejected == 4 && l.n_samples == 7 && !l.get(a, b, 0.0), "non-finite / negative samples rejected");
    l.add(2.0, 12.0);
    check(l.get(a, b, 0.0) && std::fabs(a - 10.0) < 1e-9 && std::fabs(b - 1.0) < 1e-9, "the 8th sample warms the fit");

    common_spec_tput_cost c;
    c.d0 = 1.0; c.d1 = 2.2; c.v0 = 25.0; c.v1 = 1.5; c.c0 = 0.8; c.c1 = 0.2; c.host = 1.2;
    double rate = 1.0;
    common_spec_tput_cost z = c;
    z.v0 = -27.0; // n = 0 costs -27 + 1.5 + 0.8 + 0.2 + 1.2 < 0
    check(common_spec_tput_best_n({ 0.9f }, 2, z, &rate) == -1 && rate == 0.0, "a non-positive cost -> invalid (-1)");
    common_spec_tput_cost n = c;
    n.d1 = NAN;
    check(common_spec_tput_best_n({ 0.9f }, 2, n) == -1, "a non-finite cost -> invalid (-1)");
    common_spec_tput_cost i = c;
    i.r0 = INFINITY; i.n_rb_free = 0;
    check(common_spec_tput_best_n({ 0.9f }, 2, i) == -1, "an infinite rollback cost -> invalid (-1)");
    check(common_spec_tput_best_n({ 0.9f }, 2, c) >= 0, "a valid model -> n*");
}

// (v51 r1, C Minors) {0} U [n_min, n_max] compared directly; never-reached positions take the last observed estimate
static void test_rule_hygiene() {
    common_spec_tput_cost c;
    c.d0 = 1.0; c.d1 = 2.2; c.v0 = 25.0; c.v1 = 1.5; c.c0 = 0.8; c.c1 = 0.2; c.host = 1.2;
    const std::vector<float> p = { 0.3f, 0.05f, 0.05f, 0.05f };
    check(common_spec_tput_best_n(p, 4, c) == 1, "unconstrained: n* = 1");
    // n_min = 3: the old rule promoted 1 -> 3 after the choice; the direct comparison finds 0 better than 3
    const int n3 = common_spec_tput_best_n(p, 4, c, nullptr, 3);
    int ref = 0;
    double best = ref_rate(p, 0, c);
    for (int n = 3; n <= 4; ++n) {
        if (ref_rate(p, n, c) > best + 1e-12) {
            best = ref_rate(p, n, c);
            ref  = n;
        }
    }
    check(n3 == 0 && ref == 0, "n_min 3: {0} U [3, 4] -> 0 (not a promoted 3)");
    check(common_spec_tput_best_n({ 0.95f, 0.95f, 0.95f, 0.95f }, 4, c, nullptr, 3) == 4, "n_min 3, easy text -> n_max");
    check(common_spec_tput_best_n({ 0.95f }, 2, c, nullptr, 3) == 0, "n_min above n_max: only 0 is a candidate");
    for (int a = 0; a <= 10; ++a) {
        for (int n_min = 1; n_min <= 5; ++n_min) {
            const std::vector<float> q = { a / 10.0f, 0.7f, 0.4f, 0.8f, 0.6f };
            const int got = common_spec_tput_best_n(q, 5, c, nullptr, n_min);
            int r = 0;
            double rb = ref_rate(q, 0, c);
            for (int n = n_min; n <= 5; ++n) {
                if (ref_rate(q, n, c) > rb + 1e-12) {
                    rb = ref_rate(q, n, c);
                    r  = n;
                }
            }
            check(got == r, "n_min sweep vs the restricted enumeration");
        }
    }

    // p_eff: an unobserved position takes the deepest observed estimate before it, 1.0 only with no observation
    const auto e1 = common_spec_tput_p_eff({ 0.8f, 0.5f, 1.0f, 1.0f }, { 10, 3, 0, 0 });
    check(e1 == std::vector<float>({ 0.8f, 0.5f, 0.5f, 0.5f }), "never-reached tail -> the last observed estimate");
    const auto e2 = common_spec_tput_p_eff({ 0.7f, 1.0f, 0.4f }, { 5, 0, 2 });
    check(e2 == std::vector<float>({ 0.7f, 0.7f, 0.4f }), "a gap takes the estimate before it");
    const auto e3 = common_spec_tput_p_eff({ 1.0f, 1.0f }, { 0, 0 });
    check(e3 == std::vector<float>({ 1.0f, 1.0f }), "nothing observed -> 1.0");
    // the optimistic tail drafted n_max; the conservative one stops earlier
    const std::vector<float> ema = { 0.6f, 0.6f, 1.0f, 1.0f, 1.0f };
    const int n_opt  = common_spec_tput_best_n(ema, 5, c);
    const int n_cons = common_spec_tput_best_n(common_spec_tput_p_eff(ema, { 50, 20, 0, 0, 0 }), 5, c);
    printf("rule hygiene: ema 0.6/0.6 + unobserved tail: optimistic 1.0 n* = %d, last-observed n* = %d\n", n_opt, n_cons);
    check(n_cons < n_opt, "the conservative tail prior shortens the draft");
}

// v51 r2 (Codex sol §4 Major): identifiability. Fits whose samples sit at one attempted length / row count have no rank
//   (8 samples are a warm-up, not a rank): the policy is `off` there, with an exploration round every 16th drafting
//   round at max(n_lo, n_max - 2) until both the draft and the verify fit span >= 2 lengths with >= 4 samples each.
static void test_rank() {
    common_spec_ew_line l;
    for (int k = 0; k < 20; ++k) {
        l.add(8.0, 4.0);
    }
    check(l.n_distinct(4) == 1, "one x: one distinct length");
    double a = 0.0, b = 0.0;
    check(l.get(a, b, 0.25), "one x: warm (8+ samples) - the warm-up alone is not the rank");
    l.add(6.0, 4.0); l.add(6.0, 4.0); l.add(6.0, 4.0);
    check(l.n_distinct(4) == 1, "3 samples at a second x: not yet");
    l.add(6.0, 4.0);
    check(l.n_distinct(4) == 2, "4 samples at a second x: 2 distinct");
    l.add(1e9, 1.0);
    check(l.n_at[common_spec_ew_line::N_X - 1] == 1, "x clamped into the last count");

    // the sol counterexample's samples: constant-width rounds n = 8 (draft 4 ms, verify 9 ms of 9 rows) -> no rank
    common_spec_tput_fits f;
    for (int k = 0; k < 64; ++k) {
        f.draft.add(8.0, 4.0); f.ckpt.add(8.0, 0.0); f.verify.add(9.0, 9.0); f.catchup.add(9.0, 0.0); f.host_d.add(9.0, 0.0);
    }
    check(!f.ranked(), "constant-width samples: no rank -> the policy plays n_max (off)");

    // the exploration cadence: rounds 16, 32 draft n_max - 2, the others n_max
    std::string seq;
    for (int r = 1; r <= 32; ++r) {
        const int n = common_spec_tput_explore_n(f, 1, 8);
        if (n != 8) {
            seq += std::to_string(r) + ":" + std::to_string(n) + " ";
        }
    }
    check(seq == "16:6 32:6 ", "exploration every 16th unranked round at n_max - 2 (got " + seq + ")");
    common_spec_tput_fits g;
    seq.clear();
    for (int r = 1; r <= 16; ++r) {
        seq += std::to_string(common_spec_tput_explore_n(g, 3, 4));
    }
    check(seq == "4444444444444443", "n_max - 2 below n_lo: explores at n_lo = 3 (got " + seq + ")");
    common_spec_tput_fits h;
    bool all_max = true;
    for (int r = 1; r <= 64; ++r) {
        all_max = all_max && common_spec_tput_explore_n(h, 4, 4) == 4;
    }
    check(all_max, "no keepable shorter length (n_lo = n_max): never explores");

    // exploration rounds reach the rank: 4 samples at the second length in both the draft and the verify fit
    for (int k = 0; k < 3; ++k) {
        f.draft.add(6.0, 4.0); f.verify.add(7.0, 9.0);
    }
    check(!f.ranked(), "3 exploration samples: no rank yet");
    f.verify.add(7.0, 9.0);
    check(!f.ranked(), "verify ranked, draft not: no rank");
    f.draft.add(6.0, 4.0);
    check(f.ranked(), "4 exploration samples in draft and verify: rank");
}

// v51 r2 (Codex sol §4 counterexample + safety bound): a zero decision must beat n_max by the margin
static void test_margin() {
    // the sol counterexample, end to end: the world has constant costs (draft 4 ms for any n >= 1, verify 9 ms for any
    //   width, everything else 0), p_i = 0.4, n_max = 8. Samples at n = 8 alone: no rank -> n_max (test_rank). With the
    //   exploration rounds (n = 6, 7 rows) the slopes are identified as 0 and the rule keeps n_max: off's 1.666/13 =
    //   0.128 tok/ms beats the true zero's 1/9 = 0.111.
    common_spec_tput_fits f;
    for (int k = 0; k < 64; ++k) {
        const bool x = k % 16 == 15;
        f.draft  .add(x ? 6.0 : 8.0, 4.0);
        f.ckpt   .add(x ? 6.0 : 8.0, 0.0);
        f.verify .add(x ? 7.0 : 9.0, 9.0);
        f.catchup.add(x ? 7.0 : 9.0, 0.0);
        f.host_d .add(x ? 7.0 : 9.0, 0.0);
    }
    check(f.ranked(), "counterexample world with exploration: ranked");
    common_spec_tput_cost c;
    check(f.cost(0.5, c), "counterexample world: cost model warm");
    check(std::fabs(c.d1) < 1e-9 && std::fabs(c.v1) < 1e-9 && std::fabs(c.d0 - 4.0) < 1e-9 && std::fabs(c.v0 - 9.0) < 1e-9,
          "counterexample world: identified slopes 0 (draft 4 + 0n, verify 9 + 0m)");
    const std::vector<float> p(8, 0.4f);
    double rate = 0.0;
    const int n = common_spec_tput_best_n(p, 8, c, &rate, 1, COMMON_SPEC_TPUT_MARGIN_DEFAULT);
    check(n == 8, "counterexample: chooses n_max, not 0 (got " + std::to_string(n) + ")");
    check(rate > 1.0/9.0, "counterexample: the chosen rate beats the true zero");

    // the review's mis-identified model (d = 2 + 0.25n, v = 4.5 + 0.5m) models zero 56 % ahead: beyond any small
    //   margin - only the rank condition (no rank at one width) protects that case, the margin covers fit noise
    common_spec_tput_cost m;
    m.d0 = 2.0; m.d1 = 0.25; m.v0 = 4.5; m.v1 = 0.5;
    check(common_spec_tput_best_n(p, 8, m, nullptr, 1, COMMON_SPEC_TPUT_MARGIN_DEFAULT) == 0,
          "a 56 % modeled advantage passes the margin (the rank guards the unidentified case)");

    // a zero within the margin keeps drafting: rate(0) = 0.1, rate(1) = 1.1/11.55 = 0.1/1.05 (zero 5 % ahead)
    common_spec_tput_cost t;
    t.v0 = 10.0; t.d0 = 1.55;
    const std::vector<float> q(1, 0.1f);
    double r = 0.0;
    check(common_spec_tput_best_n(q, 1, t, &r, 1, 0.0) == 0, "no margin: the 5 % zero wins");
    check(common_spec_tput_best_n(q, 1, t, &r, 1, 0.04) == 0, "margin 4 %: the 5 % zero wins");
    check(common_spec_tput_best_n(q, 1, t, &r, 1, 0.10) == 1 && std::fabs(r - 1.1/11.55) < 1e-9,
          "margin 10 %: the 5 % zero keeps drafting (n_max), rate_out = the drafting rate");
    // an exact tie: drafting (ties no longer go to the zero)
    common_spec_tput_cost e;
    e.v0 = 10.0; e.d0 = 1.0;
    check(common_spec_tput_best_n(q, 1, e, nullptr, 1, 0.10) == 1, "an exact tie keeps drafting");
    // the margin compares against n_max: zero 20 % ahead of n_max passes even if a shorter draft is closer
    common_spec_tput_cost g;
    g.v0 = 10.0; g.d0 = 0.5; g.d1 = 3.0;
    const std::vector<float> q2(4, 0.1f);
    // rate(0) = 0.1; rate(1) = 1.1/13.5 = 0.0815; rate(4) = 1.1111/22.5 = 0.0494
    check(common_spec_tput_best_n(q2, 4, g, nullptr, 1, 0.10) == 0, "zero far ahead of n_max: zero");
}

// v51 r2 (Codex sol §1): until the rollback fit has W_MIN samples the replay is priced conservatively: the measured
//   restore + a whole plain pass of a+2 rows, never cheaper than the rb samples so far
static void test_replay_fallback() {
    common_spec_tput_fits f;
    for (int k = 0; k < 16; ++k) {
        const double n = 1 + k % 4, m = n + 1;
        f.draft  .add(n, 1.0 + 2.0*n);
        f.ckpt   .add(n, 3.0);
        f.verify .add(m, 10.0 + 1.0*m);
        f.catchup.add(m, 1.0 + 0.5*m);
        f.host   .add(0.0, 2.0);
    }
    common_spec_tput_cost c;
    check(f.cost(0.5, c), "replay fallback: cost model warm");
    check(std::fabs(c.r1 - 1.5) < 1e-9, "replay fallback: r1 = verify + catch-up per-row cost");
    check(std::fabs(c.r0 - (3.0 + 10.0 + 1.0 + 2.0)) < 1e-9,
          "replay fallback before any restore: the checkpoint save's mean stands in for the restore (r0 = 16)");
    f.restore.add(0.0, 5.0);
    f.cost(0.5, c);
    check(std::fabs(c.r0 - (5.0 + 13.0)) < 1e-9, "one measured restore: r0 = restore + v0 + c0 + host = 18");
    check(c.r0 > c.v0 + c.c0 + c.host, "the restore is never excluded from the replay price");
    // rb samples below the fallback do not cheapen it
    for (int k = 0; k < 3; ++k) {
        f.rb.add(4.0, 10.0);
    }
    f.cost(0.5, c);
    check(std::fabs(c.r0 - 18.0) < 1e-9, "cheap partial rb samples: the fallback stands (r0 = 18)");
    // rb samples above it lift it: mean 40 ms at r = 4 -> r0 = 40 - 1.5*4 = 34 (bit-for-bit the measured mean there)
    common_spec_tput_fits g = f;
    g.rb = common_spec_ew_line();
    for (int k = 0; k < 3; ++k) {
        g.rb.add(4.0, 40.0);
    }
    g.cost(0.5, c);
    check(std::fabs(c.r0 - 34.0) < 1e-9 && std::fabs(c.r0 + 4.0*c.r1 - 40.0) < 1e-9,
          "expensive partial rb samples: the fallback is lifted to them (r0 = 34)");
    // the rollback raises the cost of a rejecting draft: with a FULL target, a long draft at p = 0.5 costs more than
    //   the same draft with the old restore-free replay estimate
    common_spec_tput_cost old_c = c;
    old_c.r0 = c.v0 + c.c0 + c.host;
    c.n_rb_free = old_c.n_rb_free = 0;
    double g1 = 0.0, t1 = 0.0, g0 = 0.0, t0 = 0.0;
    const std::vector<float> p(4, 0.5f);
    common_spec_tput_eval(p, 4, c, g1, t1);
    common_spec_tput_eval(p, 4, old_c, g0, t0);
    check(g1 == g0 && t1 > t0, "conservative replay: same gain, higher cost than the restore-free estimate");
    // W_MIN rb samples: the fit takes over
    g.rb = common_spec_ew_line();
    for (int k = 0; k < 8; ++k) {
        g.rb.add(2.0 + k % 3, 30.0 + 2.0*(2.0 + k % 3));
    }
    g.cost(0.5, c);
    check(c.r1 > 1.9 && c.r1 < 2.1, "warm rb: the fit's slope (r1 ~ 2)");
}

// v51 r2 (Hermes): with n_min > 1 a probe shorter than n_min used to be cleared by the n_min rule and counted as a
//   failed probe - a cooldown that could sustain itself. The probe is max(n_min, 1) tokens, kept as drafted; a draft
//   the n_min rule cleared (outside a probe) is no cooldown observation and never a failure at position 0.
static void test_probe_n_min() {
    check(common_spec_tput_probe_n(0, 8) == 1 && common_spec_tput_probe_n(1, 8) == 1, "probe = 1 for n_min <= 1");
    check(common_spec_tput_probe_n(3, 8) == 3, "probe = n_min = 3");
    check(common_spec_tput_probe_n(3, 2) == 0, "probe does not fit n_max: no probe");

    check(common_spec_tput_empty_draft(1, 0) == COMMON_SPEC_TPUT_EMPTY_FAIL0, "stop at 0: a failure at 0");
    check(common_spec_tput_empty_draft(0, 0) == COMMON_SPEC_TPUT_EMPTY_FAIL0, "nothing attempted, nothing kept: failure");
    check(common_spec_tput_empty_draft(3, 2) == COMMON_SPEC_TPUT_EMPTY_STOP_K, "cleared 2 kept, stop at 2: the stop only");
    check(common_spec_tput_empty_draft(2, 2) == COMMON_SPEC_TPUT_EMPTY_NONE, "cleared without a stop: no observation");

    // the Hermes scenario, n_min = 3: the probe (3 tokens) stops at position 2 (p_min) every time, 1 of the 2 accepted.
    //   Old: cleared -> observe(0, 0) -> C doubles to c_max and stays there. New: kept -> observe(2, 1) -> C halves.
    const int n_min = 3;
    common_spec_cooldown old_cd, new_cd;
    old_cd.decide(0);
    new_cd.decide(0);
    for (int r = 0; r < 200; ++r) {
        if (old_cd.decide(5) > 0) {
            old_cd.observe(0, 0); // cleared probe = failed probe
        }
        if (new_cd.cooling() && new_cd.decide(5) > 0) {
            const int n_probe = common_spec_tput_probe_n(n_min, 8);
            const int n_kept  = std::min(n_probe, 2); // keep_short: the 2 drafted tokens are verified
            new_cd.observe(n_kept, 1);
        }
    }
    check(old_cd.cooling() && old_cd.c == old_cd.c_max, "old: cleared probes keep the cooldown at c_max (self-sustaining)");
    check(!new_cd.cooling(), "new: a kept short probe that is accepted ends the cooldown");

    // non-probe rounds whose drafts the n_min rule clears do not start a cooldown (no observation): 10 such rounds
    common_spec_cooldown k;
    for (int r = 0; r < 10; ++r) {
        k.decide(4);
        if (common_spec_tput_empty_draft(4, 2) == COMMON_SPEC_TPUT_EMPTY_FAIL0) {
            k.observe(0, 0);
        }
    }
    check(!k.cooling() && k.n_fail == 0, "cleared drafts never start a cooldown");
}

static void test_cooldown() {
    common_spec_cooldown cd;

    // outside a cooldown the rule passes through
    check(cd.decide(3) == 3 && !cd.cooling(), "rule passes through");

    // a true-zero decision starts a cooldown of c_min rounds (this one included), then a probe n = 1
    std::string seq;
    for (int r = 0; r < 4; ++r) {
        seq += std::to_string(cd.decide(r == 0 ? 0 : 5));
    }
    check(seq == "0000", "zero decision -> c_min zero rounds (got " + seq + ")");
    check(cd.decide(5) == 1, "then a probe of 1");
    // a failed probe doubles C: 8 zero rounds, then a probe
    cd.observe(1, 0);
    check(cd.c == 8, "failed probe doubles C");
    seq.clear();
    for (int r = 0; r < 8; ++r) {
        seq += std::to_string(cd.decide(5));
    }
    check(seq == "00000000" && cd.decide(5) == 1, "8 zero rounds then a probe");
    // C is capped
    for (int k = 0; k < 10; ++k) {
        cd.observe(1, 0);
        while (cd.decide(5) == 0) {}
    }
    check(cd.c == cd.c_max, "C capped at c_max (" + std::to_string(cd.c) + ")");
    // an empty probe (nothing proposed) is a failure too
    const int c_before = cd.c;
    cd.observe(0, 0);
    check(cd.c == c_before && cd.cooling(), "empty probe at the cap stays cooling");
    while (cd.decide(5) == 0) {}
    // a success halves C (spec §3): 64 -> 32 -> 16 -> 8 -> 4, then below c_min the cooldown ends
    cd.observe(1, 1);
    check(cd.cooling() && cd.c == 32 && cd.wait == 32, "a success halves C (64 -> 32)");
    seq.clear();
    for (int k = 0; k < 4; ++k) {
        cd.observe(1, 1);
        seq += std::to_string(cd.c) + " ";
    }
    check(seq == "16 8 4 0 " && !cd.cooling(), "successes halve C down to the end of the cooldown (got " + seq + ")");
    check(cd.decide(4) == 4, "out of the cooldown the rule passes through again");
    // the next zero decision starts from c_min again; one success at c_min ends it
    cd.decide(0);
    check(cd.c == cd.c_min, "a new cooldown starts at c_min");
    cd.observe(1, 1);
    check(!cd.cooling(), "a success at c_min ends the cooldown");

    // K = 3 consecutive failed rounds (0 accepted or empty) start a cooldown although the rule says draft
    common_spec_cooldown k;
    k.decide(4); k.observe(4, 0);
    k.decide(4); k.observe(0, 0);
    check(!k.cooling(), "2 failures: still drafting");
    k.decide(4); k.observe(4, 0);
    check(k.cooling() && k.c == k.c_min, "3rd consecutive failure starts a cooldown");
    seq.clear();
    for (int r = 0; r < 4; ++r) {
        seq += std::to_string(k.decide(4));
    }
    check(seq == "0000" && k.decide(4) == 1, "c_min zero rounds then a probe");
    // a success in between resets the failure count
    common_spec_cooldown m;
    m.observe(4, 0); m.observe(4, 0); m.observe(4, 2); m.observe(4, 0); m.observe(4, 0);
    check(!m.cooling(), "a success resets the consecutive-failure count");
}

int main() {
    test_rule();
    test_rollback_cycle();
    test_depth_buckets();
    test_ckpt_cost();
    test_timing_hygiene();
    test_rule_hygiene();
    test_cooldown();
    test_rank();
    test_margin();
    test_replay_fallback();
    test_probe_n_min();
    if (n_fail) {
        printf("test-spec-draft-policy: %d failures (%d checks ok)\n", n_fail, n_ok);
        return 1;
    }
    printf("test-spec-draft-policy: OK (%d checks)\n", n_ok);
    return 0;
}
