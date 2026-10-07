// [TAG_SPEC_DRAFT_POLICY] --spec-draft-policy tput (v51): the throughput-max draft length as a pure function
// (acceptance p_i, round costs -> n*) and the cooldown state machine around true-zero rounds.

#include "speculative.h"

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
    // a non-zero acceptance leaves the cooldown, the rule passes through again
    cd.observe(1, 1);
    check(!cd.cooling() && cd.decide(4) == 4, "success resets the cooldown");
    // the next zero decision starts from c_min again
    cd.decide(0);
    check(cd.c == cd.c_min, "a new cooldown starts at c_min");
    cd.observe(1, 1);

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
    test_cooldown();
    if (n_fail) {
        printf("test-spec-draft-policy: %d failures (%d checks ok)\n", n_fail, n_ok);
        return 1;
    }
    printf("test-spec-draft-policy: OK (%d checks)\n", n_ok);
    return 0;
}
