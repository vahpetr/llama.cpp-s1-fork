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

// brute force reference: tokens per ms of every n, argmax with ties to the shorter draft
static int ref_best(const std::vector<float> & p, int n_max, const common_spec_tput_cost & c) {
    int best = 0;
    double best_r = -1.0;
    for (int n = 0; n <= n_max; ++n) {
        double gain = 1.0;
        for (int i = 1; i <= n; ++i) {
            double P = 1.0;
            for (int j = 1; j <= i; ++j) {
                P *= j - 1 < (int) p.size() ? p[j - 1] : 1.0;
            }
            gain += P;
        }
        const double m = n + 1.0;
        const double t = (n ? c.d0 + c.d1*n : 0.0) + c.v0 + c.v1*m + c.c0 + c.c1*m + c.host;
        if (gain / t > best_r) {
            best_r = gain / t;
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
    test_cooldown();
    if (n_fail) {
        printf("test-spec-draft-policy: %d failures (%d checks ok)\n", n_fail, n_ok);
        return 1;
    }
    printf("test-spec-draft-policy: OK (%d checks)\n", n_ok);
    return 0;
}
