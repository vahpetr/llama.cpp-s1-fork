// [TAG_G68_H3_FIX] LLAMA_SPEC_DRAFT_STOP_EOG=1 (H3) x --spec-draft-n-min under rejection sampling (astra v68 delta,
// Important): the n_min rule cleared an EOG-shortened sampled draft, so the kept drafts were conditioned on the drawn
// token. With p = q, first-position EOG probability e and n_min = n_max = 2 the first-token EOG probability was e^2.
//
// A synthetic drafter (prefix-independent q per position, vocab {A, B, EOG}) + the real n_min rule
// common_spec_draft_clear_short + the rejection verifier (accept x with min(1, p/q), else the residual (p - q)+; an
// empty draft = the target samples p). The exact first-token distribution is enumerated over every draft branch, and
// 200 000 seeded trials repeat it by sampling. The pre-fix arm (= f3cbd94cb) is the same rule with eog_end = false.
// (pkg9-fix) n_max here is the chain's effective cap, which the caller's per-sequence cap (context left, output limit)
// can put below n_min: the pkg9 rule (EOG exempt whatever the cap, = n_cap unlimited) is biased there (astra v69: p = q,
// e = 0.5, n_min 2, cap 1 -> 0.75), the fixed rule (exempt only while cap >= n_min) is not.

#include "speculative.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;
static int n_ok   = 0;

static void check(bool ok, const std::string & what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    (ok ? n_ok : n_fail)++;
}

enum { TOK_A = 0, TOK_B = 1, TOK_EOG = 2, NV = 3 };

using dist = std::vector<double>; // NV entries

struct setup {
    dist p;            // target, same at every position (only the first token is checked)
    std::vector<dist> q; // draft per position
    int  n_min;
    int  n_max;
    bool stop_eog;
    bool sampled;      // rejection-verified round (T > 0); false = greedy
    bool prefix;       // pre-fix rule (f3cbd94cb): the EOG end is not exempt
    bool pkg9 = false; // the pkg9 rule (1d7bcd1c5): the EOG end exempt whatever the cap
};

// the n_min decision as the drafter makes it
static bool clears(const setup & s, size_t n, bool eog_end) {
    return common_spec_draft_clear_short(n, s.n_min, s.pkg9 ? INT32_MAX : s.n_max, /*keep_short=*/ false, s.sampled,
                                         s.prefix ? false : eog_end);
}

// first output token given the first drafted token x (rejection sampling)
static dist verify_first(const setup & s, int x) {
    const dist & q = s.q[0];
    dist out(NV, 0.0);
    const double a = q[x] > 0 ? std::min(1.0, s.p[x] / q[x]) : 0.0;
    out[x] += a;
    double z = 0.0;
    for (int v = 0; v < NV; ++v) z += std::max(0.0, s.p[v] - q[v]);
    for (int v = 0; v < NV; ++v) out[v] += (1.0 - a) * (z > 0 ? std::max(0.0, s.p[v] - q[v]) / z : 0.0);
    return out;
}

// exact: enumerate every draft branch
static dist exact_first(const setup & s) {
    dist out(NV, 0.0);
    std::vector<int> d;
    // recursive over positions
    struct rec {
        static void go(const setup & s, std::vector<int> & d, double pr, dist & out) {
            const int i = (int) d.size();
            const bool eog_end = s.stop_eog && i > 0 && d.back() == TOK_EOG;
            if (i == s.n_max || eog_end) {
                if (clears(s, d.size(), eog_end)) {
                    for (int v = 0; v < NV; ++v) out[v] += pr * s.p[v];
                } else {
                    const dist f = verify_first(s, d[0]);
                    for (int v = 0; v < NV; ++v) out[v] += pr * f[v];
                }
                return;
            }
            for (int v = 0; v < NV; ++v) {
                if (s.q[i][v] <= 0) continue;
                d.push_back(v);
                go(s, d, pr * s.q[i][v], out);
                d.pop_back();
            }
        }
    };
    rec::go(s, d, 1.0, out);
    return out;
}

static int draw(std::mt19937_64 & rng, const dist & w) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng), c = 0.0;
    for (int v = 0; v < NV; ++v) { c += w[v]; if (r < c) return v; }
    return NV - 1;
}

static double mc_first_eog(const setup & s, int n_trials, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int n_eog = 0;
    for (int t = 0; t < n_trials; ++t) {
        std::vector<int> d;
        bool eog_end = false;
        while ((int) d.size() < s.n_max) {
            d.push_back(draw(rng, s.q[d.size()]));
            if (s.stop_eog && d.back() == TOK_EOG) { eog_end = true; break; }
        }
        int first;
        if (clears(s, d.size(), eog_end)) {
            first = draw(rng, s.p);
        } else {
            const int x = d[0];
            const dist & q = s.q[0];
            if (u(rng) < std::min(1.0, s.p[x] / q[x])) {
                first = x;
            } else {
                dist r(NV);
                double z = 0.0;
                for (int v = 0; v < NV; ++v) { r[v] = std::max(0.0, s.p[v] - q[v]); z += r[v]; }
                for (int v = 0; v < NV; ++v) r[v] /= z;
                first = draw(rng, r);
            }
        }
        n_eog += first == TOK_EOG;
    }
    return (double) n_eog / n_trials;
}

static double tv(const dist & a, const dist & b) {
    double t = 0.0;
    for (int v = 0; v < NV; ++v) t += std::fabs(a[v] - b[v]);
    return 0.5 * t;
}

int main(int argc, char ** argv) {
    const int n_trials = argc > 1 ? std::max(1, atoi(argv[1])) : 200000;

    // 1. the astra case: p = q, e = 0.5, n_min = n_max = 2, no other stops
    {
        const dist e = { 0.5, 0.0, 0.5 };
        setup fix = { e, { e, e }, 2, 2, true, true, false };
        setup pre = fix; pre.prefix = true;
        const double x_fix = exact_first(fix)[TOK_EOG];
        const double x_pre = exact_first(pre)[TOK_EOG];
        const double m_fix = mc_first_eog(fix, n_trials, 42);
        const double m_pre = mc_first_eog(pre, n_trials, 42);
        printf("astra case: first-token EOG exact fix %.6f pre-fix %.6f | %d seeded trials fix %.5f pre-fix %.5f\n",
               x_fix, x_pre, n_trials, m_fix, m_pre);
        check(std::fabs(x_fix - 0.5)  < 1e-12, "exact fix = e = 0.5");
        check(std::fabs(x_pre - 0.25) < 1e-12, "exact pre-fix = e^2 = 0.25 (the bias reproduced)");
        const double se = std::sqrt(0.25 / n_trials); // binomial sd at 0.5 (larger than at 0.25)
        check(std::fabs(m_fix - 0.5)  < 5 * se, "sampled fix within 5 sd of 0.5");
        check(std::fabs(m_pre - 0.25) < 5 * se, "sampled pre-fix within 5 sd of 0.25");
    }

    // 2. general: fixed rule is distribution-exact (first token ~ p) for p != q, n_min in [0, n_max]; the pre-fix rule is
    //    not whenever n_min > 1 can clear an EOG end
    {
        const std::vector<dist> ps = { { 0.6, 0.1, 0.3 }, { 0.2, 0.2, 0.6 }, { 0.5, 0.0, 0.5 } };
        const std::vector<dist> qs = { { 0.3, 0.3, 0.4 }, { 0.7, 0.2, 0.1 }, { 0.5, 0.0, 0.5 } };
        double max_fix = 0.0, max_pre = 0.0;
        int n_cases = 0;
        for (const auto & p : ps) for (const auto & q0 : qs) for (const auto & q1 : qs)
        for (int n_max = 1; n_max <= 4; ++n_max) for (int n_min = 0; n_min <= n_max; ++n_min) {
            setup s = { p, { q0, q1, q0, q1 }, n_min, n_max, true, true, false };
            max_fix = std::max(max_fix, tv(exact_first(s), p));
            if (n_min >= 2) { s.prefix = true; max_pre = std::max(max_pre, tv(exact_first(s), p)); }
            n_cases++;
        }
        printf("general (%d cases): max TV(first token, p) fix %.3g, pre-fix %.3g\n", n_cases, max_fix, max_pre);
        check(max_fix < 1e-12, "fix: first token ~ p in every case");
        check(max_pre > 1e-3,  "pre-fix: biased in some n_min >= 2 case");
    }

    // 3. knob off (stop_eog = 0): fix and pre-fix identical and exact
    {
        const dist p = { 0.6, 0.1, 0.3 }, q = { 0.3, 0.3, 0.4 };
        setup s = { p, { q, q, q, q }, 2, 3, false, true, false };
        setup t = s; t.prefix = true;
        check(tv(exact_first(s), exact_first(t)) == 0.0 && tv(exact_first(s), p) < 1e-12, "knob off: rule unchanged, exact");
    }

    // 4. greedy (T = 0) rounds: the rule is the old one for every input
    {
        bool same = true;
        for (size_t n = 0; n <= 8; ++n) for (int n_min = -1; n_min <= 6; ++n_min)
        for (int ks = 0; ks < 2; ++ks) for (int ee = 0; ee < 2; ++ee) {
            const bool old_rule = !ks && n < (size_t) std::max(n_min, 0);
            for (int n_cap = 0; n_cap <= 8; ++n_cap) {
                same = same && common_spec_draft_clear_short(n, n_min, n_cap, ks, /*sampled=*/ false, ee) == old_rule;
                // sampled rounds differ from the old rule only by the EOG exemption, and only while cap >= n_min
                same = same && common_spec_draft_clear_short(n, n_min, n_cap, ks, /*sampled=*/ true, ee) ==
                                   (old_rule && !(ee && n_cap >= n_min));
            }
        }
        check(same, "T = 0 rule identical to f3cbd94cb; sampled differs only for an EOG end with cap >= n_min");
    }

    // 5. (pkg9-fix) effective cap below n_min: the astra v69 case, then every case of 2. with n_min > cap
    {
        const dist e = { 0.5, 0.0, 0.5 };
        setup fix = { e, { e, e }, 2, 1, true, true, false };
        setup p9 = fix; p9.pkg9 = true;
        const double x_fix = exact_first(fix)[TOK_EOG];
        const double x_p9  = exact_first(p9)[TOK_EOG];
        const double m_fix = mc_first_eog(fix, n_trials, 7);
        const double m_p9  = mc_first_eog(p9, n_trials, 7);
        printf("cap < n_min (n_min 2, cap 1): first-token EOG exact fix %.6f pkg9 %.6f | %d seeded trials fix %.5f pkg9 %.5f\n",
               x_fix, x_p9, n_trials, m_fix, m_p9);
        check(std::fabs(x_fix - 0.5)  < 1e-12, "cap < n_min: exact fix = e = 0.5");
        check(std::fabs(x_p9  - 0.75) < 1e-12, "cap < n_min: exact pkg9 rule = 0.75 (the astra v69 bias reproduced)");
        const double se = std::sqrt(0.25 / n_trials);
        check(std::fabs(m_fix - 0.5)  < 5 * se, "cap < n_min: sampled fix within 5 sd of 0.5");
        check(std::fabs(m_p9  - 0.75) < 5 * se, "cap < n_min: sampled pkg9 rule within 5 sd of 0.75");

        const std::vector<dist> ps = { { 0.6, 0.1, 0.3 }, { 0.2, 0.2, 0.6 }, { 0.5, 0.0, 0.5 } };
        const std::vector<dist> qs = { { 0.3, 0.3, 0.4 }, { 0.7, 0.2, 0.1 }, { 0.5, 0.0, 0.5 } };
        double max_fix = 0.0, max_p9 = 0.0;
        int n_cases = 0;
        for (const auto & p : ps) for (const auto & q0 : qs) for (const auto & q1 : qs)
        for (int n_max = 1; n_max <= 4; ++n_max) for (int n_min = n_max + 1; n_min <= 6; ++n_min) {
            setup s = { p, { q0, q1, q0, q1 }, n_min, n_max, true, true, false };
            max_fix = std::max(max_fix, tv(exact_first(s), p));
            s.pkg9 = true;
            max_p9 = std::max(max_p9, tv(exact_first(s), p));
            n_cases++;
        }
        printf("cap < n_min (%d cases): max TV(first token, p) fix %.3g, pkg9 rule %.3g\n", n_cases, max_fix, max_p9);
        check(max_fix < 1e-12, "cap < n_min, fix: first token ~ p in every case");
        check(max_p9 > 1e-3,   "cap < n_min, pkg9 rule: biased in some case");
    }

    printf("%d ok, %d failed\n", n_ok, n_fail);
    return n_fail == 0 ? 0 : 1;
}
