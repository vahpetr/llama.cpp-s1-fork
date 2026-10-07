#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // [TAG_SPEC_DRAFT_POLICY] outputs of the last _draft() call (-1: the implementation does not report them):
    //   n_attempted = draft positions sampled for this sequence (incl. the one a p_min exit stopped at), n_kept = tokens
    //   drafted before the n_min rule (a draft shorter than n_min is cleared from result, not from n_kept)
    int32_t n_attempted = -1;
    int32_t n_kept      = -1;

    // [TAG_SPEC_DRAFT_POLICY] (v51 r2, Hermes) input: true = the n_min rule does not clear this draft (a tput cooldown
    //   probe of max(n_min, 1) tokens that stopped early is kept as drafted - a cleared probe would count as a failed
    //   probe and could keep a cooldown alive by itself). false (the default, every other caller): unchanged.
    bool keep_short = false;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
// false: the blob was not restored (no implementation claims it, or the claiming one rejects its width / size);
//   an empty blob is true only when no implementation is stateful (Codex gate r3 Major 2: MTP / eagle3 with a
//   boundary stash require their carry - an empty one is a missing state)
bool common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// the number of active implementations whose per-seq carry is required (MTP, eagle3 with a boundary stash);
//   get_state() serializes one carry, so persisting with more than one is refused by the callers
int common_speculative_n_stateful(const common_speculative * spec);

// the restore verdict of common_speculative_set_state (exposed for an offline unit check): nullptr = restored /
//   legitimately nothing to restore, else the reason. n_required: stateful implementations, n_bytes: the carry's
//   size, n_ok / n_failed: implementations that restored / rejected it
const char * common_speculative_carry_verdict(int n_required, size_t n_bytes, int n_ok, int n_failed);

// the per-seq carry schemas of the stateful implementations (schema id, tag and row width), "," joined; "" = none
std::string common_speculative_state_schema(const common_speculative * spec);

// [TAG_SPEC_RESET] forget the per-seq carry of every implementation (the seq's target state was replaced)
void common_speculative_reset_state(common_speculative * spec, llama_seq_id seq_id);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);

// [TAG_SPEC_DRAFT_POLICY] (v51) --spec-draft-policy tput: the throughput-max draft length (server only).
//   One round with n drafted tokens commits 1 + sum_{i<=n} P_i tokens on average (P_i = prod_{j<=i} p_j, p_j the
//   per-position conditional acceptance) and costs T_draft(n) + T_verify(n+1) + T_catchup(n+1) + T_host; n = 0 is a
//   plain one-token decode (no draft call: T_draft(0) = 0, no checkpoint: T_ckpt(0) = 0). All costs in ms, affine in
//   the row count; T_draft(n) below includes the checkpoint term k0 + k1*n.
//   Complete cycle (v51 r1, Codex C Major): a round whose rejection rolls back deeper than n_rb_free (a FULL target:
//   every rejection, an RS target: deeper than n_rs_seq) restores the per-round checkpoint and replays the accepted
//   prefix in the next pass. With a accepted (first rejection at position a+1, probability q_a = P_a*(1 - p_{a+1}))
//   the replay pass decodes a + 2 rows and commits one token more than a plain rejection (the replay's own sample);
//   its cost r0 + r1*(a+2) (restore + the whole replay pass, fitted) and that token are attributed to the round:
//     gain(n) = 1 + sum_{i<=n} P_i + sum_{a<n, n-a>n_rb_free} q_a
//     cost(n) = T_draft(n) + T_verify(n+1) + T_catchup(n+1) + T_host + sum_{a<n, n-a>n_rb_free} q_a*(r0 + r1*(a+2))
struct common_spec_tput_cost {
    double d0 = 0.0, d1 = 0.0; // the draft call for n >= 1 tokens: d0 + d1*n
    double v0 = 0.0, v1 = 0.0; // the target decode of m = n+1 rows: v0 + v1*m
    double c0 = 0.0, c1 = 0.0; // the MTP process() catch-up of m rows: c0 + c1*m
    double k0 = 0.0, k1 = 0.0; // the per-round checkpoint save + draft-context restore for n >= 1: k0 + k1*n (n = 0
                               //   takes none: not part of the host intercept, v51 r1 Codex D Major)
    double host = 0.0;         // per-round host time (from zero rounds: no draft, no checkpoint)
    double r0 = 0.0, r1 = 0.0; // a rollback cycle: checkpoint restore + the replay pass of r = a+2 rows: r0 + r1*r
    int    n_rb_free = 1 << 30; // the deepest rollback that needs no restore (FULL: 0, RS: n_rs_seq, else never)
};

// the expected committed tokens (gain) and wall ms (cost) of one round that drafts n tokens (see above)
void common_spec_tput_eval(const std::vector<float> & p, int n, const common_spec_tput_cost & c, double & gain, double & cost);

// n* = argmax over n in {0} U [max(1, n_min), n_max] of gain/cost (v51 r1, Codex C Minor: the drafter clears a draft
// shorter than n_min, so those lengths are not candidates - compared directly, not promoted after the choice);
// p[i] = p_{i+1} (missing entries count as 1, see common_spec_tput_p_eff for the server's estimate);
// ties between drafting lengths go to the shorter draft (a zero never wins a tie, see below). rate_out (optional): the expected tokens per ms at n*.
//   -1: the cost model is invalid (a candidate's cost is non-finite or <= 0) - the caller takes its prior path.
// v51 r2 (Codex sol §4 safety bound): n* = 0 only when the modeled rate of 0 exceeds the modeled rate of n_max by
//   more than zero_margin (rate(0) > (1 + zero_margin)*rate(n_max)); otherwise (ties, inside the margin) the best
//   drafting length - the fits are not exact, so an uncertain zero keeps drafting (server: LLAMA_SPEC_TPUT_MARGIN, 0.10)
int common_spec_tput_best_n(const std::vector<float> & p, int n_max, const common_spec_tput_cost & c,
                            double * rate_out = nullptr, int n_min = 1, double zero_margin = 0.0);

#define COMMON_SPEC_TPUT_MARGIN_DEFAULT 0.10

// the acceptance the rule sees: p_eff[i] = ema[i] where position i was observed (n_obs[i] > 0); a never-reached
//   position takes the estimate of the deepest observed position before it (v51 r1, Codex C Minor: not the optimistic
//   1.0 of the EMA's start, which overvalues long tails after a workload change); 1.0 only while nothing was observed
std::vector<float> common_spec_tput_p_eff(const std::vector<float> & ema, const std::vector<uint32_t> & n_obs);

// exponentially weighted least squares t = a + b*n of one round component (ms), for --spec-draft-policy tput
struct common_spec_ew_line {
    static constexpr double ALPHA   = 1.0/64.0;
    static constexpr double W_MIN   = 8.0;
    static constexpr double VAR_MIN = 0.25;

    static constexpr int N_X = 64; // the per-x sample counts (x rounded, clamped to [0, N_X-1]) of the rank condition

    double w = 0.0, s_n = 0.0, s_t = 0.0, s_nn = 0.0, s_nt = 0.0;
    int    n_samples  = 0; // accepted samples (the warm-up counts these: W_MIN of them, not the decayed weight)
    int    n_rejected = 0; // non-finite or negative samples dropped
    int    n_at[N_X]  = { 0 }; // accepted samples per distinct x (not decayed)

    // the distinct x values with at least min_count accepted samples each (v51 r2, Codex sol §4: the slope is
    //   identified only by samples at >= 2 distinct x; 8 samples at one x are a warm-up, not a rank)
    int n_distinct(int min_count) const;

    // a non-finite n or t, or t < 0, is rejected (counted, not fitted)
    void add(double n, double t_ms);
    // the weighted means of x and t: from the first sample on (false before any)
    bool mean(double & n, double & t) const;
    // false before W_MIN samples, or when the fit is not finite; a slope that the samples do not identify (one row count only, or a negative fit)
    // falls back to b_prior, capped so that the intercept through the mean keeps at least half of the mean time (a
    // prior from another machine must not turn the measured fixed cost into a per-row one)
    bool get(double & a, double & b, double b_prior) const;
};

// the fits are kept per KV-depth bucket (v51 r1, Codex D Major): every component's cost grows with the depth the
//   round attends over, so a deep slot is priced only from rounds at its own depth. log2 buckets: [0, 8k) -> 0,
//   [2^(12+b), 2^(13+b)) -> b (107k -> 4, 524k -> 7, 1M -> 8), >= 2M -> 9.
#define COMMON_SPEC_TPUT_N_BUCKETS 10
int common_spec_tput_bucket(int n_kv);

struct common_spec_tput_fits {
    common_spec_ew_line draft;   // draft wall vs drafted n
    common_spec_ew_line verify;  // target decode wall vs batch rows
    common_spec_ew_line catchup; // MTP process() wall vs batch rows
    common_spec_ew_line ckpt;    // the checkpoint save + draft-context restore of a drafting round vs drafted n
    common_spec_ew_line host;    // the rest of a zero round (the host intercept)
    common_spec_ew_line host_d;  // the rest of a drafting round (checkpoint excluded) vs rows: fallback for host
    common_spec_ew_line rb;      // the rollback cycle (restore + replay pass) vs the replay rows
    common_spec_ew_line restore; // every checkpoint restore of a rollback (x = 0; multi-slot passes included)

    uint32_t n_unranked_rounds = 0; // drafting decisions taken in this bucket while it had no rank (exploration cadence)

    // v51 r2 (Codex sol §4 Major): the fits may decide a zero or a shorter draft only once both slopes they
    //   extrapolate are identified - the draft fit's samples span >= 2 distinct attempted lengths and the verify fit's
    //   >= 2 distinct row counts, each with >= COMMON_SPEC_TPUT_RANK_MIN samples. Until then the policy is `off`.
    bool ranked() const;

    // the round cost model of this bucket (n_rb_free left to the caller); b_prior: the draft / verify slope while a
    //   fit does not identify it. false while a component is not warm (or not finite): the caller's prior path.
    //   The rollback cycle while rb has < W_MIN samples (v51 r2, Codex sol §1): priced conservatively as the measured
    //   restore (its mean; before the first restore: the checkpoint save's mean) + a whole plain pass of r = a+2 rows
    //   (verify + catch-up + host), r0 = t_restore + v0 + c0 + host, r1 = v1 + c1 - lifted so that it is never
    //   cheaper than the rb samples taken so far (at their mean rows)
    bool cost(double b_prior, common_spec_tput_cost & c) const;
};

#define COMMON_SPEC_TPUT_RANK_MIN     4
#define COMMON_SPEC_TPUT_EXPLORE_EVERY 16

// the draft length of a drafting round in a bucket WITHOUT rank: n_max (as `off`), except every
//   COMMON_SPEC_TPUT_EXPLORE_EVERY-th such round, which drafts max(n_lo, n_max - 2) (an exploration round: fitted like
//   any other round, never a cooldown failure). Cost: <= 1/16 of the unranked rounds lose the expected acceptance of
//   the last two positions (P_{n_max-1} + P_{n_max} tokens) and save two draft steps + two verify rows; it ends once
//   the bucket has rank. n_max - 2 <= n_lo - 1 (no shorter keepable length): always n_max (the bucket stays `off`).
int common_spec_tput_explore_n(common_spec_tput_fits & f, int n_lo, int n_max);

struct common_spec_tput_model {
    common_spec_tput_fits bucket[COMMON_SPEC_TPUT_N_BUCKETS];

    common_spec_tput_fits       & at(int n_kv)       { return bucket[common_spec_tput_bucket(n_kv)]; }
    const common_spec_tput_fits & at(int n_kv) const { return bucket[common_spec_tput_bucket(n_kv)]; }
};

// the cooldown around true-zero decisions: without a draft the acceptance estimate never moves, so after a zero
//   decision (or K consecutive rounds that proposed nothing / had nothing accepted) the slot plays n = 0 for C rounds and
//   then probes (the server drafts max(n_min, 1) tokens for it, kept even below n_min); a failed probe doubles C (c_min .. c_max), a success (a non-zero acceptance) halves C (spec §3;
//   below c_min the cooldown ends).
struct common_spec_cooldown {
    int k_fail = 3;  // consecutive failed rounds that start a cooldown
    int c_min  = 4;
    int c_max  = 64;

    int n_fail = 0;  // consecutive failed rounds (outside a cooldown)
    int c      = 0;  // the current cooldown length, 0 = not cooling
    int wait   = 0;  // zero rounds left before the next probe

    // the draft length for this round from the rule's n_rule: n_rule outside a cooldown (a zero starts one),
    // inside it 0 or the probe (1)
    int  decide(int n_rule);
    // the outcome of a round that called the drafter: n_proposed tokens drafted (0 = empty), n_accepted of them taken
    void observe(int n_proposed, int n_accepted);
    bool cooling() const { return c > 0; }
};

// v51 r2 (Hermes): the cooldown probe's length: max(n_min, 1) - the shortest draft the n_min rule keeps - or 0 when it
//   does not fit n_max (no probe this round). The probe is drafted with keep_short (never cleared by the n_min rule).
int common_spec_tput_probe_n(int n_min, int n_max);

// v51 r2 (Hermes): what an EMPTY draft of a tput round observes, from the drafter's n_attempted / n_kept
enum common_spec_tput_empty_kind {
    COMMON_SPEC_TPUT_EMPTY_FAIL0 = 0, // nothing drafted (a stop at position 0): a failure at 0, a failed round
    COMMON_SPEC_TPUT_EMPTY_STOP_K,    // the n_min rule cleared n_kept > 0 tokens and the drafter stopped at n_kept: only
                                      //   that stop is observed (positions < n_kept unverified); no cooldown observation
    COMMON_SPEC_TPUT_EMPTY_NONE,      // cleared without a stop (n_kept = n_attempted): no observation at all
};
common_spec_tput_empty_kind common_spec_tput_empty_draft(int n_attempted, int n_kept);
