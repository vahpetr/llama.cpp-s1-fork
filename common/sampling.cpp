#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

//
// G32: candidate-window fast path + sampling time counters
//
// LLAMA_SAMPLER_FASTPATH:
//   0 / off / false - never take the fast path (exactly today's code)
//   1 / unset       - take it whenever it is provably identical (default)
//   2 / verify      - take it, but also build the reference window and log any disagreement (for FAST_TOPK_PEN
//                     the reference is the whole vocabulary *after* the real penalties sampler, see below)
//
static int common_sampler_fastpath_env() {
    static const int val = []() {
        const char * s = getenv("LLAMA_SAMPLER_FASTPATH");
        if (s == nullptr || *s == '\0') {
            return 1;
        }
        if (strcmp(s, "0") == 0 || strcmp(s, "off") == 0 || strcmp(s, "false") == 0) {
            return 0;
        }
        if (strcmp(s, "2") == 0 || strcmp(s, "verify") == 0) {
            return 2;
        }
        return 1;
    }();

    return val;
}

// process-wide, so that a server can report them without reaching into per-slot samplers
static std::atomic<uint64_t> g_smpl_t_us  {0};
static std::atomic<uint64_t> g_smpl_n     {0};
static std::atomic<uint64_t> g_smpl_n_fast{0};
static std::atomic<uint64_t> g_smpl_n_fast_pen{0};

uint64_t common_sampler_perf_time_us  () { return g_smpl_t_us      .load(std::memory_order_relaxed); }
uint64_t common_sampler_perf_n_sample () { return g_smpl_n         .load(std::memory_order_relaxed); }
uint64_t common_sampler_perf_n_fast   () { return g_smpl_n_fast    .load(std::memory_order_relaxed); }
uint64_t common_sampler_perf_n_fast_pen() { return g_smpl_n_fast_pen.load(std::memory_order_relaxed); }

// FAST_TOPK_PEN refuses a window larger than this: past it the saving over the full vocabulary stops being the
// point of the exercise, and the per-sample scratch work (one extra penalties pass + a sort) stops being free.
static constexpr int COMMON_SAMPLER_FAST_WINDOW_MAX = 4096;

// the comparator llama_token_data_array_partial_sort() uses; descending by logit
static bool common_sampler_logit_desc(const llama_token_data & a, const llama_token_data & b) {
    return a.logit > b.logit;
}

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    // today's path: one llama_token_data per vocabulary entry
    void set_logits_full(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };

        fast_used = false;
    }

    void set_logits(struct llama_context * ctx, int idx) {
        if (set_logits_fast(ctx, idx)) {
            return;
        }

        set_logits_full(ctx, idx);
    }

    // ---------------------------------------------------------------------------------------------------------
    // G32: build only the window of the vocabulary that the chain can reach.
    //
    // set_logits_full() writes n_vocab * sizeof(llama_token_data) (248 320 * 12 B = 2.98 MB on qwen4exp) on every
    // sample call, and the chain then walks all of it (temp's -INF sweep, top-k's partial_sort over the full
    // array, ...). ~5 calls per speculative round. When the chain provably cannot reach outside a small window,
    // the identical result can be produced from a window built straight from the raw logits.
    //
    // Both modes reproduce the *state of cur_p at a point in today's chain*, so everything from that point on is
    // bit-identical, including the RNG draw of the `dist` sampler:
    //
    //   FAST_GREEDY - the chain reduces to "argmax of the raw logits" (temp <= 0, dynatemp off, and nothing in
    //       the chain that can reorder the logits or drop the maximum: penalties/DRY/XTC/typical/top-n-sigma are
    //       all statically disabled, and top-k / top-p / min-p can only truncate a descending array, so they
    //       always keep the maximum). llama_sampler_temp_impl(<=0) then sets every other candidate to -INF and
    //       `dist` selects the single finite one. The scan below is llama_sampler_temp_impl's own scan (strict
    //       `>`, so the earliest index wins a tie), and the fast path only fires when the maximum is *unique*,
    //       so no tie order can be observed. The chain is then applied to the 1-element array: top-k/top-p/min-p
    //       keep it, temp_impl(<=0) leaves it untouched, and `dist` takes its size == 1 branch, which draws from
    //       the RNG exactly once - as the full-array branch does.
    //
    //   FAST_TOPK - top-k is the first sampler in the chain that is not statically a no-op, so today's chain
    //       hands everything downstream exactly `top_k` candidates in descending order. The window below is
    //       built from the raw logits and is accepted only when the top k+1 logits are *strictly* decreasing;
    //       the descending order of the top k is then unique, so it cannot depend on how std::partial_sort
    //       happens to order equal logits. cur_p is handed to the chain with .sorted = true and .size = k, which
    //       is exactly the state llama_sampler_top_k_impl leaves behind (it then re-runs as a no-op:
    //       k = min(k, size) = size and !sorted is false). Nothing downstream is inspected: it sees the same
    //       bytes in the same order as today.
    //
    //   FAST_TOPK_PEN - FAST_TOPK with one *enabled* `penalties` sampler allowed in front of top-k; this is the
    //       production hands recipe (temp 0.7 / top-p 0.8 / top-k 20 / presence 1.5 / penalty_last_n 64), which
    //       FAST_TOPK has to refuse. llama_sampler_penalties_apply() (src/llama-sampler.cpp) only touches ids
    //       that are in the sampler's `token_count`, i.e. at most `penalty_last_n` distinct ids in the whole
    //       vocabulary, and under the gate below it can only *lower* a logit. The window is then the raw
    //       top-(k + penalty_last_n + 1), and it provably contains the post-penalty top-k of the vocabulary:
    //
    //         Let W be that window, m = k + penalty_last_n + 1, its raw logits strictly decreasing (first
    //         witness), L = the smallest of them, and P the penalised set, |P| <= penalty_last_n. For any token
    //         t outside W: post(t) <= raw(t) <= L (the penalty is non-increasing, so this covers t in P too).
    //         At least m - |P| >= k + 1 members of W are outside P and therefore keep their raw logit; call them
    //         u_1 > ... > u_{k+1} >= L - strictly decreasing, by the first witness. Then u_k > u_{k+1} >= L >=
    //         post(t), so k members of W beat t after the penalties: t cannot be in the post-penalty top-k, and
    //         the k-th largest post-penalty logit *of the window* is also the k-th largest of the vocabulary.
    //         Those same k also beat every remaining member of W, so the top-k is the same set on both paths;
    //         and once the retained top k+1 is strictly decreasing (second witness) it is the same *ordered*
    //         sequence too, whatever order std::partial_sort would have produced from the full array.
    //
    //       The window is handed to the chain RAW, at .size = m and .sorted = true, and the chain's own
    //       penalties instance then lowers what it lowers (bit for bit today's values - it is a pure function of
    //       token_count, applied per element) before top_k partial-sorts the m entries down to k. Everything
    //       downstream sees the same bytes in the same order as today. The second witness is taken beforehand on
    //       a scratch copy of the window by applying the same sampler to it; `apply` reads token_count and
    //       writes no sampler state, so running it twice has no side effect.
    //
    //       Gate (common_sampler_fast_mode_of): the penalties sampler must be the only non-no-op before top-k
    //       (so DRY / XTC / typical / top-n-sigma, wherever they sit before it, must be statically disabled -
    //       DRY in particular reads the whole array), penalty_last_n > 0, the window must fit in
    //       COMMON_SAMPLER_FAST_WINDOW_MAX, and the penalty must be non-increasing:
    //       penalty_repeat >= 1 && penalty_freq >= 0 && penalty_present >= 0.
    //
    //       That last condition is load-bearing and is NOT implied by the containment argument, which only ever
    //       reasons about tokens whose logit did not move. repeat_penalty flips sign at zero
    //       (`logit *= penalty_repeat` for logit <= 0, `logit /= penalty_repeat` above), so a repeat penalty
    //       below 1 - or a negative frequency / presence penalty - *raises* the logit of a penalised token, and
    //       a penalised token sitting far outside the window could then be lifted into the top-k, which no
    //       window built from the raw logits can see. With repeat >= 1 both branches are non-increasing
    //       (x <= 0: x*r <= x; x > 0: x/r <= x) and subtracting count*freq + present with freq, present >= 0
    //       only lowers it further.
    //
    // Known, documented differences (neither can be avoided without reproducing libstdc++ internals):
    //   * FAST_GREEDY and a `dist` draw of exactly 0.0: today's full-array branch selects cur_p->data[0] (the
    //     accumulator starts at 0.0 and `sum_run >= sum_tgt` is true at i == 0), i.e. token id 0 rather than the
    //     argmax; the 1-element branch returns the argmax. P(rnd == 0.0) ~ 2^-53 per token.
    //   * NaN logits: FAST_GREEDY's scan is temp_impl's scan, so it behaves identically; FAST_TOPK and
    //     FAST_TOPK_PEN bail to the slow path (the strict-decrease test fails on any NaN), where
    //     std::partial_sort's comparator is not a strict weak ordering and the outcome is undefined anyway.
    // ---------------------------------------------------------------------------------------------------------
    bool set_logits_fast(struct llama_context * ctx, int idx) {
        if (fast_mode == FAST_NONE || !fast_enabled) {
            return false;
        }

        // a backend sampler is attached -> the candidate window is whatever it returned, not the raw logits
        if (llama_get_sampled_probs_ith(ctx, idx) || llama_get_sampled_logits_ith(ctx, idx)) {
            return false;
        }

        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

        const int n_vocab = llama_vocab_n_tokens(vocab);

        const float * logits = llama_get_logits_ith(ctx, idx);
        GGML_ASSERT(logits != nullptr);

        bool ok = false;

        switch (fast_mode) {
            case FAST_GREEDY:   ok = set_logits_fast_greedy   (logits, n_vocab); break;
            case FAST_TOPK:     ok = set_logits_fast_top_k    (logits, n_vocab); break;
            case FAST_TOPK_PEN: ok = set_logits_fast_top_k_pen(logits, n_vocab); break;
            default:            return false;
        }

        if (!ok) {
            return false;
        }

        cur_p = { cur.data(), cur.size(), -1, /* .sorted = */ true };

        fast_used = true;

        g_smpl_n_fast.fetch_add(1, std::memory_order_relaxed);

        if (fast_mode == FAST_TOPK_PEN) {
            g_smpl_n_fast_pen.fetch_add(1, std::memory_order_relaxed);
        }

        return true;
    }

    // argmax of the raw logits, with llama_sampler_temp_impl's tie handling; refuses a non-unique maximum
    bool set_logits_fast_greedy(const float * logits, int n_vocab) {
        if (n_vocab <= 0) {
            return false;
        }

        int   max_i   = 0;
        float max_l   = logits[0];
        bool  max_uni = true;

        for (int i = 1; i < n_vocab; ++i) {
            const float l = logits[i];
            if (l > max_l) {
                max_i   = i;
                max_l   = l;
                max_uni = true;
            } else if (l == max_l) {
                max_uni = false;
            }
        }

        if (!max_uni) {
            return false;
        }

        if (fastpath_verify) {
            const float * ref = std::max_element(logits, logits + n_vocab);
            if (ref != logits + max_i) {
                LOG_ERR("%s: fast path (greedy) disagrees with std::max_element: %d vs %d\n",
                        __func__, max_i, (int) (ref - logits));
            }
        }

        cur.resize(1);
        cur[0] = llama_token_data{ max_i, max_l, 0.0f };

        return true;
    }

    // the top `m` raw logits, descending, into fast_heap; refuses any tie inside the window - with a tie the
    // descending order is not unique, so it could differ from the one std::partial_sort happens to produce
    bool build_raw_window(const float * logits, int n_vocab, int m) {
        if (m <= 0 || n_vocab < m) {
            return false;
        }

        // with this comparator (the partial-sort one), heap.front() is the *smallest* logit in the heap
        const auto comp = common_sampler_logit_desc;

        auto & heap = fast_heap;

        heap.clear();
        heap.reserve(m);

        for (int i = 0; i < m; ++i) {
            heap.push_back(llama_token_data{ i, logits[i], 0.0f });
        }

        std::make_heap(heap.begin(), heap.end(), comp);

        float thresh = heap.front().logit;

        for (int i = m; i < n_vocab; ++i) {
            const float l = logits[i];
            if (l > thresh) {
                std::pop_heap(heap.begin(), heap.end(), comp);
                heap.back() = llama_token_data{ i, l, 0.0f };
                std::push_heap(heap.begin(), heap.end(), comp);

                thresh = heap.front().logit;
            }
        }

        std::sort(heap.begin(), heap.end(), comp);

        // a tie anywhere in the window - or at its boundary - makes the order ambiguous: hand it to the slow path
        for (int i = 0; i + 1 < m; ++i) {
            if (!(heap[i].logit > heap[i + 1].logit)) {
                return false;
            }
        }

        return true;
    }

    // the top `fast_k` logits in descending order; refuses anything but a strictly decreasing top k+1
    bool set_logits_fast_top_k(const float * logits, int n_vocab) {
        // one witness past the window, to prove the boundary is not a tie
        if (!build_raw_window(logits, n_vocab, fast_k + 1)) {
            return false;
        }

        auto & heap = fast_heap;

        if (fastpath_verify) {
            std::vector<llama_token_data> ref(n_vocab);
            for (int i = 0; i < n_vocab; ++i) {
                ref[i] = llama_token_data{ i, logits[i], 0.0f };
            }
            std::partial_sort(ref.begin(), ref.begin() + fast_k, ref.end(), common_sampler_logit_desc);
            for (int i = 0; i < fast_k; ++i) {
                if (ref[i].id != heap[i].id || ref[i].logit != heap[i].logit) {
                    LOG_ERR("%s: fast path (top-k) disagrees with std::partial_sort at %d: "
                            "(%d, %g) vs (%d, %g)\n",
                            __func__, i, heap[i].id, (double) heap[i].logit, ref[i].id, (double) ref[i].logit);
                    break;
                }
            }
        }

        cur.assign(heap.begin(), heap.begin() + fast_k);

        return true;
    }

    // the raw top-(fast_k + penalty_last_n + 1), handed to the chain unpenalised: the chain's own penalties
    // sampler lowers what it lowers and top_k truncates. See the block comment above set_logits_fast().
    bool set_logits_fast_top_k_pen(const float * logits, int n_vocab) {
        if (fast_k <= 0 || fast_m < fast_k + 2 || fast_pen_idx < 0) {
            return false;
        }

        // first witness: the whole raw window is strictly decreasing (this is what makes u_k > u_{k+1} >= L hold)
        if (!build_raw_window(logits, n_vocab, fast_m)) {
            return false;
        }

        auto * pen = llama_sampler_chain_get(chain, fast_pen_idx);
        if (pen == nullptr) {
            return false;
        }

        // second witness: the post-penalty top k+1 must be strictly decreasing too, else the retained top-k is
        // not a unique ordered sequence and the two partial sorts could break the tie differently. `apply` is a
        // pure function of the sampler's token_count, so running it here as well changes no state.
        auto & scratch = fast_pen_scratch;

        scratch.assign(fast_heap.begin(), fast_heap.end());

        llama_token_data_array pen_p = { scratch.data(), scratch.size(), -1, /* .sorted = */ true };

        llama_sampler_apply(pen, &pen_p);

        std::sort(scratch.begin(), scratch.end(), common_sampler_logit_desc);

        for (int i = 0; i < fast_k; ++i) {
            if (!(scratch[i].logit > scratch[i + 1].logit)) {
                return false;
            }
        }

        if (fastpath_verify) {
            std::vector<llama_token_data> ref(n_vocab);
            for (int i = 0; i < n_vocab; ++i) {
                ref[i] = llama_token_data{ i, logits[i], 0.0f };
            }

            llama_token_data_array ref_p = { ref.data(), ref.size(), -1, /* .sorted = */ false };

            llama_sampler_apply(pen, &ref_p);

            std::partial_sort(ref.begin(), ref.begin() + fast_k, ref.end(), common_sampler_logit_desc);

            for (int i = 0; i < fast_k; ++i) {
                if (ref[i].id != scratch[i].id || ref[i].logit != scratch[i].logit) {
                    LOG_ERR("%s: fast path (top-k+penalties) disagrees with the penalised full vocabulary at %d: "
                            "(%d, %g) vs (%d, %g)\n",
                            __func__, i, scratch[i].id, (double) scratch[i].logit, ref[i].id, (double) ref[i].logit);
                    break;
                }
            }
        }

        // the chain gets the RAW window: it applies the real penalties sampler itself
        cur.assign(fast_heap.begin(), fast_heap.end());

        return true;
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;

    // G32 fast path state - see set_logits_fast()
    enum fast_mode_t {
        FAST_NONE = 0,
        FAST_GREEDY,
        FAST_TOPK,
        FAST_TOPK_PEN,
    };

    fast_mode_t fast_mode       = FAST_NONE;  // what the chain provably reduces to (decided in common_sampler_init)
    bool        fast_enabled    = false;      // caller opt-in: the candidate array is a window, not the vocabulary
    bool        fastpath_verify = false;
    bool        fast_used       = false;      // whether the last set_logits() built a window
    int         fast_k          = 0;
    int         fast_m          = 0;          // FAST_TOPK_PEN window: fast_k + penalty_last_n + 1
    int         fast_pen_idx    = -1;         // FAST_TOPK_PEN: index of the penalties sampler inside `chain`

    std::vector<llama_token_data> fast_heap;
    std::vector<llama_token_data> fast_pen_scratch;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

// G32: decide once what the chain provably reduces to. See the block comment above set_logits_fast().
//
// The "statically a no-op" predicates below mirror the early returns of the corresponding samplers in
// src/llama-sampler.cpp; keep them in sync (they are named there, next to the code they mirror).
static common_sampler::fast_mode_t common_sampler_fast_mode_of(
        const common_params_sampling & params,
        bool has_grmr,
        bool has_rbudget,
        bool has_logit_bias,
        int * pen_idx) {
    *pen_idx = -1;

    // these rewrite or reorder the logits before any window can be taken
    if (has_grmr || has_rbudget || has_logit_bias || params.backend_sampling) {
        return common_sampler::FAST_NONE;
    }

    // mirostat replaces params.samplers with its own two-sampler chain
    if (params.mirostat != 0) {
        return common_sampler::FAST_NONE;
    }

    // llama_sampler_penalties::is_disabled()
    const bool no_penalties = params.penalty_last_n == 0 ||
        (params.penalty_repeat == 1.0f && params.penalty_freq == 0.0f && params.penalty_present == 0.0f);
    // llama_sampler_dry_apply()
    const bool no_dry       = params.dry_multiplier == 0.0f || params.dry_base < 1.0f || params.dry_penalty_last_n == 0;
    // llama_sample_xtc_apply()
    const bool no_xtc       = params.xtc_probability <= 0.0f || params.xtc_threshold > 0.5f;
    // llama_sampler_typical_apply(), llama_sampler_top_n_sigma_apply(), llama_sampler_top_p_apply(),
    // llama_sampler_min_p_apply(), llama_sampler_top_k_impl()
    const bool no_typical   = params.typ_p       >= 1.0f;
    const bool no_top_n_sig = params.top_n_sigma <= 0.0f;
    const bool no_top_p     = params.top_p       >= 1.0f;
    const bool no_min_p     = params.min_p       <= 0.0f;
    const bool no_top_k     = params.top_k       <= 0;

    auto is_noop = [&](enum common_sampler_type cnstr) {
        switch (cnstr) {
            case COMMON_SAMPLER_TYPE_PENALTIES:  return no_penalties;
            case COMMON_SAMPLER_TYPE_DRY:        return no_dry;
            case COMMON_SAMPLER_TYPE_XTC:        return no_xtc;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:  return no_typical;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:return no_top_n_sig;
            case COMMON_SAMPLER_TYPE_TOP_P:      return no_top_p;
            case COMMON_SAMPLER_TYPE_MIN_P:      return no_min_p;
            case COMMON_SAMPLER_TYPE_TOP_K:      return no_top_k;
            default:                             return false;
        }
    };

    bool has_temp = false;
    for (const auto & cnstr : params.samplers) {
        has_temp |= cnstr == COMMON_SAMPLER_TYPE_TEMPERATURE;
    }

    // greedy: temp <= 0 turns the chain into "argmax of the raw logits" as long as nothing in it can reorder the
    // logits or drop the maximum. What is left after the no-op tests is top-k / top-p / min-p, and all three only
    // truncate a descending array, so the maximum always survives them - in any order.
    if (has_temp && params.temp <= 0.0f && params.dynatemp_range <= 0.0f) {
        bool ok = true;
        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_TOP_K:
                case COMMON_SAMPLER_TYPE_TOP_P:
                case COMMON_SAMPLER_TYPE_MIN_P:
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    break;
                default:
                    ok &= is_noop(cnstr);
            }
        }
        if (ok) {
            return common_sampler::FAST_GREEDY;
        }
    }

    // top-k: everything before top-k in the chain must be a no-op, and then the window handed downstream is
    // exactly the top k in descending order - whatever the rest of the chain happens to be.
    //
    // FAST_TOPK_PEN relaxes "everything" by exactly one enabled `penalties` sampler, and pays for it with a
    // window of k + penalty_last_n + 1 instead of k + 1. See the block comment above set_logits_fast().
    if (params.top_k > 0 && params.top_k <= 1024) {
        // llama_sampler_penalties_apply() is non-increasing only under this gate; without it a penalised token
        // outside the window can be *raised* into the top-k and no raw window can see it coming
        const bool pen_monotone = params.penalty_repeat  >= 1.0f &&
                                  params.penalty_freq    >= 0.0f &&
                                  params.penalty_present >= 0.0f;

        // written so that it cannot overflow for an absurd penalty_last_n (top_k <= 1024 here)
        const bool pen_windowable = !no_penalties && pen_monotone && params.penalty_last_n > 0 &&
            params.penalty_last_n <= COMMON_SAMPLER_FAST_WINDOW_MAX - 1 - params.top_k;

        // index inside the chain: every sampler type that can be reached before top-k here pushes exactly one
        // entry (adaptive-p, the one that does not, is never a no-op, so the loop breaks on it)
        int idx = 0;
        int pen = -1;

        for (const auto & cnstr : params.samplers) {
            if (cnstr == COMMON_SAMPLER_TYPE_TOP_K) {
                if (pen < 0) {
                    return common_sampler::FAST_TOPK;
                }

                *pen_idx = pen;

                return common_sampler::FAST_TOPK_PEN;
            }
            if (is_noop(cnstr)) {
                ++idx;
                continue;
            }
            // the single enabled penalties sampler the window is sized for; a second one would need a second
            // penalty_last_n of slack, so take only the first
            if (cnstr == COMMON_SAMPLER_TYPE_PENALTIES && pen < 0 && pen_windowable) {
                pen = idx++;
                continue;
            }
            break;
        }
    }

    return common_sampler::FAST_NONE;
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    bool has_logit_bias = false;
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));

            has_logit_bias = true;
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };

    // G32: what the chain reduces to is fixed by the parameters, so decide it once. Whether the fast path is
    // actually taken is the caller's call (common_sampler_set_fastpath): it truncates the candidate array, so it
    // must stay off for callers that read the full array back out of common_sampler_get_candidates().
    int fast_pen_idx = -1;

    result->fast_mode       = common_sampler_fast_mode_of(params, grmr != nullptr, rbudget != nullptr, has_logit_bias, &fast_pen_idx);
    result->fast_k          = params.top_k;
    // only meaningful (and only bounded) in FAST_TOPK_PEN, which gates the window to COMMON_SAMPLER_FAST_WINDOW_MAX
    result->fast_m          = result->fast_mode == common_sampler::FAST_TOPK_PEN
        ? params.top_k + params.penalty_last_n + 1
        : 0;
    result->fast_pen_idx    = fast_pen_idx;
    result->fastpath_verify = common_sampler_fastpath_env() == 2;

    // fast_pen_idx is derived from params.samplers; make sure it really does address the chain's penalties
    // sampler before anything relies on it
    if (result->fast_mode == common_sampler::FAST_TOPK_PEN) {
        llama_sampler * pen = (fast_pen_idx >= 0 && fast_pen_idx < llama_sampler_chain_n(chain))
            ? llama_sampler_chain_get(chain, fast_pen_idx)
            : nullptr;

        if (pen == nullptr || strcmp(llama_sampler_name(pen), "penalties") != 0) {
            LOG_WRN("%s: penalties sampler not found at chain index %d, disabling the sampling fast path\n",
                    __func__, fast_pen_idx);

            result->fast_mode    = common_sampler::FAST_NONE;
            result->fast_pen_idx = -1;
        }
    }

    return result;
}

void common_sampler_set_fastpath(struct common_sampler * gsmpl, bool enabled) {
    if (!gsmpl) {
        return;
    }

    gsmpl->fast_enabled = enabled && common_sampler_fastpath_env() != 0;
}

const char * common_sampler_fastpath_str(const struct common_sampler * gsmpl) {
    if (!gsmpl || !gsmpl->fast_enabled) {
        return "off";
    }

    switch (gsmpl->fast_mode) {
        case common_sampler::FAST_GREEDY:   return "greedy";
        case common_sampler::FAST_TOPK:     return "top-k";
        case common_sampler::FAST_TOPK_PEN: return "top-k+penalties";
        default:                            return "none";
    }
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    auto * result = new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
    };

    // a clone must sample exactly like the original, so it inherits the fast-path decision too
    result->fast_mode       = gsmpl->fast_mode;
    result->fast_enabled    = gsmpl->fast_enabled;
    result->fastpath_verify = gsmpl->fastpath_verify;
    result->fast_used       = gsmpl->fast_used;
    result->fast_k          = gsmpl->fast_k;
    result->fast_m          = gsmpl->fast_m;
    result->fast_pen_idx    = gsmpl->fast_pen_idx;

    return result;
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->t_total_us = src->t_total_us;

    dst->fast_mode       = src->fast_mode;
    dst->fast_enabled    = src->fast_enabled;
    dst->fastpath_verify = src->fastpath_verify;
    dst->fast_used       = src->fast_used;
    dst->fast_k          = src->fast_k;
    dst->fast_m          = src->fast_m;
    dst->fast_pen_idx    = src->fast_pen_idx;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

// G32: accumulate the wall time spent in common_sampler_sample() process-wide, so that a server can report it
// without a profiler (llamacpp:sampling_seconds_total). Two clock reads per sample call.
struct common_sampler_perf_meas {
    common_sampler_perf_meas() : t_start_us(ggml_time_us()) {}

    ~common_sampler_perf_meas() {
        g_smpl_t_us.fetch_add((uint64_t) (ggml_time_us() - t_start_us), std::memory_order_relaxed);
        g_smpl_n   .fetch_add(1,                                        std::memory_order_relaxed);
    }

    const int64_t t_start_us;
};

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    const common_sampler_perf_meas pm;

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            // a backend sampler that returns only the token (e.g. greedy) leaves the raw-logits branch in place,
            // so the window may not contain the token it picked - rebuild the full candidate array to find it
            if (gsmpl->fast_used) {
                gsmpl->set_logits_full(ctx, idx);
            }

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}
