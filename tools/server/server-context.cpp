#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-disk-cache.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "../../src/llama-ext.h" // llama_get_ctx_other (v33 B2: shared-memory drafts keep the draft copy)

#include <optional>
#include <algorithm>
#include <cstddef>
#include <cinttypes>
#include <cerrno>
#include <exception>
#include <memory>
#include <filesystem>
#include <random>
#include <utility>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstdlib>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// [TAG_MULTISLOT] stage 5.1 item 2 (see server-context.h). The knob is parsed strictly like LLAMA_MTP_CARRIER_ZERO:
// only a plain 0 / 1; anything else is reported in the proof and treated as unset (never as qualification mode)
server_multislot_envelope server_multislot_envelope_check(const char * knob, int n_parallel, bool has_mmproj, bool ctx_shift) {
    server_multislot_envelope r;
    const bool knob_set   = knob != nullptr && knob[0] != '\0';
    const bool knob_one   = knob_set && strcmp(knob, "1") == 0;
    const bool knob_valid = !knob_set || knob_one || strcmp(knob, "0") == 0;
    if (n_parallel <= 1) {
        return r; // single slot: unchanged, nothing printed
    }
    r.strict = knob_one;
    r.proof  = string_format("[multislot-envelope] np=%d text_only=%d shared_prefix=%d ctx_shift=%d strict=%d",
                             n_parallel, has_mmproj ? 0 : 1, r.strict ? 0 : 1, ctx_shift ? 1 : 0, r.strict ? 1 : 0);
    if (!knob_valid) {
        r.proof += string_format(" (invalid LLAMA_QSA_HOMOGENEOUS='%s', 0 or 1 - treated as unset)", knob);
    }
    if (!r.strict) {
        r.proof += " - multi-slot exactness is NOT qualified outside LLAMA_QSA_HOMOGENEOUS=1 (mmproj, shared-prompt requests, context shift allowed)";
        return r;
    }
    if (has_mmproj) {
        r.ok    = false;
        r.error = string_format("[multislot-envelope] LLAMA_QSA_HOMOGENEOUS=1 with n_parallel=%d refuses --mmproj (text-only envelope)", n_parallel);
    } else if (ctx_shift) {
        r.ok    = false;
        r.error = string_format("[multislot-envelope] LLAMA_QSA_HOMOGENEOUS=1 with n_parallel=%d requires --no-context-shift", n_parallel);
    }
    return r;
}

// [TAG_BS_GUARD] 0226 (see server-context.h)
server_backend_sampling_support server_backend_sampling_check(const common_params & params) {
    server_backend_sampling_support r;

    std::vector<common_speculative_type> types;
    for (const auto t : params.speculative.types) {
        if (t != COMMON_SPECULATIVE_TYPE_NONE) {
            types.push_back(t);
        }
    }
    if (!types.empty()) {
        r.spec_type = common_speculative_type_name_str(types);
        // the same ceiling server_output_limits sizes the target with: MTP / draft-simple / eagle3 report at least 1
        // (they append a token before testing n_max), dflash / dspark / ngram their configured maximum
        r.n_max = common_speculative_n_max(&params.speculative);
    }

    // the target context's per-sequence output limit as llama_context resolves it from these params
    // (n_outputs_max 0 = n_batch; n_outputs_max_per_seq 0 = n_outputs_max, else capped by it)
    const auto lim = server_output_limits(params);
    const int32_t total = lim.total > 0 ? lim.total : params.n_batch;
    r.per_seq_max = lim.per_seq > 0 ? std::min(lim.per_seq, total) : total;

    if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        r.supported = false;
        r.error     = "backend_sampling is not supported on this server: tensor split";
        return r;
    }

    const int32_t need = 1 + r.n_max; // one verify round: the sampled token + the drafts, all in one sequence
    if (r.per_seq_max < need) {
        r.supported = false;
        r.error     = string_format("backend_sampling is not supported on this server: speculative verify needs 1+%d "
                                    "outputs per sequence but the context allows %d", r.n_max, r.per_seq_max);
    }

    return r;
}

// [TAG_MULTISLOT] (stage 5.1 item 4a, see server-context.h) the routing-class planner of LLAMA_QSA_HOMOGENEOUS=1
int server_qsa_route_class(int64_t d_lo, int64_t d_hi, int64_t n_min) {
    if (d_lo >= n_min) {
        return 1; // above (equality at N is above; N = 0: everything is above)
    }
    if (d_hi < n_min) {
        return 0; // below
    }
    return -1;    // straddles N: a singleton
}

std::vector<std::vector<int32_t>> server_qsa_route_groups(const std::vector<server_qsa_route_run> & runs, int64_t n_min, int64_t max_ub) {
    std::vector<int32_t> by_id(runs.size());
    for (size_t r = 0; r < runs.size(); ++r) {
        by_id[r] = (int32_t) r;
    }
    std::stable_sort(by_id.begin(), by_id.end(), [&](int32_t a, int32_t b) { return runs[a].id_slot < runs[b].id_slot; });

    std::vector<int>  cls(runs.size());
    std::vector<bool> big(runs.size());
    for (size_t r = 0; r < runs.size(); ++r) {
        cls[r] = runs[r].partial ? -1 : server_qsa_route_class((int64_t) runs[r].p_lo + 1, (int64_t) runs[r].p_hi + 1, n_min);
        big[r] = max_ub > 0 && (int64_t) runs[r].n_rows > max_ub;
    }

    std::vector<std::pair<int32_t, std::vector<int32_t>>> tagged; // (first original row, runs)
    for (int key = 0; key < 4; ++key) {
        const int  c = key >> 1;
        const bool b = key & 1;
        std::vector<int32_t> cur;
        int64_t cur_rows = 0;
        auto flush = [&]() {
            if (!cur.empty()) {
                int32_t first = INT32_MAX;
                for (const int32_t r : cur) {
                    first = std::min(first, runs[r].first_row);
                }
                tagged.emplace_back(first, cur);
                cur.clear();
                cur_rows = 0;
            }
        };
        for (const int32_t r : by_id) {
            if (cls[r] != c || big[r] != b) {
                continue;
            }
            const int64_t n = runs[r].n_rows;
            if (!b && max_ub > 0 && cur_rows + n > max_ub) {
                flush();
            }
            cur.push_back(r);
            cur_rows += n;
        }
        flush();
    }
    for (size_t r = 0; r < runs.size(); ++r) {
        if (cls[r] < 0) {
            tagged.emplace_back(runs[r].first_row, std::vector<int32_t>{ (int32_t) r });
        }
    }
    std::stable_sort(tagged.begin(), tagged.end(), [](const auto & a, const auto & b) { return a.first < b.first; });

    std::vector<std::vector<int32_t>> groups;
    for (auto & t : tagged) {
        groups.push_back(std::move(t.second));
    }
    return groups;
}

int64_t server_qsa_route_target_cap(int64_t max_ub, const char * env_target_cap, const char * env_catchup_attn) {
    const bool knob      = env_target_cap != nullptr && strcmp(env_target_cap, "1") == 0;
    const bool attn_free = env_catchup_attn != nullptr && strcmp(env_catchup_attn, "0") == 0;
    return knob || !attn_free ? max_ub : 0;
}

// [TAG_MULTISLOT] (stage 5.1 item 5b, see server-context.h) the operation-start gate
const char * server_adm_op_name(server_adm_op op) {
    switch (op) {
        case SERVER_ADM_OP_DRAFT:        return "draft";
        case SERVER_ADM_OP_RESTORE_RAM:  return "restore_ram";
        case SERVER_ADM_OP_RESTORE_DISK: return "restore_disk";
        case SERVER_ADM_OP_RESTORE_SLOT: return "restore_slot";
        default:                         return "unknown";
    }
}

server_adm_op_decision server_admission_op_check(bool enabled, double bar_mib, double reserve_mib, bool concurrent,
                                                 const std::string & used_path) {
    server_adm_op_decision d;
    if (!enabled || reserve_mib <= 0.0 || !concurrent) {
        return d;
    }
    std::ifstream f(used_path);
    uint64_t used_b = 0;
    if (!f || !(f >> used_b)) {
        return d; // unreadable: start (the prompt admission disables itself on its next check and logs it)
    }
    d.gated         = true;
    d.used_mib      = used_b / 1024.0 / 1024.0;
    d.projected_mib = d.used_mib + reserve_mib;
    d.start         = d.projected_mib <= bar_mib;
    return d;
}

// [TAG_SPEC_TRACE_IDS] LLAMA_SPEC_TRACE_IDS=1: read once
static bool server_spec_trace_ids() {
    static const bool on = [] {
        const char * s = getenv("LLAMA_SPEC_TRACE_IDS");
        const bool v = s != nullptr && atoi(s) != 0;
        if (v) {
            fprintf(stderr, "[spec-ids] on (LLAMA_SPEC_TRACE_IDS=1): one line per verify round\n");
        }
        return v;
    }();
    return on;
}

// [TAG_B68_DRAFT_KNOBS] strict integer env knob in [lo, hi]; unset / invalid -> 0 (off), read once with a proof line
static int server_env_int(const char * name, int lo, int hi, bool print_on = true) {
    const char * s = getenv(name);
    if (s == nullptr) {
        return 0;
    }
    char * end = nullptr;
    errno = 0;
    const long v = (*s >= '0' && *s <= '9') ? strtol(s, &end, 10) : -1;
    if (end != nullptr && end != s && *end == '\0' && errno == 0 && v >= lo && v <= hi) {
        if (print_on) {
            SRV_WRN("%s=%ld: on\n", name, v);
        }
        return (int) v;
    }
    SRV_WRN("invalid %s='%s' (an integer in [%d, %d]) - stays off\n", name, s, lo, hi);
    return 0;
}

// H2 LLAMA_SPEC_DRAFT_DEPTH_AUTO=d0: per-slot draft depth from the acceptance history, starting at d0: +1 after a round
//   whose full-depth draft was accepted whole, -1 (floor LLAMA_SPEC_DRAFT_DEPTH_MIN, default 2, and --spec-draft-n-min) after a rejection;
//   bounded by --spec-draft-n-max; restarts at d0 with every request. H5a LLAMA_SPEC_DRAFT_N_MAX_GREEDY=n: rounds verified
//   by sample-and-match (T = 0, or no U7 candidates) draft at most n tokens, so --spec-draft-n-max can be raised for the
//   sampled (U7) rounds alone. The cap is clamped to [--spec-draft-n-min, --spec-draft-n-max]: below n_min the drafter
//   would throw every capped draft away (speculation silently off at T = 0); the proof line names the clamp.
static int server_spec_depth_auto() {
    static const int v = server_env_int("LLAMA_SPEC_DRAFT_DEPTH_AUTO", 1, 64);
    return v;
}
static int server_spec_depth_min() {
    static const int v = [] { const int m = server_env_int("LLAMA_SPEC_DRAFT_DEPTH_MIN", 1, 64); return m > 0 ? m : 2; }();
    return v;
}
static int server_spec_n_max_greedy(int n_min, int n_max) {
    static const int v = [n_min, n_max] {
        const char * name = "LLAMA_SPEC_DRAFT_N_MAX_GREEDY";
        const int g = server_env_int(name, 1, 64, /*print_on =*/ false);
        if (g <= 0) {
            return 0;
        }
        int c = std::max(g, n_min);
        if (n_max > 0) {
            c = std::min(c, n_max);
        }
        if (c > g) {
            SRV_WRN("%s=%d: on, clamped to n_min %d\n", name, g, c);
        } else if (c < g) {
            SRV_WRN("%s=%d: on, clamped to n_max %d\n", name, g, c);
        } else {
            SRV_WRN("%s=%d: on\n", name, g);
        }
        return c;
    }();
    return v;
}

// Replay after a checkpoint restore (partial draft acceptance on a context that cannot seq_rm partially).
//
// The leading `n_committed` tokens of `draft` were already sampled AND accepted on this very sampler during the
// verification round - common_sampler_sample_and_accept_n() returns exactly the tokens it accepted, and that is
// what the caller stored in spec_draft/spec_n_committed. The verification round then returned early, before
// emitting them, so the replay round is what emits them. The live sampler therefore already holds the correct
// post-verification state and the caller must NOT restore a pre-verification snapshot over it.
//
// So the committed tokens are neither re-sampled nor re-accepted here:
//   - re-sampling is unsafe: the replay batch has a different size than the verification batch, so the
//     recomputed logits are not bit-identical and a near-tie can flip the argmax (or a temperature > 0 sample
//     can differ) -> the token is "rejected" -> restore -> replay -> rejected again, forever;
//   - re-accepting would advance the grammar, the reasoning budget and the sampling chain (mirostat mu, the
//     dist sampler's RNG, ...) a second time for a token that was already accepted.
//
// Invariant: every emitted token is accepted on the live sampler exactly once - the committed ones during
// verification, the final one here.
//
// `probe_mismatch` (LLAMA_TRACE only) counts how many committed tokens the target would pick differently now.
// It samples a CLONE of the sampler, never the live one, so turning the trace on cannot change the generated
// output. The clone starts from the post-verification state, so for history-dependent samplers (penalties,
// grammar) the probe is only an approximation - it is a diagnostic counter, not a correctness check.
std::vector<llama_token> server_sample_and_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int> & idxs,
        const llama_tokens & draft,
        size_t n_committed,       // leading tokens already sampled and accepted during verification: just re-emit
        bool probe_mismatch,      // LLAMA_TRACE only: re-sample the committed tokens on a clone to count rounding flips
        int & n_mismatch) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(n_committed <= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());
    n_mismatch = 0;

    // diagnostics only - the live sampler is never sampled at a committed position
    common_sampler_ptr smpl_probe(probe_mismatch && n_committed > 0 ? common_sampler_clone(smpl) : nullptr);

    for (size_t i = 0; i < n_committed; i++) {
        if (smpl_probe) {
            if (common_sampler_sample(smpl_probe.get(), ctx, idxs[i]) != draft[i]) {
                n_mismatch++;
            }
            common_sampler_accept(smpl_probe.get(), draft[i], true);
        }
        result.push_back(draft[i]);
    }

    // verify the (optional) uncommitted tail exactly like common_sampler_sample_and_accept_n
    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    for (size_t i = n_committed; i < idxs.size(); i++) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        // [TAG_UP29638] no draft token after an EOG either (a trailing EOG is kept, as there)
        if (i < draft.size() && (id != draft[i] || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size()))) {
            break;
        }
    }

    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// the leading `n_committed` tokens are a replay after a checkpoint restore: they were already sampled and accepted
// during verification (see server_sample_and_accept_replay), so they are only re-emitted here, never re-sampled and
// never re-accepted. In particular the last committed token no longer needs a special "accept with is_generated =
// true" case to advance the grammar / reasoning state - verification already accepted it exactly that way.
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        size_t n_committed) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());
    GGML_ASSERT(n_committed <= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    for (size_t i = 0; i < n_committed; ++i) {
        result.push_back(draft[i]);
    }

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = n_committed; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        if (accept && !llama_vocab_is_eog(vocab, draft[i])) {
            // synthetic draft tokens do not advance grammar or reasoning state
            common_sampler_accept(smpl, draft[i], false);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    // [TAG_MULTISLOT] (stage 5.1 item 4a) homogeneous routing ranges of LLAMA_QSA_HOMOGENEOUS=1 (np > 1): the end
    //   offsets (ascending) of the ranges each decode call is clamped to, and of the whole-slot runs inside them (the
    //   capacity-retry cut points). Empty = one range = today's chunking.
    std::vector<int32_t> range_end;
    std::vector<int32_t> run_end;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output, is_prompt });
        return true;
    }

    bool add(int32_t id_slot, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        range_end.clear();
        run_end.clear();
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    // [TAG_MULTISLOT] (item 4a) reorder the unrendered rows: order[new] = old; embedding rows move with their token
    //   rows. Returns map[old] = new (the caller remaps every stored batch index through it).
    std::vector<int32_t> permute(const std::vector<int32_t> & order) {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT((int32_t) order.size() == size());
        std::vector<int32_t> map(order.size(), -1);
        std::vector<token> tokens_new;
        tokens_new.reserve(tokens.size());
        std::vector<float> embd_new;
        if (has_embd) {
            GGML_ASSERT(embd.size() == tokens.size() * (size_t) n_embd);
            embd_new.reserve(embd.size());
        }
        for (size_t i = 0; i < order.size(); ++i) {
            const int32_t o = order[i];
            GGML_ASSERT(o >= 0 && o < size() && map[o] == -1);
            map[o] = (int32_t) i;
            tokens_new.push_back(tokens[o]);
            if (has_embd) {
                embd_new.insert(embd_new.end(), embd.begin() + (size_t) o * n_embd, embd.begin() + (size_t) (o + 1) * n_embd);
            }
        }
        tokens.swap(tokens_new);
        if (has_embd) {
            embd.swap(embd_new);
        }
        return map;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;

    // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
    std::vector<std::vector<llama_token_data>> spec_draft_q;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    // [TAG_UP27694] the round whose committed tokens a replay re-emits was verified by rejection sampling
    bool spec_round_rejection = false;
    size_t spec_n_committed = 0; // number of committed (already-accepted) tokens at the front of spec_draft this round

    // [TAG_GTT_ADMISSION] (v33 #13) prompt admission state of this slot
    bool    adm_deferred       = false; // the last admission check of the current prompt deferred it
    int64_t adm_t_defer_us     = 0;     // start of the current deferral
    int64_t adm_t_log_us       = 0;     // last deferral log line (rate limit)
    double  adm_projected_mib  = 0.0;   // projection of the last check

    // [TAG_B0_RESTORE_LOG] (v33 B0) per-task restore telemetry, logged once the prompt's first batch is set up
    bool    b0_pending         = false;
    int32_t b0_n_cached_live   = 0;     // tokens held by the slot before the task (the live state)
    int32_t b0_prefix_match    = 0;     // common prefix of the live tokens and the new prompt (after cache reuse)
    int64_t b0_ckpt_n_tokens   = -1;    // n_tokens of the restored checkpoint, -1 = none restored
    bool    b0_full_reset      = false; // no usable checkpoint -> full re-processing

    // [TAG_CKPT_PIN_RESTORED] (v33 fix round 2) n_tokens of the checkpoint this task restored (and kept), -1 = none;
    //   the capacity eviction in create_checkpoint() skips it for the rest of the task
    int64_t ckpt_pinned_n_tokens = -1;

    // [TAG_DEEP_CKPT] a user-message start this task branched at after a deep-checkpoint rewind: the replay takes a
    //   checkpoint there and stores it, so the next branch from the same point is a rewind, not a replay; -1 = none
    int64_t ckpt_branch_pos = -1;
    // [TAG_CACHE_PIN] (SC1) the request's cache_pin position still to be reached by this prompt processing (a split
    //   and a checkpoint there, stored as the conversation's pin); -1 = none
    int64_t ckpt_pin_pos = -1;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;
    bool release_cancel = false; // [TAG_MULTISLOT] released by a SERVER_TASK_TYPE_CANCEL (logged as stop=cancel)

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft, state_desc());
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
        // [TAG_CACHE_COMPAT2] the speculative carry belongs to this prompt: it travels with the entry (one carry is
        //   serializable; with more stateful implementations it stays empty and the swap-in resets the draft side)
        if (spec && common_speculative_n_stateful(spec) <= 1) {
            common_speculative_get_state(spec, id, cur->data.spec);
        }

        return true;
    }

    // [TAG_DISK_PROMPT_CACHE] keep this slot's prompt on disk before it is overwritten
    bool disk_save(server_disk_cache & disk_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        return disk_cache.save(ctx_tgt, ctx_dft, spec, id, prompt);
    }

    server_disk_load_result disk_load(server_disk_cache & disk_cache, int idx) {
        prompt_clear(); // both models' memory of this sequence, through the slot's own memory wrapper

        return disk_cache.load(idx, ctx_tgt, ctx_dft, spec, id, prompt);
    }

    // [TAG_CACHE_COMPAT2] the RAM cache's descriptor guard (server_cache_state_desc)
    std::string state_desc() const {
        return server_cache_state_desc(ctx_tgt, ctx_dft, spec ? common_speculative_state_schema(spec) : std::string());
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        std::vector<uint8_t> carry;
        bool swapped = false;
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id, state_desc(), &carry, &swapped);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
            if (spec) {
                common_speculative_reset_state(spec, id); // [TAG_SPEC_RESET] the caller clears the sequence
            }
        } else if (swapped && spec && !common_speculative_set_state(spec, id, carry)) {
            // [TAG_CACHE_COMPAT2] (astra compat2 r2) a RAM swap-in replaced the target + draft states: the drafter's
            //   carry must be the swapped-in prompt's. Not restorable (none saved, several stateful implementations,
            //   schema mismatch): drop the draft sequence and the carry. [TAG_DRAFT_PARTIAL] the target state is
            //   canonical, only the draft history is gone: the prompt is draft-partial (gap = n_tokens, as the
            //   target-only adopted load), so it stays reusable and saveable (draft_cover = the rows the draft
            //   computes after the swap-in) - not `partial`
            common_speculative_reset_state(spec, id);
            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), id, -1, -1);
                prompt.draft_gap = (int32_t) prompt.tokens.size();
            }
            SLT_INF(*this, "RAM swap-in of %zu tokens: speculative carry not restorable (%zu bytes) - draft sequence "
                    "and carry reset%s\n", prompt.tokens.size(), carry.size(),
                    prompt.draft_partial() ? ", the prompt is draft-partial" : "");
        } else if (swapped && spec) {
            SLT_DBG(*this, "RAM swap-in of %zu tokens: speculative carry restored (%zu bytes)\n", prompt.tokens.size(),
                    carry.size());
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        mem.seq_rm(id, -1, -1);

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    // --- adaptive draft length (--spec-draft-n-max-adaptive) ------------------------------------
    //
    // A verification decode of D+1 tokens costs t_dec + D*t_tok, while every accepted draft token
    // saves a whole plain decode (t_dec).
    // Drafting position d (1-based) costs t_tok unconditionally, but it only pays off when the whole
    // prefix 1..d is accepted, i.e. with probability P_d = prod_{i<=d} p_i, where
    // p_i = P(pos i accepted | 1..i-1 accepted). So we draft the longest prefix whose marginal value
    // is positive: P_d * t_dec > t_tok, i.e. P_d > r with r = t_tok/t_dec.
    // p_i is estimated per slot as an EMA over this slot's verification rounds (window ~32,
    // alpha = 1/32), initialized optimistically at 1.0, so a slot drafts the configured n-max until
    // its own acceptance says otherwise, and re-adapts within ~32 rounds when the text changes kind.
    // The result is clamped to [1, configured n-max], so the recurrent-state rollback depth
    // (n_rs_seq = n-max, allocated at startup) is never exceeded.
    //
    // [TAG_SPEC_ADAPTIVE_DEPTH] t_tok and t_dec are NOT constants: each drafted token adds one more
    // KV pass to the verification graph (and to the draft loop), so both grow with the KV depth.
    // The server therefore evaluates the affine model
    //     t_tok(n_kv) = t_tok0 + k_tok * n_kv ,  t_dec(n_kv) = t_dec0 + k_dec * n_kv
    // at the slot's current depth, and prefers its own measured round times when they are available
    // (see server_context_impl::spec_cost). Because r(n_kv) rises with n_kv, the chosen draft length
    // shrinks with depth - which is the point: at 213k a drafted token costs a much larger fraction
    // of the decode it might save than at 8k.
    //
    // Quality: this only changes HOW MANY tokens are drafted. Verification and acceptance are
    // untouched (greedy accept-if-equal at temp 0, unchanged rejection sampling above it), so the
    // emitted distribution is unchanged; only the round structure differs.
    std::vector<float> spec_acc_ema;    // p_i per draft position, EMA
    std::vector<uint32_t> spec_acc_nobs; // [TAG_SPEC_DRAFT_POLICY] observations per position (0 = never reached)
    int                spec_n_max_last = 0; // last adaptive draft length (also reported in /slots)

    // last cost model evaluation, for /slots and /metrics (diagnostics only)
    float spec_cost_t_tok    = 0.0f;
    float spec_cost_t_dec    = 0.0f;
    int   spec_cost_n_kv     = 0;
    bool  spec_cost_measured = false;

    // [TAG_SPEC_DRAFT_POLICY] --spec-draft-policy tput: the cooldown state and the length requested this round
    common_spec_cooldown spec_cd;
    int                  spec_tput_n_req = 0;
    int                  spec_tput_n_att = 0;   // draft positions the drafter sampled this round (p_min stop included)
    int                  spec_tput_n_kept = 0;  // tokens it drafted before the n_min rule
    int                  spec_tput_n_draft_rounds = 0; // drafting rounds of this task (the first one is cold)
    bool                 spec_tput_ranked = false; // this round's bucket had rank (else `off`: no cooldown observation)
    bool                 spec_tput_probe  = false; // this round is a cooldown probe (its draft is kept below n_min)
    // the last decision's state for /slots (also at n = 0): the rule's n*, the cost model of the slot's depth bucket
    bool                  spec_tput_active   = false;
    bool                  spec_tput_measured = false;
    int                   spec_tput_rule     = 0;
    int                   spec_tput_n_kv     = 0;
    double                spec_tput_rate     = 0.0;
    common_spec_tput_cost spec_tput_cost;
    int                   spec_tput_fit_n[8] = { 0 }; // samples: draft, ckpt, verify, catchup, host, host_d, rb, restore
    int64_t              spec_rb_restore_us = 0; // the checkpoint restore of the last rollback (charged to its replay)

    static constexpr float SPEC_ACC_EMA_ALPHA = 1.0f/32.0f;

    // [TAG_B68_DRAFT_KNOBS] H2 the current auto depth (0 = not started; restarts at every request so that a seeded
    //   T > 0 request replays exactly - the depth changes how the rejection sampler consumes its RNG) and the depth
    //   requested in the round being verified
    int spec_depth_cur = 0;
    int spec_depth_req = 0;

    // Update the per-position conditional acceptance from one verification round.
    //
    // Called once per verified round with that round's FRESH draft tail only: n_draft is the
    // number of tokens drafted this round (spec_draft minus the leading committed tokens a
    // replay/fold round re-emits) and n_accepted how many of them matched.  A pure replay round
    // has no fresh tail, so n_draft == 0 and this returns immediately.  The call site is above
    // the checkpoint-restore branch so that a round which ends in a restore - i.e. a rejection -
    // is observed exactly like one that does not.
    void spec_acc_update(size_t n_draft, size_t n_accepted, int n_max_cfg) {
        if (n_draft == 0 || n_max_cfg <= 0) {
            return;
        }

        if (spec_acc_ema.empty()) {
            spec_acc_ema.assign(n_max_cfg, 1.0f);
        }

        // positions 0..n_accepted-1 were accepted; position n_accepted (if it was drafted) was
        // rejected. positions after a rejection are not observed - they follow a broken prefix.
        spec_acc_nobs.resize(spec_acc_ema.size(), 0);
        for (size_t i = 0; i < n_draft && i <= n_accepted && i < spec_acc_ema.size(); ++i) {
            const float obs = (i < n_accepted) ? 1.0f : 0.0f;
            spec_acc_ema[i] += SPEC_ACC_EMA_ALPHA * (obs - spec_acc_ema[i]);
            spec_acc_nobs[i]++;
        }
    }

    // [TAG_SPEC_DRAFT_POLICY] a drafter stop at position k with positions < k unverified (a draft the n_min rule
    //   cleared): only the stop itself is observed, as a failure at k
    void spec_acc_observe_stop(size_t k, int n_max_cfg) {
        if (n_max_cfg <= 0) {
            return;
        }
        if (spec_acc_ema.empty()) {
            spec_acc_ema.assign(n_max_cfg, 1.0f);
        }
        spec_acc_nobs.resize(spec_acc_ema.size(), 0);
        if (k < spec_acc_ema.size()) {
            spec_acc_ema[k] += SPEC_ACC_EMA_ALPHA * (0.0f - spec_acc_ema[k]);
            spec_acc_nobs[k]++;
        }
    }

    // the draft length to use for the next round, in [1, n_max_cfg]
    int spec_n_max_adaptive(int n_max_cfg, float t_tok, float t_dec, float * p_cum_out = nullptr) const {
        float p_cum = 1.0f;
        int   d     = 0;

        for (int i = 0; i < n_max_cfg; ++i) {
            p_cum *= (i < (int) spec_acc_ema.size()) ? spec_acc_ema[i] : 1.0f;

            if (p_cum * t_dec <= t_tok) {
                break; // the marginal value of drafting this position is negative
            }

            d = i + 1;
        }

        if (p_cum_out) {
            *p_cum_out = p_cum;
        }

        return std::max(1, d);
    }

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    // [TAG_SPEC_TRACE_IDS] the verify-round counter of the current request (LLAMA_SPEC_TRACE_IDS)
    uint64_t spec_trace_round = 0;

    void reset() {
        spec_trace_round = 0;
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;
        spec_round_rejection = false;
        spec_n_committed = 0;

        adm_deferred      = false;
        adm_t_defer_us    = 0;
        adm_t_log_us      = 0;
        adm_projected_mib = 0.0;

        b0_pending       = false;
        b0_n_cached_live = 0;
        b0_prefix_match  = 0;
        b0_ckpt_n_tokens = -1;
        b0_full_reset    = false;

        ckpt_pinned_n_tokens = -1;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        release_cancel = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();
        spec_acc_ema.clear();
        spec_acc_nobs.clear();
        spec_n_max_last    = 0;
        spec_depth_cur     = 0; // [TAG_B68_DRAFT_KNOBS] H2 restarts per request: a seeded T > 0 request replays exactly
        spec_depth_req     = 0;
        spec_cost_t_tok    = 0.0f;
        spec_cost_t_dec    = 0.0f;
        spec_cost_n_kv     = 0;
        spec_cost_measured = false;
        spec_cd            = common_spec_cooldown();
        spec_tput_n_req    = 0;
        spec_tput_n_att    = 0;
        spec_tput_n_kept   = 0;
        spec_tput_n_draft_rounds = 0;
        spec_tput_ranked   = false;
        spec_tput_probe    = false;
        spec_tput_active   = false;
        spec_tput_measured = false;
        spec_tput_rule     = 0;
        spec_tput_n_kv     = 0;
        spec_tput_rate     = 0.0;
        spec_tput_cost     = common_spec_tput_cost();
        std::fill(std::begin(spec_tput_fit_n), std::end(spec_tput_fit_n), 0);
        spec_rb_restore_us = 0;

        n_predict_max = -1;

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
    // [TAG_UP27694] stateful samplers (mirostat v1/v2, adaptive-p) would update their state with the chain's own
    //   sample instead of the emitted token inside the rejection verifier: those keep sample-and-match
    bool use_spec_rejection() const {
        return task && task->params.sampling.temp > 0.0f &&
               task->params.sampling.mirostat == 0 && task->params.sampling.adaptive_target < 0.0f;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            stats.n_spec_plain_rounds += 1;

            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(id, inp_embd, prompt.tokens.pos_next(), true, false);
            } else {
                add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            if (spec_n_committed == 0) {
                stats.n_spec_verify_rounds += 1;
            } else if (spec_n_committed == spec_draft.size()) {
                stats.n_spec_replay_rounds += 1;
            } else {
                stats.n_spec_fold_rounds += 1;
            }

            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            // [TAG_MULTISLOT] one INFO line per finished generation with the stop type, so a cut reply can be
            // classified from the INFO log (stable format for grep: stop=<eos|limit|word|cancel|none> n_gen= n_predict= truncated=)
            const char * stop_str = "none";
            if (release_cancel) {
                stop_str = "cancel";
            } else {
                switch (stop) {
                    case STOP_TYPE_EOS:   stop_str = "eos";   break;
                    case STOP_TYPE_WORD:  stop_str = "word";  break;
                    case STOP_TYPE_LIMIT: stop_str = "limit"; break;
                    case STOP_TYPE_NONE:  stop_str = "none";  break;
                }
            }
            SLT_INF(*this, "stop processing: n_tokens = %d, stop=%s n_gen=%d n_predict=%d truncated=%d\n",
                    prompt.n_tokens(), stop_str, (int) stats.n_gen, (int) task->params.n_predict, truncated ? 1 : 0);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            // nor of the stateless slots - no task ever reuses it
            if (task->is_child() || task->is_stateless()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;
            if (stats.n_draft_post_eog > 0) {
                SLT_INF(*this, "draft tokens after a drafted EOG = %" PRIu64 " (never acceptable; LLAMA_SPEC_DRAFT_STOP_EOG=1 ends the chain there)\n", stats.n_draft_post_eog);
            }
            if (stats.n_draft_eog_rejected > 0) {
                SLT_INF(*this, "drafted EOG not accepted = %" PRIu64 " rounds (non-terminal; LLAMA_SPEC_DRAFT_STOP_EOG=1 shortens these rounds)\n", stats.n_draft_eog_rejected);
            }
            if (stats.n_replay_mismatch > 0) {
                SLT_INF(*this, "replay mismatches = %" PRIu64 " (committed tokens kept; would have livelocked before)\n", stats.n_replay_mismatch);
            }

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);

            SLT_INF(*this, "rounds: verify %" PRIu64 ", replay %" PRIu64 ", fold %" PRIu64 ", plain %" PRIu64 " (empty drafts %" PRIu64 ", fold rejected %" PRIu64 ")\n",
                    stats.n_spec_verify_rounds, stats.n_spec_replay_rounds, stats.n_spec_fold_rounds, stats.n_spec_plain_rounds,
                    stats.n_spec_empty_drafts, stats.n_spec_fold_rejected);

            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        if (stats.n_spec_zero_rounds + stats.n_spec_probe_rounds > 0) { // [TAG_SPEC_DRAFT_POLICY]
            SLT_INF(*this, "draft policy tput: %" PRIu64 " zero rounds (no draft call), %" PRIu64 " probes, cooldown %d\n",
                    stats.n_spec_zero_rounds, stats.n_spec_probe_rounds, spec_cd.c);
        }

        if (stats.n_spec_timed_rounds > 0) { // [TAG_SPEC_ROUND_TIMING]
            const double nr = (double) stats.n_spec_timed_rounds;
            SLT_INF(*this, "round time: %" PRIu64 " rounds, per round draft %.3f ms, catch-up %.3f ms, verify %.3f ms, host %.3f ms\n",
                    stats.n_spec_timed_rounds, stats.t_spec_draft_us/1e3/nr, stats.t_spec_catchup_us/1e3/nr,
                    stats.t_spec_verify_us/1e3/nr, stats.t_spec_host_us/1e3/nr);
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        if (adm_deferred || adm_projected_mib > 0.0) {
            res["admission"] = {
                {"deferred",      adm_deferred},
                {"projected_mib", adm_projected_mib},
            };
        }

        // adaptive draft length (--spec-draft-n-max-adaptive): the effective n-max this slot last
        // asked the drafter for, and the cost model it came from - so a bench can see the rule work
        if (spec_n_max_last > 0 || spec_tput_active) {
            json acc_ema = json::array();
            for (const float p : spec_acc_ema) {
                acc_ema.push_back(p);
            }

            res["spec_draft"] = {
                {"n_max_eff",    spec_n_max_last},
                {"acc_ema",      acc_ema},
                {"cost_t_tok_ms", spec_cost_t_tok},
                {"cost_t_dec_ms", spec_cost_t_dec},
                {"cost_n_kv",     spec_cost_n_kv},
                {"cost_measured", spec_cost_measured},
            };

            if (spec_tput_active) { // [TAG_SPEC_DRAFT_POLICY] the tput state, also while the slot plays n = 0
                const auto & c = spec_tput_cost;
                const auto ab = [](double x, double y) { return json::array({ x, y }); };
                json n_samples = json::object();
                const char * fit_names[8] = { "draft", "ckpt", "verify", "catchup", "host", "host_d", "rollback", "restore" };
                for (int k = 0; k < 8; ++k) {
                    n_samples[fit_names[k]] = spec_tput_fit_n[k];
                }
                json fit = json::object();
                fit["draft"]     = ab(c.d0, c.d1);
                fit["ckpt"]      = ab(c.k0, c.k1);
                fit["verify"]    = ab(c.v0, c.v1);
                fit["catchup"]   = ab(c.c0, c.c1);
                fit["host"]      = c.host;
                fit["rollback"]  = ab(c.r0, c.r1);
                fit["n_rb_free"] = c.n_rb_free;
                fit["n_samples"] = n_samples;
                json t = json::object();
                t["n_rule"]          = spec_tput_rule;
                t["cooldown_c"]      = spec_cd.c;
                t["probe_in"]        = spec_cd.c > 0 ? spec_cd.wait : -1;
                t["n_fail"]          = spec_cd.n_fail;
                t["n_zero_rounds"]   = stats.n_spec_zero_rounds;
                t["n_probe_rounds"]  = stats.n_spec_probe_rounds;
                t["p_eff"]           = common_spec_tput_p_eff(spec_acc_ema, spec_acc_nobs);
                t["acc_n_obs"]       = spec_acc_nobs;
                t["n_kv"]            = spec_tput_n_kv;
                t["bucket"]          = common_spec_tput_bucket(spec_tput_n_kv);
                t["measured"]        = spec_tput_measured;
                t["ranked"]          = spec_tput_ranked;
                t["rate_tok_per_ms"] = spec_tput_rate;
                t["fit"]             = fit;
                res["spec_draft"]["tput"] = t;
            }
        }

        const auto & ptask = task ? task : task_prev;

        // n_prompt_tokens: the slot's tracked token count, also for a slot that ran no task yet (a legacy /slots
        //   restore, a fresh start). (Codex gate r1 C2) on an idle slot it is the resident count; while the slot is
        //   processing it is NOT committed residency: it already includes rows queued for a decode that has not run
        //   yet (yielding prefill) and unaccepted draft proposals (speculative verification), and the draft-side
        //   catch-up may lag it
        res["n_prompt_tokens"] = (int32_t) prompt.tokens.size();

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.id,     -1, -1);
        mem.seq_cp(id, other.id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                void * cb_data = slot.spec;
                static auto cb = [](llama_batch batch, void * user_data) {
                    common_speculative * spec = static_cast<common_speculative *>(user_data);
                    if (!common_speculative_process(spec, batch)) {
                        return 1;
                    }
                    return 0;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

// [TAG_SPEC_ADAPTIVE_DEPTH] running cost fit for the adaptive draft length
//
// Every target decode gives us one (n_tokens, wall_time) sample. The cost of a decode is affine in
// the batch size, cost(n) = a + b*n, so an exponentially-weighted least-squares fit of the wall time
// against the token count recovers exactly the two numbers the adaptive rule needs:
//     t_tok = b       (one extra drafted token in the verification decode)
//     t_dec = a + b   (a plain 1-token decode)
// The fit is taken over a short window (~256 rounds), so it describes the traffic that is actually
// running. It does NOT replace the affine-in-n_kv model: a fit is a single (t_tok, t_dec) pair at the
// mean depth of its own samples, so on its own it would say nothing about a slot sitting at a
// different depth. Instead the fit CALIBRATES the model - it fixes the level at the depth it was
// measured at, and the model keeps supplying the depth derivative:
//     t(n_kv) = [ t_measured / model(kv_window) ] * model(n_kv)
// which is exactly the measurement when the slot is at the window's depth, and the model's own
// extrapolation away from it. When the fit is not usable - too few samples, or the batch sizes seen
// were all the same, so the slope is not identified - the configured model is used unscaled.
//
// Only decodes that are pure generation (no prompt tokens) and that were synchronized are fed in, so
// prefill ubatches cannot contaminate the fit.
struct server_spec_cost_fit {
    static constexpr double ALPHA   = 1.0/256.0; // forgetting factor
    static constexpr double W_MIN   = 32.0;      // effective samples before the fit is trusted
    static constexpr double VAR_MIN = 0.25;      // spread in n (var, tokens^2) needed to identify the slope
    static constexpr double R_MAX   = 0.98;      // a drafted token cannot cost a whole decode - see solve()
    static constexpr double F_MIN   = 0.1;       // clamp on the calibration factor: the model and the
    static constexpr double F_MAX   = 10.0;      // machine may disagree, but not by 10x in both directions

    double w = 0.0, s_n = 0.0, s_t = 0.0, s_nn = 0.0, s_nt = 0.0, s_kv = 0.0;

    void add(int n_tokens, double t_ms, int n_kv) {
        if (n_tokens <= 0 || !(t_ms > 0.0)) {
            return;
        }

        // reject gross outliers (a stall, a page-fault storm, a task processed inside the yield):
        // they would drag the slope around for a whole window
        if (w >= W_MIN) {
            const double mean_t = s_t / w;
            if (t_ms > 8.0*mean_t) {
                return;
            }
        }

        const double n = (double) n_tokens;
        const double d = 1.0 - ALPHA;

        w    = d*w    + 1.0;
        s_n  = d*s_n  + n;
        s_t  = d*s_t  + t_ms;
        s_nn = d*s_nn + n*n;
        s_nt = d*s_nt + n*t_ms;
        s_kv = d*s_kv + (double) std::max(n_kv, 0);
    }

    // model(n_kv) = c0 + k*n_kv for each cost. Returns true and fills t_tok/t_dec (ms) with the
    // measurement-calibrated model at n_kv, or false when the fit is not usable.
    bool solve(int n_kv, float t_tok0, float k_tok, float t_dec0, float k_dec,
               float & t_tok, float & t_dec) const {
        if (w < W_MIN) {
            return false;
        }

        const double mean_n = s_n / w;
        const double mean_t = s_t / w;
        const double var_n  = s_nn / w - mean_n*mean_n;

        if (var_n < VAR_MIN) {
            return false; // all batches had the same size - the slope is not identified
        }

        const double cov   = s_nt / w - mean_n*mean_t;
        const double slope = cov / var_n;
        const double icpt  = mean_t - slope*mean_n;

        if (!(slope > 0.0) || !(icpt > 0.0)) {
            return false; // nonsense fit (noise-dominated); keep the model
        }

        // a verification decode of d+1 tokens is one graph, so one extra drafted token cannot cost as
        // much as the whole decode it might save; a fit that says otherwise is noise, not physics
        if (slope > R_MAX*(icpt + slope)) {
            return false;
        }

        // depth the fit describes, and the model there
        const double kv_win    = s_kv / w;
        const double m_tok_win = (double) t_tok0 + (double) k_tok*kv_win;
        const double m_dec_win = (double) t_dec0 + (double) k_dec*kv_win;

        if (!(m_tok_win > 0.0) || !(m_dec_win > 0.0)) {
            return false;
        }

        const double f_tok = std::min(std::max(slope        / m_tok_win, F_MIN), F_MAX);
        const double f_dec = std::min(std::max((icpt+slope) / m_dec_win, F_MIN), F_MAX);

        t_tok = (float) (f_tok*((double) t_tok0 + (double) k_tok*(double) n_kv));
        t_dec = (float) (f_dec*((double) t_dec0 + (double) k_dec*(double) n_kv));

        return true;
    }
};

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        server_metrics res = metrics;

        // G32: process-wide, not tracked per slot - see the SERVER_TASK_TYPE_METRICS handler
        res.n_sampling_us       = common_sampler_perf_time_us();
        res.n_sampling_calls    = common_sampler_perf_n_sample();
        res.n_sampling_fast     = common_sampler_perf_n_fast();
        res.n_sampling_fast_pen = common_sampler_perf_n_fast_pen();

        if (disk_cache) {
            res.disk.n_saves        = disk_cache->n_saves;
            res.disk.n_loads        = disk_cache->n_loads;
            res.disk.n_evict        = disk_cache->n_evict;
            res.disk.n_deep_rewinds = disk_cache->n_deep_rewinds;
            res.disk.n_load_failed  = disk_cache->n_load_failed;
            res.disk.n_load_capacity = disk_cache->n_load_capacity;
            res.disk.n_load_other   = disk_cache->n_load_other;
            res.disk.bytes_written  = disk_cache->bytes_written;
            res.disk.bytes_loaded   = disk_cache->bytes_loaded;
            res.disk.save_us        = disk_cache->save_us;
            res.disk.load_us        = disk_cache->load_us;
            res.disk.n_entries      = disk_cache->n_entries();
            res.disk.n_deep_ckpts   = disk_cache->n_deep_ckpts();
            res.disk.n_blocks       = disk_cache->n_blocks();
            res.disk.bytes_on_disk  = disk_cache->total_bytes();
            // [TAG_DISK_ASYNC_SAVE]
            res.disk.save_async_us         = disk_cache->save_async_us;
            res.disk.save_copy_us          = disk_cache->save_copy_us;
            res.disk.save_wait_us          = disk_cache->save_wait_us;
            res.disk.bytes_copied          = disk_cache->bytes_copied;
            res.disk.n_saves_async         = disk_cache->n_saves_async;
            res.disk.n_saves_async_failed  = disk_cache->n_saves_async_failed;
            res.disk.n_pending             = disk_cache->n_pending();
            res.disk.last_save_blocking_us = disk_cache->last_save_blocking_us;
            res.disk.last_save_total_us    = disk_cache->last_save_total_us;
            res.disk.last_save_bytes       = disk_cache->last_save_bytes;
            res.disk.last_save_written     = disk_cache->last_save_written;
            res.disk.last_load_us          = disk_cache->last_load_us;
            res.disk.last_load_bytes       = disk_cache->last_load_bytes;
        }

        return res;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

    json disk_cache_status() const {
        return disk_cache ? disk_cache->status() : json{ { "enabled", false } };
    }

    // [TAG_DISK_PROMPT_CACHE] the server is stopping (the loop has returned): keep every slot's conversation, so a
    //   restart or a deploy loses nothing; runs on the main thread before the contexts are freed
    void save_on_exit() {
        save_all_slots("exit");
    }

    // [TAG_DISK_PROMPT_CACHE] best-effort: a failed save is logged and the conversation is lost (also used before
    //   the idle sleep frees the contexts, [TAG_SLEEP_SAVE])
    void save_all_slots(const char * why) {
        if (!disk_cache || sleeping || ctx_tgt == nullptr) {
            return;
        }
        for (auto & slot : slots) {
            if (slot.prompt.n_tokens() > 0) {
                SLT_INF(slot, "saving %d tokens to the disk cache before %s\n", slot.prompt.n_tokens(), why);
                if (!slot.disk_save(*disk_cache)) {
                    SLT_WRN(slot, "disk cache: the %d tokens were not saved before %s\n", slot.prompt.n_tokens(), why);
                }
            }
        }
        // [TAG_DISK_ASYNC_SAVE] every background save committed before the contexts go; a slot whose save failed in the
        //   background (its disk_save() returned true when the copy was queued) is reported here
        for (const int32_t id : disk_cache->drain()) { // ids deduplicated; slot ids are their indices
            if (id >= 0 && id < (int32_t) slots.size()) {
                SLT_WRN(slots[id], "disk cache: the background save before %s failed, the conversation is not on disk\n", why);
            }
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // [TAG_MULTISLOT] LLAMA_QSA_HOMOGENEOUS=1 with n_parallel > 1 (server_multislot_envelope_check): shared-prompt
    //   parent/child requests are rejected before any slot is chosen
    bool multislot_strict = false;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    // [TAG_SLOT_INVALIDATE] slots released on a failure path in this update_slots() pass: their sequence may hold part
    //   of a batch, and a batch built before the failure may still write it, so they are cleared again after decoding
    std::vector<int32_t> slots_invalidated;

    // debug-only fault injection for the failure-path tests, read once at load (absent in every deployment):
    //   LLAMA_SERVER_DEBUG_FAULT=slot   - a per-slot exception in post_decode() once a slot has generated 2 tokens
    //   LLAMA_SERVER_DEBUG_FAULT=decode - an exception after a successful decode (as a failed speculative batch)
    //   LLAMA_SERVER_DEBUG_FAULT=ckpt   - an exception in the first create_checkpoint() (a pre-decode batch-building
    //                                     failure after the slot's prompt rows were appended)
    int32_t dbg_fault = 0;

    // [TAG_PREDECODE_ABORT] (Codex gate r1 A1) true while pre_decode() builds the batch: a per-slot exception there
    //   cannot be confined to the slot (its rows are already in the batch, slot_batched may be unset or point at it),
    //   so iterate() rethrows it to the outer abort-and-return path in update_slots(), which discards the batch
    bool iterate_rethrow = false;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // [TAG_CKPT_NO_DFT] (v33 B2) prompt checkpoints leave the draft KV out when it is a plain (partially
    //   removable) KV cache of its own: it is causal, so on restore seq_rm(pos_max + 1, -1) leaves exactly the
    //   values a checkpoint copy would have restored (only the physical cell placement differs), and the
    //   speculative pending state (MTP pending_h, eagle3 boundary row) still travels in data_spec. The plain
    //   KV cache ignores PARTIAL_ONLY (llama-kv-cache.cpp state_write), so the copy was the draft's WHOLE KV of
    //   the sequence (2 KiB x n_past on the prod MTP head) in host RAM per checkpoint. Off for a draft whose
    //   memory is recurrent (FULL/RS: it cannot be rewound) or shared with the target (ctx_other).
    bool ckpt_skip_dft = false;

    // [TAG_GTT_ADMISSION] (v33, night-1 #13; fix round 1) headroom admission for prefill chunks. OFF unless
    //   LLAMA_GTT_ADMISSION_MIB is set > 0 (the compose sets bar, slope and step per profile). Before a slot's prompt
    //   tokens enter the batch:
    //     growth    = min(max(0, resident + chunk - high_water), n_ctx - high_water)        [cells]
    //     projected = used (sysfs mem_info_gtt_used) + slope x growth + (growth > 0 ? step : 0)
    //   growth == 0 -> ADMIT (the chunk only refills cells the pool already had: first-fit, see below);
    //   projected <= bar -> ADMIT; else DEFER (slot skipped this iteration, re-checked every iteration, 50 ms
    //   back-off when nothing else runs); deferred longer than LLAMA_GTT_ADMISSION_MAX_WAIT_S (default 600,
    //   0 = forever) -> REJECT: send_error(ERROR_TYPE_UNAVAILABLE) = HTTP 503 and the slot is released. A
    //   stream=true request already got its 200 headers in STARTED, so it receives the error as an SSE error event.
    //   Scheduling only - no numerics change.
    //   resident = sum of the slots' cached tokens; high_water = its maximum over this process. Under llama_kv_cache
    //   (the attention/indexer pool) seq_rm rewinds the head to the lowest freed cell and find_slot takes empty
    //   cells first-fit from the head (llama-kv-cache.cpp seq_rm / find_slot), so every cell below the head is in
    //   use and the pool's used_max_p1 grows only once all holes are refilled: used_max_p1 == max(used) == the
    //   high-water of resident while cells are not shared between sequences. Over-count (under-projection) risk:
    //   seq_cp-shared cells (n > 1 child slots) count once per slot. LLAMA_GTT_HOLES_FULL_CHUNK=1 instead projects
    //   the whole chunk as growth whenever resident < high_water (conservative; it disables the growth == 0 admit
    //   after any trim).
    //   step (only when growth > 0), the largest of:
    //     LLAMA_GTT_STAGING_STEP_MIB (explicit, calibrated on the device);
    //     geometric prealloc_x (GGML_VK_QSA_SCRATCH_PRESIZE + _PRESIZE_FA): 0.25 x LLAMA_GTT_STAGING_B_PER_CELL (2048)
    //       x cells_after;
    //     allocator regrow slack: LLAMA_GTT_SLACK_PCT (default = GGML_ALLOC_GROW_SLACK, the percent the run-time
    //       regrow adds to a growing compute chunk) / 100 x slope x cells_after (the growing buffers ~ slope x cells).
    //   [TAG_QSA_VIS] (A4 T4) + once (only when growth > 0): LLAMA_GTT_ONCE_MIB (default 0 = off) = what is still to be
    //     allocated ONCE after load, not per cell - with --kv-graph-span 0 every compute buffer is reserved at load and
    //     LLAMA_QSA_VIS=1 + LLAMA_QSA_PREFILL=indexed leave no per-cell FA staging, so the run-time growth is the one-time
    //     prealloc_x presize to the reserve hint (the QSA select scratch, pool x 1 026 B/cell at C 256) at the first
    //     >= 64-token FA / QSA dispatch, plus first-use objects. Pending part = max(0, once - (used - used at init)): it
    //     vanishes as soon as GTT has grown that much since admission_init. The A4 brain layout then runs with a small
    //     slope (first-use, allocator rounding) and LLAMA_GTT_STAGING_B_PER_CELL=0 (no geometric prealloc_x growth: the
    //     presize covers every graph of the span) instead of the dense 2 560 B/cell + 0.25 x 2 048 B/cell step.
    struct {
        bool        enabled = false;
        std::string path;
        double      bar_mib            = 0.0;
        double      slope_b_per_cell   = 18432.0;
        double      staging_b_per_cell = 2048.0;
        double      step_override_mib  = -1.0;
        double      slack_pct          = 0.0;
        double      once_mib           = 0.0;   // [TAG_QSA_VIS] LLAMA_GTT_ONCE_MIB
        double      used_init_mib      = 0.0;   // GTT used at admission_init (the baseline of the pending once part)
        double      max_wait_s         = 600.0;
        bool        geometric          = false;
        bool        holes_full_chunk   = false;
        int64_t     hw_cells           = 0;     // resident-cell high-water seen by this server
        bool        deferred_this_iter = false;
        // [TAG_MULTISLOT] (stage 5.1 item 5a) LLAMA_GTT_CONCURRENT_RESERVE_MIB: a measured, deployment-selected
        //   reserve for concurrent operations (another slot's verify / restore / checkpoint while this prompt runs),
        //   added to the projection whenever another slot is processing - also for a zero-growth chunk (hole reuse),
        //   which is otherwise admitted unconditionally. Constant (never retired by unrelated GTT growth, unlike
        //   LLAMA_GTT_ONCE_MIB). Default 0 = today's admission bit for bit.
        double      reserve_mib        = 0.0;
    } adm;

    enum admission_result { ADM_ADMIT, ADM_DEFER, ADM_REJECT };

    common_speculative_ptr spec;

    // [TAG_BS_GUARD] decided in load_model from the loaded configuration
    server_backend_sampling_support bs_support;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    std::unique_ptr<server_disk_cache> disk_cache; // [TAG_DISK_PROMPT_CACHE]

    // [TAG_CACHE_COMPAT2] the short-context measurement (spec D4): on sequence 0 before serving, from a cleared
    //   target and draft memory, a temporary fresh speculative object driven through target decode + its process(),
    //   a fixed 64-token probe (output on the last token only, like a prompt); hashes the target, draft and carry
    //   host payloads (flags 0). "" on failure.
    //   Isolation (Codex compat2 B): it runs BEFORE the serving speculative object exists (load_model builds that one
    //   afterwards, afresh), so a temporary object's sampler attach / detach on the draft context and its
    //   embeddings_nextn switches never reach the serving configuration; both memories are cleared on every exit,
    //   an exception included. Not restored, only warmed: the scheduler's reserved graph / compute buffers and the
    //   output buffers (sizes grown to the probe's ubatch, a reserve pending) - no stored state depends on them.
    //   Schedule: ubatches of min(16, effective n_batch / n_ubatch of both contexts) tokens (a context with
    //   n_batch < 16 would assert in decode); the chunk and the effective sizes are part of the hashed bytes
    std::string cache_compat_measure() {
        constexpr int N = 64;
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        int CH = 16;
        for (llama_context * c : { ctx_tgt, ctx_dft }) {
            if (c) {
                CH = std::min<int>(CH, (int) std::min(llama_n_batch(c), llama_n_ubatch(c)));
            }
        }
        if (n_vocab <= 0 || CH < 1 || llama_n_ctx_seq(ctx_tgt) < (uint32_t) N ||
            (ctx_dft && llama_n_ctx_seq(ctx_dft) < (uint32_t) N)) {
            SRV_WRN("cache-compat: no measurement possible (n_vocab %d, chunk %d, n_ctx_seq %u)\n", n_vocab, CH,
                    llama_n_ctx_seq(ctx_tgt));
            return "";
        }
        std::string bytes = string_format("probe=%u;n=%d;chunk=%d;tgt_nb=%u;tgt_ub=%u;dft_nb=%u;dft_ub=%u;",
                                          DISK_CACHE_PROBE_REVISION, N, CH, llama_n_batch(ctx_tgt),
                                          llama_n_ubatch(ctx_tgt), ctx_dft ? llama_n_batch(ctx_dft) : 0u,
                                          ctx_dft ? llama_n_ubatch(ctx_dft) : 0u);
        common_speculative_ptr tmp;
        llama_batch b = llama_batch_init(CH, 0, 1);
        const auto cleanup = [&]() {
            tmp.reset(); // detaches its samplers from the draft context (the serving object does not exist yet)
            llama_batch_free(b);
            b = {};
            llama_synchronize(ctx_tgt);
            llama_memory_clear(llama_get_memory(ctx_tgt), true);
            if (ctx_dft) {
                llama_synchronize(ctx_dft);
                llama_memory_clear(llama_get_memory(ctx_dft), true);
            }
        };
        bool ok = true;
        try {
            llama_memory_clear(llama_get_memory(ctx_tgt), true);
            if (ctx_dft) {
                llama_memory_clear(llama_get_memory(ctx_dft), true);
            }
            if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                tmp.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            }
            for (int i0 = 0; ok && i0 < N; i0 += CH) {
                common_batch_clear(b);
                for (int i = i0; i < std::min(N, i0 + CH); i++) {
                    common_batch_add(b, (llama_token) ((1009 + 7919LL * i) % n_vocab), i, { 0 }, i == N - 1);
                }
                ok = llama_decode(ctx_tgt, b) == 0;
                llama_synchronize(ctx_tgt);
                if (ok && tmp) {
                    ok = common_speculative_process(tmp.get(), b);
                }
            }
            const auto section = [&](const char * name, llama_context * ctx) {
                const size_t n = llama_state_seq_get_size_ext(ctx, 0, 0);
                std::vector<uint8_t> buf(n);
                const size_t got = n ? llama_state_seq_get_data_ext(ctx, buf.data(), n, 0, 0) : 0;
                bytes += string_format("%s=%zu:", name, got);
                bytes.append(reinterpret_cast<const char *>(buf.data()), got);
                return got == n;
            };
            if (ok) {
                llama_synchronize(ctx_tgt);
                ok = section("tgt", ctx_tgt);
                if (ok && ctx_dft) {
                    llama_synchronize(ctx_dft);
                    ok = section("dft", ctx_dft);
                }
                std::vector<uint8_t> carry;
                if (ok && tmp && common_speculative_n_stateful(tmp.get()) > 0) {
                    ok = common_speculative_get_state(tmp.get(), 0, carry);
                }
                bytes += string_format("carry=%zu:", carry.size());
                bytes.append(reinterpret_cast<const char *>(carry.data()), carry.size());
            }
        } catch (const std::exception & e) {
            SRV_WRN("cache-compat: measurement failed: %s\n", e.what());
            ok = false;
        } catch (...) {
            SRV_WRN("%s", "cache-compat: measurement failed: unknown exception\n");
            ok = false;
        }
        try {
            cleanup();
        } catch (...) {
            SRV_WRN("%s", "cache-compat: cleanup after the measurement threw\n");
            ok = false;
        }
        return ok ? server_cache_measure_hash(bytes) : "";
    }

    // [TAG_CACHE_COMPAT2] the measurement for this load, computed by cache_compat_prepare() before the serving
    //   speculative object is built; consumed by cache_compat_key()
    std::string cache_meas;
    std::string cache_meas_how;  // "memo hit" / "computed N ms" / "failed"
    std::string cache_memo_key_pending; // a fresh measurement to publish once the root lock is held ("": none)

    // [TAG_MULTISLOT] the context size in the disk cache's structural fingerprint is the PER-SLOT one (n_ctx_seq): a
    //   stored state is one sequence, and per-seq state is stream-count agnostic - so an np1 namespace is adoptable
    //   by an np2 per-stream server of the same slot size. Equal to llama_n_ctx for np1 and for --kv-unified, so
    //   those fingerprints (and keys) are unchanged; n_parallel / kv_unified stay in the parsed-param digest (key)
    int32_t cache_n_ctx_seq() const {
        return ctx_tgt ? (int32_t) llama_n_ctx_seq(ctx_tgt) : n_ctx;
    }

    bool cache_disk_wanted() const {
        return params_base.lora_adapters.empty() && !params_base.cache_disk_path.empty() && params_base.cache_disk_gib > 0;
    }

    // [TAG_CACHE_COMPAT2] memo lookup or measurement; called in load_model() before the serving speculative object
    void cache_compat_prepare() {
        cache_meas.clear();
        cache_meas_how.clear();
        cache_memo_key_pending.clear();
        const std::string cheap = server_disk_cache_fingerprint(params_base, cache_n_ctx_seq());
        const std::string memo = server_cache_memo_key(cheap, server_cache_runtime_identity(),
                                                       server_cache_placement_identity(params_base, ctx_tgt, ctx_dft));
        if (memo.empty()) {
            SRV_WRN("%s", "cache-compat: runtime libraries unverifiable (mapping != file) - no measurement memo\n");
        } else if (server_cache_memo_get(params_base.cache_disk_path, memo, cache_meas)) {
            cache_meas_how = "memo hit";
            return;
        }
        const int64_t t0 = ggml_time_us();
        cache_meas = cache_compat_measure();
        const int64_t ms = (ggml_time_us() - t0) / 1000;
        if (cache_meas.empty()) {
            cache_meas_how = string_format("failed %" PRId64 " ms", ms);
        } else {
            cache_meas_how = string_format("computed %" PRId64 " ms", ms);
            cache_memo_key_pending = memo;
        }
    }

    // [TAG_CACHE_COMPAT2] key v2 from the cheap identity: the measurement comes from the memo (cheap identity + probe
    //   revision + GPU identity) or is computed now - a memo miss means a new cheap identity, i.e. the cache root can
    //   only hold entries of another cheap identity, or none (this server is then the first writer and seeds its own
    //   measurement before its first save)
    std::string cache_compat_key(const std::string & cheap) {
        const std::string meas = server_cache_measure_component(cache_meas);
        if (cache_meas.empty()) {
            cache_memo_key_pending.clear(); // fail-closed: nothing published
            SRV_WRN("cache-compat: measurement %s - non-reusable namespace %s (no stored state is reused or kept "
                    "for a later start)\n", cache_meas_how.empty() ? "not prepared" : cache_meas_how.c_str(), meas.c_str());
        } else {
            SRV_INF("cache-compat: measurement %s (%s)\n", meas.c_str(), cache_meas_how.c_str());
        }
        server_cache_key_parts p;
        p.model_id   = server_disk_cache_model_identity(params_base);
        p.desc_h     = disk_cache->descriptor_key();
        p.measure_h  = meas;
        p.env_digest = server_cache_env_digest(server_cache_environ());
        p.param_digest = server_cache_param_digest(params_base, ctx_tgt, ctx_dft);
        const std::string fp = server_disk_cache_fingerprint_v2(cheap, p);
        SRV_INF("cache-compat: key %s = model %s | desc %s | meas %s | depth %u | env %s | param %s (cheap %s, "
                "place %s)\n", fp.substr(0, 16).c_str(), p.model_id.c_str(), p.desc_h.c_str(), meas.c_str(),
                p.depth_version, p.env_digest.c_str(), p.param_digest.c_str(), cheap.substr(0, 16).c_str(),
                server_cache_placement_identity(params_base, ctx_tgt, ctx_dft).c_str());
        return fp;
    }

    // [TAG_CACHE_COMPAT2] publish a fresh measurement to the memo - only while this process holds the cache root's
    //   lock (after init()), never for a failed measurement
    void cache_compat_publish() {
        if (!cache_memo_key_pending.empty() && disk_cache && disk_cache->owns_root() && !cache_meas.empty()) {
            if (!server_cache_memo_put(disk_cache->root_dir(), cache_memo_key_pending, cache_meas)) {
                SRV_WRN("%s", "cache-compat: the measurement memo could not be written\n");
            }
        }
        cache_memo_key_pending.clear();
    }

    // [TAG_DEEP_CKPT] a checkpoint leaving the slot's in-RAM list goes to the deep store (if the schedule keeps it)
    // [TAG_SLOT_FILE_CKPT_NO_DEEP] (SC1 review m5, merge with U5) a checkpoint restored from a /slots file (id_task =
    //   SLOT_FILE_CKPT_ID_TASK) is never stored in the deep store, whatever its kind (depth / semantic / pin): its bytes
    //   come from a file outside the disk cache, and before U5 such checkpoints did not exist at all, so the deep store
    //   stays exactly what v62 writes. It is still used from RAM (restore within the slot)
    void deep_spill(server_slot & slot, const common_prompt_checkpoint & ckpt, uint8_t kind) {
        if (ckpt.id_task == SLOT_FILE_CKPT_ID_TASK) {
            SLT_TRC(slot, "deep checkpoint at %" PRId64 " not stored: restored from a slot file\n", ckpt.n_tokens);
            return;
        }
        // (Codex gate r3 Major 2) not persisted with more than one stateful speculative implementation (one carry)
        if (disk_cache && common_speculative_n_stateful(spec.get()) <= 1) {
            disk_cache->deep_put(slot.prompt, ckpt, kind);
        }
    }

    // [TAG_SEMANTIC_CKPT] (SC1) the kind a checkpoint leaving the slot's in-RAM list is stored as: semantic when it sits
    //   at a message boundary of the slot's current prompt - a user turn start (the prompt processing splits there), or
    //   within SEM_SLACK tokens of the end of the system prompt or of the input of a turn (a user message or a tool result:
    //   the start of an assistant message that follows a non-assistant one), or of a tool result's span. That end is
    //   where a request's generation prompt starts, and the prompt-tail split puts a checkpoint 4 tokens before the
    //   prompt end, i.e. inside the generation prompt right after it (a template whose tool-response delimiter does not
    //   tokenize like the prompt shows tool results as user messages - the assistant-start rule still finds their end).
    //   Only positions the current task shares with the slot's prompt are judged (its message spans index its tokens)
    uint8_t deep_kind_at(const server_slot & slot, int64_t n_tokens) const {
        static constexpr int64_t SEM_SLACK = 16;
        if (!disk_cache || disk_cache->ckpt_sem_max <= 0 || !slot.task || n_tokens > (int64_t) slot.prompt.n_tokens()) {
            return server_deep_ref::DEEP_KIND_DEPTH;
        }
        const auto & spans = slot.task->params.message_spans.spans;
        for (size_t i = 0; i < spans.size(); i++) {
            const auto & sp = spans[i];
            if (sp.role == COMMON_CHAT_ROLE_USER && (int64_t) sp.pos == n_tokens) {
                return server_deep_ref::DEEP_KIND_SEMANTIC;
            }
            const bool input_end = (sp.role == COMMON_CHAT_ROLE_SYSTEM || sp.role == COMMON_CHAT_ROLE_TOOL) ||
                    (i + 1 < spans.size() && sp.role != COMMON_CHAT_ROLE_ASSISTANT && spans[i + 1].role == COMMON_CHAT_ROLE_ASSISTANT);
            if (input_end && std::llabs(n_tokens - (int64_t) (sp.pos + sp.len)) <= SEM_SLACK) {
                return server_deep_ref::DEEP_KIND_SEMANTIC;
            }
        }
        return server_deep_ref::DEEP_KIND_DEPTH;
    }

    // [TAG_CACHE_PIN] (SC1) the request's `cache_pin: {"tokens": N}` once the kept prefix n_past is known: N beyond it is
    //   reached by this prompt processing (a split and a checkpoint at N, stored as the pin); N inside it is pinned only
    //   if a checkpoint at exactly N exists (in RAM or stored) - the state at N is gone otherwise; N must leave at least
    //   one prompt token to process (a restore needs it)
    // (review m2) the outcome as it is: pinned only if the conversation's deep refs now carry a pin at n (deep_put declines
    //   noncanonical / partial prompts, more than one stateful speculative carry, on-device data, quota 0)
    void cache_pin_report(const server_slot & slot, int64_t n, const char * src) const {
        const bool ok = std::any_of(slot.prompt.deep.begin(), slot.prompt.deep.end(), [&](const auto & r) {
            return r.n_tokens == n && r.kind == server_deep_ref::DEEP_KIND_PIN;
        });
        if (ok) {
            SLT_INF(slot, "cache_pin: checkpoint at n_tokens = %" PRId64 " (%s) is this conversation's pin\n", n, src);
        } else {
            SLT_WRN(slot, "cache_pin: the checkpoint at %" PRId64 " was not stored (the disk cache declined it), not pinned\n", n);
        }
    }

    void cache_pin_setup(server_slot & slot, int64_t n_past) {
        slot.ckpt_pin_pos = -1;
        const int64_t pin = slot.task->params.cache_pin_tokens;
        if (pin <= 0 || !disk_cache) {
            return;
        }
        if (pin >= (int64_t) slot.task->n_tokens()) {
            SLT_WRN(slot, "cache_pin: %" PRId64 " tokens is not inside the prompt (%d tokens), ignored\n", pin, slot.task->n_tokens());
            return;
        }
        if (pin > n_past) {
            slot.ckpt_pin_pos = pin;
            return;
        }
        for (const auto & ck : slot.prompt.checkpoints) {
            // [TAG_SLOT_FILE_CKPT_NO_DEEP] (review m5) a slot-file checkpoint cannot become the pin (deep_spill declines it)
            if (ck.n_tokens == pin && ck.id_task != SLOT_FILE_CKPT_ID_TASK) {
                deep_spill(slot, ck, server_deep_ref::DEEP_KIND_PIN);
                cache_pin_report(slot, pin, "in RAM");
                return;
            }
        }
        for (const auto & m : disk_cache->deep_candidates(slot.task->tokens, pin)) {
            if (m.n_tokens == pin) {
                disk_cache->deep_adopt(slot.prompt, m, server_deep_ref::DEEP_KIND_PIN);
                cache_pin_report(slot, pin, "stored");
                return;
            }
            break; // deepest first
        }
        SLT_WRN(slot, "cache_pin: %" PRId64 " tokens are already processed without a checkpoint there, not pinned\n", pin);
    }

    // collect deep checkpoints nothing references (see live_deep_refs)
    void deep_gc() {
        if (disk_cache) {
            disk_cache->deep_gc();
        }
    }

    // the deep refs of every live prompt: the slots and the RAM prompt cache's states
    std::set<std::string> live_deep_refs() const {
        std::set<std::string> keys;
        for (const auto & slot : slots) {
            for (const auto & r : slot.prompt.deep) {
                keys.insert(r.key);
            }
        }
        if (prompt_cache) {
            for (const auto & st : prompt_cache->states) {
                for (const auto & r : st.prompt.deep) {
                    keys.insert(r.key);
                }
            }
        }
        return keys;
    }

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    // measured cost model for --spec-draft-n-max-adaptive (see server_spec_cost_fit)
    server_spec_cost_fit spec_cost;

    // [TAG_SPEC_DRAFT_POLICY] --spec-draft-policy tput: per-round cost fits (draft wall vs drafted n, verify and
    // catch-up wall vs batch rows, host wall, the rollback cycle vs the replay rows), one set per KV-depth bucket
    common_spec_tput_model spec_tput_fits;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            // [TAG_SLEEP_SAVE] destroy() frees the slots' state: keep their conversations first (RAM-tier entries are
            //   lost with it; the RAM path already wrote its outgoing slots to disk, [TAG_RAM_TIER_DISK])
            save_all_slots("sleep");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        if (const char * f = getenv("LLAMA_SERVER_DEBUG_FAULT")) {
            dbg_fault = std::string(f) == "slot" ? 1 : std::string(f) == "decode" ? 2 : std::string(f) == "ckpt" ? 3 : 0;
            SRV_WRN("debug fault injection armed: %s\n", f);
        }

        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        // [TAG_BS_GUARD] decide backend sampling support before the context is created: a server started with
        //   --backend-sampling attaches backend samplers at context creation (common_init_from_params), which the
        //   loaded speculation / tensor split cannot serve - force the default off (params is also the routes'
        //   request default, so requests without the field stay unaffected)
        bs_support = server_backend_sampling_check(params);
        if (!bs_support.supported && params.sampling.backend_sampling) {
            SRV_WRN("[bs-guard] --backend-sampling forced off: %s\n", bs_support.error.c_str());
            params.sampling.backend_sampling = false;
        }
        SRV_INF("[bs-guard] backend_sampling_supported=%d spec_type=%s n_max=%d outputs_per_seq=%d\n",
                bs_support.supported ? 1 : 0, bs_support.spec_type.c_str(), bs_support.n_max, bs_support.per_seq_max);

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();

        // [TAG_MULTISLOT] the support envelope, decided before anything (mmproj included) is loaded
        {
            const auto env = server_multislot_envelope_check(getenv("LLAMA_QSA_HOMOGENEOUS"), params.n_parallel,
                                                             has_mmproj, params.ctx_shift);
            if (!env.ok) {
                SRV_ERR("%s\n", env.error.c_str());
                return false;
            }
            if (!env.proof.empty()) {
                if (env.strict) {
                    SRV_INF("%s\n", env.proof.c_str());
                } else {
                    SRV_WRN("%s\n", env.proof.c_str());
                }
            }
            multislot_strict = env.strict;
            if (multislot_strict) {
                SRV_INF("[qsa-route] row cap: target ranges %lld, draft groups %lld (0 = none; LLAMA_MTP_QSA_MAX_UBATCH, target cap only with an attention catch-up or LLAMA_QSA_HOMOGENEOUS_TARGET_CAP=1)\n",
                        (long long) qsa_env_target_max_ubatch(), (long long) qsa_env_mtp_max_ubatch());
            }
        }

        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        // [TAG_MULTISLOT] the context-creation line `qsa_history_dense = ...` is library INFO (filtered at -lv 3): the
        //   same information at the server's INFO level (hands proof), next to the [multislot-envelope] line
        {
            const char *  reason = nullptr;
            const int32_t hd     = llama_qsa_history_dense_info(ctx_tgt, &reason);
            if (hd >= 0) {
                const char * hom = getenv("LLAMA_QSA_HOMOGENEOUS");
                SRV_INF("qsa_history_dense = %s (%s)%s\n", hd ? "on" : "off", reason,
                        hom != nullptr && strcmp(hom, "1") == 0 && !hd ? "; coverage check still on: LLAMA_QSA_HOMOGENEOUS=1" : "");
            }
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // [TAG_CACHE_COMPAT2] the cache-compatibility measurement runs before the serving speculative object exists
        //   (isolation: its temporary object attaches / detaches draft samplers); the serving one is built afresh below
        spec.reset();
        if (cache_disk_wanted()) {
            cache_compat_prepare();
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        // [TAG_BS_GUARD] a speculation that failed to initialize runs without one: report (and judge) the server as such
        if (!spec && bs_support.spec_type != "none") {
            common_params params_nospec = params_base;
            params_nospec.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
            bs_support = server_backend_sampling_check(params_nospec);
            SRV_INF("[bs-guard] speculation not loaded: backend_sampling_supported=%d spec_type=none\n",
                    bs_support.supported ? 1 : 0);
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
            ckpt_skip_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART && llama_get_ctx_other(ctx_dft) == nullptr;
            SRV_INF("prompt checkpoints %s the draft KV (draft seq_rm type = %d)\n",
                    ckpt_skip_dft ? "exclude (restore = draft seq_rm)" : "include", (int) ctx_dft_seq_rm_type);
        }

        if (spec && ctx_dft) {
            // [TAG_CACHE_COMPAT2] the serving draft context's backend samplers (equal on a memo hit and a miss)
            std::string att;
            for (int i = 0; i < params_base.n_parallel; i++) {
                const llama_sampler * sm = llama_get_sampler(ctx_dft, i);
                att += string_format("%s%d=", i ? "," : "", i);
                if (!sm) {
                    att += "none";
                    continue;
                }
                const int n = std::string(llama_sampler_name(sm)) == "chain" ? llama_sampler_chain_n(sm) : 0;
                att += llama_sampler_name(sm);
                for (int k = 0; k < n; k++) {
                    att += std::string(k ? "+" : "[") + llama_sampler_name(llama_sampler_chain_get(const_cast<llama_sampler *>(sm), k));
                }
                att += n > 0 ? "]" : "";
            }
            SRV_INF("speculative: serving draft samplers %s\n", att.c_str());
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft;
            slot.mem.init(ctx_tgt, ctx_dft);
            slot.spec    = spec.get();
            slot.n_ctx   = n_ctx_slot();

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel), n_embd);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        // created once: a wake-up from idle sleep reloads the model but keeps this object, so GET /disk-cache (HTTP
        // threads) never sees it replaced; entries are matched by tokens only, so LoRA adapters disable it
        if (!params_base.lora_adapters.empty() && !params_base.cache_disk_path.empty()) {
            SRV_WRN("%s", "--cache-disk is disabled while LoRA adapters are loaded (stored states are keyed by tokens only)\n");
        }
        if (!disk_cache && params_base.lora_adapters.empty() && !params_base.cache_disk_path.empty() && params_base.cache_disk_gib > 0) {
            disk_cache = std::make_unique<server_disk_cache>(
                params_base.cache_disk_path, (uint64_t) params_base.cache_disk_gib << 30,
                std::max(1, params_base.cache_disk_min_tokens), params_base.cache_disk_ttl_days,
                server_disk_cache_fingerprint(params_base, cache_n_ctx_seq()), mctx != nullptr,
                params_base.cache_disk_ckpt_ratio, params_base.checkpoint_min_step, params_base.cache_disk_ckpt_max);
            disk_cache->live_refs = [this]() { return live_deep_refs(); };
            disk_cache->ckpt_sem_max = params_base.cache_disk_ckpt_semantic; // [TAG_SEMANTIC_CKPT]
            disk_cache->on_async_done = [this]() { queue_tasks.wake(); }; // [TAG_DISK_ASYNC_SAVE] commit on an idle loop
            disk_cache->set_async((uint64_t) params_base.cache_disk_async_mib << 20); // [TAG_DISK_ASYNC_SAVE]
            disk_cache->adopt_keys = string_split<std::string>(params_base.cache_disk_adopt, ',');
            disk_cache->set_descriptor(ctx_tgt, ctx_dft, common_speculative_type_name_str(params_base.speculative.types) +
                                      "/" + common_speculative_state_schema(spec.get()));
            disk_cache->set_fingerprint(cache_compat_key(server_disk_cache_fingerprint(params_base, cache_n_ctx_seq())));
            disk_cache->init();
            cache_compat_publish();
        } else if (disk_cache) {
            disk_cache->set_descriptor(ctx_tgt, ctx_dft, common_speculative_type_name_str(params_base.speculative.types) +
                                      "/" + common_speculative_state_schema(spec.get()));
            disk_cache->revalidate(cache_compat_key(server_disk_cache_fingerprint(params_base, cache_n_ctx_seq())));
            cache_compat_publish();
        }

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        admission_init();

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool enabled = it != params_base.default_template_kwargs.end();
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);

                // a busy pinned slot is returned as is: the caller defers the task and selection re-runs later, so
                // nothing here (similarity bookkeeping, the RAM tier, the disk hook) may touch its running state
                if (ret->is_processing()) {
                    return ret;
                }
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing());
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                // select the current slot if the criteria match
                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            f_sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find an empty slot, or the one that has been least recently used
        if (ret == nullptr) {
            const auto free_slots = get_free_slots(1, -1);

            if (!free_slots.empty()) {
                ret = free_slots[0];

                SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt.n_tokens(), ret->t_last_used);

                update_cache = true;
            }
        }

        if (ret) {
            update_cache = update_cache && prompt_cache;

            // cache prompts only for the tasks that reuse them
            update_cache = update_cache && !task.is_stateless();

            // [TAG_MULTISLOT] (item 5b) a RAM swap-in may follow: within the concurrent reserve only, else the task
            //   waits in the queue (nothing saved or loaded yet; selection re-runs on the next pass)
            if (update_cache && !admission_op_gate(*ret, SERVER_ADM_OP_RESTORE_RAM)) {
                return nullptr;
            }

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // [TAG_RAM_TIER_DISK] the RAM tier is volatile (lost on a restart, evicted by size): entering the
                //   replacement path writes the outgoing slot to disk first, regardless of the divergence rules; the
                //   disk hook below then sees the slot as the RAM load left it (its LCP is recomputed there)
                if (disk_cache && ret->prompt.n_tokens() > 0) {
                    ret->disk_save(*disk_cache);
                }

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        // [TAG_DISK_PROMPT_CACHE] a stored conversation that shares clearly more with the new prompt than this slot
        //   is loaded instead of re-processed; whatever the slot is about to lose (all of it on a load, or most of it
        //   when the new prompt diverges early) is written to disk first
        // [TAG_REPLACE_ANY_TASK] an embedding / rerank / infill may replace the whole sequence of the slot: the
        //   outgoing conversation is saved regardless of the LCP (spec: unconditional). (Codex gate r1 B3 Major 4) an
        //   infill's textual LCP does not prove its prefix is kept - infill makes no prompt checkpoints, so a rewind
        //   past a recurrent model's rollback window still resets the whole sequence; the cost is one save per
        //   infill on a slot holding another prompt (a no-op when an entry already holds exactly these tokens).
        //   No lookup / load for them
        if (ret && disk_cache && !ret->is_processing() && task.type != SERVER_TASK_TYPE_COMPLETION &&
            ret->prompt.n_tokens() > 0) {
            ret->disk_save(*disk_cache);
        }

        if (ret && disk_cache && !ret->is_processing() && task.type == SERVER_TASK_TYPE_COMPLETION) {
            // [TAG_NO_CACHE_PROMPT] cache_prompt:false re-processes the whole prompt: no lookup / load, and the slot
            //   keeps nothing of its prompt (the retained prefix counts as 0 for the outgoing save)
            const bool   reuse    = task.params.cache_prompt;
            const size_t n_slot   = ret->prompt.tokens.size();
            const size_t lcp_slot = reuse ? ret->prompt.tokens.get_common_prefix(task.tokens) : 0;
            const size_t margin   = 1024; // below this a load does not pay for itself

            // [TAG_DISK_ASYNC_SAVE] this conversation's own save still in flight (it left this or another slot just
            //   now): wait for its commit, so the lookup finds it instead of re-processing the prompt
            if (reuse) {
                disk_cache->await_pending(task.tokens, lcp_slot + margin);
            }

            const bool will_load = reuse && disk_cache->find(task.tokens, lcp_slot + margin, ret->n_ctx, nullptr) >= 0;

            // [TAG_MULTISLOT] (item 5b) a disk load allocates: within the concurrent reserve only, else the task waits
            //   in the queue before anything is saved or loaded
            if (will_load && !admission_op_gate(*ret, SERVER_ADM_OP_RESTORE_DISK)) {
                return nullptr;
            }

            // [TAG_TURN_BOUNDARY] a late branch: a whole later user turn of the resident prompt is cut off and the
            //   lost tail is worth a load. Best-effort: needs the user-message starts (message_spans of a chat
            //   request); without them (raw /completion, a legacy snapshot, a v5 entry) only the rules above apply.
            //   A normal next turn or a regenerate diverges inside the last assistant answer - no user start there
            const bool late_branch = n_slot - lcp_slot >= margin && ret->prompt.user_start_from((int32_t) lcp_slot);

            if (n_slot > 0 && (will_load || 2*lcp_slot < n_slot || late_branch)) {
                if (late_branch && !will_load && 2*lcp_slot >= n_slot) {
                    SLT_INF(*ret, "disk cache: a late branch at %zu of %zu tokens cuts off a user turn, saving\n", lcp_slot, n_slot);
                }
                ret->disk_save(*disk_cache);
            }

            if (will_load) {
                size_t lcp_disk = 0;
                const int idx = disk_cache->find(task.tokens, lcp_slot + margin, ret->n_ctx, &lcp_disk);
                if (idx >= 0) {
                    SLT_INF(*ret, "disk cache: loading a stored prompt, common prefix %zu vs %zu in the slot\n", lcp_disk, lcp_slot);
                    server_disk_load_result res = ret->disk_load(*disk_cache, idx);

                    // [TAG_DISK_LOAD_RESULT] no room in a unified pool: purge idle slots (they are saved first) and
                    //   retry until the entry loads or nothing is left to purge; the entry is addressed by its id and
                    //   kept from eviction meanwhile (a victim's save can evict entries and move the others)
                    if (res == SERVER_DISK_LOAD_CAPACITY) {
                        disk_cache->protect_id = disk_cache->entry_id(idx);
                        // [TAG_DISK_VERIFY] (Codex gate r1 A3 Major 2) the library reports no room while reading
                        //   the stream's metadata, before the tensor blocks: only a completely verified entry
                        //   (maps, every block hash, .aux, checkpoints) may cost the other slots a purge
                        ret->prompt_clear();
                        const int idx_verify = disk_cache->find_id(disk_cache->protect_id);
                        const server_disk_load_result verified =
                            idx_verify >= 0 ? disk_cache->verify(idx_verify) : SERVER_DISK_LOAD_INVALID;
                        if (verified != SERVER_DISK_LOAD_OK) {
                            SLT_WRN(*ret, "disk cache: %s did not verify, no purge\n", disk_cache->protect_id.c_str());
                            res = verified;
                        }
                        while (res == SERVER_DISK_LOAD_CAPACITY) {
                            ret->prompt_clear();
                            if (!try_clear_idle_slots()) {
                                break;
                            }
                            const int idx_retry = disk_cache->find_id(disk_cache->protect_id);
                            if (idx_retry < 0) {
                                break;
                            }
                            SLT_INF(*ret, "disk cache: retrying the load after a purge (%s)\n", disk_cache->protect_id.c_str());
                            res = ret->disk_load(*disk_cache, idx_retry);
                        }
                        disk_cache->protect_id.clear();
                    }

                    if (res != SERVER_DISK_LOAD_OK) {
                        ret->prompt_clear();
                    }
                }
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        // purge the idle slot that is the cheapest to rebuild: the one with the fewest tokens (its prompt
        // has to be re-processed if the client comes back). LRU breaks ties. The caller retries the decode
        // and purges again if this did not free enough room.
        server_slot * victim = nullptr;

        for (auto & slot : slots) {
            if (slot.is_processing() || slot.prompt.n_tokens() == 0) {
                continue;
            }

            if (victim == nullptr ||
                slot.prompt.n_tokens() <  victim->prompt.n_tokens() ||
               (slot.prompt.n_tokens() == victim->prompt.n_tokens() && slot.t_last_used < victim->t_last_used)) {
                victim = &slot;
            }
        }

        if (victim != nullptr) {
            SRV_WRN("purging slot %d with %zu tokens\n", victim->id, victim->prompt.tokens.size());
            SRV_DBG("__TEST_TAG_PURGE_SLOT_%d__\n", victim->id);

            // keep a host copy when a prompt cache exists, so the purged context can be restored instead of
            // re-processed if the client comes back (see get_available_slot)
            if (prompt_cache && victim->prompt_save(*prompt_cache)) {
                SRV_DBG("__TEST_TAG_PURGE_SAVED_%d__\n", victim->id);
                prompt_cache->update();
            }

            if (disk_cache) {
                victim->disk_save(*disk_cache); // [TAG_DISK_PROMPT_CACHE]
            }

            victim->prompt_clear();

            res = true;
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            // G32: build only the candidate window the chain can reach, instead of one llama_token_data per
            // vocabulary entry (2.98 MB per sample call on qwen4exp, ~5 calls per speculative round). The
            // selected token and the RNG stream are unchanged, but the candidate array is then a window, so it
            // has to stay off when the response reports post-sampling probabilities - that is the only place in
            // the server that reads cur_p back (populate_token_probs). Pre-sampling probabilities (n_probs
            // without post_sampling_probs, what the census sends) go through get_token_probabilities(), which
            // reads the raw logits and is unaffected.
            const bool need_post_sample_probs = task.params.sampling.n_probs > 0 && task.params.post_sampling_probs;

            common_sampler_set_fastpath(slot.smpl.get(), !need_post_sample_probs);

            bool use_backend_sampling = task.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (use_backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler fast path: %s\n", common_sampler_fastpath_str(slot.smpl.get()));
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
        }

        std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot * a, const server_slot * b) {
            if ((a->prompt.n_tokens() == 0) != (b->prompt.n_tokens() == 0)) {
                return a->prompt.n_tokens() == 0;
            }
            return a->t_last_used < b->t_last_used;
        });

        free_slots.resize(std::min(free_slots.size(), n_slots_needed));

        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            // keep the state that the child task is about to overwrite
            if (prompt_cache && slot->prompt_save(*prompt_cache)) {
                prompt_cache->update();
            }

            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    static bool env_truthy(const char * name) {
        const char * v = getenv(name);
        if (v == nullptr || *v == '\0') {
            return false;
        }
        const std::string s = v;
        return !(s == "0" || s == "false" || s == "off" || s == "no" || s == "FALSE" || s == "OFF" || s == "NO");
    }

    static double env_double(const char * name, double def) {
        const char * v = getenv(name);
        if (v == nullptr || *v == '\0') {
            return def;
        }
        char * end = nullptr;
        const double d = strtod(v, &end);
        if (end == v) {
            SRV_WRN("ignoring unparsable %s = '%s'\n", name, v);
            return def;
        }
        return d;
    }

    static bool read_u64_file(const std::string & path, uint64_t & out) {
        std::ifstream f(path);
        if (!f) {
            return false;
        }
        uint64_t v = 0;
        f >> v;
        if (f.fail()) {
            return false;
        }
        out = v;
        return true;
    }

    // [TAG_GTT_ADMISSION]
    void admission_init() {
        // [TAG_MULTISLOT] (item 5a) the shape the calibration applies to: N slots, pool C, graph, KV types, rollback
        adm.reserve_mib = std::max(0.0, env_double("LLAMA_GTT_CONCURRENT_RESERVE_MIB", 0.0));
        metrics.admission_n_slots     = slots.size();
        metrics.admission_reserve_mib = adm.reserve_mib;
        SRV_INF("admission: shape N = %zu slots, pool C = %u cells (kv_unified = %d), graph n_batch = %u n_ubatch = %u, "
                "KV types K = %s V = %s, rollback depth = %u, concurrent reserve = %.0f MiB (LLAMA_GTT_CONCURRENT_RESERVE_MIB), "
                "slope = %s B/cell, once = %s MiB\n",
                slots.size(), ctx_tgt ? llama_n_ctx(ctx_tgt) : 0, params_base.kv_unified ? 1 : 0,
                ctx_tgt ? llama_n_batch(ctx_tgt) : 0, ctx_tgt ? llama_n_ubatch(ctx_tgt) : 0,
                ggml_type_name(params_base.cache_type_k), ggml_type_name(params_base.cache_type_v),
                ctx_tgt ? llama_n_rs_seq(ctx_tgt) : 0, adm.reserve_mib,
                getenv("LLAMA_GTT_SLOPE_B_PER_CELL") ? getenv("LLAMA_GTT_SLOPE_B_PER_CELL") : "default",
                getenv("LLAMA_GTT_ONCE_MIB") ? getenv("LLAMA_GTT_ONCE_MIB") : "0");

        adm.bar_mib            = env_double("LLAMA_GTT_ADMISSION_MIB",      0.0);
        adm.slope_b_per_cell   = env_double("LLAMA_GTT_SLOPE_B_PER_CELL",   18432.0);
        adm.staging_b_per_cell = env_double("LLAMA_GTT_STAGING_B_PER_CELL", 2048.0);
        adm.step_override_mib  = env_double("LLAMA_GTT_STAGING_STEP_MIB",   -1.0);
        adm.slack_pct          = env_double("LLAMA_GTT_SLACK_PCT",          env_double("GGML_ALLOC_GROW_SLACK", 0.0));
        adm.once_mib           = std::max(0.0, env_double("LLAMA_GTT_ONCE_MIB", 0.0)); // [TAG_QSA_VIS]
        adm.max_wait_s         = env_double("LLAMA_GTT_ADMISSION_MAX_WAIT_S", 600.0);
        adm.geometric          = env_truthy("GGML_VK_QSA_SCRATCH_PRESIZE") && env_truthy("GGML_VK_QSA_SCRATCH_PRESIZE_FA");
        adm.holes_full_chunk   = env_truthy("LLAMA_GTT_HOLES_FULL_CHUNK");
        adm.enabled            = false;

        if (adm.bar_mib <= 0.0) {
            SRV_INF("%s", "admission: GTT headroom admission off (set LLAMA_GTT_ADMISSION_MIB > 0 to enable)\n");
            return;
        }

        const char * env_path = getenv("LLAMA_GTT_USED_PATH");
        if (env_path != nullptr && *env_path != '\0') {
            adm.path = env_path;
        } else {
            std::vector<std::string> cands;
            std::error_code ec;
            for (const auto & e : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
                const std::string name = e.path().filename().string();
                if (name.rfind("card", 0) == 0 && name.find('-') == std::string::npos) {
                    const std::string f = e.path().string() + "/device/mem_info_gtt_used";
                    if (std::filesystem::exists(f, ec)) {
                        cands.push_back(f);
                    }
                }
            }
            std::sort(cands.begin(), cands.end());
            if (!cands.empty()) {
                adm.path = cands.front();
            }
        }

        uint64_t used = 0;
        if (adm.path.empty() || !read_u64_file(adm.path, used)) {
            SRV_WRN("admission: GTT usage not readable (path = '%s') - headroom admission disabled\n", adm.path.c_str());
            return;
        }

        adm.enabled = true;
        adm.used_init_mib = used / 1024.0 / 1024.0;
        SRV_INF("admission: GTT headroom admission on: bar = %.0f MiB, slope = %.0f B/cell, step = max(explicit %.0f MiB, geometric %s, slack %.1f %% x slope x cells) when growing, max wait = %.0f s, holes = %s, used now = %.0f MiB (%s)\n",
                adm.bar_mib, adm.slope_b_per_cell, std::max(0.0, adm.step_override_mib),
                adm.geometric ? string_format("0.25 x %.0f B/cell x cells", adm.staging_b_per_cell).c_str() : "off",
                adm.slack_pct, adm.max_wait_s, adm.holes_full_chunk ? "full chunk" : "first-fit",
                used / 1024.0 / 1024.0, adm.path.c_str());
        if (adm.once_mib > 0.0) {
            // the proof line of the A4 accounting (a WARN, visible at the server's default verbosity)
            SRV_WRN("admission: + once %.0f MiB (LLAMA_GTT_ONCE_MIB) while GTT has grown less than that since now (%.0f MiB)\n",
                    adm.once_mib, adm.used_init_mib);
        }

        metrics.admission_enabled = 1;
        metrics.admission_bar_mib = adm.bar_mib;
    }

    // cells resident in the KV pool: every slot's cached tokens (idle slots keep theirs)
    int64_t admission_resident_cells() const {
        int64_t n = 0;
        for (const auto & s : slots) {
            n += s.prompt.n_tokens();
        }
        return n;
    }

    // [TAG_MULTISLOT] (stage 5.1 item 5b) the operation-start gate (server_admission_op_check) for slot `slot`: false =
    //   defer the operation (counted per kind, logged at most every 10 s per kind)
    int64_t adm_op_t_log_us[SERVER_ADM_OP_COUNT] = { 0 };

    bool admission_op_gate(const server_slot & slot, server_adm_op op) {
        if (!adm.enabled || adm.reserve_mib <= 0.0) {
            return true; // the default: nothing read, today's behaviour
        }
        bool concurrent = false;
        for (const auto & other : slots) {
            concurrent |= other.id != slot.id && other.is_processing();
        }
        const auto d = server_admission_op_check(adm.enabled, adm.bar_mib, adm.reserve_mib, concurrent, adm.path);
        if (!d.gated) {
            return true;
        }
        metrics.admission_last_reserve_mib = adm.reserve_mib;
        if (d.start) {
            return true;
        }
        metrics.admission_op_deferred_total[op]++;
        const int64_t t_now = ggml_time_us();
        if (adm_op_t_log_us[op] == 0 || t_now - adm_op_t_log_us[op] > 10 * 1000 * 1000) {
            adm_op_t_log_us[op] = t_now;
            SRV_WRN("admission: deferred %s of slot %d (projected %.0f MiB > bar %.0f MiB: used %.0f + concurrent reserve %.0f; another slot is processing; %" PRIu64 " deferred so far)\n",
                    server_adm_op_name(op), slot.id, d.projected_mib, adm.bar_mib, d.used_mib, adm.reserve_mib,
                    metrics.admission_op_deferred_total[op]);
        }
        return false;
    }

    // ADM_DEFER keeps the slot's prompt out of this batch (re-checked on the next iteration)
    admission_result admission_check(server_slot & slot, int32_t n_room) {
        const int64_t n_chunk = std::min<int64_t>((int64_t) slot.task->n_tokens() - slot.prompt.n_tokens(), n_room);
        if (n_chunk <= 0) {
            return ADM_ADMIT;
        }

        const int64_t t_now = ggml_time_us();

        auto admitted = [&](const char * why, double projected) {
            metrics.admission_last_deferred = 0;
            if (slot.adm_deferred) {
                SRV_INF("admission: admitted slot %d after %.1f s (%s, projected %.0f MiB, bar %.0f MiB)\n",
                        slot.id, (t_now - slot.adm_t_defer_us) / 1e6, why, projected, adm.bar_mib);
                slot.adm_deferred = false;
            }
            return ADM_ADMIT;
        };

        uint64_t used_b = 0;
        if (!read_u64_file(adm.path, used_b)) {
            SRV_WRN("admission: GTT usage no longer readable (path = '%s') - headroom admission disabled\n", adm.path.c_str());
            adm.enabled = false;
            metrics.admission_enabled = 0;
            return admitted("GTT usage unreadable, admission disabled", 0.0);
        }

        const int64_t n_ctx_pool = (int64_t) llama_n_ctx(ctx_tgt);
        const int64_t resident   = admission_resident_cells();
        adm.hw_cells = std::max(adm.hw_cells, resident);

        const int64_t cells_after = std::min(resident + n_chunk, n_ctx_pool);
        int64_t growth = std::max<int64_t>(0, cells_after - adm.hw_cells);
        if (adm.holes_full_chunk && resident < adm.hw_cells) {
            growth = n_chunk;
        }
        growth = std::min(growth, std::max<int64_t>(0, n_ctx_pool - adm.hw_cells));

        const double used_mib = used_b / 1024.0 / 1024.0;

        double step_mib = 0.0;
        if (growth > 0) {
            step_mib = std::max(step_mib, adm.step_override_mib);
            if (adm.geometric) {
                step_mib = std::max(step_mib, 0.25 * adm.staging_b_per_cell * (double) cells_after / 1024.0 / 1024.0);
            }
            step_mib = std::max(step_mib, adm.slack_pct / 100.0 * adm.slope_b_per_cell * (double) cells_after / 1024.0 / 1024.0);
        }

        // [TAG_QSA_VIS] the one-time allocation still pending (LLAMA_GTT_ONCE_MIB), only for a growing chunk like the step
        const double once_mib = growth > 0 && adm.once_mib > 0.0
            ? std::max(0.0, adm.once_mib - std::max(0.0, used_mib - adm.used_init_mib)) : 0.0;

        // [TAG_MULTISLOT] (item 5a) the concurrent-operation reserve while another slot is processing
        bool concurrent = false;
        for (const auto & other : slots) {
            concurrent |= other.id != slot.id && other.is_processing();
        }
        const double reserve_mib = concurrent ? adm.reserve_mib : 0.0;

        const double projected = used_mib + adm.slope_b_per_cell * (double) growth / 1024.0 / 1024.0 + step_mib + once_mib + reserve_mib;

        slot.adm_projected_mib = projected;
        metrics.admission_last_reserve_mib = reserve_mib;

        metrics.admission_last_used_mib      = used_mib;
        metrics.admission_last_projected_mib = projected;

        if (growth == 0 && (reserve_mib <= 0.0 || projected <= adm.bar_mib)) {
            return admitted("no new cells past the high-water", projected);
        }
        if (growth == 0) {
            metrics.admission_reserve_deferred_total++; // a zero-growth chunk held back by the concurrent reserve
        }

        if (projected <= adm.bar_mib) {
            return admitted("under the bar", projected);
        }

        if (!slot.adm_deferred) {
            slot.adm_deferred   = true;
            slot.adm_t_defer_us = t_now;
            slot.adm_t_log_us   = 0;
            metrics.admission_deferred_total++;
        }

        const double waited_s = (t_now - slot.adm_t_defer_us) / 1e6;

        if (adm.max_wait_s > 0.0 && waited_s > adm.max_wait_s) {
            SRV_WRN("admission: rejected slot %d after %.1f s (projected %.0f MiB > bar %.0f MiB, max wait %.0f s)\n",
                    slot.id, waited_s, projected, adm.bar_mib, adm.max_wait_s);
            slot.adm_deferred = false;
            metrics.admission_last_deferred = 0;
            metrics.admission_rejected_total++;
            return ADM_REJECT;
        }

        adm.deferred_this_iter = true;
        metrics.admission_last_deferred = 1;

        if (slot.adm_t_log_us == 0 || t_now - slot.adm_t_log_us > 10 * 1000 * 1000) {
            slot.adm_t_log_us = t_now;
            SRV_WRN("admission: deferred slot %d (projected %.0f MiB > bar %.0f MiB: used %.0f + %" PRId64 " new cells x %.0f B + step %.0f + once %.0f + concurrent reserve %.0f; chunk %" PRId64 " tokens, resident %" PRId64 ", high-water %" PRId64 ", waiting %.1f s)\n",
                    slot.id, projected, adm.bar_mib, used_mib, growth, adm.slope_b_per_cell, step_mib, once_mib, reserve_mib,
                    n_chunk, resident, adm.hw_cells, waited_s);
        }
        return ADM_DEFER;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        if (dbg_fault == 3) {
            dbg_fault = 0;
            throw std::runtime_error("debug fault: ckpt");
        }

        const int id_task = slot.task->id;

        // [TAG_CKPT_KEEP_SAME_POS] (v33 fix round 1) a checkpoint that already exists at this position is the one just
        //   restored (a checkpoint the live state has moved past is erased on the next rewind, and a prompt cannot pass
        //   the same position twice without one), so its content is identical - keep it instead of re-copying the
        //   recurrent state (112.6 MiB device -> host on prod) on every follow-up; only re-own it for the min-step rule.
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto & ckpt : slot.prompt.checkpoints) {
                if (ckpt.n_tokens == n_tokens_new && ckpt.pos_max == pos_max && ckpt.pos_min == pos_min) {
                    // [TAG_SLOT_FILE_CKPT_NO_DEEP] (review m5) a slot-file checkpoint keeps its origin mark: re-owning
                    //   it would let it reach the deep store when it leaves the RAM list. The min-step rule still spares
                    //   it - it is the pinned restored checkpoint for the rest of this task (ckpt_pinned_n_tokens below)
                    if (ckpt.id_task != SLOT_FILE_CKPT_ID_TASK) {
                        ckpt.id_task = id_task;
                    }
                    slot.ckpt_pinned_n_tokens = ckpt.n_tokens; // [TAG_CKPT_PIN_RESTORED]
                    SLT_INF(slot, "context checkpoint at n_tokens = %" PRId64 " already present - kept, no copy (size = %.3f MiB)\n",
                            ckpt.n_tokens, (float) ckpt.size() / 1024 / 1024);
                    return;
                }
            }
        }

        // [TAG_CKPT_28302] (v33, upstream 5d806aa25 / #28302, de-duplication moved first so that a superseded
        //   same-position checkpoint frees its entry before the capacity loop below could evict the oldest one).
        //   Retention side effect: the spacing rule no longer thins a non-full list, so a slot keeps up to
        //   n_ctx_checkpoints checkpoints (hard cap unchanged); host RAM per slot <= n_ctx_checkpoints x one
        //   checkpoint (with [TAG_CKPT_NO_DFT]: ~112.6 MiB each on the prod model -> 0.44 GiB/slot at --ctx-checkpoints 4).
        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if (it->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", it->n_tokens);
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // evict checkpoints within min-step of a previous checkpoint, unless they were
        // created by the current task
        // only when the list is full, otherwise short prompts keep just the oldest checkpoint
        int64_t last = -1;
        for (auto it = slot.prompt.checkpoints.begin();
                slot.prompt.checkpoints.size() + 1 >= (size_t) params_base.n_ctx_checkpoints &&
                it != slot.prompt.checkpoints.end(); ) {
            // [TAG_CKPT_PIN_RESTORED] the pinned restored checkpoint is never a spacing victim; it still counts as
            //   the predecessor of the next one. At capacity 1 the pin and a new checkpoint cannot both be kept (the
            //   capacity eviction below takes the pin then).
            if (it->id_task != id_task && last >= 0 && it->n_tokens <= last + params_base.checkpoint_min_step &&
                it->n_tokens != slot.ckpt_pinned_n_tokens) {
                SLT_TRC(slot, "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, (float) it->size() / 1024 / 1024);

                deep_spill(slot, *it, deep_kind_at(slot, it->n_tokens)); // [TAG_SEMANTIC_CKPT]

                it = slot.prompt.checkpoints.erase(it);
                continue;
            }

            last = it->n_tokens;
            ++it;
        }

        // [TAG_CKPT_PIN_RESTORED] (v33 fix round 2) the list is kept in ascending position order - the restore search
        //   (rbegin, first usable) and the min-step loop above rely on it - so the checkpoint this task restored is
        //   NOT moved to the back; instead the capacity eviction takes the oldest checkpoint other than it. Without
        //   this, a follow-up that restores the lowest surviving checkpoint (the branch point of the conversation)
        //   and then creates n_ctx_checkpoints new ones evicted that branch point first. Exempting all current-task
        //   entries (like the min-step loop) would not help: after a keep, every entry the follow-up adds is
        //   current-task too, so the fallback would again be the front.
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            auto victim = slot.prompt.checkpoints.begin();
            if (slot.prompt.checkpoints.size() > 1 && victim->n_tokens == slot.ckpt_pinned_n_tokens) {
                ++victim;
            }
            const auto & cur = *victim;

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB%s)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024,
                    victim == slot.prompt.checkpoints.begin() ? "" : "; skipped the pinned restored checkpoint");

            deep_spill(slot, *victim, deep_kind_at(slot, victim->n_tokens)); // [TAG_SEMANTIC_CKPT]

            slot.prompt.checkpoints.erase(victim);
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (!ckpt_skip_dft) {
            cur.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        } // else [TAG_CKPT_NO_DFT]: data_dft stays empty, the restore rewinds the draft KV instead
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_INF(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB: tgt %.3f + dft %.3f + spec %.3f)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024,
                (float) cur.data_tgt.size() / 1024 / 1024, (float) cur.data_dft.size() / 1024 / 1024,
                (float) cur.data_spec.size() / 1024 / 1024);

        // [TAG_DEEP_CKPT] the first checkpoint of a conversation (system prompt + tools, shared by every chat and agent
        //   with the same setup) and a branch point are stored at once, the rest when they leave the in-RAM list
        if (disk_cache) {
            if (cur.n_tokens == slot.ckpt_pin_pos) { // [TAG_CACHE_PIN]
                deep_spill(slot, cur, server_deep_ref::DEEP_KIND_PIN);
                cache_pin_report(slot, cur.n_tokens, "new");
                slot.ckpt_pin_pos = -1;
            } else if (cur.n_tokens == slot.ckpt_branch_pos) {
                deep_spill(slot, cur, server_deep_ref::DEEP_KIND_BRANCH);
                slot.ckpt_branch_pos = -1;
            } else if (slot.prompt.deep.empty() && slot.prompt.checkpoints.size() == 1) {
                deep_spill(slot, cur, server_deep_ref::DEEP_KIND_ANCHOR);
            }
            deep_gc();
        }
    }

    // [TAG_SLOT_FILE_CKPT] (upstream 033df86b6, #26004) the context checkpoints are appended to the slot save file, after
    //   the llama state payload: they cannot be recreated from the final state alone (a recurrent state cannot be
    //   rewound). Fork layout (version 2): magic, version, desc_tgt, desc_dft (blobs: server_cache_state_desc of the
    //   target side / of the draft side + carry schema), ident (blob: slot_file_ident(), the disk-cache namespace
    //   identity), count, count x {n_tokens, pos_min, pos_max, flags_tgt, flags_dft, tgt, dft, spec}. The flags are the
    //   ones each blob was saved with (common_prompt_checkpoint); the descriptors gate the restore like the RAM cache's
    //   guard: another target descriptor drops the appendix, another draft descriptor drops the draft blobs (all
    //   checkpoints when the drafter's carry is required, review m1). The identity gates their reuse beyond RAM: only an
    //   equal, non-empty one leaves them publishable, otherwise foreign_unverified (review M2). Version 1 (upstream, no
    //   flags / descriptors) still reads: blobs saved PARTIAL_ONLY, marked foreign_unverified (usable from RAM, never
    //   written into own storage). Restored checkpoints carry id_task = SLOT_FILE_CKPT_ID_TASK (their origin)
    static constexpr uint32_t SLOT_CKPT_MAGIC      = 0x504b4353; // "SCKP"
    static constexpr uint32_t SLOT_CKPT_VERSION_V1 = 1;          // upstream
    static constexpr uint32_t SLOT_CKPT_VERSION    = 2;          // fork: + descriptors + identity + per-blob flags
    static constexpr int      SLOT_FILE_CKPT_ID_TASK = SERVER_SLOT_FILE_CKPT_ID_TASK; // the origin mark of a checkpoint read from a slot file

    static bool ckpt_read(std::ifstream & ifs, void * dst, size_t size, size_t & n_read) {
        if (!ifs.read((char *) dst, size)) {
            return false;
        }
        n_read += size;
        return true;
    }

    static bool ckpt_read_buf(std::ifstream & ifs, std::vector<uint8_t> & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
        }
        buf.resize(n);
        return n == 0 || ckpt_read(ifs, buf.data(), n, n_read);
    }

    static void ckpt_write(std::ofstream & ofs, const void * src, size_t size, size_t & n_written) {
        ofs.write((const char *) src, size);
        n_written += size;
    }

    static void ckpt_write_buf(std::ofstream & ofs, const std::vector<uint8_t> & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
        }
    }

    static std::vector<uint8_t> slot_file_bytes(const std::string & s) {
        return std::vector<uint8_t>(s.begin(), s.end());
    }

    // [TAG_SLOT_FILE_CKPT] the state descriptors of the target side and of the draft side (+ the carry schema)
    std::string slot_file_desc_tgt() const {
        return server_cache_state_desc(ctx_tgt, nullptr, "");
    }

    std::string slot_file_desc_dft() const {
        return server_cache_state_desc(nullptr, ctx_dft, spec ? common_speculative_state_schema(spec.get()) : std::string());
    }

    // [TAG_SLOT_FILE_CKPT] (review M2) the disk-cache namespace identity minus the state layout (in the descriptors):
    //   model files (sampled content + size), the numerics measurement, the semantic env and the parsed-param digest -
    //   the parts of cache_compat_key(). Empty without a disk cache (nothing is published then) or without a
    //   measurement: an empty identity never matches
    std::string slot_file_ident() const {
        if (!disk_cache || cache_meas.empty()) {
            return std::string();
        }
        return "model=" + server_disk_cache_model_identity(params_base) + "|meas=" + server_cache_measure_component(cache_meas) +
               "|env=" + server_cache_env_digest(server_cache_environ()) + "|param=" +
               server_cache_param_digest(params_base, ctx_tgt, ctx_dft);
    }

    // returns false if the appendix could not be written completely
    bool save_slot_checkpoints(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        // a device snapshot is no byte stream, an adopted pre-format-8 one has no full-descriptor proof: not written
        std::vector<const common_prompt_checkpoint *> ckpts;
        for (const auto & cur : slot.prompt.checkpoints) {
            if (((cur.flags_tgt | cur.flags_dft) & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) || cur.foreign_unverified || cur.data_tgt.empty()) {
                continue;
            }
            ckpts.push_back(&cur);
        }
        if (ckpts.empty()) {
            return true;
        }
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic   = SLOT_CKPT_MAGIC;
        const uint32_t version = SLOT_CKPT_VERSION;
        const uint32_t count   = (uint32_t) ckpts.size();
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, slot_file_bytes(slot_file_desc_tgt()), n_written);
        ckpt_write_buf(ofs, slot_file_bytes(slot_file_desc_dft()), n_written);
        ckpt_write_buf(ofs, slot_file_bytes(slot_file_ident()),    n_written);
        ckpt_write(ofs, &count,   sizeof(count),   n_written);
        for (const auto * cur : ckpts) {
            const uint32_t flags_tgt = cur->flags_tgt;
            const uint32_t flags_dft = cur->flags_dft;
            ckpt_write(ofs, &cur->n_tokens, sizeof(cur->n_tokens), n_written);
            ckpt_write(ofs, &cur->pos_min,  sizeof(cur->pos_min),  n_written);
            ckpt_write(ofs, &cur->pos_max,  sizeof(cur->pos_max),  n_written);
            ckpt_write(ofs, &flags_tgt,     sizeof(flags_tgt),     n_written);
            ckpt_write(ofs, &flags_dft,     sizeof(flags_dft),     n_written);
            ckpt_write_buf(ofs, cur->data_tgt,  n_written);
            ckpt_write_buf(ofs, cur->data_dft,  n_written);
            ckpt_write_buf(ofs, cur->data_spec, n_written);
        }
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n",
                count, (float) n_written / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed (the appendix parsed completely, its checkpoints kept or dropped), 0 if
    //   there is no well-formed appendix at offset
    size_t load_slot_checkpoints(const std::string & filepath, size_t offset, server_slot & slot) const {
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset; // bytes after the llama state payload
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        uint32_t count   = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_CKPT_MAGIC) {
            return 0;
        }
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) ||
            (version != SLOT_CKPT_VERSION && version != SLOT_CKPT_VERSION_V1)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        const bool v1 = version == SLOT_CKPT_VERSION_V1;
        std::vector<uint8_t> desc_tgt;
        std::vector<uint8_t> desc_dft;
        std::vector<uint8_t> ident;
        if ((!v1 && (!ckpt_read_buf(ifs, desc_tgt, n_avail, n_read) || !ckpt_read_buf(ifs, desc_dft, n_avail, n_read) ||
                     !ckpt_read_buf(ifs, ident, n_avail, n_read))) ||
            !ckpt_read(ifs, &count, sizeof(count), n_read)) {
            SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        // (review M2) a layout match alone is no proof of the same model / numerics: only the namespace identity is
        const std::string ident_own = slot_file_ident();
        const bool foreign = v1 || ident_own.empty() || std::string(ident.begin(), ident.end()) != ident_own;
        std::list<common_prompt_checkpoint> checkpoints;
        for (uint32_t i = 0; i < count; ++i) {
            common_prompt_checkpoint cur;
            cur.id_task = SLOT_FILE_CKPT_ID_TASK; // not created by a task - read from a slot file (review m1: its origin)
            uint32_t flags_tgt = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY; // v1: the upstream server saves them PARTIAL_ONLY
            uint32_t flags_dft = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
            if (!ckpt_read(ifs, &cur.n_tokens, sizeof(cur.n_tokens), n_read) ||
                !ckpt_read(ifs, &cur.pos_min,  sizeof(cur.pos_min),  n_read) ||
                !ckpt_read(ifs, &cur.pos_max,  sizeof(cur.pos_max),  n_read) ||
                (!v1 && !ckpt_read(ifs, &flags_tgt, sizeof(flags_tgt), n_read)) ||
                (!v1 && !ckpt_read(ifs, &flags_dft, sizeof(flags_dft), n_read)) ||
                !ckpt_read_buf(ifs, cur.data_tgt,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_dft,  n_avail, n_read) ||
                !ckpt_read_buf(ifs, cur.data_spec, n_avail, n_read)) {
                SLT_WRN(slot, "truncated context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            // a saved checkpoint always holds a target state - an empty blob would roll back without restoring
            //   anything; a device snapshot is never a file stream
            if (cur.data_tgt.empty() || ((flags_tgt | flags_dft) & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
                SLT_WRN(slot, "invalid context checkpoint appendix in '%s' - ignored\n", filepath.c_str());
                return 0;
            }
            cur.flags_tgt = flags_tgt;
            cur.flags_dft = cur.data_dft.empty() ? 0 : flags_dft;
            cur.foreign_unverified = foreign; // [TAG_CACHE_COMPAT2] no identity proof: RAM use only
            // a checkpoint must lie inside the restored prompt (its key is a prefix of these tokens)
            if (cur.n_tokens <= 0 || cur.n_tokens > (int64_t) slot.prompt.tokens.size()) {
                continue;
            }
            checkpoints.push_back(std::move(cur));
            if (checkpoints.size() > (size_t) std::max(0, params_base.n_ctx_checkpoints)) {
                checkpoints.pop_front();
            }
        }
        if (!v1 && std::string(desc_tgt.begin(), desc_tgt.end()) != slot_file_desc_tgt()) {
            SLT_WRN(slot, "%zu context checkpoint(s) in '%s' were saved by another target context - dropped\n",
                    checkpoints.size(), filepath.c_str());
            return n_read;
        }
        // (review m1) without its draft blobs a checkpoint has no carry either (clear_dft): with a drafter whose carry
        //   is required (MTP, eagle3 boundary) it could not restore later - drop such checkpoints instead
        const bool carry_required = spec && common_speculative_n_stateful(spec.get()) > 0;
        auto drop_draft_side = [&](const char * why) {
            if (carry_required) {
                SLT_WRN(slot, "%zu context checkpoint(s) in '%s': %s and the drafter's carry is required - dropped\n",
                        checkpoints.size(), filepath.c_str(), why);
                checkpoints.clear();
                return;
            }
            SLT_WRN(slot, "draft data of the context checkpoints in '%s': %s - dropped\n", filepath.c_str(), why);
            for (auto & cur : checkpoints) {
                cur.clear_dft();
            }
        };
        if (!v1 && std::string(desc_dft.begin(), desc_dft.end()) != slot_file_desc_dft()) {
            drop_draft_side("saved by another draft context");
        }
        // the draft blobs are not covered by the payload checks - test-load one, drop the draft data if it does not fit
        //   (the draft sequence is reset here; the draft-state appendix, when present, is loaded after this)
        if (ctx_dft != nullptr && !checkpoints.empty() && !checkpoints.back().data_dft.empty()) {
            const bool ok = checkpoints.back().load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            if (!ok) {
                drop_draft_side("the draft data does not match the draft context");
            }
        }
        slot.prompt.checkpoints = std::move(checkpoints);
        SLT_INF(slot, "restored %zu context checkpoint(s) from '%s'%s\n", slot.prompt.checkpoints.size(), filepath.c_str(),
                foreign ? " (no namespace-identity proof: RAM use only)" : "");
        return n_read;
    }

    // [TAG_SLOT_FILE_DRAFT] (upstream ea736c1d5, #30574) the live draft state follows the checkpoints: the full draft
    //   context and the speculative carry (MTP's pending_h, eagle3's boundary; draft-simple has none). The llama state
    //   payload holds the target context only. Fork layout (version 3; version 2 = the same layout without a provenance proof): magic, version, desc_tgt, desc_dft, ident
    //   (blobs, see [TAG_SLOT_FILE_CKPT]), draft_gap (i32), bits (u32: 1 = the prompt was `partial`, 2 = `noncanonical`), dft, spec. Version 1
    //   (upstream: magic, version, dft, spec) still reads. A draft state that does not fit the draft context, another
    //   draft descriptor or a carry the drafter rejects leave the legacy restore (draft sequence and carry reset, the
    //   prompt partial), as does a file without the section. (review M1) The draft state is streamed between the
    //   context and the file (llama_state_seq_write_cb / read_cb, as the disk cache): at 1M the MTP draft state is
    //   ~2.2 GiB and is never held in host RAM as a whole: save and restore move it in pieces of at most 16 MiB
    //   (SLOT_DRAFT_CHUNK here, LLAMA_STATE_READ_CHUNK in llama_io_read_cb::read_tensor, astra pkg2 Important)
    static constexpr uint32_t SLOT_DRAFT_MAGIC      = 0x54464453; // "SDFT"
    static constexpr uint32_t SLOT_DRAFT_VERSION_V1 = 1;          // upstream
    static constexpr uint32_t SLOT_DRAFT_VERSION_V2 = 2;          // fork r1/r2: + descriptors + identity + draft coverage
    // (astra pkg2 delta r3) version 3 = the version 2 layout with a known provenance: every save writes the section
    //   (also without a draft state) and its `noncanonical` bit is reliable. A version 2 section (pkg2 r1 wrote a shifted
    //   prompt with bits 0) or a file without the section has no provenance proof: the draft may restore warm, but the
    //   prompt stays `partial` - never published - on every profile (draft model, n-gram, none)
    static constexpr uint32_t SLOT_DRAFT_VERSION    = 3;
    static constexpr uint32_t SLOT_DRAFT_BIT_PARTIAL = 1u << 0;
    // (astra pkg2 Critical) the prompt was `noncanonical` (context shift / n_cache_reuse shift): the target KV is not
    //   the state of its token prefix - the restore keeps the mark, so neither the slot nor its checkpoints are ever
    //   published (disk save, deep_put) until a full recompute clears it, as in the live slot
    static constexpr uint32_t SLOT_DRAFT_BIT_NONCANONICAL = 1u << 1;
    static constexpr size_t   SLOT_DRAFT_CHUNK       = 16u << 20; // tensor ranges go to the file in 16 MiB pieces
    // the dft blob is the llama_state_seq_get_data_ext() stream: its 8-byte head (the sequence-state magic, io_magic in
    //   src/llama-context.cpp, + the source seq_id) and the state_seq_write_data body that write_cb / read_cb carry
    static constexpr uint32_t SLOT_DRAFT_SEQ_MAGIC   = 0xaf143cd8;

    // returns false if the draft state could not be written completely
    bool save_slot_draft(const std::string & filepath, const server_slot & slot, size_t & n_written) const {
        n_written = 0;
        const uint64_t n_dft = ctx_dft != nullptr ? llama_state_seq_get_size_ext(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
        std::vector<uint8_t> data_spec;
        // one carry is serializable (as the RAM cache): with more stateful implementations it stays empty and the
        //   restore resets the draft side
        if (spec && common_speculative_n_stateful(spec.get()) <= 1) {
            common_speculative_get_state(spec.get(), slot.id, data_spec);
        }
        // (astra pkg2 delta r3) the section is always written, draft-less when there is no draft state: it is the
        //   provenance record (bits) without which the restore keeps the prompt unpublishable
        std::ofstream ofs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::app);
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s'\n", filepath.c_str());
            return false;
        }
        const uint32_t magic     = SLOT_DRAFT_MAGIC;
        const uint32_t version   = SLOT_DRAFT_VERSION;
        const int32_t  draft_gap = slot.prompt.draft_gap;
        const uint32_t bits      = (slot.prompt.partial      ? SLOT_DRAFT_BIT_PARTIAL      : 0u) |
                                   (slot.prompt.noncanonical ? SLOT_DRAFT_BIT_NONCANONICAL : 0u);
        ckpt_write(ofs, &magic,   sizeof(magic),   n_written);
        ckpt_write(ofs, &version, sizeof(version), n_written);
        ckpt_write_buf(ofs, slot_file_bytes(slot_file_desc_tgt()), n_written);
        ckpt_write_buf(ofs, slot_file_bytes(slot_file_desc_dft()), n_written);
        ckpt_write_buf(ofs, slot_file_bytes(slot_file_ident()),    n_written);
        ckpt_write(ofs, &draft_gap, sizeof(draft_gap), n_written);
        ckpt_write(ofs, &bits,      sizeof(bits),      n_written);
        ckpt_write(ofs, &n_dft,     sizeof(n_dft),     n_written);
        if (n_dft > 0) {
            // the same byte stream as llama_state_seq_get_data_ext(FLAGS_NONE), written as it is produced
            uint64_t n_stream = 0;
            std::vector<uint8_t> chunk;
            llama_state_seq_writer w;
            w.write = [&](const void * src, size_t size) {
                ofs.write((const char *) src, size);
                n_stream += size;
            };
            w.write_tensor = [&](ggml_tensor * t, size_t offset, size_t size) {
                chunk.resize(std::min(size, SLOT_DRAFT_CHUNK));
                for (size_t done = 0; done < size; ) {
                    const size_t n = std::min(size - done, chunk.size());
                    ggml_backend_tensor_get(t, chunk.data(), offset + done, n);
                    ofs.write((const char *) chunk.data(), n);
                    done += n;
                }
                n_stream += size;
            };
            const uint32_t     seq_magic = SLOT_DRAFT_SEQ_MAGIC;
            const llama_seq_id seq_src   = slot.id;
            w.write(&seq_magic, sizeof(seq_magic));
            w.write(&seq_src,   sizeof(seq_src));
            const size_t n_cb = llama_state_seq_write_cb(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE, w);
            n_written += n_stream;
            if (n_cb == 0 || n_cb + sizeof(seq_magic) + sizeof(seq_src) != n_dft || n_stream != n_dft) {
                SLT_WRN(slot, "the draft state for '%s' streamed %" PRIu64 " of %" PRIu64 " bytes - the file is incomplete\n",
                        filepath.c_str(), n_stream, n_dft);
                return false;
            }
        }
        ckpt_write_buf(ofs, data_spec, n_written);
        ofs.flush();
        if (!ofs) {
            SLT_WRN(slot, "failed to append the draft state to '%s' - the draft state is incomplete\n", filepath.c_str());
            return false;
        }
        SLT_INF(slot, "appended the draft state (%.3f MiB: dft %.3f + spec %.3f) to '%s'\n", (float) n_written / 1024 / 1024,
                (float) n_dft / 1024 / 1024, (float) data_spec.size() / 1024 / 1024, filepath.c_str());
        return true;
    }

    // returns the number of bytes consumed (the section parsed completely, its state restored or dropped), 0 if there
    //   is no well-formed draft state at offset. Called after the legacy restore reset the draft sequence and the
    //   carry and marked the prompt partial (speculative profiles): only a restored draft state lifts that
    size_t load_slot_draft(const std::string & filepath, size_t offset, server_slot & slot) const {
        std::ifstream ifs(std::filesystem::u8path(filepath), std::ios::binary | std::ios::ate);
        const size_t file_size = ifs ? (size_t) ifs.tellg() : 0;
        if (!ifs || file_size < offset || !ifs.seekg(offset)) {
            return 0;
        }
        const size_t n_avail = file_size - offset;
        size_t n_read = 0;
        uint32_t magic   = 0;
        uint32_t version = 0;
        if (!ckpt_read(ifs, &magic, sizeof(magic), n_read) || magic != SLOT_DRAFT_MAGIC) {
            return 0;
        }
        std::vector<uint8_t> desc_tgt;
        std::vector<uint8_t> desc_dft;
        std::vector<uint8_t> ident;
        int32_t  draft_gap = 0;
        uint32_t bits      = SLOT_DRAFT_BIT_PARTIAL; // v1: no identity proof - never published (see below)
        uint64_t n_dft     = 0;
        std::vector<uint8_t> data_spec;
        if (!ckpt_read(ifs, &version, sizeof(version), n_read) ||
            (version != SLOT_DRAFT_VERSION && version != SLOT_DRAFT_VERSION_V2 && version != SLOT_DRAFT_VERSION_V1)) {
            SLT_WRN(slot, "invalid draft state in '%s' - ignored\n", filepath.c_str());
            return 0;
        }
        const bool v1 = version == SLOT_DRAFT_VERSION_V1;
        const bool provenance_known = version == SLOT_DRAFT_VERSION; // (r3) v1 / v2: never published
        if ((!v1 && (!ckpt_read_buf(ifs, desc_tgt, n_avail, n_read) ||
                     !ckpt_read_buf(ifs, desc_dft, n_avail, n_read) ||
                     !ckpt_read_buf(ifs, ident,    n_avail, n_read) ||
                     !ckpt_read(ifs, &draft_gap, sizeof(draft_gap), n_read) ||
                     !ckpt_read(ifs, &bits, sizeof(bits), n_read))) ||
            !ckpt_read(ifs, &n_dft, sizeof(n_dft), n_read) || n_dft > n_avail - n_read) {
            // (astra pkg2 Critical) a damaged v2 head hides the provenance bits: the restored state is never published
            //   (legacy restore otherwise - the tokens are the payload's)
            if (!v1) {
                slot.prompt.noncanonical = true;
            }
            SLT_WRN(slot, "invalid draft state in '%s' - ignored%s\n", filepath.c_str(), v1 ? "" : ", the prompt is not published");
            return 0;
        }
        // (astra pkg2 Critical) the provenance holds whatever happens to the draft side below: the target state is the
        //   payload's, and a shifted payload stays unpublishable (prompt.clear() at the restore reset the mark)
        if ((bits & SLOT_DRAFT_BIT_NONCANONICAL) != 0) {
            slot.prompt.noncanonical = true;
            SLT_INF(slot, "the slot file '%s' holds a shifted (noncanonical) state - never published\n", filepath.c_str());
        }
        // drop the draft side: whatever the checkpoint test-load left, and a partially streamed state
        auto drop = [&](const char * why) {
            if (ctx_dft != nullptr) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            }
            if (spec) {
                common_speculative_reset_state(spec.get(), slot.id);
            }
            SLT_WRN(slot, "draft state in '%s' %s - dropped, the draft starts empty\n", filepath.c_str(), why);
        };
        const char * why = nullptr;
        if (!v1 && std::string(desc_dft.begin(), desc_dft.end()) != slot_file_desc_dft()) {
            why = "was saved by another draft context";
        } else if (ctx_dft == nullptr ? n_dft > 0 : n_dft == 0) {
            why = ctx_dft == nullptr ? "has a draft context state but this server has no draft context" : "has no draft context state";
        } else if (n_dft > 0) {
            // stream the draft state from the file into the draft context, bounded by its stored size
            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
            uint64_t n_left = n_dft;
            uint32_t     seq_magic = 0;
            llama_seq_id seq_src   = 0;
            llama_state_seq_reader r;
            r.read = [&](void * dst, size_t size) {
                if (size > n_left || !ifs.read((char *) dst, size)) {
                    return false;
                }
                n_left -= size;
                return true;
            };
            if (!r.read(&seq_magic, sizeof(seq_magic)) || !r.read(&seq_src, sizeof(seq_src)) || seq_magic != SLOT_DRAFT_SEQ_MAGIC) {
                why = "has no sequence state stream";
            } else {
                const size_t n_cb = llama_state_seq_read_cb(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_NONE, r);
                if (n_cb == 0 || n_left != 0) {
                    why = "does not match the draft context";
                }
            }
        }
        // past the draft state, whatever was consumed of it
        ifs.clear();
        if (!ifs.seekg((std::streamoff) (offset + n_read + n_dft))) {
            drop("is truncated");
            return 0;
        }
        n_read += n_dft;
        if (!ckpt_read_buf(ifs, data_spec, n_avail, n_read)) {
            drop("is truncated");
            return 0;
        }
        // a provenance-only section (no draft context here or at save, no carry): nothing to restore on the draft side,
        //   only the publication rule below applies
        const bool draft_less = why == nullptr && ctx_dft == nullptr && n_dft == 0 && data_spec.empty();
        if (why != nullptr) {
            drop(why);
            return n_read;
        }
        if (!draft_less && spec && !common_speculative_set_state(spec.get(), slot.id, data_spec)) {
            // the carry belongs to the draft history just restored: without it the two disagree (the RAM swap-in rule)
            drop("has a speculative carry the drafter does not restore");
            return n_read;
        }
        // the draft side is the saved one: the prompt is as complete as it was at save. Publishing (partial = false)
        //   additionally needs the target descriptor and the namespace identity (review M2: a layout match is no proof
        //   of the same model / numerics) - otherwise the warm draft stays, but the prompt is never published
        const std::string ident_own = slot_file_ident();
        // (r3) and a known provenance: only a version 3 section of a canonical prompt clears `partial`
        const bool tgt_ok = provenance_known && (bits & SLOT_DRAFT_BIT_NONCANONICAL) == 0 &&
                            std::string(desc_tgt.begin(), desc_tgt.end()) == slot_file_desc_tgt() &&
                            !ident_own.empty() && std::string(ident.begin(), ident.end()) == ident_own;
        slot.prompt.partial   = !tgt_ok || (bits & SLOT_DRAFT_BIT_PARTIAL) != 0;
        slot.prompt.draft_gap = (int32_t) std::clamp<int64_t>(draft_gap, 0, (int64_t) slot.prompt.tokens.size());
        SLT_INF(slot, "restored the %s (%.3f MiB: dft %.3f + spec %.3f) from '%s'%s%s\n",
                draft_less ? "slot-file provenance" : "draft state", (float) n_read / 1024 / 1024,
                (float) n_dft / 1024 / 1024, (float) data_spec.size() / 1024 / 1024, filepath.c_str(),
                slot.prompt.partial ? (provenance_known ? ", the prompt stays partial" : ", the prompt stays partial (no provenance proof: a pre-r3 section)")
                                    : (slot.prompt.noncanonical ? ", noncanonical (never published)" : ""),
                slot.prompt.draft_partial() ? ", draft-partial" : "");
        return n_read;
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    // [TAG_MULTISLOT] qualification mode: a parent/child (n_cmpl > 1) request would copy one slot's KV
                    //   into others (shared physical prefix, outside the envelope) - rejected before any slot is touched
                    if (multislot_strict && task.is_parent()) {
                        SRV_WRN("[multislot-envelope] rejected shared-prompt request id_task=%d (n_cmpl=%zu, LLAMA_QSA_HOMOGENEOUS=1)\n",
                                id_task, task.child_tasks.size() + 1);
                        send_error(task, "n_cmpl > 1 (shared-prompt parent/child slots) is not supported with LLAMA_QSA_HOMOGENEOUS=1 and n_parallel > 1",
                                   ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        // [TAG_REPLACE_ANY_TASK] the child slots are overwritten by copy_state_to() (and cleared on
                        //   release): their conversations are saved now that all required slots are available, before
                        //   the launch - not inside the copy, which a cancel or a launch failure would never reach
                        if (disk_cache) {
                            for (server_slot * child : child_slots) {
                                if (child->prompt.n_tokens() > 0) {
                                    child->disk_save(*disk_cache);
                                }
                            }
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR] [TAG_RAM_TIER_DISK] the cleared slot survives a restart
                                    if (disk_cache && slot.prompt.n_tokens() > 0) {
                                        slot.disk_save(*disk_cache);
                                    }
                                    slot.prompt_clear();
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release_cancel = true;
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    if (disk_cache) {
                        disk_cache->poll(); // [TAG_DISK_ASYNC_SAVE] finished background saves show as committed
                    }

                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    // get_metrics() fills in the process-wide sampling counters (G32)
                    res->metrics             = get_metrics();

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    // [TAG_PARTIAL_PROMPT] /slots save writes the target context's sequence and the tokens, then the
                    //   prompt checkpoints ([TAG_SLOT_FILE_CKPT]) and the draft state + speculative carry
                    //   ([TAG_SLOT_FILE_DRAFT]) as appendices - no deep refs, no turn boundaries. A file without the
                    //   draft state (v62 and older) or with a pre-r3 one restores partial on every profile and is not
                    //   published to the disk cache until recomputed (SLOT_RESTORE below)
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_ckpt = 0;
                    if (!save_slot_checkpoints(filepath, *slot, nwrite_ckpt)) {
                        send_error(task, "Unable to save slot: incomplete context checkpoints", ERROR_TYPE_SERVER);
                        break;
                    }

                    size_t nwrite_dft = 0;
                    if (!save_slot_draft(filepath, *slot, nwrite_dft)) {
                        send_error(task, "Unable to save slot: incomplete draft state", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite + nwrite_ckpt + nwrite_dft;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // [TAG_MULTISLOT] (item 5b) the restore allocates: within the concurrent reserve only
                    if (!admission_op_gate(*slot, SERVER_ADM_OP_RESTORE_SLOT)) {
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    if (disk_cache) {
                        slot->disk_save(*disk_cache); // [TAG_DISK_PROMPT_CACHE] the restore overwrites the slot
                    }

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);

                        // [TAG_PARTIAL_PROMPT] the payload holds the target state only: the draft context's sequence
                        //   and the drafter's carry belong to the previous conversation - drop them ([TAG_SPEC_RESET]);
                        //   the empty draft KV is not rebuilt for the restored prefix, so the prompt is partial unless
                        //   the file's draft-state appendix restores the draft side (load_slot_draft below)
                        if (ctx_dft) {
                            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot->id, -1, -1);
                        }
                        if (spec) {
                            common_speculative_reset_state(spec.get(), slot->id);
                        }
                        // (astra pkg2 delta r3) on every profile, also n-gram / no draft: until a version 3 draft-state
                        //   section proves the provenance (load_slot_draft below), the restored state is not published
                        slot->prompt.partial = true;
                        SLT_INF(*slot, "legacy restore of %zu tokens: draft sequence and carry reset, the prompt is partial (not published to the disk cache until fully recomputed or proven by the slot file)\n",
                                slot->prompt.tokens.size());
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    // nread is the end offset of the llama state payload within the file
                    const size_t nread_ckpt = load_slot_checkpoints(filepath, nread, *slot);
                    const size_t nread_dft  = load_slot_draft(filepath, nread + nread_ckpt, *slot);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread + nread_ckpt + nread_dft;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // no disk save here: an erase is the client discarding the slot (pi after a failed restore,
                    // a released hand), unlike a restore that overwrites someone else's conversation

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear();

                    // [TAG_DISK_GC] a transaction boundary: deep checkpoints only this slot referenced are collected
                    deep_gc();

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                if (iterate_rethrow) {
                    throw;
                }
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
                invalidate_slot(slot);
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                if (iterate_rethrow) {
                    throw;
                }
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
                invalidate_slot(*slot);
            }
        }
    }

    // [TAG_SLOT_INVALIDATE] a slot released on a failure path: its sequence (target + draft) may hold part of a batch,
    //   so it is cleared now and once more after this pass's decode, which may still write it (same end state as the
    //   decode-error branch in decode())
    void invalidate_slot(server_slot & slot) {
        slot.prompt_clear();
        slots_invalidated.push_back(slot.id);
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            // a slot with tokens in the batch was selected as processing in this pass, even if it was released since
            bool in_batch = false;
            for (const auto & t : batch.tokens) {
                if (t.id_slot == slot.id) {
                    in_batch = true;
                    break;
                }
            }
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
                invalidate_slot(slot);
            } else if (in_batch) {
                invalidate_slot(slot);
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;

    // [TAG_SPEC_ROUND_TIMING] (R80.1 v49) wall times of the current update_slots pass; a pass that ran the drafter is one
    // speculative round. Timers only: nothing here feeds back into scheduling, sampling or the batch.
    struct spec_round_timing {
        int64_t t0_us      = 0;
        int64_t draft_us   = 0; // the common_speculative_draft yield block
        int64_t catchup_us = 0; // the common_speculative_process yield blocks of the pass's batch views
        int64_t verify_us  = 0; // the llama_decode (+ sync) yield blocks of the pass's batch views
        std::vector<int> slot_ids; // slots that drafted in this pass (empty: not a round)
        // [TAG_SPEC_DRAFT_POLICY] tput: slots that took a true-zero round, the drafted n requested, the target rows
        //   decoded and the rows a pure round of the single drafting / zero slot has (1 + its draft)
        std::vector<int> zero_ids;
        int n_draft_req   = 0;
        int n_rows        = 0;
        int n_rows_expect = 0;
        int n_kv          = 0; // the depth of the round's slot: its fits bucket
        int n_rows_gen    = 0; // tput: target rows of the pass's true-zero and replay slots (the rest: prefill)
        int n_draft_att   = 0; // tput: draft positions sampled (the draft fit's x), tokens drafted (the ckpt fit's x)
        int n_draft_out   = 0;
        bool cold         = false; // tput: the slot's first drafting round of the task (allocation / graph warm-up)
        // tput: slots whose pass is the replay after a checkpoint restore, that restore's wall, and the restores
        //   done in this pass (not host time of this pass: charged to the replay that follows)
        std::vector<int> replay_ids;
        int64_t replay_restore_us = 0;
        int64_t restore_us        = 0;
        int64_t ckpt_us           = 0; // tput: the per-round checkpoint save + draft-context restore of a drafting round
    } spec_rt;

    // [TAG_SPEC_DRAFT_POLICY] one single-slot pure round -> the cost fits
    void spec_tput_feed() {
        const size_t n_slots = spec_rt.slot_ids.size() + spec_rt.zero_ids.size() + spec_rt.replay_ids.size();
        if (n_slots != 1 || spec_rt.n_rows <= 0 || spec_rt.n_rows != spec_rt.n_rows_expect || spec_rt.cold) {
            return;
        }
        const int64_t t_round = ggml_time_us() - spec_rt.t0_us;
        if (!spec_rt.replay_ids.empty()) {
            // the complete rollback cycle: the restore (in the rejecting pass) + this whole replay pass
            spec_tput_fits.at(spec_rt.n_kv).rb.add(spec_rt.n_rows, (spec_rt.replay_restore_us + t_round) / 1e3);
            return;
        }
        const int64_t t_host  = std::max<int64_t>(0, t_round - spec_rt.draft_us - spec_rt.catchup_us - spec_rt.verify_us -
                                                     spec_rt.restore_us - spec_rt.ckpt_us);
        auto & f = spec_tput_fits.at(spec_rt.n_kv);
        if (!spec_rt.slot_ids.empty()) {
            f.draft .add(spec_rt.n_draft_att, spec_rt.draft_us / 1e3);
            f.ckpt  .add(spec_rt.n_draft_out, spec_rt.ckpt_us  / 1e3);
            f.host_d.add(spec_rt.n_rows,      t_host           / 1e3);
        } else {
            f.host  .add(0.0,                 t_host           / 1e3); // the host intercept: zero rounds only
        }
        f.verify .add(spec_rt.n_rows, spec_rt.verify_us  / 1e3);
        f.catchup.add(spec_rt.n_rows, spec_rt.catchup_us / 1e3);
    }

    // [TAG_SPEC_DRAFT_POLICY] the draft length of this round for a slot that may draft up to n_draft_max (0 = true zero)
    int spec_tput_n(server_slot & slot, int n_draft_max) {
        const auto & dp = params_base.speculative.draft;
        const int n_max_cfg = std::min(n_draft_max, common_speculative_n_max(spec.get()));
        // slope priors while one row count dominates the samples: the depth-aware model's cost of one drafted token,
        // half on the draft call and half on the verify decode
        const int    n_kv  = spec_n_kv_depth(slot);
        const double t_tok = (double) dp.n_max_adaptive_t_tok + (double) dp.n_max_adaptive_k_tok*(double) n_kv;
        common_spec_tput_cost c;
        // the fits of this slot's depth bucket only (samples from other depths never price this one); a bucket
        //   that is not warm yet takes the prior path below (the configured length, cooldown still on)
        auto & f = spec_tput_fits.at(n_kv);
        const bool ok = f.cost(0.5*t_tok, c);
        c.n_rb_free = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ? 0
                    : ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS   ? (int) llama_n_rs_seq(ctx_tgt)
                    : (1 << 30);
        double rate = 0.0;
        // before the fits are usable, or when they give no valid cost model (non-finite / non-positive costs): the
        //   prior path = today's length (n_max), the cooldown still applies
        // candidates {0} U [n_lo, n_max]: the drafter clears a draft shorter than --spec-draft-n-min
        const int n_lo = std::max(1, dp.n_min);
        // a zero must beat n_max by the margin (v51 r2, Codex sol §4: the fits carry no performance bound by themselves)
        static const double zero_margin = [] {
            const char * e = getenv("LLAMA_SPEC_TPUT_MARGIN");
            const double m = e ? atof(e) : COMMON_SPEC_TPUT_MARGIN_DEFAULT;
            return std::isfinite(m) && m >= 0.0 ? m : COMMON_SPEC_TPUT_MARGIN_DEFAULT;
        }();
        int n_rule = ok ? common_spec_tput_best_n(common_spec_tput_p_eff(slot.spec_acc_ema, slot.spec_acc_nobs),
                                                  n_max_cfg, c, &rate, n_lo, zero_margin) : -1;
        if (n_rule < 0) {
            n_rule = n_max_cfg;
        }
        // v51 r2 (Codex sol §4 Major): without rank in this bucket the fits do not identify the slopes a zero or a
        //   shorter draft extrapolates along - the policy is `off` there (n_max, no zero, no cooldown entry: the
        //   cooldown is dropped and the round is not observed), except the exploration rounds that build the rank
        const bool ranked = f.ranked();
        slot.spec_tput_ranked = ranked;
        int n = 0;
        slot.spec_tput_probe = false;
        if (!ranked) {
            slot.spec_cd = common_spec_cooldown();
            n_rule = common_spec_tput_explore_n(f, std::min(n_lo, n_max_cfg), n_max_cfg);
            n      = n_rule;
        } else {
            const bool was_cooling = slot.spec_cd.cooling();
            n = std::min(slot.spec_cd.decide(n_rule), n_max_cfg);
            if (was_cooling && n > 0) {
                // the probe: the shortest length the drafter keeps (none fits the remaining space: no probe this round)
                n = common_spec_tput_probe_n(dp.n_min, n_max_cfg);
            }
            if (was_cooling && n > 0) {
                slot.stats.n_spec_probe_rounds++;
                slot.spec_tput_probe = true; // v51 r2 (Hermes): the n_min rule must not clear the probe
            }
        }
        if (trace > 0 && n != slot.spec_n_max_last) {
            SLT_INF(slot, "draft policy tput: %d -> %d (rule %d, %s, cooldown %d, n_kv %d bucket %d, draft %.2f+%.2fn, verify %.2f+%.2fm, "
                    "catch-up %.2f+%.2fm, ckpt %.2f+%.2fn, host %.2f ms, rollback %.2f+%.2fr free %d, %.4f tok/ms)\n", slot.spec_n_max_last, n, n_rule,
                    !ranked ? "no rank" : ok ? "measured" : "warm-up", slot.spec_cd.c, n_kv, common_spec_tput_bucket(n_kv), c.d0, c.d1, c.v0, c.v1, c.c0, c.c1, c.k0, c.k1, c.host,
                    c.r0, c.r1, c.n_rb_free, rate);
        }
        slot.spec_n_max_last = n;
        slot.spec_tput_n_req = n;
        spec_rt.n_kv         = n_kv;

        slot.spec_tput_active   = true;
        slot.spec_tput_measured = ok && ranked;
        slot.spec_tput_rule     = n_rule;
        slot.spec_tput_n_kv     = n_kv;
        slot.spec_tput_rate     = rate;
        slot.spec_tput_cost     = c;
        const common_spec_ew_line * fl[8] = { &f.draft, &f.ckpt, &f.verify, &f.catchup, &f.host, &f.host_d, &f.rb, &f.restore };
        for (int k = 0; k < 8; ++k) {
            slot.spec_tput_fit_n[k] = fl[k]->n_samples;
        }
        metrics.spec_adaptive_n_max = (uint64_t) n;
        return n;
    }

    // end of a pass: a round -> the slots' stats, the /metrics counters and the main context's kqvis-prof window
    void spec_round_finish() {
        if (params_base.speculative.draft.policy == COMMON_SPEC_DRAFT_POLICY_TPUT) {
            spec_tput_feed();
            if (spec_rt.slot_ids.empty() && (!spec_rt.zero_ids.empty() || !spec_rt.replay_ids.empty())) {
                // no drafting slot, but true-zero / replay rows: their catch-up is theirs, the prefill rows' share of a
                //   mixed pass (by rows) keeps the prefill accounting
                const int     n_gen = std::min(spec_rt.n_rows_gen, spec_rt.n_rows);
                const int64_t t_pf  = spec_rt.n_rows > 0 ? spec_rt.catchup_us * (spec_rt.n_rows - n_gen) / spec_rt.n_rows : 0;
                const int64_t t_gen = spec_rt.catchup_us - t_pf;
                if (t_pf > 0) {
                    metrics.n_spec_catchup_pf_passes += 1;
                    metrics.t_spec_catchup_pf_us     += t_pf;
                    llama_kqvis_prof_add_catchup_pf(ctx_tgt, t_pf);
                }
                if (!spec_rt.zero_ids.empty()) {
                    metrics.n_spec_zero_passes     += 1;
                    metrics.t_spec_zero_catchup_us += spec_rt.replay_ids.empty() ? t_gen : t_gen / 2;
                }
                if (!spec_rt.replay_ids.empty()) {
                    metrics.n_spec_replay_passes     += 1;
                    metrics.t_spec_replay_catchup_us += spec_rt.zero_ids.empty() ? t_gen : t_gen - t_gen / 2;
                }
                spec_rt.catchup_us = 0;
                spec_rt.zero_ids.clear();
                spec_rt.replay_ids.clear();
                return;
            }
        }
        if (spec_rt.slot_ids.empty()) {
            if (spec_rt.catchup_us > 0) { // a pass without a drafting slot (prompt processing): its catch-up, separately
                metrics.n_spec_catchup_pf_passes += 1;
                metrics.t_spec_catchup_pf_us     += spec_rt.catchup_us;
                llama_kqvis_prof_add_catchup_pf(ctx_tgt, spec_rt.catchup_us);
                spec_rt.catchup_us = 0;
            }
            return;
        }
        const int64_t t_round  = ggml_time_us() - spec_rt.t0_us;
        const int64_t t_host   = std::max<int64_t>(0, t_round - spec_rt.draft_us - spec_rt.catchup_us - spec_rt.verify_us);
        for (const int id : spec_rt.slot_ids) {
            auto & st = slots[id].stats;
            st.n_spec_timed_rounds += 1;
            st.t_spec_draft_us     += spec_rt.draft_us;
            st.t_spec_catchup_us   += spec_rt.catchup_us;
            st.t_spec_verify_us    += spec_rt.verify_us;
            st.t_spec_host_us      += t_host;
        }
        metrics.n_spec_timed_rounds += 1;
        metrics.t_spec_draft_us     += spec_rt.draft_us;
        metrics.t_spec_catchup_us   += spec_rt.catchup_us;
        metrics.t_spec_verify_us    += spec_rt.verify_us;
        metrics.t_spec_host_us      += t_host;
        llama_kqvis_prof_add_spec_round(ctx_tgt, spec_rt.draft_us, spec_rt.catchup_us, spec_rt.verify_us, t_host);
        spec_rt.slot_ids.clear();
    }
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    // [TAG_MULTISLOT] (stage 5.1 item 4a) homogeneous routing-class scheduling (LLAMA_QSA_HOMOGENEOUS=1, np > 1).
    //   The QSA dense/sparse topology is chosen per UBATCH from its deepest sequence (qwen4exp_qsa_depth), also under
    //   per-stream KV, so sequences below and at/above N = LLAMA_QSA_MIN_KV never share a decode call: after pre_decode()
    //   appended every slot's run, the whole runs are stable-partitioned into homogeneous ranges and each range is
    //   decoded, caught up (MTP) and post-decoded on its own. Classes: depth = pos + 1 of a run's rows; below = every
    //   row < N, above = every row >= N (equality at N is above); a run straddling N, or a slot whose target history
    //   does not start at 0 / does not continue its cached positions (restore, rewind hole) is a singleton range.
    //   Runs of <= LLAMA_MTP_QSA_MAX_UBATCH rows are grouped only while the range total stays <= it (the MTP catch-up
    //   / draft limit counts TOTAL tokens), larger runs only with larger runs. (r3) Target ranges take that cap only
    //   when the catch-up runs attention or LLAMA_QSA_HOMOGENEOUS_TARGET_CAP=1 (server_qsa_route_target_cap); drafts
    //   always. Compatible runs are sorted by slot (=
    //   sequence / stream) id: split_equal needs consecutive increasing ids.
    static int64_t qsa_env_min_kv() {
        static const int64_t v = [] {
            const char * s = getenv("LLAMA_QSA_MIN_KV");
            const long long n = s ? atoll(s) : 0;   // same normalisation as qwen4exp_qsa_min_kv()
            return (int64_t) (n > 0 ? n : 0);
        }();
        return v;
    }

    static int64_t qsa_env_mtp_max_ubatch() {
        static const int64_t v = [] {
            const char * s = getenv("LLAMA_MTP_QSA_MAX_UBATCH");
            const long long n = s ? atoll(s) : 8;   // same normalisation as qwen4exp_mtp_qsa_max_ubatch()
            return (int64_t) (n > 0 ? n : 0);
        }();
        return v;
    }

    // target ranges (r3): the cap only with an attention catch-up or LLAMA_QSA_HOMOGENEOUS_TARGET_CAP=1
    static int64_t qsa_env_target_max_ubatch() {
        static const int64_t v = server_qsa_route_target_cap(qsa_env_mtp_max_ubatch(),
                getenv("LLAMA_QSA_HOMOGENEOUS_TARGET_CAP"), getenv("LLAMA_MTP_CATCHUP_ATTN"));
        return v;
    }

    static int qsa_route_class(int64_t d_lo, int64_t d_hi) {
        return server_qsa_route_class(d_lo, d_hi, qsa_env_min_kv());
    }

    uint64_t qsa_route_rounds   = 0; // rounds with >= 1 range
    uint64_t qsa_route_mixed    = 0; // rounds with > 1 range
    uint64_t qsa_route_n_log    = 0;
    std::string qsa_route_desc;      // the current round's range description (logged per range)

    struct qsa_route_run {
        int32_t id_slot;
        std::vector<int32_t> rows;
        llama_pos p_lo = 0;
        llama_pos p_hi = 0;
        bool partial = false; // partial target history: a singleton
    };

    void qsa_route_partition() {
        batch.range_end.clear();
        batch.run_end.clear();
        if (!multislot_strict || batch.size() == 0) {
            return;
        }

        // whole runs per slot, in the order of first appearance (within-slot order preserved)
        std::vector<qsa_route_run> runs;
        std::vector<int32_t> run_of(slots.size(), -1);
        for (int32_t i = 0; i < batch.size(); ++i) {
            const auto & t = batch.tokens[i];
            GGML_ASSERT(t.id_slot >= 0 && t.id_slot < (int32_t) slots.size());
            int32_t & r = run_of[t.id_slot];
            if (r < 0) {
                r = (int32_t) runs.size();
                runs.push_back({ t.id_slot, {}, t.pos, t.pos });
            }
            auto & run = runs[r];
            run.rows.push_back(i);
            run.p_lo = std::min(run.p_lo, t.pos);
            run.p_hi = std::max(run.p_hi, t.pos);
        }

        const int64_t max_ub = qsa_env_target_max_ubatch();
        llama_memory_t mem = llama_get_memory(ctx_tgt);
        for (auto & run : runs) {
            // partial history: a draft-partial prompt (target-only load / adoption, [TAG_DRAFT_PARTIAL]), or rows not
            // continuing the cached positions (a hole). NOT llama_memory_seq_pos_min: on the hybrid memory it reports
            // the recurrent part's single (latest) position, > 0 for every complete history.
            const llama_pos c_max = llama_memory_seq_pos_max(mem, run.id_slot);
            if (slots[run.id_slot].prompt.draft_partial() || run.p_lo != c_max + 1) {
                run.partial = true;
            }
        }

        std::vector<server_qsa_route_run> infos;
        for (const auto & run : runs) {
            infos.push_back({ run.id_slot, (int32_t) run.rows.size(), run.p_lo, run.p_hi, run.partial, run.rows.front() });
        }
        const auto groups = server_qsa_route_groups(infos, qsa_env_min_kv(), max_ub);

        std::vector<int32_t> order;
        order.reserve(batch.size());
        qsa_route_desc.clear();
        for (const auto & g : groups) {
            std::string gd;
            for (int32_t r : g) {
                const auto & run = runs[r];
                order.insert(order.end(), run.rows.begin(), run.rows.end());
                batch.run_end.push_back((int32_t) order.size());
                gd += string_format("%s%d:%d-%d", gd.empty() ? "" : ",", run.id_slot, (int) run.p_lo, (int) run.p_hi);
            }
            const auto & r0 = runs[g.front()];
            const int cls = r0.partial ? -1 : qsa_route_class((int64_t) r0.p_lo + 1, (int64_t) r0.p_hi + 1);
            qsa_route_desc += string_format(" {%s %s}", cls < 0 ? "single" : cls ? "above" : "below", gd.c_str());
            batch.range_end.push_back((int32_t) order.size());
        }
        GGML_ASSERT((int32_t) order.size() == batch.size());

        // i_batch is remapped only when it points at one of the slot's own rows of this batch (else it is stale / -1)
        std::vector<bool> own_row(slots.size(), false);
        for (size_t k = 0; k < slots.size(); ++k) {
            const int32_t ib = slots[k].i_batch;
            own_row[k] = ib >= 0 && ib < batch.size() && batch.tokens[ib].id_slot == slots[k].id;
        }
        const auto map = batch.permute(order);
        for (size_t k = 0; k < slots.size(); ++k) {
            auto & slot = slots[k];
            if (own_row[k]) {
                slot.i_batch = map[slot.i_batch];
            }
            for (auto & i : slot.spec_i_batch) {
                GGML_ASSERT(i >= 0 && i < (int32_t) map.size());
                i = map[i];
            }
        }

        qsa_route_rounds++;
        if (groups.size() > 1) {
            qsa_route_mixed++;
        }
    }

    // [TAG_MULTISLOT] (item 4a) routing groups of the drafting slots: a draft step's row of a slot sits at n_past + i
    //   (i < n_max), depth n_past + 1 .. n_past + n_max; a proposal whose possible range straddles N, or a slot with a
    //   partial target history, drafts alone; a group holds at most LLAMA_MTP_QSA_MAX_UBATCH slots (one row each per
    //   step, TOTAL tokens); sorted by slot id
    std::vector<std::vector<server_slot *>> qsa_route_draft_groups(const std::vector<server_slot *> & drafting) {
        std::vector<server_qsa_route_run> infos;
        std::string desc;
        for (size_t k = 0; k < drafting.size(); ++k) {
            const server_slot * s = drafting[k];
            const auto & dp = common_speculative_get_draft_params(spec.get(), s->id);
            const int32_t n_max = dp.n_max > 0 ? dp.n_max : std::max(1, params_base.speculative.draft.n_max);
            const bool partial = s->prompt.draft_partial(); // see qsa_route_partition()
            // one row per step: rows at n_past .. n_past + n_max - 1
            infos.push_back({ s->id, 1, dp.n_past, dp.n_past + n_max - 1, partial, (int32_t) k });
            const int cls = partial ? -1 : qsa_route_class((int64_t) dp.n_past + 1, (int64_t) dp.n_past + n_max);
            desc += string_format(" %d:%d+%d=%s", s->id, (int) dp.n_past, (int) n_max, cls < 0 ? "single" : cls ? "above" : "below");
        }
        const auto groups = server_qsa_route_groups(infos, qsa_env_min_kv(), qsa_env_mtp_max_ubatch());
        std::vector<std::vector<server_slot *>> res;
        for (const auto & g : groups) {
            res.emplace_back();
            for (const int32_t r : g) {
                res.back().push_back(drafting[r]);
            }
        }
        if (res.size() > 1 && (qsa_route_n_log < 64 || qsa_route_n_log % 256 == 0)) {
            SRV_INF("[qsa-route] draft groups=%zu:%s\n", res.size(), desc.c_str());
        }
        return res;
    }

    // [TAG_MULTISLOT] (item 4a) the decode length at off: clamped to its routing range; on a capacity retry
    //   (n_batch < the rest of the range) cut at a whole-slot run - a run that alone exceeds n_batch may only be split
    //   when it has at most one output row (a prompt chunk), never a verification run
    int32_t qsa_route_clamp(int32_t off, int32_t n_tokens) const {
        if (batch.range_end.empty()) {
            return n_tokens;
        }
        const int32_t r_end = *std::upper_bound(batch.range_end.begin(), batch.range_end.end(), off);
        if (off + n_tokens >= r_end) {
            return r_end - off;
        }
        int32_t cut = off;
        for (const int32_t e : batch.run_end) {
            if (e > off && e - off <= n_tokens) {
                cut = e;
            }
        }
        if (cut > off) {
            return cut - off;
        }
        // the run at off is longer than n_batch
        const int32_t e = *std::upper_bound(batch.run_end.begin(), batch.run_end.end(), off);
        int n_out = 0;
        for (int32_t i = off; i < e; ++i) {
            n_out += batch.tokens[i].output ? 1 : 0;
        }
        if (n_out > 1) {
            throw std::runtime_error(string_format("[qsa-route] the verification run of slot %d (%d rows) does not fit n_batch %d",
                    batch.tokens[off].id_slot, e - off, n_tokens));
        }
        return n_tokens;
    }

    void update_slots() {
        slots_invalidated.clear();

        if (disk_cache) {
            disk_cache->poll(); // [TAG_DISK_ASYNC_SAVE] commit the background saves that finished
        }

#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        if (spec) { // [TAG_SPEC_ROUND_TIMING] the pass starts
            spec_rt.t0_us      = ggml_time_us();
            spec_rt.draft_us   = 0;
            spec_rt.catchup_us = 0;
            spec_rt.verify_us  = 0;
            spec_rt.slot_ids.clear();
            spec_rt.zero_ids.clear();
            spec_rt.n_draft_req   = 0;
            spec_rt.n_rows        = 0;
            spec_rt.n_rows_expect = 0;
            spec_rt.n_kv          = 0;
            spec_rt.n_rows_gen    = 0;
            spec_rt.n_draft_att   = 0;
            spec_rt.n_draft_out   = 0;
            spec_rt.cold          = false;
            spec_rt.replay_ids.clear();
            spec_rt.replay_restore_us = 0;
            spec_rt.restore_us        = 0;
            spec_rt.ckpt_us           = 0;
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            qsa_route_partition(); // [TAG_MULTISLOT] (item 4a) before rendering
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // [TAG_PREDECODE_ABORT] the batch is half-built and not rendered: discard it (rows of the aborted slots,
            //   slot_batched) so nothing of it is decoded, and skip now to avoid UB
            batch.clear();
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        // [TAG_GTT_ADMISSION] everything runnable is deferred: nothing to decode (the decode loop below is skipped),
        //   so back off a little instead of spinning on NEXT_RESPONSE
        if (adm.deferred_this_iter && batch.size() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                n_tokens = qsa_route_clamp(off, n_tokens); // [TAG_MULTISLOT] (item 4a)

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
                if (!batch.range_end.empty() && (qsa_route_n_log < 64 || qsa_route_n_log % 256 == 0 || !ok)) {
                    std::string ids;
                    for (int32_t i = off; i < off + n_tokens; ++i) {
                        if (i == off || batch.tokens[i].id_slot != batch.tokens[i - 1].id_slot) {
                            ids += string_format("%s%d@%d", ids.empty() ? "" : ",", batch.tokens[i].id_slot, (int) batch.tokens[i].pos);
                        }
                    }
                    SRV_INF("[qsa-route] round=%" PRIu64 " ranges=%zu off=%d n=%d slots=[%s] decode=%s (mixed rounds %" PRIu64 ") plan:%s\n",
                            qsa_route_rounds, batch.range_end.size(), off, n_tokens, ids.c_str(), ok ? "ok" : "retry",
                            qsa_route_mixed, qsa_route_desc.c_str());
                }
                qsa_route_n_log++;
                if (ok && dbg_fault == 2 && batch_view.n_tokens == 1) { // the first generation step
                    dbg_fault = 0;
                    throw std::runtime_error("debug fault: decode");
                }
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }

        spec_round_finish(); // [TAG_SPEC_ROUND_TIMING]

        // [TAG_SLOT_INVALIDATE] what the batch wrote for a slot invalidated in this pass goes too
        for (const int32_t id : slots_invalidated) {
            server_slot & slot = slots[id];
            if (!slot.is_processing()) {
                slot.prompt_clear();
            }
        }
    }

    void pre_decode() {
        adm.deferred_this_iter = false;

        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);

                    // [TAG_NONCANONICAL] the KV past n_keep was computed with the discarded tokens before it
                    slot.prompt.noncanonical  = true;
                    slot.ckpt_pinned_n_tokens = -1;
                    slot.ckpt_branch_pos      = -1;
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // [TAG_PREDECODE_ABORT] from here on a per-slot failure aborts the whole pass (see update_slots())
        iterate_rethrow = true;
        struct rethrow_reset {
            bool & f;
            ~rethrow_reset() { f = false; }
        } rethrow_guard { iterate_rethrow };

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    // stale candidates: a replay never reads them, a new draft refills them
                    slot.spec_draft_q.clear();

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                        // [TAG_SPEC_DRAFT_POLICY] tput: the replay after a restore closes its round's rollback cycle
                        if (params_base.speculative.draft.policy == COMMON_SPEC_DRAFT_POLICY_TPUT &&
                                slot.spec_is_replay && slot.spec_n_committed == slot.spec_draft.size()) {
                            spec_rt.replay_ids.push_back(slot.id);
                            spec_rt.replay_restore_us = slot.spec_rb_restore_us;
                            spec_rt.n_kv              = spec_n_kv_depth(slot);
                            spec_rt.n_rows_expect     = 1 + (int) slot.spec_draft.size();
                            spec_rt.n_rows_gen       += 1 + (int) slot.spec_draft.size();
                            slot.spec_rb_restore_us   = 0;
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        // [TAG_SPEC_DRAFT_POLICY] tput: the length first; a true zero is a plain decode (no draft
                        //   call, no per-round checkpoint, process() keeps the draft KV), like n_draft_max == 0
                        int n_tput = -1;
                        if (params_base.speculative.draft.policy == COMMON_SPEC_DRAFT_POLICY_TPUT) {
                            n_tput = spec_tput_n(slot, n_draft_max);
                            if (n_tput == 0) {
                                slot.stats.n_spec_zero_rounds++;
                                spec_rt.zero_ids.push_back(slot.id);
                                spec_rt.n_rows_expect = 1;
                                spec_rt.n_rows_gen   += 1;
                                return;
                            }
                        }

                        // [TAG_MULTISLOT] (item 5b) draft + speculative checkpoint work starts only within the
                        //   concurrent reserve; deferred = a plain decode this round (no draft call, no checkpoint)
                        if (!admission_op_gate(slot, SERVER_ADM_OP_DRAFT)) {
                            return;
                        }

                        const int64_t t_ckpt0 = n_tput > 0 ? ggml_time_us() : 0; // [TAG_SPEC_DRAFT_POLICY] ckpt cost

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                        }

                        if (n_tput > 0) {
                            spec_rt.ckpt_us += ggml_time_us() - t_ckpt0;
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        int n_draft_req = n_draft_max;

                        if (n_tput > 0) {
                            n_draft_req = std::min(n_draft_req, n_tput);
                        } else if (params_base.speculative.draft.n_max_adaptive) {
                            const auto & dparams = params_base.speculative.draft;

                            // [TAG_SPEC_ADAPTIVE_DEPTH] the cost of a drafted token and of the decode it
                            // might save both grow with the KV depth - evaluate them at this slot's depth,
                            // or take the server's own measured round times when the fit is usable
                            const int n_kv = spec_n_kv_depth(slot);

                            float t_tok = dparams.n_max_adaptive_t_tok + dparams.n_max_adaptive_k_tok*(float) n_kv;
                            float t_dec = dparams.n_max_adaptive_t_dec + dparams.n_max_adaptive_k_dec*(float) n_kv;

                            const bool measured = dparams.n_max_adaptive_measure &&
                                spec_cost.solve(n_kv,
                                        dparams.n_max_adaptive_t_tok, dparams.n_max_adaptive_k_tok,
                                        dparams.n_max_adaptive_t_dec, dparams.n_max_adaptive_k_dec,
                                        t_tok, t_dec);

                            const int n_max_cfg = common_speculative_n_max(spec.get());

                            float p_cum = 1.0f;
                            int n_adapt = slot.spec_n_max_adaptive(n_max_cfg, t_tok, t_dec, &p_cum);

                            // --spec-draft-n-min: the drafter throws away a draft shorter than n_min, so
                            // shortening below it would silently disable speculation instead of cheapening it
                            n_adapt = std::max(n_adapt, std::min(dparams.n_min, n_max_cfg));

                            // never above the configured n-max (the recurrent rollback depth n_rs_seq is
                            // sized from it at startup) and never below 1 - both are enforced above, this
                            // only adds the remaining context space
                            n_draft_req = std::min(n_draft_req, n_adapt);

                            if (trace > 0 && n_adapt != slot.spec_n_max_last) {
                                std::string ema;
                                for (size_t i = 0; i < slot.spec_acc_ema.size(); ++i) {
                                    ema += string_format(i ? " %.3f" : "%.3f", slot.spec_acc_ema[i]);
                                }
                                SLT_INF(slot, "adaptive draft len: %d -> %d (p_cum %.3f, ema [%s], n_kv %d, "
                                        "t_tok %.2f ms, t_dec %.2f ms, %s)\n",
                                        slot.spec_n_max_last, n_adapt, p_cum, ema.c_str(), n_kv,
                                        t_tok, t_dec, measured ? "measured" : "model");
                            }

                            slot.spec_n_max_last    = n_adapt;
                            slot.spec_cost_t_tok    = t_tok;
                            slot.spec_cost_t_dec    = t_dec;
                            slot.spec_cost_n_kv     = n_kv;
                            slot.spec_cost_measured = measured;

                            metrics.spec_adaptive_n_max    = (uint64_t) n_adapt;
                            metrics.spec_adaptive_n_kv     = (uint64_t) n_kv;
                            metrics.spec_adaptive_t_tok_ms = t_tok;
                            metrics.spec_adaptive_t_dec_ms = t_dec;
                            metrics.spec_adaptive_measured = measured ? 1 : 0;
                        }

                        // [TAG_B68_DRAFT_KNOBS] H2 auto depth (only when neither tput nor the cost model decides)
                        slot.spec_depth_req = 0;
                        if (n_tput <= 0 && !params_base.speculative.draft.n_max_adaptive && server_spec_depth_auto() > 0) {
                            const int n_cfg = std::min(n_draft_max, common_speculative_n_max(spec.get()));
                            if (slot.spec_depth_cur <= 0) {
                                slot.spec_depth_cur = server_spec_depth_auto();
                            }
                            // floor at --spec-draft-n-min: the drafter throws away a draft shorter than n_min, so a
                            // lower auto depth would silently turn speculation off instead of cheapening it
                            const int n_floor = std::max(1, std::min(params_base.speculative.draft.n_min, n_cfg));
                            slot.spec_depth_cur = std::max(n_floor, std::min(slot.spec_depth_cur, n_cfg));
                            n_draft_req         = std::min(n_draft_req, slot.spec_depth_cur);
                            slot.spec_depth_req = n_draft_req;
                        }

                        common_speculative_get_draft_params(spec.get(), slot.id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_req,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };
                        {
                            // [TAG_UP27694] the candidates are asked for only at T > 0 under the flag; the drafter
                            //   itself drops the request when --spec-draft-sampling is greedy (the default)
                            auto & dpr = common_speculative_get_draft_params(spec.get(), slot.id);
                            dpr.result_q = params_base.speculative.draft.probabilistic && slot.use_spec_rejection() ?
                                           &slot.spec_draft_q : nullptr;
                            dpr.temp     = slot.task->params.sampling.temp;
                            dpr.seed     = slot.task->params.sampling.seed;
                            // [TAG_B68_DRAFT_KNOBS] H5a a round verified by sample-and-match keeps the greedy depth
                            const int n_greedy = server_spec_n_max_greedy(params_base.speculative.draft.n_min,
                                                                          params_base.speculative.draft.n_max);
                            if (dpr.result_q == nullptr && n_greedy > 0) {
                                dpr.n_max = std::min(dpr.n_max > 0 ? dpr.n_max : n_draft_max, n_greedy);
                            }
                            // [TAG_B68_DRAFT_KNOBS] H2 judges the round against the depth the drafter was really allowed
                            //   (after the greedy cap): recorded before the cap, a capped T = 0 round never reached the
                            //   requested depth, so the auto depth could only ratchet down
                            if (slot.spec_depth_req > 0) {
                                slot.spec_depth_req = dpr.n_max;
                            }
                        }
                        // [TAG_SPEC_DRAFT_POLICY] a tput cooldown probe is kept as drafted (false on every other path)
                        common_speculative_get_draft_params(spec.get(), slot.id).keep_short = n_tput > 0 && slot.spec_tput_probe;

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            const int64_t t_draft0 = ggml_time_us(); // [TAG_SPEC_ROUND_TIMING]
            // [TAG_MULTISLOT] (item 4a) draft generation homogeneous too: the draft decode steps put one row per drafting
            //   sequence into one ubatch (common/speculative.cpp), so the drafting slots run in the same routing groups
            //   - one common_speculative_draft() per group with only its members' dparams.drafting enabled. The wrapper
            //   resets every flag to false at its end, so a later group re-enables only its own (not yet run, empty
            //   result) members; a completed proposal is never re-enabled.
            std::vector<std::vector<server_slot *>> dgroups;
            if (multislot_strict && drafting.size() > 1) {
                dgroups = qsa_route_draft_groups(drafting);
            } else {
                dgroups.push_back(drafting);
            }
            for (size_t g = 0; g < dgroups.size(); ++g) {
                if (dgroups.size() > 1) {
                    for (server_slot * s : drafting) {
                        auto & dp = common_speculative_get_draft_params(spec.get(), s->id);
                        const bool member = std::find(dgroups[g].begin(), dgroups[g].end(), s) != dgroups[g].end();
                        GGML_ASSERT(!member || dp.result->empty()); // not yet drafted
                        dp.drafting = member;
                    }
                }
                queue_tasks.yield_to_queue([&]() {
                    common_speculative_draft(spec.get());
                });
            }
            spec_rt.draft_us += ggml_time_us() - t_draft0;
            for (const server_slot * s : drafting) {
                spec_rt.slot_ids.push_back(s->id);
            }
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();
            // [TAG_B68_DRAFT_KNOBS] H3 measurement: tokens drafted after a drafted EOG (the verifier never takes them)
            for (size_t k = 0; k + 1 < draft.size(); ++k) {
                if (llama_vocab_is_eog(vocab, draft[k])) {
                    slot.stats.n_draft_post_eog += draft.size() - 1 - k;
                    break;
                }
            }

            if (params_base.speculative.draft.policy == COMMON_SPEC_DRAFT_POLICY_TPUT) {
                // [TAG_SPEC_DRAFT_POLICY] what the drafter attempted (a drafter that does not report it: the request)
                const auto & dpo = common_speculative_get_draft_params(spec.get(), slot.id);
                slot.spec_tput_n_att  = dpo.n_attempted >= 0 ? dpo.n_attempted : slot.spec_tput_n_req;
                slot.spec_tput_n_kept = dpo.n_kept      >= 0 ? dpo.n_kept      : (int) draft.size();
                spec_rt.cold          = slot.spec_tput_n_draft_rounds++ == 0;
                spec_rt.n_draft_req   = slot.spec_tput_n_req;
                spec_rt.n_draft_att   = slot.spec_tput_n_att;
                spec_rt.n_draft_out   = (int) draft.size();
                spec_rt.n_rows_expect = 1 + (int) draft.size();
            }
            if (draft.empty()) {
                slot.stats.n_spec_empty_drafts += 1;
                if (params_base.speculative.draft.policy == COMMON_SPEC_DRAFT_POLICY_TPUT) {
                    // [TAG_SPEC_DRAFT_POLICY] an empty draft: a stop at position 0 is a failure there (and a failed
                    //   round for the cooldown); a draft the n_min rule cleared stopped at n_kept > 0 (positions before
                    //   it were never verified: unobserved; only the stop at n_kept is) and is no cooldown observation
                    //   (v51 r2, Hermes: counted as a failure it could keep a cooldown alive when n_min > 1)
                    const int n_max_cfg = common_speculative_n_max(spec.get());
                    const auto kind = common_spec_tput_empty_draft(slot.spec_tput_n_att, slot.spec_tput_n_kept);
                    if (kind == COMMON_SPEC_TPUT_EMPTY_FAIL0) {
                        slot.spec_acc_update(1, 0, n_max_cfg);
                        if (slot.spec_tput_ranked) {
                            slot.spec_cd.observe(0, 0);
                        }
                    } else if (kind == COMMON_SPEC_TPUT_EMPTY_STOP_K) {
                        slot.spec_acc_observe_stop((size_t) slot.spec_tput_n_kept, n_max_cfg);
                    }
                }
            }

            // [TAG_SPEC_DRAFT_POLICY] tput: the draft-context restore and the per-round checkpoints below are the
            //   checkpoint cost of a drafting round (charged to n > 0 only, not host time)
            const bool    tput_ckpt = params_base.speculative.draft.policy == COMMON_SPEC_DRAFT_POLICY_TPUT;
            const int64_t t_ckpt0   = tput_ckpt ? ggml_time_us() : 0;

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    if (!ckpt.load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
                        GGML_ABORT("speculative draft checkpoint size mismatch\n"); // this run's own state: a bug
                    }
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

                    // [TAG_MTP_CKPT_PENDING_H] (audit B#2b) the speculative pending state (MTP pending_h, eagle3
                    //   boundary row) travels with the per-round checkpoint too, as with the prompt checkpoints
                    //   (create_checkpoint): the verification's process() overwrites it with the batch's last row,
                    //   and a restore below replays the committed tokens from this position. Taken after the
                    //   draft, which only reads it.
                    ckpt.data_spec.clear();
                    common_speculative_get_state(spec.get(), slot.id, ckpt.data_spec);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                }
            }

            if (tput_ckpt) {
                spec_rt.ckpt_us += ggml_time_us() - t_ckpt0;
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        slot.b0_pending       = true;
                        slot.b0_n_cached_live = slot.prompt.n_tokens();
                        slot.b0_ckpt_n_tokens = -1;
                        slot.b0_full_reset    = false;
                        slot.b0_prefix_match  = 0;
                        slot.ckpt_pinned_n_tokens = -1;
                        slot.ckpt_branch_pos      = -1;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            // [TAG_NONSPLIT_REPLACE] n_past stays 0: the whole sequence is replaced, so the previous
                            //   conversation's checkpoints, deep refs and pin go before the new tokens are observable
                            slot.prompt.checkpoints.clear();
                            slot.prompt.deep.clear();
                            slot.ckpt_pinned_n_tokens = -1;
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt && !slot.task->is_stateless()) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    int64_t first_edit = -1; // [TAG_NONCANONICAL] first position whose KV was shifted

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            if (first_edit < 0) {
                                                first_edit = (int64_t) head_p;
                                            }

                                            slot.mem.seq_rm (slot.id, head_p, head_c);
                                            slot.mem.seq_add(slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    // [TAG_NONCANONICAL] the shifted chunks were computed in another context: the prompt
                                    //   is noncanonical, and checkpoints / deep refs / pin / branch marker past the first
                                    //   edited position describe a sequence that no longer exists
                                    if (first_edit >= 0) {
                                        slot.prompt.noncanonical = true;
                                        slot.prompt.user_starts.clear(); // [TAG_TURN_BOUNDARY] positions moved
                                        slot.prompt.checkpoints.remove_if([&](const auto & c) { return c.n_tokens > first_edit; });
                                        slot.prompt.deep.erase(std::remove_if(slot.prompt.deep.begin(), slot.prompt.deep.end(),
                                                    [&](const auto & r) { return r.n_tokens > first_edit; }), slot.prompt.deep.end());
                                        if (slot.ckpt_pinned_n_tokens > first_edit) {
                                            slot.ckpt_pinned_n_tokens = -1;
                                        }
                                        if (slot.ckpt_branch_pos > first_edit) {
                                            slot.ckpt_branch_pos = -1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            slot.b0_prefix_match = n_past;

                            // [TAG_PREFIX_DIVERGE] the client changed already-cached history: one line with the text
                            //   around the first differing token, so a re-prefill can be traced to its cause (a reordered
                            //   tool list, a dropped message, a re-rendered answer) without LLAMA_SERVER_SLOTS_DEBUG
                            if (slot.task->params.cache_prompt && slot.task->params.n_cache_reuse == 0 &&
                                n_past < (int) slot.prompt.tokens.size() && n_past < slot.task->n_tokens()) {
                                const auto around = [&](const server_tokens & toks) {
                                    const int i0 = std::max(0, n_past - 16);
                                    const int i1 = std::min<int>(n_past + 32, toks.size());
                                    std::string s;
                                    for (int i = i0; i < i1; i++) {
                                        if (i == n_past) {
                                            s += "<<|>>";
                                        }
                                        const llama_token t = toks[i];
                                        s += t != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, t) : "[mtmd]";
                                    }
                                    std::string out;
                                    for (const char c : s) {
                                        out += c == '\n' ? std::string("\\n") : std::string(1, c);
                                    }
                                    return out;
                                };
                                SLT_INF(slot, "prompt diverges from the cache at token %d (cached %d, prompt %d): old '%s' new '%s'\n",
                                        n_past, (int) slot.prompt.tokens.size(), slot.task->n_tokens(),
                                        around(slot.prompt.tokens).c_str(), around(slot.task->tokens).c_str());
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // number of cached tokens shared with the new prompt, before any checkpoint is restored
                            const int n_prefix = n_past;

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    const auto usable = [&](const auto & cur) {
                                        // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                        SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                        // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                        if (cur.pos_max > pos_next) {
                                            return false;
                                        }
                                        return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                    };

                                    // search for a context checkpoint
                                    auto it = std::find_if(slot.prompt.checkpoints.rbegin(), slot.prompt.checkpoints.rend(), usable);

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    // [TAG_DEEP_CKPT] none of the in-RAM checkpoints (all near the old end) reaches back to the
                                    //   divergence: rewind to the deepest stored checkpoint of this prefix instead of re-processing
                                    //   everything, and take a checkpoint at the branch point on the way back up
                                    //   (also when the usable in-RAM one lies far below the divergence: an old pinned branch point)
                                    const int64_t n_ram = do_reset ? -1 : it->n_tokens;
                                    const common_prompt_checkpoint * disk_ck = nullptr; // the deep checkpoint taken from disk
                                    std::string disk_ck_key;
                                    if (disk_cache && n_past - n_ram > params_base.checkpoint_min_step) {
                                        const int n_div = n_past;
                                        for (const auto & m : disk_cache->deep_candidates(slot.task->tokens, n_past)) {
                                            if (m.n_tokens <= n_ram) {
                                                break; // deepest first: nothing better than the in-RAM one is left
                                            }
                                            if (m.pos_max > pos_next || !(m.pos_min < pos_min_thold || m.pos_min == 0)) {
                                                continue;
                                            }
                                            common_prompt_checkpoint ck;
                                            if (!disk_cache->deep_get(m.key, ck)) {
                                                continue;
                                            }
                                            const int64_t n_ck = ck.n_tokens;
                                            auto pos_it = std::find_if(slot.prompt.checkpoints.begin(), slot.prompt.checkpoints.end(),
                                                    [&](const auto & c) { return c.n_tokens > n_ck; });
                                            disk_ck_key = m.key;
                                            disk_ck = &*slot.prompt.checkpoints.insert(pos_it, std::move(ck));
                                            disk_cache->deep_adopt(slot.prompt, m);
                                            disk_cache->n_deep_rewinds++;
                                            SLT_INF(slot, "deep checkpoint at n_tokens = %" PRId64 " for a divergence at %d (replay %" PRId64 " tokens instead of %d)\n",
                                                    n_ck, n_div, (int64_t) n_div - n_ck, n_div);

                                            for (const auto & sp : slot.task->params.message_spans.spans) {
                                                if (sp.role == COMMON_CHAT_ROLE_USER && (int64_t) sp.pos > n_ck + params_base.checkpoint_min_step / 4 &&
                                                    (int64_t) sp.pos <= n_div) {
                                                    slot.ckpt_branch_pos = (int64_t) sp.pos;
                                                }
                                            }
                                            break;
                                        }
                                        it = std::find_if(slot.prompt.checkpoints.rbegin(), slot.prompt.checkpoints.rend(), usable);
                                        do_reset = it == slot.prompt.checkpoints.rend();
                                    }

                                    bool restored = false;
                                    // [TAG_CELL_EXT_PROVENANCE] a pre-fix (unmarked) checkpoint loads only with the disk cache's
                                    //   proof: an entry referencing it whose stored token stream has no media chunk in its prefix
                                    //   (orphans and checkpoints of this process's own making get none; the latter are marked)
                                    bool ck_legacy = false;
                                    if (!do_reset && disk_cache) {
                                        try {
                                            ck_legacy = disk_cache->ckpt_legacy_text_ok(
                                                    slot.prompt.tokens.prefix_keys({ (size_t) it->n_tokens }).at(0), it->n_tokens);
                                        } catch (const std::exception &) {
                                            ck_legacy = false;
                                        }
                                    }
                                    // scoped: cleared on every exit, a bad_alloc in a state reader included
                                    std::optional<llama_state_legacy_accept_scope> ck_scope;
                                    ck_scope.emplace(ctx_tgt, ctx_dft, ck_legacy);
                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        restored = it->load_tgt(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                                    }
                                    if (!do_reset && restored) {
                                        if (ctx_dft && ckpt_skip_dft && it->data_dft.empty()) {
                                            // [TAG_CKPT_NO_DFT] every surviving checkpoint is a prefix of the live tokens (the
                                            //   pos_max > pos_next ones are erased below), so the live draft KV already holds
                                            //   positions <= pos_max as the checkpoint saw them - drop what lies past it
                                            if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, it->pos_max + 1, -1)) {
                                                // cannot happen for a PART cache; if it does, drop the draft sequence (drafts
                                                // degrade until it refills, the target is unaffected) instead of aborting
                                                SLT_WRN(slot, "draft KV rewind to pos %d refused - resetting the draft sequence\n", it->pos_max);
                                                llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                                            }
                                        } else if (ctx_dft && it->data_dft.empty()) {
                                            // [TAG_DRAFT_PARTIAL] a draft-less checkpoint on a context whose checkpoints
                                            //   carry the draft (a target-only one from an adopt source): the draft
                                            //   sequence restarts empty, the prompt becomes draft-partial up to it
                                            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                                            slot.prompt.draft_gap = std::max<int32_t>(slot.prompt.draft_gap, (int32_t) it->n_tokens);
                                        } else {
                                            restored = it->load_dft(ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
                                        }
                                    }
                                    ck_scope.reset();
                                    if (!do_reset && restored && !common_speculative_set_state(spec.get(), slot.id, it->data_spec)) {
                                        // the draft's speculative state did not restore (Codex gate r2 Major 1): the same miss
                                        common_speculative_reset_state(spec.get(), slot.id);
                                        restored = false;
                                    }
                                    if (!do_reset && !restored) {
                                        // [TAG_CKPT_LOAD_CHECKED] (Codex gate r2 Blocker 1) the stored state's length differs from
                                        //   what the context consumed: a cache miss, not an abort - clear the partially restored
                                        //   destination in both contexts, drop that checkpoint (from disk too when it came from
                                        //   there) and re-process the prompt
                                        const bool from_disk = disk_ck != nullptr && &*it == disk_ck;
                                        if (disk_cache && !from_disk && it->id_task == SLOT_FILE_CKPT_ID_TASK) {
                                            // [TAG_SLOT_FILE_CKPT] (review m1) its bytes came from a /slots file: a stored deep
                                            //   checkpoint under the same prefix key is another, valid state - keep it
                                            disk_ck_key.clear();
                                        } else if (disk_cache && !from_disk) {
                                            // an in-RAM checkpoint that came with a loaded entry lives in the checkpoint
                                            //   store under its prefix key: drop it there too
                                            try {
                                                disk_ck_key = slot.prompt.tokens.prefix_keys({ (size_t) it->n_tokens }).at(0);
                                            } catch (const std::exception &) {
                                                disk_ck_key.clear();
                                            }
                                        }
                                        SLT_WRN(slot, "context checkpoint (n_tokens = %" PRId64 ", key %s) does not restore - dropping it, full prompt re-processing\n",
                                                it->n_tokens, disk_ck_key.empty() ? "-" : disk_ck_key.c_str());
                                        if (disk_cache && !disk_ck_key.empty()) {
                                            disk_cache->deep_drop(disk_ck_key, "state length mismatch on restore");
                                            auto & dr = slot.prompt.deep;
                                            dr.erase(std::remove_if(dr.begin(), dr.end(), [&](const auto & r) { return r.key == disk_ck_key; }), dr.end());
                                        }
                                        slot.prompt.checkpoints.erase(std::next(it).base());
                                        llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, -1, -1);
                                        if (ctx_dft) {
                                            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, -1, -1);
                                        }
                                        do_reset = true;
                                    }
                                    if (!do_reset) {
                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        slot.b0_ckpt_n_tokens = it->n_tokens;
                                        slot.ckpt_pinned_n_tokens = it->n_tokens; // [TAG_CKPT_PIN_RESTORED]
                                        SLT_INF(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB: tgt %.3f + dft %.3f + spec %.3f)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024,
                                                (float) it->data_tgt.size() / 1024 / 1024, (float) it->data_dft.size() / 1024 / 1024, (float) it->data_spec.size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        slot.b0_full_reset = true;
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next, or that cover tokens not shared with the new prompt
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next || cur.n_tokens > n_prefix) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);
                        slot.prompt.draft_trim(n_past); // [TAG_DRAFT_PARTIAL] the draft rows [gap, n_past) stay
                        if (n_past == 0) {
                            // [TAG_FULL_RESET] (Codex gate r1 C1) nothing of the old prompt survives a full recompute:
                            //   the trim above keeps checkpoints with pos_max <= 0 (e.g. the n_tokens = 1 checkpoint
                            //   of a one-token legacy restore), which would otherwise be published under the new
                            //   prompt once the flags below no longer block the disk save
                            slot.prompt.checkpoints.clear();
                            slot.prompt.deep.clear(); // [TAG_DEEP_CKPT]
                            slot.ckpt_pinned_n_tokens = -1;
                            slot.ckpt_branch_pos      = -1;
                            slot.prompt.noncanonical = false; // [TAG_NONCANONICAL] a full recompute
                            slot.prompt.partial      = false; // [TAG_PARTIAL_PROMPT] the draft side is rebuilt too
                            slot.prompt.draft_gap    = 0;     // [TAG_DRAFT_PARTIAL]
                            slot.prompt.adopted_from.clear(); // [TAG_DISK_ADOPT] recomputed, not the adopted state
                        }

                        // [TAG_TURN_BOUNDARY] the user-message starts of the prompt this slot is about to hold: the
                        //   task's own spans describe all of it; without spans (raw /completion) the known starts of
                        //   the kept prefix stay
                        {
                            auto & us = slot.prompt.user_starts;
                            std::vector<int32_t> task_us;
                            for (const auto & sp : slot.task->params.message_spans.spans) {
                                if (sp.role == COMMON_CHAT_ROLE_USER) {
                                    task_us.push_back((int32_t) sp.pos);
                                }
                            }
                            if (!task_us.empty()) {
                                std::sort(task_us.begin(), task_us.end());
                                us = std::move(task_us);
                            } else {
                                us.erase(std::remove_if(us.begin(), us.end(), [&](int32_t p) { return p >= n_past; }), us.end());
                            }
                        }

                        // [TAG_DEEP_CKPT] stored checkpoints past the kept prefix belong to the old continuation
                        slot.prompt.deep.erase(std::remove_if(slot.prompt.deep.begin(), slot.prompt.deep.end(),
                                    [&](const auto & r) { return r.n_tokens > n_past; }), slot.prompt.deep.end());

                        cache_pin_setup(slot, n_past); // [TAG_CACHE_PIN]

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // [TAG_GTT_ADMISSION] keep the prompt queued while its next ubatch would push GTT past the bar
                    if (adm.enabled) {
                        const auto adm_res = admission_check(slot, n_batch - batch.size());
                        if (adm_res == ADM_DEFER) {
                            return;
                        }
                        if (adm_res == ADM_REJECT) {
                            send_error(slot,
                                    string_format("GTT headroom: prompt deferred for more than %.0f s (projected %.0f MiB > bar %.0f MiB), retry later",
                                        adm.max_wait_s, slot.adm_projected_mib, adm.bar_mib),
                                    ERROR_TYPE_UNAVAILABLE);
                            slot.release();
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    // [TAG_RECURRENT_ROLLBACK_SPLITS]
                    // a bounded (recurrent) context only keeps the snapshot planes that the last ubatch of this
                    //   sequence wrote, so a deeper trim is refused - the only safe fallback is to drop the
                    //   sequence and reprocess the whole prompt
                    if (!slot.mem.seq_rm_try(slot.id, p0, -1)) {
                        SLT_WRN(slot, "partial memory removal refused at pos %d - reprocessing the whole prompt\n", p0);

                        slot.mem.seq_rm(slot.id, -1, -1);

                        slot.prompt.tokens.keep_first(0);
                        slot.prompt.checkpoints.clear();
                        slot.prompt.deep.clear(); // [TAG_DEEP_CKPT]
                        slot.ckpt_pinned_n_tokens = -1; // [TAG_FULL_RESET]
                        slot.ckpt_branch_pos      = -1;
                        slot.prompt.noncanonical = false; // [TAG_NONCANONICAL]
                        slot.prompt.partial      = false; // [TAG_PARTIAL_PROMPT]
                        slot.prompt.draft_gap    = 0;     // [TAG_DRAFT_PARTIAL]
                        slot.prompt.adopted_from.clear(); // [TAG_DISK_ADOPT]
                        cache_pin_setup(slot, 0); // [TAG_CACHE_PIN] (review m1) re-arm: the prefix is re-processed

                        slot.stats.n_prompt_cached = 0;

                        slot.b0_full_reset = true;
                    }

                    // [TAG_B0_RESTORE_LOG] (v33 B0) one line per prompt-processing task: how much of the common prefix
                    //   is re-processed. replayed = prefix_match - n_past_restored (tokens that matched the live state
                    //   but are decoded again because the recurrent state could only be restored at an earlier point),
                    //   ~ n_ubatch when the n - (4 + n_ubatch) checkpoint was used, ~ 4 for n - 4.
                    if (slot.b0_pending) {
                        slot.b0_pending = false;

                        const int32_t n_restored = slot.prompt.n_tokens();
                        const int32_t replayed   = std::max(0, slot.b0_prefix_match - n_restored);
                        const int32_t n_new      = std::max(0, slot.task->n_tokens() - std::max(slot.b0_prefix_match, n_restored));

                        metrics.n_prompt_restores       += slot.b0_ckpt_n_tokens >= 0 ? 1 : 0;
                        metrics.n_prompt_replayed_tokens += replayed;

                        const std::string ckpt_str = slot.b0_ckpt_n_tokens >= 0
                            ? string_format("checkpoint at %" PRId64, slot.b0_ckpt_n_tokens)
                            : slot.b0_full_reset ? std::string("no checkpoint, full reset") : std::string("live state, no checkpoint");

                        SLT_INF(slot, "restore: slot %d n_past_restored=%d prefix_match=%d replayed=%d tokens (%s; cached_live=%d, new=%d, prompt=%d)\n",
                                slot.id, n_restored, slot.b0_prefix_match, replayed, ckpt_str.c_str(),
                                slot.b0_n_cached_live, n_new, slot.task->n_tokens());
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            invalidate_slot(slot); // [TAG_SLOT_INVALIDATE] part of the chunk may be in the sequence
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        slot.prompt.tokens.push_back(cur_tok);

                        // [TAG_CACHE_PIN] the pinned prefix ends here: split, the next batch starts with its checkpoint
                        if (do_checkpoint && slot.prompt.n_tokens() == slot.ckpt_pin_pos) {
                            break;
                        }

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step ||
                                pos == slot.ckpt_branch_pos) {
                                break;
                            }
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    if (adm.enabled) {
                        adm.hw_cells = std::max(adm.hw_cells, admission_resident_cells());
                    }

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end && n_tokens_start != slot.ckpt_pin_pos) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start == slot.ckpt_branch_pos || n_tokens_start == slot.ckpt_pin_pos ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        if (spec && batch.has_embd) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;

        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // active recurrent-state rollback depth of this decode: a verification round can reject at most
        //   spec_draft.size() - spec_n_committed of its tokens, prompt and plain rounds never roll back and
        //   have an empty spec_draft -> depth 0. One decode can carry several slots, so the depth passed to
        //   llama_decode() is the maximum over the slots in this view; the memory still records the number of
        //   planes each sequence actually got, so a seq_rm deeper than that is refused per sequence.
        uint32_t rs_depth = 0;

        const uint32_t rs_depth_max = llama_n_rs_seq(ctx_tgt);

        for (int i = off; i < off + batch_view.n_tokens; ++i) {
            has_output |= batch.tokens[i].output;

            const auto & slot = slots[batch.tokens[i].id_slot];
            GGML_ASSERT(slot.spec_n_committed <= slot.spec_draft.size());

            rs_depth = std::max(rs_depth, (uint32_t) std::min<size_t>(slot.spec_draft.size() - slot.spec_n_committed, rs_depth_max));
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        const int64_t t_verify0 = ggml_time_us(); // [TAG_SPEC_ROUND_TIMING]
        queue_tasks.yield_to_queue([&]() {
            llama_set_rs_rollback_depth(ctx_tgt, rs_depth);
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });
        spec_rt.verify_us += ggml_time_us() - t_verify0;
        spec_rt.n_rows    += batch_view.n_tokens; // [TAG_SPEC_DRAFT_POLICY]

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch_view.n_tokens, has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            bool ok = true;
            const int64_t t_catchup0 = ggml_time_us(); // [TAG_SPEC_ROUND_TIMING]
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch_view);
            });
            spec_rt.catchup_us += ggml_time_us() - t_catchup0;

            if (!ok) {
                SRV_ERR("%s", "failed to process speculative batch\n");

                // TODO: handle error
                throw std::runtime_error("failed to process speculative batch");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            if (dbg_fault == 1 && slot.is_processing() && slot.stats.n_gen >= 2) {
                dbg_fault = 0;
                throw std::runtime_error("debug fault: slot");
            }
            // [TAG_MULTISLOT] (item 4a) a slot whose verification rows are all outside this view is not part of it
            //   (another routing range); a participating slot must have ALL its rows here (never split)
            size_t n_inside = 0;
            for (auto & i : slot.spec_i_batch) {
                n_inside += is_inside_view(i) ? 1 : 0;
            }
            if (n_inside > 0 && n_inside < slot.spec_i_batch.size()) {
                throw std::runtime_error(string_format("speculative batch indices of slot %d are split across sub-batches (%zu of %zu inside [%d, %d))",
                        slot.id, n_inside, slot.spec_i_batch.size(), off, off + n_batch_tokens));
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // [TAG_MULTISLOT] (item 4a) only the slots verified in this view (all rows inside, checked above)
            if (!is_inside_view(slot.spec_i_batch.front())) {
                return;
            }

            // the samplers read the logits of the last decode = this view: view-local indices (global - off); the
            // stored indices stay global until here (identical to before when the view starts at 0)
            std::vector<int32_t> spec_i_view(slot.spec_i_batch);
            for (auto & i : spec_i_view) {
                i -= off;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            {
                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                // [TAG_UP27694] drafters that fill no distribution (or a draft whose candidates do not line up with it)
                //   fall back to sample-and-match; a replay re-emits its committed tokens in both modes (the fork's
                //   server_sample_and_accept_replay, which never re-samples them - upstream's server_accept_replay is
                //   not needed here)
                const bool use_rejection = params_base.speculative.draft.probabilistic && slot.use_spec_rejection() &&
                                           !slot.spec_is_replay &&
                                           !slot.spec_draft_q.empty() && slot.spec_draft_q.size() == slot.spec_draft.size();
                if (!slot.spec_is_replay) {
                    // [TAG_UP27694] a replay re-emits the tokens committed by this round: remember how they were verified
                    slot.spec_round_rejection = use_rejection && synth_probs.empty();
                }

                std::vector<llama_token> accepted;
                if (!synth_probs.empty()) {
                    accepted = server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, spec_i_view, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_n_committed);
                } else if (slot.spec_is_replay) {
                    int n_mismatch = 0;
                    accepted = server_sample_and_accept_replay(slot.smpl.get(), slot.ctx_tgt, spec_i_view, slot.spec_draft,
                            slot.spec_n_committed,
                            // [TAG_UP27694] a token kept by rejection sampling is not the target's own sample, so
                            //   the (trace-only) re-sample probe would count it as a flip: off for those rounds
                            trace > 0 && !slot.spec_round_rejection,
                            n_mismatch);
                    if (n_mismatch > 0) {
                        // would have been rejected again by the old code -> checkpoint restore -> replay -> ... (livelock)
                        SLT_WRN(slot, "replay: %d of %zu committed tokens re-sampled differently (batch-size dependent rounding or an inexact restore); kept the committed tokens\n",
                                n_mismatch, slot.spec_draft.size());
                        slot.stats.n_replay_mismatch += n_mismatch;
                    }
                } else if (use_rejection) {
                    // [TAG_UP27694] probabilistic draft: verify by rejection sampling against the draft's candidates
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, spec_i_view, slot.spec_draft, slot.spec_draft_q);
                } else {
                    accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, spec_i_view, slot.spec_draft);
                }
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                // [TAG_SPEC_TRACE_IDS] LLAMA_SPEC_TRACE_IDS=1 (debug, default off): one machine-readable line per verify
                // round - the proposals (the drafted ids in order, committed replay tokens first), the verify width and the
                // accepted ids - for an external comparer (proposal identity across two runs, R83 stats)
                if (server_spec_trace_ids()) {
                    std::string sd, sa;
                    for (size_t i = 0; i < slot.spec_draft.size(); ++i) {
                        sd += (i ? "," : "") + std::to_string(slot.spec_draft[i]);
                    }
                    for (size_t i = 0; i < accepted.size(); ++i) {
                        sa += (i ? "," : "") + std::to_string(accepted[i]);
                    }
                    // t_*_us: this pass's host wall times of the draft / catch-up / verify phases (spec_rt, all slots of the pass);
                    // n_att: the draft positions the drafter sampled this round (incl. a p_min stop; -1 = not reported) = the
                    // number of draft-step graphs, the x of the per-step cost fit (spec_cost_attrib.py)
                    const int n_att = common_speculative_get_draft_params(spec.get(), slot.id).n_attempted;
                    fprintf(stderr, "[spec-ids] task=%d slot=%d round=%llu n_past=%d n_draft=%zu committed=%zu replay=%d synth=%d width=%zu n_acc=%zu draft=%s acc=%s"
                            " t_draft_us=%lld t_catchup_us=%lld t_verify_us=%lld n_att=%d\n",
                            slot.task ? slot.task->id : -1, slot.id, (unsigned long long) slot.spec_trace_round++,
                            (int) slot.prompt.n_tokens(), n_draft, std::min(slot.spec_n_committed, n_draft),
                            (int) slot.spec_is_replay, (int) !synth_probs.empty(), n_draft + 1, accepted.size() - 1,
                            sd.empty() ? "-" : sd.c_str(), sa.c_str(),
                            (long long) spec_rt.draft_us, (long long) spec_rt.catchup_us, (long long) spec_rt.verify_us, n_att);
                }

                // adaptive draft length: record this round's per-position acceptance here, before
                // the branch, so it happens exactly once per verified round on every exit.
                //
                // Only the tokens that were freshly drafted this round carry information about the
                // draft model.  slot.spec_n_committed leading tokens of spec_draft (nonzero on a
                // replay or a fold round, see 0006-0010) were sampled and accepted during an
                // earlier round and are merely re-emitted here, so they are not observations - and
                // on a pure replay round there is no fresh tail at all, which makes n_fresh_draft
                // zero and spec_acc_update a no-op.
                //
                // Doing this before the branch is the fix: the checkpoint-restore branch below
                // returns early, and the round it schedules next is a replay, which used to be
                // excluded wholesale by "!was_replay".  A FULL-rollback target (or an RS target
                // whose rollback is deeper than n_rs_seq) therefore fed the estimator every
                // success and no failure, and the EMA converged to 1.0 - i.e. always draft n-max -
                // no matter how badly the draft model was doing.  A fold round's fresh tail was
                // dropped for the same reason.
                // this round's fresh tail, shared by the adaptive / tput estimator and the H2 auto depth below
                const size_t n_committed_in = std::min(slot.spec_n_committed, n_draft);
                const size_t n_fresh_draft  = n_draft - n_committed_in;
                const size_t n_matched      = accepted.size() - 1;
                const size_t n_fresh_acc    = n_matched > n_committed_in
                                            ? std::min(n_matched - n_committed_in, n_fresh_draft)
                                            : 0;

                // [TAG_B68_DRAFT_KNOBS] H3 diagnostic: the fresh draft held an EOG that the verifier did not take (it stopped
                //   before it, or sampled another token there), so the round was not the last one. A matched EOG ends the
                //   accepted list as the target's own token (accepted[k] == draft[k], n_matched == k): terminal, not counted
                for (size_t k = n_committed_in; k < n_draft; ++k) {
                    if (llama_vocab_is_eog(vocab, slot.spec_draft[k])) {
                        if (n_matched < k || (n_matched == k && accepted[k] != slot.spec_draft[k])) {
                            slot.stats.n_draft_eog_rejected++;
                        }
                        break;
                    }
                }

                if (params_base.speculative.draft.n_max_adaptive ||
                        params_base.speculative.draft.policy == COMMON_SPEC_DRAFT_POLICY_TPUT) {
                    if (params_base.speculative.draft.policy == COMMON_SPEC_DRAFT_POLICY_TPUT) {
                        // [TAG_SPEC_DRAFT_POLICY] a drafter that stopped short (p_min exit) failed at the next position
                        const bool short_draft = n_committed_in == 0 && n_fresh_draft > 0 &&
                                                 slot.spec_tput_n_att > (int) n_fresh_draft;
                        slot.spec_acc_update(n_fresh_draft + (short_draft ? 1 : 0), n_fresh_acc, common_speculative_n_max(spec.get()));
                        if (n_fresh_draft > 0 && slot.spec_tput_ranked) {
                            slot.spec_cd.observe((int) n_fresh_draft, (int) n_fresh_acc);
                        }
                    } else {
                    slot.spec_acc_update(n_fresh_draft, n_fresh_acc, common_speculative_n_max(spec.get()));
                    }
                }

                // [TAG_B68_DRAFT_KNOBS] H2: adapt the slot's depth from this round's fresh tail
                if (slot.spec_depth_req > 0) {
                    if (n_fresh_draft > 0) {
                        const int depth_prev = slot.spec_depth_cur;
                        if (n_fresh_acc == n_fresh_draft) {
                            if ((int) n_fresh_draft >= slot.spec_depth_req) {
                                // clamped to n-max at the next round; a greedy-capped req never lowers the depth
                                slot.spec_depth_cur = std::max(slot.spec_depth_cur, slot.spec_depth_req) + 1;
                            }
                        } else {
                            slot.spec_depth_cur = std::max(server_spec_depth_min(), slot.spec_depth_req - 1);
                        }
                        if (trace > 0 && slot.spec_depth_cur != depth_prev) {
                            SLT_INF(slot, "auto draft depth: %d -> %d (allowed %d, fresh accepted %zu/%zu)\n",
                                    depth_prev, slot.spec_depth_cur, slot.spec_depth_req, n_fresh_acc, n_fresh_draft);
                        }
                    }
                    slot.spec_depth_req = 0;
                }

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_n_committed = accepted.size();
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        const int64_t t_restore0 = ggml_time_us(); // [TAG_SPEC_DRAFT_POLICY] the rollback cycle

                        // this run's own speculative checkpoint: a size mismatch is a bug, not a cache miss
                        if (!ckpt.load_tgt(slot.ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
                            GGML_ABORT("speculative checkpoint size mismatch\n");
                        }

                        if (slot.ctx_dft) {
                            if (!ckpt.load_dft(slot.ctx_dft, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
                                GGML_ABORT("speculative draft checkpoint size mismatch\n");
                            }
                        }

                        slot.mem.seq_rm(slot.id, ckpt.pos_max + 1, -1);

                        // [TAG_MTP_CKPT_PENDING_H] (audit B#2b) the carrier back to the checkpoint's: without it the
                        //   replay paired its first token with the last row of the rejected verification batch
                        common_speculative_set_state(spec.get(), slot.id, ckpt.data_spec);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);

                        if (params_base.speculative.draft.policy == COMMON_SPEC_DRAFT_POLICY_TPUT) {
                            slot.spec_rb_restore_us = ggml_time_us() - t_restore0;
                            spec_rt.restore_us     += slot.spec_rb_restore_us;
                            // every restore is measured (multi-slot passes too): the rollback fallback's restore term
                            spec_tput_fits.at(spec_n_kv_depth(slot)).restore.add(0.0, slot.spec_rb_restore_us / 1e3);
                        }

                        // the sampler is deliberately NOT rolled back: the tokens accepted just above are the
                        // ones the replay round will emit, so the live (post-verification) sampler already has
                        // the state that belongs after them - see server_sample_and_accept_replay()

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;
            slot.spec_n_committed = 0;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // (the adaptive draft length's acceptance EMA was updated in the verification block
            // above, before the checkpoint-restore branch, so that a restore round is observed too)

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());
        });
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
    int n_ctx_slot() const {
        int res = llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    // [TAG_SPEC_ADAPTIVE_DEPTH] KV depth the next decode of this slot will attend over - the cost
    // driver for the adaptive draft length. With a unified KV cache the attention walks the whole used
    // span of the cache, so the depth is the max over the slots that are running; without it every
    // sequence has its own cells and only this slot's depth matters.
    int spec_n_kv_depth(const server_slot & slot) const {
        return params_base.kv_unified ? std::max((int) slot.prompt.n_tokens(), spec_n_kv_busy()) : slot.prompt.n_tokens();
    }

    // deepest sequence currently running - the depth a shared (unified) decode pays for
    int spec_n_kv_busy() const {
        int n_kv = 0;

        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_kv = std::max(n_kv, (int) slot.prompt.n_tokens());
            }
        }

        return n_kv;
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto & slot = slots[t.id_slot];
            if (slot.stats.is_set()) {
                slot.stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();

            // [TAG_SPEC_ADAPTIVE_DEPTH] one (batch size, wall time) sample for the measured cost model.
            // Pure-generation batches only: a prefill ubatch would swamp the fit.
            if (n_prompt_tokens == 0 &&
                    params_base.speculative.draft.n_max_adaptive &&
                    params_base.speculative.draft.n_max_adaptive_measure) {
                spec_cost.add(n_tokens, (ggml_time_us() - t_decode_start) / 1000.0, spec_n_kv_busy());
            }
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto & slot = slots[t.id_slot];
            if (t.is_prompt && slot.stats.is_set()) {
                slot.stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        metrics.n_spec_verify_rounds += slot.stats.n_spec_verify_rounds;
        metrics.n_spec_replay_rounds += slot.stats.n_spec_replay_rounds;
        metrics.n_spec_fold_rounds   += slot.stats.n_spec_fold_rounds;
        metrics.n_spec_plain_rounds  += slot.stats.n_spec_plain_rounds;
        metrics.n_spec_empty_drafts  += slot.stats.n_spec_empty_drafts;
        metrics.n_spec_fold_rejected += slot.stats.n_spec_fold_rejected;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

void server_context::save_on_exit() {
    impl->save_on_exit();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->n_ctx_slot(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,

        /* backend_sampling_supported */ impl->bs_support.supported,
        /* backend_sampling_error     */ impl->bs_support.error,
        /* spec_type                  */ impl->bs_support.spec_type,
        /* spec_n_max                 */ impl->bs_support.n_max,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            // [TAG_BS_GUARD] reject here (400, nothing posted, no slot touched) instead of letting llama_decode fail
            //   the verify batch (500 + the slot's prompt cache cleared)
            if (task.params.sampling.backend_sampling && !meta->backend_sampling_supported) {
                throw std::invalid_argument(meta->backend_sampling_error);
            }

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
        // [TAG_BS_GUARD] whether a request may set backend_sampling: true, and the loaded speculation
        { "backend_sampling_supported",  meta.backend_sampling_supported },
        { "speculative",                 json {
            {"type",  meta.spec_type},
            {"n_max", meta.spec_n_max},
        } },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    // [TAG_DISK_PROMPT_CACHE] stored conversations + totals (a snapshot, safe from the HTTP threads)
    this->get_disk_cache = [this](const server_http_req &) {
        auto res = create_response(true);
        res->ok(ctx_server.disk_cache_status());
        return res;
    };

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i], ctx_server.init_opt);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}
