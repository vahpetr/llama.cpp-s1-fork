#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "json.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <set>

struct server_context_impl; // private implementation

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    bool has_inp_video;
    json json_ui_settings;
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
    std::string model_ftype;

    // [TAG_BS_GUARD] backend sampling support + loaded speculation state (/props, request validation)
    bool        backend_sampling_supported;
    std::string backend_sampling_error;
    std::string spec_type;
    int32_t     spec_n_max; // effective drafter ceiling (as sized for the target's output limit)
};

enum server_state {
    SERVER_STATE_DOWNLOADING,
    SERVER_STATE_LOADING,
    SERVER_STATE_READY,
    SERVER_STATE_SLEEPING,
};

static std::string server_state_to_str(server_state state) {
    switch (state) {
        case SERVER_STATE_DOWNLOADING: return "downloading";
        case SERVER_STATE_LOADING:     return "loading";
        case SERVER_STATE_READY:       return "ready";
        case SERVER_STATE_SLEEPING:    return "sleeping";
        default: GGML_ASSERT(false && "invalid server_state");
    }
}

static server_state server_state_from_str(const std::string & str) {
    if (str == "downloading") return SERVER_STATE_DOWNLOADING;
    if (str == "loading")     return SERVER_STATE_LOADING;
    if (str == "ready")       return SERVER_STATE_READY;
    if (str == "sleeping")    return SERVER_STATE_SLEEPING;
    GGML_ASSERT(false && "invalid server_state string");
}

using server_state_callback_t = std::function<void(server_state, json /* payload */)>;

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // after start_loop returned: write the slots to the disk prompt cache (no-op without --cache-disk)
    void save_on_exit();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // note: must be set before load_model() is called
    void set_state_callback(server_state_callback_t callback);
};


// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_disk_cache;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_chat_completions_tok;
    server_http_context::handler_t post_control;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_responses_tok_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;

    // to be used in router mode
    json get_model_info() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);

    // cached responses, to be used during sleep
    std::mutex     mutex_cache;
    json           cached_models  = nullptr;
    json           cached_props   = nullptr;
    server_metrics cached_metrics;
    // set when a scrape during sleep already reported the throughput buckets
    bool           should_reset_buckets = false;
    // call right before sleep to update the cached responses
    void update_cached_responses(bool is_sleeping);
};

// replay round after a checkpoint restore: re-emit the n_committed leading draft tokens, verify the rest
// (defined in server-context.cpp; declared here for tests/test-draft-accept-eog)
std::vector<llama_token> server_sample_and_accept_replay(
        struct common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int> & idxs,
        const llama_tokens & draft,
        size_t n_committed,
        bool probe_mismatch,
        int & n_mismatch);

// [TAG_MULTISLOT] stage 5.1 item 2: the multi-slot support envelope. LLAMA_QSA_HOMOGENEOUS=1 with n_parallel > 1 is
// the qualification mode: text-only (no mmproj), no shared-prefix parent/child requests, no context shift - startup is
// refused outside it. Unset / 0 = today's behaviour (the hands' 5 slots + mmproj keep running), a WARN only.
// (defined in server-context.cpp; declared here for tests/test-cache-compat-key)
struct server_multislot_envelope {
    bool        strict = false; // the knob is 1 AND n_parallel > 1: shared-prompt (n_cmpl > 1) requests are rejected
    bool        ok     = true;  // false: refuse startup (error says why)
    std::string error;
    std::string proof;          // the [multislot-envelope] line ("" for n_parallel <= 1: unchanged, nothing printed)
};
server_multislot_envelope server_multislot_envelope_check(const char * knob, int n_parallel, bool has_mmproj, bool ctx_shift);

// [TAG_BS_GUARD] 0226: can this server honour a request's `backend_sampling: true`? Decided once from the loaded
//   configuration. llama_decode() rejects a batch in which a sequence with a backend sampler has more outputs than the
//   TARGET context's n_outputs_max_per_seq; a speculative verify round asks for 1 + n_draft outputs per sequence. The
//   target limit is sized by server_output_limits (min(n_batch, 1 + n_max)), so speculation is supported whenever that
//   limit covers 1 + the drafter's ceiling (the draft context's own limits are separate and not looked at). Tensor split
//   never offloads samplers. Unsupported -> requests asking for it get a 400 (error) before any slot is touched, and a
//   server started with --backend-sampling has it forced off at startup (warning).
//   (defined in server-context.cpp; tests/test-backend-sampling-guard)
struct server_backend_sampling_support {
    bool        supported   = true;
    std::string spec_type   = "none"; // loaded speculation: canonical --spec-type name(s), "none" without speculation
    int32_t     n_max       = 0;      // the drafter's effective ceiling used for the output limit (0 = none)
    int32_t     per_seq_max = 1;      // the target context's resolved n_outputs_max_per_seq
    std::string error;                // the 400 text when !supported
};
server_backend_sampling_support server_backend_sampling_check(const common_params & params);

// [TAG_MULTISLOT] (stage 5.1 item 4a) homogeneous routing-class planner (LLAMA_QSA_HOMOGENEOUS=1, np > 1).
// class of a depth interval [d_lo, d_hi] (depth = pos + 1) against N = LLAMA_QSA_MIN_KV: 1 above (d_lo >= N, equality
// is above), 0 below (d_hi < N), -1 straddles N (a singleton)
int server_qsa_route_class(int64_t d_lo, int64_t d_hi, int64_t n_min);

// one whole run (a slot's rows of one round, or a drafting slot with one row per draft step)
struct server_qsa_route_run {
    int32_t   id_slot;
    int32_t   n_rows;
    llama_pos p_lo;      // min row position
    llama_pos p_hi;      // max row position
    bool      partial;   // partial history: a singleton
    int32_t   first_row; // original order key (the first row's batch index)
};

// groups of run indices: singletons for straddling / partial runs; same-class runs together, sorted by slot id
// (split_equal wants consecutive increasing sequence ids), runs of <= max_ub rows chunked at max_ub TOTAL rows
// (LLAMA_MTP_QSA_MAX_UBATCH, 0 = no cap), larger runs only with larger runs; groups ordered by first_row
std::vector<std::vector<int32_t>> server_qsa_route_groups(const std::vector<server_qsa_route_run> & runs, int64_t n_min, int64_t max_ub);

// (r3, astra r2 Minor) the row cap of TARGET ranges: the MAX_UBATCH limit (max_ub) belongs to the MTP layers, so it
// applies to a target range only when that range's MTP catch-up runs attention (LLAMA_MTP_CATCHUP_ATTN unset / not
// "0": combining rows can change the catch-up's QSA topology) or LLAMA_QSA_HOMOGENEOUS_TARGET_CAP=1 (the r2
// conservative cap); with the attention-free catch-up (prod "0") target ranges are uncapped (0). Draft groups always
// keep max_ub. Arguments are the raw env values (nullptr = unset).
int64_t server_qsa_route_target_cap(int64_t max_ub, const char * env_target_cap, const char * env_catchup_attn);

// [TAG_MULTISLOT] (stage 5.1 item 5b, astra r2 Major 2) operation-start gate of LLAMA_GTT_CONCURRENT_RESERVE_MIB: before
// a slot starts draft / speculative-checkpoint work (device GTT) or a restore allocation (RAM swap-in, disk load,
// /slots restore) while ANOTHER slot is processing, the live GTT usage + the reserve must stay <= the admission bar;
// otherwise the operation is deferred (never aborted): a draft round becomes a plain decode, a restore waits in the
// task queue. Off (start = true, nothing read) with admission off, reserve 0 (the default) or no concurrency.
enum server_adm_op {
    SERVER_ADM_OP_DRAFT = 0,     // draft generation + its on-device speculative checkpoints (target / draft)
    SERVER_ADM_OP_RESTORE_RAM,   // RAM prompt-cache swap-in on slot selection
    SERVER_ADM_OP_RESTORE_DISK,  // disk prompt-cache load on slot selection
    SERVER_ADM_OP_RESTORE_SLOT,  // /slots/{id}?action=restore
    SERVER_ADM_OP_COUNT,
};
const char * server_adm_op_name(server_adm_op op);

struct server_adm_op_decision {
    bool   start         = true;  // false = defer
    bool   gated         = false; // the gate was evaluated (enabled, reserve > 0, concurrent, usage readable)
    double used_mib      = 0.0;
    double projected_mib = 0.0;   // used + reserve
};
server_adm_op_decision server_admission_op_check(bool enabled, double bar_mib, double reserve_mib, bool concurrent,
                                                 const std::string & used_path);
