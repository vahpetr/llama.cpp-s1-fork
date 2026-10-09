#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama-sampler.h"
#include "llama.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

//
// llama_context
//

// env: GGML_SCHED_PROFILE=1 - host-side timing of the decode path, printed to stderr as "[sched-prof]" lines
// (the same env drives ggml_backend_sched_compute_splits and ggml_backend_vk_graph_compute). Read once and
// cached: with the env off every instrumented site is a single branch on this bool and nothing else.
static bool sched_prof_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("GGML_SCHED_PROFILE");
        return s != nullptr && atoi(s) != 0;
    }();
    return enabled;
}

// n_kv of the attention cache the graph was built against; -1 when there is none (encoder, recurrent-only)
static int64_t sched_prof_n_kv(const llama_memory_context_i * mctx) {
    if (mctx == nullptr) {
        return -1;
    }
    if (const auto * c = dynamic_cast<const llama_kv_cache_context *>(mctx)) {
        return c->get_n_kv();
    }
    if (const auto * c = dynamic_cast<const llama_kv_cache_iswa_context *>(mctx)) {
        return c->get_base()->get_n_kv();
    }
    if (const auto * c = dynamic_cast<const llama_memory_hybrid_context *>(mctx)) {
        return c->get_attn()->get_n_kv();
    }
    if (const auto * c = dynamic_cast<const llama_memory_hybrid_iswa_context *>(mctx)) {
        return c->get_attn()->get_base()->get_n_kv();
    }
    return -1;
}

static const char * sched_prof_ctx_name(llama_context_type ctx_type) {
    return ctx_type == LLAMA_CONTEXT_TYPE_MTP ? "mtp" : "main";
}

// [TAG_KQVIS_PROF] LLAMA_KQ_VIS_PROF=1 (llama_kqvis_prof_enabled, llama-kv-cache.cpp): per context, windows of <= 64
// graphs of one class and one fill mode, one WARN line each:
//   kqvis-prof: ctx=<main|mtp> mode=<mask|vis|none|both> class=<dec|pp> win=<n> reason=<full|class|mode|idle|exit>
//     graphs reused tok nkv_min nkv_max nkv_avg build_us inputs_us fill_us fill_avg in_wait_us in_copy_us kq_copy_us
//     kq_kib compute_us compute_avg sync_us sync_avg n_sync t0_ms t1_ms
//     plan_us n_plan plan_avg plan_inc cen_inc cen_full fb=<mark>/<shape>/<shrink>/<region>
//     qin_us mir_kib mir_runs mir_full n_round t_draft_ms t_catchup_ms t_verify_ms t_host_ms
// [TAG_QSA_CENSUS_INC] plan_* (appended; all 0 without LLAMA_QSA_POOLED_CACHE): plan_us = the QSA store plan time of the window
// (census + stamps + slot plan, llama_qsa_pooled_store::stats.t_plan_us; it runs inside build_us), n_plan = plans,
// plan_avg = plan_us/n_plan, plan_inc = plans whose every stream took the incremental fast path, cen_inc / cen_full =
// stream censuses fast / full, fb = the full ones by reason: mark (a mutator's full-census mark or the first plan), shape
// (> 1 sequence, ranked / dup, oor), shrink (n_kv shrank), region (re-grouped region too large)
// [TAG_QSA_INPUTS_INC] then (appended) qin_us mir_kib mir_runs mir_full: qin_us = time in the QSA store inputs'
// set_input (inside inputs_us; either mode), mir_kib / mir_runs / mir_full = KiB and runs written to the device-resident
// input mirrors and the (array, stream) writes of a whole array (all 0 without LLAMA_QSA_INPUTS_INC=1)
// [TAG_SPEC_ROUND_TIMING] then (appended) n_round t_draft_ms t_catchup_ms t_verify_ms t_host_ms: the speculative rounds
// the server reported to this context (llama_kqvis_prof_add_spec_round, at the end of each round, so a round lands in the
// window open when it ended) and their summed host wall times: the drafter call (common_speculative_draft), the catch-up
// (common_speculative_process), the verify decode (llama_decode + its sync) and the rest of the round (host = round wall
// - the three); all 0 without a speculative server / in the MTP context
// per graph: build = process_ubatch() start .. set_inputs (apply, reuse check, build + alloc); inputs = all set_inputs;
// fill = the KQ mask / metadata fill inside it; compute = graph_compute() host time, which contains the scheduler's
// user-input uploads (in_wait = the synchronize before each upload, in_copy = the uploads, kq_copy/kq_kib = the
// attn_inp_kq_* part); sync = synchronize() waits with queued work (the device time not already waited in in_wait).
// t0/t1 = unix ms of the window, so a reader attributes windows to request phases by time; a window never spans an idle
// gap > 1.5 s of all contexts (g_kqvis_epoch), so a request's windows are closed by the next request's first graph.
static std::atomic<int64_t>  g_kqvis_last_us{0}; // ggml_time_us() at the end of the last graph / sync, any context
static std::atomic<uint64_t> g_kqvis_epoch{0};   // +1 at every idle gap > 1.5 s

static int64_t kqvis_unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void llama_context::kqvis_prof_add_spec_round(int64_t t_draft_us, int64_t t_catchup_us, int64_t t_verify_us,
        int64_t t_host_us) const {
    auto & w = kqvis_win; // [TAG_SPEC_ROUND_TIMING]
    w.n_round      += 1;
    w.t_draft_us   += t_draft_us;
    w.t_catchup_us += t_catchup_us;
    w.t_verify_us  += t_verify_us;
    w.t_host_us    += t_host_us;
}

void llama_context::kqvis_prof_add_catchup_pf(int64_t t_catchup_us) const {
    kqvis_win.t_catchup_pf_us += t_catchup_us; // [TAG_SPEC_ROUND_TIMING]
}

void llama_context::kqvis_prof_flush(const char * reason) const {
    const auto & w = kqvis_win;
    if (w.graphs > 0) {
        static const char * mode_name[4] = { "none", "mask", "vis", "both" };
        const double g = (double) w.graphs;
        LLAMA_LOG_WARN("kqvis-prof: ctx=%s mode=%s class=%s win=%" PRId64 " reason=%s graphs=%d reused=%d tok=%" PRId64
                " nkv_min=%" PRId64 " nkv_max=%" PRId64 " nkv_avg=%.0f build_us=%" PRId64 " inputs_us=%" PRId64
                " fill_us=%" PRId64 " fill_avg=%.1f in_wait_us=%" PRId64 " in_copy_us=%" PRId64 " kq_copy_us=%" PRId64
                " kq_kib=%" PRId64 " compute_us=%" PRId64 " compute_avg=%.1f sync_us=%" PRId64 " sync_avg=%.1f n_sync=%" PRId64
                " t0_ms=%" PRId64 " t1_ms=%" PRId64
                " plan_us=%" PRId64 " n_plan=%" PRId64 " plan_avg=%.1f plan_inc=%" PRId64 " cen_inc=%" PRId64
                " cen_full=%" PRId64 " fb=%" PRId64 "/%" PRId64 "/%" PRId64 "/%" PRId64
                " qin_us=%" PRId64 " mir_kib=%" PRId64 " mir_runs=%" PRId64 " mir_full=%" PRId64
                " n_round=%" PRId64 " t_draft_ms=%.3f t_catchup_ms=%.3f t_verify_ms=%.3f t_host_ms=%.3f t_catchup_pf_ms=%.3f\n",
                sched_prof_ctx_name(cparams.ctx_type), mode_name[w.mode & 3], w.cls == 1 ? "pp" : "dec", kqvis_n_win, reason,
                w.graphs, w.reused, w.tokens, w.n_kv_min, w.n_kv_max, w.n_kv_sum/g, w.build_us, w.inputs_us,
                w.fill_us, w.fill_us/g, w.in_wait_us, w.in_copy_us, w.kq_copy_us,
                w.kq_bytes/1024, w.compute_us, w.compute_us/g, w.sync_us, w.sync_us/g, w.n_sync,
                w.t0_ms, w.t1_ms,
                w.plan_us, w.n_plan, w.n_plan > 0 ? (double) w.plan_us/w.n_plan : 0.0, w.n_plan_inc, w.n_cen_inc,
                w.n_cen_full, w.n_cen_fb[1], w.n_cen_fb[2], w.n_cen_fb[3], w.n_cen_fb[4],
                w.qin_us, w.mir_bytes/1024, w.mir_runs, w.mir_full,
                w.n_round, w.t_draft_us/1000.0, w.t_catchup_us/1000.0, w.t_verify_us/1000.0, w.t_host_us/1000.0,
                w.t_catchup_pf_us/1000.0);
        if (graph_cache_k > 0) {
            // [TAG_GRAPH_CACHE] the graph-cache counters of the window (only with LLAMA_GRAPH_CACHE > 0)
            LLAMA_LOG_WARN("kqvis-prof: ctx=%s win=%" PRId64 " graph-cache hit=%d miss=%d check=%d inval=%d\n",
                    sched_prof_ctx_name(cparams.ctx_type), kqvis_n_win, w.gc_hit, w.gc_miss, w.gc_check, w.gc_inval);
        }
        kqvis_n_win++;
    }
    kqvis_win = kqvis_prof_window();
}

// [TAG_QSA_DUMP_IDS] (M4.2b item 5) LLAMA_QSA_DUMP_IDS=1 (default off): the QSA selection of every graph, as one
// stderr line per (graph, layer)
//   [qsa-ids] ctx=<main|mtp> g=<graph counter> il=<il> n_tps=<n> width=<w> fnv=<FNV-1a 64 of the i32 ids> first=<id0>
// so two runs (chunk 256 vs 0, gather q8 vs f32) can be compared on the device: identical selection <=> identical
// hashes at every (g, il). Installed as the context's eval callback (only if the caller set none), which asks for
// every tensor named "indexer_top_k-<il>" - the QSA_TOP_K output, or the reshape of the unfused chain's top-k,
// [width, n_tps, 1, n_stream] i32 - and reads it back after it is computed. A chunked layer (LLAMA_QSA_CHUNK) has
// one such tensor per chunk; they are hashed as one stream in execution order, which for n_stream = 1 is exactly the
// byte sequence of the whole ubatch's [width, n_tps] tensor, so chunked and unchunked hashes compare directly
// (n_stream > 1: the chunk order interleaves streams - compare like with like). Cost: nothing when unset (one static
// bool at context creation); when set, the scheduler cuts every split at each selection and synchronizes to read
// it back, which also changes where backend fusions can form - compare runs that BOTH have the dump on.
// [TAG_QSA_POOLED_CACHE] (M3) the same callback also hashes every "indexer_k-<il>" - the pooled block keys the
// selection scores - as one `[qsa-pooled]` line per (graph, layer) (see dump_pooled), so the pooled-key cache
// (LLAMA_QSA_POOLED_CACHE=1) and the recompute can be compared value for value, not only by the ids they select.
static bool llama_qsa_dump_ids_enabled() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_QSA_DUMP_IDS");
        return s != nullptr && atoi(s) != 0;
    }();

    return enabled;
}

struct llama_qsa_ids_dump {
    const char * ctx_name = "main";
    uint64_t graph = 0;          // graphs computed by this context so far
    bool     open  = false;      // a layer is being accumulated
    int      il    = -1;
    int64_t  width = 0;
    int64_t  n_tps = 0;
    int32_t  first = 0;
    uint64_t fnv   = 0;
    std::vector<uint8_t> buf;

    void flush() {
        if (open) {
            fprintf(stderr, "[qsa-ids] ctx=%s g=%llu il=%d n_tps=%lld width=%lld fnv=%016llx first=%d\n", ctx_name,
                    (unsigned long long) graph, il, (long long) n_tps, (long long) width, (unsigned long long) fnv, first);
            open = false;
        }
    }

    void end_graph() {
        flush();
        graph++;
    }

    // exactly "indexer_top_k-<il>": the views the attention takes of it are auto-named "... (view)" and must not be
    // hashed a second time
    static bool is_selection(const ggml_tensor * t) {
        if (strncmp(t->name, "indexer_top_k-", 14) != 0) {
            return false;
        }
        const char * p = t->name + 14;
        if (*p == '\0') {
            return false;
        }
        for (; *p; ++p) {
            if (*p < '0' || *p > '9') {
                return false;
            }
        }
        return true;
    }

    // [TAG_QSA_POOLED_CACHE] exactly "indexer_k-<il>": the pooled block keys the selection scores against,
    // [idx_dim, n_blocks, n_stream] f32 - the recompute's rope output, or the pooled-key cache's remap of its store
    static bool is_pooled(const ggml_tensor * t) {
        if (strncmp(t->name, "indexer_k-", 10) != 0) {
            return false;
        }
        const char * p = t->name + 10;
        if (*p == '\0') {
            return false;
        }
        for (; *p; ++p) {
            if (*p < '0' || *p > '9') {
                return false;
            }
        }
        return true;
    }

    //   [qsa-pooled] ctx=<main|mtp> g=<graph> il=<il> n_blocks=<n> n_stream=<ns> fnv=<FNV-1a 64 of the f32 bytes>
    // one line per (graph, layer), printed when the tensor is computed, i.e. before that layer's selection line
    void dump_pooled(ggml_tensor * t) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            LLAMA_LOG_WARN("%s: %s is not a contiguous f32 tensor - not hashed\n", __func__, t->name);
            return;
        }
        const size_t nb = ggml_nbytes(t);
        std::vector<uint8_t> pb(nb);
        ggml_backend_tensor_get(t, pb.data(), 0, nb);
        uint64_t h = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < nb; ++i) {
            h = (h ^ pb[i]) * 0x100000001b3ull;
        }
        fprintf(stderr, "[qsa-pooled] ctx=%s g=%llu il=%d n_blocks=%lld n_stream=%lld fnv=%016llx\n", ctx_name,
                (unsigned long long) graph, atoi(t->name + 10), (long long) t->ne[1], (long long) t->ne[2],
                (unsigned long long) h);
    }

    static bool eval(ggml_tensor * t, bool ask, void * user_data) {
        if (is_pooled(t)) {
            if (!ask) {
                ((llama_qsa_ids_dump *) user_data)->dump_pooled(t);
            }
            return true;
        }
        if (!is_selection(t)) {
            return !ask; // ask: not needed; never reached with ask == false for a tensor we did not ask for
        }
        if (ask) {
            return true;
        }
        auto * d = (llama_qsa_ids_dump *) user_data;
        const int t_il = atoi(t->name + 14);
        if (d->open && t_il != d->il) {
            d->flush();
        }
        if (t->type != GGML_TYPE_I32 || !ggml_is_contiguous(t)) {
            LLAMA_LOG_WARN("%s: %s is not a contiguous i32 tensor - not hashed\n", __func__, t->name);
            return true;
        }
        if (!d->open) {
            d->open  = true;
            d->il    = t_il;
            d->width = t->ne[0];
            d->n_tps = 0;
            d->fnv   = 0xcbf29ce484222325ull; // FNV-1a 64 offset basis
        }
        const size_t nb = ggml_nbytes(t);
        d->buf.resize(nb);
        ggml_backend_tensor_get(t, d->buf.data(), 0, nb);
        if (d->n_tps == 0) {
            memcpy(&d->first, d->buf.data(), sizeof(int32_t));
        }
        for (size_t i = 0; i < nb; ++i) {
            d->fnv = (d->fnv ^ d->buf[i]) * 0x100000001b3ull;
        }
        d->n_tps += t->ne[1]*t->ne[3]; // queries x streams
        return true; // keep computing the split
    }
};

// [TAG_QSA_POOLED_CACHE] after a graph of a hybrid-idx memory: publish or drop its pooled-key stamps
static void llama_qsa_store_publish(llama_memory_context_i * mctx, bool ok) {
    if (const auto * c = dynamic_cast<const llama_memory_hybrid_idx_context *>(mctx)) {
        c->qsa_store_publish(ok);
    }
}

// [TAG_QSA_CENSUS_INC] the plan counters of a hybrid-idx memory's store (zeros without one), for the kqvis-prof window
static llama_qsa_pooled_store::stats_t llama_qsa_store_stats(llama_memory_context_i * mctx) {
    if (const auto * c = dynamic_cast<const llama_memory_hybrid_idx_context *>(mctx)) {
        if (const auto * st = c->get_qsa_store()) {
            return st->stats;
        }
    }
    return llama_qsa_pooled_store::stats_t();
}

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

static const llm_fused_op_probe llm_fused_op_lid_probe = {
    /*.op               =*/ LLM_FUSED_OP_LIGHTNING_INDEXER,
    /*.name             =*/ "Lightning Indexer",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_pre_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_PRE,
    /*.name             =*/ "fused DeepSeek V4 HC pre",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_comb_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_COMB,
    /*.name             =*/ "fused DeepSeek V4 HC comb",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_dsv4_hc_post_probe = {
    /*.op               =*/ LLM_FUSED_OP_DSV4_HC_POST,
    /*.name             =*/ "fused DeepSeek V4 HC post",
    /*.n_tokens_per_seq =*/ 1,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    // +1: id n_layer() taps the output of the last layer ("input" of the head)
    cparams.embeddings_layer_inp.resize(hparams.n_layer() + 1, false);
    embd_layer_inp.resize(hparams.n_layer() + 1);

    cparams.ctx_type     = params.ctx_type;
    cparams.pooling_type = params.pooling_type;

    // [TAG_MTP_CATCHUP_NOATTN] LLAMA_MTP_CATCHUP_ATTN (MTP contexts only, read once per context): 1 / unset = today's
    // full draft graph for every ubatch; 0 = a ubatch with zero outputs and no unmasked hidden export (the speculative
    // catch-up of prefill and verification rows) builds only the draft block's K/V projections and the cache writes -
    // no attention, head or FFN (R80.4, item 7). Exact: the written K/V bytes do not depend on the attention output.
    // Strict parse as LLAMA_MTP_CARRIER_ZERO: only a plain 0 or 1; anything else keeps the default with a WARN.
    cparams.mtp_catchup_attn = true;
    if (cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        const char * s = getenv("LLAMA_MTP_CATCHUP_ATTN");
        if (s != nullptr && !(strcmp(s, "0") == 0 || strcmp(s, "1") == 0)) {
            LLAMA_LOG_WARN("%s: invalid LLAMA_MTP_CATCHUP_ATTN='%s' (0 or 1) - the full catch-up graph stays\n", __func__, s);
        } else if (s != nullptr) {
            cparams.mtp_catchup_attn = s[0] == '1';
        }
        LLAMA_LOG_INFO("%s: LLAMA_MTP_CATCHUP_ATTN=%d%s: %s\n", __func__, cparams.mtp_catchup_attn ? 1 : 0,
                s == nullptr ? " (default)" : "",
                cparams.mtp_catchup_attn ? "zero-output MTP catch-up ubatches build the full draft graph"
                                         : "zero-output MTP catch-up ubatches store K/V only (attention-free)");
    }

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    // [TAG_QSA_DUMP_IDS]
    if (llama_qsa_dump_ids_enabled()) {
        if (cparams.cb_eval == nullptr) {
            qsa_ids_dump = std::make_shared<llama_qsa_ids_dump>();
            qsa_ids_dump->ctx_name    = sched_prof_ctx_name(params.ctx_type);
            cparams.cb_eval           = llama_qsa_ids_dump::eval;
            cparams.cb_eval_user_data = qsa_ids_dump.get();
            LLAMA_LOG_WARN("%s: LLAMA_QSA_DUMP_IDS=1: dumping the QSA selection hashes of the %s context to stderr\n",
                    __func__, qsa_ids_dump->ctx_name);
        } else {
            LLAMA_LOG_WARN("%s: LLAMA_QSA_DUMP_IDS=1 ignored: the caller installed its own eval callback\n", __func__);
        }
    }

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    cparams.fused_lid    = true;
    cparams.auto_flid    = true;

    cparams.fused_dsv4_hc_pre  = true;
    cparams.fused_dsv4_hc_comb = true;
    cparams.fused_dsv4_hc_post = true;
    cparams.auto_fhc           = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;
    cparams.n_outputs_max_per_seq = params.n_outputs_max_per_seq == 0 ?
            cparams.n_outputs_max : std::min(params.n_outputs_max_per_seq, cparams.n_outputs_max);

    // [TAG_KV_GRAPH_SPAN] cap for the worst-case graph reserve only - see llama_kv_cache::init_full()
    cparams.kv_graph_span = params.kv_graph_span;

    // [TAG_QSA_RATIO_OVERRIDE] must be set before create_memory() below: the indexer KV cache is
    // allocated from the *effective* ratios, so a forced-dense context must not allocate it and a
    // forced-QSA context must
    cparams.qsa_ratio = params.qsa_ratio;

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }

        // [TAG_GRAPH_CACHE]
        const char * LLAMA_GRAPH_CACHE       = getenv("LLAMA_GRAPH_CACHE");
        const char * LLAMA_GRAPH_CACHE_CHECK = getenv("LLAMA_GRAPH_CACHE_CHECK");
        graph_cache_k     = LLAMA_GRAPH_CACHE ? std::max(0, atoi(LLAMA_GRAPH_CACHE)) : 0;
        graph_cache_check = graph_cache_k > 0 && LLAMA_GRAPH_CACHE_CHECK ? std::max(0, atoi(LLAMA_GRAPH_CACHE_CHECK)) : 0;
        if (graph_cache_check > 0) {
            const char * inj = getenv("LLAMA_GRAPH_CACHE_CHECK_INJECT");
            graph_cache_inject = inj ? inj : "";
        }
        if (graph_cache_k > 0) {
            LLAMA_LOG_INFO("%s: graph cache: K = %d, check = %d\n", __func__, graph_cache_k, graph_cache_check);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max             = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx                 = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq             = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch               = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch              = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn           = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn            = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified            = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base             = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale            = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq              = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max         = %u\n",   __func__, cparams.n_outputs_max);
    LLAMA_LOG_INFO("%s: n_outputs_max_per_seq = %u\n",   __func__, cparams.n_outputs_max_per_seq);
    LLAMA_LOG_INFO("%s: kv_graph_span         = %u%s\n", __func__, cparams.kv_graph_span,
            cparams.kv_graph_span == 0 ? " (full KV size)" : " (worst-case graph reserve only)");
    if (hparams.indexer_head_size > 0) {
        // [TAG_QSA_RATIO_OVERRIDE]
        LLAMA_LOG_INFO("%s: qsa_ratio             = %d%s\n", __func__, cparams.qsa_ratio,
                cparams.qsa_ratio < 0  ? " (from the GGUF)" :
                cparams.qsa_ratio == 0 ? " (forced dense)"  : " (forced sparse)");
        // [TAG_MULTISLOT] round 2: the complete-history dense fallback, once per context
        const char * reason = nullptr;
        const bool   hd     = llama_qsa_history_dense(&reason);
        const char * hom    = getenv("LLAMA_QSA_HOMOGENEOUS");
        LLAMA_LOG_INFO("%s: qsa_history_dense = %s (%s)%s\n", __func__, hd ? "on" : "off", reason,
                hom != nullptr && strcmp(hom, "1") == 0 && !hd ? "; coverage check still on: LLAMA_QSA_HOMOGENEOUS=1" : "");
    }

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));

        // [TAG_KV_GRAPH_SPAN]
        // shrink the n_kv that init_full() simulates for the worst-case compute-buffer reserve.
        // this only changes the size of the buffers that are reserved up front - the n_kv of real
        // batches still comes from llama_kv_cache::get_n_kv(sinfo). if a later graph genuinely needs
        // more, ggml-alloc reallocates it on demand (see ggml_backend_sched_alloc_splits).
        if (memory && cparams.kv_graph_span > 0) {
            memory->set_n_kv_reserve_max(cparams.kv_graph_span);
        }
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    // wait for any pending asynchronous copies into the output buffers before they are freed
    synchronize();

    if (llama_kqvis_prof_enabled()) {
        kqvis_prof_flush("exit"); // [TAG_KQVIS_PROF]
    }

    if (graph_cache_k > 0) {
        // [TAG_GRAPH_CACHE] totals of this context
        const auto & t = graph_cache_tot;
        LLAMA_LOG_INFO("graph-cache: ctx=%s K=%d hit=%" PRId64 " miss=%" PRId64 " check=%" PRId64 " inval=%" PRId64
                " evict=%" PRId64 " clear=%" PRId64 " check2=%" PRId64 " check_sched=%" PRId64 " park_fail=%" PRId64 " bypass=%" PRId64 " check2_state_only=%" PRId64 "\n",
                sched_prof_ctx_name(cparams.ctx_type), graph_cache_k, t.hit, t.miss, t.check, t.inval, t.evict, t.clear,
                t.check2, t.check_sched, t.park_fail, t.bypass, t.check2_state_only);
        graph_cache.clear(); // before the scheduler (the snapshots reference its tensors only by pointer)
    }

    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }

    if (cparams.auto_flid) {
        LLAMA_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", func);
        resolve(llm_fused_op_lid_probe, cparams.fused_lid);
        cparams.auto_flid = false;
    }

    if (cparams.auto_fhc) {
        LLAMA_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", func);
        resolve(llm_fused_op_dsv4_hc_pre_probe,  cparams.fused_dsv4_hc_pre);
        resolve(llm_fused_op_dsv4_hc_comb_probe, cparams.fused_dsv4_hc_comb);
        resolve(llm_fused_op_dsv4_hc_post_probe, cparams.fused_dsv4_hc_post);
        cparams.auto_fhc = false;
    }
}

// [TAG_RESERVE_RUNTIME_SHAPE] (M4.2b item 1) LLAMA_RESERVE_RUNTIME_SHAPE=1 (default off) adds the single-sequence
// prefill shape to the worst-case pp reserve. graph_reserve rounds n_tokens UP to a multiple of n_seqs (2048 tokens and
// --parallel 5 reserve 5 x 410 = 2050 tokens), a ubatch that cannot occur at run time, while the ubatch every long
// prefill really runs is one sequence of n_ubatch tokens. A graph whose buffer need is not monotone in n_tokens is then
// under-reserved at the reserved n_kv itself - measured with LLAMA_QSA_CHUNK=256: 2050 tokens are cut into 9 x 228
// query chunks, 2048 into 8 x 256, and at the same n_kv 196 608 the run-time graph needs 2 381.01 MiB against the
// 2 336.45 MiB reserve, so the compute buffer regrows from n_kv 190 464 on, 3 ubatches BEFORE the --kv-graph-span
// (qsa_fusion_replay REPLAY_NKV_TABLE, M4.2b report). With the knob the reserve is the max over both shapes, so the
// first regrow moves to the first ubatch that really exceeds the span. Cost: the difference, at idle (+44.56 MiB at
// span 196 608, +75.41 MiB at 245 760 with chunk 256; 0 for a graph whose need grows with n_tokens, e.g. chunk 0).
static bool llama_reserve_runtime_shape() {
    static const bool enabled = [] {
        const char * s = getenv("LLAMA_RESERVE_RUNTIME_SHAPE");
        return s != nullptr && atoi(s) != 0;
    }();

    return enabled;
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    graph_cache_clear(); // [TAG_GRAPH_CACHE] the scheduler is replaced below
    graph_cache_max_nodes = max_nodes;
    graph_cache_check_res.reset(); // the CHECK scratch objects follow the scheduler and its node budget
    graph_cache_check_sched.reset();
    graph_cache_check_owner = nullptr;

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // [TAG_RESERVE_RUNTIME_SHAPE] the single-sequence prefill ubatch (see llama_reserve_runtime_shape). Allocating
    // path: ggml-alloc only grows a single-chunk buffer, so the buffer ends at the max of both shapes. Estimate path
    // (no_alloc): take the max of the two size estimates explicitly.
    // Unified KV only: with --no-kv-unified the full-memory context spans n_seq_max KV streams, and build_attn splits
    // the queries into k->ne[3] streams (llm_graph_context::build_attn_mha), so a single-sequence ubatch of n_tokens
    // (not a multiple of n_stream) yields an attention output with n_stream*floor(n_tokens/n_stream) rows while the
    // other operands keep n_tokens - GGML_ASSERT(ggml_can_repeat) in the first residual/gate op (Qwen3.8-27B, -ub 512,
    // --parallel 5 --no-kv-unified). There a run-time single-sequence ubatch also sees one stream, not the n_seq_max
    // the full-memory context holds, so this reserve would not match the run-time graph anyway.
    if (llama_reserve_runtime_shape() && n_seqs > 1 && cparams.kv_unified) {
        std::vector<size_t> sizes_rt(backend_buf_exp_size.size(), 0);

        auto * gf = graph_reserve(n_tokens, 1, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? sizes_rt.data() : nullptr);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers (single-sequence shape)");
        }
        if (model.hparams.no_alloc) {
            for (size_t i = 0; i < backend_buf_exp_size.size(); ++i) {
                backend_buf_exp_size[i] = std::max(backend_buf_exp_size[i], sizes_rt[i]);
            }
        }

        LLAMA_LOG_INFO("%s: LLAMA_RESERVE_RUNTIME_SHAPE: also reserved the %u-token single-sequence pp graph\n", __func__, n_tokens);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    const bool prof = sched_prof_enabled();

    int64_t t_prof_us = 0;
    if (prof) {
        t_prof_us = ggml_time_us();
    }

    // [TAG_KQVIS_PROF] the device wait for queued work, charged to the open window
    const int64_t t_kqvis_us = (n_queued_tokens > 0 && llama_kqvis_prof_enabled()) ? ggml_time_us() : 0;

    ggml_backend_sched_synchronize(sched.get());

    if (t_kqvis_us > 0) {
        const int64_t t_now_us = ggml_time_us();
        kqvis_win.sync_us += t_now_us - t_kqvis_us;
        kqvis_win.n_sync++;
        kqvis_win.t1_ms = kqvis_unix_ms();
        g_kqvis_last_us.store(t_now_us);
    }

    if (prof && n_queued_tokens > 0) {
        // the GPU wait for everything decode() enqueued since the previous synchronize (the async graph
        // compute plus the async output reads); printed only when there was queued work, so the many
        // no-op synchronize() calls from the getters stay silent
        fprintf(stderr, "[sched-prof] sync ctx=%s n_queued=%" PRId64 " wait=%.3fms\n",
                sched_prof_ctx_name(cparams.ctx_type), n_queued_tokens, (ggml_time_us() - t_prof_us)/1000.0);
    }

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();
        graph_cache_clear(); // [TAG_GRAPH_CACHE]

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

void llama_context::set_rs_rollback_depth(uint32_t depth) {
    rs_rollback_depth = depth;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

// [TAG_GRAPH_CACHE]
void llama_context::graph_cache_clear() {
    if (graph_cache.empty() && !graph_cache_active) {
        return;
    }
    bool had_resident = false;
    for (const auto & e : graph_cache) {
        had_resident |= e.resident;
    }
    if (!graph_cache.empty()) {
        graph_cache_tot.clear++;
    }
    graph_cache.clear();
    if (had_resident && sched) {
        // the live state belonged to the dropped resident entry
        ggml_backend_sched_reset(sched.get());
    }
    graph_cache_active = false;
}

llm_graph_result * llama_context::graph_cache_get(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret, bool & reused) {
    reused = false;
    if (!graph_cache_active) {
        // the live scheduler state may still belong to gf_res_prev: never reuse it while the cache owns the scheduler
        gf_res_prev->reset();
        graph_cache_active = true;
        // every cached state keeps the split context it was made in: size those from the graphs, not from the
        // worst case of graph_max_nodes (~ graph_max_nodes * 60 tensors, 1.6 GiB virtual at 81920 nodes)
        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_release_context(sched.get());
    }

    ggml_backend_sched_t s = sched.get();

    // the resident entry, if any (its state is the scheduler's live state)
    auto resident_it = std::find_if(graph_cache.begin(), graph_cache.end(), [](const graph_cache_entry & e) { return e.resident; });

    // lookup: most recently used first
    for (size_t i = 0; i < graph_cache.size(); ++i) {
        auto & e = graph_cache[i];
        const auto gparams = graph_params(e.res.get(), ubatch, mctx, gtype);
        if (!e.res->can_reuse(gparams)) {
            continue;
        }

        if (!e.resident) {
            const bool ok = ggml_backend_sched_restore(s, e.snap.get());
            // the snapshot object now holds the previous live state: it goes back to the previous resident
            std::unique_ptr<ggml_backend_sched_snapshot, sched_snapshot_deleter> prev(e.snap.release());
            if (resident_it != graph_cache.end()) {
                resident_it->snap     = std::move(prev);
                resident_it->resident = false;
            }
            if (!ok) {
                // stale: the compute buffers were reallocated since this graph was allocated
                graph_cache_tot.inval++;
                graph_cache.erase(graph_cache.begin() + i);
                resident_it = graph_cache.end();
                --i;
                continue;
            }
            e.resident = true;
        }

        if (graph_cache_check) {
            // rebuild into a scratch result on a scratch scheduler (nothing is allocated) and compare
            if (!graph_cache_check_sched || graph_cache_check_owner != s || graph_cache_check_nodes != graph_cache_max_nodes) {
                // (re)made for the current scheduler and node budget (a replaced scheduler / a grown budget, e.g.
                // after a LoRA load, must not leave CHECK on the old capacity)
                graph_cache_check_res.reset();
                graph_cache_check_sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                        graph_cache_max_nodes, false, cparams.op_offload));
                ggml_backend_sched_release_context(graph_cache_check_sched.get());
                graph_cache_check_res.reset(new llm_graph_result(graph_cache_max_nodes));
                graph_cache_check_owner = s;
                graph_cache_check_nodes = graph_cache_max_nodes;
                graph_cache_tot.check_sched++;
            }
            ggml_backend_sched_reset(graph_cache_check_sched.get());
            graph_cache_check_res->reset();
            auto gp_check = graph_params(graph_cache_check_res.get(), ubatch, mctx, gtype);
            gp_check.sched = graph_cache_check_sched.get();
            ggml_cgraph * gf_check = model.build_graph(gp_check);
            if (gf_check != nullptr) {
                // the cached graph went through the split (backend graph_optimize may reorder nodes in place):
                // split the fresh one the same way (no allocation)
                ggml_backend_sched_split_graph(graph_cache_check_sched.get(), gf_check);
            }
            llm_graph_topology_cmp cmp;
            if (graph_cache_inject == "ptr" || graph_cache_inject == "view" || graph_cache_inject == "route") {
                // test seam: corrupt the cached entry once; the checks below must abort
                for (const auto & r : e.manifest) {
                    auto * t = const_cast<ggml_tensor *>(r.t);
                    if (graph_cache_inject == "ptr" && t->view_src == nullptr && r.size > 0) {
                        t->data = (char *) t->data + 64;
                        break;
                    }
                    if (graph_cache_inject == "view" && t->view_src != nullptr) {
                        t->view_offs += 16;
                        break;
                    }
                }
                cmp.inject_route = graph_cache_inject == "route";
                graph_cache_inject.clear();
            }
            graph_cache_manifest_check(e);
            if (gf_check == nullptr || !cmp.graph_eq(e.res->get_gf(), gf_check)) {
                GGML_ABORT("graph-cache check: the cached graph differs from a fresh build: %s", gf_check ? cmp.why.c_str() : "rebuild failed");
            }
            if (!cmp.sched_eq(s, e.res->get_gf(), graph_cache_check_sched.get(), gf_check)) {
                GGML_ABORT("graph-cache check: the cached split plan differs from a fresh build: %s", cmp.why.c_str());
            }
            graph_cache_tot.check++;
            graph_cache_check_res->reset();
        }

        // move to the front
        std::rotate(graph_cache.begin(), graph_cache.begin() + i, graph_cache.begin() + i + 1);
        graph_cache_tot.hit++;
        reused = true;
        return graph_cache.front().res.get();
    }

    // miss: park the resident state, evict the least recently used entry, build and allocate a new graph
    graph_cache_tot.miss++;
    resident_it = std::find_if(graph_cache.begin(), graph_cache.end(), [](const graph_cache_entry & e) { return e.resident; });
    if (resident_it != graph_cache.end()) {
        ggml_backend_sched_snapshot_t snap = ggml_backend_sched_take(s);
        if (snap != nullptr) {
            resident_it->snap.reset(snap);
            resident_it->resident = false;
        } else {
            // the live state cannot be parked (allocation failure): drop its entry, the scheduler starts over
            LLAMA_LOG_WARN("%s: graph cache: cannot park the resident graph, dropping it\n", __func__);
            ggml_backend_sched_reset(s);
            graph_cache.erase(resident_it);
            graph_cache_tot.park_fail++;
        }
    }
    while ((int32_t) graph_cache.size() >= graph_cache_k) {
        graph_cache.pop_back();
        graph_cache_tot.evict++;
    }
    graph_cache.reserve(graph_cache_k); // the insertion below cannot reallocate (no throw after the allocation)

    graph_cache_entry e;
    e.res.reset(new llm_graph_result(graph_cache_max_nodes));

    ggml_backend_sched_reset(s);
    ggml_backend_sched_set_eval_callback(s, cparams.cb_eval, cparams.cb_eval_user_data);

    const auto gparams = graph_params(e.res.get(), ubatch, mctx, gtype);
    ggml_cgraph * gf = model.build_graph(gparams);
    if (!gf) {
        LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
        ggml_backend_sched_reset(s); // no live state may refer to the result destroyed on return
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    if (!ggml_backend_sched_alloc_graph(s, gf)) {
        LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
        ggml_backend_sched_reset(s);
        ret = GGML_STATUS_ALLOC_FAILED;
        return nullptr;
    }
    try {
        e.resident   = true;
        e.buffer_gen = ggml_backend_sched_buffer_gen(s);
        if (graph_cache_check) {
            graph_cache_manifest_take(e);
        }
        graph_cache.insert(graph_cache.begin(), std::move(e));
    } catch (...) {
        // the live state refers to e's graph, which is destroyed on unwinding: leave the scheduler reset
        ggml_backend_sched_reset(s);
        throw;
    }
    return graph_cache.front().res.get();
}

// every tensor the graph and its scheduler state reach, each once: the graph's nodes, then the scheduler's split
// graph copy (sched != nullptr: its nodes = the split input dependency views + the input copies + the split nodes,
// its leafs = the graph's leafs), and from each of these the sources and view sources. A split replaces node->src by a backend
// copy with no link back, so an original input consumed only through a copy is reached here through its leaf entry
// and through the dependency view's src[0]. Two equal plans (CHECK=1 compared them) are walked in the same order.
static void graph_cache_collect(ggml_cgraph * gf, ggml_backend_sched_t sched, std::vector<const ggml_tensor *> & out) {
    std::unordered_set<const ggml_tensor *> seen;
    std::vector<const ggml_tensor *> stack;
    auto walk = [&](const ggml_tensor * root) {
        stack.push_back(root);
        while (!stack.empty()) {
            const ggml_tensor * t = stack.back();
            stack.pop_back();
            if (t == nullptr || !seen.insert(t).second) {
                continue;
            }
            out.push_back(t);
            stack.push_back(t->view_src);
            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                stack.push_back(t->src[j]);
            }
        }
    };
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        walk(ggml_graph_node(gf, i));
    }
    if (sched != nullptr) {
        ggml_tensor ** leafs = nullptr;
        int n_leafs = 0;
        ggml_cgraph * gs = ggml_backend_sched_get_graph_copy(sched, &leafs, &n_leafs);
        for (int i = 0; i < ggml_graph_n_nodes(gs); ++i) {
            walk(ggml_graph_node(gs, i));
        }
        for (int i = 0; i < n_leafs; ++i) {
            walk(leafs[i]);
        }
    }
}

static bool graph_cache_is_compute(const ggml_tensor * t) {
    return t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE;
}

void llama_context::graph_cache_manifest_take(graph_cache_entry & e) {
    std::vector<const ggml_tensor *> ts;
    graph_cache_collect(e.res->get_gf(), sched.get(), ts);
    e.manifest.clear();
    for (const ggml_tensor * t : ts) {
        if (!graph_cache_is_compute(t) || t->data == nullptr) {
            continue;
        }
        const char * base = (const char *) ggml_backend_buffer_get_base(t->buffer);
        e.manifest.push_back({ t, t->buffer, (size_t) ((const char *) t->data - base), ggml_nbytes(t), t->view_src, t->view_offs });
    }
}

void llama_context::graph_cache_manifest_check(const graph_cache_entry & e) {
    if (e.buffer_gen != ggml_backend_sched_buffer_gen(sched.get())) {
        GGML_ABORT("graph-cache check: hit on an entry of buffer generation %" PRIu64 ", current %" PRIu64,
                e.buffer_gen, ggml_backend_sched_buffer_gen(sched.get()));
    }
    // every compute-buffer tensor the graph reaches now must be in the manifest, unchanged, in bounds and consistent
    std::unordered_map<const ggml_tensor *, const graph_cache_alloc_rec *> by_t;
    for (const auto & r : e.manifest) {
        by_t[r.t] = &r;
    }
    std::vector<const ggml_tensor *> ts;
    graph_cache_collect(e.res->get_gf(), sched.get(), ts);
    for (const ggml_tensor * t : ts) {
        if (t->buffer == nullptr && t->data == nullptr) {
            continue;
        }
        if (t->buffer == nullptr || !graph_cache_is_compute(t)) {
            if (by_t.count(t)) {
                GGML_ABORT("graph-cache check: %s: left its compute buffer", t->name);
            }
            continue; // persistent: compared by buffer + data in graph_eq
        }
        const auto it = by_t.find(t);
        if (it == by_t.end()) {
            GGML_ABORT("graph-cache check: %s: compute tensor not in the allocation manifest", t->name);
        }
        const auto & r = *it->second;
        const char * base = (const char *) ggml_backend_buffer_get_base(t->buffer);
        const size_t offs = (size_t) ((const char *) t->data - base);
        if (t->buffer != r.buf || offs != r.offs || ggml_nbytes(t) != r.size) {
            GGML_ABORT("graph-cache check: %s: allocation differs from the manifest (buffer %p/%p offs %zu/%zu size %zu/%zu)",
                    t->name, (void *) t->buffer, (void *) r.buf, offs, r.offs, ggml_nbytes(t), r.size);
        }
        if ((const char *) t->data < base || offs + r.size > ggml_backend_buffer_get_size(t->buffer)) {
            GGML_ABORT("graph-cache check: %s: out of the bounds of its buffer (offs %zu size %zu buffer %zu)",
                    t->name, offs, r.size, ggml_backend_buffer_get_size(t->buffer));
        }
        if (t->view_src != r.view_src || t->view_offs != r.view_offs ||
            (t->view_src != nullptr && (const char *) t->data != (const char *) t->view_src->data + t->view_offs)) {
            GGML_ABORT("graph-cache check: %s: view inconsistent (data != view_src->data + view_offs or view changed)", t->name);
        }
    }
}

static void graph_cache_read(const ggml_tensor * t, std::vector<uint8_t> & out) {
    out.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size());
}

// CHECK=3: fill every compute-buffer INPUT tensor the graph (and its scheduler state) reaches with a sentinel. Split
// input copies carry no INPUT flag (one copy slot): the scheduler rewrites them from their source on every compute
static void graph_cache_poison_inputs(ggml_cgraph * gf, ggml_backend_sched_t sched, uint8_t v) {
    std::vector<const ggml_tensor *> ts;
    graph_cache_collect(gf, sched, ts);
    for (const ggml_tensor * t : ts) {
        if ((t->flags & GGML_TENSOR_FLAG_INPUT) && t->view_src == nullptr && t->data != nullptr && graph_cache_is_compute(t)) {
            ggml_backend_tensor_memset(const_cast<ggml_tensor *>(t), v, 0, ggml_nbytes(t));
        }
    }
}

void llama_context::graph_cache_check_exec_pre(llm_graph_result * res, const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx) {
    ggml_backend_sched_synchronize(sched.get()); // the previous graph may still be writing the state
    auto & x = graph_cache_exec;
    x.outputs.clear();
    x.state.clear();

    ggml_cgraph * gf_c = res->get_gf();
    const int n = ggml_graph_n_nodes(gf_c);

    // every persistent non-weight tensor the graph reaches (KV / recurrent / rollback / QSA state, also when an op
    // writes it as a side effect of a source): its bytes before. Weights are never written and are skipped.
    std::vector<std::pair<const ggml_tensor *, std::vector<uint8_t>>> before;
    {
        std::vector<const ggml_tensor *> ts;
        graph_cache_collect(gf_c, sched.get(), ts);
        std::set<std::pair<const void *, size_t>> seen;
        for (const ggml_tensor * t : ts) {
            if (t->buffer == nullptr || t->data == nullptr || graph_cache_is_compute(t) ||
                ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                !seen.insert({ t->data, ggml_nbytes(t) }).second) {
                continue;
            }
            before.emplace_back(t, std::vector<uint8_t>());
            graph_cache_read(t, before.back().second);
        }
    }

    // a fresh plan of the same ubatch on the scratch scheduler, allocated in its own compute buffers
    ggml_backend_sched_t cs = graph_cache_check_sched.get();
    ggml_backend_sched_reset(cs);
    graph_cache_check_res->reset();
    auto gp = graph_params(graph_cache_check_res.get(), ubatch, mctx, gtype);
    gp.sched = cs;
    ggml_cgraph * gf_f = model.build_graph(gp);
    if (gf_f == nullptr || ggml_graph_n_nodes(gf_f) != n || !ggml_backend_sched_alloc_graph(cs, gf_f)) {
        GGML_ABORT("graph-cache check=2: the fresh plan could not be built/allocated");
    }
    graph_cache_exec_n_tokens = ubatch.n_tokens;
    if (graph_cache_check >= 3) {
        // CHECK=3: the fresh plan prepares its inputs independently - a second set_inputs over a sentinel fill (0x5A;
        // the cached graph's inputs were filled with 0xA5 before its own set_inputs), so a region either set_inputs
        // leaves unwritten differs between the two plans. The one-shot inputs (get_one_shot: the recurrent s_copy,
        // whose first preparation consumed a pending rollback) are copied from the cached graph instead.
        // Other set_input side effects are idempotent: the QSA incremental mirrors (persistent store buffers, the
        // second call finds nothing dirty) and the plan caches keyed by the ubatch.
        graph_cache_poison_inputs(gf_f, cs, 0x5A);
        graph_cache_check_res->set_inputs(&ubatch);
        std::vector<ggml_tensor *> oc, of;
        res->get_one_shot_inputs(oc);
        graph_cache_check_res->get_one_shot_inputs(of);
        GGML_ASSERT(oc.size() == of.size());
        std::vector<uint8_t> buf;
        for (size_t i = 0; i < oc.size(); ++i) {
            if (oc[i]->data == nullptr || of[i]->data == nullptr) {
                continue;
            }
            GGML_ASSERT(ggml_nbytes(oc[i]) == ggml_nbytes(of[i]));
            graph_cache_read(oc[i], buf);
            ggml_backend_tensor_set(of[i], buf.data(), 0, buf.size());
        }
    } else {
        // CHECK=2: the inputs exactly as the cached graph got them: set_inputs is not repeated (some inputs are
        // one-shot, see above). This proves same-input execution equivalence only: a stale input the cached graph
        // got is given to the fresh plan as well (CHECK=3 prepares them independently). Equal topologies are
        // collected in the same order, so the i-th tensors correspond.
        std::vector<const ggml_tensor *> tc, tf;
        graph_cache_collect(gf_c, sched.get(), tc);
        graph_cache_collect(gf_f, cs, tf);
        GGML_ASSERT(tc.size() == tf.size());
        std::vector<uint8_t> buf;
        for (size_t i = 0; i < tc.size(); ++i) {
            if (!(tc[i]->flags & GGML_TENSOR_FLAG_INPUT) || tc[i]->data == nullptr || tc[i]->data == tf[i]->data) {
                continue;
            }
            GGML_ASSERT((tf[i]->flags & GGML_TENSOR_FLAG_INPUT) && ggml_nbytes(tc[i]) == ggml_nbytes(tf[i]));
            graph_cache_read(tc[i], buf);
            ggml_backend_tensor_set(const_cast<ggml_tensor *>(tf[i]), buf.data(), 0, buf.size());
        }
    }
    if (graph_cache_inject == "split-input") {
        // test seam (once): zero the largest original INPUT that the fresh plan consumes only through a backend copy
        // (a split input; zero keeps index inputs in bounds). The fresh outputs then differ: the check must abort
        ggml_tensor * victim = nullptr;
        for (int i = 0; i < ggml_backend_sched_get_n_splits(cs); ++i) {
            int b, i0, i1;
            const int n_in = ggml_backend_sched_get_split(cs, i, &b, &i0, &i1);
            for (int j = 0; j < n_in; ++j) {
                ggml_tensor * in = nullptr, * cpy = nullptr;
                ggml_backend_sched_get_split_input(cs, i, j, &in, &cpy);
                if ((in->flags & GGML_TENSOR_FLAG_INPUT) && (victim == nullptr || ggml_nbytes(in) > ggml_nbytes(victim))) {
                    victim = in;
                }
            }
        }
        if (victim != nullptr) { // else: stays armed for a later hit
            LLAMA_LOG_WARN("graph-cache: inject split-input: zeroing %s (consumed through a backend copy)\n", victim->name);
            ggml_backend_tensor_memset(victim, 0, 0, ggml_nbytes(victim));
            graph_cache_inject.clear();
        }
    }
    if (graph_compute(gf_f, ubatch.n_tokens > 1, cs) != GGML_STATUS_SUCCESS) {
        GGML_ABORT("graph-cache check=2: the fresh plan failed to compute");
    }
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * tf = ggml_graph_node(gf_f, i);
        if (tf->flags & GGML_TENSOR_FLAG_OUTPUT) {
            x.outputs.emplace_back(ggml_graph_node(gf_c, i), std::vector<uint8_t>());
            graph_cache_read(tf, x.outputs.back().second);
        }
    }
    // the state the fresh plan wrote, then the state back as it was for the cached graph
    for (auto & [t, bytes] : before) {
        x.state.emplace_back(t, std::vector<uint8_t>());
        graph_cache_read(t, x.state.back().second);
    }
    for (auto & [t, bytes] : before) {
        ggml_backend_tensor_set(const_cast<ggml_tensor *>(t), bytes.data(), 0, bytes.size());
    }
    ggml_backend_sched_reset(cs);
    graph_cache_check_res->reset();
}

void llama_context::graph_cache_check_exec_post(llm_graph_result * res) {
    GGML_UNUSED(res);
    if (graph_cache_exec.outputs.empty() && graph_cache_exec.state.empty()) {
        // every decode graph produces outputs or writes the memory state: an empty set means the check saw nothing
        GGML_ABORT("graph-cache check=2: nothing to compare (no outputs, no state tensors)");
    }
    if (graph_cache_exec.outputs.empty()) {
        // a state-only graph (the KV-only MTP catch-up: the cache writes are its only roots): the state bytes decide
        graph_cache_tot.check2_state_only++;
    }
    ggml_backend_sched_synchronize(sched.get());
    std::vector<uint8_t> got;
    for (const auto * set : { &graph_cache_exec.outputs, &graph_cache_exec.state }) {
        for (const auto & [t, want] : *set) {
            graph_cache_read(t, got);
            if (got != want) {
                size_t i0 = 0;
                while (i0 < got.size() && got[i0] == want[i0]) { ++i0; }
                GGML_ABORT("graph-cache check=2: %s %s differs between the cached and a fresh plan (hit %" PRId64 ", n_tokens %u, byte %zu of %zu)",
                        set == &graph_cache_exec.outputs ? "output" : "state", t->name, graph_cache_tot.hit, graph_cache_exec_n_tokens, i0, got.size());
            }
        }
    }
    graph_cache_exec.outputs.clear();
    graph_cache_exec.state.clear();
    graph_cache_tot.check2++;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    const bool prof = sched_prof_enabled();

    int64_t t_prof_us = 0;
    if (prof) {
        t_prof_us = ggml_time_us();
        sched_prof.t_start_us = t_prof_us;
    }

    // [TAG_KQVIS_PROF]
    const bool kqprof = llama_kqvis_prof_enabled();
    int64_t kq_t_start_us = 0;
    int64_t kq_t_start_ms = 0;
    bool    kq_reused     = false;
    int64_t kq_gc_inval   = 0; // [TAG_GRAPH_CACHE]
    llama_qsa_pooled_store::stats_t kq_plan0;
    if (kqprof) {
        kq_plan0      = llama_qsa_store_stats(mctx);
        kq_t_start_us = ggml_time_us();
        kq_t_start_ms = kqvis_unix_ms();
        const int64_t last_us = g_kqvis_last_us.load();
        if (last_us > 0 && kq_t_start_us - last_us > 1500000) {
            g_kqvis_epoch.fetch_add(1); // every context was idle > 1.5 s: close all open windows at their next graph
        }
    }

    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // [TAG_GRAPH_CACHE] the backend samplers keep their input tensors outside the graph result (the uniforms of the
    // last built graph, AUDIT-graph-cache-inputs.md), and pipeline parallelism rotates the input copies: neither
    // can switch between cached graphs, both keep the single previous graph
    const bool use_graph_cache = graph_cache_k > 0 && !graph_reuse_disable && !cparams.pipeline_parallel && sampling.samplers.empty();
    if (graph_cache_k > 0 && !use_graph_cache) {
        graph_cache_tot.bypass++;
    }
    if (!use_graph_cache && graph_cache_active) {
        graph_cache_clear();
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (use_graph_cache) {
        kq_gc_inval = graph_cache_tot.inval;
        res = graph_cache_get(ubatch, gtype, mctx, ret, kq_reused);
        kq_gc_inval = graph_cache_tot.inval - kq_gc_inval;
        if (!res) {
            llama_qsa_store_publish(mctx, false); // [TAG_QSA_POOLED_CACHE]
            return nullptr;
        }
        if (kq_reused) {
            n_reused++;
        }
        if (prof) {
            const int64_t t_now_us = ggml_time_us();
            sched_prof.reused     = kq_reused;
            sched_prof.t_build_us = t_now_us - t_prof_us;
            sched_prof.t_alloc_us = 0; // included in t_build_us on this path
            t_prof_us = t_now_us;
        }
    } else if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
        kq_reused = true;

        if (prof) {
            // reuse path: build = mctx->apply() + graph_params() + the can_reuse() check (+ the
            // pipeline-parallel synchronize); no graph is built and nothing is allocated
            const int64_t t_now_us = ggml_time_us();
            sched_prof.reused     = true;
            sched_prof.t_build_us = t_now_us - t_prof_us;
            sched_prof.t_alloc_us = 0;
            t_prof_us = t_now_us;
        }
    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            llama_qsa_store_publish(mctx, false); // [TAG_QSA_POOLED_CACHE]
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (prof) {
            // build path: build = mctx->apply() + graph_params() + the failed can_reuse() check +
            // ggml_backend_sched_reset() + model.build_graph()
            const int64_t t_now_us = ggml_time_us();
            sched_prof.reused     = false;
            sched_prof.t_build_us = t_now_us - t_prof_us;
            t_prof_us = t_now_us;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            llama_qsa_store_publish(mctx, false); // [TAG_QSA_POOLED_CACHE]
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        if (prof) {
            const int64_t t_now_us = ggml_time_us();
            sched_prof.t_alloc_us = t_now_us - t_prof_us;
            t_prof_us = t_now_us;
        }
    }

    // [TAG_QSA_INPUTS_INC] the QSA input mirrors are device tensors written in place by set_inputs (a host-visible buffer
    // is a plain memcpy): the previous graph of this context must be done reading them
    if (llama_qsa_inputs_inc()) {
        ggml_backend_sched_synchronize(sched.get());
    }

    // [TAG_KQVIS_PROF]
    int64_t kq_t_inputs_us = 0;
    if (kqprof) {
        llama_kqvis_prof_fill_tls() = llama_kqvis_prof_fill();
        kq_t_inputs_us = ggml_time_us();
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // [TAG_GRAPH_CACHE] CHECK=3: the cached graph's inputs get a sentinel first (see graph_cache_check_exec_pre)
        const bool gc_check3 = use_graph_cache && kq_reused && graph_cache_check >= 3;
        if (gc_check3) {
            ggml_backend_sched_synchronize(sched.get()); // the previous graph may still use the shared buffers
            graph_cache_poison_inputs(res->get_gf(), sched.get(), 0xA5);
        }
        // test seam LLAMA_GRAPH_CACHE_CHECK_INJECT=stale (CHECK >= 2, once): the first token's position is left
        // unrefreshed (its bytes from before set_inputs are put back), as by a set_input that skips a region
        ggml_tensor * gc_stale = nullptr;
        std::vector<uint8_t> gc_stale_old;
        if (use_graph_cache && kq_reused && graph_cache_check >= 2 && graph_cache_inject == "stale") {
            for (const auto & inp : res->inputs) {
                if (auto * ip = dynamic_cast<llm_graph_input_pos *>(inp.get()); ip != nullptr && ip->pos && ip->pos->data) {
                    gc_stale = ip->pos;
                    graph_cache_read(gc_stale, gc_stale_old);
                }
            }
        }

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        if (gc_stale != nullptr) {
            ggml_backend_tensor_set(gc_stale, gc_stale_old.data(), 0, ggml_type_size(gc_stale->type));
            LLAMA_LOG_WARN("graph-cache: inject stale: %s left partly unrefreshed\n", gc_stale->name);
            graph_cache_inject.clear();
        }

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    if (prof) {
        const int64_t t_now_us = ggml_time_us();
        sched_prof.t_inputs_us = t_now_us - t_prof_us;
        t_prof_us = t_now_us;
    }

    int64_t kq_t_compute_us = 0;
    if (kqprof) {
        ggml_backend_sched_set_input_prof(sched.get(), true);
        kq_t_compute_us = ggml_time_us();
    }

    // [TAG_GRAPH_CACHE] CHECK=2: execute a fresh plan first (state restored), compare after the cached one ran
    const bool gc_exec = use_graph_cache && kq_reused && graph_cache_check >= 2;
    if (gc_exec) {
        graph_cache_check_exec_pre(res, ubatch, gtype, mctx);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);

    if (gc_exec && status == GGML_STATUS_SUCCESS) {
        graph_cache_check_exec_post(res);
    }

    if (kqprof) {
        // [TAG_KQVIS_PROF] account this graph; first close the open window if the graph does not belong to it
        const int64_t t_now_us = ggml_time_us();
        const llama_kqvis_prof_fill fill = llama_kqvis_prof_fill_tls();
        int64_t in_wait_us = 0, in_copy_us = 0, kq_copy_us = 0, kq_bytes = 0;
        ggml_backend_sched_get_input_prof(sched.get(), &in_wait_us, &in_copy_us, &kq_copy_us, &kq_bytes);

        const int32_t  cls   = ubatch.n_tokens >= 64 ? 1 : 0;
        const int32_t  mode  = (fill.n_mask > 0 ? 1 : 0) | (fill.n_vis > 0 ? 2 : 0);
        const uint64_t epoch = g_kqvis_epoch.load();
        const int64_t  n_kv  = sched_prof_n_kv(mctx);

        auto & w = kqvis_win;
        if (w.graphs > 0) {
            const char * reason = w.epoch != epoch ? "idle" : w.cls != cls ? "class" : w.mode != mode ? "mode" : w.graphs >= 64 ? "full" : nullptr;
            if (reason) {
                kqvis_prof_flush(reason);
            }
        }
        if (w.graphs == 0) {
            w.cls      = cls;
            w.mode     = mode;
            w.epoch    = epoch;
            w.t0_ms    = kq_t_start_ms;
            w.n_kv_min = n_kv;
            w.n_kv_max = n_kv;
        }
        w.graphs++;
        w.reused     += kq_reused ? 1 : 0;
        if (use_graph_cache) {
            w.gc_hit   += kq_reused ? 1 : 0;
            w.gc_miss  += kq_reused ? 0 : 1;
            w.gc_check += (kq_reused && graph_cache_check) ? 1 : 0;
            w.gc_inval += (int32_t) kq_gc_inval;
        }
        w.tokens     += ubatch.n_tokens;
        w.n_kv_min    = std::min(w.n_kv_min, n_kv);
        w.n_kv_max    = std::max(w.n_kv_max, n_kv);
        w.n_kv_sum   += n_kv;
        w.build_us   += kq_t_inputs_us - kq_t_start_us;
        w.inputs_us  += kq_t_compute_us - kq_t_inputs_us;
        w.fill_us    += fill.us;
        w.in_wait_us += in_wait_us;
        w.in_copy_us += in_copy_us;
        w.kq_copy_us += kq_copy_us;
        w.kq_bytes   += kq_bytes;
        w.compute_us += t_now_us - kq_t_compute_us;
        w.t1_ms       = kqvis_unix_ms();
        {
            const auto p1 = llama_qsa_store_stats(mctx);
            w.plan_us    += (int64_t) (p1.t_plan_us  - kq_plan0.t_plan_us);
            w.n_plan     += (int64_t) (p1.n_plan     - kq_plan0.n_plan);
            w.n_plan_inc += (int64_t) (p1.n_plan_inc - kq_plan0.n_plan_inc);
            w.n_cen_inc  += (int64_t) (p1.n_cen_inc  - kq_plan0.n_cen_inc);
            w.n_cen_full += (int64_t) (p1.n_cen_full - kq_plan0.n_cen_full);
            for (int k = 0; k < 5; ++k) {
                w.n_cen_fb[k] += (int64_t) (p1.n_cen_fb[k] - kq_plan0.n_cen_fb[k]);
            }
            w.qin_us    += (int64_t) (p1.t_inputs_us - kq_plan0.t_inputs_us);
            w.mir_bytes += (int64_t) (p1.mir_bytes   - kq_plan0.mir_bytes);
            w.mir_runs  += (int64_t) (p1.mir_runs    - kq_plan0.mir_runs);
            w.mir_full  += (int64_t) (p1.mir_full    - kq_plan0.mir_full);
        }
        g_kqvis_last_us.store(t_now_us);
    }

    // [TAG_QSA_POOLED_CACHE] the graph has been submitted in its build order (raw indexer write, then the
    // pooling of the dirty blocks into the store, then the selection that reads it): its pooled-key stamps
    // become valid now, or its streams are invalidated if it failed. A later graph reading the store is
    // queued after this one.
    llama_qsa_store_publish(mctx, status == GGML_STATUS_SUCCESS);

    if (prof) {
        // graph_compute() is ggml_backend_sched_graph_compute_async(): on an asynchronous backend (Vulkan)
        // this is the host-side split walk + command recording + vkQueueSubmit time only - the GPU wait
        // lands in synchronize() and is printed there as "[sched-prof] sync ... wait="
        sched_prof.t_compute_us = ggml_time_us() - t_prof_us;
        sched_prof.n_kv         = sched_prof_n_kv(mctx);
    }

    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

// one line per ubatch; t_outputs_start_us is the time process_ubatch() returned, so "outputs" brackets the
// ggml_backend_tensor_get_async() calls for logits / embeddings / h_nextn / backend-sampler rows and "total"
// runs from the start of process_ubatch() to the end of those queued reads
void llama_context::sched_prof_print_ubatch(const char * ctx_name, uint32_t n_tokens, int64_t t_outputs_start_us) const {
    const int64_t t_now_us = ggml_time_us();

    fprintf(stderr, "[sched-prof] ubatch ctx=%s n_tokens=%u n_kv=%" PRId64 " reuse=%d build=%.3fms alloc=%.3fms inputs=%.3fms compute=%.3fms outputs=%.3fms total=%.3fms\n",
            ctx_name, n_tokens, sched_prof.n_kv, sched_prof.reused ? 1 : 0,
            sched_prof.t_build_us/1000.0, sched_prof.t_alloc_us/1000.0, sched_prof.t_inputs_us/1000.0, sched_prof.t_compute_us/1000.0,
            (t_now_us - t_outputs_start_us)/1000.0, (t_now_us - sched_prof.t_start_us)/1000.0);
}

int llama_context::encode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_embd = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    balloc->set_rs_rollback_depth(0);
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    int64_t t_prof_outputs_us = 0;
    if (sched_prof_enabled()) {
        t_prof_outputs_us = ggml_time_us();
    }

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    if (sched_prof_enabled()) {
        sched_prof_print_ubatch("enc", n_tokens, t_prof_outputs_us);
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

template<typename T>
static void copy_tensor_async_rows(
    const std::vector<ggml_tensor *> & tensors,
    const buffer_view<T> & dst,
    size_t stride,
    uint32_t row_offset,
    ggml_backend_sched_t sched,
    std::vector<uint32_t> * counts = nullptr) {
    if (!dst.has_data()) {
        return;
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto * tensor = tensors[i];
        if (tensor == nullptr) {
            continue;
        }

        const uint32_t row = row_offset + i;
        const size_t n_elements = ggml_nelements(tensor);
        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampling tensor must be contiguous for async copy");
        GGML_ASSERT(n_elements <= stride);
        GGML_ASSERT((size_t) row * stride + n_elements <= dst.size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        T * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        if (counts) {
            GGML_ASSERT(row < counts->size());
            (*counts)[row] = n_elements;
        }
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    const bool prof = sched_prof_enabled();

    // pre = batch init + memory init_batch (slot search) + output_reserve, ubatches = the ubatch loop,
    // post = output-id mapping; the per-ubatch lines are printed inside the loop
    int64_t t_prof_decode_us = 0;
    int64_t t_prof_loop0_us  = 0;
    int64_t t_prof_loop1_us  = 0;
    int32_t n_prof_ubatch    = 0;
    if (prof) {
        t_prof_decode_us = ggml_time_us();
    }
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    const uint32_t rs_rollback_depth_cur = std::min(rs_rollback_depth, cparams.n_rs_seq);
    rs_rollback_depth = 0;

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const bool    mtp_embd = cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP && batch_inp.embd;
    const int64_t n_embd  = mtp_embd ? hparams.n_embd_out() : hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // embedding contexts output every token even when batch.logits is not set
    if (has_samplers && (output_all || batch_inp.logits)) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (!output_all && batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    continue;
                }

                seq_output_count[seq_id]++;
                auto sampler = sampling.samplers.find(seq_id);
                if (sampler != sampling.samplers.end() &&
                        seq_output_count[seq_id] > (int32_t) cparams.n_outputs_max_per_seq) {
                    LLAMA_LOG_ERROR("%s: backend sampling supports at most %u outputs per sequence "
                            "(seq_id %d had %d)\n", __func__, cparams.n_outputs_max_per_seq,
                            seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    balloc->set_rs_rollback_depth(rs_rollback_depth_cur);
    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    // [TAG_MULTISLOT] round 2: an embedding batch of an M-RoPE indexer model (an mtmd image chunk) switches the default
    // complete-history fallback off for the process (llama_qsa_history_dense)
    if (batch_inp.token == nullptr && batch_inp.embd != nullptr && model.hparams.n_pos_per_embd() > 1 &&
        model.hparams.indexer_head_size > 0) {
        llama_qsa_note_mm_positions();
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    // TODO: this clear of the buffer can easily be forgotten - need something better
    // sync first so any in-flight async copies into embd_seq complete before it is freed
    if (!embd_seq.empty()) {
        synchronize();
    }
    embd_seq.clear();

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    // start a new sampling transaction for this logical batch
    for (const auto & entry : sampling.samplers) {
        llama_sampler_backend_begin(entry.second);
    }

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    if (prof) {
        t_prof_loop0_us = ggml_time_us();
    }

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        int64_t t_prof_outputs_us = 0;
        if (prof) {
            t_prof_outputs_us = ggml_time_us();
        }

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
            }
        }

        if (has_samplers) {
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_rows(res->t_sampled,        sampling.sampled,    1,      n_outputs_prev, sched.get());
            copy_tensor_async_rows(res->t_sampled_logits, sampling.logits,     stride, n_outputs_prev, sched.get(), &sampling.logits_count);
            copy_tensor_async_rows(res->t_sampled_probs,  sampling.probs,      stride, n_outputs_prev, sched.get(), &sampling.probs_count);
            copy_tensor_async_rows(res->t_candidates,     sampling.candidates, stride, n_outputs_prev, sched.get(), &sampling.candidates_count);
        }

        if (prof) {
            n_prof_ubatch++;
            sched_prof_print_ubatch(sched_prof_ctx_name(cparams.ctx_type), ubatch.n_tokens, t_prof_outputs_us);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    if (prof) {
        t_prof_loop1_us = ggml_time_us();
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    if (prof) {
        const int64_t t_now_us = ggml_time_us();
        fprintf(stderr, "[sched-prof] decode ctx=%s n_tokens=%u n_ubatch=%d pre=%.3fms ubatches=%.3fms post=%.3fms total=%.3fms\n",
                sched_prof_ctx_name(cparams.ctx_type), n_tokens_all, n_prof_ubatch,
                (t_prof_loop0_us - t_prof_decode_us)/1000.0, (t_prof_loop1_us - t_prof_loop0_us)/1000.0,
                (t_now_us - t_prof_loop1_us)/1000.0, (t_now_us - t_prof_decode_us)/1000.0);
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab     = model.vocab.n_tokens();
    const uint64_t n_embd      = model.hparams.n_embd;
    const uint64_t n_embd_out  = model.hparams.n_embd_out();

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd.data[i0*n_embd_out + k], embd.data[i1*n_embd_out + k]);
            }
        }

        if (embd_nextn.size > 0) {
            for (uint64_t k = 0; k < n_embd_out; k++) {
                std::swap(embd_nextn.data[i0*n_embd_out + k], embd_nextn.data[i1*n_embd_out + k]);
            }
        }

        if (embd_layer_inp.size() > 0) {
            for (int lid = 0; lid < (int) embd_layer_inp.size(); ++lid) {
                if (embd_layer_inp[lid].size > 0) {
                    for (uint64_t k = 0; k < n_embd; ++k) {
                        std::swap(embd_layer_inp[lid].data[i0*n_embd + k], embd_layer_inp[lid].data[i1*n_embd + k]);
                    }
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint32_t res;
    if (model.arch == LLM_ARCH_KIMI_K3) {
        // the n_tokens*40 budget below is exhausted at ubatch 3840
        res = std::max<uint32_t>(n_tokens * 160, 64u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_BAILINGMOE3 ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_QWEN4EXP ||
        model.arch == LLM_ARCH_DEEPSEEK4 ||
        (model.arch == LLM_ARCH_DFLASH && model.hparams.dsv4_hc_mult > 0) ||
        model.arch == LLM_ARCH_NANBEIGE ||
        model.arch == LLM_ARCH_MINIMAX_01 ||
        model.arch == LLM_ARCH_MINIMAX_M3) {
        res = std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    } else if (model.arch == LLM_ARCH_DFLASH && model.hparams.dflash_selector_rank > 0) {
        // DFlash2's convolutions and selector are shape work rather than matmuls,
        // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
        res = std::max<uint32_t>(1024u, 12u*model.n_tensors());
    } else {
        res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
        for (const auto & lora : model.loras) {
            res += lora->get_n_nodes();
        }
    }

    uint32_t n_sampling_nodes = 0;
    uint32_t n_sampling_nodes_max = 0;
    for (const auto & [seq_id, sampler] : sampling.samplers) {
        const uint32_t n_nodes = llama_sampler_backend_n_nodes(sampler);
        n_sampling_nodes += n_nodes;
        if (cparams.n_outputs_max_per_seq > 1) {
            n_sampling_nodes_max = std::max(n_sampling_nodes_max, n_nodes);
        }
    }

    const uint32_t n_sampling_outputs_max = std::min<uint64_t>(
            std::min(n_tokens, cparams.n_outputs_max),
            (uint64_t) cparams.n_seq_max * cparams.n_outputs_max_per_seq);

    res += n_sampling_nodes;
    if (n_sampling_outputs_max > 1) {
        res += (n_sampling_outputs_max - 1) * n_sampling_nodes_max;
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void ubatch_prepare_reserve(
              llama_ubatch                            & ubatch,
              uint32_t                                  n_outputs,
        const std::map<llama_seq_id, llama_sampler *> & samplers,
              uint32_t                                  n_outputs_max_per_seq) {
    const uint32_t n_seqs       = ubatch.n_seqs;
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_seq_tokens; ++t) {
            const uint32_t i = s * n_seq_tokens + t;
            ubatch.n_seq_id[i] = 1;
            ubatch.seq_id[i] = &ubatch.seq_id_unq[s];
        }
    }

    // sequences with a sampler that fit in this ubatch
    std::vector<uint32_t> sampler_seqs;
    std::vector<bool> has_sampler(n_seqs, false);
    for (const auto & entry : samplers) {
        const llama_seq_id seq_id = entry.first;
        if (seq_id < 0 || (uint32_t) seq_id >= n_seqs) {
            continue;
        }

        sampler_seqs.push_back(seq_id);
        has_sampler[seq_id] = true;
    }

    uint32_t n_outputs_set = 0;

    const uint32_t n_outputs_per_seq = std::min(n_seq_tokens, n_outputs_max_per_seq);
    for (uint32_t s : sampler_seqs) {
        if (n_outputs_set >= n_outputs) {
            break;
        }

        for (uint32_t t = 0; t < n_outputs_per_seq && n_outputs_set < n_outputs; ++t) {
            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }

    // use sequences without samplers for any remaining outputs
    for (uint32_t t = 0; t < n_seq_tokens && n_outputs_set < n_outputs; ++t) {
        for (uint32_t s = 0; s < n_seqs && n_outputs_set < n_outputs; ++s) {
            if (has_sampler[s]) {
                continue;
            }

            ubatch.output[s * n_seq_tokens + t] = true;
            ++n_outputs_set;
        }
    }
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();
    graph_cache_clear(); // [TAG_GRAPH_CACHE]

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    balloc.set_rs_rollback_depth(cparams.n_rs_seq);
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    ubatch_prepare_reserve(ubatch, n_outputs, sampling.samplers, cparams.n_outputs_max_per_seq);

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched,
   ggml_backend_sched_t   s_other) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    if (s_other != nullptr) {
        // [TAG_GRAPH_CACHE] CHECK=2: the fresh plan on the scratch scheduler, synchronously, no dumps
        return ggml_backend_sched_graph_compute(s_other, gf);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // [TAG_QSA_DUMP_IDS] with an eval callback the scheduler computes (and calls back) synchronously, so every
    // selection of this graph has been hashed by now
    if (qsa_ids_dump) {
        qsa_ids_dump->end_graph();
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            // [TAG_MTP_QSA] elements, not blocks: ggml_element_size is the BLOCK size of a quantized type, so the old
            // size/element_size made a q8_0 view 1/32 of the range (an ON_DEVICE state of a q8_0 cache lost 31/32 of it)
            const int64_t n = winfo.size/ggml_type_size(winfo.tensor->type)*ggml_blck_size(winfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            // [TAG_MTP_QSA] elements, not blocks: ggml_element_size is the BLOCK size of a quantized type, so the old
            // size/element_size made a q8_0 view 1/32 of the range (an ON_DEVICE state of a q8_0 cache lost 31/32 of it)
            const int64_t n = rinfo.size/ggml_type_size(rinfo.tensor->type)*ggml_blck_size(rinfo.tensor->type);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
            }
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (tokens_out == nullptr) {
            const size_t n_token_max = (file.size() - file.tell()) / sizeof(llama_token);
            if (n_token_count > n_token_max) {
                LLAMA_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__, n_token_count, n_token_max);
                return 0;
            }

            *n_token_count_out = n_token_count;
            return file.tell();
        }

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_seq_write_io(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return state_seq_write_data(io, seq_id, flags);
}

size_t llama_context::state_seq_read_io(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return state_seq_read_data(io, seq_id, flags);
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        balloc->set_rs_rollback_depth(0);
        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            graph_cache_clear(); // [TAG_GRAPH_CACHE] the training path uses the scheduler directly

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_outputs_max_per_seq       =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
        /*.kv_graph_span               =*/ 0,
        /*.qsa_ratio                   =*/ -1,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if ((model->hparams.is_mla() || model->arch == LLM_ARCH_DEEPSEEK4) && params.type_k != params.type_v) {
        LLAMA_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        return nullptr;
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        (model->hparams.n_layer_nextn == 0 || model->hparams.router_layer >= 0)) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

int32_t llama_qsa_history_dense_info(const llama_context * ctx, const char ** reason) {
    if (ctx == nullptr || ctx->get_model().hparams.indexer_head_size == 0) {
        return -1;
    }
    return llama_qsa_history_dense(reason) ? 1 : 0;
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_set_rs_rollback_depth(llama_context * ctx, uint32_t depth) {
    ctx->set_rs_rollback_depth(depth);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_sampler * llama_get_sampler(const llama_context * ctx, llama_seq_id seq_id) {
    return ctx->get_sampler(seq_id);
}

llama_sampler * llama_context::get_sampler(llama_seq_id seq_id) const {
    const auto it = sampling.samplers.find(seq_id);
    return it == sampling.samplers.end() ? nullptr : it->second;
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}

void llama_kqvis_prof_add_spec_round(struct llama_context * ctx, int64_t t_draft_us, int64_t t_catchup_us,
        int64_t t_verify_us, int64_t t_host_us) {
    if (ctx != nullptr && llama_kqvis_prof_enabled()) {
        ctx->kqvis_prof_add_spec_round(t_draft_us, t_catchup_us, t_verify_us, t_host_us);
    }
}

void llama_kqvis_prof_add_catchup_pf(struct llama_context * ctx, int64_t t_catchup_us) {
    if (ctx != nullptr && llama_kqvis_prof_enabled()) {
        ctx->kqvis_prof_add_catchup_pf(t_catchup_us);
    }
}

//
// streamed sequence state [TAG_DISK_DELTA]
//

class llama_io_write_cb : public llama_io_write_i {
public:
    explicit llama_io_write_cb(const llama_state_seq_writer & w) : w(w) {}

    void write(const void * src, size_t size) override {
        w.write(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        w.write_tensor(tensor, offset, size);
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const llama_state_seq_writer & w;
    size_t size_written = 0;
};

class llama_io_read_cb : public llama_io_read_i {
public:
    explicit llama_io_read_cb(const llama_state_seq_reader & r) : r(r) {}

    void read(void * dst, size_t size) override {
        if (!r.read(dst, size)) {
            throw std::runtime_error("sequence state stream ended early");
        }
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const llama_state_seq_reader & r;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t llama_state_seq_write_cb(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags,
                                const llama_state_seq_writer & writer) {
    ctx->synchronize();

    try {
        llama_io_write_cb io(writer);
        return ctx->state_seq_write_io(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error writing sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_read_cb(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags,
                               const llama_state_seq_reader & reader, llama_state_read_status * status) {
    ctx->synchronize();

    llama_io_read_cb io(reader);
    size_t n_read = 0;
    try {
        n_read = ctx->state_seq_read_io(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error reading sequence state: %s\n", __func__, err.what());
        n_read = 0;
    }
    if (status) {
        *status = n_read > 0      ? LLAMA_STATE_READ_OK
                : io.status != 0  ? (llama_state_read_status) io.status
                                  : LLAMA_STATE_READ_OTHER;
    }
    return n_read;
}

std::string llama_state_seq_descriptor(const llama_context * ctx) {
    const auto & cp = ctx->get_cparams();
    const auto & hp = ctx->get_model().hparams;
    // stream flags: PARTIAL_ONLY skips the full-only sections (marked "(full-only)" below), ON_DEVICE is a snapshot
    //   location, never a disk stream
    // [TAG_MULTISLOT] per-seq state is stream-count agnostic (llama_kv_cache::state_read_sinfo): the slot count and
    //   the KV layout (n_seq_max, kv_unified, n_stream) do not change the per-sequence bytes, so they print the values
    //   of a single-slot unified context (1 / 1 / 1) - the prod np1 --kv-unified descriptor stays byte-identical, and
    //   np1 / np2 per-stream / np2 unified agree whenever the per-sequence capacity (n_ctx_seq, kv size) does
    std::string res = format("abi=%d;flags=partial-only:1;ctx_type=%d;other=%d;causal=%d;n_ctx_seq=%u;n_seq_max=%u;"
                                    "unified=%d;flash_attn=%d;", LLAMA_STATE_ABI_REV, (int) cp.ctx_type,
                                    (int) (cp.ctx_other != nullptr), (int) cp.causal_attn, cp.n_ctx_seq, 1u,
                                    1, (int) cp.flash_attn);
    res += format("rope{type=%d,sections=%d/%d/%d/%d,base=%a,scale=%a,orig=%u,ext=%a,attn=%a,beta=%a/%a,rot=[",
                         (int) hp.rope_type, hp.rope_sections[0], hp.rope_sections[1], hp.rope_sections[2],
                         hp.rope_sections[3], cp.rope_freq_base, cp.rope_freq_scale, cp.n_ctx_orig_yarn,
                         cp.yarn_ext_factor, cp.yarn_attn_factor, cp.yarn_beta_fast, cp.yarn_beta_slow);
    std::string qsa;
    for (uint32_t il = 0; il < hp.n_layer(); il++) {
        res += format("%u;", hp.n_rot(il));
        qsa += format("%u;", llama_hparams_qsa_ratio(hp, cp.qsa_ratio, il));
    }
    res += "]};qsa=[" + qsa + "];mem=";
    const llama_memory_t mem = ctx->get_memory();
    res += mem ? mem->state_descriptor() : std::string("none");
    return res;
}
