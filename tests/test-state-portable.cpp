// [TAG_MULTISLOT] per-seq state is stream-count agnostic.
//
// A slot / sequence state saved by an np1 server (n_stream 1) used to fail in an np2 per-stream server with
// "llama_state_seq_load_file: error loading sequence state file: n_stream mismatch". The CPU qwen4exp fixture of
// test-mtp-catchup (hybrid-idx target: attention KV + recurrent + QSA indexer K; MTP draft: attention KV + indexer K
// under LLAMA_MTP_QSA=1) checks, for the target (llama_state_seq_save_file / load_file) + draft (seq data) + carry:
//  1. np1 -> np2 per-stream into seq 0 and into seq 1 (seq 0 busy with another prompt), np1 -> np2 unified (seq 1),
//     np2 per-stream (seq 1) -> np1, np2 unified -> np2 per-stream, np2 per-stream -> np2 unified: the continuation
//     (target logits of every row, draft tokens + draft logits, carry, and the canonical state streams) is BYTE-equal
//     to the np1 continuation, and the "n_stream mismatch" error never occurs;
//  2. a whole-context state (llama_state_get_data) stays strict: np1 -> np2 per-stream is refused ("n_stream
//     mismatch"), np2 -> np2 still loads;
//  3. a per-seq blob with two non-empty stream blocks is rejected (np2 and np1 readers), the sequence left empty;
//  4. the state descriptor (disk-cache envelope / key part) is equal for np1 / np2 per-stream / np2 unified of one
//     per-slot size, and the np1 unified descriptor prints the single-slot values.

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"
#include "server-disk-cache.h" // [TAG_MULTISLOT] the disk-cache descriptor + adopt structure

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-ext.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

static const uint32_t N_VOCAB   = 128;
static const uint32_t N_CTX_SEQ = 512;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-1f);
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (auto & x : tmp) { x = dis(gen); }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (auto & x : tmp) { x = ggml_fp32_to_fp16(dis(gen)); }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// the test-mtp-catchup qwen4exp fixture (hc 4, QSA ratios 4, MoE, 1 MTP block)
static gguf_context_ptr make_meta(uint32_t n_ctx_train) {
    const llm_arch arch = LLM_ARCH_QWEN4EXP;
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_embd = 256, n_head = 2, n_ff = 384, n_layer_trunk = 2, n_nextn = 1;
    const uint32_t n_layer_all = n_layer_trunk + n_nextn;
    const uint32_t n_embd_head = n_embd / n_head;
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              n_ctx_train);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer_all);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,        n_nextn);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,     uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,     std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,      uint32_t(4));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK,   uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,   std::vector<uint32_t>(n_layer_all, 4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        n_ff);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);
    ms.add_kv(LLM_KV_EXPERT_COUNT,                uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,           uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,         uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,          uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,        1.0f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,             uint32_t(2));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");
    return ret;
}

static int n_fail = 0;
static int n_ok   = 0;
static int n_mismatch_logs = 0; // occurrences of the gate's error string

static void check(bool ok, const std::string & what) {
    if (!ok) {
        printf("FAIL %s\n", what.c_str());
        n_fail++;
    } else {
        n_ok++;
    }
}

template <typename T>
static bool eq(const std::vector<T> & a, const std::vector<T> & b) {
    return a.size() == b.size() && (a.empty() || memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

static bool silent_progress(float, void *) { return true; }

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (strstr(text, "n_stream mismatch") != nullptr) {
        n_mismatch_logs++;
    }
    if (level >= GGML_LOG_LEVEL_WARN && getenv("TEST_STATE_PORTABLE_VERBOSE")) {
        fputs(text, stderr);
    }
}

static llama_model_ptr g_model;

struct env {
    uint32_t          np;
    bool              unified;
    llama_context_ptr ctx_tgt;
    llama_context_ptr ctx_dft;
    common_params_speculative params;
    common_speculative * spec = nullptr;
    ~env() { common_speculative_free(spec); }
};

static std::unique_ptr<env> make_env(uint32_t np, bool unified) {
    auto e = std::make_unique<env>();
    e->np = np;
    e->unified = unified;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = unified ? N_CTX_SEQ : N_CTX_SEQ * np; // one per-slot size everywhere
    cp.n_batch         = 128;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = np;
    cp.kv_unified      = unified;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    e->ctx_tgt.reset(llama_init_from_model(g_model.get(), cp));
    GGML_ASSERT(e->ctx_tgt && "target context failed");
    GGML_ASSERT(llama_n_ctx_seq(e->ctx_tgt.get()) == N_CTX_SEQ);
    llama_context_params cd = cp;
    cd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cd.n_rs_seq = 0;
    e->ctx_dft.reset(llama_init_from_model(g_model.get(), cd));
    GGML_ASSERT(e->ctx_dft && "MTP context failed");
    e->params.types           = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    e->params.draft.n_max     = 4;
    e->params.draft.ctx_tgt   = e->ctx_tgt.get();
    e->params.draft.ctx_dft   = e->ctx_dft.get();
    e->spec = common_speculative_init(e->params, np);
    GGML_ASSERT(e->spec && "draft-mtp init failed");
    return e;
}

static std::vector<llama_token> tokens(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<llama_token> dis(0, N_VOCAB - 1);
    std::vector<llama_token> t(n);
    for (auto & x : t) { x = dis(gen); }
    return t;
}

// target decode of seq (every row an output) + the MTP catch-up; returns the target logits
static std::vector<float> feed(env & e, llama_seq_id seq, const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(b, toks[i], pos0 + (llama_pos) i, { seq }, true);
    }
    GGML_ASSERT(llama_decode(e.ctx_tgt.get(), b) == 0);
    std::vector<float> logits(toks.size() * N_VOCAB);
    for (size_t i = 0; i < toks.size(); ++i) {
        memcpy(logits.data() + i * N_VOCAB, llama_get_logits_ith(e.ctx_tgt.get(), (int32_t) i), N_VOCAB * sizeof(float));
    }
    GGML_ASSERT(common_speculative_process(e.spec, b));
    llama_batch_free(b);
    return logits;
}

struct seq_state {
    std::vector<llama_token> toks; // the target seq file's prompt
    std::string              file; // target: llama_state_seq_save_file
    std::vector<uint8_t>     dft;  // draft: llama_state_seq_get_data_ext, flags 0
    std::vector<uint8_t>     carry;
};

static std::vector<uint8_t> seq_data(llama_context * ctx, llama_seq_id seq) {
    std::vector<uint8_t> d(llama_state_seq_get_size_ext(ctx, seq, 0));
    GGML_ASSERT(llama_state_seq_get_data_ext(ctx, d.data(), d.size(), seq, 0) == d.size());
    return d;
}

static seq_state save(env & e, llama_seq_id seq, const std::vector<llama_token> & toks, const std::string & file) {
    seq_state s;
    s.toks = toks;
    s.file = file;
    GGML_ASSERT(llama_state_seq_save_file(e.ctx_tgt.get(), file.c_str(), seq, toks.data(), toks.size()) > 0);
    s.dft = seq_data(e.ctx_dft.get(), seq);
    GGML_ASSERT(common_speculative_get_state(e.spec, seq, s.carry));
    return s;
}

static bool load(env & e, llama_seq_id seq, const seq_state & s) {
    std::vector<llama_token> toks(s.toks.size() + 8);
    size_t n_tok = 0;
    if (llama_state_seq_load_file(e.ctx_tgt.get(), s.file.c_str(), seq, toks.data(), toks.size(), &n_tok) == 0) {
        return false;
    }
    if (n_tok != s.toks.size() || memcmp(toks.data(), s.toks.data(), n_tok * sizeof(llama_token)) != 0) {
        return false;
    }
    llama_memory_seq_rm(llama_get_memory(e.ctx_dft.get()), seq, -1, -1);
    if (llama_state_seq_set_data_ext(e.ctx_dft.get(), s.dft.data(), s.dft.size(), seq, 0) == 0) {
        return false;
    }
    return common_speculative_set_state(e.spec, seq, s.carry);
}

// the continuation: 3 rounds of (draft from the last token, then feed a fixed chunk)
struct cont {
    std::vector<std::vector<llama_token>> draft_toks;
    std::vector<std::vector<float>>       draft_logits;
    std::vector<std::vector<float>>       tgt_logits;
    std::vector<std::vector<uint8_t>>     carry;
    std::vector<uint8_t>                  tgt_final, dft_final; // canonical (np1 re-save) final state streams
};

// a state stream in the np1 format: loaded into a fresh np1 context and saved again (the readers are portable)
static void canon(const std::vector<uint8_t> & tgt, const std::vector<uint8_t> & dft, std::vector<uint8_t> & tgt_out,
                  std::vector<uint8_t> & dft_out) {
    auto c = make_env(1, false);
    GGML_ASSERT(llama_state_seq_set_data_ext(c->ctx_tgt.get(), tgt.data(), tgt.size(), 0, 0) > 0);
    GGML_ASSERT(llama_state_seq_set_data_ext(c->ctx_dft.get(), dft.data(), dft.size(), 0, 0) > 0);
    tgt_out = seq_data(c->ctx_tgt.get(), 0);
    dft_out = seq_data(c->ctx_dft.get(), 0);
}

static cont run_cont(env & e, llama_seq_id seq, llama_pos n_past, llama_token id_last) {
    cont r;
    const std::vector<std::vector<llama_token>> chunks = { tokens(1, 101), tokens(5, 102), tokens(3, 103) };
    for (const auto & ch : chunks) {
        std::vector<llama_token> out;
        auto & dp = common_speculative_get_draft_params(e.spec, seq);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.n_past   = n_past;
        dp.id_last  = id_last;
        dp.prompt   = nullptr;
        dp.result   = &out;
        common_speculative_draft(e.spec);
        const float * l = llama_get_logits_ith(e.ctx_dft.get(), -1);
        r.draft_toks.push_back(out);
        r.draft_logits.emplace_back(l ? std::vector<float>(l, l + N_VOCAB) : std::vector<float>());
        llama_memory_seq_rm(llama_get_memory(e.ctx_dft.get()), seq, n_past, -1);
        r.tgt_logits.push_back(feed(e, seq, ch, n_past));
        n_past += (llama_pos) ch.size();
        id_last = ch.back();
        std::vector<uint8_t> c;
        GGML_ASSERT(common_speculative_get_state(e.spec, seq, c));
        r.carry.push_back(c);
    }
    canon(seq_data(e.ctx_tgt.get(), seq), seq_data(e.ctx_dft.get(), seq), r.tgt_final, r.dft_final);
    return r;
}

static void cmp_cont(const cont & a, const cont & b, const std::string & what) {
    bool toks = true, dl = true, tl = true, ca = true;
    for (size_t i = 0; i < a.tgt_logits.size(); ++i) {
        toks = toks && eq(a.draft_toks[i], b.draft_toks[i]) && !a.draft_toks[i].empty();
        dl   = dl   && eq(a.draft_logits[i], b.draft_logits[i]) && !a.draft_logits[i].empty();
        tl   = tl   && eq(a.tgt_logits[i], b.tgt_logits[i]);
        ca   = ca   && eq(a.carry[i], b.carry[i]);
    }
    check(toks, what + ": draft proposals byte-equal");
    check(dl,   what + ": draft logits byte-equal");
    check(tl,   what + ": target logits (every row) byte-equal");
    check(ca,   what + ": speculative carry byte-equal");
    check(eq(a.tgt_final, b.tgt_final), what + ": final target state stream (canonical) byte-equal");
    check(eq(a.dft_final, b.dft_final), what + ": final draft state stream (canonical) byte-equal");
}

// two seq blobs of the same draft prompt in stream 0 / stream 1 of an np2 per-stream context -> one blob whose
// attention section holds BOTH blocks non-empty (the indexer section kept as written for stream 0)
static std::vector<uint8_t> two_block_body(const std::vector<uint8_t> & b0, const std::vector<uint8_t> & b1);

// llama_state_seq_get_data blobs open with (magic, source seq id); the memory sections follow
static std::vector<uint8_t> two_block_blob(const std::vector<uint8_t> & b0, const std::vector<uint8_t> & b1) {
    if (b0.size() < 8 || b1.size() < 8) {
        return {};
    }
    auto body = two_block_body(std::vector<uint8_t>(b0.begin() + 8, b0.end()), std::vector<uint8_t>(b1.begin() + 8, b1.end()));
    if (body.empty()) {
        return {};
    }
    body.insert(body.begin(), b0.begin(), b0.begin() + 8);
    return body;
}

static std::vector<uint8_t> two_block_body(const std::vector<uint8_t> & b0, const std::vector<uint8_t> & b1) {
    // b0 = [2][c][A][0] [2][c][I][0], b1 = [2][0][c][A] [2][0][c][I]
    if (b0.size() != b1.size() || b0.size() < 16) {
        return {};
    }
    uint32_t c = 0;
    memcpy(&c, b0.data() + 4, 4);
    // the cell metadata names the sequence, so A differs between the two blobs: the section boundary is found from the
    //   framing instead - both blobs' next section (if any) starts at 12 + |A|: b0 "[2][c]", b1 "[2][0]"
    for (size_t a = 1; 12 + a <= b0.size(); ++a) {
        uint32_t z = 1;
        memcpy(&z, b0.data() + 8 + a, 4);
        bool end_ok = 12 + a == b0.size();
        if (!end_ok && 20 + a <= b0.size()) {
            uint32_t s0 = 0, c0 = 1, s1 = 0, z1 = 1, c1 = 0;
            memcpy(&s0, b0.data() + 12 + a, 4);
            memcpy(&c0, b0.data() + 16 + a, 4);
            memcpy(&s1, b1.data() + 12 + a, 4);
            memcpy(&z1, b1.data() + 16 + a, 4);
            if (24 + a <= b1.size()) {
                memcpy(&c1, b1.data() + 20 + a, 4);
            }
            end_ok = s0 == 2 && c0 == c && s1 == 2 && z1 == 0 && c1 == c;
        }
        if (z == 0 && end_ok) {
            std::vector<uint8_t> out(b0.begin(), b0.begin() + 8 + a);           // [2][c][A]
            out.insert(out.end(), b0.begin() + 4, b0.begin() + 8 + a);           // [c][A] again: a 2nd non-empty block
            out.insert(out.end(), b0.begin() + 12 + a, b0.end());                // the indexer section (if any)
            return out;
        }
    }
    return {};
}

int main() {
    llama_log_set(log_cb, nullptr);
    common_log_set_verbosity_thold(-1);
    llama_backend_init();

    gguf_context_ptr meta = make_meta(4096);
    {
        llama_model_params mp = llama_model_default_params();
        mp.progress_callback = silent_progress;
        mp.load_mtp          = true;
        ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
        mp.devices = devs;
        size_t seed = 31;
        g_model.reset(llama_model_init_from_user(meta.get(), set_tensor_data, &seed, mp));
        GGML_ASSERT(g_model && "model init failed");
    }
    const char * mode = getenv("LLAMA_MTP_QSA");
    printf("mode: %s MTP draft\n", mode && atoi(mode) ? "sparse (LLAMA_MTP_QSA=1)" : "dense");

    const std::string dir = "test-state-portable-" + std::to_string((long) getpid());
    const auto P     = tokens(70, 7);   // the saved prompt (crosses a ubatch and several QSA blocks)
    const auto other = tokens(45, 8);   // seq 0's own prompt in the busy np2 case
    const llama_pos n_past = (llama_pos) P.size();

    // the np1 reference: prompt, save, continue
    auto ref = make_env(1, false);
    feed(*ref, 0, P, 0);
    const seq_state s_np1 = save(*ref, 0, P, dir + "-np1.bin");
    const cont c_ref = run_cont(*ref, 0, n_past, P.back());

    // 4. descriptors: one per-slot size, any slot count / KV layout
    {
        auto a = make_env(2, false);
        auto b = make_env(2, true);
        auto u = make_env(1, true);
        for (int t = 0; t < 2; ++t) {
            auto d = [&](env & e) { return llama_state_seq_descriptor(t ? e.ctx_dft.get() : e.ctx_tgt.get()); };
            const char * n = t ? "draft" : "target";
            check(d(*ref) == d(*a) && d(*ref) == d(*b) && d(*ref) == d(*u),
                  std::string("descriptor: ") + n + " equal for np1 / np2 per-stream / np2 unified / np1 unified");
            const std::string du = d(*u);
            check(du.find("n_seq_max=1;unified=1;") != std::string::npos && du.find("n_stream=1,n_seq_max=1,") != std::string::npos,
                  std::string("descriptor: ") + n + " prints the single-slot values (np1 --kv-unified unchanged)");
        }
        // the disk cache: the state descriptor hash (key v2 component + per-entry envelope) and the adopt structure
        //   (the server passes the per-slot context, llama_n_ctx_seq) agree across the slot count / KV layout
        const std::string carry = "draft-mtp/" + common_speculative_state_schema(ref->spec);
        const auto sd = [&](env & e) { return server_cache_state_desc(e.ctx_tgt.get(), e.ctx_dft.get(), carry); };
        check(sd(*ref) == sd(*a) && sd(*ref) == sd(*b), "disk cache: state descriptor hash equal np1 / np2 per-stream / np2 unified");
        common_params p;
        const auto st = [&](env & e) {
            return server_disk_cache_fingerprint_structure(server_disk_cache_fingerprint(p, (int32_t) llama_n_ctx_seq(e.ctx_tgt.get())));
        };
        check(!st(*ref).empty() && st(*ref) == st(*a) && st(*ref) == st(*b),
              "disk cache: adopt structure (per-slot n_ctx) equal np1 / np2 per-stream / np2 unified");
        // the server's INFO copy of the context-creation qsa_history_dense line reads this (the fixture has a QSA indexer)
        const char * hd_reason = nullptr;
        check(llama_qsa_history_dense_info(ref->ctx_tgt.get(), &hd_reason) >= 0 && hd_reason != nullptr,
              "qsa_history_dense_info: defined for a QSA-indexer context, with a reason");
        check(llama_n_ctx(a->ctx_tgt.get()) != llama_n_ctx(ref->ctx_tgt.get()),
              "disk cache: (the np2 per-stream total n_ctx differs - the per-slot value is what makes them equal)");
    }

    const int mism0 = n_mismatch_logs;

    // 1a. np1 -> np2 per-stream, seq 0
    {
        auto e = make_env(2, false);
        check(load(*e, 0, s_np1), "np1 -> np2 per-stream seq 0: load");
        cmp_cont(c_ref, run_cont(*e, 0, n_past, P.back()), "np1 -> np2 per-stream seq 0");
    }
    // 1b. np1 -> np2 per-stream, seq 1 (seq 0 busy); then the reverse: that seq 1 saved at np2 -> np1, -> np2 unified
    seq_state s_np2_s1;
    {
        auto e = make_env(2, false);
        feed(*e, 0, other, 0);
        check(load(*e, 1, s_np1), "np1 -> np2 per-stream seq 1: load");
        s_np2_s1 = save(*e, 1, P, dir + "-np2s1.bin");
        cmp_cont(c_ref, run_cont(*e, 1, n_past, P.back()), "np1 -> np2 per-stream seq 1 (seq 0 busy)");
        std::vector<uint8_t> t0 = seq_data(e->ctx_tgt.get(), 0);
        check(!t0.empty(), "np2 per-stream: seq 0 untouched by the seq 1 restore");
    }
    {
        auto e = make_env(1, false);
        check(load(*e, 0, s_np2_s1), "np2 per-stream seq 1 -> np1: load");
        cmp_cont(c_ref, run_cont(*e, 0, n_past, P.back()), "np2 per-stream seq 1 -> np1");
    }
    seq_state s_uni;
    {
        auto e = make_env(2, true);
        check(load(*e, 1, s_np1), "np1 -> np2 unified seq 1: load");
        s_uni = save(*e, 1, P, dir + "-uni.bin");
        cmp_cont(c_ref, run_cont(*e, 1, n_past, P.back()), "np1 -> np2 unified seq 1");
    }
    {
        auto e = make_env(2, true);
        check(load(*e, 0, s_np2_s1), "np2 per-stream seq 1 -> np2 unified seq 0: load");
        cmp_cont(c_ref, run_cont(*e, 0, n_past, P.back()), "np2 per-stream -> np2 unified");
    }
    {
        auto e = make_env(2, false);
        check(load(*e, 1, s_uni), "np2 unified -> np2 per-stream seq 1: load");
        cmp_cont(c_ref, run_cont(*e, 1, n_past, P.back()), "np2 unified -> np2 per-stream");
    }
    check(n_mismatch_logs == mism0, "the gate's \"n_stream mismatch\" error never occurs on a per-seq load");

    // 2. whole-context state stays strict
    {
        auto a = make_env(1, false);
        feed(*a, 0, P, 0);
        std::vector<uint8_t> whole(llama_state_get_size(a->ctx_tgt.get()));
        whole.resize(llama_state_get_data(a->ctx_tgt.get(), whole.data(), whole.size()));
        auto b = make_env(2, false);
        const int m = n_mismatch_logs;
        check(llama_state_set_data(b->ctx_tgt.get(), whole.data(), whole.size()) == 0,
              "whole-context: np1 -> np2 per-stream refused");
        check(n_mismatch_logs > m, "whole-context: refused with n_stream mismatch");
        auto c = make_env(2, false);
        feed(*c, 1, P, 0);
        std::vector<uint8_t> w2(llama_state_get_size(c->ctx_tgt.get()));
        w2.resize(llama_state_get_data(c->ctx_tgt.get(), w2.data(), w2.size()));
        check(llama_state_set_data(b->ctx_tgt.get(), w2.data(), w2.size()) > 0, "whole-context: np2 -> np2 loads");
        check(llama_memory_seq_pos_max(llama_get_memory(b->ctx_tgt.get()), 1) == n_past - 1,
              "whole-context: np2 -> np2 restores seq 1");
    }

    // 3. a per-seq blob with two non-empty stream blocks is rejected
    {
        auto e = make_env(2, false);
        feed(*e, 0, P, 0);
        feed(*e, 1, P, 0);
        const auto bad = two_block_blob(seq_data(e->ctx_dft.get(), 0), seq_data(e->ctx_dft.get(), 1));
        check(!bad.empty(), "corrupted blob: built (two non-empty attention blocks)");
        if (!bad.empty()) {
            auto f = make_env(2, false);
            check(llama_state_seq_set_data_ext(f->ctx_dft.get(), bad.data(), bad.size(), 1, 0) == 0,
                  "corrupted blob: rejected by an np2 per-stream reader");
            check(llama_memory_seq_pos_max(llama_get_memory(f->ctx_dft.get()), 1) == -1,
                  "corrupted blob: the sequence is left empty");
            auto g = make_env(1, false);
            check(llama_state_seq_set_data_ext(g->ctx_dft.get(), bad.data(), bad.size(), 0, 0) == 0,
                  "corrupted blob: rejected by an np1 reader");
        }
    }

    for (const char * sfx : { "-np1.bin", "-np2s1.bin", "-uni.bin" }) {
        std::remove((dir + sfx).c_str());
    }
    g_model.reset();
    llama_backend_free();

    if (n_fail) {
        printf("test-state-portable: %d failures (%d checks ok)\n", n_fail, n_ok);
        return 1;
    }
    printf("test-state-portable: OK (%d checks)\n", n_ok);
    return 0;
}
