// llama-mtp-replay: teacher-forced crossed replay with direct logit capture - separates H3 (int8 prompt processing changes the
// KV/GDN state + the MTP carrier, so the draft agrees less) from H5 (verify-batch numerics != plain decode) and measures the
// draft-target disagreement per path (piagent docs/verification/2026-10-02-counsel-astra-int8-acceptance.md §2 + §5). The
// int8 knobs (GGML_VK_DISABLE_MMQ_CM1, GGML_VK_MMQ_INT8_MIN_N / _OPS) are read at Vulkan device init: one process = one mode,
// prefill mode P x verify mode V is crossed through state files.
//   prefill: --prompt-text F [--chat] | --prompt-ids F, --state-out S [--gen G [--gen-out F]] - the prompt in -b chunks (target
//     decode + draft-mtp catch-up, the server's order); saves the target seq state, the MTP draft KV, the carrier pending_h and
//     the last-position logits. --gen G: G greedy plain (n = 1) decodes from the live state = an exact-id continuation.
//   replay: (that state and/or --state-in S ...) x (--cont TEXT | --cont-ids IDS ...) x --n-list 1,6,30 - restore it all,
//     teacher-force the continuation in batches of n with logits on every row; at every row j run the MTP draft from what the
//     server holds there (draft KV < N+j, carrier h_{N+j-1}, id_last = c_j): D = --spec-draft-n-max greedy steps by an own
//     loop = draft-mtp's single-head draft(), checked at row 0 against common_speculative_draft (draft_check).
//   compare: --compare A B - two state files (carrier L2, last-position logits KL / top-1 / margin, blob hashes) or two
//     --dump-logits files (per-row full-vocab KL, top-1 agreement, mean |d top-1 margin|).
// --out F.jsonl: a "row" line per position, a "summary" line per (state, cont, n); stdout: the summary table. restore_check:
// the gen continuation replayed at n = 1 from the restored state must reproduce the live logits bit for bit. Acceptance-
// equivalent length at row j: the leading draft tokens with top-10 confidence >= --spec-draft-p-min (the draft's p_min stop)
// equal to the target argmax of rows j, j+1, ...; censored (left out of acc / acclen) when a teacher-forced token is not the
// previous accepted draft token (that target row is off the draft's path).
// GATE INSTRUMENT (v49, Codex astra r2 Major 1): --compare is bytewise as well - state files: carrier_bytes_equal (the whole
// blob incl. its tag), logits_bytes_equal, tgt_bytes_equal / dft_bytes_equal (memcmp of the full blobs), nan_mismatch,
// bytes_equal = all of them + same prompt / n_past / last token; logit dumps: bytes_equal_rows / bytes_equal; two snapshot
// files: first differing line. last_maxdiff / maxdiff are NaN-safe (a NaN on one side only -> null = not equal).
// --snapshot-every-ubatch F: a JSON line after every draft catch-up (prefill -b chunk = ubatch at -b == -ub; replay before /
// after the draft side), after every restore, every draft chain (all D steps' logits + nextn embeddings hashed) with the
// draft-KV rollback after it, the server-path draft at row 0 and every --gen decode: the draft seq state hash, the carrier
// blob hash, the target logits hash of the rows (full vocabulary, bytewise). Equal files = equal draft state at every step.
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "speculative.h"

#include "../../src/llama-ext.h" // staging API: llama_get_embeddings_nextn_ith (as common/speculative.cpp)

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using tokens = std::vector<llama_token>;

static const char STATE_MAGIC[8] = { 'M', 'T', 'P', 'R', 'S', 'T', '1', 0 };
static const char DUMP_MAGIC[8]  = { 'M', 'T', 'P', 'R', 'L', 'G', '1', 0 };
[[noreturn]] static void fail(const std::string & msg) { fprintf(stderr, "mtp-replay: %s\n", msg.c_str()); exit(1); }

static uint64_t fnv(const void * p, size_t n) { // FNV-1a 64
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h = (h ^ ((const uint8_t *) p)[i]) * 1099511628211ull; }
    return h;
}
static uint64_t hash64(const void * p, size_t n) { // FNV-style over 64-bit words (snapshots of multi-GiB blobs); deterministic
    const uint8_t * b = (const uint8_t *) p; uint64_t h = 1469598103934665603ull; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, b + i, 8); h = (h ^ w) * 1099511628211ull; h ^= h >> 31; }
    for (; i < n; ++i) { h = (h ^ b[i]) * 1099511628211ull; }
    return h ^ (uint64_t) n;
}
static uint64_t hmix(uint64_t h, uint64_t x) { return (h ^ x) * 1099511628211ull ^ ((h ^ x) >> 29); }

struct state_t {
    int32_t n_vocab = 0, n_embd = 0, n_past = 0, last_tok = -1;
    uint64_t prompt_hash = 0, tgt_hash = 0, dft_hash = 0;
    std::string mode, tag;
    std::vector<uint8_t> carrier, tgt, dft; // carrier = the common_speculative_get_state blob (4-byte tag + n_embd floats)
    std::vector<float> logits;              // target, last prompt position
    const float * h() const { return (const float *) (carrier.data() + 4); }
};

static void wr(FILE * f, const void * p, size_t n) { if (n && fwrite(p, 1, n, f) != n) { fail("write failed"); } }
static void rd(FILE * f, void * p, size_t n) { if (n && fread(p, 1, n, f) != n) { fail("read failed (truncated file?)"); } }
template <class T> static void wr_vec(FILE * f, const T * p, uint64_t cnt) { const uint64_t n = cnt * sizeof(T); wr(f, &n, 8); wr(f, p, n); }
template <class T> static void rd_vec(FILE * f, std::vector<T> & v, bool skip = false) {
    uint64_t n = 0;
    rd(f, &n, 8);
    if (skip) { if (fseek(f, (long) n, SEEK_CUR) != 0) { fail("seek failed"); } } else { v.resize(n / sizeof(T)); rd(f, v.data(), n); }
}

static std::string base_tag(const std::string & path) {
    const std::string b = path.substr(path.find_last_of('/') + 1); const size_t dot = b.find_last_of('.');
    return dot == std::string::npos || dot == 0 ? b : b.substr(0, dot);
}

static FILE * open_magic(const std::string & path, const char * magic) { // positioned after the magic, or nullptr
    FILE * f = fopen(path.c_str(), "rb"); char m[8] = { 0 };
    if (f && (fread(m, 1, 8, f) != 8 || memcmp(m, magic, 8) != 0)) { fclose(f); f = nullptr; }
    return f;
}

static void save_state(const std::string & path, const state_t & s) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fail("cannot write " + path); }
    const int32_t  h[4]  = { s.n_vocab, s.n_embd, s.n_past, s.last_tok };
    const uint64_t hs[3] = { s.prompt_hash, s.tgt_hash, s.dft_hash };
    wr(f, STATE_MAGIC, 8); wr(f, h, sizeof(h)); wr(f, hs, sizeof(hs));
    wr_vec(f, s.mode.data(), s.mode.size()); wr_vec(f, s.carrier.data(), s.carrier.size());
    wr_vec(f, s.logits.data(), s.logits.size()); wr_vec(f, s.tgt.data(), s.tgt.size()); wr_vec(f, s.dft.data(), s.dft.size());
    if (fclose(f) != 0) { fail("close failed: " + path); }
}

static state_t load_state(const std::string & path, bool blobs) {
    FILE * f = open_magic(path, STATE_MAGIC);
    if (!f) { fail("not a mtp-replay state file: " + path); }
    state_t s; int32_t h[4]; uint64_t hs[3]; std::vector<char> mode;
    rd(f, h, sizeof(h)); rd(f, hs, sizeof(hs));
    s.n_vocab = h[0]; s.n_embd = h[1]; s.n_past = h[2]; s.last_tok = h[3]; s.prompt_hash = hs[0]; s.tgt_hash = hs[1]; s.dft_hash = hs[2];
    rd_vec(f, mode); rd_vec(f, s.carrier); rd_vec(f, s.logits); rd_vec(f, s.tgt, !blobs); rd_vec(f, s.dft, !blobs);
    fclose(f);
    if (s.carrier.size() != 4 + (size_t) s.n_embd * sizeof(float) || s.logits.size() != (size_t) s.n_vocab) { fail("corrupt state file: " + path); }
    s.mode.assign(mode.begin(), mode.end()); s.tag = base_tag(path);
    return s;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fail("cannot read " + path); }
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static tokens read_ids(const std::string & path) { // integers separated by anything else (whitespace, commas, brackets)
    tokens t; std::string cur;
    for (char c : read_file(path) + " ") {
        if (isdigit((unsigned char) c)) { cur += c; } else if (!cur.empty()) { t.push_back(std::stoi(cur)); cur.clear(); }
    }
    return t;
}

// the int8 mode of THIS process, as ggml-vulkan reads it (ggml_vk_env_bool: atoi != 0)
static std::string mode_str() {
    const char * d = getenv("GGML_VK_DISABLE_MMQ_CM1"), * mn = getenv("GGML_VK_MMQ_INT8_MIN_N"), * ops = getenv("GGML_VK_MMQ_INT8_OPS");
    std::string m = (d && *d && atoi(d) != 0) ? "int8-off" : "int8-on";
    if (mn && *mn)   { m += "+min_n=" + std::string(mn); }
    if (ops && *ops) { m += "+ops=" + std::string(ops); }
    return m;
}

// ---- math; -inf entries (a vocabulary-pruned draft head scatters -inf into the pruned rows) have probability 0
static double lse(const float * x, int n) {
    const float mx = *std::max_element(x, x + n); double s = 0;
    for (int i = 0; i < n; ++i) { s += std::exp((double) x[i] - mx); }
    return mx + std::log(s);
}
static double kl(const float * p, const float * q, int n) { // KL(softmax p || softmax q) over the full vocabulary
    const double lp = lse(p, n), lq = lse(q, n);
    double s = 0;
    for (int i = 0; i < n; ++i) { if (!std::isinf(p[i])) { const double a = p[i] - lp; s += std::exp(a) * (a - (q[i] - lq)); } }
    return std::max(0.0, s);
}
static void topk(const float * x, int n, int k, std::vector<int> & ids) { // descending; ties -> lower id first
    ids.clear();
    for (int i = 0; i < n; ++i) {
        if ((int) ids.size() == k && x[i] <= x[ids.back()]) { continue; }
        ids.insert(std::upper_bound(ids.begin(), ids.end(), i, [&](int a, int b) { return x[a] > x[b]; }), i);
        if ((int) ids.size() > k) { ids.pop_back(); }
    }
}
static float margin(const float * x, const std::vector<int> & ids) { return x[ids[0]] - x[ids[1]]; }
static double sum_d(const float * x, int n) { double s = 0; for (int i = 0; i < n; ++i) { s += x[i]; } return s; }
static std::string jnum(double v) { char b[40]; snprintf(b, sizeof(b), "%.7g", v); return std::isfinite(v) ? b : "null"; }
template <class T> static std::string jarr(const std::vector<T> & v) {
    std::string s; for (const T & x : v) { s += (s.empty() ? "[" : ",") + jnum((double) x); } return s.empty() ? "[]" : s + "]";
}

// ---- the model side
struct env_t {
    llama_context * tgt = nullptr, * dft = nullptr; common_speculative * spec = nullptr;
    int n_vocab = 0, n_embd = 0, D = 0; float p_min = 0; llama_batch b1 {}; std::string mode;
    FILE * snap = nullptr; std::vector<uint8_t> sbuf; // --snapshot-every-ubatch
};

static uint64_t seq_hash(llama_context * ctx, std::vector<uint8_t> & buf) {
    buf.resize(llama_state_seq_get_size(ctx, 0));
    if (llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0) != buf.size()) { fail("snapshot: state capture failed"); }
    return hash64(buf.data(), buf.size());
}
static uint64_t logits_hash(llama_context * ctx, int i0, int m, int V) { // rows i0 .. i0+m-1 of the last decode, full vocabulary
    uint64_t h = 0;
    for (int i = i0; i < i0 + m; ++i) { h = hmix(h, hash64(llama_get_logits_ith(ctx, i), (size_t) V * sizeof(float))); }
    return h;
}
// one snapshot line: where, position, rows, the draft seq state hash (+ size), the carrier blob hash, an extra hash (target
// logits / draft chain / draft tokens), optionally the target seq state hash
static void snap(env_t & E, const char * at, const std::string & ctx_tag, int pos, int m, uint64_t extra, bool tgt = false) {
    if (!E.snap) { return; }
    std::vector<uint8_t> c;
    const uint64_t hc = common_speculative_get_state(E.spec, 0, c) ? hash64(c.data(), c.size()) : 0;
    const uint64_t hd = seq_hash(E.dft, E.sbuf); const size_t nd = E.sbuf.size();
    std::string ht = "null";
    if (tgt) { char b[24]; snprintf(b, sizeof(b), "\"%016llx\"", (unsigned long long) seq_hash(E.tgt, E.sbuf)); ht = b; }
    E.sbuf.clear(); E.sbuf.shrink_to_fit();
    fprintf(E.snap, "{\"t\":\"snap\",\"at\":\"%s\",\"ctx\":\"%s\",\"pos\":%d,\"m\":%d,\"dft_size\":%zu,\"dft\":\"%016llx\",\"carrier_size\":%zu,"
            "\"carrier\":\"%016llx\",\"extra\":\"%016llx\",\"tgt\":%s}\n", at, ctx_tag.c_str(), pos, m, nd, (unsigned long long) hd,
            c.size(), (unsigned long long) hc, (unsigned long long) extra, ht.c_str());
    fflush(E.snap);
}

static void restore(env_t & E, const state_t & s) {
    llama_memory_clear(llama_get_memory(E.tgt), true);
    llama_memory_clear(llama_get_memory(E.dft), true);
    if (llama_state_seq_set_data(E.tgt, s.tgt.data(), s.tgt.size(), 0) == 0) { fail("target state restore failed: " + s.tag); }
    if (llama_state_seq_set_data(E.dft, s.dft.data(), s.dft.size(), 0) == 0) { fail("draft state restore failed: " + s.tag); }
    common_speculative_set_state(E.spec, 0, s.carrier);
}

// = common_speculative_impl_draft_mtp::draft() for one sequence (single MTP head, own draft KV) with p_min 0: D greedy steps
// from (id_last = tok at pos, carrier h0) -> tokens, top-10 softmax confidence per step (what p_min gates), step-0 logits;
// then drops its draft-KV cells (the server's rollback of a draft)
static void draft_chain(env_t & E, llama_token tok, llama_pos pos, const float * h0, std::vector<int> & d, std::vector<float> & p10,
                        std::vector<float> & lg0, uint64_t * chain = nullptr) {
    std::vector<float> h(h0, h0 + E.n_embd); std::vector<int> ids;
    d.clear(); p10.clear();
    for (int k = 0; k < E.D; ++k) {
        E.b1.n_tokens = 1; E.b1.token[0] = tok; E.b1.pos[0] = pos + k; E.b1.n_seq_id[0] = 1; E.b1.seq_id[0][0] = 0; E.b1.logits[0] = 1;
        memcpy(E.b1.embd, h.data(), h.size() * sizeof(float));
        if (llama_decode(E.dft, E.b1) != 0) { fail("draft decode failed"); }
        const float * l = llama_get_logits_ith(E.dft, 0);
        topk(l, E.n_vocab, 10, ids);
        double s = 0;
        for (int i : ids) { s += std::exp((double) l[i] - l[ids[0]]); }
        d.push_back(tok = ids[0]); p10.push_back((float) (1.0 / s));
        if (k == 0) { lg0.assign(l, l + E.n_vocab); }
        const float * hn = llama_get_embeddings_nextn_ith(E.dft, 0);
        h.assign(hn, hn + E.n_embd);
        if (chain) { *chain = hmix(hmix(*chain, hash64(l, (size_t) E.n_vocab * sizeof(float))), hash64(hn, (size_t) E.n_embd * sizeof(float))); }
    }
    llama_memory_seq_rm(llama_get_memory(E.dft), 0, pos, -1);
}

struct row_t {
    int tok = 0, next = -1, top1 = 0, drafted = 0, acc = 0; bool cens = false, ref1 = true;
    float margin = 0, kl_ref = 0, dmargin = 0, p_tgt = 0, kl_dt = 0; std::vector<int> top, d; std::vector<float> topl, p10;
};
struct ref_t { std::vector<float> logits, margin; std::vector<int> top1; int n = 0; }; // the first n of --n-list
static void replay(env_t & E, const state_t & S, const tokens & C, const std::string & ctag, int n, ref_t & ref, FILE * out,
                   const std::string & dump_dir, int dump_rows, const std::vector<double> * gen_sums) {
    restore(E, S);
    const std::string sctx = S.tag + "/" + ctag + "/n" + std::to_string(n);
    snap(E, "restore", sctx, (int) S.n_past, 0, 0, true);
    const int L = (int) C.size(), V = E.n_vocab, n_dump = dump_rows > 0 ? std::min(dump_rows, L) : L;
    const llama_pos N = S.n_past;
    const bool is_ref = ref.logits.empty();
    if (is_ref) { ref.logits.resize((size_t) L * V); ref.margin.resize(L); ref.top1.resize(L); ref.n = n; }
    std::vector<row_t> rows(L); std::vector<uint8_t> carrier; std::vector<float> lg0;
    int draft_check = -1, restore_bad = 0;
    FILE * dump = nullptr;
    if (!dump_dir.empty()) {
        const std::string p = dump_dir + "/" + S.tag + "__" + ctag + "__n" + std::to_string(n) + "__V" + E.mode + ".logits";
        const std::string meta = "state=" + S.tag + " P=" + S.mode + " cont=" + ctag + " n=" + std::to_string(n) + " V=" + E.mode;
        const int32_t h[2] = { n_dump, V };
        if (!(dump = fopen(p.c_str(), "wb"))) { fail("cannot write " + p); }
        wr(dump, DUMP_MAGIC, 8); wr(dump, h, sizeof(h)); wr_vec(dump, meta.data(), meta.size());
    }
    llama_batch b = llama_batch_init(n, 0, 1);
    auto * mem_dft = llama_get_memory(E.dft);
    for (int b0 = 0; b0 < L; b0 += n) {
        const int m = std::min(n, L - b0);
        common_batch_clear(b);
        for (int i = 0; i < m; ++i) { common_batch_add(b, C[b0 + i], N + b0 + i, { 0 }, true); }
        if (!common_speculative_get_state(E.spec, 0, carrier)) { fail("no draft-mtp carrier"); }
        if (llama_decode(E.tgt, b) != 0) { fail("target decode failed at row " + std::to_string(b0)); }
        if (!common_speculative_process(E.spec, b)) { fail("draft-mtp catch-up failed"); }
        snap(E, "catchup", sctx, (int) (N + b0), m, E.snap ? logits_hash(E.tgt, 0, m, V) : 0);
        for (int i = 0; i < m; ++i) { // target side, forward
            const int j = b0 + i;
            const float * lt = llama_get_logits_ith(E.tgt, i);
            row_t & r = rows[j];
            topk(lt, V, 8, r.top);
            for (int t : r.top) { r.topl.push_back(lt[t]); }
            r.tok = C[j]; r.next = j + 1 < L ? C[j + 1] : -1; r.top1 = r.top[0]; r.margin = margin(lt, r.top);
            if (is_ref) {
                std::copy(lt, lt + V, ref.logits.begin() + (size_t) j * V); ref.margin[j] = r.margin; ref.top1[j] = r.top1;
            } else {
                r.kl_ref = (float) kl(&ref.logits[(size_t) j * V], lt, V);
                r.dmargin = std::fabs(r.margin - ref.margin[j]); r.ref1 = r.top1 == ref.top1[j];
            }
            if (dump && j < n_dump) { wr(dump, lt, (size_t) V * sizeof(float)); }
            if (gen_sums && j < (int) gen_sums->size() && sum_d(lt, V) != (*gen_sums)[j]) { restore_bad++; }
        }
        for (int i = m - 1; i >= 0; --i) { // draft side, backward: the draft at row j may only see the draft KV < N+j
            const int j = b0 + i;
            row_t & r = rows[j];
            llama_memory_seq_rm(mem_dft, 0, N + j, -1);
            uint64_t chain = 0;
            draft_chain(E, C[j], N + j, i == 0 ? (const float *) (carrier.data() + 4) : llama_get_embeddings_nextn_ith(E.tgt, i - 1), r.d, r.p10, lg0,
                        E.snap ? &chain : nullptr);
            snap(E, "draft+rollback", sctx, (int) (N + j), E.D, chain);
            if (j == 0) { // the same draft through common_speculative_draft (the server's path; p_min 0, n_max D)
                static const tokens no_prompt;
                tokens res;
                common_speculative_set_state(E.spec, 0, carrier);
                auto & dp = common_speculative_get_draft_params(E.spec, 0);
                dp.drafting = true; dp.n_max = -1; dp.n_past = N; dp.id_last = C[0]; dp.prompt = &no_prompt; dp.result = &res;
                common_speculative_draft(E.spec);
                llama_memory_seq_rm(mem_dft, 0, N, -1);
                draft_check = std::vector<int>(res.begin(), res.end()) == r.d ? 0 : 1;
                snap(E, "server-draft+rollback", sctx, (int) N, (int) res.size(), hash64(res.data(), res.size() * sizeof(llama_token)));
            }
            const float * lt = llama_get_logits_ith(E.tgt, i);
            r.kl_dt = (float) kl(lg0.data(), lt, V);
            r.p_tgt = (float) std::exp(lg0[r.top1] - lse(lg0.data(), V));
        }
        common_speculative_set_state(E.spec, 0, carrier); // rebuild this batch's draft-KV cells; the carrier -> its last row
        if (!common_speculative_process(E.spec, b)) { fail("draft-mtp catch-up failed"); }
        snap(E, "re-catchup", sctx, (int) (N + b0), m, 0);
    }
    llama_batch_free(b);
    if (dump) { fclose(dump); }
    double s_kl = 0, s_dm = 0, s_p10 = 0, s_pt = 0, s_kdt = 0, mx_kl = 0;
    int n_ref1 = 0, n_agree = 0, n_empty = 0, n_cens = 0, s_acc = 0, s_drafted = 0, n_greedy = 0;
    for (int j = 0; j < L; ++j) { // acceptance-equivalent lengths (need the argmax of the following rows)
        row_t & r = rows[j];
        while (r.drafted < E.D && r.p10[r.drafted] >= E.p_min) { r.drafted++; }
        for (int k = 0; k < r.drafted; ++k) {
            if (j + k >= L || (k > 0 && C[j + k] != r.d[k - 1])) { r.cens = true; break; }
            if (r.d[k] != rows[j + k].top1) { break; }
            r.acc++;
        }
        n_agree += r.d[0] == r.top1; n_empty += r.drafted == 0; n_ref1 += r.ref1; n_greedy += r.next == r.top1;
        s_kl += r.kl_ref; mx_kl = std::max(mx_kl, (double) r.kl_ref); s_dm += r.dmargin; s_p10 += r.p10[0]; s_pt += r.p_tgt; s_kdt += r.kl_dt;
        if (r.cens) { n_cens++; } else { s_acc += r.acc; s_drafted += r.drafted; }
        if (out) {
            fprintf(out, "{\"t\":\"row\",\"state\":\"%s\",\"P\":\"%s\",\"V\":\"%s\",\"cont\":\"%s\",\"n\":%d,\"j\":%d,\"tok\":%d,\"next\":%d,"
                    "\"top_id\":%s,\"top_logit\":%s,\"margin\":%s,\"kl_ref\":%s,\"dmargin\":%s,\"top1_ref\":%d,\"draft\":%s,\"p10\":%s,"
                    "\"p_tgt\":%s,\"kl_dt\":%s,\"drafted\":%d,\"acc\":%d,\"cens\":%d}\n",
                    S.tag.c_str(), S.mode.c_str(), E.mode.c_str(), ctag.c_str(), n, j, r.tok, r.next, jarr(r.top).c_str(),
                    jarr(r.topl).c_str(), jnum(r.margin).c_str(), is_ref ? "null" : jnum(r.kl_ref).c_str(), is_ref ? "null" : jnum(r.dmargin).c_str(),
                    (int) r.ref1, jarr(r.d).c_str(), jarr(r.p10).c_str(), jnum(r.p_tgt).c_str(), jnum(r.kl_dt).c_str(), r.drafted, r.acc, (int) r.cens);
        }
    }
    const double acc_rate = s_drafted ? (double) s_acc / s_drafted : NAN, acc_len = L > n_cens ? (double) s_acc / (L - n_cens) : NAN;
    const double greedy = (double) n_greedy / std::max(1, L - 1);
    const std::string chk = draft_check < 0 ? "n/a" : draft_check == 0 ? "ok" : "MISMATCH";
    const std::string rchk = gen_sums ? (restore_bad == 0 ? "exact" : "MISMATCH(" + std::to_string(restore_bad) + ")") : "n/a";
    auto opt = [&](double v) { return is_ref ? std::string("null") : jnum(v); };
    if (out) {
        fprintf(out, "{\"t\":\"summary\",\"state\":\"%s\",\"P\":\"%s\",\"V\":\"%s\",\"cont\":\"%s\",\"n\":%d,\"ref_n\":%d,\"rows\":%d,"
                "\"n_past\":%d,\"greedy\":%s,\"kl_ref_mean\":%s,\"kl_ref_max\":%s,\"top1_ref\":%s,\"dmargin_mean\":%s,\"agree0\":%s,"
                "\"empty\":%s,\"acc_rate\":%s,\"acc_len\":%s,\"n_cens\":%d,\"p10_0\":%s,\"p_tgt\":%s,\"kl_dt\":%s,\"D\":%d,\"p_min\":%s,"
                "\"draft_check\":\"%s\",\"restore_check\":\"%s\"}\n",
                S.tag.c_str(), S.mode.c_str(), E.mode.c_str(), ctag.c_str(), n, ref.n, L, N, jnum(greedy).c_str(), opt(s_kl / L).c_str(),
                opt(mx_kl).c_str(), opt((double) n_ref1 / L).c_str(), opt(s_dm / L).c_str(), jnum((double) n_agree / L).c_str(),
                jnum((double) n_empty / L).c_str(), jnum(acc_rate).c_str(), jnum(acc_len).c_str(), n_cens, jnum(s_p10 / L).c_str(),
                jnum(s_pt / L).c_str(), jnum(s_kdt / L).c_str(), E.D, jnum(E.p_min).c_str(), chk.c_str(), rchk.c_str());
        fflush(out);
    }
    printf("%-20s %-10s %-10s %-12s %3d %5d %6.3f %10.3g %10.3g %6.3f %8.4f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %7.4f %-5s %s\n",
           S.tag.c_str(), S.mode.c_str(), E.mode.c_str(), ctag.c_str(), n, L, greedy, s_kl / L, mx_kl, (double) n_ref1 / L, s_dm / L,
           (double) n_agree / L, (double) n_empty / L, acc_rate, acc_len, s_p10 / L, s_pt / L, s_kdt / L, chk.c_str(), rchk.c_str());
    fflush(stdout);
}

static int compare(const std::string & a, const std::string & b) {
    std::vector<int> ia, ib;
    if (FILE * fs = open_magic(a, STATE_MAGIC)) {
        fclose(fs);
        const state_t A = load_state(a, true), B = load_state(b, true); // the full blobs: bytewise
        if (A.n_vocab != B.n_vocab || A.n_embd != B.n_embd) { fail("incompatible states"); }
        double l2 = 0, n2 = 0, mxd = 0; int nan_mm = 0;
        for (int i = 0; i < A.n_embd; ++i) { const double d = A.h()[i] - B.h()[i]; l2 += d * d; n2 += (double) A.h()[i] * A.h()[i]; }
        for (int i = 0; i < A.n_vocab; ++i) { // NaN-safe: a NaN on one side only = not equal (maxdiff -> inf -> null)
            const float x = A.logits[i], y = B.logits[i];
            if (std::isnan(x) != std::isnan(y)) { nan_mm++; mxd = INFINITY; } else if (!std::isnan(x)) { mxd = std::max(mxd, (double) std::fabs(x - y)); }
        }
        auto beq = [](const std::vector<uint8_t> & x, const std::vector<uint8_t> & y) { return x.size() == y.size() && (x.empty() || memcmp(x.data(), y.data(), x.size()) == 0); };
        const bool c_eq = beq(A.carrier, B.carrier), t_eq = beq(A.tgt, B.tgt), d_eq = beq(A.dft, B.dft);
        const bool l_eq = A.logits.size() == B.logits.size() && memcmp(A.logits.data(), B.logits.data(), A.logits.size() * sizeof(float)) == 0;
        const bool all_eq = c_eq && t_eq && d_eq && l_eq && A.prompt_hash == B.prompt_hash && A.n_past == B.n_past && A.last_tok == B.last_tok;
        topk(A.logits.data(), A.n_vocab, 2, ia); topk(B.logits.data(), B.n_vocab, 2, ib);
        printf("{\"t\":\"compare_state\",\"a\":\"%s\",\"b\":\"%s\",\"Pa\":\"%s\",\"Pb\":\"%s\",\"same_prompt\":%d,\"n_past\":%d,"
               "\"carrier_l2\":%s,\"carrier_rel\":%s,\"last_kl\":%s,\"last_top1_eq\":%d,\"last_dmargin\":%s,\"last_maxdiff\":%s,"
               "\"tgt_equal\":%d,\"dft_equal\":%d,\"carrier_bytes_equal\":%d,\"logits_bytes_equal\":%d,\"tgt_bytes_equal\":%d,"
               "\"dft_bytes_equal\":%d,\"nan_mismatch\":%d,\"tgt_size\":%zu,\"dft_size\":%zu,\"bytes_equal\":%d}\n", A.tag.c_str(), B.tag.c_str(), A.mode.c_str(), B.mode.c_str(),
               (int) (A.prompt_hash == B.prompt_hash && A.n_past == B.n_past), A.n_past, jnum(std::sqrt(l2)).c_str(),
               jnum(std::sqrt(l2 / std::max(n2, 1e-30))).c_str(), jnum(kl(A.logits.data(), B.logits.data(), A.n_vocab)).c_str(),
               (int) (ia[0] == ib[0]), jnum(std::fabs(margin(A.logits.data(), ia) - margin(B.logits.data(), ib))).c_str(),
               jnum(mxd).c_str(), (int) (A.tgt_hash == B.tgt_hash), (int) (A.dft_hash == B.dft_hash), (int) c_eq, (int) l_eq, (int) t_eq,
               (int) d_eq, nan_mm, A.tgt.size(), A.dft.size(), (int) all_eq);
        return 0;
    }
    FILE * f[2] = { open_magic(a, DUMP_MAGIC), open_magic(b, DUMP_MAGIC) };
    if (!f[0] && !f[1]) { // two snapshot files (--snapshot-every-ubatch): line by line, bytewise
        std::ifstream x(a, std::ios::binary), y(b, std::ios::binary);
        if (!x || !y) { fail("cannot read " + a + " / " + b); }
        std::string lx, ly; int n = 0, first = -1, na = 0, nb = 0;
        while (true) {
            const bool gx = (bool) std::getline(x, lx), gy = (bool) std::getline(y, ly);
            na += gx; nb += gy; if (!gx && !gy) { break; }
            if (first < 0 && (gx != gy || lx != ly)) { first = n; }
            n++;
        }
        if (na > 0 && read_file(a).rfind("{\"t\":\"snap\"", 0) != 0) { fail("not a state file, a logit dump or a snapshot file: " + a); }
        printf("{\"t\":\"compare_snap\",\"a\":\"%s\",\"b\":\"%s\",\"lines_a\":%d,\"lines_b\":%d,\"first_diff\":%d,\"bytes_equal\":%d}\n",
               base_tag(a).c_str(), base_tag(b).c_str(), na, nb, first, (int) (first < 0 && na > 0 && read_file(a) == read_file(b)));
        return 0;
    }
    int32_t h[2][2]; std::string meta[2];
    for (int k = 0; k < 2; ++k) {
        if (!f[k]) { fail("not a state file or a logit dump: " + (k ? b : a)); }
        std::vector<char> m; rd(f[k], h[k], sizeof(h[k])); rd_vec(f[k], m); meta[k].assign(m.begin(), m.end());
    }
    if (h[0][1] != h[1][1]) { fail("vocab sizes differ"); }
    const int R = std::min(h[0][0], h[1][0]), V = h[0][1];
    std::vector<float> x((size_t) V), y((size_t) V); double s_kl = 0, mx_kl = 0, s_dm = 0, mxd = 0; int n_eq = 0, n_beq = 0, nan_mm = 0;
    for (int r = 0; r < R; ++r) {
        rd(f[0], x.data(), (size_t) V * 4); rd(f[1], y.data(), (size_t) V * 4);
        const double k = kl(x.data(), y.data(), V); s_kl += k; mx_kl = std::max(mx_kl, k);
        topk(x.data(), V, 2, ia); topk(y.data(), V, 2, ib);
        n_eq += ia[0] == ib[0]; s_dm += std::fabs(margin(x.data(), ia) - margin(y.data(), ib));
        for (int i = 0; i < V; ++i) {
            if (std::isnan(x[i]) != std::isnan(y[i])) { nan_mm++; mxd = INFINITY; } else if (!std::isnan(x[i])) { mxd = std::max(mxd, (double) std::fabs(x[i] - y[i])); }
        }
        n_beq += memcmp(x.data(), y.data(), (size_t) V * 4) == 0;
    }
    fclose(f[0]); fclose(f[1]);
    printf("{\"t\":\"compare_dump\",\"a\":\"%s\",\"b\":\"%s\",\"rows\":%d,\"kl_mean\":%s,\"kl_max\":%s,\"top1_agree\":%s,\"dmargin_mean\":%s,"
           "\"maxdiff\":%s,\"rows_a\":%d,\"rows_b\":%d,\"bytes_equal_rows\":%d,\"nan_mismatch\":%d,\"bytes_equal\":%d}\n", meta[0].c_str(),
           meta[1].c_str(), R, jnum(R ? s_kl / R : NAN).c_str(), jnum(mx_kl).c_str(), jnum(R ? (double) n_eq / R : NAN).c_str(),
           jnum(R ? s_dm / R : NAN).c_str(), jnum(mxd).c_str(), h[0][0], h[1][0], n_beq, nan_mm,
           (int) (R > 0 && h[0][0] == h[1][0] && n_beq == R));
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    bool chat = false; int gen = 0, max_rows = 0, dump_rows = 0;
    std::string prompt_text, prompt_ids, state_out, gen_out, out_path, dump_dir, snap_path;
    std::vector<std::string> states_in, cmp;
    std::vector<std::pair<bool, std::string>> cont_files; // (ids?, path)
    std::vector<int> n_list = { 1, 6, 30 };
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { fail("missing value for " + a); } return argv[++i]; };
        if      (a == "--chat")        { chat = true; }
        else if (a == "--prompt-text") { prompt_text = val(); }
        else if (a == "--prompt-ids")  { prompt_ids = val(); }
        else if (a == "--state-out")   { state_out = val(); }
        else if (a == "--state-in")    { states_in.push_back(val()); }
        else if (a == "--cont")        { cont_files.emplace_back(false, val()); }
        else if (a == "--cont-ids")    { cont_files.emplace_back(true, val()); }
        else if (a == "--gen")         { gen = std::stoi(val()); }
        else if (a == "--gen-out")     { gen_out = val(); }
        else if (a == "--n-list")      { n_list.clear(); for (const auto & s : string_split<std::string>(val(), ',')) { n_list.push_back(std::stoi(s)); } }
        else if (a == "--out")         { out_path = val(); }
        else if (a == "--dump-logits") { dump_dir = val(); }
        else if (a == "--dump-rows")   { dump_rows = std::stoi(val()); }
        else if (a == "--max-rows")    { max_rows = std::stoi(val()); }
        else if (a == "--compare")     { cmp.push_back(val()); cmp.push_back(val()); }
        else if (a == "--snapshot-every-ubatch") { snap_path = val(); }
        else                           { rest.push_back(argv[i]); }
    }
    if (!cmp.empty()) { return compare(cmp[0], cmp[1]); }
    if (n_list.empty() || *std::min_element(n_list.begin(), n_list.end()) < 1) { fail("--n-list needs n >= 1"); }
    const bool do_prefill = !prompt_text.empty() || !prompt_ids.empty();
    if ((do_prefill && state_out.empty()) || (gen > 0 && !do_prefill)) { fail("a prefill needs --state-out; --gen needs a prefill"); }
    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_SPECULATIVE)) { return 1; }
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.sampling.backend_sampling = false;
    const int n_out = std::max(*std::max_element(n_list.begin(), n_list.end()), params.speculative.draft.n_max + 1);
    params.n_outputs_max = n_out; params.n_outputs_max_per_seq = n_out; // every row of a teacher-forced batch has logits

    llama_backend_init(); llama_numa_init(params.numa);
    auto init = common_init_from_params(params);
    llama_model * model = init->model(); env_t E; E.tgt = init->context();
    if (!model || !E.tgt) { fail("model / context init failed"); }
    common_params params_dft = common_base_params_to_speculative(params); // = the server: --model-draft or the fused MTP head
    auto spec_init = common_speculative_init_from_params(params_dft, model, E.tgt);
    if (!(E.dft = spec_init->context())) { fail("MTP draft context init failed"); }
    E.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    E.n_embd = llama_model_n_embd_out(model);
    if (llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(E.dft))) != E.n_vocab) { fail("draft / target vocabulary sizes differ"); }
    E.D = params.speculative.draft.n_max; E.p_min = params.speculative.draft.p_min; E.mode = mode_str();
    // the impl drafts all D steps (p_min 0, n_min 0) on host logits; the p_min gate is applied in the analysis
    params.speculative.draft.p_min = 0.0f; params.speculative.draft.n_min = 0; params.speculative.draft.backend_sampling = false;
    params.speculative.draft.ctx_tgt = E.tgt; params.speculative.draft.ctx_dft = E.dft;
    if (!(E.spec = common_speculative_init(params.speculative, 1))) { fail("draft-mtp init failed"); }
    if (!snap_path.empty() && !(E.snap = fopen(snap_path.c_str(), "w"))) { fail("cannot write " + snap_path); }
    E.b1 = llama_batch_init(1, E.n_embd, 1);
    E.b1.token = (llama_token *) malloc(sizeof(llama_token)); // the MTP batch carries token + embd (as the impl)
    LOG_INF("mtp-replay: mode %s, n_vocab %d, n_embd %d, D %d, p_min %.2f, n_ctx %u, n_batch %d\n", E.mode.c_str(), E.n_vocab, E.n_embd,
            E.D, E.p_min, llama_n_ctx(E.tgt), params.n_batch);

    std::vector<std::pair<std::string, tokens>> conts;
    std::vector<double> gen_sums;
    const std::string gen_tag = gen_out.empty() ? "gen" : base_tag(gen_out);
    if (do_prefill) {
        tokens prompt;
        if (!prompt_ids.empty()) {
            prompt = read_ids(prompt_ids);
        } else {
            std::string text = read_file(prompt_text);
            if (chat) { // = a /v1/chat/completions request with one user message and enable_thinking false (the needle probe)
                common_chat_templates_inputs in; common_chat_msg msg; msg.role = "user"; msg.content = text;
                in.messages = { msg }; in.add_generation_prompt = true; in.use_jinja = true; in.enable_thinking = false;
                in.chat_template_kwargs["enable_thinking"] = "false";
                text = common_chat_templates_apply(common_chat_templates_init(model, params.chat_template).get(), in).prompt;
            }
            prompt = common_tokenize(E.tgt, text, true, true);
        }
        if (prompt.empty() || (int64_t) prompt.size() + 2 > (int64_t) llama_n_ctx(E.tgt)) { fail("empty prompt or prompt >= n_ctx"); }
        state_t S;
        const int64_t t0 = ggml_time_ms();
        llama_batch b = llama_batch_init(params.n_batch, 0, 1);
        for (size_t i0 = 0; i0 < prompt.size(); i0 += params.n_batch) {
            const size_t m = std::min((size_t) params.n_batch, prompt.size() - i0);
            common_batch_clear(b);
            for (size_t i = 0; i < m; ++i) { common_batch_add(b, prompt[i0 + i], (llama_pos) (i0 + i), { 0 }, i0 + i + 1 == prompt.size()); }
            if (llama_decode(E.tgt, b) != 0) { fail("prefill decode failed at " + std::to_string(i0)); }
            if (!common_speculative_process(E.spec, b)) { fail("prefill draft-mtp catch-up failed at " + std::to_string(i0)); }
            snap(E, "prefill-catchup", "prefill", (int) i0, (int) m, i0 + m == prompt.size() && E.snap ? logits_hash(E.tgt, -1, 1, E.n_vocab) : 0);
            if ((i0 / params.n_batch) % 16 == 15) { LOG_INF("prefill %zu / %zu tokens, %.1f s\n", i0 + m, prompt.size(), (ggml_time_ms() - t0) / 1e3); }
        }
        llama_batch_free(b);
        common_speculative_begin(E.spec, 0, prompt);
        const float * lg = llama_get_logits_ith(E.tgt, -1);
        S.n_vocab = E.n_vocab; S.n_embd = E.n_embd; S.n_past = (int32_t) prompt.size(); S.last_tok = prompt.back(); S.mode = E.mode;
        S.prompt_hash = fnv(prompt.data(), prompt.size() * sizeof(llama_token)); S.logits.assign(lg, lg + E.n_vocab);
        if (!common_speculative_get_state(E.spec, 0, S.carrier)) { fail("no draft-mtp carrier"); }
        S.tgt.resize(llama_state_seq_get_size(E.tgt, 0)); S.dft.resize(llama_state_seq_get_size(E.dft, 0));
        if (llama_state_seq_get_data(E.tgt, S.tgt.data(), S.tgt.size(), 0) != S.tgt.size() ||
            llama_state_seq_get_data(E.dft, S.dft.data(), S.dft.size(), 0) != S.dft.size()) { fail("state capture failed"); }
        S.tgt_hash = fnv(S.tgt.data(), S.tgt.size()); S.dft_hash = fnv(S.dft.data(), S.dft.size());
        if (E.snap) {
            fprintf(E.snap, "{\"t\":\"snap\",\"at\":\"prefill-end\",\"n_past\":%d,\"tgt_size\":%zu,\"tgt\":\"%016llx\",\"dft_size\":%zu,"
                    "\"dft\":\"%016llx\",\"carrier\":\"%016llx\",\"logits\":\"%016llx\"}\n", S.n_past, S.tgt.size(),
                    (unsigned long long) hash64(S.tgt.data(), S.tgt.size()), S.dft.size(), (unsigned long long) hash64(S.dft.data(), S.dft.size()),
                    (unsigned long long) hash64(S.carrier.data(), S.carrier.size()), (unsigned long long) hash64(S.logits.data(), S.logits.size() * sizeof(float)));
            fflush(E.snap);
        }
        save_state(state_out, S);
        std::vector<int> ids; double h2 = 0;
        topk(S.logits.data(), E.n_vocab, 2, ids);
        for (int i = 0; i < E.n_embd; ++i) { h2 += (double) S.h()[i] * S.h()[i]; }
        LOG_INF("prefill: %zu tokens (hash %016llx) in %.1f s (mode %s), state %s: target %.1f MiB, draft %.1f MiB, carrier |h| %.4g, "
                "last top-1 %d margin %.4f\n",
                prompt.size(), (unsigned long long) S.prompt_hash, (ggml_time_ms() - t0) / 1e3, E.mode.c_str(), state_out.c_str(),
                S.tgt.size() / 1048576.0, S.dft.size() / 1048576.0, std::sqrt(h2), ids[0], margin(S.logits.data(), ids));
        if (gen > 0) { // greedy plain decodes from the LIVE state (target only): exact-id continuation + restore_check reference
            tokens g = { (llama_token) ids[0] };
            llama_batch b1 = llama_batch_init(1, 0, 1);
            for (int k = 0; k < gen; ++k) {
                common_batch_clear(b1);
                common_batch_add(b1, g[k], S.n_past + k, { 0 }, true);
                if (llama_decode(E.tgt, b1) != 0) { fail("gen decode failed"); }
                const float * l = llama_get_logits_ith(E.tgt, 0);
                gen_sums.push_back(sum_d(l, E.n_vocab));
                snap(E, "gen", "gen", S.n_past + k, 1, E.snap ? hash64(l, (size_t) E.n_vocab * sizeof(float)) : 0);
                if (k + 1 < gen) { topk(l, E.n_vocab, 1, ids); g.push_back(ids[0]); }
            }
            llama_batch_free(b1);
            if (!gen_out.empty()) { std::ofstream f(gen_out); for (auto t : g) { f << t << "\n"; } if (!f) { fail("cannot write " + gen_out); } }
            LOG_INF("%s: %d tokens: %s\n", gen_tag.c_str(), gen, common_detokenize(E.tgt, g, false).substr(0, 240).c_str());
            conts.emplace_back(gen_tag, g);
        }
    }
    for (const auto & [ids, path] : cont_files) {
        conts.emplace_back(base_tag(path), ids ? read_ids(path) : common_tokenize(E.tgt, read_file(path), false, false));
    }
    std::vector<std::string> replay_states = states_in;
    if (do_prefill) { replay_states.insert(replay_states.begin(), state_out); }
    if (!conts.empty() && !replay_states.empty()) {
        FILE * out = out_path.empty() ? nullptr : fopen(out_path.c_str(), "a");
        if (!out_path.empty() && !out) { fail("cannot write " + out_path); }
        printf("%-20s %-10s %-10s %-12s %3s %5s %6s %10s %10s %6s %8s %6s %6s %6s %6s %6s %6s %7s %-5s %s\n", "state", "P", "V", "cont", "n",
               "rows", "greedy", "KL(ref)", "KLmax", "top1=", "|dmarg|", "agree0", "empty", "acc", "acclen", "p10_0", "p_tgt", "KL(d|t)", "check", "restore");
        for (size_t si = 0; si < replay_states.size(); ++si) {
            const state_t S = load_state(replay_states[si], true); // one state's blobs in memory at a time
            if (S.n_vocab != E.n_vocab || S.n_embd != E.n_embd) { fail("state " + S.tag + " is from another model"); }
            for (const auto & [ctag, C0] : conts) {
                tokens C = C0;
                if (max_rows > 0 && (int) C.size() > max_rows) { C.resize(max_rows); }
                if (C.empty() || (int64_t) S.n_past + (int64_t) C.size() + E.D + 1 > (int64_t) llama_n_ctx(E.tgt)) {
                    fail("continuation " + ctag + " empty or beyond n_ctx"); }
                ref_t ref;
                const bool own_gen = do_prefill && gen > 0 && si == 0 && ctag == gen_tag;
                for (int n : n_list) { replay(E, S, C, ctag, n, ref, out, dump_dir, dump_rows, own_gen && n == 1 ? &gen_sums : nullptr); }
            }
        }
        if (out) { fclose(out); }
    }
    if (E.snap) { fclose(E.snap); }
    llama_batch_free(E.b1); common_speculative_free(E.spec); llama_backend_free();
    return 0;
}
