// [TAG_QSA_DUMP_QIN] (fr15, ruling R130, contrib 0260) GGML_OP_QSA_HASH and LLAMA_QSA_DUMP_QIN.
//
// Part 1, the op (CPU, and the first GPU device when there is one): the partials of ggml_qsa_hash equal, bit for bit, an
// independent re-implementation of the definition in ggml.h over q-like planes, strided views (a chunk of q, gaps between
// rows), the store + rows gather (f32 and f16, repeated rows, an out-of-range row) and tiny inputs;
//   determinism: the same input hashed twice (and on the GPU vs the CPU) gives the same partials and the same h64; the
//                h64 does not depend on the number of partials (1 vs 128)
//   sensitivity: flipping one bit of one element (first, last, 32 random positions) changes h64, restoring it restores
//                h64; swapping two different elements changes h64; in the gather, changing a store row that no index
//                names leaves h64 unchanged, changing a named row or one index changes it
// Part 2, the knob through the model graph under the PROD profile (the test-qsa-f16-mirror fixture: qwen4exp, 2 trunk
// layers + 1 MTP block, QSA ratio 4, 4 indexer heads; target (q8_0 KV) + MTP draft contexts; the 1M brain compose env
// verbatim incl. LLAMA_GRAPH_CACHE=8, LLAMA_QSA_GATHER_Q8=1, the scratch presizes, chunk 128 - LLAMA_QSA_MIN_KV scaled to
// 256): one child per arm (the knobs are read once), on the first GPU (CPU without one), GGML_VK_FUSION_COUNT=1 on a GPU:
//   A  ids              LLAMA_QSA_DUMP_IDS=1                                    (today's arm-ids pass)
//   B  ids + qin 2      A + LLAMA_QSA_DUMP_QIN=2                                (the gate mode)
//   C  ids + qin 2      B again (determinism)
//   D  plain            no dump knob                                            E  qin 1   D + LLAMA_QSA_DUMP_QIN=1
// PASS: target and draft logits lines A == B == C and D == E (the knob does not perturb numerics: the oracle's C2 rule);
// the [qsa-ids-chunk] / [qsa-ids] / [qsa-pooled] records A == B; the [graph-cache] trace A == B and D == E with cache hits
// in B and E (K = 8 reuse with the knob); the [vk-fusion-count] lines A == B and D == E once qsa_hash=<n> is taken out
// (same nodes, fusions, operand misalignment, pipelines and workgroup counts: the hashes changed no allocation and no
// dispatch) and B / E carry qsa_hash; the [qsa-qin-chunk] records B == C, both contexts (main, mtp), q / bias_blk /
// bias_qry / pooled (and on a GPU pooled16) records present, every record of B and E placed=1 with an h64; E (level 1)
// has q and bias records and no pooled ones.
// usage: test-qsa-qin-hash                (parent: part 1 here, part 2 through the children)
//        test-qsa-qin-hash --child        (one part-2 arm, the env decides)
//        test-qsa-qin-hash --perf         (ms per hash dispatch at the deployed shapes, first GPU)
//        test-qsa-qin-hash --part1        (part 1 only)

#include "common.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------------
// part 1: the op

// the definition (ggml.h ggml_qsa_hash), written out again here independently of ggml-cpu and the shader
static uint32_t ref_fmix32(uint32_t h) {
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

// the logical word stream of (a, rows) from host copies; a_bytes is the whole underlying buffer of a, a_off a's offset in it
static std::vector<uint32_t> ref_partials(const std::vector<uint8_t> & a_bytes, size_t a_off, const int64_t ne[4], const size_t nb[4],
                                          size_t row_bytes, const std::vector<int32_t> * rows, int n_part) {
    const uint64_t row_words = row_bytes/4;
    const uint64_t n_rows    = rows ? rows->size() : (uint64_t) (ne[1]*ne[2]*ne[3]);
    const uint64_t n_words   = row_words*n_rows;
    const uint64_t chunk     = ((n_words + n_part - 1)/n_part + 3)/4*4; // ceil(n / n_part), rounded up to 4 words
    std::vector<uint32_t> out(2*n_part, 0);
    for (uint64_t e = 0; e < n_words; ++e) {
        const uint64_t p = e/chunk;
        const uint64_t k = e/row_words, j = e%row_words;
        const uint32_t ei = (uint32_t) e;
        size_t off;
        if (rows) {
            const int32_t r = (*rows)[k];
            if (r < 0 || r >= ne[1]) {
                out[2*p] += ref_fmix32(ei ^ 0xA5A5A5A5u) + 1u;
                continue;
            }
            off = a_off + (size_t) r*nb[1];
        } else {
            const uint64_t i1 = k % ne[1], i2 = (k/ne[1]) % ne[2], i3 = k/(ne[1]*ne[2]);
            off = a_off + i1*nb[1] + i2*nb[2] + i3*nb[3];
        }
        uint32_t w;
        memcpy(&w, a_bytes.data() + off + 4*j, 4);
        out[2*p]     += ref_fmix32(w ^ (ei*0x9E3779B1u));
        out[2*p + 1] += ref_fmix32(((w << 16) | (w >> 16)) ^ (ei*0xC2B2AE3Du + 0x27D4EB2Fu));
    }
    return out;
}

static const int LLAMA_QSA_QIN_PARTS_TEST = 128; // = LLAMA_QSA_QIN_PARTS (src/llama-graph.h)

static int g_fail = 0;
#define EXPECT(cond, ...) do { if (!(cond)) { g_fail++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

// one hash case: a [ne] of type t, viewed from a base tensor of base_ne (a = view with nb / offset), rows optional
struct op_case {
    std::string           name;
    ggml_type             type;
    int64_t               base_ne[4];
    int64_t               ne[4];       // the view's extent
    int64_t               off_rows;    // view offset in base rows (along ne1 of the base)
    bool                  use_view;    // false: a is the base itself
    std::vector<int32_t>  rows;        // empty: no gather
};

struct op_runner {
    ggml_backend_t backend;
    const char *   name;

    // hashes a with n_part partials; base holds the bytes (row-major, contiguous), view per the case
    std::vector<uint32_t> run(const op_case & c, const std::vector<uint8_t> & base, int n_part, std::vector<uint8_t> * a_bytes_out,
                              size_t * a_off_out, int64_t ne_out[4], size_t nb_out[4], size_t * row_bytes_out) {
        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_4d(ctx, c.type, c.base_ne[0], c.base_ne[1], c.base_ne[2], c.base_ne[3]);
        ggml_tensor * rows = nullptr;
        if (!c.rows.empty()) {
            rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) c.rows.size());
        }
        ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2*n_part);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buf);
        ggml_backend_tensor_set(t, base.data(), 0, base.size());
        if (rows) {
            ggml_backend_tensor_set(rows, c.rows.data(), 0, c.rows.size()*sizeof(int32_t));
        }
        // junk in the partials: every slot must be written
        std::vector<int32_t> junk(2*n_part, 0x5A5A5A5A);
        ggml_backend_tensor_set(dst, junk.data(), 0, junk.size()*sizeof(int32_t));

        ggml_tensor * a = t;
        if (c.use_view) {
            a = ggml_view_4d(ctx, t, c.ne[0], c.ne[1], c.ne[2], c.ne[3], t->nb[1], t->nb[2], t->nb[3], c.off_rows*t->nb[1]);
        }
        ggml_tensor * h = ggml_qsa_hash(ctx, a, rows, dst);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, h);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
        GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        std::vector<uint32_t> out(2*n_part);
        ggml_backend_tensor_get(dst, out.data(), 0, out.size()*sizeof(uint32_t));
        if (a_bytes_out) {
            *a_bytes_out = base;
            *a_off_out   = a->view_offs;
            for (int i = 0; i < 4; ++i) { ne_out[i] = a->ne[i]; nb_out[i] = a->nb[i]; }
            *row_bytes_out = ggml_row_size(a->type, a->ne[0]);
        }
        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return out;
    }

    bool supports(const op_case & c) {
        ggml_init_params ip = { ggml_tensor_overhead()*8, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_4d(ctx, c.type, c.base_ne[0], c.base_ne[1], c.base_ne[2], c.base_ne[3]);
        ggml_tensor * a = c.use_view ? ggml_view_4d(ctx, t, c.ne[0], c.ne[1], c.ne[2], c.ne[3], t->nb[1], t->nb[2], t->nb[3], c.off_rows*t->nb[1]) : t;
        ggml_tensor * rows = c.rows.empty() ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) c.rows.size());
        ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 256);
        const bool ok = ggml_backend_supports_op(backend, ggml_qsa_hash(ctx, a, rows, dst));
        ggml_free(ctx);
        return ok;
    }
};

static uint64_t h64_of(const std::vector<uint32_t> & p, uint64_t n_words) {
    return ggml_qsa_hash_combine((const int32_t *) p.data(), (int64_t) p.size()/2, n_words);
}

static std::vector<uint8_t> random_bytes(const op_case & c, uint32_t seed) {
    const size_t n = (size_t) c.base_ne[0]*c.base_ne[1]*c.base_ne[2]*c.base_ne[3];
    std::mt19937 rng(seed);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    std::vector<uint8_t> b(n*ggml_type_size(c.type));
    for (size_t i = 0; i < n; ++i) {
        if (c.type == GGML_TYPE_F32) {
            const float v = dis(rng);
            memcpy(b.data() + 4*i, &v, 4);
        } else {
            const ggml_fp16_t v = ggml_fp32_to_fp16(dis(rng));
            memcpy(b.data() + 2*i, &v, 2);
        }
    }
    return b;
}

static void part1(op_runner & cpu, op_runner * gpu) {
    std::vector<op_case> cases;
    // q of one chunk: [idx_dim 128, 4 heads x 37 queries, 1 stream]
    cases.push_back({ "q-plane", GGML_TYPE_F32, { 128, 4*37, 1, 1 }, { 128, 4*37, 1, 1 }, 0, false, {} });
    // a chunk of q (rows 4*10 .. 4*29 of a 4*37-row q) - the q_idx view
    cases.push_back({ "q-chunk-view", GGML_TYPE_F32, { 128, 4*37, 1, 1 }, { 128, 4*20, 1, 1 }, 4*10, true, {} });
    // gaps: 3 streams, only 7 of the 11 rows of each stream (nb2 > 7*nb1)
    cases.push_back({ "strided-3d", GGML_TYPE_F32, { 128, 11, 3, 1 }, { 128, 7, 3, 1 }, 2, true, {} });
    // a 4-d plane, odd sizes
    cases.push_back({ "plane-4d", GGML_TYPE_F32, { 72, 5, 3, 2 }, { 72, 5, 3, 2 }, 0, false, {} });
    // the gather: store [128, 300] rows remap (repeats, two streams' worth)
    {
        std::vector<int32_t> r;
        std::mt19937 rng(5);
        for (int i = 0; i < 2*64; ++i) { r.push_back((int32_t) (rng() % 300)); }
        r[3] = r[7]; // a repeated row
        cases.push_back({ "store-rows-f32", GGML_TYPE_F32, { 128, 300, 1, 1 }, { 128, 300, 1, 1 }, 0, false, r });
        cases.push_back({ "store-rows-f16", GGML_TYPE_F16, { 128, 300, 1, 1 }, { 128, 300, 1, 1 }, 0, false, r });
        std::vector<int32_t> bad = r;
        bad[10] = 300; bad[11] = -1;
        cases.push_back({ "store-rows-oor", GGML_TYPE_F32, { 128, 300, 1, 1 }, { 128, 300, 1, 1 }, 0, false, bad });
    }
    // tiny: one f16 pair = one word; 3 words
    cases.push_back({ "tiny-1w", GGML_TYPE_F16, { 2, 1, 1, 1 }, { 2, 1, 1, 1 }, 0, false, {} });
    cases.push_back({ "tiny-3w", GGML_TYPE_F32, { 3, 1, 1, 1 }, { 3, 1, 1, 1 }, 0, false, {} });
    // larger than one partial per invocation: 512 x 1024 f32 (2 MiB)
    cases.push_back({ "big", GGML_TYPE_F32, { 512, 1024, 1, 1 }, { 512, 1024, 1, 1 }, 0, false, {} });

    std::vector<op_runner *> runners = { &cpu };
    if (gpu) { runners.push_back(gpu); }

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const op_case & c = cases[ci];
        const std::vector<uint8_t> base = random_bytes(c, 1000 + (uint32_t) ci);
        std::vector<uint8_t> ab; size_t a_off = 0; int64_t ne[4]; size_t nb[4]; size_t row_bytes = 0;
        const auto p_cpu = cpu.run(c, base, LLAMA_QSA_QIN_PARTS_TEST, &ab, &a_off, ne, nb, &row_bytes);
        const auto p_ref = ref_partials(ab, a_off, ne, nb, row_bytes, c.rows.empty() ? nullptr : &c.rows, LLAMA_QSA_QIN_PARTS_TEST);
        const uint64_t n_words = (uint64_t) row_bytes/4*(c.rows.empty() ? (uint64_t) (ne[1]*ne[2]*ne[3]) : c.rows.size());
        const uint64_t h_ref = h64_of(p_ref, n_words);
        EXPECT(p_cpu == p_ref, "%s: CPU partials != the reference", c.name.c_str());
        // n_part does not change the hash
        const auto p_one = cpu.run(c, base, 1, nullptr, nullptr, nullptr, nullptr, nullptr);
        EXPECT(h64_of(p_one, n_words) == h_ref, "%s: h64 with 1 partial != with %d", c.name.c_str(), LLAMA_QSA_QIN_PARTS_TEST);
        for (op_runner * r : runners) {
            if (!r->supports(c)) {
                EXPECT(r == &cpu ? false : true, "%s: %s does not support the case", c.name.c_str(), r->name);
                printf("  %-16s %-8s unsupported\n", c.name.c_str(), r->name);
                continue;
            }
            const auto p1 = r->run(c, base, LLAMA_QSA_QIN_PARTS_TEST, nullptr, nullptr, nullptr, nullptr, nullptr);
            const auto p2 = r->run(c, base, LLAMA_QSA_QIN_PARTS_TEST, nullptr, nullptr, nullptr, nullptr, nullptr);
            EXPECT(p1 == p_ref, "%s: %s partials != the reference", c.name.c_str(), r->name);
            EXPECT(p1 == p2,    "%s: %s partials differ between two runs", c.name.c_str(), r->name);
            printf("  %-16s %-8s words=%-8" PRIu64 " h64=%016" PRIx64 " %s\n", c.name.c_str(), r->name, n_words, h64_of(p1, n_words),
                   p1 == p_ref && p1 == p2 ? "ok" : "MISMATCH");
        }

        // sensitivity (every runner): one flipped bit of one element changes the hash; restoring restores it
        const size_t esz = ggml_type_size(c.type);
        std::vector<size_t> pos;   // byte offsets of elements inside the hashed region
        if (c.rows.empty()) {
            // first / last hashed element and 32 random ones
            const int64_t n_el = ne[0]*ne[1]*ne[2]*ne[3];
            std::mt19937 rng(77 + (uint32_t) ci);
            std::vector<int64_t> idx = { 0, n_el - 1 };
            for (int i = 0; i < 32 && n_el > 2; ++i) { idx.push_back((int64_t) (rng() % (uint64_t) n_el)); }
            for (int64_t e : idx) {
                const int64_t i0 = e % ne[0], i1 = (e/ne[0]) % ne[1], i2 = (e/(ne[0]*ne[1])) % ne[2], i3 = e/(ne[0]*ne[1]*ne[2]);
                pos.push_back(a_off + i0*esz + i1*nb[1] + i2*nb[2] + i3*nb[3]);
            }
        } else {
            for (size_t k : { (size_t) 0, c.rows.size() - 1, c.rows.size()/2 }) {
                const int32_t r = c.rows[k];
                if (r >= 0 && r < ne[1]) { pos.push_back((size_t) r*nb[1] + esz*(k % ne[0])); }
            }
        }
        for (op_runner * r : runners) {
            if (!r->supports(c)) { continue; }
            int n_changed = 0;
            for (size_t bp : pos) {
                std::vector<uint8_t> m = base;
                m[bp] ^= 0x01; // the lowest mantissa bit
                const auto pm = r->run(c, m, LLAMA_QSA_QIN_PARTS_TEST, nullptr, nullptr, nullptr, nullptr, nullptr);
                n_changed += h64_of(pm, n_words) != h_ref;
            }
            EXPECT(n_changed == (int) pos.size(), "%s: %s: %d of %zu one-bit flips changed the hash", c.name.c_str(), r->name, n_changed, pos.size());
            const auto pr = r->run(c, base, LLAMA_QSA_QIN_PARTS_TEST, nullptr, nullptr, nullptr, nullptr, nullptr);
            EXPECT(h64_of(pr, n_words) == h_ref, "%s: %s: restored input, different hash", c.name.c_str(), r->name);
            // swap two different elements of the hashed region (the first two positions)
            if (pos.size() >= 2 && c.rows.empty()) {
                std::vector<uint8_t> m = base;
                if (memcmp(m.data() + pos[0], m.data() + pos[1], esz) != 0) {
                    std::vector<uint8_t> tmp(m.begin() + pos[0], m.begin() + pos[0] + esz);
                    memcpy(m.data() + pos[0], m.data() + pos[1], esz);
                    memcpy(m.data() + pos[1], tmp.data(), esz);
                    const auto pm = r->run(c, m, LLAMA_QSA_QIN_PARTS_TEST, nullptr, nullptr, nullptr, nullptr, nullptr);
                    EXPECT(h64_of(pm, n_words) != h_ref, "%s: %s: swapping two elements kept the hash", c.name.c_str(), r->name);
                }
            }
            if (!c.rows.empty()) {
                // a row no index names: unchanged; a different index: changed
                std::vector<bool> named(ne[1], false);
                for (int32_t x : c.rows) { if (x >= 0 && x < ne[1]) { named[x] = true; } }
                const int64_t free_row = std::find(named.begin(), named.end(), false) - named.begin();
                if (free_row < ne[1]) {
                    std::vector<uint8_t> m = base;
                    m[(size_t) free_row*nb[1]] ^= 0xFF;
                    const auto pm = r->run(c, m, LLAMA_QSA_QIN_PARTS_TEST, nullptr, nullptr, nullptr, nullptr, nullptr);
                    EXPECT(h64_of(pm, n_words) == h_ref, "%s: %s: an unnamed store row changed the hash", c.name.c_str(), r->name);
                }
                op_case c2 = c;
                c2.rows[1] = (c2.rows[1] + 1) % (int32_t) ne[1];
                if (memcmp(base.data() + (size_t) c.rows[1]*nb[1], base.data() + (size_t) c2.rows[1]*nb[1], row_bytes) != 0) {
                    const auto pm = r->run(c2, base, LLAMA_QSA_QIN_PARTS_TEST, nullptr, nullptr, nullptr, nullptr, nullptr);
                    EXPECT(h64_of(pm, n_words) != h_ref, "%s: %s: a changed row index kept the hash", c.name.c_str(), r->name);
                }
            }
            printf("  %-16s %-8s sensitivity: %d/%zu flips changed h64\n", c.name.c_str(), r->name, n_changed, pos.size());
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// part 2: the knob through the model graph

static const uint32_t N_VOCAB = 128;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-1f);
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) { tmp[i] = dis(gen); }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) { tmp[i] = ggml_fp32_to_fp16(dis(gen)); }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// qwen4exp: the test-qsa-f16-mirror fixture
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
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        n_ff);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);
    ms.add_kv(LLM_KV_EXPERT_COUNT,                uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,           uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,         uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,          uint32_t(2)); // sigmoid
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,        1.0f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,             uint32_t(2));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");
    return ret;
}

static bool silent_progress(float, void *) { return true; }

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR || strstr(text, "LLAMA_QSA_DUMP_QIN") != nullptr) {
        fputs(text, stderr);
    }
}

static uint64_t fnv(const void * p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    for (size_t i = 0; i < n; ++i) { h = (h ^ ((const uint8_t *) p)[i]) * 0x100000001b3ull; }
    return h;
}

static int child() {
    llama_log_set(quiet_log, nullptr);
    setvbuf(stdout, nullptr, _IOLBF, 0);
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); }
    printf("[qin] device %s\n", dev ? ggml_backend_dev_name(dev) : "CPU");

    gguf_context_ptr meta = make_meta(4096);
    llama_model_params mp = llama_model_default_params();
    mp.progress_callback = silent_progress;
    mp.load_mtp          = true;
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    if (dev) {
        mp.n_gpu_layers = 999;
        mp.devices      = devs;
    } else {
        mp.n_gpu_layers = 0;
    }
    size_t seed = 31;
    llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, &seed, mp));
    GGML_ASSERT(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 256; cp.n_ubatch = 256; cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // the prod target cache type (-ctk / -ctv q8_0); the draft context keeps the fixture's f16
    if (getenv("QIN_TEST_KV_Q8")) { cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0; }
    llama_context_ptr tgt(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(tgt);
    llama_context_params cd = cp;
    cd.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cd.type_k = GGML_TYPE_F16; cd.type_v = GGML_TYPE_F16;
    llama_context_ptr dft(llama_init_from_model(model.get(), cd));
    GGML_ASSERT(dft);

    common_params_speculative params;
    params.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.draft.n_max   = 4;
    params.draft.n_min   = 0;
    params.draft.p_min   = 0.0f;
    params.draft.ctx_tgt = tgt.get();
    params.draft.ctx_dft = dft.get();
    common_speculative * spec = common_speculative_init(params, 1);
    GGML_ASSERT(spec);
    common_speculative_begin(spec, 0, {});

    std::mt19937 rng(77);
    llama_pos pos = 0;
    llama_token id_last = 0;
    int step = 0;

    auto feed = [&](int n, const char * what) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            common_batch_add(b, (llama_token) (rng() % N_VOCAB), pos + i, { 0 }, true);
        }
        GGML_ASSERT(llama_decode(tgt.get(), b) == 0);
        uint64_t h = 0xcbf29ce484222325ull;
        for (int i = 0; i < n; ++i) { h = fnv(llama_get_logits_ith(tgt.get(), i), N_VOCAB*sizeof(float), h); }
        const float * last = llama_get_logits_ith(tgt.get(), n - 1);
        id_last = (llama_token) (std::max_element(last, last + N_VOCAB) - last);
        GGML_ASSERT(common_speculative_process(spec, b));
        llama_synchronize(dft.get());
        printf("[qin] %d %s n=%d pos=%d tgt=%016" PRIx64 "\n", step++, what, n, (int) pos, h);
        llama_batch_free(b);
        pos += n;
    };
    auto rounds = [&](int n) {
        for (int r = 0; r < n; ++r) {
            std::vector<llama_token> drafted;
            auto & dp = common_speculative_get_draft_params(spec, 0);
            dp.drafting = true; dp.n_max = -1; dp.n_past = pos; dp.id_last = id_last; dp.prompt = nullptr; dp.result = &drafted;
            common_speculative_draft(spec);
            llama_synchronize(dft.get());
            uint64_t hd = 0;
            if (!drafted.empty()) {
                hd = fnv(llama_get_logits_ith(dft.get(), -1), N_VOCAB*sizeof(float));
                hd = fnv(drafted.data(), drafted.size()*sizeof(llama_token), hd);
            }
            printf("[qin] %d draft n=%zu dft=%016" PRIx64 "\n", step++, drafted.size(), hd);
            llama_memory_seq_rm(llama_get_memory(dft.get()), 0, pos, -1);
            feed(1 + r % 6, "round");
            common_speculative_accept(spec, 0, 0);
        }
    };

    // a cold fill in 256-row ubatches (chunked QSA: 2 chunks of 128), then draft -> verify rounds of 1..6 rows
    for (int c = 0; c < 2; ++c) { feed(256, "fill"); }
    rounds(14);
    feed(200, "fill2");
    rounds(6);
    common_speculative_free(spec);
    fflush(stdout);
    return 0;
}

static std::vector<std::string> run_child(const char * self, const std::string & env) {
    const std::string cmd = env + " '" + self + "' --child 2>&1";
    std::vector<std::string> lines;
    FILE * p = popen(cmd.c_str(), "r");
    GGML_ASSERT(p);
    std::string cur;
    char buf[4096];
    while (fgets(buf, sizeof(buf), p)) {
        cur += buf;
        if (!cur.empty() && cur.back() == '\n') { lines.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) { lines.push_back(cur + "\n"); }
    const int rc = pclose(p);
    lines.push_back("rc=" + std::to_string(rc) + "\n");
    return lines;
}

static std::vector<std::string> pick(const std::vector<std::string> & v, const char * prefix) {
    std::vector<std::string> r;
    for (const auto & l : v) { if (l.rfind(prefix, 0) == 0) { r.push_back(l); } }
    return r;
}

// the [vk-fusion-count] line without its qsa_hash=<n> entry
static std::string strip_qsa_hash(const std::string & l) {
    const size_t i = l.find(" qsa_hash=");
    if (i == std::string::npos) { return l; }
    const size_t j = l.find(' ', i + 1);
    return l.substr(0, i) + (j == std::string::npos ? std::string("\n") : l.substr(j));
}

static bool same(const char * what, const std::vector<std::string> & a, const std::vector<std::string> & b, bool need_nonempty = true) {
    size_t n_diff = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (a[i] != b[i] && n_diff++ == 0) { printf("  %s first difference (line %zu):\n    %s    %s", what, i, a[i].c_str(), b[i].c_str()); }
    }
    const bool ok = a.size() == b.size() && n_diff == 0 && (!need_nonempty || !a.empty());
    printf("  %-34s %s (%zu vs %zu lines, %zu differ)\n", what, ok ? "same" : "DIFFERENT", a.size(), b.size(), n_diff);
    return ok;
}

static size_t count_has(const std::vector<std::string> & v, const char * a, const char * b = nullptr) {
    size_t n = 0;
    for (const auto & l : v) { n += l.find(a) != std::string::npos && (b == nullptr || l.find(b) != std::string::npos); }
    return n;
}

// M1 (review of 1ea219d9c): the identity is proven under the PROD profile - the 1M brain compose's env
// (deploy/docker-compose.qwen38-flash-next-uncensored-brain-1m.yml) verbatim, K = 8 graph cache included, q8_0 target KV -
// with two test-size substitutions: LLAMA_QSA_MIN_KV 65536 -> 256 (so the 1024-cell fixture runs both the dense and the
// sparse graph) and the GTT admission / path knobs left out (scheduling only, and the host differs)
static void part2(const char * self, bool gpu) {
    const std::string PROD_ENV =
        "env -u LLAMA_QSA_DUMP_QIN -u LLAMA_QSA_DUMP_IDS -u LLAMA_QSA_DUMP_SCORE -u GGML_VK_FUSION_COUNT -u LLAMA_QSA_POOL_REMAP "
        "GGML_VK_DISABLE_MMQ_CM1=1 LLAMA_QSA_COMPACT_BIAS=1 LLAMA_KQ_VIS=1 LLAMA_KQ_VIS_MIN_TOKENS=64 "
        "GGML_VK_QSA_SCRATCH_PRESIZE=1 GGML_VK_QSA_SCRATCH_PRESIZE_FA=1 GGML_VK_QSA_FUSED_SELECT=1 LLAMA_QSA_CHUNK=128 "
        "LLAMA_QSA_GATHER_Q8=1 LLAMA_QSA_POOLED_CACHE=1 LLAMA_QSA_MIN_KV=256 LLAMA_QSA_GATHER_MAX_WINDOWS=64 "
        "LLAMA_QSA_PREFILL=indexed LLAMA_QSA_VIS=1 LLAMA_QSA_CENSUS_INC=1 LLAMA_QSA_INPUTS_INC=1 LLAMA_QSA_RELU_INPLACE=1 "
        "LLAMA_MTP_CATCHUP_ATTN=0 LLAMA_MTP_QSA=1 LLAMA_GRAPH_CACHE=8 GGML_VK_FREE_WAIT_IDLE=1 GGML_ALLOC_GROW_SLACK=10 "
        "LLAMA_RESERVE_RUNTIME_SHAPE=1 GGML_VK_FA_GQA_TOKENS=0 GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296 "
        "QIN_TEST_KV_Q8=1 LLAMA_GRAPH_CACHE_TRACE=1";
    const std::string FC  = gpu ? " GGML_VK_FUSION_COUNT=1" : "";
    const std::string IDS = PROD_ENV + " LLAMA_QSA_DUMP_IDS=1" + FC;
    const auto A = run_child(self, IDS);
    const auto B = run_child(self, IDS + " LLAMA_QSA_DUMP_QIN=2");
    const auto C = run_child(self, IDS + " LLAMA_QSA_DUMP_QIN=2");
    const auto D = run_child(self, PROD_ENV + FC);
    const auto E = run_child(self, PROD_ENV + FC + " LLAMA_QSA_DUMP_QIN=1");

    for (const auto * v : { &A, &B, &C, &D, &E }) {
        if (v->back() != "rc=0\n") {
            g_fail++;
            printf("  FAIL: a child exited with %s  last lines:\n", v->back().c_str());
            for (size_t i = v->size() > 12 ? v->size() - 12 : 0; i < v->size(); ++i) { printf("    %s", (*v)[i].c_str()); }
        }
    }
    const auto dev = pick(A, "[qin] device");
    printf("  part 2 on %s", dev.empty() ? "?\n" : dev[0].c_str() + 13);

    bool ok = true;
    ok &= same("logits A (ids) == B (ids+qin2)",  pick(A, "[qin] "), pick(B, "[qin] "));
    ok &= same("logits B == C (qin2 again)",       pick(B, "[qin] "), pick(C, "[qin] "));
    ok &= same("logits D (plain) == E (qin1)",     pick(D, "[qin] "), pick(E, "[qin] "));
    ok &= same("[qsa-ids-chunk] A == B",           pick(A, "[qsa-ids-chunk] "), pick(B, "[qsa-ids-chunk] "));
    ok &= same("[qsa-ids] A == B",                 pick(A, "[qsa-ids] "), pick(B, "[qsa-ids] "));
    ok &= same("[qsa-pooled] A == B",              pick(A, "[qsa-pooled] "), pick(B, "[qsa-pooled] "));
    // the K = 8 graph cache: the same hit / miss sequence with the knob, and real hits
    ok &= same("[graph-cache] A == B",             pick(A, "[graph-cache] "), pick(B, "[graph-cache] "));
    ok &= same("[graph-cache] D == E",             pick(D, "[graph-cache] "), pick(E, "[graph-cache] "));
    const size_t n_hit_b = count_has(B, "[graph-cache] ", " reused=1"), n_hit_e = count_has(E, "[graph-cache] ", " reused=1");
    printf("  graph cache hits: B %zu, E %zu\n", n_hit_b, n_hit_e);
    ok &= n_hit_b > 0 && n_hit_e > 0;
    if (gpu) {
        for (int pair = 0; pair < 2; ++pair) {
            const auto & off = pair == 0 ? A : D;
            const auto & on  = pair == 0 ? B : E;
            std::vector<std::string> f_on;
            size_t n_hash_lines = 0;
            for (const auto & l : pick(on, "[vk-fusion-count] ")) { n_hash_lines += l.find(" qsa_hash=") != std::string::npos; f_on.push_back(strip_qsa_hash(l)); }
            ok &= same(pair == 0 ? "[vk-fusion-count] A == B - qsa_hash" : "[vk-fusion-count] D == E - qsa_hash", pick(off, "[vk-fusion-count] "), f_on);
            printf("  %s: %zu fusion lines carry qsa_hash=\n", pair == 0 ? "B" : "E", n_hash_lines);
            ok &= n_hash_lines > 0;
        }
    }
    const auto qb = pick(B, "[qsa-qin-chunk] "), qc = pick(C, "[qsa-qin-chunk] "), qe = pick(E, "[qsa-qin-chunk] ");
    ok &= same("[qsa-qin-chunk] B == C",           qb, qc);
    const size_t n_main = count_has(qb, "ctx=main "), n_mtp = count_has(qb, "ctx=mtp "), n_q = count_has(qb, " what=q ");
    const size_t n_blk = count_has(qb, " what=bias_blk "), n_qry = count_has(qb, " what=bias_qry "), n_bias = count_has(qb, " what=bias ");
    const size_t n_pooled = count_has(qb, " what=pooled "), n_pooled16 = count_has(qb, " what=pooled16 ");
    const size_t n_chunk1 = count_has(qb, " chunk=1 ", " what=q "), n_occ1 = count_has(qb, " occ=1 ");
    size_t n_bad = 0;
    for (const auto & l : qb) { n_bad += l.find(" h64=") == std::string::npos || l.find(" placed=1") == std::string::npos; }
    for (const auto & l : qe) { n_bad += l.find(" h64=") == std::string::npos || l.find(" placed=1") == std::string::npos; }
    const size_t n_qe = count_has(qe, " what=q "), n_pe = count_has(qe, " what=pooled"), n_be = count_has(qe, " what=bias_blk ");
    printf("  B qin records: %zu (main %zu, mtp %zu; q %zu (chunk 1: %zu), bias_blk %zu, bias_qry %zu, bias %zu, pooled %zu, pooled16 %zu, occ=1 %zu); "
           "not placed / no h64 (B+E) %zu; E: q %zu, bias_blk %zu, pooled %zu\n", qb.size(), n_main, n_mtp, n_q, n_chunk1, n_blk, n_qry, n_bias,
           n_pooled, n_pooled16, n_occ1, n_bad, n_qe, n_be, n_pe);
    ok &= n_main > 0 && n_mtp > 0 && n_q > 0 && n_pooled > 0 && n_chunk1 > 0 && n_bad == 0 && n_qe > 0 && n_pe == 0;
    ok &= n_blk > 0 && n_qry > 0 && n_be > 0;          // the bias fold's inputs
    if (gpu) {
        ok &= n_pooled16 > 0;                          // the f16 mirror the coopmat1 score reads
    }
    // the knob off prints nothing
    ok &= pick(A, "[qsa-qin-chunk] ").empty() && pick(D, "[qsa-qin-chunk] ").empty();
    if (!ok) { g_fail++; }
    printf("  part 2: %s\n", ok ? "PASS" : "FAIL");
}

// --perf: ms per QSA_HASH dispatch on the first GPU (graph compute incl. submit + wait, median of 15) at the deployed 1M shapes
// (262144 pooled blocks): q of one 128-query chunk, the bias fold's block table, the folded bias plane of one chunk (r1),
// the pooled operand as f32 / f16 rows
static int perf() {
    ggml_backend_load_all();
    ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gdev) { gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); }
    if (!gdev) { printf("perf: SKIP no GPU\n"); return 0; }
    ggml_backend_t be = ggml_backend_dev_init(gdev, nullptr);
    struct pc { const char * name; ggml_type type; int64_t ne0, ne1; int64_t n_rows; };
    const pc cases[] = {
        { "q-chunk128",          GGML_TYPE_F32, 128, 4*128,  0 },      // what=q, per chunk (prod LLAMA_QSA_CHUNK=128)
        { "bias-blk-1M",         GGML_TYPE_I32, 2,   262144, 0 },      // what=bias_blk, once per graph
        { "bias-plane-1M-chunk", GGML_TYPE_F32, 262144, 128, 0 },      // r1's what=bias (the folded plane, per chunk) - for comparison
        { "pooled-1M-f32-rows",  GGML_TYPE_F32, 128, 262144, 262144 }, // what=pooled, per (graph, layer), level 2
        { "pooled-1M-f16-rows",  GGML_TYPE_F16, 128, 262144, 262144 }, // what=pooled16, per (graph, layer), level 2
    };
    for (const auto & c : cases) {
        ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_2d(ctx, c.type, c.ne0, c.ne1);
        ggml_tensor * rows = c.n_rows ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_rows) : nullptr;
        ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2*LLAMA_QSA_QIN_PARTS_TEST);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_buffer_clear(buf, 0);
        if (rows) {
            std::vector<int32_t> r(c.n_rows);
            std::mt19937 rng(3);
            for (auto & x : r) { x = (int32_t) (rng() % c.ne1); }
            ggml_backend_tensor_set(rows, r.data(), 0, r.size()*sizeof(int32_t));
        }
        ggml_tensor * h = ggml_qsa_hash(ctx, a, rows, dst);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, h);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
        std::vector<double> ms;
        for (int i = 0; i < 16; ++i) {
            const int64_t t0 = ggml_time_us();
            GGML_ASSERT(ggml_backend_graph_compute(be, gf) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(be);
            if (i > 0) { ms.push_back((ggml_time_us() - t0)/1000.0); }
        }
        std::sort(ms.begin(), ms.end());
        printf("perf: %-20s bytes=%-10" PRIu64 " median %.3f ms\n", c.name, ggml_qsa_hash_n_words(a, rows)*4, ms[ms.size()/2]);
        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "--child") == 0) {
        return child();
    }
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        return perf();
    }
    const bool part1_only = argc > 1 && strcmp(argv[1], "--part1") == 0; // e.g. a GGML_VULKAN_CHECK_RESULTS build
    setvbuf(stdout, nullptr, _IOLBF, 0);
    ggml_backend_load_all();

    printf("part 1: GGML_OP_QSA_HASH\n");
    op_runner cpu { ggml_backend_cpu_init(), "CPU" };
    ggml_backend_dev_t gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!gdev) { gdev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU); }
    op_runner gpu { gdev ? ggml_backend_dev_init(gdev, nullptr) : nullptr, gdev ? ggml_backend_dev_name(gdev) : "none" };
    part1(cpu, gpu.backend ? &gpu : nullptr);
    if (!gpu.backend) { printf("  (no GPU: CPU only)\n"); }
    if (gpu.backend) { ggml_backend_free(gpu.backend); }
    ggml_backend_free(cpu.backend);
    const int fail1 = g_fail;
    printf("part 1: %s\n", fail1 == 0 ? "PASS" : "FAIL");

    if (!part1_only) {
        printf("part 2: LLAMA_QSA_DUMP_QIN through the model graph (prod profile)\n");
        part2(argv[0], gdev != nullptr);
    }

    printf("test-qsa-qin-hash: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
