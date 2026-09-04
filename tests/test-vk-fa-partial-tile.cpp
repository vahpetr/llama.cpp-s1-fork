// Flash-attention partial-Br-tile reproducibility probe.
//
// Motivation: on Vulkan, two decode dispatches that share the same Q/K/V/mask
// data disagree on the *shared* query rows when only the batch shape (the
// number of query rows N) changes. The recurrent-state rollback path in the
// server relies on re-running a prefix of a batch and getting bit-identical
// activations back, so this is a correctness defect for it.
//
// Geometry mirrors the production ornith-1.0-9b attention op:
//   hsk = hsv = 256, 4 KV heads, GQA ratio 4 (16 Q heads), physical KV = 256,
//   Q F32, K/V F16, mask F16, GGML_PREC_F32, scale 1/16,
//   no sinks, no ALiBi, no logit softcap.
//
// The test builds two graphs over one shared set of device buffers: one with
// N = 5 query rows and one with N = 6.  Query rows 0..4 are literally the same
// bytes (the N=5 tensors are prefix views of the N=6 tensors); only row 5 and
// the batch shape differ.  The mask is causal with cutoff L for query row 4
// (mask[k] == 0 for k < L, -INF otherwise), row r < 4 uses cutoff L-4+r and
// row 5 uses cutoff L+1.  L is swept over 27..35 so that both parities of the
// causal cutoff are covered.
//
// Requirement: output row 4 must be BIT IDENTICAL between the N=5 and the N=6
// run.  The deviation against the CPU backend is reported for information
// only; it is not a pass criterion.
//
// Two buffer layouts are exercised:
//   view  - K/V are views of a 2x larger buffer and Q/mask are prefix views,
//           i.e. the layout used by tests/test-backend-ops.cpp (kv_view=true).
//   cache - the exact strides llama.cpp produces at run time: Q permuted from
//           [head_dim, n_head, n_tokens] and K/V permuted views of the
//           [n_embd_k_gqa, kv_size] KV cache.
//
// --kvtail selects what the two runs see in the mask-suppressed KV tail:
//   shared - both runs read the same data in all 256 physical KV cells.
//   prod   - models the KV cache: the short run has written cells [0, L) and
//            zeroes beyond, the long run has written cells [0, L] (one more,
//            for its extra token) and zeroes beyond. This is the only input
//            difference the two production decodes have besides the batch
//            shape, and the mask suppresses it for the compared row.
//   poison - like prod, but the unwritten tail carries distinct finite data in
//            each run, so a non-inert masked tail shows up immediately.
//
// --pollute runs a full-Br (N=16) flash-attention dispatch on the same pipeline
// in between the two compared dispatches, so that the shared-memory query tile
// holds large live values from an unrelated dispatch. Without it, two
// back-to-back dispatches of the same shape are likely to see the same stale
// shared memory, which would hide a padded-row leak.
//
// Relevant environment knobs (read once at process start, so run the binary
// once per mode): GGML_VK_FA_NO_GQA, GGML_VK_FA_NO_SPLIT_K,
// GGML_VK_FA_SPLIT_K_LOG, GGML_VK_DISABLE_COOPMAT.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int64_t HSK   = 256;
static const int64_t HSV   = 256;
static const int64_t NH_KV = 4;                 // KV heads
static const int64_t GQA   = 4;                 // Q heads per KV head
static const int64_t NH_Q  = NH_KV * GQA;       // 16 Q heads
static const int64_t KVLEN = 256;               // physical KV length
static int64_t NB_HI = 6;                 // long dispatch
static int64_t NB_LO = 5;                 // short dispatch
static int64_t TROW  = 4;                 // compared query row (last row of the short dispatch)

static const size_t ROW_ELEMS = (size_t) HSV * NH_Q;

static const int64_t NB_POL = 16;               // full-Br polluting dispatch

enum layout_t { LAYOUT_VIEW = 0, LAYOUT_CACHE = 1 };
enum kvtail_t { KVTAIL_SHARED = 0, KVTAIL_PROD = 1, KVTAIL_POISON = 2 };

static const char * layout_name(layout_t l) {
    return l == LAYOUT_VIEW ? "view" : "cache";
}

static const char * kvtail_name(kvtail_t t) {
    return t == KVTAIL_SHARED ? "shared" : (t == KVTAIL_PROD ? "prod" : "poison");
}

// index-deterministic pseudo random values in [-1, 1)
static float hashf(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (float) ((int64_t) (x & 0xFFFFFF) - 0x800000) / (float) 0x800000;
}

struct fa_ctx {
    layout_t              layout  = LAYOUT_VIEW;
    ggml_backend_t        backend = nullptr;
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    // parent (allocated) buffers
    ggml_tensor * qsrc = nullptr;
    ggml_tensor * ksrc = nullptr;
    ggml_tensor * vsrc = nullptr;
    ggml_tensor * msrc = nullptr;
    ggml_tensor * qpol = nullptr;   // polluting dispatch, full Br tile
    ggml_tensor * mpol = nullptr;

    // op inputs
    ggml_tensor * k      = nullptr;
    ggml_tensor * v      = nullptr;
    ggml_tensor * q[3]   = { nullptr, nullptr, nullptr };   // [0] -> N=5, [1] -> N=6, [2] -> N=16 polluter
    ggml_tensor * m[3]   = { nullptr, nullptr, nullptr };
    ggml_tensor * out[3] = { nullptr, nullptr, nullptr };
    ggml_cgraph * gf[3]  = { nullptr, nullptr, nullptr };
};

// byte offset of a logical element of `t` relative to the start of `src`
static size_t elem_off(const ggml_tensor * t, const ggml_tensor * src,
                       int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    const ptrdiff_t base = (const char *) t->data - (const char *) src->data;
    return (size_t) (base + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3]);
}

static void fa_build(fa_ctx & c, ggml_backend_t backend, layout_t layout) {
    c.backend = backend;
    c.layout  = layout;

    ggml_init_params ip = {
        /*.mem_size   =*/ 8*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    c.ctx = ggml_init(ip);
    GGML_ASSERT(c.ctx != nullptr);

    ggml_context * ctx = c.ctx;

    if (layout == LAYOUT_CACHE) {
        // Q source: [head_dim, n_head, n_tokens] contiguous, as produced by the
        // Q projection + rope; the op input is permute(0, 2, 1, 3) of it.
        c.qsrc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HSK, NH_Q, NB_HI, 1);
        c.qpol = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HSK, NH_Q, NB_POL, 1);
        // K/V cache: [n_embd_gqa, kv_size]; the op input is a permuted view.
        c.ksrc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HSK*NH_KV, 2*KVLEN, 1, 1);
        c.vsrc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HSV*NH_KV, 2*KVLEN, 1, 1);
    } else {
        c.qsrc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HSK, NB_HI,  NH_Q, 1);
        c.qpol = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HSK, NB_POL, NH_Q, 1);
        c.ksrc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HSK, 2*KVLEN, NH_KV, 1);
        c.vsrc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HSV, 2*KVLEN, NH_KV, 1);
    }
    c.msrc = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KVLEN, NB_HI,  1, 1);
    c.mpol = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KVLEN, NB_POL, 1, 1);

    ggml_set_name(c.qsrc, "qsrc");
    ggml_set_name(c.ksrc, "ksrc");
    ggml_set_name(c.vsrc, "vsrc");
    ggml_set_name(c.msrc, "msrc");
    ggml_set_name(c.qpol, "qpol");
    ggml_set_name(c.mpol, "mpol");

    if (layout == LAYOUT_CACHE) {
        ggml_tensor * kview = ggml_view_4d(ctx, c.ksrc, HSK, NH_KV, KVLEN, 1,
                                           ggml_row_size(GGML_TYPE_F16, HSK),
                                           ggml_row_size(GGML_TYPE_F16, HSK*NH_KV),
                                           ggml_row_size(GGML_TYPE_F16, HSK*NH_KV*2*KVLEN), 0);
        ggml_tensor * vview = ggml_view_4d(ctx, c.vsrc, HSV, NH_KV, KVLEN, 1,
                                           ggml_row_size(GGML_TYPE_F16, HSV),
                                           ggml_row_size(GGML_TYPE_F16, HSV*NH_KV),
                                           ggml_row_size(GGML_TYPE_F16, HSV*NH_KV*2*KVLEN), 0);
        c.k = ggml_permute(ctx, kview, 0, 2, 1, 3);
        c.v = ggml_permute(ctx, vview, 0, 2, 1, 3);
    } else {
        c.k = ggml_view_4d(ctx, c.ksrc, HSK, KVLEN, NH_KV, 1,
                           c.ksrc->nb[1], c.ksrc->nb[2], c.ksrc->nb[3], 0);
        c.v = ggml_view_4d(ctx, c.vsrc, HSV, KVLEN, NH_KV, 1,
                           c.vsrc->nb[1], c.vsrc->nb[2], c.vsrc->nb[3], 0);
    }

    const int64_t nbs[3] = { NB_LO, NB_HI, NB_POL };
    for (int j = 0; j < 3; ++j) {
        const int64_t   n  = nbs[j];
        ggml_tensor * qp = j == 2 ? c.qpol : c.qsrc;
        ggml_tensor * mp = j == 2 ? c.mpol : c.msrc;

        if (layout == LAYOUT_CACHE) {
            ggml_tensor * qv = ggml_view_4d(ctx, qp, HSK, NH_Q, n, 1,
                                            qp->nb[1], qp->nb[2], qp->nb[3], 0);
            c.q[j] = ggml_permute(ctx, qv, 0, 2, 1, 3);
        } else {
            c.q[j] = ggml_view_4d(ctx, qp, HSK, n, NH_Q, 1,
                                  qp->nb[1], qp->nb[2], qp->nb[3], 0);
        }

        // the mask must stay contiguous
        const size_t mb1 = mp->nb[1];
        c.m[j] = ggml_view_4d(ctx, mp, KVLEN, n, 1, 1, mb1, mb1*n, mb1*n, 0);

        c.out[j] = ggml_flash_attn_ext(ctx, c.q[j], c.k, c.v, c.m[j], 1.0f/16.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(c.out[j], GGML_PREC_F32);
        ggml_set_name(c.out[j], j == 0 ? "out_lo" : (j == 1 ? "out_hi" : "out_pol"));
    }

    c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, backend);
    GGML_ASSERT(c.buf != nullptr);

    for (int j = 0; j < 3; ++j) {
        c.gf[j] = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(c.gf[j], c.out[j]);
    }
}

static void fa_free(fa_ctx & c) {
    if (c.buf) ggml_backend_buffer_free(c.buf);
    if (c.ctx) ggml_free(c.ctx);
    c.buf = nullptr;
    c.ctx = nullptr;
}

// Q and the mask do not depend on the KV-tail mode; they are filled once per L.
static void fa_fill_q(fa_ctx & c) {
    std::vector<uint8_t> s(ggml_nbytes(c.qsrc), 0);
    for (int64_t h = 0; h < NH_Q; ++h) {
        for (int64_t r = 0; r < NB_HI; ++r) {
            for (int64_t d = 0; d < HSK; ++d) {
                const float val = hashf(((uint64_t) h << 40) ^ ((uint64_t) r << 24) ^ (uint64_t) d);
                memcpy(s.data() + elem_off(c.q[1], c.qsrc, d, r, h, 0), &val, sizeof(float));
            }
        }
    }
    ggml_backend_tensor_set(c.qsrc, s.data(), 0, s.size());

    // the polluting dispatch uses much larger query values so that any leak of
    // its shared-memory tile into a later partial tile is unmistakable
    std::vector<uint8_t> p(ggml_nbytes(c.qpol), 0);
    for (int64_t h = 0; h < NH_Q; ++h) {
        for (int64_t r = 0; r < NB_POL; ++r) {
            for (int64_t d = 0; d < HSK; ++d) {
                const float val = 64.0f * hashf(0x3Cull ^ ((uint64_t) h << 40) ^ ((uint64_t) r << 24) ^ (uint64_t) d);
                memcpy(p.data() + elem_off(c.q[2], c.qpol, d, r, h, 0), &val, sizeof(float));
            }
        }
    }
    ggml_backend_tensor_set(c.qpol, p.data(), 0, p.size());
}

// Diagnostic: multiply the K/V values written at physical positions >= g_gain_from
// by g_gain.  Used to measure the weight with which a mask-suppressed KV cell
// leaks into a live query row (the leak, if any, is linear in the cell value).
static float   g_gain      = 1.0f;
static int64_t g_gain_from = -1;
static bool    g_tailval_set = false;
static float   g_tailval   = 0.0f;
// --extra <pos>: both runs write exactly the same KV cells [0,L); the SECOND run
// additionally overwrites V at physical position <pos> with g_tailval.  Isolates
// which mask-suppressed KV position leaks into the compared live row.
static int64_t g_extra_pos = -1;
static bool    g_extra_on  = false;

// K / V.  Physical KV positions [0, n_written) hold data, the rest hold either
// zeroes (as a freshly cleared KV cache does) or, when tail_id != 0, distinct
// finite values so that a non-inert masked tail shows up.
static void fa_fill_kv(fa_ctx & c, int64_t nk_written, int64_t nv_written, int tail_id) {
    std::vector<uint8_t> ks(ggml_nbytes(c.ksrc), 0);
    std::vector<uint8_t> vs(ggml_nbytes(c.vsrc), 0);
    std::vector<float>       f(HSK);
    std::vector<ggml_fp16_t> h16(HSK);
    for (int64_t hh = 0; hh < NH_KV; ++hh) {
        for (int64_t p = 0; p < KVLEN; ++p) {
            const bool live_k = p < nk_written;
            const bool live_v = p < nv_written;
            if (live_k || tail_id != 0) {
                const uint64_t seed = live_k ? 0x51ull : (0x51ull ^ ((uint64_t) tail_id << 56));
                for (int64_t d = 0; d < HSK; ++d) {
                    f[d] = hashf(seed ^ ((uint64_t) hh << 44) ^ ((uint64_t) p << 24) ^ (uint64_t) d);
                    if (g_gain_from >= 0 && p >= g_gain_from) f[d] = g_tailval_set ? g_tailval : f[d] * g_gain;
                }
                ggml_fp32_to_fp16_row(f.data(), h16.data(), HSK);
                for (int64_t d = 0; d < HSK; ++d) {
                    memcpy(ks.data() + elem_off(c.k, c.ksrc, d, p, hh, 0), &h16[d], sizeof(ggml_fp16_t));
                }
            }
            if (live_v || tail_id != 0) {
                const uint64_t seedv = live_v ? 0xA7ull : (0xA7ull ^ ((uint64_t) tail_id << 56));
                for (int64_t d = 0; d < HSV; ++d) {
                    f[d] = hashf(seedv ^ ((uint64_t) hh << 44) ^ ((uint64_t) p << 24) ^ (uint64_t) d);
                    if (g_gain_from >= 0 && p >= g_gain_from) f[d] = g_tailval_set ? g_tailval : f[d] * g_gain;
                }
                ggml_fp32_to_fp16_row(f.data(), h16.data(), HSV);
                for (int64_t d = 0; d < HSV; ++d) {
                    memcpy(vs.data() + elem_off(c.v, c.vsrc, d, p, hh, 0), &h16[d], sizeof(ggml_fp16_t));
                }
            }
        }
    }
    if (g_extra_on && g_extra_pos >= 0) {
        ggml_fp16_t hv = ggml_fp32_to_fp16(g_tailval);
        for (int64_t hh = 0; hh < NH_KV; ++hh) {
            for (int64_t d = 0; d < HSV; ++d) {
                memcpy(vs.data() + elem_off(c.v, c.vsrc, d, g_extra_pos, hh, 0), &hv, sizeof(ggml_fp16_t));
            }
        }
    }
    ggml_backend_tensor_set(c.ksrc, ks.data(), 0, ks.size());
    ggml_backend_tensor_set(c.vsrc, vs.data(), 0, vs.size());
}

// causal mask: query row TROW has cutoff L, row r has cutoff L - TROW + r
static void fa_fill_mask(fa_ctx & c, int L) {
    std::vector<uint8_t>     s(ggml_nbytes(c.msrc), 0);
    std::vector<float>       f(KVLEN);
    std::vector<ggml_fp16_t> h16(KVLEN);
    for (int64_t r = 0; r < NB_HI; ++r) {
        const int64_t cut = (int64_t) L - TROW + r;
        for (int64_t kk = 0; kk < KVLEN; ++kk) {
            f[kk] = kk < cut ? 0.0f : -INFINITY;
        }
        ggml_fp32_to_fp16_row(f.data(), h16.data(), KVLEN);
        for (int64_t kk = 0; kk < KVLEN; ++kk) {
            memcpy(s.data() + elem_off(c.m[1], c.msrc, kk, r, 0, 0), &h16[kk], sizeof(ggml_fp16_t));
        }
    }
    ggml_backend_tensor_set(c.msrc, s.data(), 0, s.size());

    // the polluter attends to everything
    std::vector<uint8_t> p(ggml_nbytes(c.mpol), 0);
    for (int64_t kk = 0; kk < KVLEN; ++kk) {
        f[kk] = 0.0f;
    }
    ggml_fp32_to_fp16_row(f.data(), h16.data(), KVLEN);
    for (int64_t r = 0; r < NB_POL; ++r) {
        for (int64_t kk = 0; kk < KVLEN; ++kk) {
            memcpy(p.data() + elem_off(c.m[2], c.mpol, kk, r, 0, 0), &h16[kk], sizeof(ggml_fp16_t));
        }
    }
    ggml_backend_tensor_set(c.mpol, p.data(), 0, p.size());
}

static void fa_dispatch(fa_ctx & c, int j) {
    const ggml_status st = ggml_backend_graph_compute(c.backend, c.gf[j]);
    GGML_ASSERT(st == GGML_STATUS_SUCCESS);
}

static void fa_run(fa_ctx & c, int j, std::vector<float> & row) {
    fa_dispatch(c, j);
    row.resize(ROW_ELEMS);
    ggml_backend_tensor_get(c.out[j], row.data(),
                            (size_t) TROW*ROW_ELEMS*sizeof(float), ROW_ELEMS*sizeof(float));
}

// One reference/rollback pair: the short (N=5) dispatch then the long (N=6) one,
// with the KV cache in the state each of them would see in production.
static int g_jlo = 0;   // graph used for the "short" side of the pair
static int g_jhi = 1;   // graph used for the "long"  side of the pair

static void fa_run_pair(fa_ctx & c, int L, kvtail_t kvtail, int tailsel, bool pollute,
                        std::vector<float> & r_lo, std::vector<float> & r_hi) {
    const int tail_id = kvtail == KVTAIL_POISON ? 1 : 0;
    g_gain_from = L;
    g_extra_on  = false;

    if (kvtail == KVTAIL_SHARED) {
        fa_fill_kv(c, KVLEN, KVLEN, 0);
    } else {
        fa_fill_kv(c, L, L, tail_id);
    }
    if (pollute) fa_dispatch(c, 2);
    fa_run(c, g_jlo, r_lo);

    g_extra_on = true;
    if (kvtail != KVTAIL_SHARED) {
        if (g_extra_pos >= 0) {
            fa_fill_kv(c, L, L, tail_id);   // identical KV, plus the single extra V cell
        } else {
            fa_fill_kv(c, (tailsel & 1) ? L + 1 : L, (tailsel & 2) ? L + 1 : L,
                       tail_id == 0 ? 0 : 2);
        }
    }
    if (pollute) fa_dispatch(c, 2);
    fa_run(c, g_jhi, r_hi);
}

struct cmp_res {
    size_t ndiff = 0;
    double maxabs = 0.0;
};

static bool g_dump = false;

// print the first few differing elements as (head, hsv) with raw bits
static void dump_diff(const std::vector<float> & a, const std::vector<float> & b, int L) {
    int shown = 0;
    for (size_t i = 0; i < a.size() && shown < 10; ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) == 0) continue;
        uint32_t ua, ub;
        memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
        printf("    L=%d diff #%d: head=%d d=%d  a=%.9g (0x%08x)  b=%.9g (0x%08x)  d=%.3g\n",
               L, shown, (int)(i / HSV), (int)(i % HSV), a[i], ua, b[i], ub, (double) b[i] - (double) a[i]);
        shown++;
    }
}

static cmp_res compare(const std::vector<float> & a, const std::vector<float> & b) {
    cmp_res r;
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            r.ndiff++;
        }
        const double d = std::fabs((double) a[i] - (double) b[i]);
        if (d > r.maxabs) r.maxabs = d;
    }
    return r;
}

static ggml_backend_dev_t find_dev(const char * prefix) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(d), prefix, strlen(prefix)) == 0) {
            return d;
        }
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    const char * dev_prefix = "Vulkan";
    int  Lmin       = 27;
    int  Lmax       = 35;
    int  reps       = 1;
    bool do_view    = true;
    bool do_cache   = true;
    kvtail_t kvtail = KVTAIL_SHARED;
    int  tailsel    = 3;   // bit0 = K differs, bit1 = V differs
    bool pollute    = false;
    bool trow_set   = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-b" && i + 1 < argc) {
            dev_prefix = argv[++i];
        } else if (a == "--lmin" && i + 1 < argc) {
            Lmin = atoi(argv[++i]);
        } else if (a == "--lmax" && i + 1 < argc) {
            Lmax = atoi(argv[++i]);
        } else if (a == "--reps" && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (a == "--layout" && i + 1 < argc) {
            const std::string l = argv[++i];
            do_view  = (l == "view"  || l == "both");
            do_cache = (l == "cache" || l == "both");
        } else if (a == "--kvtail" && i + 1 < argc) {
            const std::string t = argv[++i];
            kvtail = t == "prod" ? KVTAIL_PROD : (t == "poison" ? KVTAIL_POISON : KVTAIL_SHARED);
        } else if (a == "--tailsel" && i + 1 < argc) {
            const std::string t = argv[++i];
            tailsel = t == "k" ? 1 : (t == "v" ? 2 : (t == "none" ? 0 : 3));
        } else if (a == "--tailval" && i + 1 < argc) {
            g_tailval_set = true;
            g_tailval = (float) atof(argv[++i]);
        } else if (a == "--gain" && i + 1 < argc) {
            g_gain = (float) atof(argv[++i]);
        } else if (a == "--pair" && i + 1 < argc) {
            const std::string t = argv[++i];
            g_jlo = t == "hihi" ? 1 : 0;
            g_jhi = t == "lolo" ? 0 : 1;
        } else if (a == "--extra" && i + 1 < argc) {
            g_extra_pos = atoi(argv[++i]);
        } else if (a == "--nlo" && i + 1 < argc) {
            NB_LO = atoi(argv[++i]);
        } else if (a == "--nhi" && i + 1 < argc) {
            NB_HI = atoi(argv[++i]);
        } else if (a == "--trow" && i + 1 < argc) {
            TROW = atoi(argv[++i]);
            trow_set = true;
        } else if (a == "--dump") {
            g_dump = true;
        } else if (a == "--pollute") {
            pollute = true;
        } else {
            fprintf(stderr,
                    "usage: %s [-b <backend-name-prefix>] [--lmin L] [--lmax L] [--reps N] "
                    "[--layout view|cache|both] [--kvtail shared|prod|poison] [--tailsel k|v|kv] [--pollute] "
                    "[--gain F] [--pair lohi|lolo|hihi] [--nlo N] [--nhi N] [--trow R]\n", argv[0]);
            return 2;
        }
    }

    if (!trow_set) {
        TROW = NB_LO - 1;   // last row of the short dispatch
    }
    if (NB_LO < 1 || NB_HI < NB_LO || TROW < 0 || TROW >= NB_LO || NB_HI > NB_POL) {
        fprintf(stderr, "bad --nlo/--nhi/--trow (%d/%d/%d): need 1 <= nlo <= nhi <= %d and 0 <= trow < nlo\n",
                (int) NB_LO, (int) NB_HI, (int) TROW, (int) NB_POL);
        return 2;
    }

    ggml_backend_dev_t dev_gpu = find_dev(dev_prefix);
    if (dev_gpu == nullptr) {
        fprintf(stderr, "no backend device with prefix '%s' found\n", dev_prefix);
        return 2;
    }
    ggml_backend_t backend_gpu = ggml_backend_dev_init(dev_gpu, nullptr);
    GGML_ASSERT(backend_gpu != nullptr);

    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(backend_cpu != nullptr);

    printf("device : %s (%s)\n", ggml_backend_dev_name(dev_gpu), ggml_backend_dev_description(dev_gpu));
    printf("env    : GGML_VK_FA_NO_GQA=%s GGML_VK_FA_NO_SPLIT_K=%s GGML_VK_DISABLE_COOPMAT=%s\n",
           getenv("GGML_VK_FA_NO_GQA")       ? "1" : "-",
           getenv("GGML_VK_FA_NO_SPLIT_K")   ? "1" : "-",
           getenv("GGML_VK_DISABLE_COOPMAT") ? "1" : "-");
    printf("geom   : hsk=hsv=%d nh_kv=%d nh_q=%d gqa=%d kv=%d N=%d vs %d target row=%d\n",
           (int) HSK, (int) NH_KV, (int) NH_Q, (int) GQA, (int) KVLEN,
           (int) NB_LO, (int) NB_HI, (int) TROW);
    printf("opts   : kvtail=%s tailsel=%s pollute=%s reps=%d gain=%g pair=%d/%d\n", kvtail_name(kvtail),
           tailsel == 1 ? "k" : (tailsel == 2 ? "v" : (tailsel == 0 ? "none" : "kv")), pollute ? "yes" : "no", reps, g_gain, g_jlo, g_jhi);
    if (g_tailval_set) printf("         tailval=%g (constant written into the extra KV cell)\n", g_tailval);

    int n_cases = 0;
    int n_fail  = 0;
    int n_race  = 0;

    for (int li = 0; li < 2; ++li) {
        const layout_t layout = li == 0 ? LAYOUT_VIEW : LAYOUT_CACHE;
        if (layout == LAYOUT_VIEW  && !do_view)  continue;
        if (layout == LAYOUT_CACHE && !do_cache) continue;

        fa_ctx gpu, cpu;
        fa_build(gpu, backend_gpu, layout);
        fa_build(cpu, backend_cpu, layout);

        printf("\nlayout %s  (q_stride=%zu nbq2=%zu  k_stride=%zu nbk2=%zu)\n",
               layout_name(layout),
               gpu.q[1]->nb[1]/sizeof(float), gpu.q[1]->nb[2],
               gpu.k->nb[1]/sizeof(ggml_fp16_t), gpu.k->nb[2]);
        printf("  %-4s %-10s %8s %12s %10s %12s %12s\n",
               "L", "Nlo==Nhi", "ndiff", "max|d|", "stable", "max|d| cpulo", "max|d| cpuhi");

        std::vector<float> g5, g6, g5b, g6b, c5, c6;

        fa_fill_q(gpu);
        fa_fill_q(cpu);

        for (int L = Lmin; L <= Lmax; ++L) {
            fa_fill_mask(gpu, L);
            fa_fill_mask(cpu, L);

            bool identical = true;
            bool stable    = true;
            cmp_res worst;
            cmp_res vs_cpu5, vs_cpu6;

            fa_run_pair(cpu, L, kvtail, tailsel, false, c5, c6);

            for (int rep = 0; rep < reps; ++rep) {
                fa_run_pair(gpu, L, kvtail, tailsel, pollute, g5,  g6);
                fa_run_pair(gpu, L, kvtail, tailsel, pollute, g5b, g6b);

                const cmp_res r  = compare(g5, g6);
                const cmp_res s5 = compare(g5, g5b);
                const cmp_res s6 = compare(g6, g6b);

                if (r.ndiff)  identical = false;
                if (r.ndiff && g_dump && rep == 0) dump_diff(g5, g6, L);
                if (s5.ndiff || s6.ndiff) stable = false;
                if (r.ndiff > worst.ndiff)   worst.ndiff  = r.ndiff;
                if (r.maxabs > worst.maxabs) worst.maxabs = r.maxabs;

                const cmp_res a5 = compare(g5, c5);
                const cmp_res a6 = compare(g6, c6);
                if (a5.maxabs > vs_cpu5.maxabs) vs_cpu5.maxabs = a5.maxabs;
                if (a6.maxabs > vs_cpu6.maxabs) vs_cpu6.maxabs = a6.maxabs;
            }

            n_cases++;
            if (!identical) n_fail++;
            if (!stable)    n_race++;

            printf("  %-4d %-10s %5zu/%zu %12.6g %10s %12.6g %12.6g\n",
                   L, identical ? "IDENTICAL" : "DIFFERENT",
                   worst.ndiff, ROW_ELEMS, worst.maxabs,
                   stable ? "yes" : "NO",
                   vs_cpu5.maxabs, vs_cpu6.maxabs);
        }

        fa_free(gpu);
        fa_free(cpu);
    }

    printf("\nsummary: %d/%d cases differ between N=%d and N=%d (row %d); %d cases were unstable across repeats\n",
           n_fail, n_cases, (int) NB_LO, (int) NB_HI, (int) TROW, n_race);

    ggml_backend_free(backend_gpu);
    ggml_backend_free(backend_cpu);

    return n_fail == 0 && n_race == 0 ? 0 : 1;
}
