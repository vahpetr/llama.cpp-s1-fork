// [TAG_ROPE_PRECISE] test-rope-precision: numerical error of GGML_OP_ROPE at large positions against an exact reference,
// for Qwen3.8-Flash-Next's rope (qwen4exp: IMROPE, 64 rotated dims, freq_base 1e7, sections [11,11,10,0], native and
// YaRN x4 over 262 144 with llama_context's attention-factor handling).
//
//   test-rope-precision                 every backend (CPU, Vulkan0 if present): reference self-check + error report; exit 0
//                                       unless the reference or a backend is grossly wrong (NMSE > 1e-3)
//   test-rope-precision --check         the same, and FAIL unless NMSE <= 1e-9 and the relative-position dot error <= 1e-6
//                                       on every backend (meant for LLAMA_ROPE_PRECISE=1)
//   test-rope-precision --dump FILE     write the raw outputs of a fixed op matrix (every rope mode / type / fusion) for a
//                                       cross-build byte-identity comparison; prints one hash per case
//   test-rope-precision --perf          time rope_multi at the model's prefill/decode shapes
//   options: --backend NAME (CPU | Vulkan0 | all, default all), --tsv FILE (per-pair rows), --check-ref FILE (compare the
//            reference with arbitrary-precision values: lines "cfg p j theta_mod_2pi cos sin")
//
// The reference: theta_j(p) = p * freq_base^(-2j/n_dims) * (freq_scale*(1-mix_j) + mix_j), mix_j = the YaRN ramp over the
// op's corr dims (ggml_rope_yarn_corr_dims, integers) * ext_factor; mscale = attn_factor * (1 + 0.1 ln(1/freq_scale)) when
// ext_factor != 0; out = mscale * R(theta) x. Evaluated in long double (x87 80-bit, 64-bit mantissa): error <= 1e-12 rad
// at p = 2^20 (checked against 60-digit decimal arithmetic with --check-ref).

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

typedef long double ld;

static const ld PI_L     = 3.141592653589793238462643383279502884L;
static const ld TWO_PI_L = 6.283185307179586476925286766559005768L;

struct rope_cfg {
    const char * name;
    int   D;          // head size
    int   n_dims;     // rotated dims
    int   mode;
    int   sections[4];
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    int   n_ctx_orig;
    float beta_fast;
    float beta_slow;
};

// llama_context's YaRN attention factor for a model without rope_yarn_log_mul (qwen4exp), same float operations
static float fork_yarn_attn_factor(float freq_scale) {
    volatile float fs = freq_scale; // keep it a run-time computation like the fork's
    const float factor = 1.0f / fs;
    const float mscale = 1.0f;
    float af = factor <= 1.0f ? 1.0f : (0.1f * mscale * logf(factor) + 1.0f);
    af *= 1.0f / (1.0f + 0.1f * logf(factor));
    af *= 1.0f; // hparams.rope_attn_factor: absent in the qwen4exp GGUF
    return af;
}

static rope_cfg make_cfg(const char * name, int D, bool yarn) {
    rope_cfg c {};
    c.name        = name;
    c.D           = D;
    c.n_dims      = 64;
    c.mode        = GGML_ROPE_TYPE_IMROPE;
    c.sections[0] = 11; c.sections[1] = 11; c.sections[2] = 10; c.sections[3] = 0;
    c.freq_base   = 10000000.0f;
    c.freq_scale  = yarn ? 0.25f : 1.0f;
    c.ext_factor  = yarn ? 1.0f : 0.0f;
    c.attn_factor = yarn ? fork_yarn_attn_factor(0.25f) : 1.0f;
    c.n_ctx_orig  = 262144;
    c.beta_fast   = 32.0f;
    c.beta_slow   = 1.0f;
    return c;
}

// ---------------------------------------------------------------------------------------------------------------------
// reference

static int mrope_component(const rope_cfg & c, int j) {
    if (!(c.mode & GGML_ROPE_TYPE_MROPE)) {
        return 0;
    }
    const int * s = c.sections;
    const int sect_dims = s[0] + s[1] + s[2] + s[3];
    const int sec_w     = s[0] + s[1];
    const int sector    = j % sect_dims;
    if (c.mode == GGML_ROPE_TYPE_IMROPE) {
        if (sector % 3 == 1 && sector < 3 * s[1]) return 1;
        if (sector % 3 == 2 && sector < 3 * s[2]) return 2;
        if (sector % 3 == 0 && sector < 3 * s[0]) return 0;
        return 3;
    }
    if (sector < s[0])         return 0;
    if (sector < sec_w)        return 1;
    if (sector < sec_w + s[2]) return 2;
    return 3;
}

struct rope_ref {
    rope_cfg        c;
    float           corr[2];
    std::vector<ld> freq;  // radians per position, pair j (YaRN mix included)
    std::vector<int> comp; // position component of pair j
    ld              mscale;
};

static rope_ref make_ref(const rope_cfg & c) {
    rope_ref r;
    r.c = c;
    ggml_rope_yarn_corr_dims(c.n_dims, c.n_ctx_orig, c.freq_base, c.beta_fast, c.beta_slow, r.corr);
    const int np = c.n_dims / 2;
    r.freq.resize(np);
    r.comp.resize(np);
    for (int j = 0; j < np; ++j) {
        const ld f = powl((ld) c.freq_base, -(ld) (2 * j) / (ld) c.n_dims);
        ld scale = c.freq_scale;
        if (c.ext_factor != 0.0f) {
            const ld y    = ((ld) j - (ld) r.corr[0]) / std::max((ld) 0.001L, (ld) r.corr[1] - (ld) r.corr[0]);
            const ld ramp = 1.0L - std::min((ld) 1.0L, std::max((ld) 0.0L, y));
            const ld mix  = ramp * (ld) c.ext_factor;
            scale = (ld) c.freq_scale * (1.0L - mix) + mix;
        }
        r.freq[j] = f * scale;
        r.comp[j] = mrope_component(c, j);
    }
    r.mscale = (ld) c.attn_factor * (c.ext_factor != 0.0f ? 1.0L + 0.1L * logl(1.0L / (ld) c.freq_scale) : 1.0L);
    return r;
}

// exact rotation of one head (NEOX/MROPE layout, n_offs 0): pairs (j, j + n_dims/2)
static void ref_rotate(const rope_ref & r, const int32_t pos4[4], const float * x, ld * y) {
    const int nh = r.c.n_dims / 2;
    for (int k = 0; k < r.c.D; ++k) {
        y[k] = x[k];
    }
    for (int j = 0; j < nh; ++j) {
        const ld th = (ld) pos4[r.comp[j]] * r.freq[j];
        const ld cs = cosl(th) * r.mscale;
        const ld sn = sinl(th) * r.mscale;
        const ld x0 = x[j], x1 = x[j + nh];
        y[j]      = x0 * cs - x1 * sn;
        y[j + nh] = x0 * sn + x1 * cs;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// emulations of the fp32 rope other stacks use (angle per (p, j); cos/sin treated as exact unless bf16 is asked)

static float bf16_round(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; // round to nearest even
    float r;
    memcpy(&r, &u, 4);
    return r;
}

// HF transformers / Megatron / vLLM: inv_freq = 1 / base**(arange(0,dim,2).float()/dim) in fp32 (YaRN: interpolation
// 1/(factor*pos_freqs), ramp mask in fp32), angle = fp32(p) * inv_freq (one fp32 rounding), cos/sin in fp32
static float hf_inv_freq(const rope_ref & r, int j) {
    const rope_cfg & c = r.c;
    const float pos_freq = powf(c.freq_base, (float) (2 * j) / (float) c.n_dims);
    const float extrap   = 1.0f / pos_freq;
    if (c.ext_factor == 0.0f) {
        return c.freq_scale == 1.0f ? extrap : extrap * c.freq_scale;
    }
    const float factor = 1.0f / c.freq_scale;
    const float interp = 1.0f / (factor * pos_freq);
    float lo = r.corr[0], hi = r.corr[1];
    if (lo == hi) {
        hi += 0.001f;
    }
    const float lin = std::min(1.0f, std::max(0.0f, ((float) j - lo) / (hi - lo)));
    const float ef  = 1.0f - lin; // inv_freq_extrapolation_factor
    return interp * (1.0f - ef) + extrap * ef;
}

// ---------------------------------------------------------------------------------------------------------------------
// backend helpers

struct host_input {
    ggml_tensor *        t;
    std::vector<uint8_t> data;
};

static std::vector<uint8_t> run_graph(ggml_backend_t be,
        const std::function<ggml_tensor * (ggml_context *, std::vector<host_input> &)> & build) {
    ggml_init_params ip = { 256 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<host_input> inputs;
    ggml_tensor * out = build(ctx, inputs);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf == nullptr) {
        fprintf(stderr, "alloc failed\n");
        exit(2);
    }
    for (auto & in : inputs) {
        GGML_ASSERT(in.data.size() == ggml_nbytes(in.t));
        ggml_backend_tensor_set(in.t, in.data.data(), 0, in.data.size());
    }
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        exit(2);
    }
    std::vector<uint8_t> res(ggml_nbytes(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

template <typename T>
static std::vector<uint8_t> to_bytes(const std::vector<T> & v) {
    std::vector<uint8_t> b(v.size() * sizeof(T));
    memcpy(b.data(), v.data(), b.size());
    return b;
}

// rope_multi of x [D, H, T] (f32) at 4-component positions pos4 [4*T] (component-major, as llama's input)
static std::vector<float> backend_rope(ggml_backend_t be, const rope_cfg & c, const std::vector<float> & x, int H, int T,
        const std::vector<int32_t> & pos) {
    auto b = run_graph(be, [&](ggml_context * ctx, std::vector<host_input> & in) {
        ggml_tensor * tx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.D, H, T);
        ggml_tensor * tp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * (int64_t) T);
        in.push_back({ tx, to_bytes(x) });
        in.push_back({ tp, to_bytes(pos) });
        int sections[4] = { c.sections[0], c.sections[1], c.sections[2], c.sections[3] };
        return ggml_rope_multi(ctx, tx, tp, nullptr, c.n_dims, sections, c.mode, c.n_ctx_orig, c.freq_base, c.freq_scale,
                c.ext_factor, c.attn_factor, c.beta_fast, c.beta_slow);
    });
    std::vector<float> y(b.size() / 4);
    memcpy(y.data(), b.data(), b.size());
    return y;
}

static std::vector<int32_t> text_pos4(const std::vector<int64_t> & p) {
    const size_t T = p.size();
    std::vector<int32_t> pos(4 * T);
    for (size_t i = 0; i < T; ++i) {
        pos[i] = pos[T + i] = pos[2 * T + i] = (int32_t) p[i];
        pos[3 * T + i] = 0; // llm_graph_input_pos: the 4th component of a text token is 0
    }
    return pos;
}

// ---------------------------------------------------------------------------------------------------------------------
// error analysis

static const int W = 2048; // positions per window

struct window {
    const char * name;
    int64_t      end; // positions end-W .. end-1
};

static const window WINDOWS[] = {
    { "0",    W       },
    { "8k",   8192    },
    { "64k",  65536   },
    { "262k", 262144  },
    { "524k", 524288  },
    { "1M",   1048576 },
};
static const int NWIN = sizeof(WINDOWS) / sizeof(WINDOWS[0]);

struct pair_stats {
    double sum2 = 0, maxabs = 0;
    int    n = 0;
    void add(double e) { sum2 += e * e; maxabs = std::max(maxabs, std::fabs(e)); n++; }
    double rms() const { return n ? std::sqrt(sum2 / n) : 0; }
};

struct win_result {
    std::vector<pair_stats> pairs; // per j
    double nmse_full = 0, nmse_rot = 0, mag_max = 0;
};

static double wrap_err(ld a) {
    return (double) remainderl(a, TWO_PI_L);
}

static FILE * g_tsv = nullptr;

// measured = a backend; for the emulations (backend == nullptr) the angle comes from emu(p, j)
static std::vector<win_result> analyse(const char * bname, ggml_backend_t be, const rope_cfg & c,
        const std::function<ld(int64_t, int)> & emu) {
    const rope_ref r = make_ref(c);
    const int np = c.n_dims / 2;
    const int R  = 4;         // random heads
    const int H  = 1 + R;     // head 0 = unit vectors (x0 = 1, x1 = 0 per pair) → the angle
    const int T  = NWIN * W;

    std::vector<int64_t> p(T);
    for (int w = 0; w < NWIN; ++w) {
        for (int i = 0; i < W; ++i) {
            p[w * W + i] = WINDOWS[w].end - W + i;
        }
    }
    std::vector<float> x(be ? (size_t) c.D * H * T : 0);
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int t = 0; be && t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            float * xr = &x[((size_t) t * H + h) * c.D];
            for (int k = 0; k < c.D; ++k) {
                xr[k] = h == 0 ? (k < np ? 1.0f : 0.0f) : nd(rng);
            }
        }
    }
    std::vector<float> y;
    if (be) {
        y = backend_rope(be, c, x, H, T, text_pos4(p));
    }

    std::vector<win_result> res(NWIN);
    std::vector<ld> yr(c.D);
    for (int w = 0; w < NWIN; ++w) {
        win_result & wr = res[w];
        wr.pairs.resize(np);
        ld se_full = 0, sr_full = 0, se_rot = 0, sr_rot = 0;
        for (int i = 0; i < W; ++i) {
            const int t = w * W + i;
            const int32_t pos4[4] = { (int32_t) p[t], (int32_t) p[t], (int32_t) p[t], 0 };
            for (int j = 0; j < np; ++j) {
                const ld th = (ld) pos4[r.comp[j]] * r.freq[j];
                double e;
                if (be) {
                    const float * yu = &y[((size_t) t * H + 0) * c.D];
                    const ld a = atan2l((ld) yu[j + np], (ld) yu[j]);
                    e = wrap_err(a - th);
                    wr.mag_max = std::max(wr.mag_max, (double) fabsl(hypotl(yu[j], yu[j + np]) / r.mscale - 1.0L));
                } else {
                    e = wrap_err(emu(p[t], j) - th);
                }
                wr.pairs[j].add(e);
            }
            if (!be) {
                continue;
            }
            for (int h = 1; h < H; ++h) {
                const float * xr = &x[((size_t) t * H + h) * c.D];
                const float * yb = &y[((size_t) t * H + h) * c.D];
                ref_rotate(r, pos4, xr, yr.data());
                for (int k = 0; k < c.D; ++k) {
                    const ld d = (ld) yb[k] - yr[k];
                    se_full += d * d;
                    sr_full += yr[k] * yr[k];
                    if (k < c.n_dims) {
                        se_rot += d * d;
                        sr_rot += yr[k] * yr[k];
                    }
                }
            }
        }
        wr.nmse_full = be ? (double) (se_full / sr_full) : NAN;
        wr.nmse_rot  = be ? (double) (se_rot / sr_rot) : NAN;
        if (g_tsv) {
            for (int j = 0; j < np; ++j) {
                fprintf(g_tsv, "pair\t%s\t%s\t%s\t%d\t%.6e\t%.6e\n", bname, c.name, WINDOWS[w].name, j, wr.pairs[j].rms(),
                        wr.pairs[j].maxabs);
            }
        }
    }
    return res;
}

static void print_summary(const char * bname, const rope_cfg & c, const std::vector<win_result> & res, bool per_pair) {
    const int np = c.n_dims / 2;
    printf("\n[%s] %s (D %d, n_dims %d, fs %.2f, ext %.0f, attn_factor %.9g, mscale %.9Lg)\n", bname, c.name, c.D, c.n_dims,
            c.freq_scale, c.ext_factor, c.attn_factor, make_ref(c).mscale);
    printf("  %-5s %12s %4s %12s %12s %12s %10s\n", "pos", "max|dθ| rad", "@j", "rms|dθ| rad", "NMSE full", "NMSE rot", "|mag|-1");
    for (int w = 0; w < NWIN; ++w) {
        double mx = 0, s2 = 0;
        int jm = 0;
        for (int j = 0; j < np; ++j) {
            if (res[w].pairs[j].maxabs > mx) {
                mx = res[w].pairs[j].maxabs;
                jm = j;
            }
            s2 += res[w].pairs[j].sum2 / std::max(1, res[w].pairs[j].n);
        }
        printf("  %-5s %12.3e %4d %12.3e %12.3e %12.3e %10.2e\n", WINDOWS[w].name, mx, jm, std::sqrt(s2 / np),
                res[w].nmse_full, res[w].nmse_rot, res[w].mag_max);
    }
    if (per_pair) {
        printf("  max|dθ| per pair j at 262k / 524k / 1M (rad):\n");
        for (int j = 0; j < np; ++j) {
            printf("    j %2d  λ %12.1f  %10.3e %10.3e %10.3e\n", j, (double) (TWO_PI_L / make_ref(c).freq[j]),
                    res[3].pairs[j].maxabs, res[4].pairs[j].maxabs, res[5].pairs[j].maxabs);
        }
    }
}

// relative-position consistency: q at m, k at n = m - d; error of q·k against the exact value, / (|q||k|)
struct dot_result {
    double rms = 0, maxabs = 0;
};

static std::map<std::string, dot_result> dot_test(ggml_backend_t be, const rope_cfg & c, const char * bname) {
    const rope_ref r = make_ref(c);
    const int64_t DS[] = { 1, 1024, 65536 };
    const char *  RG[] = { "near0", "near1M" };
    std::map<std::string, dot_result> out;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int64_t d : DS) {
        for (int g = 0; g < 2; ++g) {
            const int T = 2 * W;
            std::vector<int64_t> p(T);
            for (int i = 0; i < W; ++i) {
                const int64_t m = g == 0 ? i + d : (1 << 20) - W + i;
                p[i]     = m;
                p[W + i] = m - d;
            }
            std::vector<float> x((size_t) c.D * T);
            for (auto & v : x) {
                v = nd(rng);
            }
            std::vector<float> y = backend_rope(be, c, x, 1, T, text_pos4(p));
            std::vector<ld> yq(c.D), yk(c.D);
            double s2 = 0, mx = 0;
            for (int i = 0; i < W; ++i) {
                const float * q  = &x[(size_t) i * c.D];
                const float * k  = &x[(size_t) (W + i) * c.D];
                const int32_t pq[4] = { (int32_t) p[i], (int32_t) p[i], (int32_t) p[i], 0 };
                const int32_t pk[4] = { (int32_t) p[W + i], (int32_t) p[W + i], (int32_t) p[W + i], 0 };
                ref_rotate(r, pq, q, yq.data());
                ref_rotate(r, pk, k, yk.data());
                ld dref = 0, db = 0, nq = 0, nk = 0;
                for (int e = 0; e < c.D; ++e) {
                    dref += yq[e] * yk[e];
                    db   += (ld) y[(size_t) i * c.D + e] * (ld) y[(size_t) (W + i) * c.D + e];
                    nq   += (ld) q[e] * q[e];
                    nk   += (ld) k[e] * k[e];
                }
                const double err = (double) ((db - dref) / sqrtl(nq * nk));
                s2 += err * err;
                mx = std::max(mx, std::fabs(err));
            }
            char key[64];
            snprintf(key, sizeof(key), "d=%" PRId64 " %s", d, RG[g]);
            out[key] = { std::sqrt(s2 / W), mx };
            if (g_tsv) {
                fprintf(g_tsv, "dot\t%s\t%s\t%s\t%.6e\t%.6e\n", bname, c.name, key, std::sqrt(s2 / W), mx);
            }
        }
    }
    return out;
}

// storage noise floors in the same units (q·k error / (|q||k|)): q8_0 K (target KV), f16 K (draft KV), bf16 q and k
// (training / vLLM activations)
static void storage_floors(int D) {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int N = 4096;
    double s_q8 = 0, s_f16 = 0, s_bf = 0;
    const auto * tt = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<float> q(D), k(D), kq(D);
    std::vector<uint8_t> blk(ggml_row_size(GGML_TYPE_Q8_0, D));
    for (int n = 0; n < N; ++n) {
        for (int e = 0; e < D; ++e) {
            q[e] = nd(rng);
            k[e] = nd(rng);
        }
        ggml_quantize_chunk(GGML_TYPE_Q8_0, k.data(), blk.data(), 0, 1, D, nullptr);
        tt->to_float(blk.data(), kq.data(), D);
        double d0 = 0, d8 = 0, d16 = 0, dbf = 0, nq = 0, nk = 0;
        for (int e = 0; e < D; ++e) {
            d0  += (double) q[e] * k[e];
            d8  += (double) q[e] * kq[e];
            d16 += (double) q[e] * ggml_fp16_to_fp32(ggml_fp32_to_fp16(k[e]));
            dbf += (double) bf16_round(q[e]) * bf16_round(k[e]);
            nq  += (double) q[e] * q[e];
            nk  += (double) k[e] * k[e];
        }
        const double s = std::sqrt(nq * nk);
        s_q8  += (d8 - d0) * (d8 - d0) / (s * s);
        s_f16 += (d16 - d0) * (d16 - d0) / (s * s);
        s_bf  += (dbf - d0) * (dbf - d0) / (s * s);
    }
    printf("  storage floors, rms q·k error / (|q||k|), D %d: q8_0 K %.3e | f16 K %.3e | bf16 q and k %.3e\n", D,
            std::sqrt(s_q8 / N), std::sqrt(s_f16 / N), std::sqrt(s_bf / N));
}

// ---------------------------------------------------------------------------------------------------------------------
// --check-ref: compare the long-double reference with arbitrary-precision values

static int check_ref(const char * path) {
    FILE * f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }
    std::map<std::string, rope_ref> refs;
    refs["native"] = make_ref(make_cfg("native", 256, false));
    refs["yarn"]   = make_ref(make_cfg("yarn", 256, true));
    char cfg[32];
    long long p;
    int j;
    char s_th[128], s_c[128], s_s[128];
    double max_th = 0, max_cs = 0;
    int n = 0;
    while (fscanf(f, "%31s %lld %d %127s %127s %127s", cfg, &p, &j, s_th, s_c, s_s) == 6) {
        const rope_ref & r = refs.at(cfg);
        const ld th  = (ld) p * r.freq[j];
        const ld thr = strtold(s_th, nullptr);
        const ld e   = remainderl(th - thr, TWO_PI_L);
        max_th = std::max(max_th, (double) fabsl(e));
        max_cs = std::max(max_cs, (double) std::max(fabsl(cosl(th) - strtold(s_c, nullptr)), fabsl(sinl(th) - strtold(s_s, nullptr))));
        n++;
    }
    fclose(f);
    printf("reference check vs %s: %d points, max |dθ| %.3e rad, max |d cos|,|d sin| %.3e\n", path, n, max_th, max_cs);
    return (n > 0 && max_th < 1e-11 && max_cs < 1e-11) ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------------------------------
// --dump: a fixed op matrix for byte-identity across builds

static uint64_t fnv1a(const std::vector<uint8_t> & b) {
    uint64_t h = 1469598103934665603ULL;
    for (uint8_t c : b) {
        h = (h ^ c) * 1099511628211ULL;
    }
    return h;
}

struct dump_rng {
    std::mt19937 rng;
    explicit dump_rng(uint32_t s) : rng(s) {}
    std::vector<uint8_t> f32(size_t n, float lo = -2.0f, float hi = 2.0f) {
        std::uniform_real_distribution<float> u(lo, hi);
        std::vector<float> v(n);
        for (auto & x : v) x = u(rng);
        return to_bytes(v);
    }
    std::vector<uint8_t> f16(size_t n) {
        std::uniform_real_distribution<float> u(-2.0f, 2.0f);
        std::vector<ggml_fp16_t> v(n);
        for (auto & x : v) x = ggml_fp32_to_fp16(u(rng));
        return to_bytes(v);
    }
    std::vector<uint8_t> pos(size_t n, int32_t maxp) {
        std::uniform_int_distribution<int32_t> u(0, maxp);
        std::vector<int32_t> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = i < 8 ? (int32_t) (i * (maxp / 8)) : u(rng);
        return to_bytes(v);
    }
};

typedef std::function<ggml_tensor * (ggml_context *, std::vector<host_input> &)> builder;

static std::vector<std::pair<std::string, builder>> dump_cases() {
    std::vector<std::pair<std::string, builder>> cs;
    const int MAXP = 1048575;
    auto rope_multi = [](ggml_context * ctx, ggml_tensor * a, ggml_tensor * p, ggml_tensor * ff, int n_dims, std::array<int, 4> s, int mode,
                         float fs, float ef, float af) {
        int sec[4] = { s[0], s[1], s[2], s[3] };
        return ggml_rope_multi(ctx, a, p, ff, n_dims, sec, mode, 262144, 1e7f, fs, ef, af, 32.0f, 1.0f);
    };
    // 1-3: qwen4exp IMROPE, f32 yarn / native, f16
    for (int v = 0; v < 3; ++v) {
        cs.push_back({ v == 0 ? "imrope_yarn_f32" : v == 1 ? "imrope_native_f32" : "imrope_yarn_f16",
            [=](ggml_context * ctx, std::vector<host_input> & in) {
                dump_rng g(100 + v);
                const ggml_type ty = v == 2 ? GGML_TYPE_F16 : GGML_TYPE_F32;
                ggml_tensor * a = ggml_new_tensor_3d(ctx, ty, 256, 4, 512);
                ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * 512);
                in.push_back({ a, v == 2 ? g.f16(256 * 4 * 512) : g.f32(256 * 4 * 512) });
                in.push_back({ p, g.pos(4 * 512, MAXP) });
                const bool yarn = v != 1;
                return rope_multi(ctx, a, p, nullptr, 64, { 11, 11, 10, 0 }, GGML_ROPE_TYPE_IMROPE, yarn ? 0.25f : 1.0f,
                        yarn ? 1.0f : 0.0f, yarn ? fork_yarn_attn_factor(0.25f) : 1.0f);
            } });
    }
    // 4: MROPE with distinct t/h/w positions
    cs.push_back({ "mrope_f32", [=](ggml_context * ctx, std::vector<host_input> & in) {
        dump_rng g(4);
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 2, 300);
        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * 300);
        in.push_back({ a, g.f32(128 * 2 * 300) });
        in.push_back({ p, g.pos(4 * 300, MAXP) });
        return rope_multi(ctx, a, p, nullptr, 64, { 8, 12, 12, 0 }, GGML_ROPE_TYPE_MROPE, 1.0f, 0.0f, 1.0f);
    } });
    // 5-8: NEOX / NORMAL through ggml_rope_ext, with freq factors, YaRN, partial dims, f16
    struct ext_case { const char * name; int mode; int D; int n_dims; bool ff; bool yarn; ggml_type ty; };
    const ext_case ecs[] = {
        { "neox_yarn_ff_f32",  GGML_ROPE_TYPE_NEOX,   128, 128, true,  true,  GGML_TYPE_F32 },
        { "neox_partial_f16",  GGML_ROPE_TYPE_NEOX,   128, 32,  false, false, GGML_TYPE_F16 },
        { "norm_f32",          GGML_ROPE_TYPE_NORMAL, 64,  64,  false, true,  GGML_TYPE_F32 },
        { "norm_ff_f16",       GGML_ROPE_TYPE_NORMAL, 64,  48,  true,  false, GGML_TYPE_F16 },
    };
    for (const ext_case & e : ecs) {
        cs.push_back({ e.name, [=](ggml_context * ctx, std::vector<host_input> & in) {
            dump_rng g(55 + e.D + e.n_dims + e.mode);
            ggml_tensor * a = ggml_new_tensor_3d(ctx, e.ty, e.D, 3, 400);
            ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 400);
            in.push_back({ a, e.ty == GGML_TYPE_F16 ? g.f16((size_t) e.D * 3 * 400) : g.f32((size_t) e.D * 3 * 400) });
            in.push_back({ p, g.pos(400, MAXP) });
            ggml_tensor * ff = nullptr;
            if (e.ff) {
                ff = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, e.n_dims / 2);
                in.push_back({ ff, g.f32(e.n_dims / 2, 1.0f, 4.0f) });
            }
            return ggml_rope_ext(ctx, a, p, ff, e.n_dims, e.mode, 4096, 1e6f, e.yarn ? 0.5f : 1.0f, e.yarn ? 1.0f : 0.0f,
                    e.yarn ? 0.9f : 1.0f, 32.0f, 1.0f);
        } });
    }
    // 9-10: RMS_NORM + MUL + ROPE (NEOX) (+ VIEW + SET_ROWS into an f16 cache) - the Vulkan fused shaders
    for (int sr = 0; sr < 2; ++sr) {
        cs.push_back({ sr ? "rms_mul_rope_setrows_f16" : "rms_mul_rope_f32", [=](ggml_context * ctx, std::vector<host_input> & in) {
            dump_rng g(90 + sr);
            const int D = 128, H = 4, T = 64;
            ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
            ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
            ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
            in.push_back({ a, g.f32((size_t) D * H * T) });
            in.push_back({ w, g.f32(D, 0.5f, 1.5f) });
            in.push_back({ p, g.pos(T, MAXP) });
            ggml_tensor * x = ggml_mul(ctx, ggml_rms_norm(ctx, a, 1e-6f), w);
            x = ggml_rope_ext(ctx, x, p, nullptr, D, GGML_ROPE_TYPE_NEOX, 262144, 1e7f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            if (!sr) {
                return x;
            }
            ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D * H, 2 * T);
            ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
            std::vector<ggml_fp16_t> z((size_t) D * H * 2 * T, ggml_fp32_to_fp16(0.0f));
            std::vector<int64_t> ix(T);
            for (int i = 0; i < T; ++i) ix[i] = 2 * i + 1;
            in.push_back({ cache, to_bytes(z) });
            in.push_back({ idx, to_bytes(ix) });
            ggml_tensor * v = ggml_view_2d(ctx, x, D * H, T, x->nb[2], 0);
            return ggml_set_rows(ctx, cache, v, idx);
        } });
    }
    // 11-12: ROPE + VIEW + SET_ROWS (f16 cache), NEOX and MROPE
    for (int m = 0; m < 2; ++m) {
        cs.push_back({ m ? "mrope_setrows_f16" : "neox_setrows_f16", [=](ggml_context * ctx, std::vector<host_input> & in) {
            dump_rng g(120 + m);
            const int D = 128, H = 2, T = 96;
            ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, T);
            ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (m ? 4 : 1) * T);
            in.push_back({ a, g.f32((size_t) D * H * T) });
            in.push_back({ p, g.pos((m ? 4 : 1) * T, MAXP) });
            ggml_tensor * x = m ? rope_multi(ctx, a, p, nullptr, 64, { 16, 8, 8, 0 }, GGML_ROPE_TYPE_MROPE, 0.25f, 1.0f, 1.0f)
                                : ggml_rope_ext(ctx, a, p, nullptr, 64, GGML_ROPE_TYPE_NEOX, 262144, 1e7f, 0.25f, 1.0f, 1.0f, 32.0f, 1.0f);
            ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, D * H, T + 5);
            ggml_tensor * idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
            std::vector<ggml_fp16_t> z((size_t) D * H * (T + 5), ggml_fp32_to_fp16(0.0f));
            std::vector<int64_t> ix(T);
            for (int i = 0; i < T; ++i) ix[i] = (T - 1 - i);
            in.push_back({ cache, to_bytes(z) });
            in.push_back({ idx, to_bytes(ix) });
            ggml_tensor * v = ggml_view_2d(ctx, x, D * H, T, x->nb[2], 0);
            return ggml_set_rows(ctx, cache, v, idx);
        } });
    }
    // 13: VISION
    cs.push_back({ "vision_f32", [=](ggml_context * ctx, std::vector<host_input> & in) {
        dump_rng g(13);
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 2, 200);
        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * 200);
        in.push_back({ a, g.f32(128 * 2 * 200) });
        in.push_back({ p, g.pos(4 * 200, 4096) });
        return rope_multi(ctx, a, p, nullptr, 64, { 16, 16, 0, 0 }, GGML_ROPE_TYPE_VISION, 1.0f, 0.0f, 1.0f);
    } });
    // 14: ROPE_BACK (NEOX, YaRN)
    cs.push_back({ "neox_back_f32", [=](ggml_context * ctx, std::vector<host_input> & in) {
        dump_rng g(14);
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 2, 200);
        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 200);
        in.push_back({ a, g.f32(128 * 2 * 200) });
        in.push_back({ p, g.pos(200, MAXP) });
        return ggml_rope_ext_back(ctx, a, p, nullptr, 64, GGML_ROPE_TYPE_NEOX, 262144, 1e7f, 0.25f, 1.0f, 1.0f, 32.0f, 1.0f);
    } });
    // 15: IMROPE with a rotated-dims offset
    cs.push_back({ "imrope_offs_f32", [=](ggml_context * ctx, std::vector<host_input> & in) {
        dump_rng g(15);
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, 128);
        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * 128);
        in.push_back({ a, g.f32(256 * 2 * 128) });
        in.push_back({ p, g.pos(4 * 128, MAXP) });
        ggml_tensor * x = rope_multi(ctx, a, p, nullptr, 64, { 11, 11, 10, 0 }, GGML_ROPE_TYPE_IMROPE, 0.25f, 1.0f, 1.0f);
        return ggml_rope_set_offset(x, 32);
    } });
    return cs;
}

static int run_dump(const std::vector<std::pair<std::string, ggml_backend_t>> & bes, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        return 2;
    }
    const auto cs = dump_cases();
    uint64_t all = 1469598103934665603ULL;
    for (const auto & be : bes) {
        for (const auto & c : cs) {
            std::vector<uint8_t> b = run_graph(be.second, c.second);
            fwrite(b.data(), 1, b.size(), f);
            const uint64_t h = fnv1a(b);
            all = (all ^ h) * 1099511628211ULL;
            printf("dump %-8s %-26s %8zu B  %016" PRIx64 "\n", be.first.c_str(), c.first.c_str(), b.size(), h);
        }
    }
    fclose(f);
    printf("dump total %016" PRIx64 "\n", all);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// --perf

static void run_perf(const std::string & bname, ggml_backend_t be) {
    struct shape { const char * name; int D; int H; int T; };
    const shape shapes[] = {
        { "trunk Q [256,24,2048]", 256, 24, 2048 },
        { "trunk K [256,2,2048]",  256, 2,  2048 },
        { "indexer q [128,4,2048]", 128, 4, 2048 },
        { "decode Q [256,24,1]",   256, 24, 1    },
    };
    const int CHAIN = 16, REPS = 10;
    for (const shape & s : shapes) {
        const rope_cfg c = make_cfg("yarn", s.D, true);
        ggml_init_params ip = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.D, s.H, s.T);
        ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * s.T);
        int sec[4] = { 11, 11, 10, 0 };
        ggml_tensor * x = a;
        for (int i = 0; i < CHAIN; ++i) {
            x = ggml_rope_multi(ctx, x, p, nullptr, 64, sec, c.mode, c.n_ctx_orig, c.freq_base, c.freq_scale, c.ext_factor,
                    c.attn_factor, c.beta_fast, c.beta_slow);
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, x);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        dump_rng g(5);
        auto xb = g.f32((size_t) s.D * s.H * s.T);
        ggml_backend_tensor_set(a, xb.data(), 0, xb.size());
        std::vector<int64_t> pp(s.T);
        for (int i = 0; i < s.T; ++i) pp[i] = 1048576 - s.T + i;
        auto pb = to_bytes(text_pos4(pp));
        ggml_backend_tensor_set(p, pb.data(), 0, pb.size());
        for (int i = 0; i < 2; ++i) ggml_backend_graph_compute(be, gf);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPS; ++i) ggml_backend_graph_compute(be, gf);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (REPS * CHAIN);
        printf("perf %-8s %-24s %10.1f us/op\n", bname.c_str(), s.name, us);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
}

// ---------------------------------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string backend = "all", dump_path, tsv_path, ref_path;
    bool check = false, perf = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--check") check = true;
        else if (a == "--perf") perf = true;
        else if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--dump" && i + 1 < argc) dump_path = argv[++i];
        else if (a == "--tsv" && i + 1 < argc) tsv_path = argv[++i];
        else if (a == "--check-ref" && i + 1 < argc) ref_path = argv[++i];
        else {
            fprintf(stderr, "usage: %s [--check] [--perf] [--dump FILE] [--tsv FILE] [--check-ref FILE] [--backend CPU|Vulkan0|all]\n", argv[0]);
            return 2;
        }
    }
    const char * env = getenv("LLAMA_ROPE_PRECISE");
    printf("LLAMA_ROPE_PRECISE=%s\n", env ? env : "(unset)");

    if (!ref_path.empty()) {
        const int rc = check_ref(ref_path.c_str());
        if (rc != 0) return rc;
    }

    std::vector<std::pair<std::string, ggml_backend_t>> bes;
    for (const char * name : { "CPU", "Vulkan0" }) {
        if (backend != "all" && backend != name) continue;
        ggml_backend_t be = ggml_backend_init_by_name(name, nullptr);
        if (!be) {
            printf("backend %s: not available\n", name);
            continue;
        }
        if (strcmp(name, "CPU") == 0) {
            ggml_backend_cpu_set_n_threads(be, 8);
        }
        bes.push_back({ name, be });
    }
    if (bes.empty()) {
        fprintf(stderr, "no backend\n");
        return 2;
    }

    if (!dump_path.empty()) {
        return run_dump(bes, dump_path.c_str());
    }
    if (perf) {
        for (auto & be : bes) run_perf(be.first, be.second);
        return 0;
    }
    if (!tsv_path.empty()) {
        g_tsv = fopen(tsv_path.c_str(), "w");
    }

    const rope_cfg cfgs[] = {
        make_cfg("trunk/draft yarn", 256, true),
        make_cfg("trunk/draft native", 256, false),
        make_cfg("indexer yarn", 128, true),
        make_cfg("indexer native", 128, false),
    };

    // YaRN facts
    {
        const rope_ref r = make_ref(cfgs[0]);
        printf("YaRN x4: corr dims %.0f / %.0f, attn_factor (llama_context) %.9g, mscale %.9Lg; pairs <= %d unscaled, >= %d x1/4\n",
                r.corr[0], r.corr[1], cfgs[0].attn_factor, r.mscale, (int) r.corr[0], (int) r.corr[1]);
    }

    bool fail = false;
    double worst_nmse = 0, worst_dot = 0;
    for (const auto & c : cfgs) {
        const rope_ref r = make_ref(c);
        // emulations (backend-independent)
        std::vector<std::pair<std::string, std::function<ld(int64_t, int)>>> emus = {
            { "emu fp32(theta) floor", [&](int64_t p, int j) { return (ld) (float) ((ld) p * r.freq[j]); } },
            { "emu HF/Megatron/vLLM fp32", [&](int64_t p, int j) { return (ld) ((float) p * hf_inv_freq(r, j)); } },
            { "emu same + bf16 cos/sin", [&](int64_t p, int j) {
                  const float a  = (float) p * hf_inv_freq(r, j);
                  const float ms = (float) r.mscale;
                  return atan2l((ld) bf16_round(sinf(a) * ms), (ld) bf16_round(cosf(a) * ms));
              } },
        };
        for (auto & e : emus) {
            auto res = analyse(e.first.c_str(), nullptr, c, e.second);
            print_summary(e.first.c_str(), c, res, false);
        }
        for (auto & be : bes) {
            auto res = analyse(be.first.c_str(), be.second, c, nullptr);
            print_summary(be.first.c_str(), c, res, c.D == 256 && c.ext_factor != 0.0f);
            for (int w = 0; w < NWIN; ++w) {
                worst_nmse = std::max(worst_nmse, res[w].nmse_rot);
                if (res[w].nmse_full > 1e-3) {
                    printf("  FAIL: %s %s %s NMSE %.3e > 1e-3 (gross)\n", be.first.c_str(), c.name, WINDOWS[w].name, res[w].nmse_full);
                    fail = true;
                }
            }
            if (c.ext_factor != 0.0f || c.D == 256) {
                auto dr = dot_test(be.second, c, be.first.c_str());
                printf("  rel-pos q·k error / (|q||k|), rms / max:");
                for (auto & kv : dr) {
                    printf("  %s %.2e/%.2e", kv.first.c_str(), kv.second.rms, kv.second.maxabs);
                    worst_dot = std::max(worst_dot, kv.second.maxabs);
                }
                printf("\n");
            }
        }
    }
    printf("\n");
    storage_floors(256);
    storage_floors(128);

    printf("\nworst NMSE (rotated dims) %.3e, worst rel-pos dot error %.3e\n", worst_nmse, worst_dot);
    if (check) {
        const bool ok = worst_nmse <= 1e-9 && worst_dot <= 1e-6;
        printf("check (NMSE <= 1e-9, rel-pos <= 1e-6): %s\n", ok ? "PASS" : "FAIL");
        fail = fail || !ok;
    }
    if (g_tsv) fclose(g_tsv);
    for (auto & be : bes) ggml_backend_free(be.second);
    return fail ? 1 : 0;
}
