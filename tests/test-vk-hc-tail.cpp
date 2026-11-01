// [TAG_HC_TAIL_FUSE] the qwen4exp build_hc_mix tail on a Vulkan device, for the old-vs-new byte comparison of hc_mixed:
//   g = MUL_MAT(W, L)  (K = 1, L[t] = +-2^e: g = W[m] * L[t] exactly; it stands in for the hc_up GEMM output)
//   gated = MUL(xn, SIGMOID(g)); RESHAPE_3D; S stream VIEWs; ADD chain; SCALE(1/S) -> hc_mixed
// built exactly as qwen4exp.cpp build_hc_mix does and run through ggml_backend_sched (so graph_optimize orders the nodes
// as in the model). For every n and "sublayer" the fixtures differ (normal, wide, signed zeros, subnormals,
// saturation / overflow / inf / NaN, cancellation patterns that tell left-associated from balanced sums, mixed).
// Output: the raw hc_mixed bytes of sublayers 0..7 (one per fixture category) of every n appended to --dump <file>, one "[hc-tail] n= l= pre= mixed=" line
// each (FNV-1a of the GEMM output and of hc_mixed), and "[hc-tail] DONE all=". Compare two runs (GGML_VK_HC_TAIL_FUSE=0 vs
// default, or a negative-control build with GGML_VK_HC_TAIL_MUTANT=<m>) with cmp; GGML_VK_FUSION_COUNT=1 shows which
// fusion ran (hc_tail_mix vs sigmoid_mul + multi_add_scale). --extra marks the gated plane as a graph output (an external
// consumer of an elided node): the fusion must not fire. --perf <iters>: only n = 2048, layer 0, repeated (for
// GGML_VK_PERF_LOGGER). Not a ctest: needs a Vulkan device; without one it prints SKIP and exits 0.
// usage: test-vk-hc-tail [--dump <file>] [--ns 1,9,64,1020,2044,2048] [--layers 96] [--extra] [--perf <iters>]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const int64_t k_n_embd = 2560;
static const int64_t k_hc     = 4;

static uint64_t fnv1a(const void * data, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ p[i]) * 0x100000001b3ull;
    }
    return h;
}

struct rng {
    uint64_t s;
    uint64_t next() { uint64_t z = (s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
    double   uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    float    normal(float sd) { double u1 = uni() + 1e-300, u2 = uni(); return (float) (sd * sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)); }
    int      pick(int k) { return (int) (next() % (uint64_t) k); }
};

static float f16_exact(float x) {
    // keep the GEMM input representable in f16 too, so W * 2^e is exact whatever the K = 1 GEMM converts it to
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
}

static float bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// sigmoid-input value for row m of category cat
static float gen_w(rng & r, int cat) {
    switch (cat) {
        case 0: return f16_exact(r.normal(3.0f));
        case 1: return f16_exact(r.normal(20.0f));
        case 2: { int k = r.pick(4); return k == 0 ? 0.0f : k == 1 ? -0.0f : f16_exact(r.normal(2.0f)); }
        case 3: return f16_exact(r.normal(1.0f));
        case 4: {
            static const float sat[] = { 89.0f, -89.0f, 100.0f, -100.0f, 200.0f, -200.0f, 1e4f, -1e4f, 60.0f, -60.0f };
            int k = r.pick(14);
            if (k < 10) { return sat[k]; }
            if (k == 10) { return INFINITY; }
            if (k == 11) { return -INFINITY; }
            if (k == 12) { return NAN; }
            return f16_exact(r.normal(30.0f));
        }
        case 5: return 100.0f; // sigmoid(100 * L) = 1 for L > 0 (the stream value passes unchanged), ~0 for L < 0
        default: return f16_exact(r.normal(4.0f));
    }
}

static float gen_xn(rng & r, int cat, int s, float big, float s1, float s2) {
    switch (cat) {
        case 0: return r.normal(1.0f);
        case 1: return r.normal(100.0f);
        case 2: { int k = r.pick(3); return k == 0 ? 0.0f : k == 1 ? -0.0f : r.normal(1.0f); }
        case 3: {
            int k = r.pick(3);
            uint32_t u = (uint32_t) (r.next() & 0x007fffffu) | ((r.next() & 1u) << 31);
            if (k == 0) { return bits(u == 0 ? 1u : u); }               // subnormal
            if (k == 1) { return (r.pick(2) ? 1.0f : -1.0f) * 1.5e-38f * (float) (1 + r.pick(8)); } // tiny normal -> subnormal product
            return r.normal(1e-30f);
        }
        case 4: {
            int k = r.pick(10);
            if (k == 0) { return INFINITY; }
            if (k == 1) { return -INFINITY; }
            if (k < 5) { return r.normal(1e30f); }
            return r.normal(1.0f);
        }
        case 5: {
            // cancellation: [B, s1, -B, s2] in a per-element stream order; ((B + s1) - B) + s2 != (B + s1) + (-B + s2)
            switch (s) { case 0: return big; case 1: return s1; case 2: return -big; default: return s2; }
        }
        default: return r.normal(1.0f) * (float) (1 << r.pick(12));
    }
}

int main(int argc, char ** argv) {
    std::string dump;
    std::vector<int64_t> ns = { 1, 9, 64, 1020, 2044, 2048 };
    int  n_layers = 96;
    bool extra    = false;
    int  perf     = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dump") && i + 1 < argc) { dump = argv[++i]; }
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) { n_layers = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--extra")) { extra = true; }
        else if (!strcmp(argv[i], "--perf") && i + 1 < argc) { perf = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--ns") && i + 1 < argc) {
            ns.clear();
            for (char * t = strtok(argv[++i], ","); t; t = strtok(nullptr, ",")) { ns.push_back(atoll(t)); }
        } else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (perf > 0) { ns = { 2048 }; n_layers = 1; }

    ggml_backend_load_all();
    ggml_backend_dev_t vk_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if ((ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU) && strncmp(ggml_backend_dev_name(d), "Vulkan", 6) == 0) { vk_dev = d; break; }
    }
    if (!vk_dev) { printf("[hc-tail] SKIP no Vulkan device\n"); return 0; }
    ggml_backend_t vk  = ggml_backend_dev_init(vk_dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t backends[2] = { vk, cpu };

    FILE * fd = dump.empty() ? nullptr : fopen(dump.c_str(), "wb");
    uint64_t all = 0xcbf29ce484222325ull;
    const int64_t hc_dim = k_hc * k_n_embd;

    for (const int64_t n : ns) {
        ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * W  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, hc_dim);
        ggml_tensor * L  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n);
        ggml_tensor * xn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, n);
        ggml_set_input(W); ggml_set_input(L); ggml_set_input(xn);

        // build_hc_mix, from the hc_up matmul on
        ggml_tensor * g = ggml_mul_mat(ctx, W, L);
        ggml_set_output(g);
        ggml_tensor * gate  = ggml_sigmoid(ctx, g);
        ggml_tensor * gated = ggml_mul(ctx, xn, gate);
        if (extra) { ggml_set_output(gated); }
        gated = ggml_reshape_3d(ctx, gated, k_n_embd, k_hc, n);
        ggml_tensor * mixed = ggml_view_2d(ctx, gated, k_n_embd, n, ggml_row_size(gated->type, k_n_embd) * k_hc, 0);
        for (int64_t c = 1; c < k_hc; ++c) {
            ggml_tensor * s = ggml_view_2d(ctx, gated, k_n_embd, n, ggml_row_size(gated->type, k_n_embd) * k_hc,
                                           ggml_row_size(gated->type, k_n_embd) * c);
            mixed = ggml_add(ctx, mixed, s);
        }
        mixed = ggml_scale(ctx, mixed, 1.0f / (float) k_hc);
        ggml_set_output(mixed);

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, mixed);
        if (extra) { ggml_build_forward_expand(gf, gated); }

        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 256, false, false);
        // everything on the GPU (the inputs too): one split, graph_optimize'd like the model's graph
        for (ggml_tensor * t : { W, L, xn }) { ggml_backend_sched_set_tensor_backend(sched, t, vk); }
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) { ggml_backend_sched_set_tensor_backend(sched, ggml_graph_node(gf, i), vk); }
        if (!ggml_backend_sched_alloc_graph(sched, gf)) { fprintf(stderr, "alloc failed\n"); return 1; }

        std::vector<float> w(hc_dim), l(n), x(hc_dim * n), out(k_n_embd * n), pre(hc_dim * n);
        for (int il = 0; il < n_layers; ++il) {
            rng r { 0x5eedull ^ ((uint64_t) n << 20) ^ (uint64_t) il * 0x9e3779b97f4a7c15ull };
            const int layer_cat = il % 8;
            auto cat_of = [&](int64_t) { return layer_cat == 7 ? r.pick(7) : layer_cat; };
            std::vector<int> rcat(hc_dim);
            for (int64_t m = 0; m < hc_dim; ++m) { rcat[m] = cat_of(m); w[m] = gen_w(r, rcat[m]); }
            static const float pw[] = { 1.0f, 2.0f, 0.5f, -1.0f };
            for (int64_t t = 0; t < n; ++t) { l[t] = (layer_cat == 5) ? (r.pick(8) ? 1.0f : 2.0f) : pw[r.pick(4)]; }
            for (int64_t t = 0; t < n; ++t) {
                for (int64_t e = 0; e < k_n_embd; ++e) {
                    const float big = ldexpf(1.0f, 26 + r.pick(4)) * (r.pick(2) ? 1.0f : -1.0f);
                    const float s1 = (float) (1 + r.pick(7)) * (r.pick(2) ? 1.0f : -1.0f) * 0.25f;
                    const float s2 = (float) (1 + r.pick(7)) * (r.pick(2) ? 1.0f : -1.0f) * 0.25f;
                    const int rot = r.pick(4); // which slot holds which of the four values
                    for (int64_t c = 0; c < k_hc; ++c) {
                        const int64_t m = c * k_n_embd + e;
                        x[t * hc_dim + m] = gen_xn(r, rcat[m], (int) ((c + rot) % k_hc), big, s1, s2);
                    }
                }
            }
            ggml_backend_tensor_set(W, w.data(), 0, ggml_nbytes(W));
            ggml_backend_tensor_set(L, l.data(), 0, ggml_nbytes(L));
            ggml_backend_tensor_set(xn, x.data(), 0, ggml_nbytes(xn));

            const int reps = perf > 0 ? perf : 1;
            for (int rep = 0; rep < reps; ++rep) {
                if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
            }
            ggml_backend_tensor_get(mixed, out.data(), 0, ggml_nbytes(mixed));
            ggml_backend_tensor_get(g, pre.data(), 0, ggml_nbytes(g));
            const uint64_t hm = fnv1a(out.data(), out.size() * 4);
            const uint64_t hp = fnv1a(pre.data(), pre.size() * 4);
            all = fnv1a(&hm, 8, all);
            if (fd && il < 8) { fwrite(out.data(), 4, out.size(), fd); } // bytes of the first 8 sublayers (all 8 categories)
            printf("[hc-tail] n=%" PRId64 " l=%d cat=%d pre=0x%016" PRIx64 " mixed=0x%016" PRIx64 "\n", n, il, layer_cat, hp, hm);
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }
    if (fd) { fclose(fd); }
    printf("[hc-tail] DONE all=0x%016" PRIx64 "\n", all);
    ggml_backend_free(vk);
    ggml_backend_free(cpu);
    return 0;
}
