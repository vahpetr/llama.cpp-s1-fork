// [TAG_MMID_SCALE_EPI] bit-exactness of the MUL_MAT_ID + MUL fusion on the GEMM (prefill) path: the routing weight applied in the
// mul_mm.comp tile store (GGML_VK_MMID_SCALE_EPILOGUE, port of Nathanw1014 30d8bb02b / 3b8d43efa) must give the bits of the unfused
// pair. Per case, on the first Vulkan device, two graphs over the same tensors:
//   y = mul_mat_id(as, b, ids)            (a lone MUL_MAT_ID: never fused, the mul_mm GEMM path)
//   z = mul(mul_mat_id(as, b, ids), w)    (fused when the knob is on and the gates pass)
// and z[i] must equal the f32 product y[i] * w[slot, token] computed on the host (IEEE single rounding, the separate MUL's op),
// bit for bit, for every element. The FNV-1a of z is printed per case, so the knob-off arm (GGML_VK_MMID_SCALE_EPILOGUE=0: the
// separate MUL dispatch) can be compared byte for byte with the default arm; run with GGML_VK_FUSION_COUNT=1 to see that the
// default arm fused (mul_mat_id_mul=1 lines). Fallback arm (GGML_VK_MMID_SCALE_EPI_TEST_FALLBACK=1, a test hook): the pair is still
// fused at graph level but the GEMM dispatcher refuses the scale, as it would should its pipeline choice disagree with the
// ggml_vk_can_fuse mirror (ggml_vk_mmid_gemm_uses_q8_1) - it must run the unfused GEMM + MUL (one "mmid_scale_epi: fallback" WARN),
// never assert, and print the default arm's hashes. Shapes: the production expert GEMMs (gate/up 640 x 2560 q4_K / q6_K / iq3_s, down
// 2560 x 640 q5_0 / q8_0; top-10, broadcast B for gate/up; 32 experts, not 512, to bound host RAM) at n = 65 / 512 / 2048 and the
// f32 / f16 A paths.
// Not a ctest: needs a Vulkan device (SKIP otherwise).
// usage: test-mmid-scale-epi [seed]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct epi_case {
    ggml_type type_a;
    int64_t m, k, n;
    int n_mats, n_used;
    bool bcast;
};

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

// returns 0 = exact, 1 = mismatch, 2 = setup failure
static int run_case(ggml_backend_t be, const epi_case & c, std::mt19937 & rng, uint64_t * hash_out, size_t * n_diff_out) {
    const size_t n_tensors = 8;
    ggml_init_params ip = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead() * 2 + 4096, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, c.type_a, c.k, c.m, c.n_mats);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, c.bcast ? 1 : c.n_used, c.n);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, c.n_used, c.n);

    ggml_tensor * y  = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_tensor * y2 = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_tensor * z  = ggml_mul(ctx, y2, w);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return 2; }

    // data: A from N(0, 1) quantised by ggml (or f32 / f16 direct), B and w from N(0, 1) / U(0, 1), ids = n_used distinct experts
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    {
        const int64_t nel = c.k * c.m * c.n_mats;
        std::vector<float> fa(nel);
        for (auto & v : fa) { v = nd(rng); }
        std::vector<uint8_t> qa(ggml_nbytes(as));
        if (c.type_a == GGML_TYPE_F32) {
            memcpy(qa.data(), fa.data(), qa.size());
        } else {
            ggml_quantize_chunk(c.type_a, fa.data(), qa.data(), 0, c.m * c.n_mats, c.k, nullptr);
        }
        ggml_backend_tensor_set(as, qa.data(), 0, qa.size());
    }
    {
        std::vector<float> fb(ggml_nelements(b));
        for (auto & v : fb) { v = nd(rng); }
        ggml_backend_tensor_set(b, fb.data(), 0, ggml_nbytes(b));
    }
    std::vector<float> fw(ggml_nelements(w));
    for (auto & v : fw) { v = ud(rng); }
    ggml_backend_tensor_set(w, fw.data(), 0, ggml_nbytes(w));
    {
        std::vector<int32_t> fi(ggml_nelements(ids));
        std::vector<int32_t> perm(c.n_mats);
        for (int i = 0; i < c.n_mats; i++) { perm[i] = i; }
        // a skewed router: half of the tokens draw from the first 16 experts (deep, multi-tile experts), the rest from all
        for (int64_t t = 0; t < c.n; t++) {
            const int pool = (t % 2 == 0 && c.n_mats > 16 && c.n_used <= 16) ? 16 : c.n_mats;
            for (int s = 0; s < c.n_used; s++) {
                std::uniform_int_distribution<int> pick(s, pool - 1);
                std::swap(perm[s], perm[pick(rng)]);
                fi[t * c.n_used + s] = perm[s];
            }
        }
        ggml_backend_tensor_set(ids, fi.data(), 0, ggml_nbytes(ids));
    }

    ggml_cgraph * g1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g1, y);
    ggml_cgraph * g2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g2, z);
    if (ggml_backend_graph_compute(be, g1) != GGML_STATUS_SUCCESS || ggml_backend_graph_compute(be, g2) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return 2;
    }

    std::vector<float> hy(ggml_nelements(y)), hz(ggml_nelements(z));
    ggml_backend_tensor_get(y, hy.data(), 0, ggml_nbytes(y));
    ggml_backend_tensor_get(z, hz.data(), 0, ggml_nbytes(z));

    size_t n_diff = 0;
    for (int64_t t = 0; t < c.n; t++) {
        for (int s = 0; s < c.n_used; s++) {
            const float ws = fw[t * c.n_used + s];
            for (int64_t r = 0; r < c.m; r++) {
                const size_t i = (size_t) (t * c.n_used + s) * c.m + r;
                volatile float ref = hy[i] * ws;   // one f32 rounding, as the separate MUL
                const float ref_v = ref;
                if (memcmp(&ref_v, &hz[i], sizeof(float)) != 0) {
                    if (n_diff < 3) {
                        fprintf(stderr, "  diff t=%" PRId64 " slot=%d row=%" PRId64 ": z=%.9g ref=%.9g (y=%.9g w=%.9g)\n",
                                t, s, r, hz[i], ref_v, hy[i], ws);
                    }
                    n_diff++;
                }
            }
        }
    }
    *hash_out = fnv1a(hz.data(), hz.size() * sizeof(float));
    *n_diff_out = n_diff;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return n_diff == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    const unsigned seed = argc > 1 ? (unsigned) atoi(argv[1]) : 1234u;
    ggml_backend_t be = ggml_backend_init_by_name("Vulkan0", nullptr);
    if (!be) {
        printf("test-mmid-scale-epi: SKIP (no Vulkan0 backend)\n");
        return 0;
    }
    std::vector<epi_case> cases;
    for (int64_t n : { 65, 512, 2048 }) {
        for (ggml_type t : { GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ3_S }) {
            cases.push_back({ t, 640, 2560, n, 32, 10, true });      // gate / up (32 of the 512 experts: host RAM)
        }
        for (ggml_type t : { GGML_TYPE_Q5_0, GGML_TYPE_Q8_0 }) {
            cases.push_back({ t, 2560, 640, n, 32, 10, false });     // down
        }
    }
    cases.push_back({ GGML_TYPE_F32, 256, 512, 300, 32, 4, false });
    cases.push_back({ GGML_TYPE_F16, 256, 512, 300, 32, 4, false });
    cases.push_back({ GGML_TYPE_Q4_K, 768, 2048, 512, 128, 8, false });   // the test-backend-ops MUL_MAT_ID_FUSION shape

    std::mt19937 rng(seed);
    int n_fail = 0, n_err = 0;
    for (const epi_case & c : cases) {
        uint64_t h = 0;
        size_t nd = 0;
        const int rc = run_case(be, c, rng, &h, &nd);
        printf("[mmid-scale-epi] type=%s m=%" PRId64 " k=%" PRId64 " n=%" PRId64 " n_mats=%d n_used=%d bcast=%d diff=%zu hash=%016" PRIx64 " %s\n",
               ggml_type_name(c.type_a), c.m, c.k, c.n, c.n_mats, c.n_used, (int) c.bcast, nd, h,
               rc == 0 ? "EXACT" : rc == 1 ? "DIFFERS" : "ERROR");
        n_fail += rc == 1;
        n_err  += rc == 2;
    }
    ggml_backend_free(be);
    printf("test-mmid-scale-epi: %s (%zu cases, %d differ, %d errors; GGML_VK_MMID_SCALE_EPILOGUE=%s)\n",
           n_fail == 0 && n_err == 0 ? "PASS" : "FAIL", cases.size(), n_fail, n_err,
           getenv("GGML_VK_MMID_SCALE_EPILOGUE") ? getenv("GGML_VK_MMID_SCALE_EPILOGUE") : "unset");
    return n_fail == 0 && n_err == 0 ? 0 : 1;
}
