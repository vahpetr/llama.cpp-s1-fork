// ggml_permute must keep 64-bit ne and byte strides (upstream #29227).
// Metadata only: a no_alloc context never allocates the multi-GiB data.

#include "ggml.h"

#include <cinttypes>
#include <cstdio>
#include <cstdint>

static int n_fail = 0;

static void check_permute(ggml_context * ctx, ggml_tensor * a, int ax0, int ax1, int ax2, int ax3, const char * what) {
    ggml_tensor * p = ggml_permute(ctx, a, ax0, ax1, ax2, ax3);
    const int axis[4] = { ax0, ax1, ax2, ax3 };
    for (int i = 0; i < 4; ++i) {
        const int j = axis[i];
        if (p->ne[j] != a->ne[i] || p->nb[j] != a->nb[i]) {
            fprintf(stderr, "FAIL %s: dim %d -> %d: ne %" PRId64 " (want %" PRId64 "), nb %zu (want %zu)\n",
                    what, i, j, p->ne[j], a->ne[i], p->nb[j], a->nb[i]);
            n_fail++;
        }
    }
    if (p->data != a->data || p->view_src != a || ggml_nelements(p) != ggml_nelements(a)) {
        fprintf(stderr, "FAIL %s: view metadata changed\n", what);
        n_fail++;
    }
}

int main() {
    ggml_init_params params = { /*mem_size*/ 16 * ggml_tensor_overhead(), /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(params);

    // the upstream example: F32 [1536, 2048, 288, 1], nb[2] = 12 MiB, nb[3] = 3 623 878 656 > INT32_MAX
    ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1536, 2048, 288, 1);
    if (a->nb[3] <= (size_t) INT32_MAX) {
        fprintf(stderr, "FAIL setup: nb[3] = %zu is not above INT32_MAX\n", a->nb[3]);
        return 1;
    }
    check_permute(ctx, a, 0, 2, 1, 3, "f32 [1536,2048,288,1] (0,2,1,3)");
    check_permute(ctx, a, 3, 2, 1, 0, "f32 [1536,2048,288,1] (3,2,1,0)");
    check_permute(ctx, a, 1, 3, 0, 2, "f32 [1536,2048,288,1] (1,3,0,2)");

    // a byte stride of 2^31 exactly and one of 2^32 + 4 (both wrap through int)
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1 << 20, 1 << 9, 3, 2);
    check_permute(ctx, b, 0, 2, 1, 3, "f32 [2^20,2^9,3,2] (0,2,1,3)");
    ggml_tensor * c = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (1 << 30) + 1, 1, 4, 1);
    check_permute(ctx, c, 1, 0, 3, 2, "f32 [2^30+1,1,4,1] (1,0,3,2)");

    // ne above INT32_MAX (I8, 3e9 elements in dim 0)
    ggml_tensor * d = ggml_new_tensor_4d(ctx, GGML_TYPE_I8, 3000000000LL, 1, 1, 1);
    check_permute(ctx, d, 1, 0, 2, 3, "i8 [3e9,1,1,1] (1,0,2,3)");

    ggml_free(ctx);

    if (n_fail) {
        fprintf(stderr, "test-permute-large-stride: %d failures\n", n_fail);
        return 1;
    }
    printf("test-permute-large-stride: OK (6 permutes, strides up to %zu bytes)\n", c->nb[2] > a->nb[3] ? c->nb[2] : a->nb[3]);
    return 0;
}
