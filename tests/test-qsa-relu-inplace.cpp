// [TAG_QSA_RELU_INPLACE] (R80.1 v49, QSA depth plan B5 step 2) LLAMA_QSA_RELU_INPLACE: the indexer score chain of
// build_qsa_select (src/models/qwen4exp.cpp) - score GEMM -> reshape_4d -> ReLU -> per-head slice sums - built with
// ggml_relu and with ggml_relu_inplace on the CPU backend:
//   1. bitwise: the ReLU plane (kept as an output) and the head-summed score are memcmp-identical, over several shapes
//      and random inputs with a negative share (so the ReLU actually clamps);
//   2. memory: the graph allocator's compute buffer for the chain (only the summed score kept) with either form, and
//      whether the out-of-place ReLU already landed on the GEMM output's memory (the allocator's automatic in-place
//      reuse) - the report says what the knob saves on top of that (0 = nothing).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct shape { int64_t n_blocks, n_idx_h, n_tps, n_stream, d; };

struct built {
    ggml_context * ctx  = nullptr;
    ggml_cgraph *  gf   = nullptr;
    ggml_tensor *  pooled = nullptr, * q = nullptr, * mm = nullptr, * relu = nullptr, * out = nullptr;
};

static built build(const shape & sh, bool inplace, bool keep_relu) {
    built b;
    ggml_init_params ip = { 64*1024*1024, nullptr, true };
    b.ctx = ggml_init(ip);
    b.pooled = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, sh.d, sh.n_blocks, sh.n_stream);
    b.q      = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, sh.d, sh.n_idx_h*sh.n_tps, sh.n_stream);
    ggml_set_input(b.pooled);
    ggml_set_input(b.q);
    // as build_qsa_select
    ggml_tensor * score = ggml_mul_mat(b.ctx, b.pooled, b.q);
    b.mm  = score;
    score = ggml_reshape_4d(b.ctx, score, sh.n_blocks, sh.n_idx_h, sh.n_tps, sh.n_stream);
    score = inplace ? ggml_relu_inplace(b.ctx, score) : ggml_relu(b.ctx, score);
    b.relu = score;
    if (keep_relu) {
        ggml_set_output(score);
    }
    ggml_tensor * summed = nullptr;
    for (int64_t h = 0; h < sh.n_idx_h; ++h) {
        ggml_tensor * slice = ggml_view_3d(b.ctx, score, sh.n_blocks, sh.n_tps, sh.n_stream, score->nb[2], score->nb[3], h*score->nb[1]);
        summed = summed ? ggml_add(b.ctx, summed, slice) : ggml_cont(b.ctx, slice);
    }
    ggml_set_output(summed);
    b.out = summed;
    b.gf = ggml_new_graph_custom(b.ctx, 1024, false);
    if (keep_relu) {
        ggml_build_forward_expand(b.gf, score);
    }
    ggml_build_forward_expand(b.gf, summed);
    return b;
}

int main() {
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(be, 4);
    const shape shapes[] = {
        { 512, 4, 1, 1, 128 }, { 1024, 4, 6, 1, 128 }, { 2048, 8, 64, 1, 128 }, { 777, 2, 33, 2, 64 }, { 4096, 4, 256, 1, 128 },
    };
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0;
    for (const auto & sh : shapes) {
        std::vector<float> pooled(sh.d*sh.n_blocks*sh.n_stream), q(sh.d*sh.n_idx_h*sh.n_tps*sh.n_stream);
        for (auto & x : pooled) { x = nd(rng); }
        for (auto & x : q)      { x = nd(rng) - 0.1f; }

        std::vector<float> res_relu[2], res_out[2];
        size_t buf_size[2] = {};
        bool   reused[2]   = {};
        for (int inplace = 0; inplace < 2; ++inplace) {
            // 1. bitwise (ReLU plane kept)
            {
                built b = build(sh, inplace != 0, true);
                ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                if (!ggml_gallocr_alloc_graph(ga, b.gf)) { printf("alloc failed\n"); return 1; }
                ggml_backend_tensor_set(b.pooled, pooled.data(), 0, pooled.size()*sizeof(float));
                ggml_backend_tensor_set(b.q,      q.data(),      0, q.size()*sizeof(float));
                if (ggml_backend_graph_compute(be, b.gf) != GGML_STATUS_SUCCESS) { printf("compute failed\n"); return 1; }
                res_relu[inplace].resize(ggml_nelements(b.relu));
                res_out[inplace].resize(ggml_nelements(b.out));
                ggml_backend_tensor_get(b.relu, res_relu[inplace].data(), 0, ggml_nbytes(b.relu));
                ggml_backend_tensor_get(b.out,  res_out[inplace].data(),  0, ggml_nbytes(b.out));
                ggml_gallocr_free(ga);
                ggml_free(b.ctx);
            }
            // 2. memory (only the summed score kept, as in the model graph)
            {
                built b = build(sh, inplace != 0, false);
                ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                if (!ggml_gallocr_alloc_graph(ga, b.gf)) { printf("alloc failed\n"); return 1; }
                buf_size[inplace] = ggml_gallocr_get_buffer_size(ga, 0);
                reused[inplace]   = b.relu->data == b.mm->data;
                ggml_gallocr_free(ga);
                ggml_free(b.ctx);
            }
        }
        size_t n_neg = 0;
        for (float x : res_relu[0]) { n_neg += x == 0.0f; }
        const bool same_relu = memcmp(res_relu[0].data(), res_relu[1].data(), res_relu[0].size()*sizeof(float)) == 0;
        const bool same_out  = memcmp(res_out[0].data(),  res_out[1].data(),  res_out[0].size()*sizeof(float)) == 0;
        const size_t plane = (size_t) sh.n_blocks*sh.n_idx_h*sh.n_tps*sh.n_stream*sizeof(float);
        printf("shape blocks=%lld heads=%lld tps=%lld streams=%lld: relu plane %s, score %s (%zu of %zu clamped); "
               "compute buffer relu %zu B / relu_inplace %zu B (saves %lld B, plane %zu B); out-of-place relu on the GEMM "
               "output's memory: %s\n",
               (long long) sh.n_blocks, (long long) sh.n_idx_h, (long long) sh.n_tps, (long long) sh.n_stream,
               same_relu ? "identical" : "DIFFERS", same_out ? "identical" : "DIFFERS", n_neg, res_relu[0].size(),
               buf_size[0], buf_size[1], (long long) buf_size[0] - (long long) buf_size[1], plane,
               reused[0] ? "yes (allocator reuse)" : "no");
        fails += !same_relu || !same_out || n_neg == 0 || !reused[1] || buf_size[1] > buf_size[0];
    }
    ggml_backend_free(be);
    printf("%s\n", fails == 0 ? "OK" : "FAIL");
    return fails == 0 ? 0 : 1;
}
