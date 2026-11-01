// [TAG_HC_MERGE_KPROOF] P3 K-proof: hc_down + hc_inject as two matmuls over the views of one load-merged container
// (llama_model_loader::merge_rows layout: down rows 0..n_lo-1, inject rows n_lo.., inject at a 256-aligned byte offset)
// vs ONE matmul over the container (qwen4exp build_hc_mix, merged path). Real weights from a GGUF holding
// blk.<il>.hc_{attn,ffn}_{down,inject,norm}.weight (an extract of the brain GGUF is enough). Bitwise compare of every
// output element, per sublayer, per n. Not a ctest (needs the weights and a GPU).
// usage: test-qwen4exp-hc-merge-kproof <gguf>   env: KP_NS (default 9,64,1020,2044,2048), KP_REPS (timing, default 0),
// KP_MUT_X=1 (sensitivity: the merged arm's input differs by one ulp in element 0 -> must FAIL), KP_LAYERS=a-b
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

static std::vector<int> parse_list(const char * s, std::vector<int> def) {
    if (!s || !*s) return def;
    std::vector<int> v; std::string t(s); size_t p = 0;
    while (p <= t.size()) { size_t q = t.find(',', p); if (q == std::string::npos) q = t.size();
        if (q > p) v.push_back(atoi(t.substr(p, q - p).c_str())); p = q + 1; }
    return v;
}

struct pair_t { int il; const char * kind; ggml_tensor * cont = nullptr; ggml_tensor * down = nullptr; ggml_tensor * inj = nullptr; std::vector<float> norm; bool merged = false; };

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <gguf>\n", argv[0]); return 2; }
    const std::vector<int> ns = parse_list(getenv("KP_NS"), {9, 64, 1020, 2044, 2048});
    const int reps = getenv("KP_REPS") ? atoi(getenv("KP_REPS")) : 0;
    const bool mut_x = getenv("KP_MUT_X") && atoi(getenv("KP_MUT_X"));
    int la = 0, lb = 1 << 20; if (const char * s = getenv("KP_LAYERS")) sscanf(s, "%d-%d", &la, &lb);

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(argv[1], gp);
    if (!g) { fprintf(stderr, "gguf open failed\n"); return 2; }
    FILE * f = fopen(argv[1], "rb");
    const size_t data_off = gguf_get_data_offset(g);

    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    if (!be) { fprintf(stderr, "no GPU backend\n"); return 2; }
    printf("[kp] backend %s mut_x=%d reps=%d\n", ggml_backend_name(be), (int) mut_x, reps);

    auto read_t = [&](const char * name, std::vector<uint8_t> & out) -> ggml_tensor * {
        int64_t id = gguf_find_tensor(g, name); if (id < 0) return nullptr;
        ggml_tensor * t = ggml_get_tensor(meta, name); out.resize(ggml_nbytes(t));
        fseek(f, (long) (data_off + gguf_get_tensor_offset(g, id)), SEEK_SET);
        if (fread(out.data(), 1, out.size(), f) != out.size()) { fprintf(stderr, "short read %s\n", name); exit(2); }
        return t;
    };

    // weights: one context, containers + views (merge_rows layout) or separate tensors (type mismatch)
    std::vector<pair_t> pairs;
    ggml_init_params wp = { 4096 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    struct upload_t { ggml_tensor * dst; std::vector<uint8_t> bytes; };
    std::vector<upload_t> ups;
    for (int il = la; il <= lb && il < 1000; ++il) {
        for (const char * kind : { "attn", "ffn" }) {
            char nd[128], ni[128], nn[128];
            snprintf(nd, sizeof nd, "blk.%d.hc_%s_down.weight", il, kind);
            snprintf(ni, sizeof ni, "blk.%d.hc_%s_inject.weight", il, kind);
            snprintf(nn, sizeof nn, "blk.%d.hc_%s_norm.weight", il, kind);
            std::vector<uint8_t> bd, bi, bn;
            ggml_tensor * md = read_t(nd, bd); ggml_tensor * mi = read_t(ni, bi); ggml_tensor * mn = read_t(nn, bn);
            if (!md || !mi) continue;
            pair_t p; p.il = il; p.kind = kind;
            if (mn) { p.norm.resize(ggml_nelements(mn)); memcpy(p.norm.data(), bn.data(), bn.size()); }
            const int64_t K = md->ne[0], n_lo = md->ne[1], hc = mi->ne[1];
            if (md->type == mi->type && ggml_nbytes(md) % 256 == 0) {
                p.merged = true;
                p.cont = ggml_new_tensor_2d(wctx, md->type, K, n_lo + hc);
                p.down = ggml_view_2d(wctx, p.cont, K, n_lo, p.cont->nb[1], 0);
                p.inj  = ggml_view_2d(wctx, p.cont, K, hc,   p.cont->nb[1], ggml_nbytes(md));
            } else {
                p.down = ggml_new_tensor_2d(wctx, md->type, K, n_lo);
                p.inj  = ggml_new_tensor_2d(wctx, mi->type, K, hc);
            }
            snprintf(p.down->name, sizeof p.down->name, "%s", nd);
            ups.push_back({ p.down, std::move(bd) }); ups.push_back({ p.inj, std::move(bi) });
            printf("[kp] pair il=%d %s down=%s inject=%s %s\n", il, kind, ggml_type_name(md->type), ggml_type_name(mi->type), p.merged ? "MERGED" : "separate-only");
            pairs.push_back(std::move(p));
        }
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    for (auto & u : ups) ggml_backend_tensor_set(u.dst, u.bytes.data(), 0, u.bytes.size());
    ups.clear();

    long total_el = 0, total_bad = 0; int n_pairs_merged = 0;
    for (auto & p : pairs) n_pairs_merged += p.merged;
    for (int n : ns) {
        long n_bad_n = 0;
        for (auto & p : pairs) {
            if (!p.merged) continue;
            const int64_t K = p.down->ne[0], n_lo = p.down->ne[1], hc = p.inj->ne[1], M = n_lo + hc;
            // realistic hc_norm input: grouped RMS norm over each of the hc streams of a deterministic pseudo-normal x, times gamma
            std::vector<float> x((size_t) K * n);
            uint64_t st = 0x9E3779B97F4A7C15ull ^ ((uint64_t) p.il * 131 + (p.kind[0] == 'f') * 7 + (uint64_t) n * 1000003);
            auto rnd = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (double) ((st >> 11) * (1.0 / 9007199254740992.0)); };
            const int64_t ne = K / hc;
            for (int c = 0; c < n; ++c) for (int64_t s = 0; s < hc; ++s) {
                double ss = 0; float * r = &x[(size_t) c * K + s * ne];
                for (int64_t e = 0; e < ne; ++e) { double u1 = rnd() + 1e-12, u2 = rnd(); r[e] = (float) (sqrt(-2 * log(u1)) * cos(6.283185307179586 * u2) * (1 + 3 * (e % 97 == 0))); ss += (double) r[e] * r[e]; }
                const float sc = (float) (1.0 / sqrt(ss / ne + 1e-6));
                for (int64_t e = 0; e < ne; ++e) r[e] = r[e] * sc * (p.norm.empty() ? 1.0f : p.norm[s * ne + e]);
            }
            ggml_init_params cp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() * 2, nullptr, true };
            ggml_context * cctx = ggml_init(cp);
            ggml_tensor * xa = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, K, n); ggml_set_input(xa);
            ggml_tensor * xb = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, K, n); ggml_set_input(xb);
            ggml_tensor * od = ggml_mul_mat(cctx, p.down, xa); ggml_set_output(od); ggml_set_name(od, "hc_down");
            ggml_tensor * oi = ggml_mul_mat(cctx, p.inj,  xa); ggml_set_output(oi); ggml_set_name(oi, "hc_inject");
            ggml_tensor * om = ggml_mul_mat(cctx, p.cont, xb); ggml_set_output(om); ggml_set_name(om, "hc_down_inject");
            ggml_cgraph * ga = ggml_new_graph(cctx); ggml_build_forward_expand(ga, od); ggml_build_forward_expand(ga, oi);
            ggml_cgraph * gb = ggml_new_graph(cctx); ggml_build_forward_expand(gb, om);
            ggml_backend_buffer_t cbuf = ggml_backend_alloc_ctx_tensors(cctx, be);
            ggml_backend_tensor_set(xa, x.data(), 0, x.size() * 4);
            if (mut_x) { uint32_t u; memcpy(&u, &x[0], 4); u ^= 1; memcpy(&x[0], &u, 4); }
            ggml_backend_tensor_set(xb, x.data(), 0, x.size() * 4);
            ggml_backend_graph_compute(be, ga); ggml_backend_graph_compute(be, gb); ggml_backend_synchronize(be);
            std::vector<float> vd((size_t) n_lo * n), vi((size_t) hc * n), vm((size_t) M * n);
            ggml_backend_tensor_get(od, vd.data(), 0, vd.size() * 4);
            ggml_backend_tensor_get(oi, vi.data(), 0, vi.size() * 4);
            ggml_backend_tensor_get(om, vm.data(), 0, vm.size() * 4);
            long bad_d = 0, bad_i = 0; double maxd = 0; int fr = -1, fc = -1; float fa = 0, fb = 0;
            for (int c = 0; c < n; ++c) for (int64_t r = 0; r < M; ++r) {
                const float a = r < n_lo ? vd[(size_t) c * n_lo + r] : vi[(size_t) c * hc + (r - n_lo)];
                const float b = vm[(size_t) c * M + r];
                if (memcmp(&a, &b, 4) != 0) {
                    (r < n_lo ? bad_d : bad_i)++; maxd = std::max(maxd, (double) fabsf(a - b));
                    if (fr < 0) { fr = (int) r; fc = c; fa = a; fb = b; }
                }
            }
            total_el += (long) M * n; total_bad += bad_d + bad_i; n_bad_n += bad_d + bad_i;
            double us_a = 0, us_b = 0;
            if (reps > 0) {
                for (int w = 0; w < 2; ++w) { ggml_backend_graph_compute(be, ga); ggml_backend_graph_compute(be, gb); }
                ggml_backend_synchronize(be);
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) ggml_backend_graph_compute(be, ga);
                ggml_backend_synchronize(be);
                auto t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) ggml_backend_graph_compute(be, gb);
                ggml_backend_synchronize(be);
                auto t2 = std::chrono::steady_clock::now();
                us_a = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
                us_b = std::chrono::duration<double, std::micro>(t2 - t1).count() / reps;
            }
            printf("[kp] n=%d il=%d %s el=%lld bad_down=%ld bad_inject=%ld maxabs=%.3g first=(row %d, col %d, sep %.9g, merged %.9g) us_sep=%.1f us_merged=%.1f %s\n",
                   n, p.il, p.kind, (long long) (M * n), bad_d, bad_i, maxd, fr, fc, fa, fb, us_a, us_b, (bad_d + bad_i) ? "DIFF" : "EQUAL");
            ggml_backend_buffer_free(cbuf); ggml_free(cctx);
        }
        printf("[kp] n=%d pairs=%d bad=%ld %s\n", n, n_pairs_merged, n_bad_n, n_bad_n ? "DIFF" : "EQUAL");
    }
    printf("[kp] DONE pairs_merged=%d pairs_total=%zu elements=%ld bad=%ld verdict=%s\n", n_pairs_merged, pairs.size(), total_el, total_bad, total_bad ? "DIFF" : "EQUAL");
    ggml_backend_buffer_free(wbuf); ggml_free(wctx); ggml_backend_free(be); gguf_free(g); fclose(f);
    return total_bad ? 1 : 0;
}
