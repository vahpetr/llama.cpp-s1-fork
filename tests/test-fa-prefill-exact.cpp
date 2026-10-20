// [TAG_FA1] bit-identity + timing harness for GGML_OP_FLASH_ATTN_EXT at the Flash-Next attention shape
// (head 256, 2 KV heads, GQA 12, K/V as cache views [D, n_head_kv, n_kv] permuted, f32 precision),
// causal cold-prompt geometry: the ubatch's nb queries are the last nb positions of the n_kv cells.
//
// Prints one line per case with an FNV-1a 64 hash of the output bytes, so two builds (or two knob
// settings of one build) can be compared bit for bit:   diff <(base --hash) <(cand --hash)
// The inputs are a pure function of (case, --seed), never of the device or the build.
//
// usage: test-fa-prefill-exact [--device NAME] [--kv a,b,c] [--nb a,b] [--types q8_0,f16] [--mask vis,plane]
//                              [--reps N] [--warmup N] [--seed S] [--extra]
//   --device  ggml_backend_dev_by_name spelling (default Vulkan0)
//   --kv      n_kv list (default 8192,32768,65536)
//   --nb      query batch list (default 2048)
//   --types   K/V cache types (default q8_0,f16)
//   --mask    vis = ggml_flash_attn_ext_vis over cell/token metadata (the prod LLAMA_KQ_VIS path),
//             plane = an f16 mask plane (default vis,plane)
//   --reps    timed graph computes per case (default 0 = hash only); prints min / median ms
//   --extra   also run the off-prod corner cases (unaligned n_kv, odd batches, decode/verify GQA folds)
//
// Every vis case is also run over the equivalent plane and the two outputs must be identical (the
// shader contract of ggml_flash_attn_ext_vis); a mismatch fails the run.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int64_t HS = 256, NH_KV = 2, GQA = 12, NH = NH_KV * GQA;

struct fa_case {
    int64_t   n_kv;
    int64_t   nb;
    ggml_type type;
    bool      vis;
};

std::vector<int64_t> parse_list(const char * s) {
    std::vector<int64_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) v.push_back(std::stoll(tok));
    }
    return v;
}

uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

struct result {
    uint64_t hash = 0;
    std::vector<float> out;
    double t_min = 0, t_med = 0;
    bool ok = false;
};

// One graph: FA over the plane (vis == false) or over the metadata (vis == true).
result run_case(ggml_backend_t be, const fa_case & c, uint32_t seed, int reps, int warmup) {
    result r;
    const int64_t n_kv = c.n_kv, nb = c.nb;
    const int64_t nb_pad = (nb + 63) / 64 * 64; // llama pads the mask rows to GGML_KQ_MASK_PAD

    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    // q as llama builds it: [D, n_head, n_tokens] -> permute(0, 2, 1, 3) = [D, n_tokens, n_head]
    ggml_tensor * q0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HS, NH, nb);
    ggml_tensor * q  = ggml_permute(ctx, q0, 0, 2, 1, 3);
    // K/V cache: [D, n_head_kv, n_kv] contiguous, the attention reads permute(0, 2, 1, 3) = [D, n_kv, n_head_kv]
    ggml_tensor * kc = ggml_new_tensor_3d(ctx, c.type, HS, NH_KV, n_kv);
    ggml_tensor * vc = ggml_new_tensor_3d(ctx, c.type, HS, NH_KV, n_kv);
    ggml_tensor * k  = ggml_permute(ctx, kc, 0, 2, 1, 3);
    ggml_tensor * v  = ggml_permute(ctx, vc, 0, 2, 1, 3);

    ggml_tensor * m = nullptr, * cells = nullptr, * toks = nullptr, * out = nullptr;
    const float scale = 1.0f / 16.0f;
    if (c.vis) {
        cells = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_kv, 4, 1);
        toks  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, nb, 1);
        out = ggml_flash_attn_ext_vis(ctx, q, k, v, cells, toks, scale, 0.0f, 0.0f, true, false, 0, 0);
    } else {
        m   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, nb_pad, 1, 1);
        out = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    }
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) {
        fprintf(stderr, "alloc failed\n");
        ggml_free(ctx);
        return r;
    }

    // inputs: a pure function of (n_kv, nb, type, seed) - identical for the vis and the plane graph
    std::mt19937 rng(seed ^ (uint32_t) (n_kv * 2654435761u) ^ (uint32_t) (nb * 40503u) ^ (uint32_t) c.type * 97u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    {
        // Q with a per-head scale so the softmax is peaked for some heads and flat for others
        std::vector<float> qd((size_t) (HS * NH * nb));
        for (int64_t t = 0; t < nb; ++t)
            for (int64_t h = 0; h < NH; ++h) {
                const float s = 0.5f + 0.25f * (float) (h % 8);
                for (int64_t d = 0; d < HS; ++d) qd[(size_t) ((t * NH + h) * HS + d)] = s * nd(rng);
            }
        ggml_backend_tensor_set(q0, qd.data(), 0, qd.size() * sizeof(float));
    }
    for (ggml_tensor * t : { kc, vc }) {
        std::vector<float> f((size_t) ggml_nelements(t));
        for (float & x : f) x = nd(rng);
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        if (c.type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) bytes.data(), (int64_t) f.size());
        } else {
            ggml_quantize_chunk(c.type, f.data(), bytes.data(), 0, f.size() / HS, HS, nullptr);
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    const int64_t p0 = n_kv - nb; // query t sits at position p0 + t; cell j holds position j
    if (c.vis) {
        std::vector<int32_t> cd((size_t) (4 * n_kv));
        for (int64_t j = 0; j < n_kv; ++j) {
            cd[j] = (int32_t) j;               // pos
            cd[n_kv + j] = 1;                  // seq 0
            cd[2 * n_kv + j] = 0;
            cd[3 * n_kv + j] = 0;
        }
        ggml_backend_tensor_set(cells, cd.data(), 0, cd.size() * sizeof(int32_t));
        std::vector<int32_t> td((size_t) (4 * nb));
        for (int64_t t = 0; t < nb; ++t) {
            td[4 * t + 0] = (int32_t) (p0 + t);
            td[4 * t + 1] = 1;
            td[4 * t + 2] = INT32_MAX;
            td[4 * t + 3] = INT32_MAX;
        }
        ggml_backend_tensor_set(toks, td.data(), 0, td.size() * sizeof(int32_t));
    } else {
        std::vector<float> mf((size_t) (n_kv * nb_pad), -INFINITY);
        for (int64_t t = 0; t < nb; ++t)
            for (int64_t j = 0; j <= p0 + t; ++j) mf[(size_t) (t * n_kv + j)] = 0.0f;
        std::vector<ggml_fp16_t> mh(mf.size());
        ggml_fp32_to_fp16_row(mf.data(), mh.data(), (int64_t) mf.size());
        ggml_backend_tensor_set(m, mh.data(), 0, mh.size() * sizeof(ggml_fp16_t));
    }

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
    } else {
        r.out.resize((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, r.out.data(), 0, ggml_nbytes(out));
        r.hash = fnv1a(r.out.data(), ggml_nbytes(out));
        r.ok = true;
        for (float x : r.out) if (!std::isfinite(x)) { r.ok = false; break; }
    }

    if (r.ok && reps > 0) {
        for (int i = 0; i < warmup; ++i) ggml_backend_graph_compute(be, gf);
        std::vector<double> ts;
        for (int i = 0; i < reps; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, gf);
            ggml_backend_synchronize(be);
            ts.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        r.t_min = ts.front();
        r.t_med = ts[ts.size() / 2];
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

} // namespace

int main(int argc, char ** argv) {
    std::string dev_name = "Vulkan0";
    std::vector<int64_t> kvs = { 8192, 32768, 65536 }, nbs = { 2048 };
    std::vector<ggml_type> types = { GGML_TYPE_Q8_0, GGML_TYPE_F16 };
    bool do_vis = true, do_plane = true, extra = false;
    int reps = 0, warmup = 2;
    uint32_t seed = 20261020u;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--device") dev_name = next();
        else if (a == "--kv") kvs = parse_list(next());
        else if (a == "--nb") nbs = parse_list(next());
        else if (a == "--reps") reps = atoi(next());
        else if (a == "--warmup") warmup = atoi(next());
        else if (a == "--seed") seed = (uint32_t) strtoul(next(), nullptr, 10);
        else if (a == "--extra") extra = true;
        else if (a == "--types") {
            types.clear();
            std::stringstream ss(next()); std::string t;
            while (std::getline(ss, t, ',')) {
                if (t == "q8_0") types.push_back(GGML_TYPE_Q8_0);
                else if (t == "f16") types.push_back(GGML_TYPE_F16);
                else { fprintf(stderr, "type %s?\n", t.c_str()); return 2; }
            }
        } else if (a == "--mask") {
            const std::string s = next();
            do_vis = s.find("vis") != std::string::npos;
            do_plane = s.find("plane") != std::string::npos;
        } else { fprintf(stderr, "unknown arg %s (see the header of %s)\n", a.c_str(), __FILE__); return 2; }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) { fprintf(stderr, "no device %s\n", dev_name.c_str()); return 2; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    std::vector<fa_case> cases;
    for (ggml_type t : types)
        for (int64_t nb : nbs)
            for (int64_t kv : kvs) {
                if (do_vis)   cases.push_back({ kv, nb, t, true });
                if (do_plane) cases.push_back({ kv, nb, t, false });
            }
    if (extra) {
        for (ggml_type t : types) {
            for (const auto & kn : std::vector<std::pair<int64_t, int64_t>>{
                     { 4133, 2048 }, { 8192, 77 }, { 8200, 512 }, { 2048, 2048 }, { 32768, 64 },
                     { 32768, 1 }, { 32768, 6 }, { 32769, 5 } }) {
                cases.push_back({ kn.first, kn.second, t, true });
                cases.push_back({ kn.first, kn.second, t, false });
            }
        }
    }

    int fails = 0;
    std::vector<float> last_vis;
    uint64_t last_vis_hash = 0;
    for (const fa_case & c : cases) {
        const result r = run_case(be, c, seed, reps, warmup);
        printf("FA1 kv=%" PRId64 " nb=%" PRId64 " type=%s mask=%s hash=%016" PRIx64 " ok=%d",
               c.n_kv, c.nb, ggml_type_name(c.type), c.vis ? "vis" : "plane", r.hash, (int) r.ok);
        if (reps > 0) printf(" ms_min=%.3f ms_med=%.3f", r.t_min, r.t_med);
        if (!r.ok) ++fails;
        if (c.vis) {
            last_vis = r.out;
            last_vis_hash = r.hash;
        } else if (do_vis && !last_vis.empty()) {
            // the case before this plane case is its vis twin (same inputs)
            const bool same = last_vis_hash == r.hash && last_vis.size() == r.out.size() &&
                              memcmp(last_vis.data(), r.out.data(), r.out.size() * sizeof(float)) == 0;
            printf(" vis==plane=%d", (int) same);
            if (!same) ++fails;
            last_vis.clear();
        }
        printf("\n");
        fflush(stdout);
    }
    printf("FA1 cases=%zu fails=%d\n", cases.size(), fails);
    ggml_backend_free(be);
    return fails == 0 ? 0 : 1;
}
