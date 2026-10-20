// E0 kernel-level classifier (Program v8.1 §2 0.6): run every case of scripts/bench/probe/e0-cases.tsv once on the
// first GPU backend and write the output bytes to <out_dir>/<case_id>.bin, so two processes with different
// GGML_VK_* knobs can be compared byte for byte (e0_classify.py). One line per case on stdout:
//   [e0-case] id=<id> op=<op> type=<type_a> m=<m used> m_case=<m> k= n= n_mats= n_used= mode= repeat_ok=<0|1> fnv=0x..
// GGML_VK_DISPATCH_LOG=1 adds the backend's [vk-dispatch] lines (the kernel each case took).
// Inputs are deterministic per case id (the same bytes in every process): A uniform [-1, 1] quantized to type_a, B
// uniform [-1, 1] f32, mul_mat_id ids = the first n_used entries of a shuffled expert permutation per token.
// Memory cap (balkon runs next to the hands server): when the f32 output would exceed --max-out-mib (default 256),
// m is cut to a multiple of 256 that fits (m=<m used> m_case=<m>): same kernel family, fewer rows.
// mode repeat3 runs the graph 3x in one process and checks the 3 outputs are identical (repeat_ok).
//
// usage: test-vk-gemm-class <cases.tsv> <out_dir> [--only id1,id2,...] [--max-out-mib N]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct ecase {
    std::string id, op, type_a, type_b, mode;
    int64_t m = 0, k = 0, n = 0, n_mats = 1, n_used = 1;
    bool bcast = false;
    std::string router = "uniform";
};

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const uint8_t *) p)[i]) * 0x100000001b3ull;
    }
    return h;
}

static ggml_type type_from(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * nm = ggml_type_name((ggml_type) t);
        if (nm && s == nm) {
            return (ggml_type) t;
        }
    }
    fprintf(stderr, "unknown type %s\n", s.c_str());
    exit(2);
}

static std::vector<ecase> read_cases(const char * path) {
    std::vector<ecase> v;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line.rfind("case_id", 0) == 0) {
            continue;
        }
        std::vector<std::string> c;
        std::stringstream ss(line);
        std::string x;
        while (std::getline(ss, x, '\t')) {
            c.push_back(x);
        }
        if (c.size() < 12) {
            continue;
        }
        ecase e;
        e.id = c[0]; e.op = c[1]; e.type_a = c[2]; e.type_b = c[3];
        e.m = atoll(c[4].c_str()); e.k = atoll(c[5].c_str()); e.n = atoll(c[6].c_str());
        e.n_mats = c[7] == "-" ? 1 : atoll(c[7].c_str());
        e.n_used = c[8] == "-" ? 1 : atoll(c[8].c_str());
        e.bcast  = c[9] == "yes";
        e.router = c[10] == "-" ? "uniform" : c[10];
        e.mode   = c[11];
        v.push_back(e);
    }
    return v;
}

static void fill_f32(std::vector<float> & v, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto & x : v) {
        x = u(rng);
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <cases.tsv> <out_dir> [--only id,...] [--max-out-mib N]\n", argv[0]);
        return 2;
    }
    std::string only;
    int64_t max_out_mib = 256;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--only") && i + 1 < argc) {
            only = std::string(",") + argv[++i] + ",";
        } else if (!strcmp(argv[i], "--max-out-mib") && i + 1 < argc) {
            max_out_mib = atoll(argv[++i]);
        }
    }
    const std::string out_dir = argv[2];
    {   // the A cache dir next to out_dir (created here: a missing dir silently disabled the cache)
        std::error_code ec;
        std::filesystem::create_directories(out_dir + "/../acache", ec);
        if (ec) {
            fprintf(stderr, "warning: cannot create %s/../acache: %s (A is regenerated per case)\n", out_dir.c_str(), ec.message().c_str());
        }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (!dev) {
        fprintf(stderr, "no GPU device\n");
        return 1;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);

    for (const ecase & e : read_cases(argv[1])) {
        if (!only.empty() && only.find("," + e.id + ",") == std::string::npos) {
            continue;
        }
        const bool    is_id = e.op == "MUL_MAT_ID";
        const int64_t rows  = is_id ? e.n_used : 1;
        int64_t m = e.m;
        const int64_t cap = max_out_mib * 1024 * 1024;
        if (m * e.n * rows * 4 > cap) {
            m = std::max<int64_t>(256, (cap / (e.n * rows * 4)) / 256 * 256);
        }
        const ggml_type ta = type_from(e.type_a);
        const ggml_type tb = type_from(e.type_b);
        // and A (quantized) under 512 MiB: only the lm_head is cut by this (the 512-expert tensor is 472 MiB)
        const int64_t a_cap = 512ll * 1024 * 1024;
        if ((int64_t) ggml_row_size(ta, e.k) * m * (is_id ? e.n_mats : 1) > a_cap) {
            m = std::max<int64_t>(256, (a_cap / ((int64_t) ggml_row_size(ta, e.k) * (is_id ? e.n_mats : 1))) / 256 * 256);
        }

        ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_3d(ctx, ta, e.k, m, is_id ? e.n_mats : 1);
        ggml_tensor * b = is_id ? ggml_new_tensor_3d(ctx, tb, e.k, e.bcast ? 1 : e.n_used, e.n)
                                : ggml_new_tensor_2d(ctx, tb, e.k, e.n);
        ggml_tensor * ids = is_id ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, e.n_used, e.n) : nullptr;
        ggml_tensor * out = is_id ? ggml_mul_mat_id(ctx, a, b, ids) : ggml_mul_mat(ctx, a, b);
        ggml_set_name(out, e.id.c_str());
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        if (!buf) {
            printf("[e0-case] id=%s ERROR alloc\n", e.id.c_str());
            fflush(stdout);
            ggml_free(ctx);
            continue;
        }

        const uint32_t seed = (uint32_t) std::hash<std::string>{}(e.id);
        {   // A: f32 uniform per case, generated and quantized in chunks of rows (the f32 copy of a 512-expert tensor
            // or of the lm_head would be GiBs of host RAM)
            const int64_t nrows = m * (is_id ? e.n_mats : 1);
            const int64_t chunk = 4096;
            const size_t  row_q = ggml_row_size(ta, e.k);
            std::vector<uint8_t> aq(ggml_nbytes(a));
            // the quantized A is cached across processes (deterministic per case id): <out_dir>/../acache/<id>-<type_a>-m<m>-k<k>-e<n_mats>.bin
            // (keyed by every input of the generator, so two case lists reusing an id cannot share a stale A)
            const std::string cpath = out_dir + "/../acache/" + e.id + "-" + e.type_a + "-m" + std::to_string(m) + "-k" +
                                      std::to_string(e.k) + "-e" + std::to_string(is_id ? e.n_mats : 1) + ".bin";
            if (FILE * fc = fopen(cpath.c_str(), "rb")) {
                const size_t got = fread(aq.data(), 1, aq.size(), fc);
                fclose(fc);
                if (got == aq.size()) {
                    ggml_backend_tensor_set(a, aq.data(), 0, aq.size());
                    goto a_done;
                }
            }
            {
            std::vector<float>   af(chunk * e.k);
            std::mt19937 rng(seed ^ 0xa5a5a5a5u);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            ggml_quantize_init(ta);
            for (int64_t r0 = 0; r0 < nrows; r0 += chunk) {
                const int64_t nr = std::min(chunk, nrows - r0);
                for (int64_t i = 0; i < nr * e.k; ++i) {
                    af[i] = u(rng);
                }
                if (ta == GGML_TYPE_F32) {
                    memcpy(aq.data() + r0 * row_q, af.data(), nr * row_q);
                } else {
                    ggml_quantize_chunk(ta, af.data(), aq.data() + r0 * row_q, 0, nr, e.k, nullptr);
                }
            }
            ggml_backend_tensor_set(a, aq.data(), 0, aq.size());
            if (FILE * fc = fopen(cpath.c_str(), "wb")) {
                fwrite(aq.data(), 1, aq.size(), fc);
                fclose(fc);
            }
            }
        }
a_done:
        {
            std::vector<float> bf(ggml_nelements(b));
            fill_f32(bf, seed ^ 0x5a5a5a5au);
            ggml_backend_tensor_set(b, bf.data(), 0, bf.size() * sizeof(float));
        }
        if (ids) {
            // routers: uniform = a shuffled permutation per token; skewed = realistic uneven loads: expert 0 in every token
            // (> 64 rows from n = 65), expert 1 in token 0 only (exactly 1 row), the rest drawn without replacement with
            // Zipf weights 1/(e+1)^1.2 over experts 2..n_as-1 (the tail gets 0 rows); hot = half the tokens use the same
            // n_used experts (one expert block of n/2 rows each), the rest uniform
            std::mt19937 rng(seed ^ 0x3c3c3c3cu);
            std::vector<int32_t> perm(e.n_mats), iv(e.n_used * e.n);
            std::vector<double> zw(e.n_mats);
            for (int64_t x = 0; x < e.n_mats; ++x) zw[x] = x < 2 ? 0.0 : 1.0 / std::pow((double) (x + 1), 1.2);
            // partial = partial expert tiles: expert e gets min(n, 1 + e % 15) rows (1..15, never a tile multiple) until the
            // n * n_used slots are used up; each token takes the n_used distinct experts with the most rows still to place
            std::vector<int64_t> left;
            if (e.router == "partial") {
                int64_t need = e.n * e.n_used;
                for (int64_t x = 0; x < e.n_mats && need > 0; ++x) { const int64_t c = std::min<int64_t>({e.n, 1 + x % 15, need}); left.push_back(c); need -= c; }
                for (int64_t x = 0; need > 0; x = (x + 1) % (int64_t) left.size()) { if (left[x] < e.n) { left[x]++; need--; } }
            }
            for (int64_t t = 0; t < e.n; ++t) {
                if (e.router == "partial") {
                    std::vector<int32_t> ord(left.size()); std::iota(ord.begin(), ord.end(), 0);
                    std::stable_sort(ord.begin(), ord.end(), [&](int32_t a, int32_t b) { return left[a] > left[b]; });
                    for (int64_t u = 0; u < e.n_used; ++u) { iv[t * e.n_used + u] = ord[u]; left[ord[u]]--; }
                    continue;
                }
                if (e.router == "skewed") {
                    std::vector<double> w = zw;
                    std::vector<int32_t> sel = { 0 };
                    if (t == 0 && e.n_used > 1) sel.push_back(1);
                    while ((int64_t) sel.size() < e.n_used) {
                        std::discrete_distribution<int> dd(w.begin(), w.end());
                        const int x = dd(rng); w[x] = 0.0; sel.push_back(x);
                    }
                    std::shuffle(sel.begin(), sel.end(), rng);
                    for (int64_t u = 0; u < e.n_used; ++u) iv[t * e.n_used + u] = sel[u];
                } else if (e.router == "hot" && (t % 2) == 0) {
                    for (int64_t u = 0; u < e.n_used; ++u) iv[t * e.n_used + u] = (int32_t) (7 + 13 * u) % e.n_mats;
                } else {
                    std::iota(perm.begin(), perm.end(), 0);
                    std::shuffle(perm.begin(), perm.end(), rng);
                    for (int64_t u = 0; u < e.n_used; ++u) iv[t * e.n_used + u] = perm[u];
                }
            }
            ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
        }

        const int reps = e.mode == "repeat3" ? 3 : 1;
        std::vector<uint8_t> first, cur(ggml_nbytes(out));
        bool repeat_ok = true;
        for (int r = 0; r < reps; ++r) {
            if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
                printf("[e0-case] id=%s ERROR compute\n", e.id.c_str());
                break;
            }
            ggml_backend_tensor_get(out, cur.data(), 0, cur.size());
            if (r == 0) {
                first = cur;
            } else {
                repeat_ok = repeat_ok && cur == first;
            }
        }
        if (!first.empty()) {
            FILE * fo = fopen((out_dir + "/" + e.id + ".bin").c_str(), "wb");
            if (fo) {
                fwrite(first.data(), 1, first.size(), fo);
                fclose(fo);
            }
            printf("[e0-case] id=%s op=%s type=%s m=%" PRId64 " m_case=%" PRId64 " k=%" PRId64 " n=%" PRId64
                   " n_mats=%" PRId64 " n_used=%" PRId64 " router=%s mode=%s repeat_ok=%d fnv=0x%016" PRIx64 "\n",
                   e.id.c_str(), e.op.c_str(), e.type_a.c_str(), m, e.m, e.k, e.n, e.n_mats, e.n_used, e.router.c_str(), e.mode.c_str(),
                   (int) repeat_ok, fnv1a(first.data(), first.size()));
        }
        fflush(stdout);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(be);
    return 0;
}
