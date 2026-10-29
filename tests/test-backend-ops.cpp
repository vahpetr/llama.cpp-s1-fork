// This file defines tests for various GGML ops and backends.
// For the forward pass it asserts that the results of multiple backends computing the same GGML ops are consistent.
// For the backward pass it asserts that the gradients from backpropagation are consistent
// with the gradients obtained via the method of finite differences ("grad" mode, this is optional).
// It is also possible to check the performance ("perf" mode).
//
// this file has three sections: Section 1 does general setup, section 2 defines the GGML ops to be tested,
// and section 3 defines which tests to run.
// Quick start for adding a new GGML op: Go to section 2 and create a struct that inherits from test_case,
// then go to section 3 and add an instantiation of your struct.
//
// Environment variables:
//   GGML_TEST_TOPK_PROD=1     also run the selection cases at the deployed Qwen 3.8 Flash Next shapes
//                             (n_kv 213000, width 2051). Needs ~10 GB of backend and as much host
//                             memory, and runs for a while.
//   GGML_TEST_DUMP_IDS=<dir>  write the ids each TOP_K / TOPK_QSA / QSA_TOP_K case produced on the
//                             BACKEND UNDER TEST to <dir>/<case>.ids, row-major, one row per line,
//                             so two runs can be diffed byte-for-byte. The directory must exist and
//                             should be fresh per run. See [TAG_QSA_ID_DUMP].


// ##############################
// ## Section 1: General Setup ##
// ##############################


#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include "qsa-select-ref.h" // [TAG_QSA_WHOLE_BLOCKS] census-shaped QSA selection inputs

#include <algorithm>
#include <atomic>
#include <array>
#include <cfloat>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <ctime>
#include <future>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <unordered_map>

#ifdef __EMSCRIPTEN__
#   define N_THREADS 1
#else
#   define N_THREADS std::thread::hardware_concurrency()
#endif

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    size_t nels = ggml_nelements(tensor);
    std::vector<float> data(nels);
    {
        // parallel initialization
        static const size_t n_threads = N_THREADS;

        auto init_thread = [&](size_t start, size_t end) {
            thread_local std::default_random_engine gen(std::random_device{}());
            std::uniform_real_distribution<float> distribution(min, max);
            for (size_t i = start; i < end; i++) {
                data[i] = distribution(gen);
            }
        };

        if (n_threads == 1) {
            init_thread(0, nels);
        } else {
            std::vector<std::future<void>> tasks;
            tasks.reserve(n_threads);
            for (size_t i = 0; i < n_threads; i++) {
                size_t start =     i*nels/n_threads;
                size_t end   = (i+1)*nels/n_threads;
                tasks.push_back(std::async(std::launch::async, init_thread, start, end));
            }
            for (auto & t : tasks) {
                t.get();
            }
        }
    }

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(nels % ggml_blck_size(tensor->type) == 0);

         // dummy importance matrix
        std::vector<float> imatrix(tensor->ne[0], 1.0f);
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }

        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, nels));
        {
            // parallel quantization by block
            size_t blck_size = ggml_blck_size(tensor->type);
            size_t n_blocks = nels / blck_size;

            auto quantize_thread = [&](size_t start, size_t end) {
                ggml_quantize_chunk(tensor->type, data.data(), dataq.data(),
                    start * blck_size, end - start, blck_size, im);
            };

            const size_t min_blocks_per_thread = 1;
            const size_t n_quant_threads = std::min<size_t>(std::max<size_t>(N_THREADS/2, 1),
                                                            std::max<size_t>(1, n_blocks / min_blocks_per_thread));

            if (n_quant_threads == 1) {
                // single-threaded quantization: do all blocks in the current thread
                quantize_thread(0, n_blocks);
            } else {
                std::vector<std::future<void>> tasks;
                tasks.reserve(n_quant_threads);
                for (size_t i = 0; i < n_quant_threads; i++) {
                    size_t start =     i*n_blocks/n_quant_threads;
                    size_t end   = (i+1)*n_blocks/n_quant_threads;
                    tasks.push_back(std::async(std::launch::async, quantize_thread, start, end));
                }
                for (auto & t : tasks) {
                    t.get();
                }
            }
        }
        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, nels * ggml_type_size(tensor->type));
    } else if (tensor->type == GGML_TYPE_I64) {
        // Integers with a size of 8 bytes can be set by mirroring the float data, the specific values are again not really meaningful.
        const size_t nbytes_half = nels * sizeof(float);
        ggml_backend_tensor_set(tensor, data.data(), 0*nbytes_half, nbytes_half);
        ggml_backend_tensor_set(tensor, data.data(), 1*nbytes_half, nbytes_half);
    } else {
        GGML_ABORT("fatal error");
    }
}

// generate an F16 mask where certain blocks are randomly masked with -INF value
static void init_tensor_kq_mask(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS( int32_t, ne, tensor, ne);

    std::vector<float>       data_f32(ne0*ne1*ne2*ne3);
    std::vector<ggml_fp16_t> data_f16(ne0*ne1*ne2*ne3);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (size_t i = 0; i < data_f32.size(); i++) {
        data_f32[i] = dis(gen);
    }

    // block size
    const int blck0 = 128;
    const int blck1 = 64;

    // number of INF/zero blocks
    const int n_inf_zero_blocks = 0.2*(ne0*ne1*ne2*ne3)/(blck0*blck1);

    for (int b = 0; b < n_inf_zero_blocks; b++) {
        const int p3 = (rd() % ne3);
        const int p2 = (rd() % ne2);
        const int p1 = (rd() % ne1);
        const int p0 = (rd() % ne0);

        bool inf = rd() & 1;

        for (int i1 = 0; i1 < blck1 && p1 + i1 < ne1; i1++) {
            const int idx = p3*ne2*ne1*ne0 + p2*ne1*ne0 + (p1 + i1)*ne0 + p0;

            for (int i0 = 0; i0 < blck0 && p0 + i0 < ne0; i0++) {
                data_f32[idx + i0] = inf ? -INFINITY : 0.0f;
            }
        }
    }

    ggml_fp32_to_fp16_row(data_f32.data(), data_f16.data(), ne0*ne1*ne2*ne3);

    ggml_backend_tensor_set(tensor, data_f16.data(), 0, data_f16.size()*sizeof(ggml_fp16_t));
}

// generate a lower triangular matrix
static void init_tensor_tril(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(tensor->ne[0] == tensor->ne[1]);

    GGML_TENSOR_LOCALS(int32_t, ne, tensor, ne);
    GGML_TENSOR_LOCALS(size_t, nb, tensor, nb);

    std::vector<float> data_f32(ne0*ne1*ne2*ne3);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                for (int64_t i0 = 0; i0 < ne0; i0++) {
                    int64_t idx = (i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3) / sizeof(float);
                    if (i0 <= i1) {
                        data_f32[idx] = dis(gen);
                    } else {
                        data_f32[idx] = 0.0f;
                    }
                }
            }
        }
    }

    ggml_backend_tensor_set(tensor, data_f32.data(), 0, ggml_nbytes(tensor));
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    const auto * tt = ggml_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I64) {
                        tv.push_back((float)*(int64_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt->to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }

    return tv;
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// difference between 2 sets (Jaccard distance, 0 - no difference, 1 - no overlap)
template <typename T>
static double jdst(const T * a, const T * b, size_t n) {
    std::unordered_map<T, size_t> set_a;
    std::unordered_map<T, size_t> set_b;

    for (size_t i = 0; i < n; ++i) {
        set_a[a[i]]++;
        set_b[b[i]]++;
    }

    size_t diff = 0;

    for (const auto & p : set_a) {
        const int64_t na = p.second;
        const int64_t nb = set_b.find(p.first) != set_b.end() ? set_b.at(p.first) : 0;

        diff += std::abs(na - nb);
    }

    for (const auto & p : set_b) {
        if (set_a.find(p.first) == set_a.end()) {
            diff += p.second;
        }
    }

    return (double) diff / (2*n);
}

// [TAG_QSA_ID_DUMP] GGML_TEST_DUMP_IDS=<dir>
//
// Determinism evidence for the selection ops. The ids a TOP_K / TOPK_QSA / QSA_TOP_K case produced
// ON THE BACKEND UNDER TEST (backend1, not the CPU reference) are written row-major to
// <dir>/<case>.ids, so two runs of the same binary on the same backend can be diffed
// byte-for-byte:
//
//   GGML_TEST_DUMP_IDS=/tmp/ids.a ./bin/test-backend-ops -b Vulkan0 -o QSA_TOP_K
//   GGML_TEST_DUMP_IDS=/tmp/ids.b ./bin/test-backend-ops -b Vulkan0 -o QSA_TOP_K
//   diff -r /tmp/ids.a /tmp/ids.b && echo "selection is reproducible"
//
// This is the check "two runs produced the same OK/FAIL lines" does not make: a pass/fail digest is
// identical whenever both runs are correct, and a tie broken by a race is correct both times. The
// ORDER within a row is dumped as the backend wrote it - the comparator is order-insensitive on
// purpose (the CPU reference even swaps its first two ids), but a run-to-run difference in that
// order still means the kernel raced, so the dump keeps it.
//
// The directory must exist; a file that cannot be opened is reported once per case and the run
// continues. Use a FRESH directory per run: files are truncated, not appended, and a stale file
// from an earlier filter would diff clean by accident.
static const char * dump_ids_dir() {
    static const char * dir = getenv("GGML_TEST_DUMP_IDS");
    return dir;
}

// case string -> a file name that survives a shell and a filesystem
static std::string dump_ids_path(const char * dir, const std::string & case_name) {
    std::string name;
    name.reserve(case_name.size() + 8);
    for (const char c : case_name) {
        name += (std::isalnum((unsigned char) c) || strchr("._-=,+()[]", c) != nullptr) ? c : '_';
    }
    return std::string(dir) + "/" + name + ".ids";
}

static void dump_ids(const char * dir, const std::string & case_name, const char * backend_name,
                     const ggml_tensor * t, const std::vector<float> & vals) {
    const int64_t row_len = t->ne[0];
    const int64_t nrows   = ggml_nrows(t);

    if (row_len <= 0 || nrows <= 0 || (size_t) (row_len*nrows) != vals.size()) {
        return;
    }

    const std::string path = dump_ids_path(dir, case_name);

    std::ofstream f(path, std::ios::out | std::ios::trunc);
    if (!f) {
        printf("[GGML_TEST_DUMP_IDS] cannot write %s (does the directory exist?) ", path.c_str());
        return;
    }

    f << "# case=" << case_name << "\n";
    f << "# backend=" << backend_name << " tensor=" << ggml_get_name(t)
      << " op=" << ggml_op_name(t->op) << " nrows=" << nrows << " row_len=" << row_len << "\n";

    std::string line;
    for (int64_t r = 0; r < nrows; r++) {
        line.clear();
        for (int64_t c = 0; c < row_len; c++) {
            if (c) {
                line += ' ';
            }
            line += std::to_string((int32_t) vals[r*row_len + c]);
        }
        f << line << "\n";
    }
}

// Exact-id comparison for the selection ops (TOP_K / TOPK_QSA / QSA_TOP_K), PER ROW.
//
// A selection's contract is per row: row r of the destination must hold the ids of the top
// `row_len` values of row r of the source, each id in [0, ncols) and each id at most once. The
// ORDER inside a row is explicitly not part of it (both CPU references swap dst[0] and dst[1] to
// say so), so rows are compared as sets.
//
// Comparing the whole tensor as one multiset - which is what jdst() over all n elements does -
// accepts three wrong results: an id that migrated from one row to another, a row that repeats an
// id while another row drops one, and (since the multiset only ever sees the reference's own
// values) it never checks the range at all. On a many-row backend dispatch (one workgroup per row,
// which is exactly how the radix select is dispatched) row-crossing is the failure mode to expect,
// so the whole-tensor multiset was checking the one property that cannot fail.
//
// Returns 0 for a match; every violation adds at least 1, and max_err() for these tests is 0.
static double cmp_ids_per_row(const float * a, const float * b, size_t n, int64_t row_len, int64_t ncols) {
    GGML_ASSERT(row_len > 0 && n % (size_t) row_len == 0);

    const size_t nrows = n / (size_t) row_len;

    double diff = 0.0;

    std::vector<int32_t> ia(row_len), ib(row_len);
    for (size_t r = 0; r < nrows; r++) {
        const float * ar = a + r*row_len;
        const float * br = b + r*row_len;

        for (int64_t c = 0; c < row_len; c++) {
            ia[c] = (int32_t) ar[c];
            ib[c] = (int32_t) br[c];

            // penalize the result if the data is not integer valued
            diff += std::fabs(ar[c] - ia[c]);
            diff += std::fabs(br[c] - ib[c]);

            // ... or if an id does not point into the row it selects from
            diff += (ia[c] < 0 || ia[c] >= ncols) ? 1.0 : 0.0;
            diff += (ib[c] < 0 || ib[c] >= ncols) ? 1.0 : 0.0;
        }

        // the same SET of ids as the reference, order-insensitively
        diff += jdst(ia.data(), ib.data(), row_len);

        // ... and a selection, i.e. no id twice in a row
        std::sort(ia.begin(), ia.end());
        std::sort(ib.begin(), ib.end());
        if (std::adjacent_find(ia.begin(), ia.end()) != ia.end()) {
            diff += 1.0;
        }
        if (std::adjacent_find(ib.begin(), ib.end()) != ib.end()) {
            diff += 1.0;
        }
    }

    return diff;
}

// maximum absolute asymmetry between a and b
// asymmetry: (a - b) / (a + b)
// This is more stable than relative error if one of the values fluctuates towards zero.
// n: number of values to compare.
// expected_vals: optional vector of expected values for a. If expected_vals is not empty, filter out all comparisons where
//     a does not match any of the expected values. Needed for noncontinuous gradients where the numerical calculation can fail.
static double mean_abs_asymm(const float * a, const float * b, const size_t n, const std::vector<float> & expected_vals) {
    double sum = 0.0f;

    size_t nvalid = 0;
    for (size_t i = 0; i < n; i++) {
        if (!expected_vals.empty()) {
            bool matches_any = false;
            for (const float & ev : expected_vals) {
                if (fabsf(a[i] - ev) < 1e-3f) {
                    matches_any = true;
                    break;
                }
            }
            if (!matches_any) {
                continue;
            }
        }

        const float asymm = (a[i] - b[i]) / (a[i] + b[i]);

        sum += fabsf(asymm);
        nvalid++;
    }

    return sum/nvalid;
}

// utils for printing the variables of the test cases

static std::string var_to_str(const std::string & x) {
    return x;
}

template<typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template<typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template<typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

static std::string var_to_str(ggml_prec prec) {
    return prec == GGML_PREC_F32 ? "f32" : "def";
}

static std::string var_to_str(ggml_op_pool pool) {
    switch (pool) {
        case GGML_OP_POOL_AVG:  return "avg";
        case GGML_OP_POOL_MAX:  return "max";
        default:                return std::to_string(pool);
    }
}

static std::string var_to_str(ggml_scale_mode mode) {
    std::string str;
    switch (mode & 0xFF) {
        case GGML_SCALE_MODE_NEAREST:  str = "nearest"; break;
        case GGML_SCALE_MODE_BILINEAR: str = "bilinear"; break;
        case GGML_SCALE_MODE_BICUBIC:  str = "bicubic"; break;
        default:                       str = std::to_string(mode); break;
    }
    if (mode & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        str += "|align_corners";
    }
    if (mode & GGML_SCALE_FLAG_ANTIALIAS) {
        str += "|antialias";
    }
    return str;
}

#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, b) VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c) VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d) VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e) VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f) VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g) VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h) VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i) VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j) VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k) VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)
#define VARS_TO_STR12(a, b, c, d, e, f, g, h, i, j, k, l) VAR_TO_STR(a) + "," + VARS_TO_STR11(b, c, d, e, f, g, h, i, j, k, l)
#define VARS_TO_STR13(a, b, c, d, e, f, g, h, i, j, k, l, m) VAR_TO_STR(a) + "," + VARS_TO_STR12(b, c, d, e, f, g, h, i, j, k, l, m)
#define VARS_TO_STR14(a, b, c, d, e, f, g, h, i, j, k, l, m, n) VAR_TO_STR(a) + "," + VARS_TO_STR13(b, c, d, e, f, g, h, i, j, k, l, m, n)
#define VARS_TO_STR15(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) VAR_TO_STR(a) + "," + VARS_TO_STR14(b, c, d, e, f, g, h, i, j, k, l, m, n, o)
#define VARS_TO_STR16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) VAR_TO_STR(a) + "," + VARS_TO_STR15(b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
#define VARS_TO_STR17(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q) VAR_TO_STR(a) + "," + VARS_TO_STR16(b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)

#ifdef GGML_USE_SYCL
static bool inline _isinf(float f) {
    return (*(uint32_t *)&f & 0x7fffffff) == 0x7f800000;
}
#else
static bool inline _isinf(float f) { return std::isinf(f); }
#endif

// accept FLT_MAX as infinity
static bool isinf_or_max(float f) {
    return _isinf(f) || f == FLT_MAX || f == -FLT_MAX;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool backend_has_feature(ggml_backend_t backend, const char * feature_name) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

    auto get_features = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    if (!get_features) {
        return false;
    }

    const ggml_backend_feature * features = get_features(reg);
    if (!features) {
        return false;
    }

    for (const ggml_backend_feature * f = features; f->name; ++f) {
        if (strcmp(f->name, feature_name) == 0 && strcmp(f->value, "1") == 0) {
            return true;
        }
    }
    return false;
}

enum test_mode {
    MODE_TEST,
    MODE_PERF,
    MODE_GRAD,
    MODE_SUPPORT,
};

// Output format support similar to llama-bench
enum output_formats { CONSOLE, SQL, CSV };

static const char * output_format_str(output_formats format) {
    switch (format) {
        case CONSOLE:
            return "console";
        case SQL:
            return "sql";
        case CSV:
            return "csv";
        default:
            GGML_ABORT("invalid output format");
    }
}

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "console") {
        format = CONSOLE;
    } else if (s == "sql") {
        format = SQL;
    } else if (s == "csv") {
        format = CSV;
    } else {
        return false;
    }
    return true;
}

static std::string test_time_now() {
    time_t t = time(NULL);
    struct tm tm_buf;
#ifdef _WIN32
    if (gmtime_s(&tm_buf, &t) != 0) {
        return "";
    }
#else
    if (gmtime_r(&t, &tm_buf) == nullptr) {
        return "";
    }
#endif
    char buf[32];
    if (std::strftime(buf, sizeof(buf), "%FT%TZ", &tm_buf) == 0) {
        return "";
    }
    return buf;
}

// Test result structure for SQL output
struct test_result {
    std::string test_time;
    std::string build_commit;
    std::string backend_name;
    std::string op_name;
    std::string op_params;
    std::string test_mode;
    bool        supported;
    bool        passed;
    std::string error_message;
    double      time_us;
    double      flops;
    double      bandwidth_gb_s;
    size_t      memory_kb;
    int         n_runs;
    std::string device_description;
    std::string backend_reg_name;

    test_result() {
        // Initialize with default values
        time_us        = 0.0;
        flops          = 0.0;
        bandwidth_gb_s = 0.0;
        memory_kb      = 0;
        n_runs         = 0;
        supported      = false;
        passed         = false;

        test_time = test_time_now();

        // Set build info
        build_commit = ggml_commit();
    }

    test_result(const std::string & backend_name, const std::string & op_name, const std::string & op_params,
                const std::string & test_mode, bool supported, bool passed, const std::string & error_message = "",
                double time_us = 0.0, double flops = 0.0, double bandwidth_gb_s = 0.0, size_t memory_kb = 0,
                int n_runs = 0, const std::string & device_description = "", const std::string & backend_reg_name = "") :
        backend_name(backend_name),
        op_name(op_name),
        op_params(op_params),
        test_mode(test_mode),
        supported(supported),
        passed(passed),
        error_message(error_message),
        time_us(time_us),
        flops(flops),
        bandwidth_gb_s(bandwidth_gb_s),
        memory_kb(memory_kb),
        n_runs(n_runs),
        device_description(device_description),
        backend_reg_name(backend_reg_name) {
        test_time = test_time_now();

        // Set build info
        build_commit = ggml_commit();
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {
            "test_time", "build_commit",  "backend_name", "op_name", "op_params",      "test_mode", "supported",
            "passed",    "error_message", "time_us",      "flops",   "bandwidth_gb_s", "memory_kb", "n_runs",
            "device_description", "backend_reg_name"
        };
        return fields;
    }

    enum field_type { STRING, BOOL, INT, FLOAT };

    static field_type get_field_type(const std::string & field) {
        if (field == "supported" || field == "passed") {
            return BOOL;
        }
        if (field == "memory_kb" || field == "n_runs") {
            return INT;
        }
        if (field == "time_us" || field == "flops" || field == "bandwidth_gb_s") {
            return FLOAT;
        }
        return STRING;
    }

    std::vector<std::string> get_values() const {
        return { test_time,
                 build_commit,
                 backend_name,
                 op_name,
                 op_params,
                 test_mode,
                 std::to_string(supported),
                 std::to_string(passed),
                 error_message,
                 std::to_string(time_us),
                 std::to_string(flops),
                 std::to_string(bandwidth_gb_s),
                 std::to_string(memory_kb),
                 std::to_string(n_runs),
                 device_description,
                 backend_reg_name };
    }
};

// Printer classes for different output formats
enum class test_status_t { NOT_SUPPORTED, OK, FAIL, SKIPPED };

struct test_operation_info {
    std::string   op_name;
    std::string   op_params;
    std::string   backend_name;
    test_status_t status = test_status_t::OK;
    std::string   failure_reason;

    // Additional information fields that were previously in separate structs
    std::string error_component;
    std::string error_details;

    // Gradient info
    int64_t     gradient_index = -1;
    std::string gradient_param_name;
    float       gradient_value = 0.0f;

    // MAA error info
    double maa_error     = 0.0;
    double maa_threshold = 0.0;

    // Flags for different types of information
    bool has_error            = false;
    bool has_gradient_info    = false;
    bool has_maa_error        = false;
    bool is_compare_failure   = false;
    bool is_large_tensor_skip = false;

    test_operation_info() = default;

    test_operation_info(const std::string & op_name, const std::string & op_params, const std::string & backend_name,
                        test_status_t status = test_status_t::OK, const std::string & failure_reason = "") :
        op_name(op_name),
        op_params(op_params),
        backend_name(backend_name),
        status(status),
        failure_reason(failure_reason) {}

    // Set error information
    void set_error(const std::string & component, const std::string & details) {
        has_error       = true;
        error_component = component;
        error_details   = details;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set gradient information
    void set_gradient_info(int64_t index, const std::string & param_name, float value) {
        has_gradient_info   = true;
        gradient_index      = index;
        gradient_param_name = param_name;
        gradient_value      = value;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set MAA error information
    void set_maa_error(double error, double threshold) {
        has_maa_error = true;
        maa_error     = error;
        maa_threshold = threshold;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set compare failure
    void set_compare_failure() {
        is_compare_failure = true;
        if (status == test_status_t::OK) {
            status = test_status_t::FAIL;
        }
    }

    // Set large tensor skip
    void set_large_tensor_skip() { is_large_tensor_skip = true; }
};

struct test_summary_info {
    size_t tests_passed;
    size_t tests_total;
    bool   is_backend_summary = false;  // true for backend summary, false for test summary

    test_summary_info() = default;

    test_summary_info(size_t tests_passed, size_t tests_total, bool is_backend_summary = false) :
        tests_passed(tests_passed),
        tests_total(tests_total),
        is_backend_summary(is_backend_summary) {}
};

struct testing_start_info {
    size_t device_count;

    testing_start_info() = default;

    testing_start_info(size_t device_count) : device_count(device_count) {}
};

struct backend_init_info {
    size_t      device_index;
    size_t      total_devices;
    std::string device_name;
    bool        skipped = false;
    std::string skip_reason;
    std::string description;
    size_t      memory_total_mb = 0;
    size_t      memory_free_mb  = 0;
    bool        has_memory_info = false;

    backend_init_info() = default;

    backend_init_info(size_t device_index, size_t total_devices, const std::string & device_name, bool skipped = false,
                      const std::string & skip_reason = "", const std::string & description = "",
                      size_t memory_total_mb = 0, size_t memory_free_mb = 0, bool has_memory_info = false) :
        device_index(device_index),
        total_devices(total_devices),
        device_name(device_name),
        skipped(skipped),
        skip_reason(skip_reason),
        description(description),
        memory_total_mb(memory_total_mb),
        memory_free_mb(memory_free_mb),
        has_memory_info(has_memory_info) {}
};

struct backend_status_info {
    std::string   backend_name;
    test_status_t status;

    backend_status_info() = default;

    backend_status_info(const std::string & backend_name, test_status_t status) :
        backend_name(backend_name),
        status(status) {}
};

struct overall_summary_info {
    size_t backends_passed;
    size_t backends_total;
    bool   all_passed;

    overall_summary_info() = default;

    overall_summary_info(size_t backends_passed, size_t backends_total, bool all_passed) :
        backends_passed(backends_passed),
        backends_total(backends_total),
        all_passed(all_passed) {}
};

struct printer {
    virtual ~printer() {}

    FILE * fout = stdout;

    virtual void print_header() {}

    virtual void print_test_result(const test_result & result) = 0;

    virtual void print_footer() {}

    virtual void print_operation(const test_operation_info & info) { (void) info; }

    virtual void print_summary(const test_summary_info & info) { (void) info; }

    virtual void print_testing_start(const testing_start_info & info) { (void) info; }

    virtual void print_backend_init(const backend_init_info & info) { (void) info; }

    virtual void print_backend_status(const backend_status_info & info) { (void) info; }

    virtual void print_overall_summary(const overall_summary_info & info) { (void) info; }

    virtual void print_failed_tests(const std::vector<std::string> & failed_tests) { (void) failed_tests; }
};

struct console_printer : public printer {
    void print_test_result(const test_result & result) override {
        if (result.test_mode == "test") {
            print_test_console(result);
        } else if (result.test_mode == "perf") {
            print_perf_console(result);
        } else if (result.test_mode == "support") {
            print_support_console(result);
        }
    }

    void print_operation(const test_operation_info & info) override {
        printf("  %s(%s): ", info.op_name.c_str(), info.op_params.c_str());
        fflush(stdout);

        // Handle large tensor skip first
        if (info.is_large_tensor_skip) {
            printf("skipping large tensors for speed \n");
            return;
        }

        // Handle not supported status
        if (info.status == test_status_t::NOT_SUPPORTED) {
            if (!info.failure_reason.empty()) {
                printf("not supported [%s]\n", info.failure_reason.c_str());
            } else {
                printf("not supported [%s]\n", info.backend_name.c_str());
            }
            return;
        }

        // Handle errors and additional information
        if (info.has_error) {
            if (info.error_component == "allocation") {
                fprintf(stderr, "failed to allocate tensors [%s] ", info.backend_name.c_str());
            } else if (info.error_component == "backend") {
                fprintf(stderr, "  Failed to initialize %s backend\n", info.backend_name.c_str());
            } else {
                fprintf(stderr, "Error in %s: %s\n", info.error_component.c_str(), info.error_details.c_str());
            }
        }

        // Handle gradient info
        if (info.has_gradient_info) {
            printf("[%s] nonfinite gradient at index %" PRId64 " (%s=%f) ", info.op_name.c_str(), info.gradient_index,
                   info.gradient_param_name.c_str(), info.gradient_value);
        }

        // Handle MAA error
        if (info.has_maa_error) {
            printf("[%s] MAA = %.9f > %.9f ", info.op_name.c_str(), info.maa_error, info.maa_threshold);
        }

        // Handle compare failure
        if (info.is_compare_failure) {
            printf("compare failed ");
        }

        // Print final status
        if (info.status == test_status_t::OK) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_summary(const test_summary_info & info) override {
        if (info.is_backend_summary) {
            printf("%zu/%zu backends passed\n", info.tests_passed, info.tests_total);
        } else {
            printf("  %zu/%zu tests passed\n", info.tests_passed, info.tests_total);
        }
    }

    void print_backend_status(const backend_status_info & info) override {
        printf("  Backend %s: ", info.backend_name.c_str());
        if (info.status == test_status_t::OK) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_testing_start(const testing_start_info & info) override {
        printf("Testing %zu devices\n\n", info.device_count);
    }

    void print_backend_init(const backend_init_info & info) override {
        printf("Backend %zu/%zu: %s\n", info.device_index + 1, info.total_devices, info.device_name.c_str());

        if (info.skipped) {
            printf("  %s\n", info.skip_reason.c_str());
            return;
        }

        if (!info.description.empty()) {
            printf("  Device description: %s\n", info.description.c_str());
        }

        if (info.has_memory_info) {
            printf("  Device memory: %zu MB (%zu MB free)\n", info.memory_total_mb, info.memory_free_mb);
        }

        printf("\n");
    }

    void print_overall_summary(const overall_summary_info & info) override {
        printf("%zu/%zu backends passed\n", info.backends_passed, info.backends_total);
        if (info.all_passed) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
    }

    void print_failed_tests(const std::vector<std::string> & failed_tests) override {
        if (failed_tests.empty()) {
            return;
        }

        printf("\nFailing tests:\n");
        for (const auto & test_name : failed_tests) {
            printf("  %s\n", test_name.c_str());
        }
    }

  private:
    void print_test_console(const test_result & result) {
        printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (!result.supported) {
            printf("not supported [%s] ", result.backend_name.c_str());
            printf("\n");
            return;
        }

        if (result.passed) {
            printf("\033[1;32mOK\033[0m\n");
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        // when stdout is a pipe the verdict would otherwise sit in the stdio buffer, and an
        // abort() in the next case drops it - which misattributes that crash to this case
        fflush(stdout);
    }

    void print_perf_console(const test_result & result) {
        int len = printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (!result.supported) {
            printf("not supported\n");
            return;
        }

        // align while also leaving some margin for variations in parameters
        int align = 8;
        int last  = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        printf("%*s", last - len, "");

        printf("    %8d runs - %8.2f us/run - ", result.n_runs, result.time_us);

        if (result.flops > 0) {
            auto format_flops = [](double flops) -> std::string {
                char buf[256];
                if (flops >= 1e12) {
                    snprintf(buf, sizeof(buf), "%6.2f TFLOP", flops / 1e12);
                } else if (flops >= 1e9) {
                    snprintf(buf, sizeof(buf), "%6.2f GFLOP", flops / 1e9);
                } else if (flops >= 1e6) {
                    snprintf(buf, sizeof(buf), "%6.2f MFLOP", flops / 1e6);
                } else {
                    snprintf(buf, sizeof(buf), "%6.2f kFLOP", flops / 1e3);
                }
                return buf;
            };
            uint64_t op_flops_per_run = result.flops * result.time_us / 1e6;
            printf("%s/run - \033[1;34m%sS\033[0m", format_flops(op_flops_per_run).c_str(),
                   format_flops(result.flops).c_str());
        } else {
            printf("%8zu kB/run - \033[1;34m%7.2f GB/s\033[0m", result.memory_kb, result.bandwidth_gb_s);
        }
        printf("\n");
    }

    void print_support_console(const test_result & result) {
        printf("  %s(%s): ", result.op_name.c_str(), result.op_params.c_str());
        fflush(stdout);

        if (result.supported) {
            printf("\033[1;32mSUPPORTED\033[0m\n");
        } else {
            printf("\033[1;31mNOT SUPPORTED\033[0m\n");
        }
    }
};

struct sql_printer : public printer {
    static std::string get_sql_field_type(const std::string & field) {
        switch (test_result::get_field_type(field)) {
            case test_result::STRING:
                return "TEXT";
            case test_result::BOOL:
            case test_result::INT:
                return "INTEGER";
            case test_result::FLOAT:
                return "REAL";
            default:
                GGML_ABORT("invalid field type");
        }
    }

    void print_header() override {
        std::vector<std::string> fields = test_result::get_fields();
        fprintf(fout, "CREATE TABLE IF NOT EXISTS test_backend_ops (\n");
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "  %s %s%s\n", fields[i].c_str(), get_sql_field_type(fields[i]).c_str(),
                    i < fields.size() - 1 ? "," : "");
        }
        fprintf(fout, ");\n\n");
    }

    void print_test_result(const test_result & result) override {
        fprintf(fout, "INSERT INTO test_backend_ops (");
        std::vector<std::string> fields = test_result::get_fields();
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "%s%s", fields[i].c_str(), i < fields.size() - 1 ? ", " : "");
        }
        fprintf(fout, ") VALUES (");
        std::vector<std::string> values = result.get_values();
        for (size_t i = 0; i < values.size(); i++) {
            fprintf(fout, "'%s'%s", values[i].c_str(), i < values.size() - 1 ? ", " : "");
        }
        fprintf(fout, ");\n");
    }
};

struct csv_printer : public printer {
    void print_header() override {

        std::vector<std::string> fields     = test_result::get_fields();
        std::vector<std::string> fields_csv = get_fields_csv();
        for (size_t i = 0; i < fields.size(); i++) {
            if (std::find(std::begin(fields_csv), std::end(fields_csv), fields[i]) == std::end(fields_csv)) {
                continue;
            }
            printf("\"%s\"%s", fields[i].c_str(), i < fields.size() - 1 ? "," : "");
        }
        printf("\n");
    }

    void print_test_result(const test_result & result) override {

        std::vector<std::string> values     = result.get_values();
        std::vector<std::string> fields     = test_result::get_fields();
        std::vector<std::string> fields_csv = get_fields_csv();

        for (size_t i = 0; i < values.size(); i++) {

            if (std::find(std::begin(fields_csv), std::end(fields_csv), fields[i]) == std::end(fields_csv)) {
                continue;
            }

            // Escape quotes and wrap in quotes for CSV
            std::string escaped_value = values[i];
            size_t pos = 0;
            while ((pos = escaped_value.find("\"", pos)) != std::string::npos) {
                escaped_value.replace(pos, 1, "\"\"");
                pos += 2;
            }
            printf("\"%s\"%s", escaped_value.c_str(), i < values.size() - 1 ? "," : "");
        }
        printf("\n");
    }

    static std::vector<std::string> get_fields_csv() {
        return {
            "op_name",
            "op_params",
            "supported",
            "error_message",
            "test_mode",
            "backend_reg_name",
            "backend_name",
        };
    }

};

static std::unique_ptr<printer> create_printer(output_formats format) {
    switch (format) {
        case CONSOLE:
            return std::make_unique<console_printer>();
        case SQL:
            return std::make_unique<sql_printer>();
        case CSV:
            return std::make_unique<csv_printer>();
    }
    GGML_ABORT("invalid output format");
}

static std::mutex g_test_output_mutex;

static void print_test_result_locked(printer * output_printer, const test_result & result) {
    if (output_printer == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> guard(g_test_output_mutex);
    output_printer->print_test_result(result);
}

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) {
        return ggml_op_desc(t);
    }

    virtual std::string vars() {
        return "";
    }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;
    virtual ggml_tensor * build_graph(ggml_context * ctx, ggml_context * ctx_weights) {
        GGML_UNUSED(ctx_weights);
        return build_graph(ctx);
    }

    virtual double max_nmse_err() {
        return 1e-7;
    }

    virtual double max_nmse_err(ggml_backend_t backend) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        // See https://github.com/ggml-org/llama.cpp/pull/22976 for explanation.
        if (contains_f16 && strcmp(ggml_backend_reg_name(reg), "WebGPU") == 0) {
            return std::max(max_nmse_err(), 1e-6);
        }
        return max_nmse_err();
    }

    virtual double max_maa_err() {
        return 1e-4;
    }

    virtual double max_err() {
        return max_nmse_err();
    }

    virtual double max_err(ggml_backend_t backend) {
        return max_nmse_err(backend);
    }

    virtual double err(const float * a, const float * b, size_t n) {
        return nmse(a, b, n);
    }

    virtual float grad_eps() {
        return 1e-1f;
    }

    // If false, estimate gradient with 2 points, neglects 3rd order derivative and higher.
    // If true,  estimate gradient with 4 points, neglects 5th order derivative and higher.
    virtual bool grad_precise() {
        return false;
    }

    // Skip gradient checks if total number of gradients to be checked is larger than this (to speed up the tests).
    virtual int64_t grad_nmax() {
        return 10000;
    }

    // No effect if empty.
    // If not empty, skip all gradient checks where the numerical result does not match any of the values.
    // Needed for dealing with noncontinuous gradients (e.g. ReLU) where estimation using finite differences is unreliable.
    virtual std::vector<float> grad_expect() {
        return {};
    }

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    virtual uint64_t op_flops(ggml_tensor * t) {
        GGML_UNUSED(t);
        return 0;
    }

    virtual bool run_whole_graph() { return false; }
    virtual std::vector<ggml_tensor *> fusion_test_nodes() { return {}; }
    virtual bool use_weight_context() { return false; }

    ggml_cgraph * gf = nullptr;
    ggml_cgraph * gb = nullptr;

    static const int sentinel_size = 1024;

    test_mode mode;

    std::vector<ggml_tensor *> sentinels;

    std::string current_op_name;
    bool contains_f16 = false;

    // Used by the WebGPU backend to relax error thresholds on ops on f16 tensors
    void check_for_f16_tensor(ggml_context * ctx) {
        contains_f16 = false;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_F16) {
                contains_f16 = true;
                break;
            }
        }
    }

    void add_sentinel(ggml_context * ctx) {
        if (mode == MODE_PERF || mode == MODE_GRAD || mode == MODE_SUPPORT) {
            return;
        }
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    // hijack ggml_new_tensor to add sentinels after each tensor to check for overflows in the backend

    ggml_tensor * ggml_new_tensor(ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ::ggml_new_tensor(ctx, type, n_dims, ne);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t ne0) {
        ggml_tensor * t = ::ggml_new_tensor_1d(ctx, type, ne0);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_2d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ::ggml_new_tensor_2d(ctx, type, ne0, ne1);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_3d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ::ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_4d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_tensor * t = ::ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        add_sentinel(ctx);
        return t;
    }

    // Checks an op against the test filter, which is a comma separated list of OP names or specific variations
    bool matches_filter(ggml_tensor * op, const char * op_names_filter) {
        if (op_names_filter) {
            const auto op_name = op_desc(op);
            const auto op_full_name = op_name + "(" + vars() + ")";
            std::string_view filter(op_names_filter);
            while (!filter.empty()) {
                auto comma_pos = filter.find_first_of(',');
                const auto lparen_pos = filter.find_first_of('(');
                if (lparen_pos < comma_pos) {
                    auto rparen_pos = filter.find_first_of(')');
                    comma_pos = filter.find_first_of(',', rparen_pos);
                    const auto op_filter = filter.substr(0, comma_pos);
                    if (op_filter == op_full_name) {
                        return true;
                    }
                } else {
                    const auto op_filter = filter.substr(0, comma_pos);
                    if (op_filter == op_name) {
                        return true;
                    }
                }
                filter = comma_pos != std::string_view::npos ? filter.substr(comma_pos + 1) : "";
            }
            return false;
        } else {
            return true;
        }
    }

    test_status_t eval(ggml_backend_t backend1,
                       ggml_backend_t backend2,
                       const char *   op_names_filter,
                       printer *      output_printer) {
        mode = MODE_TEST;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        const bool use_weights = use_weight_context();

        ggml_context_ptr ctx(ggml_init(params));
        GGML_ASSERT(ctx);
        ggml_context_ptr ctx_weights(use_weights ? ggml_init(params) : nullptr);
        GGML_ASSERT(!use_weights || ctx_weights);

        gf = ggml_new_graph(ctx.get());

        // pre-graph sentinel
        add_sentinel(ctx.get());
        if (ctx_weights) {
            add_sentinel(ctx_weights.get());
        }

        ggml_tensor * out = build_graph(ctx.get(), ctx_weights.get());
        current_op_name   = op_desc(out);
        check_for_f16_tensor(ctx.get());

        if (!matches_filter(out, op_names_filter)) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            return test_status_t::SKIPPED;
        }

        // check if the backends support the ops
        bool supported = true;
        std::string unsupported_str;
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    supported = false;
                    if (unsupported_str.empty()) {
                        unsupported_str = std::string(ggml_backend_name(backend));
                    } else {
                        unsupported_str += ", " + std::string(ggml_backend_name(backend));
                    }
                }
            }
        }

        if (!supported) {
            test_result result(unsupported_str, current_op_name, vars(), "test",
                             false, false, "not supported");

            print_test_result_locked(output_printer, result);

            return test_status_t::NOT_SUPPORTED;
        }

        // post-graph sentinel
        add_sentinel(ctx.get());
        if (ctx_weights) {
            add_sentinel(ctx_weights.get());
        }

        ggml_backend_buffer_ptr buf_weights(nullptr);
        if (ctx_weights) {
            buf_weights.reset(ggml_backend_alloc_ctx_tensors(ctx_weights.get(), backend1));
            if (buf_weights == NULL) {
                printf("failed to allocate weight tensors [%s] ", ggml_backend_name(backend1));
                return test_status_t::FAIL;
            }
            ggml_backend_buffer_set_usage(buf_weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        // allocate
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend1));

        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
            return test_status_t::FAIL;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            ggml_graph_add_node(gf, sentinel);
        }

        // randomize tensors
        initialize_tensors(ctx.get());
        if (ctx_weights) {
            initialize_tensors(ctx_weights.get());
        }

        // compare
        struct callback_userdata {
            bool   ok;
            test_case * tc;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
        };

        callback_userdata ud {
            true,
            this,
            backend1,
            backend2,
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            // [TAG_QSA_ID_DUMP] determinism evidence: the ids the backend under test produced,
            // before any comparison can short-circuit the callback
            // the case's output node, named "out" by every selection test - one file per case, so a
            // graph that happens to carry a second TOP_K cannot overwrite it
            // [TAG_QSA_TOPK_NOKEYS] also the (plane, metadata) concat of QSA_TOP_K_VIS / QSA_TOP_K_NOKEYS, ids as f32
            if (dump_ids_dir() && (t1->op == GGML_OP_TOP_K || t1->op == GGML_OP_QSA_TOP_K ||
                                   (t1->op == GGML_OP_CONCAT && (ud->tc->current_op_name == "QSA_TOP_K_VIS" ||
                                                                 ud->tc->current_op_name == "QSA_TOP_K_NOKEYS"))) &&
                strcmp(ggml_get_name(t1), "out") == 0) {
                dump_ids(dump_ids_dir(), ud->tc->current_op_name + "(" + ud->tc->vars() + ")", bn1, t1, f1);
            }

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = ud->tc->err(f1.data(), f2.data(), f1.size());
            if (err > ud->tc->max_err(ud->backend1)) {
                printf("[%s] ERR = %.9f > %.9f ", ggml_op_desc(t1), err, ud->tc->max_err(ud->backend1));
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        std::vector<ggml_tensor *> fused_nodes_to_verify = fusion_test_nodes();
        if (fused_nodes_to_verify.size() == 0 && run_whole_graph()) {
            fused_nodes_to_verify.push_back(out);
        }
        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud,
                                                               run_whole_graph() ? fused_nodes_to_verify.data() : nullptr,
                                                               fused_nodes_to_verify.size());

        // Create test result
        bool        test_passed = ud.ok && cmp_ok;
        std::string error_msg   = test_passed ? "" : (!cmp_ok ? "compare failed" : "test failed");
        test_result result(ggml_backend_name(backend1), current_op_name, vars(), "test", supported, test_passed,
                           error_msg);

        print_test_result_locked(output_printer, result);

        return test_passed ? test_status_t::OK : test_status_t::FAIL;
    }

    bool eval_perf(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        mode = MODE_PERF;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        const bool use_weights = use_weight_context();

        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);
        ggml_context_ptr ctx_weights(use_weights ? ggml_init(params) : nullptr);
        GGML_ASSERT(!use_weights || ctx_weights);

        ggml_tensor * out             = build_graph(ctx.get(), ctx_weights.get());
        current_op_name               = op_desc(out);
        if (!matches_filter(out, op_names_filter)) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            return true;
        }

        if (!ggml_backend_supports_op(backend, out)) {
            // Create test result for unsupported performance test
            test_result result(ggml_backend_name(backend), current_op_name, vars(), "perf", false, false,
                               "not supported");

            output_printer->print_test_result(result);

            return true;
        }

        ggml_backend_buffer_ptr buf_weights(nullptr);
        if (ctx_weights) {
            buf_weights.reset(ggml_backend_alloc_ctx_tensors(ctx_weights.get(), backend));
            if (buf_weights == NULL) {
                printf("failed to allocate weight tensors\n");
                return false;
            }
            ggml_backend_buffer_set_usage(buf_weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        // allocate
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend)); // smart ptr

        if (buf == NULL) {
            printf("failed to allocate tensors\n");
            return false;
        }

        // randomize tensors
        initialize_tensors(ctx.get());
        if (ctx_weights) {
            initialize_tensors(ctx_weights.get());
        }

        // build graph
        ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
            return false;
        }

        // determine number of runs
        int n_runs;
        bool is_cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        if (op_flops(out) > 0) {
            // based on flops
            const uint64_t GFLOP = 1000 * 1000 * 1000;
            const uint64_t target_flops_cpu =   8ULL * GFLOP;
            const uint64_t target_flops_gpu = 100ULL * GFLOP;
            uint64_t target_flops = is_cpu ? target_flops_cpu : target_flops_gpu;
            n_runs = (int)std::min<int64_t>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_flops / op_flops(out)) + 1;
        } else {
            // based on memory size
            const size_t GB = 1ULL << 30;
            const size_t target_size_cpu =  8 * GB;
            const size_t target_size_gpu = 32 * GB;
            size_t target_size = is_cpu ? target_size_cpu : target_size_gpu;
            n_runs = (int)std::min<int64_t>(ggml_graph_size(gf) - ggml_graph_n_nodes(gf), target_size / op_size(out)) + 1;
        }

        // duplicate the op
        for (int i = 1; i < n_runs; i++) {
            ggml_graph_add_node(gf, out);
        }

        // calculate memory
        size_t mem = n_runs * op_size(out);
        auto tensor_op_size = [](ggml_tensor * t) {
            size_t size = ggml_nbytes(t);
            // add source tensors
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (t->src[i] != NULL) {
                    size += ggml_nbytes(t->src[i]);
                }
            }
            return size;
        };
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            if (ggml_is_view_op(ggml_graph_node(gf, i)->op) || ggml_graph_node(gf, i) == out) {
                continue;
            }
            mem += tensor_op_size(ggml_graph_node(gf, i));
        }

        // run
        int64_t total_time_us = 0;
        int64_t total_mem = 0;
        int total_runs = 0;
        do {
            int64_t start_time = ggml_time_us();
            ggml_status status = ggml_backend_graph_compute(backend, gf);
            if (status != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                return false;
            }
            int64_t end_time = ggml_time_us();

            total_time_us += end_time - start_time;
            total_mem += mem;
            total_runs += n_runs;
        } while (total_time_us < 1000*1000); // run for at least 1 second

        // Create test result
        double avg_time_us      = (double) total_time_us / total_runs;
        double calculated_flops = (op_flops(out) > 0) ? (op_flops(out) * total_runs) / (total_time_us / 1e6) : 0.0;
        double calculated_bandwidth =
            (op_flops(out) == 0) ? total_mem / (total_time_us / 1e6) / 1024.0 / 1024.0 / 1024.0 : 0.0;
        size_t calculated_memory_kb = op_size(out) / 1024;

        test_result result(ggml_backend_name(backend), current_op_name, vars(), "perf", true, true, "", avg_time_us,
                           calculated_flops, calculated_bandwidth, calculated_memory_kb, total_runs);

        if (output_printer) {
            output_printer->print_test_result(result);
        }

        return true;
    }

    bool eval_support(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        mode = MODE_SUPPORT;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);

        gf = ggml_new_graph_custom(ctx.get(), graph_nodes, false);

        ggml_tensor * out = build_graph(ctx.get());
        current_op_name   = op_desc(out);

        if (!matches_filter(out, op_names_filter)) {
            return true;
        }

        bool supported = ggml_backend_supports_op(backend, out);

        std::string device_desc = ggml_backend_dev_description(ggml_backend_get_device(backend));
        std::string backend_reg_name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)));

        test_result result(ggml_backend_name(backend), current_op_name, vars(), "support", supported, supported,
                           supported ? "yes" : "no", 0.0, 0.0, 0.0, 0, 0, device_desc, backend_reg_name);

        output_printer->print_test_result(result);

        return true;
    }

    bool eval_grad(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        mode = MODE_GRAD;
        const std::vector<float> expect = grad_expect();

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + 2*ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context_ptr ctx(ggml_init(params)); // smart ptr
        GGML_ASSERT(ctx);

        gf = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);
        gb = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, true);

        ggml_tensor * out = build_graph(ctx.get());

        if (!matches_filter(out, op_names_filter) || out->op == GGML_OP_OPT_STEP_ADAMW) {
            return true;
        }

        if (out->type != GGML_TYPE_F32) {
            output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                                                test_status_t::NOT_SUPPORTED,
                                                                out->name + std::string("->type != FP32")));
            return true;
        }

        // Print operation info first
        output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend)));

        // check if the backend supports the ops
        bool        supported  = true;
        bool        any_params = false;
        std::string failure_reason;

        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                supported      = false;
                failure_reason = ggml_backend_name(backend);
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM)) {
                any_params = true;
                if (t->type != GGML_TYPE_F32) {
                    supported      = false;
                    failure_reason = std::string(t->name) + "->type != FP32";
                    break;
                }
            }
        }
        if (!any_params) {
            supported      = false;
            failure_reason = op_desc(out);
        }

        if (!supported) {
            output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                                                test_status_t::NOT_SUPPORTED, failure_reason));
            return true;
        }

        int64_t ngrads = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (t->flags & GGML_TENSOR_FLAG_PARAM) {
                ngrads += ggml_nelements(t);
            }
        }
        if (ngrads > grad_nmax()) {
            test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
            info.set_large_tensor_skip();
            output_printer->print_operation(info);
            return true;
        }


        if (!ggml_is_scalar(out)) {
            out = ggml_sum(ctx.get(), out);
            ggml_set_name(out, "sum_of_out");
        }
        ggml_set_loss(out);

        ggml_build_forward_expand(gf, out);
        ggml_graph_cpy(gf, gb);
        ggml_build_backward_expand(ctx.get(), gb, nullptr);
        if (expect.size() != 1 || expect[0] != 0.0f) {
            GGML_ASSERT(ggml_graph_n_nodes(gb) > ggml_graph_n_nodes(gf));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
                GGML_ASSERT(!(t->flags & GGML_TENSOR_FLAG_PARAM) || ggml_graph_get_grad(gb, t)->op != GGML_OP_NONE);
            }
        }

        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != NULL; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!ggml_backend_supports_op(backend, t)) {
                output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                                                    test_status_t::NOT_SUPPORTED,
                                                                    ggml_backend_name(backend)));
                supported = false;
                break;
            }
            if ((t->flags & GGML_TENSOR_FLAG_PARAM) && t->type != GGML_TYPE_F32) {
                output_printer->print_operation(test_operation_info(op_desc(out), vars(), ggml_backend_name(backend),
                                                                    test_status_t::NOT_SUPPORTED,
                                                                    std::string(t->name) + "->type != FP32"));
                supported = false;
                break;
            }
        }
        if (!supported) {
            return true;
        }

        // allocate
        ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend)); // smart ptr
        if (buf == NULL) {
            test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
            info.set_error("allocation", "");
            output_printer->print_operation(info);
            return false;
        }

        initialize_tensors(ctx.get()); // Randomizes all tensors (including gradients).
        ggml_graph_reset(gb);    // Sets gradients to 1 if loss, 0 otherwise.

        ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
            return false;
        }
        status = ggml_backend_graph_compute(backend, gb);
        if (status != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
            return false;
        }

        bool ok = true;
        for (struct ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (!(t->flags & GGML_TENSOR_FLAG_PARAM)) {
                continue;
            }

            const char * bn = ggml_backend_name(backend);
            const int64_t ne = ggml_nelements(t);

            std::vector<float> ga;
            struct ggml_tensor * grad = ggml_graph_get_grad(gb, t);
            if (grad) {
                ga = tensor_to_float(grad);
            } else {
                ga.resize(ne); // default value is 0.0f
            }

            for (int64_t i = 0; i < ne; ++i) { // gradient algebraic
                // check for nans
                if (!std::isfinite(ga[i])) {
                    test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
                    info.set_gradient_info(i, bn, ga[i]);
                    output_printer->print_operation(info);
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                break;
            }

            std::vector<float> gn(ne); // gradient numeric
            GGML_ASSERT(ga.size() == gn.size());

            std::vector<float> x0 = tensor_to_float(t); // original t data
            GGML_ASSERT(ggml_is_scalar(out));
            GGML_ASSERT(out->type == GGML_TYPE_F32);

            const float eps = grad_eps();
            for (int64_t i = 0; i < ne; ++i) {
                const float xiu  = x0[i] + 1.0f*eps; // x, index i, up
                const float xiuh = x0[i] + 0.5f*eps; // x, index i, up half
                const float xidh = x0[i] - 0.5f*eps; // x, index i, down half
                const float xid  = x0[i] - 1.0f*eps; // x, index i, down

                float fu, fuh, fdh, fd; // output values for xiu, xiuh, xid, xidh

                ggml_backend_tensor_set(t, &xiu, i*sizeof(float), sizeof(float));
                status = ggml_backend_graph_compute(backend, gf);
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                    return false;
                }
                ggml_backend_tensor_get(out, &fu, 0, ggml_nbytes(out));

                ggml_backend_tensor_set(t, &xid, i*sizeof(float), sizeof(float));
                status = ggml_backend_graph_compute(backend, gf);
                if (status != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                    return false;
                }
                ggml_backend_tensor_get(out, &fd, 0, ggml_nbytes(out));

                if (grad_precise()) {
                    ggml_backend_tensor_set(t, &xiuh, i*sizeof(float), sizeof(float));
                    status = ggml_backend_graph_compute(backend, gf);
                    if (status != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                        return false;
                    }
                    ggml_backend_tensor_get(out, &fuh, 0, ggml_nbytes(out));

                    ggml_backend_tensor_set(t, &xidh, i*sizeof(float), sizeof(float));
                    status = ggml_backend_graph_compute(backend, gf);
                    if (status != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "%s: ggml_backend_graph_compute failed. status=%s \n", __func__, ggml_status_to_string(status));
                        return false;
                    }
                    ggml_backend_tensor_get(out, &fdh, 0, ggml_nbytes(out));

                    gn[i] = (8.0*(double)fuh + (double)fd - (8.0*(double)fdh + (double)fu)) / (6.0*(double)eps);
                } else {
                    gn[i] = (fu - fd) / (2.0f*eps);
                }

                ggml_backend_tensor_set(t, x0.data(), 0, ggml_nbytes(t));
            }

            const double err = mean_abs_asymm(gn.data(), ga.data(), gn.size(), expect);
            if (err > max_maa_err()) {
                test_operation_info info(op_desc(out), vars(), ggml_backend_name(backend));
                info.set_maa_error(err, max_maa_err());
                output_printer->print_operation(info);
                ok = false;
                break;
            }
            if (!ok) {
                break;
            }
        }

        // Create final test result
        test_operation_info final_info(op_desc(out), vars(), ggml_backend_name(backend));
        if (!ok) {
            final_info.set_compare_failure();
        }
        final_info.status = ok ? test_status_t::OK : test_status_t::FAIL;
        output_printer->print_operation(final_info);

        if (ok) {
            return true;
        }

        return false;
    }
};


// ####################################
// ## Section 2: GGML Op Definitions ##
// ####################################


// The following is an example showing the bare minimum for creating a test for a GGML op.

// GGML_OP_EXAMPLE
struct test_example : public test_case {
    // Always define these 2 or variants thereof:
    const ggml_type type; // The type of the input tensors.
    const std::array<int64_t, 4> ne; // The shape of the input tensors.
    // For some ops it's necessary to define multiple types or shapes for the inputs.
    // Or they may need additional parameters.

    // Put all parameters needed to fully define the test into one of the VARS_TO_STR macros.
    // In most cases these are just the properties of the struct that you defined above.
    // This is needed for info prints.
    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    // Define a constructor for the struct.
    // In most cases it will be sufficient to have the same arguments as the struct has properties
    // and just use initializer lists.
    test_example(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    // Define how a simple GGML compute graph can be constructed for the new GGML op.
    ggml_tensor * build_graph(ggml_context * ctx) override {
        // Step 1: create input tensors that don't depend on any other tensors:
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a"); // Setting names is optional but it's useful for debugging.

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        // Step 2: use the op that you want to test in the GGML compute graph.
        ggml_tensor * out = ggml_add(ctx, a, b); // For this example we're just doing a simple addition.
        ggml_set_name(out, "out");

        // Step 3: return the output tensor.
        return out;
    }
    // In order to also check the gradients for your op, add calls like ggml_set_param(a)
    // immediately after you create the tensors.
    // This is optional and only makes sense if a backward pass has actually been implemented for the new op.
};


// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, v);
    }

    test_unary(ggml_unary_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {128, 2, 2, 2},
            int v = 0)
        : op(op), type(type), ne_a(ne_a), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const bool grad_supported = op == GGML_UNARY_OP_ABS || op == GGML_UNARY_OP_SGN || op == GGML_UNARY_OP_NEG ||
            op == GGML_UNARY_OP_STEP || op == GGML_UNARY_OP_RELU || op == GGML_UNARY_OP_SILU ||
            op == GGML_UNARY_OP_EXPM1 || op == GGML_UNARY_OP_SOFTPLUS;

        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a;
            ne[0] *= 3;
            ne[1] *= 2;
            ne[2] *= 5;
            ne[3] *= 4;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            if (grad_supported) {
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            if (grad_supported) {
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");
        }

        ggml_tensor * out = ggml_unary(ctx, a, op);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        float min = -150.f;
        float max =  150.f;

        // Keep FP16 exp/expm1 inputs in-range so all backends stay finite instead of
        // disagreeing on whether overflow saturates to max-F16 or produces +inf.
        if (type == GGML_TYPE_F16 && (op == GGML_UNARY_OP_EXP || op == GGML_UNARY_OP_EXPM1)) {
            min = -10.f;
            max =  10.f;
        }

        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, min, max);
        }
    }

    float grad_eps() override {
        return 15.0f;
    }

    std::vector<float> grad_expect() override {
        if (op == GGML_UNARY_OP_ABS) {
            return {-1.0f, 1.0f};
        }
        if (op == GGML_UNARY_OP_SGN || op == GGML_UNARY_OP_STEP) {
            return {0.0f};
        }
        if (op == GGML_UNARY_OP_RELU) {
            return {0.0f, 1.0f};
        }
        return {};
    }

};

// GGML_OP_GLU
struct test_glu : public test_case {
    const ggml_glu_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)
    bool swapped;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, v, swapped);
    }

    test_glu(ggml_glu_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {128, 2, 2, 2},
            int v = 0,
            bool swapped = false)
        : op(op), type(type), ne_a(ne_a), v(v), swapped(swapped) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_name(a, "a");
        }

        ggml_tensor * out = ggml_glu(ctx, a, op, swapped);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

struct test_glu_split : public test_case {
    const ggml_glu_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, v) + ",split";
    }

    test_glu_split(ggml_glu_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {128, 2, 2, 2},
            int v = 0)
        : op(op), type(type), ne_a(ne_a), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        ggml_tensor * b;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_param(a);
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");

            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_param(b);
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_a[0], ne_a[1], ne_a[2], ne_a[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(a, "view_of_b");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_param(a);
            ggml_set_name(a, "a");

            b = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_param(b);
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_glu_split(ctx, a, b, op);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

struct test_swiglu_oai : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)
    float alpha;
    float limit;

    std::string vars() override {
        return VARS_TO_STR5(type, ne_a, v, alpha, limit);
    }

    test_swiglu_oai(ggml_type type = GGML_TYPE_F32,
                    std::array<int64_t, 4> ne_a = {128, 2, 2, 2},
                    int v = 0,
                    float alpha = 1.702f,
                    float limit = 7.0f)
        : type(type), ne_a(ne_a), v(v), alpha(alpha), limit(limit) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        ggml_tensor * b;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_param(a);
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");

            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_param(b);
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_a[0], ne_a[1], ne_a[2], ne_a[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(a, "view_of_b");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_param(a);
            ggml_set_name(a, "a");

            b = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_param(b);
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_swiglu_oai(ctx, a, b, alpha, limit);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

// GGML_OP_GET_ROWS
struct test_get_rows : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int be1; // batch size
    const int be2; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR7(type, n, m, r, be1, be2, v);
    }

    test_get_rows(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int be1 = 1, int be2 = 1, bool v = false)
        : type(type), n(n), m(m), r(r), be1(be1), be2(be2), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor_4d(ctx, type, n, m, be1, be2);
        ggml_set_name(in, "in");

        ggml_tensor * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, r, be1, be2);
        ggml_set_name(rows, "rows");
        if (v) {
            rows = ggml_view_3d(ctx, rows, r/2, be1, be2, rows->nb[1], rows->nb[2], 0);
            ggml_set_name(rows, "view_of_rows");
        }

        const bool grad_supported = ggml_is_matrix(in) && ggml_is_vector(rows);
        if (grad_supported) {
            ggml_set_param(in);
            // rows is a constant input -> no gradients
        }

        ggml_tensor * out = ggml_get_rows(ctx, in, rows);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows, and an i32 source: sized by the tensor itself (an i32 source smaller than the row ids, e.g.
                // (4, 8704) gathered by 1M ids, was written out of bounds when both were filled with r*be1*be2 ids)
                std::vector<int> data(ggml_nelements(t));
                for (size_t i = 0; i < data.size(); i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_GET_ROWS, type-preserving ([TAG_QSA_GATHER_Q8] ggml_get_rows_type(a, b, a->type)):
// the result is the source's own rows, byte for byte (f16 or whole q8_0 blocks)
struct test_get_rows_type : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int be1; // batch size
    const int be2; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR7(type, n, m, r, be1, be2, v) + ",type_dst=" + ggml_type_name(type);
    }

    test_get_rows_type(ggml_type type = GGML_TYPE_Q8_0, int n = 512, int m = 5, int r = 3, int be1 = 1, int be2 = 1, bool v = false)
        : type(type), n(n), m(m), r(r), be1(be1), be2(be2), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor_4d(ctx, type, n, m, be1, be2);
        ggml_set_name(in, "in");

        ggml_tensor * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, r, be1, be2);
        ggml_set_name(rows, "rows");
        if (v) {
            rows = ggml_view_3d(ctx, rows, r/2, be1, be2, rows->nb[1], rows->nb[2], 0);
            ggml_set_name(rows, "view_of_rows");
        }

        ggml_tensor * out = ggml_get_rows_type(ctx, in, rows, type);
        ggml_set_name(out, "out");

        return out;
    }

    // a row copy: anything but an exact match is a bug
    double max_nmse_err() override {
        return 0.0;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows
                std::vector<int> data(r*be1*be2);
                for (int i = 0; i < r*be1*be2; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * be1 * be2 * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_GET_ROWS_BACK
struct test_get_rows_back : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int b; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR6(type, n, m, r, b, v);
    }

    test_get_rows_back(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int b = 1, bool v = false)
        : type(type), n(n), m(m), r(r), b(b), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in_forward = ggml_new_tensor_3d(ctx, type, n, m, b);
        ggml_set_name(in_forward, "in_forward");

        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r, b);
        ggml_set_name(rows, "rows");
        if (v) {
            rows = ggml_view_2d(ctx, rows, r/2, b, rows->nb[1], 0);
            ggml_set_name(rows, "view_of_rows");
        }

        ggml_tensor * grad = ggml_new_tensor_3d(ctx, type, n, r, b);
        ggml_set_name(grad, "grad");

        ggml_tensor * out = ggml_get_rows_back(ctx, grad, rows, in_forward);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows
                std::vector<int> data(r*b);
                for (int i = 0; i < r*b; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * b * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

static void init_set_rows_row_ids(ggml_tensor * t, int num_rows) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    for (int i2 = 0; i2 < t->ne[2]; i2++) {
        for (int i1 = 0; i1 < t->ne[1]; i1++) {
            // generate a shuffled subset of row indices
            std::vector<int64_t> data(num_rows);
            for (int i = 0; i < num_rows; i++) {
                data[i] = i;
            }
            std::shuffle(data.begin(), data.end(), rng);
            data.resize(t->ne[0]);

            const size_t offs = i1*t->nb[1] + i2*t->nb[2];
            if (t->type == GGML_TYPE_I32) {
                // TODO: Make a template or something
                std::vector<int32_t> data_i32(t->ne[0]);
                for (int i = 0; i < t->ne[0]; i++) {
                    data_i32[i] = static_cast<int32_t>(data[i]);
                }
                ggml_backend_tensor_set(t, data_i32.data(), offs, t->ne[0]*sizeof(int32_t));
            } else {
                ggml_backend_tensor_set(t, data.data(), offs, t->ne[0]*sizeof(int64_t));
            }
        }
    }
}

// GGML_OP_SET_ROWS
struct test_set_rows : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const ggml_type type_idx;
    const std::array<int64_t, 4> ne;
    const std::array<int, 2> nr23; // broadcast only dims 2 and 3
    const int r; // rows to set
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR7(type_src, type_dst, type_idx, ne, nr23, r, v);
    }

    test_set_rows(ggml_type type_src,
            ggml_type type_dst,
            ggml_type type_idx,
            std::array<int64_t, 4> ne,
            std::array<int, 2> nr23,
            int r, bool v = false)
        : type_src(type_src), type_dst(type_dst), type_idx(type_idx), ne(ne), nr23(nr23), r(r), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * dst = ggml_new_tensor_4d(ctx, type_dst, ne[0], ne[1], ne[2]*nr23[0], ne[3]*nr23[1]);
        ggml_set_name(dst, "dst");

        ggml_tensor * src = ggml_new_tensor_4d(ctx, type_src, ne[0], r,     ne[2]*nr23[0], ne[3]*nr23[1]);
        ggml_set_name(src, "src");

        ggml_tensor * row_idxs = ggml_new_tensor_3d(ctx, type_idx, r, ne[2], ne[3]);
        ggml_set_name(row_idxs, "row_idxs");

        if (v) {
            src      = ggml_view_4d(ctx, src, ne[0], r/2, ne[2]*nr23[0], ne[3]*nr23[1], src->nb[1], src->nb[2], src->nb[3], 0);
            row_idxs = ggml_view_3d(ctx, row_idxs, r/2, ne[2], ne[3], row_idxs->nb[1], row_idxs->nb[2], 0);
            ggml_set_name(row_idxs, "view_of_rows");
        }

        ggml_tensor * out = ggml_set_rows(ctx, dst, src, row_idxs);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_is_view_op(t->op)) {
                continue;
            }
            if (t->type == GGML_TYPE_I64 || t->type == GGML_TYPE_I32) {
                init_set_rows_row_ids(t, ne[1]);
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    double max_nmse_err() override {
        if (type_dst == GGML_TYPE_Q2_0 || type_dst == GGML_TYPE_Q4_0 || type_dst == GGML_TYPE_Q4_1 ||
            type_dst == GGML_TYPE_IQ4_NL ||
            type_dst == GGML_TYPE_Q5_0 || type_dst == GGML_TYPE_Q5_1 || type_dst == GGML_TYPE_Q8_0) {
            // estimate what the max nmse error would be if one quantized value is
            // off by one. The test values are distributed in [-1,1], so it'll be
            // roughly (2.0 / 2^bits)^2, divided by the mean square value of the reference,
            // which is roughly 0.25 times the number of elements.
            double err_estimate = 1.0f/8.0f;
            if (type_src == GGML_TYPE_F16 && type_dst == GGML_TYPE_Q2_0) {
                err_estimate *= 4.0f;
            }
            if (type_dst == GGML_TYPE_Q5_0 || type_dst == GGML_TYPE_Q5_1) {
                err_estimate /= 2.0f;
            }
            if (type_dst == GGML_TYPE_Q8_0) {
                err_estimate /= 8.0f;
            }
            err_estimate *= err_estimate;
            if (type_src == GGML_TYPE_F16) {
                err_estimate *= 16.0f;
            }
            err_estimate /= 0.25f*float(ne[0] * r * ne[2]*nr23[0] * ne[3]*nr23[1]);
            return err_estimate;
        }
        return 1e-7;
    }

    // See dicussion here: https://github.com/ggml-org/llama.cpp/pull/23760#issuecomment-4566312209
    double max_nmse_err(ggml_backend_t backend) override {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        if (type_dst == GGML_TYPE_Q8_0) {
            if (strcmp(ggml_backend_reg_name(reg), "WebGPU") == 0) {
                return std::max(test_case::max_nmse_err(backend), 2e-7);
            }
            if (strcmp(ggml_backend_reg_name(reg), "HTP") == 0) {
                return std::max(test_case::max_nmse_err(backend), 5e-6);
            }
        }
        return test_case::max_nmse_err(backend);
    }
};

// GGML_OP_ROPE + GGML_OP_VIEW + GGML_OP_SET_ROWS
struct test_rope_set_rows : public test_case {
    const ggml_type type;
    const ggml_type type_idx;
    const std::array<int64_t, 4> ne_a;
    int mode;
    const int n_ctx{512};
    const int n_dims{128};

    std::string vars() override {
        return VARS_TO_STR4(type, type_idx, ne_a, mode);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "ROPE_SET_ROWS";
    }

    bool run_whole_graph() override { return true; }

    test_rope_set_rows(ggml_type type,
            ggml_type type_idx,
            std::array<int64_t, 4> ne_a,
            int mode)
        : type(type), type_idx(type_idx), ne_a(ne_a), mode(mode) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne_a[0], ne_a[1], ne_a[2], 1);
        ggml_set_name(a, "a");

        const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
        const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

        ggml_tensor * pos;
        if (is_mrope || is_vision) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2] * 4);
        } else {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2]);
        }
        ggml_set_name(pos, "pos");

        float fs = 1.4245f;
        float ef = 0.7465f;
        float af = 1.4245f;
        ggml_tensor * freq = nullptr;

        ggml_tensor * rope = nullptr;
        if (is_mrope) {
            if (is_vision) {
                GGML_ASSERT(n_dims/4 > 0);
                int rope_sections[4] = {n_dims/4, n_dims/4, 0, 0}; // Vision-RoPE only use first two dimension for image (x, y) coordinate
                rope = ggml_rope_multi(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
            } else {
                GGML_ASSERT(n_dims/3 > 0);
                int rope_sections[4] = {n_dims/3, n_dims/3, n_dims/3, 0};
                rope = ggml_rope_multi(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
            }
        } else {
            rope = ggml_rope(ctx, a, pos, ne_a[0], mode);
        }

        ggml_tensor * view = ggml_view_2d(ctx, rope, ne_a[0] * ne_a[1], ne_a[2], rope->nb[2], 0);

        ggml_tensor * dst = ggml_new_tensor_4d(ctx, type, ne_a[0] * ne_a[1], ne_a[2] * ne_a[3], 1, 1);
        ggml_set_name(dst, "dst");

        ggml_tensor * row_idxs = ggml_new_tensor_3d(ctx, type_idx, ne_a[2], 1, 1);
        ggml_set_name(row_idxs, "row_idxs");

        ggml_tensor * out = ggml_set_rows(ctx, dst, view, row_idxs);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "row_idxs") == 0) {
                if (ggml_is_view_op(t->op)) {
                    continue;
                }
                init_set_rows_row_ids(t, ne_a[2]);
            } else if (t->type == GGML_TYPE_I32) {
                // pos
                const int num_pos_ids = (mode & GGML_ROPE_TYPE_MROPE) ? ne_a[2] * 4 : ne_a[2];
                std::vector<int> data(num_pos_ids);
                for (int i = 0; i < num_pos_ids; i++) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, num_pos_ids * sizeof(int));
            } else {
                if (t->ne[0] == n_dims/2) {
                    // frequency factors in the range [0.9f, 1.1f]
                    init_tensor_uniform(t, 0.9f, 1.1f);
                } else {
                    init_tensor_uniform(t);
                }
            }
        }
    }
};

// GGML_OP_RMS_NORM + GGML_OP_MUL + GGML_OP_ROPE (+ GGML_OP_VIEW + GGML_OP_SET_ROWS)
struct test_rms_norm_mul_rope : public test_case {
    const std::array<int64_t, 4> ne;
    const float eps;
    const bool multi_add; // test a sequence of adds feeding into rms_norm
    const bool set_rows;
    const bool broadcast; // multiply by a 1D [ne0] weight, as model norm weights are
    int mode;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "RMS_NORM_MUL_ROPE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR6(ne, eps, multi_add, set_rows, broadcast, mode);
    }

    test_rms_norm_mul_rope(std::array<int64_t, 4> ne, float eps = 1e-6f, bool multi_add = false,
                           bool set_rows = false, bool broadcast = false, int mode = GGML_ROPE_TYPE_NORMAL)
        : ne(ne), eps(eps), multi_add(multi_add), set_rows(set_rows), broadcast(broadcast), mode(mode) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], 1);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], 1);
        ggml_tensor * c = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], 1);

        if (multi_add) {
            a = ggml_add(ctx, ggml_add(ctx, a, b), c);
        }

        ggml_tensor * w = broadcast ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]) : b;

        a = ggml_mul(ctx, ggml_rms_norm(ctx, a, eps), w);

        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne[2]);

        ggml_tensor * rope = ggml_rope(ctx, a, pos, ne[0], mode);

        ggml_tensor * out;

        if (set_rows) {
            ggml_tensor * view = ggml_view_2d(ctx, rope, ne[0] * ne[1], ne[2], rope->nb[2], 0);

            ggml_tensor * dst = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ne[0] * ne[1], ne[2] * ne[3], 1, 1);
            ggml_set_name(dst, "dst");

            ggml_tensor * row_idxs = ggml_new_tensor_3d(ctx, GGML_TYPE_I64, ne[2], 1, 1);
            ggml_set_name(row_idxs, "row_idxs");

            out = ggml_set_rows(ctx, dst, view, row_idxs);
            ggml_set_name(out, "out");
        } else {
            out = rope;
        }

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I64 || t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) {
                    continue;
                }

                init_set_rows_row_ids(t, ne[2]);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_ARGMAX
struct test_argmax : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_argmax(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 100, 1, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_argmax(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    double max_nmse_err() override {
        return 0.0;
    }
};

// GGML_OP_COUNT_EQUAL
struct test_count_equal : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_count_equal(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {4, 500, 1, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * a_argmax = ggml_argmax(ctx, a);
        ggml_set_name(a_argmax, "a_argmax");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        ggml_tensor * b_argmax = ggml_argmax(ctx, b);
        ggml_set_name(b_argmax, "b_argmax");

        ggml_tensor * out = ggml_count_equal(ctx, a_argmax, b_argmax);
        ggml_set_name(out, "out");

        return out;
    }

    double max_nmse_err() override {
        return 0.0;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_REPEAT
struct test_repeat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 2;
    }

    test_repeat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            std::array<int, 4> nr = {2, 2, 2, 2})
        : type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * target = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_set_name(target, "target");

        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        ggml_tensor * out = ggml_repeat(ctx, src, target);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_REPEAT_BACK
struct test_repeat_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;
    const bool v; // whether src is a noncontiguous view

    std::string vars() override {
        return VARS_TO_STR4(type, ne, nr, v);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 2;
    }

    test_repeat_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {8, 6, 4, 2},
            std::array<int, 4> nr = {2, 2, 2, 2},
            bool v = false)
        : type(type), ne(ne), nr(nr), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_set_name(src, "src");

        if (v) {
            GGML_ASSERT(ne[0] % 2 == 0);
            GGML_ASSERT(ne[1] % 2 == 0);
            GGML_ASSERT(ne[2] % 2 == 0);
            GGML_ASSERT(ne[3] % 2 == 0);
            GGML_ASSERT(nr[0] % 2 == 0 || nr[0] == 1);
            GGML_ASSERT(nr[1] % 2 == 0 || nr[1] == 1);
            GGML_ASSERT(nr[2] % 2 == 0 || nr[2] == 1);
            GGML_ASSERT(nr[3] % 2 == 0 || nr[3] == 1);

            const int64_t ne00 = nr[0] == 1 ? src->ne[0] : src->ne[0] / 2;
            const int64_t ne01 = nr[1] == 1 ? src->ne[1] : src->ne[1] / 2;
            const int64_t ne02 = nr[2] == 1 ? src->ne[2] : src->ne[2] / 2;
            const int64_t ne03 = nr[3] == 1 ? src->ne[3] : src->ne[3] / 2;

            src = ggml_view_4d(ctx, src, ne00, ne01, ne02, ne03, src->nb[1], src->nb[2], src->nb[3], 0);
        }

        ggml_tensor * target = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(target, "target");

        ggml_tensor * out = ggml_repeat_back(ctx, src, target);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_DUP
struct test_dup : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool _use_permute;

    std::string vars() override {
        std::string v = VARS_TO_STR2(type, ne);
        if (_use_permute) v += "," + VAR_TO_STR(permute);
        return v;
    }

    test_dup(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 20, 1},
            std::array<int64_t, 4> permute = {0, 0, 0, 0})
        : type(type), ne(ne), permute(permute),
            _use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        if (_use_permute) {
            src = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
            ggml_set_name(src, "src_permuted");
        }

        ggml_tensor * out = ggml_dup(ctx, src);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SET
struct test_set : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne;
    const int dim;
    const bool inplace;

    std::string vars() override {
        return VARS_TO_STR5(type_src, type_dst, ne, dim, inplace);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) + ggml_nbytes(t->src[0]);
    }

    test_set(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {6, 5, 4, 3}, int dim = 1, bool inplace = false)
        : type_src(type_src), type_dst(type_dst), ne(ne), dim(dim), inplace(inplace) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        auto ne_dst = ne;
        for (int i = 0; i < dim; ++i) {
            ne_dst[i] *= 2;
        }
        ggml_tensor * dst = ggml_new_tensor(ctx, type_dst, 4, ne_dst.data());
        ggml_set_param(dst);
        ggml_set_name(dst, "dst");

        size_t offset = 0;
        for (int i = 0; i < dim; ++i) {
            offset += ((ne_dst[i] - ne[i])/2)*dst->nb[i];
        }
        ggml_tensor * out;
        if (inplace) {
            out = ggml_set_inplace(ctx, dst, src,
                    // The backward pass requires setting a contiguous region:
                    src->nb[1], src->nb[2], src->nb[3], offset);
        } else {
            out = ggml_set(ctx, dst, src,
                    // The backward pass requires setting a contiguous region:
                    src->nb[1], src->nb[2], src->nb[3], offset);
        }
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_CPY
struct test_cpy : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne_src;
    const std::array<int64_t, 4> ne_dst;
    const std::array<int64_t, 4> permute_src;
    const std::array<int64_t, 4> permute_dst;
    const std::array<int64_t, 4> dst_alloc; // if set, dst is a view into a larger buffer (strided)
    bool _src_use_permute;
    bool _dst_use_permute;
    bool _src_transpose;
    bool _use_dst_shape;
    bool _use_dst_alloc;

    std::string vars() override {
        if (_use_dst_alloc) {
            return VARS_TO_STR8(type_src, type_dst, ne_src, ne_dst, permute_src, permute_dst, _src_transpose, dst_alloc);
        }
        if (_use_dst_shape) {
            return VARS_TO_STR7(type_src, type_dst, ne_src, ne_dst, permute_src, permute_dst, _src_transpose);
        }
        return VARS_TO_STR6(type_src, type_dst, ne_src, permute_src, permute_dst, _src_transpose);
    }

    int64_t total_elements() const {
        return ne_src[0] * ne_src[1] * ne_src[2] * ne_src[3];
    }

    double max_nmse_err() override {
        if (type_src == type_dst) {
            return 0.0;
        }
        if (type_dst == GGML_TYPE_Q4_0 || type_dst == GGML_TYPE_Q4_1 || type_dst == GGML_TYPE_IQ4_NL ||
            type_dst == GGML_TYPE_Q5_0 || type_dst == GGML_TYPE_Q5_1 || type_dst == GGML_TYPE_Q8_0) {
            // estimate what the max nmse error would be if one quantized value is
            // off by one. The test values are distributed in [-150,150], so it'll be
            // roughly (150*2.0 / 2^bits)^2, divided by the mean square value of the reference,
            // which is roughly 0.25*150^2 times the number of elements.
            double err_estimate = 1.0f/8.0f * 150.0f;
            if (type_dst == GGML_TYPE_IQ4_NL) {
                // iq4_nl values are a bit more spread out
                err_estimate *= 2.0f;
            }
            if (type_dst == GGML_TYPE_Q5_0 || type_dst == GGML_TYPE_Q5_1) {
                err_estimate /= 2.0f;
            }
            if (type_dst == GGML_TYPE_Q8_0) {
                err_estimate /= 8.0f;
            }
            err_estimate *= err_estimate;
            err_estimate /= (150.0f*150.0f*0.25f)*float(total_elements());
            return err_estimate;
        }
        return 1e-6;
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) + ggml_nbytes(t->src[0]);
    }

    test_cpy(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_src = {10, 10, 10, 1},
            std::array<int64_t, 4> ne_dst = {-1, -1, -1, -1},
            std::array<int64_t, 4> permute_src = {0, 0, 0, 0},
            std::array<int64_t, 4> permute_dst = {0, 0, 0, 0},
            bool transpose_src = false,
            std::array<int64_t, 4> dst_alloc = {0, 0, 0, 0})
        : type_src(type_src), type_dst(type_dst), ne_src(ne_src), ne_dst(ne_dst), permute_src(permute_src), permute_dst(permute_dst),
          dst_alloc(dst_alloc),
          _src_use_permute(permute_src[0] + permute_src[1] + permute_src[2] + permute_src[3] > 0),
          _dst_use_permute(permute_dst[0] + permute_dst[1] + permute_dst[2] + permute_dst[3] > 0),
          _src_transpose(transpose_src),
          _use_dst_shape(ne_dst[0] >= 0 && ne_dst[1] >= 0 && ne_dst[2] >= 0 && ne_dst[3] >= 0),
          _use_dst_alloc(dst_alloc[0] > 0){}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne_src.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        if (_src_use_permute) {
            src = ggml_permute(ctx, src, permute_src[0], permute_src[1], permute_src[2], permute_src[3]);
            ggml_set_name(src, "src_permuted");
        }

        if (_src_transpose) {
            src = ggml_transpose(ctx, src);
            ggml_set_name(src, "src_transposed");
        }

        std::array<int64_t, 4> dst_ne = _use_dst_shape ? ne_dst : std::array<int64_t, 4>{src->ne[0], src->ne[1], src->ne[2], src->ne[3]};
        ggml_tensor * dst;

        if (_use_dst_alloc) {
            // view a sub-block of a larger buffer -> strided dst
            ggml_tensor * dst_buf = ggml_new_tensor(ctx, type_dst, 4, dst_alloc.data());
            ggml_set_name(dst_buf, "dst_buf");
            dst = ggml_view_4d(ctx, dst_buf, dst_ne[0], dst_ne[1], dst_ne[2], dst_ne[3],
                dst_buf->nb[1], dst_buf->nb[2], dst_buf->nb[3], 0);
            ggml_set_name(dst, "dst_view");
        } else {
            dst = ggml_new_tensor(ctx, type_dst, 4, dst_ne.data());
            ggml_set_name(dst, "dst");

            if (_dst_use_permute) {
                dst = ggml_permute(ctx, dst, permute_dst[0], permute_dst[1], permute_dst[2], permute_dst[3]);
                ggml_set_name(dst, "dst_permuted");
            }
        }

        ggml_tensor * out = ggml_cpy(ctx, src, dst);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check if casting between f32 and i32 is consistent
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

// GGML_OP_CONT
// permute = {0, 0, 0, 0} means no permutation: the source is transposed (or
// view-sliced). A non-identity permute applies ggml_permute before ggml_cont.
struct test_cont : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    bool use_view_slice;
    const std::array<int64_t, 4> permute;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, use_view_slice, permute);
    }

    test_cont(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1},
            bool use_view_slice = false,
            std::array<int64_t, 4> permute = {0, 0, 0, 0})
        : type(type), ne(ne), use_view_slice(use_view_slice), permute(permute) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(src);
        ggml_set_name(src, "src");

        const bool permuted = permute[0] != 0 || permute[1] != 0 || permute[2] != 0 || permute[3] != 0;

        ggml_tensor * dst;
        if (permuted) {
            dst = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
            ggml_set_name(dst, "src_permuted");
        } else if (use_view_slice) {
            dst = ggml_view_4d(ctx, src, src->ne[0], 1, src->ne[2], src->ne[3],
                src->nb[1], src->nb[2], src->nb[3], src->nb[0] * (src->ne[1] - 1));
            ggml_set_name(dst, "src_view_slice");
        } else {
            dst = ggml_transpose(ctx, src);
            ggml_set_name(dst, "src_transposed");
        }

        ggml_tensor * out = ggml_cont(ctx, dst);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ADD
// GGML_OP_SUB
// GGML_OP_MUL
// GGML_OP_DIV
struct test_bin_bcast : public test_case {
    using op_t = ggml_tensor * (*) (ggml_context *, ggml_tensor *, ggml_tensor *);
    op_t op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;
    int nf; // number of fused ops, nf == 1 -> single op (no fusion)
    bool perm1; // permute src1?
    bool src_overlap; // src0 and src1 are overlapping views of the same buffer

    bool run_whole_graph() override { return nf > 1; }

    std::string vars() override {
        return VARS_TO_STR6(type, ne, nr, nf, perm1, src_overlap);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 3;
    }

    test_bin_bcast(op_t op, ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 1, 1},
            std::array<int, 4> nr = {1, 2, 1, 1},
            int nf = 1,
            bool perm1 = false, bool src_overlap = false)
        : op(op), type(type), ne(ne), nr(nr), nf(nf), perm1(perm1), src_overlap(src_overlap) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        GGML_ASSERT(nf <= 16);

        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_set_name(a, "a");

        ggml_tensor * b[16];
        for (int i = 0; i < nf; ++i) {
            if (perm1) {
                const int p[4] = { 1, 2, 0, 3 }; // hardcoded for now

                b[i] = ggml_new_tensor_4d(ctx, type, ne[p[0]], ne[p[1]], ne[p[2]], ne[p[3]]);
                b[i] = ggml_permute(ctx, b[i], p[0], p[1], p[2], p[3]);
            } else if (src_overlap) {
                b[i] = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], 2 * (ne[3] / 3), a->nb[1], a->nb[2], a->nb[3], (ne[3] / 3) * a->nb[3]);
            } else {
                b[i] = ggml_new_tensor(ctx, type, 4, ne.data());
            }
            ggml_set_name(b[i], (std::string("b") + std::to_string(i)).c_str());
        }

        // The backward pass supports broadcasting only for GGML_ADD:
        const bool grad_supported = op == ggml_add && ggml_are_same_shape(a, b[0]) && nf == 1 && !perm1;
        if (grad_supported) {
            ggml_set_param(a);
            ggml_set_param(b[0]);
        }

        ggml_tensor *out;

        if (src_overlap) {
            out = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], 2 * (ne[3] / 3), a->nb[1], a->nb[2], a->nb[3], 0);
        } else {
            out = a;
        }

        for (int i = 0; i < nf; ++i) {
            out = op(ctx, out, b[i]);
        }

        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (op == ggml_mul || op == ggml_div) {
                // MUL and DIV have numerical issues around zero:
                init_tensor_uniform(t, 0.9f, 1.1f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    float grad_eps() override {
        return 0.1f * (op == ggml_mul ? ne[0]*ne[1]*ne[2]*ne[3] : 1);
    }

    bool grad_precise() override {
        return op == ggml_div;
    }

    double max_maa_err() override {
        return op == ggml_add ? 1e-4 : 1e-3;
    }
};

// GGML_OP_ADD_ID
struct test_add_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t n_embd;
    const int64_t n_experts;
    const int64_t n_experts_used;
    const int64_t n_token;

    std::string vars() override {
        return VARS_TO_STR6(type_a, type_b, n_embd, n_experts, n_experts_used, n_token);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) + ggml_nbytes(t->src[0]) + ggml_nbytes(t->src[2]);
    }

    test_add_id(ggml_type type_a = GGML_TYPE_F32,
            ggml_type type_b = GGML_TYPE_F32,
            int64_t n_embd = 128,
            int64_t n_experts = 16,
            int64_t n_experts_used = 8,
            int64_t n_token = 10)
        : type_a(type_a), type_b(type_b), n_embd(n_embd),
          n_experts(n_experts), n_experts_used(n_experts_used), n_token(n_token) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_3d(ctx, type_a, n_embd, n_experts_used, n_token);
        ggml_tensor * b = ggml_new_tensor_2d(ctx, type_b, n_embd, n_experts);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts, n_token);
        if (n_experts_used != n_experts) {
            ids = ggml_view_2d(ctx, ids, n_experts_used, n_token, ids->nb[1], 0);
            ggml_set_name(ids, "view_of_ids");
        }

        ggml_tensor * out = ggml_add_id(ctx, a, b, ids);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                std::random_device rd;
                std::default_random_engine rng(rd());
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_experts;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_SCALE
struct test_scale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float scale;
    float bias;
    bool inplace;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, scale, bias, inplace);
    }

    test_scale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float scale = 2.0f,
            float bias = 0.0f,
            bool inplace = false)
        : type(type), ne(ne), scale(scale), bias(bias), inplace(inplace) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out;
        if (inplace) {
            out = ggml_scale_bias_inplace(ctx, a, scale, bias);
        } else {
            out = ggml_scale_bias(ctx, a, scale, bias);
        }
        ggml_set_name(out, "out");

        return out;
    }
};

// [TAG_MERGE_ROWS] GGML_OP_SCALE (+ the SILU of build_hc_mix) over a row view of a wider tensor: the qwen4exp merged
// projection output cut at row `offs` (strided at ne1 >= 2)
struct test_scale_rows : public test_case {
    const std::array<int64_t, 4> ne;   // the view
    const int64_t ne0_full;
    const int64_t offs;
    const bool silu;
    const float bias;

    std::string vars() override {
        return VARS_TO_STR5(ne, ne0_full, offs, silu, bias);
    }

    test_scale_rows(std::array<int64_t, 4> ne = {320, 6, 1, 1}, int64_t ne0_full = 324, int64_t offs = 0, bool silu = false, float bias = 0.0f)
        : ne(ne), ne0_full(ne0_full), offs(offs), silu(silu), bias(bias) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0_full, ne[1], ne[2], ne[3]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * v = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a->nb[1], a->nb[2], a->nb[3], offs*ggml_element_size(a));
        ggml_set_name(v, "view");

        ggml_tensor * out = bias != 0.0f ? ggml_scale_bias(ctx, v, 0.25f, bias) : ggml_scale(ctx, v, 0.25f);
        if (silu) {
            out = ggml_silu(ctx, out);
        }
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
struct test_softcap : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float softcap;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "SOFTCAP";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR3(type, ne, softcap);
    }

    test_softcap(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float softcap = 30.0f)
        : type(type), ne(ne), softcap(softcap) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());

        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, a, 1.0f / softcap)), softcap);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SILU_BACK
struct test_silu_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_silu_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * grad = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(grad, "grad");

        ggml_tensor * out = ggml_silu_back(ctx, a, grad);
        ggml_set_name(out, "out");

        return out;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_NORM
struct test_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool v; // whether a is a non-contiguous view
    const float eps;
    const bool noncontig_rows;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, v, eps, noncontig_rows);
    }

    test_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            bool v = false,
            float eps = 1e-6f,
            bool noncontig_rows = false)
        : type(type), ne(ne), v(v), eps(eps), noncontig_rows(noncontig_rows) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const std::array<int64_t, 4> ne_a = noncontig_rows ?
            std::array<int64_t, 4>{ ne[1], ne[0], ne[2], ne[3] } : ne;
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        if (noncontig_rows) {
            a = ggml_permute(ctx, a, 1, 0, 2, 3);
            ggml_set_name(a, "permuted a");
        }
        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0]/2, a->ne[1]/2, a->ne[2]/2, a->ne[3]/2, a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }

        ggml_tensor * out = ggml_norm(ctx, a, eps);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_NORM + GGML_OP_MUL + GGML_OP_ADD
struct test_norm_mul_add : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;
    const bool broadcast;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "NORM_MUL_ADD";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(type, ne, eps, broadcast);
    }

    test_norm_mul_add(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {128, 2, 1, 1},
            float eps = 1e-5f,
            bool broadcast = false)
        : type(type), ne(ne), eps(eps), broadcast(broadcast) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        std::array<int64_t, 4> broadcast_dims = {ne[0], ne[1] * 2, ne[2] * 2, ne[3] * 2};

        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, broadcast ? broadcast_dims.data() : ne.data());
        ggml_tensor * w = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a); ggml_set_param(w); ggml_set_param(b);
        ggml_set_name(a, "a"); ggml_set_name(w, "w"); ggml_set_name(b, "b");

        // Use a, w and b early to avoid OP_NONE in graph
        a = ggml_add(ctx, ggml_add(ctx, a, w), b);

        ggml_tensor * n = ggml_norm(ctx, a, eps);
        ggml_tensor * m = ggml_mul(ctx, n, w);
        ggml_tensor * out = ggml_add(ctx, m, b);
        ggml_set_name(out, "out");
        return out;
    }
};
// GGML_OP_RMS_NORM
struct test_rms_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool v; // whether a is a non-contiguous view
    const float eps;
    const bool inplace; // whether to do the operation inplace

    std::string vars() override {
        return VARS_TO_STR5(type, ne, v, eps, inplace);
    }

    test_rms_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            bool v = false,
            float eps = 1e-6f,
            bool inplace = false)
        : type(type), ne(ne), v(v), eps(eps), inplace(inplace) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0]/2, a->ne[1]/2, a->ne[2]/2, a->ne[3]/2, a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }

        ggml_tensor * out;
        if (inplace) {
            out = ggml_rms_norm_inplace(ctx, a, eps);
        } else {
            out = ggml_rms_norm(ctx, a, eps);
        }
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }

    float grad_eps() override {
        return 1.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_RMS_NORM_BACK
struct test_rms_norm_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_rms_norm_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_rms_norm_back(ctx, a, b, eps);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }
};

// GGML_OP_RMS_NORM + GGML_OP_MUL + GGML_OP_ADD
struct test_rms_norm_mul_add : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;
    const bool broadcast;
    const bool multi_add; // test a sequence of adds feeding into rms_norm

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "RMS_NORM_MUL_ADD";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR5(type, ne, eps, broadcast, multi_add);
    }

    test_rms_norm_mul_add(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            float eps = 1e-6f, bool broadcast = false, bool multi_add = false)
        : type(type), ne(ne), eps(eps), broadcast(broadcast), multi_add(multi_add) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        std::array<int64_t, 4> broadcast_dims = {ne[0]*2, ne[1]*3, ne[2]*3, ne[3]*4};

        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, broadcast ? broadcast_dims.data() : ne.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * c = ggml_new_tensor(ctx, type, 4, ne.data());

        ggml_set_param(a);
        ggml_set_name(a, "a");
        ggml_set_param(b);
        ggml_set_name(b, "b");
        ggml_set_param(c);
        ggml_set_name(c, "c");

        // Use a, b and c early, so we don't end up with an OP_NONE between rms_norm and mul
        a = ggml_add(ctx, ggml_add(ctx, a, b), c);
        if (multi_add) {
            a = ggml_add(ctx, ggml_add(ctx, a, b), c);
        }
        ggml_tensor * out = ggml_add(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, a, eps), b), c);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }

    float grad_eps() override {
        return 1.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_ADD + GGML_OP_RMS_NORM (fused operation)
struct test_add_rms_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;
    const bool broadcast;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "ADD_RMS_NORM";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(type, ne, eps, broadcast);
    }

    test_add_rms_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 5, 4, 3},
            float eps = 1e-6f, bool broadcast = false)
        : type(type), ne(ne), eps(eps), broadcast(broadcast) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        std::array<int64_t, 4> broadcast_dims = {ne[0]*2, ne[1]*3, ne[2]*3, ne[3]*4};

        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, broadcast ? broadcast_dims.data() : ne.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());

        ggml_set_param(a);
        ggml_set_name(a, "a");
        ggml_set_param(b);
        ggml_set_name(b, "b");

        // ADD operation followed by RMS_NORM
        ggml_tensor * add_result = ggml_add(ctx, a, b);
        ggml_set_name(add_result, "add_result");

        ggml_tensor * out = ggml_rms_norm(ctx, add_result, eps);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.f, 10.f);
        }
    }

    float grad_eps() override {
        return 1.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_UNARY(RELU) + GGML_OP_SQR (fused operation)
struct test_relu_sqr : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "RELU_SQR";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_relu_sqr(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {128, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * r = ggml_relu(ctx, a);
        ggml_set_name(r, "relu");

        ggml_tensor * out = ggml_sqr(ctx, r);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_UNARY(SILU|SIGMOID|SOFTPLUS) + GGML_OP_MUL (fused operation).
// `layout` and `tail` are used for fallback cases where fusion must be skipped
struct test_unary_mul : public test_case {
    const ggml_unary_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool swap;          // unary result is the second MUL operand
    const std::string layout; // operand layout, see build_graph()
    const std::string tail;   // extra consumer past the MUL, see build_graph()

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return std::string(ggml_unary_op_name(op)) + "_MUL";
    }

    bool run_whole_graph() override { return true; }

    double max_nmse_err() override {
        // the fused kernel elides the rounding of the unary result that the CPU chain
        // performs; relax the tolerance to match that drift
        switch (type) {
            case GGML_TYPE_F16: return 5e-5;
            default:            return 1e-7;
        }
    }

    std::string vars() override {
        return VARS_TO_STR5(type, ne, swap, layout, tail);
    }

    test_unary_mul(ggml_unary_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {128, 2, 2, 2},
            bool swap = false,
            std::string layout = "packed",
            std::string tail = "")
        : op(op), type(type), ne(ne), swap(swap), layout(std::move(layout)), tail(std::move(tail)) {}

    // `ne` viewed out of a wider tensor: rows stay contiguous, but the stride exceeds the width
    ggml_tensor * padded(ggml_context * ctx, const char * name, int64_t mul0, int64_t off0) {
        std::array<int64_t, 4> ne_w = ne;
        ne_w[0] *= mul0;
        ggml_tensor * base = ggml_new_tensor(ctx, type, 4, ne_w.data());
        ggml_set_name(base, name);
        return ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3],
                            base->nb[1], base->nb[2], base->nb[3], off0 * base->nb[0]);
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = nullptr; // unary source
        ggml_tensor * b = nullptr; // other MUL operand

        if (layout == "packed") {
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            b = ggml_new_tensor(ctx, type, 4, ne.data());
        } else if (layout == "pad_unary") {
            a = padded(ctx, "a", 3, 0);
            b = ggml_new_tensor(ctx, type, 4, ne.data());
        } else if (layout == "pad_other") {
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            b = padded(ctx, "b", 3, 0);
        } else if (layout == "halves") {
            // the shape the Conformer audio encoders build: one tensor split in two
            std::array<int64_t, 4> ne_w = ne;
            ne_w[0] *= 2;
            ggml_tensor * base = ggml_new_tensor(ctx, type, 4, ne_w.data());
            ggml_set_name(base, "base");
            b = ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3], base->nb[1], base->nb[2], base->nb[3], 0);
            a = ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3], base->nb[1], base->nb[2], base->nb[3],
                             ne[0] * base->nb[0]);
        } else if (layout == "strided_dim1") {
            // contiguous rows but a strided dim 1: not ggml_is_contiguous_1, must not fuse
            std::array<int64_t, 4> ne_w = ne;
            ne_w[1] *= 3;
            ggml_tensor * base = ggml_new_tensor(ctx, type, 4, ne_w.data());
            ggml_set_name(base, "a");
            a = ggml_view_4d(ctx, base, ne[0], ne[1], ne[2], ne[3], base->nb[1], base->nb[2], base->nb[3], 0);
            b = ggml_new_tensor(ctx, type, 4, ne.data());
        } else if (layout == "bcast") {
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            b = ggml_new_tensor_4d(ctx, type, ne[0], 1, 1, 1);
        } else if (layout == "bcast_unary") {
            // the shared-expert gate: a [1, T] unary broadcast over an [n_embd, T] activation
            // (use with swap = true, so the unary is the second MUL operand)
            a = ggml_new_tensor_4d(ctx, type, 1, ne[1], ne[2], ne[3]);
            b = ggml_new_tensor(ctx, type, 4, ne.data());
        } else {
            GGML_ABORT("unknown layout %s", layout.c_str());
        }
        ggml_set_name(a, "a");
        ggml_set_name(b, "b");

        ggml_tensor * u = ggml_unary(ctx, a, op);
        ggml_set_name(u, "unary");

        // a broadcasting operand can only be the second one
        const bool second = swap && layout != "bcast";
        ggml_tensor * out = second ? ggml_mul(ctx, b, u) : ggml_mul(ctx, u, b);

        if (tail == "reuse") {
            // a second read of the unary result must block the fusion
            ggml_set_name(out, "mul");
            out = ggml_add(ctx, out, u);
        } else if (tail == "consumer") {
            // fusion still applies; catches a dispatcher that skips one node too many
            ggml_set_name(out, "mul");
            out = ggml_add(ctx, out, b);
        } else if (!tail.empty()) {
            GGML_ABORT("unknown tail %s", tail.c_str());
        }
        ggml_set_name(out, "out");

        return out;
    }
};

// SNAKE activation fusion: y = x + sin(a*x)^2 * inv_b
// CUDA backend matches the naive 5-op chain (mul, sin, sqr, mul, add)
// and dispatches a single fused kernel.
struct test_snake_fuse : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;   // [T, C, D2, D3]

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "SNAKE_FUSE";
    }

    bool run_whole_graph() override { return true; }

    double max_nmse_err() override {
        // BF16 epsilon ~ 7.8e-3, F16 epsilon ~ 9.7e-4: relax tolerance to match
        // the natural roundoff drift between the naive CPU chain and the fused
        // CUDA kernel. F32 keeps the default tight bound.
        switch (type) {
            case GGML_TYPE_BF16: return 5e-3;
            case GGML_TYPE_F16:  return 5e-5;
            default:             return 1e-7;
        }
    }

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_snake_fuse(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {256, 192, 1, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(x, "x");

        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, ne[1]);
        ggml_set_name(a, "a");

        ggml_tensor * inv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, ne[1]);
        ggml_set_name(inv_b, "inv_b");

        // exact 5-op chain that BigVGAN / Vocos frontends emit
        ggml_tensor * ax     = ggml_mul(ctx, x, a);
        ggml_tensor * sin_ax = ggml_sin(ctx, ax);
        ggml_tensor * sin_sq = ggml_sqr(ctx, sin_ax);
        ggml_tensor * scaled = ggml_mul(ctx, sin_sq, inv_b);
        ggml_tensor * out    = ggml_add(ctx, x, scaled);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // x in [-pi, pi] to exercise sin periodicity, params in default [-1, 1]
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            const std::string name = ggml_get_name(t);
            if (name == "x") {
                init_tensor_uniform(t, -3.14159f, 3.14159f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};


// qwen4exp hyper-connection combine (build_hc_combine):
//   out[i, c, t] = residual[i, c, t] + block[i, t] * (2 * sigmoid(inject[c, t] / hc))
// emitted as SCALE -> SIGMOID -> SCALE -> REPEAT -> MUL -> ADD. The Vulkan backend matches
// the chain and dispatches one fused kernel (GGML_VK_FUSE_HC_COMBINE=0 turns it off).
struct test_hc_combine_fuse : public test_case {
    const int64_t n_embd;
    const int64_t hc;
    const int64_t n_tokens;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "HC_COMBINE_FUSE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR3(n_embd, hc, n_tokens);
    }

    test_hc_combine_fuse(int64_t n_embd = 2560, int64_t hc = 4, int64_t n_tokens = 5)
        : n_embd(n_embd), hc(hc), n_tokens(n_tokens) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_name(residual, "residual");

        ggml_tensor * block = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(block, "block");

        ggml_tensor * inject = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n_tokens);
        ggml_set_name(inject, "inject");

        ggml_tensor * w = ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float) hc));
        w = ggml_scale(ctx, w, 2.0f);
        w = ggml_reshape_3d(ctx, w, 1, hc, n_tokens);

        ggml_tensor * b = ggml_reshape_3d(ctx, block, n_embd, 1, n_tokens);
        b = ggml_repeat_4d(ctx, b, n_embd, hc, n_tokens, 1);

        ggml_tensor * out = ggml_add(ctx, residual, ggml_mul(ctx, b, w));
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            // the injection feeds a sigmoid: keep it wide enough to cover both tails
            init_tensor_uniform(t, std::string(ggml_get_name(t)) == "inject" ? -8.0f : -1.0f,
                                   std::string(ggml_get_name(t)) == "inject" ?  8.0f :  1.0f);
        }
    }
};

// [TAG_HC_MIX_TAIL] qwen4exp hyper-connection mix tail (build_hc_mix), two forms:
//
//   HC_MIX_TAIL_FUSE   build_hc_mix node for node: RMS_NORM+MUL(w_norm) -> reshape -> MUL_MAT(w_down)
//                      -> SCALE(1/hc) -> SILU -> MUL_MAT(w_up) -> SIGMOID -> MUL(xn, gate) -> reshape ->
//                      hc-1 ADDs of strided stream views -> SCALE(1/hc), against the CPU backend (NMSE). On
//                      Vulkan the graph exercises rms_norm_mul, scale_silu, sigmoid_mul and multi_add_scale
//                      (GGML_VK_FUSION_LIST=1 names what fired).
//   HC_MIX_TAIL_EXACT  the two new fusions against the unfused chain ON THE SAME BACKEND, bit for bit. Each
//                      tail is built twice from the same inputs: arm A as the model emits it (fusable), arm B
//                      with the would-be-elided intermediate flagged as a graph output, which every fusion
//                      predicate refuses (ggml_node_has_n_uses) - so arm B runs exactly the v59 dispatches
//                      (SCALE, SILU apart; multi_add, then SCALE). The graph returns concat(A - B, B - A) of
//                      both tails: +0 everywhere iff A and B are the same floats, a sign-of-zero difference
//                      leaves a -0 in one half. The CPU computes +0 everywhere and the outputs are compared
//                      BITWISE (err = number of differing bit patterns, max 0). Inputs carry exact +0 / -0
//                      entries and a wide silu range.
struct test_hc_mix_tail_fuse : public test_case {
    const int64_t n_embd;
    const int64_t hc;
    const int64_t n_tokens;
    const int64_t rank;
    const bool    exact;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return exact ? "HC_MIX_TAIL_EXACT" : "HC_MIX_TAIL_FUSE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(n_embd, hc, n_tokens, rank);
    }

    double max_nmse_err() override { return exact ? 0.0 : 5e-4; }
    double max_err() override { return exact ? 0.0 : max_nmse_err(); }
    double max_err(ggml_backend_t backend) override { GGML_UNUSED(backend); return max_err(); }

    double err(const float * a, const float * b, size_t n) override {
        if (!exact) {
            return nmse(a, b, n);
        }
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) {
            bad += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        }
        return (double) bad;
    }

    test_hc_mix_tail_fuse(int64_t n_embd = 2560, int64_t hc = 4, int64_t n_tokens = 5, int64_t rank = 320, bool exact = false)
        : n_embd(n_embd), hc(hc), n_tokens(n_tokens), rank(rank), exact(exact) {}

    // the stream mean of build_hc_mix: (((s0 + s1) + s2) + s3) over strided views, then SCALE(1/hc);
    // mark_sum flags the sum as an output so nothing can fuse the SCALE into it.
    //
    // Node order: the model's depth-first expansion interleaves VIEW s_c and ADD; in the model the
    // scheduler's ggml_vk_graph_optimize then emits the (empty) views first, which is what makes the
    // ADDs consecutive and lets multi_add match. ggml_backend_compare_graph_backend computes the graph
    // as built, without graph_optimize, so the views are expanded into gf here first, and the chain
    // right after it: the order graph_optimize produces, not the raw expansion.
    ggml_tensor * stream_mean(ggml_context * ctx, ggml_tensor * gated, bool mark_sum) {
        const size_t row = ggml_row_size(gated->type, n_embd);
        std::vector<ggml_tensor *> views;
        for (int64_t c = 0; c < hc; ++c) {
            views.push_back(ggml_view_2d(ctx, gated, n_embd, n_tokens, row * hc, row * c));
            if (gf && mode == MODE_TEST) {   // only the test mode builds gf before build_graph
                ggml_build_forward_expand(gf, views.back());
            }
        }
        ggml_tensor * mixed = views[0];
        for (int64_t c = 1; c < hc; ++c) {
            mixed = ggml_add(ctx, mixed, views[c]);
        }
        if (mark_sum && hc > 1) {
            ggml_set_output(mixed);
        }
        ggml_tensor * out = ggml_scale(ctx, mixed, 1.0f / (float) hc);
        if (gf && mode == MODE_TEST) {
            ggml_build_forward_expand(gf, out);
        }
        return out;
    }

    // silu(scale(lo, 1/hc)); mark_scale flags the scale as an output so the pair cannot fuse
    ggml_tensor * scaled_silu(ggml_context * ctx, ggml_tensor * lo, bool mark_scale) {
        ggml_tensor * s = ggml_scale(ctx, lo, 1.0f / (float) hc);
        if (mark_scale) {
            ggml_set_output(s);
        }
        return ggml_silu(ctx, s);
    }

    ggml_tensor * flat_diff(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
        ggml_tensor * d0 = ggml_sub(ctx, a, b);
        ggml_tensor * d1 = ggml_sub(ctx, b, a);
        return ggml_concat(ctx, ggml_reshape_1d(ctx, d0, ggml_nelements(d0)),
                                ggml_reshape_1d(ctx, d1, ggml_nelements(d1)), 0);
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t hc_dim = hc * n_embd;

        if (exact) {
            ggml_tensor * lo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, n_tokens);
            ggml_set_name(lo, "lo");
            ggml_tensor * gated = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
            ggml_set_name(gated, "gated");

            ggml_tensor * silu_a = scaled_silu(ctx, lo, false);
            ggml_tensor * silu_b = scaled_silu(ctx, lo, true);
            ggml_tensor * mix_a  = stream_mean(ctx, gated, false);
            ggml_tensor * mix_b  = stream_mean(ctx, gated, true);

            ggml_tensor * out = ggml_concat(ctx, flat_diff(ctx, silu_a, silu_b), flat_diff(ctx, mix_a, mix_b), 0);
            ggml_set_name(out, "out");
            return out;
        }

        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_name(x, "x");
        ggml_tensor * w_norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, hc);
        ggml_set_name(w_norm, "w_norm");
        ggml_tensor * w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc_dim, rank);
        ggml_set_name(w_down, "w_down");
        ggml_tensor * w_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, hc_dim);
        ggml_set_name(w_up, "w_up");

        ggml_tensor * xn = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), w_norm);
        xn = ggml_reshape_2d(ctx, xn, hc_dim, n_tokens);

        ggml_tensor * lo = scaled_silu(ctx, ggml_mul_mat(ctx, w_down, xn), false);
        ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo));

        ggml_tensor * gated = ggml_mul(ctx, xn, gate);
        gated = ggml_reshape_3d(ctx, gated, n_embd, hc, n_tokens);

        ggml_tensor * out = stream_mean(ctx, gated, false);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::mt19937 rng(0x9e3779b9u ^ (uint32_t) (n_embd*31 + hc*131 + n_tokens*17 + rank*7 + exact));
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE || t->type != GGML_TYPE_F32) {
                continue;
            }
            const std::string name = ggml_get_name(t);
            float lo = -1.0f, hi = 1.0f;
            if (name == "lo") {
                lo = -12.0f; hi = 12.0f;    // both silu tails
            } else if (name == "w_norm") {
                lo = 0.5f; hi = 1.5f;       // the converter folds gamma to (1 + w)
            } else if (name == "w_down" || name == "w_up") {
                const float a = 2.0f / sqrtf((float) t->ne[0]);
                lo = -a; hi = a;
            }
            std::uniform_real_distribution<float> u(lo, hi);
            std::vector<float> v(ggml_nelements(t));
            for (auto & e : v) {
                e = u(rng);
                // exact zeros of both signs in the exact form's inputs (1 in 32)
                if (exact && (rng() & 31) == 0) {
                    e = (rng() & 1) ? -0.0f : 0.0f;
                }
            }
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        }
    }
};

// [TAG_HC_MIX_TAIL] the second multi_add_scale site: build_qsa_pool's key mean (qwen4exp), r = 4 CONT slices of the
// gathered block members, ADD chain, SCALE(1/r) - [idx_dim, n_rows, n_stream] f32. Exact form as HC_MIX_TAIL_EXACT:
// arm A as the model emits it, arm B with the sum flagged as an output (multi_add then SCALE: the v59 dispatches),
// concat(A - B, B - A) compared bitwise against the CPU's +0. The slices are pre-expanded into gf, the order
// ggml_vk_graph_optimize gives them (the CONTs are independent of each other, so they lead and the ADDs follow).
struct test_qsa_pool_mean_fuse : public test_case {
    const int64_t idx_dim;
    const int64_t r;
    const int64_t n_rows;
    const int64_t n_stream;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "QSA_POOL_MEAN_EXACT";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(idx_dim, r, n_rows, n_stream);
    }

    double max_nmse_err() override { return 0.0; }
    double max_err() override { return 0.0; }
    double max_err(ggml_backend_t backend) override { GGML_UNUSED(backend); return 0.0; }

    double err(const float * a, const float * b, size_t n) override {
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) {
            bad += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        }
        return (double) bad;
    }

    test_qsa_pool_mean_fuse(int64_t idx_dim = 128, int64_t r = 4, int64_t n_rows = 512, int64_t n_stream = 1)
        : idx_dim(idx_dim), r(r), n_rows(n_rows), n_stream(n_stream) {}

    ggml_tensor * mean(ggml_context * ctx, ggml_tensor * members, bool mark_sum) {
        std::vector<ggml_tensor *> slices;
        for (int64_t i = 0; i < r; ++i) {
            slices.push_back(ggml_cont(ctx, ggml_view_3d(ctx, members, idx_dim, n_rows, n_stream,
                    members->nb[2], members->nb[3], i*members->nb[1])));
            if (gf && mode == MODE_TEST) {
                ggml_build_forward_expand(gf, slices.back());
            }
        }
        ggml_tensor * pooled = slices[0];
        for (int64_t i = 1; i < r; ++i) {
            pooled = ggml_add(ctx, pooled, slices[i]);
        }
        if (mark_sum && r > 1) {
            ggml_set_output(pooled);
        }
        ggml_tensor * out = ggml_scale(ctx, pooled, 1.0f/(float) r);
        if (gf && mode == MODE_TEST) {
            ggml_build_forward_expand(gf, out);
        }
        return out;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // the gathered members as get_rows leaves them, viewed [idx_dim, r, n_rows, n_stream]
        ggml_tensor * members = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, idx_dim, r, n_rows, n_stream);
        ggml_set_name(members, "members");

        ggml_tensor * a = mean(ctx, members, false);
        ggml_tensor * b = mean(ctx, members, true);

        ggml_tensor * d0 = ggml_sub(ctx, a, b);
        ggml_tensor * d1 = ggml_sub(ctx, b, a);
        ggml_tensor * out = ggml_concat(ctx, ggml_reshape_1d(ctx, d0, ggml_nelements(d0)),
                                             ggml_reshape_1d(ctx, d1, ggml_nelements(d1)), 0);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::mt19937 rng(0x2545f491u ^ (uint32_t) (idx_dim*31 + r*131 + n_rows*17 + n_stream*7));
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE || t->type != GGML_TYPE_F32) {
                continue;
            }
            std::vector<float> v(ggml_nelements(t));
            for (auto & e : v) {
                e = u(rng);
                if ((rng() & 31) == 0) {
                    e = (rng() & 1) ? -0.0f : 0.0f;
                }
            }
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        }
    }
};

// [TAG_CONV_INPUT_SPLIT] the qwen4exp GDN conv input: concat(state, transpose(x), dim 0) -> ssm_conv (-> + bias)
// (-> silu), with x [nr, n_t, n_s] as the qkv projection leaves it. The Vulkan backend reads the conv window from
// state | x without writing the concat (concat_ssm_conv*). exact: arm A as the model emits it, arm B with the
// concat flagged as an output (the concat is written and the conv reads it: the v59 dispatches), concat(A - B,
// B - A) compared bitwise (max 0), +/-0 inputs; otherwise arm A against the CPU (NMSE).
struct test_concat_ssm_conv_fuse : public test_case {
    const int64_t nr;
    const int64_t n_t;
    const int64_t n_s;
    const int     extra;   // 0: conv only, 1: + silu, 2: + bias + silu
    const bool    exact;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return exact ? "CONCAT_SSM_CONV_EXACT" : "CONCAT_SSM_CONV_FUSE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(nr, n_t, n_s, extra);
    }

    double max_nmse_err() override { return exact ? 0.0 : 1e-7; }
    double max_err() override { return exact ? 0.0 : max_nmse_err(); }
    double max_err(ggml_backend_t backend) override { GGML_UNUSED(backend); return max_err(); }

    double err(const float * a, const float * b, size_t n) override {
        if (!exact) {
            return nmse(a, b, n);
        }
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) {
            bad += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        }
        return (double) bad;
    }

    test_concat_ssm_conv_fuse(int64_t nr = 10240, int64_t n_t = 5, int64_t n_s = 1, int extra = 1, bool exact = true)
        : nr(nr), n_t(n_t), n_s(n_s), extra(extra), exact(exact) {}

    ggml_tensor * arm(ggml_context * ctx, ggml_tensor * state, ggml_tensor * x, ggml_tensor * w, ggml_tensor * bias, bool mark) {
        ggml_tensor * ci = ggml_concat(ctx, state, ggml_transpose(ctx, x), 0);
        if (mark) {
            ggml_set_output(ci);
        }
        ggml_tensor * y = ggml_ssm_conv(ctx, ci, w);
        if (extra == 2) {
            y = ggml_add(ctx, y, bias);
        }
        if (extra >= 1) {
            y = ggml_silu(ctx, y);
        }
        if (gf && mode == MODE_TEST) {
            ggml_build_forward_expand(gf, y);
        }
        return y;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, nr, n_s);
        ggml_set_name(state, "state");
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nr, n_t, n_s);
        ggml_set_name(x, "x");
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, nr);
        ggml_set_name(w, "w");
        ggml_tensor * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nr);
        ggml_set_name(bias, "bias");

        ggml_tensor * a = arm(ctx, state, x, w, bias, false);
        if (!exact) {
            ggml_set_name(a, "out");
            return a;
        }
        ggml_tensor * b = arm(ctx, state, x, w, bias, true);
        ggml_tensor * d0 = ggml_sub(ctx, a, b);
        ggml_tensor * d1 = ggml_sub(ctx, b, a);
        ggml_tensor * out = ggml_concat(ctx, ggml_reshape_1d(ctx, d0, ggml_nelements(d0)),
                                             ggml_reshape_1d(ctx, d1, ggml_nelements(d1)), 0);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::mt19937 rng(0x68e31da4u ^ (uint32_t) (nr*31 + n_t*131 + n_s*17 + extra*7 + exact));
        std::uniform_real_distribution<float> u(-2.0f, 2.0f);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE || t->type != GGML_TYPE_F32) {
                continue;
            }
            std::vector<float> v(ggml_nelements(t));
            for (auto & e : v) {
                e = u(rng);
                if (exact && (rng() & 31) == 0) {
                    e = (rng() & 1) ? -0.0f : 0.0f;
                }
            }
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        }
    }
};

// [TAG_CONV_INPUT_SPLIT] the conv history tails qwen4exp's build_conv_state_at writes back: the v59 form (a view of
// concat(state, x^T) - its last state_cols + slot columns) against the stage8 form (transpose of a token-range view of
// x [ch, n_t, n_s]), each written into its own history row exactly as the model does it: n_rs_seq == 0 -> cpy(cont(tail),
// row); rollback -> one cpy(tail, plane) per slot < K_write. Only the slots the model takes from x
// (n_t >= state_cols + slot) compare the two forms; the others compare the concat form with itself. The graph returns
// concat(X - C, C - X) over every written row, and err() demands an all-+0 bit pattern on BOTH backends, so a form
// that is wrong the same way on the CPU fails too. sc 3 = the GDN conv (kernel 4), sc 9 = the PLE conv
// ((kernel 4 - 1) x dilation 3).
struct test_conv_tail : public test_case {
    const int64_t sc;
    const int64_t ch;
    const int64_t n_t;
    const int64_t n_s;
    const int64_t k_write;   // 0 = the n_rs_seq == 0 form (cont + cpy); else the rollback planes 0..k_write-1

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "CONV_TAIL_EXACT";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR5(sc, ch, n_t, n_s, k_write);
    }

    double max_nmse_err() override { return 0.0; }
    double max_err() override { return 0.0; }
    double max_err(ggml_backend_t backend) override { GGML_UNUSED(backend); return 0.0; }

    double err(const float * a, const float * b, size_t n) override {
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) {
            uint32_t ua, ub;
            memcpy(&ua, &a[i], 4);
            memcpy(&ub, &b[i], 4);
            bad += (ua != 0) + (ub != 0);
        }
        return (double) bad;
    }

    test_conv_tail(int64_t sc = 3, int64_t ch = 10240, int64_t n_t = 2048, int64_t n_s = 1, int64_t k_write = 0)
        : sc(sc), ch(ch), n_t(n_t), n_s(n_s), k_write(k_write) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sc, ch, n_s);
        ggml_set_name(state, "state");
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ch, n_t, n_s);
        ggml_set_name(x, "x");
        ggml_tensor * ci = ggml_concat(ctx, state, ggml_transpose(ctx, x), 0);

        auto tail_c = [&](int64_t back) {
            return ggml_view_3d(ctx, ci, sc, ch, n_s, ci->nb[1], ci->nb[2], ggml_row_size(ci->type, ci->ne[0] - back));
        };
        // GGML_TEST_CONV_TAIL_NEGATIVE=1: the e787a6b86 form (a view OF x^T, nb[0] forced to 4) - must FAIL (review C1)
        static const bool negative = getenv("GGML_TEST_CONV_TAIL_NEGATIVE") != nullptr;
        auto tail_x = [&](int64_t back) {
            if (negative) {
                ggml_tensor * x_t = ggml_transpose(ctx, x);
                return ggml_view_3d(ctx, x_t, sc, ch, n_s, x_t->nb[1], x_t->nb[2], (n_t - back)*x_t->nb[0]);
            }
            return ggml_transpose(ctx, ggml_view_3d(ctx, x, ch, sc, n_s, x->nb[1], x->nb[2], (n_t - back)*x->nb[1]));
        };

        ggml_tensor * out = nullptr;
        const int64_t n_slots = k_write == 0 ? 1 : std::min<int64_t>(k_write, n_t);
        for (int64_t slot = 0; slot < n_slots; ++slot) {
            const int64_t back = sc + slot;
            const bool    from_x = n_t >= back;
            ggml_tensor * row_c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sc*ch, n_s);
            ggml_tensor * row_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sc*ch, n_s);
            ggml_tensor * tc = tail_c(back);
            ggml_tensor * tx = from_x ? tail_x(back) : tail_c(back);
            ggml_tensor * wc = ggml_cpy(ctx, k_write == 0 ? ggml_cont(ctx, tc) : tc, row_c);
            ggml_tensor * wx = ggml_cpy(ctx, k_write == 0 ? ggml_cont(ctx, tx) : tx, row_x);
            ggml_tensor * d = ggml_concat(ctx, ggml_reshape_1d(ctx, ggml_sub(ctx, wx, wc), sc*ch*n_s),
                                               ggml_reshape_1d(ctx, ggml_sub(ctx, wc, wx), sc*ch*n_s), 0);
            out = out ? ggml_concat(ctx, out, d, 0) : d;
        }
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::mt19937 rng(0x1b873593u ^ (uint32_t) (sc*31 + ch*131 + n_t*17 + n_s*7 + k_write));
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE || t->type != GGML_TYPE_F32) {
                continue;
            }
            std::vector<float> v(ggml_nelements(t));
            for (auto & e : v) {
                e = u(rng);   // finite, distinct: a wrong element shows as a non-zero difference
            }
            ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(float));
        }
    }
};

// Fused Gated-DeltaNet decay pre-chain (qwen4exp build_layer_attn_linear):
//   ADD(alpha, ssm_dt) -> UNARY(SOFTPLUS) -> MUL(., ssm_a)
// Production geometry is nh = num_v_heads = 48 over the draft-length token range.
// With extra_consumer the softplus feeds a second op, which must veto the fusion and
// fall back to the three-dispatch chain.
struct test_add_softplus_mul_fuse : public test_case {
    const int64_t nh;
    const int64_t n_tokens;
    const int64_t n_seqs;
    const bool    extra_consumer;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "ADD_SOFTPLUS_MUL_FUSE";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(nh, n_tokens, n_seqs, extra_consumer);
    }

    test_add_softplus_mul_fuse(int64_t nh = 48, int64_t n_tokens = 5, int64_t n_seqs = 1, bool extra_consumer = false)
        : nh(nh), n_tokens(n_tokens), n_seqs(n_seqs), extra_consumer(extra_consumer) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * alpha = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nh, n_tokens, n_seqs);
        ggml_set_name(alpha, "alpha");

        ggml_tensor * dt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nh);
        ggml_set_name(dt, "dt");

        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nh);
        ggml_set_name(a, "a");

        ggml_tensor * softplus = ggml_softplus(ctx, ggml_add(ctx, alpha, dt));
        ggml_tensor * out      = ggml_mul(ctx, softplus, a);

        if (extra_consumer) {
            out = ggml_add(ctx, out, softplus);
        }

        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            // alpha + dt has to cross the x > 20 branch of op_softplus in both directions
            const bool wide = std::string(ggml_get_name(t)) == "alpha";
            init_tensor_uniform(t, wide ? -30.0f : -2.0f, wide ? 30.0f : 2.0f);
        }
    }
};

struct test_dsv4_hc : public test_case {
    static constexpr int64_t hc = 4;

    ggml_tensor * out = nullptr;

    static uint32_t tensor_seed(const ggml_tensor * t) {
        uint32_t seed = 2166136261u;
        for (const char * p = ggml_get_name(t); *p; ++p) {
            seed ^= (uint8_t) *p;
            seed *= 16777619u;
        }
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            seed ^= (uint32_t) t->ne[i];
            seed *= 16777619u;
        }
        return seed;
    }

    static bool tensor_range(const std::string & name, float & lo, float & hi) {
        if (name == "mixes") {
            lo = -2.0f; hi = 2.0f; return true;
        }
        if (name == "scale") {
            lo = -0.5f; hi = 0.5f; return true;
        }
        if (name == "base") {
            lo = -0.25f; hi = 0.25f; return true;
        }
        if (name == "weights" || name == "comb") {
            lo = 0.0f; hi = 1.0f; return true;
        }
        if (name == "post") {
            lo = 0.0f; hi = 2.0f; return true;
        }
        if (name == "x" || name == "residual") {
            lo = -1.0f; hi = 1.0f; return true;
        }
        return false;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            const std::string name = ggml_get_name(t);
            float lo;
            float hi;
            if (!tensor_range(name, lo, hi)) {
                init_tensor_uniform(t);
                continue;
            }

            GGML_ASSERT(t->type == GGML_TYPE_F32);
            std::mt19937 rng(tensor_seed(t));
            std::uniform_real_distribution<float> dist(lo, hi);
            std::vector<float> data(ggml_nelements(t));
            for (float & v : data) {
                v = dist(rng);
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size()*sizeof(float));
        }
    }
};

struct test_dsv4_hc_comb : public test_dsv4_hc {
    const int64_t n_tokens;
    const int32_t n_iter;
    const float eps;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_HC_COMB";
    }

    std::string vars() override {
        return VARS_TO_STR3(n_tokens, n_iter, eps);
    }

    test_dsv4_hc_comb(int64_t n_tokens = 17, int32_t n_iter = 4, float eps = 1e-6f)
        : n_tokens(n_tokens), n_iter(n_iter), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (2 + hc)*hc, n_tokens);
        ggml_set_name(mixes, "mixes");

        ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
        ggml_set_name(scale, "scale");

        ggml_tensor * base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (2 + hc)*hc);
        ggml_set_name(base, "base");

        out = ggml_dsv4_hc_comb(ctx, mixes, scale, base, eps, n_iter);
        ggml_set_name(out, "out");
        return out;
    }
};

struct test_dsv4_hc_pre : public test_dsv4_hc {
    const int64_t n_embd;
    const int64_t n_tokens;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_HC_PRE";
    }

    std::string vars() override {
        return VARS_TO_STR2(n_embd, n_tokens);
    }

    test_dsv4_hc_pre(int64_t n_embd = 31, int64_t n_tokens = 17)
        : n_embd(n_embd), n_tokens(n_tokens) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_name(x, "x");

        ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n_tokens);
        ggml_set_name(weights, "weights");

        out = ggml_dsv4_hc_pre(ctx, x, weights);
        ggml_set_name(out, "out");
        return out;
    }
};

struct test_dsv4_hc_post : public test_dsv4_hc {
    const int64_t n_embd;
    const int64_t n_tokens;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "DSV4_HC_POST";
    }

    std::string vars() override {
        return VARS_TO_STR2(n_embd, n_tokens);
    }

    test_dsv4_hc_post(int64_t n_embd = 31, int64_t n_tokens = 17)
        : n_embd(n_embd), n_tokens(n_tokens) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(x, "x");

        ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_name(residual, "residual");

        ggml_tensor * post = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n_tokens);
        ggml_set_name(post, "post");

        ggml_tensor * comb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);
        ggml_set_name(comb, "comb");

        out = ggml_dsv4_hc_post(ctx, x, residual, post, comb);
        ggml_set_name(out, "out");
        return out;
    }
};


// GGML_OP_SSM_CONV
struct test_ssm_conv : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, ne_b);
    }

    test_ssm_conv(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 1},
            std::array<int64_t, 4> ne_b = {3, 3, 1, 1})
        : type(type), ne_a(ne_a), ne_b(ne_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a   = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * b   = ggml_new_tensor(ctx, type, 4, ne_b.data());
        ggml_tensor * out = ggml_ssm_conv(ctx, a, b);
        return out;
    }
};

// GGML_OP_SSM_CONV + GGML_OP_ADD (channel-wise bias, optional) + GGML_OP_UNARY(SILU) (fused operation)
struct test_ssm_conv_bias_silu : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;
    const bool fuse_bias;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "SSM_CONV_BIAS_SILU";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, ne_b, fuse_bias);
    }

    test_ssm_conv_bias_silu(ggml_type type, std::array<int64_t, 4> ne_a, std::array<int64_t, 4> ne_b,
            bool fuse_bias)
        : type(type), ne_a(ne_a), ne_b(ne_b), fuse_bias(fuse_bias) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne_b.data());
        ggml_set_name(a, "a");
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_ssm_conv(ctx, a, b);

        if (fuse_bias) {
            ggml_tensor * bias = ggml_new_tensor_1d(ctx, type, out->ne[0]);
            ggml_set_name(bias, "bias");
            out = ggml_add(ctx, out, bias);
        }

        out = ggml_silu(ctx, out);

        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_SSM_SCAN
struct test_ssm_scan : public test_case {
    const ggml_type type;

    const int64_t d_state;
    const int64_t head_dim;
    const int64_t n_head;
    const int64_t n_group;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;
    const bool    xbc_overlap;
    const int64_t K;
    const bool    weak_decay;

    std::string vars() override {
        return VARS_TO_STR10(type, d_state, head_dim, n_head, n_group, n_seq_tokens, n_seqs, xbc_overlap, K, weak_decay);
    }

    test_ssm_scan(ggml_type type = GGML_TYPE_F32,
            int64_t d_state = 32,
            int64_t head_dim = 1, // 1 = Mamba-1; > 1 = Mamba-2 (scalar A per head)
            int64_t n_head  = 32,
            int64_t n_group = 1,
            int64_t n_seq_tokens = 32,
            int64_t n_seqs = 32,
            bool xbc_overlap = false,
            int64_t K = 1,
            bool weak_decay = false)
        : type(type), d_state(d_state), head_dim(head_dim), n_head(n_head), n_group(n_group), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs), xbc_overlap(xbc_overlap), K(K), weak_decay(weak_decay) {}

    double max_nmse_err() override {
        // SSD path (head_dim > 1) uses FP16 intermediates (M matrix, X_dt); Mamba-1 is pure FP32.
        return (head_dim > 1) ? 2e-7 : 1e-7;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * s   = ggml_new_tensor_4d(ctx, type, d_state,  head_dim,     n_head,       n_seqs);
        ggml_tensor * dt  = ggml_new_tensor_3d(ctx, type, n_head,   n_seq_tokens, n_seqs);
        ggml_tensor * A   = ggml_new_tensor_2d(ctx, type, (head_dim > 1) ? 1 : d_state, n_head);
        ggml_tensor * x;
        ggml_tensor * B;
        ggml_tensor * C;

        if (xbc_overlap) {
            ggml_tensor * xbc = ggml_new_tensor_4d(ctx, type, d_state, n_head, n_seq_tokens, 2 * n_seqs);
            x = ggml_view_4d(ctx, xbc, head_dim, n_head, n_seq_tokens, n_seqs,
                             xbc->nb[1], xbc->nb[2], xbc->nb[3], xbc->nb[3]);
            B = ggml_view_4d(ctx, xbc, d_state, n_group, n_seq_tokens, n_seqs,
                             xbc->nb[1], xbc->nb[2], xbc->nb[3], 0);
            C = ggml_view_4d(ctx, xbc, d_state, n_group, n_seq_tokens, n_seqs,
                             xbc->nb[1], xbc->nb[2], xbc->nb[3], 2 * xbc->nb[3]);
        } else {
            x = ggml_new_tensor_4d(ctx, type, head_dim, n_head, n_seq_tokens, n_seqs);
            B = ggml_new_tensor_4d(ctx, type, d_state,  n_group, n_seq_tokens, n_seqs);
            C = ggml_new_tensor_4d(ctx, type, d_state,  n_group, n_seq_tokens, n_seqs);
        }
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,  n_seqs);
        ggml_tensor * out = ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, K);
        return out;
    }


    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // ids: permutation of [0..n_seqs)
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else if (ggml_is_view_op(t->op)) {
                continue;
            } else if (t->ne[1] == n_head && t->ne[2] == 1) {
                // A {1 or d_state, n_head}: negative decay (2-D tensor, ne[2]==1 distinguishes from 3-D/4-D tensors)
                init_tensor_uniform(t, weak_decay ? -0.02f : -1.0f, weak_decay ? -0.005f : -0.5f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

struct test_ssm_scan_rollback : public test_case {
    const ggml_type type;

    const int64_t d_state;
    const int64_t head_dim;
    const int64_t n_head;
    const int64_t n_group;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;
    const int64_t K;

    std::string vars() override {
        return VARS_TO_STR8(type, d_state, head_dim, n_head, n_group, n_seq_tokens, n_seqs, K);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "SSM_SCAN_ROLLBACK";
    }

    bool run_whole_graph() override {
        return true;
    }

    double max_err() override {
        return 1e-6;
    }

    double err(const float * a, const float * b, size_t n) override {
        double result = 0.0;
        for (size_t i = 0; i < n; ++i) {
            result = std::max(result, (double) fabsf(a[i]));
            result = std::max(result, (double) fabsf(b[i]));
        }
        return result;
    }

    test_ssm_scan_rollback(ggml_type type = GGML_TYPE_F32,
            int64_t d_state = 32,
            int64_t head_dim = 64,
            int64_t n_head  = 16,
            int64_t n_group = 2,
            int64_t n_seq_tokens = 8,
            int64_t n_seqs = 2,
            int64_t K = 3)
        : type(type), d_state(d_state), head_dim(head_dim), n_head(n_head), n_group(n_group),
          n_seq_tokens(n_seq_tokens), n_seqs(n_seqs), K(K) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * s   = ggml_new_tensor_4d(ctx, type, d_state,  head_dim,     n_head,       n_seqs);
        ggml_tensor * x   = ggml_new_tensor_4d(ctx, type, head_dim, n_head,       n_seq_tokens, n_seqs);
        ggml_tensor * dt  = ggml_new_tensor_3d(ctx, type, n_head,   n_seq_tokens, n_seqs);
        ggml_tensor * A   = ggml_new_tensor_2d(ctx, type, 1,        n_head);
        ggml_tensor * B   = ggml_new_tensor_4d(ctx, type, d_state,  n_group,      n_seq_tokens, n_seqs);
        ggml_tensor * C   = ggml_new_tensor_4d(ctx, type, d_state,  n_group,      n_seq_tokens, n_seqs);
        ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,  n_seqs);

        ggml_tensor * full = ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids, K);

        const int64_t y_elems     = head_dim * n_head * n_seq_tokens * n_seqs;
        const int64_t state_elems = d_state  * head_dim * n_head      * n_seqs;

        ggml_tensor * out = nullptr;
        for (int64_t slot = 0; slot < K; ++slot) {
            const int64_t prefix_tokens = n_seq_tokens - slot;

            ggml_tensor * x_prefix  = ggml_cont(ctx, ggml_view_4d(ctx, x,  head_dim, n_head,  prefix_tokens, n_seqs, x->nb[1],  x->nb[2],  x->nb[3],  0));
            ggml_tensor * dt_prefix = ggml_cont(ctx, ggml_view_3d(ctx, dt, n_head,   prefix_tokens, n_seqs, dt->nb[1], dt->nb[2], 0));
            ggml_tensor * B_prefix  = ggml_cont(ctx, ggml_view_4d(ctx, B,  d_state,  n_group, prefix_tokens, n_seqs, B->nb[1],  B->nb[2],  B->nb[3],  0));
            ggml_tensor * C_prefix  = ggml_cont(ctx, ggml_view_4d(ctx, C,  d_state,  n_group, prefix_tokens, n_seqs, C->nb[1],  C->nb[2],  C->nb[3],  0));

            ggml_tensor * prefix = ggml_ssm_scan(ctx, s, x_prefix, dt_prefix, A, B_prefix, C_prefix, ids, /*K=*/1);

            ggml_tensor * full_state   = ggml_view_1d(ctx, full,   state_elems, (y_elems + slot*state_elems)*ggml_element_size(full));
            ggml_tensor * prefix_state = ggml_view_1d(ctx, prefix, state_elems, (head_dim*n_head*prefix_tokens*n_seqs)*ggml_element_size(prefix));
            ggml_tensor * diff         = ggml_sum(ctx, ggml_sqr(ctx, ggml_sub(ctx, full_state, prefix_state)));

            out = out == nullptr ? diff : ggml_add(ctx, out, diff);
        }

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else if (ggml_is_view_op(t->op)) {
                continue;
            } else if (t->ne[1] == n_head && t->ne[2] == 1) {
                init_tensor_uniform(t, -1.0f, -0.5f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_RWKV_WKV6
struct test_rwkv_wkv6 : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return VARS_TO_STR5(type, head_count, head_size, n_seq_tokens, n_seqs);
    }

    test_rwkv_wkv6(ggml_type type = GGML_TYPE_F32,
            int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor * r   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * k   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * v   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * tf  = ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size, head_count }.data());
        ggml_tensor * td  = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * s   = ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size * head_size * head_count, n_seqs }.data());
        ggml_tensor * out = ggml_rwkv_wkv6(ctx, k, v, r, tf, td, s);
        return out;
    }
};

// GGML_OP_GATED_DELTA_NET
struct test_gated_delta_net : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;
    const int     v_repeat;
    const bool    permuted;
    const bool    kda;
    const int64_t K; // snapshot slot count: 1 = final-only, >1 = last K states

    std::string vars() override {
        return VARS_TO_STR9(type, head_count, head_size, n_seq_tokens, n_seqs, v_repeat, permuted, kda, K);
    }

    test_gated_delta_net(ggml_type type = GGML_TYPE_F32,
            int64_t head_count = 4, int64_t head_size = 16, int64_t n_seq_tokens = 1, int64_t n_seqs = 1,
            int v_repeat = 1, bool permuted = false, bool kda = false, int64_t K = 1)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs),
          v_repeat(v_repeat), permuted(permuted), kda(kda), K(K) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * q;
        ggml_tensor * k;
        ggml_tensor * v;
        if (permuted) {
            // create with dims 1 and 2 swapped, then permute back to get non-contiguous layout
            q = ggml_permute(ctx, ggml_new_tensor_4d(ctx, type, head_size, n_seq_tokens, head_count, n_seqs), 0, 2, 1, 3);
            k = ggml_permute(ctx, ggml_new_tensor_4d(ctx, type, head_size, n_seq_tokens, head_count, n_seqs), 0, 2, 1, 3);
            v = ggml_permute(ctx, ggml_new_tensor_4d(ctx, type, head_size, n_seq_tokens, head_count * v_repeat, n_seqs), 0, 2, 1, 3);
        } else {
            q = ggml_new_tensor_4d(ctx, type, head_size, head_count, n_seq_tokens, n_seqs);
            k = ggml_new_tensor_4d(ctx, type, head_size, head_count, n_seq_tokens, n_seqs);
            v = ggml_new_tensor_4d(ctx, type, head_size, head_count * v_repeat, n_seq_tokens, n_seqs);
        }
        ggml_set_name(q, "q");
        ggml_set_name(k, "k");
        ggml_set_name(v, "v");
        const int64_t g_ne0 = kda ? head_size : 1;
        ggml_tensor * g     = ggml_new_tensor_4d(ctx, type, g_ne0, head_count * v_repeat, n_seq_tokens, n_seqs);
        ggml_tensor * beta  = ggml_new_tensor_4d(ctx, type, 1, head_count * v_repeat, n_seq_tokens, n_seqs);
        ggml_tensor * state = ggml_new_tensor_4d(ctx, type, head_size, head_size, head_count * v_repeat, n_seqs);
        ggml_set_name(g,     "g");
        ggml_set_name(beta,  "beta");
        ggml_set_name(state, "state");
        // q/k are L2-normalised in qwen35/kimi-linear before delta_net
        q = ggml_l2_norm(ctx, q, 1e-6f);
        k = ggml_l2_norm(ctx, k, 1e-6f);
        ggml_tensor * out   = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_is_view_op(t->op)) { continue; }
            if (strcmp(t->name, "g") == 0) {
                init_tensor_uniform(t, -20.0f, -1e-4f);
            } else if (strcmp(t->name, "beta") == 0) {
                init_tensor_uniform(t, 0.0f, 1.0f);
            } else if (strcmp(t->name, "v") == 0) {
                init_tensor_uniform(t, -0.3f, 5.0f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_GATED_LINEAR_ATTN
struct test_gla : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return VARS_TO_STR5(type, head_count, head_size, n_seq_tokens, n_seqs);
    }

    test_gla(ggml_type type = GGML_TYPE_F32,
            int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor * q   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * k   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * v   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * g   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * s   = ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size * head_size * head_count, n_seqs }.data());
        ggml_tensor * out = ggml_gated_linear_attn(ctx, k, v, q, g, s, pow(head_size, -0.5));
        return out;
    }
};

// GGML_OP_RWKV_WKV7
struct test_rwkv_wkv7 : public test_case {
    const ggml_type type;

    const int64_t head_count;
    const int64_t head_size;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;

    std::string vars() override {
        return VARS_TO_STR5(type, head_count, head_size, n_seq_tokens, n_seqs);
    }

    test_rwkv_wkv7(ggml_type type = GGML_TYPE_F32,
            int64_t head_count = 32, int64_t head_size = 64, int64_t n_seq_tokens = 32, int64_t n_seqs = 32)
        : type(type), head_count(head_count), head_size(head_size), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_tokens = n_seq_tokens * n_seqs;
        ggml_tensor * r   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * w   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * k   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * v   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * a   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        ggml_tensor * b   = ggml_new_tensor(ctx, type, 3, std::vector<int64_t>{ head_size, head_count, n_tokens }.data());
        // Outputs may become NaN with long seqlen without these normalization
        a = ggml_l2_norm(ctx, a, 1e-7F);
        b = ggml_l2_norm(ctx, b, 1e-7F);
        ggml_tensor * s   = ggml_new_tensor(ctx, type, 2, std::vector<int64_t>{ head_size * head_size * head_count, n_seqs }.data());
        ggml_tensor * out = ggml_rwkv_wkv7(ctx, r, w, k, v, a, b, s);
        return out;
    }
};

// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs;  // dims 3 and 4
    const std::array<int64_t, 2> nr;  // repeat in dims 3 and 4
    const std::array<int64_t, 4> per; // permutation of dimensions
    const int64_t k_v; // size of k in memory, resulting in a non-contiguous view for k_v > k, no view for k_v == 0
    const uint32_t o; // number of outputs
    const bool src_overlap; // a and b are overlapping views of the same tensor
    const int64_t m_v; // rows of a in memory, the batches of a are strided for m_v > m, no view for m_v == 0 (upstream #28956)
    const int64_t pad; // bytes after the m_v rows of each batch of a, so nb[2] of a is not a multiple of nb[1]

    std::string vars() override {
        // m_v / pad only named when set, so the names of the existing cases do not change
        std::string s = VARS_TO_STR11(type_a, type_b, m, n, k, bs, nr, per, k_v, o, src_overlap);
        if (m_v != 0 || pad != 0) {
            s += "," + VARS_TO_STR2(m_v, pad);
        }
        return s;
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    double max_nmse_err(ggml_backend_t backend) override {
        // for blackwell we quantize activations to mxfp4 instead of q8_1 so we add higher tolerance
        if ((type_a == GGML_TYPE_MXFP4 || type_a == GGML_TYPE_NVFP4) && backend_has_feature(backend, "BLACKWELL_NATIVE_FP4")) {
            return 2e-2;
        }
        return max_nmse_err();
    }

    int64_t grad_nmax() override {
        return 20000;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * n * k * bs[0] * nr[0] * bs[1] * nr[1];
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2},
            std::array<int64_t, 4> per = {0, 1, 2, 3},
            int64_t k_v = 0, uint32_t o = 1, bool src_overlap = false, int64_t m_v = 0, int64_t pad = 0)
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr), per(per), k_v(k_v), o(o), src_overlap(src_overlap), m_v(m_v), pad(pad) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a;
        ggml_tensor * b;

        const int npermuted = (per[0] != 0) + (per[1] != 1) + (per[2] != 2) + (per[3] != 3);
        if (npermuted > 0) {
            GGML_ASSERT(npermuted == 2);
            GGML_ASSERT(k_v == 0); // not handled
            GGML_ASSERT(m_v == 0); // not handled
            GGML_ASSERT(!ggml_is_quantized(type_a) || per[0] == 0);
            GGML_ASSERT(!ggml_is_quantized(type_b) || per[0] == 0);

            // Create tensors with the permuted dimensions, then permute them back to the dimensions given by m,n,k.
            const int64_t ne_a[4] = {k, m, bs[0],       bs[1]};
            const int64_t ne_b[4] = {k, n, bs[0]*nr[0], bs[1]*nr[1]};

            a = ggml_new_tensor_4d(ctx, type_a, ne_a[per[0]], ne_a[per[1]], ne_a[per[2]], ne_a[per[3]]);
            b = ggml_new_tensor_4d(ctx, type_b, ne_b[per[0]], ne_b[per[1]], ne_b[per[2]], ne_b[per[3]]);
            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(a);
                }
                ggml_set_param(b);
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");

            a = ggml_permute(ctx, a, per[0], per[1], per[2], per[3]);
            b = ggml_permute(ctx, b, per[0], per[1], per[2], per[3]);
            ggml_set_name(a, "a_permuted");
            ggml_set_name(b, "b_permuted");
        } else if (src_overlap) {
            GGML_ASSERT(type_a == type_b);
            GGML_ASSERT(k_v == 0);
            GGML_ASSERT(m_v == 0);

            // a and b are interleaved views of the same tensor: (e.g. fused QKV in MiniMax-01)
            ggml_tensor * base = ggml_new_tensor_4d(ctx, type_a, 2*k, std::max(m, n), bs[0]*nr[0], bs[1]*nr[1]);
            ggml_set_name(base, "base");

            a = ggml_view_4d(ctx, base, k, m, bs[0],       bs[1],       base->nb[1], base->nb[2], base->nb[3], 0);
            b = ggml_view_4d(ctx, base, k, n, bs[0]*nr[0], bs[1]*nr[1], base->nb[1], base->nb[2], base->nb[3], k*ggml_type_size(type_a));
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");
        } else {
            const int64_t k_physical = k_v == 0 ? k : k_v;
            const int64_t m_physical = m_v == 0 ? m : m_v + (pad != 0);
            a = ggml_new_tensor_4d(ctx, type_a, k_physical, m_physical, bs[0], bs[1]);
            b = ggml_new_tensor_4d(ctx, type_b, k_physical, n,          bs[0]*nr[0], bs[1]*nr[1]);

            if (!ggml_is_quantized(type_a)) {
                if (bs[1] == 1 && nr[1] == 1) {
                    ggml_set_param(a);
                }
                ggml_set_param(b);
            }

            if (k_v != 0) {
                GGML_ASSERT(k_v > k);
                a = ggml_view_4d(ctx, a, k, m, bs[0],       bs[1],       a->nb[1], a->nb[2], a->nb[3], 0);
                b = ggml_view_4d(ctx, b, k, n, bs[0]*nr[0], bs[1]*nr[1], b->nb[1], b->nb[2], b->nb[3], 0);
            }
            if (m_v != 0) {
                GGML_ASSERT(m_v > m);
                GGML_ASSERT(pad < (int64_t) a->nb[1]);
                a = ggml_view_4d(ctx, a, k, m, bs[0], bs[1], a->nb[1], m_v*a->nb[1] + pad, a->nb[3], 0);
            }
            ggml_set_name(a, "a");
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        ggml_set_name(out, "out");
        for (uint32_t i = 1; i < o; ++i) {
            ggml_tensor * out2 = ggml_mul_mat(ctx, a, b);
            ggml_set_name(out2, "out2");
            out = ggml_add(ctx, out, out2);
        }

        return out;
    }

    bool run_whole_graph() override { return o > 1; }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return ggml_op_name(GGML_OP_MUL_MAT);
    }
};

// [TAG_DMMV_SHAPES] The Vulkan backend routes a MUL_MAT with ne1 <= 8 and a batch of one to a
// separate mat-vec kernel family (ggml_vk_mul_mat -> ggml_vk_mul_mat_vec_q_f16), and picks its
// pipeline - including the rows-per-workgroup the GGML_VK_DMMV_RM_* knobs set - from
// (type, m, n, k).  Whether it even quantizes B to q8_1 first depends on n and the type
// (ggml_vk_should_use_mmvq), so the shapes have to be the real ones to exercise the real
// pipelines.  Give them their own op name so `test-backend-ops -o MUL_MAT_VEC` is exactly that
// kernel at exactly the production shapes, instead of being buried in the generic MUL_MAT set.
struct test_mul_mat_vec : public test_mul_mat {
    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT_VEC";
    }

    test_mul_mat_vec(ggml_type type_a = GGML_TYPE_F32, int64_t m = 4096, int64_t n = 6, int64_t k = 2560)
        : test_mul_mat(type_a, GGML_TYPE_F32, m, n, k, {1, 1}, {1, 1}) {}
};

// GGML_HINT_SRC0_IS_HADAMARD
struct test_mul_mat_hadamard : public test_mul_mat {
    test_mul_mat_hadamard(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {1, 1},
            std::array<int64_t, 2> nr = {1, 1})
        : test_mul_mat(type_a, type_b, m, n, k, bs, nr) {
            GGML_ASSERT(type_a == GGML_TYPE_F32);
        }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * out = test_mul_mat::build_graph(ctx);
        // Find the mul_mat op in the graph and set the hint
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op == GGML_OP_MUL_MAT) {
                ggml_mul_mat_set_hint(t, GGML_HINT_SRC0_IS_HADAMARD);
            }
        }
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "a") == 0) {
                const int64_t n_cols = t->ne[0];
                const int64_t n_rows = ggml_nrows(t);
                std::vector<float> data(n_cols * n_rows);
                float scale = 1.0f / sqrtf((float)n_cols);
                for (int64_t r = 0; r < n_rows; r++) {
                    float * row_data = data.data() + r * n_cols;
                    for (int64_t i = 0; i < n_cols; i++) {
                        int pop = 0;
                        int64_t val = r & i;
                        while (val) {
                            pop += (val & 1);
                            val >>= 1;
                        }
                        row_data[i] = (pop % 2 == 0) ? scale : -scale;
                    }
                }
                ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
            } else if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16) {
                init_tensor_uniform(t);
            }
        }
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT_HADAMARD";
    }
};

// [TAG_MMID_COMPACT] Router shapes for the ids tensor.  The flattened (expert, N-tile) MoE grid
// dispatches a host-computed bound on sum_e ceil(count_e / BN); how close the real tile total gets
// to that bound depends entirely on how skewed the routing is, so the skewed shapes below are what
// exercise both ends of it.
enum mul_mat_id_router {
    MMID_ROUTER_UNIFORM = 0,  // a shuffled permutation of (i % n_mats) per token: counts ~ equal
    MMID_ROUTER_SKEWED  = 1,  // every token also picks expert 0: one expert holds all nei1 rows
    MMID_ROUTER_TAIL    = 2,  // every token picks expert 0 in slot 0 and expert n_mats - j in slot j > 0: expert 0
                              // and the last n_used - 1 experts hold nei1 rows each, the other n_mats - n_used
                              // are empty. count_e <= nei1 for every expert, which is the most one expert can
                              // legally take (the legacy Vulkan grid and CUDA assume it); the former
                              // MMID_ROUTER_SINGLE (nei0*nei1 rows on expert 0) violated it and was dropped.
};

static void init_mul_mat_id_tensors(ggml_context * ctx, int n_mats, int router = MMID_ROUTER_UNIFORM) {
    std::random_device rd;
    std::default_random_engine rng(rd());
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->type == GGML_TYPE_I32) {
            if (ggml_is_view_op(t->op)) { continue; }
            // ids
            for (int64_t r = 0; r < ggml_nrows(t); r++) {
                std::vector<int32_t> data(t->ne[0]);
                for (int i = 0; i < t->ne[0]; i++) {
                    data[i] = i % n_mats;
                }
                std::shuffle(data.begin(), data.end(), rng);
                if (router == MMID_ROUTER_SKEWED) {
                    // move expert 0 into slot 0 (the used view keeps the leading n_used slots),
                    // swapping so the row still holds distinct experts
                    for (int i = 1; i < t->ne[0]; i++) {
                        if (data[i] == 0) {
                            data[i] = data[0];
                            break;
                        }
                    }
                    data[0] = 0;
                } else if (router == MMID_ROUTER_TAIL) {
                    // slot 0 -> expert 0, slot j -> expert n_mats - j: distinct within the row for every
                    // j < n_mats, so the used view (the leading n_used slots) is always a legal routing
                    data[0] = 0;
                    for (int i = 1; i < t->ne[0]; i++) {
                        data[i] = n_mats - i;
                    }
                }
                ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
            }
        } else {
            init_tensor_uniform(t);
        }
    }
}

// GGML_OP_MUL_MAT_ID
struct test_mul_mat_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int n_mats;
    const int n_used;
    const bool b; // broadcast b matrix
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const int router;
    const int64_t m_v; // rows of as in memory, the experts of as are strided for m_v > m, no view for m_v == 0 (upstream #28956)
    const bool ids_offset; // ids is a view one element into its rows (upstream #29998)

    std::string vars() override {
        std::string s = VARS_TO_STR9(type_a, type_b, n_mats, n_used, b, m, n, k, router);
        if (m_v != 0) {
            s += "," + VAR_TO_STR(m_v);
        }
        if (ids_offset) {
            s += "," + VAR_TO_STR(ids_offset);
        }
        return s;
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    double max_nmse_err(ggml_backend_t backend) override {
        // for blackwell we quantize activations to mxfp4 instead of q8_1 so we add higher tolerance
        if ((type_a == GGML_TYPE_MXFP4 || type_a == GGML_TYPE_NVFP4) && backend_has_feature(backend, "BLACKWELL_NATIVE_FP4")) {
            return 2e-2;
        }
        return max_nmse_err();
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * k * n * n_used;
    }

    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int n_mats = 8, int n_used = 2, bool b = false,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            int router = MMID_ROUTER_UNIFORM, int64_t m_v = 0, bool ids_offset = false)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), n_used(n_used), b(b),
            m(m), n(n), k(k), router(router), m_v(m_v), ids_offset(ids_offset) {
            GGML_ASSERT(n_used <= n_mats);
            GGML_ASSERT(m_v == 0 || m_v > m);
        }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m_v == 0 ? m : m_v, n_mats);
        if (m_v != 0) {
            as = ggml_view_3d(ctx, as, k, m, n_mats, as->nb[1], as->nb[2], 0);
        }
        ggml_set_name(as, "as");

        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats + int(ids_offset), n);
        ggml_set_name(ids, "ids");
        if (n_used != n_mats || ids_offset) {
            ids = ggml_view_2d(ctx, ids, n_used, n, ids->nb[1], ids_offset ? sizeof(int32_t) : 0);
            ggml_set_name(ids, "view_of_ids");
        }

        ggml_tensor * b = ggml_new_tensor_3d(ctx, type_b, k, this->b ? 1 : n_used, n);
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        init_mul_mat_id_tensors(ctx, n_mats, router);
    }
};

// MUL_MAT_ID with expert ids repeated within a token's row (upstream #29998): an expert can then hold
// more than nei1 rows, which the per-expert tile list must cover in full
struct test_mul_mat_id_dup : public test_mul_mat_id {
    using test_mul_mat_id::test_mul_mat_id;
    std::string vars() override { return test_mul_mat_id::vars() + ",dup=1"; }
    void initialize_tensors(ggml_context * ctx) override {
        std::default_random_engine rng(1234);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = (rng() % 4 == 0) ? (int32_t) (rng() % n_mats) : 0;   // repeated ids, mostly expert 0
                    }
                    if (r == ggml_nrows(t) - 1) {
                        data[ids_offset ? 1 : 0] = n_mats - 1;
                    }
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_MUL_MAT_ID + GGML_OP_ADD or GGML_OP_MUL
struct test_mul_mat_id_fusion : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int n_mats;
    const int n_used;
    const bool b; // broadcast b matrix
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const uint32_t o; // number of outputs
    const bool mul;

    std::string vars() override {
        return VARS_TO_STR10(type_a, type_b, n_mats, n_used, b, m, n, k, o, mul);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * m * k * n * n_used;
    }

    test_mul_mat_id_fusion(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int n_mats = 8, int n_used = 2, bool b = false,
            int64_t m = 32, int64_t n = 32, int64_t k = 32, uint32_t o = 1, bool mul = false)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), n_used(n_used), b(b),
            m(m), n(n), k(k), o(o), mul(mul) {
            GGML_ASSERT(n_used <= n_mats);
        }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_set_name(as, "as");

        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n);
        ggml_set_name(ids, "ids");
        if (n_used != n_mats) {
            ids = ggml_view_2d(ctx, ids, n_used, n, ids->nb[1], 0);
            ggml_set_name(ids, "view_of_ids");
        }

        ggml_tensor * b = ggml_new_tensor_3d(ctx, type_b, k, this->b ? 1 : n_used, n);
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
        ggml_set_name(out, "out");

        for (uint32_t i = 1; i < o; ++i) {
            ggml_tensor * a2 = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
            ggml_tensor * out2 = ggml_mul_mat_id(ctx, a2, b, ids);
            ggml_set_name(out2, "out2");
            out = ggml_add(ctx, out, out2);
        }

        if (mul) {
            std::array<int64_t, 4> ne { 1, out->ne[1], out->ne[2], out->ne[3] };
            ne[0] = 1;
            ggml_tensor * m = ggml_new_tensor(ctx, out->type, 4, ne.data());
            out = ggml_mul(ctx, out, m);
        }

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        init_mul_mat_id_tensors(ctx, n_mats);
    }

    bool run_whole_graph() override { return true; }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT_ID_FUSION";
    }
};

// GGML_OP_OUT_PROD
struct test_out_prod : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs; // dims 3 and 4
    const std::array<int64_t, 2> nr; // repeat in dims 3 and 4
    const bool trans_b;

    std::string vars() override {
        return VARS_TO_STR8(type_a, type_b, m, n, k, bs, nr, trans_b);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    test_out_prod(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2},
            bool trans_b = false)
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr), trans_b(trans_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type_a, m, k, bs[0], bs[1]);
        ggml_set_name(a, "a");

        ggml_tensor * b;
        if (trans_b) {
            b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0]*nr[0], bs[1]*nr[1]);
            b = ggml_transpose(ctx, b);
        } else {
            b = ggml_new_tensor_4d(ctx, type_b, n, k, bs[0]*nr[0], bs[1]*nr[1]);
        }
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_out_prod(ctx, a, b);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SQR
struct test_sqr : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqr(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sqr(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * 0.25f*ne[0]*ne[1]*ne[2]*ne[3]; // 10% of expected value of sum.
    }
};

// GGML_OP_SQRT
struct test_sqrt : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqrt(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 3, 3, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sqrt(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // fill with positive values
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, 50.0f, 100.0f);
        }
    }

    float grad_eps() override {
        return 20.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_LOG
struct test_log : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_log(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_log(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // log(1) == 0, cluster values there to keep the sum low for better precision in the backward pass:
            init_tensor_uniform(t, 0.9f, 1.1f);
        }
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_SIN
struct test_sin : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sin(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_sin(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -6.5f, 6.5f); // Covers interval [-2*pi, 2*pi].
        }
    }

    double max_maa_err() override {
        return 1e-3;
    }

    float grad_eps() override {
        return 0.2f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_COS
struct test_cos : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cos(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_cos(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -6.5f, 6.5f); // Covers interval [-2*pi, 2*pi].
        }
    }

    double max_maa_err() override {
        return 1e-3;
    }

    float grad_eps() override {
        return 0.2f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_CLAMP
struct test_clamp : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float min;
    float max;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, min, max);
    }

    test_clamp(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            float min = -0.5f, float max = 0.5f)
        : type(type), ne(ne), min(min), max(max) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_clamp(ctx, a, min, max);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 1e-2f;
    }

    std::vector<float> grad_expect() override {
        return {0.0f, 1.0f};
    }
};

// GGML_OP_FLOOR
struct test_floor : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_floor(ggml_type type = GGML_TYPE_F32,
               std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_floor(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.0f, 10.0f);
        }
    }
};

// GGML_OP_CEIL
struct test_ceil : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_ceil(ggml_type type = GGML_TYPE_F32,
              std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_ceil(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.0f, 10.0f);
        }
    }
};

// GGML_OP_ROUND
struct test_round : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_round(ggml_type type = GGML_TYPE_F32,
               std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_round(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.0f, 10.0f);
        }
    }
};

// GGML_OP_TRUNC
struct test_trunc : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_trunc(ggml_type type = GGML_TYPE_F32,
               std::array<int64_t, 4> ne = {10, 2, 2, 2})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_trunc(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -10.0f, 10.0f);
        }
    }
};

// GGML_OP_DIAG_MASK_INF
struct test_diag_mask_inf : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int n_past;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, n_past);
    }

    test_diag_mask_inf(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 3, 2},
            int n_past = 5)
        : type(type), ne(ne), n_past(n_past) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_diag_mask_inf(ctx, a, n_past);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SOFT_MAX
struct test_soft_max : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool mask;
    const bool sinks;
    const ggml_type m_prec;
    const std::array<int64_t, 2> nr23; // broadcast only dims 2 and 3
    const float scale;
    const float max_bias;
    const bool inplace;

    std::string vars() override {
        return VARS_TO_STR9(type, ne, mask, sinks, m_prec, nr23, scale, max_bias, inplace);
    }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override {
        return 1e-6;
    }

    test_soft_max(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            bool mask = false,
            bool sinks = false,
            ggml_type m_prec = GGML_TYPE_F32,
            std::array<int64_t, 2> nr23 = {1, 1},
            float scale = 1.0f,
            float max_bias = 0.0f,
            bool inplace = false)
        : type(type), ne(ne), mask(mask), sinks(sinks), m_prec(m_prec), nr23(nr23), scale(scale), max_bias(max_bias), inplace(inplace) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2]*nr23[0], ne[3]*nr23[1]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * mask = nullptr;
        if (this->mask) {
            mask = ggml_new_tensor_4d(ctx, m_prec, ne[0], ne[1], ne[2], ne[3]);
            ggml_set_name(mask, "mask");
        }

        ggml_tensor * sinks = nullptr;
        if (this->sinks) {
            sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[2]*nr23[0]);
            ggml_set_name(sinks, "sinks");
        }

        ggml_tensor * out;
        if (inplace) {
            out = ggml_soft_max_ext_inplace(ctx, a, mask, scale, max_bias);
        } else {
            out = ggml_soft_max_ext(ctx, a, mask, scale, max_bias);
        }
        ggml_soft_max_add_sinks(out, sinks);
        ggml_set_name(out, "out");

        return out;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_SOFT_MAX_BACK
struct test_soft_max_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float scale;
    const float max_bias;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, scale, max_bias);
    }

    test_soft_max_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            float scale = 1.0f,
            float max_bias = 0.0f)
        : type(type), ne(ne), scale(scale), max_bias(max_bias) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_soft_max_ext_back(ctx, a, b, scale, max_bias);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ROPE + GGML_OP_ROPE_BACK
struct test_rope : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int n_dims;
    int mode;
    int n_ctx; // used to generate positions
    float fs; // freq_scale
    float ef; // ext_factor
    float af; // attn_factor
    bool ff;
    int v; // view (1 : non-contiguous a)
    bool forward;
    bool inplace;
    int n_offs; // offset of the rotated dims window, set via ggml_rope_set_offset()

    std::string vars() override {
        // forward can be inferred from the op, does not need to be printed
        return VARS_TO_STR12(type, ne_a, n_dims, mode, n_ctx, fs, ef, af, ff, v, inplace, n_offs);
    }

    test_rope(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 5, 3, 1},
            int n_dims = 10, int mode = GGML_ROPE_TYPE_NORMAL, int n_ctx = 512, float fs = 1.0f,
            float ef = 0.0f, float af = 0.0f, bool ff = false, int v = 0, bool forward = true, bool inplace = false,
            int n_offs = 0)
        : type(type), ne_a(ne_a), n_dims(n_dims), mode(mode), n_ctx(n_ctx), fs(fs), ef(ef), af(af), ff(ff), v(v), forward(forward), inplace(inplace), n_offs(n_offs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            if (forward && n_offs == 0) {
                // FIXME: support gradients with n_offs > 0
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else if (v == 2) {
            // second-half slice along dim 0 (mimics build_rope_2d in clip.cpp).
            // The non-zero view offset (ne_a[0] * elem_size) often produces a
            // non-aligned buffer offset, which exercises backends' alignment paths.
            auto ne = ne_a; ne[0] *= 2;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            if (forward && n_offs == 0) {
                // FIXME: support gradients with n_offs > 0
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3],
                             a->nb[1], a->nb[2], a->nb[3],
                             ne_a[0] * ggml_element_size(a));
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            if (forward && n_offs == 0) {
                // FIXME: support gradients with n_offs > 0
                ggml_set_param(a);
            }
            ggml_set_name(a, "a");
        }

        const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
        const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

        ggml_tensor * pos;
        if (is_mrope || is_vision) {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2] * 4);
        } else {
            pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2]);
        }
        ggml_set_name(pos, "pos");

        ggml_tensor * freq = nullptr;
        if (ff) {
            freq = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims/2);
            ggml_set_name(freq, "freq");
        }

        ggml_tensor * out;
        if (is_mrope) {
            if (is_vision) {
                GGML_ASSERT(n_dims/4 > 0);
                int rope_sections[4] = {n_dims/4, n_dims/4, 0, 0}; // Vision-RoPE only use first two dimension for image (x, y) coordinate
                if (forward) {
                    if (inplace) {
                        out = ggml_rope_multi_inplace(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    } else {
                        out = ggml_rope_multi(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    }
                } else {
                    out = ggml_rope_multi_back(ctx, a, pos, freq, n_dims/2, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            } else {
                GGML_ASSERT(n_dims/3 > 0);
                int rope_sections[4] = {n_dims/3, n_dims/3, n_dims/3, 0};
                if (forward) {
                    if (inplace) {
                        out = ggml_rope_multi_inplace(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    } else {
                        out = ggml_rope_multi(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                    }
                } else {
                    out = ggml_rope_multi_back(ctx, a, pos, freq, n_dims, rope_sections, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            }
        } else {
            if (forward) {
                if (inplace) {
                    out = ggml_rope_ext_inplace(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                } else {
                    out = ggml_rope_ext(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
                }
            } else {
                out = ggml_rope_ext_back(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
            }
        }
        if (n_offs != 0) {
            out = ggml_rope_set_offset(out, n_offs);
        }
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                const int num_pos_ids = (mode & GGML_ROPE_TYPE_MROPE) ? ne_a[2] * 4 : ne_a[2];
                std::vector<int> data(num_pos_ids);
                for (int i = 0; i < num_pos_ids; i++) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, num_pos_ids * sizeof(int));
            } else {
                if (t->ne[0] == n_dims/2) {
                    // frequency factors in the range [0.9f, 1.1f]
                    init_tensor_uniform(t, 0.9f, 1.1f);
                } else {
                    init_tensor_uniform(t);
                }
            }
        }
    }

    double max_maa_err() override {
        return 1e-3;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_POOL2D
struct test_pool2d : public test_case {
    enum ggml_op_pool pool_type;
    const ggml_type type_input;
    const std::array<int64_t, 4> ne_input;
    // kernel size
    const int k0;
    const int k1;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;

    std::string vars() override {
        return VARS_TO_STR9(pool_type, type_input, ne_input, k0, k1, s0, s1, p0, p1);
    }

    test_pool2d(ggml_op_pool pool_type = GGML_OP_POOL_AVG,
            ggml_type type_input = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
            int k0 = 3, int k1 = 3,
            int s0 = 1, int s1 = 1,
            int p0 = 1, int p1 = 1)
        : pool_type(pool_type), type_input(type_input), ne_input(ne_input), k0(k0), k1(k1), s0(s0), s1(s1), p0(p0), p1(p1) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(input);
        ggml_set_name(input, "input");

        ggml_tensor * out = ggml_pool_2d(ctx, input, pool_type, k0, k1, s0, s1, p0, p1);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_POOL1D
struct test_pool1d : public test_case {
    enum ggml_op_pool pool_type;
    const ggml_type type_input;
    const std::array<int64_t, 4> ne_input;
    const int k0;
    const int s0;
    const int p0;

    std::string vars() override {
        return VARS_TO_STR6(pool_type, type_input, ne_input, k0, s0, p0);
    }

    test_pool1d(ggml_op_pool pool_type = GGML_OP_POOL_AVG,
                ggml_type type_input = GGML_TYPE_F32,
                std::array<int64_t,4> ne_input = {10, 1, 1, 1},
                int k0 = 3, int s0 = 3, int p0 = 0)
        : pool_type(pool_type), type_input(type_input), ne_input(ne_input), k0(k0), s0(s0), p0(p0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(input);
        ggml_set_name(input, "input");

        ggml_tensor * out = ggml_pool_1d(ctx, input, pool_type, k0, s0, p0);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_CONV_TRANSPOSE_1D
struct test_conv_transpose_1d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;

    const int s0; // stride
    const int p0; // padding
    const int d0; // dilation

    std::string vars() override {
        return VARS_TO_STR5(ne_input, ne_kernel, s0, p0, d0);
    }

    test_conv_transpose_1d(std::array<int64_t, 4> ne_input = {197, 32, 1, 1}, // [input_width, input_channels, 1 /* assert in cpu kernel*/, 1 (should be batch)]
                           std::array<int64_t, 4> ne_kernel = {16, 32, 32, 1}, // [kernel_width, output_channels, input_channels, 1 (should be batch)]
                           int s0 = 1, int p0 = 0, int d0 = 1)
        : ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), p0(p0), d0(d0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, s0, p0, d0);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_COL2IM_1D
struct test_col2im_1d : public test_case {
    const ggml_type type;
    const int64_t K;    // kernel size
    const int64_t OC;   // output channels
    const int64_t T_in; // input length (number of columns)
    const int s0;       // stride
    const int p0;       // padding cropped from both sides

    std::string vars() override {
        return VARS_TO_STR6(type, K, OC, T_in, s0, p0);
    }

    double max_nmse_err() override {
        return type == GGML_TYPE_F32 ? 1e-7 : 5e-4;
    }

    test_col2im_1d(ggml_type type = GGML_TYPE_F32,
            int64_t K = 4, int64_t OC = 3, int64_t T_in = 7,
            int s0 = 2, int p0 = 0)
        : type(type), K(K), OC(OC), T_in(T_in), s0(s0), p0(p0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * cols = ggml_new_tensor_2d(ctx, type, K*OC, T_in);
        ggml_set_name(cols, "cols");

        ggml_tensor * out = ggml_col2im_1d(ctx, cols, s0, (int) OC, p0);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_CONV_TRANSPOSE_2D
struct test_conv_transpose_2d : public test_case {
    // Dimensions
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    const int stride;
    // Types
    const ggml_type kernel_type;

    std::string vars() override {
        return VARS_TO_STR4(kernel_type, ne_input, ne_kernel, stride);
    }

    double max_nmse_err() override {
        return 5e-4; // The default 1e-7 is too small for Vulkan.
    }

    test_conv_transpose_2d(
        std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
        std::array<int64_t, 4> ne_kernel = {3, 3, 3, 1}, // [kernel_width, kernel_height, input_channels, 1]
        int stride = 1,
        ggml_type kernel_type = GGML_TYPE_F16
    ) : ne_input(ne_input), ne_kernel(ne_kernel), stride(stride), kernel_type(kernel_type) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, kernel_type, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_conv_transpose_2d_p0(ctx, kernel, input, stride);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_IM2COL
struct test_im2col : public test_case {
    const ggml_type type_input;
    const ggml_type type_kernel;
    const ggml_type dst_type;
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;
    // dilation
    const int d0;
    const int d1;
    // mode
    const bool is_2D;

    std::string vars() override {
        return VARS_TO_STR12(type_input, type_kernel, dst_type, ne_input, ne_kernel, s0, s1, p0, p1, d0, d1, is_2D);
    }

    test_im2col(ggml_type type_input = GGML_TYPE_F32, ggml_type type_kernel = GGML_TYPE_F16, ggml_type dst_type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
            std::array<int64_t, 4> ne_kernel = {3, 3, 3, 1}, // [kernel_width, kernel_height, input_channels, 1]
            int s0 = 1, int s1 = 1,
            int p0 = 1, int p1 = 1,
            int d0 = 1, int d1 = 1,
            bool is_2D = true)
        : type_input(type_input), type_kernel(type_kernel), dst_type(dst_type), ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), is_2D(is_2D) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(input);
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_im2col(ctx, kernel, input, s0, s1, p0, p1, d0, d1, is_2D, dst_type);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_IM2COL_3D
struct test_im2col_3d : public test_case {
    const ggml_type type_input;
    const ggml_type type_kernel;
    const ggml_type dst_type;
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    // stride
    const int s0;
    const int s1;
    const int s2;
    // padding
    const int p0;
    const int p1;
    const int p2;
    // dilation
    const int d0;
    const int d1;
    const int d2;

    const int64_t IC;
    const bool v;

    std::string vars() override {
        return VARS_TO_STR16(type_input, type_kernel, dst_type, ne_input, ne_kernel, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, v);
    }

    test_im2col_3d(ggml_type type_input = GGML_TYPE_F32, ggml_type type_kernel = GGML_TYPE_F16, ggml_type dst_type = GGML_TYPE_F32,
                std::array<int64_t, 4> ne_input = {10, 10, 10, 9}, // [OC*IC, KD, KH, KW]
                std::array<int64_t, 4> ne_kernel = {3, 3, 3, 1}, // [N*IC, ID, IH, IW]
                int64_t IC = 3,
                int s0 = 1, int s1 = 1, int s2 = 1,
                int p0 = 1, int p1 = 1, int p2 = 1,
                int d0 = 1, int d1 = 1, int d2 = 1,
                bool v = false)
        : type_input(type_input), type_kernel(type_kernel), dst_type(dst_type), ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), s1(s1), s2(s2), p0(p0), p1(p1), p2(p2), d0(d0), d1(d1), d2(d2), IC(IC), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_set_param(input);
        ggml_set_name(input, "input");

        if (v) {
            input = ggml_view_4d(ctx, input, ne_input[0] - 2, ne_input[1] - 2, ne_input[2] - 2, ne_input[3] - 2, input->nb[1], input->nb[2], input->nb[3], 0);
            ggml_set_name(input, "view_of_input");
        }

        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_im2col_3d(ctx, kernel, input, IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, dst_type);
        ggml_set_name(out, "out");

        return out;
    }
};

// CONV_2D
struct test_conv_2d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    const ggml_type              type_kernel;
    const int                    stride0;
    const int                    stride1;
    const int                    padding0;
    const int                    padding1;
    const int                    dilation0;
    const int                    dilation1;
    // Whether the inputs are contiguous in the channel dim or the width dim
    const bool                   cwhn;

    // If true, the direct CONV_2D will be used in the graph, otherwise it
    // uses ggml_conv_2d:
    // * if the program is called with -o CONV_2D_DIRECT_IMPL, the
    // CONV_2D graph will be built, while
    // * if the program is called with -o CONV_2D_INDIRECT_IMPL, the
    // IM2COL -> MUL_MM graph will be built.

    std::string vars() override {
        return VARS_TO_STR10(ne_input, ne_kernel, type_kernel, stride0, stride1, padding0, padding1, dilation0, dilation1, cwhn);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        // Just counting matmul costs:
        // KxCRS @ CRSxNPQ = KxNPQ --> KxNPQx(CRS+CRS-1) flops

        // Copied from ggml.c: int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d)
        auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
            return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
        };

        int64_t W    = ne_input[0];
        int64_t H    = ne_input[1];
        int64_t KW   = ne_kernel[0];
        int64_t KH   = ne_kernel[1];
        int64_t Cin  = ne_kernel[2];
        int64_t Cout = ne_kernel[3];
        int64_t N    = ne_input[3];
        int64_t OH   = calc_conv_output_size(H, KH, stride0, padding0, dilation0);
        int64_t OW   = calc_conv_output_size(W, KW, stride0, padding0, dilation0);

        int64_t K   = Cout;
        int64_t CRS = Cin * KH * KW;
        int64_t NPQ = N * OH * OW;

        return K * NPQ * (2 * CRS - 1);
    }

    test_conv_2d(std::array<int64_t, 4> ne_input  = { 64, 64, 16, 1 },
                 std::array<int64_t, 4> ne_kernel = { 3, 3, 1, 16 }, ggml_type type_kernel = GGML_TYPE_F32, int stride0 = 1,
                 int stride1 = 1, int padding0 = 0, int padding1 = 0, int dilation0 = 1, int dilation1 = 1, bool cwhn = false) :
        ne_input(ne_input),
        ne_kernel(ne_kernel),
        type_kernel(type_kernel),
        stride0(stride0),
        stride1(stride1),
        padding0(padding0),
        padding1(padding1),
        dilation0(dilation0),
        dilation1(dilation1),
        cwhn(cwhn) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        if (cwhn) {
            // change memory layout to channel-most-contiguous (CWHN),
            // then permute it back so NE matches the original input
            input  = ggml_cont(ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
            input  = ggml_permute(ctx, input, 2, 0, 1, 3);
            kernel = ggml_cont(ctx, ggml_permute(ctx, kernel, 2, 3, 1, 0));
            kernel = ggml_permute(ctx, kernel, 3, 2, 0, 1);
        }

        ggml_tensor * out =
            ggml_conv_2d_direct(ctx, kernel, input, stride0, stride1, padding0, padding1, dilation0, dilation1);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_CONV_2D_DW
struct test_conv_2d_dw : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    const ggml_type type_kernel;
    const int stride;
    const int padding;
    const int dilation;
    const bool cwhn;

    std::string vars() override {
        return VARS_TO_STR7(ne_input, ne_kernel, type_kernel, stride, padding, dilation, cwhn);
    }

    test_conv_2d_dw(
            std::array<int64_t, 4> ne_input = {64, 64, 16, 1},
            std::array<int64_t, 4> ne_kernel = {3, 3, 1, 16},
            ggml_type type_kernel = GGML_TYPE_F32,
            int stride = 1, int padding = 0, int dilation = 1, bool cwhn = false)
        : ne_input(ne_input), ne_kernel(ne_kernel), type_kernel(type_kernel), stride(stride), padding(padding), dilation(dilation), cwhn(cwhn) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_set_name(input, "input");

        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_set_name(kernel, "kernel");

        if (cwhn) {
            // change memory layout to channel-most-contiguous (CWHN),
            // then permute it back so NE matches the original input
            input = ggml_cont(ctx, ggml_permute(ctx, input, 1, 2, 0, 3));
            input = ggml_permute(ctx, input, 2, 0, 1, 3);
            kernel = ggml_cont(ctx, ggml_permute(ctx, kernel, 2, 3, 1, 0));
            kernel = ggml_permute(ctx, kernel, 3, 2, 0, 1);
        }

        ggml_tensor * out = ggml_conv_2d_dw_direct(
            ctx, kernel, input,
            stride, stride, padding, padding, dilation, dilation);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_CONV_3D
struct test_conv_3d : public test_case {
    // Logical 5D dimensions
    const int64_t N, IC, ID, IH, IW;
    const int64_t OC, KD, KH, KW;
    // Conv params
    const int s0, s1, s2;
    const int p0, p1, p2;
    const int d0, d1, d2;
    // Types
    const ggml_type type_kernel;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "CONV_3D";
    }

    std::string vars() override {
        return VARS_TO_STR11(N, IC, ID, IH, IW, OC, KD, KH, KW, s0, s1) + "," +
               VARS_TO_STR8(s2, p0, p1, p2, d0, d1, d2, type_kernel);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
            return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
        };
        const int64_t OD = calc_conv_output_size(ID, KD, s2, p2, d2);
        const int64_t OH = calc_conv_output_size(IH, KH, s1, p1, d1);
        const int64_t OW = calc_conv_output_size(IW, KW, s0, p0, d0);

        return (uint64_t)N * OC * OD * OH * OW * (2 * IC * KD * KH * KW - 1);
    }

    test_conv_3d(
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW,
        int64_t OC, int64_t KD, int64_t KH, int64_t KW,
        int s0, int s1, int s2,
        int p0, int p1, int p2,
        int d0, int d1, int d2,
        ggml_type type_kernel
    ) : N(N), IC(IC), ID(ID), IH(IH), IW(IW),
        OC(OC), KD(KD), KH(KH), KW(KW),
        s0(s0), s1(s1), s2(s2),
        p0(p0), p1(p1), p2(p2),
        d0(d0), d1(d1), d2(d2),
        type_kernel(type_kernel) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // GGML input tensor is packed as [W, H, D, C*N]
        const int64_t ne_input[] = {IW, IH, ID, IC * N};
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input);
        ggml_set_name(input, "input");

        // GGML kernel tensor is packed as [KW, KH, KD, IC*OC]
        const int64_t ne_kernel[] = {KW, KH, KD, IC * OC};
        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel);
        ggml_set_name(kernel, "kernel");

        ggml_tensor * out = ggml_conv_3d_direct(ctx, kernel, input, s0, s1, s2, p0, p1, p2, d0, d1, d2, (int)IC, (int)N, (int)OC);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_CONCAT
struct test_concat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int64_t ne_b_d;
    const int dim;
    const int v; // view (1 << 0: non-cont a (first 3 dim), 1 << 1: non-cont b (first 3 dim), 1 << 2: non-cont a (last 2 dim), 1 << 3: non-cont b (last 2 dim), 1 << 4: transposed b (dim 1 innermost))

    std::string vars() override {
        return VARS_TO_STR5(type, ne_a, ne_b_d, dim, v);
    }

    test_concat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 5, 5, 5},
            int64_t ne_b_d = 5,
            int dim = 2, int v = 0)
        : type(type), ne_a(ne_a), ne_b_d(ne_b_d), dim(dim), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        auto ne_b = ne_a;
        ne_b[dim] = ne_b_d;
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else if (v & 4) {
            auto ne = ne_a; ne[2] *= 2; ne[3] *= 4;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(a, "a");

            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view_of_a");
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
            ggml_set_name(a, "a");
        }
        ggml_tensor * b;
        if (v & 2) {
            auto ne = ne_b; ne[0] *= 3; ne[1] *= 2; ne[2] *= 4;
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(b, "view_of_b");
        } else if (v & 8) {
            auto ne = ne_b; ne[2] *= 3; ne[3] *= 2;
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(b, "view_of_b");
        } else if (v & 16) {
            // Transposed b: nb[0] is strided and nb[1] == type_size. This is the
            // concat(state, ggml_transpose(x), 0) shape class that every GDN / Mamba-style
            // conv-state build produces, and the one the Vulkan backend splits into a generic
            // copy of a plus a tiled 32x32 transpose-copy of b.
            auto ne = ne_b; std::swap(ne[0], ne[1]);
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(b, "b");

            b = ggml_transpose(ctx, b);
            ggml_set_name(b, "transpose_of_b");
        } else {
            b = ggml_new_tensor(ctx, type, 4, ne_b.data());
            ggml_set_name(b, "b");
        }

        ggml_tensor * out = ggml_concat(ctx, a, b, dim);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ARGSORT
struct test_argsort : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    ggml_sort_order order;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, order);
    }

    test_argsort(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {16, 10, 10, 10},
            ggml_sort_order order = GGML_SORT_ORDER_ASC)
        : type(type), ne(ne), order(order) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_argsort(ctx, a, order);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // indices
                std::vector<int> data(ggml_nelements(t));
                for (int i = 0; i < ggml_nelements(t); i++) {
                    data[i] = rand();
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), 0, ne[0]*ne[1]*ne[2]*ne[3] * sizeof(int));
            } else if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                GGML_ABORT("fatal error");
            }
        }
    }
};

// GGML_OP_TOP_K
struct test_top_k : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int k;
    // 0: distinct values, compare the indices exactly.
    // 1: random duplicates, compare the selected VALUES as a multiset - any resolution of a
    //    tie that straddles the k-th value passes.
    // 2: ties by construction, compare the INDICES exactly. The reference is the top k under
    //    (value descending, index ascending) - cmp_top_k in ggml-cpu/ops.cpp - so a backend
    //    passes only if it resolves ties the same way, which is the property that makes a
    //    selection reproducible run to run. Vulkan's radix-select (k >= 1024) ranks ties by
    //    index; keep these cases at large k, the bitonic small-k fallback does not.
    const int ties;
    ggml_tensor * input {};

    std::string vars() override {
        return VARS_TO_STR4(type, ne, k, ties);
    }

    test_top_k(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {16, 10, 10, 10},
            int k = 4, int ties = 0)
        : type(type), ne(ne), k(k), ties(ties) {}

    double max_err() override {
        return 0.0;
    }

    // When there are ties, only validate the final result.
    // The logic in err can't handle the sentinel tensors.
    bool run_whole_graph() override { return ties != 0; }

    double err(const float * a, const float * b, size_t n) override {
        // When there are no ties, we expect the exact same set of indices,
        // but possibly in a different order. When there are ties, the indices
        // can be different but the input values they correspond to should be
        // the same. The logic for ties could work for non-ties, but only for
        // the output tensor, not for the sentinel tensors.
        if (ties == 1) {
            std::vector<float> src(ggml_nelements(input));

            ggml_backend_tensor_get(input, src.data(), 0, ggml_nelements(input) * ggml_type_size(type));

            double diff = 0.0f;

            GGML_ASSERT(n == (size_t)(ggml_nrows(input) * k));
            int64_t cols = input->ne[0];
            std::vector<int32_t> ia(k);
            std::vector<int32_t> ib(k);
            std::vector<float> asrc(k);
            std::vector<float> bsrc(k);
            for (int64_t r = 0; r < ggml_nrows(input); r++) {
                // Convert indices for the row back to integer
                for (int64_t c = 0; c < k; c++) {
                    ia[c] = (int32_t)a[r * k + c];
                    ib[c] = (int32_t)b[r * k + c];
                }
                // The src values for each row should match.
                for (int64_t c = 0; c < k; c++) {
                    asrc[c] = src[r * cols + ia[c]];
                    bsrc[c] = src[r * cols + ib[c]];
                }
                diff += jdst(asrc.data(), bsrc.data(), k);
                // There should be no duplicate indices
                std::sort(ia.begin(), ia.end());
                std::sort(ib.begin(), ib.end());
                if (std::adjacent_find(ia.begin(), ia.end()) != ia.end()) {
                    diff += 1;
                }
                if (std::adjacent_find(ib.begin(), ib.end()) != ib.end()) {
                    diff += 1;
                }
            }
            return diff;
        } else {
            // ties == 0 does not set run_whole_graph(), so err() is also called on the harness
            // sentinels; only the output tensor has the (nrows x k) id layout, everything else
            // keeps the flat comparison (a sentinel is memcmp'd for equality before this anyway).
            if (n == (size_t) (ggml_nrows(input) * k)) {
                return cmp_ids_per_row(a, b, n, k, input->ne[0]);
            }

            std::vector<int32_t> ia(n);
            std::vector<int32_t> ib(n);

            double diff = 0.0f;

            for (size_t i = 0; i < n; i++) {
                ia[i] = (int32_t) a[i];
                ib[i] = (int32_t) b[i];

                // penalize the result if the data is not integer valued
                diff += std::fabs(a[i] - ia[i]);
                diff += std::fabs(b[i] - ib[i]);
            }

            return diff + jdst(ia.data(), ib.data(), n);
        }
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        // Save 'a' for err()
        input = a;

        ggml_tensor * out = ggml_top_k(ctx, a, k);
        ggml_set_name(out, "out");

        return out;
    }

    // ties == 2: one structure per row, all of them tie ON the k-th largest value, so the
    // selection is only reproducible if the backend breaks the tie the way the reference does.
    // Every value is a small integer, exact in f32.
    void fill_tie_row(std::vector<float> & data, int64_t n, int64_t row) const {
        const int64_t rot = 7919*row; // keep the rows from selecting alike
        switch (row % 4) {
            case 0: {
                // block-constant: 4 consecutive cells share a value, the qwen4exp indexer
                // layout, and k is not a multiple of 4 so it cuts inside a block - the tie
                // group at the threshold is 4 wide with fewer slots left
                const int64_t nb = (n + 3)/4;
                for (int64_t i = 0; i < n; i++) {
                    data[i] = (float) ((i/4 + rot) % nb);
                }
            } break;
            case 1:
                // one value: the answer is exactly the k lowest indices
                for (int64_t i = 0; i < n; i++) {
                    data[i] = 1.0f;
                }
                break;
            case 2: {
                // a plateau far wider than the slots the distinct values leave: the k-th
                // largest has ~n - k/2 ties and only k - k/2 places to put them
                for (int64_t i = 0; i < n; i++) {
                    data[i] = 1.0f;
                }
                const int64_t nhi    = std::max(1, k/2);
                const int64_t stride = std::max<int64_t>(1, n/nhi);
                for (int64_t j = 0; j < nhi && j*stride < n; j++) {
                    data[(j*stride + rot) % n] = 2.0f + (float) j;
                }
            } break;
            default: {
                // row % 4 == 3: the same plateau, but at zero and with both signs: IEEE compare - and so
                // the reference - calls -0.0 and +0.0 equal, while the bit pattern of -0.0
                // sorts strictly below +0.0. A backend keying on the bits without folding
                // the two together selects every +0.0 before any -0.0 instead of the lowest
                // indices of the two combined.
                for (int64_t i = 0; i < n; i++) {
                    data[i] = (i % 2) ? -0.0f : 0.0f;
                }
                const int64_t nhi    = std::max(1, k/2);
                const int64_t stride = std::max<int64_t>(1, n/nhi);
                for (int64_t j = 0; j < nhi && j*stride < n; j++) {
                    data[(j*stride + rot) % n] = 1.0f + (float) j;
                }
            } break;
        }
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            int tie_denom = std::max(1, std::min(10, k / 2));
            for (int64_t r = 0; r < ggml_nrows(t); r++) {
                std::vector<float> data(t->ne[0]);
                if (ties == 2) {
                    fill_tie_row(data, t->ne[0], r);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                    continue;
                }
                for (int i = 0; i < t->ne[0]; i++) {
                    if (ties) {
                        // integer division to introduce duplicates
                        data[i] = i / tie_denom;
                    } else {
                        data[i] = i;
                    }
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
            }
        }
    }
};

// top_k over rows like log-probabilities: distinct negative values, fewer
// than k +inf (none for k = 1, so the expected indices are unique) and many
// -inf (masked tokens)
struct test_top_k_inf : public test_top_k {
    test_top_k_inf(std::array<int64_t, 4> ne, int k)
        : test_top_k(GGML_TYPE_F32, ne, k, false) {}

    std::string vars() override {
        return test_top_k::vars() + ",inf=1";
    }

    // compare only the output: the input holds infinities, which err() would
    // read as indices
    bool run_whole_graph() override { return true; }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            for (int64_t r = 0; r < ggml_nrows(t); r++) {
                std::vector<float> data(t->ne[0]);
                for (int i = 0; i < t->ne[0]; i++) {
                    data[i] = -1.0f - i;
                }
                std::shuffle(data.begin(), data.end(), rng);
                const int n_pinf = k / 2;
                for (int i = 0; i < t->ne[0]; i++) {
                    if (i < n_pinf) {
                        data[i] = INFINITY;
                    } else if (i % 3 == 0) {
                        data[i] = -INFINITY;
                    }
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
            }
        }
    }
};
// Initialization shared by test_topk_qsa (the six-node chain) and test_qsa_top_k (the single
// GGML_OP_QSA_TOP_K node): the two feed the SAME selection, so the only way to read one against the
// other is for them to be filled identically. Tensors are keyed by name - "score", "cell_blk",
// "kq_mask" - and everything else (the harness sentinels) keeps the uniform fill.
//
// Two tie-free initializations plus the tie modes. The comparison is a set of indices, so before
// the reference's tie-break existed a tie that straddled the k-th value would let CPU and backend
// disagree legitimately; now the ties ARE the interesting case.
//
// default: distinct mask ramp + small scores. Only valid while n_kv <= 2048: f16 stops
//   representing consecutive integers there and saturates to inf past 65504.
// blocked: the production layout - cell i belongs to block i/r, the mask carries the in-block
//   offset - so value(i) = ((i/r + 7919*row) % n_blocks) + 0.25*(i%r), unique for every cell of a
//   row and exact in both f16 and f32 at any n_kv.
//
// `width` is only read by ties == 4, which needs to place its -inf cutoff below the selection width.
static void qsa_select_init_tensors(ggml_context * ctx, int64_t n_blocks, int64_t n_kv, int64_t n_tps,
                                    int64_t n_stream, bool blocked, int ties, int width) {
    // [TAG_QSA_WHOLE_BLOCKS] ties == 5: a census-shaped selection (tests/qsa-select-ref.h) - one sequence in
    // permuted cells, the +1e9 tail boost on the tail block and the spare dead id, -inf ids past it, a causal 0/-inf
    // mask, ReLU-style tied block scores, stream 0 a full cache - i.e. the inputs the deployed census produces,
    // which the whole-block form's tail / eligibility rules read. n_blocks must be ceil(n_kv / 4).
    if (ties == 5) {
        GGML_ASSERT(n_blocks == (n_kv + 3)/4);
        const qsa_census_case c = qsa_census_make(n_kv, n_tps, n_stream, (uint32_t) (n_kv*31 + n_tps*7 + n_stream), false);
        int seen = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE) {
                continue;
            }
            const char * name = ggml_get_name(t);
            if (strcmp(name, "cell_blk") == 0) {
                seen++;
                ggml_backend_tensor_set(t, c.cell_blk.data(), 0, c.cell_blk.size()*sizeof(int32_t));
            } else if (strcmp(name, "score") == 0) {
                seen++;
                ggml_backend_tensor_set(t, c.score.data(), 0, c.score.size()*sizeof(float));
            } else if (strcmp(name, "kq_mask") == 0) {
                seen++;
                if (t->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> h(c.mask.size());
                    for (size_t i = 0; i < h.size(); ++i) {
                        h[i] = ggml_fp32_to_fp16(c.mask[i]);
                    }
                    ggml_backend_tensor_set(t, h.data(), 0, h.size()*sizeof(ggml_fp16_t));
                } else {
                    ggml_backend_tensor_set(t, c.mask.data(), 0, c.mask.size()*sizeof(float));
                }
            } else {
                init_tensor_uniform(t, 0.0f, 0.5f);
            }
        }
        GGML_ASSERT(seen == 3);
        GGML_UNUSED(blocked);
        GGML_UNUSED(width);
        return;
    }

    GGML_UNUSED(n_stream);
    const int64_t r = (n_kv + n_blocks - 1)/n_blocks;
    // The three inputs are found BY NAME, so a rename in either build_graph would silently fall
    // through to the uniform fill: the cell->block map would become random floats reinterpreted as
    // indices and the tie constructions would stop constructing ties, i.e. the tests would still
    // pass and cover nothing. Count them and require all three.
    int seen_score = 0, seen_cell = 0, seen_mask = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op != GGML_OP_NONE) {
            continue;
        }
        const char * name = ggml_get_name(t);
        if (strcmp(name, "cell_blk") == 0) {
            // cell_blk [n_kv, n_stream]
            seen_cell++;
            GGML_ASSERT(t->type == GGML_TYPE_I32);
            std::vector<int32_t> data(ggml_nelements(t));
            for (int64_t row = 0; row < ggml_nrows(t); row++) {
                for (int64_t i = 0; i < n_kv; i++) {
                    data[row * n_kv + i] = blocked ? (int32_t) std::min(i/r, n_blocks - 1) : rand() % n_blocks;
                }
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int32_t));
        } else if (strcmp(name, "kq_mask") == 0) {
            // kq_mask [n_kv, n_tps, (1,) n_stream], f16 in the flash-attention graph the fusion and
            // the op are built for, f32 in the one case that covers the reference's other branch
            seen_mask++;
            GGML_ASSERT(t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32);
            std::vector<char> data(ggml_nbytes(t));
            for (int64_t row = 0; row < ggml_nrows(t); row++) {
                // ties == 4: the masked tail every prefill ubatch boundary produces. Cells at or
                // past a per-row cutoff are masked with -INFINITY, and the cutoff is BELOW the
                // selection width, so the finite values do not fill the row and the selection has
                // to reach into the masked set. score + -inf == -inf for every one of them, so the
                // tie group at the threshold is (n_kv - cut) wide with (width - cut) slots left -
                // by far the widest tie the deployed selection ever sees, and one no backend can
                // resolve by luck: only an index-ordered tie-break gives the reference's answer
                // (cells 0..width-1). Not a synthetic shape - it is what the mask does at every
                // ubatch whose causal window is shorter than indexer_top_k.
                const int64_t cut = std::max<int64_t>(1, width - 1 - (row % 4)*(width/8));
                for (int64_t i = 0; i < n_kv; i++) {
                    // ties: no in-block offset, so a block's cells are indistinguishable
                    const float v = ties == 4 ? (i < cut ? 0.0f : -INFINITY)
                                  : ties      ? 0.0f
                                  : blocked   ? 0.25f*(float) (i % r)
                                              : (float) i;
                    if (t->type == GGML_TYPE_F16) {
                        ((ggml_fp16_t *) data.data())[row * n_kv + i] = ggml_fp32_to_fp16(v);
                    } else {
                        ((float *) data.data())[row * n_kv + i] = v;
                    }
                }
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size());
        } else if (strcmp(name, "score") == 0) {
            seen_score++;
            if (!blocked) {
                // the default initialization gets its distinct values from the mask ramp instead
                init_tensor_uniform(t, 0.0f, 0.5f);
                continue;
            }
            // score [n_blocks, n_tps, n_stream]; +7919*row keeps the rows from selecting alike.
            // Only the score gets the block ramp: the harness plants 1024-element F32 sentinels
            // after every input (add_sentinel), and writing n_blocks values into one of those
            // overflows its buffer as soon as n_blocks > 1024 - the GGML_TEST_TOPK_PROD shapes.
            // The sentinels keep the uniform init below.
            GGML_ASSERT(t->ne[0] == n_blocks && ggml_nrows(t) == n_tps*n_stream);
            std::vector<float> data(ggml_nelements(t));
            for (int64_t row = 0; row < ggml_nrows(t); row++) {
                for (int64_t b = 0; b < n_blocks; b++) {
                    const int64_t v = (b + 7919*row) % n_blocks;
                    // ties == 4 shares the ties == 1 score (one value per block): its tie group is
                    // built by the mask, not by the score
                    data[row * n_blocks + b] = ties == 2 ? 0.0f : (float) (ties == 3 ? v/8 : v);
                }
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
        } else {
            init_tensor_uniform(t, 0.0f, 0.5f);
        }
    }
    // exactly one of each, or a rename has made this initializer vacuous
    GGML_ASSERT(seen_score == 1 && seen_cell == 1 && seen_mask == 1);
}

// qwen4exp QSA indexer top-k fusion: expand per-block scores to cells, add the f16 mask, top-k.
struct test_topk_qsa : public test_case {
    const int64_t n_blocks;
    const int64_t n_kv;
    const int64_t n_tps;
    const int64_t n_stream;
    const int     width;
    // production cell->block layout; the only tie-free choice once n_kv > 2048 (see initialize_tensors)
    const bool    blocked;
    // 0 = off. Otherwise place the TOP_K destination inside the storage of one of the three operands
    // the FUSED dispatch reads at the TOP_K node - 1 = scores (the cont(permute(score)) the shader
    // gathers from), 2 = cell_blk, 3 = the raw f16 mask - by pointing out->view_src at it. That is
    // what ggml-alloc does by accident in the real graph: all three die inside the pattern's window
    // (scores at the GET_ROWS, cell_blk after the LAST QSA layer's GET_ROWS since one input serves
    // every layer of the ratio) and the TOP_K is a fresh best-fit allocation that can land in the
    // freed block. Unfused the graph is still correct - every read precedes the TOP_K in graph order
    // - so a backend that fuses here MUST veto (ggml_vk_can_fuse_topk_qsa tests exactly these three
    // against the destination; for cell_blk and the mask the generic src/dst loop cannot, their op is
    // GGML_OP_NONE and it skips those). test-backend-ops allocates linearly and would never produce
    // the alias on its own, so without this no TOPK_QSA case reaches that veto.
    const int     alias;
    // 0 = the tie-free initialisations below. Otherwise the tie structure the deployed
    // selection actually runs into, which those deliberately avoid: the mask is all zero, so
    // every cell of a block carries the identical block score exactly as qwen4exp expands it,
    // and the width cuts inside a block. 1 = one score per block (tie group = ratio cells),
    // 2 = every cell of the row equal, 3 = eight blocks share a score, so the threshold's tie
    // group is 8*ratio cells wide with 3 slots left, 4 = a -INFINITY mask past a per-row cutoff
    // below the width, so the threshold value is -inf and the selection reaches into the masked
    // set (qsa_select_init_tensors). Requires blocked. The ids are compared
    // exactly, which is only meaningful because the CPU reference resolves ties by index
    // (cmp_top_k) and the Vulkan radix-select does too.
    const int     ties;
    ggml_tensor * out {};

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "TOPK_QSA";
    }

    std::string vars() override {
        return VARS_TO_STR8(n_blocks, n_kv, n_tps, n_stream, width, blocked, alias, ties);
    }

    test_topk_qsa(int64_t n_blocks = 512, int64_t n_kv = 2048, int64_t n_tps = 2, int64_t n_stream = 1, int width = 1500,
                  bool blocked = false, int alias = 0, int ties = 0)
        : n_blocks(n_blocks), n_kv(n_kv), n_tps(n_tps), n_stream(n_stream), width(width), blocked(blocked), alias(alias),
          ties(ties) {
        GGML_ASSERT(ties == 0 || blocked);
    }

    double max_err() override { return 0.0; }
    bool run_whole_graph() override { return true; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, n_tps, n_stream);
        ggml_set_name(score, "score");
        ggml_tensor * cell_blk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_stream);
        ggml_set_name(cell_blk, "cell_blk");
        ggml_tensor * kq_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_kv, n_tps, n_stream);
        ggml_set_name(kq_mask, "kq_mask");

        ggml_tensor * a = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));
        ggml_tensor * e = ggml_get_rows(ctx, a, cell_blk);
        e = ggml_cont(ctx, ggml_permute(ctx, e, 1, 0, 2, 3));
        ggml_tensor * m = ggml_cast(ctx, kq_mask, GGML_TYPE_F32);
        e = ggml_add(ctx, e, ggml_reshape_3d(ctx, m, n_kv, n_tps, n_stream));
        out = ggml_top_k(ctx, e, width);
        ggml_set_name(out, "out");

        if (alias) {
            ggml_tensor * target = alias == 1 ? a : alias == 2 ? cell_blk : kq_mask;
            GGML_ASSERT(out->view_src == nullptr && ggml_nbytes(out) <= ggml_nbytes(target));
            out->view_src  = target;
            out->view_offs = 0;
        }
        return out;
    }

    std::vector<ggml_tensor *> fusion_test_nodes() override { return { out }; }

    void initialize_tensors(ggml_context * ctx) override {
        qsa_select_init_tensors(ctx, n_blocks, n_kv, n_tps, n_stream, blocked, ties, width);
    }

    // top-k output order is unspecified; compare each ROW as a set of indices, and require that
    // each row is a selection out of its own n_kv cells (see cmp_ids_per_row). Only the `out`
    // tensor reaches err() here - run_whole_graph() is on and fusion_test_nodes() names it.
    double err(const float * a, const float * b, size_t n) override {
        return cmp_ids_per_row(a, b, n, width, n_kv);
    }
};

// [TAG_QSA_FUSED_SELECT] the same selection as test_topk_qsa above, emitted as ONE node
// (GGML_OP_QSA_TOP_K) instead of six. Same inputs, same initialization, same comparison, so the two
// tests read against each other directly: -o TOPK_QSA is the seven-node pattern the Vulkan backend
// fuses, -o QSA_TOP_K is the op that replaces it under GGML_VK_QSA_FUSED_SELECT=1.
//
// No `alias` mode here, unlike test_topk_qsa, and that absence is the point of the task. The fused
// pattern reads three buffers at its TOP_K that ggml-alloc believes died six nodes earlier, so it
// needs a hand-written veto and a test that forces each aliasing placement. The op's three
// operands are its own srcs: ggml_gallocr allocates a node's destination BEFORE it frees the
// parents whose use count the node just dropped to zero (ggml-alloc.c, the parent-update loop runs
// after ggml_gallocr_allocate_node), and QSA_TOP_K is not in ggml_op_can_inplace, so a src's block
// is not on the free list when the dst is placed and no in-place reuse is offered either. The
// alias is unreachable, and unlike the fusion there is no unfused fallback that would be correct
// under it - the shader writes ids while other workgroups are still reading the score - so forcing
// it would assert a graph the allocator cannot build. The reachability claim is checked where it
// can be: the M2.1 replay runs the real ggml_gallocr over the production reserve graph and reports
// dst-over-src overlaps for every QSA_TOP_K node (an external fusion-replay probe).
struct test_qsa_top_k : public test_case {
    const int64_t   n_blocks;
    const int64_t   n_kv;
    const int64_t   n_tps;
    const int64_t   n_stream;
    const int       width;
    // production cell->block layout; the only tie-free choice once n_kv > 2048 (see qsa_select_init_tensors)
    const bool      blocked;
    // as test_topk_qsa::ties: 1 = one score per block, 2 = the whole row equal, 3 = eight blocks
    // share a score, 4 = a -INFINITY mask past a per-row cutoff below the width, so the threshold
    // value is -inf and the selection reaches into the masked set. Requires blocked.
    const int       ties;
    // the graph builds the op with an f16 mask (flash attention); GGML_TYPE_F32 covers the other
    // branch of the CPU reference. The Vulkan shader binds a float16_t mask, so an f32 case is
    // expected to report "not supported" there rather than to run.
    const ggml_type mask_type;
    // [TAG_QSA_WHOLE_BLOCKS] 0 = the cell form (ggml_qsa_top_k); > 0 = ggml_qsa_top_k_blocks with n_sel whole
    // blocks. Its ids are in a canonical order (ascending, padded with a repeat of the last), so the rows are
    // compared element for element - including the padding, which a set comparison would call a repeated id.
    const int       n_sel;

    std::string vars() override {
        // n_sel only when set, so the cell-form cases keep their names
        return n_sel > 0 ? VARS_TO_STR9(n_blocks, n_kv, n_tps, n_stream, width, blocked, ties, mask_type, n_sel)
                         : VARS_TO_STR8(n_blocks, n_kv, n_tps, n_stream, width, blocked, ties, mask_type);
    }

    test_qsa_top_k(int64_t n_blocks = 512, int64_t n_kv = 2048, int64_t n_tps = 2, int64_t n_stream = 1,
                   int width = 1500, bool blocked = false, int ties = 0, ggml_type mask_type = GGML_TYPE_F16,
                   int n_sel = 0)
        : n_blocks(n_blocks), n_kv(n_kv), n_tps(n_tps), n_stream(n_stream), width(width), blocked(blocked),
          ties(ties), mask_type(mask_type), n_sel(n_sel) {
        GGML_ASSERT(ties == 0 || blocked);
    }

    double max_err() override { return 0.0; }
    bool run_whole_graph() override { return true; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // the score in its natural per-block layout: no permute, no cont, no expansion
        ggml_tensor * score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks, n_tps, n_stream);
        ggml_set_name(score, "score");
        ggml_tensor * cell_blk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_stream);
        ggml_set_name(cell_blk, "cell_blk");
        // [n_kv, n_tps, 1, n_stream], exactly as llm_graph_input_attn_kv builds it
        ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx, mask_type, n_kv, n_tps, 1, n_stream);
        ggml_set_name(kq_mask, "kq_mask");

        ggml_tensor * out = n_sel > 0 ? ggml_qsa_top_k_blocks(ctx, score, cell_blk, kq_mask, width, n_sel)
                                      : ggml_qsa_top_k(ctx, score, cell_blk, kq_mask, width);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        qsa_select_init_tensors(ctx, n_blocks, n_kv, n_tps, n_stream, blocked, ties, width);
    }

    // top-k output order is unspecified; compare each ROW as a set of indices, and require that
    // each row is a selection out of its own n_kv cells (see cmp_ids_per_row). Only the `out`
    // tensor reaches err() here - run_whole_graph() is on, so only the graph's `out` is compared.
    // [TAG_QSA_WHOLE_BLOCKS] the block form: element for element (canonical order), every id in range.
    double err(const float * a, const float * b, size_t n) override {
        if (n_sel > 0) {
            double diff = 0.0;
            for (size_t i = 0; i < n; i++) {
                diff += a[i] != b[i] ? 1.0 : 0.0;
                diff += (a[i] < 0 || a[i] >= (float) n_kv) ? 1.0 : 0.0;
            }
            return diff;
        }
        return cmp_ids_per_row(a, b, n, width, n_kv);
    }
};

// [TAG_QSA_TOPK_V2] GGML_OP_QSA_TOP_K, EXACT: the ids AND their order, against a reference implemented
// here from the backend's documented selection contract (topk_radix_select.comp / _v2.comp):
//   key(x) = the order-preserving float->uint map with -0.0 folded onto +0.0,
//   T      = the k-th largest key of the row,
//   ids    = every cell with key > T in ascending id order, then the lowest-id cells with key == T,
//            ascending, until k ids are out.
// The QSA gather path sums the attention window in id order, so the order is part of what the
// deployed graph sees - test_qsa_top_k compares sets only. The CPU backend (partial_sort, first two
// swapped) is still checked against the reference as a SET, which validates the reference itself.
// The two Vulkan kernels (GGML_VK_QSA_TOPK=v1|v2) both have to match it id for id, i.e. they are
// byte-identical to each other on every case here; GGML_TEST_DUMP_IDS diffs them directly.
//
// Modes (value(i) = score[cell_blk[i]] + mask[i], f32 add of the f16 mask, as every backend does):
//   random  scores U[0,8), mask -U[0,2] in f16, random cell->block map: (almost) distinct values
//   quant   16 score levels, mask 0 / -1: wide tie groups at every threshold
//   equal   every cell equal: the whole row is one tie group
//   mask    -inf masked cells; on even rows the finite cells do not fill k, so T = -inf
//   zero    -0/+0 scores and -0/+0 f16 masks around a handful of positives: T is a zero, the
//           tie group mixes both signs (folded keys), negatives below
//   inf     +inf / -inf scores and -inf masks (never inf + -inf): T = +inf once enough +inf cells
//   nan     as inf plus inf + -inf = NaN cells and NaN scores of both signs. The NaN a device add
//           produces is the device's, and the CPU comparator does not order NaN, so only the
//           structure is checked here (ids in range, no repeats); v1 vs v2 is by dump.
//   prod    the deployed layout: cell i in block i/4, U[0,4) block scores (4-cell tie groups),
//           a -inf padded tail and the causal mask of the verify rows
//   census  the census-shaped inputs of tests/qsa-select-ref.h (ReLU-tied block scores, +1e9 tail boost on the
//           tail block and the spare id, -inf ids past it, permuted cells, causal 0/-inf mask; n_blocks =
//           ceil(n_kv/4)) - what the deployed graph feeds the op
//
// [TAG_QSA_WHOLE_BLOCKS] n_sel > 0 builds the block-level form (ggml_qsa_top_k_blocks) instead, against its own
// reference from the ggml.h contract: eligible blocks (score finite and < GGML_QSA_TAIL_MIN), the min(n_sel, |E|)
// best by (key desc, block asc), selected cells = tail (value finite and >= GGML_QSA_TAIL_MIN) + the visible cells
// (mask not -inf) of those blocks, ascending, truncated to k, padded with the largest selected id (or 0..k-1). That
// order is canonical, so the CPU backend is compared element for element as well; every mode applies, NaN included
// (eligibility and the tail test reject NaN whatever its bits, and the masks are 0 / -inf).
struct test_qsa_top_k_exact : public test_case {
    enum { M_RANDOM, M_QUANT, M_EQUAL, M_MASK, M_ZERO, M_INF, M_NAN, M_PROD, M_CENSUS, M_COUNT };

    const int64_t n_kv;
    const int64_t n_tps;
    const int64_t n_stream;
    const int     width;
    const int     mode;
    const int     n_sel; // 0 = the cell form, > 0 = the block-level form
    std::vector<std::vector<int32_t>> expected; // per row, filled by initialize_tensors

    static const char * mode_name(int m) {
        static const char * names[M_COUNT] = { "random", "quant", "equal", "mask", "zero", "inf", "nan", "prod", "census" };
        return m >= 0 && m < M_COUNT ? names[m] : "?";
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "QSA_TOP_K_EXACT";
    }

    std::string vars() override {
        const std::string data = mode_name(mode);
        // n_sel only when set, so the cell-form cases keep their names
        return n_sel > 0 ? VARS_TO_STR6(n_kv, n_tps, n_stream, width, data, n_sel)
                         : VARS_TO_STR5(n_kv, n_tps, n_stream, width, data);
    }

    test_qsa_top_k_exact(int64_t n_kv, int64_t n_tps, int64_t n_stream, int width, int mode, int n_sel = 0)
        : n_kv(n_kv), n_tps(n_tps), n_stream(n_stream), width(width), mode(mode), n_sel(n_sel) {
        GGML_ASSERT(width > 0 && width <= n_kv && mode >= 0 && mode < M_COUNT && n_sel >= 0);
    }

    double max_err() override { return 0.0; }
    bool run_whole_graph() override { return true; }

    int64_t n_blocks() const { return mode == M_CENSUS ? (n_kv + 3)/4 : std::max<int64_t>(1, n_kv / 4); }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks(), n_tps, n_stream);
        ggml_set_name(score, "score");
        ggml_tensor * cell_blk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_stream);
        ggml_set_name(cell_blk, "cell_blk");
        ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_tps, 1, n_stream);
        ggml_set_name(kq_mask, "kq_mask");
        ggml_tensor * out = n_sel > 0 ? ggml_qsa_top_k_blocks(ctx, score, cell_blk, kq_mask, width, n_sel)
                                      : ggml_qsa_top_k(ctx, score, cell_blk, kq_mask, width);
        ggml_set_name(out, "out");
        return out;
    }

    // the backends' key: order-preserving, -0.0 == +0.0
    static uint32_t key_of(float x) {
        uint32_t y;
        memcpy(&y, &x, sizeof(y));
        if (y == 0x80000000u) {
            y = 0u;
        }
        return (y & 0x80000000u) ? ~y : (y | 0x80000000u);
    }

    void initialize_tensors(ggml_context * ctx) override {
        const int64_t nb    = n_blocks();
        const int64_t nrows = n_tps*n_stream;
        std::mt19937 rng(0x9e3779b9u ^ (uint32_t) (mode*7919 + n_kv*31 + n_tps*131 + n_stream*17 + width));
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        const float inf = INFINITY;
        float qnan_pos, qnan_neg;
        {
            const uint32_t bp = 0x7FC00000u, bn = 0xFFC00000u;
            memcpy(&qnan_pos, &bp, 4);
            memcpy(&qnan_neg, &bn, 4);
        }

        std::vector<int32_t> cb(n_kv*n_stream);
        std::vector<float> sc(nb*nrows);
        std::vector<ggml_fp16_t> mk(n_kv*nrows);
        if (mode == M_CENSUS) {
            const qsa_census_case c = qsa_census_make(n_kv, n_tps, n_stream, (uint32_t) (n_kv*31 + n_tps*7 + n_stream + width), false);
            GGML_ASSERT(c.n_blocks == nb && c.score.size() == sc.size() && c.cell_blk.size() == cb.size() && c.mask.size() == mk.size());
            cb = c.cell_blk;
            sc = c.score;
            for (size_t i = 0; i < mk.size(); i++) {
                mk[i] = ggml_fp32_to_fp16(c.mask[i]);
            }
        }
        for (int64_t s = 0; s < n_stream && mode != M_CENSUS; s++) {
            for (int64_t i = 0; i < n_kv; i++) {
                cb[s*n_kv + i] = mode == M_PROD ? (int32_t) std::min(i/4, nb - 1) : (int32_t) (rng() % (uint32_t) nb);
            }
        }
        for (int64_t row = 0; row < nrows && mode != M_CENSUS; row++) {
            for (int64_t b = 0; b < nb; b++) {
                const float x = u01(rng);
                float v = 0.0f;
                switch (mode) {
                    case M_RANDOM: v = 8.0f*x; break;
                    case M_QUANT:  v = 0.25f*(float) (rng() % 16); break;
                    case M_EQUAL:  v = 1.5f; break;
                    case M_MASK:   v = 8.0f*x; break;
                    case M_ZERO:   v = x < 0.001f ? 1.0f + u01(rng) : x < 0.25f ? -(1.0f + u01(rng)) : ((rng() & 1) ? -0.0f : 0.0f); break;
                    case M_INF:    v = x < 0.02f ? inf : x < 0.04f ? -inf : 8.0f*u01(rng); break;
                    case M_NAN:    v = x < 0.02f ? inf : x < 0.04f ? -inf : x < 0.05f ? qnan_pos : x < 0.06f ? qnan_neg : 8.0f*u01(rng); break;
                    case M_PROD:   v = 4.0f*x; break;
                }
                sc[row*nb + b] = v;
            }
        }
        const int64_t used = n_kv - n_kv/16;
        for (int64_t row = 0; row < nrows && mode != M_CENSUS; row++) {
            const int64_t t = row % n_tps;
            const int64_t s = row / n_tps;
            for (int64_t i = 0; i < n_kv; i++) {
                const float sv = sc[row*nb + cb[s*n_kv + i]];
                float m = 0.0f;
                switch (mode) {
                    case M_RANDOM: m = -2.0f*u01(rng); break;
                    case M_QUANT:  m = (rng() % 4 == 0) ? -1.0f : 0.0f; break;
                    case M_EQUAL:  m = 0.0f; break;
                    case M_MASK:   m = ((row % 2 == 0 && i >= width/2) || u01(rng) < 0.4f) ? -inf : 0.0f; break;
                    case M_ZERO:   m = (rng() & 1) ? -0.0f : 0.0f; break;
                    case M_INF:    m = (sv != inf && u01(rng) < 0.1f) ? -inf : 0.0f; break;
                    case M_NAN:    m = u01(rng) < 0.1f ? -inf : 0.0f; break;
                    case M_PROD:   m = i < used - (n_tps - 1 - t) ? 0.0f : -inf; break;
                }
                mk[row*n_kv + i] = ggml_fp32_to_fp16(m);
            }
        }

        int seen = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE) {
                continue;
            }
            const char * name = ggml_get_name(t);
            if (strcmp(name, "score") == 0) {
                GGML_ASSERT(ggml_nbytes(t) == sc.size()*sizeof(float));
                ggml_backend_tensor_set(t, sc.data(), 0, ggml_nbytes(t));
                seen++;
            } else if (strcmp(name, "cell_blk") == 0) {
                GGML_ASSERT(ggml_nbytes(t) == cb.size()*sizeof(int32_t));
                ggml_backend_tensor_set(t, cb.data(), 0, ggml_nbytes(t));
                seen++;
            } else if (strcmp(name, "kq_mask") == 0) {
                GGML_ASSERT(ggml_nbytes(t) == mk.size()*sizeof(ggml_fp16_t));
                ggml_backend_tensor_set(t, mk.data(), 0, ggml_nbytes(t));
                seen++;
            } else {
                init_tensor_uniform(t, 0.0f, 0.5f);
            }
        }
        GGML_ASSERT(seen == 3);

        expected.assign(nrows, {});

        // [TAG_QSA_WHOLE_BLOCKS] the block-level reference (the ggml.h contract, the GPU's visibility test)
        if (n_sel > 0) {
            std::vector<int32_t> el;
            std::vector<uint8_t> flag(nb);
            for (int64_t row = 0; row < nrows; row++) {
                const int64_t s  = row / n_tps;
                const float * rs = sc.data() + row*nb;
                el.clear();
                for (int64_t b = 0; b < nb; b++) {
                    if (std::isfinite(rs[b]) && rs[b] < GGML_QSA_TAIL_MIN) {
                        el.push_back((int32_t) b);
                    }
                }
                const size_t n_take = std::min<size_t>((size_t) n_sel, el.size());
                std::partial_sort(el.begin(), el.begin() + n_take, el.end(), [&](int32_t x, int32_t y) {
                    const uint32_t kx = key_of(rs[x]), ky = key_of(rs[y]);
                    return kx > ky || (kx == ky && x < y);
                });
                std::fill(flag.begin(), flag.end(), 0);
                for (size_t j = 0; j < n_take; j++) {
                    flag[el[j]] = 1;
                }
                std::vector<int32_t> & e = expected[row];
                for (int64_t i = 0; i < n_kv && (int64_t) e.size() < width; i++) {
                    const int32_t b = cb[s*n_kv + i];
                    const float   m = ggml_fp16_to_fp32(mk[row*n_kv + i]);
                    const float   v = rs[b] + m;
                    const bool tail    = std::isfinite(v) && v >= GGML_QSA_TAIL_MIN;
                    const bool visible = !(std::isinf(m) && m < 0.0f);
                    if (tail || (flag[b] && visible)) {
                        e.push_back((int32_t) i);
                    }
                }
                const int64_t c = (int64_t) e.size();
                for (int64_t j = c; j < width; j++) {
                    e.push_back(c > 0 ? e[c - 1] : (int32_t) j);
                }
            }
            return;
        }

        // the reference, row by row: T by nth_element over the keys, then the two classes by id
        if (mode == M_NAN) {
            return;
        }
        std::vector<uint32_t> keys(n_kv), tmp(n_kv);
        for (int64_t row = 0; row < nrows; row++) {
            const int64_t s = row / n_tps;
            for (int64_t i = 0; i < n_kv; i++) {
                const float v = sc[row*nb + cb[s*n_kv + i]] + ggml_fp16_to_fp32(mk[row*n_kv + i]);
                keys[i] = key_of(v);
            }
            tmp = keys;
            std::nth_element(tmp.begin(), tmp.begin() + (width - 1), tmp.end(), std::greater<uint32_t>());
            const uint32_t T = tmp[width - 1];
            std::vector<int32_t> & e = expected[row];
            e.reserve(width);
            for (int64_t i = 0; i < n_kv; i++) {
                if (keys[i] > T) {
                    e.push_back((int32_t) i);
                }
            }
            int64_t desired = width - (int64_t) e.size();
            GGML_ASSERT(desired >= 1);
            for (int64_t i = 0; i < n_kv && desired > 0; i++) {
                if (keys[i] == T) {
                    e.push_back((int32_t) i);
                    desired--;
                }
            }
            GGML_ASSERT((int64_t) e.size() == width);
        }
    }

    // a: the backend under test (exact, order included), b: the CPU backend (as a set)
    double err(const float * a, const float * b, size_t n) override {
        GGML_ASSERT(n % (size_t) width == 0);
        const int64_t nrows = (int64_t) (n / (size_t) width);
        double diff = 0.0;
        std::vector<int32_t> ids(width), ref(width);
        int64_t bad_row = -1, bad_col = -1;
        for (int64_t r = 0; r < nrows; r++) {
            const float * ar = a + r*width;
            for (int64_t c = 0; c < width; c++) {
                ids[c] = (int32_t) ar[c];
                if (ids[c] < 0 || ids[c] >= n_kv) {
                    diff += 1.0;
                }
            }
            if (mode == M_NAN && n_sel == 0) {
                std::sort(ids.begin(), ids.end());
                if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
                    diff += 1.0;
                }
                continue;
            }
            GGML_ASSERT((int64_t) expected.size() == nrows);
            for (int64_t c = 0; c < width; c++) {
                if (ids[c] != expected[r][c]) {
                    diff += 1.0;
                    if (bad_row < 0) {
                        bad_row = r;
                        bad_col = c;
                    }
                }
            }
            // the CPU reference's set == the reference's set (block form: its canonical order too)
            for (int64_t c = 0; c < width; c++) {
                ids[c] = (int32_t) b[r*width + c];
            }
            if (n_sel > 0) {
                if (ids != expected[r]) {
                    diff += 1.0;
                    printf("[cpu != reference, row %" PRId64 "] ", r);
                }
                continue;
            }
            ref = expected[r];
            std::sort(ids.begin(), ids.end());
            std::sort(ref.begin(), ref.end());
            if (ids != ref) {
                diff += 1.0;
                printf("[cpu set != reference set, row %" PRId64 "] ", r);
            }
        }
        if (bad_row >= 0) {
            printf("[first order mismatch row %" PRId64 " col %" PRId64 ": got %d want %d] ", bad_row, bad_col,
                   (int) a[bad_row*width + bad_col], expected[bad_row][bad_col]);
        }
        return diff;
    }
};

// [TAG_QSA_POOL_REMAP] GGML_OP_MUL_MAT_ROWS (ggml_mul_mat_rows): the qwen4exp indexer score read from the pooled-key
// store through the remap, instead of get_rows (a copy of the whole pool, every QSA layer) + mul_mat.
//
//   MUL_MAT_ROWS_EXACT  the op against get_rows + mul_mat ON THE SAME BACKEND, bit for bit - the contract of the op.
//                       The graph computes both and returns concat(idx - copy, copy - idx) along dim 1: an element is
//                       +0 exactly where the two results are the same float (finite data), and a sign-of-zero
//                       difference leaves a -0 in one of the halves. The CPU computes +0 everywhere (its op IS the
//                       gather + the unchanged mul_mat), and the two outputs are compared BITWISE (err = the number of
//                       differing bit patterns, max 0). Shapes: every kernel the plain MUL_MAT picks for an f32 A -
//                       the mat-vec (N <= 8 on one batch, N = 1 on several), the matmul's small / medium / large
//                       tiles, aligned (K % 8 == 0) and unaligned K, partial tiles (M not a tile multiple), batches.
//   MUL_MAT_ROWS        the op against the CPU backend (NMSE, as MUL_MAT). In perf mode also form 1 / 2: the two
//                       halves of the copy path the op replaces (form 1 = the get_rows gather, form 2 = the mul_mat
//                       of the gathered copy), so copy = form 1 + form 2 against the op's form 0.
//
// Data modes: random (rows drawn with repeats), perm (distinct rows in random order), ident (row = block, per stream:
// the layout after a full rebuild), spare (the upper half of the blocks names one row: the store's spare row), quant
// (five levels in A and B: exact ties and signed zeros in the products), prod (unit-norm keys and queries, a near-
// identity remap with 1 % of the blocks moved, as the stable slots after a restore plus some dirty reassignments).
struct test_mul_mat_rows : public test_case {
    enum { M_RANDOM, M_PERM, M_IDENT, M_SPARE, M_QUANT, M_PROD, M_COUNT };

    const int64_t K;
    const int64_t M;      // rows named per batch (the blocks)
    const int64_t N;      // columns of B (heads x queries)
    const int64_t n_b;    // batches (streams)
    const int     mode;
    const bool    exact;
    const int     form;   // 0 = the op, 1 = the copy path's get_rows alone, 2 = its mul_mat alone (perf)

    static const char * mode_name(int m) {
        static const char * names[M_COUNT] = { "random", "perm", "ident", "spare", "quant", "prod" };
        return m >= 0 && m < M_COUNT ? names[m] : "?";
    }

    // the store: each stream owns M + M/8 + 1 rows (the slots, the spare row last)
    int64_t rows_per_stream() const { return M + M/8 + 1; }
    int64_t n_rows()          const { return n_b*rows_per_stream(); }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return exact ? "MUL_MAT_ROWS_EXACT" : "MUL_MAT_ROWS";
    }

    std::string vars() override {
        const std::string data = mode_name(mode);
        return form == 0 ? VARS_TO_STR5(K, M, N, n_b, data) : VARS_TO_STR6(K, M, N, n_b, data, form);
    }

    test_mul_mat_rows(int64_t K, int64_t M, int64_t N, int64_t n_b, int mode, bool exact, int form = 0)
        : K(K), M(M), N(N), n_b(n_b), mode(mode), exact(exact), form(form) {
        GGML_ASSERT(mode >= 0 && mode < M_COUNT && form >= 0 && form <= 2 && !(exact && form != 0));
    }

    double max_nmse_err() override { return 5e-4; }
    double max_err() override { return exact ? 0.0 : max_nmse_err(); }
    double max_err(ggml_backend_t backend) override { GGML_UNUSED(backend); return max_err(); }

    double err(const float * a, const float * b, size_t n) override {
        if (!exact) {
            return nmse(a, b, n);
        }
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) {
            bad += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        }
        return (double) bad;
    }

    // the exact form compares only its output (the products themselves are not bitwise equal to the CPU's)
    bool run_whole_graph() override { return exact; }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return form == 1 ? 0 : (uint64_t) 2*M*N*K*n_b;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N, n_b);
        ggml_set_name(b, "b");

        if (form == 2) {
            ggml_tensor * pooled = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, n_b);
            ggml_set_name(pooled, "pooled");
            ggml_tensor * out = ggml_mul_mat(ctx, pooled, b);
            ggml_set_name(out, "out");
            return out;
        }

        ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_rows());
        ggml_set_name(a, "a");
        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, M, n_b);
        ggml_set_name(rows, "rows");

        if (form == 1) {
            ggml_tensor * out = ggml_get_rows(ctx, a, ggml_reshape_1d(ctx, rows, M*n_b));
            ggml_set_name(out, "out");
            return out;
        }

        ggml_tensor * idx = ggml_mul_mat_rows(ctx, a, rows, b);
        if (!exact) {
            ggml_set_name(idx, "out");
            return idx;
        }
        ggml_set_name(idx, "idx");

        // the qwen4exp copy path, node for node: get_rows of the flat remap, reshaped to [K, M, n_b], then mul_mat
        ggml_tensor * pooled = ggml_get_rows(ctx, a, ggml_reshape_1d(ctx, rows, M*n_b));
        pooled = ggml_reshape_3d(ctx, pooled, K, M, n_b);
        ggml_tensor * copy = ggml_mul_mat(ctx, pooled, b);
        ggml_set_name(copy, "copy");

        ggml_tensor * out = ggml_concat(ctx, ggml_sub(ctx, idx, copy), ggml_sub(ctx, copy, idx), 1);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::mt19937 rng(0x5bd1e995u ^ (uint32_t) (mode*7919 + K*31 + M*131 + N*17 + n_b*7 + form));
        std::uniform_real_distribution<float> u11(-1.0f, 1.0f);
        std::normal_distribution<float>       nrm(0.0f, 1.0f);

        auto fill_f32 = [&](ggml_tensor * t, int64_t row_len) {
            const int64_t n = ggml_nelements(t);
            std::vector<float> v(n);
            if (mode == M_QUANT) {
                static const float lv[5] = { -1.0f, -0.5f, 0.0f, 0.5f, 1.0f };
                for (auto & x : v) {
                    x = lv[rng() % 5];
                    if (x == 0.0f && (rng() & 1)) {
                        x = -0.0f;
                    }
                }
            } else if (mode == M_PROD) {
                for (int64_t r = 0; r < n/row_len; r++) {
                    double ss = 0.0;
                    for (int64_t k = 0; k < row_len; k++) {
                        v[r*row_len + k] = nrm(rng);
                        ss += (double) v[r*row_len + k]*v[r*row_len + k];
                    }
                    const float sc = ss > 0.0 ? (float) (1.0/std::sqrt(ss)) : 1.0f;
                    for (int64_t k = 0; k < row_len; k++) {
                        v[r*row_len + k] *= sc;
                    }
                }
            } else {
                for (auto & x : v) {
                    x = u11(rng);
                }
            }
            ggml_backend_tensor_set(t, v.data(), 0, n*sizeof(float));
        };

        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            const char * name = ggml_get_name(t);
            if (strcmp(name, "rows") == 0) {
                const int64_t rps = rows_per_stream();
                std::vector<int32_t> ids(M*n_b);
                for (int64_t s = 0; s < n_b; s++) {
                    std::vector<int32_t> slot(rps);
                    for (int64_t i = 0; i < rps; i++) {
                        slot[i] = (int32_t) i;
                    }
                    std::shuffle(slot.begin(), slot.end(), rng);
                    for (int64_t m = 0; m < M; m++) {
                        int64_t r = 0;
                        switch (mode) {
                            case M_PERM:  r = slot[m]; break;
                            case M_IDENT: r = m; break;
                            case M_SPARE: r = m < M/2 ? m : rps - 1; break;
                            case M_PROD:  r = (rng() % 100 == 0) ? (int64_t) (rng() % (uint32_t) rps) : m; break;
                            default:      r = (int64_t) (rng() % (uint32_t) rps); break;
                        }
                        ids[s*M + m] = (int32_t) (s*rps + r);
                    }
                }
                ggml_backend_tensor_set(t, ids.data(), 0, ids.size()*sizeof(int32_t));
            } else if (strcmp(name, "a") == 0 || strcmp(name, "b") == 0 || strcmp(name, "pooled") == 0) {
                fill_f32(t, t->ne[0]);
            } else if (t->type == GGML_TYPE_F32 && !ggml_is_view_op(t->op)) {
                init_tensor_uniform(t);
            }
        }
    }
};

// [TAG_QSA_SCORE_EPILOGUE] GGML_OP_QSA_SCORE (ggml_qsa_score): the qwen4exp indexer score with its ReLU, ordered head sum
// and per-block bias as one node.
//
//   QSA_SCORE_EXACT  the op against the chain it replaces ON THE SAME BACKEND, bit for bit - the contract of the op. The
//                    graph computes both and returns concat(op, chain) along dim 1; err() compares the two halves of the
//                    tested backend's output AND the two halves of the CPU's, bitwise (err = differing bit patterns, max
//                    0) - the CPU op must equal the CPU chain too. The chain is build_qsa_select's, node for node:
//                    MUL_MAT (pooled) / MUL_MAT_ROWS (store + remap) -> reshape [M, H, N, n_b] -> RELU (in place) ->
//                    CONT(head 0) + ADD(head 1) + ... -> ADD(bias). Bias modes: 0 none, 1 the QSA_BIAS constants
//                    (+0, 1e9, -inf: the -inf rows stay -inf in both halves), 2 uniform. Shapes: ragged M (not a tile
//                    multiple), every chunk width 1..128 (the widths the Vulkan op refuses - the mat-vec ones - report
//                    "not supported"), unaligned K, two streams, other head counts, the deployed depths.
//   QSA_SCORE        the op against the CPU backend (NMSE); in perf mode the op alone (the perf harness repeats only the
//                    output node, so the chain it replaces cannot be timed here - the A1 classifier times both arms).
struct test_qsa_score : public test_case {
    const int64_t K;
    const int64_t M;      // blocks
    const int64_t N;      // queries (the product has n_head*N columns)
    const int64_t n_b;    // streams
    const int     n_head;
    const bool    rows;   // MUL_MAT_ROWS form (store + remap) vs MUL_MAT (pooled)
    const int     bias;   // 0 none, 1 QSA_BIAS constants, 2 uniform
    const bool    exact;
    const bool    f16a;   // [TAG_QSA_POOLED_F16_MIRROR] the op reads an f16 mirror of the store (set_rows of every row)
    const float   scale;  // [TAG_QSA_POOLED_F16_MIRROR] the store's values x scale (1e-5: f16 denormals in the mirror)

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return exact ? "QSA_SCORE_EXACT" : "QSA_SCORE";
    }

    std::string vars() override {
        if (f16a || scale != 1.0f) {
            return VARS_TO_STR10(K, M, N, n_b, n_head, rows, bias, exact, f16a, scale);
        }
        return VARS_TO_STR8(K, M, N, n_b, n_head, rows, bias, exact);
    }

    test_qsa_score(int64_t K, int64_t M, int64_t N, int64_t n_b, int n_head, bool rows, int bias, bool exact,
                   bool f16a = false, float scale = 1.0f)
        : K(K), M(M), N(N), n_b(n_b), n_head(n_head), rows(rows), bias(bias), exact(exact), f16a(f16a), scale(scale) {
        GGML_ASSERT(!f16a || rows);
    }

    double max_nmse_err() override { return 5e-4; }
    double max_err() override { return exact ? 0.0 : max_nmse_err(); }
    double max_err(ggml_backend_t backend) override { GGML_UNUSED(backend); return max_err(); }

    double err(const float * a, const float * b, size_t n) override {
        if (!exact) {
            return nmse(a, b, n);
        }
        GGML_ASSERT(n == (size_t) (M*2*N*n_b));
        size_t bad = 0;
        for (int64_t s = 0; s < n_b; s++) {
            for (int64_t t = 0; t < N; t++) {
                const size_t i0 = (size_t) ((s*2*N + t)*M);
                const size_t i1 = (size_t) ((s*2*N + N + t)*M);
                for (int64_t m = 0; m < M; m++) {
                    bad += memcmp(&a[i0 + m], &a[i1 + m], sizeof(float)) != 0;
                    bad += memcmp(&b[i0 + m], &b[i1 + m], sizeof(float)) != 0;
                }
            }
        }
        return (double) bad;
    }

    bool run_whole_graph() override { return exact; }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return (uint64_t) 2*M*N*n_head*K*n_b;
    }

    int64_t n_rows() const { return n_b*(M + M/8 + 1); }

    ggml_tensor * chain(ggml_context * ctx, ggml_tensor * a, ggml_tensor * ids, ggml_tensor * b, ggml_tensor * bi) {
        ggml_tensor * score = ids ? ggml_mul_mat_rows(ctx, a, ids, b) : ggml_mul_mat(ctx, a, b);
        score = ggml_reshape_4d(ctx, score, M, n_head, N, n_b);
        score = ggml_relu_inplace(ctx, score);
        ggml_tensor * summed = nullptr;
        for (int h = 0; h < n_head; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx, score, M, N, n_b, score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx, summed, slice) : ggml_cont(ctx, slice);
        }
        return bi ? ggml_add(ctx, summed, bi) : summed;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_head*N, n_b);
        ggml_set_name(b, "b");

        ggml_tensor * a   = rows ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_rows()) : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, M, n_b);
        ggml_set_name(a, "a");
        ggml_tensor * ids = nullptr;
        if (rows) {
            ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, M, n_b);
            ggml_set_name(ids, "rows");
        }
        ggml_tensor * bi = nullptr;
        if (bias) {
            bi = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, M, N, n_b);
            ggml_set_name(bi, "bias");
        }

        ggml_tensor * op = ggml_qsa_score(ctx, a, ids, b, bi, n_head);
        if (f16a) {
            // the store's f16 mirror as qwen4exp writes it: set_rows of the f32 rows into an F16 store
            ggml_tensor * a16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, n_rows());
            ggml_set_name(a16, "a16");
            ggml_tensor * all = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows());
            ggml_set_name(all, "allrows");
            ggml_qsa_score_set_a_f16(op, ggml_set_rows(ctx, a16, a, all));
        }
        if (!exact) {
            ggml_set_name(op, "out");
            return op;
        }
        ggml_set_name(op, "op");
        ggml_tensor * ref = chain(ctx, a, ids, b, bi);
        ggml_set_name(ref, "chain");

        ggml_tensor * out = ggml_concat(ctx, op, ref, 1);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::mt19937 rng(0x2545f491u ^ (uint32_t) (K*31 + M*131 + N*17 + n_b*7 + n_head*1009 + rows*3 + bias*5));
        std::normal_distribution<float>       nrm(0.0f, 1.0f);
        std::uniform_real_distribution<float> u11(-1.0f, 1.0f);

        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            const char * name = ggml_get_name(t);
            if (strcmp(name, "rows") == 0) {
                // a near-identity remap with 1 % of the blocks moved (the stable slots after a restore), per stream
                const int64_t rps = M + M/8 + 1;
                std::vector<int32_t> v(M*n_b);
                for (int64_t s = 0; s < n_b; s++) {
                    for (int64_t m = 0; m < M; m++) {
                        const int64_t r = (rng() % 100 == 0) ? (int64_t) (rng() % (uint32_t) rps) : m;
                        v[s*M + m] = (int32_t) (s*rps + r);
                    }
                }
                ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(int32_t));
            } else if (strcmp(name, "allrows") == 0) {
                std::vector<int32_t> v(ggml_nelements(t));
                for (size_t i = 0; i < v.size(); i++) {
                    v[i] = (int32_t) i;
                }
                ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(int32_t));
            } else if (strcmp(name, "a16") == 0) {
                // every row is overwritten by the set_rows; a recognisable junk value elsewhere
                std::vector<ggml_fp16_t> v(ggml_nelements(t), ggml_fp32_to_fp16(-7.0f));
                ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(ggml_fp16_t));
            } else if (strcmp(name, "a") == 0 || strcmp(name, "b") == 0) {
                // unit-norm rows (the normed, roped keys and queries), so the scores straddle 0 and the ReLU matters
                const int64_t n = ggml_nelements(t);
                std::vector<float> v(n);
                for (int64_t r = 0; r < n/K; r++) {
                    double ss = 0.0;
                    for (int64_t k = 0; k < K; k++) {
                        v[r*K + k] = nrm(rng);
                        ss += (double) v[r*K + k]*v[r*K + k];
                    }
                    const float sc = ss > 0.0 ? (float) (1.0/std::sqrt(ss)) : 1.0f;
                    const float sc2 = strcmp(name, "a") == 0 ? sc*scale : sc;
                    for (int64_t k = 0; k < K; k++) {
                        v[r*K + k] *= sc2;
                    }
                }
                ggml_backend_tensor_set(t, v.data(), 0, n*sizeof(float));
            } else if (strcmp(name, "bias") == 0) {
                const int64_t n = ggml_nelements(t);
                std::vector<float> v(n);
                for (auto & x : v) {
                    if (bias == 1) {
                        const uint32_t c = rng() % 8;
                        x = c == 0 ? -INFINITY : (c == 1 ? 1e9f : 0.0f);
                    } else {
                        x = u11(rng);
                    }
                }
                ggml_backend_tensor_set(t, v.data(), 0, n*sizeof(float));
            } else if (t->type == GGML_TYPE_F32 && !ggml_is_view_op(t->op)) {
                init_tensor_uniform(t);
            }
        }
    }
};

// [TAG_QSA_COMPACT_BIAS] GGML_OP_QSA_BIAS: the qwen4exp per-block indexer bias built from the block table
// blk [2, n_blocks, n_stream] (key, vis) and the query table qry [2, n_tps, n_stream] (tail, bit), read
// through rows [np, n_stream] (NULL = identity). The op writes one of three constants (+0.0f, 1e9f, -inf)
// and nothing else, so the backends must agree BITWISE: err() counts the elements whose bit patterns
// differ (-inf == -inf, +0.0 != -0.0), and the bar is 0.
//   rows_mode 0 = NULL, 1 = the arange slice t0..t0+np (a [TAG_QSA_CHUNK] chunk), 2 = the PAD chunk (arange
//   clamped to n_tps - 1), 3 = random rows with repeats
struct test_qsa_bias : public test_case {
    const int64_t n_blocks;
    const int64_t n_tps;
    const int64_t np;
    const int64_t n_stream;
    const int     rows_mode;

    std::string vars() override {
        return VARS_TO_STR5(n_blocks, n_tps, np, n_stream, rows_mode);
    }

    test_qsa_bias(int64_t n_blocks = 512, int64_t n_tps = 7, int64_t np = 7, int64_t n_stream = 1, int rows_mode = 0)
        : n_blocks(n_blocks), n_tps(n_tps), np(np), n_stream(n_stream), rows_mode(rows_mode) {
        GGML_ASSERT(rows_mode != 0 || np == n_tps);
        GGML_ASSERT(rows_mode != 1 || np <= n_tps);
    }

    double max_err() override { return 0.0; }
    bool run_whole_graph() override { return true; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * blk = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 2, n_blocks, n_stream);
        ggml_set_name(blk, "qsa_blk");
        ggml_tensor * qry = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 2, n_tps, n_stream);
        ggml_set_name(qry, "qsa_qry");
        ggml_tensor * rows = nullptr;
        if (rows_mode != 0) {
            rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, np, n_stream);
            ggml_set_name(rows, "qsa_rows");
        }

        ggml_tensor * out = ggml_qsa_bias(ctx, blk, qry, rows);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::mt19937 rng(1234 + (uint32_t) (n_blocks*31 + n_tps*7 + np*3 + n_stream + rows_mode));
        // keys and tails are multiples of 4 (r = 4) in [0, 2^24), drawn near one base so both sides of
        // key >= tail occur in every row
        const int32_t base = (int32_t) (rng() % ((1u << 24) - (1u << 16))) & ~3;
        auto near = [&]() { return base + (int32_t) (rng() % 4096)*4; };

        int seen = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE) {
                continue;
            }
            const char * name = ggml_get_name(t);
            std::vector<int32_t> data(ggml_nelements(t));
            if (strcmp(name, "qsa_blk") == 0) {
                seen++;
                for (int64_t e = 0; e < n_blocks*n_stream; ++e) {
                    const uint32_t u = rng() % 100;
                    // ~5 % the spare/dead ids (INT32_MAX), with vis ~0 (the boosted spare) or 0 (dead)
                    const bool spare = u < 5;
                    uint32_t vis = rng() & 31u;                  // 5-bit sequence masks, 0 included
                    if (u >= 95) vis = 0xFFFFFFFFu;              // every sequence (incl. bit 31)
                    if (spare) vis = (u & 1) ? 0xFFFFFFFFu : 0u;
                    data[2*e + 0] = spare ? INT32_MAX : near();
                    data[2*e + 1] = (int32_t) vis;
                }
            } else if (strcmp(name, "qsa_qry") == 0) {
                seen++;
                for (int64_t e = 0; e < n_tps*n_stream; ++e) {
                    data[2*e + 0] = near();
                    data[2*e + 1] = (int32_t) (e == 0 ? 0x80000000u : 1u << (rng() % 5));   // one row on bit 31
                }
            } else if (strcmp(name, "qsa_rows") == 0) {
                seen++;
                for (int64_t s = 0; s < n_stream; ++s) {
                    const int64_t t0 = rows_mode == 1 ? (n_tps - np)/2 : rows_mode == 2 ? std::max<int64_t>(0, n_tps - np/2) : 0;
                    for (int64_t t = 0; t < np; ++t) {
                        int64_t q = 0;
                        switch (rows_mode) {
                            case 1: q = t0 + t; break;
                            case 2: q = std::min<int64_t>(t0 + t, n_tps - 1); break;
                            default: q = (int64_t) (rng() % (uint32_t) n_tps); break;
                        }
                        data[s*np + t] = (int32_t) q;
                    }
                }
            } else {
                init_tensor_uniform(t);
                continue;
            }
            ggml_backend_tensor_set(t, data.data(), 0, data.size()*sizeof(int32_t));
        }
        // found by name: a rename would silently fall through to the uniform fill
        GGML_ASSERT(seen == (rows_mode != 0 ? 3 : 2));
    }

    // bitwise: the number of elements whose f32 bit patterns differ
    double err(const float * a, const float * b, size_t n) override {
        size_t bad = 0;
        for (size_t i = 0; i < n; ++i) {
            bad += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        }
        return (double) bad;
    }
};

enum MoeGatingFunc {
    GATING_FUNC_SOFTMAX,
    GATING_FUNC_SIGMOID,
    GATING_FUNC_SOFTMAX_WEIGHT,
    GATING_FUNC_SQRT_SOFTPLUS,
};

struct test_topk_moe : public test_case {
    const std::array<int64_t, 4> ne;
    const int n_expert_used;
    const bool with_norm;
    const bool bias_probs;
    const MoeGatingFunc gating_func;
    const float scale_w;
    ggml_tensor * weights {};
    ggml_tensor * selected_experts {};

    test_topk_moe(std::array<int64_t, 4> ne              = { 10, 5, 1, 1 },
                  int                    n_expert_used   = 1,
                  bool                   with_norm       = false,
                  bool                   bias_probs      = false,
                  MoeGatingFunc          gating_func     = GATING_FUNC_SOFTMAX,
                  float                  scale_w         = 0.0f) :
        ne(ne),
        n_expert_used(n_expert_used),
        with_norm(with_norm),
        bias_probs(bias_probs),
        gating_func(gating_func),
        scale_w(scale_w) {
        GGML_ASSERT(n_expert_used <= ne[0]);
    }

    std::string vars() override { return VARS_TO_STR6(ne, n_expert_used, with_norm, bias_probs, gating_func, scale_w); }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "TOPK_MOE";
    }

    bool run_whole_graph() override { return true; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int n_expert = ne[0];
        const int n_tokens = ne[1];

        ggml_tensor * logits = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data());
        ggml_tensor * probs            =
            (gating_func == GATING_FUNC_SOFTMAX) ? ggml_soft_max(ctx, logits) :
            (gating_func == GATING_FUNC_SIGMOID) ? ggml_sigmoid(ctx, logits) :
            (gating_func == GATING_FUNC_SQRT_SOFTPLUS) ? ggml_sqrt(ctx, ggml_softplus(ctx, logits)) : logits;
        ggml_set_name(probs, "probs");

        ggml_tensor * selection_probs = probs;
        if (bias_probs) {
            ggml_tensor * exp_probs_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
            ggml_set_name(exp_probs_b, "exp_probs_b");
            selection_probs = ggml_add(ctx, probs, exp_probs_b);
            ggml_set_name(selection_probs, "selection_probs");
        }

        selected_experts = ggml_argsort_top_k(ctx, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        ggml_set_name(selected_experts, "selected_experts");

        weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
        ggml_set_name(weights, "weights");

        if (gating_func == GATING_FUNC_SOFTMAX_WEIGHT) {
            weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
            weights = ggml_soft_max(ctx, weights);  // [n_expert_used, n_tokens]
            weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
        }

        if (with_norm) {
            weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
            ggml_tensor * weights_sum = ggml_sum_rows(ctx, weights); // [1, n_tokens]
            ggml_set_name(weights_sum, "weights_sum");

            weights_sum = ggml_clamp(ctx, weights_sum, 6.103515625e-5, INFINITY);
            weights = ggml_div(ctx, weights, weights_sum); // [n_expert_used, n_tokens]
            weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
        }

        if (scale_w) {
            weights = ggml_scale(ctx, weights, scale_w);
        }

        ggml_set_name(weights, "weights");
        return weights;
    }
    // Verify two outputs
    std::vector<ggml_tensor *> fusion_test_nodes() override { return { selected_experts, weights }; }

    // allow output in arbitrary order
    double err(const float * a, const float * b, size_t n) override {
        std::vector<float> a2(n);
        std::vector<float> b2(n);
        for (size_t i = 0; i < n; ++i) {
            a2[i] = a[i];
            b2[i] = b[i];
        }
        std::sort(a2.begin(), a2.end());
        std::sort(b2.begin(), b2.end());
        return nmse(a2.data(), b2.data(), n);
    }
};

struct test_mul_mat_vec_fusion : public test_case {
    const ggml_type type;
    const ggml_glu_op glu_op;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const bool use_id;
    const int n_mats;
    const int n_used;
    const bool b;        // broadcast b matrix (only for use_id)
    const bool with_bias;
    const bool with_gate;
    const bool with_lane_scale;
    std::array<int64_t, 2> batch_dims;

    test_mul_mat_vec_fusion(ggml_type type, ggml_glu_op op, int64_t m, int64_t n, int64_t k,
                        bool use_id = false, int n_mats = 1, int n_used = 1, bool b = false, bool with_bias = false, bool with_gate = true,
                        bool with_lane_scale = false, std::array<int64_t, 2> batch_dims = {4, 2})
    : type(type), glu_op(op), m(m), n(n), k(k), use_id(use_id), n_mats(n_mats), n_used(n_used), b(b), with_bias(with_bias),
        with_gate(with_gate), with_lane_scale(with_lane_scale), batch_dims(batch_dims) {
        if (use_id) {
            GGML_ASSERT(n_used <= n_mats);
        }
    }

    std::string vars() override {
        return VARS_TO_STR13(type, glu_op, m, n, k, use_id, n_mats, n_used, b, with_bias, with_gate, with_lane_scale, batch_dims);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT_VEC_FUSION";
    }

    bool run_whole_graph() override { return true; }
    bool use_weight_context() override { return use_id && with_lane_scale; }

    ggml_tensor * build_gate(ggml_context * ctx, ggml_tensor * ffn_gate, ggml_tensor * ffn_up) {
        ggml_tensor * out = nullptr;
        if (with_gate) {
            if (glu_op == GGML_GLU_OP_SWIGLU_OAI) {
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                out = ggml_swiglu_oai(ctx, ffn_gate, ffn_up, alpha, limit);
            } else {
                out = ggml_glu_split(ctx, ffn_gate, ffn_up, glu_op);
            }
        }
        return out;
    }

    ggml_tensor * build_lane_scale_dense(ggml_context * ctx, ggml_tensor * out) {
        ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        return ggml_mul(ctx, out, scale);
    }

    ggml_tensor * build_lane_scale_id(ggml_context * ctx, ggml_context * ctx_weights, ggml_tensor * out, ggml_tensor * ids) {
        GGML_ASSERT(ctx_weights);
        ggml_tensor * scale = ggml_new_tensor_1d(ctx_weights, GGML_TYPE_F32, n_mats);
        ggml_tensor * s = ggml_reshape_3d(ctx, scale, 1, n_mats, 1);
        s = ggml_repeat_4d(ctx, s, 1, n_mats, m, 1);
        s = ggml_get_rows(ctx, s, ids);
        return ggml_mul(ctx, out, s);
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        GGML_ASSERT(!use_weight_context());
        return build_graph(ctx, nullptr);
    }

    ggml_tensor * build_graph(ggml_context * ctx, ggml_context * ctx_weights) override {
        if (!use_id) {
            const int              channels = batch_dims[0];
            const int              samples  = batch_dims[1];
            std::array<int64_t, 4> ne       = { k, m, channels, samples };
            std::array<int64_t, 4> ne0      = { k, n, channels, samples };

            ggml_tensor * cur  = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne.data());
            ggml_tensor * gate = with_gate ? ggml_new_tensor(ctx, type, 4, ne0.data()) : nullptr;
            ggml_tensor * up   = ggml_new_tensor(ctx, type, 4, ne0.data());

            auto build_lane_up = [&]() {
                ggml_tensor * ffn_up = ggml_mul_mat(ctx, up, cur);
                if (with_lane_scale) {
                    ffn_up = build_lane_scale_dense(ctx, ffn_up);
                }
                if (with_bias) {
                    std::array<int64_t, 4> bias_ne = { ffn_up->ne[0], 1, channels, samples };
                    ggml_tensor * up_bias = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, bias_ne.data());
                    ffn_up = ggml_add(ctx, ffn_up, up_bias);
                }
                return ffn_up;
            };

            auto build_lane_gate = [&]() {
                ggml_tensor * ffn_gate = ggml_mul_mat(ctx, gate, cur);
                if (with_lane_scale) {
                    ffn_gate = build_lane_scale_dense(ctx, ffn_gate);
                }
                if (with_bias) {
                    std::array<int64_t, 4> bias_ne   = { ffn_gate->ne[0], 1, channels, samples };
                    ggml_tensor * gate_bias = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, bias_ne.data());
                    ffn_gate = ggml_add(ctx, ffn_gate, gate_bias);
                }
                return ffn_gate;
            };

            ggml_tensor * ffn_up = build_lane_up();
            ggml_tensor * ffn_gate = with_gate ? build_lane_gate() : nullptr;

            ggml_tensor * out = with_gate ? build_gate(ctx, ffn_gate, ffn_up) : ffn_up;

            std::array<int64_t, 4> bias2_ne   = { out->ne[0], 1, channels, samples };
            ggml_tensor * bias2 = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, bias2_ne.data());
            out = ggml_add(ctx, out, bias2);

            ggml_set_name(out, "out");
            return out;
        } else {
            ggml_tensor * gates = ggml_new_tensor_3d(ctx, type, k, n, n_mats);
            ggml_tensor * ups   = ggml_new_tensor_3d(ctx, type, k, n, n_mats);
            ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, m);

            if (n_used != n_mats) {
                ids = ggml_view_2d(ctx, ids, n_used, m, ids->nb[1], 0);
            }

            ggml_tensor * cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, this->b ? 1 : n_used, m);
            ggml_set_name(cur, "cur");

            auto build_lane_up = [&]() {
                ggml_tensor * ffn_up = ggml_mul_mat_id(ctx, ups, cur, ids);
                if (with_lane_scale) {
                    ffn_up = build_lane_scale_id(ctx, ctx_weights, ffn_up, ids);
                }
                if (with_bias) {
                    ggml_tensor * up_bias_param = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_up->ne[0], n_mats);
                    ffn_up = ggml_add_id(ctx, ffn_up, up_bias_param, ids);
                }
                return ffn_up;
            };

            auto build_lane_gate = [&]() {
                ggml_tensor * ffn_gate = ggml_mul_mat_id(ctx, gates, cur, ids);
                if (with_lane_scale) {
                    ffn_gate = build_lane_scale_id(ctx, ctx_weights, ffn_gate, ids);
                }
                if (with_bias) {
                    ggml_tensor * gate_bias_param = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_gate->ne[0], n_mats);
                    ffn_gate = ggml_add_id(ctx, ffn_gate, gate_bias_param, ids);
                }
                return ffn_gate;
            };

            ggml_tensor * ffn_up = build_lane_up();
            ggml_tensor * ffn_gate = with_gate ? build_lane_gate() : nullptr;

            ggml_tensor * out = with_gate ? build_gate(ctx, ffn_gate, ffn_up) : ffn_up;

            std::array<int64_t, 4> scale_ne { 1, out->ne[1], out->ne[2], out->ne[3] };
            ggml_tensor * scale = ggml_new_tensor(ctx, out->type, 4, scale_ne.data());
            out = ggml_mul(ctx, out, scale);

            ggml_set_name(out, "out");
            return out;
        }
    }

    void initialize_tensors(ggml_context * ctx) override {
        if (!use_id) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                init_tensor_uniform(t);
            }
        } else {
            init_mul_mat_id_tensors(ctx, n_mats);
        }
    }

    double max_nmse_err() override {
        return 5e-3;
    }
};

// GGML_OP_SUM
struct test_sum : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool _use_permute;

    std::string vars() override {
        std::string v = VARS_TO_STR2(type, ne);
        if (_use_permute) v += "," + VAR_TO_STR(permute);
        return v;
    }

    test_sum(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            std::array<int64_t, 4> permute = {0, 0, 0, 0})
        : type(type), ne(ne), permute(permute),
            _use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        if (_use_permute) {
            a = ggml_permute(ctx, a, permute[0], permute[1], permute[2], permute[3]);
            ggml_set_name(a, "a_permuted");
        }

        ggml_tensor * out = ggml_sum(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * sqrtf(ne[0]*ne[1]*ne[2]*ne[3]);
    }

    // Don't center the distribution around zero. Helps to avoid catastrophic cancellation.
    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -0.9f, 1.1f);
        }
    }
};

// GGML_OP_SUM_ROWS
struct test_sum_rows : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool permute;
    const bool slice;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, permute, slice);
    }

    test_sum_rows(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3},
            bool permute = false, bool slice = false)
        : type(type), ne(ne), permute(permute), slice(slice) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        if (slice) {
            a = ggml_view_4d(ctx, a,
                             ne[0], ne[1], ne[2] / 2, ne[3] - 1,
                             a->nb[1], a->nb[2] * 2, a->nb[3], /*offset=*/a->nb[3]);
        }
        if (permute) {
            a = ggml_permute(ctx, a, 0, 2, 3, 1);
        }

        ggml_tensor * out = ggml_sum_rows(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_MEAN
struct test_mean : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_mean(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_mean(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }

    float grad_eps() override {
        return 0.1f * ne[0]*ne[1]*ne[2]*ne[3];
    }

    // Don't center the distribution around zero. Helps to avoid catastrophic cancellation.
    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -0.9f, 1.1f);
        }
    }
};

// GGML_OP_UPSCALE
struct test_upscale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t scale_factor;
    const bool transpose;
    const ggml_scale_mode mode;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, scale_factor, mode, transpose);
    }

    test_upscale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {512, 512, 3, 1},
            int32_t scale_factor = 2, ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST, bool transpose = false)
        : type(type), ne(ne), scale_factor(scale_factor), transpose(transpose), mode(mode) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        if (transpose) {
            a = ggml_transpose(ctx, a);
            ggml_set_name(a, "a_transposed");
        }

        ggml_tensor * out = ggml_upscale(ctx, a, scale_factor, mode);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_UPSCALE (via ggml_interpolate)
struct test_interpolate : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> ne_tgt;
    const ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, ne_tgt, mode);
    }

    test_interpolate(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne     = {2, 5,  7, 11},
            std::array<int64_t, 4> ne_tgt = {5, 7, 11, 13},
            ggml_scale_mode mode = GGML_SCALE_MODE_NEAREST)
        : type(type), ne(ne), ne_tgt(ne_tgt), mode(mode) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_interpolate(ctx, a, ne_tgt[0], ne_tgt[1],ne_tgt[2], ne_tgt[3], mode);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_GROUP_NORM
struct test_group_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t num_groups;
    const float eps;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, num_groups, eps);
    }

    test_group_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 64, 320, 1},
            int32_t num_groups = 32,
            float eps = 1e-6f)
        : type(type), ne(ne), num_groups(num_groups), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_group_norm(ctx, a, num_groups, eps);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_GROUP_NORM + GGML_OP_MUL + GGML_OP_ADD
struct test_group_norm_mul_add : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    int num_groups;
    float eps;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "GROUP_NORM_MUL_ADD";
    }

    bool run_whole_graph() override { return true; }

    std::string vars() override {
        return VARS_TO_STR4(type, ne, num_groups, eps);
    }

    test_group_norm_mul_add(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {128, 1, 1, 1},
            int num_groups = 4,
            float eps = 1e-5f)
        : type(type), ne(ne), num_groups(num_groups), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * w = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(a); ggml_set_param(w); ggml_set_param(b);
        ggml_set_name(a, "a"); ggml_set_name(w, "w"); ggml_set_name(b, "b");
        ggml_tensor * n = ggml_group_norm(ctx, a, num_groups, eps);
        ggml_tensor * m = ggml_mul(ctx, n, w);
        ggml_tensor * out = ggml_add(ctx, m, b);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_L2_NORM
struct test_l2_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;
    bool v;
    bool noncontig_rows;

    std::string vars() override {
        return VARS_TO_STR5(type, ne, eps, v, noncontig_rows);
    }

    test_l2_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 64, 320, 1},
            float eps = 1e-12f,
            bool v = false,
            bool noncontig_rows = false)
        : type(type), ne(ne), eps(eps), v(v), noncontig_rows(noncontig_rows) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const std::array<int64_t, 4> ne_a = noncontig_rows ?
            std::array<int64_t, 4>{ ne[1], ne[0], ne[2], ne[3] } : ne;
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        if (noncontig_rows) {
            a = ggml_permute(ctx, a, 1, 0, 2, 3);
            ggml_set_name(a, "permuted a");
        }
        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0]/2, a->ne[1]/2, a->ne[2]/2, a->ne[3]/2, a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }

        ggml_tensor * out = ggml_l2_norm(ctx, a, eps);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ACC
struct test_acc : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;
    const int64_t stride_dim;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, ne_b, stride_dim);
    }

    test_acc(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {256, 17, 2, 3},
            std::array<int64_t, 4> ne_b = {256, 16, 2, 3},
            uint64_t stride_dim = -1)
        : type(type), ne_a(ne_a), ne_b(ne_b), stride_dim(stride_dim) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * b;
        if (stride_dim == 1 || stride_dim == 2 || stride_dim == 3) {
            // Create a larger tensor and take a view at a non-zero offset.
            // This tests that the backend correctly handles b's data offset
            std::array<int64_t, 4> ne_b_pad = {ne_b[0], ne_b[1], ne_b[2], ne_b[3]};
            ne_b_pad[stride_dim] += 1;
            ggml_tensor * b_pad = ggml_new_tensor(ctx, type, 4, ne_b_pad.data());
            ggml_set_param(b_pad);
            ggml_set_name(b_pad, "b_pad");
            // View that skips the first row, so b has a non-zero byte offset
            b = ggml_view_4d(ctx, b_pad,
                ne_b[0], ne_b[1], ne_b[2], ne_b[3],
                b_pad->nb[1], b_pad->nb[2], b_pad->nb[3],
                b_pad->nb[1]);
        } else {
            b = ggml_new_tensor(ctx, type, 4, ne_b.data());
            ggml_set_param(b);
        }
        ggml_set_name(b, "b");

        // When ne_b[0] < ne_a[0], a->nb[1] != b->nb[1], so the stride
        // parameters to ggml_acc don't match b's natural stride.
        ggml_tensor * out = ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], 0);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_PAD
struct test_pad : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int pad_0;
    const int pad_1;
    const bool circular;

    std::string vars() override {
        return VARS_TO_STR5(type, ne_a, pad_0, pad_1, circular);
    }

    test_pad(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {512, 512, 1, 1},
            int pad_0 = 1, int pad_1 = 1, bool circular = false)
        : type(type), ne_a(ne_a), pad_0(pad_0), pad_1(pad_1), circular(circular) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = circular
            ? ggml_pad_circular(ctx, a, pad_0, pad_1, 0, 0)
            : ggml_pad(ctx, a, pad_0, pad_1, 0, 0);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_PAD (with extension)
struct test_pad_ext : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int lp0;
    const int rp0;
    const int lp1;
    const int rp1;
    const int lp2;
    const int rp2;
    const int lp3;
    const int rp3;
    const int tfrm; // 0 - none, 1 - non-cont, 2 - perm
    const bool circular;

    std::string vars() override {
        return VARS_TO_STR12(type, ne_a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, tfrm, circular);
    }

    test_pad_ext(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {512, 512, 3, 1},
            int lp0 = 1, int rp0 = 1, int lp1 = 1, int rp1 = 1,
            int lp2 = 1, int rp2 = 1, int lp3 = 1, int rp3 = 1,
            int tfrm = 0, bool circular = false)
        : type(type), ne_a(ne_a), lp0(lp0), rp0(rp0), lp1(lp1), rp1(rp1), lp2(lp2), rp2(rp2), lp3(lp3), rp3(rp3),
          tfrm(tfrm), circular(circular) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        if (tfrm == 1) {
            a = ggml_view_4d(ctx, a, (a->ne[0] + 1) / 2, (a->ne[1] + 1) / 2, (a->ne[2] + 1) / 2, (a->ne[3] + 1) / 2, a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view of a");
        } else if (tfrm == 2) {
            a = ggml_permute(ctx, a, 2, 1, 0, 3);
            ggml_set_name(a, "permuted a");
        }

        ggml_tensor * out = circular
            ? ggml_pad_ext_circular(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
            : ggml_pad_ext         (ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_PAD_REFLECT_1D
struct test_pad_reflect_1d : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int pad_0;
    const int pad_1;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, pad_0, pad_1);
    }

    test_pad_reflect_1d(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {512, 34, 2, 1},
            int pad_0 = 10, int pad_1 = 9)
        : type(type), ne_a(ne_a), pad_0(pad_0), pad_1(pad_1)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 2, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_pad_reflect_1d(ctx, a, pad_0, pad_1);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ROLL
struct test_roll : public test_case {
    const int shift0;
    const int shift1;
    const int shift3;
    const int shift4;
    const bool permute;

    std::string vars() override {
        return VARS_TO_STR5(shift0, shift1, shift3, shift4, permute);
    }

    test_roll(int shift0 = 3, int shift1 = -2, int shift3 = 1, int shift4 = -1, bool permute = false)
        : shift0(shift0), shift1(shift1), shift3(shift3), shift4(shift4), permute(permute) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        int64_t ne[4] = {10, 5, 4, 3};
        ggml_tensor * a = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
        ggml_set_name(a, "a");

        if (permute) {
            // ggml_roll only requires nb[0] == type size, so a permuted src is valid
            a = ggml_permute(ctx, a, 0, 2, 1, 3);
            ggml_set_name(a, "a_permuted");
        }

        ggml_tensor * out = ggml_roll(ctx, a, shift0, shift1, shift3, shift4);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_ARANGE
struct test_arange : public test_case {
    const ggml_type type;
    const float start;
    const float stop;
    const float step;

    std::string vars() override {
        return VARS_TO_STR4(type, start, stop, step);
    }

    test_arange(ggml_type type = GGML_TYPE_F32,
            float start = 0.f, float stop = 10.f, float step = 1.f)
        : type(type), start(start), stop(stop), step(step)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * out = ggml_arange(ctx, start, stop, step);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_TIMESTEP_EMBEDDING
struct test_timestep_embedding : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int dim;
    const int max_period;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, dim, max_period);
    }

    test_timestep_embedding(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {2, 1, 1, 1},
            int dim = 320, int max_period=10000)
        : type(type), ne_a(ne_a), dim(dim), max_period(max_period)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_timestep_embedding(ctx, a, dim, max_period);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_LEAKY_RELU
struct test_leaky_relu : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const float negative_slope;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, negative_slope);
    }

    test_leaky_relu(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 5, 4, 3},
            float negative_slope = 0.1f)
        : type(type), ne_a(ne_a), negative_slope(negative_slope)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_leaky_relu(ctx, a, negative_slope, true);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_FLASH_ATTN_EXT
struct test_flash_attn_ext : public test_case {
    const int64_t hsk; // K head size
    const int64_t hsv; // V head size
    const int64_t nh; // num heads
    const std::array<int64_t, 2> nr23; // repeat in dim 2 and 3, tests for grouped-query attention
    const int64_t kv; // kv size
    const int64_t nb; // batch size

    const bool mask; // use mask
    const bool sinks; // use sinks

    const float max_bias; // ALiBi
    const float logit_softcap; // Gemma 2

    const ggml_prec prec;
    const ggml_type type_K;
    const ggml_type type_V;
    std::array<int32_t, 4> permute;
    const bool kv_view; // create K/V as views of a larger buffer (like a KV cache)
    const bool v_is_view_of_k;
    // rows of the mask, when it is padded past the batch. llama.cpp always pads the KQ mask to
    // GGML_KQ_MASK_PAD (64) rows, so at decode/verify batch sizes nem1 is 64 and not nb - which
    // is what enables the Vulkan mask-opt summary path (nem1 >= 32). 0 means "exactly nb".
    const int64_t mask_rows;

    std::string vars() override {
        return VARS_TO_STR17(hsk, hsv, nh, nr23, kv, nb, mask, sinks, max_bias, logit_softcap, prec, type_K, type_V, permute, kv_view, v_is_view_of_k, mask_rows);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        // Just counting matmul costs:
        // Q*K^T is nb x hsk x kv, P*V is nb x kv x hsv, per head
        return (2 * nh*nr23[0] * nb * (hsk + hsv) * kv)*nr23[1];
    }

    test_flash_attn_ext(int64_t hsk = 128, int64_t hsv = 128, int64_t nh = 32, std::array<int64_t, 2> nr23 = {1, 1}, int64_t kv = 96, int64_t nb = 8,
                        bool mask = true, bool sinks = false, float max_bias = 0.0f, float logit_softcap = 0.0f, ggml_prec prec = GGML_PREC_F32,
                        ggml_type type_K = GGML_TYPE_F16, ggml_type type_V = GGML_TYPE_F16, std::array<int32_t, 4> permute = {0, 1, 2, 3},
                        bool kv_view = true, bool v_is_view_of_k = false, int64_t mask_rows = 0)
        : hsk(hsk), hsv(hsv), nh(nh), nr23(nr23), kv(kv), nb(nb), mask(mask), sinks(sinks), max_bias(max_bias), logit_softcap(logit_softcap), prec(prec),
          type_K(type_K), type_V(type_V), permute(permute), kv_view(kv_view), v_is_view_of_k(v_is_view_of_k), mask_rows(mask_rows) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t hsk_padded = GGML_PAD(hsk, ggml_blck_size(type_K));
        const int64_t hsv_padded = GGML_PAD(hsv, ggml_blck_size(type_V));

        auto const &create_permuted = [&](ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, bool is_view) -> ggml_tensor * {
            int64_t ne[4] = {ne0, ne1, ne2, ne3};
            int64_t ne_perm[4];
            for (int i = 0; i < 4; ++i) {
                ne_perm[permute[i]] = ne[i];
            }
            ggml_tensor * t;
            if (is_view) {
                ggml_tensor * t0 = ggml_new_tensor_4d(ctx, type, ne_perm[0], 2*ne_perm[1], ne_perm[2], ne_perm[3]);
                t = ggml_view_4d(ctx, t0, ne_perm[0], ne_perm[1], ne_perm[2], ne_perm[3], t0->nb[1], t0->nb[2], t0->nb[3], 0);
            } else {
                t = ggml_new_tensor_4d(ctx, type, ne_perm[0], ne_perm[1], ne_perm[2], ne_perm[3]);
            }
            if (permute != std::array<int32_t, 4>{0, 1, 2, 3}) {
                t = ggml_permute(ctx, t, permute[0], permute[1], permute[2], permute[3]);
            }
            return t;
        };

        ggml_tensor * q = create_permuted(GGML_TYPE_F32, hsk_padded, nb, nh*nr23[0], nr23[1], false);
        ggml_set_name(q, "q");

        ggml_tensor * k = create_permuted(type_K,        hsk_padded, kv, nh,         nr23[1], kv_view); // the K tensor is usually a view of the K cache
        ggml_set_name(k, "k");

        ggml_tensor * v = nullptr;
        if (v_is_view_of_k) {
            // the V cache is a sub-view of the K cache. this is used by some MLA-based models
            // for more info:
            //   - https://github.com/ggml-org/llama.cpp/pull/13435
            //   - https://github.com/ggml-org/llama.cpp/pull/18953#issuecomment-3774948392
            //   - https://github.com/ggml-org/llama.cpp/pull/18986
            GGML_ASSERT(type_K == type_V && hsv_padded <= hsk_padded);

            v = ggml_view_4d(ctx, k, hsv_padded, kv, nh, nr23[1], k->nb[1], k->nb[2], k->nb[3], 0);
        } else {
            v = create_permuted(type_V,        hsv_padded, kv, nh,         nr23[1], kv_view); // the V tensor is usually a view of the V cache
        }
        ggml_set_name(v, "v");

        ggml_tensor * m = nullptr;
        if (mask) {
            m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, std::max(nb, mask_rows), 1, nr23[1]);
            ggml_set_name(m, "m");
        }

        ggml_tensor * s = nullptr;
        if (sinks) {
            s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q->ne[2]);
            ggml_set_name(s, "s");
        }

        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(hsk), max_bias, logit_softcap);
        ggml_flash_attn_ext_add_sinks(out, s);
        ggml_flash_attn_ext_set_prec (out, prec);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "s") == 0) {
                // make the sink values more noticeable in order to trigger a test failure when the implementation is wrong
                init_tensor_uniform(t, -10.0f, 10.0f);
            } else if (strcmp(t->name, "m") == 0) {
                init_tensor_kq_mask(t);
            } else {
                init_tensor_uniform(t);
            }
        }
    }

    bool grad_precise() override {
        return true;
    }
};

// [TAG_KQ_VIS] visibility metadata for GGML_OP_KQ_MASK / ggml_flash_attn_ext_vis (see ggml.h):
// cells [n_kv, 4, n_stream] I32 planes {pos, seq bitmask, x, y}, toks [4, n_tps, n_stream] I32
// {pos, 1 << seq, x, y}. A layout like a unified or per-stream KV cache: up to 5 sequences per stream,
// holes, seqless used cells, shared-prefix (multi-bit) cells, and - so no query row is fully masked,
// which would make the FA reference and the backend disagree on 0/0 - every token's own cell.
// rope2d: tokens of one sequence share a position in runs of 4 (an image row) with distinct (x, y);
// otherwise x = y = INT32_MAX (no 2-D positions).
// [TAG_QSA_VIS] the generator behind init_kq_vis (the same draws, so its cases are unchanged); own (if not NULL) receives
// each token's own cell index, [s*n_tps + i]
static void gen_kq_vis(int64_t n_kv, int64_t n_tps, int64_t n_stream, bool rope2d, uint32_t seed,
                       std::vector<int32_t> & c, std::vector<int32_t> & t, std::vector<int64_t> * own) {
    GGML_ASSERT(n_kv >= n_tps);

    std::mt19937 rng(seed);
    auto rnd = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };

    c.assign((size_t) (4*n_kv*n_stream), 0);
    t.assign((size_t) (4*n_tps*n_stream), 0);
    if (own) {
        own->assign((size_t) (n_tps*n_stream), 0);
    }

    for (int64_t s = 0; s < n_stream; ++s) {
        int32_t * c_pos = c.data() + 4*n_kv*s;
        int32_t * c_seq = c_pos + n_kv;
        int32_t * c_x   = c_pos + 2*n_kv;
        int32_t * c_y   = c_pos + 3*n_kv;

        const int n_seq = rnd(1, 5);
        const int32_t depth = (int32_t) n_kv; // positions of the cached cells
        std::vector<int32_t> next(n_seq);
        for (int q = 0; q < n_seq; ++q) {
            next[q] = rnd(depth/2, depth);
        }

        // cached cells
        for (int64_t j = 0; j < n_kv; ++j) {
            const int r = rnd(0, 99);
            if (r < 10) { // hole
                c_pos[j] = -1; c_seq[j] = 0; c_x[j] = 0; c_y[j] = 0;
                continue;
            }
            c_pos[j] = rnd(0, depth + (int32_t) n_tps);
            c_x[j]   = rnd(0, 3);
            c_y[j]   = rnd(0, 3);
            uint32_t m = r < 13 ? 0u : (1u << rnd(0, n_seq - 1)); // 3 % used without a sequence
            if (r >= 13 && r < 30) {
                m |= 1u << rnd(0, n_seq - 1); // shared prefix
            }
            c_seq[j] = (int32_t) m;
        }

        // tokens, each with its own cell at a distinct index
        std::vector<int64_t> idx((size_t) n_kv);
        for (int64_t j = 0; j < n_kv; ++j) idx[j] = j;
        std::shuffle(idx.begin(), idx.end(), rng);
        for (int64_t i = 0; i < n_tps; ++i) {
            const int q = rnd(0, n_seq - 1);
            int32_t * tk = t.data() + 4*(s*n_tps + i);
            if (rope2d) {
                tk[0] = next[q] + (int32_t) (i/4);
                tk[2] = (int32_t) (i % 4);
                tk[3] = (int32_t) (i % 3);
            } else {
                tk[0] = next[q]++;
                tk[2] = INT32_MAX;
                tk[3] = INT32_MAX;
            }
            tk[1] = (int32_t) (1u << q);

            const int64_t j = idx[i];
            c_pos[j] = tk[0];
            c_seq[j] = (int32_t) (1u << q);
            c_x[j]   = rope2d ? tk[2] : 0;
            c_y[j]   = rope2d ? tk[3] : 0;
            if (own) {
                (*own)[s*n_tps + i] = j;
            }
        }
    }
}

static void init_kq_vis(ggml_tensor * cells, ggml_tensor * toks, bool rope2d, uint32_t seed) {
    std::vector<int32_t> c, t;
    gen_kq_vis(cells->ne[0], toks->ne[1], toks->ne[2], rope2d, seed, c, t, nullptr);

    ggml_backend_tensor_set(cells, c.data(), 0, c.size()*sizeof(int32_t));
    ggml_backend_tensor_set(toks,  t.data(), 0, t.size()*sizeof(int32_t));
}

// GGML_OP_KQ_MASK: the backend must reproduce the CPU reference bit for bit
struct test_kq_mask : public test_case {
    const int64_t   n_kv;
    const int64_t   n_tps;
    const int64_t   n_stream;
    const ggml_type type;
    const bool      causal;
    const bool      alibi;
    const int       swa_type;
    const int       n_swa;
    const bool      rope2d;

    std::string vars() override {
        return VARS_TO_STR9(n_kv, n_tps, n_stream, type, causal, alibi, swa_type, n_swa, rope2d);
    }

    test_kq_mask(int64_t n_kv = 1024, int64_t n_tps = 32, int64_t n_stream = 1, ggml_type type = GGML_TYPE_F16,
                 bool causal = true, bool alibi = false, int swa_type = 0, int n_swa = 0, bool rope2d = false)
        : n_kv(n_kv), n_tps(n_tps), n_stream(n_stream), type(type), causal(causal), alibi(alibi),
          swa_type(swa_type), n_swa(n_swa), rope2d(rope2d) {}

    double max_err() override { return 0.0; }

    // number of elements whose bits differ (the values are f16/f32 0, -inf or -|d|)
    double err(const float * a, const float * b, size_t n) override {
        size_t nd = 0;
        for (size_t i = 0; i < n; ++i) {
            nd += memcmp(&a[i], &b[i], sizeof(float)) != 0;
        }
        return (double) nd;
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * cells = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_kv, 4, n_stream);
        ggml_set_name(cells, "cells");
        ggml_tensor * toks = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, n_tps, n_stream);
        ggml_set_name(toks, "toks");

        ggml_tensor * out = ggml_kq_mask(ctx, cells, toks, type, causal, alibi, swa_type, n_swa);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        init_kq_vis(ggml_get_tensor(ctx, "cells"), ggml_get_tensor(ctx, "toks"), rope2d, 1234u + (uint32_t) (n_kv + 7*n_tps + 13*n_stream));
    }
};

// FLASH_ATTN_EXT with the mask computed from the metadata (ggml_flash_attn_ext_vis), against the
// CPU backend with the usual tolerance. (Bit identity plane vs metadata on ONE backend is the job of
// test-kq-mask-meta --device, which can compare two graphs on the same device.)
struct test_flash_attn_ext_vis : public test_case {
    const int64_t   hs;
    const int64_t   nh_kv;
    const int64_t   gqa;
    const int64_t   kv;
    const int64_t   nb;
    const int64_t   n_stream;
    const ggml_type type_KV;
    const bool      causal;
    const float     max_bias; // > 0: ALiBi mask values (-|p0 - p1|) with slopes
    const int       swa_type;
    const int       n_swa;
    const bool      rope2d;

    std::string vars() override {
        return VARS_TO_STR12(hs, nh_kv, gqa, kv, nb, n_stream, type_KV, causal, max_bias, swa_type, n_swa, rope2d);
    }

    double max_nmse_err() override { return 5e-4; }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return 2 * nh_kv*gqa * nb * (hs + hs) * kv * n_stream;
    }

    test_flash_attn_ext_vis(int64_t hs = 256, int64_t nh_kv = 2, int64_t gqa = 12, int64_t kv = 1024, int64_t nb = 6,
                            int64_t n_stream = 1, ggml_type type_KV = GGML_TYPE_F16, bool causal = true,
                            float max_bias = 0.0f, int swa_type = 0, int n_swa = 0, bool rope2d = false)
        : hs(hs), nh_kv(nh_kv), gqa(gqa), kv(kv), nb(nb), n_stream(n_stream), type_KV(type_KV), causal(causal),
          max_bias(max_bias), swa_type(swa_type), n_swa(n_swa), rope2d(rope2d) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, nb, nh_kv*gqa, n_stream);
        ggml_set_name(q, "q");
        ggml_tensor * k = ggml_new_tensor_4d(ctx, type_KV, hs, kv, nh_kv, n_stream);
        ggml_set_name(k, "k");
        ggml_tensor * v = ggml_new_tensor_4d(ctx, type_KV, hs, kv, nh_kv, n_stream);
        ggml_set_name(v, "v");
        ggml_tensor * cells = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, kv, 4, n_stream);
        ggml_set_name(cells, "cells");
        ggml_tensor * toks = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, nb, n_stream);
        ggml_set_name(toks, "toks");

        ggml_tensor * out = ggml_flash_attn_ext_vis(ctx, q, k, v, cells, toks, 1.0f/sqrtf((float) hs), max_bias, 0.0f,
                                                    causal, max_bias > 0.0f, swa_type, n_swa);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "cells") == 0 || strcmp(t->name, "toks") == 0) {
                continue;
            }
            init_tensor_uniform(t);
        }
        init_kq_vis(ggml_get_tensor(ctx, "cells"), ggml_get_tensor(ctx, "toks"), rope2d, 4321u + (uint32_t) (kv + 7*nb + 13*n_stream));
    }
};

// [TAG_UP29591] GGML_OP_MUL_MAT twice on one input, with an op between them that uses the backend's scratch the
// first matmul converted that input into (Vulkan: prealloc_y). The second matmul must not reuse the conversion.
// The second weight depends on the op in between, so no scheduler can move it before that op.
// Upstream #29591's test (dropped there as backend-specific), plus mode 2: FA over the visibility metadata, whose
// vis-opt pass writes the same scratch in the fork.
// y_nc: the cases without it never reach the stale read on a KHR_coopmat device (RDNA3 / RDNA3.5): their contiguous
// f32 input goes into prealloc_y only as MMQ's q8_1, and the coopmat matmul pipelines have no q8_1 variant, so the
// matmul reads the input in place (with GGML_VK_DISABLE_COOPMAT=1 the q4_0 / q8_0 ones do fail without the fix). With
// y_nc the input is a small non-contiguous view, which every device converts to f16 into prealloc_y; the op in
// between also runs once before the first matmul, as an earlier layer's would, so it has already sized the scratch
// (a reallocation there would clear the record and hide the stale read), and its write covers >= 25 % of the input.
struct test_mul_mat_reuse_across_scratch : public test_case {
    const ggml_type type_w;
    const int       mode; // 0: a wide soft_max (> 16384 cols), 1: flash attention with a large mask, 2: 1 over the metadata
    const bool      y_nc;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT";
    }

    std::string vars() override {
        return y_nc ? VARS_TO_STR3(type_w, mode, y_nc) : VARS_TO_STR2(type_w, mode);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    bool run_whole_graph() override { return true; }

    test_mul_mat_reuse_across_scratch(ggml_type type_w = GGML_TYPE_Q4_0, int mode = 1, bool y_nc = false)
        : type_w(type_w), mode(mode), y_nc(y_nc) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t k = y_nc ? 32 : 256, m = 128, n = y_nc ? 16 : 64;
        const int64_t hs = 128, nq = y_nc ? 512 : n, kv = y_nc ? 2048 : 1024; // the op in between

        ggml_tensor * x = nullptr;
        if (y_nc) {
            ggml_tensor * xb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2*k, n);
            ggml_set_name(xb, "xb");
            x = ggml_view_2d(ctx, xb, k, n, xb->nb[1], 0);
        } else {
            x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        }
        ggml_set_name(x, "x");

        ggml_tensor * q = nullptr, * kk = nullptr, * v = nullptr, * mk = nullptr, * cells = nullptr, * toks = nullptr, * logits = nullptr;
        if (mode == 1) {
            const int64_t nh = 4;

            q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hs, nq, nh);
            kk = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hs, kv, nh);
            v  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hs, kv, nh);
            mk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, nq);
            ggml_set_name(mk, "m");
        } else if (mode == 2) {
            const int64_t nh_kv = 2, gqa = 2;

            q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, nq, nh_kv*gqa, 1);
            kk    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hs, kv, nh_kv,     1);
            v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hs, kv, nh_kv,     1);
            cells = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, kv, 4, 1);
            ggml_set_name(cells, "cells");
            toks  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, nq, 1);
            ggml_set_name(toks, "toks");
        } else {
            logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 20000, 4);
        }

        // the op in between; the result is zero, but only known once the op has run
        auto op_then_zero = [&]() {
            ggml_tensor * op = nullptr;
            if (mode == 1) {
                op = ggml_flash_attn_ext(ctx, q, kk, v, mk, 1.0f/sqrtf((float) hs), 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(op, GGML_PREC_F32);
            } else if (mode == 2) {
                op = ggml_flash_attn_ext_vis(ctx, q, kk, v, cells, toks, 1.0f/sqrtf((float) hs), 0.0f, 0.0f, true, false, 0, 0);
                ggml_flash_attn_ext_set_prec(op, GGML_PREC_F32);
            } else {
                op = ggml_soft_max(ctx, logits);
            }
            return ggml_scale(ctx, ggml_sum(ctx, op), 0.0f);
        };

        ggml_tensor * w1 = ggml_new_tensor_2d(ctx, type_w, k, m);
        ggml_set_name(w1, "w1");
        if (y_nc) {
            ggml_tensor * w1f = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
            w1 = ggml_cpy(ctx, ggml_add(ctx, w1f, op_then_zero()), w1);
        }

        ggml_tensor * y1 = ggml_mul_mat(ctx, w1, x);

        ggml_tensor * after = op_then_zero();

        ggml_tensor * w2f = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
        ggml_tensor * w2  = ggml_cpy(ctx, ggml_add(ctx, w2f, after), ggml_new_tensor_2d(ctx, type_w, k, m));

        ggml_tensor * y2 = ggml_mul_mat(ctx, w2, x);

        ggml_tensor * out = ggml_add(ctx, y1, y2);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "m") == 0) {
                init_tensor_kq_mask(t);
            } else if (strcmp(t->name, "cells") == 0 || strcmp(t->name, "toks") == 0) {
                continue;
            } else if (y_nc && strcmp(t->name, "x") == 0) {
                continue; // a view of xb
            } else {
                init_tensor_uniform(t);
            }
        }
        if (mode == 2) {
            init_kq_vis(ggml_get_tensor(ctx, "cells"), ggml_get_tensor(ctx, "toks"), false, 29591u);
        }
    }
};

// [TAG_QSA_INDEXED_FA] mask [n_kv, n_tps, 1, n_stream] f16 + ids [W, n_tps, 1, n_stream] i32; each row W unique ids
// (a shuffled prefix, ascending if sorted). dead: 0 = 0..30 % (floor) of the row's ids -inf; 1 = row 0 keeps one live id;
// 2 = causal (query t sees cells < n_kv - n_tps + t + 1) and every list holds cell n_kv - 1; 3 = row 0 all dead.
static void init_kv_ids_case(ggml_tensor * mask, ggml_tensor * ids, int dead, bool sorted) {
    GGML_ASSERT(mask->type == GGML_TYPE_F16 && ids->type == GGML_TYPE_I32);
    const int64_t n_kv = mask->ne[0], n_tps = mask->ne[1], n_stream = mask->ne[3], W = ids->ne[0];
    std::mt19937 gen(4242); std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::vector<float> m(n_kv*n_tps*n_stream);
    std::vector<int32_t> id(W*n_tps*n_stream), perm(n_kv);
    for (int64_t row = 0; row < n_tps*n_stream; ++row) {
        const int64_t n_vis = n_kv - n_tps + row % n_tps + 1;
        float * mr = m.data() + row*n_kv; int32_t * ir = id.data() + row*W;
        for (int64_t c = 0; c < n_kv; ++c) { mr[c] = (dead == 2 && c >= n_vis) ? -INFINITY : dis(gen); perm[c] = (int32_t) c; }
        std::shuffle(perm.begin(), perm.end(), gen);
        if (dead == 2) { std::iter_swap(perm.begin(), std::find(perm.begin(), perm.end(), (int32_t) (n_kv - 1))); }
        std::copy(perm.begin(), perm.begin() + W, ir);
        if (sorted) { std::sort(ir, ir + W); }
        const int64_t n_dead = dead == 0 ? W*(1 + row % 30)/100 : row != 0 ? 0 : dead == 1 ? W - 1 : dead == 3 ? W : 0;
        for (int64_t c = 0; c < n_dead; ++c) { mr[ir[W - 1 - c]] = -INFINITY; }
    }
    std::vector<ggml_fp16_t> m16(m.size()); ggml_fp32_to_fp16_row(m.data(), m16.data(), m.size());
    ggml_backend_tensor_set(mask, m16.data(), 0, m16.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(ids, id.data(), 0, id.size()*sizeof(int32_t));
}
// the Flash-Next attention shape: head 256, 2 KV heads, GQA 12, K/V as cache views ([D, n_head_kv, n_kv] permuted)
struct fa_indexed_io { ggml_tensor * q, * k, * v, * m, * ids; };
static fa_indexed_io fa_indexed_new(ggml_context * ctx, ggml_type type_KV, int64_t n_tps, int64_t n_kv, int64_t W, int64_t n_stream) {
    fa_indexed_io io;
    io.q   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, n_tps, 24, n_stream);
    io.k   = ggml_permute(ctx, ggml_new_tensor_4d(ctx, type_KV, 256, 2, n_kv, n_stream), 0, 2, 1, 3);
    io.v   = ggml_permute(ctx, ggml_new_tensor_4d(ctx, type_KV, 256, 2, n_kv, n_stream), 0, 2, 1, 3);
    io.m   = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_tps, 1, n_stream); ggml_set_name(io.m, "m");
    io.ids = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, W, n_tps, 1, n_stream);    ggml_set_name(io.ids, "ids");
    return io;
}
static ggml_tensor * fa_indexed_op(ggml_context * ctx, const fa_indexed_io & io, ggml_tensor * mask, ggml_tensor * ids) {
    ggml_tensor * out = ggml_flash_attn_ext(ctx, io.q, io.k, io.v, mask, 1.0f/16.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32); ggml_flash_attn_ext_set_kv_ids(out, ids);
    return out;
}
static void fa_indexed_init(ggml_context * ctx, int dead, bool sorted) {
    ggml_tensor * m = nullptr, * ids = nullptr;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        if (t->op == GGML_OP_NONE) { if (!strcmp(t->name, "m")) { m = t; } else if (!strcmp(t->name, "ids")) { ids = t; } else { init_tensor_uniform(t); } }
    }
    GGML_ASSERT(m && ids); // found by name: a rename must not fall through to the uniform fill
    init_kv_ids_case(m, ids, dead, sorted);
}
struct test_flash_attn_ext_indexed : public test_case {
    const ggml_type type_KV; const int64_t n_tps, n_kv, idx_W, n_stream; const int dead; const bool sorted;
    std::string vars() override { return VARS_TO_STR7(type_KV, n_tps, n_kv, idx_W, n_stream, dead, sorted); }
    double max_nmse_err() override { return 5e-4; }
    test_flash_attn_ext_indexed(ggml_type type_KV, int64_t n_tps, int64_t n_kv, int64_t idx_W, int64_t n_stream = 1, int dead = 0, bool sorted = false)
        : type_KV(type_KV), n_tps(n_tps), n_kv(n_kv), idx_W(idx_W), n_stream(n_stream), dead(dead), sorted(sorted) {}
    ggml_tensor * build_graph(ggml_context * ctx) override {
        const fa_indexed_io io = fa_indexed_new(ctx, type_KV, n_tps, n_kv, idx_W, n_stream);
        ggml_tensor * out = fa_indexed_op(ctx, io, io.m, io.ids);
        ggml_set_name(out, "out");
        return out;
    }
    void initialize_tensors(ggml_context * ctx) override { fa_indexed_init(ctx, dead, sorted); }
};
// spec §3.6: out = concat(indexed(sorted ids) - dense(scan mask), dense) along dim 3; the difference half must be
// +0.0 on the CPU reference (b). The backend under test (a) is held to a relative bar on its own dense half,
// sum(diff^2) / sum(dense^2) <= 5e-5, AND to a per-element cap max|diff| <= 5e-3 (the ratio alone would let one
// element be off by ~0.06 at Q8_0 64/4096/2051). Estimate: uniform q/k/v in [-1, 1] at scale 1/16 give a near-uniform
// softmax, so |dense| ~ 0.577/sqrt(W) ~ 0.013 RMS at W 2051 (the cap alone is ~1/3 of it); dropping 3 live columns
// shifts the output by ~sqrt(3)*0.577/2051 ~ 4.9e-4 RMS, a ratio ~1.3e-3 (~25x over the bar), while the two paths
// differ only in f16 P rounding / summation order (ratio ~1e-6). Tail-tile coverage comes from the UNSORTED indexed
// cases (NMSE vs CPU), not from this selfcheck: its sorted ids + dead layout make the tail tile all -inf.
// cases must be q8_0, or f16 with n_tps < GGML_FA_TILE_Q (64): the dense node must stay off the CPU tiled path to be bit-exact
struct test_flash_attn_ext_indexed_selfcheck : public test_case {
    const ggml_type type_KV; const int64_t n_tps, n_kv, idx_W; const int dead; const int64_t n_stream;
    std::string vars() override { return VARS_TO_STR6(type_KV, n_tps, n_kv, idx_W, dead, n_stream) + ",selfcheck"; }
    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "FLASH_ATTN_EXT"; }
    bool run_whole_graph() override { return true; }
    double max_nmse_err() override { return 0.0; }
    double err(const float * a, const float * b, size_t n) override {
        const size_t nd = n/2; // [0, nd) = indexed - dense, [nd, n) = dense
        double d2 = 0.0, s2 = 0.0, worst = 0.0;
        for (size_t i = 0; i < nd; ++i) {
            if (b[i] != 0.0f) { return 1.0; }
            d2 += (double) a[i]*a[i]; worst = std::max(worst, (double) std::fabs(a[i]));
        }
        for (size_t i = nd; i < n; ++i) { s2 += (double) a[i]*a[i]; }
        if (!(s2 > 0.0)) { return 1.0; }
        if (worst > 5e-3) { return worst; }
        return d2/s2 <= 5e-5 ? 0.0 : d2/s2;
    }
    test_flash_attn_ext_indexed_selfcheck(ggml_type type_KV, int64_t n_tps, int64_t n_kv, int64_t idx_W, int dead, int64_t n_stream = 1)
        : type_KV(type_KV), n_tps(n_tps), n_kv(n_kv), idx_W(idx_W), dead(dead), n_stream(n_stream) {}
    ggml_tensor * build_graph(ggml_context * ctx) override {
        const fa_indexed_io io = fa_indexed_new(ctx, type_KV, n_tps, n_kv, idx_W, n_stream);
        // the scan's mask, as build_qsa_scan builds it (src/models/qwen4exp.cpp:1737-1763)
        ggml_tensor * all = ggml_fill(ctx, io.m, -INFINITY);
        all = ggml_view_4d(ctx, all, 1, all->ne[0], all->ne[1], all->ne[3], all->nb[0], all->nb[1], all->nb[2], 0);
        ggml_tensor * ids3 = ggml_view_4d(ctx, io.ids, io.ids->ne[0], io.ids->ne[1], io.ids->ne[3], 1,
                                          io.ids->nb[1], io.ids->nb[2], io.ids->ne[3]*io.ids->nb[3], 0);
        ggml_tensor * zeros = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, ids3->ne[0], ids3->ne[1], ids3->ne[2]), 0.0f);
        ggml_tensor * sm = ggml_set_rows(ctx, all, zeros, ids3);
        sm = ggml_view_4d(ctx, sm, sm->ne[1], sm->ne[2], 1, sm->ne[3], sm->nb[2], sm->nb[3], sm->nb[3], 0);
        sm = ggml_add(ctx, sm, io.m);
        ggml_tensor * dense = fa_indexed_op(ctx, io, sm, nullptr);
        ggml_tensor * out = ggml_concat(ctx, ggml_sub(ctx, fa_indexed_op(ctx, io, io.m, io.ids), dense), dense, 3);
        ggml_set_name(out, "out");
        return out;
    }
    void initialize_tensors(ggml_context * ctx) override { fa_indexed_init(ctx, dead, true); }
};

// [TAG_QSA_VIS] (A4 T3/T4) the metadata forms against their plane forms ON THE SAME BACKEND, bit for bit. Each graph
// computes both and `out` concatenates them (as f32: ids < 2^24 and f16 mask values convert exactly), so err() sees the
// (plane, vis) pair of the backend under test AND of the CPU reference and requires plane == vis in both (EXACT), then the
// backend against the CPU. Metadata: gen_kq_vis (holes, seqless used cells, 1-5 sequences with shared-prefix cells, rope2d
// position ties) + optional extra holes (never a token's own cell). mutate = the negative control: the PLANE side reads cells
// with one sequence bit of row 0's own cell flipped (and that cell made the row's best), so the halves MUST differ - those
// cases are registered only with GGML_TEST_A4_NEGATIVE=1, where every case must FAIL.
struct qsa_vis_meta {
    std::vector<int32_t> c, cp, t; // cells, the plane side's cells, toks
    std::vector<int64_t> own;      // each token's own cell
};

// mutate_tok >= 0: the plane side loses that token's (stream 0) own cell
static qsa_vis_meta qsa_vis_meta_gen(int64_t n_kv, int64_t n_tps, int64_t n_stream, bool rope2d, uint32_t seed, int holes_pct,
                                     int64_t mutate_tok) {
    qsa_vis_meta m;
    gen_kq_vis(n_kv, n_tps, n_stream, rope2d, seed, m.c, m.t, &m.own);
    if (holes_pct > 0) {
        std::mt19937 rng(seed ^ 0x5eedu);
        std::vector<char> is_own((size_t) (n_kv*n_stream), 0);
        for (int64_t q = 0; q < n_tps*n_stream; ++q) {
            is_own[(size_t) ((q / n_tps)*n_kv + m.own[q])] = 1;
        }
        for (int64_t s = 0; s < n_stream; ++s) {
            int32_t * c = m.c.data() + 4*n_kv*s;
            for (int64_t j = 0; j < n_kv; ++j) {
                if (!is_own[(size_t) (s*n_kv + j)] && (int) (rng() % 100) < holes_pct) {
                    c[j] = -1; c[n_kv + j] = 0; c[2*n_kv + j] = 0; c[3*n_kv + j] = 0;
                }
            }
        }
    }
    m.cp = m.c;
    if (mutate_tok >= 0) {
        // the token loses its own cell's sequence bit on the plane side only
        m.cp[(size_t) (n_kv + m.own[mutate_tok])] ^= m.t[4*mutate_tok + 1];
    }
    return m;
}

// two halves of `n` floats (a = plane form, b = metadata form) bitwise equal
static double qsa_vis_halves_differ(const float * x, size_t n, size_t half_stride, size_t half_len, size_t n_blocks) {
    double d = 0.0;
    for (size_t blk = 0; blk < n_blocks; ++blk) {
        const float * a = x + blk*half_stride;
        const float * b = a + half_len;
        for (size_t i = 0; i < half_len; ++i) {
            d += memcmp(&a[i], &b[i], sizeof(float)) != 0 ? 1.0 : 0.0;
        }
    }
    GGML_UNUSED(n);
    return d;
}

// [TAG_QSA_TOPK_NOKEYS] modes 3.. (QSA_TOP_K_NOKEYS, the A3 domain and tie stress): 3 every block equal, 4 signed zeros
// (+0 / -0) among quantized scores, 5 the normal boundary (FLT_MIN, its neighbours, +-0, 1e-30, -inf), 6 ties across
// tiles (a long run of equal blocks that k cuts inside, spanning tile boundaries), 7 subnormal scores (OUTSIDE the
// integer key's domain: a diagnostic registered only with GGML_TEST_NOKEYS_SUBNORMAL=1)
struct test_qsa_top_k_vis : public test_case {
    const int64_t n_kv, n_tps, n_stream;
    const int     width, n_sel, mode, holes;
    const bool    rope2d, pad, mutate;
    bool          nokeys_family = false;   // [TAG_QSA_TOPK_NOKEYS] reported as QSA_TOP_K_NOKEYS

    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return nokeys_family ? "QSA_TOP_K_NOKEYS" : "QSA_TOP_K_VIS"; }
    std::string vars() override {
        return VARS_TO_STR11(n_kv, n_tps, n_stream, width, n_sel, mode, holes, rope2d, pad, mutate, n_full());
    }
    test_qsa_top_k_vis(int64_t n_kv, int64_t n_tps, int64_t n_stream, int width, int n_sel, int mode, int holes = 0,
                       bool rope2d = false, bool pad = false, bool mutate = false)
        : n_kv(n_kv), n_tps(n_tps), n_stream(n_stream), width(width), n_sel(n_sel), mode(mode), holes(holes),
          rope2d(rope2d), pad(pad), mutate(mutate) {}

    double max_err() override { return 0.0; }
    bool run_whole_graph() override { return true; }

    int64_t n_blocks() const { return (n_kv + 3)/4; }
    // pad: the chunk is the ubatch's last np = n_tps rows from t0 = n_full - n_tps + 3, its last 3 rows padding (clamped
    // to the last query), as build_qsa_chunked builds a [TAG_QSA_CHUNK_PAD] tail chunk
    int64_t n_full() const { return pad ? n_tps + 8 : n_tps; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * score    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_blocks(), n_tps, n_stream); ggml_set_name(score, "score");
        ggml_tensor * cell_blk = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_stream);              ggml_set_name(cell_blk, "cell_blk");
        ggml_tensor * cells    = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_kv, 4, n_stream);           ggml_set_name(cells, "cells");
        ggml_tensor * cells_p  = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_kv, 4, n_stream);           ggml_set_name(cells_p, "cells_p");
        ggml_tensor * toks     = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, n_full(), n_stream);       ggml_set_name(toks, "toks");
        ggml_tensor * plane    = ggml_kq_mask(ctx, cells_p, toks, GGML_TYPE_F16, true, false, 0, 0);
        ggml_tensor * toks_c   = toks;
        if (pad) {
            ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_tps, n_stream); ggml_set_name(rows, "rows");
            toks_c = ggml_get_rows(ctx, toks, rows);
            plane  = ggml_get_rows_type(ctx, plane, ggml_reshape_3d(ctx, rows, n_tps, 1, n_stream), GGML_TYPE_F16);
        }
        ggml_tensor * a = n_sel > 0 ? ggml_qsa_top_k_blocks(ctx, score, cell_blk, plane, width, n_sel)
                                    : ggml_qsa_top_k(ctx, score, cell_blk, plane, width);
        ggml_tensor * b = ggml_qsa_top_k_vis(ctx, score, cell_blk, cells, toks_c, width, n_sel, true);
        ggml_tensor * out = ggml_concat(ctx, ggml_cast(ctx, a, GGML_TYPE_F32), ggml_cast(ctx, b, GGML_TYPE_F32), 1);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        const int64_t t0 = n_full() - n_tps + 3;
        // the negative control hides row 0's own cell (row 0 = token t0 of a pad chunk) on the plane side
        const qsa_vis_meta m = qsa_vis_meta_gen(n_kv, n_full(), n_stream, rope2d, 4242u + (uint32_t) (n_kv + n_tps + mode), holes,
                                                mutate ? (pad ? t0 : 0) : -1);
        std::mt19937 rng(77u + (uint32_t) mode);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        const int64_t nb = n_blocks();
        std::vector<int32_t> rows((size_t) (n_tps*n_stream));
        for (int64_t s = 0; s < n_stream; ++s) {
            for (int64_t i = 0; i < n_tps; ++i) {
                rows[s*n_tps + i] = (int32_t) (pad ? std::min<int64_t>(t0 + i, n_full() - 1) : i);
            }
        }
        std::vector<float> sc((size_t) (nb*n_tps*n_stream));
        for (int64_t s = 0; s < n_stream; ++s) {
            for (int64_t i = 0; i < n_tps; ++i) {
                float * r = sc.data() + (s*n_tps + i)*nb;
                for (int64_t b = 0; b < nb; ++b) {
                    const float x = u01(rng);
                    r[b] = mode == 0 ? 8.0f*x                                                   // (almost) distinct
                         : (x < 0.6f ? 0.0f : std::floor(16.0f*u01(rng))/4.0f);                  // ReLU-tied, quantized
                    if (mode == 2 && u01(rng) < 0.05f) { r[b] = -INFINITY; }                    // dead blocks
                    if (mode == 3) { r[b] = 0.5f; }                                             // every block equal
                    if (mode == 4 && x < 0.5f) { r[b] = (rng() & 1u) ? 0.0f : -0.0f; }          // signed zeros
                    if (mode == 5) {                                                            // the normal boundary
                        static const float vals[8] = { FLT_MIN, 2.0f*FLT_MIN, std::nextafter(FLT_MIN, 1.0f), 0.0f, -0.0f,
                                                       1e-30f, -INFINITY, 4.0f*FLT_MIN };
                        r[b] = vals[rng() % 8u];
                    }
                    if (mode == 6) { r[b] = (b % 1500) < 900 ? 1.0f : 0.25f*std::floor(4.0f*x); } // ties across tiles
                    if (mode == 7) { r[b] = x < 0.5f ? std::ldexp(1.0f + std::floor(8.0f*x), -140) : 0.0f; } // subnormal
                }
                if (mode == 2) {
                    // the census's +1e9 boost on the block of the row's own cell (the tail)
                    r[m.own[s*n_full() + rows[s*n_tps + i]] / 4] = 1e9f;
                }
            }
        }
        if (mutate) {
            // row 0's own cell is its best finite block: selected by the metadata form, hidden on the plane side
            sc[(size_t) (m.own[rows[0]] / 4)] = 1e6f;
        }
        std::vector<int32_t> cb((size_t) (n_kv*n_stream));
        for (int64_t s = 0; s < n_stream; ++s) {
            for (int64_t j = 0; j < n_kv; ++j) {
                cb[s*n_kv + j] = (int32_t) (j / 4);
            }
        }
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE) { continue; }
            if      (!strcmp(t->name, "score"))    { ggml_backend_tensor_set(t, sc.data(),   0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "cell_blk")) { ggml_backend_tensor_set(t, cb.data(),   0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "cells"))    { ggml_backend_tensor_set(t, m.c.data(),  0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "cells_p"))  { ggml_backend_tensor_set(t, m.cp.data(), 0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "toks"))     { ggml_backend_tensor_set(t, m.t.data(),  0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "rows"))     { ggml_backend_tensor_set(t, rows.data(), 0, ggml_nbytes(t)); }
        }
    }

    // out: [width, 2*n_tps, 1, n_stream] - per stream n_tps plane rows, then n_tps metadata rows
    double err(const float * a, const float * b, size_t n) override {
        const size_t half = (size_t) (width*n_tps), stride = 2*half;
        double d = qsa_vis_halves_differ(a, n, stride, half, (size_t) n_stream)  // backend: plane == vis
                 + qsa_vis_halves_differ(b, n, stride, half, (size_t) n_stream); // CPU:     plane == vis
        for (int64_t s = 0; s < n_stream; ++s) {                                 // backend vs CPU (the vis half)
            const float * av = a + s*stride + half;
            const float * bv = b + s*stride + half;
            if (n_sel > 0) {
                for (size_t i = 0; i < half; ++i) { d += av[i] != bv[i] ? 1.0 : 0.0; }
            } else {
                d += cmp_ids_per_row(av, bv, half, width, n_kv);
            }
        }
        return d;
    }
};

struct test_kq_mask_ids : public test_case {
    const int64_t n_kv, n_tps, n_stream, W;
    const bool    rope2d, mutate;

    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "KQ_MASK_IDS"; }
    std::string vars() override { return VARS_TO_STR6(n_kv, n_tps, n_stream, W, rope2d, mutate); }
    test_kq_mask_ids(int64_t n_kv, int64_t n_tps, int64_t n_stream, int64_t W, bool rope2d = false, bool mutate = false)
        : n_kv(n_kv), n_tps(n_tps), n_stream(n_stream), W(W), rope2d(rope2d), mutate(mutate) {}

    double max_err() override { return 0.0; }
    bool run_whole_graph() override { return true; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t n_q = n_tps*n_stream;
        ggml_tensor * cells   = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_kv, 4, n_stream);     ggml_set_name(cells, "cells");
        ggml_tensor * cells_p = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_kv, 4, n_stream);     ggml_set_name(cells_p, "cells_p");
        ggml_tensor * toks    = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, n_tps, n_stream);    ggml_set_name(toks, "toks");
        ggml_tensor * ids     = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, W, n_tps, 1, n_stream); ggml_set_name(ids, "ids");
        // the gather's plane form (build_qsa_gather): the f16 plane's rows flattened over (t, s), its cells at the ids
        ggml_tensor * plane = ggml_kq_mask(ctx, cells_p, toks, GGML_TYPE_F16, true, false, 0, 0);
        ggml_tensor * cellv = ggml_view_3d(ctx, plane, 1, n_kv, n_q, plane->nb[0], plane->nb[1], 0);
        ggml_tensor * a = ggml_reshape_4d(ctx, ggml_get_rows_type(ctx, cellv, ggml_reshape_3d(ctx, ids, W, n_q, 1), GGML_TYPE_F16),
                                          W, 1, 1, n_q);
        ggml_tensor * b = ggml_kq_mask_ids(ctx, cells, toks, ids, GGML_TYPE_F16, true, false, 0, 0);
        ggml_tensor * out = ggml_concat(ctx, ggml_cast(ctx, a, GGML_TYPE_F32), ggml_cast(ctx, b, GGML_TYPE_F32), 0);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        const qsa_vis_meta m = qsa_vis_meta_gen(n_kv, n_tps, n_stream, rope2d, 99u + (uint32_t) (n_kv + n_tps), 0, mutate ? 0 : -1);
        std::mt19937 rng(1234u);
        std::vector<int32_t> id((size_t) (W*n_tps*n_stream)), perm((size_t) n_kv);
        for (int64_t q = 0; q < n_tps*n_stream; ++q) {
            for (int64_t j = 0; j < n_kv; ++j) { perm[j] = (int32_t) j; }
            std::shuffle(perm.begin(), perm.end(), rng);
            // the row's own cell first (row 0's is the one the negative control flips)
            std::iter_swap(perm.begin(), std::find(perm.begin(), perm.end(), (int32_t) m.own[q]));
            std::copy(perm.begin(), perm.begin() + W, id.begin() + q*W);
        }
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE) { continue; }
            if      (!strcmp(t->name, "cells"))   { ggml_backend_tensor_set(t, m.c.data(),  0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "cells_p")) { ggml_backend_tensor_set(t, m.cp.data(), 0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "toks"))    { ggml_backend_tensor_set(t, m.t.data(),  0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "ids"))     { ggml_backend_tensor_set(t, id.data(),   0, ggml_nbytes(t)); }
        }
    }

    // out: [2W, 1, 1, n_q] - per row W plane values, then W metadata values; everything bitwise
    double err(const float * a, const float * b, size_t n) override {
        double d = qsa_vis_halves_differ(a, n, 2*W, W, (size_t) (n_tps*n_stream))
                 + qsa_vis_halves_differ(b, n, 2*W, W, (size_t) (n_tps*n_stream));
        for (size_t i = 0; i < n; ++i) { d += memcmp(&a[i], &b[i], sizeof(float)) != 0 ? 1.0 : 0.0; }
        return d;
    }
};

// indexed flash attention over the metadata vs over the device plane, the Flash-Next shape (head 256, 2 KV heads, GQA 12);
// dead_tail: the last 3 ids of every row are -1 (dead columns, ggml.h - the [TAG_QSA_WHOLE_BLOCKS] padding guard)
struct test_flash_attn_ext_indexed_vis : public test_case {
    const ggml_type type_KV;
    const int64_t   n_tps, n_kv, W, n_stream;
    const bool      rope2d, dead_tail, mutate;

    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "FLASH_ATTN_EXT_INDEXED_VIS"; }
    std::string vars() override { return VARS_TO_STR8(type_KV, n_tps, n_kv, W, n_stream, rope2d, dead_tail, mutate); }
    test_flash_attn_ext_indexed_vis(ggml_type type_KV, int64_t n_tps, int64_t n_kv, int64_t W, int64_t n_stream = 1,
                                    bool rope2d = false, bool dead_tail = false, bool mutate = false)
        : type_KV(type_KV), n_tps(n_tps), n_kv(n_kv), W(W), n_stream(n_stream), rope2d(rope2d), dead_tail(dead_tail),
          mutate(mutate) {}

    double max_err() override { return 0.0; }
    bool run_whole_graph() override { return true; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * q       = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, n_tps, 24, n_stream);   ggml_set_name(q, "q");
        ggml_tensor * k       = ggml_permute(ctx, ggml_new_tensor_4d(ctx, type_KV, 256, 2, n_kv, n_stream), 0, 2, 1, 3);
        ggml_tensor * v       = ggml_permute(ctx, ggml_new_tensor_4d(ctx, type_KV, 256, 2, n_kv, n_stream), 0, 2, 1, 3);
        ggml_tensor * cells   = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_kv, 4, n_stream);          ggml_set_name(cells, "cells");
        ggml_tensor * cells_p = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, n_kv, 4, n_stream);          ggml_set_name(cells_p, "cells_p");
        ggml_tensor * toks    = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 4, n_tps, n_stream);         ggml_set_name(toks, "toks");
        ggml_tensor * ids     = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, W, n_tps, 1, n_stream);      ggml_set_name(ids, "ids");
        ggml_tensor * plane   = ggml_kq_mask(ctx, cells_p, toks, GGML_TYPE_F16, true, false, 0, 0);
        ggml_tensor * a = ggml_flash_attn_ext(ctx, q, k, v, plane, 1.0f/16.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(a, GGML_PREC_F32); ggml_flash_attn_ext_set_kv_ids(a, ids);
        ggml_tensor * b = ggml_flash_attn_ext_vis(ctx, q, k, v, cells, toks, 1.0f/16.0f, 0.0f, 0.0f, true, false, 0, 0);
        ggml_flash_attn_ext_set_prec(b, GGML_PREC_F32); ggml_flash_attn_ext_set_kv_ids(b, ids);
        ggml_tensor * out = ggml_concat(ctx, a, b, 3);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        const qsa_vis_meta m = qsa_vis_meta_gen(n_kv, n_tps, n_stream, rope2d, 7u + (uint32_t) (n_kv + n_tps), 0, mutate ? 0 : -1);
        std::mt19937 rng(4243u);
        std::vector<int32_t> id((size_t) (W*n_tps*n_stream)), perm((size_t) n_kv);
        for (int64_t q = 0; q < n_tps*n_stream; ++q) {
            for (int64_t j = 0; j < n_kv; ++j) { perm[j] = (int32_t) j; }
            std::shuffle(perm.begin(), perm.end(), rng);
            // every row attends its own cell (no all-dead row: 0/0 is unspecified), at a random slot
            std::iter_swap(perm.begin(), std::find(perm.begin(), perm.end(), (int32_t) m.own[q]));
            std::iter_swap(perm.begin(), perm.begin() + (int64_t) (rng() % (uint32_t) std::max<int64_t>(1, W - (dead_tail ? 3 : 0))));
            std::copy(perm.begin(), perm.begin() + W, id.begin() + q*W);
            if (dead_tail && W > 4) {
                for (int64_t c = W - 3; c < W; ++c) { id[q*W + c] = -1; }
            }
        }
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE) { continue; }
            if      (!strcmp(t->name, "cells"))   { ggml_backend_tensor_set(t, m.c.data(),  0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "cells_p")) { ggml_backend_tensor_set(t, m.cp.data(), 0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "toks"))    { ggml_backend_tensor_set(t, m.t.data(),  0, ggml_nbytes(t)); }
            else if (!strcmp(t->name, "ids"))     { ggml_backend_tensor_set(t, id.data(),   0, ggml_nbytes(t)); }
            else { init_tensor_uniform(t); }
        }
    }

    // out: [256, 24, n_tps, 2*n_stream] - streams 0..n_stream-1 the plane form, then the metadata form. Each backend: the
    // halves bitwise equal (one kernel, the same mask values). Backend vs CPU: the indexed tests' NMSE bar on the vis half.
    double err(const float * a, const float * b, size_t n) override {
        const size_t half = n/2;
        double d = qsa_vis_halves_differ(a, n, n, half, 1) + qsa_vis_halves_differ(b, n, n, half, 1);
        if (d != 0.0) { return 1.0 + d; }
        double e2 = 0.0, r2 = 0.0;
        for (size_t i = half; i < n; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { return 1e9; }
            e2 += (double) (a[i] - b[i])*(a[i] - b[i]); r2 += (double) b[i]*b[i];
        }
        const double nmse = r2 > 0.0 ? e2/r2 : e2;
        return nmse <= 5e-4 ? 0.0 : nmse;
    }
};

// GGML_OP_CROSS_ENTROPY_LOSS
struct test_cross_entropy_loss : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cross_entropy_loss(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * logits = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_param(logits);
        ggml_set_name(logits, "logits");

        ggml_tensor * labels = ggml_new_tensor(ctx, type, 4, ne.data());
        // The labels are assumed to be constant -> no gradients.
        ggml_set_name(labels, "labels");

        // Ensure labels add up to 1:
        labels = ggml_soft_max(ctx, labels);
        ggml_set_name(labels, "labels_normalized");

        ggml_tensor * out = ggml_cross_entropy_loss(ctx, logits, labels);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // For larger abs. diffs between logits softmax is more linear, therefore more precise num. gradients.
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -100.0f, 100.0f);
        }
    }

    float grad_eps() override {
        return 1.0f;
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_CROSS_ENTROPY_LOSS_BACK
struct test_cross_entropy_loss_back : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cross_entropy_loss_back(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * grad = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        ggml_set_name(grad, "grad");

        ggml_tensor * logits = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(logits, "logits");

        ggml_tensor * labels = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(labels, "labels");

        // Ensure labels add up to 1:
        labels = ggml_soft_max(ctx, labels);
        ggml_set_name(labels, "labels_normalized");

        ggml_tensor * out = ggml_cross_entropy_loss_back(ctx, grad, logits, labels);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_OPT_STEP_ADAMW
struct test_opt_step_adamw : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_opt_step_adamw(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 5, 4, 3})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_param(a); // Despite tensor a having gradients the output tensor will not.
        ggml_set_name(a, "a");

        ggml_tensor * grad = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(grad, "grad");

        ggml_tensor * grad_m = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(grad_m, "grad_m");

        ggml_tensor * grad_v = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(grad_v, "grad_v");

        ggml_tensor * adamw_params = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
        ggml_set_name(adamw_params, "adamw_params");

        ggml_tensor * out = ggml_opt_step_adamw(ctx, a, grad, grad_m, grad_v, adamw_params);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, 0.0f, 1.0f); // grad_v and adamw_params need non-negative values.
        }
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_OPT_STEP_SGD
struct test_opt_step_sgd : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_opt_step_sgd(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_param(a);  // Despite tensor a having gradients the output tensor will not.
        ggml_set_name(a, "a");

        ggml_tensor * grad = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(grad, "grad");

        ggml_tensor * sgd_params = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
        ggml_set_name(sgd_params, "sgd_params");

        ggml_tensor * out = ggml_opt_step_sgd(ctx, a, grad, sgd_params);

        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, 0.0f, 1.0f);  // sgd_params need non-negative values.
        }
    }

    bool grad_precise() override {
        return true;
    }
};

// GGML_OP_CUMSUM
struct test_cumsum : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_cumsum(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_cumsum(ctx, a);

        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -1.0f, 1.0f);
        }
    }
};

// GGML_OP_XIELU
struct test_xielu : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_xielu(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = { 10, 5, 4, 3 })
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        float alpha_n = 4.0f;
        float alpha_p = 20.0f;
        float beta = 0.5f;
        float eps = 0.0000001f;

        ggml_tensor * out = ggml_xielu(ctx, a, alpha_n, alpha_p, beta, eps);

        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -1.0f, 1.0f);
        }
    }
};

// GGML_OP_TRI
struct test_tri : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    const ggml_tri_type          tri_type;

    std::string vars() override { return VARS_TO_STR3(type, ne, tri_type); }

    test_tri(ggml_tri_type tri_type, ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = { 10, 10, 4, 3 })
        : type(type), ne(ne), tri_type(tri_type) {
            GGML_ASSERT(ne[0] == ne[1]);
        }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_tri(ctx, a, tri_type);

        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -1.0f, 1.0f);
        }
    }
};

// GGML_OP_FILL
struct test_fill : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;
    float                        c;

    std::string vars() override { return VARS_TO_STR3(type, ne, c); }

    test_fill(float c, ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = { 10, 10, 4, 3 })
        : type(type), ne(ne), c(c) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_fill(ctx, a, c);

        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_SOLVE_TRI
struct test_solve_tri : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne_lhs;
    const std::array<int64_t, 4> ne_rhs;

    std::string vars() override { return VARS_TO_STR3(type, ne_lhs, ne_rhs); }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        int64_t n = ne_lhs[0];
        int64_t k = ne_rhs[0];
        int64_t batch = ne_lhs[2] * ne_lhs[3];
        // n * (n + 1) / 2 non-zero elements of lhs, 2 flops each, for each col of rhs
        return n * (n + 1) * k * batch;
    }

    test_solve_tri(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_lhs = { 10, 10, 4, 3 },
            std::array<int64_t, 4> ne_rhs = { 3, 10, 4, 3 }
        )
        : type(type), ne_lhs(ne_lhs), ne_rhs(ne_rhs) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne_lhs[0], ne_lhs[1], ne_lhs[2], ne_lhs[3]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor_4d(ctx, type, ne_rhs[0], ne_rhs[1], ne_rhs[2], ne_rhs[3]);
        ggml_set_param(b);
        ggml_set_name(b, "b");

        ggml_tensor * out = ggml_solve_tri(ctx, a, b, true, true, false);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "a") == 0) {
                // note: avoid zeros in the diagonal
                init_tensor_tril(t, 0.1, 1.0f);
            } else {
                init_tensor_uniform(t, -1.0f, 1.0f);
            }
        }
    }
};

// GGML_OP_DIAG
struct test_diag : public test_case {
    const ggml_type              type;
    const std::array<int64_t, 4> ne;

    std::string vars() override { return VARS_TO_STR2(type, ne); }

    test_diag(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = { 10, 1, 4, 3 })
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        GGML_ASSERT(ne[1] == 1);
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_param(a);
        ggml_set_name(a, "a");

        ggml_tensor * out = ggml_diag(ctx, a);
        ggml_set_name(out, "out");

        return out;
    }
};

// GGML_OP_LIGHTNING_INDEXER
struct test_lightning_indexer : public test_case {
    const int64_t hsk; // indexer K head size
    const int64_t nh; // num indexer heads
    const int64_t kv; // kv size
    const int64_t nb; // batch size
    const int64_t ns; // num streams
    const int64_t nm; // ne[3] of mask

    const ggml_type type_K;

    std::string vars() override {
        return VARS_TO_STR7(hsk, nh, kv, nb, ns, nm, type_K);
    }

    double max_nmse_err() override {
        return 1e-6;
    }

    uint64_t op_flops(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return ((2 * hsk + 2) * nh + 1) * kv * nb * ns;
    }

    test_lightning_indexer(int64_t hsk = 128, int64_t nh = 64, int64_t kv = 256, int64_t nb = 128, int64_t ns = 1, int64_t nm = 1, ggml_type type_K = GGML_TYPE_F16)
        : hsk(hsk), nh(nh), kv(kv), nb(nb), ns(ns), nm(nm), type_K(type_K) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk, nh, nb, ns);
        ggml_set_param(q);
        ggml_set_name(q, "q");

        ggml_tensor * k = ggml_new_tensor_4d(ctx, type_K, hsk, 1, kv, ns);
        ggml_set_param(k);
        ggml_set_name(k, "k");

        ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, nh, nb, 1, ns);
        ggml_set_param(w);
        ggml_set_name(w, "w");

        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nb, 1, nm);
        ggml_set_param(m);
        ggml_set_name(m, "m");

        ggml_tensor * out = ggml_lightning_indexer(ctx, q, k, w, m);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "m") == 0) {
                init_tensor_kq_mask(t);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// Deserializable generic test case
struct input_tensor {
    ggml_type type;
    std::array<int64_t, 4> ne;
    std::array<size_t, 4> nb; // strides (0 = use default contiguous strides)
};

static bool is_non_contiguous(const input_tensor & src) {
    if (src.nb[0] == 0) {
        return false;
    }
    const size_t default_nb0 = ggml_type_size(src.type);
    const size_t default_nb1 = default_nb0 * (src.ne[0] / ggml_blck_size(src.type));
    const size_t default_nb2 = default_nb1 * src.ne[1];
    const size_t default_nb3 = default_nb2 * src.ne[2];
    return src.nb[0] != default_nb0 ||
           src.nb[1] != default_nb1 ||
           src.nb[2] != default_nb2 ||
           src.nb[3] != default_nb3;
}

static std::string var_to_str(const std::vector<input_tensor>& sources) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& src : sources) {
        if (!first) oss << ",";
        oss << ggml_type_name(src.type) << "[" << src.ne[0] << "," << src.ne[1] << "," << src.ne[2] << "," << src.ne[3] << "]";
        if (is_non_contiguous(src)) {
            oss << "nb[" << src.nb[0] << "," << src.nb[1] << "," << src.nb[2] << "," << src.nb[3] << "]";
        }
        first = false;
    }
    return oss.str();
}

static std::string var_to_str(const std::array<int32_t, GGML_MAX_OP_PARAMS / sizeof(int32_t)>& params) {
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i] != 0) {
            if (!first) oss << ",";
            oss << i << ":" << params[i];
            first = false;
        }
    }
    oss << "]";
    return oss.str();
}


struct test_generic_op : public test_case {
    const ggml_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int32_t, GGML_MAX_OP_PARAMS / sizeof(int32_t)> op_params;

    const std::vector<input_tensor> sources;
    const std::string name;

    std::string vars() override {
        if (name.empty()) {
            return VARS_TO_STR4(type, ne, op_params, sources);
        }

        return VARS_TO_STR5(name, type, ne, op_params, sources);
    }

    test_generic_op(ggml_op op, ggml_type type, std::array<int64_t, 4> ne,
                    std::array<int32_t, GGML_MAX_OP_PARAMS / sizeof(int32_t)> op_params,
                    std::vector<input_tensor> sources, std::string name = "")
        : op(op), type(type), ne(ne), op_params(op_params), sources(sources), name(std::move(name)) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const size_t source_count = std::min(sources.size(), (size_t)GGML_MAX_SRC);

        std::array<ggml_tensor *, GGML_MAX_SRC> source_tensors;
        for (size_t i = 0; i < source_count; ++i) {
            const input_tensor& src = sources[i];

            if (is_non_contiguous(src)) {
                size_t total_size;
                const size_t blck_size = ggml_blck_size(src.type);
                if (blck_size == 1) {
                    total_size = ggml_type_size(src.type);
                    for (int d = 0; d < 4; d++) {
                        total_size += (src.ne[d] - 1) * src.nb[d];
                    }
                } else {
                    total_size = src.ne[0] * src.nb[0] / blck_size;
                    for (int d = 1; d < 4; d++) {
                        total_size += (src.ne[d] - 1) * src.nb[d];
                    }
                }

                // Convert bytes to elements, padded to block size for quantized types
                const size_t type_size = ggml_type_size(src.type);
                size_t backing_elements = (total_size * blck_size + type_size - 1) / type_size;
                backing_elements = ((backing_elements + blck_size - 1) / blck_size) * blck_size;
                ggml_tensor * backing = ggml_new_tensor_1d(ctx, src.type, backing_elements);
                source_tensors[i] = ggml_view_4d(ctx, backing,
                    src.ne[0], src.ne[1], src.ne[2], src.ne[3],
                    src.nb[1], src.nb[2], src.nb[3], 0);
                // nb[0] does not get set by view_4d, so set it manually
                source_tensors[i]->nb[0] = src.nb[0];
            } else {
                source_tensors[i] = ggml_new_tensor_4d(ctx, src.type, src.ne[0], src.ne[1], src.ne[2], src.ne[3]);
            }
        }

        // Ops with an inplace flag create a view of src[0] as their output.
        bool inplace = false;
        if (op == GGML_OP_SET || op == GGML_OP_ACC) {
            inplace = op_params[4] != 0;
        } else if (op == GGML_OP_ADD_REL_POS) {
            inplace = op_params[0] != 0;
        }

        ggml_tensor * out;
        if (inplace && source_count > 0) {
            out = ggml_view_tensor(ctx, source_tensors[0]);
        } else {
            out = ggml_new_tensor_4d(ctx, type, ne[0], ne[1], ne[2], ne[3]);
        }
        out->op = op;
        for (size_t i = 0; i < source_count; ++i) {
            out->src[i] = source_tensors[i];
        }

        memcpy(out->op_params, op_params.data(), GGML_MAX_OP_PARAMS);
        ggml_set_name(out, "out");

        return out;
    }

    double max_nmse_err() override {
        switch (op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_IM2COL:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_SET_ROWS:
        case GGML_OP_CPY:
            return 5e-4;
        case GGML_OP_SOFT_MAX:
            return 1e-6;
        case GGML_OP_RWKV_WKV7:
            return 5e-3;
        case GGML_OP_FLASH_ATTN_EXT:
        {
            // Scale error with kv length to account for accumulating floating point error
            const int64_t kv = sources[1].ne[1];
            return 5e-4 * std::max(1.0, kv / 20000.0);
        }
        default:
            return 1e-7;
        }
    }

    void initialize_tensors(ggml_context * ctx) override {
        ggml_tensor * out = ggml_get_tensor(ctx, "out");

        std::random_device rd;
        std::default_random_engine rng(rd());

        for (size_t i = 0; i < sources.size() && i < GGML_MAX_SRC; i++) {
            ggml_tensor * t = out->src[i];
            if (!t) {
                break;
            }

            // FLASH_ATTN_EXT: src[3] is the KQ mask
            if (op == GGML_OP_FLASH_ATTN_EXT && i == 3) {
                init_tensor_kq_mask(t);
                continue;
            }

            if (t->type == GGML_TYPE_I32 || t->type == GGML_TYPE_I64) {
                if (op == GGML_OP_GET_ROWS || op == GGML_OP_GET_ROWS_BACK) {
                    const int64_t num_rows = sources[0].ne[1];
                    const int64_t nels = ggml_nelements(t);
                    std::vector<int32_t> data(nels);
                    std::uniform_int_distribution<int32_t> dist(0, num_rows - 1);
                    for (int64_t i = 0; i < nels; i++) {
                        data[i] = dist(rng);
                    }
                    ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(int32_t));
                } else if (op == GGML_OP_SET_ROWS) {
                    init_set_rows_row_ids(t, ne[1]);
                } else if (op == GGML_OP_ROPE) {
                    const int mode = op_params[2];
                    const int64_t nels = (mode & GGML_ROPE_TYPE_MROPE) ? ne[2] * 4 : ne[2];
                    std::vector<int32_t> data(nels);
                    std::uniform_int_distribution<int32_t> dist(0, ne[2] - 1);
                    for (int64_t i = 0; i < nels; i++) {
                        data[i] = dist(rng);
                    }
                    ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(int32_t));
                } else if (op == GGML_OP_MUL_MAT_ID || op == GGML_OP_ADD_ID) {
                    const int64_t n_expert = (op == GGML_OP_MUL_MAT_ID) ? sources[0].ne[2] : sources[1].ne[1];
                    for (int64_t r = 0; r < ggml_nrows(t); r++) {
                        std::vector<int32_t> data(t->ne[0]);
                        for (int32_t i = 0; i < t->ne[0]; i++) {
                            data[i] = i % n_expert;
                        }
                        std::shuffle(data.begin(), data.end(), rng);
                        ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                    }
                } else if (op == GGML_OP_SSM_SCAN) {
                    for (int64_t r = 0; r < ggml_nrows(t); r++) {
                        std::vector<int32_t> data(t->ne[0]);
                        for (int32_t i = 0; i < t->ne[0]; i++) {
                            data[i] = i;
                        }
                        std::shuffle(data.begin(), data.end(), rng);
                        ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                    }
                } else {
                    init_tensor_uniform(t);
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};


enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};

struct llama_hparams {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    static constexpr uint32_t n_layer = 1;
    uint32_t n_rot;
    uint32_t n_embd_head; // dimension of values (d_v)
    uint32_t n_ff;

    float f_norm_eps;
    float f_norm_rms_eps;

    // cparams
    static constexpr uint32_t n_ctx = 512; // user-specified context size
    static constexpr uint32_t n_ctx_orig = n_ctx;

    // batch
    int32_t n_tokens;

    // llm_build_context
    static constexpr int32_t n_kv    = 32; // size of KV cache to consider (n_kv <= n_ctx
    static constexpr int32_t kv_head = 1;  // index of where we store new KV data in the cache

    uint32_t n_embd_gqa() const { // dimension of key embeddings across all k-v heads
        return n_embd_head * n_head_kv;
    }
};

// LLM base class
struct test_llm : public test_case {
    llama_hparams hp;

protected:
    test_llm(llama_hparams hp)
        : hp(std::move(hp)) {
    }

public:
    struct ggml_tensor * llm_build_norm(
            struct ggml_context * ctx,
             struct ggml_tensor * cur,
             struct ggml_tensor * mw,
             struct ggml_tensor * mb,
                  llm_norm_type   type) {
        switch (type) {
            case LLM_NORM:     cur = ggml_norm    (ctx, cur, hp.f_norm_eps); break;
            case LLM_NORM_RMS: cur = ggml_rms_norm(ctx, cur, hp.f_norm_rms_eps); break;
        }
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cur = ggml_add(ctx, cur, mb);
        }
        return cur;
    }

    void llm_build_kv_store(
            struct ggml_context * ctx,
             struct ggml_tensor * k_l,
             struct ggml_tensor * v_l,
             struct ggml_tensor * k_cur,
             struct ggml_tensor * v_cur) {
        // compute the transposed [n_tokens, n_embd] V matrix
        struct ggml_tensor * v_cur_t = ggml_transpose(ctx, ggml_reshape_2d(ctx, v_cur, hp.n_embd_gqa(), hp.n_tokens));

        struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, k_l, hp.n_tokens*hp.n_embd_gqa(),
                (ggml_row_size(k_l->type, hp.n_embd_gqa()))*hp.kv_head);

        struct ggml_tensor * v_cache_view = ggml_view_2d(ctx, v_l, hp.n_tokens, hp.n_embd_gqa(),
                (  hp.n_ctx)*ggml_element_size(v_l),
                (hp.kv_head)*ggml_element_size(v_l));

        // important: storing RoPE-ed version of K in the KV cache!
        ggml_cpy(ctx, k_cur,   k_cache_view);
        ggml_cpy(ctx, v_cur_t, v_cache_view);
    }

    struct ggml_tensor * llm_build_kqv(
            struct ggml_context * ctx,
             struct ggml_tensor * k_l,
             struct ggml_tensor * v_l,
             struct ggml_tensor * q_cur,
             struct ggml_tensor * kq_mask,
                        float     kq_scale) {
        struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);

        struct ggml_tensor * k =
            ggml_view_3d(ctx, k_l,
                    hp.n_embd_head, hp.n_kv, hp.n_head_kv,
                    ggml_row_size(k_l->type, hp.n_embd_gqa()),
                    ggml_row_size(k_l->type, hp.n_embd_head),
                    0);

        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

        // split cached v into n_head heads
        struct ggml_tensor * v =
            ggml_view_3d(ctx, v_l,
                    hp.n_kv, hp.n_embd_head, hp.n_head_kv,
                    ggml_element_size(v_l)*hp.n_ctx,
                    ggml_element_size(v_l)*hp.n_ctx*hp.n_embd_head,
                    0);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);

        struct ggml_tensor * cur = ggml_cont_2d(ctx, kqv_merged, hp.n_embd_head*hp.n_head, hp.n_tokens);

        struct ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
        cur = ggml_mul_mat(ctx, wo, cur);

        return cur;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(hp.n_tokens);
                for (int i = 0; i < hp.n_tokens; i++) {
                    data[i] = rand() % hp.n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, hp.n_tokens * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// Llama
struct test_llama : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;
    bool fused;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "LLAMA";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    bool run_whole_graph() override { return fused; }

    test_llama(int n_tokens = 1, bool fused = false)
        : test_llm({
            /*n_vocab        =*/ 32000,
            /*n_embd         =*/ 3200,
            /*n_head         =*/ 32,
            /*n_head_kv      =*/ 32,
            /*n_rot          =*/ 100,
            /*n_embd_head    =*/ 100,
            /*n_ff           =*/ 8640,
            /*f_norm_eps     =*/ 0.f,
            /*f_norm_rms_eps =*/ 1e-5f,
            /*n_tokens       =*/ n_tokens,
        })
        , fused(fused)
    {
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            ggml_tensor * attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur = llm_build_norm(ctx, inpL, attn_norm, nullptr, LLM_NORM_RMS);

            // self-attention
            {
                ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
                ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());
                ggml_tensor * wv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx, wq, cur);
                struct ggml_tensor * Kcur = ggml_mul_mat(ctx, wk, cur);
                struct ggml_tensor * Vcur = ggml_mul_mat(ctx, wv, cur);

                Qcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f/sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

            // feed-forward network
            ggml_tensor * ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur = llm_build_norm(ctx, ffn_inp, ffn_norm, nullptr, LLM_NORM_RMS);

            ggml_tensor * ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff,   hp.n_embd);
            ggml_tensor * ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            struct ggml_tensor * tmp = ggml_mul_mat(ctx, ffn_up, cur);
            cur = ggml_mul_mat(ctx, ffn_gate, cur);
            cur = ggml_silu(ctx, cur);
            cur = ggml_mul(ctx, cur, tmp);
            cur = ggml_mul_mat(ctx, ffn_down, cur);

            cur = ggml_add(ctx, cur, ffn_inp);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur = llm_build_norm(ctx, cur, output_norm, nullptr, LLM_NORM_RMS);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_vocab);
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

// Falcon
struct test_falcon : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "FALCON";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    test_falcon(int n_tokens = 1)
        : test_llm({
            /*n_vocab        =*/ 32000,
            /*n_embd         =*/ 3200,
            /*n_head         =*/ 50,
            /*n_head_kv      =*/ 1,
            /*n_rot          =*/ 64,
            /*n_embd_head    =*/ 64,
            /*n_ff           =*/ 8640,
            /*f_norm_eps     =*/ 1e-5f,
            /*f_norm_rms_eps =*/ 0.f,
            /*n_tokens       =*/ n_tokens,
        }) {
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            // norm
            ggml_tensor * attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm = llm_build_norm(ctx, inpL, attn_norm_w, attn_norm_b, LLM_NORM);

            // self-attention
            {
                cur = attn_norm;

                ggml_tensor * wqkv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd + 2*hp.n_embd_gqa());

                cur = ggml_mul_mat(ctx, wqkv, cur);

                struct ggml_tensor * Qcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd,     hp.n_tokens, cur->nb[1], 0*sizeof(float)*(hp.n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1*sizeof(float)*(hp.n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1*sizeof(float)*(hp.n_embd + hp.n_embd_gqa())));

                Qcur = ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    hp.n_tokens);
                Kcur = ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx, Qcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, Kcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f/sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = cur;

            // feed forward
            {
                ggml_tensor * ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
                ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff, hp.n_embd);
                cur = attn_norm;
                cur = ggml_mul_mat(ctx, ffn_up, cur);
                cur = ggml_gelu(ctx, cur);
                cur = ggml_mul_mat(ctx, ffn_down, cur);
            }

            cur = ggml_add(ctx, cur, ffn_inp);

            cur = ggml_add(ctx, cur, inpL);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        ggml_tensor * output_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur = llm_build_norm(ctx, cur, output_norm, output_norm_b, LLM_NORM);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, hp.n_embd, hp.n_vocab);
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};


// ###########################################
// ## Section 3: GGML Op Test Instantiation ##
// ###########################################
static const ggml_type all_types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q1_0,
    GGML_TYPE_Q2_0,
    GGML_TYPE_MXFP4, GGML_TYPE_NVFP4,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_TQ2_0,
    // GGML_TYPE_TQ1_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
};

static const ggml_type base_types[] = {
    GGML_TYPE_F32, GGML_TYPE_F16,
    GGML_TYPE_Q8_0, // for I8MM tests
    GGML_TYPE_Q1_0,
    GGML_TYPE_Q2_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1, // for I8MM tests
    GGML_TYPE_Q4_K,
    GGML_TYPE_MXFP4, GGML_TYPE_NVFP4, // TODO: or "other"
    GGML_TYPE_IQ2_XXS
};

static const ggml_type other_types[] = {
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q1_0,
    GGML_TYPE_Q2_0,
    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_TQ2_0,
    // GGML_TYPE_TQ1_0, // TODO: implement for all backends
    GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
    GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
    GGML_TYPE_BF16,
};

#ifdef _MSC_VER
// Workaround long compile time with msvc
#pragma optimize("", off)
#endif

// Test cases for evaluation: should try to cover edge cases while using small input sizes to keep the runtime low
static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    // unary ops
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int v : {0, 1}) {
            for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
                if (op == GGML_UNARY_OP_XIELU) {
                    continue; // need extra params, separate test
                }
                test_cases.emplace_back(new test_unary((ggml_unary_op) op, type, { 128, 2, 2, 2 }, v));
                test_cases.emplace_back(new test_unary((ggml_unary_op) op, type, { 5, 7, 11, 13 }, v));
            }
        }
    }

    // fused relu + sqr (squared ReLU)
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        test_cases.emplace_back(new test_relu_sqr(type, { 128, 2, 2, 2 }));
        test_cases.emplace_back(new test_relu_sqr(type, { 5, 7, 11, 13 }));
    }

    // fused unary + mul (gated activations that are not expressed as GGML_OP_GLU)
    for (ggml_unary_op op : { GGML_UNARY_OP_SILU, GGML_UNARY_OP_SIGMOID, GGML_UNARY_OP_SOFTPLUS }) {
        for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
            for (bool swap : { false, true }) {
                test_cases.emplace_back(new test_unary_mul(op, type, { 128, 2, 2, 2 }, swap));
            }
            test_cases.emplace_back(new test_unary_mul(op, type, { 5, 7, 11, 13 }));
            test_cases.emplace_back(new test_unary_mul(op, type, { 128, 2, 2, 2 }, false, "pad_unary"));
            // a view only stays out from between the two ops when the unary result is second
            test_cases.emplace_back(new test_unary_mul(op, type, { 128, 2, 2, 2 }, true, "pad_other"));
            test_cases.emplace_back(new test_unary_mul(op, type, { 128, 2, 2, 2 }, true, "halves"));
            test_cases.emplace_back(new test_unary_mul(op, type, { 128, 2, 2, 2 }, false, "packed", "consumer"));
            // must not fuse
            test_cases.emplace_back(new test_unary_mul(op, type, { 128, 2, 2, 2 }, false, "strided_dim1"));
            test_cases.emplace_back(new test_unary_mul(op, type, { 128, 2, 2, 2 }, false, "bcast"));
            test_cases.emplace_back(new test_unary_mul(op, type, { 128, 2, 2, 2 }, false, "packed", "reuse"));
        }
    }

    // qwen4exp gated activations at the production geometry over the draft-length range
    // (GGML_VK_FUSE_SIGMOID_MUL=0 turns the Vulkan fusion off)
    for (int64_t n : { 1, 2, 3, 4, 5, 8 }) {
        test_cases.emplace_back(new test_unary_mul(GGML_UNARY_OP_SIGMOID, GGML_TYPE_F32, { 10240, n, 1, 1 }, true));  // build_hc_mix gate
        test_cases.emplace_back(new test_unary_mul(GGML_UNARY_OP_SIGMOID, GGML_TYPE_F32, {  6144, n, 1, 1 }, true));  // attention output gate
        test_cases.emplace_back(new test_unary_mul(GGML_UNARY_OP_SIGMOID, GGML_TYPE_F32, {   128, 48, n, 1 }, true)); // build_norm_gated
        test_cases.emplace_back(new test_unary_mul(GGML_UNARY_OP_SIGMOID, GGML_TYPE_F32, {  2560, n, 1, 1 }, true, "bcast_unary")); // shared-expert gate
    }

    // SNAKE activation fusion: x + sin(a*x)^2 * inv_b
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        test_cases.emplace_back(new test_snake_fuse(type, {   5,   7, 1, 1}));   // primes sub-block
        test_cases.emplace_back(new test_snake_fuse(type, {  33,  32, 1, 1}));   // boundary
        test_cases.emplace_back(new test_snake_fuse(type, {1025,  13, 1, 1}));   // large prime, grid-stride
        test_cases.emplace_back(new test_snake_fuse(type, { 128,  16, 1, 1}));   // power-of-two
        test_cases.emplace_back(new test_snake_fuse(type, { 256, 192, 1, 1}));   // BigVGAN-ish
        // higher-rank shapes: matcher must reject fusion, fallback to naive chain
        test_cases.emplace_back(new test_snake_fuse(type, {  64,  32, 2, 1}));   // ne[2] > 1
        test_cases.emplace_back(new test_snake_fuse(type, {  64,  32, 1, 2}));   // ne[3] > 1
        test_cases.emplace_back(new test_snake_fuse(type, {  64,  32, 2, 3}));   // ne[2] > 1 and ne[3] > 1
    }

    // qwen4exp hyper-connection combine fusion, at the production geometry (n_embd 2560, hc 4)
    // over the draft-length range plus the n_outputs=1 tail, and off-geometry shapes
    for (int64_t nt : { 1, 2, 3, 4, 5, 8 }) {
        test_cases.emplace_back(new test_hc_combine_fuse(2560, 4, nt));
    }
    test_cases.emplace_back(new test_hc_combine_fuse(31, 4, 17));     // non-power-of-two row
    test_cases.emplace_back(new test_hc_combine_fuse(2560, 4, 1));    // n_outputs == 1
    test_cases.emplace_back(new test_hc_combine_fuse(1, 4, 5));       // degenerate row
    test_cases.emplace_back(new test_hc_combine_fuse(2560, 1, 5));    // hc == 1
    test_cases.emplace_back(new test_hc_combine_fuse(257, 3, 7));     // all primes

    // [TAG_HC_MIX_TAIL] qwen4exp hc_mix tail: the build_hc_mix chain against the CPU, and the two new fusions
    // (multi_add_scale, scale_silu) bit for bit against the unfused chain on the same backend - production
    // geometry (n_embd 2560, hc 4, rank 320) at decode / draft / prefill-ubatch token counts, then hc == 1,
    // hc == 2 (one ADD: no multi_add), odd hc and off-geometry rows
    for (bool exact : { false, true }) {
        for (int64_t nt : { 1, 4, 5, 2048 }) {
            test_cases.emplace_back(new test_hc_mix_tail_fuse(2560, 4, nt, 320, exact));
        }
        test_cases.emplace_back(new test_hc_mix_tail_fuse(2560, 1, 5, 320, exact));    // hc == 1
        test_cases.emplace_back(new test_hc_mix_tail_fuse(2560, 2, 5, 320, exact));    // hc == 2
        test_cases.emplace_back(new test_hc_mix_tail_fuse(2560, 3, 5, 320, exact));    // odd hc
        test_cases.emplace_back(new test_hc_mix_tail_fuse(2560, 5, 4, 320, exact));    // odd hc, 4 adds
        test_cases.emplace_back(new test_hc_mix_tail_fuse(257, 3, 7, 31, exact));      // all primes
        test_cases.emplace_back(new test_hc_mix_tail_fuse(31, 4, 17, 5, exact));       // non-power-of-two row
    }
    // [TAG_HC_MIX_TAIL] the QSA pooled-key mean (--qsa-ratio 4): pooled-cache dirty blocks, a full recompute, np2 streams,
    // r = 2 (a single ADD: not fused) and r = 8
    test_cases.emplace_back(new test_qsa_pool_mean_fuse(128, 4, 1, 1));
    test_cases.emplace_back(new test_qsa_pool_mean_fuse(128, 4, 512, 1));
    test_cases.emplace_back(new test_qsa_pool_mean_fuse(128, 4, 65536, 1));
    test_cases.emplace_back(new test_qsa_pool_mean_fuse(128, 4, 4096, 2));
    test_cases.emplace_back(new test_qsa_pool_mean_fuse(128, 2, 512, 1));
    test_cases.emplace_back(new test_qsa_pool_mean_fuse(128, 8, 333, 1));

    // [TAG_CONV_INPUT_SPLIT] the history tails (GDN sc 3, PLE sc 9; n_rs_seq == 0 and rollback planes)
    for (int64_t sc : { 3, 9 }) {
        for (int64_t n_t : { 1, 4, 5, 2044, 2048 }) {
            for (int64_t n_s : { 1, 2 }) {
                for (int64_t kw : { 0, 1, 5 }) {
                    test_cases.emplace_back(new test_conv_tail(sc, 10240, n_t, n_s, kw));
                }
            }
        }
    }

    // [TAG_CONV_INPUT_SPLIT] the GDN conv input without the concat (qwen4exp: 10240 channels, kernel 4) at
    // decode-verify / prefill-ubatch token counts, 2 sequences, the three conv tails, odd channel counts
    for (bool exact : { true, false }) {
        for (int extra : { 0, 1, 2 }) {
            for (int64_t n_t : { 1, 3, 4, 5, 2044, 2048 }) {
                test_cases.emplace_back(new test_concat_ssm_conv_fuse(10240, n_t, 1, extra, exact));
            }
            test_cases.emplace_back(new test_concat_ssm_conv_fuse(10240, 64, 2, extra, exact));
            test_cases.emplace_back(new test_concat_ssm_conv_fuse(1027, 17, 3, extra, exact));
            // sliding-window kernel (8 tokens per thread, 32 per workgroup, 64 channels): fewer tokens than the
            // window, ragged token tiles, ragged channels, two sequences
            for (int64_t n_t : { 2, 8, 9, 31, 33, 300 }) {
                test_cases.emplace_back(new test_concat_ssm_conv_fuse(130, n_t, 2, extra, exact));
            }
            test_cases.emplace_back(new test_concat_ssm_conv_fuse(10240, 257, 1, extra, exact));
        }
    }

    // qwen4exp Gated-DeltaNet decay pre-chain fusion, at the production geometry
    // (num_v_heads 48) over the draft-length range, plus off-geometry shapes and the
    // veto case where the softplus has a second consumer
    for (int64_t nt : { 1, 2, 3, 4, 5, 8 }) {
        test_cases.emplace_back(new test_add_softplus_mul_fuse(48, nt, 1));
    }
    test_cases.emplace_back(new test_add_softplus_mul_fuse(48, 2048, 1));      // prefill ubatch
    test_cases.emplace_back(new test_add_softplus_mul_fuse(48, 5, 4));         // n_seqs > 1
    test_cases.emplace_back(new test_add_softplus_mul_fuse(1, 5, 1));          // degenerate row
    test_cases.emplace_back(new test_add_softplus_mul_fuse(31, 17, 1));        // non-power-of-two row
    test_cases.emplace_back(new test_add_softplus_mul_fuse(257, 7, 3));        // all primes
    test_cases.emplace_back(new test_add_softplus_mul_fuse(48, 5, 1, true));   // extra consumer: no fusion

    test_cases.emplace_back(new test_dsv4_hc_comb(1, 1));
    test_cases.emplace_back(new test_dsv4_hc_comb(17, 4));
    test_cases.emplace_back(new test_dsv4_hc_comb(257, 8));
    test_cases.emplace_back(new test_dsv4_hc_comb(17, 20));

    test_cases.emplace_back(new test_dsv4_hc_pre(1, 1));
    test_cases.emplace_back(new test_dsv4_hc_pre(31, 17));
    test_cases.emplace_back(new test_dsv4_hc_pre(128, 257));
    test_cases.emplace_back(new test_dsv4_hc_pre(4096, 21));

    test_cases.emplace_back(new test_dsv4_hc_post(1, 1));
    test_cases.emplace_back(new test_dsv4_hc_post(31, 17));
    test_cases.emplace_back(new test_dsv4_hc_post(128, 257));
    test_cases.emplace_back(new test_dsv4_hc_post(4096, 21));

    // glu ops
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int v : {0, 1}) {
            for (int op = 0; op < GGML_GLU_OP_COUNT; op++) {
                if (op == GGML_GLU_OP_SWIGLU_OAI) {
                    // SWIGLU_OAI is handled separately
                    continue;
                }

                for (bool swapped : {false, true}) {
                    test_cases.emplace_back(new test_glu((ggml_glu_op) op, type, { 128, 2, 2, 2 }, v, swapped));
                    test_cases.emplace_back(new test_glu((ggml_glu_op) op, type, { 5, 7, 11, 13 }, v, swapped));
                }

                test_cases.emplace_back(new test_glu_split((ggml_glu_op) op, type, { 128, 2, 2, 2 }, v));
                test_cases.emplace_back(new test_glu_split((ggml_glu_op) op, type, { 5, 7, 11, 13 }, v));
            }
        }
    }

    for (int v : {0, 1}) {
        for (float alpha : {.5f, 1.702f}) {
            for (float limit : {2.0f, 7.0f}) {
                test_cases.emplace_back(new test_swiglu_oai(GGML_TYPE_F32, { 128, 2, 2, 2 }, v, alpha, limit));
            }
        }
    }

    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q4_0}) {
        test_cases.emplace_back(new test_get_rows(type, 300*256,   5,         4,   1,   2, false));
        test_cases.emplace_back(new test_get_rows(type,     256,   80000, 70000,   2,   1, false));
        test_cases.emplace_back(new test_get_rows(type,     256,   5,         4, 700, 100, false));
    }

    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 1, 8, 2, 1, 1, false));
    for (ggml_type type : all_types) {
        for (int b : {1, 7}) {
            for (bool v : {false, true}) {
                test_cases.emplace_back(new test_get_rows(type, 256, 5, 4, b, 1, v));
            }
        }
    }
    for (int b : {1, 7}) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 256, 5, 4, b, 1, v));
        }
    }
    // the QSA block-selection gathers (qwen4exp_select_complete_blocks): 1-wide score / block-id rows per query and 4-wide
    // block-cell rows (GET_ROWS(1, 512, 2048) and (4, 1048576) at a 2048-token ubatch), plus bf16 and odd widths
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 1, 8704, 512, 2048, 1, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 1, 512, 512, 2048, 1, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 4, 8704, 1048576, 1, 1, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_BF16, 4, 300, 777, 3, 2, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F16, 5, 300, 777, 3, 2, true));

    // [TAG_QSA_GATHER_Q8] type-preserving gathers (ggml_get_rows_type): q8_0 -> q8_0 and f16 -> f16
    for (int b : {1, 7}) {
        for (bool v : {false, true}) {
            for (int n : {64, 256, 512}) {
                test_cases.emplace_back(new test_get_rows_type(GGML_TYPE_Q8_0, n, 5, 4, b, 1, v));
            }
            for (int n : {1, 2, 256}) {
                test_cases.emplace_back(new test_get_rows_type(GGML_TYPE_F16, n, 5, 4, b, 1, v));
            }
        }
    }
    test_cases.emplace_back(new test_get_rows_type(GGML_TYPE_Q8_0,  32,  5, 4, 1, 1, false)); // one block: 34 B rows (Vulkan copies 4-byte words -> not supported there)
    test_cases.emplace_back(new test_get_rows_type(GGML_TYPE_Q8_0, 512,  7, 4, 3, 2, false)); // 4-d source
    // the QSA gather at the deployed shape: 512-value cells (2 heads x 256) of a 196 608-cell layer,
    // 2 051 cells for each of 6 queries (an MTP verify round), one stream
    test_cases.emplace_back(new test_get_rows_type(GGML_TYPE_Q8_0, 512, 196608, 2051*6, 1, 1, false));
    // its mask: 1-element f16 rows of [1, n_kv, n_q], width rows per query
    test_cases.emplace_back(new test_get_rows_type(GGML_TYPE_F16,    1, 196608, 2051, 6, 1, false));
    // the M4.2 chunk mask: whole f16 mask rows [n_kv] of a chunk of queries
    test_cases.emplace_back(new test_get_rows_type(GGML_TYPE_F16, 65536,   300,  228, 1, 1, false));

    test_cases.emplace_back(new test_get_rows_back(GGML_TYPE_F32, 1, 8, 2, 1, false));
    test_cases.emplace_back(new test_get_rows_back(GGML_TYPE_F32, 1, 70000, 4, 1, false)); // row count > CUDA grid-y limit (65535)
    for (ggml_type type : all_types) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_get_rows_back(type, 256, 5, 4, 1, v));
        }
    }
    for (bool v : {false, true}) {
        test_cases.emplace_back(new test_get_rows_back(GGML_TYPE_I32, 256, 5, 4, 1, v));
    }

    test_cases.emplace_back(new test_set_rows(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_I64, { 1, 8, 1, 3 }, { 1, 1 }, 2, false));
    test_cases.emplace_back(new test_set_rows(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_I32, { 1, 8, 1, 3 }, { 1, 1 }, 2, false));
    test_cases.emplace_back(new test_set_rows(GGML_TYPE_F32, GGML_TYPE_Q8_0, GGML_TYPE_I32, { 256, 5, 1, 3 }, { 1, 1, }, 1, false));
    for (ggml_type src_type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (ggml_type type : all_types) {
            for (int b : {1, 7}) {
                for (bool v : {false, true}) {
                    test_cases.emplace_back(new test_set_rows(src_type, type, GGML_TYPE_I64, { 256, 5,  b, 3 }, { 1, 1, }, 1, v));
                    test_cases.emplace_back(new test_set_rows(src_type, type, GGML_TYPE_I64, { 256, 11, 1, b }, { 2, 3, }, 7, v));

                    test_cases.emplace_back(new test_set_rows(src_type, type, GGML_TYPE_I64, { 3*ggml_blck_size(type), 3, b, 1 }, { 2, 3, }, 2, v));

                    if (ggml_blck_size(type) == 1) {
                        test_cases.emplace_back(new test_set_rows(src_type, type, GGML_TYPE_I64, { 31, 3, b, 1 }, { 2, 3, }, 2, v));
                        test_cases.emplace_back(new test_set_rows(src_type, type, GGML_TYPE_I64, { 33, 5, 1, b }, { 2, 3, }, 1, v));
                    }
                }
            }
        }
    }
    test_cases.emplace_back(new test_set_rows(GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_I64, { 1, 8, 1, 3 }, { 1, 1 }, 2, false));
    test_cases.emplace_back(new test_set_rows(GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_I32, { 1, 8, 1, 3 }, { 1, 1 }, 2, false));
    test_cases.emplace_back(new test_set_rows(GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_I64, { 1, 8, 1, 3 }, { 1, 1 }, 2, true));
    test_cases.emplace_back(new test_set_rows(GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_I32, { 1, 8, 1, 3 }, { 1, 1 }, 2, true));

    for (int mode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_MROPE, GGML_ROPE_TYPE_VISION }) {
        for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            for (int ne2 : {1, 8, 512}) {
                test_cases.emplace_back(new test_rope_set_rows(type, GGML_TYPE_I64, { 128, 32, ne2, 1 }, mode));
                test_cases.emplace_back(new test_rope_set_rows(type, GGML_TYPE_I64, { 128, 32, ne2, 3 }, mode));
            }
        }
    }

    for (ggml_type type_input : {GGML_TYPE_F32}) {
        for (ggml_op_pool pool_type : {GGML_OP_POOL_AVG, GGML_OP_POOL_MAX}) {
            for (int k0 : {1, 3}) {
                for (int k1 : {1, 3}) {
                    for (int s0 : {1, 2}) {
                        for (int s1 : {1, 2}) {
                            for (int p0 : {0, 1}) {
                                for (int p1 : {0, 1}) {
                                    test_cases.emplace_back(new test_pool2d(pool_type, type_input, {10, 10, 3, 1}, k0, k1, s0, s1, p0, p1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type_input : {GGML_TYPE_F32}) {
        for (ggml_op_pool pool_type : {GGML_OP_POOL_AVG, GGML_OP_POOL_MAX}) {
            for (int k0 : {1, 2, 3}) {
                for (int s0 : {1, 2, 3}) {
                    for (int p0 : {0, 1, 2, 3}) {
                        test_cases.emplace_back(new test_pool1d(pool_type, type_input, { 10,  3, 2, 1 }, k0, s0, p0));
                        test_cases.emplace_back(new test_pool1d(pool_type, type_input, { 11,  1, 3, 2 }, k0, s0, p0));
                        test_cases.emplace_back(new test_pool1d(pool_type, type_input, { 128, 2, 1, 3 }, k0, s0, p0));
                    }
                }
            }
        }
    }

#if 0
    // >4GB im2col destination. Too slow to run by default.
    // Test cases taken from Wan2.1 T2V 1.3B.
    test_cases.emplace_back(new test_im2col   (GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {832, 480, 192, 4}, {3, 3, 192, 96}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col_3d(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {834, 482, 6, 96},  {3, 3,3, 9216}, 96, 1, 1, 1, 0, 0, 0, 1, 1, 1, false));
#endif

    // im2col 1D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {3000, 384, 1, 1}, {3, 384,  384, 1}, 1, 0, 1, 0, 1, 0, false));
    for (int s0 : {1, 3}) {
        for (int p0 : {0, 3}) {
            for (int d0 : {1, 3}) {
                test_cases.emplace_back(new test_im2col(
                    GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {20, 2, 2, 1}, {3, 2, 2, 1},
                    s0, 0, p0, 0, d0, 0, false));
            }
        }
    }

    // im2col 2D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F16));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    for (int s0 : {1, 3}) {
        for (int s1 : {1, 3}) {
            for (int p0 : {0, 3}) {
                for (int p1 : {0, 3}) {
                    for (int d0 : {1, 3}) {
                        for (int d1 : {1, 3}) {
                            test_cases.emplace_back(new test_im2col(
                                GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {20, 20, 2, 2}, {3, 3, 2, 2},
                                s0, s1, p0, p1, d0, d1, true));
                        }
                    }
                }
            }
        }
    }

    // extra tests for im2col 2D
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 1, 32}, {3, 3, 1, 32}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 2, 32}, {3, 3, 2, 32}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 1, 1024}, {3, 3, 1, 1024}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 2, 1024}, {3, 3, 2, 1024}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 1, 2048}, {3, 3, 1, 2048}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 2, 2048}, {3, 3, 2, 2048}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 1, 2560}, {3, 3, 1, 2560}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {12, 12, 2, 2560}, {3, 3, 2, 2560}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {5, 5, 1, 32}, {3, 4, 1, 32}, 1, 1, 0, 0, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {2, 2, 1536, 729}, {2, 2, 1536, 4096}, 1, 1, 0, 0, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {128, 128, 1, 2}, {32, 33, 1, 2}, 1, 1, 1, 1, 1, 1, true));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {128, 128, 2, 1}, {33, 34, 2, 1}, 1, 1, 1, 1, 1, 1, true));

    // im2col 3D
    test_cases.emplace_back(new test_im2col_3d(GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col_3d(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col_3d(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    for (int s0 : {1, 3}) {
        for (int s1 : {1, 3}) {
            for (int s2 : {1, 3}) {
                for (int p0 : {0, 3}) {
                    for (int p1 : {0, 3}) {
                        for (int p2 : {0, 3}) {
                            for (int d0 : {1, 3}) {
                                for (int d1 : {1, 3}) {
                                    for (int d2 : {1, 3}) {
                                        for (int IC : {1, 3}) {
                                            for (bool v : {false, true}) {
                                                test_cases.emplace_back(new test_im2col_3d(
                                                    GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, {20, 20, 10, 3}, {3, 3, 3, 3},
                                                    IC, s0, s1, s2, p0, p1, p2, d0, d1, d2, v));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

// Conv_2D test cases
#ifdef DETAILED_TESTS
    // Probably we do not have enough time to execute these in the pipeline.
    uint32_t iwh_idx  = 0;
    uint32_t kwh_idx  = 1;
    uint32_t Cout_idx = 2;
    uint32_t Cin_idx  = 3;
    uint32_t B_idx    = 4;

    std::vector<std::array<int, 5>> cases = {
  //{IWH, KWH, Cout, Cin, B}
  // K=CRS=NPQ=4096 conv_2d matmul performance
        {19,   4, 4096, 256, 16},
 // K=128, CRS=128, NPQ=4096
        { 19,  4, 128,  8,   16},
 // K=130, CRS=128, NPQ=4096
        { 19,  4, 130,  8,   16},
 // Edge case: K x CRS is small
        { 19,  2, 4,    4,   16},
 // A ConvNet's first layer
        { 224, 3, 8,    3,   1 },
 // A ConvNet's first layer with 2x2 convolution, and 1 channel
        { 224, 2, 8,    1,   1 },
 // A ConvNet's first layer with 2x2 convolution, and 1 channel, several images in the batch
        { 224, 2, 8,    1,   8 },
 // A middle layer of a ConvNet
        { 58,  3, 64,   32,  1 },
 // A middle layer of a ConvNet, several images in the batch
        { 58,  3, 64,   32,  8 },
 // A deep layer of a ConvNet, several images in the batch
        { 16,  3, 256,  128, 8 }
    };

    for (auto kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (auto act_case : cases) {
            test_cases.emplace_back(new test_conv_2d(
                { act_case[iwh_idx], act_case[iwh_idx], act_case[Cin_idx], act_case[B_idx] },
                { act_case[kwh_idx], act_case[kwh_idx], act_case[Cin_idx], act_case[Cout_idx] },
                kernel_type, 1, 1, 0, 0, 1, 1, false));  // bool cwhn = false
            test_cases.emplace_back(new test_conv_2d(
                { act_case[iwh_idx], act_case[iwh_idx], act_case[Cin_idx], act_case[B_idx] },
                { act_case[kwh_idx], act_case[kwh_idx], act_case[Cin_idx], act_case[Cout_idx] },
                kernel_type, 1, 1, 0, 0, 1, 1, true));  // bool cwhn = true
        }
    }
#endif

    // CONV_2D:
    auto calc_conv_output_size = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
        return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
    };

    //uint32_t s0 = 3;
    uint32_t s1 = 5;
    uint32_t p0 = 5;
    //uint32_t p1 = 2;
    uint32_t d0 = 2;
    uint32_t d1 = 4;

    for (uint32_t s0 : { 1, 3 }) {
        for (uint32_t p1 : { 2, 5 }) {
            for (uint32_t Cin : { 1, 25 }) {
                for (uint32_t Cout : { 1, 12 }) {
                    for (uint32_t KH : { 1, 2, 3, 11 }) {
                        for (uint32_t KW : { 1, 2, 3, 11 }) {
                            for (uint32_t H : { 1, 133 }) {
                                for (uint32_t W : { 1, 141 }) {
                                    if (calc_conv_output_size(W, KW, s0, p0, d0) > 0 &&
                                        calc_conv_output_size(H, KH, s1, p1, d1) > 0) {
                                        for (auto kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                                            test_cases.emplace_back(new test_conv_2d(
                                                { W, H, Cin, 2 }, { KW, KH, Cin, Cout }, kernel_type, s0, s1, p0, p1, d0, d1, false)); // bool cwhn = false
                                            test_cases.emplace_back(new test_conv_2d(
                                                { W, H, Cin, 2 }, { KW, KH, Cin, Cout }, kernel_type, s0, s1, p0, p1, d0, d1, true));  // bool cwhn = true
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (auto kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        test_cases.emplace_back(new test_conv_2d({ 256, 256, 192, 1 }, { 3, 3, 192, 96 }, kernel_type, 1, 1, 1, 1, 1, 1, false)); // bool cwhn = false
        test_cases.emplace_back(new test_conv_2d({ 256, 256, 192, 1 }, { 3, 3, 192, 96 }, kernel_type, 1, 1, 1, 1, 1, 1, true));  // bool cwhn = true
    }

    // sycl backend will limit task global_range < MAX_INT
    // test cases for 2D im2col with large input W and H (occurs in stable-diffusion)
    // however these cases need to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // these cases are verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));
    // test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {1024, 1024, 256, 1}, {3, 3, 256, 1}, 1, 1, 1, 1, 1, 1, true));

    test_cases.emplace_back(new test_conv_2d_dw({17, 34, 9, 1}, {3, 3, 1, 9},  GGML_TYPE_F32, 1, 0, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({17, 34, 9, 1}, {3, 3, 1, 9},  GGML_TYPE_F32, 1, 0, 1, true));
    test_cases.emplace_back(new test_conv_2d_dw({32, 8, 64, 1}, {3, 3, 1, 64}, GGML_TYPE_F32, 2, 1, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({32, 8, 64, 1}, {3, 3, 1, 64}, GGML_TYPE_F32, 2, 1, 1, true));

    test_cases.emplace_back(new test_conv_2d_dw({17, 34, 9, 1}, {3, 3, 1, 9},  GGML_TYPE_F16, 1, 0, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({17, 34, 9, 1}, {3, 3, 1, 9},  GGML_TYPE_F16, 1, 0, 1, true));
    test_cases.emplace_back(new test_conv_2d_dw({32, 8, 64, 1}, {3, 3, 1, 64}, GGML_TYPE_F16, 2, 1, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({32, 8, 64, 1}, {3, 3, 1, 64}, GGML_TYPE_F16, 2, 1, 1, true));

    // CONV_3D
    auto calc_conv_output_size_3d = [](int64_t ins, int64_t ks, int s, int p, int d) -> int64_t {
        return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
    };

    for (ggml_type kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (int N : {1, 2}) {
            for (int IC : {1, 3}) {
                for (int OC : {1, 4}) {
                    for (int s0 : {1, 2}) {
                        for (int p1 : {0, 1}) {
                            for (int d2 : {1, 2}) {
                                int64_t IW = 20, IH = 22, ID = 18;
                                int64_t KW = 3,  KH = 3,  KD = 3;
                                int s1 = s0, s2 = s0;
                                int p0 = p1, p2 = p1;
                                int d0 = d2, d1 = d2;

                                if (calc_conv_output_size_3d(IW, KW, s0, p0, d0) <= 0 ||
                                    calc_conv_output_size_3d(IH, KH, s1, p1, d1) <= 0 ||
                                    calc_conv_output_size_3d(ID, KD, s2, p2, d2) <= 0) {
                                    continue;
                                }
                                test_cases.emplace_back(new test_conv_3d(
                                    N, IC, ID, IH, IW,
                                    OC, KD, KH, KW,
                                    s0, s1, s2, p0, p1, p2, d0, d1, d2,
                                    kernel_type));

                                // Asymmetric kernel and params
                                int64_t asym_KW = 5, asym_KH = 1, asym_KD = 3;
                                int asym_s0 = 2, asym_s1 = 1, asym_s2 = 1;
                                int asym_p0 = 2, asym_p1 = 0, asym_p2 = 1;
                                int asym_d0 = 1, asym_d1 = 1, asym_d2 = 2;

                                if (calc_conv_output_size_3d(IW, asym_KW, asym_s0, asym_p0, asym_d0) <= 0 ||
                                    calc_conv_output_size_3d(IH, asym_KH, asym_s1, asym_p1, asym_d1) <= 0 ||
                                    calc_conv_output_size_3d(ID, asym_KD, asym_s2, asym_p2, asym_d2) <= 0) {
                                    continue;
                                }
                                test_cases.emplace_back(new test_conv_3d(
                                    N, IC, ID, IH, IW,
                                    OC, asym_KD, asym_KH, asym_KW,
                                    asym_s0, asym_s1, asym_s2, asym_p0, asym_p1, asym_p2, asym_d0, asym_d1, asym_d2,
                                    kernel_type));
                            }
                        }
                    }
                }
            }
        }
        // Case with kernel size 1
        test_cases.emplace_back(new test_conv_3d(1, 4, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, kernel_type));
    }

    for(uint32_t Cout : {1, 9}){
        for(uint32_t Cin : {1, 7}){
            for(uint32_t K : {1, 3, 1337}){
                for(uint32_t L : {1, 2, 13}){
                    for(uint32_t s0: {1, 2, 3}){
                        test_cases.emplace_back(new test_conv_transpose_1d({L,Cin,1,1}, {K,Cout,Cin,1}, s0, 0, 1));
                    }
                }
            }
        }
    }

    test_cases.emplace_back(new test_conv_transpose_1d());
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 3, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,1,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({2,1,1,1}, {3,1,1,1}, 1, 0, 1));

    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        // ConvTranspose1d expressed as mul_mat + col2im (DAC decoder upsampling)
        test_cases.emplace_back(new test_col2im_1d(type, 16, 32, 197, 8, 0)); // kernel = 2*stride
        test_cases.emplace_back(new test_col2im_1d(type, 4, 3, 7, 2, 0));
        test_cases.emplace_back(new test_col2im_1d(type, 1, 5, 13, 1, 0));    // stride 1, no overlap
        test_cases.emplace_back(new test_col2im_1d(type, 6, 4, 11, 3, 1));    // with cropping
        test_cases.emplace_back(new test_col2im_1d(type, 2, 3, 9, 3, 0));     // kernel < stride, gap positions are zeroed
        test_cases.emplace_back(new test_col2im_1d(type, 5, 4, 11, 2, 0));    // kernel not a multiple of stride, alternating overlap
        test_cases.emplace_back(new test_col2im_1d(type, 8, 4, 13, 4, 2));    // padding = stride/2 (DAC causal cropping)
        test_cases.emplace_back(new test_col2im_1d(type, 4, 3, 1, 2, 0));     // single column, pure kernel unfold
        test_cases.emplace_back(new test_col2im_1d(type, 16, 1, 197, 8, 0));   // OC = 1, mono output stage
        test_cases.emplace_back(new test_col2im_1d(type, 1, 5, 13, 3, 0));     // K = 1 with stride > 1, sparse scatter
        test_cases.emplace_back(new test_col2im_1d(type, 8, 2, 3, 2, 5));      // cropping eats most of the signal, T_out = 2
    }

    for (ggml_type kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        test_cases.emplace_back(new test_conv_transpose_2d({3, 2, 3, 1}, {2, 2, 1, 3}, 1, kernel_type));
        test_cases.emplace_back(new test_conv_transpose_2d({10, 10, 9, 1}, {3, 3, 1, 9}, 2, kernel_type));
        test_cases.emplace_back(new test_conv_transpose_2d({129, 63, 35, 1}, {3, 3, 48, 35}, 1, kernel_type));
    }

    test_cases.emplace_back(new test_count_equal(GGML_TYPE_F32, {4,  500, 1, 1}));
    test_cases.emplace_back(new test_count_equal(GGML_TYPE_F32, {4, 5000, 1, 1}));

    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {32,    1, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {32,  513, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {100,  10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {1024, 10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {1024, 12, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {2000, 10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {5438,  3, 1, 1}));

    for (int ne3 : {1, 3}) { // CUDA backward pass only supports ne3 == 1
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {1, 1, 1, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {2, 1, 1, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {1, 2, 1, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {1, 1, 2, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 5, 4, ne3}, {1, 1, 1, 2}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_F16, {10, 5, 4, ne3}, {2, 1, 1, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_I32, {10, 5, 4, ne3}, {2, 1, 1, 1}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_I16, {10, 5, 4, ne3}, {1, 1, 1, 2}));
        test_cases.emplace_back(new test_repeat(GGML_TYPE_BF16, {10, 5, 4, ne3}, {2, 1, 1, 1}));
    }

    for (bool view : {false, true}) {
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {1, 1, 1, 1}, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {2, 1, 1, 1}, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {1, 2, 1, 1}, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {1, 1, 2, 1}, view));
        test_cases.emplace_back(new test_repeat_back(GGML_TYPE_F32, {8, 6, 4, 2}, {1, 1, 1, 2}, view));
    }

    test_cases.emplace_back(new test_dup(GGML_TYPE_F32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, {10, 10, 5, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, {10, 10, 5, 1}, {0, 2, 1, 3})); // dup by rows
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, {10, 10, 5, 1}, {1, 0, 2, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, {10, 10, 5, 1}, {1, 0, 2, 3})); // dup dst not-contiguous
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, {10,  8, 3, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, {10,  8, 3, 1}, {1, 2, 0, 3}));

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        test_cases.emplace_back(new test_set(GGML_TYPE_F32, GGML_TYPE_F32, {6, 5, 4, 3}, dim, false));
        test_cases.emplace_back(new test_set(GGML_TYPE_F32, GGML_TYPE_F32, {6, 5, 4, 3}, dim, true));
    }

    for (int dim = 1; dim < GGML_MAX_DIMS; ++dim) {
        test_cases.emplace_back(new test_set(GGML_TYPE_I32, GGML_TYPE_I32, {6, 5, 4, 3}, dim, false));
        test_cases.emplace_back(new test_set(GGML_TYPE_I32, GGML_TYPE_I32, {6, 5, 4, 3}, dim, true));
    }

    // same-type copy
    for (ggml_type type : all_types) {
        const auto nk = ggml_blck_size(type);

        for (int k = 1; k < 4; ++k) {
            test_cases.emplace_back(new test_cpy(type, type, {k*nk, 2, 3, 4}));
            test_cases.emplace_back(new test_cpy(type, type, {k*nk, 2, 3, 4}, {-1,-1,-1,-1}, {0, 2, 1, 3}));
            test_cases.emplace_back(new test_cpy(type, type, {k*nk, 2, 3, 4}, {-1,-1,-1,-1}, {0, 3, 1, 2}, {0, 2, 1, 3}));
        }
    }

    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : all_types) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 4, 4, 4}));
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {-1,-1,-1,-1}, {0, 2, 1, 3})); // cpy by rows
        }
    }
    for (ggml_type type_src : all_types) {
        for (ggml_type type_dst : {GGML_TYPE_F32}) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 4, 4, 4}));
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {-1,-1,-1,-1}, {0, 2, 1, 3})); // cpy by rows
        }
    }
    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {-1,-1,-1,-1}, {1, 0, 2, 3})); // cpy not-contiguous
        }
    }
    // quant block count not a multiple of the kernel block size
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_Q4_0, {96, 1, 1, 1}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_Q4_0, GGML_TYPE_F32, {96, 1, 1, 1}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_I32, {256, 2, 3, 4}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_I32, {256, 2, 3, 4}, {-1,-1,-1,-1}, {1, 0, 2, 3}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_I32, GGML_TYPE_F32, {256, 2, 3, 4}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_I32, GGML_TYPE_F32, {256, 2, 3, 4}, {-1,-1,-1,-1}, {1, 0, 2, 3}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F16, GGML_TYPE_F16, {256, 4, 3, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {256, 4, 3, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {256, 4, 3, 3}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_BF16, GGML_TYPE_BF16, {256, 4, 3, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F16, GGML_TYPE_F16, {256, 4, 1, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {256, 4, 1, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_BF16, GGML_TYPE_BF16, {256, 4, 1, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_I32, GGML_TYPE_I32, {256, 4, 1, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_I32, GGML_TYPE_I32, {256, 1, 4, 1}, {-1,-1,-1,-1}, {1, 2, 0, 3}, {0, 0, 0, 0}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {256, 1, 4, 1}, {-1,-1,-1,-1}, {1, 2, 0, 3}, {0, 0, 0, 0}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {2, 2097121, 1, 1}, {-1,-1,-1,-1}, {1, 0, 2, 3}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {2, 2, 524281, 1}, {-1,-1,-1,-1}, {1, 0, 2, 3}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {128, 2, 3, 1}, {128, 2, 3, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, false, {128, 4, 3, 1})); // strided dst
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F16, GGML_TYPE_F16, {128, 2, 3, 1}, {128, 2, 3, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, false, {128, 4, 3, 1})); // strided dst

    // CPY - different src/dst shapes (reshaping via CPY)
    // Use permutations of {3, 5, 7, 32}. Total elements: 3*5*7*32 = 3360.
    // Each src permutation is tested against canonical sorted and reverse dst (skip self).
    {
        std::array<int64_t, 4> dims = {3, 5, 7, 32};
        std::sort(dims.begin(), dims.end());
        std::array<int64_t, 4> canonical = dims;
        std::array<int64_t, 4> reversed  = {32, 7, 5, 3};
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            std::array<int64_t, 4> cur = dims;
            do {
                if (cur != canonical) {
                    test_cases.emplace_back(new test_cpy(type, type, cur, canonical));
                }
                if (cur != reversed) {
                    test_cases.emplace_back(new test_cpy(type, type, cur, reversed));
                }
                if (cur[0] == 32 && type == GGML_TYPE_F32) {
                    if (canonical[0] == 32) {
                        test_cases.emplace_back(new test_cpy(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, cur, canonical));
                    }
                    if (reversed[0] == 32) {
                        test_cases.emplace_back(new test_cpy(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, cur, reversed));
                    }
                }
                std::next_permutation(cur.begin(), cur.end());
            } while (cur != canonical);
        }
    }

    for (ggml_type type_dst : { GGML_TYPE_F32, GGML_TYPE_I32, GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        for (bool use_view_slice : { true, false }) {
            for (std::array<int64_t, 4> ne : std::initializer_list<std::array<int64_t, 4>>{ {2, 1, 1, 1}, {2, 1, 3, 5},
                {2, 3, 5, 7}, {1, 4, 4, 1}, {1, 8, 17, 1}, {10, 10, 10, 1} }) {
                if (use_view_slice && (type_dst == GGML_TYPE_F16 || type_dst == GGML_TYPE_BF16)) {
                    continue; // TODO: add after WebGPU is fixed
                }
                test_cases.emplace_back(new test_cont(type_dst, ne, use_view_slice));
            }
        }
    }

    for (ggml_type type_dst : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        for (std::array<int64_t, 4> ne : std::initializer_list<std::array<int64_t, 4>>{
                {10, 10, 10, 1}, {33, 5, 7, 1}, {64, 3, 65, 1}, {2, 3, 5, 7},
                // large, tile-aligned and tile-unaligned, matching the perf cases
                {1024, 64, 64, 1}, {2304, 64, 64, 1}, {1000, 33, 65, 1} }) {
            for (std::array<int64_t, 4> perm : std::initializer_list<std::array<int64_t, 4>>{
                    {2, 1, 0, 3},   // 0<->2 swap
                    {1, 2, 0, 3},   // 3-cycle
                    {0, 2, 1, 3} }) {
                test_cases.emplace_back(new test_cont(type_dst, ne, false, perm));
            }
        }
    }

    auto add_test_bin_bcast = [&](ggml_type type, std::array<int64_t, 4> ne, std::array<int, 4> nr, bool perm1 = false, bool src_overlap = false) {
        for (auto op : {ggml_add, ggml_sub, ggml_mul, ggml_div}) {
            test_cases.emplace_back(new test_bin_bcast(op, type, ne, nr, 1, perm1, src_overlap));
        }
    };
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (bool perm1 : {false, true}) {
            add_test_bin_bcast(type, {1,  1,   8,   1}, {1,  1, 1, 1}, perm1);
            add_test_bin_bcast(type, {1,  1,   1,   1}, {32, 1, 1, 1}, perm1);
            add_test_bin_bcast(type, {1,  1, 320, 320}, {1,  1, 1, 1}, perm1);
            add_test_bin_bcast(type, {10, 5,   1,   1}, {1,  1, 1, 1}, perm1);
            add_test_bin_bcast(type, {10, 5,   4,   1}, {1,  1, 1, 1}, perm1);
            add_test_bin_bcast(type, {10, 5,   4,   3}, {1,  1, 1, 1}, perm1);
            add_test_bin_bcast(type, {10, 5,   4,   3}, {2,  1, 1, 1}, perm1);
            add_test_bin_bcast(type, {10, 5,   4,   3}, {1,  2, 1, 1}, perm1);
            add_test_bin_bcast(type, {10, 5,   4,   3}, {1,  1, 2, 1}, perm1);
            add_test_bin_bcast(type, {10, 5,   4,   3}, {1,  1, 1, 2}, perm1);
            add_test_bin_bcast(type, {10, 5,   4,   3}, {1,  1, 2, 2}, perm1);
            add_test_bin_bcast(type, {10, 5,   4,   3}, {1,  2, 2, 2}, perm1);
            add_test_bin_bcast(type, {10, 5,   4,   3}, {2,  2, 2, 2}, perm1);
        }

        // src_overlap
        add_test_bin_bcast(type, {10, 5, 4, 6}, {1, 1, 1, 1}, false, true);
        add_test_bin_bcast(type, {10, 5, 4, 5}, {1, 1, 1, 1}, false, true);
        add_test_bin_bcast(type, {1, 1, 120, 120}, {1, 1, 1, 1}, false, true);
        add_test_bin_bcast(type, {1, 1, 4, 320}, {1, 1, 1, 1}, false, true);

        // test case for k_bin_bcast_unravel in CUDA backend
        add_test_bin_bcast(type, {1, 1, 65536, 1}, {256, 1, 1, 1});

        // stable diffusion
        add_test_bin_bcast(type, {1280, 1, 1, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {1280, 1, 1, 1}, {1, 16, 16, 1});
        add_test_bin_bcast(type, {1280, 16, 16, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {1280, 1, 1, 1}, {1, 256, 1, 1});
        add_test_bin_bcast(type, {1, 1, 1280, 1}, {16, 16, 1, 1});
        add_test_bin_bcast(type, {16, 16, 1280, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {1, 1, 1920, 1}, {16, 16, 1, 1});
        add_test_bin_bcast(type, {1, 1, 2560, 1}, {16, 16, 1, 1});
        add_test_bin_bcast(type, {1, 1, 1280, 1}, {32, 32, 1, 1});
        add_test_bin_bcast(type, {1, 1, 1920, 1}, {32, 32, 1, 1});
        add_test_bin_bcast(type, {1, 1, 640, 1}, {32, 32, 1, 1});
        add_test_bin_bcast(type, {5120, 1, 1, 1}, {1, 256, 1, 1});
        add_test_bin_bcast(type, {640, 1, 1, 1}, {1, 1, 1, 1});
        add_test_bin_bcast(type, {64, 262144, 1, 1}, {1, 1, 1, 1});
        //add_test_bin_bcast(type, {3, 3, 2560, 1280}, {1, 1, 1, 1});
        //add_test_bin_bcast(type, {3, 3, 2560, 1280}, {2, 1, 1, 1});
    }

    // single inplace tests, especially important for WebGPU backend since kernels for inplace vs. not are different
    test_cases.emplace_back(new test_bin_bcast(ggml_add_inplace, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));
    test_cases.emplace_back(new test_bin_bcast(ggml_mul_inplace, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));
    test_cases.emplace_back(new test_bin_bcast(ggml_sub_inplace, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));
    test_cases.emplace_back(new test_bin_bcast(ggml_div_inplace, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));

    // fusion
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 5, 4, 3}, {2, 1, 1, 1}, 2));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 2, 1, 1}, 3));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 5, 4, 3}, {1, 1, 2, 1}, 4));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 2}, 5));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 5, 4, 3}, {1, 1, 2, 2}, 6));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {10, 5, 4, 3}, {1, 2, 2, 2}, 7));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {16, 5, 4, 3}, {2, 2, 2, 2}, 8));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {16, 5, 4, 3}, {1, 1, 1, 1}, 16));

    test_cases.emplace_back(new test_scale());
    test_cases.emplace_back(new test_scale(GGML_TYPE_F32, {10, 10, 10, 10}, 2.0f, 1.0f));
    test_cases.emplace_back(new test_scale(GGML_TYPE_F32, {10, 10, 10, 10}, 2.0f, 1.0f, true)); // inplace test
    test_cases.emplace_back(new test_scale(GGML_TYPE_F32, {100, 10, 10, 10}, 2.0f, 1.0f));
    // [TAG_MERGE_ROWS] row views of the merged hc down | inject output: lo (offs 0) and inject (offs 320), nt 1..8 and wider
    for (int64_t nt : {1, 2, 5, 6, 8, 9, 64}) {
        for (bool silu : {false, true}) {
            test_cases.emplace_back(new test_scale_rows({320, nt, 1, 1}, 324, 0,   silu));
        }
        test_cases.emplace_back(new test_scale_rows({4, nt, 1, 1}, 324, 320, false));
        test_cases.emplace_back(new test_scale_rows({48, nt, 1, 1}, 96, 48, false));
    }
    test_cases.emplace_back(new test_scale_rows({16, 3, 2, 2}, 40, 7, true));
    // bias != 0 over a strided source (the CPU reference's bias branch must step the source by nb01)
    test_cases.emplace_back(new test_scale_rows({320, 6, 1, 1}, 324, 0, false, 1.5f));
    test_cases.emplace_back(new test_scale_rows({48, 9, 1, 1}, 96, 48, false, -0.75f));
    test_cases.emplace_back(new test_scale_rows({16, 3, 2, 2}, 40, 7, false, 0.5f));
    test_cases.emplace_back(new test_softcap(GGML_TYPE_F32, {10, 10, 10, 10}, 50.0f));
    test_cases.emplace_back(new test_silu_back());

    for (float eps : { 0.0f, 1e-6f, 1e-4f, 1e-1f, 10.f }) {
        for (uint32_t n : { 64, 1025 }) {
            for (bool v : { false, true }) {
                test_cases.emplace_back(new test_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, v, eps));
                test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, v, eps));
            }
            test_cases.emplace_back(new test_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, false, eps, true));
            test_cases.emplace_back(new test_rms_norm_back(GGML_TYPE_F32, { n, 5, 4, 3 }, eps));
            test_cases.emplace_back(new test_l2_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, eps, false));
            test_cases.emplace_back(new test_l2_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, eps, true));
            test_cases.emplace_back(new test_l2_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, eps, false, true));
        }
        // row lengths that are not a multiple of 32, for the scalar (33) and float4 (132, 260) paths
        for (uint32_t n : { 33, 132, 260 }) {
            for (bool v : { false, true }) {
                test_cases.emplace_back(new test_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, v, eps));
                test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, v, eps));
            }
        }
    }

    // in-place tests
    test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {64, 5, 4, 3}, false, 1e-6f, true));

    for (float eps : { 0.0f, 1e-6f, 1e-4f, 1e-1f, 1.0f }) {
        for (uint32_t n : { 64, 1025 }) {
            test_cases.emplace_back(new test_rms_norm_mul_add(GGML_TYPE_F32, { n, 5, 4, 3 }, eps, false));
            test_cases.emplace_back(new test_rms_norm_mul_add(GGML_TYPE_F32, { n, 5, 4, 3 }, eps, true));
            test_cases.emplace_back(new test_norm_mul_add(GGML_TYPE_F32, { n, 5, 4, 3 }, eps, false));
            test_cases.emplace_back(new test_norm_mul_add(GGML_TYPE_F32, { n, 5, 4, 3 }, eps, true));
            test_cases.emplace_back(new test_add_rms_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, eps, false));
            test_cases.emplace_back(new test_add_rms_norm(GGML_TYPE_F32, { n, 5, 4, 3 }, eps, true));
        }
    }
    for (uint32_t n : {1, 511, 1025, 8192, 33*512}) {
        for (bool multi_add : {false, true}) {
            test_cases.emplace_back(new test_rms_norm_mul_add(GGML_TYPE_F32, {n, 1, 1, 1}, 1e-6f, false, multi_add));
        }
        test_cases.emplace_back(new test_add_rms_norm(GGML_TYPE_F32, {n, 1, 1, 1}, 1e-6f, false));
    }

    for (auto multi_add : {false, true}) {
        for (auto set_rows : {false, true}) {
            for (auto broadcast : {false, true}) {
                for (auto rope : {GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX}) {
                    test_cases.emplace_back(new test_rms_norm_mul_rope({768, 1, 1, 1}, 1e-6f, multi_add, set_rows, broadcast, rope));
                    test_cases.emplace_back(new test_rms_norm_mul_rope({768, 3, 1, 1}, 1e-6f, multi_add, set_rows, broadcast, rope));
                    test_cases.emplace_back(new test_rms_norm_mul_rope({768, 3, 5, 1}, 1e-6f, multi_add, set_rows, broadcast, rope));
                    test_cases.emplace_back(new test_rms_norm_mul_rope({128, 32, 2, 1}, 1e-6f, multi_add, set_rows, broadcast, rope));
                    test_cases.emplace_back(new test_rms_norm_mul_rope({128, 4, 2, 1}, 1e-6f, multi_add, set_rows, broadcast, rope));
                    test_cases.emplace_back(new test_rms_norm_mul_rope({128, 32, 50, 1}, 1e-6f, multi_add, set_rows, broadcast, rope));
                    test_cases.emplace_back(new test_rms_norm_mul_rope({128, 4, 50, 1}, 1e-6f, multi_add, set_rows, broadcast, rope));
                    test_cases.emplace_back(new test_rms_norm_mul_rope({8192, 2, 2, 1}, 1e-6f, multi_add, set_rows, broadcast, rope));
                    test_cases.emplace_back(new test_rms_norm_mul_rope({8192, 2, 2, 1}, 1e-6f, multi_add, set_rows, broadcast, rope));
                }
            }
        }
    }
    for (int64_t d_conv : {3, 4, 9}) {
        for (int64_t d_inner: {1024, 1536, 2048}) {
            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_conv, d_inner, 1, 1}, {d_conv, d_inner, 1, 1}));
            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {2 * d_conv, d_inner, 1, 1}, {d_conv, d_inner, 1, 1}));
            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_conv, d_inner, 4, 1}, {d_conv, d_inner, 1, 1}));
            // long token (n_t > 32, exercises the long_token kernel path)
            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_conv - 1 + 64, d_inner, 1, 1}, {d_conv, d_inner, 1, 1}));
            test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {d_conv - 1 + 64, d_inner, 4, 1}, {d_conv, d_inner, 1, 1}));
        }
    }

    // fused ssm_conv + (optional) bias_add + silu. The bias-only graph (no silu) is intentionally
    // not tested since there's no fusion for that pattern in ggml_cuda_can_fuse.
    for (int64_t d_conv : {3, 4, 9}) {
        for (int64_t d_inner : {1024, 1536, 2048}) {
            for (bool fuse_bias : {false, true}) {
                // short token path (n_t <= 32)
                test_cases.emplace_back(new test_ssm_conv_bias_silu(
                    GGML_TYPE_F32, {d_conv, d_inner, 1, 1}, {d_conv, d_inner, 1, 1}, fuse_bias));
                test_cases.emplace_back(new test_ssm_conv_bias_silu(
                    GGML_TYPE_F32, {2 * d_conv, d_inner, 1, 1}, {d_conv, d_inner, 1, 1}, fuse_bias));
                test_cases.emplace_back(new test_ssm_conv_bias_silu(
                    GGML_TYPE_F32, {d_conv, d_inner, 4, 1}, {d_conv, d_inner, 1, 1}, fuse_bias));
                // long token path (n_t > 32)
                test_cases.emplace_back(new test_ssm_conv_bias_silu(
                    GGML_TYPE_F32, {d_conv - 1 + 64, d_inner, 1, 1}, {d_conv, d_inner, 1, 1}, fuse_bias));
                test_cases.emplace_back(new test_ssm_conv_bias_silu(
                    GGML_TYPE_F32, {d_conv - 1 + 64, d_inner, 4, 1}, {d_conv, d_inner, 1, 1}, fuse_bias));
            }
        }
    }

    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 16, 1, 1024, 1, 32, 4)); // Mamba-1
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 16, 2, 32, 4)); // Mamba-2
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 256, 64,  8, 2, 32, 4)); // Falcon-H1
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 128, 4, 4, 16, 2, true)); // x/B/C overlap
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 80, 128, 1, 256, 1)); // Nemotron-9B SSD path
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 80, 128, 1, 512, 1)); // Nemotron-9B SSD multi-chunk (2 aligned chunks)
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 80, 8, 300, 2)); // Mamba-2 SSD multi-chunk (partial 2nd chunk, 2 seqs)
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 16, 2, 4, 2, false, /*K=*/4)); // Mamba-2 rollback snapshots
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 16, 2, 8, 2, false, /*K=*/3)); // Mamba-2 rollback overflow
    test_cases.emplace_back(new test_ssm_scan_rollback(GGML_TYPE_F32, 128, 64, 16, 2, 8, 2, /*K=*/3)); // rollback snapshots match prefix states
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 16, 2, 64, 4)); // Metal SSD one chunk MMA only, no seq tail
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 16, 2, 65, 2)); // SSD one chunk + 1-token sequential tail
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 16, 2, 128, 2)); // SSD multi-chunk, no tail (exercises the chunk-to-chunk state handoff)
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 16, 2, 128, 2, false, /*K=*/1, /*weak_decay=*/true)); // SSD multi-chunk, carried state not numerically negligible

    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_rwkv_wkv6(GGML_TYPE_F32, 32, 64, 128, 4));

    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 1, 4));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_rwkv_wkv7(GGML_TYPE_F32, 32, 64, 128, 4));

    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 1, 1));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 32, 1));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 32, 4));
    test_cases.emplace_back(new test_gla(GGML_TYPE_F32, 32, 64, 128, 4));

    // FWHT tests
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 128, 1, 128));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 64, 1, 64));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 256, 1, 256));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 512, 1, 512));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 128, 32, 128));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 128, 4, 128, {2, 3}));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 256, 512, 256)); // many rows
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 32, 1, 32)); // too small (N<64)
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 1024, 1, 1024)); // too big (N>512)

    // [TAG_MMVQ_HOIST] C1: the decode mat-vec shapes at 1..8 columns (f32 router, HC up / down / inject, q8_0 hands
    // shapes, a q4_0 one) - n = 1 is the untouched path, n >= 2 the hoisted twins
    for (int64_t n = 1; n <= 8; ++n) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32,  GGML_TYPE_F32,   512, n,  2560, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q5_0, GGML_TYPE_F32, 10240, n,   320, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,   320, n, 10240, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,     4, n, 10240, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32,  2560, n,   640, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32,   512, n,  2560, {1, 1}, {1, 1}));
    }

    // [TAG_DENSE_TILE_B2] the B2 hc tile (m <= 384, split_k 1) and its boundaries vs the CPU (review-B2 I1): the production
    // q4_K 320 x 10240, q5_0 / q6_K inside 65..384, the m 64 / 65 / 384 / 385 edges; m 2048 n 256 reaches the opt-in xl tile
    for (int64_t n : { 128, 1024, 2044, 2048 }) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 320, n, 10240, {1, 1}, {1, 1}));
    }
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q5_0, GGML_TYPE_F32, 200, 512, 2560, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q5_0, GGML_TYPE_F32, 200, 512,  320, {1, 1}, {1, 1}));   // unaligned K
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 384, 1024, 2560, {1, 1}, {1, 1}));
    for (int64_t m : { 64, 65, 384, 385 }) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, m, 2047, 2560, {1, 1}, {1, 1}));
    }
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 2048, 256, 2560, {1, 1}, {1, 1}));

    // [TAG_DENSE_TILE_G1] the G1 production dense shapes (P.1 profile, m x k) vs the CPU at partial-tile widths. Heavy (CPU
    // reference of 12288 x 2047 x 2560), so only emitted with GGML_VK_DISABLE_MMQ_CM1 set non-zero - the env under which the
    // _g1 tiles are reached (GGML_VK_DENSE_TILE_G1 unset = the table, or =B:BM:BN:WM:WN:PAD = sweep mode)
    const char * g1_env = getenv("GGML_VK_DISABLE_MMQ_CM1");
    for (int64_t n : { 513, 2047 }) {
        if (g1_env == nullptr || atoi(g1_env) == 0) {
            break;
        }
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q5_0, GGML_TYPE_F32, 10240, n,   320, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,  2560, n,  6144, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,   320, n, 10240, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 10240, n,  2560, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 12288, n,  2560, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32,   320, n, 10240, {1, 1}, {1, 1}));
    }

    // [TAG_INJECT_TILE_M16] the qwen4exp hc inject GEMM (m = 4, k = 4 x 2560) and the m 16 / 17 edges of the BM 16 tile vs the CPU
    for (ggml_type type_a : {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
        for (int64_t m : { 4, 16, 17 }) {
            for (int64_t n : { 9, 17, 64, 128, 2044, 2048 }) {
                test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, m, n, 10240, {1, 1}, {1, 1}));
            }
        }
    }

    // [TAG_MMID_TILE_B1] [TAG_MMID_PRECAST_B] the qwen4exp expert GEMMs vs the CPU, uniform and skewed routing, decode to prefill widths
    for (int router : { MMID_ROUTER_UNIFORM, MMID_ROUTER_SKEWED }) {
        for (int64_t n : { 8, 64, 65, 128, 2044, 2048 }) {   // review B1 M7: + 64 and 2048
            test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 512, 10, true, 640, n, 2560, router));
            test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_IQ3_S, GGML_TYPE_F32, 512, 10, true, 640, n, 2560, router));
            test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q6_K, GGML_TYPE_F32, 512, 10, true, 640, n, 2560, router));
            test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q5_0, GGML_TYPE_F32, 512, 10, false, 2560, n, 640, router));
            test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q8_0, GGML_TYPE_F32, 512, 10, false, 2560, n, 640, router));
        }
    }

#if 0
    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q8_0, GGML_TYPE_F32, 128, 128, false, 8192, 2, 5120)); // Llama-4-Maverick-17B-128E-PAB-Q8_0
    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q8_0, GGML_TYPE_F32, 128, 128, false, 8192, 1, 5120)); // Llama-4-Maverick-17B-128E-PAB-Q8_0
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 8192, 1, 5120, {128, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 8192, 512, 5120, {128, 1}, {1, 1}));
    // > 4GB A matrix. Too slow to be enabled by default.
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16,  900000,  3, 2592, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 1700000, 96, 2592, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 1700000,  3, 2592, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 1700000,  1, 2592, {1, 1}, {1, 1}));

#endif

    // repeated expert ids within a row (upstream #29998)
    for (ggml_type ta : {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
        for (int n : {9, 16, 33, 64}) {
            test_cases.emplace_back(new test_mul_mat_id_dup(ta, GGML_TYPE_F32, 28, 10, false, 1024, n, 256));
        }
    }

    for (ggml_type ta : {GGML_TYPE_F16, GGML_TYPE_Q4_0}) {
        test_cases.emplace_back(new test_mul_mat_id_dup(ta, GGML_TYPE_F32, 1025, 10, false, 64, 33, 256, MMID_ROUTER_UNIFORM, 0, true));
        test_cases.emplace_back(new test_mul_mat_id_dup(ta, GGML_TYPE_F32, 2050, 10, true, 64, 33, 256));
    }

    // [TAG_MMID_TILE_G2A] the 8-warp mul_mat_id l tile (the B1-classified types, m > 64) with duplicate ids and hot routes whose
    // per-expert rows cross BN 64, and n_as > 512
    for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_IQ3_S}) {
        for (int n_as : {512, 513, 1025}) {
            for (int n : {65, 513, 2047}) {
                test_cases.emplace_back(new test_mul_mat_id_dup(ta, GGML_TYPE_F32, n_as, 10, false, 128, n, 256));
                test_cases.emplace_back(new test_mul_mat_id(ta, GGML_TYPE_F32, n_as, 10, false, 128, n, 256, MMID_ROUTER_SKEWED));
            }
        }
    }

    for (ggml_type type_a : all_types) {
        for (int i = 1; i < 10; ++i) {
            test_cases.emplace_back(new test_mul_mat(type_a,    GGML_TYPE_F32, 16,  i, 1*256, { 1,  1}, {1, 1}));
            //test_cases.emplace_back(new test_mul_mat(type_a,    GGML_TYPE_F32, 12,  i, 2*256, { 2,  1}, {1, 1}));
            //test_cases.emplace_back(new test_mul_mat(type_a,    GGML_TYPE_F32, 11,  i, 3*256, { 1,  3}, {5, 1}));
            //test_cases.emplace_back(new test_mul_mat(type_a,    GGML_TYPE_F32, 13,  i, 4*256, { 2,  3}, {1, 1}));
            //test_cases.emplace_back(new test_mul_mat(type_a,    GGML_TYPE_F32, 17,  i, 31*256, { 4,  1}, {1, 1}));
            //test_cases.emplace_back(new test_mul_mat(type_a,    GGML_TYPE_F32, 18,  i, 32*256, { 1,  1}, {8, 1}));
            //test_cases.emplace_back(new test_mul_mat(type_a,    GGML_TYPE_F32, 19,  i, 33*256, { 1,  1}, {1, 1}));
        }
    }

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32, 2880, 32, 2880, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 2880, 32, 2880, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_MXFP4, GGML_TYPE_F32, 2880, 32, 2880, {1, 1}, {1, 1}));

    // m == 1, with n on both sides of MMVF_MAX_BATCH_SIZE (8): mmvf below, operand swap above
    for (int64_t n : {1, 7, 8, 9, 16, 128, 512}) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 1, n, 2048, {1, 1}, {1, 1}));
    }

#if 0
    {
        // Test paths in OpenCL
        std::vector<int> ns = {32, 64, 128, 256, 512, 1024, 4096};
        std::vector<int> ks = {896, 1536, 4096};
        for (auto n : ns) {
            for (auto k : ks) {
                test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 1024, n, k, {1, 1}, {1, 1}));
            }
        }
    }
#endif

#if 1
    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            std::vector<int> ks = { 256 };
            if (ggml_blck_size(type_a) == 1) {
                ks.push_back(4);
            }
            for (auto k : ks) {
                // test cases without permutation
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {1, 1}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {1, 1}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {1, 1}, {1, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 1}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 1}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 2}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 2}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 2}, {1, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {3, 2}, {2, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  4, k, {3, 2}, {2, 2}));

                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {1, 1}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {1, 1}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {1, 1}, {1, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 1}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 1}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 2}, {1, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 2}, {2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 2}, {1, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {3, 2}, {2, 2}));

                // test cases with permutation
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {2, 3}, {1, 1}, {0, 2, 1, 3}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {2, 3}, {1, 1}, {0, 1, 3, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, k, {2, 3}, {1, 1}, {0, 3, 2, 1}));

                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  4, k, {2, 3}, {1, 1}, {0, 3, 2, 1}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  8, k, {2, 3}, {1, 1}, {0, 2, 1, 3}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  8, k, {2, 3}, {1, 1}, {0, 1, 3, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  8, k, {2, 3}, {1, 1}, {0, 3, 2, 1}));

                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {2, 3}, {1, 1}, {0, 2, 1, 3}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {2, 3}, {1, 1}, {0, 1, 3, 2}));
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, k, {2, 3}, {1, 1}, {0, 3, 2, 1}));
            }

            // test cases with large ne00/ne10 to cover stream-k fixup
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  1, 1024, {3, 2}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16,  8, 1024, {3, 2}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 1024, {3, 2}, {1, 1}));

            // test cases with large batch size
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 8, 256, {1536, 1}, {1, 1}));
        }
    }

    // BF16 is absent from base_types: add the 3 standard non-contig permutations explicitly
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 16,  1, 256, {2, 3}, {1, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 16,  1, 256, {2, 3}, {1, 1}, {0, 1, 3, 2}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 16,  1, 256, {2, 3}, {1, 1}, {0, 3, 2, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 16,  8, 256, {2, 3}, {1, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 16,  8, 256, {2, 3}, {1, 1}, {0, 1, 3, 2}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 16,  8, 256, {2, 3}, {1, 1}, {0, 3, 2, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 16, 16, 256, {2, 3}, {1, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 16, 16, 256, {2, 3}, {1, 1}, {0, 1, 3, 2}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_BF16, GGML_TYPE_F32, 16, 16, 256, {2, 3}, {1, 1}, {0, 3, 2, 1}));

    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32}) {
            if (ggml_blck_size(type_a) != 256) {
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, ggml_blck_size(type_a), {1,  1}, {1, 1}));
            }
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {1,  1}, {1, 1}));
        }
    }
#else
    // m = a rows
    // n = b rows
    // k = cols
    std::uniform_int_distribution<> dist_m(1, 128);
    std::uniform_int_distribution<> dist_n(16, 128);
    std::uniform_int_distribution<> dist_k(1, 16);
    for (int i = 0; i < 1000; i++) {
        for (ggml_type type_a : all_types) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                int m = dist_m(rng);
                int n = dist_n(rng);
                int k = dist_k(rng) * ggml_blck_size(type_a);
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, m, n, k, { 1,  1}, {1, 1}));
            }
        }
    }
#endif

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 2,  128, { 8,  1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  83, 2,  128, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 2,   64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  83, 2,   64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 45, 128, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128, 45,  64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 1056, 1, 193, {1,  1}, {4, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 1056, 1, 67,  {1,  1}, {4, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 16, 32, 32, { 1,  1}, {1, 1}, {0, 1, 2, 3}, 64, 3));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 77, 77, {12,1}, {1,1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 32, 4, 96, {3, 2}, {1, 1}, {0, 1, 2, 3}, 0, 1, true));
    // upstream #28956: the first rows of a KV cache: a is a view whose batches are strided by the cache length
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 8,  1, 64, {8, 1}, {1, 1}, {0, 1, 2, 3}, 0, 1, false, 5120));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 8, 16, 64, {8, 1}, {1, 1}, {0, 1, 2, 3}, 0, 1, false, 5120));
    // the batches of a are padded, nb[2] is not a multiple of nb[1]
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 8, 1, 64, {8, 1}, {1, 1}, {0, 1, 2, 3}, 0, 1, false, 16, 16));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 8, 16, 64, {8, 1}, {1, 1}, {0, 1, 2, 3}, 0, 1, false, 16, 16));
    // as is a view whose experts are strided by more rows than it uses
    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F32, GGML_TYPE_F32, 4, 2, false, 8,  1, 64, MMID_ROUTER_UNIFORM, 64));
    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 4, 2, false, 8, 16, 64, MMID_ROUTER_UNIFORM, 64));

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32, 576, 512, 576, {1,1}, {1,1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_0, GGML_TYPE_F32, 1, 2048, 8192, {1,  1}, {1, 1}));
    for (ggml_type type_a : all_types) {
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 1, 64, 256, {1,  1}, {1, 1}));
    }

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 6, 4096, 5120, {1, 1}, {1, 1}));

    // K not a multiple of 32
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 64, 32,  65, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 64, 32,  80, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 32,  80, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 32,  80, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 64, 32, 588, {1, 1}, {1, 1})); // 14*14*3, e.g. conv_2d im2col
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 64, 32,  80, {4, 1}, {1, 1}));

#if 0
    // test the mat-mat path for Metal
    for (int k = 1; k < 512; ++k) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 127, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 127, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 77, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 77, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 64, 128, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 64, 128, k, {12,1}, {1,1}));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, false, 50, 200, k));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, true, 50, 200, k));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_BF16, GGML_TYPE_F32, 16, 16, false, 50, 200, k));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_BF16, GGML_TYPE_F32, 16, 16, true, 50, 200, k));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F32, GGML_TYPE_F32, 16, 16, false, 50, 200, k));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F32, GGML_TYPE_F32, 16, 16, true, 50, 200, k));
    }
#endif

    for (auto bs2 : {1,3}) {
        for (auto bs : {1,2,4,8}) {
            for (auto nr : {1,4}) {
                for (uint32_t m = 0; m < 2; ++m) {
                    for (uint32_t k = 0; k < 2; ++k) {
                        for (ggml_type type: {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32}) {
                            test_cases.emplace_back(new test_mul_mat(type, GGML_TYPE_F32, 1056 + m, 1, 128 + k,  {bs,  bs2}, {nr, 1}, {0, 2, 1, 3}));
                            test_cases.emplace_back(new test_mul_mat(type, GGML_TYPE_F32, 128 + m,  1, 1056 + k, {bs,  bs2}, {nr, 1}, {0, 1, 2, 3}, 2*1056 + k));
                        }
                    }
                }
            }
        }
    }

    // [TAG_DMMV_SHAPES] the dense mat-vec dispatches of a Flash-Next verify graph, at the weight
    // types and the (m, k) the model actually has: q4_K/q6_K attention and FFN projections,
    // q5_0 and q8_0 heads, and the f32 weights.  n = 6 is a 5-token draft plus the verified
    // token (the shapes the M3 capture shows at 53 % of the DRAM roofline); n = 1 is the same
    // kernel at NUM_COLS = 1, which is a separate pipeline object.  k = 320 is skipped for the
    // 256-element k-quants, which cannot have it.
    for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
        for (int64_t k : {320, 2560, 10240}) {
            if (k % ggml_blck_size(type_a) != 0) {
                continue;
            }
            for (int64_t m : {4, 512, 2560, 6144, 10240}) {
                for (int64_t n : {1, 6}) {
                    test_cases.emplace_back(new test_mul_mat_vec(type_a, m, n, k));
                }
            }
        }
    }

    // sycl backend will limit task global_range < MAX_INT
    // test case for f16-type-convert-to-fp32 kernel with large k under fp32 compute dtype (occurs in stable-diffusion)
    // however this case needs to alloc more memory which may fail in some devices (Intel Arc770, etc.)
    // this case is verified (pass) in Intel(R) Data Center GPU Max 1100 (sycl backend) and NV A30 (cuda backend)
    // test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F16, 512, 262144, 9216, {1, 1}, {1, 1}));

    // test large experts*tokens
    for (bool b : {false, true}) {
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, b, 32, 1024, 16));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 2, 2, b, 32, 8192, 64));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, b, 50, 200, 64));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_BF16, GGML_TYPE_F32, 16, 16, b, 32, 1024, 16));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_BF16, GGML_TYPE_F32, 16, 16, b, 50, 200, 64));
    }

    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16, GGML_TYPE_F32, 1, 1, false, 8, 16, 1));
    test_cases.emplace_back(new test_mul_mat_id_fusion(GGML_TYPE_F16, GGML_TYPE_F32, 16, 16, false, 32, 32, 32, 3));

    // [TAG_MMID_COMPACT] Flash-Next MoE routing: 512 experts, 10 used per token.  These are the
    // shapes the flattened (expert, N-tile) grid is built for - at n = 2048 the 3-D grid launches
    // 512 x ceil(2048/128) = 8 192 tile-columns of which ~512 hold rows - and they are also what
    // pins the Vulkan row id hoisting (n_as <= 512).  m and k are kept small: what is under test is
    // the launch enumeration and the tile prefix sum, not the tile arithmetic.
    //   m = 128, n = 2048 -> large tile;  m = 64, n = 2048 -> medium tile (the G53 arm);
    //   m = 128, n = 16   -> small tile;  m = 128, n = 6 is the production verify-round shape, which
    //                                     Vulkan routes to mul_mat_vec_id (ids->ne[1] <= 8) and which
    //                                     must therefore stay unaffected by the compact grid.
    // The three routers bracket the host-side bound min(n_as, nei0*nei1) + nei0*nei1/BN (672 at n = 2048,
    // n_used 10, BN 128): uniform is ~one tile per expert (512 of 672), skewed gives expert 0 all nei1 rows
    // (16 tiles) and ~one tile to each of the other 511 (~527 of 672), and tail gives nei1 rows to expert 0
    // and to each of the last n_used - 1 = 9 experts (10 x 16 = 160 of 672, 502 empty prefixes) - the
    // heaviest legal load on a single expert (count_e <= nei1). Every shape is a legal routing (distinct
    // experts per row, no expert above nei1 rows), so the same list runs on the legacy grid
    // (GGML_VK_MMID_COMPACT=0) and on CUDA.
    for (int router : {MMID_ROUTER_UNIFORM, MMID_ROUTER_SKEWED, MMID_ROUTER_TAIL}) {
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16,  GGML_TYPE_F32, 512, 10, false, 128, 2048, 256, router));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16,  GGML_TYPE_F32, 512, 10, false,  64, 2048, 256, router));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16,  GGML_TYPE_F32, 512, 10, false, 128,   16, 256, router));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_F16,  GGML_TYPE_F32, 512, 10, false, 128,    6, 256, router));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 512, 10, false, 128, 2048, 256, router));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 512, 10, false, 128,   16, 256, router));
    }

    // gpt-oss issue with Vulkan mmq_id
    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_MXFP4, GGML_TYPE_F32, 32, 2, false, 2880, 32, 2880));
    test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_0, GGML_TYPE_F32, 32, 2, false, 2880, 32, 2880));

    for (ggml_type type_a : all_types) {
        test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 4, 2, false, 64, 16, 3*ggml_blck_size(type_a)));
    }

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4, 8}) {
                for (int n_used : {1, 2, 4}) {
                    for (bool b : {false, true}) {
                        for (int n : {1, 4, 5, 17, 32, 129}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4}) {
                for (int n_used : {2}) {
                    for (bool b : {false}) {
                        for (int n : {1, 32}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    for (int bs : {1, 4, 512}) {
        for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K}) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                // test with mul after (ffn_moe_weighted)
                test_cases.emplace_back(new test_mul_mat_id_fusion(type_a, type_b, 128, 8, false, 768, bs, 2048, 1, true));
            }
        }
    }

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (int n : {1, 16}) {
                for (int k : {1, 16}) {
                    for (int bs2 : {1, 3}) {
                        for (int bs3 : {1, 3}) {
                            for (int nr2 : {1, 2}) {
                                for (int nr3 : {1, 2}) {
                                    test_cases.emplace_back(new test_out_prod(type_a, type_b, 256, n, k, {bs2, bs3}, {nr2, nr3}));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ne2 sweep to cover the cublasSgemmStridedBatched path (dps2 == 1, ne2 > 1)
    for (int64_t ne2 : {1, 8, 16, 32}) {
        test_cases.emplace_back(new test_out_prod(GGML_TYPE_F32, GGML_TYPE_F32,
                                                  256, 16, 16, {ne2, 1}, {1, 1}));
    }

    // nr2 sweep to cover the cublasSgemmBatched pointer-array path (dps2 > 1)
    for (int64_t nr2 : {8, 16, 32}) {
        test_cases.emplace_back(new test_out_prod(GGML_TYPE_F32, GGML_TYPE_F32,
                                                  256, 16, 16, {1, 1}, {nr2, 1}));
    }

    // add_id
    for (ggml_type type_a : {GGML_TYPE_F32}) {
        for (ggml_type type_b : {GGML_TYPE_F32}) {
            for (int n_mats : {4, 8}) {
                for (int n_used : {1, 2, 4}) {
                    for (int n_embd : {32, 129}) {
                        for (int n_token : {1, 32, 129}) {
                            test_cases.emplace_back(new test_add_id(type_a, type_b, n_embd, n_mats, n_used, n_token));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        test_cases.emplace_back(new test_sqr       (type));
        test_cases.emplace_back(new test_sqrt      (type));
        test_cases.emplace_back(new test_log       (type));
        test_cases.emplace_back(new test_sin       (type));
        test_cases.emplace_back(new test_cos       (type));
        test_cases.emplace_back(new test_clamp     (type));
        test_cases.emplace_back(new test_leaky_relu(type));
        test_cases.emplace_back(new test_floor     (type));
        test_cases.emplace_back(new test_ceil      (type));
        test_cases.emplace_back(new test_round     (type));
        test_cases.emplace_back(new test_trunc     (type));
        test_cases.emplace_back(new test_sqr       (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_sqr       (type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_sqrt      (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_sqrt      (type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_log       (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_log       (type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_sin       (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_sin       (type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_cos       (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_cos       (type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_clamp     (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_clamp     (type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_leaky_relu(type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_leaky_relu(type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_floor     (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_floor     (type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_ceil      (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_ceil      (type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_round     (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_round     (type, {1024, 1024, 1, 1}));
        test_cases.emplace_back(new test_trunc     (type, {7, 1, 5, 3}));
        test_cases.emplace_back(new test_trunc     (type, {1024, 1024, 1, 1}));
    }

    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 1, 1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 3, 1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 3, 2}, 5));

#if 0
    std::uniform_int_distribution<> dist_ne1(1, 50);
    int exponent = 1;
    while (exponent < (1 << 17)) {
        std::uniform_int_distribution<> dist_ne0(exponent, 2*exponent);

        for (int n = 0; n < 10; ++n) {
            int64_t ne0 = dist_ne0(rng);
            int64_t ne1 = dist_ne1(rng);
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, GGML_TYPE_F32, {ne0, ne1, 1, 1}, n/2 == 0, 0.1f, ne0 < 1000 ? 4.0f : 0.0f));
        }

        exponent <<= 1;
    }
#endif
    for (bool mask : {false, true}) {
        for (bool sinks : {false, true}) {
            for (float max_bias : {0.0f, 8.0f}) {
                if (!mask && max_bias > 0.0f) continue;
                for (float scale : {1.0f, 0.1f}) {
                    for (int64_t ne0 : {16, 1024}) {
                        for (int64_t ne1 : {16, 1024}) {
                            if (mask) {
                                for (ggml_type m_prec : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                                    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0,   ne1,   1, 1}, mask, sinks, m_prec, {1, 1}, scale, max_bias));
                                    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, mask, sinks, m_prec, {1, 1}, scale, max_bias));

                                    if (ne0 <= 32 && ne1 <= 32) {
                                        test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0,   ne1,   1, 3}, mask, sinks, m_prec, {3, 1}, scale, max_bias));
                                        test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, mask, sinks, m_prec, {2, 3}, scale, max_bias));
                                    }
                                }
                            } else {
                                /* The precision of mask here doesn't matter as boolean mask is false */
                                test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0,   ne1,   1, 1}, mask, sinks, GGML_TYPE_F32, {1, 1}, scale, max_bias));
                                test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, mask, sinks, GGML_TYPE_F32, {1, 1}, scale, max_bias));
                            }
                        }
                    }
                }
            }
            // inplace tests
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, mask, sinks, GGML_TYPE_F32, {1, 1}, 0.1f, 0.0f, true));
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, mask, sinks, GGML_TYPE_F16, {1, 1}, 0.1f, 0.0f, true));
        }
    }
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, true,  true,  GGML_TYPE_F32, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, true,  false, GGML_TYPE_F16, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, false, true,  GGML_TYPE_F32, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  true,  GGML_TYPE_F32, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  false, GGML_TYPE_F16, {1, 1}, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  true,  GGML_TYPE_F32, {1, 1}, 0.1f, 8.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  true,  GGML_TYPE_F16, {1, 1}, 0.1f, 8.0f));

    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {200001, 2, 3, 1}, true,   true,  GGML_TYPE_F32, {1, 1}, 0.1f, 8.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {200001, 2, 3, 1}, true,   true,  GGML_TYPE_F16, {1, 1}, 0.1f, 8.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {200000, 1, 1, 1}, false,  false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {200000, 4, 1, 1}, false,  false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {643251, 3, 1, 1}, false,  false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));

    for (float max_bias : {0.0f, 8.0f}) {
        for (float scale : {1.0f, 0.1f}) {
            for (int64_t ne0 : {16, 1024}) {
                for (int64_t ne1 : {16, 1024}) {
                    test_cases.emplace_back(new test_soft_max_back(GGML_TYPE_F32, {ne0,   ne1,   1, 1}, scale, max_bias));
                    test_cases.emplace_back(new test_soft_max_back(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, scale, max_bias));
                    test_cases.emplace_back(new test_soft_max_back(GGML_TYPE_F32, {ne0,   ne1,   2, 3}, scale, max_bias));
                }
            }
        }
    }

    for (bool fw : {true, false}) { // fw == forward
        bool all = true;

        for (float fs : { 1.0f, 1.4245f }) {
            for (float ef : { 0.0f, 0.7465f }) {
                for (float af : { 1.0f, 1.4245f }) {
                    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                        for (bool ff : {false, true}) { // freq_factors
                            for (float v : { 0, 1 }) {
                                test_cases.emplace_back(new test_rope(type, {128,  32, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 7B

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, {128,  40, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 13B
                                    test_cases.emplace_back(new test_rope(type, {128,  52, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 30B
                                    test_cases.emplace_back(new test_rope(type, {128,  64, 2, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw)); // llama 65B
                                    test_cases.emplace_back(new test_rope(type, {16, 16, 8192, 1}, 16, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));
                                }

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, { 64,   1, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,  71, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,   8, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)

                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  20, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  32, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 4, 1},  32, GGML_ROPE_TYPE_NORMAL, 512, fs, ef, af, ff, v, fw));

                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  20, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (stablelm)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 2, 1},  32, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (phi-2)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 4, 1},  32, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (phi-2)
                                    test_cases.emplace_back(new test_rope(type, { 16, 16, 8192, 1},  16, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw));
                                }

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1}, 128, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 2B)
                                    test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1}, 128, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl 7B)
                                    test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1},  20, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1},  32, GGML_ROPE_TYPE_MROPE,  512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1}, 128, GGML_ROPE_TYPE_IMROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,imrope (qwen3vl 2B)
                                    test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1}, 128, GGML_ROPE_TYPE_IMROPE,  512, fs, ef, af, ff, v, fw)); // rope_multi,imrope (qwen3vl 7B)
                                    test_cases.emplace_back(new test_rope(type, {128,  12, 2, 1},  20, GGML_ROPE_TYPE_IMROPE,  512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, {128,  28, 2, 1},  32, GGML_ROPE_TYPE_IMROPE,  512, fs, ef, af, ff, v, fw));
                                    test_cases.emplace_back(new test_rope(type, { 80,  16, 2, 1},  80, GGML_ROPE_TYPE_VISION, 512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen2vl ViT)
                                    test_cases.emplace_back(new test_rope(type, {128,  16, 2, 1}, 128, GGML_ROPE_TYPE_IMROPE, 512, fs, ef, af, ff, v, fw)); // rope_multi,m-rope (qwen3vl)
                                    test_cases.emplace_back(new test_rope(type, {16, 16, 8192, 1}, 16, GGML_ROPE_TYPE_IMROPE, 512, fs, ef, af, ff, v, fw));
                                }

                                test_cases.emplace_back(new test_rope(type, { 64, 128, 2, 1},  64, GGML_ROPE_TYPE_NEOX, 512, fs, ef, af, ff, v, fw)); // neox (falcon 40B)
                            }

                            // build_rope_2d-style: ROPE on a non-contiguous view
                            // that starts at a non-zero offset along dim 0
                            // (e.g. gemma4v vision second-half view).
                            for (int rmode : { GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_MROPE, GGML_ROPE_TYPE_IMROPE, GGML_ROPE_TYPE_VISION }) {
                                test_cases.emplace_back(new test_rope(type, { 36, 16, 2457, 1}, 36, rmode, 512, fs, ef, af, ff, 2, fw));
                            }
                        }

                        all = false;
                    }
                }
            }
        }
    }

    // single inplace test per type/mode/ff
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (int mode : {GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_MROPE, GGML_ROPE_TYPE_IMROPE, GGML_ROPE_TYPE_VISION}) {
            for (bool ff : {false, true}) {
                test_cases.emplace_back(new test_rope(type, {128,  32, 2, 1}, 128, mode, 512, 1.4245f, 0.7465f, 1.4245f, ff, 0, true, true));
                test_cases.emplace_back(new test_rope(type, {128,  32, 2, 1}, 128, mode, 512, 1.4245f, 0.7465f, 1.4245f, ff, 1, true, true));
                test_cases.emplace_back(new test_rope(type, {128,  32, 2, 3}, 128, mode, 512, 1.4245f, 0.7465f, 1.4245f, ff, 1, true, true));
            }
        }
    }

    // rotated dims window at an offset (ggml_rope_set_offset), not supported for vision mode
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (bool fw : {true, false}) { // fw == forward
            for (bool ff : {false, true}) {
                test_cases.emplace_back(new test_rope(type, {128, 32, 2, 1}, 32, GGML_ROPE_TYPE_NORMAL, 512, 1.4245f, 0.7465f, 1.4245f, ff, 0, fw, false, 32));
                test_cases.emplace_back(new test_rope(type, {128, 32, 2, 1}, 32, GGML_ROPE_TYPE_NEOX,   512, 1.4245f, 0.7465f, 1.4245f, ff, 0, fw, false, 32));
                test_cases.emplace_back(new test_rope(type, {128, 12, 2, 1}, 24, GGML_ROPE_TYPE_MROPE,  512, 1.4245f, 0.7465f, 1.4245f, ff, 0, fw, false, 32));
                test_cases.emplace_back(new test_rope(type, {128, 12, 2, 1}, 24, GGML_ROPE_TYPE_IMROPE, 512, 1.4245f, 0.7465f, 1.4245f, ff, 0, fw, false, 32));
            }
        }
        // inplace with an offset
        test_cases.emplace_back(new test_rope(type, {128, 32, 2, 1}, 32, GGML_ROPE_TYPE_NEOX, 512, 1.4245f, 0.7465f, 1.4245f, false, 0, true, true, 32));
    }

    for (int v : { 0, 1, 2, 3 }) {
        for (int dim : { 0, 1, 2, 3, }) {
            test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {11, 12, 13, 14}, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_F16, {11, 12, 13, 14}, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_BF16, {11, 12, 13, 14}, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_I8, {11, 12, 13, 14}, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_I16, {11, 12, 13, 14}, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_I32, {11, 12, 13, 14}, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_I64, {11, 12, 13, 14}, 7, dim, v));
        }
    }

    // Transposed src1 (v = 16). concat(state, ggml_transpose(x), 0) is the conv-state build of
    // qwen4exp/mamba/delta-net/arwkv7; on Vulkan it is split into a copy of src0 plus a tiled
    // 32x32 transpose-copy of src1, which must be byte-exact with the single-dispatch path.
    test_cases.emplace_back(new test_concat(GGML_TYPE_F32,  {3, 10240, 1, 1}, 2048, 0, 16)); // production shape
    test_cases.emplace_back(new test_concat(GGML_TYPE_F32,  {5,  1027, 1, 1},   67, 0, 16)); // odd sizes, partial tiles
    test_cases.emplace_back(new test_concat(GGML_TYPE_F16,  {3,  1027, 1, 1},   67, 0, 16)); // 2-byte tiled transpose
    test_cases.emplace_back(new test_concat(GGML_TYPE_BF16, {3,  1027, 1, 1},   67, 0, 16));
    test_cases.emplace_back(new test_concat(GGML_TYPE_F32,  {3,    64, 3, 2},   67, 0, 16)); // dims 2/3 as batch
    // Same transposed layout but below the tile threshold / on another axis: these must stay on the
    // generic single-dispatch path, so they guard the gating conditions rather than the fast path.
    test_cases.emplace_back(new test_concat(GGML_TYPE_F32,  {3,  1027, 1, 1},    7, 0, 16));
    for (int dim : { 1, 2, 3 }) {
        test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {67, 129, 1, 1}, 7, dim, 16));
    }

    for (ggml_type type_a : { GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0 }) {
        for (int v : { 0, 4, 8, 12 }) {
            for (int dim : { 0, 1, 2, 3, }) {
                test_cases.emplace_back(new test_concat(type_a, {128, 12, 13, 14}, dim == 0 ? 256 : 7, dim, v));
            }
        }
    }

    for (ggml_sort_order order : {GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC}) {
        for (uint32_t i = 4; i <= 1024*1024; i *= 2) {
            test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {i-1, 1, 1, 1}));
            test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {i, 1, 1, 1}));
        }
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {16, 10, 10, 10}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {60, 10, 10, 10}, order)); // qwen
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {1023, 2, 1, 3}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {1024, 2, 1, 3}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {1025, 2, 1, 3}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {1025, 256, 1, 1}, order)); // test ceildiv in CUDA's CUB's DeviceSegmentedSort
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {2047, 2, 1, 3}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {2048, 2, 1, 3}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {2049, 2, 1, 3}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {2, 8, 8192, 1}, order)); // bailingmoe2 (group selection)
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {2048, 512, 1, 1}, order)); // test CUDA dispatching to radix sort for nrows > = 1 in graph mode
    }

    for (int n = 1; n < 5; ++n) {
        for (int k = 1; k <= n; ++k) {
            test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {n, 2, 1, 3}, k, true));
        }
    }
    for (int k : {1, 10, 40}) {
        test_cases.emplace_back(new test_top_k_inf({4096, 2, 1, 1}, k));
        test_cases.emplace_back(new test_top_k_inf({248320, 1, 1, 1}, k));
    }

    for (int i = 0; i < 20; ++i) {
        for (int k : {1, 2, 3, 7, 15, 100, 500, 1023, 9999}) {
            if (k <= 1<<i) {
                test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {(1<<i), 1, 1, 1}, k));
                test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {(1<<i) + 11, 1, 2, 1}, k));
                test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {(1<<i) + 11, 1, 2, 1}, k, true));
            }
        }
    }
    for (int k : {4, 8, 16, 32}) {
        for (int nrows : {1, 8, 16}) {
            test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {202048, nrows, 1, 1}, k));
            test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {151936, nrows, 1, 1}, k));
            test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {8192,   nrows, 1, 1}, k));
            test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {8193,   nrows, 1, 1}, k));
            test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {8192,   nrows, 1, 1}, k, true));
            test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {202048, nrows, 1, 1}, k, true));
        }
    }

    for (int k : {1, 2, 3, 7, 15}) {
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {16, 10, 10, 10}, k));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {60, 10, 10, 10}, k));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {1023, 2, 1, 3}, k));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {1024, 2, 1, 3}, k));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {1025, 2, 1, 3}, k));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {16384, 1, 1, 1}, k));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {2047, 2, 1, 3}, k));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {2048, 2, 1, 3}, k));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {2049, 2, 1, 3}, k));
    }

    //test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {262144, 8192, 1, 1}, 1024));
    //test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {1048576, 512, 1, 1}, 2048));
    // Large-k, including multi-row and ties (qwen4exp)
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 1024,  1, 1, 1 }, 1024));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 2048,  2, 1, 1 }, 1024));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 4096,  1, 1, 1 }, 2048));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 8192,  2, 1, 1 }, 2051));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 33024, 1, 1, 1 }, 2051));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 33024, 4, 1, 1 }, 2051));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 8192,  2, 1, 1 }, 2051, 1));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 33024, 4, 1, 1 }, 2051, 1));

    // ... and ties the k-th largest value actually lands on, with the ids compared exactly
    // (ties == 2, four structures spread over the rows: block-constant, all-equal, plateau,
    // signed-zero plateau). k >= 1024 on purpose - that is the radix-select path, the one the
    // QSA selection uses; the small-k pipelines are not required to rank ties by index.
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 2048,  4, 1, 1 }, 1025, 2));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 8192,  4, 1, 1 }, 2051, 2));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 16384, 4, 1, 1 }, 2051, 2));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 33024, 4, 1, 1 }, 2051, 2));

    // qwen4exp QSA indexer top-k fusion (get_rows + f16 mask + top_k)
    test_cases.emplace_back(new test_topk_qsa(512,  2048,  1, 1, 1500));
    test_cases.emplace_back(new test_topk_qsa(512,  2048,  2, 1, 1500));
    test_cases.emplace_back(new test_topk_qsa(256,  2048,  4, 2, 2000));
    test_cases.emplace_back(new test_topk_qsa(64,   256,   2, 1, 200));  // small k: unfused fallback
    // production width (indexer_top_k 2048 + ratio 4 - 1) and decode n_tps (5 draft + 1)
    test_cases.emplace_back(new test_topk_qsa(1024, 4096,  6, 1, 2051, true));

    // ... and with the fused destination forced to alias each operand the dispatch reads at the
    // TOP_K (1 = scores, 2 = cell_blk, 3 = the raw f16 mask; see test_topk_qsa::alias). Production
    // width and ratio 4, at decode (n_tps 1) and verify (n_tps 6) shapes; n_kv/n_blocks are sized so
    // the destination fits inside each target. A backend that fuses one of these instead of vetoing
    // reads a buffer it is writing, and the result stops matching the CPU reference.
    for (int64_t n_tps : { 1, 6 }) {
        for (int alias = 1; alias <= 3; ++alias) {
            test_cases.emplace_back(new test_topk_qsa(4096, 16384, n_tps, 1, 2051, true, alias));
        }
    }

    // ... and with the ties the selection is built out of (see test_topk_qsa::ties): the mask
    // is zero, so a block's cells are indistinguishable and width 2051 = 512*4 + 3 cuts inside
    // one. Production width and ratio 4 at the 4k/16k shapes. The ids are compared with zero
    // tolerance, so these fail on any backend whose large-k top_k picks a different member of
    // the tie group than (value descending, index ascending) - which is the point: that choice
    // is what makes a decode reproducible.
    test_cases.emplace_back(new test_topk_qsa(1024, 4096,  6, 1, 2051, true, 0, 1));
    test_cases.emplace_back(new test_topk_qsa(4096, 16384, 1, 1, 2051, true, 0, 1));
    test_cases.emplace_back(new test_topk_qsa(4096, 16384, 6, 1, 2051, true, 0, 2));
    test_cases.emplace_back(new test_topk_qsa(4096, 16384, 6, 1, 2051, true, 0, 3));
    // ... and the tie the deployed graph hits on EVERY prefill ubatch whose causal window is
    // shorter than the width: the mask is -INFINITY past a per-row cutoff below the width, so the
    // k-th largest value is -inf and the selection has to pick (width - cut) cells out of a tie
    // group (n_kv - cut) wide. ties == 4; see qsa_select_init_tensors.
    test_cases.emplace_back(new test_topk_qsa(4096, 16384, 1, 1, 2051, true, 0, 4));
    test_cases.emplace_back(new test_topk_qsa(4096, 16384, 6, 1, 2051, true, 0, 4));
    // ... and the below-budget exactness path (LLAMA_QSA_BYPASS_EXHAUSTIVE=0): width == n_kv, so
    // the selection covers every cell and the only correct answer is a permutation of 0..n_kv-1.
    // The comparator checks exactly that per row (range + uniqueness + the reference's set).
    // n_kv 2048 keeps it in the radix regime (k >= 1024) and inside the f16 mask ramp.
    test_cases.emplace_back(new test_topk_qsa(512,  2048,  2, 1, 2048, true));

    // [TAG_QSA_FUSED_SELECT] ... and the same selection as ONE node (GGML_OP_QSA_TOP_K, the
    // GGML_VK_QSA_FUSED_SELECT=1 graph). Deliberately the same shapes as the pattern cases above,
    // minus the alias modes - see test_qsa_top_k for why the alias is unreachable for a node whose
    // operands are its own srcs - so -o TOPK_QSA and -o QSA_TOP_K cover the same ground.
    // [TAG_KQ_VIS] GGML_OP_KQ_MASK, bit for bit: every branch of the mask fill (causal / not, the three SWA
    // kinds, ALiBi, M-RoPE ties), F16 and F32, one and three streams, the prod-like shapes
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_F32 }) {
        for (int64_t n_stream : { 1, 3 }) {
            test_cases.emplace_back(new test_kq_mask(1024, 32, n_stream, type));
            test_cases.emplace_back(new test_kq_mask(1000, 7,  n_stream, type, true,  false, 0, 0,   true));
            test_cases.emplace_back(new test_kq_mask(1024, 32, n_stream, type, false));
            test_cases.emplace_back(new test_kq_mask(1024, 32, n_stream, type, true,  true));
            test_cases.emplace_back(new test_kq_mask(1024, 32, n_stream, type, true,  false, 1, 128));
            test_cases.emplace_back(new test_kq_mask(1024, 32, n_stream, type, true,  false, 2, 256));
            test_cases.emplace_back(new test_kq_mask(1024, 32, n_stream, type, false, false, 3, 100));
        }
    }
    for (int64_t n_tps : { 1, 5, 6, 64, 2048 }) {
        test_cases.emplace_back(new test_kq_mask(8192, n_tps, 1, GGML_TYPE_F16));
    }
    test_cases.emplace_back(new test_kq_mask(65536, 6, 1, GGML_TYPE_F16));
    test_cases.emplace_back(new test_kq_mask(70000, 2, 1, GGML_TYPE_F16, true, true)); // ALiBi overflow to -inf
    // ALiBi distances > 2048: f16 rounds -|d| to 11 significant bits, f32 keeps it exact
    test_cases.emplace_back(new test_kq_mask(8192, 16, 1, GGML_TYPE_F32, true,  true));
    test_cases.emplace_back(new test_kq_mask(8192, 16, 1, GGML_TYPE_F16, true,  true));
    test_cases.emplace_back(new test_kq_mask(8192, 16, 1, GGML_TYPE_F32, false, true, 1, 5000, true));

    // [TAG_KQ_VIS] FLASH_ATTN_EXT over the metadata: the model's shape (24 heads over 2 KV heads, head dim 256;
    // target q8_0 K/V, MTP draft f16), decode 1/5/6 rows (GQA fold, split_k), verify, prefill (mask-opt pass)
    for (ggml_type type_KV : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        for (int64_t nb : { 1, 2, 5, 6, 64, 512 }) {
            for (int64_t kv : { 1024, 4096 }) {
                test_cases.emplace_back(new test_flash_attn_ext_vis(256, 2, 12, kv, nb, 1, type_KV));
            }
        }
        test_cases.emplace_back(new test_flash_attn_ext_vis(256, 2, 12, 2048, 64, 3, type_KV));             // 3 streams
        test_cases.emplace_back(new test_flash_attn_ext_vis(256, 2, 12, 4096, 64, 1, type_KV, true, 0.0f, 0, 0, true)); // M-RoPE ties
    }
    test_cases.emplace_back(new test_flash_attn_ext_vis(256, 2, 12, 4096, 64, 1, GGML_TYPE_F16, true,  8.0f));          // ALiBi
    test_cases.emplace_back(new test_flash_attn_ext_vis(256, 2, 12, 4096, 64, 1, GGML_TYPE_F16, true,  0.0f, 1, 512));  // SWA standard
    test_cases.emplace_back(new test_flash_attn_ext_vis(256, 2, 12, 4096, 64, 1, GGML_TYPE_F16, true,  0.0f, 2, 1024)); // SWA chunked
    test_cases.emplace_back(new test_flash_attn_ext_vis(256, 2, 12, 4096, 64, 1, GGML_TYPE_F16, false, 0.0f, 3, 700));  // non-causal, symmetric
    test_cases.emplace_back(new test_flash_attn_ext_vis(128, 4, 4,  1024, 8,  2, GGML_TYPE_F16));                      // another head size

    // [TAG_UP29591] a second matmul on the same input after an op that reuses the conversion scratch
    for (bool y_nc : {false, true}) {
        for (ggml_type type_w : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
            for (int mode : {0, 1, 2}) {
                test_cases.emplace_back(new test_mul_mat_reuse_across_scratch(type_w, mode, y_nc));
            }
        }
    }

    test_cases.emplace_back(new test_qsa_top_k(512,  2048,  1, 1, 1500));
    test_cases.emplace_back(new test_qsa_top_k(512,  2048,  2, 1, 1500));
    test_cases.emplace_back(new test_qsa_top_k(256,  2048,  4, 2, 2000));
    test_cases.emplace_back(new test_qsa_top_k(64,   256,   2, 1, 200));   // small k: still the radix path
    test_cases.emplace_back(new test_qsa_top_k(1024, 4096,  6, 1, 2051, true));
    for (int64_t n_tps : { 1, 6 }) {
        test_cases.emplace_back(new test_qsa_top_k(4096, 16384, n_tps, 1, 2051, true));
    }
    test_cases.emplace_back(new test_qsa_top_k(1024, 4096,  6, 1, 2051, true, 1));
    test_cases.emplace_back(new test_qsa_top_k(4096, 16384, 1, 1, 2051, true, 1));
    test_cases.emplace_back(new test_qsa_top_k(4096, 16384, 6, 1, 2051, true, 2));
    test_cases.emplace_back(new test_qsa_top_k(4096, 16384, 6, 1, 2051, true, 3));
    // the -inf masked tail and the width == n_kv selection, as above for the seven-node pattern
    test_cases.emplace_back(new test_qsa_top_k(4096, 16384, 1, 1, 2051, true, 4));
    test_cases.emplace_back(new test_qsa_top_k(4096, 16384, 6, 1, 2051, true, 4));
    test_cases.emplace_back(new test_qsa_top_k(512,  2048,  2, 1, 2048, true));
    // the f32-mask branch of the CPU reference. The graph never builds this (it emits the op only
    // for the f16 flash-attention mask) and the Vulkan shader binds a float16_t mask, so this case
    // is expected to report "not supported" there rather than to run.
    test_cases.emplace_back(new test_qsa_top_k(1024, 4096,  6, 1, 2051, true, 0, GGML_TYPE_F32));

    // [TAG_QSA_WHOLE_BLOCKS] the census-shaped inputs (ties == 5, tests/qsa-select-ref.h) in the cell form - the
    // deployed selection on the inputs it really gets - and in the block form (n_sel 512 = indexer_top_k / r): decode,
    // verify and prefill-chunk rows, one and two streams (stream 0 a full cache: padding has no masked cell to use),
    // rows with fewer complete blocks than the budget. Then the block form over the other initializations (their
    // tie structures, no tail boost), a small budget (n_sel 50 over 64 blocks) and the f32-mask CPU branch.
    for (int n_sel : { 0, 512 }) {
        test_cases.emplace_back(new test_qsa_top_k(1024,  4096,  6,  1, 2051, true, 5, GGML_TYPE_F16, n_sel));
        test_cases.emplace_back(new test_qsa_top_k(4096,  16384, 1,  1, 2051, true, 5, GGML_TYPE_F16, n_sel));
        test_cases.emplace_back(new test_qsa_top_k(4096,  16384, 6,  2, 2051, true, 5, GGML_TYPE_F16, n_sel));
        test_cases.emplace_back(new test_qsa_top_k(16384, 65536, 64, 1, 2051, true, 5, GGML_TYPE_F16, n_sel));
        test_cases.emplace_back(new test_qsa_top_k(650,   2600,  8,  1, 2051, true, 5, GGML_TYPE_F16, n_sel));
    }
    for (int ties = 0; ties <= 4; ++ties) {
        test_cases.emplace_back(new test_qsa_top_k(1024, 4096,  6, 1, 2051, true, ties, GGML_TYPE_F16, 512));
        test_cases.emplace_back(new test_qsa_top_k(4096, 16384, 6, 1, 2051, true, ties, GGML_TYPE_F16, 512));
    }
    test_cases.emplace_back(new test_qsa_top_k(64,   256,   2, 1, 200,  true, 0, GGML_TYPE_F16, 50));
    test_cases.emplace_back(new test_qsa_top_k(512,  2048,  2, 1, 2048, true, 0, GGML_TYPE_F16, 512));
    test_cases.emplace_back(new test_qsa_top_k(1024, 4096,  6, 1, 2051, true, 5, GGML_TYPE_F32, 512));

    // [TAG_QSA_TOPK_V2] the selection EXACTLY - ids and order - against the contract's reference (see
    // test_qsa_top_k_exact): every data mode at the decode/verify shapes (1 and 6 queries, k 512 and the
    // deployed 2051) from one tile to 128 tiles of the v2 kernel, ...
    for (int64_t n_kv : { 4096, 65536, 131072, 262144 }) {
        for (int64_t n_tps : { 1, 6 }) {
            for (int width : { 512, 2051 }) {
                for (int mode = 0; mode < test_qsa_top_k_exact::M_COUNT; mode++) {
                    test_cases.emplace_back(new test_qsa_top_k_exact(n_kv, n_tps, 1, width, mode));
                }
            }
        }
    }
    // ... tail residues (n_kv not a multiple of the 2048-cell tile, of 256, or of 4) ...
    for (int64_t n_kv : { 2051, 4096 + 77, 65536 + 2047, 131072 + 1000, 262144 + 1 }) {
        for (int64_t n_tps : { 1, 6 }) {
            for (int mode : { test_qsa_top_k_exact::M_RANDOM, test_qsa_top_k_exact::M_QUANT, test_qsa_top_k_exact::M_MASK,
                              test_qsa_top_k_exact::M_ZERO, test_qsa_top_k_exact::M_PROD }) {
                test_cases.emplace_back(new test_qsa_top_k_exact(n_kv, n_tps, 1, 2051, mode));
            }
        }
    }
    // ... k = 1, k = n_kv (every cell), k = n_kv - 1, two streams, a prefill-sized query batch
    test_cases.emplace_back(new test_qsa_top_k_exact(4096,  6,  1, 1,    test_qsa_top_k_exact::M_RANDOM));
    test_cases.emplace_back(new test_qsa_top_k_exact(4096,  6,  1, 1,    test_qsa_top_k_exact::M_EQUAL));
    test_cases.emplace_back(new test_qsa_top_k_exact(4096,  6,  1, 4096, test_qsa_top_k_exact::M_RANDOM));
    test_cases.emplace_back(new test_qsa_top_k_exact(4096,  6,  1, 4096, test_qsa_top_k_exact::M_QUANT));
    test_cases.emplace_back(new test_qsa_top_k_exact(4096,  6,  1, 4095, test_qsa_top_k_exact::M_QUANT));
    test_cases.emplace_back(new test_qsa_top_k_exact(2051,  6,  1, 2051, test_qsa_top_k_exact::M_EQUAL));
    test_cases.emplace_back(new test_qsa_top_k_exact(65536, 3,  2, 2051, test_qsa_top_k_exact::M_RANDOM));
    test_cases.emplace_back(new test_qsa_top_k_exact(65536, 3,  2, 2051, test_qsa_top_k_exact::M_QUANT));
    test_cases.emplace_back(new test_qsa_top_k_exact(65536, 3,  2, 2051, test_qsa_top_k_exact::M_PROD));
    test_cases.emplace_back(new test_qsa_top_k_exact(65536, 64, 1, 2051, test_qsa_top_k_exact::M_PROD));
    test_cases.emplace_back(new test_qsa_top_k_exact(65536, 64, 1, 2051, test_qsa_top_k_exact::M_QUANT));
    // [TAG_QSA_WHOLE_BLOCKS] ... and the block-level form (ggml_qsa_top_k_blocks), element for element: every data
    // mode at the decode / verify shapes with the deployed (k 2051, 512 blocks) and a truncating (k 512, 128 blocks)
    // budget, tail residues, more blocks requested than exist, one tie group, k = 1, two streams, 64 rows
    for (int64_t n_kv : { 4096, 65536, 131072, 262144 }) {
        for (int64_t n_tps : { 1, 6 }) {
            for (const auto & kb : { std::pair<int, int>{ 2051, 512 }, std::pair<int, int>{ 512, 128 } }) {
                for (int mode = 0; mode < test_qsa_top_k_exact::M_COUNT; mode++) {
                    test_cases.emplace_back(new test_qsa_top_k_exact(n_kv, n_tps, 1, kb.first, mode, kb.second));
                }
            }
        }
    }
    for (int64_t n_kv : { 2051, 4096 + 77, 65536 + 2047, 262144 + 1 }) {
        for (int64_t n_tps : { 1, 6 }) {
            for (int mode : { test_qsa_top_k_exact::M_CENSUS, test_qsa_top_k_exact::M_QUANT, test_qsa_top_k_exact::M_PROD }) {
                test_cases.emplace_back(new test_qsa_top_k_exact(n_kv, n_tps, 1, 2051, mode, 512));
            }
        }
    }
    test_cases.emplace_back(new test_qsa_top_k_exact(1024,  6,  1, 1024, test_qsa_top_k_exact::M_CENSUS, 512));
    test_cases.emplace_back(new test_qsa_top_k_exact(4096,  6,  1, 4096, test_qsa_top_k_exact::M_EQUAL,  1024));
    test_cases.emplace_back(new test_qsa_top_k_exact(4096,  6,  1, 4096, test_qsa_top_k_exact::M_MASK,   1024));
    test_cases.emplace_back(new test_qsa_top_k_exact(4096,  6,  1, 1,    test_qsa_top_k_exact::M_RANDOM, 1));
    test_cases.emplace_back(new test_qsa_top_k_exact(65536, 3,  2, 2051, test_qsa_top_k_exact::M_CENSUS, 512));
    test_cases.emplace_back(new test_qsa_top_k_exact(65536, 3,  2, 2051, test_qsa_top_k_exact::M_QUANT,  512));
    test_cases.emplace_back(new test_qsa_top_k_exact(65536, 64, 1, 2051, test_qsa_top_k_exact::M_CENSUS, 512));

    // [TAG_QSA_POOL_REMAP] the indexed score, bit for bit against get_rows + mul_mat on the backend itself:
    // every data mode x every kernel the plain f32 MUL_MAT picks (mat-vec N 1 / 4 / 8, matmul small N 12 / 24 / 32,
    // medium 36 / 64, large 96 / 1024), aligned K 128 and unaligned K 72 / 70, partial tiles, a deployed-depth pool
    for (int mode = 0; mode < test_mul_mat_rows::M_COUNT; mode++) {
        for (int64_t N : { 1, 4, 8, 12, 24, 32, 36, 64, 96 }) {
            for (int64_t M : { 1000, 4096 }) {
                test_cases.emplace_back(new test_mul_mat_rows(128, M, N, 1, mode, true));
            }
        }
        for (int64_t K : { 72, 70 }) {
            for (int64_t N : { 4, 24, 36, 96 }) {
                test_cases.emplace_back(new test_mul_mat_rows(K, 777, N, 1, mode, true));
            }
        }
        // two streams: the batched mat-vec (N 1), the batched matmul (N 4 / 24)
        for (int64_t N : { 1, 4, 24 }) {
            test_cases.emplace_back(new test_mul_mat_rows(128, 3000, N, 2, mode, true));
        }
        // small M (fewer rows than one tile), a prefill chunk (N 1024)
        test_cases.emplace_back(new test_mul_mat_rows(128, 7,    24,   1, mode, true));
        test_cases.emplace_back(new test_mul_mat_rows(128, 1,    4,    1, mode, true));
        test_cases.emplace_back(new test_mul_mat_rows(128, 2048, 1024, 1, mode, true));
    }
    // the decode / verify shapes at deployed pool depths: 131072 / 262144 / 524288 cells / 4 = blocks, Q = 1 / 2 / 6
    for (int64_t M : { 32768, 65536, 131072 }) {
        for (int64_t N : { 4, 8, 24 }) {
            test_cases.emplace_back(new test_mul_mat_rows(128, M, N, 1, test_mul_mat_rows::M_PROD,   true));
            test_cases.emplace_back(new test_mul_mat_rows(128, M, N, 1, test_mul_mat_rows::M_RANDOM, true));
        }
    }
    // ... and the op against the CPU (NMSE)
    for (int mode : { test_mul_mat_rows::M_RANDOM, test_mul_mat_rows::M_PROD }) {
        for (int64_t N : { 1, 4, 24, 96 }) {
            test_cases.emplace_back(new test_mul_mat_rows(128, 4096, N, 1, mode, false));
        }
        test_cases.emplace_back(new test_mul_mat_rows(72,  777,  24, 1, mode, false));
        test_cases.emplace_back(new test_mul_mat_rows(128, 3000, 24, 2, mode, false));
    }

    // [TAG_QSA_SCORE_EPILOGUE] the fused indexer score, bit for bit against the chain on the backend itself: ragged M
    // (fewer rows than a tile, not a multiple of the 128-row tile), every chunk width 1..128 (1 / 2 = the mat-vec widths
    // the Vulkan op refuses), both product forms, the three bias modes; unaligned K; two streams; head counts 1 / 2 / 8 /
    // 16 (and 3: refused by Vulkan, TN % 3 != 0); the deployed pool depths at the deployed chunk (C 128) and its tail
    for (bool r : { false, true }) {
        for (int bm : { 0, 1, 2 }) {
            for (int64_t M : { 1, 17, 129, 1000 }) {
                for (int64_t N = 1; N <= 128; N += (N < 20 ? 1 : (N < 64 ? 7 : 16))) {
                    test_cases.emplace_back(new test_qsa_score(128, M, N, 1, 4, r, bm, true));
                }
                test_cases.emplace_back(new test_qsa_score(128, M, 127, 1, 4, r, bm, true));
                test_cases.emplace_back(new test_qsa_score(128, M, 128, 1, 4, r, bm, true));
            }
            for (int64_t N : { 1, 3, 37, 128 }) {
                test_cases.emplace_back(new test_qsa_score(128, 4133, N, 2, 4, r, bm, true));
                test_cases.emplace_back(new test_qsa_score(72,  777,  N, 1, 4, r, bm, true));
                test_cases.emplace_back(new test_qsa_score(64,  65,   N, 1, 4, r, bm, true));
            }
            for (int H : { 1, 2, 3, 8, 16 }) {
                test_cases.emplace_back(new test_qsa_score(128, 1000, 9,  1, H, r, bm, true));
                test_cases.emplace_back(new test_qsa_score(128, 1000, 64, 1, H, r, bm, true));
            }
        }
        for (int64_t M : { 16384, 16384 + 37, 65536 }) {
            for (int64_t N : { 128, 121, 64, 5 }) {
                test_cases.emplace_back(new test_qsa_score(128, M, N, 1, 4, r, 1, true));
            }
        }
    }
    // [TAG_QSA_POOLED_F16_MIRROR] the same op reading the store's f16 mirror (set_rows f32 -> f16 of every row), still bit for
    // bit against the f32 chain: the widths, ragged M, both bias modes, unaligned K, two streams, head counts, the deployed
    // depths, and the store scaled by 1e-5 (f16 denormals in the mirror) and 1e3
    for (int bm : { 0, 1 }) {
        for (int64_t M : { 1, 17, 129, 1000 }) {
            for (int64_t N = 1; N <= 128; N += (N < 20 ? 1 : (N < 64 ? 7 : 16))) {
                test_cases.emplace_back(new test_qsa_score(128, M, N, 1, 4, true, bm, true, true));
            }
            test_cases.emplace_back(new test_qsa_score(128, M, 128, 1, 4, true, bm, true, true));
        }
        for (int64_t N : { 1, 3, 6, 37, 128 }) {
            test_cases.emplace_back(new test_qsa_score(128, 4133, N, 2, 4, true, bm, true, true));
            test_cases.emplace_back(new test_qsa_score(72,  777,  N, 1, 4, true, bm, true, true));
            test_cases.emplace_back(new test_qsa_score(128, 1000, N, 1, 4, true, bm, true, true, 1e-5f));
            test_cases.emplace_back(new test_qsa_score(128, 1000, N, 1, 4, true, bm, true, true, 1e3f));
        }
        for (int H : { 1, 2, 8, 16 }) {
            test_cases.emplace_back(new test_qsa_score(128, 1000, 9, 1, H, true, bm, true, true));
        }
    }
    for (int64_t M : { 65536, 262144 }) {
        for (int64_t N : { 3, 4, 5, 6, 9, 128 }) {
            test_cases.emplace_back(new test_qsa_score(128, M, N, 1, 4, true, 1, true, true));
        }
    }

    // ... and the op against the CPU (NMSE)
    for (bool r : { false, true }) {
        for (int64_t N : { 3, 37, 128 }) {
            test_cases.emplace_back(new test_qsa_score(128, 4096, N, 1, 4, r, 2, false));
        }
        test_cases.emplace_back(new test_qsa_score(128, 3000, 24, 2, 4, r, 2, false));
        test_cases.emplace_back(new test_qsa_score(72,  777,  24, 1, 4, r, 0, false));
    }

    // Qwen 3.8 Flash Next at the deployed context: n_kv 213000, ratio 4, width 2051, decode
    // and -ub 2048 prefill. Opt-in: the n_tps 2048 pair needs ~10 GB of backend and as much
    // host memory for the reference, and runs for a while.
    if (getenv("GGML_TEST_TOPK_PROD")) {
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 213000, 6,    1, 1 }, 2051));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 213000, 2048, 1, 1 }, 2051));
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, { 213000, 6,    1, 1 }, 2051, 2));
        test_cases.emplace_back(new test_topk_qsa(53250, 213000, 6,    1, 2051, true));
        test_cases.emplace_back(new test_topk_qsa(53250, 213000, 2048, 1, 2051, true));
        test_cases.emplace_back(new test_topk_qsa(53250, 213000, 6,    1, 2051, true, 0, 1));
        test_cases.emplace_back(new test_topk_qsa(53250, 213000, 6,    1, 2051, true, 0, 4));
        test_cases.emplace_back(new test_qsa_top_k(53250, 213000, 6,    1, 2051, true));
        test_cases.emplace_back(new test_qsa_top_k(53250, 213000, 2048, 1, 2051, true));
        test_cases.emplace_back(new test_qsa_top_k(53250, 213000, 6,    1, 2051, true, 1));
        test_cases.emplace_back(new test_qsa_top_k(53250, 213000, 6,    1, 2051, true, 4));
    }

    // [TAG_QSA_COMPACT_BIAS] the compact bias: the deployed shapes (C 128 / the reserve's balanced chunk),
    // decode, the PAD chunk, two and five streams (n_blocks not a multiple of 256), random rows with repeats
    test_cases.emplace_back(new test_qsa_bias(131072, 2048, 128, 1, 1));
    test_cases.emplace_back(new test_qsa_bias(61440,  2050, 228, 1, 1));
    test_cases.emplace_back(new test_qsa_bias(512,    7,    7,   1, 0));
    test_cases.emplace_back(new test_qsa_bias(4096,   265,  256, 1, 2));
    test_cases.emplace_back(new test_qsa_bias(2048,   64,   64,  2, 3));
    test_cases.emplace_back(new test_qsa_bias(1000,   5,    5,   5, 0));
    test_cases.emplace_back(new test_qsa_bias(1000,   40,   64,  5, 2));

    // exhaustive top_k tests
    //for (int i = 1; i < 9999; ++i) {
    //    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {i, 2, 1, 3}, rand() % i + 1));
    //}

    for (ggml_scale_mode mode : {GGML_SCALE_MODE_NEAREST, GGML_SCALE_MODE_BILINEAR, GGML_SCALE_MODE_BICUBIC, ggml_scale_mode(GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS)}) {
        test_cases.emplace_back(new test_upscale(GGML_TYPE_F32, {512, 512, 3, 2}, 2, mode));
        test_cases.emplace_back(new test_upscale(GGML_TYPE_F32, {512, 512, 3, 2}, 2, mode, true));
        test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {2, 5,  7, 11}, {5, 7, 11, 13}, mode));
        test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {5, 7, 11, 13}, {2, 5,  7, 11}, mode));
    }
    for (ggml_scale_mode mode : {GGML_SCALE_MODE_BILINEAR, GGML_SCALE_MODE_BICUBIC}) {
        test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {2, 5, 7, 11}, {5, 7, 11, 13}, (ggml_scale_mode)(mode | GGML_SCALE_FLAG_ALIGN_CORNERS)));
        test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {1, 4, 3, 2}, {2, 8, 3, 2}, (ggml_scale_mode)(mode | GGML_SCALE_FLAG_ALIGN_CORNERS)));
        test_cases.emplace_back(new test_interpolate(GGML_TYPE_F32, {4, 1, 3, 2}, {1, 1, 3, 2}, (ggml_scale_mode)(mode | GGML_SCALE_FLAG_ALIGN_CORNERS)));
    }

    test_cases.emplace_back(new test_sum());
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, {11, 5, 6, 3}, {0, 2, 1, 3}));  // row-contiguous but non-contiguous
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, {11, 5, 6, 3}, {0, 3, 2, 1}));
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, {11, 5, 6, 3}, {0, 1, 3, 2}));
    test_cases.emplace_back(new test_mean());
    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, { 33, 1, 1, 1 }));
    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, { 33, 256, 1, 1 }));
    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, { 32769, 1, 1, 1 }));
    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, { 32, 1, 1, 1 }));
    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, { 32, 256, 1, 1 }));
    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, { 32768, 1, 1, 1 }));
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, { 33, 1, 1, 1 }));
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, { 33, 1024, 1, 1 }));
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, { 33, 256, 1, 1 }));
    test_cases.emplace_back(new test_sum(GGML_TYPE_F32, { 33, 256, 1, 1 }, { 1, 0, 2, 3 })); // sum dst not-contiguous
    test_cases.emplace_back(new test_sum_rows());
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 11, 5, 6, 3 }, true, false));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 11, 5, 6, 3 }, false, true));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 11, 5, 6, 3 }, true, true));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 16, 5, 6, 3 }, true, false));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 16, 5, 6, 3 }, false, true));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 16, 5, 6, 3 }, true, true));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 33, 1, 1, 1 }));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 33, 1024, 1, 1 }));
    test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, { 33, 256, 1, 1 }));
    test_cases.emplace_back(new test_group_norm(GGML_TYPE_F32, {64, 64, 320, 1}));
    test_cases.emplace_back(new test_group_norm(GGML_TYPE_F32, {9, 9, 1280, 1}));
    test_cases.emplace_back(new test_group_norm_mul_add(GGML_TYPE_F32, {64, 64, 320, 1}));
    test_cases.emplace_back(new test_group_norm_mul_add(GGML_TYPE_F32, {9, 9, 1280, 1}));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 1, 1}, {256, 16, 1, 1}, -1));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {256, 16, 2, 3}, -1));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {128, 16, 2, 3}, -1));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {256, 16, 2, 3}, 1));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {128, 16, 2, 3}, 2));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {64, 16, 2, 3}, 3));

    test_cases.emplace_back(new test_pad());
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {33, 17, 2, 1}, 4, 3, true)); // circular
    test_cases.emplace_back(new test_pad_ext());
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {1024, 1, 1, 1}, 1, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {1024, 2, 1, 1}, 1, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {1024, 16, 1, 1}, 0, 1, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {1023, 1, 1, 1}, 1, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {1023, 8, 1, 1}, 1, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {1025, 1, 1, 1}, 1, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {1025, 8, 1, 1}, 1, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {2048, 1, 1, 1}, 1, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {2048, 4, 1, 1}, 1, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {2049, 1, 1, 1}, 1, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {100, 1, 1, 1}, 100, 0, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {100, 1, 1, 1}, 0, 100, false));
    test_cases.emplace_back(new test_pad(GGML_TYPE_F32, {100, 100, 1, 1}, 50, 50, false));

    test_cases.emplace_back(new test_pad_reflect_1d());
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 384, 4, 1}));
    test_cases.emplace_back(new test_roll());
    test_cases.emplace_back(new test_roll(3, -2, 1, -1, true));
    test_cases.emplace_back(new test_arange());
    test_cases.emplace_back(new test_arange(GGML_TYPE_F32, 0.0f, 1048576.0f, 1.0f));
    test_cases.emplace_back(new test_timestep_embedding());
    test_cases.emplace_back(new test_leaky_relu());

    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 10, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 127, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 128, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 128, 128, 4, 4 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 255, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 256, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 511, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 512, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 1023, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 1024, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 2047, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 2048, 5, 4, 3 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 201*1204, 1, 1, 1 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 312*1205, 1, 1, 1 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 20481, 4, 1, 1 }));

    test_cases.emplace_back(new test_xielu());
    test_cases.emplace_back(new test_xielu(GGML_TYPE_F16));
    test_cases.emplace_back(new test_xielu(GGML_TYPE_F32, { 512, 16, 1, 1 }));
    test_cases.emplace_back(new test_xielu(GGML_TYPE_F16, { 512, 16, 1, 1 }));

    test_cases.emplace_back(new test_tri(GGML_TRI_TYPE_LOWER));
    test_cases.emplace_back(new test_tri(GGML_TRI_TYPE_LOWER_DIAG));
    test_cases.emplace_back(new test_tri(GGML_TRI_TYPE_UPPER));
    test_cases.emplace_back(new test_tri(GGML_TRI_TYPE_UPPER_DIAG));

    test_cases.emplace_back(new test_fill(0.0f));
    test_cases.emplace_back(new test_fill(2.0f, GGML_TYPE_F32, { 303, 207, 11, 3 }));
    test_cases.emplace_back(new test_fill(-152.0f, GGML_TYPE_F32, { 800, 600, 4, 4 }));
    test_cases.emplace_back(new test_fill(3.5f, GGML_TYPE_F32, { 2048, 512, 2, 2 }));

    test_cases.emplace_back(new test_diag());
    test_cases.emplace_back(new test_diag(GGML_TYPE_F32, { 79, 1, 19, 13 }));
    test_cases.emplace_back(new test_diag(GGML_TYPE_F32, { 256, 1, 8, 16 }));

    test_cases.emplace_back(new test_solve_tri());
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 11, 11, 1, 1 }, { 5, 11, 1, 1 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 17, 17, 2, 4 }, { 9, 17, 2, 4 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 30, 30, 7, 1 }, { 8, 30, 7, 1 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 42, 42, 5, 2 }, { 10, 42, 5, 2 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 64, 64, 2, 2 }, { 10, 64, 2, 2 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 64, 64, 2, 2 }, { 64, 64, 2, 2 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 79, 79, 5, 3 }, { 417, 79, 5, 3 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 128, 128, 4, 2 }, { 32, 128, 4, 2 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 80, 80, 2, 8 }, { 80, 80, 2, 8 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 80, 80, 2, 8 }, { 79, 80, 2, 8 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 80, 80, 2, 8 }, { 81, 80, 2, 8 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 80, 80, 8, 8 }, { 80, 80, 8, 8 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 80, 80, 8, 8 }, { 79, 80, 8, 8 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 80, 80, 8, 8 }, { 81, 80, 8, 8 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 84, 84, 4, 4 }, { 32, 84, 4, 4 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 95, 95, 8, 8 }, { 40, 95, 8, 8 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 100, 100, 4, 4 }, { 41, 100, 4, 4 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 128, 128, 4, 4 }, { 31, 128, 4, 4 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 128, 128, 4, 4 }, { 32, 128, 4, 4 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 128, 128, 3, 4 }, { 32, 128, 3, 4 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 128, 128, 4, 1 }, { 32, 128, 4, 1 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 64, 64, 4, 4 }, { 200, 64, 4, 4 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 64, 64, 4, 4 }, { 384, 64, 4, 4 }));

    for (int tfrm : {0, 1, 2}) {
        for (bool circular : {false, true}) {
            test_cases.emplace_back(new test_pad_ext(GGML_TYPE_F32, {512, 512, 1, 1}, 0, 1, 0, 1, 0, 0, 0, 0, tfrm, circular));
            test_cases.emplace_back(new test_pad_ext(GGML_TYPE_F32, {11, 22, 33, 44}, 1, 2, 3, 4, 5, 6, 7, 8, tfrm, circular));
        }
    }

    // prefill-shaped cases with long KV (nb >= 32, kv >= 1024): covers the
    // XMX/GEMM-accelerated SYCL FA path which only activates for these shapes.
    for (int kv : { 1024, 2048, }) {
        for (int hs : { 64, 128, 256, }) {
            for (int nb : { 32, 64, }) {
                for (ggml_type type_KV : { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, }) {
                    test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 8, {4, 1}, kv, nb, true, false, 0, 0, GGML_PREC_F32, type_KV, type_KV));
                }
            }
        }
    }

    for (int hsk : { 40, 64, 72, 80, 96, 128, 192, 256, 320, 512, 576 }) {
        for (int hsv : { 40, 64, 72, 80, 96, 128, 192, 256, 512 }) {
            if (hsk != 192 && hsk != 320 && hsk != 576 && hsk != hsv) continue;
            if (hsk == 192 && (hsv != 128 && hsv != 192)) continue;
            if (hsk == 576 && hsv != 512) continue; // DeepSeek MLA
            if (hsk == 320 && hsv != 256) continue; // Mistral4 MLA

            for (bool mask : { true, false } ) {
                for (bool sinks : { true, false } ) {
                    for (float max_bias : { 0.0f, 8.0f }) {
                        if (!mask && max_bias > 0.0f) continue;
                        for (float logit_softcap : {0.0f, 10.0f}) {
                            if (hsk != 128 && logit_softcap != 0.0f) continue;
                            for (int nh : { 1, 4 }) {
                                if (nh == 1 && hsk != 320 && hsk != 576) continue;
                                for (int nr3 : { 1, 3, }) {
                                    if (hsk > 64 && nr3 > 1) continue; // skip broadcast for large head sizes
                                    for (int nr2 : { 1, 4, 8, 12, 16, 20, 32 }) {
                                        if (nr2 ==  8 && hsk != 192) continue;
                                        if (nr2 == 12 && hsk != 128) continue;
                                        if (nr2 == 16 && hsk != 192) continue;
                                        if (nr2 == 20 && (nh != 1 || hsk != 576)) continue;
                                        if (nr2 == 32 && (nh != 1 || hsk != 320)) continue;
                                        //for (int kv : { 1, 17, 31, 33, 61, 113, 65, 127, 129, 130, 255, 260, 371, 380, 407, 512, 1024, }) {
                                        for (int kv : { 113, 512, 1024, }) {
                                            if (nr2 != 1 && kv != 512) continue;
                                            for (int nb : { 1, 3, 32, 75, }) {
                                                for (ggml_prec prec : {GGML_PREC_F32, GGML_PREC_DEFAULT}) {
                                                    if (hsk != 128 && prec == GGML_PREC_DEFAULT) continue;
                                                    for (ggml_type type_KV : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL}) {
                                                        if (type_KV != GGML_TYPE_F16 && hsk != 64 && hsk != 72) continue;
                                                        // DeepSeek MLA: the V cache is a sub-view of the K cache
                                                        const bool v_is_view_of_k = hsk == 576;
                                                        test_cases.emplace_back(new test_flash_attn_ext(
                                                                    hsk, hsv, nh, {nr2, nr3}, kv, nb, mask, sinks, max_bias, logit_softcap, prec, type_KV, type_KV, {0, 1, 2, 3}, true, v_is_view_of_k));
                                                        // run fewer test cases permuted
                                                        if (mask == true && max_bias == 0.0f && logit_softcap == 0 && kv == 512) {
                                                            test_cases.emplace_back(new test_flash_attn_ext(
                                                                        hsk, hsv, nh, {nr2, nr3}, kv, nb, mask, sinks, max_bias, logit_softcap, prec, type_KV, type_KV, {0, 2, 1, 3}, true, v_is_view_of_k));
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // mixed quant and Q1_0 test cases
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 4, {1, 1}, 128, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 4, {1, 1}, 128, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_F16));
    test_cases.emplace_back(new test_flash_attn_ext(72, 72, 4, {1, 1}, 96, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 4, {1, 1}, 96, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F32));
    test_cases.emplace_back(new test_flash_attn_ext(128, 128, 4, {1, 1}, 256, 1, false, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0));
    test_cases.emplace_back(new test_flash_attn_ext(128, 128, 4, {1, 1}, 96, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q1_0, GGML_TYPE_Q1_0));
    test_cases.emplace_back(new test_flash_attn_ext(128, 64, 4, {1, 1}, 128, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q1_0, GGML_TYPE_Q4_0));
    test_cases.emplace_back(new test_flash_attn_ext(64, 128, 4, {1, 1}, 128, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q1_0));
    test_cases.emplace_back(new test_flash_attn_ext(128, 64, 4, {1, 1}, 64, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q1_0, GGML_TYPE_F16));
    test_cases.emplace_back(new test_flash_attn_ext(128, 128, 4, {1, 1}, 96, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q2_0, GGML_TYPE_Q2_0));
    test_cases.emplace_back(new test_flash_attn_ext(128, 64, 4, {1, 1}, 128, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q2_0, GGML_TYPE_Q4_0));
    test_cases.emplace_back(new test_flash_attn_ext(64, 128, 4, {1, 1}, 128, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q2_0));
    test_cases.emplace_back(new test_flash_attn_ext(128, 64, 4, {1, 1}, 64, 2, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q2_0, GGML_TYPE_F16));

    // q8_0 KV cases: decode and prompt batches, KV pad, permuted KV, feature flags, and long context
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},   113,   1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},  1024,   1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},  1024,   1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 2},  1025,   1, true, true,  8, 30, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},  1025,  64, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 16384,   1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    // [TAG_QSA_INDEXED_FA] indexed flash attention (docs/superpowers/specs/2026-09-24-indexed-fa-prefill-design.md §5)
    for (ggml_type type_KV : { GGML_TYPE_Q8_0, GGML_TYPE_F16 }) {
        for (int64_t n_tps : { 9, 64, 228, 256 }) {                                                        // (a)
            const int64_t kv_w[4][2] = { { 4096, 64 }, { 4096, 2051 }, { 20000, 64 }, { 20000, 2051 } };   // (n_kv, W)
            for (const auto & p : kv_w) { test_cases.emplace_back(new test_flash_attn_ext_indexed(type_KV, n_tps, p[0], p[1])); }
        }
    }
    for (int dead : { 1, 2, 3 }) { test_cases.emplace_back(new test_flash_attn_ext_indexed(GGML_TYPE_Q8_0, 64, 4096, 2051, 1, dead)); } // (b)(c)
    test_cases.emplace_back(new test_flash_attn_ext_indexed(GGML_TYPE_Q8_0,  64, 4096, 2051, 1, 0, true));   // (d) sorted
    test_cases.emplace_back(new test_flash_attn_ext_indexed(GGML_TYPE_Q8_0,  64, 4096, 2051, 2));            // (e)
    test_cases.emplace_back(new test_flash_attn_ext_indexed(GGML_TYPE_F16,    9, 4096,   77, 2, 2));         // (e)(f)(c)
    test_cases.emplace_back(new test_flash_attn_ext_indexed(GGML_TYPE_Q8_0, 228, 4096,   77));               // (f)
    test_cases.emplace_back(new test_flash_attn_ext_indexed(GGML_TYPE_Q8_0,   9, 4096,    1));               // (f) one id
    for (int64_t n_tps : { 1, 2 }) {  // n_tps <= 8 with ids: the forced plain fold, GQA_TOKENS off
        test_cases.emplace_back(new test_flash_attn_ext_indexed(GGML_TYPE_Q8_0, n_tps, 4096, 2051));
    }
    for (int dead : { 0, 2 }) {
        test_cases.emplace_back(new test_flash_attn_ext_indexed_selfcheck(GGML_TYPE_Q8_0, 64, 4096, 2051, dead));
    }
    for (int dead : { 0, 1, 2, 3 }) {
        test_cases.emplace_back(new test_flash_attn_ext_indexed_selfcheck(GGML_TYPE_F16,   9, 4096,   77, dead));
    }
    test_cases.emplace_back(new test_flash_attn_ext_indexed_selfcheck(GGML_TYPE_F16, 9, 4096, 77, 2, /*n_stream*/ 2));

    // [TAG_QSA_VIS] (A4 T3/T4) metadata vs plane, bitwise on the backend; GGML_TEST_A4_NEGATIVE=1 registers ONLY the negative
    // controls instead (one seq bit flipped on the plane side: every one of them must FAIL)
    {
        const bool neg = getenv("GGML_TEST_A4_NEGATIVE") != nullptr;
        auto qv = [&](int64_t n_kv, int64_t n_tps, int64_t ns, int w, int n_sel, int mode, int holes, bool rope2d, bool pad) {
            test_cases.emplace_back(new test_qsa_top_k_vis(n_kv, n_tps, ns, w, n_sel, mode, holes, rope2d, pad, neg));
        };
        for (int n_sel : { 0, 512 }) {
            for (int mode : { 0, 1, 2 }) {
                qv(4096,  6,   1, 2051, n_sel, mode, 0,  false, false);  // decode / verify rows
                qv(4096,  228, 1, 2051, n_sel, mode, 50, false, false);  // a balanced prefill chunk, 50 % extra holes
                qv(16384, 64,  2, 2051, n_sel, mode, 5,  false, false);  // two streams
                qv(16384, 256, 1, 2051, n_sel, mode, 0,  false, true);   // a CHUNK_PAD tail chunk (3 padding rows)
            }
            qv(4096,   9, 1, 2051, n_sel, 1, 0, true,  false);           // M-RoPE ties at p0 == p1
            qv(65536,  6, 1, 2051, n_sel, 2, 0, false, false);
            qv(65536, 64, 1, 2051, n_sel, 1, 0, false, true);
            qv(262144, 1, 1, 2051, n_sel, 2, 5, false, false);
        }
        qv(256, 8, 1, 200, 0, 1, 0, false, false);                       // n_kv < k + visible: -inf fills, T = -inf
        auto mi = [&](int64_t n_kv, int64_t n_tps, int64_t ns, int64_t w, bool rope2d) {
            test_cases.emplace_back(new test_kq_mask_ids(n_kv, n_tps, ns, w, rope2d, neg));
        };
        mi(4096, 1, 1, 77, false); mi(4096, 6, 1, 2051, false); mi(65536, 64, 1, 2051, false);
        mi(8192, 6, 2, 2051, false); mi(4096, 8, 1, 77, true); mi(262144, 6, 1, 2051, false);
        for (ggml_type type_KV : { GGML_TYPE_Q8_0, GGML_TYPE_F16 }) {
            auto fv = [&](int64_t n_tps, int64_t n_kv, int64_t w, int64_t ns, bool rope2d, bool dead_tail) {
                test_cases.emplace_back(new test_flash_attn_ext_indexed_vis(type_KV, n_tps, n_kv, w, ns, rope2d, dead_tail, neg));
            };
            fv(9,   4096, 77,   1, false, false);
            fv(228, 4096, 2051, 1, false, false);
            fv(256, 8192, 2051, 1, false, false);
            fv(64,  4096, 2051, 2, false, false);   // two streams
            fv(64,  4096, 2051, 1, true,  false);   // M-RoPE ties
            fv(64,  4096, 2051, 1, false, true);    // dead (-1) ids
            fv(9,   4096, 1,    1, false, false);   // one id: the row's own cell
        }
    }

    // [TAG_QSA_TOPK_NOKEYS] (A3) QSA_TOP_K_NOKEYS: the metadata cell form against its plane form ON THE SAME BACKEND, bit for
    // bit in order (the plane form always keeps the keyed passes, so with GGML_VK_QSA_TOPK_NOKEYS=1 this compares keyed and
    // nokeys within one run), and against the CPU per row. The production shapes: decode / verify rows 1..9, the 128-query
    // chunk and a ragged 100, depths 65 536 (one tile ragged: + 2047) and 262 145, two streams, k = 2051, the tie / domain
    // modes 0..6 (7 = subnormal scores, outside the domain: only with GGML_TEST_NOKEYS_SUBNORMAL=1), holes 0 / 100 (100 =
    // fewer visible cells than k), pad (the chunk tail). tests/test-qsa-topk-nokeys runs these with the knob off and on and
    // compares the dumped ids (GGML_TEST_DUMP_IDS) byte for byte.
    {
        const bool sub = getenv("GGML_TEST_NOKEYS_SUBNORMAL") != nullptr;
        auto nk = [&](int64_t n_kv, int64_t n_tps, int64_t ns, int mode, int holes, bool pad) {
            auto * tc = new test_qsa_top_k_vis(n_kv, n_tps, ns, 2051, 0, mode, holes, false, pad, false);
            tc->nokeys_family = true;
            test_cases.emplace_back(tc);
        };
        const int modes_hi = sub ? 8 : 7;
        for (int mode = 0; mode < modes_hi; ++mode) {
            for (int64_t n_tps : { 1, 2, 3, 6, 9 }) {
                nk(65536, n_tps, 1, mode, 0, false);
            }
            nk(65536 + 2047, 128, 1, mode, 0,   false);
            nk(65536 + 2047, 100, 1, mode, 0,   true);
            nk(65536,        128, 2, mode, 5,   false);
            nk(65536,        128, 1, mode, 100, false);
            nk(262144 + 1,   128, 1, mode, 0,   false);
        }
    }

    // Speculative-decode verify rounds at the Flash-Next attention shape (2 KV heads, GQA 12,
    // HSK = HSV = 256, q8_0 or f16 KV, long KV).  The upstream matrix only ever runs the GQA
    // decode path at nb == 1, so nothing covered a batch of drafted tokens; the Vulkan coopmat1
    // backend folds those tokens into the row dimension (one KV pass per round instead of one
    // per token), which is only exercised at nb > 1.  nb 5 and 6 leave the last row tile
    // partly empty, nb 2 fills a single tile.
    for (int64_t nb : { 2, 5, 6 }) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 16384, nb, true, false, 0, 0,
                                                        GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 16384, nb, true, false, 0, 0,
                                                        GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    }
    // ... the same fold with a KV that is not a multiple of the KV block (bounds-checked
    // variant), with ALiBi + sinks + softcap (per-row slope and sink indexing) over a batch
    // dimension, and at a second head/GQA shape.
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 16385, 6, true, false, 0,  0,
                                                    GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 2}, 16384, 6, true, true,  8, 30,
                                                    GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(128, 128, 4, { 8, 1}, 16384, 4, true, false, 0,  0,
                                                    GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));

    // ... and the same fold as production actually dispatches it.  The cases above build a mask
    // with exactly nb rows, which is not what llama.cpp does: the KQ mask is padded to
    // GGML_KQ_MASK_PAD (64) rows, and that is what puts nem1 over the Vulkan backend's
    // use_mask_opt threshold (nem1 >= 32, nem0*nem1 > 32768).  Without it the mask-opt summary
    // path - including the folded dispatch's "read block row 0 for every row tile" rule - is
    // never reached.  The Q/K/V layout is the permuted {0, 2, 1, 3} one a real graph produces.
    for (int64_t nb : { 2, 5, 6 }) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 16384, nb, true, false, 0, 0,
                                                        GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                                                        {0, 1, 2, 3}, true, false, 64));
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 16384, nb, true, false, 0, 0,
                                                        GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                                                        {0, 2, 1, 3}, true, false, 64));
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 16384, nb, true, false, 0, 0,
                                                        GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16,
                                                        {0, 2, 1, 3}, true, false, 64));
    }
    // padded mask + KV that is not a multiple of the KV block (bounds-checked variant), and
    // padded mask + ALiBi/sinks/softcap over a batch dimension, both in the permuted layout.
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 16385, 6, true, false, 0,  0,
                                                    GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                                                    {0, 2, 1, 3}, true, false, 64));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 2}, 16384, 6, true, true,  8, 30,
                                                    GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                                                    {0, 2, 1, 3}, true, false, 64));
    // a short KV that leaves the folded dispatch with a single split (no split_k reduce), so the
    // fold is compared against the per-token dispatch on the non-split store path too.
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 512, 6, true, false, 0,  0,
                                                    GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0,
                                                    {0, 2, 1, 3}, true, false, 64));

    // MLA shape: the V cache is a sub-view of the K cache, with quantized KV
    test_cases.emplace_back(new test_flash_attn_ext(576, 512, 1, {20, 1},  113,   1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 1, 2, 3}, true, true));
    test_cases.emplace_back(new test_flash_attn_ext(576, 512, 1, {20, 1}, 1024,   1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 1, 2, 3}, true, true));
    test_cases.emplace_back(new test_flash_attn_ext(576, 512, 1, {20, 1}, 1024,  64, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 1, 2, 3}, true, true));

    // more V-is-sub-view-of-K cases: other head shapes, and full views with equal head sizes
    test_cases.emplace_back(new test_flash_attn_ext(320, 256, 1, {32, 1}, 512, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, true));
    test_cases.emplace_back(new test_flash_attn_ext(192, 128, 4, {8, 1},  512, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, true));
    test_cases.emplace_back(new test_flash_attn_ext(128, 128, 8, {4, 1},  512, 8, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, true));
    test_cases.emplace_back(new test_flash_attn_ext(64,  64,  4, {1, 1},  512, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 1, 2, 3}, true, true));

    // large-KV F16 cases (Qwen3.6-27B geometry and a llama-class control): the upstream matrix
    // stops at kv=1024, blind to long-context FA bugs (e.g. the oneDNN SDPA ordering race on BMG).
    for (int64_t kv : { 4096, 16384 }) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 4, {6, 1}, kv, 512, true, false, 0, 0,
                                                        GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
        test_cases.emplace_back(new test_flash_attn_ext(128, 128, 8, {4, 1}, kv, 512, true, false, 0, 0,
                                                        GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    }

    // dense-allocated (non-view) quant K/V at batch >= 64, in cache and native layouts
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 4, {1, 1}, 512, 75, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 2, 1, 3}, false));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 4, {4, 1}, 512, 75, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 2, 1, 3}, false));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 4, {1, 1}, 1024, 75, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 2, 1, 3}, false));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 4, {1, 1}, 512, 75, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, {0, 1, 2, 3}, false));

    test_cases.emplace_back(new test_cross_entropy_loss     (GGML_TYPE_F32, {   10, 5, 4, 3}));
    test_cases.emplace_back(new test_cross_entropy_loss     (GGML_TYPE_F32, {30000, 1, 1, 1}));
    test_cases.emplace_back(new test_cross_entropy_loss_back(GGML_TYPE_F32, {   10, 5, 4, 3}));
    test_cases.emplace_back(new test_cross_entropy_loss_back(GGML_TYPE_F32, {30000, 1, 1, 1}));

    test_cases.emplace_back(new test_opt_step_adamw(GGML_TYPE_F32, {10, 5, 4, 3}));
    test_cases.emplace_back(new test_opt_step_sgd(GGML_TYPE_F32, {10, 5, 4, 3}));

    for (ggml_type type : base_types) {
        for (bool with_gate : {false, true}) {
            for (bool use_id : {false, true}) {
                for (bool b : {false, true}) {
                    if (!use_id && b) {
                        continue;
                    }
                    for (bool with_bias : {false, true}) {
                        if (!with_gate && !with_bias) {
                            continue;
                        }
                        for (ggml_glu_op glu_op : {GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU}) {
                            if (!with_bias && glu_op == GGML_GLU_OP_SWIGLU_OAI) {
                                continue;
                            }
                            if (!with_gate && glu_op != GGML_GLU_OP_SWIGLU) {
                                continue;
                            }
                            for (bool with_lane_scale : {false, true}) {
                                if (with_lane_scale && type != GGML_TYPE_NVFP4) {
                                    continue;
                                }
                                test_cases.emplace_back(new test_mul_mat_vec_fusion(type, glu_op, 1, 32, 256,
                                    use_id, 16, 8, b, with_bias, with_gate, with_lane_scale));
                                test_cases.emplace_back(new test_mul_mat_vec_fusion(type, glu_op, 1, 32, 256,
                                    use_id, 16, 8, b, with_bias, with_gate, with_lane_scale, {1, 1}));
                                if (!use_id && with_gate && !with_bias) {
                                    // small multi-token batches (speculative decoding / MTP verify)
                                    for (int64_t m_batch : { 2, 4, 8 }) {
                                        test_cases.emplace_back(new test_mul_mat_vec_fusion(type, glu_op, m_batch, 32, 256,
                                            use_id, 16, 8, b, with_bias, with_gate, with_lane_scale, {1, 1}));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto gate : {GATING_FUNC_SOFTMAX, GATING_FUNC_SIGMOID, GATING_FUNC_SOFTMAX_WEIGHT, GATING_FUNC_SQRT_SOFTPLUS}) {
        for (bool with_norm : {false, true}) {
            for (bool bias_probs : {false, true}) {
                for (float scale_w : {0.0f, 2.0f}) {
                    test_cases.emplace_back(new test_topk_moe({8, 22, 1, 1}, 4, with_norm, bias_probs, gate, scale_w));
                    test_cases.emplace_back(new test_topk_moe({31, 22, 1, 1}, 8, with_norm, bias_probs, gate, scale_w));
                    test_cases.emplace_back(new test_topk_moe({32, 22, 1, 1}, 8, with_norm, bias_probs, gate, scale_w));
                    test_cases.emplace_back(new test_topk_moe({40, 22, 1, 1}, 8, with_norm, bias_probs, gate, scale_w));
                    test_cases.emplace_back(new test_topk_moe({71, 22, 1, 1}, 8, with_norm, bias_probs, gate, scale_w));
                    test_cases.emplace_back(new test_topk_moe({128, 1, 1, 1}, 128, with_norm, bias_probs, gate, scale_w));
                    test_cases.emplace_back(new test_topk_moe({129, 1, 1, 1}, 128, with_norm, bias_probs, gate, scale_w));
                    test_cases.emplace_back(new test_topk_moe({160, 4, 1, 1}, 160, with_norm, bias_probs, gate, scale_w));
                    test_cases.emplace_back(new test_topk_moe({256, 22, 1, 1}, 6, with_norm, bias_probs, gate, scale_w)); // Used by DeepSeek-V4
                    test_cases.emplace_back(new test_topk_moe({288, 22, 1, 1}, 8, with_norm, bias_probs, gate, scale_w)); // Used by StepFun 3.7
                }
            }
        }
    }

    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 128, 1, 1));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 16, 1, 1));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 16, 1, 1, 1, true, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 16, 1, 1, 1, false, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 16, 64, 1, 2));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 4, 1));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 4, 2));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 8, 32, 4, 2, 2));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 4, 2, 1, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 4, 1, 1, true));
    // KDA (vector gate)
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 1, 1, 1, false, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 1, 2, 1, false, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 16, 1, 2, 1, false, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 32, 4, 1, 1, false, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 4, 2, 1, false, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 8, 32, 4, 2, 2, false, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 4, 2, 1, true,  true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 16, 4, 2, 1, true,  true));
    // chunked path: multi-chunk and non-multiple-of-chunk-size (chunk_size=64 GDN, 16 KDA)
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64,  64, 1));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 127, 1));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 256, 1));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64,  65, 1));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 100, 1));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 200, 1));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 127, 2));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64,  64, 1, 1, false, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64,  33, 1, 1, false, true));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64, 100, 1, 1, false, true));

    // K > 1: output keeps the last min(n_tokens, K) per-token snapshots, ordered most-recent-first
    // (slot 0 = final state, slot s = state s tokens back).
    // exact-match cases (K == n_seq_tokens):
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 16,   2, 1, 1, false, false, /*K=*/2));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 32,   4, 1, 1, false, false, /*K=*/4));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64,   4, 2, 1, false, false, /*K=*/4));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 8, 128,  4, 1, 1, false, false, /*K=*/4));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64,   4, 2, 1, false, true,  /*K=*/4));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 8, 32,   4, 2, 2, false, true,  /*K=*/4));
    // overflow: n_tokens > K — only the last K snapshots kept.
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 32,   8, 1, 1, false, false, /*K=*/3));
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 64,  16, 2, 1, false, false, /*K=*/4));

#if 0
    // these tests are disabled to save execution time, sbut they can be handy for debugging
    test_cases.emplace_back(new test_llama(2, true));
    test_cases.emplace_back(new test_llama(1));
    test_cases.emplace_back(new test_llama(2));
    test_cases.emplace_back(new test_falcon(1));
    test_cases.emplace_back(new test_falcon(2));
#endif

    // lightning_indexer
    for (int kv : { 256 }) {
        for (int bs : { 1, 512 }) {
            for (int nh : { 32, 64 }) {
                for (auto [ns, nm] : { std::pair{1, 1}, std::pair{4, 4}, std::pair{4, 1} }) {
                    for (ggml_type type_K : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL}) {
                        test_cases.emplace_back(new test_lightning_indexer(128, nh, kv, bs, ns, nm, type_K));
                    }
                }
            }
        }
    }

    for (int kv : { 1, 7, 8, 63, 64, 65 }) {
        for (ggml_type type_K : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0}) {
            test_cases.emplace_back(new test_lightning_indexer(128, 64, kv, 32, 4, 1, type_K));
        }
    }

    return test_cases;
}
#ifdef _MSC_VER
#pragma optimize("", on)
#endif

// Test cases for performance evaluation: should be representative of real-world use cases
static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {
    std::vector<std::unique_ptr<test_case>> test_cases;

    // [TAG_MMID_TILE_B1] [TAG_MMID_PRECAST_B] B1: the production expert GEMMs (qwen4exp gate/up q4_K 640 x 2560, 512 experts,
    // 10 used, broadcast B; down q5_0 / q8_0 2560 x 640) at prefill widths; GGML_VK_MMID_PRECAST_B / _TILE_BM128 = 0 / 1 A/B
    for (int64_t n : { 128, 512, 2044, 2048 }) {
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_K, GGML_TYPE_F32, 512, 10, true, 640, n, 2560, MMID_ROUTER_SKEWED));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q5_0, GGML_TYPE_F32, 512, 10, false, 2560, n, 640, MMID_ROUTER_SKEWED));
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q8_0, GGML_TYPE_F32, 512, 10, false, 2560, n, 640, MMID_ROUTER_SKEWED));
    }

    // [TAG_HC_INJECT_GEMM] B4 item 3: the qwen4exp hc_mix GEMMs of one prefill ubatch (n = 2048 tokens,
    // k = hc_dim = 10240) at the brain's hc weight types - the m = 4 inject projection (the N-parallel
    // question) next to the m = 320 down projection and the m = 10240 up projection (k = 320) for scale
    for (ggml_type type_a : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_F32}) {
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32,     4, 2048, 10240, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32,   320, 2048, 10240, {1, 1}, {1, 1}));
        if (type_a == GGML_TYPE_Q5_0 || type_a == GGML_TYPE_Q8_0 || type_a == GGML_TYPE_F32) {
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 10240, 2048,   320, {1, 1}, {1, 1}));
        }
    }
    // [TAG_MMVQ_HOIST] C1: the N-sweep of the decode mat-vec shapes (sol 2026-10-09 G2: router f32 512 x 2560, HC q5_0
    // 10240 x 320, HC q4_K 320 x 10240 + inject 4 x 10240, q4_K 2560 x 6144, q4_K 10240 x 2560, q6_K lm_head-like 10240 x 2560)
    for (int64_t n = 1; n <= 8; ++n) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32,  GGML_TYPE_F32,   512, n,  2560, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q5_0, GGML_TYPE_F32, 10240, n,   320, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,   320, n, 10240, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,     4, n, 10240, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,  2560, n,  6144, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 10240, n,  2560, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 10240, n,  2560, {1, 1}, {1, 1}));
    }
    // [TAG_DENSE_TILE_B2] B2: the production dense prefill GEMMs at n 128 / 512 / 2044 / 2048 (q4_K 10240 x 2560 and
    // 6144 x 2560, q4_K 320 x 10240, q5_0 10240 x 320; m x k)
    for (int64_t n : { 128, 512, 2044, 2048 }) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 10240, n,  2560, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,  6144, n,  2560, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,   320, n, 10240, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q5_0, GGML_TYPE_F32, 10240, n,   320, {1, 1}, {1, 1}));
    }
    // [TAG_DENSE_TILE_G1] G1: the P.1 dense prefill GEMMs (docs/verification/2026-10-15-e0-gemm-screen.md §3, m x k) at a
    // full 2048 ubatch and at n 512; tools/perf/g1-sweep.sh perf runs them (-p on these shapes) once per tile config. q4_K
    // 10240 / 6144 x 2560, q4_K 320 x 10240 and q5_0 10240 x 320 come from the B2 block above (same n)
    for (int64_t n : { 512, 2048 }) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32,  2560, n,  6144, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 10240, n,  2560, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q4_K, GGML_TYPE_F32, 12288, n,  2560, {1, 1}, {1, 1}));
        if (n != 2048) {   // q6_K 320 x 10240 at n 2048 = the HC block above
            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q6_K, GGML_TYPE_F32, 320, n, 10240, {1, 1}, {1, 1}));
        }
    }
    // [TAG_B4_CONCAT_COST] B4 sizing: the GDN conv-input concat (state | x^T) and its ssm_conv at a 2048-token
    // ubatch (10240 channels, kernel 4), and the QSA chunk concat chain (left-deep, 9 chunks of ~228 rows of a
    // 4096-wide attention output; the k-th concat copies k+1 chunks)
    test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {3, 10240, 1, 1}, 2048, 0, 16));
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {2051, 10240, 1, 1}, {4, 10240, 1, 1}));
    // [TAG_CONV_INPUT_SPLIT] the whole GDN conv input + conv + silu: GGML_VK_FUSE_CONCAT_SSM_CONV=1 (default) vs 0
    test_cases.emplace_back(new test_concat_ssm_conv_fuse(10240, 2048, 1, 1, false));
    for (int64_t k = 1; k <= 8; ++k) {
        test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {4096, 228*k, 1, 1}, 228, 1, 0));
    }

    // [TAG_QSA_TOPK_V2] the QSA selection (GGML_OP_QSA_TOP_K, width 2051) at decode (1) / verify (6) depths,
    // deployed data layout, plus random data and prefill-chunk query batches: GGML_VK_QSA_TOPK=v1|v2 A/B
    for (int64_t n_kv : { 65536, 131072, 262144, 524288 }) {
        for (int64_t n_tps : { 1, 6 }) {
            test_cases.emplace_back(new test_qsa_top_k_exact(n_kv, n_tps, 1, 2051, test_qsa_top_k_exact::M_PROD));
        }
    }
    test_cases.emplace_back(new test_qsa_top_k_exact(262144, 1,   1, 2051, test_qsa_top_k_exact::M_RANDOM));
    test_cases.emplace_back(new test_qsa_top_k_exact(262144, 6,   1, 2051, test_qsa_top_k_exact::M_RANDOM));
    test_cases.emplace_back(new test_qsa_top_k_exact(131072, 64,  1, 2051, test_qsa_top_k_exact::M_PROD));
    test_cases.emplace_back(new test_qsa_top_k_exact(131072, 256, 1, 2051, test_qsa_top_k_exact::M_PROD));
    // [TAG_QSA_WHOLE_BLOCKS] the block-level form at the same depths, census-shaped inputs, 512 blocks
    for (int64_t n_kv : { 131072, 262144, 524288 }) {
        for (int64_t n_tps : { 1, 6 }) {
            test_cases.emplace_back(new test_qsa_top_k_exact(n_kv, n_tps, 1, 2051, test_qsa_top_k_exact::M_CENSUS, 512));
        }
    }

    // [TAG_QSA_POOL_REMAP] the indexer score at decode (Q 1: N 4) / verify (Q 6: N 24) at pool depths 131072 / 262144 /
    // 524288 cells (32768 / 65536 / 131072 blocks): the op (form 0) against the copy path it replaces, get_rows
    // (form 1) + mul_mat of the copy (form 2); remap near-identity (prod) and fully shuffled (perm)
    for (int64_t M : { 32768, 65536, 131072 }) {
        for (int64_t N : { 4, 24 }) {
            for (int form = 0; form <= 2; form++) {
                test_cases.emplace_back(new test_mul_mat_rows(128, M, N, 1, test_mul_mat_rows::M_PROD, false, form));
            }
            test_cases.emplace_back(new test_mul_mat_rows(128, M, N, 1, test_mul_mat_rows::M_PERM, false, 0));
        }
    }

    // [TAG_QSA_SCORE_EPILOGUE] the prefill indexer score per 128-query chunk at pool depths 65536 / 262144 / 524288 /
    // 1048576 cells (16384 / 65536 / 131072 / 262144 blocks), the deployed form (store + remap, the QSA_BIAS constants):
    // (the chain it replaces is timed arm against arm by the A1 classifier: the perf harness repeats only the output node)
    for (int64_t M : { 16384, 65536, 131072, 262144 }) {
        test_cases.emplace_back(new test_qsa_score(128, M, 128, 1, 4, true, 1, false));
    }
    // [TAG_QSA_POOLED_F16_MIRROR] decode / verify rounds at 1M (262144 blocks) and 524k: widths 3..6 (N 12..24), f32 store vs
    // the f16 mirror (the mirror's set_rows of every row is part of the f16a graph: subtract test-backend-ops' SET_ROWS time
    // or read the perf logger's QSA_SCORE line)
    for (int64_t M : { 131072, 262144 }) {
        for (int64_t N : { 3, 4, 5, 6 }) {
            test_cases.emplace_back(new test_qsa_score(128, M, N, 1, 4, true, 1, false));
            test_cases.emplace_back(new test_qsa_score(128, M, N, 1, 4, true, 1, false, true));
        }
    }

    // [TAG_QSA_POOL_REMAP] the selected-K/V gather of the same verify graph ([TAG_QSA_GATHER_Q8]: q8_0 cells of 2 x 256
    // values = 136 words copied by 512-invocation workgroups), Q = 1 / 6 windows of 2051 cells from a 524288-cell cache
    test_cases.emplace_back(new test_get_rows_type(GGML_TYPE_Q8_0, 512, 524288, 2051,   1, 1, false));
    test_cases.emplace_back(new test_get_rows_type(GGML_TYPE_Q8_0, 512, 524288, 2051*6, 1, 1, false));

    // SWIGLU at a 27B-class FFN width, fused [gate|up] vs split operands
    // note: same bytes either way, so a backend that indexes them differently shows it here
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int64_t n_tokens : {512, 2048}) {
            test_cases.emplace_back(new test_glu(GGML_GLU_OP_SWIGLU, type, { 2*17408, n_tokens, 1, 1 }, 0, false));
            test_cases.emplace_back(new test_glu_split(GGML_GLU_OP_SWIGLU, type, { 17408, n_tokens, 1, 1 }, 0));
        }
    }

    // CONT of a 0<->2 permute at DeepSeek-V4 lightning-indexer shapes:
    // indexer_kq is [n_kv, n_tokens, n_head=64] and gets ggml_cont(ggml_permute(.., 2,1,0,3)).
    for (int64_t n_kv : { 1024, 1280, 2048, 2304 }) {
        test_cases.emplace_back(new test_cont(
            GGML_TYPE_F32, {n_kv, 64, 64, 1}, false, {2, 1, 0, 3}));
    }
    for (int64_t n_kv : { 2048, 2304 }) {
        test_cases.emplace_back(new test_cont(
            GGML_TYPE_F32, {n_kv, 512, 64, 1}, false, {2, 1, 0, 3}));
    }

    // Conv2d: K=CRS=NPQ=4096 matmul performance
    uint32_t                        iwh_idx  = 0;
    uint32_t                        kwh_idx  = 1;
    uint32_t                        Cout_idx = 2;
    uint32_t                        Cin_idx  = 3;
    uint32_t                        B_idx    = 4;
    std::vector<std::array<int, 5>> cases    = {
  //{IWH, KWH, Cout, Cin, B}
  // K=CRS=NPQ=4096 conv2d matmul performance
        {19,   4, 4096, 256, 16},
 // K=128, CRS=128, NPQ=4096
        { 19,  4, 128,  8,   16},
 // K=130, CRS=128, NPQ=4096
        { 19,  4, 130,  8,   16},
 // Edge case: K x CRS is small
        { 19,  2, 4,    4,   16},
 // A ConvNet's first layer
        { 224, 3, 8,    3,   1 },
 // A ConvNet's first layer with 2x2 convolution, and 1 channel
        { 224, 2, 8,    1,   1 },
 // A ConvNet's first layer with 2x2 convolution, and 1 channel, several images in the batch
        { 224, 2, 8,    1,   8 },
 // A middle layer of a ConvNet
        { 58,  3, 64,   32,  1 },
 // A middle layer of a ConvNet, several images in the batch
        { 58,  3, 64,   32,  8 },
 // A deep layer of a ConvNet, several images in the batch
        { 16,  3, 512,  128, 8 },
 // High resolution output (large NPQ)
        {1536, 3, 64,   32,  1 },
    };

    for (auto kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (auto act_case : cases) {
            // Direct CONV_2D
            test_cases.emplace_back(new test_conv_2d(
                { act_case[iwh_idx], act_case[iwh_idx], act_case[Cin_idx], act_case[B_idx] },
                { act_case[kwh_idx], act_case[kwh_idx], act_case[Cin_idx], act_case[Cout_idx] },
                kernel_type, 1, 1, 0, 0, 1, 1, false));   // bool cwhn = false
            test_cases.emplace_back(new test_conv_2d(
                { act_case[iwh_idx], act_case[iwh_idx], act_case[Cin_idx], act_case[B_idx] },
                { act_case[kwh_idx], act_case[kwh_idx], act_case[Cin_idx], act_case[Cout_idx] },
                kernel_type, 1, 1, 0, 0, 1, 1, true));    // bool cwhn = true
        }
    }

    struct conv3d_perf_case {
        int N, IC, ID, IH, IW, OC, KD, KH, KW, s0, s1, s2, p0, p1, p2, d0, d1, d2;
    };

    const std::vector<conv3d_perf_case> conv3d_cases = {
        {1,  320, 8,  38,  26, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1280, 8,  38,  26, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1,  320, 8,  76,  52, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1280, 8,  76,  52, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1,  320, 8, 152, 104, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
#if 0
        // too slow on some devices
        {1, 1280, 8, 152, 104, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1,  320, 4, 304, 208, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1,  640, 4, 304, 208, 1280, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1},
#endif
    };

    for (ggml_type kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (const conv3d_perf_case & c : conv3d_cases) {
            test_cases.emplace_back(new test_conv_3d(
                c.N, c.IC, c.ID, c.IH, c.IW,
                c.OC, c.KD, c.KH, c.KW,
                c.s0, c.s1, c.s2, c.p0, c.p1, c.p2, c.d0, c.d1, c.d2,
                kernel_type));
        }
    }

    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {4096, 1, 1, 1}, {1,   1, 1, 1}));
    test_cases.emplace_back(new test_bin_bcast(ggml_add, GGML_TYPE_F32, {4096, 1, 1, 1}, {1, 512, 1, 1}));

    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F16,  {512, 3072, 1, 1}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F32,  {8192, 512, 2, 1}, {-1,-1,-1,-1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_F32,  {3072, 512, 2, 1}, {-1,-1,-1,-1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32,  GGML_TYPE_Q4_0, {8192, 512, 2, 1}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_Q4_0, GGML_TYPE_F32,  {8192, 512, 2, 1}));

    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {768*1024, 256, 1, 1}, {-1,-1,-1,-1}, {1, 0, 2, 3}, {0, 0, 0, 0}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F16, GGML_TYPE_F16, {768*1024, 256, 1, 1}, {-1,-1,-1,-1}, {1, 0, 2, 3}, {0, 0, 0, 0}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F16, GGML_TYPE_F16, {768, 1024, 256, 1}, {-1,-1,-1,-1}, {1, 0, 2, 3}, {0, 0, 0, 0}));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_BF16, GGML_TYPE_BF16, {768, 1024, 256, 1}, {-1,-1,-1,-1}, {1, 0, 2, 3}, {0, 0, 0, 0}));

    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {768*1024, 256, 1, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F32, GGML_TYPE_F32, {768, 1024, 256, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F16, GGML_TYPE_F16, {768*1024, 256, 1, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_F16, GGML_TYPE_F16, {768, 1024, 256, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));
    test_cases.emplace_back(new test_cpy(GGML_TYPE_BF16, GGML_TYPE_BF16, {768, 1024, 256, 1}, {-1,-1,-1,-1}, {0, 0, 0, 0}, {0, 0, 0, 0}, true));

    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {4096, 4096, 5, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {12888, 256, 5, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {77, 4096, 5, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {1024, 1024, 10, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {77, 1024, 10, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {256, 256, 20, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {64, 64, 20, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {77, 64, 20, 1}, false, false, GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));

    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {32, 10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {1024, 10, 1, 1}));
    test_cases.emplace_back(new test_argmax(GGML_TYPE_F32, {32000, 512, 1, 1}));

    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {512, 34, 2, 1}));
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 80, 1, 1}));
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 80, 4, 1}));
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 384, 1, 1}));
    test_cases.emplace_back(new test_pad_reflect_1d(GGML_TYPE_F32, {3000, 384, 4, 1}));

    // SNAKE activation fusion at BigVGAN scale (T=7680 = 24 kHz x 320 ms, C=192)
    test_cases.emplace_back(new test_snake_fuse(GGML_TYPE_F32,  {7680, 192, 1, 1}));
    test_cases.emplace_back(new test_snake_fuse(GGML_TYPE_F16,  {7680, 192, 1, 1}));
    test_cases.emplace_back(new test_snake_fuse(GGML_TYPE_BF16, {7680, 192, 1, 1}));

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 16416, 1, 128, {8,  1}, {4, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128, 1, 16416, {8,  1}, {4, 1}, {0, 1, 2, 3}, 2*16416));

    // FWHT tests
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 128, 1, 128));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 64, 1, 64));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 256, 1, 256));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 128, 32, 128));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 64, 2048, 64));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 128, 2048, 128));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 256, 2048, 256));
    test_cases.emplace_back(new test_mul_mat_hadamard(GGML_TYPE_F32, GGML_TYPE_F32, 512, 2048, 512));

    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 64, 64, 4, 4 }, { 32, 64, 4, 4 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 128, 128, 4, 2 }, { 32, 128, 4, 2 }));
    // qwen3next with CHUNK_SIZE 64
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 64, 64, 8, 32 }, { 64, 64, 8, 32 }));
    // qwen3next with CHUNK_SIZE 128
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 128, 128, 4, 32 }, { 128, 128, 4, 32 }));
    test_cases.emplace_back(new test_solve_tri(GGML_TYPE_F32, { 256, 256, 4, 2 }, { 128, 256, 4, 2 }));

    test_cases.emplace_back(new test_tri(GGML_TRI_TYPE_LOWER, GGML_TYPE_F32, { 256, 256, 4, 4 }));
    test_cases.emplace_back(new test_tri(GGML_TRI_TYPE_UPPER_DIAG, GGML_TYPE_F32, { 1024, 1024, 8, 4 }));

    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 128, 128, 4, 4 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 2048, 16, 5, 4 }));
    test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, { 20000, 10, 4, 1 }));

    for (int bs : {1, 2, 3, 4, 5, 8, 512}) {
        for (ggml_type type_a : all_types) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 4096, bs, 14336, {1,  1}, {1, 1}));
            }
        }
    }

    // qwen3-30b-a3b
    for (int bs : {1, 4, 8, 32, 64, 128, 256, 512}) {
        for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS}) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, 128, 8, false, 768, bs, 2048));
                test_cases.emplace_back(new test_mul_mat_id_fusion(type_a, type_b, 128, 8, false, 768, bs, 2048, 1));
            }
        }
    }

    for (int bs : {1, 4, 8, 32, 64, 128, 256, 512}) {
        for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS}) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, 32, 4, false, 1792, bs, 2048));
                test_cases.emplace_back(new test_mul_mat_id_fusion(type_a, type_b, 32, 4, false, 1792, bs, 2048, 1));
            }
        }
    }


    // gpt-oss-20b
    for (int bs : {1, 4, 8, 512}) {
        for (ggml_type type_a : {GGML_TYPE_MXFP4}) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, 32, 4, false, 2880, bs, 2880));
                test_cases.emplace_back(new test_mul_mat_id_fusion(type_a, type_b, 32, 4, false, 2880, bs, 2880, 1));
            }
        }
    }

    for (int K : {3, 5}) {
        for (int IC : {256, 2560}) {
            for (int IW_IH : {32, 64, 256}) {
                if (IC == 2560 && IW_IH == 256) {
                    // too big
                    continue;
                }
                test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {IW_IH, IW_IH, IC, 1}, {K, K, IC, 1}, 1, 1, 1, 1, 1, 1, true));
            }
        }
    }

    // Qwen3-VL-8B https://github.com/ggml-org/llama.cpp/issues/17012
    test_cases.emplace_back(new test_flash_attn_ext(72, 72, 16, {1, 1}, 5776, 5776, false, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));

    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 8, {8, 1}, 7680, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 8, {8, 1}, 7680, 4, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 8, {8, 1}, 7680,   1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 8, {8, 1}, 7680, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 8, {8, 1}, 7680,   1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(64, 64, 8, {8, 1}, 7680, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));

    // q8_0 KV cases with long context (decode and prompt)
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},   128, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},   512, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},  1024, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},  2048, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1},  4096, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 10000, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 20000, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 10000, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 20000, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 10000, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 20000, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 10000, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {16, 1}, 20000, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));

    for (int kv : { 4096, 8192, 16384, }) {
        for (int hs : { 64, 128, }) {
            for (int nr : { 1, 4, }) {
                test_cases.emplace_back(new test_flash_attn_ext(hs, hs, 8, {nr, 1}, kv, 1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16));
            }
        }
    }

    for (int col : {8192, 16384, 32768, 65536, 131072, 262144, 524288}) {
        for (int rows : {1, 4, 16}){
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {col, rows, 1, 1}, false,  false,  GGML_TYPE_F32, {1, 1}, 1.0f, 0.0f));
        }
    }

    test_cases.emplace_back(new test_conv_2d_dw({512, 512, 256, 1}, {3, 3, 1, 256}, GGML_TYPE_F32, 1, 1, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({512, 512, 256, 1}, {3, 3, 1, 256}, GGML_TYPE_F32, 1, 1, 1, true));
    test_cases.emplace_back(new test_conv_2d_dw({112, 112, 32,  1}, {3, 3, 1, 32},  GGML_TYPE_F32, 1, 1, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({112, 112, 32,  1}, {3, 3, 1, 32},  GGML_TYPE_F32, 1, 1, 1, true));
    test_cases.emplace_back(new test_conv_2d_dw({56,  56,  128, 1}, {5, 5, 1, 128}, GGML_TYPE_F32, 2, 2, 1, false));
    test_cases.emplace_back(new test_conv_2d_dw({56,  56,  128, 1}, {5, 5, 1, 128}, GGML_TYPE_F32, 2, 2, 1, true));

    for (ggml_type kernel_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        test_cases.emplace_back(new test_conv_transpose_2d({256, 256, 256, 1}, {3, 3, 16, 256}, 1, kernel_type));
        test_cases.emplace_back(new test_conv_transpose_2d({16, 16, 16, 1}, {3, 3, 8, 16}, 1, kernel_type));
        test_cases.emplace_back(new test_conv_transpose_2d({10, 10, 9, 1}, {3, 3, 1, 9}, 2, kernel_type));
    }

    // Memory bound overlap-add of the GEMM + col2im_1d transposed conv path, real vocoder stage shapes
    test_cases.emplace_back(new test_col2im_1d(GGML_TYPE_F32, 16, 512, 2048, 8, 0));
    test_cases.emplace_back(new test_col2im_1d(GGML_TYPE_F32, 4, 128, 65536, 2, 0));
    test_cases.emplace_back(new test_col2im_1d(GGML_TYPE_F16, 16, 512, 2048, 8, 0));

    test_cases.emplace_back(new test_mean(GGML_TYPE_F32, {256, 256, 3, 1}));


    for (int n_token : {1, 512}) {
        test_cases.emplace_back(new test_add_id(GGML_TYPE_F32, GGML_TYPE_F32, 2880, 128, 4, n_token));
        test_cases.emplace_back(new test_add_id(GGML_TYPE_F32, GGML_TYPE_F32, 2880, 32, 4, n_token));
    }

    for (bool fw : {true, false}) { // fw == forward
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (bool ff : {false, true}) { // freq_factors
                for (float v : { 0, 1 }) {
                    test_cases.emplace_back(new test_rope(type, {128,  32, 512, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, 1.0f, 0.0f, 1.0f, ff, v, fw)); // llama 7B
                    test_cases.emplace_back(new test_rope(type, {128,  64, 512, 1}, 128, GGML_ROPE_TYPE_NORMAL, 512, 1.0f, 0.0f, 1.0f, ff, v, fw)); // llama 65B
                    test_cases.emplace_back(new test_rope(type, { 80,  32, 512, 1},  20, GGML_ROPE_TYPE_NEOX, 512, 1.0f, 0.0f, 1.0f, ff, v, fw)); // neox (stablelm)
                    test_cases.emplace_back(new test_rope(type, { 64,   8, 512, 1},  64, GGML_ROPE_TYPE_NEOX, 512, 1.0f, 0.0f, 1.0f, ff, v, fw)); // neox (falcon 40B)
                    test_cases.emplace_back(new test_rope(type, {128,  12, 512, 1}, 128, GGML_ROPE_TYPE_MROPE,  512, 1.0f, 0.0f, 1.0f, ff, v, fw)); // rope_multi,m-rope (qwen2vl 2B)
                    test_cases.emplace_back(new test_rope(type, {128,  12, 512, 1}, 128, GGML_ROPE_TYPE_IMROPE,  512, 1.0f, 0.0f, 1.0f, ff, v, fw)); // rope_multi,imrope (qwen3vl 2B)
                    test_cases.emplace_back(new test_rope(type, { 80,  16, 2, 1},  80, GGML_ROPE_TYPE_VISION, 512, 1.0f, 0.0f, 1.0f, ff, v, fw)); // rope_multi,m-rope (qwen2vl ViT)
                }
            }
        }
    }

    std::vector<std::array<int64_t, 4>> reduce_rows_cases = {
        { 8192, 1,    1, 1 },
        { 8192, 8192, 1, 1 },
        { 128,  8192, 1, 1 },
    };

    for (auto it: reduce_rows_cases){
        test_cases.emplace_back(new test_mean(GGML_TYPE_F32, it));
        test_cases.emplace_back(new test_sum_rows(GGML_TYPE_F32, it));
        test_cases.emplace_back(new test_sum(GGML_TYPE_F32, it));
    }

    test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {65000,  16, 1, 1}));
    test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {200000, 1,  1, 1}));
    test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {200000, 16, 1, 1}));

    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {2, 1, 1, 1}, 1));
    // widths around the tiling threshold
    for (auto cols : {4096, 8192, 12288, 16384, 24576, 32768, 65536, 131072}) {
        for (auto nrows : {1, 16}) {
            test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {cols, nrows, 1, 1}, 16));
        }
    }
    for (auto k : {1, 4, 8, 10, 16, 32, 40, 400}) {
        for (auto nrows : {1, 16}) {
            for (auto cols : {k, 1000, 65000, 200000}) {
                test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {cols, nrows, 1, 1}, k));
            }
        }
    }

    for (auto nrows : {1, 4, 8, 16}) {
        for (auto cols : {128, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000, 2000000}) {
            test_cases.emplace_back(new test_cumsum(GGML_TYPE_F32, {cols, nrows, 1, 1}));
        }
    }

    // Examples from granite-4.0-h-1b/ggml-model-Q8_0.gguf
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {515, 3328, 1, 1}, {4, 3328, 1, 1})); // prefill
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {937, 8192, 1, 1}, {4, 8192, 1, 1})); // prefill
    test_cases.emplace_back(new test_ssm_conv(GGML_TYPE_F32, {4,   3328, 1, 1}, {4, 3328, 1, 1})); // generate
    test_cases.emplace_back(new test_ssm_conv_bias_silu(GGML_TYPE_F32, {515, 3328, 1, 1}, {4, 3328, 1, 1}, true));  // prefill
    test_cases.emplace_back(new test_ssm_conv_bias_silu(GGML_TYPE_F32, {4,   3328, 1, 1}, {4, 3328, 1, 1}, true));  // generate
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 48, 1, 512, 1)); // prefill
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 64, 48, 1, 1,   1)); // generate
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 80, 128, 1, 512, 1)); // Nemotron-9B prefill
    test_cases.emplace_back(new test_ssm_scan(GGML_TYPE_F32, 128, 80, 128, 1, 1,   1)); // Nemotron-9B generate

    // acc
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 1, 1}, {256, 16, 1, 1}, -1));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {256, 16, 2, 3}, -1));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {128, 16, 2, 3}, -1));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {256, 16, 2, 3}, 1));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {128, 16, 2, 3}, 2));
    test_cases.emplace_back(new test_acc(GGML_TYPE_F32, {256, 17, 2, 3}, {64, 16, 2, 3}, 3));

    // GATED_DELTA_NET: realistic model configurations
    // TG: n_seq_tokens=1 (autoregressive)
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 128, 1, 1));   // Qwen3.5-like: 32 heads, d=128
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 16, 64,  1, 1));   // smaller model
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 128, 1, 1, 1, false, true)); // KDA
    // PP: n_seq_tokens=64,256 (prompt processing)
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 128, 64, 1));  // PP-64
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 128, 256, 1)); // PP-256
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 128, 512, 1)); // PP-512
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 128, 1024, 1)); // PP-1024
    // Small model configs (fewer heads = less GPU occupancy for autoregressive)
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 128, 64, 1));   // 4h PP-64
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 128, 256, 1));  // 4h PP-256
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 128, 512, 1));  // 4h PP-512
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 4, 128, 1024, 1)); // 4h PP-1024
    test_cases.emplace_back(new test_gated_delta_net(GGML_TYPE_F32, 32, 128, 64, 1, 1, false, true)); // KDA PP-64

    // lightning_indexer
    for (int kv : { 256, 4096, 65536 }) {
        for (int bs : { 1, 512, 2048 }) {
            for (int nh : { 32, 64 }) {
                for (int ns : { 1, 4 }) {
                    for (ggml_type type_K : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL}) {
                        test_cases.emplace_back(new test_lightning_indexer(128, nh, kv, bs, ns, ns, type_K));
                    }
                }
            }
        }
    }

    return test_cases;
}

static std::vector<std::unique_ptr<test_case>> make_test_cases_from_file(const char * path) {
    std::ifstream f(path);

    if (!f.is_open()) {
        throw std::runtime_error("Unable to read test file");
    }

    std::vector<std::unique_ptr<test_case>> test_cases;

    std::string line;

    while (std::getline(f, line)) {
        std::istringstream iss(line);

        ggml_op op;
        ggml_type type;
        std::array<int64_t, 4> ne;
        std::array<int32_t, GGML_MAX_OP_PARAMS / sizeof(int32_t)> op_params = {};
        std::string name;
        uint64_t tmp;

        iss >> tmp;
        op = (ggml_op)tmp;
        iss >> tmp;
        type = (ggml_type)tmp;

        for (size_t i = 0; i < 4; i++) {
            iss >> ne[i];
        }

        iss >> tmp;
        for (size_t i = 0; i < tmp && i < op_params.size(); i++) {
            iss >> op_params[i];
        }

        iss >> tmp;

        size_t num_src = std::min((uint64_t)GGML_MAX_SRC, tmp);
        std::vector<input_tensor> sources(num_src);
        for (size_t i = 0; i < num_src; i++) {
            input_tensor& src = sources[i];
            iss >> tmp;
            src.type = (ggml_type)tmp;

            for (size_t i = 0; i < 4; i++) {
                iss >> src.ne[i];
            }
            for (size_t i = 0; i < 4; i++) {
                iss >> src.nb[i];
            }
        }

        iss >> name;

        if (name.length() == 1 && name[0] == '-') {
            name = "";
        }

        test_cases.emplace_back(new test_generic_op(op, type, ne, op_params, sources, std::move(name)));
    }

    return test_cases;
}

// ---- FA vec (Q,NE): forced-config numerical slice (Metal only) ----
using set_fa_vec_override_t   = void (*)(int, int);
using clear_fa_vec_override_t = void (*)(void);

// NL = 32/NE must divide both dk/4 and dv/4.
static std::vector<int> fa_vec_legal_ne(int dk, int dv) {
    std::vector<int> r;
    for (int ne : {1, 2, 4}) {
        const int nl = 32 / ne;
        if ((dk/4) % nl == 0 && (dv/4) % nl == 0) {
            r.push_back(ne);
        }
    }
    return r;
}

static bool op_names_filter_selects(const char * op_names_filter, const char * op_name) {
    if (!op_names_filter) {
        return true;
    }
    std::string_view filter(op_names_filter);
    while (!filter.empty()) {
        auto comma_pos = filter.find_first_of(',');
        const auto lparen_pos = filter.find_first_of('(');
        std::string_view entry;
        if (lparen_pos < comma_pos) {
            const auto rparen_pos = filter.find_first_of(')');
            comma_pos = filter.find_first_of(',', rparen_pos);
            entry = filter.substr(0, lparen_pos);
        } else {
            entry = filter.substr(0, comma_pos);
        }
        if (entry == op_name) {
            return true;
        }
        filter = comma_pos != std::string_view::npos ? filter.substr(comma_pos + 1) : "";
    }
    return false;
}

// Covers padded rows, sinks, kvpad, multi-SIMDgroup reduction, quantized K/V, and MLA views.
// The override is backend-global, so this runs after all parallel workers have joined.
static bool run_fa_vec_slice(ggml_backend_t backend, ggml_backend_t backend_cpu, const char * op_names_filter) {
    if (!op_names_filter_selects(op_names_filter, "FLASH_ATTN_EXT")) {
        return true;
    }

    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));

    auto set_ov   = (set_fa_vec_override_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_set_fa_vec_override");
    auto clear_ov = (clear_fa_vec_override_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_tuning_clear_fa_vec_override");
    if (!set_ov || !clear_ov) {
        return true;  // not the Metal backend: nothing to force
    }

    struct shape_t { int dk, dv; };
    const shape_t   shapes[] = { { 128, 128 }, { 576, 512 } };  // mainstream head size + MLA shared K/V view
    const int       ne01_pts[] = { 1, 3 };                      // decode, and padded rows for Q=2 and Q=4
    const int       ne11_pts[] = { 512, 4097 };                 // nsg=1, and nsg>=2 together with kvpad
    const ggml_type types[]    = { GGML_TYPE_F16, GGML_TYPE_Q4_0 };

    int n_run = 0, n_fail = 0;
    for (auto s : shapes) {
        for (int ne : fa_vec_legal_ne(s.dk, s.dv)) {
            for (int Q : { 1, 2, 4 }) {
                for (ggml_type type_kv : types) {
                    for (bool sinks : { false, true }) {
                        for (int ne01 : ne01_pts) {
                            for (int ne11 : ne11_pts) {
                                set_ov(Q, ne);
                                test_flash_attn_ext tc(s.dk, s.dv, /*nh=*/4, { 1, 1 }, /*kv=*/ne11, /*nb=*/ne01,
                                                       /*mask=*/true, sinks, 0.0f, 0.0f, GGML_PREC_F32,
                                                       type_kv, type_kv);
                                auto st = tc.eval(backend, backend_cpu, "FLASH_ATTN_EXT", nullptr);
                                clear_ov();

                                if (st == test_status_t::FAIL) {
                                    printf("  FAIL fa_vec slice: dk=%d dv=%d Q=%d ne=%d type=%s ne01=%d ne11=%d sinks=%d\n",
                                           s.dk, s.dv, Q, ne, ggml_type_name(type_kv), ne01, ne11, (int) sinks);
                                    n_fail++;
                                }
                                n_run++;
                            }
                        }
                    }
                }
            }
        }
    }

    printf("  fa_vec (Q,NE) slice: %d cases run, %d failed\n", n_run, n_fail);

    return n_fail == 0;
}

static bool test_backend(ggml_backend_t backend, ggml_backend_dev_t dev, test_mode mode, const char * op_names_filter, const char * params_filter,
                         printer * output_printer, const char * test_file_path, int parallel_workers) {
    auto filter_test_cases = [](std::vector<std::unique_ptr<test_case>> & test_cases, const char * params_filter) {
        if (params_filter == nullptr) {
            return;
        }

        std::regex params_filter_regex(params_filter);

        for (auto it = test_cases.begin(); it != test_cases.end();) {
            if (!std::regex_search((*it)->vars(), params_filter_regex)) {
                it = test_cases.erase(it);
                continue;
            }

            it++;
        }
    };

    std::vector<std::unique_ptr<test_case>> test_cases;

    if (test_file_path == nullptr) {
        switch (mode) {
        case MODE_TEST:
        case MODE_GRAD:
        case MODE_SUPPORT:
            test_cases = make_test_cases_eval();
            break;
        case MODE_PERF:
            test_cases = make_test_cases_perf();
            break;
        }
    } else {
        test_cases = make_test_cases_from_file(test_file_path);
    }

    filter_test_cases(test_cases, params_filter);

    if (mode == MODE_TEST) {
        ggml_backend_ptr backend_cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL));
        if (backend_cpu == NULL) {
            test_operation_info info("", "", "CPU");
            info.set_error("backend", "Failed to initialize CPU backend");
            output_printer->print_operation(info);
            return false;
        }
        // Use reference implementation on the CPU backend for comparison
        using ggml_backend_cpu_set_use_ref_t = void (*)(ggml_backend_t, bool);
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu.get()));
        auto * set_use_ref = (ggml_backend_cpu_set_use_ref_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_use_ref");
        if (set_use_ref) {
            set_use_ref(backend_cpu.get(), true);
        }

        std::atomic<size_t> n_ok = 0;
        std::atomic<size_t> tests_run = 0;
        std::vector<std::string> failed_tests;
        std::mutex failed_tests_mutex;

        // Each worker grabs a chunk of cases at a time. The chunk shrinks as we
        // run out of work so that a few slow tests at the tail get spread across
        // workers instead of landing on one unlucky thread.
        constexpr size_t MAX_TESTS_PER_ITER = 100;
        std::atomic<size_t> test_idx = 0;

        const auto & next_chunk = [&](size_t & my_begin, size_t & my_end) {
            const size_t cur = test_idx.load(std::memory_order_relaxed);
            const size_t remaining = cur < test_cases.size() ? test_cases.size() - cur : 0;
            const size_t chunk = std::max<size_t>(1, std::min<size_t>(MAX_TESTS_PER_ITER, remaining / parallel_workers));
            my_begin = test_idx.fetch_add(chunk);
            my_end = std::min(my_begin + chunk, test_cases.size());
        };

        const auto & run_tests = [&](ggml_backend_t b, ggml_backend_t b_cpu) {
            size_t my_begin, my_end;
            next_chunk(my_begin, my_end);
            while (my_begin < test_cases.size()) {
                for (size_t i = my_begin; i < my_end; ++i) {
                    auto & test = test_cases[i];
                    test_status_t status = test->eval(b, b_cpu, op_names_filter, output_printer);
                    if (status == test_status_t::SKIPPED || status == test_status_t::NOT_SUPPORTED) {
                        continue;
                    }
                    tests_run++;
                    if (status == test_status_t::OK) {
                        n_ok++;
                    } else if (status == test_status_t::FAIL) {
                        std::lock_guard<std::mutex> guard(failed_tests_mutex);
                        failed_tests.push_back(test->current_op_name + "(" + test->vars() + ")");
                    }
                }
                next_chunk(my_begin, my_end);
            }
        };

        if (parallel_workers <= 1) {
            // Reuse the outer backend / backend_cpu so we don't pay an
            // extra CPU backend init.
            run_tests(backend, backend_cpu.get());
        } else {
            std::atomic<size_t> workers_started = 0;

            const auto & eval_worker = [&]() {
                ggml_backend_ptr b(ggml_backend_dev_init(dev, NULL));
                if (b == NULL) {
                    return;
                }

                ggml_backend_ptr b_cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL));
                if (b_cpu == NULL) {
                    return;
                }

                if (set_use_ref) {
                    set_use_ref(b_cpu.get(), true);
                }
                workers_started++;
                run_tests(b.get(), b_cpu.get());
            };

            std::vector<std::thread> threads;
            threads.reserve(parallel_workers);
            for (int i = 0; i < parallel_workers; ++i) {
                threads.emplace_back(eval_worker);
            }
            for (auto & t : threads) {
                t.join();
            }

            if (workers_started == 0 && !test_cases.empty()) {
                return false;
            }
        }

        output_printer->print_summary(test_summary_info(n_ok, tests_run, false));
        output_printer->print_failed_tests(failed_tests);

        const bool slice_ok = run_fa_vec_slice(backend, backend_cpu.get(), op_names_filter);

        return n_ok == tests_run && slice_ok;
    }

    if (mode == MODE_GRAD) {
        test_cases.erase(
            std::remove_if(test_cases.begin(), test_cases.end(), [](const std::unique_ptr<test_case> & tc) {
                return tc->run_whole_graph();
            }),
            test_cases.end()
        );

        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval_grad(backend, op_names_filter, output_printer)) {
                n_ok++;
            }
        }
        output_printer->print_summary(test_summary_info(n_ok, test_cases.size(), false));

        return n_ok == test_cases.size();
    }

    if (mode == MODE_PERF) {
        for (auto & test : test_cases) {
            test->eval_perf(backend, op_names_filter, output_printer);
        }
        return true;
    }

    if (mode == MODE_SUPPORT) {
        // Filter out fusion cases
        test_cases.erase(
            std::remove_if(test_cases.begin(), test_cases.end(), [](const std::unique_ptr<test_case> & tc) {
                return tc->run_whole_graph();
            }),
            test_cases.end()
        );

        for (auto & test : test_cases) {
            test->eval_support(backend, op_names_filter, output_printer);
        }
        return true;
    }

    GGML_ABORT("fatal error");
}

static void list_all_ops() {
    printf("GGML operations:\n");
    std::set<std::string> all_ops;

    for (int i = 1; i < GGML_OP_COUNT; i++) {
        all_ops.insert(ggml_op_name((enum ggml_op)i));
    }
    for (int i = 0; i < GGML_UNARY_OP_COUNT; i++) {
        all_ops.insert(ggml_unary_op_name((enum ggml_unary_op)i));
    }
    for (int i = 0; i < GGML_GLU_OP_COUNT; i++) {
        all_ops.insert(ggml_glu_op_name((enum ggml_glu_op)i));
    }
    for (const auto & op : all_ops) {
        printf("  %s\n", op.c_str());
    }
    printf("\nTotal: %zu operations\n", all_ops.size());
}

static void show_test_coverage() {
    std::set<std::string> all_ops;
    for (int i = 1; i < GGML_OP_COUNT; i++) {
        auto op = (enum ggml_op)i;
        if (op == GGML_OP_VIEW      ||
            op == GGML_OP_RESHAPE   ||
            op == GGML_OP_PERMUTE   ||
            op == GGML_OP_TRANSPOSE ||
            op == GGML_OP_CONT      ||
            op == GGML_OP_GLU       ||
            op == GGML_OP_UNARY) {
            continue;
        }
        all_ops.insert(ggml_op_name(op));
    }
    for (int i = 0; i < GGML_UNARY_OP_COUNT; i++) {
        all_ops.insert(ggml_unary_op_name((enum ggml_unary_op)i));
    }
    for (int i = 0; i < GGML_GLU_OP_COUNT; i++) {
        all_ops.insert(ggml_glu_op_name((enum ggml_glu_op)i));
    }
    auto test_cases = make_test_cases_eval();
    // Filter out fusion cases
    test_cases.erase(
        std::remove_if(test_cases.begin(), test_cases.end(), [](const std::unique_ptr<test_case> & tc) {
            return tc->run_whole_graph();
        }),
        test_cases.end()
    );

    std::set<std::string> tested_ops;

    ggml_init_params params = {
        /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
        /* .mem_base = */ NULL,
        /* .no_alloc = */ true,
    };

    for (auto & test_case : test_cases) {
        ggml_context_ptr ctx(ggml_init(params));
        if (ctx) {
            test_case->mode = MODE_TEST;
            ggml_tensor * out = test_case->build_graph(ctx.get());
            if (out && out->op != GGML_OP_NONE) {
                if (out->op == GGML_OP_UNARY) {
                    tested_ops.insert(ggml_unary_op_name(ggml_get_unary_op(out)));
                } else if (out->op == GGML_OP_GLU) {
                    tested_ops.insert(ggml_glu_op_name(ggml_get_glu_op(out)));
                } else {
                    tested_ops.insert(ggml_op_name(out->op));
                }
            }
        }
    }
    std::set<std::string> covered_ops;
    std::set<std::string> uncovered_ops;
    for (const auto & op : all_ops) {
        if (tested_ops.count(op) > 0) {
            covered_ops.insert(op);
        } else {
            uncovered_ops.insert(op);
        }
    }

    printf("Operations covered by tests (%zu):\n", covered_ops.size());
    for (const auto & op : covered_ops) {
        printf("  ✓ %s\n", op.c_str());
    }
    printf("\nOperations without tests (%zu):\n", uncovered_ops.size());
    for (const auto & op : uncovered_ops) {
        printf("  ✗ %s\n", op.c_str());
    }

    printf("\nCoverage Summary:\n");
    printf("  Total operations: %zu\n", all_ops.size());
    printf("  Tested operations: %zu\n", covered_ops.size());
    printf("  Untested operations: %zu\n", uncovered_ops.size());
    printf("  Coverage: %.1f%%\n", (double)covered_ops.size() / all_ops.size() * 100.0);
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-o <op,..>] [-b <backend>] [-p <params regex>] [--output <console|sql|csv>] [--list-ops]", argv[0]);
    printf(" [--show-coverage] [--test-file <path>] [-j <n>]\n");
    printf("    valid modes:\n");
    printf("      - test (default, compare with CPU backend for correctness)\n");
    printf("      - grad (compare gradients from backpropagation with method of finite differences)\n");
    printf("      - perf (performance evaluation)\n");
    printf("      - support (probe backend operation support)\n");
    printf("    op names for -o are as given by ggml_op_desc() (e.g. ADD, MUL_MAT, etc),\n");
    printf("        optionally including the full test case string (e.g. \"ADD(type=f16,ne=[1,1,8,1],nr=[1,1,1,1],nf=1)\")\n");
    printf("    --output specifies output format (default: console, options: console, sql, csv)\n");
    printf("    --list-ops lists all available GGML operations\n");
    printf("    --show-coverage shows test coverage\n");
    printf("    --test-file reads test operators from a test file generated by test-export-graph-ops\n");
    printf("    -j <n> runs tests using <n> parallel worker threads (default: 1, test mode only)\n");
    printf("    environment:\n");
    printf("      GGML_TEST_TOPK_PROD=1    also run the selection cases at the deployed 213k-context shapes\n");
    printf("      GGML_TEST_DUMP_IDS=<dir> write each TOP_K/TOPK_QSA/QSA_TOP_K case's output ids (row-major,\n");
    printf("                               from the backend under test) to <dir>/<case>.ids, so two runs can\n");
    printf("                               be diffed byte-for-byte. <dir> must exist; use a fresh one per run\n");
}

int main(int argc, char ** argv) {
    test_mode mode = MODE_TEST;
    output_formats output_format = CONSOLE;
    const char * op_names_filter = nullptr;
    const char * backend_filter = nullptr;
    const char * params_filter = nullptr;
    const char * test_file_path = nullptr;
    int parallel_workers = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        } else if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        } else if (strcmp(argv[i], "grad") == 0) {
            mode = MODE_GRAD;
        } else if (strcmp(argv[i], "support") == 0) {
            mode = MODE_SUPPORT;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_names_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) {
                params_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                if (!output_format_from_str(argv[++i], output_format)) {
                    usage(argv);
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--list-ops") == 0) {
            list_all_ops();
            return 0;
        } else if (strcmp(argv[i], "--show-coverage") == 0) {
            show_test_coverage();
            return 0;
        } else if (strcmp(argv[i], "--test-file") == 0) {
            if (i + 1 < argc) {
                test_file_path = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-j") == 0) {
            if (i + 1 < argc) {
                parallel_workers = atoi(argv[++i]);
                if (parallel_workers < 1) {
                    usage(argv);
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        } else {
            usage(argv);
            return 1;
        }
    }

    // load and enumerate backends
    ggml_backend_load_all();

    // Create printer for output format
    std::unique_ptr<printer> output_printer = create_printer(output_format);
    if (output_printer) {
        output_printer->print_header();
    }

    output_printer->print_testing_start(testing_start_info(ggml_backend_dev_count()));

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (backend_filter != NULL && strcmp(backend_filter, ggml_backend_dev_name(dev)) != 0) {
            output_printer->print_backend_init(
                backend_init_info(i, ggml_backend_dev_count(), ggml_backend_dev_name(dev), true, "Skipping"));
            n_ok++;
            continue;
        }

        if (backend_filter == NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && mode != MODE_GRAD) {
            output_printer->print_backend_init(backend_init_info(
                i, ggml_backend_dev_count(), ggml_backend_dev_name(dev), true, "Skipping CPU backend"));
            n_ok++;
            continue;
        }

        ggml_backend_ptr backend(ggml_backend_dev_init(dev, NULL));
        GGML_ASSERT(backend != NULL);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            // TODO: better value for n_threads
            ggml_backend_set_n_threads_fn(backend.get(), N_THREADS);
        }

        size_t free, total;  // NOLINT
        ggml_backend_dev_memory(dev, &free, &total);
        output_printer->print_backend_init(backend_init_info(i, ggml_backend_dev_count(), ggml_backend_dev_name(dev),
                                                             false, "", ggml_backend_dev_description(dev),
                                                             total / 1024 / 1024, free / 1024 / 1024, true));

        bool ok = test_backend(backend.get(), dev, mode, op_names_filter, params_filter, output_printer.get(), test_file_path, parallel_workers);

        if (ok) {
            n_ok++;
        }
        output_printer->print_backend_status(
            backend_status_info(ggml_backend_name(backend.get()), ok ? test_status_t::OK : test_status_t::FAIL));
    }

    ggml_quantize_free();

    if (output_printer) {
        output_printer->print_footer();
    }

    output_printer->print_overall_summary(
        overall_summary_info(n_ok, ggml_backend_dev_count(), n_ok == ggml_backend_dev_count()));

    if (n_ok != ggml_backend_dev_count()) {
        return 1;
    }

    return 0;
}
