#pragma once
// [TAG_PLE_DIRECT] shared by test-ple-direct-read (ctest) and test-ple-direct-bench (microbench): write a synthetic GGUF
// whose tensors are filled with deterministic pseudo-random bytes, streamed (a table of tens of GiB never sits in RAM).

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

struct ple_gguf_tensor {
    std::string name;
    ggml_type   type;
    int64_t     ne0;
    int64_t     ne1;
};

// bytes of tensor i are a function of (seed, i, byte index) only: fill_bytes() regenerates any range
static inline void ple_gguf_fill(uint64_t seed, int64_t ti, uint64_t first, uint8_t * dst, size_t n) {
    // counter-based: splitmix64 of (seed, tensor, 8-byte word index)
    for (size_t k = 0; k < n; ) {
        const uint64_t pos  = first + k;
        const uint64_t word = pos >> 3;
        uint64_t z = seed ^ ((uint64_t) ti * 0xD1B54A32D192ED03ull) ^ (word * 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        for (uint64_t b = pos & 7; b < 8 && k < n; ++b, ++k) {
            dst[k] = (uint8_t) (z >> (8*b));
        }
    }
}

// writes the file; returns false on an I/O error
static inline bool ple_gguf_write(const std::string & path, const std::vector<ple_gguf_tensor> & ts, uint64_t seed) {
    ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead() * (ts.size() + 4), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "ple-direct-test");
    gguf_set_val_str(g, "general.name", "synthetic rows for the PLE direct-read tests");
    for (const auto & t : ts) {
        ggml_tensor * x = ggml_new_tensor_2d(ctx, t.type, t.ne0, t.ne1);
        ggml_set_name(x, t.name.c_str());
        gguf_add_tensor(g, x);
    }
    bool ok = gguf_write_to_file(g, path.c_str(), /*only_meta =*/ true);
    const size_t meta = gguf_get_meta_size(g);
    FILE * f = ok ? fopen(path.c_str(), "r+b") : nullptr;
    ok = f != nullptr;
    std::vector<uint8_t> buf(8u << 20);
    for (size_t i = 0; ok && i < ts.size(); ++i) {
        const size_t off   = meta + gguf_get_tensor_offset(g, (int64_t) i);
        const size_t bytes = ggml_row_size(ts[i].type, ts[i].ne0) * (size_t) ts[i].ne1;
        ok = fseeko(f, (off_t) off, SEEK_SET) == 0;
        for (size_t done = 0; ok && done < bytes; ) {
            const size_t n = std::min(buf.size(), bytes - done);
            ple_gguf_fill(seed, (int64_t) i, done, buf.data(), n);
            ok = fwrite(buf.data(), 1, n, f) == n;
            done += n;
        }
    }
    if (f) {
        ok = (fclose(f) == 0) && ok;
    }
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

// file offset of a tensor's data and its geometry, read back through the gguf parser (as the model loader does)
struct ple_gguf_info {
    size_t    offs  = 0;
    size_t    rs    = 0;
    int64_t   rows  = 0;
    ggml_type type  = GGML_TYPE_COUNT;
};

static inline bool ple_gguf_locate(const std::string & path, const std::string & name, ple_gguf_info & out) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (g == nullptr) {
        return false;
    }
    const int64_t ti = gguf_find_tensor(g, name.c_str());
    bool ok = false;
    if (ti >= 0) {
        const ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
        out.offs = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti);
        out.rs   = ggml_row_size(t->type, t->ne[0]);
        out.rows = t->ne[1];
        out.type = t->type;
        ok = true;
    }
    gguf_free(g);
    ggml_free(meta);
    return ok;
}
