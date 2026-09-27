// [TAG_KQ_VIS_FILL] the reference for llama_kv_cells_fill_kq_vis: its body before P3-F1 (631cdb55f), verbatim but for
// the name - per cell an is_empty branch, a 32-step seq_has loop and four stores. Shared by test-kq-vis-fill-equiv
// (randomized byte-for-byte equivalence) and test-kq-vis-fill-bench (the "before" row).
#pragma once

#include "../src/llama-batch.h"
#include "../src/llama-kv-cells.h"

#include "ggml.h"

#include <cstdint>
#include <vector>

static bool llama_kv_cells_fill_kq_vis_ref(
        int32_t * cells_data, int32_t * toks_data, int64_t n_kv, int64_t n_stream, const llama_ubatch * ubatch,
        const std::vector<llama_kv_cells> & v_cells, const std::vector<uint32_t> & seq_to_stream) {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream > 0 && n_tokens%n_stream == 0);

    const int64_t n_tps = n_tokens/n_stream;
    const bool    is_2d = ubatch->is_pos_2d();

    for (int64_t s = 0; s < n_stream; ++s) {
        uint32_t stream = UINT32_MAX;

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];
            if (seq_id < 0 || seq_id >= 32) {
                return false;
            }
            if (stream == UINT32_MAX) {
                stream = seq_to_stream[seq_id];
            } else if (stream != seq_to_stream[seq_id]) {
                return false;
            }

            int32_t * t = toks_data + 4*i;
            t[0] = ubatch->pos[i];
            t[1] = (int32_t) (1u << seq_id);
            t[2] = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : INT32_MAX;
            t[3] = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : INT32_MAX;
        }

        const auto & cells = v_cells.at(stream);
        GGML_ASSERT((int64_t) cells.size() >= n_kv);

        // one plane per field: pos, seqmask, x, y
        int32_t * c_pos = cells_data + 4*s*n_kv;
        int32_t * c_seq = c_pos + n_kv;
        int32_t * c_x   = c_pos + 2*n_kv;
        int32_t * c_y   = c_pos + 3*n_kv;
        for (int64_t j = 0; j < n_kv; ++j) {
            if (cells.is_empty(j)) {
                c_pos[j] = -1; c_seq[j] = 0; c_x[j] = 0; c_y[j] = 0;
                continue;
            }
            uint32_t seqmask = 0;
            for (llama_seq_id k = 0; k < 32; ++k) {
                seqmask |= cells.seq_has(j, k) ? (1u << k) : 0u;
            }
            const auto & ext = cells.ext_get(j);
            c_pos[j] = cells.pos_get(j);
            c_seq[j] = (int32_t) seqmask;
            c_x[j]   = ext.x;
            c_y[j]   = ext.y;
        }
    }

    return true;
}
