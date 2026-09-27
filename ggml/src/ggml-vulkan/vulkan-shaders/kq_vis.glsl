// [TAG_KQ_VIS] visibility from metadata: the KQ-mask value llama_kv_cache::set_input_kq_mask writes,
// computed from the per-cell / per-token records of ggml_kq_mask (see ggml.h). Shared by
// kq_mask.comp, flash_attn_vis_opt.comp and the FA shaders (through flash_attn_vis.glsl).
//
// The includer defines, before including this file:
//     int kqv_cell_load(uint idx)   - element idx of the cell planes [n_kv, 4, n_stream] (I32)
//
// cells: per stream s, base = s * 4 * n_kv: pos at base + j, seq bitmask at base + n_kv + j,
//        ext.x at base + 2 n_kv + j, ext.y at base + 3 n_kv + j
// token: ivec4 {pos, 1 << seq, x, y} (x = y = INT32_MAX without 2-D positions)
// flags: bit 0 causal, bit 1 alibi, bits 2..3 swa type (0 none, 1 standard, 2 chunked, 3 symmetric)
//
// Every value returned is exactly representable in f16 (+0, -inf, or -n with n rounded to f16
// precision), so converting it to float16_t is exact under any rounding mode: the result is the
// f16 plane's element bit for bit.

#define KQV_FLAG_CAUSAL 1u
#define KQV_FLAG_ALIBI  2u

float kqv_neg_inf() {
    return uintBitsToFloat(0xFF800000u);
}

// -|d| as ggml_fp32_to_fp16(float(-|d|)) produces it (round to nearest even at 11 significant bits,
// overflow to -inf), returned as f32
float kqv_alibi_value(const int d) {
    uint n = uint(abs(d));
    if (n > 2048u) {
        const uint sh   = uint(findMSB(n)) - 10u;
        uint       q    = n >> sh;
        const uint rem  = n & ((1u << sh) - 1u);
        const uint half_ = 1u << (sh - 1u);
        if (rem > half_ || (rem == half_ && (q & 1u) != 0u)) {
            q += 1u;
        }
        n = q << sh;
        if (n >= 65536u) {
            return kqv_neg_inf();
        }
    }
    // n == 0 (p0 == p1) must stay +0: the reference computes float(-abs(0)) = float(int 0) = +0,
    // while -float(0u) would be -0 (0x8000 in the f16 plane)
    return n == 0u ? 0.0 : -float(n);
}

// a copy of llama_hparams::is_masked_swa
bool kqv_swa_masked(const uint swa_type, const int n_swa, const int p0, const int p1) {
    if (swa_type == 1u) {
        return p1 - p0 >= n_swa;
    }
    if (swa_type == 2u) {
        return p0 < (p1 / n_swa) * n_swa;
    }
    if (swa_type == 3u) {
        const int half_n_swa = n_swa / 2;
        const int pos_diff   = p1 - p0;
        return pos_diff < -half_n_swa || pos_diff > half_n_swa;
    }
    return false;
}

// the mask value for a cell record (p0, sm, x0, y0) and a token record t. x0 / y0 are read only at
// a causal tie (p0 == t.x), so a caller may pass anything for them when p0 is no token's position.
float kqv_value_of(const int p0, const uint sm, const int x0, const int y0, const ivec4 t, const uint flags, const int n_swa) {
    if ((sm & uint(t.y)) == 0u) {
        return kqv_neg_inf();
    }
    const int p1 = t.x;
    if ((flags & KQV_FLAG_CAUSAL) != 0u) {
        if (p0 > p1) {
            return kqv_neg_inf();
        }
        if (p0 == p1 && (y0 > t.w || (y0 == t.w && x0 > t.z))) {
            return kqv_neg_inf();
        }
    }
    const uint swa_type = (flags >> 2) & 3u;
    if (swa_type != 0u && kqv_swa_masked(swa_type, n_swa, p0, p1)) {
        return kqv_neg_inf();
    }
    if ((flags & KQV_FLAG_ALIBI) != 0u) {
        return kqv_alibi_value(p0 - p1);
    }
    return 0.0;
}

// the mask value of cell j (of the stream whose planes start at cbase) for token record t, loading
// the cell planes lazily: seq bits, then pos, then x / y only at a causal tie
float kqv_value(const uint cbase, const uint n_kv, const uint j, const ivec4 t, const uint flags, const int n_swa) {
    const uint sm = uint(kqv_cell_load(cbase + n_kv + j));
    if ((sm & uint(t.y)) == 0u) {
        return kqv_neg_inf();
    }
    const int p0 = kqv_cell_load(cbase + j);
    int x0 = 0;
    int y0 = 0;
    if ((flags & KQV_FLAG_CAUSAL) != 0u && p0 == t.x) {
        x0 = kqv_cell_load(cbase + 2u * n_kv + j);
        y0 = kqv_cell_load(cbase + 3u * n_kv + j);
    }
    return kqv_value_of(p0, sm, x0, y0, t, flags, n_swa);
}
