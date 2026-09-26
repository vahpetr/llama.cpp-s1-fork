// [TAG_KQ_VIS] flash attention over the visibility metadata instead of a mask plane.
// Included by flash_attn.comp (scalar) and flash_attn_cm1.comp after flash_attn_base.glsl.
//
// With VIS set, binding 3 holds the cell planes [n_kv, 4, n_stream] (I32) instead of the f16 mask,
// binding 7 the token records [4, n_tps, n_stream], and every load of a mask element becomes
// fa_vis_value() of the same (row, column): the same f16 value, so everything downstream (the
// all -inf skip, slope * mask, ZERO_DEAD_V, GQA_TOKENS rows, split_k) is unchanged. The host keeps
// the mask push constants as the plane would have them: nem1 = n_tps, nem2 = 1, nem3 = n_stream.
//
// Flags bits: 512 VIS, 1024 causal, 2048 alibi, 4096/8192 swa type; n_swa is spec constant 16.

layout (constant_id = 16) const uint32_t VisNSwa = 0;

const bool VIS           = (Flags & 512) != 0;
const uint32_t VIS_FLAGS = (((Flags & 1024) != 0) ? 1u : 0u) |
                           (((Flags & 2048) != 0) ? 2u : 0u) |
                           (((Flags >> 12) & 3u) << 2);

layout (binding = 3) readonly buffer VC {int data_vc[];};
layout (binding = 7) readonly buffer VT {ivec4 data_vt[];};

int kqv_cell_load(uint idx) {
    return data_vc[idx];
}

#include "kq_vis.glsl"

// token record of mask row 0 of this workgroup's stream, and the stream's cell planes
uint vis_tok0, vis_cbase;

void fa_vis_init() {
    const uint s = iq3 % p.nem3;
    vis_tok0  = gqa_iq1 + s * p.nem1;
    vis_cbase = s * 4u * KV;
}

// the mask element the plane path loads at data_m[m_offset + fa_mask_row(R) * m_stride + col]
float fa_vis_value(const uint R, const uint col) {
    const uint tok = vis_tok0 + ((!GQA_TOKENS && p.gqa_ratio > 1) ? 0u : fa_mask_row(R));
    return kqv_value(vis_cbase, KV, col, data_vt[tok], VIS_FLAGS, int(VisNSwa));
}
