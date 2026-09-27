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

// [TAG_KQ_VIS] staging (spec §2.4a): the workgroup's Br token records once per workgroup, and the
// Bc cell records of a KV tile once per tile, in shared memory; the load sites then evaluate the
// predicate from shared memory (fa_vis_value_sh). The arrays shrink to one element when VIS is off.
const uint32_t VIS_SH_BC = VIS ? Bc : 1;
const uint32_t VIS_SH_BR = VIS ? Br : 1;
shared int   vis_sh_pos[VIS_SH_BC];
shared uint  vis_sh_seq[VIS_SH_BC];
shared int   vis_sh_x[VIS_SH_BC];
shared int   vis_sh_y[VIS_SH_BC];
shared ivec4 vis_sh_tok[VIS_SH_BR];

// token record of mask row 0 of this workgroup's stream, the stream's cell planes, and the range of
// the workgroup's token positions (a causal tie p0 == p1 is only possible inside it)
uint vis_tok0, vis_cbase;
int  vis_tmin, vis_tmax;

// the token record index of query row R (the row the plane path reads at fa_mask_row(R) * m_stride);
// rows past the batch are clamped - their mask values are never used
uint fa_vis_tok_index(const uint R) {
    return vis_tok0 + ((!GQA_TOKENS && p.gqa_ratio > 1) ? 0u : min(fa_mask_row(R), p.nem1 - 1u));
}

// after init_indices(); every invocation of the workgroup must call it (it has a barrier)
void fa_vis_init() {
    const uint s = iq3 % p.nem3;
    vis_tok0  = gqa_iq1 + s * p.nem1;
    vis_cbase = s * 4u * KV;

    for (uint r = gl_LocalInvocationIndex; r < Br; r += gl_WorkGroupSize.x) {
        vis_sh_tok[r] = data_vt[fa_vis_tok_index(i * Br + r)];
    }
    barrier();

    vis_tmin = 2147483647;
    vis_tmax = int(0x80000000u);
    for (uint r = 0; r < Br; ++r) {
        vis_tmin = min(vis_tmin, vis_sh_tok[r].x);
        vis_tmax = max(vis_tmax, vis_sh_tok[r].x);
    }
}

// stage the cell records of KV tile j; workgroup-uniform call (it ends with a barrier). x / y are
// loaded only for cells whose position some token of this workgroup has (the only cells where a
// causal tie can read them); pos is loaded only for cells with a sequence bit.
void fa_vis_stage_tile(const uint j) {
    for (uint c = gl_LocalInvocationIndex; c < Bc; c += gl_WorkGroupSize.x) {
        const uint col = j * Bc + c;
        int  p0 = -1;
        uint sm = 0u;
        int  x0 = 0;
        int  y0 = 0;
        if (col < KV) {
            sm = uint(data_vc[vis_cbase + KV + col]);
            if (sm != 0u) {
                p0 = data_vc[vis_cbase + col];
                if (p0 >= vis_tmin && p0 <= vis_tmax) {
                    x0 = data_vc[vis_cbase + 2u * KV + col];
                    y0 = data_vc[vis_cbase + 3u * KV + col];
                }
            }
        }
        vis_sh_pos[c] = p0;
        vis_sh_seq[c] = sm;
        vis_sh_x[c]   = x0;
        vis_sh_y[c]   = y0;
    }
    barrier();
}

// the mask element the plane path loads at data_m[m_offset + fa_mask_row(R) * m_stride + col], for the
// tile-local row r_local = R - i * Br and tile-local column c_local = col - j * Bc of the staged tile
float fa_vis_value_sh(const uint r_local, const uint c_local) {
    return kqv_value_of(vis_sh_pos[c_local], vis_sh_seq[c_local], vis_sh_x[c_local], vis_sh_y[c_local],
                        vis_sh_tok[r_local], VIS_FLAGS, int(VisNSwa));
}
