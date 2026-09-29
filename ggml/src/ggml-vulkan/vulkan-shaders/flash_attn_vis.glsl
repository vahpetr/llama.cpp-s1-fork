// [TAG_KQ_VIS] flash attention over the visibility metadata instead of a mask plane.
// Included by flash_attn.comp (scalar) and flash_attn_cm1.comp after flash_attn_base.glsl.
//
// With VIS set, binding 3 holds the cell planes [n_kv, 4, n_stream] (I32) instead of the f16 mask,
// binding 7 the token records [4, n_tps, n_stream], and every load of a mask element becomes
// fa_vis_value() of the same (row, column): the same f16 value, so everything downstream (the
// all -inf skip, slope * mask, ZERO_DEAD_V, GQA_TOKENS rows, split_k) is unchanged. The host keeps
// the mask push constants as the plane would have them: nem1 = n_tps, nem2 = 1, nem3 = n_stream.
//
// Flags bits: 512 VIS, 1024 causal, 2048 alibi, 4096/8192 swa type; n_swa is spec constant 16,
// n_sink ([TAG_MTP_DRAFT_WINDOW], standard window only) spec constant 17.

layout (constant_id = 16) const uint32_t VisNSwa = 0;
layout (constant_id = 17) const uint32_t VisNSink = 0;

const bool VIS           = (Flags & 512) != 0;
const uint32_t VIS_FLAGS = (((Flags & 1024) != 0) ? 1u : 0u) |
                           (((Flags & 2048) != 0) ? 2u : 0u) |
                           (((Flags >> 12) & 3u) << 2) |
                           16u; // KQV_FLAG_F16: FA reads the values of an f16 plane

layout (binding = 3) readonly buffer VC {int data_vc[];};
layout (binding = 7) readonly buffer VT {ivec4 data_vt[];};

int kqv_cell_load(uint idx) {
    return data_vc[idx];
}

#include "kq_vis.glsl"

// [TAG_KQ_VIS] staging (spec §2.4a/§2.4c): the workgroup's Br token records once per workgroup and the
// Bc cell records of a KV tile once per tile, in shared memory, so the load sites never go to global
// memory. Two forms:
//  * rows share tokens (the plain GQA fold: every row is the one token gqa_iq1; GQA_TOKENS: rows are
//    (token, head) pairs, at most VIS_SLOTS distinct tokens per Br tile): the staging evaluates the
//    predicate ONCE per (cell, token) into vis_sh_val and a load site is one shared read
//    (vis_sh_val[slot of the row][column]) - the q heads folded onto a kv head share it;
//  * rows are tokens (no fold, prefill): the staging stores the cell records, the load site
//    evaluates the predicate from shared memory per element (fa_vis_value_b).
// flash_attn_cm1.comp double-buffers the staging (buffer j & 1) and stages tile j + 1 at the top of
// iteration j, so the barriers the tile already has order it (no barrier of its own); the scalar
// shader stages single-buffered with a barrier (fa_vis_stage_tile). The arrays shrink to one element
// when VIS is off.
const uint32_t VIS_SLOTS  = Br >= 16 ? Br / 16 : 1;   // distinct tokens per Br tile when rows share tokens
const uint32_t VIS_SH_BC  = VIS ? 2 * Bc : 1;          // two buffers of Bc cell records
const uint32_t VIS_SH_BR  = VIS ? Br : 1;
const uint32_t VIS_SH_VAL = VIS ? 2 * VIS_SLOTS * Bc : 1;
const uint32_t VIS_SH_SL  = VIS ? VIS_SLOTS : 1;
shared int   vis_sh_pos[VIS_SH_BC];
shared uint  vis_sh_seq[VIS_SH_BC];
shared int   vis_sh_x[VIS_SH_BC];
shared int   vis_sh_y[VIS_SH_BC];
shared ivec4 vis_sh_tok[VIS_SH_BR];      // the token record of each row of the Br tile
shared uint  vis_sh_rowslot[VIS_SH_BR];  // rows share tokens: the slot of each row
shared ivec4 vis_sh_slottok[VIS_SH_SL];  // rows share tokens: the token record of each slot
shared float vis_sh_val[VIS_SH_VAL];     // rows share tokens: [buffer][slot][column] mask values

// token record of mask row 0 of this workgroup's stream, the stream's cell planes, the range of the
// workgroup's token positions (a causal tie p0 == p1 is only possible inside it), and the form
bool vis_fold;
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
    vis_fold  = GQA_TOKENS || p.gqa_ratio > 1;

    for (uint r = gl_LocalInvocationIndex; r < Br; r += gl_WorkGroupSize.x) {
        vis_sh_tok[r] = data_vt[fa_vis_tok_index(i * Br + r)];
        // the rows of a Br tile span at most VIS_SLOTS consecutive tokens: FA_GQA_ROWS >= 16 rows
        // per token and i * Br is a multiple of 16
        vis_sh_rowslot[r] = GQA_TOKENS ? fa_mask_row(i * Br + r) - fa_mask_row(i * Br) : 0u;
    }
    for (uint sl = gl_LocalInvocationIndex; sl < VIS_SLOTS; sl += gl_WorkGroupSize.x) {
        const uint tok = GQA_TOKENS ? vis_tok0 + min(fa_mask_row(i * Br) + sl, p.nem1 - 1u) : vis_tok0;
        vis_sh_slottok[sl] = data_vt[tok];
    }
    barrier();

    vis_tmin = 2147483647;
    vis_tmax = int(0x80000000u);
    for (uint r = 0; r < Br; ++r) {
        vis_tmin = min(vis_tmin, vis_sh_tok[r].x);
        vis_tmax = max(vis_tmax, vis_sh_tok[r].x);
    }
}

// load the cell record of column col (x / y only where a causal tie can read them: pos inside the
// workgroup's token range; pos only for cells with a sequence bit)
void fa_vis_load_cell(const uint col, out int p0, out uint sm, out int x0, out int y0) {
    p0 = -1;
    sm = 0u;
    x0 = 0;
    y0 = 0;
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
}

// stage KV tile jt into buffer b, no barrier (the caller's barriers order it)
void fa_vis_stage(const uint jt, const uint b) {
    for (uint c = gl_LocalInvocationIndex; c < Bc; c += gl_WorkGroupSize.x) {
        int p0; uint sm; int x0; int y0;
        fa_vis_load_cell(jt * Bc + c, p0, sm, x0, y0);
        if (vis_fold) {
            [[unroll]] for (uint sl = 0; sl < VIS_SLOTS; ++sl) {
                vis_sh_val[(b * VIS_SLOTS + sl) * Bc + c] =
                    kqv_value_of(p0, sm, x0, y0, vis_sh_slottok[sl], VIS_FLAGS, int(VisNSwa), int(VisNSink));
            }
        } else {
            vis_sh_pos[b * Bc + c] = p0;
            vis_sh_seq[b * Bc + c] = sm;
            vis_sh_x  [b * Bc + c] = x0;
            vis_sh_y  [b * Bc + c] = y0;
        }
    }
}

// the scalar shader: stage KV tile j into buffer 0 and wait (workgroup-uniform call)
void fa_vis_stage_tile(const uint j) {
    fa_vis_stage(j, 0u);
    barrier();
}

// the mask element the plane path loads at data_m[m_offset + fa_mask_row(R) * m_stride + col], for the
// tile-local row r_local = R - i * Br and tile-local column c_local = col - j * Bc of the tile staged
// in buffer b
float fa_vis_value_b(const uint b, const uint r_local, const uint c_local) {
    if (vis_fold) {
        // the plain GQA fold has one token (slot 0): no per-row slot read
        return vis_sh_val[(b * VIS_SLOTS + (GQA_TOKENS ? vis_sh_rowslot[r_local] : 0u)) * Bc + c_local];
    }
    const uint e = b * Bc + c_local;
    return kqv_value_of(vis_sh_pos[e], vis_sh_seq[e], vis_sh_x[e], vis_sh_y[e],
                        vis_sh_tok[r_local], VIS_FLAGS, int(VisNSwa), int(VisNSink));
}

float fa_vis_value_sh(const uint r_local, const uint c_local) {
    return fa_vis_value_b(0u, r_local, c_local);
}
