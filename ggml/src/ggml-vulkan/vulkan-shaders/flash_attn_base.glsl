
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id =  0) const uint32_t WorkGroupSize = 128;
layout (constant_id =  1) const uint32_t Br = 1;
layout (constant_id =  2) const uint32_t Bc = 32;
layout (constant_id =  3) const uint32_t HSK = 32;
layout (constant_id =  4) const uint32_t HSV = 32;
layout (constant_id =  5) const uint32_t Clamp = 0;
layout (constant_id =  6) const uint32_t D_split = 16;
layout (constant_id =  7) const uint32_t row_split = 1;
layout (constant_id =  8) const uint32_t SubGroupSize = 32;
layout (constant_id =  9) const uint32_t SHMEM_STAGING = 0;
layout (constant_id = 10) const uint32_t Flags = 0;
layout (constant_id = 11) const uint32_t LIMIT_OCCUPANCY_SHMEM = 0;
// ggml_type enumerant for K/V
layout (constant_id = 12) const uint32_t FaTypeK = 0;
layout (constant_id = 13) const uint32_t FaTypeV = 0;
// sizeof(decode buffer): quants -> ggml block size; F32 -> 16 (decodeBufF32 vec4).
layout (constant_id = 14) const uint32_t FaBlockBytesK = 2;
layout (constant_id = 15) const uint32_t FaBlockBytesV = 2;

const bool USE_MASK_OPT    = (Flags & 1) != 0;
const bool MASK_ENABLE     = (Flags & 2) != 0;
const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;
// Replace V by +0 in KV columns that this Br tile's mask suppresses for every row.
const bool ZERO_DEAD_V     = (Flags & 16) != 0;
// Derive ZERO_DEAD_V's dead-column predicate by re-reading Psh from shared memory (the
// original implementation) instead of from the P values that are still in registers.
// Same predicate either way; this is the A/B knob (GGML_VK_FA_DEADV_LDS).
const bool DEADV_LDS       = (Flags & 32) != 0;
// [TAG_FA_GQA_TOKENS] Fold the query tokens into the row dimension on top of the GQA head
// fold: a Br tile holds rows of the (token, head) product space, decoded as
// token = r / gqa_ratio, head = iq2 + r % gqa_ratio, instead of one token's gqa_ratio heads.
// One workgroup then covers ceil(gqa_ratio * n_tokens / Br) tiles' worth of KV passes instead
// of one pass per token, which is the point: at decode depth this kernel is bound by how many
// times it streams its KV head.  Rows still index gl_WorkGroupID.x the way the non-gqa path
// does (i = row tile), and the mask gets a real per-row stride.
//
// The packing is dense: a Br tile is a contiguous slice of the row space and a token's rows
// may straddle two tiles.  The alternative - round each tile down to a whole number of tokens,
// floor(Br / gqa_ratio) * gqa_ratio rows - keeps every tile's rows inside one token, but at
// the shape this exists for (gqa_ratio 12, Br 32) that is 24 rows per tile, so it needs the
// same ceil(6 / 2) = 3 tiles as dense packing while leaving a quarter of every tile's QK and
// P*V work on padding rows.  It buys nothing in return: no reduction in this kernel is over
// rows, so a tile that mixes tokens costs nothing, and the mask is read per row either way
// (the four rows of one mask f16vec4 already come from four independent scalar loads).  What
// the alternative WOULD buy is a tile-uniform mask row, i.e. one mask load broadcast to all
// rows - but at decode the mask block is 64 x Br f16 served out of L1, not a DRAM cost, so it
// is not worth a quarter of the tile.
// Bit 64 is deliberately skipped: the prod-v7-fa line uses it for CM1_SF_PAD (the coopmat1
// QK^T scratch stride pad, [TAG_FA_CM1_SF_PAD]).  The two series are independent and either
// could land first, so they must not alias - a merge that gave one branch's dispatch the
// other's shader meaning would pad sfshstride under a GQA_TOKENS dispatch (mismatching the
// host-side shared-memory estimate) and mis-decode the rows.  Keep new FA flags above 64.
const bool GQA_TOKENS      = (Flags & 128) != 0;

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
const uint32_t HSK_pad = (HSK + 15) & ~15;
const uint32_t HSV_pad = (HSV + 15) & ~15;

const bool KV_bounds_check = Clamp != 0;

layout (push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

#define SINK_ENABLE_BIT (1<<24)
#define N_LOG2_MASK 0xFFFF

layout (binding = 4) readonly buffer S {float data_s[];};

layout (binding = 5) writeonly buffer O {D_TYPE data_o[];};
layout (binding = 5) writeonly buffer OV4 {D_TYPEV4 data_ov4[];};

layout (binding = 6) readonly buffer MO {uint32_t data_mask_opt[];};

#define MASK_OPT_ALL_NEG_INF 1
#define MASK_OPT_ALL_ZERO 2

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1

#include "fa_types.glsl"

#if defined(BFLOAT16)
#define O_TYPE float
#define O_TYPEV4 vec4
#elif defined(FA_PV_ACC_TYPE)
// The P*V cooperative matmul accumulates in ACC_TYPE, i.e. f32 whenever the op
// asks for GGML_PREC_F32, instead of always in FLOAT_TYPE (f16 for f16 K/V).
//
// Mask-suppressed KV columns have P == +0 exactly, so they are mathematically
// inert - but they are still fed to coopMatMulAdd, where 0*V evaluates to -0
// for V < 0.  RDNA3's cooperative matmul is not invariant to the sign of a zero
// product: a -0 term perturbs the result by one ULP of the accumulator whenever
// its f16 k-pair partner is a live column.  A live query row's output therefore
// depends on the V bytes of KV cells it has masked away, which breaks
// reproducibility across batch shapes (the same row computed against a KV cache
// with one more written cell).  Accumulating in f32 does not remove the
// dependency, but it shrinks it by 2^13 - from ~2e-4 (one f16 ULP, ~0.15 on the
// logits after 32 layers) to ~3e-8 - which is below the noise of the rest of
// the graph.
#define O_TYPE ACC_TYPE
#define O_TYPEV4 ACC_TYPEV4
#else
#define O_TYPE FLOAT_TYPE
#define O_TYPEV4 FLOAT_TYPEV4
#endif

// These can't be `const` globals because GLSL forbids function calls in global
// const initializers, even when the spec constants would let the driver fold
// them. Macros expand at the use site and fold after specialization.
#define BLOCK_SIZE_K fa_block_elems(FaTypeK)
#define BLOCK_SIZE_V fa_block_elems(FaTypeV)
// F16 reads f16 elements directly from the binding; everything else routes
// through dequantize4 / the MMQ helpers to unpack from the packed block layout.
#define USE_DECODE_K (FaTypeK != FA_TYPE_F16)
#define USE_DECODE_V (FaTypeV != FA_TYPE_F16)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Store column zero. This is used to save per-row m and L values for split_k.
ACC_TYPE perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = D_TYPE(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
ACC_TYPE perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const ACC_TYPE base = ACC_TYPE(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return ACC_TYPE(pow(base, ACC_TYPE(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
ACC_TYPE perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return ACC_TYPE(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    // gl_WorkGroupID.x is a query token when a workgroup covers one token's gqa_ratio heads
    // (the plain GQA fold), and a Br-row tile of the (token, head) row space under GQA_TOKENS,
    // which is the same layout the non-gqa path uses.  Written out at each site rather than
    // hoisted into a local so that, with GQA_TOKENS specialized to false, each test is the
    // "p.gqa_ratio > 1" the compiler saw before.
    if (p.k_num > 1) {
        if (!GQA_TOKENS && p.gqa_ratio > 1) {
            i = 0;
            // batch and split_k share gl_WorkGroupID.x
            gqa_iq1 = gl_WorkGroupID.x / p.k_num;
            split_k_index = gl_WorkGroupID.x % p.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % p.k_num;
            i = gl_WorkGroupID.x / p.k_num;
        }
    } else if (!GQA_TOKENS && p.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    Tr = CEIL_DIV(N, Br);

    start_j = split_k_index * p.split_kv / Bc;
    end_j = CEIL_DIV(min(KV, (split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    // broadcast factors
    rk2 = p.neq2/p.nek2;
    rk3 = p.neq3/p.nek3;

    rv2 = p.neq2/p.nev2;
    rv3 = p.neq3/p.nev3;

    // k indices
    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    // v indices
    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;
    // When one workgroup covers one token's heads, all rows use the same mask (stride 0) and
    // the token is folded into m_offset instead.  "p.gqa_ratio >> 16" is just a roundabout way
    // of writing zero that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    // With GQA_TOKENS the rows are (token, head) pairs, so the mask gets its real row stride
    // back and fa_mask_row() below turns a row into its token.
    m_stride = (!GQA_TOKENS && p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;
}

// The mask row of row r of the N-row query space.  Rows past the end of the batch (the padding
// of the last Br tile) are clamped so the read stays inside the mask; their result is discarded.
uint32_t fa_mask_row(const in uint32_t r)
{
    return GQA_TOKENS ? min(r / p.gqa_ratio, p.nem1 - 1) : r;
}

// Q row r as a vec4 offset relative to q_offset / 4.  Without GQA_TOKENS this is exactly the
// expression the callers used to inline; with it, the row carries its own token and head.
uint32_t fa_q_row_offset4(const in uint32_t r)
{
    return GQA_TOKENS ? ((r / p.gqa_ratio) * p.nb01 + (r % p.gqa_ratio) * q_stride) / 4
                      : r * q_stride / 4;
}

// The token and the head-within-group of row r.  Only meaningful when GQA_TOKENS is set.
#define FA_ROW_TOKEN(r) ((r) / p.gqa_ratio)
#define FA_ROW_HEAD(r)  ((r) % p.gqa_ratio)

// Bias applied to softmax to stay in fp16 range.
// Based on ggml-cuda issue https://github.com/ggml-org/llama.cpp/issues/18606
const float FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f;

// Store the output when doing grouped query attention.
// Rows index by Q's dimension 2, and the first N rows are valid.
void gqaStore(const in uint32_t r, const in uint32_t c, const in O_TYPEV4 elems, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    uint32_t offset = (iq2 + r) * HSV / 4 + c;
    data_ov4[o_offset + offset] = D_TYPEV4(elems);
}
