#pragma once

// [TAG_MMQ_INT8_KNOBS] Diagnostic restrictions of the int8 coopmat1 MMQ path ([TAG_MMQ_CM1], contrib 0137), for an
// A/B that separates where the int8 path changes the numerics (batched verify vs prompt processing, dense vs MoE).
// Header-only and free of Vulkan types, so tests/test-vk-mmq-knobs.cpp checks the parsers on the CPU.
//
//   GGML_VK_MMQ_INT8_MIN_N=<N>   take the int8 path only when the matmul's N >= N: MUL_MAT N = src1->ne[1] (the
//                                columns of one GEMM), MUL_MAT_ID N = ids->ne[1] (the batch's tokens - the value the
//                                mat-vec/mat-mat split and the int8 pipeline size choice use; each expert sees at
//                                most that many columns). Decimal digits only, 0..4294967295. Default 0 = no limit.
//                                Matmuls with N <= 8 never reach the int8 path (they run the mat-vec shaders), so any
//                                N <= 9 equals the default.
//   GGML_VK_MMQ_INT8_OPS=<list>  restrict the int8 path to these ops: "all" (default), "none", or a comma-separated
//                                list of "mul_mat" and "mul_mat_id" (exact lower case, no spaces).
//
// Absent or empty = the default (as [TAG_ENV_BOOL]); an invalid value keeps the default and is warned about. Both are
// read once at device init and printed in the mmq_cm1 proof line. A matmul they exclude takes quantize_y = false, i.e.
// the pre-port f16/f32 dequant coopmat1 path - for that call exactly what GGML_VK_DISABLE_MMQ_CM1=1 does for all of
// them (no q8_1 pipeline exists on a coopmat1 device besides the int8 coopmat1 ones). They only act when the int8
// coopmat1 path is on (device->mmq_cm1); the integer-dot (dp4a) MMQ of other devices is untouched.

#include <cstdint>
#include <cstring>

enum : uint32_t {
    GGML_VK_MMQ_INT8_OP_MUL_MAT    = 1u << 0,
    GGML_VK_MMQ_INT8_OP_MUL_MAT_ID = 1u << 1,
    GGML_VK_MMQ_INT8_OPS_ALL       = GGML_VK_MMQ_INT8_OP_MUL_MAT | GGML_VK_MMQ_INT8_OP_MUL_MAT_ID,
};

// nullptr / "" -> true, out = 0 (the default). A valid value -> true, out = the value. Invalid -> false, out untouched.
static inline bool ggml_vk_parse_mmq_int8_min_n(const char * s, uint32_t & out) {
    if (s == nullptr || *s == '\0') {
        out = 0;
        return true;
    }
    uint64_t v = 0;
    for (const char * p = s; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            return false; // sign, space, hex prefix, trailing garbage
        }
        v = v * 10 + (uint64_t) (*p - '0');
        if (v > UINT32_MAX) {
            return false;
        }
    }
    out = (uint32_t) v;
    return true;
}

// nullptr / "" -> true, out = GGML_VK_MMQ_INT8_OPS_ALL (the default). "all" / "none" alone, or a comma-separated list
// of "mul_mat" / "mul_mat_id" -> true, out = the mask. Anything else (empty item, unknown name, upper case, spaces,
// "all"/"none" inside a list) -> false, out untouched.
static inline bool ggml_vk_parse_mmq_int8_ops(const char * s, uint32_t & out) {
    if (s == nullptr || *s == '\0') {
        out = GGML_VK_MMQ_INT8_OPS_ALL;
        return true;
    }
    if (strcmp(s, "all") == 0) {
        out = GGML_VK_MMQ_INT8_OPS_ALL;
        return true;
    }
    if (strcmp(s, "none") == 0) {
        out = 0;
        return true;
    }
    uint32_t mask = 0;
    const char * p = s;
    for (;;) {
        const char * comma = strchr(p, ',');
        const size_t len = comma ? (size_t) (comma - p) : strlen(p);
        if (len == 7 && strncmp(p, "mul_mat", 7) == 0) {
            mask |= GGML_VK_MMQ_INT8_OP_MUL_MAT;
        } else if (len == 10 && strncmp(p, "mul_mat_id", 10) == 0) {
            mask |= GGML_VK_MMQ_INT8_OP_MUL_MAT_ID;
        } else {
            return false;
        }
        if (comma == nullptr) {
            break;
        }
        p = comma + 1;
    }
    out = mask;
    return true;
}

static inline const char * ggml_vk_mmq_int8_ops_name(uint32_t mask) {
    switch (mask & GGML_VK_MMQ_INT8_OPS_ALL) {
        case GGML_VK_MMQ_INT8_OPS_ALL:       return "all";
        case GGML_VK_MMQ_INT8_OP_MUL_MAT:    return "mul_mat";
        case GGML_VK_MMQ_INT8_OP_MUL_MAT_ID: return "mul_mat_id";
        default:                             return "none";
    }
}

// The per-call decision: may this matmul (op = GGML_VK_MMQ_INT8_OP_*, N as above) take the int8 path? It still needs
// a q8_1 pipeline for its weight type and a contiguous f32 src1, as before.
static inline bool ggml_vk_mmq_int8_allowed(uint32_t ops_mask, uint32_t min_n, uint32_t op, uint64_t n) {
    return (ops_mask & op) != 0 && n >= min_n;
}
