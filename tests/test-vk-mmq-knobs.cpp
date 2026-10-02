// [TAG_MMQ_INT8_KNOBS] CPU check of the GGML_VK_MMQ_INT8_MIN_N / GGML_VK_MMQ_INT8_OPS parsers and the per-call
// decision of ggml/src/ggml-vulkan/ggml-vulkan-mmq-knobs.h (the Vulkan backend reads both once at device init and
// prints them in the "mmq_cm1:" proof line; their effect on the GPU is checked with test-backend-ops on a coopmat1
// device). No Vulkan, no model.
//
//  1. MIN_N: absent / empty -> 0; decimal digits 0..UINT32_MAX accepted; sign, spaces, hex, trailing garbage and
//     overflow rejected with the output untouched (the backend then keeps the default and warns).
//  2. OPS: absent / empty / "all" -> both ops; "none" -> neither; "mul_mat", "mul_mat_id" and lists of them; empty
//     items, unknown names, upper case, spaces and "all"/"none" inside a list rejected, output untouched.
//  3. the proof-line names round-trip, and ggml_vk_mmq_int8_allowed() = (op in mask) && n >= min_n; the defaults
//     allow every call (= the behaviour before the knobs).

#include "../ggml/src/ggml-vulkan/ggml-vulkan-mmq-knobs.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static const uint32_t SENTINEL = 0xdeadbeefu;

static void check_min_n_ok(const char * s, uint32_t want) {
    uint32_t v = SENTINEL;
    const bool ok = ggml_vk_parse_mmq_int8_min_n(s, v);
    if (!ok || v != want) {
        fprintf(stderr, "FAIL min_n \"%s\": ok=%d v=%u, want ok=1 v=%u\n", s ? s : "(null)", ok, v, want);
        n_fail++;
    }
}

static void check_min_n_bad(const char * s) {
    uint32_t v = SENTINEL;
    const bool ok = ggml_vk_parse_mmq_int8_min_n(s, v);
    if (ok || v != SENTINEL) {
        fprintf(stderr, "FAIL min_n \"%s\": ok=%d v=%u, want rejected and untouched\n", s, ok, v);
        n_fail++;
    }
}

static void check_ops_ok(const char * s, uint32_t want) {
    uint32_t v = SENTINEL;
    const bool ok = ggml_vk_parse_mmq_int8_ops(s, v);
    if (!ok || v != want) {
        fprintf(stderr, "FAIL ops \"%s\": ok=%d v=%u, want ok=1 v=%u\n", s ? s : "(null)", ok, v, want);
        n_fail++;
    }
}

static void check_ops_bad(const char * s) {
    uint32_t v = SENTINEL;
    const bool ok = ggml_vk_parse_mmq_int8_ops(s, v);
    if (ok || v != SENTINEL) {
        fprintf(stderr, "FAIL ops \"%s\": ok=%d v=%u, want rejected and untouched\n", s, ok, v);
        n_fail++;
    }
}

int main() {
    const uint32_t MM  = GGML_VK_MMQ_INT8_OP_MUL_MAT;
    const uint32_t MID = GGML_VK_MMQ_INT8_OP_MUL_MAT_ID;
    const uint32_t ALL = GGML_VK_MMQ_INT8_OPS_ALL;

    // 1. MIN_N
    check_min_n_ok(nullptr, 0);
    check_min_n_ok("", 0);
    check_min_n_ok("0", 0);
    check_min_n_ok("9", 9);
    check_min_n_ok("32", 32);
    check_min_n_ok("064", 64);
    check_min_n_ok("4294967295", UINT32_MAX);
    for (const char * s : { "-1", "+1", " 1", "1 ", "1x", "0x10", "abc", "1.5", "4294967296", "99999999999999999999", "1e3", "\t2" }) {
        check_min_n_bad(s);
    }

    // 2. OPS
    check_ops_ok(nullptr, ALL);
    check_ops_ok("", ALL);
    check_ops_ok("all", ALL);
    check_ops_ok("none", 0);
    check_ops_ok("mul_mat", MM);
    check_ops_ok("mul_mat_id", MID);
    check_ops_ok("mul_mat,mul_mat_id", ALL);
    check_ops_ok("mul_mat_id,mul_mat", ALL);
    check_ops_ok("mul_mat_id,mul_mat_id", MID);
    for (const char * s : { "ALL", "None", "MUL_MAT", "mul_mat ", " mul_mat", "mul_mat, mul_mat_id", "mul_mat,",
                            ",mul_mat", "mul_mat,,mul_mat_id", "mul_mat,all", "none,mul_mat", "all,none", "mul",
                            "mul_mat_i", "mul_mat_idx", "mul_mat_id;mul_mat", "0", "1", "moe" }) {
        check_ops_bad(s);
    }

    // 3. names and the per-call decision
    CHECK(strcmp(ggml_vk_mmq_int8_ops_name(ALL), "all") == 0);
    CHECK(strcmp(ggml_vk_mmq_int8_ops_name(0), "none") == 0);
    CHECK(strcmp(ggml_vk_mmq_int8_ops_name(MM), "mul_mat") == 0);
    CHECK(strcmp(ggml_vk_mmq_int8_ops_name(MID), "mul_mat_id") == 0);
    for (const char * s : { "all", "none", "mul_mat", "mul_mat_id" }) {
        uint32_t v = SENTINEL;
        CHECK(ggml_vk_parse_mmq_int8_ops(s, v));
        CHECK(strcmp(ggml_vk_mmq_int8_ops_name(v), s) == 0);
    }

    // defaults (0, all): every call allowed, whatever N
    for (uint64_t n : { (uint64_t) 0, (uint64_t) 1, (uint64_t) 9, (uint64_t) 30, (uint64_t) 512, (uint64_t) 1 << 40 }) {
        CHECK(ggml_vk_mmq_int8_allowed(ALL, 0, MM, n));
        CHECK(ggml_vk_mmq_int8_allowed(ALL, 0, MID, n));
    }
    // MIN_N = 64: n >= 64 only
    CHECK(!ggml_vk_mmq_int8_allowed(ALL, 64, MM, 9));
    CHECK(!ggml_vk_mmq_int8_allowed(ALL, 64, MID, 30));
    CHECK(!ggml_vk_mmq_int8_allowed(ALL, 64, MM, 63));
    CHECK(ggml_vk_mmq_int8_allowed(ALL, 64, MM, 64));
    CHECK(ggml_vk_mmq_int8_allowed(ALL, 64, MID, 512));
    // OPS = mul_mat_id: dense matmuls excluded at any N
    CHECK(!ggml_vk_mmq_int8_allowed(MID, 0, MM, 512));
    CHECK(ggml_vk_mmq_int8_allowed(MID, 0, MID, 9));
    // OPS = mul_mat: MoE excluded
    CHECK(ggml_vk_mmq_int8_allowed(MM, 0, MM, 9));
    CHECK(!ggml_vk_mmq_int8_allowed(MM, 0, MID, 512));
    // OPS = none: nothing
    CHECK(!ggml_vk_mmq_int8_allowed(0, 0, MM, 512));
    CHECK(!ggml_vk_mmq_int8_allowed(0, 0, MID, 512));
    // both: MIN_N 32 + mul_mat_id
    CHECK(!ggml_vk_mmq_int8_allowed(MID, 32, MID, 31));
    CHECK(ggml_vk_mmq_int8_allowed(MID, 32, MID, 32));
    CHECK(!ggml_vk_mmq_int8_allowed(MID, 32, MM, 32));

    if (n_fail == 0) {
        printf("test-vk-mmq-knobs: OK\n");
        return 0;
    }
    fprintf(stderr, "test-vk-mmq-knobs: %d failure(s)\n", n_fail);
    return 1;
}
