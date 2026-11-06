// [TAG_VK_READ_CHUNK] CPU check of the chunked staging plan of ggml/src/ggml-vulkan/ggml-vulkan-staging-plan.h, which
// ggml_vk_buffer_read_2d uses for a synchronous staged read larger than GGML_VK_SYNC_READ_CHUNK_MIB. No Vulkan.
//
//  1. byte identity: for contiguous, row-padded and destination-padded 2D reads, rows smaller than, equal to and larger
//     than the slot, the plan executed in the backend's ring order (copy chunk i+1 into slot (i+1)%2 BEFORE the memcpy of
//     chunk i out of slot i%2) yields exactly the bytes of the direct 2D copy, and destination bytes outside the rows
//     stay untouched;
//  2. bounds: every chunk uses <= cap staging bytes, every piece lies inside its chunk, pieces cover each row once,
//     a contiguous read is one piece per chunk and ceil(total / cap) chunks;
//  3. mutant: the same ring order with a one-slot staging (no ring) must corrupt the output - the check above would
//     catch a ring that lets the in-flight copy overwrite the slot being read;
//  4. the GGML_VK_SYNC_READ_CHUNK_MIB parser (digits via ggml_vk_parse_mmq_int8_min_n);
//  5. MiB -> bytes in 64-bit, clamped for a 32-bit size_t (4096 MiB must not wrap to 0).

#include "../ggml/src/ggml-vulkan/ggml-vulkan-staging-plan.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

// executes a plan as ggml_vk_buffer_read_2d_chunked does: submit(0); for i: wait; submit(i+1); memcpy(i).
// "submit" is executed eagerly here, i.e. the copy of chunk i+1 has landed before chunk i is read - the worst case.
static void run_ring(const std::vector<uint8_t> & src, std::vector<uint8_t> & dst, size_t cap, int n_slots,
                     const std::vector<ggml_vk_staging_piece> & pieces, const std::vector<ggml_vk_staging_chunk> & chunks) {
    std::vector<uint8_t> staging(cap * n_slots, 0xEE);
    auto copy = [&](size_t ci) {
        const size_t base = (ci % n_slots) * cap;
        for (size_t k = 0; k < chunks[ci].count; k++) {
            const auto & p = pieces[chunks[ci].first + k];
            memcpy(staging.data() + base + p.stg, src.data() + p.src, p.n);
        }
    };
    copy(0);
    for (size_t ci = 0; ci < chunks.size(); ci++) {
        if (ci + 1 < chunks.size()) {
            copy(ci + 1);
        }
        const size_t base = (ci % n_slots) * cap;
        for (size_t k = 0; k < chunks[ci].count; k++) {
            const auto & p = pieces[chunks[ci].first + k];
            memcpy(dst.data() + p.dst, staging.data() + base + p.stg, p.n);
        }
    }
}

static bool check_case(size_t offset, size_t spitch, size_t dpitch, size_t width, size_t height, size_t cap, std::mt19937 & rng) {
    const size_t src_size = offset + (height - 1) * spitch + width + 7;
    const size_t dst_size = (height - 1) * dpitch + width + 5;
    std::vector<uint8_t> src(src_size);
    for (auto & b : src) {
        b = (uint8_t) rng();
    }

    // the legacy route: whole read in one staging buffer = the direct 2D copy
    std::vector<uint8_t> want(dst_size, 0x5A);
    for (size_t r = 0; r < height; r++) {
        memcpy(want.data() + r * dpitch, src.data() + offset + r * spitch, width);
    }

    std::vector<ggml_vk_staging_piece> pieces;
    std::vector<ggml_vk_staging_chunk> chunks;
    ggml_vk_staging_plan_2d(offset, spitch, dpitch, width, height, cap, pieces, chunks);

    bool ok = !chunks.empty();
    size_t covered = 0;
    size_t next_first = 0;
    for (const auto & ch : chunks) {
        ok = ok && ch.first == next_first && ch.count > 0 && ch.bytes > 0 && ch.bytes <= cap;
        next_first = ch.first + ch.count;
        size_t used = 0;
        for (size_t k = 0; k < ch.count; k++) {
            const auto & p = pieces[ch.first + k];
            ok = ok && p.n > 0 && p.stg == used && p.stg + p.n <= cap;
            used += p.n;
            covered += p.n;
        }
        ok = ok && used == ch.bytes;
    }
    ok = ok && next_first == pieces.size() && covered == width * height;
    if (width == spitch && width == dpitch) {
        ok = ok && chunks.size() == (width * height + cap - 1) / cap && pieces.size() == chunks.size();
    }

    std::vector<uint8_t> got(dst_size, 0x5A);
    run_ring(src, got, cap, 2, pieces, chunks);
    ok = ok && got == want;

    if (!ok) {
        fprintf(stderr, "FAIL case offset=%zu spitch=%zu dpitch=%zu width=%zu height=%zu cap=%zu (chunks=%zu pieces=%zu)\n",
                offset, spitch, dpitch, width, height, cap, chunks.size(), pieces.size());
        n_fail++;
    }
    return ok;
}

int main() {
    std::mt19937 rng(42);
    int n_cases = 0;

    // 1 + 2: hand-picked shapes around the slot boundary
    const size_t cap = 64;
    const size_t widths[]  = { 1, 7, 31, 32, 63, 64, 65, 127, 128, 129, 200, 1000 };
    const size_t heights[] = { 1, 2, 3, 5, 17 };
    for (size_t w : widths) {
        for (size_t h : heights) {
            for (size_t off : { (size_t) 0, (size_t) 3, (size_t) 64 }) {
                check_case(off, w,      w,      w, h, cap, rng); // contiguous (the slot-save case)
                check_case(off, w + 9,  w,      w, h, cap, rng); // padded source rows
                check_case(off, w,      w + 4,  w, h, cap, rng); // padded destination rows
                check_case(off, w * 3,  w * 2,  w, h, cap, rng); // both strided
                n_cases += 4;
            }
        }
    }
    // random shapes and caps
    for (int i = 0; i < 2000; i++) {
        const size_t w   = 1 + rng() % 300;
        const size_t h   = 1 + rng() % 40;
        const size_t sp  = w + (rng() % 3 == 0 ? 0 : rng() % 50);
        const size_t dp  = w + (rng() % 3 == 0 ? 0 : rng() % 50);
        const size_t off = rng() % 100;
        const size_t c   = 1 + rng() % 512;
        check_case(off, sp, dp, w, h, c, rng);
        n_cases++;
    }
    // the production geometry scaled down: 64 MiB ring -> slot 32 MiB; a 1 GiB layer = 32 chunks; here 1 MiB / 32 KiB
    check_case(0, 1u << 20, 1u << 20, 1u << 20, 1, 32u << 10, rng);
    check_case(4096, (1u << 20) + 4096, 1u << 20, 1u << 20, 1, 32u << 10, rng);
    n_cases += 2;

    // 3: mutant - one slot instead of the ring must corrupt a multi-chunk read
    {
        const size_t w = 1000, h = 1;
        std::vector<uint8_t> src(w);
        for (auto & b : src) {
            b = (uint8_t) rng();
        }
        std::vector<ggml_vk_staging_piece> pieces;
        std::vector<ggml_vk_staging_chunk> chunks;
        ggml_vk_staging_plan_2d(0, w, w, w, h, 64, pieces, chunks);
        std::vector<uint8_t> ring(w, 0), mono(w, 0);
        run_ring(src, ring, 64, 2, pieces, chunks);
        run_ring(src, mono, 64, 1, pieces, chunks);
        CHECK(ring == src);
        CHECK(mono != src); // the check would see a broken ring
    }

    // empty inputs
    {
        std::vector<ggml_vk_staging_piece> pieces;
        std::vector<ggml_vk_staging_chunk> chunks;
        ggml_vk_staging_plan_2d(0, 0, 0, 0, 5, 64, pieces, chunks);
        CHECK(chunks.empty() && pieces.empty());
        ggml_vk_staging_plan_2d(0, 8, 8, 8, 0, 64, pieces, chunks);
        CHECK(chunks.empty() && pieces.empty());
    }

    // 4: parser
    {
        uint32_t v = 7;
        CHECK(ggml_vk_parse_sync_read_chunk_mib(nullptr, v) && v == 64);
        v = 7; CHECK(ggml_vk_parse_sync_read_chunk_mib("", v) && v == 64);
        v = 7; CHECK(ggml_vk_parse_sync_read_chunk_mib("0", v) && v == 0);
        v = 7; CHECK(ggml_vk_parse_sync_read_chunk_mib("1", v) && v == 2);
        v = 7; CHECK(ggml_vk_parse_sync_read_chunk_mib("128", v) && v == 128);
        v = 7; CHECK(ggml_vk_parse_sync_read_chunk_mib("4096", v) && v == 4096);
        for (const char * bad : { "4097", "-1", " 64", "64 ", "0x40", "64M", "abc", "99999999999999999999" }) {
            v = 7;
            CHECK(!ggml_vk_parse_sync_read_chunk_mib(bad, v) && v == 7);
        }
    }

    // 5: MiB -> bytes in 64-bit; a 32-bit size_t must not wrap 4096 MiB to 0 (chunking silently off)
    {
        const uint64_t m64 = UINT64_MAX, m32 = UINT32_MAX;
        CHECK(ggml_vk_sync_read_chunk_bytes(0,    m64) == 0);
        CHECK(ggml_vk_sync_read_chunk_bytes(64,   m64) == (64ull << 20));
        CHECK(ggml_vk_sync_read_chunk_bytes(4096, m64) == (4096ull << 20));
        CHECK(ggml_vk_sync_read_chunk_bytes(64,   m32) == (64ull << 20));
        CHECK(ggml_vk_sync_read_chunk_bytes(2047, m32) == (2047ull << 20));
        CHECK(ggml_vk_sync_read_chunk_bytes(2048, m32) == (2047ull << 20));
        CHECK(ggml_vk_sync_read_chunk_bytes(4096, m32) == (2047ull << 20));
        CHECK(ggml_vk_sync_read_chunk_bytes(4096, m32) != 0 && ggml_vk_sync_read_chunk_bytes(4096, m32) <= m32);
    }

    if (n_fail == 0) {
        printf("test-vk-staging-plan: OK (%d cases)\n", n_cases);
        return 0;
    }
    fprintf(stderr, "test-vk-staging-plan: %d failure(s)\n", n_fail);
    return 1;
}
