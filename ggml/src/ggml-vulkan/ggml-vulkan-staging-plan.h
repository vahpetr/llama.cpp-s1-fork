#pragma once

// [TAG_VK_READ_CHUNK] Chunked staging plan for the synchronous device->host read (ggml_vk_buffer_read_2d) of a
// host-visible UMA buffer that is not HOST_CACHED (write-combined: a direct CPU read is uncached, so the bytes go
// through vkCmdCopyBuffer into the HOST_CACHED device->sync_staging and a memcpy out of it). Before this the staging
// buffer was grown to the whole read (one KV layer of a slot save: ~0.5-1 GiB of GTT at 1M) and never shrunk; now a
// read larger than GGML_VK_SYNC_READ_CHUNK_MIB (default 64) streams through a fixed ring of two slots of half that
// size each, so the staging held after a save is <= the chunk.
//
// Header-only and free of Vulkan types, so tests/test-vk-staging-plan.cpp checks on the CPU that executing a plan
// (copy src -> staging, memcpy staging -> dst) yields the same bytes as the direct 2D copy and never touches more
// than `cap` bytes of a slot.
//
//   ggml_vk_staging_plan_2d(offset, spitch, dpitch, width, height, cap, pieces, chunks)
//     splits the 2D read of `height` rows of `width` bytes (source row i at offset + i*spitch, destination row i at
//     i*dpitch) into pieces {src, stg, dst, n}: copy n bytes from src to staging slot offset stg, then from there to
//     dst. Pieces are grouped into chunks of <= cap bytes, rows split across chunks when width > cap; adjacent pieces
//     that are contiguous in all three address spaces are merged (a contiguous read is one piece per chunk).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_vk_staging_piece {
    size_t src; // byte offset in the source buffer
    size_t stg; // byte offset in the staging slot (0 .. cap)
    size_t dst; // byte offset from the destination pointer
    size_t n;
};

struct ggml_vk_staging_chunk {
    size_t first; // index of the first piece
    size_t count; // number of pieces
    size_t bytes; // staging bytes used (<= cap)
};

static inline void ggml_vk_staging_plan_2d(size_t offset, size_t spitch, size_t dpitch, size_t width, size_t height, size_t cap,
                                           std::vector<ggml_vk_staging_piece> & pieces, std::vector<ggml_vk_staging_chunk> & chunks) {
    pieces.clear();
    chunks.clear();
    if (width == 0 || height == 0 || cap == 0) {
        return;
    }
    size_t used = 0;
    auto close_chunk = [&]() {
        const size_t first = chunks.empty() ? 0 : chunks.back().first + chunks.back().count;
        chunks.push_back({ first, pieces.size() - first, used });
        used = 0;
    };
    for (size_t r = 0; r < height; r++) {
        for (size_t c = 0; c < width; ) {
            const size_t n   = std::min(width - c, cap - used);
            const size_t src = offset + r*spitch + c;
            const size_t dst = r*dpitch + c;
            const size_t first = chunks.empty() ? 0 : chunks.back().first + chunks.back().count;
            if (pieces.size() > first) {
                ggml_vk_staging_piece & p = pieces.back();
                if (p.src + p.n == src && p.stg + p.n == used && p.dst + p.n == dst) {
                    p.n += n;
                    used += n;
                    c += n;
                    if (used == cap) {
                        close_chunk();
                    }
                    continue;
                }
            }
            pieces.push_back({ src, used, dst, n });
            used += n;
            c += n;
            if (used == cap) {
                close_chunk();
            }
        }
    }
    if (used > 0) {
        close_chunk();
    }
}

// GGML_VK_SYNC_READ_CHUNK_MIB: absent / empty -> 64; decimal digits 0..4096 accepted (0 = off: the staging buffer is
// sized to the whole read, the behaviour before the chunking); 1 is raised to 2 (a slot is half the chunk). Anything
// else is rejected with the output untouched (the backend keeps the default and warns).
static inline bool ggml_vk_parse_sync_read_chunk_mib(const char * s, uint32_t & out) {
    if (s == nullptr || *s == '\0') {
        out = 64;
        return true;
    }
    uint64_t v = 0;
    for (const char * p = s; *p; p++) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        v = v*10 + (uint64_t) (*p - '0');
        if (v > 4096) {
            return false;
        }
    }
    out = v == 1 ? 2 : (uint32_t) v;
    return true;
}
