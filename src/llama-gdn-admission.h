#pragma once

// [TAG_GDN_REPLAY] free device memory for the gdn replay snapshot admission (llama_memory_recurrent::gdn_commit), header-only
//   so tests/test-gdn-replay.cpp ("admission") checks it with fake heaps.
//
// ggml_backend_vk_get_device_memory sums heapBudget - heapUsage over the heaps: (a) per heap it wrapped to ~2^64 when the
//   usage is over the budget (unsigned; clamped at 0 in ggml-vulkan since this change), so the admission passed exactly when
//   memory was overcommitted, and (b) on a UMA device (integrated GPU) every heap draws from the one shared system memory
//   pool (GTT), so the sum counts that pool more than once. Here: each heap's free = max(0, budget - usage); UMA -> the
//   largest of them (one pool), discrete -> the sum over the device-local heaps.

#include <cstddef>
#include <cstdint>

struct llama_gdn_heap {
    uint64_t budget;       // VkPhysicalDeviceMemoryBudgetPropertiesEXT::heapBudget
    uint64_t usage;        // ... ::heapUsage
    bool     device_local; // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
};

static inline uint64_t llama_gdn_heaps_free(const llama_gdn_heap * heaps, size_t n, bool uma) {
    uint64_t free_sum = 0;
    uint64_t free_max = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!uma && !heaps[i].device_local) {
            continue;
        }
        const uint64_t f = heaps[i].budget > heaps[i].usage ? heaps[i].budget - heaps[i].usage : 0;
        free_sum += f;
        free_max  = f > free_max ? f : free_max;
    }
    return uma ? free_max : free_sum;
}

// true = bytes + margin fit into free (no wrap: free < margin refuses)
static inline bool llama_gdn_admission_fits(uint64_t free, uint64_t bytes, uint64_t margin) {
    return free >= margin && bytes <= free - margin;
}
