// [TAG_BS_GUARD] 0226: server_backend_sampling_check - support follows the target context's resolved per-sequence output
//   limit (server_output_limits) against one verify round (1 + the drafter's ceiling); tensor split never; the spec type
//   and n_max reported in /props; the exact 400 texts
#include "server-context.h"

#include <cstdio>
#include <string>

static int n_fail = 0;

#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            n_fail++;                                                    \
        }                                                                \
    } while (0)

int main() {
    {   // no speculation: supported
        common_params p;
        const auto r = server_backend_sampling_check(p);
        CHECK(r.supported && r.spec_type == "none" && r.n_max == 0 && r.error.empty());
    }
    {   // draft-mtp (prod shape, n_max 3 <= n_batch - 1): supported, the target limit is 1 + 3
        common_params p;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        p.speculative.draft.n_max = 3;
        const auto r = server_backend_sampling_check(p);
        CHECK(r.supported && r.spec_type == "draft-mtp" && r.n_max == 3 && r.per_seq_max == 4);
    }
    {   // draft-mtp with n_max 0 still drafts one token: the effective ceiling 1 is reported and covered
        common_params p;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        p.speculative.draft.n_max = 0;
        const auto r = server_backend_sampling_check(p);
        CHECK(r.supported && r.n_max == 1 && r.per_seq_max == 2);
    }
    {   // multi-slot draft-mtp: the per-sequence limit is unchanged
        common_params p;
        p.n_parallel = 4;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        p.speculative.draft.n_max = 3;
        const auto r = server_backend_sampling_check(p);
        CHECK(r.supported && r.per_seq_max == 4);
    }
    {   // ngram-simple: same formula, supported
        common_params p;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE };
        const auto r = server_backend_sampling_check(p);
        CHECK(r.supported && r.spec_type == "ngram-simple" && r.per_seq_max == 1 + r.n_max);
    }
    {   // dflash with and without draft backend sampling: the target limit does not depend on it
        common_params p;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
        p.speculative.draft.n_max = 8;
        p.speculative.draft.backend_sampling = false;
        CHECK(server_backend_sampling_check(p).supported);
        p.speculative.draft.backend_sampling = true;
        CHECK(server_backend_sampling_check(p).supported);
    }
    {   // contrived mismatch: a verify round (1 + 16) larger than the batch (8) - the limit is capped at n_batch
        common_params p;
        p.n_batch = 8;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        p.speculative.draft.n_max = 16;
        const auto r = server_backend_sampling_check(p);
        CHECK(!r.supported && r.n_max == 16 && r.per_seq_max == 8);
        CHECK(r.error == "backend_sampling is not supported on this server: speculative verify needs 1+16 outputs per "
                         "sequence but the context allows 8");
        p.speculative.draft.n_max = 7; // 1 + 7 == n_batch: fits
        CHECK(server_backend_sampling_check(p).supported);
    }
    {   // contrived mismatch: an embedding context outputs one per sequence
        common_params p;
        p.embedding = true;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE };
        const auto r = server_backend_sampling_check(p);
        CHECK(!r.supported && r.per_seq_max == 1);
    }
    {   // tensor split: never, with or without speculation
        common_params p;
        p.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        auto r = server_backend_sampling_check(p);
        CHECK(!r.supported && r.error == "backend_sampling is not supported on this server: tensor split");
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        r = server_backend_sampling_check(p);
        CHECK(!r.supported && r.error == "backend_sampling is not supported on this server: tensor split");
    }

    if (n_fail) {
        fprintf(stderr, "%d failure(s)\n", n_fail);
        return 1;
    }
    printf("test-backend-sampling-guard: OK\n");
    return 0;
}
