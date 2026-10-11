// [TAG_BS_GUARD] 0226: server_backend_sampling_check - which loaded configurations can serve backend_sampling: true,
//   the spec type reported in /props and the exact 400 text
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

static const std::string spec_err(const std::string & type) {
    return "backend_sampling is not supported on this server: speculative decoding (" + type +
           ") verifies more than one output per sequence";
}

int main() {
    {   // no speculation: supported
        common_params p;
        const auto r = server_backend_sampling_check(p);
        CHECK(r.supported && r.spec_type == "none" && r.error.empty());
    }
    {   // draft-mtp (prod): unsupported
        common_params p;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        const auto r = server_backend_sampling_check(p);
        CHECK(!r.supported && r.spec_type == "draft-mtp" && r.error == spec_err("draft-mtp"));
    }
    {   // classic draft model without --spec-type: reported as draft-simple, unsupported
        common_params p;
        p.speculative.draft.mparams.path = "/nonexistent/draft.gguf";
        const auto r = server_backend_sampling_check(p);
        CHECK(!r.supported && r.spec_type == "draft-simple" && r.error == spec_err("draft-simple"));
    }
    {   // ngram: unsupported
        common_params p;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE };
        CHECK(!server_backend_sampling_check(p).supported);
    }
    {   // dflash / dspark with draft backend sampling (the default): supported
        common_params p;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
        p.speculative.draft.backend_sampling = true;
        auto r = server_backend_sampling_check(p);
        CHECK(r.supported && r.spec_type == "draft-dflash");
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        r = server_backend_sampling_check(p);
        CHECK(r.supported && r.spec_type == "draft-dspark");
        // ... but not without it (n_outputs_max_per_seq stays 1)
        p.speculative.draft.backend_sampling = false;
        CHECK(!server_backend_sampling_check(p).supported);
        // ... nor mixed with another speculation
        p.speculative.draft.backend_sampling = true;
        p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, COMMON_SPECULATIVE_TYPE_NGRAM_MOD };
        r = server_backend_sampling_check(p);
        CHECK(!r.supported && r.spec_type == "draft-dflash,ngram-mod");
    }
    {   // tensor split: never
        common_params p;
        p.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        const auto r = server_backend_sampling_check(p);
        CHECK(!r.supported && r.error == "backend_sampling is not supported on this server: tensor split");
    }

    if (n_fail) {
        fprintf(stderr, "%d failure(s)\n", n_fail);
        return 1;
    }
    printf("test-backend-sampling-guard: OK\n");
    return 0;
}
