// [TAG_MERGE_ROWS] end-to-end evidence for the qwen4exp load-time projection merges: a real model is fed a fixed
// pseudo-random prompt (prefill in -b batches of -ub ubatches) and then a fixed sequence of decode calls of n tokens each, with the
// logits of every row of every decode call hashed (FNV-1a 64 over the f32 bytes, full vocabulary). Two runs of the same
// binary - LLAMA_QWEN4EXP_MERGE=0 and the default - must print identical [merge-e2e] lines. Combine with
// LLAMA_QSA_DUMP_IDS=1 ([qsa-ids] per QSA layer) and GGML_VK_FUSION_COUNT=1 (fusion counts + dispatch signature).
//
// usage: test-qwen4exp-merge-e2e -m model.gguf [common args: -c, -ub, -b, -ngl, -ot, --load-mode, -fa ...]
//   E2E_N_PROMPT  prompt length (default 3000)
//   E2E_STEPS     comma list of decode-call sizes (default 1,2,3,4,5,6,7,8,9,16,6,1)
//   E2E_SEED      token stream seed (default 1)
// Output (stdout):
//   [merge-e2e] placement len=<n> fnv=0x<fnv of llama_model_placement_desc> model_size=<bytes>
//   [merge-e2e] prefill n=<n> last_logits=0x<fnv>
//   [merge-e2e] call=<k> n=<n> pos=<p> logits=0x<fnv of all rows>
//   [merge-e2e] DONE calls=<k> all=0x<fnv over every line's hash>
// (not a ctest: it needs a model)

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const void * p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const uint8_t *) p)[i]) * 0x100000001b3ull;
    }
    return h;
}

static int env_int(const char * name, int def) {
    const char * s = getenv(name);
    return (s && *s) ? atoi(s) : def;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int n_prompt = env_int("E2E_N_PROMPT", 3000);
    std::vector<int> steps = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 6, 1 };
    if (const char * s = getenv("E2E_STEPS")) {
        steps.clear();
        std::string l = s;
        size_t pos = 0;
        while (pos <= l.size()) {
            const size_t e = std::min(l.find(',', pos), l.size());
            if (e > pos) {
                steps.push_back(atoi(l.substr(pos, e - pos).c_str()));
            }
            pos = e + 1;
        }
    }
    uint32_t st = (uint32_t) env_int("E2E_SEED", 1) * 2654435761u + 7u;
    auto next_tok = [&]() -> llama_token {
        st = st * 1664525u + 1013904223u;
        // stay below the special-token range at the top of the vocabulary
        return (llama_token) ((st >> 4) % (uint32_t) (n_vocab > 1024 ? n_vocab - 1024 : n_vocab));
    };

    // the disk-cache placement key input (tensor names, order and buffer types): must not depend on the merge
    {
        const int32_t n = llama_model_placement_desc(model, nullptr, 0);
        std::string pd(n > 0 ? (size_t) n + 1 : 1, '\0');
        llama_model_placement_desc(model, pd.data(), pd.size());
        pd.resize(strlen(pd.c_str()));
        printf("[merge-e2e] placement len=%zu fnv=0x%016" PRIx64 " model_size=%" PRIu64 "\n", pd.size(),
               fnv1a(pd.data(), pd.size()), (uint64_t) llama_model_size(model));
    }

    uint64_t all = 0xcbf29ce484222325ull;
    int pos = 0;

    // prefill: llama_decode calls of -b tokens (the context splits each into -ub ubatches); logits of the last row only
    {
        const int n_batch = (int) llama_n_batch(ctx);
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        int last = 0;
        for (int i0 = 0; i0 < n_prompt; i0 += n_batch) {
            const int n = std::min(n_batch, n_prompt - i0);
            common_batch_clear(batch);
            for (int i = 0; i < n; ++i) {
                common_batch_add(batch, next_tok(), pos++, { 0 }, i0 + i == n_prompt - 1);
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "prefill decode failed\n");
                return 1;
            }
            last = n - 1;
        }
        const float * lg = llama_get_logits_ith(ctx, last);
        const uint64_t h = fnv1a(lg, (size_t) n_vocab * sizeof(float));
        all = fnv1a(&h, sizeof(h), all);
        printf("[merge-e2e] prefill n=%d last_logits=0x%016" PRIx64 "\n", n_prompt, h);
        fflush(stdout);
        llama_batch_free(batch);
    }

    int call = 0;
    for (const int n : steps) {
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            common_batch_add(batch, next_tok(), pos++, { 0 }, true);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode call %d failed\n", call);
            return 1;
        }
        uint64_t h = 0xcbf29ce484222325ull;
        for (int i = 0; i < n; ++i) {
            h = fnv1a(llama_get_logits_ith(ctx, i), (size_t) n_vocab * sizeof(float), h);
        }
        all = fnv1a(&h, sizeof(h), all);
        printf("[merge-e2e] call=%d n=%d pos=%d logits=0x%016" PRIx64 "\n", call, n, pos - n, h);
        fflush(stdout);
        llama_batch_free(batch);
        call++;
    }
    printf("[merge-e2e] DONE calls=%d all=0x%016" PRIx64 "\n", call, all);

    llama_backend_free();
    return 0;
}
