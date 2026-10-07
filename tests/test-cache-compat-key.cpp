// [TAG_CACHE_COMPAT2] compatibility key v2 of the disk prompt cache: key composition, identity comparison, env
//   registry classification and digest, sampled model identity, measurement memo
#include "server-disk-cache.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

static int n_fail = 0;

#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            n_fail++;                                                    \
        }                                                                \
    } while (0)

static server_cache_key_parts parts() {
    server_cache_key_parts p;
    p.model_id   = "m:0123456789abcdef|d:none|p:none";
    p.desc_h     = "aaaaaaaaaaaaaaaa";
    p.measure_h  = "bbbbbbbbbbbbbbbb";
    p.env_digest = "-";
    return p;
}

int main() {
    // a v1 (cheap) fingerprint as server_disk_cache_fingerprint writes it
    const std::string cheap_a = "1111111111111111 v7L|5:/m.gg|10@20|0:|0@0|ctx4096|k1|v1|dk1|dv1|0:|0@0|libllama.so=ab-1;|A=1;";
    const std::string cheap_b = "2222222222222222 v7L|5:/m.gg|10@21|0:|0@0|ctx4096|k1|v1|dk1|dv1|0:|0@0|libllama.so=cd-1;|A=2;";

    // key composition: the namespace key depends on the v2 parts only, never on the cheap identity
    const std::string fa = server_disk_cache_fingerprint_v2(cheap_a, parts());
    const std::string fb = server_disk_cache_fingerprint_v2(cheap_b, parts());
    CHECK(fa.substr(0, 16) == fb.substr(0, 16));
    CHECK(fa != fb);
    CHECK(server_disk_cache_same_identity(fa, fb));
    CHECK(server_disk_cache_compat2_body(fa) == "c2|model=m:0123456789abcdef|d:none|p:none|desc=aaaaaaaaaaaaaaaa|"
                                                "meas=bbbbbbbbbbbbbbbb|depth=" + std::to_string(DISK_CACHE_DEPTH_VERSION) + "|env=-");
    CHECK(server_disk_cache_compat2_body(cheap_a).empty());
    // the cheap body stays parseable for adopt eligibility, and equal structures stay equal
    CHECK(server_disk_cache_fingerprint_structure(fa) == server_disk_cache_fingerprint_structure(cheap_a));
    CHECK(!server_disk_cache_fingerprint_structure(fa).empty());

    // every component changes the key
    {
        const std::string k0 = fa.substr(0, 16);
        auto p = parts(); p.model_id = "m:fedcba9876543210|d:none|p:none";
        CHECK(server_disk_cache_fingerprint_v2(cheap_a, p).substr(0, 16) != k0);
        p = parts(); p.desc_h = "cccccccccccccccc";
        CHECK(server_disk_cache_fingerprint_v2(cheap_a, p).substr(0, 16) != k0);
        p = parts(); p.measure_h = "dddddddddddddddd";
        CHECK(server_disk_cache_fingerprint_v2(cheap_a, p).substr(0, 16) != k0);
        p = parts(); p.depth_version = DISK_CACHE_DEPTH_VERSION + 1;
        CHECK(server_disk_cache_fingerprint_v2(cheap_a, p).substr(0, 16) != k0);
        p = parts(); p.env_digest = "eeeeeeeeeeeeeeee";
        const std::string fe = server_disk_cache_fingerprint_v2(cheap_a, p);
        CHECK(fe.substr(0, 16) != k0);
        CHECK(!server_disk_cache_same_identity(fa, fe));
    }
    // v1 fingerprints keep exact comparison; v1 vs v2 never match
    CHECK(server_disk_cache_same_identity(cheap_a, cheap_a));
    CHECK(!server_disk_cache_same_identity(cheap_a, cheap_b));
    CHECK(!server_disk_cache_same_identity(cheap_a, fa));
    // a truncated / forged compat2 tail is no v2 body
    CHECK(server_disk_cache_compat2_body(fa.substr(0, fa.size() - 1)).empty());

    // env registry: exact knobs (appendix) vs semantic (registered or unregistered)
    for (const char * n : { "LLAMA_TRACE", "LLAMA_KQ_VIS_PROF", "LLAMA_GTT_ADMISSION_MIB", "LLAMA_SERVER_SLOTS_DEBUG",
                            "GGML_VK_PERF_LOGGER", "GGML_VK_PREFER_HOST_MEMORY", "GGML_SCHED_DEBUG", "GGML_VULKAN_SKIP_CHECKS",
                            "LLAMA_ARG_CACHE_DISK", "LLAMA_ARG_CACHE_DISK_ADOPT", "LLAMA_ARG_PORT", "LLAMA_CACHE" }) {
        if (server_cache_env_classify(n) != SERVER_CACHE_ENV_EXACT) {
            fprintf(stderr, "expected exact: %s\n", n);
            n_fail++;
        }
    }
    for (const char * n : { "LLAMA_ATTN_ROT_DISABLE", "LLAMA_MTP_CARRIER_ZERO", "LLAMA_QSA_MIN_KV", "LLAMA_KQ_VIS",
                            "LLAMA_QSA_POOL_REMAP", "GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_QSA_TOPK",
                            "GGML_TOTAL_THREADS", "LLAMA_MTP_QSA", "LLAMA_ARG_ROPE_SCALING", "LLAMA_PROBE_DUMMY",
                            "GGML_SOMETHING_NEW", "LLAMA_TRACEX", "GGML_VK_PERF_LOGGER_X" }) {
        if (server_cache_env_classify(n) != SERVER_CACHE_ENV_SEMANTIC) {
            fprintf(stderr, "expected semantic: %s\n", n);
            n_fail++;
        }
    }
    // digest: exact knobs do not count, semantic ones do (name and value), order does not matter
    {
        const std::string d0 = server_cache_env_digest({});
        CHECK(d0 == "-");
        CHECK(server_cache_env_digest({ "LLAMA_TRACE=1", "GGML_VK_PERF_LOGGER=1" }) == d0);
        const std::string d1 = server_cache_env_digest({ "LLAMA_KQ_VIS=1", "LLAMA_TRACE=1" });
        CHECK(d1 != d0);
        CHECK(server_cache_env_digest({ "LLAMA_TRACE=2", "LLAMA_KQ_VIS=1" }) == d1);
        CHECK(server_cache_env_digest({ "LLAMA_KQ_VIS=0" }) != d1);
        CHECK(server_cache_env_digest({ "LLAMA_KQ_VIS=1", "LLAMA_PROBE_DUMMY=1" }) != d1);
    }

    // sampled model identity and the memo
    {
        const fs::path dir = fs::temp_directory_path() / ("test-cache-compat-" + std::to_string(getpid()));
        fs::create_directories(dir);
        const std::string f = (dir / "m.gguf").string();
        {
            std::ofstream o(f, std::ios::binary);
            std::string blob(3u << 20, 'x');
            o << blob;
        }
        const std::string i0 = server_cache_file_identity(f);
        CHECK(i0.size() == 16);
        CHECK(server_cache_file_identity(f) == i0);
        {
            std::fstream o(f, std::ios::in | std::ios::out | std::ios::binary);
            o.seekp((3u << 20) - 1); // the tail sample covers the last byte
            o.put('y');
        }
        CHECK(server_cache_file_identity(f) != i0);
        CHECK(server_cache_file_identity((dir / "missing").string()) == "absent");
        CHECK(server_cache_file_identity("") == "none");

        const std::string mk = server_cache_memo_key(cheap_a, "cpu");
        CHECK(mk != server_cache_memo_key(cheap_b, "cpu"));
        CHECK(mk != server_cache_memo_key(cheap_a, "Vulkan0/AMD Radeon 780M/0000:c4:00.0;"));
        std::string m;
        CHECK(!server_cache_memo_get(dir.string(), mk, m));
        CHECK(server_cache_memo_put(dir.string(), mk, "0123456789abcdef"));
        CHECK(server_cache_memo_get(dir.string(), mk, m) && m == "0123456789abcdef");
        fs::remove_all(dir);
    }
    CHECK(server_cache_measure_hash("abc") == server_cache_measure_hash("abc"));
    CHECK(server_cache_measure_hash("abc") != server_cache_measure_hash("abd"));

    if (n_fail) {
        fprintf(stderr, "%d failure(s)\n", n_fail);
        return 1;
    }
    printf("test-cache-compat-key: OK\n");
    return 0;
}
