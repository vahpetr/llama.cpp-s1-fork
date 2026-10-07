// [TAG_CACHE_COMPAT2] compatibility key v2 of the disk prompt cache: key composition, identity comparison, env
//   registry classification and digest, sampled model identity, measurement memo
#include "server-disk-cache.h"
#include "arg.h"

#include <atomic>
#include <cstdio>
#include <iterator>
#include <thread>
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
    p.param_digest = "ffffffffffffffff";
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
                                                "meas=bbbbbbbbbbbbbbbb|depth=" + std::to_string(DISK_CACHE_DEPTH_VERSION) + "|env=-|param=ffffffffffffffff");
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
        p = parts(); p.param_digest = "0000000000000000";
        CHECK(server_disk_cache_fingerprint_v2(cheap_a, p).substr(0, 16) != k0);
    }

    // [TAG_CACHE_COMPAT2] parsed-param digest: argv and LLAMA_ARG_* env aliases of one option give one digest;
    //   a semantic option changes it (effective batch, threads, offload, speculative, --override-kv)
    {
        const auto canon = [](std::vector<const char *> argv, std::vector<std::pair<const char *, const char *>> env) {
            for (auto & e : env) {
                setenv(e.first, e.second, 1);
            }
            common_params params;
            argv.insert(argv.begin(), "llama-server");
            const bool ok = common_params_parse((int) argv.size(), const_cast<char **>(argv.data()), params,
                                                LLAMA_EXAMPLE_SERVER);
            for (auto & e : env) {
                unsetenv(e.first);
            }
            return ok ? server_cache_param_canon(params) : std::string("parse-failed");
        };
        const std::string base = canon({ "-m", "x.gguf" }, {});
        CHECK(base != "parse-failed");
        const std::string a1 = canon({ "-m", "x.gguf", "-b", "64", "-ub", "32", "-t", "3", "-nkvo", "--spec-draft-n-max", "5" }, {});
        const std::string e1 = canon({ "-m", "x.gguf" }, { { "LLAMA_ARG_BATCH", "64" }, { "LLAMA_ARG_UBATCH", "32" },
                                                          { "LLAMA_ARG_THREADS", "3" }, { "LLAMA_ARG_NO_KV_OFFLOAD", "1" },
                                                          { "LLAMA_ARG_SPEC_DRAFT_N_MAX", "5" } });
        CHECK(a1 != "parse-failed");
        CHECK(a1 == e1);
        CHECK(a1 != base);
        for (auto args : std::vector<std::vector<const char *>>{
                 { "-b", "64" }, { "-ub", "32" }, { "-t", "3" }, { "-tb", "3" }, { "-nkvo" }, { "-fa", "off" },
                 { "--spec-draft-n-max", "5" }, { "--override-kv", "qwen4exp.indexer_top_k=int:512" }, { "--no-op-offload" } }) {
            args.insert(args.begin(), { "-m", "x.gguf" });
            const std::string c = canon(args, {});
            if (c == base || c == "parse-failed") {
                fprintf(stderr, "FAIL param digest unchanged by %s\n", args[2]);
                n_fail++;
            }
        }
        // order of options does not matter
        CHECK(canon({ "-m", "x.gguf", "-t", "3", "-b", "64" }, {}) == canon({ "-b", "64", "-m", "x.gguf", "-t", "3" }, {}));
        // a control-plane option (exact-qualified) leaves it
        CHECK(canon({ "-m", "x.gguf", "--port", "18999" }, {}) == base);
        CHECK(server_cache_env_digest({ "LLAMA_ARG_BATCH=64" }) == "-");
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
                            "LLAMA_ARG_CACHE_DISK", "LLAMA_ARG_CACHE_DISK_ADOPT", "LLAMA_ARG_PORT", "LLAMA_CACHE", "LLAMA_ARG_ROPE_SCALING", "LLAMA_ARG_BATCH" }) {
        if (server_cache_env_classify(n) != SERVER_CACHE_ENV_EXACT) {
            fprintf(stderr, "expected exact: %s\n", n);
            n_fail++;
        }
    }
    for (const char * n : { "LLAMA_ATTN_ROT_DISABLE", "LLAMA_MTP_CARRIER_ZERO", "LLAMA_QSA_MIN_KV", "LLAMA_KQ_VIS",
                            "LLAMA_QSA_POOL_REMAP", "GGML_VK_DISABLE_F16", "GGML_VK_DISABLE_COOPMAT", "GGML_VK_QSA_TOPK",
                            "GGML_TOTAL_THREADS", "LLAMA_MTP_QSA", "LLAMA_PROBE_DUMMY",
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
        // [TAG_CACHE_COMPAT2] malformed / truncated / foreign records are rejected; a failed value is never published
        const fs::path mp = dir / "compat-memo" / (mk + ".txt");
        std::string rec;
        {
            std::ifstream f(mp, std::ios::binary);
            rec.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        const auto put_raw = [&](const std::string & text) {
            std::ofstream f(mp, std::ios::binary | std::ios::trunc);
            f << text;
        };
        for (size_t n = 0; n < rec.size(); n++) {
            put_raw(rec.substr(0, n)); // every truncation
            if (server_cache_memo_get(dir.string(), mk, m)) {
                fprintf(stderr, "FAIL truncated memo (%zu of %zu bytes) accepted\n", n, rec.size());
                n_fail++;
            }
        }
        for (size_t i = 0; i < rec.size(); i++) {
            std::string bad = rec;
            bad[i] = bad[i] == 'a' ? 'b' : 'a'; // every single-byte corruption
            put_raw(bad);
            if (server_cache_memo_get(dir.string(), mk, m)) {
                fprintf(stderr, "FAIL corrupted memo (byte %zu) accepted\n", i);
                n_fail++;
            }
        }
        put_raw(rec + "x");
        CHECK(!server_cache_memo_get(dir.string(), mk, m));
        put_raw("0123456789abcdef\n"); // the step-2 r0 format
        CHECK(!server_cache_memo_get(dir.string(), mk, m));
        const std::string mk2 = server_cache_memo_key(cheap_b, "cpu");
        CHECK(server_cache_memo_put(dir.string(), mk2, "1111111111111111"));
        fs::copy_file(dir / "compat-memo" / (mk2 + ".txt"), mp, fs::copy_options::overwrite_existing);
        CHECK(!server_cache_memo_get(dir.string(), mk, m)); // another key's record under this name
        CHECK(!server_cache_memo_put(dir.string(), mk, "unmeasured-0123"));
        // concurrent publishers + a reader: never a torn or invalid record, no temporary left behind
        {
            std::atomic<int> bad_reads{0};
            std::atomic<bool> stop{false};
            std::vector<std::thread> th;
            for (int t = 0; t < 6; t++) {
                th.emplace_back([&, t]() {
                    for (int k = 0; k < 40; k++) {
                        if (!server_cache_memo_put(dir.string(), mk, t % 2 ? "aaaaaaaaaaaaaaaa" : "bbbbbbbbbbbbbbbb")) {
                            bad_reads++;
                        }
                    }
                });
            }
            std::thread rd([&]() {
                while (!stop) {
                    std::string v;
                    if (server_cache_memo_get(dir.string(), mk, v) && v != "aaaaaaaaaaaaaaaa" && v != "bbbbbbbbbbbbbbbb") {
                        bad_reads++;
                    }
                }
            });
            for (auto & x : th) {
                x.join();
            }
            stop = true;
            rd.join();
            CHECK(bad_reads == 0);
            CHECK(server_cache_memo_get(dir.string(), mk, m));
            size_t n_tmp = 0;
            for (const auto & de : fs::directory_iterator(dir / "compat-memo")) {
                n_tmp += de.path().filename().string().find(".tmp") != std::string::npos;
            }
            CHECK(n_tmp == 0);
        }
        fs::remove_all(dir);
    }
    // [TAG_CACHE_COMPAT2] runtime identity of the memo: stable within a process, carries the CPU feature list and the
    //   driver environment (a driver switch forces a re-measurement)
    {
        const std::string r0 = server_cache_runtime_identity();
        CHECK(r0 == server_cache_runtime_identity());
        CHECK(r0.find("|feat=") != std::string::npos && r0.find("|libs=") != std::string::npos);
        setenv("RADV_DEBUG", "nocache", 1);
        const std::string r1 = server_cache_runtime_identity();
        unsetenv("RADV_DEBUG");
        CHECK(r1 != r0);
        CHECK(server_cache_memo_key(cheap_a, r0) != server_cache_memo_key(cheap_a, r1));
    }
    // [TAG_CACHE_COMPAT2] fail-closed: a failed measurement gives a non-reusable namespace (two failures never share)
    {
        const std::string u1 = server_cache_measure_component("");
        const std::string u2 = server_cache_measure_component("");
        CHECK(u1.rfind("unmeasured-", 0) == 0 && u1 != u2);
        CHECK(server_cache_measure_component("bbbbbbbbbbbbbbbb") == "bbbbbbbbbbbbbbbb");
        auto p1 = parts(); p1.measure_h = u1;
        auto p2 = parts(); p2.measure_h = u2;
        CHECK(server_disk_cache_fingerprint_v2(cheap_a, p1).substr(0, 16) != server_disk_cache_fingerprint_v2(cheap_a, p2).substr(0, 16));
        CHECK(server_disk_cache_fingerprint_v2(cheap_a, p1).substr(0, 16) != fa.substr(0, 16));
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
