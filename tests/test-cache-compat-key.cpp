// [TAG_CACHE_COMPAT2] compatibility key v2 of the disk prompt cache: key composition, identity comparison, env
//   registry classification and digest, sampled model identity, measurement memo
#include "server-disk-cache.h"
#include "arg.h"
#include "server-task.h"
#include "server-context.h" // [TAG_MULTISLOT] server_multislot_envelope_check
#include "../src/llama-ext.h" // [TAG_CELL_EXT_PROVENANCE] LLAMA_STATE_IXY_MARK

#include <atomic>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <thread>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/sysmacros.h>
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
                 { "--spec-draft-n-max", "5" }, { "--override-kv", "qwen4exp.indexer_top_k=int:512" }, { "--no-op-offload" },
                 { "--numa", "distribute" }, { "--ctx-checkpoints", "4" }, { "--checkpoint-min-step", "64" } }) {
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
        // the checkpoint schedule: argv and env give one digest
        CHECK(canon({ "-m", "x.gguf", "--ctx-checkpoints", "4" }, {}) ==
              canon({ "-m", "x.gguf" }, { { "LLAMA_ARG_CTX_CHECKPOINTS", "4" } }));
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
    // [TAG_T2_SEQ_RM_COUNTED] LLAMA_SPEC_OVERLAP is output-exact (class A): unset / =0 / =1 give one env digest, so the
    //   namespace key of a v69 prod start with OVERLAP=1 is the v68 one; GGML_VK_F32ACC(_SCOPE) changes numerics and
    //   stays semantic (a start with it set gets its own namespace)
    {
        CHECK(server_cache_env_classify("LLAMA_SPEC_OVERLAP") == SERVER_CACHE_ENV_EXACT);
        const std::string d0 = server_cache_env_digest({});
        CHECK(server_cache_env_digest({ "LLAMA_SPEC_OVERLAP=0" }) == d0);
        CHECK(server_cache_env_digest({ "LLAMA_SPEC_OVERLAP=1" }) == d0);
        const std::string dk = server_cache_env_digest({ "LLAMA_KQ_VIS=1" });
        CHECK(server_cache_env_digest({ "LLAMA_KQ_VIS=1", "LLAMA_SPEC_OVERLAP=1" }) == dk);
        CHECK(server_cache_env_classify("GGML_VK_F32ACC") == SERVER_CACHE_ENV_SEMANTIC);
        CHECK(server_cache_env_classify("GGML_VK_F32ACC_SCOPE") == SERVER_CACHE_ENV_SEMANTIC);
        CHECK(server_cache_env_digest({ "GGML_VK_F32ACC=1" }) != d0);
        CHECK(server_cache_env_digest({ "GGML_VK_F32ACC=1", "LLAMA_SPEC_OVERLAP=1" }) ==
              server_cache_env_digest({ "GGML_VK_F32ACC=1" }));
        CHECK(server_cache_env_digest({ "GGML_VK_F32ACC=1", "GGML_VK_F32ACC_SCOPE=trunk" }) !=
              server_cache_env_digest({ "GGML_VK_F32ACC=1" }));
        printf("env registry: LLAMA_SPEC_OVERLAP unset/0/1 = one digest (exact), GGML_VK_F32ACC(_SCOPE) semantic\n");
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
        // KNOWN LIMITATION (design R80.2: sampled, not complete, content identity): an equal-size edit outside the
        //   four 1 MiB samples keeps the identity - here byte 1.5 MiB of an 8 MiB file (samples at 0, 2.33, 4.67, 7 MiB)
        {
            const std::string g = (dir / "big.gguf").string();
            {
                std::ofstream o(g, std::ios::binary);
                o << std::string(8u << 20, 'x');
            }
            const std::string g0 = server_cache_file_identity(g);
            {
                std::fstream o(g, std::ios::in | std::ios::out | std::ios::binary);
                o.seekp((3u << 20) / 2);
                o.put('y');
            }
            CHECK(server_cache_file_identity(g) == g0); // documents the limitation; a sampled edit changes it (above)
        }
        CHECK(server_cache_file_identity((dir / "missing").string()) == "absent");
        CHECK(server_cache_file_identity("") == "none");

        const std::string mk = server_cache_memo_key(cheap_a, "cpu");
        CHECK(mk != server_cache_memo_key(cheap_b, "cpu"));
        CHECK(mk != server_cache_memo_key(cheap_a, "Vulkan0/AMD Radeon 780M/0000:c4:00.0;"));
        // [TAG_CACHE_COMPAT2] the effective placement (astra r2): another placement / NUMA strategy is another memo
        {
            CHECK(mk == server_cache_memo_key(cheap_a, "cpu", "-"));
            CHECK(mk != server_cache_memo_key(cheap_a, "cpu", "1111111111111111"));
            CHECK(server_cache_memo_key(cheap_a, "cpu", "1111111111111111") !=
                  server_cache_memo_key(cheap_a, "cpu", "2222222222222222"));
            CHECK(server_cache_model_placement(nullptr) == "none");
            common_params pn;
            const std::string p0 = server_cache_placement_identity(pn, nullptr, nullptr);
            CHECK(p0 == server_cache_placement_identity(pn, nullptr, nullptr));
            pn.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
            CHECK(p0 != server_cache_placement_identity(pn, nullptr, nullptr));
            CHECK(server_cache_memo_key(cheap_a, "cpu", p0) !=
                  server_cache_memo_key(cheap_a, "cpu", server_cache_placement_identity(pn, nullptr, nullptr)));
        }
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
    // [TAG_CACHE_COMPAT2] descriptor_full is mandatory for own / format >= 8 entries (astra r2)
    {
        const std::string dk = "0123456789abcdef";
        CHECK(server_disk_cache_desc_ok(true, 8, dk, dk));
        CHECK(!server_disk_cache_desc_ok(true, 8, "", dk));       // own, absent => miss
        CHECK(!server_disk_cache_desc_ok(true, 8, "fedcba9876543210", dk));
        CHECK(!server_disk_cache_desc_ok(false, 8, "", dk));      // adopted format 8 without it => miss
        CHECK(server_disk_cache_desc_ok(false, 7, "", dk));       // adopted pre-8: per-stream envelope checks decide
        CHECK(!server_disk_cache_desc_ok(false, 7, "fedcba9876543210", dk));
        CHECK(server_disk_cache_desc_ok(false, 8, dk, dk));
    }
    // [TAG_CACHE_COMPAT2] the RAM cache swaps in only a state of the loading contexts' descriptor
    {
        server_prompt_cache pc(64, 0);
        server_prompt pr;
        pr.tokens = server_tokens(llama_tokens{ 1, 2, 3, 4, 5, 6, 7, 8 }, false);
        CHECK(pc.alloc(pr, 0, 0, "aaaaaaaaaaaaaaaa") != nullptr);
        server_prompt slot;
        const server_tokens next(llama_tokens{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, false);
        CHECK(pc.load(slot, next, nullptr, nullptr, 0, "bbbbbbbbbbbbbbbb")); // no swap: other descriptor
        CHECK(slot.tokens.size() == 0 && pc.states.size() == 1);
        CHECK(server_cache_state_desc(nullptr, nullptr, "") != server_cache_state_desc(nullptr, nullptr, "mtp:1"));
    }
    // [TAG_CACHE_COMPAT2] mapped-library identity (astra r2): the device + inode of a maps line must match stat() of
    //   the path; a replaced / unlinked / missing file is unverifiable, and an unverifiable runtime has no memo key
    {
        const fs::path dir = fs::temp_directory_path() / ("cache-compat-maps-" + std::to_string(getpid()));
        fs::create_directories(dir);
        const fs::path so = dir / "libfoo.so.1";
        { std::ofstream o(so); o << "x"; }
        struct stat st {};
        CHECK(stat(so.c_str(), &st) == 0);
        const auto line = [&](unsigned long long ino, unsigned maj, unsigned mnr, const std::string & path) {
            char b[64];
            snprintf(b, sizeof(b), "%02x:%02x %llu", maj, mnr, ino);
            return "7f0000000000-7f0000001000 r-xp 00000000 " + std::string(b) + "    " + path;
        };
        const unsigned maj = major(st.st_dev), mnr = minor(st.st_dev);
        std::string e;
        CHECK(server_cache_maps_lib(line(st.st_ino, maj, mnr, so.string()), e) == 1 && e.find(so.string()) == 0);
        CHECK(server_cache_maps_lib(line(st.st_ino + 1, maj, mnr, so.string()), e) == -1); // replaced: other inode
        CHECK(server_cache_maps_lib(line(st.st_ino, maj, mnr + 1, so.string()), e) == -1); // other device
        CHECK(server_cache_maps_lib(line(st.st_ino, maj, mnr, so.string() + " (deleted)"), e) == -1);
        CHECK(server_cache_maps_lib(line(st.st_ino, maj, mnr, (dir / "libgone.so").string()), e) == -1);
        CHECK(server_cache_maps_lib("7f0000000000-7f0000001000 r-xp 00000000 garbage " + so.string(), e) == -1);
        // untracked: no path, not a shared object, our own libraries (content-hashed in the cheap identity)
        CHECK(server_cache_maps_lib("7f0000000000-7f0000001000 rw-p 00000000 00:00 0    [heap]", e) == 0);
        CHECK(server_cache_maps_lib(line(1, 0, 0, "/x/model.gguf"), e) == 0);
        CHECK(server_cache_maps_lib(line(1, 0, 0, "/x/libggml-base.so (deleted)"), e) == 0);
        // replace the file under the same path: the old mapping's inode no longer verifies
        fs::remove(so);
        { std::ofstream o(so); o << "y"; }
        struct stat st2 {};
        CHECK(stat(so.c_str(), &st2) == 0);
        if (st2.st_ino != st.st_ino) {
            CHECK(server_cache_maps_lib(line(st.st_ino, maj, mnr, so.string()), e) == -1);
        }
        CHECK(server_cache_memo_key(cheap_a, SERVER_CACHE_RUNTIME_UNVERIFIED, "-").empty());
        CHECK(!server_cache_memo_key(cheap_a, "cpu", "-").empty());
        // this process's own mappings verify
        CHECK(server_cache_runtime_identity() != SERVER_CACHE_RUNTIME_UNVERIFIED);
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
    // [TAG_CACHE_COMPAT2] full coverage: a change at EVERY byte position of a descriptor-sized string changes the
    //   hash16 used for the descriptor key / envelope v2 / measurement (the weak v1 hash16 saw lanes a / c only)
    {
        std::string d;
        for (int i = 0; d.size() < 3000; i++) {
            d += "layer" + std::to_string(i) + "{k=q8_0,v=q8_0,dims=128x8,rot=64};";
        }
        const std::string h0 = server_cache_measure_hash(d);
        size_t n_same = 0;
        for (size_t i = 0; i < d.size(); i++) {
            std::string m = d;
            m[i] ^= 0x01;
            n_same += server_cache_measure_hash(m) == h0;
        }
        CHECK(n_same == 0);
        CHECK(server_cache_measure_hash(d + "x") != h0);
        CHECK(server_cache_measure_hash(d.substr(0, d.size() - 1)) != h0);
    }
    CHECK(server_cache_measure_hash("abc") == server_cache_measure_hash("abc"));
    CHECK(server_cache_measure_hash("abc") != server_cache_measure_hash("abd"));

    // [TAG_MULTISLOT] stage 5.1 item 2: the support-envelope decision (server_multislot_envelope_check)
    {
        // np 1: unchanged, nothing printed, whatever the knob / mmproj / context shift
        for (const char * k : { (const char *) nullptr, "", "0", "1", "yes" }) {
            const auto r = server_multislot_envelope_check(k, 1, true, true);
            CHECK(r.ok && !r.strict && r.proof.empty());
        }
        // knob unset / 0 / invalid with np > 1: today's behaviour (mmproj, shift, fork allowed), a WARN proof only
        for (const char * k : { (const char *) nullptr, "", "0", "on", "2", "1 " }) {
            const auto r = server_multislot_envelope_check(k, 5, true, true);
            CHECK(r.ok && !r.strict);
            CHECK(r.proof.rfind("[multislot-envelope] np=5 text_only=0 shared_prefix=1 ctx_shift=1 strict=0", 0) == 0);
            CHECK(r.proof.find("NOT qualified") != std::string::npos);
        }
        CHECK(server_multislot_envelope_check("on", 2, false, false).proof.find("invalid LLAMA_QSA_HOMOGENEOUS='on'") != std::string::npos);
        // strict + text only + no shift: starts, the exact startup proof
        {
            const auto r = server_multislot_envelope_check("1", 2, false, false);
            CHECK(r.ok && r.strict && r.error.empty());
            CHECK(r.proof == "[multislot-envelope] np=2 text_only=1 shared_prefix=0 ctx_shift=0 strict=1");
            printf("%s\n", r.proof.c_str());
        }
        // strict + mmproj: refused; strict + context shift: refused
        {
            const auto r = server_multislot_envelope_check("1", 2, true, false);
            CHECK(!r.ok && r.strict && r.error.find("--mmproj") != std::string::npos);
            const auto s = server_multislot_envelope_check("1", 4, false, true);
            CHECK(!s.ok && s.strict && s.error.find("--no-context-shift") != std::string::npos);
            printf("%s\n%s\n", r.error.c_str(), s.error.c_str());
        }
    }

    // (R92) an adopted deep checkpoint dropped at a read for a state descriptor mismatch (balkon: "deep checkpoint
    //   ... unreadable (state descriptor mismatch (v2 ..., expected v2 ...)), dropping it") counts in the adopt
    //   source's `dropped` of /disk-cache (it reported 0): source A (descriptor X) stores a deep checkpoint, server B
    //   (descriptor Y, another key, the same structure) adopts A's namespace and reads it
    {
        const fs::path base = fs::temp_directory_path() / ("cache-compat-r92-" + std::to_string(getpid()));
        const fs::path ra = base / "a", rb = base / "b"; // two roots: the cache root lock is held for the process lifetime
        std::error_code ec;
        fs::remove_all(base, ec);
        server_cache_key_parts px = parts(), py = parts();
        py.desc_h = "cccccccccccccccc";
        const std::string fpa = server_disk_cache_fingerprint_v2(cheap_a, px);
        const std::string fpb = server_disk_cache_fingerprint_v2(cheap_a, py);
        const std::string ka = fpa.substr(0, 16);
        CHECK(ka != fpb.substr(0, 16));
        CHECK(server_disk_cache_fingerprint_structure(fpa) == server_disk_cache_fingerprint_structure(fpb));
        llama_tokens toks;
        for (int i = 0; i < 64; i++) {
            toks.push_back(i + 1);
        }
        std::string ck_key;
        {
            server_disk_cache a(ra.string(), 1ull << 30, 8, 30, fpa, false, 0.5f, 8, 8);
            a.set_descriptor(nullptr, nullptr, "carry-x");
            a.init();
            server_prompt pr;
            pr.tokens = server_tokens(toks, false);
            common_prompt_checkpoint c;
            c.n_tokens = 32;
            c.pos_min  = 0;
            c.pos_max  = 31;
            c.data_tgt.assign(64, 7);
            a.deep_put(pr, c, server_deep_ref::DEEP_KIND_ANCHOR);
            CHECK(pr.deep.size() == 1 && a.n_deep_ckpts() == 1);
            if (!pr.deep.empty()) {
                ck_key = pr.deep[0].key;
            }
        }
        // an entry of A (adoption needs one; this one does not read: counted as not adopted at the start)
        fs::create_directories(rb, ec);
        fs::copy(ra / ("ns-" + ka), rb / ("ns-" + ka), fs::copy_options::recursive, ec);
        CHECK(!ec);
        {
            std::ofstream m(rb / ("ns-" + ka) / "17081d812d27c40a.meta");
            m << "{\"fingerprint\":\"" << fpa << "\"}";
        }
        server_disk_cache b(rb.string(), 1ull << 30, 8, 30, fpb, false, 0.5f, 8, 8);
        b.set_descriptor(nullptr, nullptr, "carry-y");
        b.adopt_keys = { ka };
        b.init();
        const auto src = [&]() {
            const json st = b.status();
            return st.contains("adopt") && st["adopt"].size() == 1 ? st["adopt"][0] : json();
        };
        CHECK(src().value("key", std::string()) == ka);
        const uint64_t d0 = src().value("dropped", (uint64_t) 0);
        CHECK(src().value("dropped_ckpts", (uint64_t) 99) == 0);
        CHECK(b.n_deep_ckpts() == 1);
        common_prompt_checkpoint out;
        bool io = false;
        CHECK(!ck_key.empty() && !b.deep_get(ck_key, out, &io));
        CHECK(!io);
        CHECK(b.n_deep_ckpts() == 0);
        CHECK(src().value("dropped", (uint64_t) 0) == d0 + 1);
        CHECK(src().value("dropped_ckpts", (uint64_t) 0) == 1);
        printf("R92 adopt drop: dropped %llu -> %llu, dropped_ckpts %llu\n", (unsigned long long) d0,
               (unsigned long long) src().value("dropped", (uint64_t) 0),
               (unsigned long long) src().value("dropped_ckpts", (uint64_t) 0));
        // the source's file stays (read-only)
        CHECK(fs::exists(rb / ("ns-" + ka) / "ckpt" / (ck_key + ".ck")));
        fs::remove_all(base, ec);
    }

    // (R92, sol review) two LEGACY (flat) adopt sources share <root>/ckpt: a dropped deep checkpoint is charged to the
    //   source whose key its header names, once — not to every source whose ckpt/ directory matches. Sources A
    //   (descriptor X) and C (descriptor Z) each store one checkpoint; their files are laid out flat in B's root; B
    //   (descriptor Y) adopts both and reads A's checkpoint, then C's
    {
        const fs::path base = fs::temp_directory_path() / ("cache-compat-r92-flat-" + std::to_string(getpid()));
        const fs::path ra = base / "a", rc = base / "c", rb = base / "b";
        std::error_code ec;
        fs::remove_all(base, ec);
        server_cache_key_parts px = parts(), py = parts(), pz = parts();
        py.desc_h = "cccccccccccccccc";
        pz.desc_h = "dddddddddddddddd";
        const std::string fpa = server_disk_cache_fingerprint_v2(cheap_a, px);
        const std::string fpb = server_disk_cache_fingerprint_v2(cheap_a, py);
        const std::string fpc = server_disk_cache_fingerprint_v2(cheap_a, pz);
        const std::string ka = fpa.substr(0, 16), kc = fpc.substr(0, 16);
        CHECK(ka != kc && ka != fpb.substr(0, 16) && kc != fpb.substr(0, 16));
        const auto put_one = [&](const fs::path & root, const std::string & fp, const char * carry, int seed) {
            server_disk_cache s(root.string(), 1ull << 30, 8, 30, fp, false, 0.5f, 8, 8);
            s.set_descriptor(nullptr, nullptr, carry);
            s.init();
            llama_tokens toks;
            for (int i = 0; i < 64; i++) {
                toks.push_back(seed + i);
            }
            server_prompt pr;
            pr.tokens = server_tokens(toks, false);
            common_prompt_checkpoint c;
            c.n_tokens = 32;
            c.pos_min  = 0;
            c.pos_max  = 31;
            c.data_tgt.assign(64, (uint8_t) seed);
            s.deep_put(pr, c, server_deep_ref::DEEP_KIND_ANCHOR);
            CHECK(pr.deep.size() == 1);
            return pr.deep.empty() ? std::string() : pr.deep[0].key;
        };
        const std::string ck_a = put_one(ra, fpa, "carry-x", 3);
        const std::string ck_c = put_one(rc, fpc, "carry-z", 5);
        CHECK(!ck_a.empty() && !ck_c.empty() && ck_a != ck_c);
        // the flat (legacy) layout: <rb>/<id>.meta of each key + both checkpoints in <rb>/ckpt
        fs::create_directories(rb / "ckpt", ec);
        fs::copy_file(ra / ("ns-" + ka) / "ckpt" / (ck_a + ".ck"), rb / "ckpt" / (ck_a + ".ck"), ec);
        CHECK(!ec);
        fs::copy_file(rc / ("ns-" + kc) / "ckpt" / (ck_c + ".ck"), rb / "ckpt" / (ck_c + ".ck"), ec);
        CHECK(!ec);
        {
            std::ofstream m(rb / "17081d812d27c40a.meta");
            m << "{\"fingerprint\":\"" << fpa << "\"}";
            std::ofstream n(rb / "27081d812d27c40b.meta");
            n << "{\"fingerprint\":\"" << fpc << "\"}";
        }
        server_disk_cache b(rb.string(), 1ull << 30, 8, 30, fpb, false, 0.5f, 8, 8);
        b.set_descriptor(nullptr, nullptr, "carry-y");
        b.adopt_keys = { ka, kc };
        b.init();
        const auto src = [&](const std::string & k) {
            const json st = b.status();
            if (st.contains("adopt")) {
                for (const auto & a : st["adopt"]) {
                    if (a.value("key", std::string()) == k) {
                        return a;
                    }
                }
            }
            return json();
        };
        CHECK(src(ka).value("legacy", false) && src(kc).value("legacy", false));
        CHECK(b.n_deep_ckpts() == 2);
        const uint64_t da0 = src(ka).value("dropped", (uint64_t) 0), dc0 = src(kc).value("dropped", (uint64_t) 0);
        common_prompt_checkpoint out;
        bool io = false;
        CHECK(!b.deep_get(ck_a, out, &io) && !io);
        CHECK(src(ka).value("dropped_ckpts", (uint64_t) 0) == 1 && src(ka).value("dropped", (uint64_t) 0) == da0 + 1);
        CHECK(src(kc).value("dropped_ckpts", (uint64_t) 99) == 0 && src(kc).value("dropped", (uint64_t) 0) == dc0);
        CHECK(!b.deep_get(ck_c, out, &io) && !io);
        CHECK(src(ka).value("dropped_ckpts", (uint64_t) 0) == 1 && src(kc).value("dropped_ckpts", (uint64_t) 0) == 1);
        CHECK(src(kc).value("dropped", (uint64_t) 0) == dc0 + 1);
        CHECK(b.n_deep_ckpts() == 0);
        printf("R92 flat two-source drop: A dropped_ckpts %llu, C dropped_ckpts %llu\n",
               (unsigned long long) src(ka).value("dropped_ckpts", (uint64_t) 0),
               (unsigned long long) src(kc).value("dropped_ckpts", (uint64_t) 0));
        CHECK(fs::exists(rb / "ckpt" / (ck_a + ".ck")) && fs::exists(rb / "ckpt" / (ck_c + ".ck")));
        // (sol review) the learned target-compatible descriptor ([TAG_ADOPT_TARGET_ONLY] tgt_compat_dh) is per source,
        //   not per <root>/ckpt: a fresh B learns A's descriptor X from A's checkpoint, then C's Z from C's; keyed by the
        //   shared directory, C's learn found A's X and C's checkpoint was dropped as a descriptor mismatch. B2 gets its
        //   own root with the same flat layout (B still holds <rb>/.lock)
        {
            const fs::path rb2 = base / "b2";
            fs::create_directories(rb2 / "ckpt", ec);
            for (const std::string & f : { "ckpt/" + ck_a + ".ck", "ckpt/" + ck_c + ".ck",
                                           std::string("17081d812d27c40a.meta"), std::string("27081d812d27c40b.meta") }) {
                fs::copy_file(rb / f, rb2 / f, ec);
                CHECK(!ec);
            }
            server_disk_cache b2(rb2.string(), 1ull << 30, 8, 30, fpb, false, 0.5f, 8, 8);
            b2.set_descriptor(nullptr, nullptr, "carry-y");
            b2.adopt_keys = { ka, kc };
            b2.init();
            CHECK(b2.n_deep_ckpts() == 2);
            CHECK(b2.n_ckpt_target_only == 0);
            common_prompt_checkpoint oa, oc;
            bool io2 = false;
            CHECK(b2.deep_get(ck_a, oa, &io2, true) && !io2 && oa.n_tokens == 32 && oa.data_dft.empty());
            CHECK(b2.deep_get(ck_c, oc, &io2, true) && !io2 && oc.n_tokens == 32 && oc.data_dft.empty());
            CHECK(b2.n_ckpt_target_only == 2 && b2.n_deep_ckpts() == 2);
            printf("R92 flat two-source learn: %llu target-only checkpoint loads, %zu deep checkpoints kept\n",
                   (unsigned long long) b2.n_ckpt_target_only, b2.n_deep_ckpts());
        }
        fs::remove_all(base, ec);
    }

    // [TAG_CELL_EXT_PROVENANCE] a checkpoint stored by a build before the indexer x/y fix (its target blob does not
    //   open with the decoy header {1, 0, LLAMA_STATE_IXY_MARK} at byte 8) is replaced by a marked one under the same prefix key, never linked; a marked one
    //   is reused; an unmarked blob over an unmarked one (a model without the mark) reuses
    {
        const fs::path root = fs::temp_directory_path() / ("cache-compat-ixy-" + std::to_string(getpid()));
        std::error_code ec;
        fs::remove_all(root, ec);
        const std::string fp = server_disk_cache_fingerprint_v2(cheap_a, parts());
        llama_tokens toks;
        for (int i = 0; i < 64; i++) {
            toks.push_back(100 + i);
        }
        const auto blob = [](bool marked, uint8_t fill) {
            std::vector<uint8_t> d(64, fill);
            // marked: the decoy header {1, 0, mark}; unmarked: a recurrent section {cell_count 1, pos 31, n_seq_id 0}
            const uint32_t w[3] = { 1u, marked ? 0u : 31u, marked ? LLAMA_STATE_IXY_MARK : 0u };
            memcpy(d.data() + 8, w, sizeof(w));
            return d;
        };
        const auto put = [&](server_disk_cache & s, bool marked, uint8_t fill) {
            server_prompt pr;
            pr.tokens = server_tokens(toks, false);
            common_prompt_checkpoint c;
            c.n_tokens = 32;
            c.pos_min  = 0;
            c.pos_max  = 31;
            c.data_tgt = blob(marked, fill);
            s.deep_put(pr, c, server_deep_ref::DEEP_KIND_ANCHOR);
            return pr.deep.empty() ? std::string() : pr.deep[0].key;
        };
        const auto stored = [&](server_disk_cache & s, const std::string & key) {
            common_prompt_checkpoint o;
            bool io = false;
            return s.deep_get(key, o, &io) && !io ? o.data_tgt : std::vector<uint8_t>();
        };
        {
            server_disk_cache s(root.string(), 1ull << 30, 8, 30, fp, false, 0.5f, 8, 8);
            s.init();
            const std::string k = put(s, false, 7);              // pre-fix shape
            CHECK(!k.empty() && stored(s, k) == blob(false, 7));
            CHECK(put(s, false, 8) == k && stored(s, k) == blob(false, 7) && s.n_ckpt_unmarked_replaced == 0); // unmarked: reused
            // a replacement whose write fails is not counted (the counter moves with the commit): the ckpt/ dir read-only
            fs::path ckdir;
            for (const auto & de : fs::recursive_directory_iterator(root)) {
                if (de.is_directory() && de.path().filename() == "ckpt") {
                    ckdir = de.path();
                }
            }
            if (geteuid() != 0 && !ckdir.empty()) {
                fs::permissions(ckdir, fs::perms::owner_read | fs::perms::owner_exec, fs::perm_options::replace, ec);
                put(s, true, 12); // the write fails: no reference is added either
                CHECK(stored(s, k) == blob(false, 7) && s.n_ckpt_unmarked_replaced == 0);
                fs::permissions(ckdir, fs::perms::owner_all, fs::perm_options::replace, ec);
            } else {
                printf("IXY checkpoint provenance: failed-commit case skipped (root or no ckpt dir)\n");
            }
            CHECK(put(s, true, 9) == k && stored(s, k) == blob(true, 9) && s.n_ckpt_unmarked_replaced == 1); // replaced, not linked
            CHECK(put(s, true, 10) == k && stored(s, k) == blob(true, 9) && s.n_ckpt_unmarked_replaced == 1); // marked: reused
            printf("IXY checkpoint provenance: unmarked replaced once by a marked one, marked reused\n");
            // the proof for a pre-fix checkpoint: an entry referencing it with a media-free stored token stream - an orphan
            //   checkpoint (no entry) has none
            CHECK(!s.ckpt_legacy_text_ok(k, 32));
        }
        // (the startup scan's mark read is not covered here: init() collects checkpoints no entry references)
        // the strict switch: LLAMA_LEGACY_UNMARKED=0 - no pre-fix state is ever vouched for (status shows it); the env
        //   is classified exact (which stored states load, not what a loaded one computes: it does not move the key)
        {
            CHECK(server_cache_env_classify("LLAMA_LEGACY_UNMARKED") == SERVER_CACHE_ENV_EXACT);
            setenv("LLAMA_LEGACY_UNMARKED", "0", 1);
            server_disk_cache off((root / "sw0").string(), 1ull << 30, 8, 30, fp, false, 0.5f, 8, 8);
            off.init();
            unsetenv("LLAMA_LEGACY_UNMARKED");
            server_disk_cache on((root / "sw1").string(), 1ull << 30, 8, 30, fp, false, 0.5f, 8, 8);
            on.init();
            CHECK(!off.legacy_unmarked && off.status().value("legacy_unmarked", true) == false);
            CHECK(on.legacy_unmarked && on.status().value("legacy_unmarked", false) == true);
            printf("IXY legacy switch: LLAMA_LEGACY_UNMARKED=0 -> refused, unset -> accepted with proof\n");
        }
        // server_tokens::has_media: the proof's media test on a stored token stream (LLAMA_TOKEN_NULL = a media position)
        {
            llama_tokens t = toks;
            CHECK(!server_tokens(t, true).has_media());
            t[40] = LLAMA_TOKEN_NULL;
            const server_tokens st(t, true);
            CHECK(st.has_media() && st.has_media(41) && !st.has_media(40));
        }
        fs::remove_all(root, ec);
    }

    if (n_fail) {
        fprintf(stderr, "%d failure(s)\n", n_fail);
        return 1;
    }
    printf("test-cache-compat-key: OK\n");
    return 0;
}
