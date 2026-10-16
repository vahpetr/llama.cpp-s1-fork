// [TAG_QSA_TOPK_NOKEYS] (Program v8.1 A3 = P.2b, contrib 0242) the QSA selection without the key rows, knob off vs on, the
// ordered ids bit for bit, on a GPU. Runs test-backend-ops (next to this binary) twice on the first Vulkan device:
//   off: GGML_VK_QSA_TOPK_NOKEYS=0 (the keyed v2 passes)    on: =1 + GGML_VK_QSA_TOPK_NOKEYS_CHECK=1
// over the QSA_TOP_K_NOKEYS family (decode / verify rows 1..9, the 128-query chunk and a ragged 100-row tail chunk, depths
// 65 536 / 67 583 / 262 145 cells, two streams, k 2051, fewer visible cells than k, the tie / domain modes: distinct,
// ReLU-tied quantized, dead blocks + boost, all equal, signed zeros, the normal boundary, ties across tiles), each case
// comparing the metadata form against the plane form on the backend (the plane form always keeps the keyed passes, so the
// on arm compares nokeys with keyed in one run) and against the CPU, with GGML_TEST_DUMP_IDS writing every case's ids.
// PASS: every case passes in both arms; the dumped ids of the two arms are byte-identical (ordered rows, both forms); the
// off arm dispatched no nokeys selection, the on arm did; the CHECK word stayed clear (no out-of-domain score).
// Negative control: in a -DGGML_VULKAN_QSA_TOPK_NOKEYS_NEGCTL=ON build (the nokeys emit ranks ties in reverse; the proof line
// says NEGATIVE-CONTROL BUILD) the off arm must still pass, the on arm must FAIL in the tie modes (all-equal and ties-across-
// tiles cases), and the dumps must differ; it then prints NEGATIVE CONTROL PASS.
// Not a ctest: needs a Vulkan device with the v2 selection; without one it prints SKIP and exits 0.
// usage: test-qsa-topk-nokeys [extra test-backend-ops args, e.g. -p "n_kv=65536"]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

struct arm_result {
    std::vector<std::string> lines;
    int    passed = -1, total = -1, rc = -1;
    size_t n_nokeys = 0, n_keyed = 0, n_fail = 0;
    bool   skip = false, negctl = false, check_warn = false;
    std::map<std::string, bool> fail_case;   // case vars -> FAIL
};

static arm_result run_arm(const std::string & tbo, const std::string & env, const std::string & dump, const std::string & extra) {
    arm_result r;
    fs::remove_all(dump);
    fs::create_directories(dump);
    // stderr (the dispatch log, the proof lines) to its own file: interleaved with the buffered stdout it could split a result line
    const std::string err = dump + ".stderr";
    const std::string cmd = env + " GGML_VK_DISPATCH_LOG=1 GGML_TEST_DUMP_IDS='" + dump + "' '" + tbo + "' -b Vulkan0 -o QSA_TOP_K_NOKEYS " +
                            extra + " 2>'" + err + "'";
    FILE * p = popen(cmd.c_str(), "r");
    if (!p) { return r; }
    char buf[8192];
    while (fgets(buf, sizeof(buf), p)) {
        r.lines.emplace_back(buf);
    }
    r.rc = pclose(p);
    {
        std::ifstream f(err);
        std::string l;
        while (std::getline(f, l)) { r.lines.push_back(l + "\n"); }
    }
    fs::remove(err);
    for (const std::string & l : r.lines) {
        int a, b;
        if (sscanf(l.c_str(), " %d/%d tests passed", &a, &b) == 2 && r.passed < 0) { r.passed = a; r.total = b; }
        if (l.find("[vk-dispatch] op=QSA_TOP_K path=v2_nokeys") != std::string::npos) { r.n_nokeys++; }
        if (l.find("[vk-dispatch] op=QSA_TOP_K path=v2 ") != std::string::npos) { r.n_keyed++; }
        if (l.find("NEGATIVE-CONTROL BUILD: the nokeys emit") != std::string::npos) { r.negctl = true; }
        if (l.find("qsa_topk nokeys CHECK") != std::string::npos) { r.check_warn = true; }
        if (l.find("QSA_TOP_K_NOKEYS(") != std::string::npos) {
            const size_t a0 = l.find("QSA_TOP_K_NOKEYS("), a1 = l.find(')', a0);
            const std::string key = l.substr(a0, a1 == std::string::npos ? std::string::npos : a1 - a0 + 1);
            const bool fail = l.find("FAIL") != std::string::npos;
            r.fail_case[key] = r.fail_case[key] || fail;
            r.n_fail += fail;
        }
        if (l.find("Skipping") != std::string::npos && l.find("Vulkan0") != std::string::npos) { r.skip = true; }
    }
    return r;
}

static std::map<std::string, std::string> read_dir(const std::string & d) {
    std::map<std::string, std::string> m;
    for (const auto & e : fs::directory_iterator(d)) {
        std::ifstream f(e.path(), std::ios::binary);
        m[e.path().filename().string()] = std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    return m;
}

int main(int argc, char ** argv) {
    const std::string self = argv[0];
    const std::string dir  = self.find('/') != std::string::npos ? self.substr(0, self.rfind('/')) : ".";
    const std::string tbo  = dir + "/test-backend-ops";
    if (!fs::exists(tbo)) { printf("test-qsa-topk-nokeys: test-backend-ops not found next to %s\n", self.c_str()); return 1; }
    std::string extra;
    for (int i = 1; i < argc; ++i) { extra += std::string(" '") + argv[i] + "'"; }
    const char * tmp = getenv("TMPDIR");
    const std::string base = std::string(tmp && *tmp ? tmp : "/tmp") + "/test-qsa-topk-nokeys-" + std::to_string((long long) getpid());

    const arm_result off = run_arm(tbo, "GGML_VK_QSA_TOPK_NOKEYS=0", base + "-off", extra);
    const arm_result on  = run_arm(tbo, "GGML_VK_QSA_TOPK_NOKEYS=1 GGML_VK_QSA_TOPK_NOKEYS_CHECK=1", base + "-on", extra);
    if (off.total <= 0 || on.total <= 0) {
        bool any_vk = false;
        for (const auto & l : on.lines) { any_vk = any_vk || l.find("Vulkan0") != std::string::npos; }
        if (!any_vk) { printf("test-qsa-topk-nokeys: SKIP (no Vulkan device)\n"); return 0; }
    }
    const auto d_off = read_dir(base + "-off");
    const auto d_on  = read_dir(base + "-on");
    size_t n_same = 0, n_diff = 0, n_missing = 0;
    for (const auto & [name, body] : d_off) {
        const auto it = d_on.find(name);
        if (it == d_on.end()) { n_missing++; continue; }
        if (it->second == body) { n_same++; } else { n_diff++; if (n_diff == 1) { printf("  first differing dump: %s\n", name.c_str()); } }
    }
    n_missing += d_on.size() > d_off.size() ? d_on.size() - d_off.size() : 0;
    fs::remove_all(base + "-off");
    fs::remove_all(base + "-on");

    printf("[nokeys-arm] off: %d/%d passed, FAIL lines %zu, dispatches nokeys %zu keyed %zu, rc %d\n", off.passed, off.total, off.n_fail,
           off.n_nokeys, off.n_keyed, off.rc);
    printf("[nokeys-arm] on:  %d/%d passed, FAIL lines %zu, dispatches nokeys %zu keyed %zu, rc %d, CHECK warning %d\n", on.passed,
           on.total, on.n_fail, on.n_nokeys, on.n_keyed, on.rc, (int) on.check_warn);
    printf("[nokeys-dump] %zu case files: %zu identical, %zu differ, %zu missing\n", d_off.size(), n_same, n_diff, n_missing);

    if (on.negctl) {
        // the control holds when the keyed arm is clean and the reversed tie order is caught: FAILs in the all-equal (mode 3)
        // and the ties-across-tiles (mode 6) cases, differing dumps
        size_t f3 = 0, f6 = 0, n3 = 0, n6 = 0;
        for (const auto & [k, f] : on.fail_case) {
            if (k.find("mode=3,") != std::string::npos) { n3++; f3 += f; }
            if (k.find("mode=6,") != std::string::npos) { n6++; f6 += f; }
        }
        const bool ok = off.passed == off.total && off.n_fail == 0 && n3 > 0 && f3 == n3 && n6 > 0 && f6 == n6 && n_diff > 0;
        printf("test-qsa-topk-nokeys: NEGATIVE CONTROL %s (reversed tie ranks: all-equal %zu/%zu FAIL, ties-across-tiles %zu/%zu FAIL, "
               "%zu of %zu cases FAIL in all, %zu dumps differ; keyed arm %d/%d)\n", ok ? "PASS" : "FAIL", f3, n3, f6, n6,
               (size_t) std::count_if(on.fail_case.begin(), on.fail_case.end(), [](const auto & kv) { return kv.second; }),
               on.fail_case.size(), n_diff, off.passed, off.total);
        return ok ? 0 : 1;
    }

    const bool ok = off.total > 0 && off.passed == off.total && on.passed == on.total && off.total == on.total && off.n_fail == 0 &&
                    on.n_fail == 0 && n_diff == 0 && n_missing == 0 && n_same > 0 && off.n_nokeys == 0 && on.n_nokeys > 0 &&
                    !on.check_warn;
    printf("test-qsa-topk-nokeys: %s (%d cases per arm, %zu dumps byte-identical off vs on, %zu nokeys selections in the on arm)\n",
           ok ? "PASS" : "FAIL", on.total, n_same, on.n_nokeys);
    return ok ? 0 : 1;
}
