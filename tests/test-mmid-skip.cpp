// [TAG_MMID_SKIP] (upstream #25483, v63 63.1) the mul_mat_id padding skip on a GPU: in a MUL_MAT_ID workgroup only the first
// (_ne1 - ic*BN) B columns are rows of the expert, and mul_mm.comp skips the B loads / coopmat1 MACs past them. EXACT by
// design, so a shipped build has no knob to compare against; this driver runs test-backend-ops (next to this binary) on the
// first Vulkan device over the partial-tile MUL_MAT_ID cases - the production expert GEMM (q4_K 640 x 2560, 512 experts,
// top-10, broadcast B) at n 64 / 65 / 128 with uniform and skewed routing, where nearly every expert ends in a partial tile,
// and the repeated-id cases (dup=1: f16 / q8_0 / q4_0, n_mats 28 / 1025 / 2050) - with GGML_VK_DISABLE_MMQ_CM1=1 (the
// production env: the quantized experts take the coopmat1 mul_mm path, B1's f16-B twin, not the int8 MMQ path).
// PASS (shipped build): every case passes against the CPU, the mm_id path was dispatched, and the proof line of the
// negative-control build is absent.
// Negative control: in a -DGGML_VULKAN_MMID_SKIP_NEGCTL=ON build (the skip counts one row too few: the last real row of each
// expert's last partial tile is never loaded; the proof line says NEGATIVE-CONTROL BUILD) the production cases and the
// repeated-id cases (when present) must each FAIL at least once; it then prints NEGATIVE CONTROL PASS.
// Not a ctest: needs a Vulkan device; without one it prints SKIP and exits 0.
// usage: test-mmid-skip [-p <test-backend-ops params regex>]   (default: the partial-tile cases above)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

static const char * k_default_filter =
    "(type_a=q4_K,type_b=f32,n_mats=512,n_used=10,b=1,m=640,n=(64|65|128),k=2560|dup=1)";

struct arm_result {
    std::vector<std::string> lines;
    int    passed = -1, total = -1, rc = -1;
    size_t n_ok = 0, n_fail = 0, n_fail_prod = 0, n_fail_dup = 0, n_cases_prod = 0, n_cases_dup = 0, n_mm_id = 0;
    bool   skip = false, negctl = false;
};

static arm_result run_arm(const std::string & tbo, const std::string & filter, const std::string & tmp) {
    arm_result r;
    const std::string err = tmp + ".stderr";
    const std::string cmd = "GGML_VK_DISABLE_MMQ_CM1=1 GGML_VK_DISPATCH_LOG=1 '" + tbo + "' -b Vulkan0 -o MUL_MAT_ID -p '" + filter +
                            "' 2>'" + err + "'";
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
        if (l.find("NEGATIVE-CONTROL BUILD: the mul_mat_id padding skip") != std::string::npos) { r.negctl = true; }
        if (l.find("[vk-dispatch] op=MUL_MAT_ID path=mm_id") != std::string::npos) { r.n_mm_id++; }
        if (l.find("Skipping") != std::string::npos && l.find("Vulkan") != std::string::npos) { r.skip = true; }
        int a = 0, b = 0;
        // "N/M tests passed" (per backend); not "N/M backends passed", which sscanf would also match up to the literal
        if (l.find("tests passed") != std::string::npos && sscanf(l.c_str(), " %d/%d", &a, &b) == 2) { r.passed = a; r.total = b; }
        const size_t at = l.find("MUL_MAT_ID(");
        if (at == std::string::npos || l.find("):") == std::string::npos) { continue; }
        const bool dup  = l.find("dup=1") != std::string::npos;
        const bool ok   = l.find("OK") != std::string::npos;
        const bool fail = l.find("FAIL") != std::string::npos;
        if (!ok && !fail) { continue; }
        (dup ? r.n_cases_dup : r.n_cases_prod)++;
        if (ok)   { r.n_ok++; }
        if (fail) { r.n_fail++; (dup ? r.n_fail_dup : r.n_fail_prod)++; }
    }
    return r;
}

int main(int argc, char ** argv) {
    std::string filter = k_default_filter;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            filter = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [-p <test-backend-ops params regex>]\n", argv[0]);
            return 2;
        }
    }

    const fs::path self = fs::canonical("/proc/self/exe");
    const std::string tbo = (self.parent_path() / "test-backend-ops").string();
    if (!fs::exists(tbo)) {
        fprintf(stderr, "test-mmid-skip: %s not found\n", tbo.c_str());
        return 1;
    }
    const std::string tmp = (fs::temp_directory_path() / ("test-mmid-skip-" + std::to_string(getpid()))).string();

    const arm_result r = run_arm(tbo, filter, tmp);
    for (const std::string & l : r.lines) {
        if (l.find("MUL_MAT_ID(") != std::string::npos && l.find("FAIL") != std::string::npos) { fputs(l.c_str(), stdout); }
    }
    if (r.skip || r.total == 0) {
        printf("test-mmid-skip: SKIP (no Vulkan device or no case ran)\n");
        return 0;
    }

    printf("test-mmid-skip: %d/%d cases passed (production expert GEMM %zu cases, %zu FAIL; repeated ids %zu cases, %zu FAIL); "
           "mm_id dispatches %zu; test-backend-ops rc %d%s\n",
           r.passed, r.total, r.n_cases_prod, r.n_fail_prod, r.n_cases_dup, r.n_fail_dup, r.n_mm_id, r.rc,
           r.negctl ? "; NEGATIVE-CONTROL BUILD" : "");

    if (r.negctl) {
        // the repeated-id cases arrive with #29998 (0254); without them only the production family is required
        const bool pass = r.n_fail_prod > 0 && (r.n_cases_dup == 0 || r.n_fail_dup > 0);
        printf("test-mmid-skip: NEGATIVE CONTROL %s (build with the padding skip one row short: %zu/%zu production cases and "
               "%zu/%zu repeated-id cases FAIL, each family present must fail at least once)\n",
               pass ? "PASS" : "FAIL", r.n_fail_prod, r.n_cases_prod, r.n_fail_dup, r.n_cases_dup);
        return pass ? 0 : 1;
    }

    const bool pass = r.rc == 0 && r.total > 0 && r.passed == r.total && r.n_fail == 0 && r.n_mm_id > 0 &&
                      r.n_cases_prod > 0;
    printf("test-mmid-skip: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
