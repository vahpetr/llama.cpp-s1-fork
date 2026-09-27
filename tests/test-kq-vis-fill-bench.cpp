// [TAG_KQVIS_PROF] CPU microbench of the per-graph host fills behind LLAMA_KQ_VIS (diagnostic probe P3-D1, not a test):
//   ON : llama_kv_cells_fill_kq_vis  - the visibility metadata, 16 B per cell over all n_kv cells + 16 B per token
//        (= llama_kv_cache::set_input_kq_vis, run once per graph when LLAMA_KQ_VIS=1)
//   OFF: llama_kv_cells_fill_kq_mask - the F16 [n_kv, n_tokens] mask plane
//        (= llama_kv_cache::set_input_kq_mask, run once per graph with the knob off)
// on synthetic cells laid out like the prod pool: one unified stream of 393 216 cells (-c 393216 --kv-unified), n_used
// cells over 5 sequences in contiguous blocks (seq 0 = the brain, 60 %; seqs 1..4 = 10 % each), positions 0.. per
// sequence, causal, no SWA, no ALiBi, 1-D positions (the qwen4exp attention layers). The ubatch is the newest n_tokens
// positions of seq 0, already stored in its cells (as at set_input time). n_kv = n_used rounded up to 256 (the cache pad).
// Single thread (set_input runs on the thread that calls llama_decode). Median / min over repetitions of >= min_ms.
// [TAG_KQ_VIS_FILL] three rows per point: "mask" (OFF), "vis-before" (the pre-P3-F1 metadata fill, kq-vis-fill-ref.h:
// 32 seq_has tests per cell) and "vis-after" (llama_kv_cells_fill_kq_vis: the low-32 shadow word, one branch-free pass);
// the after records must equal the before records byte for byte.
//
// usage: test-kq-vis-fill-bench [n_used = 213248,380160] [n_tokens = 1,6,2048] [min_ms = 400]
//   (the pool grows to the largest n_used when that exceeds 393 216; a 2 048-token mask plane at 1M cells is 4 GiB)

#include "ggml.h"

#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"

#include "kq-vis-fill-ref.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<int64_t> parse_list(const char * s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        out.push_back(std::strtoll(item.c_str(), nullptr, 10));
    }
    return out;
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct timing {
    double med_us = 0.0;
    double min_us = 0.0;
    int    reps   = 0;
};

template <typename F>
timing time_it(F && f, double min_ms) {
    f(); // warm-up (page-faults the destination buffers)
    std::vector<double> t;
    const double t_end = now_us() + min_ms*1000.0;
    while (t.size() < 3 || (now_us() < t_end && t.size() < 2000)) {
        const double t0 = now_us();
        f();
        t.push_back(now_us() - t0);
    }
    std::sort(t.begin(), t.end());
    timing r;
    r.med_us = t[t.size()/2];
    r.min_us = t[0];
    r.reps   = (int) t.size();
    return r;
}

} // namespace

int main(int argc, char ** argv) {
    const std::vector<int64_t> v_used   = parse_list(argc > 1 ? argv[1] : "213248,380160");
    const std::vector<int64_t> v_tokens = parse_list(argc > 2 ? argv[2] : "1,6,2048");
    const double min_ms = argc > 3 ? std::atof(argv[3]) : 400.0;

    // the prod pool, or larger when a point asks for more used cells (e.g. 1048576 for the single-tenant 1M profile)
    const uint32_t kv_size = (uint32_t) std::max<int64_t>(393216, ((*std::max_element(v_used.begin(), v_used.end()) + 255)/256)*256);
    const int      n_seq   = 5;
    const double   share[n_seq] = { 0.6, 0.1, 0.1, 0.1, 0.1 };

    llama_hparams hp{};
    std::vector<uint32_t> seq_to_stream(LLAMA_MAX_SEQ, 0);

    printf("kqvis-fill-bench: pool %u cells, %d seqs (seq 0 = 60 %%), ubatch = newest n_tokens of seq 0, F16 mask, 1 thread, min %.0f ms/point\n",
            kv_size, n_seq, min_ms);
    printf("%8s %8s %6s | %-10s | %10s %10s %8s | %9s %9s | %9s\n",
            "n_used", "n_kv", "n_tok", "fill", "med_us", "min_us", "ns/cell", "-mask_us", "x mask", "out_KiB");

    for (const int64_t n_used : v_used) {
        GGML_ASSERT(n_used > 2048 && n_used <= (int64_t) kv_size);

        std::vector<llama_kv_cells> v_cells(1);
        auto & cells = v_cells[0];
        cells.resize(kv_size);

        // contiguous blocks: [seq 0][seq 1]...[seq 4]
        int64_t n_seq_cells[n_seq];
        int64_t acc = 0;
        for (int s = 0; s < n_seq; ++s) {
            n_seq_cells[s] = s == n_seq - 1 ? n_used - acc : (int64_t) (share[s]*n_used);
            acc += n_seq_cells[s];
        }
        uint32_t j = 0;
        for (int s = 0; s < n_seq; ++s) {
            for (int64_t p = 0; p < n_seq_cells[s]; ++p, ++j) {
                cells.pos_set(j, (llama_pos) p);
                cells.seq_add(j, s);
            }
        }
        const int64_t n_kv = std::min<int64_t>(kv_size, ((n_used + 255)/256)*256);

        for (const int64_t n_tokens : v_tokens) {
            GGML_ASSERT(n_tokens >= 1 && n_tokens <= n_seq_cells[0]);

            // the ubatch: the newest n_tokens positions of seq 0 (cells already hold them)
            std::vector<llama_pos>      pos(n_tokens);
            std::vector<int32_t>        n_seq_id(n_tokens, 1);
            std::vector<llama_seq_id>   seq_data(n_tokens, 0);
            std::vector<llama_seq_id *> seq_ptr(n_tokens);
            std::vector<int8_t>         output(n_tokens, 1);
            std::vector<int32_t>        seq_idx(LLAMA_MAX_SEQ, -1);
            std::vector<llama_seq_id>   seq_unq = { 0 };
            seq_idx[0] = 0;
            for (int64_t i = 0; i < n_tokens; ++i) {
                pos[i]     = (llama_pos) (n_seq_cells[0] - n_tokens + i);
                seq_ptr[i] = &seq_data[i];
            }

            llama_ubatch ub{};
            ub.b_equal_seqs = 0;
            ub.n_tokens     = (uint32_t) n_tokens;
            ub.n_seq_tokens = (uint32_t) n_tokens;
            ub.n_seqs       = 1;
            ub.n_seqs_unq   = 1;
            ub.n_pos        = 1;
            ub.pos          = pos.data();
            ub.n_seq_id     = n_seq_id.data();
            ub.seq_id       = seq_ptr.data();
            ub.seq_id_unq   = seq_unq.data();
            ub.seq_idx      = seq_idx.data();
            ub.output       = output.data();

            std::vector<ggml_fp16_t> plane((size_t) n_kv*n_tokens);
            std::vector<int32_t>     cells_rec((size_t) n_kv*4);
            std::vector<int32_t>     toks_rec((size_t) 4*n_tokens);
            std::vector<int32_t>     cells_ref((size_t) n_kv*4);
            std::vector<int32_t>     toks_ref((size_t) 4*n_tokens);

            const timing tm = time_it([&] {
                llama_kv_cells_fill_kq_mask(plane.data(), GGML_TYPE_F16, n_kv, 1, &ub, true, hp, 0, LLAMA_SWA_TYPE_NONE,
                        v_cells, seq_to_stream);
            }, min_ms);
            bool ok = true;
            const timing tr = time_it([&] {
                ok &= llama_kv_cells_fill_kq_vis_ref(cells_ref.data(), toks_ref.data(), n_kv, 1, &ub, v_cells, seq_to_stream);
            }, min_ms);
            const timing tv = time_it([&] {
                ok &= llama_kv_cells_fill_kq_vis(cells_rec.data(), toks_rec.data(), n_kv, 1, &ub, v_cells, seq_to_stream);
            }, min_ms);
            GGML_ASSERT(ok && "the metadata fill rejected the layout");
            GGML_ASSERT(memcmp(cells_rec.data(), cells_ref.data(), cells_rec.size()*sizeof(int32_t)) == 0 &&
                        memcmp(toks_rec.data(),  toks_ref.data(),  toks_rec.size()*sizeof(int32_t))  == 0 &&
                        "vis-after records differ from vis-before");

            // sanity: the last token sees exactly the seq 0 cells (all of them: positions <= its own)
            int64_t kept = 0;
            const ggml_fp16_t * row = plane.data() + (size_t) (n_tokens - 1)*n_kv;
            for (int64_t c = 0; c < n_kv; ++c) {
                kept += ggml_fp16_to_fp32(row[c]) == 0.0f;
            }
            GGML_ASSERT(kept == n_seq_cells[0]);

            const double out_mask = plane.size()*sizeof(ggml_fp16_t)/1024.0;
            const double out_vis  = (cells_rec.size() + toks_rec.size())*sizeof(int32_t)/1024.0;
            const struct { const char * name; const timing & t; double out; } rows[] = {
                { "mask",       tm, out_mask },
                { "vis-before", tr, out_vis  },
                { "vis-after",  tv, out_vis  },
            };
            for (const auto & r : rows) {
                printf("%8lld %8lld %6lld | %-10s | %10.1f %10.1f %8.2f | %+9.1f %9.2f | %9.1f   (reps %d)\n",
                        (long long) n_used, (long long) n_kv, (long long) n_tokens, r.name,
                        r.t.med_us, r.t.min_us, 1000.0*r.t.med_us/n_kv, r.t.med_us - tm.med_us, r.t.med_us/tm.med_us,
                        r.out, r.t.reps);
            }
            printf("%8lld %8lld %6lld | vis after/before: %.3fx (%.2f -> %.2f ns/cell), records identical\n",
                    (long long) n_used, (long long) n_kv, (long long) n_tokens, tv.med_us/tr.med_us,
                    1000.0*tr.med_us/n_kv, 1000.0*tv.med_us/n_kv);
            fflush(stdout);
        }
    }

    return 0;
}
