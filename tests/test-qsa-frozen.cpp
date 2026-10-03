// [TAG_QSA_FROZEN] test-qsa-frozen (CPU): the llama-qsa-frozen core (tools/qsa-frozen/qsa-frozen-core.h) and the
// LLAMA_QSA_INDEXED_SORT sort chain.
//   1. parser: the tool's integer lists (--depths / --layers / --probe ...).
//   2. selection: the sort chain qwen4exp_qsa_sort_ids builds (cast F32 -> argsort ASC -> get_rows -> cast I32) gives each
//      row's ids sorted, exactly (ids < 2^24), on rows of the production width 2051.
//   3. a seeded synthetic capture (q f32, K/V q8_0 cache layout, permuted cells, 3 chunks): replay on the CPU backend -
//      membership PASS, the regenerated mask hash matches, S / I / Is vs the f64 oracle NMSE <= 5e-4.
//   4. negative control: one duplicated id -> membership `unique` false and the outputs are not compared.
//   5. capture round trip: a tiny graph shaped like build_attn_mha's (permuted q view of "Qcur-3", cell-major q8_0 K/V cache
//      views, f16 mask, "indexer_top_k-3" as src[5]) computed through ggml_backend_sched with the capture callback armed:
//      q / K / V / ids / kv_ids byte-identical to their sources, lengths = ne x type size, mask hash = FNV of the mask.
#include "../tools/qsa-frozen/qsa-frozen-core.h"

#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static std::string tmpdir() {
    char tmpl[] = "/tmp/test-qsa-frozen.XXXXXX";
    const char * d = mkdtemp(tmpl);
    if (!d) {
        fprintf(stderr, "mkdtemp failed\n");
        exit(1);
    }
    return d;
}

static void test_parser() {
    CHECK(qf::parse_list("65536,212992,393216") == (std::vector<int64_t>{ 65536, 212992, 393216 }), "depth list");
    CHECK(qf::parse_list("3") == (std::vector<int64_t>{ 3 }), "single item");
    for (const char * bad : { "", "3,", ",3", "3,,7", "-1", "3 7", "1e5", "abc", "0x10" }) {
        bool threw = false;
        try { qf::parse_list(bad); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "parse_list(\"%s\") must throw", bad);
    }
    printf("parser: ok\n");
}

static void test_sort_chain() {
    const int64_t W = 2051, R = 6;
    std::mt19937 rng(7);
    std::vector<int32_t> ids(W*R);
    for (int64_t r = 0; r < R; ++r) {
        std::vector<int32_t> pool;
        while ((int64_t) pool.size() < W) {
            pool.push_back((int32_t) (rng() % (1u << 22)) + (r == 0 ? (int32_t) ((1u << 24) - (1u << 22) - 1) : 0));
            std::sort(pool.begin(), pool.end());
            pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
        }
        std::shuffle(pool.begin(), pool.end(), rng);
        std::copy(pool.begin(), pool.end(), ids.begin() + r*W);
    }
    ggml_init_params ip = { (size_t) 64 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * in = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, W, R, 1, 1);
    memcpy(in->data, ids.data(), ids.size()*4);
    // the chain of qwen4exp_qsa_sort_ids (src/models/qwen4exp.cpp, [TAG_QSA_INDEXED_SORT])
    ggml_tensor * f    = ggml_reshape_2d(ctx, ggml_cast(ctx, in, GGML_TYPE_F32), W, R);
    ggml_tensor * perm = ggml_argsort(ctx, f, GGML_SORT_ORDER_ASC);
    ggml_tensor * val  = ggml_get_rows(ctx, ggml_reshape_3d(ctx, f, 1, W, R), perm);
    ggml_tensor * out  = ggml_cast(ctx, ggml_reshape_4d(ctx, val, W, R, 1, 1), GGML_TYPE_I32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);
    int64_t bad = 0;
    for (int64_t r = 0; r < R; ++r) {
        std::vector<int32_t> ref(ids.begin() + r*W, ids.begin() + (r + 1)*W);
        std::sort(ref.begin(), ref.end());
        bad += memcmp(ref.data(), (const int32_t *) out->data + r*W, W*4) != 0;
    }
    CHECK(bad == 0, "sort chain: %lld of %lld rows differ from std::sort", (long long) bad, (long long) R);
    ggml_free(ctx);
    printf("sort chain: %s (W %lld, ids up to 2^24 - 1)\n", bad == 0 ? "ok" : "FAIL", (long long) W);
}

// ---- 3 + 4: synthetic capture

struct synth {
    int64_t d = 64, n_head = 4, n_head_kv = 2, n_kv = 320, n_used = 300, n_tps = 48, W = 40;
    std::vector<std::pair<int64_t, int64_t>> chunks = { { 0, 16 }, { 16, 16 }, { 32, 16 } };
};

static void write_synth(const std::string & dir, const synth & S, uint32_t seed, int dup_row) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const ggml_type tk = GGML_TYPE_Q8_0;
    const size_t row = ggml_row_size(tk, S.d);

    std::vector<float> q(S.d*S.n_tps*S.n_head);
    for (auto & x : q) { x = nd(rng); }
    std::vector<float> kf(S.n_kv*S.n_head_kv*S.d), vf(kf.size());
    for (auto & x : kf) { x = nd(rng); }
    for (auto & x : vf) { x = nd(rng); }
    std::vector<uint8_t> kq(row*S.n_head_kv*S.n_kv), vq(kq.size());
    ggml_quantize_chunk(tk, kf.data(), kq.data(), 0, S.n_kv*S.n_head_kv, S.d, nullptr);
    ggml_quantize_chunk(tk, vf.data(), vq.data(), 0, S.n_kv*S.n_head_kv, S.d, nullptr);

    // positions 0..n_used-1 in permuted cells, the rest empty; the queries are the last n_tps positions
    std::vector<int32_t> perm(S.n_kv);
    for (int64_t j = 0; j < S.n_kv; ++j) { perm[j] = (int32_t) j; }
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> cells(S.n_kv, -1), toks(S.n_tps);
    for (int64_t p = 0; p < S.n_used; ++p) { cells[perm[p]] = (int32_t) p; }
    for (int64_t t = 0; t < S.n_tps; ++t) { toks[t] = (int32_t) (S.n_used - S.n_tps + t); }

    // ids: W distinct visible cells per row, in a random order (the radix select's order is not ascending)
    std::vector<int32_t> ids(S.W*S.n_tps);
    for (int64_t t = 0; t < S.n_tps; ++t) {
        std::vector<int32_t> vis;
        for (int64_t c = 0; c < S.n_kv; ++c) {
            if (cells[c] >= 0 && cells[c] <= toks[t]) { vis.push_back((int32_t) c); }
        }
        std::shuffle(vis.begin(), vis.end(), rng);
        std::copy(vis.begin(), vis.begin() + S.W, ids.begin() + t*S.W);
    }
    if (dup_row >= 0) {
        ids[dup_row*S.W + 1] = ids[dup_row*S.W + 0];
    }

    // q.bin per chunk in the FA's [d, nc, n_head] order
    std::vector<float> qfile;
    for (const auto & [t0, nc] : S.chunks) {
        for (int64_t h = 0; h < S.n_head; ++h) {
            for (int64_t i = 0; i < nc; ++i) {
                const float * src = q.data() + ((t0 + i)*S.n_head + h)*S.d;
                qfile.insert(qfile.end(), src, src + S.d);
            }
        }
    }
    // mask.fnv of the indexed path's mask (the causal rows of each chunk)
    std::string fnv;
    qf::json chunks = qf::json::array();
    for (const auto & [t0, nc] : S.chunks) {
        std::vector<ggml_fp16_t> m(S.n_kv*nc, qf::f16_ninf());
        for (int64_t i = 0; i < nc; ++i) {
            for (int64_t c = 0; c < S.n_kv; ++c) {
                if (cells[c] >= 0 && cells[c] <= toks[t0 + i]) { m[i*S.n_kv + c] = qf::f16_zero(); }
            }
        }
        fnv += std::to_string(t0) + " " + std::to_string(nc) + " " + qf::hex64(qf::fnv1a(m.data(), m.size()*2)) + "\n";
        chunks.push_back(qf::json { { "t0", t0 }, { "nc", nc } });
    }
    if (system(("mkdir -p '" + dir + "'").c_str()) != 0) { exit(1); }
    qf::write_file(dir + "/q.bin", qfile.data(), qfile.size()*4);
    qf::write_file(dir + "/k.bin", kq.data(), kq.size());
    qf::write_file(dir + "/v.bin", vq.data(), vq.size());
    qf::write_file(dir + "/ids.bin", ids.data(), ids.size()*4);
    qf::write_file(dir + "/cells.bin", cells.data(), cells.size()*4);
    qf::write_file(dir + "/toks.bin", toks.data(), toks.size()*4);
    qf::write_file(dir + "/mask.fnv", fnv.data(), fnv.size());
    const qf::json meta = {
        { "depth", 0 }, { "il", 3 }, { "n_kv", S.n_kv }, { "n_tps", S.n_tps }, { "width", S.W }, { "chunk", 16 }, { "chunks", chunks },
        { "d_k", S.d }, { "d_v", S.d }, { "n_head", S.n_head }, { "n_head_kv", S.n_head_kv },
        { "k", { { "type", "q8_0" } } }, { "v", { { "type", "q8_0" } } },
        { "kq_scale", 1.0f/8.0f }, { "max_bias", 0.0f }, { "softcap", 0.0f }, { "prec", (int) GGML_PREC_F32 },
        { "hadamard", false }, { "kv_ids", false }, { "qsa_prefill", "indexed" }, { "git_rev", "test" },
    };
    const std::string ms = meta.dump(1);
    qf::write_file(dir + "/meta.json", ms.data(), ms.size());
}

static void test_replay(ggml_backend_t cpu) {
    const synth S;
    const std::string dir = tmpdir() + "/cap-0-il3";
    write_synth(dir, S, 1234, -1);
    const qf::replay_result R = qf::replay(dir, cpu, 4, true, 16);
    const qf::json j = qf::replay_json(R, "CPU");
    printf("replay: %s\n", j.dump().c_str());
    CHECK(R.error.empty(), "replay error: %s", R.error.c_str());
    for (const char * k : { "unique", "in_range", "visible_ok", "sorted_same_set", "mask_fnv_match", "kv_ids_match" }) {
        CHECK(R.membership.value(k, false) == true, "membership %s not true", k);
    }
    for (const char * p : { "S-O", "I-O", "Is-O" }) {
        CHECK(R.pairs.count(p) == 1, "pair %s missing", p);
        if (R.pairs.count(p)) {
            const auto & s = R.pairs.at(p);
            const double nmse = s.den > 0 ? s.num/s.den : 1.0;
            CHECK(nmse <= 5e-4, "%s NMSE %.3e > 5e-4", p, nmse);
        }
    }
    for (const char * p : { "I-S", "Is-S", "I-Is" }) {
        CHECK(R.pairs.count(p) == 1 && R.pairs.at(p).den > 0, "pair %s missing", p);
    }

    // 4. negative control: a duplicated id must fail `unique` and stop the comparison
    const std::string dir2 = tmpdir() + "/cap-0-il3";
    write_synth(dir2, S, 1234, 5);
    const qf::replay_result N = qf::replay(dir2, cpu, 4, true, 16);
    CHECK(N.membership.value("unique", true) == false, "negative control: a duplicated id passed `unique`");
    CHECK(!N.error.empty() && N.pairs.empty(), "negative control: outputs were compared on a failed membership");
    printf("negative control (duplicated id): unique=%s, error='%s'\n", N.membership.value("unique", true) ? "true" : "false",
           N.error.c_str());
}

// ---- 5: capture round trip through the scheduler callback

static void test_capture_roundtrip(ggml_backend_t cpu) {
    const int64_t d = 64, n_head = 4, n_head_kv = 2, n_kv = 96, n_q = 12, W = 20;
    const ggml_type tk = GGML_TYPE_Q8_0;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    // q_cur [d, n_head, n_q] as the attention block builds it, viewed / permuted as build_attn_mha does
    ggml_tensor * qcur = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, n_head, n_q);
    ggml_set_name(qcur, "Qcur-3");
    ggml_tensor * kc = ggml_new_tensor_2d(ctx, tk, d*n_head_kv, n_kv);
    ggml_tensor * vc = ggml_new_tensor_2d(ctx, tk, d*n_head_kv, n_kv);
    ggml_set_name(kc, "cache_k_l3");
    ggml_set_name(vc, "cache_v_l3");
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, 1, 1);
    ggml_tensor * ids_in = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, W, n_q, 1, 1);
    for (ggml_tensor * x : { qcur, kc, vc, mask, ids_in }) { ggml_set_input(x); }

    ggml_tensor * q = ggml_permute(ctx, qcur, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(ctx, ggml_view_3d(ctx, kc, d, n_head_kv, n_kv, ggml_row_size(tk, d), kc->nb[1], 0), 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx, ggml_view_3d(ctx, vc, d, n_head_kv, n_kv, ggml_row_size(tk, d), vc->nb[1], 0), 0, 2, 1, 3);
    ggml_tensor * ids = ggml_scale(ctx, ggml_cast(ctx, ids_in, GGML_TYPE_F32), 1.0f);       // a node, as QSA_TOP_K is
    ids = ggml_cast(ctx, ids, GGML_TYPE_I32);
    ggml_set_name(ids, "indexer_top_k-3");
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_flash_attn_ext_set_kv_ids(fa, ids);
    ggml_set_output(fa);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    std::vector<float> qh(d*n_head*n_q), kf(d*n_head_kv*n_kv), vf(kf.size());
    for (auto & x : qh) { x = nd(rng); }
    for (auto & x : kf) { x = nd(rng); }
    for (auto & x : vf) { x = nd(rng); }
    std::vector<uint8_t> kq(ggml_nbytes(kc)), vq(ggml_nbytes(vc));
    ggml_quantize_chunk(tk, kf.data(), kq.data(), 0, n_kv*n_head_kv, d, nullptr);
    ggml_quantize_chunk(tk, vf.data(), vq.data(), 0, n_kv*n_head_kv, d, nullptr);
    std::vector<ggml_fp16_t> mh(n_kv*n_q);
    std::vector<int32_t> cells(n_kv), toks(n_q), idh(W*n_q);
    for (int64_t c = 0; c < n_kv; ++c) { cells[c] = (int32_t) c; }
    for (int64_t t = 0; t < n_q; ++t) {
        toks[t] = (int32_t) (n_kv - n_q + t);
        for (int64_t c = 0; c < n_kv; ++c) { mh[t*n_kv + c] = c <= toks[t] ? qf::f16_zero() : qf::f16_ninf(); }
        std::vector<int32_t> vis;
        for (int64_t c = 0; c <= toks[t]; ++c) { vis.push_back((int32_t) c); }
        std::shuffle(vis.begin(), vis.end(), rng);
        std::copy(vis.begin(), vis.begin() + W, idh.begin() + t*W);
    }
    ggml_backend_tensor_set(qcur, qh.data(), 0, qh.size()*4);
    ggml_backend_tensor_set(kc, kq.data(), 0, kq.size());
    ggml_backend_tensor_set(vc, vq.data(), 0, vq.size());
    ggml_backend_tensor_set(mask, mh.data(), 0, mh.size()*2);
    ggml_backend_tensor_set(ids_in, idh.data(), 0, idh.size()*4);

    const std::string root = tmpdir();
    qf::capture cap;
    cap.layers = { 3 };
    cap.dir = root;
    cap.qsa_prefill = "indexed";
    cap.git_rev = "test";
    cap.arm(0);

    ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, 256, false, true);
    ggml_backend_sched_set_eval_callback(sched, qf::capture_eval_cb, &cap);
    const bool ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    CHECK(ok, "round trip: graph compute failed");
    const qf::json res = cap.finish(cells, toks);
    ggml_backend_sched_free(sched);

    const std::string dir = root + "/cap-0-il3";
    const std::vector<uint8_t> qb = qf::read_file(dir + "/q.bin"), kb = qf::read_file(dir + "/k.bin"), vb = qf::read_file(dir + "/v.bin");
    const std::vector<uint8_t> ib = qf::read_file(dir + "/ids.bin"), kib = qf::read_file(dir + "/kv_ids.bin");
    // q.bin = the permuted view in its logical [d, n_q, n_head] order
    std::vector<float> qref;
    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t t = 0; t < n_q; ++t) {
            qref.insert(qref.end(), qh.begin() + (t*n_head + h)*d, qh.begin() + (t*n_head + h + 1)*d);
        }
    }
    CHECK(qb.size() == (size_t) d*n_q*n_head*4 && memcmp(qb.data(), qref.data(), qb.size()) == 0, "round trip: q.bin differs");
    CHECK(kb.size() == ggml_row_size(tk, d)*n_head_kv*n_kv && kb == kq, "round trip: k.bin differs (%zu B)", kb.size());
    CHECK(vb.size() == vq.size() && vb == vq, "round trip: v.bin differs");
    CHECK(ib.size() == idh.size()*4 && memcmp(ib.data(), idh.data(), ib.size()) == 0, "round trip: ids.bin differs");
    CHECK(kib == ib, "round trip: kv_ids.bin differs from ids.bin");
    const std::vector<uint8_t> fnv = qf::read_file(dir + "/mask.fnv");
    const std::string line(fnv.begin(), fnv.end());
    CHECK(line == "0 " + std::to_string(n_q) + " " + qf::hex64(qf::fnv1a(mh.data(), mh.size()*2)) + "\n", "round trip: mask.fnv '%s'",
          line.c_str());
    const std::vector<uint8_t> mj = qf::read_file(dir + "/meta.json");
    const qf::json meta = qf::json::parse(std::string(mj.begin(), mj.end()));
    CHECK(meta["n_kv"] == n_kv && meta["n_tps"] == n_q && meta["width"] == W && meta["n_head"] == n_head &&
          meta["n_head_kv"] == n_head_kv && meta["k"]["type"] == "q8_0" && meta["kv_ids"] == true && meta["prec"] == (int) GGML_PREC_F32,
          "round trip: meta.json %s", meta.dump().c_str());
    printf("capture round trip: %s (%s)\n", g_fail == 0 ? "ok" : "FAIL", res.dump().c_str());

    // and the replay of that real capture agrees with itself and the oracle
    const qf::replay_result R = qf::replay(dir, cpu, 4, true);
    CHECK(R.error.empty() && R.membership.value("mask_fnv_match", false) && R.membership.value("kv_ids_match", false),
          "round trip replay: %s", qf::replay_json(R, "CPU").dump().c_str());
    for (const char * p : { "S-O", "I-O", "Is-O" }) {
        const double nmse = R.pairs.count(p) && R.pairs.at(p).den > 0 ? R.pairs.at(p).num/R.pairs.at(p).den : 1.0;
        CHECK(nmse <= 5e-4, "round trip replay %s NMSE %.3e", p, nmse);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        fprintf(stderr, "no CPU backend\n");
        return 1;
    }
    test_parser();
    test_sort_chain();
    test_replay(cpu);
    test_capture_roundtrip(cpu);
    ggml_backend_free(cpu);
    printf("%s (%d failure(s))\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
