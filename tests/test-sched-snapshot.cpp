// [TAG_GRAPH_CACHE] ggml_backend_sched_take / _restore: two allocated graphs alternate on one scheduler through
// snapshots; every result is bit-identical to a fresh scheduler; a buffer reallocation (a larger graph) and an
// explicit ggml_backend_sched_reserve invalidate the snapshots.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

struct test_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * a   = nullptr;
    ggml_tensor  * b   = nullptr;
    ggml_tensor  * out = nullptr;
    int n = 0;

    explicit test_graph(int n) : n(n) {
        ggml_init_params params = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(params);
        a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
        b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 4);
        ggml_set_input(a);
        ggml_set_input(b);
        ggml_tensor * t = ggml_mul_mat(ctx, a, b);           // [n, 4]
        t = ggml_relu(ctx, t);
        t = ggml_add(ctx, t, ggml_scale(ctx, t, 0.5f));
        t = ggml_soft_max(ctx, t);
        out = ggml_sum_rows(ctx, ggml_mul(ctx, t, t));       // [1, 4]
        ggml_set_output(out);
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
    }
    ~test_graph() { ggml_free(ctx); }
};

static std::vector<float> rnd(size_t count, std::mt19937 & rng) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(count);
    for (auto & x : v) {
        x = d(rng);
    }
    return v;
}

static std::vector<float> run_inputs(ggml_backend_sched_t sched, test_graph & g, const std::vector<float> & va, const std::vector<float> & vb) {
    ggml_backend_tensor_set(g.a, va.data(), 0, ggml_nbytes(g.a));
    ggml_backend_tensor_set(g.b, vb.data(), 0, ggml_nbytes(g.b));
    CHECK(ggml_backend_sched_graph_compute(sched, g.gf) == GGML_STATUS_SUCCESS);
    std::vector<float> out(ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
    return out;
}

static std::vector<float> run_fresh(ggml_backend_t backend, int n, const std::vector<float> & va, const std::vector<float> & vb) {
    ggml_backend_sched_t s = ggml_backend_sched_new(&backend, nullptr, 1, 1024, false, false);
    test_graph g(n);
    CHECK(ggml_backend_sched_alloc_graph(s, g.gf));
    auto out = run_inputs(s, g, va, vb);
    ggml_backend_sched_free(s);
    return out;
}

// the resident-entry bookkeeping a cache does: the scheduler's live state belongs to at most one entry
struct snap_cache {
    ggml_backend_sched_t sched;
    ggml_backend_sched_snapshot_t snap[2] = { nullptr, nullptr };
    int resident = -1;

    bool activate(int e) {
        if (resident == e) {
            return true;
        }
        CHECK(snap[e] != nullptr);
        const bool ok = ggml_backend_sched_restore(sched, snap[e]);
        ggml_backend_sched_snapshot_t prev = snap[e];
        snap[e] = nullptr;
        if (resident >= 0) {
            snap[resident] = prev;
        } else {
            ggml_backend_sched_snapshot_free(prev);
        }
        resident = ok ? e : -1;
        return ok;
    }
};

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    CHECK(backend);
    std::mt19937 rng(42);

    const int NA = 32, NB = 48;

    // 1. a larger graph allocated after a snapshot reallocates the buffer: the snapshot is stale
    {
        ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 1024, false, false);
        test_graph ga(NA), gb(NB);
        CHECK(ggml_backend_sched_alloc_graph(sched, ga.gf));
        const uint64_t gen0 = ggml_backend_sched_buffer_gen(sched);
        ggml_backend_sched_snapshot_t sa = ggml_backend_sched_take(sched);
        CHECK(ggml_backend_sched_snapshot_valid(sched, sa));
        CHECK(ggml_backend_sched_alloc_graph(sched, gb.gf));
        CHECK(ggml_backend_sched_buffer_gen(sched) != gen0);
        CHECK(!ggml_backend_sched_snapshot_valid(sched, sa));
        ggml_backend_sched_snapshot_t sb = ggml_backend_sched_take(sched);
        CHECK(ggml_backend_sched_snapshot_valid(sched, sb));
        CHECK(!ggml_backend_sched_restore(sched, sa)); // stale: refused, the scheduler is reset
        ggml_backend_sched_snapshot_free(sa);          // now holds the (empty) previous state
        ggml_backend_sched_snapshot_free(sb);
        ggml_backend_sched_free(sched);
        printf("stale-after-grow: ok\n");
    }

    // 2. alternate two snapshot graphs; results equal a fresh scheduler; then a reserve invalidates both
    {
        ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, 1024, false, false);
        test_graph ga(NA), gb(NB), gm(NB);
        CHECK(ggml_backend_sched_reserve(sched, gm.gf)); // worst case first: no regrow below
        CHECK(ggml_backend_sched_alloc_graph(sched, ga.gf));
        snap_cache c { sched };
        c.snap[0] = ggml_backend_sched_take(sched);
        CHECK(ggml_backend_sched_alloc_graph(sched, gb.gf));
        c.snap[1] = ggml_backend_sched_take(sched);
        CHECK(ggml_backend_sched_snapshot_valid(sched, c.snap[0]));
        CHECK(ggml_backend_sched_snapshot_valid(sched, c.snap[1]));

        test_graph * gs[2] = { &ga, &gb };
        int n_cmp = 0;
        for (int round = 0; round < 20; ++round) {
            const int e = (round % 3 == 2) ? (round & 1) : (round % 2); // A B A B ... with repeats
            CHECK(c.activate(e));
            test_graph & g = *gs[e];
            const auto va = rnd((size_t) g.n*g.n, rng);
            const auto vb = rnd((size_t) g.n*4, rng);
            const auto got  = run_inputs(sched, g, va, vb);
            const auto want = run_fresh(backend, g.n, va, vb);
            CHECK(got.size() == want.size());
            CHECK(memcmp(got.data(), want.data(), got.size()*sizeof(float)) == 0);
            n_cmp++;
        }
        printf("alternate: %d rounds bit-identical to a fresh scheduler\n", n_cmp);

        // park the live state, then an explicit reserve invalidates every snapshot
        const int r = c.resident;
        c.snap[r] = ggml_backend_sched_take(sched);
        c.resident = -1;
        CHECK(ggml_backend_sched_snapshot_valid(sched, c.snap[0]) && ggml_backend_sched_snapshot_valid(sched, c.snap[1]));
        test_graph gm2(NA);
        CHECK(ggml_backend_sched_reserve(sched, gm2.gf));
        CHECK(!ggml_backend_sched_snapshot_valid(sched, c.snap[0]));
        CHECK(!ggml_backend_sched_snapshot_valid(sched, c.snap[1]));
        CHECK(!c.activate(0));
        CHECK(c.resident == -1);
        printf("reserve-invalidates: ok\n");

        ggml_backend_sched_snapshot_free(c.snap[0]);
        ggml_backend_sched_snapshot_free(c.snap[1]);
        ggml_backend_sched_free(sched);
    }

    ggml_backend_free(backend);
    printf("test-sched-snapshot: OK\n");
    return 0;
}
