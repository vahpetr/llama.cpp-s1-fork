// [TAG_PLE_DIRECT] see llama-ple-direct.h
#include "llama-ple-direct.h"

#include "llama-impl.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <cerrno>
#define LLAMA_PLE_DIRECT_SUPPORTED 1
#else
#define LLAMA_PLE_DIRECT_SUPPORTED 0
#endif

// default budget of the exact LRU row cache: 1 GiB of row data = 9 761 289 rows of 110 B (+ ~18 % index)
static constexpr size_t LLAMA_PLE_DIRECT_DEFAULT_CACHE_MIB = 1024;
// two reads closer than this are merged into one (bytes between them are read and discarded)
static constexpr size_t LLAMA_PLE_DIRECT_MERGE_GAP = 4096;
// ... up to this many bytes per merged read
static constexpr size_t LLAMA_PLE_DIRECT_MAX_READ  = 128*1024;

static int ple_env_int(const char * name, int def) {
    const char * s = getenv(name);
    if (s == nullptr || s[0] == '\0') {
        return def;
    }
    return atoi(s);
}

llama_ple_direct_params llama_ple_direct_params::from_env() {
    llama_ple_direct_params p;
    p.mode = ple_env_int("LLAMA_PLE_DIRECT", 0);
    if (p.mode != 1 && p.mode != 2) {
        p.mode = 0;
        return p;
    }
    const int mib = ple_env_int("LLAMA_PLE_DIRECT_CACHE_MIB", (int) LLAMA_PLE_DIRECT_DEFAULT_CACHE_MIB);
    p.cache_bytes = mib > 0 ? (size_t) mib * 1024 * 1024 : 0;
    p.n_threads   = std::max(1, std::min(256, ple_env_int("LLAMA_PLE_DIRECT_THREADS", 16)));
    p.nowait      = ple_env_int("LLAMA_PLE_DIRECT_NOWAIT", 1) != 0;
    p.mlock_cache = ple_env_int("LLAMA_PLE_DIRECT_MLOCK", 0) != 0;
    p.stats_every = std::max(0, ple_env_int("LLAMA_PLE_DIRECT_STATS", 0));
    return p;
}

#if LLAMA_PLE_DIRECT_SUPPORTED

// anonymous zero-filled memory, committed page by page as it is written (so the cache's RSS follows its fill level)
static void * ple_anon_alloc(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}

static void ple_anon_free(void * p, size_t bytes) {
    if (p != nullptr && bytes > 0) {
        munmap(p, bytes);
    }
}

// exact LRU over row ids: slot arrays + an open-addressing hash (linear probing, backward-shift deletion, entry = slot+1
// so the zero pages of a fresh mapping read as "empty") + a doubly linked recency list (head = most recent)
struct ple_row_cache {
    size_t    rs    = 0;
    int64_t   cap   = 0;
    int64_t   size  = 0;
    int32_t   head  = -1;
    int32_t   tail  = -1;
    uint64_t  mask  = 0;
    int       shift = 64;

    uint8_t * data  = nullptr; size_t data_bytes = 0;
    int32_t * key   = nullptr;
    int32_t * prv   = nullptr;
    int32_t * nxt   = nullptr; size_t slot_bytes = 0;  // key, prv, nxt: one allocation
    int32_t * tab   = nullptr; size_t tab_bytes  = 0;

    bool init(size_t row_size, size_t budget, bool lock, std::string & note) {
        rs  = row_size;
        cap = (int64_t) (budget / row_size);
        if (cap <= 0) {
            cap = 0;
            return true;
        }
        cap = std::min<int64_t>(cap, INT32_MAX / 2);
        uint64_t n_tab = 1;
        int      bits  = 0;
        while (n_tab < (uint64_t) cap + (uint64_t) cap / 3 + 1) { // load factor <= 3/4
            n_tab <<= 1;
            bits++;
        }
        mask  = n_tab - 1;
        shift = 64 - std::max(bits, 1);

        data_bytes = (size_t) cap * rs;
        slot_bytes = (size_t) cap * 3 * sizeof(int32_t);
        tab_bytes  = (size_t) n_tab * sizeof(int32_t);
        data = (uint8_t *) ple_anon_alloc(data_bytes);
        auto * s = (int32_t *) ple_anon_alloc(slot_bytes);
        tab  = (int32_t *) ple_anon_alloc(tab_bytes);
        if (data == nullptr || s == nullptr || tab == nullptr) {
            ple_anon_free(s, slot_bytes); slot_bytes = 0;
            release();
            cap = 0;
            note = "row cache allocation failed, running without a cache";
            return false;
        }
        key = s;
        prv = s + cap;
        nxt = s + 2*cap;
        if (lock) {
            if (mlock(data, data_bytes) != 0 || mlock(s, slot_bytes) != 0 || mlock(tab, tab_bytes) != 0) {
                note = std::string("mlock of the row cache failed (") + strerror(errno) + "), cache stays swappable";
            } else {
                note = "row cache mlocked";
            }
        }
        return true;
    }

    void release() {
        ple_anon_free(data, data_bytes); data = nullptr; data_bytes = 0;
        ple_anon_free(key,  slot_bytes); key  = prv = nxt = nullptr; slot_bytes = 0;
        ple_anon_free(tab,  tab_bytes);  tab  = nullptr; tab_bytes  = 0;
        size = 0; head = tail = -1;
    }

    ~ple_row_cache() { release(); }

    uint64_t home(int32_t id) const {
        return ((uint64_t) (uint32_t) id * 0x9E3779B97F4A7C15ull) >> shift & mask;
    }

    int32_t find(int32_t id) const {
        for (uint64_t h = home(id);; h = (h + 1) & mask) {
            const int32_t e = tab[h];
            if (e == 0) {
                return -1;
            }
            if (key[e - 1] == id) {
                return e - 1;
            }
        }
    }

    void unlink(int32_t s) {
        if (prv[s] >= 0) { nxt[prv[s]] = nxt[s]; } else { head = nxt[s]; }
        if (nxt[s] >= 0) { prv[nxt[s]] = prv[s]; } else { tail = prv[s]; }
    }

    void push_front(int32_t s) {
        prv[s] = -1;
        nxt[s] = head;
        if (head >= 0) { prv[head] = s; }
        head = s;
        if (tail < 0) { tail = s; }
    }

    void touch(int32_t s) {
        if (s != head) {
            unlink(s);
            push_front(s);
        }
    }

    void tab_erase(int32_t id) {
        uint64_t i = home(id);
        while (tab[i] != 0 && key[tab[i] - 1] != id) {
            i = (i + 1) & mask;
        }
        GGML_ASSERT(tab[i] != 0 && "ple-direct: evicted row not in the cache index");
        // backward-shift deletion: move every later entry of the probe run whose home is not in (i, j] into the hole
        uint64_t j = i;
        for (;;) {
            j = (j + 1) & mask;
            if (tab[j] == 0) {
                break;
            }
            const uint64_t k = home(key[tab[j] - 1]);
            const bool stays = (i <= j) ? (i < k && k <= j) : (i < k || k <= j);
            if (!stays) {
                tab[i] = tab[j];
                i = j;
            }
        }
        tab[i] = 0;
    }

    // returns true when a row was evicted to make room
    bool insert(int32_t id, const uint8_t * row) {
        int32_t s;
        bool    evicted = false;
        if (size < cap) {
            s = (int32_t) size++;
        } else {
            s = tail;
            tab_erase(key[s]);
            unlink(s);
            evicted = true;
        }
        key[s] = id;
        memcpy(data + (size_t) s * rs, row, rs);
        uint64_t h = home(id);
        while (tab[h] != 0) {
            h = (h + 1) & mask;
        }
        tab[h] = s + 1;
        push_front(s);
        return evicted;
    }

    const uint8_t * row(int32_t s) const { return data + (size_t) s * rs; }

    std::string check() const {
        char buf[256];
        // the list visits exactly `size` slots, both directions agree
        int64_t n = 0;
        int32_t last = -1;
        for (int32_t s = head; s >= 0; s = nxt[s]) {
            if (prv[s] != last) { snprintf(buf, sizeof(buf), "list: prv[%d]=%d, expected %d", s, prv[s], last); return buf; }
            last = s;
            if (++n > size) { return "list: longer than size (cycle?)"; }
        }
        if (n != size || last != tail) { snprintf(buf, sizeof(buf), "list: %" PRId64 " slots, size %" PRId64, n, size); return buf; }
        // every slot is findable, and the index holds exactly `size` entries
        for (int64_t s = 0; s < size; ++s) {
            if (find(key[s]) != (int32_t) s) { snprintf(buf, sizeof(buf), "index: slot %" PRId64 " (row %d) not found", s, key[s]); return buf; }
        }
        int64_t e = 0;
        for (uint64_t h = 0; h <= mask; ++h) { e += tab[h] != 0; }
        if (e != size) { snprintf(buf, sizeof(buf), "index: %" PRId64 " entries, size %" PRId64, e, size); return buf; }
        return "";
    }
};

// fixed worker pool: run(n, f) calls f(0..n-1) on the workers and the caller, returns when all are done and every
// worker is idle again (so f and the job state are never touched after run() returns)
struct ple_pool {
    std::vector<std::thread> threads;
    std::mutex               m;
    std::condition_variable  cv_work;
    std::condition_variable  cv_done;
    uint64_t                 gen    = 0;
    bool                     stop   = false;
    int                      active = 0;
    const std::function<void(int64_t)> * job = nullptr;
    int64_t                  n_jobs = 0;
    std::atomic<int64_t>     next{0};

    explicit ple_pool(int n_threads) {
        for (int i = 1; i < n_threads; ++i) {
            threads.emplace_back([this] { worker(); });
        }
    }

    ~ple_pool() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv_work.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }

    void drain(const std::function<void(int64_t)> & f, int64_t n) {
        for (int64_t i = next.fetch_add(1); i < n; i = next.fetch_add(1)) {
            f(i);
        }
    }

    void worker() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int64_t)> * f;
            int64_t n;
            {
                std::unique_lock<std::mutex> lk(m);
                cv_work.wait(lk, [&] { return stop || gen != seen; });
                if (stop) {
                    return;
                }
                seen = gen;
                if (job == nullptr) {
                    continue; // woke after that run() had already finished
                }
                f = job;
                n = n_jobs;
                active++;
            }
            drain(*f, n);
            {
                std::lock_guard<std::mutex> lk(m);
                active--;
            }
            cv_done.notify_one();
        }
    }

    void run(int64_t n, const std::function<void(int64_t)> & f) {
        if (n <= 0) {
            return;
        }
        if (n == 1 || threads.empty()) {
            for (int64_t i = 0; i < n; ++i) {
                f(i);
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m);
            job    = &f;
            n_jobs = n;
            next.store(0);
            gen++;
        }
        if (n - 1 >= (int64_t) threads.size()) {
            cv_work.notify_all();
        } else {
            for (int64_t i = 0; i < n - 1; ++i) {
                cv_work.notify_one();
            }
        }
        drain(f, n);
        std::unique_lock<std::mutex> lk(m);
        // every index is claimed once the caller's drain returns; each claimed one is finished once no worker is active.
        // A worker that has not picked the job up yet finds job == nullptr below and skips it (all under m).
        cv_done.wait(lk, [&] { return active == 0; });
        job    = nullptr;
        n_jobs = 0;
    }
};

struct llama_ple_direct_reader::impl {
    llama_ple_direct_params params;

    int     fd_dio   = -1;
    int     fd_buf   = -1;
    int     mode     = 2;       // effective: 1 = O_DIRECT, 2 = buffered
    size_t  align    = 1;       // O_DIRECT offset/length/buffer alignment
    size_t  offs     = 0;
    size_t  rs       = 0;
    int64_t n_rows   = 0;
    size_t  fsize    = 0;
    size_t  page     = 4096;
    std::string notes;

    ple_row_cache cache;
    std::unique_ptr<ple_pool> pool;

    mutable std::mutex mtx;
    llama_ple_direct_stats st;
    bool warned_fallback = false;

    // per-gather scratch, reused
    std::vector<uint64_t> pairs;   // (id << 32) | position
    std::vector<int64_t>  miss;    // index into pairs of the first position of each missing distinct id
    std::vector<uint8_t>  mrows;   // the missing rows, row_size each
    std::vector<uint8_t>  served;  // per miss: 1 = served by the page cache probe
    struct req { size_t a0, a1, soff; int64_t m0, m1; bool ok; int err; };
    std::vector<req>      reqs;
    uint8_t * scratch = nullptr; size_t scratch_bytes = 0;

    ~impl() {
        pool.reset();
        if (fd_dio >= 0) { close(fd_dio); }
        if (fd_buf >= 0) { close(fd_buf); }
        free(scratch);
    }

    size_t row_off(int32_t id) const { return offs + (size_t) id * rs; }

    uint8_t * scratch_reserve(size_t bytes) {
        if (bytes > scratch_bytes) {
            free(scratch);
            scratch_bytes = std::max(bytes, scratch_bytes * 2);
            scratch_bytes = (scratch_bytes + 4095) & ~(size_t) 4095;
            if (posix_memalign((void **) &scratch, std::max<size_t>(align, 4096), scratch_bytes) != 0) {
                scratch = nullptr;
                scratch_bytes = 0;
                GGML_ABORT("ple-direct: cannot allocate %zu bytes of read scratch", bytes);
            }
        }
        return scratch;
    }

    // read [a0, a1) into dst; true when at least `need` bytes arrived (a short read at EOF is fine); *err = errno
    // of a failed read (0 for a short read)
    bool read_range(int fd, size_t a0, size_t a1, size_t need, uint8_t * dst, int * err) {
        *err = 0;
        if (params.fail_reads_for_test) {
            *err = EIO;
            return false;
        }
        size_t got = 0;
        while (a0 + got < a1) {
            const ssize_t r = pread(fd, dst + got, a1 - a0 - got, (off_t) (a0 + got));
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                *err = errno;
                return false;
            }
            if (r == 0) {
                break; // EOF
            }
            got += (size_t) r;
        }
        return got >= need;
    }
};

llama_ple_direct_reader::llama_ple_direct_reader(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

llama_ple_direct_reader::~llama_ple_direct_reader() {
    if (pimpl && pimpl->st.gathers > 0) {
        LLAMA_LOG_INFO("%s\n", stats_line().c_str());
    }
}

std::unique_ptr<llama_ple_direct_reader> llama_ple_direct_reader::open(int fd_src, size_t file_offs, size_t row_size,
        int64_t n_rows, const llama_ple_direct_params & params, std::string * err) {
    auto fail = [&](const std::string & why) -> std::unique_ptr<llama_ple_direct_reader> {
        if (err) { *err = why; }
        return nullptr;
    };
    if (params.mode != 1 && params.mode != 2) {
        return fail("mode is off");
    }
    if (row_size == 0 || n_rows <= 0 || n_rows > INT32_MAX) {
        return fail("bad table geometry");
    }

    auto p = std::make_unique<impl>();
    p->params = params;
    p->offs   = file_offs;
    p->rs     = row_size;
    p->n_rows = n_rows;

    struct stat s_src, s_buf;
    if (fstat(fd_src, &s_src) != 0) {
        return fail(std::string("fstat: ") + strerror(errno));
    }
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd_src);

    // buffered descriptor: mode 2 (and the same-file check); no readahead on it
    p->fd_buf = ::open(path, O_RDONLY | O_CLOEXEC);
    if (p->fd_buf < 0) {
        return fail(std::string("re-open ") + path + ": " + strerror(errno));
    }
    if (fstat(p->fd_buf, &s_buf) != 0 || s_buf.st_dev != s_src.st_dev || s_buf.st_ino != s_src.st_ino || s_buf.st_size != s_src.st_size) {
        return fail("re-opened descriptor is not the model file");
    }
    p->fsize = (size_t) s_src.st_size;
    p->page  = (size_t) std::max(4096L, sysconf(_SC_PAGESIZE));
    if (file_offs + (size_t) n_rows * row_size > p->fsize) {
        return fail("table extends past the end of the file");
    }
    posix_fadvise(p->fd_buf, 0, 0, POSIX_FADV_RANDOM);

    p->mode = params.mode;
    if (p->mode == 1) {
        p->fd_dio = ::open(path, O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (p->fd_dio < 0) {
            p->notes += std::string("O_DIRECT refused (") + strerror(errno) + "), buffered reads instead; ";
            p->mode = 2;
        } else {
            size_t a = 0;
#if defined(STATX_DIOALIGN)
            struct statx stx;
            if (statx(p->fd_dio, "", AT_EMPTY_PATH, STATX_DIOALIGN, &stx) == 0 && (stx.stx_mask & STATX_DIOALIGN) &&
                stx.stx_dio_offset_align > 0) {
                a = std::max<size_t>(stx.stx_dio_offset_align, stx.stx_dio_mem_align);
            }
#endif
            if (a == 0) {
                a = 4096;
            }
            // probe: one aligned block at the table start must read
            bool ok = false;
            for (size_t try_a : { a, (size_t) 4096 }) {
                void * buf = nullptr;
                if (posix_memalign(&buf, std::max<size_t>(try_a, 4096), try_a) != 0) {
                    break;
                }
                const size_t a0 = file_offs & ~(try_a - 1);
                const ssize_t r = pread(p->fd_dio, buf, try_a, (off_t) a0);
                free(buf);
                if (r > 0) {
                    a  = try_a;
                    ok = true;
                    break;
                }
            }
            if (!ok) {
                p->notes += std::string("O_DIRECT probe read failed (") + strerror(errno) + "), buffered reads instead; ";
                close(p->fd_dio);
                p->fd_dio = -1;
                p->mode = 2;
            } else {
                p->align = a;
            }
        }
    }

    std::string note;
    if (params.cache_bytes > 0) {
        p->cache.init(row_size, params.cache_bytes, params.mlock_cache, note);
        if (!note.empty()) {
            p->notes += note + "; ";
        }
    }
    p->pool = std::make_unique<ple_pool>(std::max(1, params.n_threads));

    return std::unique_ptr<llama_ple_direct_reader>(new llama_ple_direct_reader(std::move(p)));
}

void llama_ple_direct_reader::gather(const int32_t * ids, int64_t n, uint8_t * dst, const uint8_t * mmap_rows) {
    if (n <= 0) {
        return;
    }
    impl & I = *pimpl;
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(I.mtx);

    const size_t rs = I.rs;

    // (id, position) sorted by id: distinct ids become runs, the reads go out in file order
    I.pairs.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t id = ids[i];
        GGML_ASSERT(id >= 0 && id < I.n_rows && "ple-direct: row id out of range");
        I.pairs[i] = ((uint64_t) (uint32_t) id << 32) | (uint64_t) i;
    }
    std::sort(I.pairs.begin(), I.pairs.end());

    auto id_at  = [&](int64_t k) { return (int32_t) (I.pairs[k] >> 32); };
    auto pos_at = [&](int64_t k) { return (int64_t) (I.pairs[k] & 0xffffffffull); };
    auto run_end = [&](int64_t k) { const uint64_t id = I.pairs[k] >> 32; int64_t e = k + 1; while (e < n && (I.pairs[e] >> 32) == id) { ++e; } return e; };
    // write one row to every position of the run that starts at k
    auto emit = [&](int64_t k, const uint8_t * src) {
        const int64_t e = run_end(k);
        for (int64_t j = k; j < e; ++j) {
            memcpy(dst + (size_t) pos_at(j) * rs, src, rs);
        }
        return e;
    };

    I.miss.clear();
    uint64_t n_unique = 0, n_hit = 0;
    for (int64_t k = 0; k < n; ) {
        n_unique++;
        const int32_t s = I.cache.cap > 0 ? I.cache.find(id_at(k)) : -1;
        if (s >= 0) {
            I.cache.touch(s);
            n_hit++;
            k = emit(k, I.cache.row(s));
        } else {
            I.miss.push_back(k);
            k = run_end(k);
        }
    }

    const int64_t nm = (int64_t) I.miss.size();
    uint64_t n_pc = 0, n_dev = 0, n_reads = 0, n_bytes = 0, n_fb = 0, n_ev = 0;
    int      fb_err = 0;
    if (nm > 0) {
        I.mrows.resize((size_t) nm * rs);
        I.served.assign(nm, 0);

        // page cache first (mode 1 + nowait, needs the mapping): mincore() on the page(s) of the row -- a pure lookup that
        // can never start I/O -- and a copy through the mapping when they are all resident. (preadv2(RWF_NOWAIT) looked
        // like the obvious probe but is not one: on a miss the kernel still starts page_cache_sync_readahead() before it
        // returns EAGAIN, so every probe of a cold row read its 4 KiB page into the page cache on top of the O_DIRECT read
        // -- measured 2026-09-30: 2.0 GiB of storage reads for 0.26 GiB of O_DIRECT data.) A page evicted between the
        // mincore and the copy costs a major fault, not a wrong byte.
        if (I.mode == 1 && I.params.nowait && mmap_rows != nullptr && !I.params.fail_reads_for_test) {
            const uintptr_t ps = (uintptr_t) I.page;
            for (int64_t m = 0; m < nm; ++m) {
                const int32_t   id  = id_at(I.miss[m]);
                const uint8_t * src = mmap_rows + (size_t) id * rs;
                const uintptr_t b   = (uintptr_t) src & ~(ps - 1);
                const uintptr_t e   = ((uintptr_t) src + rs + ps - 1) & ~(ps - 1);
                unsigned char vec[4] = { 0, 0, 0, 0 };
                if (e - b <= 4 * ps && mincore((void *) b, e - b, vec) == 0) {
                    bool all = true;
                    for (uintptr_t k = 0; k < (e - b) / ps; ++k) {
                        all = all && (vec[k] & 1);
                    }
                    if (all) {
                        memcpy(I.mrows.data() + (size_t) m * rs, src, rs);
                        I.served[m] = 1;
                        n_pc++;
                    }
                }
            }
        }
        // the rest: merged, aligned (O_DIRECT) reads, in file order, spread over the pool
        const size_t A   = I.mode == 1 ? I.align : 1;
        const int    fd  = I.mode == 1 ? I.fd_dio : I.fd_buf;
        I.reqs.clear();
        size_t soff = 0;
        for (int64_t m = 0; m < nm; ++m) {
            if (I.served[m]) {
                continue;
            }
            n_dev++;
            const size_t o  = I.row_off(id_at(I.miss[m]));
            const size_t a0 = o & ~(A - 1);
            const size_t a1 = (o + rs + A - 1) & ~(A - 1);
            if (!I.reqs.empty()) {
                auto & r = I.reqs.back();
                if (a0 <= r.a1 + LLAMA_PLE_DIRECT_MERGE_GAP && std::max(a1, r.a1) - r.a0 <= LLAMA_PLE_DIRECT_MAX_READ) {
                    soff  += std::max(a1, r.a1) - r.a1;
                    r.a1   = std::max(a1, r.a1);
                    r.m1   = m + 1;
                    continue;
                }
            }
            soff = (soff + std::max<size_t>(A, 64) - 1) & ~(std::max<size_t>(A, 64) - 1);
            I.reqs.push_back({ a0, a1, soff, m, m + 1, false, 0 });
            soff += a1 - a0;
        }
        if (!I.reqs.empty()) {
            uint8_t * sc = I.scratch_reserve(soff);
            const std::function<void(int64_t)> job = [&](int64_t q) {
                auto & r = I.reqs[q];
                // the bytes this request must deliver: up to the end of its last row
                const size_t need = I.row_off(id_at(I.miss[r.m1 - 1])) + rs - r.a0;
                r.ok = I.read_range(fd, r.a0, r.a1, need, sc + r.soff, &r.err);
            };
            I.pool->run((int64_t) I.reqs.size(), job);
            for (const auto & r : I.reqs) {
                n_reads++;
                n_bytes += r.a1 - r.a0;
                for (int64_t m = r.m0; m < r.m1; ++m) {
                    if (I.served[m]) {
                        continue;
                    }
                    const int32_t id = id_at(I.miss[m]);
                    uint8_t * out = I.mrows.data() + (size_t) m * rs;
                    if (r.ok) {
                        memcpy(out, sc + r.soff + (I.row_off(id) - r.a0), rs);
                    } else {
                        if (mmap_rows == nullptr) {
                            GGML_ABORT("ple-direct: read of row %d failed and there is no mapping to fall back to", id);
                        }
                        memcpy(out, mmap_rows + (size_t) id * rs, rs);
                        n_fb++;
                        fb_err = r.err;
                    }
                }
            }
        }
        for (int64_t m = 0; m < nm; ++m) {
            const uint8_t * src = I.mrows.data() + (size_t) m * rs;
            emit(I.miss[m], src);
            if (I.cache.cap > 0) {
                n_ev += I.cache.insert(id_at(I.miss[m]), src) ? 1 : 0;
            }
        }
    }

    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    auto & st = I.st;
    st.gathers++;
    st.ids           += (uint64_t) n;
    st.unique        += n_unique;
    st.cache_hits    += n_hit;
    st.pc_hits       += n_pc;
    st.dev_rows      += n_dev;
    st.dev_reads     += n_reads;
    st.dev_bytes     += n_bytes;
    st.fallback_rows += n_fb;
    st.evictions     += n_ev;
    st.us_total      += us;
    st.us_max         = std::max(st.us_max, us);

    if (n_fb > 0 && !I.warned_fallback) {
        I.warned_fallback = true;
        LLAMA_LOG_WARN("ple-direct: a read failed (%s) -- %" PRIu64 " rows of this gather copied from the mapping instead; "
                       "later failures are only counted (fallback in the stats line)\n", fb_err ? strerror(fb_err) : "short read", n_fb);
    }
    if (I.params.stats_every > 0 && st.gathers % (uint64_t) I.params.stats_every == 0) {
        // the lock is held: format from the counters directly
        LLAMA_LOG_INFO("%s\n", [&] {
            char b[512];
            snprintf(b, sizeof(b), "ple-direct stats: gathers %" PRIu64 " ids %" PRIu64 " unique %" PRIu64 " row-cache hits %" PRIu64
                     " page-cache hits %" PRIu64 " device rows %" PRIu64 " reads %" PRIu64 " bytes %" PRIu64 " fallback %" PRIu64
                     " evictions %" PRIu64 " cached %" PRId64 " | %.1f us/gather mean, %.1f max",
                     st.gathers, st.ids, st.unique, st.cache_hits, st.pc_hits, st.dev_rows, st.dev_reads, st.dev_bytes,
                     st.fallback_rows, st.evictions, I.cache.size, st.us_total / (double) st.gathers, st.us_max);
            return std::string(b);
        }().c_str());
    }
}

llama_ple_direct_stats llama_ple_direct_reader::stats() const {
    std::lock_guard<std::mutex> lk(pimpl->mtx);
    return pimpl->st;
}

std::string llama_ple_direct_reader::describe() const {
    const impl & I = *pimpl;
    char b[768];
    snprintf(b, sizeof(b), "mode %d (%s), cache %zu MiB = %" PRId64 " rows (+%zu MiB index)%s, %d threads, page-cache probe %s, "
             "table %" PRId64 " rows x %zu B at file offset %zu%s%s",
             I.mode, I.mode == 1 ? (std::string("O_DIRECT, align ") + std::to_string(I.align)).c_str() : "buffered pread, POSIX_FADV_RANDOM",
             I.cache.data_bytes >> 20, I.cache.cap, (I.cache.slot_bytes + I.cache.tab_bytes) >> 20,
             I.params.mlock_cache ? " mlock requested" : "", I.params.n_threads,
             (I.mode == 1 && I.params.nowait) ? "on (mincore on the mapping)" : "off",
             I.n_rows, I.rs, I.offs, I.notes.empty() ? "" : "; ", I.notes.c_str());
    return b;
}

std::string llama_ple_direct_reader::stats_line() const {
    const auto st = stats();
    const double u = st.unique > 0 ? 100.0 / (double) st.unique : 0.0;
    char b[640];
    snprintf(b, sizeof(b), "ple-direct stats: gathers %" PRIu64 ", ids %" PRIu64 ", distinct %" PRIu64 " = row cache %.1f %% + page cache %.1f %% + "
             "device %.1f %% (%" PRIu64 " reads, %.1f MiB), fallback %" PRIu64 ", evictions %" PRIu64 ", %.1f us/gather mean, %.1f max",
             st.gathers, st.ids, st.unique, st.cache_hits * u, st.pc_hits * u, st.dev_rows * u, st.dev_reads,
             st.dev_bytes / 1048576.0, st.fallback_rows, st.evictions, st.gathers ? st.us_total / (double) st.gathers : 0.0, st.us_max);
    return b;
}

std::string llama_ple_direct_reader::self_check() const {
    std::lock_guard<std::mutex> lk(pimpl->mtx);
    return pimpl->cache.cap > 0 ? pimpl->cache.check() : "";
}

#else // !LLAMA_PLE_DIRECT_SUPPORTED

struct llama_ple_direct_reader::impl {};

llama_ple_direct_reader::llama_ple_direct_reader(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}
llama_ple_direct_reader::~llama_ple_direct_reader() = default;

std::unique_ptr<llama_ple_direct_reader> llama_ple_direct_reader::open(int, size_t, size_t, int64_t,
        const llama_ple_direct_params &, std::string * err) {
    if (err) { *err = "direct reads are implemented for Linux only"; }
    return nullptr;
}

void llama_ple_direct_reader::gather(const int32_t *, int64_t, uint8_t *, const uint8_t *) { GGML_ABORT("ple-direct: unsupported"); }
llama_ple_direct_stats llama_ple_direct_reader::stats() const { return {}; }
std::string llama_ple_direct_reader::describe() const { return "unsupported"; }
std::string llama_ple_direct_reader::stats_line() const { return ""; }
std::string llama_ple_direct_reader::self_check() const { return ""; }

#endif
