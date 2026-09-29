#pragma once

// [TAG_PLE_DIRECT] direct-read gather of the rows of one row-major table that lives in the model file (the qwen4exp PLE
// n-gram table, 320 M rows x 110 B = 32.8 GiB, TENSOR_READ_LAZY): pread / O_DIRECT reads of exactly the rows a ubatch
// needs instead of page faults on the mmap, with an exact LRU row cache in front.
//
// Exactness: gather() writes, for every id, the bytes of that row as they are in the file -- the same bytes the mmap
// path reads (a copy in the row cache is a copy of those bytes; the model file is opened read-only and checked to be the
// same inode as the loader's descriptor). A failed read falls back to a memcpy from the mapping when one is given.
//
// Linux only (O_DIRECT, preadv2, /proc/self/fd); open() returns nullptr elsewhere and the caller keeps the mmap path.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct llama_ple_direct_params {
    // 1 = O_DIRECT reads (the page cache is neither used nor filled; falls back to 2 if the file system refuses
    //     O_DIRECT), 2 = buffered pread with POSIX_FADV_RANDOM (page cache, one page per missing row, no readahead)
    int    mode        = 1;
    size_t cache_bytes = 0;       // exact LRU row cache budget in bytes of row data (0 = no cache)
    int    n_threads   = 16;      // reads in flight at most (the calling thread is one of them)
    bool   nowait      = false;   // mode 1: serve a row whose page(s) the page cache already holds (mincore on the mapping,
                                  // then a copy from it; never starts I/O); needs the mapping in gather()
    bool   mlock_cache = false;   // mlock() the row cache (warns and continues if the limit refuses it)
    int    stats_every = 0;       // log one stats line every N gathers (0 = only at destruction)
    int    fail_reads_for_test = 0; // test hook: every device read fails -> the mmap fallback must deliver the rows

    // LLAMA_PLE_DIRECT (0 = off), _CACHE_MIB, _THREADS, _NOWAIT, _MLOCK, _STATS; returns mode 0 when off
    static llama_ple_direct_params from_env();
};

struct llama_ple_direct_stats {
    uint64_t gathers       = 0;
    uint64_t ids           = 0;   // ids asked for
    uint64_t unique        = 0;   // distinct ids per gather, summed
    uint64_t cache_hits    = 0;   // distinct ids served by the row cache
    uint64_t pc_hits       = 0;   // distinct ids served by the page cache (mincore probe, mode 1 + nowait)
    uint64_t dev_rows      = 0;   // distinct ids read from the device / buffered pread
    uint64_t dev_reads     = 0;   // read syscalls issued for them (adjacent rows are merged)
    uint64_t dev_bytes     = 0;   // bytes those reads transferred
    uint64_t fallback_rows = 0;   // rows copied from the mapping after a failed read
    uint64_t evictions     = 0;
    double   us_total      = 0.0; // wall time inside gather()
    double   us_max        = 0.0;
};

struct llama_ple_direct_reader {
    // fd_src: an open descriptor of the model file; it is re-opened through /proc/self/fd, so the caller may close it
    // afterwards. file_offs: byte offset of row 0 in the file; row_size: bytes per row; n_rows: rows in the table.
    // Returns nullptr (and a reason in *err) when direct reads are unavailable.
    static std::unique_ptr<llama_ple_direct_reader> open(int fd_src, size_t file_offs, size_t row_size, int64_t n_rows,
                                                         const llama_ple_direct_params & params, std::string * err);

    ~llama_ple_direct_reader();

    // dst + i*row_size <- the file bytes of row ids[i], for i in [0, n). ids must lie in [0, n_rows) (asserted, as the
    // CPU get_rows does). mmap_rows (row 0 of the table in memory, or nullptr): the source of the fallback copy when a
    // read fails; without it a failed read aborts. Thread-safe (one gather at a time per reader).
    void gather(const int32_t * ids, int64_t n, uint8_t * dst, const uint8_t * mmap_rows);

    llama_ple_direct_stats stats() const;
    std::string describe() const;      // one line: mode, alignment, cache, threads
    std::string stats_line() const;    // one line: the counters

    // test hook: verifies the row cache's hash table and LRU list; returns an empty string when consistent
    std::string self_check() const;

    struct impl;

private:
    explicit llama_ple_direct_reader(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};
