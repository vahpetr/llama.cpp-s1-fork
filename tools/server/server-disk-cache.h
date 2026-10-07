#pragma once

#include "server-common.h"
#include "server-task.h"

#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

// Disk tier of the prompt cache [TAG_DISK_PROMPT_CACHE]
//
// A slot's state (target KV + recurrent state, the draft model's KV, the context checkpoints and the speculative
// pending state) is written to <dir>/<id>.* when another conversation is about to overwrite it, and loaded back
// into a slot when a prompt arrives whose common prefix with a stored entry is longer than with the slot. On a
// slow GPU with fast NVMe this turns a re-prefill of minutes into a load of about a second (s1: ~15 KB per token,
// ~6.8 GB/s read), so several chats or agent sessions can share one slot.
//
// Files of one entry: <id>.tok (serialized server_tokens, used for the prefix match), <id>.tgt / <id>.dft
// (llama_state_seq_save_file), <id>.aux (checkpoints, speculative state, deep checkpoint refs), <id>.meta (written
// last; an entry without it is incomplete and deleted). Entries from another model / context size / binary / flags
// are deleted on start (fingerprint). The server owns the directory: quota, TTL and the value order are enforced
// after every save and on start.
//
// Deep checkpoints [TAG_DEEP_CKPT]: a hybrid model can only rewind its recurrent state to a checkpoint, and the
// slot keeps a few, all near the end. Checkpoints that leave that list go to <dir>/ckpt/<key>.ck, keyed by the
// content of the prompt prefix they belong to, so a branch from the middle of a conversation, a copy of a
// conversation in a new chat, or any chat with the same system prompt + tools rewinds there and replays only the
// gap. Each conversation keeps them on a depth schedule (dense near the end, gap >= ratio x distance to the end),
// plus its anchor (first checkpoint) and the points branches were taken from. Unreferenced files are collected.
struct server_disk_cache_entry {
    std::string   id;
    server_tokens tokens;

    std::vector<server_deep_ref> deep;

    std::vector<std::string> blobs; // [TAG_DISK_DELTA] content blocks the .tgt / .dft maps reference
    std::vector<std::string> ckpt_keys; // its in-RAM checkpoints, kept in the checkpoint store

    uint64_t bytes   = 0; // logical size (what a full copy would take)
    uint64_t own_bytes = 0; // its own files (.tok, .aux, maps); the blocks are shared
    int64_t  t_saved = 0; // unix seconds
    int64_t  t_used  = 0; // unix seconds, last save or load
    int32_t  hits    = 0; // loads
    bool     has_dft = false;
};

// [TAG_DISK_LOAD_RESULT] why a load failed decides what happens to the entry: only an invalid entry (seal / hash /
//   format of any of its files) is erased; a capacity failure (the verified state does not fit the free cells) and an
//   inconclusive one keep it
enum server_disk_load_result {
    SERVER_DISK_LOAD_OK,
    SERVER_DISK_LOAD_INVALID,
    SERVER_DISK_LOAD_CAPACITY,
    SERVER_DISK_LOAD_OTHER,
};

struct server_deep_ckpt_meta {
    std::string key;
    int64_t     n_tokens = 0;
    int32_t     pos_min  = 0;
    int32_t     pos_max  = 0;
    uint64_t    bytes    = 0;
    int64_t     t_used   = 0;
    int32_t     hits     = 0;
};

struct server_disk_cache {
    server_disk_cache(std::string dir, uint64_t quota_bytes, int32_t min_tokens, int32_t ttl_days,
                      std::string fingerprint, bool has_mtmd, float ckpt_ratio, int32_t ckpt_min_step, int32_t ckpt_max);

    // scan the directory: keep the complete entries with this fingerprint, delete the rest, enforce the limits
    void init();

    // write the state of seq id_slot; a no-op when an entry already holds exactly these tokens. Entries that are a
    // prefix of the new one are replaced. Returns true if the state is on disk.
    bool save(llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, int32_t id_slot,
              const server_prompt & prompt);

    // index of the entry with the longest common prefix with tokens, if longer than lcp_min and it fits n_ctx_slot
    int find(const server_tokens & tokens, size_t lcp_min, size_t n_ctx_slot, size_t * lcp_out) const;

    // load entry idx into seq id_slot and prompt; unless OK the slot's memory is undefined (caller clears it)
    server_disk_load_result load(int idx, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec,
                                 int32_t id_slot, server_prompt & prompt);

    // [TAG_DEEP_CKPT] store a checkpoint of prompt (taken at c.n_tokens) and reference it from prompt.deep, then
    // re-apply the depth schedule to prompt.deep
    void deep_put(server_prompt & prompt, const common_prompt_checkpoint & c, uint8_t kind);

    // stored checkpoints with n_tokens <= n_max whose prefix matches tokens, deepest first
    std::vector<server_deep_ckpt_meta> deep_candidates(const server_tokens & tokens, int64_t n_max) const;

    bool deep_get(const std::string & key, common_prompt_checkpoint & out);

    // reference a stored checkpoint a deep rewind used from prompt.deep (it then survives the collection)
    void deep_adopt(server_prompt & prompt, const server_deep_ckpt_meta & m);

    // delete checkpoint files that no entry and no live prompt (live_refs) references
    void deep_gc();

    // keys referenced by the server's live prompts (slots, the RAM prompt cache); set by the server
    std::function<std::set<std::string>()> live_refs;

    int32_t min_tokens() const { return n_min_tokens; }

    uint64_t total_bytes() const;

    // counters for the logs and /metrics
    uint64_t n_saves = 0;
    uint64_t n_loads = 0;
    uint64_t n_evict = 0;
    uint64_t n_deep_rewinds = 0;
    uint64_t n_load_failed  = 0; // invalid: the entry was dropped
    uint64_t n_load_capacity = 0; // the state did not fit the free cells: the entry was kept
    uint64_t n_load_other   = 0; // inconclusive: the entry was kept

    // the entry with this id is never evicted (a load retried across purges) - empty: none
    std::string protect_id;

    int find_id(const std::string & id) const;
    const std::string & entry_id(int idx) const { return entries[idx].id; }
    uint64_t bytes_written  = 0;
    uint64_t bytes_loaded   = 0;
    uint64_t save_us        = 0;
    uint64_t load_us        = 0;

    size_t n_entries()    const { return entries.size(); }
    size_t n_deep_ckpts() const { return ckpts.size(); }
    size_t n_blocks()     const { return blobs.size(); }

    // the stored conversations and totals for GET /disk-cache; thread-safe (a snapshot the server loop refreshes)
    json status() const;
    void refresh_status();

    // after a model reload (wake-up from idle sleep): a changed fingerprint switches to that fingerprint's namespace
    //   (the old one stays as a foreign namespace)
    void revalidate(const std::string & fingerprint);

    // [TAG_STATE_DESCRIPTOR] what the state streams of these contexts contain (llama_state_seq_descriptor + the
    //   speculative carry schema, the token / media serialization versions and the checkpoint semantics revision);
    //   its hash heads every .tgt / .dft / .ck this cache writes and must match on every read. Called before init()
    //   and before every revalidate().
    void set_descriptor(llama_context * ctx_tgt, llama_context * ctx_dft, const std::string & spec_types);
    const std::string & descriptor_hash() const { return desc_h; }

  private:
    // [TAG_DISK_NAMESPACE] <root>/ns-<key>/ holds this fingerprint's entries and ckpt/; <root>/blk/ the shared blocks;
    //   the flat top level (entries + ckpt/ of a format <= 6 build) is a legacy namespace, kept in place for a rollback
    struct foreign_ns {
        std::string key;      // hash16 of its fingerprint
        std::string path;     // its directory (the root for the legacy namespace)
        bool        legacy  = false;
        size_t      n_entries = 0;
        uint64_t    bytes   = 0; // its own files (entries + checkpoints), not the shared blocks
        int64_t     t_newest = 0; // mtime of its newest file
        std::vector<std::string> ids; // legacy: its entry ids at the top level
        std::set<std::string> blocks; // blocks its maps reference (kept by the block collection)
    };
    std::vector<foreign_ns> foreign;
    void scan_foreign();
    void remove_foreign(const foreign_ns & ns);
    bool foreign_protected(const foreign_ns & ns) const;

    std::string root;
    std::string dir; // the own namespace
    uint64_t    quota;
    int32_t     n_min_tokens;
    int32_t     ttl_days;
    std::string fingerprint;
    bool        has_mtmd;

    std::string desc_h;     // hash16 of the combined descriptor (target + draft): .meta, .ck, /disk-cache
    std::string desc_tgt_h; // envelope hash of the .tgt streams
    std::string desc_dft_h; // envelope hash of the .dft streams
    float       ckpt_ratio;
    int32_t     ckpt_min_step;
    int32_t     ckpt_max;

    std::vector<server_disk_cache_entry> entries;

    std::map<std::string, server_deep_ckpt_meta> ckpts;

    mutable std::mutex status_mtx;
    json               status_snap;

    // [TAG_DISK_DELTA] content-addressed blocks of the state streams: key -> bytes on disk
    std::map<std::string, uint64_t> blobs;
    uint64_t blob_bytes = 0;

    bool ckpt_store(const std::string & key, const common_prompt_checkpoint & c, bool * written);
    std::string blob_path(const std::string & key) const;
    bool save_stream(llama_context * ctx, int32_t id_slot, const std::string & map_path, const std::string & env_h,
                     std::vector<std::string> & keys, uint64_t & n_logical, uint64_t & n_written);
    // env_h: the envelope hash the stream must carry; legacy_ok: a stream without an envelope (format 5 / 6) is accepted
    server_disk_load_result load_stream(llama_context * ctx, int32_t id_slot, const std::string & map_path,
                                        const std::string & env_h, bool legacy_ok);
    static bool read_map_keys(const std::string & map_path, std::vector<std::string> & keys);
    void blob_gc();

    std::string path(const std::string & id, const char * ext) const;
    std::string ckpt_path(const std::string & key) const;
    void        remove_files(const std::string & id) const;
    bool        write_meta(const server_disk_cache_entry & e) const;
    void        erase(size_t idx);
    void        enforce(const std::string & protect);
    void        thin(std::vector<server_deep_ref> & refs, int64_t n_end) const;
};

// identity of everything a stored state depends on: model + draft files, context size, KV types, the binary,
// its command line and the LLAMA_* / GGML_* environment knobs
std::string server_disk_cache_fingerprint(const common_params & params, int32_t n_ctx);
