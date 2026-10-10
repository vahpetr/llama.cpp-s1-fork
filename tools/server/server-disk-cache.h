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

    uint32_t    version = 0;  // payload format as written (a 5 / 6 entry of an adopt source stays 5 / 6)
    std::string src;          // [TAG_DISK_ADOPT] the adopt source's key; empty: the own namespace
    std::string adopted_from; // own entry saved after a load from an adopt source: that source's key
    // [TAG_PARTIAL_PROMPT] the number of prompt rows the draft history held at save time (.meta "draft_cover"); a load
    //   with draft_cover < the token count marks the prompt draft-partial ([TAG_DRAFT_PARTIAL]). A .meta without it: the token count (its writer
    //   kept C1), except an own entry with adopted_from (v50rc4 published adopted target-only states): 0
    int64_t     draft_cover = 0;
    std::string desc_full;    // [TAG_CACHE_COMPAT2] the writer's full-coverage descriptor hash (format >= 8; "" before)
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

// [TAG_DISK_READ_STATUS] (Codex gate r1 A3 Major 1) a read that could not be done (open / read / allocation error)
//   is inconclusive and keeps the files; only verified damage (missing file, truncation, format, checksum) is BAD
enum sealed_status { SEALED_OK, SEALED_IO, SEALED_BAD };

struct server_deep_ckpt_meta {
    std::string key;
    int64_t     n_tokens = 0;
    int32_t     pos_min  = 0;
    int32_t     pos_max  = 0;
    uint64_t    bytes    = 0;
    int64_t     t_used   = 0;
    int32_t     hits     = 0;
    std::string src_dir; // [TAG_DISK_ADOPT] the adopt source's ckpt/ (read-only); empty: the own namespace
    uint32_t    src_version = 0; // [TAG_CACHE_COMPAT2] the adopt source's payload format (its fingerprint's v<N>)
    std::string src_key; // (R92, sol review) the adopt source key written in the checkpoint's header: its owner (two legacy
                         //   sources share <root>/ckpt, so src_dir does not identify one)
};

// the structural identity of a disk-cache fingerprint (named, length-delimited fields; shard identity arrays) for
//   adoption eligibility; "" with *why set when it does not parse unambiguously
// draft_kv = false: the draft KV types (dk / dv) are left out ([TAG_ADOPT_TARGET_ONLY]: a source that differs only there
//   is adopted for target-only loads - the draft stream's descriptor names its KV types, so it never loads)
std::string server_disk_cache_fingerprint_structure(const std::string & fp, std::string * why = nullptr, bool draft_kv = true);

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

    // io_error (optional): set when the file could not be read (inconclusive, the checkpoint is kept)
    // [TAG_ADOPT_TARGET_ONLY] learn: the caller loads an entry target-only (its target stream matched) - its prompt
    //   checkpoints, written in the same save as that target stream, teach the source's combined descriptor; a
    //   checkpoint of an adopt source that carries its learned descriptor loads target-only (draft data dropped)
    bool deep_get(const std::string & key, common_prompt_checkpoint & out, bool * io_error = nullptr, bool learn = false);

    // reference a stored checkpoint a deep rewind used from prompt.deep (it then survives the collection)
    void deep_adopt(server_prompt & prompt, const server_deep_ckpt_meta & m);

    // [TAG_CKPT_LOAD_CHECKED] a checkpoint that passed deep_get but did not restore (state length mismatch): forget
    //   it and remove its file (an adopt source's file stays, read-only)
    void deep_drop(const std::string & key, const char * why);

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
    uint64_t n_ckpt_dropped = 0;
    uint64_t n_blk_copied_total = 0; // (Codex gate r3 Major 1) adopted / legacy blocks copied into the own store at a save
    uint64_t n_deep_copied  = 0; // adopted deep checkpoints copied into the own namespace at a resave // deep checkpoints that did not restore
    uint64_t n_load_failed  = 0; // invalid: the entry was dropped
    uint64_t n_load_capacity = 0; // the state did not fit the free cells: the entry was kept
    uint64_t n_load_other   = 0; // inconclusive: the entry was kept
    uint64_t n_save_partial        = 0; // [TAG_PARTIAL_PROMPT] saves refused: a partial prompt (C1)
    uint64_t n_load_partial_cover  = 0; // [TAG_DRAFT_PARTIAL] own loads with draft_cover < n_tokens (draft-partial)
    uint64_t n_load_draft_partial  = 0; // [TAG_DRAFT_PARTIAL] loads that left the prompt draft-partial (target-only incl.)
    uint64_t n_save_draft_partial  = 0; // [TAG_DRAFT_PARTIAL] saves of a draft-partial prompt (draft_cover < n_tokens)
    uint64_t n_ckpt_target_only    = 0; // [TAG_ADOPT_TARGET_ONLY] adopt-source checkpoints loaded target-only
    uint64_t n_adopted_target_only = 0; // [TAG_ADOPT_TARGET_ONLY] adopted loads of the target stream only (draft reset)

    // [TAG_DISK_ADOPT] --cache-disk-adopt: foreign namespace keys opened read-only as extra load sources (set before
    //   init()); eligibility (structural fingerprint part) at init, usability at load
    std::vector<std::string> adopt_keys;

    // the entry with this id is never evicted (a load retried across purges) - empty: none
    std::string protect_id;

    int find_id(const std::string & id) const;

    // [TAG_DISK_VERIFY] (Codex gate r1 A3 Major 2) check entry idx completely without touching any context: the .aux,
    //   its checkpoint files, the .tgt / .dft maps and every block they reference (hashes). OK = verified; INVALID =
    //   damaged, the entry is dropped (as by load()); OTHER = inconclusive (I/O), kept. Gates a capacity-driven purge.
    server_disk_load_result verify(int idx);
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
    json storage_split() const; // own / foreign / shared / held bytes (Codex gate r2 Major 5)

    // after a model reload (wake-up from idle sleep): a changed fingerprint switches to that fingerprint's namespace
    //   (the old one stays as a foreign namespace)
    void revalidate(const std::string & fingerprint);

    // [TAG_STATE_DESCRIPTOR] what the state streams of these contexts contain (llama_state_seq_descriptor + the
    //   speculative carry schema, the token / media serialization versions and the checkpoint semantics revision);
    //   its hash heads every .tgt / .dft / .ck this cache writes and must match on every read. Called before init()
    //   and before every revalidate().
    void set_descriptor(llama_context * ctx_tgt, llama_context * ctx_dft, const std::string & spec_types);
    const std::string & descriptor_hash() const { return desc_h; }
    const std::string & descriptor_key() const { return desc_key; } // [TAG_CACHE_COMPAT2] the key v2 component

    // [TAG_CACHE_COMPAT2] replace the fingerprint (and the own namespace) before init(): the v2 key is known only
    //   after set_descriptor() and the measurement
    void set_fingerprint(const std::string & fp);
    const std::string & root_dir() const { return root; }
    bool owns_root() const { return lock_fd >= 0; } // [TAG_DISK_LOCK] this process holds the cache root's lock

  private:
    // [TAG_DISK_NAMESPACE] <root>/ns-<key>/ holds this fingerprint's entries, ckpt/ and blk/ (its new blocks,
    //   [TAG_DISK_BLK_NS]); <root>/blk/ is the legacy (format <= 6) block store, read-only for new blocks. Rollback-
    //   forward: a v47 rollback never touches ns-<key>/ (own entries + exclusive blocks survive it); a legacy block a
    //   v48 entry references (dedup) can be collected by v47 when v47 evicts its flat owner - that v48 entry is then
    //   dropped as incomplete at the next v48 start (a miss, never a wrong load);
    //   the flat top level (entries + ckpt/ of a format <= 6 build) is a legacy namespace, kept in place for a rollback
    struct foreign_ns {
        std::string key;      // hash16 of its fingerprint
        std::string path;     // its directory (the root for the legacy namespace)
        bool        legacy  = false;
        bool        incomplete = false; // its listing failed: never removed this scan
        size_t      n_entries = 0;
        uint64_t    bytes   = 0; // its own files (entries + checkpoints), not the shared blocks
        int64_t     t_newest = 0; // mtime of its newest file
        std::vector<std::string> ids; // legacy: its entry ids at the top level
        std::set<std::string> blocks; // blocks its maps reference (kept by the block collection)
        std::set<std::string> stored; // blocks in its own <ns>/blk (Codex gate r4 Major: an own copy of one is redundant)
    };
    std::vector<foreign_ns> foreign;

    struct adopt_src {
        std::string key;
        std::string fingerprint;
        uint32_t    version = 0; // [TAG_CACHE_COMPAT2] its payload format (fingerprint_format)
        bool        legacy = false;
        size_t      n_entries = 0;
        uint64_t    hits = 0;
        uint64_t    dropped = 0; // entries whose load failed invalid: no longer eligible (files kept); (R92) plus
                                 //   its deep checkpoints dropped at a read (descriptor mismatch, damage, no restore)
        uint64_t    dropped_ckpts = 0; // (R92) the deep-checkpoint part of `dropped`
        uint64_t    target_only = 0; // [TAG_ADOPT_TARGET_ONLY] hits that loaded the target stream only
        bool        draft_kv_differs = false; // its draft KV types differ: its entries can only load target-only
    };
    std::vector<adopt_src> adopted;
    void count_src_ckpt_drop(const server_deep_ckpt_meta & m); // (R92) charge a dropped adopted checkpoint to its owning source
    std::map<std::string, std::string> src_dir_of; // adopt entry id -> its directory
    // [TAG_ADOPT_TARGET_ONLY] adopt source checkpoint directory -> its writer's combined state descriptor, learned from the prompt
    //   checkpoints of an entry whose target stream matched this target (target_only()): its checkpoints and deep
    //   checkpoints carry that descriptor and load target-only
    std::map<std::string, std::string> tgt_compat_dh;
    void open_adopt_sources();
    sealed_status read_entry(const std::string & id, const std::string & fp, server_disk_cache_entry & e) const;
    bool scan_foreign(); // false: a listing failed, the foreign block sets are incomplete
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
    std::string desc_key;   // [TAG_CACHE_COMPAT2] full-coverage hash16 of the combined descriptor (key v2)
    std::string desc_tgt_full; // [TAG_CACHE_COMPAT2] envelope v2: full-coverage hash16 of the target descriptor
    std::string desc_dft_full; // envelope v2: full-coverage hash16 of the draft descriptor
    // which state envelopes a stream may carry (Codex compat2 D): v2 with the full-coverage hash; v1 (weak hash)
    //   only for an adopted format-7 entry; none only for an adopted format 5 / 6 entry
    struct stream_rule {
        std::string h2;               // required v2 hash ("": nothing passes)
        std::string h1;               // v1 hash, when v1_ok
        bool        v1_ok     = false;
        bool        legacy_ok = false;
    };
    // tgt_only: [TAG_ADOPT_TARGET_ONLY] an adopt source's entry judged per stream (its descriptor_full names its own
    //   draft, which differs by design): the stream's own full-coverage hash binds it
    stream_rule rule_for(const server_disk_cache_entry & e, bool tgt, bool tgt_only = false) const;
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
    // [TAG_DISK_BLK_NS] (Codex gate r4 Major) physical inventories, apart from the read location: a key can sit in
    //   several stores at once (an own copy of an adopted / legacy block, its original kept). Read location: own >
    //   legacy > adopt source. Accounting charges each physical file; collection decides per store
    std::set<std::string> blk_own;              // keys with a file in <dir>/blk
    std::set<std::string> blk_legacy;           // keys with a file in <root>/blk (never written, only collected)
    std::map<std::string, std::string> blk_src; // key -> an adopt source's <ns>/blk holding it (read-only)
    void blob_drop_copy(const std::string & key); // a damaged read-location copy: forget it (remove it if writable)
    // (Codex gate r5 Major 1) blob_gc() removal: true when the file is gone (removed or already absent); any other
    //   error keeps the file inventoried and charged (warned once per run)
    bool gc_unlink(const std::string & path);
    bool warned_gc_unlink = false;
    uint64_t blob_bytes = 0;
    // [TAG_DISK_READ_STATUS] (Hermes gate Major 1) an own entry / block / checkpoint skipped at start on an I/O error:
    //   what it references is unknown, so no block or checkpoint is collected for the rest of this run
    bool gc_hold = false;
    int  lock_fd = -1; // [TAG_DISK_LOCK] the cache root's .lock, flock'ed for the process lifetime; -2: refused
    bool warned_unidentified = false; // an unreadable legacy .meta was reported (once per run)

    bool ckpt_store(const std::string & key, const common_prompt_checkpoint & c, bool * written);
    std::string blob_path(const std::string & key) const;
    bool        blob_writable(const std::string & key) const;
    bool save_stream(llama_context * ctx, int32_t id_slot, const std::string & map_path, const std::string & env_h,
                     std::vector<std::string> & keys, uint64_t & n_logical, uint64_t & n_written);
    // env_h: the envelope hash the stream must carry; legacy_ok: a stream without an envelope (format 5 / 6) is accepted
    server_disk_load_result load_stream(llama_context * ctx, int32_t id_slot, const std::string & map_path,
                                        const stream_rule & rule);
    static bool read_map_keys(const std::string & map_path, std::vector<std::string> & keys, sealed_status * st = nullptr);
    server_disk_load_result verify_stream(const std::string & map_path, const stream_rule & rule);
    // [TAG_ADOPT_TARGET_ONLY] an adopt source's entry whose .tgt envelope names this target descriptor while its .dft
    //   envelope names another draft descriptor (or envelope version): its target stream alone is loadable
    bool target_only(const server_disk_cache_entry & e) const;
    bool draft_kv_src(const server_disk_cache_entry & e) const; // its adopt source differs in the draft KV types
    void drop_invalid(int idx); // erase an invalid entry, count it, collect what it alone referenced
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
// its command line and the LLAMA_* / GGML_* environment knobs. Since key v2 [TAG_CACHE_COMPAT2] this is the CHEAP
// identity: it keys the measurement memo and carries the adopt structure; the namespace key is the v2 key below.
// [TAG_MULTISLOT] n_ctx is the PER-SLOT context (llama_n_ctx_seq; = llama_n_ctx for np1 and --kv-unified): a stored
// state is one sequence and per-seq state is stream-count agnostic, so the adopt structure does not bind to the slot
// count / KV layout (those stay in the parsed-param digest, i.e. in the namespace key)
std::string server_disk_cache_fingerprint(const common_params & params, int32_t n_ctx);

// [TAG_CACHE_COMPAT2] compatibility key v2 (R80.2 parts 2-3, spec series D4): namespace key = hash of
//   {model content identity (sampled hash per shard), state descriptor hash, short-context measurement hash,
//    depth version, semantic-env digest}. The libraries, sources, argv and exact env knobs stay in the cheap identity
//   only: a rebuild / flag change that writes the same states keeps the namespace.

// bumped only by the gate (outputs at depth differ)
static constexpr uint32_t DISK_CACHE_DEPTH_VERSION = 1;
// the measurement's probe (tokens, positions, outputs, ubatch schedule, hashed sections); a change re-measures
static constexpr uint32_t DISK_CACHE_PROBE_REVISION = 1;

enum server_cache_env_class {
    SERVER_CACHE_ENV_EXACT,    // speed / diagnostics / control by contract: not in the key
    SERVER_CACHE_ENV_SEMANTIC, // results / representation, unclear or unregistered: digested into the key
};

// the env registry (spec appendix "Registry seed"); unregistered LLAMA_* / GGML_* names are semantic
server_cache_env_class server_cache_env_classify(const std::string & name);

// digest (hash16) of the semantic knobs among "NAME=value" entries; order-independent; "-" when there are none
std::string server_cache_env_digest(const std::vector<std::string> & env);

// the process's LLAMA_* / GGML_* environment entries ("NAME=value")
std::vector<std::string> server_cache_environ();

// sampled content hash of a file (size + fixed-offset samples, a few MiB): "<hash16>"; "absent" when it cannot be read
std::string server_cache_file_identity(const std::string & path);

// model + draft + mmproj content identity: every shard of a split GGUF, "," joined per file, "|" between files
std::string server_disk_cache_model_identity(const common_params & params);

// [TAG_CACHE_COMPAT2] the parsed-param digest (Codex compat2 A): a normalized digest of the semantic parsed
//   configuration of BOTH contexts. In: requested + effective n_batch / n_ubatch, n_parallel / n_seq_max, threads
//   (generation + batch, target + draft, requested + effective), ctx-checkpoints + checkpoint-min-step (they
//   place the prefill ubatch boundaries before a checkpoint), NUMA strategy, devices, n_gpu_layers, main GPU,
//   split mode, the effective placement of both loaded models (after fit-params),
//   tensor split, tensor-buffer overrides, KV / op offload, extra buffer types (repack), host buffers, flash
//   attention, KV types, SWA-full, unified KV, the QSA ratio, --override-kv (model metadata overrides), the loaded
//   indexer / QSA metadata (selection budget + policy per layer), and the speculative configuration (types, synthetic
//   acceptance, draft limits, adaptive policy, split / min probabilities, backend sampling, draft placement + KV
//   types, n-gram parameters). Argv and LLAMA_ARG_* env give the same parsed values, hence the same digest.
//   Exact-qualified exclusions (do not change the state bytes a token sequence produces):
//   - model / draft / mmproj paths, HF / URL sources: the content identity (model_id) is keyed instead
//   - n_ctx, RoPE / YaRN, n_ctx_seq: the effective values are in the state descriptor (desc)
//   [TAG_MULTISLOT] n_parallel / n_seq_max / unified KV are IN (the namespace key differs across them by design) but
//     NOT in the state descriptor (llama_state_seq_descriptor prints the single-slot values): the per-seq bytes do
//     not depend on them, so another slot count reaches an np1 namespace through --cache-disk-adopt
//   - mmap / mlock / direct-io / check-tensors / fit-params*: loading method; fit's effects (GPU layers, tensor
//     split, buffer overrides) are keyed through the effective placement of the loaded model
//     (server_cache_model_placement, in the ctx canon), its n_ctx through the descriptor
//   - cpu masks / priority / poll / strict placement: scheduling of the same per-row work, thread counts are in
//   - sampling defaults, grammar, chat template / jinja / reasoning, stop words: they choose tokens; a stored state
//     is keyed by its tokens
//   - host / port / API key / timeouts / slots / metrics / webui / log options: control plane
//   - cache-ram, cache-disk-*: where / which states are kept, not their bytes
//   - cache-reuse: changes the bytes, but a reused (shifted) prompt is marked non-canonical and never saved
//   - context shift, n_keep: change the bytes, but a shifted prompt is non-canonical and never saved; n_predict:
//     generation policy after a restore
//   - LoRA adapters: the disk cache is disabled with LoRA
//   - ngram lookup cache paths: draft proposals only, verified against the target like any draft (n-gram params in)
std::string server_cache_param_canon(const common_params & params);
// [TAG_CACHE_COMPAT2] effective placement (astra r2): hash16 of llama_model_placement_desc - GPU-layer count, split
//   mode, devices, tensor split, per-layer device and per-weight buffer type AS LOADED (after fit-params); "none"
//   without a model. It enters the ctx canon (hence the key digest) and, with the NUMA strategy, the memo identity
std::string server_cache_model_placement(const llama_model * m);
std::string server_cache_placement_identity(const common_params & params, const llama_context * ctx_tgt,
                                            const llama_context * ctx_dft);
std::string server_cache_ctx_canon(const llama_context * ctx);
// [TAG_CACHE_COMPAT2] an entry's metadata descriptor (descriptor_full) against this context's: mandatory for an own
//   entry and for any format >= 8 entry (absent / empty => miss), optional only for an adopted pre-8 entry
bool server_disk_cache_desc_ok(bool own, uint32_t version, const std::string & desc_full, const std::string & desc_key);
// [TAG_CACHE_COMPAT2] the RAM cache's state descriptor hash: target + draft llama_state_seq_descriptor and the carry
//   schema (common_speculative_state_schema) - a RAM entry is swapped in only under the same descriptor
std::string server_cache_state_desc(const llama_context * ctx_tgt, const llama_context * ctx_dft,
                                    const std::string & carry_schema);
std::string server_cache_param_digest(const common_params & params, const llama_context * ctx_tgt,
                                      const llama_context * ctx_dft);

struct server_cache_key_parts {
    std::string model_id;   // server_disk_cache_model_identity
    std::string desc_h;     // server_disk_cache::descriptor_key()
    std::string measure_h;  // the short-context measurement (hash16)
    uint32_t    depth_version = DISK_CACHE_DEPTH_VERSION;
    std::string env_digest; // server_cache_env_digest
    std::string param_digest; // server_cache_param_digest
};

// the v2 fingerprint: "<key16> <cheap body>|compat2=<len>:<v2 body>" - the cheap body keeps the adopt structure
//   (server_disk_cache_fingerprint_structure) parseable, key16 = hash of the v2 body alone
std::string server_disk_cache_fingerprint_v2(const std::string & cheap, const server_cache_key_parts & p);

// the v2 body of a fingerprint ("" for a v1 fingerprint)
std::string server_disk_cache_compat2_body(const std::string & fp);

// same namespace identity: two v2 fingerprints compare by key16 + v2 body (the cheap part may differ), else exact
bool server_disk_cache_same_identity(const std::string & a, const std::string & b);

// the measurement memo, on disk under <root>/compat-memo/: key = cheap identity + probe revision + GPU device identity
// [TAG_CACHE_COMPAT2] the runtime identity of the memo (Codex compat2 A "stale memo"): GPU devices (name,
//   description, PCI id), every backend's feature list (CPU ISA), the mapped driver / runtime shared objects (path +
//   size + mtime + inode; ours are content-hashed in the cheap identity) and the driver environment (RADV_ / MESA_ /
//   AMD_ / VK_ / ...) - a driver change forces a re-measurement. Gap: ggml's device description carries no Vulkan
//   driverVersion / apiVersion; a driver update that replaces no mapped file (e.g. firmware) is not seen
//   A mapped library whose device + inode in /proc/self/maps differ from stat() of its path, or that is unlinked /
//   unreadable, makes the identity SERVER_CACHE_RUNTIME_UNVERIFIED: its memo key is "" (no memo read or written)
static constexpr const char * SERVER_CACHE_RUNTIME_UNVERIFIED = "unverified";
std::string server_cache_runtime_identity();
// one /proc/self/maps line: 0 = not a tracked library, 1 = verified (entry = path:size:mtime:inode), -1 = unverifiable
int server_cache_maps_lib(const std::string & line, std::string & entry);
// placement_id = server_cache_placement_identity (the effective placement + NUMA strategy of this load)
std::string server_cache_memo_key(const std::string & cheap, const std::string & runtime_id,
                                  const std::string & placement_id = "-");
bool server_cache_memo_get(const std::string & root, const std::string & memo_key, std::string & measure_h);
bool server_cache_memo_put(const std::string & root, const std::string & memo_key, const std::string & measure_h);

// [TAG_CACHE_COMPAT2] the key's measurement component: the hash, or "unmeasured-<random>" for a failed measurement
//   (a non-reusable namespace; the caller publishes no memo)
std::string server_cache_measure_component(const std::string & measure_h);

// the hash16 of measurement bytes (target, draft, carry sections as given)
std::string server_cache_measure_hash(const std::string & bytes);
