#pragma once

#include "server-common.h"
#include "server-task.h"

#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <cstdint>
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
// (llama_state_seq_save_file), <id>.aux (checkpoints + speculative state), <id>.meta (written last; an entry
// without it is incomplete and deleted). Entries from another model / context size / server binary are deleted
// on start (fingerprint). The server owns the directory: quota, TTL and the value order are enforced after every
// save and on start.
struct server_disk_cache_entry {
    std::string   id;
    server_tokens tokens;

    uint64_t bytes   = 0;
    int64_t  t_saved = 0; // unix seconds
    int64_t  t_used  = 0; // unix seconds, last save or load
    int32_t  hits    = 0; // loads
    bool     has_dft = false;
};

struct server_disk_cache {
    server_disk_cache(std::string dir, uint64_t quota_bytes, int32_t min_tokens, int32_t ttl_days,
                      std::string fingerprint, bool has_mtmd);

    // scan the directory: keep the complete entries with this fingerprint, delete the rest, enforce the limits
    void init();

    // write the state of seq id_slot; a no-op when an entry already holds these tokens (or a longer prompt with
    // them as prefix). Entries that are a prefix of the new one are replaced. Returns true if the state is on disk.
    bool save(llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, int32_t id_slot,
              const server_prompt & prompt);

    // index of the entry with the longest common prefix with tokens, if longer than lcp_min and it fits n_ctx_slot
    int find(const server_tokens & tokens, size_t lcp_min, size_t n_ctx_slot, size_t * lcp_out) const;

    // load entry idx into seq id_slot and prompt; on false the slot's memory is undefined (caller clears it)
    bool load(int idx, llama_context * ctx_tgt, llama_context * ctx_dft, common_speculative * spec, int32_t id_slot,
              server_prompt & prompt);

    int32_t min_tokens() const { return n_min_tokens; }

    uint64_t total_bytes() const;

    // counters for /metrics and the logs
    uint64_t n_saves = 0;
    uint64_t n_loads = 0;
    uint64_t n_evict = 0;

  private:
    std::string dir;
    uint64_t    quota;
    int32_t     n_min_tokens;
    int32_t     ttl_days;
    std::string fingerprint;
    bool        has_mtmd;

    std::vector<server_disk_cache_entry> entries;

    std::string path(const std::string & id, const char * ext) const;
    void        remove_files(const std::string & id) const;
    bool        write_meta(const server_disk_cache_entry & e) const;
    void        erase(size_t idx);
    void        enforce(const std::string & protect);
};

// identity of everything a stored state depends on: model + draft files, context size, KV types and the binary
std::string server_disk_cache_fingerprint(const common_params & params, int32_t n_ctx);
