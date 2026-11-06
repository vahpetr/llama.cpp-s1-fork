#pragma once

// [TAG_CACHE_ALIGN] byte-aligned prompt-cache splice (P16e "canonical generated tokens", part 1: the core only)
//
// A reply the model generated is cached as the ids it sampled; a valid but non-canonical BPE segmentation (` (`
// `prec` `ache` instead of ` (` `pre` `cache`) re-tokenizes canonically when the client re-sends the text, and the
// id LCP of the next turn stops at the first such span. align() walks the cached ids C and the new prompt N byte by
// byte past their id LCP and returns a splice S = C[0:n_c] ++ N[n_n:] that keeps the cached ids for a byte-identical
// prefix. Rules (design r2 §4.1-4.3 + astra r2 OPEN 2):
//   - structural identity: a token whose bytes cannot identify it (CONTROL / USER_DEFINED / UNKNOWN, or an empty
//     piece) matches only the same id at the same byte offset on both sides; a literal `<|im_end|>` spelled by
//     ordinary pieces never matches the structural token, in either direction
//   - a splice point is a joint token boundary (equal byte prefix, both sides at a token boundary) that is usable:
//     (a) immediately before a common structural id, (b) the end of C with a structural id next in N, or (c) the end
//     of N; byte syncs inside a fragment are kept in `syncs` (position mapping) but never chosen as the splice point
//   - the walk stops at the first byte mismatch, a media placeholder (LLAMA_TOKEN_NULL) or the end of either side;
//     a splice with media at or after the splice point in N is declined (media remap is not implemented)
//   - complete delimiter identity: with message delimiters given, the message spans of S must equal the spans of N
//     mapped through the splice (same roles, every start an exact joint boundary); otherwise the next earlier usable
//     point is tried, and in the end the splice is declined (= the plain id LCP) - never an assert
// The result never keeps less than the id LCP: the declined / no-gain answer is n_c = n_n = id LCP (S = N).

#include "llama.h"
#include "chat.h"
#include "json.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// piece bytes + structural flag per token id; built once per vocab
struct server_cache_align_vocab {
    std::vector<std::string> piece;      // raw bytes of the token rendered with special = true
    std::vector<uint8_t>     structural; // 1 = matches only by id

    static server_cache_align_vocab from_llama(const llama_vocab * vocab);

    // a vocab from an explicit table (tests); an empty piece is always structural
    static server_cache_align_vocab from_pieces(std::vector<std::string> pieces, const std::vector<llama_token> & structural_ids);

    size_t size() const { return piece.size(); }
    bool is_structural(llama_token t) const;
};

enum server_cache_align_status {
    CACHE_ALIGN_NONE,              // no gain: the best usable point is the id LCP itself (S = N)
    CACHE_ALIGN_ALIGNED,           // a splice past the id LCP
    CACHE_ALIGN_DECLINED_SPANS,    // every usable point past the LCP broke the message-span identity
    CACHE_ALIGN_DECLINED_MEDIA,    // media at or after the splice point in N
};

const char * server_cache_align_status_str(server_cache_align_status s);

enum server_cache_align_stop {
    CACHE_ALIGN_STOP_END,          // the end of C or N
    CACHE_ALIGN_STOP_BYTES,        // first byte mismatch
    CACHE_ALIGN_STOP_STRUCTURAL,   // a structural token against a different id / a byte offset mismatch, or an id
                                   // outside the vocab table
    CACHE_ALIGN_STOP_LITERAL,      // ... where the other side spells the same bytes with ordinary pieces (WARN)
    CACHE_ALIGN_STOP_MEDIA,        // a media placeholder
};

struct server_cache_align_result {
    server_cache_align_status status = CACHE_ALIGN_NONE;
    server_cache_align_stop   stop   = CACHE_ALIGN_STOP_END;

    size_t n_lcp   = 0; // id LCP of C and N (the walk's start)
    size_t n_c     = 0; // tokens of C kept by the splice
    size_t n_n     = 0; // tokens of N they replace
    size_t n_bytes = 0; // bytes of C[0:n_c] (= bytes of N[0:n_n]); rank candidates by this, not by token counts
    size_t n_tried = 0; // usable points rejected by the span check before the chosen one

    // joint token boundaries (n index, c index) from (n_lcp, n_lcp) up to the farthest one the walk reached,
    // ascending; the chosen splice point is one of them
    std::vector<std::pair<size_t, size_t>> syncs;

    // message spans of S (meaningful when delimiters were given; equal to the spans of N when not aligned)
    common_chat_msg_spans spans;

    bool aligned() const { return status == CACHE_ALIGN_ALIGNED; }

    // S length minus N length
    int64_t delta() const { return (int64_t) n_c - (int64_t) n_n; }

    // number of re-segmented spans between the LCP and the splice point (sync intervals whose id runs differ)
    size_t n_spans = 0;
};

struct server_cache_align_params {
    // message delimiters (tokenized); empty = no span identity check (no spans requested)
    const common_chat_msg_delimiters * delims = nullptr;

    // id LCP computed by the caller (server_tokens::get_common_prefix, media-aware); SIZE_MAX = compute it here
    // (raw ids, stops at the first LLAMA_TOKEN_NULL)
    size_t n_lcp = SIZE_MAX;

    // earlier usable points to try after a span-check failure before declining
    int max_span_retries = 8;

    // prefix byte sums of the prompt N (size |N| + 1) for n_bytes; nullptr = summed here over C[0:n_lcp]
    const std::vector<size_t> * prompt_bytes = nullptr;
};

server_cache_align_result server_cache_align(
        const llama_tokens & cached,
        const llama_tokens & prompt,
        const server_cache_align_vocab & vocab,
        const server_cache_align_params & params = {});

// S = C[0:n_c] ++ N[n_n:] (= N when not aligned)
llama_tokens server_cache_align_splice(const llama_tokens & cached, const llama_tokens & prompt, const server_cache_align_result & r);

// a request-coordinate position p (a boundary in N, e.g. cache_pin_tokens) in S coordinates:
//   p <= n_lcp -> p; p >= n_n -> p + delta; inside the re-segmented region -> the C index of the joint boundary at
//   p, or of the previous one when p falls inside a re-segmented token (snapped = true, the caller WARNs)
struct server_cache_align_pos {
    size_t pos     = 0;
    bool   snapped = false;
};
server_cache_align_pos server_cache_align_map_pos(const server_cache_align_result & r, size_t p);

// provenance: only prompts the server tokenized from text are eligible; ids the caller chose (numeric arrays, mixed
// arrays with explicit ids) are authoritative and never rewritten
bool server_cache_align_prompt_from_text(const common_json & subprompt);

struct server_cache_align_gate {
    bool enabled      = false; // --cache-align-bytes
    bool from_text    = false; // server_cache_align_prompt_from_text / chat rendering
    bool cache_prompt = true;
    int  n_cmpl       = 1;     // n > 1 parents/children are excluded
    bool is_child     = false;
};
bool server_cache_align_eligible(const server_cache_align_gate & g);

// prefix byte sums of a token sequence (media placeholders and ids outside the table count 0)
std::vector<size_t> server_cache_align_prefix_bytes(const llama_tokens & toks, const server_cache_align_vocab & vocab);

// the split of tokenize_input_prompts: one provenance flag per subprompt (string / prompt_string = text)
std::vector<bool> server_cache_align_prompts_from_text(const common_json & prompt);

//
// [TAG_CACHE_ALIGN] part 2: the server's matching sites (slot selection, RAM tier, disk find, pending-save match)
//
struct server_tokens;

// one per request; vocab == nullptr = knob off or the request is ineligible: every match is the plain id LCP
struct server_cache_matcher {
    const server_cache_align_vocab   * vocab  = nullptr;
    const common_chat_msg_delimiters * delims = nullptr; // the request's message delimiters (span identity)
    std::vector<size_t>                prompt_bytes;     // prefix byte sums of the request's prompt

    bool active() const { return vocab != nullptr; }
};

// a match of a cached sequence C against the new prompt N in two coordinates:
//   n_n     - new-prompt coverage (f_sim, gain thresholds)
//   n_c     - cached tokens kept (f_keep, loss rules)
//   n_bytes - rank key; off: n_lcp (token count), so every comparison is the v75 one
struct server_cache_match {
    size_t n_lcp   = 0;
    size_t n_c     = 0;
    size_t n_n     = 0;
    size_t n_bytes = 0;
    bool   aligned = false;
};

server_cache_match server_cache_match_tokens(const server_tokens & cached, const server_tokens & prompt, const server_cache_matcher * m);
