// [TAG_CACHE_ALIGN] byte-aligned prompt-cache splice core (P16e part 1): synthetic cases for every boundary rule on a
// toy vocab table, then the three logged v75 incidents replayed from their disk-cache .tok entries on the prod vocab
// (vocab-only load; skipped with a note when the GGUF / .tok files are absent, CACHE_ALIGN_REQUIRE_REAL=1 = fail)
#include "server-cache-align.h"
#include "server-common.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(c)                                                         \
    do {                                                                 \
        if (!(c)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            n_fail++;                                                    \
        }                                                                \
    } while (0)

// toy vocab
enum : llama_token {
    IM_START = 0, IM_END = 1, THINK_END = 2, USER = 3, NL = 4, SP_PAREN = 5, PREC = 6, ACHE = 7, PRE = 8, CACHE = 9,
    LT = 10, BAR = 11, IM = 12, UEND = 13, BAR_GT = 14, LT_SLASH = 15, THINK = 16, GT = 17, B_D0 = 18, B_BC = 19,
    CYR_M = 20, ASSISTANT = 21, HELLO = 22, WORLD = 23, BOX = 24, BOX2 = 25, USER_NL = 26, UNUSED = 27, X = 28,
    N_TOY = 29,
};

static server_cache_align_vocab toy_vocab() {
    std::vector<std::string> p(N_TOY);
    p[IM_START] = "<|im_start|>"; p[IM_END] = "<|im_end|>"; p[THINK_END] = "</think>";
    p[USER] = "user"; p[NL] = "\n"; p[SP_PAREN] = " ("; p[PREC] = "prec"; p[ACHE] = "ache"; p[PRE] = "pre";
    p[CACHE] = "cache"; p[LT] = "<"; p[BAR] = "|"; p[IM] = "im"; p[UEND] = "_end"; p[BAR_GT] = "|>";
    p[LT_SLASH] = "</"; p[THINK] = "think"; p[GT] = ">"; p[B_D0] = "\xD0"; p[B_BC] = "\xBC"; p[CYR_M] = "\xD0\xBC";
    p[ASSISTANT] = "assistant"; p[HELLO] = "hello"; p[WORLD] = " world"; p[BOX] = "\xE2\x95\x90";
    p[BOX2] = "\xE2\x95\x90\xE2\x95\x90"; p[USER_NL] = "user\n"; p[UNUSED] = ""; p[X] = "x";
    return server_cache_align_vocab::from_pieces(p, { IM_START, IM_END, THINK_END });
}

static common_chat_msg_delimiters toy_delims() {
    // "<|im_start|>user\n" tokenized canonically in the toy vocab
    common_chat_msg_delimiters d;
    d.delimiters.push_back({ COMMON_CHAT_ROLE_ASSISTANT, "<|im_start|>assistant", { IM_START, ASSISTANT } });
    d.delimiters.push_back({ COMMON_CHAT_ROLE_USER,      "<|im_start|>user\n",    { IM_START, USER, NL } });
    return d;
}

static std::string bytes_of(const server_cache_align_vocab & v, const llama_tokens & t) {
    std::string s;
    for (auto x : t) {
        s += v.piece[x];
    }
    return s;
}

static void synthetic() {
    const auto v = toy_vocab();
    const auto d = toy_delims();
    server_cache_align_params pd;
    pd.delims = &d;

    // structural flags: listed ids + the empty piece
    CHECK(v.is_structural(IM_END) && v.is_structural(UNUSED) && !v.is_structural(USER) && v.is_structural(-5));

    // 1. forced split ` (` `prec` `ache` vs ` (` `pre` `cache`, next turn: rule (b) at the end of C, delta 0
    {
        const llama_tokens C = { IM_START, ASSISTANT, NL, HELLO, SP_PAREN, PREC, ACHE };
        const llama_tokens N = { IM_START, ASSISTANT, NL, HELLO, SP_PAREN, PRE, CACHE, IM_END, NL, IM_START, USER, NL, X };
        const auto r = server_cache_align(C, N, v, pd);
        CHECK(r.status == CACHE_ALIGN_ALIGNED);
        CHECK(r.n_lcp == 5 && r.n_c == 7 && r.n_n == 7 && r.delta() == 0 && r.n_spans == 1);
        CHECK(r.n_bytes == bytes_of(v, C).size());
        const auto S = server_cache_align_splice(C, N, r);
        CHECK(bytes_of(v, S) == bytes_of(v, N));
        CHECK(llama_tokens(S.begin(), S.begin() + 7) == C);
        CHECK(r.spans.spans.size() == 2 && r.spans.spans[1].role == COMMON_CHAT_ROLE_USER && r.spans.spans[1].pos == 9);
        // no delimiters: same splice, no span check
        const auto r0 = server_cache_align(C, N, v);
        CHECK(r0.aligned() && r0.n_c == 7 && r0.n_n == 7);
    }
    // 2. `═` `═` (cached) vs `══` (canonical): delta +1, spans after the splice shift by delta
    {
        const llama_tokens C = { IM_START, USER, NL, HELLO, BOX, BOX, WORLD };
        const llama_tokens N = { IM_START, USER, NL, HELLO, BOX2, WORLD, IM_END, NL, IM_START, USER, NL, X };
        const auto r = server_cache_align(C, N, v, pd);
        CHECK(r.aligned() && r.n_lcp == 4 && r.n_c == 7 && r.n_n == 6 && r.delta() == 1);
        CHECK(r.spans.spans.size() == 2 && r.spans.spans[0].pos == 0 && r.spans.spans[1].pos == 9); // 8 + 1
        const auto S = server_cache_align_splice(C, N, r);
        CHECK(S.size() == N.size() + 1 && bytes_of(v, S) == bytes_of(v, N));
    }
    // 3. literal vs structural collisions, both directions, `<|im_end|>` and `</think>`: stop before, no splice
    {
        const llama_tokens C = { HELLO, SP_PAREN, PREC, ACHE, LT, BAR, IM, UEND, BAR_GT };     // literal spelling
        const llama_tokens N = { HELLO, SP_PAREN, PRE, CACHE, IM_END, NL };                    // structural
        auto r = server_cache_align(C, N, v, pd);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 2 && r.n_n == 2 && r.stop == CACHE_ALIGN_STOP_LITERAL);
        r = server_cache_align(N, C, v, pd);                                                     // reversed
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 2 && r.stop == CACHE_ALIGN_STOP_LITERAL);
        const llama_tokens C2 = { HELLO, SP_PAREN, PREC, ACHE, THINK_END, X };
        const llama_tokens N2 = { HELLO, SP_PAREN, PRE, CACHE, LT_SLASH, THINK, GT, X };
        r = server_cache_align(C2, N2, v);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 2 && r.stop == CACHE_ALIGN_STOP_LITERAL);
        r = server_cache_align(N2, C2, v);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 2 && r.stop == CACHE_ALIGN_STOP_LITERAL);
        // a structural token that does not start at a joint boundary (`══` cached vs `═` + `<|im_end|>`)
        r = server_cache_align({ HELLO, BOX2, X }, { HELLO, BOX, IM_END }, v);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 1 && r.stop == CACHE_ALIGN_STOP_STRUCTURAL);
    }
    // 4. UTF-8: byte pieces vs the full character, joint boundary after the character
    {
        const llama_tokens C = { HELLO, B_D0, B_BC, WORLD };
        const llama_tokens N = { HELLO, CYR_M, WORLD, IM_END };
        const auto r = server_cache_align(C, N, v);
        CHECK(r.aligned() && r.n_c == 4 && r.n_n == 3 && r.delta() == 1);
        CHECK(r.syncs.size() == 3 && r.syncs[1] == std::make_pair((size_t) 2, (size_t) 3));
        // a pin between the byte pieces (C side) is not a request coordinate; the request's 2 = after `м` -> C 3
        const auto mp = server_cache_align_map_pos(r, 2);
        CHECK(mp.pos == 3 && !mp.snapped);
    }
    // 5. Continue (rule c): prompt ends inside the cached reply
    {
        const llama_tokens C = { HELLO, SP_PAREN, PREC, ACHE, WORLD };
        // at a joint boundary
        auto r = server_cache_align(C, { HELLO, SP_PAREN, PRE, CACHE }, v);
        CHECK(r.aligned() && r.n_c == 4 && r.n_n == 4);
        CHECK(server_cache_align_splice(C, { HELLO, SP_PAREN, PRE, CACHE }, r) == llama_tokens({ HELLO, SP_PAREN, PREC, ACHE }));
        // mid-token (`pre` ends inside `prec`): the last common boundary = the LCP
        r = server_cache_align(C, { HELLO, SP_PAREN, PRE }, v);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 2 && r.n_n == 2);
        // mid-UTF-8 character: N ends after the lead byte, C holds the full character
        r = server_cache_align({ HELLO, CYR_M, WORLD }, { HELLO, B_D0 }, v);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 1 && r.n_n == 1);
        // `═` vs `══` at the end of N
        r = server_cache_align({ HELLO, BOX2 }, { HELLO, BOX }, v);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 1);
    }
    // 6. byte mismatch after a re-segmented span inside the same fragment: the intermediate sync is not usable
    {
        const llama_tokens C = { IM_START, ASSISTANT, NL, HELLO, SP_PAREN, PREC, ACHE, WORLD };
        const llama_tokens N = { IM_START, ASSISTANT, NL, HELLO, SP_PAREN, PRE, CACHE, X, IM_END };
        const auto r = server_cache_align(C, N, v, pd);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 5 && r.n_n == 5 && r.stop == CACHE_ALIGN_STOP_BYTES);
        CHECK(server_cache_align_splice(C, N, r) == N);
    }
    // 7. delimiter ambiguity: the cached `<|im_start|>` `user\n` (explicit-id origin) breaks the user span at the
    //    later point (n 10 / c 9) -> retry the earlier usable point (before the common `<|im_start|>`)
    {
        const llama_tokens C = { HELLO, SP_PAREN, PREC, ACHE, IM_END, NL, IM_START, USER_NL, HELLO, IM_END };
        const llama_tokens N = { HELLO, SP_PAREN, PRE, CACHE, IM_END, NL, IM_START, USER, NL, HELLO, IM_END, NL, IM_START, ASSISTANT, NL };
        auto r = server_cache_align(C, N, v, pd);
        CHECK(r.aligned() && r.n_c == 6 && r.n_n == 6 && r.n_tried == 1);
        CHECK(r.spans.spans.size() == 2 && r.spans.spans[0].pos == 6 && r.spans.spans[0].role == COMMON_CHAT_ROLE_USER);
        // without delimiters the later point wins
        r = server_cache_align(C, N, v);
        CHECK(r.aligned() && r.n_c == 9 && r.n_n == 10);
        // the only usable point breaks the span -> declined, never an assert
        const llama_tokens C2 = { HELLO, IM_START, USER_NL, PREC, ACHE, IM_END };
        const llama_tokens N2 = { HELLO, IM_START, USER, NL, PRE, CACHE, IM_END, NL };
        r = server_cache_align(C2, N2, v, pd);
        CHECK(r.status == CACHE_ALIGN_DECLINED_SPANS && r.n_c == 2 && r.n_n == 2 && r.n_tried == 1);
        CHECK(server_cache_align_splice(C2, N2, r) == N2);
        CHECK(r.spans.spans.size() == 1 && r.spans.spans[0].pos == 1);
        // retries exhausted
        server_cache_align_params p0 = pd;
        p0.max_span_retries = 0;
        r = server_cache_align(C, N, v, p0);
        CHECK(r.status == CACHE_ALIGN_DECLINED_SPANS && r.n_c == 2 && r.n_n == 2);
    }
    // 8. pin mapping: exact boundary, shift past the splice, snap inside a re-segmented token
    {
        const llama_tokens C = { HELLO, BOX2, WORLD };
        const llama_tokens N = { HELLO, BOX, BOX, WORLD, IM_END };
        const auto r = server_cache_align(C, N, v);
        CHECK(r.aligned() && r.n_c == 3 && r.n_n == 4 && r.delta() == -1);
        auto mp = server_cache_align_map_pos(r, 1);
        CHECK(mp.pos == 1 && !mp.snapped);
        mp = server_cache_align_map_pos(r, 2);                 // between `═` and `═` = inside the cached `══`
        CHECK(mp.pos == 1 && mp.snapped);
        mp = server_cache_align_map_pos(r, 3);
        CHECK(mp.pos == 2 && !mp.snapped);
        mp = server_cache_align_map_pos(r, 4);
        CHECK(mp.pos == 3 && !mp.snapped);
        mp = server_cache_align_map_pos(r, 5);
        CHECK(mp.pos == 4 && !mp.snapped);
    }
    // 9. media: the walk stops at a placeholder; media after the splice point in N declines
    {
        const llama_tokens C = { HELLO, SP_PAREN, PREC, ACHE };
        const llama_tokens N = { HELLO, SP_PAREN, PRE, CACHE, IM_END, LLAMA_TOKEN_NULL, LLAMA_TOKEN_NULL };
        auto r = server_cache_align(C, N, v);
        CHECK(r.status == CACHE_ALIGN_DECLINED_MEDIA && r.n_c == 2 && r.n_n == 2);
        r = server_cache_align({ HELLO, LLAMA_TOKEN_NULL, X }, { HELLO, LLAMA_TOKEN_NULL, WORLD }, v);
        CHECK(r.n_lcp == 1 && r.stop == CACHE_ALIGN_STOP_MEDIA && r.status == CACHE_ALIGN_NONE);
    }
    // 10. ids outside the table, empty pieces, identical prompts, caller-given LCP
    {
        auto r = server_cache_align({ HELLO, 999 }, { HELLO, 998 }, v);
        CHECK(r.status == CACHE_ALIGN_NONE && r.stop == CACHE_ALIGN_STOP_STRUCTURAL);
        r = server_cache_align({ HELLO, UNUSED, PREC, ACHE }, { HELLO, PRE, CACHE }, v);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 1);
        const llama_tokens A = { IM_START, USER, NL, HELLO };
        r = server_cache_align(A, A, v, pd);
        CHECK(r.status == CACHE_ALIGN_NONE && r.n_c == 4 && r.n_n == 4 && server_cache_align_splice(A, A, r) == A);
        server_cache_align_params pl;
        pl.n_lcp = 2;
        r = server_cache_align({ HELLO, SP_PAREN, PREC, ACHE }, { HELLO, SP_PAREN, PRE, CACHE }, v, pl);
        CHECK(r.n_lcp == 2 && r.aligned() && r.n_c == 4);
    }
    // 11. provenance + gate (knob off = never)
    {
        CHECK(server_cache_align_prompt_from_text(common_json("text")));
        CHECK(server_cache_align_prompt_from_text(common_json::parse(R"({"prompt_string":"x","multimodal_data":[]})")));
        CHECK(!server_cache_align_prompt_from_text(common_json::parse("[1,2,3]")));
        CHECK(!server_cache_align_prompt_from_text(common_json::parse(R"([1,"a",3])")));
        CHECK(!server_cache_align_prompt_from_text(common_json::parse(R"(["a","b"])")));
        CHECK(!server_cache_align_prompt_from_text(common_json::parse(R"({"prompt_string":[1,2]})")));
        server_cache_align_gate g;
        g.from_text = true;
        CHECK(!server_cache_align_eligible(g));                // knob off (default)
        g.enabled = true;
        CHECK(server_cache_align_eligible(g));
        g.n_cmpl = 2;     CHECK(!server_cache_align_eligible(g)); g.n_cmpl = 1;
        g.is_child = true; CHECK(!server_cache_align_eligible(g)); g.is_child = false;
        g.cache_prompt = false; CHECK(!server_cache_align_eligible(g)); g.cache_prompt = true;
        g.from_text = false; CHECK(!server_cache_align_eligible(g));
    }
}

// ---- part 2: the two-coordinate match the server's matching sites use ----
static void coordinates() {
    const auto v = toy_vocab();
    const auto d = toy_delims();
    const auto st = [](const llama_tokens & t) { return server_tokens(t, false); };

    const llama_tokens N = { HELLO, CYR_M, CYR_M, IM_END, NL, X };
    server_cache_matcher m;
    m.vocab        = &v;
    m.delims       = &d;
    m.prompt_bytes = server_cache_align_prefix_bytes(N, v);
    CHECK(m.prompt_bytes.size() == N.size() + 1 && m.prompt_bytes[3] == 9 && m.prompt_bytes[4] == 19);

    // inactive (knob off / ineligible / nullptr): n_c = n_n = n_bytes = the id LCP - the v75 comparisons
    const llama_tokens C1 = { HELLO, B_D0, B_BC, B_D0, B_BC };   // finer segmentation, fewer bytes
    const llama_tokens C2 = { HELLO, CYR_M, CYR_M, IM_END };     // fewer tokens, more bytes
    for (const server_cache_matcher * off : { (const server_cache_matcher *) nullptr, (const server_cache_matcher *) new server_cache_matcher() }) {
        const auto a = server_cache_match_tokens(st(C1), st(N), off);
        CHECK(a.n_lcp == 1 && a.n_c == 1 && a.n_n == 1 && a.n_bytes == 1 && !a.aligned);
        const auto b = server_cache_match_tokens(st(C2), st(N), off);
        CHECK(b.n_lcp == 4 && b.n_c == 4 && b.n_n == 4 && b.n_bytes == 4);
        delete off;
    }

    // active: two coordinates. C1 keeps 5 cached tokens (retention n_c) for 3 prompt tokens (coverage n_n)
    const auto a = server_cache_match_tokens(st(C1), st(N), &m);
    CHECK(a.aligned && a.n_lcp == 1 && a.n_c == 5 && a.n_n == 3 && a.n_bytes == 9);
    const auto b = server_cache_match_tokens(st(C2), st(N), &m);
    CHECK(!b.aligned && b.n_c == 4 && b.n_n == 4 && b.n_bytes == 19);
    // the rank key: C1 keeps more cached tokens, C2 matches more bytes and covers more of the prompt -> C2
    CHECK(a.n_c > b.n_c && b.n_bytes > a.n_bytes && b.n_n > a.n_n);
    // f_keep = n_c / |C| (1.0 for C1: the whole cache is kept), f_sim = n_n / |N|
    CHECK(a.n_c == C1.size() && b.n_c == C2.size());

    // the media-aware id LCP of server_tokens is the walk's start
    const auto c = server_cache_match_tokens(st({ HELLO, CYR_M, B_D0, B_BC }), st(N), &m);
    CHECK(c.n_lcp == 2 && c.aligned && c.n_c == 4 && c.n_n == 3 && c.n_bytes == 9);

    // provenance per subprompt, mirroring tokenize_input_prompts' split
    auto pf = server_cache_align_prompts_from_text(common_json::parse(R"(["a", {"prompt_string":"b"}, [1,2]])"));
    CHECK(pf.size() == 3 && pf[0] && pf[1] && !pf[2]);
    pf = server_cache_align_prompts_from_text(common_json::parse(R"([1, "a"])")); // one mixed prompt
    CHECK(pf.size() == 1 && !pf[0]);
    pf = server_cache_align_prompts_from_text(common_json("text"));
    CHECK(pf.size() == 1 && pf[0]);
}

// ---- the three logged v75 incidents (design §1 / §7.2) ----

static bool read_tok(const std::string & path, llama_tokens & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    int32_t hdr[3];
    if (!f.read((char *) hdr, sizeof(hdr)) || hdr[2] <= 0) {
        return false;
    }
    out.resize(hdr[2]);
    return (bool) f.read((char *) out.data(), sizeof(llama_token) * out.size());
}

static int real_cases() {
    const char * e_gguf = getenv("CACHE_ALIGN_VOCAB_GGUF");
    const char * e_dir  = getenv("CACHE_ALIGN_TOK_DIR");
    const std::string gguf = e_gguf ? e_gguf : "/mnt/data/models/qwen3.8-flash-next-uncensored/Qwen3.8-Flash-Next-Uncensored.i1-Q4_K_M.gguf";
    const std::string dir  = e_dir ? e_dir : "/mnt/data/piagent-slots/prompt-cache/qwen38-flash-next-uncensored-brain-1m/ns-36557823e9dde84e/";
    const bool require = getenv("CACHE_ALIGN_REQUIRE_REAL") != nullptr;

    struct incident { const char * key; size_t reply_start, n_cache, id_lcp; };
    const incident cases[] = {
        { "ba543f058ab248da",  89623, 124316,  92653 },
        { "95ea69c1eb053e22", 234824, 257289, 238290 },
        { "a32a45b7e9f14c89", 306875, 336262, 310735 },
    };

    std::ifstream probe(gguf);
    if (!probe) {
        fprintf(stderr, "real cases: SKIPPED (no vocab GGUF %s)\n", gguf.c_str());
        return require ? 1 : 0;
    }

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(gguf.c_str(), mp);
    if (model == nullptr) {
        fprintf(stderr, "real cases: cannot load the vocab of %s\n", gguf.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto v = server_cache_align_vocab::from_llama(vocab);

    common_chat_msg_delimiters d;
    d.add(COMMON_CHAT_ROLE_ASSISTANT, "<|im_start|>assistant");
    d.add(COMMON_CHAT_ROLE_TOOL,      "<|im_start|>user\n<tool_response>");
    d.add(COMMON_CHAT_ROLE_TOOL,      "<|im_start|>tool_response");
    d.add(COMMON_CHAT_ROLE_USER,      "<|im_start|>user");
    d.add(COMMON_CHAT_ROLE_SYSTEM,    "<|im_start|>system");
    d.tokenize(vocab);
    server_cache_align_params pd;
    pd.delims = &d;

    int rc = 0;
    for (const auto & c : cases) {
        llama_tokens C;
        if (!read_tok(dir + c.key + ".tok", C)) {
            fprintf(stderr, "real case %s: SKIPPED (no .tok)\n", c.key);
            if (require) {
                rc = 1;
            }
            continue;
        }
        CHECK(C.size() == c.n_cache);
        // the client re-sends the reply as text: N = C up to the last structural token at/before the reply start,
        // then the canonical tokenization of the same bytes + the next turn (splice_sim.py)
        size_t s = c.reply_start;
        while (s > 0 && !v.is_structural(C[s])) {
            s--;
        }
        s++;
        const std::string tail = bytes_of(v, llama_tokens(C.begin() + s, C.end()));
        llama_tokens N(C.begin(), C.begin() + s);
        const auto can = common_tokenize(vocab, tail + "<|im_end|>\n<|im_start|>user\nnext turn", false, true);
        N.insert(N.end(), can.begin(), can.end());

        const auto r = server_cache_align(C, N, v, pd);
        const auto S = server_cache_align_splice(C, N, r);
        fprintf(stderr, "real case %s: |C| %zu |N| %zu id-lcp %zu (lost %zu) -> %s n_c %zu n_n %zu delta %lld spans %zu, "
                "dropped %zu, message spans %zu\n", c.key, C.size(), N.size(), r.n_lcp, C.size() - r.n_lcp,
                server_cache_align_status_str(r.status), r.n_c, r.n_n, (long long) r.delta(), r.n_spans,
                C.size() - r.n_c, r.spans.spans.size());
        CHECK(r.n_lcp == c.id_lcp);           // the bug, reproduced
        CHECK(r.aligned());
        CHECK(r.n_c == C.size());             // 0 dropped
        CHECK(r.n_spans > 0);
        CHECK(bytes_of(v, S) == bytes_of(v, N));
        CHECK(llama_tokens(S.begin(), S.begin() + C.size()) == C);
        CHECK(r.spans.spans.size() == d.split(N).spans.size());
    }

    llama_model_free(model);
    llama_backend_free();
    return rc;
}

int main() {
    synthetic();
    coordinates();
    const int rc = real_cases();
    if (n_fail > 0 || rc != 0) {
        fprintf(stderr, "test-cache-align: %d check(s) failed, real rc %d\n", n_fail, rc);
        return 1;
    }
    fprintf(stderr, "test-cache-align: OK\n");
    return 0;
}
