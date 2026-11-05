// [TAG_DRIFT_GUARD] duplicate reasoning close guard (server_drift_guard)
//
// A. state machine on SYNTHETIC ids (no vocabulary): trigger at the duplicate close, natural EOS, drift without a
//    duplicate close, tool calls (disarm), continuation prefills (closed / open / thinking off), inactive, and the
//    single walk of a speculative block incl. a close that is also an EOG
// B. template tags (no vocabulary): the Flash-Next template the composes load gives "<think>" / {"</think>",
//    "<tool_call>"}, the generation prompts the guard is initialised from, and no message delimiter contains
//    "</think>"; a missing template file is a FAIL
// C. the real vocabulary (vocab_only load of the GGUF given as argv[1] or LLAMA_DRIFT_GUARD_VOCAB; default = the prod
//    brain path inside the llama-dev container): the think-close resolves to one USER_DEFINED "</think>" token, spelled
//    pieces never contain it, the tokenizer-side gap. Without the GGUF: FAIL, or SKIP (exit 77) only when
//    DRIFT_GUARD_ALLOW_SKIP=1 is set
// D. serializers (no vocabulary): think_reenter -> native stop_type, OAI text / chat / chat stream finish_reason "stop"
//    + stop_type in the choice (stream: choices[0] of the final chunk), Anthropic end_turn + top-level stop_type
//    (stream: the message_delta data); an EOS reply carries no stop_type anywhere
#include "server-common.h"
#include "server-task.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
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

static llama_tokens cat(std::initializer_list<llama_tokens> parts) {
    llama_tokens out;
    for (const auto & p : parts) {
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

// index of the first trigger in the generated tokens, -1 if none (one token at a time, the non-speculative path)
static int run(server_drift_guard g, const llama_tokens & gen) {
    for (size_t i = 0; i < gen.size(); ++i) {
        if (g.step(gen[i]) == server_drift_guard::EV_TRIGGER) {
            return (int) i;
        }
    }
    return -1;
}

// synthetic vocabulary: ids only
enum : llama_token { T_IM = 1, T_ASSIST, T_NL, T_OPEN, T_CLOSE, T_TCALL, T_EOS, T_A, T_B, T_C, T_ALT_EOG };

static const std::vector<llama_tokens> ENDS = { { T_CLOSE }, { T_TCALL } };
static const llama_tokens GP = { T_IM, T_ASSIST, T_NL, T_OPEN, T_NL }; // "<|im_start|>assistant\n<think>\n"

static server_drift_guard armed(const llama_tokens & prefill = GP) {
    server_drift_guard g;
    g.arm(T_OPEN, T_CLOSE, ENDS, prefill);
    return g;
}

static void test_state_machine() {
    server_drift_guard g0 = armed();
    CHECK(g0.phase == server_drift_guard::REASONING);
    CHECK(g0.disarm_ids.size() == 1 && g0.disarm_ids[0] == T_TCALL);

    // the incident shape
    CHECK(run(g0, { T_A, T_CLOSE, T_NL, T_B, T_C, T_CLOSE, T_A }) == 5);
    // natural EOS; drift prose without a duplicate close
    CHECK(run(g0, { T_A, T_CLOSE, T_NL, T_B, T_EOS }) == -1);
    CHECK(run(g0, { T_A, T_CLOSE, T_NL, T_B, T_C, T_B, T_C, T_EOS }) == -1);
    // a second <think> in the content does not trigger by itself; its close does
    CHECK(run(g0, { T_A, T_CLOSE, T_B, T_OPEN, T_C, T_CLOSE }) == 5);
    // tool calls: <tool_call> ends the reasoning and disarms; also after the content
    CHECK(run(g0, { T_A, T_TCALL, T_B, T_CLOSE, T_EOS }) == -1);
    CHECK(run(g0, { T_A, T_CLOSE, T_NL, T_TCALL, T_B, T_CLOSE, T_EOS }) == -1);

    // continuation prefills
    {
        // reasoning closed in the prefill ("...<think>\nR\n</think>\n\nHello, ")
        server_drift_guard g = armed(cat({ GP, { T_A, T_NL, T_CLOSE, T_NL, T_B } }));
        CHECK(g.phase == server_drift_guard::CONTENT);
        CHECK(run(g, { T_C, T_EOS }) == -1);
        CHECK(run(g, { T_C, T_CLOSE }) == 1);
        // a prefill that quotes the close twice (tokenizer gap) is never a trigger, only the phase
        server_drift_guard g2 = armed(cat({ GP, { T_A, T_CLOSE, T_B, T_CLOSE, T_C } }));
        CHECK(g2.phase == server_drift_guard::CONTENT);
    }
    {
        // reasoning still open in the prefill
        server_drift_guard g = armed(cat({ GP, { T_A } }));
        CHECK(g.phase == server_drift_guard::REASONING);
        CHECK(run(g, { T_B, T_CLOSE, T_NL, T_C, T_EOS }) == -1);
    }
    {
        // thinking disabled: the template closes an empty block in the prefill
        server_drift_guard g = armed(cat({ GP, { T_NL, T_CLOSE, T_NL } }));
        CHECK(g.phase == server_drift_guard::CONTENT);
        CHECK(run(g, { T_A, T_CLOSE }) == 1);
    }
    {
        // no start tag (lenient): the first close is the reasoning close
        server_drift_guard g;
        g.arm(LLAMA_TOKEN_NULL, T_CLOSE, ENDS, { T_IM, T_ASSIST, T_NL });
        CHECK(g.phase == server_drift_guard::PRE);
        CHECK(run(g, { T_A, T_CLOSE, T_B, T_CLOSE }) == 3);
    }
    {
        // inactive: no resolved close
        server_drift_guard g;
        g.arm(T_OPEN, LLAMA_TOKEN_NULL, ENDS, GP);
        CHECK(!g.active());
        CHECK(run(g, { T_CLOSE, T_CLOSE }) == -1);
        CHECK(server_drift_guard::resolve_close(nullptr, ENDS) == LLAMA_TOKEN_NULL);
    }

    // the single walk of a speculative block (the server uses its events for the trim AND the stop)
    {
        const auto is_eog = [](llama_token t) { return t == T_EOS || t == T_ALT_EOG; };
        using ev = server_drift_guard::event_t;

        server_drift_guard g = armed();
        auto e = g.walk({ T_A, T_CLOSE, T_B, T_CLOSE, T_C }, is_eog);
        CHECK(e.size() == 4);
        CHECK(e[1] == ev::EV_CLOSED && e[3] == ev::EV_TRIGGER);

        // an EOG before the duplicate close ends the walk (the reply ends at the EOG): no trigger
        g = armed(cat({ GP, { T_A, T_CLOSE } }));
        e = g.walk({ T_B, T_EOS, T_CLOSE }, is_eog);
        CHECK(e.size() == 2 && e.back() == ev::EV_NONE);

        // a close that is ALSO an EOG: in the content phase the guard wins (checked before the EOG) ...
        const auto close_is_eog = [](llama_token t) { return t == T_CLOSE || t == T_EOS; };
        g = armed(cat({ GP, { T_A, T_CLOSE } }));
        e = g.walk({ T_B, T_CLOSE, T_C }, close_is_eog);
        CHECK(e.size() == 2 && e.back() == ev::EV_TRIGGER);
        // ... in the reasoning phase it is the reasoning close, and the EOG ends the walk there
        g = armed();
        e = g.walk({ T_A, T_CLOSE, T_B, T_CLOSE }, close_is_eog);
        CHECK(e.size() == 2 && e.back() == ev::EV_CLOSED);
        CHECK(g.phase == server_drift_guard::CONTENT);

        // the walk and the one-token path agree on every token
        const llama_tokens blk = { T_A, T_NL, T_CLOSE, T_B, T_TCALL, T_CLOSE, T_C };
        server_drift_guard ga = armed(), gb = armed();
        e = ga.walk(blk, is_eog);
        CHECK(e.size() == blk.size());
        for (size_t i = 0; i < blk.size() && i < e.size(); ++i) {
            CHECK(gb.step(blk[i]) == e[i]);
        }
        // an inactive guard walks to the end with no events
        server_drift_guard off;
        e = off.walk(blk, is_eog);
        CHECK(e.size() == blk.size());
    }

    // the trim decision for an interior trigger (block index i) of a speculative round
    {
        using T = common_speculative_type;
        const std::vector<T> mtp    = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        const std::vector<T> eagle3 = { COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 };
        // MTP: PART always, RS within the snapshot depth and among the fresh drafts
        CHECK( server_drift_guard_can_trim(COMMON_CONTEXT_SEQ_RM_TYPE_PART, 4, 0, 1, 0, mtp));
        CHECK( server_drift_guard_can_trim(COMMON_CONTEXT_SEQ_RM_TYPE_RS,   4, 4, 1, 0, mtp));
        CHECK(!server_drift_guard_can_trim(COMMON_CONTEXT_SEQ_RM_TYPE_RS,   5, 4, 1, 0, mtp)); // deeper than n_rs_seq
        CHECK(!server_drift_guard_can_trim(COMMON_CONTEXT_SEQ_RM_TYPE_RS,   4, 4, 1, 2, mtp)); // inside the committed tokens
        CHECK( server_drift_guard_can_trim(COMMON_CONTEXT_SEQ_RM_TYPE_RS,   4, 4, 2, 2, mtp));
        CHECK(!server_drift_guard_can_trim(COMMON_CONTEXT_SEQ_RM_TYPE_FULL, 4, 4, 1, 0, mtp));
        CHECK( server_drift_guard_can_trim(COMMON_CONTEXT_SEQ_RM_TYPE_PART, 4, 0, 1, 0, { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE }));
        CHECK( server_drift_guard_can_trim(COMMON_CONTEXT_SEQ_RM_TYPE_PART, 4, 0, 1, 0, { COMMON_SPECULATIVE_TYPE_NGRAM_MOD, COMMON_SPECULATIVE_TYPE_DRAFT_MTP }));
        // eagle3 (and dflash / dspark): never - the no-trim path (decoded suffix kept, no carry rebase), on any target
        for (const auto st : { COMMON_CONTEXT_SEQ_RM_TYPE_PART, COMMON_CONTEXT_SEQ_RM_TYPE_RS }) {
            CHECK(!server_drift_guard_can_trim(st, 4, 4, 1, 0, eagle3));
            CHECK(!server_drift_guard_can_trim(st, 4, 4, 1, 0, { COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 }));
            CHECK(!server_drift_guard_can_trim(st, 4, 4, 1, 0, { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH }));
            CHECK(!server_drift_guard_can_trim(st, 4, 4, 1, 0, { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK }));
        }
    }
}

static common_chat_templates_ptr flash_next_templates() {
    const char * path = "models/templates/qwen3.8-flash-next-preserve-ws.jinja";
    std::ifstream f(path);
    if (!f.good()) {
        fprintf(stderr, "FAIL: template file %s missing (run from the source root)\n", path);
        n_fail++;
        return nullptr;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return common_chat_templates_ptr(common_chat_templates_init(nullptr, ss.str()));
}

static void test_template_tags() {
    auto tmpls = flash_next_templates();
    if (!tmpls) {
        return;
    }
    common_chat_msg user;
    user.role    = "user";
    user.content = "Hi";

    common_chat_templates_inputs inputs;
    inputs.messages              = { user };
    inputs.add_generation_prompt = true;
    inputs.enable_thinking       = true;
    const auto cp = common_chat_templates_apply(tmpls.get(), inputs);

    CHECK(cp.thinking_start_tag == "<think>");
    CHECK(cp.thinking_end_tags.size() == 2 && cp.thinking_end_tags[0] == "</think>" && cp.thinking_end_tags[1] == "<tool_call>");
    CHECK(cp.generation_prompt == "<|im_start|>assistant\n<think>\n");
    CHECK(!cp.message_delimiters.delimiters.empty());
    for (const auto & d : cp.message_delimiters.delimiters) {
        CHECK(d.delimiter.find("</think>") == std::string::npos);
    }

    common_chat_msg prefill;
    prefill.role              = "assistant";
    prefill.content           = "Hello, ";
    prefill.reasoning_content = "I'm thinking\n\n";
    common_chat_templates_inputs ci;
    ci.messages               = { user, prefill };
    ci.add_generation_prompt  = false;
    ci.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
    ci.enable_thinking        = true;
    CHECK(common_chat_templates_apply(tmpls.get(), ci).generation_prompt ==
          "<|im_start|>assistant\n<think>\nI'm thinking\n\n</think>\n\nHello, ");
    ci.continue_final_message = COMMON_CHAT_CONTINUATION_REASONING;
    CHECK(common_chat_templates_apply(tmpls.get(), ci).generation_prompt == "<|im_start|>assistant\n<think>\nI'm thinking\n\n");
}

static void test_serializers() {
    for (const stop_type st : { STOP_TYPE_THINK_REENTER, STOP_TYPE_EOS }) {
        const bool tr = st == STOP_TYPE_THINK_REENTER;
        server_task_result_cmpl_final r;
        r.stop                 = st;
        r.stream               = false;
        r.content              = "A";
        r.oaicompat_msg.role    = "assistant";
        r.oaicompat_msg.content = "A";

        const json nat = r.to_json_non_oaicompat();
        CHECK(nat.at("stop_type") == (tr ? "think_reenter" : "eos"));

        const json txt = r.to_json_oaicompat();
        CHECK(txt.at("choices")[0].at("finish_reason") == "stop");
        CHECK(txt.at("choices")[0].contains("stop_type") == tr);

        const json chat = r.to_json_oaicompat_chat();
        CHECK(chat.at("choices")[0].at("finish_reason") == "stop");
        CHECK(chat.at("choices")[0].contains("stop_type") == tr);
        if (tr) {
            CHECK(chat.at("choices")[0].at("stop_type") == "think_reenter");
        }

        r.stream = true;
        const json chunks = r.to_json_oaicompat_chat_stream();
        CHECK(chunks.is_array() && !chunks.empty());
        int n_fin = 0;
        for (const auto & c : chunks) {
            if (!c.contains("choices") || c.at("choices").empty()) {
                continue; // the usage chunk
            }
            const json & ch = c.at("choices")[0];
            if (ch.at("finish_reason").is_null()) {
                CHECK(!ch.contains("stop_type"));
                continue;
            }
            n_fin++;
            CHECK(ch.at("finish_reason") == "stop");
            CHECK(ch.contains("stop_type") == tr);
        }
        CHECK(n_fin == 1);
        r.stream = false;

        const json an = r.to_json_anthropic();
        CHECK(an.at("stop_reason") == "end_turn");
        CHECK(an.contains("stop_type") == tr);

        const json evs = r.to_json_anthropic_stream();
        bool seen = false;
        for (const auto & e : evs) {
            if (e.at("event") == "message_delta") {
                seen = true;
                CHECK(e.at("data").at("delta").at("stop_reason") == "end_turn");
                CHECK(e.at("data").contains("stop_type") == tr);
            }
        }
        CHECK(seen);
    }
}

// returns false when the vocabulary is unavailable
static bool test_real_vocab(const std::string & path) {
    {
        std::ifstream probe(path);
        if (!probe.good()) {
            return false;
        }
    }
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "FAIL: cannot load the vocabulary of %s\n", path.c_str());
        n_fail++;
        return true;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const auto tok = [&](const std::string & t, bool special) { return common_tokenize(vocab, t, false, special); };

    const std::vector<llama_tokens> ends = { tok("</think>", true), tok("<tool_call>", true) };
    const llama_token close = server_drift_guard::resolve_close(vocab, ends);
    CHECK(close != LLAMA_TOKEN_NULL);
    fprintf(stderr, "think-close resolved from the vocabulary: id %d, attr 0x%x\n", close, (unsigned) llama_vocab_get_attr(vocab, close));
    CHECK(llama_vocab_get_attr(vocab, close) & LLAMA_TOKEN_ATTR_USER_DEFINED);

    // only a "</think>" special arms: <tool_call> (special, other text) and a NORMAL token do not
    CHECK(server_drift_guard::resolve_close(vocab, { tok("<tool_call>", true) }) == LLAMA_TOKEN_NULL);
    CHECK(server_drift_guard::resolve_close(vocab, { tok("hello", false) }) == LLAMA_TOKEN_NULL);
    CHECK(server_drift_guard::resolve_close(vocab, { tok("</end>", true) }) == LLAMA_TOKEN_NULL);

    // spelled pieces ("</", "think", ">") never contain the special id -> a spelled close never triggers
    const llama_tokens spelled = cat({ tok("</", false), tok("think", false), tok(">", false) });
    for (llama_token t : spelled) {
        CHECK(t != close);
    }
    server_drift_guard g;
    g.arm(tok("<think>", true).front(), close, ends, tok("<|im_start|>assistant\n<think>\n", true));
    CHECK(g.phase == server_drift_guard::REASONING);
    CHECK(run(g, cat({ tok("R", false), { close }, tok("\n\nUse `", false), spelled, tok("`.", false) })) == -1);
    CHECK(run(g, cat({ tok("R", false), { close }, tok("\n\nA", false), { close } })) >= 0);

    // the tokenizer-side gap: USER_DEFINED -> the TEXT maps to the special id even without parse_special
    bool text_special = false;
    for (llama_token t : tok("Use `</think>` here", false)) {
        text_special |= t == close;
    }
    CHECK(text_special);

    llama_model_free(model);
    llama_backend_free();
    return true;
}

int main(int argc, char ** argv) {
    test_state_machine();
    test_template_tags();
    test_serializers();

    const char * env = getenv("LLAMA_DRIFT_GUARD_VOCAB");
    const std::string vocab_path = argc > 1 ? argv[1] : env != nullptr ? env
        : "/models/qwen3.8-flash-next-uncensored/Qwen3.8-Flash-Next-Uncensored.i1-Q4_K_M.gguf";
    const bool have_vocab = test_real_vocab(vocab_path);

    if (n_fail != 0) {
        fprintf(stderr, "test-drift-guard: %d FAILED\n", n_fail);
        return 1;
    }
    if (!have_vocab) {
        const char * allow = getenv("DRIFT_GUARD_ALLOW_SKIP");
        if (allow != nullptr && strcmp(allow, "1") == 0) {
            printf("test-drift-guard: synthetic + template sections OK; REAL-VOCAB SECTION SKIPPED: %s not found "
                   "(set LLAMA_DRIFT_GUARD_VOCAB or pass the GGUF path)\n", vocab_path.c_str());
            return 77;
        }
        fprintf(stderr, "test-drift-guard: FAIL - the real-vocab section needs %s (set LLAMA_DRIFT_GUARD_VOCAB or pass "
                        "the GGUF path; DRIFT_GUARD_ALLOW_SKIP=1 turns this into SKIP, exit 77)\n", vocab_path.c_str());
        return 1;
    }
    printf("test-drift-guard: OK\n");
    return 0;
}
