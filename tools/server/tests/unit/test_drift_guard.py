# [TAG_DRIFT_GUARD] duplicate reasoning close guard, end to end on a small model whose vocabulary has a "</think>"
# special token (the guard arms only on that): MiniCPM5-1B Q4_K_M from /models (the llama-dev mount), CPU.
# Override with DRIFT_GUARD_TEST_MODEL. A grammar forces the token sequence with the special id:
#   "R" </think> "A" </think> " B"
# The first close ends the reasoning, the second one is the duplicate close in the content phase.
# Not here: /v1/messages (the Anthropic conversion passes no grammar / drift_guard - its serializers are covered in
# tests/test-drift-guard.cpp section D) and a dangling partial UTF-8 byte before the trigger (llama-grammar rejects a
# token terminal after a partial UTF-8 byte token, so the sequence cannot be forced).
import os
import pytest
from utils import *

MODEL = os.environ.get("DRIFT_GUARD_TEST_MODEL", "/models/minicpm/MiniCPM5-1B-Q4_K_M.gguf")

pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason=f"drift guard e2e model {MODEL} not found")

server: ServerProcess


def make_server() -> ServerProcess:
    s = ServerProcess()
    s.model_file = MODEL
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_alias = "drift-guard-test"
    s.n_ctx = 1024
    s.n_batch = 64
    s.n_slots = 1
    s.n_predict = 32
    s.seed = 42
    s.jinja = True
    return s


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = make_server()


def _id(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    toks = res.body["tokens"]
    assert len(toks) == 1, f"{text!r} is not one token: {toks}"
    return toks[0]


def _grammar(*parts) -> str:
    # parts: str literals or int token ids
    items = [f"<[{p}]>" if isinstance(p, int) else '"' + p.replace('"', '\\"') + '"' for p in parts]
    return "root ::= " + " ".join(items)


def _completion(extra, grammar=None):
    close = _id("</think>")
    data = {"prompt": "Hello", "n_predict": 32, "temperature": 0.0,
            "grammar": grammar or _grammar("R", close, "A", close, " B"),
            "reasoning_budget_start_tag": "<think>", "reasoning_budget_end_tags": ["</think>"], **extra}
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return res.body


def test_guard_off_by_default_unchanged():
    server.start()
    a = _completion({})
    b = _completion({"drift_guard": False})
    assert a["generation_settings"]["drift_guard"] is False
    assert a["stop_type"] in ("eos", "limit")
    assert a["content"].endswith(" B")
    assert a["content"] == b["content"]
    assert a["tokens_predicted"] == b["tokens_predicted"]


def test_guard_stops_before_duplicate_close():
    server.start()
    close = _id("</think>")
    on = _completion({"drift_guard": True, "return_tokens": True})
    assert on["stop_type"] == "think_reenter"
    assert on["content"].endswith("A")
    assert " B" not in on["content"]
    # the suppressed close is neither visible nor counted
    assert on["tokens"].count(close) == 1
    assert on["tokens_predicted"] == len(on["tokens"])


def test_guard_needs_the_think_close_special():
    server.start()
    # a reasoning end tag that is not the "</think>" special leaves the guard inactive
    a = _id("A")
    res = _completion({"drift_guard": True, "reasoning_budget_end_tags": ["A"]}, grammar=_grammar("R", a, "R", a, " B"))
    assert res["stop_type"] in ("eos", "limit")
    assert res["content"].endswith(" B")


def test_guard_flag_default_and_reasoning_none():
    server.extra_args = ["--drift-guard", "stop"]
    server.start()
    on = _completion({})
    assert on["generation_settings"]["drift_guard"] is True
    assert on["stop_type"] == "think_reenter"
    off = _completion({"drift_guard": False})
    assert off["stop_type"] in ("eos", "limit")
    none = _completion({"reasoning_format": "none"})
    assert none["stop_type"] in ("eos", "limit")


def test_guard_stream_flushes_held_stop_prefix():
    server.start()
    # "A" is a prefix of the stop string "AZ": it is held back, then the trigger arrives; the held bytes are content
    # (the stop string never completed) and must reach the client
    close = _id("</think>")
    data = {"prompt": "Hello", "n_predict": 32, "temperature": 0.0, "grammar": _grammar("R", close, "A", close, " B"),
            "reasoning_budget_end_tags": ["</think>"], "drift_guard": True, "stream": True, "stop": ["AZ"]}
    content = ""
    last = None
    for chunk in server.make_stream_request("POST", "/completion", data=data):
        content += chunk.get("content", "")
        last = chunk
    assert last is not None and last["stop"] is True
    assert last["stop_type"] == "think_reenter"
    assert content.endswith("A")


@pytest.mark.parametrize("stream", [False, True])
def test_guard_oai_chat(stream):
    server.start()
    close = _id("</think>")
    data = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "temperature": 0.0,
            "grammar": _grammar("R", close, "A", close, " B"),
            "reasoning_budget_start_tag": "<think>", "reasoning_budget_end_tags": ["</think>"],
            "drift_guard": True, "stream": stream}
    if not stream:
        res = server.make_request("POST", "/chat/completions", data=data)
        assert res.status_code == 200, res.body
        choice = res.body["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["stop_type"] == "think_reenter"
        assert " B" not in (choice["message"].get("content") or "")
    else:
        finish = None
        for chunk in server.make_stream_request("POST", "/chat/completions", data=data):
            for c in chunk.get("choices", []):
                if c.get("finish_reason") is not None:
                    finish = c
        assert finish is not None
        assert finish["finish_reason"] == "stop"
        assert finish["stop_type"] == "think_reenter"


def test_guard_oai_chat_untriggered_has_no_stop_type():
    server.start()
    close = _id("</think>")
    data = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "temperature": 0.0,
            "grammar": _grammar("R", close, "A", close, " B"),
            "reasoning_budget_end_tags": ["</think>"], "drift_guard": False}
    res = server.make_request("POST", "/chat/completions", data=data)
    assert res.status_code == 200, res.body
    choice = res.body["choices"][0]
    assert "stop_type" not in choice


def test_guard_oai_text_completion():
    server.start()
    close = _id("</think>")
    data = {"prompt": "Hello", "max_tokens": 32, "temperature": 0.0, "grammar": _grammar("R", close, "A", close, " B"),
            "reasoning_budget_end_tags": ["</think>"], "drift_guard": True}
    res = server.make_request("POST", "/v1/completions", data=data)
    assert res.status_code == 200, res.body
    choice = res.body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["stop_type"] == "think_reenter"
