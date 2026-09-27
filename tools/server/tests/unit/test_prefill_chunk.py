import os
import re
import tempfile
import threading
import time
import pytest
from utils import *

# [TAG_PRIO_CHUNK] (E11) LLAMA_PREFILL_CHUNK_WITH_DECODE = C: while a slot decodes, the prompt tokens of the other
# (non-exempt) slots are capped at C per step; with no decoder nothing changes.
#
# Assertions on log lines run once at the end, after a graceful stop has flushed the log (the log is written by a
# worker thread, see test_kv_unified_purge.py).

server = ServerPreset.tinyllama2()

N_BATCH = 256
C_REQ = 32      # requested; values in (0, 64) are raised to 64
C_EFF = 64

SPLIT_RE = re.compile(r"prio-chunk: split decode view \[0, (\d+)\) \+ prompt view \[(\d+), (\d+)\)")
PRIO_RE = re.compile(r"prio-chunk: n_gen_slots=(\d+) n_gen_tokens=(\d+) n_prompt_tokens=(\d+) budget=(\d+) slots=\[([^\]]*)\]")


def words(n: int, seed: int) -> str:
    # fixed-width pseudo-random "words", ~6 tokens each on stories260K (one per digit)
    return " ".join(f"w{(seed * 7919 + i * 104729) % 99991:05d}" for i in range(n))


PROMPT_1 = words(100, 11)   # ~600 tokens
PROMPT_2 = words(100, 12)


def make_server(env: dict | None, n_slots: int = 2) -> ServerProcess:
    s = ServerPreset.tinyllama2()
    s.n_slots = n_slots
    s.n_ctx = 2048 * n_slots
    s.n_batch = N_BATCH
    s.n_ubatch = N_BATCH
    s.temperature = 0.0
    s.server_slots = True
    s.server_metrics = True
    s.env = env
    fd, s.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    return s


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = make_server(None)
    yield
    server.stop()


def full_log(s: ServerProcess) -> str:
    s.stop()   # SIGTERM -> graceful exit -> the log is flushed
    with open(s.log_path) as f:
        return f.read()


def metric(s: ServerProcess, name: str) -> int:
    body = s.make_request("GET", "/metrics").body
    line = [l for l in body.splitlines() if l.startswith("llamacpp:" + name + " ")][0]
    return int(float(line.split(" ")[1]))


def n_decoded(s: ServerProcess, slot_id: int) -> int:
    res = s.make_request("GET", "/slots")
    assert res.status_code == 200
    for slot in res.body:
        if slot["id"] == slot_id and slot["is_processing"] and "next_token" in slot:
            return slot["next_token"][0]["n_decoded"]
    return 0


def co_schedule(s: ServerProcess, prompt: str, n_predict: int = 4, n_gen: int = 1024, extra: dict | None = None) -> tuple[dict, dict]:
    """slot 0 generates a long answer; once it is decoding, slot 1 gets `prompt`. Returns both response bodies."""
    out = {}

    def gen():
        res = s.make_request("POST", "/completion", data={
            "prompt": "Once upon a time",
            "id_slot": 0,
            "n_predict": n_gen,
            "ignore_eos": True,
            "cache_prompt": False,
            **(extra or {}),
        }, timeout=300)
        out["gen"] = res

    t = threading.Thread(target=gen)
    t.start()
    t_end = time.time() + 30
    while n_decoded(s, 0) < 4:
        assert time.time() < t_end, "slot 0 never started decoding"
        time.sleep(0.005)
    res1 = s.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 1,
        "n_predict": n_predict,
        "cache_prompt": False,
        **(extra or {}),
    }, timeout=300)
    assert res1.status_code == 200
    t.join()
    assert out["gen"].status_code == 200
    # slot 0 was still decoding when slot 1's prompt finished - the whole prefill was co-scheduled
    assert out["gen"].body["timings"]["predicted_n"] == n_gen
    return out["gen"].body, res1.body


def prio_lines(log: str) -> list[tuple[int, int, int, int, str]]:
    return [(int(m[0]), int(m[1]), int(m[2]), int(m[3]), m[4]) for m in PRIO_RE.findall(log)]


# (a) a prompt that arrives while another slot decodes is fed in chunks of <= C; without a decoder it is not
def test_chunked_while_decoding():
    global server
    server = make_server({"LLAMA_PREFILL_CHUNK_WITH_DECODE": str(C_REQ)})
    server.debug = True     # the split-view line is DBG
    server.start()

    _, body1 = co_schedule(server, PROMPT_1)
    n_prompt_1 = body1["timings"]["prompt_n"]
    assert n_prompt_1 > 4 * C_EFF

    # task 4: counters of the co-scheduled phase
    steps_1    = metric(server, "prefill_chunked_steps_total")
    tokens_1   = metric(server, "prefill_chunked_tokens_total")
    deferred_1 = metric(server, "prefill_chunk_deferred_token_steps_total")
    assert steps_1 >= -(-n_prompt_1 // C_EFF)
    assert tokens_1 == n_prompt_1
    assert deferred_1 > 0

    # slot 0 is done: a new prompt on slot 1 runs in full n_batch steps again
    dec0 = metric(server, "n_decode_total")
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT_2, "id_slot": 1, "n_predict": 1, "cache_prompt": False,
    })
    assert res.status_code == 200
    n_prompt_2 = res.body["timings"]["prompt_n"]
    n_steps_2 = metric(server, "n_decode_total") - dec0
    assert n_steps_2 <= -(-n_prompt_2 // N_BATCH) + 1     # un-chunked: ceil(n / n_batch) prompt decodes
    assert n_prompt_2 / n_steps_2 > C_EFF                 # i.e. its steps are > C
    assert metric(server, "prefill_chunked_steps_total") == steps_1

    log = full_log(server)
    assert f"LLAMA_PREFILL_CHUNK_WITH_DECODE = {C_EFF} (requested {C_REQ}, n_batch = {N_BATCH}, on)" in log
    lines = prio_lines(log)
    assert len(lines) >= 1
    assert len(lines) == steps_1
    for n_gen_slots, n_gen_tokens, n_prompt, budget, slots in lines:
        assert n_gen_slots >= 1 and n_gen_tokens >= 1
        assert budget == C_EFF
        assert 0 < n_prompt <= C_EFF
        assert slots.startswith("1:")
    # every prompt token of slot 1 went through a budgeted step
    assert sum(l[2] for l in lines) == n_prompt_1

    print(f"EVIDENCE (a): n_prompt_1={n_prompt_1} prio_lines={len(lines)} first={lines[0]} last={lines[-1]} "
          f"counters steps={steps_1} tokens={tokens_1} deferred={deferred_1}; solo n_prompt_2={n_prompt_2} decodes={n_steps_2}")

    # task 2: every budgeted step is decoded as two calls - the generating tokens first, then the prompt chunk
    splits = [(int(a), int(b), int(c)) for a, b, c in SPLIT_RE.findall(log)]
    assert len(splits) == len(lines)
    for (n_gen_view, off2, n_total), (_, n_gen_tokens, n_prompt, _, _) in zip(splits, lines):
        assert n_gen_view == n_gen_tokens == off2
        assert n_total == n_gen_tokens + n_prompt
        assert f"n_batch (effective) = {N_BATCH}, off = {off2}" in log


# (b) single slot: the budget never engages, the output is identical to the knob unset
def test_single_slot_unchanged():
    global server
    contents = []
    for env in (None, {"LLAMA_PREFILL_CHUNK_WITH_DECODE": str(C_REQ)}):
        server = make_server(env, n_slots=1)
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": PROMPT_1, "n_predict": 32, "cache_prompt": False,
        })
        assert res.status_code == 200
        contents.append(res.body["content"])
        log = full_log(server)
        assert len(prio_lines(log)) == 0
    print(f"EVIDENCE (b): outputs identical={contents[0] == contents[1]} len={len(contents[0])}")
    assert contents[0] == contents[1]


# (c) an exempt slot is never chunked
def test_exempt_slot_not_chunked():
    global server
    server = make_server({"LLAMA_PREFILL_CHUNK_WITH_DECODE": str(C_REQ), "LLAMA_PREFILL_CHUNK_EXEMPT_SLOTS": "1"})
    server.start()

    co_schedule(server, PROMPT_1)

    log = full_log(server)
    assert "LLAMA_PREFILL_CHUNK_EXEMPT_SLOTS = [1]" in log
    assert len(prio_lines(log)) == 0


# (d) the checkpoint count of a chunked prompt equals the un-chunked schedule's (LFM2 = hybrid, checkpointed)
LFM2_Q8 = "/models/_test/LFM2-350M-Q8_0.gguf"
CKPT_RE = re.compile(r"id +1 \| task \d+ \| created context checkpoint")


@pytest.mark.skipif(not os.path.exists(LFM2_Q8), reason="LFM2 test model not mounted")
def test_checkpoint_count_unchanged():
    global server
    prompt = words(200, 13)

    def run(env, co: bool) -> tuple[int, int, str]:
        global server
        server = make_server(env)
        server.model_hf_repo = None     # HF resolution would win over model_file
        server.model_hf_file = None
        server.model_file = LFM2_Q8
        server.fa = "off"
        server.start()
        if co:
            _, body = co_schedule(server, prompt, n_gen=384)
        else:
            res = server.make_request("POST", "/completion", data={
                "prompt": prompt, "id_slot": 1, "n_predict": 4, "cache_prompt": False,
            }, timeout=300)
            assert res.status_code == 200
            body = res.body
        log = full_log(server)
        return len(CKPT_RE.findall(log)), body["timings"]["prompt_n"], log

    n_chunked, n_prompt, log_chunked = run({"LLAMA_PREFILL_CHUNK_WITH_DECODE": str(C_REQ)}, co=True)
    n_solo,    _,        _           = run(None, co=False)
    n_co,      _,        _           = run(None, co=True)

    print(f"EVIDENCE (d): n_prompt={n_prompt} checkpoints chunked={n_chunked} solo={n_solo} co-scheduled-unchunked={n_co} "
          f"prio_lines={len(prio_lines(log_chunked))}")
    assert n_prompt > 2 * N_BATCH
    assert len(prio_lines(log_chunked)) >= (n_prompt - (4 + N_BATCH)) // C_EFF    # the tail really was chunked
    assert n_chunked == n_solo
    assert n_chunked <= n_co


# (e) review I1: a DRAFTING slot generates while another slot's long prompt is chunked. View 1 carries the verify
#     batch (1 + n_draft tokens), so verify/accept/restore run before the chunk; the tokens of both slots equal the
#     un-chunked co-scheduled run at temperature 0 (stories15M_MOE + stories15M-q4_0 draft, as test_speculative.py;
#     the slot context is capped at the model's n_ctx_train = 256, so the prompt is ~200 tokens and slot 0 makes 200)
MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


def test_chunked_while_drafting():
    global server
    draft = download_file(MODEL_DRAFT_FILE_URL)

    def run(env) -> tuple[dict, dict, dict, str]:
        global server
        server = ServerPreset.stories15m_moe()
        server.model_draft = draft
        server.spec_type = "draft-simple"
        server.spec_draft_n_min = 4
        server.spec_draft_n_max = 8
        server.fa = "off"
        server.n_slots = 2
        server.n_ctx = 512
        server.n_batch = N_BATCH
        server.n_ubatch = N_BATCH
        server.temperature = 0.0
        server.server_slots = True
        server.server_metrics = True
        server.env = env
        fd, server.log_path = tempfile.mkstemp(suffix='.log')
        os.close(fd)
        server.start()
        gen, body1 = co_schedule(server, words(40, 14), n_predict=8, n_gen=200, extra={"return_tokens": True})
        counters = {k: metric(server, "spec_decode_num_" + k + "_rounds_total") for k in ("verify", "replay", "plain")}
        counters["chunked_steps"] = metric(server, "prefill_chunked_steps_total")
        return gen, body1, counters, full_log(server)

    gen_c, body_c, cnt_c, log_c = run({"LLAMA_PREFILL_CHUNK_WITH_DECODE": str(C_REQ)})
    gen_u, body_u, cnt_u, log_u = run(None)

    lines = prio_lines(log_c)
    print(f"EVIDENCE (e): prio_lines={len(lines)} max_gen_tokens={max(l[1] for l in lines)} "
          f"counters chunked={cnt_c} unchunked={cnt_u} draft_n={gen_c['timings']['draft_n']} "
          f"accepted={gen_c['timings']['draft_n_accepted']} tokens_equal={gen_c['tokens'] == gen_u['tokens']}/{body_c['tokens'] == body_u['tokens']}")

    assert body_c["timings"]["prompt_n"] > 2 * C_EFF
    # the prompt was chunked, and at least one budgeted step carried a verify batch (> 1 generating token)
    assert len(lines) >= -(-body_c["timings"]["prompt_n"] // C_EFF)
    assert all(0 < l[2] <= C_EFF for l in lines)
    assert max(l[1] for l in lines) > 1
    assert cnt_c["chunked_steps"] == len(lines)
    assert len(prio_lines(log_u)) == 0

    # speculation still works and is exact
    assert gen_c["timings"]["draft_n"] > 0
    assert gen_c["timings"]["draft_n_accepted"] > 0
    assert cnt_c["verify"] > 0
    assert gen_c["tokens"] == gen_u["tokens"]
    assert body_c["tokens"] == body_u["tokens"]
