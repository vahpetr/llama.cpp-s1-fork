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


def co_schedule(s: ServerProcess, prompt: str, n_predict: int = 4) -> tuple[dict, dict]:
    """slot 0 generates a long answer; once it is decoding, slot 1 gets `prompt`. Returns both response bodies."""
    out = {}

    def gen():
        res = s.make_request("POST", "/completion", data={
            "prompt": "Once upon a time",
            "id_slot": 0,
            "n_predict": 1024,
            "ignore_eos": True,
            "cache_prompt": False,
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
    }, timeout=300)
    assert res1.status_code == 200
    t.join()
    assert out["gen"].status_code == 200
    # slot 0 was still decoding when slot 1's prompt finished - the whole prefill was co-scheduled
    assert out["gen"].body["timings"]["predicted_n"] == 1024
    return out["gen"].body, res1.body


def prio_lines(log: str) -> list[tuple[int, int, int, int, str]]:
    return [(int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5]) for m in PRIO_RE.findall(log)]


# (a) a prompt that arrives while another slot decodes is fed in chunks of <= C; without a decoder it is not
def test_chunked_while_decoding():
    global server
    server = make_server({"LLAMA_PREFILL_CHUNK_WITH_DECODE": str(C_REQ)})
    server.start()

    _, body1 = co_schedule(server, PROMPT_1)
    n_prompt_1 = body1["timings"]["prompt_n"]
    assert n_prompt_1 > 4 * C_EFF

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

    log = full_log(server)
    assert f"LLAMA_PREFILL_CHUNK_WITH_DECODE = {C_EFF} (requested {C_REQ}, n_batch = {N_BATCH}, on)" in log
    lines = prio_lines(log)
    assert len(lines) >= 1
    for n_gen_slots, n_gen_tokens, n_prompt, budget, slots in lines:
        assert n_gen_slots >= 1 and n_gen_tokens >= 1
        assert budget == C_EFF
        assert 0 < n_prompt <= C_EFF
        assert slots.startswith("1:")
    # every prompt token of slot 1 went through a budgeted step
    assert sum(l[2] for l in lines) == n_prompt_1


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
