import os
import tempfile
import pytest
from utils import *

# With --kv-unified the slots share one KV pool. When it fills, the server frees room by purging an idle slot
# (try_clear_idle_slots). These tests pin down which slot is purged and what happens to its context.
#
# Assertions use server state (/slots, timings) rather than log timing: the log is written by a worker
# thread, so the lines of the main loop are not guaranteed to be on disk when the HTTP response arrives.
# The full log is checked once at the end, after a graceful stop has flushed it.

server = ServerPreset.tinyllama2()


def words(n: int, seed: int) -> str:
    # n pseudo-random fixed-width "words"; each is ~6 tokens on stories260K (one per digit), so token counts
    # scale with n and two prompts built from the same n have the same token count
    return " ".join(f"w{(seed * 7919 + i * 104729) % 99991:05d}" for i in range(n))


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 3
    server.n_ctx = 1024           # the shared pool (--kv-unified)
    server.kv_unified = True
    server.cache_ram = 0          # no host cache: purging is the only way to make room
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


def fill(slot: int, prompt: str) -> int:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": slot,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body["timings"]["prompt_n"]


def slot_tokens() -> dict:
    # tokens currently held by each slot (0 after a purge)
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    return {s["id"]: s.get("n_prompt_tokens", 0) for s in res.body}


def full_log() -> str:
    server.stop()   # SIGTERM → graceful exit → the log is flushed
    with open(server.log_path) as f:
        return f.read()


# the idle slot with the fewest tokens is the cheapest to rebuild → it is the one purged
def test_purge_evicts_smallest_idle_slot():
    global server
    server.start()

    big, small, third = words(60, 1), words(20, 2), words(100, 3)
    n_big = fill(0, big)
    n_small = fill(1, small)
    assert n_big > 2 * n_small

    n_third = fill(2, third)                     # does not fit next to big + small → one idle slot must go
    assert n_big + n_small + n_third > 1024      # the pool really did overflow ...
    assert n_big + n_third + 8 <= 1024           # ... and purging the small slot alone is enough

    held = slot_tokens()
    assert held[1] == 0                          # the cheap one went
    assert held[0] > 0                           # the expensive one stayed

    # the big context survived: continuing it only processes the new tail
    res = server.make_request("POST", "/completion", data={
        "prompt": big + " " + words(4, 9),
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] < n_big // 4

    log = full_log()
    assert "purging slot 1 with" in log
    assert "purging slot 0 with" not in log


# equal sizes → the least recently used one is purged
def test_purge_prefers_lru_among_equal_sizes():
    global server
    server.start()

    a, b = words(30, 4), words(30, 5)            # same token count (fixed-width words), different content
    n_a = fill(0, a)
    n_b = fill(1, b)
    assert n_a == n_b

    n_third = fill(2, words(120, 6))
    assert n_a + n_b + n_third > 1024
    assert n_a + n_third + 8 <= 1024

    held = slot_tokens()
    assert held[0] == 0                          # slot 0 finished first → least recently used
    assert held[1] > 0

    log = full_log()
    assert "purging slot 0 with" in log
    assert "purging slot 1 with" not in log


# with a prompt cache, the purged context is saved before it is cleared, so a returning client restores it
# instead of re-processing it (--no-cache-idle-slots isolates the purge path from the launch-time save)
def test_purge_saves_to_prompt_cache():
    global server
    server.cache_ram = 64
    server.no_cache_idle_slots = True
    server.slot_save_path = "/tmp"   # enables the /slots/:id?action=... endpoints (erase is used below)
    server.start()

    small, big = words(20, 7), words(60, 8)
    n_small = fill(0, small)
    n_big = fill(1, big)

    n_third = fill(2, words(100, 9))             # overflow → slot 0 (smallest) is purged ...
    assert n_small + n_big + n_third > 1024
    assert n_big + n_third + 8 <= 1024
    assert slot_tokens()[0] == 0

    # the third client is done and releases its slot (a restore needs room in the pool) ...
    res = server.make_request("POST", "/slots/2?action=erase")
    assert res.status_code == 200
    assert slot_tokens()[2] == 0

    # ... and the purged client coming back with the same prefix restores it from the cache instead of
    # re-processing it
    res = server.make_request("POST", "/completion", data={
        "prompt": small + " " + words(4, 10),
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0
    assert res.body["timings"]["prompt_n"] < n_small // 2

    log = full_log()
    assert "__TEST_TAG_PURGE_SAVED_0__" in log
