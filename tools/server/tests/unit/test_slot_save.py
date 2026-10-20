import pytest
from utils import *
import base64
import requests
import struct

# sequence state file: magic(4) version(4) payload_size(4), then payload_size llama_token words
STATE_FILE_HEADER_SIZE = 12

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84
    slot_file = os.path.join(server.slot_save_path, "slot1.bin")
    assert res.body["n_written"] == os.path.getsize(slot_file)

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84
    assert res.body["n_read"] == os.path.getsize(slot_file)

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_legacy_token_list():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # rewrite the token payload into a plain token list, as written by servers that predate the packed server_tokens format
    path = os.path.join("tmp", "slot_legacy.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    # the payload written by this server starts with a packed header: LLAMA_TOKEN_NULL(4) version(4) n_tokens(4)
    packed_header_size = 12

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    assert n_tokens == 84

    tokens_start = STATE_FILE_HEADER_SIZE + packed_header_size
    data = data[:STATE_FILE_HEADER_SIZE] + data[tokens_start:tokens_start + n_tokens * 4] + data[payload_end:]
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, n_tokens)

    with open(path, "wb") as f:
        f.write(data)

    # the plain token list must restore, and the restored KV must be reusable
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 6  # only the different part is processed



# the slot file carries the live draft state: a restored slot drafts like a slot that was never saved
#   (upstream ea736c1d5, #30574)
def test_slot_restore_draft_state():
    global server
    server.model_draft_hf_repo = server.model_hf_repo
    server.spec_type = "draft-simple"
    server.start()

    prompt = "What is the capital of France?"
    turn = {}
    for id_slot in [0, 1]:
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "n_predict": 16,
            "id_slot": id_slot,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        turn[id_slot] = res.body["content"]
    assert turn[0] == turn[1]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "draft_slot0.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "draft_slot0.bin",
    })
    assert res.status_code == 200

    timings = {}
    for id_slot in [0, 1]:
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt + turn[id_slot] + " And",
            "n_predict": 32,
            "id_slot": id_slot,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        timings[id_slot] = res.body["timings"]

    assert timings[0]["prompt_n"] == timings[1]["prompt_n"]
    assert timings[0]["draft_n_accepted"] > 0
    assert timings[0]["draft_n"]          == timings[1]["draft_n"]
    assert timings[0]["draft_n_accepted"] == timings[1]["draft_n_accepted"]


# [TAG_SLOT_FILE_DRAFT] round trip with a draft model (stories15M target + stories15M draft, T = 0): after a restore the
#   continuation is byte-identical to the continuous run and the draft is warm from the first drafting round; a file
#   without the appendices (the v62 layout: the llama state payload only) still restores, with the draft cold
STORIES15M_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"
DRAFT_MAGIC = struct.pack("<I", 0x54464453)  # "SDFT"


# the offset of the version-3 (r3; version 2 = the same layout) draft appendix that ends the file: magic(4) version(4), desc_tgt and desc_dft blobs,
#   ident blob, draft_gap(4) bits(4), dft and spec blobs (each blob size(8) + data)
def find_draft_appendix(data):
    o = data.find(DRAFT_MAGIC)
    while o > 0:
        if struct.unpack_from("<I", data, o + 4)[0] in (2, 3):
            pos = o + 8
            try:
                for i in range(5):
                    if i == 3:
                        pos += 8
                    n = struct.unpack_from("<Q", data, pos)[0]
                    pos += 8 + n
            except struct.error:
                pos = -1
            if pos == len(data):
                return o
        o = data.find(DRAFT_MAGIC, o + 1)
    assert False, "no draft appendix"


def stories15m_draft_server():
    srv = ServerPreset.tinyllama2()
    srv.model_hf_repo = None
    srv.model_file = download_file(STORIES15M_URL)
    srv.model_draft = srv.model_file
    srv.spec_type = "draft-simple"
    srv.spec_draft_n_min = 1
    srv.spec_draft_n_max = 8
    srv.n_slots = 1
    srv.n_ctx = 1024
    srv.slot_save_path = "./tmp"
    srv.temperature = 0.0
    srv.fa = "off"
    return srv


def continue_turn(srv, prompt, n_predict):
    res = srv.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": n_predict,
        "id_slot": 0,
        "cache_prompt": True,
        "return_tokens": True,
    })
    assert res.status_code == 200
    return res.body


@pytest.mark.parametrize("n_predict", [9, 48])  # 9 = the first token + one drafting round of up to 8
def test_slot_restore_draft_round_trip(n_predict):
    global server
    server = stories15m_draft_server()
    server.start()

    prompt = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends."
    first = continue_turn(server, prompt, 32)
    follow = prompt + first["content"] + " One day,"

    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "draft_rt.bin"})
    assert res.status_code == 200
    path = os.path.join(server.slot_save_path, "draft_rt.bin")
    assert res.body["n_written"] == os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read()
    off_dft = find_draft_appendix(data)

    # continuous: the slot was never saved / restored
    cont = continue_turn(server, follow, n_predict)
    assert cont["timings"]["draft_n_accepted"] > 0

    # the old layout: the payload alone (what v62 wrote - the payload writer is unchanged)
    # stories15M has no SWA / recurrent layers: no checkpoint appendix, the draft state follows the payload
    with open(os.path.join(server.slot_save_path, "draft_rt_old.bin"), "wb") as f:
        f.write(data[:off_dft])

    for fname, warm in (("draft_rt.bin", True), ("draft_rt_old.bin", False)):
        res = server.make_request("POST", "/slots/0?action=erase")
        assert res.status_code == 200
        res = server.make_request("POST", "/slots/0?action=restore", data={"filename": fname})
        assert res.status_code == 200
        assert res.body["n_read"] == os.path.getsize(os.path.join(server.slot_save_path, fname))

        rest = continue_turn(server, follow, n_predict)
        # byte-identical continuation at T = 0 (the target state is the saved one either way)
        assert rest["tokens"] == cont["tokens"]
        assert rest["content"] == cont["content"]
        assert rest["timings"]["prompt_n"] == cont["timings"]["prompt_n"]
        if warm:
            # the draft continues the saved draft history: the same drafts, accepted from the first round
            assert rest["timings"]["draft_n"] == cont["timings"]["draft_n"]
            assert rest["timings"]["draft_n_accepted"] == cont["timings"]["draft_n_accepted"]
            assert rest["timings"]["draft_n_accepted"] > 0
        print(f"{fname}: n_predict={n_predict} draft_n={rest['timings']['draft_n']} "
              f"draft_n_accepted={rest['timings']['draft_n_accepted']} (continuous {cont['timings']['draft_n']}/"
              f"{cont['timings']['draft_n_accepted']})")


# a draft state saved with another draft context is dropped (the draft starts empty) without failing the restore
def test_slot_restore_draft_state_other_draft_context():
    global server
    server = stories15m_draft_server()
    server.ctkd = "f16"
    server.start()

    prompt = "Once upon a time, there was a little girl named Lily."
    first = continue_turn(server, prompt, 16)
    cont_prompt = prompt + first["content"] + " One day,"
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "draft_other.bin"})
    assert res.status_code == 200
    cont = continue_turn(server, cont_prompt, 16)

    server.stop()
    server.ctkd = "q8_0"
    server.start()
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "draft_other.bin"})
    assert res.status_code == 200
    rest = continue_turn(server, cont_prompt, 16)
    assert rest["tokens"] == cont["tokens"]
    assert rest["timings"]["prompt_n"] == cont["timings"]["prompt_n"]


# a truncated draft section is ignored: the restore is the legacy one (cold draft), the tokens are unaffected
def test_slot_restore_draft_state_truncated():
    global server
    server = stories15m_draft_server()
    server.start()

    prompt = "Once upon a time, there was a little girl named Lily."
    first = continue_turn(server, prompt, 16)
    follow = prompt + first["content"] + " One day,"
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "draft_trunc.bin"})
    assert res.status_code == 200
    cont = continue_turn(server, follow, 16)

    path = os.path.join(server.slot_save_path, "draft_trunc.bin")
    with open(path, "rb") as f:
        data = f.read()
    off_dft = find_draft_appendix(data)
    with open(path, "wb") as f:
        f.write(data[:-5])

    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "draft_trunc.bin"})
    assert res.status_code == 200
    assert res.body["n_read"] == off_dft  # the payload only
    rest = continue_turn(server, follow, 16)
    assert rest["tokens"] == cont["tokens"]


# (review M2) the publish guard: with a disk cache, a restored warm draft lifts `partial` only for a file of the same
#   namespace identity; a file from another identity (here: another semantic LLAMA_* env) keeps the warm draft but
#   stays partial - the overwritten slot is refused by the disk cache (saves_refused_partial), never published
@pytest.mark.parametrize("same_namespace", [True, False])
def test_slot_restore_draft_namespace_guard(same_namespace):
    global server
    cache_dir = os.path.abspath(os.path.join("./tmp", f"u5-disk-cache-{int(same_namespace)}"))
    if os.path.isdir(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
    server = stories15m_draft_server()
    server.extra_args = ["--cache-disk", cache_dir, "--cache-disk-gib", "1", "--cache-disk-min-tokens", "1"]
    server.start()

    prompt = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park."
    first = continue_turn(server, prompt, 16)
    follow = prompt + first["content"] + " One day,"
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "draft_ns.bin"})
    assert res.status_code == 200
    cont = continue_turn(server, follow, 9)

    server.stop()
    if not same_namespace:
        server.env = {"LLAMA_U5_NAMESPACE_PROBE": "1"}  # an unregistered LLAMA_* name: semantic, in the env digest
    server.start()
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "draft_ns.bin"})
    assert res.status_code == 200
    rest = continue_turn(server, follow, 9)
    assert rest["tokens"] == cont["tokens"]
    # warm either way: the draft descriptor matches
    assert rest["timings"]["draft_n_accepted"] == cont["timings"]["draft_n_accepted"] > 0

    refused0 = server.make_request("GET", "/disk-cache").body["saves_refused_partial"]
    continue_turn(server, "Unrelated text about a big red bus driving through the town at night.", 4)
    refused = server.make_request("GET", "/disk-cache").body["saves_refused_partial"] - refused0
    print(f"same_namespace={same_namespace}: saves_refused_partial delta {refused}")
    if same_namespace:
        assert refused == 0
    else:
        assert refused >= 1



def draft_appendix_bits(data):
    pos = find_draft_appendix(data) + 8
    for _ in range(3):  # desc_tgt, desc_dft, ident
        pos += 8 + struct.unpack_from("<Q", data, pos)[0]
    return struct.unpack_from("<I", data, pos + 4)[0]  # after draft_gap(4)


def disk_cache_saves(srv):
    res = srv.make_request("GET", "/disk-cache")
    assert res.status_code == 200
    return res.body["saves"], res.body["n_entries"]


# (astra pkg2 Critical) a shifted (noncanonical) slot keeps that provenance through the slot file: shift -> save ->
#   restore x2 publishes nothing. Without the mark the first restore lifted both guards (prompt.clear() reset
#   noncanonical, the warm same-identity draft cleared partial) and the second restore's disk save of the overwritten
#   slot published the first restored - shifted - state as the canonical state of its tokens
def test_slot_restore_noncanonical_never_published():
    global server
    cache_dir = os.path.abspath(os.path.join("./tmp", "u5-disk-cache-noncanonical"))
    if os.path.isdir(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
    server = stories15m_draft_server()
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.extra_args = ["--cache-disk", cache_dir, "--cache-disk-gib", "1", "--cache-disk-min-tokens", "1"]
    server.start()

    prompt = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park."
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 320,
        "id_slot": 0,
        "cache_prompt": True,
        "ignore_eos": True,
    })
    assert res.status_code == 200
    assert res.body["truncated"] is True  # the context was shifted
    assert res.body["timings"]["predicted_n"] == 320

    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "noncanonical.bin"})
    assert res.status_code == 200
    with open(os.path.join(server.slot_save_path, "noncanonical.bin"), "rb") as f:
        data = f.read()
    assert draft_appendix_bits(data) & 2, "the noncanonical provenance bit is not in the slot file"

    saves0, entries0 = disk_cache_saves(server)
    for i in range(2):
        res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "noncanonical.bin"})
        assert res.status_code == 200
        assert res.body["n_read"] == len(data)
    # an unrelated turn overwrites the slot: one more disk save of the restored state
    continue_turn(server, "A big red bus drove through the town at night.", 4)
    saves, entries = disk_cache_saves(server)
    print(f"noncanonical: saves {saves0} -> {saves}, entries {entries0} -> {entries}")
    assert saves == saves0
    assert entries == entries0


# (astra pkg2 delta r3) only a version-3 draft section of a canonical prompt makes a restored slot publishable. A pkg2 r1
#   file (section version 2) wrote a shifted prompt with bits 0 - the same bytes as a canonical one - so a version-2
#   section is never trusted: the draft restores warm, the prompt stays partial. "v2_shifted" is the r1 file of a shifted
#   slot (this build's file with the version set to 2 and the bits cleared, the exact r1 layout), "v2" the same for a
#   canonical slot, "v3" the unmodified canonical file (the positive control: published). The "nodraft_" variants run
#   without a draft model (the n-gram / no-draft profiles, where a legacy restore used to stay publishable): the
#   draft-less v3 section of a canonical slot is published, a shifted slot's file cut before the section (the r1 / v62
#   layout of a draft-less slot: no provenance at all) is not
@pytest.mark.parametrize("variant", ["v3", "v2", "v2_shifted", "nodraft_v3", "nodraft_nosection_shifted"])
def test_slot_restore_pre_r3_section_never_published(variant):
    global server
    cache_dir = os.path.abspath(os.path.join("./tmp", f"u5-disk-cache-prov-{variant}"))
    if os.path.isdir(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
    server = stories15m_draft_server()
    if variant.startswith("nodraft_"):
        server.model_draft = None
        server.spec_type = None
        server.spec_draft_n_min = None
        server.spec_draft_n_max = None
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.extra_args = ["--cache-disk", cache_dir, "--cache-disk-gib", "1", "--cache-disk-min-tokens", "1"]
    server.start()

    shifted = variant.endswith("_shifted")
    prompt = "Once upon a time, there was a little girl named Lily. She loved to play outside in the park."
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 320 if shifted else 16,
        "id_slot": 0,
        "cache_prompt": True,
        "ignore_eos": True,
    })
    assert res.status_code == 200
    assert res.body["truncated"] is shifted

    fname = f"prov_{variant}.bin"
    path = os.path.join(server.slot_save_path, fname)
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": fname})
    assert res.status_code == 200
    with open(path, "rb") as f:
        data = bytearray(f.read())
    off = find_draft_appendix(data)
    assert struct.unpack_from("<I", data, off + 4)[0] == 3
    pos = off + 8
    for _ in range(3):  # desc_tgt, desc_dft, ident
        pos += 8 + struct.unpack_from("<Q", data, pos)[0]
    bits_off = pos + 4  # after draft_gap(4)
    assert (struct.unpack_from("<I", data, bits_off)[0] & 2 != 0) is shifted
    if variant.startswith("v2"):
        struct.pack_into("<I", data, off + 4, 2)
        struct.pack_into("<I", data, bits_off, 0)
    elif variant.endswith("_nosection_shifted"):
        assert struct.unpack_from("<Q", data, bits_off + 4)[0] == 0  # n_dft: draft-less
        data = data[:off]
    if variant not in ("v3", "nodraft_v3"):
        with open(path, "wb") as f:
            f.write(data)

    res = server.make_request("POST", "/slots/0?action=erase")  # no disk save: the restores below land in an empty slot
    assert res.status_code == 200
    saves0, entries0 = disk_cache_saves(server)
    for _ in range(2):  # the second restore's disk save is the restored state's
        res = server.make_request("POST", "/slots/0?action=restore", data={"filename": fname})
        assert res.status_code == 200
        assert res.body["n_read"] == len(data)
    rest = continue_turn(server, "A big red bus drove through the town at night.", 4)
    saves, entries = disk_cache_saves(server)
    print(f"{variant}: saves {saves0} -> {saves}, entries {entries0} -> {entries}")
    if variant in ("v3", "nodraft_v3"):
        assert saves > saves0
    else:
        assert saves == saves0
        assert entries == entries0

# (astra pkg2 Important) the restore's host transient is one 16 MiB piece, not a whole tensor range: a draft K range of
#   ~30k cells x 288 x f32 = ~33 MiB (above glibc's 32 MiB mmap threshold cap, so a whole-range buffer is fresh pages)
#   was allocated at once by llama_io_read_cb::read_tensor; chunked, the peak RSS delta of the restore stays below it.
#   The target K is q8_0 (~9 MiB a range), so the draft side decides. Linux only (VmHWM, clear_refs)
def test_slot_restore_draft_memory_bound():
    global server
    if not os.path.exists("/proc/self/clear_refs"):
        pytest.skip("needs /proc/<pid>/clear_refs (Linux)")
    server = stories15m_draft_server()
    server.n_ctx = 32768
    server.n_batch = 2048
    server.n_ubatch = 256
    server.ctk = "q8_0"
    server.ctkd = "f32"
    # stories15M is trained at 128 tokens and the server caps the slot context there: lift the cap through the metadata
    server.extra_args = ["--override-kv", "llama.context_length=int:32768"]
    server.start()

    unit = " Once upon a time, a girl played in the park."
    res = server.make_request("POST", "/tokenize", data={"content": unit * 10})
    assert res.status_code == 200
    prompt = unit * (30000 * 10 // len(res.body["tokens"]))
    res = server.make_request("POST", "/tokenize", data={"content": prompt})
    assert res.status_code == 200
    n_prompt = len(res.body["tokens"])
    assert 28000 < n_prompt < 32000, n_prompt
    first = continue_turn(server, prompt, 2)
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "draft_mem.bin"})
    assert res.status_code == 200
    n_file = os.path.getsize(os.path.join(server.slot_save_path, "draft_mem.bin"))

    pid = server.process.pid

    def vm(field):
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])  # kB
        raise AssertionError(field)

    deltas = []
    for _ in range(2):
        res = server.make_request("POST", "/slots/0?action=erase")
        assert res.status_code == 200
        with open(f"/proc/{pid}/clear_refs", "w") as f:
            f.write("5")  # reset VmHWM to the current RSS
        rss0 = vm("VmRSS")
        res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "draft_mem.bin"})
        assert res.status_code == 200
        assert res.body["n_read"] == n_file
        deltas.append((vm("VmHWM") - rss0) / 1024.0)
    range_mib = n_prompt * 288 * 4 / 1024.0 / 1024.0
    print(f"draft memory bound: n_prompt={n_prompt} file={n_file / 1048576:.1f} MiB draft K range={range_mib:.1f} MiB "
          f"restore peak RSS delta={', '.join(f'{d:.1f}' for d in deltas)} MiB")
    assert range_mib > 32
    assert max(deltas) < 26, deltas


def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


#
# Multimodal server (mmproj loaded) slot save/restore.
#
# A pure-text slot on a multimodal server and a slot containing images must both support save/restore.
# Erase remains gated on the slot's content.
#

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"
IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture
def mmproj_server():
    # tinygemma3 is a small multimodal model: the mmproj is provided by the HF registry API and auto-downloaded on first run.
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
    return mm_server


def test_slot_save_restore_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    # A pure-text prompt processed on slot 1 of a multimodal server.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Saving a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0  # the slot KV (prompt + generated tokens) was written

    # Restore the saved state into slot 0; it must round-trip exactly.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # Prefix reuse is not checked with the default SWA cache.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_save_restore_with_image(mmproj_server):
    server = mmproj_server
    # Use the full SWA cache so the restored image prefix can be reused.
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content_cat = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert res.body["timings"]["cache_n"] == 0
    assert prompt_n_full > 32  # text plus image tokens are all processed

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    n_written = res.body["n_written"]
    assert n_saved > 0
    assert n_written > 0

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert res.body["n_read"] == n_written

    # a different image must not reuse the restored image tokens; only the text prefix before the image is common
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_TRUCK)],
        },
    })
    assert res.status_code == 200
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n < 16
    assert res.body["timings"]["prompt_n"] == prompt_n_full - cache_n

    # restore again and resend the same image: the image tokens must be reused and greedy sampling must reproduce the original content
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content_cat


def test_slot_save_restore_with_two_images(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.n_ctx = 2048  # two images need more than the default 512 per slot
    server.start()

    prompt = {
        "prompt_string": "A: <__media__> B: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT), _get_img_base64(IMG_URL_TRUCK)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 64

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    assert res.body["content"] == content


def test_slot_save_restore_with_image_across_restart(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # restart the server with the same model and mmproj: the saved file must restore in the new process and the image KV must be reused
    server.stop()
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content


def test_slot_save_restore_image_payload_larger_than_context(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    # the slot context, as the server computed it (n_ctx split across the slots)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    n_ctx_slot = res.body["default_generation_settings"]["n_ctx"]

    # a filler token, used to grow the prompt up to the slot context
    res = server.make_request("POST", "/tokenize", data={"content": " hello" * 8})
    assert res.status_code == 200
    assert len(res.body["tokens"]) == 8

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n" + " hello" * (n_ctx_slot - res.body["timings"]["prompt_n"] - 8),
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    path = os.path.join("tmp", "mm_slot_large_payload.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    assert payload_size > n_ctx_slot  # the scenario under test: the payload does not fit in n_ctx

    # drop the image from the slot, then restore it from the file
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_media_file_without_mmproj(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 200

    # restart the same model without the mmproj: restoring the media file must fail gracefully and leave the slot usable
    server.stop()
    server.no_mmproj = True
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 400
    assert "Cannot restore media tokens without an mmproj" in res.body["error"]["message"]

    # A failed restore must leave the slot empty and usable.
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    content = res.body["content"]

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["content"] == content


@pytest.fixture
def swa_server():
    swa = ServerPreset.tinygemma3()
    swa.slot_save_path = "./tmp"
    swa.temperature = 0.0
    swa.cache_ram = 0
    # Keep the first prompt checkpoint before the divergence point.
    swa.n_ubatch = 32
    return swa


# the non-ASCII name checks that the appendix lands in the same file as the llama state on Windows
@pytest.mark.parametrize("filename", ["ckpt_slot1.bin", "ckpt_slot1_é.bin"])
def test_slot_restore_preserves_context_checkpoints(swa_server, filename):
    server = swa_server
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]
    assert n_live < n_full

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": filename,
    })
    assert res.status_code == 200
    assert res.body["n_saved"] > 0
    ckpt_file = os.path.join(server.slot_save_path, filename)
    assert res.body["n_written"] == os.path.getsize(ckpt_file)

    res = server.make_request("POST", "/completion", data={
        "prompt": "Unrelated text with no common prefix occupies the slot now.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": filename,
    })
    assert res.status_code == 200
    assert res.body["n_read"] == os.path.getsize(ckpt_file)

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_live


# checkpoint appendix (fork version 2): magic(4) version(4), desc_tgt, desc_dft and ident blobs (size(8) + data), count(4),
# then per checkpoint n_tokens(8) pos_min(4) pos_max(4) flags_tgt(4) flags_dft(4) and three blobs (target, draft,
# speculative), each size(8) + data
CKPT_HEADER_SIZE = 24


def parse_ckpt_appendix(data, end=None):
    off = data.find(struct.pack("<II", 0x504b4353, 2))
    assert off > 0
    pos = off + 8
    for _ in range(3):  # desc_tgt, desc_dft, ident
        n = struct.unpack_from("<Q", data, pos)[0]
        pos += 8 + n
    count_off = pos
    count = struct.unpack_from("<I", data, pos)[0]
    ckpts = []
    pos += 4
    for _ in range(count):
        start = pos
        pos += CKPT_HEADER_SIZE
        blobs = []
        for _ in range(3):
            n = struct.unpack_from("<Q", data, pos)[0]
            blobs.append(pos + 8)
            pos += 8 + n
        ckpts.append((start, pos, blobs[0]))
    assert pos == (len(data) if end is None else end)
    return off, count_off, ckpts


# [TAG_SLOT_FILE_CKPT_NO_DEEP] (astra pkg2 Important, merge review M1 / m7) checkpoints restored from a slot file (id_task
#   -2) are RAM-only: the ordinary conversation save of the restored slot writes the entry without them, so they never
#   reach the shared checkpoint store (n_deep_ckpts). Before the fix the same-identity restore left them publishable and
#   the .aux path stored them (a later load turns them into id_task -1 deep candidates). The default
#   --checkpoint-min-step (8192) keeps deep_spill out of this short conversation: every stored checkpoint would be one
#   of the entry save's
def test_slot_restore_checkpoints_never_reach_disk_store(swa_server):
    server = swa_server
    cache_dir = os.path.abspath(os.path.join("./tmp", "pkg2-disk-cache-slot-ckpt"))
    if os.path.isdir(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
    server.extra_args = ["--cache-disk", cache_dir, "--cache-disk-gib", "1", "--cache-disk-min-tokens", "1"]
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20
    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "ckpt_disk.bin"})
    assert res.status_code == 200
    with open(os.path.join(server.slot_save_path, "ckpt_disk.bin"), "rb") as f:
        data = f.read()
    _, _, ckpts = parse_ckpt_appendix(data, find_draft_appendix(data))  # (r3) the draft-less provenance section ends the file
    assert len(ckpts) > 0

    res = server.make_request("POST", "/slots/1?action=erase")  # no disk save: the restore below lands in an empty slot
    assert res.status_code == 200

    def status():
        res = server.make_request("GET", "/disk-cache")
        assert res.status_code == 200
        return res.body["n_entries"], res.body["n_deep_ckpts"]

    entries0, deep0 = status()
    for _ in range(2):  # the second restore saves the first restored state (the ordinary entry save)
        res = server.make_request("POST", "/slots/1?action=restore", data={"filename": "ckpt_disk.bin"})
        assert res.status_code == 200
    entries, deep = status()
    print(f"slot-file checkpoints: {len(ckpts)} in the file; entries {entries0} -> {entries}, n_deep_ckpts {deep0} -> {deep}")
    assert entries == entries0 + 1  # the restored conversation was saved
    assert deep == deep0 == 0       # without the slot-file checkpoints

    # still used from RAM: the second ending reuses the restored checkpoint, as without a disk cache
    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] > 0

# a damaged appendix must be ignored, or its checkpoints dropped when they fail to load, without aborting the server
@pytest.mark.parametrize("damage", ["oversized_blob", "empty_target", "corrupt_state", "many_checkpoints"])
def test_slot_restore_damaged_checkpoint_appendix(swa_server, damage):
    server = swa_server
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "ckpt_damaged.bin",
    })
    assert res.status_code == 200

    path = os.path.join(server.slot_save_path, "ckpt_damaged.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    off, count_off, ckpts = parse_ckpt_appendix(data, find_draft_appendix(data))  # (r3) the draft section ends every file

    if damage == "oversized_blob":
        # the first target blob declares a size that cannot be allocated, it must be rejected before allocating
        data = data[:ckpts[0][0] + CKPT_HEADER_SIZE] + struct.pack("<Q", 1 << 62)
    elif damage == "empty_target":
        # the target blobs are removed and their size set to 0, a valid save never writes an empty target state
        for start, end, tgt in reversed(ckpts):
            size = struct.unpack_from("<Q", data, tgt - 8)[0]
            data = data[:tgt - 8] + struct.pack("<Q", 0) + data[tgt + size:]
    elif damage == "corrupt_state":
        # the sizes are intact, but the target states do not load
        for _, _, tgt in ckpts:
            struct.pack_into("<I", data, tgt, 0xdeadbeef)
    else:
        # more than 1024 entries: one-byte fillers that never match go first, the real checkpoints stay last
        filler = struct.pack("<qiiIIQBQQ", 0, 0, 1 << 30, 1, 1, 1, 0, 0, 0)
        data = data[:count_off + 4] + filler * (1025 - len(ckpts)) + data[count_off + 4:]
        struct.pack_into("<I", data, count_off, 1025)

    with open(path, "wb") as f:
        f.write(data)

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "ckpt_damaged.bin",
    })
    assert res.status_code == 200
    if damage in ("oversized_blob", "empty_target"):
        assert res.body["n_read"] == off

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    if damage == "many_checkpoints":
        assert res.body["timings"]["prompt_n"] == n_live
    else:
        assert res.body["timings"]["prompt_n"] > n_live


# the draft blobs of the checkpoint appendix are not covered by the main payload checks,
# so restoring into a server with another draft KV cache type must not abort
@pytest.mark.parametrize("ctkd_restore", ["f16", "q8_0"])
def test_slot_restore_checkpoints_draft_kv_type_change(swa_server, ctkd_restore):
    server = swa_server
    server.model_draft_hf_repo = "ggml-org/tinygemma3-GGUF:Q8_0"  # same file as the target, already in the HF cache
    server.spec_type = "draft-simple"
    server.ctkd = "f16"
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "ckpt_draft_slot1.bin",
    })
    assert res.status_code == 200

    server.stop()
    server.ctkd = ctkd_restore
    server.start()

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "ckpt_draft_slot1.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_live


# [TAG_DISK_ASYNC_SAVE] (SC2) the asynchronous disk-cache save (--cache-disk-async N MiB): the server loop copies the
#   state off the device, workers hash / write the blocks, the loop commits. Compared with the synchronous path
#   (--cache-disk-async 0) byte by byte, and exercised with a save in flight against a lookup of its conversation, two saves
#   of one slot in flight, an idle server, an injected background failure and shutdown. LLAMA_DISK_ASYNC_TEST_DELAY_MS
#   slows every background block task, LLAMA_DISK_ASYNC_TEST_FAIL_MAPS=N makes the first N map writes throw.
#   Model: LFM2-350M (hybrid: checkpoints / .ck files; 128k training context - the server caps the slot context at it),
#   also as the draft (every entry has a .dft stream)
DC_MODEL_URL = "https://huggingface.co/LiquidAI/LFM2-350M-GGUF/resolve/main/LFM2-350M-Q8_0.gguf"
DC_WORDS = ("apple river stone cloud garden lantern window forest silver morning candle harbor meadow thunder "
            "pencil violet copper island marble whisper orchard falcon velvet canyon").split()


def dc_server(tag, async_mib, env=None, n_ctx=16384, n_slots=1):
    import shutil
    cache_dir = os.path.abspath(os.path.join("./tmp", f"sc2-{tag}"))
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    srv = ServerPreset.tinyllama2()
    srv.model_hf_repo = None
    srv.model_file = download_file(DC_MODEL_URL)
    srv.model_draft = srv.model_file
    srv.spec_type = "draft-simple"
    srv.spec_draft_n_min = 1
    srv.spec_draft_n_max = 8
    srv.n_slots = n_slots
    srv.kv_unified = n_slots > 1
    srv.slot_save_path = "./tmp"
    srv.n_ctx = n_ctx
    srv.n_batch = 1024
    srv.temperature = 0.0
    srv.server_metrics = True
    srv.extra_args = ["--cache-disk", cache_dir, "--cache-disk-gib", "1", "--cache-disk-min-tokens", "1",
                      "--cache-ram", "0",  # as the s1 / balkon profiles: no RAM tier between the slot and the disk
                      "--cache-disk-async", str(async_mib)]
    srv.env = env
    return srv, cache_dir


def dc_text(seed, n_words):
    import random
    rnd = random.Random(seed)
    return " ".join(rnd.choice(DC_WORDS) for _ in range(n_words))


def dc_complete(srv, prompt, n=8, slot=0):
    res = srv.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n, "id_slot": slot, "cache_prompt": True, "temperature": 0.0,
        "return_tokens": True})
    assert res.status_code == 200
    return res.body


def dc_metrics(srv):
    res = srv.make_request("GET", "/metrics")  # a task through the loop: finished background saves are committed
    assert res.status_code == 200
    out = {}
    for line in res.body.splitlines() if isinstance(res.body, str) else []:
        if line.startswith("llamacpp:"):
            k, v = line.split(" ", 1)
            out[k[len("llamacpp:"):]] = float(v)
    return out


def dc_status(srv, through_loop=True):
    if through_loop:
        dc_metrics(srv)
    res = srv.make_request("GET", "/disk-cache")
    assert res.status_code == 200
    return res.body


def dc_settle(srv, tries=1200):
    for _ in range(tries):
        if dc_metrics(srv)["disk_cache_saves_pending"] == 0:
            return dc_status(srv)
        time.sleep(0.05)
    assert False, "saves still pending"


def dc_snapshot(cache_dir):
    """the namespace by content: entries keyed by n_tokens (ids are random) -> sha of .tok / .aux / .tgt / .dft; the
    block store as {name: sha}; no .tmp file may be left"""
    import glob, hashlib, json as js
    d = glob.glob(os.path.join(cache_dir, "ns-*"))
    assert len(d) == 1, d
    ns = d[0]
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    entries = {}
    for meta in glob.glob(os.path.join(ns, "*.meta")):
        eid = os.path.basename(meta)[:-5]
        m = js.load(open(meta))
        files = {ext: (sha(os.path.join(ns, eid + ext)) if os.path.exists(os.path.join(ns, eid + ext)) else None)
                 for ext in (".tok", ".aux", ".tgt", ".dft")}
        files["bytes"] = m["bytes"]
        files["draft_cover"] = m["draft_cover"]
        entries[m["n_tokens"]] = files
    blk = {os.path.basename(p): sha(p) for p in glob.glob(os.path.join(ns, "blk", "*"))}
    tmp = glob.glob(os.path.join(ns, "**", "*.tmp"), recursive=True)
    return entries, blk, tmp


DC_A = "Archive A. " + dc_text(1, 3000)
DC_B = "Archive B. " + dc_text(2, 3000)
DC_A_MORE = " Then " + dc_text(3, 600)


def dc_scenario(srv, n_slots):
    """A -> B (A saved) -> A grown (A loaded, B saved) -> B (A grown re-saved). With two slots X sits in slot 1 first
    and is erased after A's first save, so A comes back at another cell offset"""
    outs = []
    if n_slots > 1:
        dc_complete(srv, "Archive X. " + dc_text(9, 1500), slot=1)
    outs.append(dc_complete(srv, DC_A)["tokens"])
    if n_slots > 1:
        assert srv.make_request("POST", "/slots/1?action=erase").status_code == 200
    outs.append(dc_complete(srv, DC_B)["tokens"])
    r = dc_complete(srv, DC_A + DC_A_MORE)
    outs.append(r["tokens"])
    loaded_a = r["timings"]["cache_n"]
    r = dc_complete(srv, DC_B + " and so on")
    outs.append(r["tokens"])
    return outs, loaded_a, r["timings"]["cache_n"]


# staging 1 MiB: every block waits for the previous one (back-pressure); n_slots 2: the cells move between the saves
@pytest.mark.parametrize("staging_mib,n_slots", [(2048, 1), (1, 1), (2048, 2)])
def test_disk_async_save_identical_to_sync(staging_mib, n_slots):
    global server
    res = {}
    for mode, mib in (("sync", 0), ("async", staging_mib)):
        server, cache_dir = dc_server(f"ident-{mode}-{staging_mib}-{n_slots}", mib, n_ctx=16384 * n_slots, n_slots=n_slots)
        server.start()
        outs, loaded_a, loaded_b = dc_scenario(server, n_slots)
        st = dc_settle(server)
        m = dc_metrics(server)
        server.stop()
        res[mode] = (outs, loaded_a, loaded_b, dc_snapshot(cache_dir), st, m)
        print(f"{mode}: loaded A {loaded_a} B {loaded_b}, saves {st['saves']}, async {st['saves_async']}, written "
              f"{st['bytes_written']}, re-save written {m['disk_cache_last_save_written_bytes']:.0f} of "
              f"{m['disk_cache_last_save_state_bytes']:.0f}, blocks {len(res[mode][3][1])}")
    s, a = res["sync"], res["async"]
    assert s[0] == a[0], "continuations differ"
    assert s[1] == a[1] and s[2] == a[2] and s[1] > 0 and s[2] > 0, "the loads differ / did not happen"
    assert s[3][0] == a[3][0], "entries differ (.tok / .aux / .tgt / .dft content)"
    assert s[3][1] == a[3][1], "block stores differ"
    assert not s[3][2] and not a[3][2], "a .tmp file was left"
    assert s[4]["saves"] == a[4]["saves"] and s[4]["bytes_written"] == a[4]["bytes_written"]
    assert a[4]["saves_async"] == a[4]["saves"] and s[4]["saves_async"] == 0 and a[4]["saves_async_failed"] == 0
    assert a[4]["save_async"] > 0 and s[4]["save_async"] == 0
    assert a[5]["disk_cache_last_save_seconds"] >= a[5]["disk_cache_last_save_blocking_seconds"] > 0


@pytest.mark.parametrize("n_slots", [1, 2])
def test_disk_save_delta_resave(n_slots):
    """[TAG_DISK_DELTA_REL] a conversation re-saved after it grew writes only the changed 4 MiB blocks (its K / V ranges
    of ~11k rows x 1 KiB span 3 blocks), on both paths - with two slots in one pool its cells sit at another offset at
    the re-save than at the first save (X erased in between), which the v63 tensor-aligned cut turned into a rewrite"""
    global server
    res = {}
    for mode, mib in (("sync", 0), ("async", 2048)):
        server, cache_dir = dc_server(f"delta-{mode}-{n_slots}", mib, n_ctx=32768 * n_slots, n_slots=n_slots)
        server.start()
        a = "Archive A. " + dc_text(22, 9000)
        b = "Archive B. " + dc_text(23, 2000)
        if n_slots > 1:
            dc_complete(server, "Archive X. " + dc_text(21, 1500), slot=1)
        dc_complete(server, a)
        if n_slots > 1:
            assert server.make_request("POST", "/slots/1?action=erase").status_code == 200
        dc_complete(server, b)  # the first save of A
        first = dc_settle(server)["last_save_written"]
        r = dc_complete(server, a + " Then " + dc_text(24, 200))  # A loaded (elsewhere), grows; B saved
        assert r["timings"]["cache_n"] > 5000, r["timings"]
        dc_complete(server, b + " x")  # the re-save of A
        dc_settle(server)
        m = dc_metrics(server)
        res[mode] = (m["disk_cache_last_save_state_bytes"], m["disk_cache_last_save_written_bytes"], r["tokens"], first)
        assert dc_complete(server, a + " Then " + dc_text(24, 200) + " end")["timings"]["cache_n"] > 5000
        server.stop()
        print(f"{mode} np {n_slots}: first save written {first}, re-save state {res[mode][0]:.0f} written "
              f"{res[mode][1]:.0f} ({100.0 * res[mode][1] / res[mode][0]:.1f} %)")
    assert res["sync"][2] == res["async"][2]
    assert abs(res["sync"][1] - res["async"][1]) < 1e4  # the same blocks written (/metrics prints %g)
    for mode in res:
        assert res[mode][1] < 0.3 * res[mode][0], f"{mode}: the re-save rewrote unchanged blocks"


def test_disk_async_save_in_flight_lookup_and_second_save():
    """A's save still in flight when A comes back: the lookup waits for it and loads it (no re-processing); two saves of
    the same slot in flight at once, committed in order"""
    global server
    server, _ = dc_server("inflight", 2048, env={"LLAMA_DISK_ASYNC_TEST_DELAY_MS": "1500"})
    server.start()
    short_b = "Archive B. " + dc_text(2, 200)  # a short turn: A's save is still running when it returns
    dc_complete(server, DC_A)
    dc_complete(server, short_b)
    m = dc_metrics(server)
    assert m["disk_cache_saves_pending"] >= 1 and m["disk_cache_entries"] == 0, m
    r = dc_complete(server, DC_A + DC_A_MORE)  # waits for A's save, then loads it
    assert r["timings"]["cache_n"] >= 3000, r["timings"]
    dc_complete(server, "Archive C. " + dc_text(4, 200))  # C evicts A+ (a long save)
    dc_complete(server, "Archive D. " + dc_text(5, 200))  # D evicts C: two saves of slot 0 in flight
    assert dc_metrics(server)["disk_cache_saves_pending"] >= 2
    st = dc_settle(server)
    # A (its tokens incl. the 8 generated ones are no prefix of A+: kept), B, A+, C; D is in the slot
    assert len(st["entries"]) == 4 and st["saves_async_failed"] == 0
    got = dc_complete(server, DC_A + DC_A_MORE + " end")
    assert got["timings"]["cache_n"] >= 3000, got["timings"]
    server.stop()


def test_disk_async_save_commits_on_an_idle_server():
    """(review C1) a background save that finishes while no request comes is committed by the woken server loop"""
    global server
    server, _ = dc_server("idle", 2048, env={"LLAMA_DISK_ASYNC_TEST_DELAY_MS": "300"})
    server.start()
    dc_complete(server, DC_A)
    dc_complete(server, "Archive B. " + dc_text(2, 200))  # A's save starts; then the server is idle
    t0 = time.time()
    st = dc_status(server, through_loop=False)
    assert st["n_entries"] == 0
    while time.time() - t0 < 30:  # GET /disk-cache is not a task: only the wake commits
        st = dc_status(server, through_loop=False)
        if st["n_entries"] == 1:
            break
        time.sleep(0.1)
    waited = time.time() - t0
    print(f"committed after {waited:.1f} s, pending {st['saves_pending']}")
    assert st["n_entries"] == 1 and st["saves_pending"] == 0
    # the background part: 300 ms per block over the workers (the server's min(4, cores / 4)); the commit follows within 2 s
    n_workers = min(4, max(1, (os.cpu_count() or 1) // 4))
    assert waited < 2.0 + 0.3 * st["n_blocks"] / n_workers + 1.0
    server.stop()


@pytest.mark.parametrize("hook", ["LLAMA_DISK_ASYNC_TEST_FAIL_MAPS", "LLAMA_DISK_ASYNC_TEST_FAIL_META"])
def test_disk_async_save_background_failure(hook):
    """(review C2, r2 C2) a save that fails in the background - a map write that throws on a worker, or its .meta at
    the commit - is committed as a failure: its files and fresh blocks go (the block inventory matches the files), the
    staging and job accounting stay consistent (later saves do not block, the drain at exit does not hang)"""
    import glob
    global server
    server, cache_dir = dc_server(f"fail-{hook[-4:].lower()}", 64, env={hook: "1"})
    server.start()
    dc_complete(server, DC_A)
    dc_complete(server, DC_B)  # A's save fails
    st = dc_settle(server)
    assert st["saves_async_failed"] == 1 and st["n_entries"] == 0, st
    blk_files = lambda: len(glob.glob(os.path.join(glob.glob(os.path.join(cache_dir, "ns-*"))[0], "blk", "*.blk")))
    assert st["n_blocks"] == blk_files() == 0, (st["n_blocks"], blk_files())
    r = dc_complete(server, DC_A + " again")  # nothing to load; B's save works
    assert r["timings"]["cache_n"] < 100
    st = dc_settle(server)
    assert st["saves_async_failed"] == 1 and st["saves_async"] == 1 and st["n_entries"] == 1
    assert st["n_blocks"] == blk_files()
    t0 = time.time()
    server.stop()  # save on exit + drain
    assert time.time() - t0 < 5
    entries, blk, tmp = dc_snapshot(cache_dir)
    assert len(entries) == 2 and not tmp  # B and the last A, no file of the failed save
    server.start()
    st = dc_status(server)
    assert st["n_entries"] == 2 and st["n_blocks"] == len(blk)  # no unreferenced block was left on disk
    server.stop()


def test_disk_async_save_before_sleep():
    """(review r2 C1) the idle sleep saves every slot with saves in flight and one worker: a worker that finishes
    wakes the server loop while the loop holds the task mutex in the sleeping callback - no deadlock, both saves land"""
    global server
    server, cache_dir = dc_server("sleep", 2048, n_ctx=32768, n_slots=2,
                                  env={"LLAMA_DISK_ASYNC_TEST_THREADS": "1", "LLAMA_DISK_ASYNC_TEST_DELAY_MS": "20"})
    server.sleep_idle_seconds = 1
    server.start()
    dc_complete(server, DC_A, slot=0)
    dc_complete(server, DC_B, slot=1)
    t0 = time.time()
    slept = False
    while time.time() - t0 < 30:
        res = server.make_request("GET", "/props", timeout=5)
        if res.status_code == 200 and res.body.get("is_sleeping"):
            slept = True
            break
        time.sleep(0.2)
    print(f"asleep after {time.time() - t0:.1f} s")
    assert slept, "the server did not reach the sleeping state (pre-sleep save deadlocked?)"
    t1 = time.time()
    server.stop()
    assert time.time() - t1 < 5
    entries, blk, tmp = dc_snapshot(cache_dir)
    assert len(entries) == 2 and not tmp


def test_disk_async_save_at_shutdown():
    """a save in flight when the server stops: save_on_exit drains it - the entry is on disk and loads after a restart"""
    global server
    server, cache_dir = dc_server("shutdown", 2048, env={"LLAMA_DISK_ASYNC_TEST_DELAY_MS": "40"})
    server.start()
    dc_complete(server, DC_A)
    dc_complete(server, DC_B)  # A in flight; B stays in the slot (saved on exit)
    server.stop()
    entries, blk, tmp = dc_snapshot(cache_dir)
    assert len(entries) == 2 and not tmp
    server.start()
    assert dc_complete(server, DC_A + " again")["timings"]["cache_n"] >= 3000
    assert dc_complete(server, DC_B + " again")["timings"]["cache_n"] >= 3000
    server.stop()
