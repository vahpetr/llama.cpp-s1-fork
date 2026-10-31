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
#   of one slot in flight, an idle server, an injected background failure and shutdown. The server's test hooks
#   (LLAMA_DISK_ASYNC_TEST=key=value,... with --cache-disk-test; keys in server-disk-cache.cpp) slow or fail the
#   background work: dc_server(hooks={...}).
#   Model: LFM2-350M (hybrid: checkpoints / .ck files; 128k training context - the server caps the slot context at it),
#   also as the draft (every entry has a .dft stream)
DC_MODEL_URL = "https://huggingface.co/LiquidAI/LFM2-350M-GGUF/resolve/main/LFM2-350M-Q8_0.gguf"
DC_WORDS = ("apple river stone cloud garden lantern window forest silver morning candle harbor meadow thunder "
            "pencil violet copper island marble whisper orchard falcon velvet canyon").split()


def dc_server(tag, async_mib, env=None, n_ctx=16384, n_slots=1, fresh=True, hooks=None):
    """hooks: the server's LLAMA_DISK_ASYNC_TEST keys (server-disk-cache.cpp), passed with --cache-disk-test"""
    import shutil
    cache_dir = os.path.abspath(os.path.join("./tmp", f"sc2-{tag}"))
    if fresh and os.path.isdir(cache_dir):
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
    srv.env = dict(env or {})
    if hooks is None and "LLAMA_DISK_ASYNC_TEST" not in srv.env:
        hooks = {}
    if hooks is not None:
        hooks = {"check_pins": "1", **hooks}  # (review r11 6) cached vs freshly derived pins after every commit
        srv.extra_args.append("--cache-disk-test")
        srv.env["LLAMA_DISK_ASYNC_TEST"] = ",".join(f"{k}={v}" for k, v in hooks.items())
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
            st = dc_status(srv)
            assert st.get("pin_check_failures", 0) == 0, "the cached pin set differed from a fresh derivation"
            return st
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


def dc_streams(cache_dir):
    """the state streams by content: n_tokens -> {.tgt / .dft: sha of the concatenation (inline segments + blocks)} -
    independent of where the blocks are cut (a map lists its segments; a block file is its raw bytes)"""
    import glob, hashlib, json as js, struct
    ns = glob.glob(os.path.join(cache_dir, "ns-*"))[0]
    out = {}
    for meta in glob.glob(os.path.join(ns, "*.meta")):
        eid = os.path.basename(meta)[:-5]
        res = {}
        for ext in (".tgt", ".dft"):
            mp = os.path.join(ns, eid + ext)
            if not os.path.exists(mp):
                res[ext] = None
                continue
            b = open(mp, "rb").read()
            magic, _ver, n_h = struct.unpack_from("<III", b, 0)
            assert magic == 0x3350414d, hex(magic)  # MAP3
            at = 12 + n_h
            (n_stream,) = struct.unpack_from("<Q", b, at)
            at += 8
            h, n = hashlib.sha256(), 0
            while True:
                kind = b[at]
                at += 1
                if kind == 255:
                    break
                if kind == 0:
                    (k,) = struct.unpack_from("<Q", b, at)
                    at += 8
                    h.update(b[at:at + k]); n += k
                    at += k
                    (ks,) = struct.unpack_from("<I", b, at)
                    at += 4 + ks
                else:
                    assert kind == 1, kind
                    (ks,) = struct.unpack_from("<I", b, at)
                    key = b[at + 4:at + 4 + ks].decode()
                    at += 4 + ks
                    blk = open(os.path.join(ns, "blk", key + ".blk"), "rb").read()
                    h.update(blk); n += len(blk)
            assert n == n_stream, (mp, n, n_stream)
            res[ext] = h.hexdigest()
        out[js.load(open(meta))["n_tokens"]] = res
    return out


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


# staging 1 MiB: every block waits for the previous one (back-pressure); n_slots 2: the cells move between the saves.
#   [TAG_DISK_DELTA_REL] async 0 cuts the blocks as v68 (aligned within the tensor), async > 0 from the start of the range:
#   with one slot (ranges at cell 0) both cuts and so every file are the same; with two slots the streams (and .tok / .aux)
#   are the same, the maps / block files differ by the cut
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
        res[mode] = (outs, loaded_a, loaded_b, dc_snapshot(cache_dir), st, m, dc_streams(cache_dir))
        print(f"{mode}: loaded A {loaded_a} B {loaded_b}, saves {st['saves']}, async {st['saves_async']}, written "
              f"{st['bytes_written']}, re-save written {m['disk_cache_last_save_written_bytes']:.0f} of "
              f"{m['disk_cache_last_save_state_bytes']:.0f}, blocks {len(res[mode][3][1])}")
    s, a = res["sync"], res["async"]
    assert s[0] == a[0], "continuations differ"
    assert s[1] == a[1] and s[2] == a[2] and s[1] > 0 and s[2] > 0, "the loads differ / did not happen"
    assert s[6] == a[6], "state streams differ"
    strip = lambda ents: {n: {k: v for k, v in f.items() if k not in (".tgt", ".dft")} for n, f in ents.items()}
    assert strip(s[3][0]) == strip(a[3][0]), "entries differ (.tok / .aux content)"
    if n_slots == 1:
        assert s[3][0] == a[3][0], "entries differ (.tgt / .dft maps)"
        assert s[3][1] == a[3][1], "block stores differ"
        assert s[4]["bytes_written"] == a[4]["bytes_written"]
    assert not s[3][2] and not a[3][2], "a .tmp file was left"
    assert s[4]["saves"] == a[4]["saves"]
    assert a[4]["saves_async"] == a[4]["saves"] and s[4]["saves_async"] == 0 and a[4]["saves_async_failed"] == 0
    assert a[4]["save_async"] > 0 and s[4]["save_async"] == 0
    assert a[5]["disk_cache_last_save_seconds"] >= a[5]["disk_cache_last_save_blocking_seconds"] > 0


@pytest.mark.skipif("LLAMA_SERVER_V68_BIN_PATH" not in os.environ,
                    reason="needs a v68 (f3cbd94cb, pre-SC2) llama-server: LLAMA_SERVER_V68_BIN_PATH")
@pytest.mark.parametrize("n_slots", [1, 2])
def test_disk_sync_save_bytes_equal_v68(n_slots):
    """[TAG_DISK_DELTA_REL] --cache-disk-async 0 writes v68's disk bytes: the dc_scenario on a v68 server (f3cbd94cb, no
    SC2) and on this one with async 0 give the same entries (.tok / .aux / .tgt / .dft maps) and the same block files
    (names and content) - also in a unified 2-slot pool, where the cells sit away from cell 0 and only the v68
    tensor-aligned cut gives v68's keys"""
    global server
    res = {}
    for mode in ("v68", "sync"):
        server, cache_dir = dc_server(f"v68cut-{mode}-{n_slots}", 0, n_ctx=16384 * n_slots, n_slots=n_slots)
        if mode == "v68":  # no SC2 options / hooks / metrics there
            server.server_path = os.environ["LLAMA_SERVER_V68_BIN_PATH"]
            i = server.extra_args.index("--cache-disk-async")
            del server.extra_args[i:i + 2]
            server.extra_args.remove("--cache-disk-test")
            server.env.pop("LLAMA_DISK_ASYNC_TEST", None)
        server.start()
        outs, loaded_a, loaded_b = dc_scenario(server, n_slots)
        if mode == "sync":
            dc_settle(server)
        server.stop()
        res[mode] = (outs, loaded_a, loaded_b, dc_snapshot(cache_dir))
        print(f"{mode} np {n_slots}: loaded A {loaded_a} B {loaded_b}, entries {len(res[mode][3][0])}, "
              f"blocks {len(res[mode][3][1])}")
    v, s = res["v68"], res["sync"]
    assert v[0] == s[0] and v[1] == s[1] and v[2] == s[2] and v[1] > 0 and v[2] > 0
    assert v[3][0] == s[3][0], "entries differ from v68 (.tok / .aux / .tgt / .dft)"
    assert v[3][1] == s[3][1], "block files differ from v68"
    assert not v[3][2] and not s[3][2]


@pytest.mark.parametrize("n_slots", [1, 2])
def test_disk_save_delta_resave(n_slots):
    """[TAG_DISK_DELTA_REL] a conversation re-saved after it grew writes only the changed 4 MiB blocks (its K / V ranges
    of ~11k rows x 1 KiB span 3 blocks) - with two slots in one pool its cells sit at another offset at the re-save than
    at the first save (X erased in between), which the tensor-aligned cut turns into a rewrite. The asynchronous path
    (--cache-disk-async > 0) cuts from the start of the range; the synchronous one (async 0) keeps the v68 cut, so its
    disk bytes are v68's: one slot = the same delta on both paths, two slots = only async writes the delta"""
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
    if n_slots == 1:
        assert abs(res["sync"][1] - res["async"][1]) < 1e4  # the same blocks written (/metrics prints %g)
    for mode in (("sync", "async") if n_slots == 1 else ("async",)):
        assert res[mode][1] < 0.3 * res[mode][0], f"{mode}: the re-save rewrote unchanged blocks"
    if n_slots > 1:  # async 0 = the v68 cut: the moved cells shift every block, the re-save rewrites the state
        assert res["sync"][1] > 0.5 * res["sync"][0], "sync (async 0): not the v68 tensor-aligned cut"


def test_disk_async_save_in_flight_lookup_and_second_save():
    """A's save still in flight when A comes back: the lookup waits for it and loads it (no re-processing); two saves of
    the same slot in flight at once, committed in order"""
    global server
    server, _ = dc_server("inflight", 2048, hooks={"delay_ms": "1500"})
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
    n_workers = 2  # explicit (review r3 I6): the bound below does not depend on the host's cores / affinity
    server, _ = dc_server("idle", 2048, hooks={"delay_ms": "300",
                                             "threads": str(n_workers)})
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
    # the background part: 300 ms per block over the workers; the commit follows within 2 s
    assert waited < 2.0 + 0.3 * st["n_blocks"] / n_workers + 1.0
    server.stop()


def test_disk_async_test_hooks_need_the_flag():
    """(review r7 5) LLAMA_DISK_ASYNC_TEST is honoured only with --cache-disk-test: a leaked variable (here a 1 MiB quota)
    leaves the cache as configured"""
    global server
    server, _ = dc_server("hooksoff", 2048, env={"LLAMA_DISK_ASYNC_TEST": "quota_mib=1,threads=1"})
    server.start()
    assert dc_status(server, through_loop=False)["quota_bytes"] == 1 << 30  # --cache-disk-gib 1
    server.stop()
    server, _ = dc_server("hookson", 2048, hooks={"quota_mib": "1"})
    server.start()
    assert dc_status(server, through_loop=False)["quota_bytes"] == 1 << 20
    server.stop()


def test_disk_async_enqueue_failure_leaves_no_phantom_task():
    """(review r10 2) a block task whose queue insertion fails (hook fail_enqueue: std::bad_alloc) fails its save without a
    phantom task: later saves still commit in order and the exit drain does not hang"""
    global server
    server, _ = dc_server("enqfail", 2048, hooks={"fail_enqueue": "3"})
    server.start()
    dc_complete(server, DC_A)
    dc_complete(server, DC_B)  # A's save: its 3rd block's enqueue throws
    st = dc_settle(server)
    assert st["saves_async_failed"] == 1 and st["n_entries"] == 0, st
    dc_complete(server, "Archive C. " + dc_text(8, 50))  # B's save commits after the failed one
    st = dc_settle(server)
    assert st["saves_async"] == 1 and st["n_entries"] == 1
    t0 = time.time()
    server.stop()
    assert time.time() - t0 < 5


def test_disk_async_finalize_enqueue_failure_ends_the_job():
    """(pkg9-fix 4) a save whose state is all inline (a short prompt: no block task) queues only its finalizer task; when
    that enqueue fails (hook fail_finalize: std::bad_alloc) the save fails at once (save() false, one warning, no second
    "background save failed" report), the job is finished on the server thread, later saves commit and the exit drain
    does not hang"""
    global server
    server, _ = dc_server("finfail", 2048, hooks={"fail_finalize": "1"})
    server.start()
    dc_complete(server, "Archive S. " + dc_text(31, 6))
    dc_complete(server, "Archive T. " + dc_text(32, 6))  # S's save: its finalizer enqueue throws
    st = dc_settle(server)
    assert st["saves_async_failed"] == 1 and st["n_entries"] == 0, st
    dc_complete(server, "Archive U. " + dc_text(33, 6))  # T's save commits
    st = dc_settle(server)
    assert st["saves_async"] == 1 and st["n_entries"] == 1, st
    t0 = time.time()
    server.stop()
    assert time.time() - t0 < 5


def test_disk_sync_fallback_shares_blocks_with_a_save_in_flight():
    """(review r12 1, r13 7) the synchronous fallback writes blocks on the workers' claims and never waits for them: two
    sibling conversations share ~7.5k tokens of prefix (whole blocks); slot 0's save is in flight, slot 1's runs
    synchronously (max_jobs 1). Run 1: the worker is slow per block; run 2: the worker's first block write (a shared
    block) is held 6 s while marked "being written" - the synchronous save does not wait for it (the request returns
    well inside the hold) and writes that block once more (bytes_written_twice). Both runs: every block file intact,
    the block set equals an all-synchronous control run, bytes_written = the control's + bytes_written_twice, both
    entries load after a restart"""
    import glob
    global server
    shared = DC_A + DC_B
    p0 = shared + " Sibling one " + dc_text(41, 100)
    p1 = shared + " Sibling two " + dc_text(42, 100)

    def run(tag, async_mib, hooks):
        global server
        server, cache_dir = dc_server(tag, async_mib, n_ctx=32768, n_slots=2, hooks=hooks)
        server.start()
        dc_complete(server, p0, slot=0)
        dc_complete(server, p1, slot=1)
        dc_complete(server, "Archive X. " + dc_text(43, 50), slot=0)  # (async runs) job 1: sibling one
        t0 = time.monotonic()
        dc_complete(server, "Archive Y. " + dc_text(44, 50), slot=1)  # (async runs) sibling two: synchronous
        waited = time.monotonic() - t0
        st = dc_settle(server)
        server.stop()
        ns = glob.glob(os.path.join(cache_dir, "ns-*"))[0]
        blk = {os.path.basename(f): os.path.getsize(f) for f in glob.glob(os.path.join(ns, "blk", "*"))}
        return st, blk, waited

    ctl, ctl_blk, _ = run("syncshare-ctl", 0, None)
    for tag, hooks in (("syncshare", {"max_jobs": "1", "threads": "1", "delay_ms": "150", "finalize_delay_job1_ms": "6000"}),
                       ("syncshare-held", {"max_jobs": "1", "threads": "1", "hold_write_job1_ms": "6000"})):
        st, blk, waited = run(tag, 2048, hooks)
        twice = st["bytes_written_twice"]
        print(f"{tag}: sync request {waited:.1f} s, saves {st['saves']} async {st['saves_async']} backpressure "
              f"{st['save_backpressure']} written {st['bytes_written']} (control {ctl['bytes_written']}, twice {twice}), "
              f"blocks {len(blk)} (control {len(ctl_blk)})")
        assert st["save_backpressure"] == 1 and st["saves_async"] == 1 and st["saves"] == 2 and st["saves_async_failed"] == 0
        for name, size in blk.items():
            assert ".tmp" not in name, f"a temp block file was left: {name}"
            assert size == int(name[:-4].rsplit("-", 1)[1]), f"block {name} is truncated"
        assert blk == ctl_blk, "the block set differs from the synchronous control"
        assert st["bytes_written"] == ctl["bytes_written"] + twice, "bytes counted twice or skipped vs the control"
        if tag == "syncshare-held":
            assert waited < 6.0 - 2.5, "the synchronous save waited for the held worker write"
            assert 0 < twice <= 4 << 20, "the held block was not written by the synchronous save"
        else:
            assert twice <= 4 << 20
        server, _ = dc_server(tag, 2048, n_ctx=32768, n_slots=2, fresh=False)
        server.start()
        for prompt in (p0, p1):
            assert dc_complete(server, prompt + " end", slot=0)["timings"]["cache_n"] >= 7000
        server.stop()


def test_disk_async_save_superseded_by_a_sync_extension():
    """(review r12 2, r13 1) A is registered; A+ (slot 0) is in flight with a held end and pins A; A++ (slot 1, extends
    A+) is saved synchronously and commits first: A+ is superseded at once (it stops pinning, so A++'s commit replaces
    A) and dropped at its commit - only A++ is registered of the three"""
    global server
    server, _ = dc_server("supersede", 2048, n_ctx=32768, n_slots=2,
                          hooks={"max_jobs": "1", "finalize_delay_job3_ms": "5000"})
    server.start()
    o1 = dc_complete(server, DC_A, slot=0)["content"]
    dc_complete(server, "Archive Z. " + dc_text(50, 50), slot=0)  # job 1: A
    n_a = [e["n_tokens"] for e in dc_settle(server)["entries"]]
    a_plus = DC_A + o1 + DC_A_MORE
    o2 = dc_complete(server, a_plus, slot=0)["content"]  # job 2: Archive Z
    dc_settle(server)
    a_pp = a_plus + o2 + " Later " + dc_text(51, 200)
    dc_complete(server, a_pp, slot=1)
    dc_complete(server, "Archive X. " + dc_text(52, 50), slot=0)  # job 3: A+ (held)
    dc_complete(server, "Archive Y. " + dc_text(53, 50), slot=1)  # A++: synchronous, commits first
    st = dc_settle(server)
    long_entries = sorted(e["n_tokens"] for e in st["entries"] if e["n_tokens"] > 1000)
    print(f"A {n_a}, entries {[(e['id'], e['n_tokens']) for e in st['entries']]} saves {st['saves']} "
          f"superseded {st['saves_superseded']}")
    assert st["saves_superseded"] == 1 and st["saves_async_failed"] == 0
    assert len(long_entries) == 1 and long_entries[0] > max(n_a), st["entries"]
    assert dc_complete(server, a_pp + " end", slot=0)["timings"]["cache_n"] >= 3000
    server.stop()


def test_disk_async_superseded_save_commits_when_its_extension_is_gone():
    """(review r14 1) A+ (slot 0) is in flight with a held end; A++ (slot 1) is saved synchronously and supersedes it;
    under a tight quota a later synchronous save evicts A++ before A+ commits - A+ then commits normally (it is not
    dropped: nothing else holds that conversation)"""
    global server
    server, _ = dc_server("supersede-evicted", 2048, n_ctx=32768, n_slots=2,
                          hooks={"max_jobs": "1", "finalize_delay_job1_ms": "45000", "quota_mib": "150"})
    server.start()
    a_plus = DC_A + DC_A_MORE
    o = dc_complete(server, a_plus, slot=0)["content"]
    a_pp = a_plus + o + " Later " + dc_text(57, 200)
    dc_complete(server, a_pp, slot=1)
    dc_complete(server, "Archive X. " + dc_text(58, 50), slot=0)  # job 1: A+ (held)
    dc_complete(server, "Archive Y. " + dc_text(59, 50), slot=1)  # A++: synchronous, supersedes A+
    n_app = max(e["n_tokens"] for e in dc_status(server)["entries"])
    dc_complete(server, DC_B, slot=1)                              # (Archive Y saved synchronously)
    dc_complete(server, "Archive W. " + dc_text(60, 50), slot=1)  # DC_B saved synchronously: evicts A++
    mid = dc_status(server)
    st = dc_settle(server)
    print(f"A++ {n_app}; before A+ commits {[(e['id'], e['n_tokens']) for e in mid['entries']]}; "
          f"after {[(e['id'], e['n_tokens']) for e in st['entries']]}; superseded {st['saves_superseded']}, "
          f"evictions {st.get('evictions')}")
    assert n_app not in [e["n_tokens"] for e in mid["entries"]], "A++ was not evicted before A+ committed"
    assert st["saves_superseded"] == 0 and st["saves_async_failed"] == 0
    long_entries = [e["n_tokens"] for e in st["entries"] if e["n_tokens"] > 4000 and e["n_tokens"] < n_app]
    assert len(long_entries) == 1, st["entries"]  # A+
    assert dc_complete(server, a_plus + " end", slot=0)["timings"]["cache_n"] >= 4000
    server.stop()


def test_disk_async_superseded_save_neither_delays_a_lookup_nor_blocks_an_exact_save():
    """(review r14 2, 3) A+ (slot 0) is in flight with a held end and superseded by A++ (slot 1, synchronous): an exact
    save of A+'s tokens (slot 2) is not taken for "already in flight" (it is registered), and a request continuing A++
    does not wait for the superseded save (it loads A++ well inside the hold)"""
    global server
    server, _ = dc_server("supersede-lookup", 2048, n_ctx=32768, n_slots=3,
                          hooks={"max_jobs": "1", "finalize_delay_job1_ms": "30000"})
    server.start()
    a_plus = DC_A + DC_A_MORE
    o = dc_complete(server, a_plus, slot=0)["content"]
    assert dc_complete(server, a_plus, slot=2)["content"] == o
    a_pp = a_plus + o + " Later " + dc_text(61, 200)
    dc_complete(server, a_pp, slot=1)
    dc_complete(server, "Archive X. " + dc_text(62, 50), slot=0)  # job 1: A+ (held)
    dc_complete(server, "Archive Y. " + dc_text(63, 50), slot=1)  # A++: synchronous, supersedes A+
    dc_complete(server, "Archive V. " + dc_text(64, 50), slot=2)  # exactly A+'s tokens: saved synchronously
    mid = dc_status(server)
    t0 = time.monotonic()
    r = dc_complete(server, a_pp + " More " + dc_text(65, 20), slot=1)
    waited = time.monotonic() - t0
    st = dc_settle(server)
    long_mid = sorted(e["n_tokens"] for e in mid["entries"] if e["n_tokens"] > 4000)
    print(f"entries while A+ is held {long_mid}; lookup {waited:.1f} s, cache_n {r['timings']['cache_n']}; "
          f"after {sorted(e['n_tokens'] for e in st['entries'] if e['n_tokens'] > 4000)}, superseded {st['saves_superseded']}")
    assert len(long_mid) == 2, "the exact save of A+'s tokens was skipped"
    assert waited < 30.0 - 20.0, "the lookup waited for the superseded save"
    assert r["timings"]["cache_n"] >= long_mid[1] - 16
    assert st["saves_superseded"] == 1 and st["saves_async_failed"] == 0
    assert sorted(e["n_tokens"] for e in st["entries"] if e["n_tokens"] > 4000) == long_mid
    server.stop()


def test_disk_async_shorter_save_after_a_longer_entry_is_kept():
    """(review r13 1) A++ registered in an earlier session; A+ saved asynchronously now: a longer entry that existed
    before the save is no reason to drop it (as the synchronous path) - both are registered, nothing superseded"""
    global server
    server, _ = dc_server("nosupersede", 2048, n_ctx=32768, n_slots=1)
    server.start()
    a_plus = DC_A + DC_A_MORE
    o = dc_complete(server, a_plus)["content"]
    a_pp = a_plus + o + " Later " + dc_text(54, 200)
    dc_complete(server, a_pp)
    dc_complete(server, "Archive X. " + dc_text(55, 50))  # A++
    st0 = dc_settle(server)
    server.stop()
    server, _ = dc_server("nosupersede", 2048, n_ctx=32768, n_slots=1, fresh=False)
    server.start()
    dc_complete(server, a_plus)
    dc_complete(server, "Archive Y. " + dc_text(56, 50))  # A+ (async)
    st = dc_settle(server)
    long0 = sorted(e["n_tokens"] for e in st0["entries"] if e["n_tokens"] > 1000)
    long1 = sorted(e["n_tokens"] for e in st["entries"] if e["n_tokens"] > 1000)
    print(f"before {long0}, after {long1}, saves_async {st['saves_async']}, superseded {st['saves_superseded']}")
    assert st["saves_superseded"] == 0 and st["saves_async"] >= 1
    assert len(long1) == 2 and long1[1] == long0[-1] and long1[0] < long1[1], (long0, long1)
    server.stop()


def test_disk_async_backpressure_with_a_stalled_oldest_save():
    """(review r10 4, r11 1-2) with 2 saves in flight (the oldest held 8 s at its end) the next save runs synchronously:
    the loop does not wait for the in-flight saves (the request returns well inside the stall), the save is counted as
    backpressure, the entry the in-flight A+ save extends (A) stays pinned under quota pressure (80 MiB, LFM2: prompt
    checkpoints on every entry), and after a restart every committed entry is complete (no checkpoint file of an
    unregistered save was collected)"""
    global server
    server, cache_dir = dc_server("bp", 2048, hooks={"finalize_delay_job2_ms": "8000", "quota_mib": "80"})
    server.start()
    o = dc_complete(server, DC_A)["content"]
    a_plus = DC_A + o + DC_A_MORE
    dc_complete(server, "Archive B. " + dc_text(2, 3000))  # job 1: A
    st = dc_settle(server)
    id_a = [e["id"] for e in st["entries"]]
    assert len(id_a) == 1
    r = dc_complete(server, a_plus)                        # A loaded; job 2: B (held 8 s)
    assert r["timings"]["cache_n"] >= 3000, r["timings"]
    dc_complete(server, "Archive C. " + dc_text(3, 200))   # job 3: A+ (pins A)
    assert dc_metrics(server)["disk_cache_saves_pending"] == 2
    t0 = time.time()
    dc_complete(server, "Archive D. " + dc_text(4, 200))   # C's save: 2 in flight -> synchronous
    waited = time.time() - t0
    st = dc_status(server)
    print(f"4th request took {waited:.1f} s, pending {st['saves_pending']}, backpressure {st['save_backpressure']}, "
          f"entries {[(e['id'], e['n_tokens']) for e in st['entries']]}, evictions {st['evictions']}")
    assert st["save_backpressure"] == 1
    assert waited < 8.0 - 3.0, "the synchronous fallback waited for the held in-flight save (8 s stall)"
    assert id_a[0] in [e["id"] for e in st["entries"]], "A (pinned by A+'s save in flight) was evicted"
    st = dc_settle(server)
    ids = [e["id"] for e in st["entries"]]
    print(f"settled: entries {[(e['id'], e['n_tokens']) for e in st['entries']]}, evictions {st['evictions']}")
    assert st["saves_async"] == 3 and st["saves_async_failed"] == 0 and st["saves"] == 4
    assert id_a[0] not in ids and any(e["n_tokens"] > 4000 for e in st["entries"]), "A+ did not replace A"
    server.stop()  # (the slot's D conversation is saved on exit)
    import glob
    metas = glob.glob(os.path.join(cache_dir, "ns-*", "*.meta"))
    server, _ = dc_server("bp", 2048, fresh=False)  # the default quota: the start evicts nothing
    server.start()
    st = dc_status(server)
    print(f"restart: {len(metas)} committed entries on disk, {st['n_entries']} loaded")
    assert st["n_entries"] == len(metas), "an entry was dropped as incomplete at the restart (missing checkpoint file?)"
    server.stop()

def test_disk_async_save_idle_commit_stress():
    """(review r3 C1) 200 x (a turn that replaces the slot's conversation - its save starts - then idle): every
    background save is committed by the woken loop with no further task. No quota is set, nothing is evicted. The wake is
    lock-free and can be late by up to the loop's 1 s wait (never lost): the per-iteration bound = 2 s + the save's new
    blocks x the per-block delay over the workers + 1 s; "slow" iterations are collected, a save that never commits
    fails after 30 s of waiting for it (the request time not included; review r9 8)"""
    global server
    delay_s, n_workers = 0.02, 2
    server, _ = dc_server("stress", 2048, hooks={"delay_ms": str(int(delay_s * 1000)),
                                               "threads": str(n_workers)})
    server.start()
    worst, slow = 0.0, []
    n_blocks = dc_status(server, through_loop=False)["n_blocks"]
    for i in range(201):
        dc_complete(server, f"Note {i}. " + dc_text(1000 + i, 300), n=1)
        if i == 0:
            continue
        t0 = time.time()
        while True:
            st = dc_status(server, through_loop=False)
            if st["saves"] == i and st["saves_pending"] == 0:
                break
            assert time.time() - t0 < 30, f"iteration {i}: save never committed ({st['saves']} saves, {st['saves_pending']} pending)"
            time.sleep(0.01)
        waited = time.time() - t0
        new_blocks = max(1, st["n_blocks"] - n_blocks)
        n_blocks = st["n_blocks"]
        bound = 2.0 + delay_s * 2 * new_blocks / n_workers + 1.0  # x2: target + draft streams claim each block
        if waited > bound:
            slow.append((i, round(waited, 2), round(bound, 2)))
        worst = max(worst, waited)
    print(f"200 idle commits, worst {worst:.2f} s, slow {slow}")
    assert not slow, f"saves committed, but slower than the bound: {slow}"
    server.stop()


def test_disk_async_failed_commits_leave_no_orphan_block():
    """(review r6 1) two saves share prefix blocks: J1 (slot 0) writes them and fails at its end while J2 (slot 1) holds
    claims on them, then J2's commit fails (.meta) - the shared blocks J1 wrote and J2 inherited go with J2's failure
    (no block file outside the inventory, none on disk)"""
    import glob
    global server
    server, cache_dir = dc_server("orphan", 2048, n_ctx=32768, n_slots=2,
                                  hooks={"delay_ms": "50", "threads": "2",
                                       # J1's end lags its blocks by 4 s: J2 claims the shared blocks before J1 commits;
                                       #   J2's by 12 s: still in flight when J1's failure is observed (review r7 6)
                                       "finalize_delay_job1_ms": "4000", "finalize_delay_job2_ms": "12000",
                                       "fail_job": "1", "fail_meta": "1"})
    server.start()
    shared = DC_A + DC_B  # ~7.5k tokens: past one 4 MiB block of each K / V range (4096 rows of 1 KiB)
    dc_complete(server, shared + " tail one", slot=0)
    dc_complete(server, shared + " tail two, longer", slot=1)  # the same prefix: the same leading blocks
    dc_complete(server, "Archive C. " + dc_text(8, 50), slot=0)  # J1: slot 0's conversation (fails at its end)
    dc_complete(server, "Archive D. " + dc_text(9, 50), slot=1)  # J2: slot 1's (claims J1's blocks; .meta fails)
    blk_files = lambda: glob.glob(os.path.join(cache_dir, "ns-*", "blk", "*.blk"))
    t0 = time.time()
    while dc_status(server, through_loop=False)["saves_async_failed"] < 1:
        assert time.time() - t0 < 60, "J1 never committed"
        time.sleep(0.02)
    st = dc_status(server, through_loop=False)
    print(f"after J1's failure: pending {st['saves_pending']} files {len(blk_files())} (J2 holds the shared ones)")
    assert st["saves_pending"] >= 1, "J2 finished before J1's failure: the shared claim was not exercised"
    assert len(blk_files()) > 0, "no block survived J1's failure: the two saves share no block (nothing to test)"
    st = dc_settle(server)
    blk = blk_files()
    print(f"failed {st['saves_async_failed']} entries {st['n_entries']} inventory {st['n_blocks']} files {len(blk)}")
    assert st["saves_async_failed"] == 2 and st["n_entries"] == 0
    assert st["n_blocks"] == 0 and len(blk) == 0, "a block of the failed saves was left on disk"
    server.stop()


def test_disk_async_quota_pins_replaced_prefix():
    """(review r4 1) while a save of A+ is in flight, the entry A it will replace is pinned: an enforce() run by an
    earlier save's commit (over quota) evicts neither A nor, in a cascade, anything else it cannot free; A+ then
    commits and replaces A"""
    global server
    server, _ = dc_server("pin", 2048, hooks={"delay_ms": "400", "threads": "1",
                                            "quota_mib": "70"})
    server.start()
    a_out = dc_complete(server, DC_A)["content"]
    a_plus = DC_A + a_out + DC_A_MORE                    # the conversation continues: A (with its answer) is a prefix
    dc_complete(server, DC_B)                            # A saved; B in the slot
    st = dc_settle(server)
    id_a = [e["id"] for e in st["entries"]]
    assert len(id_a) == 1, st["entries"]
    saves0 = st["saves"]
    dc_complete(server, a_plus)                          # A loaded; B saved (job 1, slow)
    dc_complete(server, "Archive C. " + dc_text(8, 50))  # A+ saved (job 2, replaces A: pinned)
    # job 1 commits while job 2 is in flight: A + B are over the 70 MiB quota, A is pinned
    t0 = time.time()
    while True:
        st = dc_status(server, through_loop=False)
        if st["saves"] == saves0 + 1:
            break  # B committed (review r5 7: by the save count, not a token window)
        assert time.time() - t0 < 120, f"B's save never committed ({st['saves']} saves)"
        time.sleep(0.05)
    id_b = [e["id"] for e in st["entries"] if e["id"] not in id_a]
    print(f"after B's commit: pending {st['saves_pending']} entries {[e['n_tokens'] for e in st['entries']]} "
          f"evictions {st['evictions']} bytes {st['bytes_on_disk']} pending bytes {st['bytes_pending']}")
    assert st["saves_pending"] >= 1, "A+'s save finished too early for the check"
    assert id_a[0] in [e["id"] for e in st["entries"]], "the pinned prefix entry was evicted"
    assert st["evictions"] == 0, "enforce() evicted while only pinned / protected entries could free the charge"
    st = dc_settle(server)
    ids = [e["id"] for e in st["entries"]]
    assert id_a[0] not in ids and len(set(ids) - set(id_a) - set(id_b)) == 1, st["entries"]  # A+ replaced A
    assert dc_complete(server, a_plus + " end")["timings"]["cache_n"] >= 3000
    server.stop()


def test_disk_async_continuation_charge():
    """(review r5 1) a continuation save (A+ replacing A) is charged what it writes minus A's own files - not minus A's
    logical size, which made its charge 0 while its blocks were being written"""
    global server
    server, _ = dc_server("contq", 2048, hooks={"delay_ms": "300", "threads": "1"})
    server.start()
    a_out = dc_complete(server, DC_A)["content"]
    a_plus = DC_A + a_out + DC_A_MORE
    dc_complete(server, DC_B)       # A saved
    dc_settle(server)
    dc_complete(server, a_plus)     # A loaded, B saved
    dc_settle(server)
    dc_complete(server, "Archive C. " + dc_text(8, 50))  # A+ saved (replaces A), slow
    peak, t0 = 0.0, time.time()
    while True:
        m = dc_metrics(server)
        if m["disk_cache_saves_pending"] == 0:
            break
        peak = max(peak, m["disk_cache_bytes_pending"])
        assert time.time() - t0 < 120, "the save never committed"
        time.sleep(0.1)
    print(f"continuation save: peak bytes_pending {peak:.0f}, written {m['disk_cache_last_save_written_bytes']:.0f}")
    assert peak > 0, "the continuation save was charged nothing while writing"
    server.stop()


def test_disk_async_chained_saves_keep_the_middle_entry():
    """(review r5 2) A+ (slot 0) and A++ (slot 1, extends A+) are saved back to back; when A+'s save commits, the new
    A+ entry is handed to A++'s save in flight (pinned): a deep-checkpoint enforce() while A++ is still writing does not
    evict it (quota 1 MiB: everything evictable would go); A++'s save then fails (injected)"""
    global server
    server, _ = dc_server("chain", 2048, n_ctx=32768, n_slots=2,
                          hooks={"delay_ms": "700", "threads": "1",
                               "quota_mib": "1", "fail_job": "2"})
    server.extra_args += ["--ctx-checkpoints", "1", "--checkpoint-min-step", "64"]
    server.start()
    a_plus = DC_A + DC_A_MORE
    o1 = dc_complete(server, a_plus, slot=0)["content"]                 # slot 0: A+
    dc_complete(server, a_plus + o1 + " Later " + dc_text(31, 200), slot=1)  # slot 1: A++ (extends A+)
    dc_complete(server, "Archive C. " + dc_text(8, 50), slot=0)         # job 1: A+
    dc_complete(server, "Archive D. " + dc_text(9, 50), slot=1)         # job 2: A++ (fails at its end)
    t0 = time.time()
    while True:
        st = dc_status(server, through_loop=False)
        if st["saves"] == 1:
            break
        assert time.time() - t0 < 120, "A+'s save never committed"
        time.sleep(0.05)
    id_ap = st["entries"][0]["id"]
    deep0 = st["n_deep_ckpts"]
    dc_complete(server, "Archive E. " + dc_text(10, 3000), slot=0)      # spills deep checkpoints: enforce()
    st = dc_status(server, through_loop=False)
    print(f"after E: pending {st['saves_pending']} failed {st['saves_async_failed']} deep {deep0} -> {st['n_deep_ckpts']} "
          f"entries {[e['id'] for e in st['entries']]} evictions {st['evictions']}")
    assert st["saves_async_failed"] == 0 and st["saves_pending"] >= 1, "A++'s save finished before the check"
    assert id_ap in [e["id"] for e in st["entries"]], "A+ was evicted while A++'s save (which replaces it) was in flight"
    assert st["n_deep_ckpts"] > deep0, "no deep checkpoint was stored meanwhile (the enforce() path was not exercised)"
    st = dc_settle(server)  # A++'s failure commits (A+ is then an ordinary entry again: the 1 MiB quota may take it)
    assert st["saves_async_failed"] == 1
    server.stop()


def test_disk_async_deep_checkpoint_quota_counts_in_flight():
    """(review r4 2, r9 9) a deep checkpoint stored while a save is in flight enforces the quota with the in-flight charge:
    the disk total alone is under the quota, total + the save's written bytes is over it -> an unrelated entry goes
    before that save commits. Deterministic: B's save (the 3rd) writes its blocks at once and then holds its end for
    40 s; the long checkpoint-spilling prefill (D) starts only after bytes_pending shows B's bytes"""
    global server
    # quota 55 MiB: A + Q + checkpoints (~48 MB) fit, + B's ~44 MB charge does not
    server, _ = dc_server("deepq", 2048, hooks={"threads": "1", "finalize_delay_job3_ms": "40000", "quota_mib": "55"})
    server.extra_args += ["--ctx-checkpoints", "1", "--checkpoint-min-step", "64"]
    server.start()
    dc_complete(server, DC_A)
    dc_complete(server, "Archive Q. " + dc_text(7, 50))  # A saved (~46 MiB on disk)
    dc_settle(server)
    dc_complete(server, DC_B)                            # Q saved; B in the slot
    st = dc_settle(server)
    ev0, deep0, saves0 = st["evictions"], st["n_deep_ckpts"], st["saves"]
    dc_complete(server, "Archive C. " + dc_text(9, 50))  # B saved (job 3: in flight for 40 s)
    t0 = time.time()
    while True:
        m = dc_metrics(server)
        if m["disk_cache_bytes_pending"] >= 7 * 2**20:
            break
        assert time.time() - t0 < 30, f"B's blocks were not written ({m['disk_cache_bytes_pending']:.0f} pending bytes)"
        time.sleep(0.1)
    assert m["disk_cache_saves_pending"] >= 1
    dc_complete(server, "Archive D. " + dc_text(10, 3000))  # D's prefill spills checkpoints: deep_put -> enforce
    st = dc_status(server, through_loop=False)
    print(f"after D: pending {st['saves_pending']} pending bytes {st['bytes_pending']} bytes {st['bytes_on_disk']} "
          f"deep {deep0} -> {st['n_deep_ckpts']} evictions {ev0} -> {st['evictions']} entries {[e['n_tokens'] for e in st['entries']]}")
    assert st["saves_pending"] >= 1, "B's save finished before the check"
    # no save was committed since ev0 (B is in flight, C / D are in the slot): the only enforce() runs are the
    #   deep-checkpoint store's (C's and D's prefills spill checkpoints)
    #   (the deep-checkpoint count itself may not grow: the evicted entries' checkpoints are collected)
    assert st["saves"] == saves0, "a commit ran enforce() meanwhile"
    assert st["evictions"] > ev0, "the deep-checkpoint path did not apply the in-flight charge"
    server.stop()


def test_disk_async_branch_commit_keeps_the_shared_prefix():
    """(review r9 10) A' and A+ both extend the stored A (two slots). A' commits while A+'s save is still in flight: A is
    pinned by A+, so A''s commit does not replace it; A+'s save then fails - A is still on disk (it was erased before)"""
    global server
    server, _ = dc_server("branch", 2048, n_ctx=32768, n_slots=2,
                          hooks={"finalize_delay_job3_ms": "3000", "finalize_delay_job4_ms": "1000", "fail_job": "4"})
    server.start()
    o = dc_complete(server, DC_A, slot=0)["content"]
    a = DC_A + o
    dc_complete(server, "Archive Q. " + dc_text(7, 50), slot=0)          # job 1: A
    st = dc_settle(server)
    id_a = [e["id"] for e in st["entries"]]
    assert len(id_a) == 1
    r = dc_complete(server, a + " Plus " + dc_text(31, 100), slot=1)      # slot 1 loads A -> A+
    assert r["timings"]["cache_n"] >= 3000, r["timings"]
    r = dc_complete(server, a + " Prime " + dc_text(32, 100), slot=0)     # slot 0 loads A -> A' (job 2: Q)
    assert r["timings"]["cache_n"] >= 3000, r["timings"]
    dc_settle(server)
    dc_complete(server, "Archive R. " + dc_text(8, 50), slot=0)           # job 3: A' (ends after 3 s)
    dc_complete(server, "Archive S. " + dc_text(9, 50), slot=1)           # job 4: A+ (fails), submitted before job 3 commits
    st = dc_settle(server)
    ids = [e["id"] for e in st["entries"]]
    print(f"entries {[(e['id'], e['n_tokens']) for e in st['entries']]} failed {st['saves_async_failed']}")
    assert st["saves_async_failed"] == 1
    assert id_a[0] in ids, "A' commit replaced A although A+'s save (which extends A) was in flight - and A+ failed"
    server.stop()


def test_disk_async_save_failure_frees_own_copies_of_adopted_blocks():
    """(review r3 I4, r7 1) a re-save of an adopted conversation writes own-store copies of blocks the index already holds
    from the adopt source; when that save fails on a worker, those own copies are unlinked too (the source stays).
    The conversation is longer than one 4 MiB block per K / V range (4096 rows of 1 KiB) - a shorter one has no block
    in common with its continuation (its single partial block changes), so no own copy of an adopted block is written
    and the case is not exercised (r1-r7 of this test: own blocks 0 for that reason)"""
    import glob
    global server
    # namespace 1: another identity (an unregistered LLAMA_* env is semantic) saves A synchronously
    server, cache_dir = dc_server("adopt", 0, env={"LLAMA_SC2_ADOPT_PROBE": "1"})
    server.start()
    long_a = DC_A + DC_B       # ~7.5k tokens: whole blocks the continuation shares
    a_out = dc_complete(server, long_a)["content"]
    dc_complete(server, "Archive Q. " + dc_text(7, 50))  # A saved
    key_src = dc_status(server)["key"]
    server.stop()
    # namespace 2 adopts it; its first save (A grown, evicted) fails in the map write
    server, _ = dc_server("adopt", 2048, hooks={"fail_maps": "1"}, fresh=False)
    server.extra_args += ["--cache-disk-adopt", key_src]
    server.start()
    st = dc_status(server)
    assert st["key"] != key_src and any(a["key"] == key_src for a in st["adopt"]), st["adopt"]
    a_plus = long_a + a_out + DC_A_MORE
    r = dc_complete(server, a_plus)  # loads the adopted entry
    assert r["timings"]["cache_n"] >= 7000, r["timings"]
    dc_complete(server, "Archive R. " + dc_text(8, 50))  # the re-save of A+ (own copies of shared blocks + new) fails
    st = dc_settle(server)
    assert st["saves_async_failed"] == 1 and st["saves_async"] == 0, st
    own_blk = glob.glob(os.path.join(cache_dir, "ns-" + st["key"], "blk", "*.blk"))
    src_blk = glob.glob(os.path.join(cache_dir, "ns-" + key_src, "blk", "*.blk"))
    print(f"own blocks {len(own_blk)}, source blocks {len(src_blk)}")
    assert len(own_blk) == 0 and len(src_blk) > 0
    # the adopted entry still loads
    assert dc_complete(server, a_plus + " end")["timings"]["cache_n"] >= 7000
    server.stop()


@pytest.mark.parametrize("hook", ["fail_maps", "fail_meta"])
def test_disk_async_save_background_failure(hook):
    """(review C2, r2 C2) a save that fails in the background - a map write that throws on a worker, or its .meta at
    the commit - is committed as a failure: its files and fresh blocks go (the block inventory matches the files), the
    staging and job accounting stay consistent (later saves do not block, the drain at exit does not hang)"""
    import glob
    global server
    server, cache_dir = dc_server(f"fail-{hook[-4:]}", 64, hooks={hook: "1"})
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
                                  hooks={"threads": "1", "delay_ms": "20"})
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
    server, cache_dir = dc_server("shutdown", 2048, hooks={"delay_ms": "40"})
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


# [TAG_DISK_INCR] --cache-disk-incr: an asynchronous save of a slot whose KV rows did not change since its last committed
#   save takes those 4 MiB blocks from that save instead of the device. Scenario: A alone (its prompt checkpoint lands
#   near A's end); A + B1 continues it; A + B2 diverges after A (2 * lcp < n_slot: A + B1 is saved, the slot rolls back
#   to A's checkpoint and keeps those cells); A + B3 saves A + B2 - its A blocks were recorded at the previous save and
#   A's cells are untouched since; a short Z saves A + B3 (reuse once more). f32 K / V (LFM2-350M: 2 KiB rows, a 4 MiB
#   block = 2048 cells) so A (~4.4k tokens) spans 2 full blocks per range at a CPU-affordable length; no draft model
#   (the draft memory's ranges are covered by tests/test-state-incr.cpp). Byte identity: the cache directory (entries +
#   block store) of each arm equals the arm without the flag; the verify arm (test hook incr_verify) copies the
#   reusable blocks and compares the keys.
DC_INCR_A = "Archive A. " + dc_text(31, 3500)


def dc_incr_server(arm):
    hooks = {"incr_verify": "1"} if arm == "verify" else {}
    srv, cache_dir = dc_server(f"incr-{arm}", 2048, n_ctx=16384, hooks=hooks)
    srv.model_draft = None
    srv.spec_type = None
    srv.spec_draft_n_min = None
    srv.spec_draft_n_max = None
    srv.extra_args += ["-ctk", "f32", "-ctv", "f32"]
    if arm != "full":
        srv.extra_args.append("--cache-disk-incr")
    return srv, cache_dir


def dc_incr_scenario(srv):
    outs = [dc_complete(srv, DC_INCR_A)["tokens"]]
    for k, seed in enumerate((32, 33, 34)):
        r = dc_complete(srv, DC_INCR_A + f" Part {k}. " + dc_text(seed, 4000))
        outs.append((r["tokens"], r["timings"]["cache_n"]))
    dc_complete(srv, "Archive Z. " + dc_text(35, 300))  # A + B3 saved
    return outs


def test_disk_incr_save_identical_to_full():
    global server
    res = {}
    for arm in ("full", "incr", "verify"):
        server, cache_dir = dc_incr_server(arm)
        server.start()
        outs = dc_incr_scenario(server)
        st = dc_settle(server)
        m = dc_metrics(server)
        server.stop()
        res[arm] = (outs, dc_snapshot(cache_dir), st, m)
        print(f"{arm}: saves {st['saves']}, reused {st.get('incr_blocks_reused')} blocks "
              f"({st.get('incr_bytes_skipped')} B), verify planned {st.get('incr_verify_planned')} checked "
              f"{st.get('incr_verify_checked')} mismatch {st.get('incr_verify_mismatch')}, copied "
              f"{m['disk_cache_bytes_copied_total']:.0f} B, copy {m['disk_cache_save_copy_seconds_total']:.3f} s")
    f = res["full"]
    assert f[2]["saves"] >= 3 and f[2]["incr_blocks_reused"] == 0 and f[2]["incr_verify_planned"] == 0
    for arm in ("incr", "verify"):
        x = res[arm]
        assert f[0] == x[0], f"{arm}: continuations differ"
        assert f[1][0] == x[1][0], f"{arm}: entries differ (.tok / .aux / .tgt content)"
        assert f[1][1] == x[1][1], f"{arm}: block stores differ"
        assert not x[1][2], f"{arm}: a .tmp file was left"
        assert f[2]["saves"] == x[2]["saves"] and x[2]["saves_async_failed"] == 0
    i, v = res["incr"], res["verify"]
    assert i[2]["incr_blocks_reused"] > 0, "nothing was reused"
    assert i[2]["incr_bytes_skipped"] == i[2]["incr_blocks_reused"] * 4 * 1024 * 1024
    assert abs(i[3]["disk_cache_incr_bytes_skipped_total"] - i[2]["incr_bytes_skipped"]) < 1e4  # /metrics prints %g
    for arm in ("full", "incr", "verify"):  # (sc3fix 5) device read = captured - reused; every claim released
        x = res[arm]
        assert x[2]["bytes_device_read"] == x[2]["bytes_copied"] - x[2]["incr_bytes_skipped"], arm
        assert x[2]["async_claims"] == 0, arm
    assert i[2]["bytes_device_read"] < f[2]["bytes_device_read"] == f[2]["bytes_copied"] == i[2]["bytes_copied"]
    assert v[2]["incr_blocks_reused"] == 0  # verify copies everything
    assert v[2]["incr_verify_planned"] == i[2]["incr_blocks_reused"]  # the same blocks were reusable
    assert v[2]["incr_verify_checked"] == v[2]["incr_verify_planned"] and v[2]["incr_verify_mismatch"] == 0


def test_disk_incr_needs_async():
    """--cache-disk-incr without --cache-disk-async: warned, the synchronous save runs as before"""
    global server
    server, _ = dc_server("incr-sync", 0)
    server.extra_args.append("--cache-disk-incr")
    server.start()
    dc_complete(server, DC_A)
    dc_complete(server, DC_B)
    st = dc_settle(server)
    assert st["saves"] >= 1 and st["incr_blocks_reused"] == 0 and st["saves_async"] == 0
    server.stop()


# (sc3fix) regression tests for the SC3 items of the astra v69 DELTA review: each one fails on a09430242 (with only the
#   test hooks / status fields of the sc3fix test commit) and passes with the fixes


def dc_incr_server_np(tag, hooks, n_slots=1, kv_unified=None):
    """dc_incr_server's incr arm (f32 K / V, no draft, --cache-disk-incr) with other hooks / slots"""
    srv, cache_dir = dc_server(tag, 2048, n_ctx=16384 * n_slots, n_slots=n_slots, hooks=hooks)
    srv.model_draft = None
    srv.spec_type = None
    srv.spec_draft_n_min = None
    srv.spec_draft_n_max = None
    if kv_unified is not None:
        srv.kv_unified = kv_unified
    srv.extra_args += ["-ctk", "f32", "-ctv", "f32", "--cache-disk-incr"]
    return srv, cache_dir


def dc_entry_blocks(cache_dir):
    """every block key the registered entries' maps reference: n_tokens -> [keys] (.tgt then .dft)"""
    import glob, json as js, struct
    ns = glob.glob(os.path.join(cache_dir, "ns-*"))[0]
    out = {}
    for meta in glob.glob(os.path.join(ns, "*.meta")):
        eid = os.path.basename(meta)[:-5]
        keys = []
        for ext in (".tgt", ".dft"):
            mp = os.path.join(ns, eid + ext)
            if not os.path.exists(mp):
                continue
            b = open(mp, "rb").read()
            _magic, _ver, n_h = struct.unpack_from("<III", b, 0)
            at = 12 + n_h + 8
            while b[at] != 255:
                kind = b[at]
                at += 1
                if kind == 0:
                    (k,) = struct.unpack_from("<Q", b, at)
                    at += 8 + k
                    (ks,) = struct.unpack_from("<I", b, at)
                    at += 4 + ks
                else:
                    (ks,) = struct.unpack_from("<I", b, at)
                    keys.append(b[at + 4:at + 4 + ks].decode())
                    at += 4 + ks
        out[js.load(open(meta))["n_tokens"]] = keys
    return ns, out


def test_disk_incr_kshift_marks_every_row():
    """(sc3fix 1) the K-shift graph ropes the K rows of every cell of every stream, the unshifted ones too (shift 0 is not
    the identity with a YaRN magnitude != 1, or a quantized K round trip). Slot 0 holds A + P1 with a committed record
    (its A blocks were reused once: non-vacuous); slot 1 context-shifts (--yarn-attn-factor 1.25: every K row of slot
    0's stream is rescaled); slot 0's next save must not reuse a block recorded before the shift. incr_verify copies
    the reusable blocks and compares their keys: a09430242 marked only the shifted cells -> stale "clean" blocks ->
    incr_verify_mismatch > 0"""
    global server
    server, _ = dc_incr_server_np("incr-kshift", {"incr_verify": "1"}, n_slots=2, kv_unified=False)
    server.extra_args += ["--context-shift", "--yarn-attn-factor", "1.25"]
    server.start()
    dc_complete(server, DC_INCR_A)
    for k, seed in enumerate((41, 42, 43)):  # P1 saves A + P0 (the record), P2 saves A + P1 (A's blocks reusable)
        dc_complete(server, DC_INCR_A + f" Part {k}. " + dc_text(seed, 4000))  # 2 * lcp < n: the divergence saves
    st = dc_settle(server)
    planned = st["incr_verify_planned"]
    assert planned > 0 and st["incr_verify_mismatch"] == 0, st
    # slot 1 (its own stream, 16384 cells): a prompt close to the slot context, then generation past it -> context shift
    n_ctx_slot = server.make_request("GET", "/props").body["default_generation_settings"]["n_ctx"]
    text = "Archive C. " + dc_text(44, 100)
    while len(server.make_request("POST", "/tokenize", data={"content": text}).body["tokens"]) < n_ctx_slot:
        text += " " + dc_text(len(text), 400)
    toks = server.make_request("POST", "/tokenize", data={"content": text}).body["tokens"][:n_ctx_slot - 200]
    res = server.make_request("POST", "/completion", data={
        "prompt": toks, "n_predict": 400, "id_slot": 1, "cache_prompt": True, "temperature": 0.0, "ignore_eos": True,
        "n_keep": 16})
    assert res.status_code == 200 and res.body["truncated"], "slot 1 did not context-shift"
    dc_complete(server, DC_INCR_A + " Part 3. " + dc_text(45, 4000))  # saves A + P2 (slot 0) after the shift
    st = dc_settle(server)
    print(f"verify planned {planned} -> {st['incr_verify_planned']}, mismatch {st['incr_verify_mismatch']}")
    assert st["incr_verify_mismatch"] == 0, "a block recorded before the K-shift was reused after it"
    assert st["incr_verify_planned"] == planned, "a block of the shifted stream was still planned for reuse"
    assert st["async_claims"] == 0 and st["saves_async_failed"] == 0
    server.stop()


def test_disk_incr_reuse_alloc_failure_leaks_no_claim():
    """(sc3fix 2) the first reused block's record allocation throws (hook fail_incr_alloc): the save fails, and no claim
    stays behind (a09430242 claimed the key before the allocation: pending[key] leaked, blob_gc would keep the block
    forever); the later save commits"""
    global server
    server, cache_dir = dc_incr_server_np("incr-allocfail", {"fail_incr_alloc": "1"})
    server.start()
    dc_incr_scenario(server)
    st = dc_settle(server)
    print(f"failed {st['saves_async_failed']} claims {st['async_claims']} reused {st['incr_blocks_reused']}")
    assert st["saves_async_failed"] == 1, st
    assert st["async_claims"] == 0, "the failed reuse left a claim behind"
    ns, ents = dc_entry_blocks(cache_dir)
    for n, keys in ents.items():
        for k in keys:
            assert os.path.exists(os.path.join(ns, "blk", k + ".blk")), (n, k)
    server.stop()


def test_disk_incr_record_failure_finishes_the_commit():
    """(sc3fix 3) building the slot's incremental record throws at the first commit (hook fail_incr_rec), after the
    entry was published: the commit still releases the job's claims and the server goes on (the record is dropped,
    the next save copies everything); a09430242 let the exception out of commit_job"""
    global server
    server, cache_dir = dc_incr_server_np("incr-recfail", {"fail_incr_rec": "1"})
    server.start()
    dc_incr_scenario(server)
    st = dc_settle(server)
    print(f"saves {st['saves']} failed {st['saves_async_failed']} claims {st['async_claims']} "
          f"reused {st['incr_blocks_reused']} entries {st['n_entries']}")
    assert st["saves_async_failed"] == 0 and st["async_claims"] == 0, st
    assert st["n_entries"] >= 3
    ns, ents = dc_entry_blocks(cache_dir)
    for n, keys in ents.items():
        for k in keys:
            assert os.path.exists(os.path.join(ns, "blk", k + ".blk")), (n, k)
    server.stop()


def test_disk_drop_of_a_claimed_block_fails_the_claimant():
    """(sc3fix 4) a block a save in flight claimed (found present by its worker) turns out corrupt at another load and
    is dropped (blob_drop_copy): the claimant must not commit an entry that references the dropped file. E1 = A + B1;
    A + B3 is loaded from E1 and saved (job 3, its end held 15 s) - its first block = E1's first block, claimed; the
    file is damaged, slot 1 loads E1 (longest prefix) -> corrupt -> dropped; job 3 commits after that. a09430242: job 3
    is registered with a missing block (every later load of it fails)"""
    global server
    server, cache_dir = dc_server("dropclaim", 2048, n_ctx=32768, n_slots=2,
                                  hooks={"finalize_delay_job3_ms": "15000", "hold_write_job4_ms": "30000"})
    server.model_draft = None
    server.spec_type = None
    server.spec_draft_n_min = None
    server.spec_draft_n_max = None
    server.start()
    ab1 = DC_INCR_A + " Part 1. " + dc_text(51, 200)
    dc_complete(server, DC_INCR_A, slot=0)
    dc_complete(server, ab1, slot=0)
    dc_complete(server, "Archive Z1. " + dc_text(52, 50), slot=0)  # job 1 saves A + B1 (E1)
    st = dc_settle(server)
    ns, ents = dc_entry_blocks(cache_dir)
    e1 = max(ents, key=lambda n: n if ents[n] else -1)
    k0 = ents[e1][0]
    # slot 1 must load E1 while job 3 is in flight: a lookup waits for an in-flight save sharing more than the slot's
    # LCP + 1024 with its prompt (await_pending), so slot 1 first holds a prefix of A + B1 long enough that job 3
    # (A + B3) is not waited for, short enough that E1 still clearly beats the slot (find: > LCP + 1024)
    a3 = DC_INCR_A + " Part 3. " + dc_text(53, 200)
    tok = lambda t: server.make_request("POST", "/tokenize", data={"content": t, "add_special": True}).body["tokens"]
    lcp = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    t_ab1, t_q = tok(ab1), tok(ab1 + " more")
    lo, hi = lcp(tok(a3), t_q) - 1024, lcp(t_ab1, t_q) - 1024 - 1
    assert 0 < lo <= hi, (lo, hi)
    dc_complete(server, t_q[:(lo + hi) // 2], n=1, slot=1)  # no save (slot 1 was empty), no job number taken
    # job 4 = slot 1's own save before its load (shares A's blocks): its first block write is held past job 3's commit,
    #   so k0 is not written back before job 3 commits (else job 3 commits normally - the after2 run)
    r = dc_complete(server, a3, slot=0)  # job 2 saves Z1; A loaded from E1
    assert r["timings"]["cache_n"] >= 4000, r["timings"]
    n_sub = st["saves"]
    dc_complete(server, "Archive Z2. " + dc_text(54, 50), slot=0)  # the save of A + B3 (claims k0), its end held
    time.sleep(3)  # its blocks are hashed / claimed by now (no per-block delay)
    st = dc_status(server, through_loop=False)
    assert st["saves_pending"] >= 1, "the A + B3 save is not in flight"
    p = os.path.join(ns, "blk", k0 + ".blk")
    with open(p, "r+b") as f:  # damage E1's first block (same size)
        f.seek(100)
        b = f.read(16)
        f.seek(100)
        f.write(bytes(x ^ 0xFF for x in b))
    r = dc_complete(server, ab1 + " more", slot=1)  # E1 is the longest prefix: its load meets the damaged block
    st = dc_status(server, through_loop=False)
    print(f"slot 1 cache_n {r['timings']['cache_n']}, load failures {st['load_failures']}, saves {n_sub} -> {st['saves']}")
    assert st["load_failures"] >= 1, "the damaged block was not met"
    assert not os.path.exists(p), "the damaged block was not dropped"
    st = dc_settle(server)
    print(f"failed {st['saves_async_failed']} claims {st['async_claims']} entries {st['n_entries']}")
    assert st["async_claims"] == 0
    assert st["saves_async_failed"] >= 1, "job 3 committed although its claimed block was dropped"
    ns, ents = dc_entry_blocks(cache_dir)
    for n, keys in ents.items():
        for k in keys:
            assert os.path.exists(os.path.join(ns, "blk", k + ".blk")), f"entry {n} references the dropped block {k}"
    server.stop()
