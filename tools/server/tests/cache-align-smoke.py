#!/usr/bin/env python3
# [TAG_CACHE_ALIGN] CPU smoke of the byte-aligned splice through a real server (any small model whose template has a
# single-token <|im_end|>, e.g. LFM2-350M). Start the server with -np 1 -c N_CTX --metrics and --cache-align-bytes on|off:
#   llama-server -m LFM2-350M-Q8_0.gguf -c 512 -np 1 --metrics --cache-align-bytes on --port 18931
#   python3 -I tools/server/tests/cache-align-smoke.py --server http://127.0.0.1:18931 --n-ctx 512 --expect on
# With an mmproj loaded (e.g. LFM2-VL-450M + its mmproj; every server_tokens then has has_mtmd = true) add --media: the
#   text checks must give the same results, plus an image turn and a text follow-up on the image cache stay healthy.
#   The server needs LLAMA_MEDIA_MARKER='<__media__>' in its environment (the /completion prompt_string marker).
# Checks
#   splice   : a numeric fill with forced re-splits, then the same text + a next turn: on = n_past == |C|, a few new
#              tokens, cache_align_total +1; off = the v75 re-processing. Both: usage prompt_tokens == |N| (review r1 3)
#   admission: a text prompt of n_ctx - 1 tokens whose splice would be longer (delta > 0) is accepted warm exactly as cold
#              (on: the splice is declined for context, cache_align_declined_total +1) (review r1 3)
#   media    : (--media) an image turn, then the same image + a longer text: both 200, the server stays healthy
import argparse
import base64
import struct
import zlib
import json
import sys
import urllib.error
import urllib.request


def post(base, ep, body):
    req = urllib.request.Request(base + ep, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def metric(base, name):
    with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("llamacpp:" + name + " "):
                return float(line.split()[1])
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:18931")
    ap.add_argument("--n-ctx", type=int, required=True)
    ap.add_argument("--expect", choices=["on", "off"], required=True)
    ap.add_argument("--media", action="store_true", help="the server has an mmproj loaded: also run the image turns")
    a = ap.parse_args()
    B = a.server.rstrip("/")

    def tok(s, special=True):
        return post(B, "/tokenize", {"content": s, "add_special": True, "parse_special": special, "with_pieces": True})[1]["tokens"]

    def piece_ids(s):
        return [t["id"] for t in post(B, "/tokenize", {"content": s, "with_pieces": True})[1]["tokens"]]

    def resplit(toks, limit):
        out, n = [], 0
        for t in toks:
            p = t["piece"]
            if n < limit and isinstance(p, str) and len(p) >= 6 and p.isascii():
                x, y = piece_ids(p[:3]), piece_ids(p[3:])
                if len(x) == 1 and len(y) == 1 and x[0] != t["id"]:
                    out += [x[0], y[0]]
                    n += 1
                    continue
            out.append(t["id"])
        return out, n

    def fill(ids):
        post(B, "/slots/0?action=erase", {})
        st, r = post(B, "/completion", {"prompt": ids, "n_predict": 0, "cache_prompt": True, "id_slot": 0})
        assert st == 200, r

    def turn(text):
        return post(B, "/completion", {"prompt": text, "n_predict": 2, "cache_prompt": True, "id_slot": 0, "temperature": 0})

    fails = 0
    reply = " ".join(["The precache mechanism stores generated replies so that a following request reuses them."] * 6)

    # splice
    C, n_res = resplit(tok(reply), 1000)
    text = reply + "<|im_end|>\n<|im_start|>user\nnext turn"
    N = [t["id"] for t in tok(text)]
    fill(C)
    m0 = metric(B, "cache_align_total")
    st, r = turn(text)
    d = metric(B, "cache_align_total") - m0
    t = r.get("timings", {})
    ok = st == 200 and r.get("tokens_evaluated", len(N)) == len(N)
    ok = ok and ((t.get("cache_n") == len(C) and t.get("prompt_n", 99) <= 16 and d == 1) if a.expect == "on" else d == 0)
    print(f"splice: |C| {len(C)} ({n_res} re-splits) |N| {len(N)} -> status {st} cache_n {t.get('cache_n')} prompt_n "
          f"{t.get('prompt_n')} tokens_evaluated {r.get('tokens_evaluated')} align +{d}: {'PASS' if ok else 'FAIL'}")
    fails += not ok

    # admission at n_ctx - 1: C = the re-split head, N = head + <|im_end|> + filler up to n_ctx - 1 tokens
    head = " ".join(["The precache mechanism stores replies."] * 4)
    C, n_res = resplit(tok(head), 4)
    filler_n = a.n_ctx
    text = None
    for _ in range(64):
        cand = head + "<|im_end|>\n" + " x" * filler_n
        n = len(tok(cand))
        if n == a.n_ctx - 1:
            text = cand
            break
        filler_n += (a.n_ctx - 1) - n
    assert text is not None, "could not build a prompt of n_ctx - 1 tokens"
    N = [t["id"] for t in tok(text)]
    post(B, "/slots/0?action=erase", {})
    st_cold, r_cold = turn(text)
    fill(C)
    dec0 = metric(B, "cache_align_declined_total")
    st_warm, r_warm = turn(text)
    dec = metric(B, "cache_align_declined_total") - dec0
    ok = st_cold == st_warm and r_cold.get("tokens_evaluated") == r_warm.get("tokens_evaluated") == len(N)
    ok = ok and (dec == 1 if a.expect == "on" else dec == 0)
    print(f"admission: |N| {len(N)} (n_ctx {a.n_ctx}), |C| {len(C)} ({n_res} re-splits): cold {st_cold} warm {st_warm} "
          f"tokens_evaluated {r_cold.get('tokens_evaluated')}/{r_warm.get('tokens_evaluated')} declined +{dec}: "
          f"{'PASS' if ok else 'FAIL'}")
    fails += not ok

    if a.media:
        img = base64.b64encode(png(32, 32)).decode()
        p1 = "<|im_start|>user\n<__media__>\nDescribe the precache picture.<|im_end|>\n<|im_start|>assistant\n"
        post(B, "/slots/0?action=erase", {})
        st1, r1 = post(B, "/completion", {"prompt": {"prompt_string": p1, "multimodal_data": [img]}, "n_predict": 4,
                                          "cache_prompt": True, "id_slot": 0, "temperature": 0})
        p2 = p1 + "It is a square.<|im_end|>\n<|im_start|>user\nnext turn<|im_end|>\n<|im_start|>assistant\n"
        st2, r2 = post(B, "/completion", {"prompt": {"prompt_string": p2, "multimodal_data": [img]}, "n_predict": 4,
                                          "cache_prompt": True, "id_slot": 0, "temperature": 0})
        st3, _ = post(B, "/completion", {"prompt": p2.replace("<__media__>", "a picture"), "n_predict": 2, "cache_prompt": True,
                                         "id_slot": 0, "temperature": 0})
        with urllib.request.urlopen(B + "/health", timeout=30) as h:
            alive = h.status == 200
        ok = st1 == 200 and st2 == 200 and st3 == 200 and alive
        print(f"media: image turn {st1}, image follow-up {st2} (cache_n {r2.get('timings', {}).get('cache_n')}), text turn {st3}, "
              f"health {alive}: {'PASS' if ok else 'FAIL'}")
        fails += not ok
    sys.exit(1 if fails else 0)


def png(w, h):
    raw = b"".join(b"\x00" + bytes(v for x in range(w) for v in (x * 8 % 256, y * 8 % 256, 128)) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    main()
