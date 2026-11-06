#!/usr/bin/env python3
# [TAG_CACHE_ALIGN] server-level replay of the three logged v75 re-segmentation incidents (P16e, design §7.2)
#
# For each incident: the cached ids C (the disk-cache .tok entry: prompt + the reply the model sampled) are put into
# slot 0 as a numeric prompt (n_predict 0); then the client turn is replayed as TEXT: detokenize(C) + the next user
# turn, which the server re-tokenizes canonically (N). Checked from the second response's timings:
#   --expect on : cache_n == |C| (0 dropped: the cached reply is kept, n_past restored to the cache end),
#                 prompt_n <= --max-new (only the new turn is processed), llamacpp:cache_align_total +1
#   --expect off: cache_n <= the id LCP (v75: re-processing from the first re-segmented token, a recurrent model may
#                 restore an earlier checkpoint), prompt_n >= |N| - id LCP, cache_align_total unchanged
#
# Not a CI test (needs the brain vocab, the .tok files and a real model). Run on s1 against a dev image of the branch,
# never next to prod, through a batch unit, WITHOUT --cache-disk (gate arms never share the prod cache dir):
#   1. dev image of p16e-align; start one llama-server with the brain profile minus --cache-disk*, plus --metrics,
#      --slots, -np 1, and either --cache-align-bytes on or off (or LLAMA_ARG_CACHE_ALIGN_BYTES=on|off)
#   2. scripts/bench/batch-run.sh cache-align-replay-on -- python3 -I tools/server/tests/cache-align-replay.py \
#        --server http://127.0.0.1:8080 --expect on [--case ba543f058ab248da]
#   3. restart with the knob off, same command with --expect off
# The 1M brain prefills 124k / 257k / 336k tokens per case (minutes each); --case runs one.
import argparse
import json
import struct
import sys
import urllib.request

TOK_DIR = "/mnt/data/piagent-slots/prompt-cache/qwen38-flash-next-uncensored-brain-1m/ns-36557823e9dde84e/"
CASES = [  # key, reply start, |C|, id LCP (design §1, test-cache-align)
    ("ba543f058ab248da", 89623, 124316, 92653),
    ("95ea69c1eb053e22", 234824, 257289, 238290),
    ("a32a45b7e9f14c89", 306875, 336262, 310735),
]
NEXT_TURN = "<|im_end|>\n<|im_start|>user\nnext turn"


def post(base, ep, body, timeout=7200):
    req = urllib.request.Request(base + ep, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metric(base, name):
    with urllib.request.urlopen(base + "/metrics", timeout=60) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("llamacpp:" + name + " "):
                return float(line.split()[1])
    return None


def read_tok(path):
    b = open(path, "rb").read()
    n = struct.unpack("<i", b[8:12])[0]
    return list(struct.unpack("<%di" % n, b[12:12 + 4 * n]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--tok-dir", default=TOK_DIR)
    ap.add_argument("--expect", choices=["on", "off"], required=True)
    ap.add_argument("--case", action="append", help="entry key (default: all three)")
    ap.add_argument("--max-new", type=int, default=64, help="on: the most prompt tokens the new turn may process")
    a = ap.parse_args()
    base = a.server.rstrip("/")

    fails = 0
    for key, reply_start, n_cache, id_lcp in CASES:
        if a.case and key not in a.case:
            continue
        C = read_tok(a.tok_dir + key + ".tok")
        assert len(C) == n_cache, (key, len(C))
        text = post(base, "/detokenize", {"tokens": C})["content"] + NEXT_TURN
        N = post(base, "/tokenize", {"content": text, "add_special": True, "parse_special": True})["tokens"]
        lcp = 0
        while lcp < min(len(C), len(N)) and C[lcp] == N[lcp]:
            lcp += 1

        try:
            post(base, "/slots/0?action=erase", {})
        except Exception as e:  # noqa: BLE001 - erase needs --slots; a fresh server has an empty slot anyway
            print(f"{key}: slot erase failed ({e}), continuing", file=sys.stderr)
        r1 = post(base, "/completion", {"prompt": C, "n_predict": 0, "cache_prompt": True, "id_slot": 0})
        m0 = metric(base, "cache_align_total")
        r2 = post(base, "/completion", {"prompt": text, "n_predict": 1, "cache_prompt": True, "id_slot": 0,
                                        "temperature": 0})
        m1 = metric(base, "cache_align_total")
        t = r2.get("timings", {})
        cache_n, prompt_n = t.get("cache_n"), t.get("prompt_n")
        d_align = None if m0 is None or m1 is None else m1 - m0
        print(f"{key}: |C| {len(C)} |N| {len(N)} id-lcp {lcp} (design {id_lcp}) | fill prompt_n {r1.get('timings', {}).get('prompt_n')}"
              f" | turn cache_n {cache_n} prompt_n {prompt_n} cache_align_total +{d_align}")
        if a.expect == "on":
            ok = cache_n == len(C) and prompt_n is not None and prompt_n <= a.max_new and d_align == 1
        else:
            ok = cache_n is not None and cache_n <= lcp and prompt_n >= len(N) - lcp and d_align in (0, None)
        print(f"{key}: {'PASS' if ok else 'FAIL'} (expect {a.expect})")
        fails += 0 if ok else 1
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
