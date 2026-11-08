#!/usr/bin/env python3
# [TAG_CACHE_ALIGN] continuation oracle of the byte-aligned prompt cache (P16e part 3, design sec. 7.3, astra r2 OPEN 5).
#
# Per scenario, on ONE running llama-server (the unit restarts it per argv variant):
#   arm A  the scenario on the cache path: the cache is built (numeric fills / T=0 generation), then the client turn is sent
#          as TEXT; with --cache-align-bytes on the server splices it onto the cache (INFO "cache-align: ... prompt=|N| -> |S|").
#          The spliced ids S = C[:n_c] + N[n_n:] are rebuilt here from the log line (C = the slot's cached ids, N = the
#          server's tokenization of the text) and checked against the logged |N|, |S| and id LCP.
#   arm R  the same server, slot erased, cache_prompt false: prompt = S as a numeric array (ineligible = verbatim).
#   CTL    (diagnostic only, --ctl) the cache rebuilt identically, then S sent numerically with cache_prompt true (the v75
#          id-LCP path on the same warm cache). A == CTL != R localises a difference to cache-vs-fresh evaluation.
#          Criterion r3 (i): A == CTL bitwise (splice exactness).
#   CTL_off (--ctl-from DIR, knob-off run; astra r3 "matched control") the knob-on run's S (DIR/<scenario>.json) sent
#          numerically onto the identically rebuilt warm cache of THIS (knob-off) server: CTL_off == CTL_on and R_off == R_on
#          bitwise = the splice adds nothing beyond the v75 warm numeric path (C identity checked by sha256).
#   RR     (--rr) arm R run twice: R == R2 bitwise = the decoding itself is reproducible (criterion r3 (iv), MTP on).
# Sampler (both arms): T = 0, n_probs 20 (pre-sampling: softmax of the full logit row), ignore_eos, no penalties, --steps
# (256). Gate (astra r2 OPEN 5, R231): A == R BITWISE on every step: chosen id + the 20 (id, logprob) pairs (JSON doubles,
# nlohmann round-trip = the float bits). With MTP on the server fills no probs for drafted tokens (server-context.cpp
# "TODO: set result.probs"): the ids of all steps are compared and the probs where both arms have them (reported).
# Knob off (--expect off): no splice may happen (no cache-align line, cache_align_total unchanged); S = N; same A/R.
#
# Scenarios (np 1 unless noted): single (fresh single turn; parity), turn (decode-built reply with re-split prefix + next
# user turn), rewind (re-split user message, regenerate -> n_c < |C|, checkpoint restore), pin_out / pin_in (turn +
# cache_pin outside / inside the re-segmented span), edit (earlier assistant message edited after its re-splits),
# cont_tok (Continue cut inside the last cached token), cont_utf8 (Continue after a partial UTF-8 char), preencode (webui:
# chat pre-encode n_predict 0, then the real chat turn), numturn (numeric next turn on a warm cache = ineligible; parity),
# compete (np 2: two conversations sharing a prefix, concurrent text turns, slot selection), natturn (criterion r3 (iii):
# a decode-built reply resent as text WITHOUT re-splits, so the ids match the cache naturally and the v75 id-LCP path
# reuses the decode-built KV; A vs R = the cached-vs-fresh divergence of prod with the knob off), disk-build / disk-turn
# (--cache-disk: build, server restart, entry load + splice; with a key-changing env + --cache-disk-adopt = adoption).
#
# Not a CI test: needs the brain (or any ChatML model) on a dev server WITHOUT the prod --cache-disk dir; see the unit
#   ~/claude-tmp/codex/train/canon-oracle/canon-oracle.sh. Usage:
#   python3 -I cache-align-oracle.py --server http://127.0.0.1:8093 --log-cmd "docker logs ARM" --out DIR --label V1 \
#       --expect on --mtp off --scenarios single,turn,... [--ctl turn,rewind] [--steps 256]
#   python3 -I cache-align-oracle.py --parity DIR_ON/V2 DIR_OFF/V3 --scenarios single,numturn
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DELIMS = [  # = common/chat.cpp ChatML family (Qwen3.5 / Qwen3.8): what the chat path passes as message_delimiters
    {"role": "assistant", "delimiter": "<|im_start|>assistant"},
    {"role": "tool", "delimiter": "<|im_start|>user\n<tool_response>"},
    {"role": "tool", "delimiter": "<|im_start|>tool_response"},
    {"role": "user", "delimiter": "<|im_start|>user"},
    {"role": "system", "delimiter": "<|im_start|>system"},
]
GEN = "<|im_start|>assistant\n<think>\n\n</think>\n\n"  # the template's generation prompt with --reasoning off
END = "<|im_end|>\n"
PARTIAL_CHAR = "\U0001D518"  # Qwen3.8 vocab: [240,157] [148] [152] -> the cache keeps the first two tokens (3 of 4 bytes)
RE_ALIGN = re.compile(r"slot (\d+) \| task (\d+) \| cache-align: lcp_ids=(\d+) aligned_n_c=(\d+) n_n=(\d+) delta=([+-]?\d+) "
                      r"spans=(\d+) dropped=(\d+) prompt=(\d+) -> (\d+)")
RE_ALIGN_ANY = re.compile(r"cache-align: (?!on \()")
WORDS = ("the cache stores generated replies so that a following request reuses them without recomputing every token "
         "precache mechanism tokenizer boundary segmentation canonical spliced continuation oracle checkpoint "
         "recurrent state draft verification sampling logits probability distribution bitwise identical evaluation "
         "факта проверка кэша токенизатор граница ═══ ║ → ≤ ≥ данных вывод").split()


class Srv:
    def __init__(self, base, log_cmd):
        self.base = base.rstrip("/")
        self.log_cmd = log_cmd

    def post(self, ep, body, timeout=3600):
        req = urllib.request.Request(self.base + ep, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{ep}: HTTP {e.code} {e.read()[:400]!r}") from None

    def get(self, ep):
        with urllib.request.urlopen(self.base + ep, timeout=60) as r:
            return r.read().decode()

    def tok(self, s):
        return self.post("/tokenize", {"content": s, "add_special": True, "parse_special": True})["tokens"]

    def pieces(self, s):
        return self.post("/tokenize", {"content": s, "add_special": False, "parse_special": True, "with_pieces": True})["tokens"]

    def detok(self, ids):
        """text of ids; a leading BOS (what tokenize adds) is dropped so that tok(detok(ids)) does not add a second one"""
        if not hasattr(self, "bos"):
            self.bos = self.tok("")
        if self.bos and ids[:len(self.bos)] == self.bos:
            ids = ids[len(self.bos):]
        return self.post("/detokenize", {"tokens": ids})["content"]

    def metric(self, name):
        for line in self.get("/metrics").splitlines():
            if line.startswith("llamacpp:" + name + " "):
                return float(line.split()[1])
        return None

    def erase(self, slot=0):
        self.post(f"/slots/{slot}?action=erase", {})

    def log_lines(self):
        p = subprocess.run(self.log_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        return p.stdout.decode(errors="replace").splitlines()


def text_block(n_words, seed):
    r = random.Random(seed)
    out, line = [], []
    for i in range(n_words):
        line.append(r.choice(WORDS))
        if len(line) >= 14:
            out.append(" ".join(line).capitalize() + ".")
            line = []
    return "\n".join(out)


def resplit(srv, text, limit):
    """canonical pieces of text, ASCII pieces >= 6 chars split into two single tokens (cached ids != canonical ids)"""
    out, n = [], 0
    for t in srv.pieces(text):
        p = t["piece"]
        if n < limit and isinstance(p, str) and len(p) >= 6 and p.isascii():
            for k in range(3, len(p) - 1):
                x, y = srv.pieces(p[:k]), srv.pieces(p[k:])
                if len(x) == 1 and len(y) == 1:
                    out += [x[0]["id"], y[0]["id"]]
                    n += 1
                    break
            else:
                out.append(t["id"])
            continue
        out.append(t["id"])
    return out, n


def samp(steps, extra=None):
    d = {"n_predict": steps, "temperature": 0, "n_probs": 20, "ignore_eos": True, "presence_penalty": 0, "frequency_penalty": 0,
         "repeat_penalty": 1, "dry_multiplier": 0, "return_tokens": True, "message_delimiters": DELIMS}
    d.update(extra or {})
    return d


def dump_completion(r):
    probs = [[e.get("id"), e.get("logprob"), [[t.get("id"), t.get("logprob")] for t in e.get("top_logprobs", [])]]
             for e in r.get("completion_probabilities", [])]
    return {"ids": r.get("tokens", []), "probs": probs, "timings": r.get("timings", {}), "tokens_evaluated": r.get("tokens_evaluated"),
            "stop_type": r.get("stop_type")}


def dump_chat(r):
    lp = (r["choices"][0].get("logprobs") or {}).get("content") or []
    probs = [[e.get("id"), e.get("logprob"), [[t.get("id"), t.get("logprob")] for t in e.get("top_logprobs", [])]] for e in lp]
    return {"ids": [e[0] for e in probs], "probs": probs, "timings": r.get("timings", {}), "usage": r.get("usage")}


def compare(a, b, steps, mtp):
    """first difference of two dumps: (equal, where, n_prob_steps_compared)"""
    ia, ib = a["ids"], b["ids"]
    if len(ia) < steps or len(ib) < steps:
        return False, f"short: {len(ia)} / {len(ib)} ids of {steps}", 0
    for i in range(steps):
        if ia[i] != ib[i]:
            return False, f"step {i}: id {ia[i]} vs {ib[i]}", 0
    pa, pb, n = a["probs"], b["probs"], 0
    for i in range(min(len(pa), len(pb), steps)):
        ta, tb = pa[i][2], pb[i][2]
        if not ta or not tb:
            if mtp:
                continue
            return False, f"step {i}: no top_logprobs ({len(ta)} / {len(tb)}) with MTP off", n
        if len(ta) != 20 or len(tb) != 20:
            return False, f"step {i}: top_logprobs {len(ta)} / {len(tb)} != 20", n
        for k in range(20):
            if ta[k][0] != tb[k][0]:
                return False, f"step {i}: top-{k} id {ta[k][0]} vs {tb[k][0]}", n
            if ta[k][1] != tb[k][1]:
                return False, f"step {i}: top-{k} (id {ta[k][0]}) logprob {ta[k][1]!r} vs {tb[k][1]!r}", n
        if pa[i][1] != pb[i][1]:
            return False, f"step {i}: chosen logprob {pa[i][1]!r} vs {pb[i][1]!r}", n
        n += 1
    if not mtp and n < steps:
        return False, f"only {n} prob steps of {steps}", n
    return True, "", n


class Oracle:
    def __init__(self, a):
        self.a = a
        self.srv = Srv(a.server, a.log_cmd)
        self.steps = a.steps
        self.mtp = a.mtp == "on"
        self.res = []
        self.rr = set(x for x in (a.rr or "").split(",") if x)

    # ---------------------------------------------------------------- building blocks
    def fill(self, ids, slot=0):
        r = self.srv.post("/completion", {"prompt": ids, "n_predict": 0, "cache_prompt": True, "id_slot": slot,
                                          "message_delimiters": DELIMS})
        return r

    def gen(self, ids, n, slot=0):
        r = self.srv.post("/completion", samp(n, {"prompt": ids, "cache_prompt": True, "id_slot": slot, "n_probs": 0}))
        return r["tokens"]

    def head(self, n_words, seed, user):
        """ids of system + user + generation prompt, canonical"""
        return self.srv.tok(f"<|im_start|>system\n{text_block(n_words, seed)}{END}<|im_start|>user\n{user}{END}{GEN}")

    def reply_prefix(self, k=6):
        ids, n = resplit(self.srv, "Here is the precache summary: tokenizer boundaries, segmentation, continuation, "
                                   "verification, distribution and evaluation. ═══ факта", k)
        assert n >= 3, f"resplit made {n} splits"
        return ids

    def mark(self):
        return len(self.srv.log_lines())

    def since(self, m):
        return self.srv.log_lines()[m:]

    # ---------------------------------------------------------------- the arms
    def splice_of(self, lines, N, C_cands, want_slot=None):
        """the spliced ids S from the newest cache-align line whose prompt size is |N|"""
        hits = []
        for l in lines:
            g = RE_ALIGN.search(l)
            if g:
                hits.append(tuple(int(x) for x in g.groups()))
        hits = [h for h in hits if h[8] == len(N) and (want_slot is None or h[0] == want_slot)]
        if not hits:
            return None, None, "no cache-align line for |N| = %d" % len(N)
        slot, task, lcp, n_c, n_n, delta, spans, dropped, n_req, n_s = hits[-1]
        info = dict(slot=slot, task=task, lcp=lcp, n_c=n_c, n_n=n_n, delta=delta, spans=spans, dropped=dropped, n_N=n_req, n_S=n_s)
        for C in C_cands:
            if len(C) < n_c + dropped:
                continue
            C = C[:n_c + dropped]
            l2 = 0
            while l2 < min(len(C), len(N)) and C[l2] == N[l2]:
                l2 += 1
            S = C[:n_c] + N[n_n:]
            if l2 == lcp and len(S) == n_s and n_c - n_n == delta:
                return S, info, ""
        return None, info, "no candidate cache content matches the line (lcp / |S| / delta)"

    def run_A_text(self, text, extra=None, slot=0):
        return self.srv.post("/completion", samp(self.steps, dict({"prompt": text, "cache_prompt": True, "id_slot": slot}, **(extra or {}))))

    def run_R(self, S, extra=None):
        self.srv.erase(0)
        r = self.srv.post("/completion", samp(self.steps, dict({"prompt": S, "cache_prompt": False, "id_slot": 0}, **(extra or {}))))
        return dump_completion(r)

    def finish(self, name, A, S, info, err, lines, C_cands, R_extra=None, ctl_build=None, expect_splice=True, extra_info=None):
        rr = name in self.rr or name.split("_")[0] in self.rr
        rec = {"scenario": name, "label": self.a.label, "expect": self.a.expect, "mtp": self.a.mtp, "info": info, "err": err,
               "extra": extra_info or {}, "log": [l for l in lines if "cache-align" in l or "disk_cache" in l or "cache_pin" in l
                                                  or "checkpoint" in l.lower() and "restor" in l.lower()][-40:]}
        verdict, where, nprob = "FAIL", err, 0
        if S is not None:
            R = self.run_R(S, R_extra)
            ok, where, nprob = compare(A, R, self.steps, self.mtp)
            verdict = "PASS" if ok else "FAIL"
            if name == "natturn":  # a control, not a gate: it documents the knob-independent cached-vs-fresh divergence
                verdict = "CTRL-EQ" if ok else "CTRL-DIFF"
            rec["R"] = R
            if rr:
                R2 = self.run_R(S, R_extra)
                okrr, wrr, _ = compare(R, R2, self.steps, self.mtp)
                rec["R2"] = R2
                rec["rr"] = "EQ" if okrr else wrr
            if ctl_build is not None:
                ctl_build()
                C = self.srv.post("/completion", samp(self.steps, dict({"prompt": S, "cache_prompt": True, "id_slot": 0}, **(R_extra or {}))))
                C = dump_completion(C)
                okc, wc, _ = compare(A, C, self.steps, self.mtp)
                okr, wr, _ = compare(C, R, self.steps, self.mtp)
                rec["CTL"] = C
                rec["ctl"] = {"A_vs_CTL": "EQ" if okc else wc, "CTL_vs_R": "EQ" if okr else wr}
        if self.a.expect == "on" and expect_splice and (info is None or info.get("n_c", 0) <= info.get("lcp", 0)):
            verdict, where = "NOT-EXERCISED", (where or "") + " | no splice past the id LCP"
        if S is not None and ctl_build is not None and self.a.ctl_from and self.a.expect == "off":
            rec["ctl_off"] = self.ctl_off(name, ctl_build, R_extra, rec.get("R"))
        rec.update(A=A, S=S, C_sha=csha(C_cands), S_len=None if S is None else len(S), verdict=verdict, first_diff=where,
                   n_prob_steps=nprob)
        os.makedirs(self.a.out, exist_ok=True)
        with open(os.path.join(self.a.out, f"{name}.json"), "w") as f:
            json.dump(rec, f)
        ctl = dict(rec.get("ctl", {}), **({"off": rec["ctl_off"]} if "ctl_off" in rec else {}))
        line = (f"{self.a.label}\t{name}\texpect={self.a.expect}\tmtp={self.a.mtp}\t{verdict}\t{where or '-'}\tprob_steps={nprob}\t"
                f"align={'-' if not info or 'lcp' not in info else 'lcp %(lcp)d n_c %(n_c)d n_n %(n_n)d delta %(delta)+d dropped %(dropped)d |N| %(n_N)d |S| %(n_S)d slot %(slot)d' % info}"
                f"\tctl={ctl or '-'}\trr={rec.get('rr', '-')}\textra={json.dumps(extra_info or {}, ensure_ascii=True)}")
        print(line, flush=True)
        with open(os.path.join(self.a.out, "summary.tsv"), "a") as f:
            f.write(line + "\n")
        self.res.append(verdict)

    def ctl_off(self, name, ctl_build, R_extra, R_off):
        try:
            on = json.load(open(os.path.join(self.a.ctl_from, name + ".json")))
        except OSError as e:
            return {"err": f"no knob-on record: {e}"}
        if not on.get("S") or "CTL" not in on:
            return {"err": "knob-on record has no S / CTL"}
        cands = ctl_build()
        out = {"C_same": csha(cands) == on.get("C_sha")}
        Coff = dump_completion(self.srv.post("/completion", samp(self.steps, dict({"prompt": on["S"], "cache_prompt": True,
                                                                                   "id_slot": 0}, **(R_extra or {})))))
        ok, w, _ = compare(on["CTL"], Coff, self.steps, self.mtp)
        out["CTLon_vs_CTLoff"] = "EQ" if ok else w
        # R_off here is the fresh prefill of N (knob off, S = N); the matched fresh pair is a fresh prefill of the on-run S
        Roff = self.run_R(on["S"], R_extra)
        ok, w, _ = compare(on["R"], Roff, self.steps, self.mtp)
        out["Ron_vs_Roff"] = "EQ" if ok else w
        return out

    def text_turn(self, name, build, text, extra=None, R_extra_fn=None, ctl=False, expect_splice=True, extra_info=None):
        """build() -> list of candidate cache contents; text = the client turn"""
        C_cands = build()
        N = self.srv.tok(text)
        m = self.mark()
        a0 = self.srv.metric("cache_align_total")
        A = dump_completion(self.run_A_text(text, extra))
        a1 = self.srv.metric("cache_align_total")
        lines = self.since(m)
        ei = dict(extra_info or {}, align_total_delta=None if a0 is None or a1 is None else a1 - a0)
        if self.a.expect == "on":
            S, info, err = self.splice_of(lines, N, C_cands)
            if S is None and not expect_splice and not any(RE_ALIGN.search(l) for l in lines):
                S, err = N, ""  # no splice by design (empty cache / nothing past the id LCP): S = N
        else:
            S, info, err = N, None, ""
            if any(RE_ALIGN_ANY.search(l) for l in lines) or (ei["align_total_delta"] not in (0, None)):
                err = "knob off but a cache-align line / metric moved"
                S = None
        R_extra = R_extra_fn(info) if R_extra_fn else None
        self.finish(name, A, S, info, err, lines, C_cands, R_extra, (lambda: build()) if ctl else None, expect_splice, ei)

    # ---------------------------------------------------------------- scenarios
    def sc_single(self, ctl):
        sysb = text_block(900, 11)
        text = f"<|im_start|>system\n{sysb}{END}<|im_start|>user\nSummarise the text above in detail.{END}{GEN}"
        self.text_turn("single", lambda: (self.srv.erase(0), [[]])[1], text, ctl=ctl, expect_splice=False)

    def _turn_build(self, seed=21, words=2600, slot=0, user="Explain the cache design described above."):
        st = {}

        def build():
            self.srv.erase(slot)
            P = self.head(words, seed, user)
            pre = self.reply_prefix()
            g = self.gen(P + pre, 128, slot)
            st["P"], st["pre"], st["g"] = P, pre, g
            return [P + pre + g]
        return st, build

    def _turn_text(self, st, user2="Now list the open questions."):
        # the reply without its last sampled token: that token is never decoded into the cache, exactly like the <|im_end|>
        #   that ends a real reply; the client text then continues with <|im_end|> (structural, rule b of the walk)
        return self.srv.detok(st["P"] + st["pre"] + st["g"][:-1]) + f"{END}<|im_start|>user\n{user2}{END}{GEN}"

    def sc_turn(self, ctl, name="turn", pin=None):
        st, build = self._turn_build()
        C = build()
        text = self._turn_text(st)
        N = self.srv.tok(text)
        first = True

        def build2():
            nonlocal first
            if first:  # the cache is already built above (the text needs the generated reply)
                first = False
                return C
            return build()
        extra, rfn, ei = None, None, {}
        if pin == "out":
            x = len(st["P"]) - 50
            extra, rfn, ei = {"cache_pin": {"tokens": x}}, (lambda info: {"cache_pin": {"tokens": x}}), {"pin_N": x, "pin_S": x}
        elif pin == "in":
            Cf = C[0]
            l = 0
            while Cf[l] == N[l]:
                l += 1
            x = l + 1  # N position right after the first canonical token that re-merges a split pair of C
            y = l + 2  # the same byte offset in C / S (the pair C[l], C[l+1] = N[l])
            assert self.srv.detok(Cf[:y]) == self.srv.detok(N[:x]), "pin_in: byte offsets differ"
            extra, rfn, ei = {"cache_pin": {"tokens": x}}, (lambda info: {"cache_pin": {"tokens": y}}), {"pin_N": x, "pin_S": y}
        self.text_turn(name, build2, text, extra, rfn, ctl=ctl, extra_info=ei)

    def sc_turnfill(self, ctl):
        """= turn, but the reply ids are PREFILLED (numeric fill of P + prefix + the same reply ids) instead of decoded"""
        st, build0 = self._turn_build(seed=23)
        build0()  # generates the reply ids once (decode); the cache is then rebuilt by a numeric fill

        def build():
            self.srv.erase(0)
            ids = st["P"] + st["pre"] + st["g"][:-1]
            self.fill(ids)
            return [ids]
        self.text_turn("turnfill", build, self._turn_text(st), ctl=ctl)

    def sc_rewind(self, ctl):
        st = {}

        def build():
            self.srv.erase(0)
            u_ids, n = resplit(self.srv, text_block(700, 33), 40)
            P = (self.srv.tok(f"<|im_start|>system\n{text_block(5200, 31)}{END}<|im_start|>user\n") + u_ids +
                 self.srv.tok(f"{END}{GEN}"))
            g = self.gen(P, 128)
            st["P"], st["n"] = P, n
            return [P + g]
        C = build()
        text = self.srv.detok(st["P"])  # regenerate: the client resends the prompt up to the reply start
        first = [True]

        def build2():
            if first[0]:
                first[0] = False
                return C
            return build()
        self.text_turn("rewind", build2, text, ctl=ctl, extra_info={"resplits": st["n"], "C": len(C[0]), "P": len(st["P"])})

    def sc_edit(self, ctl):
        """re-splits in the first user message, the edit in the earlier assistant reply a1, then u2 / a2 (generated) / u3"""
        st = {}
        u1 = "Describe the precache design. " + text_block(260, 42)
        a1 = "The cache keeps generated replies. " + text_block(260, 41) + " Segmentation boundaries matter for continuation."

        def build():
            self.srv.erase(0)
            u1_ids, n = resplit(self.srv, u1, 30)
            P = (self.srv.tok(f"<|im_start|>system\n{text_block(1800, 43)}{END}<|im_start|>user\n") + u1_ids +
                 self.srv.tok(f"{END}{GEN}{a1}{END}<|im_start|>user\nAnd the failure modes?{END}{GEN}"))
            g = self.gen(P, 96)
            st["P"], st["g"], st["n"] = P, g, n
            return [P + g]
        C = build()
        full = self.srv.detok(C[0][:-1])
        k = full.index("Segmentation boundaries matter")
        text = full[:k] + "Segmentation EDITED boundaries matter" + full[k + len("Segmentation boundaries matter"):] + \
            f"{END}<|im_start|>user\nContinue.{END}{GEN}"
        first = [True]

        def build2():
            if first[0]:
                first[0] = False
                return C
            return build()
        self.text_turn("edit", build2, text, ctl=ctl, extra_info={"resplits": st["n"], "C": len(C[0])})

    def _cont_head(self, seed, user):
        """system + a re-split user message + generation prompt: the splice reaches the structural tokens after the user
        message (rule a); the Continue cut then lies in the canonical reply after it"""
        u_ids, n = resplit(self.srv, user + " " + text_block(300, seed + 1), 30)
        return (self.srv.tok(f"<|im_start|>system\n{text_block(2000, seed)}{END}<|im_start|>user\n") + u_ids +
                self.srv.tok(f"{END}{GEN}")), n

    def sc_cont_tok(self, ctl):
        reply = "Here is the summary of tokenizer boundaries and then the segmentation"

        def build():
            self.srv.erase(0)
            P, n = self._cont_head(51, "Write about tokenizers and precache segmentation.")
            ids = P + [t["id"] for t in self.srv.pieces(reply)]
            self.fill(ids)
            return [ids]
        C = build()
        last = self.srv.pieces(reply)[-1]["piece"]
        assert isinstance(last, str) and len(last) >= 4, last
        text = self.srv.detok(C[0][:-1]) + last[:2]  # Continue: the text ends inside the last cached token
        first = [True]

        def build2():
            if first[0]:
                first[0] = False
                return C
            return build()
        self.text_turn("cont_tok", build2, text, ctl=ctl, extra_info={"last_piece": last, "cut_to": last[:2]})

    def sc_cont_utf8(self, ctl):
        pu = [t["id"] for t in self.srv.pieces(PARTIAL_CHAR)]
        assert len(pu) >= 2, f"this vocab encodes U+1D518 as {pu}: no partial-UTF-8 token boundary"
        PARTIAL_U8 = pu[:-1]
        reply = "Here is the summary of Unicode letters, for example the letter"
        st = {}

        def build():
            self.srv.erase(0)
            P, n = self._cont_head(61, "Write about Unicode and precache segmentation.")
            ids = P + [t["id"] for t in self.srv.pieces(reply)] + PARTIAL_U8
            try:
                self.fill(ids)
            except RuntimeError as e:
                # v77: a prompt ending in an incomplete UTF-8 char makes the (empty) response fail the Content-only parse
                #   (HTTP 500) AFTER the slot processed it; the splice line of the next turn proves the cached ids
                if "Content-only" not in str(e):
                    raise
                st["fill_http500"] = True
            return [ids]
        C = build()
        text = self.srv.detok(C[0][:-len(PARTIAL_U8)])  # Continue: the client drops the incomplete UTF-8 bytes
        first = [True]

        def build2():
            if first[0]:
                first[0] = False
                return C
            return build()
        self.text_turn("cont_utf8", build2, text, ctl=ctl, extra_info={"partial_tail": PARTIAL_U8, "char_ids": pu,
                                                                          "fill_http500_content_only": st.get("fill_http500", False)})

    def sc_numturn(self, ctl):
        """numeric next turn on a warm cache: ineligible, must be the v75 path with the knob on or off (parity)"""
        st, build = self._turn_build(seed=71)
        C = build()
        N = self.srv.tok(self._turn_text(st))
        m = self.mark()
        A = dump_completion(self.srv.post("/completion", samp(self.steps, {"prompt": N, "cache_prompt": True, "id_slot": 0})))
        lines = self.since(m)
        err = "a cache-align line on a numeric request" if any(RE_ALIGN.search(l) for l in lines) else ""
        self.finish("numturn", A, None if err else N, None, err, lines, C, expect_splice=False)

    def sc_natturn(self, ctl):
        """criterion r3 (iii): decode-built reply, NO re-split prefix; the text turn's ids match the cache naturally (v75 id-LCP
        reuse of decode-built KV, knob on or off alike); A (text, warm) vs R (fresh prefill of the same ids)"""
        st = {}

        def build():
            self.srv.erase(0)
            P = self.head(2600, 27, "Explain the cache design described above.")
            g = self.gen(P, 128)
            st["P"], st["g"] = P, g
            return [P + g]
        C = build()
        text = self.srv.detok(st["P"] + st["g"][:-1]) + f"{END}<|im_start|>user\nNow list the open questions.{END}{GEN}"
        N = self.srv.tok(text)
        l = 0
        while l < min(len(C[0]), len(N)) and C[0][l] == N[l]:
            l += 1
        first = [True]

        def build2():
            if first[0]:
                first[0] = False
                return C
            return build()
        self.text_turn("natturn", build2, text, ctl=ctl, expect_splice=False,
                       extra_info={"C": len(C[0]), "id_lcp": l, "natural": l >= len(C[0]) - 1})

    def sc_preencode(self, ctl):
        sysb = text_block(2400, 81)
        msgs = [{"role": "system", "content": sysb}, {"role": "user", "content": "Explain the cache design."}]
        self.srv.erase(0)
        P = self.srv.tok(self.srv.post("/apply-template", {"messages": msgs})["prompt"])
        pre = self.reply_prefix()
        g = self.gen(P + pre, 128)
        reply = self.srv.detok(pre + g[:-1])
        C0 = P + pre + g
        msgs1 = msgs + [{"role": "assistant", "content": reply}]
        csamp = {"temperature": 0, "ignore_eos": True, "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1,
                 "dry_multiplier": 0, "logprobs": True, "top_logprobs": 20}
        m = self.mark()
        self.srv.post("/v1/chat/completions", dict(csamp, messages=msgs1, n_predict=0))  # the webui pre-encode (ChatService.preEncode)
        lines_pre = self.since(m)
        N_pre = self.srv.tok(self.srv.post("/apply-template", {"messages": msgs1})["prompt"])
        S_pre, info_pre, err_pre = self.splice_of(lines_pre, N_pre, [C0])
        msgs2 = msgs1 + [{"role": "user", "content": "Now list the open questions."}]
        N = self.srv.tok(self.srv.post("/apply-template", {"messages": msgs2})["prompt"])
        m = self.mark()
        A = dump_chat(self.srv.post("/v1/chat/completions", dict(csamp, messages=msgs2, max_tokens=self.steps)))
        lines = self.since(m)
        cands = ([S_pre] if S_pre else []) + [C0, N_pre]
        S, info, err = self.splice_of(lines, N, cands)
        if S is None and not any(RE_ALIGN.search(l) for l in lines):
            # the pre-encode already spliced and the real turn extends the spliced cache by plain id LCP (no second splice)
            base = S_pre if S_pre else C0
            l = 0
            while l < min(len(base), len(N)) and base[l] == N[l]:
                l += 1
            S, err = (N if l < len(base) else base + N[len(base):]), ""
            info = dict(info_pre or {}, note="real turn: no second splice; S = cache(after pre-encode) + N tail by id LCP")
            if l < len(base):
                info["note"] = f"real turn: id LCP {l} < cache {len(base)}, S = N (re-processed)"
        self.finish("preencode", A, S, info, err, lines_pre + lines, cands, None, None, S_pre is None,
                    {"pre_encode": info_pre, "pre_err": err_pre})

    def sc_compete(self, ctl):
        """np 2: conversations X (slot 0) and Y (slot 1) share the system prompt; concurrent text turns, id_slot -1"""
        stx, bx = self._turn_build(seed=91, slot=0, user="Explain the cache design described above.")
        sty, by = self._turn_build(seed=91, slot=1, user="Explain the checkpoint design described above, briefly please.")
        Cx, Cy = bx(), by()
        tx = self._turn_text(stx, "Now list the open questions.")
        ty = self._turn_text(sty, "Now list the risks, in order of severity, with one line each.")
        Nx, Ny = self.srv.tok(tx), self.srv.tok(ty)
        assert len(Nx) != len(Ny)
        m = self.mark()
        out = {}

        def go(k, t):
            out[k] = dump_completion(self.srv.post("/completion", samp(self.steps, {"prompt": t, "cache_prompt": True, "id_slot": -1})))
        th = [threading.Thread(target=go, args=("x", tx)), threading.Thread(target=go, args=("y", ty))]
        [t.start() for t in th]
        [t.join() for t in th]
        lines = self.since(m)
        for k, N, C in (("x", Nx, Cx), ("y", Ny, Cy)):
            S, info, err = self.splice_of(lines, N, C)
            ei = {"want_slot": 0 if k == "x" else 1}
            if info and info["slot"] != ei["want_slot"]:
                err = (err + "; " if err else "") + f"selected slot {info['slot']} != {ei['want_slot']}"
            self.finish(f"compete_{k}", out[k], S, info, err, lines, C, extra_info=ei)

    def sc_disk_build(self, ctl):
        st, build = self._turn_build(seed=101)
        C = build()
        with open(os.path.join(self.a.out, "..", "disk-state.json"), "w") as f:
            json.dump({"C": C[0], "text": self._turn_text(st)}, f)
        print(f"{self.a.label}\tdisk-build\tC={len(C[0])}\t(saved by the server at SIGTERM)", flush=True)

    def sc_disk_turn(self, ctl, name="disk_turn"):
        d = json.load(open(os.path.join(self.a.out, "..", "disk-state.json")))
        cands = [d["C"]] + ([d["S"] + d["g"]] if "S" in d else [])
        m = self.mark()
        A = dump_completion(self.run_A_text(d["text"]))
        lines = self.since(m)
        N = self.srv.tok(d["text"])
        S, info, err = self.splice_of(lines, N, cands)
        loaded = [l for l in lines if "disk_cache" in l and ("load" in l or "adopt" in l or "restor" in l)]
        if not loaded:
            err = (err + "; " if err else "") + "no disk_cache load line"
        if S is not None and "S" not in d:
            d["S"], d["g"] = S, A["ids"]
            with open(os.path.join(self.a.out, "..", "disk-state.json"), "w") as f:
                json.dump(d, f)
        self.finish(name, A, S, info, err, lines, cands, extra_info={"disk_lines": loaded[-6:]})

    def sc_disk_adopt(self, ctl):
        self.sc_disk_turn(ctl, "disk_adopt")


def csha(cands):
    return hashlib.sha256(json.dumps(cands[0] if cands else []).encode()).hexdigest()[:16]


def parity(a):
    fails = 0
    for sc in a.scenarios.split(","):
        x = json.load(open(os.path.join(a.parity[0], sc + ".json")))
        y = json.load(open(os.path.join(a.parity[1], sc + ".json")))
        for arm in ("A", "R"):
            if arm not in x or arm not in y:
                print(f"parity\t{sc}\t{arm}\tMISSING")
                fails += 1
                continue
            ok, where, n = compare(x[arm], y[arm], a.steps, x["mtp"] == "on")
            print(f"parity\t{sc}\t{arm}\t{x['label']}({x['expect']}) vs {y['label']}({y['expect']})\t{'PASS' if ok else 'FAIL'}\t{where or '-'}\tprob_steps={n}")
            fails += not ok
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8093")
    ap.add_argument("--log-cmd", default="docker logs piagent-llm-canon-oracle")
    ap.add_argument("--out")
    ap.add_argument("--label", default="run")
    ap.add_argument("--expect", choices=["on", "off"], default="on")
    ap.add_argument("--mtp", choices=["on", "off"], default="on")
    ap.add_argument("--scenarios", default="single,turn,turnfill,rewind,pin_out,pin_in,edit,cont_tok,cont_utf8,preencode,numturn")
    ap.add_argument("--ctl", default="", help="scenarios that also run the CTL diagnostic")
    ap.add_argument("--ctl-from", default="", help="knob-off run: the knob-on run dir whose S / CTL / R the matched control uses")
    ap.add_argument("--rr", default="", help="scenarios whose arm R runs twice (reproducibility control)")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--parity", nargs=2)
    a = ap.parse_args()
    if a.parity:
        sys.exit(1 if parity(a) else 0)
    os.makedirs(a.out, exist_ok=True)
    o = Oracle(a)
    ctl = set(x for x in a.ctl.split(",") if x)
    for sc in a.scenarios.split(","):
        t0 = time.time()
        fn = {"pin_out": lambda c: o.sc_turn(c, "pin_out", "out"), "pin_in": lambda c: o.sc_turn(c, "pin_in", "in"),
              "disk-build": o.sc_disk_build, "disk-turn": o.sc_disk_turn, "disk-adopt": o.sc_disk_adopt}.get(sc) or getattr(o, "sc_" + sc)
        try:
            fn(sc in ctl)
        except Exception as e:  # noqa: BLE001 - one scenario's error must not hide the others
            line = f"{a.label}\t{sc}\texpect={a.expect}\tmtp={a.mtp}\tERROR\t{type(e).__name__}: {e}"
            print(line, flush=True)
            with open(os.path.join(a.out, "summary.tsv"), "a") as f:
                f.write(line + "\n")
            o.res.append("ERROR")
        print(f"# {sc}: {time.time() - t0:.1f} s", flush=True)
    bad = [v for v in o.res if v not in ("PASS", "CTRL-EQ", "CTRL-DIFF")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
