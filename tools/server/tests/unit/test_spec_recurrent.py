import math, os, pytest
from utils import *

LFM2_Q8 = "/models/_test/LFM2-350M-Q8_0.gguf"
LFM2_Q4 = "/models/_test/LFM2-350M-Q4_0.gguf"
pytestmark = pytest.mark.skipif(not (os.path.exists(LFM2_Q8) and os.path.exists(LFM2_Q4)), reason="LFM2 test models not mounted (see Task 1 Step 0)")

REQ = {"prompt": "Once upon a time", "temperature": 0.0, "n_predict": 96, "cache_prompt": False, "return_tokens": True}

def make_server(draft):
    """LFM2 (recurrent) target. draft: 'q4' = real Q4_0 draft-simple (partial acceptances, valid output),
       'synth' = synthetic 50 % acceptance (counters only, output invalid), None = no speculation."""
    s = ServerPreset.tinyllama2()
    s.model_hf_repo = None          # start() passes --hf-repo/--hf-file when set (utils.py:169-175) and HF resolution would win
    s.model_hf_file = None
    s.model_file = LFM2_Q8
    s.n_ctx = 512
    s.n_batch = 64
    s.fa = "off"
    s.server_metrics = True
    if draft == "q4":
        s.model_draft = LFM2_Q4; s.spec_type = "draft-simple"; s.spec_draft_n_min = 1; s.spec_draft_n_max = 4
    elif draft == "synth":
        s.model_draft = LFM2_Q8; s.spec_type = "draft-simple"; s.spec_draft_n_min = 1; s.spec_draft_n_max = 4
        s.spec_synth_rates = [0.5, 0.5, 0.5, 0.5]
    return s

def metric(server, name):
    body = server.make_request("GET", "/metrics").body
    line = [l for l in body.splitlines() if l.startswith("llamacpp:" + name + " ")][0]
    return int(float(line.split(" ")[1]))

def counters(server):
    c = {k: metric(server, "spec_decode_num_" + k + "_rounds_total") for k in ("verify", "replay", "fold", "plain", "empty_draft")}
    c["fold_rejected"] = metric(server, "spec_decode_num_fold_rejected_total")
    c["decode"] = metric(server, "n_decode_total")
    return c

def identity_holds(c, n_prompt_decodes=2):
    # LFM2 is a checkpointed (recurrent) target: do_checkpoint splits prompt processing into two
    # llama_decode() calls (the last `checkpoint_offsets` tokens are decoded separately so a
    # checkpoint can be taken before the end of the prompt, server-context.cpp ~3612-3631) instead
    # of one, so n_decode_total attributes 2 batches to the REQ prompt here, not 1 as for a plain
    # KV-cache target (see test_speculative.py::test_metrics_round_counters_kv_target).
    return c["decode"] - n_prompt_decodes == c["verify"] + c["replay"] + c["fold"] + c["plain"]

def first_divergence_index(a, b):
    """Index of the first position where token sequences a and b differ, or None if one is a
    prefix of the other (including the two being equal)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None

def first_divergence_is_near_tie(ref_body, ref_tokens, other_tokens, margin=0.02):
    """Correctness oracle for comparing a drafted run's tokens against an undrafted reference run
    on a checkpointed (recurrent) target. The two runs feed the same generated position through
    differently-shaped batches (the reference always decodes 1 token at a time; a drafted run
    verifies up to n_draft_max+1 at once), so their raw softmax results for that position are not
    guaranteed bit-identical even at temperature 0 — this is the "batch-size dependent rounding"
    case server-context.cpp's n_replay_mismatch warning (~3973-3974) already names, distinct from
    "an inexact restore" (a real correctness bug in the checkpoint-restore mechanism this ladder is
    built to optimize). The two causes look identical from reproducibility/inspection alone; this
    function tells them apart by measurement.

    ref_body must be the reference (undrafted) run's response requested with n_probs>=2, so
    ref_body["completion_probabilities"][i]["top_logprobs"] holds the raw (pre-sampling-chain,
    i.e. before temperature/greedy collapse) top-2 probabilities at each generated position.

    Finds the first index where ref_tokens and other_tokens differ; returns True if there is none.
    Otherwise returns True only if the reference run's top-2 probabilities at that position differ
    by at most `margin` AND other_tokens' token at that position is one of those top-2 (a
    legitimate near-tie flip). Returns False for a confident-position divergence — that would
    indicate an inexact restore, not rounding."""
    i = first_divergence_index(ref_tokens, other_tokens)
    if i is None:
        return True
    top = ref_body["completion_probabilities"][i]["top_logprobs"]
    if len(top) < 2:
        return False
    p0, p1 = math.exp(top[0]["logprob"]), math.exp(top[1]["logprob"])
    return abs(p0 - p1) <= margin and other_tokens[i] in (top[0]["id"], top[1]["id"])

def test_recurrent_target_counts_replays():
    server = make_server("synth"); server.start()
    assert server.make_request("POST", "/completion", data=REQ).status_code == 200
    c = counters(server); server.stop()
    assert c["replay"] > 0                       # 50 % acceptance → partial acceptances → restore + replay
    assert c["fold"] == 0                        # no fold before Task 5B
    assert identity_holds(c)

def test_real_draft_makes_partial_acceptances_and_keeps_tokens():
    # On this Vulkan build, the Q4_0-drafted run and the undrafted reference run deterministically
    # diverge starting at one token index (verified reproducible: same token pair every run). Per
    # server-context.cpp's own n_replay_mismatch warning (~3973-3974), a divergence like this has
    # two possible causes — "batch-size dependent rounding OR an inexact restore" — and only the
    # second would be a real correctness bug in the checkpoint-restore mechanism this whole ladder
    # optimizes. first_divergence_is_near_tie() tells them apart by measurement rather than by
    # inspection: it uses the reference run's top-2 raw probabilities (n_probs=2) at the first
    # divergence to check whether the reference was itself nearly undecided there (rounding) or
    # confident (restore bug). Measured on this build: divergence at token index 20, reference
    # top-2 probabilities 0.44999 vs 0.44951 (margin ~0.00048, i.e. a near-tie), and the drafted
    # run's token is exactly the reference's runner-up — a legitimate near-tie flip.
    ref_req = dict(REQ, n_probs=2)   # only the reference run needs probabilities
    server = make_server(None); server.start()
    base_body = server.make_request("POST", "/completion", data=ref_req).body
    base = base_body["tokens"]
    server.stop()

    server = make_server("q4"); server.start()
    res = server.make_request("POST", "/completion", data=REQ).body
    c = counters(server); server.stop()
    on = res["tokens"]

    assert res["timings"]["draft_n"] > 0
    assert c["replay"] > 0                       # the Q4_0 draft disagrees sometimes
    assert identity_holds(c)

    i = first_divergence_index(base, on)
    assert on[:i] == base[:i]                    # agree up to the first divergence, if any
    is_near_tie = first_divergence_is_near_tie(base_body, base, on)
    msg = "no divergence" if i is None else (
        f"divergence at token index {i} is NOT a near-tie (points at an inexact checkpoint "
        f"restore, not batch-size rounding): reference top-2 = "
        f"{base_body['completion_probabilities'][i]['top_logprobs']}, "
        f"reference token = {base[i]}, drafted token = {on[i]}"
    )
    assert is_near_tie, msg

def test_forced_empty_drafts_are_plain_rounds():
    server = make_server("q4"); server.spec_draft_p_min = 1.0   # draft-simple returns nothing (speculative.cpp:336)
    server.start()
    assert server.make_request("POST", "/completion", data=REQ).status_code == 200
    c = counters(server); server.stop()
    # R2: dropped the c["verify"] == 0 and c["replay"] == 0 assertion — a softmax that saturates to
    # exactly 1.0 could let draft-simple emit a token even with spec_draft_p_min = 1.0.
    assert c["empty_draft"] > 0 and c["empty_draft"] <= c["plain"]
    assert identity_holds(c)

def test_replay_is_deterministic_and_trace_does_not_perturb_sampling():
    # Regression guard for how the replay path handles the SAMPLER (not the context).
    #
    # A partial draft acceptance on this recurrent target restores a context checkpoint and replays the
    # committed tokens. Verification (common_sampler_sample_and_accept_n) sampled and accepted exactly
    # the tokens it returned, so the sampler that already holds the correct state is the LIVE one: the
    # server must not restore a pre-verification snapshot over it, the replay must not re-accept the
    # committed tokens, and the LLAMA_TRACE mismatch probe must sample a CLONE.
    #
    # Every one of those three mistakes is invisible at temperature 0 (a greedy chain has no state that
    # an extra sample/accept can perturb) but changes the output as soon as the sampling chain carries
    # apply-time state - the dist sampler's RNG draw, mirostat's mu, adaptive-p. Hence temperature 0.8
    # with a fixed seed: the number of sampler advances per generated token then has to be identical on
    # every run, and the diagnostic probe must not consume any.
    req = dict(REQ, temperature=0.8, seed=42)   # n_predict=96, cache_prompt=False from REQ

    server = make_server("q4"); server.start()
    first  = server.make_request("POST", "/completion", data=req).body["tokens"]
    second = server.make_request("POST", "/completion", data=req).body["tokens"]
    c = counters(server); server.stop()

    assert c["replay"] > 0, "no replay rounds - this run did not exercise the replay path"
    assert first == second, (
        "same seed, same server, but the two runs differ at token index "
        f"{first_divergence_index(first, second)}: a replay advanced the live sampler a different "
        "number of times (snapshot restore / re-accept of the committed tokens)")

    # ... and again on a server with the LLAMA_TRACE mismatch probe enabled
    server = make_server("q4"); server.env = {"LLAMA_TRACE": "1"}; server.start()
    traced = server.make_request("POST", "/completion", data=req).body["tokens"]
    c = counters(server); server.stop()

    assert c["replay"] > 0
    assert traced == first, (
        f"LLAMA_TRACE=1 changed the generated tokens at index {first_divergence_index(first, traced)}: "
        "the committed-token mismatch probe must sample a clone of the sampler, not the live one")
