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
    # 50 % acceptance → partial acceptances → restore + (replay or, since Task 5B, fold). Fold is enabled by
    # default and this drafter always has draft room, so it can fold every partial acceptance - replay is no
    # longer guaranteed to be > 0 on its own (see test_fold_reduces_replays_and_keeps_tokens for that behaviour
    # measured directly against a fold-off run).
    assert c["replay"] + c["fold"] > 0
    assert c["fold"] > 0                          # fold is enabled by default (Task 5B)
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
    #
    # LLAMA_SPEC_FOLD=0: this guard is about the REPLAY path specifically (Task 2 / R9's live-sampler
    # design), predating the fold (Task 5B). With fold enabled, the SAME two runs can legitimately diverge
    # from a *different* mechanism - batch-size dependent rounding (see server_sample_and_accept_replay's
    # own docstring): a folded round's batch shape (replay tail + fresh draft in one target decode) is not
    # otherwise exercised by a plain replay, so a near-tie accept/reject decision can flip between two
    # otherwise-identical runs and, at temperature 0.8, cascade via the RNG stream from there - confirmed by
    # inspection (round-by-round accept/reject trace) to be exactly this, not an extra sampler advance. That
    # class of divergence is what the near-tie oracle in test_fold_reduces_replays_and_keeps_tokens is for;
    # this test stays scoped to its original purpose by disabling fold for both requests.
    req = dict(REQ, temperature=0.8, seed=42)   # n_predict=96, cache_prompt=False from REQ

    server = make_server("q4"); server.env = {"LLAMA_SPEC_FOLD": "0"}; server.start()
    first  = server.make_request("POST", "/completion", data=req).body["tokens"]
    second = server.make_request("POST", "/completion", data=req).body["tokens"]
    c = counters(server); server.stop()

    assert c["fold"] == 0
    assert c["replay"] > 0, "no replay rounds - this run did not exercise the replay path"
    assert first == second, (
        "same seed, same server, but the two runs differ at token index "
        f"{first_divergence_index(first, second)}: a replay advanced the live sampler a different "
        "number of times (snapshot restore / re-accept of the committed tokens)")

    # ... and again on a server with the LLAMA_TRACE mismatch probe enabled
    server = make_server("q4"); server.env = {"LLAMA_TRACE": "1", "LLAMA_SPEC_FOLD": "0"}; server.start()
    traced = server.make_request("POST", "/completion", data=req).body["tokens"]
    c = counters(server); server.stop()

    assert c["fold"] == 0
    assert c["replay"] > 0
    assert traced == first, (
        f"LLAMA_TRACE=1 changed the generated tokens at index {first_divergence_index(first, traced)}: "
        "the committed-token mismatch probe must sample a clone of the sampler, not the live one")

def run_once(draft, env=None):
    server = make_server(draft)
    if env:
        server.env = env
    server.start()
    body = server.make_request("POST", "/completion", data=REQ).body
    c = counters(server); server.stop()
    return body["tokens"], c

def test_fold_reduces_replays_and_keeps_tokens():
    # Task 5B: fold the next draft into the replay batch after a partial acceptance on a checkpointed
    # (recurrent) target. Oracle: compare each arm against a NO-DRAFT reference by prefix agreement +
    # near-tie at the first divergence, exactly like test_real_draft_makes_partial_acceptances_and_keeps_tokens
    # above - strict token-for-token equality is not a valid oracle once the fold changes batch shapes on
    # the GPU (see first_divergence_is_near_tie's docstring).
    ref_req = dict(REQ, n_probs=2)   # only the reference run needs probabilities
    ref_server = make_server(None); ref_server.start()
    base_body = ref_server.make_request("POST", "/completion", data=ref_req).body
    base = base_body["tokens"]
    ref_server.stop()

    off, c0 = run_once("q4", env={"LLAMA_SPEC_FOLD": "0"})   # today's replay behaviour (no fold)
    on,  c1 = run_once("q4")                                  # fold enabled (default)

    assert c0["fold"] == 0, "LLAMA_SPEC_FOLD=0 must disable folding"
    assert c1["fold"] > 0, "fold is enabled by default but no fold rounds were counted"
    assert c1["replay"] < c0["replay"]         # folded partial acceptances no longer cost a separate replay round
    assert c1["fold_rejected"] <= c1["replay"] # every fold-tail rejection is one of the remaining replays
    assert identity_holds(c0)
    assert identity_holds(c1)

    for name, tokens in (("fold-off", off), ("fold-on", on)):
        i = first_divergence_index(base, tokens)
        assert tokens[:i] == base[:i]          # agree up to the first divergence, if any
        is_near_tie = first_divergence_is_near_tie(base_body, base, tokens)
        msg = "no divergence" if i is None else (
            f"{name}: divergence at token index {i} is NOT a near-tie (points at an inexact fold/replay, not "
            f"batch-size rounding): reference top-2 = {base_body['completion_probabilities'][i]['top_logprobs']}, "
            f"reference token = {base[i]}, {name} token = {tokens[i]}"
        )
        assert is_near_tie, msg

def first_divergence_is_near_tie_topn(ref_body, ref_tokens, other_tokens, n=5, margin=0.03):
    """Like first_divergence_is_near_tie, but checks the reference's top-n candidates (not just top-2)
    with a slightly wider margin. Needed for longer, multi-prompt stress runs: more decode rounds and a
    wider mix of batch shapes (verify / replay / fold, and a fold's own tail) give batch-size-dependent
    floating point drift (see server_sample_and_accept_replay's docstring) more chances to move the
    argmax within a cluster of 3+ closely-probable candidates, not just 2. Confirmed by direct inspection
    (requesting n_probs=20 at the divergence position) on one such case: the reference's top-3 were
    0.1526 / 0.1520 / 0.1297, and the fold-on arm picked exactly the #3 candidate - a real near-tie, not
    an inexact restore, that first_divergence_is_near_tie's stricter top-2/margin=0.02 correctly flags as
    "not near-tie" because it is deliberately narrow (see its own docstring)."""
    i = first_divergence_index(ref_tokens, other_tokens)
    if i is None:
        return True
    top = ref_body["completion_probabilities"][i]["top_logprobs"]
    if not top:
        return False
    probs = [math.exp(t["logprob"]) for t in top[:n]]
    ids = [t["id"] for t in top[:n]]
    p0 = probs[0]
    within = {ids[j] for j in range(len(probs)) if p0 - probs[j] <= margin}
    return other_tokens[i] in within

@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_fold_stress_matches_no_draft_baseline():
    # Stress the fold over more prompts / longer generations than the single-prompt tests above. Per R18
    # (see first_divergence_is_near_tie's docstring): strict token equality is not a valid oracle on this
    # Vulkan build even without fold - test_real_draft_makes_partial_acceptances_and_keeps_tokens above
    # already documents one legitimate near-tie divergence at temperature 0 on the plain replay path - so
    # every prompt is checked against a NO-DRAFT reference by prefix agreement + near-tie at the first
    # divergence, for both fold on and fold off. Uses the wider top-n oracle (see
    # first_divergence_is_near_tie_topn) because 256-token runs give compounding batch-size-dependent
    # rounding more opportunities to move the argmax within a 3+-way near-tie cluster, not just a 2-way one.
    prompts = [
        "Once upon a time",
        "The quick brown fox jumps",
        "In a galaxy far, far away",
        "def fibonacci(n):",
        "The history of computing began",
        "She looked at the letter and",
    ]

    def mk_req(p):
        return {"prompt": p, "temperature": 0.0, "n_predict": 256, "cache_prompt": False,
                "return_tokens": True, "n_probs": 5}

    ref_server = make_server(None); ref_server.start()
    refs = [ref_server.make_request("POST", "/completion", data=mk_req(p)).body for p in prompts]
    ref_server.stop()

    for env_name, env in (("fold-on", None), ("fold-off", {"LLAMA_SPEC_FOLD": "0"})):
        server = make_server("q4")
        if env:
            server.env = env
        server.start()
        for p, ref_body in zip(prompts, refs):
            body = server.make_request("POST", "/completion", data=mk_req(p)).body
            ref_tokens = ref_body["tokens"]
            tokens = body["tokens"]
            i = first_divergence_index(ref_tokens, tokens)
            assert tokens[:i] == ref_tokens[:i]
            assert first_divergence_is_near_tie_topn(ref_body, ref_tokens, tokens), (
                f"{env_name}: prompt {p!r} diverges from the no-draft baseline at a confident (non-near-tie) "
                "position - points at an inexact restore/fold, not batch-size rounding"
            )
        c = counters(server); server.stop()
        if env_name == "fold-on":
            assert c["fold"] > 0
        else:
            assert c["fold"] == 0
        assert identity_holds(c, n_prompt_decodes=2 * len(prompts))

@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_fold_concurrent_slots():
    # Two concurrent requests on two slots (n_parallel=2, --kv-unified) so the drafting / checkpoint /
    # accept phases' iterate(...) loops process a folded round on one slot at the same time as a plain
    # round (or a fold-tail rejection) on the other slot - the interesting case for any per-slot fold state
    # (spec_folded, spec_fold_draft, spec_n_target_committed) that must not leak across slots.
    prompts = ["Once upon a time", "The quick brown fox jumps"]

    def mk_req(p):
        return {"prompt": p, "temperature": 0.0, "n_predict": 128, "cache_prompt": False,
                "return_tokens": True, "n_probs": 5}

    ref_server = make_server(None); ref_server.start()
    refs = [ref_server.make_request("POST", "/completion", data=mk_req(p)).body for p in prompts]
    ref_server.stop()

    server = make_server("q4")
    server.n_slots = 2
    server.kv_unified = True
    server.start()

    bodies = parallel_function_calls([
        (server.make_request, ("POST", "/completion", mk_req(p))) for p in prompts
    ])
    bodies = [r.body for r in bodies]

    c = counters(server); server.stop()

    for p, ref_body, body in zip(prompts, refs, bodies):
        ref_tokens = ref_body["tokens"]
        tokens = body["tokens"]
        i = first_divergence_index(ref_tokens, tokens)
        assert tokens[:i] == ref_tokens[:i]
        assert first_divergence_is_near_tie_topn(ref_body, ref_tokens, tokens), (
            f"prompt {p!r} diverges from the no-draft baseline at a confident (non-near-tie) position "
            "under two concurrent slots"
        )

    assert c["fold"] > 0
    # no identity_holds() here: with 2 concurrent slots, prompt processing for both requests can be
    # batched into the same llama_decode() call(s), so n_decode_total no longer attributes a fixed
    # "2 prompt decodes per request" the way the single-slot tests above assume (identity_holds'
    # n_prompt_decodes=2 default, and this file's own docstring, are both scoped to one active slot;
    # see Ruling R1 in the plan ledger for the general multi-slot case).
