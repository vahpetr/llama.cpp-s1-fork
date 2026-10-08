# Multi-slot stage 6 — notes for item 4 (per-stream KV) [TAG_MULTISLOT]

Source: Codex gpt-6-astra review r1 of items 1–3 (piagent `docs/verification/2026-10-09-codex-astra-stage6-r1.md`),
spec piagent `docs/superpowers/specs/2026-10-09-multislot-pregate-spec.md`. State at this note: items 1–3 + round 2
(`LLAMA_QSA_HISTORY_DENSE`, tests) on `prod-v11-stage6`.

## Design points item 4 must honour under per-stream KV (`kv_unified = false`, `n_stream = n_seq_max`)

1. **Routing is per ubatch, not per stream.** `qwen4exp_qsa_depth()` takes one maximum over the ubatch rows and
   `qsa_exhaustive` picks ONE topology for every stream in it; any partial-history sequence routes the whole ubatch
   dense (`qsa_history_complete`), B's rows included (test-mtp-qsa-restore "simul"). Per-stream score tensors isolate
   score ownership, not the routing decision. → Homogeneous below/above-N groups stay required in BOTH target
   execution and draft generation; isolate partial histories and threshold-crossing runs in their own ubatches.
2. **`n_kv` = the maximum padded high-water over the participating streams.** An inactive stream does not inflate a
   batch's `n_kv`, but an active multi-stream batch shares one width/window: unequal depths still share dimensions.
   `oor` handling stays necessary, and partial-history detection must NOT fall back to `depth > n_kv`.
3. **Sort compatible runs by stream (sequence) id.** Sequential `split_equal` needs consecutive increasing ids and
   emits equal-length stream-major slices; gaps (A=0, B=2) legitimately split the batch (less batching, not an error).
4. **`LLAMA_MTP_QSA_MAX_UBATCH` counts total tokens of the ubatch.** Combined streams must not cross it when each solo
   run would stay sparse; classify ALL effective routing triggers (MIN_KV N, the exhaustive bypass `width == n_kv`,
   MAX_UBATCH, partial history), not only N. Conservative singleton exceptions are acceptable at first.
5. **Server fixes from the spec remain:** row-index remapping, view-boundary clamping, participating-slot filtering,
   and the temporary `global_index - off` sampling indices — post-decode currently checks every slot's verification
   indices and hands global indices to the speculative samplers (`server-context.cpp` ~5912, ~6033).
6. Keep: one preparation pass, whole verification runs, fair group execution, once-per-round accounting; build/reuse
   assertions on the actually executed ubatches.

## Minimal test list (np2 / np4, WITHOUT unified KV, N = 64)

- depths 63 / 64 / 65 around N; simultaneous same-class streams; mixed classes (must not share a ubatch);
  unequal lengths; non-consecutive ids; partial A + complete B (exists: test-mtp-qsa-restore "simul");
  threshold-crossing verification; cancellation; rejection / replay; capacity retry.
- Assert positive `n_stream > 1` coverage, the expected per-sequence routes, no mixed-class ubatch.
- Compare against matched solo histories with fixed schedules K = 0 / K = 8 / CHECK = 3; byte equality only against a
  reference with the same row count per decode (CPU mul_mat rounds a 2-row decode differently from a 1-row one,
  ~2e-8) — vs solo: proposals equal + tolerance.
- Explicit unchanged-shape transitions complete → partial and partial → complete (exists for np1:
  test-mtp-qsa-restore "transitions", `llama_perf_context().n_reused` deltas vs a `LLAMA_QSA_HISTORY_DENSE=0` child).

## Real-model gate before item 4 lands (items 1–3)

Positive multi-sequence contiguity coverage; startup / request refusal with an unchanged active neighbour; partial
fallback for restore / hole / rewind; normalized selection / state comparisons; graph-cache transition coverage;
attention-free catch-up; actual rejection / replay; unset-knob hands compatibility. Mixed-class exactness is not a
PASS requirement yet. Measure per-stream startup / restore peaks against 102400 MiB; incomplete coverage = PARTIAL.

## Knobs (round 2)

- `LLAMA_QSA_HISTORY_DENSE` = 1 on / 0 off (a25b663c1 routing) / unset: on until the process decodes an M-RoPE
  embedding batch (mtmd image), then off, sticky, logged once. Start log per context:
  `qsa_history_dense = on|off (reason)`. A hands process with mmproj therefore drops the fallback after its first
  image; a general multimodal-safe fix needs explicit partial-history provenance instead of positional contiguity.
- `LLAMA_QSA_HOMOGENEOUS=1` keeps the coverage term on regardless (multi-slot qualification mode).
