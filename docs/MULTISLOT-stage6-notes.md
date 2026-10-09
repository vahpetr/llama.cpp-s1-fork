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

## Item 4 + 5a status (implemented on `prod-v11-stage6`)

- 4b model guard: `qwen4exp_qsa_routing_check()` inside `qwen4exp_qsa_dense()` (build + reuse), strict mode only;
  per-sequence depth = max pos + 1 of its rows, `GGML_ABORT` on a below + at/above-N ubatch; proof
  `[qsa-routing] checks=… multi_seq=… below=… above=… mixed=0` (every 1024 checks + exit).
- 4a server: `qsa_route_partition()` after `pre_decode()` (one preparation pass), before `render()`; planner
  `server_qsa_route_groups()` / `server_qsa_route_class()` (server-context.h, test-driven); views clamped per range,
  capacity retries cut at whole-slot runs (a verification run that alone exceeds n_batch throws); post_decode skips
  slots without verify rows in the view, a split run throws, samplers get `global - off`; drafts per group
  (`qsa_route_draft_groups()`). Log: `[qsa-route] round=… ranges=… off=… n=… slots=[id@pos…] decode=ok|retry plan: {…}`.
- Partial history server-side = `prompt.draft_partial()` or rows not continuing `seq_pos_max + 1`.
  `llama_memory_seq_pos_min` is NOT usable: on the hybrid memory it is the recurrent part's latest position.
  A partial indexer history inside complete target/draft metadata (rewind hole in the draft only) is not visible to
  the server; the model-side dense fallback still routes that ubatch dense (one class, no guard violation).
- MAX_UBATCH cap also on target ranges (as specified): runs of <= 8 rows group only up to 8 rows total, so two
  4-row verify runs share a call but three do not. With `LLAMA_MTP_CATCHUP_ATTN=0` (prod) the catch-up has no
  attention; the cap then only costs calls - candidate relaxation: cap the catch-up / draft only.
- 5a: admission shape log (N, pool C, kv_unified, n_batch/n_ubatch, KV types, rollback depth, reserve, slope/once),
  `LLAMA_GTT_CONCURRENT_RESERVE_MIB` (default 0) added while another slot is processing, also on zero-growth chunks;
  /metrics `admission_n_slots`, `admission_reserve_mib`, `admission_last_reserve_mib`,
  `admission_reserve_deferred_total`. Restore / verify / checkpoint starts are not admission-gated yet.
- Tests: `test-mtp-catchup-homog-np2/-np4` (planner + range execution on the synthetic fixture, per-stream KV,
  N = 64) + negative child. Not covered: server e2e (fixture tokenizer `no_vocab`, the server's detokenization
  asserts on LLAMA_VOCAB_TYPE_NONE), the server's capacity-retry cut, an explicit unchanged-shape transition test.

## Round 3 status (Codex astra review r2, piagent `docs/verification/2026-10-09-codex-astra-stage6-r2.md`)

- **Major 1 — guard dedup (fixed).** `qwen4exp_qsa_routing_check()` validates EVERY invocation (no early return on
  a remembered ubatch). Only the proof counters stay "once per ubatch, not per QSA layer", deduplicated by a word-wise
  hash over (memory context, n_tokens, every row's pos / n_seq_id / seq ids); a collision can only under-count, never
  skip validation. Cost of the undeduplicated pass on the fixture (`LLAMA_QSA_ROUTING_COST=1` →
  `[qsa-routing-cost] calls=… rows=… ns=… ns_per_call=… ns_per_row=…`, a separate line; the `[qsa-routing]` proof line
  is unchanged): ~85 ns per call, ~20 ns per row (np2: 200 calls / 808 rows; np4: 319 / 1394). Negative child
  `--homog-neg-reuse`: one `llama_batch` decoded twice with the r2 key unchanged (first/last row seq 0/2 @ 80, n_tokens
  3, first seq 0), interior seq 1 rewound 80 → 61 by checkpoint → must abort. Note: on this fixture the r2 guard ALSO
  aborted (each ubatch's `pos` array is a fresh heap allocation, so the remembered pointer differed); the hole was
  allocator-dependent, the test is a requirement test, not a reproduction.
- **Major 2 — 5b operation-start gate (implemented).** `server_admission_op_check()` / `admission_op_gate()`: while
  another slot is processing, live GTT used + `LLAMA_GTT_CONCURRENT_RESERVE_MIB` must be <= the bar before (a) draft
  generation + its on-device speculative checkpoints (deferred = a plain decode this round), (b) a RAM prompt-cache
  swap-in, (c) a disk prompt-cache load (deferred = the task stays queued, nothing saved/loaded yet), (d) a `/slots`
  restore (re-queued). Never an abort, no timeout (the concurrency ends when the neighbour releases). /metrics
  `admission_reserve_deferred_{draft,restore_ram,restore_disk,restore_slot}_total`; WARN at most every 10 s per kind.
  Reserve 0 (default) / admission off: returns before reading anything. NOT gated: host prompt checkpoints (RSS
  envelope, spec §5), checkpoint restores into existing cells (rollback after rejection: correctness, no new cells),
  startup allocations. A waiting task is not "processing", so an active neighbour is never gated by a pending
  restore. The RAM gate is conservative: every LRU replacement with a prompt cache may swap in.
- **Minor — target row cap (relaxed).** `server_qsa_route_target_cap()`: target ranges are uncapped when the MTP
  catch-up is attention-free (`LLAMA_MTP_CATCHUP_ATTN=0`, prod); capped at `LLAMA_MTP_QSA_MAX_UBATCH` with the
  attention catch-up (unset / 1) or `LLAMA_QSA_HOMOGENEOUS_TARGET_CAP=1`. Draft groups always keep the cap. Startup
  log `[qsa-route] row cap: target ranges N, draft groups M`.
- **Major 3 — exactness standard (tests).** homog np2/np4 add an environment G running M's IDENTICAL schedule
  (history construction, ranges, draft groups, restore/replay, cancellation) with `LLAMA_GRAPH_CACHE=8
  LLAMA_GRAPH_CACHE_CHECK=3`; M is K = 0. Required byte equality: target logits per run, last draft logits per draft
  group, proposals, per-sequence carry; positive K=8 target reuse. Fixture: np2 18/18, np4 31/31 byte equal. Grouped
  vs solo is R81 composition: tolerance 1e-3 + equal proposals kept only there (different rows per decode → different
  CPU accumulation), first divergence printed (row, pos, vocab; ~1e-8).
- **Major 4 — provenance caveat (accepted, unchanged).** Server-side partial-history detection
  (`draft_partial()` or rows not continuing `seq_pos_max + 1`) is conservative and cannot prove the absence of older
  holes or a partial draft indexer history. The model's `qsa_history_complete()` (indexer minimum / count / maximum)
  is the source of truth: a partial history the server misses routes that ubatch dense on the model side (one class,
  no guard violation) — it costs one extra call / a dense neighbour, and a complete above-N neighbour sharing that
  ubatch runs dense too (not its solo sparse semantics), so exactness evidence for restore / adoption / hole cases
  must show isolation or equal coverage at the gate. The optional "singleton when the model reported partial last
  round" was not added (needs a model → server channel).
- **Gate contract.** Review r2 gate items 1–6 (shape/memory first at 102 400 MiB with a measured reserve; arms
  R/B2/O2/C2/S4; server execution coverage; routing/cache proof incl. K=0/K=8 identical-history byte comparison and
  CHECK=3 windows; admission/cache lifecycle incl. reserve defer/recovery; PASS rule — byte equality for identical
  histories and grouping, composition differences localised + teacher-forced replay, missing coverage = PARTIAL) are
  the multi-slot gate contract; see the r2 review file for the verbatim list.
- Still not covered here: server e2e (fixture `no_vocab`), the capacity-retry cut, real GTT calibration of the reserve.

## Per-seq state portability across the stream count (np1 <-> np2)

- Found by the multi-slot gate: an np1 slot / sequence state failed in an np2 per-stream server
  (`llama_state_seq_load_file: error loading sequence state file: n_stream mismatch`). Format rule now: per-seq state
  is stream-count agnostic — the writer is unchanged (`n_stream` + one cell block per writer stream; a per-seq save
  fills only its own stream), `llama_kv_cache::state_read_sinfo` accepts any writer `n_stream` for `seq_id >= 0`, puts
  the one non-empty block into the stream of `seq_id` (mirrored `sinfos_in/out` indexed by the process's streams, so
  the hybrid-idx indexer still adopts the attention layout) and rejects a second non-empty block. Whole-context
  (`seq_id == -1`) restores stay strict. The recurrent memory and the speculative carries (MTP "MPTH", eagle3
  boundary) have no stream / slot field; the seq-data envelope's source seq id is ignored on the host path.
- Disk prompt cache: the namespace key keeps n_parallel / n_seq_max / kv_unified (param digest, by design). Before,
  the state descriptor (key component + per-entry envelope) printed n_seq_max / unified / kv n_stream / n_seq_max /
  per-layer stream views / recurrent pool size, and the adopt structure used the TOTAL n_ctx — np2 could not adopt an
  np1 namespace. After: the descriptor prints the single-slot values (1) for those (prod np1 `--kv-unified`
  descriptor byte-identical, so its key does not move), and the structural fingerprint's ctx is the per-slot
  `llama_n_ctx_seq` (= `llama_n_ctx` for np1 and unified; changes only for np>1 without `--kv-unified`). So an np2
  per-stream (or unified) server with the same per-slot size adopts the np1 namespace via `--cache-disk-adopt`.
  One-time consequence: an np>1 server's descriptor (e.g. balkon hands, `--kv-unified --parallel N`) changes, so its
  own disk-cache namespace moves on the next deploy of this code (adopt the previous key to keep it).
- Tests: `test-state-portable` / `-qsa`.
