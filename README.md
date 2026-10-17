# llama.cpp for Strix Halo: Qwen3.8-Flash-Next at 1M context on Vulkan

This is my fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). I tuned it for one machine: an AMD Ryzen AI Max+ 395
("Strix Halo", Radeon 8060S iGPU, gfx1151) with 128 GB of unified memory, running on **Vulkan (RADV), not ROCm**. A second, smaller
box with a Ryzen 9 7940HS and a Radeon 780M (RDNA3, 96 GB) serves as the "hands" for parallel coding agents.

The model is **Qwen3.8-Flash-Next** (`qwen4exp`): a 177B-parameter MoE (125B weights + a 51B n-gram embedding table, ~6B active
per token, MTP head, vision). Stock llama.cpp runs it at ~24-27 t/s with a dense 262k window and no working speculative
decoding. This fork runs it with a **1 048 576-token context in one slot at 35-60 t/s decode**, in about 103 GiB of GPU memory.

Everything here was measured on my two boxes. Numbers, dates and image versions: [docs/s1/BENCHMARKS.md](docs/s1/BENCHMARKS.md).
Deploying it, upgrading and rolling back: [docs/s1/DEPLOY.md](docs/s1/DEPLOY.md). Flags and knobs: [docs/s1/KNOBS.md](docs/s1/KNOBS.md).

## What v61 adds (2026-10-17)

v61 is v60 plus three patches. Two are speed levers for the QSA indexer at depth, both on by default (no compose change):

- **0241, A2b: pooled-key reuse in the indexer score GEMM.** A 128-query prefill chunk is four query tiles. v60 streamed the
  whole pooled-key store once per query tile; v61 numbers the score GEMM's workgroups key-tile-major, so the four workgroups
  that read the same tile of keys are dispatched back to back and the later ones usually find it in cache (an observed
  effect; the bits do not depend on it). Each workgroup runs the same per-tile program as before, so every output bit is
  unchanged. Knob `GGML_VK_QSA_SCORE_KREUSE` (coopmat1 devices).
- **0243, A3: top-k select without key rows.** The select pass over the score plane no longer writes and re-reads a key row
  per (row, cell); it computes the key from the score bits and the cell visibility with integer instructions. This also removes
  the key region from the selection scratch (258 -> 32.25 MiB preallocated in a 524k test). Knob `GGML_VK_QSA_TOPK_NOKEYS`.
- 0242: a test fix (a private temp dir per run) and comment wording; no code change.

The package is EXACT, gated the same way as v60: on gfx1151 with the 1M production configuration minus the vision projector,
one slot, my S0.1 oracle corpus (text), v61 matched the v59 reference bit for bit (logits at 1 502 positions, 224 state
points, ordered selected ids, restore / replay histories); with A2b off, A2b forced onto its fallback (a 4-workgroup limit)
and A3 off, the short case set matched the same reference. 91 checks PASS, 4 INFO, 0 FAIL (13 h 35 min wall, 13 h 21 min of sections). The state format
and the disk-cache key are v60's: an install on v60 switches by changing the image (the v59 adopt line, if you have it, stays
as it is).
Release notes: [docs/s1/RELEASE-NOTES.md](docs/s1/RELEASE-NOTES.md).

1M brain profile, i1-Q4_K_M + MTP draft, KV q8_0, one slot, v60 -> v61:

| metric | v60 | v61 | change | method |
|---|---|---|---|---|
| prefill of the 65 536-token step ending at 262k / 524k / 1M | 474.3 / 413.2 / 328.9 t/s | **491.4 / 433.9 / 357.1 t/s** | +3.6 / +5.0 / +8.6 % | oracle probe, continuous cold fill, draft on |
| cold fill 0 -> 1M | 44.0 min | **41.8 min** | -5.0 % time | oracle probe, sum of 18 prefill steps |
| cold prompt 8k / 32k / 64k | 548.8 / 458.6 / 357.8 t/s | 550.0 / 457.5 / 357.9 t/s | +0.2 / -0.2 / +0.0 % | oracle probe; dense range, not touched by the levers |
| step on a restored prefix, -> 262k / -> 524k / -> 1M | 472.8 / 413.1 / 326.0 t/s | 488.4 / 432.3 / 355.6 t/s | +3.3 / +4.6 / +9.1 % | oracle probe |
| GTT peak, oracle probe (no projector) | 106 357 MiB | 106 095 MiB | -262 MiB | the gate's score-capture pass; admission bar 109 216 |
| GTT peak, oracle probe fill pass | 104 900 MiB | 104 544 MiB | -356 MiB | the gate's fill pass |
| GTT, llama-server with the vision projector, after load | 105 414 MiB | 105 495 MiB (105 519 peak after the smoke requests) | +81 MiB | deploy smokes, 2026-10-16 / 2026-10-17 |
| decode at depth (draft on, worst window) / TTFT at 1M + 256 new tokens / server GTT over a 1M fill | not measured on v60; v59: 26.07 / 22.10 / 24.38 ms/tok, 2 452 ms, 105 196 MiB | 25.83 / 21.04 / 19.92 ms/tok at 32k / 262k / 1M; 1 886 ms median (1 592 warm); 105 825 MiB at load, 105 627 over the fill | vs v59: -0.9 / -4.8 / -18.3 %; -23 %; +434 MiB | llama-server S0.2 rows, corpus tokens-s02, 2026-10-15 (v59) and 2026-10-18 (v61), BENCHMARKS section 2a |

Kernel level, s1: the A2b score GEMM per call at the deployed chunk shape, 16 shapes (pool 16 384-262 144 blocks, key forms
f32 rows and f16 mirror), median time ratio on/off 0.877 (0.673-1.097) in the gate's run; at pools of 98 304 blocks and more
-30 to -32 % on f32 rows and -12 to -23 % on the f16 mirror (the production form). The
A3 select of one QSA layer for the 64k step ending at 262k / 524k / 1M: 1.834 -> 1.219 / 3.757 -> 2.993 / 7.616 -> 6.097 s.
The probe rows are single runs, the v60 and v61 passes a day apart; the dense rows (8k-64k) moved within 0.4 %.
Server-side rows (llama-server, the S0.2 corpus, the same driver and positions) exist for v59 and v61, not for v60: cold prompt
128k 377.9 -> 427.4 t/s, fill 0 -> 1M 1.07 -> 0.70 h, decode with the draft 26.07 / 22.10 / 24.38 -> 25.83 / 21.04 / 19.92 ms/tok
at 32k / 262k / 1M with the same accepted-draft counts, TTFT at 1M 2 452 -> 1 886 ms; docs/s1/BENCHMARKS.md section 2a.

## What v60 adds (2026-10-16)

v60 is v59 plus fourteen patches (0227-0240) aimed at prefill on this GPU, mostly at depth: fused hyper-connection tails and a
GDN conv input without the concat pass, MoE expert GEMMs with a one-time f16 cast of the activations and a larger tile, a
16 x 16 tile for the m = 4 hyper-connection inject, the QSA indexer score as one op with an f16 mirror of its pooled keys, a
mat-vec that unpacks the weights once per row for the speculative verify widths, and a device-resident KQ visibility mirror.
The package is EXACT against v59, as a whole: gated on gfx1151 with the 1M production configuration minus the vision
projector, one slot, on my S0.1 oracle corpus (text), it gave bit-identical logits, saved states and ordered selected ids,
including restore / replay histories, at verify widths 1-6 in full and the last 6 positions of widths 7-9; this is no claim
for other model families, and not a claim of equality with upstream. The B2 tile (0233) is built but off by default and was
not part of the gated configuration. One defect is
fixed (0239: a restored conversation with images lost the image tokens' positions in the indexer cache); because of it the
saved-state format changed, so v60 has a new disk-cache key and adopts the v59 cache read-only (DEPLOY.md, section 4).
Release notes: [docs/s1/RELEASE-NOTES.md](docs/s1/RELEASE-NOTES.md).

1M brain profile, i1-Q4_K_M + MTP draft, KV q8_0, one slot, v59 -> v60:

| metric | v59 | v60 | change | method |
|---|---|---|---|---|
| prefill of the 65 536-token step ending at 262k / 524k / 1M | 368.0 / 284.1 / 194.9 t/s | **474.3 / 413.2 / 328.9 t/s** | +28.9 / +45.4 / +68.7 % | oracle probe, continuous cold fill, draft on |
| cold fill 0 -> 1M | 64.3 min | **44.0 min** | -31.6 % time | oracle probe, sum of 18 prefill steps |
| cold prompt 8k / 32k / 64k | 503.4 / 423.0 / 335.4 t/s | 548.8 / 458.6 / 357.8 t/s | +9.0 / +8.4 / +6.7 % | oracle probe, cold prompts in one process (8k = its first prompt) |
| GTT, llama-server with the vision projector | 104 591 MiB (v59 live, just before the switch) | 105 414 MiB after load, 106 032 after the first requests | | deploy smoke, 2026-10-16 |
| GTT peak, oracle probe (no projector) | 105 538 MiB | 106 357 MiB | +819 MiB | the gate's score-capture pass; admission bar 109 216 |
| decode at depth | 26.1 / 22.1 / 24.4 ms/tok at 32k / 262k / 1M | not measured on v60 | v61: 25.83 / 21.04 / 19.92 ms/tok (BENCHMARKS section 2a) | llama-server, my corpus, draft on, worst of 3 windows of 128 tokens |
| TTFT, 256 new tokens on a 1M history | 2.45 s (median of 3; 2.05 s warm) | not measured on v60 | v61: 1.89 s median (1.59 s warm) | llama-server |

The probe is a libllama program that runs the compose's configuration (without the projector) through a fixed corpus; on v59
it matched llama-server within 1.2 % per step. Single runs, the v59 and v60 passes about a day apart.

## What it gives

Same box, one conversation, one slot for the fork's current column (the fork columns use the i1-Q4_K_M weights). The last
column is the v53 suite of 2026-10-09/10 (v54-v59 have the same numerics); v60's and v61's prefill gains are in the tables above:

| | stock llama.cpp (Aug 2026) | this fork, dense (Sep 9) | this fork, v53-v59 |
|---|---|---|---|
| decode, empty context | 24-27 t/s | 49.6 t/s (Sep 6) | **61 t/s** |
| decode at 107k / 213k | - | 41.8 / 29.4 t/s | **51.6 / 59.9 t/s** |
| decode at 524k / 1M | not possible | not possible | **45 / 35.5 t/s** |
| prefill 8k / 32k | 365 t/s at 6.5k | 386 / 329 t/s | **550 / 450 t/s** |
| prefill at 213k depth | - | 167 t/s | **421 t/s** |
| cold fill 0 -> 524k / 0 -> 1M | - | - | **25.6 min / 1.09 h** |
| GPU memory (GTT) | - | - | **92.3 GiB at 524k, 102.7 GiB at 1M**, flat at every depth |

The stock row used IQ4_XS / UD-Q4_K_XL of the base model, so it is a rough reference. Decode at depth depends on how predictable
the text is (draft acceptance), which is why 213k can read faster than 107k.

### Stock llama.cpp b10689 vs this fork (v60), same GGUF, gfx1151, n_ctx 262 144 dense

Measured 2026-10-16 on s1: the official `server-vulkan-b10689` image against v60, same i1-Q4_K_M GGUF, one slot, KV q8_0,
YaRN x4, dense attention on both sides. This is the 262k window stock can run, not the fork's 1M profile.

| measurement | stock b10689 | v60 | v60 vs stock |
|---|---:|---:|---:|
| cold prompt 8k / 32k / 72k / 128k, t/s (128k: n = 1; both MTP off) | 353.8 / 312.6 / 247.8 / 192.0 | 544.3 / 455.8 / 338.2 / 245.9 | +53.9 / +45.8 / +36.5 / +28.1 % |
| prefill steps 65 536 -> 261 120, aggregate, t/s | 107.3 | 124.1 | +15.7 % |
| last step 207 110 -> 261 120, t/s | 81.4 | 91.4 | +12.2 % |
| decode at 26k / 131k / 261k, MTP off, ms/token (restore path both) | 37.82 / 56.04 / 77.81 | 31.76 / 41.05 / 52.18 | +19.1 / +36.5 / +49.1 % t/s |
| decode at 26k / 131k / 261k, fork MTP on (continuous), ms/token | no stock MTP | 20.00 / 24.69 / 40.88 (acceptance 0.782 / 0.949 / 0.852) | +89.1 / +127.0 / +90.3 % vs stock MTP off |
| GTT at 262k, MTP off | 88 137 MiB | 85 420 MiB | -3.1 % |
| TTFT, continuation at 261.6k + 256 new tokens, median of 3 | 5 359 ms | 3 659 ms (MTP on) | x1.46 |

GTT point: stock at the end of its continuous fill, the fork in the MTP-off restore band at 261 120 (the fork reserves the
full KV at load).

- **No stock MTP number.** b10689 failed to load the sidecar draft GGUF (`failed to load draft model`); upstream `qwen4exp` has no MTP graph.
- **Greedy-64 identity, stock vs fork, both MTP off: 62/64 tokens equal (against stock rep 0), first divergence at token 48.** Reported, not graded:
  stock lacks the GDN normalization fix (upstream PR #28068) and the fork's exactness patches.
- **The stock server was not run-to-run deterministic in this run:** greedy-64 rep 0 != rep 1 on the same 2 000-token prompt
  within one process; the fork arms were deterministic (fk0 with the strict top-5 check, fkm tokens only). Observed once (2 reps); reported, not diagnosed.

Caveats (524k/1M and QSA are fork-only, MTP needs the sidecar draft, the fork's base is b10664 + PR #27742 + the GDN fix rather
than b10689, YaRN x4 in both arms, decode bands by the restore path) and the method: [docs/s1/BENCHMARKS.md](docs/s1/BENCHMARKS.md),
section 9.

What does that:

- **QSA sparse attention for 524k-1M.** The model's 12 full-attention layers ship an indexer that the GGUFs leave switched off.
  The fork runs it: each query attends to 2 051 selected cells instead of the whole context, with a persistent pooled-key
  cache, an incremental block census, a fused top-k selection kernel and indexed flash attention for prefill. Dense below 64k,
  so short prompts are unchanged.
- **MTP speculative decoding that pays off on Vulkan.** On the early MTP builds the draft was a 5x loss on this GPU. Here it is
  roughly x1.5 decode (23.4 vs 35.1 ms/token at 81k), and the draft itself runs sparse at depth (-18 % ms/token at 524k).
- **Graph + allocation-plan cache.** Decode reuses built graphs and their memory plans on an exact match (-5.8 % ms/token).
- **Flat memory.** The KQ mask is computed from cell metadata instead of a [context x batch] plane (at 1M that plane alone would
  be 4 GiB), and every compute buffer is reserved at load. GTT does not grow during a 1M fill; an admission check makes a
  prefill chunk wait rather than run the box out of memory.
- **Disk prompt cache with recurrent checkpoints.** A conversation evicted from the slot is written to NVMe (~15 KB/token) and
  loaded back instead of re-prefilled; checkpoints are kept along the conversation, so a branch or a copy rewinds to the nearest
  one. Measured: 72.5k-token branch resumed in 0.6 s (re-prefill: 209 s), identical answer.
- **webui for long agent sessions.** Context-reserve auto-compaction (summarise and continue before the window is full, continue a
  reply that hit the cap from its exact stop point, a guard against unfinished replies), per-chat context settings, a frozen tool
  list per conversation (tool changes do not invalidate the cache). The one context element is upstream's ring by the send
  button: amber and a "compact?" link from 80 % full (setting), red below the pre-turn minimum, pulsing while it compacts.
  Hover opens its card, click or Enter pins it: upstream's usage details plus last-request prefill / decode t/s, MTP % with
  accepted / drafted, cache share, the chat's compute time, the reserve with its breakdown, the compaction state with Cancel,
  links to the previous / next compacted chat, "Compact now" and the settings; `/compact` in the composer. Timing chips
  collapse to one muted line per reply (details on hover; "Show timing chips" hides them). A continuation drops the prefill
  the server echoes back (whitespace-normalised match; a diverging echo is joined verbatim and the message marked
  `echoMismatch`). Reasoning per chat: "+" menu -> Reasoning -> Default / Off / Low / Medium / High / Max. When the template
  supports it (Qwen3.8-Flash-Next), Low / Medium / High / Max also send the native `reasoning_effort` low / medium / xhigh /
  xhigh. An explicit level gets the model card's thinking sampling, Off the instruct sampling, but only for parameters you
  have not set yourself; Default sends neither and follows the server. Both servers here run `--reasoning off`, so Default
  means no thinking. Settings -> Sampling & Penalties -> "Presets by reasoning state" makes both presets editable (Thinking
  and Instruct, model-card defaults, "Reset to model card") with optional per-level overrides for Low / Medium / High / Max;
  precedence: your own Sampling values > level override > preset. The presets follow Save / Cancel and are included in
  settings export / import.
- **Backend sampling.** `--backend-sampling` (or the webui checkbox) works with draft-mtp on this fork, but it bypasses the
  graph cache and the CPU fast path (slower decode) and is not bit-exact at temperature > 0: keep it off for speed. Clients can
  read `/props` -> `backend_sampling_supported` and `speculative {type, n_max}`; where it cannot work, a request asking for it
  gets a clear 400 before any slot is touched.
- **Telemetry.** The stop type of every reply at INFO (`stop=eos|limit|word|cancel`), per-round speculative timings, `kqvis-prof`
  lines, expert-occupancy stats, `disk_cache_*` in `/metrics`, `GET /disk-cache`.

The full list (240 numbered changes, including upstream back-ports): [docs/s1/PATCHES.md](docs/s1/PATCHES.md).

## What is hardware-specific and what is not

- **Vulkan / RDNA3 and RDNA3.5:** the new kernels (QSA top-k radix select, indexed and metadata flash attention, KQ mask, GDN and
  hyper-connection fusions), the MoE `mul_mat_id` changes (row-id hoisting up to 512 experts, compact grid, tile choice), flash
  attention coopmat1 tiles with RDNA3 signed-zero handling, UMA readback. They should help any RADV GPU with cooperative
  matrices on a large MoE; I measured them on gfx1151 and gfx1103 only.
- **Generic (libllama / server / webui, any backend):** the QSA model logic and memory, graph and plan cache, disk prompt cache,
  GTT admission (reads an AMD sysfs file; the path is a knob), the webui, the speculative-decoding fixes, the logs.
- **Model-specific:** QSA and the MTP work apply to `qwen4exp` only. Other models run upstream's model code with the fork's
  generic kernels.
- **Not tested:** ROCm/HIP and NVIDIA. The new ops have CPU and Vulkan implementations only.

### Other models

Which v60-era levers can help another model:

| lever | patches / knob | scope |
|---|---|---|
| B1: MoE expert GEMMs, f16 pre-cast of B + BM 128 x BN 64 tile | 0231, 0232 | generic: any `mul_mat_id` on AMD coopmat1 with q4_K / q5_0 / q8_0 / q6_K / iq3_s experts |
| B4: GDN conv input without the concat pass | 0228 | generic fusion (CONCAT + SSM_CONV); fires where a model's graph has that shape. On Qwen3.6-35B-A3B it does not fire (probe, 2026-10-17) |
| B4c: 16 x 16 tile for m <= 16 matmuls | 0234 | generic: dense matmul with m <= 16, q4_K / q8_0 weights, AMD coopmat1 |
| C1: mat-vec unpack hoist | 0236 | generic: mat-vec at 2-8 columns (speculative verify widths), MMVQ q4_0 / q5_0 / q8_0 / q4_K and f32 x f32 |
| graph + allocation-plan cache | 0192-0196, `LLAMA_GRAPH_CACHE` | generic (libllama) |
| disk prompt cache | `--cache-disk` | generic (server) |
| B3: hc_mix tail fusion | 0227 | a generic graph-pattern fusion (multi_add + SCALE, SCALE + SILU); the sites I know are Qwen3.8-Flash-Next's hyper-connections and QSA pooling. On Qwen3.6-35B-A3B it does not fire (probe, 2026-10-17) |
| QSA line: A1 score epilogue, A2a f16 mirror, A2b score key reuse, A3 select without key rows, the indexer cache and 0239 | 0118-0161, 0235, 0239, 0240, 0241, 0243 | Qwen3.8-Flash-Next only |

One A/B so far on another family: Qwen3.6-35B-A3B UD-Q4_K_XL on the 780M (KV f16, 262k context, one slot), upstream
b10689 vs this fork at the v60 revision with `GGML_VK_DISABLE_MMQ_CM1=1 GGML_VK_FREE_WAIT_IDLE=1 GGML_ALLOC_GROW_SLACK=10
LLAMA_RESERVE_RUNTIME_SHAPE=1 LLAMA_GRAPH_CACHE=8` (the int8 MMQ switch keeps both arms on the same matmul path), run 2026-10-17.
MTP off: cold prefill 8k / 32k / 72k 304.1 / 212.9 / 130.7 -> 310.7 / 225.1 / 152.6 t/s (+2.2 / +5.7 / +16.8 %); decode at depth
4k / 32k / 128k 49.77 / 56.03 / 85.28 -> 46.22 / 54.62 / 84.38 ms/token (one request per depth, no noise band). The fork's base
is b10664 plus drluoto's branch, so "fork vs upstream" also contains about 25 upstream commits. Per-lever split done (the same
fork with the levers off): +3.7 / +1.5 / +1.1 % prefill at 8k / 32k / 72k, attributed mainly to B1; decode below 1 % (n = 1,
not measurable); B3/B4 do not fire; MTP on +8.9 / +3.3 % at 4k / 32k and -7.8 % at 128k with lower draft acceptance (0.890 vs
0.924) (n = 1); greedy tokens identical 64/64 in every pair. No EXACT claim for this
family (other numerics). Details: [docs/s1/BENCHMARKS.md](docs/s1/BENCHMARKS.md), section 8.

## Hardware and host setup

s1: Ryzen AI Max+ 395, 128 GB LPDDR5x, VRAM carve-out at the minimum, kernel 7.0, Ubuntu. Kernel parameters
`ttm.pages_limit=28835840 ttm.page_pool_size=28835840` (110 GiB GTT), `amd_iommu=off` (+13 % decode, +25 % prefill),
`amdgpu.lockup_timeout=600000`, dirty-page caps at 2 GiB, 64 GiB swap, a hardware watchdog, GPU clock pinned high.
balkon: Ryzen 9 7940HS + 780M, 96 GB DDR5, 78 GiB GTT, same idea scaled down. Details and the exact lines:
[docs/s1/HOST-TUNING.md](docs/s1/HOST-TUNING.md). A 1M cold fill keeps the GPU busy for an hour; check your cooling.

## Running it

Two commands per box (details, disk needs and the manual route: [deploy/README.md](deploy/README.md)):

```sh
deploy/s1/setup-brain.sh           # Strix Halo: 1M brain   (BRAIN_CTX=524k for the 524k profile)
deploy/balkon/setup-hands.sh       # 780M: 5-slot hands
```

Each script checks docker, the Hugging Face CLI, disk space, the GTT limit and the Vulkan device; downloads the weights and
checks their sha256; (hands only) re-encodes the BF16 hyper-connection weights and the LM head to Q8_0 with
`tools/s1/gguf_hc_q8.py`; pulls the image (or builds it with `BUILD=1`); starts the compose file and smoke-tests it.
`--dry-run` shows every step first. Which weights and why: [docs/s1/WEIGHTS.md](docs/s1/WEIGHTS.md). I redistribute no weights;
they come from their authors' Hugging Face repos.
The published image `vahpetr/llama-cpp-s1-fork:v61` = `:v61k` = `:latest` (one digest,
`sha256:46fc98c6f86902b220309e172691758e9806eaf2fc411a640d4ac2f5595b7c63`) is built from my `prod-v11-stage8-pkg8` revision
(v61 = `7a438f3ba`; `:v60` / `:v60k` stay as the rollback, v60 = `845c6fbcd`, digest
`sha256:794113766af1129c65e9541bcc4b9b3bac4d378fa38b38cd4d7c5f00fbf7a0d1`); this `s1-public` branch adds docs, deploy files,
`tools/s1` and comment scrubs on top.
Upgrading from v60: same disk-cache key, change the image only. Upgrading an install with stored conversations from v59: the
disk-cache key changes, see [docs/s1/DEPLOY.md](docs/s1/DEPLOY.md), section 4 (one `--cache-disk-adopt` line).
My production draft head is a vocabulary-pruned variant built from my own traffic, so the public setup uses dzannotti's
full-vocabulary head, and the decode numbers above are an upper bound for it.

First start on s1: GTT after load ~105 500 MiB (1M, v61; v60 105 414, ~106 000 MiB after the first requests on v60, bar
109 216) or ~94 500 MiB (524k, v59; v60 adds the f16 mirror, not measured there); the first long prompt fills at ~550 t/s
falling to ~357 t/s near 1M (v61; v60 ~330 t/s).

## Known limitations

- **No context shift.** The model has recurrent (Gated-DeltaNet) layers. The client must compact before the window is full;
  the bundled webui does. Replies near the end of the window are capped by the client to the room that is left.
- **Accuracy at 1M is not gated.** One long-context probe scored 76.8 % at 1M (vs ~77 % at 524k). Use the 1M window for recall
  over long inputs and check answers that depend on it.
- **Two slots give no extra throughput** on these boxes (-15 % total, measured). A second slot is for concurrency only.
- **After a checkpoint rewind** (disk cache, branch, copy) the continuation can differ from a fresh prefill at near-tie tokens:
  the same math in a different batch composition. The state itself is verified, the text is not guaranteed byte-identical.
- **Pre-v60 disk-cache entries with images are not loaded by v60 / v61** (they may carry the defect 0239 fixes); text entries
  are. A rollback to v59 does not see conversations saved by v60 or v61 (v60 and v61 share one key).
- One model load per GPU at a time; a build or a second load next to a loaded 1M model has run my box out of memory.
- Single-user setups. No multi-tenant hardening beyond upstream's.

## For AI agents

If you are an agent working on or with this code, start with [AGENTS.md](AGENTS.md) (repo map, tags, test contract, exactness
classes, how to measure) and [docs/s1/KNOBS.md](docs/s1/KNOBS.md) (every flag and env knob with default, effect, class and the
value I run). The webui's compaction and timing features exist so that agents can run long sessions on this box without
re-prefilling and without guessing where the time went.

## How I work

I run this box with my own agent harness: a controller model plans and reviews, worker agents ("hands") edit code and run tests,
and every change goes through a gate on the real model before it ships. That harness is private; this repository is the runtime
it runs on.

## Upstream and license

Based on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) via drluoto's `strix-halo-flash-next` branch (b10664) and
upstream PR #27742 (Qwen3.8-Flash-Next support). Upstream's documentation is in `docs/`; its README is at
<https://github.com/ggml-org/llama.cpp#readme>. One change is open upstream as PR #28118; several others are PR candidates, not submitted yet.

MIT License, Copyright (c) 2023-2026 The ggml authors, see [LICENSE](LICENSE). The fork's changes are released under the same
license. Third-party components keep their licenses (`licenses/`, and the acknowledgements in upstream's README: cpp-httplib,
stb, nlohmann/json, miniaudio, subprocess.h).
