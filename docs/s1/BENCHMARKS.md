# Benchmarks

All numbers are measured on my two boxes, single runs unless a range is given. "vNN" is my image version (one image per fork
build; v53 = fork `prod-v11-stage6` @ f0fa8ddf1, v54 = v53 + webui and logging changes, v55 = v54 + webui
changes only, v56 = v55 + webui changes only, v57 = v56 + webui changes only, v58 = v57 + webui
changes only, v59 = v58 + a server backend-sampling guard and webui changes; same numerics, gated: G6 same-key,
bit-identical to v58; v60 = this tree's HEAD = v59 + patches 0227-0240, EXACT against v59, see section 0).
Run-to-run noise on the speed track is about ±1.5 %.

Brain model everywhere below: Qwen3.8-Flash-Next-Uncensored i1-Q4_K_M + MTP draft head (vocab-pruned, see deploy/README.md),
KV q8_0, one slot, YaRN x4, QSA above 64k, s1 (Ryzen AI Max+ 395, GTT 110 GiB, Vulkan/RADV). Sections 7 and 8 are the
780M box.

## 0. v60 vs v59 (2026-10-15/16, 1M profile, oracle probe)

Method: my oracle probe, a small libllama program that drives the 1M brain configuration (same model files, flags and env as
the compose, without the vision projector) through a fixed token corpus and logs the wall time of every operation. Fill rows:
one continuous cold fill in steps of 65 536 tokens with the MTP draft on; t/s = new tokens / wall of the step. Cold prompt rows:
cold (not cached) prompts in one fresh process; the 8k prompt is its first operation, 32k and 64k come later. v59 = rev `a1b6c324e` (plus the dump-only patch 0229, so both arms log the same records),
v60 = rev `845c6fbcd`; same corpus, same argv and env; single runs, the two passes ran about a day apart. Before trusting the
probe I checked it against llama-server on v59: the fill steps agree within 1.2 % and the whole fill within 0.2 % (section 2a).
The probe's GTT has no vision projector, so it is not the server's GTT.

| metric | v59 | v60 | change | v60 with the f16 mirror off (0240) |
|---|---|---|---|---|
| fill step 196 608 -> 262 144 | 368.0 t/s | **474.3 t/s** | +28.9 % | 460.7 |
| fill step 262 144 -> 327 680 | 340.2 t/s | 453.9 t/s | +33.4 % | 443.0 |
| fill step 458 752 -> 524 288 | 284.1 t/s | **413.2 t/s** | +45.4 % | 399.9 |
| fill step 983 040 -> 1 048 576 | 194.9 t/s | **328.9 t/s** | +68.7 % | 311.0 |
| cold fill 0 -> 262k / 0 -> 524k (sum of prefill walls) | 687.3 s / 1 534.6 s | 584.3 s / 1 190.2 s | -15.0 % / -22.4 % time | 591.2 s / 1 215.7 s |
| cold fill 0 -> 1 048 576 (sum of 18 prefill walls) | 64.30 min | **43.99 min** | -31.6 % time (x1.46) | 45.65 min |
| cold prompt 8k | 503.4 t/s | 548.8 t/s | +9.0 % | 535.3 |
| cold prompt 32k | 423.0 t/s | **458.6 t/s** | +8.4 % | 445.5 |
| cold prompt 64k | 335.4 t/s | 357.8 t/s | +6.7 % | 350.6 |
| step on a restored prefix, 196 608 / 458 752 / 983 040 + 65 536 | 368.2 / 283.6 / 195.0 t/s | 472.8 / 413.1 / 326.0 t/s | +28.4 / +45.7 / +67.2 % | 452.6 / 391.6 / 305.4 |
| GTT peak, fill pass (probe, no projector) | 104 078 MiB | 104 900 MiB | +822 MiB | 104 121 MiB |

The gain grows with depth: the dense part below 64k gains 5-9.5 %, every QSA step above it more, up to +68.7 % at 1M. The
f16 mirror alone is worth +2.9 % (->262k) to +5.8 % (->1M) in the back-to-back fill pair and costs 779 MiB.

Server rows, llama-server with the full compose (vision projector loaded):

| metric | v59 | v60 | source |
|---|---|---|---|
| GTT after load | 104 591 MiB | 105 414 MiB (+823) | the v60 deploy on 2026-10-16: v59 live just before the switch, v60 after load |
| GTT after the deploy smoke requests | - | 106 032 MiB | same deploy; the admission bar is 109 216 |
| GTT peak over a 1M fill | 105 196 MiB (section 2a) | pending (at least 106 032 MiB, the after-smoke value) | |
| GTT peak, oracle probe score-capture pass (no projector) | 105 538 MiB | 106 357 MiB (+819) | the gate's maximum; capture buffers, not a serving state |
| decode at depth (worst of 3 windows, draft on) | 26.07 / 22.10 / 24.38 ms/tok at 32k / 262k / 1M (section 2a) | pending | |
| TTFT, 256 new tokens on a 1M history | 2 452 ms median of 3 (2 054 ms warm, section 2a) | pending | |
| cold prompt 128k | 377.9 t/s (section 2a) | pending | |

Class: EXACT. On my S0.1 oracle (fixed prompts, greedy; every compared position's logits, the saved states and the QSA
selected ids in order, at verify widths 1-6 in full and the last 6 positions of widths 7-9, depths 8k -> 1M, and restore /
replay histories) v60 matched v59 bit for bit at all 1 502 compared logit positions and 224 state points. The gate ran on
gfx1151 with the production configuration minus the vision projector (one slot, a text corpus); it covers the package as a
whole, not each patch alone, and the B2 tile (0233, off by default) was not in it;
multi-slot stream offsets were not covered. On the live server after the switch, three fixed requests (a 32k prompt, a chat,
a needle at 26.7k) gave the same tokens and the same number of accepted drafts as on v59; on the 32k prompt the top-5
logprobs differed by exactly 0.

Observed and not explained yet: in the probe, state saves were about 1.8x slower on v60 than on v59 (21.4 s vs 11.6 s for the
15.6 GB state at 1M) while loads and the state size were the same. A llama-server slot-save timing on v60 is pending.

### Version lineage on the brain

| version | date | what changed | class vs the previous | disk-cache key |
|---|---|---|---|---|
| v52 | 2026-10-09 | graph + allocation-plan cache (decode 524k 23.6 -> 22.2 ms/tok) | EXACT | - |
| v53 | 2026-10-09 | multi-slot fixes (one-slot behaviour = v52) | one slot = v52 | - |
| v54 | 2026-10-10 | webui compaction, logs and counters | same numerics | same as v53 |
| v55-v58 | 2026-10-10/11 | webui only | same numerics | same |
| v59 | 2026-10-11 | server backend-sampling guard + webui | bit-identical to v58 | same |
| v60 | 2026-10-16 | patches 0227-0240: prefill kernels (B1, B3, B4, B4c), QSA score epilogue (A1) and f16 pooled-key mirror (A2a), mat-vec hoist (C1), KQ visibility mirror (C2), indexer cell-ext restore (0239) | EXACT (oracle above) | new: adopts the v59 namespace read-only (docs/s1/DEPLOY.md) |

## 1. Current build, 524k profile - suite runs 2026-10-09 21:50 and 2026-10-10 02:30 (image v53)

| metric | value |
|---|---|
| decode, cold (empty context), mean of code / C# / essay | 61.0 t/s (61.5 / 70.9 / 50.7) |
| decode at 27k depth | 57.1-58.2 t/s (draft acceptance 0.95) |
| decode at 107k depth | 51.6 t/s (acceptance 0.94) |
| decode at 213k depth | 59.9 t/s (acceptance 0.93) |
| decode at 524k depth | 22.2 ms/tok = 45 t/s (v52 gate, 2026-10-09) |
| prefill 8k (synthetic / real text: wiki en, wiki ru, C#, HTML) | 550 / 524-528 t/s |
| prefill 32k | 450 t/s |
| prefill of a 2k chunk at 107k / 213k depth | 407 / 421 t/s |
| cold fill 0 -> 524k | 25.6 min |
| 5 parallel short streams, total | 220 t/s |
| GTT peak | 94 493 MiB = 92.3 GiB |
| needle in a haystack 32k / 107k / 213k | 2/2, 2/2, 2/2 |

Decode at 213k reads faster than at 107k in these runs; the prompts differ in how predictable they are (acceptance), so compare
depths within one row of section 3 rather than across this table.

## 2. 1M profile - 2026-10-09 (image v52, same numerics as v53), cold fill in 16 steps of 65 536 tokens

| depth | prefill of the step, t/s | decode ms/tok | decode t/s | draft acceptance | GTT MiB |
|---|---|---|---|---|---|
| 65 536 | 333 | 21.4 | 46.8 | 0.77 | 105 188 |
| 131 072 | 435 | 19.5 | 51.2 | 0.81 | 105 193 |
| 262 144 | 361 | 21.2 | 47.1 | 0.82 | 105 193 |
| 524 288 | 278 | 22.4 | 44.7 | 0.82 | 105 193 |
| 786 432 | 227 | 22.1 | 45.3 | 0.85 | 105 193 |
| 917 504 | 208 | 25.1 | 39.8 | 0.82 | 105 193 |
| 1 048 320 | 192 | 28.2 | 35.5 | 0.78 | 105 193 |

Whole fill 0 -> 1M: 1.09 h. GTT is flat from the first step (102.7 GiB) because every compute buffer is reserved at load. No
admission waits. Host RSS 3.3-7.5 GiB. The same 1M arm on v47 (2026-10-06): 17.8 t/s at 1M, fill 1.76 h, GTT 106 043 MiB.

## 2a. S0.2 H0 rows (2026-10-15, v59, corpus tokens-s02)

Image v59, 1M brain profile without the disk cache, corpus tokens-s02 (sha256 e77e4bea67adebe8..., 1 098 504 tokens), single runs
(noise +-1.5 %). Decode rows are worst-of-3 windows of 128 tokens on a continuous history; they are a new baseline on this
corpus and not comparable 1:1 with the rows above (other text, other acceptance).

| row | value |
|---|---|
| cold prompt 128k (first request of the process) | 377.9 t/s (warm repeat 378.4) |
| prefill 72k | 346.4 / 348.3 t/s |
| fill step ->262k / ->524k / ->1M | 370.1 / 280.7 / 195.1 t/s (rows above: 361 / 281 / 192) |
| whole fill 0 -> 1 046 528 | 1.07 h (rows above: 1.09 h) |
| TTFT follow-up at 1M, 256 new tokens | 2 054 ms warm; 2 511 / 2 452 ms after a slot restore |
| slot save at 32k / 262k / 1M | 0.57 / 3.72 / 14.48 GiB (3.2 s at 1M) |
| GTT peak | 105 196 MiB |

Serial decode without the draft versus draft on:

| depth | no draft ms/tok | draft on ms/tok | draft acceptance | gain |
|---|---|---|---|---|
| 32 768 | 33.07 | 26.07 | 0.817 (223/273) | -21 % |
| 262 144 | 33.76 | 22.10 | 0.800 (264/330) | -35 % |
| 1 046 528 | 42.45 | 24.38 | 0.877 (306/349) | -43 % |

Per-round verify width with the draft on (width = sampled token + drafts):

| depth | share of rounds with width >= 3 | mean width |
|---|---|---|
| 32 768 | 41.0 % | 2.73 |
| 262 144 | 65.8 % | 3.77 |
| 1 046 528 | 97.4 % | 5.51 |

## 3. How we got here (same box, same weights)

| build | date | decode cold | prefill 8k | decode 107k | prefill 107k | decode 213k | prefill 213k | notes |
|---|---|---|---|---|---|---|---|---|
| stock llama.cpp PR #27742 image | 2026-08-27 | ~24 t/s | 365 t/s (6.5k) | - | - | - | - | IQ4_XS; 17.5 t/s decode at 26k; no MTP head in any GGUF |
| stock b10689 | 2026-08-30 | 27 t/s | - | - | - | - | - | MTP not supported by the stock converter/loader |
| v9 (first fork build with MTP) | 2026-09-06 | 49.6 | 386 | - | - | - | - | 5 slots, dense attention |
| v31-det (dense attention) | 2026-09-09 | - | - | 41.8 | 271 | 29.4 | 167 | depth track, 5 slots |
| v53 (QSA + sparse draft + graph cache) | 2026-10-09 | 61.0 | 550 | 51.6 | 407 | 59.9 | 421 | 1 slot |

The stock rows used a different quant (IQ4_XS / UD-Q4_K_XL of the base model), so they are a rough reference, not a paired A/B.
The paired A/B on the same GGUF (stock b10689 vs v60, n_ctx 262 144 dense) is section 9.
At depth the gain comes from QSA (sparse attention over 2 051 selected cells per query instead of the whole context); at 1M it is
the difference between possible and not.

## 4. What single levers were worth (paired gates on s1)

| lever | class | effect | when |
|---|---|---|---|
| MTP speculative decoding (draft n-max 5, p-min 0.7) vs no draft | output-preserving | 23.4 vs 35.1 ms/tok at 81k, 27.4 vs 39.9 at 524k (profiler on) | 2026-10-09 |
| sparse MTP draft (`LLAMA_MTP_QSA=1`) | target exact | decode at 524k -18.1 % ms/tok | 2026-10-08 |
| attention-free draft catch-up (`LLAMA_MTP_CATCHUP_ATTN=0`) | exact | 524k fill -29.6 % | 2026-10-08 |
| graph + plan cache (`LLAMA_GRAPH_CACHE=8`) | exact | decode at 524k 23.6 -> 22.2 ms/tok (-5.8 %) | 2026-10-09 |
| incremental QSA census (`LLAMA_QSA_CENSUS_INC=1`) | exact | 524k ms/tok -7.6 %; plan step 5.6 ms -> 38 us | 2026-10-04 |
| QSA top-k v2 selection | exact | decode +22.8 % at 524k | 2026-10-03 |
| pooled-key remap without the copy | exact | -8.3 % ms/tok | 2026-10-03 |
| indexed prefill (`LLAMA_QSA_PREFILL=indexed`) | composition | fill x1.88, GTT -2.9 GiB; accuracy non-inferior, 1 of 339 needles lost in its gate | 2026-10-03 |
| `amd_iommu=off` (host) | - | +13 % decode, +25 % prefill | 2026-08-27 |
| GPU DPM pinned high (host) | - | +3 % prefill | 2026-09-07 |

Measured and NOT adopted: int8 coopmat1 MMQ (lowered draft acceptance at 32k/107k), q8_0 draft KV, a throughput-driven draft
length policy, draft width 3 instead of 5 (+1 %, within noise), `-ub 4096`.

## 5. Two slots give no total throughput

Gate 2026-10-09, two concurrent streams at 393k each (`-np 2`, per-stream KV): each round costs 2.4x a solo round, so the total
is ~39 t/s vs 46 t/s for one stream (-15 %). Speculative decoding already batches the rows of one context with high expert
overlap; a second context shares fewer expert reads and adds its own depth work. A second slot is a concurrency feature (two
agents served at ~0.45x speed each), not a speed-up.

## 6. Disk prompt cache (2026-10-04, image v43)

132k-token chat, 12 turns. Branch after turn 6 (72.5k tokens): rewind to a stored checkpoint and replay 28 tokens in 0.6 s, vs a
fresh prefill of the same prompt in 209 s; the answer was identical to the fresh reference. New chat with the same system prompt:
0.7 s. The first conversation copied after another one took the slot: 0.9 s. Entry size ~15 KB/token + 0.3-0.5 GiB fixed;
loads at 6.8 GB/s cold, 8.8-10 GB/s from page cache. For comparison, re-prefill costs 58 s at 27k, 288 s at 108k, 468 s at 139k.

## 7. Hands box (balkon, Radeon 780M, 96 GB) - Qwen3.8-Flash-Next GSQ-RCO abliterated

| weights | date | decode | prefill | other |
|---|---|---|---|---|
| IQ3_XXS-hc8 | 2026-10-07 | 18 t/s | 129 t/s | 40-task agent coding bench 38/40 (Qwen3.6-35B-A3B on the same box: 35/40, decode 31, prefill 343); wall per task 510 vs 632 s |
| IQ3_XXS-hc8, 400k context | 2026-10-07 | 13.3 t/s | 72.6 t/s | needles at 399k found |
| IQ3_S-hc8-o8 (the compose) | 2026-10-10 | ~15-16 t/s | ~140 t/s at 1.7k | early end-of-reply P(EOS) 0.001 vs 0.090 on IQ3_XXS-hc8 |

QSA on the 780M at 128k: +48 % prefill, +68 % decode vs dense. `-ub 4096`, forced MMVQ and int8 MMQ: no gain.

## 8. Another model family: Qwen3.6-35B-A3B on the 780M, upstream vs this fork (2026-10-16, partial)

Question: do the generic levers help a model the fork was not built for? Model Qwen3.6-35B-A3B UD-Q4_K_XL (unsloth's file, MTP head included), balkon
(Radeon 780M, Vulkan/RADV, Mesa 25.2.8), the hands server stopped. Both arms: `-c 262144 --parallel 1 -ngl 999 -b 2048 -ub 2048
-t 8 -tb 8 --no-mmap --fit off --cache-ram 0`, KV f16 and flash attention at upstream's defaults, MTP off. Upstream = the official
image `ghcr.io/ggml-org/llama.cpp:server-vulkan-b10689`. Fork = this tree at the v60 revision with `GGML_VK_DISABLE_MMQ_CM1=1
GGML_VK_FREE_WAIT_IDLE=1 GGML_ALLOC_GROW_SLACK=10 LLAMA_RESERVE_RUNTIME_SHAPE=1 LLAMA_GRAPH_CACHE=8` (the int8 MMQ switch keeps
the arms on the same matmul path: b10689 predates upstream's int8 coopmat1 MMQ).

Caveat: the fork's base is b10664 plus drluoto's branch, not b10689, so "fork vs upstream" also contains about 25 upstream
commits and the drluoto changes. The per-lever split (the same fork with the listed levers switched off) is pending.

Cold prefill, t/s, median of 3:

| prompt | upstream b10689 | fork (v60 rev) | change |
|---|---|---|---|
| 8k | 304.7 | 312.5 | +2.6 % |
| 32k | 212.5 | 226.1 | +6.4 % |
| 72k | 131.4 | 153.1 | +16.5 % |

Decode, ms/token (lower is better):

| point | upstream b10689 | fork (v60 rev) | change |
|---|---|---|---|
| right after the 8k / 32k / 72k prompt (128 tokens, median of 3) | 48.96 / 58.44 / 71.68 | 48.04 / 55.05 / 66.72 | -1.9 / -5.8 / -6.9 % |
| at depth 4k (median of 3 windows of 128 tokens, one request) | 53.86 | 46.64 | -13.4 % |
| at depth 32k | 63.13 | 55.19 | -12.6 % |
| at depth 128k | 91.64 | 86.23 | -5.9 % |

The depth rows are n = 1 (one request per depth) with no noise band; each arm continues with its own generated tokens, so the
contexts differ by a few hundred tokens. GTT peak: 27 984 MiB upstream, 28 289 MiB fork (the fork arm runs the allocator env
above, so this is not a like-for-like allocation comparison).

Identity: greedy 64 tokens on a ~1k prompt were identical in both builds; the top-5 logprobs were not (top-5 id lists equal at
45 of 64 positions, max |delta logprob| 1.9 on matching ids). The fork claims EXACT only for Qwen3.8-Flash-Next against its own
previous version; for this model it is a different build with different numerics, not an EXACT claim.

Why these decode numbers are higher than the A3B row in section 7 (decode 31 t/s) and my earlier 31-33 ms/token band for
this model on this box: those were measured with the MTP draft on (my A3B hands compose: draft-mtp, KV q8_0); this A/B runs MTP
off, KV f16 and contexts up to 128k. It is a different configuration, not a regression.

Pending (scheduled): the per-lever attribution arms (fork with B1 / B3 / B4 / B4c / C1 and the graph cache off, and a second
arm with every other fork-only kernel off), the MTP-on arms, the backend-sampling arms, and a probe of which fork fusions
actually fire on this model.

## 9. Stock llama.cpp b10689 vs this fork (v60), same GGUF, gfx1151, n_ctx 262 144 dense (2026-10-16)

Run `20261016-203641` on s1. Stock = the official image `ghcr.io/ggml-org/llama.cpp:server-vulkan-b10689` (build 10689 /
57291f264); fork = the v60 image (845c6fbcd). Same files in every arm: Qwen3.8-Flash-Next-Uncensored i1-Q4_K_M + mmproj f16;
the MTP draft only in the fork's MTP arm. n_ctx 262 144, attention dense on both sides (fork `--qsa-ratio 0`), so this is not
the 1M profile of the sections above. The stock argv is the fork's minus `--qsa-ratio 0 --kv-graph-span 0`.

| measurement | stock b10689 | v60 | v60 vs stock |
|---|---:|---:|---:|
| cold prompt 8k / 32k / 72k / 128k, t/s (128k: n = 1; both MTP off) | 353.8 / 312.6 / 247.8 / 192.0 | 544.3 / 455.8 / 338.2 / 245.9 | +53.9 / +45.8 / +36.5 / +28.1 % |
| prefill steps 65 536 -> 261 120, aggregate, t/s | 107.3 | 124.1 | +15.7 % |
| last step 207 110 -> 261 120, t/s | 81.4 | 91.4 | +12.2 % |
| decode at 26k / 131k / 261k, MTP off, ms/token (restore path both) | 37.82 / 56.04 / 77.81 | 31.76 / 41.05 / 52.18 | +19.1 / +36.5 / +49.1 % t/s |
| decode at 26k / 131k / 261k, fork MTP on (continuous), ms/token | no stock MTP | 20.00 / 24.69 / 40.88 (acceptance 0.782 / 0.949 / 0.852) | +89.1 / +127.0 / +90.3 % vs stock MTP off |
| GTT at 262k, MTP off | 88 137 MiB | 85 420 MiB | -3.1 % |
| TTFT, continuation at 261.6k + 256 new tokens, median of 3 | 5 359 ms | 3 659 ms (MTP on) | x1.46 |

- **No stock MTP number.** b10689 failed to load the sidecar draft GGUF (`failed to load draft model`); upstream `qwen4exp` has no MTP graph.
- **Greedy-64 identity, stock vs fork, both MTP off: 62/64 tokens equal (against stock rep 0), first divergence at token 48.** Reported, not graded:
  stock lacks the GDN normalization fix (upstream PR #28068) and the fork's exactness patches.
- **The stock server was not run-to-run deterministic in this run:** greedy-64 rep 0 != rep 1 on the same 2 000-token prompt
  within one process; the fork arms were deterministic (fk0 with the strict top-5 check, fkm tokens only). Observed once (2 reps); reported, not diagnosed.

Method: corpus `tokens-s02.bin` (identical prompts in every arm), 1 slot, `-ub 2048`, KV q8_0. Cold prompts with
`cache_prompt` false, median of 3 (128k: n = 1). Fill: continuous history, one request per step; the fork fills in its MTP-on
process (the draft catches up on every prompt token), stock without a draft. Decode bands: greedy, 64 warm-up + 3 x 128 timed
tokens of one request, median window; both MTP-off sides by the restore path (each from its own saved history), the fork MTP
column continuous (a restore drops the draft KV). GTT at 262k: stock at the end of its continuous fill, fork in the MTP-off
restore band at 261 120 (the fork reserves the full KV at load). TTFT: median of 3 (1 warm, 2 after a restore).

Not comparable:

- 524k and 1M exist only in the fork (QSA + YaRN x4 over the native 262 144 window); stock stops at 262 144. QSA itself is
  fork-only and off here; the 1M-profile step numbers (474.3 t/s at 196 608 -> 262 144 etc.) are QSA numbers, not these.
- MTP needs the fork's sidecar draft GGUF (blk.48 nextn, vocabulary pruned to 65 536); upstream b10689 has no MTP graph.
- Fork base is b10664 + PR #27742 + drluoto + the GDN fix, not b10689: "fork vs b10689" also contains ~25 upstream commits.
- YaRN x4 (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144`) runs in both arms, as in the committed compose; a stock
  user at 262k would not set it.
- Decode bands differ by path: the MTP-off columns are restore-path on both sides, the fork MTP column is continuous.
- Fork-only features (disk prompt cache, slot adoption, GTT admission, `--kv-graph-span`) are off or inert in every arm; the
  fork's env knobs stay set in the fork arms.
