# Benchmarks

All numbers are measured on my two boxes, single runs unless a range is given. "vNN" is my image version (one image per fork
build; v53 = fork `prod-v11-stage6` @ f0fa8ddf1, v54 = v53 + webui and logging changes, v55 = v54 + webui
changes only, v56 = v55 + webui changes only, v57 = v56 + webui changes only, v58 = v57 + webui
changes only, v59 = v58 + a server backend-sampling guard and webui changes; same numerics, gated: G6 same-key,
bit-identical to v58; v60 = v59 + patches 0227-0240, EXACT against v59, see section 0; v61 = this tree's HEAD = v60 +
patches 0241-0243, EXACT against v60 and the v59 reference, see section 0a).
Run-to-run noise on the speed track is about ±1.5 %.

Latest build: **v61 (2026-10-17)**. Its numbers against v60 are in section 0a (with the version lineage table at its end),
the v60-vs-v59 package is section 0, and both have a v61 column below. Sections 1 and 2 are the v52/v53 suite baselines (older,
slower at depth) with a v61 column where v61 was measured; "not re-measured" marks the rows that need a new server run.

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

| metric | v59 | v60 | v61 (section 0a) | v60 vs v59 | v60 with the f16 mirror off (0240) |
|---|---|---|---|---|---|
| fill step 196 608 -> 262 144 | 368.0 t/s | **474.3 t/s** | 491.4 t/s | +28.9 % | 460.7 |
| fill step 262 144 -> 327 680 | 340.2 t/s | 453.9 t/s | 470.8 t/s | +33.4 % | 443.0 |
| fill step 458 752 -> 524 288 | 284.1 t/s | **413.2 t/s** | 433.9 t/s | +45.4 % | 399.9 |
| fill step 983 040 -> 1 048 576 | 194.9 t/s | **328.9 t/s** | 357.1 t/s | +68.7 % | 311.0 |
| cold fill 0 -> 262k / 0 -> 524k (sum of prefill walls) | 687.3 s / 1 534.6 s | 584.3 s / 1 190.2 s | 572.0 s / 1 153.0 s | -15.0 % / -22.4 % time | 591.2 s / 1 215.7 s |
| cold fill 0 -> 1 048 576 (sum of 18 prefill walls) | 64.30 min | **43.99 min** | 41.80 min | -31.6 % time (x1.46) | 45.65 min |
| cold prompt 8k | 503.4 t/s | 548.8 t/s | 550.0 t/s | +9.0 % | 535.3 |
| cold prompt 32k | 423.0 t/s | **458.6 t/s** | 457.5 t/s | +8.4 % | 445.5 |
| cold prompt 64k | 335.4 t/s | 357.8 t/s | 357.9 t/s | +6.7 % | 350.6 |
| step on a restored prefix, 196 608 / 458 752 / 983 040 + 65 536 | 368.2 / 283.6 / 195.0 t/s | 472.8 / 413.1 / 326.0 t/s | 488.4 / 432.3 / 355.6 t/s | +28.4 / +45.7 / +67.2 % | 452.6 / 391.6 / 305.4 |
| GTT peak, fill pass (probe, no projector) | 104 078 MiB | 104 900 MiB | 104 544 MiB | +822 MiB | 104 121 MiB |

The gain grows with depth: the dense part below 64k gains 5-9.5 %, every QSA step above it more, up to +68.7 % at 1M. The
f16 mirror alone is worth +2.9 % (->262k) to +5.8 % (->1M) in the back-to-back fill pair and costs 779 MiB.

Server rows, llama-server with the full compose (vision projector loaded):

| metric | v59 | v60 | v61 | source |
|---|---|---|---|---|
| GTT after load | 104 591 MiB | 105 414 MiB (+823) | 105 495 MiB (+81) | the v60 deploy on 2026-10-16: v59 live just before the switch, v60 after load |
| GTT after the deploy smoke requests | - | 106 032 MiB | 105 519 MiB | each version's deploy smoke; the admission bar is 109 216 |
| GTT peak over a 1M fill | 105 196 MiB (section 2a) | not measured (at least 106 032 MiB, the after-smoke value) | 105 825 MiB at load, 105 627 over the fill | S0.2 runs, llama-server, corpus tokens-s02: v59 2026-10-15, v61 2026-10-18 (section 2a) |
| GTT peak, oracle probe score-capture pass (no projector) | 105 538 MiB | 106 357 MiB (+819) | 106 095 MiB (-262) | the gate's maximum; capture buffers, not a serving state |
| decode at depth (worst of 3 windows, draft on) | 26.07 / 22.10 / 24.38 ms/tok at 32k / 262k / 1M (section 2a) | not measured | 25.83 / 21.04 / 19.92 ms/tok (-0.9 / -4.8 / -18.3 %) | S0.2 runs, llama-server, corpus tokens-s02: v59 2026-10-15, v61 2026-10-18 (section 2a) |
| TTFT, 256 new tokens on a 1M history | 2 452 ms median of 3 (2 054 ms warm, section 2a) | not measured | 1 886 ms median of 3 (1 592 ms warm), -23 % | S0.2 runs, llama-server, corpus tokens-s02: v59 2026-10-15, v61 2026-10-18 (section 2a) |
| cold prompt 128k | 377.9 t/s (section 2a) | not measured | 427.4 t/s (+13.1 %) | S0.2 runs, llama-server, corpus tokens-s02: v59 2026-10-15, v61 2026-10-18 (section 2a) |

Class: EXACT. On my S0.1 oracle (fixed prompts, greedy; every compared position's logits, the saved states and the QSA
selected ids in order, at verify widths 1-6 in full and the last 6 positions of widths 7-9, depths 8k -> 1M, and restore /
replay histories) v60 matched v59 bit for bit at all 1 502 compared logit positions and 224 state points. The gate ran on
gfx1151 with the production configuration minus the vision projector (one slot, a text corpus); it covers the package as a
whole, not each patch alone, and the B2 tile (0233, off by default) was not in it;
multi-slot stream offsets were not covered. On the live server after the switch, three fixed requests (a 32k prompt, a chat,
a needle at 26.7k) gave the same tokens and the same number of accepted drafts as on v59; on the 32k prompt the top-5
logprobs differed by exactly 0.

Observed and not explained yet: in the probe, state saves were about 1.8x slower on v60 than on v59 (21.4 s vs 11.6 s for the
15.6 GB state at 1M) while loads and the state size were the same. On llama-server (section 2a, v61 vs v59, same bytes) the save took 0.5 -> 1.4 s at 262k and 3.2 -> 3.9 s at 1M; cause not established.

## 0a. v61 vs v60 (2026-10-17, 1M profile, oracle probe = the gate's fill/full passes)

Method as in section 0, with one difference: these rows come from the EXACT gate's own probe passes (the fill pass and the
full pass of each gate), not from separate timing runs. v60 = gate of 2026-10-15/16 (rev `845c6fbcd`, fill pass 2026-10-16),
v61 = gate of 2026-10-17 (rev `7a438f3ba`, fill pass 2026-10-17); same corpus, argv and env; single runs. The v59 and v60
columns of section 0 were recomputed from the same kind of records with the same extraction and reproduce exactly.

| metric | v60 | v61 | change |
|---|---|---|---|
| fill step 65 536 -> 131 072 | 512.2 t/s | 524.4 t/s | +2.4 % |
| fill step 196 608 -> 262 144 | 474.3 t/s | **491.4 t/s** | +3.6 % |
| fill step 262 144 -> 327 680 | 453.9 t/s | 470.8 t/s | +3.7 % |
| fill step 458 752 -> 524 288 | 413.2 t/s | **433.9 t/s** | +5.0 % |
| fill step 983 040 -> 1 048 576 | 328.9 t/s | **357.1 t/s** | +8.6 % |
| cold fill 0 -> 262k / 0 -> 524k (sum of prefill walls) | 584.3 s / 1 190.2 s | 572.0 s / 1 153.0 s | -2.1 % / -3.1 % time |
| cold fill 0 -> 1 048 576 (sum of 18 prefill walls) | 43.99 min | **41.80 min** | -5.0 % time (x1.05; x1.54 vs v59) |
| cold prompt 8k | 548.8 t/s | 550.0 t/s | +0.2 % |
| cold prompt 32k | 458.6 t/s | 457.5 t/s | -0.2 % |
| cold prompt 64k | 357.8 t/s | 357.9 t/s | +0.0 % |
| step on a restored prefix, 196 608 / 458 752 / 983 040 + 65 536 | 472.8 / 413.1 / 326.0 t/s | 488.4 / 432.3 / 355.6 t/s | +3.3 / +4.6 / +9.1 % |
| GTT peak, fill pass (probe, no projector) | 104 900 MiB | 104 544 MiB | -356 MiB |
| GTT peak, score-capture pass (the gate's maximum) | 106 357 MiB | 106 095 MiB | -262 MiB |

The dense range (below 64k) is unchanged within 0.4 %; every QSA step gains, more with depth (+2.4 % at 131k to +8.6 % at
1M), because both levers work on the indexer, whose cost grows with the pool. All 18 fill steps, v60 -> v61 wall (s):
14.938 -> 14.881, 57.345 -> 57.475, 112.284 -> 112.250, 127.954 -> 124.964, 133.551 -> 129.088, 138.180 -> 133.356,
144.386 -> 139.187, 149.070 -> 143.344, 153.853 -> 147.416, 158.620 -> 151.047, 164.056 -> 155.915, 168.396 -> 159.050,
172.816 -> 162.442, 178.724 -> 166.930, 183.262 -> 171.082, 189.147 -> 176.088, 193.810 -> 179.741, 199.249 -> 183.503.

Kernel level (s1, rev `7a438f3ba`, knob off vs on in one build):

| measurement | off | on | change |
|---|---|---|---|
| A2b: QSA_SCORE ms per call at the deployed chunk shape, 16 shapes (pool 16 384-262 144 blocks; f32 rows and f16-mirror forms), the gate's run | | | median ratio 0.877 (min 0.673, max 1.097) |
| A2b: pool 262 144 blocks, f16-mirror form (the production form at 3+ rows) / f32 rows, the gate's run | 2.929 / 3.870 ms | 2.568 / 2.696 ms | -12.3 / -30.3 % |
| A2b: pools 98 304-262 144 blocks, gate's run / a second run the same day (medians 0.877 / 0.830) | | | f32 rows -30.3 to -32.2 / -30.2 to -32.5 %; f16 mirror -12.3 to -22.6 / -15.9 to -21.2 % |
| A2b: pools 16 384-65 536 blocks, gate's run | | | -32.7 to +9.7 % (small calls, mixed) |
| A3: QSA_TOP_K GPU time, one QSA layer, 64k step -> 262k / -> 524k / -> 1M | 1.834 / 3.757 / 7.616 s | 1.219 / 2.993 / 6.097 s | -33.6 / -20.3 / -20.0 % |
| A3: selection scratch preallocated (524k test fill) | 258.00 MiB | 32.25 MiB | -225.75 MiB (the cell-form scratch 257 -> 1 MiB) |
| A3: whole-pass wall inside the gate, prefill 8k / decode-like tails at 262k | | | ratio 0.980 / 0.993 |

Gate section walls (v60 gate vs v61 gate; coarse: they include state hashing and capture I/O, not only compute):

| section | v60 | v61 |
|---|---|---|
| arm (logits + states pass) | 116:36 | 115:03 |
| ids capture | 125:59 | 124:00 |
| score capture | 200:08 | 196:08 |
| dispatch log pass | 35:53 | 35:27 |
| fill pass | 45:48 | 43:35 |

Server rows, llama-server with the full compose (vision projector loaded):

| metric | v60 | v61 | source |
|---|---|---|---|
| GTT after load | 105 414 MiB | 105 495 MiB (+81) | the v61 deploy on 2026-10-17: v60 live just before the switch, v61 after load |
| GTT after the deploy smoke requests | 106 032 MiB | 105 519 MiB | each version's deploy smoke; the admission bar is 109 216 |
| GTT peak over a 1M fill | not measured | 105 825 MiB at load, 105 627 over the fill (v59: 105 196 / 105 193) | S0.2 runs, llama-server, corpus tokens-s02: v59 2026-10-15, v61 2026-10-18 (section 2a) |
| decode at depth (worst of 3 windows, draft on) | not measured | 25.83 / 21.04 / 19.92 ms/tok at 32k / 262k / 1M (v59: 26.07 / 22.10 / 24.38) | S0.2 runs, llama-server, corpus tokens-s02: v59 2026-10-15, v61 2026-10-18 (section 2a) |
| TTFT, 256 new tokens on a 1M history | not measured | 1 886 ms median of 3, 1 592 ms warm (v59: 2 452 / 2 054) | S0.2 runs, llama-server, corpus tokens-s02: v59 2026-10-15, v61 2026-10-18 (section 2a) |
| cold prompt 128k | not measured | 427.4 t/s (v59: 377.9) | S0.2 runs, llama-server, corpus tokens-s02: v59 2026-10-15, v61 2026-10-18 (section 2a) |

Class: EXACT, the same gate as v60's (section 0) against the same v59 reference: 1 502 logit positions and 224 state points
identical, ordered selected ids identical; plus an off arm per lever (A2b off, A2b forced onto its fallback with a 4-workgroup
limit, A3 off) equal to the reference on the short case set. 91 PASS, 4 INFO, 0 FAIL. On the live server after the switch, the
three fixed requests (a 32k prompt, a chat, a needle) gave the same tokens and accepted drafts as a fresh v60 reference; on the
32k prompt the top-5 logprobs differed by exactly 0.

### Version lineage on the brain

| version | date | what changed | class vs the previous | disk-cache key |
|---|---|---|---|---|
| v52 | 2026-10-09 | graph + allocation-plan cache (decode 524k 23.6 -> 22.2 ms/tok) | EXACT | - |
| v53 | 2026-10-09 | multi-slot fixes (one-slot behaviour = v52) | one slot = v52 | - |
| v54 | 2026-10-10 | webui compaction, logs and counters | same numerics | same as v53 |
| v55-v58 | 2026-10-10/11 | webui only | same numerics | same |
| v59 | 2026-10-11 | server backend-sampling guard + webui | bit-identical to v58 | same |
| v60 | 2026-10-16 | patches 0227-0240: prefill kernels (B1, B3, B4, B4c), QSA score epilogue (A1) and f16 pooled-key mirror (A2a), mat-vec hoist (C1), KQ visibility mirror (C2), indexer cell-ext restore (0239) | EXACT (oracle above) | new: adopts the v59 namespace read-only (docs/s1/DEPLOY.md) |
| v61 | 2026-10-17 | patches 0241-0243: QSA score GEMM key-tile-major dispatch (A2b), QSA select without key rows (A3), test / comment polish | EXACT (section 0a) | same as v60 (the v59 adopt stays) |
| v62 | 2026-10-19 | contrib 0244-0251 at rev e4af0a6ca: correctness and diagnostics, `--spec-draft-n-ubatch` ships unset | numbers = v61 (section 0b) | same as v61 |

## 0b. v62 (2026-10-19): correctness release, numbers = v61


v62 = v61 + patches 0244-0251 (correctness and diagnostics; `--spec-draft-n-ubatch` ships unset). Prefill, decode and fill
are unchanged by design, so every v61 number below stands for v62. Slot save identity (u1-server-ab, llama-server): the saved
bytes are identical v61 <-> v62 at 262k and 1M; save 527 -> 549 ms at 262k and 2 269 -> 2 316 ms at 1M (noise band).
Source: docs/releases/v62-regressions.md in the piagent repo.

## 1. 524k profile baseline - suite runs 2026-10-09 21:50 and 2026-10-10 02:30 (image v53; v54-v61 keep these numerics,
v60/v61 gated EXACT on the 1M oracle; their speed gains at depth are in sections 0 and 0a)

| metric | v53 suite (524k profile) | v61 (1M profile live server, deploy smoke 2026-10-17, and the oracle probe of section 0a) |
|---|---|---|
| decode, cold (empty context), mean of code / C# / essay | 61.0 t/s (61.5 / 70.9 / 50.7) | 75.3 t/s on one 116-token chat reply (draft 95/100); not the suite's 3-prompt mean |
| decode at 27k depth | 57.1-58.2 t/s (draft acceptance 0.95) | 57.2 t/s worst decode window at 26.7k (acceptance 0.954) |
| decode at 107k depth | 51.6 t/s (acceptance 0.94) | not re-measured |
| decode at 213k depth | 59.9 t/s (acceptance 0.93) | not re-measured |
| decode at 524k depth | 22.2 ms/tok = 45 t/s (v52 gate, 2026-10-09) | not re-measured at 524k; S0.2 corpus, worst window: 21.04 ms/tok at 262k, 19.92 at 1M (section 2a) |
| prefill 8k (synthetic / real text: wiki en, wiki ru, C#, HTML) | 550 / 524-528 t/s | 550.0 t/s (probe, text) |
| prefill 32k | 450 t/s | 457.5 t/s (probe); 501.3 t/s on the 26.7k smoke prompt |
| prefill of a 2k chunk at 107k / 213k depth | 407 / 421 t/s | not re-measured (65k steps at 131k / 262k: 524.4 / 491.4 t/s, probe) |
| cold fill 0 -> 524k | 25.6 min | 19.2 min (1 153.0 s, probe, 1M profile) |
| 5 parallel short streams, total | 220 t/s | not re-measured (one slot in the 1M profile) |
| GTT peak | 94 493 MiB = 92.3 GiB | 1M profile: 105 495 MiB after load, 105 519 after the smoke (524k profile not run) |
| needle in a haystack 32k / 107k / 213k | 2/2, 2/2, 2/2 | 2/2 at 26.7k (smoke); 107k / 213k not re-measured |

Decode at 213k reads faster than at 107k in these runs; the prompts differ in how predictable they are (acceptance), so compare
depths within one row of section 3 rather than across this table.

## 2. 1M profile - 2026-10-09 (image v52, same numerics as v53), cold fill in 16 steps of 65 536 tokens; v60/v61 fill and step numbers: sections 0 and 0a

| depth | prefill of the step, t/s | decode ms/tok | decode t/s | draft acceptance | GTT MiB | v61 prefill of the step, t/s (oracle probe 2026-10-17, section 0a) |
|---|---|---|---|---|---|---|
| 65 536 | 333 | 21.4 | 46.8 | 0.77 | 105 188 | 550.5 / 427.6 / 291.9 (the probe fills 0 -> 8k -> 32k -> 64k) |
| 131 072 | 435 | 19.5 | 51.2 | 0.81 | 105 193 | 524.4 |
| 262 144 | 361 | 21.2 | 47.1 | 0.82 | 105 193 | 491.4 |
| 524 288 | 278 | 22.4 | 44.7 | 0.82 | 105 193 | 433.9 |
| 786 432 | 227 | 22.1 | 45.3 | 0.85 | 105 193 | 392.6 |
| 917 504 | 208 | 25.1 | 39.8 | 0.82 | 105 193 | 372.2 |
| 1 048 320 | 192 | 28.2 | 35.5 | 0.78 | 105 193 | 357.1 (step 983 040 -> 1 048 576) |

Whole fill 0 -> 1M: 1.09 h (v61: 0.70 h both in the probe, section 0a, and on llama-server, section 2a; v61 decode on the S0.2
corpus, worst window, draft on: 21.04 ms/tok at 262k and 19.92 at 1M; server GTT over the v61 fill 105 627 MiB, section 2a). GTT is flat from the first step (102.7 GiB)
because every compute buffer is reserved at load. No admission waits. Host RSS 3.3-7.5 GiB. The same 1M arm on v47 (2026-10-06): 17.8 t/s at 1M, fill 1.76 h, GTT 106 043 MiB.

## 2a. S0.2 H0 rows: v59 (2026-10-15) and v61 (2026-10-18), llama-server, corpus tokens-s02

Images v59 and v61, 1M brain profile without the disk cache (one slot, vision projector loaded), corpus tokens-s02 (sha256
e77e4bea67adebe8..., 1 098 504 tokens); the same driver, argv, env and token positions in both runs; single runs (noise
+-1.5 %). Decode rows are worst-of-3 windows of 128 tokens on a continuous history; they are a baseline on this corpus and not
comparable 1:1 with the rows of sections 1 and 2 (other text, other acceptance). v60 was not measured this way (it was live
for one day); v59 -> v61 covers the v60 and v61 packages together.

| row | v59 | v61 | change |
|---|---|---|---|
| cold prompt 128k (first request of the process) | 377.9 t/s (warm repeat 378.4) | 427.4 t/s (warm repeat 429.5) | +13.1 % |
| prefill 72k, two repeats | 346.4 / 348.3 t/s | 377.2 / 377.9 t/s | +8.9 / +8.5 % |
| fill step ->262k / ->524k / ->1M | 370.1 / 280.7 / 195.1 t/s | 483.2 / 425.0 / 351.5 t/s | +30.6 / +51.4 / +80.2 % |
| whole fill 0 -> 1 046 528 (17 steps, bands inside) | 1.07 h | 0.70 h | -34.6 % time |
| TTFT follow-up at 1M, 256 new tokens | 2 054 ms warm; 2 511 / 2 452 ms after a slot restore (median 2 452) | 1 592 ms warm; 1 886 / 1 901 ms after a slot restore (median 1 886) | -22.5 % warm, -23.1 % median |
| slot save at 32k / 262k / 1M | 0.57 / 3.72 / 14.48 GiB in 0.1 / 0.5 / 3.2 s | 0.57 / 3.72 / 14.48 GiB in 0.1 / 1.4 / 3.9 s | same bytes; +0.9 s at 262k, +0.7 s at 1M |
| GTT peak (server, projector loaded): at load / over the fill | 105 196 / 105 193 MiB | 105 825 / 105 627 MiB | +629 / +434 MiB |
| run wall (= prod downtime) | 87.5 min | 63.5 min | -27 % |

Serial decode without the draft versus draft on (ms/tok, worst window). The accepted / drafted counts and the per-round widths
are identical in the two runs: the same tokens came out, as the EXACT class says.

| depth | no draft v59 | no draft v61 | draft on v59 | draft on v61 | draft acceptance (both) | gain v59 | gain v61 |
|---|---|---|---|---|---|---|---|
| 32 768 | 33.07 | 32.69 | 26.07 | 25.83 | 0.817 (223/273) | -21 % | -21 % |
| 262 144 | 33.76 | 33.33 | 22.10 | 21.04 | 0.800 (264/330) | -35 % | -37 % |
| 1 046 528 | 42.45 | 40.65 | 24.38 | 19.92 | 0.877 (306/349) | -43 % | -51 % |

v61 vs v59: with the draft -0.9 / -4.8 / -18.3 % at 32k / 262k / 1M, without it -1.1 / -1.3 / -4.2 %. The draft-on path gains
more at 1M because it runs the QSA indexer more often per emitted token (the sparse MTP draft has its own indexer pass and the
verify round scores a plane of 5-6 rows); the levers were not isolated one by one here.

Per-round verify width with the draft on (width = sampled token + drafts; identical in both runs):

| depth | share of rounds with width >= 3 | mean width |
|---|---|---|
| 32 768 | 41.0 % | 2.73 |
| 262 144 | 65.8 % | 3.77 |
| 1 046 528 | 97.4 % | 5.51 |

The slot save is slower on v61 at the same size (262k 0.5 -> 1.4 s, 1M 3.2 -> 3.9 s): the server-side form of the probe's
state-save slowdown (section 0, last paragraph). Loads were not timed here.

## 3. How we got here (same box, same weights)

| build | date | decode cold | prefill 8k | decode 107k | prefill 107k | decode 213k | prefill 213k | notes |
|---|---|---|---|---|---|---|---|---|
| stock llama.cpp PR #27742 image | 2026-08-27 | ~24 t/s | 365 t/s (6.5k) | - | - | - | - | IQ4_XS; 17.5 t/s decode at 26k; no MTP head in any GGUF |
| stock b10689 | 2026-08-30 | 27 t/s | - | - | - | - | - | MTP not supported by the stock converter/loader |
| v9 (first fork build with MTP) | 2026-09-06 | 49.6 | 386 | - | - | - | - | 5 slots, dense attention |
| v31-det (dense attention) | 2026-09-09 | - | - | 41.8 | 271 | 29.4 | 167 | depth track, 5 slots |
| v53 (QSA + sparse draft + graph cache) | 2026-10-09 | 61.0 | 550 | 51.6 | 407 | 59.9 | 421 | 1 slot |
| v60 (v59 + prefill kernels, QSA score epilogue, f16 pooled-key mirror) | 2026-10-16 | - | 549 (probe) | - | - | - | - | 1 slot; decode at depth not re-measured; 65k fill steps at ->262k / ->524k / ->1M 474 / 413 / 329 t/s, cold fill 0 -> 1M 44.0 min (section 0) |
| v61 (v60 + QSA score GEMM dispatch A2b, select without key rows A3) | 2026-10-17 | 75.3 (one chat, deploy smoke; not the 3-prompt mean) | 550 (probe) | - | - | - | - | 1 slot; S0.2 corpus (section 2a): decode 47.5 / 50.2 t/s at 262k / 1M (draft on, worst window), 65k fill steps 483 / 425 / 352 t/s on llama-server (491 / 434 / 357 in the probe), cold fill 0 -> 1M 0.70 h |

The stock rows used a different quant (IQ4_XS / UD-Q4_K_XL of the base model), so they are a rough reference, not a paired A/B.
The paired A/B on the same GGUF (stock b10689 vs v60, n_ctx 262 144 dense) is section 9.
At depth the gain comes from QSA (sparse attention over 2 051 selected cells per query instead of the whole context); at 1M it is
the difference between possible and not.

## 4. What single levers were worth (paired gates on s1)

| lever | class | effect | when |
|---|---|---|---|
| QSA select without key rows (A3, patch 0243, v61) | exact (gate section 0a) | QSA_TOP_K GPU time per layer -33.6 / -20.3 / -20.0 % at the 64k step -> 262k / -> 524k / -> 1M; selection scratch 258 -> 32 MiB; GTT fill peak -356 MiB | 2026-10-17 |
| QSA score GEMM key-tile-major dispatch (A2b, patch 0241, v61) | exact (gate section 0a) | QSA_SCORE kernel ms median x0.877 over 16 shapes (f32 rows -30 %, f16 mirror -12 to -23 % at pools 98k-262k); A2b + A3 together: fill step -> 1M +8.6 %, -> 524k +5.0 %, cold fill 0 -> 1M -5.0 % | 2026-10-17 |
| f16 pooled-key mirror (A2a, patch 0240, v60) | exact | fill step +2.9 % (-> 262k) to +5.8 % (-> 1M), +779 MiB GTT (section 0, last column) | 2026-10-16 |
| v60 package 0227-0240 as a whole (B1, B3, B4, B4c, A1, A2a, C1, C2) | exact (gate section 0) | fill step -> 262k / -> 524k / -> 1M +28.9 / +45.4 / +68.7 %, cold fill 0 -> 1M 64.3 -> 44.0 min, cold prompt 8k / 32k / 64k +9.0 / +8.4 / +6.7 % | 2026-10-16 |
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
| IQ3_S-hc8-o8, image v61k (the compose since 2026-10-17) | 2026-10-17 | 24.3 t/s on one short chat reply (deploy smoke, draft 183/189) | not re-measured | disk-cache restore of a stored 6 584-token entry: PASS (prompt_n 4, cache_n 6 580); v61's levers act on the QSA indexer above 64k only |

QSA on the 780M at 128k: +48 % prefill, +68 % decode vs dense. `-ub 4096`, forced MMVQ and int8 MMQ: no gain.

## 8. Another model family: Qwen3.6-35B-A3B on the 780M, upstream vs this fork (2026-10-17; fork arm = pkg7, the v60 code; not re-run on v61)

Question: do the generic levers help a model the fork was not built for? Model Qwen3.6-35B-A3B UD-Q4_K_XL (unsloth's file, MTP head included), balkon
(Radeon 780M, Vulkan/RADV, Mesa 25.2.8, kernel 7.0.0-38), the hands server stopped. All arms: `-c 262144 --parallel 1 -ngl 999 -b 2048 -ub 2048
-t 8 -tb 8 --no-mmap --fit off --cache-ram 0`, no `--cache-disk`, KV f16 and flash attention at upstream's defaults, one slot. Upstream = the official
image `ghcr.io/ggml-org/llama.cpp:server-vulkan-b10689`, no env. Fork = this tree at the v60 revision (image `flash-next:vulkan-pkg7-845c6fb`).
Arms and their env, verbatim:

- **fork** (fk; fkc = the same env on the same night as fk0a/fk0b): `GGML_VK_DISABLE_MMQ_CM1=1 GGML_VK_FREE_WAIT_IDLE=1 GGML_ALLOC_GROW_SLACK=10 LLAMA_RESERVE_RUNTIME_SHAPE=1 LLAMA_GRAPH_CACHE=8`
- **fk0a** (the listed levers off, graph cache off): `GGML_VK_DISABLE_MMQ_CM1=1 GGML_VK_FREE_WAIT_IDLE=1 GGML_ALLOC_GROW_SLACK=10 LLAMA_RESERVE_RUNTIME_SHAPE=1 GGML_VK_MMID_PRECAST_B=0 GGML_VK_MMID_TILE_BM128=0 GGML_VK_FUSE_MULTI_ADD_SCALE=0 GGML_VK_FUSE_SCALE_SILU=0 GGML_VK_FUSE_CONCAT_SSM_CONV=0 GGML_VK_DENSE_TILE_M16=0 GGML_VK_MMVQ_HOIST=0`
- **fk0b** (every fork-only knob off): `GGML_VK_DISABLE_MMQ_CM1=1 GGML_VK_FREE_WAIT_IDLE=1 GGML_ALLOC_GROW_SLACK=10 LLAMA_RESERVE_RUNTIME_SHAPE=1 GGML_VK_MMID_PRECAST_B=0 GGML_VK_MMID_TILE_BM128=0 GGML_VK_FUSE_MULTI_ADD_SCALE=0 GGML_VK_FUSE_SCALE_SILU=0 GGML_VK_FUSE_CONCAT_SSM_CONV=0 GGML_VK_DENSE_TILE_M16=0 GGML_VK_MMVQ_HOIST=0 GGML_VK_FUSE_GDN_PRE=0 GGML_VK_FUSE_SIGMOID_MUL=0 GGML_VK_FUSE_HC_COMBINE=0 GGML_VK_CONCAT_TRANSPOSE=0 GGML_VK_MMID_COMPACT=0 GGML_VK_DISABLE_HOIST_ROW_IDS=1 GGML_VK_FA_GQA_TOKENS=0 GGML_VK_FA_NO_ZERO_DEAD_V=1 LLAMA_GDN_NO_DIRECT=1 LLAMA_SAMPLER_FASTPATH=0`

`GGML_VK_DISABLE_MMQ_CM1=1` keeps the fork on the same matmul path as upstream (b10689 predates upstream's int8 coopmat1 MMQ).
MTP-on arms add `draft-mtp` (n-max 3, p-min 0.7); the -bs arms add backend sampling.

Mixed reference: the fork's base is b10664 plus drluoto's branch and a GDN fix, not b10689, so "fork vs upstream" mixes the
fork's levers with about 25 upstream commits, the drluoto changes and fork-only default kernels. Per-lever attribution comes
only from fk vs fk0a. fk0b vs upstream bounds what is not a fork knob; some fk0b knobs may also switch off paths upstream has,
so it is a lower bound, not an upstream equivalent. Gain columns: positive = the first-named build is better (higher t/s,
lower ms/token).

Cold prefill, t/s, median (MTP off):

| prompt | upstream | fork | fork vs up | fkc | fk0a | fk vs fk0a | fk0b | fk0b vs up | n up/fk/fkc/fk0a/fk0b |
|---|---|---|---|---|---|---|---|---|---|
| 8k | 304.1 | 310.7 | +2.2 % | 330.7 | 318.9 | **+3.7 %** | 280.1 | -7.9 % | 3/3/1/1/1 |
| 32k | 212.9 | 225.1 | +5.7 % | 227.8 | 224.5 | **+1.5 %** | 199.4 | -6.3 % | 3/3/1/1/1 |
| 72k | 130.7 | 152.6 | +16.8 % | 153.3 | 151.6 | **+1.1 %** | 122.8 | -6.0 % | 3/3/1/1/1 |

The fk vs fk0a column uses fkc (same night as fk0a/fk0b).

Decode right after each cold prefill, ms/token (128 tokens, median over the repeats):

| prompt | upstream | fork | fork vs up | fk0a | fk vs fk0a |
|---|---|---|---|---|---|
| 8k | 46.48 | 46.91 | -0.9 % | 46.78 | +0.0 % |
| 32k | 53.50 | 54.42 | -1.7 % | 54.30 | -0.2 % |
| 72k | 65.17 | 66.84 | -2.6 % | 66.86 | -0.1 % |

Decode at depth, MTP off, ms/token (median of 3 windows of 128 tokens of one request per depth: n = 1, no noise band):

| depth | upstream | fork | fork vs up | fkc | fk0a | fk vs fk0a | fk0b |
|---|---|---|---|---|---|---|---|
| 4k | 49.77 | 46.22 | +7.1 % | 46.39 | 46.13 | -0.6 % | 50.30 |
| 32k | 56.03 | 54.62 | +2.5 % | 54.48 | 54.54 | +0.1 % | 58.70 |
| 128k | 85.28 | 84.38 | +1.1 % | - | - | - | - |

Decode at depth, MTP on (draft-mtp n-max 3, p-min 0.7, the multi-row verify path), ms/token, n = 1:

| depth | upstream | fork | fork vs up | draft acceptance up / fork |
|---|---|---|---|---|
| 4k | 35.54 | 32.39 | +8.9 % | 0.934 / 0.932 |
| 32k | 39.48 | 38.17 | +3.3 % | 0.958 / 0.965 |
| 128k | 69.70 | 75.16 | -7.8 % | 0.924 / 0.890 |

At 128k the fork is 7.8 % slower with a lower draft acceptance, at n = 1. Acceptance is content-dependent and each arm continues
the conversation with its own generated tokens (the 32k/128k contexts differ by at most 2 x 448 tokens between arms), so this
row does not separate kernel speed from acceptance; no fk0a MTP arm exists to isolate it.

Backend sampling (`-bs`), MTP off, depth 4k only: upstream 45.51, fork 45.57 ms/token (-0.1 %).

Which fork levers fire on this model (probe: 2k cold prompt + 32 tokens, MTP on, `GGML_VK_FUSION_LIST=1 GGML_VK_DISPATCH_LOG=1`,
fk vs fk0a):

- B4 `concat_ssm_conv`, B3 `multi_add_scale` and B3 `scale_silu` do not fire on qwen35moe (11 distinct fusions in both arms, none
  only in one of them).
- B1 mmid precast fires: 83 staging cpys in fk, 0 in fk0a; the expert GEMMs run `matmul_id_subgroup_q4_k/q6_k_f16_f16acc_aligned_l`
  in fk vs `matmul_id_subgroup_q4_k/q6_k_f32_f16acc_aligned_m` in fk0a.
- C1 MMVQ hoist fires: the `mul_mat_vec_*_hoist` pipelines appear in fk only.
- A1/A2a (QSA; no QSA metadata in the GGUF), C2 (needs `LLAMA_KQ_VIS=1`, set in no arm) and hc_combine (qwen4exp only) do not
  apply by construction. The graph cache K = 8 is set by env in fk; its proof line does not reach the server log.

Reading per lever class (attribution = fk vs fk0a only, n = 1 per point):

- **B1 (expert GEMM pre-cast + BM128 tile):** fires; the fk vs fk0a prefill gain of +3.7 / +1.5 / +1.1 % at 8k / 32k / 72k is
  attributed mainly to B1 (fk0a also turns off C1, B4c and the graph cache).
- **C1 + graph cache (decode, MTP off):** fk vs fk0a at depth -0.6 / +0.1 % (4k / 32k) and +0.0 / -0.2 / -0.1 % right after the
  prefills: below 1 % (n = 1, no noise band), so neither is measurable on this model. fk0a drops both together, so they are not separated.
- **B3 / B4:** contribute nothing here because they do not fire.
- **B4c (m <= 16 tile):** not isolated (there is no fk0a MTP process).
- The large "fork vs up" prefill gain at 72k (+16.8 %) is mostly not the listed levers (fk vs fk0a +1.1 %); it is the mixed
  reference. fk0b is -7.9 / -6.3 / -6.0 % vs upstream in prefill; that is a lower bound for the non-knob part, not an
  upstream-equivalent arm.

Identity (greedy 64 tokens, ignore_eos, top-5 logprobs, ~1k prompt): the generated tokens were IDENTICAL (64/64) in every pair
(upstream vs fork MTP off and MTP on, fk vs fk0a, fk0a vs fk0b, MTP off vs on in each build). fk vs fk0a: top-5 lists equal at
64/64 positions, max |delta logprob| 0 (fkc vs fk0a), so the levers do not change the numerics on this 64-token probe. Upstream vs fork (MTP off): top-5
lists equal at 45/64 positions, max |delta logprob| 1.899 on matching ids; this is a difference between two builds with different
numerics, reported, not a defect. The fork claims EXACT only for Qwen3.8-Flash-Next against its own previous version; for this
model it makes no EXACT claim.

GTT peak per process, MiB:

| arm | MTP off | MTP on | -bs |
|---|---|---|---|
| upstream | 27 984 | 31 453 | 28 256 |
| fork | 28 289 | 31 477 | 28 292 |

The fork arms run the A3B-profile allocator env (`GGML_VK_FREE_WAIT_IDLE=1 GGML_ALLOC_GROW_SLACK=10 LLAMA_RESERVE_RUNTIME_SHAPE=1`),
so this is "A3B profile vs stock", not a like-for-like allocation comparison.

Why these decode numbers are higher than the A3B row in section 7 (decode 31 t/s) and my earlier 31-33 ms/token band for
this model on this box: those were measured with the MTP draft on (my A3B hands compose: draft-mtp, KV q8_0); in this A/B only the
MTP-off tables run without the draft, and all arms use KV f16 and contexts up to 128k. The MTP-on rows here are closer (fork at
4k: 32.39 ms/token, inside the old 31-33 band). It is a different configuration, not a regression.

Evidence: run 20261016T222558Z on balkon; upstream image `ghcr.io/ggml-org/llama.cpp:server-vulkan-b10689` sha256:584ed154…,
fork image `flash-next:vulkan-pkg7-845c6fb` sha256:d21f904a…, model sha256 55983c5a…; kernel 7.0.0-38, Mesa 25.2.8; 24 driver
checks and all runbook checks (pinned sums, images, model, argv/env as frozen per arm, hands restore) 0 FAIL.

## 9. Stock llama.cpp b10689 vs this fork (v60), same GGUF, gfx1151, n_ctx 262 144 dense (2026-10-16; holds for v61: its two levers are QSA indexer kernels, which this dense arm never runs)

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
