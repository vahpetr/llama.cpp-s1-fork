# Benchmarks

All numbers are measured on my two boxes, single runs unless a range is given. "vNN" is my image version (one image per fork
build; v53 = fork `prod-v11-stage6` @ f0fa8ddf1, v54 = v53 + webui and logging changes, v55 = v54 + webui
changes only, v56 = this tree's HEAD = v55 + webui changes only; same numerics).
Run-to-run noise on the speed track is about ±1.5 %.

Brain model everywhere below: Qwen3.8-Flash-Next-Uncensored i1-Q4_K_M + MTP draft head (vocab-pruned, see deploy/README.md),
KV q8_0, one slot, YaRN x4, QSA above 64k, s1 (Ryzen AI Max+ 395, GTT 110 GiB, Vulkan/RADV).

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

## 3. How we got here (same box, same weights)

| build | date | decode cold | prefill 8k | decode 107k | prefill 107k | decode 213k | prefill 213k | notes |
|---|---|---|---|---|---|---|---|---|
| stock llama.cpp PR #27742 image | 2026-08-27 | ~24 t/s | 365 t/s (6.5k) | - | - | - | - | IQ4_XS; 17.5 t/s decode at 26k; no MTP head in any GGUF |
| stock b10689 | 2026-08-30 | 27 t/s | - | - | - | - | - | MTP not supported by the stock converter/loader |
| v9 (first fork build with MTP) | 2026-09-06 | 49.6 | 386 | - | - | - | - | 5 slots, dense attention |
| v31-det (dense attention) | 2026-09-09 | - | - | 41.8 | 271 | 29.4 | 167 | depth track, 5 slots |
| v53 (QSA + sparse draft + graph cache) | 2026-10-09 | 61.0 | 550 | 51.6 | 407 | 59.9 | 421 | 1 slot |

The stock rows used a different quant (IQ4_XS / UD-Q4_K_XL of the base model), so they are a rough reference, not a paired A/B.
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
