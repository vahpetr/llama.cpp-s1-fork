# Benchmarks

Numbers for release v71 (image `flash-next:vulkan-v71`, fork revision `8bcde80da`, `prod-v11-stage9-pkg11`) on s1, plus one
dated comparison with stock llama.cpp (section 6). Single runs unless reps are given; the speed noise band is about ±1.5 %.

## 1. Setup and method

Host s1: Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5x, GTT 108 GiB (`ttm.pages_limit=28311552`), GPU DPM level
`high` while busy, kernel 7.0.0-38-generic; Mesa 26.0.8 RADV (`mesa-vulkan-drivers 26.0.8-1ubuntu0.3`) and `libvulkan1`
1.4.341 inside the image.

Model: `Qwen3.8-Flash-Next-Uncensored.i1-Q4_K_M.gguf` (mradermacher imatrix, 119 150 723 360 B = 110.97 GiB) + mmproj f16
(0.84 GiB). Draft: MTP head `mtp-Qwen3.8-Flash-Next.Q4_K_M.v65536-v5.gguf`, vocabulary pruned to 65 536 (not published; the
public setup uses the full-vocabulary head, so decode rows are an upper bound for it); `--spec-type draft-mtp
--spec-draft-n-max 5 --spec-draft-p-min 0.7 --spec-draft-sampling probabilistic`.

Server flags: the 1M compose (`deploy/s1/docker-compose.brain-1m.yml`): `-c 1048576 --parallel 1 --kv-unified`, `-ctk q8_0
-ctv q8_0`, `-b 2048 -ub 2048`, `-fa on`, `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144`, `--qsa-ratio 4` with
`LLAMA_QSA_MIN_KV=65536` (dense below 65 536 tokens), `-ot per_layer_token_embd=CPU`, `--load-mode mmap --tensor-read-lazy
auto`, `--ctx-checkpoints 4`, `--cache-disk-async 1024 --cache-disk-incr`; temp 0.7, top-p 0.8, top-k 20, min-p 0, presence
penalty 1.5, `--reasoning off`; `GGML_VK_DISABLE_MMQ_CM1=1` (int8 MMQ off). Top-k v3, the QSA bias fold and GDN replay run at
their v71 code defaults on gfx1151 (startup proof lines). The S0.2 runs strip `--cache-disk*` from the compose.

Method:

- **Cold prompt**: one prompt with `cache_prompt` false; t/s of the prompt.
- **Step → N**: the 65 536-token append that reaches depth N, in one continuous history; t/s of that step.
- **Fill**: sum of the raw step walls from 0 to the last depth.
- **Decode**: ms per token with MTP on, worst of 3 windows of 128 tokens at that depth, continuous history, fixed corpus
  (`tokens-s02`). No-draft decode is the restore path with the draft off.
- **TTFT**: 256 new tokens after a slot restore of a 1M conversation.
- **GTT**: `mem_info_gtt_used` of the GPU.

No stock llama.cpp or other-engine numbers were measured on v71.

## 2. S0.2 run 2026-11-04 (llama-server, corpus `tokens-s02`, 1M profile)

| measurement | v71 |
|---|---:|
| cold prompt 128k, t/s (first request; warm repeat 584.4) | 580.5 |
| 72k prompt, probe M2 (range of its reps), t/s | 511.2–512.2 |
| step 197 070 → 262 606, t/s | 676.9 |
| step 459 676 → 525 212, t/s | 648.2 |
| step 983 964 → 1 046 528 (62 564 new tokens), t/s | 585.3 |
| fill 0 → 1 046 528 (17 steps), min | 27.76 |
| decode with MTP at 32k / 262k / 1M, ms/token | 24.01 / 19.33 / 17.69 |
| the same in t/s | 41.6 / 51.7 / 56.5 |
| draft acceptance at 32k / 262k / 1M | 0.817 / 0.800 / 0.877 |
| decode without draft (restore path) at 32k / 262k / 1M, ms/token | 31.80 / 32.37 / 39.39 |
| TTFT, 256 new tokens at 1 046 989 after a slot restore, mean of the 2 restored reps, ms (1 266.9 / 1 303.6; warm rep without restore 1 302.9) | 1 285 |
| GTT peak over the 1M fill, MiB (admission bar 109 216) | 105 407 |
| 1M slot file: size / save time | 17.09 GiB / 3.0 s |

Decode at depth depends on how predictable the text is (draft acceptance); on this corpus 1M decodes faster than 32k.

## 3. AB64 run 2026-11-01 (llama-server, cold prompts, 2 reps, arm A3 = the v71 image)

| measurement | v71 |
|---|---:|
| cold prompt 32k, t/s (reps 623.41 / 623.58) | 623.5 |
| cold prompt 72k, t/s | 510.0 |
| GTT peak over the cold prompts, MiB | 104 739 |
| greedy identity against the reference arm | PASS |

## 4. Deploy smoke 2026-11-01 (the 1M compose, with the disk cache)

| measurement | v71 |
|---|---:|
| GTT after load, MiB | 104 580 |
| cold prompt 8k (8 192 tokens, n = 1), t/s | 789.1 |
| short chat (32-token prompt, 116 tokens), t/s | 84.3 |
| draft accepted / drafted on that reply | 95 / 100 |

## 5. Two slots and the hands box

Two concurrent streams (`-np 2`, per-stream KV) cost 2.4x a solo round each, so the total was 15 % below one stream: speculative
decoding already batches the rows of one context with high expert overlap, and a second context shares fewer expert reads. A
second slot is a concurrency feature, not a speed-up.

Hands (Radeon 780M): the v71 hands build (`:v71k`) is not published and has no measurements; the hands run `:v70k`.

## 6. Stock llama.cpp b10689 vs this fork (v60), same GGUF, gfx1151, n_ctx 262 144 dense (2026-10-16)

This is the last stock baseline; it was not re-run on v71.

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
  fork-only and off here; the 1M-profile step numbers above are QSA numbers, not these.
- MTP needs the fork's sidecar draft GGUF (blk.48 nextn, vocabulary pruned to 65 536); upstream b10689 has no MTP graph.
- Fork base is b10664 + PR #27742 + drluoto + the GDN fix, not b10689: "fork vs b10689" also contains ~25 upstream commits.
- YaRN x4 (`--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144`) runs in both arms, as in the committed compose; a stock
  user at 262k would not set it.
- Decode bands differ by path: the MTP-off columns are restore-path on both sides, the fork MTP column is continuous.
- Fork-only features (disk prompt cache, slot adoption, GTT admission, `--kv-graph-span`) are off or inert in every arm; the
  fork's env knobs stay set in the fork arms.
