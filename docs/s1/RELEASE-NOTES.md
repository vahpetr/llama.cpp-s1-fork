# Release notes

One entry per published image. Older versions (v52-v59) are summarised in the lineage table of
[BENCHMARKS.md](BENCHMARKS.md), section 0; every change is a numbered row in [PATCHES.md](PATCHES.md).

## v64 - 2026-10-21

Images: `vahpetr/llama-cpp-s1-fork:v64` = `:v64k` = `:latest` (one digest,
`sha256:b3c12a3a7f680d9d1683edbc1b5728b285766618bfacb9aa2159d9eed891ff28`), revision `c5b3462ed` (`prod-v11-stage9-pkg3`).
In production since 2026-10-21 22:12 (s1 brain; local build `flash-next:vulkan-v64`, image id `bb175dd1070a`) / 22:16
(balkon hands; `flash-next:vulkan-v64k`, image id `7e8d8fe2bbe9`). Rollback: `:v63` / `:v63k` (`8c1a468aa`), same key and
slot-file format; the v63 image accepts the v64 composes' `--spec-draft-sampling` line (the flag exists there, off by default),
so a rollback compose keeps it unless you remove the two lines.

**What it is.** v63 plus three kernel patches (0262, 0263, 0265: ggml-vulkan and tests only, no server change), on by their
code defaults, and one compose change: `--spec-draft-sampling probabilistic` (0252, in the code since v63) is set in every
compose with a draft model.

- Flash attention (0263, FA1): two shared-memory layouts for the coopmat1 kernel on dense multi-row dispatches - the P.V
  staging aliased onto the softmax staging (disjoint lifetimes) and the dead-column V staged per 16-row chunk; 31 -> 21 KiB
  of LDS at head size 256, 4 -> 6 workgroups per WGP on RDNA3. Decode, verify and the indexed QSA path keep their pipelines.
  Knobs `GGML_VK_FA_SFPV_ALIAS`, `GGML_VK_FA_V_CHUNK`.
- Expert GEMMs (0262, G2a): v60's BM 128 x BN 64 `mul_mat_id` tile runs 8 warps of 64 x 16 instead of 4 of 64 x 32, with an
  LDS row pad of 2 (a 72-byte stride; conflict-free but not 16-byte aligned, so RADV only by design - `_PAD=4` is the
  conformant layout and gives no gain). Same fragments, same K loop, same bits. Knob `GGML_VK_MMID_TILE_G2A`; startup line
  `mmid_tile_g2a: on (...)`.
- Dense GEMMs (0265, G1): a 3-rule tile table for the large prefill matmuls (q4_K / q6_K m >= 2048 k >= 2560 -> BM 64 x BN
  128; q5_0 m >= 2048 k <= 640 -> BM 128 x BN 64; 2 warps, LDS pad 4, subgroup 64 pinned); the hyper-connection shapes stay on
  the base tile. Knob `GGML_VK_DENSE_TILE_G1`; startup line `dense_tile_g1: table on (3 rules) (...)`.
- Probabilistic draft sampling as the compose default (0252; rulings R134 / R135): at T > 0 the MTP draft is sampled at the
  request's temperature and verified by rejection sampling; a T = 0 request is unchanged (greedy, bit-identical).

**Measured** (1M profile on s1; the kernel A/B = one image with the three knobs off and on, llama-server, 2026-10-21; details
in BENCHMARKS.md sections 0d and 2c):

| metric | v63 kernels | v64 | change |
|---|---|---|---|
| prefill, fresh 32k / 72k prompt (cold, 2 reps per arm) | 467.5 / 385.6 t/s | 500.9 / 417.3 t/s | +7.15 / +8.22 % |
| node time of one cold 32k request under the perf logger: FLASH_ATTN_EXT / MUL_MAT_ID / G1-eligible MUL_MAT / all nodes | 19.39 / 16.14 / 17.62 / 68.45 s | 16.20 / 15.48 / 16.89 / 63.83 s | -16.5 / -4.1 / -4.1 / -6.8 % (attribution, not a timing run) |
| identity, kernels off vs on | 32 greedy tokens + top-8 logprobs at 32k, the first token at 72k | identical | |
| decode at 32k, live server, 256 tokens x 3, T = 0 / pi's thinking card / pi's instruct card (both runs with the probabilistic flag) | 22.07 / 22.51 / 22.91 ms/tok | 21.27 / 21.68 / 22.28 ms/tok | -3.6 / -3.7 / -2.7 %, same accepted / drafted counts |
| GTT, llama-server with the projector, after load / after the smoke requests | 105 495 / 105 519 MiB (v63) | 105 495 / 105 519 MiB | unchanged (the v63 live level before the switch was 105 414) |
| 780M kernel tests (balkon): G2a MUL_MAT_ID n 2048, FA1 8k-64k, G1 large q4_K shapes | | -12 ... -18 %, -13 ... -18 %, -31 ... -38 % | the G1 gains did not transfer shape for shape to gfx1151 (section 2c) |
| hands `:v64k`, one short chat reply | 27.8 t/s (v63k, free GPU) | 24.6 t/s (the deploy smoke right after the restart) | re-measure pending; 40-task coding bench with the probabilistic draft 39/40 (references 38/40) |

**Class: EXACT on the short gate tier, by ruling.** For this release the S0.1 oracle ran its short tier (R133 / R137), not the
full 16.7 h run of v63: the candidate passes (1M production configuration minus the vision projector, one slot, text corpus)
are complete and bit-identical to the v59 reference - 558 logit positions and 81 state points, ordered selected ids, with the
score and dispatch captures compared as extra evidence; the per-lever control arms (production-eligibility rows, not the
candidate's check) were cut and are re-run by the deep tripwire the night after the release, with an automatic rollback to v63
on any difference (result not in this text). On top: each kernel's byte classifier against the v63 build (G1 48/48, G2a 42/42 +
27/27 + 90/90 duplicate-id / hot-route cases, FA1 44/44 hashes), `test-backend-ops` MUL_MAT 1139 / MUL_MAT_ID 1058 /
FLASH_ATTN_EXT 5253, three negative controls, the A/B identity above, and on the live server three fixed greedy requests
identical to v63 (top-5 logprobs equal). Not covered: multi-slot stream offsets, probabilistic drafting (the gate runs greedy),
FA1 under the QSA masks beyond 64k and the 72-byte stride on gfx1151 beyond the short case set. The full oracle and the pair
gate for probabilistic draft sampling return at v67. No claim for other model families.

**Disk cache and slot files: same key and format as v63.** v64 reads v63's entries and v63 reads v64's (13 of 13 same-key
checks); slot files restore warm in both directions. The draft-sampling flag is not part of the key. Switching v63 <-> v64 is
an image change; the v64 composes add the flag, the v63 composes on the v64 image keep greedy drafting.

**Known open items.**

- Probabilistic draft sampling: the live probe around the switch is not an off -> on comparison (both runs had the flag), so
  the flag's own decode effect at T > 0 on the brain is unmeasured; the off -> on comparison and the distribution check (pair
  gate) are scheduled (v67 / the next measurement window). Greedy outputs are unaffected.
- The G1 table on gfx1151: under the perf logger q5_0 10240 x 320 is -19 % and q4_K 2560 x 6144 -8 %, but q4_K 12288 x 2560
  +8 %, 6144 x 2560 +5 % and q6_K 10240 x 2560 +3 % (net -4.1 % on the eligible nodes); a per-rule re-cut is open.
- The S0.2 server rows (128k, the 65k fill steps, decode bands at depth, TTFT at 1M, slot saves) and the 524k profile are not
  measured on v64 yet.
- Hands throughput: the post-restart smoke (24.6 t/s) is below the v63k free-GPU value (27.8); a free-GPU re-measure is pending.

## v63 - 2026-10-21

Images: `vahpetr/llama-cpp-s1-fork:v63` = `:v63k` = `:latest` (one digest,
`sha256:d16b1cdc9139573f07c380f6a5fc6913866348bb6a121593c2e52c76f538a5d4`), revision `8c1a468aa` (`prod-v11-stage9-pkg2`).
In production from 2026-10-21 06:32 (s1 brain) / 06:38 (balkon hands; the local balkon build `flash-next:vulkan-v63k`,
image id `f82fdd2a4091`) to 22:12 / 22:16 (v64).
Rollback: `:v62` / `:v62k` (`e4af0a6ca`). v62 (2026-10-19) was a correctness and diagnostics release with v61's numbers
(patches 0244-0251, BENCHMARKS.md section 0b).

**What it is.** v62 plus patches 0252-0259. No new setting in the composes: every new flag runs at its default
(KNOBS.md, "v63 package").

- MoE matmul (0253, 0254; upstream #25483, #29998): in a MUL_MAT_ID tile, invocations and coopmat warps whose B columns are
  not rows of the expert skip their loads and multiply-adds; the matmul_id shaders read their (expert, tile) list from the
  expert-count prepass instead of a binary search; duplicate expert ids and more than 512 experts are handled. A test-only
  negative-control build option (`GGML_VULKAN_MMID_SKIP_NEGCTL`) proves the gate's test catches a broken skip.
- Disk cache (0255, 0256): recurrent checkpoints at message boundaries (user turn starts, the end of the system prompt, of a tool
  result or of a turn's input) are kept per conversation on top of the depth schedule (`--cache-disk-ckpt-semantic`, default
  16), so a branch after such a boundary restores close to it; a request can pin a checkpoint at N prompt tokens
  (`cache_pin: {"tokens": N}`).
- Slot files (0257, 0258, 0259; upstream #26004, #30574): `/slots` save files carry the slot's context checkpoints and the live
  MTP draft state, so a restored slot drafts warm instead of starting with an empty draft context; checkpoints restored from a
  file never enter the disk cache's deep store or become a pin.
- Speculative decoding (0252, upstream #27694): `--spec-draft-sampling probabilistic` samples the draft at T > 0 and verifies
  by rejection sampling. Off by default (greedy, byte-identical to v62); not measured on the brain yet.

**Measured** (llama-server, 1M profile on s1, one measurement unit on 2026-10-21, single runs; details in BENCHMARKS.md
section 0c):

| metric | v62 | v63 | change |
|---|---|---|---|
| decode after a 32k `/slots` restore | 24.4-33.5 ms/tok, acceptance 0.50-0.62 (cold draft) | 18.3 ms/tok, acceptance 0.94 | warm draft from the file |
| decode band at 1M after a slot restore (same v61-format file) | 18.58 ms/tok, acceptance 0.968 | 18.62 ms/tok, acceptance 0.968 | unchanged |
| `/slots` file at 1M | 15.55 GB | 18.35 GB | +2.6 GiB; save 3.7 s, restore 4.1 s |
| cold fill 0 -> 1M | (v61 probe: 41.8 min) | 41.6 min | |
| fresh 32k prompt | 72.85 s | 69.79 s | -4.2 %, one sample |
| branch after a message boundary, TTFT (semantic checkpoints 16 vs 0) | 4.86 s at 0 | 1.84 s | one branch of three; the other two unchanged |

**Server rows** (llama-server, 1M profile on s1, the S0.2 corpus with the same driver and token positions as the v61 run of
2026-10-18; run of 2026-10-21, single runs, noise +-1.5 %; BENCHMARKS.md section 2b):

| metric | v61 | v63 | change |
|---|---|---|---|
| cold prompt 128k / prefill 72k | 427.4 / 377.2 t/s | 439.8 / 384.1 t/s | +2.9 / +1.8 % |
| 65k fill steps ->262k / ->524k / ->1M; whole fill 0 -> 1M | 483 / 425 / 352 t/s; 0.70 h | 495 / 434 / 358 t/s; 0.69 h | +2.4 / +2.1 / +1.9 %; -1.4 % |
| decode at depth, draft on, worst window, 32k / 262k / 1M | 25.83 / 21.04 / 19.92 ms/tok | 25.80 / 21.05 / 19.93 ms/tok | unchanged (same accepted drafts) |
| TTFT at 1M + 256 new tokens, warm / median after a slot restore | 1 592 / 1 886 ms | 1 449 / 1 728 ms | -9.0 / -8.3 % (prompt side) |
| `/slots` save at 262k / 1M | 3.72 / 14.48 GiB in 1.4 / 3.9 s | 4.71 / 17.09 GiB in 1.7 / 4.0 s | +27 / +18 % bytes (draft state + checkpoints) |
| server GTT at load / over the 1M fill | 105 825 / 105 627 MiB | 105 519 / 105 627 MiB | -306 / 0 MiB |

The prefill rows are the MUL_MAT_ID skip; decode is unchanged by design; the TTFT gain is in the prompt processing on both the
warm and the restored request (the warm draft from the slot file pays after the first token); the slot files grow by what they
now carry.

**Class: EXACT.** The full S0.1 oracle gate on gfx1151 (1M production configuration minus the vision projector, one slot,
text corpus; 16.7 h, finished 2026-10-21 04:36): bit-identical to the v59 reference at 1 502 logit positions and 224 state
points, with the ordered QSA selected ids, at verify widths 1-6 in full and the last 6 positions of widths 7-9, depths 8k ->
1M, and restore / replay histories; the A2b / A3 off arms and a MUL_MAT_ID section as well. Gate: 96 PASS, 4 INFO, 0 FAIL.
GTT peak 106 095 MiB (score-capture pass; v61 106 095, v60 106 357). Not covered: multi-slot stream offsets,
`--spec-draft-sampling probabilistic`. No claim for other model families. Inside the gate the A3 on/off wall ratio on the decode-like tails at 262k is 1.061 (v62: 0.999), but the absolute A3-on tails are unchanged vs v62 (v63/v62 = 1.006; prompt 8k 0.966): the A3-off path got 5.3 % faster with the MUL_MAT_ID skip (0253 / 0254; off arm v63/v62 = 0.947), so the shipped A3-on configuration did not regress, and the release accepted the ratio on that basis.

**Disk cache and slot files: same key as v62.** v63 reads v62's disk-cache entries and v62 reads v63's. v63 restores v62 /
v61-format `/slots` files (with a cold draft, as v62 did). A v62 server restoring a v63 `/slots` file reads the target state
and ignores the new appendices: the result is correct, but the draft starts cold, so decode is slower until it warms up (32k:
33.5 ms/tok, acceptance 0.50). Switching v62 <-> v63 is an image change.

**Known open items.**

- `--spec-draft-sampling probabilistic`: the decode band on the brain at pi's sampling settings is not measured.
- `GET /disk-cache` `deep_semantic` counts newly written semantic checkpoint files only; an existing file re-tagged as
  semantic is not counted.
- The 524k profile was not re-measured on v63.
- Hands throughput (resolved): re-measured on a free 780M at 06:57 with the smoke request (max_tokens 256, temperature 0, thinking off), three runs: 27.8 / 28.0 / 27.7 t/s, draft 184/186, prompt 189-224 ms (v62k smoke 2026-10-19: 25.8 t/s; v61k: 24.3). The deploy smoke's 20.9 t/s ran next to a GPU test container on the same GPU and is not used.

## v61 - 2026-10-17

Images: `vahpetr/llama-cpp-s1-fork:v61` = `:v61k` = `:latest`
(`sha256:46fc98c6f86902b220309e172691758e9806eaf2fc411a640d4ac2f5595b7c63`), revision `7a438f3ba` (`prod-v11-stage8-pkg8`).
Rollback: `:v60` / `:v60k` (`845c6fbcd`). In production on my s1 brain (1M profile) since 2026-10-17 15:31; the balkon hands
run `:v61k`.

**What it is.** v60 plus patches 0241-0243, two levers for the QSA indexer at depth. No new environment in the composes: both
new knobs run at their code default (on).

- QSA score (0241, A2b): the indexer score GEMM's workgroups are numbered key-tile-major, so the query tiles of a prefill
  chunk read each tile of pooled keys back to back instead of streaming the whole store once per query tile. Same per-tile
  program, same bits. Planes of one N tile (decode / verify widths) keep the v60 dispatch, and so does a plane above the
  device's workgroup-count limit.
- QSA select (0243, A3): the top-k select over the score plane computes each cell's key from the score bits and the
  visibility with integer instructions instead of writing and re-reading key rows; the selection scratch loses its key region.
- 0242: a test runs in a private temp dir (two ctest variants raced on shared names); comment wording. No code change.

**Measured** (1M profile, oracle probe, v60 -> v61; details in BENCHMARKS.md section 0a):

| metric | v60 | v61 | change |
|---|---|---|---|
| fill step -> 262k / -> 524k / -> 1M | 474.3 / 413.2 / 328.9 t/s | 491.4 / 433.9 / 357.1 t/s | +3.6 / +5.0 / +8.6 % |
| cold fill 0 -> 1M | 44.0 min | 41.8 min | -5.0 % time |
| cold prompt 8k / 32k / 64k | 548.8 / 458.6 / 357.8 t/s | 550.0 / 457.5 / 357.9 t/s | +0.2 / -0.2 / +0.0 % |
| GTT, probe peak (score-capture pass) / fill-pass peak (no projector) | 106 357 / 104 900 MiB | 106 095 / 104 544 MiB | -262 / -356 MiB |
| GTT, llama-server with the projector, after load | 105 414 MiB | 105 495 MiB (105 519 peak after the smoke requests) | +81 MiB |
| A2b score GEMM per call, 16 shapes (kernel test, gate run) | | median ratio 0.877 (0.673-1.097) | at pools >= 98 304 blocks: -30 to -32 % (f32 rows), -12 to -23 % (f16 mirror, the production form) |
| A3 select, one QSA layer, 64k step -> 262k / -> 524k / -> 1M | 1.834 / 3.757 / 7.616 s | 1.219 / 2.993 / 6.097 s | -33.6 / -20.3 / -20.0 % |
| decode at depth (draft on, worst window) / TTFT at 1M + 256 / server GTT over a 1M fill (llama-server, S0.2 corpus; BENCHMARKS section 2a, 2026-10-18) | not measured on v60; v59: 26.07 / 22.10 / 24.38 ms/tok at 32k / 262k / 1M, 2 452 ms, 105 196 MiB | 25.83 / 21.04 / 19.92 ms/tok; 1 886 ms median (1 592 warm); 105 825 MiB at load, 105 627 over the fill | vs v59: -0.9 / -4.8 / -18.3 %; -23 %; +434 MiB |

**Class: EXACT.** The package as a whole, gated on gfx1151 like v60 (1M production configuration minus the vision projector,
one slot, S0.1 oracle corpus, text): bit-identical to the v59 reference at 1 502 logit positions and 224 state points, with
the ordered QSA selected ids, at verify widths 1-6 in full and the last 6 positions of widths 7-9, depths 8k -> 1M, and
restore / replay histories. With A2b off, A2b forced onto its fallback and A3 off, the short case set matched the same
reference. Gate: 91 PASS, 4 INFO, 0 FAIL. Not covered: multi-slot stream offsets. No claim for other model families.

**Disk cache: unchanged.** Same state format and key as v60. On my box v61 reads v60's entries and v60 reads v61's (same-key
check, 13 of 13 required checks); the read-only adopt of the v59 namespace stays as it was. Switching v60 <-> v61 is an image
change.

**Known open items.**

- Decode bands, TTFT at 1M and the server GTT peak over a 1M fill, llama-server on the S0.2 corpus (2026-10-18): 25.83 / 21.04 / 19.92 ms/tok at 32k / 262k / 1M (v59 26.07 / 22.10 / 24.38), TTFT at 1M + 256 new tokens 1 886 ms median (1 592 warm; v59 2 452 / 2 054), server GTT over the fill 105 627 MiB (v59 105 193); BENCHMARKS.md section 2a, 2026-10-18; v60 was not measured this way.
- The 524k profile was not re-measured on v61.

## v60 - 2026-10-16

Images: `vahpetr/llama-cpp-s1-fork:v60` = `:v60k` = `:latest`
(`sha256:794113766af1129c65e9541bcc4b9b3bac4d378fa38b38cd4d7c5f00fbf7a0d1`), revision `845c6fbcd` (`prod-v11-stage8-pkg7`).
Rollback: `:v59` / `:v59k` (`a1b6c324e`). In production on my s1 brain (1M profile) and the balkon hands since 2026-10-16.

**What it is.** v59 plus patches 0227-0240, a package aimed at prefill on gfx1151, mostly at depth. No new environment in the
composes: every new knob runs at its code default.

- Prefill kernels: hc_mix tail fusions (0227, B3), the GDN conv input without a concat pass (0228, B4), mul_mat_id with a
  one-time f16 pre-cast of B and a BM 128 x BN 64 tile on AMD coopmat1 (0231 / 0232, B1), a 16 x 16 tile for the m = 4 hc inject
  matmul (0234, B4c).
- QSA at depth: the indexer score as one op with ReLU, head sum and bias in the GEMM's store (0235, A1); an f16 mirror of the
  pooled-key store for that GEMM (0240, A2a, +779 MiB at 1M).
- Decode-side: the mat-vec unpack hoisted out of the column loop for verify widths 2-8 (0236, C1); a device-resident KQ
  visibility mirror updated with changed cells only (0237, C2).
- Defect fix (0239): restoring a conversation with images left the indexer cache's M-RoPE x / y at 0. v60 restores them and
  marks its saved state (`IXY1`).
- Diagnostics, off by default: QSA id / score dumps and a graph-cache trace (0229), a matmul dispatch log (0230), a
  per-round speculative ids trace (0238).
- Built, off by default, not in the gated configuration: a dense tile plan for the hc shapes (0233, B2); 2.6x slower on gfx1151
  for one of its shapes. Its op tests are bit-identical; it is not EXACT-claimed.

**Measured** (1M profile, oracle probe, v59 -> v60; details in BENCHMARKS.md section 0):

| metric | v59 | v60 | change |
|---|---|---|---|
| fill step -> 262k / -> 524k / -> 1M | 368.0 / 284.1 / 194.9 t/s | 474.3 / 413.2 / 328.9 t/s | +28.9 / +45.4 / +68.7 % |
| cold fill 0 -> 1M | 64.3 min | 44.0 min | -31.6 % time |
| cold prompt 8k / 32k / 64k | 503.4 / 423.0 / 335.4 t/s | 548.8 / 458.6 / 357.8 t/s | +9.0 / +8.4 / +6.7 % |
| GTT, probe fill-pass peak (no projector) | 104 078 MiB | 104 900 MiB | +822 MiB |
| GTT, llama-server with the projector | 104 591 MiB (v59 live, just before the switch) | 105 414 MiB after load, 106 032 after the first requests | |
| decode at depth, TTFT at 1M, server GTT peak over a fill | measured on v59 | not measured on v60 | measured on v61: 25.83 / 21.04 / 19.92 ms/tok at 32k / 262k / 1M (v59 26.07 / 22.10 / 24.38), TTFT at 1M + 256 new tokens 1 886 ms median (1 592 warm; v59 2 452 / 2 054), server GTT over the fill 105 627 MiB (v59 105 193); BENCHMARKS.md section 2a, 2026-10-18 |

**Class: EXACT.** The package as a whole, gated on gfx1151 with the 1M production configuration minus the vision projector,
one slot, on my S0.1 oracle corpus (text): v60 matched v59 bit for bit, with logits at 1 502 positions, 224 saved states and the
ordered QSA selected ids, at verify widths 1-6 in full and the last 6 positions of widths 7-9, depths 8k -> 1M, and restore /
replay histories. Not covered: multi-slot stream offsets. No claim for other model families, and not a claim of equality with
upstream. The one change in behaviour is 0239's, and only for restored image
conversations, which were wrong before.

**Disk cache: new key, the old namespace is adopted read-only.** The state format changed, so v60 writes to a new namespace.
With `--cache-disk-adopt <v59 key>` it reads the v59 namespace without writing it: text conversations load, conversations
with images are refused (re-prefilled once), raw v59 slot files are refused; `LLAMA_LEGACY_UNMARKED=0` refuses all old state.
A rollback to v59 finds its namespace untouched; v59 does not see what v60 saved. Procedure: [DEPLOY.md](DEPLOY.md), section 4.

**Other models.** The B1 / B4 / B4c / C1 kernels, the graph cache and the disk cache are not specific to Qwen3.8. A first A/B
on Qwen3.6-35B-A3B (780M, MTP off) against upstream b10689: prefill +2.6 / +6.4 / +16.5 % at 8k / 32k / 72k, decode at depth
-13.4 / -12.6 / -5.9 % ms/token at 4k / 32k / 128k, greedy tokens identical, logprobs not (no EXACT claim for that model). The
per-lever attribution is pending. BENCHMARKS.md section 8.

**Known open items.**

- Decode bands, TTFT at 1M and the server GTT peak over a 1M fill on v60: not measured (v60 was live for one day); on v61 25.83 / 21.04 / 19.92 ms/tok at 32k / 262k / 1M (v59 26.07 / 22.10 / 24.38), TTFT at 1M + 256 new tokens 1 886 ms median (1 592 warm; v59 2 452 / 2 054), server GTT over the fill 105 627 MiB (v59 105 193); BENCHMARKS.md section 2a, 2026-10-18.
- In the probe, state saves took about 1.8x as long as on v59 (21.4 s vs 11.6 s for the 15.6 GB state at 1M); loads and sizes
  were unchanged. Cause not established; a server slot-save timing (2026-10-18, v61 vs v59, same bytes): 262k 0.5 -> 1.4 s, 1M 3.2 -> 3.9 s, so the server shows a smaller form of it.
- The 524k profile was not re-measured on v60.
