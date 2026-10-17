# Release notes

One entry per published image. Older versions (v52-v59) are summarised in the lineage table of
[BENCHMARKS.md](BENCHMARKS.md), section 0; every change is a numbered row in [PATCHES.md](PATCHES.md).

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
