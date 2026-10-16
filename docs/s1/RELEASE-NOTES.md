# Release notes

One entry per published image. Older versions (v52-v59) are summarised in the lineage table of
[BENCHMARKS.md](BENCHMARKS.md), section 0; every change is a numbered row in [PATCHES.md](PATCHES.md).

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
- In the tree, off: a dense tile plan for the hc shapes (0233, B2): 2.6x slower on gfx1151 for one of its shapes.

**Measured** (1M profile, oracle probe, v59 -> v60; details in BENCHMARKS.md section 0):

| metric | v59 | v60 | change |
|---|---|---|---|
| fill step -> 262k / -> 524k / -> 1M | 368.0 / 284.1 / 194.9 t/s | 474.3 / 413.2 / 328.9 t/s | +28.9 / +45.4 / +68.7 % |
| cold fill 0 -> 1M | 64.3 min | 44.0 min | -31.6 % time |
| cold prompt 8k / 32k / 64k | 503.4 / 423.0 / 335.4 t/s | 548.8 / 458.6 / 357.8 t/s | +9.0 / +8.4 / +6.7 % |
| GTT | 104 078 MiB (probe) / 104 591 MiB (server after load) | 104 900 / 105 414 MiB | +822 / +823 MiB |
| decode at depth, TTFT at 1M, server GTT peak over a fill | measured on v59 | pending | |

**Class: EXACT.** On my S0.1 oracle v60 matched v59 bit for bit: logits at 1 502 positions, 224 saved states, and the QSA
selected ids, across verify widths 1-9, depths 8k -> 1M, and restore / replay histories, on gfx1151 with the production
configuration. Not covered: multi-slot stream offsets. The one change in behaviour is 0239's, and only for restored image
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

- Decode bands, TTFT at 1M and the server GTT peak over a 1M fill on v60: pending.
- In the probe, state saves took about 1.8x as long as on v59 (21.4 s vs 11.6 s for the 15.6 GB state at 1M); loads and sizes
  were unchanged. Cause not established; a server slot-save timing is pending.
- The 524k profile was not re-measured on v60.
