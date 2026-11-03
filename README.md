# llama.cpp for Strix Halo: Qwen3.8-Flash-Next at 1M context on Vulkan

This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for one machine: an AMD Ryzen AI Max+ 395 ("Strix Halo",
Radeon 8060S iGPU, gfx1151, 128 GB unified memory) on Vulkan (RADV), not ROCm. The model is Qwen3.8-Flash-Next (`qwen4exp`), a
177B-parameter MoE with about 6B active parameters per token, an MTP head and vision. Stock llama.cpp runs this model with a
dense window of at most 262 144 tokens and no working speculative decoding. This fork runs it with a 1 048 576-token context in
one slot, with sparse attention, MTP drafting and a disk prompt cache. This README describes release v71 (fork revision
`8bcde80da`, `prod-v11-stage9-pkg11`).

## Measured on v71, by context depth

| depth | prefill or step, t/s | decode with MTP, t/s (ms/token) | draft acceptance | TTFT after slot restore |
|---|---:|---:|---:|---:|
| 8k | 789.1 (cold prompt) | — | — | — |
| 32k | 623.5 (cold prompt) | 41.6 (24.01) | 0.817 | — |
| 128k | 580.5 (cold prompt) | — | — | — |
| 262k | 676.9 (step) | 51.7 (19.33) | 0.800 | — |
| 524k | 648.2 (step) | — | — | — |
| 1M | 585.3 (step) | 56.5 (17.69) | 0.877 | 1 285 ms |

GTT after load: 104 580 MiB (admission bar 109 216 MiB). Cold fill 0 → 1 046 528 tokens: 27.76 min (sum of 17 step walls).
"—" = not measured on v71. Per-run tables and the method: [docs/s1/BENCHMARKS.md](docs/s1/BENCHMARKS.md).

Measurement setup:

- Host s1: Ryzen AI Max+ 395, Radeon 8060S (gfx1151), 128 GB LPDDR5x, GTT 108 GiB (`ttm.pages_limit=28311552`), GPU DPM level
  `high` while busy, kernel 7.0.0-38-generic, Mesa 26.0.8 RADV and Vulkan loader 1.4.341 inside the image.
- Image `flash-next:vulkan-v71` (rev `8bcde80da`), llama-server, the 1M compose of `deploy/s1/` without `--cache-disk*`.
- Weights: Qwen3.8-Flash-Next-Uncensored i1-Q4_K_M (mradermacher, 110.97 GiB) + mmproj f16. Draft: the MTP head pruned to a
  65 536-token vocabulary (not published); the public setup uses the full-vocabulary head, so the decode column is an upper
  bound for it.
- `-c 1048576 --parallel 1`, KV q8_0, `-b 2048 -ub 2048`, flash attention on, YaRN x4 over 262 144, QSA ratio 4 from 65 536
  tokens (dense below), MTP `n-max 5`, `p-min 0.7`, probabilistic draft sampling; temp 0.7, top-p 0.8, top-k 20, reasoning off.
- Cold prompt = one prompt with `cache_prompt` false (8k: one deploy smoke prompt; 32k: 2 reps; 128k: the first request).
- Step = the 65 536-token append that reaches the depth; t/s of that step. Fill = sum of the step walls.
- Decode = ms per token, worst of 3 windows of 128 tokens, continuous history, draft on, fixed text corpus. Decode speed depends
  on draft acceptance, which depends on the text.
- TTFT = 256 new tokens after restoring a saved 1M conversation, mean of 2 restored reps.
- Single runs; noise band about ±1.5 %.
- No stock llama.cpp or other-engine numbers were measured on v71; the only cross-engine number is the dated v60 stock comparison.

## Against upstream and other Strix Halo engines

Their numbers are their published ones (date, quant, context as they state); nothing below was re-measured on v71.

| engine | backend | max context on a 128 GB box | prefill vs ours | decode vs ours | source / date |
|---|---|---|---|---|---|
| stock llama.cpp b10689, same GGUF | Vulkan | 262 144 (dense); no working MTP for this model | ours (v60) +46 % on a cold 32k prompt | ours (v60) +49 % at 261k, MTP off on both | measured by us, 2026-10-16 (last stock baseline) |
| Strata (Niko1221/Strata, open) | HIP / CUDA, Strix Halo port, int8 KV | ≤ 131k | ×1.8–2 ours at 8k–128k (≈ 1 160–1 320 t/s flat, 4-bit) | parity (IQ4_XS 53–54 t/s at 8k, 51 at 128k) | their Linux Strix Halo numbers, 2026-10 |
| halogen-flash-server (closed) | HIP | below 1M (exact limit not checked) | ≈ ×3 ours | parity | their numbers, as of 2026-10-11 |
| Nathan/halo-box | Vulkan fork | not checked | ≈ ×1.7–1.9 ours at ≤ 32k | not checked | read 2026-10-28, before its GEMM loaders were ported here in v68 (+12 % here); not re-compared |

## What this fork is for

- 1M-token context in one slot on a 128 GB Strix Halo; the engines above stop at 128k–262k.
- QSA sparse attention with a bit-exact top-k: the model's own indexer, verified against the dense path by an oracle gate.
- Decode that does not slow down with depth: 1M decodes at least as fast as 32k (table above).
- A disk prompt cache that resumes a 1M conversation in the TTFT above instead of a full re-prefill (fill time above).
- MTP speculative decoding on Vulkan.
- Vulkan/RADV only: stock Mesa, no ROCm.

Short-context prefill is below Strata and halogen (×0.5–0.7 of theirs); the fork optimises depth, not the 8k–32k range.

## What is in it

- QSA sparse attention: the indexer picks 2 051 cells per query in the 12 full-attention layers; fused top-k select (v3 on
  gfx1151), indexed flash attention, persistent pooled-key cache, incremental census; dense below 65 536 tokens.
- GDN commit-by-replay: after a speculative verify the Gated-DeltaNet state is committed by replaying the accepted tokens
  instead of keeping one state plane per draft position; default on gfx1151, off elsewhere.
- MTP speculative decoding: the MTP head drafts on Vulkan, sparse at depth; probabilistic draft sampling; trimmed host phase
  per round (`LLAMA_SPEC_OVERLAP=1`).
- Graph and allocation-plan cache between decode steps.
- Flat memory: compute buffers reserved at load, KQ mask from cell metadata, GTT admission that delays a prefill chunk instead
  of overrunning GTT.
- Disk prompt cache with recurrent checkpoints, asynchronous and incremental saves, restore instead of re-prefill.
- Vulkan kernels for RDNA3/RDNA3.5 coopmat1: MoE `mul_mat_id` tiles and loaders, dense GEMM tile table, flash-attention
  layouts, QSA score epilogue, hyper-connection and GDN fusions.
- Hands profile for the Radeon 780M (gfx1103): five slots, IQ3_S experts with Q8_0 hyper-connection and head weights.
- webui for long agent sessions: auto-compaction, continuation from the stop point, per-chat reasoning levels, timing per reply.

Numbered changes, including upstream back-ports: [docs/s1/PATCHES.md](docs/s1/PATCHES.md). Flags and environment knobs:
[docs/s1/KNOBS.md](docs/s1/KNOBS.md). QSA, GDN replay and the MTP work are `qwen4exp`-only; the graph cache, disk cache, GTT
admission and webui are backend-generic; the Vulkan kernels were measured on gfx1151 and gfx1103 only. ROCm/HIP and NVIDIA are
not tested.

## Running it

```sh
git clone https://github.com/vahpetr/llama.cpp-s1-fork && cd llama.cpp-s1-fork    # branch s1-public
deploy/s1/setup-brain.sh --dry-run && deploy/s1/setup-brain.sh           # Strix Halo: 1M brain (BRAIN_CTX=524k for 524k)
deploy/balkon/setup-hands.sh --dry-run && deploy/balkon/setup-hands.sh   # Radeon 780M: 5-slot hands
```

Images on Docker Hub, `vahpetr/llama-cpp-s1-fork`: `:v71` is the brain image of this release; the hands run `:v70k` (see Known
limitations); `:v70` is the brain rollback. Host settings: [docs/s1/HOST-TUNING.md](docs/s1/HOST-TUNING.md). Upgrade, rollback
and disk-cache keys: [docs/s1/DEPLOY.md](docs/s1/DEPLOY.md). Compose files and flags: [deploy/README.md](deploy/README.md).

## For AI agents

1. Entry points: `deploy/s1/setup-brain.sh` (brain), `deploy/balkon/setup-hands.sh` (hands), `deploy/setup-common.sh` (shared
   checks), `deploy/README.md` (what the scripts do, by-hand route).
2. Compose files: `deploy/s1/docker-compose.brain-1m.yml` (1M), `deploy/s1/docker-compose.brain-524k.yml` (524k),
   `deploy/balkon/docker-compose.hands.yml` (hands).
3. Run every script with `--dry-run` first: it prints each check, file and size and changes nothing.
4. The scripts check docker and the compose plugin, the Hugging Face CLI, free disk, the GTT limit and the Vulkan device; they
   download the weights, check sha256, (hands) re-encode with `tools/s1/gguf_hc_q8.py`, pull the image, start the compose and
   smoke-test it.
5. Hardware for the 1M brain: Strix Halo with 128 GB and GTT 108 GiB (`ttm.pages_limit=28311552`). For the hands: a Radeon
   780M-class box with 96 GB and GTT 78 GiB.
6. Weights and sha256: [docs/s1/WEIGHTS.md](docs/s1/WEIGHTS.md). About 160 GB free disk for the brain, 150 GB for the hands.
7. Docker Hub tags: `:v71` brain, `:v70k` hands, `:v70` brain rollback. `BUILD=1` builds the image from this tree instead.
8. Verify: `curl -sf http://127.0.0.1:8000/health`, `/props` shows `n_ctx` 1048576, then one chat request
   ([docs/s1/DEPLOY.md](docs/s1/DEPLOY.md), "Smoke checks").
9. Flags and environment knobs with defaults and classes: [docs/s1/KNOBS.md](docs/s1/KNOBS.md).
10. Code map, tags, test contract, exactness classes: [AGENTS.md](AGENTS.md).

## Known limitations

- hands build :v71k is not published — the v71 code crashes once at warmup on the Radeon 780M (gfx1103, vk::DeviceLostError); hands run :v70k until the fix
- No context shift: the recurrent layers forbid it; the client must compact before the window is full (the bundled webui does).
- Accuracy at 1M is not gated. Check answers that depend on deep recall.
- Two slots give no extra total throughput on these boxes; a second slot is for concurrency only.
- After a checkpoint rewind the continuation can differ from a fresh prefill at near-tie tokens (same math, different batch
  shape). The restored state is verified; the text is not guaranteed byte-identical.
- Disk-cache entries are tied to a cache key. v71 shares its key with v60–v70; [docs/s1/DEPLOY.md](docs/s1/DEPLOY.md) has the details.
- One model load per GPU at a time: a build or a second load next to a loaded 1M model has run the 128 GB box out of memory.
- Single-user setups; no multi-tenant hardening beyond upstream's.
- The 524k profile and the hands profile have no v71 measurements.

## Upstream and license

Based on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) via drluoto's `strix-halo-flash-next` branch (b10664) and
upstream PR #27742 (Qwen3.8-Flash-Next), plus selected upstream back-ports. Upstream's documentation is in `docs/`. One change is
open upstream as PR #28118.

MIT License, Copyright (c) 2023-2026 The ggml authors, see [LICENSE](LICENSE); the fork's changes use the same license.
Third-party components keep their licenses (`licenses/` and upstream's acknowledgements).
