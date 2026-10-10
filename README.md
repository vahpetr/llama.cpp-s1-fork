# llama.cpp for Strix Halo: Qwen3.8-Flash-Next at 1M context on Vulkan

This is my fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). I tuned it for one machine: an AMD Ryzen AI Max+ 395
("Strix Halo", Radeon 8060S iGPU, gfx1151) with 128 GB of unified memory, running on **Vulkan (RADV), not ROCm**. A second, smaller
box with a Ryzen 9 7940HS and a Radeon 780M (RDNA3, 96 GB) serves as the "hands" for parallel coding agents.

The model is **Qwen3.8-Flash-Next** (`qwen4exp`): a 177B-parameter MoE (125B weights + a 51B n-gram embedding table, ~6B active
per token, MTP head, vision). Stock llama.cpp runs it at ~24-27 t/s with a dense 262k window and no working speculative
decoding. This fork runs it with a **1 048 576-token context in one slot at 35-60 t/s decode**, in 102.7 GiB of GPU memory.

Everything here was measured on my two boxes. Numbers, dates and image versions: [docs/s1/BENCHMARKS.md](docs/s1/BENCHMARKS.md).

## What it gives

Same box, one conversation, one slot for the fork's current column (the fork columns use the i1-Q4_K_M weights):

| | stock llama.cpp (Aug 2026) | this fork, dense (Sep 9) | this fork now (v53-v55) |
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
  list per conversation (tool changes do not invalidate the cache). The permanent surface is a thin `used / n_ctx` meter that
  tints and offers "compact?" from 80 % full (setting); its popover (hover, click pins) holds prefill / decode t/s, cache share,
  MTP %, the chat's compute time, the reserve, links to the previous / next compacted chat, "Compact now" and the settings;
  `/compact` in the composer. Timing chips collapse to one muted line per reply (details on hover; "Show timing chips" hides
  them). A continuation drops the prefill the server echoes back (whitespace-normalised match; a diverging echo is joined
  verbatim and the message marked `echoMismatch`).
- **Telemetry.** The stop type of every reply at INFO (`stop=eos|limit|word|cancel`), per-round speculative timings, `kqvis-prof`
  lines, expert-occupancy stats, `disk_cache_*` in `/metrics`, `GET /disk-cache`.

The full list (191 numbered changes, including upstream back-ports): [docs/s1/PATCHES.md](docs/s1/PATCHES.md).

## What is hardware-specific and what is not

- **Vulkan / RDNA3 and RDNA3.5:** the new kernels (QSA top-k radix select, indexed and metadata flash attention, KQ mask, GDN and
  hyper-connection fusions), the MoE `mul_mat_id` changes (row-id hoisting up to 512 experts, compact grid, tile choice), flash
  attention coopmat1 tiles with RDNA3 signed-zero handling, UMA readback. They should help any RADV GPU with cooperative
  matrices on a large MoE; I measured them on gfx1151 and gfx1103 only.
- **Generic (libllama / server / webui, any backend):** the QSA model logic and memory, graph and plan cache, disk prompt cache,
  GTT admission (reads an AMD sysfs file; the path is a knob), the webui, the speculative-decoding fixes, the logs.
- **Model-specific:** QSA and the MTP work apply to `qwen4exp` only. Other models run as on upstream.
- **Not tested:** ROCm/HIP and NVIDIA. The new ops have CPU and Vulkan implementations only.

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
The published images (`vahpetr/llama-cpp-s1-fork:v55` = `:latest`, `:v55k`) are built from my `prod-v11-stage7` revision
(v55 = `af7485f3c`; `:v54` / `:v54k` stay as the rollback); this `s1-public` branch only adds docs and deploy files on top.
My production draft head is a vocabulary-pruned variant built from my own traffic, so the public setup uses dzannotti's
full-vocabulary head, and the decode numbers above are an upper bound for it.

First start on s1: GTT after load ~105 000 MiB (1M) or ~94 500 MiB (524k), flat afterwards; the first long prompt fills at
~550 t/s falling to ~190 t/s near 1M.

## Known limitations

- **No context shift.** The model has recurrent (Gated-DeltaNet) layers. The client must compact before the window is full;
  the bundled webui does. Replies near the end of the window are capped by the client to the room that is left.
- **Accuracy at 1M is not gated.** One long-context probe scored 76.8 % at 1M (vs ~77 % at 524k). Use the 1M window for recall
  over long inputs and check answers that depend on it.
- **Two slots give no extra throughput** on these boxes (-15 % total, measured). A second slot is for concurrency only.
- **After a checkpoint rewind** (disk cache, branch, copy) the continuation can differ from a fresh prefill at near-tie tokens:
  the same math in a different batch composition. The state itself is verified, the text is not guaranteed byte-identical.
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
