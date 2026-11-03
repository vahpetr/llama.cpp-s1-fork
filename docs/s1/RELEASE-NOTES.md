# Release notes

## v71

Fork revision `8bcde80da` (`prod-v11-stage9-pkg11`). Brain image `vahpetr/llama-cpp-s1-fork:v71`; the hands run `:v70k`; the
brain rollback is `:v70`. Measurements: [BENCHMARKS.md](BENCHMARKS.md). Deploy, upgrade and rollback: [DEPLOY.md](DEPLOY.md).

What the fork is at v71:

- **1M context in one slot** on the Strix Halo (gfx1151, 128 GB, GTT 108 GiB): YaRN x4 over the native 262 144 tokens, KV q8_0,
  every compute buffer reserved at load, GTT admission for prefill chunks.
- **QSA sparse attention** on the 12 full-attention layers from 65 536 tokens: the model's indexer selects 2 051 cells per
  query; the top-k select is single-pass with the v3 block form on gfx1151 (`GGML_VK_QSA_TOPK_V3`, v3 from 2 rows per
  selection, single-pass below), the block bias is folded into the score epilogue (`GGML_VK_QSA_BIAS_FOLD`), and prefill runs
  indexed flash attention.
- **GDN commit-by-replay** on gfx1151 (`LLAMA_GDN_REPLAY`, default on there, off elsewhere): the recurrent state after a
  speculative verify is committed by replaying the accepted tokens; the snapshot admission clamps the per-heap free value at 0
  and uses the largest heap on UMA devices. The startup line `gdn_replay: on|off (<reason>)` names the reason.
- **MTP speculative decoding** with the model's MTP head on Vulkan, probabilistic draft sampling at temperature > 0, and the
  trimmed host phase per round (`LLAMA_SPEC_OVERLAP=1` in the composes).
- **Disk prompt cache** with recurrent checkpoints, asynchronous saves with 1 024 MiB of host staging (`--cache-disk-async`)
  and incremental re-saves of unchanged KV blocks (`--cache-disk-incr`).
- **Vulkan kernels** for AMD coopmat1 (MoE and dense GEMM tiles and loaders, flash-attention layouts, fusions), measured on
  gfx1151 and gfx1103.
- **webui** for long agent sessions (compaction, continuation, per-chat reasoning levels, timing per reply).

Exactness: on gfx1151 the v71 package passed the D-gate as EXACT against the v70 reference, the oracle manifest (12 of 12),
the AB64 greedy identity check and the same-key disk-cache gate (13 of 13 checks). v71 shares its disk-cache key with v60–v70.

Known limitations: hands build :v71k is not published — the v71 code crashes once at warmup on the Radeon 780M (gfx1103,
vk::DeviceLostError); hands run :v70k until the fix. The 524k profile has no v71 measurements. The full list is in
[README.md](../../README.md#known-limitations).
