# Release notes

## v72

Fork revision `608af6ea8` (`prod-v11-stage9-pkg12`). Images `vahpetr/llama-cpp-s1-fork:v72` (brain) and `:v72k` (hands), one
image; the rollback is `:v71` / `:v71k`. Measurements: [BENCHMARKS.md](BENCHMARKS.md). Deploy, upgrade and rollback:
[DEPLOY.md](DEPLOY.md).

New in v72 (all code defaults, no compose change; on gfx1151 against v71: step 1M +4.6 %, step 262k +3.1 %, fill 1M −3.3 %,
cold 32k prefill +2.9 %, decode −0.4…−0.7 %):

- **hc-mix tail in one shader** (`GGML_VK_HC_TAIL_FUSE`, default on): SIGMOID_MUL + MULTI_ADD_SCALE of the hyper-connection
  mix run as one dispatch; a destination that overlaps a source falls back to the two ops.
- **HC_COMBINE + RMS_NORM_MUL in one dispatch** (`GGML_VK_ADD_RMS_ROWS`, default on).
- **hc_inject merged into hc_down** above `LLAMA_QWEN4EXP_MERGE_MAX_N` where the Vulkan dispatch plan gives bit-identical rows
  (`LLAMA_QWEN4EXP_MERGE_PLAN`, default on gfx1151, off elsewhere).
- **Pipeline prewarm** (`GGML_VK_PREWARM_SHAPES`, default on): the first graphs compile the other decode-width and prefill-tile
  pipelines instead of compiling them lazily mid-request.
- Code only, off by default: `LLAMA_SPEC_OVERLAP=2` (the composes keep `=1`) and the f16 indexer score for decode rows
  (`LLAMA_QSA_IDX_F16_DECODE`, not exact, in the disk-cache key).

Each new path prints one startup line (`hc_tail_fuse: on`, `add_rms_rows: on`, `qwen4exp merge plan: on|off`,
`prewarm_shapes: on`, `qsa_idx_f16_decode: off`); [KNOBS.md](KNOBS.md) lists them.

Exactness: on gfx1151 v72 passed the D-gate as EXACT against the v71 reference at 262k and 1M, with every new knob off as a
second arm (also EXACT) and a negative control that the gate caught; the oracle manifest (12 of 12); the AB64 greedy identity
check; the same-key disk-cache gate (13 of 13). On the 780M (gfx1103) the hands test binary gave identical logits, ids and states
against v71 in 5 of 5 arms. v72 shares its disk-cache key with v60–v71.

Known limitations: the 524k profile has no v72 measurements. The full list is in [README.md](../../README.md#known-limitations).
