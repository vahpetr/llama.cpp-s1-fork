# Release notes

## v74

Fork revision `e5f793eae` (`ui-agentic-actions` = v73 `7597206a9` + six webui commits, patches 0469–0474). Images
`vahpetr/llama-cpp-s1-fork:v74` (brain) and `:v74k` (hands), one image; the rollback is `:v73` / `:v73k`. Webui only: the server,
ggml and every shared library are byte-identical to v73 (only the embedded UI changed), the disk-cache key is unchanged, so every
number in BENCHMARKS.md is v73's.

- **Agentic bubbles:** Fork, Continue and Copy act on the flow's last assistant message (the root holds only reasoning and tool calls).
- **Continue after a cut reply:** a reply that ends mid-sentence offers Continue (judged on the last line only); Continue joins the
  continuation without repeating the prefilled text.

## v73 (server; unchanged in v74)

Fork revision `7597206a9` (`prod-v11-stage9-pkg13`). Images `vahpetr/llama-cpp-s1-fork:v73` (brain) and `:v73k` (hands), one
image; the rollback was `:v72` / `:v72k`. Measurements: [BENCHMARKS.md](BENCHMARKS.md). Deploy, upgrade and rollback:
[DEPLOY.md](DEPLOY.md).

New in v73 (all code defaults, no compose change; on gfx1151 against the published v72 numbers: step 262k +3.0 %, step 1M
+0.5 %, fill 1M −0.9 %, cold 32k prefill +0.4 % (AB64, same session), cold 128k −2.3 %, decode with MTP −0.2 % / +0.4 % / +0.6 %
at 32k / 262k / 1M, TTFT after a 1M restore −7.5 %; hands chat 28.4 t/s against 29.4 on v72k, one request each; noise band
about ±1.5 %):

- **D3 decode fusions** (`GGML_VK_DECODE_FUSE`, default on): at ≤ 32 rows (tokens × sequences) the shared-expert tail
  (`shexp_tail`: MUL by the gate + ADD into the MoE output), the shared-expert GLU with sigmoid gate (`shexp_glu_sig`) and the
  GDN q/k norm + gate (`gdn_qk_gate`) each run as one dispatch. `=0` gives the v72 dispatches. Exact-listed in CACHE_ENV_EXACT,
  so it is not in the disk-cache key. Tests: `DECODE_FUSE_EXACT` bitwise cases, plus a mutant build option
  (`GGML_VULKAN_DECODE_FUSE_MUTANT`, tests only) whose cases must fail.
- **Upstream ports:** #30283 (int8 coopmat1 MMQ A-prefetch clamp; int8 MMQ stays off in the composes), #31350 (speculative
  checkpoint for drafts longer than `n_rs_seq`), #30213 (rejection-sampling test), #31035 (DRY sequence-breaker cap 64), #31310
  (gguf-py `np_roundf` sign).

The new path prints one startup line (`decode_fuse: on (GGML_VK_DECODE_FUSE, default on)`); [KNOBS.md](KNOBS.md) lists it.

Exactness: on gfx1151 v73 passed the D-gate as EXACT against the v72 reference at 262k and 1M, with every new knob off as a
second arm and a negative control that the gate caught; the oracle manifest; the AB64 greedy identity check; the same-key
disk-cache gate (13 of 13). `DECODE_FUSE_EXACT` passed 62 of 62 cases on gfx1151 and on the 780M (gfx1103); the #30283 int8
path is bitwise equal to v72. v73 shares its disk-cache key with v60–v72.

Known limitations: the 524k profile has no v73 measurements. The full list is in [README.md](../../README.md#known-limitations).
