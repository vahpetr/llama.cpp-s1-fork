# Release notes

## v75

Fork revision `4a55d9173` (`prod-v11-stage9-pkg15` = v74 `e5f793eae` + twelve commits, patches 0475–0486). Images
`vahpetr/llama-cpp-s1-fork:v75` (brain) and `:v75k` (hands), one image; the rollback is `:v74` / `:v74k`. Released 2026-11-05
(the previous release, v74, was 2026-11-05 01:30). libggml* and libllama.so are byte-identical to v74, the disk-cache key is
unchanged, so every speed and accuracy number is v74's (measured on v73, [BENCHMARKS.md](BENCHMARKS.md)). Hands deploy smoke
on `:v75k`: 28.6 t/s on a short chat, first request after start.

- **Reasoning boundary as generated** (0475, 0476): the GGUF template renders a past reply's reasoning as trimmed text plus
  `\n</think>`; the model often generates `\n\n</think>`. The rendered history then differed from the cached tokens, and the
  next request re-prefilled from that point (about 35k tokens after a long reasoning reply). The prefill continuation in
  `common/chat.cpp` and two template files, `models/templates/qwen3.8-flash-next-preserve-ws.jinja` (brain) and
  `qwen3.8-flash-next-gsq-preserve-ws.jinja` (hands), keep a trailing newline as generated (lstrip; a reasoning that ends in
  `\n` is kept, any other is trimmed plus one `\n`). For trimmed or empty reasoning the output is byte-identical to the GGUF
  template. The composes mount `models/templates/` and pass `--chat-template-file`.
- **Finished-reply save** (0477): when the next request diverges inside a finished reply (EOS or stop word, at least 1024
  tokens lost), the disk cache now saves that reply's tail first. This is a temporary rule; it misses a divergence at the
  first generated token. The stop log line appends `eog_id=N` on EOS stops.
- **Fingerprint-neutral flags** (0478, 0483): `--chat-template`, `--chat-template-file` and `--drift-guard` (and their
  `LLAMA_ARG_*` forms) are not part of the disk-cache fingerprint; changing the template keeps the stored conversations.
- **Drift guard, opt-in** (0479–0483, webui 0485–0486): `--drift-guard stop` ends a reply when the model generates a second
  special `</think>` in the content phase. The token is not emitted; the response carries `finish_reason` `stop` and
  `stop_type` `think_reenter`; a speculative draft past the trigger is trimmed and the MTP carry is re-picked for the kept
  prefix. The webui records `stop_type` and offers one bounded content-phase continuation. Default `off`; the composes set
  `off`. Tests: `test-drift-guard`, `test-mtp-carrier` case 5, `test_drift_guard.py` (0482).
- **Server README** (0484): `tools/server/README.md` documents the limits of 0475–0478.

Release checks passed on s1 and the 780M: T = 0 identity against v74 on six prompts with MTP, a live guard trigger with a speculative
trim, the same-key disk-cache check, the brain smoke with the template mount, the hands smoke with the GSQ template.

## Engine (ggml and libllama, unchanged since v73)

The engine came with v73 (`7597206a9`). Measurements: [BENCHMARKS.md](BENCHMARKS.md). Deploy, upgrade and rollback:
[DEPLOY.md](DEPLOY.md).

What v73 added (all code defaults, no compose change):

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
path is bitwise equal to v72. v75 shares the disk-cache key with v60–v74.

Known limitations: the 524k profile has no measurements on this engine. The full list is in [README.md](../../README.md#known-limitations).
