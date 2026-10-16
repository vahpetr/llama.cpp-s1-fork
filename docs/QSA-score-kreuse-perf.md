# QSA_SCORE pooled-key reuse across query tiles (A2b) - perf note

`[TAG_QSA_SCORE_KREUSE]`, knob `GGML_VK_QSA_SCORE_KREUSE` (default on; `0` = the A2a dispatch order). Class EXACT.

## What changed

The indexer score GEMM (`GGML_OP_QSA_SCORE`, K = 128, 4 heads, one 128-query chunk = N 512 = four 128-wide N tiles on
the l tile) used to be dispatched as `x = M tile, y = N tile`. The GPU walks x fastest, so it streamed the whole pooled-key
store (16 / 32 / 64 MiB of f16 keys at 262k / 524k / 1M) once per N tile: 4 times per chunk, 16 chunks per 2048-token
ubatch, per QSA layer.

The `*_qsa_kr` twins (same `mul_mm.comp`, `QSA_SCORE_KREUSE` defined) number the workgroups key tile major:
`x = ir * blocks_n + ic`, `y = 0`. The `blocks_n` workgroups that read one BM x K tile of keys therefore run back to back,
and all but the first find that tile in cache. Each workgroup still runs the per-tile program unchanged for its (ir, ic):
the same A / B loads, shared-memory layout, fragments, `coopMatMulAdd` sequence over K, +0 accumulator start and epilogue.
So every output's bits are the per-tile twin's. The SPIR-V of every other shader is byte-identical to the base.

Eligible: knob on, coopmat1, plane of at least 2 N tiles, split_k 1, and `blocks_m * blocks_n` within
`maxComputeWorkGroupCount[0]`. Everything else keeps the A2a dispatch:
- decode / verify widths 3..9 (one N tile);
- the mat-vec widths 1..2, which never reach QSA_SCORE.

## 780M (balkon, RADV KHR_coopmat, GGML_VK_DISABLE_MMQ_CM1=1)

`test-qsa-score-kreuse --perf 11 <M...>`: deployed chunk shape (store + remap + f16 mirror + QSA_BIAS constants). The
figure is ms per call, 16 calls per graph, median of 11 graphs; the table shows the median of 3 alternating rounds per arm.
The per-step figure is the trapezoid mean over 5 depths in the step x 6144 calls (32 ubatches x 16 chunks x 12 QSA
layers).

| step | off ms/call | on ms/call | per 64k step off -> on | saving |
|---|---|---|---|---|
| ->262k (M 49152..65536) | 1.963 | 1.804 | 12.06 -> 11.09 s | 0.97 s (-8.1 %) |
| ->524k (M 114688..131072) | 4.211 | 3.792 | 25.87 -> 23.30 s | 2.57 s (-10.0 %) |
| ->1M (M 245760..262144) | 8.662 | 7.869 | 53.22 -> 48.34 s | 4.87 s (-9.2 %) |

With the f32 store (mirror off), the saving is -26..-27 % (4.5 / 9.9 / 20.5 s per step): the key bytes are twice as large
there.

Real graph cross-check: the tiny qwen4exp fixture (one QSA layer; prod 1M QSA env, ub 2048, chunk 128, f16 mirror) was
filled with `GGML_VK_PERF_LOGGER=1`. The sum of its QSA_SCORE GPU time per 64k step x 12 layers:
- ->262k: 12.13 -> 10.76 s (-1.37 s, -11.3 %);
- ->524k: 26.22 -> 22.86 s (-3.36 s, -12.8 %);
- ->1M, off arm only: 53.29 s (31 of 32 ubatches).

The ->1M on arm was stopped by the GTT guard next to the hands server. The 1M off arm hit a perf-logger-only assert
(`compute_ctx.expired()`) at its last ubatch, on the unchanged A2a path.

Power-of-two pools gain the most: M 65536 2.46 -> 2.03 ms, 131072 4.75 -> 4.00 ms. The per-tile order shows a jump
there (61440 -> 65536: +18 % time for +6.7 % keys), and the key-major order removes it.

## Variants measured and not kept (780M, same harness)

| variant | per-call change vs off (f16 mirror) | why it is not kept |
|---|---|---|
| keys resident in shared memory for the whole K (BM x 128 f16), one workgroup iterating all N tiles (bit-identical: test-qsa-score-kreuse PASS) | +13..+35 % (slower) | 48 KiB LDS per workgroup: 2 instead of 5 workgroups per WGP, so global-load latency is no longer hidden |
| one workgroup iterating the N tiles, keys re-loaded per tile (cache-hot), base LDS (perf only, not bit-tested) | -8..-25 % | fewer workgroups in flight than the key-major order |
| **key-major workgroup order (kept)** | **-4..-37 %** (-13..-37 % at power-of-two pools, -4..-11 % between them) | - |

Negative control: the kept twin with only the two TK steps of each K block swapped makes all 119 kr cases of
`test-qsa-score-kreuse` differ, and 0 of the non-kr cases. The test therefore detects a per-output summation-order change.

## Not done

Cross-chunk reuse. The 16 chunks of a ubatch are separate nodes, each releasing its plane before the next. Reusing a key
tile across chunks needs all chunk planes alive at once, about 2 GiB at 1M after A1 (bounded storage, plan A1 -> A2b
inter-chunk). This step leaves it out.
