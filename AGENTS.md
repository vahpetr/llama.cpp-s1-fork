# Notes for coding agents (and people) working on this fork

This is a runtime fork of llama.cpp for one model family (Qwen3.8-Flash-Next, `qwen4exp`) on AMD iGPUs with Vulkan. Upstream's
contributor rules for llama.cpp itself are in
[upstream AGENTS.md](https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md); they apply to anything sent upstream from here.
Read README.md first, then docs/s1/KNOBS.md (every flag and env knob) and docs/s1/PATCHES.md (every change, numbered).

This tree is **v61** (2026-10-17): fork revision `7a438f3ba` (`prod-v11-stage8-pkg8`) plus the docs and deploy files of this
`s1-public` branch. Image `vahpetr/llama-cpp-s1-fork:v61` (= `:v61k` = `:latest`); deploy steps in docs/s1/DEPLOY.md, what
changed in docs/s1/RELEASE-NOTES.md.

## Where the fork's code lives

| area | files |
|---|---|
| model graph: QSA sparse attention (incl. the fused score op), MTP draft block, PLE gather, hyper-connections, GDN conv input split | `src/models/qwen4exp.cpp`, `src/models/delta-net-base.cpp` |
| hybrid memory with the QSA indexer cache: pooled keys (+ f16 mirror), census, store plan, incremental inputs, cell-ext restore | `src/llama-memory-hybrid-idx.cpp/.h` |
| state provenance: the `IXY1` mark, the legacy-state accept API | `src/llama-ext.h`, `src/llama-kv-cache.cpp` (`seq_history_intact`) |
| KQ mask from metadata (`LLAMA_KQ_VIS`), its device mirror with per-consumer dirty ranges, cells | `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`, `src/llama-kv-cells.h` |
| graph + allocation-plan cache, reserve shape, MTP catch-up | `src/llama-context.cpp`, `ggml/src/ggml-backend.cpp`, `ggml/src/ggml-alloc.c` |
| speculative decoding (MTP draft, replay, adaptive / tput policy) | `common/speculative.cpp`, `common/sampling.cpp` |
| server: slots, checkpoints, GTT admission, multi-slot routing, stop-type log | `tools/server/server-context.cpp`, `server-task.cpp` |
| disk prompt cache (tiers, deep checkpoints, namespaces, adopt, fingerprint, legacy-state policy `LLAMA_LEGACY_UNMARKED`) | `tools/server/server-disk-cache.cpp/.h` |
| Vulkan kernels: QSA top-k, indexed / metadata FA, KQ mask, fusions, MMQ int8, expert telemetry; v60: QSA score epilogue and the matmul tile plans (`mul_mm`), mat-vec hoist, hc_mix tail and conv-input fusions | `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `vulkan-shaders/{topk_radix_select*,flash_attn_*,kq_mask,qsa_bias,hc_combine,add_softplus_mul,mul_mmq_cm1*,mul_mm,mul_mat_vec*,multi_add,scale_silu,ssm_conv}` |
| new ops (`GGML_OP_QSA_TOP_K`, `QSA_BIAS`, `QSA_SCORE`, `KQ_MASK`, FA with ids / metadata, `get_rows_type`) | `ggml/include/ggml.h`, `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c` |
| webui (compaction, per-chat settings, timing chips, frozen tools) | `tools/ui/` (Svelte; tests under `tools/ui/tests/`) |
| tools | `tools/mtp-replay/`, `tools/qsa-frozen/`, `tools/s1/gguf_hc_q8.py` |
| deploy + docs | `deploy/`, `docs/s1/` |

## Tags: how to find a feature

Fork code and commits carry `[TAG_<NAME>]` markers. `git grep -n TAG_GRAPH_CACHE` finds every site of a feature;
`git log --grep TAG_MULTISLOT` finds its commits. Main tags: `TAG_QSA_*` (sparse attention: `_POOLED_CACHE`, `_CENSUS_INC`,
`_INPUTS_INC`, `_VIS`, `_INDEXED_FA`, `_POOL_REMAP`, `_COMPACT_BIAS`, `_CHUNK`, `_TOPK_V2`, `_FUSED_SELECT`, `_RATIO_OVERRIDE`,
`_MIN_KV`, `_SCRATCH`, `_WHOLE_BLOCKS`), `TAG_KQ_VIS`, `TAG_MTP_QSA`, `TAG_MTP_CATCHUP_NOATTN`, `TAG_GRAPH_CACHE`,
`TAG_MULTISLOT`, `TAG_GTT_ADMISSION`, `TAG_KV_GRAPH_SPAN`, `TAG_DISK_PROMPT_CACHE`, `TAG_DEEP_CKPT`, `TAG_DISK_ADOPT`,
`TAG_CACHE_COMPAT2`, `TAG_STATE_DESCRIPTOR`, `TAG_SPEC_DRAFT_POLICY`, `TAG_SPEC_ROUND_TIMING`, `TAG_WEBUI`, `TAG_VK_EXPERT_OCC`,
`TAG_UP<pr>` (an upstream back-port). v60 added `TAG_HC_MIX_TAIL` (0227), `TAG_CONV_INPUT_SPLIT` (0228), `TAG_QSA_DUMP_IDS` /
`TAG_QSA_DUMP_SCORE` (0229), `TAG_VK_DISPATCH_LOG` (0230), `TAG_MMID_PRECAST_B` / `TAG_MMID_TILE_B1` (0231 / 0232),
`TAG_DENSE_TILE_B2` (0233), `TAG_INJECT_TILE_M16` (0234), `TAG_QSA_SCORE_EPILOGUE` (0235), `TAG_MMVQ_HOIST` (0236),
`TAG_KQ_VIS_MIRROR` (0237), `TAG_SPEC_TRACE_IDS` (0238), `TAG_CELL_EXT_RESTORE` / `TAG_CELL_EXT_PROVENANCE` (0239),
`TAG_QSA_POOLED_F16_MIRROR` (0240); v61 added `TAG_QSA_SCORE_KREUSE` (0241) and `TAG_QSA_TOPK_NOKEYS` (0243). New work gets a new tag in both the code comment and the commit subject.

## Build and test contract

1. CPU build and the unit tests first, on any machine:
   `cmake -B build -DLLAMA_BUILD_TESTS=ON && cmake --build build -j && ctest --test-dir build --output-on-failure`.
   Fork tests: `test-qsa-census-inc` (census vs a frozen reference, fuzzed), `test-qsa-select`, `test-qsa-relu-inplace`,
   `test-kq-mask-meta` / `test-kq-vis-fill-equiv` (metadata mask == plane), `test-graph-cache` / `test-sched-snapshot` (cache
   off == on == CHECK), `test-mtp-catchup` / `test-mtp-qsa-restore` / `test-mtp-qsa-phase` / `test-mtp-carrier`,
   `test-state-portable` (np1 <-> np2 state), `test-cache-compat-key` (incl. the unmarked-checkpoint rules),
   `test-spec-draft-policy`, `test-draft-accept-eog`, `test-cell-ext-restore` / `-qsa` (state restore with and without PLE,
   text and image entries, marked and unmarked).
   Built but not ctests (need a Vulkan coopmat1 GPU): `test-qsa-f16-mirror` (mirror off vs on + CHECK through the model graph),
   `test-vk-graph-order` (fusion / dispatch order through the scheduler, `tests/test-vk-graph-order.sh`).
2. Vulkan op tests on a box with the GPU: build with `-DGGML_VULKAN=ON`, then
   `test-backend-ops -b Vulkan0 -o <OP>` for every op you touched (`FLASH_ATTN_EXT`, `MUL_MAT_ID`, `TOPK_QSA`, `QSA_TOP_K`,
   `KQ_MASK`, `GET_ROWS`, ...). The fork added cases at the deployed shapes; `GGML_TEST_TOPK_PROD=1` enables the large ones.
3. Server: `tools/server/tests` (pytest; `test_kv_unified_purge.py`, `test_spec_recurrent.py`). webui:
   `cd tools/ui && npm ci && npm run test:unit -- --run && npm run check`.
4. Anything that can change outputs or speed is then gated on the real model (below). Unit tests alone never prove a change.

## Exactness classes: every change states its class and how it was proven

- **EXACT**: same tokens, same probabilities, same state bytes. Proof = per-step slot state hashes identical (save the slot,
  sha256), `[qsa-ids]` selection dumps identical (`LLAMA_QSA_DUMP_IDS=1`), the census oracle identical (fixed prompts, greedy,
  512 tokens, every position compared with probabilities), and CHECK modes where they exist (`LLAMA_GRAPH_CACHE_CHECK`,
  `LLAMA_QSA_CENSUS_CHECK`, `LLAMA_KQ_VIS_MIRROR_CHECK`, `LLAMA_QSA_POOLED_F16_MIRROR_CHECK`). Since v60 the gate is my S0.1
  oracle: one probe program (private) drives the production configuration through fixed prompts and compares the new build
  with the previous one bit for bit - logits at every captured position, the saved states, the QSA ids per chunk (`[qsa-ids]`
  sha256 records, patch 0229) - at verify widths 1-6 in full and the last 6 positions of widths 7-9, depths 8k -> 1M, and
  restore / replay histories. v60 vs v59: 1 502 logit positions and 224 state points identical, on gfx1151, the package as a
  whole, production configuration minus the vision projector, one slot, text corpus; v61 the same against the same v59
  reference, plus an off arm per lever (docs/s1/BENCHMARKS.md section 0a).
- **COMPOSITION**: same per-op math, different batching / routing / selection order, so near-ties can flip a token. Proof =
  teacher-forced replay after the first divergence: every divergence must be a near-tie or a selection flip that the reference
  also accepts (my "R81" classifier), plus needles and speed. A change that cannot be classified this way is treated as LOSSY.
- **LOSSY**: different numerics (quant types, int8 paths, budget caps). Needs a quality gate: KL <= 1.1x the quant's own KL vs
  Q8_0, needles / census / agent bench unchanged, and a person's decision. Never on by default.
- Commit messages and PATCHES.md rows say which class and which proof. Speed claims carry the measured noise band (~±1.5 %).

## How to measure

- `LLAMA_KQ_VIS_PROF=1`: `kqvis-prof:` lines per window of <= 64 graphs (build / input / sync time, graph-cache hits, QSA plan).
- `GGML_VK_PERF_LOGGER=1` (read once at backend init): per-op GPU times; adds barriers, so totals run ~7 % high.
- `GGML_VK_EXPERT_OCCUPANCY=1`: `[vk-expert-occ]` rows-per-expert stats for prefill `mul_mat_id` (adds a barrier).
- Server log: one `stop processing: n_tokens = N, stop=<eos|limit|word|cancel|none> n_gen= n_predict= truncated=` line per
  finished generation; per-round speculative timing lines; `[qsa-history]`, `[qsa-routing]`, `[mtp-contiguity]` proof counters.
- HTTP: `/metrics` (with `--metrics`; `disk_cache_*`, admission counters), `/slots`, `GET /disk-cache` (own key, entries,
  adopt sources), `/props`.
- One GPU-heavy job at a time on a box: two model loads, or a build next to a loaded model, have run my 128 GB box out of memory.

## Known no-gos

- No context shift: the Gated-DeltaNet layers are recurrent. Run with `--no-context-shift`; the client compacts before the
  window is full (the webui does, see patch 0209).
- Never point an experiment at a production `--cache-disk` directory: a server started with another fingerprint drops the
  entries it cannot use. Give every arm its own directory.
- Two slots give no total throughput on these boxes (MoE + speculative verify; measured -15 %). Use one slot per box unless you
  need concurrency.
- Do not trust a speed number without the profiler off and a paired control on the same image.

## Contributing a patch

One commit per change, with its tag, the test that covers it, and its class + proof in the message. Keep defaults unchanged
(new behaviour behind a knob) until a gate says otherwise, then flip the default in a separate commit. Add a row to
docs/s1/PATCHES.md and docs/s1/KNOBS.md. Issues and PRs are welcome; I review them as time allows.
