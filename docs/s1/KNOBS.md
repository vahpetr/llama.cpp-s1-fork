# Fork flags and env knobs

Everything this fork adds on top of upstream's options: server flags first, then environment variables by area. Env knobs are
read once at start (or at context/memory creation). Upstream flags I rely on but did not add (`--ctx-checkpoints`,
`--checkpoint-min-step`, `--spec-*` except the ones below, `--kv-unified`, `--no-context-shift`) are documented upstream.

Class: **EXACT** = outputs and state bytes identical with the knob on and off (proven by a gate); **COMPOSITION** = same math per
op but a different batching / selection route, so near-ties can flip a token (classified by replay, see AGENTS.md); **LOSSY**
= changes numerics; **SCHED** = scheduling or memory only, never outputs; **DIAG** = logging / checks / test-only.
"prod" = the value in my s1 brain composes (`deploy/s1/`); "-" = unset (the default). Rows marked (?) I have not re-checked
against the code for this table.

## Server flags

| flag | default | effect | class | prod |
|---|---|---|---|---|
| `--qsa-ratio N` | -1 | -1 = QSA ratio from the GGUF, 0 = force dense attention, N = force ratio N on every full-attention layer that has indexer tensors | LOSSY vs dense (it is the sparse-attention switch) | 4 |
| `--kv-graph-span N` | 0 | size the worst-case compute reserve for N KV cells instead of the full cache; bigger batches regrow on demand | SCHED | 0 (full span: flat memory) |
| `--cache-disk DIR` | off | disk tier of the prompt cache: an evicted conversation is written to DIR and loaded back; one server per DIR | EXACT (gated: identical continuation) | per profile |
| `--cache-disk-gib N` | 200 | quota; over it the entry with the lowest re-prefill seconds per GiB is deleted | SCHED | 200 |
| `--cache-disk-min-tokens N` | 8192 | only prompts this long go to disk | SCHED | 4096 |
| `--cache-disk-ttl-days N` | 14 | delete entries unused this long (0 = never) | SCHED | 14 |
| `--cache-disk-ckpt-ratio F` | 0.125 | keep a stored recurrent checkpoint when its gap is >= F x its distance to the conversation end | SCHED | 0.125 |
| `--cache-disk-ckpt-max N` | 32 | stored checkpoints per conversation (113 MiB each for this model) | SCHED | 32 |
| `--cache-disk-adopt KEY,...` | - | open another build's cache namespace read-only (after an exact upgrade); entries are verified at load | EXACT when the source build is exact | - |
| `--spec-draft-n-max-adaptive` | off | adapt the draft length per slot from the running acceptance | output-preserving | - |
| `--spec-draft-n-max-adaptive-cost a,b[,c,d]` | built-in | depth-aware cost model for the rule above | - | - |
| `--spec-draft-n-max-adaptive-measure` | on | prefer measured round times over the cost model | - | - |
| `--backend-sampling` | off | upstream flag; on this fork it works with draft-mtp but bypasses the graph cache and the CPU fast path (slower decode) and is not bit-exact at temperature > 0, so keep it off for speed. Where the target context cannot hold a verify round (1 + n_max outputs per sequence) or under tensor split it is forced off at start (WRN `[bs-guard] --backend-sampling forced off: <reason>`; INF `[bs-guard] backend_sampling_supported=<0\|1> spec_type=<type> n_max=<N> outputs_per_seq=<M>` always) and a request with `backend_sampling: true` gets a 400; `/props` reports `backend_sampling_supported` and `speculative {type, n_max}` | LOSSY at temperature > 0 (sampling route) | off |
| `--spec-draft-policy off\|tput` | off | tput = pick the draft length that maximises committed tokens per ms (gate: no gain here) | output-preserving | off |

## Sparse attention (QSA) - `LLAMA_QSA_*`, `GGML_VK_QSA_*`

| knob | default | effect | class | prod |
|---|---|---|---|---|
| `LLAMA_QSA_MIN_KV=N` | 0 | QSA layers run dense while the sequence is shorter than N | EXACT below N (dense) | 65536 |
| `LLAMA_QSA_POOLED_CACHE=1` | 0 | persistent store of the indexer's pooled block keys, re-pool only changed blocks | EXACT | 1 |
| `LLAMA_QSA_CENSUS_INC=1` | 0 | incremental block census / store plan (needs the pooled cache) | EXACT | 1 |
| `LLAMA_QSA_INPUTS_INC=1` | 0 | incremental QSA graph inputs with device-resident mirrors | EXACT | 1 |
| `LLAMA_QSA_COMPACT_BIAS=1` | 0 | selection bias from per-block tables (`GGML_OP_QSA_BIAS`), no [blocks x tokens] plane | EXACT | 1 |
| `LLAMA_QSA_GATHER_Q8=1` | 0 | gather selected K/V rows as q8_0 (type-preserving get_rows) | EXACT (?) | 1 |
| `LLAMA_QSA_GATHER_MAX_WINDOWS=N` | 0 (no cap) | cap on gather windows per ubatch | EXACT (?) | 64 |
| `LLAMA_QSA_CHUNK=C` | 0 | run a QSA layer's query half in chunks of <= C queries | EXACT (ids identical) | 128 |
| `LLAMA_QSA_CHUNK_PAD=1` | 0 | pad the last chunk | DIAG | - |
| `LLAMA_QSA_PREFILL=indexed` | scan | prefill: flash attention over the selected ids instead of a masked scan | COMPOSITION | indexed |
| `LLAMA_QSA_VIS=1` | 0 | QSA selection / indexed FA read cell metadata, no KQ mask plane | EXACT | 1 |
| `LLAMA_QSA_RELU_INPLACE=1` | 0 | in-place ReLU on the indexer score | EXACT | 1 |
| `LLAMA_QSA_POOL_REMAP=indexed\|copy` | indexed | score reads the pooled store through the remap (no copy) | EXACT | - |
| `LLAMA_QSA_HISTORY_DENSE=0\|1` | on | route a sequence dense while its indexer history is incomplete (after a restore / rewind) | EXACT for full histories | - |
| `LLAMA_QSA_HOMOGENEOUS=1` | 0 | multi-slot: homogeneous routing classes + strict envelope | COMPOSITION | - (np2 profile only) |
| `LLAMA_QSA_HOMOGENEOUS_TARGET_CAP=N` | derived | target ubatch cap for the routing classes | SCHED (?) | - |
| `LLAMA_QSA_WHOLE_BLOCKS=1` | 0 | block-level selection as in the HF reference | COMPOSITION (measured, not adopted) | - |
| `LLAMA_QSA_PATH=auto\|gather\|scan` | auto | force the attention branch | DIAG | - |
| `LLAMA_QSA_BYPASS_EXHAUSTIVE=0` | on | 0 keeps the full QSA graph when the selection already covers every key | DIAG | - |
| `LLAMA_QSA_TAIL_FIX=0` | on | 0 restores the old tail-cell bias (bug) | DIAG | - |
| `LLAMA_QSA_INDEXED_SORT=1` | 0 | sorted ids for the frozen-state tool | DIAG | - |
| `LLAMA_QSA_ROUTING_COST=1`, `LLAMA_QSA_DUMP_IDS=1`, `LLAMA_QSA_CENSUS_CHECK=1`, `LLAMA_QSA_COMPACT_BIAS_CHECK=1`, `LLAMA_QSA_TAIL_ASSERT=1` | 0 | cost logging, `[qsa-ids]` dumps, oracle checks | DIAG | - |
| `GGML_VK_QSA_FUSED_SELECT=1` | 0 | score + block map + mask -> top-k ids in one op | EXACT | 1 |
| `GGML_VK_QSA_SCRATCH_PRESIZE=1`, `..._FA=1` | 0 | presize the selection / indexed-FA scratch at load | SCHED | 1, 1 |
| `GGML_VK_QSA_TOPK=v1\|v2` | v2 | top-k kernel (v2 = distributed radix threshold, byte-identical to v1) | EXACT | - |
| `GGML_VK_QSA_TOPK_WAVE`, `GGML_VK_QSA_TOPK_CPT` | auto | top-k subgroup size / cells per invocation | EXACT | - |

## MTP draft - `LLAMA_MTP_*`

| knob | default | effect | class | prod |
|---|---|---|---|---|
| `LLAMA_MTP_QSA=1` | 0 | sparse draft: the MTP block runs QSA with its own indexer | EXACT for the target (draft acceptance +0.5 pp) | 1 |
| `LLAMA_MTP_QSA_MAX_UBATCH=N` | 8 | draft catch-up ubatches above N run dense | EXACT target | - |
| `LLAMA_MTP_QSA_TOP_K=N` | model | draft selection width | changes draft only | - |
| `LLAMA_MTP_QSA_PHASE=1` | 0 | reference block phase for the draft (lost to the default on acceptance) | changes draft only | - |
| `LLAMA_MTP_CATCHUP_ATTN=0` | 1 | attention-free draft catch-up on prefill (store K/V only) | EXACT | 0 |
| `LLAMA_MTP_EHPROJ_FLAT=0`, `..._MIN_TOKENS=N` | on, 32 | MTP `eh_proj` as one flat GEMM for >= N tokens | COMPOSITION (oracle-gated) | - |
| `LLAMA_MTP_CARRIER_ZERO=1` | 0 | upstream #28333 carrier zeroing | changes draft only | - |

## Graph, memory, admission

| knob | default | effect | class | prod |
|---|---|---|---|---|
| `LLAMA_GRAPH_CACHE=K` | 0 | keep K built graphs + allocation plans, reuse on an exact match | EXACT | 8 |
| `LLAMA_GRAPH_CACHE_CHECK=1\|2\|3` | 0 | rebuild and compare on every hit (1 plan, 2 outputs, 3 stale inputs); aborts on a difference | DIAG | - |
| `LLAMA_GRAPH_CACHE_CHECK_INJECT=...` | - | fault injection for the check | DIAG | - |
| `LLAMA_KQ_VIS=1` | 0 | KQ mask from cell/token metadata, no mask plane | EXACT | 1 |
| `LLAMA_KQ_VIS_MIN_TOKENS=N` | 64 | below N tokens per batch use the classic mask plane | EXACT | 64 |
| `LLAMA_KQ_VIS_PROF=1` | 0 | `kqvis-prof:` timing lines per window of <= 64 graphs (incl. graph-cache hit/miss) | DIAG | - |
| `LLAMA_RESERVE_RUNTIME_SHAPE=1` | 0 | reserve max(worst-case, runtime) graph shape: no regrow mid-fill | SCHED | 1 |
| `GGML_ALLOC_GROW_SLACK=P` | 0 | grow compute buffers with P % slack | SCHED | 10 |
| `LLAMA_GTT_ADMISSION_MIB=N` | 0 (off) | before a prefill chunk grows GTT past N MiB, the slot waits | SCHED | GTT - 3424 |
| `LLAMA_GTT_SLOPE_B_PER_CELL`, `LLAMA_GTT_STAGING_B_PER_CELL`, `LLAMA_GTT_STAGING_STEP_MIB`, `LLAMA_GTT_ONCE_MIB`, `LLAMA_GTT_SLACK_PCT` | 18432, 2048, auto, 0, = grow slack | the admission's growth model | SCHED | 256, 0, -, 1200, 0 |
| `LLAMA_GTT_ADMISSION_MAX_WAIT_S` | 600 | then HTTP 503 | SCHED | 300 |
| `LLAMA_GTT_USED_PATH` | auto | sysfs file with GTT used bytes | SCHED | card1 |
| `LLAMA_GTT_CONCURRENT_RESERVE_MIB`, `LLAMA_GTT_HOLES_FULL_CHUNK` | 0, 0 | multi-slot admission reserve | SCHED | - |
| `LLAMA_PLE_PREFETCH=0` | on | madvise(WILLNEED) on the PLE rows of each ubatch | EXACT | - |
| `LLAMA_PLE_HOIST=1` | 0 | PLE gather as the first node (measured -3.5 % prefill) | EXACT | - |
| `LLAMA_PLE_GATHER_SORT` / `GGML_GET_ROWS_SORTED` | auto | page-ordered get_rows for large sources | EXACT | - |
| `GGML_GET_ROWS_THREADS=N` | 1 | multi-threaded gather (neutral here) | EXACT | - |
| `LLAMA_SAMPLER_FASTPATH=0\|1\|2` | 1 | candidate-window sampling fast path; 2 = verify against the full path | EXACT | - |
| `LLAMA_SPEC_TPUT_MARGIN=F` | 0.10 | margin for the tput policy | - | - |

## Vulkan backend - `GGML_VK_*`

| knob | default | effect | class | prod |
|---|---|---|---|---|
| `GGML_VK_DISABLE_MMQ_CM1=1` | 0 | turn off the int8 coopmat1 MMQ path (upstream #27952 back-port) | LOSSY when on (lowered draft acceptance here) | 1 |
| `GGML_VK_MMQ_INT8_MIN_N`, `GGML_VK_MMQ_INT8_OPS` | 0, all | where int8 MMQ applies | LOSSY | - |
| `GGML_VK_FA_GQA_TOKENS=0` | 1 | fold verify tokens x GQA heads into FA tiles; 0 = per-token dispatch | COMPOSITION | 0 |
| `GGML_VK_FREE_WAIT_IDLE=1` | 0 | wait for idle queues before freeing a >= 256 MiB buffer | SCHED | 1 |
| `GGML_VK_SUBALLOCATION_BLOCK_SIZE` | upstream | (upstream knob) 4 GiB blocks here | SCHED | 4294967296 |
| `GGML_VK_EXPERT_OCCUPANCY=1`, `..._EVERY=N` | 0, 16 | `[vk-expert-occ]` line per sampled prefill mul_mat_id: rows per expert, percentiles | DIAG (adds a barrier) | - |
| `GGML_VK_MMID_COMPACT=0` | 1 | flattened (expert, tile) grid for mul_mat_id | EXACT | - |
| `GGML_VK_DISABLE_HOIST_ROW_IDS=1` | 0 | per-workgroup ids scan instead of the hoisted table | EXACT | - |
| `GGML_VK_CONCAT_TRANSPOSE=0`, `GGML_VK_FUSE_HC_COMBINE=0`, `GGML_VK_FUSE_SIGMOID_MUL=0`, `GGML_VK_FUSE_GDN_PRE=0` | on | disable one fork kernel / fusion (A/B) | EXACT | - |
| `GGML_VK_FUSE_MULTI_ADD_SCALE=0`, `GGML_VK_FUSE_SCALE_SILU=0` | on | disable one hc_mix tail fusion (patch 0227: multi_add + trailing SCALE - qwen4exp hc_mix stream mean and QSA pooled-key mean - and SCALE + SILU; pattern names `multi_add_scale`, `scale_silu`; no node-order change) (B3 arm, not in a release yet) | EXACT | - |
| `GGML_VK_FUSE_CONCAT_SSM_CONV=0` | on | read the GDN conv window from state | x instead of writing the concat (patch 0228; patterns `concat_ssm_conv`, `_silu`, `_bias_silu`; prefill ubatches only); =0 also restores the graph_optimize order (B4 arm, not in a release yet) | EXACT | - |
| `GGML_VK_MMID_PRECAST_B=0` | on | coopmat1 mul_mat_id converts B to f16 once (shared by gate/up) and runs the f16-B twin of the shader instead of converting per tile (patch 0231, B1 arm, not in a release yet) | EXACT | - |
| `GGML_VK_MMID_TILE_BM128=0` | on | AMD coopmat1 mul_mat_id large tile BM 128 x BN 64 (m > 64 and n > 64; upstream: off on AMD; q4_K / q5_0 / q8_0 / q6_K / iq3_s only, like PRECAST_B) (patch 0232, B1 arm, not in a release yet) | EXACT | - |
| `GGML_VK_DENSE_TILE_M16=0` | on | AMD coopmat1 dense matmul with m <= 16 (the qwen4exp hc inject, m = 4) on a BM 16 x BN 16 tile instead of the s tile; q4_K / q8_0 A (patch 0234, B4c arm, not in a release yet) | EXACT | - |
| `GGML_VK_DENSE_TILE_B2=1`, `GGML_VK_DENSE_TILE_B2_XL=1` | off / off | AMD coopmat1 dense matmul tile plan: q4_K / q5_0 / q6_K with m <= 384 on BM 64 x BN 128 instead of the l tile (split_k 1 only); _XL=1 also puts m >= 2048 on BM 128 x BN 256 (slower on the 780M) (patch 0233; default off since the gfx1151 run4 ruling: the hc tile is 2.6x slower there on q4_K 320 x 10240; A/B only) | EXACT | - |
| `GGML_VK_MMVQ_HOIST=0`, `GGML_VK_MMVQ_HOIST_MIN_COLS=N` | on, 2 | mat-vec at 2..8 columns (decode verify widths): the MMVQ q4_0 / q5_0 / q8_0 / q4_K and f32 x f32 twins unpack A once per row instead of once per column, from N columns on (patch 0236, C1 arm, not in a release yet) | EXACT | - |
| `GGML_VK_QSA_SCORE_EPILOGUE=0` | on | indexer score as one node (GGML_OP_QSA_SCORE): ReLU, the ordered 4-head sum and the block bias in the coopmat1 store of the score GEMM, no per-head plane; decode / 2-token verify and non-coopmat1 backends keep the chain; read by qwen4exp, =0 = the v59 graph (patch 0235, A1 arm, not in a release yet) | EXACT | - |
| `GGML_VK_FA_DEADV_LDS=1`, `GGML_VK_FA_NO_ZERO_DEAD_V=1` | 0 | FA dead-column variants (A/B; the second reopens an RDNA3 signed-zero hazard) | DIAG | - |
| `GGML_VK_OPT_WINDOW=N` | 20 | graph_optimize independence window | COMPOSITION | - |
| `GGML_VK_DMMV_RM_{KQ,STDQ,KQ_INT,STDQ_INT,F32}` | built-in | rows per workgroup for dense mat-vec | EXACT | - |
| `GGML_VK_ASYNC_READ_UMA=1`, `GGML_VK_ASYNC_READ_COPYBUF=1`, `GGML_VK_FLOPS_SEED_LEGACY=1`, `GGML_VK_DISABLE_DESCRIPTOR_REUSE=1` | 0 | readback / submit / descriptor variants (A/B) | EXACT | - |
| `GGML_VK_DISABLE_FUSION_PATTERNS=a,b`, `GGML_VK_FUSION_LIST=1`, `GGML_VK_FUSION_COUNT=1`, `GGML_VK_FUSION_DIAG=1`, `GGML_VK_SUBMIT_LOG=1` | - | fusion / submit diagnostics | DIAG | - |
| `GGML_VK_DISPATCH_LOG=1` | 0 | one `[vk-dispatch]` line per MUL_MAT / MUL_MAT_VEC / MUL_MAT_ID dispatch: path, types, m/n/k, n_as/n_used, pipeline, split_k, mmvq/mmq (patch 0230, not in a release yet) | DIAG (no barrier) | - |
| `GGML_SCHED_PROFILE=1`, `GGML_GET_ROWS_STATS=1\|2` | 0 | host-side scheduler timers, get_rows census | DIAG | - |

## Test / debug only

`LLAMA_GDN_NO_DIRECT`, `LLAMA_SERVER_DEBUG_FAULT=slot|decode|ckpt`, `LLAMA_PLE_DUMP=<file>`, `LLAMA_KLD_DUMP=<file>`
(llama-perplexity), `GGML_TEST_*` (test-backend-ops).
