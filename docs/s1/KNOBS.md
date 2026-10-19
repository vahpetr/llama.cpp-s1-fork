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
| `--cache-disk-adopt KEY,...` | - | open other builds' cache namespaces read-only (`KEY` = the 16-hex key a build shows in `GET /disk-cache` and logs as `own key`; its directory is `ns-<KEY>`). Only a source with the same model / draft / mmproj files, per-slot context size and KV types is opened; entries are verified at load, new saves go to the own namespace, the source is never written. Use it after an upgrade that changes the key, e.g. v59 -> v60 (docs/s1/DEPLOY.md) | EXACT when the source build is exact; v60 loading pre-v60 entries: see `LLAMA_LEGACY_UNMARKED` | the previous version's key (1M profile, since v60) |
| `--spec-draft-n-max-adaptive` | off | adapt the draft length per slot from the running acceptance | output-preserving | - |
| `--spec-draft-n-max-adaptive-cost a,b[,c,d]` | built-in | depth-aware cost model for the rule above | - | - |
| `--spec-draft-n-max-adaptive-measure` | on | prefer measured round times over the cost model | - | - |
| `--backend-sampling` | off | upstream flag; on this fork it works with draft-mtp but bypasses the graph cache and the CPU fast path (slower decode) and is not bit-exact at temperature > 0, so keep it off for speed. Where the target context cannot hold a verify round (1 + n_max outputs per sequence) or under tensor split it is forced off at start (WRN `[bs-guard] --backend-sampling forced off: <reason>`; INF `[bs-guard] backend_sampling_supported=<0\|1> spec_type=<type> n_max=<N> outputs_per_seq=<M>` always) and a request with `backend_sampling: true` gets a 400; `/props` reports `backend_sampling_supported` and `speculative {type, n_max}` | LOSSY at temperature > 0 (sampling route) | off |
| `--spec-draft-policy off\|tput` | off | tput = pick the draft length that maximises committed tokens per ms (gate: no gain here) | output-preserving | off |
| `--spec-draft-n-ubatch N` (env `LLAMA_ARG_SPEC_DRAFT_N_UBATCH`) | 0 | 0250 (upstream #30459, v62 62.7): physical batch of the draft context; 0 = follow `-ub` (the default, identical to v61); a smaller N shrinks the draft's compute reserve (the second `sched_reserve … compute buffer size` of the start log; 1033.54 MiB at ub 2048 on the 1M brain) and chunks the MTP catch-up of a prefill ubatch into N-row pieces. Part of the disk-cache key today (cache_compat_measure hashes dft_ub: setting it = a new own key) | EXACT* (R110: draft identity across the catch-up width must hold - measured by the u4 unit before a compose sets it) | unset |

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
| `LLAMA_QSA_ROUTING_COST=1`, `LLAMA_QSA_DUMP_IDS=1\|2`, `LLAMA_QSA_CENSUS_CHECK=1`, `LLAMA_QSA_COMPACT_BIAS_CHECK=1`, `LLAMA_QSA_TAIL_ASSERT=1` | 0 | cost logging, `[qsa-ids]` dumps (since 0229 with a sha256 of the ordered id stream per layer and one `[qsa-ids-chunk]` record per selection chunk; `=2` also writes the raw ids to `LLAMA_QSA_DUMP_DIR`, optional graph range `LLAMA_QSA_DUMP_G=lo:hi`), oracle checks | DIAG | - |
| `LLAMA_QSA_DUMP_SCORE=1`, `LLAMA_QSA_DUMP_TIES=1` (+ `_TIES_K=N`, `_TIES_P=lo:hi`) | 0 | `[qsa-score]` hash of every indexer score plane (pre-bias); `[qsa-score-ties]` tied scores at the top-k boundary (patch 0229). No compute op changes, but the observed tensors become scheduler split points, so fusions across them cannot form: compare capture-on runs with capture-on runs only | DIAG | - |
| `GGML_VK_QSA_FUSED_SELECT=1` | 0 | score + block map + mask -> top-k ids in one op | EXACT | 1 |
| `GGML_VK_QSA_SCRATCH_PRESIZE=1`, `..._FA=1` | 0 | presize the selection / indexed-FA scratch at load | SCHED | 1, 1 |
| `GGML_VK_QSA_TOPK=v1\|v2` | v2 | top-k kernel (v2 = distributed radix threshold, byte-identical to v1) | EXACT | - |
| `GGML_VK_QSA_TOPK_WAVE`, `GGML_VK_QSA_TOPK_CPT` | auto | top-k subgroup size / cells per invocation | EXACT | - |
| `GGML_VK_QSA_SCORE_KREUSE=0` | on | v61, 0241: key-tile-major dispatch of the indexer score GEMM (A2b); details in "v61 package" below | EXACT | - |
| `GGML_VK_QSA_TOPK_NOKEYS=0` | on | v61, 0243: top-k select without key rows (A3); details in "v61 package" below | EXACT | - |

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
| `LLAMA_GRAPH_CACHE_TRACE=1` | 0 | one `[graph-cache] ctx= n_past= n_tokens= cache= reused= inval=` line per computed ubatch (patch 0229) | DIAG | - |
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
| `GGML_VK_FA_DEADV_LDS=1`, `GGML_VK_FA_NO_ZERO_DEAD_V=1` | 0 | FA dead-column variants (A/B; the second reopens an RDNA3 signed-zero hazard) | DIAG | - |
| `GGML_VK_OPT_WINDOW=N` | 20 | graph_optimize independence window | COMPOSITION | - |
| `GGML_VK_DMMV_RM_{KQ,STDQ,KQ_INT,STDQ_INT,F32}` | built-in | rows per workgroup for dense mat-vec | EXACT | - |
| `GGML_VK_ASYNC_READ_UMA=1`, `GGML_VK_ASYNC_READ_COPYBUF=1`, `GGML_VK_FLOPS_SEED_LEGACY=1`, `GGML_VK_DISABLE_DESCRIPTOR_REUSE=1` | 0 | readback / submit / descriptor variants (A/B) | EXACT | - |
| `GGML_VK_DISABLE_FUSION_PATTERNS=a,b`, `GGML_VK_FUSION_LIST=1`, `GGML_VK_FUSION_COUNT=1`, `GGML_VK_FUSION_DIAG=1`, `GGML_VK_SUBMIT_LOG=1` | - | fusion / submit diagnostics | DIAG | - |
| `GGML_VK_BUF_LOG=1` | 0 | 0244 (v62 62.1, `[TAG_VK_BUF_LOG]`): one INFO line per Vulkan buffer allocation, `[vk-buf] <device> kind=device\|host\|other size=<B> (<MiB>) type=<memory type index> heap=<heap> flags=<VkMemoryPropertyFlags> wc=<0\|1> import=<0\|1>`; kind device = `ggml_vk_create_buffer_device` (weights, KV / recurrent / pool buffers, preallocs), host = pinned host memory, other = staging / aux; wc=1 = host-visible without HOST_CACHED (write-combined for CPU reads). Prints only (the #30049 read-path change is parked, R129) | DIAG | - |
| `GGML_SCHED_PROFILE=1`, `GGML_GET_ROWS_STATS=1\|2` | 0 | host-side scheduler timers, get_rows census | DIAG | - |

## v61 package - patches 0241-0243

Both are in v61 and neither is set in my composes: the "default" column is what v61 runs. On by default, EXACT as part of the
gated package (bit-identical logits, states and ordered selected ids against the v59 reference on my oracle, and each lever's
off arm equal to the same reference; docs/s1/BENCHMARKS.md section 0a). The `=0` switches exist for A/B runs and to bisect.
Each prints one startup line (shown in the effect column). 0242 has no knob.

| knob | default | effect | class | prod |
|---|---|---|---|---|
| `GGML_VK_QSA_SCORE_KREUSE=0` | on | 0241 (A2b): a QSA_SCORE plane of >= 2 N tiles (a prefill chunk: 128 queries x 4 heads = 4 N tiles) runs on the `*_qsa_kr` twin of the pipeline v60 picks, with the workgroups numbered key tile major, so the workgroups reading one tile of pooled keys are dispatched back to back. Each workgroup runs the same per-tile program, so the bits do not change. Needs coopmat1 (not coopmat2) and split_k 1; one-N-tile planes (decode / verify widths) keep the v60 dispatch. `=0`: the v60 dispatch order, the kr pipelines are not created. Startup line `qsa_score_kreuse: on\|off (...; max <n> workgroups)` | EXACT | - |
| `GGML_VK_QSA_SCORE_KREUSE_MAX_WG=N` | unset (device limit) | 0241: a plane whose workgroup count exceeds `maxComputeWorkGroupCount[0]` falls back to the v60 dispatch; this test hook lowers that limit (only values below the device's are taken) so the fallback can be exercised on a real device | EXACT (the gate ran it at 4) | - |
| `GGML_VK_QSA_TOPK_NOKEYS=0` | on (with the v2 select) | 0243 (A3): the cell form of the QSA top-k select (`topk_radix_select_v2_vis_nk`) computes each cell's key from the score bits and the visibility with integer instructions instead of storing and re-loading per-cell key rows; the scratch has no key region (a 524k test: 258 -> 32.25 MiB preallocated). Equal to the keyed path for finite / -inf, non-subnormal scores. The block form and v1 are unchanged. `=0`: the v60 keyed path. Startup line `qsa_topk: v2 ...; nokeys on\|off` | EXACT | - |
| `GGML_VK_QSA_TOPK_NOKEYS_CHECK=1` | off | 0243, debug: flags visible scores outside the domain where the integer key equals the keyed one (subnormal / NaN / +inf) in a host-visible word and logs one WARN at synchronize; it does not change the selection | DIAG | - |

## v60 package - patches 0227-0240

All of these are in v60 and none is set in my composes: the "default" column is what v60 runs. Every compute change below
that is on by default is EXACT as part of the gated package (bit-identical logits, states and ordered selected ids against v59
on my oracle, see docs/s1/BENCHMARKS.md section 0); the `=0` switches exist for A/B runs and to bisect. The B2 tile (0233) is off
and was not in the gated configuration. Patch numbers: docs/s1/PATCHES.md.

**On by default in v60:** `GGML_VK_FUSE_MULTI_ADD_SCALE`, `GGML_VK_FUSE_SCALE_SILU` (0227), `GGML_VK_FUSE_CONCAT_SSM_CONV` (0228),
`GGML_VK_MMID_PRECAST_B` (0231), `GGML_VK_MMID_TILE_BM128` (0232), `GGML_VK_DENSE_TILE_M16` (0234), `GGML_VK_QSA_SCORE_EPILOGUE`
(0235), `GGML_VK_MMVQ_HOIST` from 2 columns (0236), `LLAMA_KQ_VIS_MIRROR` (0237, only with `LLAMA_KQ_VIS=1`),
`LLAMA_QSA_POOLED_F16_MIRROR` (0240, only with the score epilogue on), `LLAMA_LEGACY_UNMARKED=1` (0239).
**Off by default in v60:** `GGML_VK_DENSE_TILE_B2`, `GGML_VK_DENSE_TILE_B2_XL` (0233, slower on gfx1151), and every debug knob:
`GGML_VK_DISPATCH_LOG` (0230), `LLAMA_SPEC_TRACE_IDS` (0238), `LLAMA_QSA_DUMP_*` / `LLAMA_GRAPH_CACHE_TRACE` (0229), the
`_CHECK` modes of 0237 and 0240.

| knob | default | effect | class | prod |
|---|---|---|---|---|
| `GGML_VK_FUSE_MULTI_ADD_SCALE=0`, `GGML_VK_FUSE_SCALE_SILU=0` | on, on | 0227: a multi_add followed by a SCALE runs as one dispatch (qwen4exp hc_mix stream mean and the QSA pooled-key mean), SCALE + SILU after the hc w_down projection runs as one; pattern names `multi_add_scale`, `scale_silu` for `GGML_VK_DISABLE_FUSION_PATTERNS`; the node order is v59's with the knobs on or off | EXACT | - |
| `GGML_VK_FUSE_CONCAT_SSM_CONV=0` | on | 0228: the GDN conv reads its window from the conv state and x instead of a written concat (prefill ubatches only; patterns `concat_ssm_conv`, `_silu`, `_bias_silu`); `=0` also restores the v59 graph_optimize order | EXACT | - |
| `GGML_VK_MMID_PRECAST_B=0` | on | 0231: AMD coopmat1 mul_mat_id converts B to f16 once (shared by gate and up) and runs the f16-B twin of the shader, instead of converting in every tile; q4_K / q5_0 / q8_0 / q6_K / iq3_s experts only | EXACT | - |
| `GGML_VK_MMID_TILE_BM128=0` | on | 0232: AMD coopmat1 mul_mat_id uses a BM 128 x BN 64 tile when m > 64 and n > 64 (upstream keeps the large tile off on AMD); same types as above | EXACT | - |
| `GGML_VK_DENSE_TILE_M16=0` | on | 0234: AMD coopmat1 dense matmul with m <= 16 (the qwen4exp hc inject, m = 4) on a BM 16 x BN 16 tile instead of the s tile; q4_K / q8_0 A, split_k 1 | EXACT | - |
| `GGML_VK_DENSE_TILE_B2=1`, `GGML_VK_DENSE_TILE_B2_XL=1` | off, off | 0233: AMD coopmat1 dense tile plan for q4_K / q5_0 / q6_K with m <= 384 on BM 64 x BN 128 (split_k 1 only); `_XL=1` also m >= 2048 on BM 128 x BN 256. Off because the hc tile is 2.6x slower on gfx1151 for q4_K 320 x 10240 (137.5 -> 356.4 ms per 2048-token ubatch); A/B only | op tests bit-identical; not covered by the package gate | - |
| `GGML_VK_QSA_SCORE_EPILOGUE=0` | on | 0235: the QSA indexer score is one node (`GGML_OP_QSA_SCORE`): ReLU, the ordered 4-head sum and the block bias happen in the coopmat1 store of the score GEMM, no per-head plane. Decode / 2-token verify and non-coopmat1 backends keep the op chain; `=0` = the v59 graph node for node | EXACT | - |
| `LLAMA_QSA_POOLED_F16_MIRROR=0`, `LLAMA_QSA_POOLED_F16_MIRROR_CHECK=1` | on, off | 0240: an f16 mirror of the QSA pooled-key store, written in the same graph as the f32 rows; the indexer score GEMM at 3+ rows (prefill chunks, verify rounds) reads it instead of converting the f32 keys (half the bytes, the same f16 values). 1-2 rows and the CPU keep the f32 store. Costs half the f32 store: about 0.8 GiB at 1M (measured +779 MiB). Off as well when the score epilogue is off. `_CHECK=1` compares every mirror row with its f32 row after each graph and aborts on a difference (debug) | EXACT | - |
| `GGML_VK_MMVQ_HOIST=0`, `GGML_VK_MMVQ_HOIST_MIN_COLS=N` | on, 2 | 0236: mat-vec at 2..8 columns (the speculative verify widths): the MMVQ q4_0 / q5_0 / q8_0 / q4_K and the f32 x f32 kernels unpack A once per row instead of once per column, from N columns on (0 = off, > 8 = 8). Startup line `mmvq_hoist: on from N columns` | EXACT | - |
| `LLAMA_KQ_VIS_MIRROR=0`, `LLAMA_KQ_VIS_MIRROR_MAX_TOKENS=N`, `LLAMA_KQ_VIS_MIRROR_CHECK=1` | on, 16, off | 0237, with `LLAMA_KQ_VIS=1`: the KQ visibility planes of a single-stream cache live in a device mirror updated with the changed cells only; ubatches of <= N tokens read it instead of a per-graph upload (at 1M that upload was 16.6 MiB per round). `_CHECK=1` reads it back and aborts on any difference from a fresh fill. Startup line `kq_vis_mirror = on` | EXACT | - |
| `LLAMA_LEGACY_UNMARKED=0\|1` | 1 | 0239: how v60 treats state written by v59 and older ("unmarked": no `IXY1` mark). 1 = a disk-cache entry or a context checkpoint is loaded only when its stored token stream has no image, and its cells pass the intact-text-history check; image entries are refused (a cache miss). 0 = strict: no unmarked state is ever loaded. A raw `/slots` restore of an unmarked file is refused either way. Shown in the disk-cache startup line (`legacy unmarked states: ...`) and in `GET /disk-cache` (`legacy_unmarked`) | EXACT (cache policy, no key change) | - (= 1) |
| `GGML_VK_DISPATCH_LOG=1` | 0 | 0230: one `[vk-dispatch]` line per MUL_MAT / MUL_MAT_ID / MUL_MAT_ROWS dispatch decision: path, types, m / n / k, n_as / n_used, pipeline, split_k, mmvq / mmq; no barrier | DIAG | - |
| `LLAMA_SPEC_TRACE_IDS=1` | 0 | 0238: one `[spec-ids]` line per speculative verify round: round, n_past, the drafted ids, width, the accepted ids, draft / catch-up / verify wall time, n_att (draft positions sampled). For comparing draft proposals between two builds | DIAG | - |

## Test / debug only

`LLAMA_GDN_NO_DIRECT`, `LLAMA_SERVER_DEBUG_FAULT=slot|decode|ckpt`, `LLAMA_PLE_DUMP=<file>`, `LLAMA_KLD_DUMP=<file>`
(llama-perplexity), `GGML_TEST_*` (test-backend-ops).
