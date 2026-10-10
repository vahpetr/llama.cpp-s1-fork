# Patch list

Every change this fork carries on top of its base, numbered in the order I made them. The base is drluoto's
`strix-halo-flash-next` branch of llama.cpp (b10664, `e70802a01`), which already had Qwen3.8-Flash-Next (`qwen4exp`) support
from upstream PR #27742 and an MTP draft graph. Upstream back-ports are marked as such.

Status column:
- **deployed (vNN)**: in my builds from image vNN on (my production line), still in this tree.
- **v54**: in this tree (HEAD = v54, going to production on 2026-10-10).
- **in tree, off**: compiled in, off unless you set its knob (docs/s1/KNOBS.md).
- **diagnostic / tool / test / comment / doc**: no effect on serving.
- **not adopted**: measured and rejected; may or may not be in this tree, never on by default.
- **QSA line**: an intermediate step of the sparse-attention work; its final form is in later rows (0118-0161).

Numbers in the descriptions are from the gate of that change on my s1 box. The per-patch files with full descriptions, gate
records and upstream status live in my private repo; ask if you want one.

| # | what | status |
|---|---|---|
| 0001 | server: keep per-round speculative recurrent-state checkpoints on the device | deployed (inert with this model) |
| 0002 | rs-rollback for QWEN3NEXT / QWEN4EXP (native recurrent rollback) | not adopted: corrupts output |
| 0003 | recurrent on-device save: throw + host fallback instead of GGML_ABORT | deployed |
| 0004 | --kv-unified purge victim = idle slot with the fewest tokens, LRU tie-break | deployed |
| 0005 | save a purged slot to the RAM prompt cache before clearing it | deployed |
| 0006 | speculative replay after a checkpoint restore accepts committed tokens (fixes a hang on recurrent targets) | deployed (v3) |
| 0007 | qwen4exp: vocabulary-pruned MTP draft head via the d2t scatter | deployed (v4) |
| 0008 | count speculative round kinds (verify / replay / fold / plain) | deployed (v5) |
| 0009 | replay keeps the live sampler; committed tokens are not re-sampled | deployed (v5) |
| 0010 | fold the next draft into the replay batch after a partial acceptance | not adopted (+1.7 %) |
| 0011 v2 | qwen4exp conv-state rollback snapshot planes without temporaries | not adopted |
| 0012 | ggml-vulkan graph_optimize: storage hazards through views (upstream base() rule) | deployed (v6) |
| 0013 | Vulkan FA coopmat1: P*V in f32 under GGML_PREC_F32 (RDNA3 signed-zero determinism) | not adopted (no gain) |
| 0014 | per-decode recurrent rollback depth (llama_set_rs_rollback_depth) | deployed (v9) |
| 0015 | gated delta net writes its snapshot planes straight into the recurrent cache | deployed (v9) |
| 0016 | Vulkan MUL_MAT_ID row-id hoisting up to 512 experts (prefill +17 % at 8k) | deployed (v18) |
| 0017 | qwen4exp: no indexer KV cache when every QSA ratio is 0 (-2.4 GiB) | deployed (v18) |
| 0018 | qwen4exp: madvise(WILLNEED) prefetch of the PLE rows | deployed (v18) |
| 0019 | qwen4exp: drop a ggml_cont in build_hc_mix (-97 dispatches per decode graph) | deployed (v18) |
| 0020 | Vulkan FA coopmat1: dead-column flags from registers instead of an LDS re-scan | deployed (v18) |
| 0021 | test-backend-ops: TOPK_QSA / TOP_K at the deployed shapes | test |
| 0022 | qwen4exp: PLE get_rows as the first graph node (LLAMA_PLE_HOIST) | in tree, off (-3.5 % prefill) |
| 0023 | qwen4exp: LLAMA_PLE_DUMP row-id dump (observation only) | diagnostic |
| 0024 | Vulkan: seed flops_per_submit from the graph's own flops | deployed (v20), no speed effect |
| 0025 | Vulkan: graph_optimize window as GGML_VK_OPT_WINDOW (default 20) | deployed (v20) |
| 0026 | Vulkan: async readback takes the UMA memcpy shortcut | in tree, gated off |
| 0027 | --kv-graph-span: cap the KV span of the worst-case reserve (GTT -5.1 GiB at 192k) | deployed (v21) |
| 0028–0035 | review fixes for 0016-0027 (PLE prefetch alignment, UMA readback gate, knob parsing) | deployed (v21) |
| 0036 | Vulkan: transposed CONCAT as copy + tiled copy_transpose (prefill +7.8 % at 8k) | deployed (v21) |
| 0037 | qwen4exp MTP: eh_proj as one flat GEMM for >= 32 tokens (prefill +3.5 %) | deployed (v21) |
| 0038 | Vulkan: flush deferred UMA host readbacks before the next compute | deployed (v22) |
| 0039 | ggml-alloc: check reserve on the regrow path (segfault fix) | deployed (v22) |
| 0040 | llama: indexer cache reserve honours --kv-graph-span | deployed (v22) |
| 0041 | Vulkan: GGML_VK_OPT_WINDOW clamp, overflow-free bound | deployed (v22) |
| 0042 | Vulkan: one ggml_vk_env_bool for boolean knobs | deployed (v22) |
| 0043 | Vulkan: graph_optimize comment fix (bit-exactness claim) | deployed (v22) |
| 0044 | Vulkan FA coopmat1: Br = 2 x MatBr query rows per workgroup | deployed (v23) |
| 0045 | Vulkan FA coopmat1: Br=32 for prefill batches | deployed (v23) |
| 0046 | test-backend-ops: FLASH_ATTN_EXT at the Flash-Next prefill shape | test |
| 0047 | Vulkan FA coopmat1: Br=32 only for n_rows >= 512 | not adopted |
| 0048 | Vulkan FA coopmat1: padded QK^T scratch stride (fewer LDS bank conflicts) | deployed (v24) |
| 0049 | Vulkan FA coopmat1: GGML_VK_FA_CM1_BC knob | in tree, off |
| 0050 | qwen4exp: PLE gather as a graph input (no CPU node in the graph) | deployed (v25) |
| 0051 | docs: why the input path already overlaps ubatches | doc |
| 0052 | qwen4exp: PLE row-length / index-size asserts | deployed (v25) |
| 0053 | Vulkan: mul_mat_id tile chosen from rows per expert | deployed (v25) |
| 0054 | Vulkan: count_experts once per ids tensor per graph | deployed (v25) |
| 0055 | comment: safe ways to bound mul_mat_id's empty workgroups | comment |
| 0056 | server: adaptive draft length per slot (--spec-draft-n-max-adaptive) | in tree, off |
| 0057 | server: depth-aware cost model for the adaptive draft rule | in tree, off |
| 0058 | server: measured decode time calibrates the depth model | in tree, off |
| 0059 | sampling: build only the reachable candidate window (greedy / top-k first) | deployed (v26) |
| 0060 | server: per-slot opt-in to the sampling fast path | deployed (v26) |
| 0061 | Vulkan: fuse GDN decay ADD -> SOFTPLUS -> MUL (-72 dispatches per graph) | deployed (v26) |
| 0062 | test-backend-ops: ADD_SOFTPLUS_MUL fusion cases | test |
| 0063 | Vulkan FA coopmat1: row_tiles = Br/MatBr generalisation | deployed (v26) |
| 0064 | Vulkan FA coopmat1: (token, head) rows for the speculative verify batch | in tree, off (GGML_VK_FA_GQA_TOKENS=0 in my composes) |
| 0065 | Vulkan: fold verify tokens x GQA heads into Br=32 tiles | in tree, off |
| 0066 | test-backend-ops: token-folded verify FA cases | test |
| 0067 | comment: why (token, head) packing is dense | comment |
| 0068 | ggml-alloc: clear every alias before freeing on regrow (double-free fix) | deployed (v27) |
| 0069 | Vulkan: flush pending host reads at event_record (multi-GPU, latent) | deployed (v27) |
| 0070 | server: adaptive-draft acceptance update on every verified round | deployed (v27) |
| 0071 | Vulkan FA: GQA_TOKENS moved to flags bit 128 | deployed (v27) |
| 0072 | Vulkan FA: token-aligned packing (keeps the RDNA3 signed-zero containment) | deployed (v27) |
| 0073 | Vulkan FA: split_k mirrored from the per-token geometry | deployed (v27) |
| 0074 | test-backend-ops: folded-verify FA cases with a 64-row mask | test |
| 0075 | Vulkan: rows-per-workgroup knobs for dense mat-vec (GGML_VK_DMMV_RM_*) | in tree, defaults unchanged |
| 0076 | test-backend-ops: MUL_MAT_VEC at the production shapes | test |
| 0077 | speculative: MTP draft sampler on the candidate-window fast path | deployed (v28) |
| 0078 | sampling: penalty-aware candidate window (FAST_TOPK_PEN) | deployed (v28) |
| 0079 | ggml-cpu: get_rows in page order for large sources (bit-exact) | deployed (v28) |
| 0080 | ggml-cpu: GGML_GET_ROWS_THREADS (multi-threaded gather) | in tree, off (neutral) |
| 0081 | ggml-cpu: GGML_GET_ROWS_STATS census | diagnostic |
| 0082 | Vulkan: count_experts also writes the per-expert tile prefix sum | deployed (v28) |
| 0083 | Vulkan: flattened (expert, N-tile) grid for mul_mat_id | deployed (v28) |
| 0084 | test-backend-ops: MUL_MAT_ID flattened-grid cases | test |
| 0085 | GGML_SCHED_PROFILE host-side scheduler timers | diagnostic |
| 0086 | test-backend-ops: legal single-expert router case | test |
| 0087 | --qsa-ratio N: force dense or a QSA ratio regardless of the GGUF | deployed (v29+) |
| 0088 | QSA: dense fall-through when the selection covers every key | deployed |
| 0089 | QSA: tail-cell bias fix (the query's own key) | deployed |
| 0090 | test-backend-ops: TOPK_QSA initializer overrun fix (ASan) | test |
| 0091 | Vulkan: GGML_VK_DISABLE_FUSION_PATTERNS / GGML_VK_FUSION_LIST | diagnostic |
| 0092 | Vulkan: GGML_VK_FUSION_COUNT per-graph fusion counts | diagnostic |
| 0093 | fused MoE router no longer vetoed by an allocation accident | deployed (v31) |
| 0094 | ggml-alloc: KEEP_ALIVE flag on the MoE router logits (deterministic fusion) | deployed (v31) |
| 0095–0099 | Vulkan: top-k QSA fusion gate and overlap-veto fixes | QSA line |
| 0100–0103 | radix select: deterministic tie-break at the threshold | QSA line |
| 0104–0106 | Vulkan: presized QSA scratch (GGML_VK_QSA_SCRATCH_PRESIZE) | deployed later (QSA line) |
| 0107–0112 | GGML_OP_QSA_TOP_K: score + block map + mask -> top-k ids in one op | deployed later (QSA line) |
| 0113 | comment: node-diff pairing note | comment |
| 0114 | Vulkan: radix select scratch with the Vulkan memory model | QSA line |
| 0115–0117 | test-backend-ops: -inf mask and width == n_kv selection cases | test |
| 0118 | LLAMA_QSA_CHUNK: run a QSA layer in chunks of <= C queries | deployed later (QSA line) |
| 0119 | ggml_get_rows_type: type-preserving get_rows (q8_0 gather) | QSA line |
| 0120 | compute-buffer regrow hygiene knobs (LLAMA_RESERVE_RUNTIME_SHAPE and 4 more) | deployed later (QSA line) |
| 0121 | LLAMA_QSA_POOLED_CACHE: persistent pooled block keys | deployed later (QSA line) |
| 0122 | LLAMA_QSA_PATH / LLAMA_QSA_CHUNK_PAD knobs | in tree, off |
| 0123 | LLAMA_QSA_MIN_KV (dense below N) and LLAMA_QSA_GATHER_MAX_WINDOWS | deployed (v38+) |
| 0124 | GGML_OP_FLASH_ATTN_EXT with kv ids: indexed flash attention (CPU + Vulkan coopmat1) | deployed later via A4 (0155) |
| 0125 | GGML_OP_QSA_BIAS + LLAMA_QSA_COMPACT_BIAS (no bias plane) | deployed (v33) |
| 0126 | prompt checkpoints no longer copy the MTP draft KV | deployed (v33) |
| 0127 | server GTT admission: defer a prefill chunk while GTT would exceed a bar | in tree, on in my 1M/524k composes |
| 0128 | server restore log line per prompt task | deployed (v33) |
| 0129 | checkpoint eviction keeps the restored checkpoint | deployed (v33) |
| 0130 | GGML_OP_KQ_MASK + flash attention over cell/token metadata | deployed (v35) |
| 0131 | LLAMA_KQ_VIS=1: hybrid memory fills metadata only, no mask plane | deployed (v35) |
| 0132 | prefill chunking while another slot decodes | in tree, off |
| 0133 | LLAMA_RESERVE_RUNTIME_SHAPE only with --kv-unified | deployed (v34) |
| 0134 | kq_vis startup proof line at WARN | deployed (v34) |
| 0135 | LLAMA_KQ_VIS_PROF: kqvis-prof timing lines | diagnostic |
| 0136 | exact fast LLAMA_KQ_VIS metadata fill | deployed (v35) |
| 0137 | back-port of upstream #27952: int8 coopmat1 MMQ for RDNA3/RDNA4 | in tree, off in my composes |
| 0138 | PLE rows by direct reads (LLAMA_PLE_DIRECT) | not adopted |
| 0139 | MTP draft attention window | not adopted |
| 0140 | back-port of upstream #29227: int64 strides in ggml_permute (> 2 GiB views) | deployed (v36) |
| 0141 | back-port of upstream #29591: prealloc_y reuse record cleared correctly | deployed (v36) |
| 0142 | back-port of upstream #28333: MTP carrier zeroing (LLAMA_MTP_CARRIER_ZERO) | in tree, off |
| 0143 | back-port of upstream #29638: stop accepting draft tokens after an EOG | deployed (v36) |
| 0144 | back-port of upstream #29280: skip identical descriptor-set updates | deployed (v36) |
| 0145 | pattern port of upstream #28896: RMS_NORM+MUL fusion through a reshape | deployed (v36) |
| 0146 | LLAMA_ROPE_PRECISE: rope angles in double | not adopted |
| 0147 | MTP carrier restored on the per-round checkpoint fallback | deployed (v36) |
| 0148 | --spec-draft-n-max 0 crash fix | deployed (v36) |
| 0149 | diagnostic knobs for the int8 MMQ path | in tree, off |
| 0150 | tool llama-mtp-replay: teacher-forced replay with logit capture | tool |
| 0151 | LLAMA_QSA_WHOLE_BLOCKS: block-level QSA selection as in the HF reference | in tree, off |
| 0152 | tool llama-qsa-frozen + LLAMA_QSA_INDEXED_SORT | tool / off |
| 0153 | QSA top-k v2: distributed radix threshold select, byte-identical to v1 (+22.8 % decode) | deployed (v37) |
| 0154 | QSA pooled-key remap without the copy (bit-identical, -8.3 % ms/token) | deployed (v38) |
| 0155 | A4: indexed prefill + no KQ plane for QSA layers + 1M admission accounting | deployed (v39) |
| 0156 | int8 MMQ q16 mode | not adopted |
| 0157 | A4 merged onto the pool remap | deployed (v39) |
| 0158 | incremental QSA block census + store plan (LLAMA_QSA_CENSUS_INC) | deployed (v40) |
| 0159 | sparse MTP draft: the draft block runs QSA with its own indexer | superseded by 0182-0185 |
| 0160 | MTP pre_fc_norm_hidden as a joint RMSNorm (LLAMA_MTP_HNORM) | not adopted |
| 0161 | incremental QSA inputs (device-resident mirrors, LLAMA_QSA_INPUTS_INC) | deployed (v49) |
| 0162 | webui: agentic chat prompt stays a pure extension of the cached one | deployed (v41) |
| 0163 | disk tier of the prompt cache (--cache-disk*) | deployed (v42) |
| 0164 | deep recurrent checkpoints in the disk cache (branch / copy rewinds) | deployed (v43) |
| 0165 | block-deduplicated state streams in the disk cache | deployed (v44) |
| 0166 | save slots on exit; disk_cache_* in /metrics; GET /disk-cache | deployed (v45) |
| 0167 | webui: rebuild tool-result audio | deployed (v45) |
| 0168 | review fixes for 0162-0167 (sealed cache files, fingerprint, LoRA guard) | deployed (v45) |
| 0169 | webui: never send the system-message placeholder | deployed (v46) |
| 0170 | webui: confirm a system-message change with its re-read cost | deployed (v46) |
| 0171 | review fixes for 0170 | deployed (v46) |
| 0172 | webui: frozen per-conversation tool list | deployed (v47) |
| 0173 | disk cache fingerprint = loaded library content + server state-source hash | deployed (v47) |
| 0174 | slot-state correctness fixes, series A | deployed (v48) |
| 0175 | slot-state preservation fixes, series B | deployed (v48) |
| 0176 | /slots truthfulness fixes, series C | deployed (v48) |
| 0177 | webui: attachment storage + pre-encode fixes | deployed (v48) |
| 0178 | disk-cache compatibility step 1: state descriptor, namespaces, --cache-disk-adopt | deployed (v48) |
| 0179 | per-round speculative timing telemetry | deployed (v49) |
| 0180 | select-path memory: in-place ReLU, LLAMA_QSA_CHUNK 128 | deployed (v49) |
| 0181 | attention-free MTP draft catch-up on prefill (fill -29.6 %) | deployed (v49) |
| 0182 | sparse MTP draft (LLAMA_MTP_QSA), rebased | deployed (v50) |
| 0183 | sparse MTP draft on the attention-free catch-up (decode 524k -18.1 %) | deployed (v50) |
| 0184 | LLAMA_MTP_QSA_PHASE reference block phase + restore fix | deployed (v50), phase off |
| 0185 | target-only adoption + unified-pool safety for the disk cache | deployed (v50) |
| 0186 | disk cache compatibility key v2 (runtime measurement) | deployed (v51) |
| 0187 | compat key v2 review fixes + envelope v2 | deployed (v51) |
| 0188 | compat key v2 review fixes, draft_partial carry | deployed (v51) |
| 0189 | MTP draft KV type (-ctkd / -ctvd) | in tree, off |
| 0190 | --spec-draft-policy tput | in tree, off |
| 0191 | tput policy review refinements | in tree, off |
| 0192 | graph + allocation-plan cache (LLAMA_GRAPH_CACHE) | deployed (v52) |
| 0193 | graph-cache set_input audit + end-to-end test | deployed (v52) |
| 0194 | graph-cache CHECK oracles + failure hygiene | deployed (v52) |
| 0195 | graph-cache memory sizing + transition tests | deployed (v52) |
| 0196 | graph-cache CHECK=3 stale-input oracle | deployed (v52) |
| 0197 | multi-slot: MTP catch-up contiguity assertion | deployed (v53) |
| 0198 | multi-slot support envelope (LLAMA_QSA_HOMOGENEOUS) | deployed (v53) |
| 0199 | QSA partial-history dense fallback (LLAMA_QSA_HISTORY_DENSE) | deployed (v53), on |
| 0200 | homogeneous QSA routing classes + concurrent admission reserve | deployed (v53) |
| 0201 | multi-slot review round 3: routing guard, operation-start gate | deployed (v53) |
| 0202 | per-sequence state portable across the stream count (np1 <-> np2) | deployed (v53) |
| 0203 | disk cache: count adopt drops | v54 |
| 0204 | qsa_history_dense proof line at server INFO | v54 |
| 0205 | expert-occupancy telemetry for mul_mat_id (GGML_VK_EXPERT_OCCUPANCY) | v54, off |
| 0206 | webui: context-reserve auto-compaction | v54 |
| 0207 | webui: per-conversation context / compaction settings | v54 |
| 0208 | webui: compaction hardening (review fixes) | v54 |
| 0209 | webui: compaction rules shared with my agent harness (mid-turn continue, unfinished-reply guard) | v54 |
| 0210 | stop type at INFO in the slot release line | v54 |
| 0211 | disk cache: learned target-only descriptor per adopt source | v54 |
| 0212 | webui: compaction review fixes (round 1) | v54 |
| 0213 | webui: compaction review fixes (round 2) | v54 |
| 0214 | webui: timing chips, tool-call durations, turn line, context-bar speeds | v54 |
