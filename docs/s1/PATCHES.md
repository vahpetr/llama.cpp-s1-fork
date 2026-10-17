# Patch list

Every change this fork carries on top of its base, numbered in the order I made them. The base is drluoto's
`strix-halo-flash-next` branch of llama.cpp (b10664, `e70802a01`), which already had Qwen3.8-Flash-Next (`qwen4exp`) support
from upstream PR #27742 and an MTP draft graph. Upstream back-ports are marked as such.

Status column:
- **deployed (vNN)**: in my builds from image vNN on (my production line), still in this tree.
- **v60**: in this tree since v60 (in production 2026-10-16 to 2026-10-17); "v60, on" / "v60, off" = the default of its knob.
- **v61**: in this tree (HEAD = v61, in production since 2026-10-17); "v61, on" = on by default.
- **lab image vNN, status unverified**: built into lab image vNN (v19-v30, before the v31 production build); I have not
  re-checked that each of these reached production unchanged.
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
| 0024 | Vulkan: seed flops_per_submit from the graph's own flops | lab image v20, status unverified |
| 0025 | Vulkan: graph_optimize window as GGML_VK_OPT_WINDOW (default 20) | lab image v20, status unverified |
| 0026 | Vulkan: async readback takes the UMA memcpy shortcut | in tree, gated off |
| 0027 | --kv-graph-span: cap the KV span of the worst-case reserve (GTT -5.1 GiB at 192k) | lab image v21, status unverified |
| 0028–0035 | review fixes for 0016-0027 (PLE prefetch alignment, UMA readback gate, knob parsing) | lab image v21, status unverified |
| 0036 | Vulkan: transposed CONCAT as copy + tiled copy_transpose (prefill +7.8 % at 8k) | lab image v21, status unverified |
| 0037 | qwen4exp MTP: eh_proj as one flat GEMM for >= 32 tokens (prefill +3.5 %) | lab image v21, status unverified |
| 0038 | Vulkan: flush deferred UMA host readbacks before the next compute | lab image v22, status unverified |
| 0039 | ggml-alloc: check reserve on the regrow path (segfault fix) | lab image v22, status unverified |
| 0040 | llama: indexer cache reserve honours --kv-graph-span | lab image v22, status unverified |
| 0041 | Vulkan: GGML_VK_OPT_WINDOW clamp, overflow-free bound | lab image v22, status unverified |
| 0042 | Vulkan: one ggml_vk_env_bool for boolean knobs | lab image v22, status unverified |
| 0043 | Vulkan: graph_optimize comment fix (bit-exactness claim) | lab image v22, status unverified |
| 0044 | Vulkan FA coopmat1: Br = 2 x MatBr query rows per workgroup | lab image v23, status unverified |
| 0045 | Vulkan FA coopmat1: Br=32 for prefill batches | lab image v23, status unverified |
| 0046 | test-backend-ops: FLASH_ATTN_EXT at the Flash-Next prefill shape | test |
| 0047 | Vulkan FA coopmat1: Br=32 only for n_rows >= 512 | not adopted |
| 0048 | Vulkan FA coopmat1: padded QK^T scratch stride (fewer LDS bank conflicts) | lab image v24, status unverified |
| 0049 | Vulkan FA coopmat1: GGML_VK_FA_CM1_BC knob | in tree, off |
| 0050 | qwen4exp: PLE gather as a graph input (no CPU node in the graph) | lab image v25, status unverified |
| 0051 | docs: why the input path already overlaps ubatches | doc |
| 0052 | qwen4exp: PLE row-length / index-size asserts | lab image v25, status unverified |
| 0053 | Vulkan: mul_mat_id tile chosen from rows per expert | lab image v25, status unverified |
| 0054 | Vulkan: count_experts once per ids tensor per graph | lab image v25, status unverified |
| 0055 | comment: safe ways to bound mul_mat_id's empty workgroups | comment |
| 0056 | server: adaptive draft length per slot (--spec-draft-n-max-adaptive) | in tree, off |
| 0057 | server: depth-aware cost model for the adaptive draft rule | in tree, off |
| 0058 | server: measured decode time calibrates the depth model | in tree, off |
| 0059 | sampling: build only the reachable candidate window (greedy / top-k first) | lab image v26, status unverified |
| 0060 | server: per-slot opt-in to the sampling fast path | lab image v26, status unverified |
| 0061 | Vulkan: fuse GDN decay ADD -> SOFTPLUS -> MUL (-72 dispatches per graph) | lab image v26, status unverified |
| 0062 | test-backend-ops: ADD_SOFTPLUS_MUL fusion cases | test |
| 0063 | Vulkan FA coopmat1: row_tiles = Br/MatBr generalisation | lab image v26, status unverified |
| 0064 | Vulkan FA coopmat1: (token, head) rows for the speculative verify batch | in tree, off (GGML_VK_FA_GQA_TOKENS=0 in my composes) |
| 0065 | Vulkan: fold verify tokens x GQA heads into Br=32 tiles | in tree, off |
| 0066 | test-backend-ops: token-folded verify FA cases | test |
| 0067 | comment: why (token, head) packing is dense | comment |
| 0068 | ggml-alloc: clear every alias before freeing on regrow (double-free fix) | lab image v27, status unverified |
| 0069 | Vulkan: flush pending host reads at event_record (multi-GPU, latent) | lab image v27, status unverified |
| 0070 | server: adaptive-draft acceptance update on every verified round | lab image v27, status unverified |
| 0071 | Vulkan FA: GQA_TOKENS moved to flags bit 128 | lab image v27, status unverified |
| 0072 | Vulkan FA: token-aligned packing (keeps the RDNA3 signed-zero containment) | lab image v27, status unverified |
| 0073 | Vulkan FA: split_k mirrored from the per-token geometry | lab image v27, status unverified |
| 0074 | test-backend-ops: folded-verify FA cases with a 64-row mask | test |
| 0075 | Vulkan: rows-per-workgroup knobs for dense mat-vec (GGML_VK_DMMV_RM_*) | in tree, defaults unchanged |
| 0076 | test-backend-ops: MUL_MAT_VEC at the production shapes | test |
| 0077 | speculative: MTP draft sampler on the candidate-window fast path | lab image v28, status unverified |
| 0078 | sampling: penalty-aware candidate window (FAST_TOPK_PEN) | lab image v28, status unverified |
| 0079 | ggml-cpu: get_rows in page order for large sources (bit-exact) | lab image v28, status unverified |
| 0080 | ggml-cpu: GGML_GET_ROWS_THREADS (multi-threaded gather) | in tree, off (neutral) |
| 0081 | ggml-cpu: GGML_GET_ROWS_STATS census | diagnostic |
| 0082 | Vulkan: count_experts also writes the per-expert tile prefix sum | lab image v28, status unverified |
| 0083 | Vulkan: flattened (expert, N-tile) grid for mul_mat_id | lab image v28, status unverified |
| 0084 | test-backend-ops: MUL_MAT_ID flattened-grid cases | test |
| 0085 | GGML_SCHED_PROFILE host-side scheduler timers | diagnostic |
| 0086 | test-backend-ops: legal single-expert router case | test |
| 0087 | --qsa-ratio N: force dense or a QSA ratio regardless of the GGUF | lab image v29, status unverified |
| 0088 | QSA: dense fall-through when the selection covers every key | lab image v29, status unverified |
| 0089 | QSA: tail-cell bias fix (the query's own key) | lab image v29, status unverified |
| 0090 | test-backend-ops: TOPK_QSA initializer overrun fix (ASan) | test |
| 0091 | Vulkan: GGML_VK_DISABLE_FUSION_PATTERNS / GGML_VK_FUSION_LIST | diagnostic |
| 0092 | Vulkan: GGML_VK_FUSION_COUNT per-graph fusion counts | diagnostic |
| 0093 | fused MoE router no longer vetoed by an allocation accident | lab image v30, status unverified |
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
| 0203 | disk cache: count adopt drops | deployed (v54) |
| 0204 | qsa_history_dense proof line at server INFO | deployed (v54) |
| 0205 | expert-occupancy telemetry for mul_mat_id (GGML_VK_EXPERT_OCCUPANCY) | deployed (v54), off |
| 0206 | webui: context-reserve auto-compaction | deployed (v54) |
| 0207 | webui: per-conversation context / compaction settings | deployed (v54) |
| 0208 | webui: compaction hardening (review fixes) | deployed (v54) |
| 0209 | webui: compaction rules shared with my agent harness (mid-turn continue, unfinished-reply guard) | deployed (v54) |
| 0210 | stop type at INFO in the slot release line | deployed (v54) |
| 0211 | disk cache: learned target-only descriptor per adopt source | deployed (v54) |
| 0212 | webui: compaction review fixes (round 1) | deployed (v54) |
| 0213 | webui: compaction review fixes (round 2) | deployed (v54) |
| 0214 | webui: timing chips, tool-call durations, turn line, context-bar speeds | deployed (v54) |
| 0215 | webui: strip the server-echoed prefill when joining a continuation | deployed (v55) |
| 0216 | webui: minimal permanent surface (context meter + popover, /compact, compaction hint, collapsed timing chips) | deployed (v55) |
| 0217 | webui: review fixes (whitespace-normalised echo detection + echoMismatch mark, popover pin per chat, tool chip in the block header) | deployed (v55) |
| 0218 | webui: the upstream context ring is the single context trigger (hover card, click / Enter pins; card = upstream usage details + the fork rows: prefill / decode t/s, MTP accepted / drafted, cache share, compute, reserve breakdown, compaction state + Cancel, Compact now, settings); top-right meter removed | deployed (v56) |
| 0219 | webui: review fixes (settings dialog state per conversation, dialog clicks keep the pinned card, popup cleared on ring teardown) | deployed (v56) |
| 0220 | webui: reasoning level picker in single-model mode ("+" menu -> Reasoning: Default / Off / Low / Medium / High / Max); native `reasoning_effort` low / medium / xhigh when the template supports it (High and Max differ by the thinking budget); sampling presets by thinking state (thinking T 1.0 / top_p 0.95 / top_k 20 / min_p 0 / presence 0; instruct 0.7 / 0.8 / 20 / 0 / 1.5), only for parameters the user left unset; a 400 naming the reasoning effort is retried once without it | deployed (v57) |
| 0221 | webui: review fixes (Default inherits the server's sampling, durable user overrides, per-turn template-option snapshot reused by continuations, narrow effort fallback) | deployed (v57) |
| 0222 | webui: the reasoning-effort fallback also accepts the server's HTTP 500 template refusal (llama-server answers the Jinja "Unexpected reasoning effort" exception with 500, not 400); a 400 or 500 matching the same narrow classifier is retried once without `reasoning_effort` and remembered per model; other errors unchanged | deployed (v58) |
| 0223 | webui: editable sampling presets by reasoning state (Settings -> Sampling & Penalties -> "Presets by reasoning state"): Thinking and Instruct presets (temperature / top_p / top_k / min_p / presence / repeat penalty; defaults = the model card, "Reset to model card"), "Active in this chat" line, collapsed per-level overrides for Low / Medium / High / Max (empty = inherit Thinking); precedence user Sampling values > level override > preset, Default sends nothing; the lightbulb title shows the preset source (card / custom / level override) | deployed (v58) |
| 0224 | webui: review fixes (presets follow Settings Save / Cancel, included in export / import, cleared by both reset paths; integer top_k; "Active in this chat" follows the picker's capability derivation; stricter reasoning-effort classifier; no NaN / blank sampling sent) | deployed (v58) |
| 0225 | webui: backend-sampling checkbox locked with the server's reason ("a speculative verify round needs 1+<n_max> outputs per sequence, more than the context allows" / "tensor split") only when `/props.backend_sampling_supported` is false; under speculative decoding it stays enabled with a cost note (bypasses the graph cache and the CPU fast path, not bit-exact at temperature > 0); an older server without the field keeps it enabled with a warning; the capability is enforced on the final request body after the custom JSON; reactive guard: the server's sampling-specific rejection (400 or SSE error event) unchecks the setting, toasts and is remembered per server; a locked setting is exported as false | deployed (v59) |
| 0226 | server: backend_sampling guard: at load the target context's per-sequence output limit is compared with one speculative verify round (1 + n_max); under tensor split or a limit below a verify round a request with `backend_sampling: true` gets a 400 `invalid_request_error` naming the reason before any slot is touched (was: 500 "Invalid input batch." and a cleared slot cache), and `--backend-sampling` is forced off (WRN `[bs-guard] --backend-sampling forced off: <reason>`; INF `[bs-guard] backend_sampling_supported=<0\|1> spec_type=<type> n_max=<N> outputs_per_seq=<M>` always); draft-mtp, ngram and other drafters are supported in normal configs; `/props` adds `backend_sampling_supported` and `speculative {type, n_max}`; test `tests/test-backend-sampling-guard` | deployed (v59) |
| 0227 | vulkan: hc_mix tail fusion - multi_add + trailing SCALE (qwen4exp hc stream mean, QSA pooled-key mean) and SCALE + SILU after w_down, one dispatch each; node order unchanged (B3) | v60, on (EXACT) |
| 0228 | vulkan + qwen4exp: the GDN conv reads its window from the conv state and x, no concat pass on prefill ubatches (B4) | v60, on (EXACT) |
| 0229 | QSA dump records for the EXACT oracle: `[qsa-ids]` sha256 of the ordered id stream + per-chunk records, `[qsa-score]` hashes, `[qsa-score-ties]`, `[graph-cache]` trace | v60, diagnostic, off |
| 0230 | vulkan: `GGML_VK_DISPATCH_LOG` - one line per matmul dispatch decision | v60, diagnostic, off |
| 0231 | vulkan: mul_mat_id B pre-cast to f16 once, shared by gate and up, f16-B shader twin (AMD coopmat1; q4_K / q5_0 / q8_0 / q6_K / iq3_s) (B1a); 780M: bit-identical on 90 production cases vs v59, speed within +-1 % there | v60, on (EXACT) |
| 0232 | vulkan: mul_mat_id BM 128 x BN 64 tile on AMD coopmat1 for m > 64, n > 64 (B1b); test-backend-ops MUL_MAT_ID vs CPU 890/890 | v60, on (EXACT) |
| 0233 | vulkan: dense coopmat1 tile plan for the hc shapes (BM 64 x BN 128, optional XL BM 128 x BN 256) (B2); 2.6x slower on gfx1151 for q4_K 320 x 10240, so off | v60, built, off by default, not in the gated configuration (op tests bit-identical) |
| 0234 | vulkan: BM 16 x BN 16 coopmat1 tile for dense matmul with m <= 16 (the hc inject, m = 4) (B4c); 780M q4_K m 4 x n 2048 x k 10240: 1 588 -> 1 278 us (-19.5 %) | v60, on (EXACT) |
| 0235 | qwen4exp + vulkan: the QSA indexer score as one op `GGML_OP_QSA_SCORE` (ReLU, ordered head sum and block bias in the coopmat1 store of the score GEMM) (A1) | v60, on (EXACT) |
| 0236 | vulkan: mat-vec at 2..8 columns unpacks A once per row instead of once per column (MMVQ q4_0 / q5_0 / q8_0 / q4_K, f32 x f32) (C1) | v60, on (EXACT) |
| 0237 | llama: device-resident KQ visibility mirror, updated with the changed cells only, for ubatches of <= 16 tokens (C2) | v60, on with `LLAMA_KQ_VIS=1` (EXACT) |
| 0238 | server: `LLAMA_SPEC_TRACE_IDS` - one `[spec-ids]` line per verify round (drafted / accepted ids, timings) (C3) | v60, diagnostic, off |
| 0239 | llama + server: a per-sequence state read restores the indexer cache's cell ext (M-RoPE x / y + token); before, a restored image conversation had x = y = 0 in the indexer. State carries an `IXY1` mark; unmarked (pre-v60) state is refused unless its stored tokens prove it text-only (`LLAMA_LEGACY_UNMARKED`). Changes the state format, so the disk-cache key changes. Text logits unchanged | v60 (defect fix) |
| 0240 | qwen4exp + vulkan: f16 mirror of the QSA pooled-key store, read by the score GEMM at 3+ rows (A2a); +779 MiB GTT at 1M | v60, on (EXACT) |
| 0241 | vulkan: QSA_SCORE planes of >= 2 N tiles run on `*_qsa_kr` twins whose workgroups are numbered key tile major, so the query tiles of a chunk read each pooled-key tile back to back (A2b); same per-tile program, same bits; s1 kernel test median 0.877x per call over 16 shapes | v61, on (EXACT) |
| 0242 | tests: test-cell-ext-restore uses a private temp dir per run; comments use neutral wording | v61, no code change |
| 0243 | vulkan: QSA top-k select (cell form, v2) without per-cell key rows: the key is computed from the score bits and the visibility with integer instructions; no key region in the scratch (A3); s1 select per QSA layer at 1M 7.616 -> 6.097 s | v61, on (EXACT) |

**0227-0240 apply order** on v59 (`a1b6c324e`): 0227, 0228, 0229, 0230, 0231, 0232, **0234, 0233**, 0235, 0236, 0237, 0238,
0239, 0240 (0233 sits on 0234). Applied with a strict `git am` in that order they reproduce the v60 tree (`845c6fbcd`,
`prod-v11-stage8-pkg7`). The gate covered the whole package at once, not each patch alone: the EXACT class above is the
package's, plus each patch's own unit and op tests.

**0241-0243 apply order** on v60 (`845c6fbcd`): 0242, 0241, 0243. Applied with a strict `git am` in that order they reproduce the
v61 tree (`7a438f3ba`, `prod-v11-stage8-pkg8`). 0241 and 0242 touch disjoint files, so their order does not matter. The gate
covered the package as a whole, plus an off arm per lever.
