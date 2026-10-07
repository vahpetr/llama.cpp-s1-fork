# Graph cache: set_input audit ([TAG_GRAPH_CACHE])

With `LLAMA_GRAPH_CACHE=<K>` up to K allocated graphs share one scheduler's compute buffers. A cached graph
that runs after another graph finds that graph's bytes, or older stale bytes, in its own compute-buffer
inputs. So every input that lives in the compute buffer must be rewritten in full by each `set_input`.
Inputs in persistent buffers (owned by a memory module) are only safe if their contents do not depend on
which graph ran last. The audit covers fork commit a21e11e7b. Paths are relative to `src/`.

## Verdicts

| Input (set_input / can_reuse) | Where | Writes | Verdict |
|---|---|---|---|
| embd | llama-graph.cpp:107 / 123 | tokens or embd (n_tokens rows) | fully rewritten each call |
| embd_h | llama-graph.cpp:132 / 155 | tokens or embd, h | fully rewritten each call |
| pos | llama-graph.cpp:165 / 187 | pos | fully rewritten each call |
| out_ids | llama-graph.cpp:237 / 264 | out_ids (n_outputs = whole tensor) | fully rewritten each call |
| rs | llama-graph.cpp:367 / 383 | s_copy [n_rs] | fully rewritten each call (`head`, `rs_z` keyed in can_reuse) |
| attn_kv (+ KQ_VIS) | llama-graph.cpp:508 / 533 | k_idxs, v_idxs, kq_mask or kq_vis cells/toks, k_rot, v_rot | fully rewritten each call |
| attn_kv_hyb, attn_k, attn_kv_msa | llama-graph.cpp:553, 572, 602 | idxs, mask | fully rewritten each call |
| attn_k_dsa, dsa_iswa, kv_iswa, k_iswa | llama-graph.cpp:628-782 | idxs, masks, rot | fully rewritten each call (`->buffer` guards skip only tensors not in the graph) |
| dsv4 (+ raw) | llama-graph.cpp:1048-1102, fill 932 | 10 index arrays per comp, mask, rot | fully rewritten each call (from the per-ubatch plan) |
| mem_hybrid / _k / _iswa | llama-graph.cpp:1154-1315 | attn idxs, mask or kq_vis, rot, s_copy | fully rewritten each call |
| KV helpers: set_input_k_idxs / v_idxs / pos_bucket / k_rot / v_rot | llama-kv-cache.cpp:1500, 1516, 1907, 1932, 1941 | whole tensor | fully rewritten each call |
| KQ mask fill | llama-kv-cache.cpp:1600-1724 | every row | fully rewritten each call (the row-copy shortcut copies a row written earlier in the same call) |
| KQ_VIS fill `llama_kv_cells_fill_kq_vis` | llama-kv-cache.cpp:1841 | toks for all n_tokens, 4 x n_kv cell planes per stream | fully rewritten each call (a false return asserts) |
| la (minimax-01) | models/minimax-01.cpp:71 / 174 | slopes, decays | fully rewritten each call |
| msa (minimax-m3) | models/minimax-m3.cpp:96 / 121, kv-cache-msa 285-358 | pos slots, cell_blk, mask, bias | fully rewritten each call |
| qsa (qwen4exp), compute-buffer tensors | models/qwen4exp.cpp:1557 / 1575; hybrid-idx 1120, 3279, 3390 | k_idxs, cell_blk, blk_cells, blk_pos, bias or qsa_blk/qsa_qry, dirty_* | fully rewritten each call |
| qsa device mirrors (`LLAMA_QSA_INPUTS_INC=1`) | hybrid-idx 2860-2992 (`llama_qsa_mirror_update`) | cell_blk / remap / block table, changed runs only | conditional → fixed by placement: the mirrors live in the store's own buffer (`mir_ctxs_bufs`, hybrid-idx 531-580), not in the compute buffer. Their dirty tracking is per store and stream, and every graph views the same bytes at offset 0. No graph can clobber them, and every set_input is followed by its compute (process_ubatch synchronizes first, llama-context `[TAG_QSA_INPUTS_INC]`). Safe as is |
| ple (qwen4exp) | models/qwen4exp.cpp:3214 / 3019 | rows | fully rewritten each call |
| **sampling (backend samplers)** | llama-graph.cpp:1348 / 1371 → llama-sampler.cpp:1338 | dist sampler `inp_uniforms` | **conditional → excluded** (see below) |
| attn_temp, pos_bucket, mean, cls, cross_embd, attn_no_cache, attn_cross, k_shift, granite switch | various | - | never reused: no can_reuse override (default false) |

## The conditional input and its fix

**Backend samplers.** `llama_sampler_dist_backend_apply` (llama-sampler.cpp:1280) pushes the newly built
graph's `uniform_*` tensor into `sctx->inp_uniforms`. That vector belongs to the sampler, not to the graph,
and `build_sampling` clears it on every build. So it always points at the most recently built graph. Suppose
cached graph A is reused after graph B was built. `backend_set_input` then writes into B's tensor, which is an
offset in the shared buffer that may alias a live tensor of A. A's own uniform keeps stale data. Its
`can_reuse` compares only sampler pointers, so it passes.

**Fix (minimal): excluded.** `process_ubatch` uses the cache only when `sampling.samplers` is empty (and
pipeline parallelism is off). Otherwise the cache is cleared and the old single-graph path runs.
A full fix would keep per-graph uniform tensors in `llm_graph_input_sampling` and rebind them before
`set_input`. That needs a sampler-interface change and is not done here.

## can_reuse side effects (probed on up to K entries per ubatch)

- **`mctx` rebinding (almost every attention input).** The memory-backed inputs rebind `mctx` to the probing
  params. `llm_graph_result::can_reuse` evaluates every input, so the chosen entry always ends up rebound to
  the current ubatch. A rejected entry keeps an mctx that it never uses before its next probe.
- **`qsa_store_plan` (qwen4exp.cpp:1639).** It commits slot assignments, but it is cached per (i_cur, ratio,
  n_kv, blk_bias). It is reached only by entries whose `allow_reuse` matched, which means the same ubatch
  shape and therefore the same key. Repeated probes are idempotent. The e2e test shows `[qsa-ids]` identical
  with the cache off and on.
- **dsv4 `get_*_plan`.** Rewrites only the reserve plans; idempotent.
- **sampling (llama-graph.cpp:1377).** `samplers[seq_id]` inserts a nullptr for a missing seq_id. This is
  pre-existing. Under the cache it is unreachable, because the cache runs only when the sampler map is empty.

## Verification
- `tests/test-graph-cache.cpp`: logits, draft tokens and `[qsa-ids]` are bit-identical across cache off, K=8
  and K=8 with `LLAMA_GRAPH_CACHE_CHECK=1`, using the prod QSA environment.
- `test-llama-archs`, `test-mtp-carrier` and `test-mtp-catchup*` pass with `LLAMA_GRAPH_CACHE=8
  LLAMA_GRAPH_CACHE_CHECK=1`.

## One-shot inputs and `LLAMA_GRAPH_CACHE_CHECK=3`

`CHECK=2` gives the fresh plan the cached graph's input bytes, so it proves same-input execution equivalence
only: an input the cached graph got stale is stale in both runs. `CHECK=3` fills the cached graph's
compute-buffer inputs with 0xA5 before its `set_inputs`, the fresh plan's with 0x5A, and prepares the fresh
inputs with a second `set_inputs`; a region either call leaves unwritten then differs.

A second `set_inputs` is idempotent except for the inputs listed by `llm_graph_input_i::get_one_shot`:

| Input | Why one-shot | CHECK=3 |
|---|---|---|
| recurrent `s_copy` (`llm_graph_input_rs`, also inside mem_hybrid / _k / _iswa) | `llama_memory_recurrent_context::s_copy` resets the pending rollback index `rs_idx` on the first call (llama-memory-recurrent.cpp:1363) | copied from the cached graph |

Other side effects of a repeated call: the QSA incremental mirrors live in the store's own buffers and the
first call cleared their dirty runs, so the second writes nothing; `qsa_store_plan` and the dsv4 plans are
cached per ubatch key. Without the copy of `s_copy`, the transitions test (rollback) aborts at its 9th hit.
