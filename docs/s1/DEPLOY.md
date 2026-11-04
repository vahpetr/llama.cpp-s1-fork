# Deploying v73, step by step

The short version is two commands (`deploy/s1/setup-brain.sh`, `deploy/balkon/setup-hands.sh`, see
[deploy/README.md](../../deploy/README.md)); this page is the long version, with the upgrade, the rollback and the checks after
a switch.

| box | profile | compose file | image | context | GTT after load |
|---|---|---|---|---|---|
| s1: Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB | brain, 1M | `deploy/s1/docker-compose.brain-1m.yml` | `:v73` | 1 048 576, 1 slot | 104 538 MiB on v73 (needs the 108 GiB GTT setting) |
| s1 | brain, 524k | `deploy/s1/docker-compose.brain-524k.yml` | `:v73` | 525 312, 1 slot | not measured on v73 |
| balkon: Ryzen 9 7940HS + Radeon 780M (gfx1103), 96 GB | hands | `deploy/balkon/docker-compose.hands.yml` | `:v73k` | 917 504 pool, 5 slots | not measured on v73 |

## 1. Images

`vahpetr/llama-cpp-s1-fork` on Docker Hub:

| tag | use | revision |
|---|---|---|
| `:v73` | brain (s1), this release | `7597206a9` (`prod-v11-stage9-pkg13`) |
| `:v73k` | hands (780M), this release (the same image as `:v73`) | `7597206a9` |
| `:v72` / `:v72k` | rollback (same disk-cache key) | `608af6ea8` |

The images are built with upstream's `.devops/vulkan.Dockerfile`, target `server` ("The image" in deploy/README.md; `BUILD=1`
in the setup scripts builds it from this tree). Build with no model loaded on the GPU.

## 2. s1: the 1M brain

1. **Host.** Apply [HOST-TUNING.md](HOST-TUNING.md), section s1: `ttm.pages_limit=28311552 ttm.page_pool_size=28311552`
   (108 GiB GTT), `amd_iommu=off`, `amdgpu.lockup_timeout=600000`, the dirty-page caps and swap. Reboot and check
   `cat /sys/bus/pci/devices/<bdf>/mem_info_gtt_total` (115 964 116 992 bytes = 110 592 MiB). With less GTT use the 524k
   profile.
2. **Checkout.** `git clone https://github.com/vahpetr/llama.cpp-s1-fork && cd llama.cpp-s1-fork` (branch `s1-public`).
3. **Dry run.** `deploy/s1/setup-brain.sh --dry-run` prints every step, file and size. About 160 GB free disk is needed for the
   weights (the target comes as three 40 GB parts that the script appends one by one).
4. **Run.** `deploy/s1/setup-brain.sh`: checks docker, the Hugging Face CLI, disk, GTT and the Vulkan device, downloads and
   sha256-checks the weights ([WEIGHTS.md](WEIGHTS.md)), pulls `:v73`, starts the compose file, waits for `/health` (the first
   start pre-reads the 33 GiB PLE range) and runs the smoke checks below. For the 524k profile:
   `BRAIN_CTX=524k deploy/s1/setup-brain.sh`.
5. **By hand** instead: put the files into `deploy/s1/models/`, then
   `docker compose -f deploy/s1/docker-compose.brain-1m.yml up -d`. `MODELS_DIR`, `SLOTS_DIR` and `LLAMA_S1_IMAGE` override
   the defaults.
6. **Admission bar.** `LLAMA_GTT_ADMISSION_MIB` in the compose defaults to 109 216 MiB, the value on my host. The setup
   script sets it to your GTT size − 3 424 MiB (107 168 MiB at 108 GiB) and sets `LLAMA_GTT_USED_PATH` to your GPU's stable
   PCI path (section 6).

What to expect on v73 (1M profile): GTT 104 538 MiB after load, up to 105 364 MiB over a full 1M fill; the admission bar on my
host is 109 216 MiB. Measured speeds: [BENCHMARKS.md](BENCHMARKS.md).

## 3. balkon: the 780M hands

1. **Host.** [HOST-TUNING.md](HOST-TUNING.md), section balkon: GTT 78 GiB, `amd_iommu=off`.
2. **Run.** `deploy/balkon/setup-hands.sh --dry-run`, then `deploy/balkon/setup-hands.sh`. It downloads SC117's IQ3_S shards,
   re-encodes the BF16 hyper-connection weights and the LM head to Q8_0 with `tools/s1/gguf_hc_q8.py` (CPU, a few minutes),
   checks the result's sha256, pulls `:v73k` and starts the compose. About 150 GB free disk is needed; `PRUNE=1` deletes the
   59 GB of sources after a verified re-encode.
3. The hands compose runs five slots on one 917 504-cell pool and leaves the sparse MTP draft and the graph cache off. Top-k v3
   and GDN replay default off on gfx1103.

## 4. Upgrading

**From v60–v72: same disk-cache key.** Pull `:v73` and recreate the container with the v73 compose files (or set
`LLAMA_S1_IMAGE=vahpetr/llama-cpp-s1-fork:v73` with your compose). Stored conversations stay readable; no adopt step. The v73
composes carry `LLAMA_SPEC_OVERLAP=1`, `--cache-disk-async 1024` and `--cache-disk-incr` (none of them is in the disk-cache key).
If your compose has adopt lines for an older namespace, keep them.

**From v59 or older: the key changes.** The state format changed in v60, so v60 and later start with an empty namespace
(`DIR/ns-<key>` under `--cache-disk DIR`). A namespace that is not adopted is deleted once it is idle past
`--cache-disk-ttl-days` or once foreign namespaces exceed half of `--cache-disk-gib`. To keep it:

1. Before stopping the old server, note its key: `curl -s http://127.0.0.1:8000/disk-cache | python3 -c 'import sys, json;
   print(json.load(sys.stdin)["key"])'` (or the `ns-<key>` directory name, 16 hex digits).
2. In the compose, uncomment the two `--cache-disk-adopt` lines and put that key there. Keep the same model, draft and projector
   files: adoption is refused when the files, context size or KV types differ.
3. Switch, then check `GET /disk-cache`: `adopt` lists the old key with its `n_entries`. The old namespace is opened read-only;
   text conversations load, conversations with images are re-prefilled once, raw v59 slot files are refused.
   `LLAMA_LEGACY_UNMARKED=0` refuses every pre-v60 state.

## 5. Rollback

To v72 (same key, nothing to do in the cache):

```sh
LLAMA_S1_IMAGE=vahpetr/llama-cpp-s1-fork:v72 docker compose -f deploy/s1/docker-compose.brain-1m.yml up -d
```

v72 reads the conversations v73 saved. v72 accepts every flag of the v73 compose (the v73 knob `GGML_VK_DECODE_FUSE` is a code default).

## 6. Smoke checks

After any start or switch (the setup scripts run the first four):

```sh
P=http://127.0.0.1:8000
curl -sf $P/health && echo healthy
curl -s $P/props | python3 -c 'import sys, json; d = json.load(sys.stdin); print("n_ctx", d["default_generation_settings"]["n_ctx"])'
curl -s $P/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with one word: ready"}],"max_tokens":16,"temperature":0}' \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
curl -s $P/disk-cache | python3 -c 'import sys, json; d = json.load(sys.stdin); print(d["key"], d["n_entries"], d["adopt"])'
echo "GTT used: $(( $(cat /sys/bus/pci/drivers/amdgpu/*/mem_info_gtt_used | head -1) / 1048576 )) MiB"
docker logs llama-s1-brain-1m 2>&1 | grep -E 'headroom admission (on|disabled)'   # must say "on", never "disabled"
docker logs llama-s1-brain-1m 2>&1 | grep -E 'n_ctx *=|qsa_topk: |qsa_bias_fold: |gdn_replay: |hc_tail_fuse: |add_rms_rows: |merge plan: |prewarm_shapes: |decode_fuse: '
```

- `n_ctx` 1048576 (or 525312; the hands report 917504).
- Admission: `admission: GTT headroom admission on: bar = <N> MiB, ...` and no `headroom admission disabled` line. The
  disabled line means `LLAMA_GTT_USED_PATH` points at a file that does not exist; the server still serves without GTT
  protection on long prompts.
- v73 startup lines on gfx1151: `qsa_topk: onepass on`, `qsa_topk: v3 on (... from 2 rows ...)`, `qsa_bias_fold: on`,
  `gdn_replay: on (default on gfx1151)`, `hc_tail_fuse: on`, `add_rms_rows: on`, `qwen4exp merge plan: on (default on gfx1151)`,
  `prewarm_shapes: on`, `qsa_idx_f16_decode: off`, `decode_fuse: on (GGML_VK_DECODE_FUSE, default on)`. On the 780M the merge plan, top-k v3 and GDN replay say `off (default off: not gfx1151)`. An `off` line names the reason or the environment variable that switched it off
  ([KNOBS.md](KNOBS.md)).
- GTT must stay below the admission bar. `--kv-graph-span 0` reserves the compute buffers for the full span at load, so an
  allocation error at load means the GTT limit is too small.
- Determinism: the same greedy request twice (`"temperature":0`, `"cache_prompt":false`, `"n_predict":64` on `/completion`)
  gives the same `content`. Every finished request logs one `stop processing: ... stop=<eos|limit|word|cancel>` line.
- Debug output is off: no `[spec-ids]`, `[vk-dispatch]`, `[qsa-score]` or `MISMATCH` lines unless those knobs are set.

## 7. Things that bite

- **GPU sysfs paths.** Never hard-code `/sys/class/drm/cardN/...`: the card index follows the probe order at boot and can change
  after a reboot (card1 -> card0 on s1). A stale `LLAMA_GTT_USED_PATH` disables admission. Use the PCI address:
  `/sys/bus/pci/devices/<bdf>/` or `/sys/bus/pci/drivers/amdgpu/<bdf>/`; find it with `readlink -f /sys/class/drm/card*/device`.
  This covers `LLAMA_GTT_USED_PATH`, `mem_info_gtt_*` reads in scripts and `power_dpm_force_performance_level`
  ([HOST-TUNING.md](HOST-TUNING.md)).
- One model load per GPU. A second model, or a build, next to the loaded 1M brain has run the 128 GB box out of memory.
- Never point a test server at a production `--cache-disk` directory; give every experiment its own directory.
- No context shift: the model has recurrent layers. The client must compact before the window is full (the bundled webui does).
- The PLE table lives in the page cache, so `MemAvailable` is a few GiB on s1 with the brain loaded. That is expected.
