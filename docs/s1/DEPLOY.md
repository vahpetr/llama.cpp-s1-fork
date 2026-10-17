# Deploying v61, step by step

This is how I run the fork on my two boxes, written so you can repeat it on yours. The short version is two commands
(`deploy/s1/setup-brain.sh`, `deploy/balkon/setup-hands.sh`, see [deploy/README.md](../../deploy/README.md)); this page is the
long version, with the upgrade from v60 or v59, the rollback and the checks I run after a switch.

| box | profile | compose file | image | context | GTT after load |
|---|---|---|---|---|---|
| s1: Ryzen AI Max+ 395 (Radeon 8060S, gfx1151), 128 GB | brain, 1M | `deploy/s1/docker-compose.brain-1m.yml` | `:v61` | 1 048 576, 1 slot | 105 495 MiB on v61 (v60: 105 414), 105 519 after the smoke requests (needs the 110 GiB GTT setting) |
| s1 | brain, 524k | `deploy/s1/docker-compose.brain-524k.yml` | `:v61` | 525 312, 1 slot | ~94 500 MiB on v59; v60 / v61 not measured on this profile |
| balkon: Ryzen 9 7940HS + Radeon 780M (gfx1103), 96 GB | hands | `deploy/balkon/docker-compose.hands.yml` | `:v61k` | 917 504 pool, 5 slots | ~73 GiB (interpolated, v54-era) |

## 1. Images

`vahpetr/llama-cpp-s1-fork` on Docker Hub:

| tag | revision | digest |
|---|---|---|
| `:v61`, `:v61k`, `:latest` (one image) | `7a438f3ba` (`prod-v11-stage8-pkg8`) | `sha256:46fc98c6f86902b220309e172691758e9806eaf2fc411a640d4ac2f5595b7c63` |
| `:v60`, `:v60k` (rollback, one image; same disk-cache key as v61) | `845c6fbcd` (`prod-v11-stage8-pkg7`) | `sha256:794113766af1129c65e9541bcc4b9b3bac4d378fa38b38cd4d7c5f00fbf7a0d1` |
| `:v59`, `:v59k` (older rollback, one image) | `a1b6c324e` (`prod-v11-stage7`) | `sha256:5da04b0d8bc24f022c28d0cce0d993c4ed2c5837527113625dbeade5699d5f92` |

The published image is built on the 780M box with upstream's `.devops/vulkan.Dockerfile`, target `server`. My s1 production
image is the same revision built on s1; the EXACT gate (docs/s1/BENCHMARKS.md sections 0 and 0a) ran on that one. To build your own,
see "The image" in deploy/README.md (`BUILD=1` in the setup scripts does it). Build with no model loaded on the GPU.

## 2. s1: the 1M brain

1. **Host.** Apply [HOST-TUNING.md](HOST-TUNING.md), section s1: `ttm.pages_limit=28835840 ttm.page_pool_size=28835840`
   (110 GiB GTT), `amd_iommu=off`, `amdgpu.lockup_timeout=600000`, the dirty-page caps and swap. Reboot and check
   `cat /sys/class/drm/card*/device/mem_info_gtt_total` (118 111 600 640 bytes = 112 640 MiB). With less GTT use the 524k
   profile.
2. **Checkout.** `git clone https://github.com/vahpetr/llama.cpp-s1-fork && cd llama.cpp-s1-fork` (branch `s1-public`).
3. **Dry run.** `deploy/s1/setup-brain.sh --dry-run` prints every step, every file and its size. You need about 160 GB free for
   the weights (the target comes as three 40 GB parts that the script appends one by one).
4. **Run.** `deploy/s1/setup-brain.sh`. It checks docker, the Hugging Face CLI, disk, GTT and the Vulkan device, downloads and
   sha256-checks the weights ([WEIGHTS.md](WEIGHTS.md)), pulls `:v61`, starts the compose file, waits for `/health` (the first
   start pre-reads the 33 GiB PLE range: 15-20 min) and runs the smoke checks below. For the 524k profile:
   `BRAIN_CTX=524k deploy/s1/setup-brain.sh`.
5. **By hand** instead: put the files into `deploy/s1/models/`, then
   `docker compose -f deploy/s1/docker-compose.brain-1m.yml up -d`. `MODELS_DIR`, `SLOTS_DIR` and `LLAMA_S1_IMAGE` override
   the defaults.
6. **The admission bar.** `LLAMA_GTT_ADMISSION_MIB` defaults to 109 216 = GTT size - 3 424 MiB on my box; the setup script
   computes it from yours. Check `LLAMA_GTT_USED_PATH` points at your GPU's `card` index.

What to expect on v61 (1M profile): GTT 105 495 MiB after load and 105 519 MiB peak after the smoke requests (llama-server with the
vision projector; v60: 105 414 after load, 106 032 after its first requests); the highest value I have on v61 is 106 095 MiB
(v60: 106 357), the oracle probe's peak during its score-capture pass (without the projector). The admission bar is 109 216
MiB. The server's peak over a full 1M fill on v60 / v61 is not measured yet (v59: 105 196 MiB). A cold fill of 1M tokens
takes about 42 min, 0 -> 524k about 19 min (oracle probe numbers); prefill ~550 t/s on an 8k prompt, ~357 t/s near 1M.
Decode on v60 / v61 is not re-measured yet; on v59 it was 26.1 / 22.1 / 24.4 ms/token at 32k / 262k / 1M on my corpus
(draft on, worst of 3 windows).

## 3. balkon: the 780M hands

1. **Host.** [HOST-TUNING.md](HOST-TUNING.md), section balkon: GTT 78 GiB, `amd_iommu=off`, same idea scaled down.
2. **Run.** `deploy/balkon/setup-hands.sh --dry-run`, then `deploy/balkon/setup-hands.sh`. It downloads SC117's IQ3_S shards,
   re-encodes the BF16 hyper-connection weights and the LM head to Q8_0 with `tools/s1/gguf_hc_q8.py` (CPU, a few minutes),
   checks the result's sha256, pulls `:v61k` and starts the compose. About 150 GB free needed; `PRUNE=1` deletes the 59 GB of
   sources after a verified re-encode.
3. The hands compose runs five slots on one 917 504-cell pool and leaves the sparse MTP draft and the graph cache off (not
   gated on this box). v60's kernels are on by their defaults there too: my hands smoke after the switch showed the
   `qsa_score_epilogue: on`, `mmvq_hoist: on from 2 columns`, `kq_vis_mirror = on` and `qsa_pooled_f16_mirror` lines; v61 adds
   `qsa_score_kreuse: on` and `qsa_topk: v2 ...; nokeys on`.

## Upgrading from v60: same key, change the image

v61 keeps v60's state format and disk-cache key, so nothing in the cache needs attention: pull `:v61` (`:v61k` on the hands)
and recreate the container with the v61 compose files (or `LLAMA_S1_IMAGE=vahpetr/llama-cpp-s1-fork:v61`). If your v60
compose has the two adopt lines for the v59 namespace, keep them. On my box the switch passed the same-key cache check (13 of
13 required checks, including v60 reading entries v61 wrote). A rollback to v60 is the same in reverse. From v59, follow the
next section; it applies to v61 as it did to v60.

## 4. Upgrading from v59: the disk-cache key changes

The disk prompt cache stores conversations under `--cache-disk DIR` in one subdirectory per compatibility key,
`DIR/ns-<key>`. The key covers the model files, the context size, the KV types, the build's compute libraries and the
format of the saved state. v60 changes the state format (patch 0239 adds an `IXY1` mark), so **v60 has a new key and starts
with an empty namespace**.

**Adopt the old namespace; do not skip it.** The server treats every namespace with another key as foreign and cleans foreign
namespaces up: one that is not adopted is deleted once it has been idle past `--cache-disk-ttl-days` (14 in my composes), or
as soon as all foreign namespaces together exceed half of `--cache-disk-gib` (100 GiB with 200). A v59 1M namespace can be
larger than that on its own, so a v60 start without the adopt lines can delete it at once, and a rollback to v59 would then
find nothing. Only an adopted namespace is protected. If you may want to roll back, you can also copy the old namespace
before the switch (`cp -a --reflink=auto ns-<key> <somewhere else>`; a reflink copy costs no space on XFS / Btrfs).

0. **Update the checkout** to the v60 files: `git fetch origin && git checkout s1-public && git pull`. The commented adopt
   block exists only in the v60 composes.
1. **Before you stop v59**, note its key: `curl -s http://127.0.0.1:8000/disk-cache | python3 -c 'import sys, json;
   print(json.load(sys.stdin)["key"])'`, or look at the directory name: `ls deploy/s1/slots/prompt-cache/brain-1m/` shows
   `ns-<key>` (16 hex digits). v59's log also prints `own key <key>` at start. If there are several `ns-*` directories, use the
   key `/disk-cache` reports: that is the one v59 writes to.
2. **Edit the compose (required).** In `docker-compose.brain-1m.yml` uncomment the two lines under "upgrading from v59 or older" and put
   that key in place of `<old key>`:
   ```yaml
         - --cache-disk-adopt
         - "0123456789abcdef"     # your v59 key
   ```
   Keep the same model, draft and projector files (same paths, not re-downloaded or touched): the adoption is refused when
   the files, the context size or the KV types differ, so a 524k namespace cannot be adopted by the 1M profile.
3. **Switch.** `docker compose -f deploy/s1/docker-compose.brain-1m.yml pull && docker compose -f
   deploy/s1/docker-compose.brain-1m.yml up -d` (or `LLAMA_S1_IMAGE=vahpetr/llama-cpp-s1-fork:v60`). On SIGTERM v59 writes its
   slot to the disk cache first; `stop_grace_period` is 90 s.
4. **Check** `curl -s http://127.0.0.1:8000/disk-cache`: `key` is the new key, `adopt` lists the old one with its `n_entries`,
   and `legacy_unmarked` is `true`. The log shows `adopt source <old key>: N entries + M deep checkpoints (read-only, ...)`.

What happens to the old entries (v59 and older, "unmarked"):

- **Text conversations load.** An unmarked entry is accepted when its own stored token stream has no image and its restored
  cells form an intact text history (positions 0..n-1, no holes, no image cells). The next save of that conversation goes to
  the new namespace.
- **Conversations with images are refused** and re-prefilled once. Before v60 a restored image conversation lost the image
  tokens' position data (M-RoPE x / y) in the indexer cache; v60 does not trust state that may carry that defect.
- **Raw slot files** (`/slots?action=restore`) written by v59 are refused.
- **Strict mode:** `LLAMA_LEGACY_UNMARKED=0` in the compose environment refuses every unmarked state and checkpoint (each a
  cache miss). The default is 1.
- The old namespace is never written by v60, and it is not deleted while it is adopted. Keep the adopt lines as long as you
  might roll back or want the old conversations; once you remove them, the cleanup above applies to it (idle past the TTL, or
  foreign namespaces over half the quota).

On my box the switch was gated end to end: 24 of 24 checks passed (adopted load, new-vs-old tokens identical, re-save into the
new namespace, restart, rollback load, old namespace byte-identical afterwards).

The hands follow the same procedure with their own key (`deploy/balkon/docker-compose.hands.yml` has the same commented
block; the directory is `slots/prompt-cache/hands`).

## 5. Rollback

To v60 (same key, nothing to do in the cache): `LLAMA_S1_IMAGE=vahpetr/llama-cpp-s1-fork:v60 docker compose -f
deploy/s1/docker-compose.brain-1m.yml up -d` (`:v60k` for the hands). v60 reads the conversations v61 saved. To v59:

```sh
cd llama.cpp-s1-fork
git checkout <the commit or tag you ran v59 from> -- deploy/    # the v59 composes (image :v59 / :v59k, no adopt lines)
docker compose -f deploy/s1/docker-compose.brain-1m.yml up -d   # recreates the container on :v59
# or keep the v60 file: comment the two adopt lines out again and start it with the v59 image
LLAMA_S1_IMAGE=vahpetr/llama-cpp-s1-fork:v59 docker compose -f deploy/s1/docker-compose.brain-1m.yml up -d
curl -s http://127.0.0.1:8000/disk-cache | python3 -c 'import sys, json; d = json.load(sys.stdin); print(d["key"], d["n_entries"])'
```

The last line should show the v59 key with its entries. If v60 ran with the adopt lines, v59 finds its namespace as it left
it: v60 only read it. If v60 ran without them, the namespace may have been deleted (section 4); then restore your copy.

Conversations saved by v60 are in v60's namespace, which v59 does not read (another key). To v59 it is a foreign namespace
under the same cleanup rule: it survives the rollback only while it is younger than the TTL and the foreign namespaces stay
under half the quota. A v60 state handed to v59 directly (a raw slot file) is refused cheaply: v59 reads the mark as an
invalid stream and stops, no large allocation.

## 6. Smoke checks you can run

After any start or switch (the setup scripts run the first four):

```sh
P=http://127.0.0.1:8000
curl -sf $P/health && echo healthy
curl -s $P/props | python3 -c 'import sys, json; d = json.load(sys.stdin); print("n_ctx", d["default_generation_settings"]["n_ctx"])'
curl -s $P/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with one word: ready"}],"max_tokens":16,"temperature":0}' \
  | python3 -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
curl -s $P/disk-cache | python3 -c 'import sys, json; d = json.load(sys.stdin); print(d["key"], d["n_entries"], d["adopt"], d["legacy_unmarked"])'
echo "GTT used: $(( $(cat /sys/class/drm/card1/device/mem_info_gtt_used) / 1048576 )) MiB"   # your card index may differ
docker logs llama-s1-brain-1m 2>&1 | grep -E 'n_ctx *=|qsa_score_epilogue: |mmvq_hoist: |kq_vis_mirror = |qsa_pooled_f16_mirror: |qsa_score_kreuse: |nokeys |legacy unmarked states'
```

- `n_ctx` 1048576 (or 525312; the hands report 917504).
- The v60 startup lines: `qsa_score_epilogue: on`, `mmvq_hoist: on from 2 columns`, `kq_vis_mirror = on` (target and draft on the
  brain), `qsa_pooled_f16_mirror: on`; from v61 also `qsa_score_kreuse: on (...)` and `qsa_topk: v2 ...; nokeys on`. If one says
  off, an env variable in your compose is switching it off ([KNOBS.md](KNOBS.md), "v61 package" and "v60 package").
- GTT: 105 495 MiB after load on v61 (v60: 105 414) and about 105 500-106 000 MiB after the first requests on my box; it must stay below the
  admission bar (109 216 MiB with a 110 GiB GTT). `--kv-graph-span 0` reserves the compute buffers for the full span at load, so
  an allocation error at load means the GTT limit is too small.
- **Determinism:** send the same greedy request twice (`"temperature":0`, `"cache_prompt":false`) to `/completion` with
  `"n_predict":64` and compare `content`; they should match. Every finished request logs one
  `stop processing: ... stop=<eos|limit|word|cancel>` line.
- **Debug output is off:** no `[spec-ids]`, `[vk-dispatch]`, `[qsa-score]` or `MISMATCH` lines unless you set those knobs.

## 7. Things that bite

- One model load per GPU. A second model, or a build, next to the loaded 1M brain has run my 128 GB box out of memory.
- Never point a test server at your production `--cache-disk` directory; give every experiment its own directory.
- No context shift: the model has recurrent layers. The client must compact before the window is full (the bundled webui does).
- The PLE table lives in the page cache, so `MemAvailable` is a few GiB on s1 with the brain loaded. That is expected.
