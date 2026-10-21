# Running it: two commands per box

| file | box | context | GTT after load | needs GTT |
|---|---|---|---|---|
| `s1/docker-compose.brain-1m.yml` | Ryzen AI Max+ 395, 128 GB | 1 048 576, 1 slot | 102.7 GiB, flat | 110 GiB |
| `s1/docker-compose.brain-524k.yml` | Ryzen AI Max+ 395, 128 GB | 525 312, 1 slot | 92.3 GiB, flat | ~96 GiB (I ran it at 104) |
| `balkon/docker-compose.hands.yml` | Ryzen 9 7940HS + 780M, 96 GB | 917 504 pool, 5 slots | ~73.7 GiB (75 479 MiB measured on v64k) | 78 GiB |

First set the host up ([docs/s1/HOST-TUNING.md](../docs/s1/HOST-TUNING.md)): without the GTT change none of these load.
Step by step, including the upgrade from v60-v63 (same key) or v59 (new disk-cache key) and the rollback: [docs/s1/DEPLOY.md](../docs/s1/DEPLOY.md).
Why these weights: [docs/s1/WEIGHTS.md](../docs/s1/WEIGHTS.md). Every flag and knob: [docs/s1/KNOBS.md](../docs/s1/KNOBS.md).

## The scripts

```sh
git clone https://github.com/vahpetr/llama.cpp-s1-fork && cd llama.cpp-s1-fork

deploy/s1/setup-brain.sh             # s1: 1M brain   (BRAIN_CTX=524k deploy/s1/setup-brain.sh for 524k)
deploy/balkon/setup-hands.sh         # 780M box: hands
```

Both are idempotent and resumable (re-run after an interrupted download; finished steps are skipped) and take
`--dry-run` (print every step and size, change nothing) and `--only download|encode|up`. What they do:

1. **Check** docker + the compose plugin, the Hugging Face CLI (`hf` or `huggingface-cli`; install with
   `pipx install 'huggingface_hub[cli]'` or `uv tool install 'huggingface_hub[cli]'`), free disk, the GTT limit (warns below what
   the profile needs and sets the admission bar to GTT - 3 424 MiB), the Vulkan device on the host and inside the image.
2. **Download and verify** each file with `hf download <repo> <file> --local-dir models/`, then sha256 against the values
   embedded in the script (as of 2026-10-10; equal to the HF LFS ids). A verified file is not re-hashed unless it changes.
   - The MTP draft head is dzannotti's full-vocabulary export. My production draft head is a vocabulary-pruned variant built
     from my own traffic (not published), so the README decode numbers are an upper bound for the public setup.
   - Brain: mradermacher's i1-Q4_K_M comes as three 40 GB parts. The script downloads one part, appends it, deletes it, then the
     next, so the peak is the 119 GB file + one part (~160 GB free needed in total). Plus the vision projector (0.9 GB) and the
     MTP draft head (2.6 GB).
   - Hands: SC117's IQ3_S shards (55 + 29 GB), a Q8_0 export of the official MTP head (4.1 GB, only as the source of the Q8_0
     LM head and embedding), the projector and the draft head. ~150 GB free needed; `PRUNE=1` deletes the 59 GB of sources after
     a verified re-encode.
3. **Re-encode (hands only, CPU, a few minutes):** `tools/s1/gguf_hc_q8.py` turns the 290 BF16 `hc_*` tensors into Q8_0 and
   copies `output.weight` + `token_embd.weight` from the Q8_0 head, in one pass. The script checks that exactly those 292
   tensors changed type, compares the result with my file's sha256, and hardlinks the untouched PLE shard (no 29 GB copy).
4. **Image:** `docker pull vahpetr/llama-cpp-s1-fork:v64` (s1) or `:v64k` (balkon). `BUILD=1` builds it from this tree instead.
5. **Start** the compose file, wait for `/health` (up to 15-20 min; the brain's first start pre-reads the 33 GiB PLE range),
   print the expected GTT and fill times, and smoke-test: `/props` n_ctx, one short completion, `/disk-cache` own key and entries.

Defaults: models in `deploy/<box>/models`, slots and the disk prompt cache in `deploy/<box>/slots`; override with `MODELS_DIR`,
`SLOTS_DIR`, `LLAMA_S1_IMAGE`. The scripts refuse to start a second llama-s1 container on the same GPU.

## The image

The prebuilt images are this tree built with upstream's `.devops/vulkan.Dockerfile`, target `server`, exactly as my
production images:

```sh
git archive <rev> | docker build -f .devops/vulkan.Dockerfile --build-arg APP_REVISION=<rev> --label llama.rev=<rev> \
  --target server -t vahpetr/llama-cpp-s1-fork:v64 -
```

v64 = rev `c5b3462ed` (the `prod-v11-stage9-pkg3` head; this branch adds docs, deploy files, `tools/s1` and comment scrubs on top). `:v64`, `:v64k`
and `:latest` are one image (digest `sha256:b3c12a3a7f680d9d1683edbc1b5728b285766618bfacb9aa2159d9eed891ff28`), built on the
780M box; my s1 production image is the same revision built on s1, and the gate ran on that one. Rollback: `:v63` /
`:v63k` (rev `8c1a468aa`, one image, `sha256:d16b1cdc9139573f07c380f6a5fc6913866348bb6a121593c2e52c76f538a5d4`; same
disk-cache key and slot-file format as v64), and the older `:v62` / `:v62k` (rev `e4af0a6ca`,
`sha256:4281527a03092d87200d14d18222f060aecea6b47468379a6ec24a7ac9d52a96`), `:v61` / `:v61k` (rev `7a438f3ba`,
`sha256:46fc98c6f86902b220309e172691758e9806eaf2fc411a640d4ac2f5595b7c63`),
`:v60` / `:v60k` (rev `845c6fbcd`, `sha256:794113766af1129c65e9541bcc4b9b3bac4d378fa38b38cd4d7c5f00fbf7a0d1`; same key) and
`:v59` / `:v59k` (rev `a1b6c324e`, one image,
`sha256:5da04b0d8bc24f022c28d0cce0d993c4ed2c5837527113625dbeade5699d5f92`) stay published; set
`LLAMA_S1_IMAGE=vahpetr/llama-cpp-s1-fork:v63` (or `:v63k`, `:v62`, `:v61`, `:v60`, `:v59`, ...). v60-v64 use a new disk-cache key against
v59, so read "Upgrading from v59" in docs/s1/DEPLOY.md before switching a v59 install that has stored conversations; from v60-v63
it is an image change plus one compose flag (`--spec-draft-sampling probabilistic`, new in the v64 composes; every image since
v63 accepts it, older images do not know it - drop the two lines when rolling back to v62 or older). Build with no model loaded on the GPU: on the
128 GB box a build next to a loaded model has run it out of memory. `docker compose -f <file> build` does the same from a checkout.

## By hand

The compose files work without the scripts: put the files the script would download into `models/` (names in the scripts and
in docs/s1/WEIGHTS.md), then `docker compose -f <file> up -d`.

## First start: what to expect

- **GTT after load** (`cat /sys/class/drm/card*/device/mem_info_gtt_used`): brain-1m 104 672-105 193 MiB on v59, 105 414 MiB
  on v60 (the f16 pooled-key mirror, +0.8 GiB) and 106 032 MiB after the first requests, 105 495 MiB on v61, v63 and v64 (v62: 105 414) and 105 519 MiB
  after the smoke requests (the admission bar is 109 216),
  brain-524k ~94 500 MiB (v59), hands ~73.7 GiB (75 479 MiB after the v64k switch; 76 260 MiB on v60k after a restart). On v59 the
  1M fill stayed at 105 188-105 196 MiB (per-step readings on v52, peaks on v59) because `--kv-graph-span 0` reserves the compute buffers for the full span at load;
  on v63 the server peak over a full 1M fill was 105 659 MiB. An allocation error at load means the GTT limit is too small.
- **Log proofs** (`-lv 4`): `n_ctx = 1048576` (or 525312), `freq_scale = 0.25` for target and draft, the QSA ratio line,
  `graph cache: K = 8`, `kq_vis: on`; from v60 also `qsa_score_epilogue: on`, `mmvq_hoist: on from 2 columns`,
  `kq_vis_mirror = on`, `qsa_pooled_f16_mirror: on`, and the disk-cache line `legacy unmarked states: ...`; from v61 also
  `qsa_score_kreuse: on` and `qsa_topk: v2 ...; nokeys on`; from v64 also `mmid_tile_g2a: on (...)` and `dense_tile_g1: table on
  (3 rules) (...)` (WARN lines, once per device; an `off (<reason>; ...)` line names what is missing on your device).
- **Cold fill** on s1 (one long prompt, empty cache), 1M profile: 0 -> 524k in 19.2 min, 0 -> 1M in 41.8 min on v61 (19.8 / 44
  min on v60, 25.6 / 64 min on v59), oracle probe; 0 -> 1M in 41.4 min on v63 (llama-server, 65 536-token steps; v64 not
  measured over a fill yet). Prefill ~550 t/s on short prompts, 501 t/s on a fresh 32k prompt and 417 t/s on a fresh 72k prompt on v64
  (v63: 467 / 386), ~358 t/s near 1M (v63; v60: ~330, v59: ~195 t/s).
- **Host RAM:** the PLE table lives in the page cache, so `MemAvailable` sits at a few GiB on s1 with the brain loaded. That is
  normal. Do not start a second model or a big build on the same box.
- **Disk prompt cache** (`slots/prompt-cache/<profile>`): ~15 KB per cached token (a 1M conversation is ~17 GiB). One directory
  per profile. Each build key has its own `ns-<key>` subdirectory; a new key starts empty unless you adopt the old one
  (`--cache-disk-adopt`, docs/s1/DEPLOY.md). Namespaces of other keys are left alone until they sit idle past the TTL or
  together pass half the quota; an adopted one is never removed.
- **Endpoints:** `/health`, `/metrics` (Prometheus; `disk_cache_*`), `/slots`, `GET /disk-cache`, the webui at `/`.
