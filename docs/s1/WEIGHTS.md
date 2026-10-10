# Weights: what I run, why, and how it was checked

I upload no weights. Everything below is a third-party upload on Hugging Face (mradermacher, SC117, dzannotti, kingjones777,
orcarouter, unsloth) under its own license and model card; you download it yourself. The setup scripts
(`deploy/s1/setup-brain.sh`, `deploy/balkon/setup-hands.sh`) do the downloads, the sha256 checks and the local re-encode; this
page explains the choices. Hashes are the files' full sha256 as of 2026-10-10 (they equal the HF LFS ids).

The model family is Qwen3.8-Flash-Next (architecture `qwen4exp`): 48 layers (12 x [3 Gated-DeltaNet + 1 attention]), 512
experts top-10 + 1 shared, 125B parameters + a 51B per-layer n-gram embedding table (PLE) + a 4B MTP head, ~6B active per token,
native context 262 144.

## 1. Brain (s1, Ryzen AI Max+ 395, 128 GB)

| file | repo | size | sha256 (first 16) |
|---|---|---|---|
| `Qwen3.8-Flash-Next-Uncensored.i1-Q4_K_M.gguf` (concatenation of `.part1of3..part3of3`) | `mradermacher/Qwen3.8-Flash-Next-Uncensored-i1-GGUF` | 119.15 GB | `9a3af985dcc9ae3f` |
| `Qwen3.8-Flash-Next-Uncensored.mmproj-f16.gguf` (vision) | `mradermacher/Qwen3.8-Flash-Next-Uncensored-GGUF` | 0.90 GB | `4fee3582716fe691` |
| `Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf` (MTP draft head) | `dzannotti/Qwen3.8-Flash-Next-MTP-GGUF` | 2.62 GB | `1f2a6991237d5412` |

**Why this quant.** i1-Q4_K_M is 5.39 bits per weight over the whole file. The PLE table (32.8 GiB, Q5_0: its rows are 160 wide,
so K-quants fall back to Q5_0) stays in host RAM through mmap; the rest (78.2 GiB) goes to the GPU. With a 1M-token q8_0 KV,
the QSA indexer cache and the draft, GTT sits at 102.7 GiB, which leaves the OS ~14 GiB of the 124 GiB it sees, most of it
page cache for the PLE table. By my arithmetic an i1-Q5_K_M trunk (89 GiB) would not fit next to the 1M KV; i1-IQ4_XS (64 GiB
trunk) fits easily, but Q4_K_M fit and I never needed to step down.

**Head and embedding stay as shipped** (`output.weight` Q6_K, `token_embd.weight` Q4_K). I tried the hands' Q8_0 head trick on
the brain (2026-10-10, single probe): P(EOS) at a known mid-code stop went 0.005 -> 0.002, but the brain had no early stops there
to begin with (0/3 either way), the "cut reply in the history" cascade was unchanged, GTT +147 MiB, decode -5 % and -14 % in two
noisy runs. No behavioural gain, so not adopted.

**Uncensored vs official.** orcarouter's abliteration of the official model. On a paired long-context test (same quant family,
same flags, 2026-09-30) the official model scored 7.1 points lower pooled over 262k and 524k (-7.8 points at 524k, McNemar
p 0.008), and +3.4 points (not significant) on 8k/32k controls. The two files come from different imatrix runs, so this is
"not worse at depth", not a clean abliteration effect. I kept the uncensored one.

**Draft head.** The MTP head is exported from the official checkpoint (the abliteration changed four MTP tensors; the official
head drafts the uncensored target fine: acceptance 0.77-0.95 depending on depth and text). My servers run a vocabulary-pruned
copy of the same head (65 536 of 248 320 output rows, fork patch 0007, `mtp-...Q4_K_M.v65536-v5.gguf`, sha `afeabdf62af8227b`).
The ranking behind the pruning is built partly from my own traffic, so I do not publish it. The full-vocabulary dzannotti file
loads the same way; I have not measured the speed difference on the current build, so treat the README decode numbers as an
upper bound for it.

## 2. Hands (balkon, Ryzen 9 7940HS + Radeon 780M, 96 GB)

| file | repo | size | sha256 (first 16) |
|---|---|---|---|
| `IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf` (trunk) | `SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF` | 55.06 GB | `40f369eca301ec6e` |
| `IQ3_S/...-IQ3_S-00002-of-00002.gguf` (the PLE table) | same | 28.80 GB | `316b46f3a2dbd68c` |
| `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (source of the Q8_0 head + embedding) | `kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF` | 4.14 GB | `25565dac32005bf4` |
| mmproj + MTP draft | as for the brain | | |
| result: `IQ3_S-hc8-o8/...-IQ3_S-hc8-o8-00001-of-00002.gguf` | made locally | 54.96 GB | `4b1be1167cf5303c` |

**What SC117's file is.** ISTA-DASLab's GSQ-RCO quantisation (learned grids, per-tensor bit allocation, imatrix from reasoning
traces) with orcarouter's abliteration transplanted into 144 residual-writer tensors. My header audit (2026-10-07): the
transplant re-encodes `ssm_out` / `attn_output` / `ffn_down_shexp` to Q8_0, and the IQ3_S tier's `ffn_down_exps` are not
GSQ-trained bytes: 39 layers are IQ4_NL from the donor and 9 re-encoded Q2_0. Gate/up experts are byte-identical to ISTA's.
Shard 2 (the PLE table, IQ4_NL) is identical in every tier, so one copy serves them all. The GGUF carries
`compress_ratios [0,0,0,4,...]`, so always pass `--qsa-ratio` explicitly.

**Re-encode 1: `hc_*` BF16 -> Q8_0 ("hc8").** GSQ leaves the 290 hyper-connection tensors in BF16 (1.17 GiB read per token).
The 780M has no bf16 math. Q8_0 saves 0.56 GiB and measured +1-5 % decode.

**Re-encode 2: `output.weight` + `token_embd.weight` -> Q8_0 ("o8").** The hands ended replies in mid-code. Diagnosis
(2026-10-10, chat replay at the stop points): the server stopped cleanly on the model's own end-of-turn token; the bias follows
bit depth, not the runtime, QSA, sampling or tools.

| weights | P(EOS) at the mid-code stop | short replies before that point (seeds 1-3) |
|---|---|---|
| IQ3_XXS-hc8 (my hands until 2026-10-10) | 0.074-0.090 | 3/3 |
| IQ3_S-hc8 | 0.021 | 1/3 |
| IQ3_S-hc8-o8 | 0.001 | 0/3 |
| brain i1-Q4_K_M (reference) | 0.005 | 0/3 |

The head bytes come from a Q8_0 export of the official model: the abliteration does not touch `output.weight` (byte-identical in
the official and uncensored Q8_0 files); `token_embd.weight` is therefore the official one, not the abliterated one. Once a cut
reply sits in the history every weight set (brain included) tends to announce and stop again; that part is fixed on the client
side (webui compaction rules, patch 0209), not by weights.

**Why IQ3_S and not IQ3_XXS.** Bit depth drives the EOS bias (4x lower per step). Cost: decode 17.0 vs 18.0 t/s (-5.6 %), prefill
126 vs 129 t/s, +7.2 GiB GTT at equal context, so the shared pool shrinks from 1.25M to 917 504 cells.

**Why not Qwen3.6-35B-A3B** (faster per token on this box: 31 vs 18 t/s decode): on my 40-task agent coding bench it solved 35/40
vs 38/40 for GSQ IQ3_XXS-hc8 and took 19 % more wall time per task (more turns, ~2x the completion tokens, more tool errors).

**Exact commands** (what `setup-hands.sh` runs; one pass does both re-encodes, I made mine in two passes with identical result):

```sh
python3 tools/s1/gguf_hc_q8.py IQ3_S/<P>-IQ3_S-00001-of-00002.gguf IQ3_S-hc8-o8/<P>-IQ3_S-hc8-o8-00001-of-00002.gguf \
  --from mtp-Qwen3.8-Flash-Next-Q8_0.gguf --tensors output.weight,token_embd.weight
ln IQ3_S/<P>-IQ3_S-00002-of-00002.gguf IQ3_S-hc8-o8/<P>-IQ3_S-hc8-o8-00002-of-00002.gguf     # PLE shard unchanged
```

Checks: same tensor list and shapes as the input; exactly 290 `hc_*` tensors changed type BF16 -> Q8_0 and the two head tensors
changed to Q8_0; nothing else changed type; sha256 of shard 1 = `4b1be1167cf5303c`; shard 2 is a hardlink.

## 3. What was measured on what

| weights | image | what | kind |
|---|---|---|---|
| brain i1-Q4_K_M + pruned MTP | v53 | speed suite, needles 32k-213k, exactness census (docs/s1/BENCHMARKS.md section 1) | suite run, single rep |
| brain i1-Q4_K_M + pruned MTP | v52 | 1M fill + decode per 64k step | single run |
| brain, every fork change | v9-v53 | paired gates: token / state-hash identity, census, speed with noise band | gates |
| brain uncensored vs official | v35 | long-context accuracy 262k / 524k, 244 items | paired, informational |
| brain Q8 head (o8) | v53 | EOS probe | single probe |
| hands GSQ tiers vs A3B | v40k-v48k | speed 1k-32k, 40-task coding bench | 1 rep per arm |
| hands IQ3_S-hc8-o8 | v53k | EOS probe, speed, GTT per pool size | probe; the 40-task gate on o8 was pending when I wrote this |

Accuracy at the 1M tier is not gated: one long-context probe scored 76.8 % at 1M. Use the 1M window for recall over very long
inputs, and check answers that depend on it.

## 4. What I would try next

A Q8_0 head only pays off on low-bit quants (about 3.5 bpw and below); at Q4_K_M it did nothing measurable. For the brain the open
question is a custom mix (Q5_K experts + an IQ4_NL PLE table), which would fit the GTT budget but needs a 360 GB BF16 download to
build; I have not done it.
