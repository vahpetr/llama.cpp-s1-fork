#!/usr/bin/env bash
# setup-hands.sh - one command for the hands box (Ryzen 9 7940HS + Radeon 780M, 96 GB): checks the host, downloads and
# verifies SC117's GSQ-RCO IQ3_S shards, re-encodes them locally (hc_* BF16 -> Q8_0, head + embedding -> Q8_0), pulls the
# image, starts the compose file and smoke-tests it. Idempotent and resumable; finished steps are skipped.
#
#   deploy/balkon/setup-hands.sh                  # everything
#   deploy/balkon/setup-hands.sh --dry-run        # print every step and size, change nothing
#   deploy/balkon/setup-hands.sh --only download  # or --only encode, --only up
#
# Env: MODELS_DIR (default deploy/balkon/models), SLOTS_DIR (default deploy/balkon/slots), LLAMA_S1_IMAGE (default the prebuilt
# image), BUILD=1 to build the image from this tree, PRUNE=1 to delete the original shard 1 and the head source after a
# verified re-encode (-59 GB). Why these weights and these re-encodes: docs/s1/WEIGHTS.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck source=../setup-common.sh
source "$HERE/../setup-common.sh"

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
parse_args "$@"

COMPOSE="$HERE/docker-compose.hands.yml"; CONTAINER=llama-s1-hands; NEED_GTT_MIB=79872
export MODELS_DIR="${MODELS_DIR:-$HERE/models}"
export SLOTS_DIR="${SLOTS_DIR:-$HERE/slots}"
export LLAMA_S1_IMAGE="${LLAMA_S1_IMAGE:-vahpetr/llama-cpp-s1-fork:v56k}"
PORT=8000
TOOL="$ROOT/tools/s1/gguf_hc_q8.py"

# ---- weights, sha256 as published on Hugging Face (LFS oid) and checked against my local files, as of 2026-10-10 ----------
REPO_H=SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF
P=Qwen3.8-Flash-Next-GSQ-RCO-abliterated
S1="IQ3_S/$P-IQ3_S-00001-of-00002.gguf"          # trunk; 55 063 448 064 B
S1_SHA=40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c; S1_SIZE=55063448064
S2="IQ3_S/$P-IQ3_S-00002-of-00002.gguf"          # the PLE n-gram table, identical in every SC117 tier; 28 800 138 432 B
S2_SHA=316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113; S2_SIZE=28800138432
# Source of the Q8_0 LM head + token embedding: a Q8_0 export of the official MTP head. Its output.weight / token_embd.weight
# are byte-identical to the ones I used (checked tensor by tensor, 2026-10-10).
REPO_Q=kingjones777/Qwen3.8-Flash-Next-MTP-Heads-GGUF
HEADSRC=mtp-Qwen3.8-Flash-Next-Q8_0.gguf
HEADSRC_SHA=25565dac32005bf4b9582cf81c58ef3c1a59d4225a2240821fc30459d7154a4d; HEADSRC_SIZE=4135934112
REPO_V=mradermacher/Qwen3.8-Flash-Next-Uncensored-GGUF
MMPROJ=Qwen3.8-Flash-Next-Uncensored.mmproj-f16.gguf
MMPROJ_SHA=4fee3582716fe691461576402f178882963e428ca9cbcc84dfeac9c53ebc1ae3; MMPROJ_SIZE=904004352
REPO_D=dzannotti/Qwen3.8-Flash-Next-MTP-GGUF
DRAFT=Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf
DRAFT_SHA=1f2a6991237d5412c9582b654e89e3aaa5a0fd33206d2b6d1813b15c475f4c02; DRAFT_SIZE=2622313344
# Result of the re-encode (shard 1). This is the sha256 of my file on the hands box.
O8_DIR=IQ3_S-hc8-o8
O1="$O8_DIR/$P-IQ3_S-hc8-o8-00001-of-00002.gguf"
O2="$O8_DIR/$P-IQ3_S-hc8-o8-00002-of-00002.gguf"
O1_SHA=4b1be1167cf5303c2e1e4b4e8b54f7b46df9e0aa6b8d84400aba8a992e54c6b7; O1_SIZE=54955467264

# ---- 1. prerequisites ----------------------------------------------------------------------------------------------------
log "prerequisites (models in $MODELS_DIR)"
check_docker
want download && find_hf
command -v curl >/dev/null || die "curl is missing"
if want encode; then
    python3 -c 'import numpy' 2>/dev/null || die "python3 with numpy is needed for the re-encode (apt install python3-numpy)"
fi
check_gtt "$NEED_GTT_MIB"
if want download || want encode; then
    need=0
    [ -f "$MODELS_DIR/$S1" ] || [ -f "$MODELS_DIR/$O1" ] || need=$(( need + S1_SIZE ))
    [ -f "$MODELS_DIR/$S2" ] || need=$(( need + S2_SIZE ))
    [ -f "$MODELS_DIR/$HEADSRC" ] || [ -f "$MODELS_DIR/$O1" ] || need=$(( need + HEADSRC_SIZE ))
    [ -f "$MODELS_DIR/$MMPROJ" ] || need=$(( need + MMPROJ_SIZE ))
    [ -f "$MODELS_DIR/$DRAFT" ] || need=$(( need + DRAFT_SIZE ))
    [ -f "$MODELS_DIR/$O1" ] || need=$(( need + O1_SIZE ))
    check_disk "$MODELS_DIR" "$need"
fi

# ---- 2. download + verify ------------------------------------------------------------------------------------------------
if want download; then
    log "weights"
    if [ -f "$MODELS_DIR/$O1" ] && [ "$DRY_RUN" = 0 ] && sha_ok "$MODELS_DIR/$O1" "$O1_SHA"; then
        info "ok: $O1 already built - shard 1 and the head source are not needed"
    else
        fetch "$REPO_H" "$S1" "$S1_SHA" "$S1_SIZE" "$MODELS_DIR"
        fetch "$REPO_Q" "$HEADSRC" "$HEADSRC_SHA" "$HEADSRC_SIZE" "$MODELS_DIR"
    fi
    fetch "$REPO_H" "$S2" "$S2_SHA" "$S2_SIZE" "$MODELS_DIR"
    fetch "$REPO_V" "$MMPROJ" "$MMPROJ_SHA" "$MMPROJ_SIZE" "$MODELS_DIR"
    fetch "$REPO_D" "$DRAFT" "$DRAFT_SHA" "$DRAFT_SIZE" "$MODELS_DIR"
fi

# ---- 3. re-encode (CPU only) ---------------------------------------------------------------------------------------------
if want encode; then
    log "re-encode: hc_* BF16 -> Q8_0 and output.weight + token_embd.weight from the Q8_0 head source"
    run mkdir -p "$MODELS_DIR/$O8_DIR"
    if [ -f "$MODELS_DIR/$O1" ] && [ "$DRY_RUN" = 0 ] && sha_ok "$MODELS_DIR/$O1" "$O1_SHA"; then
        info "ok: $O1"
    else
        [ "$DRY_RUN" = 1 ] || [ -f "$MODELS_DIR/$S1" ] || die "missing $S1 (run --only download first)"
        run ionice -c3 nice python3 "$TOOL" "$MODELS_DIR/$S1" "$MODELS_DIR/$O1.tmp" \
            --from "$MODELS_DIR/$HEADSRC" --tensors output.weight,token_embd.weight
        if [ "$DRY_RUN" = 0 ]; then
            # Structural check: same tensor list and shapes; exactly 290 hc_* tensors BF16 -> Q8_0 and the two head tensors
            # changed type; nothing else changed type. Then the byte check against my file.
            python3 - "$MODELS_DIR/$S1" "$MODELS_DIR/$O1.tmp" <<'PY'
import struct, sys
def tensors(p):
    h = open(p, 'rb').read(64 << 20); o = [4]
    def r(f):
        v = struct.unpack_from(f, h, o[0]); o[0] += struct.calcsize(f); return v[0]
    def s():
        n = r('<Q'); v = h[o[0]:o[0] + n]; o[0] += n; return v
    def sk(t):
        if t == 8: s(); return
        if t == 9:
            et = r('<I'); n = r('<Q')
            for _ in range(n): sk(et)
            return
        r({0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}[t])
    r('<I'); nt = r('<Q'); nk = r('<Q')
    for _ in range(nk): s(); sk(r('<I'))
    out = {}
    for _ in range(nt):
        n = s().decode(); nd = r('<I'); d = tuple(r('<Q') for _ in range(nd)); t = r('<I'); r('<Q'); out[n] = (d, t)
    return out
a, b = tensors(sys.argv[1]), tensors(sys.argv[2])
assert a.keys() == b.keys(), "tensor lists differ"
assert all(a[k][0] == b[k][0] for k in a), "shapes differ"
changed = [k for k in a if a[k][1] != b[k][1]]
hc = [k for k in changed if 'hc_' in k and a[k][1] == 30 and b[k][1] == 8]
head = [k for k in changed if k in ('output.weight', 'token_embd.weight') and b[k][1] == 8]
other = set(changed) - set(hc) - set(head)
print(f"    tensors {len(a)}, type changes: hc_* BF16->Q8_0 {len(hc)}, head/embedding ->Q8_0 {len(head)}, other {len(other)}")
assert len(hc) == 290 and len(head) == 2 and not other, "unexpected tensor changes"
PY
            mv "$MODELS_DIR/$O1.tmp" "$MODELS_DIR/$O1"
            sha_ok "$MODELS_DIR/$O1" "$O1_SHA" || warn "re-encoded shard 1 differs from my file (structure is fine; numpy rounding may differ)"
        fi
    fi
    # shard 2 (the PLE table) is not touched: link it instead of copying 29 GB
    if [ ! -e "$MODELS_DIR/$O2" ]; then
        if [ "$DRY_RUN" = 1 ] || ln "$MODELS_DIR/$S2" "$MODELS_DIR/$O2" 2>/dev/null; then
            info "hardlink: $O2 -> $S2"
        else
            run ln -s "../$S2" "$MODELS_DIR/$O2"
        fi
    else
        info "ok: $O2"
    fi
    if [ "${PRUNE:-0}" = 1 ] && [ -f "$MODELS_DIR/$O1" ]; then
        run rm -f "$MODELS_DIR/$S1" "$MODELS_DIR/$S1.sha256-ok" "$MODELS_DIR/$HEADSRC" "$MODELS_DIR/$HEADSRC.sha256-ok"
    fi
fi

# ---- 4. image, start, smoke ----------------------------------------------------------------------------------------------
if want up; then
    log "image $LLAMA_S1_IMAGE"
    if [ "${BUILD:-0}" = 1 ]; then
        info "building from this tree (same recipe as my prod images: .devops/vulkan.Dockerfile, target server)"
        run docker compose -f "$COMPOSE" build
    elif ! docker image inspect "$LLAMA_S1_IMAGE" >/dev/null 2>&1; then
        run docker pull "$LLAMA_S1_IMAGE" || die "pull failed; build it instead: BUILD=1 $0"
    else
        info "present"
    fi
    check_vulkan "$LLAMA_S1_IMAGE"
    for f in "$O1" "$O2" "$MMPROJ" "$DRAFT"; do
        [ "$DRY_RUN" = 1 ] || [ -e "$MODELS_DIR/$f" ] || die "missing $MODELS_DIR/$f (run without --only up first)"
    done
    log "start ($COMPOSE)"
    run mkdir -p "$SLOTS_DIR"
    if [ "$DRY_RUN" = 0 ] && docker ps --format '{{.Names}}' | grep -qE '^llama-s1-(brain|hands)' \
        && ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
        die "another llama-s1 container is running; stop it first (docker ps)"
    fi
    run docker compose -f "$COMPOSE" up -d
    info "expected: GTT after load ~73 GiB of a 78 GiB limit; decode ~15-16 t/s, prefill ~140 t/s on short prompts"
    if [ "$DRY_RUN" = 0 ]; then
        wait_health "$PORT" 1200
        smoke "$PORT"
    fi
    log "done - webui at http://<this host>:$PORT/ , logs: docker logs -f $CONTAINER"
fi
