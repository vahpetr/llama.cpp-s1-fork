#!/usr/bin/env bash
# setup-brain.sh - one command for the s1 brain (Ryzen AI Max+ 395 / Strix Halo, 128 GB): checks the host, downloads and
# verifies the weights, pulls the image, starts the compose file and smoke-tests it. Idempotent: re-run it any time, finished
# steps are skipped (a downloaded file is re-hashed only if its size or mtime changed).
#
#   deploy/s1/setup-brain.sh                    # 1M context (default; needs GTT 110 GiB)
#   BRAIN_CTX=524k deploy/s1/setup-brain.sh     # 524k context (fits GTT 104 GiB)
#   deploy/s1/setup-brain.sh --dry-run          # print every step and size, change nothing
#   deploy/s1/setup-brain.sh --only download    # or --only up
#
# Env: MODELS_DIR (default deploy/s1/models), SLOTS_DIR (default deploy/s1/slots), LLAMA_S1_IMAGE (default the prebuilt image),
# BUILD=1 to build the image from this tree instead of pulling it. Why these weights: docs/s1/WEIGHTS.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../setup-common.sh
source "$HERE/../setup-common.sh"

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
parse_args "$@"
[ "$ONLY" = encode ] && die "the brain has no encode step (--only download | up)"

BRAIN_CTX="${BRAIN_CTX:-1m}"
case "$BRAIN_CTX" in
    1m)   COMPOSE="$HERE/docker-compose.brain-1m.yml";   CONTAINER=llama-s1-brain-1m;   NEED_GTT_MIB=112640
          EXPECT="GTT after load ~105 000 MiB (102.7 GiB), flat at every depth; cold fill 0 -> 1M ~1.1 h, 0 -> 524k ~26 min" ;;
    524k) COMPOSE="$HERE/docker-compose.brain-524k.yml"; CONTAINER=llama-s1-brain-524k; NEED_GTT_MIB=98000
          EXPECT="GTT after load ~94 500 MiB (92.3 GiB), flat at every depth; cold fill 0 -> 524k ~26 min" ;;
    *) die "BRAIN_CTX must be 1m or 524k" ;;
esac
export MODELS_DIR="${MODELS_DIR:-$HERE/models}"
export SLOTS_DIR="${SLOTS_DIR:-$HERE/slots}"
export LLAMA_S1_IMAGE="${LLAMA_S1_IMAGE:-vahpetr/llama-cpp-s1-fork:v58}"
PORT=8000

# ---- weights, sha256 as published on Hugging Face (LFS oid) and checked against my local files, as of 2026-10-10 ----------
# Target: mradermacher's imatrix Q4_K_M of orcarouter/Qwen3.8-Flash-Next-Uncensored, published as 3 parts to concatenate.
REPO_T=mradermacher/Qwen3.8-Flash-Next-Uncensored-i1-GGUF
TARGET=Qwen3.8-Flash-Next-Uncensored.i1-Q4_K_M.gguf
TARGET_SHA=9a3af985dcc9ae3f05b5f6ff93d5df25da8bc07e0be1e44d553806d990c97a82    # the concatenated file, 119 150 723 360 B
PARTS=(part1of3 part2of3 part3of3)
PART_SIZE=(39728447488 39728447488 39693828384)
PART_SHA=(fe0b81d46ab598a3850405d038f938c4045b5795d9eca0f6a935dafe134cc601
          c68de61a2d26d5c1ca0695d92a60d8118411610639dacfbaf7bbe495b7598314
          103f1595d32979fe87a22a45817c8773dcbb4cd9dec9017ae729394416b50ce2)
# Vision projector (static repo).
REPO_V=mradermacher/Qwen3.8-Flash-Next-Uncensored-GGUF
MMPROJ=Qwen3.8-Flash-Next-Uncensored.mmproj-f16.gguf
MMPROJ_SHA=4fee3582716fe691461576402f178882963e428ca9cbcc84dfeac9c53ebc1ae3; MMPROJ_SIZE=904004352
# MTP draft head, exported from the official checkpoint.
REPO_D=dzannotti/Qwen3.8-Flash-Next-MTP-GGUF
DRAFT=Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf
DRAFT_SHA=1f2a6991237d5412c9582b654e89e3aaa5a0fd33206d2b6d1813b15c475f4c02; DRAFT_SIZE=2622313344

# ---- 1. prerequisites ----------------------------------------------------------------------------------------------------
log "prerequisites ($BRAIN_CTX profile, models in $MODELS_DIR)"
check_docker
want download && find_hf
command -v curl >/dev/null || die "curl is missing"
check_gtt "$NEED_GTT_MIB"
if [ "$GTT_TOTAL_MIB" -gt 0 ]; then
    export LLAMA_GTT_ADMISSION_MIB=$(( GTT_TOTAL_MIB - 3424 ))
    info "GTT admission bar for this host: $LLAMA_GTT_ADMISSION_MIB MiB (GTT - 3 424)"
fi
[ -n "${GTT_USED_PATH:-}" ] && export LLAMA_GTT_USED_PATH="$GTT_USED_PATH"
if want download; then
    # the biggest moment: the concatenated file grows while one 40 GB part still sits next to it
    need=0
    [ -f "$MODELS_DIR/$TARGET" ] || need=$(( 119150723360 + 39728447488 ))
    [ -f "$MODELS_DIR/$MMPROJ" ] || need=$(( need + MMPROJ_SIZE ))
    [ -f "$MODELS_DIR/$DRAFT" ] || need=$(( need + DRAFT_SIZE ))
    check_disk "$MODELS_DIR" "$need"
fi

# ---- 2. download + verify ------------------------------------------------------------------------------------------------
if want download; then
    log "weights"
    if [ -f "$MODELS_DIR/$TARGET" ] && [ ! -f "$MODELS_DIR/$TARGET.assembling" ] && [ "$DRY_RUN" = 0 ] \
        && sha_ok "$MODELS_DIR/$TARGET" "$TARGET_SHA"; then
        info "ok: $TARGET"
    else
        # Assemble part by part: download part k, check it, append it, delete it. Peak disk = the file + one part.
        # Resumable: the file is cut back to the last complete part boundary before the next part is appended.
        run touch "$MODELS_DIR/$TARGET.assembling"
        done_bytes=0
        for k in 0 1 2; do
            part="$TARGET.${PARTS[$k]}"
            end=$(( done_bytes + PART_SIZE[k] ))
            have=0; [ -f "$MODELS_DIR/$TARGET" ] && have=$(stat -c %s "$MODELS_DIR/$TARGET")
            if [ "$have" -ge "$end" ]; then
                info "ok: ${PARTS[$k]} already appended"
                [ -f "$MODELS_DIR/$part" ] && run rm -f "$MODELS_DIR/$part" "$MODELS_DIR/$part.sha256-ok"
            else
                fetch "$REPO_T" "$part" "${PART_SHA[$k]}" "${PART_SIZE[$k]}" "$MODELS_DIR"
                if [ "$DRY_RUN" = 0 ]; then
                    [ -f "$MODELS_DIR/$TARGET" ] || : > "$MODELS_DIR/$TARGET"
                    truncate -s "$done_bytes" "$MODELS_DIR/$TARGET"
                fi
                info "append ${PARTS[$k]}"
                if [ "$DRY_RUN" = 0 ]; then cat "$MODELS_DIR/$part" >> "$MODELS_DIR/$TARGET"; fi
                run rm -f "$MODELS_DIR/$part" "$MODELS_DIR/$part.sha256-ok"
            fi
            done_bytes=$end
        done
        if [ "$DRY_RUN" = 0 ]; then
            sha_ok "$MODELS_DIR/$TARGET" "$TARGET_SHA" || die "$TARGET failed its sha256 check; delete it and re-run"
        fi
        run rm -f "$MODELS_DIR/$TARGET.assembling"
    fi
    fetch "$REPO_V" "$MMPROJ" "$MMPROJ_SHA" "$MMPROJ_SIZE" "$MODELS_DIR"
    fetch "$REPO_D" "$DRAFT" "$DRAFT_SHA" "$DRAFT_SIZE" "$MODELS_DIR"
fi

# ---- 3. image, start, smoke ----------------------------------------------------------------------------------------------
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
    for f in "$TARGET" "$MMPROJ" "$DRAFT"; do
        [ "$DRY_RUN" = 1 ] || [ -f "$MODELS_DIR/$f" ] || die "missing $MODELS_DIR/$f (run without --only up first)"
    done
    log "start ($COMPOSE)"
    run mkdir -p "$SLOTS_DIR"
    if [ "$DRY_RUN" = 0 ] && docker ps --format '{{.Names}}' | grep -qE '^llama-s1-(brain|hands)' \
        && ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
        warn "another llama-s1 container is running; two model loads on one GPU run the box out of memory"
        die "stop it first: docker ps"
    fi
    run docker compose -f "$COMPOSE" up -d
    info "expected: $EXPECT"
    if [ "$DRY_RUN" = 0 ]; then
        wait_health "$PORT" 900
        smoke "$PORT"
    fi
    log "done - webui at http://<this host>:$PORT/ , logs: docker logs -f $CONTAINER"
fi
