# shellcheck shell=bash
# Shared helpers for deploy/s1/setup-brain.sh and deploy/balkon/setup-hands.sh. Sourced, not run.

DRY_RUN=0
ONLY=all

log()  { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    WARNING: %s\n' "$*" >&2; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# run CMD...: print it, and execute it unless --dry-run
run() {
    printf '    $ %s\n' "$*"
    if [ "$DRY_RUN" = 0 ]; then "$@"; fi
}

parse_args() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run) DRY_RUN=1 ;;
            --only) shift; ONLY="${1:-}" ;;
            --only=*) ONLY="${1#--only=}" ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown argument: $1 (see --help)" ;;
        esac
        shift
    done
    case "$ONLY" in all|download|encode|up) ;; *) die "--only must be download, encode or up" ;; esac
}

want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

human() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1e9 }'; }

# ---- prerequisites ---------------------------------------------------------------------------------------------------

HF=""
find_hf() {
    if command -v hf >/dev/null 2>&1; then HF="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then HF="huggingface-cli"
    elif [ "$DRY_RUN" = 1 ]; then
        HF="hf"; warn "no Hugging Face CLI found (pipx install 'huggingface_hub[cli]'); the dry run assumes hf"
    else
        die "no Hugging Face CLI found. Install it with one of:
         pipx install 'huggingface_hub[cli]'
         uv tool install 'huggingface_hub[cli]'"
    fi
    info "Hugging Face CLI: $HF"
}

check_docker() {
    command -v docker >/dev/null 2>&1 || die "docker is not installed"
    docker compose version >/dev/null 2>&1 || die "the docker compose plugin is missing (docker compose version failed)"
    info "docker: $(docker --version | cut -d, -f1), $(docker compose version --short 2>/dev/null || echo compose)"
}

# check_disk DIR BYTES_NEEDED
check_disk() {
    local dir="$1" need="$2" avail
    mkdir -p "$dir"
    avail=$(df -PB1 "$dir" | awk 'NR == 2 { print $4 }')
    info "free space in $dir: $(human "$avail") (this run may still need up to $(human "$need"))"
    if [ "$avail" -lt "$need" ]; then
        warn "not enough free space for the remaining steps"
        [ "$DRY_RUN" = 1 ] || die "free at least $(human "$need") in $dir or set MODELS_DIR"
    fi
}

GTT_TOTAL_MIB=0
# check_gtt NEEDED_MIB: read the GTT limit the amdgpu driver reports (falls back to ttm.pages_limit)
check_gtt() {
    local need="$1" f bytes pages
    # the amdgpu driver dir first: the path handed to the server must survive a reboot, and /sys/class/drm/cardN does not
    # (my s1 went card1 -> card0 at a reboot and the server silently ran with headroom admission disabled for two days)
    for f in /sys/bus/pci/drivers/amdgpu/*/mem_info_gtt_total /sys/class/drm/card*/device/mem_info_gtt_total; do
        [ -r "$f" ] || continue
        bytes=$(cat "$f"); GTT_TOTAL_MIB=$(( bytes / 1048576 ))
        case "$f" in /sys/bus/pci/drivers/amdgpu/*) f="/sys/bus/pci/devices/$(basename "$(dirname "$f")")/mem_info_gtt_total";; esac
        GTT_USED_PATH="${f%_total}_used"
        break
    done
    if [ "$GTT_TOTAL_MIB" = 0 ] && [ -r /sys/module/ttm/parameters/pages_limit ]; then
        pages=$(cat /sys/module/ttm/parameters/pages_limit); GTT_TOTAL_MIB=$(( pages * 4 / 1024 ))
    fi
    if [ "$GTT_TOTAL_MIB" = 0 ]; then
        warn "could not read the GTT limit (no amdgpu device found?)"
        return
    fi
    info "GTT limit: $GTT_TOTAL_MIB MiB (this profile needs >= $need MiB)"
    if [ "$GTT_TOTAL_MIB" -lt "$need" ]; then
        warn "GTT too small for this profile. Raise ttm.pages_limit / ttm.page_pool_size: docs/s1/HOST-TUNING.md"
    fi
}

check_vulkan() {
    local image="$1"
    [ -e /dev/dri ] || die "/dev/dri does not exist: no GPU visible to the host"
    if command -v vulkaninfo >/dev/null 2>&1; then
        info "Vulkan devices on the host:"
        vulkaninfo --summary 2>/dev/null | grep -E 'deviceName' | sed 's/^/      /' || true
    fi
    if [ "$DRY_RUN" = 0 ] && docker image inspect "$image" >/dev/null 2>&1; then
        info "devices seen inside the image:"
        docker run --rm --device /dev/dri --entrypoint /app/llama-server "$image" --list-devices 2>&1 \
            | grep -iE 'vulkan|radeon|amd' | sed 's/^/      /' || warn "the image sees no Vulkan device"
    fi
}

# ---- downloads -------------------------------------------------------------------------------------------------------

# sha_ok FILE SHA256: compare, caching a positive result next to the file (keyed by size + mtime)
sha_ok() {
    local file="$1" want="$2" stamp got key
    stamp="$file.sha256-ok"
    key="$(stat -c '%s %Y' "$file") $want"
    if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$key" ]; then return 0; fi
    info "sha256 $(basename "$file") ..."
    got=$(ionice -c3 nice sha256sum "$file" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then echo "$key" > "$stamp"; return 0; fi
    warn "sha256 mismatch for $file: got $got, want $want"
    return 1
}

# fetch REPO PATH_IN_REPO SHA256 SIZE DEST_DIR: download one file unless it is already there with the right hash
fetch() {
    local repo="$1" path="$2" sha="$3" size="$4" dest="$5" file
    file="$dest/$path"
    if [ -f "$file" ] && [ "$DRY_RUN" = 0 ] && sha_ok "$file" "$sha"; then
        info "ok: $path"
        return 0
    fi
    info "download $repo/$path ($(human "$size"))"
    run "$HF" download "$repo" "$path" --local-dir "$dest"
    if [ "$DRY_RUN" = 0 ]; then
        sha_ok "$file" "$sha" || die "$file failed its sha256 check; delete it and re-run"
    fi
}

# ---- start + smoke ---------------------------------------------------------------------------------------------------

# wait_health PORT TIMEOUT_S
wait_health() {
    local port="$1" t="$2" i=0
    info "waiting for http://127.0.0.1:$port/health (up to $t s) ..."
    while [ $i -lt "$t" ]; do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then info "healthy after ${i}s"; return 0; fi
        sleep 5; i=$((i + 5))
    done
    die "the server did not become healthy; check: docker logs $CONTAINER"
}

smoke() {
    local port="$1"
    log "smoke test"
    info "n_ctx: $(curl -s "http://127.0.0.1:$port/props" | python3 -c 'import sys, json; d = json.load(sys.stdin); print(d.get("default_generation_settings", {}).get("n_ctx", "?"))' 2>/dev/null || echo '?')"
    info "reply: $(curl -s "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Reply with one word: ready"}],"max_tokens":16,"temperature":0}' \
        | python3 -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[:60])' 2>/dev/null || echo FAILED)"
    info "disk cache: $(curl -s "http://127.0.0.1:$port/disk-cache" | python3 -c 'import sys, json; d = json.load(sys.stdin); print("own key", d.get("key", d.get("own_key", "?")), "entries", d.get("n_entries", d.get("entries", "?")))' 2>/dev/null || echo 'not enabled')"
    if [ -n "${GTT_USED_PATH:-}" ] && [ -r "$GTT_USED_PATH" ]; then
        info "GTT used now: $(( $(cat "$GTT_USED_PATH") / 1048576 )) MiB"
    fi
}
