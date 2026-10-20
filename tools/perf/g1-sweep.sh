#!/usr/bin/env bash
# [TAG_DENSE_TILE_G1] G1 dense coopmat1 tile sweep harness (plan 2026-10-18 row 2.5). Runs on balkon (780M), next to the
# hands server: every GPU run is one container under ~/piagent-tuning/.gpu.lock, nice 10, GGML_VK_DISABLE_MMQ_CM1=1 (prod).
# Vendored: test-vk-gemm-class.cpp + e0_classify.py = the E0 kernel-level byte classifier (S0.3 S9 / B1 / B2 rounds, as on balkon
# ~/llama.cpp-scratch/handA63/tools); g1-cases.tsv = the P.1 dense shapes at n 512 / 513 / 1024 / 2044 / 2047 / 2048.
#
#   g1-sweep.sh tool                         build test-vk-gemm-class against $G1_BUILD (no rpath: LD_LIBRARY_PATH picks the libs;
#                                            $G1_BUILD must lie under $G1_WORK/src, the tree the build container mounts)
#   (cfg everywhere: B:BM:BN:WM:WN:PAD | 1 table | off/base = 0 | def = unset; G1_EXTRA_ENV=VAR=val adds one env, e.g. a negative control)
#   g1-sweep.sh perf  <tag> <cfg|off>        test-backend-ops perf -o MUL_MAT on the P.1 shapes (n 512 / 2048) with
#                                            GGML_VK_PERF_LOGGER=1; cfg = B:BM:BN:WM:WN:PAD (sweep mode) | 1 (table) | off
#   g1-sweep.sh class <tag> <cfg|off|base>   byte classifier on tools/perf/g1-cases.tsv (+ [vk-dispatch] kernel names);
#                                            base = the same tool on the pristine $G1_BASE_BUILD libs (8c1a468aa)
#   g1-sweep.sh cmp   <base-tag> <var-tag>   e0_classify.py of two class runs -> logs/class-<var>.tsv + a one-line verdict
#   g1-sweep.sh configs                      every valid tile (BK 32, TM/TN/TK 16, WARP 64 fixed; pad 4) - one per line
#   g1-sweep.sh sweep <cfg-file> [every]     perf for each config, a base (off) run before the first and after every
#                                            <every> (default 8) configs (drift control); base tags off-<cfg-file name>-<i>
#   g1-sweep.sh table [--base <prefix>] <tag> [<tag>...]
#                                            per shape: steady kernel us (median of the 2nd half of the perf-logger lines)
#                                            for each tag, vs the mean of the perf-<prefix>*.log base runs (default prefix
#                                            off-: every base run; pass off-<cfg-file name>- to keep one sweep's baseline)
set -uo pipefail
W=${G1_WORK:-$HOME/llama.cpp-scratch/g1}; S=$W/src; L=$W/logs
BUILD=${G1_BUILD:-$S/build/bin}; BASE_BUILD=${G1_BASE_BUILD:-$HOME/llama.cpp-scratch/negctl-8c1a468aa/src/build/bin}
IMG=${G1_IMG:-llama-dev:vulkan}; U="$(id -u):$(id -g)"
GV=$(getent group video | cut -d: -f3); GR=$(getent group render | cut -d: -f3)
SHAPES='type_a=(q4_K|q5_0|q6_K),type_b=f32,m=(10240|2560|320|6144|12288),n=(512|2048),k=(320|6144|10240|2560),'
mkdir -p "$L"

gpu() {  # gpu <tag> <libdir> <env...> -- <cmd...>
    local tag=$1 lib=$2; shift 2; local envs=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=(-e "$1"); shift; done; shift
    flock "$HOME/piagent-tuning/.gpu.lock" timeout -k 30 30m docker run --rm --init --name "g1s-$tag-$$" --user "$U" \
        --device /dev/dri --group-add "$GV" --group-add "$GR" -v "$S:/src:ro" -v "$L:/logs" -v "$lib:/lib-under-test:ro" \
        -e HOME=/tmp -e LD_LIBRARY_PATH=/lib-under-test -e GGML_VK_DISABLE_MMQ_CM1=1 "${envs[@]}" ${G1_EXTRA_ENV:+-e "$G1_EXTRA_ENV"} -w /logs "$IMG" \
        nice -n 10 "$@"
}
cfgenv() {  # off / base = knob 0; def = knob unset (the code default); else the knob value
    case "$1" in off|base) echo GGML_VK_DENSE_TILE_G1=0 ;; def) echo G1_KNOB_UNSET=1 ;; *) echo "GGML_VK_DENSE_TILE_G1=$1" ;; esac; }

case "${1:-}" in
tool)
    case "$BUILD" in "$S"/*) ;; *) echo "G1_BUILD=$BUILD is not under $S (the container mounts only $S) - refusing" >&2; exit 2 ;; esac
    docker run --rm --user "$U" -v "$S:/src" -w /src "$IMG" g++ -O2 -std=c++17 -Iggml/include tools/perf/test-vk-gemm-class.cpp \
        -L"${BUILD#$S/}" -lggml -lggml-base -lggml-cpu -o "${BUILD#$S/}/test-vk-gemm-class" && echo "tool ok" ;;
perf)
    tag=$2 cfg=$3; lib=$BUILD; [ "$cfg" = base ] && lib=$BASE_BUILD
    gpu "$tag" "$lib" GGML_VK_PERF_LOGGER=1 "$(cfgenv "$cfg")" -- /lib-under-test/test-backend-ops perf -o MUL_MAT -b Vulkan0 \
        -p "$SHAPES" > "$L/perf-$tag.log" 2>&1
    echo "perf $tag cfg=$cfg rc=$? g1_lines=$(grep -c 'dense_tile_g1' "$L/perf-$tag.log") invalid=$(grep -c 'invalid for' "$L/perf-$tag.log")" ;;
class)
    tag=$2 cfg=$3; lib=$BUILD; [ "$cfg" = base ] && lib=$BASE_BUILD
    mkdir -p "$L/cls-$tag"; rm -f "$L/cls-$tag"/*.bin
    cp "$BUILD/test-vk-gemm-class" "$L/cls-$tag/tool"
    gpu "c$tag" "$lib" GGML_VK_DISPATCH_LOG=1 "$(cfgenv "$cfg")" -- "/logs/cls-$tag/tool" /src/tools/perf/g1-cases.tsv "/logs/cls-$tag" \
        > "$L/cls-$tag/run.log" 2>&1
    echo "class $tag cfg=$cfg rc=$? cases=$(grep -c '^\[e0-case\]' "$L/cls-$tag/run.log") errors=$(grep -c ERROR "$L/cls-$tag/run.log")" ;;
cmp)
    python3 "$S/tools/perf/e0_classify.py" "$L/cls-$2" "$L/cls-$3" "$3" > "$L/class-$3.tsv"
    awk -F'\t' -v v="$3" 'NR>1 { n++; c[$7]++; if ($12 ~ /_g1/) g++; if ($13 != "1/1" && $13 != "-/-" && $13 != "-") r++ }
        END { printf "cmp %s: cases %d identical %d differs %d missing %d on_g1 %d repeat_bad %d\n", v, n, c["identical"], c["differs"], c["missing"], g, r }' "$L/class-$3.tsv" ;;
configs)
    python3 - <<'PY'
for bm in (32, 64, 128, 256):
    for bn in (32, 64, 128, 256):
        for wm in (16, 32, 64, 128):
            for wn in (16, 32, 64, 128):
                if bm % wm or bn % wn: continue
                w = (bm // wm) * (bn // wn); b = w * 64
                if b > 1024 or (wm // 16) * (wn // 16) > 32: continue
                if any(bm % (b * lva // 32) for lva in (2, 4)) or bn % (b * 8 // 32) or bn % (b * 2 // 32): continue
                if (bm + bn) * 40 * 2 + 256 * w * 4 > 65536: continue
                print(f"{b}:{bm}:{bn}:{wm}:{wn}:4")
PY
    ;;
sweep)
    every=${3:-8}; i=0; r=0; sw=$(basename "$2"); sw=${sw%.*}
    "$0" perf "off-$sw-$r" off
    while read -r cfg; do
        [ -z "$cfg" ] && continue; case "$cfg" in \#*) continue ;; esac
        "$0" perf "cfg-${cfg//:/-}" "$cfg"; i=$((i + 1))
        if [ $((i % every)) = 0 ]; then r=$((r + 1)); "$0" perf "off-$sw-$r" off; fi
    done < "$2"
    r=$((r + 1)); "$0" perf "off-$sw-$r" off ;;
table)
    shift; bp=off-; [ "${1:-}" = --base ] && { bp=$2; shift 2; }
    python3 - "$L" "$bp" "$@" <<'PY'
import re, statistics, sys, glob, os
L, bp, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
pat = re.compile(r"^MUL_MAT (\S+) m=(\d+) n=(\d+) k=(\d+): (\d+) x ([\d.]+) us")
def load(tag):
    d = {}
    for line in open(f"{L}/perf-{tag}.log", errors="replace"):
        m = pat.match(line)
        if m:
            d.setdefault((m[1], int(m[2]), int(m[3]), int(m[4])), []).append(float(m[6]))
    return {k: statistics.median(v[len(v) // 2:]) for k, v in d.items()}
offs = [os.path.basename(f)[5:-4] for f in sorted(glob.glob(f"{L}/perf-{bp}*.log"))]
if not offs:
    sys.exit(f"no base runs perf-{bp}*.log in {L}")
base = {}
for t in offs:
    for k, v in load(t).items(): base.setdefault(k, []).append(v)
print("tag\ttype\tm\tn\tk\tbase_us\tbase_cv%\tus\tdelta%\tTFLOPS")
for t in tags:
    for k, v in sorted(load(t).items()):
        b = base.get(k)
        if not b: continue
        bm = statistics.mean(b); cv = 100 * statistics.pstdev(b) / bm
        print(f"{t}\t{k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\t{bm:.1f}\t{cv:.1f}\t{v:.1f}\t{100 * (v - bm) / bm:+.2f}\t{2 * k[1] * k[2] * k[3] / v / 1e6:.2f}")
PY
    ;;
*) sed -n 2,21p "$0"; exit 2 ;;
esac
