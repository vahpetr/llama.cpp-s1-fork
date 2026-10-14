#!/usr/bin/env bash
# [TAG_HC_MIX_TAIL] [TAG_CONV_INPUT_SPLIT] check test-vk-graph-order against the pinned v59 reference.
#   tests/test-vk-graph-order.sh <path to the test-vk-graph-order binary of THIS revision> [fixture]
# The fixture (default tests/fixtures/vk-graph-order-v59.txt) holds, per n_tokens, the line a v59 (a1b6c324e) build of
# the same test source prints in its v59 form; its header records the producing command and the device (the dispatch
# signature sig= depends on the device and driver: compare on the device that produced the fixture).
# Checks, per n_tokens in the fixture:
#   1. knobs off (GGML_VK_FUSE_MULTI_ADD_SCALE=0 GGML_VK_FUSE_SCALE_SILU=0 GGML_VK_FUSE_CONCAT_SSM_CONV=0), v59 form:
#      the whole [vk-fusion-count] line (counts + sig) and out/pool/conv_state == the fixture  -> v59 node for node
#   2. knobs on (defaults), b4 form (the model's form with the knob on): out/pool/conv_state == the fixture
#      (the counts are printed: only multi_add_scale / scale_silu / concat_ssm_conv_silu may differ)
set -uo pipefail
BIN=${1:?usage: $0 <test-vk-graph-order binary> [fixture]}
FIX=${2:-$(dirname "$0")/fixtures/vk-graph-order-v59.txt}
fail=0
run(){ env GGML_VK_FUSION_COUNT=1 "$@" 2>&1 | grep -aE '^\[vk-fusion-count\]|^\[order-test\]'; }
hashes(){ grep -aoE 'out=0x[0-9a-f]+ pool=0x[0-9a-f]+ conv_state=0x[0-9a-f]+' <<< "$1"; }
counts(){ grep -a '^\[vk-fusion-count\]' <<< "$1"; }
for nt in $(grep -aoE '^\[order-test\] form=v59 nt=[0-9]+' "$FIX" | sed 's/.*nt=//'); do
  ref=$(awk -v nt="$nt" '/^# nt=/{p=($2=="nt="nt)} p && !/^#/' "$FIX")
  off=$(run GGML_VK_FUSE_MULTI_ADD_SCALE=0 GGML_VK_FUSE_SCALE_SILU=0 GGML_VK_FUSE_CONCAT_SSM_CONV=0 "$BIN" "$nt" 1 v59)
  on=$(run "$BIN" "$nt" 1 b4)
  if [ "$(counts "$off")" = "$(counts "$ref")" ] && [ "$(hashes "$off")" = "$(hashes "$ref")" ]; then echo "PASS nt=$nt knobs-off/v59-form == fixture (counts+sig+hashes)"
  else echo "FAIL nt=$nt knobs-off/v59-form differs from the fixture"; echo "  ref: $(counts "$ref") $(hashes "$ref")"; echo "  got: $(counts "$off") $(hashes "$off")"; fail=1; fi
  if [ "$(hashes "$on")" = "$(hashes "$ref")" ]; then echo "PASS nt=$nt knobs-on/b4-form hashes == fixture  [$(counts "$on" | sed 's/^\[vk-fusion-count\] //; s/ sig=.*//')]"
  else echo "FAIL nt=$nt knobs-on/b4-form hashes differ: ref $(hashes "$ref") got $(hashes "$on")"; fail=1; fi
done
[ -n "${nt:-}" ] || { echo "FAIL no fixture lines in $FIX"; fail=1; }
exit $fail
