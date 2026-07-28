#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
python_bin="${REDCO_PYTHON:-python}"
config_root="${REDCO_CONFIG_ROOT:-configs/stage-c3/rendered-v2}"
run_root="${REDCO_RUN_ROOT:-runs/stage-c3/credit-confusion-live-v2}"
launcher="${REDCO_ARM_LAUNCHER:-scripts/run_stage_c2_campaign_arm.sh}"
invariant_timeout="${REDCO_INVARIANT_TIMEOUT_SECONDS:-900}"

cd "$repo_root"
test -f "$config_root/smoke-broadcast-s9400.toml"
test ! -e "$run_root/smoke/broadcast-s9400"

run_supervised() {
  local mode="$1"
  local config="$2"
  local output="$3"
  local result="$4"
  "$python_bin" scripts/run_stage_c3_supervised_arm.py \
    --launcher "$launcher" \
    --config "$config" \
    --output-dir "$output" \
    --result "$result" \
    --mode "$mode" \
    --invariant-timeout-seconds "$invariant_timeout"
}

# The smoke is deliberately outside the eight-run scientific campaign. No
# scientific arm starts unless one real training batch demonstrates parseable
# routes, reward variation, and a nonzero trainable fraction.
run_supervised \
  smoke \
  "$config_root/smoke-broadcast-s9400.toml" \
  "$run_root/smoke/broadcast-s9400" \
  "$run_root/smoke/invariant.json"

# Frozen alternation order. There is intentionally no retry path.
while IFS='|' read -r probe arm seed; do
  config="$config_root/${probe}-${arm}-s${seed}.toml"
  output="$run_root/${probe}/${arm}-s${seed}"
  result="$run_root/invariants/${probe}-${arm}-s${seed}.json"
  test -f "$config"
  test ! -e "$output"
  run_supervised arm "$config" "$output" "$result"
done <<'RUNS'
confusion_irrelevant|broadcast|9401
confusion_irrelevant|sliced|9401
confusion_irrelevant|sliced|9402
confusion_irrelevant|broadcast|9402
confusion_redundant|broadcast|9403
confusion_redundant|sliced|9403
confusion_lucky|sliced|9404
confusion_lucky|broadcast|9404
RUNS

touch "$run_root/CAMPAIGN_COMPLETE"
