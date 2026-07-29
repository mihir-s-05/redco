#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
config_root="${REDCO_CONFIG_ROOT:-configs/stage-c3/rendered-v3}"
run_root="${REDCO_RUN_ROOT:-runs/stage-c3/credit-confusion-live-v3}"
launcher="${REDCO_ARM_LAUNCHER:-scripts/run_stage_c2_campaign_arm.sh}"
invariant_timeout="${REDCO_INVARIANT_TIMEOUT_SECONDS:-900}"
model_path="runs/stage-c2/warmstart-merged-candidates-v2/step_23"
prime_patch="patches/prime-rl-redco-stage-c3-v3.patch"
forced_config="configs/stage-c3/credit-confusion-forced-smoke-v3.toml"
structural_config="$config_root/structural-broadcast-s9601.toml"

cd "$repo_root"
test -f "$forced_config"
test -f "$structural_config"
test -s "$prime_patch"
test -d "$model_path"
test ! -e "$run_root"
mkdir -p "$run_root/smoke" "$run_root/power" "$run_root/invariants"

export REDCO_PRIME_PATCH="$prime_patch"
export REDCO_RUN_SEED=7202901
export REDCO_UV_ENVIRONMENT="${REDCO_UV_ENVIRONMENT:-/workspace/.venv-prime-stage-c3-v3}"
export REDCO_UV_CACHE_DIR="${REDCO_UV_CACHE_DIR:-/workspace/.uv-cache-prime-stage-c3-v3}"

uv_prime=(
  uv run --frozen --project external/prime-rl
  --extra flash-attn
  --with-editable "$repo_root"
  --with-editable "$repo_root/environments/redco_credit_v1"
  --with-editable "$repo_root/external/prime-rl/deps/verifiers"
)

campaign_status=1
finish() {
  if test "$campaign_status" -ne 0; then
    touch "$run_root/CAMPAIGN_TERMINAL_FAILURE"
  fi
}
trap finish EXIT

run_supervised() {
  local mode="$1"
  local config="$2"
  local output="$3"
  local result="$4"
  "${uv_prime[@]}" python scripts/run_stage_c3_supervised_arm.py \
    --launcher "$launcher" \
    --config "$config" \
    --output-dir "$output" \
    --result "$result" \
    --mode "$mode" \
    --invariant-timeout-seconds "$invariant_timeout"
}

# Deterministic integration smoke: live model serving with constrained outputs
# exercises parsing, rewards, serialization, and one trainer step without a
# sampled pass condition.
forced_output="$run_root/smoke/forced-broadcast"
forced_invariant="$run_root/smoke/forced-invariant.json"
run_supervised smoke "$forced_config" "$forced_output" "$forced_invariant"
forced_traces="$forced_output/run_default/rollouts/step_1/train/all/traces.jsonl"
test -s "$forced_traces"
"${uv_prime[@]}" python -m redco.analysis.stage_c3_forced_smoke \
  --traces "$forced_traces" \
  --invariant "$forced_invariant" \
  --output "$run_root/smoke/forced-verification.json"

# Reuse the deterministic smoke's exact rendered prefixes to measure support
# analytically. The scientific campaign cannot begin unless the frozen power
# requirements pass.
"${uv_prime[@]}" python -m redco.analysis.stage_c_policy_audit prepare \
  --traces "$forced_traces" \
  --output "$run_root/power/action-cases.json"
"${uv_prime[@]}" python -m redco.analysis.stage_c3_power prepare-root-cases \
  --traces "$forced_traces" \
  --output "$run_root/power/root-cases.json"
"${uv_prime[@]}" python scripts/score_stage_c_policies_vllm.py \
  --cases "$run_root/power/action-cases.json" \
  --model "$model_path" \
  --model-name warmstart \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  --output "$run_root/power/action-scores.json"
"${uv_prime[@]}" python scripts/score_stage_c3_root_routes_vllm.py \
  --cases "$run_root/power/root-cases.json" \
  --model "$model_path" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  --output "$run_root/power/root-scores.json"
"${uv_prime[@]}" python -m redco.analysis.stage_c3_power analyze \
  --action-scores "$run_root/power/action-scores.json" \
  --route-scores "$run_root/power/root-scores.json" \
  --output "$run_root/power/exact-power.json"
touch "$run_root/POWER_GATE_PASS"

# A normally sampled live smoke follows, but its invariant is structural only:
# model/runtime errors and a regressed root token budget can stop the campaign;
# sampled routes, rewards, and trainability cannot.
structural_output="$run_root/smoke/structural-broadcast-s9601"
run_supervised \
  arm \
  "$structural_config" \
  "$structural_output" \
  "$run_root/smoke/structural-invariant.json"
touch "$run_root/STRUCTURAL_SMOKE_PASS"

# Frozen alternation order and fresh v3 seed block. There is no retry path.
while IFS='|' read -r probe arm seed; do
  config="$config_root/${probe}-${arm}-s${seed}.toml"
  output="$run_root/${probe}/${arm}-s${seed}"
  result="$run_root/invariants/${probe}-${arm}-s${seed}.json"
  test -f "$config"
  test ! -e "$output"
  run_supervised arm "$config" "$output" "$result"
done <<'RUNS'
confusion_irrelevant|broadcast|9501
confusion_irrelevant|sliced|9501
confusion_irrelevant|sliced|9502
confusion_irrelevant|broadcast|9502
confusion_redundant|broadcast|9503
confusion_redundant|sliced|9503
confusion_lucky|sliced|9504
confusion_lucky|broadcast|9504
RUNS

adapter_args=()
while IFS='|' read -r probe arm seed step; do
  name="${probe}--${arm}--s${seed}"
  adapter="$run_root/$probe/${arm}-s${seed}/run_default/broadcasts/step_${step}"
  test -s "$adapter/adapter_model.safetensors"
  adapter_args+=(--adapter "$name=$adapter")
done <<'ADAPTERS'
confusion_irrelevant|broadcast|9501|36
confusion_irrelevant|sliced|9501|6
confusion_irrelevant|sliced|9502|6
confusion_irrelevant|broadcast|9502|36
confusion_redundant|broadcast|9503|36
confusion_redundant|sliced|9503|6
confusion_lucky|sliced|9504|6
confusion_lucky|broadcast|9504|36
ADAPTERS

"${uv_prime[@]}" python scripts/score_stage_c_policies_vllm.py \
  --cases "$run_root/power/action-cases.json" \
  --model "$model_path" \
  --model-name warmstart \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  "${adapter_args[@]}" \
  --output "$run_root/final-policy-scores.json"
"${uv_prime[@]}" python -m redco.analysis.stage_c3_live \
  --run-root "$run_root" \
  --scores "$run_root/final-policy-scores.json" \
  --output "$run_root/frozen-decision.json"

touch "$run_root/CAMPAIGN_COMPLETE"
campaign_status=0
