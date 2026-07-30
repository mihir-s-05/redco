#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
config_root="${REDCO_CONFIG_ROOT:-configs/stage-c6/rendered-v1}"
run_root="${REDCO_RUN_ROOT:-runs/stage-c6/credit-confusion-live-v1}"
launcher="${REDCO_ARM_LAUNCHER:-scripts/run_stage_c2_campaign_arm.sh}"
invariant_timeout="${REDCO_INVARIANT_TIMEOUT_SECONDS:-900}"
stage_c2_adapter="runs/stage-c2/warmstart-selected-v2/step_23/lora_adapters"
stage_c5_adapter="runs/stage-c5/constrained-successor-v3-selection/evidence/runs/stage-c5/warmstart-selected-v3/lora-adapters"
stage_c2_merged="runs/stage-c6/stage-c2-initialization-merged"
model_path="runs/stage-c6/selected-initialization-merged"
selection_root="$run_root/initialization"
action_cases="configs/stage-c4/selection-action-cases.json"
root_cases="configs/stage-c4/selection-root-cases.json"
dataset_manifest="datasets/stage-c4/factorized-warmstart-manifest.json"
prime_patch="patches/prime-rl-redco-stage-c3-v3.patch"
structural_config="$config_root/structural-broadcast-s9900.toml"

cd "$repo_root"
test -s "$stage_c2_adapter/adapter_model.safetensors"
test -s "$stage_c5_adapter/adapter_model.safetensors"
test -s "$action_cases"
test -s "$root_cases"
test -s "$dataset_manifest"
test -s "$prime_patch"
test -f "$structural_config"
test ! -e "$stage_c2_merged"
test ! -e "$model_path"
test ! -e "$run_root"

stage_c2_sha="$(sha256sum "$stage_c2_adapter/adapter_model.safetensors" | cut -d' ' -f1)"
stage_c5_sha="$(sha256sum "$stage_c5_adapter/adapter_model.safetensors" | cut -d' ' -f1)"
test "$stage_c2_sha" = "28fba5d421ea611db2e0d9cd411e40a0fc2035a9a45eb0bb3be24c84947e0ab6"
test "$stage_c5_sha" = "e1d56f45485eef065bae42980427ee3c88176a5c864cbb350fa8494d0370e623"
git -C external/prime-rl apply --no-index --reverse --check \
  "$repo_root/$prime_patch"

export REDCO_PRIME_PATCH="$prime_patch"
export REDCO_RUN_SEED=7203101
export REDCO_UV_ENVIRONMENT="${REDCO_UV_ENVIRONMENT:-/workspace/.venv-prime-stage-c6-v1}"
export REDCO_UV_CACHE_DIR="${REDCO_UV_CACHE_DIR:-/workspace/.uv-cache-prime-stage-c6-v1}"
export UV_PROJECT_ENVIRONMENT="$REDCO_UV_ENVIRONMENT"
export UV_CACHE_DIR="$REDCO_UV_CACHE_DIR"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export XDG_CONFIG_HOME="$repo_root/.runtime-config"
export PYTHONPATH="$repo_root/src:$repo_root/scripts"

uv_prime=(
  uv run --frozen --project external/prime-rl
  --extra flash-attn
  --with-editable "$repo_root"
  --with-editable "$repo_root/environments/redco_credit_v1"
  --with-editable "$repo_root/external/prime-rl/deps/verifiers"
)

campaign_status=1
finish() {
  if test "$campaign_status" -ne 0 && test -d "$run_root"; then
    touch "$run_root/CAMPAIGN_TERMINAL_FAILURE"
  fi
}
trap finish EXIT

mkdir -p "$selection_root" "$run_root/smoke" "$run_root/invariants"

"${uv_prime[@]}" python scripts/merge_stage_c_warmstart.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter "$stage_c2_adapter" \
  --output "$stage_c2_merged" \
  --manifest "$selection_root/stage-c2-merge-manifest.json"
"${uv_prime[@]}" python scripts/merge_stage_c_warmstart.py \
  --model "$stage_c2_merged" \
  --adapter "$stage_c5_adapter" \
  --output "$model_path" \
  --manifest "$selection_root/stage-c5-merge-manifest.json"

signed_field() {
  "${uv_prime[@]}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["signed_payload_sha256"])' \
    "$1"
}
action_cases_signed="$(signed_field "$action_cases")"
root_cases_signed="$(signed_field "$root_cases")"

CUDA_VISIBLE_DEVICES=0 "${uv_prime[@]}" python scripts/run_signed_vllm_scorer.py \
  --output "$selection_root/action-scores.json" \
  --verified "$selection_root/action-scores.verified.json" \
  --expected-cases-sha256 "$action_cases_signed" \
  --expected-model "$model_path" \
  -- \
  python scripts/score_stage_c_policies_vllm.py \
    --cases "$action_cases" \
    --model "$model_path" \
    --model-name selected_initialization \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --output "$selection_root/action-scores.json" \
  >"$selection_root/action-score.log" 2>&1 &
action_pid=$!

CUDA_VISIBLE_DEVICES=1 "${uv_prime[@]}" python scripts/run_signed_vllm_scorer.py \
  --output "$selection_root/root-scores.json" \
  --verified "$selection_root/root-scores.verified.json" \
  --expected-cases-sha256 "$root_cases_signed" \
  --expected-model "$model_path" \
  --expected-analysis stage-c3-root-route-sequence-scores \
  -- \
  python scripts/score_stage_c3_root_routes_vllm.py \
    --cases "$root_cases" \
    --model "$model_path" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.7 \
    --output "$selection_root/root-scores.json" \
  >"$selection_root/root-score.log" 2>&1 &
root_pid=$!

action_status=0
root_status=0
wait "$action_pid" || action_status=$?
wait "$root_pid" || root_status=$?
test "$action_status" -eq 0
test "$root_status" -eq 0

"${uv_prime[@]}" python scripts/evaluate_stage_c5_candidate.py evaluate \
  --step 18 \
  --action-scores "$selection_root/action-scores.json" \
  --root-scores "$selection_root/root-scores.json" \
  --dataset-manifest "$dataset_manifest" \
  --output "$selection_root/support-verification.json"
support_status="$(
  "${uv_prime[@]}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$selection_root/support-verification.json"
)"
test "$support_status" = "passed"
touch "$run_root/INITIALIZATION_SUPPORT_PASS"

export REDCO_REQUIRED_MODEL_DIR="$model_path"
run_supervised() {
  local config="$1"
  local output="$2"
  local result="$3"
  "${uv_prime[@]}" python scripts/run_stage_c3_supervised_arm.py \
    --launcher "$launcher" \
    --config "$config" \
    --output-dir "$output" \
    --result "$result" \
    --mode arm \
    --invariant-timeout-seconds "$invariant_timeout"
}

structural_output="$run_root/smoke/structural-broadcast-s9900"
run_supervised \
  "$structural_config" \
  "$structural_output" \
  "$run_root/smoke/structural-invariant.json"
"${uv_prime[@]}" python scripts/verify_stage_c5_constraint_smoke.py \
  --traces "$structural_output/run_default/rollouts/step_1/train/all/traces.jsonl" \
  --batch "$structural_output/run_default/rollouts/step_1/train_rollouts.bin" \
  --root-scores "$selection_root/root-scores.json" \
  --expected-context-traces 16 \
  --output "$run_root/smoke/constrained-interface-verification.json"
smoke_status="$(
  "${uv_prime[@]}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$run_root/smoke/constrained-interface-verification.json"
)"
test "$smoke_status" = "passed"
touch "$run_root/STRUCTURAL_SMOKE_PASS"

# Frozen alternation order. An observed arm is never rerun.
while IFS='|' read -r probe arm seed; do
  config="$config_root/${probe}-${arm}-s${seed}.toml"
  output="$run_root/${probe}/${arm}-s${seed}"
  result="$run_root/invariants/${probe}-${arm}-s${seed}.json"
  test -f "$config"
  test ! -e "$output"
  run_supervised "$config" "$output" "$result"
done <<'RUNS'
confusion_irrelevant|broadcast|9901
confusion_irrelevant|sliced|9901
confusion_irrelevant|sliced|9902
confusion_irrelevant|broadcast|9902
confusion_redundant|broadcast|9903
confusion_redundant|sliced|9903
confusion_lucky|sliced|9904
confusion_lucky|broadcast|9904
RUNS

adapter_args=()
while IFS='|' read -r probe arm seed step; do
  name="${probe}--${arm}--s${seed}"
  adapter="$run_root/$probe/${arm}-s${seed}/run_default/broadcasts/step_${step}"
  test -s "$adapter/adapter_model.safetensors"
  adapter_args+=(--adapter "$name=$adapter")
done <<'ADAPTERS'
confusion_irrelevant|broadcast|9901|36
confusion_irrelevant|sliced|9901|6
confusion_irrelevant|sliced|9902|6
confusion_irrelevant|broadcast|9902|36
confusion_redundant|broadcast|9903|36
confusion_redundant|sliced|9903|6
confusion_lucky|sliced|9904|6
confusion_lucky|broadcast|9904|36
ADAPTERS

"${uv_prime[@]}" python scripts/score_stage_c_policies_vllm.py \
  --cases "$action_cases" \
  --model "$model_path" \
  --model-name warmstart \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  "${adapter_args[@]}" \
  --output "$run_root/final-policy-scores.json"
"${uv_prime[@]}" python -m redco.analysis.stage_c6_live \
  --run-root "$run_root" \
  --scores "$run_root/final-policy-scores.json" \
  --output "$run_root/frozen-decision.json"

touch "$run_root/CAMPAIGN_COMPLETE"
campaign_status=0
