#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
config_root="${REDCO_CONFIG_ROOT:-configs/stage-c6/rendered-v3}"
run_root="${REDCO_RUN_ROOT:-runs/stage-c6/credit-confusion-live-v3}"
launcher="${REDCO_ARM_LAUNCHER:-scripts/run_stage_c2_campaign_arm.sh}"
evidence_root="${REDCO_RESUME_EVIDENCE_ROOT:?v2 initialization evidence is required}"
invariant_timeout="${REDCO_INVARIANT_TIMEOUT_SECONDS:-900}"
uv_binary="${REDCO_UV_BINARY:-uv}"
stage_c2_adapter="runs/stage-c2/warmstart-selected-v2/step_23/lora_adapters"
stage_c5_adapter="runs/stage-c5/constrained-successor-v3-selection/evidence/runs/stage-c5/warmstart-selected-v3/lora-adapters"
stage_c2_merged="runs/stage-c6/stage-c2-initialization-merged"
model_path="runs/stage-c6/selected-initialization-merged"
reference_manifest="runs/stage-c5/constrained-successor-v3-selection/evidence/runs/stage-c5/warmstart-selection-v3/candidates/step_18/merge-manifest.json"
selection_root="$run_root/initialization"
canonical_root="$selection_root/canonical"
runtime_root="$selection_root/runtime"
dataset_manifest="datasets/stage-c4/factorized-warmstart-manifest.json"
prime_patch="patches/prime-rl-redco-stage-c6-v3.patch"
structural_config="$config_root/structural-broadcast-s9920.toml"

cd "$repo_root"
test -x "$uv_binary" || command -v "$uv_binary" >/dev/null
test -s "$stage_c2_adapter/adapter_model.safetensors"
test -s "$stage_c5_adapter/adapter_model.safetensors"
test -s "$reference_manifest"
test -s "$dataset_manifest"
test -s "$prime_patch"
test -f "$structural_config"
test -d "$evidence_root/canonical"
test -d "$evidence_root/runtime"
test ! -e "$stage_c2_merged"
test ! -e "$model_path"
test ! -e "$run_root"

test "$(sha256sum "$stage_c2_adapter/adapter_model.safetensors" | cut -d' ' -f1)" = \
  "28fba5d421ea611db2e0d9cd411e40a0fc2035a9a45eb0bb3be24c84947e0ab6"
test "$(sha256sum "$stage_c5_adapter/adapter_model.safetensors" | cut -d' ' -f1)" = \
  "e1d56f45485eef065bae42980427ee3c88176a5c864cbb350fa8494d0370e623"
git -C external/prime-rl apply --no-index --reverse --check \
  "$repo_root/$prime_patch"

export REDCO_PRIME_PATCH="$prime_patch"
export REDCO_RUN_SEED=7203201
export REDCO_UV_ENVIRONMENT="${REDCO_UV_ENVIRONMENT:-/home/ubuntu/.cache/redco/stage-c6-v3-env}"
export REDCO_UV_CACHE_DIR="${REDCO_UV_CACHE_DIR:-/home/ubuntu/.cache/redco/uv-cache}"
export UV_PROJECT_ENVIRONMENT="$REDCO_UV_ENVIRONMENT"
export UV_CACHE_DIR="$REDCO_UV_CACHE_DIR"
export PATH="$(dirname "$uv_binary"):$PATH"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export XDG_CONFIG_HOME="$repo_root/.runtime-config"
export PYTHONPATH="$repo_root/src:$repo_root/scripts"

uv_prime=(
  "$uv_binary" run --frozen --project external/prime-rl
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

mkdir -p "$canonical_root" "$runtime_root" "$run_root/smoke" \
  "$run_root/invariants"

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
"${uv_prime[@]}" python scripts/evaluate_stage_c6_canonical_scores.py \
  verify-model-identity \
  --reference "$reference_manifest" \
  --current "$selection_root/stage-c5-merge-manifest.json" \
  --output "$selection_root/model-identity.json"
test "$(
  "${uv_prime[@]}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$selection_root/model-identity.json"
)" = "passed"

# V2's completed, signed initialization measurements are inherited, never rerun.
cp -a "$evidence_root/canonical/." "$canonical_root/"
cp -a "$evidence_root/runtime/." "$runtime_root/"
for replicate in 1 2 3; do
  replicate_root="$canonical_root/replicate_$replicate"
  "${uv_prime[@]}" python scripts/evaluate_stage_c5_candidate.py evaluate \
    --step 18 \
    --action-scores "$replicate_root/action-scores.json" \
    --root-scores "$replicate_root/root-scores.json" \
    --dataset-manifest "$dataset_manifest" \
    --output "$replicate_root/support-verification.json"
  test "$(
    "${uv_prime[@]}" python -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "$replicate_root/support-verification.json"
  )" = "passed"
done
"${uv_prime[@]}" python scripts/evaluate_stage_c6_canonical_scores.py \
  verify-replicates \
  --action "$canonical_root/replicate_1/action-scores.json" \
  --action "$canonical_root/replicate_2/action-scores.json" \
  --action "$canonical_root/replicate_3/action-scores.json" \
  --root "$canonical_root/replicate_1/root-scores.json" \
  --root "$canonical_root/replicate_2/root-scores.json" \
  --root "$canonical_root/replicate_3/root-scores.json" \
  --output "$canonical_root/reproducibility.json"
test "$(
  "${uv_prime[@]}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$canonical_root/reproducibility.json"
)" = "passed"
touch "$run_root/CANONICAL_SUPPORT_PASS"

"${uv_prime[@]}" python scripts/evaluate_stage_c5_candidate.py evaluate \
  --step 18 \
  --action-scores "$runtime_root/action-scores.json" \
  --root-scores "$runtime_root/root-scores.json" \
  --dataset-manifest "$dataset_manifest" \
  --output "$runtime_root/support-verification.json"
"${uv_prime[@]}" python scripts/evaluate_stage_c6_canonical_scores.py \
  verify-runtime-support \
  --candidate "$runtime_root/support-verification.json" \
  --output "$runtime_root/runtime-support.json"
test "$(
  "${uv_prime[@]}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$runtime_root/runtime-support.json"
)" = "passed"
touch "$run_root/RUNTIME_POWER_PASS"

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

structural_output="$run_root/smoke/structural-broadcast-s9920"
run_supervised \
  "$structural_config" \
  "$structural_output" \
  "$run_root/smoke/structural-invariant.json"
"${uv_prime[@]}" python scripts/verify_stage_c6_v3_interface.py \
  --traces "$structural_output/run_default/rollouts/step_1/train/all/traces.jsonl" \
  --batch "$structural_output/run_default/rollouts/step_1/train_rollouts.bin" \
  --token-exports "$structural_output/run_default/token_exports" \
  --root-scores "$runtime_root/root-scores.json" \
  --expected-context-traces 8 \
  --output "$run_root/smoke/exact-constrained-interface.json"
touch "$run_root/STRUCTURAL_SMOKE_PASS"

while IFS='|' read -r probe arm seed; do
  config="$config_root/${probe}-${arm}-s${seed}.toml"
  output="$run_root/${probe}/${arm}-s${seed}"
  result="$run_root/invariants/${probe}-${arm}-s${seed}.json"
  test -f "$config"
  test ! -e "$output"
  run_supervised "$config" "$output" "$result"
done <<'RUNS'
confusion_irrelevant|broadcast|9921
confusion_irrelevant|sliced|9921
confusion_irrelevant|sliced|9922
confusion_irrelevant|broadcast|9922
confusion_redundant|broadcast|9923
confusion_redundant|sliced|9923
confusion_lucky|sliced|9924
confusion_lucky|broadcast|9924
RUNS

adapter_args=()
while IFS='|' read -r probe arm seed step; do
  name="${probe}--${arm}--s${seed}"
  adapter="$run_root/$probe/${arm}-s${seed}/run_default/broadcasts/step_${step}"
  test -s "$adapter/adapter_model.safetensors"
  adapter_args+=(--adapter "$name=$adapter")
done <<'ADAPTERS'
confusion_irrelevant|broadcast|9921|36
confusion_irrelevant|sliced|9921|6
confusion_irrelevant|sliced|9922|6
confusion_irrelevant|broadcast|9922|36
confusion_redundant|broadcast|9923|36
confusion_redundant|sliced|9923|6
confusion_lucky|sliced|9924|6
confusion_lucky|broadcast|9924|36
ADAPTERS

CUDA_VISIBLE_DEVICES=0 "${uv_prime[@]}" python \
  scripts/score_stage_c6_canonical_transformers.py \
  --model "$model_path" \
  --model-name warmstart \
  --action-cases configs/stage-c4/selection-action-cases.json \
  --root-cases configs/stage-c4/selection-root-cases.json \
  --action-output "$run_root/final-policy-scores.json" \
  "${adapter_args[@]}" \
  >"$run_root/final-policy-score.log" 2>&1
"${uv_prime[@]}" python -m redco.analysis.stage_c6_v3_live \
  --run-root "$run_root" \
  --scores "$run_root/final-policy-scores.json" \
  --output "$run_root/frozen-decision.json"

touch "$run_root/CAMPAIGN_COMPLETE"
campaign_status=0
