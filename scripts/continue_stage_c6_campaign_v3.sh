#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/home/ubuntu/redco}"
config_root="configs/stage-c6/rendered-v3"
run_root="${REDCO_RUN_ROOT:-runs/stage-c6/credit-confusion-live-v3}"
launcher="scripts/run_stage_c2_campaign_arm.sh"
uv_binary="${REDCO_UV_BINARY:-/home/ubuntu/.local/bin/uv}"
model_path="runs/stage-c6/selected-initialization-merged"
runtime_root="$run_root/initialization/runtime"
structural_output="$run_root/smoke/structural-broadcast-s9920"

cd "$repo_root"
test -f "$run_root/CAMPAIGN_TERMINAL_FAILURE"
test -f "$run_root/CANONICAL_SUPPORT_PASS"
test -f "$run_root/RUNTIME_POWER_PASS"
test ! -f "$run_root/STRUCTURAL_SMOKE_PASS"
test ! -f "$run_root/CAMPAIGN_COMPLETE"
test -d "$model_path"
test "$(
  sha256sum \
    "$structural_output/run_default/rollouts/step_1/train/all/traces.jsonl" |
    cut -d' ' -f1
)" = "e1fdbe0d4f79c6503495cb2cb6c3f0362571213f00d6d85e8de7e2d2a557017d"
test "$(
  sha256sum \
    "$structural_output/run_default/rollouts/step_1/train_rollouts.bin" |
    cut -d' ' -f1
)" = "7f4cdeaf0afb01fd4ee7e59c179777a5838ee20427dcdbec449f468d3e1886b0"
test "$(
  sha256sum \
    "$structural_output/run_default/token_exports/step_1/rank_0.jsonl" |
    cut -d' ' -f1
)" = "070fcc569141675d27648801afc9a10b4b5dc1bac576cbd6dacc30ddd517d53d"

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
export REDCO_PRIME_PATCH="patches/prime-rl-redco-stage-c6-v3.patch"
export REDCO_REQUIRED_MODEL_DIR="$model_path"

uv_prime=(
  "$uv_binary" run --frozen --project external/prime-rl
  --extra flash-attn
  --with-editable "$repo_root"
  --with-editable "$repo_root/environments/redco_credit_v1"
  --with-editable "$repo_root/external/prime-rl/deps/verifiers"
)

"${uv_prime[@]}" python scripts/verify_stage_c6_v3_interface.py \
  --traces "$structural_output/run_default/rollouts/step_1/train/all/traces.jsonl" \
  --batch "$structural_output/run_default/rollouts/step_1/train_rollouts.bin" \
  --token-exports "$structural_output/run_default/token_exports" \
  --root-scores "$runtime_root/root-scores.json" \
  --expected-context-traces 8 \
  --output "$run_root/smoke/exact-constrained-interface.json"
touch "$run_root/STRUCTURAL_SMOKE_PASS"

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
    --invariant-timeout-seconds 900
}

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
touch "$run_root/CAMPAIGN_COMPLETE_AFTER_IN_PLACE_REPAIR"
