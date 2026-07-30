#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
config_root="${REDCO_CONFIG_ROOT:-configs/stage-c9/rendered-v1}"
run_root="${REDCO_RUN_ROOT:-runs/stage-c9/practical-efficiency}"
launcher="${REDCO_ARM_LAUNCHER:-scripts/run_stage_c2_campaign_arm.sh}"
invariant_timeout="${REDCO_INVARIANT_TIMEOUT_SECONDS:-900}"
uv_binary="${REDCO_UV_BINARY:-uv}"
stage_c2_adapter="runs/stage-c2/warmstart-selected-v2/step_23/lora_adapters"
stage_c5_adapter="runs/stage-c5/constrained-successor-v3-selection/evidence/runs/stage-c5/warmstart-selected-v3/lora-adapters"
stage_c2_merged="runs/stage-c9/stage-c2-initialization-merged"
model_path="runs/stage-c9/selected-initialization-merged"
reference_manifest="runs/stage-c5/constrained-successor-v3-selection/evidence/runs/stage-c5/warmstart-selection-v3/candidates/step_18/merge-manifest.json"
prime_patch="patches/prime-rl-redco-stage-c9-practical-efficiency.patch"

cd "$repo_root"
test -x "$uv_binary" || command -v "$uv_binary" >/dev/null
test -s "$stage_c2_adapter/adapter_model.safetensors"
test -s "$stage_c5_adapter/adapter_model.safetensors"
test -s "$reference_manifest"
test -s "$config_root/manifest.json"
test -s "$prime_patch"
test ! -e "$stage_c2_merged"
test ! -e "$model_path"
test ! -e "$run_root"
test "$(sha256sum "$stage_c2_adapter/adapter_model.safetensors" | cut -d' ' -f1)" = \
  "28fba5d421ea611db2e0d9cd411e40a0fc2035a9a45eb0bb3be24c84947e0ab6"
test "$(sha256sum "$stage_c5_adapter/adapter_model.safetensors" | cut -d' ' -f1)" = \
  "e1d56f45485eef065bae42980427ee3c88176a5c864cbb350fa8494d0370e623"
git -C external/prime-rl apply --reverse --check "$repo_root/$prime_patch"

export REDCO_PRIME_PATCH="$prime_patch"
export REDCO_RUN_SEED=7309101
export REDCO_UV_ENVIRONMENT="${REDCO_UV_ENVIRONMENT:-/home/ubuntu/.cache/redco/stage-c9-env}"
export REDCO_UV_CACHE_DIR="${REDCO_UV_CACHE_DIR:-/home/ubuntu/.cache/redco/uv-cache}"
export UV_PROJECT_ENVIRONMENT="$REDCO_UV_ENVIRONMENT"
export UV_CACHE_DIR="$REDCO_UV_CACHE_DIR"
export PATH="$(dirname "$uv_binary"):$PATH"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export REDCO_DETERMINISTIC=1
export PYTHONHASHSEED=7309101
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

mkdir -p "$run_root/invariants" "$run_root/initialization"

# Frozen CPU/runtime contracts, including the real sender path.
"${uv_prime[@]}" pytest -q \
  tests/test_stage_c_training.py \
  tests/test_stage_c9_efficiency.py \
  external/prime-rl/tests/unit/orchestrator/test_orchestrator_setup.py \
  external/prime-rl/tests/unit/orchestrator/test_advantage.py \
  external/prime-rl/tests/unit/train/rl/test_redco_loss_cpu.py \
  external/prime-rl/tests/unit/train/models/test_multi_lora_linear.py

"${uv_prime[@]}" python scripts/merge_stage_c_warmstart.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter "$stage_c2_adapter" \
  --output "$stage_c2_merged" \
  --manifest "$run_root/initialization/stage-c2-merge-manifest.json"
"${uv_prime[@]}" python scripts/merge_stage_c_warmstart.py \
  --model "$stage_c2_merged" \
  --adapter "$stage_c5_adapter" \
  --output "$model_path" \
  --manifest "$run_root/initialization/stage-c5-merge-manifest.json"
"${uv_prime[@]}" python scripts/evaluate_stage_c6_canonical_scores.py \
  verify-model-identity \
  --reference "$reference_manifest" \
  --current "$run_root/initialization/stage-c5-merge-manifest.json" \
  --output "$run_root/initialization/model-identity.json"
test "$(
  "${uv_prime[@]}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$run_root/initialization/model-identity.json"
)" = "passed"

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

# One frozen integration smoke. Its scientific outcome cannot change the
# already-frozen arms; it only gates the two-epoch orchestration contract.
smoke_output="$run_root/smoke/local-e2-s10030"
run_supervised \
  "$config_root/smoke-local-e2-s10030.toml" \
  "$smoke_output" \
  "$run_root/invariants/smoke-local-e2-s10030.json"
"${uv_prime[@]}" python scripts/verify_stage_c9_smoke.py \
  --run-dir "$smoke_output" \
  --output "$run_root/smoke/reuse-contract.json"
touch "$run_root/INTEGRATION_SMOKE_PASS"

# Latin-balanced arm order. After any arm outcome, that arm is never rerun.
while IFS='|' read -r arm seed; do
  config="$config_root/${arm}-s${seed}.toml"
  output="$run_root/confusion_redundant/${arm}-s${seed}"
  result="$run_root/invariants/${arm}-s${seed}.json"
  test -f "$config"
  test ! -e "$output"
  run_supervised "$config" "$output" "$result"
done <<'RUNS'
local-e1|10031
local-e2|10031
branch-global-e2|10031
stock|10031
local-e2|10032
stock|10032
local-e1|10032
branch-global-e2|10032
branch-global-e2|10033
local-e1|10033
stock|10033
local-e2|10033
RUNS

adapter_args=()
for seed in 10031 10032 10033; do
  for arm in local-e1 local-e2 branch-global-e2 stock; do
    for collection in 1 2 3 4 5 6; do
      case "$arm" in
        local-e1) step="$collection" ;;
        local-e2|branch-global-e2) step="$((collection * 2))" ;;
        stock) step="$((collection * 6))" ;;
      esac
      name="${arm}--s${seed}--c${collection}"
      adapter="$run_root/confusion_redundant/${arm}-s${seed}/run_default/broadcasts/step_${step}"
      test -s "$adapter/adapter_model.safetensors"
      adapter_args+=(--adapter "$name=$adapter")
    done
  done
done

CUDA_VISIBLE_DEVICES=0 "${uv_prime[@]}" python \
  scripts/score_stage_c6_canonical_transformers.py \
  --model "$model_path" \
  --model-name warmstart \
  --action-cases configs/stage-c4/selection-action-cases.json \
  --root-cases configs/stage-c4/selection-root-cases.json \
  --action-output "$run_root/exact-checkpoint-scores.json" \
  "${adapter_args[@]}" \
  >"$run_root/exact-checkpoint-score.log" 2>&1

"${uv_prime[@]}" python -m redco.analysis.stage_c9_efficiency \
  --run-root "$run_root" \
  --scores "$run_root/exact-checkpoint-scores.json" \
  --output "$run_root/frozen-decision.json"

touch "$run_root/CAMPAIGN_COMPLETE"
campaign_status=0
