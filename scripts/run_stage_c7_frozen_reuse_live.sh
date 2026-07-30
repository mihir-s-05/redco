#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
run_root="${REDCO_RUN_ROOT:-runs/stage-c7/frozen-reuse-live}"
source_batch="runs/stage-c6/credit-confusion-live-v3/confusion_redundant/sliced-s9923/run_default/rollouts/step_1/train_rollouts.bin"
source_control="runs/stage-c6/credit-confusion-live-v3/confusion_redundant/sliced-s9923/run_default/control/orch.toml"
template="configs/stage-c7/frozen-reuse-trainer.template.toml"
stage_c2_adapter="runs/stage-c2/warmstart-selected-v2/step_23/lora_adapters"
stage_c5_adapter="runs/stage-c5/constrained-successor-v3-selection/evidence/runs/stage-c5/warmstart-selected-v3/lora-adapters"
stage_c2_merged="$run_root/stage-c2-initialization-merged"
model_path="$run_root/selected-initialization-merged"
uv_binary="${REDCO_UV_BINARY:-uv}"

cd "$repo_root"
test ! -e "$run_root"
test -s "$source_batch"
test -s "$source_control"
test -s "$template"
test -s "$stage_c2_adapter/adapter_model.safetensors"
test -s "$stage_c5_adapter/adapter_model.safetensors"
test "$(sha256sum "$source_batch" | cut -d' ' -f1)" = \
  "6ceb8d2d7f476bf4ec4e370262cbe6b3351a9e80b6d28758797df9162cc75ed2"
test "$(sha256sum "$stage_c2_adapter/adapter_model.safetensors" | cut -d' ' -f1)" = \
  "28fba5d421ea611db2e0d9cd411e40a0fc2035a9a45eb0bb3be24c84947e0ab6"
test "$(sha256sum "$stage_c5_adapter/adapter_model.safetensors" | cut -d' ' -f1)" = \
  "e1d56f45485eef065bae42980427ee3c88176a5c864cbb350fa8494d0370e623"
git -C external/prime-rl apply --no-index --reverse --check \
  "$repo_root/patches/prime-rl-redco-stage-c7-practical-loss.patch"

export UV_PROJECT_ENVIRONMENT="${REDCO_UV_ENVIRONMENT:-/home/ubuntu/.cache/redco/stage-c7-env}"
export UV_CACHE_DIR="${REDCO_UV_CACHE_DIR:-/home/ubuntu/.cache/redco/uv-cache}"
export PATH="$(dirname "$uv_binary"):$PATH"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export REDCO_DETERMINISTIC=1
export REDCO_RUN_SEED=7307101
export PYTHONHASHSEED=7307101
export PYTHONPATH="$repo_root/src:$repo_root/scripts"

uv_prime=(
  "$uv_binary" run --frozen --project external/prime-rl
  --extra flash-attn
  --with-editable "$repo_root"
  --with-editable "$repo_root/environments/redco_credit_v1"
  --with-editable "$repo_root/external/prime-rl/deps/verifiers"
)

mkdir -p "$run_root"
"${uv_prime[@]}" python scripts/merge_stage_c_warmstart.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter "$stage_c2_adapter" \
  --output "$stage_c2_merged" \
  --manifest "$run_root/stage-c2-merge-manifest.json"
"${uv_prime[@]}" python scripts/merge_stage_c_warmstart.py \
  --model "$stage_c2_merged" \
  --adapter "$stage_c5_adapter" \
  --output "$model_path" \
  --manifest "$run_root/stage-c5-merge-manifest.json"

for updates in 1 2 3; do
  arm="$run_root/reuse-$updates"
  config="$arm/configs/trainer.toml"
  "${uv_prime[@]}" python scripts/render_stage_c7_reuse_config.py \
    --template "$template" \
    --output "$config" \
    --output-dir "$arm" \
    --max-steps "$updates" \
    --control-template "$source_control" \
    --control-output "$arm/run_default/control/orch.toml"
  for step in $(seq 1 "$updates"); do
    rollout_dir="$arm/run_default/rollouts/step_$step"
    mkdir -p "$rollout_dir"
    cp "$source_batch" "$rollout_dir/train_rollouts.bin"
    test "$(sha256sum "$rollout_dir/train_rollouts.bin" | cut -d' ' -f1)" = \
      "6ceb8d2d7f476bf4ec4e370262cbe6b3351a9e80b6d28758797df9162cc75ed2"
  done
  mkdir -p "$arm/logs/trainer/torchrun"
  CUDA_VISIBLE_DEVICES=0 "${uv_prime[@]}" torchrun \
    --standalone \
    --role=trainer \
    --log-dir="$arm/logs/trainer/torchrun" \
    --local-ranks-filter=0 \
    --redirect=3 \
    --tee=3 \
    --nproc-per-node=1 \
    -m prime_rl.trainer.rl.train \
    @ "$config" \
    >"$arm/logs/trainer.log" 2>&1
  test -s "$arm/metrics.jsonl"
  test -s \
    "$arm/run_default/broadcasts/step_$updates/adapter_model.safetensors"
done

CUDA_VISIBLE_DEVICES=0 "${uv_prime[@]}" python \
  scripts/score_stage_c6_canonical_transformers.py \
  --model "$model_path" \
  --model-name warmstart \
  --action-cases configs/stage-c4/selection-action-cases.json \
  --root-cases configs/stage-c4/selection-root-cases.json \
  --action-output "$run_root/action-scores.json" \
  --adapter "reuse-1=$run_root/reuse-1/run_default/broadcasts/step_1" \
  --adapter "reuse-2=$run_root/reuse-2/run_default/broadcasts/step_2" \
  --adapter "reuse-3=$run_root/reuse-3/run_default/broadcasts/step_3" \
  >"$run_root/action-score.log" 2>&1

"${uv_prime[@]}" python -m redco.analysis.stage_c7_live_reuse \
  --run-root "$run_root" \
  --scores "$run_root/action-scores.json" \
  --output "$run_root/frozen-result.json"
touch "$run_root/CAMPAIGN_COMPLETE"
