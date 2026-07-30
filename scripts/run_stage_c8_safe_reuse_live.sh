#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
run_root="${REDCO_RUN_ROOT:-runs/stage-c8/safe-reuse-live}"
batch_root="runs/stage-c8/batch-contract"
batch_audit="reports/stage-c8-batch-contract-audit-v1.json"
source_control="runs/stage-c6/credit-confusion-live-v3/confusion_redundant/sliced-s9923/run_default/control/orch.toml"
template="configs/stage-c8/safe-reuse-trainer.template.toml"
stage_c2_adapter="runs/stage-c2/warmstart-selected-v2/step_23/lora_adapters"
stage_c5_adapter="runs/stage-c5/constrained-successor-v3-selection/evidence/runs/stage-c5/warmstart-selected-v3/lora-adapters"
stage_c2_merged="$run_root/stage-c2-initialization-merged"
model_path="$run_root/selected-initialization-merged"
uv_binary="${REDCO_UV_BINARY:-uv}"

cd "$repo_root"
test ! -e "$run_root"
test -s "$batch_root/step_1/train_rollouts.bin"
test -s "$batch_root/step_2/train_rollouts.bin"
test -s "$batch_audit"
test -s "$source_control"
test -s "$template"
test -s "$stage_c2_adapter/adapter_model.safetensors"
test -s "$stage_c5_adapter/adapter_model.safetensors"
test "$(sha256sum "$batch_root/step_1/train_rollouts.bin" | cut -d' ' -f1)" = \
  "6ceb8d2d7f476bf4ec4e370262cbe6b3351a9e80b6d28758797df9162cc75ed2"
test "$(sha256sum "$batch_root/step_2/train_rollouts.bin" | cut -d' ' -f1)" = \
  "a000030d42effda06d980c06254513e8327d46d98485bb4b50905b98c1ffd7a9"
test "$(sha256sum "$stage_c2_adapter/adapter_model.safetensors" | cut -d' ' -f1)" = \
  "28fba5d421ea611db2e0d9cd411e40a0fc2035a9a45eb0bb3be24c84947e0ab6"
test "$(sha256sum "$stage_c5_adapter/adapter_model.safetensors" | cut -d' ' -f1)" = \
  "e1d56f45485eef065bae42980427ee3c88176a5c864cbb350fa8494d0370e623"
grep -Fq \
  "self.use_grouped_mm = use_grouped_mm and n_adapters > 1" \
  external/prime-rl/src/prime_rl/trainer/models/layers/lora/multi_linear.py
git -C external/prime-rl apply --reverse --check \
  "$repo_root/patches/prime-rl-redco-stage-c8-single-adapter-fallback.patch"

export UV_PROJECT_ENVIRONMENT="${REDCO_UV_ENVIRONMENT:-/home/ubuntu/.cache/redco/stage-c8-env}"
export UV_CACHE_DIR="${REDCO_UV_CACHE_DIR:-/home/ubuntu/.cache/redco/uv-cache}"
export PATH="$(dirname "$uv_binary"):$PATH"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export REDCO_DETERMINISTIC=1
export REDCO_RUN_SEED=7308101
export PYTHONHASHSEED=7308101
export PYTHONPATH="$repo_root/src:$repo_root/scripts"

uv_prime=(
  "$uv_binary" run --frozen --project external/prime-rl
  --extra flash-attn
  --with-editable "$repo_root"
  --with-editable "$repo_root/environments/redco_credit_v1"
  --with-editable "$repo_root/external/prime-rl/deps/verifiers"
)

mkdir -p "$run_root"
CUDA_VISIBLE_DEVICES=0 "${uv_prime[@]}" python \
  scripts/run_stage_c8_raw_probe.py \
  --output-dir "$run_root/raw-grouped-mm" \
  --shared-cache "/home/ubuntu/.cache/redco/stage-c8-raw-inductor"

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

config="$run_root/configs/trainer.toml"
"${uv_prime[@]}" python scripts/render_stage_c7_reuse_config.py \
  --template "$template" \
  --output "$config" \
  --output-dir "$run_root" \
  --max-steps 2 \
  --control-template "$source_control" \
  --control-output "$run_root/run_default/control/orch.toml"

for step in 1 2; do
  rollout_dir="$run_root/run_default/rollouts/step_$step"
  mkdir -p "$rollout_dir"
  cp "$batch_root/step_$step/train_rollouts.bin" \
    "$rollout_dir/train_rollouts.bin"
done

mkdir -p "$run_root/logs/trainer/torchrun"
export CUDA_LAUNCH_BLOCKING=1
export TORCHINDUCTOR_CACHE_DIR="/home/ubuntu/.cache/redco/stage-c8-trainer-inductor"
CUDA_VISIBLE_DEVICES=0 "${uv_prime[@]}" torchrun \
  --standalone \
  --role=trainer \
  --log-dir="$run_root/logs/trainer/torchrun" \
  --local-ranks-filter=0 \
  --redirect=3 \
  --tee=3 \
  --nproc-per-node=1 \
  -m prime_rl.trainer.rl.train \
  @ "$config" \
  >"$run_root/logs/trainer.log" 2>&1

test -s "$run_root/metrics.jsonl"
test -s \
  "$run_root/run_default/broadcasts/step_1/adapter_model.safetensors"
test -s \
  "$run_root/run_default/broadcasts/step_2/adapter_model.safetensors"

"${uv_prime[@]}" python -m redco.analysis.stage_c8_safe_reuse \
  --run-root "$run_root" \
  --raw-summary "$run_root/raw-grouped-mm/summary.json" \
  --batch-audit "$batch_audit" \
  --output "$run_root/frozen-result.json"
touch "$run_root/CAMPAIGN_COMPLETE"
