#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
run_root="${REDCO_RUN_ROOT:-runs/stage-d/e2-4b-live-v1}"
model_root="$repo_root/.cache/models/qwen3-4b-instruct-2507-cdbee75f"
model_revision="cdbee75f17c01a7cc42f958dc650907174af0554"
uv_binary="${REDCO_UV_BINARY:-uv}"
trainer_config="configs/stage-d/stage-d-e2-trainer-v1.toml"
control_config="configs/stage-d/stage-d-e2-control-v1.toml"
golden_root="configs/stage-d/e2-golden-v1"

cd "$repo_root"
test ! -e "$run_root"
test -s "$trainer_config"
test -s "$control_config"
test -s "$golden_root/sealed-training-batch.json"
test -s "$golden_root/golden-manifest.json"
test "$(git -C external/prime-rl rev-parse HEAD)" = \
  "3b22dd951cad1036d1fe8dd0a0bfc40807a9b360"
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)" -eq 1
git -C external/prime-rl apply --reverse --check \
  "$repo_root/patches/prime-rl-redco-stage-c9-practical-efficiency.patch"
git -C external/prime-rl apply --reverse --check \
  "$repo_root/patches/prime-rl-stage-d-live-update-gate-v1.patch"

export UV_PROJECT_ENVIRONMENT="${REDCO_UV_ENVIRONMENT:-/workspace/.cache/redco/stage-d-e2-env}"
export UV_CACHE_DIR="${REDCO_UV_CACHE_DIR:-/workspace/.cache/redco/uv-cache}"
export PATH="$(dirname "$uv_binary"):$PATH"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export REDCO_DETERMINISTIC=1
export REDCO_RUN_SEED=7400201
export PYTHONHASHSEED=7400201
export PYTHONPATH="$repo_root/src:$repo_root/scripts"

uv_prime=(
  "$uv_binary" run --frozen --project external/prime-rl
  --extra flash-attn
  --with-editable "$repo_root"
  --with-editable "$repo_root/external/prime-rl/deps/verifiers"
)

"${uv_prime[@]}" pytest -q \
  tests/test_stage_d_live_update.py \
  tests/test_stage_d_live_update_torch.py \
  tests/test_stage_d_training_bridge.py

mkdir -p "$run_root"
"${uv_prime[@]}" python scripts/stage_d_e2_control.py parse-configs \
  --trainer-config "$trainer_config" \
  --control-config "$control_config" \
  --output "$run_root/prime-config-parse.json"

test ! -e "$model_root"
"${uv_prime[@]}" python -c \
  'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen3-4B-Instruct-2507", revision=sys.argv[1], local_dir=sys.argv[2])' \
  "$model_revision" "$model_root"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p \
  "$run_root/run_default/control" \
  "$run_root/receipts" \
  "$run_root/logs/trainer/torchrun"
cp "$control_config" "$run_root/run_default/control/orch.toml"

"${uv_prime[@]}" python scripts/stage_d_e2_control.py base-manifest \
  --model-root "$model_root" \
  --revision "$model_revision" \
  --output "$run_root/base-snapshot-manifest.json"
"${uv_prime[@]}" python scripts/stage_d_e2_control.py prepare \
  --sealed-batch "$golden_root/sealed-training-batch.json" \
  --golden-manifest "$golden_root/golden-manifest.json" \
  --trainer-config "$trainer_config" \
  --control-config "$run_root/run_default/control/orch.toml" \
  --base-manifest "$run_root/base-snapshot-manifest.json" \
  --rollout "$run_root/run_default/rollouts/step_1/train_rollouts.bin" \
  --binding "$run_root/live-binding.json" \
  --preflight "$run_root/preflight.json"

export REDCO_LIVE_UPDATE_BINDING="$repo_root/$run_root/live-binding.json"
export REDCO_LIVE_UPDATE_RECEIPTS="$repo_root/$run_root/receipts"
CUDA_VISIBLE_DEVICES=0 "${uv_prime[@]}" torchrun \
  --standalone \
  --role=trainer \
  --log-dir="$run_root/logs/trainer/torchrun" \
  --local-ranks-filter=0 \
  --redirect=3 \
  --tee=3 \
  --nproc-per-node=1 \
  -m prime_rl.trainer.rl.train \
  @ "$trainer_config" \
  >"$run_root/logs/trainer.log" 2>&1

touch "$run_root/TRAINER_COMPLETE"
