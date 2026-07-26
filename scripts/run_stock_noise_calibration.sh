#!/usr/bin/env bash
set -euo pipefail

cd /workspace/redco
export PATH="/workspace/redco/external/prime-rl/.venv/bin:$PATH"
export REDCO_RUN_SEED="4101"
export REDCO_DETERMINISTIC="1"
export PYTHONHASHSEED="4101"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
root="runs/stage-a/stock-noise-calibration"
expected_batch_sha="93cd7e67073d5965e41213cc214735bdf89e600f2082f3a69b6eaba063bd6c45"
names=(
  stock-c01
  stock-c02
  stock-c03
  stock-c04
  stock-c05
  stock-c06
  stock-c07
  stock-c08
)

for name in "${names[@]}"; do
  batch="$root/$name/run_default/rollouts/step_1/train_rollouts.bin"
  actual_batch_sha="$(sha256sum "$batch" | cut -d' ' -f1)"
  test "$actual_batch_sha" = "$expected_batch_sha"
  echo "STOCK_CALIBRATION_START $name $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "$root/$name/logs/trainer/torchrun"
  CUDA_VISIBLE_DEVICES=0 \
    external/prime-rl/.venv/bin/torchrun \
      --standalone \
      --role=trainer \
      --log-dir="$root/$name/logs/trainer/torchrun" \
      --local-ranks-filter=0 \
      --redirect=3 \
      --tee=3 \
      --nproc-per-node=1 \
      -m prime_rl.trainer.rl.train \
      @ "$root/$name/configs/trainer.toml" \
      >"$root/$name/logs/trainer.log" 2>&1
  test -s "$root/$name/metrics.jsonl"
  test -s \
    "$root/$name/run_default/broadcasts/step_1/adapter_model.safetensors"
  if grep -Eiq "Traceback|CUDA out of memory|process failed" \
    "$root/$name/logs/trainer.log"; then
    tail -n 100 "$root/$name/logs/trainer.log"
    exit 1
  fi
  echo "STOCK_CALIBRATION_DONE $name $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
done
