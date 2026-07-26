#!/usr/bin/env bash
set -euo pipefail

cd /workspace/redco
export PATH="/workspace/redco/external/prime-rl/.venv/bin:$PATH"
export REDCO_RUN_SEED="4101"
root="runs/stage-a/frozen-rollout"
expected_batch_sha="93cd7e67073d5965e41213cc214735bdf89e600f2082f3a69b6eaba063bd6c45"

for arm in stock-a stock-b redco; do
  batch="$root/$arm/run_default/rollouts/step_1/train_rollouts.bin"
  actual_batch_sha="$(sha256sum "$batch" | cut -d' ' -f1)"
  test "$actual_batch_sha" = "$expected_batch_sha"
  echo "FROZEN_REPLAY_START $arm $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "$root/$arm/logs/trainer/torchrun"
  CUDA_VISIBLE_DEVICES=0 \
    external/prime-rl/.venv/bin/torchrun \
      --standalone \
      --role=trainer \
      --log-dir="$root/$arm/logs/trainer/torchrun" \
      --local-ranks-filter=0 \
      --redirect=3 \
      --tee=3 \
      --nproc-per-node=1 \
      -m prime_rl.trainer.rl.train \
      @ "$root/$arm/configs/trainer.toml" \
      >"$root/$arm/logs/trainer.log" 2>&1
  test -s "$root/$arm/metrics.jsonl"
  test -s \
    "$root/$arm/run_default/broadcasts/step_1/adapter_model.safetensors"
  if grep -Eiq "Traceback|CUDA out of memory|process failed" \
    "$root/$arm/logs/trainer.log"; then
    tail -n 100 "$root/$arm/logs/trainer.log"
    exit 1
  fi
  echo "FROZEN_REPLAY_DONE $arm $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
done
