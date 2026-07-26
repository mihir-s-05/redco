#!/usr/bin/env bash
set -euo pipefail

cd /workspace/redco
export PATH="/workspace/redco/external/prime-rl/.venv/bin:$PATH"
export REDCO_DETERMINISTIC="1"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
root="runs/stage-a/noop-confirmation"
expected_batch_sha="93cd7e67073d5965e41213cc214735bdf89e600f2082f3a69b6eaba063bd6c45"
seeds=(5101 5102 5103 5104)

run_arm() {
  local seed="$1"
  local arm="$2"
  local pair="pair-s${seed}"
  local arm_root="$root/$pair/$arm"
  local batch="$arm_root/run_default/rollouts/step_1/train_rollouts.bin"
  local actual_batch_sha
  actual_batch_sha="$(sha256sum "$batch" | cut -d' ' -f1)"
  test "$actual_batch_sha" = "$expected_batch_sha"
  export REDCO_RUN_SEED="$seed"
  export PYTHONHASHSEED="$seed"
  echo "NOOP_CONFIRMATION_START $pair $arm $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
  mkdir -p "$arm_root/logs/trainer/torchrun"
  CUDA_VISIBLE_DEVICES=0 \
    external/prime-rl/.venv/bin/torchrun \
      --standalone \
      --role=trainer \
      --log-dir="$arm_root/logs/trainer/torchrun" \
      --local-ranks-filter=0 \
      --redirect=3 \
      --tee=3 \
      --nproc-per-node=1 \
      -m prime_rl.trainer.rl.train \
      @ "$arm_root/configs/trainer.toml" \
      >"$arm_root/logs/trainer.log" 2>&1
  test -s "$arm_root/metrics.jsonl"
  test -s \
    "$arm_root/run_default/broadcasts/step_1/adapter_model.safetensors"
  if grep -Eiq "Traceback|CUDA out of memory|process failed" \
    "$arm_root/logs/trainer.log"; then
    tail -n 100 "$arm_root/logs/trainer.log"
    exit 1
  fi
  echo "NOOP_CONFIRMATION_DONE $pair $arm $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
}

for seed in "${seeds[@]}"; do
  run_arm "$seed" stock
  run_arm "$seed" redco
done
