#!/usr/bin/env bash
set -euo pipefail

cd /workspace/redco
export PATH="/workspace/redco/external/prime-rl/.venv/bin:$PATH"
export REDCO_RUN_SEED="4101"
export REDCO_DETERMINISTIC="1"
export PYTHONHASHSEED="4101"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
root="runs/stage-a/deterministic-replay"
expected_batch_sha="93cd7e67073d5965e41213cc214735bdf89e600f2082f3a69b6eaba063bd6c45"

run_arm() {
  local arm="$1"
  local batch="$root/$arm/run_default/rollouts/step_1/train_rollouts.bin"
  local actual_batch_sha
  actual_batch_sha="$(sha256sum "$batch" | cut -d' ' -f1)"
  test "$actual_batch_sha" = "$expected_batch_sha"
  echo "DETERMINISTIC_REPLAY_START $arm $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
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
  echo "DETERMINISTIC_REPLAY_DONE $arm $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
}

run_arm stock-a
run_arm stock-b

PYTHONPATH="/workspace/redco/src" \
  external/prime-rl/.venv/bin/python -c \
  'from pathlib import Path
from redco.analysis.frozen_rollout import ADAPTER_RELATIVE_PATH, _core_metrics, _sha256
root = Path("runs/stage-a/deterministic-replay")
stock_a = root / "stock-a"
stock_b = root / "stock-b"
metrics_exact = _core_metrics(stock_a / "metrics.jsonl") == _core_metrics(stock_b / "metrics.jsonl")
adapters_exact = _sha256(stock_a / ADAPTER_RELATIVE_PATH) == _sha256(stock_b / ADAPTER_RELATIVE_PATH)
print(f"DETERMINISTIC_STOCK_CHECK metrics_exact={metrics_exact} adapters_exact={adapters_exact}")
raise SystemExit(0 if metrics_exact and adapters_exact else 3)'

run_arm redco
