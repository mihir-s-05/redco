#!/usr/bin/env bash
set -euo pipefail

cd /workspace/redco
audit_dir="runs/stage-b/model-seed-audit"
mkdir -p "$audit_dir"

CUDA_VISIBLE_DEVICES=0 external/prime-rl/.venv/bin/inference \
  @ runs/stage-a/ga-micro/pilot-stock-s2101-a/configs/inference.toml \
  >"$audit_dir/inference.log" 2>&1 &
inference_pid=$!

cleanup() {
  if kill -0 "$inference_pid" 2>/dev/null; then
    kill "$inference_pid"
    wait "$inference_pid" || true
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$inference_pid" 2>/dev/null; then
    tail -n 80 "$audit_dir/inference.log"
    exit 1
  fi
  sleep 5
done

if [[ "$ready" != "1" ]]; then
  tail -n 80 "$audit_dir/inference.log"
  exit 1
fi

/root/.local/bin/uv run python -m redco.analysis.model_seed_audit \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output "$audit_dir/report.json"
