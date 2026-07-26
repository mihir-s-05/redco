#!/usr/bin/env bash
set -euo pipefail

cd /workspace/redco
repo_root="$PWD"
run_root="runs/stage-b/rlm-trace-audit"
mkdir -p "$run_root"

if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
elif test -x "$HOME/.local/bin/uv"; then
  uv_bin="$HOME/.local/bin/uv"
else
  uv_bin="$(command -v uv)"
fi

inference_log="$run_root/inference.log"
control_log="$run_root/control.log"
traces="$run_root/live/traces.jsonl"

CUDA_VISIBLE_DEVICES=0 \
  "$uv_bin" run --frozen --project external/prime-rl \
  inference @ configs/stage-b/rlm-trace-audit-inference.toml \
  >"$inference_log" 2>&1 &
inference_pid=$!

cleanup() {
  if kill -0 "$inference_pid" 2>/dev/null; then
    kill "$inference_pid"
    wait "$inference_pid" || true
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 180); do
  if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$inference_pid" 2>/dev/null; then
    tail -n 100 "$inference_log"
    exit 1
  fi
  sleep 5
done
if test "$ready" != "1"; then
  tail -n 100 "$inference_log"
  exit 1
fi

VLLM_API_KEY=EMPTY \
  bash -c '
    cd "$1/external/prime-rl/deps/verifiers"
    UV_PROJECT_ENVIRONMENT=/tmp/redco-verifiers-env \
      "$2" run --frozen --no-dev --python 3.12 \
      --with-editable "$1/environments/redco_rlm_trace_v1" \
      python -m redco_rlm_trace_v1.run_audit \
      --output-dir "$1/runs/stage-b/rlm-trace-audit/live"
  ' _ "$repo_root" "$uv_bin" \
  >"$control_log" 2>&1

test -s "$traces"
"$uv_bin" run --frozen python -m redco.analysis.verifiers_trace_audit \
  --input "$traces" \
  --output "$run_root/audit-report.json"
