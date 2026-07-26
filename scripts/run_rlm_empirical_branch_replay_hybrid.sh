#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
cd "$repo_root"
run_root="${REDCO_RUN_ROOT:-runs/stage-b/rlm-empirical-branch-replay-hybrid}"
source_trace="$repo_root/runs/stage-b/rlm-multi-child-return-strict-trace/live/traces.jsonl"
inference_config="configs/stage-b/rlm-trace-audit-inference-strict.toml"
source_sha256="e3e9a4ca37ed0e44d0647cc3f5b45dcad83350f29d479f5fb391505a33fceea1"
mkdir -p "$run_root"

if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
elif test -x "$HOME/.local/bin/uv"; then
  uv_bin="$HOME/.local/bin/uv"
else
  uv_bin="$(command -v uv)"
fi

test "$(sha256sum "$source_trace" | cut -d ' ' -f 1)" = "$source_sha256"
test "$(
  sha256sum "$inference_config" | cut -d ' ' -f 1
)" = "ea4777a1947687a6ec3d1c7b9e84e81dbab6869c36508d68d384ea0a689edabc"
test "$(
  sha256sum patches/prime-rl-strict-tool-env-guard.patch | cut -d ' ' -f 1
)" = "1c52102bf79741d8a1791733397de26d7319b907531317c22d5ec1e6cd29c001"

if git -C external/prime-rl apply --check \
  "$repo_root/patches/prime-rl-strict-tool-env-guard.patch"
then
  git -C external/prime-rl apply \
    "$repo_root/patches/prime-rl-strict-tool-env-guard.patch"
else
  git -C external/prime-rl apply --reverse --check \
    "$repo_root/patches/prime-rl-strict-tool-env-guard.patch"
fi

inference_log="$run_root/inference.log"
resource_log="$run_root/resource.log"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader >"$resource_log"

CUDA_VISIBLE_DEVICES=0 \
  "$uv_bin" run --frozen --project external/prime-rl \
  inference @ "$inference_config" \
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
grep -Fx "REDCO_STRICT_TOOL_CALLING_ENV=1" "$inference_log"

"$uv_bin" run --frozen python -m redco.analysis.empirical_branch_replay \
  --input "$source_trace" \
  --output "$run_root/report.json" \
  --expected-source-sha256 "$source_sha256" \
  --alternatives-per-target 3 \
  --master-seed "redco-stage-b-empirical-replay-hybrid-v1" \
  --temperature 0.7 \
  --candidate-max-tokens 192 \
  --continuation-max-tokens 96 \
  >"$run_root/control.log" 2>&1

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader >>"$resource_log"
test -s "$run_root/report.json"
