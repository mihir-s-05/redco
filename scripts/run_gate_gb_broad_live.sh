#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
cd "$repo_root"
run_root="${REDCO_RUN_ROOT:-runs/stage-b/gate-gb-broad-live}"
source_trace="$repo_root/runs/stage-b/rlm-multi-child-return-strict-trace/live/traces.jsonl"
inference_config="configs/stage-b/rlm-trace-audit-inference-strict.toml"
static_report="runs/stage-b/gate-gb-cpu/report.json"
dynamic_report="reports/dynamic-rlm-raf-cpu-result-2026-07-26.json"
stochastic_report="reports/gate-gb-stochastic-replay-cpu-result-2026-07-26.json"
strict_report="reports/rlm-multi-child-return-strict-trace-result-2026-07-26.json"
source_sha256="e3e9a4ca37ed0e44d0647cc3f5b45dcad83350f29d479f5fb391505a33fceea1"
mkdir -p "$run_root"

if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
elif test -x "$HOME/.local/bin/uv"; then
  uv_bin="$HOME/.local/bin/uv"
else
  uv_bin="$(command -v uv)"
fi

verify_sha256() {
  local path="$1"
  local expected="$2"
  test "$(sha256sum "$path" | cut -d ' ' -f 1)" = "$expected"
}

verify_sha256 "$source_trace" "$source_sha256"
verify_sha256 "$inference_config" \
  "ea4777a1947687a6ec3d1c7b9e84e81dbab6869c36508d68d384ea0a689edabc"
verify_sha256 patches/prime-rl-strict-tool-env-guard.patch \
  "1c52102bf79741d8a1791733397de26d7319b907531317c22d5ec1e6cd29c001"
verify_sha256 src/redco/analysis/empirical_branch_replay.py \
  "4be7db532627375e18a354bd6570a7b7a50a706e303bd844adde9110e344734e"
verify_sha256 src/redco/analysis/gate_gb_aggregate.py \
  "7c4e52001be0a2dde32d75bbd62445f1bd17f5a0013542cc9cde7e58fe3f05e2"
verify_sha256 "$static_report" \
  "754eea47223d5076465c78d321cb7bb01ccba9647b19fdfac93df0232d3a3221"
verify_sha256 "$dynamic_report" \
  "9dcbb5983204f5b28c9f26877e34b2bc25d4ff5555f0219ef541b3672d5e9951"
verify_sha256 "$stochastic_report" \
  "18a46bdd53523e4174cddcd6418722723e40620591950a9d11e473d913d877cf"
verify_sha256 "$strict_report" \
  "22bff1313759c38ae290d1f07bd21b24123eb9065e4e9d6afb7895776fdbd347"

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
gpu_samples="$run_root/gpu-samples.csv"
inference_pid=""
sampler_pid=""

cleanup() {
  if test -n "$sampler_pid" && kill -0 "$sampler_pid" 2>/dev/null; then
    kill "$sampler_pid"
    wait "$sampler_pid" || true
  fi
  if test -n "$inference_pid" && kill -0 "$inference_pid" 2>/dev/null; then
    kill "$inference_pid"
    wait "$inference_pid" || true
  fi
}
trap cleanup EXIT

nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader >"$resource_log"
(
  while true; do
    epoch="$(date +%s.%N)"
    metrics="$(
      nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits
    )"
    printf '%s,%s\n' "$epoch" "$metrics"
    sleep 1
  done
) >"$gpu_samples" &
sampler_pid=$!

CUDA_VISIBLE_DEVICES=0 \
  "$uv_bin" run --frozen --project external/prime-rl \
  inference @ "$inference_config" \
  >"$inference_log" 2>&1 &
inference_pid=$!

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

timeout --signal=TERM 10800 \
  "$uv_bin" run --frozen python -m redco.analysis.empirical_branch_replay \
  --input "$source_trace" \
  --output "$run_root/broad-report.json" \
  --expected-source-sha256 "$source_sha256" \
  --alternatives-per-target 1024 \
  --master-seed "redco-stage-b-gate-gb-broad-v1" \
  --temperature 0.7 \
  --candidate-max-tokens 192 \
  --continuation-max-tokens 96 \
  --minimum-distinct-candidate-fraction 0.5 \
  --progress-every 64 \
  >"$run_root/control.log" 2>&1

kill "$sampler_pid"
wait "$sampler_pid" || true
sampler_pid=""
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader >>"$resource_log"

"$uv_bin" run --frozen python -m redco.analysis.gate_gb_aggregate \
  --static "$static_report" \
  --dynamic "$dynamic_report" \
  --stochastic "$stochastic_report" \
  --strict-trace "$strict_report" \
  --broad "$run_root/broad-report.json" \
  --gpu-samples "$gpu_samples" \
  --expected-static-sha256 \
    "754eea47223d5076465c78d321cb7bb01ccba9647b19fdfac93df0232d3a3221" \
  --expected-dynamic-sha256 \
    "9dcbb5983204f5b28c9f26877e34b2bc25d4ff5555f0219ef541b3672d5e9951" \
  --expected-stochastic-sha256 \
    "18a46bdd53523e4174cddcd6418722723e40620591950a9d11e473d913d877cf" \
  --expected-strict-trace-sha256 \
    "22bff1313759c38ae290d1f07bd21b24123eb9065e4e9d6afb7895776fdbd347" \
  --minimum-live-pairs 4096 \
  --live-confidence 0.9 \
  --overall-reward-margin 0.04 \
  --per-target-reward-margin 0.08 \
  --minimum-distinct-candidate-fraction 0.5 \
  --maximum-sliced-policy-event-fraction 0.41 \
  --minimum-gpu-samples 60 \
  --output "$run_root/gate-report.json" \
  >"$run_root/aggregate.log" 2>&1

test -s "$run_root/broad-report.json"
test -s "$run_root/gate-report.json"
test -s "$gpu_samples"
