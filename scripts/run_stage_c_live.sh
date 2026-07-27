#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
config_path="${1:?usage: run_stage_c_live.sh CONFIG OUTPUT_DIR}"
output_dir="${2:?usage: run_stage_c_live.sh CONFIG OUTPUT_DIR}"
timeout_seconds="${REDCO_TIMEOUT_SECONDS:-10800}"
run_seed="${REDCO_RUN_SEED:-7202602}"

cd "$repo_root"
test -f "$config_path"
test ! -e "$output_dir"

test -s patches/prime-rl-redco-stage-c.patch
git -C external/prime-rl apply --no-index --reverse --check \
  "$repo_root/patches/prime-rl-redco-stage-c.patch"

mkdir -p "$(dirname "$output_dir")"
control_log="${output_dir}.control.log"
resource_log="${output_dir}.resource.log"
gpu_samples="${output_dir}.gpu-samples.csv"
sampler_pid=""

cleanup() {
  if test -n "$sampler_pid" && kill -0 "$sampler_pid" 2>/dev/null; then
    kill "$sampler_pid"
    wait "$sampler_pid" || true
  fi
}
trap cleanup EXIT

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader >"$resource_log"
(
  while true; do
    epoch="$(date +%s.%N)"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
      --format=csv,noheader,nounits |
      while IFS= read -r sample; do
        printf '%s,%s\n' "$epoch" "$sample"
      done
    sleep 2
  done
) >"$gpu_samples" &
sampler_pid=$!

export REDCO_RUN_SEED="$run_seed"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export UV_PROJECT_ENVIRONMENT="${REDCO_UV_ENVIRONMENT:-/workspace/.venv-prime-stage-c}"
export UV_CACHE_DIR="${REDCO_UV_CACHE_DIR:-/workspace/.uv-cache-prime-stage-c}"

timeout --signal=TERM "$timeout_seconds" \
  uv run --frozen --project external/prime-rl \
  --extra flash-attn \
  --with-editable "$repo_root" \
  --with-editable "$repo_root/environments/redco_credit_v1" \
  --with-editable "$repo_root/external/prime-rl/deps/verifiers" \
  rl @ "$config_path" --output-dir "$output_dir" \
  >"$control_log" 2>&1

kill "$sampler_pid"
wait "$sampler_pid" || true
sampler_pid=""
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader >>"$resource_log"

test -s "$control_log"
test -s "$gpu_samples"
test -s "$output_dir/metrics.jsonl"
