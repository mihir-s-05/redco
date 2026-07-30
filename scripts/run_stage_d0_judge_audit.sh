#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
run_root="${REDCO_RUN_ROOT:-runs/stage-d0/judge-audit-v1}"
uv_binary="${REDCO_UV_BINARY:-uv}"
judge_model="Qwen/Qwen3-8B"
judge_revision="b968826d9c46dd6066d109eabc6255188de91218"
judge_port=8100
server_timeout_seconds=1200

cd "$repo_root"
test -x "$uv_binary" || command -v "$uv_binary" >/dev/null
test ! -e "$run_root"
mkdir -p "$run_root"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/home/ubuntu/.cache/redco/stage-d0-env}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/ubuntu/.cache/redco/uv-cache}"
export HF_HOME="${HF_HOME:-/home/ubuntu/.cache/redco/huggingface}"
export PATH="$(dirname "$uv_binary"):$PATH"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export STAGE_D_JUDGE_API_KEY=EMPTY

uv_prime=(
  "$uv_binary" run --frozen --project external/prime-rl
  --extra flash-attn
  --with-editable "$repo_root"
  --with-editable "$repo_root/external/prime-rl/deps/verifiers"
)

"${uv_prime[@]}" python scripts/audit_stage_d0_prerequisites.py \
  --output "$run_root/cpu-prerequisites.json"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader >"$run_root/gpu-before.csv"

judge_pid=""
finish() {
  if test -n "$judge_pid" && kill -0 "$judge_pid" 2>/dev/null; then
    kill "$judge_pid" || true
    wait "$judge_pid" || true
  fi
}
trap finish EXIT

CUDA_VISIBLE_DEVICES=1 "${uv_prime[@]}" python -m vllm.entrypoints.openai.api_server \
  --model "$judge_model" \
  --revision "$judge_revision" \
  --served-model-name "$judge_model" \
  --host 127.0.0.1 \
  --port "$judge_port" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  >"$run_root/judge-server.log" 2>&1 &
judge_pid=$!
printf '%s\n' "$judge_pid" >"$run_root/judge-server.pid"

deadline=$((SECONDS + server_timeout_seconds))
until curl --fail --silent "http://127.0.0.1:${judge_port}/health" >/dev/null; do
  if ! kill -0 "$judge_pid" 2>/dev/null; then
    wait "$judge_pid"
  fi
  if test "$SECONDS" -ge "$deadline"; then
    echo "judge server readiness timed out" >&2
    exit 1
  fi
  sleep 5
done

curl --fail --silent "http://127.0.0.1:${judge_port}/v1/models" \
  >"$run_root/models.json"
"${uv_prime[@]}" python scripts/run_stage_d_judge_calibration.py \
  --base-url "http://127.0.0.1:${judge_port}/v1" \
  --model "$judge_model" \
  --repeats 3 \
  --output "$run_root/responses.jsonl"
"${uv_prime[@]}" python scripts/audit_stage_d_judge_calibration.py \
  --responses "$run_root/responses.jsonl" \
  --output "$run_root/decision.json" \
  --repeats 3 \
  --strong-threshold 7 \
  --minimum-balanced-accuracy 0.8

test "$(
  "${uv_prime[@]}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
    "$run_root/decision.json"
)" = "pass"

nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader >"$run_root/gpu-after.csv"
touch "$run_root/JUDGE_AUDIT_PASS"
