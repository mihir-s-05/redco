#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
timeout_seconds="${REDCO_SFT_TIMEOUT_SECONDS:-3600}"
prime_env="${REDCO_UV_ENVIRONMENT:-/workspace/.venv-prime-stage-c2}"
prime_cache="${REDCO_UV_CACHE_DIR:-/workspace/.uv-cache-prime-stage-c2}"
run_dir="$repo_root/runs/stage-c2/warmstart-sft-v2"
audit_dir="$repo_root/runs/stage-c2/warmstart-audit-v2"
merged_dir="$repo_root/runs/stage-c2/warmstart-merged-v2"

cd "$repo_root"
test ! -e "$run_dir"
test ! -e "$audit_dir"
test ! -e "$merged_dir"
test -s patches/prime-rl-redco-stage-c2.patch
git -C external/prime-rl apply --no-index --reverse --check \
  "$repo_root/patches/prime-rl-redco-stage-c2.patch"

mkdir -p "$audit_dir"
export UV_PROJECT_ENVIRONMENT="$prime_env"
export UV_CACHE_DIR="$prime_cache"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export XDG_CONFIG_HOME="$repo_root/.runtime-config"
export PYTHONPATH="$repo_root/src:$repo_root/scripts"
mkdir -p "$XDG_CONFIG_HOME"

uv_nvcc="$prime_env/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"
if test ! -x /usr/local/cuda/bin/nvcc && test -x "$uv_nvcc"; then
  export CUDA_HOME="${uv_nvcc%/bin/nvcc}"
fi

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader >"$audit_dir/resource-before.csv"

timeout --signal=TERM "$timeout_seconds" \
  uv run --frozen --project external/prime-rl \
  --extra flash-attn \
  sft @ "$repo_root/configs/stage-c2/warmstart-sft-v2.toml" \
  >"$audit_dir/sft-control.log" 2>&1

test -s "$run_dir/metrics.jsonl"
for step in $(seq 1 32); do
  test -s "$run_dir/weights/step_${step}/lora_adapters/adapter_model.safetensors"
  test -f "$run_dir/weights/step_${step}/STABLE"
done

adapter_args=()
for step in $(seq 1 32); do
  adapter_args+=(
    --adapter
    "sft_step_${step}=$run_dir/weights/step_${step}/lora_adapters"
  )
done
uv run --frozen --project external/prime-rl \
  python scripts/score_stage_c_policies_vllm.py \
  --cases configs/stage-c2/policy-audit-cases.json \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  "${adapter_args[@]}" \
  --output "$audit_dir/sft-policy-scores.json" \
  >"$audit_dir/vllm-score.log" 2>&1

uv run --frozen --project external/prime-rl \
  python -m redco.analysis.stage_c_warmstart select \
  --raw-scores "$audit_dir/sft-policy-scores.json" \
  --minimum-needle-mass-t2 0.08 \
  --maximum-needle-mass-t2 0.12 \
  --maximum-needle-greedy-rate 0 \
  --branch-count 11 \
  --groups-per-step 8 \
  --minimum-expected-informative-groups 4.75 \
  --output "$audit_dir/selection.json"

selected_step="$(
  uv run --frozen --project external/prime-rl python -c \
    'import json,sys; value=json.load(open(sys.argv[1])); print("" if value["selected"] is None else value["selected"]["step"])' \
    "$audit_dir/selection.json"
)"
if test -z "$selected_step"; then
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader >"$audit_dir/resource-after.csv"
  touch "$audit_dir/WARMSTART_FAILED"
  find "$audit_dir" -maxdepth 1 -type f ! -name sha256-manifest.txt \
    -print0 | sort -z | xargs -0 sha256sum >"$audit_dir/sha256-manifest.txt"
  exit 20
fi
selected_adapter="$run_dir/weights/step_${selected_step}/lora_adapters"

uv run --frozen --project external/prime-rl \
  python scripts/merge_stage_c_warmstart.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter "$selected_adapter" \
  --output "$merged_dir" \
  --manifest "$audit_dir/merge-manifest.json"

uv run --frozen --project external/prime-rl \
  python scripts/score_stage_c_policies_vllm.py \
  --cases configs/stage-c2/policy-audit-cases.json \
  --model "$merged_dir" \
  --model-name merged \
  --output "$audit_dir/merged-policy-scores.json" \
  >"$audit_dir/vllm-merged-score.log" 2>&1

uv run --frozen --project external/prime-rl \
  python -m redco.analysis.stage_c_warmstart_gate \
  --run-dir "$run_dir" \
  --raw-scores "$audit_dir/sft-policy-scores.json" \
  --selected-scores "$audit_dir/sft-policy-scores.json" \
  --merged-scores "$audit_dir/merged-policy-scores.json" \
  --expected-steps 32 \
  --minimum-needle-mass-t2 0.08 \
  --maximum-needle-mass-t2 0.12 \
  --maximum-needle-greedy-rate 0 \
  --branch-count 11 \
  --groups-per-step 8 \
  --minimum-expected-informative-groups 4.75 \
  --output "$audit_dir/gate-report.json"

uv run --frozen --project external/prime-rl \
  python -c \
  'import json,sys; value=json.load(open(sys.argv[1])); raise SystemExit(0 if value["status"] == "pass" else 21)' \
  "$audit_dir/gate-report.json"

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader >"$audit_dir/resource-after.csv"
touch "$audit_dir/WARMSTART_PASSED"
find "$audit_dir" -maxdepth 1 -type f ! -name sha256-manifest.txt \
  -print0 | sort -z | xargs -0 sha256sum >"$audit_dir/sha256-manifest.txt"
