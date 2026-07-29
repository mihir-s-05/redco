#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
timeout_seconds="${REDCO_SFT_TIMEOUT_SECONDS:-3600}"
prime_env="${REDCO_UV_ENVIRONMENT:-/workspace/.venv-prime-stage-c4}"
prime_cache="${REDCO_UV_CACHE_DIR:-/workspace/.uv-cache-prime-stage-c4}"
config="configs/stage-c4/factorized-warmstart-sft-v4.toml"
dataset="datasets/stage-c4/factorized-warmstart.jsonl"
dataset_manifest="datasets/stage-c4/factorized-warmstart-manifest.json"
action_cases="configs/stage-c4/selection-action-cases.json"
root_cases="configs/stage-c4/selection-root-cases.json"
stage_c2_adapter="runs/stage-c2/warmstart-selected-v2/step_23/lora_adapters"
base_merged="runs/stage-c4/base-merged-step23-v4"
sft_run="runs/stage-c4/warmstart-sft-v4"
selection_root="runs/stage-c4/warmstart-selection-v4"
selected_root="runs/stage-c4/warmstart-selected-v4"
candidate_merged="$selection_root/work/merged-candidate"

cd "$repo_root"
test -f "$config"
test -s "$dataset"
test -s "$dataset_manifest"
test -s "$action_cases"
test -s "$root_cases"
test -s "$stage_c2_adapter/adapter_model.safetensors"
test -s patches/prime-rl-redco-stage-c3-v3.patch
test ! -e "$base_merged"
test ! -e "$sft_run"
test ! -e "$selection_root"
test ! -e "$selected_root"
git -C external/prime-rl apply --no-index --reverse --check \
  "$repo_root/patches/prime-rl-redco-stage-c3-v3.patch"

export UV_PROJECT_ENVIRONMENT="$prime_env"
export UV_CACHE_DIR="$prime_cache"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export XDG_CONFIG_HOME="$repo_root/.runtime-config"
export PYTHONPATH="$repo_root/src:$repo_root/scripts"
mkdir -p "$XDG_CONFIG_HOME" "$selection_root/candidates" "$selection_root/work"

uv_prime=(
  uv run --frozen --project external/prime-rl
  --extra flash-attn
  --with-editable "$repo_root"
)

selection_status=1
finish() {
  if test "$selection_status" -ne 0; then
    touch "$selection_root/SELECTION_TERMINAL_FAILURE"
  fi
}
trap finish EXIT

cleanup_candidate_merged() {
  local expected="$repo_root/runs/stage-c4/warmstart-selection-v4/work/merged-candidate"
  local resolved="$repo_root/$candidate_merged"
  if test "$resolved" != "$expected"; then
    echo "Refusing unsafe candidate cleanup: $resolved" >&2
    exit 90
  fi
  if test -e "$candidate_merged"; then
    rm -rf -- "$candidate_merged"
  fi
}

signed_field() {
  "${uv_prime[@]}" python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["signed_payload_sha256"])' \
    "$1"
}

"${uv_prime[@]}" python -m redco.analysis.stage_c4_warmstart audit \
  --dataset "$dataset" \
  --output "$selection_root/dataset-audit.json"

"${uv_prime[@]}" python scripts/merge_stage_c_warmstart.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter "$stage_c2_adapter" \
  --output "$base_merged" \
  --manifest "$selection_root/base-merge-manifest.json"

# Every SFT prompt and target must remain token-identical to the frozen live
# scorer under the campaign renderer.
"${uv_prime[@]}" python scripts/audit_stage_c4_renderer_alignment.py \
  --config "$config" \
  --dataset "$dataset" \
  --root-cases "$root_cases" \
  --action-cases "$action_cases" \
  --tokenizer "$base_merged/tokenizer.json" \
  --output "$selection_root/renderer-alignment.json"

action_cases_signed="$(signed_field "$action_cases")"
root_cases_signed="$(signed_field "$root_cases")"

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader >"$selection_root/resource-before.csv"

timeout --signal=TERM "$timeout_seconds" \
  "${uv_prime[@]}" sft @ "$repo_root/$config" \
  >"$selection_root/sft-control.log" 2>&1

test -s "$sft_run/metrics.jsonl"
for step in $(seq 2 2 32); do
  test -s "$sft_run/weights/step_${step}/lora_adapters/adapter_model.safetensors"
  test -f "$sft_run/weights/step_${step}/STABLE"
done

candidate_report_args=()
selected_step=""
for step in $(seq 2 2 32); do
  candidate_dir="$selection_root/candidates/step_${step}"
  mkdir -p "$candidate_dir"
  cleanup_candidate_merged
  "${uv_prime[@]}" python scripts/merge_stage_c_warmstart.py \
    --model "$base_merged" \
    --adapter "$sft_run/weights/step_${step}/lora_adapters" \
    --output "$candidate_merged" \
    --manifest "$candidate_dir/merge-manifest.json"

  # The pod has two GPUs. The independent exact action and route scorers run
  # simultaneously, one per GPU, to reduce billed wall time without changing
  # either score or selection rule.
  CUDA_VISIBLE_DEVICES=0 "${uv_prime[@]}" python scripts/run_signed_vllm_scorer.py \
    --output "$candidate_dir/action-scores.json" \
    --verified "$candidate_dir/action-scores.verified.json" \
    --expected-cases-sha256 "$action_cases_signed" \
    --expected-model "$candidate_merged" \
    -- \
    python scripts/score_stage_c_policies_vllm.py \
      --cases "$action_cases" \
      --model "$candidate_merged" \
      --model-name "candidate_step_${step}" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.7 \
      --output "$candidate_dir/action-scores.json" \
    >"$candidate_dir/action-score.log" 2>&1 &
  action_pid=$!

  CUDA_VISIBLE_DEVICES=1 "${uv_prime[@]}" python scripts/run_signed_vllm_scorer.py \
    --output "$candidate_dir/root-scores.json" \
    --verified "$candidate_dir/root-scores.verified.json" \
    --expected-cases-sha256 "$root_cases_signed" \
    --expected-model "$candidate_merged" \
    --expected-analysis stage-c3-root-route-sequence-scores \
    -- \
    python scripts/score_stage_c3_root_routes_vllm.py \
      --cases "$root_cases" \
      --model "$candidate_merged" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.7 \
      --output "$candidate_dir/root-scores.json" \
    >"$candidate_dir/root-score.log" 2>&1 &
  root_pid=$!

  action_status=0
  root_status=0
  wait "$action_pid" || action_status=$?
  wait "$root_pid" || root_status=$?
  if test "$action_status" -ne 0 || test "$root_status" -ne 0; then
    echo "Candidate scorer failure: action=$action_status root=$root_status" >&2
    exit 21
  fi

  "${uv_prime[@]}" python -m redco.analysis.stage_c4_warmstart evaluate \
    --step "$step" \
    --action-scores "$candidate_dir/action-scores.json" \
    --root-scores "$candidate_dir/root-scores.json" \
    --dataset-manifest "$dataset_manifest" \
    --output "$candidate_dir/report.json"
  candidate_report_args+=(--candidate-report "$candidate_dir/report.json")
  candidate_status="$(
    "${uv_prime[@]}" python -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "$candidate_dir/report.json"
  )"
  if test "$candidate_status" = "passed"; then
    selected_step="$step"
    break
  fi
done

"${uv_prime[@]}" python -m redco.analysis.stage_c4_warmstart select \
  "${candidate_report_args[@]}" \
  --output "$selection_root/selection.json"

if test -z "$selected_step"; then
  cleanup_candidate_merged
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader >"$selection_root/resource-after.csv"
  find "$selection_root" -type f ! -name sha256-manifest.txt \
    -print0 | sort -z | xargs -0 sha256sum \
    >"$selection_root/sha256-manifest.txt"
  exit 20
fi

mkdir -p "$selected_root"
mv "$candidate_merged" "$selected_root/merged-model"
cp -a "$sft_run/weights/step_${selected_step}/lora_adapters" \
  "$selected_root/lora-adapters"
cp "$selection_root/candidates/step_${selected_step}/merge-manifest.json" \
  "$selected_root/merge-manifest.json"
"${uv_prime[@]}" python -c \
  'import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])
model=p/"lora-adapters"/"adapter_model.safetensors"
selection=json.load(open(sys.argv[2]))
payload={"schema_version":1,"selected_step":selection["selected_step"],"selection_signed_payload_sha256":selection["signed_payload_sha256"],"adapter_model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),"adapter_model_bytes":model.stat().st_size}
encoded=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
payload["signed_payload_sha256"]=hashlib.sha256(encoded).hexdigest()
pathlib.Path(sys.argv[3]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' \
  "$selected_root" \
  "$selection_root/selection.json" \
  "$selected_root/manifest.json"

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader >"$selection_root/resource-after.csv"
touch "$selection_root/SELECTION_COMPLETE"
find "$selection_root" -type f ! -name sha256-manifest.txt \
  -print0 | sort -z | xargs -0 sha256sum \
  >"$selection_root/sha256-manifest.txt"
selection_status=0
