#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
run_root="${REDCO_RUN_ROOT:-runs/stage-d0/scaffold-support-v3}"
prime_env="${REDCO_UV_ENVIRONMENT:-/workspace/.venv-prime-stage-d}"
prime_cache="${REDCO_UV_CACHE_DIR:-/workspace/.uv-cache-prime-stage-d}"
verifiers_worktree="/tmp/redco-verifiers-structural"
verifiers_environment="/tmp/redco-verifiers-structural-env"
base_config="configs/stage-d/stage-d0-scaffold-inference-base-v3.toml"
sft_config="configs/stage-d/stage-d0-scaffold-inference-sft-v3.toml"
sft_train_config="configs/stage-d/stage-d0-scaffold-sft-v2.toml"
dataset="datasets/stage-d/qasper-scaffold-successor-v2.jsonl"
dataset_sha256="fbe3edb17aeaafb0e4326d5aaf83d898cfacb1490c5ee24b14468bffd0eaed80"
fixture_dataset="datasets/stage-d/evidence-selection-fixture-v1.jsonl"
fixture_sha256="06a22dfea8acc8d7e1cf36f00091610c49cae7cdb7f25ac7d647dc2fcb344783"
scaffold="configs/stage-d/stage-d0-scaffold-fewshot-v2.txt"
scaffold_sha256="b0db1850aecc8f6f65f530de4ce2ce4ecf6c15a75daf20596564ddf17d7540e2"
model_repo="Qwen/Qwen3-4B-Instruct-2507"
model_revision="cdbee75f17c01a7cc42f958dc650907174af0554"
base_model="/workspace/models/qwen3-4b-instruct-2507-cdbee75"
sft_dir="$repo_root/runs/stage-d0/scaffold-sft-v2"
sft_selected="$sft_dir/weights/step_8/lora_adapters"
sft_reloaded="$repo_root/$run_root/sft-reloaded-adapter"
sft_merged="$repo_root/$run_root/sft-merged"

cd "$repo_root"
test ! -e "$run_root"
mkdir -p "$run_root"

if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
elif test -x "$HOME/.local/bin/uv"; then
  uv_bin="$HOME/.local/bin/uv"
else
  uv_bin="$(command -v uv)"
fi

export UV_PROJECT_ENVIRONMENT="$prime_env"
export UV_CACHE_DIR="$prime_cache"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export XDG_CONFIG_HOME="$repo_root/.runtime-config"
export PYTHONPATH="$repo_root/src:$repo_root/environments/redco_evidence_selection_v2"
mkdir -p "$XDG_CONFIG_HOME" /workspace/models /workspace/.cache/huggingface

test "$(sha256sum "$dataset" | cut -d ' ' -f 1)" = "$dataset_sha256"
test "$(sha256sum "$fixture_dataset" | cut -d ' ' -f 1)" = "$fixture_sha256"
test "$(sha256sum "$scaffold" | cut -d ' ' -f 1)" = "$scaffold_sha256"
test -f "$base_config"
test -f "$sft_config"
test -f "$sft_train_config"

snapshot_path="$(
  HF_HOME=/workspace/.cache/huggingface \
    "$uv_bin" run --frozen --project external/prime-rl \
    python -c \
    "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$model_repo', revision='$model_revision'))"
)"
resolved_snapshot="$(readlink -f "$snapshot_path")"
case "$resolved_snapshot" in
  *"/snapshots/$model_revision") ;;
  *)
    echo "unexpected model snapshot: $resolved_snapshot" >&2
    exit 1
    ;;
esac
ln -sfn "$resolved_snapshot" "$base_model"

REDCO_REPO_ROOT="$repo_root" \
REDCO_PREPARE_ONLY=1 \
REDCO_RLM_TOOL_PATCH_MODE=root_initial \
REDCO_INFERENCE_CONFIG="$base_config" \
REDCO_PRIME_STRICT_ENV_GUARD=1 \
  bash scripts/run_rlm_structural_trace_audit.sh

inference_pid=""
meter_pid=""
current_inference_log=""

stop_inference() {
  if test -n "$inference_pid" && kill -0 "$inference_pid" 2>/dev/null; then
    kill "$inference_pid"
    wait "$inference_pid" || true
  fi
  if test -n "$meter_pid" && kill -0 "$meter_pid" 2>/dev/null; then
    kill "$meter_pid"
    wait "$meter_pid" || true
  fi
  inference_pid=""
  meter_pid=""
}

cleanup() {
  stop_inference
}
trap cleanup EXIT

start_inference() {
  local config="$1"
  local label="$2"
  current_inference_log="$run_root/inference-$label.log"
  CUDA_VISIBLE_DEVICES=0 \
    "$uv_bin" run --frozen --project external/prime-rl \
    inference @ "$config" >"$current_inference_log" 2>&1 &
  inference_pid=$!
  (
    while kill -0 "$inference_pid" 2>/dev/null; do
      nvidia-smi \
        --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader
      sleep 5
    done
  ) >"$run_root/gpu-resource-$label.csv" 2>&1 &
  meter_pid=$!
  for _ in $(seq 1 180); do
    if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
      grep -Fx "REDCO_STRICT_TOOL_CALLING_ENV=1" "$current_inference_log"
      return
    fi
    if ! kill -0 "$inference_pid" 2>/dev/null; then
      tail -n 120 "$current_inference_log"
      exit 1
    fi
    sleep 5
  done
  tail -n 120 "$current_inference_log"
  exit 1
}

run_eval() {
  local model="$1"
  local source="$2"
  local source_sha="$3"
  local split="$4"
  local output="$5"
  local tasks="$6"
  local replicates="$7"
  local profile="$8"
  local master_seed="$9"
  (
    cd "$verifiers_worktree"
    VLLM_API_KEY=EMPTY \
    UV_PROJECT_ENVIRONMENT="$verifiers_environment" \
      "$uv_bin" run --frozen --no-dev --python 3.12 \
      --with-editable \
      "$repo_root/environments/redco_evidence_selection_v2" \
      python -m redco_evidence_selection_v2.run_feasibility \
      --model "$model" \
      --renderer-model-name "$model_repo" \
      --dataset "$repo_root/$source" \
      --dataset-sha256 "$source_sha" \
      --split "$split" \
      --output-dir "$repo_root/$output" \
      --num-tasks "$tasks" \
      --replicates "$replicates" \
      --prompt-profile "$profile" \
      --scaffold-prompt "$repo_root/$scaffold" \
      --scaffold-prompt-sha256 "$scaffold_sha256" \
      --master-seed "$master_seed" \
      --temperature 0.7 \
      --top-p 1.0 \
      --max-completion-tokens 768 \
      --max-total-tokens 8192
  )
}

start_inference "$base_config" "base-support"
run_eval \
  "$base_model" "$dataset" "$dataset_sha256" fewshot_support \
  "$run_root/fewshot-support" 8 8 fewshot_scaffold_v2 \
  redco-stage-d0-fewshot-support-v3 \
  >"$run_root/fewshot-support-control.log" 2>&1

set +e
"$uv_bin" run --frozen python scripts/audit_stage_d_scaffold_support.py \
  --traces "$run_root/fewshot-support/traces.jsonl" \
  --output "$run_root/fewshot-support-audit.json"
support_status=$?
set -e

selected_model="$base_model"
selected_config="$base_config"
selected_kind="shared-fewshot-base"
if test "$support_status" = "20"; then
  stop_inference
  CUDA_VISIBLE_DEVICES=0 \
  timeout --signal=TERM 3600 \
    "$uv_bin" run --frozen --project external/prime-rl \
    --extra flash-attn \
    sft @ "$sft_train_config" \
    >"$run_root/sft-control.log" 2>&1

  for step in 2 4 6 8; do
    test -s "$sft_dir/weights/step_${step}/lora_adapters/adapter_model.safetensors"
    test -f "$sft_dir/weights/step_${step}/STABLE"
  done
  tar -C "$sft_selected" -czf "$run_root/selected-adapter.tar.gz" .
  mkdir "$sft_reloaded"
  tar -xzf "$run_root/selected-adapter.tar.gz" -C "$sft_reloaded"
  test "$(
    sha256sum "$sft_selected/adapter_model.safetensors" | cut -d ' ' -f 1
  )" = "$(
    sha256sum "$sft_reloaded/adapter_model.safetensors" | cut -d ' ' -f 1
  )"

  CUDA_VISIBLE_DEVICES=0 \
    "$uv_bin" run --frozen --project external/prime-rl \
    python scripts/score_stage_c_policies_vllm.py \
    --cases configs/stage-c2/policy-audit-cases.json \
    --model "$base_model" \
    --adapter "original=$sft_selected" \
    --adapter "reloaded=$sft_reloaded" \
    --output "$run_root/sft-reload-scores.json" \
    >"$run_root/sft-reload-score.log" 2>&1
  "$uv_bin" run --frozen python -c \
    'import json,sys; p=json.load(open(sys.argv[1])); m={x["name"]:x["temperatures"] for x in p["models"]}; assert m["original"] == m["reloaded"]' \
    "$run_root/sft-reload-scores.json"

  "$uv_bin" run --frozen --project external/prime-rl \
    python scripts/merge_stage_c_warmstart.py \
    --model "$base_model" \
    --adapter "$sft_reloaded" \
    --output "$sft_merged" \
    --manifest "$run_root/sft-merge-manifest.json"

  selected_model="$sft_merged"
  selected_config="$sft_config"
  selected_kind="shared-fewshot-plus-fixed-step8-sft"
elif test "$support_status" != "0"; then
  exit "$support_status"
fi

start_inference "$selected_config" "selected"

run_eval \
  "$selected_model" "$fixture_dataset" "$fixture_sha256" audit \
  "$run_root/selected-fixture" 1 1 fewshot_fixture_v3 \
  redco-stage-d0-selected-fixture-v3 \
  >"$run_root/selected-fixture-control.log" 2>&1
fixture_trace="$run_root/selected-fixture/traces.jsonl"
fixture_trace_sha="$(sha256sum "$fixture_trace" | cut -d ' ' -f 1)"
"$uv_bin" run --frozen python -m redco.analysis.empirical_branch_replay \
  --input "$fixture_trace" \
  --output "$run_root/selected-fixture-replay.json" \
  --expected-source-sha256 "$fixture_trace_sha" \
  --alternatives-per-target 3 \
  --maximum-targets 1 \
  --master-seed redco-stage-d0-selected-fixture-replay-v3 \
  --temperature 0.7 \
  --candidate-max-tokens 512 \
  --continuation-max-tokens 768 \
  --minimum-distinct-candidate-fraction 0.3333333333333333
"$uv_bin" run --frozen python scripts/score_stage_d_replay_fixture.py \
  --trace "$fixture_trace" \
  --replay-report "$run_root/selected-fixture-replay.json" \
  --model "$selected_model" \
  --output "$run_root/selected-fixture-scores.json"
"$uv_bin" run --frozen python scripts/audit_stage_d_target_support.py single \
  --trace "$fixture_trace" \
  --replay "$run_root/selected-fixture-replay.json" \
  --scorer "$run_root/selected-fixture-scores.json" \
  --output "$run_root/selected-fixture-eligibility.json"

run_eval \
  "$selected_model" "$dataset" "$dataset_sha256" power_audit \
  "$run_root/power-audit" 8 8 fewshot_scaffold_v2 \
  redco-stage-d0-power-audit-v3 \
  >"$run_root/power-audit-control.log" 2>&1
"$uv_bin" run --frozen python scripts/split_stage_d_traces.py \
  --input "$run_root/power-audit/traces.jsonl" \
  --output-dir "$run_root/power-traces"
mkdir "$run_root/power-replays" "$run_root/power-scores" "$run_root/power-records"

for trace in "$run_root"/power-traces/*.jsonl; do
  name="$(basename "$trace" .jsonl)"
  has_target="$(
    "$uv_bin" run --frozen python -c \
      'import sys; from pathlib import Path; from redco.integrations.verifiers_trace import load_trace_records,extract_policy_calls; t=load_trace_records(Path(sys.argv[1]))[0]; print(int(any(c.agent_depth == 1 for c in extract_policy_calls(t))))' \
      "$trace"
  )"
  if test "$has_target" = "0"; then
    set +e
    "$uv_bin" run --frozen python scripts/audit_stage_d_target_support.py single \
      --trace "$trace" \
      --replay "$run_root/nonexistent-replay.json" \
      --scorer "$run_root/nonexistent-scorer.json" \
      --output "$run_root/power-records/$name.json"
    single_status=$?
    set -e
    test "$single_status" = "21"
    continue
  fi
  trace_sha="$(sha256sum "$trace" | cut -d ' ' -f 1)"
  "$uv_bin" run --frozen python -m redco.analysis.empirical_branch_replay \
    --input "$trace" \
    --output "$run_root/power-replays/$name.json" \
    --expected-source-sha256 "$trace_sha" \
    --alternatives-per-target 3 \
    --maximum-targets 1 \
    --master-seed redco-stage-d0-power-replay-v3 \
    --temperature 0.7 \
    --candidate-max-tokens 512 \
    --continuation-max-tokens 768 \
    --minimum-distinct-candidate-fraction 0.3333333333333333
  "$uv_bin" run --frozen python scripts/score_stage_d_replay_fixture.py \
    --trace "$trace" \
    --replay-report "$run_root/power-replays/$name.json" \
    --model "$selected_model" \
    --output "$run_root/power-scores/$name.json"
  set +e
  "$uv_bin" run --frozen python scripts/audit_stage_d_target_support.py single \
    --trace "$trace" \
    --replay "$run_root/power-replays/$name.json" \
    --scorer "$run_root/power-scores/$name.json" \
    --output "$run_root/power-records/$name.json"
  single_status=$?
  set -e
  test "$single_status" = "0" -o "$single_status" = "21"
done

set +e
"$uv_bin" run --frozen python scripts/audit_stage_d_target_support.py aggregate \
  --records-dir "$run_root/power-records" \
  --output "$run_root/power-audit-aggregate.json"
power_status=$?
set -e

"$uv_bin" run --frozen python -c \
  'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"schema_version":1,"selected_initialization":sys.argv[2],"selected_model":sys.argv[3]},indent=2)+"\n")' \
  "$run_root/selection.json" "$selected_kind" "$selected_model"
find "$run_root" -type f ! -name artifact-sha256.txt -print0 |
  sort -z |
  xargs -0 sha256sum >"$run_root/artifact-sha256.txt"

if test "$power_status" = "22"; then
  touch "$run_root/POWER_FAILED"
  exit 22
fi
test "$power_status" = "0"
touch "$run_root/POWER_PASSED"
