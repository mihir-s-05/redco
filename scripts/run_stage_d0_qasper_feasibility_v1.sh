#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
run_root="${REDCO_RUN_ROOT:-runs/stage-d0/qasper-feasibility-v1}"
inference_config="configs/stage-d/stage-d0-qasper-feasibility-inference-v1.toml"
dataset="datasets/stage-d/qasper-deterministic-v2.jsonl"
dataset_sha256="de84fda40c43fa7f977e063130f3f60fbcf05f625f947d941f3b6c0a80cbd347"
model_repo="Qwen/Qwen3-4B-Instruct-2507"
model_revision="cdbee75f17c01a7cc42f958dc650907174af0554"
model_path="/workspace/models/qwen3-4b-instruct-2507-cdbee75"
verifiers_worktree="/tmp/redco-verifiers-structural"
verifiers_environment="/tmp/redco-verifiers-structural-env"
cd "$repo_root"
mkdir -p "$run_root"

if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
elif test -x "$HOME/.local/bin/uv"; then
  uv_bin="$HOME/.local/bin/uv"
else
  uv_bin="$(command -v uv)"
fi

test "$(sha256sum "$dataset" | cut -d ' ' -f 1)" = "$dataset_sha256"
test -f "$inference_config"
test -f environments/redco_evidence_selection_v2/pyproject.toml
test -f scripts/analyze_stage_d0_qasper_feasibility.py
test -f scripts/score_stage_d_replay_fixture.py

mkdir -p /workspace/models /workspace/.cache/huggingface
snapshot_path="$(
  HF_HOME=/workspace/.cache/huggingface \
    "$uv_bin" run --frozen --project external/prime-rl \
    python -c \
    "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='$model_repo', revision='$model_revision'))"
)"
test -d "$snapshot_path"
resolved_snapshot="$(readlink -f "$snapshot_path")"
case "$resolved_snapshot" in
  *"/snapshots/$model_revision") ;;
  *)
    echo "unexpected model snapshot path: $resolved_snapshot" >&2
    exit 1
    ;;
esac
ln -sfn "$resolved_snapshot" "$model_path"
test "$(readlink -f "$model_path")" = "$resolved_snapshot"

REDCO_REPO_ROOT="$repo_root" \
REDCO_PREPARE_ONLY=1 \
REDCO_RLM_TOOL_PATCH_MODE=root_initial \
REDCO_INFERENCE_CONFIG="$inference_config" \
REDCO_PRIME_STRICT_ENV_GUARD=1 \
  bash scripts/run_rlm_structural_trace_audit.sh

inference_log="$run_root/inference.log"
resource_log="$run_root/gpu-resource.csv"
control_log="$run_root/control.log"

CUDA_VISIBLE_DEVICES=0 \
  "$uv_bin" run --frozen --project external/prime-rl \
  inference @ "$inference_config" \
  >"$inference_log" 2>&1 &
inference_pid=$!

(
  while kill -0 "$inference_pid" 2>/dev/null; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader
    sleep 5
  done
) >"$resource_log" 2>&1 &
meter_pid=$!

cleanup() {
  if kill -0 "$inference_pid" 2>/dev/null; then
    kill "$inference_pid"
    wait "$inference_pid" || true
  fi
  if kill -0 "$meter_pid" 2>/dev/null; then
    kill "$meter_pid"
    wait "$meter_pid" || true
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

{
  cd "$verifiers_worktree"
  VLLM_API_KEY=EMPTY \
  UV_PROJECT_ENVIRONMENT="$verifiers_environment" \
    "$uv_bin" run --frozen --no-dev --python 3.12 \
    --with-editable \
    "$repo_root/environments/redco_evidence_selection_v2" \
    python -m redco_evidence_selection_v2.run_feasibility \
    --model "$model_path" \
    --dataset "$repo_root/$dataset" \
    --output-dir "$repo_root/$run_root/natural" \
    --num-tasks 8 \
    --replicates 4 \
    --prompt-profile natural \
    --master-seed redco-stage-d0-qasper-natural-v1 \
    --temperature 0.7 \
    --top-p 1.0 \
    --max-completion-tokens 768 \
    --max-total-tokens 8192

  VLLM_API_KEY=EMPTY \
  UV_PROJECT_ENVIRONMENT="$verifiers_environment" \
    "$uv_bin" run --frozen --no-dev --python 3.12 \
    --with-editable \
    "$repo_root/environments/redco_evidence_selection_v2" \
    python -m redco_evidence_selection_v2.run_feasibility \
    --model "$model_path" \
    --dataset "$repo_root/$dataset" \
    --output-dir "$repo_root/$run_root/fixture" \
    --num-tasks 1 \
    --replicates 1 \
    --prompt-profile forced_trace_fixture \
    --master-seed redco-stage-d0-qasper-fixture-v1 \
    --temperature 0.7 \
    --top-p 1.0 \
    --max-completion-tokens 768 \
    --max-total-tokens 8192
} >"$control_log" 2>&1

fixture_trace="$run_root/fixture/traces.jsonl"
fixture_sha256="$(sha256sum "$fixture_trace" | cut -d ' ' -f 1)"
"$uv_bin" run --frozen python -m redco.analysis.empirical_branch_replay \
  --input "$fixture_trace" \
  --output "$run_root/fixture-replay-report.json" \
  --expected-source-sha256 "$fixture_sha256" \
  --alternatives-per-target 1 \
  --master-seed redco-stage-d0-qasper-fixture-replay-v1 \
  --temperature 0.7 \
  --candidate-max-tokens 256 \
  --continuation-max-tokens 512 \
  >>"$control_log" 2>&1

PYTHONPATH="$repo_root/environments/redco_evidence_selection_v2" \
  "$uv_bin" run --frozen python scripts/score_stage_d_replay_fixture.py \
  --trace "$fixture_trace" \
  --replay-report "$run_root/fixture-replay-report.json" \
  --model "$model_path" \
  --output "$run_root/fixture-scorer-plumbing.json" \
  >>"$control_log" 2>&1

find "$run_root" -type f ! -name artifact-sha256.txt -print0 |
  sort -z |
  xargs -0 sha256sum >"$run_root/artifact-sha256.txt"
test -s "$run_root/natural/traces.jsonl"
test -s "$run_root/fixture/traces.jsonl"
test -s "$run_root/fixture-replay-report.json"
test -s "$run_root/fixture-scorer-plumbing.json"
