#!/usr/bin/env bash
set -euo pipefail

if test "${REDCO_STAGE_D_TIMEOUT_WRAPPED:-0}" != "1"; then
  export REDCO_STAGE_D_TIMEOUT_WRAPPED=1
  exec timeout --signal=TERM --kill-after=120 21600 bash "$0" "$@"
fi

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
run_root="${REDCO_RUN_ROOT:-runs/stage-d0/scaffold-support-v4-6}"
prime_env="${REDCO_UV_ENVIRONMENT:-/workspace/.venv-prime-stage-d-v4-6}"
prime_cache="${REDCO_UV_CACHE_DIR:-/workspace/.uv-cache-prime-stage-d-v4-6}"
protocol="configs/stage-d/stage-d0-scaffold-support-preregistration-v4-6.json"
selected_config="configs/stage-d/stage-d0-scaffold-inference-sft-v4.toml"
dataset="datasets/stage-d/qasper-scaffold-successor-v4.jsonl"
dataset_sha256="2ed4c2afc74b1a979558ada3899b008fcc1b259c5678b3a5ef1f7070aa4fb932"
fixture_dataset="datasets/stage-d/evidence-selection-fixture-v1.jsonl"
fixture_sha256="06a22dfea8acc8d7e1cf36f00091610c49cae7cdb7f25ac7d647dc2fcb344783"
scaffold="configs/stage-d/stage-d0-scaffold-fewshot-v2.txt"
scaffold_sha256="b0db1850aecc8f6f65f530de4ce2ce4ecf6c15a75daf20596564ddf17d7540e2"
archive="runs/stage-d0/scaffold-support-v4/selected-adapter.tar.gz"
archive_sha256="296a7d6163d1e1a7e8800f3fb1114a9646a8aaf6e9dfbd8437ba006025745263"
archive_manifest="reports/stage-d0-scaffold-step8-adapter-manifest-v1.json"
action_cases="configs/stage-c4/selection-action-cases.json"
root_cases="configs/stage-c4/selection-root-cases.json"
canonical_scorer="scripts/score_stage_d_retained_adapter_canonical_v4_6.py"
model_repo="Qwen/Qwen3-4B-Instruct-2507"
model_revision="cdbee75f17c01a7cc42f958dc650907174af0554"
base_model="/workspace/models/qwen3-4b-instruct-2507-cdbee75"
active_adapter="/tmp/redco-stage-d0-v4-6-active-adapter"
deployed_adapter="/tmp/redco-stage-d0-v4-6-deployed-adapter"
selected_model="/tmp/redco-stage-d0-scaffold-support-v4-sft-merged"
verifiers_worktree="/tmp/redco-verifiers-structural"
verifiers_environment="/tmp/redco-verifiers-structural-env"

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
export PYTHONPATH="$repo_root/src:$repo_root/scripts:$repo_root/environments/redco_evidence_selection_v2"
mkdir -p "$XDG_CONFIG_HOME" /workspace/models /workspace/.cache/huggingface

"$uv_bin" run --frozen python - "$protocol" <<'PY'
import hashlib
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for name, expected in protocol["source_sha256"].items():
    actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"frozen hash mismatch: {name}: {actual} != {expected}")
PY

test "$(sha256sum "$archive" | cut -d ' ' -f 1)" = "$archive_sha256"
test "$(sha256sum "$dataset" | cut -d ' ' -f 1)" = "$dataset_sha256"
test "$(sha256sum "$fixture_dataset" | cut -d ' ' -f 1)" = "$fixture_sha256"
test "$(sha256sum "$scaffold" | cut -d ' ' -f 1)" = "$scaffold_sha256"

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
REDCO_INFERENCE_CONFIG="$selected_config" \
REDCO_PRIME_STRICT_ENV_GUARD=1 \
  bash scripts/run_rlm_structural_trace_audit.sh
test "$(
  sha256sum external/prime-rl/src/prime_rl/inference/server.py |
    cut -d ' ' -f 1
)" = "b5030e12c3658152430360e9733754d135f59434bd0a9f8a1daa279bd3abfe2c"

"$uv_bin" run --frozen python scripts/audit_stage_d_adapter_archive.py \
  --archive "$archive" \
  --output "$run_root/archive-manifest-recomputed.json"
cmp -s "$archive_manifest" "$run_root/archive-manifest-recomputed.json"

inference_pid=""
meter_pid=""

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
  if test -L "$active_adapter"; then
    rm "$active_adapter"
  fi
  rm -rf "$deployed_adapter" "$selected_model"
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES=0 \
  "$uv_bin" run --frozen python \
  scripts/run_stage_d_retention_canonical_v4_6.py \
  --archive "$archive" \
  --frozen-archive-manifest "$archive_manifest" \
  --base-model "$base_model" \
  --action-cases "$action_cases" \
  --root-cases "$root_cases" \
  --scorer "$canonical_scorer" \
  --uv-binary "$uv_bin" \
  --prime-project external/prime-rl \
  --output-dir "$run_root/retention" \
  --stable-adapter-path "$active_adapter" \
  --output "$run_root/retention-ledger.json"

mkdir "$deployed_adapter"
tar -xzf "$archive" -C "$deployed_adapter"
"$uv_bin" run --frozen python scripts/audit_stage_d_adapter_directory.py \
  --directory "$deployed_adapter" \
  --output "$run_root/deployed-adapter-manifest.json"
"$uv_bin" run --frozen python - \
  "$archive_manifest" "$run_root/deployed-adapter-manifest.json" <<'PY'
import json
import pathlib
import sys

archive = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
deployed = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
for field in ("members", "adapter_config", "safetensors", "passes"):
    if archive[field] != deployed[field]:
        raise SystemExit(f"deployed adapter differs in {field}")
PY

"$uv_bin" run --frozen --project external/prime-rl \
  python scripts/merge_stage_c_warmstart.py \
  --model "$base_model" \
  --adapter "$deployed_adapter" \
  --output "$selected_model" \
  --manifest "$run_root/selected-merge-manifest.json"

CUDA_VISIBLE_DEVICES=0 \
  "$uv_bin" run --frozen --project external/prime-rl \
  python scripts/score_stage_c_policies_vllm.py \
  --cases "$action_cases" \
  --model "$selected_model" \
  --model-name selected \
  --output "$run_root/runtime-action-scores.json" \
  >"$run_root/runtime-action-score.log" 2>&1
CUDA_VISIBLE_DEVICES=0 \
  "$uv_bin" run --frozen --project external/prime-rl \
  python scripts/score_stage_c3_root_routes_vllm.py \
  --cases "$root_cases" \
  --model "$selected_model" \
  --output "$run_root/runtime-root-scores.json" \
  >"$run_root/runtime-root-score.log" 2>&1
"$uv_bin" run --frozen python scripts/audit_stage_d_vllm_health_v4_6.py \
  --canonical-action "$run_root/retention/canonical-action-1.json" \
  --runtime-action "$run_root/runtime-action-scores.json" \
  --canonical-root "$run_root/retention/canonical-root-1.json" \
  --runtime-root "$run_root/runtime-root-scores.json" \
  --expected-runtime-model "$selected_model" \
  --output "$run_root/runtime-health.json"

start_inference() {
  local config="$1"
  local label="$2"
  CUDA_VISIBLE_DEVICES=0 \
    "$uv_bin" run --frozen --project external/prime-rl \
    inference @ "$config" >"$run_root/inference-$label.log" 2>&1 &
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
      grep -Fx "REDCO_STRICT_TOOL_CALLING_ENV=1" \
        "$run_root/inference-$label.log"
      return
    fi
    if ! kill -0 "$inference_pid" 2>/dev/null; then
      tail -n 120 "$run_root/inference-$label.log"
      exit 1
    fi
    sleep 5
  done
  tail -n 120 "$run_root/inference-$label.log"
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

start_inference "$selected_config" "selected"

run_eval \
  "$selected_model" "$fixture_dataset" "$fixture_sha256" audit \
  "$run_root/selected-fixture" 1 1 fewshot_fixture_v3 \
  redco-stage-d0-selected-fixture-v4 \
  >"$run_root/selected-fixture-control.log" 2>&1
fixture_trace="$run_root/selected-fixture/traces.jsonl"
"$uv_bin" run --frozen python scripts/run_stage_d_branch_group.py \
  --trace "$fixture_trace" \
  --output "$run_root/selected-fixture-replay.json" \
  --model "$selected_model" \
  --master-seed redco-stage-d0-selected-fixture-replay-v4 \
  --temperature 0.7 \
  --candidate-max-tokens 512 \
  --continuation-max-tokens 768
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
  "$run_root/power-audit" 64 1 fewshot_scaffold_v2 \
  redco-stage-d0-power-audit-v4 \
  >"$run_root/power-audit-control.log" 2>&1
"$uv_bin" run --frozen python scripts/split_stage_d_traces.py \
  --input "$run_root/power-audit/traces.jsonl" \
  --output-dir "$run_root/power-traces"
mkdir "$run_root/power-replays" "$run_root/power-scores" \
  "$run_root/power-target-records"

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
      --output "$run_root/power-target-records/$name.json"
    single_status=$?
    set -e
    test "$single_status" = "21"
    continue
  fi

  "$uv_bin" run --frozen python scripts/run_stage_d_branch_group.py \
    --trace "$trace" \
    --output "$run_root/power-replays/$name.json" \
    --model "$selected_model" \
    --master-seed redco-stage-d0-power-replay-v4 \
    --temperature 0.7 \
    --candidate-max-tokens 512 \
    --continuation-max-tokens 768
  replay_status="$(
    "$uv_bin" run --frozen python -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status","ok"))' \
      "$run_root/power-replays/$name.json"
  )"
  scorer="$run_root/nonexistent-scorer.json"
  if test "$replay_status" = "ok"; then
    scorer="$run_root/power-scores/$name.json"
    "$uv_bin" run --frozen python scripts/score_stage_d_replay_fixture.py \
      --trace "$trace" \
      --replay-report "$run_root/power-replays/$name.json" \
      --model "$selected_model" \
      --output "$scorer"
  fi
  set +e
  "$uv_bin" run --frozen python scripts/audit_stage_d_target_support.py single \
    --trace "$trace" \
    --replay "$run_root/power-replays/$name.json" \
    --scorer "$scorer" \
    --output "$run_root/power-target-records/$name.json"
  single_status=$?
  set -e
  test "$single_status" = "0" -o "$single_status" = "21"
done

selected_initialization_sha="$(
  printf '%s\n' \
    "d2490951aa35bd1bfa52f0b4dbeb6be5ffb98aab74f794736e3f71a58e238412|$model_repo@$model_revision|$scaffold_sha256" |
    sha256sum | cut -d ' ' -f 1
)"
"$uv_bin" run --frozen python scripts/materialize_stage_d_power_records.py \
  --summary "$run_root/power-audit/run-summary.json" \
  --traces-dir "$run_root/power-traces" \
  --target-records-dir "$run_root/power-target-records" \
  --dataset "$dataset" \
  --selected-initialization-sha256 "$selected_initialization_sha" \
  --output-dir "$run_root/power-records"

set +e
"$uv_bin" run --frozen python scripts/audit_stage_d_target_support.py aggregate \
  --records-dir "$run_root/power-records" \
  --output "$run_root/power-audit-aggregate.json"
power_status=$?
set -e

"$uv_bin" run --frozen python -c \
  'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.write_text(json.dumps({"schema_version":1,"selected_initialization":"shared-fewshot-plus-fixed-step8-sft","selected_model":sys.argv[2],"selected_initialization_sha256":sys.argv[3]},indent=2)+"\n")' \
  "$run_root/selection.json" "$selected_model" \
  "$selected_initialization_sha"

stop_inference
rm -rf "$deployed_adapter" "$selected_model"
if test "$power_status" = "22"; then
  touch "$run_root/POWER_FAILED"
else
  test "$power_status" = "0"
  touch "$run_root/POWER_PASSED"
fi
find "$run_root" -type f ! -name artifact-sha256.txt -print0 |
  sort -z |
  xargs -0 sha256sum >"$run_root/artifact-sha256.txt"

if test "$power_status" = "22"; then
  exit 22
fi
test "$power_status" = "0"
