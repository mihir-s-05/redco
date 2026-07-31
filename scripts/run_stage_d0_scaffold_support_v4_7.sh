#!/usr/bin/env bash
set -euo pipefail

if test "${REDCO_STAGE_D_TIMEOUT_WRAPPED:-0}" != "1"; then
  export REDCO_STAGE_D_TIMEOUT_WRAPPED=1
  exec timeout --signal=TERM --kill-after=120 21600 bash "$0" "$@"
fi

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
run_root="${REDCO_RUN_ROOT:-runs/stage-d0/scaffold-support-v4-7}"
prime_env="${REDCO_UV_ENVIRONMENT:-/workspace/.venv-prime-stage-d-v4-7}"
prime_cache="${REDCO_UV_CACHE_DIR:-/workspace/.uv-cache-prime-stage-d-v4-7}"
protocol="configs/stage-d/stage-d0-scaffold-support-preregistration-v4-7.json"
selected_config="configs/stage-d/stage-d0-scaffold-inference-sft-v4.toml"
dataset="datasets/stage-d/qasper-scaffold-successor-v4.jsonl"
dataset_sha256="2ed4c2afc74b1a979558ada3899b008fcc1b259c5678b3a5ef1f7070aa4fb932"
parent_fixture="datasets/stage-d/evidence-selection-fixture-v1.jsonl"
fixture_dataset="datasets/stage-d/evidence-selection-fixture-v2.jsonl"
fixture_sha256="d809c2d7acd721f38117fc4a4abfc6e3fee19e5a36184fa7df8b592ff87bb65d"
scaffold="configs/stage-d/stage-d0-scaffold-fewshot-v2.txt"
scaffold_sha256="b0db1850aecc8f6f65f530de4ce2ce4ecf6c15a75daf20596564ddf17d7540e2"
migration_report="reports/stage-d0-fixture-v1-to-v2-migration-v4-7.json"
archive="runs/stage-d0/scaffold-support-v4/selected-adapter.tar.gz"
archive_sha256="296a7d6163d1e1a7e8800f3fb1114a9646a8aaf6e9dfbd8437ba006025745263"
archive_manifest="reports/stage-d0-scaffold-step8-adapter-manifest-v1.json"
model_repo="Qwen/Qwen3-4B-Instruct-2507"
model_revision="cdbee75f17c01a7cc42f958dc650907174af0554"
base_model="/workspace/models/qwen3-4b-instruct-2507-cdbee75"
deployed_adapter="/tmp/redco-stage-d0-v4-7-deployed-adapter"
selected_model="/tmp/redco-stage-d0-scaffold-support-v4-sft-merged"
verifiers_worktree="/tmp/redco-verifiers-structural"
verifiers_environment="/tmp/redco-verifiers-structural-env"
inherited_tail="scripts/run_stage_d0_scaffold_support_v4_6.sh"
instrumented_tail="/tmp/redco-stage-d0-v4-7-fixture-power.sh"

cd "$repo_root"
if test "${REDCO_STAGE_D_V4_7_TAIL_TEST_ONLY:-0}" = "1"; then
  awk '
    /^start_inference\(\)/ { emit = 1 }
    emit {
      if ($0 ~ /^run_eval \\$/) {
        calls += 1
        if (calls == 1) {
          print "touch \"$run_root/FIXTURE_REQUESTS_STARTED\""
        } else if (calls == 2) {
          print "touch \"$run_root/POWER_REQUESTS_STARTED\""
        }
      }
      print
    }
  ' "$inherited_tail" >"$instrumented_tail"
  test "$(grep -c 'REQUESTS_STARTED' "$instrumented_tail")" = "2"
  bash -n "$instrumented_tail"
  rm "$instrumented_tail"
  exit 0
fi
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

"$uv_bin" run --frozen python scripts/audit_stage_d_fixture_schema_v4_7.py \
  --fixture "$fixture_dataset" \
  --fixture-sha256 "$fixture_sha256" \
  --parent-fixture "$parent_fixture" \
  --output "$run_root/fixture-schema.json"
"$uv_bin" run --frozen python scripts/audit_stage_d_fixture_migration_v4_7.py \
  --parent-fixture "$parent_fixture" \
  --successor-fixture "$fixture_dataset" \
  --scaffold "$scaffold" \
  --taskset-source \
    environments/redco_evidence_selection_v2/redco_evidence_selection_v2/taskset.py \
  --seeding-source \
    environments/redco_evidence_selection_v2/redco_evidence_selection_v2/seeding.py \
  --master-seed redco-stage-d0-selected-fixture-v4 \
  --output "$run_root/fixture-migration-recomputed.json"
cmp -s "$migration_report" "$run_root/fixture-migration-recomputed.json"

verifiers_source="external/prime-rl/deps/verifiers"
test "$(
  git -C "$verifiers_source" rev-parse HEAD
)" = "b13ba60da63cea91389e7575766b7270d0d11fc5"
test -z "$(git -C "$verifiers_source" status --porcelain)"
test "$(
  sha256sum "$verifiers_source/verifiers/v1/interception/server.py" |
    cut -d ' ' -f 1
)" = "179b117e04910c179605d88cc81260dc567aaf5be385fd7bced15ecd56411ee0"
test "$(
  sha256sum "$verifiers_source/verifiers/v1/trace.py" |
    cut -d ' ' -f 1
)" = "0a756de2539fcb310ea1722ffd7490aa1630c58efb4eefcefff14b355153671b"

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

(
  cd "$verifiers_worktree"
  UV_PROJECT_ENVIRONMENT="$verifiers_environment" \
    "$uv_bin" run --frozen --no-dev --python 3.12 \
    --with-editable \
    "$repo_root/environments/redco_evidence_selection_v2" \
    python "$repo_root/scripts/audit_stage_d_fixture_loader_v4_7.py" \
    --fixture "$repo_root/$fixture_dataset" \
    --fixture-sha256 "$fixture_sha256" \
    --scaffold "$repo_root/$scaffold" \
    --scaffold-sha256 "$scaffold_sha256" \
    --migration-report "$repo_root/$migration_report" \
    --master-seed redco-stage-d0-selected-fixture-v4 \
    --output "$repo_root/$run_root/fixture-loader.json"
)

(
  cd "$verifiers_worktree"
  VLLM_API_KEY=EMPTY \
  UV_PROJECT_ENVIRONMENT="$verifiers_environment" \
    "$uv_bin" run --frozen --no-dev --python 3.12 \
    --with-editable \
    "$repo_root/environments/redco_evidence_selection_v2" \
    python -m redco_evidence_selection_v2.run_feasibility \
    --model "$selected_model" \
    --renderer-model-name "$model_repo" \
    --dataset "$repo_root/$fixture_dataset" \
    --dataset-sha256 "$fixture_sha256" \
    --split audit \
    --output-dir "$repo_root/$run_root/fixture-dry-run-unused" \
    --num-tasks 1 \
    --replicates 1 \
    --prompt-profile fewshot_fixture_v3 \
    --scaffold-prompt "$repo_root/$scaffold" \
    --scaffold-prompt-sha256 "$scaffold_sha256" \
    --master-seed redco-stage-d0-selected-fixture-v4 \
    --temperature 0.7 \
    --top-p 1.0 \
    --max-completion-tokens 768 \
    --max-total-tokens 8192 \
    --dry-run
) >"$run_root/fixture-production-dry-run.json"
"$uv_bin" run --frozen python \
  scripts/audit_stage_d_fixture_dry_run_v4_7.py \
  --plan "$run_root/fixture-production-dry-run.json" \
  --loader-report "$run_root/fixture-loader.json" \
  --expected-model "$selected_model" \
  --expected-fixture "$fixture_dataset" \
  --expected-fixture-sha256 "$fixture_sha256" \
  --expected-scaffold "$scaffold" \
  --expected-scaffold-sha256 "$scaffold_sha256" \
  --output "$run_root/fixture-production-dry-run-audit.json"
touch "$run_root/PREFLIGHT_PASSED"

"$uv_bin" run --frozen python scripts/audit_stage_d_adapter_archive.py \
  --archive "$archive" \
  --output "$run_root/archive-manifest-recomputed.json"
cmp -s "$archive_manifest" "$run_root/archive-manifest-recomputed.json"

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
  rm -rf "$deployed_adapter" "$selected_model" "$instrumented_tail"
}
trap cleanup EXIT

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

awk '
  /^start_inference\(\)/ { emit = 1 }
  emit {
    if ($0 ~ /^run_eval \\$/) {
      calls += 1
      if (calls == 1) {
        print "touch \"$run_root/FIXTURE_REQUESTS_STARTED\""
      } else if (calls == 2) {
        print "touch \"$run_root/POWER_REQUESTS_STARTED\""
      }
    }
    print
  }
' "$inherited_tail" >"$instrumented_tail"
test "$(grep -c 'REQUESTS_STARTED' "$instrumented_tail")" = "2"
source "$instrumented_tail"
