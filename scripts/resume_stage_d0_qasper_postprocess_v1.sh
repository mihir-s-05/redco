#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
run_root="${REDCO_RUN_ROOT:-runs/stage-d0/qasper-feasibility-v1}"
inference_config="configs/stage-d/stage-d0-qasper-feasibility-inference-v1.toml"
model_path="/workspace/models/qwen3-4b-instruct-2507-cdbee75"
natural_trace="$run_root/natural/traces.jsonl"
fixture_trace="$run_root/fixture/traces.jsonl"
natural_sha256="317d031b1528dd1191e5415102d592d1070fc262c5df4eafe684652e99e87c6b"
fixture_sha256="c826975f1bbe459318fe12e51f32886d69b61e95ccdeac18c74622d87edbdf1f"

cd "$repo_root"
if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
elif test -x "$HOME/.local/bin/uv"; then
  uv_bin="$HOME/.local/bin/uv"
else
  uv_bin="$(command -v uv)"
fi

test "$(sha256sum "$natural_trace" | cut -d ' ' -f 1)" = "$natural_sha256"
test "$(sha256sum "$fixture_trace" | cut -d ' ' -f 1)" = "$fixture_sha256"
test ! -e "$run_root/fixture-replay-report.json"
test ! -e "$run_root/fixture-scorer-plumbing.json"
test "$(readlink -f "$model_path")" = "$(
  readlink -f /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
)"

inference_log="$run_root/postprocess-inference.log"
postprocess_log="$run_root/postprocess-control.log"
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
test "$ready" = "1"
grep -Fx "REDCO_STRICT_TOOL_CALLING_ENV=1" "$inference_log"

"$uv_bin" run --frozen python -m redco.analysis.empirical_branch_replay \
  --input "$fixture_trace" \
  --output "$run_root/fixture-replay-report.json" \
  --expected-source-sha256 "$fixture_sha256" \
  --alternatives-per-target 1 \
  --master-seed redco-stage-d0-qasper-fixture-replay-v1 \
  --temperature 0.7 \
  --candidate-max-tokens 256 \
  --continuation-max-tokens 512 \
  >"$postprocess_log" 2>&1

PYTHONPATH="$repo_root/environments/redco_evidence_selection_v2" \
  "$uv_bin" run --frozen python scripts/score_stage_d_replay_fixture.py \
  --trace "$fixture_trace" \
  --replay-report "$run_root/fixture-replay-report.json" \
  --model "$model_path" \
  --output "$run_root/fixture-scorer-plumbing.json" \
  >>"$postprocess_log" 2>&1

find "$run_root" -type f ! -name artifact-sha256.txt -print0 |
  sort -z |
  xargs -0 sha256sum >"$run_root/artifact-sha256.txt"
test -s "$run_root/fixture-replay-report.json"
test -s "$run_root/fixture-scorer-plumbing.json"
