#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
cd "$repo_root"
run_root="runs/stage-b/rlm-structural-trace-audit"
rlm_worktree="/tmp/redco-rlm-structural"
rlm_tool_root="/tmp/vf-rlm"
verifiers_worktree="/tmp/redco-verifiers-structural"
verifiers_environment="/tmp/redco-verifiers-structural-env"
mkdir -p "$run_root"

if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
elif test -x "$HOME/.local/bin/uv"; then
  uv_bin="$HOME/.local/bin/uv"
else
  uv_bin="$(command -v uv)"
fi

test "$(
  sha256sum patches/rlm-structural-trace-headers.patch | cut -d ' ' -f 1
)" = "589d412de4aff70ecfd52e35e474ef42c9033e5a221c7db9929ee838b24bcfb9"
test "$(
  sha256sum patches/verifiers-rlm-structural-trace.patch | cut -d ' ' -f 1
)" = "95db874f84fdd1487399d6ee77b11f1726e7ff27c14d0626a1a7e6f2c664b577"

rm -rf "$rlm_worktree" "$rlm_tool_root" "$verifiers_worktree" \
  "$verifiers_environment"
git clone --quiet https://github.com/PrimeIntellect-ai/rlm.git "$rlm_worktree"
git -C "$rlm_worktree" checkout --quiet \
  56218f33796ecbe465445bc43948886354fde196
git -C "$rlm_worktree" apply \
  "$repo_root/patches/rlm-structural-trace-headers.patch"
mkdir -p "$rlm_tool_root/bin" "$rlm_tool_root/tools"
UV_TOOL_BIN_DIR="$rlm_tool_root/bin" \
  UV_TOOL_DIR="$rlm_tool_root/tools" \
  "$uv_bin" tool install --python 3.12 --editable "$rlm_worktree"
test -x "$rlm_tool_root/bin/rlm"

git clone --quiet \
  "$repo_root/external/prime-rl/deps/verifiers" \
  "$verifiers_worktree"
git -C "$verifiers_worktree" apply \
  "$repo_root/patches/verifiers-rlm-structural-trace.patch"
UV_PROJECT_ENVIRONMENT="$verifiers_environment" \
  "$uv_bin" run --frozen --no-dev --python 3.12 \
  --project "$verifiers_worktree" \
  python -c \
  "from verifiers.v1.interception.server import _rlm_structure; assert _rlm_structure({'X-RLM-Depth': '0'}) is not None"

if test "${REDCO_PREPARE_ONLY:-0}" = "1"; then
  exit 0
fi

inference_log="$run_root/inference.log"
control_log="$run_root/control.log"
traces="$run_root/live/traces.jsonl"

CUDA_VISIBLE_DEVICES=0 \
  "$uv_bin" run --frozen --project external/prime-rl \
  inference @ configs/stage-b/rlm-trace-audit-inference.toml \
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
if test "$ready" != "1"; then
  tail -n 100 "$inference_log"
  exit 1
fi

(
  cd "$verifiers_worktree"
  VLLM_API_KEY=EMPTY \
    UV_PROJECT_ENVIRONMENT="$verifiers_environment" \
    "$uv_bin" run --frozen --no-dev --python 3.12 \
    --with-editable "$repo_root/environments/redco_rlm_trace_v1" \
    python -m redco_rlm_trace_v1.run_audit \
    --output-dir "$repo_root/$run_root/live"
) >"$control_log" 2>&1

test -s "$traces"
"$uv_bin" run --frozen python -m redco.analysis.verifiers_trace_audit \
  --input "$traces" \
  --output "$run_root/trace-audit-report.json"
"$uv_bin" run --frozen python -m redco.analysis.verifiers_provenance \
  --input "$traces" \
  --output "$run_root/provenance-report.json" \
  --require-ready
