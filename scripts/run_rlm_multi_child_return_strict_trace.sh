#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
cd "$repo_root"

export REDCO_REPO_ROOT="$repo_root"
export REDCO_RUN_ROOT="runs/stage-b/rlm-multi-child-return-strict-trace"
export REDCO_TASK_PROFILE="multi_child"
export REDCO_MAX_TOTAL_TOKENS="8192"
export REDCO_RLM_TOOL_PATCH_MODE="root_initial"
export REDCO_FORWARD_REQUIRED_TOOL_CHOICE_ENV="1"
export REDCO_INFERENCE_CONFIG="configs/stage-b/rlm-trace-audit-inference-strict.toml"
export REDCO_PRIME_STRICT_ENV_GUARD="1"
export RLM_FORCE_TOOL_CHOICE_REQUIRED="1"

bash scripts/run_rlm_structural_trace_audit.sh

if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
elif test -x "$HOME/.local/bin/uv"; then
  uv_bin="$HOME/.local/bin/uv"
else
  uv_bin="$(command -v uv)"
fi

"$uv_bin" run --frozen python -m redco.analysis.recorded_raf \
  --input "$REDCO_RUN_ROOT/live/traces.jsonl" \
  --output "$REDCO_RUN_ROOT/recorded-raf-projection.json" \
  --alternatives-per-target 3 \
  --require-broader-trace
