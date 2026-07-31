#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
cd "$repo_root"
run_root="${REDCO_RUN_ROOT:-runs/stage-b/rlm-structural-trace-audit}"
task_profile="${REDCO_TASK_PROFILE:-single}"
max_total_tokens="${REDCO_MAX_TOTAL_TOKENS:-4096}"
tool_patch_mode="${REDCO_RLM_TOOL_PATCH_MODE:-all_turns}"
forward_required_tool_choice_env="${REDCO_FORWARD_REQUIRED_TOOL_CHOICE_ENV:-0}"
inference_config="${REDCO_INFERENCE_CONFIG:-configs/stage-b/rlm-trace-audit-inference.toml}"
prime_strict_env_guard="${REDCO_PRIME_STRICT_ENV_GUARD:-0}"
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
  sha256sum patches/rlm-mcp-client-symbol-compat.patch | cut -d ' ' -f 1
)" = "0706fe4aa96c8c9e648ca55312c433797d923958b422ca27386085b4cfed87bd"
case "$tool_patch_mode" in
  all_turns)
    tool_patch="patches/rlm-required-tool-choice.patch"
    tool_patch_sha256="93a7456cf48d9150c83add3e3139eb8d38a936e0b87fe8adf4b766183430a673"
    ;;
  root_initial)
    tool_patch="patches/rlm-root-initial-required-tool-choice.patch"
    tool_patch_sha256="9730e59d2fc161e2b0dc69bccafca5fc07fe1b94ded06b8c9647e8d3d5a41c75"
    ;;
  *)
    echo "unsupported REDCO_RLM_TOOL_PATCH_MODE: $tool_patch_mode" >&2
    exit 2
    ;;
esac
test "$(
  sha256sum "$tool_patch" | cut -d ' ' -f 1
)" = "$tool_patch_sha256"
test "$(
  sha256sum patches/verifiers-rlm-structural-trace.patch | cut -d ' ' -f 1
)" = "95db874f84fdd1487399d6ee77b11f1726e7ff27c14d0626a1a7e6f2c664b577"
test -f "$inference_config"
if test "$prime_strict_env_guard" = "1"; then
  test "$(
    sha256sum patches/prime-rl-strict-tool-env-guard.patch | cut -d ' ' -f 1
  )" = "1c52102bf79741d8a1791733397de26d7319b907531317c22d5ec1e6cd29c001"
  if git -C external/prime-rl apply --check \
    "$repo_root/patches/prime-rl-strict-tool-env-guard.patch"
  then
    git -C external/prime-rl apply \
      "$repo_root/patches/prime-rl-strict-tool-env-guard.patch"
  else
    git -C external/prime-rl apply --reverse --check \
      "$repo_root/patches/prime-rl-strict-tool-env-guard.patch"
  fi
fi

rm -rf "$rlm_worktree" "$rlm_tool_root" "$verifiers_worktree" \
  "$verifiers_environment"
git clone --quiet https://github.com/PrimeIntellect-ai/rlm.git "$rlm_worktree"
git -C "$rlm_worktree" checkout --quiet \
  56218f33796ecbe465445bc43948886354fde196
git -C "$rlm_worktree" apply \
  "$repo_root/patches/rlm-structural-trace-headers.patch"
git -C "$rlm_worktree" apply \
  "$repo_root/patches/rlm-mcp-client-symbol-compat.patch"
git -C "$rlm_worktree" apply \
  "$repo_root/$tool_patch"
mkdir -p "$rlm_tool_root/bin" "$rlm_tool_root/tools"
UV_TOOL_BIN_DIR="$rlm_tool_root/bin" \
  UV_TOOL_DIR="$rlm_tool_root/tools" \
  "$uv_bin" tool install --python 3.12 --editable "$rlm_worktree"
test -x "$rlm_tool_root/bin/rlm"
"$rlm_tool_root/tools/rlm/bin/python" -c \
  "import rlm.mcp; assert hasattr(rlm.mcp, 'streamable_http_client')"

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
if test "$ready" != "1"; then
  tail -n 100 "$inference_log"
  exit 1
fi
if test "$prime_strict_env_guard" = "1"; then
  grep -Fx "REDCO_STRICT_TOOL_CALLING_ENV=1" "$inference_log"
fi

(
  cd "$verifiers_worktree"
  control_args=(
    --output-dir "$repo_root/$run_root/live"
    --task-profile "$task_profile"
    --max-total-tokens "$max_total_tokens"
  )
  if test "$forward_required_tool_choice_env" = "1"; then
    control_args+=(--forward-required-tool-choice-env)
  fi
  VLLM_API_KEY=EMPTY \
    UV_PROJECT_ENVIRONMENT="$verifiers_environment" \
    "$uv_bin" run --frozen --no-dev --python 3.12 \
    --with-editable "$repo_root/environments/redco_rlm_trace_v1" \
    python -m redco_rlm_trace_v1.run_audit \
    "${control_args[@]}"
) >"$control_log" 2>&1

test -s "$traces"
"$uv_bin" run --frozen python -m redco.analysis.verifiers_trace_audit \
  --input "$traces" \
  --output "$run_root/trace-audit-report.json"
"$uv_bin" run --frozen python -m redco.analysis.verifiers_provenance \
  --input "$traces" \
  --output "$run_root/provenance-report.json" \
  --require-ready
