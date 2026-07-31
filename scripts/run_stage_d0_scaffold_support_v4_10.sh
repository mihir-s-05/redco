#!/usr/bin/env bash
set -euo pipefail

if test "${REDCO_STAGE_D_V4_10_TIMEOUT_WRAPPED:-0}" != "1"; then
  timeout_seconds="${REDCO_V4_10_AUTHORIZED_TIMEOUT_SECONDS:?required}"
  test "$timeout_seconds" -ge 1
  test "$timeout_seconds" -le 17100
  export REDCO_STAGE_D_V4_10_TIMEOUT_WRAPPED=1
  exec timeout --signal=TERM --kill-after=120 \
    "$timeout_seconds" bash "$0" "$@"
fi

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
runtime_root="$repo_root/.runtime/stage-d-v4-8"
run_root="runs/stage-d0/scaffold-support-v4-10"
protocol="configs/stage-d/stage-d0-scaffold-support-preregistration-v4-10.json"
generated_root_rel="runs/stage-d0/scaffold-support-v4-10-cpu-preflight"
generated_root="$repo_root/$generated_root_rel"
generated_eager_tail_rel="$generated_root_rel/generated-eager-tail.sh"
generated_eager_tail="$repo_root/$generated_eager_tail_rel"
generated_eager_audit="$runtime_root/generated-eager-tail-v4-10-audit.json"
generated_inner_rel="$generated_root_rel/generated-inner.sh"
generated_inner="$repo_root/$generated_inner_rel"
generated_inner_audit="$runtime_root/generated-inner-v4-10-audit.json"

test "$repo_root" = "/workspace/redco"
test ! -e "$repo_root/$run_root"

cd "$repo_root"
python3 - "$protocol" <<'PY'
import hashlib
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for name, expected in protocol["source_sha256"].items():
    actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"v4.10 bootstrap hash mismatch: {name}")
PY

sudo -n install -d -o ubuntu -g ubuntu -m 0755 \
  /workspace/models \
  /workspace/.cache \
  /workspace/.cache/huggingface
sudo -n install -o ubuntu -g ubuntu -m 0644 \
  /dev/null /workspace/evidence_context.txt

export REDCO_REPO_ROOT="$repo_root"
export REDCO_RUNTIME_ROOT="$runtime_root"
export REDCO_RUNTIME_PREFLIGHT_REPORT="$runtime_root/runtime-path-preflight.tsv"
export REDCO_MINIMUM_FREE_KIB=47185920
export REDCO_RUN_ROOT="$run_root"
export REDCO_UV_ENVIRONMENT="$runtime_root/prime-env"
export REDCO_UV_CACHE_DIR="$runtime_root/uv-cache"
export REDCO_V4_10_EAGER_TAIL="$generated_eager_tail"

mkdir -p "$REDCO_UV_ENVIRONMENT" "$REDCO_UV_CACHE_DIR" "$generated_root"
if test -x "$HOME/.local/uv-latest/uv"; then
  uv_bin="$HOME/.local/uv-latest/uv"
elif test -x "$HOME/.local/bin/uv"; then
  uv_bin="$HOME/.local/bin/uv"
else
  uv_bin="$(command -v uv)"
fi

UV_PROJECT_ENVIRONMENT="$REDCO_UV_ENVIRONMENT" \
UV_CACHE_DIR="$REDCO_UV_CACHE_DIR" \
  "$uv_bin" run --frozen --no-dev python \
  scripts/generate_stage_d_v4_10_eager_tail.py \
  --parent scripts/run_stage_d0_scaffold_support_v4_6.sh \
  --output "$generated_eager_tail_rel" \
  --report "$generated_eager_audit"
bash -n "$generated_eager_tail"

UV_PROJECT_ENVIRONMENT="$REDCO_UV_ENVIRONMENT" \
UV_CACHE_DIR="$REDCO_UV_CACHE_DIR" \
  "$uv_bin" run --frozen --no-dev python \
  scripts/generate_stage_d_v4_10_inner_runner.py \
  --parent scripts/run_stage_d0_scaffold_support_v4_7.sh \
  --output "$generated_inner_rel" \
  --report "$generated_inner_audit"
bash -n "$generated_inner"

python3 - \
  "$protocol" \
  "$generated_eager_tail" \
  "$generated_eager_audit" \
  "$generated_inner" \
  "$generated_inner_audit" <<'PY'
import hashlib
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = protocol["runtime"]
for label, path, expected_hash, audit_path, expected_signature in (
    (
        "eager tail",
        pathlib.Path(sys.argv[2]),
        runtime["generated_eager_tail_sha256"],
        pathlib.Path(sys.argv[3]),
        runtime["generated_eager_tail_audit_signature"],
    ),
    (
        "inner runner",
        pathlib.Path(sys.argv[4]),
        runtime["generated_inner_sha256"],
        pathlib.Path(sys.argv[5]),
        runtime["generated_inner_audit_signature"],
    ),
):
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if actual_hash != expected_hash:
        raise SystemExit(f"v4.10 generated {label} hash mismatch")
    if audit.get("signed_payload_sha256") != expected_signature:
        raise SystemExit(f"v4.10 generated {label} audit signature mismatch")
    if not audit.get("passes"):
        raise SystemExit(f"v4.10 generated {label} audit failed")
PY

bash "$repo_root/scripts/preflight_stage_d_runtime_paths_v4_8.sh"

set +e
bash "$generated_inner"
status=$?
set -e

if test -d "$repo_root/$run_root"; then
  install -m 0644 \
    "$REDCO_RUNTIME_PREFLIGHT_REPORT" \
    "$repo_root/$run_root/runtime-path-preflight.tsv"
  install -m 0644 \
    "$generated_eager_audit" \
    "$repo_root/$run_root/generated-eager-tail-v4-10-audit.json"
  install -m 0644 \
    "$generated_inner_audit" \
    "$repo_root/$run_root/generated-inner-v4-10-audit.json"
  find "$repo_root/$run_root" \
    -type f ! -name artifact-sha256.txt -print0 |
    sort -z |
    xargs -0 sha256sum \
      >"$repo_root/$run_root/artifact-sha256.txt"
fi
exit "$status"
