#!/usr/bin/env bash
set -euo pipefail

if test "${REDCO_STAGE_D_V4_8_TIMEOUT_WRAPPED:-0}" != "1"; then
  timeout_seconds="${REDCO_V4_8_AUTHORIZED_TIMEOUT_SECONDS:?required}"
  test "$timeout_seconds" -ge 1
  test "$timeout_seconds" -le 21600
  export REDCO_STAGE_D_V4_8_TIMEOUT_WRAPPED=1
  exec timeout --signal=TERM --kill-after=120 \
    "$timeout_seconds" bash "$0" "$@"
fi

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
runtime_root="$repo_root/.runtime/stage-d-v4-8"
run_root="runs/stage-d0/scaffold-support-v4-8"
protocol="configs/stage-d/stage-d0-scaffold-support-preregistration-v4-8.json"

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
        raise SystemExit(f"v4.8 bootstrap hash mismatch: {name}")
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

bash "$repo_root/scripts/preflight_stage_d_runtime_paths_v4_8.sh"

set +e
bash "$repo_root/scripts/run_stage_d0_scaffold_support_v4_7.sh"
status=$?
set -e

if test -d "$repo_root/$run_root"; then
  install -m 0644 \
    "$REDCO_RUNTIME_PREFLIGHT_REPORT" \
    "$repo_root/$run_root/runtime-path-preflight.tsv"
  find "$repo_root/$run_root" \
    -type f ! -name artifact-sha256.txt -print0 |
    sort -z |
    xargs -0 sha256sum \
      >"$repo_root/$run_root/artifact-sha256.txt"
fi
exit "$status"
