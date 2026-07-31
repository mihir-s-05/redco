#!/usr/bin/env bash
set -euo pipefail

if test "${REDCO_STAGE_D_ALL_CHILD_REPAIR_TIMEOUT_WRAPPED:-0}" != "1"; then
  timeout_seconds="${REDCO_ALL_CHILD_AUTHORIZED_TIMEOUT_SECONDS:?required}"
  test "$timeout_seconds" -ge 1
  test "$timeout_seconds" -le 18000
  export REDCO_STAGE_D_ALL_CHILD_REPAIR_TIMEOUT_WRAPPED=1
  exec timeout --signal=TERM --kill-after=120 \
    "$timeout_seconds" bash "$0" "$@"
fi

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
runtime_root="$repo_root/.runtime/stage-d-v4-8"
run_root="runs/stage-d0/all-child-support-v1"
base_protocol="configs/stage-d/stage-d0-all-child-support-preregistration-v1.json"
amendment="configs/stage-d/stage-d0-all-child-support-repair-amendment-v1-1.json"
merged_protocol="$runtime_root/all-child-support-repair-protocol-v1-1.json"
generated_root_rel="runs/stage-d0/all-child-support-v1-1-cpu-preflight"
generated_root="$repo_root/$generated_root_rel"
generated_runner_rel="$generated_root_rel/generated-runner.sh"
generated_runner="$repo_root/$generated_runner_rel"
generated_audit="$runtime_root/generated-all-child-runner-audit-v1-1.json"

test "$repo_root" = "/workspace/redco"
test ! -e "$repo_root/$run_root"
cd "$repo_root"
mkdir -p "$runtime_root" "$generated_root"

python3 - "$base_protocol" "$amendment" "$merged_protocol" <<'PY'
import hashlib
import json
import pathlib
import sys

base_path, amendment_path, output_path = map(pathlib.Path, sys.argv[1:])
amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
if hashlib.sha256(base_path.read_bytes()).hexdigest() != amendment["base_protocol_sha256"]:
    raise SystemExit("bounded-repair base protocol hash mismatch")
base = json.loads(base_path.read_text(encoding="utf-8"))
merged_sources = dict(base["source_sha256"])
merged_sources.update(amendment["source_sha256_overrides"])
merged_sources.update(amendment["repair_source_sha256"])
for name, expected in merged_sources.items():
    actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"bounded-repair source hash mismatch: {name}")
base["source_sha256"] = merged_sources
encoded = json.dumps(base, sort_keys=True, separators=(",", ":")) + "\n"
if hashlib.sha256(encoded.encode()).hexdigest() != amendment["merged_protocol_sha256"]:
    raise SystemExit("bounded-repair merged protocol hash mismatch")
output_path.write_text(encoded, encoding="utf-8")
PY

sudo -n install -d -o ubuntu -g ubuntu -m 0755 \
  /workspace/models /workspace/.cache /workspace/.cache/huggingface
sudo -n install -o ubuntu -g ubuntu -m 0644 \
  /dev/null /workspace/evidence_context.txt

export REDCO_REPO_ROOT="$repo_root"
export REDCO_RUNTIME_ROOT="$runtime_root"
export REDCO_RUNTIME_PREFLIGHT_REPORT="$runtime_root/runtime-path-preflight.tsv"
export REDCO_MINIMUM_FREE_KIB=47185920
export REDCO_RUN_ROOT="$run_root"
export REDCO_UV_ENVIRONMENT="$runtime_root/prime-env"
export REDCO_UV_CACHE_DIR="$runtime_root/uv-cache"

mkdir -p "$REDCO_UV_ENVIRONMENT" "$REDCO_UV_CACHE_DIR"
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
  scripts/generate_stage_d_all_child_live_runner_v1_1.py \
  --parent scripts/run_stage_d0_scaffold_support_v4_6.sh \
  --output "$generated_runner_rel" --report "$generated_audit"
bash -n "$generated_runner"

python3 - "$amendment" "$generated_runner" "$generated_audit" <<'PY'
import hashlib
import json
import pathlib
import sys

amendment = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
runner = pathlib.Path(sys.argv[2])
audit = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
if hashlib.sha256(runner.read_bytes()).hexdigest() != amendment["generated_runner_sha256"]:
    raise SystemExit("bounded-repair generated runner hash mismatch")
if audit.get("signed_payload_sha256") != amendment["generated_runner_audit_signature"]:
    raise SystemExit("bounded-repair generator audit signature mismatch")
if not audit.get("passes"):
    raise SystemExit("bounded-repair generator audit failed")
PY

bash "$repo_root/scripts/preflight_stage_d_runtime_paths_v4_8.sh"

set +e
bash "$generated_runner"
status=$?
set -e

if test -d "$repo_root/$run_root"; then
  install -m 0644 "$REDCO_RUNTIME_PREFLIGHT_REPORT" \
    "$repo_root/$run_root/runtime-path-preflight.tsv"
  install -m 0644 "$generated_audit" \
    "$repo_root/$run_root/generated-all-child-runner-audit-v1-1.json"
  install -m 0644 "$merged_protocol" \
    "$repo_root/$run_root/merged-repair-protocol-v1-1.json"
  find "$repo_root/$run_root" \
    -type f ! -name artifact-sha256.txt -print0 |
    sort -z | xargs -0 sha256sum \
      >"$repo_root/$run_root/artifact-sha256.txt"
fi
exit "$status"
