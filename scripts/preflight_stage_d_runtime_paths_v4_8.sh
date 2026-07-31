#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
runtime_root="${REDCO_RUNTIME_ROOT:-$repo_root/.runtime/stage-d-v4-8}"
report="${REDCO_RUNTIME_PREFLIGHT_REPORT:-$runtime_root/runtime-path-preflight.tsv}"
minimum_free_kib="${REDCO_MINIMUM_FREE_KIB:-47185920}"

expected_paths=(
  "/workspace/redco"
  "/workspace/redco/runs/stage-d0"
  "/workspace/redco/.runtime/stage-d-v4-8"
  "/workspace/redco/.runtime/stage-d-v4-8/prime-env"
  "/workspace/redco/.runtime/stage-d-v4-8/uv-cache"
  "/workspace/redco/.runtime-config"
  "/workspace/models"
  "/workspace/.cache"
  "/workspace/.cache/huggingface"
  "/home/ubuntu/.local/uv-latest"
  "/tmp"
)
context_path="/workspace/evidence_context.txt"

test "$repo_root" = "/workspace/redco"
test "$runtime_root" = "/workspace/redco/.runtime/stage-d-v4-8"
test "$report" = "$runtime_root/runtime-path-preflight.tsv"
test "$(id -un)" = "ubuntu"

mkdir -p \
  "$repo_root/runs/stage-d0" \
  "$runtime_root/prime-env" \
  "$runtime_root/uv-cache" \
  "$repo_root/.runtime-config"

tmp_report="$(mktemp /tmp/redco-stage-d-v4-8-runtime-paths.XXXXXX)"
trap 'rm -f "$tmp_report"' EXIT
printf 'path\twritable\tprobe_created_and_removed\n' >"$tmp_report"

for path in "${expected_paths[@]}"; do
  test -d "$path"
  test -w "$path"
  probe="$(mktemp -d "$path/.redco-stage-d-v4-8-write-probe.XXXXXX")"
  test -d "$probe"
  rmdir "$probe"
  test ! -e "$probe"
  printf '%s\ttrue\ttrue\n' "$path" >>"$tmp_report"
done

test -f "$context_path"
test -w "$context_path"
context_probe="redco-stage-d-v4-8-context-write-read-truncate-probe"
printf '%s' "$context_probe" >"$context_path"
test "$(cat "$context_path")" = "$context_probe"
: >"$context_path"
test ! -s "$context_path"
printf '%s\ttrue\twrite-read-truncate\n' "$context_path" >>"$tmp_report"

workspace_free_kib="$(df -Pk /workspace/redco | awk 'NR == 2 {print $4}')"
tmp_free_kib="$(df -Pk /tmp | awk 'NR == 2 {print $4}')"
test "$workspace_free_kib" -ge "$minimum_free_kib"
test "$tmp_free_kib" -ge "$minimum_free_kib"
printf 'minimum_free_kib\t%s\n' "$minimum_free_kib" >>"$tmp_report"
printf 'workspace_free_kib\t%s\n' "$workspace_free_kib" >>"$tmp_report"
printf 'tmp_free_kib\t%s\n' "$tmp_free_kib" >>"$tmp_report"
printf 'uid\t%s\n' "$(id -u)" >>"$tmp_report"
printf 'user\t%s\n' "$(id -un)" >>"$tmp_report"

install -m 0644 "$tmp_report" "$report"
test "$(wc -l <"$report")" = "18"
