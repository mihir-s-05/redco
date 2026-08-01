#!/usr/bin/env bash
set -euo pipefail

test "$(uname -s)" = "Linux"
repo_root="/mnt/c/Users/mihir/Documents/redco"
output="${1:?output required}"
evidence_dir="${output%.json}-evidence"
report_staging="${output}.staging.$$"
evidence_staging="${evidence_dir}.staging.$$"
if test -e "$output" || test -e "$evidence_dir" || \
   test -e "$report_staging" || test -e "$evidence_staging"; then
  echo "refusing to overwrite Stage D replay evidence" >&2
  exit 2
fi
production_python="/home/mihir/.venvs/redco-prime-cpu/bin/python"
rlm_python="/home/mihir/.venvs/redco-rlm-replay/bin/python"
scratch="$(mktemp -d /home/mihir/.cache/redco/stage-d-production.XXXXXX)"
scratch="$(readlink -f -- "$scratch")"
case "$scratch" in
  /home/mihir/.cache/redco/stage-d-production.*) ;;
  *) echo "unsafe Stage D scratch path" >&2; exit 1 ;;
esac
cleanup() {
  status="$?"
  if test "$status" -ne 0; then
    failure_dir="${output%.json}-failure-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mkdir "$failure_dir"
    for path in \
      "$scratch/work/production/traces.jsonl" \
      "$scratch/work/production/config.toml" \
      "$scratch/work/precommit.json" \
      "$scratch/work/production.log" \
      "$scratch/work/production.stdout" \
      "$scratch/work/production.stderr" \
      "$scratch/runner.stdout" \
      "$scratch/runner.stderr"; do
      if test -f "$path"; then
        cp -- "$path" "$failure_dir/$(basename "$path")"
      fi
    done
  fi
  rm -f -- "$report_staging"
  rm -rf -- "$evidence_staging"
  rm -rf -- "$scratch"
  return "$status"
}
trap cleanup EXIT

mkdir -p "$scratch/workspace" "$scratch/vf-rlm/bin"
printf '%s\n' "$scratch" > \
  "$scratch/workspace/.redco-stage-d-production-replay-sentinel"
ln -s "$rlm_python" "$scratch/vf-rlm/bin/python"
ln -s /home/mihir/.venvs/redco-rlm-replay/bin/rlm "$scratch/vf-rlm/bin/rlm"

export PYTHONPATH="$repo_root/src:$repo_root:$repo_root/environments/redco_evidence_selection_v2"
export HF_HOME="/home/mihir/.cache/huggingface-redco"
export UV_CACHE_DIR="/home/mihir/.cache/uv-redco"

bwrap \
  --unshare-user --uid 0 --gid 0 --unshare-net \
  --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp \
  --bind "$scratch" "$scratch" \
  --bind "$scratch/workspace" /workspace \
  --bind "$scratch/vf-rlm" /tmp/vf-rlm \
  --chdir "$repo_root" \
  --setenv PYTHONPATH "$PYTHONPATH" \
  --setenv HF_HOME "$HF_HOME" \
  --setenv UV_CACHE_DIR "$UV_CACHE_DIR" \
  --setenv HF_HUB_OFFLINE 1 \
  --setenv TRANSFORMERS_OFFLINE 1 \
  --setenv REDCO_STAGE_D_PRODUCTION_SANDBOX_ROOT "$scratch" \
  "$production_python" \
    "$repo_root/scripts/run_stage_d_production_replay_regression_v1.py" \
    --repo "$repo_root" \
    --work "$scratch/work" \
    --output "$scratch/report.json" \
    --production-python "$production_python" \
    --rlm-python "$rlm_python" \
    >"$scratch/runner.stdout" 2>"$scratch/runner.stderr"

cp -- "$scratch/report.json" "$report_staging"
mkdir "$evidence_staging"
for path in \
  "$scratch/work/fixture.jsonl" \
  "$scratch/work/precommit.json" \
  "$scratch/work/production/traces.jsonl" \
  "$scratch/work/production/config.toml" \
  "$scratch/work/production.log" \
  "$scratch/work/production.stdout" \
  "$scratch/work/production.stderr"; do
  cp -- "$path" "$evidence_staging/$(basename "$path")"
done
(
  cd "$evidence_staging"
  find . -maxdepth 1 -type f ! -name artifact-sha256.txt -print0 |
    sort -z | xargs -0 sha256sum >artifact-sha256.txt
  sha256sum --check artifact-sha256.txt
)
mv -- "$evidence_staging" "$evidence_dir"
mv -- "$report_staging" "$output"
