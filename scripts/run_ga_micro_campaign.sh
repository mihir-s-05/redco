#!/usr/bin/env bash
set -euo pipefail

phase="${1:?usage: run_ga_micro_campaign.sh pilot|confirm}"
cd /workspace/redco
export PATH="/workspace/redco/external/prime-rl/.venv/bin:$PATH"
config_dir="runs/stage-a/ga-micro/configs"
control_dir="runs/stage-a/ga-micro/control"
mkdir -p "$control_dir"

case "$phase" in
  pilot)
    names=(
      pilot-stock-s2101-a
      pilot-stock-s2101-b
      pilot-stock-s2102-a
      pilot-stock-s2102-b
    )
    ;;
  confirm)
    names=(
      confirm-stock-s3101
      confirm-redco-s3101
      confirm-stock-s3102
      confirm-redco-s3102
      confirm-stock-s3103
      confirm-redco-s3103
      confirm-stock-s3104
      confirm-redco-s3104
    )
    ;;
  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac

for name in "${names[@]}"; do
  echo "GA_MICRO_START $name $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
  external/prime-rl/.venv/bin/rl @ "$config_dir/$name.toml" \
    >"$control_dir/$name.launcher.log" 2>&1
  test -s "runs/stage-a/ga-micro/$name/metrics.jsonl"
  test -s "runs/stage-a/ga-micro/$name/run_default/metrics.jsonl"
  if grep -Eiq "Traceback|CUDA out of memory|process failed" \
    "$control_dir/$name.launcher.log"; then
    tail -n 100 "$control_dir/$name.launcher.log"
    exit 1
  fi
  echo "GA_MICRO_DONE $name $(date --utc +%Y-%m-%dT%H:%M:%SZ)"
done
