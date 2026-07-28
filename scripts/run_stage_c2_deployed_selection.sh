#!/usr/bin/env bash
set -euo pipefail

repo_root="${REDCO_REPO_ROOT:-/workspace/redco}"
prime_env="${REDCO_UV_ENVIRONMENT:-/workspace/.venv-prime-stage-c2}"
prime_cache="${REDCO_UV_CACHE_DIR:-/workspace/.uv-cache-prime-stage-c2}"
audit_dir="$repo_root/runs/stage-c2/warmstart-audit-v2"
sft_dir="$repo_root/runs/stage-c2/warmstart-sft-v2"
candidate_root="$repo_root/runs/stage-c2/warmstart-merged-candidates-v2"
selection_path="$audit_dir/deployed-selection.json"

cd "$repo_root"
test -s "$audit_dir/merged-policy-scores.json"
test -s "$audit_dir/gate-report.json"
test ! -e "$selection_path"
test ! -e "$candidate_root"
mkdir -p "$candidate_root"

export UV_PROJECT_ENVIRONMENT="$prime_env"
export UV_CACHE_DIR="$prime_cache"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="$repo_root/src:$repo_root/scripts"

scores=(--score "21=$audit_dir/merged-policy-scores.json")
selection_args=(
  --start-step 21
  --minimum-needle-mass-t2 0.08
  --maximum-needle-mass-t2 0.12
  --maximum-needle-greedy-rate 0
  --branch-count 11
  --groups-per-step 8
  --minimum-expected-informative-groups 4.75
)

uv run --frozen --project external/prime-rl \
  python -m redco.analysis.stage_c_deployed_selection \
  "${scores[@]}" "${selection_args[@]}" --output "$selection_path"

for step in $(seq 22 32); do
  candidate_dir="$candidate_root/step_${step}"
  candidate_score="$audit_dir/merged-step-${step}-policy-scores.json"
  candidate_manifest="$audit_dir/merged-step-${step}-manifest.json"
  candidate_log="$audit_dir/merged-step-${step}-score.log"
  test ! -e "$candidate_dir"
  test ! -e "$candidate_score"
  test ! -e "$candidate_manifest"
  test ! -e "$candidate_log"

  uv run --frozen --project external/prime-rl \
    python scripts/merge_stage_c_warmstart.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --adapter "$sft_dir/weights/step_${step}/lora_adapters" \
    --output "$candidate_dir" \
    --manifest "$candidate_manifest"

  uv run --frozen --project external/prime-rl \
    python scripts/score_stage_c_policies_vllm.py \
    --cases configs/stage-c2/policy-audit-cases.json \
    --model "$candidate_dir" \
    --model-name "merged_step_${step}" \
    --output "$candidate_score" \
    >"$candidate_log" 2>&1

  scores+=(--score "$step=$candidate_score")
  uv run --frozen --project external/prime-rl \
    python -m redco.analysis.stage_c_deployed_selection \
    "${scores[@]}" "${selection_args[@]}" --output "$selection_path.tmp"
  mv "$selection_path.tmp" "$selection_path"

  selected_step="$(
    uv run --frozen --project external/prime-rl python -c \
      'import json,sys; value=json.load(open(sys.argv[1])); print("" if value["selected"] is None else value["selected"]["step"])' \
      "$selection_path"
  )"
  if test -n "$selected_step"; then
    test "$selected_step" = "$step"
    printf '%s\n' "$selected_step" >"$audit_dir/DEPLOYED_WARMSTART_SELECTED"
    exit 0
  fi
done

touch "$audit_dir/DEPLOYED_WARMSTART_FAILED"
exit 20
