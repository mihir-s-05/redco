"""Generate the frozen Stage D all-child live runner from the proven v4.6 base."""

from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload

TAIL = r"""start_inference "$selected_config" "selected"
grep -F "enforce_eager=True" "$run_root/inference-selected.log"
if grep -E "profile_cudagraph_memory|Capturing CUDA graphs" \
  "$run_root/inference-selected.log"; then
  echo "eager runtime unexpectedly entered CUDA graph setup" >&2
  exit 25
fi
touch "$run_root/EAGER_RUNTIME_PREFLIGHT_PASSED"

slots="configs/stage-d/stage-d0-all-child-live-slots-v1.json"
materialized="$run_root/materialized-slots"
"$uv_bin" run --frozen python \
  scripts/build_stage_d_all_child_live_slots_v1.py materialize \
  --manifest "$slots" \
  --fixture-dataset "$fixture_dataset" \
  --support-dataset "$dataset" \
  --output-dir "$materialized"

dry_run_eval() {
  local source="$1"
  local source_sha="$2"
  local split="$3"
  local tasks="$4"
  local profile="$5"
  local master_seed="$6"
  local output="$7"
  (
    cd "$verifiers_worktree"
    VLLM_API_KEY=EMPTY \
    UV_PROJECT_ENVIRONMENT="$verifiers_environment" \
      "$uv_bin" run --frozen --no-dev --python 3.12 \
      --with-editable \
      "$repo_root/environments/redco_evidence_selection_v2" \
      python -m redco_evidence_selection_v2.run_feasibility \
      --model "$selected_model" \
      --renderer-model-name "$model_repo" \
      --dataset "$repo_root/$source" \
      --dataset-sha256 "$source_sha" \
      --split "$split" \
      --output-dir "$repo_root/$run_root/dry-unused" \
      --num-tasks "$tasks" \
      --replicates 1 \
      --prompt-profile "$profile" \
      --scaffold-prompt "$repo_root/$scaffold" \
      --scaffold-prompt-sha256 "$scaffold_sha256" \
      --master-seed "$master_seed" \
      --temperature 0.7 \
      --top-p 1.0 \
      --max-completion-tokens 768 \
      --max-total-tokens 8192 \
      --dry-run
  ) >"$output" 2>"$output.stderr"
}

dry_run_eval "$fixture_dataset" "$fixture_sha256" successor_fixture 2 \
  fewshot_fixture_v3 redco-stage-d0-all-child-fixture-v1 \
  "$run_root/fixture-dry-run.json"
"$uv_bin" run --frozen python \
  scripts/audit_stage_d_all_child_live_plan_v1.py dry-run \
  --slots "$slots" --kind fixture \
  --input "$run_root/fixture-dry-run.json" \
  --output "$run_root/fixture-dry-run-audit.json"
dry_run_eval "$dataset" "$dataset_sha256" successor_support 64 \
  fewshot_scaffold_v2 redco-stage-d0-all-child-support-v1 \
  "$run_root/support-dry-run.json"
"$uv_bin" run --frozen python \
  scripts/audit_stage_d_all_child_live_plan_v1.py dry-run \
  --slots "$slots" --kind support \
  --input "$run_root/support-dry-run.json" \
  --output "$run_root/support-dry-run-audit.json"

mkdir "$run_root/work" "$run_root/completed" \
  "$run_root/support-records" "$run_root/progress"

process_slot() {
  local kind="$1"
  local slot="$2"
  local example_id="$3"
  local paper_id="$4"
  local expected_seed="$5"
  local master_seed="$6"
  local replay_seed="$7"
  local source="$8"
  local source_sha="$9"
  local profile="fewshot_scaffold_v2"
  if test "$kind" = "fixture"; then
    profile="fewshot_fixture_v3"
  fi
  local work="$run_root/work/$kind-$slot"
  local recorded="$work/recorded"
  mkdir "$work"
  printf '%s\t%s\t%s\t%s\n' \
    "$kind" "$slot" "$example_id" "$paper_id" >"$work/ADDRESS_STARTED"
  run_eval "$selected_model" "$source" "$source_sha" \
    "$(test "$kind" = fixture && echo successor_fixture || echo successor_support)" \
    "$recorded" 1 1 "$profile" "$master_seed" \
    >"$work/recorded-control.log" 2>&1
  touch "$work/RECORDED_COMPLETION_OBSERVED"
  "$uv_bin" run --frozen python \
    scripts/audit_stage_d_all_child_live_plan_v1.py summary \
    --slots "$slots" --kind "$kind" --index "$((10#$slot))" \
    --input "$recorded/run-summary.json" \
    --output "$work/summary-audit.json"
  "$uv_bin" run --frozen python scripts/split_stage_d_traces.py \
    --input "$recorded/traces.jsonl" --output-dir "$work/traces"
  local trace
  trace="$(find "$work/traces" -maxdepth 1 -type f -name '*.jsonl')"
  test -n "$trace"
  test "$(find "$work/traces" -maxdepth 1 -type f -name '*.jsonl' | wc -l)" = 1
  "$uv_bin" run --frozen python scripts/audit_stage_d_runtime_context_v1.py \
    --trace "$trace" --output "$work/context-audit.json"
  "$uv_bin" run --frozen python scripts/audit_stage_d_all_child_support.py \
    precommit --trace "$trace" --output "$work/precommit.json"
  "$uv_bin" run --frozen python scripts/run_stage_d_all_child_branch_group.py \
    --trace "$trace" --precommit "$work/precommit.json" \
    --output "$work/replay.json" --model "$selected_model" \
    --master-seed "$replay_seed" --temperature 0.7 \
    --candidate-max-tokens 512 --continuation-max-tokens 768
  "$uv_bin" run --frozen python scripts/score_stage_d_all_child_replay.py \
    --trace "$trace" --precommit "$work/precommit.json" \
    --replay-report "$work/replay.json" --master-seed "$replay_seed" \
    --model "$selected_model" --output "$work/scorer.json"
  set +e
  "$uv_bin" run --frozen python scripts/audit_stage_d_all_child_support.py \
    single --trace "$trace" --precommit "$work/precommit.json" \
    --replay "$work/replay.json" --scorer "$work/scorer.json" \
    --master-seed "$replay_seed" --output "$work/target-audit.json"
  local audit_status=$?
  set -e
  test "$audit_status" = 0 -o "$audit_status" = 21
  if test "$kind" = "fixture"; then
    "$uv_bin" run --frozen python \
      scripts/audit_stage_d_fixture_integration_v1.py \
      --target-audit "$work/target-audit.json" \
      --output "$work/fixture-integration.json"
  else
    install -m 0644 "$work/target-audit.json" \
      "$run_root/support-records/$slot.json"
  fi
  mv "$work" "$run_root/completed/$kind-$slot"
}

touch "$run_root/FIXTURE_REQUESTS_STARTED"
while IFS=$'\t' read -r slot example_id paper_id expected_seed \
  master_seed replay_seed source source_sha; do
  process_slot fixture "$slot" "$example_id" "$paper_id" \
    "$expected_seed" "$master_seed" "$replay_seed" "$source" "$source_sha"
done <"$materialized/fixture.tsv"
touch "$run_root/FIXTURE_INTEGRATION_PASSED"

touch "$run_root/SUPPORT_REQUESTS_STARTED"
while IFS=$'\t' read -r slot example_id paper_id expected_seed \
  master_seed replay_seed source source_sha; do
  process_slot support "$slot" "$example_id" "$paper_id" \
    "$expected_seed" "$master_seed" "$replay_seed" "$source" "$source_sha"
  "$uv_bin" run --frozen python \
    scripts/audit_stage_d_all_child_live_plan_v1.py progress \
    --slots "$slots" --records-dir "$run_root/support-records" \
    --output "$run_root/progress/$slot.json"
  decision="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
    "$run_root/progress/$slot.json")"
  if test "$decision" = terminal_fail; then
    touch "$run_root/SUPPORT_EARLY_FAILED"
    break
  fi
done <"$materialized/support.tsv"

latest_progress="$(find "$run_root/progress" -type f -name '*.json' | sort | tail -n 1)"
decision="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
  "$latest_progress")"
stop_inference
rm -rf "$deployed_adapter" "$selected_model"
if test "$decision" = pass; then
  touch "$run_root/SUPPORT_PASSED"
else
  touch "$run_root/SUPPORT_FAILED"
fi
find "$run_root" -type f ! -name artifact-sha256.txt -print0 |
  sort -z | xargs -0 sha256sum >"$run_root/artifact-sha256.txt"
if test "$decision" != pass; then
  exit 22
fi
"""


REPLACEMENTS = {
    'exec timeout --signal=TERM --kill-after=120 21600 bash "$0" "$@"': (
        'exec timeout --signal=TERM --kill-after=120 18000 bash "$0" "$@"'
    ),
    'run_root="${REDCO_RUN_ROOT:-runs/stage-d0/scaffold-support-v4-6}"': (
        'run_root="${REDCO_RUN_ROOT:-runs/stage-d0/all-child-support-v1}"'
    ),
    'protocol="configs/stage-d/stage-d0-scaffold-support-preregistration-v4-6.json"': (
        'protocol="configs/stage-d/stage-d0-all-child-support-preregistration-v1.json"'
    ),
    'selected_config="configs/stage-d/stage-d0-scaffold-inference-sft-v4.toml"': (
        'selected_config="configs/stage-d/stage-d0-scaffold-inference-sft-v5-midpoint-eager.toml"'
    ),
    'dataset="datasets/stage-d/qasper-scaffold-successor-v4.jsonl"': (
        'dataset="datasets/stage-d/qasper-successor-extension-v1.jsonl"'
    ),
    'dataset_sha256="2ed4c2afc74b1a979558ada3899b008fcc1b259c5678b3a5ef1f7070aa4fb932"': (
        'dataset_sha256="8a1b55482d3c5f151741f42ba645b1cad2a0d20e2a445f69432c4e31c0c744b8"'
    ),
    'fixture_dataset="datasets/stage-d/evidence-selection-fixture-v1.jsonl"': (
        'fixture_dataset="datasets/stage-d/evidence-selection-complementary-fixtures-v1.jsonl"'
    ),
    'fixture_sha256="06a22dfea8acc8d7e1cf36f00091610c49cae7cdb7f25ac7d647dc2fcb344783"': (
        'fixture_sha256="d5712f21e74d935707dae66832c916cb2bb3722c1c3c9ef47b551c7f949fd5fa"'
    ),
    'scaffold="configs/stage-d/stage-d0-scaffold-fewshot-v2.txt"': (
        'scaffold="configs/stage-d/stage-d0-scaffold-fewshot-v3.txt"'
    ),
    'scaffold_sha256="b0db1850aecc8f6f65f530de4ce2ce4ecf6c15a75daf20596564ddf17d7540e2"': (
        'scaffold_sha256="1d67c4bf6dc01efada3721b6026128bb56eb49704b6c4789e4057af0b1e1fea7"'
    ),
}


def generate(parent: str) -> str:
    output = parent
    for old, new in REPLACEMENTS.items():
        if output.count(old) != 1:
            raise ValueError(f"expected exactly one parent match: {old}")
        output = output.replace(old, new)
    marker = 'start_inference "$selected_config" "selected"\n'
    if output.count(marker) != 1:
        raise ValueError("parent request-tail marker is not unique")
    output = output[: output.index(marker)] + TAIL
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent.read_text(encoding="utf-8")
    generated = generate(parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(generated.encode("utf-8"))
    report = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-live-runner-generation-v1",
            "parent": args.parent.as_posix(),
            "parent_sha256": hashlib.sha256(parent.encode("utf-8")).hexdigest(),
            "generated": args.output.as_posix(),
            "generated_sha256": hashlib.sha256(generated.encode("utf-8")).hexdigest(),
            "replacement_count": len(REPLACEMENTS),
            "request_tail_replaced": True,
            "unified_diff": list(
                difflib.unified_diff(parent.splitlines(), generated.splitlines(), lineterm="")
            ),
            "passes": True,
        }
    )
    atomic_write_json(args.report, report)


if __name__ == "__main__":
    main()
