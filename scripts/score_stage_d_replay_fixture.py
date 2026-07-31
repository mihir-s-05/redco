from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(
    0, str(ROOT / "environments" / "redco_evidence_selection_v2")
)

from redco_evidence_selection_v2.scoring import (  # noqa: E402
    parse_evidence,
    score_exact_spans,
)

from redco.analysis.empirical_branch_replay import (  # noqa: E402
    TokenInferenceClient,
    _clean_action_text,
    _token_hash,
)
from redco.integrations.verifiers_trace import (  # noqa: E402
    extract_policy_calls,
)


def load_single_trace(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traces = [
        trace
        for record in records
        for trace in (
            record.get("traces")
            if isinstance(record.get("traces"), list)
            else [record]
        )
    ]
    if len(traces) != 1 or not isinstance(traces[0], dict):
        raise ValueError("fixture scorer requires exactly one trace")
    return traces[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = load_single_trace(args.trace)
    replay = json.loads(args.replay_report.read_text(encoding="utf-8"))
    task = (trace.get("task") or {}).get("data") or {}
    paper = task.get("paper")
    reference = task.get("reference_evidence")
    if not isinstance(paper, str) or not isinstance(reference, list):
        raise TypeError("fixture trace lacks paper/reference evidence")

    calls = {
        call.call_index: call for call in extract_policy_calls(trace)
    }
    roots = [
        call
        for call in calls.values()
        if call.agent_depth == 0 and call.turn_index is not None
    ]
    if not roots:
        raise ValueError("fixture trace has no root calls")
    final_root = max(roots, key=lambda call: call.turn_index or 0)
    original_final_prompt_hash = _token_hash(final_root.prompt_token_ids)
    client = TokenInferenceClient(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=120.0,
    )

    scored_pairs = []
    all_distinct = True
    all_changed = True
    for pair in replay.get("pairs") or []:
        target_index = int(pair["target_call_index"])
        target = calls[target_index]
        alternative_hash = str(pair["candidate_action_sha256"])
        distinct = alternative_hash != _token_hash(target.action_token_ids)
        changed = pair["branch_prompt_sha256"] != original_final_prompt_hash
        token_ids = tuple(
            int(token_id)
            for token_id in pair["downstream_generation"]["token_ids"]
        )
        text = _clean_action_text(client.detokenize(token_ids))
        parsed = parse_evidence(text)
        score = score_exact_spans(paper, parsed.spans, reference)
        scored_pairs.append(
            {
                "target_call_index": target_index,
                "target_node_id": pair["target_node_id"],
                "alternative_index": pair["alternative_index"],
                "alternative_distinct_from_original": distinct,
                "downstream_prompt_changed": changed,
                "terminal_token_sha256": _token_hash(token_ids),
                "terminal_text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "parseable": parsed.parseable,
                "parsed_spans": list(parsed.spans),
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1"],
            }
        )
        all_distinct = all_distinct and distinct
        all_changed = all_changed and changed

    result = {
        "schema_version": 1,
        "scope": (
            "offline deterministic scorer plumbing for shared paired terminal "
            "actions; not independent full-vs-sliced reward equivalence"
        ),
        "source_trace_id": trace.get("id"),
        "all_alternatives_distinct_from_original": all_distinct,
        "all_downstream_prompts_changed": all_changed,
        "pairs": scored_pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not scored_pairs or not all_distinct or not all_changed:
        raise SystemExit("fixture scorer plumbing failed")


if __name__ == "__main__":
    main()
