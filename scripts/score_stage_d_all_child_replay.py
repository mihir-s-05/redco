"""Score a Stage D all-child replay with a complete provenance chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "environments" / "redco_evidence_selection_v2"))

from redco_evidence_selection_v2.scoring import (  # noqa: E402
    parse_evidence,
    score_exact_spans,
)

from redco.analysis.empirical_branch_replay import (  # noqa: E402
    TokenInferenceClient,
    _clean_action_text,
    _token_hash,
)
from redco.analysis.stage_d_all_child_support import (  # noqa: E402
    verify_canonical_precommit,
    verify_replay_chain,
)
from redco.integrations.signed_subprocess import (  # noqa: E402
    atomic_write_json,
    sign_payload,
)
from redco.integrations.verifiers_trace import (  # noqa: E402
    extract_policy_calls,
)


def load_single_trace(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    traces = [
        trace
        for record in records
        for trace in (record.get("traces") if isinstance(record.get("traces"), list) else [record])
    ]
    if len(traces) != 1 or not isinstance(traces[0], dict):
        raise ValueError("all-child scorer requires exactly one trace")
    return traces[0]


def _chain(committed: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_trace_sha256": committed["source_trace_sha256"],
        "precommit_signed_payload_sha256": committed["signed_payload_sha256"],
        "candidate_set_sha256": committed["candidate_set_sha256"],
        "replay_signed_payload_sha256": replay["signed_payload_sha256"],
        "decision_unit_weights": replay["decision_unit_weights"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--precommit", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--master-seed", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = load_single_trace(args.trace)
    committed = json.loads(args.precommit.read_text(encoding="utf-8"))
    replay = json.loads(args.replay_report.read_text(encoding="utf-8"))
    verify_canonical_precommit(args.trace, committed)
    verify_replay_chain(
        trace_path=args.trace,
        committed=committed,
        replay=replay,
        master_seed=args.master_seed,
    )
    chain = _chain(committed, replay)
    if replay.get("status") == "deterministic_ineligible":
        atomic_write_json(
            args.output,
            sign_payload(
                {
                    "schema_version": 1,
                    "analysis": "stage-d-all-child-replay-scorer",
                    "status": "deterministic_ineligible",
                    "source_trace_id": trace.get("id"),
                    **chain,
                    "regenerated_originals": [],
                    "pairs": [],
                }
            ),
        )
        return

    task = (trace.get("task") or {}).get("data") or {}
    paper = task.get("paper")
    reference = task.get("reference_evidence")
    if not isinstance(paper, str) or not isinstance(reference, list):
        raise TypeError("all-child trace lacks paper/reference evidence")
    calls = {call.call_index: call for call in extract_policy_calls(trace)}
    roots = [
        call for call in calls.values() if call.agent_depth == 0 and call.turn_index is not None
    ]
    if not roots:
        raise ValueError("all-child trace has no returning root")
    final_root = max(roots, key=lambda call: call.turn_index or 0)
    original_final_prompt_hash = _token_hash(final_root.prompt_token_ids)
    client = TokenInferenceClient(
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=120.0,
    )

    scored_originals = []
    for original in replay["regenerated_originals"]:
        token_ids = tuple(
            int(token_id) for token_id in original["downstream_generation"]["token_ids"]
        )
        text = _clean_action_text(client.detokenize(token_ids))
        parsed = parse_evidence(text)
        score = score_exact_spans(paper, parsed.spans, reference)
        scored_originals.append(
            {
                "target_call_index": int(original["target_call_index"]),
                "target_node_id": original["target_node_id"],
                "continuation_seed": int(original["continuation_seed"]),
                "terminal_token_sha256": _token_hash(token_ids),
                "terminal_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "parseable": parsed.parseable,
                "all_predicted_spans_verbatim": score["all_predicted_spans_verbatim"],
                "parsed_spans": list(parsed.spans),
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1"],
            }
        )

    scored_pairs = []
    all_distinct = True
    all_changed = True
    for pair in replay["pairs"]:
        target_index = int(pair["target_call_index"])
        target = calls[target_index]
        alternative_hash = str(pair["candidate_action_sha256"])
        distinct = alternative_hash != _token_hash(target.action_token_ids)
        changed = pair["branch_prompt_sha256"] != original_final_prompt_hash
        token_ids = tuple(int(token_id) for token_id in pair["downstream_generation"]["token_ids"])
        text = _clean_action_text(client.detokenize(token_ids))
        parsed = parse_evidence(text)
        score = score_exact_spans(paper, parsed.spans, reference)
        scored_pairs.append(
            {
                "target_call_index": target_index,
                "target_node_id": pair["target_node_id"],
                "alternative_index": int(pair["alternative_index"]),
                "alternative_distinct_from_original": distinct,
                "downstream_prompt_changed": changed,
                "terminal_token_sha256": _token_hash(token_ids),
                "terminal_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "parseable": parsed.parseable,
                "all_predicted_spans_verbatim": score["all_predicted_spans_verbatim"],
                "parsed_spans": list(parsed.spans),
                "precision": score["precision"],
                "recall": score["recall"],
                "f1": score["f1"],
            }
        )
        all_distinct = all_distinct and distinct
        all_changed = all_changed and changed

    result = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-all-child-replay-scorer",
            "status": "scored",
            "source_trace_id": trace.get("id"),
            **chain,
            "regenerated_originals": scored_originals,
            "all_alternatives_distinct_from_original": all_distinct,
            "all_downstream_prompts_changed": all_changed,
            "pairs": scored_pairs,
        }
    )
    atomic_write_json(args.output, result)


if __name__ == "__main__":
    main()
