"""Record the retrospective CPU diagnosis of the terminal Stage D v4.10 fixture."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redco.analysis.stage_d_all_child_support import (  # noqa: E402
    precommit_all_depth_one_targets,
)
from redco.integrations.signed_subprocess import (  # noqa: E402
    atomic_write_json,
    sign_payload,
    verify_signed_payload,
)
from redco.integrations.verifiers_trace import (  # noqa: E402
    extract_policy_calls,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_trace(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    traces = record.get("traces")
    if not isinstance(traces, list) or len(traces) != 1:
        raise ValueError("diagnostic requires one wrapped trace")
    trace = traces[0]
    if not isinstance(trace, dict):
        raise TypeError("trace must be an object")
    return trace


def _tool_return_items(trace: dict[str, Any]) -> list[str]:
    for node in reversed(trace["nodes"]):
        message = node.get("message") or {}
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        start = content.find("['")
        if start < 0:
            continue
        parsed = ast.literal_eval(content[start:].strip())
        if isinstance(parsed, list) and all(
            isinstance(item, str) for item in parsed
        ):
            return parsed
    raise ValueError("no parseable two-child tool return found")


def _excerpt_chars(prompt: str) -> int:
    marker = "in this excerpt:\n"
    if marker not in prompt:
        raise ValueError("child prompt lacks excerpt marker")
    return len(prompt.split(marker, 1)[1])


def build_report(
    *,
    trace_path: Path,
    eligibility_path: Path,
) -> dict[str, Any]:
    trace = _load_trace(trace_path)
    calls = extract_policy_calls(trace)
    roots = [call for call in calls if call.agent_depth == 0]
    children = [call for call in calls if call.agent_depth == 1]
    if len(roots) != 3 or len(children) != 2:
        raise ValueError("unexpected v4.10 trace topology")
    nodes = trace["nodes"]
    child_rows = []
    tool_items = _tool_return_items(trace)
    for call in children:
        prompt_message = nodes[call.node_index - 1]["message"]
        completion_message = nodes[call.node_index]["message"]
        prompt = str(prompt_message.get("content", ""))
        completion = str(completion_message.get("content", ""))
        matching_slots = [
            index for index, item in enumerate(tool_items) if item == completion
        ]
        child_rows.append(
            {
                "native_call_index": call.call_index,
                "session_id": call.session_id,
                "parent_turn_index": call.parent_turn_index,
                "excerpt_characters": _excerpt_chars(prompt),
                "completion_sha256": hashlib.sha256(
                    completion.encode("utf-8")
                ).hexdigest(),
                "tool_return_slot": (
                    matching_slots[0] if len(matching_slots) == 1 else None
                ),
                "mentions_exact_evidence": (
                    "420 milliseconds to 260 milliseconds" in completion
                    and "81.4 percent" in completion
                ),
            }
        )
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    verify_signed_payload(eligibility)
    precommit = precommit_all_depth_one_targets(trace_path)
    verify_signed_payload(precommit)
    first_root_tool = nodes[roots[0].node_index]["message"]["tool_calls"][0]
    first_code = json.loads(first_root_tool["arguments"])["code"]
    first_tool_output = str(nodes[roots[0].node_index + 1]["message"]["content"])
    selected_index = int(eligibility["target_call_index"])
    selected = next(
        row for row in child_rows if row["native_call_index"] == selected_index
    )
    sibling = next(
        row for row in child_rows if row["native_call_index"] != selected_index
    )
    checks = {
        "first_root_omitted_asyncio_import": "import asyncio" not in first_code,
        "first_tool_failed_with_name_error": (
            "NameError" in first_tool_output
            and "asyncio" in first_tool_output
        ),
        "later_root_recovered": trace.get("ok") is True,
        "native_first_child_was_empty_excerpt": (
            selected["excerpt_characters"] == 0
        ),
        "native_second_child_had_exact_evidence": (
            sibling["mentions_exact_evidence"] is True
        ),
        "semantic_tool_order_differs_from_native_call_order": (
            selected["tool_return_slot"] == 1
            and sibling["tool_return_slot"] == 0
        ),
        "all_branch_scores_flat": (
            eligibility["reward_informativeness"]["range"] == 0.0
        ),
        "all_child_precommit_contains_two_targets": (
            precommit["candidate_count"] == 2
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"v4.10 diagnostic check failed: {checks}")
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-v4-10-retrospective-trace-diagnostic",
            "scope": "CPU-only descriptive diagnosis; not confirmatory data",
            "source_trace": {
                "path": trace_path.as_posix(),
                "sha256": _sha256(trace_path),
            },
            "source_eligibility": {
                "path": eligibility_path.as_posix(),
                "sha256": _sha256(eligibility_path),
            },
            "root_calls": len(roots),
            "child_calls": len(children),
            "legacy_selected_call_index": selected_index,
            "children_in_native_call_order": child_rows,
            "all_child_precommit": precommit,
            "checks": checks,
            "interpretation": {
                "topology_failure": (
                    "The extra root was a self-correction after omitting the "
                    "demonstrated asyncio import, not an absent return path."
                ),
                "selector_failure": (
                    "Native call order under asyncio.gather differed from "
                    "semantic gather order, so first-native-child selected "
                    "the guaranteed empty second shard."
                ),
                "informativeness_failure": (
                    "Changing the empty child could not affect reward while "
                    "the fixed sibling already supplied the complete answer."
                ),
                "v4_10_disposition": (
                    "The frozen gate remains failed and terminal; these facts "
                    "justify a fresh all-child successor, not reinterpretation."
                ),
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(
        args.output,
        build_report(
            trace_path=args.trace,
            eligibility_path=args.eligibility,
        ),
    )


if __name__ == "__main__":
    main()
