"""Audit the frozen synthetic Stage D scaffold-usage SFT fallback."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _qasper_strings(path: Path) -> tuple[set[str], set[str]]:
    paper_ids: set[str] = set()
    evidence: set[str] = set()
    for row in _rows(path):
        paper_ids.add(str(row["paper_id"]))
        evidence.update(
            span
            for span in row["reference_evidence"]
            if isinstance(span, str) and len(span) >= 8
        )
    return paper_ids, evidence


def evaluate(
    *,
    dataset_path: Path,
    manifest_path: Path,
    qasper_path: Path,
) -> dict[str, Any]:
    rows = _rows(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paper_ids, evidence = _qasper_strings(qasper_path)
    serialized = dataset_path.read_text(encoding="utf-8")
    final_lists_parse = True
    exact_structure = True
    for row in rows:
        messages = row.get("messages") or []
        tools = row.get("tools") or []
        try:
            final = ast.literal_eval(messages[-1]["content"])
        except (IndexError, KeyError, SyntaxError, ValueError):
            final_lists_parse = False
            final = None
        final_lists_parse = final_lists_parse and (
            isinstance(final, list)
            and bool(final)
            and all(isinstance(item, str) and item for item in final)
        )
        tool_calls = messages[2].get("tool_calls") if len(messages) > 2 else []
        arguments = (
            tool_calls[0]["function"]["arguments"] if tool_calls else "{}"
        )
        try:
            code = json.loads(arguments)["code"]
        except (KeyError, TypeError, json.JSONDecodeError):
            code = ""
        exact_structure = exact_structure and (
            len(messages) == 5
            and [message["role"] for message in messages]
            == ["system", "user", "assistant", "tool", "assistant"]
            and len(tools) == 1
            and tools[0]["function"]["name"] == "ipython"
            and len(tool_calls) == 1
            and tool_calls[0]["function"]["name"] == "ipython"
            and "await asyncio.gather" in code
            and "rlm(prompt)" in code
            and messages[3].get("tool_call_id") == "call_0"
        )
    checks = {
        "example_count_is_32_and_at_most_50": len(rows) == 32,
        "synthetic_ids_are_unique": (
            len({row.get("synthetic_id") for row in rows}) == len(rows)
        ),
        "classification_is_honest": (
            manifest["classification"]
            == "shared_synthetic_scaffold_and_task_sft"
        ),
        "manifest_dataset_hash_matches": (
            manifest["dataset_sha256"] == _sha256(dataset_path)
        ),
        "all_final_payloads_are_nonempty_bare_lists": final_lists_parse,
        "all_rows_have_exact_ipython_rlm_structure": exact_structure,
        "no_reward_or_advantage_fields": all(
            not ({"reward", "rewards", "advantage", "advantages"} & set(row))
            for row in rows
        ),
        "no_qasper_paper_ids": not any(
            paper_id in serialized for paper_id in paper_ids
        ),
        "no_qasper_reference_spans": not any(
            span in serialized for span in evidence
        ),
        "fixed_final_checkpoint_selection": (
            manifest["selection"]
            == "fixed final step 8; no adaptive checkpoint selection"
        ),
    }
    return sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-synthetic-scaffold-sft-audit",
            "dataset_sha256": _sha256(dataset_path),
            "manifest_sha256": _sha256(manifest_path),
            "examples": len(rows),
            "checks": checks,
            "passes": all(checks.values()),
            "claim": (
                "Shared synthetic scaffold-and-task SFT only. It teaches "
                "IPython child invocation and exact-evidence workflow but "
                "contains no QASPER papers or reference spans."
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qasper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write_json(
        args.output,
        evaluate(
            dataset_path=args.dataset,
            manifest_path=args.manifest,
            qasper_path=args.qasper,
        ),
    )


if __name__ == "__main__":
    main()
