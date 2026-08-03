#!/usr/bin/env python3
"""Audit the evidence-honest Stage-D action-closure corpus without writing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from redco.analysis.stage_d_exact_action import BehaviorAction
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} is not a lowercase SHA-256")
    return value


def _json_object(
    path: Path, *, require_canonical: bool = False
) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    if require_canonical and canonical_json(value) != raw:
        raise ValueError(f"{path} is not a canonical JSON object")
    return raw, value


def _audit_ledger(root: Path) -> dict[str, Any]:
    records_root = root / "records"
    evidence_root = root / "evidence"
    record_paths = sorted(records_root.glob("*.json"))
    if not record_paths or not evidence_root.is_dir():
        raise ValueError(f"{root} lacks retained ledger evidence")
    prior = "0" * 64
    ledger_id: str | None = None
    receipt_kinds: Counter[str] = Counter()
    referenced: set[str] = set()
    completed_actions: list[tuple[str, str, str]] = []
    for offset, path in enumerate(record_paths):
        if path.name != f"{offset:020d}.json":
            raise ValueError(f"{root} has noncontiguous ledger records")
        raw, record = _json_object(path, require_canonical=True)
        if set(record) != {
            "schema_version",
            "domain",
            "ledger_id",
            "offset",
            "prior_record_sha256",
            "record_kind",
            "body",
        }:
            raise ValueError(f"{path} has different ledger fields")
        if record["schema_version"] != 1 or record["domain"] != (
            "redco-stage-d-receipt-ledger-v2"
        ):
            raise ValueError(f"{path} has a different ledger schema")
        if ledger_id is None:
            ledger_id = record["ledger_id"]
        if not isinstance(ledger_id, str) or record["ledger_id"] != ledger_id:
            raise ValueError(f"{path} changes ledger identity")
        if record["offset"] != offset or record["prior_record_sha256"] != prior:
            raise ValueError(f"{path} breaks the ledger hash chain")
        body = record["body"]
        if not isinstance(body, dict):
            raise ValueError(f"{path} lacks a ledger body")
        if offset == 0:
            if record["record_kind"] != "genesis":
                raise ValueError(f"{path} is not the genesis record")
            refs: list[str] = []
        elif record["record_kind"] == "receipt":
            if set(body) != {
                "evidence_refs",
                "receipt",
                "receipt_kind",
                "receipt_sha256",
            }:
                raise ValueError(f"{path} has a different receipt envelope")
            receipt = body["receipt"]
            if not isinstance(receipt, dict):
                raise ValueError(f"{path} lacks a receipt payload")
            kind = body["receipt_kind"]
            if not isinstance(kind, str) or not kind:
                raise ValueError(f"{path} has an invalid receipt kind")
            if (
                receipt.get("schema_version") != 1
                or receipt.get("receipt_kind") != kind
                or receipt.get("ledger_id") != ledger_id
                or receipt.get("ledger_offset") != offset
                or receipt.get("prior_chain_sha256") != prior
                or body["receipt_sha256"] != _sha256(canonical_json(receipt))
            ):
                raise ValueError(f"{path} receipt identity or hash differs")
            refs = body["evidence_refs"]
            if not isinstance(refs, list) or refs != sorted(set(refs)):
                raise ValueError(f"{path} has invalid evidence references")
            receipt_kinds[kind] += 1
            if kind == "source_policy_call_completed":
                response_sha256 = _require_sha256(
                    receipt.get("response_sha256"), f"{path} response"
                )
                action_digest = _require_sha256(
                    receipt.get("action_digest"), f"{path} action"
                )
                key_digest = _require_sha256(
                    receipt.get("exact_action_key_digest"), f"{path} action key"
                )
                if response_sha256 not in refs:
                    raise ValueError(f"{path} omits completed action evidence")
                completed_actions.append((response_sha256, action_digest, key_digest))
        elif record["record_kind"] in {
            "action_reservation",
            "model_call_completed",
            "model_call_started",
            "recorded_action_materialized",
        }:
            if set(body) != {"event", "evidence_refs"} or not isinstance(
                body["event"], dict
            ):
                raise ValueError(f"{path} has a different lifecycle event envelope")
            refs = body["evidence_refs"]
            if not isinstance(refs, list) or refs != sorted(set(refs)):
                raise ValueError(f"{path} has invalid evidence references")
        else:
            raise ValueError(f"{path} has an unknown record kind")
        for digest in refs:
            digest = _require_sha256(digest, f"{path} evidence reference")
            evidence = evidence_root / digest
            if not evidence.is_file() or _sha256(evidence.read_bytes()) != digest:
                raise ValueError(f"{path} references absent or changed evidence")
            referenced.add(digest)
        prior = _sha256(raw)

    terminations: Counter[str] = Counter()
    action_count = 0
    for path in sorted(evidence_root.iterdir()):
        if not path.is_file() or _sha256(path.read_bytes()) != path.name:
            raise ValueError(f"{path} violates content addressing")
    for response_sha256, action_digest, key_digest in completed_actions:
        path = evidence_root / response_sha256
        envelope = json.loads(path.read_bytes())
        if not isinstance(envelope, dict) or envelope.get("domain") not in {
            "redco-stage-d-behavior-action-v1",
            "redco-stage-d-behavior-action-v2",
        }:
            raise ValueError(f"{path} is not completed action evidence")
        prompt = tuple(envelope["action"]["key"]["prompt_token_ids"])

        def render_prompt(
            _request: Mapping[str, Any], prompt: tuple[int, ...] = prompt
        ) -> tuple[int, ...]:
            return prompt

        action = BehaviorAction.from_bytes(
            path.read_bytes(),
            validate_action=lambda _request, _message, _tokens: None,
            render_prompt=render_prompt,
        )
        if action.digest != action_digest or action.key.digest != key_digest:
            raise ValueError(f"{path} differs from its completed-call receipt")
        action_count += 1
        terminations[action.termination_kind] += 1
    return {
        "record_count": len(record_paths),
        "head_sha256": prior,
        "referenced_evidence_count": len(referenced),
        "receipt_counts": dict(sorted(receipt_kinds.items())),
        "cryptographic_behavior_action_envelopes_reloaded": action_count,
        "semantic_renderer_replay_performed": False,
        "termination_counts": dict(sorted(terminations.items())),
    }


def audit(repository: Path, corpus_path: Path) -> dict[str, Any]:
    corpus_raw, corpus = _json_object(corpus_path)
    if corpus.get("domain") != "redco-stage-d-action-closure-corpus-v1":
        raise ValueError("action-closure corpus domain differs")
    cases = corpus.get("cases")
    if not isinstance(cases, list) or [case.get("version") for case in cases] != list(
        range(1, 11)
    ):
        raise ValueError("action-closure corpus must describe versions 1 through 10")
    audited: list[dict[str, Any]] = []
    total_actions = 0
    for case in cases:
        report_path = repository / case["report_path"]
        report_raw, _ = _json_object(report_path)
        if _sha256(report_raw) != case["report_sha256"]:
            raise ValueError(f"v{case['version']} terminal report changed")
        result: dict[str, Any] = {
            "version": case["version"],
            "evidence_level": case["evidence_level"],
            "report_sha256": case["report_sha256"],
        }
        if case["ledger_root"] is not None:
            ledger = _audit_ledger(repository / case["ledger_root"])
            expected = case["expected_ledger"]
            if ledger != expected:
                raise ValueError(f"v{case['version']} ledger audit differs")
            result["ledger"] = ledger
            total_actions += ledger["cryptographic_behavior_action_envelopes_reloaded"]
        archive = case["archive"]
        if archive is not None:
            archive_path = repository / archive["path"]
            if _sha256(archive_path.read_bytes()) != archive["sha256"]:
                raise ValueError(f"v{case['version']} terminal archive changed")
            result["archive_sha256"] = archive["sha256"]
        audited.append(result)
    return {
        "schema_version": 1,
        "domain": "redco-stage-d-action-closure-corpus-audit-v1",
        "passes": True,
        "corpus_sha256": _sha256(corpus_raw),
        "case_count": len(audited),
        "cryptographic_behavior_action_envelopes_reloaded": total_actions,
        "semantic_renderer_replay_performed": False,
        "cases": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("configs/stage-d/stage-d1-action-closure-corpus-v1.json"),
    )
    args = parser.parse_args()
    print(canonical_json(audit(args.repository, args.repository / args.corpus)).decode())


if __name__ == "__main__":
    main()
