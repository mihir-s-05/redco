"""Focused cross-owner evidence polarity and wallet-before regressions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import test_stage_d_v13_prime_test_one_shot_evidence_v2 as evidence_tests
import test_stage_d_v13_prime_test_one_shot_lifecycle_v2 as lifecycle_tests

from redco.analysis import stage_d_v13_prime_test_one_shot_evidence_v2 as evidence
from redco.analysis import stage_d_v13_prime_test_one_shot_handoff_v2 as handoff_owner
from redco.analysis import stage_d_v13_prime_test_one_shot_lifecycle_v2 as lifecycle
from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    HANDOFF_NAMESPACE,
    READINESS_AUTHORITY,
    canonical_json,
    sha256_bytes,
)


@pytest.mark.parametrize(
    "mutation", ["forged", "noncanonical", "identity", "pagination", "row", "digest"]
)
def test_wallet_before_without_after_rejects_every_mutation(
    tmp_path: Path, mutation: str
) -> None:
    context, root = evidence_tests._failed_fixture(tmp_path / mutation)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    cleanup = cast(dict[str, Any], json.loads((root / "cleanup.json").read_bytes()))
    cleanup["wallet_after"] = None
    cleanup["errors"] = ["wallet:RuntimeError"]
    terminal["cleanup_proven"] = False
    terminal["cleanup_failures"] = ["wallet:RuntimeError"]
    wallet = cast(dict[str, Any], json.loads((root / "wallet-before.json").read_bytes()))
    if mutation == "forged":
        wallet = {"forged": True}
    elif mutation == "identity":
        wallet["wallet_identity_sha256"] = "0" * 64
    elif mutation == "pagination":
        wallet["pagination"]["pages"][0]["offset"] = 100
    elif mutation == "row":
        wallet["rows"][0]["amount_usd"] = "9"
    elif mutation == "digest":
        wallet["rows"][0]["semantic_row_sha256"] = "0" * 64
    wallet_raw = (
        json.dumps(wallet, indent=2).encode()
        if mutation == "noncanonical"
        else canonical_json(wallet)
    )
    evidence_tests._replace_bound_bytes(root, terminal, "wallet-before", wallet_raw)
    evidence_tests._replace_bound_json(root, terminal, "cleanup", cleanup)
    evidence_tests._resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


def test_coherent_wire_rewrite_is_accepted_only_when_wire_parser_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, root = evidence_tests._completed_with_history(tmp_path / "parser-disabled")
    terminal = json.loads((root / "terminal.json").read_bytes())
    lifecycle_tests._rewrite_coherent_wire_chain(context, root, terminal, "Z2FyYmFnZQ==")

    def framing_only(_blob: bytes, _algorithm: str) -> None:
        return None

    monkeypatch.setattr(handoff_owner, "_validate_ssh_key_blob", framing_only)
    assert evidence.verify_terminal_evidence(root, context.identity)["state"] == "completed"

def _replace_journal(
    context: Any, root: Path,
    terminal: dict[str, Any], records: list[dict[str, Any]],
) -> None:
    raw = b"".join(canonical_json(record) + b"\n" for record in records)
    evidence_tests._replace_bound_bytes(root, terminal, "command-journal", raw)
    evidence_tests._resign_terminal(context, root, terminal)


@evidence_tests._parametrize(
    "mutation",
    [
        "unknown_operation",
        "unknown_detail",
        "detail_type",
        "unpaired",
        "duplicate_outcome",
        "out_of_order",
        "wrong_operation_pair",
    ],
)
def test_resigned_terminal_rejects_journal_schema_and_pairing_mutations(
    tmp_path: Path, mutation: str
) -> None:
    context, root = evidence_tests._completed_with_history(tmp_path / mutation)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    records = evidence_tests._journal_records(root)
    if mutation == "unknown_operation":
        records[0]["operation"] = "unknown-operation"
    elif mutation == "unknown_detail":
        cast(dict[str, object], records[0]["details"])["unknown"] = None
    elif mutation == "detail_type":
        records[0]["details"] = []
    elif mutation == "unpaired":
        records.pop(1)
    elif mutation == "duplicate_outcome":
        records.insert(2, deepcopy(records[1]))
    elif mutation == "out_of_order":
        records[0], records[1] = records[1], records[0]
    else:
        records[1]["operation"] = "prime-api-create"
    for ordinal, record in enumerate(records, start=1):
        record["ordinal"] = ordinal
    _replace_journal(context, root, terminal, records)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


@evidence_tests._parametrize(
    "field,value",
    [
        ("raw_team_id", "team-secret"),
        ("provider_row", {"provider": "secret-provider"}),
        ("private_key_path", r"C:\\secret\\id_rsa"),
        ("credential", "fixture-secret"),
        ("raw_body", "provider-response"),
    ],
)
def test_resigned_terminal_rejects_every_journal_privacy_sentinel(
    tmp_path: Path, field: str, value: object
) -> None:
    context, root = evidence_tests._completed_with_history(tmp_path / field)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    records = evidence_tests._journal_records(root)
    cast(dict[str, object], records[0]["details"])[field] = value
    _replace_journal(context, root, terminal, records)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)
    assert b"team-secret" not in canonical_json(terminal)


@evidence_tests._parametrize("artifact", ["create-dispatch", "create-result"])
def test_resigned_terminal_rejects_arbitrary_create_artifact(
    tmp_path: Path, artifact: str
) -> None:
    context, root = evidence_tests._completed_with_history(tmp_path / artifact)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    evidence_tests._replace_bound_json(root, terminal, artifact, {"forged": True})
    evidence_tests._resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


def test_resigned_terminal_rejects_arbitrary_handoff_payload(tmp_path: Path) -> None:
    context, root = evidence_tests._completed_with_history(tmp_path / "handoff")
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    payload = canonical_json({"forged": True})
    signature = lifecycle._sign(context, payload, HANDOFF_NAMESPACE, timeout=30)
    envelope = evidence.signed_envelope(
        payload,
        signature,
        HANDOFF_NAMESPACE,
        context.identity,
        authority=READINESS_AUTHORITY,
    )
    evidence_tests._replace_bound_bytes(root, terminal, "handoff", payload)
    evidence_tests._replace_bound_bytes(root, terminal, "handoff-signature", signature)
    evidence_tests._replace_bound_bytes(root, terminal, "handoff-envelope", envelope)
    evidence_tests._resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)


@evidence_tests._parametrize(
    "mutation",
    [
        "unknown", "missing", "authorization", "claim", "transcript", "assessment",
        "resource", "pod", "ssh", "runtime", "ttl", "retry", "authority", "path",
        "nonce", "signature_file", "signature_replay", "known_hash", "known_empty",
        "known_malformed", "known_oversize", "known_host", "known_port",
        "known_same_endpoint_key", "keyscan_digest", "keyscan_reorder",
        "keyscan_duplicate", "keyscan_missing",
    ],
)
def test_resigned_terminal_rejects_handoff_semantic_and_signature_matrix(
    tmp_path: Path, mutation: str
) -> None:
    context, root = evidence_tests._completed_with_history(tmp_path / mutation)
    terminal = cast(dict[str, Any], json.loads((root / "terminal.json").read_bytes()))
    handoff = cast(dict[str, Any], json.loads((root / "handoff.json").read_bytes()))
    if mutation == "unknown":
        handoff["unknown"] = None
    elif mutation == "missing":
        handoff.pop("state")
    elif mutation == "authorization":
        handoff["authorization"]["commit"] = "0" * 40
    elif mutation in {"claim", "transcript"}:
        handoff[mutation]["sha256"] = "0" * 64
    elif mutation == "assessment":
        handoff["assessment"]["envelope_sha256"] = "0" * 64
    elif mutation == "resource":
        handoff["selected_resource_sha256"] = "0" * 64
    elif mutation == "pod":
        handoff["pod"]["identity_sha256"] = "0" * 64
    elif mutation == "ssh":
        handoff["ssh"]["port"] = 0
    elif mutation == "runtime":
        handoff["runtime"]["test_nodes"].reverse()
    elif mutation == "ttl":
        handoff["expires_at_epoch"] += 1
    elif mutation == "retry":
        handoff["retry"] = True
    elif mutation == "authority":
        handoff["authority"]["science_authorized"] = True
    elif mutation == "path":
        handoff["evidence_paths"]["claim"] = "elsewhere.json"
    elif mutation == "nonce":
        handoff["nonce"] = "0"
    known_hosts = (root / "known-hosts.txt").read_bytes()
    known_mutations = {
        "known_empty": b"",
        "known_malformed": b"[8.8.8.8]:2222 ssh-rsa fixture\n",
        "known_oversize": b"x" * (64 * 1024 + 1),
        "known_host": known_hosts.replace(b"[8.8.8.8]:2222", b"[1.1.1.1]:2222"),
        "known_port": known_hosts.replace(b":2222", b":22"),
        "known_same_endpoint_key": known_hosts.replace(b"QC3G", b"QC4G", 1),
    }
    if mutation == "known_hash":
        handoff["ssh"]["known_hosts_sha256"] = "0" * 64
    elif mutation in known_mutations:
        known_hosts = known_mutations[mutation]
        handoff["ssh"]["known_hosts_sha256"] = sha256_bytes(known_hosts)
        evidence_tests._replace_bound_bytes(root, terminal, "known-hosts", known_hosts)
    if mutation.startswith("keyscan_"):
        records = evidence_tests._journal_records(root)
        scans = [
            index for index, record in enumerate(records) if record["phase"] == "dispatch"
            and cast(str, record["operation"]).startswith("ssh-keyscan.exe ")
        ]
        if mutation == "keyscan_digest":
            cast(dict[str, object], records[scans[0] + 1]["details"])["stdout_sha256"] = "0" * 64
        elif mutation == "keyscan_reorder":
            second = scans[1]
            records[second : second + 4] = (
                records[second + 2 : second + 4] + records[second : second + 2]
            )
        elif mutation == "keyscan_duplicate":
            records[scans[1] : scans[1]] = deepcopy(records[scans[0] : scans[0] + 2])
        else:
            del records[scans[1] : scans[1] + 2]
        evidence_tests._bind_journal(root, terminal, records)
    raw = canonical_json(handoff)
    envelope_signature = lifecycle._sign(
        context, b"replay" if mutation == "signature_replay" else raw,
        HANDOFF_NAMESPACE, timeout=30,
    )
    envelope = evidence.signed_envelope(
        raw, envelope_signature, HANDOFF_NAMESPACE, context.identity,
        authority=READINESS_AUTHORITY,
    )
    evidence_tests._replace_bound_bytes(root, terminal, "handoff", raw)
    signature_file = b"forged" if mutation == "signature_file" else envelope_signature
    evidence_tests._replace_bound_bytes(root, terminal, "handoff-signature", signature_file)
    evidence_tests._replace_bound_bytes(root, terminal, "handoff-envelope", envelope)
    evidence_tests._resign_terminal(context, root, terminal)
    with pytest.raises(ValueError):
        evidence.verify_terminal_evidence(root, context.identity)
