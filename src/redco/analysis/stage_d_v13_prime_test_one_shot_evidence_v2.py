"""Strict signed-evidence codec and verifier for the Prime test one-shot."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    ALLOWED_GPU_LABELS,
    ARTIFACT_FILENAMES,
    ASSESSMENT_DOMAIN,
    ASSESSMENT_TTL_SECONDS,
    AUTHORIZATION_PATH,
    CLAIM_DOMAIN,
    CLEANUP_TIMEOUT_SECONDS,
    HANDOFF_NAMESPACE,
    MAX_PRIME_CLI_CALLS,
    MAX_WALLET_API_CALLS,
    MAXIMUM_POD_SECONDS,
    MAXIMUM_RATE_USD,
    READINESS_AUTHORITY,
    RUNTIME_AUTHORITY,
    SIGNED_ENVELOPE_DOMAIN,
    CommandJournalSummary,
    CreateDispatchSummary,
    CreateResultSummary,
    SigningIdentity,
    authority_value,
    canonical_json,
    closed_authority,
    sha256_bytes,
    strict_object,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_prime_v2 import (
    assess_pages,
    replay_command_journal,
    validate_create_dispatch,
    validate_create_result,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_remote_v2 import (
    HandoffSummary,
    remote_test_script,
    validate_gpu_facts,
    validate_handoff_payload,
    validate_junit,
    verify_openssh_sshsig,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_runtime_binding_v2 import (
    V2_RUNTIME_BINDING,
    RuntimeBinding,
    _is_trusted_binding,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_wallet_v2 import (
    SanitizedWalletSnapshot,
    decimal_value,
    replay_wallet_reconciliation,
    validate_wallet_snapshot_bytes,
    validate_wallet_snapshot_journal,
)

MAX_TRANSCRIPT_BYTES = 34 * 1024 * 1024
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_FAILURE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?::[A-Za-z][A-Za-z0-9_]*)?")
_TERMINAL_STATES = frozenset(
    {"ambiguous_capacity", "completed", "failed_terminal", "no_qualifying_capacity"}
)
_TERMINAL_EXCLUSIONS = {
    "terminal",
    "terminal-envelope",
    "terminal-publication-failure",
}
_REMOTE_ARTIFACTS = {"gpu-facts", "junit", "remote-status"}
_SIGNED_ASSESSMENT_ARTIFACTS = {"transcript", "assessment", "assessment-envelope"}
_HANDOFF_ARTIFACTS = {
    "known-hosts",
    "handoff",
    "handoff-signature",
    "handoff-envelope",
}


@dataclass(frozen=True, slots=True)
class CleanupSummary:
    errors: tuple[str, ...]
    proven: bool
    owned_identity_sha256s: tuple[str, ...]
    terminated_identity_sha256s: tuple[str, ...]
    wallet_reconciliation: Mapping[str, object] | None


def signed_envelope(
    payload: bytes,
    signature: bytes,
    namespace: str,
    identity: SigningIdentity,
    *,
    authority: Mapping[str, bool] = RUNTIME_AUTHORITY,
    domain: str = SIGNED_ENVELOPE_DOMAIN,
) -> bytes:
    return canonical_json(
        {
            "schema_version": 2,
            "domain": domain,
            "state": "detached_signature",
            "namespace": namespace,
            "payload": {
                "base64": base64.b64encode(payload).decode(),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            },
            "signature": {
                "base64": base64.b64encode(signature).decode(),
                "bytes": len(signature),
                "sha256": sha256_bytes(signature),
            },
            "identity": identity.sanitized(),
            "authority": dict(authority),
        }
    )


def verify_signed_envelope(
    raw: bytes,
    payload: bytes,
    namespace: str,
    identity: SigningIdentity,
    *,
    authority: Mapping[str, bool] = RUNTIME_AUTHORITY,
    domain: str = SIGNED_ENVELOPE_DOMAIN,
) -> bytes:
    authority = closed_authority(authority, "signed envelope")
    value = strict_object(
        raw,
        {
            "schema_version",
            "domain",
            "state",
            "namespace",
            "payload",
            "signature",
            "identity",
            "authority",
        },
        "signed envelope",
    )
    if (
        value["schema_version"] != 2
        or value["domain"] != domain
        or value["namespace"] != namespace
        or value["identity"] != identity.sanitized()
        or value["state"] != "detached_signature"
    ):
        raise ValueError("Prime one-shot signed envelope differs")
    authority_value(value["authority"], authority, "signed envelope")
    payload_value = value["payload"]
    signature_value = value["signature"]
    identity_value = value["identity"]
    if (
        type(payload_value) is not dict
        or set(payload_value) != {"base64", "bytes", "sha256"}
        or type(signature_value) is not dict
        or set(signature_value) != {"base64", "bytes", "sha256"}
        or type(identity_value) is not dict
        or set(identity_value)
        != {"principal", "key_type", "fingerprint_sha256", "allowed_signers_sha256"}
    ):
        raise ValueError("signed envelope projection differs")
    if type(payload_value["base64"]) is not str or type(signature_value["base64"]) is not str:
        raise ValueError("signed envelope Base64 differs")
    try:
        decoded = base64.b64decode(payload_value["base64"], validate=True)
        signature = base64.b64decode(signature_value["base64"], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("signed envelope Base64 differs") from error
    if (
        decoded != payload
        or base64.b64encode(decoded).decode() != payload_value["base64"]
        or base64.b64encode(signature).decode() != signature_value["base64"]
        or payload_value["sha256"] != sha256_bytes(payload)
        or type(payload_value["bytes"]) is not int
        or payload_value["bytes"] != len(payload)
        or signature_value["sha256"] != sha256_bytes(signature)
        or type(signature_value["bytes"]) is not int
        or signature_value["bytes"] != len(signature)
    ):
        raise ValueError("Prime one-shot signed envelope digest differs")
    verify_openssh_sshsig(payload, signature, identity.public_key, namespace)
    return signature


def artifact_dag(paths: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, path in sorted(paths.items()):
        if path.is_file() and not path.is_symlink():
            raw = path.read_bytes()
            result[name] = {"path": path.name, "bytes": len(raw), "sha256": sha256_bytes(raw)}
    return result


def _authorization_projection(
    value: object, *, expected_authorization_path: str = AUTHORIZATION_PATH
) -> dict[str, str]:
    keys = {
        "commit", "tree", "parent",
        "authorization_path", "authorization_sha256", "authorization_blob",
    }
    if type(value) is not dict or set(cast(dict[object, object], value)) != keys:
        raise ValueError("Prime one-shot terminal authorization schema differs")
    result = cast(dict[str, object], value)
    if any(type(result[key]) is not str for key in keys):
        raise ValueError("Prime one-shot terminal authorization type differs")
    normalized = cast(dict[str, str], result)
    if (
        any(_HEX40.fullmatch(normalized[key]) is None for key in ("commit", "tree", "parent"))
        or _HEX40.fullmatch(normalized["authorization_blob"]) is None
        or _HEX64.fullmatch(normalized["authorization_sha256"]) is None
        or normalized["authorization_path"] != expected_authorization_path
        or normalized["commit"] == normalized["parent"]
    ):
        raise ValueError("Prime one-shot terminal authorization binding differs")
    return normalized


def _claim_projection(
    raw: bytes,
    authorization: Mapping[str, str],
    *,
    expected_claim_domain: str = CLAIM_DOMAIN,
    expected_authority: Mapping[str, bool] = RUNTIME_AUTHORITY,
) -> dict[str, Any]:
    expected_authority = closed_authority(expected_authority, "claim")
    claim = strict_object(
        raw,
        {
            "schema_version", "domain", "state", "authorization",
            "created_at_epoch", "nonce", "availability_attempt_limit",
            "create_dispatch_limit", "monitoring", "retry", "authority",
        },
        "Prime one-shot claim",
    )
    created = claim["created_at_epoch"]
    if (
        claim["schema_version"] != 2
        or claim["domain"] != expected_claim_domain
        or claim["state"] != "observation_attempt_consumed"
        or claim["authorization"] != dict(authorization)
        or type(created) is not int
        or created < 0
        or type(claim["nonce"]) is not str
        or _HEX64.fullmatch(claim["nonce"]) is None
        or type(claim["availability_attempt_limit"]) is not int
        or claim["availability_attempt_limit"] != 1
        or type(claim["create_dispatch_limit"]) is not int
        or claim["create_dispatch_limit"] != 1
        or claim["monitoring"] is not False
        or claim["retry"] is not False
    ):
        raise ValueError("Prime one-shot claim binding differs")
    authority_value(claim["authority"], expected_authority, "claim")
    return claim


def _assessment_projection(
    raw: bytes,
    authorization: Mapping[str, str],
    *,
    expected_domain: str = ASSESSMENT_DOMAIN,
    expected_schema_version: int = 2,
    expected_authority: Mapping[str, bool] = RUNTIME_AUTHORITY,
    expected_ttl_seconds: int = ASSESSMENT_TTL_SECONDS,
) -> dict[str, Any]:
    expected_authority = closed_authority(expected_authority, "assessment")
    assessment = strict_object(
        raw,
        {
            "schema_version", "domain", "state", "reason", "captured_at_epoch",
            "expires_at_epoch", "checkout", "transcript_payload_sha256",
            "row_count", "eligible_count", "duplicate_identity", "selection_order",
            "selected_resource_sha256", "selected_facts", "attempt_consumed",
            "retry", "authority",
        },
        "Prime one-shot assessment",
    )
    captured = assessment["captured_at_epoch"]
    expires = assessment["expires_at_epoch"]
    row_count = assessment["row_count"]
    eligible_count = assessment["eligible_count"]
    state = assessment["state"]
    reason = assessment["reason"]
    if (
        assessment["schema_version"] != expected_schema_version
        or assessment["domain"] != expected_domain
        or type(captured) is not int
        or captured < 0
        or type(expires) is not int
        or expires != captured + expected_ttl_seconds
        or assessment["checkout"] != dict(authorization)
        or type(assessment["transcript_payload_sha256"]) is not str
        or _HEX64.fullmatch(assessment["transcript_payload_sha256"]) is None
        or type(row_count) is not int
        or row_count < 0
        or type(eligible_count) is not int
        or not 0 <= eligible_count <= row_count
        or type(assessment["duplicate_identity"]) is not bool
        or assessment["selection_order"]
        != "hourly_rate,gpu_label,canonical_resource_sha256"
        or assessment["attempt_consumed"] is not True
        or assessment["retry"] is not False
    ):
        raise ValueError("Prime one-shot assessment binding differs")
    authority_value(assessment["authority"], expected_authority, "assessment")
    selected_hash = assessment["selected_resource_sha256"]
    selected_facts = assessment["selected_facts"]
    duplicate = assessment["duplicate_identity"]
    if state == "no_qualifying_capacity":
        valid = (
            reason == "no_eligible_resource"
            and eligible_count == 0
            and duplicate is False
            and selected_hash is None
            and selected_facts is None
        )
    elif state == "ambiguous_capacity":
        valid = (
            reason == "duplicate_resource_identity"
            and duplicate is True
            and selected_hash is None
            and selected_facts is None
        )
    elif state == "qualifying_capacity":
        facts = cast(dict[str, object], selected_facts) if type(selected_facts) is dict else None
        valid = (
            reason == "deterministic_first_eligible"
            and duplicate is False
            and eligible_count >= 1
            and type(selected_hash) is str
            and _HEX64.fullmatch(selected_hash) is not None
            and facts is not None
            and set(facts)
            == {
                "gpu_type", "gpu_count", "gpu_memory_gb",
                "is_spot", "hourly_rate_usd", "disk_size",
            }
            and type(facts["gpu_type"]) is str
            and facts["gpu_type"] in ALLOWED_GPU_LABELS
            and type(facts["gpu_count"]) is int
            and facts["gpu_count"] == 2
            and type(facts["gpu_memory_gb"]) is int
            and facts["gpu_memory_gb"] == 96
            and facts["is_spot"] is False
            and type(facts["disk_size"]) is int
            and facts["disk_size"] == 0
            and Decimal() < decimal_value(facts["hourly_rate_usd"], "assessment rate")
            <= Decimal(str(MAXIMUM_RATE_USD))
        )
    else:
        valid = False
    if not valid:
        raise ValueError("Prime one-shot assessment state differs")
    return assessment


def _transcript_projection(
    raw: bytes,
    assessment_raw: bytes | None,
    assessment: Mapping[str, Any] | None,
    authorization: Mapping[str, str],
    assessment_binding: RuntimeBinding,
) -> dict[str, object]:
    if len(raw) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("Prime one-shot transcript exceeds bound")
    transcript = strict_object(
        raw,
        {"pages", "diagnostic", "failure", "request_count"},
        "Prime one-shot transcript",
    )
    replay = v5._replay_transcript(
        transcript["pages"],
        transcript["diagnostic"],
        transcript["failure"],
        transcript["request_count"],
    )
    if assessment_raw is not None and assessment is not None:
        if assessment["transcript_payload_sha256"] != replay["payload_sha256"]:
            raise ValueError("Prime one-shot transcript payload binding differs")
        expected, _resource = assess_pages(
            cast(list[dict[str, object]], transcript["pages"]),
            authorization,
            cast(int, assessment["captured_at_epoch"]),
            assessment_binding,
        )
        if expected != assessment_raw:
            raise ValueError("Prime one-shot transcript assessment replay differs")
    return replay


def _failure_list(value: object, label: str) -> list[str]:
    if type(value) is not list or any(
        type(item) is not str or _FAILURE_NAME.fullmatch(item) is None
        for item in cast(list[object], value)
    ):
        raise ValueError(f"Prime one-shot terminal {label} differs")
    return cast(list[str], value)


def _journal_summary(root: Path, dag: Mapping[str, object]) -> CommandJournalSummary:
    if "command-records" not in dag:
        raise ValueError("Prime one-shot terminal lacks command records")
    records = (root / ARTIFACT_FILENAMES["command-records"]).read_bytes()
    if "command-journal" not in dag:
        if records != b"[]":
            raise ValueError("Prime one-shot commands lack their journal")
        return CommandJournalSummary(0, 0, 0, None, None, None, None, None, (), (), (), ())
    return replay_command_journal(
        records,
        (root / ARTIFACT_FILENAMES["command-journal"]).read_bytes(),
    )


def _bound_dag(root: Path, value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("Prime one-shot terminal DAG schema differs")
    dag = cast(dict[str, object], value)
    if (
        "claim" not in dag
        or any(key not in ARTIFACT_FILENAMES for key in dag)
        or any(key in dag for key in _TERMINAL_EXCLUSIONS)
    ):
        raise ValueError("Prime one-shot terminal DAG topology differs")
    expected_paths: set[str] = set()
    bound = {"terminal.json", "terminal-envelope.json"}
    for name, raw_binding in dag.items():
        if type(name) is not str or type(raw_binding) is not dict:
            raise ValueError("Prime one-shot terminal DAG binding schema differs")
        binding = cast(dict[str, object], raw_binding)
        if set(binding) != {"path", "bytes", "sha256"}:
            raise ValueError("Prime one-shot terminal DAG binding schema differs")
        expected_path = ARTIFACT_FILENAMES[name]
        byte_count = binding["bytes"]
        digest = binding["sha256"]
        if (
            binding["path"] != expected_path
            or expected_path in expected_paths
            or type(byte_count) is not int
            or byte_count < 0
            or type(digest) is not str
            or _HEX64.fullmatch(digest) is None
        ):
            raise ValueError("Prime one-shot terminal DAG binding differs")
        expected_paths.add(expected_path)
        path = root / expected_path
        info = path.lstat()
        if (
            path.parent != root
            or path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise ValueError("terminal artifact path differs")
        raw = path.read_bytes()
        if byte_count != len(raw) or digest != sha256_bytes(raw):
            raise ValueError("terminal artifact digest differs")
        bound.add(path.name)
    if {path.name for path in root.iterdir() if path.is_file()} != bound:
        raise ValueError("terminal has unbound evidence")
    return dag


def _cleanup_projection(
    root: Path,
    dag: Mapping[str, object],
    wallet_before_value: object | None,
    journal: CommandJournalSummary,
    wallet_before_requests: int,
) -> CleanupSummary | None:
    if "cleanup" not in dag:
        return None
    cleanup = strict_object(
        (root / ARTIFACT_FILENAMES["cleanup"]).read_bytes(),
        {
            "owned_identity_sha256s",
            "terminated_identity_sha256s",
            "pods_after_count",
            "disks_after_count",
            "wallet_after",
            "errors",
        },
        "Prime one-shot cleanup",
    )
    owned = cleanup["owned_identity_sha256s"]
    terminated = cleanup["terminated_identity_sha256s"]
    pods_after = cleanup["pods_after_count"]
    disks_after = cleanup["disks_after_count"]
    if (
        type(owned) is not list
        or any(type(item) is not str or _HEX64.fullmatch(item) is None for item in owned)
        or owned != sorted(set(owned))
        or type(terminated) is not list
        or any(type(item) is not str or _HEX64.fullmatch(item) is None for item in terminated)
        or terminated != sorted(set(terminated))
        or not (pods_after is None or (type(pods_after) is int and pods_after >= 0))
        or not (disks_after is None or (type(disks_after) is int and disks_after >= 0))
    ):
        raise ValueError("Prime one-shot cleanup binding differs")
    errors = _failure_list(cleanup["errors"], "cleanup evidence failures")
    wallet_reconciliation: Mapping[str, object] | None = None
    if cleanup["wallet_after"] is not None:
        if wallet_before_value is None:
            raise ValueError("Prime one-shot cleanup lacks wallet-before evidence")
        wallet_reconciliation = replay_wallet_reconciliation(
            wallet_before_value, cleanup["wallet_after"]
        )
        reconciliation = cast(dict[str, object], cleanup["wallet_after"])
        after_value = cast(dict[str, object], reconciliation["after_snapshot"])
        after_pagination = cast(dict[str, object], after_value["pagination"])
        after_requests = cast(int, after_pagination["request_count"])
        outcomes = journal.wallet_outcomes[
            wallet_before_requests : wallet_before_requests + after_requests
        ]
        validate_wallet_snapshot_journal(
            after_value, expected_phase="postcleanup", outcomes=outcomes
        )
        if len(journal.wallet_outcomes) != wallet_before_requests + after_requests:
            raise ValueError("Prime wallet journal has an unbound outcome")
    elif len(journal.wallet_outcomes) != wallet_before_requests:
        raise ValueError("Prime wallet journal has an unbound outcome")
    proven = (
        pods_after == 0
        and disks_after == 0
        and cleanup["wallet_after"] is not None
        and not errors
    )
    return CleanupSummary(
        tuple(errors), proven, tuple(cast(list[str], owned)),
        tuple(cast(list[str], terminated)), wallet_reconciliation,
    )


def _remote_projection(
    root: Path, dag: Mapping[str, object], assessment: Mapping[str, Any] | None
) -> tuple[bool, bool]:
    present = _REMOTE_ARTIFACTS & set(dag)
    if present and present != _REMOTE_ARTIFACTS:
        raise ValueError("Prime one-shot remote evidence is incomplete")
    if not present:
        return False, False
    if assessment is None or assessment["state"] != "qualifying_capacity":
        raise ValueError("Prime one-shot remote evidence lacks a qualifying assessment")
    status = strict_object(
        (root / ARTIFACT_FILENAMES["remote-status"]).read_bytes(),
        {"schema_version", "returncode"},
        "Prime one-shot remote status",
    )
    if (
        status["schema_version"] != 2
        or type(status["returncode"]) is not int
        or not 0 <= status["returncode"] <= 255
    ):
        raise ValueError("Prime one-shot remote status differs")
    passed = status["returncode"] == 0
    validate_gpu_facts(
        (root / ARTIFACT_FILENAMES["gpu-facts"]).read_bytes(),
        assessment["selected_facts"],
    )
    validate_junit(
        (root / ARTIFACT_FILENAMES["junit"]).read_bytes(),
        require_success=passed,
    )
    return True, passed


def _wallet_before_projection(
    root: Path, dag: Mapping[str, object], journal: CommandJournalSummary
) -> tuple[object | None, SanitizedWalletSnapshot | None, int]:
    if "wallet-before" not in dag:
        return None, None, 0
    raw = (root / ARTIFACT_FILENAMES["wallet-before"]).read_bytes()
    summary = validate_wallet_snapshot_bytes(raw, expected_phase="precreate")
    value = cast(dict[str, object], json.loads(raw))
    pagination = cast(dict[str, object], value["pagination"])
    requests = cast(int, pagination["request_count"])
    validate_wallet_snapshot_journal(
        value, expected_phase="precreate", outcomes=journal.wallet_outcomes[:requests]
    )
    return value, summary, requests


def _create_projections(
    root: Path,
    dag: Mapping[str, object],
    authorization: Mapping[str, str],
    assessment: Mapping[str, Any] | None,
    journal: CommandJournalSummary,
    expected_authority: Mapping[str, bool] = READINESS_AUTHORITY,
) -> tuple[CreateDispatchSummary | None, CreateResultSummary | None]:
    dispatch = (
        validate_create_dispatch(
            (root / ARTIFACT_FILENAMES["create-dispatch"]).read_bytes(),
            authority=expected_authority,
        )
        if "create-dispatch" in dag
        else None
    )
    result = (
        validate_create_result(
            (root / ARTIFACT_FILENAMES["create-result"]).read_bytes(),
            authorization,
            authority=expected_authority,
        )
        if "create-result" in dag
        else None
    )
    selected = None if assessment is None else assessment["selected_resource_sha256"]
    if dispatch is None:
        if journal.create_payload_sha256 is not None:
            raise ValueError("Prime create journal lacks its dispatch artifact")
    elif (
        assessment is None
        or assessment["state"] != "qualifying_capacity"
        or dispatch.resource_sha256 != selected
        or dispatch.payload_sha256 != journal.create_payload_sha256
    ):
        raise ValueError("Prime create dispatch semantic binding differs")
    if result is not None and (
        dispatch is None
        or result.status_code != journal.create_status_code
        or result.pod_identity_sha256 != journal.create_pod_identity_sha256
        or result.response_sha256 != journal.create_response_sha256
        or result.response_bytes != journal.create_response_bytes
    ):
        raise ValueError("Prime create result semantic binding differs")
    return dispatch, result


def _handoff_projection(
    root: Path,
    dag: Mapping[str, object],
    authorization: Mapping[str, str],
    assessment: Mapping[str, Any] | None,
    result: CreateResultSummary | None,
    identity: SigningIdentity,
    journal: CommandJournalSummary,
    *,
    expected_namespace: str = HANDOFF_NAMESPACE,
    expected_authority: Mapping[str, bool] = READINESS_AUTHORITY,
    expected_domain: str = SIGNED_ENVELOPE_DOMAIN,
) -> HandoffSummary | None:
    if not (_HANDOFF_ARTIFACTS & set(dag)):
        return None
    if not set(dag) >= _HANDOFF_ARTIFACTS or assessment is None or result is None:
        raise ValueError("Prime one-shot handoff evidence is incomplete")
    handoff = (root / ARTIFACT_FILENAMES["handoff"]).read_bytes()
    known_hosts = (root / ARTIFACT_FILENAMES["known-hosts"]).read_bytes()
    known_hosts_sha256 = sha256_bytes(known_hosts)
    dispatches = journal.ssh_keyscan_dispatch_ordinals
    outcomes = journal.ssh_keyscan_outcome_ordinals
    if (
        len(dispatches) != 2
        or outcomes != (dispatches[0] + 1, dispatches[1] + 1)
        or dispatches[1] != outcomes[0] + 1
        or journal.ssh_keyscan_stdout_sha256s
        != (known_hosts_sha256, known_hosts_sha256)
    ):
        raise ValueError("Prime known-hosts differ from keyscan outcomes")
    summary = validate_handoff_payload(
        handoff,
        authorization=authorization,
        claim_sha256=sha256_bytes((root / ARTIFACT_FILENAMES["claim"]).read_bytes()),
        transcript_sha256=sha256_bytes(
            (root / ARTIFACT_FILENAMES["transcript"]).read_bytes()
        ),
        assessment_sha256=sha256_bytes(
            (root / ARTIFACT_FILENAMES["assessment"]).read_bytes()
        ),
        assessment_envelope_sha256=sha256_bytes(
            (root / ARTIFACT_FILENAMES["assessment-envelope"]).read_bytes()
        ),
        selected_resource_sha256=cast(str, assessment["selected_resource_sha256"]),
        selected_facts=assessment["selected_facts"],
        known_hosts=known_hosts,
        test_script=remote_test_script(
            authorization["commit"], assessment["selected_facts"]
        ),
        authority=expected_authority,
    )
    if summary.pod_identity_sha256 != result.pod_identity_sha256:
        raise ValueError("Prime handoff pod identity differs from create result")
    signature = verify_signed_envelope(
        (root / ARTIFACT_FILENAMES["handoff-envelope"]).read_bytes(),
        handoff,
        expected_namespace,
        identity,
        authority=expected_authority,
        domain=expected_domain,
    )
    if signature != (root / ARTIFACT_FILENAMES["handoff-signature"]).read_bytes():
        raise ValueError("Prime one-shot handoff signature differs")
    return summary


def verify_terminal_evidence(
    root: Path,
    identity: SigningIdentity,
    *,
    binding: RuntimeBinding = V2_RUNTIME_BINDING,
) -> dict[str, Any]:
    if not _is_trusted_binding(binding):
        raise ValueError("Prime one-shot verifier binding is not the canonical singleton")
    expected_authorization_path = binding.authorization_path
    expected_claim_domain = binding.claim_domain
    expected_terminal_domain = binding.terminal_domain
    expected_terminal_namespace = binding.terminal_namespace
    expected_terminal_purpose = binding.terminal_purpose
    expected_claim_authority = dict(binding.claim_authority)
    expected_assessment_domain = binding.assessment_domain
    expected_assessment_schema_version = binding.assessment_schema_version
    expected_assessment_namespace = binding.assessment_namespace
    expected_assessment_authority = dict(binding.assessment_authority)
    expected_assessment_ttl_seconds = binding.assessment_ttl_seconds
    expected_signed_envelope_domain = binding.signed_envelope_domain
    expected_create_authority = dict(binding.create_authority)
    expected_terminal_authority = dict(binding.terminal_authority)
    expected_handoff_namespace = binding.handoff_namespace
    expected_handoff_authority = dict(binding.handoff_authority)
    expected_claim_authority = closed_authority(expected_claim_authority, "claim")
    expected_assessment_authority = closed_authority(expected_assessment_authority, "assessment")
    expected_create_authority = closed_authority(expected_create_authority, "create")
    expected_terminal_authority = closed_authority(
        expected_terminal_authority, "terminal", readiness=True
    )
    expected_handoff_authority = closed_authority(
        expected_handoff_authority, "handoff", readiness=True
    )
    if (
        not _is_trusted_binding(binding)
        or dict(binding.assessment_authority)
        != dict(expected_assessment_authority)
    ):
        raise ValueError("Prime one-shot assessment binding is not canonical")
    terminal = (root / "terminal.json").read_bytes()
    verify_signed_envelope(
        (root / "terminal-envelope.json").read_bytes(),
        terminal,
        expected_terminal_namespace,
        identity,
        authority=expected_terminal_authority,
        domain=expected_signed_envelope_domain,
    )
    value = strict_object(
        terminal,
        {
            "schema_version", "domain", "state", "disposition", "purpose", "monitoring",
            "authorization", "assessment_sha256", "create_dispatched", "tests_passed",
            "cleanup_proven", "primary_failure", "recovery_failures", "cleanup_failures",
            "publication_failures", "evidence_dag", "command_count",
            "prime_cli_call_count", "wallet_api_call_count", "elapsed_seconds",
            "attempt_consumed", "retry", "authority",
        },
        "Prime one-shot terminal",
    )
    state = value["state"]
    elapsed = value["elapsed_seconds"]
    if (
        value["schema_version"] != 2
        or value["domain"] != expected_terminal_domain
        or type(state) is not str
        or state not in _TERMINAL_STATES
        or value["disposition"] != state
        or value["purpose"] != expected_terminal_purpose
        or value["monitoring"] is not False
        or type(value["create_dispatched"]) is not bool
        or type(value["tests_passed"]) is not bool
        or type(value["cleanup_proven"]) is not bool
        or type(elapsed) not in {int, float}
        or not math.isfinite(float(elapsed))
        or not 0 <= float(elapsed) <= MAXIMUM_POD_SECONDS + CLEANUP_TIMEOUT_SECONDS
        or value["attempt_consumed"] is not True
        or value["retry"] is not False
    ):
        raise ValueError("Prime one-shot terminal binding differs")
    authority_value(value["authority"], expected_terminal_authority, "terminal")
    authorization = _authorization_projection(
        value["authorization"], expected_authorization_path=expected_authorization_path
    )
    dag = _bound_dag(root, value["evidence_dag"])
    _claim_projection(
        (root / ARTIFACT_FILENAMES["claim"]).read_bytes(),
        authorization,
        expected_claim_domain=expected_claim_domain,
        expected_authority=expected_claim_authority,
    )

    assessment: dict[str, Any] | None = None
    assessment_raw: bytes | None = None
    if "assessment" in dag:
        assessment_raw = (root / ARTIFACT_FILENAMES["assessment"]).read_bytes()
        assessment = _assessment_projection(
            assessment_raw,
            authorization,
            expected_domain=expected_assessment_domain,
            expected_schema_version=expected_assessment_schema_version,
            expected_authority=expected_assessment_authority,
            expected_ttl_seconds=expected_assessment_ttl_seconds,
        )
        if (
            "assessment-envelope" not in dag
            or value["assessment_sha256"] != sha256_bytes(assessment_raw)
        ):
            raise ValueError("Prime one-shot terminal assessment binding differs")
        verify_signed_envelope(
            (root / ARTIFACT_FILENAMES["assessment-envelope"]).read_bytes(),
            assessment_raw,
            expected_assessment_namespace,
            identity,
            authority=expected_assessment_authority,
            domain=expected_signed_envelope_domain,
        )
    elif value["assessment_sha256"] is not None:
        raise ValueError("Prime one-shot terminal has an unbound assessment")
    if "assessment-envelope" in dag and assessment is None:
        raise ValueError("Prime one-shot terminal has an orphan assessment envelope")
    if "transcript" in dag:
        _transcript_projection(
            (root / ARTIFACT_FILENAMES["transcript"]).read_bytes(),
            assessment_raw,
            assessment,
            authorization,
            binding,
        )
    elif assessment is not None:
        raise ValueError("Prime one-shot assessment lacks a transcript")

    journal = _journal_summary(root, dag)
    wallet_before_value, _wallet_before, wallet_before_requests = (
        _wallet_before_projection(root, dag, journal)
    )
    create_dispatch, create_result = _create_projections(
        root, dag, authorization, assessment, journal, expected_create_authority
    )
    handoff = _handoff_projection(
        root,
        dag,
        authorization,
        assessment,
        create_result,
        identity,
        journal,
        expected_namespace=expected_handoff_namespace,
        expected_authority=expected_handoff_authority,
        expected_domain=expected_signed_envelope_domain,
    )
    cleanup = _cleanup_projection(
        root, dag, wallet_before_value, journal, wallet_before_requests
    )
    primary = value["primary_failure"]
    if not (
        primary is None
        or (type(primary) is str and _FAILURE_NAME.fullmatch(primary) is not None)
    ):
        raise ValueError("Prime one-shot terminal primary failure differs")
    recovery_failures = _failure_list(value["recovery_failures"], "recovery failures")
    terminal_cleanup_failures = _failure_list(value["cleanup_failures"], "cleanup failures")
    publication_failures = _failure_list(value["publication_failures"], "publication failures")
    if cleanup is not None and terminal_cleanup_failures != list(cleanup.errors):
        raise ValueError("Prime one-shot cleanup failure binding differs")
    for actual, expected, maximum, label in (
        (value["command_count"], journal.command_count, 10_000, "command count"),
        (
            value["prime_cli_call_count"], journal.prime_cli_call_count,
            MAX_PRIME_CLI_CALLS, "Prime count",
        ),
        (
            value["wallet_api_call_count"], journal.wallet_api_call_count,
            MAX_WALLET_API_CALLS, "wallet count",
        ),
    ):
        if type(actual) is not int or actual != expected or actual > maximum:
            raise ValueError(f"Prime one-shot terminal {label} differs")
    remote_present, remote_passed = _remote_projection(root, dag, assessment)
    failures = [
        *(item for item in (primary,) if item is not None),
        *recovery_failures,
        *terminal_cleanup_failures,
        *publication_failures,
    ]
    assessment_state = None if assessment is None else assessment["state"]
    create_dispatched = value["create_dispatched"]
    tests_passed = value["tests_passed"]
    cleanup_proven = value["cleanup_proven"]

    if create_dispatched:
        create_consistent = (
            assessment_state == "qualifying_capacity"
            and {"wallet-before", "create-dispatch", "command-journal", "cleanup"} <= set(dag)
            and create_dispatch is not None
            and journal.create_payload_sha256 is not None
            and cleanup is not None
            and cleanup_proven is cleanup.proven
        )
    else:
        create_consistent = (
            create_dispatch is None
            and journal.create_payload_sha256 is None
            and "create-dispatch" not in dag
            and "create-result" not in dag
            and "cleanup" not in dag
            and cleanup is None
            and wallet_before_value is None
            and cleanup_proven is True
        )
    if not create_consistent:
        raise ValueError("Prime one-shot create or cleanup state differs")
    if (
        (create_result is None) is not (journal.create_status_code is None)
        and (state != "failed_terminal" or not failures)
    ):
        raise ValueError("Prime one-shot create result topology differs")
    if handoff is not None and create_result is None:
        raise ValueError("Prime one-shot handoff lacks a create result")
    if create_result is not None:
        pod_hash = create_result.pod_identity_sha256
        if cleanup is None or pod_hash not in cleanup.owned_identity_sha256s:
            raise ValueError("Prime one-shot create pod is absent from cleanup ownership")
        if cleanup.proven and pod_hash not in cleanup.terminated_identity_sha256s:
            raise ValueError("Prime one-shot create pod was not terminated")
        reconciliation = cleanup.wallet_reconciliation
        if reconciliation is not None:
            owned = set(cast(list[str], reconciliation["owned_pod_identity_sha256s"]))
            resources = set(
                cast(list[str], reconciliation["new_resource_identity_sha256s"])
            )
            if pod_hash not in owned or not resources or not resources <= owned:
                raise ValueError("Prime one-shot wallet pod ownership differs")
    if tests_passed is not (remote_present and remote_passed):
        raise ValueError("Prime one-shot remote test result differs")

    if state == "completed":
        valid = (
            set(dag) == set(ARTIFACT_FILENAMES) - _TERMINAL_EXCLUSIONS
            and assessment_state == "qualifying_capacity"
            and create_dispatched is True
            and tests_passed is True
            and cleanup_proven is True
            and not failures
        )
    elif state in {"no_qualifying_capacity", "ambiguous_capacity"}:
        valid = (
            set(dag)
            == {"claim", "transcript", "assessment", "assessment-envelope", "command-records"}
            and assessment_state == state
            and create_dispatched is False
            and tests_passed is False
            and cleanup_proven is True
            and journal.command_count == journal.prime_cli_call_count
            == journal.wallet_api_call_count == 0
            and not failures
        )
    else:
        complete_remote = not tests_passed or (
            remote_present
            and set(dag) >= _HANDOFF_ARTIFACTS
            and {"create-result", "wallet-before", "cleanup"} <= set(dag)
        )
        valid = (
            bool(failures)
            and assessment_state in {None, "qualifying_capacity"}
            and complete_remote
        )
    if not valid:
        raise ValueError("Prime one-shot terminal disposition or state fields differ")
    return value


__all__ = [
    "ARTIFACT_FILENAMES",
    "MAX_TRANSCRIPT_BYTES",
    "artifact_dag",
    "authority_value",
    "closed_authority",
    "signed_envelope",
    "verify_signed_envelope",
    "verify_terminal_evidence",
]
