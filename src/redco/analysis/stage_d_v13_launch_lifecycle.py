"""Small durable owners for provisioning and provider dispatch state.

Provisioning is deliberately separate from the outcome-bearing attempt claim:
one zero-call replacement may continue the same campaign, while the first
provider POST irreversibly closes that replacement choice.  The module is
source-free and is used by the launch runtime and by deterministic tests.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes

SUPPORT_CAP_USD = 12.0
SCIENCE_RESERVE_USD = 16.0
TEARDOWN_RESERVE_USD = 2.0
MAX_PROVISIONING_ATTEMPTS = 2
PROVISIONING_LEDGER_ENV = "REDCO_STAGE_D_PROVISIONING_LEDGER"
DISPATCH_JOURNAL_SUFFIX = ".dispatch.jsonl"
HANDOFF_V2_DOMAIN = "redco-stage-d1-support-v13-execute-handoff-v2"
HANDOFF_V2_SCHEMA_VERSION = 2
HANDOFF_V2_NAMESPACE = "redco-stage-d1-support-v13-execute-handoff-v2-signing"
HANDOFF_V2_EXPIRY_SECONDS = 900
HANDOFF_V2_LEDGER_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-launch/provisioning-ledger-v1.json"
)
HANDOFF_V2_RESOURCE_KEYS = frozenset(
    {
        "resource_id",
        "provider",
        "location",
        "gpu_type",
        "gpu_count",
        "memory_gb",
        "is_spot",
        "security",
    }
)
HANDOFF_V2_SSH_KEYS = frozenset({"user", "host", "port"})
HANDOFF_V2_AUTHORITY_KEYS = frozenset(
    {
        "support_only",
        "provider_calls_authorized",
        "model_calls_authorized",
        "science_authorized",
        "training_authorized",
        "heldout_evaluation_authorized",
        "scientific_transition_authorized",
        "prime_gpu_scientific_launch_authorized",
    }
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_DISPATCH_SYNC_LOCK = threading.RLock()

_T = TypeVar("_T")


def _strict_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise RuntimeError(f"execute handoff {label} is not a nonempty string")
    return value


def _strict_integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"execute handoff {label} is not an integer")
    return value


def _strict_digest(value: object, label: str, *, length: int = 64) -> str:
    result = _strict_string(value, label)
    pattern = _HEX64 if length == 64 else _HEX40
    if not pattern.fullmatch(result):
        raise RuntimeError(f"execute handoff {label} is not a lowercase hex digest")
    return result


@dataclass(frozen=True, slots=True)
class SigningIdentity:
    """Public-only OpenSSH identity bound by the launch authorization."""

    public_key_type: str
    public_key_base64: str
    fingerprint_sha256: str
    principal: str
    namespace: str
    allowed_signers_sha256: str

    @property
    def public_key_line(self) -> bytes:
        return f"{self.public_key_type} {self.public_key_base64}\n".encode("ascii")

    @property
    def allowed_signers_bytes(self) -> bytes:
        return (
            f"{self.principal} {self.public_key_type} {self.public_key_base64}\n"
        ).encode("ascii")

    @classmethod
    def from_payload(cls, value: object) -> SigningIdentity:
        if not isinstance(value, dict):
            raise RuntimeError("launch signing identity is not an object")
        expected = {
            "public_key_type",
            "public_key_base64",
            "fingerprint_sha256",
            "principal",
            "namespace",
            "allowed_signers_sha256",
        }
        if set(value) != expected:
            raise RuntimeError("launch signing identity fields differ")
        identity = cls(
            public_key_type=_strict_string(value["public_key_type"], "public_key_type"),
            public_key_base64=_strict_string(value["public_key_base64"], "public_key_base64"),
            fingerprint_sha256=_strict_string(
                value["fingerprint_sha256"], "fingerprint_sha256"
            ),
            principal=_strict_string(value["principal"], "principal"),
            namespace=_strict_string(value["namespace"], "namespace"),
            allowed_signers_sha256=_strict_digest(
                value["allowed_signers_sha256"], "allowed_signers_sha256"
            ),
        )
        if identity.namespace != HANDOFF_V2_NAMESPACE:
            raise RuntimeError("launch signing namespace differs")
        if hashlib.sha256(identity.allowed_signers_bytes).hexdigest() != (
            identity.allowed_signers_sha256
        ):
            raise RuntimeError("allowed-signers binding differs")
        if " " in identity.public_key_type or " " in identity.public_key_base64:
            raise RuntimeError("launch public key encoding is not canonical")
        try:
            base64.b64decode(identity.public_key_base64, validate=True)
        except (ValueError, binascii.Error) as error:
            raise RuntimeError("launch public key is not base64") from error
        return identity

    def to_payload(self) -> dict[str, str]:
        return {
            "public_key_type": self.public_key_type,
            "public_key_base64": self.public_key_base64,
            "fingerprint_sha256": self.fingerprint_sha256,
            "principal": self.principal,
            "namespace": self.namespace,
            "allowed_signers_sha256": self.allowed_signers_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExecuteHandoffV2:
    """The single lossless schema shared by issuer, verifier, and claim owner."""

    domain: str
    schema_version: int
    state: str
    campaign_id: str
    outcome_attempt_id: str
    provisioning_ordinal: int
    bundle_commit: str
    bundle_tree: str
    launch_authorization_sha256: str
    frozen_support_protocol_sha256: str
    prime_observation_sha256: str
    resource_identity: dict[str, Any]
    resource_price_usd: float
    pod_id: str
    pod_name: str
    pod_status_sha256: str
    ssh: dict[str, Any]
    known_hosts_sha256: str
    known_hosts_fingerprints: tuple[str, ...]
    ledger_path: str
    ledger_sha256: str
    ledger_state: str
    signer_principal: str
    signer_namespace: str
    signer_fingerprint: str
    nonce: str
    issued_at_epoch: int
    expires_at_epoch: int
    authority: dict[str, bool]

    @classmethod
    def from_payload(cls, value: object) -> ExecuteHandoffV2:
        if not isinstance(value, dict):
            raise RuntimeError("execute handoff v2 is not an object")
        expected = {
            "authority",
            "bundle_commit",
            "bundle_tree",
            "campaign_id",
            "domain",
            "expires_at_epoch",
            "frozen_support_protocol_sha256",
            "issued_at_epoch",
            "known_hosts_fingerprints",
            "known_hosts_sha256",
            "launch_authorization_sha256",
            "ledger_path",
            "ledger_sha256",
            "ledger_state",
            "nonce",
            "outcome_attempt_id",
            "pod_id",
            "pod_name",
            "pod_status_sha256",
            "prime_observation_sha256",
            "provisioning_ordinal",
            "resource_identity",
            "resource_price_usd",
            "schema_version",
            "signer_fingerprint",
            "signer_namespace",
            "signer_principal",
            "ssh",
            "state",
        }
        if set(value) != expected:
            raise RuntimeError("execute handoff v2 fields differ")
        identity = value["resource_identity"]
        if not isinstance(identity, dict) or set(identity) != HANDOFF_V2_RESOURCE_KEYS:
            raise RuntimeError("execute handoff resource identity fields differ")
        if (
            type(identity["resource_id"]) is not str
            or type(identity["provider"]) is not str
            or type(identity["location"]) is not str
            or type(identity["gpu_type"]) is not str
            or type(identity["gpu_count"]) is not int
            or type(identity["memory_gb"]) not in {int, float}
            or type(identity["is_spot"]) is not bool
            or type(identity["security"]) not in {str, dict}
        ):
            raise RuntimeError("execute handoff resource identity types differ")
        ssh = value["ssh"]
        if not isinstance(ssh, dict) or set(ssh) != HANDOFF_V2_SSH_KEYS:
            raise RuntimeError("execute handoff SSH fields differ")
        if (
            type(ssh["user"]) is not str
            or type(ssh["host"]) is not str
            or type(ssh["port"]) is not int
            or not ssh["host"]
            or not 1 <= ssh["port"] <= 65535
        ):
            raise RuntimeError("execute handoff SSH types differ")
        fingerprints = value["known_hosts_fingerprints"]
        if (
            not isinstance(fingerprints, list)
            or not fingerprints
            or any(type(item) is not str or not item for item in fingerprints)
            or fingerprints != sorted(set(fingerprints))
            or any(not item.startswith("SHA256:") for item in fingerprints)
        ):
            raise RuntimeError("execute handoff known-host fingerprints differ")
        authority = value["authority"]
        if not isinstance(authority, dict) or set(authority) != HANDOFF_V2_AUTHORITY_KEYS:
            raise RuntimeError("execute handoff authority fields differ")
        if any(type(item) is not bool for item in authority.values()):
            raise RuntimeError("execute handoff authority types differ")
        expected_authority = {
            "support_only": True,
            "provider_calls_authorized": True,
            "model_calls_authorized": True,
            "science_authorized": False,
            "training_authorized": False,
            "heldout_evaluation_authorized": False,
            "scientific_transition_authorized": False,
            "prime_gpu_scientific_launch_authorized": False,
        }
        if authority != expected_authority:
            raise RuntimeError("execute handoff authority is outside support-only scope")
        if (
            value["domain"] != HANDOFF_V2_DOMAIN
            or value["schema_version"] != HANDOFF_V2_SCHEMA_VERSION
        ):
            raise RuntimeError("execute handoff v2 identity differs")
        if value["state"] not in {"issued", "consumed"}:
            raise RuntimeError("execute handoff v2 state differs")
        if (
            type(value["provisioning_ordinal"]) is not int
            or value["provisioning_ordinal"] not in {1, 2}
        ):
            raise RuntimeError("execute handoff provisioning ordinal differs")
        for field in (
            "campaign_id",
            "outcome_attempt_id",
            "pod_id",
            "pod_name",
            "ledger_path",
            "ledger_state",
            "signer_principal",
            "signer_namespace",
            "signer_fingerprint",
            "nonce",
        ):
            _strict_string(value[field], field)
        _strict_digest(value["bundle_commit"], "bundle_commit", length=40)
        _strict_digest(value["bundle_tree"], "bundle_tree", length=40)
        for field in (
            "launch_authorization_sha256",
            "frozen_support_protocol_sha256",
            "prime_observation_sha256",
            "pod_status_sha256",
            "known_hosts_sha256",
            "ledger_sha256",
        ):
            _strict_digest(value[field], field)
        if not _HEX64.fullmatch(cast(str, value["nonce"])):
            raise RuntimeError("execute handoff nonce is not 256-bit hex")
        if (
            type(value["resource_price_usd"]) not in {int, float}
            or type(value["resource_price_usd"]) is bool
        ):
            raise RuntimeError("execute handoff resource price type differs")
        issued = _strict_integer(value["issued_at_epoch"], "issued_at_epoch")
        expires = _strict_integer(value["expires_at_epoch"], "expires_at_epoch")
        if expires != issued + HANDOFF_V2_EXPIRY_SECONDS or expires <= issued:
            raise RuntimeError("execute handoff expiry differs")
        if cast(str, value["ledger_state"]) != "provisioned":
            raise RuntimeError("execute handoff ledger state differs")
        if cast(str, value["ledger_path"]) != HANDOFF_V2_LEDGER_RELATIVE:
            raise RuntimeError("execute handoff ledger path differs")
        return cls(
            domain=cast(str, value["domain"]),
            schema_version=cast(int, value["schema_version"]),
            state=cast(str, value["state"]),
            campaign_id=cast(str, value["campaign_id"]),
            outcome_attempt_id=cast(str, value["outcome_attempt_id"]),
            provisioning_ordinal=value["provisioning_ordinal"],
            bundle_commit=cast(str, value["bundle_commit"]),
            bundle_tree=cast(str, value["bundle_tree"]),
            launch_authorization_sha256=cast(str, value["launch_authorization_sha256"]),
            frozen_support_protocol_sha256=cast(str, value["frozen_support_protocol_sha256"]),
            prime_observation_sha256=cast(str, value["prime_observation_sha256"]),
            resource_identity=cast(dict[str, Any], identity),
            resource_price_usd=float(value["resource_price_usd"]),
            pod_id=cast(str, value["pod_id"]),
            pod_name=cast(str, value["pod_name"]),
            pod_status_sha256=cast(str, value["pod_status_sha256"]),
            ssh=cast(dict[str, Any], ssh),
            known_hosts_sha256=cast(str, value["known_hosts_sha256"]),
            known_hosts_fingerprints=tuple(cast(list[str], fingerprints)),
            ledger_path=cast(str, value["ledger_path"]),
            ledger_sha256=cast(str, value["ledger_sha256"]),
            ledger_state=cast(str, value["ledger_state"]),
            signer_principal=cast(str, value["signer_principal"]),
            signer_namespace=cast(str, value["signer_namespace"]),
            signer_fingerprint=cast(str, value["signer_fingerprint"]),
            nonce=cast(str, value["nonce"]),
            issued_at_epoch=issued,
            expires_at_epoch=expires,
            authority=cast(dict[str, bool], authority),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "authority": dict(self.authority),
            "bundle_commit": self.bundle_commit,
            "bundle_tree": self.bundle_tree,
            "campaign_id": self.campaign_id,
            "domain": self.domain,
            "expires_at_epoch": self.expires_at_epoch,
            "frozen_support_protocol_sha256": self.frozen_support_protocol_sha256,
            "issued_at_epoch": self.issued_at_epoch,
            "known_hosts_fingerprints": list(self.known_hosts_fingerprints),
            "known_hosts_sha256": self.known_hosts_sha256,
            "launch_authorization_sha256": self.launch_authorization_sha256,
            "ledger_path": self.ledger_path,
            "ledger_sha256": self.ledger_sha256,
            "ledger_state": self.ledger_state,
            "nonce": self.nonce,
            "outcome_attempt_id": self.outcome_attempt_id,
            "pod_id": self.pod_id,
            "pod_name": self.pod_name,
            "pod_status_sha256": self.pod_status_sha256,
            "prime_observation_sha256": self.prime_observation_sha256,
            "provisioning_ordinal": self.provisioning_ordinal,
            "resource_identity": dict(self.resource_identity),
            "resource_price_usd": self.resource_price_usd,
            "schema_version": self.schema_version,
            "signer_fingerprint": self.signer_fingerprint,
            "signer_namespace": self.signer_namespace,
            "signer_principal": self.signer_principal,
            "ssh": dict(self.ssh),
            "state": self.state,
        }


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
            descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_parent(path)


def _replace(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("lifecycle temporary path already exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("provisioning ledger is unreadable") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != path.read_bytes():
        raise RuntimeError("provisioning ledger is not canonical")
    expected = {
        "schema_version",
        "domain",
        "campaign_id",
        "outcome_attempt_limit",
        "provisioning_attempt_limit",
        "provider_post_observed",
        "provider_post_count",
        "provider_operation_ids",
        "evidence_observation_count",
        "closed",
        "cumulative_cost_usd",
        "provisions",
        "replacement_rule",
        "dispatch_journal_path",
        "billing",
    }
    if (
        set(value) != expected
        or value["schema_version"] != 2
        or value["domain"] != "redco-stage-d1-support-v13-provisioning-ledger-v1"
        or type(value["provider_post_observed"]) is not bool
        or type(value["closed"]) is not bool
        or type(value["provider_post_count"]) is not int
        or not isinstance(value["provider_operation_ids"], list)
        or value["provider_post_count"] != len(value["provider_operation_ids"])
        or len(set(value["provider_operation_ids"]))
        != len(value["provider_operation_ids"])
        or any(
            type(item) is not str or not item
            for item in value["provider_operation_ids"]
        )
        or value["dispatch_journal_path"] != path.name + DISPATCH_JOURNAL_SUFFIX
        or not isinstance(value["billing"], dict)
    ):
        raise RuntimeError("provisioning ledger state is malformed")
    return cast(dict[str, Any], value)


@dataclass(slots=True)
class ProvisioningLedger:
    """Durable campaign provisioning and irreversible dispatch state."""

    path: Path
    campaign_id: str

    @classmethod
    def create(
        cls,
        path: Path,
        campaign_id: str,
        *,
        wallet_before: Mapping[str, Any] | None = None,
    ) -> ProvisioningLedger:
        if not campaign_id or path.exists() or path.is_symlink():
            raise RuntimeError("provisioning ledger already exists or has no campaign identity")
        ledger = cls(path.resolve(), campaign_id)
        _write_once(
            ledger.path,
            ledger._bytes(
                (),
                provider_post=False,
                provider_operations=(),
                evidence=0,
                closed=False,
                billing={
                    "status": "unreconciled",
                    "wallet_before": None if wallet_before is None else dict(wallet_before),
                    "wallet_after": None,
                    "delta_usd": None,
                },
            ),
        )
        return ledger

    @classmethod
    def open(cls, path: Path) -> ProvisioningLedger:
        """Open an existing ledger using only its durable campaign identity."""

        resolved = path.resolve()
        state = _load(resolved)
        campaign_id = state.get("campaign_id")
        if not isinstance(campaign_id, str) or not campaign_id:
            raise RuntimeError("provisioning ledger lacks its durable campaign identity")
        return cls(resolved, campaign_id)

    def _payload(
        self,
        provisions: tuple[dict[str, Any], ...],
        *,
        provider_post: bool,
        provider_operations: tuple[str, ...],
        evidence: int,
        closed: bool,
        billing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        cumulative = sum(
            float(item["billed_cost_usd"])
            for item in provisions
            if type(item.get("billed_cost_usd")) in {int, float}
        )
        return {
            "schema_version": 2,
            "domain": "redco-stage-d1-support-v13-provisioning-ledger-v1",
            "campaign_id": self.campaign_id,
            "outcome_attempt_limit": 1,
            "provisioning_attempt_limit": MAX_PROVISIONING_ATTEMPTS,
            "provider_post_observed": provider_post,
            "provider_post_count": len(provider_operations),
            "provider_operation_ids": list(provider_operations),
            "evidence_observation_count": evidence,
            "closed": closed,
            "cumulative_cost_usd": cumulative,
            "provisions": list(provisions),
            "dispatch_journal_path": self.path.name + DISPATCH_JOURNAL_SUFFIX,
            "billing": dict(
                billing
                or {
                    "status": "unreconciled",
                    "wallet_before": None,
                    "wallet_after": None,
                    "delta_usd": None,
                }
            ),
            "replacement_rule": {
                "same_campaign": True,
                "only_before_provider_post_or_evidence": True,
                "max_attempts": MAX_PROVISIONING_ATTEMPTS,
                "cap_usd": SUPPORT_CAP_USD,
            },
        }

    def _bytes(
        self,
        provisions: tuple[dict[str, Any], ...],
        *,
        provider_post: bool,
        provider_operations: tuple[str, ...],
        evidence: int,
        closed: bool,
        billing: Mapping[str, Any] | None = None,
    ) -> bytes:
        return cast(
            bytes,
            canonical_json_bytes(
            self._payload(
                provisions,
                provider_post=provider_post,
                provider_operations=provider_operations,
                evidence=evidence,
                closed=closed,
                billing=billing,
            )
            ),
        )

    def _state(self) -> dict[str, Any]:
        state = _load(self.path)
        if state.get("campaign_id") != self.campaign_id:
            raise RuntimeError("provisioning ledger campaign identity differs")
        operations = self._journal_operations()
        if operations != tuple(cast(list[str], state["provider_operation_ids"])):
            state["provider_operation_ids"] = list(operations)
            state["provider_post_count"] = len(operations)
            state["provider_post_observed"] = bool(operations)
        return state

    def _journal_path(self) -> Path:
        return self.path.with_name(self.path.name + DISPATCH_JOURNAL_SUFFIX)

    def _journal_operations(self) -> tuple[str, ...]:
        journal = self._journal_path()
        if not journal.exists():
            return ()
        operations: list[str] = []
        for line in journal.read_bytes().splitlines():
            if not line:
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("provider dispatch journal is malformed") from error
            if (
                not isinstance(value, dict)
                or set(value)
                != {
                    "schema_version",
                    "domain",
                    "campaign_id",
                    "operation_id",
                    "request_sha256",
                }
                or value["schema_version"] != 1
                or value["domain"]
                != "redco-stage-d1-support-v13-provider-dispatch-v1"
                or value["campaign_id"] != self.campaign_id
            ):
                raise RuntimeError("provider dispatch journal identity differs")
            operation_id = value["operation_id"]
            if not isinstance(operation_id, str) or not operation_id or operation_id in operations:
                raise RuntimeError("provider dispatch journal contains duplicate identity")
            if not isinstance(value["request_sha256"], str) or len(value["request_sha256"]) != 64:
                raise RuntimeError("provider dispatch journal request hash is invalid")
            operations.append(operation_id)
        return tuple(operations)

    def _append_dispatch(self, operation_id: str, request_sha256: str = "") -> None:
        value = canonical_json_bytes(
            {
                "schema_version": 1,
                "domain": "redco-stage-d1-support-v13-provider-dispatch-v1",
                "campaign_id": self.campaign_id,
                "operation_id": operation_id,
                "request_sha256": request_sha256 or "0" * 64,
            }
        ) + b"\n"
        journal = self._journal_path()
        journal.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(journal, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, value)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_parent(journal)

    def record_provision(
        self,
        *,
        provision_id: str,
        resource_id: str,
        cost_usd: int | float | None = None,
        billing_cursor: str,
        pod_id: str | None = None,
    ) -> dict[str, Any]:
        if not provision_id or not resource_id or not billing_cursor:
            raise ValueError("provisioning requires durable identities")
        if cost_usd is not None and (
            type(cost_usd) not in {int, float}
            or not math.isfinite(float(cost_usd))
            or cost_usd < 0
        ):
            raise ValueError("uncertain or invalid provisioning cost")
        state = self._state()
        if (
            state["closed"]
            or state["provider_post_observed"]
            or state["evidence_observation_count"]
        ):
            raise RuntimeError("provisioning cannot continue after outcome evidence")
        provisions = tuple(cast(list[dict[str, Any]], state["provisions"]))
        if len(provisions) >= MAX_PROVISIONING_ATTEMPTS:
            raise RuntimeError("provisioning attempt limit exhausted")
        if (
            cost_usd is not None
            and float(state["cumulative_cost_usd"]) + float(cost_usd)
            > SUPPORT_CAP_USD
        ):
            raise RuntimeError("cumulative provisioning cost exceeds support cap")
        item = {
            "attempt": len(provisions) + 1,
            "provision_id": provision_id,
            "resource_id": resource_id,
            "pod_id": pod_id,
            "provider_provision_id": None,
            "state": "provisioned" if pod_id is not None else "claimed",
            "billed_cost_usd": None if cost_usd is None else float(cost_usd),
            "billing_status": "reconciled_test" if cost_usd is not None else "pending",
            "billing_cursor": billing_cursor,
            "provider_post_observed": False,
        }
        updated = (*provisions, item)
        billing = dict(cast(dict[str, Any], state["billing"]))
        if cost_usd is not None and billing.get("wallet_before") is None:
            billing["status"] = "reconciled_test"
        _replace(
            self.path,
            self._bytes(
                updated,
                provider_post=False,
                provider_operations=(),
                evidence=0,
                closed=False,
                billing=billing,
            ),
        )
        return item

    def bind_provision(self, provision_id: str, pod_id: str) -> None:
        """Bind the real Prime pod identity to the pre-create attempt record."""

        if not provision_id or not pod_id:
            raise ValueError("provision identity is required")
        with _DISPATCH_SYNC_LOCK:
            state = self._state()
            provisions = tuple(cast(list[dict[str, Any]], state["provisions"]))
            if not provisions or provisions[-1].get("provision_id") != provision_id:
                raise RuntimeError("provision identity is not the current campaign attempt")
            if provisions[-1].get("pod_id") not in {None, pod_id}:
                raise RuntimeError("provision identity was already bound differently")
            latest = {
                **provisions[-1],
                "pod_id": pod_id,
                "provider_provision_id": pod_id,
                "state": "provisioned",
            }
            updated = (*provisions[:-1], latest)
            _replace(
                self.path,
                self._bytes(
                    updated,
                    provider_post=bool(state["provider_post_observed"]),
                    provider_operations=tuple(self._journal_operations()),
                    evidence=int(state["evidence_observation_count"]),
                    closed=bool(state["closed"]),
                    billing=state["billing"],
                ),
            )

    def reconcile_billing(self, wallet_after: Mapping[str, Any]) -> float:
        state = self._state()
        billing = cast(dict[str, Any], state["billing"])
        before = billing.get("wallet_before")
        if not isinstance(before, dict):
            raise RuntimeError("billing reconciliation lacks the pre-provision wallet")
        before_id = before.get("wallet_id", before.get("account_id"))
        after_id = wallet_after.get("wallet_id", wallet_after.get("account_id"))
        if before_id != after_id or before.get("team_id") != wallet_after.get("team_id"):
            raise RuntimeError("billing wallet identity changed")
        before_balance = before.get("wallet_usd", before.get("balance_usd"))
        after_balance = wallet_after.get("wallet_usd", wallet_after.get("balance_usd"))
        if type(before_balance) not in {int, float} or type(after_balance) not in {int, float}:
            raise RuntimeError("billing balance observations are missing")
        before_balance_value = cast(int | float, before_balance)
        after_balance_value = cast(int | float, after_balance)
        delta = round(float(before_balance_value) - float(after_balance_value), 6)
        if (
            delta < 0
            or delta > SUPPORT_CAP_USD
            or float(after_balance_value) < SCIENCE_RESERVE_USD + TEARDOWN_RESERVE_USD
        ):
            raise RuntimeError("billing delta is outside the support and reserve bounds")
        before_rows = {
            str(row.get("id"))
            for row in cast(list[Any], before.get("recent_billings", []))
            if isinstance(row, dict)
        }
        after_rows = [
            row
            for row in cast(list[Any], wallet_after.get("recent_billings", []))
            if isinstance(row, dict)
        ]
        new_rows = [row for row in after_rows if str(row.get("id")) not in before_rows]
        row_total_value = 0.0
        for row in new_rows:
            amount = row.get("amount_usd")
            if type(amount) in {int, float}:
                row_total_value += float(cast(int | float, amount))
        row_total = round(row_total_value, 6)
        if abs(row_total - delta) > 0.000001 and not (delta == 0 and row_total == 0):
            raise RuntimeError("wallet delta does not equal the authenticated billing rows")
        provisions = tuple(cast(list[dict[str, Any]], state["provisions"]))
        if provisions:
            latest = {**provisions[-1], "billed_cost_usd": delta, "billing_status": "reconciled"}
            provisions = (*provisions[:-1], latest)
        updated_billing = {
            "status": "reconciled",
            "wallet_before": dict(before),
            "wallet_after": dict(wallet_after),
            "delta_usd": delta,
            "new_billing_ids": sorted(str(row.get("id")) for row in new_rows),
        }
        _replace(
            self.path,
            self._bytes(
                provisions,
                provider_post=bool(state["provider_post_observed"]),
                provider_operations=tuple(self._journal_operations()),
                evidence=int(state["evidence_observation_count"]),
                closed=bool(state["closed"]),
                billing=updated_billing,
            ),
        )
        return delta

    def record_provider_post(
        self,
        *,
        operation_id: str,
        request_sha256: str = "",
    ) -> None:
        if not operation_id:
            raise ValueError("provider operation needs an identity")
        with _DISPATCH_SYNC_LOCK:
            state = self._state()
            if state["closed"]:
                raise RuntimeError("provider dispatch is closed")
            provisions = tuple(cast(list[dict[str, Any]], state["provisions"]))
            if not provisions:
                raise RuntimeError("provider dispatch requires a provisioned runtime")
            operations = self._journal_operations()
            if operation_id in operations:
                raise RuntimeError("provider operation was already observed")
            self._append_dispatch(operation_id, request_sha256)
            updated_operations = self._journal_operations()
            updated = tuple(
                {
                    **item,
                    "provider_post_observed": index == len(provisions) - 1,
                    **(
                        {"first_provider_operation_id": operation_id}
                        if index == 0
                        else {}
                    ),
                }
                if index == len(provisions) - 1
                else item
                for index, item in enumerate(provisions)
            )
            _replace(
                self.path,
                self._bytes(
                    updated,
                    provider_post=True,
                    provider_operations=updated_operations,
                    evidence=int(state["evidence_observation_count"]),
                    closed=False,
                    billing=state["billing"],
                ),
            )

    def record_evidence(self) -> None:
        state = self._state()
        if state["closed"]:
            raise RuntimeError("evidence cannot be recorded after terminal closure")
        _replace(
            self.path,
            self._bytes(
                tuple(cast(list[dict[str, Any]], state["provisions"])),
                provider_post=bool(state["provider_post_observed"]),
                provider_operations=tuple(
                    cast(list[str], state.get("provider_operation_ids", []))
                ),
                evidence=int(state["evidence_observation_count"]) + 1,
                closed=False,
                billing=state["billing"],
            ),
        )

    def close(self) -> None:
        state = self._state()
        if state["closed"]:
            # Recovery may replace the local file with the already-closed copy
            # returned by the remote owner.  Closing that authenticated state is
            # an idempotent handoff, not a second terminal disposition.
            return
        _replace(
            self.path,
            self._bytes(
                tuple(cast(list[dict[str, Any]], state["provisions"])),
                provider_post=bool(state["provider_post_observed"]),
                provider_operations=tuple(
                    cast(list[str], state.get("provider_operation_ids", []))
                ),
                evidence=int(state["evidence_observation_count"]),
                closed=True,
                billing=state["billing"],
            ),
        )

    def replacement_allowed(self) -> bool:
        state = self._state()
        return (
            not state["closed"]
            and not state["provider_post_observed"]
            and int(state["evidence_observation_count"]) == 0
            and state["billing"].get("status") in {"reconciled", "reconciled_test"}
            and len(state["provisions"]) < MAX_PROVISIONING_ATTEMPTS
        )

    def has_provider_post(self) -> bool:
        return bool(self._state()["provider_post_observed"])


@dataclass(slots=True)
class ProviderDispatchBoundary:
    """Call the irreversible ledger marker immediately before the POST."""

    ledger: ProvisioningLedger

    def send(self, operation_id: str, post: Callable[[], _T]) -> _T:
        self.ledger.record_provider_post(operation_id=operation_id)
        return post()


def record_provider_post_from_path(path: Path, request: bytes) -> None:
    """Record a provider POST from the actual renderer/client boundary.

    The child source/branch process opens the parent-owned ledger itself.  The
    operation ID is derived from the exact prepared request, so the durable
    marker is written immediately before the network owner sends that request.
    """

    if type(request) is not bytes or not request:
        raise ValueError("provider dispatch requires the exact prepared request bytes")
    operation_id = hashlib.sha256(
        b"redco-stage-d1-support-provider-operation-v1\0" + request
    ).hexdigest()
    ledger = ProvisioningLedger.open(path)
    ledger.record_provider_post(
        operation_id=operation_id,
        request_sha256=hashlib.sha256(request).hexdigest(),
    )


def _ssh_keygen(argv: list[str], *, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["ssh-keygen", *argv],
            input=b"" if input_bytes is None else input_bytes,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("required OpenSSH ssh-keygen operation failed") from error
    return bytes(result.stdout)


def _public_key_from_private(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("signing private key is missing")
    raw = _ssh_keygen(["-y", "-f", str(path)]).decode("ascii").strip()
    fields = raw.split()
    if len(fields) < 2 or not fields[0] or not fields[1]:
        raise RuntimeError("signing private key did not produce a public key")
    return fields[0], fields[1]


def _public_key_from_file(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("signing public key file is missing")
    fields = path.read_text(encoding="ascii").strip().split()
    if len(fields) < 2 or not fields[0] or not fields[1]:
        raise RuntimeError("signing public key file is malformed")
    return fields[0], fields[1]


def _public_fingerprint(public_key_type: str, public_key_base64: str) -> str:
    output = _ssh_keygen(
        ["-lf", "-", "-E", "sha256"],
        input_bytes=f"{public_key_type} {public_key_base64}\n".encode("ascii"),
    ).decode("ascii").strip()
    fields = output.split()
    if len(fields) < 2 or not fields[1].startswith("SHA256:"):
        raise RuntimeError("signing public key fingerprint is unavailable")
    return fields[1]


def validate_signing_key(path: Path, identity: SigningIdentity) -> None:
    public_path = path.with_name(path.name + ".pub")
    public_type, public_base64 = _public_key_from_file(public_path)
    key_type, key_base64 = _public_key_from_private(path)
    if (public_type, public_base64) != (key_type, key_base64):
        raise RuntimeError("signing public key file differs from private key")
    if (key_type, key_base64) != (identity.public_key_type, identity.public_key_base64):
        raise RuntimeError("operator signing key does not match launch authorization")
    if _public_fingerprint(key_type, key_base64) != identity.fingerprint_sha256:
        raise RuntimeError("operator signing fingerprint differs")
    challenge = b"redco-stage-d1-support-v13-execute-handoff-v2-signing-challenge"
    signature = _sign_bytes(path, identity.namespace, challenge)
    _verify_bytes(identity, challenge, signature)


def _sign_bytes(path: Path, namespace: str, value: bytes) -> bytes:
    with tempfile.TemporaryDirectory(prefix="redco-handoff-sign-") as directory:
        payload_path = Path(directory) / "payload"
        payload_path.write_bytes(value)
        _ssh_keygen(
            ["-Y", "sign", "-f", str(path), "-n", namespace, str(payload_path)]
        )
        signature_path = payload_path.with_name(payload_path.name + ".sig")
        if not signature_path.is_file() or signature_path.is_symlink():
            raise RuntimeError("ssh-keygen did not produce a detached signature")
        return signature_path.read_bytes()


def _verify_bytes(identity: SigningIdentity, value: bytes, signature: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="redco-handoff-verify-") as directory:
        directory_path = Path(directory)
        payload_path = directory_path / "payload"
        signature_path = directory_path / "payload.sig"
        signers_path = directory_path / "allowed_signers"
        payload_path.write_bytes(value)
        signature_path.write_bytes(signature)
        signers_path.write_bytes(identity.allowed_signers_bytes)
        _ssh_keygen(
            [
                "-Y",
                "verify",
                "-f",
                str(signers_path),
                "-I",
                identity.principal,
                "-n",
                identity.namespace,
                "-s",
                str(signature_path),
            ],
            input_bytes=value,
        )


def issue_execute_handoff_v2(
    path: Path,
    signature_path: Path,
    *,
    bundle: Mapping[str, Any],
    launch_authorization_sha256: str,
    frozen_support_protocol_sha256: str,
    prime_observation_sha256: str,
    resource_identity: Mapping[str, Any],
    resource_price_usd: float,
    pod_id: str,
    pod_name: str,
    pod_status_sha256: str,
    ssh: Mapping[str, Any],
    known_hosts_sha256: str,
    known_hosts_fingerprints: tuple[str, ...],
    ledger: ProvisioningLedger,
    signing_key: Path,
    signer: SigningIdentity,
    provisioning_ordinal: int,
    now_epoch: int | None = None,
) -> bytes:
    """Issue and sign the one-use handoff after all pre-provision bindings."""

    if set(bundle) != {"commit", "tree"}:
        raise RuntimeError("execute handoff bundle fields differ")
    commit = _strict_digest(bundle["commit"], "bundle_commit", length=40)
    tree = _strict_digest(bundle["tree"], "bundle_tree", length=40)
    state = ledger._state()
    provisions = cast(list[dict[str, Any]], state["provisions"])
    if (
        provisioning_ordinal not in {1, 2}
        or not provisions
        or provisions[-1].get("pod_id") != pod_id
        or type(resource_price_usd) not in {int, float}
        or type(resource_price_usd) is bool
    ):
        raise RuntimeError("execute handoff requires the bound provisioning state")
    normalized_resource = dict(resource_identity)
    if set(normalized_resource) != HANDOFF_V2_RESOURCE_KEYS:
        raise RuntimeError("execute handoff resource identity fields differ")
    issued = int(time.time()) if now_epoch is None else now_epoch
    payload = ExecuteHandoffV2(
        domain=HANDOFF_V2_DOMAIN,
        schema_version=HANDOFF_V2_SCHEMA_VERSION,
        state="issued",
        campaign_id=ledger.campaign_id,
        outcome_attempt_id=f"{ledger.campaign_id}:outcome-attempt-1",
        provisioning_ordinal=provisioning_ordinal,
        bundle_commit=commit,
        bundle_tree=tree,
        launch_authorization_sha256=_strict_digest(
            launch_authorization_sha256, "launch_authorization_sha256"
        ),
        frozen_support_protocol_sha256=_strict_digest(
            frozen_support_protocol_sha256, "frozen_support_protocol_sha256"
        ),
        prime_observation_sha256=_strict_digest(
            prime_observation_sha256, "prime_observation_sha256"
        ),
        resource_identity=normalized_resource,
        resource_price_usd=float(resource_price_usd),
        pod_id=_strict_string(pod_id, "pod_id"),
        pod_name=_strict_string(pod_name, "pod_name"),
        pod_status_sha256=_strict_digest(pod_status_sha256, "pod_status_sha256"),
        ssh=dict(ssh),
        known_hosts_sha256=_strict_digest(known_hosts_sha256, "known_hosts_sha256"),
        known_hosts_fingerprints=tuple(known_hosts_fingerprints),
        ledger_path=HANDOFF_V2_LEDGER_RELATIVE,
        ledger_sha256=hashlib.sha256(ledger.path.read_bytes()).hexdigest(),
        ledger_state="provisioned",
        signer_principal=signer.principal,
        signer_namespace=signer.namespace,
        signer_fingerprint=signer.fingerprint_sha256,
        nonce=secrets.token_hex(32),
        issued_at_epoch=issued,
        expires_at_epoch=issued + HANDOFF_V2_EXPIRY_SECONDS,
        authority={
            "support_only": True,
            "provider_calls_authorized": True,
            "model_calls_authorized": True,
            "science_authorized": False,
            "training_authorized": False,
            "heldout_evaluation_authorized": False,
            "scientific_transition_authorized": False,
            "prime_gpu_scientific_launch_authorized": False,
        },
    )
    canonical = canonical_json_bytes(payload.to_payload())
    ExecuteHandoffV2.from_payload(json.loads(canonical))
    validate_signing_key(signing_key, signer)
    signature = _sign_bytes(signing_key, signer.namespace, canonical)
    _write_once(path, canonical)
    _write_once(signature_path, signature)
    return cast(bytes, canonical)


def consume_execute_handoff_v2(
    path: Path,
    signature_path: Path,
    claim_path: Path,
    *,
    identity: SigningIdentity,
    bundle: Mapping[str, Any],
    launch_authorization_sha256: str,
    frozen_support_protocol_sha256: str,
    prime_observation_sha256: str,
    resource_identity: Mapping[str, Any],
    known_hosts_sha256: str,
    ledger: ProvisioningLedger,
    pod_id: str | None = None,
    pod_name: str | None = None,
    pod_status_sha256: str | None = None,
    ssh: Mapping[str, Any] | None = None,
    known_hosts_fingerprints: tuple[str, ...] | None = None,
    now_epoch: int | None = None,
) -> ExecuteHandoffV2:
    """Verify the detached handoff once, then create its durable O_EXCL claim."""

    for candidate in (path, signature_path, claim_path):
        if candidate.is_symlink():
            raise RuntimeError("execute handoff path is a symlink")
    if not path.is_file() or not signature_path.is_file():
        raise RuntimeError("execute handoff or detached signature is missing")
    # The production consumer has one authority: the fixed files at ``path``
    # and ``signature_path``.  In particular, callers cannot supply alternate
    # bytes that differ from the bytes later retained in the durable claim.
    raw = path.read_bytes()
    signature = signature_path.read_bytes()
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("execute handoff v2 is not JSON") from error
    if canonical_json_bytes(decoded) != raw:
        raise RuntimeError("execute handoff v2 is not canonical")
    handoff = ExecuteHandoffV2.from_payload(decoded)
    if handoff.state != "issued":
        raise RuntimeError("execute handoff v2 was already consumed")
    if (
        handoff.signer_namespace != identity.namespace
        or handoff.signer_principal != identity.principal
    ):
        raise RuntimeError("execute handoff signer identity differs")
    if handoff.signer_fingerprint != identity.fingerprint_sha256:
        raise RuntimeError("execute handoff signer fingerprint differs")
    _verify_bytes(identity, raw, signature)
    expected_bundle = dict(bundle)
    if set(expected_bundle) != {"commit", "tree"} or {
        "commit": handoff.bundle_commit,
        "tree": handoff.bundle_tree,
    } != expected_bundle:
        raise RuntimeError("execute handoff bundle binding differs")
    if (
        handoff.campaign_id != ledger.campaign_id
        or handoff.outcome_attempt_id
        != f"{ledger.campaign_id}:outcome-attempt-1"
        or handoff.launch_authorization_sha256 != launch_authorization_sha256
        or handoff.frozen_support_protocol_sha256 != frozen_support_protocol_sha256
        or handoff.prime_observation_sha256 != prime_observation_sha256
        or handoff.resource_identity != dict(resource_identity)
        or handoff.known_hosts_sha256 != known_hosts_sha256
        or handoff.ledger_path != HANDOFF_V2_LEDGER_RELATIVE
        or handoff.ledger_state != "provisioned"
        or handoff.ledger_sha256 != hashlib.sha256(ledger.path.read_bytes()).hexdigest()
    ):
        raise RuntimeError("execute handoff v2 does not match current launch state")
    if (
        (pod_id is not None and handoff.pod_id != pod_id)
        or (pod_name is not None and handoff.pod_name != pod_name)
        or (
            pod_status_sha256 is not None
            and handoff.pod_status_sha256 != pod_status_sha256
        )
        or (ssh is not None and handoff.ssh != dict(ssh))
        or (
            known_hosts_fingerprints is not None
            and handoff.known_hosts_fingerprints != tuple(known_hosts_fingerprints)
        )
    ):
        raise RuntimeError("execute handoff pod binding differs from current launch state")
    provisions = cast(list[dict[str, Any]], ledger._state()["provisions"])
    if handoff.provisioning_ordinal != len(provisions):
        raise RuntimeError("execute handoff provisioning ordinal differs from ledger")
    current = int(time.time()) if now_epoch is None else now_epoch
    if current < handoff.issued_at_epoch or current > handoff.expires_at_epoch:
        raise RuntimeError("execute handoff v2 is expired or future-dated")
    claim = {
        "schema_version": 2,
        "domain": "redco-stage-d1-support-v13-execute-handoff-claim-v2",
        "state": "consumed",
        "handoff_sha256": hashlib.sha256(raw).hexdigest(),
        "signature_sha256": hashlib.sha256(signature).hexdigest(),
        "handoff_bytes_b64": base64.b64encode(raw).decode("ascii"),
        "signature_bytes_b64": base64.b64encode(signature).decode("ascii"),
        "campaign_id": ledger.campaign_id,
        "provisioning_ordinal": handoff.provisioning_ordinal,
        "consumed_at_epoch": current,
    }
    _write_once(claim_path, canonical_json_bytes(claim))
    consumed = ExecuteHandoffV2(
        **{**handoff.to_payload(), "state": "consumed"}
    )
    _replace(path, canonical_json_bytes(consumed.to_payload()))
    return consumed


def dispatch_callback_from_environment() -> Callable[[bytes], None] | None:
    """Return the launch-owned callback installed for a real child process."""

    raw_path = os.environ.get(PROVISIONING_LEDGER_ENV)
    if raw_path is None:
        return None
    if not raw_path:
        raise RuntimeError("provider dispatch ledger environment is empty")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("provider dispatch ledger environment is not a file")
    return lambda request: record_provider_post_from_path(path, request)


def dispatch_callback(ledger: ProvisioningLedger, operation_id: str) -> Callable[[], None]:
    boundary = ProviderDispatchBoundary(ledger)
    return lambda: boundary.ledger.record_provider_post(operation_id=operation_id)


def evidence_blocks_replacement(state: Mapping[str, Any]) -> bool:
    return bool(
        state.get("provider_post_observed")
        or int(state.get("evidence_observation_count", 0)) > 0
        or state.get("closed")
    )


__all__ = [
    "DISPATCH_JOURNAL_SUFFIX",
    "HANDOFF_V2_DOMAIN",
    "HANDOFF_V2_LEDGER_RELATIVE",
    "HANDOFF_V2_NAMESPACE",
    "HANDOFF_V2_SCHEMA_VERSION",
    "MAX_PROVISIONING_ATTEMPTS",
    "PROVISIONING_LEDGER_ENV",
    "SCIENCE_RESERVE_USD",
    "SUPPORT_CAP_USD",
    "TEARDOWN_RESERVE_USD",
    "ExecuteHandoffV2",
    "ProviderDispatchBoundary",
    "ProvisioningLedger",
    "SigningIdentity",
    "consume_execute_handoff_v2",
    "dispatch_callback",
    "dispatch_callback_from_environment",
    "evidence_blocks_replacement",
    "issue_execute_handoff_v2",
    "record_provider_post_from_path",
    "validate_signing_key",
]
