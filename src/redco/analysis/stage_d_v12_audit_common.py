"""Shared pure values and bounded comparisons for the v12 offline audit."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, cast

from redco.analysis.stage_d_source_producer import (
    _normalize_openai_message,
    _normalize_openai_tools,
)
from redco.contracts import canonical_json

ARCHIVE_SHA256: Final = "c2bb6713234653fc08e10778aa17815ad5a26f769406806037f50c390820b894"
EVIDENCE_MANIFEST_SHA256: Final = "90c694a45ece9887ea658adc36a8b30f6b7d78f8b111c017f54b5e5d51003671"
TERMINAL_REPORT_SHA256: Final = "3b79bbf541ee4210744b37f2df49531ca4e6b601f500e0ecaa43e3fa9e8ca9ec"
KNOWN_FAILURE_DECISION_ID: Final = "decision-9eb4cf0ed5732ad818483e5f"
KNOWN_FAILURE_LINEAGE: Final = "root/cbb0fb0fe8ca42bac12b7225"
KNOWN_FAILURE_NODE: Final = 10
FROZEN_RUNTIME_CODE_COMMIT: Final = "7b54f25912a9842c000291d20314dc831eca776b"
FROZEN_TRACE_ID: Final = "c7834d223cdb4f568d5835314e04d5ba"
FROZEN_TERMINAL_REPORT_SCHEMA_VERSION: Final = 1
AUDIT_SCHEMA_VERSION: Final = 2
AUDIT_DOMAIN: Final = "redco-stage-d1-v12-finalization-engineering-audit-v2"

FROZEN_ARCHIVE_RELATIVE: Final = Path("runs/stage-d/stage-d1-support-v12-terminal.tar.gz")
FROZEN_MANIFEST_RELATIVE: Final = Path("runs/stage-d/stage-d1-support-v12-evidence-sha256.txt")
FROZEN_REPORT_RELATIVE: Final = Path("reports/stage-d1-support-v12-terminal.json")
FROZEN_ARCHIVE_ROOT_RELATIVE: Final = Path("runs/stage-d/stage-d1-support-v12")

FROZEN_REPO_FILE_SHA256: Final[dict[str, str]] = {
    "configs/stage-d/stage-d1-dependency-stack-v12.json": (
        "cda524c6ecea9821b1e36290da64df465aa46fad9ec174881c24d3dc895b2831"
    ),
    "configs/stage-d/stage-d1-support-collection-plan-v11.json": (
        "9870c18fd43ee3a08bb212d5f9b506104e39b3d2dc2ccd7b689240799d2696cb"
    ),
    "configs/stage-d/stage-d1-support-genesis-v12.json": (
        "94df4470e5c285597023a357e0f179feece26be652b54b7717c83545370f2d14"
    ),
    "configs/stage-d/stage-d1-support-preregistration-v12.json": (
        "8cf086e4b198c45306a0cb7d3289b72e65fa2ee9ae34940a42e141542003b429"
    ),
    "configs/stage-d/stage-d1-support-protocol-v12.json": (
        "2be6b64916ef3620dc15fade89916b616de1ea8f54db0109c7f0ff5c3be8e9fd"
    ),
    "configs/stage-d/stage-d1-support-rules-v1.json": (
        "088e3990c270881435214c725c0ca984462f9a824a1e0708d1bcbebd4264d235"
    ),
    "configs/stage-d/stage-d1-support-source-eval-v12.toml": (
        "704eca5bb15a7ee52572653110639857dcffe422a2d5cfe1a66b498959e88351"
    ),
    "configs/stage-d/stage-d1-support-source-v12.json": (
        "034a9bc05d8ff28699d29e6e6d649dbfb9cb57191af0fb6af34983f9d18d9141"
    ),
    "configs/stage-d/stage-d1-support-deployment-amendment-v12-2.json": (
        "6737dcf0957dab91ce201bceea9bea8e096ca4611e810460d65088c32100a022"
    ),
    "configs/stage-d/stage-d1-support-deployment-amendment-v12-3.json": (
        "116abe769bfc8ffce819e2d500af3f381f45f6ac669e9468253bc71962bd3a2e"
    ),
    "configs/stage-d/stage-d1-support-deployment-amendment-v12-4.json": (
        "d38d32935f0be6de5914c3f53de5ccdcad595a62bd4501eb0757ecf6ce011eac"
    ),
    "environments/redco_evidence_selection_v2/redco_evidence_selection_v2/source_env.py": (
        "b2971989f10ab354d6f8367848fb2ab9eb3a2ccd1e89a892e9e01e69406d28aa"
    ),
    "pyproject.toml": "94c85ca6ffd627b07cfee14ce8ba80b3cb19fb279e7b98792fddf12695e0699b",
    "src/redco/analysis/stage_d_live_observer.py": (
        "79b914d8722efd64ab42b485dfb7b78c69a9352f077353657a69dacd626b20c9"
    ),
    "src/redco/analysis/stage_d_source_producer.py": (
        "4e2f59b3ae973eaaa8aab8dca378e196430acb650cf8c005dbb2227b1d0923b1"
    ),
    "uv.lock": "60e9fe7396d45d8e8edd13d2de708fa4895452410b43e1ad860f720047634d31",
}

_ABSENT = object()
_STATUS = Literal["pass", "fail", "not_observable_from_persisted_schema"]
_METHOD = Literal[
    "directly_verified_from_archive",
    "reconstructed_on_disposable_copy",
    "not_observable_from_persisted_schema",
]
_POST_CALL_INVARIANT_NAMES: tuple[str, ...] = (
    "trace_fields_exact",
    "trace_artifact_hash_binding",
    "trace_success_state",
    "sampled_nodes_biject_calls",
    "parent_graph_paths",
    "durable_address_mapping",
    "trace_reward_and_metadata",
    "ledger_terminal_poison",
    "exactly_one_finalization_abort",
    "finalization_abort_error_digest",
    "source_artifacts_absent",
    "source_eligibility_not_recovered",
)
_SEMANTIC_RECONSTRUCTION_NAMES: tuple[str, ...] = (
    "episode_schema_and_trace_contract",
    "deployed_parent_links",
    "strict_scaffold_eligibility",
    "sampled_node_call_bijection",
    "sampled_node_mask_shape",
    "leaf_path_sample_derivation",
    "exactly_once_sampled_node_routing",
    "finite_reward_summation",
    "child_target_roster",
    "graph_to_source_mappings",
    "child_weight_normalization",
    "source_semantic_equivalence",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str, name: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{name} hash differs from its frozen value: {actual}")
    return actual


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _mapping_list(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of JSON objects")
    return value


def _canonical(value: Any) -> bytes:
    return cast(bytes, canonical_json(value))


def _bounded(value: object) -> dict[str, Any]:
    """Describe a value without emitting generated text or long arrays."""
    if value is _ABSENT:
        return {"presence": "absent"}
    if value is None:
        return {
            "presence": "present-null",
            "type": "null",
            "sha256": sha256_bytes(_canonical(None)),
        }
    if isinstance(value, str):
        return {
            "presence": "present-value",
            "type": "string",
            "length": len(value),
            "sha256": sha256_bytes(_canonical(value)),
        }
    if isinstance(value, dict):
        return {
            "presence": "present-value",
            "type": "object",
            "key_count": len(value),
            "keys_sha256": sha256_bytes(_canonical(sorted(value))),
            "sha256": sha256_bytes(_canonical(value)),
        }
    if isinstance(value, list):
        return {
            "presence": "present-value",
            "type": "array",
            "length": len(value),
            "sha256": sha256_bytes(_canonical(value)),
        }
    return {
        "presence": "present-value",
        "type": type(value).__name__,
        "sha256": sha256_bytes(_canonical(value)),
    }


def _pointer_key(parent: str, key: object) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"/{escaped}" if parent == "" else f"{parent}/{escaped}"


def json_pointer_differences(
    left: object, right: object, pointer: str = ""
) -> list[dict[str, Any]]:
    """Return RFC 6901 presence-aware differences with bounded subvalues."""
    if left is _ABSENT or right is _ABSENT:
        if left is right:
            return []
        return [
            {
                "pointer": pointer,
                "left": _bounded(left),
                "right": _bounded(right),
                "reason": "presence_or_value_difference",
            }
        ]
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            differences.extend(
                json_pointer_differences(
                    left.get(key, _ABSENT),
                    right.get(key, _ABSENT),
                    _pointer_key(pointer, key),
                )
            )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            differences.append(
                {
                    "pointer": pointer,
                    "left": _bounded(left),
                    "right": _bounded(right),
                    "reason": "array_length_difference",
                }
            )
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            differences.extend(
                json_pointer_differences(
                    left_item,
                    right_item,
                    _pointer_key(pointer, index),
                )
            )
        return differences
    if left != right or type(left) is not type(right):
        return [
            {
                "pointer": pointer,
                "left": _bounded(left),
                "right": _bounded(right),
                "reason": "value_or_type_difference",
            }
        ]
    return []


def _status(
    name: str,
    status: _STATUS,
    method: _METHOD,
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "status": status, "method": method}
    if detail:
        result["detail"] = detail
    return result


def _normalize_message_for_audit(value: object) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return _normalize_openai_message(value), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def _normalize_tools_for_audit(value: object) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        return _normalize_openai_tools(value), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def _accept_archived_action(
    _request: Mapping[str, Any],
    _message: Mapping[str, Any],
    _tokens: Sequence[int],
) -> None:
    """Load archived actions without requiring the frozen tokenizer callable."""


def _message_audit(transport: object, trace: object) -> dict[str, Any]:
    raw_differences = json_pointer_differences(transport, trace)
    normalized_transport, transport_error = _normalize_message_for_audit(transport)
    normalized_trace, trace_error = _normalize_message_for_audit(trace)
    normalized_differences = (
        json_pointer_differences(normalized_transport, normalized_trace)
        if normalized_transport is not None and normalized_trace is not None
        else []
    )
    return {
        "raw_equal": not raw_differences,
        "canonical_equal_under_current_finalizer": (
            not normalized_differences and transport_error is None and trace_error is None
        ),
        "transport_normalization_error": transport_error,
        "trace_normalization_error": trace_error,
        "differences": raw_differences,
        "normalized_differences": normalized_differences,
    }
