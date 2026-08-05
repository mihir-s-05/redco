"""Authenticated historical receipts and cumulative Phase-A exclusions."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import sha256_bytes, sha256_json
from redco.analysis.stage_d_v13_draft_inputs import (
    HISTORICAL_ADDRESS_HASHES,
    HISTORICAL_ROLLOUT_HASHES,
    V4_DATASET_SHA256,
    V6_SUCCESSOR_DATASET_SHA256,
    historical_identity_witness,
    sha256_file,
)
from redco.analysis.stage_d_v13_source_phase_a_decoder import (
    SOURCE_ARTIFACT_RELATIVE,
    SOURCE_REVISION,
    SOURCE_SEMANTIC_COMMIT,
    SOURCE_SHA256,
    bounded_source_rows,
    source_row_sha256,
)
from redco.analysis.stage_d_v13_source_phase_a_selector import (
    select_first_eligible,
)

HISTORICAL_RECEIPT_MANIFEST_HASHES: dict[str, str] = {
    "datasets/stage-d/qasper-support-successor-manifest-v1.json": (
        "f090d4cd382fce120ab3bd3ae15a2123102941e53fbc344f3a2098880d2daba6"
    ),
    "datasets/stage-d/qasper-support-successor-manifest-v2.json": (
        "979fb8420fa5b9f4db7d02d6d2d0eb39f3edaa18e2b03ce9d7cd10837f8e6d81"
    ),
    "datasets/stage-d/qasper-support-successor-manifest-v3.json": (
        "04de661ac20c81332e7e8dc68d2b81ed1279107bf99392b488dd438ffb9e8e4b"
    ),
    "datasets/stage-d/qasper-support-successor-manifest-v4.json": (
        "b248f8635c625cc2b4dab5ee0d06587b4120d0a6e264debca3c17f4544e6c8d6"
    ),
    "datasets/stage-d/qasper-support-successor-manifest-v5.json": (
        "35f9f903ab7ac53ed0456237521109ea1afa5f9ff10aa01ba2208ce8f7e479eb"
    ),
    "datasets/stage-d/qasper-support-successor-manifest-v6.json": (
        "5b1667fa9f17c7e733276b17534de6453598e60b8e52a733a2028c7dab671697"
    ),
}

HISTORICAL_RECEIPT_EXPECTATIONS: tuple[dict[str, Any], ...] = (
    {
        "version": 1,
        "source_ordinal": 174,
        "example_id": "qasper-225a567eeb2698a9d3f1024a8b270313a6d15f82",
        "paper_id": "1605.04655",
        "source_row_sha256": "190ad2d2927390ec3fe6e59401add8b10aaf202a2517bace5acb736b4bd3e720",
    },
    {
        "version": 2,
        "source_ordinal": 175,
        "example_id": "qasper-6e4505609a280acc45b0a821755afb1b3b518ffd",
        "paper_id": "1911.09483",
        "source_row_sha256": "d985a629c04ee42609b0f35842971c6384c8879e0ab0afe3f8e620b9605b3521",
    },
    {
        "version": 3,
        "source_ordinal": 176,
        "example_id": "qasper-282aa4e160abfa7569de7d99b8d45cabee486ba4",
        "paper_id": "1805.00760",
        "source_row_sha256": "01b57344f5307fdb66efc94b9551bf41f1679f32a39f6414eff7de4e0fb2b11a",
    },
    {
        "version": 4,
        "source_ordinal": 177,
        "example_id": "qasper-221e9189a9d2431902d8ea833f486a38a76cbd8e",
        "paper_id": "1909.05358",
        "source_row_sha256": "923424557289b6d6fc6bfe6d6117f3c0bd44c1cc17208ec9b89902db523536f5",
    },
    {
        "version": 5,
        "source_ordinal": 178,
        "example_id": "qasper-ec8043290356fcb871c2f5d752a9fe93a94c2f71",
        "paper_id": "2003.06279",
        "source_row_sha256": "61f54eeda499aca305b3df7a24fbe5a0812b1dfad98e55b89a531c4cf9231511",
    },
    {
        "version": 6,
        "source_ordinal": 179,
        "example_id": "qasper-f33236ebd6f5a9ccb9b9dbf05ac17c3724f93f91",
        "paper_id": "2004.03744",
        "source_row_sha256": "b0facff2e4aaab643ea3cb5b1595a29bc6b082a40e0d46db364e1d0b9fa27b45",
    },
)

RETIRED_PAPERS = (
    "1911.03894",
    "2001.09899",
    "1710.01492",
    "1912.01673",
    "1909.12231",
    "1706.08032",
)
OBSERVED_EXAMPLE_ID = "qasper-69a7a6675c59a4c5fb70006523b9fe0f01ca415c"
OBSERVED_PAPER_ID = "1811.01399"
OBSERVED_ROW_SHA256 = "564ee955475039ddfdd284e48aa089ca91ef4070a123d6f4d4f89ec70974f32a"
EXPECTED_CARDINALITIES = {
    "historical_address_sha256": 74,
    "old_snapshot_paper_ids": 120,
    "old_snapshot_example_ids": 120,
    "old_snapshot_row_sha256": 120,
    "predecessor_paper_ids": 112,
    "predecessor_example_ids": 112,
    "predecessor_row_sha256": 112,
    "predecessor_reference_span_sha256": 210,
    "retired_paper_ids": 7,
    "retired_example_ids": 7,
    "retired_row_sha256": 7,
    "rendered_paper_sha256": 7,
    "old_snapshot_rendered_paper_sha256": 120,
    "predecessor_rendered_paper_sha256": 112,
    "old_snapshot_reference_span_sha256": 197,
    "reference_span_sha256": 8,
    "source_address_sha256": 7,
    "historical_paper_ids": 70,
    "historical_example_ids": 70,
    "historical_row_sha256": 70,
    "historical_seeds": 70,
    "historical_groups": 70,
    "historical_slots": 71,
    "historical_cache_salts": 70,
    "historical_call_ids": 1,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_json(root: Path, relative: str) -> dict[str, Any]:
    value = json.loads((root / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"source-auth input is not a JSON object: {relative}")
    return value


def authenticated_historical_inputs(root: Path) -> dict[str, str]:
    expected = {
        **HISTORICAL_RECEIPT_MANIFEST_HASHES,
        **HISTORICAL_ADDRESS_HASHES,
        **HISTORICAL_ROLLOUT_HASHES,
    }
    actual: dict[str, str] = {}
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required historical input is missing: {relative}")
        observed = sha256_file(path)
        if observed != digest:
            raise ValueError(f"historical input hash mismatch for {relative}")
        actual[relative] = observed
    return actual


def _address_core(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "paper_id",
        "example_id",
        "seed",
        "scientific_group_id",
        "slot_id",
        "rollout_slot",
        "cache_salt",
        "row_sha256",
        "canonical_row_sha256",
    )
    return {field: record[field] for field in fields if field in record}


def historical_receipts(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    addresses = {
        version: read_json(
            root, f"reports/stage-d1-support-successor-address-audit-v{version}.json"
        )
        for version in range(1, 7)
    }
    receipts: list[dict[str, Any]] = []
    for version, fixed in enumerate(HISTORICAL_RECEIPT_EXPECTATIONS, start=1):
        manifest = read_json(
            root, f"datasets/stage-d/qasper-support-successor-manifest-v{version}.json"
        )
        receipt = cast(
            Mapping[str, Any], cast(Mapping[str, Any], manifest["successor"])["selection_receipt"]
        )
        if (
            receipt.get("source_ordinal") != fixed["source_ordinal"]
            or receipt.get("selected_example_id") != fixed["example_id"]
            or receipt.get("selected_paper_id") != fixed["paper_id"]
            or receipt.get("source_row_sha256") != fixed["source_row_sha256"]
            or manifest.get("source_revision") != SOURCE_SEMANTIC_COMMIT
            or manifest.get("converted_parquet_revision") != SOURCE_REVISION
        ):
            raise ValueError(f"historical receipt v{version} differs from its pinned manifest")
        address_version = version + 1 if version < 6 else 6
        address = addresses[address_version]
        checks = cast(Mapping[str, Any], address.get("checks", {}))
        if not checks or not all(value is True for value in checks.values()):
            raise ValueError(f"historical address audit v{address_version} is not passing")
        records: Iterable[Mapping[str, Any]]
        role = "preserved"
        if version == 6:
            role = "reserve"
            records = (cast(Mapping[str, Any], address["reserve"]),)
        else:
            records = cast(Iterable[Mapping[str, Any]], address["preserved"])
        matching = [record for record in records if record.get("example_id") == fixed["example_id"]]
        if len(matching) != 1:
            raise ValueError(f"receipt v{version} is not bound by address audit")
        receipts.append(
            {
                **fixed,
                "manifest_path": (
                    f"datasets/stage-d/qasper-support-successor-manifest-v{version}.json"
                ),
                "address_audit_path": (
                    f"reports/stage-d1-support-successor-address-audit-v{address_version}.json"
                ),
                "address_role": role,
                "address_record": dict(matching[0]),
            }
        )
    retired: dict[str, dict[str, Any]] = {}
    for version, address in addresses.items():
        retired_record = cast(Mapping[str, Any], address["retired"])
        paper_id = str(retired_record["paper_id"])
        if paper_id not in RETIRED_PAPERS:
            raise ValueError(f"unexpected retired paper in address audit v{version}")
        retired[paper_id] = {
            "address_audit_version": int(version),
            "address_audit_path": (
                f"reports/stage-d1-support-successor-address-audit-v{version}.json"
            ),
            "address_record": dict(retired_record),
        }
    if set(retired) != set(RETIRED_PAPERS):
        raise ValueError("six historical retired papers are not authenticated")
    return receipts, retired


def _hash_identity_rows(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"historical row artifact is malformed: {relative}")
    raw_reference_spans = sorted(
        str(span) for row in rows for span in cast(Iterable[Any], row["reference_evidence"])
    )
    identity_sets = {
        "paper_ids": sorted(str(row["paper_id"]) for row in rows),
        "example_ids": sorted(str(row["example_id"]) for row in rows),
        "row_sha256": sorted(source_row_sha256(cast(Mapping[str, Any], row)) for row in rows),
        "rendered_paper_sha256": sorted(
            sha256_bytes(str(row["paper"]).encode("utf-8")) for row in rows
        ),
        "reference_span_sha256": sorted(
            sha256_bytes(span.encode("utf-8")) for span in raw_reference_spans
        ),
    }
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "identity_sets": identity_sets,
        "identity_witness_sha256": sha256_json(identity_sets),
        "_raw_reference_spans": raw_reference_spans,
    }


def reconstruct_retired_units(
    root: Path,
    *,
    prefix_rows: Iterable[tuple[int, Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct the six historical rows plus the observed v12 unit."""

    _receipts, retired_addresses = historical_receipts(root)
    by_paper: dict[str, dict[str, Any]] = {}
    source_rows = prefix_rows or bounded_source_rows(root / SOURCE_ARTIFACT_RELATIVE)
    for ordinal, row in source_rows:
        paper_id = str(row["id"])
        if paper_id not in RETIRED_PAPERS:
            continue
        selected = select_first_eligible(row)
        if selected is None:
            raise ValueError(f"retired paper has no historical eligible question: {paper_id}")
        chosen, question_index = selected
        address = retired_addresses[paper_id]["address_record"]
        selected_hash = source_row_sha256(chosen)
        expected_hash = address.get("row_sha256") or address.get("canonical_row_sha256")
        if chosen["example_id"] != address["example_id"] or selected_hash != expected_hash:
            raise ValueError(f"retired row reconstruction differs for {paper_id}")
        references = [str(value) for value in chosen["reference_evidence"]]
        by_paper[paper_id] = {
            "paper_id": paper_id,
            "example_id": str(chosen["example_id"]),
            "source_ordinal": ordinal,
            "question_index": question_index,
            "canonical_row_sha256": selected_hash,
            "rendered_paper_sha256": [sha256_bytes(str(chosen["paper"]).encode("utf-8"))],
            "reference_span_sha256": [sha256_bytes(value.encode("utf-8")) for value in references],
            "source_address_sha256": sha256_json(_address_core(address)),
            "address_audit_version": retired_addresses[paper_id]["address_audit_version"],
            "address_audit_path": retired_addresses[paper_id]["address_audit_path"],
            "status": "authenticated",
            "_raw_reference_evidence": references,
        }
    if set(by_paper) != set(RETIRED_PAPERS):
        raise ValueError("bounded source prefix did not reconstruct all six retired rows")

    observed_rows = [
        json.loads(line)
        for line in (root / "datasets/stage-d/qasper-support-successor-v6.jsonl")
        .read_bytes()
        .splitlines()
        if line
    ]
    if not observed_rows or not isinstance(observed_rows[0], dict):
        raise ValueError("v12 observed row is malformed")
    observed = observed_rows[0]
    observed_hash = source_row_sha256(observed)
    if (
        observed_hash != OBSERVED_ROW_SHA256
        or observed.get("example_id") != OBSERVED_EXAMPLE_ID
        or observed.get("paper_id") != OBSERVED_PAPER_ID
    ):
        raise ValueError("v12 observed row does not match authenticated predecessor")
    address_v6 = read_json(root, "reports/stage-d1-support-successor-address-audit-v6.json")
    observed_addresses = [
        record
        for record in cast(Iterable[Mapping[str, Any]], address_v6["preserved"])
        if record.get("example_id") == OBSERVED_EXAMPLE_ID
    ]
    if len(observed_addresses) != 1:
        raise ValueError("v12 observed address is not authenticated")
    references = [str(value) for value in observed["reference_evidence"]]
    by_paper[OBSERVED_PAPER_ID] = {
        "paper_id": OBSERVED_PAPER_ID,
        "example_id": OBSERVED_EXAMPLE_ID,
        "source_ordinal": None,
        "canonical_row_sha256": observed_hash,
        "rendered_paper_sha256": [sha256_bytes(str(observed["paper"]).encode("utf-8"))],
        "reference_span_sha256": [sha256_bytes(value.encode("utf-8")) for value in references],
        "source_address_sha256": sha256_json(_address_core(observed_addresses[0])),
        "address_audit_version": 6,
        "address_audit_path": "reports/stage-d1-support-successor-address-audit-v6.json",
        "status": "authenticated_observed_v12_unit",
        "_raw_reference_evidence": references,
    }
    return [by_paper[paper] for paper in (*RETIRED_PAPERS, OBSERVED_PAPER_ID)]


def _public_units(units: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in unit.items() if not key.startswith("_")} for unit in units
    ]


def _build_witness(
    root: Path,
    units: list[dict[str, Any]],
    address_inputs: Mapping[str, str],
) -> tuple[dict[str, Any], set[str]]:
    historical = historical_identity_witness(root)
    historical_sets = cast(Mapping[str, Any], historical["identity_sets"])
    sanitized_historical: dict[str, list[str]] = {}
    raw_reference_spans: set[str] = set()
    for name, values in historical_sets.items():
        if name == "reference_spans":
            raw_reference_spans.update(str(value) for value in cast(Iterable[Any], values))
            sanitized_historical["reference_span_sha256"] = sorted(
                sha256_bytes(str(value).encode("utf-8")) for value in cast(Iterable[Any], values)
            )
        else:
            sanitized_historical[name] = sorted(str(value) for value in cast(Iterable[Any], values))
    if len(sanitized_historical.get("addresses", [])) != 74:
        raise ValueError("historical address witness cardinality differs")
    old_snapshot = _hash_identity_rows(root, "datasets/stage-d/qasper-deterministic-v4.jsonl")
    predecessor = _hash_identity_rows(root, "datasets/stage-d/qasper-support-successor-v6.jsonl")
    if (
        old_snapshot["sha256"] != V4_DATASET_SHA256
        or predecessor["sha256"] != V6_SUCCESSOR_DATASET_SHA256
    ):
        raise ValueError("authenticated historical dataset hash differs")
    raw_reference_spans.update(old_snapshot["_raw_reference_spans"])
    raw_reference_spans.update(predecessor["_raw_reference_spans"])
    raw_reference_spans.update(
        str(span)
        for unit in units
        for span in cast(Iterable[str], unit.get("_raw_reference_evidence", ()))
    )
    public_units = _public_units(units)
    sets: dict[str, list[str]] = {
        "retired_paper_ids": sorted(str(unit["paper_id"]) for unit in public_units),
        "retired_example_ids": sorted(str(unit["example_id"]) for unit in public_units),
        "retired_row_sha256": sorted(str(unit["canonical_row_sha256"]) for unit in public_units),
        "rendered_paper_sha256": sorted(
            str(value)
            for unit in public_units
            for value in cast(Iterable[str], unit["rendered_paper_sha256"])
        ),
        "reference_span_sha256": sorted(
            str(value)
            for unit in public_units
            for value in cast(Iterable[str], unit["reference_span_sha256"])
        ),
        "source_address_sha256": sorted(
            str(unit["source_address_sha256"]) for unit in public_units
        ),
        "old_snapshot_paper_ids": old_snapshot["identity_sets"]["paper_ids"],
        "old_snapshot_example_ids": old_snapshot["identity_sets"]["example_ids"],
        "old_snapshot_row_sha256": old_snapshot["identity_sets"]["row_sha256"],
        "old_snapshot_rendered_paper_sha256": old_snapshot["identity_sets"][
            "rendered_paper_sha256"
        ],
        "old_snapshot_reference_span_sha256": old_snapshot["identity_sets"][
            "reference_span_sha256"
        ],
        "predecessor_paper_ids": predecessor["identity_sets"]["paper_ids"],
        "predecessor_example_ids": predecessor["identity_sets"]["example_ids"],
        "predecessor_row_sha256": predecessor["identity_sets"]["row_sha256"],
        "predecessor_rendered_paper_sha256": predecessor["identity_sets"]["rendered_paper_sha256"],
        "predecessor_reference_span_sha256": predecessor["identity_sets"]["reference_span_sha256"],
        "historical_address_sha256": sanitized_historical["addresses"],
        "historical_reference_span_sha256": sanitized_historical.get("reference_span_sha256", []),
        "historical_paper_ids": sanitized_historical.get("paper_ids", []),
        "historical_example_ids": sanitized_historical.get("example_ids", []),
        "historical_row_sha256": sanitized_historical.get("row_hashes", []),
        "historical_seeds": sanitized_historical.get("seeds", []),
        "historical_groups": sanitized_historical.get("groups", []),
        "historical_slots": sanitized_historical.get("slots", []),
        "historical_cache_salts": sanitized_historical.get("cache_salts", []),
        "historical_call_ids": sanitized_historical.get("call_ids", []),
    }
    exclusion_hashes = {name: sha256_json(values) for name, values in sets.items()}
    witness: dict[str, Any] = {
        "schema_version": 2,
        "source_artifact_sha256": SOURCE_SHA256,
        "authenticated_address_inputs": dict(sorted(address_inputs.items())),
        "authenticated_historical_identity_witness": {
            "artifact_hashes": historical["artifacts"],
            "witness_sha256": historical["witness_sha256"],
            "identity_sets": sanitized_historical,
            "address_count": len(sanitized_historical["addresses"]),
        },
        "retired_units": public_units,
        "old_deterministic_snapshot": {
            key: value for key, value in old_snapshot.items() if not key.startswith("_")
        },
        "complete_v6_predecessor": {
            key: value for key, value in predecessor.items() if not key.startswith("_")
        },
        "forbidden_sets": sets,
        "exclusion_hashes": exclusion_hashes,
        "collision_rule": {
            "paper_or_reference_collision": "continue_source_order_scan",
            "address_or_identity_collision": "terminal_fail_closed",
            "candidate_dependent_checks_before_candidate": None,
        },
    }
    witness["witness_sha256"] = sha256_json(witness)
    return witness, raw_reference_spans


def build_forbidden_witness(
    root: Path,
    units: list[dict[str, Any]],
    address_inputs: Mapping[str, str],
) -> tuple[dict[str, Any], set[str]]:
    return _build_witness(root, units, address_inputs)


def validate_forbidden_witness(
    root: Path,
    witness: Mapping[str, Any],
    *,
    expected_units: list[dict[str, Any]] | None = None,
    address_inputs: Mapping[str, str] | None = None,
) -> None:
    """Rebuild the expected structure from authenticated inputs, not self-hashes."""

    # Ignore caller-provided structures for the authoritative comparison. They
    # are accepted only as a compatibility aid for diagnostics; rebuilding
    # them from the authenticated source and receipts prevents a mutated
    # witness and a mutated expected object from validating together.
    del expected_units, address_inputs
    authenticated_inputs = authenticated_historical_inputs(root)
    expected_units_from_source = reconstruct_retired_units(root)
    expected, _raw = _build_witness(root, expected_units_from_source, authenticated_inputs)
    actual_without_hash = {key: value for key, value in witness.items() if key != "witness_sha256"}
    expected_without_hash = {
        key: value for key, value in expected.items() if key != "witness_sha256"
    }
    if actual_without_hash != expected_without_hash:
        raise ValueError(
            "forbidden witness differs from independently rebuilt authenticated structure"
        )
    if witness.get("witness_sha256") != sha256_json(actual_without_hash):
        raise ValueError("forbidden witness hash is not self-authenticating")
    sets = cast(Mapping[str, Any], witness["forbidden_sets"])
    for name, expected_count in EXPECTED_CARDINALITIES.items():
        values = sets.get(name)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(f"forbidden witness cardinality differs for {name}")
    for name, values in sets.items():
        if name.endswith("sha256") and (
            not isinstance(values, list)
            or any(
                not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
                for value in values
            )
        ):
            raise ValueError(f"forbidden witness digest list is malformed for {name}")
    exclusion_hashes = cast(Mapping[str, Any], witness["exclusion_hashes"])
    if {name: sha256_json(values) for name, values in sets.items()} != dict(exclusion_hashes):
        raise ValueError("forbidden witness exclusion hashes differ")


__all__ = [
    "EXPECTED_CARDINALITIES",
    "HISTORICAL_RECEIPT_EXPECTATIONS",
    "OBSERVED_EXAMPLE_ID",
    "OBSERVED_PAPER_ID",
    "OBSERVED_ROW_SHA256",
    "RETIRED_PAPERS",
    "authenticated_historical_inputs",
    "build_forbidden_witness",
    "historical_receipts",
    "read_json",
    "reconstruct_retired_units",
    "validate_forbidden_witness",
]
