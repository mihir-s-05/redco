"""Build a fresh QASPER extension disjoint from the immutable Stage D v4 snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

QASPER_SOURCE_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
QASPER_PARQUET_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
PARQUET_BASE = (
    "https://huggingface.co/datasets/allenai/qasper/resolve/"
    f"{QASPER_PARQUET_REVISION}/qasper"
)
EXTENSION_SPLITS = {
    "successor_support": 64,
    "successor_science_train": 16,
    "successor_science_eval": 32,
}


def _encode_rows(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )


def _decode_rows(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.decode("utf-8").splitlines() if line]


def _require_fresh_outputs(paths: Iterable[Path]) -> tuple[Path, ...]:
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("bundle output paths must be distinct")
    existing = [path for path in resolved if path.exists()]
    if existing:
        raise FileExistsError(f"bundle outputs already exist: {existing}")
    return resolved


def _write_fresh_bundle(outputs: list[tuple[Path, bytes]]) -> None:
    """Promote a validated bundle once, with its manifest strictly last."""
    paths = _require_fresh_outputs(path for path, _ in outputs)
    temporary: list[tuple[Path, Path]] = []
    try:
        for (_path, payload), resolved in zip(outputs, paths, strict=True):
            resolved.parent.mkdir(parents=True, exist_ok=True)
            pending = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
            with pending.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.append((pending, resolved))
        for pending, destination in temporary:
            os.replace(pending, destination)
    finally:
        for pending, _ in temporary:
            pending.unlink(missing_ok=True)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_paper(row: dict[str, Any]) -> str:
    parts = [
        f"### PAPER: {row['title']}",
        "<abstract>",
        row["abstract"],
        "</abstract>",
    ]
    sections = row["full_text"]
    for name, paragraphs in zip(
        sections["section_name"], sections["paragraphs"], strict=True
    ):
        parts.append(f"\n## {name}")
        parts.extend(paragraphs)
    return "\n".join(parts)


def _answer_type(annotation: dict[str, Any]) -> str:
    if annotation.get("yes_no") is not None:
        return "yes_no"
    if annotation.get("extractive_spans"):
        return "extractive"
    if annotation.get("free_form_answer"):
        return "abstractive"
    return "other"


def _exact_reference(
    paper: str,
    answers: dict[str, Any],
    *,
    minimum_span_characters: int,
) -> tuple[tuple[str, ...], str] | None:
    candidates: list[tuple[tuple[str, ...], str]] = []
    for annotation in answers["answer"]:
        if annotation["unanswerable"]:
            continue
        evidence = tuple(
            dict.fromkeys(
                span.strip()
                for span in annotation["evidence"]
                if len(span.strip()) >= minimum_span_characters
                and span.strip() in paper
            )
        )
        if evidence:
            candidates.append((evidence, _answer_type(annotation)))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (sum(map(len, item[0])), len(item[0]), item[1]),
    )


def select_extension_examples(
    rows: Iterable[dict[str, Any]],
    *,
    output_split: str,
    limit: int,
    maximum_paper_characters: int,
    minimum_span_characters: int,
    forbidden_paper_ids: set[str],
    forbidden_reference_spans: set[str],
    selection_receipts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for source_ordinal, row in enumerate(rows):
        if row["id"] in forbidden_paper_ids:
            continue
        paper = render_paper(row)
        if len(paper) > maximum_paper_characters:
            continue
        qas = row["qas"]
        chosen = None
        for index, question in enumerate(qas["question"]):
            reference = _exact_reference(
                paper,
                qas["answers"][index],
                minimum_span_characters=minimum_span_characters,
            )
            if reference is None:
                continue
            evidence, kind = reference
            if set(evidence) & forbidden_reference_spans:
                continue
            chosen = {
                "example_id": f"qasper-{qas['question_id'][index]}",
                "paper_id": row["id"],
                "title": row["title"],
                "question": question,
                "answer_type": kind,
                "split": output_split,
                "paper": paper,
                "reference_evidence": list(evidence),
            }
            break
        if chosen is not None:
            examples.append(chosen)
            if selection_receipts is not None:
                selection_receipts.append(
                    {
                        "source_ordinal": source_ordinal,
                        "source_row_sha256": _sha256(_encode_rows([row])),
                        "selected_example_id": chosen["example_id"],
                        "selected_paper_id": chosen["paper_id"],
                    }
                )
            forbidden_paper_ids.add(chosen["paper_id"])
            forbidden_reference_spans.update(chosen["reference_evidence"])
        if len(examples) == limit:
            return examples
    raise RuntimeError(
        f"only found {len(examples)} fresh {output_split} papers, needed {limit}"
    )


def materialize_extension(
    *,
    old_rows: list[dict[str, Any]],
    train_rows: Iterable[dict[str, Any]],
    validation_rows: Iterable[dict[str, Any]],
    maximum_paper_characters: int,
    minimum_span_characters: int,
) -> list[dict[str, Any]]:
    forbidden_paper_ids = {str(row["paper_id"]) for row in old_rows}
    forbidden_reference_spans = {
        str(span)
        for row in old_rows
        for span in row["reference_evidence"]
    }
    support = select_extension_examples(
        train_rows,
        output_split="successor_support",
        limit=64,
        maximum_paper_characters=maximum_paper_characters,
        minimum_span_characters=minimum_span_characters,
        forbidden_paper_ids=forbidden_paper_ids,
        forbidden_reference_spans=forbidden_reference_spans,
    )
    science_train = select_extension_examples(
        train_rows,
        output_split="successor_science_train",
        limit=16,
        maximum_paper_characters=maximum_paper_characters,
        minimum_span_characters=minimum_span_characters,
        forbidden_paper_ids=forbidden_paper_ids,
        forbidden_reference_spans=forbidden_reference_spans,
    )
    science_eval = select_extension_examples(
        validation_rows,
        output_split="successor_science_eval",
        limit=32,
        maximum_paper_characters=maximum_paper_characters,
        minimum_span_characters=minimum_span_characters,
        forbidden_paper_ids=forbidden_paper_ids,
        forbidden_reference_spans=forbidden_reference_spans,
    )
    return [*support, *science_train, *science_eval]


def materialize_support_successor(
    *,
    old_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    train_rows: Iterable[dict[str, Any]],
    retired_example_id: str,
    maximum_paper_characters: int,
    minimum_span_characters: int,
    historically_retired_paper_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace one observed support address while preserving the other rows exactly."""
    prior_counts = Counter(str(row["split"]) for row in prior_rows)
    if len(prior_rows) != 112 or prior_counts != Counter(EXTENSION_SPLITS):
        raise ValueError("prior extension partition differs from the frozen 112 rows")
    support = [row for row in prior_rows if row["split"] == "successor_support"]
    retired = [row for row in support if row["example_id"] == retired_example_id]
    if len(retired) != 1:
        raise ValueError("retired support example must identify exactly one row")
    if support[0]["example_id"] != retired_example_id:
        raise ValueError("the observed retired address must be the first support row")
    retained = [row for row in support if row["example_id"] != retired_example_id]
    forbidden_rows = [*old_rows, *prior_rows]
    retired_history = set(historically_retired_paper_ids or ())
    excluded_paper_ids = {
        *(str(row["paper_id"]) for row in forbidden_rows),
        *retired_history,
    }
    excluded_reference_spans = {
        str(span)
        for row in forbidden_rows
        for span in row["reference_evidence"]
    }
    selection_receipts: list[dict[str, Any]] = []
    reserve = select_extension_examples(
        train_rows,
        output_split="successor_support",
        limit=1,
        maximum_paper_characters=maximum_paper_characters,
        minimum_span_characters=minimum_span_characters,
        forbidden_paper_ids=set(excluded_paper_ids),
        forbidden_reference_spans=set(excluded_reference_spans),
        selection_receipts=selection_receipts,
    )[0]
    inherited_science = [
        row for row in prior_rows if row["split"] != "successor_support"
    ]
    rows = [*retained, reserve, *inherited_science]
    exclusion_hashes = {
        "paper_ids_sha256": _sha256(
            _encode_rows(
                [{"value": value} for value in sorted(excluded_paper_ids)]
            )
        ),
        "example_ids_sha256": _sha256(
            _encode_rows(
                [
                    {"value": value}
                    for value in sorted(
                        str(row["example_id"]) for row in forbidden_rows
                    )
                ]
            )
        ),
        "rendered_paper_sha256s_sha256": _sha256(
            _encode_rows(
                [
                    {"value": value}
                    for value in sorted(
                        _sha256(str(row["paper"]).encode("utf-8"))
                        for row in forbidden_rows
                    )
                ]
            )
        ),
        "reference_spans_sha256": _sha256(
            _encode_rows(
                [
                    {"value": value}
                    for value in sorted(excluded_reference_spans)
                ]
            )
        ),
    }
    checks = {
        "retired_exactly_one_observed_row": len(retired) == 1,
        "retained_63_original_rows_in_order": retained == support[1:] and len(retained) == 63,
        "successor_has_112_unique_papers": len(rows)
        == len({str(row["paper_id"]) for row in rows})
        == 112,
        "inherited_science_48_rows_in_order": inherited_science
        == [row for row in prior_rows if row["split"] != "successor_support"],
        "reserve_excluded_from_every_prior_partition": reserve["paper_id"]
        not in {str(row["paper_id"]) for row in prior_rows},
        "reserve_example_id_fresh": reserve["example_id"]
        not in {str(row["example_id"]) for row in forbidden_rows},
        "reserve_rendered_paper_fresh": _sha256(
            str(reserve["paper"]).encode("utf-8")
        )
        not in {
            _sha256(str(row["paper"]).encode("utf-8")) for row in forbidden_rows
        },
        "reserve_excluded_from_immutable_snapshot": reserve["paper_id"]
        not in {str(row["paper_id"]) for row in old_rows},
        "reserve_excluded_from_retired_history": reserve["paper_id"]
        not in retired_history,
        "reserve_reference_fresh": not set(reserve["reference_evidence"])
        & {
            str(span)
            for row in forbidden_rows
            for span in row["reference_evidence"]
        },
        "reserve_not_in_science_partitions": reserve["example_id"]
        not in {row["example_id"] for row in inherited_science},
        "every_reference_exact": all(
            span in row["paper"]
            for row in rows
            for span in row["reference_evidence"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"support successor audit failed: {checks}")
    return rows, {
        "checks": checks,
        "retired": {
            "example_id": retired[0]["example_id"],
            "paper_id": retired[0]["paper_id"],
            "row_sha256": _sha256(_encode_rows(retired)),
        },
        "reserve": {
            "example_id": reserve["example_id"],
            "paper_id": reserve["paper_id"],
            "row_sha256": _sha256(_encode_rows([reserve])),
        },
        "selection_receipt": selection_receipts[0],
        "historically_retired_paper_ids": sorted(retired_history),
        "exclusion_hashes": exclusion_hashes,
        "retained": [
            {
                "example_id": row["example_id"],
                "row_sha256": _sha256(_encode_rows([row])),
            }
            for row in retained
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-snapshot", type=Path, required=True)
    parser.add_argument("--old-snapshot-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-extension", type=Path)
    parser.add_argument("--prior-extension-sha256")
    parser.add_argument("--retired-example-id")
    parser.add_argument("--prior-collection-plan", type=Path)
    parser.add_argument("--prior-collection-plan-sha256")
    parser.add_argument("--collection-plan-output", type=Path)
    parser.add_argument("--address-audit", type=Path)
    parser.add_argument("--master-seed")
    parser.add_argument("--group-namespace")
    parser.add_argument(
        "--historically-retired-paper-id",
        action="append",
        default=[],
        help="Previously observed paper ID that may never be selected as a reserve.",
    )
    parser.add_argument("--maximum-paper-characters", type=int, default=60_000)
    parser.add_argument("--minimum-span-characters", type=int, default=20)
    args = parser.parse_args()

    successor_args = (
        args.prior_extension,
        args.prior_extension_sha256,
        args.retired_example_id,
        args.prior_collection_plan,
        args.prior_collection_plan_sha256,
        args.collection_plan_output,
        args.address_audit,
        args.master_seed,
        args.group_namespace,
    )
    if any(value is not None for value in successor_args) and not all(successor_args):
        raise ValueError("support successor mode requires all successor arguments")
    successor_mode = args.prior_extension is not None
    output_paths = [args.output]
    if successor_mode:
        output_paths.extend([args.collection_plan_output, args.address_audit])
    output_paths.append(args.manifest)
    _require_fresh_outputs(output_paths)

    old_bytes = args.old_snapshot.read_bytes()
    if _sha256(old_bytes) != args.old_snapshot_sha256:
        raise ValueError("immutable old snapshot hash mismatch")
    old_rows = _decode_rows(old_bytes)
    if len(old_rows) != 120:
        raise ValueError("immutable v4 snapshot must contain 120 papers")

    from datasets import load_dataset

    train_rows = load_dataset(
        "parquet",
        data_files={
            "train": f"{PARQUET_BASE}/train/0000.parquet",
        },
        split="train",
        streaming=True,
    )
    successor_details = None
    prior_line_by_example_id: dict[str, bytes] | None = None
    if args.prior_extension is not None:
        prior_bytes = args.prior_extension.read_bytes()
        if _sha256(prior_bytes) != args.prior_extension_sha256:
            raise ValueError("prior extension hash mismatch")
        prior_rows = _decode_rows(prior_bytes)
        if _encode_rows(prior_rows) != prior_bytes:
            raise ValueError("prior extension rows are not canonical JSONL")
        prior_lines = [line + b"\n" for line in prior_bytes.splitlines()]
        prior_line_by_example_id = {
            row["example_id"]: line
            for row, line in zip(prior_rows, prior_lines, strict=True)
        }
        if len(prior_line_by_example_id) != len(prior_rows):
            raise ValueError("prior extension example IDs are not unique")
        rows, successor_details = materialize_support_successor(
            old_rows=old_rows,
            prior_rows=prior_rows,
            train_rows=train_rows,
            retired_example_id=args.retired_example_id,
            maximum_paper_characters=args.maximum_paper_characters,
            minimum_span_characters=args.minimum_span_characters,
            historically_retired_paper_ids=set(args.historically_retired_paper_id),
        )
    else:
        validation_rows = load_dataset(
            "parquet",
            data_files={
                "validation": f"{PARQUET_BASE}/validation/0000.parquet",
            },
            split="validation",
            streaming=True,
        )
        rows = materialize_extension(
            old_rows=old_rows,
            train_rows=train_rows,
            validation_rows=validation_rows,
            maximum_paper_characters=args.maximum_paper_characters,
            minimum_span_characters=args.minimum_span_characters,
        )
    if successor_details is None:
        payload = _encode_rows(rows)
    else:
        assert prior_line_by_example_id is not None
        reserve_id = successor_details["reserve"]["example_id"]
        payload = b"".join(
            _encode_rows([row])
            if row["example_id"] == reserve_id
            else prior_line_by_example_id[row["example_id"]]
            for row in rows
        )
    address_audit = None
    address_audit_bytes = None
    successor_plan_bytes = None
    if successor_details is not None:
        from redco.analysis.stage_d_collection import (
            StageDCollectionPlan,
            derive_scientific_group_id,
        )
        from redco.contracts import canonical_json

        prior_plan_bytes = args.prior_collection_plan.read_bytes()
        if _sha256(prior_plan_bytes) != args.prior_collection_plan_sha256:
            raise ValueError("prior collection plan hash mismatch")
        prior_plan = StageDCollectionPlan.from_bytes(prior_plan_bytes)
        support_rows = [row for row in rows if row["split"] == "successor_support"]
        successor_plan = StageDCollectionPlan.build(
            [
                {
                    "scientific_group_id": derive_scientific_group_id(
                        namespace=args.group_namespace,
                        example_id=row["example_id"],
                    ),
                    "example_id": row["example_id"],
                    "rollout_slot": 0,
                }
                for row in support_rows
            ],
            master_seed=args.master_seed,
        )
        prior_slots = {slot.example_id: slot for slot in prior_plan.slots}
        successor_slots = {slot.example_id: slot for slot in successor_plan.slots}
        row_by_id = {row["example_id"]: row for row in support_rows}
        preserved = []
        for retained_row in successor_details["retained"]:
            example_id = retained_row["example_id"]
            old_slot = prior_slots[example_id]
            new_slot = successor_slots[example_id]
            old_address = old_slot.to_payload()
            new_address = new_slot.to_payload()
            row_sha256 = _sha256(prior_line_by_example_id[example_id])
            preserved.append(
                {
                    "example_id": example_id,
                    "paper_id": row_by_id[example_id]["paper_id"],
                    "canonical_row_sha256": row_sha256,
                    **new_address,
                    "matches_prior_row": row_sha256 == retained_row["row_sha256"],
                    "matches_prior_address": new_address == old_address,
                }
            )
        reserve_id = successor_details["reserve"]["example_id"]
        retired_id = successor_details["retired"]["example_id"]
        audit_checks = {
            "prior_plan_has_64_unique_slots": len(prior_plan.slots)
            == len(prior_slots)
            == 64,
            "successor_plan_has_64_unique_slots": len(successor_plan.slots)
            == len(successor_slots)
            == 64,
            "retired_address_absent": retired_id not in successor_slots,
            "reserve_address_fresh": reserve_id not in prior_slots,
            "preserved_63_addresses_exact": len(preserved) == 63
            and all(
                row["matches_prior_address"] and row["matches_prior_row"]
                for row in preserved
            ),
        }
        if not all(audit_checks.values()):
            raise ValueError(f"successor address audit failed: {audit_checks}")
        address_audit = {
            "schema_version": 1,
            "domain": "redco-stage-d-support-successor-address-audit-v1",
            "prior_collection_plan_sha256": args.prior_collection_plan_sha256,
            "successor_collection_plan_sha256": successor_plan.plan_sha256,
            "master_seed_sha256": _sha256(args.master_seed.encode("utf-8")),
            "scientific_group_namespace": args.group_namespace,
            "retired": {
                **successor_details["retired"],
                **prior_slots[retired_id].to_payload(),
            },
            "reserve": {
                **successor_details["reserve"],
                **successor_slots[reserve_id].to_payload(),
            },
            "preserved": preserved,
            "checks": audit_checks,
        }
        address_audit_bytes = canonical_json(address_audit)
        successor_plan_bytes = successor_plan.to_bytes()
    old_ids = {str(row["paper_id"]) for row in old_rows}
    old_refs = {
        str(span)
        for row in old_rows
        for span in row["reference_evidence"]
    }
    new_ids = {str(row["paper_id"]) for row in rows}
    new_refs = {
        str(span) for row in rows for span in row["reference_evidence"]
    }
    split_counts = Counter(str(row["split"]) for row in rows)
    expected_splits = EXTENSION_SPLITS
    checks = {
        "immutable_old_snapshot_retained": _sha256(old_bytes)
        == args.old_snapshot_sha256,
        (
            "support_successor_has_112_unique_papers"
            if successor_details
            else "extension_has_112_unique_papers"
        ): len(rows) == len(new_ids) == sum(expected_splits.values()),
        "split_sizes_exact": split_counts == Counter(expected_splits),
        "old_and_extension_papers_disjoint": not old_ids & new_ids,
        "old_and_extension_references_disjoint": not old_refs & new_refs,
        "extension_references_unique": len(new_refs)
        == sum(len(row["reference_evidence"]) for row in rows),
        "every_reference_exact": all(
            span in row["paper"]
            for row in rows
            for span in row["reference_evidence"]
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"QASPER extension audit failed: {checks}")
    manifest = {
        "schema_version": 1,
        "dataset": "allenai/qasper",
        "source_revision": QASPER_SOURCE_REVISION,
        "converted_parquet_revision": QASPER_PARQUET_REVISION,
        "license": "cc-by-4.0",
        "old_snapshot": {
            "path": args.old_snapshot.as_posix(),
            "sha256": args.old_snapshot_sha256,
            "papers": 120,
            "disposition": "immutable development/retired quarantine",
        },
        "selection": {
            "source_order": True,
            "seed_forbidden_ids_and_spans_from_all_old_120": True,
            "one_question_per_paper": True,
            "maximum_paper_characters": args.maximum_paper_characters,
            "minimum_span_characters": args.minimum_span_characters,
            "fresh_reference_positions_or_midpoint_coverage_inspected": False,
        },
        "partitions": {
            split: {
                "papers": expected,
                "paper_ids": [
                    row["paper_id"] for row in rows if row["split"] == split
                ],
                "answer_types": dict(
                    sorted(
                        Counter(
                            row["answer_type"]
                            for row in rows
                            if row["split"] == split
                        ).items()
                    )
                ),
            }
            for split, expected in expected_splits.items()
        },
        "checks": checks,
        "output": {
            "path": args.output.as_posix(),
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
    }
    if successor_details is not None:
        manifest["mode"] = "support_successor"
        manifest["successor"] = successor_details
        manifest["prior_extension"] = {
            "path": args.prior_extension.as_posix(),
            "sha256": args.prior_extension_sha256,
            "papers": 112,
        }
        manifest["collection_plan"] = {
            "path": args.collection_plan_output.as_posix(),
            "sha256": address_audit["successor_collection_plan_sha256"],
        }
        manifest["address_audit"] = {
            "path": args.address_audit.as_posix(),
            "sha256": _sha256(canonical_json(address_audit)),
        }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    outputs = [(args.output, payload)]
    if successor_details is not None:
        outputs.extend(
            [
                (args.collection_plan_output, successor_plan_bytes),
                (args.address_audit, address_audit_bytes),
            ]
        )
    outputs.append((args.manifest, manifest_bytes))
    _write_fresh_bundle(outputs)


if __name__ == "__main__":
    main()
