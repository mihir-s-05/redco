"""Build fresh answer-blind complementary-evidence Stage D fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from redco.contracts import canonical_json  # noqa: E402

SHARD_CHARS = 2000
FIXTURES = (
    (
        "adaptive-cache",
        "What changed in latency, and what happened to exact-answer accuracy?",
        "Adaptive caching reduced median latency from 0420 ms to 0260 ms.",
        "Exact-answer accuracy remained 081.4 percent after deployment.",
    ),
    (
        "branch-allocation",
        "How did success change, and what happened to rollout time?",
        "Directed allocation increased held-out success from 0037 to 0046 percent.",
        "The ninety-fifth percentile rollout time increased by 0012 percent.",
    ),
    (
        "typed-replay",
        "How many interventions were tested, and how many disagreements occurred?",
        "The evaluation covered exactly 10000 deterministic interventions.",
        "The observed full-versus-sliced disagreement count was exactly 00000.",
    ),
    (
        "retrieval-index",
        "How did index size and query time change?",
        "The compressed index size fell from 0840 MB to 0520 MB.",
        "Median query time increased from 0031 ms to 0037 ms.",
    ),
    (
        "batch-scheduler",
        "What happened to throughput and tail latency?",
        "Scheduler throughput rose from 0128 to 0176 requests per second.",
        "The ninety-ninth percentile latency rose from 0210 ms to 0248 ms.",
    ),
    (
        "evidence-filter",
        "How did precision and recall change?",
        "Evidence precision increased from 067.0 to 079.0 percent.",
        "Evidence recall decreased from 088.0 to 083.0 percent.",
    ),
    (
        "checkpoint-codec",
        "What happened to checkpoint size and load time?",
        "Checkpoint size decreased from 1600 MB to 0940 MB.",
        "Median checkpoint load time increased from 0018 s to 0022 s.",
    ),
    (
        "token-router",
        "How did routing accuracy and compute change?",
        "Routing accuracy increased from 072.0 to 078.5 percent.",
        "Mean compute per request increased from 0041 to 0048 GFLOP.",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shard(label: str, evidence: str) -> str:
    prefix = f"### SECTION {label}\n"
    filler = (
        "Background measurements were collected under a fixed protocol. "
        "This neutral sentence carries no answer to the paired question. "
    )
    before_target = 900
    before = (prefix + filler * 20)[:before_target]
    if len(before) < before_target:
        before += "x" * (before_target - len(before))
    body = before + evidence + "\n"
    tail = (filler * 30)[: SHARD_CHARS - len(body)]
    body += tail
    if len(body) < SHARD_CHARS:
        body += "x" * (SHARD_CHARS - len(body))
    if len(body) != SHARD_CHARS:
        raise ValueError("fixture shard length is not exact")
    return body


def _order_for(fixture_id: str) -> tuple[int, int]:
    # The order depends only on the public fixture ID, never on answer text,
    # reference fields, reward, model behavior, or a relevance score.
    first = hashlib.sha256(fixture_id.encode("utf-8")).digest()[0] % 2
    return (first, 1 - first)


def build(dataset_path: Path, manifest_path: Path) -> None:
    rows = []
    audit_rows = []
    for index, (slug, question, first_fact, second_fact) in enumerate(FIXTURES):
        fixture_id = f"complementary-{index:03d}-{slug}"
        facts = (first_fact, second_fact)
        order = _order_for(fixture_id)
        shards = [
            _shard(f"{slot + 1:02d}", facts[fact_index])
            for slot, fact_index in enumerate(order)
        ]
        paper = "".join(shards)
        references = [facts[fact_index] for fact_index in order]
        positions = [paper.index(reference) for reference in references]
        row = {
            "example_id": fixture_id,
            "paper_id": fixture_id,
            "title": slug.replace("-", " ").title(),
            "question": question,
            "split": "successor_fixture",
            "paper": paper,
            "reference_evidence": references,
            "answer_type": "extractive",
        }
        rows.append(row)
        audit_rows.append(
            {
                "fixture_id": fixture_id,
                "shard_order": list(order),
                "paper_chars": len(paper),
                "shard_chars": [len(shard) for shard in shards],
                "reference_positions": positions,
                "references_per_shard": [
                    sum(
                        start <= position < start + SHARD_CHARS
                        for position in positions
                    )
                    for start in (0, SHARD_CHARS)
                ],
                "each_shard_alone_insufficient": all(
                    sum(
                        start <= position < start + SHARD_CHARS
                        for position in positions
                    )
                    == 1
                    for start in (0, SHARD_CHARS)
                ),
                "union_sufficient": all(position >= 0 for position in positions),
            }
        )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in rows)
    )
    checks = {
        "eight_fresh_fixture_ids": len({row["example_id"] for row in rows}) == 8,
        "all_shards_exactly_2000_chars": all(
            audit["shard_chars"] == [SHARD_CHARS, SHARD_CHARS]
            for audit in audit_rows
        ),
        "one_reference_per_shard": all(
            audit["references_per_shard"] == [1, 1]
            for audit in audit_rows
        ),
        "each_shard_alone_insufficient": all(
            audit["each_shard_alone_insufficient"] for audit in audit_rows
        ),
        "union_sufficient": all(audit["union_sufficient"] for audit in audit_rows),
        "both_orderings_present": {
            tuple(audit["shard_order"]) for audit in audit_rows
        }
        == {(0, 1), (1, 0)},
    }
    if not all(checks.values()):
        raise ValueError(f"complementary fixture audit failed: {checks}")
    manifest = {
        "schema_version": 1,
        "generator": "scripts/build_stage_d_complementary_fixtures_v1.py",
        "dataset": dataset_path.as_posix(),
        "dataset_sha256": _sha256(dataset_path),
        "fixtures": len(rows),
        "partition_contract": {
            "shards_per_paper": 2,
            "characters_per_shard": SHARD_CHARS,
            "ordering_input": "public fixture_id only",
            "ordering_forbidden_inputs": [
                "answer text",
                "reference field",
                "reward",
                "model output",
                "relevance score",
            ],
            "model_visible_reference_fields": False,
        },
        "checks": checks,
        "rows": audit_rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    build(args.dataset, args.manifest)


if __name__ == "__main__":
    main()
