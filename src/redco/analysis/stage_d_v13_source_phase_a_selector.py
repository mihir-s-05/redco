"""One canonical QASPER eligibility/collision selector for Phase A.

The implementation is the executable projection of
``scripts/build_stage_d_qasper_extension_v1.py``: rendering, exact-evidence
normalization, answer-type ordering, and first-question traversal are kept in
one owner and are reused by the historical builder and the Phase-A audit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from redco.analysis.stage_d_v13_draft import sha256_bytes
from redco.analysis.stage_d_v13_source_phase_a_decoder import canonical_source_row_bytes

MAXIMUM_PAPER_CHARACTERS = 60_000
MINIMUM_SPAN_CHARACTERS = 20
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
CollisionClass = Literal[
    "paper_id_collision",
    "example_id_collision",
    "rendered_paper_collision",
    "reference_span_collision",
    "source_address_collision",
    "source_row_collision",
]
CONTINUABLE_COLLISIONS = frozenset(
    {"paper_id_collision", "reference_span_collision"}
)
TERMINAL_COLLISIONS = frozenset(
    {
        "example_id_collision",
        "rendered_paper_collision",
        "source_address_collision",
        "source_row_collision",
    }
)
COLLISION_CLASS_ORDER: tuple[CollisionClass, ...] = (
    "paper_id_collision",
    "reference_span_collision",
    "example_id_collision",
    "rendered_paper_collision",
    "source_row_collision",
    "source_address_collision",
)
TERMINAL_COLLISION_ORDER: tuple[CollisionClass, ...] = (
    "example_id_collision",
    "rendered_paper_collision",
    "source_row_collision",
    "source_address_collision",
)


@dataclass(frozen=True, slots=True)
class CollisionClassification:
    """Complete, deterministic collision evidence for one candidate question."""

    collision_set: tuple[CollisionClass, ...]
    primary_terminal: CollisionClass | None

    @property
    def continuable(self) -> bool:
        return bool(self.collision_set) and self.primary_terminal is None

    @property
    def empty(self) -> bool:
        return not self.collision_set


class TerminalIdentityCollision(ValueError):
    """A collision whose frozen law retires the source scan immediately."""

    collision_class: CollisionClass
    collision_set: tuple[CollisionClass, ...]

    def __init__(self, classification: CollisionClassification) -> None:
        if classification.primary_terminal is None:
            raise ValueError("terminal collision requires a terminal classification")
        self.collision_class = classification.primary_terminal
        self.collision_set = classification.collision_set
        suffix = "" if len(self.collision_set) == 1 else (
            f"; set={','.join(self.collision_set)}"
        )
        super().__init__(
            "terminal identity collision: "
            f"{self.collision_class}{suffix}"
        )


def normalize_digest_values(values: Iterable[str], *, field: str) -> tuple[str, ...]:
    """Normalize digest collections without accidentally iterating one string."""

    if isinstance(values, str):
        raise ValueError(f"{field} must be a collection of digest strings")
    normalized = tuple(sorted(set(values)))
    if any(
        not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None for value in normalized
    ):
        raise ValueError(f"{field} contains a non-canonical SHA-256 digest")
    return normalized


def render_paper(row: Mapping[str, Any]) -> str:
    parts = [
        f"### PAPER: {row['title']}",
        "<abstract>",
        row["abstract"],
        "</abstract>",
    ]
    sections = cast(Mapping[str, Any], row["full_text"])
    for name, paragraphs in zip(
        cast(Iterable[str], sections["section_name"]),
        cast(Iterable[Iterable[str]], sections["paragraphs"]),
        strict=True,
    ):
        parts.append(f"\n## {name}")
        parts.extend(paragraphs)
    return "\n".join(parts)


def _answer_type(annotation: Mapping[str, Any]) -> str:
    if annotation.get("yes_no") is not None:
        return "yes_no"
    if annotation.get("extractive_spans"):
        return "extractive"
    if annotation.get("free_form_answer"):
        return "abstractive"
    return "other"


def exact_reference(
    paper: str,
    answers: Mapping[str, Any],
    *,
    minimum_span_characters: int = MINIMUM_SPAN_CHARACTERS,
) -> tuple[tuple[str, ...], str] | None:
    candidates: list[tuple[tuple[str, ...], str]] = []
    for annotation in cast(Iterable[Mapping[str, Any]], answers["answer"]):
        if annotation["unanswerable"]:
            continue
        evidence = tuple(
            dict.fromkeys(
                span.strip()
                for span in cast(Iterable[str], annotation["evidence"])
                if len(span.strip()) >= minimum_span_characters and span.strip() in paper
            )
        )
        if evidence:
            candidates.append((evidence, _answer_type(annotation)))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (sum(map(len, item[0])), len(item[0]), item[1]))


def classify_candidate_collisions(
    *,
    row: Mapping[str, Any],
    example_id: str,
    paper: str,
    evidence: Sequence[str],
    forbidden_paper_ids: set[str],
    forbidden_example_ids: set[str],
    forbidden_rendered_paper_sha256: set[str],
    forbidden_reference_spans: set[str],
    forbidden_row_sha256: set[str],
    candidate_address_sha256: str | None,
    forbidden_address_sha256: set[str],
) -> CollisionClassification:
    collisions: dict[CollisionClass, bool] = {
        "paper_id_collision": str(row["id"]) in forbidden_paper_ids,
        "reference_span_collision": any(
            span in forbidden_reference_spans for span in evidence
        ),
        "example_id_collision": example_id in forbidden_example_ids,
        "rendered_paper_collision": False,
        "source_row_collision": False,
        "source_address_collision": False,
    }
    rendered_digest = sha256_bytes(paper.encode("utf-8"))
    collisions["rendered_paper_collision"] = rendered_digest in forbidden_rendered_paper_sha256
    collisions["source_address_collision"] = (
        candidate_address_sha256 is not None
        and candidate_address_sha256 in forbidden_address_sha256
    )
    row_digest = sha256_bytes(canonical_source_row_bytes(row))
    collisions["source_row_collision"] = row_digest in forbidden_row_sha256
    collision_set = tuple(
        collision_class
        for collision_class in COLLISION_CLASS_ORDER
        if collisions[collision_class]
    )
    primary_terminal = next(
        (
            collision_class
            for collision_class in TERMINAL_COLLISION_ORDER
            if collision_class in collision_set
        ),
        None,
    )
    return CollisionClassification(
        collision_set=collision_set,
        primary_terminal=primary_terminal,
    )


_candidate_collision = classify_candidate_collisions


def select_first_eligible(
    row: Mapping[str, Any],
    *,
    split: str = "successor_support",
    maximum_paper_characters: int = MAXIMUM_PAPER_CHARACTERS,
    minimum_span_characters: int = MINIMUM_SPAN_CHARACTERS,
    forbidden_paper_ids: set[str] | None = None,
    forbidden_example_ids: set[str] | None = None,
    forbidden_rendered_paper_sha256: Iterable[str] = (),
    forbidden_reference_spans: set[str] | None = None,
    forbidden_row_sha256: set[str] | None = None,
    forbidden_address_sha256: Iterable[str] = (),
    candidate_address_sha256: str | None = None,
) -> tuple[dict[str, Any], int] | None:
    """Apply the frozen historical selector and complete collision universe."""

    paper_id_forbidden = forbidden_paper_ids or set()
    example_id_forbidden = forbidden_example_ids or set()
    reference_forbidden = forbidden_reference_spans or set()
    row_forbidden = forbidden_row_sha256 or set()
    rendered_forbidden = set(
        normalize_digest_values(forbidden_rendered_paper_sha256, field="rendered_paper_sha256")
    )
    address_forbidden = set(
        normalize_digest_values(forbidden_address_sha256, field="source_address_sha256")
    )
    paper = render_paper(row)
    if len(paper) > maximum_paper_characters:
        return None
    qas = cast(Mapping[str, Any], row["qas"])
    questions = cast(Iterable[str], qas["question"])
    answers = cast(Iterable[Mapping[str, Any]], qas["answers"])
    question_ids = cast(Iterable[str], qas["question_id"])
    for index, (question, answer, question_id) in enumerate(
        zip(questions, answers, question_ids, strict=True)
    ):
        reference = exact_reference(
            paper,
            answer,
            minimum_span_characters=minimum_span_characters,
        )
        if reference is None:
            continue
        evidence, kind = reference
        example_id = f"qasper-{question_id}"
        collision = classify_candidate_collisions(
            row=row,
            example_id=example_id,
            paper=paper,
            evidence=evidence,
            forbidden_paper_ids=paper_id_forbidden,
            forbidden_example_ids=example_id_forbidden,
            forbidden_rendered_paper_sha256=rendered_forbidden,
            forbidden_reference_spans=reference_forbidden,
            forbidden_row_sha256=row_forbidden,
            candidate_address_sha256=candidate_address_sha256,
            forbidden_address_sha256=address_forbidden,
        )
        if not collision.empty:
            if collision.continuable:
                continue
            raise TerminalIdentityCollision(collision)
        return (
            {
                "example_id": example_id,
                "paper_id": row["id"],
                "title": row["title"],
                "question": question,
                "answer_type": kind,
                "split": split,
                "paper": paper,
                "reference_evidence": list(evidence),
            },
            index,
        )
    return None


def selector_decision(
    row: Mapping[str, Any],
    *,
    forbidden_paper_ids: set[str],
    forbidden_example_ids: set[str],
    forbidden_rendered_paper_sha256: Iterable[str],
    forbidden_reference_spans: set[str],
    forbidden_row_sha256: set[str],
    forbidden_address_sha256: Iterable[str],
    candidate_address_sha256: str | None = None,
) -> str:
    selected = select_first_eligible(
        row,
        forbidden_paper_ids=forbidden_paper_ids,
        forbidden_example_ids=forbidden_example_ids,
        forbidden_rendered_paper_sha256=forbidden_rendered_paper_sha256,
        forbidden_reference_spans=forbidden_reference_spans,
        forbidden_row_sha256=forbidden_row_sha256,
        forbidden_address_sha256=forbidden_address_sha256,
        candidate_address_sha256=candidate_address_sha256,
    )
    if selected is None:
        return "reject_no_exact_evidence_or_authenticated_collision"
    return "eligible_not_materialized_phase_a"


def derivation_golden_vectors() -> dict[str, Any]:
    """Return the frozen scientific group/seed law and its deterministic vector."""

    from redco.analysis.stage_d_collection import (
        SourceCollectionSlot,
        derive_scientific_group_id,
        derive_source_episode_seed_and_salt,
    )

    namespace = "redco-stage-d1-support-v1"
    example_id = "qasper-f33236ebd6f5a9ccb9b9dbf05ac17c3724f93f91"
    group_id = derive_scientific_group_id(namespace=namespace, example_id=example_id)
    seed, cache_salt = derive_source_episode_seed_and_salt(
        master_seed="redco-stage-d1-support-v1-20260802-78b65e4cc16ac31f",
        scientific_group_id=group_id,
        rollout_slot=0,
    )
    slot = SourceCollectionSlot.build(
        {
            "scientific_group_id": group_id,
            "example_id": example_id,
            "rollout_slot": 0,
        },
        master_seed="redco-stage-d1-support-v1-20260802-78b65e4cc16ac31f",
    )
    return {
        "namespace": namespace,
        "master_seed": "redco-stage-d1-support-v1-20260802-78b65e4cc16ac31f",
        "example_id": example_id,
        "group_id": group_id,
        "rollout_slot": 0,
        "seed": seed,
        "cache_salt": cache_salt,
        "slot_id": slot.slot_id,
        "group_domain": "redco-stage-d-scientific-group-v1",
        "seed_domain": "redco-stage-d-source-episode-seed-v1",
        "slot_domain": "redco-stage-d-source-slot-v1",
        "hmac": {
            "algorithm": "HMAC-SHA256",
            "key": "master_seed",
            "seed_bytes": "first_8_big_endian_mod_2^31",
            "cache_salt_prefix": "stage-d-source-",
        },
        "canonical_json": {
            "sort_keys": True,
            "ensure_ascii": False,
            "allow_nan": False,
            "trailing_newline": False,
        },
    }


__all__ = [
    "CONTINUABLE_COLLISIONS",
    "MAXIMUM_PAPER_CHARACTERS",
    "MINIMUM_SPAN_CHARACTERS",
    "TERMINAL_COLLISIONS",
    "TerminalIdentityCollision",
    "derivation_golden_vectors",
    "exact_reference",
    "normalize_digest_values",
    "render_paper",
    "select_first_eligible",
    "selector_decision",
]
