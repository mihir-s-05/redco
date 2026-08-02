"""Paper-level support decision over verified Stage-D replay artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast

from redco.analysis.stage_d_branch_artifacts import (
    StageDBranchTarget,
    StageDBranchTargetRoster,
)
from redco.analysis.stage_d_scientific_branch_group import BranchGroupArtifact
from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.contracts import canonical_json


@dataclass(frozen=True, slots=True)
class StageDSupportRules:
    required_papers: int
    required_successes: int
    minimum_targets: int
    maximum_targets: int
    minimum_reward_range: float
    rules_sha256: str

    @classmethod
    def from_bytes(cls, value: bytes) -> StageDSupportRules:
        payload = json.loads(value)
        fields = {
            "schema_version",
            "domain",
            "required_papers",
            "required_successes",
            "minimum_targets",
            "maximum_targets",
            "minimum_reward_range",
        }
        if (
            not isinstance(payload, dict)
            or canonical_json(payload) != value
            or set(payload) != fields
            or payload.get("schema_version") != 1
            or payload.get("domain") != "redco-stage-d-support-rules-v1"
        ):
            raise ValueError("support rules differ from the frozen schema")
        rules = cls(
            payload["required_papers"],
            payload["required_successes"],
            payload["minimum_targets"],
            payload["maximum_targets"],
            payload["minimum_reward_range"],
            hashlib.sha256(value).hexdigest(),
        )
        if any(
            type(number) is not int
            for number in (
                rules.required_papers,
                rules.required_successes,
                rules.minimum_targets,
                rules.maximum_targets,
            )
        ) or type(rules.minimum_reward_range) not in {int, float}:
            raise ValueError("support rule numbers have invalid types")
        if not 1 <= rules.required_successes <= rules.required_papers:
            raise ValueError("support success floor is outside the paper denominator")
        if not 1 <= rules.minimum_targets <= rules.maximum_targets:
            raise ValueError("support target bounds are invalid")
        if rules.minimum_reward_range <= 0:
            raise ValueError("support reward range must be positive")
        return rules


def load_support_rules(path: Path, expected_sha256: str) -> StageDSupportRules:
    value = path.read_bytes()
    if hashlib.sha256(value).hexdigest() != expected_sha256:
        raise ValueError("support rules differ from the protocol manifest")
    return StageDSupportRules.from_bytes(value)


def evaluate_support_gate(
    sources: Sequence[SourceRollout],
    artifacts: Sequence[BranchGroupArtifact],
    roster: StageDBranchTargetRoster,
    *,
    paper_ids: Mapping[str, str],
    rules: StageDSupportRules,
) -> bytes:
    """Evaluate frozen rules without reinterpreting source or branch evidence."""
    if (
        len(sources) != rules.required_papers
        or roster.planned_source_count != rules.required_papers
    ):
        raise ValueError("support source count differs from the frozen denominator")
    source_by_sha = {source.source_sha256: source for source in sources}
    if len(source_by_sha) != len(sources):
        raise ValueError("support sources are not unique")
    if tuple(sorted(source_by_sha)) != roster.source_sha256s:
        raise ValueError("support sources differ from the committed target roster")
    if set(paper_ids) != set(source_by_sha):
        raise ValueError("support paper identities differ from the source roster")
    if any(not paper_id for paper_id in paper_ids.values()) or len(
        set(paper_ids.values())
    ) != len(sources):
        raise ValueError("support inferential units are not unique papers")

    targets_by_source: dict[str, list[StageDBranchTarget]] = defaultdict(list)
    for target in roster.targets:
        targets_by_source[target.source_sha256].append(target)
    artifact_by_key = {
        (artifact.commitment.group_id, artifact.commitment.target_id): artifact
        for artifact in artifacts
    }
    roster_keys = {(target.group_id, target.target_id) for target in roster.targets}
    if len(artifact_by_key) != len(artifacts) or set(artifact_by_key) != roster_keys:
        raise ValueError("support artifacts differ from the complete committed target set")

    papers = []
    for source in sorted(sources, key=lambda item: item.source_sha256):
        targets = targets_by_source[source.source_sha256]
        target_rows = []
        outer_weight = Fraction(0)
        for target in targets:
            artifact = artifact_by_key[(target.group_id, target.target_id)]
            commitment = artifact.commitment
            artifact_address = {
                **commitment.target_address.as_payload(),
                "turn": commitment.target_address.turn,
            }
            if (
                commitment.group_id != source.group_id
                or commitment.rollout_id != source.rollout_id
                or commitment.target_id != target.target_id
                or commitment.target_ordinal != target.target_ordinal
                or artifact_address != target.event_address
            ):
                raise ValueError("support artifact differs from its committed source target")
            rewards = [arm.q_value for arm in artifact.arms]
            reward_range = max(rewards) - min(rewards)
            outer_weight += commitment.outer_weight
            target_rows.append(
                {
                    "artifact_sha256": hashlib.sha256(artifact.to_bytes()).hexdigest(),
                    "target_id": target.target_id,
                    "target_ordinal": target.target_ordinal,
                    "reconstruction_qa_passed": artifact.reconstruction_qa.passed,
                    "reward_range": reward_range,
                    "informative": reward_range >= rules.minimum_reward_range,
                    "outer_weight": {
                        "numerator": commitment.outer_weight.numerator,
                        "denominator": commitment.outer_weight.denominator,
                    },
                }
            )
        target_count_ok = rules.minimum_targets <= len(targets) <= rules.maximum_targets
        all_exact = bool(targets) and all(
            row["reconstruction_qa_passed"] for row in target_rows
        )
        informative = any(row["informative"] for row in target_rows)
        paper_success = (
            source.branch_eligible
            and target_count_ok
            and all_exact
            and informative
            and outer_weight == 1
        )
        papers.append(
            {
                "source_sha256": source.source_sha256,
                "paper_id": paper_ids[source.source_sha256],
                "group_id": source.group_id,
                "rollout_id": source.rollout_id,
                "branch_eligible": source.branch_eligible,
                "target_count": len(targets),
                "target_count_ok": target_count_ok,
                "all_targets_exact": all_exact,
                "has_informative_target": informative,
                "outer_weight_sum": {
                    "numerator": outer_weight.numerator,
                    "denominator": outer_weight.denominator,
                },
                "success": paper_success,
                "targets": target_rows,
            }
        )

    successes = sum(1 for row in papers if row["success"] is True)
    unsigned = {
        "schema_version": 1,
        "analysis": "stage-d-state-aware-support-gate-v1",
        "inferential_unit": "unique-paper",
        "rules_sha256": rules.rules_sha256,
        "required_papers": rules.required_papers,
        "required_successes": rules.required_successes,
        "minimum_targets_per_paper": rules.minimum_targets,
        "maximum_targets_per_paper": rules.maximum_targets,
        "minimum_reward_range": rules.minimum_reward_range,
        "paper_successes": successes,
        "paper_failures": rules.required_papers - successes,
        "decision": "pass" if successes >= rules.required_successes else "fail",
        "papers": papers,
    }
    return canonical_json(
        {
            **unsigned,
            "signed_payload_sha256": hashlib.sha256(canonical_json(unsigned)).hexdigest(),
        }
    )


def verify_support_pass(
    report: bytes,
    *,
    expected_rules_sha256: str,
    source_sha256s: Sequence[str],
    artifact_sha256s: Sequence[str],
) -> str:
    payload = json.loads(report)
    if not isinstance(payload, dict) or canonical_json(payload) != report:
        raise ValueError("support report is not canonical JSON")
    signature = payload.get("signed_payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "signed_payload_sha256"}
    papers = payload.get("papers")
    report_sources = (
        [paper.get("source_sha256") for paper in papers]
        if isinstance(papers, list) and all(isinstance(paper, dict) for paper in papers)
        else []
    )
    report_paper_ids = (
        [paper.get("paper_id") for paper in papers]
        if isinstance(papers, list) and all(isinstance(paper, dict) for paper in papers)
        else []
    )
    report_artifacts = [
        target.get("artifact_sha256")
        for paper in papers if isinstance(paper, dict)
        for target in paper.get("targets", []) if isinstance(target, dict)
    ] if isinstance(papers, list) else []
    if any(
        not isinstance(value, str) or not value
        for value in (*report_sources, *report_paper_ids, *report_artifacts)
    ):
        raise RuntimeError("frozen Stage-D support gate has malformed evidence identities")
    typed_sources = cast(list[str], report_sources)
    typed_paper_ids = cast(list[str], report_paper_ids)
    typed_artifacts = cast(list[str], report_artifacts)
    required_papers = payload.get("required_papers")
    required_successes = payload.get("required_successes")
    successes = (
        sum(paper.get("success") is True for paper in papers)
        if isinstance(papers, list)
        else 0
    )
    reported_rules = canonical_json(
        {
            "schema_version": 1,
            "domain": "redco-stage-d-support-rules-v1",
            "required_papers": required_papers,
            "required_successes": required_successes,
            "minimum_targets": payload.get("minimum_targets_per_paper"),
            "maximum_targets": payload.get("maximum_targets_per_paper"),
            "minimum_reward_range": payload.get("minimum_reward_range"),
        }
    )
    if (
        payload.get("decision") != "pass"
        or payload.get("rules_sha256") != expected_rules_sha256
        or signature != hashlib.sha256(canonical_json(unsigned)).hexdigest()
        or sorted(typed_sources) != sorted(source_sha256s)
        or len(typed_sources) != len(set(typed_sources))
        or len(typed_paper_ids) != len(typed_sources)
        or len(typed_paper_ids) != len(set(typed_paper_ids))
        or sorted(typed_artifacts) != sorted(artifact_sha256s)
        or len(typed_artifacts) != len(set(typed_artifacts))
        or type(required_papers) is not int
        or type(required_successes) is not int
        or len(typed_sources) != required_papers
        or not 1 <= required_successes <= required_papers
        or successes < required_successes
        or payload.get("paper_successes") != successes
        or payload.get("paper_failures") != required_papers - successes
        or hashlib.sha256(reported_rules).hexdigest() != expected_rules_sha256
    ):
        raise RuntimeError("frozen Stage-D support gate failed authentication")
    return hashlib.sha256(report).hexdigest()
