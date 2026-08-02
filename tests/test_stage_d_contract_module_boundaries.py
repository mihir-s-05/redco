from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

import redco.analysis.stage_d_arm_contracts as arm_contracts
import redco.analysis.stage_d_source_contracts as source_contracts
import redco.analysis.stage_d_three_arm_bridge as bridge

SOURCE_FIELDS = {
    "FrozenTrainingSequence": (
        "token_ids",
        "mask",
        "behavior_logprobs",
        "temperatures",
        "rl_weights",
        "rl_normalizer",
    ),
    "DecisionProvenance": (
        "reservation_receipt",
        "completion_receipt",
        "ledger_id",
        "group_id",
        "rollout_id",
        "decision_id",
        "node_kind",
        "target_id",
        "target_ordinal",
        "event_address",
        "branch_selected",
        "target_commitment_receipt_sha256",
        "exact_action_key_digest",
        "action_digest",
        "request_sha256",
        "response_sha256",
        "request_sequence",
        "completion_sequence",
    ),
    "RolloutDecision": (
        "decision_id",
        "event_address",
        "action",
        "node_kind",
        "target_id",
        "target_ordinal",
        "outer_weight",
        "provenance",
    ),
    "SourceRollout": (
        "group_id",
        "rollout_id",
        "reward",
        "stock_sequences",
        "stock_sequence_decision_ids",
        "decisions",
        "child_target_roster",
        "branch_eligible",
        "ineligibility_reason",
        "trace_sha256",
        "reward_evidence_sha256",
        "stock_sequences_evidence_sha256",
        "base_model_manifest_sha256",
        "evidence_class",
        "producer_receipt",
        "source_sha256",
    ),
}
ARM_FIELDS = {
    "ArmTrainerRecord": (
        "arm",
        "record_kind",
        "source_sha256",
        "group_id",
        "rollout_id",
        "decision_id",
        "target_id",
        "action_slot",
        "token_ids",
        "mask",
        "behavior_logprobs",
        "temperatures",
        "advantages",
        "rl_weights",
        "rl_normalizer",
    ),
    "SealedArmBatch": (
        "arm",
        "records",
        "source_sha256s",
        "branch_artifact_sha256s",
        "evidence_class",
        "objective_binding",
        "policy_sha256",
        "trainer_step",
        "seq_len",
        "batch_identity",
    ),
    "ThreeArmCompilation": (
        "stock",
        "branch_global",
        "local",
        "common_branch_layout_sha256",
    ),
}


@pytest.mark.parametrize(
    ("module", "expected"),
    ((source_contracts, SOURCE_FIELDS), (arm_contracts, ARM_FIELDS)),
)
def test_moved_contract_shape_is_frozen(
    module: object, expected: dict[str, tuple[str, ...]]
) -> None:
    for name, field_names in expected.items():
        contract = getattr(module, name)
        assert tuple(field.name for field in fields(contract)) == field_names
        assert contract.__dataclass_params__.frozen is True
        assert "__slots__" in contract.__dict__
    assert tuple(inspect.signature(source_contracts.DecisionProvenance).parameters) == ()
    assert tuple(inspect.signature(source_contracts.SourceRollout).parameters) == ()


def test_bridge_reexports_are_exact_aliases() -> None:
    for name in SOURCE_FIELDS:
        assert getattr(bridge, name) is getattr(source_contracts, name)
    for name in ARM_FIELDS:
        assert getattr(bridge, name) is getattr(arm_contracts, name)
    assert bridge.ArmName is arm_contracts.ArmName
    assert bridge.RecordKind is arm_contracts.RecordKind


@pytest.mark.parametrize(
    "order",
    (
        ("stage_d_source_contracts", "stage_d_arm_contracts", "stage_d_three_arm_bridge"),
        ("stage_d_three_arm_bridge", "stage_d_arm_contracts", "stage_d_source_contracts"),
    ),
)
def test_contract_modules_are_import_order_independent(order: tuple[str, ...]) -> None:
    imports = ";".join(f"import redco.analysis.{name}" for name in order)
    subprocess.run([sys.executable, "-c", imports], check=True)


def test_contract_dependency_boundaries_are_acyclic() -> None:
    analysis_root = Path(source_contracts.__file__).parent
    forbidden = {
        "stage_d_source_contracts.py": {
            "redco.analysis.stage_d_three_arm_bridge",
            "redco.analysis.stage_d_arm_contracts",
        },
        "stage_d_arm_contracts.py": {
            "redco.analysis.stage_d_three_arm_bridge",
            "redco.analysis.stage_d_source_contracts",
            "redco.analysis.stage_d_source_producer",
        },
    }
    for filename, blocked in forbidden.items():
        tree = ast.parse((analysis_root / filename).read_text(encoding="utf-8"))
        top_level_imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert top_level_imports.isdisjoint(blocked)
