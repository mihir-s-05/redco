import json
from pathlib import Path

from scripts.audit_stage_c3_v2_preregistration import audit


def test_audit_rejects_changed_decision_rule(tmp_path: Path) -> None:
    v1 = {
        "frozen_metrics_and_decision": {"threshold": 0.75},
        "design": {"runs": [{"seed": 9301}]},
    }
    v2 = {
        "frozen_metrics_and_decision": {"threshold": 0.76},
        "decision_rule_identity": {
            "v1_canonical_sha256": "wrong",
            "v2_canonical_sha256": "wrong",
        },
        "design": {
            "runs": [
                {"seed": 9401},
                {"seed": 9402},
                {"seed": 9403},
                {"seed": 9404},
            ]
        },
        "v1_terminal_record": {
            "bundle_sha256": (
                "1a4c239da6c878a1513964be2041dcd9e73066cc567456de8c896d5b8ddb13d9"
            ),
            "scientific_gate_evaluated": False,
        },
        "execution": {
            "smoke": {"position": "before_every_scientific_arm"},
            "early_abort": {"scope": "smoke_and_all_eight_arms"},
        },
        "source": {"sha256": {}},
    }
    v1_path = tmp_path / "v1.json"
    v2_path = tmp_path / "v2.json"
    v1_path.write_text(json.dumps(v1), encoding="utf-8")
    v2_path.write_text(json.dumps(v2), encoding="utf-8")

    result = audit(v1_path, v2_path)

    assert result["passed"] is False
    assert result["checks"]["decision_rules_byte_identical_to_v1"] is False
