from __future__ import annotations

import json
from pathlib import Path

from redco.analysis import stage_c9_partial_recovery


def _rows(delta_five: float, causal_five: float) -> list[dict[str, object]]:
    return [
        {
            "probe_name": "confusion_redundant",
            "case_id": "delta",
            "context_route": "delta",
            "action_probabilities": {
                "4": 1.0 - delta_five,
                "5": delta_five,
            },
        },
        {
            "probe_name": "confusion_redundant",
            "case_id": "alpha",
            "context_route": "alpha",
            "action_probabilities": {
                "4": 1.0 - causal_five,
                "5": causal_five,
            },
        },
    ]


def test_partial_recovery_never_promotes_endpoint_to_auc_gate(
    tmp_path: Path, monkeypatch
) -> None:
    models = [
        {"name": "warmstart", "temperatures": {"2.0": _rows(0.2, 0.2)}}
    ]
    for seed in stage_c9_partial_recovery.SEEDS:
        for arm in stage_c9_partial_recovery.ARMS:
            if arm == "local-e1":
                rows = _rows(0.2, 0.3)
            elif arm == "local-e2":
                rows = _rows(0.2, 0.6)
            elif arm == "branch-global-e2":
                rows = _rows(0.8, 0.6)
            else:
                rows = _rows(0.5, 0.9)
            models.append(
                {
                    "name": f"{arm}--s{seed}--final",
                    "temperatures": {"2.0": rows},
                }
            )
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({"models": models}), encoding="utf-8")
    monkeypatch.setattr(
        stage_c9_partial_recovery,
        "_usage",
        lambda _path: {"policy_calls": 576},
    )
    monkeypatch.setattr(
        stage_c9_partial_recovery,
        "_practical_diagnostics",
        lambda _path, _updates: {},
    )
    monkeypatch.setattr(
        stage_c9_partial_recovery,
        "_reuse_contract",
        lambda _path: {
            "all_pairs_passed": True,
            "fresh_example_stream_between_collections": True,
        },
    )

    result = stage_c9_partial_recovery.evaluate(tmp_path, scores)

    assert result["status"].startswith("terminal_postprocessing_failure")
    assert result["engineering_reuse_and_ledger_pass"]
    assert result["reuse_efficiency_gate"]["status"] == "indeterminate"
    assert result["reuse_efficiency_gate"]["endpoint_gain_component_pass"]
    assert result["checks"][
        "local_e2_auc_exceeds_e1_in_at_least_two_seeds"
    ] is None
    assert result["matched_data_credit_pass"]
