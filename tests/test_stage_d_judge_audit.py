from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.audit_stage_d_judge_calibration import balanced_accuracy
from scripts.audit_stage_d_judge_calibration_v2 import verdict_signature
from scripts.run_stage_d_judge_calibration import render_prompt


def test_balanced_accuracy_weights_both_classes_equally() -> None:
    assert balanced_accuracy(
        [True, True, False, False],
        [True, False, False, False],
    ) == pytest.approx(0.75)


def test_judge_prompt_contains_reference_and_prediction() -> None:
    pytest.importorskip("verifiers")
    rendered = render_prompt(
        "{question}\n{reference}\n{response}\n{criteria}",
        {
            "criteria": [
                {
                    "name": "precision",
                    "text": "No padding.",
                    "choices": ["1", "10"],
                }
            ]
        },
        {
            "question": "What changed?",
            "reference_evidence": ["Latency fell."],
            "response": "['Latency fell.']",
        },
    )
    assert "Latency fell." in rendered
    assert "['Latency fell.']" in rendered
    assert '"verdicts"' in rendered


def test_v2_repeat_signature_ignores_reason_text() -> None:
    first = {
        "verdicts": [
            {"name": "precision", "reason": "first", "verdict": "8"},
            {"name": "recall", "reason": "first", "verdict": "7"},
        ]
    }
    second = {
        "verdicts": [
            {"name": "recall", "reason": "different", "verdict": "7"},
            {"name": "precision", "reason": "different", "verdict": "8"},
        ]
    }
    assert verdict_signature(first) == verdict_signature(second)
