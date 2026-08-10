from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.audit_stage_d_judge_calibration import balanced_accuracy
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
