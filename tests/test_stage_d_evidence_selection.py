from __future__ import annotations

import sys
from pathlib import Path

ENV_ROOT = (
    Path(__file__).parents[1]
    / "environments"
    / "redco_evidence_selection_v1"
)
sys.path.insert(0, str(ENV_ROOT))

from redco_evidence_selection_v1.scoring import (  # noqa: E402
    parse_evidence,
    score_exact_spans,
)


def test_safe_parser_accepts_only_literal_string_lists() -> None:
    assert parse_evidence("['one', 'two']").spans == ("one", "two")
    assert parse_evidence("'one'").spans == ("one",)
    assert not parse_evidence("__import__('os').system('echo nope')").parseable
    assert not parse_evidence("{'span': 'one'}").parseable


def test_exact_span_score_uses_character_union() -> None:
    paper = "prefix alpha beta gamma suffix"
    result = score_exact_spans(
        paper,
        ["alpha beta", "beta gamma"],
        ["alpha beta gamma"],
    )
    assert result == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "exact_substring_fraction": 1.0,
    }


def test_nonverbatim_predictions_do_not_receive_credit() -> None:
    result = score_exact_spans(
        "The measured latency was 260 milliseconds.",
        ["Latency fell to 260 ms."],
        ["The measured latency was 260 milliseconds."],
    )
    assert result["f1"] == 0.0
    assert result["exact_substring_fraction"] == 0.0
