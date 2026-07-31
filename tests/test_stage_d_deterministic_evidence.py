from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ENV_ROOT = (
    Path(__file__).parents[1]
    / "environments"
    / "redco_evidence_selection_v2"
)
sys.path.insert(0, str(ENV_ROOT))

from redco_evidence_selection_v2.scoring import (  # noqa: E402
    parse_evidence,
    score_evidence_reply,
    score_exact_spans,
)
from redco_evidence_selection_v2.seeding import (  # noqa: E402
    derive_episode_seed,
)

PAPER = (
    "Alpha result improved by 20 percent. "
    "Noise was unrelated. "
    "Alpha result improved by 20 percent."
)
REFERENCE = ("Alpha result improved by 20 percent.",)


def test_parser_requires_a_literal_list_and_preserves_empty_strings() -> None:
    assert parse_evidence("['exact']").spans == ("exact",)
    assert parse_evidence("['']").spans == ("",)
    assert not parse_evidence("'exact'").parseable
    assert not parse_evidence("('exact',)").parseable
    assert not parse_evidence("__import__('os').system('echo no')").parseable


def test_exact_complete_evidence_scores_one() -> None:
    score = score_evidence_reply(PAPER, repr(list(REFERENCE)), REFERENCE)
    assert score["f1"] == 1.0
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0


def test_hallucinated_or_empty_extra_span_zeroes_reward() -> None:
    hallucinated = score_evidence_reply(
        PAPER,
        repr([REFERENCE[0], "This string is not in the paper."]),
        REFERENCE,
    )
    empty = score_evidence_reply(PAPER, repr([REFERENCE[0], ""]), REFERENCE)
    assert hallucinated["f1"] == 0.0
    assert hallucinated["exact_substring_fraction"] == 0.5
    assert empty["f1"] == 0.0


def test_verbatim_padding_and_whole_paper_are_penalized() -> None:
    padded = score_exact_spans(
        PAPER,
        [REFERENCE[0], "Noise was unrelated."],
        REFERENCE,
    )
    whole = score_exact_spans(PAPER, [PAPER], REFERENCE)
    assert 0.0 < padded["precision"] < 1.0
    assert 0.0 < padded["f1"] < 1.0
    assert 0.0 < whole["precision"] < 1.0


def test_single_word_cannot_receive_full_recall() -> None:
    score = score_exact_spans(PAPER, ["Alpha"], REFERENCE)
    assert score["recall"] < 1.0
    assert score["f1"] < 1.0


def test_split_overlapping_and_duplicate_spans_use_character_unions() -> None:
    split = score_exact_spans(
        PAPER,
        ["Alpha result improved", " by 20 percent."],
        REFERENCE,
    )
    overlapping = score_exact_spans(
        PAPER,
        [REFERENCE[0], "result improved by 20 percent."],
        REFERENCE,
    )
    duplicate = score_exact_spans(
        PAPER,
        [REFERENCE[0], REFERENCE[0]],
        REFERENCE,
    )
    assert split["f1"] == 1.0
    assert overlapping["f1"] == 1.0
    assert duplicate["f1"] == 1.0
    assert duplicate["predicted_span_count"] == 2.0


def test_unparseable_empty_prediction_and_empty_reference_score_zero() -> None:
    assert score_evidence_reply(PAPER, "FINAL(nope)", REFERENCE)["f1"] == 0.0
    assert score_evidence_reply(PAPER, "[]", REFERENCE)["f1"] == 0.0
    assert score_exact_spans(PAPER, [REFERENCE[0]], [])["f1"] == 0.0


def test_repeated_reference_text_has_deterministic_occurrence_semantics() -> None:
    score = score_exact_spans(PAPER, [REFERENCE[0]], REFERENCE)
    assert score["f1"] == 1.0


def test_prompt_profiles_separate_science_from_trace_fixture(
    tmp_path: Path,
) -> None:
    pytest.importorskip("verifiers.v1")
    from redco_evidence_selection_v2.taskset import (
        EvidenceSelectionConfig,
        EvidenceSelectionTaskset,
    )

    row = {
        "example_id": "fixture-1",
        "paper_id": "paper-1",
        "title": "Fixture",
        "question": "What is the evidence?",
        "paper": PAPER,
        "reference_evidence": list(REFERENCE),
        "answer_type": "extractive",
        "split": "train",
    }
    dataset = tmp_path / "fixture.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()

    natural = EvidenceSelectionTaskset(
        EvidenceSelectionConfig(
            dataset_path=dataset,
            dataset_sha256=digest,
            split="train",
            prompt_profile="natural",
        )
    ).load()[0]
    fixture = EvidenceSelectionTaskset(
        EvidenceSelectionConfig(
            dataset_path=dataset,
            dataset_sha256=digest,
            split="train",
            prompt_profile="forced_trace_fixture",
        )
    ).load()[0]

    assert "call exactly two `rlm(...)` children" not in natural.data.prompt
    assert "minimum child" not in natural.data.prompt.lower()
    assert "call exactly two `rlm(...)` children" in fixture.data.prompt
    assert "excluded from scientific feasibility metrics" in fixture.data.prompt


def test_episode_seed_uses_task_and_replicate_address() -> None:
    seeds = {
        derive_episode_seed("master", example_id, replicate)
        for example_id in ("example-a", "example-b")
        for replicate in range(4)
    }
    assert len(seeds) == 8
    assert derive_episode_seed("master", "example-a", 0) == derive_episode_seed(
        "master", "example-a", 0
    )


def test_served_snapshot_and_renderer_identity_are_separate(
    tmp_path: Path,
) -> None:
    pytest.importorskip("verifiers.v1")
    from redco_evidence_selection_v2.run_feasibility import build_config

    args = Namespace(
        model="/workspace/models/exact-snapshot",
        renderer_model_name="Qwen/Qwen3-4B-Instruct-2507",
        base_url="http://127.0.0.1:8000/v1",
        dataset=tmp_path / "dataset.jsonl",
        dataset_sha256="a" * 64,
        split="train",
        prompt_profile="natural",
        output_dir=tmp_path / "output",
        num_tasks=8,
        replicates=4,
        master_seed="master",
        temperature=0.7,
        top_p=1.0,
        max_completion_tokens=768,
        max_total_tokens=8192,
        rlm_version="56218f33796ecbe465445bc43948886354fde196",
        setup_timeout=900.0,
        harness_timeout=900.0,
    )
    config = build_config(args)
    assert config.model == "/workspace/models/exact-snapshot"
    assert (
        config.client.renderer_model_name
        == "Qwen/Qwen3-4B-Instruct-2507"
    )
