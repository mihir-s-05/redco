from __future__ import annotations

import json
import sys
from pathlib import Path

from redco.analysis.stage_d_scaffold_sft import evaluate

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.build_stage_d_scaffold_sft import build


def test_scaffold_sft_generator_is_deterministic_and_disjoint(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_manifest = tmp_path / "first-manifest.json"
    second_manifest = tmp_path / "second-manifest.json"
    scaffold = (
        root / "configs" / "stage-d" / "stage-d0-scaffold-fewshot-v2.txt"
    )
    build(
        scaffold_path=scaffold,
        dataset_path=first,
        manifest_path=first_manifest,
        examples=32,
    )
    build(
        scaffold_path=scaffold,
        dataset_path=second,
        manifest_path=second_manifest,
        examples=32,
    )

    assert first.read_bytes() == second.read_bytes()
    first_payload = json.loads(first_manifest.read_text(encoding="utf-8"))
    second_payload = json.loads(second_manifest.read_text(encoding="utf-8"))
    assert (
        first_payload["dataset_sha256"]
        == second_payload["dataset_sha256"]
    )
    report = evaluate(
        dataset_path=first,
        manifest_path=first_manifest,
        qasper_path=root
        / "datasets"
        / "stage-d"
        / "qasper-deterministic-v2.jsonl",
    )
    assert report["passes"]
    assert report["examples"] == 32
