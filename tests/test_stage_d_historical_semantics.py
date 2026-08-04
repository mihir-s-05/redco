from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_retained_stage_d_responses_pass_current_semantic_path(tmp_path: Path) -> None:
    tokenizer = os.environ.get("REDCO_STAGE_D_TOKENIZER_PATH")
    renderers = os.environ.get("REDCO_STAGE_D_RENDERERS_ROOT")
    verifiers = os.environ.get("REDCO_STAGE_D_VERIFIERS_ROOT")
    if tokenizer is None or renderers is None or verifiers is None:
        pytest.skip("the three REDCO_STAGE_D_* paths are required for live-stack preflight")

    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "historical-semantic-replay.json"
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts/audit_stage_d_historical_semantics.py"),
            "--repository",
            str(repository),
            "--tokenizer-path",
            tokenizer,
            "--renderers-root",
            renderers,
            "--verifiers-root",
            verifiers,
            "--output",
            str(output),
        ],
        check=True,
        cwd=repository,
        stdout=subprocess.DEVNULL,
    )
    report = json.loads(output.read_bytes())
    checked_in = repository / "reports/stage-d1-historical-semantic-replay-v1.json"
    assert output.read_bytes() == checked_in.read_bytes()
    assert report["passes"] is True
    assert report["live_support_run_authorized"] is False
    assert report["scientific_training_authorized"] is False
    assert report["historical_topology_replay_performed"] is False
    assert report["historical_versions_semantically_replayed"] == [4, 7, 8, 9, 10]
    assert report["semantic_renderer_observer_replay_count"] == 14
    assert report["completed_action_replay_count"] == 12
    assert report["unavailable_versions_were_not_reconstructed"] == [1, 2, 3, 5, 6]
    assert report["support_density"]["confirmatory_probability_identifiable"] is False
    assert report["support_density"]["descriptive_scaffold_proxy_successes"] == 3
    assert report["support_density"]["N_eligible"] is None
    assert report["support_density"]["N_joint"] is None
    assert all(
        replay.get("response_derived_non_key_fields_exact", True) is True
        for replay in report["replays"]
    )
    completed_replays = [
        replay
        for replay in report["replays"]
        if replay["historical_completed_action_available"]
    ]
    assert all(
        "prompt_tokens" in replay["all_non_key_behavior_fields_compared"]
        for replay in completed_replays
    )
    v10_uncompleted = [
        replay
        for replay in report["replays"]
        if replay["version"] == 10
        and not replay["historical_completed_action_available"]
    ]
    assert len(v10_uncompleted) == 1
    assert v10_uncompleted[0]["finish_reason"] == "length"
    assert v10_uncompleted[0]["termination_kind"] == "max_tokens"
    assert v10_uncompleted[0]["completion_tokens"] == 768
    assert v10_uncompleted[0]["request_max_tokens"] == 768
    assert v10_uncompleted[0]["current_observer_accepted"] is True
    assert isinstance(v10_uncompleted[0]["current_action_digest"], str)
