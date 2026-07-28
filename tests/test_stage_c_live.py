from __future__ import annotations

import json
from pathlib import Path

import pytest

from redco.analysis.stage_c_live import smoke_verification_report, verify_smoke


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> Path:
    output = tmp_path / "output"
    rollouts = output / "rollouts" / "step_1"
    trace_rows: list[dict[str, object]] = []
    for episode in ("episode-a", "episode-b"):
        target = f"{episode}:target"
        trace_rows.append(
            {
                "id": f"{episode}-context",
                "agent": {"name": "context"},
                "info": {
                    "episode_id": episode,
                    "policy_version": 0,
                    "redco": {
                        "record_kind": "context",
                        "target_node_id": target,
                    },
                },
            }
        )
        for index in range(4):
            trace_rows.append(
                {
                    "id": f"{episode}-branch-{index}",
                    "agent": {"name": f"branch-{index}"},
                    "info": {
                        "episode_id": episode,
                        "policy_version": 0,
                        "redco": {
                            "record_kind": "branch",
                            "branch_index": index,
                            "target_node_id": target,
                            "selected_pre_action": True,
                            "replay_equivalent": True,
                            "checkpoint_contract": "episode-policy-version",
                            "action_seed": index + (100 if episode == "episode-a" else 200),
                            "branch_cache_salt": f"{episode}-salt-{index}",
                            "action_token_id": 15 + index,
                            "branch_temperature": 2.0,
                            "parsed_action": f"alias-{index}",
                            "canonical_action": f"action-{index}",
                            "full_suffix_reward": float(index),
                        },
                    },
                }
            )
    _write_jsonl(rollouts / "train" / "all" / "traces.jsonl", trace_rows)
    _write_jsonl(rollouts / "train" / "effective" / "traces.jsonl", trace_rows)
    _write_jsonl(output / "metrics.jsonl", [{"step": 1, "optim/grad_norm": 0.25}])
    _write_jsonl(
        output / "run_default" / "metrics.jsonl",
        [{"step": 1, "train/agg/effective/reward/mean": 1.5}],
    )
    (rollouts / "train_rollouts.bin").write_bytes(b"batch")
    adapter = output / "broadcasts" / "step_1" / "adapter_model.safetensors"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"adapter")
    return tmp_path


def test_verify_smoke_accepts_complete_snapshot(tmp_path: Path) -> None:
    report = verify_smoke(_fixture(tmp_path))
    assert report["status"] == "pass"
    assert report["checks"]["branch_records"] == 8
    assert report["checks"]["policy_versions"] == [0]


def test_verify_smoke_accepts_branch_only_effective_subset(tmp_path: Path) -> None:
    run_dir = _fixture(tmp_path)
    effective = next(run_dir.glob("**/train/effective/traces.jsonl"))
    rows = [json.loads(line) for line in effective.read_text().splitlines()]
    branch_only = [
        row
        for row in rows
        if row["info"]["episode_id"] == "episode-a"
        and row["info"]["redco"]["record_kind"] == "branch"
    ]
    _write_jsonl(effective, branch_only)

    report = verify_smoke(run_dir)

    assert report["status"] == "pass"
    assert report["checks"]["effective_traces"] == 4
    assert report["checks"]["effective_episodes"] == 1
    assert report["checks"]["branch_records"] == 4
    assert report["checks"]["context_records"] == 0


def test_verify_smoke_accepts_preregistered_empty_batch_retries(
    tmp_path: Path,
) -> None:
    run_dir = _fixture(tmp_path)
    all_traces = next(run_dir.glob("**/train/all/traces.jsonl"))
    rows = [json.loads(line) for line in all_traces.read_text().splitlines()]
    retry_rows = json.loads(json.dumps(rows))
    for row in retry_rows:
        row["id"] += "-retry"
        row["info"]["episode_id"] += "-retry"
    _write_jsonl(all_traces, [*rows, *retry_rows])

    report = verify_smoke(run_dir)

    assert report["status"] == "pass"
    assert report["checks"]["collected_traces"] == 20
    assert report["checks"]["collected_episodes"] == 4
    assert report["checks"]["collection_attempts"] == 2


def test_verify_smoke_rejects_gathered_master_weights(tmp_path: Path) -> None:
    run_dir = _fixture(tmp_path)
    gathered = run_dir / "output" / "weights" / "step_1" / "model.safetensors"
    gathered.parent.mkdir(parents=True)
    gathered.write_bytes(b"redundant full model")

    with pytest.raises(
        ValueError, match="prohibited gathered full-model checkpoint"
    ):
        verify_smoke(run_dir)


def test_verify_smoke_rejects_replay_disagreement(tmp_path: Path) -> None:
    run_dir = _fixture(tmp_path)
    traces = next(run_dir.glob("**/train/effective/traces.jsonl"))
    rows = [json.loads(line) for line in traces.read_text().splitlines()]
    rows[1]["info"]["redco"]["replay_equivalent"] = False
    _write_jsonl(traces, rows)
    try:
        verify_smoke(run_dir)
    except ValueError as error:
        assert "replay equivalence" in str(error)
    else:
        raise AssertionError("expected replay disagreement to fail")


def test_smoke_verification_report_records_missing_artifact_failure(
    tmp_path: Path,
) -> None:
    report = smoke_verification_report(tmp_path)

    assert report["status"] == "fail"
    assert report["error"]["type"] == "ValueError"
    assert "metrics.jsonl" in report["error"]["message"]
    assert len(report["signed_payload_sha256"]) == 64
