from __future__ import annotations

import json
from pathlib import Path

from redco.analysis.stage_c_live import verify_smoke


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
                        },
                    },
                }
            )
    _write_jsonl(rollouts / "train" / "all" / "traces.jsonl", trace_rows)
    _write_jsonl(output / "metrics.jsonl", [{"step": 1, "optim/grad_norm": 0.25}])
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


def test_verify_smoke_rejects_replay_disagreement(tmp_path: Path) -> None:
    run_dir = _fixture(tmp_path)
    traces = next(run_dir.glob("**/train/all/traces.jsonl"))
    rows = [json.loads(line) for line in traces.read_text().splitlines()]
    rows[1]["info"]["redco"]["replay_equivalent"] = False
    _write_jsonl(traces, rows)
    try:
        verify_smoke(run_dir)
    except ValueError as error:
        assert "replay equivalence" in str(error)
    else:
        raise AssertionError("expected replay disagreement to fail")
