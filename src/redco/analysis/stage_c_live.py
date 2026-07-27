from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object JSON line")
                records.append(value)
    return records


def _single_match(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {pattern!r} below {run_dir}, found {len(matches)}"
        )
    return matches[0]


def verify_smoke(run_dir: Path) -> dict[str, Any]:
    metrics_path = _single_match(run_dir, "**/metrics.jsonl")
    all_traces_path = _single_match(
        run_dir, "**/rollouts/step_1/train/all/traces.jsonl"
    )
    effective_traces_path = _single_match(
        run_dir, "**/rollouts/step_1/train/effective/traces.jsonl"
    )
    batch_path = _single_match(
        run_dir, "**/rollouts/step_1/train_rollouts.bin"
    )
    adapter_path = _single_match(
        run_dir, "**/broadcasts/step_1/adapter_model.safetensors"
    )

    all_traces = _read_jsonl(all_traces_path)
    traces = _read_jsonl(effective_traces_path)
    if len(traces) != 10:
        raise ValueError(
            f"smoke must serialize exactly 10 effective traces, found {len(traces)}"
        )
    if len(all_traces) < len(traces) or len(all_traces) % 5 != 0:
        raise ValueError("collected smoke traces must contain complete five-trace episodes")
    effective_ids = {trace.get("id") for trace in traces}
    all_ids = {trace.get("id") for trace in all_traces}
    if (
        None in effective_ids
        or len(effective_ids) != len(traces)
        or len(all_ids) != len(all_traces)
        or not effective_ids <= all_ids
    ):
        raise ValueError("effective smoke traces are not a subset of collected traces")
    collected_episodes: dict[str, list[dict[str, Any]]] = {}
    for trace in all_traces:
        info = trace.get("info")
        if not isinstance(info, dict):
            raise ValueError("collected trace is missing info metadata")
        episode_id = info.get("episode_id")
        redco = info.get("redco")
        if (
            not isinstance(episode_id, str)
            or not episode_id
            or not isinstance(redco, dict)
        ):
            raise ValueError("collected trace is missing its Stage C episode record")
        if info.get("policy_version") != 0:
            raise ValueError("collected smoke trace is not from snapshot version 0")
        collected_episodes.setdefault(episode_id, []).append(trace)
    if any(len(episode) != 5 for episode in collected_episodes.values()):
        raise ValueError("collected smoke episode is not a complete five-trace group")

    policy_versions: set[int] = set()
    episodes: dict[str, list[dict[str, Any]]] = {}
    branch_records: list[dict[str, Any]] = []
    context_records: list[dict[str, Any]] = []
    for trace in traces:
        info = trace.get("info")
        agent = trace.get("agent")
        if not isinstance(info, dict) or not isinstance(agent, dict):
            raise ValueError("trace is missing info or agent metadata")
        policy_version = info.get("policy_version")
        episode_id = info.get("episode_id")
        redco = info.get("redco")
        if not isinstance(policy_version, int) or not isinstance(episode_id, str):
            raise ValueError("trace is missing policy version or episode id")
        if not isinstance(redco, dict):
            raise ValueError("trace is missing the Stage C redco record")
        policy_versions.add(policy_version)
        episodes.setdefault(episode_id, []).append(trace)
        if redco.get("record_kind") == "branch":
            branch_records.append(redco)
        elif redco.get("record_kind") == "context":
            context_records.append(redco)
        else:
            raise ValueError("trace has an unknown Stage C record kind")

    if policy_versions != {0}:
        raise ValueError(f"smoke must use only snapshot version 0: {policy_versions}")
    if len(episodes) != 2 or any(len(group) != 5 for group in episodes.values()):
        raise ValueError("smoke must contain two complete five-trace episodes")
    if len(branch_records) != 8 or len(context_records) != 2:
        raise ValueError("smoke must contain eight branch and two context records")

    episodes_with_branch_signal = 0
    for episode in episodes.values():
        redco_records = [trace["info"]["redco"] for trace in episode]
        branches = [
            record for record in redco_records if record["record_kind"] == "branch"
        ]
        contexts = [
            record for record in redco_records if record["record_kind"] == "context"
        ]
        if sorted(record.get("branch_index") for record in branches) != [0, 1, 2, 3]:
            raise ValueError("episode does not contain branch indices 0 through 3")
        if len(contexts) != 1:
            raise ValueError("episode must contain exactly one context record")
        target_ids = {record.get("target_node_id") for record in redco_records}
        if len(target_ids) != 1 or None in target_ids:
            raise ValueError("episode target identity is inconsistent")
        action_seeds = [record.get("action_seed") for record in branches]
        if not all(isinstance(seed, int) for seed in action_seeds):
            raise ValueError("episode is missing a structural action seed")
        if len(set(action_seeds)) != 4:
            raise ValueError("episode branch action seeds are not distinct")
        cache_salts = [record.get("branch_cache_salt") for record in branches]
        if not all(isinstance(salt, str) and salt for salt in cache_salts):
            raise ValueError("episode is missing an auditable branch cache salt")
        if len(set(cache_salts)) != 4:
            raise ValueError("episode branch cache salts are not distinct")
        parsed_actions = [record.get("parsed_action") for record in branches]
        canonical_actions = [record.get("canonical_action") for record in branches]
        if not all(isinstance(action, str) and action for action in parsed_actions):
            raise ValueError("effective episode contains an invalid displayed action")
        if not all(isinstance(action, str) and action for action in canonical_actions):
            raise ValueError("effective episode contains an invalid canonical action")
        branch_rewards = [record.get("full_suffix_reward") for record in branches]
        if not all(
            isinstance(reward, (int, float)) and math.isfinite(float(reward))
            for reward in branch_rewards
        ):
            raise ValueError("effective episode contains an invalid branch reward")
        if len({float(reward) for reward in branch_rewards}) > 1:
            episodes_with_branch_signal += 1

    if episodes_with_branch_signal < 1:
        raise ValueError("effective smoke batch has no counterfactual branch signal")

    if not all(record.get("selected_pre_action") is True for record in branch_records):
        raise ValueError("a target was not committed before its action")
    if not all(record.get("replay_equivalent") is True for record in branch_records):
        raise ValueError("in-loop sliced/full replay equivalence failed")
    snapshot_contracts = (
        record.get("checkpoint_contract") == "episode-policy-version"
        for record in branch_records
    )
    if not all(snapshot_contracts):
        raise ValueError("a branch is missing the snapshot checkpoint contract")

    metric_rows = _read_jsonl(metrics_path)
    grad_norms = [
        float(row["optim/grad_norm"])
        for row in metric_rows
        if "optim/grad_norm" in row
    ]
    if len(grad_norms) != 1:
        raise ValueError(f"expected one optimizer grad norm, found {len(grad_norms)}")
    grad_norm = grad_norms[0]
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise ValueError(f"optimizer gradient must be finite and positive: {grad_norm}")

    files = {
        "metrics": metrics_path,
        "all_traces": all_traces_path,
        "effective_traces": effective_traces_path,
        "train_batch": batch_path,
        "adapter": adapter_path,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "gate": "stage-c-live-smoke",
        "status": "pass",
        "run_dir": run_dir.as_posix(),
        "checks": {
            "optimizer_steps": 1,
            "policy_versions": sorted(policy_versions),
            "collected_traces": len(all_traces),
            "collected_episodes": len(collected_episodes),
            "effective_traces": len(traces),
            "episodes": len(episodes),
            "branch_records": len(branch_records),
            "context_records": len(context_records),
            "episodes_with_branch_signal": episodes_with_branch_signal,
            "all_targets_precommitted": True,
            "all_replays_equivalent": True,
            "grad_norm": grad_norm,
            "adapter_bytes": adapter_path.stat().st_size,
            "batch_bytes": batch_path.stat().st_size,
        },
        "artifacts": {
            name: {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in files.items()
        },
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def smoke_verification_report(run_dir: Path) -> dict[str, Any]:
    """Return a signed pass/fail record, including failures before artifact creation."""
    try:
        return verify_smoke(run_dir)
    except Exception as error:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "gate": "stage-c-live-smoke",
            "status": "fail",
            "run_dir": run_dir.as_posix(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen Stage C live smoke")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = smoke_verification_report(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
