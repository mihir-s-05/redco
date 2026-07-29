"""Verify the deterministic Stage-C3 forced-output integration smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _last_reply(trace: dict[str, Any]) -> str:
    return str(trace["nodes"][-1]["message"]["content"])


def _episode_index(trace: dict[str, Any]) -> int:
    salt = str(trace["agent"]["sampling"]["extra_body"]["cache_salt"])
    marker = ":episode:"
    if marker not in salt:
        raise ValueError("trace cache salt has no episode address")
    value = salt.split(marker, maxsplit=1)[1].split(":", maxsplit=1)[0]
    if not value.isdigit():
        raise ValueError("trace cache salt has an invalid episode index")
    return int(value)


def verify(
    traces_path: Path,
    invariant_path: Path,
) -> dict[str, Any]:
    traces = [
        json.loads(line)
        for line in traces_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_episode[str(trace["info"]["episode_id"])].append(trace)
    observed: dict[int, tuple[str, str, float]] = {}
    serialized = True
    for episode in by_episode.values():
        by_role = {str(trace["agent"]["name"]): trace for trace in episode}
        if set(by_role) != {"context", "original"}:
            raise ValueError("forced smoke episode has unexpected roles")
        context = by_role["context"]
        original = by_role["original"]
        if _episode_index(context) != _episode_index(original):
            raise ValueError("episode roles have different rollout addresses")
        index = _episode_index(context)
        observed[index] = (
            _last_reply(context),
            _last_reply(original),
            float(original["rewards"]["deterministic_reward"]),
        )
        serialized = serialized and all(
            isinstance(node.get("token_ids"), list)
            for trace in episode
            for node in trace["nodes"]
        )
    expected = {
        0: ("<route>gamma</route>", "5", 1.0),
        1: ("<route>delta</route>", "1", 1.0),
        2: ("<route>gamma</route>", "0", 0.0),
        3: ("<route>alpha</route>", "2", 0.0),
        4: ("<route>beta</route>", "3", 0.0),
        5: ("<route>gamma</route>", "4", 0.0),
        6: ("<route>alpha</route>", "6", 0.0),
        7: ("<route>beta</route>", "7", 0.0),
    }
    invariant = json.loads(invariant_path.read_text(encoding="utf-8"))
    checks = {
        "eight_exact_episode_addresses": set(observed) == set(expected),
        "forced_outputs_and_rewards_exact": observed == expected,
        "trace_token_serialization_present": serialized,
        "supervisor_invariant_passed": invariant.get("passed") is True,
        "supervisor_reward_exposure_exact": (
            invariant.get("observed", {}).get("reward_min") == 0.0
            and invariant.get("observed", {}).get("reward_max") == 1.0
        ),
        "supervisor_trainable_fraction_nonzero": (
            float(
                invariant.get("observed", {}).get(
                    "trainable_fraction",
                    0.0,
                )
            )
            > 0.0
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis": "stage-c3-v3-forced-integration-smoke",
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            str(index): {
                "route": values[0],
                "action": values[1],
                "reward": values[2],
            }
            for index, values in sorted(observed.items())
        },
        "route_counts": dict(
            Counter(values[0] for values in observed.values())
        ),
        "action_counts": dict(
            Counter(values[1] for values in observed.values())
        ),
    }
    signed = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--invariant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.traces, args.invariant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
