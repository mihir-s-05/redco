"""Execute a scripted Stage D replay through the real RLM/IPython runtime."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path
from typing import Any

from redco.analysis.rlm_episode_replay import (
    RLMEventAddress,
    ScriptedCompletionRouter,
    ScriptedEvent,
    ScriptedModelServer,
    inject_child_answer,
    load_scripted_events,
)
from redco.integrations.signed_subprocess import sign_payload


def _workspace_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _source_hashes(repo_root: Path) -> dict[str, str]:
    paths = (
        "patches/rlm-event-replay-provenance.patch",
        "src/redco/analysis/rlm_episode_replay.py",
        "scripts/run_stage_d_rlm_replay_fixture.py",
        "scripts/run_stage_d_rlm_replay_wsl.sh",
        "tests/fixtures/stage_d_rlm_replay_cassette_v1.json",
    )
    return {path: hashlib.sha256((repo_root / path).read_bytes()).hexdigest() for path in paths}


async def run(
    cassette: Path,
    workspace: Path,
    output: Path,
    *,
    require_network_blocked: bool,
) -> None:
    network_check: dict[str, Any] = {"required": require_network_blocked}
    if require_network_blocked:
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=0.25)
        except OSError as error:
            network_check.update({"blocked": True, "error_type": type(error).__name__})
        else:
            raise RuntimeError("non-loopback network is reachable")
    events = load_scripted_events(str(cassette))
    workspace.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        {
            "RLM_API_KEY": "scripted-local-only",
            "RLM_MODEL": "scripted-replay",
            "RLM_DEPTH": "0",
            "RLM_MAX_DEPTH": "1",
            "RLM_SDK_MAX_RETRIES": "0",
            "RLM_MAX_TOKENS": "8192",
        }
    )
    from rlm.api import run as run_rlm  # type: ignore[import-not-found]

    async def run_arm(
        name: str,
        arm_events: tuple[ScriptedEvent, ...],
    ) -> dict[str, Any]:
        arm_workspace = workspace / name
        arm_workspace.mkdir()
        os.environ["RLM_HOME"] = str(arm_workspace / ".rlm-home")
        router = ScriptedCompletionRouter(arm_events)
        with ScriptedModelServer(router) as server:
            os.environ["RLM_BASE_URL"] = server.base_url
            result = await run_rlm("Execute the scripted replay fixture.", cwd=str(arm_workspace))
        terminal = arm_workspace / "result.txt"
        terminal_text = terminal.read_text(encoding="utf-8")
        audit = router.audit()
        semantic = {
            "answer": result.answer,
            "terminal_text": terminal_text,
            "turns": result.turns,
            "addresses": audit["seen_addresses"],
        }
        return {
            **semantic,
            "semantic_sha256": hashlib.sha256(
                json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "router_audit": audit,
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
            },
            "workspace_manifest": _workspace_manifest(arm_workspace),
        }

    target = RLMEventAddress(
        depth=1,
        turn=0,
        call_kind="policy",
        parent_lineage="root",
        parent_turn=0,
        parent_tool_call_id="call_0",
        invocation_id="shard-0",
    )
    original = await run_arm("original", events)
    branch_events = inject_child_answer(events, target=target, answer="changed")
    branch = await run_arm("branch", branch_events)
    repo_root = Path(__file__).resolve().parents[1]
    payload = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-rlm-whole-episode-replay-fixture",
            "rlm_upstream_commit": "56218f33796ecbe465445bc43948886354fde196",
            "source_sha256": _source_hashes(repo_root),
            "runtime": {
                "platform": platform.platform(),
                "python": sys.version,
            },
            "cassette_sha256": hashlib.sha256(cassette.read_bytes()).hexdigest(),
            "target_address": target.key(),
            "intervention": {"candidate_answer": "changed", "event_count": 1},
            "network_check": network_check,
            "original": original,
            "branch": branch,
            "passes": (
                original["router_audit"]["complete"]
                and branch["router_audit"]["complete"]
                and all(
                    row["session_id_present"]
                    for arm in (original, branch)
                    for row in arm["router_audit"]["requests"]
                )
                and original["answer"] == "same|same\nline"
                and original["terminal_text"] == "same|same\nline"
                and branch["answer"] == "changed|same\nline"
                and branch["terminal_text"] == "changed|same\nline"
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not payload["passes"]:
        raise SystemExit("whole-episode replay fixture failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cassette", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-network-blocked", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        run(
            args.cassette,
            args.workspace,
            args.output,
            require_network_blocked=args.require_network_blocked,
        )
    )


if __name__ == "__main__":
    main()
