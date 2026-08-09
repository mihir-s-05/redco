"""Source-free tests for the candidate-null v13 support launch bundle."""

from __future__ import annotations

import base64
import inspect
import json
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import run_stage_d_v13_local_orchestrator as local_orchestrator
import run_stage_d_v13_remote_bootstrap as bootstrap
from run_stage_d_v13_local_orchestrator import LocalLaunchOrchestrator

from redco.analysis import stage_d_v13_launch_lifecycle as lifecycle
from redco.analysis import stage_d_v13_launch_observations as observations
from redco.analysis import stage_d_v13_support_launch as launch
from redco.analysis import stage_d_v13_support_launch_runtime as runtime
from redco.analysis.stage_d_dependency_stack import (
    live_owner_dependency_payload,
    write_canonical_tree_tar,
)
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_draft_publication import atomic_publish_set
from redco.analysis.stage_d_v13_support_contract import CANDIDATE_RELATIVE
from redco.analysis.stage_d_v13_support_launch import (
    LAUNCH_AUDIT_RELATIVE,
    LAUNCH_AUTH_RELATIVE,
    LAUNCH_BRANCH_RUNTIME_RELATIVE,
    LAUNCH_DATASET_RELATIVE,
    LAUNCH_PLAN_RELATIVE,
    LAUNCH_PROTOCOL_RELATIVE,
    LAUNCH_SOURCE_EVAL_RELATIVE,
    PARENT_COMMIT,
    PARENT_TREE,
    build_launch_artifacts,
    build_preflight_snapshot,
    execute_support_once,
    preflight_validate,
    verify_launch_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (
    LAUNCH_AUTH_RELATIVE,
    LAUNCH_DATASET_RELATIVE,
    LAUNCH_PLAN_RELATIVE,
    LAUNCH_SOURCE_EVAL_RELATIVE,
    LAUNCH_PROTOCOL_RELATIVE,
    LAUNCH_BRANCH_RUNTIME_RELATIVE,
    LAUNCH_AUDIT_RELATIVE,
)


def _prime_must_not_run(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("Prime must not be called")


def test_terminal_payload_preserves_owned_process_failure_bytes_as_hashes() -> None:
    error = subprocess.CalledProcessError(
        17,
        ["owner", "--execute"],
        output=b"exact stdout\x00",
        stderr=b"exact stderr\xff",
    )
    payload = json.loads(
        runtime.terminal_payload(
            state="failed_terminal_no_retry",
            provider_dispatch_observed=True,
            error=error,
        )
    )
    assert payload["error_returncode"] == 17
    assert payload["error_output_sha256"] == sha256_bytes(b"exact stdout\x00")
    assert payload["error_stderr_sha256"] == sha256_bytes(b"exact stderr\xff")


@pytest.fixture(scope="module")  # type: ignore[untyped-decorator]
def payloads() -> dict[str, bytes]:
    return build_launch_artifacts(ROOT)


@pytest.fixture  # type: ignore[untyped-decorator]
def published_bundle(tmp_path: Path, payloads: dict[str, bytes]) -> Path:
    atomic_publish_set(
        tmp_path,
        payloads,
        immutable_paths=launch._bundle_immutable_paths(ROOT),
        manifest_path=LAUNCH_AUDIT_RELATIVE,
        require_draft_envelope=False,
    )
    return tmp_path


def test_launch_bundle_rebuild_is_byte_identical(payloads: dict[str, bytes]) -> None:
    assert build_launch_artifacts(ROOT) == payloads


def test_launch_bundle_is_exactly_support_only(payloads: dict[str, bytes]) -> None:
    authorization = json.loads(payloads[LAUNCH_AUTH_RELATIVE])
    assert authorization["parent"] == {"commit": PARENT_COMMIT, "tree": PARENT_TREE}
    assert authorization["authorization"]["text_byte_length"] == 189
    assert authorization["signing"] == launch.launch_signing_identity().to_payload()
    scope = authorization["scope"]
    assert scope["support_attempt_limit"] == 1
    assert scope["support_launch_authorized"] is True
    assert scope["provider_calls_authorized"] is True
    assert scope["model_calls_authorized"] is True
    assert scope["support_spend_authorized"] is True
    assert all(
        scope[name] is False
        for name in (
            "science_authorized",
            "training_authorized",
            "heldout_evaluation_authorized",
            "scientific_transition_authorized",
            "prime_gpu_scientific_launch_authorized",
        )
    )
    rows = payloads[LAUNCH_DATASET_RELATIVE].splitlines()
    assert len(rows) == 64
    assert all(json.loads(row)["split"] == "successor_support" for row in rows)


def test_launch_tomls_pass_pinned_eval_config_and_reject_unsupported_tables(
    payloads: dict[str, bytes],
) -> None:
    """The exact pinned owner parser, not a permissive TOML parser, owns this gate."""

    from verifiers.v1.configs.eval import EvalConfig

    expected_sampling = {
        **launch.LAUNCH_PERSISTED_SAMPLING,
        "extra_body": {"cache_salt": "placeholder-only-before-episode-addressing"},
    }
    source_raw = tomllib.loads(payloads[LAUNCH_SOURCE_EVAL_RELATIVE].decode("utf-8"))
    branch_raw = tomllib.loads(payloads[LAUNCH_BRANCH_RUNTIME_RELATIVE].decode("utf-8"))
    source = EvalConfig.model_validate(source_raw)
    branch = EvalConfig.model_validate(branch_raw)
    assert source.num_tasks == 64
    assert source.num_rollouts == source.max_concurrent == 1
    assert branch.num_tasks == branch.num_rollouts == branch.max_concurrent == 1
    assert source.sampling.model_dump(mode="json", exclude_none=False) == expected_sampling
    assert branch.sampling.model_dump(mode="json", exclude_none=False) == expected_sampling
    assert source.env.branch_count == branch.env.branch_count == 4
    assert branch.env.taskset.rollouts_per_task == 1
    assert "branch" not in branch_raw
    assert "network_fallback" not in source_raw
    assert "network_fallback" not in branch_raw

    with pytest.raises(ValueError):
        EvalConfig.model_validate(
            tomllib.loads(
                payloads[LAUNCH_BRANCH_RUNTIME_RELATIVE].decode("utf-8")
                + "\n[branch]\nk = 4\n"
            )
        )
    with pytest.raises(ValueError):
        EvalConfig.model_validate(
            tomllib.loads(
                "network_fallback = false\n"
                + payloads[LAUNCH_SOURCE_EVAL_RELATIVE].decode("utf-8")
            )
        )


def test_launch_bundle_verifies_and_binds_plan_and_manifest(
    published_bundle: Path,
    payloads: dict[str, bytes],
) -> None:
    actual = verify_launch_bundle(ROOT, published_bundle)
    assert set(actual) == set(OUTPUTS)
    plan = json.loads((published_bundle / LAUNCH_PLAN_RELATIVE).read_bytes())
    protocol = json.loads((published_bundle / LAUNCH_PROTOCOL_RELATIVE).read_bytes())
    assert len(plan["slots"]) == 64
    assert protocol["collection_plan_sha256"] == actual[LAUNCH_PLAN_RELATIVE]
    assert actual[LAUNCH_SOURCE_EVAL_RELATIVE] == sha256_bytes(
        payloads[LAUNCH_SOURCE_EVAL_RELATIVE]
    )


def test_source_free_build_never_reads_authenticated_parquet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = launch._read_bound

    def guarded(root: Path, relative: str, expected: str) -> bytes:
        assert not relative.endswith("0000.parquet")
        return original(root, relative, expected)

    monkeypatch.setattr(launch, "_read_bound", guarded)
    build_launch_artifacts(ROOT)


def test_check_only_does_not_write_or_repair(
    published_bundle: Path,
    payloads: dict[str, bytes],
) -> None:
    before = {
        relative: (
            (published_bundle / relative).read_bytes(),
            os.stat(published_bundle / relative).st_mtime_ns,
        )
        for relative in OUTPUTS
    }
    atomic_publish_set(
        published_bundle,
        payloads,
        immutable_paths=launch._bundle_immutable_paths(ROOT),
        manifest_path=LAUNCH_AUDIT_RELATIVE,
        check_only=True,
        require_draft_envelope=False,
    )
    after = {
        relative: (
            (published_bundle / relative).read_bytes(),
            os.stat(published_bundle / relative).st_mtime_ns,
        )
        for relative in OUTPUTS
    }
    assert after == before


def test_check_only_rejects_tamper_without_mutation(
    published_bundle: Path,
    payloads: dict[str, bytes],
) -> None:
    target = published_bundle / LAUNCH_AUTH_RELATIVE
    tampered = json.loads(target.read_bytes())
    tampered["scope"]["science_authorized"] = True
    tampered_bytes = canonical_json_bytes(tampered)
    target.write_bytes(tampered_bytes)
    before = target.stat()
    with pytest.raises(ValueError, match="differ"):
        atomic_publish_set(
            published_bundle,
            payloads,
            immutable_paths=launch._bundle_immutable_paths(ROOT),
            manifest_path=LAUNCH_AUDIT_RELATIVE,
            check_only=True,
            require_draft_envelope=False,
        )
    after = target.stat()
    assert target.read_bytes() == tampered_bytes
    assert (after.st_size, after.st_mtime_ns, after.st_ino) == (
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
    )


def test_coordinated_authorization_mutation_fails_closed(
    published_bundle: Path,
    payloads: dict[str, bytes],
) -> None:
    target = published_bundle / LAUNCH_AUTH_RELATIVE
    mutated: dict[str, Any] = json.loads(target.read_bytes())
    mutated["parent"]["commit"] = "0" * 40
    target.write_bytes(canonical_json_bytes(mutated))
    with pytest.raises(ValueError, match="differ"):
        verify_launch_bundle(ROOT, published_bundle)


def test_output_and_immutable_input_aliases_fail_closed(
    published_bundle: Path,
    payloads: dict[str, bytes],
) -> None:
    output_alias = published_bundle / LAUNCH_PLAN_RELATIVE
    output_alias.unlink()
    os.link(published_bundle / LAUNCH_AUTH_RELATIVE, output_alias)
    try:
        with pytest.raises(ValueError, match="aliases"):
            atomic_publish_set(
                published_bundle,
                payloads,
                immutable_paths=launch._bundle_immutable_paths(ROOT),
                manifest_path=LAUNCH_AUDIT_RELATIVE,
                check_only=True,
                require_draft_envelope=False,
            )
    finally:
        output_alias.unlink()
        output_alias.write_bytes(payloads[LAUNCH_PLAN_RELATIVE])

    immutable_alias = published_bundle / LAUNCH_AUTH_RELATIVE
    immutable_alias.unlink()
    os.link(ROOT / CANDIDATE_RELATIVE, immutable_alias)
    try:
        with pytest.raises(ValueError, match="hard-link alias"):
            atomic_publish_set(
                published_bundle,
                payloads,
                immutable_paths=launch._bundle_immutable_paths(ROOT),
                manifest_path=LAUNCH_AUDIT_RELATIVE,
                check_only=True,
                require_draft_envelope=False,
            )
    finally:
        immutable_alias.unlink()
        immutable_alias.write_bytes(payloads[LAUNCH_AUTH_RELATIVE])


def test_preflight_snapshot_is_canonical_and_real(tmp_path: Path) -> None:
    snapshot = tmp_path / "preflight.json"
    snapshot.write_bytes(
        build_preflight_snapshot(
            ROOT,
            location="synthetic-cpu-location",
            captured_at_epoch=int(time.time()),
            expires_at_epoch=int(time.time()) + 3600,
        )
    )
    result = preflight_validate(ROOT, snapshot, synthetic=True)
    assert set(result) == set(OUTPUTS)
    mutated = json.loads(snapshot.read_bytes())
    mutated["resource"]["wallet_usd"] = 29
    snapshot.write_bytes(canonical_json_bytes(mutated))
    with pytest.raises(ValueError, match="resource witness"):
        preflight_validate(ROOT, snapshot, synthetic=True)


def test_precommit_execution_is_non_authorizing_and_claim_free(tmp_path: Path) -> None:
    snapshot = tmp_path / "preflight.json"
    snapshot.write_bytes(
        build_preflight_snapshot(
            ROOT,
            location="synthetic-cpu-location",
            captured_at_epoch=int(time.time()),
            expires_at_epoch=int(time.time()) + 3600,
        )
    )
    with pytest.raises(ValueError, match="synthetic preflight"):
        execute_support_once(ROOT, preflight_snapshot=snapshot)
    assert not (ROOT / launch.LAUNCH_ATTEMPT_RELATIVE).exists()


def test_synthetic_preflight_cannot_reach_execute_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        launch,
        "preflight_validate",
        lambda *_args, **_kwargs: {LAUNCH_AUTH_RELATIVE: "synthetic-auth"},
    )
    with pytest.raises(ValueError, match="synthetic preflight"):
        execute_support_once(
            tmp_path,
            preflight_snapshot=tmp_path / "synthetic.json",
        )
    assert not (tmp_path / launch.LAUNCH_ATTEMPT_RELATIVE).exists()


def test_synthetic_post_commit_gate_requires_exact_direct_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "1" * 40
    values = {
        ("rev-parse", "HEAD"): head,
        ("rev-parse", f"{PARENT_COMMIT}^{{tree}}"): PARENT_TREE,
        ("rev-parse", "HEAD^"): PARENT_COMMIT,
        ("rev-list", "--parents", "-n", "1", head): f"{head} {PARENT_COMMIT}",
    }
    monkeypatch.setattr(launch, "_git_value", lambda _root, *args: values[tuple(args)])
    monkeypatch.setattr(launch, "_status_paths", lambda _root: ((" M", "external/prime-rl"),))
    seen: list[str] = []

    def exact_diff(_root: Path, actual_head: str) -> None:
        seen.append(actual_head)

    monkeypatch.setattr(launch, "_authenticate_committed_diff", exact_diff)
    assert launch._authenticate_parent(ROOT, require_post_commit=True) == "committed_direct_child"
    assert seen == [head]
    values[("rev-parse", "HEAD^")] = "2" * 40
    with pytest.raises(ValueError, match="direct child"):
        launch._authenticate_parent(ROOT, require_post_commit=True)


def test_real_temp_git_post_commit_allowlist_and_dirty_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = tuple(sorted(launch.LAUNCH_BUNDLE_PATHS))

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    git("init", "--quiet")
    git("config", "user.email", "redco-test@example.invalid")
    git("config", "user.name", "Redco test")
    (tmp_path / "base.txt").write_bytes(b"base")
    git("add", "base.txt")
    git("commit", "--quiet", "-m", "baseline")
    baseline = git("rev-parse", "HEAD")
    baseline_tree = git("rev-parse", "HEAD^{tree}")
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"bundle")
    git("add", "-f", "--", *paths)
    git("commit", "--quiet", "-m", "bundle")
    monkeypatch.setattr(launch, "PARENT_COMMIT", baseline)
    monkeypatch.setattr(launch, "PARENT_TREE", baseline_tree)
    monkeypatch.setattr(launch, "LAUNCH_BUNDLE_PATHS", frozenset(paths))
    assert (
        launch._authenticate_parent(tmp_path, require_post_commit=True)
        == "committed_direct_child"
    )

    (tmp_path / "unallowlisted.txt").write_bytes(b"dirty")
    with pytest.raises(ValueError, match="authenticated clean view"):
        launch._authenticate_parent(tmp_path, require_post_commit=True)


def test_real_launch_owners_run_source_free_in_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the public execute-once path against only external-boundary fakes.

    The temporary repository is a disposable direct child of the reviewed
    baseline.  The test signs the fixed handoff, invokes the real CLI, and
    lets ProductionSupportActuator spawn the real collection and campaign
    owners.  The only substituted code is the provider transport/renderer
    boundary; all durable collection, roster, branch, scoring, recovery, and
    terminal artifacts are produced by the launch owners.
    """

    clone = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--local", "--no-hardlinks", str(ROOT), str(clone)],
        check=True,
        capture_output=True,
    )
    for owner_script in (
        "scripts/run_stage_d_source_collection.py",
        "scripts/run_stage_d_scientific_campaign.py",
    ):
        shutil.copy2(ROOT / owner_script, clone / owner_script)
    prime_checkout = clone / "external" / "prime-rl"
    if prime_checkout.exists():
        shutil.rmtree(prime_checkout)
    prime_checkout.mkdir(parents=True)
    for dependency in ("renderers", "verifiers"):
        dependency_path = prime_checkout / "deps" / dependency
        if dependency_path.exists():
            shutil.rmtree(dependency_path)
        subprocess.run(
            [
                "git",
                "clone",
                "--local",
                "--no-hardlinks",
                str(ROOT / "external" / "prime-rl" / "deps" / dependency),
                str(dependency_path),
            ],
            check=True,
            capture_output=True,
        )
    live_owner = live_owner_dependency_payload(clone)
    for component in live_owner["components"]:
        component_name = str(component["name"])
        dependency_path = prime_checkout / "deps" / component_name
        checkout_command = [
            "git",
            "-C",
            str(dependency_path),
            "checkout",
            "-q",
            "--detach",
            str(component["base_commit"]),
        ]
        subprocess.run(
            checkout_command,
            check=True,
            capture_output=True,
        )
        for patch in component["patches"]:
            patch_name = str(patch["name"])
            subprocess.run(
                ["git", "-C", str(dependency_path), "apply", str(clone / "patches" / patch_name)],
                check=True,
            capture_output=True,
        )
    fixture_root = tmp_path / "authenticated-rlm-fixture"
    rlm_tree = fixture_root / "rlm-tree"
    (rlm_tree / "src" / "rlm").mkdir(parents=True)
    (rlm_tree / "src" / "rlm" / "provenance.py").write_bytes(
        b'DOMAIN = "redco.stage-d.spawn-lineage.v2"\n'
    )
    (rlm_tree / "src" / "rlm" / "session.py").write_bytes(b"# fixture session\n")
    (rlm_tree / "src" / "rlm" / "cli.py").write_bytes(b"# fixture cli\n")
    (rlm_tree / "install.sh").write_bytes(b"#!/bin/sh\nset -eu\n")
    (rlm_tree / "pyproject.toml").write_bytes(
        b'[project]\nname = "rlm"\nversion = "0.0.0"\n'
    )
    (rlm_tree / "uv.lock").write_bytes(b"version = 1\n")
    archive_path = fixture_root / "rlm-patched-v1.tar"
    write_canonical_tree_tar(rlm_tree, archive_path)
    uv_path = fixture_root / "uv"
    uv_path.write_bytes(b"#!/bin/sh\nset -eu\nexit 0\n")
    uv_path.chmod(0o755)
    cache_path = fixture_root / "rlm-cache-v2.tar.gz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(cache_path, mode="w:gz") as cache_archive:
        directory = tarfile.TarInfo("cache")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        directory.mtime = 0
        cache_archive.addfile(directory)
    launcher_path = fixture_root / "rlm-wrapper"
    launcher_path.write_bytes(
        textwrap.dedent(
            r'''
            #!/usr/bin/env python3
            import concurrent.futures
            import hashlib
            import json
            import os
            import sys
            import urllib.request

            def post(messages, headers):
                body = {
                    "model": os.environ["RLM_MODEL"],
                    "messages": messages,
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "ipython",
                            "description": "Execute code.",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                                "required": ["code"],
                            },
                        },
                    }],
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "reasoning_effort": None,
                    "min_p": 0.0,
                    "repetition_penalty": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "seed": 1,
                    "max_tokens": 768,
                    "n": 1,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
                wire_headers = {
                    "Authorization": "Bearer " + os.environ["RLM_API_KEY"],
                    "Content-Type": "application/json",
                    **headers,
                }
                request = urllib.request.Request(
                    os.environ["RLM_BASE_URL"].rstrip("/") + "/chat/completions",
                    data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
                    headers=wire_headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.loads(response.read())

            def assistant(payload):
                return payload["choices"][0]["message"]

            def root_headers(ordinal):
                return {
                    "X-RLM-Provenance-Version": "2",
                    "X-RLM-Depth": "0",
                    "X-RLM-Session-ID": "fixture-root",
                    "X-RLM-Lineage": "root",
                    "X-RLM-Session-Call-Ordinal": str(ordinal),
                    "X-RLM-Turn": str(ordinal),
                    "X-RLM-Call-Kind": "policy",
                    "X-RLM-Completed-Episode-Spawn-Ordinals": "",
                }

            def child_lineage(spawn):
                payload = json.dumps(
                    {
                        "domain": "redco.stage-d.spawn-lineage.v2",
                        "depth": 1,
                        "parent_lineage": "root",
                        "parent_call_ordinal": 0,
                        "parent_tool_call_slot": 0,
                        "spawn_ordinal": spawn,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                return "root/" + hashlib.sha256(payload).hexdigest()[:24]

            def child_headers(spawn, parent_tool_call_id):
                return {
                    "X-RLM-Provenance-Version": "2",
                    "X-RLM-Depth": "1",
                    "X-RLM-Session-ID": "fixture-child-" + str(spawn),
                    "X-RLM-Lineage": child_lineage(spawn),
                    "X-RLM-Session-Call-Ordinal": "0",
                    "X-RLM-Turn": "0",
                    "X-RLM-Call-Kind": "policy",
                    "X-RLM-Completed-Episode-Spawn-Ordinals": "",
                    "X-RLM-Parent-Session-ID": "fixture-root",
                    "X-RLM-Parent-Turn": "0",
                    "X-RLM-Parent-Tool-Call-ID": parent_tool_call_id,
                    "X-RLM-Invocation-ID": "fixture-child-" + str(spawn),
                    "X-RLM-Parent-Lineage": "root",
                    "X-RLM-Parent-Call-Ordinal": "0",
                    "X-RLM-Parent-Tool-Call-Slot": "0",
                    "X-RLM-Spawn-Ordinal": str(spawn),
                    "X-RLM-Episode-Spawn-Ordinal": str(spawn),
                    "X-RLM-Completed-Predecessor-Spawn-Ordinals": "",
                }

            prompt = sys.argv[-1] if len(sys.argv) > 1 else "Return candidate evidence."
            eligible_marker = os.environ.get("REDCO_FIXTURE_ELIGIBLE_MARKER", "")
            is_eligible = bool(eligible_marker) and eligible_marker in prompt
            root_zero = assistant(post(
                [{"role": "user", "content": prompt}],
                root_headers(0),
            ))
            root_tool_calls = root_zero.get("tool_calls")
            if not isinstance(root_tool_calls, list) or not root_tool_calls:
                raise RuntimeError("fixture renderer did not produce the root tool call")
            parent_tool_call = root_tool_calls[0]
            if not isinstance(parent_tool_call, dict):
                raise RuntimeError("fixture renderer returned an invalid tool call")
            parent_tool_call_id = parent_tool_call.get("id")
            if not isinstance(parent_tool_call_id, str):
                raise RuntimeError("fixture renderer returned no tool-call ID")
            root_one = assistant(post(
                [
                    {"role": "user", "content": prompt},
                    root_zero,
                    {"role": "tool", "tool_call_id": parent_tool_call_id, "content": "computed"},
                ],
                root_headers(1),
            ))
            root_one_tool_calls = root_one.get("tool_calls")
            if not isinstance(root_one_tool_calls, list) or not root_one_tool_calls:
                root_one_tool_call_id = None
            else:
                root_one_tool_call = root_one_tool_calls[0]
                if not isinstance(root_one_tool_call, dict):
                    raise RuntimeError("fixture renderer returned an invalid second tool call")
                root_one_tool_call_id = root_one_tool_call.get("id")
                if not isinstance(root_one_tool_call_id, str):
                    raise RuntimeError("fixture renderer returned no second tool-call ID")
            root_two = None
            root_two_tool_call_id = None
            if not is_eligible:
                if root_one_tool_call_id is None:
                    raise RuntimeError("fixture noneligible root lost its second tool call")
                root_two = assistant(post(
                    [
                        {"role": "user", "content": prompt},
                        root_zero,
                        {
                            "role": "tool",
                            "tool_call_id": parent_tool_call_id,
                            "content": "computed",
                        },
                        root_one,
                        {
                            "role": "tool",
                            "tool_call_id": root_one_tool_call_id,
                            "content": "computed",
                        },
                    ],
                    root_headers(2),
                ))
                root_two_tool_calls = root_two.get("tool_calls")
                if not isinstance(root_two_tool_calls, list) or not root_two_tool_calls:
                    raise RuntimeError("fixture noneligible root lost its third tool call")
                root_two_tool_call = root_two_tool_calls[0]
                if not isinstance(root_two_tool_call, dict):
                    raise RuntimeError("fixture renderer returned an invalid third tool call")
                root_two_tool_call_id = root_two_tool_call.get("id")
                if not isinstance(root_two_tool_call_id, str):
                    raise RuntimeError("fixture renderer returned no third tool-call ID")
            child_messages = [
                {"role": "user", "content": prompt},
                root_zero,
                {"role": "tool", "tool_call_id": parent_tool_call_id, "content": "computed"},
                root_one,
            ]
            if root_one_tool_call_id is not None:
                child_messages.append(
                    {"role": "tool", "tool_call_id": root_one_tool_call_id, "content": "computed"}
                )
            if root_two is not None and root_two_tool_call_id is not None:
                child_messages.extend(
                    [
                        root_two,
                        {
                            "role": "tool",
                            "tool_call_id": root_two_tool_call_id,
                            "content": "computed",
                        },
                    ]
                )
            child_spawns = (0, 1) if is_eligible else (0,)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(child_spawns)
            ) as pool:
                futures = [
                    pool.submit(
                        post,
                        child_messages + [{"role": "user", "content": "child-" + str(spawn)}],
                        child_headers(spawn, parent_tool_call_id),
                    )
                    for spawn in child_spawns
                ]
            children = [future.result() for future in futures]
            print(json.dumps({"children": len(children), "status": "fixture evidence"}))
            '''
        ).lstrip().encode("utf-8")
    )
    launcher_path.chmod(0o755)
    lock_sha256 = sha256_bytes((rlm_tree / "uv.lock").read_bytes())
    fixture_bindings = {
        "checkout_archive_path": str(archive_path),
        "checkout_archive_sha256": sha256_bytes(archive_path.read_bytes()),
        "checkout_uv_path": str(uv_path),
        "checkout_uv_sha256": sha256_bytes(uv_path.read_bytes()),
        "checkout_cache_archive_path": str(cache_path),
        "checkout_cache_archive_sha256": sha256_bytes(cache_path.read_bytes()),
        "checkout_uv_lock_sha256": lock_sha256,
        "checkout_launcher_path": str(launcher_path),
        "checkout_launcher_sha256": sha256_bytes(launcher_path.read_bytes()),
    }
    fixture_dependency = json.loads(
        (ROOT / "configs/stage-d/stage-d1-dependency-stack-v12.json").read_bytes()
    )
    fixture_dependency["rlm_archive_sha256"] = fixture_bindings["checkout_archive_sha256"]
    fixture_dependency["rlm_uv_binary_sha256"] = fixture_bindings["checkout_uv_sha256"]
    fixture_dependency["rlm_uv_cache_archive_sha256"] = fixture_bindings[
        "checkout_cache_archive_sha256"
    ]
    fixture_dependency["rlm_executable_sha256"] = fixture_bindings[
        "checkout_launcher_sha256"
    ]
    fixture_dependency["uv_lock_sha256"] = lock_sha256
    dependency_path = fixture_root / "dependency-stack.json"
    dependency_path.write_bytes(canonical_json_bytes(fixture_dependency))
    dependency_sha256 = sha256_bytes(dependency_path.read_bytes())
    original_dependency_path = launch.DEPENDENCY_MANIFEST_RELATIVE
    monkeypatch.setattr(launch, "DEPENDENCY_MANIFEST_RELATIVE", str(dependency_path))
    monkeypatch.delitem(launch.FROZEN_ROOT_HASHES, original_dependency_path)
    monkeypatch.setitem(launch.FROZEN_ROOT_HASHES, str(dependency_path), dependency_sha256)
    monkeypatch.setattr(launch, "EXPECTED_DEPENDENCY_STACK_SHA256", dependency_sha256)
    monkeypatch.setattr(launch, "EXPECTED_UV_LOCK_SHA256", lock_sha256)
    monkeypatch.setattr(launch, "OFFLINE_RLM_BINDINGS", fixture_bindings)
    for relative in sorted(launch.LAUNCH_BUNDLE_PATHS):
        source = ROOT / relative
        target = clone / relative
        if not source.is_file() or source.is_symlink():
            raise AssertionError(f"launch allowlist input is missing: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in sorted(set(launch.FROZEN_ROOT_HASHES) | set(launch.POLICY_FILES)):
        source = ROOT / relative
        target = clone / relative
        if not source.is_file() or source.is_symlink():
            raise AssertionError(f"authenticated launch input is missing: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            shutil.copyfile(source, target)
    vendor_lock = clone / ".redco" / "vendor" / "rlm" / "uv.lock"
    vendor_lock.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / ".redco" / "vendor" / "rlm" / "uv.lock", vendor_lock)
    key_path = tmp_path / "fixture-signing-key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)],
        check=True,
        capture_output=True,
    )
    public = key_path.with_name(key_path.name + ".pub").read_text(encoding="ascii")
    key_type, key_base64, *_ = public.strip().split()
    fingerprint = subprocess.run(
        [
            "ssh-keygen",
            "-lf",
            str(key_path.with_name(key_path.name + ".pub")),
            "-E",
            "sha256",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[1]
    identity = lifecycle.SigningIdentity(
        public_key_type=key_type,
        public_key_base64=key_base64,
        fingerprint_sha256=fingerprint,
        principal="fixture-launch",
        namespace=lifecycle.HANDOFF_V2_NAMESPACE,
        allowed_signers_sha256=sha256_bytes(
            f"fixture-launch {key_type} {key_base64}\n".encode("ascii")
        ),
    )
    monkeypatch.setattr(launch, "launch_signing_identity", lambda: identity)
    payloads = launch.build_launch_artifacts(clone)
    for relative, value in payloads.items():
        path = clone / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    subprocess.run(
        ["git", "-C", str(clone), "config", "user.email", "fixture@redco.test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "config", "user.name", "Redco fixture"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "add",
            "-f",
            "--",
            *sorted(launch.LAUNCH_BUNDLE_PATHS),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "commit", "-qm", "disposable launch integration"],
        check=True,
        capture_output=True,
    )
    head = (
        subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    tree = (
        subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    assert subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD^"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == launch.PARENT_COMMIT
    assert launch.verify_launch_bundle(clone, require_post_commit=True)

    outputs = _prime_fixture_outputs()

    def _run_command(
        _self: Any,
        argv: tuple[str, ...],
    ) -> subprocess.CompletedProcess[bytes]:
        name = next(
            candidate
            for candidate, command in observations.PRIME_READ_ONLY_COMMANDS.items()
            if tuple(command) == tuple(argv)
        )
        return subprocess.CompletedProcess(argv, 0, outputs[name], b"")

    monkeypatch.setattr(observations.PrimeObservationProducer, "_run_command", _run_command)
    fixture_prime = observations.PrimeObservationProducer(clone)

    prime_path = clone / launch.LAUNCH_PRIME_OBSERVATION_RELATIVE
    prime_path.parent.mkdir(parents=True, exist_ok=True)
    prime_path.write_bytes(
        fixture_prime.capture(captured_at_epoch=int(time.time()))
    )
    asset_paths = {
        relative: (
            clone / relative,
            digest,
        )
        for relative, digest in launch.POLICY_FILES.items()
    }
    asset_paths["uv.lock"] = (
        rlm_tree / "uv.lock",
        lock_sha256,
    )

    class Response:
        def __init__(self, body: bytes) -> None:
            self.status = 200
            self._body = body

        def read(self) -> bytes:
            return self._body

    health = canonical_json_bytes({"status": "ready"})
    models = canonical_json_bytes(
        {"data": [{"id": "/workspace/models/stage-d1-merged"}]}
    )

    def opener(request: Any) -> Response:
        suffix = request.full_url.rsplit("8000", 1)[-1]
        return Response(health if suffix == "/health" else models)

    pod_path = clone / launch.LAUNCH_POD_OBSERVATION_RELATIVE
    pod_path.parent.mkdir(parents=True, exist_ok=True)
    pod_path.write_bytes(
        observations.capture_pod_runtime_observation(
            base_url="http://127.0.0.1:8000",
            asset_paths=asset_paths,
            opener=opener,
            runtime_probe=canonical_json_bytes(
                {"datasets": "5.0.0", "pyarrow": "25.0.0", "python": "3.12.3"}
            ),
        )
    )
    prime = json.loads(prime_path.read_bytes())
    resource = prime["resource"]
    assert isinstance(resource, dict)
    wallet = json.loads(outputs["wallet"])
    ledger_path = clone / launch.LAUNCH_PROVISIONING_LEDGER_RELATIVE
    provisioning = lifecycle.ProvisioningLedger.create(
        ledger_path,
        "fixture-launch-campaign",
        wallet_before={
            "wallet_id": wallet["wallet_id"],
            "team_id": wallet["team_id"],
            "wallet_usd": wallet["balance_usd"],
            "recent_billings": wallet["recent_billings"],
        },
    )
    resource_identity = {
        key: resource[key] for key in lifecycle.HANDOFF_V2_RESOURCE_KEYS
    }
    provisioning.record_provision(
        provision_id="fixture-provision-1",
        resource_id=resource_identity["resource_id"],
        cost_usd=0,
        billing_cursor="fixture-billing-cursor",
    )
    provisioning.bind_provision("fixture-provision-1", "fixture-pod-1")
    known_hosts = clone / launch.LAUNCH_KNOWN_HOSTS_RELATIVE
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    known_hosts.write_bytes(
        b"fixture.example ssh-ed25519 "
        b"AAAAC3NzaC1lZDI1NTE5AAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
    )
    handoff = clone / launch.LAUNCH_HANDOFF_RELATIVE
    signature = clone / launch.LAUNCH_HANDOFF_SIGNATURE_RELATIVE
    lifecycle.issue_execute_handoff_v2(
        handoff,
        signature,
        bundle={"commit": head, "tree": tree},
        launch_authorization_sha256=sha256_bytes(
            (clone / launch.LAUNCH_AUTH_RELATIVE).read_bytes()
        ),
        frozen_support_protocol_sha256=launch.PROTOCOL_ROOT_SHA256,
        prime_observation_sha256=sha256_bytes(prime_path.read_bytes()),
        resource_identity=resource_identity,
        resource_price_usd=float(resource["hourly_rate_usd"]),
        pod_id="fixture-pod-1",
        pod_name="redco-fixture-launch",
        pod_status_sha256="f" * 64,
        ssh={"user": "fixture", "host": "fixture.example", "port": 2201},
        known_hosts_sha256=sha256_bytes(known_hosts.read_bytes()),
        known_hosts_fingerprints=("SHA256:fixture",),
        ledger=provisioning,
        signing_key=key_path,
        signer=identity,
        provisioning_ordinal=1,
    )

    fixture_dir = tmp_path / "fixture-boundary"
    fixture_dir.mkdir()
    event_log = tmp_path / "owner-events.jsonl"
    launch_rows = [
        json.loads(line)
        for line in (clone / LAUNCH_DATASET_RELATIVE).read_bytes().splitlines()
    ]
    assert launch_rows and isinstance(launch_rows[0], dict)
    eligible_marker = "Question: " + str(launch_rows[0]["question"])
    eligible_reply = repr(launch_rows[0]["reference_evidence"])
    sitecustomize = textwrap.dedent(
        r"""
        from __future__ import annotations

        import json
        import os
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        _source_overlay = os.environ.get("REDCO_TEST_SOURCE_OVERLAY")
        if _source_overlay:
            sys.path.insert(0, _source_overlay)

        from redco.analysis import stage_d_v13_support_launch as launch
        from redco.analysis import stage_d_v13_support_launch_runtime as runtime
        from redco.analysis import stage_d_v13_launch_lifecycle as lifecycle
        from redco.contracts import canonical_json
        from redco_evidence_selection_v2 import source_env as fixture_source_env
        from renderers.base import (
            ParsedResponse,
            ParsedToolCall,
            RenderedTokens,
            ToolCallParseStatus,
        )

        _events = Path(os.environ["REDCO_FIXTURE_EVENT_LOG"])
        _journal = Path(os.environ["REDCO_FIXTURE_LEDGER"]) .with_name(
            Path(os.environ["REDCO_FIXTURE_LEDGER"]).name + ".dispatch.jsonl"
        )
        _identity = lifecycle.SigningIdentity.from_payload(
            json.loads(os.environ["REDCO_FIXTURE_SIGNING_IDENTITY"])
        )
        _fixture_dependency_path = os.environ["REDCO_FIXTURE_DEPENDENCY_PATH"]
        _fixture_dependency_sha256 = os.environ["REDCO_FIXTURE_DEPENDENCY_SHA256"]
        _fixture_lock_sha256 = os.environ["REDCO_FIXTURE_LOCK_SHA256"]
        _fixture_bindings = json.loads(os.environ["REDCO_FIXTURE_RLM_BINDINGS"])
        _eligible_marker = os.environ["REDCO_FIXTURE_ELIGIBLE_MARKER"]
        _eligible_reply = os.environ["REDCO_FIXTURE_ELIGIBLE_REPLY"]
        _original_dependency_path = launch.DEPENDENCY_MANIFEST_RELATIVE
        launch.DEPENDENCY_MANIFEST_RELATIVE = _fixture_dependency_path
        launch.FROZEN_ROOT_HASHES.pop(_original_dependency_path, None)
        launch.FROZEN_ROOT_HASHES[_fixture_dependency_path] = _fixture_dependency_sha256
        launch.EXPECTED_DEPENDENCY_STACK_SHA256 = _fixture_dependency_sha256
        launch.EXPECTED_UV_LOCK_SHA256 = _fixture_lock_sha256
        launch.OFFLINE_RLM_BINDINGS = _fixture_bindings
        fixture_source_env.CONTEXT_PATH = Path(
            os.environ["REDCO_FIXTURE_CONTEXT_PATH"]
        )

        def _event(value: str) -> None:
            with _events.open("ab") as stream:
                stream.write(value.encode("utf-8") + bytes((10,)))

        launch.launch_signing_identity = lambda: _identity
        _original_execute = launch.execute_support_once

        def _execute(*args: object, **kwargs: object) -> object:
            _event("execute_support_once")
            return _original_execute(*args, **kwargs)

        launch.execute_support_once = _execute
        _original_run_once = runtime.ProductionSupportActuator.run_once

        def _run_once(self: object, *args: object, **kwargs: object) -> object:
            _event("ProductionSupportActuator.run_once")
            return _original_run_once(self, *args, **kwargs)

        runtime.ProductionSupportActuator.run_once = _run_once

        class _Renderer:
            supports_tools = True

            def __init__(self) -> None:
                self._eligible_value: bool | None = None

            def _eligible(self, messages: list[dict[str, object]]) -> bool:
                user_messages = [
                    message
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "user"
                ]
                if self._eligible_value is None:
                    self._eligible_value = bool(user_messages) and _eligible_marker in str(
                        user_messages[-1].get("content", "")
                    )
                return self._eligible_value

            def _ids(self, messages: list[dict[str, object]]) -> list[int]:
                if any(
                    str(message.get("content", "")).startswith("child-")
                    for message in messages
                    if isinstance(message, dict)
                ):
                    return [102, 2]
                if not self._eligible(messages):
                    return [104, 2]
                if any(
                    message.get("role") == "tool"
                    for message in messages
                    if isinstance(message, dict)
                ):
                    return [103, 2]
                return [101, 2]

            def render_ids(
                self,
                messages: list[dict[str, object]],
                *,
                tools: object = None,
                add_generation_prompt: bool = False,
            ) -> list[int]:
                del tools
                if not add_generation_prompt:
                    raise AssertionError("fixture renderer requires a generation prompt")
                return self._ids(messages)

            def render(
                self,
                messages: list[dict[str, object]],
                *,
                tools: object = None,
                add_generation_prompt: bool = False,
            ) -> RenderedTokens:
                ids = self.render_ids(
                    messages,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                )
                return RenderedTokens(
                    token_ids=ids,
                    message_indices=[0] * len(ids),
                    sampled_mask=[False] * len(ids),
                    is_content=[False] * len(ids),
                    message_roles=["user"] * len(ids),
                )

            def bridge_to_next_turn(
                self,
                prompt_ids: list[int],
                completion_ids: list[int],
                messages: list[dict[str, object]],
                *,
                tools: object = None,
            ) -> RenderedTokens:
                if any(
                    str(message.get("content", "")).startswith("child-")
                    for message in messages
                    if isinstance(message, dict)
                ):
                    return None
                tail = self.render(messages, tools=tools, add_generation_prompt=True)
                prefix_length = len(prompt_ids) + len(completion_ids)
                return RenderedTokens(
                    token_ids=prompt_ids + completion_ids + tail.token_ids,
                    message_indices=[-1] * prefix_length + tail.message_indices,
                    sampled_mask=[False] * prefix_length + tail.sampled_mask,
                    is_content=[False] * prefix_length + tail.is_content,
                    message_roles=["unknown"] * prefix_length + tail.message_roles,
                )

            def get_stop_token_ids(self) -> list[int]:
                return [151645]

            def parse_response(
                self,
                completion_ids: list[int],
                *,
                tools: object = None,
            ) -> ParsedResponse:
                del tools
                _event("parse_response:" + json.dumps(completion_ids))
                if completion_ids and completion_ids[0] in {101, 103, 104}:
                    code = "\n".join(
                        (
                        "import asyncio",
                        "children = await asyncio.gather(",
                        '    rlm("Return candidate verbatim evidence.", '
                        'redco_invocation_id="midpoint-shard-0"),',
                        '    rlm("Return candidate verbatim evidence.", '
                        'redco_invocation_id="midpoint-shard-1"),',
                            ")",
                            "children",
                        )
                    )
                    raw = json.dumps(
                        {"name": "ipython", "arguments": {"code": code}},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    return ParsedResponse(
                        content="",
                        tool_calls=[
                            ParsedToolCall(
                                raw=raw,
                                name="ipython",
                                arguments={"code": code},
                                status=ToolCallParseStatus.OK,
                                id="call_0",
                            )
                        ],
                    )
                return ParsedResponse(content=_eligible_reply)

        class _OpenAI:
            base_url = "http://127.0.0.1:8000/v1"
            max_retries = 0

            async def post(self, *_args: object, **kwargs: object) -> object:
                body = kwargs.get("body")
                if not isinstance(body, dict):
                    raise AssertionError("fixture provider request body is missing")
                token_ids = body.get("token_ids")
                _event("renderer_post:" + json.dumps(token_ids))
                count = _journal.read_bytes().count(bytes((10,))) if _journal.is_file() else 0
                _event("provider_post:" + str(count))
                turn_marker = (
                    token_ids[-2]
                    if isinstance(token_ids, list) and len(token_ids) >= 2
                    else None
                )
                if turn_marker in {101, 103, 104}:
                    completion = [int(turn_marker), 2]
                else:
                    completion = [102, 151645]
                finish_reason = "tool_calls" if completion[0] in {101, 103, 104} else "stop"
                return SimpleNamespace(
                    content=canonical_json(
                        {
                            "request_id": "fixture-provider-" + str(count),
                            "choices": [
                                {
                                    "token_ids": completion,
                                    "logprobs": {
                                        "content": [
                                            {"logprob": -0.2},
                                            {"logprob": -0.1},
                                        ]
                                    },
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }
                    )
                )

            async def close(self) -> None:
                return None

        def _configure(client: object) -> object:
            from verifiers.v1.clients.train import TrainClient

            if not isinstance(client, TrainClient):
                raise TypeError("fixture resolver did not return the pinned TrainClient")
            client._pool = _Renderer()
            client.openai = _OpenAI()
            if hasattr(client, "_openai"):
                client._openai = client.openai
            return client

        import verifiers.v1.cli.eval.runner as eval_runner
        import verifiers.v1.clients as clients
        import verifiers.v1.clients.train as train

        for _module in (eval_runner, clients, train):
            _original_resolve = getattr(_module, "resolve_client", None)
            if callable(_original_resolve):
                def _resolve(config: object, _original=_original_resolve) -> object:
                    return _configure(_original(config))

                _module.resolve_client = _resolve
        """
    )
    (fixture_dir / "sitecustomize.py").write_text(sitecustomize, encoding="utf-8")
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            (
                str(fixture_dir),
                # The disposable child contains the committed launch bundle;
                # import the current source-contract implementation so this
                # uncommitted recovery check exercises the patch under test.
                str(ROOT / "src"),
                str(clone / "src"),
                str(ROOT / "scripts"),
                str(clone / "scripts"),
                str(clone / "environments" / "redco_evidence_selection_v2"),
                str(clone / "external" / "prime-rl" / "deps" / "renderers"),
                str(clone / "external" / "prime-rl" / "deps" / "verifiers"),
            )
        ),
        "REDCO_FIXTURE_EVENT_LOG": str(event_log),
        "REDCO_FIXTURE_LEDGER": str(ledger_path),
        "REDCO_FIXTURE_SIGNING_IDENTITY": json.dumps(identity.to_payload()),
        "REDCO_FIXTURE_DEPENDENCY_PATH": str(dependency_path),
        "REDCO_FIXTURE_DEPENDENCY_SHA256": dependency_sha256,
        "REDCO_FIXTURE_LOCK_SHA256": lock_sha256,
        "REDCO_FIXTURE_RLM_BINDINGS": json.dumps(fixture_bindings),
        "REDCO_FIXTURE_CONTEXT_PATH": str(fixture_root / "evidence_context.txt"),
        "REDCO_TEST_SOURCE_OVERLAY": str(ROOT / "src"),
        "REDCO_FIXTURE_ELIGIBLE_MARKER": eligible_marker,
        "REDCO_FIXTURE_ELIGIBLE_REPLY": eligible_reply,
        "VLLM_API_KEY": "fixture",
    }
    command = [
        sys.executable,
        str(clone / "scripts/run_stage_d_v13_support.py"),
        "--execute-once",
        "--repository",
        str(clone),
        "--preflight-observation",
        str(prime_path),
        "--pod-runtime-observation",
        str(pod_path),
        "--capability",
        str(handoff),
        "--capability-signature",
        str(signature),
    ]
    completed = subprocess.run(
        command,
        cwd=clone,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, (
        completed.stderr
        + "\nfixture events:\n"
        + (event_log.read_text(encoding="utf-8") if event_log.is_file() else "")
    )
    output = json.loads(completed.stdout)
    assert output["mode"] == "execute-once"
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert "execute_support_once" in events
    assert "ProductionSupportActuator.run_once" in events
    provider_events = [event for event in events if event.startswith("provider_post:")]
    assert provider_events and all(int(event.rsplit(":", 1)[1]) > 0 for event in provider_events)

    runtime_root = clone / runtime.RUNTIME_ROOT_RELATIVE
    manifest_path = runtime_root / runtime.EXECUTION_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["source_count"] == 64
    assert manifest["branch_count_k"] == 4
    assert manifest["branch_target_artifact_bijection"] is True
    assert manifest["retry"] is False
    receipt_path = runtime_root / "collection-receipt.json"
    roster_path = runtime_root / "target-roster" / "target-roster.json"
    roster = json.loads(roster_path.read_bytes())
    targets = roster["targets"]
    eligible = roster["eligible_source_count"]
    assert eligible == 1
    assert len(targets) == 2 * eligible
    branch_paths = tuple(sorted((runtime_root / "branch-results").glob("*.json")))
    assert {path.name for path in branch_paths} == set(
        runtime.expected_branch_artifact_keys(roster)
    )
    assert len(branch_paths) == 2 * eligible
    assert all(len(json.loads(path.read_bytes())["arms"]) == 4 for path in branch_paths)
    report_path = clone / launch.LAUNCH_SUPPORT_REPORT_RELATIVE
    report = json.loads(report_path.read_bytes())
    nested = report["nested_support"]
    assert nested["N_scaffold"] == 1
    assert nested["N_eligible"] == 1
    assert nested["N_joint"] == 1
    assert nested["N_scaffold"] <= nested["N_eligible"] <= 64
    assert nested["N_joint"] <= nested["N_eligible"]
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["planned_slot_count"] == 64
    provisioning_state = json.loads(ledger_path.read_bytes())
    assert provisioning_state["provider_post_count"] > 0
    assert provisioning_state["closed"] is True
    assert json.loads(
        (clone / launch.LAUNCH_ATTEMPT_RELATIVE).read_bytes()
    ).get("retry") is False
    terminal = json.loads((clone / launch.LAUNCH_TERMINAL_RELATIVE).read_bytes())
    assert terminal["state"] == "completed_support_only"
    assert terminal["science_authorized"] is False
    assert terminal["scientific_transition_authorized"] is False
    assert terminal["provider_dispatch_observed"] is True
    assert receipt_path.is_file()
    assert report_path.is_file()
    assert manifest_path.is_file()
def test_operator_asset_root_symlink_is_rejected_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    observation = repository / "observation.json"
    observation.write_bytes(b"{}")
    ssh_key = repository / "id_ed25519"
    ssh_key.write_bytes(b"synthetic-key")
    target = tmp_path / "assets"
    target.mkdir()
    linked = tmp_path / "assets-link"
    try:
        os.symlink(target, linked, target_is_directory=True)
    except OSError:
        monkeypatch.setattr(
            local_orchestrator,
            "_is_link_or_reparse",
            lambda path: path == linked.absolute(),
        )
    monkeypatch.setattr(
        local_orchestrator,
        "_run",
        _prime_must_not_run,
    )
    with pytest.raises(ValueError, match="linked/reparse"):
        local_orchestrator.LocalLaunchOrchestrator(
            repository,
            observation,
            ssh_key,
            linked,
        )
    assert not (repository / "runs").exists()


def test_operator_asset_root_reparse_ancestor_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    observation = repository / "observation.json"
    observation.write_bytes(b"{}")
    ssh_key = repository / "id_ed25519"
    ssh_key.write_bytes(b"synthetic-key")
    reparse_parent = tmp_path / "junction-parent"
    reparse_parent.mkdir()
    target = reparse_parent / "assets"
    target.mkdir()
    monkeypatch.setattr(
        local_orchestrator,
        "_is_link_or_reparse",
        lambda path: path == reparse_parent.absolute(),
    )
    monkeypatch.setattr(
        local_orchestrator,
        "_run",
        _prime_must_not_run,
    )
    with pytest.raises(ValueError, match="linked/reparse"):
        local_orchestrator.LocalLaunchOrchestrator(
            repository,
            observation,
            ssh_key,
            target,
        )
    assert not (repository / "runs").exists()


def test_operator_ssh_key_symlink_is_rejected_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    observation = repository / "observation.json"
    observation.write_bytes(b"{}")
    asset_root = repository / "assets"
    asset_root.mkdir()
    target = tmp_path / "id_ed25519"
    target.write_bytes(b"synthetic-key")
    linked = tmp_path / "id-ed25519-link"
    try:
        os.symlink(target, linked)
    except OSError:
        monkeypatch.setattr(
            local_orchestrator,
            "_is_link_or_reparse",
            lambda path: path == linked.absolute(),
        )
    monkeypatch.setattr(
        local_orchestrator,
        "_run",
        _prime_must_not_run,
    )
    with pytest.raises(ValueError, match="linked/reparse"):
        local_orchestrator.LocalLaunchOrchestrator(
            repository,
            observation,
            linked,
            asset_root,
        )
    assert not (repository / "runs").exists()


def test_operator_ssh_key_reparse_ancestor_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    observation = repository / "observation.json"
    observation.write_bytes(b"{}")
    asset_root = repository / "assets"
    asset_root.mkdir()
    reparse_parent = tmp_path / "ssh-junction-parent"
    reparse_parent.mkdir()
    target = reparse_parent / "id_ed25519"
    target.write_bytes(b"synthetic-key")
    monkeypatch.setattr(
        local_orchestrator,
        "_is_link_or_reparse",
        lambda path: path == reparse_parent.absolute(),
    )
    monkeypatch.setattr(
        local_orchestrator,
        "_run",
        _prime_must_not_run,
    )
    with pytest.raises(ValueError, match="linked/reparse"):
        local_orchestrator.LocalLaunchOrchestrator(
            repository,
            observation,
            target,
            asset_root,
        )
    assert not (repository / "runs").exists()


def test_operator_paths_accept_normal_exact_paths(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    ssh_key = tmp_path / "id_ed25519"
    ssh_key.write_bytes(b"synthetic-key")
    assert local_orchestrator._operator_path(
        asset_root,
        "asset root",
        require_directory=True,
    ) == asset_root.resolve()
    assert local_orchestrator._operator_path(
        ssh_key,
        "SSH key",
        require_file=True,
    ) == ssh_key.resolve()


def test_handoff_consumer_has_no_caller_byte_overrides() -> None:
    parameters = inspect.signature(lifecycle.consume_execute_handoff_v2).parameters
    assert "handoff_bytes" not in parameters
    assert "signature_bytes" not in parameters


def test_asset_contract_separates_windows_sources_from_linux_destinations(
    tmp_path: Path,
) -> None:
    contract = launch.asset_binding_contract(ROOT)
    assert contract
    assert all(
        value["local_locator"].startswith(("repository:", "artifact-store:"))
        and value["remote_destination"].startswith("/workspace/")
        for value in contract.values()
    )
    base = next(
        value for name, value in contract.items() if name.startswith("base-model:")
    )
    adapter = contract["adapter:model"]
    artifact_root = tmp_path / "D-artifact-store"
    artifact_root.mkdir()
    base_path = launch.resolve_local_asset_locator(
        ROOT,
        artifact_root,
        base["local_locator"],
    )
    adapter_path = launch.resolve_local_asset_locator(
        ROOT,
        artifact_root,
        adapter["local_locator"],
    )
    assert base_path.is_relative_to(artifact_root)
    assert adapter_path.is_relative_to(artifact_root)
    assert "workspace" not in str(base_path).replace("\\", "/").lower()
    assert "workspace" not in str(adapter_path).replace("\\", "/").lower()
    for value in contract.values():
        local_path = launch.resolve_local_asset_locator(
            ROOT,
            artifact_root,
            value["local_locator"],
        )
        approved_root = (
            ROOT
            if value["local_locator"].startswith("repository:")
            else artifact_root
        )
        assert local_path.is_relative_to(approved_root)
        assert not value["remote_destination"].startswith(("C:", "D:"))
        assert "/workspace" not in value["local_locator"]
    with pytest.raises(ValueError, match="escapes"):
        launch.resolve_local_asset_locator(
            ROOT,
            artifact_root,
            "artifact-store:../workspace/model",
        )


def test_extracted_uv_cache_is_the_runtime_cache(tmp_path: Path) -> None:
    uv_source = shutil.which("uv")
    if uv_source is None:
        raise AssertionError("the uv executable is required for the offline cache probe")
    root = tmp_path / "remote"
    uv_path = root / bootstrap.UV_RELATIVE
    uv_path.parent.mkdir(parents=True)
    shutil.copy2(uv_source, uv_path)
    cache = root / bootstrap.UV_CACHE_RELATIVE
    cache.mkdir(parents=True)
    bootstrap._verify_uv_cache_directory(root)
    assert bootstrap._runtime_env(root)["UV_CACHE_DIR"] == str(cache)


def test_execute_once_requires_signed_handoff_before_any_attempt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fixed Prime handoff path"):
        execute_support_once(
            tmp_path,
            preflight_observation=tmp_path / "observation.json",
            pod_runtime_observation=tmp_path / "pod-observation.json",
            capability=tmp_path / "handoff.json",
            capability_signature=tmp_path / "handoff.sig",
        )
    assert not (tmp_path / launch.LAUNCH_ATTEMPT_RELATIVE).exists()
def _prime_fixture_outputs() -> dict[str, bytes]:
    resource = {
        "id": "resource-001",
        "cloud_id": "cloud-001",
        "gpu_type": "L40S",
        "provider": "prime",
        "location": "us-test-1",
        "gpu_count": 2,
        "socket": "socket-0",
        "stock_status": "available",
        "price_per_hour": 2,
        "price_value": 2,
        "security": {"ssh": True},
        "vcpus": 16,
        "memory_gb": 128,
        "disk_gb": 128,
        "gpu_memory": 48,
        "is_spot": False,
    }
    return {
        "version": b"Prime CLI version: 0.6.20\n",
        "wallet": canonical_json_bytes(
            {
                "wallet_id": "acct-1",
                "team_id": "team-1",
                "balance_usd": 30,
                "currency": "USD",
                "total_billings": 0,
                "recent_billings": [],
            }
        ),
        "pods": canonical_json_bytes({"pods": [], "total_count": 0, "offset": 0, "limit": 100}),
        "disks": canonical_json_bytes({"disks": [], "total_count": 0, "offset": 0, "limit": 100}),
        "availability": canonical_json_bytes(
            {"gpu_resources": [resource], "total_count": 1, "filters": {"gpu_count": 2}}
        ),
    }


def test_prime_observation_is_raw_and_bundle_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs = _prime_fixture_outputs()

    def run_command(self: Any, argv: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        name = next(
            name
            for name, command in observations.PRIME_READ_ONLY_COMMANDS.items()
            if command == argv
        )
        return subprocess.CompletedProcess(argv, 0, outputs[name], b"")

    monkeypatch.setattr(observations.PrimeObservationProducer, "_run_command", run_command)
    producer = observations.PrimeObservationProducer(ROOT)
    value = producer.capture(captured_at_epoch=int(time.time()))
    path = tmp_path / "prime-observation.json"
    path.write_bytes(value)
    parsed = observations.validate_prime_observation(ROOT, path)
    assert parsed["resource"]["resource_id"] == "resource-001"
    assert parsed["commands"]["wallet"]["stdout_sha256"] == sha256_bytes(outputs["wallet"])
    mutated = json.loads(value)
    mutated["resource"]["resource_id"] = "operator-asserted"
    path.write_bytes(canonical_json_bytes(mutated))
    with pytest.raises(ValueError, match="differs"):
        observations.validate_prime_observation(ROOT, path)


def test_prime_observation_rejects_stale_inventory_and_resource_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs = _prime_fixture_outputs()

    def run_command(self: Any, argv: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        name = next(
            name
            for name, command in observations.PRIME_READ_ONLY_COMMANDS.items()
            if command == argv
        )
        return subprocess.CompletedProcess(argv, 0, outputs[name], b"")

    monkeypatch.setattr(observations.PrimeObservationProducer, "_run_command", run_command)
    path = tmp_path / "prime-observation.json"
    path.write_bytes(observations.PrimeObservationProducer(ROOT).capture())
    mutated = json.loads(path.read_bytes())
    mutated["inventory"]["pods"] = [{"pod_id": "duplicate"}]
    path.write_bytes(canonical_json_bytes(mutated))
    with pytest.raises(ValueError, match="differs"):
        observations.validate_prime_observation(ROOT, path)


def test_pod_runtime_observation_uses_health_and_models_only(tmp_path: Path) -> None:
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"asset")
    expected = sha256_bytes(b"asset")

    class Response:
        def __init__(self, body: bytes) -> None:
            self.status = 200
            self._body = body

        def read(self) -> bytes:
            return self._body

    responses = {
        "/health": Response(canonical_json_bytes({"status": "ready"})),
        "/v1/models": Response(canonical_json_bytes({"data": [{"id": "model"}]})),
    }

    def opener(request: Any) -> Response:
        return responses[request.full_url.split("8000", 1)[1]]

    value = observations.capture_pod_runtime_observation(
        base_url="http://127.0.0.1:8000",
        asset_paths={"asset": (asset, expected)},
        opener=opener,
        expected_model_ids=("model",),
    )
    parsed = observations.validate_pod_runtime_observation(
        value,
        expected_model_ids=("model",),
    )
    assert parsed["completion_requests"] == 0


def test_branch_artifact_keys_are_roster_derived_and_two_per_eligible() -> None:
    roster: dict[str, Any] = {
        "schema_version": 2,
        "domain": "redco-stage-d-branch-target-roster-v2",
        "planned_source_count": 4,
        "completed_source_count": 4,
        "eligible_source_count": 2,
        "ineligible_source_count": 2,
        "minimum_eligible_sources": 2,
        "eligibility_passed": True,
        "source_sha256s": ["a", "b", "c", "d"],
        "targets": [
            {"group_id": "g1", "target_id": "t1", "source_sha256": "a"},
            {"group_id": "g1", "target_id": "t2", "source_sha256": "a"},
            {"group_id": "g2", "target_id": "t1", "source_sha256": "b"},
            {"group_id": "g2", "target_id": "t2", "source_sha256": "b"},
        ],
        "excluded_targets": [
            {"ineligibility_reason": "natural_topology_ineligible"},
            {"ineligibility_reason": "natural_topology_ineligible"},
        ],
    }
    assert runtime.expected_branch_artifact_keys(roster) == (
        "g1--t1.json",
        "g1--t2.json",
        "g2--t1.json",
        "g2--t2.json",
    )
    roster["targets"].pop()
    with pytest.raises(ValueError, match="exactly two"):
        runtime.expected_branch_artifact_keys(roster)


def test_provisioning_replacement_and_dispatch_are_irreversible(tmp_path: Path) -> None:
    ledger = lifecycle.ProvisioningLedger.create(tmp_path / "provisioning.json", "campaign-1")
    ledger.record_provision(
        provision_id="p1",
        resource_id="r1",
        cost_usd=2,
        billing_cursor="c1",
    )
    assert ledger.replacement_allowed()
    ledger.record_provision(
        provision_id="p2",
        resource_id="r2",
        cost_usd=2,
        billing_cursor="c2",
    )
    assert not ledger.replacement_allowed()
    order: list[str] = []
    boundary = lifecycle.ProviderDispatchBoundary(ledger)
    boundary.send("op-1", lambda: order.append("post"))
    assert order == ["post"]
    boundary.send("op-2", lambda: order.append("post-2"))
    assert order == ["post", "post-2"]
    with pytest.raises(RuntimeError, match="already observed"):
        boundary.send("op-2", lambda: order.append("duplicate"))
    assert order == ["post", "post-2"]


def test_real_child_dispatch_callback_uses_the_durable_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "provisioning.json"
    ledger = lifecycle.ProvisioningLedger.create(path, "campaign-callback")
    ledger.record_provision(
        provision_id="p1",
        resource_id="r1",
        cost_usd=2,
        billing_cursor="c1",
    )
    monkeypatch.setenv(lifecycle.PROVISIONING_LEDGER_ENV, str(path))
    callback = lifecycle.dispatch_callback_from_environment()
    assert callback is not None
    callback(b"prepared-source-request")
    callback(b"prepared-branch-request")
    state = json.loads(path.read_bytes())
    assert state["provider_post_count"] == 2
    assert state["provider_post_observed"] is True
    assert lifecycle.ProvisioningLedger.open(path).replacement_allowed() is False


def test_execute_handoff_v2_is_signed_and_consumed_once(tmp_path: Path) -> None:
    key_path = tmp_path / "signing-key"
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)],
        check=True,
        capture_output=True,
    )
    public = key_path.with_name(key_path.name + ".pub").read_text(encoding="ascii")
    key_type, key_base64, *_ = public.strip().split()
    fingerprint = subprocess.run(
        ["ssh-keygen", "-lf", str(key_path.with_name(key_path.name + ".pub")), "-E", "sha256"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[1]
    identity = lifecycle.SigningIdentity(
        public_key_type=key_type,
        public_key_base64=key_base64,
        fingerprint_sha256=fingerprint,
        principal="test-principal",
        namespace=lifecycle.HANDOFF_V2_NAMESPACE,
        allowed_signers_sha256=sha256_bytes(
            f"test-principal {key_type} {key_base64}\n".encode("ascii")
        ),
    )
    ledger_path = tmp_path / "provisioning.json"
    ledger = lifecycle.ProvisioningLedger.create(
        ledger_path,
        "campaign-handoff",
        wallet_before={"wallet_id": "wallet-1", "team_id": "team-1", "wallet_usd": 30},
    )
    ledger.record_provision(
        provision_id="provision-1",
        resource_id="resource-1",
        billing_cursor="cursor-1",
    )
    ledger.bind_provision("provision-1", "pod-1")
    handoff = tmp_path / "execute-handoff-v2.json"
    signature = tmp_path / "execute-handoff-v2.sig"
    claim = tmp_path / "provision-claim-v2.json"
    bundle = {"commit": "c" * 40, "tree": "d" * 40}
    resource_identity = {
        "resource_id": "resource-1",
        "provider": "prime",
        "location": "test-zone",
        "gpu_type": "L40",
        "gpu_count": 2,
        "memory_gb": 48,
        "is_spot": False,
        "security": {"persistent_storage": False},
    }
    ssh = {"user": "operator", "host": "example.test", "port": 2201}
    lifecycle.issue_execute_handoff_v2(
        handoff,
        signature,
        bundle=bundle,
        launch_authorization_sha256="a" * 64,
        frozen_support_protocol_sha256="b" * 64,
        prime_observation_sha256="e" * 64,
        resource_identity=resource_identity,
        resource_price_usd=2.0,
        pod_id="pod-1",
        pod_name="redco-test-pod",
        pod_status_sha256="f" * 64,
        ssh=ssh,
        known_hosts_sha256="a" * 64,
        known_hosts_fingerprints=("SHA256:test",),
        ledger=ledger,
        signing_key=key_path,
        signer=identity,
        provisioning_ordinal=1,
        now_epoch=1000,
    )
    original = handoff.read_bytes()
    forged = json.loads(original)
    forged["unknown_null"] = None
    forged_path = tmp_path / "forged-handoff-v2.json"
    forged_signature = tmp_path / "forged-handoff-v2.sig"
    forged_claim = tmp_path / "forged-claim-v2.json"
    forged_path.write_bytes(canonical_json_bytes(forged))
    shutil.copyfile(signature, forged_signature)
    with pytest.raises(RuntimeError, match="fields differ"):
        lifecycle.consume_execute_handoff_v2(
            forged_path,
            forged_signature,
            forged_claim,
            identity=identity,
            bundle=bundle,
            launch_authorization_sha256="a" * 64,
            frozen_support_protocol_sha256="b" * 64,
            prime_observation_sha256="e" * 64,
            resource_identity=resource_identity,
            pod_id="pod-1",
            pod_name="redco-test-pod",
            pod_status_sha256="f" * 64,
            ssh=ssh,
            known_hosts_sha256="a" * 64,
            known_hosts_fingerprints=("SHA256:test",),
            ledger=lifecycle.ProvisioningLedger.open(ledger_path),
            now_epoch=1001,
        )
    assert not forged_claim.exists()
    assert handoff.read_bytes() == original
    lifecycle.consume_execute_handoff_v2(
        handoff,
        signature,
        claim,
        identity=identity,
        bundle=bundle,
        launch_authorization_sha256="a" * 64,
        frozen_support_protocol_sha256="b" * 64,
        prime_observation_sha256="e" * 64,
        resource_identity=resource_identity,
        pod_id="pod-1",
        pod_name="redco-test-pod",
        pod_status_sha256="f" * 64,
        ssh=ssh,
        known_hosts_sha256="a" * 64,
        known_hosts_fingerprints=("SHA256:test",),
        ledger=lifecycle.ProvisioningLedger.open(ledger_path),
        now_epoch=1001,
    )
    consumed_claim = json.loads(claim.read_bytes())
    assert consumed_claim["handoff_bytes_b64"] == base64.b64encode(original).decode(
        "ascii"
    )
    assert consumed_claim["signature_bytes_b64"] == base64.b64encode(
        signature.read_bytes()
    ).decode("ascii")
    with pytest.raises(RuntimeError, match="already consumed"):
        lifecycle.consume_execute_handoff_v2(
            handoff,
            signature,
            tmp_path / "second-claim.json",
            identity=identity,
            bundle=bundle,
            launch_authorization_sha256="a" * 64,
            frozen_support_protocol_sha256="b" * 64,
            prime_observation_sha256="e" * 64,
            resource_identity=resource_identity,
            pod_id="pod-1",
            pod_name="redco-test-pod",
            pod_status_sha256="f" * 64,
            ssh=ssh,
            known_hosts_sha256="a" * 64,
            known_hosts_fingerprints=("SHA256:test",),
            ledger=lifecycle.ProvisioningLedger.open(ledger_path),
            now_epoch=1001,
        )


def test_prime_0620_command_contract_is_exact() -> None:
    assert observations.PRIME_CLI_VERSION == "0.6.20"
    assert observations.PRIME_READ_ONLY_COMMANDS == {
        "version": ("prime", "--version"),
        "wallet": ("prime", "--plain", "wallet", "--limit", "100", "--output", "json"),
        "pods": ("prime", "--plain", "pods", "list", "--output", "json"),
        "disks": ("prime", "--plain", "disks", "list", "--output", "json"),
        "availability": (
            "prime",
            "--plain",
            "availability",
            "list",
            "--gpu-count",
            "2",
            "--output",
            "json",
        ),
    }


def test_prime_0620_ssh_endpoint_requires_documented_port_for_ip() -> None:
    assert LocalLaunchOrchestrator._parse_ssh_endpoint(
        "operator@pod.example -p 2201,operator@other.example -p 2202"
    ) == ("operator", "pod.example", 2201)
    with pytest.raises(RuntimeError, match="documented SSH endpoint"):
        LocalLaunchOrchestrator._parse_ssh_endpoint("203.0.113.10")
    assert LocalLaunchOrchestrator._endpoint_from_status(
        {
            "ip": "203.0.113.10",
            "port_mappings": [{"internal": "22", "external": 2201}],
        }
    ) == "203.0.113.10 -p 2201"


def test_unsigned_execute_handoff_v1_is_retired() -> None:
    assert not hasattr(lifecycle, "issue_execute_handoff")
    assert not hasattr(lifecycle, "consume_execute_handoff")


def test_dispatch_journal_preserves_concurrent_posts(tmp_path: Path) -> None:
    path = tmp_path / "campaign.json"
    ledger = lifecycle.ProvisioningLedger.create(path, "concurrent-campaign")
    ledger.record_provision(
        provision_id="pod-1",
        resource_id="resource-1",
        cost_usd=0,
        billing_cursor="synthetic",
    )

    def send(index: int) -> None:
        lifecycle.record_provider_post_from_path(path, f"request-{index}".encode())

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(send, range(8)))
    state = lifecycle.ProvisioningLedger.open(path)._state()
    assert state["provider_post_count"] == 8
    assert len(state["provider_operation_ids"]) == 8
    journal = path.with_name(path.name + lifecycle.DISPATCH_JOURNAL_SUFFIX)
    assert journal.read_bytes().count(b"\n") == 8


def test_billing_reconciliation_uses_wallet_rows_not_rate(tmp_path: Path) -> None:
    path = tmp_path / "campaign.json"
    before = {
        "wallet_id": "wallet-1",
        "team_id": "team-1",
        "currency": "USD",
        "wallet_usd": 30,
        "recent_billings": [],
    }
    after = {
        **before,
        "wallet_usd": 28.5,
        "recent_billings": [
            {
                "id": "billing-1",
                "created_at": "2026-08-07T00:00:00Z",
                "updated_at": "2026-08-07T00:00:00Z",
                "amount_usd": 1.5,
                "currency": "USD",
                "resource_type": "pod",
            }
        ],
    }
    ledger = lifecycle.ProvisioningLedger.create(path, "billing-campaign", wallet_before=before)
    ledger.record_provision(
        provision_id="pod-1",
        resource_id="resource-1",
        billing_cursor="cursor-before",
    )
    assert ledger.reconcile_billing(after) == 1.5
    state = lifecycle.ProvisioningLedger.open(path)._state()
    assert state["cumulative_cost_usd"] == 1.5
    assert state["provisions"][0]["billing_status"] == "reconciled"
