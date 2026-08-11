from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from test_stage_d_v13_prime_test_one_shot_evidence_v2 import (
    _Client,
    _Commands,
    _item,
    _Transport,
)

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis import stage_d_v13_prime_test_one_shot_contract_v2 as v2
from redco.analysis import stage_d_v13_prime_test_one_shot_evidence_v2 as evidence
from redco.analysis import stage_d_v13_prime_test_one_shot_lifecycle_v2 as lifecycle
from redco.analysis import stage_d_v13_prime_test_one_shot_prime_v2 as prime
from redco.analysis import stage_d_v13_prime_test_one_shot_runtime_binding_v2 as runtime_binding
from redco.analysis import stage_d_v13_prime_test_one_shot_runtime_v3 as runtime_v3
from redco.analysis import stage_d_v13_prime_test_one_shot_successor_v3 as successor
from redco.analysis.stage_d_v13_prime_test_one_shot_runtime_v3 import V3_RUNTIME_BINDING

ROOT = Path(__file__).parents[1].resolve()


def test_public_v2_context_and_lifecycle_have_no_runtime_overrides() -> None:
    assert "production_context" not in prime.__all__
    assert not inspect.signature(runtime_binding._production_context_v2).parameters
    assert "evidence_root" not in inspect.signature(lifecycle._run_one_shot).parameters
    assert "_run_one_shot" not in lifecycle.__all__


def test_successor_is_non_authorizing_and_binds_terminal_history() -> None:
    # The historical names are relative to the v2 evidence root.
    before = {
        name: hashlib.sha256(
            (ROOT / successor.V2_EVIDENCE_ROOT / name).read_bytes()
        ).hexdigest()
        for name in successor.V2_TERMINAL_FILE_NAMES
    }
    state = successor.authenticate_successor(ROOT, committed=False)
    assert state == {"commit": successor.PARENT_COMMIT, "tree": successor.PARENT_TREE}
    first = successor.build_readiness_artifacts(ROOT, committed=False)
    second = successor.build_readiness_artifacts(ROOT, committed=False)
    assert first == second
    value = json.loads(first[successor.CONTRACT_PATH])
    assert value["state"] == "non_authorizing_successor_readiness"
    assert value["threat_model"] == successor.THREAT_MODEL
    assert value["evidence"]["predecessor_terminal"]["terminal"]["state"] == (
        "no_qualifying_capacity"
    )
    runtime = value["runtime_binding"]
    assert runtime["authorization_path"] == V3_RUNTIME_BINDING.authorization_path
    assert runtime["evidence_root"] == V3_RUNTIME_BINDING.evidence_root
    assert runtime["claim_domain"] == V3_RUNTIME_BINDING.claim_domain
    assert runtime["assessment"]["domain"] == V3_RUNTIME_BINDING.assessment_domain
    assert runtime["assessment"]["schema_version"] == V3_RUNTIME_BINDING.assessment_schema_version
    assert runtime["assessment"]["namespace"] == V3_RUNTIME_BINDING.assessment_namespace
    assert runtime["signed_envelope_domain"] == V3_RUNTIME_BINDING.signed_envelope_domain
    assert runtime["handoff_namespace"] == V3_RUNTIME_BINDING.handoff_namespace
    assert runtime["terminal"]["domain"] == V3_RUNTIME_BINDING.terminal_domain
    assert runtime["terminal"]["namespace"] == V3_RUNTIME_BINDING.terminal_namespace
    assert runtime["terminal"]["purpose"] == V3_RUNTIME_BINDING.terminal_purpose
    assert runtime["artifact_filenames"] == dict(V3_RUNTIME_BINDING.artifact_filenames)
    assert runtime["assessment"]["authority"] == dict(V3_RUNTIME_BINDING.assessment_authority)
    assert runtime["terminal"]["authority"] == dict(V3_RUNTIME_BINDING.terminal_authority)
    assert not any(value["authority"].values())
    assert not (ROOT / successor.AUTHORIZATION_PATH).exists()
    assert not (ROOT / successor.EVIDENCE_ROOT).exists()
    after = {
        name: hashlib.sha256(
            (ROOT / successor.V2_EVIDENCE_ROOT / name).read_bytes()
        ).hexdigest()
        for name in successor.V2_TERMINAL_FILE_NAMES
    }
    assert before == after


def test_successor_artifacts_are_deterministic_and_exclusive() -> None:
    spec = importlib.util.spec_from_file_location(
        "stage_d_v3_readiness_builder", ROOT / successor.BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build = cast(Callable[..., dict[str, str]], module.build)

    built = successor.build_readiness_artifacts(ROOT, committed=False)
    assert build(check_only=False, prepare_authorization=False) == {
        relative: successor.sha256_bytes(raw) for relative, raw in built.items()
    }
    successor.verify_readiness_artifacts(ROOT, committed=False)
    assert set(built) == {successor.CONTRACT_PATH, successor.AUDIT_PATH}
    audit = json.loads(built[successor.AUDIT_PATH])
    assert audit["successor"]["live_capture_executed"] is False
    assert audit["successor"]["prime_calls"] == 0
    assert audit["threat_model"] == successor.THREAT_MODEL
    assert audit["authority"] == successor.READINESS_AUTHORITY


def test_future_authorization_is_blocked_before_a_successor_commit() -> None:
    with pytest.raises(ValueError, match="committed v3 readiness child"):
        spec = importlib.util.spec_from_file_location(
            "stage_d_v3_readiness_builder_for_auth", ROOT / successor.BUILDER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        build = cast(Callable[..., dict[str, str]], module.build)

        build(check_only=False, prepare_authorization=True)
    assert not (ROOT / successor.AUTHORIZATION_PATH).exists()
    assert not (ROOT / successor.EVIDENCE_ROOT).exists()


def test_scope_text_is_exact_and_forbidden_surfaces_are_false() -> None:
    raw = successor.SUCCESSOR_AUTHORIZATION_TEXT.encode("utf-8")
    assert len(raw) == len(successor.SUCCESSOR_AUTHORIZATION_TEXT)
    assert hashlib.sha256(raw).hexdigest() == successor.SUCCESSOR_AUTHORIZATION_SHA256
    contract = json.loads(
        successor.build_readiness_artifacts(ROOT, committed=False)[successor.CONTRACT_PATH]
    )
    scope = contract["user_scope"]
    assert scope["no_monitoring"] is True
    assert scope["no_retry_after_observation"] is True
    assert scope["model_training_science_source_parquet"] is False
    assert all(value is False for value in successor.READINESS_AUTHORITY.values())


def test_predecessor_uses_canonical_signed_v2_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValueError("sentinel verifier")

    monkeypatch.setattr(successor, "verify_terminal_evidence", reject)
    with pytest.raises(ValueError, match="signed evidence"):
        successor.authenticate_successor(ROOT, committed=False)


def test_v3_rejects_v2_lineage_and_public_runner_is_pre_auth_fail_closed() -> None:
    with pytest.raises(ValueError):
        successor.authenticate_authorization_v3(ROOT)
    assert successor.EVIDENCE_ROOT != successor.V2_EVIDENCE_ROOT
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, str(ROOT / successor.RUNNER_PATH)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 20
    assert result.stderr == b""
    assert b"Traceback" not in result.stdout
    assert json.loads(result.stdout) == {
        "schema_version": 2,
        "state": "failed_terminal",
        "live_result": False,
    }
    assert not (ROOT / successor.EVIDENCE_ROOT).exists()
    assert not (ROOT / successor.AUTHORIZATION_PATH).exists()


def test_public_v3_completed_e2e_uses_committed_a3_and_v3_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--no-hardlinks", "--local", str(ROOT), str(repository)],
        check=True,
        capture_output=True,
    )
    for relative in sorted(successor.SOURCE_PATHS):
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    shutil.copytree(
        ROOT / successor.V2_EVIDENCE_ROOT,
        repository / successor.V2_EVIDENCE_ROOT,
    )
    def synthetic_status(_root: Path) -> set[tuple[str, str]]:
        if v2.git_output(repository, "rev-parse", "HEAD") != successor.PARENT_COMMIT:
            return {(" M", v2.EXTERNAL_GITLINK)}
        artifacts = (repository / successor.CONTRACT_PATH).is_file() and (
            repository / successor.AUDIT_PATH
        ).is_file()
        paths = successor.READINESS_PATHS if artifacts else successor.SOURCE_PATHS
        return (
            {(" M", v2.EXTERNAL_GITLINK)}
            | {(" M", path) for path in successor.MODIFIED_OWNER_PATHS if path in paths}
            | {("??", path) for path in paths if path not in successor.MODIFIED_OWNER_PATHS}
        )

    monkeypatch.setattr(successor, "_git_status", synthetic_status)
    monkeypatch.setattr(v2, "_external_binding", lambda _root: None)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "test"],
        check=True,
    )
    built = successor.build_readiness_artifacts(repository, committed=False)
    for relative, raw in built.items():
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    subprocess.run(
        ["git", "-C", str(repository), "add", *sorted(successor.READINESS_PATHS)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "readiness"],
        check=True,
        capture_output=True,
    )
    readiness = successor.current_head(repository)
    builder_spec = importlib.util.spec_from_file_location(
        "stage_d_v3_readiness_builder_after_r3", repository / successor.BUILDER_PATH
    )
    assert builder_spec is not None and builder_spec.loader is not None
    builder_module = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder_module)
    vars(builder_module)["ROOT"] = repository
    prepared = builder_module.build(check_only=False, prepare_authorization=True)
    assert prepared == {
        successor.AUTHORIZATION_PATH: successor.sha256_bytes(
            (repository / successor.AUTHORIZATION_PATH).read_bytes()
        )
    }
    subprocess.run(
        ["git", "-C", str(repository), "add", successor.AUTHORIZATION_PATH],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "authorization"],
        check=True,
        capture_output=True,
    )
    projection = successor.authenticate_authorization_v3(repository)
    assert projection["parent"] == readiness
    assert projection["authorization_path"] == successor.AUTHORIZATION_PATH
    assert projection["authorization_sha256"] == successor.sha256_bytes(
        (repository / successor.AUTHORIZATION_PATH).read_bytes()
    )

    fake_prime = tmp_path / "prime.exe"
    fake_keygen = tmp_path / "ssh-keygen.exe"
    fake_ssh = tmp_path / "ssh.exe"
    fake_scp = tmp_path / "scp.exe"
    fake_keyscan = tmp_path / "ssh-keyscan.exe"
    fake_uv = tmp_path / "uv"
    signing_key = tmp_path / "id_rsa"
    for path, content in (
        (fake_prime, b"prime-fixture"),
        (fake_keygen, b"keygen-fixture"),
        (fake_ssh, b"ssh-fixture"),
        (fake_scp, b"scp-fixture"),
        (fake_keyscan, b"keyscan-fixture"),
        (fake_uv, b"uv-fixture"),
        (signing_key, b"private-key-fixture"),
    ):
        path.write_bytes(content)
    terminal_identity = v5._load_terminal_signing_identity()
    identity = v2.SigningIdentity(
        terminal_identity.principal,
        terminal_identity.key_type,
        terminal_identity.public_key_base64,
        terminal_identity.fingerprint_sha256,
        terminal_identity.allowed_signers_sha256,
    )

    transport = _Transport([_item(1)])
    commands = _Commands(tmp_path)
    transport.commands = commands
    fake_client = _Client(transport)
    client_constructions: list[object] = []

    def construct_client() -> _Client:
        client_constructions.append(fake_client)
        return fake_client

    context_identity_ids: list[int] = []
    real_build_context = runtime_binding._build_production_context

    def build_context_once(binding: runtime_binding.RuntimeBinding) -> object:
        context = real_build_context(binding)
        context_identity_ids.append(id(cast(Any, context).identity))
        return context

    verified_identity_ids: list[int] = []
    real_verify_terminal = evidence.verify_terminal_evidence

    def verify_terminal_once(
        root: Path,
        identity: v2.SigningIdentity,
        *,
        binding: runtime_binding.RuntimeBinding = runtime_binding.V2_RUNTIME_BINDING,
    ) -> dict[str, Any]:
        verified_identity_ids.append(id(identity))
        return real_verify_terminal(root, identity, binding=binding)

    monkeypatch.setattr(runtime_binding, "_repository_root", lambda: repository)
    monkeypatch.setattr(runtime_binding, "_build_production_context", build_context_once)
    monkeypatch.setattr(lifecycle, "_build_production_context", build_context_once)
    monkeypatch.setattr(lifecycle, "verify_terminal_evidence", verify_terminal_once)
    monkeypatch.setattr(evidence, "verify_terminal_evidence", verify_terminal_once)
    monkeypatch.setattr(sys, "version_info", (3, 13, 2))
    monkeypatch.setattr(
        runtime_binding,
        "OPENSSH_EXECUTABLES",
        {
            name: {"path": str(path), "bytes": path.stat().st_size,
                   "sha256": runtime_binding.sha_file(path)}
            for name, path in {
                "ssh": fake_ssh,
                "scp": fake_scp,
                "ssh-keyscan": fake_keyscan,
            }.items()
        },
    )
    monkeypatch.setattr(runtime_binding, "LINUX_UV_PATH", fake_uv)
    monkeypatch.setattr(runtime_binding, "LINUX_UV_BYTES", fake_uv.stat().st_size)
    monkeypatch.setattr(runtime_binding, "LINUX_UV_SHA256", runtime_binding.sha_file(fake_uv))
    monkeypatch.setattr(
        v5,
        "authenticate_installed_capture_owners",
        lambda: {
            "prime_uv_tool": {
                "canonical_path": str(fake_prime),
                "sha256": runtime_binding.sha_file(fake_prime),
            }
        },
    )
    monkeypatch.setattr(runtime_binding, "_authenticate_source", lambda *_args: None)
    monkeypatch.setattr(
        v5,
        "authenticate_approved_openssh_executable",
        lambda: {"path": str(fake_keygen)},
    )
    monkeypatch.setattr(
        v5,
        "_load_terminal_signing_identity",
        lambda: terminal_identity,
    )
    monkeypatch.setattr(v5, "_authenticate_operator_key", lambda *_args: None)
    monkeypatch.setattr(v5, "_construct_api_client", construct_client)
    monkeypatch.setattr(v5, "_httpx_request_error_types", lambda: (TimeoutError,))
    assert not (repository / successor.EVIDENCE_ROOT).exists()

    home = tmp_path / "operator-home"
    key = home / ".ssh" / "id_rsa"
    key.parent.mkdir(parents=True)
    subprocess.run(
        [
            str(v5.OPENSSH_EXECUTABLE_PATH), "-q", "-t", "rsa", "-b", "2048", "-N", "",
            "-f", str(key),
        ],
        check=True,
        capture_output=True,
    )
    key_type, public, *_ = key.with_suffix(".pub").read_text(encoding="ascii").split()
    allowed = f"mihir {key_type} {public}\n".encode()
    monkeypatch.setattr(
        v5,
        "authenticate_approved_openssh_executable",
        lambda: {"path": str(v5.OPENSSH_EXECUTABLE_PATH)},
    )
    live_identity = v5._TerminalSigningIdentity(
        "mihir", key_type, public, v5._fingerprint(key_type, public),
        v2.sha256_bytes(allowed),
    )
    monkeypatch.setattr(successor, "_v2_signing_identity", lambda: identity)

    monkeypatch.setattr(runtime_binding, "_production_run", commands)
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setattr(
        v5,
        "authenticate_approved_openssh_executable",
        lambda: {"path": str(v5.OPENSSH_EXECUTABLE_PATH)},
    )
    monkeypatch.setattr(
        v5,
        "_load_terminal_signing_identity",
        lambda: live_identity,
    )
    monkeypatch.setattr(time, "time", lambda: 1_000.0)
    monkeypatch.setattr(time, "monotonic", lambda: 1.0)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    result = runtime_v3.run_prime_test_one_shot_v3()
    assert len(context_identity_ids) == 1
    assert len(client_constructions) == 1
    assert verified_identity_ids == context_identity_ids
    assert result.state == "completed"
    assert result.tests_passed and result.cleanup_proven and result.create_dispatched
    assert [method for method, _url, _payload in transport.calls].count("POST") == 1
    terminal = json.loads(
        (repository / successor.EVIDENCE_ROOT / "terminal.json").read_bytes()
    )
    assert set(terminal["authority"]) == set(successor.READINESS_AUTHORITY)
    assert terminal["authority"]["parquet_access_authorized"] is False
    assert terminal["tests_passed"] is True
    assert terminal["cleanup_proven"] is True
    assert "junit" in terminal["evidence_dag"]
    with pytest.raises(ValueError):
        runtime_v3.run_prime_test_one_shot_v3()
    assert [method for method, _url, _payload in transport.calls].count("POST") == 1


def test_v3_lifecycle_binds_a3_and_v3_root_without_external_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    root_calls = 0

    def forbidden_root(*_args: object, **_kwargs: object) -> Path:
        nonlocal root_calls
        root_calls += 1
        raise AssertionError("evidence root was reached")

    monkeypatch.setattr(lifecycle, "exclusive_runtime_root", forbidden_root)
    with pytest.raises(ValueError, match="canonical singleton"):
        lifecycle._run_one_shot(cast(runtime_binding.RuntimeBinding, object()))
    assert root_calls == 0
    assert not (repository / successor.EVIDENCE_ROOT).exists()
