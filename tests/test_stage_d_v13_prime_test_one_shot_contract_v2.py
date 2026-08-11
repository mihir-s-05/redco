"""CPU-only contract tests for the current-lineage Prime test one-shot."""

from __future__ import annotations

import ast
import base64
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis import stage_d_v13_prime_test_one_shot_contract_v2 as contract
from redco.analysis.stage_d_v13_prime_test_one_shot_remote_v2 import (
    ALLOWED_DEVICE_NAMES,
    remote_test_script,
    transitive_test_bindings,
    validate_gpu_facts,
    validate_junit,
    validate_known_hosts,
)

ROOT = Path(__file__).parents[1].resolve()


_GPU_MEMORY = {
    "NVIDIA L40": 46_068 * 1024 * 1024,
    "NVIDIA L40S": 46_068 * 1024 * 1024,
    "NVIDIA RTX 6000 Ada": 49_140 * 1024 * 1024,
    "NVIDIA RTX 6000 Ada Generation": 49_140 * 1024 * 1024,
}


def _gpu(name: str = "NVIDIA L40S", memory: int | None = None) -> bytes:
    resolved_memory = _GPU_MEMORY[name] if memory is None else memory
    return contract.canonical_json(
        {
            "schema_version": 2,
            "device_count": 2,
            "names": [name, name],
            "memory_bytes": [resolved_memory, resolved_memory],
            "selected_nominal_aggregate_gb": 96,
            "observed_aggregate_bytes": 2 * resolved_memory,
            "torch": "fixture",
            "cuda": "fixture",
        }
    )


def _selected(gpu_type: str = "L40S 48GB") -> dict[str, object]:
    return {
        "gpu_type": gpu_type,
        "gpu_count": 2,
        "gpu_memory_gb": 96,
        "is_spot": False,
        "hourly_rate_usd": 1.64,
        "disk_size": 0,
    }


def _ssh_blob(algorithm: str, *fields: bytes) -> str:
    values = (algorithm.encode(), *fields)
    return base64.b64encode(
        b"".join(struct.pack(">I", len(field)) + field for field in values)
    ).decode()


_VALID_RSA_BLOB = (
    "AAAAB3NzaC1yc2EAAAADAQABAAABAQC3Gb/rOlR0rMY/ilVefyI9LXjn9n8XsOpI/6wdeEDe8O7a2MweozgxrFFPuSx7H/W1WoR3apBwvVOOyyTK4YYMw1Yc/oSPHlYg5nO1mEmZePV6YaQIGUk3UqkJXONYYdj7XBijrnmzI+w48DilRaQoOL98R14ClnIzi6V0eOibacw2u1RmjKLJl6FTAcUnjxdMQm+sl6/7xs5dhvSGg+06nZMuv2ncyVDCKiibRxCykWMkUQTueryA/0/iiaTJ7ye5/Oz8/5WF5T2/J+xOj3bkxuqkzAqsypLOVwp2UYNBkAS6LDMVq4ZFTIPlnj8zuvNgdkE43TRZXJPMI99FvTfB"
)


def test_readiness_is_non_authorizing_and_deterministic() -> None:
    contract.authenticate_readiness(ROOT, committed=False)
    first = contract.build_readiness_artifacts(ROOT)
    second = contract.build_readiness_artifacts(ROOT)
    assert first == second
    value = json.loads(first[contract.CONTRACT_PATH])
    assert value["state"] == "non_authorizing_readiness"
    assert not any(value["authority"].values())
    assert contract.AUTHORIZATION_PATH not in contract.READINESS_PATHS
    assert not (ROOT / contract.AUTHORIZATION_PATH).exists()


def test_production_context_reaches_authenticated_client_without_request_or_evidence(
    tmp_path: Path,
) -> None:
    key = tmp_path / "key"
    subprocess.run(
        [
            str(v5.OPENSSH_EXECUTABLE_PATH),
            "-q",
            "-t",
            "rsa",
            "-b",
            "2048",
            "-N",
            "",
            "-f",
            str(key),
        ],
        check=True,
        capture_output=True,
    )
    key_type, public, *_rest = key.with_suffix(".pub").read_text(encoding="ascii").split()
    allowed = f"mihir {key_type} {public}\n".encode()
    identity = [
        "mihir",
        key_type,
        public,
        v5._fingerprint(key_type, public),
        contract.sha256_bytes(allowed),
    ]
    authorization = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "parent": "c" * 40,
        "authorization_path": "authorization.json",
        "authorization_sha256": "d" * 64,
        "authorization_blob": "e" * 40,
    }
    script = """
import json, sys
from pathlib import Path
from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis import stage_d_v13_prime_test_one_shot_prime_v2 as owner
identity = v5._TerminalSigningIdentity(*json.loads(sys.argv[2]))
authorization = json.loads(sys.argv[3])
v5._load_terminal_signing_identity = lambda: identity
owner.authenticate_authorization = lambda _root: authorization
context = owner._production_context(Path(sys.argv[1]))
assert context.client.base_url == v5.BASE_URL
print("production-context-ok")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    before = (ROOT / contract.EVIDENCE_ROOT).exists()
    result = subprocess.run(
        [
            str(Path(os.environ["APPDATA"]) / "uv/tools/prime/Scripts/python.exe"),
            "-c",
            script,
            str(key),
            json.dumps(identity),
            json.dumps(authorization),
        ],
        check=False,
        capture_output=True,
        timeout=30,
        env=environment,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b"production-context-ok\r\n"
    assert (ROOT / contract.EVIDENCE_ROOT).exists() == before


def test_authorization_cannot_be_self_issued_from_parent_checkout() -> None:
    with pytest.raises(ValueError, match="readiness parent"):
        contract.authorization_value(ROOT, current_is_authorization=False)
    with pytest.raises(ValueError):
        contract.authenticate_authorization(ROOT)


def test_public_runner_canonicalizes_git_subprocess_failures_without_side_effects(
    tmp_path: Path,
) -> None:
    runner_path = ROOT / contract.RUNNER_SCRIPT
    script = """
import importlib.util, subprocess, sys
from pathlib import Path
from redco.analysis import stage_d_v13_prime_test_one_shot_contract_v2 as contract
spec = importlib.util.spec_from_file_location("one_shot_runner", sys.argv[1])
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
kind = sys.argv[2]
root = Path(sys.argv[3])
def fail():
    target = root if kind == "TimeoutExpired" else root / "not-a-repository"
    timeout = 0 if kind == "TimeoutExpired" else None
    return subprocess.run(
        [str(contract.GIT_EXECUTABLE["path"]), "-C", str(target), "rev-parse", "HEAD"],
        check=True, capture_output=True, timeout=timeout,
    )
runner.run_prime_test_one_shot_v2 = fail
raise SystemExit(runner.main())
"""
    before_authorization = (ROOT / contract.AUTHORIZATION_PATH).exists()
    before_evidence = (ROOT / contract.EVIDENCE_ROOT).exists()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    for failure in ("CalledProcessError", "TimeoutExpired"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(runner_path), failure, str(ROOT)],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            timeout=30,
            env=environment,
        )
        expected = contract.canonical_json(
            {
                "schema_version": 2,
                "state": "failed_terminal",
                "failure": failure,
                "live_result": False,
            }
        ) + b"\r\n"
        assert result.returncode == 20 and result.stdout == expected
        assert result.stderr == b""
        assert not (tmp_path / contract.EVIDENCE_ROOT).exists()
    assert (ROOT / contract.AUTHORIZATION_PATH).exists() == before_authorization
    assert (ROOT / contract.EVIDENCE_ROOT).exists() == before_evidence


def test_test_plan_transitive_bindings_are_source_model_free() -> None:
    bindings = transitive_test_bindings(ROOT)
    paths = {str(item["path"]) for item in bindings}
    assert {node.partition("::")[0] for node in contract.TEST_NODES} <= paths
    script = remote_test_script("a" * 40, _selected()).decode()
    assert "--junitxml=.runtime/pytest.xml" in script
    assert "--frozen --no-sync" in script
    assert "load_dataset" not in script
    assert "from_pretrained" not in script
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in script
    assert script.index("git clone --no-checkout") < script.index("git checkout --detach")
    assert script.index("git checkout --detach") < script.index("git submodule update")


def test_gpu_contract_rejects_wrong_model_or_memory() -> None:
    mutations = [
        ("NVIDIA A100-SXM4-80GB", 46_068 * 1024 * 1024),
        ("NVIDIA H100 80GB HBM3", 46_068 * 1024 * 1024),
        ("NVIDIA L40S", 47_185_919_999),
        ("NVIDIA L40S", 48_305_799_169),
    ]
    for name, memory in mutations:
        with pytest.raises(ValueError, match="hardware"):
            validate_gpu_facts(_gpu(name, memory), _selected())


def test_gpu_contract_accepts_only_frozen_names() -> None:
    selected = {
        "NVIDIA L40": "L40 48GB",
        "NVIDIA L40S": "L40S 48GB",
        "NVIDIA RTX 6000 Ada": "RTX6000Ada 48GB",
        "NVIDIA RTX 6000 Ada Generation": "RTX6000Ada 48GB",
    }
    for name in ALLOWED_DEVICE_NAMES:
        assert validate_gpu_facts(_gpu(name), _selected(selected[name]))["device_count"] == 2


def test_gpu_contract_is_bound_to_label_specific_repository_telemetry() -> None:
    bindings = contract.GPU_TELEMETRY_BINDING
    for key, marker, total in (
        ("L40S", b"NVIDIA L40S", 46_068),
        ("RTX6000Ada", b"NVIDIA RTX 6000 Ada Generation", 49_140),
    ):
        binding = bindings[key]
        raw = (ROOT / cast(str, binding["path"])).read_bytes()
        assert len(raw) == binding["bytes"]
        assert contract.sha256_bytes(raw) == binding["sha256"]
        assert raw.count(marker) == binding["device_count"] == 2
        assert raw.count(f"{total} MiB".encode()) == 2
    l40 = bindings["L40"]
    assert l40["evidence_kind"] == "conservative_l40s_48gb_class_bound"
    assert l40["bound_source"] == "L40S"
    assert validate_gpu_facts(_gpu(), _selected())["device_count"] == 2


def test_gpu_contract_rejects_mixed_or_wrong_inventory_aggregate() -> None:
    value = json.loads(_gpu())
    value["names"][1] = "NVIDIA L40"
    with pytest.raises(ValueError, match="hardware"):
        validate_gpu_facts(contract.canonical_json(value), _selected())
    value = json.loads(_gpu())
    value["selected_nominal_aggregate_gb"] = 95
    with pytest.raises(ValueError, match="hardware"):
        validate_gpu_facts(contract.canonical_json(value), _selected())


def test_gpu_contract_rejects_every_cross_class_and_selected_projection_mutation() -> None:
    pairs = (
        ("NVIDIA L40", "L40S 48GB"),
        ("NVIDIA L40", "RTX6000Ada 48GB"),
        ("NVIDIA L40S", "L40 48GB"),
        ("NVIDIA L40S", "RTX6000Ada 48GB"),
        ("NVIDIA RTX 6000 Ada", "L40 48GB"),
        ("NVIDIA RTX 6000 Ada Generation", "L40S 48GB"),
    )
    for observed, selected in pairs:
        with pytest.raises(ValueError, match="hardware"):
            validate_gpu_facts(_gpu(observed), _selected(selected))
    for key, bad in (("gpu_count", 1), ("gpu_count", 4), ("gpu_memory_gb", 95)):
        projection = _selected()
        projection[key] = bad
        with pytest.raises(ValueError):
            validate_gpu_facts(_gpu(), projection)
    value = json.loads(_gpu())
    value["observed_aggregate_bytes"] += 1
    with pytest.raises(ValueError, match="hardware"):
        validate_gpu_facts(contract.canonical_json(value), _selected())


def test_junit_requires_each_frozen_node_exactly_once() -> None:
    cases = [
        (
            node.partition("::")[0].removesuffix(".py").replace("/", "."),
            node.partition("::")[2],
        )
        for node in contract.TEST_NODES
    ]

    def raw(values: list[tuple[str, str]], *, failures: int = 0) -> bytes:
        body = "".join(
            f'<testcase classname="{classname}" name="{name}"/>' for classname, name in values
        )
        return (
            f'<testsuite tests="{len(values)}" failures="{failures}" errors="0" '
            f'skipped="0">{body}</testsuite>'
        ).encode()

    validate_junit(raw(cases))
    for values in (cases[:-1], [*cases, cases[0]], [cases[0], cases[0], *cases[2:]]):
        with pytest.raises(ValueError):
            validate_junit(raw(values))
    with pytest.raises(ValueError):
        validate_junit(raw(cases, failures=1))


def test_git_executable_is_absolute_authenticated_and_path_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = tmp_path / "git.exe"
    fake.write_bytes(b"shadow")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert contract.authenticate_git_executable() == Path(
        cast(str, contract.GIT_EXECUTABLE["path"])
    )
    monkeypatch.setitem(contract.GIT_EXECUTABLE, "sha256", "0" * 64)
    with pytest.raises(ValueError, match="Git owner differs"):
        contract.authenticate_git_executable()


def test_git_launcher_owner_and_redirect_environment_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_DIR", "hostile")
    with pytest.raises(ValueError, match="environment redirects"):
        contract.authenticate_git_executable()
    monkeypatch.delenv("GIT_DIR")
    monkeypatch.setitem(contract.GIT_LAUNCHER, "sha256", "0" * 64)
    with pytest.raises(ValueError, match="launcher differs"):
        contract.authenticate_git_executable()


def test_git_replace_refs_are_ignored_and_metadata_redirects_rejected(tmp_path: Path) -> None:
    executable = contract.authenticate_git_executable()
    repository = tmp_path / "repository"
    environment = contract._git_environment()
    setup_environment = dict(environment)
    setup_environment.pop("GIT_NO_REPLACE_OBJECTS")

    def run(*arguments: str) -> str:
        return subprocess.run(
            (str(executable), *arguments),
            check=True,
            capture_output=True,
            text=True,
            env=setup_environment,
        ).stdout.strip()

    run("init", str(repository))
    run("-C", str(repository), "config", "user.name", "fixture")
    run("-C", str(repository), "config", "user.email", "fixture@example.invalid")
    tracked = repository / "value.txt"
    tracked.write_text("original", encoding="utf-8")
    run("-C", str(repository), "add", "value.txt")
    run("-C", str(repository), "commit", "-m", "original")
    original = run("-C", str(repository), "rev-parse", "HEAD")
    tracked.write_text("replacement", encoding="utf-8")
    run("-C", str(repository), "commit", "-am", "replacement")
    replacement = run("-C", str(repository), "rev-parse", "HEAD")
    run("-C", str(repository), "replace", original, replacement)
    assert run("-C", str(repository), "show", f"{original}:value.txt") == "replacement"
    assert contract.git_output(repository, "show", f"{original}:value.txt") == "original"
    for relative in ("info/grafts", "shallow", "objects/info/alternates"):
        candidate = repository / ".git" / relative
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text("hostile", encoding="utf-8")
        with pytest.raises(ValueError, match="object substitution"):
            contract.git_output(repository, "rev-parse", "HEAD")
        candidate.unlink()


def test_installed_wallet_api_owner_freezes_offset_pagination_contract() -> None:
    owner = (
        Path(os.environ["APPDATA"]) / "uv/tools/prime/Lib/site-packages" / contract.WALLET_API_OWNER
    )
    assert contract.sha256_bytes(owner.read_bytes()) == contract.WALLET_API_OWNER_SHA256
    tree = ast.parse(owner.read_text(encoding="utf-8"))
    wallet_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WalletClient"
    )
    get = next(
        node
        for node in wallet_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "get"
    )
    assert [ast.literal_eval(value) for value in get.args.defaults] == [20, 0, None]
    rendered = ast.unparse(get)
    assert "self.client.get" in rendered and "'/billing/wallet'" in rendered
    assert "'limit': limit" in rendered and "'offset': offset" in rendered
    assert contract.WALLET_PAGE_LIMIT == 100
    assert contract.MAX_PREEXISTING_WALLET_ROWS == 4096
    assert contract.MAX_NEW_BILLING_ROWS == 4096
    assert contract.MAX_POST_WALLET_ROWS == 8192
    assert contract.MAX_PRE_WALLET_PAGES == 41
    assert contract.MAX_POST_WALLET_PAGES == 82
    assert contract.MAX_PRE_WALLET_REQUESTS == 42
    assert contract.MAX_POST_WALLET_REQUESTS == 83
    assert os.path.normcase(str(owner)).endswith(os.path.normcase(contract.WALLET_API_OWNER))
    package = owner.parents[2]
    for relative, expected in (
        (contract.PRIME_CLIENT_OWNER, contract.PRIME_CLIENT_OWNER_SHA256),
        (contract.PRIME_CONFIG_OWNER, contract.PRIME_CONFIG_OWNER_SHA256),
    ):
        assert contract.sha256_bytes((package / relative).read_bytes()) == expected


def test_readiness_path_and_budget_are_frozen() -> None:
    assert contract.MAX_OPERATIONAL_PRIME_CLI_CALLS == 362
    assert contract.MAX_CLEANUP_PRIME_CLI_CALLS == 190
    assert contract.MAX_PRIME_CLI_CALLS == 552
    assert contract.MAX_WALLET_API_CALLS == 1038
    assert contract.MAXIMUM_POD_SECONDS * contract.MAXIMUM_RATE_USD / 3600 <= 12
    assert {
        contract.CONTRACT_MODULE,
        contract.REMOTE_MODULE,
        contract.PRIME_MODULE,
        contract.WALLET_MODULE,
        contract.EVIDENCE_MODULE,
        contract.LIFECYCLE_MODULE,
        contract.BUILDER_SCRIPT,
        contract.RUNNER_SCRIPT,
        contract.CONTRACT_PATH,
        contract.AUDIT_PATH,
        contract.CONTRACT_TEST,
        contract.EVIDENCE_TEST,
        contract.WALLET_TEST,
        contract.LIFECYCLE_TEST,
    } == contract.READINESS_PATHS


@pytest.mark.parametrize("algorithm,blob", [
    ("ssh-rsa", _VALID_RSA_BLOB),
    ("ssh-ed25519", _ssh_blob("ssh-ed25519", b"\x03" * 32)),
    (
        "ecdsa-sha2-nistp256",
        _ssh_blob(
            "ecdsa-sha2-nistp256",
            b"nistp256",
            b"\x04" + bytes.fromhex(
                "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"
                "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"
            ),
        ),
    ),
    (
        "ecdsa-sha2-nistp384",
        _ssh_blob(
            "ecdsa-sha2-nistp384",
            b"nistp384",
            b"\x04" + bytes.fromhex(
                "aa87ca22be8b05378eb1c71ef320ad746e1d3b628ba79b9859f741e082542a385502f25dbf55296c3a545e3872760ab7"
                "3617de4a96262c6f5d9e98bf9292dc29f8f41dbd289a147ce9da3113b5f0b8c00a60b1ce1d7e819d7a431d7c90ea0e5f"
            ),
        ),
    ),
])
def test_known_hosts_accepts_each_supported_wire_algorithm(algorithm: str, blob: str) -> None:
    validate_known_hosts(
        f"[8.8.8.8]:2222 {algorithm} {blob}\n".encode(),
        contract.sha256_bytes(b"8.8.8.8"), 2222,
    )
