"""Authenticated runtime bindings and context factories for the Prime one-shot."""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    ARTIFACT_FILENAMES,
    ASSESSMENT_DOMAIN,
    ASSESSMENT_NAMESPACE,
    ASSESSMENT_TTL_SECONDS,
    AUTHORIZATION_PATH,
    CLAIM_DOMAIN,
    EVIDENCE_ROOT,
    HANDOFF_NAMESPACE,
    OPENSSH_EXECUTABLES,
    PODS_API_OWNER,
    PODS_API_OWNER_SHA256,
    PODS_COMMAND_OWNER,
    PODS_COMMAND_OWNER_SHA256,
    PRIME_CLIENT_OWNER,
    PRIME_CLIENT_OWNER_SHA256,
    PRIME_CONFIG_OWNER,
    PRIME_CONFIG_OWNER_SHA256,
    READINESS_AUTHORITY,
    RUNTIME_AUTHORITY,
    SIGNED_ENVELOPE_DOMAIN,
    TERMINAL_DOMAIN,
    TERMINAL_NAMESPACE,
    TERMINAL_PURPOSE,
    V3_READINESS_AUTHORITY,
    V3_RUNTIME_AUTHORITY,
    V5_CONTRACT_PATH,
    V5_CONTRACT_SHA256,
    V5_OWNER_PATH,
    V5_OWNER_SHA256,
    CommandResult,
    SigningIdentity,
    sha256_bytes,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_remote_v2 import (
    LINUX_UV_BYTES,
    LINUX_UV_SHA256,
)


class APIClient(Protocol):
    base_url: str
    api_key: str
    client: Any
    config: Any


LINUX_UV_PATH = Path(r"\\wsl.localhost\Ubuntu\home\mihir\.local\uv-latest\uv")
_V2_AUTHENTICATOR = "authenticate_authorization"
_V3_AUTHENTICATOR = "authenticate_authorization_v3"
_V3_AUTHORIZATION_PATH = "configs/stage-d/stage-d1-prime-test-one-shot-authorization-v3.json"
_V3_EVIDENCE_ROOT = "runs/stage-d/stage-d1-prime-test-one-shot-v3"
_V3_CLAIM_DOMAIN = "redco-stage-d1-prime-test-one-shot-claim-v3"
_V3_TERMINAL_DOMAIN = "redco-stage-d1-prime-test-one-shot-terminal-v3"
_V3_TERMINAL_NAMESPACE = "redco-stage-d1-prime-test-one-shot-terminal-v3"
_V3_TERMINAL_PURPOSE = "source_free_prime_integration_tests_only"
_SCHEMA_KEYS = frozenset(
    {
        "claim",
        "assessment",
        "assessment-envelope",
        "handoff",
        "handoff-envelope",
        "terminal",
        "terminal-envelope",
        "result",
    }
)
@dataclass(frozen=True, slots=True)
class RuntimeBinding:
    authorization_path: str
    authorization_authenticator_name: str
    evidence_root: str
    claim_domain: str
    assessment_domain: str
    assessment_schema_version: int
    assessment_namespace: str
    signed_envelope_domain: str
    handoff_namespace: str
    terminal_domain: str
    terminal_namespace: str
    terminal_purpose: str
    claim_authority: tuple[tuple[str, bool], ...]
    assessment_authority: tuple[tuple[str, bool], ...]
    create_authority: tuple[tuple[str, bool], ...]
    terminal_authority: tuple[tuple[str, bool], ...]
    result_authority: tuple[tuple[str, bool], ...]
    handoff_authority: tuple[tuple[str, bool], ...]
    artifact_filenames: tuple[tuple[str, str], ...]
    assessment_ttl_seconds: int
    schema_versions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        strings = (
            self.authorization_path,
            self.evidence_root,
            self.authorization_authenticator_name,
            self.claim_domain,
            self.assessment_domain,
            self.assessment_namespace,
            self.signed_envelope_domain,
            self.handoff_namespace,
            self.terminal_domain,
            self.terminal_namespace,
            self.terminal_purpose,
        )
        if any(type(value) is not str or not value or any(ord(c) < 32 for c in value)
               for value in strings):
            raise ValueError("Prime one-shot runtime binding text differs")
        _safe_relative(self.authorization_path, directory=False)
        _safe_relative(self.evidence_root, directory=True)
        if self.authorization_authenticator_name not in {
            _V2_AUTHENTICATOR,
            _V3_AUTHENTICATOR,
        }:
            raise ValueError("Prime one-shot authenticator is not an approved owner")
        if type(self.assessment_schema_version) is not int or self.assessment_schema_version != 2:
            raise ValueError("Prime one-shot assessment schema differs")
        if (
            type(self.assessment_ttl_seconds) is not int
            or not 0 < self.assessment_ttl_seconds <= 900
        ):
            raise ValueError("Prime one-shot assessment TTL differs")
        version = self.authorization_authenticator_name
        if version == _V2_AUTHENTICATOR:
            expected_paths = (AUTHORIZATION_PATH, EVIDENCE_ROOT)
            expected_domains = (
                CLAIM_DOMAIN,
                ASSESSMENT_DOMAIN,
                ASSESSMENT_NAMESPACE,
                SIGNED_ENVELOPE_DOMAIN,
                HANDOFF_NAMESPACE,
                TERMINAL_DOMAIN,
                TERMINAL_NAMESPACE,
                TERMINAL_PURPOSE,
            )
            expected_keys = frozenset(READINESS_AUTHORITY)
            expected_maps = (
                RUNTIME_AUTHORITY,
                RUNTIME_AUTHORITY,
                READINESS_AUTHORITY,
                READINESS_AUTHORITY,
                READINESS_AUTHORITY,
                READINESS_AUTHORITY,
            )
        else:
            expected_paths = (_V3_AUTHORIZATION_PATH, _V3_EVIDENCE_ROOT)
            expected_domains = (
                _V3_CLAIM_DOMAIN,
                ASSESSMENT_DOMAIN,
                ASSESSMENT_NAMESPACE,
                SIGNED_ENVELOPE_DOMAIN,
                HANDOFF_NAMESPACE,
                _V3_TERMINAL_DOMAIN,
                _V3_TERMINAL_NAMESPACE,
                _V3_TERMINAL_PURPOSE,
            )
            expected_keys = frozenset(V3_READINESS_AUTHORITY)
            expected_maps = (
                V3_RUNTIME_AUTHORITY,
                V3_RUNTIME_AUTHORITY,
                V3_RUNTIME_AUTHORITY,
                V3_READINESS_AUTHORITY,
                V3_READINESS_AUTHORITY,
                V3_READINESS_AUTHORITY,
            )
        if (self.authorization_path, self.evidence_root) != expected_paths or (
            self.claim_domain,
            self.assessment_domain,
            self.assessment_namespace,
            self.signed_envelope_domain,
            self.handoff_namespace,
            self.terminal_domain,
            self.terminal_namespace,
            self.terminal_purpose,
        ) != expected_domains:
            raise ValueError("Prime one-shot runtime binding semantics differ")
        for actual, expected in zip(
            (
                self.claim_authority,
                self.assessment_authority,
                self.create_authority,
                self.terminal_authority,
                self.result_authority,
                self.handoff_authority,
            ),
            expected_maps,
            strict=True,
        ):
            if _authority(actual, expected_keys) != expected:
                raise ValueError("Prime one-shot authority binding differs")
        if _filenames(self.artifact_filenames) != ARTIFACT_FILENAMES:
            raise ValueError("Prime one-shot artifact binding differs")
        schema_versions = _schemas(self.schema_versions)
        if set(schema_versions) != _SCHEMA_KEYS or any(
            value != 2 for value in schema_versions.values()
        ):
            raise ValueError("Prime one-shot schema projection differs")

    @property
    def authorization_authenticator(self) -> Callable[[Path], Mapping[str, object]]:
        if self.authorization_authenticator_name == _V2_AUTHENTICATOR:
            from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
                authenticate_authorization as v2_authenticator,
            )

            return cast(
                Callable[[Path], Mapping[str, object]],
                v2_authenticator,
            )
        from redco.analysis.stage_d_v13_prime_test_one_shot_successor_v3 import (
            authenticate_authorization_v3 as v3_authenticator,
        )

        return cast(
            Callable[[Path], Mapping[str, object]],
            v3_authenticator,
        )


def _safe_relative(value: str, *, directory: bool) -> None:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if normalized != value or path.is_absolute() or ":" in normalized:
        raise ValueError("Prime one-shot runtime path is not repository-relative")
    parts = normalized.rstrip("/").split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("Prime one-shot runtime path is not canonical")
    if not directory and len(parts[-1]) == 0:
        raise ValueError("Prime one-shot runtime filename differs")


def _authority(
    entries: tuple[tuple[str, bool], ...], expected_keys: frozenset[str]
) -> dict[str, bool]:
    if type(entries) is not tuple:
        raise ValueError("Prime one-shot authority entries differ")
    result: dict[str, bool] = {}
    for entry in entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError("Prime one-shot authority entries differ")
        key, value = entry
        if type(key) is not str or type(value) is not bool or key in result:
            raise ValueError("Prime one-shot authority entries differ")
        result[key] = value
    if set(result) != expected_keys:
        raise ValueError("Prime one-shot authority key universe differs")
    return result


def _filenames(entries: tuple[tuple[str, str], ...]) -> dict[str, str]:
    if type(entries) is not tuple:
        raise ValueError("Prime one-shot artifact entries differ")
    result: dict[str, str] = {}
    for entry in entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError("Prime one-shot artifact entries differ")
        key, filename = entry
        if type(key) is not str or type(filename) is not str or key in result:
            raise ValueError("Prime one-shot artifact entries differ")
        if "/" in filename or "\\" in filename or filename in {"", ".", ".."}:
            raise ValueError("Prime one-shot artifact filename differs")
        if filename in result.values():
            raise ValueError("Prime one-shot artifact filenames are not unique")
        result[key] = filename
    return result


def _schemas(entries: tuple[tuple[str, int], ...]) -> dict[str, int]:
    if type(entries) is not tuple:
        raise ValueError("Prime one-shot schema entries differ")
    result: dict[str, int] = {}
    for entry in entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError("Prime one-shot schema entries differ")
        key, value = entry
        if type(key) is not str or type(value) is not int or key in result:
            raise ValueError("Prime one-shot schema entries differ")
        result[key] = value
    return result


V2_RUNTIME_BINDING = RuntimeBinding(
    authorization_path=AUTHORIZATION_PATH,
    authorization_authenticator_name=_V2_AUTHENTICATOR,
    evidence_root=EVIDENCE_ROOT,
    claim_domain=CLAIM_DOMAIN,
    assessment_domain=ASSESSMENT_DOMAIN,
    assessment_schema_version=2,
    assessment_namespace=ASSESSMENT_NAMESPACE,
    signed_envelope_domain=SIGNED_ENVELOPE_DOMAIN,
    handoff_namespace=HANDOFF_NAMESPACE,
    terminal_domain=TERMINAL_DOMAIN,
    terminal_namespace=TERMINAL_NAMESPACE,
    terminal_purpose=TERMINAL_PURPOSE,
    claim_authority=tuple(RUNTIME_AUTHORITY.items()),
    assessment_authority=tuple(RUNTIME_AUTHORITY.items()),
    create_authority=tuple(READINESS_AUTHORITY.items()),
    terminal_authority=tuple(READINESS_AUTHORITY.items()),
    result_authority=tuple(READINESS_AUTHORITY.items()),
    handoff_authority=tuple(READINESS_AUTHORITY.items()),
    artifact_filenames=tuple(ARTIFACT_FILENAMES.items()),
    assessment_ttl_seconds=ASSESSMENT_TTL_SECONDS,
    schema_versions=tuple((key, 2) for key in sorted(_SCHEMA_KEYS)),
)

def _is_trusted_binding(binding: RuntimeBinding) -> bool:
    if binding is V2_RUNTIME_BINDING:
        return True
    try:
        from redco.analysis import stage_d_v13_prime_test_one_shot_runtime_v3 as runtime_v3
    except ImportError:
        return False
    return binding is runtime_v3.V3_RUNTIME_BINDING


@dataclass(frozen=True, slots=True)
class _RuntimeContext:
    repository: Path
    authorization: Mapping[str, str]
    client: APIClient
    wallet_team_id: str | None
    transport_errors: tuple[type[BaseException], ...]
    prime_executable: Path
    openssh: Mapping[str, Path]
    keygen_executable: Path
    signing_key: Path
    identity: SigningIdentity
    linux_uv: Path
    run: Callable[[Sequence[str], bytes | None, float], CommandResult]
    now: Callable[[], int]
    monotonic: Callable[[], float]
    sleep: Callable[[float], None]
    binding: RuntimeBinding = V2_RUNTIME_BINDING


def sha_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _authenticate_source(path: str, expected: str) -> None:
    spec = importlib.util.find_spec(path.replace("/", ".").removesuffix(".py"))
    if spec is None or spec.origin is None or sha_file(Path(spec.origin)) != expected:
        raise ValueError(f"Prime one-shot installed owner differs: {path}")


def _production_run(
    argv: Sequence[str], input_bytes: bytes | None, timeout: float
) -> CommandResult:
    environment = None
    if Path(argv[0]).name.lower() == "ssh-keygen.exe":
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith(("SSH_", "PRIME_", "GIT_SSH"))
        }
        environment["PATH"] = r"C:\Windows\System32\OpenSSH;C:\Windows\System32;C:\Windows"
    result = subprocess.run(
        argv,
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=environment,
    )
    return CommandResult(tuple(argv), result.returncode, result.stdout, result.stderr)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_production_context(binding: RuntimeBinding) -> _RuntimeContext:
    if not _is_trusted_binding(binding):
        raise ValueError("Prime one-shot runtime binding is not the canonical singleton")
    repository = _repository_root()
    signing_key = Path.home() / ".ssh" / "id_rsa"
    if sys.version_info[:3] != (3, 13, 2):
        raise ValueError("Prime one-shot requires CPython 3.13.2")
    authorization = cast(dict[str, str], binding.authorization_authenticator(repository))
    for relative, expected in (
        (V5_OWNER_PATH, V5_OWNER_SHA256),
        (V5_CONTRACT_PATH, V5_CONTRACT_SHA256),
    ):
        if sha_file(repository / relative) != expected:
            raise ValueError("historical v5 binding differs")
    tool = cast(dict[str, object], v5.authenticate_installed_capture_owners()["prime_uv_tool"])
    prime = Path(cast(str, tool["canonical_path"]))
    if sha_file(prime) != tool["sha256"]:
        raise ValueError("Prime executable differs")
    for owner, digest in (
        (PODS_API_OWNER, PODS_API_OWNER_SHA256),
        (PODS_COMMAND_OWNER, PODS_COMMAND_OWNER_SHA256),
        (PRIME_CLIENT_OWNER, PRIME_CLIENT_OWNER_SHA256),
        (PRIME_CONFIG_OWNER, PRIME_CONFIG_OWNER_SHA256),
    ):
        _authenticate_source(owner, digest)
    openssh = {
        name: Path(cast(str, item["path"])) for name, item in OPENSSH_EXECUTABLES.items()
    }
    for name, path in openssh.items():
        expected_executable = OPENSSH_EXECUTABLES[name]
        info = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_size != expected_executable["bytes"]
            or sha_file(path) != expected_executable["sha256"]
        ):
            raise ValueError("OpenSSH executable differs")
    keygen = v5.authenticate_approved_openssh_executable()
    raw_identity = v5._load_terminal_signing_identity()
    v5._authenticate_operator_key(signing_key, raw_identity)
    identity = SigningIdentity(
        raw_identity.principal,
        raw_identity.key_type,
        raw_identity.public_key_base64,
        raw_identity.fingerprint_sha256,
        raw_identity.allowed_signers_sha256,
    )
    linux_uv = LINUX_UV_PATH
    if (
        not linux_uv.is_file()
        or linux_uv.stat().st_size != LINUX_UV_BYTES
        or sha_file(linux_uv) != LINUX_UV_SHA256
    ):
        raise ValueError("Linux uv asset differs")
    client = cast(APIClient, v5._construct_api_client())
    configured_team = getattr(client.config, "team_id", None)
    if configured_team is not None and (
        not isinstance(configured_team, str) or not configured_team
    ):
        raise ValueError("Prime configured team identity differs")
    return _RuntimeContext(
        repository,
        authorization,
        client,
        configured_team,
        v5._httpx_request_error_types(),
        prime,
        openssh,
        Path(cast(str, keygen["path"])),
        signing_key,
        identity,
        linux_uv,
        _production_run,
        lambda: int(time.time()),
        time.monotonic,
        time.sleep,
        binding,
    )


def _production_context_v2() -> _RuntimeContext:
    return _build_production_context(V2_RUNTIME_BINDING)


def _production_context_v3() -> _RuntimeContext:
    from redco.analysis.stage_d_v13_prime_test_one_shot_runtime_v3 import (
        V3_RUNTIME_BINDING,
    )

    return _build_production_context(V3_RUNTIME_BINDING)


__all__ = [
    "LINUX_UV_PATH",
    "V2_RUNTIME_BINDING",
    "V3_READINESS_AUTHORITY",
    "V3_RUNTIME_AUTHORITY",
    "APIClient",
    "RuntimeBinding",
    "sha_file",
]
