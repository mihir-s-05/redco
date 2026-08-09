"""Endpoint-level, terminal, non-authorizing Prime inventory receipts v5."""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from redco.analysis import stage_d_v13_prime_inventory_v3 as v3
from redco.analysis import stage_d_v13_prime_inventory_v4 as v4
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes
from redco.analysis.stage_d_v13_support_readiness import _status_paths

ROOT = Path(__file__).parents[3].resolve()
PARENT_COMMIT = "b1c2532b35194a3b0d1ee4000ec347d810a87405"
PARENT_TREE = "cf247bc69f8ae8d750d62df4185845ef7a35e475"
SCHEMA_VERSION = 5
CLAIM_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-endpoint-claim-v5"
RAW_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-endpoint-raw-v5"
ASSESSMENT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-assessment-v5"
TRANSCRIPT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-transcript-v5"
TERMINAL_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-terminal-v5"
TERMINAL_AUTH_PAYLOAD_DOMAIN = (
    "redco-stage-d1-support-v13-prime-inventory-terminal-auth-payload-v5"
)
TERMINAL_AUTH_ENVELOPE_DOMAIN = (
    "redco-stage-d1-support-v13-prime-inventory-terminal-auth-envelope-v5"
)
TERMINAL_AUTH_NAMESPACE = (
    "redco-stage-d1-support-v13-prime-inventory-v5-terminal-auth"
)
CONTRACT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-contract-v5"
AUDIT_DOMAIN = "redco-stage-d1-support-v13-prime-inventory-audit-v5"

CLAIM_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-readiness/"
    "prime-inventory-endpoint-claim-v5.json"
)
RAW_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-readiness/prime-inventory-raw-v5.json"
)
ASSESSMENT_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-readiness/prime-inventory-assessment-v5.json"
)
TRANSCRIPT_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-readiness/prime-inventory-transcript-v5.json"
)
TERMINAL_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-readiness/prime-inventory-terminal-v5.json"
)
TERMINAL_AUTH_RELATIVE = (
    "runs/stage-d/stage-d1-support-v13-readiness/"
    "prime-inventory-terminal-auth-v5.json"
)
OWNER_RELATIVE = "src/redco/analysis/stage_d_v13_prime_inventory_v5.py"
BUILDER_RELATIVE = "scripts/build_stage_d_v13_prime_inventory_v5.py"
TEST_RELATIVE = "tests/test_stage_d_v13_prime_inventory_v5.py"
CONTRACT_RELATIVE = "configs/stage-d/stage-d1-support-prime-inventory-contract-v5.json"
AUDIT_RELATIVE = "reports/stage-d1-support-prime-inventory-audit-v5.json"
CHECKPOINT_PATHS = frozenset(
    {OWNER_RELATIVE, BUILDER_RELATIVE, TEST_RELATIVE, CONTRACT_RELATIVE, AUDIT_RELATIVE}
)

BASE_URL = "https://api.primeintellect.ai"
ENDPOINTS = (
    "/api/v1/availability/gpus",
    "/api/v1/availability/multi-node",
)
PAGE_SIZE = 100
MAX_PAGES_PER_ENDPOINT = 100
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_CUMULATIVE_BODY_BYTES = 32 * 1024 * 1024
TTL_SECONDS = 900
FORBIDDEN_ENVIRONMENT = frozenset(
    {"PRIME_CONTEXT", "PRIME_API_BASE_URL", "PRIME_BASE_URL"}
)

API_SOURCE_RELATIVE = "prime_cli/api/availability.py"
API_SOURCE_SHA256 = "fe366aea5b501ae278902e55a4d1d3059e2fbbcde48d0beeffe980d36603e938"
API_SOURCE_BYTES = 7_439
CLIENT_SOURCE_RELATIVE = "prime_cli/core/client.py"
CLIENT_SOURCE_SHA256 = "bdfb0e6de11980c3c30d402b88828ecac4d5fb56c99ea27f5a09acd2a6b609c0"
CLIENT_SOURCE_BYTES = 12_726
CONFIG_SOURCE_RELATIVE = "prime_cli/core/config.py"
CONFIG_SOURCE_SHA256 = "ec4b68730b1aafd9638ef076889ef3433719317c95da77fa4cfbdca1d3eaf90a"
CONFIG_SOURCE_BYTES = 16_568
HTTPX_SOURCE_RELATIVE = "httpx/__init__.py"
HTTPX_SOURCE_SHA256 = "0ac6997bac998f4ac783adf6d8058a587193315afdb718047c3e4fdff46bcfad"
HTTPX_SOURCE_BYTES = 2_171
HTTPX_VERSION = "0.28.1"

LAUNCH_AUTHORIZATION_RELATIVE = (
    "configs/stage-d/v13-draft/"
    "stage-d1-support-v13-launch-authorization-v1.json"
)
LAUNCH_AUTHORIZATION_SHA256 = (
    "30020b15b5929af1bf668de1bd6b3eb15fe068ec86b24d2dc9a05a8b3b72a7be"
)
OPENSSH_OWNER_RELATIVE = "src/redco/analysis/stage_d_v13_launch_lifecycle.py"
OPENSSH_OWNER_SHA256 = (
    "9b0e38b548e01b014426d0adf39ce07bf72547ed16c05c73ce60a59deb491602"
)
OPENSSH_EXECUTABLE_PATH = Path(r"C:\Windows\System32\OpenSSH\ssh-keygen.exe")
OPENSSH_EXECUTABLE_NORMALIZED_PATH = (
    r"c:\windows\system32\openssh\ssh-keygen.exe"
)
OPENSSH_EXECUTABLE_SHA256 = (
    "3e2f8579e998bc77870b4544efa852391d20afb1bdcf6f48fccb34383ab4b730"
)
OPENSSH_EXECUTABLE_BYTES = 862_208
OPENSSH_PRODUCT_VERSION = "OpenSSH_9.5p2 for Windows"
OPENSSH_SERVICING_HARDLINK_PATH = Path(
    r"C:\Windows\WinSxS\amd64_openssh-common-components-onecore_31bf3856ad364e35_"
    r"10.0.26100.7705_none_584785c55fded901\ssh-keygen.exe"
)
OPENSSH_HARDLINK_COUNT = 2
SIGNING_PRINCIPAL = "mihir"
SIGNING_KEY_TYPE = "ssh-rsa"
SIGNING_FINGERPRINT = "SHA256:LNuExn82n/p//myB4cc0pYv7yA1rzlbeI8Qyi6mXM3U"
ALLOWED_SIGNERS_SHA256 = (
    "ff2a200af8bbdf8aea8d724c804dc9ba638534a33f490473ddcc668b449a9dd4"
)
PUBLIC_KEY_LINE_SHA256 = (
    "9dd0f8ccda00b9151d0e4ffb9a80daab87e284d68b3d55819cd057e486c5a8c0"
)
_HEX64 = re.compile(r"[0-9a-f]{64}")
TERMINAL_AUTH_REJECTION_MATRIX = (
    "path_shadowed_ssh_keygen",
    "wrong_openssh_executable_path",
    "wrong_openssh_executable_bytes",
    "wrong_openssh_link_or_alias_topology",
    "coherent_page_body_transcript_raw_rewrite",
    "recomputed_unsigned_hashes",
    "wrong_private_key",
    "wrong_detached_signature",
    "wrong_principal",
    "wrong_namespace",
    "wrong_fingerprint",
    "wrong_claim",
    "wrong_commit",
    "wrong_tree",
    "wrong_capture_owner_projection",
    "wrong_request_count",
    "wrong_diagnostic",
    "wrong_authority",
    "retry_true",
    "attempt_not_consumed",
    "wrong_transcript_hash",
    "wrong_raw_hash",
    "missing_field",
    "extra_field",
    "noncanonical_base64",
    "replayed_signature",
    "expired_payload",
)

V4_TRACKED_BINDINGS = {
    v4.OWNER_RELATIVE: "59b0896e6df14a4f78c79b2fbbf7fe5ac945b8933bd32718b02e2455989f989c",
    v4.BUILDER_RELATIVE: "12ea27de831f04c4b5e9435280d4d2d5f9af2570aad09bd738fe6ce25e7e8516",
    v4.TEST_RELATIVE: "bffaf913773398a2fc832340f13f2def9c09b824be2e8defc845a85a878551f6",
    v4.CONTRACT_RELATIVE: "004134d6214250dcc4582d29081627a69ef333645f2a08a159fb5172441d24ea",
    v4.AUDIT_RELATIVE: "2d400bcf4843f04baf15ce22bc2f637c20d50b240262d3d94fa489fbfdb36159",
}
HISTORICAL_BINDINGS = {**v4.HISTORICAL_BINDINGS, **V4_TRACKED_BINDINGS}
AUTHORIZATION_FALSE = dict(v3.AUTHORIZATION_FALSE)

ALLOWED_ITEM_KEYS = frozenset(
    {
        "cloudId",
        "gpuType",
        "socket",
        "provider",
        "dataCenter",
        "country",
        "gpuCount",
        "gpuMemory",
        "disk",
        "vcpu",
        "memory",
        "internetSpeed",
        "interconnect",
        "interconnectType",
        "provisioningTime",
        "stockStatus",
        "security",
        "prices",
        "images",
        "isSpot",
        "prepaidTime",
    }
)
DISK_KEYS = frozenset(
    {
        "minCount",
        "defaultCount",
        "maxCount",
        "pricePerUnit",
        "step",
        "defaultIncludedInPrice",
        "additionalInfo",
    }
)
PRICE_KEYS = frozenset({"onDemand", "communityPrice", "isVariable", "currency"})
ALLOWED_GPU_LABELS = frozenset({"L40 48GB", "L40S 48GB", "RTX6000Ada 48GB"})
AVAILABLE_STOCK = frozenset({"available", "ready", "in_stock"})
RECORDED_FAILURE_DIAGNOSTICS = frozenset(
    {
        "non_200_response",
        "non_json_content_type",
        "malformed_pagination_json",
        "pagination_body_not_object",
        "pagination_schema_error",
        "pagination_items_error",
        "changed_total_count",
        "empty_page_before_total",
        "duplicate_canonical_item",
        "pagination_overshoot",
    }
)
UNRECORDED_FAILURE_DIAGNOSTICS = frozenset(
    {
        "transport_failure",
        "capture_cancelled",
        "response_body_too_large",
        "cumulative_body_too_large",
        "page_limit_exceeded",
        "pagination_incomplete",
    }
)


class _Response(Protocol):
    status_code: int
    content: bytes
    headers: Mapping[str, str]


class _Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, object],
        follow_redirects: bool,
    ) -> _Response: ...


class _APIClient(Protocol):
    base_url: str
    api_key: str
    client: _Transport


@dataclass(frozen=True, slots=True)
class _TerminalSigningIdentity:
    principal: str
    key_type: str
    public_key_base64: str
    fingerprint_sha256: str
    allowed_signers_sha256: str
    namespace: str = TERMINAL_AUTH_NAMESPACE

    @property
    def public_key_bytes(self) -> bytes:
        return f"{self.key_type} {self.public_key_base64}\n".encode("ascii")

    @property
    def allowed_signers_bytes(self) -> bytes:
        return (
            f"{self.principal} {self.key_type} {self.public_key_base64}\n"
        ).encode("ascii")

    def projection(self) -> dict[str, object]:
        public_key_bytes = self.public_key_bytes
        return {
            "principal": self.principal,
            "key_type": self.key_type,
            "public_key_base64": self.public_key_base64,
            "public_key_bytes_b64": base64.b64encode(public_key_bytes).decode("ascii"),
            "public_key_bytes_sha256": sha256_bytes(public_key_bytes),
            "fingerprint_sha256": self.fingerprint_sha256,
            "allowed_signers_sha256": self.allowed_signers_sha256,
            "namespace": self.namespace,
        }


def _strict_json_object(raw: bytes, keys: set[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Prime v5 {label} is malformed") from error
    if not isinstance(value, dict) or set(value) != keys or canonical_json_bytes(value) != raw:
        raise ValueError(f"Prime v5 {label} schema/canonical bytes differ")
    return cast(dict[str, Any], value)


def _strict_b64(value: object, length: object, digest: object, label: str) -> bytes:
    if type(value) is not str or type(length) is not int or type(digest) is not str:
        raise ValueError(f"Prime v5 {label} binding types differ")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"Prime v5 {label} encoding differs") from error
    if (
        base64.b64encode(raw).decode("ascii") != value
        or len(raw) != length
        or not _HEX64.fullmatch(digest)
        or sha256_bytes(raw) != digest
    ):
        raise ValueError(f"Prime v5 {label} binding differs")
    return raw


def _load_terminal_signing_identity() -> _TerminalSigningIdentity:
    launch_path = ROOT / LAUNCH_AUTHORIZATION_RELATIVE
    owner_path = ROOT / OPENSSH_OWNER_RELATIVE
    if (
        launch_path.is_symlink()
        or not launch_path.is_file()
        or sha256_bytes(launch_path.read_bytes()) != LAUNCH_AUTHORIZATION_SHA256
        or owner_path.is_symlink()
        or not owner_path.is_file()
        or sha256_bytes(owner_path.read_bytes()) != OPENSSH_OWNER_SHA256
    ):
        raise ValueError("Prime v5 terminal signing owners differ")
    launch = _strict_json_object(
        launch_path.read_bytes(),
        {
            "schema_version",
            "domain",
            "state",
            "parent",
            "orchestrator_thread_id",
            "authorization",
            "scope",
            "obligations",
            "frozen_bindings",
            "input_bindings",
            "execution_gate",
            "signing",
        },
        "launch authorization",
    )
    signing = launch["signing"]
    if not isinstance(signing, dict) or set(signing) != {
        "allowed_signers_sha256",
        "fingerprint_sha256",
        "namespace",
        "principal",
        "public_key_base64",
        "public_key_type",
    }:
        raise ValueError("Prime v5 launch signing projection differs")
    identity = _TerminalSigningIdentity(
        principal=cast(str, signing["principal"]),
        key_type=cast(str, signing["public_key_type"]),
        public_key_base64=cast(str, signing["public_key_base64"]),
        fingerprint_sha256=cast(str, signing["fingerprint_sha256"]),
        allowed_signers_sha256=cast(str, signing["allowed_signers_sha256"]),
    )
    if (
        identity.principal != SIGNING_PRINCIPAL
        or identity.key_type != SIGNING_KEY_TYPE
        or identity.fingerprint_sha256 != SIGNING_FINGERPRINT
        or identity.allowed_signers_sha256 != ALLOWED_SIGNERS_SHA256
        or sha256_bytes(identity.public_key_bytes) != PUBLIC_KEY_LINE_SHA256
        or sha256_bytes(identity.allowed_signers_bytes) != ALLOWED_SIGNERS_SHA256
    ):
        raise ValueError("Prime v5 frozen public identity differs")
    try:
        decoded = base64.b64decode(identity.public_key_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Prime v5 public key encoding differs") from error
    if not decoded:
        raise ValueError("Prime v5 public key is empty")
    return identity


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticate_approved_openssh_executable() -> dict[str, object]:
    """Authenticate the sole Windows-host OpenSSH executable without executing it."""

    if sys.platform != "win32":
        raise ValueError("Prime v5 terminal authentication is Windows-operator-host-only")
    path = OPENSSH_EXECUTABLE_PATH
    servicing_path = OPENSSH_SERVICING_HARDLINK_PATH
    if (
        os.path.normcase(os.path.abspath(path)) != OPENSSH_EXECUTABLE_NORMALIZED_PATH
        or path != Path(r"C:\Windows\System32\OpenSSH\ssh-keygen.exe")
    ):
        raise ValueError("Prime v5 approved OpenSSH executable path differs")
    for candidate in (path, servicing_path):
        _reject_linked_ancestors(candidate, "approved OpenSSH executable")
        if (
            candidate.is_symlink()
            or v3._is_link_or_reparse(candidate)
            or not candidate.is_file()
        ):
            raise ValueError("Prime v5 approved OpenSSH executable is not a regular file")
    stat = path.stat()
    servicing_stat = servicing_path.stat()
    if (
        stat.st_nlink != OPENSSH_HARDLINK_COUNT
        or servicing_stat.st_nlink != OPENSSH_HARDLINK_COUNT
        or not path.samefile(servicing_path)
        or (stat.st_dev, stat.st_ino) != (servicing_stat.st_dev, servicing_stat.st_ino)
    ):
        raise ValueError("Prime v5 approved OpenSSH servicing identity differs")
    if stat.st_size != OPENSSH_EXECUTABLE_BYTES or _sha256_file(path) != (
        OPENSSH_EXECUTABLE_SHA256
    ):
        raise ValueError("Prime v5 approved OpenSSH executable bytes differ")
    return {
        "operator_host": "windows",
        "operator_host_only": True,
        "path": str(path),
        "normalized_absolute_path": OPENSSH_EXECUTABLE_NORMALIZED_PATH,
        "sha256": OPENSSH_EXECUTABLE_SHA256,
        "bytes": OPENSSH_EXECUTABLE_BYTES,
        "product_version": OPENSSH_PRODUCT_VERSION,
        "servicing_hardlink_path": str(servicing_path),
        "hardlink_count": OPENSSH_HARDLINK_COUNT,
        "path_lookup_allowed": False,
        "linux_fallback_allowed": False,
    }


def _ssh_keygen(arguments: list[str], *, input_bytes: bytes = b"") -> bytes:
    executable = authenticate_approved_openssh_executable()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("SSH_", "PRIME_", "GIT_SSH"))
    }
    environment["PATH"] = (
        r"C:\Windows\System32\OpenSSH;C:\Windows\System32;C:\Windows"
    )
    try:
        result = subprocess.run(
            [cast(str, executable["path"]), *arguments],
            input=input_bytes,
            capture_output=True,
            check=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Prime v5 required OpenSSH operation failed") from error
    return bytes(result.stdout)


def _public_key_from_private(path: Path) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Prime v5 operator signing key is unavailable")
    fields = _ssh_keygen(["-y", "-f", str(path)]).decode("ascii").strip().split()
    if len(fields) < 2:
        raise ValueError("Prime v5 operator public key differs")
    return fields[0], fields[1]


def _fingerprint(key_type: str, key_base64: str) -> str:
    output = _ssh_keygen(
        ["-lf", "-", "-E", "sha256"],
        input_bytes=f"{key_type} {key_base64}\n".encode("ascii"),
    ).decode("ascii").strip().split()
    if len(output) < 2 or not output[1].startswith("SHA256:"):
        raise ValueError("Prime v5 operator fingerprint is unavailable")
    return output[1]


def _sign_bytes(path: Path, value: bytes) -> bytes:
    with tempfile.TemporaryDirectory(prefix="redco-prime-v5-sign-") as directory:
        payload_path = Path(directory) / "payload"
        payload_path.write_bytes(value)
        _ssh_keygen(
            ["-Y", "sign", "-f", str(path), "-n", TERMINAL_AUTH_NAMESPACE, str(payload_path)]
        )
        signature_path = payload_path.with_name("payload.sig")
        if signature_path.is_symlink() or not signature_path.is_file():
            raise ValueError("Prime v5 detached signature is unavailable")
        return signature_path.read_bytes()


def _verify_signature(
    identity: _TerminalSigningIdentity, value: bytes, signature: bytes
) -> None:
    with tempfile.TemporaryDirectory(prefix="redco-prime-v5-verify-") as directory:
        root = Path(directory)
        signature_path = root / "payload.sig"
        signers_path = root / "allowed_signers"
        signature_path.write_bytes(signature)
        signers_path.write_bytes(identity.allowed_signers_bytes)
        _ssh_keygen(
            [
                "-Y",
                "verify",
                "-f",
                str(signers_path),
                "-I",
                identity.principal,
                "-n",
                TERMINAL_AUTH_NAMESPACE,
                "-s",
                str(signature_path),
            ],
            input_bytes=value,
        )


def _authenticate_operator_key(
    path: Path, identity: _TerminalSigningIdentity
) -> None:
    _reject_linked_ancestors(path, "operator signing key")
    key_type, key_base64 = _public_key_from_private(path)
    public_path = path.with_name(path.name + ".pub")
    _reject_linked_ancestors(public_path, "operator public key")
    if public_path.is_symlink() or not public_path.is_file():
        raise ValueError("Prime v5 operator public key file is unavailable")
    public_fields = public_path.read_text(encoding="ascii").strip().split()
    if len(public_fields) < 2 or public_fields[:2] != [key_type, key_base64]:
        raise ValueError("Prime v5 operator public/private key binding differs")
    if (
        (key_type, key_base64) != (identity.key_type, identity.public_key_base64)
        or _fingerprint(key_type, key_base64) != identity.fingerprint_sha256
    ):
        raise ValueError("Prime v5 operator signing identity differs")
    challenge = b"redco-stage-d1-support-v13-prime-inventory-v5-signing-challenge"
    _verify_signature(identity, challenge, _sign_bytes(path, challenge))


def _site_packages() -> Path:
    return cast(Path, v3._prime_source_path().parents[2])


def _bound_installed_file(
    relative: str, expected_hash: str, expected_bytes: int
) -> dict[str, object]:
    path = _site_packages() / relative
    if v3._is_link_or_reparse(path) or not path.is_file():
        raise ValueError(f"installed capture owner is unavailable: {relative}")
    raw = path.read_bytes()
    if len(raw) != expected_bytes or sha256_bytes(raw) != expected_hash:
        raise ValueError(f"installed capture owner binding differs: {relative}")
    return {"path": relative, "sha256": expected_hash, "bytes": expected_bytes}


def authenticate_installed_capture_owners() -> dict[str, object]:
    """Authenticate source and executable owners without reading config contents."""

    owners = {
        "availability_api": _bound_installed_file(
            API_SOURCE_RELATIVE, API_SOURCE_SHA256, API_SOURCE_BYTES
        ),
        "core_client": _bound_installed_file(
            CLIENT_SOURCE_RELATIVE, CLIENT_SOURCE_SHA256, CLIENT_SOURCE_BYTES
        ),
        "config_owner": _bound_installed_file(
            CONFIG_SOURCE_RELATIVE, CONFIG_SOURCE_SHA256, CONFIG_SOURCE_BYTES
        ),
        "httpx": {
            **_bound_installed_file(
                HTTPX_SOURCE_RELATIVE, HTTPX_SOURCE_SHA256, HTTPX_SOURCE_BYTES
            ),
            "version": HTTPX_VERSION,
        },
        "prime_uv_tool": v3.authenticate_installed_prime_executable(),
    }
    client_text = (_site_packages() / CLIENT_SOURCE_RELATIVE).read_text(encoding="utf-8")
    config_text = (_site_packages() / CONFIG_SOURCE_RELATIVE).read_text(encoding="utf-8")
    required_client = (
        "self.config = Config()",
        "self.base_url = self.config.base_url",
        "self.client = httpx.Client(",
    )
    required_config = (
        'DEFAULT_BASE_URL: str = "https://api.primeintellect.ai"',
        'context = os.getenv("PRIME_CONTEXT")',
        'os.getenv("PRIME_API_BASE_URL") or os.getenv("PRIME_BASE_URL")',
    )
    if any(line not in client_text for line in required_client) or any(
        line not in config_text for line in required_config
    ):
        raise ValueError("installed Prime capture construction law differs")
    return owners


def _config_paths() -> tuple[Path, Path, Path]:
    root = Path.home() / ".prime"
    return root, root / "config.json", root / "environments"


def _reject_linked_ancestors(path: Path, label: str) -> None:
    current = path.absolute()
    while True:
        if current.exists() and v3._is_link_or_reparse(current):
            raise ValueError(f"{label} has a linked or reparse ancestor")
        if current == current.parent:
            return
        current = current.parent


def _authenticate_config_paths() -> None:
    root, config, environments = _config_paths()
    for path, expected_directory in ((root, True), (config, False), (environments, True)):
        _reject_linked_ancestors(path, "Prime config path")
        if v3._is_link_or_reparse(path) or not path.exists():
            raise ValueError("Prime config path is absent or linked")
        if expected_directory != path.is_dir():
            raise ValueError("Prime config path has the wrong file type")
    if config.stat().st_nlink != 1:
        raise ValueError("Prime config file is aliased")


def _load_httpx_module() -> Any:
    site = _site_packages().resolve()
    inserted = str(site)
    sys.path.insert(0, inserted)
    try:
        module = importlib.import_module("httpx")
    finally:
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
    module_path = Path(cast(str, module.__file__)).resolve()
    if not module_path.is_relative_to(site):
        raise ValueError("Prime capture imported an unauthenticated HTTPX dependency")
    if getattr(module, "__version__", None) != HTTPX_VERSION:
        raise ValueError("Prime capture httpx version differs")
    return module


def _httpx_request_error_types() -> tuple[type[BaseException], ...]:
    request_error = getattr(_load_httpx_module(), "RequestError", None)
    if not isinstance(request_error, type) or not issubclass(request_error, BaseException):
        raise ValueError("authenticated HTTPX RequestError is unavailable")
    return (request_error,)


def _construct_api_client() -> _APIClient:
    site = _site_packages().resolve()
    inserted = str(site)
    sys.path.insert(0, inserted)
    try:
        module = importlib.import_module("prime_cli.core.client")
    finally:
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
    module_path = Path(cast(str, module.__file__)).resolve()
    if not module_path.is_relative_to(site):
        raise ValueError("Prime capture imported an unauthenticated dependency")
    _load_httpx_module()
    client = cast(_APIClient, module.APIClient())
    if client.base_url != BASE_URL or not client.api_key:
        raise ValueError("Prime API client authentication/base URL differs")
    return client


def _git(root: Path, *arguments: str) -> str:
    return cast(str, v3._git(root, *arguments))


def _authenticate_committed_capture_checkout() -> dict[str, str]:
    head = _git(ROOT, "rev-parse", "HEAD")
    parents = _git(ROOT, "rev-list", "--parents", "-n", "1", head).split()
    if len(parents) != 2 or parents[1] != PARENT_COMMIT:
        raise ValueError("Prime v5 capture requires a single-parent direct child checkpoint")
    if _git(ROOT, "rev-parse", f"{PARENT_COMMIT}^{{tree}}") != PARENT_TREE:
        raise ValueError("Prime v5 parent tree differs")
    changes = _git(
        ROOT, "diff", "--name-status", "--no-renames", PARENT_COMMIT, head
    ).splitlines()
    if {line for line in changes} != {f"A\t{path}" for path in CHECKPOINT_PATHS}:
        raise ValueError("Prime v5 checkpoint diff differs from its exact five additions")
    if _status_paths(ROOT):
        raise ValueError("Prime v5 capture checkout is dirty")
    return {"commit": head, "tree": _git(ROOT, "rev-parse", "HEAD^{tree}")}


def _fixed_path(relative: str) -> Path:
    root = ROOT.absolute()
    current = root
    while True:
        if v3._is_link_or_reparse(current):
            raise ValueError("Prime v5 repository ancestor is linked or reparse")
        if current == current.parent:
            break
        current = current.parent
    path = root / relative
    if ".." in Path(relative).parts or not path.resolve(strict=False).is_relative_to(
        root.resolve()
    ):
        raise ValueError("Prime v5 fixed path escapes the repository")
    current = path.parent
    while current != root.parent:
        if current.exists() and v3._is_link_or_reparse(current):
            raise ValueError("Prime v5 output ancestor is linked or reparse")
        if current == root:
            break
        current = current.parent
    return path


def _publish_fixed(relative: str, raw: bytes) -> Path:
    path = _fixed_path(relative)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Prime v5 evidence already exists: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if v3._is_link_or_reparse(path.parent):
        raise ValueError("Prime v5 output parent is linked or reparse")
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        v3._fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_fixed(relative: str) -> bytes:
    path = _fixed_path(relative)
    if v3._is_link_or_reparse(path) or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"Prime v5 fixed evidence is unavailable or aliased: {relative}")
    return path.read_bytes()


def _request_contract() -> dict[str, object]:
    return {
        "endpoints": list(ENDPOINTS),
        "gpu_count": "2",
        "page_size": PAGE_SIZE,
        "sequential": True,
        "redirects": False,
        "retries": 0,
    }


def _projection_sha256(value: Mapping[str, object]) -> str:
    return cast(str, sha256_bytes(canonical_json_bytes(dict(value))))


def _claim_value(
    checkout: Mapping[str, str],
    owners: Mapping[str, object],
    identity: _TerminalSigningIdentity,
    openssh_executable: Mapping[str, object],
    captured: int,
    attempt_nonce: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": CLAIM_DOMAIN,
        "state": "capture_in_progress_attempt_consumed",
        "created_at_epoch": captured,
        "attempt_nonce": attempt_nonce,
        "checkout": dict(checkout),
        "capture_owners": dict(owners),
        "openssh_executable": dict(openssh_executable),
        "endpoints": list(ENDPOINTS),
        "output_paths": {
            "transcript": TRANSCRIPT_RELATIVE,
            "raw": RAW_RELATIVE,
            "assessment": ASSESSMENT_RELATIVE,
            "terminal": TERMINAL_RELATIVE,
            "terminal_auth": TERMINAL_AUTH_RELATIVE,
        },
        "terminal_authentication": {
            "path": TERMINAL_AUTH_RELATIVE,
            "payload_domain": TERMINAL_AUTH_PAYLOAD_DOMAIN,
            "payload_schema_version": SCHEMA_VERSION,
            "namespace": TERMINAL_AUTH_NAMESPACE,
            "public_identity_projection_sha256": _projection_sha256(
                identity.projection()
            ),
            "signature_required_before_assessment": True,
        },
        "attempt_consumed": True,
        "retry": False,
        "authorization": AUTHORIZATION_FALSE,
    }


def _page_record(
    endpoint: str, page: int, response: _Response, cumulative: int
) -> tuple[dict[str, object], int]:
    content = bytes(response.content)
    if len(content) > MAX_BODY_BYTES:
        raise ValueError("response_body_too_large")
    cumulative += len(content)
    if cumulative > MAX_CUMULATIVE_BODY_BYTES:
        raise ValueError("cumulative_body_too_large")
    content_type = response.headers.get("content-type", "")
    return (
        {
            "endpoint": endpoint,
            "params": {"gpu_count": "2", "page": page, "page_size": PAGE_SIZE},
            "status": response.status_code,
            "content_type": content_type,
            "decoded_application_body_b64": base64.b64encode(content).decode("ascii"),
            "decoded_application_body_sha256": sha256_bytes(content),
            "decoded_application_body_bytes": len(content),
            "page_ordinal": page,
        },
        cumulative,
    )


def _pagination(value: bytes) -> tuple[int, list[dict[str, Any]]]:
    try:
        parsed = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("malformed_pagination_json") from error
    if not isinstance(parsed, dict):
        raise ValueError("pagination_body_not_object")
    total = parsed.get("totalCount")
    items = parsed.get("items")
    if type(total) is not int or total < 0 or not isinstance(items, list):
        raise ValueError("pagination_schema_error")
    if len(items) > PAGE_SIZE or any(not isinstance(item, dict) for item in items):
        raise ValueError("pagination_items_error")
    return total, cast(list[dict[str, Any]], items)


def _response_summary(response: _Response) -> dict[str, object]:
    content = bytes(response.content)
    return {
        "status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "decoded_application_body_sha256": sha256_bytes(content),
        "decoded_application_body_bytes": len(content),
    }


def _failure_value(
    endpoint: str,
    page: int,
    diagnostic: str,
    *,
    response_recorded: bool,
    response: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "endpoint": endpoint,
        "page_ordinal": page,
        "diagnostic": diagnostic,
        "response_recorded": response_recorded,
        "response": response,
    }


def _capture_pages(
    client: _APIClient,
    transport_errors: tuple[type[BaseException], ...],
) -> tuple[
    list[dict[str, object]],
    str | None,
    dict[str, object] | None,
    int,
]:
    captures: list[dict[str, object]] = []
    cumulative = 0
    diagnostic: str | None = None
    failure: dict[str, object] | None = None
    request_count = 0
    observed_items: set[bytes] = set()
    for endpoint in ENDPOINTS:
        expected_total: int | None = None
        collected = 0
        for page in range(1, MAX_PAGES_PER_ENDPOINT + 1):
            params: dict[str, object] = {
                "gpu_count": "2",
                "page": page,
                "page_size": PAGE_SIZE,
            }
            request_count += 1
            try:
                response = client.client.request(
                    "GET",
                    f"{BASE_URL}{endpoint}",
                    params=params,
                    follow_redirects=False,
                )
                try:
                    record, cumulative = _page_record(endpoint, page, response, cumulative)
                except ValueError as error:
                    diagnostic = str(error)
                    failure = _failure_value(
                        endpoint,
                        page,
                        diagnostic,
                        response_recorded=False,
                        response=_response_summary(response),
                    )
                    break
                captures.append(record)
                if response.status_code != 200:
                    diagnostic = "non_200_response"
                    failure = _failure_value(
                        endpoint,
                        page,
                        diagnostic,
                        response_recorded=True,
                        response=None,
                    )
                    break
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
                if content_type != "application/json":
                    diagnostic = "non_json_content_type"
                    failure = _failure_value(
                        endpoint,
                        page,
                        diagnostic,
                        response_recorded=True,
                        response=None,
                    )
                    break
                try:
                    total, items = _pagination(bytes(response.content))
                except ValueError as error:
                    diagnostic = str(error)
                    failure = _failure_value(
                        endpoint,
                        page,
                        diagnostic,
                        response_recorded=True,
                        response=None,
                    )
                    break
                if expected_total is None:
                    expected_total = total
                elif total != expected_total:
                    diagnostic = "changed_total_count"
                    failure = _failure_value(
                        endpoint, page, diagnostic, response_recorded=True, response=None
                    )
                    break
                if collected < total and not items:
                    diagnostic = "empty_page_before_total"
                    failure = _failure_value(
                        endpoint, page, diagnostic, response_recorded=True, response=None
                    )
                    break
                for item in items:
                    canonical = canonical_json_bytes(item)
                    if canonical in observed_items:
                        diagnostic = "duplicate_canonical_item"
                        failure = _failure_value(
                            endpoint,
                            page,
                            diagnostic,
                            response_recorded=True,
                            response=None,
                        )
                        break
                    observed_items.add(canonical)
                if diagnostic is not None:
                    break
                collected += len(items)
                if collected > total:
                    diagnostic = "pagination_overshoot"
                    failure = _failure_value(
                        endpoint, page, diagnostic, response_recorded=True, response=None
                    )
                    break
                if collected == total:
                    break
            except transport_errors:
                diagnostic = "transport_failure"
                failure = _failure_value(
                    endpoint, page, diagnostic, response_recorded=False, response=None
                )
                break
            except KeyboardInterrupt:
                diagnostic = "capture_cancelled"
                failure = _failure_value(
                    endpoint, page, diagnostic, response_recorded=False, response=None
                )
                break
        else:
            diagnostic = "page_limit_exceeded"
            failure = _failure_value(
                endpoint,
                MAX_PAGES_PER_ENDPOINT + 1,
                diagnostic,
                response_recorded=False,
                response=None,
            )
        if diagnostic is None and expected_total is not None and collected != expected_total:
            diagnostic = "pagination_incomplete"
            failure = _failure_value(
                endpoint,
                min(MAX_PAGES_PER_ENDPOINT + 1, len(captures) + 1),
                diagnostic,
                response_recorded=False,
                response=None,
            )
        if diagnostic is not None:
            break
    return captures, diagnostic, failure, request_count


def _transcript_payload(
    pages: list[dict[str, object]],
    diagnostic: str | None,
    failure: dict[str, object] | None,
    request_count: int,
) -> dict[str, object]:
    return {
        "pages": pages,
        "diagnostic": diagnostic,
        "failure": failure,
        "request_count": request_count,
    }


def _terminal_auth_payload(
    *,
    claim: bytes,
    transcript: bytes,
    transcript_payload: Mapping[str, object],
    raw: bytes,
    raw_state: str,
    captured: int,
    checkout: Mapping[str, str],
    owners: Mapping[str, object],
    identity: _TerminalSigningIdentity,
    openssh_executable: Mapping[str, object],
) -> bytes:
    pages = cast(list[object], transcript_payload["pages"])
    diagnostic = transcript_payload["diagnostic"]
    replay = _replay_transcript(
        pages,
        diagnostic,
        transcript_payload["failure"],
        transcript_payload["request_count"],
    )
    assessment_allowed = raw_state == "captured_endpoint_terminal" and diagnostic is None
    return cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": TERMINAL_AUTH_PAYLOAD_DOMAIN,
                "state": (
                    "authenticated_capture_complete"
                    if assessment_allowed
                    else "authenticated_capture_incomplete"
                ),
                "claim": {"path": CLAIM_RELATIVE, "sha256": sha256_bytes(claim)},
                "transcript": {
                    "path": TRANSCRIPT_RELATIVE,
                    "artifact_sha256": sha256_bytes(transcript),
                    "payload_sha256": replay["payload_sha256"],
                },
                "raw": {"path": RAW_RELATIVE, "artifact_sha256": sha256_bytes(raw)},
                "request_count": transcript_payload["request_count"],
                "captured_page_count": len(pages),
                "raw_capture_state": raw_state,
                "terminal_diagnostic": diagnostic,
                "captured_at_epoch": captured,
                "expires_at_epoch": captured + TTL_SECONDS,
                "capture_checkpoint": {
                    "commit": checkout["commit"],
                    "tree": checkout["tree"],
                    "parent_commit": PARENT_COMMIT,
                },
                "capture_owner_projection_sha256": _projection_sha256(owners),
                "openssh_executable_projection": dict(openssh_executable),
                "request_contract_projection_sha256": _projection_sha256(
                    _request_contract()
                ),
                "signing": {
                    "principal": identity.principal,
                    "fingerprint_sha256": identity.fingerprint_sha256,
                    "namespace": TERMINAL_AUTH_NAMESPACE,
                    "public_identity_projection_sha256": _projection_sha256(
                        identity.projection()
                    ),
                },
                "attempt_consumed": True,
                "retry": False,
                "assessment_allowed": assessment_allowed,
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )


def _terminal_auth_envelope(
    payload: bytes,
    signature: bytes,
    identity: _TerminalSigningIdentity,
    openssh_executable: Mapping[str, object],
) -> bytes:
    return cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": TERMINAL_AUTH_ENVELOPE_DOMAIN,
                "state": "detached_signature_terminal",
                "payload": {
                    "base64": base64.b64encode(payload).decode("ascii"),
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                },
                "signature": {
                    "base64": base64.b64encode(signature).decode("ascii"),
                    "bytes": len(signature),
                    "sha256": sha256_bytes(signature),
                },
                "public_identity": identity.projection(),
                "openssh_executable": dict(openssh_executable),
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )


def _terminal_publication_failure(
    claim: bytes,
    transcript_payload: Mapping[str, object],
    raw: bytes,
    *,
    transcript_published: bool,
) -> bytes:
    pages = cast(list[object], transcript_payload["pages"])
    terminal = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": TERMINAL_DOMAIN,
                "state": "raw_publication_failed_terminal",
                "claim": {"path": CLAIM_RELATIVE, "sha256": sha256_bytes(claim)},
                "request_count": transcript_payload["request_count"],
                "captured_page_count": len(pages),
                "captured_prefix_sha256": sha256_bytes(canonical_json_bytes(pages)),
                "raw_candidate_sha256": sha256_bytes(raw),
                "transcript_published": transcript_published,
                "retry": False,
                "assessment_allowed": False,
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    _publish_fixed(TERMINAL_RELATIVE, terminal)
    return terminal


def capture_prime_inventory_raw_v5(operator_key_path: Path | None = None) -> bytes:
    """Capture endpoint bodies once; this is the sole production network entrypoint."""

    if operator_key_path is None:
        raise ValueError("Prime v5 operator signing key path is required")

    for relative in (
        CLAIM_RELATIVE,
        TRANSCRIPT_RELATIVE,
        RAW_RELATIVE,
        ASSESSMENT_RELATIVE,
        TERMINAL_RELATIVE,
        TERMINAL_AUTH_RELATIVE,
    ):
        path = _fixed_path(relative)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Prime v5 terminal path already exists: {relative}")
    if FORBIDDEN_ENVIRONMENT.intersection(os.environ):
        raise ValueError("Prime v5 forbids context/base URL environment overrides")
    checkout = _authenticate_committed_capture_checkout()
    owners = authenticate_installed_capture_owners()
    _authenticate_config_paths()
    identity = _load_terminal_signing_identity()
    openssh_executable = authenticate_approved_openssh_executable()
    _authenticate_operator_key(operator_key_path, identity)
    client = _construct_api_client()
    if client.base_url != BASE_URL:
        raise ValueError("Prime v5 API base URL differs")
    transport_errors = _httpx_request_error_types()
    captured = int(time.time())
    attempt_nonce = secrets.token_hex(32)
    claim = canonical_json_bytes(
        _claim_value(
            checkout,
            owners,
            identity,
            openssh_executable,
            captured,
            attempt_nonce,
        )
    )
    _publish_fixed(CLAIM_RELATIVE, claim)
    pages, diagnostic, failure, request_count = _capture_pages(client, transport_errors)
    transcript_payload = _transcript_payload(pages, diagnostic, failure, request_count)
    transcript = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": TRANSCRIPT_DOMAIN,
                "state": "captured_transcript_terminal",
                "claim": {"path": CLAIM_RELATIVE, "sha256": sha256_bytes(claim)},
                "transcript_sha256": sha256_bytes(canonical_json_bytes(transcript_payload)),
                "request_count": request_count,
                "diagnostic": diagnostic,
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    raw_state = (
        "captured_endpoint_terminal" if diagnostic is None else "capture_failed_terminal"
    )
    raw = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": RAW_DOMAIN,
                "state": raw_state,
                "captured_at_epoch": captured,
                "expires_at_epoch": captured + TTL_SECONDS,
                "checkout": checkout,
                "claim": {"path": CLAIM_RELATIVE, "sha256": sha256_bytes(claim)},
                "transcript": {
                    "path": TRANSCRIPT_RELATIVE,
                    "sha256": sha256_bytes(transcript),
                },
                "capture_owners": owners,
                "base_url": BASE_URL,
                "request_contract": _request_contract(),
                "request_count": request_count,
                "pages": pages,
                "diagnostic": diagnostic,
                "failure": failure,
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    transcript_published = False
    try:
        payload = _terminal_auth_payload(
            claim=claim,
            transcript=transcript,
            transcript_payload=transcript_payload,
            raw=raw,
            raw_state=raw_state,
            captured=captured,
            checkout=checkout,
            owners=owners,
            identity=identity,
            openssh_executable=openssh_executable,
        )
        signature = _sign_bytes(operator_key_path, payload)
        envelope = _terminal_auth_envelope(
            payload, signature, identity, openssh_executable
        )
        _publish_fixed(TRANSCRIPT_RELATIVE, transcript)
        transcript_published = True
        _publish_fixed(RAW_RELATIVE, raw)
        _publish_fixed(TERMINAL_AUTH_RELATIVE, envelope)
    except (OSError, ValueError):
        _terminal_publication_failure(
            claim,
            transcript_payload,
            raw,
            transcript_published=transcript_published,
        )
        raise
    return raw


def _validate_page_record(value: object) -> tuple[str, int, bytes]:
    if not isinstance(value, dict) or set(value) != {
        "endpoint",
        "params",
        "status",
        "content_type",
        "decoded_application_body_b64",
        "decoded_application_body_sha256",
        "decoded_application_body_bytes",
        "page_ordinal",
    }:
        raise ValueError("Prime v5 page record schema differs")
    record = cast(dict[str, Any], value)
    endpoint = record["endpoint"]
    page = record["page_ordinal"]
    if endpoint not in ENDPOINTS or type(page) is not int or page < 1:
        raise ValueError("Prime v5 page identity differs")
    if record["params"] != {"gpu_count": "2", "page": page, "page_size": PAGE_SIZE}:
        raise ValueError("Prime v5 page params differ")
    try:
        body = base64.b64decode(record["decoded_application_body_b64"], validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Prime v5 page body encoding differs") from error
    if (
        len(body) != record["decoded_application_body_bytes"]
        or sha256_bytes(body) != record["decoded_application_body_sha256"]
    ):
        raise ValueError("Prime v5 page body binding differs")
    return cast(str, endpoint), page, body


def _new_replay_state() -> dict[str, Any]:
    return {
        "endpoint_index": 0,
        "page": 1,
        "total": None,
        "collected": 0,
        "cumulative": 0,
        "items": set(),
    }


def _expected_boundary(state: Mapping[str, Any]) -> tuple[str, int]:
    index = cast(int, state["endpoint_index"])
    if index >= len(ENDPOINTS):
        raise ValueError("Prime v5 transcript contains pages after endpoint completion")
    return ENDPOINTS[index], cast(int, state["page"])


def _advance_success_record(record: object, state: dict[str, Any]) -> None:
    expected_endpoint, expected_page = _expected_boundary(state)
    endpoint, page, body = _validate_page_record(record)
    if endpoint != expected_endpoint or page != expected_page or page > MAX_PAGES_PER_ENDPOINT:
        raise ValueError("Prime v5 transcript endpoint/page order differs")
    typed = cast(dict[str, Any], record)
    status = typed["status"]
    content_type = typed["content_type"]
    if type(status) is not int or status != 200:
        raise ValueError("non_200_response")
    if (
        not isinstance(content_type, str)
        or content_type.split(";", 1)[0].strip() != "application/json"
    ):
        raise ValueError("non_json_content_type")
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("response_body_too_large")
    cumulative = cast(int, state["cumulative"]) + len(body)
    if cumulative > MAX_CUMULATIVE_BODY_BYTES:
        raise ValueError("cumulative_body_too_large")
    state["cumulative"] = cumulative
    total, items = _pagination(body)
    expected_total = state["total"]
    if expected_total is None:
        state["total"] = total
    elif total != expected_total:
        raise ValueError("changed_total_count")
    collected = cast(int, state["collected"])
    if collected < total and not items:
        raise ValueError("empty_page_before_total")
    seen = cast(set[bytes], state["items"])
    for item in items:
        canonical = canonical_json_bytes(item)
        if canonical in seen:
            raise ValueError("duplicate_canonical_item")
        seen.add(canonical)
    collected += len(items)
    if collected > total:
        raise ValueError("pagination_overshoot")
    state["collected"] = collected
    if collected == total:
        state["endpoint_index"] = cast(int, state["endpoint_index"]) + 1
        state["page"] = 1
        state["total"] = None
        state["collected"] = 0
    else:
        state["page"] = expected_page + 1


def _validate_failure_response(value: object) -> dict[str, object]:
    keys = {
        "status",
        "content_type",
        "decoded_application_body_sha256",
        "decoded_application_body_bytes",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("Prime v5 failure response summary differs")
    response = cast(dict[str, object], value)
    if (
        type(response["status"]) is not int
        or not isinstance(response["content_type"], str)
        or not isinstance(response["decoded_application_body_sha256"], str)
        or len(response["decoded_application_body_sha256"]) != 64
        or type(response["decoded_application_body_bytes"]) is not int
        or response["decoded_application_body_bytes"] < 0
    ):
        raise ValueError("Prime v5 failure response summary types differ")
    return response


def _replay_transcript(
    pages: object,
    diagnostic: object,
    failure: object,
    request_count: object,
) -> dict[str, object]:
    """Authenticate one complete endpoint transcript or its exact failing prefix."""

    if not isinstance(pages, list) or type(request_count) is not int or request_count < 0:
        raise ValueError("Prime v5 transcript collection types differ")
    state = _new_replay_state()
    if diagnostic is None:
        if failure is not None or request_count != len(pages):
            raise ValueError("Prime v5 successful transcript terminal fields differ")
        for record in pages:
            _advance_success_record(record, state)
        if state["endpoint_index"] != len(ENDPOINTS):
            raise ValueError("Prime v5 successful transcript is incomplete")
    else:
        failure_keys = {
            "endpoint",
            "page_ordinal",
            "diagnostic",
            "response_recorded",
            "response",
        }
        if (
            not isinstance(diagnostic, str)
            or not isinstance(failure, dict)
            or set(failure) != failure_keys
            or failure["diagnostic"] != diagnostic
            or type(failure["response_recorded"]) is not bool
        ):
            raise ValueError("Prime v5 failure transcript fields differ")
        recorded = failure["response_recorded"]
        allowed_diagnostics = (
            RECORDED_FAILURE_DIAGNOSTICS if recorded else UNRECORDED_FAILURE_DIAGNOSTICS
        )
        if diagnostic not in allowed_diagnostics:
            raise ValueError("Prime v5 failure diagnostic is not frozen")
        prefix = pages[:-1] if recorded else pages
        for record in prefix:
            _advance_success_record(record, state)
        endpoint, page = _expected_boundary(state)
        if failure["endpoint"] != endpoint or failure["page_ordinal"] != page:
            raise ValueError("Prime v5 failure boundary differs")
        if recorded:
            if request_count != len(pages) or not pages or failure["response"] is not None:
                raise ValueError("Prime v5 recorded failure count differs")
            try:
                _advance_success_record(pages[-1], state)
            except ValueError as error:
                if str(error) != diagnostic:
                    raise ValueError("Prime v5 recorded failure diagnostic differs") from error
            else:
                raise ValueError("Prime v5 failing record unexpectedly succeeded")
        else:
            page_limit = diagnostic in {"page_limit_exceeded", "pagination_incomplete"}
            expected_requests = len(pages) if page_limit else len(pages) + 1
            if request_count != expected_requests:
                raise ValueError("Prime v5 unrecorded failure count differs")
            response = failure["response"]
            if diagnostic in {"response_body_too_large", "cumulative_body_too_large"}:
                summary = _validate_failure_response(response)
                size = cast(int, summary["decoded_application_body_bytes"])
                if diagnostic == "response_body_too_large" and size <= MAX_BODY_BYTES:
                    raise ValueError("Prime v5 response size failure is unproven")
                if diagnostic == "cumulative_body_too_large" and (
                    size > MAX_BODY_BYTES
                    or cast(int, state["cumulative"]) + size <= MAX_CUMULATIVE_BODY_BYTES
                ):
                    raise ValueError("Prime v5 cumulative size failure is unproven")
            elif response is not None:
                raise ValueError("Prime v5 unrecorded failure has unexpected response")
            if page_limit and page != MAX_PAGES_PER_ENDPOINT + 1:
                raise ValueError("Prime v5 page-limit failure boundary differs")
    payload = _transcript_payload(
        cast(list[dict[str, object]], pages),
        diagnostic,
        cast(dict[str, object] | None, failure),
        request_count,
    )
    return {
        "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
        "request_count": request_count,
        "captured_page_count": len(pages),
    }


def _authenticate_terminal_envelope(
    *, now_epoch: int
) -> tuple[dict[str, Any], bytes]:
    openssh_executable = authenticate_approved_openssh_executable()
    identity = _load_terminal_signing_identity()
    envelope_raw = _read_fixed(TERMINAL_AUTH_RELATIVE)
    envelope = _strict_json_object(
        envelope_raw,
        {
            "schema_version",
            "domain",
            "state",
            "payload",
            "signature",
            "public_identity",
            "openssh_executable",
            "authorization",
        },
        "terminal authentication envelope",
    )
    if (
        envelope["schema_version"] != SCHEMA_VERSION
        or envelope["domain"] != TERMINAL_AUTH_ENVELOPE_DOMAIN
        or envelope["state"] != "detached_signature_terminal"
        or envelope["public_identity"] != identity.projection()
        or envelope["openssh_executable"] != openssh_executable
        or envelope["authorization"] != AUTHORIZATION_FALSE
    ):
        raise ValueError("Prime v5 terminal authentication envelope differs")
    payload_binding = envelope["payload"]
    signature_binding = envelope["signature"]
    if not isinstance(payload_binding, dict) or set(payload_binding) != {
        "base64",
        "bytes",
        "sha256",
    }:
        raise ValueError("Prime v5 signed payload binding differs")
    if not isinstance(signature_binding, dict) or set(signature_binding) != {
        "base64",
        "bytes",
        "sha256",
    }:
        raise ValueError("Prime v5 detached signature binding differs")
    payload_raw = _strict_b64(
        payload_binding["base64"],
        payload_binding["bytes"],
        payload_binding["sha256"],
        "signed payload",
    )
    signature = _strict_b64(
        signature_binding["base64"],
        signature_binding["bytes"],
        signature_binding["sha256"],
        "detached signature",
    )
    _verify_signature(identity, payload_raw, signature)
    payload = _strict_json_object(
        payload_raw,
        {
            "schema_version",
            "domain",
            "state",
            "claim",
            "transcript",
            "raw",
            "request_count",
            "captured_page_count",
            "raw_capture_state",
            "terminal_diagnostic",
            "captured_at_epoch",
            "expires_at_epoch",
            "capture_checkpoint",
            "capture_owner_projection_sha256",
            "openssh_executable_projection",
            "request_contract_projection_sha256",
            "signing",
            "attempt_consumed",
            "retry",
            "assessment_allowed",
            "authorization",
        },
        "signed terminal payload",
    )
    captured = payload["captured_at_epoch"]
    expires = payload["expires_at_epoch"]
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["domain"] != TERMINAL_AUTH_PAYLOAD_DOMAIN
        or payload["state"]
        not in {"authenticated_capture_complete", "authenticated_capture_incomplete"}
        or type(captured) is not int
        or type(expires) is not int
        or expires != captured + TTL_SECONDS
        or now_epoch < captured
        or now_epoch > expires
        or type(payload["request_count"]) is not int
        or payload["request_count"] < 0
        or type(payload["captured_page_count"]) is not int
        or payload["captured_page_count"] < 0
        or payload["attempt_consumed"] is not True
        or payload["retry"] is not False
        or type(payload["assessment_allowed"]) is not bool
        or payload["authorization"] != AUTHORIZATION_FALSE
    ):
        raise ValueError("Prime v5 signed terminal payload binding differs")
    signing = payload["signing"]
    if signing != {
        "principal": identity.principal,
        "fingerprint_sha256": identity.fingerprint_sha256,
        "namespace": TERMINAL_AUTH_NAMESPACE,
        "public_identity_projection_sha256": _projection_sha256(identity.projection()),
    }:
        raise ValueError("Prime v5 signed identity projection differs")
    checkout = _authenticate_committed_capture_checkout()
    if payload["capture_checkpoint"] != {
        "commit": checkout["commit"],
        "tree": checkout["tree"],
        "parent_commit": PARENT_COMMIT,
    }:
        raise ValueError("Prime v5 signed capture checkpoint differs")
    owners = authenticate_installed_capture_owners()
    if (
        payload["capture_owner_projection_sha256"] != _projection_sha256(owners)
        or payload["openssh_executable_projection"] != openssh_executable
        or payload["request_contract_projection_sha256"]
        != _projection_sha256(_request_contract())
    ):
        raise ValueError("Prime v5 signed owner/request projection differs")
    claim_binding = payload["claim"]
    transcript_binding = payload["transcript"]
    raw_binding = payload["raw"]
    if not isinstance(claim_binding, dict) or set(claim_binding) != {"path", "sha256"}:
        raise ValueError("Prime v5 signed claim binding differs")
    if not isinstance(transcript_binding, dict) or set(transcript_binding) != {
        "path",
        "artifact_sha256",
        "payload_sha256",
    }:
        raise ValueError("Prime v5 signed transcript binding differs")
    if not isinstance(raw_binding, dict) or set(raw_binding) != {
        "path",
        "artifact_sha256",
    }:
        raise ValueError("Prime v5 signed raw binding differs")
    claim = _read_fixed(CLAIM_RELATIVE)
    transcript = _read_fixed(TRANSCRIPT_RELATIVE)
    raw = _read_fixed(RAW_RELATIVE)
    if claim_binding != {"path": CLAIM_RELATIVE, "sha256": sha256_bytes(claim)}:
        raise ValueError("Prime v5 signed claim artifact differs")
    if transcript_binding["path"] != TRANSCRIPT_RELATIVE or transcript_binding[
        "artifact_sha256"
    ] != sha256_bytes(transcript):
        raise ValueError("Prime v5 signed transcript artifact differs")
    if raw_binding != {"path": RAW_RELATIVE, "artifact_sha256": sha256_bytes(raw)}:
        raise ValueError("Prime v5 signed raw artifact differs")
    claim_value = _strict_json_object(
        claim,
        {
            "schema_version",
            "domain",
            "state",
            "created_at_epoch",
            "attempt_nonce",
            "checkout",
            "capture_owners",
            "openssh_executable",
            "endpoints",
            "output_paths",
            "terminal_authentication",
            "attempt_consumed",
            "retry",
            "authorization",
        },
        "claim",
    )
    nonce = claim_value["attempt_nonce"]
    if (
        claim_value["schema_version"] != SCHEMA_VERSION
        or claim_value["domain"] != CLAIM_DOMAIN
        or claim_value["state"] != "capture_in_progress_attempt_consumed"
        or claim_value["created_at_epoch"] != captured
        or type(nonce) is not str
        or not _HEX64.fullmatch(nonce)
        or claim_value["checkout"] != checkout
        or claim_value["capture_owners"] != owners
        or claim_value["openssh_executable"] != openssh_executable
        or claim_value["endpoints"] != list(ENDPOINTS)
        or claim_value["output_paths"]
        != {
            "transcript": TRANSCRIPT_RELATIVE,
            "raw": RAW_RELATIVE,
            "assessment": ASSESSMENT_RELATIVE,
            "terminal": TERMINAL_RELATIVE,
            "terminal_auth": TERMINAL_AUTH_RELATIVE,
        }
        or claim_value["terminal_authentication"]
        != {
            "path": TERMINAL_AUTH_RELATIVE,
            "payload_domain": TERMINAL_AUTH_PAYLOAD_DOMAIN,
            "payload_schema_version": SCHEMA_VERSION,
            "namespace": TERMINAL_AUTH_NAMESPACE,
            "public_identity_projection_sha256": _projection_sha256(
                identity.projection()
            ),
            "signature_required_before_assessment": True,
        }
        or claim_value["attempt_consumed"] is not True
        or claim_value["retry"] is not False
        or claim_value["authorization"] != AUTHORIZATION_FALSE
    ):
        raise ValueError("Prime v5 signed claim content differs")
    return payload, raw


def _validate_raw_bytes(raw: bytes, *, now_epoch: int) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prime v5 raw receipt is malformed") from error
    keys = {
        "schema_version",
        "domain",
        "state",
        "captured_at_epoch",
        "expires_at_epoch",
        "checkout",
        "claim",
        "transcript",
        "capture_owners",
        "base_url",
        "request_contract",
        "request_count",
        "pages",
        "diagnostic",
        "failure",
        "authorization",
    }
    if not isinstance(value, dict) or set(value) != keys or canonical_json_bytes(value) != raw:
        raise ValueError("Prime v5 raw receipt schema/canonical bytes differ")
    receipt = cast(dict[str, Any], value)
    captured = receipt["captured_at_epoch"]
    expires = receipt["expires_at_epoch"]
    if (
        receipt["schema_version"] != SCHEMA_VERSION
        or receipt["domain"] != RAW_DOMAIN
        or receipt["state"] not in {"captured_endpoint_terminal", "capture_failed_terminal"}
        or type(captured) is not int
        or type(expires) is not int
        or expires != captured + TTL_SECONDS
        or now_epoch < captured
        or now_epoch > expires
        or receipt["base_url"] != BASE_URL
        or receipt["authorization"] != AUTHORIZATION_FALSE
        or receipt["capture_owners"] != authenticate_installed_capture_owners()
    ):
        raise ValueError("Prime v5 raw receipt binding differs")
    checkout = _authenticate_committed_capture_checkout()
    if receipt["checkout"] != checkout:
        raise ValueError("Prime v5 raw checkout binding differs")
    if receipt["request_contract"] != _request_contract():
        raise ValueError("Prime v5 request contract differs")
    claim = _read_fixed(CLAIM_RELATIVE)
    if receipt["claim"] != {
        "path": CLAIM_RELATIVE,
        "sha256": sha256_bytes(claim),
    }:
        raise ValueError("Prime v5 claim binding differs")
    replay = _replay_transcript(
        receipt["pages"],
        receipt["diagnostic"],
        receipt["failure"],
        receipt["request_count"],
    )
    transcript = _read_fixed(TRANSCRIPT_RELATIVE)
    try:
        transcript_value = json.loads(transcript)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Prime v5 transcript commitment is malformed") from error
    expected_transcript = {
        "schema_version": SCHEMA_VERSION,
        "domain": TRANSCRIPT_DOMAIN,
        "state": "captured_transcript_terminal",
        "claim": {"path": CLAIM_RELATIVE, "sha256": sha256_bytes(claim)},
        "transcript_sha256": replay["payload_sha256"],
        "request_count": replay["request_count"],
        "diagnostic": receipt["diagnostic"],
        "authorization": AUTHORIZATION_FALSE,
    }
    if (
        not isinstance(transcript_value, dict)
        or transcript_value != expected_transcript
        or canonical_json_bytes(transcript_value) != transcript
        or receipt["transcript"]
        != {"path": TRANSCRIPT_RELATIVE, "sha256": sha256_bytes(transcript)}
    ):
        raise ValueError("Prime v5 transcript commitment differs")
    if (receipt["state"] == "captured_endpoint_terminal") != (receipt["diagnostic"] is None):
        raise ValueError("Prime v5 terminal state/diagnostic differs")
    return receipt


def _positive_number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(cast(int | float, value))
    return number if math.isfinite(number) and number > 0 else None


def _assess_item(item: dict[str, Any], provenance: dict[str, object]) -> dict[str, object]:
    reasons: list[str] = []
    unknown = sorted(set(item).difference(ALLOWED_ITEM_KEYS))
    if unknown:
        reasons.append("unknown_item_keys")
    label = item.get("gpuType")
    count = item.get("gpuCount")
    memory = item.get("gpuMemory")
    spot = item.get("isSpot", "absent")
    stock = item.get("stockStatus")
    disk = item.get("disk")
    prices = item.get("prices")
    cloud_id = item.get("cloudId")
    if label not in ALLOWED_GPU_LABELS:
        reasons.append("gpu_label_not_allowed")
    if type(count) is not int or count != 2:
        reasons.append("gpu_count_not_two")
    if type(memory) is not int or memory != 96:
        reasons.append("aggregate_gpu_memory_not_96")
    if spot is not False:
        reasons.append("non_spot_not_proven")
    if not isinstance(stock, str) or stock.strip().casefold() not in AVAILABLE_STOCK:
        reasons.append("stock_not_available")
    if not isinstance(cloud_id, str) or not cloud_id:
        reasons.append("cloud_id_malformed")
    if not isinstance(disk, dict) or set(disk) != DISK_KEYS:
        reasons.append("disk_schema_unknown")
    elif type(disk.get("defaultCount")) is not int or disk["defaultCount"] <= 0:
        reasons.append("positive_disk_default_not_proven")
    rate: float | None = None
    if not isinstance(prices, dict) or set(prices) != PRICE_KEYS:
        reasons.append("price_schema_unknown")
    else:
        community_raw = prices["communityPrice"]
        rate = (
            _positive_number(prices["onDemand"])
            if community_raw is None
            else _positive_number(community_raw)
        )
        if rate is None:
            reasons.append("positive_hourly_rate_not_proven")
        elif rate > 2:
            reasons.append("hourly_rate_above_cap")
    return {
        "provenance": provenance,
        "cloud_id": cloud_id if isinstance(cloud_id, str) else None,
        "raw_item": item,
        "hourly_rate_usd": rate,
        "eligible": not reasons,
        "reasons": reasons,
    }


def _assessment_value(receipt: dict[str, Any], raw_sha256: str) -> dict[str, object]:
    _replay_transcript(
        receipt["pages"],
        receipt["diagnostic"],
        receipt["failure"],
        receipt["request_count"],
    )
    if receipt["state"] != "captured_endpoint_terminal":
        state = "capture_failed_non_authorizing"
        rows: list[dict[str, object]] = []
        reason = cast(str, receipt["diagnostic"])
    else:
        rows = []
        for record in cast(list[object], receipt["pages"]):
            endpoint, page, body = _validate_page_record(record)
            _total, items = _pagination(body)
            rows.extend(
                _assess_item(
                    item,
                    {"endpoint": endpoint, "page": page, "item_ordinal": ordinal},
                )
                for ordinal, item in enumerate(items)
            )
        cloud_ids = [row["cloud_id"] for row in rows if row["cloud_id"] is not None]
        duplicate_cloud = len(cloud_ids) != len(set(cloud_ids))
        eligible = [row for row in rows if row["eligible"] is True]
        if duplicate_cloud or len(eligible) > 1:
            state, reason = "observed_ambiguous_resources", "duplicate_or_multiple_resources"
        elif len(eligible) == 1:
            state, reason = "observed_non_authorizing_resource", "one_qualifying_resource"
        else:
            state, reason = "observed_no_qualifying_resource", "no_qualifying_resource"
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": ASSESSMENT_DOMAIN,
        "state": state,
        "reason": reason,
        "raw_receipt": {"path": RAW_RELATIVE, "sha256": raw_sha256},
        "rows": rows,
        "resource": None,
        "authorization": AUTHORIZATION_FALSE,
    }


def assess_prime_inventory_v5() -> bytes:
    """Purely assess the fixed endpoint receipt; never invokes Prime or transport."""

    terminal = _fixed_path(TERMINAL_RELATIVE)
    if terminal.exists() or terminal.is_symlink():
        raise ValueError("Prime v5 terminal publication failure forbids assessment")
    now_epoch = int(time.time())
    signed_payload, raw = _authenticate_terminal_envelope(now_epoch=now_epoch)
    receipt = _validate_raw_bytes(raw, now_epoch=now_epoch)
    replay = _replay_transcript(
        receipt["pages"],
        receipt["diagnostic"],
        receipt["failure"],
        receipt["request_count"],
    )
    expected_allowed = (
        receipt["state"] == "captured_endpoint_terminal"
        and receipt["diagnostic"] is None
    )
    if (
        signed_payload["request_count"] != receipt["request_count"]
        or signed_payload["captured_page_count"] != replay["captured_page_count"]
        or signed_payload["raw_capture_state"] != receipt["state"]
        or signed_payload["terminal_diagnostic"] != receipt["diagnostic"]
        or signed_payload["captured_at_epoch"] != receipt["captured_at_epoch"]
        or signed_payload["expires_at_epoch"] != receipt["expires_at_epoch"]
        or signed_payload["transcript"]["payload_sha256"]
        != replay["payload_sha256"]
        or signed_payload["assessment_allowed"] is not expected_allowed
        or signed_payload["state"]
        != (
            "authenticated_capture_complete"
            if expected_allowed
            else "authenticated_capture_incomplete"
        )
    ):
        raise ValueError("Prime v5 signed terminal/raw topology differs")
    if not expected_allowed:
        raise ValueError("Prime v5 incomplete capture is not assessable")
    assessment = cast(bytes, canonical_json_bytes(_assessment_value(receipt, sha256_bytes(raw))))
    _publish_fixed(ASSESSMENT_RELATIVE, assessment)
    return assessment


def _bound_file(root: Path, relative: str, expected: str) -> None:
    path = root / relative
    if path.is_symlink() or not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
        raise ValueError(f"historical Prime inventory binding differs: {relative}")


def _authenticate_precommit(root: Path) -> None:
    if _git(root, "rev-parse", "HEAD") != PARENT_COMMIT:
        raise ValueError("Prime inventory v5 build requires exact parent b1c2532")
    if _git(root, "rev-parse", "HEAD^{tree}") != PARENT_TREE:
        raise ValueError("Prime inventory v5 parent tree differs")
    unexpected = _status_paths(root).difference(CHECKPOINT_PATHS)
    if unexpected:
        raise ValueError("Prime inventory v5 worktree exceeds its exact allowlist")


def build_prime_inventory_v5_artifacts(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    _authenticate_precommit(root)
    for relative, expected in HISTORICAL_BINDINGS.items():
        _bound_file(root, relative, expected)
    owners = authenticate_installed_capture_owners()
    signing_identity = _load_terminal_signing_identity()
    openssh_executable = authenticate_approved_openssh_executable()
    owner_hash = sha256_bytes((root / OWNER_RELATIVE).read_bytes())
    builder_hash = sha256_bytes((root / BUILDER_RELATIVE).read_bytes())
    test_hash = sha256_bytes((root / TEST_RELATIVE).read_bytes())
    contract = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": CONTRACT_DOMAIN,
                "state": "non_authorizing_cpu_endpoint_receipt_checkpoint",
                "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
                "allowlist": sorted(CHECKPOINT_PATHS),
                "historical": {
                    path: {"sha256": digest, "immutable": True}
                    for path, digest in sorted(HISTORICAL_BINDINGS.items())
                },
                "installed_capture_owners": owners,
                "fixed_paths": {
                    "claim": CLAIM_RELATIVE,
                    "transcript": TRANSCRIPT_RELATIVE,
                    "raw": RAW_RELATIVE,
                    "assessment": ASSESSMENT_RELATIVE,
                    "terminal": TERMINAL_RELATIVE,
                    "terminal_auth": TERMINAL_AUTH_RELATIVE,
                    "tracked": False,
                    "atomic_no_overwrite": True,
                },
                "endpoint_contract": {
                    "base_url": BASE_URL,
                    "endpoints": list(ENDPOINTS),
                    "params": {"gpu_count": "2", "page_size": PAGE_SIZE},
                    "max_pages_per_endpoint": MAX_PAGES_PER_ENDPOINT,
                    "max_decoded_body_bytes": MAX_BODY_BYTES,
                    "max_cumulative_decoded_body_bytes": MAX_CUMULATIVE_BODY_BYTES,
                    "sequential": True,
                    "redirects": False,
                    "retries": 0,
                    "cli_and_pydantic_capture_forbidden": True,
                    "httpx_request_errors": "fixed_transport_failure",
                    "transcript_replay_required": True,
                    "raw_publication_failure_terminal": True,
                },
                "terminal_authentication": {
                    "launch_authorization": {
                        "path": LAUNCH_AUTHORIZATION_RELATIVE,
                        "sha256": LAUNCH_AUTHORIZATION_SHA256,
                    },
                    "openssh_owner": {
                        "path": OPENSSH_OWNER_RELATIVE,
                        "sha256": OPENSSH_OWNER_SHA256,
                    },
                    "openssh_executable": openssh_executable,
                    "public_identity": signing_identity.projection(),
                    "payload_domain": TERMINAL_AUTH_PAYLOAD_DOMAIN,
                    "envelope_domain": TERMINAL_AUTH_ENVELOPE_DOMAIN,
                    "namespace": TERMINAL_AUTH_NAMESPACE,
                    "publication_order": [
                        "claim",
                        "requests",
                        "sign",
                        "transcript",
                        "raw",
                        "terminal_auth",
                    ],
                    "signature_required_before_semantics": True,
                    "assessment_allowed_only_for_complete_topology": True,
                    "private_key_serialized": False,
                    "operator_host_only": "windows",
                    "linux_fallback_allowed": False,
                    "rejection_matrix": list(TERMINAL_AUTH_REJECTION_MATRIX),
                },
                "semantic_law": {
                    "allowed_gpu_labels": sorted(ALLOWED_GPU_LABELS),
                    "gpu_count": 2,
                    "aggregate_gpu_memory_gb": 96,
                    "literal_false_is_spot_required": True,
                    "available_stock": sorted(AVAILABLE_STOCK),
                    "price_precedence": "communityPrice_else_onDemand",
                    "community_price_fallback_only_when_literal_null": True,
                    "maximum_hourly_rate_usd": 2,
                    "positive_disk_default_required": True,
                    "cloud_id_deduplication": False,
                    "repeated_cloud_id_disposition": "ambiguous",
                    "unknown_keys_scope": "row_nonqualifying",
                },
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    bindings = {
        OWNER_RELATIVE: owner_hash,
        BUILDER_RELATIVE: builder_hash,
        TEST_RELATIVE: test_hash,
        CONTRACT_RELATIVE: sha256_bytes(contract),
    }
    audit = cast(
        bytes,
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "domain": AUDIT_DOMAIN,
                "state": "non_authorizing_cpu_endpoint_receipt_checkpoint",
                "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
                "allowlist": sorted(CHECKPOINT_PATHS),
                "file_bindings": dict(sorted(bindings.items())),
                "installed_capture_owners": owners,
                "terminal_authentication": {
                    "launch_authorization_sha256": LAUNCH_AUTHORIZATION_SHA256,
                    "openssh_owner_sha256": OPENSSH_OWNER_SHA256,
                    "openssh_executable": openssh_executable,
                    "public_identity_projection_sha256": _projection_sha256(
                        signing_identity.projection()
                    ),
                    "terminal_auth_path": TERMINAL_AUTH_RELATIVE,
                    "payload_domain": TERMINAL_AUTH_PAYLOAD_DOMAIN,
                    "namespace": TERMINAL_AUTH_NAMESPACE,
                    "coherent_unsigned_rewrite_rejected": True,
                    "authentication_precedes_semantic_replay": True,
                    "rejection_matrix": list(TERMINAL_AUTH_REJECTION_MATRIX),
                },
                "tests": {
                    "transport": "source_free_fake_only",
                    "live_capture_executed": False,
                    "shared_checkout_live_artifacts_created": False,
                    "disposable_ssh_keys_only": True,
                },
                "sensitive_values_tracked": False,
                "external_activity": {
                    "prime_calls": 0,
                    "network_calls": 0,
                    "provider_calls": 0,
                    "model_calls": 0,
                    "gpu_calls": 0,
                    "wallet_calls": 0,
                    "source_or_parquet_reads": 0,
                },
                "authorization": AUTHORIZATION_FALSE,
            }
        ),
    )
    return {CONTRACT_RELATIVE: contract, AUDIT_RELATIVE: audit}


def verify_prime_inventory_v5_artifacts(root: Path, output_root: Path) -> dict[str, str]:
    expected = build_prime_inventory_v5_artifacts(root)
    hashes: dict[str, str] = {}
    for relative, raw in expected.items():
        path = output_root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise ValueError(f"Prime inventory v5 artifact differs: {relative}")
        hashes[relative] = sha256_bytes(raw)
    return hashes


__all__ = [
    "ASSESSMENT_RELATIVE",
    "AUDIT_RELATIVE",
    "CLAIM_RELATIVE",
    "CONTRACT_RELATIVE",
    "RAW_RELATIVE",
    "TERMINAL_AUTH_RELATIVE",
    "TERMINAL_RELATIVE",
    "TRANSCRIPT_RELATIVE",
    "assess_prime_inventory_v5",
    "authenticate_installed_capture_owners",
    "build_prime_inventory_v5_artifacts",
    "capture_prime_inventory_raw_v5",
    "verify_prime_inventory_v5_artifacts",
]
