from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import ipaddress
import json
import re
import struct
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    ARTIFACT_FILENAMES,
    ASSESSMENT_TTL_SECONDS,
    EXTERNAL_GITLINK_OBJECT,
    GPU_TELEMETRY_BINDING,
    HANDOFF_DOMAIN,
    HANDOFF_NAMESPACE,
    MAX_COMMAND_OUTPUT_BYTES,
    MAX_RECONCILIATION_MATCHES,
    MAX_STATUS_POLLS,
    MAXIMUM_POD_SECONDS,
    POD_NAME_PREFIX,
    READINESS_AUTHORITY,
    TEST_NODES,
    CommandResult,
    authority_value,
    canonical_json,
    sha256_bytes,
    strict_object,
)

LINUX_UV_SOURCE = "/home/mihir/.local/uv-latest/uv"
LINUX_UV_BYTES = 66_081_208
LINUX_UV_SHA256 = "da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb"
REDCO_LOCK_SHA256 = "60e9fe7396d45d8e8edd13d2de708fa4895452410b43e1ad860f720047634d31"
PRIME_LOCK_SHA256 = "188b23ec2c723b8ca0c8ca80278747e0628f680449061a5404622f39114d42f5"
PRIME_POST_TREE_SHA256 = "6f87f378dd1d35032272e1797423e8fb54039e37a0100ddb1714569872469978"
PRIME_EXCLUDED_GITLINKS = (
    "deps/pydantic-config",
    "deps/renderers",
    "deps/research-environments",
    "deps/verifiers",
)
PRIME_PATCHES = (
    (
        "patches/prime-rl-redco-stage-c9-practical-efficiency.patch",
        "a91087ed7e79d18da420720ad56e73f0bc7527a5868dbaa4ba82624a019d56f0",
    ),
    (
        "patches/prime-rl-stage-d-live-update-gate-v1.patch",
        "b6474d2e7815b3412625defc08b4a5f4800504f7af23339d747441d61ad4818d",
    ),
    (
        "patches/prime-rl-stage-d-objective-gate-v1.patch",
        "f13da58a6092a2427cec0c47e5d73320bd9ddf453f86b00636d5c8394d85c1ba",
    ),
    (
        "patches/prime-rl-strict-tool-env-guard.patch",
        "1c52102bf79741d8a1791733397de26d7319b907531317c22d5ec1e6cd29c001",
    ),
)

ALLOWED_DEVICE_NAMES = frozenset(
    {
        "NVIDIA L40",
        "NVIDIA L40S",
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX 6000 Ada",
    }
)
MAX_KNOWN_HOSTS_BYTES = 64 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SSH_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,31}")
_FAILURE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?::[A-Za-z][A-Za-z0-9_]*)?")
_HOST_KEY_ALGORITHMS = frozenset(
    {"ssh-ed25519", "ssh-rsa", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384"}
)
_ECDSA_CURVES = {
    "ecdsa-sha2-nistp256": (
        "nistp256", 65,
        int("FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF", 16),
        int("5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B", 16),
    ),
    "ecdsa-sha2-nistp384": (
        "nistp384", 97,
        int(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
            "FFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFF", 16
        ),
        int(
            "B3312FA7E23EE7E4988E056BE3F82D19181D9C6EFE814112"
            "0314088F5013875AC656398D8A2ED19D2A85C8EDD3EC2AEF", 16
        ),
    ),
}
_RSA_MAX_EXPONENT, _RSA_MIN_BITS, _RSA_MAX_BITS = 2**32 - 1, 2048, 16384
_MIB = 1024 * 1024

@dataclass(frozen=True, slots=True)
class HandoffSummary:
    pod_identity_sha256: str
    pod_name: str
    pod_status_sha256: str
    ssh_user: str | None
    ssh_host_sha256: str
    ssh_port: int


class _PodLifecycle(Protocol):
    pod_name: str
    trusted_pod_id: str | None
    known_pod_ids: set[str]
    def prime(self, *arguments: str, cleanup: bool = False) -> CommandResult: ...
    def list_pods(
        self, *, cleanup: bool = False
    ) -> tuple[list[dict[str, Any]], CommandResult]: ...
    def sleep_bounded(self, *, cleanup: bool) -> None: ...

def _hash(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"Prime one-shot {label} differs")
    return value

def validate_command_journal_details(
    phase: str, operation: str, value: object
) -> dict[str, object] | None:
    if type(value) is not dict:
        raise ValueError("Prime one-shot command journal details differ")
    details = cast(dict[str, object], value)
    parts = operation.split(" ")
    executable = parts[0].lower() if parts else ""
    valid = (
        len(parts) == 3
        and operation == " ".join(parts)
        and (
            (executable == "prime.exe" and parts[1] == "--plain" and parts[2] in {"pods", "disks"})
            or (executable == "ssh-keyscan.exe" and parts[1] == "-p" and parts[2].isdigit()
                and 1 <= int(parts[2]) <= 65535)
            or (executable in {"scp.exe", "ssh.exe"} and parts[1:] == ["-o", "BatchMode=yes"])
        )
    )
    if not valid:
        raise ValueError("Prime one-shot command operation differs")
    if phase == "dispatch":
        if set(details) != {"argv_sha256", "timeout"}:
            raise ValueError("Prime one-shot command dispatch schema differs")
        timeout = details["timeout"]
        if (
            _hash(details["argv_sha256"], "command argv hash") != details["argv_sha256"]
            or type(timeout) not in {int, float}
            or not 0 < float(cast(int | float, timeout)) <= MAXIMUM_POD_SECONDS
        ):
            raise ValueError("Prime one-shot command dispatch differs")
        return details
    if set(details) == {"error"}:
        if (
            type(details["error"]) is not str
            or _FAILURE_NAME.fullmatch(details["error"]) is None
        ):
            raise ValueError("Prime one-shot command error differs")
        return None
    expected = {
        "operation", "argv_sha256", "returncode", "stdout_sha256", "stdout_bytes",
        "stderr_sha256", "stderr_bytes",
    }
    if set(details) != expected:
        raise ValueError("Prime one-shot command outcome schema differs")
    if (
        details["operation"] != parts[1:]
        or _hash(details["argv_sha256"], "command outcome argv") != details["argv_sha256"]
        or type(details["returncode"]) is not int
        or not -255 <= details["returncode"] <= 255
        or _hash(details["stdout_sha256"], "command stdout") != details["stdout_sha256"]
        or type(details["stdout_bytes"]) is not int
        or not 0 <= details["stdout_bytes"] <= MAX_COMMAND_OUTPUT_BYTES
        or _hash(details["stderr_sha256"], "command stderr") != details["stderr_sha256"]
        or type(details["stderr_bytes"]) is not int
        or not 0 <= details["stderr_bytes"] <= MAX_COMMAND_OUTPUT_BYTES
    ):
        raise ValueError("Prime one-shot command outcome differs")
    return details

def parse_endpoint(value: Mapping[str, Any]) -> tuple[str | None, str, int]:
    raw = value.get("ssh") or value.get("sshConnection")
    if not isinstance(raw, str) or raw != raw.strip() or any(ord(item) < 32 for item in raw):
        raise RuntimeError("Prime one-shot status lacks an explicit SSH endpoint")
    if "," in raw or raw.startswith("-"):
        raise RuntimeError("Prime one-shot SSH endpoint grammar differs")
    match = re.fullmatch(
        r"(?:(?P<user>[A-Za-z_][A-Za-z0-9._-]{0,31})@)?"
        r"(?P<host>\[[0-9A-Fa-f:]+\]|[A-Za-z0-9][A-Za-z0-9.-]{0,252}) -p (?P<port>[0-9]{1,5})",
        raw,
    )
    if not match or not 1 <= int(match.group("port")) <= 65535:
        raise RuntimeError("Prime one-shot SSH endpoint grammar differs")
    host = match.group("host")
    normalized = host[1:-1] if host.startswith("[") else host
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        labels = normalized.lower().split(".")
        if (
            normalized.lower() in {"localhost", "unknown", "n/a"}
            or len(labels) < 2
            or any(not label or label.startswith("-") or label.endswith("-") for label in labels)
        ):
            raise RuntimeError("Prime one-shot SSH host differs") from None
    else:
        if not address.is_global:
            raise RuntimeError("Prime one-shot SSH address is not globally routable")
    return match.group("user"), normalized, int(match.group("port"))

def status_active(owner: _PodLifecycle, pod_id: str) -> dict[str, Any]:
    for _ in range(MAX_STATUS_POLLS):
        value = json.loads(owner.prime("pods", "status", pod_id, "--output", "json").stdout)
        if type(value) is not dict:
            raise RuntimeError("Prime one-shot pod status is not an object")
        result = cast(dict[str, Any], value)
        status = result.get("status")
        if result.get("id", result.get("podId")) != pod_id or not isinstance(status, str):
            raise RuntimeError("Prime one-shot pod status identity differs")
        if status.upper() == "ACTIVE":
            return result
        if status.upper() not in {"INSTALLING", "PENDING"}:
            raise RuntimeError("Prime pod entered terminal status")
        owner.sleep_bounded(cleanup=False)
    raise TimeoutError("Prime one-shot pod readiness timed out")

def reconcile_created_pod(owner: _PodLifecycle) -> str:
    for _ in range(MAX_STATUS_POLLS):
        rows, _result = owner.list_pods()
        matches = [row for row in rows if row.get("name") == owner.pod_name]
        if len(matches) > MAX_RECONCILIATION_MATCHES:
            raise RuntimeError("Prime pod reconciliation exceeds bound")
        owner.known_pod_ids.update(
            cast(str, row["id"]) for row in matches if isinstance(row.get("id"), str)
        )
        if len(matches) == 1 and isinstance(matches[0].get("id"), str):
            identifier = cast(str, matches[0]["id"])
            if owner.trusted_pod_id is not None and identifier != owner.trusted_pod_id:
                raise RuntimeError("Prime create response and inventory identity differ")
            return identifier
        if len(matches) > 1:
            raise RuntimeError("Prime create produced duplicate named pods")
        owner.sleep_bounded(cleanup=False)
    raise TimeoutError("Prime create identity was not reconciled")

def _memory_range(label: str) -> tuple[int, int]:
    value = GPU_TELEMETRY_BINDING[label]
    return (
        cast(int, value["cuda_visible_min_mib_per_device"]) * _MIB,
        cast(int, value["cuda_visible_max_mib_per_device"]) * _MIB,
    )

DEVICE_MEMORY_RANGES = {
    "NVIDIA L40": _memory_range("L40"),
    "NVIDIA L40S": _memory_range("L40S"),
    "NVIDIA RTX 6000 Ada": _memory_range("RTX6000Ada"),
    "NVIDIA RTX 6000 Ada Generation": _memory_range("RTX6000Ada"),
}
EXPECTED_INVENTORY_AGGREGATE_GB = 96
SELECTED_GPU_NAMES = {
    "L40 48GB": frozenset({"NVIDIA L40"}),
    "L40S 48GB": frozenset({"NVIDIA L40S"}),
    "RTX6000Ada 48GB": frozenset({"NVIDIA RTX 6000 Ada", "NVIDIA RTX 6000 Ada Generation"}),
}

def _read_u32(raw: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(raw):
        raise ValueError("OpenSSH signature is truncated")
    return struct.unpack(">I", raw[offset : offset + 4])[0], offset + 4

def _read_string(raw: bytes, offset: int) -> tuple[bytes, int]:
    length, offset = _read_u32(raw, offset)
    end = offset + length
    if end > len(raw):
        raise ValueError("OpenSSH signature string is truncated")
    return raw[offset:end], end

def _string(raw: bytes) -> bytes:
    return struct.pack(">I", len(raw)) + raw

def _armored_signature(raw: bytes) -> bytes:
    try:
        text = raw.decode("ascii")
        lines = text.splitlines()
        if lines[0] != "-----BEGIN SSH SIGNATURE-----" or lines[-1] != (
            "-----END SSH SIGNATURE-----"
        ):
            raise ValueError
        return base64.b64decode("".join(lines[1:-1]), validate=True)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("OpenSSH signature armor differs") from error

def verify_openssh_sshsig(
    payload: bytes, signature: bytes, public_key: bytes, namespace: str
) -> None:
    fields = public_key.strip().split()
    if len(fields) != 2 or fields[0] != b"ssh-rsa":
        raise ValueError("OpenSSH public key differs")
    key_blob = base64.b64decode(fields[1], validate=True)
    key_type, offset = _read_string(key_blob, 0)
    exponent_raw, offset = _read_string(key_blob, offset)
    modulus_raw, offset = _read_string(key_blob, offset)
    if key_type != b"ssh-rsa" or offset != len(key_blob):
        raise ValueError("OpenSSH RSA key encoding differs")
    exponent = int.from_bytes(exponent_raw, "big")
    modulus = int.from_bytes(modulus_raw, "big")
    envelope = _armored_signature(signature)
    if envelope[:6] != b"SSHSIG":
        raise ValueError("OpenSSH SSHSIG magic differs")
    version, offset = _read_u32(envelope, 6)
    embedded_key, offset = _read_string(envelope, offset)
    signed_namespace, offset = _read_string(envelope, offset)
    reserved, offset = _read_string(envelope, offset)
    hash_algorithm, offset = _read_string(envelope, offset)
    signature_blob, offset = _read_string(envelope, offset)
    if (
        version != 1
        or embedded_key != key_blob
        or signed_namespace.decode() != namespace
        or reserved
        or hash_algorithm != b"sha512"
        or offset != len(envelope)
    ):
        raise ValueError("OpenSSH SSHSIG envelope differs")
    algorithm, signature_offset = _read_string(signature_blob, 0)
    rsa_signature, signature_offset = _read_string(signature_blob, signature_offset)
    if algorithm != b"rsa-sha2-512" or signature_offset != len(signature_blob):
        raise ValueError("OpenSSH RSA signature algorithm differs")
    payload_hash = hashlib.sha512(payload).digest()
    signed = (
        b"SSHSIG"
        + _string(namespace.encode())
        + _string(b"")
        + _string(b"sha512")
        + _string(payload_hash)
    )
    digest = hashlib.sha512(signed).digest()
    digest_info = bytes.fromhex("3051300d060960864801650304020305000440") + digest
    size = (modulus.bit_length() + 7) // 8
    recovered = pow(int.from_bytes(rsa_signature, "big"), exponent, modulus).to_bytes(size, "big")
    padding = size - len(digest_info) - 3
    expected = b"\x00\x01" + b"\xff" * padding + b"\x00" + digest_info
    if padding < 8 or recovered != expected:
        raise ValueError("OpenSSH RSA signature verification failed")

def validate_gpu_facts(raw: bytes, selected_facts: object) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("remote GPU facts are not JSON") from error
    keys = {
        "schema_version",
        "device_count",
        "names",
        "memory_bytes",
        "selected_nominal_aggregate_gb",
        "observed_aggregate_bytes",
        "torch",
        "cuda",
    }
    if not isinstance(value, dict) or set(value) != keys or canonical_json(value) != raw:
        raise ValueError("remote GPU facts schema differs")
    names = value["names"]
    memory = value["memory_bytes"]
    if not isinstance(selected_facts, dict) or set(selected_facts) != {
        "gpu_type",
        "gpu_count",
        "gpu_memory_gb",
        "is_spot",
        "hourly_rate_usd",
        "disk_size",
    }:
        raise ValueError("selected GPU facts schema differs")
    selected_type = selected_facts["gpu_type"]
    expected_names = SELECTED_GPU_NAMES.get(selected_type)
    if (
        value["schema_version"] != 2
        or value["device_count"] != 2
        or not isinstance(names, list)
        or len(names) != 2
        or expected_names is None
        or type(selected_facts["gpu_count"]) is not int
        or selected_facts["gpu_count"] != 2
        or type(selected_facts["gpu_memory_gb"]) is not int
        or selected_facts["gpu_memory_gb"] != EXPECTED_INVENTORY_AGGREGATE_GB
        or any(name not in expected_names for name in names)
        or names[0] != names[1]
        or not isinstance(memory, list)
        or len(memory) != 2
        or any(type(item) is not int for item in memory)
        or any(
            not DEVICE_MEMORY_RANGES[cast(str, name)][0]
            <= cast(int, item)
            <= DEVICE_MEMORY_RANGES[cast(str, name)][1]
            for name, item in zip(names, memory, strict=True)
        )
        or value["selected_nominal_aggregate_gb"] != selected_facts["gpu_memory_gb"]
        or type(value["observed_aggregate_bytes"]) is not int
        or value["observed_aggregate_bytes"] != sum(memory)
        or not isinstance(value["torch"], str)
        or not isinstance(value["cuda"], str)
    ):
        raise ValueError("remote GPU hardware differs from the frozen class")
    return cast(dict[str, object], value)

def validate_junit(raw: bytes, *, require_success: bool = True) -> None:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ValueError("remote JUnit is malformed") from error
    suites = [root] if root.tag == "testsuite" else list(root)
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(
        int(suite.attrib.get("failures", "0"))
        + int(suite.attrib.get("errors", "0"))
        for suite in suites
    )
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    cases = [case for suite in suites for case in suite.findall(".//testcase")]
    observed = [(case.attrib.get("classname"), case.attrib.get("name")) for case in cases]
    expected = [
        (
            node.partition("::")[0].removesuffix(".py").replace("/", "."),
            node.partition("::")[2],
        )
        for node in TEST_NODES
    ]
    if (
        tests != len(TEST_NODES)
        or skipped
        or (require_success and failures)
        or len(observed) != len(set(observed))
        or sorted(observed) != sorted(expected)
    ):
        raise ValueError("remote JUnit result differs")

def _module_candidates(root: Path, module: str) -> Iterable[Path]:
    relative = Path(*module.split("."))
    bases = (
        root / "src",
        root / "external/prime-rl/src",
        root / "external/prime-rl/packages/prime-rl-configs/src",
    )
    for base in bases:
        for candidate in (base / relative.with_suffix(".py"), base / relative / "__init__.py"):
            if candidate.is_file():
                yield candidate

def _selected_nodes(tree: ast.Module, selected: str | None) -> list[ast.AST]:
    if selected is None:
        return list(tree.body)
    definitions: dict[str, ast.AST] = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            definitions.update(
                {target.id: node for target in targets if isinstance(target, ast.Name)}
            )
    if selected not in definitions:
        imported = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and any((item.asname or item.name) == selected for item in node.names)
        ]
        if not imported:
            raise ValueError("remote selected test node is absent")
        return [*imported, ast.Name(id=selected, ctx=ast.Load())]
    result: list[ast.AST] = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    pending = [selected]
    included: set[str] = set()
    while pending:
        name = pending.pop()
        if name in included:
            continue
        definition = definitions.get(name)
        if definition is None:
            continue
        included.add(name)
        result.append(definition)
        pending.extend(
            child.id
            for child in ast.walk(definition)
            if isinstance(child, ast.Name) and child.id in definitions
        )
    return result

def transitive_test_bindings(root: Path) -> list[dict[str, object]]:
    """Conservatively scan the selected tests and every local imported owner."""
    forbidden_imports = {"datasets", "pyarrow", "transformers", "huggingface_hub"}
    forbidden_calls = {
        "load_dataset",
        "read_parquet",
        "from_pretrained",
        "snapshot_download",
    }
    pending = [(root / node.partition("::")[0], node.partition("::")[2]) for node in TEST_NODES]
    visited: set[Path] = set()
    bindings: list[dict[str, object]] = []
    while pending:
        unresolved, selected = pending.pop()
        path = unresolved.resolve()
        if path in visited:
            continue
        visited.add(path)
        if len(visited) > 256:
            raise ValueError("remote test transitive owner closure exceeds its bound")
        raw = path.read_bytes()
        tree = ast.parse(raw, filename=str(path))
        relevant = _selected_nodes(tree, selected or None)
        used_names = {
            child.id for node in relevant for child in ast.walk(node) if isinstance(child, ast.Name)
        }
        scheduled: set[tuple[str, str]] = set()
        for node in relevant:
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    for item in child.names:
                        local = item.asname or item.name.split(".", 1)[0]
                        if item.name.split(".")[0] in forbidden_imports:
                            raise ValueError(f"remote test owner {path.name} imports {item.name}")
                        if local in used_names:
                            attributes = {
                                reference.attr
                                for owner in relevant
                                for reference in ast.walk(owner)
                                if isinstance(reference, ast.Attribute)
                                and isinstance(reference.value, ast.Name)
                                and reference.value.id == local
                            }
                            if attributes:
                                scheduled.update((item.name, attr) for attr in attributes)
                            else:
                                scheduled.add((item.name, ""))
                elif isinstance(child, ast.ImportFrom) and child.module:
                    if child.module.split(".")[0] in forbidden_imports:
                        raise ValueError(f"remote test owner {path.name} imports {child.module}")
                    scheduled.update(
                        (child.module, item.name)
                        for item in child.names
                        if (item.asname or item.name) in used_names and item.name != "*"
                    )
                elif isinstance(child, ast.Call):
                    name = (
                        child.func.attr
                        if isinstance(child.func, ast.Attribute)
                        else child.func.id
                        if isinstance(child.func, ast.Name)
                        else ""
                    )
                    if name in forbidden_calls:
                        raise ValueError(f"remote test owner {path.name} calls forbidden {name}")
        for module, imported_name in scheduled:
            pending.extend(
                (candidate, imported_name) for candidate in _module_candidates(root, module)
            )
        bindings.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        )
    selected_paths = {node.partition("::")[0] for node in TEST_NODES}
    if not selected_paths.issubset({item["path"] for item in bindings}):
        raise ValueError("remote test plan lacks a selected source binding")
    return sorted(bindings, key=lambda item: cast(str, item["path"]))

def remote_test_script(authorization_commit: str, selected_facts: object) -> bytes:
    nodes = " ".join(f"'{node}'" for node in TEST_NODES)
    exclusions = repr(PRIME_EXCLUDED_GITLINKS)
    patch_lines = "\n".join(
        (
            f"test \"$(sha256sum '{path}' | cut -d' ' -f1)\" = '{digest}'\n"
            f'git -C external/prime-rl apply "$PWD/{path}"'
        )
        for path, digest in PRIME_PATCHES
    )
    if not isinstance(selected_facts, dict) or selected_facts.get("gpu_memory_gb") != 96:
        raise ValueError("selected GPU facts differ before remote script construction")
    selected_nominal = selected_facts["gpu_memory_gb"]
    return f"""set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
rm -rf /workspace/redco
git clone --no-checkout https://github.com/mihir-s-05/redco.git /workspace/redco
cd /workspace/redco
git checkout --detach "{authorization_commit}"
test "$(git rev-parse HEAD)" = "{authorization_commit}"
git submodule update --init --recursive
mkdir -p .runtime
write_status() {{
  rc=$?
  python3 - "$rc" <<'PY'
import json, sys
raw=json.dumps({{'schema_version':2,'returncode':int(sys.argv[1])}},sort_keys=True,separators=(',',':')).encode()
open('.runtime/remote-status.json','wb').write(raw)
PY
  return "$rc"
}}
trap write_status EXIT
install -m 0755 /tmp/uv .runtime/uv
test "$(sha256sum .runtime/uv | cut -d' ' -f1)" = "{LINUX_UV_SHA256}"
test "$(git rev-parse HEAD)" = "{authorization_commit}"
test "$(git ls-tree HEAD external/prime-rl | awk '{{print $3}}')" = "{EXTERNAL_GITLINK_OBJECT}"
test "$(sha256sum uv.lock | cut -d' ' -f1)" = "{REDCO_LOCK_SHA256}"
test "$(git -C external/prime-rl rev-parse HEAD)" = "{EXTERNAL_GITLINK_OBJECT}"
prime_lock="$(git -C external/prime-rl show HEAD:uv.lock | sha256sum | cut -d' ' -f1)"
test "$prime_lock" = "{PRIME_LOCK_SHA256}"
{patch_lines}
export PYTHONPATH="$PWD/src"
post_tree="$(./.runtime/uv run --no-project --offline python - <<'PY'
import hashlib
from pathlib import Path
from redco.analysis.stage_d_dependency_stack import canonical_tree_manifest_bytes
raw = canonical_tree_manifest_bytes(
    Path('external/prime-rl'), allow_relative_symlinks=True, excluded_roots={exclusions}
)
print(hashlib.sha256(raw).hexdigest())
PY
)"
test "$post_tree" = "{PRIME_POST_TREE_SHA256}"
./.runtime/uv sync --frozen --project external/prime-rl --group dev
export PYTHONPATH="$PWD/src:$PWD/external/prime-rl/src"
export PYTHONPATH="$PYTHONPATH:$PWD/external/prime-rl/packages/prime-rl-configs/src"
./.runtime/uv run --project external/prime-rl --frozen --no-sync python - <<'PY'
import json, torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
names=[torch.cuda.get_device_name(i) for i in range(2)]
memory=[torch.cuda.get_device_properties(i).total_memory for i in range(2)]
value={{
    'schema_version': 2,
    'device_count': 2,
    'names': names,
    'memory_bytes': memory,
    'selected_nominal_aggregate_gb': {selected_nominal},
    'observed_aggregate_bytes': sum(memory),
    'torch': torch.__version__,
    'cuda': torch.version.cuda,
}}
open('.runtime/gpu-facts.json','wb').write(
    json.dumps(value,sort_keys=True,separators=(',',':')).encode()
)
PY
./.runtime/uv run --project external/prime-rl --frozen --no-sync \
  python -m pytest -q --junitxml=.runtime/pytest.xml {nodes}
""".encode()

def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != keys:
        raise ValueError(f"Prime one-shot {label} schema differs")
    return cast(dict[str, object], value)

def _ssh_field(blob: bytes, offset: int) -> tuple[bytes, int]:
    if offset > len(blob) - 4:
        raise ValueError("Prime one-shot SSH key field is truncated")
    length = struct.unpack_from(">I", blob, offset)[0]
    end = offset + 4 + length
    if length > len(blob) or end > len(blob):
        raise ValueError("Prime one-shot SSH key field length differs")
    return blob[offset + 4 : end], end

def _positive_mpint(value: bytes) -> int:
    if not value or value[0] & 0x80 or (number := int.from_bytes(value, "big")) <= 0:
        raise ValueError("Prime one-shot SSH mpint differs")
    if len(value) > 1 and value[0] == 0 and not value[1] & 0x80:
        raise ValueError("Prime one-shot SSH mpint is not canonical")
    return number

def _validate_curve_point(point: bytes, size: int, prime: int, coefficient: int) -> None:
    width = (size - 1) // 2
    if len(point) != size or point[:1] != b"\x04":
        raise ValueError("Prime one-shot ECDSA key structure differs")
    x = int.from_bytes(point[1 : width + 1], "big")
    y = int.from_bytes(point[width + 1 :], "big")
    if x >= prime or y >= prime or (
        pow(y, 2, prime) - pow(x, 3, prime) + 3 * x - coefficient
    ) % prime:
        raise ValueError("Prime one-shot ECDSA point is not on curve")

def _validate_ssh_key_blob(blob: bytes, algorithm: str) -> None:
    key_type, offset = _ssh_field(blob, 0)
    if key_type != algorithm.encode():
        raise ValueError("Prime one-shot SSH key type differs")
    if algorithm == "ssh-rsa":
        exponent, offset = _ssh_field(blob, offset)
        modulus, offset = _ssh_field(blob, offset)
        exponent_value = _positive_mpint(exponent)
        modulus_value = _positive_mpint(modulus)
        if exponent_value < 3 or exponent_value % 2 == 0 or exponent_value > _RSA_MAX_EXPONENT:
            raise ValueError("Prime one-shot RSA exponent is weak")
        if (
            modulus_value % 2 == 0
            or not _RSA_MIN_BITS <= modulus_value.bit_length() <= _RSA_MAX_BITS
        ):
            raise ValueError("Prime one-shot RSA modulus is weak")
    elif algorithm == "ssh-ed25519":
        key, offset = _ssh_field(blob, offset)
        if len(key) != 32:
            raise ValueError("Prime one-shot Ed25519 key length differs")
    else:
        curve_name, offset = _ssh_field(blob, offset)
        point, offset = _ssh_field(blob, offset)
        expected_curve, expected_length, prime, coefficient = _ECDSA_CURVES[algorithm]
        if curve_name != expected_curve.encode():
            raise ValueError("Prime one-shot ECDSA key structure differs")
        _validate_curve_point(point, expected_length, prime, coefficient)
    if offset != len(blob):
        raise ValueError("Prime one-shot SSH key has trailing bytes")

def validate_known_hosts(raw: bytes, host_sha256: str, port: int) -> None:
    if not raw or len(raw) > MAX_KNOWN_HOSTS_BYTES or not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("Prime one-shot known-hosts bytes differ")
    try:
        lines = raw.decode("ascii").removesuffix("\n").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError("Prime one-shot known-hosts encoding differs") from error
    if len(lines) != len(set(lines)):
        raise ValueError("Prime one-shot known-hosts lines overlap")
    matched = False
    for line in lines:
        if any(ord(character) < 32 or ord(character) == 127 for character in line):
            raise ValueError("Prime one-shot known-hosts control differs")
        fields = line.split(" ")
        if len(fields) != 3 or any(not field for field in fields):
            raise ValueError("Prime one-shot known-hosts line differs")
        host_port, algorithm, encoded_key = fields
        match = re.fullmatch(r"\[([^\[\],\s]+)\]:([0-9]{1,5})", host_port)
        if match is None or algorithm not in _HOST_KEY_ALGORITHMS:
            raise ValueError("Prime one-shot known-hosts field differs")
        parsed_port = int(match.group(2))
        try:
            key = base64.b64decode(encoded_key, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Prime one-shot known-hosts key differs") from error
        if not key or len(key) > 16 * 1024 or base64.b64encode(key).decode() != encoded_key:
            raise ValueError("Prime one-shot known-hosts key differs")
        _validate_ssh_key_blob(key, algorithm)
        matched |= parsed_port == port and sha256_bytes(match.group(1).encode()) == host_sha256
    if not matched:
        raise ValueError("Prime one-shot known-hosts endpoint differs")


def validate_handoff_payload(
    raw: bytes,
    *,
    authorization: Mapping[str, str],
    claim_sha256: str,
    transcript_sha256: str,
    assessment_sha256: str,
    assessment_envelope_sha256: str,
    selected_resource_sha256: str,
    selected_facts: object,
    known_hosts: bytes,
) -> HandoffSummary:
    value = strict_object(
        raw,
        {
            "schema_version", "domain", "state", "authorization", "claim", "transcript",
            "assessment", "selected_resource_sha256", "selected_facts", "pod", "ssh",
            "runtime", "evidence_paths", "nonce", "issued_at_epoch", "expires_at_epoch",
            "attempt_consumed", "retry", "authority",
        },
        "Prime one-shot handoff",
    )
    claim = _object(value["claim"], {"path", "sha256"}, "handoff claim")
    transcript = _object(value["transcript"], {"path", "sha256"}, "handoff transcript")
    assessment = _object(
        value["assessment"], {"path", "sha256", "envelope_sha256"}, "handoff assessment"
    )
    pod = _object(value["pod"], {"identity_sha256", "name", "status_sha256"}, "handoff pod")
    ssh = _object(
        value["ssh"], {"user", "host_sha256", "port", "known_hosts_sha256"}, "handoff SSH"
    )
    runtime = _object(
        value["runtime"],
        {"test_script_sha256", "linux_uv_sha256", "test_nodes", "gpu_probe", "gpu_telemetry"},
        "handoff runtime",
    )
    commit = authorization["commit"]
    expected_script = remote_test_script(commit, selected_facts)
    issued, expires = value["issued_at_epoch"], value["expires_at_epoch"]
    user, port = ssh["user"], ssh["port"]
    if (
        value["schema_version"] != 2
        or value["domain"] != HANDOFF_DOMAIN
        or value["state"] != "pod_bound_one_use"
        or value["authorization"] != dict(authorization)
        or claim != {"path": ARTIFACT_FILENAMES["claim"], "sha256": claim_sha256}
        or transcript != {"path": ARTIFACT_FILENAMES["transcript"], "sha256": transcript_sha256}
        or assessment != {
            "path": ARTIFACT_FILENAMES["assessment"], "sha256": assessment_sha256,
            "envelope_sha256": assessment_envelope_sha256,
        }
        or value["selected_resource_sha256"] != selected_resource_sha256
        or value["selected_facts"] != selected_facts
        or pod["name"] != f"{POD_NAME_PREFIX}-{commit[:12]}"
        or type(user) not in {str, type(None)}
        or (type(user) is str and _SSH_USER.fullmatch(user) is None)
        or type(port) is not int
        or not 1 <= port <= 65535
        or ssh["known_hosts_sha256"] != sha256_bytes(known_hosts)
        or runtime != {
            "test_script_sha256": sha256_bytes(expected_script),
            "linux_uv_sha256": LINUX_UV_SHA256,
            "test_nodes": list(TEST_NODES),
            "gpu_probe": "exact allowed class, two devices, aggregate 96GB bounds",
            "gpu_telemetry": GPU_TELEMETRY_BINDING,
        }
        or value["evidence_paths"] != {
            name: filename for name, filename in sorted(ARTIFACT_FILENAMES.items())
        }
        or type(value["nonce"]) is not str
        or _HEX64.fullmatch(value["nonce"]) is None
        or type(issued) is not int
        or issued < 0
        or type(expires) is not int
        or expires != issued + ASSESSMENT_TTL_SECONDS
        or value["attempt_consumed"] is not True
        or value["retry"] is not False
    ):
        raise ValueError("Prime one-shot handoff binding differs")
    authority_value(value["authority"], READINESS_AUTHORITY, "handoff")
    identity_hash = _hash(pod["identity_sha256"], "handoff pod identity")
    status_hash = _hash(pod["status_sha256"], "handoff pod status")
    host_hash = _hash(ssh["host_sha256"], "handoff host")
    normalized_user = user if type(user) is str else None
    validate_known_hosts(known_hosts, host_hash, port)
    return HandoffSummary(
        identity_hash, pod["name"], status_hash, normalized_user, host_hash, port
    )


def build_handoff_payload(
    *,
    authorization: Mapping[str, str], claim_sha256: str, transcript_sha256: str,
    assessment_sha256: str, assessment_envelope_sha256: str,
    selected_resource_sha256: str, selected_facts: object, pod_identity_sha256: str,
    pod_status_sha256: str, ssh_user: str | None, ssh_host: str, ssh_port: int,
    known_hosts: bytes, nonce: str, issued_at_epoch: int,
) -> tuple[bytes, bytes]:
    test_script = remote_test_script(authorization["commit"], selected_facts)
    raw = canonical_json(
        {
            "schema_version": 2, "domain": HANDOFF_DOMAIN, "state": "pod_bound_one_use",
            "authorization": dict(authorization),
            "claim": {"path": ARTIFACT_FILENAMES["claim"], "sha256": claim_sha256},
            "transcript": {
                "path": ARTIFACT_FILENAMES["transcript"], "sha256": transcript_sha256
            },
            "assessment": {
                "path": ARTIFACT_FILENAMES["assessment"], "sha256": assessment_sha256,
                "envelope_sha256": assessment_envelope_sha256,
            },
            "selected_resource_sha256": selected_resource_sha256,
            "selected_facts": selected_facts,
            "pod": {
                "identity_sha256": pod_identity_sha256,
                "name": f"{POD_NAME_PREFIX}-{authorization['commit'][:12]}",
                "status_sha256": pod_status_sha256,
            },
            "ssh": {
                "user": ssh_user, "host_sha256": sha256_bytes(ssh_host.encode()),
                "port": ssh_port, "known_hosts_sha256": sha256_bytes(known_hosts),
            },
            "runtime": {
                "test_script_sha256": sha256_bytes(test_script),
                "linux_uv_sha256": LINUX_UV_SHA256, "test_nodes": list(TEST_NODES),
                "gpu_probe": "exact allowed class, two devices, aggregate 96GB bounds",
                "gpu_telemetry": GPU_TELEMETRY_BINDING,
            },
            "evidence_paths": {
                name: filename for name, filename in sorted(ARTIFACT_FILENAMES.items())
            },
            "nonce": nonce, "issued_at_epoch": issued_at_epoch,
            "expires_at_epoch": issued_at_epoch + ASSESSMENT_TTL_SECONDS,
            "attempt_consumed": True, "retry": False, "authority": READINESS_AUTHORITY,
        }
    )
    validate_handoff_payload(
        raw, authorization=authorization, claim_sha256=claim_sha256,
        transcript_sha256=transcript_sha256, assessment_sha256=assessment_sha256,
        assessment_envelope_sha256=assessment_envelope_sha256,
        selected_resource_sha256=selected_resource_sha256, selected_facts=selected_facts,
        known_hosts=known_hosts,
    )
    return raw, test_script


def handoff_consumer_script(
    *,
    authorization_commit: str,
    payload_sha256: str,
    public_key_sha256: str,
    test_script_sha256: str,
) -> bytes:
    verifier = inspect_verifier_source(authorization_commit)
    return f"""set -euo pipefail
umask 077
root=/tmp/redco-one-shot-handoff-v2
test -f "$root/payload.json" -a -f "$root/payload.sig" -a -f "$root/public.key"
test "$(sha256sum "$root/payload.json" | cut -d' ' -f1)" = "{payload_sha256}"
test "$(sha256sum "$root/public.key" | cut -d' ' -f1)" = "{public_key_sha256}"
test "$(sha256sum "$root/test.sh" | cut -d' ' -f1)" = "{test_script_sha256}"
python3 - "$root/payload.json" "$root/payload.sig" "$root/public.key" <<'PY'
{verifier}
PY
( set -o noclobber; : > "$root/consumed" ) 2>/dev/null
bash "$root/test.sh"
""".encode()


def inspect_verifier_source(authorization_commit: str) -> str:
    return f"""import base64,hashlib,json,struct,sys,time
def u32(raw,o):
    if o+4>len(raw): raise ValueError('truncated')
    return struct.unpack('>I',raw[o:o+4])[0],o+4
def string(raw,o):
    n,o=u32(raw,o); e=o+n
    if e>len(raw): raise ValueError('truncated')
    return raw[o:e],e
def pack(raw): return struct.pack('>I',len(raw))+raw
p=open(sys.argv[1],'rb').read(); a=open(sys.argv[2],'rb').read(); k=open(sys.argv[3],'rb').read()
v=json.loads(p)  # noqa: E501
keys={{'schema_version','domain','state','authorization','claim','transcript','assessment','selected_resource_sha256','selected_facts','pod','ssh','runtime','evidence_paths','nonce','issued_at_epoch','expires_at_epoch','attempt_consumed','retry','authority'}}
canonical=json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
if set(v)!=keys or canonical!=p:
    raise ValueError('payload')
if (v['schema_version']!=2 or
    v['domain']!='redco-stage-d1-prime-test-one-shot-handoff-v2' or
    v['state']!='pod_bound_one_use'):
    raise ValueError('domain')
if v['authorization'].get('commit')!={authorization_commit!r}: raise ValueError('authorization')
if (type(v['issued_at_epoch']) is not int or
    type(v['expires_at_epoch']) is not int or
    not v['issued_at_epoch']<=int(time.time())<=v['expires_at_epoch']):
    raise ValueError('expiry')
if v['attempt_consumed'] is not True or v['retry'] is not False: raise ValueError('attempt')
if v['authority']!={READINESS_AUTHORITY!r}: raise ValueError('authority')
f=k.strip().split()
if len(f)!=2 or f[0]!=b'ssh-rsa': raise ValueError('key')
kb=base64.b64decode(f[1],validate=True); kt,o=string(kb,0); er,o=string(kb,o); nr,o=string(kb,o)
if kt!=b'ssh-rsa' or o!=len(kb): raise ValueError('key')
lines=a.decode('ascii').splitlines()
if (lines[0]!='-----BEGIN SSH SIGNATURE-----' or
    lines[-1]!='-----END SSH SIGNATURE-----'):
    raise ValueError('armor')
s=base64.b64decode(''.join(lines[1:-1]),validate=True)
if s[:6]!=b'SSHSIG': raise ValueError('magic')
v,o=u32(s,6); ek,o=string(s,o); ns,o=string(s,o); r,o=string(s,o)
ha,o=string(s,o); sb,o=string(s,o)
if (v!=1 or ek!=kb or ns!={HANDOFF_NAMESPACE.encode()!r} or
    r or ha!=b'sha512' or o!=len(s)):
    raise ValueError('envelope')
alg,q=string(sb,0); sig,q=string(sb,q)
if alg!=b'rsa-sha2-512' or q!=len(sb): raise ValueError('algorithm')
d=hashlib.sha512(p).digest(); signed=b'SSHSIG'+pack(ns)+pack(b'')+pack(ha)+pack(d)
di=bytes.fromhex('3051300d060960864801650304020305000440')+hashlib.sha512(signed).digest()
n=int.from_bytes(nr,'big'); e=int.from_bytes(er,'big'); z=(n.bit_length()+7)//8
got=pow(int.from_bytes(sig,'big'),e,n).to_bytes(z,'big'); pad=z-len(di)-3
want=b'\\0\\1'+b'\\xff'*pad+b'\\0'+di
if pad<8 or got!=want: raise ValueError('signature')
"""


__all__ = [
    "LINUX_UV_BYTES",
    "LINUX_UV_SHA256",
    "LINUX_UV_SOURCE",
    "HandoffSummary",
    "build_handoff_payload",
    "handoff_consumer_script",
    "remote_test_script",
    "transitive_test_bindings",
    "validate_command_journal_details",
    "validate_gpu_facts",
    "validate_handoff_payload",
    "validate_junit",
    "validate_known_hosts",
    "verify_openssh_sshsig",
]
