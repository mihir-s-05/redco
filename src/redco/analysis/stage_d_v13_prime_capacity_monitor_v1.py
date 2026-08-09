"""Bounded, signed, non-authorizing Stage D Prime capacity monitor."""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5

ROOT = Path(__file__).resolve().parents[3]
PARENT_COMMIT = "5684bb67babff19ebffe697661fccdb660527ac4"
PARENT_TREE = "0dec35f9a4aaa542d18f17a8d139f6ac748ed1af"
V5_CHECKPOINT = "1e70670eaa75942eebacc9d9c720fdc03c03165d"
V5_OWNER_SHA256 = "699df3d9b591df84dcc24cbec842532683240edaaec6a0aaa62934a56c1a1b9b"
V5_CONTRACT_SHA256 = "700bad4bbdfbe2c39bc6ac607fa09f3daa7946cf9ca071c253dbea643d5f78a8"
V5_REPORT_RELATIVE = "reports/stage-d1-support-prime-inventory-v5-terminal-report-v1.json"
V5_REPORT_SHA256 = "bf482b1bc206e0d14c73797504142a1a989eb9cb22d5702cf8d3de5375df3fe7"
SOURCE_TASK_ID = "019f9ab9-ec45-7ac3-82b1-09757b92a7c3"
AUTHORIZATION_TEXT = (
    "Authorized; you may use a monitor/automation every 5 minutes to check for gpu "
    "availability in this thread as you have previously until you get it."
)
AUTHORIZATION_SHA256 = "d47a0db57b6e4a57256b083c28b4469a83809df6b6471e07977850d57309a1ac"

OWNER_RELATIVE = "src/redco/analysis/stage_d_v13_prime_capacity_monitor_v1.py"
RUNNER_RELATIVE = "scripts/run_stage_d_v13_prime_capacity_monitor_v1.py"
CONTRACT_RELATIVE = "configs/stage-d/stage-d1-support-prime-capacity-monitor-v1.json"
AUDIT_RELATIVE = "reports/stage-d1-support-prime-capacity-monitor-audit-v1.json"
TEST_RELATIVE = "tests/test_stage_d_v13_prime_capacity_monitor_v1.py"
CHECKPOINT_PATHS = frozenset(
    {OWNER_RELATIVE, RUNNER_RELATIVE, CONTRACT_RELATIVE, AUDIT_RELATIVE, TEST_RELATIVE}
)

CONTRACT_DOMAIN = "redco-stage-d1-support-v13-prime-capacity-monitor-contract-v1"
AUDIT_DOMAIN = "redco-stage-d1-support-v13-prime-capacity-monitor-audit-v1"
CLAIM_DOMAIN = "redco-stage-d1-support-v13-prime-capacity-monitor-observation-claim-v1"
TRANSCRIPT_DOMAIN = "redco-stage-d1-support-v13-prime-capacity-monitor-observation-transcript-v1"
RAW_DOMAIN = "redco-stage-d1-support-v13-prime-capacity-monitor-observation-raw-v1"
TERMINAL_AUTH_PAYLOAD_DOMAIN = (
    "redco-stage-d1-support-v13-prime-capacity-monitor-terminal-auth-payload-v1"
)
TERMINAL_AUTH_ENVELOPE_DOMAIN = (
    "redco-stage-d1-support-v13-prime-capacity-monitor-terminal-auth-envelope-v1"
)
ASSESSMENT_DOMAIN = "redco-stage-d1-support-v13-prime-capacity-monitor-observation-assessment-v1"
TERMINAL_DOMAIN = "redco-stage-d1-support-v13-prime-capacity-monitor-observation-terminal-v1"
LEDGER_PAYLOAD_DOMAIN = "redco-stage-d1-support-v13-prime-capacity-monitor-ledger-record-payload-v1"
LEDGER_ENVELOPE_DOMAIN = (
    "redco-stage-d1-support-v13-prime-capacity-monitor-ledger-record-envelope-v1"
)
OBSERVATION_NAMESPACE = "redco-stage-d1-support-v13-prime-capacity-monitor-observation-v1"
LEDGER_NAMESPACE = "redco-stage-d1-support-v13-prime-capacity-monitor-ledger-v1"

MONITOR_ID = "stage-d-v13-prime-capacity-monitor-v1"
MONITOR_ROOT_RELATIVE = "runs/stage-d/stage-d1-support-v13-prime-capacity-monitor-v1"
MINIMUM_CADENCE_SECONDS = 300
MAXIMUM_OBSERVATIONS = 288
MAXIMUM_WINDOW_SECONDS = 86_400
MAXIMUM_MONITOR_BYTES = 536_870_912
PRECLAIM_FREE_SPACE_RESERVE = 67_108_864
ATTEMPT_LIMIT = 1
RETRY = False
EXTERNAL_GITLINK_PATH = "external/prime-rl"
EXTERNAL_GITLINK_STATUS = " M external/prime-rl"
EXTERNAL_GITLINK_MODE = "160000"
EXTERNAL_GITLINK_OBJECT = "3b22dd951cad1036d1fe8dd0a0bfc40807a9b360"

MAXIMUM_PAGE_RECORD_OVERHEAD_BYTES = 4_096
MAXIMUM_NON_BODY_ARTIFACT_BYTES = 4_194_304
MAXIMUM_ENDPOINT_PAGES = len(v5.ENDPOINTS) * v5.MAX_PAGES_PER_ENDPOINT
MAXIMUM_ENCODED_BODY_BYTES = (
    4 * ((v5.MAX_CUMULATIVE_BODY_BYTES + 2) // 3) + 4 * MAXIMUM_ENDPOINT_PAGES
)
MAXIMUM_VALID_OBSERVATION_BYTES = (
    MAXIMUM_ENCODED_BODY_BYTES
    + MAXIMUM_ENDPOINT_PAGES * MAXIMUM_PAGE_RECORD_OVERHEAD_BYTES
    + MAXIMUM_NON_BODY_ARTIFACT_BYTES
)

V5_PRIMITIVE_ALLOWLIST = (
    "authenticate_installed_capture_owners",
    "authenticate_approved_openssh_executable",
    "_load_terminal_signing_identity",
    "_authenticate_operator_key",
    "_authenticate_config_paths",
    "_construct_api_client",
    "_httpx_request_error_types",
    "_request_contract",
    "_capture_pages",
    "_replay_transcript",
    "_pagination",
    "_assess_item",
    "_ssh_keygen",
)

AUTHORIZATION_FALSE: dict[str, bool] = {
    "candidate_authorized": False,
    "live_authorized": False,
    "model_calls_authorized": False,
    "prime_authorized": False,
    "provider_calls_authorized": False,
    "provisioning_authorized": False,
    "science_authorized": False,
    "support_launch_authorized": False,
}

_OBSERVATION_FILES = (
    "claim.json",
    "transcript.json",
    "raw.json",
    "terminal-auth.json",
    "assessment.json",
    "terminal.json",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


class _Client(Protocol):
    base_url: str
    api_key: str
    client: Any


class _Clock(Protocol):
    def wall(self) -> int: ...

    def monotonic(self) -> float: ...


@dataclass(frozen=True, slots=True)
class _SystemClock:
    def wall(self) -> int:
        return int(time.time())

    def monotonic(self) -> float:
        return time.monotonic()


@dataclass(frozen=True, slots=True)
class _SigningIdentity:
    principal: str
    key_type: str
    public_key_base64: str
    fingerprint_sha256: str
    allowed_signers_sha256: str

    @property
    def allowed_signers_bytes(self) -> bytes:
        return (f"{self.principal} {self.key_type} {self.public_key_base64}\n").encode("ascii")

    def projection(self, namespace: str) -> dict[str, object]:
        return {
            "principal": self.principal,
            "key_type": self.key_type,
            "fingerprint_sha256": self.fingerprint_sha256,
            "allowed_signers_sha256": self.allowed_signers_sha256,
            "namespace": namespace,
        }


@dataclass(frozen=True, slots=True)
class _Signer:
    identity: _SigningIdentity
    key_path: Path


@dataclass(frozen=True, slots=True)
class ObservationLayout:
    root: Path

    @property
    def observations(self) -> Path:
        return self.root / "observations"

    @property
    def ledger(self) -> Path:
        return self.root / "ledger"

    @property
    def lock(self) -> Path:
        return self.root / "monitor.lock"

    def observation_dir(self, ordinal: int) -> Path:
        return self.observations / f"{ordinal:08d}"

    def artifact(self, ordinal: int, filename: str) -> Path:
        if filename not in _OBSERVATION_FILES:
            raise ValueError("unknown monitor observation artifact")
        return self.observation_dir(ordinal) / filename

    def ledger_record(self, ordinal: int) -> Path:
        return self.ledger / f"{ordinal:08d}.json"

    def relative_artifact(self, ordinal: int, filename: str) -> str:
        return f"observations/{ordinal:08d}/{filename}"

    def relative_ledger(self, ordinal: int) -> str:
        return f"ledger/{ordinal:08d}.json"


@dataclass(frozen=True, slots=True)
class HeartbeatResult:
    state: str
    disposition: str
    continue_monitoring: bool
    observation_ordinal: int | None = None
    next_not_before_epoch: int | None = None
    ledger_sha256: str | None = None

    @property
    def exit_code(self) -> int:
        if self.disposition == "qualifying_capacity_found_stop":
            return 10
        if not self.continue_monitoring:
            return 20
        return 0

    def value(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "domain": CONTRACT_DOMAIN,
            "monitor_id": MONITOR_ID,
            "state": self.state,
            "disposition": self.disposition,
            "continue_monitoring": self.continue_monitoring,
            "observation_ordinal": self.observation_ordinal,
            "next_not_before_epoch": self.next_not_before_epoch,
            "ledger_sha256": self.ledger_sha256,
            "authorization": AUTHORIZATION_FALSE,
        }


@dataclass(frozen=True, slots=True)
class _HeartbeatContext:
    repository: Path
    layout: ObservationLayout
    signer: _Signer
    checkpoint: Mapping[str, str]
    capture_owners: Mapping[str, object]
    openssh_executable: Mapping[str, object]
    client_factory: Callable[[], _Client]
    transport_errors: tuple[type[BaseException], ...]
    clock: _Clock
    free_bytes: Callable[[Path], int]


@dataclass(frozen=True, slots=True)
class _ChainState:
    next_ordinal: int
    previous_ledger_sha256: str
    previous_claim_epoch: int | None
    previous_end_epoch: int | None
    first_claim_epoch: int | None
    stopped: bool
    stop_disposition: str | None


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_object(raw: bytes, keys: set[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"monitor {label} is malformed") from error
    if not isinstance(value, dict) or set(value) != keys or _canonical(value) != raw:
        raise ValueError(f"monitor {label} schema/canonical bytes differ")
    return cast(dict[str, Any], value)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _status_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    external = [
        line
        for line in lines
        if len(line) >= 4 and line[3:].replace("\\", "/") == EXTERNAL_GITLINK_PATH
    ]
    if external != [EXTERNAL_GITLINK_STATUS]:
        raise ValueError("monitor external gitlink porcelain state differs")
    stage = _git(root, "ls-files", "--stage", "--", EXTERNAL_GITLINK_PATH)
    if stage != (f"{EXTERNAL_GITLINK_MODE} {EXTERNAL_GITLINK_OBJECT} 0\t{EXTERNAL_GITLINK_PATH}"):
        raise ValueError("monitor external gitlink index binding differs")
    gitlink = root / EXTERNAL_GITLINK_PATH
    if _is_link_or_reparse(gitlink) or not gitlink.is_dir():
        raise ValueError("monitor external gitlink worktree differs")
    if _git(gitlink, "rev-parse", "HEAD") != EXTERNAL_GITLINK_OBJECT:
        raise ValueError("monitor external gitlink object differs")
    paths: set[str] = set()
    for line in lines:
        if len(line) < 4 or line[2] != " ":
            raise ValueError("monitor porcelain record is malformed")
        path = line[3:].replace("\\", "/")
        if path == EXTERNAL_GITLINK_PATH:
            continue
        if path.startswith(f"{EXTERNAL_GITLINK_PATH}/"):
            raise ValueError("monitor external gitlink descendant state differs")
        paths.add(path)
    return paths


def authenticate_monitor_checkpoint(
    root: Path = ROOT, *, precommit: bool = False
) -> dict[str, str]:
    """Authenticate the reviewed direct-child topology or explicit nonauthorizing build state."""

    head = _git(root, "rev-parse", "HEAD")
    parent_tree = _git(root, "rev-parse", f"{PARENT_COMMIT}^{{tree}}")
    if parent_tree != PARENT_TREE:
        raise ValueError("monitor parent tree differs")
    if precommit:
        if head != PARENT_COMMIT or not _status_paths(root).issubset(CHECKPOINT_PATHS):
            raise ValueError("monitor precommit state differs")
        return {"commit": PARENT_COMMIT, "tree": PARENT_TREE, "state": "precommit"}
    parents = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    if len(parents) != 2 or parents[1] != PARENT_COMMIT:
        raise ValueError("monitor checkpoint is not a direct single-parent child")
    changes = set(
        _git(root, "diff", "--name-status", "--no-renames", PARENT_COMMIT, head).splitlines()
    )
    if changes != {f"A\t{path}" for path in CHECKPOINT_PATHS} or _status_paths(root):
        raise ValueError("monitor checkpoint diff/worktree differs")
    return {"commit": head, "tree": _git(root, "rev-parse", "HEAD^{tree}"), "state": "committed"}


def _authenticate_immutable_inputs(root: Path) -> None:
    bindings = {
        "src/redco/analysis/stage_d_v13_prime_inventory_v5.py": V5_OWNER_SHA256,
        "configs/stage-d/stage-d1-support-prime-inventory-contract-v5.json": V5_CONTRACT_SHA256,
        V5_REPORT_RELATIVE: V5_REPORT_SHA256,
    }
    for relative, expected in bindings.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or _sha256(path.read_bytes()) != expected:
            raise ValueError(f"monitor immutable binding differs: {relative}")
    if _sha256(AUTHORIZATION_TEXT.encode("utf-8")) != AUTHORIZATION_SHA256:
        raise ValueError("monitor authorization bytes differ")


def _is_link_or_reparse(path: Path) -> bool:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(stat, "st_file_attributes", 0)
    reparse = getattr(os.stat_result, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return path.is_symlink() or bool(attributes & reparse)


def _reject_linked_ancestors(path: Path, stop: Path) -> None:
    current = path.absolute()
    stop_absolute = stop.absolute()
    while True:
        if current.exists() and _is_link_or_reparse(current):
            raise ValueError("monitor path has a link/reparse ancestor")
        if current == stop_absolute or current == current.parent:
            return
        current = current.parent


def _safe_path(layout: ObservationLayout, path: Path) -> Path:
    root = layout.root.absolute()
    candidate = path.absolute()
    if not candidate.is_relative_to(root):
        raise ValueError("monitor path escapes fixed root")
    _reject_linked_ancestors(candidate.parent, root.parent)
    if candidate.exists() and _is_link_or_reparse(candidate):
        raise ValueError("monitor path is linked/reparse")
    return candidate


def _prepare_monitor_root(layout: ObservationLayout) -> Path:
    root = layout.root.absolute()
    anchor = Path(root.anchor)
    _reject_linked_ancestors(root.parent, anchor)
    if root.exists():
        if _is_link_or_reparse(root) or not root.is_dir():
            raise ValueError("monitor root is not a fixed regular directory")
    else:
        root.mkdir(parents=True, exist_ok=False)
    if _is_link_or_reparse(root) or not root.is_dir():
        raise ValueError("monitor root changed during creation")
    return root


def _same_file_stats(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _read_safe_file(layout: ObservationLayout, path: Path, label: str) -> bytes:
    candidate = _safe_path(layout, path)
    try:
        before = candidate.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"monitor {label} is absent") from error
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or _is_link_or_reparse(candidate):
        raise ValueError(f"monitor {label} is linked or not regular")
    descriptor = os.open(candidate, os.O_RDONLY)
    try:
        opened = os.fstat(descriptor)
        after = candidate.lstat()
        if (
            opened.st_nlink != 1
            or after.st_nlink != 1
            or not stat.S_ISREG(opened.st_mode)
            or not _same_file_stats(before, opened)
            or not _same_file_stats(opened, after)
        ):
            raise ValueError(f"monitor {label} changed during open")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive(layout: ObservationLayout, path: Path, raw: bytes) -> None:
    target = _safe_path(layout, path)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"monitor artifact already exists: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _safe_path(layout, target.parent)
    temporary = target.parent / f".{target.name}.{secrets.token_hex(16)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _monitor_lock(layout: ObservationLayout) -> Iterator[bool]:
    if sys.platform != "win32":
        raise ValueError("Prime capacity monitor is Windows-only")
    import msvcrt

    _prepare_monitor_root(layout)
    lock = _safe_path(layout, layout.lock)
    if lock.exists() or lock.is_symlink():
        before = lock.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or _is_link_or_reparse(lock):
            raise ValueError("monitor lock is linked or not regular")
        descriptor = os.open(lock, os.O_RDWR)
    else:
        before = None
        descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        opened = os.fstat(descriptor)
        after = lock.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or after.st_nlink != 1
            or not _same_file_stats(opened, after)
            or (before is not None and not _same_file_stats(before, opened))
        ):
            raise ValueError("monitor lock changed during acquisition")
        handle = os.fdopen(descriptor, "r+b")
        descriptor = -1
    except BaseException:
        os.close(descriptor)
        raise
    with handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def _identity_from_v5(value: Any) -> _SigningIdentity:
    return _SigningIdentity(
        principal=cast(str, value.principal),
        key_type=cast(str, value.key_type),
        public_key_base64=cast(str, value.public_key_base64),
        fingerprint_sha256=cast(str, value.fingerprint_sha256),
        allowed_signers_sha256=cast(str, value.allowed_signers_sha256),
    )


def _sign_bytes(signer: _Signer, value: bytes, namespace: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="redco-prime-monitor-sign-") as directory:
        payload = Path(directory) / "payload"
        payload.write_bytes(value)
        v5._ssh_keygen(["-Y", "sign", "-f", str(signer.key_path), "-n", namespace, str(payload)])
        signature = payload.with_name("payload.sig")
        if signature.is_symlink() or not signature.is_file():
            raise ValueError("monitor detached signature is unavailable")
        return signature.read_bytes()


def _verify_signature(
    identity: _SigningIdentity, value: bytes, signature: bytes, namespace: str
) -> None:
    with tempfile.TemporaryDirectory(prefix="redco-prime-monitor-verify-") as directory:
        root = Path(directory)
        signature_path = root / "payload.sig"
        signers_path = root / "allowed_signers"
        signature_path.write_bytes(signature)
        signers_path.write_bytes(identity.allowed_signers_bytes)
        v5._ssh_keygen(
            [
                "-Y",
                "verify",
                "-f",
                str(signers_path),
                "-I",
                identity.principal,
                "-n",
                namespace,
                "-s",
                str(signature_path),
            ],
            input_bytes=value,
        )


def _envelope(
    *,
    domain: str,
    state: str,
    payload: bytes,
    signature: bytes,
    identity: _SigningIdentity,
    namespace: str,
) -> bytes:
    return _canonical(
        {
            "schema_version": 1,
            "domain": domain,
            "state": state,
            "payload": {
                "base64": base64.b64encode(payload).decode("ascii"),
                "bytes": len(payload),
                "sha256": _sha256(payload),
            },
            "signature": {
                "base64": base64.b64encode(signature).decode("ascii"),
                "bytes": len(signature),
                "sha256": _sha256(signature),
            },
            "public_identity": identity.projection(namespace),
            "authorization": AUTHORIZATION_FALSE,
        }
    )


def _decode_binding(value: object, label: str) -> bytes:
    if not isinstance(value, dict) or set(value) != {"base64", "bytes", "sha256"}:
        raise ValueError(f"monitor {label} binding differs")
    binding = cast(dict[str, object], value)
    if type(binding["base64"]) is not str or type(binding["bytes"]) is not int:
        raise ValueError(f"monitor {label} binding types differ")
    try:
        raw = base64.b64decode(binding["base64"], validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError(f"monitor {label} encoding differs") from error
    if (
        base64.b64encode(raw).decode("ascii") != binding["base64"]
        or len(raw) != binding["bytes"]
        or binding["sha256"] != _sha256(raw)
    ):
        raise ValueError(f"monitor {label} digest differs")
    return raw


def _verify_envelope(
    raw: bytes,
    *,
    domain: str,
    state: str,
    payload_domain: str,
    namespace: str,
    identity: _SigningIdentity,
) -> tuple[dict[str, Any], bytes]:
    value = _strict_object(
        raw,
        {
            "schema_version",
            "domain",
            "state",
            "payload",
            "signature",
            "public_identity",
            "authorization",
        },
        "signature envelope",
    )
    if (
        value["schema_version"] != 1
        or value["domain"] != domain
        or value["state"] != state
        or value["public_identity"] != identity.projection(namespace)
        or value["authorization"] != AUTHORIZATION_FALSE
    ):
        raise ValueError("monitor signature envelope projection differs")
    payload = _decode_binding(value["payload"], "payload")
    signature = _decode_binding(value["signature"], "signature")
    _verify_signature(identity, payload, signature, namespace)
    parsed = json.loads(payload)
    if (
        not isinstance(parsed, dict)
        or _canonical(parsed) != payload
        or parsed.get("domain") != payload_domain
    ):
        raise ValueError("monitor signed payload differs")
    return cast(dict[str, Any], parsed), payload


def _artifact_binding(layout: ObservationLayout, path: Path) -> dict[str, object]:
    raw = _read_safe_file(layout, path, path.name)
    return {
        "path": str(path.relative_to(layout.root)).replace("\\", "/"),
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _contract_hash(repository: Path) -> str:
    path = repository / CONTRACT_RELATIVE
    if path.is_symlink() or not path.is_file():
        raise ValueError("monitor contract is unavailable")
    return _sha256(path.read_bytes())


def _audit_hash(repository: Path) -> str:
    path = repository / AUDIT_RELATIVE
    if path.is_symlink() or not path.is_file():
        raise ValueError("monitor audit is unavailable")
    return _sha256(path.read_bytes())


def _genesis_projection(repository: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "domain": LEDGER_PAYLOAD_DOMAIN,
        "monitor_id": MONITOR_ID,
        "contract_sha256": _contract_hash(repository),
        "audit_sha256": _audit_hash(repository),
        "checkpoint_parent": PARENT_COMMIT,
        "checkpoint_parent_tree": PARENT_TREE,
        "v5_checkpoint": V5_CHECKPOINT,
        "v5_owner_sha256": V5_OWNER_SHA256,
        "v5_contract_sha256": V5_CONTRACT_SHA256,
        "v5_terminal_report_sha256": V5_REPORT_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "source_task_id": SOURCE_TASK_ID,
        "minimum_cadence_seconds": MINIMUM_CADENCE_SECONDS,
        "maximum_observations": MAXIMUM_OBSERVATIONS,
        "maximum_window_seconds": MAXIMUM_WINDOW_SECONDS,
        "maximum_monitor_bytes": MAXIMUM_MONITOR_BYTES,
        "preclaim_free_space_reserve": PRECLAIM_FREE_SPACE_RESERVE,
        "attempt_limit": ATTEMPT_LIMIT,
        "retry": RETRY,
        "authorization": AUTHORIZATION_FALSE,
    }


def _genesis_hash(repository: Path) -> str:
    return _sha256(_canonical(_genesis_projection(repository)))


def _authenticated_page_body(record: object) -> tuple[str, int, bytes]:
    """Project a page only after ``_replay_transcript`` authenticated it."""

    if not isinstance(record, dict):
        raise ValueError("monitor authenticated page projection differs")
    typed = cast(dict[str, object], record)
    endpoint = typed["endpoint"]
    page = typed["page_ordinal"]
    encoded = typed["decoded_application_body_b64"]
    if not isinstance(endpoint, str) or type(page) is not int or not isinstance(encoded, str):
        raise ValueError("monitor authenticated page projection types differ")
    try:
        return endpoint, page, base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("monitor authenticated page projection encoding differs") from error


def _assessment_value(pages: list[dict[str, object]]) -> dict[str, object]:
    v5._replay_transcript(pages, None, None, len(pages))
    rows: list[dict[str, object]] = []
    for record in pages:
        endpoint, page, body = _authenticated_page_body(record)
        _total, items = v5._pagination(body)
        rows.extend(
            v5._assess_item(item, {"endpoint": endpoint, "page": page, "item_ordinal": ordinal})
            for ordinal, item in enumerate(items)
        )
    identities = [row["cloud_id"] for row in rows if row["cloud_id"] is not None]
    duplicated = len(identities) != len(set(identities))
    eligible = [row for row in rows if row["eligible"] is True]
    if duplicated or len(eligible) > 1:
        state, reason = "observed_ambiguous_resources", "duplicate_or_multiple_resources"
    elif len(eligible) == 1:
        state, reason = "observed_non_authorizing_resource", "one_qualifying_resource"
    else:
        state, reason = "observed_no_qualifying_resource", "no_qualifying_resource"
    counts = {value: identities.count(value) for value in set(identities)}
    return {
        "schema_version": 1,
        "domain": ASSESSMENT_DOMAIN,
        "state": state,
        "reason": reason,
        "resource": None,
        "semantic_projection_sha256": _sha256(_canonical(rows)),
        "counts": {
            "row_occurrences": len(rows),
            "distinct_identity_values": len(set(identities)),
            "duplicated_identity_values": sum(count > 1 for count in counts.values()),
            "extra_duplicate_occurrences": sum(max(0, count - 1) for count in counts.values()),
            "qualifying_row_occurrences": len(eligible),
        },
        "authorization": AUTHORIZATION_FALSE,
    }


def _observation_id(ordinal: int) -> str:
    return f"{MONITOR_ID}:observation:{ordinal:08d}"


def _read_canonical(
    layout: ObservationLayout, path: Path, keys: set[str], label: str
) -> tuple[dict[str, Any], bytes]:
    raw = _read_safe_file(layout, path, label)
    return _strict_object(raw, keys, label), raw


def _require_binding(
    value: object,
    *,
    path: str,
    raw: bytes,
    label: str,
    extra: Mapping[str, object] | None = None,
) -> None:
    expected: dict[str, object] = {
        "path": path,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }
    if extra is not None:
        expected.update(extra)
    if value != expected:
        raise ValueError(f"monitor {label} binding differs")


def _validate_observation_artifacts(
    context: _HeartbeatContext,
    *,
    ordinal: int,
    claim: Mapping[str, object],
    claim_raw: bytes,
    ledger_payload: Mapping[str, object],
) -> None:
    layout = context.layout
    directory = layout.observation_dir(ordinal)
    transcript, transcript_raw = _read_canonical(
        layout,
        directory / "transcript.json",
        {
            "schema_version",
            "domain",
            "state",
            "observation_id",
            "ordinal",
            "claim_sha256",
            "transcript_payload_sha256",
            "request_count",
            "diagnostic",
            "authorization",
        },
        "transcript",
    )
    raw, raw_bytes = _read_canonical(
        layout,
        directory / "raw.json",
        {
            "schema_version",
            "domain",
            "state",
            "observation_id",
            "ordinal",
            "claim_sha256",
            "request_contract",
            "request_count",
            "pages",
            "diagnostic",
            "failure",
            "timing",
            "authorization",
        },
        "raw",
    )
    terminal_payload, _ = _verify_envelope(
        _read_safe_file(layout, directory / "terminal-auth.json", "terminal-auth"),
        domain=TERMINAL_AUTH_ENVELOPE_DOMAIN,
        state="signed_observation_terminal",
        payload_domain=TERMINAL_AUTH_PAYLOAD_DOMAIN,
        namespace=OBSERVATION_NAMESPACE,
        identity=context.signer.identity,
    )
    terminal_keys = {
        "schema_version",
        "domain",
        "state",
        "monitor_id",
        "observation_id",
        "ordinal",
        "claim",
        "prior_ledger_sha256",
        "transcript",
        "raw",
        "request_count",
        "endpoint_order",
        "captured_page_count",
        "row_count",
        "diagnostic",
        "checkpoint",
        "capture_owner_projection_sha256",
        "request_contract_projection_sha256",
        "public_signing",
        "openssh_executable",
        "timing",
        "attempt_consumed",
        "retry",
        "assessment_allowed",
        "authorization",
    }
    if set(terminal_payload) != terminal_keys:
        raise ValueError("monitor terminal payload schema differs")
    pages = raw["pages"]
    if not isinstance(pages, list):
        raise ValueError("monitor raw pages differ")
    typed_pages = cast(list[dict[str, object]], pages)
    replay = v5._replay_transcript(pages, raw["diagnostic"], raw["failure"], raw["request_count"])
    row_count = sum(
        len(v5._pagination(_authenticated_page_body(page)[2])[1]) for page in typed_pages
    )
    identity = _observation_id(ordinal)
    expected_complete = raw["diagnostic"] is None
    if (
        transcript["schema_version"] != 1
        or transcript["domain"] != TRANSCRIPT_DOMAIN
        or transcript["state"] != "observation_transcript_terminal"
        or transcript["observation_id"] != identity
        or transcript["ordinal"] != ordinal
        or transcript["claim_sha256"] != _sha256(claim_raw)
        or transcript["transcript_payload_sha256"] != replay["payload_sha256"]
        or transcript["request_count"] != raw["request_count"]
        or transcript["diagnostic"] != raw["diagnostic"]
        or transcript["authorization"] != AUTHORIZATION_FALSE
    ):
        raise ValueError("monitor transcript binding differs")
    expected_raw_state = (
        "captured_endpoint_terminal" if expected_complete else "capture_failed_terminal"
    )
    if (
        raw["schema_version"] != 1
        or raw["domain"] != RAW_DOMAIN
        or raw["state"] != expected_raw_state
        or raw["observation_id"] != identity
        or raw["ordinal"] != ordinal
        or raw["claim_sha256"] != _sha256(claim_raw)
        or raw["request_contract"] != v5._request_contract()
        or raw["authorization"] != AUTHORIZATION_FALSE
        or raw["timing"] != ledger_payload["timing"]
    ):
        raise ValueError("monitor raw binding differs")
    _require_binding(
        terminal_payload["transcript"],
        path=layout.relative_artifact(ordinal, "transcript.json"),
        raw=transcript_raw,
        label="terminal transcript",
        extra={"payload_sha256": replay["payload_sha256"]},
    )
    _require_binding(
        terminal_payload["raw"],
        path=layout.relative_artifact(ordinal, "raw.json"),
        raw=raw_bytes,
        label="terminal raw",
    )
    claim_binding = terminal_payload["claim"]
    if claim_binding != {
        "path": layout.relative_artifact(ordinal, "claim.json"),
        "sha256": _sha256(claim_raw),
    }:
        raise ValueError("monitor terminal claim binding differs")
    if (
        terminal_payload["schema_version"] != 1
        or terminal_payload["monitor_id"] != MONITOR_ID
        or terminal_payload["observation_id"] != identity
        or terminal_payload["ordinal"] != ordinal
        or terminal_payload["prior_ledger_sha256"] != ledger_payload["previous_ledger_sha256"]
        or terminal_payload["request_count"] != raw["request_count"]
        or terminal_payload["endpoint_order"] != list(v5.ENDPOINTS)
        or terminal_payload["captured_page_count"] != len(pages)
        or terminal_payload["row_count"] != row_count
        or terminal_payload["diagnostic"] != raw["diagnostic"]
        or terminal_payload["checkpoint"] != context.checkpoint
        or terminal_payload["capture_owner_projection_sha256"]
        != _sha256(_canonical(context.capture_owners))
        or terminal_payload["request_contract_projection_sha256"]
        != _sha256(_canonical(v5._request_contract()))
        or terminal_payload["public_signing"]
        != context.signer.identity.projection(OBSERVATION_NAMESPACE)
        or terminal_payload["openssh_executable"] != context.openssh_executable
        or terminal_payload["timing"] != raw["timing"]
        or terminal_payload["attempt_consumed"] is not True
        or terminal_payload["retry"] is not False
        or terminal_payload["assessment_allowed"] is not expected_complete
        or terminal_payload["authorization"] != AUTHORIZATION_FALSE
    ):
        raise ValueError("monitor terminal authentication binding differs")
    if expected_complete:
        if terminal_payload["state"] != "authenticated_observation_complete":
            raise ValueError("monitor completed terminal state differs")
        assessment = _canonical(_assessment_value(typed_pages))
        if _read_safe_file(layout, directory / "assessment.json", "assessment") != assessment:
            raise ValueError("monitor assessment semantics differ")
        value = cast(dict[str, object], json.loads(assessment))
        if (
            ledger_payload["formal_state"] != value["state"]
            or ledger_payload["reason"] != value["reason"]
            or ledger_payload["counts"] != value["counts"]
        ):
            raise ValueError("monitor ledger assessment binding differs")
    else:
        if terminal_payload["state"] != "authenticated_observation_incomplete":
            raise ValueError("monitor incomplete terminal state differs")
        terminal, _ = _read_canonical(
            layout,
            directory / "terminal.json",
            {
                "schema_version",
                "domain",
                "state",
                "monitor_id",
                "observation_id",
                "ordinal",
                "claim_sha256",
                "diagnostic",
                "attempt_consumed",
                "retry",
                "authorization",
            },
            "terminal",
        )
        if (
            terminal["domain"] != TERMINAL_DOMAIN
            or terminal["observation_id"] != identity
            or terminal["ordinal"] != ordinal
            or terminal["claim_sha256"] != _sha256(claim_raw)
            or terminal["diagnostic"] != raw["diagnostic"]
            or terminal["attempt_consumed"] is not True
            or terminal["retry"] is not False
            or terminal["authorization"] != AUTHORIZATION_FALSE
            or ledger_payload["formal_state"] != "capture_failed_terminal"
            or ledger_payload["reason"] != raw["diagnostic"]
        ):
            raise ValueError("monitor failure terminal binding differs")


def _validate_chain(context: _HeartbeatContext) -> _ChainState:
    layout = context.layout
    _prepare_monitor_root(layout)
    for path in (layout.observations, layout.ledger):
        candidate = _safe_path(layout, path)
        if candidate.exists():
            if not candidate.is_dir() or _is_link_or_reparse(candidate):
                raise ValueError("monitor evidence directory differs")
        else:
            candidate.mkdir(exist_ok=False)
            if not candidate.is_dir() or _is_link_or_reparse(candidate):
                raise ValueError("monitor evidence directory changed during creation")
    observation_names = sorted(path.name for path in layout.observations.iterdir())
    ledger_names = sorted(path.name for path in layout.ledger.iterdir())
    if any(not re.fullmatch(r"\d{8}", name) for name in observation_names):
        raise ValueError("monitor observation directory set differs")
    if any(not re.fullmatch(r"\d{8}\.json", name) for name in ledger_names):
        raise ValueError("monitor ledger file set differs")
    previous_hash = _genesis_hash(context.repository)
    previous_claim: int | None = None
    previous_end: int | None = None
    first_claim: int | None = None
    stopped = False
    stop_disposition: str | None = None
    expected_count = len(observation_names)
    if observation_names != [f"{value:08d}" for value in range(1, expected_count + 1)]:
        raise ValueError("monitor observation ordinals differ")
    if ledger_names != [f"{value:08d}.json" for value in range(1, expected_count + 1)]:
        raise ValueError("monitor incomplete or discontinuous allocated observation")
    for ordinal in range(1, expected_count + 1):
        directory = layout.observation_dir(ordinal)
        if _is_link_or_reparse(directory) or not directory.is_dir():
            raise ValueError("monitor observation directory is linked or absent")
        names = {path.name for path in directory.iterdir()}
        normal = {
            "claim.json",
            "transcript.json",
            "raw.json",
            "terminal-auth.json",
            "assessment.json",
        }
        failed = {
            "claim.json",
            "transcript.json",
            "raw.json",
            "terminal-auth.json",
            "terminal.json",
        }
        if names not in (normal, failed):
            raise ValueError("monitor observation artifact set differs")
        ledger_raw = _read_safe_file(layout, layout.ledger_record(ordinal), "signed ledger record")
        payload, _payload_raw = _verify_envelope(
            ledger_raw,
            domain=LEDGER_ENVELOPE_DOMAIN,
            state="signed_ledger_record",
            payload_domain=LEDGER_PAYLOAD_DOMAIN,
            namespace=LEDGER_NAMESPACE,
            identity=context.signer.identity,
        )
        required = {
            "schema_version",
            "domain",
            "state",
            "monitor_id",
            "observation_id",
            "ordinal",
            "previous_ledger_sha256",
            "artifacts",
            "counts",
            "formal_state",
            "reason",
            "resource",
            "timing",
            "disposition",
            "attempt_limit",
            "retry",
            "authorization",
        }
        if (
            set(payload) != required
            or payload["schema_version"] != 1
            or payload["monitor_id"] != MONITOR_ID
        ):
            raise ValueError("monitor ledger payload schema differs")
        if payload["ordinal"] != ordinal or payload["observation_id"] != _observation_id(ordinal):
            raise ValueError("monitor ledger identity differs")
        if payload["previous_ledger_sha256"] != previous_hash:
            raise ValueError("monitor ledger chain differs")
        if (
            payload["attempt_limit"] != 1
            or payload["retry"] is not False
            or payload["authorization"] != AUTHORIZATION_FALSE
        ):
            raise ValueError("monitor ledger authority differs")
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, dict) or set(artifacts) != names:
            raise ValueError("monitor ledger artifact key set differs")
        for name in names:
            if artifacts[name] != _artifact_binding(layout, directory / name):
                raise ValueError("monitor ledger artifact binding differs")
        claim, _claim_raw = _read_canonical(
            layout,
            directory / "claim.json",
            {
                "schema_version",
                "domain",
                "state",
                "monitor_id",
                "observation_id",
                "ordinal",
                "nonce",
                "checkpoint",
                "contract_sha256",
                "audit_sha256",
                "v5_bindings",
                "previous_ledger_sha256",
                "previous_end_epoch",
                "next_not_before_epoch",
                "paths",
                "request_contract",
                "public_signing",
                "openssh_executable",
                "attempt_limit",
                "attempt_consumed",
                "retry",
                "authorization",
            },
            "claim",
        )
        if (
            claim["schema_version"] != 1
            or claim["domain"] != CLAIM_DOMAIN
            or claim["state"] != "observation_claimed_attempt_consumed"
            or claim["monitor_id"] != MONITOR_ID
            or claim["ordinal"] != ordinal
            or claim["observation_id"] != _observation_id(ordinal)
            or not isinstance(claim["nonce"], str)
            or re.fullmatch(r"[0-9a-f]{64}", claim["nonce"]) is None
            or claim["checkpoint"] != context.checkpoint
            or claim["contract_sha256"] != _contract_hash(context.repository)
            or claim["audit_sha256"] != _audit_hash(context.repository)
            or claim["v5_bindings"]
            != {
                "checkpoint": V5_CHECKPOINT,
                "owner_sha256": V5_OWNER_SHA256,
                "contract_sha256": V5_CONTRACT_SHA256,
                "terminal_report_sha256": V5_REPORT_SHA256,
            }
            or claim["previous_ledger_sha256"] != previous_hash
            or claim["previous_end_epoch"] != previous_end
            or claim["paths"]
            != {
                name.removesuffix(".json"): layout.relative_artifact(ordinal, name)
                for name in _OBSERVATION_FILES
            }
            | {"ledger": layout.relative_ledger(ordinal)}
            or claim["request_contract"] != v5._request_contract()
            or claim["public_signing"] != context.signer.identity.projection(OBSERVATION_NAMESPACE)
            or claim["openssh_executable"] != context.openssh_executable
            or claim["attempt_limit"] != 1
            or claim["attempt_consumed"] is not True
            or claim["retry"] is not False
            or claim["authorization"] != AUTHORIZATION_FALSE
        ):
            raise ValueError("monitor claim binding differs")
        timing = payload["timing"]
        if not isinstance(timing, dict) or set(timing) != {
            "start_epoch",
            "end_epoch",
            "elapsed_monotonic_seconds",
            "next_not_before_epoch",
        }:
            raise ValueError("monitor ledger timing differs")
        start = timing["start_epoch"]
        end = timing["end_epoch"]
        elapsed = timing["elapsed_monotonic_seconds"]
        if (
            type(start) is not int
            or type(end) is not int
            or type(elapsed) not in (int, float)
            or end < start
            or elapsed < 0
        ):
            raise ValueError("monitor ledger time values differ")
        if previous_claim is not None and start < previous_claim + MINIMUM_CADENCE_SECONDS:
            raise ValueError("monitor ledger cadence differs")
        if claim["next_not_before_epoch"] != start + MINIMUM_CADENCE_SECONDS:
            raise ValueError("monitor claim cadence binding differs")
        if timing["next_not_before_epoch"] != start + MINIMUM_CADENCE_SECONDS:
            raise ValueError("monitor ledger cadence binding differs")
        _validate_observation_artifacts(
            context,
            ordinal=ordinal,
            claim=claim,
            claim_raw=_claim_raw,
            ledger_payload=payload,
        )
        if first_claim is None:
            first_claim = start
        previous_claim = start
        previous_end = end
        previous_hash = _sha256(ledger_raw)
        disposition = payload["disposition"]
        if disposition != "continue_monitoring":
            stopped = True
            stop_disposition = cast(str, disposition)
            if ordinal != expected_count:
                raise ValueError("monitor ledger continues after stop")
    return _ChainState(
        expected_count + 1,
        previous_hash,
        previous_claim,
        previous_end,
        first_claim,
        stopped,
        stop_disposition,
    )


def _root_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _authenticate_storage_law() -> None:
    if (
        MAXIMUM_ENDPOINT_PAGES != 200
        or v5.MAX_CUMULATIVE_BODY_BYTES != 33_554_432
        or MAXIMUM_VALID_OBSERVATION_BYTES > PRECLAIM_FREE_SPACE_RESERVE
    ):
        raise ValueError("monitor preclaim reservation cannot cover a valid observation")


def _validate_capture_storage(pages: list[dict[str, object]]) -> None:
    encoded_bytes = 0
    overhead_bytes = 0
    for page in pages:
        encoded = page.get("decoded_application_body_b64")
        if not isinstance(encoded, str):
            raise ValueError("monitor capture body encoding differs")
        encoded_bytes += len(encoded.encode("ascii"))
        overhead = len(_canonical(page)) - len(encoded)
        if overhead > MAXIMUM_PAGE_RECORD_OVERHEAD_BYTES:
            raise ValueError("monitor capture page metadata exceeds frozen bound")
        overhead_bytes += overhead
    if (
        len(pages) > MAXIMUM_ENDPOINT_PAGES
        or encoded_bytes > MAXIMUM_ENCODED_BODY_BYTES
        or overhead_bytes > MAXIMUM_ENDPOINT_PAGES * MAXIMUM_PAGE_RECORD_OVERHEAD_BYTES
        or encoded_bytes + overhead_bytes + MAXIMUM_NON_BODY_ARTIFACT_BYTES
        > PRECLAIM_FREE_SPACE_RESERVE
    ):
        raise ValueError("monitor capture exceeds frozen observation reservation")


def _terminal_result(state: str, ordinal: int | None = None) -> HeartbeatResult:
    return HeartbeatResult(state, state, False, ordinal)


def _write_terminal(
    context: _HeartbeatContext, ordinal: int, claim_raw: bytes, diagnostic: str
) -> None:
    terminal = _canonical(
        {
            "schema_version": 1,
            "domain": TERMINAL_DOMAIN,
            "state": "terminal_failure",
            "monitor_id": MONITOR_ID,
            "observation_id": _observation_id(ordinal),
            "ordinal": ordinal,
            "claim_sha256": _sha256(claim_raw),
            "diagnostic": diagnostic,
            "attempt_consumed": True,
            "retry": False,
            "authorization": AUTHORIZATION_FALSE,
        }
    )
    with contextlib.suppress(OSError, ValueError):
        _publish_exclusive(
            context.layout, context.layout.artifact(ordinal, "terminal.json"), terminal
        )


def _publish_ledger(
    context: _HeartbeatContext,
    *,
    ordinal: int,
    previous_hash: str,
    assessment: Mapping[str, object] | None,
    formal_state: str,
    reason: str,
    timing: Mapping[str, object],
    disposition: str,
) -> bytes:
    directory = context.layout.observation_dir(ordinal)
    names = {path.name for path in directory.iterdir()}
    counts = (
        cast(dict[str, object], assessment["counts"])
        if assessment is not None
        else {
            "row_occurrences": 0,
            "distinct_identity_values": 0,
            "duplicated_identity_values": 0,
            "extra_duplicate_occurrences": 0,
            "qualifying_row_occurrences": 0,
        }
    )
    payload = _canonical(
        {
            "schema_version": 1,
            "domain": LEDGER_PAYLOAD_DOMAIN,
            "state": "ledger_record_terminal",
            "monitor_id": MONITOR_ID,
            "observation_id": _observation_id(ordinal),
            "ordinal": ordinal,
            "previous_ledger_sha256": previous_hash,
            "artifacts": {
                name: _artifact_binding(context.layout, directory / name) for name in sorted(names)
            },
            "counts": counts,
            "formal_state": formal_state,
            "reason": reason,
            "resource": None,
            "timing": dict(timing),
            "disposition": disposition,
            "attempt_limit": 1,
            "retry": False,
            "authorization": AUTHORIZATION_FALSE,
        }
    )
    signature = _sign_bytes(context.signer, payload, LEDGER_NAMESPACE)
    envelope = _envelope(
        domain=LEDGER_ENVELOPE_DOMAIN,
        state="signed_ledger_record",
        payload=payload,
        signature=signature,
        identity=context.signer.identity,
        namespace=LEDGER_NAMESPACE,
    )
    _verify_envelope(
        envelope,
        domain=LEDGER_ENVELOPE_DOMAIN,
        state="signed_ledger_record",
        payload_domain=LEDGER_PAYLOAD_DOMAIN,
        namespace=LEDGER_NAMESPACE,
        identity=context.signer.identity,
    )
    _publish_exclusive(context.layout, context.layout.ledger_record(ordinal), envelope)
    return envelope


def _heartbeat_locked(context: _HeartbeatContext) -> HeartbeatResult:
    _authenticate_storage_law()
    try:
        chain = _validate_chain(context)
    except (OSError, ValueError):
        return _terminal_result("prior_chain_invalid_terminal")
    if chain.stopped:
        return _terminal_result(cast(str, chain.stop_disposition))
    now = context.clock.wall()
    if chain.previous_claim_epoch is not None:
        if chain.previous_end_epoch is None or now < chain.previous_end_epoch:
            return _terminal_result("clock_rollback_terminal")
        next_not_before = chain.previous_claim_epoch + MINIMUM_CADENCE_SECONDS
        if now < next_not_before:
            return HeartbeatResult(
                "cadence_not_elapsed_noop",
                "cadence_not_elapsed_noop",
                True,
                next_not_before_epoch=next_not_before,
            )
    if chain.next_ordinal > MAXIMUM_OBSERVATIONS:
        return _terminal_result("monitor_window_exhausted_no_capacity")
    if (
        chain.first_claim_epoch is not None
        and now - chain.first_claim_epoch >= MAXIMUM_WINDOW_SECONDS
    ):
        return _terminal_result("monitor_window_exhausted_no_capacity")
    if (
        _root_bytes(context.layout.root) + PRECLAIM_FREE_SPACE_RESERVE > MAXIMUM_MONITOR_BYTES
        or context.free_bytes(context.layout.root) < PRECLAIM_FREE_SPACE_RESERVE
    ):
        return _terminal_result("storage_limit_terminal")
    ordinal = chain.next_ordinal
    directory = context.layout.observation_dir(ordinal)
    try:
        directory.mkdir(parents=False, exist_ok=False)
    except OSError:
        return _terminal_result("observation_allocation_terminal", ordinal)
    start_epoch = now
    start_monotonic = context.clock.monotonic()
    next_not_before = start_epoch + MINIMUM_CADENCE_SECONDS
    claim = _canonical(
        {
            "schema_version": 1,
            "domain": CLAIM_DOMAIN,
            "state": "observation_claimed_attempt_consumed",
            "monitor_id": MONITOR_ID,
            "observation_id": _observation_id(ordinal),
            "ordinal": ordinal,
            "nonce": secrets.token_hex(32),
            "checkpoint": dict(context.checkpoint),
            "contract_sha256": _contract_hash(context.repository),
            "audit_sha256": _audit_hash(context.repository),
            "v5_bindings": {
                "checkpoint": V5_CHECKPOINT,
                "owner_sha256": V5_OWNER_SHA256,
                "contract_sha256": V5_CONTRACT_SHA256,
                "terminal_report_sha256": V5_REPORT_SHA256,
            },
            "previous_ledger_sha256": chain.previous_ledger_sha256,
            "previous_end_epoch": chain.previous_end_epoch,
            "next_not_before_epoch": next_not_before,
            "paths": {
                name.removesuffix(".json"): context.layout.relative_artifact(ordinal, name)
                for name in _OBSERVATION_FILES
            }
            | {"ledger": context.layout.relative_ledger(ordinal)},
            "request_contract": v5._request_contract(),
            "public_signing": context.signer.identity.projection(OBSERVATION_NAMESPACE),
            "openssh_executable": dict(context.openssh_executable),
            "attempt_limit": 1,
            "attempt_consumed": True,
            "retry": False,
            "authorization": AUTHORIZATION_FALSE,
        }
    )
    try:
        _publish_exclusive(context.layout, context.layout.artifact(ordinal, "claim.json"), claim)
    except (OSError, ValueError):
        return _terminal_result("claim_publication_terminal", ordinal)
    stage = "capture"
    try:
        client = context.client_factory()
        if client.base_url != v5.BASE_URL or not client.api_key:
            raise ValueError("client binding differs")
        pages, diagnostic, failure, request_count = v5._capture_pages(
            client, context.transport_errors
        )
        end_epoch = context.clock.wall()
        elapsed = context.clock.monotonic() - start_monotonic
        if end_epoch < start_epoch or elapsed < 0:
            raise ValueError("clock rollback")
        replay = v5._replay_transcript(pages, diagnostic, failure, request_count)
        _validate_capture_storage(pages)
        transcript = _canonical(
            {
                "schema_version": 1,
                "domain": TRANSCRIPT_DOMAIN,
                "state": "observation_transcript_terminal",
                "observation_id": _observation_id(ordinal),
                "ordinal": ordinal,
                "claim_sha256": _sha256(claim),
                "transcript_payload_sha256": replay["payload_sha256"],
                "request_count": request_count,
                "diagnostic": diagnostic,
                "authorization": AUTHORIZATION_FALSE,
            }
        )
        raw = _canonical(
            {
                "schema_version": 1,
                "domain": RAW_DOMAIN,
                "state": "captured_endpoint_terminal"
                if diagnostic is None
                else "capture_failed_terminal",
                "observation_id": _observation_id(ordinal),
                "ordinal": ordinal,
                "claim_sha256": _sha256(claim),
                "request_contract": v5._request_contract(),
                "request_count": request_count,
                "pages": pages,
                "diagnostic": diagnostic,
                "failure": failure,
                "timing": {
                    "start_epoch": start_epoch,
                    "end_epoch": end_epoch,
                    "elapsed_monotonic_seconds": elapsed,
                    "next_not_before_epoch": next_not_before,
                },
                "authorization": AUTHORIZATION_FALSE,
            }
        )
        stage = "terminal_authentication"
        payload = _canonical(
            {
                "schema_version": 1,
                "domain": TERMINAL_AUTH_PAYLOAD_DOMAIN,
                "state": "authenticated_observation_complete"
                if diagnostic is None
                else "authenticated_observation_incomplete",
                "monitor_id": MONITOR_ID,
                "observation_id": _observation_id(ordinal),
                "ordinal": ordinal,
                "claim": {
                    "path": context.layout.relative_artifact(ordinal, "claim.json"),
                    "sha256": _sha256(claim),
                },
                "prior_ledger_sha256": chain.previous_ledger_sha256,
                "transcript": {
                    "path": context.layout.relative_artifact(ordinal, "transcript.json"),
                    "bytes": len(transcript),
                    "sha256": _sha256(transcript),
                    "payload_sha256": replay["payload_sha256"],
                },
                "raw": {
                    "path": context.layout.relative_artifact(ordinal, "raw.json"),
                    "bytes": len(raw),
                    "sha256": _sha256(raw),
                },
                "request_count": request_count,
                "endpoint_order": list(v5.ENDPOINTS),
                "captured_page_count": len(pages),
                "row_count": sum(
                    len(v5._pagination(_authenticated_page_body(page)[2])[1]) for page in pages
                ),
                "diagnostic": diagnostic,
                "checkpoint": dict(context.checkpoint),
                "capture_owner_projection_sha256": _sha256(_canonical(context.capture_owners)),
                "request_contract_projection_sha256": _sha256(_canonical(v5._request_contract())),
                "public_signing": context.signer.identity.projection(OBSERVATION_NAMESPACE),
                "openssh_executable": dict(context.openssh_executable),
                "timing": {
                    "start_epoch": start_epoch,
                    "end_epoch": end_epoch,
                    "elapsed_monotonic_seconds": elapsed,
                    "next_not_before_epoch": next_not_before,
                },
                "attempt_consumed": True,
                "retry": False,
                "assessment_allowed": diagnostic is None,
                "authorization": AUTHORIZATION_FALSE,
            }
        )
        signature = _sign_bytes(context.signer, payload, OBSERVATION_NAMESPACE)
        terminal_auth = _envelope(
            domain=TERMINAL_AUTH_ENVELOPE_DOMAIN,
            state="signed_observation_terminal",
            payload=payload,
            signature=signature,
            identity=context.signer.identity,
            namespace=OBSERVATION_NAMESPACE,
        )
        _verify_envelope(
            terminal_auth,
            domain=TERMINAL_AUTH_ENVELOPE_DOMAIN,
            state="signed_observation_terminal",
            payload_domain=TERMINAL_AUTH_PAYLOAD_DOMAIN,
            namespace=OBSERVATION_NAMESPACE,
            identity=context.signer.identity,
        )
        stage = "artifact_publication"
        _publish_exclusive(
            context.layout, context.layout.artifact(ordinal, "transcript.json"), transcript
        )
        _publish_exclusive(context.layout, context.layout.artifact(ordinal, "raw.json"), raw)
        _publish_exclusive(
            context.layout, context.layout.artifact(ordinal, "terminal-auth.json"), terminal_auth
        )
        timing = {
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "elapsed_monotonic_seconds": elapsed,
            "next_not_before_epoch": next_not_before,
        }
        if diagnostic is not None:
            stage = "capture_terminal"
            _write_terminal(context, ordinal, claim, diagnostic)
            ledger = _publish_ledger(
                context,
                ordinal=ordinal,
                previous_hash=chain.previous_ledger_sha256,
                assessment=None,
                formal_state="capture_failed_terminal",
                reason=diagnostic,
                timing=timing,
                disposition="terminal_failure",
            )
            return HeartbeatResult(
                "capture_failed_terminal",
                "terminal_failure",
                False,
                ordinal,
                ledger_sha256=_sha256(ledger),
            )
        stage = "assessment"
        assessment = _assessment_value(pages)
        assessment_raw = _canonical(assessment)
        _publish_exclusive(
            context.layout, context.layout.artifact(ordinal, "assessment.json"), assessment_raw
        )
        state = cast(str, assessment["state"])
        reason = cast(str, assessment["reason"])
        if state == "observed_non_authorizing_resource":
            disposition = "qualifying_capacity_found_stop"
        elif ordinal >= MAXIMUM_OBSERVATIONS or (
            chain.first_claim_epoch is not None
            and end_epoch - chain.first_claim_epoch >= MAXIMUM_WINDOW_SECONDS
        ):
            disposition = "monitor_window_exhausted_no_capacity"
        else:
            disposition = "continue_monitoring"
        stage = "ledger"
        ledger = _publish_ledger(
            context,
            ordinal=ordinal,
            previous_hash=chain.previous_ledger_sha256,
            assessment=assessment,
            formal_state=state,
            reason=reason,
            timing=timing,
            disposition=disposition,
        )
        return HeartbeatResult(
            state,
            disposition,
            disposition == "continue_monitoring",
            ordinal,
            next_not_before if disposition == "continue_monitoring" else None,
            _sha256(ledger),
        )
    except (OSError, ValueError, TypeError):
        _write_terminal(context, ordinal, claim, f"{stage}_failure")
        return _terminal_result(f"{stage}_failure_terminal", ordinal)


def _run_heartbeat(context: _HeartbeatContext) -> HeartbeatResult:
    with _monitor_lock(context.layout) as acquired:
        if not acquired:
            return HeartbeatResult("overlap_noop", "overlap_noop", True)
        return _heartbeat_locked(context)


def _production_context() -> _HeartbeatContext:
    if sys.platform != "win32" or sys.version_info[:3] != (3, 13, 2):
        raise ValueError("monitor requires authenticated Prime uv-tool CPython 3.13.2")
    expected_python = (
        Path(os.environ["APPDATA"]) / "uv" / "tools" / "prime" / "Scripts" / "python.exe"
    ).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ValueError("monitor interpreter differs")
    checkpoint = authenticate_monitor_checkpoint(ROOT)
    _authenticate_immutable_inputs(ROOT)
    owners = v5.authenticate_installed_capture_owners()
    v5._authenticate_config_paths()
    openssh = v5.authenticate_approved_openssh_executable()
    source_identity = v5._load_terminal_signing_identity()
    identity = _identity_from_v5(source_identity)
    key = Path.home() / ".ssh" / "id_rsa"
    v5._authenticate_operator_key(key, source_identity)
    return _HeartbeatContext(
        repository=ROOT,
        layout=ObservationLayout(ROOT / MONITOR_ROOT_RELATIVE),
        signer=_Signer(identity, key),
        checkpoint=checkpoint,
        capture_owners=owners,
        openssh_executable=openssh,
        client_factory=lambda: cast(_Client, v5._construct_api_client()),
        transport_errors=v5._httpx_request_error_types(),
        clock=_SystemClock(),
        free_bytes=lambda path: shutil.disk_usage(path).free,
    )


def run_capacity_monitor_heartbeat_v1() -> HeartbeatResult:
    """Run at most one fixed, signed, non-authorizing availability observation."""

    return _run_heartbeat(_production_context())


def build_checkpoint_artifacts(root: Path) -> dict[str, bytes]:
    authenticate_monitor_checkpoint(root, precommit=True)
    _authenticate_immutable_inputs(root)
    source_hashes = {
        relative: _sha256((root / relative).read_bytes())
        for relative in (OWNER_RELATIVE, RUNNER_RELATIVE, TEST_RELATIVE)
    }
    contract = _canonical(
        {
            "schema_version": 1,
            "domain": CONTRACT_DOMAIN,
            "state": "non_authorizing_cpu_monitor_contract",
            "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
            "source_task_id": SOURCE_TASK_ID,
            "monitor_authorization": {
                "utf8_bytes": len(AUTHORIZATION_TEXT.encode("utf-8")),
                "text": AUTHORIZATION_TEXT,
                "sha256": AUTHORIZATION_SHA256,
            },
            "immutable_v5": {
                "checkpoint": V5_CHECKPOINT,
                "owner_sha256": V5_OWNER_SHA256,
                "contract_sha256": V5_CONTRACT_SHA256,
                "terminal_report": {"path": V5_REPORT_RELATIVE, "sha256": V5_REPORT_SHA256},
            },
            "source_hashes": source_hashes,
            "tracked_paths": sorted(CHECKPOINT_PATHS),
            "monitor_root": MONITOR_ROOT_RELATIVE,
            "domains": {
                "claim": CLAIM_DOMAIN,
                "transcript": TRANSCRIPT_DOMAIN,
                "raw": RAW_DOMAIN,
                "terminal_auth_payload": TERMINAL_AUTH_PAYLOAD_DOMAIN,
                "terminal_auth_envelope": TERMINAL_AUTH_ENVELOPE_DOMAIN,
                "assessment": ASSESSMENT_DOMAIN,
                "terminal": TERMINAL_DOMAIN,
                "ledger_payload": LEDGER_PAYLOAD_DOMAIN,
                "ledger_envelope": LEDGER_ENVELOPE_DOMAIN,
            },
            "signing_namespaces": {
                "observation": OBSERVATION_NAMESPACE,
                "ledger": LEDGER_NAMESPACE,
            },
            "v5_primitive_allowlist": list(V5_PRIMITIVE_ALLOWLIST),
            "v5_forbidden_owners": [
                "_sign_bytes",
                "_verify_signature",
                "_terminal_auth_payload",
                "_terminal_auth_envelope",
                "_assessment_value",
                "_publish_fixed",
                "_fixed_path",
                "_authenticate_committed_capture_checkout",
            ],
            "request_contract": v5._request_contract(),
            "limits": {
                "minimum_cadence_seconds": MINIMUM_CADENCE_SECONDS,
                "cadence_owner": "prior_claim_creation_epoch",
                "maximum_observations": MAXIMUM_OBSERVATIONS,
                "maximum_window_seconds": MAXIMUM_WINDOW_SECONDS,
                "maximum_monitor_bytes": MAXIMUM_MONITOR_BYTES,
                "preclaim_free_space_reserve": PRECLAIM_FREE_SPACE_RESERVE,
                "attempt_limit": 1,
                "retries": 0,
                "maximum_valid_observation_bytes": MAXIMUM_VALID_OBSERVATION_BYTES,
            },
            "external_gitlink": {
                "path": EXTERNAL_GITLINK_PATH,
                "porcelain": EXTERNAL_GITLINK_STATUS,
                "mode": EXTERNAL_GITLINK_MODE,
                "object": EXTERNAL_GITLINK_OBJECT,
            },
            "runtime": {
                "platform": "windows",
                "python": "3.13.2",
                "uv_mode": [
                    "--no-project",
                    "--offline",
                    "--python",
                    "authenticated-prime-tool-python",
                ],
                "fixed_no_argument_runner": RUNNER_RELATIVE,
            },
            "dispositions": [
                "continue_monitoring",
                "cadence_not_elapsed_noop",
                "overlap_noop",
                "qualifying_capacity_found_stop",
                "monitor_window_exhausted_no_capacity",
                "clock_rollback_terminal",
                "storage_limit_terminal",
                "terminal_failure",
            ],
            "exit_codes": {"continue_or_noop": 0, "qualifying_stop": 10, "terminal": 20},
            "authorization": AUTHORIZATION_FALSE,
            "live_monitoring_authorized_by_checkpoint": False,
        }
    )
    audit = _canonical(
        {
            "schema_version": 1,
            "domain": AUDIT_DOMAIN,
            "state": "non_authorizing_cpu_monitor_audit",
            "parent": {"commit": PARENT_COMMIT, "tree": PARENT_TREE},
            "contract": {"path": CONTRACT_RELATIVE, "sha256": _sha256(contract)},
            "file_bindings": source_hashes,
            "verification": {
                "source_free": True,
                "network_calls": 0,
                "prime_calls": 0,
                "live_observations": 0,
                "required_zero_skips": True,
                "test_path": TEST_RELATIVE,
                "ruff_version": "0.16.0",
                "mypy_version": "2.3.0",
            },
            "immutability": {
                "v5_checkpoint": V5_CHECKPOINT,
                "v5_terminal_report_sha256": V5_REPORT_SHA256,
                "external_prime_rl_untouched": True,
            },
            "authorization": AUTHORIZATION_FALSE,
        }
    )
    return {CONTRACT_RELATIVE: contract, AUDIT_RELATIVE: audit}


def verify_checkpoint_artifacts(root: Path) -> dict[str, str]:
    expected = build_checkpoint_artifacts(root)
    result: dict[str, str] = {}
    for relative, raw in expected.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise ValueError(f"monitor checkpoint artifact differs: {relative}")
        result[relative] = _sha256(raw)
    return result


__all__ = [
    "AUDIT_RELATIVE",
    "CONTRACT_RELATIVE",
    "HeartbeatResult",
    "ObservationLayout",
    "authenticate_monitor_checkpoint",
    "build_checkpoint_artifacts",
    "run_capacity_monitor_heartbeat_v1",
    "verify_checkpoint_artifacts",
]
