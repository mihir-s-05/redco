"""Immutable contracts for the current-lineage Prime test-only one-shot."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

ROOT = Path(__file__).resolve().parents[3]
IMPLEMENTATION_PARENT = "e59fe73a70c2fb15f6f87baf7d1361140ec13db1"
IMPLEMENTATION_PARENT_TREE = "57f708f320ad3088942ef4f764a0ebe95c6b53b3"
EXTERNAL_GITLINK = "external/prime-rl"
EXTERNAL_GITLINK_OBJECT = "3b22dd951cad1036d1fe8dd0a0bfc40807a9b360"

ORIGINAL_AUTHORIZATION = (
    "don't monitor for it but just try getting capacity on prime right now and running "
    "the prime tests there. so the first path but no monitoring, just one shot attempt "
    "at trying to get capacity on prime and using it"
)
ORIGINAL_AUTHORIZATION_BYTES = 211
ORIGINAL_AUTHORIZATION_SHA256 = "8de7846795d7d89fdd78567e895eff8b5481128d986d9b6b7460e767624bdb43"
CLARIFICATION = (
    "what i mean by one shot is don't monitor but if you get prime do everything you "
    "need to do on it"
)
CLARIFICATION_BYTES = 96
CLARIFICATION_SHA256 = "c7221728727753d5cdb6970fe7a98e7ae88392897b63305c944abf8ccc7fd52a"
ORCHESTRATOR_THREAD = "019f9ab9-ec45-7ac3-82b1-09757b92a7c3"

CONTRACT_MODULE = "src/redco/analysis/stage_d_v13_prime_test_one_shot_contract_v2.py"
REMOTE_MODULE = "src/redco/analysis/stage_d_v13_prime_test_one_shot_remote_v2.py"
PRIME_MODULE = "src/redco/analysis/stage_d_v13_prime_test_one_shot_prime_v2.py"
WALLET_MODULE = "src/redco/analysis/stage_d_v13_prime_test_one_shot_wallet_v2.py"
EVIDENCE_MODULE = "src/redco/analysis/stage_d_v13_prime_test_one_shot_evidence_v2.py"
LIFECYCLE_MODULE = "src/redco/analysis/stage_d_v13_prime_test_one_shot_lifecycle_v2.py"
BUILDER_SCRIPT = "scripts/build_stage_d_v13_prime_test_one_shot_readiness_v2.py"
RUNNER_SCRIPT = "scripts/run_stage_d_v13_prime_test_one_shot_v2.py"
CONTRACT_PATH = "configs/stage-d/stage-d1-prime-test-one-shot-readiness-v2.json"
AUDIT_PATH = "reports/stage-d1-prime-test-one-shot-readiness-audit-v2.json"
CONTRACT_TEST = "tests/test_stage_d_v13_prime_test_one_shot_contract_v2.py"
EVIDENCE_TEST = "tests/test_stage_d_v13_prime_test_one_shot_evidence_v2.py"
LIFECYCLE_TEST = "tests/test_stage_d_v13_prime_test_one_shot_lifecycle_v2.py"
WALLET_TEST = "tests/test_stage_d_v13_prime_test_one_shot_wallet_v2.py"
AUTHORIZATION_PATH = "configs/stage-d/stage-d1-prime-test-one-shot-authorization-v2.json"
READINESS_PATHS = frozenset(
    {
        CONTRACT_MODULE,
        REMOTE_MODULE,
        PRIME_MODULE,
        WALLET_MODULE,
        EVIDENCE_MODULE,
        LIFECYCLE_MODULE,
        BUILDER_SCRIPT,
        RUNNER_SCRIPT,
        CONTRACT_PATH,
        AUDIT_PATH,
        CONTRACT_TEST,
        EVIDENCE_TEST,
        LIFECYCLE_TEST,
        WALLET_TEST,
    }
)

READINESS_DOMAIN = "redco-stage-d1-prime-test-one-shot-readiness-v2"
READINESS_AUDIT_DOMAIN = "redco-stage-d1-prime-test-one-shot-readiness-audit-v2"
AUTHORIZATION_DOMAIN = "redco-stage-d1-prime-test-one-shot-authorization-v2"
CLAIM_DOMAIN = "redco-stage-d1-prime-test-one-shot-claim-v2"
ASSESSMENT_DOMAIN = "redco-stage-d1-prime-test-one-shot-assessment-v2"
HANDOFF_DOMAIN = "redco-stage-d1-prime-test-one-shot-handoff-v2"
TERMINAL_DOMAIN = "redco-stage-d1-prime-test-one-shot-terminal-v2"
TERMINAL_PURPOSE = "source_free_prime_integration_tests_only"
SIGNED_ENVELOPE_DOMAIN = "redco-stage-d1-prime-test-one-shot-signed-envelope-v2"
WALLET_ROW_DOMAIN = "redco-stage-d1-prime-test-one-shot-wallet-row-v2"
WALLET_SNAPSHOT_DOMAIN = "redco-stage-d1-prime-test-one-shot-wallet-snapshot-v2"
WALLET_RECONCILIATION_DOMAIN = (
    "redco-stage-d1-prime-test-one-shot-wallet-reconciliation-v2"
)
ASSESSMENT_NAMESPACE = "redco-stage-d1-prime-test-one-shot-assessment-v2"
HANDOFF_NAMESPACE = "redco-stage-d1-prime-test-one-shot-handoff-v2"
TERMINAL_NAMESPACE = "redco-stage-d1-prime-test-one-shot-terminal-v2"

EVIDENCE_ROOT = "runs/stage-d/stage-d1-prime-test-one-shot-v2"
POD_NAME_PREFIX = "redco-stage-d-prime-tests-once-v2"
MAX_COMMAND_OUTPUT_BYTES = 8 * 1024 * 1024
ARTIFACT_FILENAMES = {
    "claim": "claim.json",
    "transcript": "transcript.json",
    "assessment": "assessment.json",
    "assessment-envelope": "assessment-envelope.json",
    "create-dispatch": "create-dispatch.json",
    "create-result": "create-result.json",
    "wallet-before": "wallet-before.json",
    "known-hosts": "known-hosts.txt",
    "handoff": "handoff.json",
    "handoff-signature": "handoff.sig",
    "handoff-envelope": "handoff-envelope.json",
    "command-journal": "command-journal.jsonl",
    "command-records": "command-records.json",
    "gpu-facts": "gpu-facts.json",
    "junit": "pytest.xml",
    "remote-status": "remote-status.json",
    "cleanup": "cleanup.json",
    "terminal": "terminal.json",
    "terminal-envelope": "terminal-envelope.json",
    "terminal-publication-failure": "terminal-publication-failure.json",
}
V5_OWNER_PATH = "src/redco/analysis/stage_d_v13_prime_inventory_v5.py"
V5_OWNER_SHA256 = "699df3d9b591df84dcc24cbec842532683240edaaec6a0aaa62934a56c1a1b9b"
V5_CONTRACT_PATH = "configs/stage-d/stage-d1-support-prime-inventory-contract-v5.json"
V5_CONTRACT_SHA256 = "700bad4bbdfbe2c39bc6ac607fa09f3daa7946cf9ca071c253dbea643d5f78a8"
PODS_API_OWNER = "prime_cli/api/pods.py"
PODS_API_OWNER_SHA256 = "3ddc32fcb713555a771a6d59c54e37ece2baff75279a1d85b2e8372171b56597"
PODS_COMMAND_OWNER = "prime_cli/commands/pods.py"
PODS_COMMAND_OWNER_SHA256 = "614db41ba796dbe0924fce3e9c620f05f54d8125f471f4e34a9bdea911f317bd"
WALLET_API_OWNER = "prime_cli/api/wallet.py"
WALLET_API_OWNER_SHA256 = "13b1bfd545e4e38aedd2d091ae956f471321bdf3f6f6466db4d2ce72d569d6a6"
WALLET_API_ENDPOINT = "https://api.primeintellect.ai/api/v1/billing/wallet"
PRIME_CLIENT_OWNER = "prime_cli/core/client.py"
PRIME_CLIENT_OWNER_SHA256 = "bdfb0e6de11980c3c30d402b88828ecac4d5fb56c99ea27f5a09acd2a6b609c0"
PRIME_CONFIG_OWNER = "prime_cli/core/config.py"
PRIME_CONFIG_OWNER_SHA256 = "ec4b68730b1aafd9638ef076889ef3433719317c95da77fa4cfbdca1d3eaf90a"
PODS_CREATE_ENDPOINT = "https://api.primeintellect.ai/api/v1/pods"

OPENSSH_EXECUTABLES = {
    "ssh": {
        "path": r"C:\Windows\System32\OpenSSH\ssh.exe",
        "bytes": 1_253_888,
        "sha256": "8607ff933e769e77534b1244e39965bcf1c904dbfd4b9da819bbb71034cfef88",
    },
    "scp": {
        "path": r"C:\Windows\System32\OpenSSH\scp.exe",
        "bytes": 431_616,
        "sha256": "7758d689e2203c5e459fa5b8251f8a3ce27c3c8f0b5dcf6c2313909f25c2cb13",
    },
    "ssh-keyscan": {
        "path": r"C:\Windows\System32\OpenSSH\ssh-keyscan.exe",
        "bytes": 667_648,
        "sha256": "43ad579511e145036282f67783459906da4d58b23b46cfc62f1b9b35a8003d06",
    },
}
GIT_EXECUTABLE = {
    "path": r"C:\Program Files\Git\mingw64\bin\git.exe",
    "bytes": 4_149_624,
    "sha256": "51c6331aab2426ae2df187975590587b5a10042e3423f4bc0fdcb54aeb3efab7",
}
GIT_LAUNCHER = {
    "path": r"C:\Program Files\Git\cmd\git.exe",
    "bytes": 46_968,
    "sha256": "f668c4ba88417ecdf29470b3af92d576a701cc0f76dd083b13d032f4b3f1f247",
}

READINESS_AUTHORITY = {
    "capacity_observation_authorized": False,
    "model_calls_authorized": False,
    "prime_authorized": False,
    "provider_calls_authorized": False,
    "provisioning_authorized": False,
    "remote_tests_authorized": False,
    "science_authorized": False,
    "source_access_authorized": False,
    "training_campaign_authorized": False,
}
RUNTIME_AUTHORITY = {
    **READINESS_AUTHORITY,
    "capacity_observation_authorized": True,
    "prime_authorized": True,
    "provisioning_authorized": True,
    "remote_tests_authorized": True,
}


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True, slots=True)
class SigningIdentity:
    principal: str
    key_type: str
    public_key_base64: str
    fingerprint_sha256: str
    allowed_signers_sha256: str

    @property
    def public_key(self) -> bytes:
        return f"{self.key_type} {self.public_key_base64}\n".encode()

    def sanitized(self) -> dict[str, str]:
        return {
            "principal": self.principal,
            "key_type": self.key_type,
            "fingerprint_sha256": self.fingerprint_sha256,
            "allowed_signers_sha256": self.allowed_signers_sha256,
        }


class _WalletClient(Protocol):
    @property
    def client(self) -> Any: ...


class WalletRuntime(Protocol):
    @property
    def client(self) -> _WalletClient: ...

    @property
    def wallet_team_id(self) -> str | None: ...

    @property
    def transport_errors(self) -> tuple[type[BaseException], ...]: ...


class WalletOwner(Protocol):
    wallet_api_calls: int

    @property
    def context(self) -> WalletRuntime: ...

    @property
    def create_dispatch_epoch(self) -> int | None: ...

    @property
    def known_pod_ids(self) -> set[str]: ...

    def remaining(self, *, cleanup: bool) -> float: ...
    def journal(self, phase: str, operation: str, details: Mapping[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class CreateDispatchSummary:
    resource_sha256: str
    payload_sha256: str


@dataclass(frozen=True, slots=True)
class CreateResultSummary:
    status_code: int
    response_sha256: str
    response_bytes: int
    pod_identity_sha256: str
    pod_name: str


@dataclass(frozen=True, slots=True)
class CommandJournalSummary:
    command_count: int
    prime_cli_call_count: int
    wallet_api_call_count: int
    create_payload_sha256: str | None
    create_status_code: int | None
    create_pod_identity_sha256: str | None
    create_response_sha256: str | None
    create_response_bytes: int | None
    wallet_outcomes: tuple[Mapping[str, object], ...]
    ssh_keyscan_dispatch_ordinals: tuple[int, ...]
    ssh_keyscan_outcome_ordinals: tuple[int, ...]
    ssh_keyscan_stdout_sha256s: tuple[str, ...]

ALLOWED_GPU_LABELS = ("L40 48GB", "L40S 48GB", "RTX6000Ada 48GB")
MAXIMUM_RATE_USD = 2.0
WALLET_MINIMUM_USD = 30.0
SUPPORT_CAP_USD = 12.0
RESERVE_USD = 18.0
MAXIMUM_POD_SECONDS = 21_600
CLEANUP_TIMEOUT_SECONDS = 900
CLEANUP_POD_TIMEOUT_SECONDS = 300
CLEANUP_DISK_TIMEOUT_SECONDS = 120
CLEANUP_BILLING_TIMEOUT_SECONDS = 480
ASSESSMENT_TTL_SECONDS = 900
COMMAND_TIMEOUT_SECONDS = 120
KEYSCAN_TIMEOUT_SECONDS = 30
TRANSFER_TIMEOUT_SECONDS = 900
REMOTE_TIMEOUT_SECONDS = 3_600
POLL_TIMEOUT_SECONDS = 900
POLL_INTERVAL_SECONDS = 5
MAX_STATUS_POLLS = POLL_TIMEOUT_SECONDS // POLL_INTERVAL_SECONDS
MAX_TERMINATION_POLLS = MAX_STATUS_POLLS
MAX_BILLING_POLLS = 12
WALLET_PAGE_LIMIT = 100
MAX_PREEXISTING_WALLET_ROWS = 4_096
MAX_NEW_BILLING_ROWS = 4_096
MAX_POST_WALLET_ROWS = MAX_PREEXISTING_WALLET_ROWS + MAX_NEW_BILLING_ROWS
MAX_PRE_WALLET_PAGES = 41
MAX_POST_WALLET_PAGES = 82
MAX_PRE_WALLET_REQUESTS = MAX_PRE_WALLET_PAGES + 1
MAX_POST_WALLET_REQUESTS = MAX_POST_WALLET_PAGES + 1
MAX_WALLET_API_CALLS = MAX_PRE_WALLET_REQUESTS + (MAX_POST_WALLET_REQUESTS * MAX_BILLING_POLLS)
BILLING_RESOURCE_TYPE = "pod"
BILLING_RESOURCE_ID_NULL_ALLOWED = False
MAX_RECONCILIATION_MATCHES = 8
MAX_OWNED_POD_IDENTITIES = 1 + MAX_RECONCILIATION_MATCHES
MAX_OPERATIONAL_PRIME_CLI_CALLS = (
    2  # pre-create pods and disks; wallet uses the authenticated API owner
    + MAX_STATUS_POLLS  # create reconciliation
    + MAX_STATUS_POLLS  # readiness status
)
MAX_CLEANUP_PRIME_CLI_CALLS = (
    MAX_TERMINATION_POLLS  # cleanup inventory and late adoption
    + MAX_OWNED_POD_IDENTITIES  # trusted response plus eight late identities
    + 1  # final disk inventory
)
MAX_PRIME_CLI_CALLS = MAX_OPERATIONAL_PRIME_CLI_CALLS + MAX_CLEANUP_PRIME_CLI_CALLS
HANDOFF_SIGN_TIMEOUT_SECONDS = 30
TERMINAL_SIGN_TIMEOUT_SECONDS = 30

GPU_TELEMETRY_BINDING = {
    "L40S": {
        "path": "runs/stage-c2/warmstart-audit-v2/resource-before.csv",
        "bytes": 144,
        "sha256": "0807ed6daf0c2b5191e3616bb4508b97725c1b5ac1373711c7b4ef88b95173b4",
        "gpu_name": "NVIDIA L40S",
        "device_count": 2,
        "memory_total_mib_per_device": 46_068,
        "cuda_visible_min_mib_per_device": 45_000,
        "cuda_visible_max_mib_per_device": 46_068,
        "evidence_kind": "repository_telemetry",
    },
    "RTX6000Ada": {
        "path": "runs/stage-c4/warmstart-selection-v4/resource-before.csv",
        "bytes": 182,
        "sha256": "045cdaabb1cae91d9cacdec1c5bd42c669625a7c91d7808cf0950be4b6794ba6",
        "gpu_name": "NVIDIA RTX 6000 Ada Generation",
        "device_count": 2,
        "memory_total_mib_per_device": 49_140,
        "cuda_visible_min_mib_per_device": 48_000,
        "cuda_visible_max_mib_per_device": 49_140,
        "evidence_kind": "repository_telemetry",
    },
    "L40": {
        "gpu_name": "NVIDIA L40",
        "device_count": 2,
        "cuda_visible_min_mib_per_device": 45_000,
        "cuda_visible_max_mib_per_device": 46_068,
        "evidence_kind": "conservative_l40s_48gb_class_bound",
        "bound_source": "L40S",
    },
}

TEST_NODES = (
    "tests/test_stage_d_objective_binding_prime.py::"
    "test_pinned_prime_materializes_actual_branch_objective",
    "tests/test_stage_d_training_bridge.py::"
    "test_actual_prime_msgpack_packer_and_clean_loss_match_manual_formula",
    "tests/test_stage_d_three_arm_bridge.py::"
    "test_actual_prime_packer_losses_and_gradients_match_independent_objectives",
    "tests/test_stage_d_prime_trainer.py::"
    "test_runtime_gate_requires_exactly_one_consumed_batch_before_exit",
    "tests/test_stage_d_live_update_torch.py::"
    "test_real_torch_gate_hashes_one_adamw_step_and_saved_adapter",
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def authority_value(value: object, expected: Mapping[str, bool], label: str) -> None:
    if type(value) is not dict or set(cast(dict[object, object], value)) != set(expected):
        raise ValueError(f"Prime one-shot {label} authority differs")
    authority = cast(dict[str, object], value)
    if any(
        type(authority[key]) is not bool or authority[key] is not expected[key]
        for key in expected
    ):
        raise ValueError(f"Prime one-shot {label} authority differs")


def authenticate_git_executable() -> Path:
    harmless = {"GIT_FLUSH", "GIT_OPTIONAL_LOCKS", "GIT_PAGER", "GIT_TERMINAL_PROMPT"}
    if any(key.startswith("GIT_") and key not in harmless for key in os.environ):
        raise ValueError("Prime one-shot Git environment redirects are forbidden")
    for label, binding in (("launcher", GIT_LAUNCHER), ("owner", GIT_EXECUTABLE)):
        path = Path(cast(str, binding["path"]))
        info = path.lstat()
        reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        if (
            path.is_symlink()
            or getattr(info, "st_file_attributes", 0) & reparse
            or not stat.S_ISREG(info.st_mode)
            or info.st_size != binding["bytes"]
            or sha256_bytes(path.read_bytes()) != binding["sha256"]
        ):
            raise ValueError(f"Prime one-shot Git {label} differs")
    return Path(cast(str, GIT_EXECUTABLE["path"]))


def _git_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    return environment


def _authenticate_git_metadata(root: Path) -> None:
    marker = root / ".git"
    marker_info = marker.lstat()
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if marker.is_symlink() or getattr(marker_info, "st_file_attributes", 0) & reparse:
        raise ValueError("Prime one-shot Git metadata alias is forbidden")
    if stat.S_ISDIR(marker_info.st_mode):
        git_dir = marker
    elif stat.S_ISREG(marker_info.st_mode) and marker_info.st_nlink == 1:
        raw = marker.read_text(encoding="utf-8").strip()
        if not raw.startswith("gitdir: "):
            raise ValueError("Prime one-shot Git metadata file differs")
        git_dir = (root / raw.removeprefix("gitdir: ")).resolve()
        info = git_dir.lstat()
        if git_dir.is_symlink() or getattr(info, "st_file_attributes", 0) & reparse:
            raise ValueError("Prime one-shot Git metadata target is aliased")
    else:
        raise ValueError("Prime one-shot Git metadata differs")
    for relative in (
        "info/grafts",
        "shallow",
        "objects/info/alternates",
        "objects/info/http-alternates",
    ):
        candidate = git_dir / relative
        if candidate.exists() or candidate.is_symlink():
            raise ValueError("Prime one-shot Git object substitution is forbidden")


def git_output(root: Path, *arguments: str) -> str:
    executable = authenticate_git_executable()
    _authenticate_git_metadata(root)
    return subprocess.run(
        (
            str(executable),
            "--no-replace-objects",
            "-c",
            f"safe.directory={root.resolve()}",
            "-C",
            str(root),
            *arguments,
        ),
        check=True,
        capture_output=True,
        text=True,
        env=_git_environment(),
    ).stdout.strip()


def _status(root: Path) -> set[tuple[str, str]]:
    executable = authenticate_git_executable()
    _authenticate_git_metadata(root)
    raw = subprocess.run(
        (
            str(executable),
            "--no-replace-objects",
            "-c",
            f"safe.directory={root.resolve()}",
            "-C",
            str(root),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ),
        check=True,
        capture_output=True,
        env=_git_environment(),
    ).stdout
    records: set[tuple[str, str]] = set()
    for item in raw.split(b"\0"):
        if not item:
            continue
        if len(item) < 4 or item[2:3] != b" ":
            raise ValueError("Prime one-shot Git status is malformed")
        records.add((item[:2].decode("ascii"), item[3:].decode("utf-8")))
    return records


def _external_binding(root: Path) -> None:
    entry = git_output(root, "ls-files", "-s", EXTERNAL_GITLINK).split()
    if entry[:3] != ["160000", EXTERNAL_GITLINK_OBJECT, "0"]:
        raise ValueError("Prime one-shot external gitlink differs")
    if git_output(root / EXTERNAL_GITLINK, "rev-parse", "HEAD") != EXTERNAL_GITLINK_OBJECT:
        raise ValueError("Prime one-shot external checkout differs")


def _authenticate_gpu_telemetry(root: Path) -> None:
    for label in ("L40S", "RTX6000Ada"):
        binding = GPU_TELEMETRY_BINDING[label]
        path = root / cast(str, binding["path"])
        info = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != binding["bytes"]
            or sha256_bytes(path.read_bytes()) != binding["sha256"]
        ):
            raise ValueError("Prime one-shot GPU telemetry differs")


def authenticate_readiness(root: Path, *, committed: bool) -> dict[str, str]:
    root = root.resolve()
    _authenticate_gpu_telemetry(root)
    _external_binding(root)
    parent_tree = git_output(root, "rev-parse", f"{IMPLEMENTATION_PARENT}^{{tree}}")
    if parent_tree != IMPLEMENTATION_PARENT_TREE:
        raise ValueError("Prime one-shot implementation parent differs")
    head = git_output(root, "rev-parse", "HEAD")
    allowed = {(" M", EXTERNAL_GITLINK)}
    if committed:
        lineage = git_output(root, "rev-list", "--parents", "-n", "1", head).split()
        if len(lineage) != 2 or lineage[1] != IMPLEMENTATION_PARENT:
            raise ValueError("Prime one-shot readiness must directly follow e59fe73")
        changes = set(
            git_output(
                root, "diff", "--name-status", "--no-renames", IMPLEMENTATION_PARENT, head
            ).splitlines()
        )
        if changes != {f"A\t{path}" for path in READINESS_PATHS}:
            raise ValueError("Prime one-shot readiness diff is not exact")
    else:
        if head != IMPLEMENTATION_PARENT:
            raise ValueError("Prime one-shot readiness build requires e59fe73")
        artifact_presence = {
            path: (root / path).is_file() and not (root / path).is_symlink()
            for path in (CONTRACT_PATH, AUDIT_PATH)
        }
        if len(set(artifact_presence.values())) != 1:
            raise ValueError("Prime one-shot readiness artifacts are partially present")
        present_paths = (
            READINESS_PATHS
            if all(artifact_presence.values())
            else READINESS_PATHS - {CONTRACT_PATH, AUDIT_PATH}
        )
        allowed |= {("??", path) for path in present_paths}
    if _status(root) != allowed:
        raise ValueError("Prime one-shot readiness worktree is not exact")
    return {"commit": head, "tree": git_output(root, "show", "-s", "--format=%T", head)}


def _readiness_commit_from_authorization(root: Path, *, current_is_authorization: bool) -> str:
    head = git_output(root, "rev-parse", "HEAD")
    return git_output(root, "rev-parse", "HEAD^") if current_is_authorization else head


def authorization_value(root: Path, *, current_is_authorization: bool) -> dict[str, object]:
    readiness_commit = _readiness_commit_from_authorization(
        root, current_is_authorization=current_is_authorization
    )
    lineage = git_output(root, "rev-list", "--parents", "-n", "1", readiness_commit).split()
    if len(lineage) != 2 or lineage[1] != IMPLEMENTATION_PARENT:
        raise ValueError("Prime one-shot authorization has the wrong readiness parent")
    changes = set(
        git_output(
            root,
            "diff",
            "--name-status",
            "--no-renames",
            IMPLEMENTATION_PARENT,
            readiness_commit,
        ).splitlines()
    )
    if changes != {f"A\t{path}" for path in READINESS_PATHS}:
        raise ValueError("Prime one-shot authorization readiness diff differs")
    contract = (root / CONTRACT_PATH).read_bytes()
    if contract != build_readiness_artifacts(root)[CONTRACT_PATH]:
        raise ValueError("Prime one-shot readiness contract differs")
    return {
        "schema_version": 2,
        "domain": AUTHORIZATION_DOMAIN,
        "state": "one_shot_test_only_authorized",
        "readiness": {
            "commit": readiness_commit,
            "tree": git_output(root, "show", "-s", "--format=%T", readiness_commit),
            "contract_path": CONTRACT_PATH,
            "contract_sha256": sha256_bytes(contract),
        },
        "user": {
            "original_text": ORIGINAL_AUTHORIZATION,
            "original_bytes": ORIGINAL_AUTHORIZATION_BYTES,
            "original_sha256": ORIGINAL_AUTHORIZATION_SHA256,
            "clarification_text": CLARIFICATION,
            "clarification_bytes": CLARIFICATION_BYTES,
            "clarification_sha256": CLARIFICATION_SHA256,
            "thread_id": ORCHESTRATOR_THREAD,
        },
        "scope": {
            "availability_attempt_limit": 1,
            "provisioning_dispatch_limit": 1,
            "monitoring": False,
            "retry": False,
            "same_pod_full_test_plan": True,
            "support_cap_usd": SUPPORT_CAP_USD,
            "reserve_usd": RESERVE_USD,
            "authority": RUNTIME_AUTHORITY,
        },
    }


def authenticate_authorization(root: Path) -> dict[str, object]:
    root = root.resolve()
    head = git_output(root, "rev-parse", "HEAD")
    lineage = git_output(root, "rev-list", "--parents", "-n", "1", head).split()
    if len(lineage) != 2:
        raise ValueError("Prime one-shot authorization must have one parent")
    readiness_commit = lineage[1]
    expected = canonical_json(authorization_value(root, current_is_authorization=True))
    actual = (root / AUTHORIZATION_PATH).read_bytes()
    if actual != expected:
        raise ValueError("Prime one-shot authorization bytes differ")
    changes = set(
        git_output(
            root, "diff", "--name-status", "--no-renames", readiness_commit, head
        ).splitlines()
    )
    if changes != {f"A\t{AUTHORIZATION_PATH}"}:
        raise ValueError("Prime one-shot authorization is not the sole-path child")
    if _status(root) != {(" M", EXTERNAL_GITLINK)}:
        raise ValueError("Prime one-shot authorization checkout is not exact")
    return {
        "commit": head,
        "tree": git_output(root, "show", "-s", "--format=%T", head),
        "parent": readiness_commit,
        "authorization_path": AUTHORIZATION_PATH,
        "authorization_sha256": sha256_bytes(actual),
        "authorization_blob": git_output(root, "rev-parse", f"HEAD:{AUTHORIZATION_PATH}"),
    }


def _verify_texts() -> None:
    bindings = (
        (ORIGINAL_AUTHORIZATION, ORIGINAL_AUTHORIZATION_BYTES, ORIGINAL_AUTHORIZATION_SHA256),
        (CLARIFICATION, CLARIFICATION_BYTES, CLARIFICATION_SHA256),
    )
    for text, length, digest in bindings:
        raw = text.encode("utf-8")
        if len(raw) != length or sha256_bytes(raw) != digest:
            raise ValueError("Prime one-shot user text differs")


def build_readiness_artifacts(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    _verify_texts()
    source_paths = sorted(READINESS_PATHS - {CONTRACT_PATH, AUDIT_PATH})
    bindings = {
        path: {
            "bytes": (root / path).stat().st_size,
            "sha256": sha256_bytes((root / path).read_bytes()),
        }
        for path in source_paths
    }
    contract = canonical_json(
        {
            "schema_version": 2,
            "domain": READINESS_DOMAIN,
            "state": "non_authorizing_readiness",
            "parent": {"commit": IMPLEMENTATION_PARENT, "tree": IMPLEMENTATION_PARENT_TREE},
            "readiness_paths": sorted(READINESS_PATHS),
            "future_authorization_path": AUTHORIZATION_PATH,
            "sequence": [
                "readiness direct child of implementation parent",
                "authorization sole-path direct child of readiness",
                "one signed observation and at most one create dispatch",
                "pod-bound signed one-use handoff before SSH",
                "same-pod complete reviewed test plan",
                "evidence recovery, billing reconciliation, exhaustive teardown, terminal",
            ],
            "user_text": {
                "original_sha256": ORIGINAL_AUTHORIZATION_SHA256,
                "clarification_sha256": CLARIFICATION_SHA256,
            },
            "prime": {
                "v5_owner": {"path": V5_OWNER_PATH, "sha256": V5_OWNER_SHA256},
                "v5_contract": {"path": V5_CONTRACT_PATH, "sha256": V5_CONTRACT_SHA256},
                "pods_api_owner": {"path": PODS_API_OWNER, "sha256": PODS_API_OWNER_SHA256},
                "pods_command_owner": {
                    "path": PODS_COMMAND_OWNER,
                    "sha256": PODS_COMMAND_OWNER_SHA256,
                    "cli_create_forbidden": True,
                    "reason": "CLI refetches availability and replaces zero disk with default",
                },
                "create_endpoint": PODS_CREATE_ENDPOINT,
                "availability_attempts": 1,
                "create_dispatches": 1,
                "retries": 0,
            },
            "limits": {
                "assessment_ttl_seconds": ASSESSMENT_TTL_SECONDS,
                "maximum_pod_seconds": MAXIMUM_POD_SECONDS,
                "cleanup_timeout_seconds": CLEANUP_TIMEOUT_SECONDS,
                "cleanup_phase_timeout_seconds": {
                    "pods": CLEANUP_POD_TIMEOUT_SECONDS,
                    "disks": CLEANUP_DISK_TIMEOUT_SECONDS,
                    "billing": CLEANUP_BILLING_TIMEOUT_SECONDS,
                },
                "maximum_rate_usd": MAXIMUM_RATE_USD,
                "support_cap_usd": SUPPORT_CAP_USD,
                "reserve_usd": RESERVE_USD,
                "wallet_minimum_usd": WALLET_MINIMUM_USD,
                "maximum_prime_cli_calls": MAX_PRIME_CLI_CALLS,
                "maximum_operational_prime_cli_calls": MAX_OPERATIONAL_PRIME_CLI_CALLS,
                "maximum_cleanup_prime_cli_calls": MAX_CLEANUP_PRIME_CLI_CALLS,
                "maximum_preexisting_wallet_rows": MAX_PREEXISTING_WALLET_ROWS,
                "maximum_new_billing_rows": MAX_NEW_BILLING_ROWS,
                "maximum_post_wallet_rows": MAX_POST_WALLET_ROWS,
                "wallet_page_limit": WALLET_PAGE_LIMIT,
                "maximum_pre_wallet_pages": MAX_PRE_WALLET_PAGES,
                "maximum_post_wallet_pages": MAX_POST_WALLET_PAGES,
                "maximum_pre_wallet_requests": MAX_PRE_WALLET_REQUESTS,
                "maximum_post_wallet_requests": MAX_POST_WALLET_REQUESTS,
                "maximum_wallet_api_calls": MAX_WALLET_API_CALLS,
                "billing_reconciliation": {
                    "maximum_new_rows": MAX_NEW_BILLING_ROWS,
                    "minimum_new_rows": 1,
                    "resource_type": BILLING_RESOURCE_TYPE,
                    "resource_id_null_allowed": BILLING_RESOURCE_ID_NULL_ALLOWED,
                    "cardinality_law": "complete_paginated_pre_and_post_set_difference",
                    "stability_law": "reread_page_zero_after_complete_snapshot",
                    "historical_law": (
                        "every prior identity and exact canonical row must recur once in the "
                        "same relative order"
                    ),
                    "headroom_law": (
                        "protocol bound: at most 4096 prior rows reserves at most 4096 new rows; "
                        "not a provider billing-cadence claim"
                    ),
                    "journal_privacy_law": (
                        "durable wallet outcomes contain only request facts, body commitments, "
                        "hashed wallet/team/row identities, canonical semantic-row hashes, "
                        "currency, balance, and counts; raw wallet, team, billing, resource, and "
                        "provider identifiers remain in authenticated process memory"
                    ),
                    "snapshot_domain": WALLET_SNAPSHOT_DOMAIN,
                    "row_domain": WALLET_ROW_DOMAIN,
                    "reconciliation_domain": WALLET_RECONCILIATION_DOMAIN,
                    "terminal_replay_law": (
                        "the signed terminal verifier recomputes each strict sanitized semantic "
                        "row commitment and replays exact historical order, page offsets and "
                        "cardinalities, new owned-pod rows, timestamps, totals, balance delta, "
                        "cap, and reserve; raw page hashes remain separate capture commitments"
                    ),
                },
                "terminal_evidence": {
                    "domain": TERMINAL_DOMAIN,
                    "purpose": TERMINAL_PURPOSE,
                    "authority": READINESS_AUTHORITY,
                    "transcript_law": (
                        "strict canonical v5 replay plus exact transcript-payload hash and "
                        "regenerated assessment-byte equality"
                    ),
                    "journal_law": (
                        "strict operation-specific details, contiguous dispatch/outcome pairing, "
                        "bounded cardinality, create payload/response/pod summaries, exact wallet "
                        "page-outcome summaries, and raw-identifier privacy"
                    ),
                    "create_law": (
                        "Prime owner validates exact canonical all-false create-dispatch and "
                        "create-result schemas and cross-binds selected resource, request payload, "
                        "response bytes, status, pod identity, cleanup ownership, and billing"
                    ),
                    "handoff_law": (
                        "Remote owner alone constructs and validates the exact all-false pod-bound "
                        "one-use handoff, canonical known-hosts bytes, authorization and evidence "
                        "lineage, selected resource, pod, SSH endpoint, runtime, TTL, nonce, and "
                        "detached signature-file equality"
                    ),
                    "wallet_before_law": (
                        "every present precreate snapshot is canonical and replayed against exact "
                        "journaled page outcomes even when postcleanup wallet evidence is absent; "
                        "create dispatch requires it and capacity-only states forbid it"
                    ),
                    "law": (
                        "canonical closed schema, exact disposition-state relationship, strict "
                        "claim and assessment lineage, recomputed counts and evidence DAG, and "
                        "state-dependent success/failure closure; failed_terminal requires an "
                        "authenticated failure and cannot omit mandatory post-dispatch cleanup or "
                        "successful-test remote, JUnit, GPU, and status evidence"
                    ),
                },
                "signing_timeouts_seconds": {
                    "handoff": HANDOFF_SIGN_TIMEOUT_SECONDS,
                    "terminal": TERMINAL_SIGN_TIMEOUT_SECONDS,
                },
                "maximum_status_polls": MAX_STATUS_POLLS,
                "maximum_termination_polls": MAX_TERMINATION_POLLS,
            },
            "openssh": OPENSSH_EXECUTABLES,
            "git": {"execution_owner": GIT_EXECUTABLE, "launcher": GIT_LAUNCHER},
            "gpu_telemetry": GPU_TELEMETRY_BINDING,
            "wallet_api_owner": {
                "path": WALLET_API_OWNER,
                "sha256": WALLET_API_OWNER_SHA256,
                "endpoint": WALLET_API_ENDPOINT,
                "transport": "authenticated APIClient.httpx.Client direct request",
                "client_owner": {
                    "path": PRIME_CLIENT_OWNER,
                    "sha256": PRIME_CLIENT_OWNER_SHA256,
                },
                "credentials_owner": {
                    "path": PRIME_CONFIG_OWNER,
                    "sha256": PRIME_CONFIG_OWNER_SHA256,
                },
            },
            "test_nodes": list(TEST_NODES),
            "authority": READINESS_AUTHORITY,
        }
    )
    audit = canonical_json(
        {
            "schema_version": 2,
            "domain": READINESS_AUDIT_DOMAIN,
            "state": "cpu_only_not_executed",
            "parent": {"commit": IMPLEMENTATION_PARENT, "tree": IMPLEMENTATION_PARENT_TREE},
            "contract": {
                "path": CONTRACT_PATH,
                "bytes": len(contract),
                "sha256": sha256_bytes(contract),
            },
            "file_bindings": bindings,
            "future_authorization_present": False,
            "live_activity": {
                "prime_calls": 0,
                "network_calls": 0,
                "provisioning_calls": 0,
                "provider_calls": 0,
                "model_calls": 0,
                "source_reads": 0,
            },
            "authority": READINESS_AUTHORITY,
        }
    )
    return {CONTRACT_PATH: contract, AUDIT_PATH: audit}


def verify_readiness_artifacts(root: Path) -> dict[str, str]:
    expected = build_readiness_artifacts(root)
    result: dict[str, str] = {}
    for relative, raw in expected.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise ValueError(f"Prime one-shot readiness artifact differs: {relative}")
        result[relative] = sha256_bytes(raw)
    return result


def strict_object(raw: bytes, keys: set[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not JSON") from error
    if not isinstance(value, dict) or set(value) != keys or canonical_json(value) != raw:
        raise ValueError(f"{label} schema or canonical bytes differ")
    return cast(dict[str, Any], value)


def safe_runtime_root(repository: Path) -> Path:
    repository = repository.resolve()
    target = repository / EVIDENCE_ROOT
    current = target.parent
    while True:
        if current.exists():
            info = current.lstat()
            attributes = getattr(info, "st_file_attributes", 0)
            reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if current.is_symlink() or attributes & reparse or not stat.S_ISDIR(info.st_mode):
                raise ValueError("Prime one-shot evidence ancestor is unsafe")
        if current == repository:
            break
        if repository not in current.parents:
            raise ValueError("Prime one-shot evidence root escapes repository")
        current = current.parent
    if target.exists() or target.is_symlink():
        raise FileExistsError("Prime one-shot evidence root already exists")
    return target


def exclusive_runtime_root(repository: Path) -> Path:
    target = safe_runtime_root(repository)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(target, 0o700)
    info = target.lstat()
    if target.is_symlink() or info.st_nlink != 1 or not stat.S_ISDIR(info.st_mode):
        raise ValueError("Prime one-shot evidence root creation is unsafe")
    return target


def fixed_runtime_path(root: Path, name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)?", name):
        raise ValueError("Prime one-shot evidence name differs")
    path = root / name
    if path.parent != root or path.exists() or path.is_symlink():
        raise FileExistsError("Prime one-shot evidence path is not fresh")
    return path


def publish_once(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.urandom(16).hex()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        if os.name != "nt":
            parent = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ARTIFACT_FILENAMES",
    "ASSESSMENT_DOMAIN",
    "ASSESSMENT_NAMESPACE",
    "ASSESSMENT_TTL_SECONDS",
    "AUTHORIZATION_PATH",
    "CLAIM_DOMAIN",
    "CONTRACT_PATH",
    "GIT_EXECUTABLE",
    "GIT_LAUNCHER",
    "GPU_TELEMETRY_BINDING",
    "HANDOFF_DOMAIN",
    "HANDOFF_NAMESPACE",
    "MAXIMUM_POD_SECONDS",
    "MAX_COMMAND_OUTPUT_BYTES",
    "POD_NAME_PREFIX",
    "READINESS_AUTHORITY",
    "READINESS_PATHS",
    "REMOTE_TIMEOUT_SECONDS",
    "RUNTIME_AUTHORITY",
    "SIGNED_ENVELOPE_DOMAIN",
    "SUPPORT_CAP_USD",
    "TERMINAL_DOMAIN",
    "TERMINAL_NAMESPACE",
    "TERMINAL_PURPOSE",
    "TEST_NODES",
    "WALLET_MINIMUM_USD",
    "WALLET_ROW_DOMAIN",
    "CommandJournalSummary",
    "CommandResult",
    "CreateDispatchSummary",
    "CreateResultSummary",
    "SigningIdentity",
    "WalletOwner",
    "WalletRuntime",
    "authenticate_authorization",
    "authenticate_git_executable",
    "authenticate_readiness",
    "authority_value",
    "authorization_value",
    "build_readiness_artifacts",
    "canonical_json",
    "exclusive_runtime_root",
    "fixed_runtime_path",
    "git_output",
    "publish_once",
    "safe_runtime_root",
    "sha256_bytes",
    "strict_object",
    "verify_readiness_artifacts",
]
