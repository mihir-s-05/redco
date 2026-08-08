"""Hash-bound local Prime/SSH lifecycle owner for the support attempt.

The script is an operator-only post-review entry point.  It is not invoked by
CPU verification and accepts no scientific or resource-selection overrides.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import posixpath
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_stage_d_v13_remote_bootstrap import required_asset_mappings  # noqa: E402

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes  # noqa: E402
from redco.analysis.stage_d_v13_draft_publication import (  # noqa: E402
    atomic_publish_set,
    validate_output_paths,
)
from redco.analysis.stage_d_v13_launch_lifecycle import (  # noqa: E402
    ProvisioningLedger,
    issue_execute_handoff_v2,
    validate_signing_key,
)
from redco.analysis.stage_d_v13_launch_observations import (  # noqa: E402
    validate_current_prime_observation,
    validate_prime_observation,
)
from redco.analysis.stage_d_v13_support_launch import (  # noqa: E402
    LAUNCH_HANDOFF_RELATIVE,
    LAUNCH_HANDOFF_SIGNATURE_RELATIVE,
    LAUNCH_KNOWN_HOSTS_RELATIVE,
    LAUNCH_POD_OBSERVATION_RELATIVE,
    LAUNCH_PRIME_OBSERVATION_RELATIVE,
    LAUNCH_PROVISIONING_LEDGER_RELATIVE,
    launch_signing_identity,
)
from redco.analysis.stage_d_v13_support_launch_runtime import (  # noqa: E402
    terminal_payload,
)

REPOSITORY_URL = "https://github.com/mihir-s-05/redco.git"
POD_NAME = "redco-stage-d1-support-v13"
POD_IMAGE = "ubuntu_22_cuda_12"
POD_DISK_SIZE_GB = "0"
POD_STATUS_TIMEOUT_SECONDS = 900
_REPARSE_POINT = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _run(argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, cwd=cwd, check=True, capture_output=True)


def _is_link_or_reparse(path: Path) -> bool:
    """Inspect a path without resolving it first.

    ``Path.is_symlink`` is not sufficient on Windows because junctions and
    other reparse points can be followed by ``resolve``.  Keep this check
    lstat-based so operator input cannot escape its approved root before the
    containment check below.
    """

    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _operator_path(
    path: Path,
    label: str,
    *,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    """Authenticate an operator path before resolving or using it."""

    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if _is_link_or_reparse(candidate):
            raise ValueError(f"{label} root or ancestor is linked/reparse")
    resolved = absolute.resolve(strict=False)
    if require_file and not resolved.is_file():
        raise ValueError(f"{label} is missing or not a file")
    if require_directory and not resolved.is_dir():
        raise ValueError(f"{label} is missing or not a directory")
    return resolved


class LocalLaunchOrchestrator:
    """Own one verified pod and always recover before termination."""

    def __init__(
        self,
        repository: Path,
        observation: Path,
        ssh_key: Path,
        artifact_root: Path,
    ) -> None:
        self.repository = _operator_path(repository, "repository")
        self.observation = _operator_path(observation, "Prime observation", require_file=True)
        self.ssh_key = _operator_path(ssh_key, "operator SSH key", require_file=True)
        self.artifact_root = _operator_path(
            artifact_root,
            "approved local asset root",
            require_directory=True,
        )
        self.signer = launch_signing_identity()
        validate_signing_key(self.ssh_key, self.signer)
        try:
            self.observation.relative_to(self.repository)
        except ValueError as error:
            raise ValueError("Prime observation must be inside the repository") from error
        if self.observation.relative_to(self.repository).as_posix() != (
            LAUNCH_PRIME_OBSERVATION_RELATIVE
        ):
            raise ValueError("Prime execution requires the fixed observation handoff path")
        self.facts = validate_prime_observation(self.repository, self.observation)
        self.observation_sha256 = sha256_bytes(self.observation.read_bytes())
        resource = facts_resource(self.facts)
        self.resource_id = resource["resource_id"]
        bundle = self.facts["bundle"]
        if not isinstance(bundle, dict) or not isinstance(bundle.get("commit"), str):
            raise ValueError("Prime observation lacks the authenticated bundle commit")
        self.bundle_commit = bundle["commit"]
        self.host: str | None = None
        self.ssh_user: str | None = None
        self.ssh_port = 22
        self.pod_id: str | None = None
        self._pod_created_at_epoch: int | None = None
        self._wallet_after: dict[str, Any] | None = None
        self._ledger: ProvisioningLedger | None = None
        self._pod_status: dict[str, Any] | None = None
        self._asset_sources: tuple[tuple[str, Path, str, str], ...] | None = None
        self.pod_name = POD_NAME
        self.known_hosts_path = self.repository / LAUNCH_KNOWN_HOSTS_RELATIVE
        self.known_hosts_fingerprints: tuple[str, ...] = ()
        self.known_hosts_sha256 = ""
        self.remote_root = "/workspace/redco"

    def _prime_json(self, *args: str) -> dict[str, Any]:
        result = _run(["prime", "--plain", *args, "--output", "json"])
        try:
            value = json.loads(result.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Prime lifecycle output is not JSON") from error
        if not isinstance(value, dict):
            raise RuntimeError("Prime lifecycle output is not an object")
        return value

    def _prime_raw(self, *args: str) -> bytes:
        return _run(["prime", "--plain", *args]).stdout

    def _prime(self, *args: str) -> dict[str, Any]:
        """Compatibility alias for read-only JSON lifecycle calls."""

        return self._prime_json(*args)

    def run(self, *, remote_root: str) -> None:
        if remote_root != "/workspace/redco":
            raise ValueError("remote root must be /workspace/redco")
        self.remote_root = remote_root
        lifecycle_path = self.repository / (
            "reports/stage-d1-support-v13-launch-lifecycle-v1.json"
        )
        if lifecycle_path.exists() or lifecycle_path.is_symlink():
            raise RuntimeError("launch lifecycle is already terminal")
        self.facts = validate_current_prime_observation(self.repository, self.observation)
        resource = facts_resource(self.facts)
        self.resource_id = resource["resource_id"]
        from redco.analysis import stage_d_v13_support_launch as launch

        validate_output_paths(
            self.repository,
            launch._bundle_immutable_paths(self.repository),
            output_paths=(
                *launch.ATTEMPT_PATHS,
                launch.LAUNCH_RUNTIME_MANIFEST_RELATIVE,
                LAUNCH_POD_OBSERVATION_RELATIVE,
            ),
        )
        existing = self._prime_json("pods", "list")
        if not _inventory_is_empty(
            existing,
            "pods",
            {"id", "name", "gpu", "status", "created_at"},
        ):
            raise RuntimeError("duplicate pod inventory is not empty")
        disks = self._prime_json("disks", "list")
        if not _inventory_is_empty(
            disks,
            "disks",
            {"id", "name", "size", "status", "provider", "location", "created_at", "price_hr"},
        ):
            raise RuntimeError("persistent disk inventory is not empty")
        run_error: BaseException | None = None
        try:
            ledger_path = self.repository / LAUNCH_PROVISIONING_LEDGER_RELATIVE
            self._ledger = ProvisioningLedger.create(
                ledger_path,
                campaign_id=sha256_bytes(
                    canonical_json_bytes(
                        {
                            "bundle": self.facts["bundle"],
                            "observation_sha256": self.observation_sha256,
                            "campaign": "redco-stage-d1-support-v13",
                        }
                    )
                ),
                wallet_before=self.facts["wallet"],
            )
            provision_id = f"{self._ledger.campaign_id}:provision-1"
            self.pod_name = f"{POD_NAME}-{self._ledger.campaign_id[:12]}-p1"
            self._ledger.record_provision(
                provision_id=provision_id,
                resource_id=self.resource_id,
                billing_cursor=str(self.facts["wallet"]["billing_cursor"]),
            )
            self._asset_sources = self._verified_non_git_assets()
            created_stdout = self._prime_raw(
                "pods",
                "create",
                "--id",
                self.resource_id,
                "--name",
                self.pod_name,
                "--disk-size",
                POD_DISK_SIZE_GB,
                "--image",
                POD_IMAGE,
                "--yes",
            )
            self.pod_id = self._pod_id_from_inventory(created_stdout)
            self._pod_status = self._poll_pod_status()
            self.host = self._endpoint_from_status(self._pod_status)
            self.ssh_user, ssh_host, self.ssh_port = self._parse_ssh_endpoint(self.host)
            self.host = ssh_host
            self._ledger.bind_provision(provision_id, self.pod_id)
            self._pod_created_at_epoch = int(time.time())
            self._capture_campaign_known_hosts()
            status_sha256 = sha256_bytes(canonical_json_bytes(self._pod_status))
            from redco.analysis import stage_d_v13_support_launch as launch

            auth_path = self.repository / launch.LAUNCH_AUTH_RELATIVE
            authorization_sha256 = sha256_bytes(auth_path.read_bytes())
            resource_identity = {
                key: resource[key]
                for key in (
                    "resource_id",
                    "provider",
                    "location",
                    "gpu_type",
                    "gpu_count",
                    "memory_gb",
                    "is_spot",
                    "security",
                )
            }
            assert self._pod_status is not None
            assert self.pod_id is not None
            issue_execute_handoff_v2(
                self.repository / LAUNCH_HANDOFF_RELATIVE,
                self.repository / LAUNCH_HANDOFF_SIGNATURE_RELATIVE,
                bundle=self.facts["bundle"],
                launch_authorization_sha256=authorization_sha256,
                frozen_support_protocol_sha256=launch.PROTOCOL_ROOT_SHA256,
                prime_observation_sha256=self.observation_sha256,
                resource_identity=resource_identity,
                resource_price_usd=float(resource["hourly_rate_usd"]),
                pod_id=self.pod_id,
                pod_name=self.pod_name,
                pod_status_sha256=status_sha256,
                ssh={
                    "user": self.ssh_user or "",
                    "host": self.host or "",
                    "port": self.ssh_port,
                },
                known_hosts_sha256=self.known_hosts_sha256,
                known_hosts_fingerprints=self.known_hosts_fingerprints,
                ledger=self._ledger,
                signing_key=self.ssh_key,
                signer=self.signer,
                provisioning_ordinal=1,
            )
            self._ssh(remote_root)
        except BaseException as error:
            run_error = error
        finally:
            try:
                self._recover_and_terminate()
            except BaseException as error:
                if run_error is None:
                    run_error = error
            try:
                if self._ledger is not None:
                    self._ledger.close()
            except BaseException as error:
                if run_error is None:
                    run_error = error
            try:
                self._publish_lifecycle(run_error)
            except BaseException as error:
                if run_error is None:
                    run_error = error
        if run_error is not None:
            raise run_error

    def _capture_campaign_known_hosts(self) -> None:
        if self.host is None:
            raise RuntimeError("cannot capture known_hosts before SSH endpoint")
        scans: list[bytes] = []
        for _ in range(2):
            result = subprocess.run(
                ["ssh-keyscan", "-T", "5", "-p", str(self.ssh_port), self.host],
                check=False,
                capture_output=True,
            )
            if result.returncode != 0 or not result.stdout:
                raise RuntimeError("Prime SSH host-key scan failed")
            scans.append(bytes(result.stdout))
        if scans[0] != scans[1]:
            raise RuntimeError("Prime SSH host-key scans are not stable")
        raw = scans[0]
        fingerprint_result = subprocess.run(
            ["ssh-keygen", "-lf", "-", "-E", "sha256"],
            input=raw,
            check=True,
            capture_output=True,
        )
        fingerprints = sorted(
            {
                field
                for line in fingerprint_result.stdout.decode("ascii").splitlines()
                for field in line.split()
                if field.startswith("SHA256:")
            }
        )
        if not fingerprints:
            raise RuntimeError("Prime SSH host-key scan has no fingerprints")
        self.known_hosts_path.parent.mkdir(parents=True, exist_ok=True)
        if self.known_hosts_path.exists() or self.known_hosts_path.is_symlink():
            raise RuntimeError("campaign known_hosts already exists")
        descriptor = os.open(
            self.known_hosts_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
                descriptor = -1
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        self.known_hosts_fingerprints = tuple(fingerprints)
        self.known_hosts_sha256 = sha256_bytes(raw)

    def _pod_id_from_inventory(self, create_output: bytes) -> str:
        """Derive the created pod identity from supported list output only."""

        del create_output
        deadline = time.monotonic() + POD_STATUS_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            inventory = self._prime_json("pods", "list")
            _inventory_is_empty(
                inventory,
                "pods",
                {"id", "name", "gpu", "status", "created_at"},
            )
            pods = inventory["pods"]
            matches = [
                item
                for item in pods
                if isinstance(item, dict) and item.get("name") == self.pod_name
            ]
            if len(matches) == 1 and isinstance(matches[0].get("id"), str):
                return str(matches[0]["id"])
            if len(matches) > 1:
                raise RuntimeError("Prime created duplicate pods with the frozen name")
            time.sleep(1)
        raise TimeoutError("Prime did not expose the created pod in list output")

    def _poll_pod_status(self) -> dict[str, Any]:
        if self.pod_id is None:
            raise RuntimeError("pod status requested before pod identity")
        deadline = time.monotonic() + POD_STATUS_TIMEOUT_SECONDS
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            result = self._prime_json("pods", "status", self.pod_id)
            if not isinstance(result.get("id"), str) or result["id"] != self.pod_id:
                raise RuntimeError("Prime pod status identity differs")
            last = result
            status_value = result.get("status")
            if not isinstance(status_value, str):
                raise RuntimeError("Prime pod status is not a documented string")
            status = status_value.upper()
            if status == "ACTIVE":
                return result
            if status in {"ERROR", "TERMINATED", "DEAD", "CANCELLED"}:
                raise RuntimeError(f"Prime pod did not become ready: {status}")
            if status not in {"INSTALLING", "PENDING"}:
                raise RuntimeError(f"Prime pod status is not in the 0.6.20 lifecycle: {status}")
            time.sleep(1)
        del last
        raise TimeoutError("Prime pod status did not become ready")

    @staticmethod
    def _parse_ssh_endpoint(value: str) -> tuple[str | None, str, int]:
        for candidate in (part.strip() for part in value.split(",")):
            match = re.fullmatch(
                r"(?:(?P<user>[^@\s]+)@)?(?P<host>[^\s,-]+)(?:\s+-p\s+(?P<port>\d+))?",
                candidate,
            )
            if match is None:
                continue
            port = int(match.group("port") or "22")
            host = match.group("host")
            normalized_host = host.strip("[]")
            try:
                parsed_ip = ipaddress.ip_address(normalized_host)
            except ValueError:
                parsed_ip = None
            if (
                1 <= port <= 65535
                and host.lower()
                not in {
                    "localhost",
                    "0.0.0.0",
                    "::",
                    "[::]",
                    "unknown",
                    "n/a",
                    "na",
                    "none",
                }
                and "*" not in host
                and not host.startswith("127.")
                and not host.startswith("192.0.2.")
                and not (
                    parsed_ip is not None
                    and match.group("port") is None
                )
                and not (
                    parsed_ip is not None
                    and (
                        parsed_ip.is_loopback
                        or parsed_ip.is_unspecified
                        or parsed_ip.is_link_local
                        or parsed_ip.is_multicast
                        or parsed_ip.is_reserved
                    )
                )
            ):
                return match.group("user"), match.group("host"), port
        raise RuntimeError("Prime pod status lacks a documented SSH endpoint")

    @classmethod
    def _endpoint_from_status(cls, status: dict[str, Any]) -> str:
        ssh = status.get("ssh")
        if isinstance(ssh, list):
            ssh = ",".join(str(value) for value in ssh)
        if isinstance(ssh, str) and ssh.strip():
            cls._parse_ssh_endpoint(ssh)
            return ssh.strip()
        ip = status.get("ip")
        if isinstance(ip, list):
            ip = ",".join(str(value) for value in ip)
        host = (
            ip.strip()
            if isinstance(ip, str) and ip.strip()
            else ""
        )
        if host:
            try:
                parsed_ip = ipaddress.ip_address(host.strip("[]"))
            except ValueError:
                host = cls._parse_ssh_endpoint(host)[1]
            else:
                if (
                    parsed_ip.is_loopback
                    or parsed_ip.is_unspecified
                    or parsed_ip.is_link_local
                    or parsed_ip.is_multicast
                    or parsed_ip.is_reserved
                ):
                    raise RuntimeError("Prime pod status has an unusable SSH host")
                host = host.strip("[]")
        mappings = status.get("port_mappings")
        if isinstance(mappings, list):
            for item in mappings:
                if not isinstance(item, dict) or str(item.get("internal")) != "22":
                    continue
                external = item.get("external")
                if host and type(external) is int and 1 <= external <= 65535:
                    return f"{host} -p {external}"
        if host:
            raise RuntimeError("Prime pod status has no explicit SSH mapping")
        raise RuntimeError("Prime pod status lacks a documented SSH endpoint")

    def _endpoint_value(self) -> str:
        if self.host is None:
            raise RuntimeError("SSH endpoint is not bound")
        user = "" if self.ssh_user is None else f"{self.ssh_user}@"
        return f"{user}{self.host} -p {self.ssh_port}"

    def _ssh_options(self, program: str) -> list[str]:
        options = [
            program,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.known_hosts_path}",
            "-i",
            str(self.ssh_key),
        ]
        if self.ssh_port != 22:
            options.extend(["-P" if program == "scp" else "-p", str(self.ssh_port)])
        return options

    def _ssh_destination(self) -> str:
        return (f"{self.ssh_user}@" if self.ssh_user else "") + (self.host or "")

    def _ssh(self, remote_root: str) -> None:
        assert self.host is not None
        command = [
            *self._ssh_options("ssh"),
            self._ssh_destination(),
            "git",
            "clone",
            "--no-checkout",
            REPOSITORY_URL,
            remote_root,
        ]
        _run(command)
        observed_tree = self.facts["bundle"].get("tree")
        if not isinstance(observed_tree, str):
            raise RuntimeError("Prime observation lacks the authenticated bundle tree")
        _run(
            [
                *self._ssh_options("ssh"),
                self._ssh_destination(),
                "git",
                "-C",
                remote_root,
                "checkout",
                "--detach",
                self.bundle_commit,
            ]
        )
        checked = _run(
            [
                *self._ssh_options("ssh"),
                self._ssh_destination(),
                "git",
                "-C",
                remote_root,
                "rev-parse",
                "HEAD",
            ]
        )
        if checked.stdout.decode().strip() != self.bundle_commit:
            raise RuntimeError("remote checkout commit differs from the Prime observation")
        checked_tree = _run(
            [
                *self._ssh_options("ssh"),
                self._ssh_destination(),
                "git",
                "-C",
                remote_root,
                "rev-parse",
                "HEAD^{tree}",
            ]
        )
        if checked_tree.stdout.decode().strip() != observed_tree:
            raise RuntimeError("remote checkout tree differs from the Prime observation")
        remote_observation = f"{remote_root}/{LAUNCH_PRIME_OBSERVATION_RELATIVE}"
        remote_pod_observation = f"{remote_root}/{LAUNCH_POD_OBSERVATION_RELATIVE}"
        _run(
            [
                *self._ssh_options("ssh"),
                self._ssh_destination(),
                "mkdir",
                "-p",
                f"{remote_root}/{Path(LAUNCH_PRIME_OBSERVATION_RELATIVE).parent.as_posix()}",
            ]
        )
        _run(
            [
                *self._ssh_options("scp"),
                str(self.observation),
                f"{self._ssh_destination()}:{remote_observation}",
            ]
        )
        for relative in (
            LAUNCH_PROVISIONING_LEDGER_RELATIVE,
            LAUNCH_HANDOFF_RELATIVE,
            LAUNCH_HANDOFF_SIGNATURE_RELATIVE,
            LAUNCH_KNOWN_HOSTS_RELATIVE,
        ):
            local = self.repository / relative
            if local.is_symlink() or not local.is_file():
                raise RuntimeError(f"required local handoff file is missing: {relative}")
            _run(
                [
                    *self._ssh_options("ssh"),
                    self._ssh_destination(),
                    "mkdir",
                    "-p",
                    f"{remote_root}/{Path(relative).parent.as_posix()}",
                ]
            )
            _run(
                [
                    *self._ssh_options("scp"),
                    str(local),
                    f"{self._ssh_destination()}:{remote_root}/{relative}",
                ]
            )
        self._transfer_non_git_assets(remote_root)
        _run(
            [
                *self._ssh_options("ssh"),
                self._ssh_destination(),
                "/workspace/redco/.runtime/stage-d/uv",
                "run",
                "--offline",
                "--no-project",
                "python",
                "scripts/run_stage_d_v13_remote_bootstrap.py",
                "--repository",
                remote_root,
                "--observation",
                remote_observation,
                "--pod-observation",
                remote_pod_observation,
                "--capability",
                f"{remote_root}/{LAUNCH_HANDOFF_RELATIVE}",
                "--capability-signature",
                f"{remote_root}/{LAUNCH_HANDOFF_SIGNATURE_RELATIVE}",
                "--execute",
            ]
        )

    def _transfer_non_git_assets(self, remote_root: str) -> None:
        from redco.analysis import stage_d_v13_support_launch as launch

        for _name, source, _expected, remote_destination in self._asset_sources or ():
            if not remote_destination.startswith("/") and launch._git_path_exists(
                self.repository, "HEAD", remote_destination
            ):
                continue
            target = (
                remote_destination
                if remote_destination.startswith("/")
                else f"{remote_root}/{remote_destination}"
            )
            parent = posixpath.dirname(target.replace("\\", "/"))
            _run(
                [
                    *self._ssh_options("ssh"),
                    self._ssh_destination(),
                    "mkdir",
                    "-p",
                    parent,
                ]
            )
            _run(
                [
                    *self._ssh_options("scp"),
                    str(source),
                    f"{self._ssh_destination()}:{target}",
                ]
            )

    def _verified_non_git_assets(self) -> tuple[tuple[str, Path, str, str], ...]:
        verified: list[tuple[str, Path, str, str]] = []
        for name, binding in sorted(
            required_asset_mappings(
                self.repository,
                artifact_root=self.artifact_root,
            ).items()
        ):
            source = binding.local_source
            expected = binding.sha256
            source = source.resolve()
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(f"required launch asset is missing: {name}")
            if sha256_bytes(source.read_bytes()) != expected:
                raise RuntimeError(f"required launch asset hash differs: {name}")
            verified.append((name, source, expected, binding.remote_destination))
        return tuple(verified)

    def _recover_and_terminate(self) -> None:
        recovery_error: BaseException | None = None
        try:
            if self.host is not None:
                remote_paths = (
                    (
                        "runs/stage-d/stage-d1-support-v13-launch/attempt-v1.json",
                        "runs/stage-d/stage-d1-support-v13-launch/attempt-v1.json",
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13-launch/provisioning-ledger-v1.json",
                        "runs/stage-d/stage-d1-support-v13-launch/provisioning-ledger-v1.json",
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13-launch/provisioning-ledger-v1.dispatch.jsonl",
                        "runs/stage-d/stage-d1-support-v13-launch/provisioning-ledger-v1.dispatch.jsonl",
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13-launch/runtime",
                        "runs/stage-d/stage-d1-support-v13-launch/runtime",
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13/ledger",
                        "runs/stage-d/stage-d1-support-v13/ledger",
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13/source-artifacts",
                        "runs/stage-d/stage-d1-support-v13/source-artifacts",
                    ),
                    (
                        "reports/stage-d1-support-v13-support-report-v1.json",
                        "reports/stage-d1-support-v13-support-report-v1.json",
                    ),
                    (
                        LAUNCH_POD_OBSERVATION_RELATIVE,
                        LAUNCH_POD_OBSERVATION_RELATIVE,
                    ),
                    (
                        LAUNCH_HANDOFF_RELATIVE,
                        LAUNCH_HANDOFF_RELATIVE,
                    ),
                    (
                        LAUNCH_HANDOFF_SIGNATURE_RELATIVE,
                        LAUNCH_HANDOFF_SIGNATURE_RELATIVE,
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13-launch/provision-claim-v2.json",
                        "runs/stage-d/stage-d1-support-v13-launch/provision-claim-v2.json",
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13/source-roster",
                        "runs/stage-d/stage-d1-support-v13/source-roster",
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13/branch-artifacts",
                        "runs/stage-d/stage-d1-support-v13/branch-artifacts",
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13/score-artifacts",
                        "runs/stage-d/stage-d1-support-v13/score-artifacts",
                    ),
                    (
                        "runs/stage-d/stage-d1-support-v13/execution-manifest-v1.json",
                        "runs/stage-d/stage-d1-support-v13/execution-manifest-v1.json",
                    ),
                )
                for remote_relative, local_relative in remote_paths:
                    destination = self.repository / local_relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    remote_path = f"{self.remote_root}/{remote_relative}"
                    probe = subprocess.run(
                        [
                            *self._ssh_options("ssh"),
                            self._ssh_destination(),
                            "test",
                            "-e",
                            remote_path,
                        ],
                        check=False,
                        capture_output=True,
                    )
                    if probe.returncode != 0:
                        continue
                    _run(
                        [
                            *self._ssh_options("scp"),
                            "-r",
                            f"{self._ssh_destination()}:{remote_path}",
                            str(destination),
                        ]
                    )
        except BaseException as error:
            recovery_error = error
        finally:
            if self.pod_id is not None:
                try:
                    self._prime_raw("pods", "terminate", self.pod_id, "--yes")
                except BaseException as error:
                    if recovery_error is None:
                        recovery_error = error
        try:
            deadline = time.monotonic() + POD_STATUS_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                remaining = self._prime_json("pods", "list")
                disks = self._prime_json("disks", "list")
                if _inventory_is_empty(
                    remaining,
                    "pods",
                    {"id", "name", "gpu", "status", "created_at"},
                ) and _inventory_is_empty(
                    disks,
                    "disks",
                    {
                        "id",
                        "name",
                        "size",
                        "status",
                        "provider",
                        "location",
                        "created_at",
                        "price_hr",
                    },
                ):
                    break
                time.sleep(1)
            else:
                raise RuntimeError("owned Prime resources were not fully terminated")
        except BaseException as error:
            if recovery_error is None:
                recovery_error = error
        try:
            self._wallet_after = self._wallet_facts(
                self._prime_json("wallet", "--limit", "100")
            )
            if self._ledger is not None:
                self._ledger.reconcile_billing(self._wallet_after)
        except BaseException as error:
            if recovery_error is None:
                recovery_error = error
        if recovery_error is not None:
            raise recovery_error

    @staticmethod
    def _wallet_facts(value: dict[str, Any]) -> dict[str, Any]:
        account_id = _required_string(
            value,
            "wallet_id" if isinstance(value.get("wallet_id"), str) else "account_id",
        )
        billing_cursor = value.get("billing_cursor")
        if not isinstance(billing_cursor, str) or not billing_cursor:
            billing_cursor = sha256_bytes(
                canonical_json_bytes(
                    {
                        "total_billings": value.get("total_billings"),
                        "recent_billings": value.get("recent_billings"),
                    }
                )
            )
        balance = value.get("balance_usd", value.get("wallet_usd"))
        if type(balance) not in {int, float}:
            raise RuntimeError("Prime wallet balance is missing")
        return {
            "account_id": account_id,
            "wallet_id": account_id,
            "team_id": value.get("team_id"),
            "currency": value.get("currency"),
            "total_billings": value.get("total_billings"),
            "recent_billings": value.get("recent_billings", []),
            "billing_cursor": billing_cursor,
            "wallet_usd": balance,
        }

    def _publish_lifecycle(self, error: BaseException | None) -> None:
        from redco.analysis import stage_d_v13_support_launch as launch

        before = self._wallet_facts(self.facts["wallet"])
        after = self._wallet_after
        billing_status = "reconciled"
        delta: float | None = None
        if after is None:
            billing_status = "not_observable_pre_execution"
        elif (
            after["account_id"] != before["account_id"]
            or after.get("team_id") != before.get("team_id")
            or after.get("currency") != before.get("currency")
        ):
            billing_status = "identity_mismatch"
        else:
            delta = float(before["wallet_usd"]) - float(after["wallet_usd"])
            if delta < 0 or delta > 12 or float(after["wallet_usd"]) < 18:
                billing_status = "uncertain_or_over_cap"
        if error is None and billing_status != "reconciled":
            raise RuntimeError(f"Prime billing reconciliation is {billing_status}")

        def recovered(path: Path, relative: str) -> dict[str, Any]:
            if path.is_file() and not path.is_symlink():
                return {
                    "status": "observed",
                    "path": relative,
                    "sha256": sha256_bytes(path.read_bytes()),
                }
            return {"status": "not_produced", "path": relative, "reason": "pre_execution_failure"}

        report_path = self.repository / (
            "reports/stage-d1-support-v13-support-report-v1.json"
        )
        report_binding = recovered(
            report_path,
            "reports/stage-d1-support-v13-support-report-v1.json",
        )
        execution_manifest_path = self.repository / launch.LAUNCH_RUNTIME_MANIFEST_RELATIVE
        execution_manifest_binding = recovered(
            execution_manifest_path,
            launch.LAUNCH_RUNTIME_MANIFEST_RELATIVE,
        )
        pod_observation_path = self.repository / LAUNCH_POD_OBSERVATION_RELATIVE
        pod_observation_binding = recovered(
            pod_observation_path,
            LAUNCH_POD_OBSERVATION_RELATIVE,
        )
        payload = {
            "schema_version": 1,
            "domain": "redco-stage-d1-support-v13-launch-lifecycle-v1",
            "state": "completed" if error is None else "failed",
            "bundle": self.facts["bundle"],
            "observation": {
                "path": self.observation.relative_to(self.repository).as_posix(),
                "sha256": self.observation_sha256,
                "captured_at_epoch": self.facts["captured_at_epoch"],
                "expires_at_epoch": self.facts["expires_at_epoch"],
            },
            "resource": dict(self.facts["resource"]),
            "pod": {
                "name": self.pod_name,
                "pod_id": self.pod_id,
                "created_at_epoch": self._pod_created_at_epoch,
                "endpoint": self._endpoint_value() if self.host is not None else None,
                "status": self._pod_status,
            },
            "billing": {
                "wallet_before": before,
                "wallet_after": after,
                "wallet_delta_usd": delta,
                "status": billing_status,
                "source": "wallet_balance_delta_and_recent_billings",
                "support_cap_usd": 12,
                "science_reserve_usd": 16,
                "teardown_reserve_usd": 2,
            },
            "recovery": {
                "support_report": report_binding,
                "execution_manifest": execution_manifest_binding,
                "pod_observation": pod_observation_binding,
                "pods_empty_after_termination": True,
                "disks_empty_after_termination": True,
                "termination_seconds": 2,
            },
            "error_message_sha256": (
                None if error is None else sha256_bytes(str(error).encode("utf-8"))
            ),
        }
        lifecycle_bytes = canonical_json_bytes(payload)
        terminal_bytes = terminal_payload(
            state=(
                "completed_support_only"
                if error is None
                else (
                    "failed_terminal_no_retry"
                    if self._ledger is not None and self._ledger.has_provider_post()
                    else "failed_zero_call_pre_dispatch"
                )
            ),
            provider_dispatch_observed=(
                False
                if self._ledger is None
                else self._ledger.has_provider_post()
            ),
            error=error,
            evidence={
                "lifecycle_sha256": sha256_bytes(lifecycle_bytes),
                "support_report": report_binding,
                "execution_manifest": execution_manifest_binding,
                "wallet_delta_usd": delta,
                "billing_status": billing_status,
            },
        )
        validate_output_paths(
            self.repository,
            launch._bundle_immutable_paths(self.repository),
            output_paths=(
                *launch.ATTEMPT_PATHS,
                launch.LAUNCH_RUNTIME_MANIFEST_RELATIVE,
                LAUNCH_POD_OBSERVATION_RELATIVE,
            ),
        )
        atomic_publish_set(
            self.repository,
            {
                launch.LAUNCH_LIFECYCLE_RELATIVE: lifecycle_bytes,
                launch.LAUNCH_TERMINAL_RELATIVE: terminal_bytes,
            },
            immutable_paths=launch._bundle_immutable_paths(self.repository),
            manifest_path=launch.LAUNCH_LIFECYCLE_RELATIVE,
            require_draft_envelope=False,
        )


def _inventory_is_empty(
    value: dict[str, Any],
    key: str,
    item_keys: set[str],
) -> bool:
    if set(value) != {key, "total_count", "offset", "limit"}:
        raise RuntimeError(f"Prime {key} inventory schema is invalid")
    rows = value.get(key)
    if (
        not isinstance(rows, list)
        or type(value["total_count"]) is not int
        or type(value["offset"]) is not int
        or type(value["limit"]) is not int
        or value["total_count"] != len(rows)
    ):
        raise RuntimeError(f"Prime {key} inventory pagination is invalid")
    for row in rows:
        if not isinstance(row, dict) or set(row) != item_keys:
            raise RuntimeError(f"Prime {key} inventory item schema is invalid")
    return not rows


def facts_resource(facts: dict[str, Any]) -> dict[str, Any]:
    resource = facts.get("resource")
    if not isinstance(resource, dict) or not isinstance(resource.get("resource_id"), str):
        raise ValueError("Prime observation lacks selected resource identity")
    return resource


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise RuntimeError(f"Prime response lacks {key}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--remote-root", default="/workspace/redco")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        raise SystemExit("local orchestrator is execution-only; pass --execute after review")
    LocalLaunchOrchestrator(
        args.repository,
        args.observation,
        args.ssh_key,
        args.asset_root,
    ).run(
        remote_root=args.remote_root
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
