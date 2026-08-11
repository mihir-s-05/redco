"""Signed evidence and orchestration for the test-only Prime one-shot."""

from __future__ import annotations

import json
import secrets
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from redco.analysis import stage_d_v13_prime_inventory_v5 as v5
from redco.analysis import stage_d_v13_prime_test_one_shot_runtime_binding_v2 as _runtime_binding
from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    HANDOFF_SIGN_TIMEOUT_SECONDS,
    KEYSCAN_TIMEOUT_SECONDS,
    MAXIMUM_POD_SECONDS,
    MAXIMUM_RATE_USD,
    READINESS_AUTHORITY,
    REMOTE_TIMEOUT_SECONDS,
    SUPPORT_CAP_USD,
    TERMINAL_DOMAIN,
    TERMINAL_SIGN_TIMEOUT_SECONDS,
    TRANSFER_TIMEOUT_SECONDS,
    WALLET_MINIMUM_USD,
    CommandResult,
    SigningIdentity,
    canonical_json,
    exclusive_runtime_root,
    fixed_runtime_path,
    publish_once,
    sha256_bytes,
    strict_object,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_evidence_v2 import (
    MAX_TRANSCRIPT_BYTES,
    artifact_dag,
    signed_envelope,
    verify_terminal_evidence,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_handoff_v2 import (
    LINUX_UV_SHA256,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_prime_v2 import (
    _Lifecycle,
    assess_pages,
    cleanup,
    direct_create,
    sha_file,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_remote_v2 import (
    build_handoff_payload,
    handoff_consumer_script,
    parse_endpoint,
    reconcile_created_pod,
    remote_test_script,
    status_active,
    validate_gpu_facts,
    validate_junit,
    verify_openssh_sshsig,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_runtime_binding_v2 import (
    RuntimeBinding,
    _build_production_context,
    _RuntimeContext,
)
from redco.analysis.stage_d_v13_prime_test_one_shot_wallet_v2 import (
    WalletSnapshot,
    decimal_value,
)


@dataclass(frozen=True, slots=True)
class OneShotResult:
    state: str
    disposition: str
    create_dispatched: bool
    tests_passed: bool
    cleanup_proven: bool
    terminal_sha256: str | None
    result_authority: tuple[tuple[str, bool], ...] = tuple(READINESS_AUTHORITY.items())
    terminal_domain: str = TERMINAL_DOMAIN

    @property
    def exit_code(self) -> int:
        return (
            0 if self.state == "completed" else 10 if self.state == "no_qualifying_capacity" else 20
        )

    def value(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "domain": self.terminal_domain,
            "state": self.state,
            "disposition": self.disposition,
            "create_dispatched": self.create_dispatched,
            "tests_passed": self.tests_passed,
            "cleanup_proven": self.cleanup_proven,
            "terminal_sha256": self.terminal_sha256,
            "authority": dict(self.result_authority),
        }


def _sign(context: _RuntimeContext, payload: bytes, namespace: str, *, timeout: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="redco-prime-one-shot-sign-") as directory:
        target = Path(directory) / "payload"
        target.write_bytes(payload)
        argv = (
            str(context.keygen_executable),
            "-Y",
            "sign",
            "-f",
            str(context.signing_key),
            "-n",
            namespace,
            str(target),
        )
        result = context.run(argv, None, timeout)
        if result.argv != argv or result.returncode:
            raise RuntimeError("Prime one-shot OpenSSH signing failed")
        signature = target.with_suffix(".sig").read_bytes()
        verify_openssh_sshsig(payload, signature, context.identity.public_key, namespace)
        return signature


def _handoff(
    owner: _Lifecycle,
    paths: dict[str, Path],
    assessment: bytes,
    envelope: bytes,
    transcript: bytes,
    resource: dict[str, Any],
    pod_id: str,
    status: dict[str, Any],
    endpoint: tuple[str | None, str, int],
    known_hosts: bytes,
) -> tuple[bytes, bytes, bytes]:
    now = owner.context.now()
    selected_facts = json.loads(assessment)["selected_facts"]
    test_script = remote_test_script(owner.context.authorization["commit"], selected_facts)
    payload, test_script = build_handoff_payload(
        authorization=owner.context.authorization,
        claim_sha256=sha_file(paths["claim"]),
        transcript_sha256=sha256_bytes(transcript),
        assessment_sha256=sha256_bytes(assessment),
        assessment_envelope_sha256=sha256_bytes(envelope),
        selected_resource_sha256=sha256_bytes(canonical_json(resource)),
        selected_facts=selected_facts,
        pod_identity_sha256=sha256_bytes(pod_id.encode()),
        pod_status_sha256=sha256_bytes(canonical_json(status)),
        ssh_user=endpoint[0],
        ssh_host=endpoint[1],
        ssh_port=endpoint[2],
        known_hosts=known_hosts,
        test_script=test_script,
        nonce=secrets.token_hex(32),
        issued_at_epoch=now,
        authority=dict(owner.context.binding.handoff_authority),
    )
    signature = _sign(
        owner.context,
        payload,
        owner.context.binding.handoff_namespace,
        timeout=HANDOFF_SIGN_TIMEOUT_SECONDS,
    )
    publish_once(paths["handoff"], payload)
    publish_once(paths["handoff-signature"], signature)
    publish_once(
        paths["handoff-envelope"],
        signed_envelope(
            payload,
            signature,
            owner.context.binding.handoff_namespace,
            owner.context.identity,
            domain=owner.context.binding.signed_envelope_domain,
            authority=dict(owner.context.binding.handoff_authority),
        ),
    )
    return payload, signature, test_script


def _run_one_shot(binding: RuntimeBinding) -> OneShotResult:
    if not _runtime_binding._is_trusted_binding(binding):
        raise ValueError("Prime one-shot runtime binding is not the canonical singleton")
    context = _build_production_context(binding)
    root = exclusive_runtime_root(context.repository, context.binding.evidence_root)
    names = {
        name: fixed_runtime_path(root, filename)
        for name, filename in context.binding.artifact_filenames
    }
    started = context.monotonic()
    owner = _Lifecycle(context, root, started)
    claim = canonical_json(
        {
            "schema_version": 2,
            "domain": context.binding.claim_domain,
            "state": "observation_attempt_consumed",
            "authorization": dict(context.authorization),
            "created_at_epoch": context.now(),
            "nonce": secrets.token_hex(32),
            "availability_attempt_limit": 1,
            "create_dispatch_limit": 1,
            "monitoring": False,
            "retry": False,
            "authority": dict(context.binding.claim_authority),
        }
    )
    publish_once(names["claim"], claim)
    state = "failed_terminal"
    primary: Exception | None = None
    recovery_errors: list[str] = []
    cleanup_errors: list[str] = []
    publication_errors: list[str] = []
    wallet_before: WalletSnapshot | None = None
    tests_passed = False
    cleanup_proven = False
    assessment_raw: bytes | None = None
    try:
        pages, diagnostic, failure, request_count = v5._capture_pages(
            context.client, context.transport_errors
        )
        if diagnostic is not None or failure is not None:
            raise RuntimeError("availability observation failed")
        transcript = canonical_json(
            {
                "pages": pages,
                "diagnostic": diagnostic,
                "failure": failure,
                "request_count": request_count,
            }
        )
        if len(transcript) > MAX_TRANSCRIPT_BYTES:
            raise RuntimeError("transcript exceeds bound")
        assessment_raw, resource = assess_pages(
            pages, context.authorization, context.now(), context.binding
        )
        signature = _sign(
            context,
            assessment_raw,
            context.binding.assessment_namespace,
            timeout=HANDOFF_SIGN_TIMEOUT_SECONDS,
        )
        assessment_envelope = signed_envelope(
            assessment_raw,
            signature,
            context.binding.assessment_namespace,
            context.identity,
            domain=context.binding.signed_envelope_domain,
            authority=dict(context.binding.assessment_authority),
        )
        publish_once(names["transcript"], transcript)
        publish_once(names["assessment"], assessment_raw)
        publish_once(names["assessment-envelope"], assessment_envelope)
        assessment = json.loads(assessment_raw)
        if resource is None:
            state = cast(str, assessment["state"])
        else:
            if context.now() > assessment["expires_at_epoch"]:
                raise TimeoutError("assessment expired")
            wallet_snapshot = owner.wallet()
            wallet_before = wallet_snapshot
            publish_once(
                names["wallet-before"],
                canonical_json(wallet_snapshot.evidence),
            )
            pods, _ = owner.list_pods()
            disks, _ = owner.list_disks()
            if wallet_before.balance < Decimal(str(WALLET_MINIMUM_USD)) or pods or disks:
                raise RuntimeError("pre-provision gate failed")
            rate = decimal_value(assessment["selected_facts"]["hourly_rate_usd"], "rate")
            if rate > Decimal(str(MAXIMUM_RATE_USD)) or rate * Decimal(
                MAXIMUM_POD_SECONDS
            ) / 3600 > Decimal(str(SUPPORT_CAP_USD)):
                raise RuntimeError("projected spend exceeds cap")
            if context.now() > assessment["expires_at_epoch"]:
                raise TimeoutError("assessment expired before create")
            direct_create(
                owner,
                resource,
                wallet_before.team_id,
                names["create-dispatch"],
                names["create-result"],
            )
            pod_id = reconcile_created_pod(owner)
            status = status_active(owner, pod_id)
            endpoint = parse_endpoint(status)
            scan = (
                str(context.openssh["ssh-keyscan"]),
                "-p",
                str(endpoint[2]),
                "--",
                endpoint[1],
            )
            one = owner.command(scan, timeout=KEYSCAN_TIMEOUT_SECONDS)
            two = owner.command(scan, timeout=KEYSCAN_TIMEOUT_SECONDS)
            if not one.stdout or one.stdout != two.stdout:
                raise RuntimeError("host keys unstable")
            publish_once(names["known-hosts"], one.stdout)
            handoff, handoff_signature, test_script = _handoff(
                owner,
                names,
                assessment_raw,
                assessment_envelope,
                transcript,
                resource,
                pod_id,
                status,
                endpoint,
                one.stdout,
            )
            consumer = handoff_consumer_script(
                authorization_commit=context.authorization["commit"],
                payload_sha256=sha256_bytes(handoff),
                public_key_sha256=sha256_bytes(context.identity.public_key),
                test_script_sha256=sha256_bytes(test_script),
                authority=dict(context.binding.handoff_authority),
            )
            ssh_host = f"[{endpoint[1]}]" if ":" in endpoint[1] else endpoint[1]
            destination = f"{endpoint[0] + '@' if endpoint[0] else ''}{ssh_host}"
            common = (
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                "-o",
                f"UserKnownHostsFile={names['known-hosts']}",
                "-i",
                str(context.signing_key),
            )
            with tempfile.TemporaryDirectory(prefix="redco-prime-transfer-") as directory:
                transfer = Path(directory)
                files = {
                    "payload.json": handoff,
                    "payload.sig": handoff_signature,
                    "public.key": context.identity.public_key,
                    "test.sh": test_script,
                    "consume.sh": consumer,
                }
                for name, raw in files.items():
                    transfer.joinpath(name).write_bytes(raw)
                owner.command(
                    (
                        str(context.openssh["scp"]),
                        *common,
                        "-P",
                        str(endpoint[2]),
                        "--",
                        str(context.linux_uv),
                        *(str(transfer / name) for name in files),
                        f"{destination}:/tmp/",
                    ),
                    timeout=TRANSFER_TIMEOUT_SECONDS,
                )
            bootstrap = (
                'set -euo pipefail; umask 077; root=/tmp/redco-one-shot-handoff-v2; mkdir "$root"; '
                + "; ".join(
                    "install -m "
                    + ("0700" if name.endswith(".sh") else "0600")
                    + f' /tmp/{name} "$root/{name}"'
                    for name in files
                )
                + "; test \"$(sha256sum /tmp/uv | cut -d' ' -f1)\" = "
                + LINUX_UV_SHA256
                + '; bash "$root/consume.sh"'
            )
            remote_result: CommandResult | None = None
            remote_error: Exception | None = None
            try:
                remote_result = owner.command(
                    (
                        str(context.openssh["ssh"]),
                        *common,
                        "-p",
                        str(endpoint[2]),
                        "--",
                        destination,
                        "bash",
                        "-lc",
                        bootstrap,
                    ),
                    timeout=REMOTE_TIMEOUT_SECONDS,
                    allow_failure=True,
                )
            except Exception as error:
                remote_error = error
            with tempfile.TemporaryDirectory(prefix="redco-prime-recovery-") as directory:
                recovery = Path(directory)
                try:
                    owner.command(
                        (
                            str(context.openssh["scp"]),
                            *common,
                            "-P",
                            str(endpoint[2]),
                            "--",
                            f"{destination}:/workspace/redco/.runtime/gpu-facts.json",
                            f"{destination}:/workspace/redco/.runtime/pytest.xml",
                            f"{destination}:/workspace/redco/.runtime/remote-status.json",
                            str(recovery),
                        ),
                        timeout=TRANSFER_TIMEOUT_SECONDS,
                    )
                    gpu = (recovery / "gpu-facts.json").read_bytes()
                    junit = (recovery / "pytest.xml").read_bytes()
                    remote_status = (recovery / "remote-status.json").read_bytes()
                    publish_once(names["gpu-facts"], gpu)
                    publish_once(names["junit"], junit)
                    publish_once(names["remote-status"], remote_status)
                    status_value = strict_object(
                        remote_status,
                        {"schema_version", "returncode"},
                        "remote status",
                    )
                    if status_value["schema_version"] != 2 or (
                        remote_result is not None
                        and status_value["returncode"] != remote_result.returncode
                    ):
                        raise ValueError("remote status differs from SSH outcome")
                    validate_gpu_facts(gpu, assessment["selected_facts"])
                    validate_junit(junit)
                    tests_passed = remote_result is not None and remote_result.returncode == 0
                except Exception as error:
                    recovery_errors.append(type(error).__name__)
            if remote_error is not None:
                raise remote_error
            if remote_result is None:
                raise RuntimeError("remote test command has no outcome")
            if remote_result.returncode:
                raise RuntimeError("remote test command failed")
    except Exception as error:
        primary = error
    finally:
        if owner.create_dispatched:
            try:
                cleanup_proven, evidence, errors = cleanup(owner, wallet_before)
                cleanup_errors.extend(errors)
            except Exception as error:
                cleanup_proven = False
                evidence = {"errors": [type(error).__name__]}
                cleanup_errors.append(type(error).__name__)
            try:
                publish_once(names["cleanup"], canonical_json(evidence))
            except Exception as error:
                publication_errors.append(type(error).__name__)
        else:
            cleanup_proven = True
        try:
            publish_once(names["command-records"], canonical_json(owner.commands))
        except Exception as error:
            publication_errors.append(type(error).__name__)
    if state not in {"no_qualifying_capacity", "ambiguous_capacity"}:
        state = (
            "completed"
            if not any((primary, recovery_errors, cleanup_errors, publication_errors))
            and tests_passed
            and cleanup_proven
            else "failed_terminal"
        )
    terminal_raw = canonical_json(
        {
            "schema_version": 2,
            "domain": context.binding.terminal_domain,
            "state": state,
            "disposition": state,
            "purpose": context.binding.terminal_purpose,
            "monitoring": False,
            "authorization": dict(context.authorization),
            "assessment_sha256": (
                sha256_bytes(names["assessment"].read_bytes())
                if names["assessment"].is_file()
                else None
            ),
            "create_dispatched": owner.create_dispatched,
            "tests_passed": tests_passed,
            "cleanup_proven": cleanup_proven,
            "primary_failure": None if primary is None else type(primary).__name__,
            "recovery_failures": recovery_errors,
            "cleanup_failures": cleanup_errors,
            "publication_failures": publication_errors,
            "evidence_dag": artifact_dag(names),
            "command_count": len(owner.commands),
            "prime_cli_call_count": owner.cli_calls,
            "wallet_api_call_count": owner.wallet_api_calls,
            "elapsed_seconds": context.monotonic() - started,
            "attempt_consumed": True,
            "retry": False,
            "authority": dict(context.binding.terminal_authority),
        }
    )
    terminal_sha256: str | None = None
    try:
        terminal_signature = _sign(
            context,
            terminal_raw,
            context.binding.terminal_namespace,
            timeout=TERMINAL_SIGN_TIMEOUT_SECONDS,
        )
        publish_once(names["terminal"], terminal_raw)
        publish_once(
            names["terminal-envelope"],
            signed_envelope(
                terminal_raw,
                terminal_signature,
                context.binding.terminal_namespace,
                context.identity,
                domain=context.binding.signed_envelope_domain,
                authority=dict(context.binding.terminal_authority),
            ),
        )
        terminal_sha256 = sha256_bytes(terminal_raw)
    except Exception as error:
        failure_raw = canonical_json(
            {
                "schema_version": 2,
                "domain": context.binding.terminal_domain,
                "state": "terminal_publication_failed",
                "failure": type(error).__name__,
                "cleanup_proven": cleanup_proven,
                "attempt_consumed": True,
                "retry": False,
                "authority": dict(context.binding.terminal_authority),
            }
        )
        with suppress(Exception):
            publish_once(names["terminal-publication-failure"], failure_raw)
        state = "failed_terminal"
    result = OneShotResult(
        state,
        state,
        owner.create_dispatched,
        tests_passed,
        cleanup_proven,
        terminal_sha256,
        tuple(context.binding.result_authority),
        context.binding.terminal_domain,
    )
    if terminal_sha256 is not None and context.binding is not _runtime_binding.V2_RUNTIME_BINDING:
        verify_terminal_evidence(root, context.identity, binding=context.binding)
    return result


def run_prime_test_one_shot_v2() -> OneShotResult:
    return _run_one_shot(_runtime_binding.V2_RUNTIME_BINDING)


__all__ = [
    "CommandResult",
    "OneShotResult",
    "SigningIdentity",
    "run_prime_test_one_shot_v2",
]
