"""Run one bounded MuSiQue warm-start attempt on a qualifying Prime pod."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from run_qasper_allocation_sweep_on_prime import (
    SshTransport,
    StageFailure,
    _all_pods,
    _eligible,
    _fetch,
    _price,
    _projection,
    _run,
    _wait_endpoint,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_COMMIT = "f97fae39af7934f3d43c24f5e1200c5b19ce3e3c"
CONFIG_SHA256 = "cc078c2786c0efa150c93029cf19361aa19ba56ef6ce90c8bb50cc3667574580"
SNAPSHOT_SHA256 = "99006bd20b1d5b42351fc231384ebf8d31a7e1bddf10f0d1a3c2f957b94b794d"
POD_NAME = "redco-musique-ans-support-warm-start-v1"
RESULT = ROOT / "runs/musique-ans-support-warm-start-v1/report.json"
ADAPTER = ROOT / "runs/musique-ans-support-warm-start-v1/adapter.safetensors"
REMOTE_ROOT = "/tmp/redco-musique-ans-support-warm-start-v1"
REMOTE_REPORT = f"{REMOTE_ROOT}/report.json"
REMOTE_ADAPTER = f"{REMOTE_ROOT}/adapter.safetensors"
REMOTE_REPO = f"{REMOTE_ROOT}/repo"
MAX_RATE_USD = 2.0
MIN_WALLET_USD = 2.5
MAX_COST_USD = 2.5
MAX_POD_SECONDS = 4500.0
MAX_PROCESS_SECONDS = 3600.0
TEARDOWN_RESERVE_SECONDS = 600.0
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_ADAPTER_BYTES = 512 * 1024 * 1024
_HEX64 = set("0123456789abcdef")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _finite_number(value: object, stage: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise StageFailure(stage)
    number = float(cast(int | float, value))
    if not math.isfinite(number):
        raise StageFailure(stage)
    return number


def _warm_eligible(row: dict[str, Any]) -> bool:
    return (
        _eligible(row)
        and type(row.get("isSpot")) is bool
        and isinstance(row.get("stockStatus"), str)
        and row["stockStatus"].strip().casefold() == "available"
    )


def _disk_size(row: Mapping[str, object]) -> int:
    disk = row.get("disk")
    if type(disk) is not dict or type(disk.get("defaultCount")) is not int:
        raise StageFailure("availability_disk")
    size = cast(int, disk["defaultCount"])
    if size <= 0:
        raise StageFailure("availability_disk")
    return size


def _select(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any] | None, int, int]:
    eligible_rows = [row for row in rows if _warm_eligible(row)]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in eligible_rows:
        identity = row.get("cloudId")
        if type(identity) is str and identity:
            groups.setdefault(identity, []).append(row)
    consistent: list[dict[str, Any]] = []
    collapsed = 0
    conflicts = 0
    for group in groups.values():
        if len({_projection(row) for row in group}) != 1:
            conflicts += 1
            continue
        consistent.append(group[0])
        collapsed += len(group) - 1
    consistent.sort(
        key=lambda row: (
            0 if row["isSpot"] is False else 1,
            _price(row),
            str(row["cloudId"]),
        )
    )
    return (consistent[0] if consistent else None), collapsed, conflicts


def _balance(wallet: Any) -> float:
    response = wallet.get(limit=100)
    return _finite_number(getattr(response, "balance_usd", None), "wallet_schema")


def _pod_id(pod: Any) -> str:
    value = getattr(pod, "id", None)
    if type(value) is not str or not value:
        raise StageFailure("pod_identity")
    return value


def _pod_name(pod: Any) -> str:
    value = getattr(pod, "name", None)
    return value if type(value) is str else ""


def _owned_pod_ids(rows: Sequence[Any], before: set[str]) -> set[str]:
    owned: set[str] = set()
    for pod in rows:
        identity = getattr(pod, "id", None)
        if (
            type(identity) is str
            and identity not in before
            and _pod_name(pod) == POD_NAME
        ):
            owned.add(identity)
    return owned


def _cleanup(
    pods: Any,
    before: set[str],
    known: set[str],
    deadline: float,
    *,
    ambiguous_create: bool = False,
) -> tuple[set[str], int]:
    dispatched: set[str] = set()
    first_error: BaseException | None = None
    last_count = -1

    def delete_known() -> None:
        nonlocal first_error
        for identity in sorted(known - dispatched):
            try:
                pods.delete(identity)
            except BaseException as error:
                first_error = first_error or error
            else:
                dispatched.add(identity)

    delete_known()
    while time.monotonic() < deadline:
        try:
            rows = _all_pods(pods)
            known.update(_owned_pod_ids(rows, before))
            delete_known()
            last_count = len(rows)
            if not rows and not ambiguous_create:
                if first_error is not None:
                    raise first_error
                return dispatched, 0
        except BaseException as error:
            first_error = first_error or error
        remaining = max(0.0, min(10.0, deadline - time.monotonic()))
        if remaining <= 0:
            break
        time.sleep(remaining)
    try:
        rows = _all_pods(pods)
        known.update(_owned_pod_ids(rows, before))
        delete_known()
        last_count = len(rows)
    except BaseException as error:
        first_error = first_error or error
    if first_error is not None:
        raise first_error
    if last_count != 0:
        raise StageFailure("teardown_residual_pods")
    return dispatched, last_count


def _remote_script() -> bytes:
    return f'''set -Eeuo pipefail
failure_phase=workspace_setup
trap 'status=$?; printf "REDCO_REMOTE_FAILURE:%s\\n" "$failure_phase" >&2; exit "$status"' ERR
mkdir -p {REMOTE_ROOT}
rm -rf {REMOTE_REPO} {REMOTE_ROOT}/result
failure_phase=repository_checkout
git clone --no-checkout {REMOTE_ROOT}/redco.bundle {REMOTE_REPO}
cd {REMOTE_REPO}
git checkout --detach {EXPERIMENT_COMMIT}
test "$(git rev-parse HEAD)" = "{EXPERIMENT_COMMIT}"
export PYTHONPATH="$PWD/src:$PWD/scripts"
failure_phase=uv_bootstrap
if ! command -v uv >/dev/null 2>&1; then
  if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
    test -x "$HOME/.local/bin/uv"
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1
export UV_CACHE_DIR={REMOTE_ROOT}/uv-cache
export CUDA_VISIBLE_DEVICES=0
run_uv() {{
  uv run --no-project --python 3.12 --index-strategy unsafe-best-match \\
    --extra-index-url https://download.pytorch.org/whl/cu128 \\
    --with torch==2.11.0 --with transformers==5.6.2 --with safetensors "$@"
}}
failure_phase=dependency_preflight
run_uv python scripts/run_musique_support_warm_start.py --check
failure_phase=cuda_probe
run_uv python -c 'import torch
name=torch.cuda.get_device_name(0)
memory=torch.cuda.get_device_properties(0).total_memory
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert "A100" in name and "80GB" in name
assert 79*1024**3 <= memory <= 81*1024**3'
failure_phase=warm_start
run_uv python scripts/run_musique_support_warm_start.py --run --output {REMOTE_REPORT}
failure_phase=adapter_recovery
if run_uv python -c 'import json
from pathlib import Path
value=json.loads(Path("{REMOTE_REPORT}").read_bytes())
raise SystemExit(0 if value.get("gate",{{}}).get("passed") is True else 1)'; then
  test -f {REMOTE_REPO}/runs/musique-ans-support-warm-start-v1/adapter.safetensors
  cp {REMOTE_REPO}/runs/musique-ans-support-warm-start-v1/adapter.safetensors {REMOTE_ADAPTER}
else
  test ! -e {REMOTE_REPO}/runs/musique-ans-support-warm-start-v1/adapter.safetensors
fi
trap - ERR
'''.encode()


def _validate_adapter(raw: bytes, record: Mapping[str, object]) -> None:
    if len(raw) <= 0 or len(raw) > MAX_ADAPTER_BYTES:
        raise StageFailure("adapter_size")
    digest = record.get("sha256")
    size = record.get("bytes")
    path = record.get("path")
    if (
        type(digest) is not str
        or len(digest) != 64
        or set(digest.casefold()) - _HEX64
        or type(size) is not int
        or size != len(raw)
        or path != "runs/musique-ans-support-warm-start-v1/adapter.safetensors"
        or digest != _sha256_bytes(raw)
    ):
        raise StageFailure("adapter_integrity")


def _validate_report(raw: bytes) -> tuple[bool, Mapping[str, object] | None]:
    if len(raw) <= 0 or len(raw) > MAX_REPORT_BYTES:
        raise StageFailure("report_size")
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise StageFailure("report_schema") from error
    if type(value) is not dict:
        raise StageFailure("report_schema")
    if _canonical(value) != raw:
        raise StageFailure("report_canonical")
    required = {
        "schema_version",
        "experiment",
        "status",
        "config_sha256",
        "snapshot_sha256",
        "model",
        "base",
        "training",
        "adapter",
        "post",
        "gate",
        "runtime",
        "authority",
        "gold_decomposition_in_prompt",
    }
    if set(value) != required:
        raise StageFailure("report_schema")
    if (
        value["schema_version"] != 1
        or value["experiment"] != "musique-ans-support-warm-start-v1"
        or value["status"] != "complete"
        or value["config_sha256"] != CONFIG_SHA256
        or value["snapshot_sha256"] != SNAPSHOT_SHA256
        or value["gold_decomposition_in_prompt"] is not False
    ):
        raise StageFailure("report_integrity")
    model = value["model"]
    if (
        type(model) is not dict
        or model.get("name") != "Qwen/Qwen3.5-4B"
        or model.get("revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or model.get("dtype") != "bfloat16"
        or model.get("cuda_devices") != 1
    ):
        raise StageFailure("report_model")
    authority = value["authority"]
    if type(authority) is not dict or any(
        authority.get(name) is not expected
        for name, expected in {
            "prime": False,
            "source": False,
            "parquet": False,
            "science": False,
            "model_calls": True,
            "training": True,
        }.items()
    ):
        raise StageFailure("report_authority")
    gate = value["gate"]
    if type(gate) is not dict or type(gate.get("passed")) is not bool:
        raise StageFailure("report_gate")
    adapter = value["adapter"]
    if gate["passed"]:
        if type(adapter) is not dict:
            raise StageFailure("adapter_missing")
        return True, cast(Mapping[str, object], adapter)
    if adapter is not None:
        raise StageFailure("adapter_on_failed_gate")
    return False, None


def _exclusive_bytes(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".prime-warm-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _publish_local(report: bytes, adapter: bytes | None) -> None:
    root = RESULT.parent
    if root.exists() or RESULT.exists() or ADAPTER.exists():
        raise StageFailure("local_output_exists")
    root.mkdir(parents=True, exist_ok=False)
    try:
        _exclusive_bytes(RESULT, report)
        if adapter is not None:
            _exclusive_bytes(ADAPTER, adapter)
    except BaseException:
        RESULT.unlink(missing_ok=True)
        ADAPTER.unlink(missing_ok=True)
        root.rmdir()
        raise


def _check() -> None:
    if RESULT.exists() or ADAPTER.exists() or RESULT.parent.exists():
        raise StageFailure("local_output_exists")
    _run(
        "experiment_commit",
        ["git", "cat-file", "-e", f"{EXPERIMENT_COMMIT}^{{commit}}"],
        10,
        cwd=ROOT,
    )
    config_path = ROOT / "configs/musique-ans-support-warm-start-v1.json"
    if _sha256_bytes(config_path.read_bytes()) != CONFIG_SHA256:
        raise StageFailure("config_hash")
    snapshot_path = ROOT / "data/musique-ans-support-warm-start-v1.json"
    if _sha256_bytes(snapshot_path.read_bytes()) != SNAPSHOT_SHA256:
        raise StageFailure("snapshot_hash")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "scripts"), environment.get("PYTHONPATH", "")]
    )
    _run(
        "experiment_check",
        [sys.executable, "scripts/run_musique_support_warm_start.py", "--check"],
        60,
        cwd=ROOT,
        env=environment,
    )


def _run_once() -> int:
    PodsClient = importlib.import_module("prime_cli.api.pods").PodsClient
    WalletClient = importlib.import_module("prime_cli.api.wallet").WalletClient
    APIClient = importlib.import_module("prime_cli.core").APIClient
    Config = importlib.import_module("prime_cli.core.config").Config
    api = APIClient(user_agent="redco-musique-ans-support-warm-start-v1")
    pods, wallet = PodsClient(api), WalletClient(api)
    before = {pod.id for pod in _all_pods(pods)}
    if before:
        raise StageFailure("preexisting_pods")
    before_balance = _balance(wallet)
    if before_balance < MIN_WALLET_USD:
        raise StageFailure("wallet_reserve")
    selected, collapsed, conflicts = _select(
        [*_fetch(api, "/availability/gpus"), *_fetch(api, "/availability/multi-node")]
    )
    if selected is None:
        print(
            json.dumps(
                {
                    "collapsed_duplicate_rows": collapsed,
                    "conflicts": conflicts,
                    "state": "no_capacity",
                },
                sort_keys=True,
            )
        )
        return 2
    image_values = selected["images"]
    image = (
        "ubuntu_22_cuda_12"
        if "ubuntu_22_cuda_12" in image_values
        else next((item for item in image_values if "cuda" in item.casefold()), image_values[0])
    )
    spec = {
        "pod": {
            "name": POD_NAME,
            "cloudId": selected["cloudId"],
            "gpuType": "A100_80GB",
            "socket": selected["socket"],
            "gpuCount": 1,
            "diskSize": _disk_size(selected),
            "vcpus": selected.get("vcpu", {}).get("defaultCount"),
            "memory": selected.get("memory", {}).get("defaultCount"),
            "image": image,
            "dataCenterId": selected.get("dataCenter"),
            "maxPrice": None,
            "country": None,
            "security": None,
            "jupyterPassword": None,
            "autoRestart": False,
            "customTemplateId": None,
            "envVars": [],
        },
        "provider": {"type": selected["provider"]},
        "disks": None,
        "team": {"teamId": api.config.team_id} if api.config.team_id else None,
    }
    known: set[str] = set()
    failure: BaseException | None = None
    started = time.monotonic()
    report: bytes | None = None
    adapter: bytes | None = None
    passed = False
    create_dispatched = False
    create_acknowledged = False
    try:
        create_dispatched = True
        pod = pods.create(spec)
        identity = _pod_id(pod)
        known.add(identity)
        create_acknowledged = True
        target, port = _wait_endpoint(pods, next(iter(known)), started + 15 * 60)
        with tempfile.TemporaryDirectory(prefix="redco-prime-") as raw_temp:
            temp = Path(raw_temp)
            bundle = temp / "redco.bundle"
            _run(
                "bundle_create",
                ["git", "bundle", "create", str(bundle), "main"],
                60,
                cwd=ROOT,
            )
            transport = SshTransport(
                target, port, Path(Config().ssh_key_path), temp / "known_hosts"
            )
            transport.establish_trust()
            transport.upload_file(f"{REMOTE_ROOT}/redco.bundle", bundle)
            remaining = min(
                MAX_PROCESS_SECONDS,
                started + MAX_POD_SECONDS - time.monotonic() - TEARDOWN_RESERVE_SECONDS,
            )
            if remaining <= 60:
                raise StageFailure("experiment_time_reserve")
            transport.run_script(_remote_script(), remaining)
            report = transport.download(REMOTE_REPORT)
            passed, adapter_record = _validate_report(report)
            if passed:
                adapter = transport.download(REMOTE_ADAPTER, MAX_ADAPTER_BYTES)
                if adapter_record is None:
                    raise StageFailure("adapter_missing")
                _validate_adapter(adapter, adapter_record)
        _publish_local(report, adapter)
    except BaseException as error:
        failure = error
    cleanup_deadline = min(started + MAX_POD_SECONDS, time.monotonic() + TEARDOWN_RESERVE_SECONDS)
    try:
        _cleanup(
            pods,
            before,
            known,
            cleanup_deadline,
            ambiguous_create=create_dispatched and not create_acknowledged,
        )
    except BaseException as error:
        failure = failure or error
    try:
        after_balance = _balance(wallet)
        cost = max(0.0, before_balance - after_balance)
        if cost > MAX_COST_USD:
            failure = failure or StageFailure("cost_cap")
    except BaseException as error:
        cost = None
        failure = failure or error
    try:
        remaining_pods = _all_pods(pods)
        pods_remaining: int | None = len(remaining_pods)
        if remaining_pods:
            failure = failure or StageFailure("teardown_residual_pods")
    except BaseException as error:
        pods_remaining = None
        failure = failure or error
    terminal = {
        "observed_cost_usd": round(cost, 4) if cost is not None else None,
        "pods_remaining": pods_remaining,
        "state": "terminal" if failure is None else "failed_terminal",
        "failure_stage": (
            failure.stage
            if isinstance(failure, StageFailure)
            else type(failure).__name__
            if failure is not None
            else None
        ),
    }
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return 0 if failure is None else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if args.check:
            _check()
            print(json.dumps({"state": "ready"}, sort_keys=True))
            return 0
        _check()
        return _run_once()
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        stage = error.stage if isinstance(error, StageFailure) else type(error).__name__
        print(
            json.dumps(
                {
                    "failure_stage": stage,
                    "observed_cost_usd": None,
                    "pods_remaining": None,
                    "state": "failed_terminal",
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
