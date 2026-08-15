"""Run the reviewed QASPER allocation sweep on one bounded Prime pod."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_COMMIT = "ac3c06341a030e4e3b57464802485c6643c9bf5b"
POD_NAME = "redco-qasper-allocation-sweep-v1"
RESULT = ROOT / "runs/qasper-allocation-sweep-v1/report.json"
MAX_RATE_USD = 2.0
MAX_COST_USD = 3.0
MAX_POD_SECONDS = 75 * 60
MAX_REPORT_BYTES = 16 * 1024 * 1024
_REMOTE_PATH = re.compile(r"/[A-Za-z0-9._/-]+")


class StageFailure(RuntimeError):
    """A sanitized lifecycle failure whose stage is safe to report."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def _run(
    stage: str,
    args: list[str],
    timeout: float,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            env=env,
            input=stdin,
            check=True,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise StageFailure(stage) from error


class SshTransport:
    """A strict SSH channel that streams bytes without keyscan or scp."""

    def __init__(self, target: str, port: str, key: Path, known_hosts: Path) -> None:
        if "@" not in target or not port.isdigit() or not key.is_file():
            raise StageFailure("ssh_configuration")
        self.target = target
        self.port = port
        self.key = key
        self.known_hosts = known_hosts
        self._trusted = False

    def _options(self, trust: str) -> list[str]:
        return [
            "-i",
            str(self.key),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
            "-o",
            f"StrictHostKeyChecking={trust}",
            "-o",
            f"UserKnownHostsFile={self.known_hosts}",
            "-p",
            self.port,
            self.target,
        ]

    def establish_trust(self) -> None:
        _run("ssh_trust", ["ssh", *self._options("accept-new"), "true"], 30)
        if not self.known_hosts.is_file() or not self.known_hosts.read_bytes().strip():
            raise StageFailure("ssh_trust_evidence")
        self._trusted = True

    def _require_trust(self) -> None:
        if not self._trusted:
            raise StageFailure("ssh_not_trusted")

    @staticmethod
    def _path(path: str) -> str:
        if _REMOTE_PATH.fullmatch(path) is None or ".." in Path(path).parts:
            raise StageFailure("remote_path")
        return path

    def upload(self, path: str, data: bytes, timeout: float = 180) -> None:
        self._require_trust()
        remote = self._path(path)
        _run(
            "ssh_upload",
            ["ssh", *self._options("yes"), f"umask 077; cat > {remote}"],
            timeout,
            stdin=data,
        )

    def upload_file(self, path: str, source: Path, timeout: float = 180) -> None:
        self.upload(path, source.read_bytes(), timeout)

    def run_script(self, script: bytes, timeout: float) -> subprocess.CompletedProcess[bytes]:
        self._require_trust()
        return _run(
            "remote_experiment",
            ["ssh", *self._options("yes"), "bash", "-s"],
            timeout,
            stdin=script,
        )

    def download(self, path: str, max_bytes: int = MAX_REPORT_BYTES) -> bytes:
        self._require_trust()
        remote = self._path(path)
        result = _run(
            "ssh_download",
            ["ssh", *self._options("yes"), "cat", remote],
            180,
        )
        if len(result.stdout) > max_bytes:
            raise StageFailure("ssh_download_size")
        return result.stdout


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _price(row: dict[str, Any]) -> float:
    prices = row.get("prices")
    if type(prices) is not dict:
        return math.inf
    value = prices.get("communityPrice")
    if value is None:
        value = prices.get("onDemand")
    if type(value) not in (int, float) or isinstance(value, bool):
        return math.inf
    number = float(cast(int | float, value))
    return number if math.isfinite(number) and number > 0 else math.inf


def _eligible(row: dict[str, Any]) -> bool:
    disk, images = row.get("disk"), row.get("images")
    return (
        row.get("gpuType") == "A100_80GB"
        and type(row.get("gpuCount")) is int
        and row["gpuCount"] == 1
        and type(row.get("gpuMemory")) is int
        and row["gpuMemory"] == 80
        and row.get("isSpot") in {False, True, None}
        and isinstance(row.get("stockStatus"), str)
        and row["stockStatus"].strip().casefold() in {"available", "ready", "in_stock"}
        and isinstance(row.get("cloudId"), str)
        and bool(row["cloudId"])
        and isinstance(row.get("provider"), str)
        and bool(row["provider"])
        and isinstance(row.get("socket"), str)
        and bool(row["socket"])
        and type(disk) is dict
        and type(disk.get("defaultCount")) is int
        and disk["defaultCount"] > 0
        and type(images) is list
        and bool(images)
        and all(type(image) is str and bool(image) for image in images)
        and _price(row) <= MAX_RATE_USD
    )


def _projection(row: dict[str, Any]) -> bytes:
    def default(name: str) -> object:
        value = row.get(name)
        return value.get("defaultCount") if type(value) is dict else None

    return _canonical(
        {
            "cloud": row.get("cloudId"),
            "count": row.get("gpuCount"),
            "data_center": row.get("dataCenter"),
            "disk": default("disk"),
            "gpu": row.get("gpuType"),
            "gpu_memory": row.get("gpuMemory"),
            "images": row.get("images"),
            "memory": default("memory"),
            "provider": row.get("provider"),
            "rate": _price(row),
            "socket": row.get("socket"),
            "spot": row.get("isSpot"),
            "stock": row.get("stockStatus"),
            "vcpu": default("vcpu"),
        }
    )


def _choose(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, int, int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _eligible(row):
            groups[row["cloudId"]].append(row)
    unique: list[dict[str, Any]] = []
    collapsed = 0
    conflicts = 0
    for group in groups.values():
        if len({_projection(row) for row in group}) != 1:
            conflicts += 1
        else:
            unique.append(group[0])
            collapsed += len(group) - 1
    if not unique:
        return None, collapsed, conflicts
    return (
        min(
            unique,
            key=lambda row: (
                0 if row.get("isSpot") is False else 2 if row.get("isSpot") is True else 1,
                _price(row),
            ),
        ),
        collapsed,
        conflicts,
    )


def _fetch(client: Any, endpoint: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total: int | None = None
    page = 1
    while total is None or len(rows) < total:
        value = client.get(endpoint, params={"page": page, "page_size": 100})
        if (
            type(value) is not dict
            or type(value.get("items")) is not list
            or type(value.get("totalCount")) is not int
        ):
            raise StageFailure("availability_schema")
        if total is None:
            total = value["totalCount"]
        elif total != value["totalCount"]:
            raise StageFailure("availability_changed")
        items = value["items"]
        if any(type(item) is not dict for item in items):
            raise StageFailure("availability_schema")
        rows.extend(items)
        if len(rows) > total or (not items and len(rows) < total):
            raise StageFailure("availability_pagination")
        page += 1
    return rows


def _all_pods(client: Any) -> list[Any]:
    rows: list[Any] = []
    offset = 0
    while True:
        page = client.list(offset=offset, limit=100)
        rows.extend(page.data)
        if len(rows) >= page.total_count:
            return rows
        if not page.data:
            raise StageFailure("pod_pagination")
        offset += len(page.data)


def _endpoint(raw: str | list[str]) -> tuple[str, str]:
    tokens = shlex.split(raw) if isinstance(raw, str) else list(raw)
    if tokens and tokens[0].casefold() == "ssh":
        tokens.pop(0)
    target = next((token for token in tokens if "@" in token), None)
    port = tokens[tokens.index("-p") + 1] if "-p" in tokens else "22"
    if target is None or not port.isdigit():
        raise StageFailure("ssh_endpoint")
    return target, port


def _wait_endpoint(client: Any, pod_id: str, deadline: float) -> tuple[str, str]:
    while time.monotonic() < deadline:
        values = client.get_status([pod_id])
        if len(values) != 1:
            raise StageFailure("pod_status")
        if values[0].installation_failure:
            raise StageFailure("pod_installation")
        if values[0].ssh_connection:
            return _endpoint(values[0].ssh_connection)
        time.sleep(10)
    raise StageFailure("pod_ssh_timeout")


def _remote_script() -> bytes:
    return f"""set -euo pipefail
rm -rf /workspace/redco
git clone /workspace/redco.bundle /workspace/redco
cd /workspace/redco
git checkout --detach {EXPERIMENT_COMMIT}
test "$(git rev-parse HEAD)" = "{EXPERIMENT_COMMIT}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR=/workspace/.uv-cache
export CUDA_VISIBLE_DEVICES=0
uv run --no-project --python 3.12 --index-strategy unsafe-best-match \\
  --extra-index-url https://download.pytorch.org/whl/cu128 \\
  --with torch==2.11.0 --with transformers==5.6.2 \\
  python scripts/run_qasper_allocation_sweep.py \\
  --output /workspace/qasper-allocation-sweep-v1
""".encode()


def _validate_report(raw: bytes) -> str:
    value = json.loads(raw)
    payload = value.get("payload") if type(value) is dict else None
    if type(payload) is not dict or value.get("schema_version") != 1:
        raise StageFailure("report_schema")
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    runs = payload.get("runs")
    if payload.get("git_commit") != EXPERIMENT_COMMIT or value.get("payload_sha256") != digest:
        raise StageFailure("report_integrity")
    if (
        type(runs) is not list
        or len(runs) != 5
        or any(type(item) is not dict or len(item.get("arms", [])) != 4 for item in runs)
    ):
        raise StageFailure("report_cardinality")
    return digest


def _check() -> None:
    if RESULT.exists():
        raise StageFailure("result_exists")
    _run(
        "experiment_commit",
        ["git", "cat-file", "-e", f"{EXPERIMENT_COMMIT}^{{commit}}"],
        10,
        cwd=ROOT,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "scripts"), environment.get("PYTHONPATH", "")]
    )
    _run(
        "experiment_check",
        [sys.executable, "scripts/run_qasper_allocation_sweep.py", "--check"],
        60,
        cwd=ROOT,
        env=environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        _check()
        print(json.dumps({"state": "ready"}, sort_keys=True))
        return 0

    PodsClient = importlib.import_module("prime_cli.api.pods").PodsClient
    WalletClient = importlib.import_module("prime_cli.api.wallet").WalletClient
    APIClient = importlib.import_module("prime_cli.core").APIClient
    Config = importlib.import_module("prime_cli.core.config").Config

    api = APIClient(user_agent="redco-qasper-allocation-sweep-v1")
    pods, wallet = PodsClient(api), WalletClient(api)
    before = {pod.id for pod in _all_pods(pods)}
    balance = float(wallet.get(limit=100).balance_usd)
    if before or balance < MAX_COST_USD or RESULT.exists():
        raise StageFailure("preflight_state")
    selected, collapsed, conflicts = _choose(
        [*_fetch(api, "/availability/gpus"), *_fetch(api, "/availability/multi-node")]
    )
    if selected is None:
        print(
            json.dumps(
                {
                    "collapsed_duplicate_rows": collapsed,
                    "conflicting_identities": conflicts,
                    "state": "no_capacity",
                },
                sort_keys=True,
            )
        )
        return 2
    images = selected["images"]
    image = (
        "ubuntu_22_cuda_12"
        if "ubuntu_22_cuda_12" in images
        else next((item for item in images if "cuda" in item.casefold()), images[0])
    )
    spec = {
        "pod": {
            "name": POD_NAME,
            "cloudId": selected["cloudId"],
            "gpuType": selected["gpuType"],
            "socket": selected["socket"],
            "gpuCount": 1,
            "diskSize": selected["disk"]["defaultCount"],
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
    pod_id: str | None = None
    failure: BaseException | None = None
    started = time.monotonic()
    try:
        pod = pods.create(spec)
        pod_id = pod.id
        print(
            json.dumps(
                {
                    "gpu": "A100_80GB",
                    "rate_usd_per_hour": _price(selected),
                    "spot": selected.get("isSpot"),
                    "state": "pod_created",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        target, port = _wait_endpoint(pods, pod_id, started + 15 * 60)
        with tempfile.TemporaryDirectory(prefix="redco-prime-") as raw_temp:
            temp = Path(raw_temp)
            bundle = temp / "redco.bundle"
            _run("bundle_create", ["git", "bundle", "create", str(bundle), "main"], 30, cwd=ROOT)
            transport = SshTransport(
                target, port, Path(Config().ssh_key_path), temp / "known_hosts"
            )
            transport.establish_trust()
            transport.upload_file("/workspace/redco.bundle", bundle)
            remaining = min(65 * 60, started + MAX_POD_SECONDS - time.monotonic() - 300)
            if remaining <= 60:
                raise StageFailure("experiment_time_reserve")
            transport.run_script(_remote_script(), remaining)
            report = transport.download("/workspace/qasper-allocation-sweep-v1/report.json")
        digest = _validate_report(report)
        RESULT.parent.mkdir(parents=True, exist_ok=False)
        RESULT.write_bytes(report)
        print(json.dumps({"payload_sha256": digest, "state": "report_recovered"}, sort_keys=True))
    except BaseException as error:
        failure = error
    finally:
        cleanup: set[str] = {pod_id} if pod_id is not None else set()
        try:
            cleanup.update(
                pod.id for pod in _all_pods(pods) if pod.id not in before and pod.name == POD_NAME
            )
        except BaseException as error:
            failure = failure or error
        for identity in cleanup:
            try:
                pods.delete(identity)
            except BaseException as error:
                failure = failure or error
        try:
            deadline = time.monotonic() + 180
            while cleanup and time.monotonic() < deadline:
                if not [pod for pod in _all_pods(pods) if pod.id in cleanup]:
                    break
                time.sleep(10)
            else:
                if cleanup:
                    raise StageFailure("teardown_timeout")
        except BaseException as error:
            failure = failure or error
    cost = max(0.0, balance - float(wallet.get(limit=100).balance_usd))
    if cost > MAX_COST_USD:
        failure = failure or StageFailure("cost_cap")
    terminal: dict[str, object] = {
        "observed_cost_usd": round(cost, 4),
        "pods_remaining": len(_all_pods(pods)),
        "state": "terminal" if failure is None else "failed_terminal",
    }
    terminal["failure_stage"] = (
        failure.stage
        if isinstance(failure, StageFailure)
        else type(failure).__name__
        if failure is not None
        else None
    )
    print(json.dumps(terminal, sort_keys=True), flush=True)
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
