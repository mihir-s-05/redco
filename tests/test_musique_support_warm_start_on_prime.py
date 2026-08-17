from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

import run_musique_support_warm_start_on_prime as launcher
from run_musique_support_warm_start_on_prime import (
    ADAPTER,
    EXPERIMENT_COMMIT,
    POD_NAME,
    RESULT,
    _check,
    _cleanup,
    _disk_size,
    _select,
    _validate_adapter,
    _validate_report,
)
from run_qasper_allocation_sweep_on_prime import StageFailure

ROOT = Path(__file__).parents[1]
launcher_any: Any = launcher


def _row(identity: str, *, spot: object = False, rate: float = 1.0) -> dict[str, Any]:
    return {
        "cloudId": identity,
        "dataCenter": "test-dc",
        "disk": {"defaultCount": 100},
        "gpuCount": 1,
        "gpuMemory": 80,
        "gpuType": "A100_80GB",
        "images": ["ubuntu_22_cuda_12"],
        "isSpot": spot,
        "memory": {"defaultCount": 80},
        "prices": {"communityPrice": rate},
        "provider": "test-provider",
        "socket": "test-socket",
        "stockStatus": "Available",
        "vcpu": {"defaultCount": 8},
    }


def _report(passed: bool, adapter: object = None) -> bytes:
    value = {
        "schema_version": 1,
        "experiment": "musique-ans-support-warm-start-v1",
        "status": "complete",
        "config_sha256": "cc078c2786c0efa150c93029cf19361aa19ba56ef6ce90c8bb50cc3667574580",
        "snapshot_sha256": "99006bd20b1d5b42351fc231384ebf8d31a7e1bddf10f0d1a3c2f957b94b794d",
        "model": {
            "name": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "dtype": "bfloat16",
            "cuda_devices": 1,
        },
        "base": {},
        "training": {},
        "adapter": adapter,
        "post": {},
        "gate": {"passed": passed},
        "runtime": {},
        "authority": {
            "prime": False,
            "source": False,
            "parquet": False,
            "science": False,
            "model_calls": True,
            "training": True,
        },
        "gold_decomposition_in_prompt": False,
    }
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def test_selection_collapses_only_identical_duplicate_projection() -> None:
    selected, collapsed, conflicts = _select([_row("cloud-a"), _row("cloud-a")])
    assert selected is not None
    assert selected["cloudId"] == "cloud-a"
    assert collapsed == 1
    assert conflicts == 0

    with pytest.raises(StageFailure, match="availability_duplicate_conflict"):
        _select([_row("cloud-a", rate=1.0), _row("cloud-a", rate=1.5)])
    with pytest.raises(StageFailure, match="availability_duplicate_conflict"):
        _select([_row("cloud-a"), _row("cloud-a", spot=None)])
    with pytest.raises(StageFailure, match="availability_duplicate_conflict"):
        _select([_row("cloud-a"), {**_row("cloud-a"), "gpuType": "H100"}])


def test_selection_rejects_unknown_spot_and_non_available_rows() -> None:
    selected, collapsed, conflicts = _select(
        [_row("unknown", spot=None), {**_row("ready"), "stockStatus": "Ready"}]
    )
    assert selected is None
    assert collapsed == 0
    assert conflicts == 0


def test_selected_disk_capacity_is_positive_and_explicit() -> None:
    assert _disk_size(_row("cloud-a")) == 100
    with pytest.raises(StageFailure, match="availability_disk"):
        _disk_size({"disk": {"defaultCount": 0}})


def test_cleanup_deletes_known_id_before_first_inventory_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Pods:
        def delete(self, identity: str) -> None:
            events.append(f"delete:{identity}")

        def list(self, *, offset: int, limit: int) -> object:
            del offset, limit
            events.append("inventory")
            raise StageFailure("inventory_failure")

    ticks = iter((0.0, 1.0, 2.0, 3.0))
    monkeypatch.setattr(launcher_any.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(launcher_any.time, "sleep", lambda _seconds: None)
    with pytest.raises(StageFailure, match="inventory_failure"):
        _cleanup(Pods(), set(), {"known-id"}, 2.5)
    assert events[0] == "delete:known-id"
    assert events[1] == "inventory"


def test_cleanup_failed_delete_is_not_marked_successfully_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    class Pods:
        def delete(self, identity: str) -> None:
            attempts.append(identity)
            raise StageFailure("delete_failure")

        def list(self, *, offset: int, limit: int) -> list[object]:
            del offset, limit
            return []

    with pytest.raises(StageFailure, match="delete_failure"):
        _cleanup(Pods(), set(), {"known-id"}, time.monotonic() + 5)
    assert attempts == ["known-id"]


def test_ambiguous_create_reconciles_late_exact_name_before_empty_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pod:
        def __init__(self, identity: str) -> None:
            self.id = identity
            self.name = POD_NAME

    class Page:
        def __init__(self, data: list[Pod]) -> None:
            self.data = data
            self.total_count = len(data)

    deleted: list[str] = []

    class Pods:
        def __init__(self) -> None:
            self.pages = [Page([]), Page([Pod("late-id")]), Page([])]

        def list(self, *, offset: int, limit: int) -> Page:
            del offset, limit
            return self.pages.pop(0)

        def delete(self, identity: str) -> None:
            deleted.append(identity)

    ticks = iter((0.0, 1.0, 2.0, 3.0, 4.0))
    monkeypatch.setattr(launcher_any.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(launcher_any.time, "sleep", lambda _seconds: None)
    dispatched, remaining = _cleanup(
        Pods(), set(), set(), 4.0, ambiguous_create=True
    )
    assert deleted == ["late-id"]
    assert dispatched == {"late-id"}
    assert remaining == 0


def test_run_preflight_precedes_prime_client_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def check() -> None:
        events.append("check")

    def run_once() -> int:
        events.append("run")
        return 0

    monkeypatch.setattr(launcher, "_check", check)
    monkeypatch.setattr(launcher, "_run_once", run_once)
    monkeypatch.setattr(sys, "argv", ["launcher", "--run"])
    assert launcher.main() == 0
    assert events == ["check", "run"]


def test_run_preflight_failure_blocks_prime_import_and_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail() -> None:
        raise StageFailure("config_hash")

    def run_once() -> int:
        calls.append("run")
        return 0

    monkeypatch.setattr(launcher, "_check", fail)
    monkeypatch.setattr(launcher, "_run_once", run_once)
    monkeypatch.setattr(sys, "argv", ["launcher", "--run"])
    assert launcher.main() == 1
    assert calls == []


def test_check_authenticates_commit_object_without_head_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        _stage: str,
        args: list[str],
        _timeout: float,
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(launcher, "_run", fake_run)
    _check()
    assert ["git", "cat-file", "-e", f"{EXPERIMENT_COMMIT}^{{commit}}"] in calls
    assert not any(args[:2] == ["git", "rev-parse"] for args in calls)


def test_report_and_adapter_integrity_are_bound_to_exact_bytes() -> None:
    passed, adapter = _validate_report(_report(False))
    assert passed is False
    assert adapter is None

    raw_adapter = b"safe-adapter"
    adapter_record = {
        "path": "runs/musique-ans-support-warm-start-v1/adapter.safetensors",
        "bytes": len(raw_adapter),
        "sha256": hashlib.sha256(raw_adapter).hexdigest(),
    }
    passed, adapter = _validate_report(_report(True, adapter_record))
    assert passed is True
    assert adapter == adapter_record
    _validate_adapter(raw_adapter, adapter_record)
    with pytest.raises(StageFailure, match="adapter_integrity"):
        _validate_adapter(b"changed", adapter_record)


def test_failed_gate_rejects_stranded_adapter() -> None:
    raw_adapter = b"stranded"
    adapter_record = {
        "path": "runs/musique-ans-support-warm-start-v1/adapter.safetensors",
        "bytes": len(raw_adapter),
        "sha256": hashlib.sha256(raw_adapter).hexdigest(),
    }
    with pytest.raises(StageFailure, match="adapter_on_failed_gate"):
        _validate_report(_report(False, adapter_record))


def test_check_is_source_free_and_requires_no_local_outputs() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + ";" + str(ROOT / "scripts")
    result = subprocess.run(
        [sys.executable, "scripts/run_musique_support_warm_start_on_prime.py", "--check"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"state": "ready"}
    assert not RESULT.exists()
    assert not ADAPTER.exists()
    assert not RESULT.parent.exists()
