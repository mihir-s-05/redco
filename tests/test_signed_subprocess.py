from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from redco.integrations.signed_subprocess import (
    atomic_write_json,
    run_and_hard_exit,
    sign_payload,
    verify_signed_payload,
)


def test_atomic_signed_payload_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "score.json"
    payload = sign_payload({"schema_version": 1, "models": [{"name": "candidate"}]})
    atomic_write_json(path, payload)

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert verify_signed_payload(loaded) == payload["signed_payload_sha256"]
    assert not list(path.parent.glob("*.tmp"))

    loaded["models"][0]["name"] = "tampered"
    with pytest.raises(ValueError, match="mismatch"):
        verify_signed_payload(loaded)


def test_hard_exit_reports_success_and_failure() -> None:
    statuses: list[int] = []

    def fake_exit(status: int) -> NoReturn:
        statuses.append(status)
        raise SystemExit(status)

    with pytest.raises(SystemExit):
        run_and_hard_exit(lambda: None, exit_process=fake_exit)
    assert statuses == [0]

    def fail() -> None:
        raise RuntimeError("native worker failed")

    with pytest.raises(SystemExit):
        run_and_hard_exit(fail, exit_process=fake_exit)
    assert statuses == [0, 1]


def _write_dummy_scorer(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "from redco.integrations.signed_subprocess import atomic_write_json, sign_payload",
                "output = Path(sys.argv[1])",
                "status = int(sys.argv[2])",
                "payload = sign_payload({",
                "  'schema_version': 1,",
                "  'source': {'cases_sha256': 'cases', 'model': 'model'},",
                "  'models': [],",
                "})",
                "atomic_write_json(output, payload)",
                "raise SystemExit(status)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_supervisor_accepts_only_zero_exit_and_valid_signature(tmp_path: Path) -> None:
    from redco.integrations.scorer_supervisor import supervise

    child = tmp_path / "child.py"
    _write_dummy_scorer(child)
    output = tmp_path / "score.json"
    verified = tmp_path / "verified.json"
    command = [sys.executable, str(child), str(output), "0"]

    sentinel = supervise(
        command,
        output=output,
        verified=verified,
        expected_cases_sha256="cases",
        expected_model="model",
        expected_analysis=None,
    )
    assert sentinel["status"] == "verified"
    assert verify_signed_payload(json.loads(verified.read_text(encoding="utf-8")))

    failed_output = tmp_path / "failed-score.json"
    with pytest.raises(RuntimeError, match="status 7"):
        supervise(
            [sys.executable, str(child), str(failed_output), "7"],
            output=failed_output,
            verified=tmp_path / "failed-verified.json",
            expected_cases_sha256="cases",
            expected_model="model",
            expected_analysis=None,
        )
    assert failed_output.exists()
    assert not (tmp_path / "failed-verified.json").exists()


def test_supervisor_rejects_tampered_payload(tmp_path: Path) -> None:
    from redco.integrations.scorer_supervisor import supervise

    output = tmp_path / "score.json"
    child = tmp_path / "tamper.py"
    child.write_text(
        "import json,sys\n"
        "json.dump({'source': {'cases_sha256': 'cases', 'model': 'model'}, "
        "'signed_payload_sha256': '0' * 64}, open(sys.argv[1], 'w'))\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mismatch"):
        supervise(
            [sys.executable, str(child), str(output)],
            output=output,
            verified=tmp_path / "verified.json",
            expected_cases_sha256="cases",
            expected_model="model",
            expected_analysis=None,
        )
