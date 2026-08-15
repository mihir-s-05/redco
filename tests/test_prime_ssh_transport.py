from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from run_qasper_allocation_sweep_on_prime import SshTransport, StageFailure


def _transport(tmp_path: Path) -> SshTransport:
    key = tmp_path / "key"
    key.write_text("test", encoding="utf-8")
    return SshTransport("root@example.test", "2222", key, tmp_path / "known_hosts")


def test_transport_streams_exact_bytes_after_isolated_trust(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    transport = _transport(tmp_path)

    def fake_run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append({"args": args, **kwargs})
        if "StrictHostKeyChecking=accept-new" in args:
            transport.known_hosts.write_text("host key", encoding="utf-8")
        stdout = b'{"ok":true}' if args[-2:] == ["cat", "/workspace/report.json"] else b""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    transport.establish_trust()
    transport.upload("/workspace/repo.bundle", b"bundle-bytes")
    assert transport.download("/workspace/report.json") == b'{"ok":true}'

    assert "StrictHostKeyChecking=accept-new" in calls[0]["args"]
    assert all("StrictHostKeyChecking=yes" in call["args"] for call in calls[1:])
    assert calls[1]["input"] == b"bundle-bytes"
    assert all(call["args"][0] == "ssh" for call in calls)


def test_transport_requires_captured_host_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b""),
    )
    with pytest.raises(StageFailure, match="ssh_trust_evidence"):
        transport.establish_trust()


def test_transport_rejects_untrusted_use_and_unsafe_paths(tmp_path: Path) -> None:
    transport = _transport(tmp_path)
    with pytest.raises(StageFailure, match="ssh_not_trusted"):
        transport.upload("/workspace/repo.bundle", b"data")
    transport._trusted = True
    with pytest.raises(StageFailure, match="remote_path"):
        transport.upload("/workspace/../secret", b"data")


def test_transport_bounds_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    transport = _transport(tmp_path)
    transport._trusted = True
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=b"12345", stderr=b""),
    )
    with pytest.raises(StageFailure, match="ssh_download_size"):
        transport.download("/workspace/report.json", max_bytes=4)


def test_transport_sanitizes_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(tmp_path)

    def fail(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.CalledProcessError(1, args, stderr=b"sensitive endpoint")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(StageFailure, match="ssh_trust") as captured:
        transport.establish_trust()
    assert "sensitive" not in str(captured.value)
