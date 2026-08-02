from __future__ import annotations

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from test_stage_d_evaluation_ledger import (
    _claim_server,
    _new_ledger,
    _receipt,
    _server_observation,
)

from redco.analysis.stage_d_evaluation_actuation import ActuatedProcessReceipt
from redco.analysis.stage_d_evaluation_codec import canonical_object
from redco.analysis.stage_d_evaluation_server import (
    EvaluationServerProcessObservation,
    probe_local_evaluation_server,
)


def test_server_process_observation_roundtrips_and_binds_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    launch = ledger.reserve_server_launch("stock")
    receipt_bytes = _receipt("stock", "server", launch.launch_record_sha256)
    receipt = ActuatedProcessReceipt.from_bytes(receipt_bytes)
    program = ledger.manifest.program("stock", "server")
    value = _server_observation(ledger, launch, receipt_bytes)
    observation = EvaluationServerProcessObservation.from_bytes(value)
    observation.verify(launch=launch, receipt=receipt, program=program)
    with pytest.raises(ValueError, match="frozen bindings"):
        replace(observation, checkpoint_manifest_sha256="f" * 64).verify(
            launch=launch,
            receipt=receipt,
            program=program,
        )


def test_server_attestation_rejects_forged_process_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    launch = ledger.reserve_server_launch("stock")
    receipt_bytes = _receipt("stock", "server", launch.launch_record_sha256)
    _claim_server(ledger, launch, receipt_bytes)
    observation = EvaluationServerProcessObservation.from_bytes(
        _server_observation(ledger, launch, receipt_bytes)
    )
    with pytest.raises(ValueError, match="frozen bindings"):
        ledger.attest_server(
            launch=launch,
            process_receipt_bytes=receipt_bytes,
            process_observation_bytes=replace(
                observation,
                endpoint="http://127.0.0.1:9999",
            ).to_bytes(),
            probe_response_bytes=b"probe",
        )
    assert not ledger.inspect().server_attestations


class _ProbeHandler(BaseHTTPRequestHandler):
    model_id = ""

    def do_GET(self) -> None:
        if self.path == "/health":
            body = b""
        elif self.path == "/v1/models":
            body = json.dumps({"data": [{"id": self.model_id}]}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


def test_server_probe_requires_the_bound_served_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProbeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    ledger = _new_ledger(
        tmp_path,
        monkeypatch,
        endpoints=(
            f"http://127.0.0.1:{port}",
            "http://127.0.0.1:61001",
            "http://127.0.0.1:61002",
        ),
    )
    program = ledger.manifest.program("stock", "server")
    try:
        _ProbeHandler.model_id = program.checkpoint_root
        evidence = canonical_object(
            probe_local_evaluation_server(program, timeout_seconds=2.0),
            "server probe",
        )
        assert evidence["expected_model_id"] == program.checkpoint_root
        _ProbeHandler.model_id = "wrong-model"
        with pytest.raises(ValueError, match="different model identity"):
            probe_local_evaluation_server(program, timeout_seconds=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
