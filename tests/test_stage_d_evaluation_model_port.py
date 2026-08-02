from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from test_stage_d_evaluation_ledger import _new_ledger, _start_arm

import redco.analysis.stage_d_evaluation_model_port as port_module
from redco.analysis.stage_d_evaluation_model_port import (
    EvaluationCallSpec,
    EvaluationModelPort,
)
from redco.analysis.stage_d_openai_response import parse_openai_response
from redco.contracts import EventAddress, canonical_json


class _Handler(BaseHTTPRequestHandler):
    calls: ClassVar[int] = 0

    def do_POST(self) -> None:
        type(self).calls += 1
        length = int(self.headers["content-length"])
        self.rfile.read(length)
        body = canonical_json(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "answer", "role": "assistant"},
                    }
                ],
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 3,
                    "total_tokens": 5,
                },
            }
        )
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        return


def _serializer(payload: dict, *, seed: int, cache_salt: str) -> bytes:
    return canonical_json(
        {
            **payload,
            "extra_body": {"cache_salt": cache_salt},
            "seed": seed,
        }
    )


def _spec() -> EvaluationCallSpec:
    return EvaluationCallSpec(EventAddress("root", 0, 0), {"messages": []})


def _port(ledger, task, session) -> EvaluationModelPort:
    return EvaluationModelPort(
        ledger=ledger,
        task=task,
        session=session,
        serialize_request=_serializer,
        timeout_seconds=1.0,
        max_calls=2,
        max_completion_tokens=8,
    )


def test_finalized_transcript_replays_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Handler.calls = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        ledger = _new_ledger(
            tmp_path,
            monkeypatch,
            endpoints=(endpoint, "http://127.0.0.1:1", "http://127.0.0.1:2"),
        )
        task, session = _start_arm(ledger, "stock")
        assert _port(ledger, task, session).call(_spec()).message["content"] == "answer"
        assert _port(ledger, task, session).call(_spec()).message["content"] == "answer"
        assert _Handler.calls == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_reserved_call_is_resumed_but_ambiguous_dispatch_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")

    def fail_before_dispatch(**_kwargs):
        raise RuntimeError("crash before dispatch")

    monkeypatch.setattr(port_module, "dispatch_reserved_local_http_once", fail_before_dispatch)
    with pytest.raises(RuntimeError, match="before dispatch"):
        _port(ledger, task, session).call(_spec())
    assert ledger.inspect().tasks[0].calls[0].dispatch_receipt_sha256 is None

    def fail_after_dispatch(*, ledger, call, session, **_kwargs):
        ledger.authorize_dispatch(call, session=session)
        raise RuntimeError("crash after dispatch")

    monkeypatch.setattr(port_module, "dispatch_reserved_local_http_once", fail_after_dispatch)
    with pytest.raises(RuntimeError, match="after dispatch"):
        _port(ledger, task, session).call(_spec())
    with pytest.raises(RuntimeError, match="ambiguous dispatched outcome"):
        _port(ledger, task, session).call(_spec())


def test_witnessed_response_is_finalized_after_parser_restart_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = _new_ledger(tmp_path, monkeypatch)
    task, session = _start_arm(ledger, "stock")
    original_parser = port_module.parse_openai_response

    def witness_then_crash(*, ledger, call, session, **_kwargs):
        dispatch = ledger.authorize_dispatch(call, session=session)
        raw = canonical_json(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "recovered", "role": "assistant"},
                    }
                ],
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 3,
                    "total_tokens": 5,
                },
            }
        )
        ledger.record_response(
            dispatch,
            session=session,
            raw_response_bytes=raw,
            status_code=200,
            headers=(("content-type", "application/json"),),
        )
        return type("Result", (), {"raw_response": raw, "status_code": 200})()

    monkeypatch.setattr(port_module, "dispatch_reserved_local_http_once", witness_then_crash)
    monkeypatch.setattr(
        port_module,
        "parse_openai_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("parser crash")),
    )
    with pytest.raises(RuntimeError, match="parser crash"):
        _port(ledger, task, session).call(_spec())

    monkeypatch.setattr(port_module, "parse_openai_response", original_parser)
    monkeypatch.setattr(
        port_module,
        "dispatch_reserved_local_http_once",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("network was retried")),
    )
    recovered = _port(ledger, task, session).call(_spec())
    assert recovered.message["content"] == "recovered"
    assert ledger.inspect().tasks[0].calls[0].outcome_sha256 is not None


def test_openai_parser_rejects_duplicate_keys_and_bad_usage() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_openai_response(b'{"choices":[],"choices":[],"usage":{}}', status_code=200)
    with pytest.raises(ValueError, match="inconsistent"):
        parse_openai_response(
            b'{"choices":[{"index":0,"message":{"role":"assistant"}}],'
            b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":9}}',
            status_code=200,
        )
