"""Source-free endpoint receipt tests for Prime inventory v5."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from redco.analysis import stage_d_v13_prime_inventory_v3 as inventory_v3
from redco.analysis import stage_d_v13_prime_inventory_v5 as inventory
from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes

ROOT = Path(__file__).parents[1].resolve()
_CAPTURE_OWNER = inventory.capture_prime_inventory_raw_v5


def _item(index: int, **changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "cloudId": f"cloud-{index}",
        "gpuType": "L40S 48GB",
        "socket": "PCIe",
        "provider": "fixture-provider",
        "dataCenter": "fixture-dc",
        "country": "US",
        "gpuCount": 2,
        "gpuMemory": 96,
        "disk": {
            "minCount": 0,
            "defaultCount": 1250,
            "maxCount": 2000,
            "pricePerUnit": 0.01,
            "step": 1,
            "defaultIncludedInPrice": True,
            "additionalInfo": None,
        },
        "vcpu": None,
        "memory": None,
        "internetSpeed": None,
        "interconnect": None,
        "interconnectType": None,
        "provisioningTime": None,
        "stockStatus": "Available",
        "security": "datacenter",
        "prices": {
            "onDemand": 1.80,
            "communityPrice": 1.64,
            "isVariable": False,
            "currency": "USD",
        },
        "images": [],
        "isSpot": False,
        "prepaidTime": None,
    }
    value.update(changes)
    return value


@dataclass(frozen=True)
class _Response:
    status_code: int
    content: bytes
    headers: Mapping[str, str]


class _FakeTransport:
    def __init__(
        self,
        endpoint_items: Mapping[str, list[dict[str, object]]],
        *,
        fail_call: int | None = None,
        status_call: int | None = None,
        total_delta_call: int | None = None,
        malformed_call: int | None = None,
    ) -> None:
        self.endpoint_items = endpoint_items
        self.fail_call = fail_call
        self.status_call = status_call
        self.total_delta_call = total_delta_call
        self.malformed_call = malformed_call
        self.calls: list[tuple[str, str, dict[str, object], bool]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, object],
        follow_redirects: bool,
    ) -> _Response:
        copied = dict(params)
        self.calls.append((method, url, copied, follow_redirects))
        call = len(self.calls)
        if call == self.fail_call:
            httpx = inventory._load_httpx_module()
            request = httpx.Request(method, url)
            raise httpx.ReadTimeout(
                "fixture timeout must not escape into evidence", request=request
            )
        if call == self.malformed_call:
            return _Response(200, b"{", {"content-type": "application/json"})
        if call == self.status_call:
            return _Response(307, b"redirect", {"content-type": "text/plain"})
        endpoint = url.removeprefix(inventory.BASE_URL)
        items = self.endpoint_items[endpoint]
        page = cast(int, copied["page"])
        start = (page - 1) * inventory.PAGE_SIZE
        body_items = items[start : start + inventory.PAGE_SIZE]
        total = len(items) + (1 if call == self.total_delta_call else 0)
        body = canonical_json_bytes({"items": body_items, "totalCount": total})
        return _Response(200, body, {"content-type": "application/json; charset=utf-8"})


class _FakeClient:
    base_url = inventory.BASE_URL
    api_key = "fixture-not-serialized"

    def __init__(self, transport: _FakeTransport) -> None:
        self.client = transport

    def request(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("APIClient.request must not be used")


def _bind_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transport: _FakeTransport,
) -> Path:
    config = tmp_path / "config-home"
    config.mkdir()
    config_file = config / "config.json"
    config_file.write_bytes(b"fixture-secret-sentinel")
    environments = config / "environments"
    environments.mkdir()
    key_path = tmp_path / "operator-signing-key"
    subprocess.run(
        [
            str(inventory.OPENSSH_EXECUTABLE_PATH),
            "-q",
            "-t",
            "rsa",
            "-b",
            "2048",
            "-N",
            "",
            "-f",
            str(key_path),
        ],
        check=True,
        capture_output=True,
    )
    public_fields = key_path.with_name(key_path.name + ".pub").read_text(
        encoding="ascii"
    ).split()
    key_type, key_base64 = public_fields[:2]
    allowed_signers = f"mihir {key_type} {key_base64}\n".encode("ascii")
    identity = inventory._TerminalSigningIdentity(
        principal="mihir",
        key_type=key_type,
        public_key_base64=key_base64,
        fingerprint_sha256=inventory._fingerprint(key_type, key_base64),
        allowed_signers_sha256=sha256_bytes(allowed_signers),
    )
    monkeypatch.setattr(inventory, "ROOT", tmp_path)
    monkeypatch.setattr(
        inventory,
        "_authenticate_committed_capture_checkout",
        lambda: {"commit": "c" * 40, "tree": "d" * 40},
    )
    monkeypatch.setattr(
        inventory,
        "authenticate_installed_capture_owners",
        lambda: {"fixture": "authenticated"},
    )
    monkeypatch.setattr(inventory, "_load_terminal_signing_identity", lambda: identity)
    monkeypatch.setattr(inventory, "_config_paths", lambda: (config, config_file, environments))
    monkeypatch.setattr(inventory, "_construct_api_client", lambda: _FakeClient(transport))
    monkeypatch.setattr(
        inventory,
        "capture_prime_inventory_raw_v5",
        lambda: _CAPTURE_OWNER(key_path),
    )
    monkeypatch.setattr(time, "time", lambda: 1_000)
    return key_path


def _transport(count: int, *, second_count: int = 0, **kwargs: int | None) -> _FakeTransport:
    return _FakeTransport(
        {
            inventory.ENDPOINTS[0]: [_item(index) for index in range(count)],
            inventory.ENDPOINTS[1]: [
                _item(10_000 + index) for index in range(second_count)
            ],
        },
        **kwargs,
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("count", "expected_calls"), ((0, 2), (1, 2), (100, 2), (101, 3), (200, 3))
)
def test_exact_pagination_and_both_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    count: int,
    expected_calls: int,
) -> None:
    transport = _transport(count)
    _bind_capture(monkeypatch, tmp_path, transport)
    raw = inventory.capture_prime_inventory_raw_v5()
    receipt = json.loads(raw)
    assert receipt["state"] == "captured_endpoint_terminal"
    assert receipt["diagnostic"] is None
    assert len(transport.calls) == expected_calls
    assert all(call[0] == "GET" and call[3] is False for call in transport.calls)
    assert all(call[2]["gpu_count"] == "2" for call in transport.calls)
    assert b"fixture-secret-sentinel" not in raw
    assessment = json.loads(inventory.assess_prime_inventory_v5())
    if count == 1:
        expected_state = "observed_non_authorizing_resource"
    elif count > 1:
        expected_state = "observed_ambiguous_resources"
    else:
        expected_state = "observed_no_qualifying_resource"
    assert assessment["state"] == expected_state
    assert assessment["resource"] is None
    assert all(value is False for value in assessment["authorization"].values())


def test_partial_terminal_receipts_stop_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for mode, call in (("timeout", 2), ("redirect", 2), ("malformed", 2)):
        root = tmp_path / mode
        root.mkdir()
        transport = _transport(
            101,
            fail_call=call if mode == "timeout" else None,
            status_call=call if mode == "redirect" else None,
            malformed_call=call if mode == "malformed" else None,
        )
        _bind_capture(monkeypatch, root, transport)
        receipt = json.loads(inventory.capture_prime_inventory_raw_v5())
        assert receipt["state"] == "capture_failed_terminal"
        assert len(transport.calls) == 2
        assert (root / inventory.CLAIM_RELATIVE).is_file()
        assert (root / inventory.RAW_RELATIVE).is_file()
        assert not (root / inventory.ASSESSMENT_RELATIVE).exists()
        with pytest.raises(ValueError, match="not assessable"):
            inventory.assess_prime_inventory_v5()
        assert not (root / inventory.ASSESSMENT_RELATIVE).exists()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "exception_name",
    ["ConnectTimeout", "ReadTimeout", "ConnectError", "RemoteProtocolError"],
)
def test_real_pinned_httpx_request_errors_are_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception_name: str,
) -> None:
    class RequestErrorTransport(_FakeTransport):
        def request(
            self,
            method: str,
            url: str,
            *,
            params: Mapping[str, object],
            follow_redirects: bool,
        ) -> _Response:
            self.calls.append((method, url, dict(params), follow_redirects))
            httpx = inventory._load_httpx_module()
            request = httpx.Request(method, url)
            exception_type = getattr(httpx, exception_name)
            raise exception_type("credential-sentinel-exception-text", request=request)

    transport = RequestErrorTransport(
        {inventory.ENDPOINTS[0]: [], inventory.ENDPOINTS[1]: []}
    )
    _bind_capture(monkeypatch, tmp_path, transport)
    raw = inventory.capture_prime_inventory_raw_v5()
    receipt = json.loads(raw)
    assert receipt["state"] == "capture_failed_terminal"
    assert receipt["diagnostic"] == "transport_failure"
    assert receipt["pages"] == []
    assert receipt["request_count"] == 1
    assert len(transport.calls) == 1
    assert (tmp_path / inventory.CLAIM_RELATIVE).is_file()
    assert (tmp_path / inventory.RAW_RELATIVE).is_file()
    assert b"credential-sentinel" not in raw


def _replace_page_body(record: dict[str, object], body_value: object) -> None:
    body = canonical_json_bytes(body_value)
    record["decoded_application_body_b64"] = base64.b64encode(body).decode()
    record["decoded_application_body_sha256"] = sha256_bytes(body)
    record["decoded_application_body_bytes"] = len(body)


def _auth_envelope(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    envelope = cast(
        dict[str, object],
        json.loads((root / inventory.TERMINAL_AUTH_RELATIVE).read_bytes()),
    )
    payload_binding = cast(dict[str, object], envelope["payload"])
    payload = cast(
        dict[str, object],
        json.loads(base64.b64decode(cast(str, payload_binding["base64"]))),
    )
    return envelope, payload


def _write_auth_envelope(
    root: Path,
    envelope: dict[str, object],
    payload: dict[str, object],
    *,
    signing_key: Path | None = None,
) -> None:
    payload_raw = canonical_json_bytes(payload)
    envelope["payload"] = {
        "base64": base64.b64encode(payload_raw).decode("ascii"),
        "bytes": len(payload_raw),
        "sha256": sha256_bytes(payload_raw),
    }
    if signing_key is not None:
        signature = inventory._sign_bytes(signing_key, payload_raw)
        envelope["signature"] = {
            "base64": base64.b64encode(signature).decode("ascii"),
            "bytes": len(signature),
            "sha256": sha256_bytes(signature),
        }
    (root / inventory.TERMINAL_AUTH_RELATIVE).write_bytes(
        canonical_json_bytes(envelope)
    )


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation",
    ["delete", "insert", "reorder", "page", "endpoint", "total", "item", "coordinated"],
)
def test_transcript_mutations_fail_through_real_assessor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    transport = _transport(101, second_count=1)
    _bind_capture(monkeypatch, tmp_path, transport)
    inventory.capture_prime_inventory_raw_v5()
    raw_path = tmp_path / inventory.RAW_RELATIVE
    receipt = cast(dict[str, object], json.loads(raw_path.read_bytes()))
    pages = cast(list[dict[str, object]], receipt["pages"])
    if mutation == "delete":
        del pages[1]
    elif mutation == "insert":
        pages.insert(1, dict(pages[0]))
    elif mutation == "reorder":
        pages[0], pages[1] = pages[1], pages[0]
    elif mutation == "page":
        pages[0]["page_ordinal"] = 2
        cast(dict[str, object], pages[0]["params"])["page"] = 2
    elif mutation == "endpoint":
        pages[0]["endpoint"] = inventory.ENDPOINTS[1]
    elif mutation == "total":
        body = json.loads(base64.b64decode(cast(str, pages[0]["decoded_application_body_b64"])))
        body["totalCount"] = 100
        _replace_page_body(pages[0], body)
    elif mutation == "item":
        body = json.loads(base64.b64decode(cast(str, pages[0]["decoded_application_body_b64"])))
        body["items"][0]["provider"] = "substituted"
        _replace_page_body(pages[0], body)
    else:
        pages.reverse()
        pages[0]["page_ordinal"] = 99
        cast(dict[str, object], pages[0]["params"])["page"] = 99
        body = json.loads(base64.b64decode(cast(str, pages[-1]["decoded_application_body_b64"])))
        body["totalCount"] = 999
        body["items"][0]["provider"] = "coordinated-substitute"
        _replace_page_body(pages[-1], body)
    raw_path.write_bytes(canonical_json_bytes(receipt))
    with pytest.raises(ValueError, match=r"signed raw artifact"):
        inventory.assess_prime_inventory_v5()


def test_valid_capture_signs_one_terminal_payload_and_publishes_auth_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(1)
    _bind_capture(monkeypatch, tmp_path, transport)
    original_sign = inventory._sign_bytes
    original_publish = inventory._publish_fixed
    terminal_signatures = 0
    publications: list[str] = []

    def observed_sign(path: Path, value: bytes) -> bytes:
        nonlocal terminal_signatures
        if inventory.TERMINAL_AUTH_PAYLOAD_DOMAIN.encode("ascii") in value:
            terminal_signatures += 1
        return original_sign(path, value)

    def observed_publish(relative: str, raw: bytes) -> Path:
        publications.append(relative)
        return original_publish(relative, raw)

    monkeypatch.setattr(inventory, "_sign_bytes", observed_sign)
    monkeypatch.setattr(inventory, "_publish_fixed", observed_publish)
    raw = inventory.capture_prime_inventory_raw_v5()
    assert terminal_signatures == 1
    assert publications == [
        inventory.CLAIM_RELATIVE,
        inventory.TRANSCRIPT_RELATIVE,
        inventory.RAW_RELATIVE,
        inventory.TERMINAL_AUTH_RELATIVE,
    ]
    envelope, payload = _auth_envelope(tmp_path)
    openssh = inventory.authenticate_approved_openssh_executable()
    claim = json.loads((tmp_path / inventory.CLAIM_RELATIVE).read_bytes())
    assert payload["raw"] == {
        "path": inventory.RAW_RELATIVE,
        "artifact_sha256": sha256_bytes(raw),
    }
    assert claim["openssh_executable"] == openssh
    assert payload["openssh_executable_projection"] == openssh
    assert envelope["openssh_executable"] == openssh
    assert payload["assessment_allowed"] is True
    assert envelope["public_identity"] == inventory._load_terminal_signing_identity().projection()
    assert json.loads(inventory.assess_prime_inventory_v5())["state"] == (
        "observed_non_authorizing_resource"
    )


def test_absolute_approved_openssh_ignores_path_shadow_for_capture_and_assessment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    sentinel = tmp_path / "shadow-invoked"
    (shadow / "ssh-keygen.cmd").write_text(
        f"@echo invoked>{sentinel}\r\n@exit /b 0\r\n", encoding="ascii"
    )
    monkeypatch.setenv("PATH", f"{shadow};{os.environ['PATH']}")
    shadow_resolution = shutil.which("ssh-keygen")
    assert shadow_resolution is not None
    assert Path(shadow_resolution).samefile(shadow / "ssh-keygen.cmd")
    transport = _transport(1)
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    _bind_capture(monkeypatch, capture_root, transport)
    inventory.capture_prime_inventory_raw_v5()
    assert not sentinel.exists()
    monkeypatch.setenv("PATH", str(shadow))
    assessment = json.loads(inventory.assess_prime_inventory_v5())
    assert assessment["state"] == "observed_non_authorizing_resource"
    assert not sentinel.exists()
    projection = inventory.authenticate_approved_openssh_executable()
    assert projection == {
        "operator_host": "windows",
        "operator_host_only": True,
        "path": r"C:\Windows\System32\OpenSSH\ssh-keygen.exe",
        "normalized_absolute_path": r"c:\windows\system32\openssh\ssh-keygen.exe",
        "sha256": inventory.OPENSSH_EXECUTABLE_SHA256,
        "bytes": 862_208,
        "product_version": "OpenSSH_9.5p2 for Windows",
        "servicing_hardlink_path": str(inventory.OPENSSH_SERVICING_HARDLINK_PATH),
        "hardlink_count": 2,
        "path_lookup_allowed": False,
        "linux_fallback_allowed": False,
    }


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation", ["path", "bytes", "link", "alias"]
)
def test_wrong_openssh_identity_fails_before_signature_or_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    transport = _transport(1)
    _bind_capture(monkeypatch, tmp_path, transport)
    inventory.capture_prime_inventory_raw_v5()
    if mutation == "path":
        fake = tmp_path / "ssh-keygen.exe"
        fake.write_bytes(b"not-openssh")
        monkeypatch.setattr(inventory, "OPENSSH_EXECUTABLE_PATH", fake)
    elif mutation == "bytes":
        monkeypatch.setattr(inventory, "OPENSSH_EXECUTABLE_SHA256", "0" * 64)
    elif mutation == "link":
        original = inventory_v3._is_link_or_reparse
        monkeypatch.setattr(
            inventory_v3,
            "_is_link_or_reparse",
            lambda path: path == inventory.OPENSSH_EXECUTABLE_PATH or original(path),
        )
    else:
        monkeypatch.setattr(inventory, "OPENSSH_HARDLINK_COUNT", 1)
    monkeypatch.setattr(
        inventory,
        "_verify_signature",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("signature verification must not run")
        ),
    )
    monkeypatch.setattr(
        inventory,
        "_replay_transcript",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("semantic replay must not run")
        ),
    )
    with pytest.raises(ValueError, match="OpenSSH"):
        inventory.assess_prime_inventory_v5()
    assert not (tmp_path / inventory.ASSESSMENT_RELATIVE).exists()


def test_wrong_openssh_bytes_fail_before_challenge_claim_or_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(0)
    key_path = _bind_capture(monkeypatch, tmp_path, transport)
    monkeypatch.setattr(inventory, "OPENSSH_EXECUTABLE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="executable bytes differ"):
        _CAPTURE_OWNER(key_path)
    assert not transport.calls
    assert not (tmp_path / inventory.CLAIM_RELATIVE).exists()


def test_coherent_unsigned_transcript_and_raw_rewrite_fails_before_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(1)
    _bind_capture(monkeypatch, tmp_path, transport)
    inventory.capture_prime_inventory_raw_v5()
    raw_path = tmp_path / inventory.RAW_RELATIVE
    transcript_path = tmp_path / inventory.TRANSCRIPT_RELATIVE
    raw = cast(dict[str, object], json.loads(raw_path.read_bytes()))
    pages = cast(list[dict[str, object]], raw["pages"])
    body = json.loads(base64.b64decode(cast(str, pages[0]["decoded_application_body_b64"])))
    body["items"][0]["provider"] = "coherent-forgery"
    _replace_page_body(pages[0], body)
    transcript_payload = inventory._transcript_payload(
        pages,
        cast(str | None, raw["diagnostic"]),
        cast(dict[str, object] | None, raw["failure"]),
        cast(int, raw["request_count"]),
    )
    transcript = cast(dict[str, object], json.loads(transcript_path.read_bytes()))
    transcript["transcript_sha256"] = sha256_bytes(
        canonical_json_bytes(transcript_payload)
    )
    transcript_bytes = canonical_json_bytes(transcript)
    transcript_path.write_bytes(transcript_bytes)
    raw["transcript"] = {
        "path": inventory.TRANSCRIPT_RELATIVE,
        "sha256": sha256_bytes(transcript_bytes),
    }
    raw_bytes = canonical_json_bytes(raw)
    raw_path.write_bytes(raw_bytes)
    envelope, payload = _auth_envelope(tmp_path)
    cast(dict[str, object], payload["transcript"])["artifact_sha256"] = sha256_bytes(
        transcript_bytes
    )
    cast(dict[str, object], payload["transcript"])["payload_sha256"] = sha256_bytes(
        canonical_json_bytes(transcript_payload)
    )
    cast(dict[str, object], payload["raw"])["artifact_sha256"] = sha256_bytes(raw_bytes)
    _write_auth_envelope(tmp_path, envelope, payload)
    monkeypatch.setattr(
        inventory,
        "_replay_transcript",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("semantic replay must not run before signature authentication")
        ),
    )
    with pytest.raises(ValueError, match="OpenSSH operation failed"):
        inventory.assess_prime_inventory_v5()
    assert not (tmp_path / inventory.ASSESSMENT_RELATIVE).exists()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mutation",
    [
        "claim",
        "commit",
        "tree",
        "owner",
        "request_count",
        "diagnostic",
        "authority",
        "retry",
        "attempt",
        "transcript_hash",
        "raw_hash",
        "principal",
        "namespace",
        "fingerprint",
        "missing",
        "extra",
    ],
)
def test_resigned_terminal_payload_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    transport = _transport(1)
    key_path = _bind_capture(monkeypatch, tmp_path, transport)
    inventory.capture_prime_inventory_raw_v5()
    envelope, payload = _auth_envelope(tmp_path)
    if mutation == "claim":
        cast(dict[str, object], payload["claim"])["sha256"] = "0" * 64
    elif mutation in {"commit", "tree"}:
        cast(dict[str, object], payload["capture_checkpoint"])[mutation] = "0" * 40
    elif mutation == "owner":
        payload["capture_owner_projection_sha256"] = "0" * 64
    elif mutation == "request_count":
        payload["request_count"] = 9
    elif mutation == "diagnostic":
        payload["terminal_diagnostic"] = "transport_failure"
    elif mutation == "authority":
        cast(dict[str, object], payload["authorization"])["prime_authorized"] = True
    elif mutation == "retry":
        payload["retry"] = True
    elif mutation == "attempt":
        payload["attempt_consumed"] = False
    elif mutation == "transcript_hash":
        cast(dict[str, object], payload["transcript"])["artifact_sha256"] = "0" * 64
    elif mutation == "raw_hash":
        cast(dict[str, object], payload["raw"])["artifact_sha256"] = "0" * 64
    elif mutation in {"principal", "fingerprint"}:
        cast(dict[str, object], payload["signing"])[
            "fingerprint_sha256" if mutation == "fingerprint" else mutation
        ] = "wrong"
    elif mutation == "namespace":
        cast(dict[str, object], payload["signing"])["namespace"] = "wrong"
    elif mutation == "missing":
        del payload["raw_capture_state"]
    else:
        payload["unknown"] = None
    _write_auth_envelope(tmp_path, envelope, payload, signing_key=key_path)
    with pytest.raises(ValueError):
        inventory.assess_prime_inventory_v5()
    assert not (tmp_path / inventory.ASSESSMENT_RELATIVE).exists()


def test_wrong_signature_key_replay_base64_and_expiry_fail_before_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(1)
    key_path = _bind_capture(monkeypatch, tmp_path, transport)
    inventory.capture_prime_inventory_raw_v5()
    auth_path = tmp_path / inventory.TERMINAL_AUTH_RELATIVE
    original = auth_path.read_bytes()
    envelope, payload = _auth_envelope(tmp_path)

    payload["request_count"] = 99
    _write_auth_envelope(tmp_path, envelope, payload)
    with pytest.raises(ValueError, match="OpenSSH operation failed"):
        inventory.assess_prime_inventory_v5()

    auth_path.write_bytes(original)
    envelope, _payload = _auth_envelope(tmp_path)
    cast(dict[str, object], envelope["signature"])["base64"] = "not-base64"
    auth_path.write_bytes(canonical_json_bytes(envelope))
    with pytest.raises(ValueError, match="signature encoding"):
        inventory.assess_prime_inventory_v5()

    auth_path.write_bytes(original)
    wrong_key = tmp_path / "wrong-key"
    subprocess.run(
        [
            str(inventory.OPENSSH_EXECUTABLE_PATH),
            "-q",
            "-t",
            "rsa",
            "-b",
            "2048",
            "-N",
            "",
            "-f",
            str(wrong_key),
        ],
        check=True,
        capture_output=True,
    )
    envelope, payload = _auth_envelope(tmp_path)
    _write_auth_envelope(tmp_path, envelope, payload, signing_key=wrong_key)
    with pytest.raises(ValueError, match="OpenSSH operation failed"):
        inventory.assess_prime_inventory_v5()

    auth_path.write_bytes(original)
    monkeypatch.setattr(time, "time", lambda: 2_000)
    with pytest.raises(ValueError, match="payload binding differs"):
        inventory.assess_prime_inventory_v5()
    assert key_path.is_file()


def test_wrong_operator_key_and_auth_publication_failure_are_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(0)
    _bind_capture(monkeypatch, tmp_path, transport)
    wrong_key = tmp_path / "wrong-operator"
    subprocess.run(
        [
            str(inventory.OPENSSH_EXECUTABLE_PATH),
            "-q",
            "-t",
            "rsa",
            "-b",
            "2048",
            "-N",
            "",
            "-f",
            str(wrong_key),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="identity differs"):
        _CAPTURE_OWNER(wrong_key)
    assert not transport.calls
    assert not (tmp_path / inventory.CLAIM_RELATIVE).exists()

    original_publish = inventory._publish_fixed

    def fail_auth(relative: str, raw: bytes) -> Path:
        if relative == inventory.TERMINAL_AUTH_RELATIVE:
            raise OSError("auth publication fixture failure")
        return original_publish(relative, raw)

    monkeypatch.setattr(inventory, "_publish_fixed", fail_auth)
    with pytest.raises(OSError, match="auth publication"):
        inventory.capture_prime_inventory_raw_v5()
    assert (tmp_path / inventory.TRANSCRIPT_RELATIVE).is_file()
    assert (tmp_path / inventory.RAW_RELATIVE).is_file()
    assert not (tmp_path / inventory.TERMINAL_AUTH_RELATIVE).exists()
    assert (tmp_path / inventory.TERMINAL_RELATIVE).is_file()
    with pytest.raises(ValueError, match="forbids assessment"):
        inventory.assess_prime_inventory_v5()


def test_terminal_signing_failure_consumes_claim_without_unsigned_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(0)
    _bind_capture(monkeypatch, tmp_path, transport)
    original_sign = inventory._sign_bytes

    def fail_terminal_sign(path: Path, value: bytes) -> bytes:
        if inventory.TERMINAL_AUTH_PAYLOAD_DOMAIN.encode("ascii") in value:
            raise ValueError("terminal signing fixture failure")
        return original_sign(path, value)

    monkeypatch.setattr(inventory, "_sign_bytes", fail_terminal_sign)
    with pytest.raises(ValueError, match="terminal signing"):
        inventory.capture_prime_inventory_raw_v5()
    assert len(transport.calls) == 2
    assert (tmp_path / inventory.CLAIM_RELATIVE).is_file()
    assert (tmp_path / inventory.TERMINAL_RELATIVE).is_file()
    assert not (tmp_path / inventory.TRANSCRIPT_RELATIVE).exists()
    assert not (tmp_path / inventory.RAW_RELATIVE).exists()
    assert not (tmp_path / inventory.TERMINAL_AUTH_RELATIVE).exists()
    assert not (tmp_path / inventory.ASSESSMENT_RELATIVE).exists()


def test_cancellation_and_size_limits_are_terminal_without_later_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class CancellingTransport(_FakeTransport):
        def request(
            self,
            method: str,
            url: str,
            *,
            params: Mapping[str, object],
            follow_redirects: bool,
        ) -> _Response:
            self.calls.append((method, url, dict(params), follow_redirects))
            raise KeyboardInterrupt

    cancelling = CancellingTransport(
        {inventory.ENDPOINTS[0]: [], inventory.ENDPOINTS[1]: []}
    )
    cancel_root = tmp_path / "cancel"
    cancel_root.mkdir()
    _bind_capture(monkeypatch, cancel_root, cancelling)
    cancelled = json.loads(inventory.capture_prime_inventory_raw_v5())
    assert cancelled["diagnostic"] == "capture_cancelled"
    assert len(cancelling.calls) == 1

    class OversizedTransport(_FakeTransport):
        def request(
            self,
            method: str,
            url: str,
            *,
            params: Mapping[str, object],
            follow_redirects: bool,
        ) -> _Response:
            self.calls.append((method, url, dict(params), follow_redirects))
            return _Response(
                200,
                b"x" * (inventory.MAX_BODY_BYTES + 1),
                {"content-type": "application/json"},
            )

    oversized = OversizedTransport(
        {inventory.ENDPOINTS[0]: [], inventory.ENDPOINTS[1]: []}
    )
    size_root = tmp_path / "oversized"
    size_root.mkdir()
    _bind_capture(monkeypatch, size_root, oversized)
    size_receipt = json.loads(inventory.capture_prime_inventory_raw_v5())
    assert size_receipt["diagnostic"] == "response_body_too_large"
    assert len(oversized.calls) == 1

    cumulative = _transport(0)
    cumulative_root = tmp_path / "cumulative"
    cumulative_root.mkdir()
    _bind_capture(monkeypatch, cumulative_root, cumulative)
    monkeypatch.setattr(inventory, "MAX_CUMULATIVE_BODY_BYTES", 1)
    cumulative_receipt = json.loads(inventory.capture_prime_inventory_raw_v5())
    assert cumulative_receipt["diagnostic"] == "cumulative_body_too_large"
    assert len(cumulative.calls) == 1


def test_empty_early_and_second_endpoint_items_are_owned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class EmptyEarlyTransport(_FakeTransport):
        def request(
            self,
            method: str,
            url: str,
            *,
            params: Mapping[str, object],
            follow_redirects: bool,
        ) -> _Response:
            self.calls.append((method, url, dict(params), follow_redirects))
            body = canonical_json_bytes({"items": [], "totalCount": 1})
            return _Response(200, body, {"content-type": "application/json"})

    empty = EmptyEarlyTransport(
        {inventory.ENDPOINTS[0]: [], inventory.ENDPOINTS[1]: []}
    )
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    _bind_capture(monkeypatch, empty_root, empty)
    receipt = json.loads(inventory.capture_prime_inventory_raw_v5())
    assert receipt["diagnostic"] == "empty_page_before_total"
    assert len(empty.calls) == 1

    second = _transport(0, second_count=1)
    second_root = tmp_path / "second"
    second_root.mkdir()
    _bind_capture(monkeypatch, second_root, second)
    inventory.capture_prime_inventory_raw_v5()
    assessment = json.loads(inventory.assess_prime_inventory_v5())
    assert assessment["state"] == "observed_non_authorizing_resource"
    assert assessment["rows"][0]["provenance"]["endpoint"] == inventory.ENDPOINTS[1]


def test_changed_total_duplicate_and_empty_early_fail_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    changed = _transport(101, total_delta_call=2)
    (tmp_path / "changed").mkdir()
    _bind_capture(monkeypatch, tmp_path / "changed", changed)
    assert json.loads(inventory.capture_prime_inventory_raw_v5())["diagnostic"] == (
        "changed_total_count"
    )

    duplicate_items = [_item(index) for index in range(100)] + [_item(0)]
    duplicate = _FakeTransport(
        {inventory.ENDPOINTS[0]: duplicate_items, inventory.ENDPOINTS[1]: []}
    )
    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    _bind_capture(monkeypatch, duplicate_root, duplicate)
    assert json.loads(inventory.capture_prime_inventory_raw_v5())["diagnostic"] == (
        "duplicate_canonical_item"
    )


def test_preflight_claim_replay_env_and_config_fail_before_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(0)
    _bind_capture(monkeypatch, tmp_path, transport)
    claim = tmp_path / inventory.CLAIM_RELATIVE
    claim.parent.mkdir(parents=True)
    claim.write_bytes(b"preexisting")
    with pytest.raises(FileExistsError):
        inventory.capture_prime_inventory_raw_v5()
    assert not transport.calls

    claim.unlink()
    monkeypatch.setenv("PRIME_CONTEXT", "shadow")
    with pytest.raises(ValueError, match="environment overrides"):
        inventory.capture_prime_inventory_raw_v5()
    assert not transport.calls
    monkeypatch.delenv("PRIME_CONTEXT")

    monkeypatch.setattr(
        inventory,
        "_config_paths",
        lambda: (tmp_path / "missing", tmp_path / "missing/config.json", tmp_path / "missing/env"),
    )
    with pytest.raises(ValueError, match="config path"):
        inventory.capture_prime_inventory_raw_v5()
    assert not transport.calls


def test_concurrent_claim_race_allows_one_capture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(0)
    _bind_capture(monkeypatch, tmp_path, transport)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(inventory.capture_prime_inventory_raw_v5) for _ in range(2)]
        outcomes: list[bytes | BaseException] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except FileExistsError as error:
                outcomes.append(error)
    assert sum(isinstance(outcome, bytes) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
    assert len(transport.calls) == 2


def test_path_alias_and_linked_ancestor_fail_before_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(0)
    _bind_capture(monkeypatch, tmp_path, transport)
    parent = tmp_path / Path(inventory.CLAIM_RELATIVE).parent
    original = inventory_v3._is_link_or_reparse
    monkeypatch.setattr(
        inventory_v3,
        "_is_link_or_reparse",
        lambda path: path == parent or original(path),
    )
    with pytest.raises(ValueError, match="linked or reparse"):
        inventory.capture_prime_inventory_raw_v5()
    assert not transport.calls


def test_body_hash_tamper_ttl_and_atomic_no_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transport = _transport(1)
    _bind_capture(monkeypatch, tmp_path, transport)
    inventory.capture_prime_inventory_raw_v5()
    with pytest.raises(FileExistsError):
        inventory.capture_prime_inventory_raw_v5()
    assert len(transport.calls) == 2
    raw_path = tmp_path / inventory.RAW_RELATIVE
    alias = tmp_path / "raw-hardlink.json"
    os.link(raw_path, alias)
    with pytest.raises(ValueError, match="aliased"):
        inventory.assess_prime_inventory_v5()
    alias.unlink()
    monkeypatch.setattr(time, "time", lambda: 2_000)
    with pytest.raises(ValueError, match="binding differs"):
        inventory.assess_prime_inventory_v5()
    monkeypatch.setattr(time, "time", lambda: 1_000)
    value = json.loads(raw_path.read_bytes())
    value["pages"][0]["decoded_application_body_sha256"] = "0" * 64
    raw_path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError, match="signed raw artifact differs"):
        inventory.assess_prime_inventory_v5()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("case", "expected_pages", "expected_requests"),
    [("zero", 0, 1), ("one", 1, 2), ("multiple", 2, 2)],
)
def test_primary_raw_publication_failure_is_terminal_and_non_reusable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    expected_pages: int,
    expected_requests: int,
) -> None:
    if case == "zero":
        transport = _transport(101, fail_call=1)
    elif case == "one":
        transport = _transport(101, fail_call=2)
    else:
        transport = _transport(0)
    _bind_capture(monkeypatch, tmp_path, transport)
    original_publish = inventory._publish_fixed

    def fail_primary(relative: str, raw: bytes) -> Path:
        if relative == inventory.RAW_RELATIVE:
            raise OSError("primary raw publication fixture failure")
        return original_publish(relative, raw)

    monkeypatch.setattr(inventory, "_publish_fixed", fail_primary)
    with pytest.raises(OSError, match="primary raw publication"):
        inventory.capture_prime_inventory_raw_v5()
    terminal_path = tmp_path / inventory.TERMINAL_RELATIVE
    terminal = json.loads(terminal_path.read_bytes())
    assert terminal["state"] == "raw_publication_failed_terminal"
    assert terminal["captured_page_count"] == expected_pages
    assert terminal["request_count"] == expected_requests
    assert terminal["retry"] is False
    assert terminal["assessment_allowed"] is False
    assert not (tmp_path / inventory.RAW_RELATIVE).exists()
    with pytest.raises(ValueError, match="forbids assessment"):
        inventory.assess_prime_inventory_v5()
    calls = len(transport.calls)
    with pytest.raises(FileExistsError):
        inventory.capture_prime_inventory_raw_v5()
    assert len(transport.calls) == calls


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "mode", ["partial", "alias"]
)
def test_partial_or_aliased_primary_raw_is_never_repaired_or_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    transport = _transport(0)
    _bind_capture(monkeypatch, tmp_path, transport)
    original_publish = inventory._publish_fixed
    victim = tmp_path / "preexisting-victim.json"
    victim.write_bytes(b"preexisting-victim")

    def alias_then_fail(relative: str, raw: bytes) -> Path:
        if relative == inventory.RAW_RELATIVE:
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if mode == "alias":
                os.link(victim, path)
            else:
                path.write_bytes(b"partial-primary")
            raise OSError("primary raw partial-or-alias fixture failure")
        return original_publish(relative, raw)

    monkeypatch.setattr(inventory, "_publish_fixed", alias_then_fail)
    with pytest.raises(OSError, match="primary raw partial-or-alias"):
        inventory.capture_prime_inventory_raw_v5()
    raw_path = tmp_path / inventory.RAW_RELATIVE
    if mode == "alias":
        assert raw_path.samefile(victim)
        assert raw_path.read_bytes() == b"preexisting-victim"
    else:
        assert raw_path.read_bytes() == b"partial-primary"
    assert victim.read_bytes() == b"preexisting-victim"
    assert (tmp_path / inventory.TERMINAL_RELATIVE).is_file()
    with pytest.raises(ValueError, match="forbids assessment"):
        inventory.assess_prime_inventory_v5()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "relative",
    [
        inventory.CLAIM_RELATIVE,
        inventory.TRANSCRIPT_RELATIVE,
        inventory.RAW_RELATIVE,
        inventory.ASSESSMENT_RELATIVE,
        inventory.TERMINAL_RELATIVE,
        inventory.TERMINAL_AUTH_RELATIVE,
    ],
)
def test_every_fixed_path_is_prevalidated_before_claim_or_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative: str,
) -> None:
    transport = _transport(0)
    _bind_capture(monkeypatch, tmp_path, transport)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"preexisting")
    with pytest.raises(FileExistsError):
        inventory.capture_prime_inventory_raw_v5()
    assert not transport.calls
    if relative != inventory.CLAIM_RELATIVE:
        assert not (tmp_path / inventory.CLAIM_RELATIVE).exists()


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "spot", [True, None, "false", 0]
)
def test_only_literal_false_spot_qualifies(spot: object) -> None:
    assessed = inventory._assess_item(
        _item(1, isSpot=spot), {"endpoint": "fixture", "page": 1, "item_ordinal": 0}
    )
    assert assessed["eligible"] is False
    assert "non_spot_not_proven" in cast(list[str], assessed["reasons"])


def test_price_disk_memory_and_label_laws_are_exact() -> None:
    mutations: tuple[tuple[dict[str, object], str], ...] = (
        ({"gpuCount": True}, "gpu_count_not_two"),
        ({"gpuMemory": 95}, "aggregate_gpu_memory_not_96"),
        ({"gpuType": "L40S_48GB"}, "gpu_label_not_allowed"),
        ({"stockStatus": "unknown"}, "stock_not_available"),
        ({"disk": {"defaultCount": 1250}}, "disk_schema_unknown"),
        (
            {
                "prices": {
                    "onDemand": 2.01,
                    "communityPrice": None,
                    "isVariable": False,
                    "currency": "USD",
                }
            },
            "hourly_rate_above_cap",
        ),
    )
    for changes, expected_reason in mutations:
        assessed = inventory._assess_item(
            _item(1, **changes),
            {"endpoint": "fixture", "page": 1, "item_ordinal": 0},
        )
        assert assessed["eligible"] is False
        assert expected_reason in cast(list[str], assessed["reasons"])


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    ("community", "eligible", "expected_rate"),
    [
        (None, True, 1.25),
        (1.50, True, 1.50),
        (0, False, None),
        (-1, False, None),
        ("1.50", False, None),
        (True, False, None),
        ({"amount": 1.50}, False, None),
        (2.01, False, 2.01),
    ],
)
def test_community_price_only_literal_none_permits_on_demand_fallback(
    community: object,
    eligible: bool,
    expected_rate: float | None,
) -> None:
    prices = {
        "onDemand": 1.25,
        "communityPrice": community,
        "isVariable": False,
        "currency": "USD",
    }
    assessed = inventory._assess_item(
        _item(1, prices=prices),
        {"endpoint": "fixture", "page": 1, "item_ordinal": 0},
    )
    assert assessed["eligible"] is eligible
    assert assessed["hourly_rate_usd"] == expected_rate
    if not eligible:
        reasons = cast(list[str], assessed["reasons"])
        assert (
            "positive_hourly_rate_not_proven" in reasons
            or "hourly_rate_above_cap" in reasons
        )


def test_unknown_key_is_row_local_and_duplicate_cloud_is_ambiguous() -> None:
    unknown = _item(1, futureField=None)
    row = inventory._assess_item(
        unknown, {"endpoint": "fixture", "page": 1, "item_ordinal": 0}
    )
    assert row["eligible"] is False
    assert row["reasons"] == ["unknown_item_keys"]

    body = canonical_json_bytes(
        {
            "items": [_item(1), _item(2, cloudId="cloud-1", provider="other")],
            "totalCount": 2,
        }
    )
    record = {
        "endpoint": inventory.ENDPOINTS[0],
        "params": {"gpu_count": "2", "page": 1, "page_size": 100},
        "status": 200,
        "content_type": "application/json",
        "decoded_application_body_b64": base64.b64encode(body).decode(),
        "decoded_application_body_sha256": sha256_bytes(body),
        "decoded_application_body_bytes": len(body),
        "page_ordinal": 1,
    }
    empty_body = canonical_json_bytes({"items": [], "totalCount": 0})
    empty_record = {
        "endpoint": inventory.ENDPOINTS[1],
        "params": {"gpu_count": "2", "page": 1, "page_size": 100},
        "status": 200,
        "content_type": "application/json",
        "decoded_application_body_b64": base64.b64encode(empty_body).decode(),
        "decoded_application_body_sha256": sha256_bytes(empty_body),
        "decoded_application_body_bytes": len(empty_body),
        "page_ordinal": 1,
    }
    value = inventory._assessment_value(
        {
            "state": "captured_endpoint_terminal",
            "pages": [record, empty_record],
            "diagnostic": None,
            "failure": None,
            "request_count": 2,
        },
        "f" * 64,
    )
    assert value["state"] == "observed_ambiguous_resources"


def test_source_owners_and_forbidden_capture_layers_are_bound() -> None:
    owners = inventory.authenticate_installed_capture_owners()
    client = cast(dict[str, object], owners["core_client"])
    config = cast(dict[str, object], owners["config_owner"])
    api = cast(dict[str, object], owners["availability_api"])
    httpx = cast(dict[str, object], owners["httpx"])
    assert client["sha256"] == inventory.CLIENT_SOURCE_SHA256
    assert config["sha256"] == inventory.CONFIG_SOURCE_SHA256
    assert api["sha256"] == inventory.API_SOURCE_SHA256
    assert httpx["version"] == "0.28.1"
    names = set(inventory._capture_pages.__code__.co_names)
    assert "AvailabilityClient" not in names
    assert "GPUAvailability" not in names
    assert "subprocess" not in names


def test_v1_through_v4_immutable_and_v5_build_is_deterministic() -> None:
    assert {
        relative: sha256_bytes((ROOT / relative).read_bytes())
        for relative in inventory.HISTORICAL_BINDINGS
    } == inventory.HISTORICAL_BINDINGS
    first = inventory.build_prime_inventory_v5_artifacts(ROOT)
    second = inventory.build_prime_inventory_v5_artifacts(ROOT)
    assert first == second
    contract = json.loads(first[inventory.CONTRACT_RELATIVE])
    audit = json.loads(first[inventory.AUDIT_RELATIVE])
    assert contract["endpoint_contract"]["endpoints"] == list(inventory.ENDPOINTS)
    assert contract["endpoint_contract"]["cli_and_pydantic_capture_forbidden"] is True
    terminal_auth = cast(dict[str, object], contract["terminal_authentication"])
    assert terminal_auth["operator_host_only"] == "windows"
    assert terminal_auth["linux_fallback_allowed"] is False
    assert terminal_auth["openssh_executable"] == (
        inventory.authenticate_approved_openssh_executable()
    )
    assert all(value is False for value in contract["authorization"].values())
    assert all(value is False for value in audit["authorization"].values())
    serialized = first[inventory.CONTRACT_RELATIVE] + first[inventory.AUDIT_RELATIVE]
    for forbidden in (b"wallet_id", b"cloudId", b"stdout", b"api_key"):
        assert forbidden not in serialized


def test_no_shared_v5_live_artifact_exists() -> None:
    for relative in (
        inventory.CLAIM_RELATIVE,
        inventory.TRANSCRIPT_RELATIVE,
        inventory.RAW_RELATIVE,
        inventory.ASSESSMENT_RELATIVE,
        inventory.TERMINAL_RELATIVE,
        inventory.TERMINAL_AUTH_RELATIVE,
    ):
        assert not (ROOT / relative).exists()
