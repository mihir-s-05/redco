"""Canonical authority-aware handoff and known-hosts owner for the one-shot."""

from __future__ import annotations

import base64
import binascii
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from redco.analysis.stage_d_v13_prime_test_one_shot_contract_v2 import (
    ARTIFACT_FILENAMES,
    ASSESSMENT_TTL_SECONDS,
    GPU_TELEMETRY_BINDING,
    HANDOFF_DOMAIN,
    HANDOFF_NAMESPACE,
    POD_NAME_PREFIX,
    READINESS_AUTHORITY,
    TEST_NODES,
    authority_value,
    canonical_json,
    closed_authority,
    sha256_bytes,
    strict_object,
)

LINUX_UV_SOURCE = "/home/mihir/.local/uv-latest/uv"
LINUX_UV_BYTES = 66_081_208
LINUX_UV_SHA256 = "da15297d6879b2cfbe5ea3cb03725c1613d51ba72892cc996468d871f0a532fb"
MAX_KNOWN_HOSTS_BYTES = 64 * 1024
_HEX64 = re.compile(r"[0-9a-f]{64}")
_SSH_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,31}")
_HOST_KEY_ALGORITHMS = frozenset(
    {"ssh-ed25519", "ssh-rsa", "ecdsa-sha2-nistp256", "ecdsa-sha2-nistp384"}
)
_ECDSA_CURVES = {
    "ecdsa-sha2-nistp256": (
        "nistp256",
        65,
        int("FFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF", 16),
        int("5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B", 16),
    ),
    "ecdsa-sha2-nistp384": (
        "nistp384",
        97,
        int(
            "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"
            "FFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFF", 16
        ),
        int(
            "B3312FA7E23EE7E4988E056BE3F82D19181D9C6EFE814112"
            "0314088F5013875AC656398D8A2ED19D2A85C8EDD3EC2AEF", 16
        ),
    ),
}
_RSA_MAX_EXPONENT, _RSA_MIN_BITS, _RSA_MAX_BITS = 2**32 - 1, 2048, 16384


@dataclass(frozen=True, slots=True)
class HandoffSummary:
    pod_identity_sha256: str
    pod_name: str
    pod_status_sha256: str
    ssh_user: str | None
    ssh_host_sha256: str
    ssh_port: int


def _hash(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"Prime one-shot {label} differs")
    return value


def _object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(cast(dict[object, object], value)) != keys:
        raise ValueError(f"Prime one-shot {label} schema differs")
    return cast(dict[str, object], value)


def _ssh_field(blob: bytes, offset: int) -> tuple[bytes, int]:
    if offset > len(blob) - 4:
        raise ValueError("Prime one-shot SSH key field is truncated")
    length = struct.unpack_from(">I", blob, offset)[0]
    end = offset + 4 + length
    if length > len(blob) or end > len(blob):
        raise ValueError("Prime one-shot SSH key field length differs")
    return blob[offset + 4 : end], end


def _positive_mpint(value: bytes) -> int:
    if not value or value[0] & 0x80 or (number := int.from_bytes(value, "big")) <= 0:
        raise ValueError("Prime one-shot SSH mpint differs")
    if len(value) > 1 and value[0] == 0 and not value[1] & 0x80:
        raise ValueError("Prime one-shot SSH mpint is not canonical")
    return number


def _validate_curve_point(point: bytes, size: int, prime: int, coefficient: int) -> None:
    width = (size - 1) // 2
    if len(point) != size or point[:1] != b"\x04":
        raise ValueError("Prime one-shot ECDSA key structure differs")
    x = int.from_bytes(point[1 : width + 1], "big")
    y = int.from_bytes(point[width + 1 :], "big")
    if x >= prime or y >= prime or (
        pow(y, 2, prime) - pow(x, 3, prime) + 3 * x - coefficient
    ) % prime:
        raise ValueError("Prime one-shot ECDSA point is not on curve")


def _validate_ssh_key_blob(blob: bytes, algorithm: str) -> None:
    key_type, offset = _ssh_field(blob, 0)
    if key_type != algorithm.encode():
        raise ValueError("Prime one-shot SSH key type differs")
    if algorithm == "ssh-rsa":
        exponent, offset = _ssh_field(blob, offset)
        modulus, offset = _ssh_field(blob, offset)
        exponent_value = _positive_mpint(exponent)
        modulus_value = _positive_mpint(modulus)
        if exponent_value < 3 or exponent_value % 2 == 0 or exponent_value > _RSA_MAX_EXPONENT:
            raise ValueError("Prime one-shot RSA exponent is weak")
        if (
            modulus_value % 2 == 0
            or not _RSA_MIN_BITS <= modulus_value.bit_length() <= _RSA_MAX_BITS
        ):
            raise ValueError("Prime one-shot RSA modulus is weak")
    elif algorithm == "ssh-ed25519":
        key, offset = _ssh_field(blob, offset)
        if len(key) != 32:
            raise ValueError("Prime one-shot Ed25519 key length differs")
    else:
        curve_name, offset = _ssh_field(blob, offset)
        point, offset = _ssh_field(blob, offset)
        expected_curve, expected_length, prime, coefficient = _ECDSA_CURVES[algorithm]
        if curve_name != expected_curve.encode():
            raise ValueError("Prime one-shot ECDSA key structure differs")
        _validate_curve_point(point, expected_length, prime, coefficient)
    if offset != len(blob):
        raise ValueError("Prime one-shot SSH key has trailing bytes")


def validate_known_hosts(raw: bytes, host_sha256: str, port: int) -> None:
    if not raw or len(raw) > MAX_KNOWN_HOSTS_BYTES or not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("Prime one-shot known-hosts bytes differ")
    try:
        lines = raw.decode("ascii").removesuffix("\n").split("\n")
    except UnicodeDecodeError as error:
        raise ValueError("Prime one-shot known-hosts encoding differs") from error
    if len(lines) != len(set(lines)):
        raise ValueError("Prime one-shot known-hosts lines overlap")
    matched = False
    for line in lines:
        if any(ord(character) < 32 or ord(character) == 127 for character in line):
            raise ValueError("Prime one-shot known-hosts control differs")
        fields = line.split(" ")
        if len(fields) != 3 or any(not field for field in fields):
            raise ValueError("Prime one-shot known-hosts line differs")
        host_port, algorithm, encoded_key = fields
        match = re.fullmatch(r"\[([^\[\],\s]+)\]:([0-9]{1,5})", host_port)
        if match is None or algorithm not in _HOST_KEY_ALGORITHMS:
            raise ValueError("Prime one-shot known-hosts field differs")
        parsed_port = int(match.group(2))
        try:
            key = base64.b64decode(encoded_key, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Prime one-shot known-hosts key differs") from error
        if not key or len(key) > 16 * 1024 or base64.b64encode(key).decode() != encoded_key:
            raise ValueError("Prime one-shot known-hosts key differs")
        _validate_ssh_key_blob(key, algorithm)
        matched |= parsed_port == port and sha256_bytes(match.group(1).encode()) == host_sha256
    if not matched:
        raise ValueError("Prime one-shot known-hosts endpoint differs")


def validate_handoff_payload(
    raw: bytes,
    *,
    authorization: Mapping[str, str],
    claim_sha256: str,
    transcript_sha256: str,
    assessment_sha256: str,
    assessment_envelope_sha256: str,
    selected_resource_sha256: str,
    selected_facts: object,
    known_hosts: bytes,
    test_script: bytes,
    authority: Mapping[str, bool] = READINESS_AUTHORITY,
) -> HandoffSummary:
    authority = closed_authority(authority, "handoff", readiness=True)
    value = strict_object(
        raw,
        {
            "schema_version", "domain", "state", "authorization", "claim", "transcript",
            "assessment", "selected_resource_sha256", "selected_facts", "pod", "ssh",
            "runtime", "evidence_paths", "nonce", "issued_at_epoch", "expires_at_epoch",
            "attempt_consumed", "retry", "authority",
        },
        "Prime one-shot handoff",
    )
    claim = _object(value["claim"], {"path", "sha256"}, "handoff claim")
    transcript = _object(value["transcript"], {"path", "sha256"}, "handoff transcript")
    assessment = _object(
        value["assessment"], {"path", "sha256", "envelope_sha256"}, "handoff assessment"
    )
    pod = _object(value["pod"], {"identity_sha256", "name", "status_sha256"}, "handoff pod")
    ssh = _object(
        value["ssh"], {"user", "host_sha256", "port", "known_hosts_sha256"}, "handoff SSH"
    )
    runtime = _object(
        value["runtime"],
        {"test_script_sha256", "linux_uv_sha256", "test_nodes", "gpu_probe", "gpu_telemetry"},
        "handoff runtime",
    )
    commit = authorization["commit"]
    issued, expires = value["issued_at_epoch"], value["expires_at_epoch"]
    user, port = ssh["user"], ssh["port"]
    if (
        value["schema_version"] != 2
        or value["domain"] != HANDOFF_DOMAIN
        or value["state"] != "pod_bound_one_use"
        or value["authorization"] != dict(authorization)
        or claim != {"path": ARTIFACT_FILENAMES["claim"], "sha256": claim_sha256}
        or transcript != {"path": ARTIFACT_FILENAMES["transcript"], "sha256": transcript_sha256}
        or assessment != {
            "path": ARTIFACT_FILENAMES["assessment"], "sha256": assessment_sha256,
            "envelope_sha256": assessment_envelope_sha256,
        }
        or value["selected_resource_sha256"] != selected_resource_sha256
        or value["selected_facts"] != selected_facts
        or pod["name"] != f"{POD_NAME_PREFIX}-{commit[:12]}"
        or type(user) not in {str, type(None)}
        or (type(user) is str and _SSH_USER.fullmatch(user) is None)
        or type(port) is not int
        or not 1 <= port <= 65535
        or ssh["known_hosts_sha256"] != sha256_bytes(known_hosts)
        or runtime != {
             "test_script_sha256": sha256_bytes(test_script),
            "linux_uv_sha256": LINUX_UV_SHA256,
            "test_nodes": list(TEST_NODES),
            "gpu_probe": "exact allowed class, two devices, aggregate 96GB bounds",
            "gpu_telemetry": GPU_TELEMETRY_BINDING,
        }
        or value["evidence_paths"] != {
            name: filename for name, filename in sorted(ARTIFACT_FILENAMES.items())
        }
        or type(value["nonce"]) is not str
        or _HEX64.fullmatch(value["nonce"]) is None
        or type(issued) is not int
        or issued < 0
        or type(expires) is not int
        or expires != issued + ASSESSMENT_TTL_SECONDS
        or value["attempt_consumed"] is not True
        or value["retry"] is not False
    ):
        raise ValueError("Prime one-shot handoff binding differs")
    authority_value(value["authority"], authority, "handoff")
    identity_hash = _hash(pod["identity_sha256"], "handoff pod identity")
    status_hash = _hash(pod["status_sha256"], "handoff pod status")
    host_hash = _hash(ssh["host_sha256"], "handoff host")
    normalized_user = user if type(user) is str else None
    validate_known_hosts(known_hosts, host_hash, port)
    return HandoffSummary(identity_hash, pod["name"], status_hash, normalized_user, host_hash, port)


def build_handoff_payload(
    *,
    authorization: Mapping[str, str], claim_sha256: str, transcript_sha256: str,
    assessment_sha256: str, assessment_envelope_sha256: str,
    selected_resource_sha256: str, selected_facts: object, pod_identity_sha256: str,
    pod_status_sha256: str, ssh_user: str | None, ssh_host: str, ssh_port: int,
    known_hosts: bytes, test_script: bytes, nonce: str, issued_at_epoch: int,
    authority: Mapping[str, bool] = READINESS_AUTHORITY,
) -> tuple[bytes, bytes]:
    authority = closed_authority(authority, "handoff", readiness=True)
    raw = canonical_json(
        {
            "schema_version": 2, "domain": HANDOFF_DOMAIN, "state": "pod_bound_one_use",
            "authorization": dict(authorization),
            "claim": {"path": ARTIFACT_FILENAMES["claim"], "sha256": claim_sha256},
            "transcript": {"path": ARTIFACT_FILENAMES["transcript"], "sha256": transcript_sha256},
            "assessment": {
                "path": ARTIFACT_FILENAMES["assessment"], "sha256": assessment_sha256,
                "envelope_sha256": assessment_envelope_sha256,
            },
            "selected_resource_sha256": selected_resource_sha256,
            "selected_facts": selected_facts,
            "pod": {
                "identity_sha256": pod_identity_sha256,
                "name": f"{POD_NAME_PREFIX}-{authorization['commit'][:12]}",
                "status_sha256": pod_status_sha256,
            },
            "ssh": {
                "user": ssh_user, "host_sha256": sha256_bytes(ssh_host.encode()),
                "port": ssh_port, "known_hosts_sha256": sha256_bytes(known_hosts),
            },
            "runtime": {
                "test_script_sha256": sha256_bytes(test_script),
                "linux_uv_sha256": LINUX_UV_SHA256, "test_nodes": list(TEST_NODES),
                "gpu_probe": "exact allowed class, two devices, aggregate 96GB bounds",
                "gpu_telemetry": GPU_TELEMETRY_BINDING,
            },
            "evidence_paths": {
                name: filename for name, filename in sorted(ARTIFACT_FILENAMES.items())
            },
            "nonce": nonce, "issued_at_epoch": issued_at_epoch,
            "expires_at_epoch": issued_at_epoch + ASSESSMENT_TTL_SECONDS,
            "attempt_consumed": True, "retry": False, "authority": dict(authority),
        }
    )
    validate_handoff_payload(
        raw,
        authorization=authorization,
        claim_sha256=claim_sha256,
        transcript_sha256=transcript_sha256,
        assessment_sha256=assessment_sha256,
        assessment_envelope_sha256=assessment_envelope_sha256,
        selected_resource_sha256=selected_resource_sha256,
        selected_facts=selected_facts,
        known_hosts=known_hosts,
        test_script=test_script,
        authority=authority,
    )
    return raw, test_script


def handoff_consumer_script(
    *,
    authorization_commit: str,
    payload_sha256: str,
    public_key_sha256: str,
    test_script_sha256: str,
    authority: Mapping[str, bool] = READINESS_AUTHORITY,
) -> bytes:
    authority = closed_authority(authority, "handoff", readiness=True)
    verifier = inspect_verifier_source(authorization_commit, authority)
    return f"""set -euo pipefail
umask 077
root=/tmp/redco-one-shot-handoff-v2
test -f "$root/payload.json" -a -f "$root/payload.sig" -a -f "$root/public.key"
test "$(sha256sum "$root/payload.json" | cut -d' ' -f1)" = "{payload_sha256}"
test "$(sha256sum "$root/public.key" | cut -d' ' -f1)" = "{public_key_sha256}"
test "$(sha256sum "$root/test.sh" | cut -d' ' -f1)" = "{test_script_sha256}"
python3 - "$root/payload.json" "$root/payload.sig" "$root/public.key" <<'PY'
{verifier}
PY
( set -o noclobber; : > "$root/consumed" ) 2>/dev/null
bash "$root/test.sh"
""".encode()


def inspect_verifier_source(
    authorization_commit: str, authority: Mapping[str, bool] = READINESS_AUTHORITY
) -> str:
    authority = closed_authority(authority, "handoff", readiness=True)
    return f"""import base64,hashlib,json,struct,sys,time
def u32(raw,o):
    if o+4>len(raw): raise ValueError('truncated')
    return struct.unpack('>I',raw[o:o+4])[0],o+4
def string(raw,o):
    n,o=u32(raw,o); e=o+n
    if e>len(raw): raise ValueError('truncated')
    return raw[o:e],e
def pack(raw): return struct.pack('>I',len(raw))+raw
p=open(sys.argv[1],'rb').read(); a=open(sys.argv[2],'rb').read(); k=open(sys.argv[3],'rb').read()
v=json.loads(p)
keys={{'schema_version','domain','state','authorization','claim','transcript','assessment','selected_resource_sha256','selected_facts','pod','ssh','runtime','evidence_paths','nonce','issued_at_epoch','expires_at_epoch','attempt_consumed','retry','authority'}}
canonical=json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
if set(v)!=keys or canonical!=p: raise ValueError('payload')
if (v['schema_version']!=2 or
    v['domain']!='redco-stage-d1-prime-test-one-shot-handoff-v2' or
    v['state']!='pod_bound_one_use'): raise ValueError('domain')
if v['authorization'].get('commit')!={authorization_commit!r}: raise ValueError('authorization')
if (type(v['issued_at_epoch']) is not int or type(v['expires_at_epoch']) is not int or
    not v['issued_at_epoch']<=int(time.time())<=v['expires_at_epoch']): raise ValueError('expiry')
if v['attempt_consumed'] is not True or v['retry'] is not False: raise ValueError('attempt')
if v['authority']!={dict(authority)!r}: raise ValueError('authority')
f=k.strip().split()
if len(f)!=2 or f[0]!=b'ssh-rsa': raise ValueError('key')
kb=base64.b64decode(f[1],validate=True); kt,o=string(kb,0); er,o=string(kb,o); nr,o=string(kb,o)
if kt!=b'ssh-rsa' or o!=len(kb): raise ValueError('key')
lines=a.decode('ascii').splitlines()
if (lines[0]!='-----BEGIN SSH SIGNATURE-----' or
    lines[-1]!='-----END SSH SIGNATURE-----'): raise ValueError('armor')
s=base64.b64decode(''.join(lines[1:-1]),validate=True)
if s[:6]!=b'SSHSIG': raise ValueError('magic')
v,o=u32(s,6); ek,o=string(s,o); ns,o=string(s,o)
r,o=string(s,o); ha,o=string(s,o); sb,o=string(s,o)
if (v!=1 or ek!=kb or ns!={HANDOFF_NAMESPACE.encode()!r} or r or
    ha!=b'sha512' or o!=len(s)): raise ValueError('envelope')
alg,q=string(sb,0); sig,q=string(sb,q)
if alg!=b'rsa-sha2-512' or q!=len(sb): raise ValueError('algorithm')
d=hashlib.sha512(p).digest(); signed=b'SSHSIG'+pack(ns)+pack(b'')+pack(ha)+pack(d)
di=bytes.fromhex('3051300d060960864801650304020305000440')+hashlib.sha512(signed).digest()
n=int.from_bytes(nr,'big'); e=int.from_bytes(er,'big'); z=(n.bit_length()+7)//8
got=pow(int.from_bytes(sig,'big'),e,n).to_bytes(z,'big'); pad=z-len(di)-3
want=b'\\0\\1'+b'\\xff'*pad+b'\\0'+di
if pad<8 or got!=want: raise ValueError('signature')
"""


__all__ = [
    "LINUX_UV_BYTES",
    "LINUX_UV_SHA256",
    "LINUX_UV_SOURCE",
    "HandoffSummary",
    "build_handoff_payload",
    "handoff_consumer_script",
    "inspect_verifier_source",
    "validate_handoff_payload",
    "validate_known_hosts",
]
