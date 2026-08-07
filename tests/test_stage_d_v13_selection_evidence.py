from __future__ import annotations

import hashlib
import json
from pathlib import Path

from redco.analysis.stage_d_v13_draft import canonical_json_bytes


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_tracked_claim_mirror_is_exact_raw_evidence() -> None:
    root = Path(__file__).parents[1]
    original = (
        root / "runs/stage-d/stage-d1-support-v13-source-selection-claim-v1.json"
    ).read_bytes()
    mirror = (root / "reports/stage-d1-support-v13-source-selection-claim-v1.json").read_bytes()
    assert len(original) == 704
    assert _sha256(original) == "240b30bc283c993bf007834f7ce7524a97a177da1678cddaeff04e60d7c8edac"
    assert mirror == original
    assert mirror[-1:] != b"\n"


def test_selection_evidence_manifest_binds_candidate_and_no_raw_transcript() -> None:
    root = Path(__file__).parents[1]
    manifest_path = root / "reports/stage-d1-support-v13-source-selection-evidence-manifest-v1.json"
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    assert raw == canonical_json_bytes(payload)
    assert raw[-1:] != b"\n"
    assert payload["claim"]["exact_byte_identity_with_mirror"] is True
    assert payload["candidate"]["source_ordinal"] == 180
    assert payload["candidate"]["paper_id"] == "2001.09332"
    assert payload["candidate"]["example_id"].startswith("qasper-")
    assert payload["attempt"] == {
        "consumed": 1,
        "no_row_after_ordinal": 180,
        "retry": False,
    }
    assert payload["transcript"]["raw_retained"] is False
    assert payload["authority"]["provider_calls_authorized"] is False
    assert payload["authority"]["science_authorized"] is False
