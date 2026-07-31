from __future__ import annotations

import hashlib
import json


def derive_episode_seed(
    master_seed: str,
    example_id: str,
    replicate: int,
) -> int:
    """Derive a nonzero signed-31-bit seed from the full episode address."""
    if replicate < 0:
        raise ValueError("replicate must be nonnegative")
    payload = json.dumps(
        {
            "master_seed": master_seed,
            "example_id": example_id,
            "replicate": replicate,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**31 - 1
    ) + 1
