"""Render one frozen Stage-C3 arm config from a committed template."""

from __future__ import annotations

import hashlib
from pathlib import Path

PROBES = {
    "confusion_irrelevant",
    "confusion_redundant",
    "confusion_lucky",
}
ARMS = {"broadcast", "sliced"}


def render(
    template: Path,
    output: Path,
    *,
    arm: str,
    probe: str,
    seed: int,
    run_root: str,
) -> dict[str, object]:
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    if probe not in PROBES:
        raise ValueError(f"unsupported probe: {probe}")
    if seed < 1:
        raise ValueError("seed must be positive")
    text = template.read_text(encoding="utf-8")
    replacements = {
        "__OUTPUT_DIR__": f"{run_root}/{probe}/{arm}-s{seed}",
        "__PROBE__": probe,
        "__TRAIN_OFFSET__": str(seed * 1000),
        "__EVAL_OFFSET__": str(seed * 1000 + 500),
        "__INFERENCE_SEED__": str(seed),
    }
    for placeholder, value in replacements.items():
        if text.count(placeholder) < 1:
            raise ValueError(f"template is missing {placeholder}")
        text = text.replace(placeholder, value)
    if "__" in text:
        raise ValueError("rendered config retains an unresolved placeholder")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "schema_version": 1,
        "arm": arm,
        "probe": probe,
        "seed": seed,
        "template": template.as_posix(),
        "output": output.as_posix(),
        "sha256": digest,
    }
