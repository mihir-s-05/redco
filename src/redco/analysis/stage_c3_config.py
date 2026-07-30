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
    smoke: bool = False,
    model_path: str | None = None,
    constrained_root_routes: bool = False,
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
    if model_path is not None:
        frozen_model = "runs/stage-c2/warmstart-merged-candidates-v2/step_23"
        if text.count(frozen_model) != 1:
            raise ValueError("template has an unexpected frozen model path")
        text = text.replace(frozen_model, model_path)
    if constrained_root_routes:
        anchor = "env.context_temperature = 2.0\n"
        if text.count(anchor) != 2:
            raise ValueError("template has an unexpected context-temperature layout")
        text = text.replace(
            anchor,
            anchor + "env.constrained_root_routes = true\n",
        )
    if "__" in text:
        raise ValueError("rendered config retains an unresolved placeholder")
    if smoke:
        if arm != "broadcast":
            raise ValueError("the frozen smoke uses the broadcast arm")
        if text.count("max_steps = 36") != 1:
            raise ValueError("broadcast template has an unexpected max_steps")
        if text.count("interval = 6") != 1:
            raise ValueError("broadcast template has an unexpected eval interval")
        if text.count("num_examples = 64") != 1:
            raise ValueError("broadcast template has an unexpected eval size")
        text = text.replace("max_steps = 36", "max_steps = 1")
        text = text.replace("interval = 6", "interval = 1")
        text = text.replace("num_examples = 64", "num_examples = 16")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "schema_version": 1,
        "arm": arm,
        "probe": probe,
        "seed": seed,
        "smoke": smoke,
        "model_path": model_path,
        "constrained_root_routes": constrained_root_routes,
        "template": template.as_posix(),
        "output": output.as_posix(),
        "sha256": digest,
    }
