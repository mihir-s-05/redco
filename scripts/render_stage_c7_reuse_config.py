"""Render one frozen Stage-C7 offline-reuse trainer configuration."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def render_control(
    template: Path,
    output: Path,
    *,
    output_dir: str,
    max_steps: int,
) -> None:
    text = template.read_text(encoding="utf-8")
    rendered, output_count = re.subn(
        r'(?m)^output_dir = "[^"]*"$',
        f'output_dir = "{output_dir}/run_default"',
        text,
    )
    rendered, steps_count = re.subn(
        r"(?m)^max_steps = \d+$",
        f"max_steps = {max_steps}",
        rendered,
    )
    if output_count != 1 or steps_count != 1:
        raise ValueError("orchestrator control fields changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--control-template", type=Path, required=True)
    parser.add_argument("--control-output", type=Path, required=True)
    args = parser.parse_args()

    text = args.template.read_text(encoding="utf-8")
    if text.count("__OUTPUT_DIR__") != 1 or text.count("__MAX_STEPS__") != 1:
        raise ValueError("trainer template placeholders changed")
    rendered = text.replace("__OUTPUT_DIR__", args.output_dir).replace(
        "__MAX_STEPS__",
        str(args.max_steps),
    )
    if "__" in rendered:
        raise ValueError("trainer template has an unresolved placeholder")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    render_control(
        args.control_template,
        args.control_output,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
