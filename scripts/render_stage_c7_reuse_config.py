"""Render one frozen Stage-C7 offline-reuse trainer configuration."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, choices=(1, 2, 3), required=True)
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


if __name__ == "__main__":
    main()
