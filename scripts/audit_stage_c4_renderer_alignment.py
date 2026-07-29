"""Audit Stage-C4 SFT tokens against the frozen live scoring cases."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from tokenizers import Tokenizer

from redco.analysis.stage_c4_renderer_alignment import audit_renderer_alignment
from redco.integrations.signed_subprocess import atomic_write_json


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root-cases", type=Path, required=True)
    parser.add_argument("--action-cases", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    report = audit_renderer_alignment(
        renderer_name=str(config["renderer"]["name"]),
        dataset=_read_jsonl(args.dataset),
        root_cases=json.loads(args.root_cases.read_text(encoding="utf-8")),
        action_cases=json.loads(args.action_cases.read_text(encoding="utf-8")),
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False).ids,
    )
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "signed_payload_sha256": report["signed_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
