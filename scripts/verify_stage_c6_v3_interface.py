"""Verify Stage-C6 v3 route traces, packed bytes, and trainer exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import msgspec
from prime_rl.transport.types import TrainingBatch

from redco.analysis.stage_c6_v3_interface import verify_interface
from redco.integrations.signed_subprocess import atomic_write_json, sign_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--token-exports", type=Path, required=True)
    parser.add_argument("--root-scores", type=Path, required=True)
    parser.add_argument("--expected-context-traces", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    traces = [
        json.loads(line)
        for line in args.traces.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    batch = msgspec.msgpack.decode(
        args.batch.read_bytes(), type=TrainingBatch
    )
    export_files = sorted(args.token_exports.rglob("*.jsonl"))
    token_exports = [
        json.loads(line)
        for path in export_files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root_scores = json.loads(args.root_scores.read_text(encoding="utf-8"))
    result = sign_payload(
        verify_interface(
            traces=traces,
            batch=batch,
            token_exports=token_exports,
            root_scores=root_scores,
            expected_context_traces=args.expected_context_traces,
        )
    )
    atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "signature": result["signed_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
