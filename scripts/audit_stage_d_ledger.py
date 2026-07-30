from __future__ import annotations

import argparse
import json
from pathlib import Path

from redco.analysis.stage_d_ledger import (
    ResourceMeters,
    build_stage_d_ledger,
    load_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-traces", type=Path, nargs="*", default=[])
    parser.add_argument("--eval-traces", type=Path, nargs="*", default=[])
    parser.add_argument("--optimizer-updates", type=int, required=True)
    parser.add_argument("--service-seconds", type=float, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--gpu-seconds", type=float, required=True)
    parser.add_argument("--storage-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_records = [
        record for path in args.train_traces for record in load_jsonl(path)
    ]
    eval_records = [
        record for path in args.eval_traces for record in load_jsonl(path)
    ]
    ledger = build_stage_d_ledger(
        train_records,
        eval_records,
        ResourceMeters(
            optimizer_updates=args.optimizer_updates,
            service_seconds=args.service_seconds,
            wall_seconds=args.wall_seconds,
            gpu_seconds=args.gpu_seconds,
            storage_bytes=args.storage_bytes,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
