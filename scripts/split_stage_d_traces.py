from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from redco.contracts import canonical_json
from redco.integrations.verifiers_trace import load_trace_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    traces = load_trace_records(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    for index, trace in enumerate(traces):
        trace_id = str(trace.get("id") or "")
        if not trace_id or trace_id in seen:
            raise ValueError("every trace must have a unique nonempty ID")
        seen.add(trace_id)
        data = canonical_json(trace) + b"\n"
        digest = hashlib.sha256(data).hexdigest()
        (args.output_dir / f"{index:03d}-{digest[:12]}.jsonl").write_bytes(data)


if __name__ == "__main__":
    main()
