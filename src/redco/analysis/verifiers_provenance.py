"""Report event provenance and measured costs from recorded verifiers traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json
from redco.integrations.verifiers_provenance import import_trace_file


def build_provenance_report(path: Path) -> dict[str, Any]:
    imported = import_trace_file(path)
    prompt_tokens = sum(trace.cost.prompt_tokens for trace in imported.traces)
    generated_tokens = sum(
        trace.cost.generated_tokens for trace in imported.traces
    )
    model_calls = sum(trace.cost.model_calls for trace in imported.traces)
    call_wall = sum(
        trace.cost.model_call_wall_seconds for trace in imported.traces
    )
    generation_wall = sum(
        trace.cost.trace_generation_wall_seconds for trace in imported.traces
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "source_sha256": imported.source_sha256,
        "completed": True,
        "ready_for_representative_raf": imported.ready_for_representative_raf,
        "summary": {
            "trace_count": len(imported.traces),
            "model_calls": model_calls,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "model_call_wall_seconds": call_wall,
            "trace_generation_wall_seconds": generation_wall,
            "exact_prompt_provenance_coverage": (
                imported.exact_prompt_provenance_coverage
            ),
            "structural_model_call_coverage": (
                imported.structural_model_call_coverage
            ),
            "recursive_model_calls": imported.recursive_model_calls,
            "exact_recursive_parent_coverage": (
                imported.exact_recursive_parent_coverage
            ),
            "recursive_components": imported.recursive_components,
            "exact_cross_component_links": (
                imported.exact_cross_component_links
            ),
            "cross_component_fallbacks": (
                imported.cross_component_fallbacks
            ),
            "cross_component_fallback_rate": (
                imported.cross_component_fallback_rate
            ),
        },
        "blocking_finding": (
            None
            if imported.ready_for_representative_raf
            else (
                "The trace lacks complete exact RLM session/turn metadata or "
                "required conservative cross-component call-edge fallbacks. "
                "A representative sliced RAF is not yet defensible."
            )
        ),
        "import": imported.as_dict(),
    }
    payload["report_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero unless provenance is ready for representative RAF.",
    )
    args = parser.parse_args()
    payload = build_provenance_report(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(payload) + b"\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return int(
        args.require_ready and not payload["ready_for_representative_raf"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
