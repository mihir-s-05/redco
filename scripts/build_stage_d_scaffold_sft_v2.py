"""Build the invocation-addressed Stage D scaffold SFT corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json

IPYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "ipython",
        "description": (
            "Execute code in a persistent IPython session. A callable rlm is "
            "already available inside IPython."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python or IPython code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in seconds.",
                },
            },
            "required": ["code"],
        },
    },
}
SUBJECTS = (
    "amber kestrel",
    "blue orchard",
    "copper lantern",
    "distant harbor",
    "emerald compass",
    "frosted willow",
    "granite meadow",
    "hollow cedar",
)
PREDICATES = (
    "was calibrated at dawn",
    "remained stable for seven cycles",
    "used a triangular reference frame",
    "was stored beside the north window",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _system(scaffold_path: Path) -> str:
    scaffold = scaffold_path.read_text(encoding="utf-8").strip()
    return (
        "You extract exact evidence from a local synthetic document. The "
        "document is stored at `/workspace/evidence_context.txt`. Use IPython "
        "and recursive subcalls, verify the result locally, and finish with "
        "exactly one bare Python list of strings.\n\n"
        + scaffold
    )


def scaffold_code(question: str) -> str:
    return (
        "import asyncio\n"
        "paper_text = open('/workspace/evidence_context.txt', "
        "encoding='utf-8').read()\n"
        "midpoint = len(paper_text) // 2\n"
        "excerpts = [paper_text[:midpoint], paper_text[midpoint:]]\n"
        f"question = {question!r}\n"
        "prompts = [\n"
        "    f\"Return candidate verbatim evidence for {question!r} "
        "from only this excerpt:\\n{excerpt}\"\n"
        "    for excerpt in excerpts\n"
        "]\n"
        "def normalize_child_answer(value):\n"
        "    answer = getattr(value, 'answer', value)\n"
        "    if not isinstance(answer, str):\n"
        "        raise TypeError('recursive child answer must be a string')\n"
        "    return answer\n"
        "children = await asyncio.gather(\n"
        "    *(\n"
        "        rlm(prompt, redco_invocation_id=f'midpoint-shard-{index}')\n"
        "        for index, prompt in enumerate(prompts)\n"
        "    )\n"
        ")\n"
        "child_answers = [normalize_child_answer(child) for child in children]\n"
        "child_answers"
    )


def _row(index: int, system: str) -> dict[str, Any]:
    subject = SUBJECTS[index % len(SUBJECTS)]
    predicate = PREDICATES[(index // len(SUBJECTS)) % len(PREDICATES)]
    evidence = f"The {subject} {predicate}."
    question = f"What does the document state about the {subject}?"
    tool_output = repr(
        [
            f"Candidate evidence: {evidence}",
            "No additional evidence in this excerpt.",
        ]
    )
    return {
        "synthetic_id": f"scaffold-sft-v2-{index:03d}",
        "source": "deterministic_synthetic_no_qasper_paper",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "ipython",
                            "arguments": json.dumps(
                                {"code": scaffold_code(question)},
                                sort_keys=True,
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "ipython",
                "tool_call_id": "call_0",
                "content": tool_output,
            },
            {"role": "assistant", "content": repr([evidence])},
        ],
        "tools": [IPYTHON_TOOL],
    }


def build(
    *,
    scaffold_path: Path,
    dataset_path: Path,
    manifest_path: Path,
    examples: int,
) -> None:
    if not 1 <= examples <= 50:
        raise ValueError("examples must be between one and 50")
    system = _system(scaffold_path)
    rows = [_row(index, system) for index in range(examples)]
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in rows)
    )
    manifest = {
        "schema_version": 2,
        "generator": "scripts/build_stage_d_scaffold_sft_v2.py",
        "examples": examples,
        "dataset": dataset_path.as_posix(),
        "dataset_sha256": _sha256(dataset_path),
        "scaffold_prompt": scaffold_path.as_posix(),
        "scaffold_prompt_sha256": _sha256(scaffold_path),
        "classification": "shared_synthetic_scaffold_and_task_sft",
        "child_answer_contract": "single idempotent normalization",
        "invocation_address_contract": "midpoint-shard-{0,1}",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=32)
    args = parser.parse_args()
    build(
        scaffold_path=args.scaffold,
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        examples=args.examples,
    )


if __name__ == "__main__":
    main()
