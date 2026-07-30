from __future__ import annotations

import hashlib
import json
from pathlib import Path

import verifiers.v1 as vf

from redco_evidence_selection_v1.scoring import parse_evidence, score_exact_spans

WORKDIR = "/workspace"
CONTEXT_PATH = f"{WORKDIR}/evidence_context.txt"

SYSTEM = f"""You extract VERBATIM evidence from one scientific paper.

The complete paper is stored at `{CONTEXT_PATH}`. Use IPython to read and search
that file. Search several keyword variants, inspect surrounding text, and return
the minimum contiguous spans that directly answer the question.

Your final answer must be a Python list of exact strings copied from the paper,
submitted with FINAL(...). Do not paraphrase. Do not include commentary or
section headings. If the paper contains no answer, submit FINAL([]).
"""


class EvidenceSelectionData(vf.TaskData):
    example_id: str
    paper_id: str
    title: str
    question: str
    paper: str
    reference_evidence: tuple[str, ...]
    split: str
    snapshot_sha256: str


class EvidenceSelectionTask(vf.Task[EvidenceSelectionData]):
    NEEDS_CONTAINER = True

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del trace
        await runtime.run(["mkdir", "-p", WORKDIR], {})
        await runtime.write(CONTEXT_PATH, self.data.paper.encode("utf-8"))

    @vf.reward(weight=0.0)
    async def exact_span_f1(self, trace: vf.Trace) -> float:
        parsed = parse_evidence(trace.last_reply)
        return score_exact_spans(
            self.data.paper, parsed.spans, self.data.reference_evidence
        )["f1"]

    @vf.metric
    async def evidence_parseable(self, trace: vf.Trace) -> float:
        return float(parse_evidence(trace.last_reply).parseable)

    @vf.metric
    async def exact_substring_fraction(self, trace: vf.Trace) -> float:
        parsed = parse_evidence(trace.last_reply)
        return score_exact_spans(
            self.data.paper, parsed.spans, self.data.reference_evidence
        )["exact_substring_fraction"]

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        parsed = parse_evidence(trace.last_reply)
        trace.info["evidence_selection"] = {
            "example_id": self.data.example_id,
            "paper_id": self.data.paper_id,
            "split": self.data.split,
            "snapshot_sha256": self.data.snapshot_sha256,
            "parseable": parsed.parseable,
            "predicted_span_count": len(parsed.spans),
            "reference_span_count": len(self.data.reference_evidence),
        }


class EvidenceSelectionConfig(vf.TasksetConfig):
    dataset_path: Path = Path(
        "datasets/stage-d/evidence-selection-fixture-v1.jsonl"
    )
    dataset_sha256: str
    split: str = "audit"


class EvidenceSelectionTaskset(
    vf.Taskset[EvidenceSelectionTask, EvidenceSelectionConfig]
):
    def load(self) -> list[EvidenceSelectionTask]:
        raw = self.config.dataset_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != self.config.dataset_sha256:
            raise ValueError(
                "evidence-selection snapshot hash mismatch: "
                f"got {digest}, expected {self.config.dataset_sha256}"
            )
        tasks: list[EvidenceSelectionTask] = []
        for row_index, line in enumerate(raw.decode("utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != self.config.split:
                continue
            paper = row["paper"]
            evidence = tuple(row["reference_evidence"])
            missing = [span for span in evidence if span not in paper]
            if missing:
                raise ValueError(
                    f"{row['example_id']} contains non-verbatim reference evidence"
                )
            data = EvidenceSelectionData(
                idx=row_index,
                name=row["example_id"],
                prompt=f"{SYSTEM}\n\nQuestion: {row['question']}",
                workdir=WORKDIR,
                example_id=row["example_id"],
                paper_id=row["paper_id"],
                title=row["title"],
                question=row["question"],
                paper=paper,
                reference_evidence=evidence,
                split=row["split"],
                snapshot_sha256=digest,
            )
            tasks.append(EvidenceSelectionTask(data, self.config.task))
        if not tasks:
            raise ValueError(
                f"snapshot has no examples for split {self.config.split!r}"
            )
        return tasks
