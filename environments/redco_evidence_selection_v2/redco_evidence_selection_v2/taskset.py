from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

from redco_evidence_selection_v2.scoring import score_evidence_reply

WORKDIR = "/workspace"
CONTEXT_PATH = f"{WORKDIR}/evidence_context.txt"
ISOLATED_EXECUTION_USER = "65534:65534"
ISOLATED_EXECUTION_HOME = "/tmp/redco-agent"
BASE_POLICY_CHECKPOINT = (
    "Qwen/Qwen3-4B-Instruct-2507@"
    "cdbee75f17c01a7cc42f958dc650907174af0554"
)

SYSTEM = f"""You extract VERBATIM evidence from one scientific paper.

The complete paper is stored at `{CONTEXT_PATH}`. Use IPython to read and search
that file without printing the entire paper into the root context. Search several
keyword variants and inspect surrounding text. Verify every proposed span against
the local paper text before answering.

Your final answer must be a Python list of nonempty exact strings copied from the
paper, submitted with FINAL(...). Every returned string must occur verbatim in
the paper. Return the minimum contiguous spans that directly answer the question.
Do not paraphrase, pad, or include commentary.
"""

ALIGNED_SYSTEM_V2 = f"""You extract VERBATIM evidence from one scientific paper.

The complete paper is stored at `{CONTEXT_PATH}`. Use IPython to read and search
that file without printing the entire paper into the root context. Search several
keyword variants and inspect surrounding text. Verify every proposed span against
the local paper text before answering.

After tool use, stop calling tools and state exactly one bare Python list of
nonempty exact strings copied from the paper. Do not wrap the list in FINAL(...)
and do not add prose. Every returned string must occur verbatim in the paper.
Return the minimum contiguous spans that directly answer the question. Do not
paraphrase or pad.
"""

FORCED_TRACE_FIXTURE = """

INTEGRATION FIXTURE ONLY: in your first IPython action, read the paper, make two
bounded excerpts, and call exactly two `rlm(...)` children concurrently, one on
each excerpt. Include the question in each child prompt. Wait for both children,
then verify and answer normally. This fixture is used only to audit trace and
replay mechanics; its reward is excluded from scientific feasibility metrics.
"""


class EvidenceSelectionData(vf.TaskData):
    example_id: str
    paper_id: str
    title: str
    question: str
    paper: str
    reference_evidence: tuple[str, ...]
    answer_type: str
    split: str
    snapshot_sha256: str
    policy_checkpoint_id: str


class EvidenceSelectionTask(vf.Task[EvidenceSelectionData]):
    # The frozen live protocol uses a dedicated ephemeral Prime pod with no
    # secrets or persistent storage. Keeping the RLM subprocess on that already
    # isolated host also preserves the patched local RLM executable and exact
    # cross-component trace headers.
    NEEDS_CONTAINER = False

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del trace
        if self.data.network_allow == []:
            await prepare_isolated_workspace(runtime, self.data.paper.encode("utf-8"))
        else:
            await runtime.run(["mkdir", "-p", WORKDIR], {})
            await runtime.write(CONTEXT_PATH, self.data.paper.encode("utf-8"))

    def _score(self, trace: vf.Trace) -> dict[str, float]:
        return score_evidence_reply(
            self.data.paper,
            trace.last_reply,
            self.data.reference_evidence,
        )

    @vf.reward
    async def exact_span_f1(self, trace: vf.Trace) -> float:
        return self._score(trace)["f1"]

    @vf.metric
    async def evidence_precision(self, trace: vf.Trace) -> float:
        return self._score(trace)["precision"]

    @vf.metric
    async def evidence_recall(self, trace: vf.Trace) -> float:
        return self._score(trace)["recall"]

    @vf.metric
    async def evidence_parseable(self, trace: vf.Trace) -> float:
        return self._score(trace)["parseable"]

    @vf.metric
    async def all_predicted_spans_verbatim(self, trace: vf.Trace) -> float:
        return self._score(trace)["all_predicted_spans_verbatim"]

    @vf.metric
    async def predicted_characters(self, trace: vf.Trace) -> float:
        return self._score(trace)["predicted_characters"]

    @vf.metric
    async def predicted_span_count(self, trace: vf.Trace) -> float:
        return self._score(trace)["predicted_span_count"]

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        score = self._score(trace)
        trace.info["evidence_selection"] = {
            "example_id": self.data.example_id,
            "paper_id": self.data.paper_id,
            "split": self.data.split,
            "answer_type": self.data.answer_type,
            "snapshot_sha256": self.data.snapshot_sha256,
            "score": score,
        }
        trace.info["checkpoint_id"] = self.data.policy_checkpoint_id


class EvidenceSelectionConfig(vf.TasksetConfig):
    dataset_path: Path
    dataset_sha256: str
    split: str = "audit"
    prompt_profile: Literal[
        "natural",
        "forced_trace_fixture",
        "fewshot_scaffold_v2",
        "fewshot_fixture_v3",
        "fewshot_fixture_v4",
    ] = "natural"
    policy_checkpoint_id: str = BASE_POLICY_CHECKPOINT
    scaffold_prompt_path: Path | None = None
    scaffold_prompt_sha256: str | None = None
    isolated_runtime_image: str | None = None


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
            if not evidence or any(
                not span or span not in paper for span in evidence
            ):
                raise ValueError(
                    f"{row['example_id']} has invalid reference evidence"
                )
            if self.config.prompt_profile in (
                "fewshot_scaffold_v2",
                "fewshot_fixture_v3",
                "fewshot_fixture_v4",
            ):
                system = ALIGNED_SYSTEM_V2
            else:
                system = SYSTEM
            if self.config.prompt_profile in (
                "forced_trace_fixture",
                "fewshot_fixture_v3",
                "fewshot_fixture_v4",
            ):
                system += FORCED_TRACE_FIXTURE
            if self.config.prompt_profile in (
                "fewshot_scaffold_v2",
                "fewshot_fixture_v3",
                "fewshot_fixture_v4",
            ):
                if (
                    self.config.scaffold_prompt_path is None
                    or self.config.scaffold_prompt_sha256 is None
                ):
                    raise ValueError(
                        "fewshot_scaffold_v2 requires a prompt path and SHA-256"
                    )
                scaffold = self.config.scaffold_prompt_path.read_bytes()
                scaffold_sha256 = hashlib.sha256(scaffold).hexdigest()
                if scaffold_sha256 != self.config.scaffold_prompt_sha256:
                    raise ValueError(
                        "shared scaffold prompt hash mismatch: "
                        f"got {scaffold_sha256}, expected "
                        f"{self.config.scaffold_prompt_sha256}"
                    )
                system += "\n\n" + scaffold.decode("utf-8")
            data = EvidenceSelectionData(
                idx=row_index,
                name=row["example_id"],
                prompt=f"{system}\n\nQuestion: {row['question']}",
                workdir=WORKDIR,
                example_id=row["example_id"],
                paper_id=row["paper_id"],
                title=row["title"],
                question=row["question"],
                paper=paper,
                reference_evidence=evidence,
                answer_type=row["answer_type"],
                split=row["split"],
                snapshot_sha256=digest,
                policy_checkpoint_id=self.config.policy_checkpoint_id,
                image=self.config.isolated_runtime_image,
                network_allow=([] if self.config.isolated_runtime_image else ["*"]),
                network_block=[],
            )
            tasks.append(EvidenceSelectionTask(data, self.config.task))
        if not tasks:
            raise ValueError(
                f"snapshot has no examples for split {self.config.split!r}"
            )
        return tasks


async def prepare_isolated_workspace(runtime: vf.Runtime, paper: bytes) -> None:
    """Create one root-owned input plus the only two non-root writable directories."""
    if type(paper) is not bytes or not paper:
        raise ValueError("isolated workspace paper must be nonempty immutable bytes")
    prepared = await runtime.run(
        [
            "sh",
            "-c",
            (
                "set -eu; "
                "install -d -m 0755 /workspace; "
                "install -d -o 65534 -g 65534 -m 0700 /workspace/.rlm; "
                "install -d -o 65534 -g 65534 -m 0700 /tmp/redco-agent; "
                "install -d -o 65534 -g 65534 -m 0700 /tmp/redco-agent/.cache"
            ),
        ],
        {},
    )
    if prepared.exit_code != 0:
        raise RuntimeError(f"isolated workspace setup failed: {prepared.stderr[-1000:]}")
    await runtime.write(CONTEXT_PATH, paper)
    frozen = await runtime.run(
        [
            "sh",
            "-c",
            (
                "set -eu; chown 0:0 /workspace/evidence_context.txt; "
                "chmod 0444 /workspace/evidence_context.txt; chmod 0555 /workspace"
            ),
        ],
        {},
    )
    if frozen.exit_code != 0:
        raise RuntimeError(f"isolated workspace freeze failed: {frozen.stderr[-1000:]}")
