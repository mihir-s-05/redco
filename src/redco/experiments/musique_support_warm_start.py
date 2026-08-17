"""Small, deterministic MuSiQue support-path warm-start boundary."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from redco.contracts import canonical_json
from redco.experiments.musique_capability import load_tasks

ARCHIVE_SHA256 = "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
TRAIN_ENTRY = "data/musique_ans_v1.0_train.jsonl"
TRAIN_ENTRY_BYTES = 241_046_755
TRAIN_ENTRY_SHA256 = "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
EVAL_SNAPSHOT_PATH = "data/musique-ans-capability-v1.json"
EVAL_SNAPSHOT_SHA256 = "07f75ea217779b754a37136d204de19f45f26679bdb6b7e056089cb5e54c70ed"
EVAL_GATE_PATH = "configs/musique-ans-capability-gate-v1.json"
EVAL_GATE_SHA256 = "9978ac70b684026b15073786c960b33bfd3d4d9973ea41bb25ccb82a80eea646"
MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PROMPT_TEMPLATE = (
    "Question:\n{question}\n\nAlready selected evidence:\n{evidence}\n\n"
    "Choose exactly one next document from the numbered candidates. Return only one integer "
    "from the displayed range.\n\nCandidates:\n{candidates}"
)
LABELS = tuple(str(index) for index in range(1, 9))
PERMUTATION_A = tuple(range(8))
PERMUTATION_B = (1, 0, 3, 2, 5, 4, 7, 6)
TRAINING_SEED = 20260817
SELECTION_NAMESPACE = "redco-musique-support-warm-start-v1"
SELECTION_LAW = "sha256(canonical_json([namespace, depth, source_row_sha256]))"
STATE_ORDER_NAMESPACE = "redco-musique-support-warm-start-v1-state-order"
STATE_ORDER_LAW = (
    "sha256(canonical_json([namespace, seed, source_row_sha256, permutation_index, step]))"
)
_REFERENCE = re.compile(r"#([1-9][0-9]*)")
_ROW_ID = re.compile(r"^(2|4)hop-train-[0-9]{3}$")
_PUNCTUATION = str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


@dataclass(frozen=True, slots=True)
class WarmCandidate:
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class WarmTask:
    row_id: str
    depth: int
    question: str
    candidates: tuple[WarmCandidate, ...]
    support_path: tuple[str, ...]
    source_row_sha256: str
    selection_key: str
    permutations: tuple[tuple[int, ...], ...]

    def states(self) -> Iterator[tuple[int, int, tuple[int, ...], int]]:
        title_to_index = {candidate.title: index for index, candidate in enumerate(self.candidates)}
        for permutation_index, permutation in enumerate(self.permutations):
            for step, target_title in enumerate(self.support_path):
                selected = {title_to_index[title] for title in self.support_path[:step]}
                active = tuple(index for index in permutation if index not in selected)
                target = next(
                    (
                        position + 1
                        for position, index in enumerate(active)
                        if self.candidates[index].title == target_title
                    ),
                    None,
                )
                if target is None:
                    raise ValueError("warm-start target is absent from its active candidate pool")
                yield permutation_index, step, active, target


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_parts(parts: Sequence[object]) -> str:
    return _sha256_bytes(canonical_json(list(parts)))


def selection_key(depth: int, source_row_sha256: str) -> str:
    return _hash_parts((SELECTION_NAMESPACE, depth, source_row_sha256))


def _state_key(source_row_sha256: str, permutation_index: int, step: int) -> str:
    return _hash_parts(
        (STATE_ORDER_NAMESPACE, TRAINING_SEED, source_row_sha256, permutation_index, step)
    )


def normalize_identity(value: str) -> str:
    """Match the compact identity normalization used for leakage checks."""
    return " ".join(value.lower().translate(_PUNCTUATION).split())


def _as_string(value: object, message: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(message)
    return value


def _as_component_id(value: object, message: str) -> str:
    if type(value) is int and value >= 0:
        return str(value)
    return _as_string(value, message)


def _parse_graph(row: Mapping[str, object], depth: int) -> tuple[tuple[str, ...], tuple[int, ...]]:
    raw = row.get("question_decomposition")
    if type(raw) is not list or len(raw) != depth:
        raise ValueError("decomposition depth is not exact")
    ids: list[str] = []
    questions: list[str] = []
    support_indices: list[int] = []
    for node in cast(list[object], raw):
        if type(node) is not dict or set(node) != {
            "answer",
            "id",
            "paragraph_support_idx",
            "question",
        }:
            raise ValueError("decomposition schema is not official")
        item = cast(dict[str, object], node)
        ids.append(_as_component_id(item["id"], "decomposition id is invalid"))
        questions.append(_as_string(item["question"], "decomposition question is invalid"))
        support = item["paragraph_support_idx"]
        if type(support) is not int or support < 0:
            raise ValueError("decomposition support index is invalid")
        support_indices.append(support)
    if len(set(ids)) != depth:
        raise ValueError("decomposition ids are not unique")
    parents: dict[str, tuple[str, ...]] = {}
    for index, question in enumerate(questions):
        refs = tuple(int(match) for match in _REFERENCE.findall(question))
        if len(set(refs)) != len(refs) or any(ref > depth for ref in refs):
            raise ValueError("decomposition reference is invalid")
        parents[ids[index]] = tuple(ids[ref - 1] for ref in refs)
    roots = [node for node, refs in parents.items() if not refs]
    children: dict[str, list[str]] = {node: [] for node in ids}
    for child, parent_names in parents.items():
        for parent in parent_names:
            if parent == child:
                raise ValueError("decomposition graph has a self-reference")
            children[parent].append(child)
    if len(roots) != 1 or any(len(value) > 1 for value in children.values()):
        raise ValueError("decomposition graph is not a single chain")
    order: list[str] = []
    current: str | None = roots[0]
    while current is not None:
        if current in order:
            raise ValueError("decomposition graph contains a cycle")
        order.append(current)
        next_nodes = children[current]
        current = next_nodes[0] if next_nodes else None
    if len(order) != depth:
        raise ValueError("decomposition graph is not connected")
    position = {node: index for index, node in enumerate(ids)}
    return tuple(order), tuple(support_indices[position[node]] for node in order)


def _parse_paragraphs(row: Mapping[str, object]) -> dict[int, tuple[str, str, bool]]:
    raw = row.get("paragraphs")
    if type(raw) is not list or not raw:
        raise ValueError("paragraph list is invalid")
    paragraphs: dict[int, tuple[str, str, bool]] = {}
    for value in cast(list[object], raw):
        if type(value) is not dict or set(value) != {
            "idx",
            "is_supporting",
            "paragraph_text",
            "title",
        }:
            raise ValueError("paragraph schema is not official")
        item = cast(dict[str, object], value)
        index = item["idx"]
        supporting = item["is_supporting"]
        if type(index) is not int or type(supporting) is not bool:
            raise ValueError("paragraph primitive type is invalid")
        if index in paragraphs:
            raise ValueError("paragraph indices are not unique")
        paragraphs[index] = (
            _as_string(item["title"], "paragraph title is invalid"),
            _as_string(item["paragraph_text"], "paragraph text is invalid"),
            supporting,
        )
    return paragraphs


def _candidate_task(
    row: Mapping[str, object],
    raw_line: bytes,
    eval_components: set[str],
    eval_titles: set[str],
    eval_questions: set[str],
    eval_text_digests: set[str],
) -> WarmTask | None:
    required = {
        "answer",
        "answer_aliases",
        "answerable",
        "id",
        "paragraphs",
        "question",
        "question_decomposition",
    }
    if set(row) != required or row.get("answerable") is not True:
        return None
    source_id = _as_string(row["id"], "source id is invalid")
    prefix = source_id.split("__", 1)[0]
    depth = 2 if prefix == "2hop" else 4 if prefix == "4hop1" else 0
    if depth == 0:
        return None
    question = _as_string(row["question"], "source question is invalid")
    if normalize_identity(question) in eval_questions:
        return None
    decomposition = cast(list[object], row["question_decomposition"])
    components = tuple(
        _as_component_id(cast(dict[str, object], item)["id"], "component id")
        for item in decomposition
    )
    if set(components) & eval_components:
        return None
    _order, support_indices = _parse_graph(row, depth)
    paragraphs = _parse_paragraphs(row)
    support_docs: list[WarmCandidate] = []
    support_titles: list[str] = []
    for index in support_indices:
        if index not in paragraphs:
            return None
        title, text, supporting = paragraphs[index]
        if not supporting:
            return None
        digest = _sha256_bytes(text.encode("utf-8"))
        if title in eval_titles or digest in eval_text_digests:
            return None
        support_docs.append(WarmCandidate(title, text))
        support_titles.append(title)
    if len(set(support_titles)) != depth:
        return None
    needed = 8 - depth
    distractors: list[WarmCandidate] = []
    used_titles = set(support_titles)
    used_texts = {_sha256_bytes(item.text.encode("utf-8")) for item in support_docs}
    for index in sorted(paragraphs):
        title, text, _supporting = paragraphs[index]
        digest = _sha256_bytes(text.encode("utf-8"))
        if title in used_titles or digest in used_texts:
            continue
        if title in eval_titles or digest in eval_text_digests:
            continue
        distractors.append(WarmCandidate(title, text))
        used_titles.add(title)
        used_texts.add(digest)
        if len(distractors) == needed:
            break
    if len(distractors) != needed:
        return None
    candidates = tuple(support_docs + distractors)
    if len({item.title for item in candidates}) != 8:
        return None
    source_digest = _sha256_bytes(raw_line)
    return WarmTask(
        "",
        depth,
        question,
        candidates,
        tuple(support_titles),
        source_digest,
        selection_key(depth, source_digest),
        (PERMUTATION_A, PERMUTATION_B),
    )


def _eval_exclusions(eval_path: Path) -> tuple[set[str], set[str], set[str], set[str]]:
    tasks = load_tasks(eval_path, expected_sha256=EVAL_SNAPSHOT_SHA256)
    components = {component for task in tasks for component in task.support_components}
    titles = {candidate.title for task in tasks for candidate in task.candidates}
    questions = {normalize_identity(task.question) for task in tasks}
    text_digests = {
        _sha256_bytes(candidate.text.encode("utf-8"))
        for task in tasks
        for candidate in task.candidates
    }
    return components, titles, questions, text_digests


def _relabel(task: WarmTask, index: int) -> WarmTask:
    return WarmTask(
        f"{task.depth}hop-train-{index:03d}",
        task.depth,
        task.question,
        task.candidates,
        task.support_path,
        task.source_row_sha256,
        task.selection_key,
        (PERMUTATION_A, PERMUTATION_B),
    )


def build_snapshot(archive_path: Path, eval_path: Path, output_path: Path) -> dict[str, object]:
    if _sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("official MuSiQue archive hash does not match")
    if _sha256_file(eval_path) != EVAL_SNAPSHOT_SHA256:
        raise ValueError("frozen MuSiQue eval snapshot hash does not match")
    exclusions = _eval_exclusions(eval_path)
    eligible: dict[int, list[WarmTask]] = {2: [], 4: []}
    entry_digest = hashlib.sha256()
    entry_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        try:
            info = archive.getinfo(TRAIN_ENTRY)
        except KeyError as error:
            raise ValueError("official train entry is missing") from error
        if info.file_size != TRAIN_ENTRY_BYTES:
            raise ValueError("official train entry size changed")
        with archive.open(info) as stream:
            for raw_line in stream:
                entry_digest.update(raw_line)
                entry_bytes += len(raw_line)
                try:
                    value = json.loads(raw_line)
                    if type(value) is dict:
                        task = _candidate_task(
                            cast(Mapping[str, object], value), raw_line, *exclusions
                        )
                        if task is not None:
                            eligible[task.depth].append(task)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
    if entry_bytes != TRAIN_ENTRY_BYTES or entry_digest.hexdigest() != TRAIN_ENTRY_SHA256:
        raise ValueError("official train entry stream is incomplete")
    tasks: list[WarmTask] = []
    for depth in (2, 4):
        ranked = sorted(
            eligible[depth], key=lambda task: (task.selection_key, task.source_row_sha256)
        )
        if len(ranked) < 256:
            raise ValueError("official train source lacks the fixed 256-task cohort")
        tasks.extend(_relabel(task, index) for index, task in enumerate(ranked[:256]))
    if len(tasks) != 512:
        raise ValueError("warm-start cohort is not exactly 256+256 tasks")
    ordered_states = training_states(tasks)
    state_digest = state_order_digest(ordered_states)
    source = {
        "archive_sha256": ARCHIVE_SHA256,
        "train_entry": TRAIN_ENTRY,
        "train_entry_bytes": entry_bytes,
        "train_entry_sha256": entry_digest.hexdigest(),
        "eval_snapshot_path": EVAL_SNAPSHOT_PATH,
        "eval_snapshot_sha256": EVAL_SNAPSHOT_SHA256,
        "eval_gate_path": EVAL_GATE_PATH,
        "eval_gate_sha256": EVAL_GATE_SHA256,
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "dataset": "MuSiQue-Ans",
        "cohort": "short_document_linear_chain",
        "selection": {"namespace": SELECTION_NAMESPACE, "law": SELECTION_LAW},
        "state_order": {
            "namespace": STATE_ORDER_NAMESPACE,
            "law": STATE_ORDER_LAW,
            "seed": TRAINING_SEED,
            "sha256": state_digest,
        },
        "source": source,
        "counts": {"train_tasks": {"2": 256, "4": 256}, "train_states": 3072},
        "tasks": [
            {
                "row_id": task.row_id,
                "depth": task.depth,
                "question": task.question,
                "candidates": [{"title": c.title, "text": c.text} for c in task.candidates],
                "support_path": list(task.support_path),
                "source_row_sha256": task.source_row_sha256,
                "selection_key": task.selection_key,
                "permutations": [list(value) for value in task.permutations],
            }
            for task in tasks
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(canonical_json(payload))
    return payload


def _validate_permutation(value: object) -> tuple[int, ...]:
    if type(value) is not list or tuple(value) not in (PERMUTATION_A, PERMUTATION_B):
        raise ValueError("warm-start permutation is not frozen")
    return tuple(cast(list[int], value))


def state_order_digest(rows: Sequence[tuple[WarmTask, int, int, int]]) -> str:
    identities = [
        [task.source_row_sha256, permutation, step] for task, permutation, step, _ in rows
    ]
    return _sha256_bytes(canonical_json(identities))


def load_snapshot(path: Path, expected_sha256: str) -> tuple[WarmTask, ...]:
    data = path.read_bytes()
    if _sha256_bytes(data) != expected_sha256:
        raise ValueError("warm-start snapshot hash does not match config")
    value = json.loads(data)
    if type(value) is not dict or set(value) != {
        "cohort",
        "counts",
        "dataset",
        "schema_version",
        "selection",
        "source",
        "state_order",
        "tasks",
    }:
        raise ValueError("warm-start snapshot schema is not exact")
    if (
        value["schema_version"] != 1
        or value["dataset"] != "MuSiQue-Ans"
        or value["cohort"] != "short_document_linear_chain"
    ):
        raise ValueError("warm-start snapshot binding is wrong")
    selection = value["selection"]
    if selection != {"namespace": SELECTION_NAMESPACE, "law": SELECTION_LAW}:
        raise ValueError("warm-start selection law is not frozen")
    state_order = value["state_order"]
    if (
        type(state_order) is not dict
        or set(state_order)
        != {
            "law",
            "namespace",
            "seed",
            "sha256",
        }
        or state_order["namespace"] != STATE_ORDER_NAMESPACE
        or state_order["law"] != STATE_ORDER_LAW
    ):
        raise ValueError("warm-start state-order law is not frozen")
    if state_order["seed"] != TRAINING_SEED or type(state_order["sha256"]) is not str:
        raise ValueError("warm-start state-order binding is invalid")
    tasks: list[WarmTask] = []
    raw_tasks = value["tasks"]
    if type(raw_tasks) is not list or len(raw_tasks) != 512:
        raise ValueError("warm-start tasks are not exactly 512")
    seen_rows: set[str] = set()
    for raw in cast(list[object], raw_tasks):
        if type(raw) is not dict or set(raw) != {
            "candidates",
            "depth",
            "permutations",
            "question",
            "row_id",
            "selection_key",
            "source_row_sha256",
            "support_path",
        }:
            raise ValueError("warm-start task schema is not exact")
        item = cast(dict[str, object], raw)
        depth = item["depth"]
        row_id = item["row_id"]
        if type(depth) is not int or depth not in (2, 4) or type(row_id) is not str:
            raise ValueError("warm-start task depth/id is invalid")
        if not _ROW_ID.fullmatch(row_id) or row_id in seen_rows:
            raise ValueError("warm-start row id is invalid")
        seen_rows.add(row_id)
        source_row = _as_string(item["source_row_sha256"], "source row hash")
        selection_digest = _as_string(item["selection_key"], "selection key")
        if selection_digest != selection_key(depth, source_row):
            raise ValueError("warm-start selection key is not authenticated")
        candidates_raw = item["candidates"]
        if type(candidates_raw) is not list or len(candidates_raw) != 8:
            raise ValueError("warm-start task candidate count is not eight")
        candidates: list[WarmCandidate] = []
        for candidate in cast(list[object], candidates_raw):
            if type(candidate) is not dict or set(candidate) != {"text", "title"}:
                raise ValueError("warm-start candidate schema is not exact")
            candidate_value = cast(dict[str, object], candidate)
            candidates.append(
                WarmCandidate(
                    _as_string(candidate_value["title"], "title"),
                    _as_string(candidate_value["text"], "text"),
                )
            )
        path_titles = item["support_path"]
        if (
            type(path_titles) is not list
            or len(path_titles) != depth
            or not all(type(x) is str for x in path_titles)
        ):
            raise ValueError("warm-start support path is invalid")
        permutations_raw = item["permutations"]
        if type(permutations_raw) is not list or len(permutations_raw) != 2:
            raise ValueError("warm-start permutation count is invalid")
        permutations = tuple(
            _validate_permutation(value) for value in cast(list[object], permutations_raw)
        )
        task = WarmTask(
            row_id,
            depth,
            _as_string(item["question"], "question"),
            tuple(candidates),
            tuple(cast(list[str], path_titles)),
            source_row,
            selection_digest,
            permutations,
        )
        if len({candidate.title for candidate in task.candidates}) != 8 or not set(
            task.support_path
        ).issubset({candidate.title for candidate in task.candidates}):
            raise ValueError("warm-start titles are invalid")
        if len(tuple(task.states())) != task.depth * 2:
            raise ValueError("warm-start state topology is invalid")
        tasks.append(task)
    counts = {str(depth): sum(task.depth == depth for task in tasks) for depth in (2, 4)}
    if counts != {"2": 256, "4": 256}:
        raise ValueError("warm-start cohort counts are not exact")
    for depth in (2, 4):
        keys = [task.selection_key for task in tasks if task.depth == depth]
        if keys != sorted(keys):
            raise ValueError("warm-start selection order is not frozen")
    rows = training_states(tasks)
    if state_order["sha256"] != state_order_digest(rows):
        raise ValueError("warm-start state-order digest does not match")
    if value["counts"] != {"train_tasks": {"2": 256, "4": 256}, "train_states": 3072}:
        raise ValueError("warm-start counts are not exact")
    return tuple(tasks)


def warm_prompt(task: WarmTask, step: int, permutation_index: int) -> str:
    if not 0 <= step < task.depth or not 0 <= permutation_index < len(task.permutations):
        raise ValueError("warm-start state index is invalid")
    permutation = task.permutations[permutation_index]
    title_to_index = {candidate.title: index for index, candidate in enumerate(task.candidates)}
    selected_titles = task.support_path[:step]
    selected_indices = {title_to_index[title] for title in selected_titles}
    evidence = [task.candidates[title_to_index[title]] for title in selected_titles]
    active = [task.candidates[index] for index in permutation if index not in selected_indices]
    evidence_text = (
        "(none)"
        if not evidence
        else "\n\n".join(f"{candidate.title}\n{candidate.text}" for candidate in evidence)
    )
    candidate_text = "\n\n".join(
        f"[{index}] {candidate.title}\n{candidate.text}"
        for index, candidate in enumerate(active, 1)
    )
    return PROMPT_TEMPLATE.format(
        question=task.question, evidence=evidence_text, candidates=candidate_text
    )


def training_states(tasks: Sequence[WarmTask]) -> tuple[tuple[WarmTask, int, int, int], ...]:
    rows: list[tuple[WarmTask, int, int, int]] = []
    for task in tasks:
        for permutation_index, step, _active, target in task.states():
            rows.append((task, permutation_index, step, target))
    rows.sort(key=lambda row: _state_key(row[0].source_row_sha256, row[1], row[2]))
    if len(rows) != 3072:
        raise ValueError("warm-start training states are not exactly 3072")
    return tuple(rows)


def config_payload(
    snapshot_sha256: str, train_entry_sha256: str, state_digest: str
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": "musique-ans-support-warm-start-v1",
        "dataset": "MuSiQue-Ans",
        "cohort": "short_document_linear_chain",
        "selection": {"namespace": SELECTION_NAMESPACE, "law": SELECTION_LAW},
        "state_order": {
            "namespace": STATE_ORDER_NAMESPACE,
            "law": STATE_ORDER_LAW,
            "seed": TRAINING_SEED,
            "sha256": state_digest,
        },
        "source": {
            "archive_sha256": ARCHIVE_SHA256,
            "train_entry": TRAIN_ENTRY,
            "train_entry_bytes": TRAIN_ENTRY_BYTES,
            "train_entry_sha256": train_entry_sha256,
            "snapshot_path": "data/musique-ans-support-warm-start-v1.json",
            "snapshot_sha256": snapshot_sha256,
            "eval_snapshot_path": EVAL_SNAPSHOT_PATH,
            "eval_snapshot_sha256": EVAL_SNAPSHOT_SHA256,
            "eval_gate_path": EVAL_GATE_PATH,
            "eval_gate_sha256": EVAL_GATE_SHA256,
            "license": "CC BY 4.0",
        },
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "transformers": "5.6.2",
            "dtype": "bfloat16",
            "cuda_devices": 1,
            "use_cache": False,
        },
        "choice": {
            "labels": list(LABELS),
            "prompt_template": PROMPT_TEMPLATE,
            "chat_template": {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
            "max_input_tokens": 8192,
            "candidate_count": 8,
            "permutations_per_train_task": 2,
        },
        "training": {
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "expected_replacements": 224,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            },
            "microbatch": 1,
            "gradient_accumulation": 16,
            "effective_batch": 16,
            "epochs": 1,
            "updates": 192,
            "learning_rate": 0.0001,
            "warmup_fraction": 0.03,
            "gradient_clip": 1.0,
            "seed": TRAINING_SEED,
            "loss": "assistant_response_only",
            "checkpoint_selection": False,
            "material_adapter_delta_l2_min": 1e-8,
        },
        "adapter": {
            "format": "safetensors",
            "path": "runs/musique-ans-support-warm-start-v1/adapter.safetensors",
            "base_only": False,
            "publish_only_after_gate": True,
        },
        "gate": {
            "base_exact": {
                "2_teacher": 31,
                "2_paths": 9,
                "4_teacher": 30,
                "4_positions": [7, 4, 2, 17],
                "4_paths": 0,
            },
            "post": {
                "2_teacher_min": 29,
                "2_paths_min": 7,
                "4_teacher_min": 22,
                "4_position_min": 8,
                "4_paths_min": 3,
                "4_path_rate_strict_gt": 0.1,
                "k10_mixed_sum_strict_gt": 5.0,
            },
        },
        "runtime": {
            "rate_cap_usd_per_hour": 2.0,
            "spot": "one qualifying A10080",
            "one_attempt": True,
            "retry": False,
            "hard_process_seconds": 3600,
            "pod_seconds": 4500,
            "cost_cap_usd": 2.5,
            "teardown_reserve_seconds": 600,
            "persistent_storage": False,
            "hard_timeout_owner": "external orchestrator",
        },
        "authority": {
            "model_calls": True,
            "training": True,
            "prime": False,
            "source_rows": False,
            "parquet": False,
            "science": False,
        },
        "state_counts": {"train_tasks": {"2": 256, "4": 256}, "train_states": 3072},
    }


def load_config(path: Path) -> tuple[dict[str, object], str]:
    data = path.read_bytes()
    digest = _sha256_bytes(data)
    value = json.loads(data)
    if (
        type(value) is not dict
        or value.get("schema_version") != 1
        or value.get("experiment") != "musique-ans-support-warm-start-v1"
        or value.get("dataset") != "MuSiQue-Ans"
        or value.get("cohort") != "short_document_linear_chain"
    ):
        raise ValueError("warm-start config identity is invalid")
    if value.get("selection") != {"namespace": SELECTION_NAMESPACE, "law": SELECTION_LAW}:
        raise ValueError("warm-start config selection law is not frozen")
    state = value.get("state_order")
    if (
        type(state) is not dict
        or state.get("namespace") != STATE_ORDER_NAMESPACE
        or state.get("law") != STATE_ORDER_LAW
        or state.get("seed") != TRAINING_SEED
    ):
        raise ValueError("warm-start config state-order law is not frozen")
    source = value.get("source")
    if (
        type(source) is not dict
        or source.get("archive_sha256") != ARCHIVE_SHA256
        or source.get("train_entry") != TRAIN_ENTRY
        or source.get("train_entry_bytes") != TRAIN_ENTRY_BYTES
        or source.get("train_entry_sha256") != TRAIN_ENTRY_SHA256
    ):
        raise ValueError("warm-start source binding is invalid")
    model = value.get("model")
    if (
        type(model) is not dict
        or model.get("name") != MODEL_NAME
        or model.get("revision") != MODEL_REVISION
        or model.get("transformers") != "5.6.2"
        or model.get("dtype") != "bfloat16"
        or model.get("cuda_devices") != 1
        or model.get("use_cache") is not False
    ):
        raise ValueError("warm-start model binding is invalid")
    source_map = cast(dict[str, object], source)
    if (
        source_map.get("snapshot_path") != "data/musique-ans-support-warm-start-v1.json"
        or type(source_map.get("snapshot_sha256")) is not str
        or len(cast(str, source_map["snapshot_sha256"])) != 64
    ):
        raise ValueError("warm-start snapshot binding is invalid")
    adapter = value.get("adapter")
    if adapter != {
        "format": "safetensors",
        "path": "runs/musique-ans-support-warm-start-v1/adapter.safetensors",
        "base_only": False,
        "publish_only_after_gate": True,
    }:
        raise ValueError("warm-start adapter binding is invalid")
    return value, digest


def check_payload(
    config: Mapping[str, object], config_sha256: str, tasks: Sequence[WarmTask]
) -> dict[str, object]:
    source = cast(Mapping[str, object], config["source"])
    state = cast(Mapping[str, object], config["state_order"])
    rows = training_states(tasks)
    digest = state_order_digest(rows)
    if state["sha256"] != digest:
        raise ValueError("config state-order digest does not match snapshot")
    return {
        "mode": "check",
        "experiment": config["experiment"],
        "config_sha256": config_sha256,
        "snapshot_sha256": source["snapshot_sha256"],
        "selection_namespace": SELECTION_NAMESPACE,
        "state_order_sha256": digest,
        "train_tasks": len(tasks),
        "train_tasks_by_depth": {
            "2": sum(task.depth == 2 for task in tasks),
            "4": sum(task.depth == 4 for task in tasks),
        },
        "train_states": len(rows),
        "model_calls": False,
        "training_updates": 0,
        "gold_decomposition_in_prompt": False,
    }


def write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json(dict(payload)))


__all__ = [
    "MODEL_NAME",
    "MODEL_REVISION",
    "PROMPT_TEMPLATE",
    "SELECTION_LAW",
    "SELECTION_NAMESPACE",
    "STATE_ORDER_LAW",
    "STATE_ORDER_NAMESPACE",
    "TRAINING_SEED",
    "WarmCandidate",
    "WarmTask",
    "build_snapshot",
    "check_payload",
    "config_payload",
    "load_config",
    "load_snapshot",
    "normalize_identity",
    "selection_key",
    "state_order_digest",
    "training_states",
    "warm_prompt",
    "write_exclusive",
]
