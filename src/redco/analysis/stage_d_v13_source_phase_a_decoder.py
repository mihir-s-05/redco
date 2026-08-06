"""Pinned source metadata and the real Phase-A bounded decoder wall.

The complete Parquet object is authenticated from bytes and metadata.  Logical
rows are decoded only through a single PyArrow record batch containing ordinals
0--179.  The historical ``datasets`` streaming adapter is probed for its
effective batch policy, but is never used to produce Phase-A rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from redco.analysis.stage_d_v13_draft import canonical_json_bytes, sha256_bytes, sha256_json
from redco.analysis.stage_d_v13_draft_inputs import sha256_file

SOURCE_REPOSITORY = "allenai/qasper"
SOURCE_REVISION = "06806e4608976fc2fac0a090ac425d5b2b29caf4"
SOURCE_SEMANTIC_COMMIT = "fdc9d8214fbab5dd782958601db4d678e6934a54"
SOURCE_PATH = "qasper/train/0000.parquet"
SOURCE_LOGICAL_URL = (
    f"https://huggingface.co/datasets/allenai/qasper/resolve/{SOURCE_REVISION}/{SOURCE_PATH}"
)
SOURCE_ARTIFACT_RELATIVE = (
    "datasets/stage-d/source-auth-v13/"
    "qasper-train-06806e4608976fc2fac0a090ac425d5b2b29caf4-0000.parquet"
)
SOURCE_BYTES = 14_374_550
SOURCE_SHA256 = "9af08092ee26c4f700202c1f90d1592b662926f23f3a308a10ff0a53345e37fe"
SOURCE_ETAG = '"87300ef1b3d9845edc33b25c47a57ee009d6f0ef9a05771df767cf83e354ffd9"'
SOURCE_LFS_OID_SHA256 = SOURCE_SHA256
SOURCE_ROW_COUNT = 888
SOURCE_ROW_GROUPS = 1
SOURCE_SCHEMA_SHA256 = "85c0addf53d5cfbcb709744f75a3a5f47272b854db453173f6cde96666cf965b"
SOURCE_FIELDS = ("id", "title", "abstract", "full_text", "qas", "figures_and_tables")
SUPPORTED_PYTHON = "3.12.3"
SUPPORTED_DATASETS = "5.0.0"
SUPPORTED_PYARROW = "25.0.0"
PHASE_A_CUTOFF = 179
PHASE_A_BATCH_SIZE = PHASE_A_CUTOFF + 1
PHASE_B_RESUME_START_ORDINAL = PHASE_A_CUTOFF + 1
PHASE_B_RESUME_BATCH_SIZE = PHASE_A_BATCH_SIZE
PHASE_B_AUTHORIZATION_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-phase-b-authorization-c-v1.json"
)
PHASE_B_BINDING_RELATIVE = (
    "configs/stage-d/v13-draft/stage-d1-support-v13-phase-b-binding-b-v1.json"
)
PHASE_C3_AUTHORIZATION_RELATIVE = (
    "configs/stage-d/v13-draft/"
    "stage-d1-support-v13-phase-b-authorization-c3-v1.json"
)
PHASE_A_VERSION = "stage-d-v13-source-authentication-phase-a-v2"

# Repair R is deliberately pinned to the already reviewed F -> B history.
# These identities are authentication constants, not values supplied by a
# future authorization artifact or by a caller.
FOUNDATION_F_COMMIT = "2970411b1ad3a68ec9a7a7f98d41428e3a8301e9"
FOUNDATION_F_PARENT_COMMIT = "c41fd18446cecf1c7c98e5aa3a962d1568072c1b"
FOUNDATION_F_TREE_SHA1 = "8a1c3bde8233075c0bdd63c3626a7c8e936bad8a"
BINDING_B_COMMIT = "ca04e22bf9c9e19f1aa2dad092403d5d9668269c"
BINDING_B_SHA256 = "93915d220f1bcb6357f0910633e6d8f2b5fa7d5727f71ae665f34d5bf36c1e8e"
BINDING_B_GIT_BLOB_SHA1 = "fc35188882620db0d7827cb82582750065bdf211"
FOUNDATION_MANIFEST_RELATIVE = (
    "reports/stage-d1-support-v13-foundation-tree-manifest-f1.json"
)
FOUNDATION_MANIFEST_SHA256 = "533ccd093d4e5f26fe77090da6dc1c16a9bfa220f449de9f2c25f55d921dd416"
FOUNDATION_MANIFEST_GIT_BLOB_SHA1 = "00cf30f1853171598ee5410bba844e4ce76da51f"
PHASE_A_CONFIG_RELATIVE = (
    "configs/stage-d/v13-draft/"
    "stage-d1-support-source-authentication-phase-a-v1.json"
)
PHASE_A_CONFIG_SHA256 = "0579823d966463f6bd33df3596174c132c553ed646d23aff000c62a2d4aa651c"
B_PRESELECTION_CHECKPOINT_SHA256 = BINDING_B_SHA256
REPAIR_DIFF_ALLOWLIST = (
    "src/redco/analysis/stage_d_v13_source_phase_a_bindings.py",
    "src/redco/analysis/stage_d_v13_source_phase_a_decoder.py",
    "tests/test_stage_d_v13_source_phase_a.py",
)

_FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = _DEFAULT_PROJECT_ROOT

_AUTH_GIT_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "PATHEXT",
        "SystemRoot",
        "SYSTEMROOT",
        "WINDIR",
        "HOME",
        "USERPROFILE",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
    }
)
_AUTH_GIT_ENV_BLOCKLIST = frozenset(
    {
        "GIT_REPLACE_REF_BASE",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_INDEX_FILE",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
    }
)


def _authentication_git_environment() -> dict[str, str]:
    """Return the minimal caller-independent environment for Git auth."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _AUTH_GIT_ENV_ALLOWLIST and key not in _AUTH_GIT_ENV_BLOCKLIST
    }
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = "NUL" if os.name == "nt" else os.devnull
    return environment


def hardened_git(
    repo_root: Path, *arguments: str, text: bool = False
) -> subprocess.CompletedProcess[Any]:
    """Run Git for authentication with replacement and path redirection disabled."""

    return subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-c",
            f"safe.directory={repo_root}",
            *arguments,
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=text,
        env=_authentication_git_environment(),
    )


class PhaseAWallError(ValueError):
    """Raised when a decoder produces a logical row past the Phase-A wall."""


class PhaseBResumeAuthorizationError(ValueError):
    """Raised before the dormant post-cutoff decoder can read any row."""


class PhaseBResumeUnavailable(PhaseBResumeAuthorizationError):
    """Foundation F has no reviewed authorization artifact C."""


_RESUME_DECODER_INVOCATIONS = 0


def git_blob_sha1(data: bytes) -> str:
    """Return the Git blob object id for exact bytes without invoking Git."""

    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def verify_git_blob_identity(
    path: Path,
    *,
    expected_sha256: str,
    expected_git_blob_sha1: str,
) -> dict[str, str | int]:
    """Authenticate a file by both content SHA-256 and its prospective Git blob id."""

    data = path.read_bytes()
    actual_sha256 = sha256_bytes(data)
    actual_blob = git_blob_sha1(data)
    if actual_sha256 != expected_sha256 or actual_blob != expected_git_blob_sha1:
        raise ValueError(f"Git-object binding differs for {path}")
    return {
        "bytes": len(data),
        "sha256": actual_sha256,
        "git_blob_sha1": actual_blob,
    }


@dataclass(slots=True)
class DecoderInstrumentation:
    """Bounded observations from the actual PyArrow decoder."""

    batch_size: int = PHASE_A_BATCH_SIZE
    use_threads: bool = False
    row_groups: tuple[int, ...] = (0,)
    decoded_objects: list[dict[str, int | str]] = field(default_factory=list)
    canonicalized_ordinals: list[int] = field(default_factory=list)

    def record_batch(self, *, start: int, rows: int) -> None:
        end = start + rows - 1
        self.decoded_objects.append(
            {
                "kind": "pyarrow_record_batch",
                "rows": rows,
                "start_ordinal": start,
                "end_ordinal": end,
            }
        )
        if rows > PHASE_A_BATCH_SIZE or end > PHASE_A_CUTOFF:
            raise PhaseAWallError("bounded decoder produced a logical object beyond ordinal 179")

    def record_canonicalized(self, ordinal: int) -> None:
        if ordinal > PHASE_A_CUTOFF:
            raise PhaseAWallError("canonicalization crossed the Phase-A ordinal wall")
        self.canonicalized_ordinals.append(ordinal)

    def to_payload(self) -> dict[str, Any]:
        max_ordinal = max(
            (int(item["end_ordinal"]) for item in self.decoded_objects),
            default=None,
        )
        max_cardinality = max((int(item["rows"]) for item in self.decoded_objects), default=0)
        post_cutoff_logical = any(
            int(item["end_ordinal"]) > PHASE_A_CUTOFF for item in self.decoded_objects
        )
        post_cutoff_canonical = any(
            ordinal > PHASE_A_CUTOFF for ordinal in self.canonicalized_ordinals
        )
        return {
            "batch_size": self.batch_size,
            "use_threads": self.use_threads,
            "row_groups": list(self.row_groups),
            "decoded_object_count": len(self.decoded_objects),
            "decoded_objects": self.decoded_objects,
            "canonicalized_ordinals": self.canonicalized_ordinals,
            "maximum_decoded_ordinal": max_ordinal,
            "maximum_decoded_cardinality": max_cardinality,
            "all_decoded_objects_within_cutoff": bool(self.decoded_objects)
            and all(
                int(item["end_ordinal"]) <= PHASE_A_CUTOFF
                and int(item["rows"]) <= PHASE_A_BATCH_SIZE
                for item in self.decoded_objects
            ),
            "post_cutoff_logical_row_materialized": post_cutoff_logical,
            "post_cutoff_row_canonicalized": post_cutoff_canonical,
            "physical_io_may_span_row_group": True,
        }


@dataclass(slots=True)
class ResumeDecoderInstrumentation:
    """Bounded observations exposed by the dormant Phase-B resume seam."""

    decoded_objects: list[dict[str, int | str]] = field(default_factory=list)
    materialized_ranges: list[list[int]] = field(default_factory=list)
    canonicalized_ordinals: list[int] = field(default_factory=list)
    evaluated_ordinals: list[int] = field(default_factory=list)
    emitted_ordinals: list[int] = field(default_factory=list)

    def record_decoded(self, *, start: int, rows: int) -> None:
        end = start + rows - 1
        previous_end = (
            int(self.decoded_objects[-1]["end_ordinal"]) if self.decoded_objects else None
        )
        if rows <= 0 or (previous_end is not None and start != previous_end + 1):
            raise ValueError("resume decoder logical batch order differs")
        self.decoded_objects.append(
            {
                "kind": "pyarrow_record_batch",
                "rows": rows,
                "start_ordinal": start,
                "end_ordinal": end,
            }
        )

    def record_materialized(self, *, start: int, rows: int) -> None:
        end = start + rows - 1
        if start < PHASE_B_RESUME_START_ORDINAL:
            raise ValueError("resume materialization began before ordinal 180")
        self.materialized_ranges.append([start, end])

    def record_canonicalized(self, ordinal: int) -> None:
        if ordinal < PHASE_B_RESUME_START_ORDINAL:
            raise ValueError("resume canonicalization began before ordinal 180")
        self.canonicalized_ordinals.append(ordinal)

    def record_evaluated(self, ordinal: int) -> None:
        if ordinal < PHASE_B_RESUME_START_ORDINAL:
            raise ValueError("resume evaluation began before ordinal 180")
        self.evaluated_ordinals.append(ordinal)

    def record_emitted(self, ordinal: int) -> None:
        if ordinal < PHASE_B_RESUME_START_ORDINAL or (
            self.emitted_ordinals and ordinal != self.emitted_ordinals[-1] + 1
        ):
            raise ValueError("resume emitted ordinals are not contiguous")
        self.emitted_ordinals.append(ordinal)

    def to_payload(self) -> dict[str, Any]:
        return {
            "decoded_objects": self.decoded_objects,
            "materialized_ranges": self.materialized_ranges,
            "canonicalized_ordinals": self.canonicalized_ordinals,
            "evaluated_ordinals": self.evaluated_ordinals,
            "emitted_ordinals": self.emitted_ordinals,
            "invocation_count_observed": 1,
        }


def _schema_descriptor(schema: Any) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def source_schema_sha256(path: Path) -> str:
    """Hash Parquet schema metadata without decoding a row."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Phase A requires the pinned PyArrow decoder") from error
    parquet_file = parquet.ParquetFile(path)
    return cast(str, sha256_bytes(str(parquet_file.schema_arrow).encode("utf-8")))


def canonical_source_row_bytes(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def source_row_sha256(row: Mapping[str, Any]) -> str:
    return cast(str, sha256_bytes(canonical_source_row_bytes(row)))


def _require_supported_versions(pyarrow: Any, datasets_version: str) -> None:
    python_version = ".".join(map(str, sys.version_info[:3]))
    if python_version != SUPPORTED_PYTHON:
        raise RuntimeError(f"Phase A requires Python {SUPPORTED_PYTHON}, got {python_version}")
    if str(pyarrow.__version__) != SUPPORTED_PYARROW:
        raise RuntimeError(
            f"Phase A requires PyArrow {SUPPORTED_PYARROW}, got {pyarrow.__version__}"
        )
    if datasets_version != SUPPORTED_DATASETS:
        raise RuntimeError(
            f"Phase A requires datasets {SUPPORTED_DATASETS}, got {datasets_version}"
        )


def authenticate_source_artifact(root: Path) -> dict[str, Any]:
    """Authenticate bytes/schema/metadata without deserializing any row."""

    path = root / SOURCE_ARTIFACT_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(
            f"exact pinned Parquet artifact is missing: {SOURCE_ARTIFACT_RELATIVE}"
        )
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != SOURCE_BYTES or digest != SOURCE_SHA256:
        raise ValueError("exact pinned Parquet artifact bytes are not authenticated")
    try:
        import pyarrow
        import pyarrow.parquet as parquet

        import datasets
    except ImportError as error:
        raise RuntimeError("Phase A requires the pinned PyArrow/datasets environment") from error
    datasets_version = str(getattr(datasets, "__version__", ""))
    pyarrow_version = str(getattr(pyarrow, "__version__", ""))
    _require_supported_versions(pyarrow, datasets_version)
    parquet_file = parquet.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata is None:
        raise ValueError("Parquet metadata is unavailable")
    schema = parquet_file.schema_arrow
    schema_digest = source_schema_sha256(path)
    if (
        metadata.num_rows != SOURCE_ROW_COUNT
        or metadata.num_row_groups != SOURCE_ROW_GROUPS
        or schema_digest != SOURCE_SCHEMA_SHA256
    ):
        raise ValueError("pinned Parquet schema or row metadata differs")
    return {
        "repository": SOURCE_REPOSITORY,
        "logical_url": SOURCE_LOGICAL_URL,
        "revision": SOURCE_REVISION,
        "semantic_source_commit": SOURCE_SEMANTIC_COMMIT,
        "path": SOURCE_PATH,
        "local_artifact": SOURCE_ARTIFACT_RELATIVE,
        "bytes": size,
        "sha256": digest,
        "git_lfs_oid_sha256": SOURCE_LFS_OID_SHA256,
        "etag": SOURCE_ETAG,
        "row_count": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "schema_sha256": schema_digest,
        "schema_fields": _schema_descriptor(schema),
        "decoder": {
            "python": SUPPORTED_PYTHON,
            "datasets": datasets_version,
            "pyarrow": pyarrow_version,
            "loader": (
                "pyarrow.parquet.ParquetFile.iter_batches(batch_size=180, "
                "row_groups=[0], use_threads=False)"
            ),
            "batch_size": PHASE_A_BATCH_SIZE,
            "use_threads": False,
            "logical_readahead": False,
            "metadata_only_for_authentication": True,
        },
    }


def legacy_datasets_decoder_probe(path: Path) -> dict[str, Any]:
    """Inspect the pinned legacy adapter without iterating or decoding a row."""

    try:
        import datasets
    except ImportError as error:
        raise RuntimeError("Phase A requires the pinned datasets environment") from error
    load_dataset = getattr(datasets, "load_dataset", None)
    if not callable(load_dataset):
        raise RuntimeError("Phase A requires the pinned datasets environment")
    dataset = load_dataset(
        "parquet", data_files={"train": str(path)}, split="train", streaming=True
    )
    iterable = dataset._ex_iterable
    generate_tables = iterable.generate_tables_fn
    config = generate_tables.__self__.config
    configured_batch_size = config.batch_size
    effective_batch_size = configured_batch_size or SOURCE_ROW_COUNT
    return {
        "adapter": "datasets.load_dataset(parquet, streaming=True)",
        "configured_batch_size": configured_batch_size,
        "effective_first_table_batch_size": effective_batch_size,
        "source_row_count": SOURCE_ROW_COUNT,
        "would_cross_phase_a_wall": effective_batch_size > PHASE_A_BATCH_SIZE,
        "rows_iterated": 0,
        "rows_deserialized": 0,
    }


def bounded_source_rows(
    path: Path,
    *,
    cutoff: int = PHASE_A_CUTOFF,
    instrumentation: DecoderInstrumentation | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Decode exactly the bounded prefix through the real pinned PyArrow path."""

    if cutoff != PHASE_A_CUTOFF:
        raise ValueError("Phase A uses the frozen ordinal-179 cutoff")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Phase A requires the pinned PyArrow decoder") from error
    observer = instrumentation or DecoderInstrumentation()
    parquet_file = parquet.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata is None or metadata.num_rows != SOURCE_ROW_COUNT:
        raise ValueError("bounded decoder metadata differs from authenticated source")
    next_ordinal = 0
    for batch in parquet_file.iter_batches(
        batch_size=PHASE_A_BATCH_SIZE,
        row_groups=[0],
        use_threads=False,
    ):
        rows = int(batch.num_rows)
        observer.record_batch(start=next_ordinal, rows=rows)
        decoded_rows = batch.to_pylist()
        if len(decoded_rows) != rows:
            raise ValueError("PyArrow batch cardinality changed during conversion")
        for row in decoded_rows:
            ordinal = next_ordinal
            if ordinal > cutoff:
                raise PhaseAWallError("PyArrow conversion crossed the Phase-A wall")
            if not isinstance(row, dict):
                raise ValueError(f"decoded source row {ordinal} is not an object")
            observer.record_canonicalized(ordinal)
            yield ordinal, row
            next_ordinal += 1
        if next_ordinal == cutoff + 1:
            return
    raise ValueError("bounded decoder ended before the frozen Phase-A cutoff")


def resume_decoder_invocation_count() -> int:
    """Return how many authorized dormant resume iterators were constructed."""

    return _RESUME_DECODER_INVOCATIONS


def _resume_contract_hash() -> str:
    """Return the frozen v2 digest used only by the committed Binding B."""

    from redco.analysis.stage_d_v13_source_phase_a_bindings import (
        PHASE_B_RESUME_CONTRACT_V2_SHA256,
    )

    return PHASE_B_RESUME_CONTRACT_V2_SHA256


def _resume_contract_v3_hash() -> str:
    """Return the separately versioned repaired runtime contract digest."""

    from redco.analysis.stage_d_v13_source_phase_a_bindings import (
        PHASE_B_RESUME_CONTRACT_V3,
    )

    return cast(str, sha256_json(PHASE_B_RESUME_CONTRACT_V3))


def _validate_synthetic_resume_authorization(
    path: Path, authorization: Mapping[str, Any]
) -> tuple[Any, Any]:
    required = {
        "state",
        "phase_b_authorized",
        "preselection_checkpoint_sha256",
        "source_sha256",
        "schema_sha256",
        "decoder_contract_sha256",
        "start_ordinal",
        "batch_size",
        "row_groups",
        "use_threads",
        "source_order",
    }
    if not required.issubset(authorization):
        raise PhaseBResumeAuthorizationError("reviewed resume authorization fields are incomplete")
    if (
        authorization["state"] != "reviewed_preselection_checkpoint"
        or authorization["phase_b_authorized"] is not True
        or not isinstance(authorization["preselection_checkpoint_sha256"], str)
        or len(authorization["preselection_checkpoint_sha256"]) != 64
        or authorization["start_ordinal"] != PHASE_B_RESUME_START_ORDINAL
        or authorization["batch_size"] != PHASE_B_RESUME_BATCH_SIZE
        or authorization["row_groups"] != [0]
        or authorization["use_threads"] is not False
        or authorization["source_order"] != "physical_ordinal"
        or authorization["decoder_contract_sha256"] != _resume_contract_hash()
    ):
        raise PhaseBResumeAuthorizationError(
            "resume authorization does not match the frozen contract"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_source_sha256 = sha256_file(path)
    actual_schema_sha256 = source_schema_sha256(path)
    if authorization["source_sha256"] != actual_source_sha256:
        raise PhaseBResumeAuthorizationError("resume source hash differs")
    if authorization["schema_sha256"] != actual_schema_sha256:
        raise PhaseBResumeAuthorizationError("resume source schema differs")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Phase B requires the pinned PyArrow decoder") from error
    parquet_file = parquet.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata is None or metadata.num_rows <= PHASE_B_RESUME_START_ORDINAL:
        raise PhaseBResumeAuthorizationError("resume source has no post-cutoff rows")
    return parquet, parquet_file


def _git_text(repo_root: Path, *arguments: str) -> str:
    result = hardened_git(repo_root, *arguments, text=True)
    if result.returncode != 0:
        raise PhaseBResumeAuthorizationError(
            f"Git object lookup failed for future C: {' '.join(arguments)}"
        )
    stdout = result.stdout
    if not isinstance(stdout, str):
        raise PhaseBResumeAuthorizationError("Git text authentication returned non-text output")
    return stdout.strip()


def _git_ancestor(repo_root: Path, ancestor: str, descendant: str) -> None:
    result = hardened_git(
        repo_root, "merge-base", "--is-ancestor", ancestor, descendant, text=True
    )
    if result.returncode != 0:
        raise PhaseBResumeAuthorizationError(
            f"future C commit is not descended from reviewed commit {ancestor}"
        )


def _git_blob_at_commit(repo_root: Path, commit: str, relative: str) -> bytes:
    blob = _git_text(repo_root, "rev-parse", "--verify", f"{commit}:{relative}")
    if _git_text(repo_root, "cat-file", "-t", blob) != "blob":
        raise PhaseBResumeAuthorizationError(f"future C path is not a Git blob: {relative}")
    result = hardened_git(repo_root, "cat-file", "blob", blob)
    if result.returncode != 0 or not isinstance(result.stdout, bytes):
        raise PhaseBResumeAuthorizationError(f"future C blob read failed: {relative}")
    return result.stdout


def _commit_parents(repo_root: Path, commit: str) -> list[str]:
    values = _git_text(repo_root, "rev-list", "--parents", "-n", "1", commit).split()
    if not values or values[0] != commit or any(
        not _FULL_COMMIT_SHA.fullmatch(value) for value in values
    ):
        raise PhaseBResumeAuthorizationError("future C commit identity is not a full Git SHA")
    return values[1:]


def _validate_c_binding(
    repo_root: Path,
    commit: str,
    record: Any,
    *,
    relative: str,
    label: str,
) -> bytes:
    if relative == SOURCE_ARTIFACT_RELATIVE:
        raise PhaseBResumeAuthorizationError(
            "production source bindings must come from the authenticated F manifest"
        )
    expected_keys = {"path", "sha256"}
    if label == "source":
        expected_keys.update({"schema_sha256", "row_count"})
    if (
        not isinstance(record, dict)
        or set(record) != expected_keys
        or record.get("path") != relative
    ):
        raise PhaseBResumeAuthorizationError(f"future C {label} path binding differs")
    raw = _git_blob_at_commit(repo_root, commit, relative)
    if record.get("sha256") != sha256_bytes(raw):
        raise PhaseBResumeAuthorizationError(f"future C {label} binding differs")
    return raw


def _git_tree_at_commit(repo_root: Path, commit: str) -> str:
    tree = _git_text(repo_root, "rev-parse", "--verify", f"{commit}^{{tree}}")
    if not _FULL_COMMIT_SHA.fullmatch(tree):
        raise PhaseBResumeAuthorizationError(
            "future checkpoint tree identity is not a full Git SHA"
        )
    return tree


def _diff_paths(repo_root: Path, parent: str, child: str) -> list[str]:
    return _git_text(
        repo_root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "--no-renames",
        "-r",
        parent,
        child,
    ).splitlines()


def _parse_canonical_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PhaseBResumeAuthorizationError(f"{label} is not JSON") from error
    if not isinstance(parsed, dict) or raw != canonical_json_bytes(parsed):
        raise PhaseBResumeAuthorizationError(f"{label} is not canonical")
    return parsed


def _require_null_candidate(parsed: Mapping[str, Any], label: str) -> None:
    expected = {
        "source_ordinal": None,
        "paper_id": None,
        "example_id": None,
        "row": None,
        "seed": None,
        "address": None,
    }
    if (
        parsed.get("candidate") != expected
        or parsed.get("seed") is not None
        or parsed.get("address") is not None
    ):
        raise PhaseBResumeAuthorizationError(f"{label} candidate identity is not null")


def _authenticated_manifest_source(
    manifest_raw: bytes,
    *,
    production: bool,
) -> dict[str, Any]:
    """Project the authenticated F manifest to the source contract.

    This reads only the small manifest Git blob.  In particular, it never
    reads the Parquet blob from Git merely to validate Binding B or C3.
    """

    manifest = _parse_canonical_object(manifest_raw, "Foundation manifest")
    source = manifest.get("source")
    qasper = manifest.get("qasper")
    required = ("path", "revision", "sha256", "schema_sha256", "row_count")
    if (
        not isinstance(source, dict)
        or not isinstance(qasper, dict)
        or not set(required).issubset(source)
        or not set(required).issubset(qasper)
    ):
        raise PhaseBResumeAuthorizationError(
            "Foundation manifest does not expose a complete source contract"
        )
    contract = {key: source[key] for key in required}
    if any(qasper.get(key) != contract[key] for key in required):
        raise PhaseBResumeAuthorizationError(
            "Foundation manifest source and QASPER contracts differ"
        )
    if production and contract != {
        "path": SOURCE_PATH,
        "revision": SOURCE_REVISION,
        "sha256": SOURCE_SHA256,
        "schema_sha256": SOURCE_SCHEMA_SHA256,
        "row_count": SOURCE_ROW_COUNT,
    }:
        raise PhaseBResumeAuthorizationError(
            "Foundation manifest source is not the pinned QASPER object"
        )
    if production:
        files = manifest.get("files")
        source_file = next(
            (
                item
                for item in files
                if isinstance(item, dict)
                and item.get("path") == SOURCE_ARTIFACT_RELATIVE
            ),
            None,
        ) if isinstance(files, list) else None
        if (
            not isinstance(source_file, dict)
            or source_file.get("bytes") != SOURCE_BYTES
            or source_file.get("sha256") != SOURCE_SHA256
        ):
            raise PhaseBResumeAuthorizationError(
                "Foundation manifest does not authenticate the source artifact file"
            )
    return contract


def _validate_b_checkpoint(
    repo_root: Path,
    foundation_commit: str,
    binding_commit: str,
    b_raw: bytes,
    *,
    production: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate B without opening or hashing the production Parquet blob."""

    from redco.analysis.stage_d_v13_source_phase_a_bindings import (
        PHASE_A_STATUS_SIGNATURE,
        PHASE_B_BINDING_DOMAIN,
    )

    parsed = _parse_canonical_object(b_raw, "Binding B")
    expected_keys = {
        "schema_version",
        "domain",
        "state",
        "draft_unfrozen",
        "foundation_only",
        "non_authorizing",
        "candidate",
        "seed",
        "address",
        "phase_b_authorized",
        "source_selection_authorized",
        "launch_authorized",
        "provider_calls_authorized",
        "model_calls_authorized",
        "prime_gpu_scientific_launch_authorized",
        "status_signature",
        "foundation_commit",
        "foundation_tree_sha1",
        "foundation_manifest",
        "bindings",
    }
    if set(parsed) != expected_keys:
        raise PhaseBResumeAuthorizationError("Binding B schema has unexpected or missing fields")
    if (
        parsed["schema_version"] != 1
        or parsed["domain"] != PHASE_B_BINDING_DOMAIN
        or parsed["state"] != "B"
        or parsed["draft_unfrozen"] is not True
        or parsed["foundation_only"] is not True
        or parsed["non_authorizing"] is not True
        or parsed["phase_b_authorized"] is not False
        or parsed["source_selection_authorized"] is not False
        or parsed["launch_authorized"] is not False
        or parsed["provider_calls_authorized"] is not False
        or parsed["model_calls_authorized"] is not False
        or parsed["prime_gpu_scientific_launch_authorized"] is not False
        or parsed["status_signature"] != PHASE_A_STATUS_SIGNATURE
        or parsed["foundation_commit"] != foundation_commit
    ):
        raise PhaseBResumeAuthorizationError(
            "Binding B state or ancestry/Foundation F identity differs"
        )
    _require_null_candidate(parsed, "Binding B")
    if production and (
        foundation_commit != FOUNDATION_F_COMMIT
        or binding_commit != BINDING_B_COMMIT
        or sha256_bytes(b_raw) != BINDING_B_SHA256
        or git_blob_sha1(b_raw) != BINDING_B_GIT_BLOB_SHA1
    ):
        raise PhaseBResumeAuthorizationError("Binding B does not match the approved checkpoint")
    tree_sha1 = _git_tree_at_commit(repo_root, foundation_commit)
    if parsed["foundation_tree_sha1"] != tree_sha1:
        raise PhaseBResumeAuthorizationError("Binding B Foundation F tree identity differs")
    if production and tree_sha1 != FOUNDATION_F_TREE_SHA1:
        raise PhaseBResumeAuthorizationError("Foundation F tree is not the approved tree")

    manifest_record = parsed["foundation_manifest"]
    if (
        not isinstance(manifest_record, dict)
        or set(manifest_record) != {"path", "sha256", "git_blob_sha1"}
        or manifest_record.get("path") != FOUNDATION_MANIFEST_RELATIVE
        or not isinstance(manifest_record.get("sha256"), str)
        or not isinstance(manifest_record.get("git_blob_sha1"), str)
    ):
        raise PhaseBResumeAuthorizationError("Binding B Foundation manifest binding is missing")
    manifest_raw = _git_blob_at_commit(repo_root, foundation_commit, FOUNDATION_MANIFEST_RELATIVE)
    actual_manifest_sha = sha256_bytes(manifest_raw)
    actual_manifest_blob = git_blob_sha1(manifest_raw)
    if (
        manifest_record["sha256"] != actual_manifest_sha
        or manifest_record["git_blob_sha1"] != actual_manifest_blob
    ):
        raise PhaseBResumeAuthorizationError("Binding B Foundation manifest binding differs")
    if production and (
        actual_manifest_sha != FOUNDATION_MANIFEST_SHA256
        or actual_manifest_blob != FOUNDATION_MANIFEST_GIT_BLOB_SHA1
    ):
        raise PhaseBResumeAuthorizationError("Foundation manifest is not the approved artifact")
    source_contract = _authenticated_manifest_source(manifest_raw, production=production)

    bindings = parsed["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != {
        "source_artifact",
        "phase_a_config",
        "decoder_contract_sha256",
    }:
        raise PhaseBResumeAuthorizationError("Binding B input bindings are missing or unexpected")
    source_binding = bindings["source_artifact"]
    expected_source_binding = {
        "path": SOURCE_ARTIFACT_RELATIVE,
        "sha256": source_contract["sha256"],
        "schema_sha256": source_contract["schema_sha256"],
        "row_count": source_contract["row_count"],
    }
    if source_binding != expected_source_binding:
        raise PhaseBResumeAuthorizationError("Binding B source metadata binding differs")
    config_binding = bindings["phase_a_config"]
    if (
        not isinstance(config_binding, dict)
        or set(config_binding) != {"path", "sha256"}
        or config_binding["path"] != PHASE_A_CONFIG_RELATIVE
        or not isinstance(config_binding["sha256"], str)
    ):
        raise PhaseBResumeAuthorizationError("Binding B phase-A config binding differs")
    config_raw = _git_blob_at_commit(repo_root, foundation_commit, PHASE_A_CONFIG_RELATIVE)
    config_sha = sha256_bytes(config_raw)
    if config_binding["sha256"] != config_sha:
        raise PhaseBResumeAuthorizationError("Binding B phase-A config bytes differ")
    if production and config_sha != PHASE_A_CONFIG_SHA256:
        raise PhaseBResumeAuthorizationError("Phase-A config is not the approved artifact")
    if bindings["decoder_contract_sha256"] != _resume_contract_hash():
        raise PhaseBResumeAuthorizationError(
            "Binding B must retain the frozen v2 decoder contract digest"
        )
    return parsed, {
        "manifest": manifest_raw,
        "config": config_raw,
        "source_contract": source_contract,
    }


def _git_path_exists_at_commit(repo_root: Path, commit: str, relative: str) -> bool:
    result = hardened_git(
        repo_root,
        "rev-parse",
        "--verify",
        f"{commit}:{relative}",
        text=True,
    )
    return result.returncode == 0


def _validate_c3_artifact(
    repo_root: Path,
    c3_raw: bytes,
    *,
    head: str,
    repair_commit: str,
    binding_commit: str,
    foundation_commit: str,
    b_raw: bytes,
    b: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    production: bool,
) -> dict[str, Any]:
    from redco.analysis.stage_d_v13_source_phase_a_bindings import (
        PHASE_A_STATUS_SIGNATURE,
        PHASE_C3_AUTHORIZATION_DOMAIN,
    )

    parsed = _parse_canonical_object(c3_raw, "Authorization C3")
    expected_keys = {
        "schema_version",
        "domain",
        "state",
        "draft_unfrozen",
        "foundation_only",
        "non_authorizing",
        "candidate",
        "seed",
        "address",
        "phase_b_authorized",
        "source_selection_authorized",
        "launch_authorized",
        "provider_calls_authorized",
        "model_calls_authorized",
        "prime_gpu_scientific_launch_authorized",
        "status_signature",
        "foundation_commit",
        "foundation_tree_sha1",
        "binding_commit",
        "repair_commit",
        "binding_artifact",
        "source",
        "bindings",
        "user_authorization",
        "resume_authorization",
    }
    if set(parsed) != expected_keys:
        raise PhaseBResumeAuthorizationError("Authorization C3 schema has unexpected fields")
    if (
        parsed["schema_version"] != 1
        or parsed["domain"] != PHASE_C3_AUTHORIZATION_DOMAIN
        or parsed["state"] != "C3"
        or parsed["draft_unfrozen"] is not False
        or parsed["foundation_only"] is not False
        or parsed["non_authorizing"] is not False
        or parsed["phase_b_authorized"] is not True
        or parsed["source_selection_authorized"] is not True
        or parsed["launch_authorized"] is not False
        or parsed["provider_calls_authorized"] is not False
        or parsed["model_calls_authorized"] is not False
        or parsed["prime_gpu_scientific_launch_authorized"] is not False
        or parsed["status_signature"] != PHASE_A_STATUS_SIGNATURE
        or parsed["foundation_commit"] != foundation_commit
        or parsed["binding_commit"] != binding_commit
        or parsed["repair_commit"] != repair_commit
        or parsed["foundation_tree_sha1"] != b["foundation_tree_sha1"]
    ):
        raise PhaseBResumeAuthorizationError("Authorization C3 state or chain identity differs")
    _require_null_candidate(parsed, "Authorization C3")
    if production and (head == BINDING_B_COMMIT or repair_commit == BINDING_B_COMMIT):
        raise PhaseBResumeAuthorizationError("Authorization C3 cannot use legacy direct B to C")

    binding_artifact = parsed["binding_artifact"]
    if (
        not isinstance(binding_artifact, dict)
        or set(binding_artifact) != {"path", "sha256", "git_blob_sha1", "status_signature"}
        or binding_artifact["path"] != PHASE_B_BINDING_RELATIVE
        or binding_artifact["sha256"] != sha256_bytes(b_raw)
        or binding_artifact["git_blob_sha1"] != git_blob_sha1(b_raw)
        or binding_artifact["status_signature"] != PHASE_A_STATUS_SIGNATURE
    ):
        raise PhaseBResumeAuthorizationError("Authorization C3 binding artifact identity differs")
    if production and (
        binding_artifact["sha256"] != BINDING_B_SHA256
        or binding_artifact["git_blob_sha1"] != BINDING_B_GIT_BLOB_SHA1
    ):
        raise PhaseBResumeAuthorizationError("Authorization C3 B artifact is not approved")

    source = parsed["source"]
    expected_source = {
        "path": source_contract["path"],
        "revision": source_contract["revision"],
        "sha256": source_contract["sha256"],
        "schema_sha256": source_contract["schema_sha256"],
        "row_count": source_contract["row_count"],
    }
    if source != expected_source:
        raise PhaseBResumeAuthorizationError("Authorization C3 source binding differs")
    bindings = parsed["bindings"]
    b_bindings = b.get("bindings")
    if not isinstance(bindings, dict) or not isinstance(b_bindings, dict):
        raise PhaseBResumeAuthorizationError("Authorization C3 input bindings are missing")
    user_authorization = parsed["user_authorization"]
    expected_user_keys = {
        "reviewed",
        "scope",
        "launch_authorized",
        "provider_calls_authorized",
        "model_calls_authorized",
    }
    if (
        not isinstance(user_authorization, dict)
        or set(user_authorization) != expected_user_keys
        or user_authorization
        != {
            "reviewed": True,
            "scope": "local_phase_b_source_selection_only",
            "launch_authorized": False,
            "provider_calls_authorized": False,
            "model_calls_authorized": False,
        }
    ):
        raise PhaseBResumeAuthorizationError("Authorization C3 user scope differs")
    expected_bindings = dict(b_bindings)
    expected_bindings["user_authorization_sha256"] = sha256_bytes(
        canonical_json_bytes(user_authorization)
    )
    if bindings != expected_bindings:
        raise PhaseBResumeAuthorizationError("Authorization C3 bindings differ from B")

    resume = parsed["resume_authorization"]
    expected_resume = {
        "state": "reviewed_preselection_checkpoint",
        "phase_b_authorized": True,
        "preselection_checkpoint_sha256": (
            BINDING_B_SHA256 if production else sha256_bytes(b_raw)
        ),
        "source_sha256": source_contract["sha256"],
        "schema_sha256": source_contract["schema_sha256"],
        "decoder_contract_v2_sha256": _resume_contract_hash(),
        "decoder_contract_v3_sha256": _resume_contract_v3_hash(),
        "start_ordinal": PHASE_B_RESUME_START_ORDINAL,
        "batch_size": PHASE_B_RESUME_BATCH_SIZE,
        "row_groups": [0],
        "use_threads": False,
        "source_order": "physical_ordinal",
    }
    if resume != expected_resume:
        raise PhaseBResumeAuthorizationError(
            "Authorization C3 nested resume contract is not exact"
        )
    return parsed


def _validate_head_authorization_c3(repo_root: Path) -> dict[str, Any]:
    """Validate only the production F -> B -> R -> C3 chain from HEAD."""

    production = repo_root.resolve() == _DEFAULT_PROJECT_ROOT
    legacy_path = repo_root / PHASE_B_AUTHORIZATION_RELATIVE
    c3_path = repo_root / PHASE_C3_AUTHORIZATION_RELATIVE
    if legacy_path.is_file():
        raise PhaseBResumeAuthorizationError(
            "legacy direct B-to-C authorization path is forbidden after Repair R"
        )
    head = _git_text(repo_root, "rev-parse", "--verify", "HEAD^{commit}")
    if not _FULL_COMMIT_SHA.fullmatch(head):
        raise PhaseBResumeAuthorizationError("current HEAD is not a full Git commit SHA")
    if _git_path_exists_at_commit(repo_root, head, PHASE_B_AUTHORIZATION_RELATIVE):
        raise PhaseBResumeAuthorizationError("legacy direct B-to-C artifact is in the HEAD tree")
    if not c3_path.is_file() or not _git_path_exists_at_commit(
        repo_root, head, PHASE_C3_AUTHORIZATION_RELATIVE
    ):
        raise PhaseBResumeAuthorizationError("Authorization C3 is absent from exact HEAD")
    c3_parents = _commit_parents(repo_root, head)
    if len(c3_parents) != 1:
        raise PhaseBResumeAuthorizationError("C3 must have exactly one direct parent R")
    repair_commit = c3_parents[0]
    repair_parents = _commit_parents(repo_root, repair_commit)
    if len(repair_parents) != 1:
        raise PhaseBResumeAuthorizationError("Repair R must have exactly one direct parent B")
    binding_commit = repair_parents[0]
    binding_parents = _commit_parents(repo_root, binding_commit)
    if len(binding_parents) != 1:
        raise PhaseBResumeAuthorizationError("Binding B must have exactly one direct parent F")
    foundation_commit = binding_parents[0]
    foundation_parents = _commit_parents(repo_root, foundation_commit)
    if len(foundation_parents) != 1:
        raise PhaseBResumeAuthorizationError("Foundation F must have exactly one direct parent")
    if production and (
        foundation_commit != FOUNDATION_F_COMMIT
        or binding_commit != BINDING_B_COMMIT
        or foundation_parents[0] != FOUNDATION_F_PARENT_COMMIT
    ):
        raise PhaseBResumeAuthorizationError("future C3 chain does not match approved F and B")
    expected_f_to_b = [f"A\t{PHASE_B_BINDING_RELATIVE}"]
    expected_b_to_r = sorted(f"M\t{path}" for path in REPAIR_DIFF_ALLOWLIST)
    expected_r_to_c3 = [f"A\t{PHASE_C3_AUTHORIZATION_RELATIVE}"]
    if sorted(_diff_paths(repo_root, foundation_commit, binding_commit)) != expected_f_to_b:
        raise PhaseBResumeAuthorizationError("F to B differs outside the Binding B artifact")
    if sorted(_diff_paths(repo_root, binding_commit, repair_commit)) != expected_b_to_r:
        raise PhaseBResumeAuthorizationError("B to R differs outside the exact repair allowlist")
    if sorted(_diff_paths(repo_root, repair_commit, head)) != expected_r_to_c3:
        raise PhaseBResumeAuthorizationError("R to C3 differs outside the C3 artifact")
    worktree_b = repo_root / PHASE_B_BINDING_RELATIVE
    b_raw = _git_blob_at_commit(repo_root, binding_commit, PHASE_B_BINDING_RELATIVE)
    c3_raw = _git_blob_at_commit(repo_root, head, PHASE_C3_AUTHORIZATION_RELATIVE)
    if not worktree_b.is_file() or worktree_b.read_bytes() != b_raw:
        raise PhaseBResumeAuthorizationError("Binding B worktree bytes differ from Git")
    if c3_path.read_bytes() != c3_raw:
        raise PhaseBResumeAuthorizationError("Authorization C3 worktree bytes differ from Git")
    b, base_inputs = _validate_b_checkpoint(
        repo_root,
        foundation_commit,
        binding_commit,
        b_raw,
        production=production,
    )
    return _validate_c3_artifact(
        repo_root,
        c3_raw,
        head=head,
        repair_commit=repair_commit,
        binding_commit=binding_commit,
        foundation_commit=foundation_commit,
        b_raw=b_raw,
        b=b,
        source_contract=cast(dict[str, Any], base_inputs["source_contract"]),
        production=production,
    )


def _require_runtime_versions_only() -> tuple[Any, str]:
    """Check exact runtime versions without touching the production source."""

    try:
        import pyarrow

        import datasets
    except ImportError as error:
        raise RuntimeError("Repair R requires the pinned PyArrow/datasets environment") from error
    datasets_version = str(getattr(datasets, "__version__", ""))
    _require_supported_versions(pyarrow, datasets_version)
    return pyarrow, datasets_version


def _validate_production_source_metadata(
    root: Path,
    source: Mapping[str, Any],
    pyarrow: Any,
) -> Any:
    """Touch the fixed source only after C3, schema, and runtime authentication."""

    expected = {
        "path": SOURCE_PATH,
        "revision": SOURCE_REVISION,
        "sha256": SOURCE_SHA256,
        "schema_sha256": SOURCE_SCHEMA_SHA256,
        "row_count": SOURCE_ROW_COUNT,
    }
    if dict(source) != expected:
        raise PhaseBResumeAuthorizationError("production source contract is not pinned")
    path = root / SOURCE_ARTIFACT_RELATIVE
    if not path.is_file():
        raise PhaseBResumeAuthorizationError("authenticated production source is absent")
    if path.stat().st_size != SOURCE_BYTES or sha256_file(path) != SOURCE_SHA256:
        raise PhaseBResumeAuthorizationError("production source bytes are not authenticated")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("Repair R requires the pinned PyArrow Parquet reader") from error
    parquet_file = parquet.ParquetFile(path)
    metadata = parquet_file.metadata
    if metadata is None or metadata.num_rows != SOURCE_ROW_COUNT:
        raise PhaseBResumeAuthorizationError("production source row count differs")
    if metadata.num_row_groups != SOURCE_ROW_GROUPS:
        raise PhaseBResumeAuthorizationError("production source row groups differ")
    if source_schema_sha256(path) != SOURCE_SCHEMA_SHA256:
        raise PhaseBResumeAuthorizationError("production source schema differs")
    return parquet_file


def _authenticate_c3_and_source() -> tuple[dict[str, Any], Any]:
    """Authenticate C3, the exact runtime, and source metadata exactly once."""

    root = _repository_root()
    if (root / PHASE_B_AUTHORIZATION_RELATIVE).is_file():
        raise PhaseBResumeAuthorizationError(
            "legacy direct B-to-C artifact cannot authorize after Repair R"
        )
    if not (root / PHASE_C3_AUTHORIZATION_RELATIVE).is_file():
        raise PhaseBResumeUnavailable(
            "Authorization C3 is absent; Repair R cannot resume Phase B"
        )
    artifact = _validate_head_authorization_c3(root)
    pyarrow, _datasets_version = _require_runtime_versions_only()
    parquet_file = _validate_production_source_metadata(
        root,
        cast(Mapping[str, Any], artifact["source"]),
        pyarrow,
    )
    return artifact, parquet_file


def validate_future_phase_b_authorization_artifact() -> dict[str, Any]:
    """Validate the non-caller-controlled F -> B -> R -> C3 activation path."""

    artifact, _parquet_file = _authenticate_c3_and_source()
    return artifact


def _repository_root() -> Path:
    return PROJECT_ROOT


def resume_source_rows() -> Iterator[tuple[int, dict[str, Any]]]:
    """Open the fixed Phase-B source only after a future committed C artifact.

    The no-argument signature prevents caller-supplied paths, hashes, flags,
    checkpoints, and mappings from acting as authority.  In Foundation F the
    absent C artifact fails before the Parquet path is even inspected.
    """

    _artifact, parquet_file = _authenticate_c3_and_source()
    global _RESUME_DECODER_INVOCATIONS
    _RESUME_DECODER_INVOCATIONS += 1
    return _resume_rows(parquet_file, ResumeDecoderInstrumentation())


def _synthetic_resume_source_rows(
    path: Path,
    *,
    authorization: Mapping[str, Any],
    instrumentation: ResumeDecoderInstrumentation | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Private synthetic-fixture seam for decoder ordering tests only."""

    _parquet, parquet_file = _validate_synthetic_resume_authorization(path, authorization)
    observer = instrumentation or ResumeDecoderInstrumentation()
    return _resume_rows(parquet_file, observer)


def _resume_rows(
    parquet_file: Any,
    observer: ResumeDecoderInstrumentation,
) -> Iterator[tuple[int, dict[str, Any]]]:
    next_ordinal = 0
    for batch in parquet_file.iter_batches(
        batch_size=PHASE_B_RESUME_BATCH_SIZE,
        row_groups=[0],
        use_threads=False,
    ):
        rows = int(batch.num_rows)
        observer.record_decoded(start=next_ordinal, rows=rows)
        batch_start = next_ordinal
        next_ordinal += rows
        if next_ordinal <= PHASE_B_RESUME_START_ORDINAL:
            continue
        if batch_start < PHASE_B_RESUME_START_ORDINAL:
            raise PhaseBResumeAuthorizationError(
                "resume batch straddles the reviewed start ordinal"
            )
        decoded_rows = batch.to_pylist()
        if len(decoded_rows) != rows:
            raise ValueError("resume PyArrow batch cardinality changed during conversion")
        observer.record_materialized(start=batch_start, rows=rows)
        for offset, row in enumerate(decoded_rows):
            ordinal = batch_start + offset
            if not isinstance(row, dict):
                raise ValueError(f"resume source row {ordinal} is not an object")
            observer.record_canonicalized(ordinal)
            observer.record_evaluated(ordinal)
            observer.record_emitted(ordinal)
            yield ordinal, row


__all__ = [
    "BINDING_B_COMMIT",
    "BINDING_B_GIT_BLOB_SHA1",
    "BINDING_B_SHA256",
    "B_PRESELECTION_CHECKPOINT_SHA256",
    "FOUNDATION_F_COMMIT",
    "FOUNDATION_F_PARENT_COMMIT",
    "FOUNDATION_F_TREE_SHA1",
    "FOUNDATION_MANIFEST_GIT_BLOB_SHA1",
    "FOUNDATION_MANIFEST_SHA256",
    "PHASE_A_BATCH_SIZE",
    "PHASE_A_CONFIG_SHA256",
    "PHASE_A_CUTOFF",
    "PHASE_A_VERSION",
    "PHASE_B_AUTHORIZATION_RELATIVE",
    "PHASE_B_BINDING_RELATIVE",
    "PHASE_B_RESUME_BATCH_SIZE",
    "PHASE_B_RESUME_START_ORDINAL",
    "PHASE_C3_AUTHORIZATION_RELATIVE",
    "PROJECT_ROOT",
    "SOURCE_ARTIFACT_RELATIVE",
    "SOURCE_BYTES",
    "SOURCE_ETAG",
    "SOURCE_FIELDS",
    "SOURCE_LFS_OID_SHA256",
    "SOURCE_LOGICAL_URL",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "SOURCE_ROW_COUNT",
    "SOURCE_ROW_GROUPS",
    "SOURCE_SCHEMA_SHA256",
    "SOURCE_SEMANTIC_COMMIT",
    "SOURCE_SHA256",
    "SUPPORTED_DATASETS",
    "SUPPORTED_PYARROW",
    "SUPPORTED_PYTHON",
    "DecoderInstrumentation",
    "PhaseAWallError",
    "PhaseBResumeAuthorizationError",
    "PhaseBResumeUnavailable",
    "ResumeDecoderInstrumentation",
    "_synthetic_resume_source_rows",
    "authenticate_source_artifact",
    "bounded_source_rows",
    "canonical_source_row_bytes",
    "git_blob_sha1",
    "hardened_git",
    "legacy_datasets_decoder_probe",
    "resume_decoder_invocation_count",
    "resume_source_rows",
    "source_row_sha256",
    "source_schema_sha256",
    "validate_future_phase_b_authorization_artifact",
    "verify_git_blob_identity",
]
