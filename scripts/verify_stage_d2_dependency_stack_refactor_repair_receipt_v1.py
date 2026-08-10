"""Verify the direct-child, non-authorizing dependency-stack repair receipt."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import py_compile
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import NoReturn, Protocol, cast

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_RELATIVE = "reports/stage-d2-dependency-stack-refactor-repair-receipt-v1.json"
VERIFIER_RELATIVE = "scripts/verify_stage_d2_dependency_stack_refactor_repair_receipt_v1.py"
CHECKPOINT_COMMIT = "fb174cea06cb1e7d45a3b4eed9b33dbf6eba8e8f"
CHECKPOINT_TREE = "6ea81ed015dc2027c44586fda895cc0f114f84d6"
CHECKPOINT_PARENT = "158678ed4566067e0e099ff02076da2fa6cb9359"
SOURCE_RELATIVE = "src/redco/analysis/stage_d_dependency_stack.py"
RECEIPT_BYTES = 7141
RECEIPT_SHA256 = "107a8126547c04e83f6cdce84c37b50493bf49c057a6aa38f9461dadf3c211c1"
HISTORICAL_AUTHORIZED_SHA256 = (
    "7feba4914177d9475ecc936447cd5b7aa0a6e9df891fcf7592fa84ccb9c4c95e"
)
CHECKPOINT_SOURCE_SHA256 = "f41995745652c430290f0bb1d541f05654a1d2ba3b9c4fb93242394a64d57d13"
V1_AUTHORITY_KEYS = frozenset(
    {
        "candidate_selection_authorized",
        "launch_authorized",
        "phase_2_authorized",
        "provider_calls_authorized",
        "science_authorized",
        "support_launch_authorized",
    }
)
V2_AUTHORITY_KEYS = frozenset(
    {
        "candidate_selection_authorized",
        "launch_authorized",
        "phase_2_live_authorized",
        "provider_calls_authorized",
        "science_authorized",
        "support_launch_authorized",
    }
)
FORBIDDEN_RECEIPT_KEYS = frozenset(
    {
        "checkpoint_2_commit",
        "checkpoint_2_tree",
        "receipt_git_blob",
        "receipt_sha256",
        "self_hash",
        "verifier_git_blob",
        "verifier_sha256",
    }
)

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class FileBinding:
    path: str
    git_blob: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class PredecessorBinding:
    file: FileBinding
    source_binding_sha256: str
    state: str


@dataclass(frozen=True, slots=True)
class SourceBinding:
    commit: str
    tree: str
    git_blob: str
    sha256: str
    bytes: int


CHECKPOINT_SOURCE = FileBinding(
    SOURCE_RELATIVE,
    "5278115fd4e30bf09d1975214c6e8712841c7db2",
    CHECKPOINT_SOURCE_SHA256,
    26090,
)
CONTRACTS_SOURCE = FileBinding(
    "src/redco/contracts.py",
    "3c1d04f076ec44bee05fc8c2ea3f35fb7d09ad31",
    "6e01b4b08c0f9a100406a7f239cd50a5de8c823f143d990b1f5234ac2b8e04e8",
    8369,
)
INTEGRITY_SOURCE = FileBinding(
    "src/redco/integrity.py",
    "3f8e3b43fc34f6fd2299aaf43df746a44b46fe02",
    "6abd798350b602415ac36609c1a524623b3a3052e3016a5b1894df2dc0b072af",
    1491,
)
SOURCE_MODULES = (
    ("redco.contracts", CONTRACTS_SOURCE),
    ("redco.integrity", INTEGRITY_SOURCE),
    ("redco.analysis.stage_d_dependency_stack", CHECKPOINT_SOURCE),
)
HOSTILE_PYC_MARKER = "__redco_hostile_unchecked_hash_pyc__"
HISTORICAL_SOURCE = SourceBinding(
    "a1dddbc4da870fe887545b23783b029f7ce1e0a4",
    "4d005e667dfa47f740ca583e6e9a4a1edcac229e",
    "8beb54c6fb0cdc5f620633a83c2f73eec4245dbd",
    HISTORICAL_AUTHORIZED_SHA256,
    24473,
)
PARENT_SOURCE = SourceBinding(
    CHECKPOINT_PARENT,
    "749d791a0fd9fadbff6416e35edcbb03e62585e4",
    "8ec559b41be2f6bc7ecfd3c231c2bdbb69ce7676",
    "1b2ad24c65c75be6e77672704e7b01468b2dd5a3cc877d37180162e765c70bb8",
    26387,
)
PREDECESSORS = (
    PredecessorBinding(
        FileBinding(
            "reports/stage-d2-qa-localization-audit-v1.json",
            "f994db5b98c2eee8bb31570d17560320d17d4c32",
            "9b6d9b1db80da5ce8c8d344923c1a009e0f13e7fa53c5df150eb556f3b684baf",
            23486,
        ),
        HISTORICAL_AUTHORIZED_SHA256,
        "non_authorizing_cpu_audit_complete",
    ),
    PredecessorBinding(
        FileBinding(
            "reports/stage-d2-qa-localization-audit-v2.json",
            "6ab4ee158d57fd4a649836c51af3adced44794df",
            "d8baa34cbd55c74213913825e186580ab9bef69ad34174f38cebfd071460d5a9",
            7946,
        ),
        PARENT_SOURCE.sha256,
        "non_authorizing_cpu_infrastructure_evidence",
    ),
)
MANIFESTS = (
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v10.json",
        "9d56268f334a634994e372cde74aeb02b1b13496",
        "4ca4b4ce0f3a5a36b18aee2892165dccb2d60afade2469198674b1c06403b58c",
        6983,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v11-1.json",
        "f2b80fd4e0bc9656aaf8d78c7777d9f2f58d394e",
        "1e29e81f6ece0f5e20e4c6f980fd0025a49f0cd9d6e8fe5746364e682883c996",
        7316,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v11-2.json",
        "7c11ef394317d02b3f729be9f90ea24d358509ee",
        "681791c039804924f0bd3ccaca42653128088442bf5f803436fb11b2d53cbe47",
        7316,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v11.json",
        "cf62093b9a79ef2f15c4b97dac0f8464e820ef12",
        "b6ec955b71c2377101653c913f0fb5e4eb30698724c7dca45cd55aa4fd997372",
        7316,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v12.json",
        "1e80216d8475fe91f00e0d6413833e2938241bcb",
        "cda524c6ecea9821b1e36290da64df465aa46fad9ec174881c24d3dc895b2831",
        7316,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v2.json",
        "48a5334f18d38cb54b2b737bea9ddd2f4d675238",
        "aee8afa82e9a3cf2e5656409eb69d81b5bd8929414f8cda4125d333986be6cc4",
        5337,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v3.json",
        "102aedb4bd68ba225512282f2277610691e0fc3d",
        "15f62e4d00777ae89e2776714c7ff49015dbca32290d1a9a87294cace3f9f92d",
        6573,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v4.json",
        "6eb05661be5fff34fb5dda66a0a91c75b4e9d3a2",
        "a18405b7a8d12af89709a52aff33f29ea8507e40e661caf75bc07f7feb4bb8e0",
        6573,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v5.json",
        "6e203befaaa9c3b12149f64b9912cc4dabeda09f",
        "57993dac96a67ca512d89088b56389eba7ad058c6e2917e92782897c0559e0f4",
        6573,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v7.json",
        "6ae12ff9905f46fd03530451ffa1a75a769a429a",
        "8e34cb551afacfb5105afcfc0e335268d41c029a7766f134df509a6a90151625",
        6774,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v8.json",
        "675d8b6a2cd93d1b919cdef02e32dc6a05c4ee3c",
        "49ec5cd10c007b698f06f6336efd39079a097dc5e59398c62a679358405b88a2",
        6774,
    ),
    FileBinding(
        "configs/stage-d/stage-d1-dependency-stack-v9.json",
        "32447da29dcfabc38580ec512ec2cab747b4d9c7",
        "2d55e1edd835df0db5359a34ca20a7c73b9d70caa1f53a6361bed8421042c8f1",
        6983,
    ),
)


class FrozenManifest(Protocol):
    def to_bytes(self) -> bytes: ...


class FrozenManifestType(Protocol):
    def from_bytes(self, raw: bytes) -> FrozenManifest: ...


class BoundMethod(Protocol):
    @property
    def __func__(self) -> object: ...


def _die(message: str) -> NoReturn:
    raise RuntimeError(message)


class _RejectRedcoFinder(MetaPathFinder):
    """Fail if authenticated Redco execution escapes into import machinery."""

    def __init__(self) -> None:
        self.attempted: list[str] = []

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        del path, target
        if fullname == "redco" or fullname.startswith("redco."):
            self.attempted.append(fullname)
            _die("authenticated Redco source escaped into import machinery")
        return None


class _SourceOnlyImporter:
    """Resolve protected Redco imports only from authenticated memory modules."""

    def __init__(self, modules: dict[str, ModuleType]) -> None:
        self._modules = modules

    def __call__(
        self,
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        package = globals.get("__package__") if globals is not None else None
        if level and type(package) is str and package.startswith("redco"):
            _die("authenticated Redco source attempted a relative import")
        if name == "redco" or name.startswith("redco."):
            module = self._modules.get(name)
            if module is None:
                _die(f"authenticated Redco source requested an unbound module: {name}")
            return module if fromlist else self._modules["redco"]
        return cast(object, builtins.__import__(name, globals, locals, fromlist, level))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in pairs:
        if key in value:
            _die(f"JSON object contains a duplicate key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> NoReturn:
    _die(f"JSON contains a non-finite constant: {value}")


def _parse_json(raw: bytes, subject: str) -> dict[str, JsonValue]:
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        _die(f"{subject} has a forbidden byte representation")
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{subject} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        _die(f"{subject} root is not an exact JSON object")
    return cast(dict[str, JsonValue], value)


def _canonical_receipt(value: Mapping[str, JsonValue]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _file_payload(binding: FileBinding) -> dict[str, JsonValue]:
    return {
        "bytes": binding.bytes,
        "git_blob": binding.git_blob,
        "path": binding.path,
        "sha256": binding.sha256,
    }


def _expected_receipt() -> dict[str, JsonValue]:
    authority: dict[str, JsonValue] = {
        key: False
        for key in (
            "candidate_selection_authorized",
            "gpu_execution_authorized",
            "heldout_evaluation_authorized",
            "launch_authorized",
            "model_calls_authorized",
            "prime_authorized",
            "provider_calls_authorized",
            "provisioning_authorized",
            "readiness_authorized",
            "science_authorized",
            "scientific_transition_authorized",
            "source_access_authorized",
            "support_launch_authorized",
            "training_authorized",
            "wallet_authorized",
        )
    }
    entries: list[JsonValue] = []
    for binding in MANIFESTS:
        payload = _file_payload(binding)
        payload["roundtrip_exact"] = True
        entries.append(payload)
    predecessor_payloads: list[JsonValue] = []
    for predecessor in PREDECESSORS:
        payload = _file_payload(predecessor.file)
        payload["source_binding_sha256"] = predecessor.source_binding_sha256
        payload["state"] = predecessor.state
        predecessor_payloads.append(payload)
    return {
        "authorization": authority,
        "checkpoint_1": {
            "commit": CHECKPOINT_COMMIT,
            "parent": CHECKPOINT_PARENT,
            "source": _file_payload(CHECKPOINT_SOURCE),
            "tree": CHECKPOINT_TREE,
        },
        "domain": "redco-stage-d2-dependency-stack-refactor-repair-receipt-v1",
        "frozen_manifest_roundtrips": {
            "entries": entries,
            "expected_count": 12,
            "verified_count": 12,
        },
        "future_launch": {
            "requirement": "separately_reviewed_readiness_v2_lineage",
            "satisfied_by_this_receipt": False,
            "separate_review_required": True,
        },
        "historical_launch_disposition": {
            "authorized_source_sha256": HISTORICAL_AUTHORIZED_SHA256,
            "checkpoint_source_sha256": CHECKPOINT_SOURCE_SHA256,
            "historical_launch_remains_fail_closed": True,
            "source_matches_authorization": False,
        },
        "predecessor_artifacts": predecessor_payloads,
        "schema_version": 1,
        "sequencing": {
            "direct_child_of_checkpoint_1_required": True,
            "future_commit_identity_intentionally_excluded": True,
            "receipt_self_identity_intentionally_excluded": True,
        },
        "source_lineage": {
            "historical_launch_source": {
                "bytes": HISTORICAL_SOURCE.bytes,
                "commit": HISTORICAL_SOURCE.commit,
                "git_blob": HISTORICAL_SOURCE.git_blob,
                "path": SOURCE_RELATIVE,
                "sha256": HISTORICAL_SOURCE.sha256,
                "tree": HISTORICAL_SOURCE.tree,
            },
            "refactor_parent_source": {
                "bytes": PARENT_SOURCE.bytes,
                "commit": PARENT_SOURCE.commit,
                "git_blob": PARENT_SOURCE.git_blob,
                "path": SOURCE_RELATIVE,
                "sha256": PARENT_SOURCE.sha256,
                "tree": PARENT_SOURCE.tree,
            },
        },
        "state": "non_authorizing_repository_refactor_receipt",
    }


def _walk_keys(value: JsonValue) -> Sequence[str]:
    keys: list[str] = []
    if type(value) is dict:
        for key, item in value.items():
            keys.append(key)
            keys.extend(_walk_keys(item))
    elif type(value) is list:
        for item in value:
            keys.extend(_walk_keys(item))
    return keys


def _contained_file(relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        _die(f"repository path is not a safe relative path: {relative}")
    path = ROOT.joinpath(*pure.parts)
    root = ROOT.resolve(strict=True)
    current = ROOT
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            _die(f"repository path crosses a symlink: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not path.is_file():
        _die(f"repository path is not a contained regular file: {relative}")
    return path


def _read_contained(relative: str) -> bytes:
    return _contained_file(relative).read_bytes()


def _git(*args: str, accepted: frozenset[int] = frozenset({0})) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode not in accepted:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        _die(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _git_text(*args: str) -> str:
    return _git(*args).decode("ascii").strip()


def _require_object(object_id: str, expected_type: str) -> None:
    if _git_text("cat-file", "-t", object_id) != expected_type:
        _die(f"Git object has the wrong type: {object_id}")


def _commit_headers(commit: str) -> tuple[str, tuple[str, ...]]:
    _require_object(commit, "commit")
    raw = _git("cat-file", "-p", commit)
    header, separator, _body = raw.partition(b"\n\n")
    if not separator:
        _die(f"Git commit lacks a header/body separator: {commit}")
    tree: str | None = None
    parents: list[str] = []
    for line in header.splitlines():
        if line.startswith(b"tree "):
            tree = line[5:].decode("ascii")
        elif line.startswith(b"parent "):
            parents.append(line[7:].decode("ascii"))
    if tree is None:
        _die(f"Git commit lacks a tree: {commit}")
    return tree, tuple(parents)


def _tree_blob(commit: str, relative: str, *, allow_absent: bool = False) -> str | None:
    raw = _git("ls-tree", "-z", commit, "--", relative)
    if not raw:
        if allow_absent:
            return None
        _die(f"Git tree lacks required path at {commit}: {relative}")
    if raw.count(b"\0") != 1 or not raw.endswith(b"\0"):
        _die(f"Git tree returned an ambiguous path entry: {relative}")
    metadata, separator, path = raw[:-1].partition(b"\t")
    if not separator or path.decode("utf-8") != relative:
        _die(f"Git tree path identity differs: {relative}")
    fields = metadata.decode("ascii").split()
    if len(fields) != 3 or fields[:2] != ["100644", "blob"]:
        _die(f"Git tree path is not a regular non-executable blob: {relative}")
    return fields[2]


def _blob_bytes(blob: str) -> bytes:
    _require_object(blob, "blob")
    return _git("cat-file", "blob", blob)


def _verify_raw(raw: bytes, binding: FileBinding, subject: str) -> None:
    if len(raw) != binding.bytes or _sha256(raw) != binding.sha256:
        _die(f"{subject} raw bytes differ: {binding.path}")


def _verify_checkpoint_file(binding: FileBinding, *, current_file: bool = True) -> bytes:
    blob = _tree_blob(CHECKPOINT_COMMIT, binding.path)
    if blob != binding.git_blob:
        _die(f"checkpoint Git blob differs: {binding.path}")
    raw = _blob_bytes(binding.git_blob)
    _verify_raw(raw, binding, "checkpoint Git blob")
    if current_file:
        current = _read_contained(binding.path)
        _verify_raw(current, binding, "retained file")
        if current != raw:
            _die(f"retained file differs from checkpoint Git bytes: {binding.path}")
    return raw


def _verify_checkpoint() -> None:
    tree, parents = _commit_headers(CHECKPOINT_COMMIT)
    if tree != CHECKPOINT_TREE or parents != (CHECKPOINT_PARENT,):
        _die("checkpoint-1 commit/tree/sole-parent binding differs")
    _require_object(CHECKPOINT_TREE, "tree")
    _verify_checkpoint_file(CHECKPOINT_SOURCE)


def _verify_source_lineage() -> None:
    for binding in (HISTORICAL_SOURCE, PARENT_SOURCE):
        tree, _parents = _commit_headers(binding.commit)
        if tree != binding.tree:
            _die(f"source-lineage commit tree differs: {binding.commit}")
        blob = _tree_blob(binding.commit, SOURCE_RELATIVE)
        if blob != binding.git_blob:
            _die(f"source-lineage Git blob differs: {binding.commit}")
        raw = _blob_bytes(binding.git_blob)
        if len(raw) != binding.bytes or _sha256(raw) != binding.sha256:
            _die(f"source-lineage raw bytes differ: {binding.commit}")
    _git("merge-base", "--is-ancestor", HISTORICAL_SOURCE.commit, CHECKPOINT_COMMIT)
    _git("merge-base", "--is-ancestor", CHECKPOINT_PARENT, CHECKPOINT_COMMIT)
    if HISTORICAL_SOURCE.sha256 == CHECKPOINT_SOURCE.sha256:
        _die("historical launch authorization unexpectedly matches the repaired source")


def _mapping(value: JsonValue, subject: str) -> dict[str, JsonValue]:
    if type(value) is not dict:
        _die(f"{subject} is not an exact JSON object")
    return value


def _false_authority(value: JsonValue, keys: frozenset[str], subject: str) -> None:
    authority = _mapping(value, subject)
    if frozenset(authority) != keys or any(
        type(item) is not bool or item for item in authority.values()
    ):
        _die(f"{subject} does not contain the exact false authority set")


def _verify_predecessors() -> None:
    parsed: list[dict[str, JsonValue]] = []
    for predecessor in PREDECESSORS:
        raw = _verify_checkpoint_file(predecessor.file)
        value = _parse_json(raw, predecessor.file.path)
        if value.get("state") != predecessor.state:
            _die(f"predecessor state differs: {predecessor.file.path}")
        bindings = _mapping(value.get("file_bindings"), "predecessor file_bindings")
        if bindings.get(SOURCE_RELATIVE) != predecessor.source_binding_sha256:
            _die(f"predecessor source binding differs: {predecessor.file.path}")
        parsed.append(value)
    _false_authority(parsed[0].get("authorization"), V1_AUTHORITY_KEYS, "v1 authorization")
    _false_authority(parsed[1].get("authorization"), V2_AUTHORITY_KEYS, "v2 authorization")
    prior = _mapping(parsed[1].get("prior_audit"), "v2 prior_audit")
    expected_prior: dict[str, JsonValue] = {
        "path": PREDECESSORS[0].file.path,
        "sha256": PREDECESSORS[0].file.sha256,
        "unchanged": True,
    }
    if prior != expected_prior:
        _die("v2 does not exactly authenticate the immutable v1 predecessor")


def _manifest_paths_at_checkpoint() -> tuple[str, ...]:
    raw = _git(
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        CHECKPOINT_COMMIT,
        "--",
        "configs/stage-d",
    )
    paths = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        relative = item.decode("utf-8")
        name = PurePosixPath(relative).name
        if name.startswith("stage-d1-dependency-stack-v") and name.endswith(".json"):
            paths.append(relative)
    return tuple(sorted(paths))


def _source_label(binding: FileBinding) -> str:
    return f"git:{CHECKPOINT_COMMIT}:{binding.path}"


def _package_shell(name: str, modules: dict[str, ModuleType]) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = f"git:{CHECKPOINT_COMMIT}:<package-shell:{name}>"
    module.__package__ = name
    module.__dict__["__cached__"] = None
    module.__dict__["__path__"] = []
    modules[name] = module
    sys.modules[name] = module
    return module


def _exec_authenticated_module(
    name: str,
    binding: FileBinding,
    raw: bytes,
    modules: dict[str, ModuleType],
    source_import: _SourceOnlyImporter,
) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = _source_label(binding)
    module.__package__ = name.rpartition(".")[0]
    module.__dict__["__cached__"] = None
    source_builtins: dict[str, object] = dict(vars(builtins))
    source_builtins["__import__"] = source_import
    module.__dict__["__builtins__"] = source_builtins
    modules[name] = module
    sys.modules[name] = module
    parent_name, _, attribute = name.rpartition(".")
    setattr(modules[parent_name], attribute, module)
    code = compile(
        raw,
        _source_label(binding),
        "exec",
        dont_inherit=True,
        optimize=0,
    )
    exec(code, module.__dict__)
    if module.__dict__.get(HOSTILE_PYC_MARKER) is True:
        _die(f"hostile unchecked-hash bytecode influenced authenticated source: {name}")
    return module


def _require_code_label(value: object, label: str, subject: str) -> None:
    code = getattr(value, "__code__", None)
    if getattr(code, "co_filename", None) != label:
        _die(f"{subject} was not compiled from authenticated checkpoint source")


def _source_only_roundtrips(
    source_raw: Mapping[str, bytes],
    raw_manifests: Sequence[bytes],
) -> int:
    preloaded_redco = tuple(
        name for name in sys.modules if name == "redco" or name.startswith("redco.")
    )
    if preloaded_redco:
        _die("Redco package state was loaded before authenticated source execution")
    modules: dict[str, ModuleType] = {}
    original_meta_path = tuple(sys.meta_path)
    finder = _RejectRedcoFinder()
    sys.meta_path.insert(0, finder)
    try:
        redco = _package_shell("redco", modules)
        analysis = _package_shell("redco.analysis", modules)
        redco.__dict__["analysis"] = analysis
        source_import = _SourceOnlyImporter(modules)
        for name, binding in SOURCE_MODULES:
            _exec_authenticated_module(
                name,
                binding,
                source_raw[name],
                modules,
                source_import,
            )
        contracts = modules["redco.contracts"]
        integrity = modules["redco.integrity"]
        dependency_stack = modules["redco.analysis.stage_d_dependency_stack"]
        manifest_type = cast(
            FrozenManifestType,
            dependency_stack.__dict__["StageDDependencyStackManifest"],
        )
        if getattr(manifest_type, "__module__", None) != dependency_stack.__name__:
            _die("manifest type does not belong to the authenticated source module")
        if dependency_stack.__dict__.get("canonical_json") is not contracts.__dict__.get(
            "canonical_json"
        ):
            _die("manifest canonical JSON helper is not the authenticated source function")
        if dependency_stack.__dict__.get(
            "resolve_contained_file"
        ) is not integrity.__dict__.get("resolve_contained_file"):
            _die("manifest path helper is not the authenticated source function")
        if dependency_stack.__dict__.get("_sha256") is not integrity.__dict__.get(
            "sha256_bytes"
        ):
            _die("manifest hash helper is not the authenticated source function")
        _require_code_label(
            contracts.__dict__["canonical_json"],
            _source_label(CONTRACTS_SOURCE),
            "canonical JSON helper",
        )
        _require_code_label(
            integrity.__dict__["resolve_contained_file"],
            _source_label(INTEGRITY_SOURCE),
            "contained-path helper",
        )
        _require_code_label(
            integrity.__dict__["sha256_bytes"],
            _source_label(INTEGRITY_SOURCE),
            "SHA-256 helper",
        )
        from_bytes = cast(BoundMethod, manifest_type.from_bytes)
        _require_code_label(
            from_bytes.__func__,
            _source_label(CHECKPOINT_SOURCE),
            "manifest parser",
        )
        passed = 0
        for raw in raw_manifests:
            parsed = manifest_type.from_bytes(raw)
            if parsed.to_bytes() != raw:
                _die("frozen manifest did not round-trip through authenticated source")
            passed += 1
        if finder.attempted:
            _die("authenticated source consulted Redco import machinery")
        return passed
    finally:
        sys.meta_path[:] = original_meta_path
        for name, module in reversed(tuple(modules.items())):
            if sys.modules.get(name) is module:
                del sys.modules[name]


def _write_hostile_unchecked_hash_pyc_tree(
    root: Path,
    source_raw: Mapping[str, bytes],
) -> None:
    cache_tag = sys.implementation.cache_tag
    if type(cache_tag) is not str or not cache_tag:
        _die("Python runtime lacks a bytecode cache tag for the hostile probe")
    (root / "redco" / "analysis").mkdir(parents=True)
    (root / "redco" / "__init__.py").write_bytes(b"")
    (root / "redco" / "analysis" / "__init__.py").write_bytes(b"")
    for name, _binding in SOURCE_MODULES:
        source = root.joinpath(*name.split(".")).with_suffix(".py")
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"{HOSTILE_PYC_MARKER} = True\n", encoding="utf-8", newline="\n")
        pyc = source.parent / "__pycache__" / f"{source.stem}.{cache_tag}.pyc"
        pyc.parent.mkdir(parents=True, exist_ok=True)
        compiled = py_compile.compile(
            str(source),
            cfile=str(pyc),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
        pyc_raw = pyc.read_bytes()
        if compiled != str(pyc) or len(pyc_raw) < 16 or int.from_bytes(pyc_raw[4:8], "little") != 1:
            _die(f"hostile probe did not create unchecked-hash bytecode: {name}")
        source.write_bytes(source_raw[name])


def _prove_hostile_unchecked_hash_pyc_is_live(root: Path) -> None:
    names = tuple(name for name, _binding in SOURCE_MODULES)
    probe = (
        "import importlib,sys\n"
        "sys.path.insert(0,sys.argv[1])\n"
        f"names={names!r}\n"
        "modules=[importlib.import_module(name) for name in names]\n"
        f"assert all(getattr(module,{HOSTILE_PYC_MARKER!r},False) is True for module in modules)\n"
        "print('hostile-unchecked-hash-pyc:3')\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, str(root)],
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0 or result.stdout.decode("ascii").strip() != (
        "hostile-unchecked-hash-pyc:3"
    ):
        _die("ordinary -B import did not execute the hostile unchecked-hash pyc fixture")


def _verify_manifests() -> int:
    expected_paths = tuple(binding.path for binding in MANIFESTS)
    if len(MANIFESTS) != 12 or len(set(expected_paths)) != 12:
        _die("frozen manifest allowlist is not exactly 12 unique paths")
    if _manifest_paths_at_checkpoint() != expected_paths:
        _die("checkpoint frozen manifest family differs from the exact allowlist")
    current_paths = tuple(
        sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "configs/stage-d").glob("stage-d1-dependency-stack-v*.json")
        )
    )
    if current_paths != expected_paths:
        _die("retained frozen manifest family differs from the exact allowlist")
    raw_manifests = [_verify_checkpoint_file(binding) for binding in MANIFESTS]
    source_raw = {
        name: _verify_checkpoint_file(binding) for name, binding in SOURCE_MODULES
    }
    with tempfile.TemporaryDirectory(prefix="redco-hostile-unchecked-pyc-") as temporary:
        hostile_root = Path(temporary)
        _write_hostile_unchecked_hash_pyc_tree(hostile_root, source_raw)
        _prove_hostile_unchecked_hash_pyc_is_live(hostile_root)
        original_path = tuple(sys.path)
        sys.path.insert(0, str(hostile_root))
        try:
            passed = _source_only_roundtrips(source_raw, raw_manifests)
        finally:
            sys.path[:] = original_path
    if passed != 12:
        _die("frozen manifest round-trip count differs from 12")
    return passed


def _untracked_paths() -> tuple[str, ...]:
    raw = _git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    return tuple(item.decode("utf-8") for item in raw.split(b"\0") if item)


def _require_no_tracked_changes(reference: str) -> None:
    _git(
        "diff",
        "--quiet",
        "--ignore-submodules=all",
        reference,
        "--",
        ".",
    )


def _verify_precommit_mode() -> str:
    if _git_text("rev-parse", "HEAD") != CHECKPOINT_COMMIT:
        _die("precommit mode requires checkpoint 1 at HEAD")
    for relative in (RECEIPT_RELATIVE, VERIFIER_RELATIVE):
        if _tree_blob(CHECKPOINT_COMMIT, relative, allow_absent=True) is not None:
            _die(f"checkpoint 1 unexpectedly contains checkpoint-2 path: {relative}")
    _require_no_tracked_changes(CHECKPOINT_COMMIT)
    expected = tuple(sorted((RECEIPT_RELATIVE, VERIFIER_RELATIVE)))
    if tuple(sorted(_untracked_paths())) != expected:
        _die("precommit working tree differs from the exact two-file checkpoint-2 boundary")
    return "precommit_candidate"


def _verify_committed_mode() -> str:
    head = _git_text("rev-parse", "HEAD")
    _tree, parents = _commit_headers(head)
    if parents != (CHECKPOINT_COMMIT,):
        _die("committed mode requires HEAD to be the direct sole child of checkpoint 1")
    raw_paths = _git(
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-z",
        head,
    )
    changed_paths = tuple(sorted(item.decode("utf-8") for item in raw_paths.split(b"\0") if item))
    expected_paths = tuple(sorted((RECEIPT_RELATIVE, VERIFIER_RELATIVE)))
    if changed_paths != expected_paths:
        _die("checkpoint 2 changed paths differ from the exact two-file boundary")
    for relative in expected_paths:
        if _tree_blob(CHECKPOINT_COMMIT, relative, allow_absent=True) is not None:
            _die(f"checkpoint-2 path already existed at checkpoint 1: {relative}")
        if _tree_blob(head, relative) is None:
            _die(f"checkpoint 2 lacks its required path: {relative}")
    _require_no_tracked_changes(head)
    if _untracked_paths():
        _die("committed verification requires a clean tree aside from ignored submodules")
    return "committed_direct_child"


def _verify_sequencing(mode: str) -> str:
    if mode == "precommit":
        return _verify_precommit_mode()
    if mode == "committed":
        return _verify_committed_mode()
    if _git_text("rev-parse", "HEAD") == CHECKPOINT_COMMIT:
        return _verify_precommit_mode()
    return _verify_committed_mode()


def verify(mode: str) -> dict[str, JsonValue]:
    receipt_raw = _read_contained(RECEIPT_RELATIVE)
    if len(receipt_raw) != RECEIPT_BYTES or _sha256(receipt_raw) != RECEIPT_SHA256:
        _die("repair receipt raw bytes differ from the reviewed verifier binding")
    receipt = _parse_json(receipt_raw, "repair receipt")
    expected = _expected_receipt()
    if receipt_raw != _canonical_receipt(expected) or receipt != expected:
        _die("repair receipt is not the exact canonical reviewed payload")
    forbidden = FORBIDDEN_RECEIPT_KEYS.intersection(_walk_keys(receipt))
    if forbidden:
        _die(f"repair receipt contains a self-referential key: {sorted(forbidden)}")
    authority = _mapping(receipt["authorization"], "receipt authorization")
    if any(type(value) is not bool or value for value in authority.values()):
        _die("repair receipt grants authority")
    _verify_checkpoint()
    _verify_source_lineage()
    _verify_predecessors()
    sequencing = _verify_sequencing(mode)
    manifest_count = _verify_manifests()
    return {
        "authority_granted": False,
        "checkpoint_1": CHECKPOINT_COMMIT,
        "frozen_manifest_roundtrips": manifest_count,
        "historical_launch_remains_fail_closed": True,
        "receipt": {
            "bytes": RECEIPT_BYTES,
            "path": RECEIPT_RELATIVE,
            "sha256": RECEIPT_SHA256,
        },
        "sequencing": sequencing,
        "unchecked_hash_pyc_isolation": True,
        "state": "verified_non_authorizing_repository_refactor_receipt",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "precommit", "committed"),
        default="auto",
        help="verify the uncommitted candidate, the committed direct child, or infer the mode",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = verify(cast(str, args.mode))
    print(json.dumps(result, allow_nan=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
