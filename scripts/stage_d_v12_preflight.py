"""Fresh-process pytest preflight for the Stage D v12 support audit."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json

RUNTIME_COMMIT = "7b54f25912a9842c000291d20314dc831eca776b"
EXPECTED_NODEID_COUNT = 134
EXPECTED_NODEID_SHA256 = "f6cf217011e208e81e1dd77e553268006e26acb3911d8c5f027237498ba4793f"
EXPECTED_SELECTOR_COUNT = 20
EXPECTED_SELECTOR_SHA256 = "612f0270e9512986f05f97a63f1d3c498358a35e303f7ca4a6ac2789cb2bfd7c"
EXPECTED_PREFLIGHT_PLUGIN_SHA256 = (
    "4a91782dd61761285dd23f3b3bf0b8f26ea757b80fdc2e17e012423132b0bdb7"
)
EXPECTED_PREFLIGHT_RUNNER = [
    "uv",
    "run",
    "--active",
    "--no-sync",
    "--with",
    "pytest-asyncio==1.3.0",
    "python",
    "-m",
    "pytest",
    "--asyncio-mode=auto",
    "-p",
    "no:cacheprovider",
    "-p",
    "scripts.stage_d_v12_preflight_plugin",
    "--strict-config",
    "--strict-markers",
    "-q",
]
EXTRA_SELECTORS = (
    "tests/test_stage_d_live_observer.py::test_max_token_malformed_action_completes_before_next_child_turn",
    "tests/test_stage_d_live_observer.py::test_observer_rejects_unknown_typed_message_fields",
    "tests/test_stage_d_live_observer.py::test_observer_rejects_tool_call_with_non_tool_finish",
    "tests/test_stage_d_exact_action.py::test_malformed_max_token_action_strictly_reloads_without_semantic_roundtrip",
    "tests/test_stage_d_source_producer.py::test_ineligible_max_token_source_strictly_reloads_after_restart",
    "tests/test_stage_d_dependency_deployment.py",
    "tests/test_stage_d_historical_semantics.py",
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_bytes(repository: Path, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{RUNTIME_COMMIT}:{path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def read_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ValueError(f"{path} is not canonical JSON")
    return value


def _read_commit_canonical(repository: Path, path: str) -> dict[str, Any]:
    raw = _git_bytes(repository, path)
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise ValueError(f"{path} is not canonical JSON at the frozen commit")
    return value


def _canonical_list_sha256(values: Sequence[str]) -> str:
    return _sha256(canonical_json(list(values)))


def _verify_selected_test_bytes(repository: Path, selectors: Sequence[str]) -> None:
    for selector in selectors:
        path_text = selector.split("::", 1)[0]
        if path_text.startswith("external/prime-rl/deps/"):
            continue
        if (repository / path_text).read_bytes() != _git_bytes(repository, path_text):
            raise ValueError(f"v12 selected test differs from frozen commit: {path_text}")


def _remove_dependency_caches(root: Path) -> None:
    for name in ("__pycache__", ".pytest_cache"):
        for cache in root.rglob(name):
            if cache.is_dir() and not cache.is_symlink():
                shutil.rmtree(cache)


def verify_active_imports(
    repository: Path,
    imported_modules: Sequence[Mapping[str, Any]],
    *,
    renderers: Path,
    verifiers: Path,
) -> dict[str, str]:
    roots = [
        repository / "src",
        repository / "external/prime-rl/src",
        repository / "environments/redco_evidence_selection_v2",
        renderers,
        verifiers,
    ]
    import sys

    for root in reversed(roots):
        root_text = str(root.resolve())
        if root_text in sys.path:
            sys.path.remove(root_text)
        sys.path.insert(0, root_text)
    bindings: dict[str, str] = {}
    for item in imported_modules:
        name = item["name"]
        if name == "renderers":
            expected_path = renderers / "renderers/__init__.py"
        elif name == "verifiers.v1":
            expected_path = verifiers / "verifiers/v1/__init__.py"
        else:
            expected_path = repository / item["absolute_path"].removeprefix(
                "/workspace/redco/"
            )
        module = importlib.import_module(name)
        actual_text = getattr(module, "__file__", None)
        if not actual_text:
            raise ValueError(f"v12 imported module has no file binding: {name}")
        actual_path = Path(actual_text).resolve()
        if actual_path != expected_path.resolve() or _sha256(
            actual_path.read_bytes()
        ) != item["sha256"]:
            raise ValueError(f"v12 active imported module differs: {name}")
        bindings[name] = item["absolute_path"]
    return bindings


class PreflightPlugin:
    def __init__(self) -> None:
        self.nodeids: list[str] = []
        self.states: dict[str, str] = {}
        self.collection_errors = 0
        self.deselected = 0

    def pytest_collection_finish(self, session: Any) -> None:
        self.nodeids = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.collection_errors += 1

    def pytest_deselected(self, items: Sequence[Any]) -> None:
        self.deselected += len(items)

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.failed:
            self.states[report.nodeid] = "failed"
        elif report.skipped:
            self.states[report.nodeid] = (
                "xfail" if hasattr(report, "wasxfail") else "skipped"
            )
        elif report.when == "call" and report.passed:
            self.states.setdefault(report.nodeid, "passed")


def build_preflight_result(
    plugin: PreflightPlugin,
    preregistration: Mapping[str, Any],
    *,
    exitstatus: int,
    renderers: Path,
    verifiers: Path,
) -> dict[str, Any]:
    selectors = preregistration["preflight"]["mandatory_prime_tests"]
    normalized = [
        nodeid.replace(f"{renderers.as_posix()}/", "external/renderers/").replace(
            f"{verifiers.as_posix()}/", "external/verifiers/"
        )
        for nodeid in plugin.nodeids
    ]
    counts = {
        name: sum(state == name for state in plugin.states.values())
        for name in ("passed", "failed", "skipped", "xfail")
    }
    return {
        "collected": len(normalized),
        "collection_errors": plugin.collection_errors,
        "deselected": plugin.deselected,
        "exit_status": int(exitstatus),
        "nodeid_manifest_sha256": _canonical_list_sha256(normalized),
        "selector_count": len(selectors),
        "selector_manifest_sha256": _canonical_list_sha256(selectors),
        **counts,
    }


def _validate_result(
    result: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> None:
    expected = preregistration["preflight"]["exact_collection"]
    candidate = {
        "collected": expected["collected"],
        "collection_errors": expected["errors"],
        "deselected": 0,
        "exit_status": 0,
        "nodeid_manifest_sha256": expected["nodeid_manifest_sha256"],
        "selector_count": expected["selector_count"],
        "selector_manifest_sha256": expected["selector_manifest_sha256"],
        "passed": expected["passed"],
        "failed": expected["failed"],
        "skipped": expected["skipped"],
        "xfail": expected["xfail"],
    }
    hard = {
        **candidate,
        "collected": EXPECTED_NODEID_COUNT,
        "nodeid_manifest_sha256": EXPECTED_NODEID_SHA256,
        "selector_count": EXPECTED_SELECTOR_COUNT,
        "selector_manifest_sha256": EXPECTED_SELECTOR_SHA256,
        "passed": EXPECTED_NODEID_COUNT,
    }
    if result != candidate or result != hard:
        raise ValueError(f"v12 exact preflight differs: {result}")


def run_preflight(
    repository: Path,
    preregistration: Mapping[str, Any],
    *,
    tokenizer: Path,
    renderers: Path,
    verifiers: Path,
) -> dict[str, Any]:
    selectors = preregistration["preflight"]["mandatory_prime_tests"]
    parent = _read_commit_canonical(
        repository, "configs/stage-d/stage-d1-support-preregistration-v11-2.json"
    )
    expected_selectors = [*parent["preflight"]["mandatory_prime_tests"], *EXTRA_SELECTORS]
    if (
        not isinstance(selectors, list)
        or not all(isinstance(item, str) for item in selectors)
        or selectors != expected_selectors
        or len(selectors) != EXPECTED_SELECTOR_COUNT
        or _canonical_list_sha256(selectors) != EXPECTED_SELECTOR_SHA256
        or importlib.metadata.version("pytest-asyncio") != "1.3.0"
    ):
        raise ValueError("v12 frozen runner or selector manifest differs")
    _verify_selected_test_bytes(repository, selectors)
    plugin_path = repository / "scripts/stage_d_v12_preflight_plugin.py"
    if _sha256(plugin_path.read_bytes()) != EXPECTED_PREFLIGHT_PLUGIN_SHA256:
        raise ValueError("v12 preflight reporter differs")
    _remove_dependency_caches(renderers)
    _remove_dependency_caches(verifiers)
    with tempfile.TemporaryDirectory(prefix="redco-stage-d-v12-preflight-") as temporary:
        root = Path(temporary)
        output = root / "report.json"
        remap = {
            "external/prime-rl/deps/renderers/tests/test_client.py": str(
                renderers / "tests/test_client.py"
            ),
            "external/prime-rl/deps/verifiers/tests/test_rlm_structure_v2.py": str(
                verifiers / "tests/test_rlm_structure_v2.py"
            ),
        }
        command = [*EXPECTED_PREFLIGHT_RUNNER, *[remap.get(item, item) for item in selectors]]
        pythonpath = [
            repository / "src",
            repository / "external/prime-rl/src",
            repository / "environments/redco_evidence_selection_v2",
            renderers,
            verifiers,
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(root / "pycache"),
                "PYTHONPATH": os.pathsep.join(str(path.resolve()) for path in pythonpath),
                "REDCO_STAGE_D_RENDERERS_ROOT": str(renderers),
                "REDCO_STAGE_D_TOKENIZER_PATH": str(tokenizer),
                "REDCO_STAGE_D_V12_PREFLIGHT_OUTPUT": str(output),
                "REDCO_STAGE_D_V12_REPOSITORY": str(repository),
                "REDCO_STAGE_D_VERIFIERS_ROOT": str(verifiers),
            }
        )
        completed = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if not output.is_file():
            raise ValueError(
                "v12 fresh-process preflight produced no report: "
                f"stdout={completed.stdout[-2000:]} stderr={completed.stderr[-2000:]}"
            )
        result = read_canonical(output)
        if completed.returncode != 0:
            raise ValueError(
                f"v12 fresh-process preflight failed: result={result} "
                f"stdout={completed.stdout[-6000:]} stderr={completed.stderr[-6000:]}"
            )
        active_imports = result.pop("active_imports")
        _validate_result(result, preregistration)
        result["active_imports"] = active_imports
        return result
