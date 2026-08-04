"""Machine reporter for the frozen Stage D v12 pytest preflight."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from redco.contracts import canonical_json
from scripts.stage_d_v12_preflight import (
    PreflightPlugin,
    build_preflight_result,
    read_canonical,
    verify_active_imports,
)

_STATE = PreflightPlugin()


def pytest_collection_finish(session: pytest.Session) -> None:
    _STATE.pytest_collection_finish(session)


def pytest_collectreport(report: pytest.CollectReport) -> None:
    _STATE.pytest_collectreport(report)


def pytest_deselected(items: list[pytest.Item]) -> None:
    _STATE.pytest_deselected(items)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _STATE.pytest_runtest_logreport(report)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    repository = Path(os.environ["REDCO_STAGE_D_V12_REPOSITORY"]).resolve()
    preregistration = read_canonical(
        repository / "configs/stage-d/stage-d1-support-preregistration-v12.json"
    )
    renderers = Path(os.environ["REDCO_STAGE_D_RENDERERS_ROOT"]).resolve()
    verifiers = Path(os.environ["REDCO_STAGE_D_VERIFIERS_ROOT"]).resolve()
    result = build_preflight_result(
        _STATE,
        preregistration,
        exitstatus=exitstatus,
        renderers=renderers,
        verifiers=verifiers,
    )
    stack = read_canonical(
        repository / "configs/stage-d/stage-d1-dependency-stack-v12.json"
    )
    result["active_imports"] = verify_active_imports(
        repository,
        stack["imported_modules"],
        renderers=renderers,
        verifiers=verifiers,
    )
    output = Path(os.environ["REDCO_STAGE_D_V12_PREFLIGHT_OUTPUT"])
    output.write_bytes(canonical_json(result))
