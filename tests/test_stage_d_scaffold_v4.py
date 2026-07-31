from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.build_stage_d_scaffold_sft_v2 import scaffold_code


async def _execute(
    tmp_path: Path,
    values: tuple[Any, Any],
) -> tuple[list[str], list[str]]:
    paper = tmp_path / "evidence_context.txt"
    paper.write_text("first half\nsecond half", encoding="utf-8")
    invocation_ids: list[str] = []

    async def rlm(_: str, *, redco_invocation_id: str) -> Any:
        invocation_ids.append(redco_invocation_id)
        return values[len(invocation_ids) - 1]

    code = scaffold_code("What happened?").replace(
        "'/workspace/evidence_context.txt'",
        repr(str(paper)),
    )
    namespace = {"rlm": rlm}
    compiled = compile(
        code,
        "<stage-d-scaffold-v4>",
        "exec",
        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
    )
    await eval(compiled, namespace)
    return namespace["child_answers"], invocation_ids


def test_scaffold_normalizes_real_shaped_results_once(tmp_path: Path) -> None:
    answers, invocation_ids = asyncio.run(
        _execute(
            tmp_path,
            (
                SimpleNamespace(answer="left"),
                SimpleNamespace(answer="right"),
            ),
        )
    )

    assert answers == ["left", "right"]
    assert invocation_ids == ["midpoint-shard-0", "midpoint-shard-1"]


def test_scaffold_accepts_defensive_string_results(tmp_path: Path) -> None:
    answers, _ = asyncio.run(_execute(tmp_path, ("left", "right")))
    assert answers == ["left", "right"]


def test_scaffold_rejects_non_string_answer(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be a string"):
        asyncio.run(
            _execute(
                tmp_path,
                (SimpleNamespace(answer=object()), "right"),
            )
        )
