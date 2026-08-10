from __future__ import annotations

import ast
import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_PROMPT_RELATIVE = Path("configs/stage-d/stage-d0-scaffold-fewshot-v4.txt")
_PROMPT_SHA256 = "b27653e90f52a20f26ac79e3d0569275e9ba0ed2b07abbe06f060dd2486aee73"


def _frozen_scaffold_source() -> str:
    raw = (Path(__file__).parents[1] / _PROMPT_RELATIVE).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _PROMPT_SHA256

    opening = b"```python\n"
    closing = b"\n```\n"
    assert raw.count(opening) == 1
    assert raw.count(closing) == 1
    start = raw.index(opening) + len(opening)
    stop = raw.index(closing, start)
    return raw[start:stop].decode("utf-8")


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

    code = _frozen_scaffold_source()
    workspace_path = "/workspace/evidence_context.txt"
    assert code.count(workspace_path) == 1
    code = code.replace(workspace_path, paper.as_posix(), 1)
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
