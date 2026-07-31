from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_stage_d_v4_9_inner_runner import NEW, OLD, generate  # noqa: E402


def _parent_text() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            OLD,
            *("# REQUESTS_STARTED" for _ in range(6)),
            'source "$instrumented_tail"',
            "",
        ]
    )


def test_exact_one_hunk_replacement(tmp_path: Path) -> None:
    parent = tmp_path / "parent.sh"
    output = tmp_path / "generated.sh"
    parent.write_text(_parent_text(), encoding="utf-8", newline="\n")

    report = generate(parent, output)

    assert report["passes"] is True
    assert OLD not in output.read_text(encoding="utf-8")
    assert output.read_text(encoding="utf-8").count(NEW) == 1
    assert output.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("count", [0, 2])
def test_rejects_missing_or_duplicate_parent_line(
    tmp_path: Path, count: int
) -> None:
    parent = tmp_path / "parent.sh"
    parent.write_text("\n".join([OLD] * count), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly once"):
        generate(parent, tmp_path / "generated.sh")
