from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_DRIVER = REPO_ROOT / "scripts" / "run_stage_c5_constrained_selection_v1.sh"


def _wsl_path(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    suffix = path.as_posix().split(":/", maxsplit=1)[1]
    return f"/mnt/{drive}/{suffix}"


def test_stage_c5_candidate_commands_expand_signed_case_variables() -> None:
    environment = os.environ.copy()
    environment["REDCO_STAGE_C5_CAMPAIGN_VERSION"] = "runtime-regression"
    environment["REDCO_STAGE_C5_RUNTIME_REGRESSION"] = "1"
    if os.name == "nt":
        command = [
            "wsl",
            "env",
            "REDCO_STAGE_C5_CAMPAIGN_VERSION=runtime-regression",
            "REDCO_STAGE_C5_RUNTIME_REGRESSION=1",
            "bash",
            _wsl_path(SHARED_DRIVER),
        ]
    else:
        command = ["bash", str(SHARED_DRIVER)]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "stage-c5-runtime-regression=passed" in result.stdout
    assert (
        "action-signature=runtime-regression-action-signature"
        in result.stdout
    )
    assert (
        "root-signature=runtime-regression-root-signature"
        in result.stdout
    )


def test_stage_c5_driver_uses_constraint_supervisor_mode() -> None:
    source = SHARED_DRIVER.read_text(encoding="utf-8")

    assert "--mode constraint" in source
    assert "--mode smoke" not in source
