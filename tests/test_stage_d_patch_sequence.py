from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
    )


def _clone_patch_base(
    tmp_path: Path,
    *,
    repo: Path,
    commit: str,
) -> Path:
    target = tmp_path / repo.name
    subprocess.run(
        ["git", "clone", "--no-hardlinks", "--no-checkout", str(repo), str(target)],
        check=True,
        capture_output=True,
    )
    _git(target, "checkout", "--detach", commit)
    return target


def _apply_stack(
    tmp_path: Path,
    *,
    repo: Path,
    commit: str,
    patch_names: tuple[str, ...],
) -> Path:
    if not (repo / ".git").exists():
        pytest.skip(f"pinned dependency checkout is absent: {repo}")
    patches = tuple(ROOT / "patches" / name for name in patch_names)
    target = _clone_patch_base(
        tmp_path,
        repo=repo,
        commit=commit,
    )
    for patch in patches:
        subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=target,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "apply", str(patch)],
            cwd=target,
            check=True,
            capture_output=True,
        )
    return target


def test_prime_runtime_patch_stack_applies_in_deployment_order(tmp_path: Path) -> None:
    target = _apply_stack(
        tmp_path,
        repo=ROOT / "external" / "prime-rl",
        commit="3b22dd951cad1036d1fe8dd0a0bfc40807a9b360",
        patch_names=(
            "prime-rl-redco-stage-c9-practical-efficiency.patch",
            "prime-rl-stage-d-live-update-gate-v1.patch",
            "prime-rl-stage-d-objective-gate-v1.patch",
        ),
    )
    train = (target / "src/prime_rl/trainer/rl/train.py").read_text()
    assert "def train(config: TrainerConfig, *, redco_runtime_gate=None):" in train
    assert "redco_runtime_gate.verify_distributed()" in train
    assert "redco_runtime_gate.verify_consumed_micro_batches(" in train
    assert "redco_runtime_gate.before_optimizer_step(" in train
    assert "redco_runtime_gate.after_optimizer_step(" in train
    assert 'parallel_dims.get_mesh("dp").get_group()' in train


def test_renderer_and_verifier_patch_stacks_apply_in_deployment_order(
    tmp_path: Path,
) -> None:
    renderer = _apply_stack(
        tmp_path,
        repo=ROOT / "external" / "prime-rl" / "deps" / "renderers",
        commit="bdb96b0c84a307e2b71c6a366c9d718c3ac7fe78",
        patch_names=("renderers-stage-d-prepared-observer-v1.patch",),
    )
    assert "PreparedGenerateObserver" in (
        renderer / "renderers/client.py"
    ).read_text()

    verifier = _apply_stack(
        tmp_path,
        repo=ROOT / "external" / "prime-rl" / "deps" / "verifiers",
        commit="b13ba60da63cea91389e7575766b7270d0d11fc5",
        patch_names=(
            "verifiers-stage-d-provenance-baseline-v1.patch",
            "verifiers-stage-d-prepared-observer-v1.patch",
        ),
    )
    train = (verifier / "verifiers/v1/clients/train.py").read_text()
    assert "prompt_token_ids: Sequence[int]" in train
    assert "add_generation_prompt=True" in train
    pytest.importorskip("anthropic", reason="full Verifiers runtime is validated in pinned WSL")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(renderer), str(verifier), str(ROOT))
    )
    subprocess.run(
        [sys.executable, "-c", "import verifiers.v1; import renderers.client"],
        check=True,
        capture_output=True,
        env=environment,
    )


def test_rlm_provenance_patch_stack_applies_and_runs_in_deployment_order(
    tmp_path: Path,
) -> None:
    default_repo = (
        Path("/home/mihir/work/redco-rlm-stage-d-clean")
        if os.name != "nt"
        else Path("Z:/__missing_rlm_checkout__")
    )
    repo = Path(os.environ.get("REDCO_RLM_REPO", str(default_repo)))
    target = _apply_stack(
        tmp_path,
        repo=repo,
        commit="56218f33796ecbe465445bc43948886354fde196",
        patch_names=(
            "rlm-event-replay-provenance.patch",
            "rlm-mcp-client-symbol-compat.patch",
            "rlm-root-initial-required-tool-choice.patch",
            "rlm-spawn-provenance-v2.patch",
        ),
    )
    assert (target / "src/rlm/provenance.py").is_file()
    assert (target / "tests/test_redco_spawn_provenance_v2.py").is_file()
    default_python = Path("/home/mihir/.venvs/redco-rlm-test/bin/python")
    python = Path(os.environ.get("REDCO_RLM_TEST_PYTHON", str(default_python)))
    if not python.is_file():
        pytest.skip("persistent uv RLM test environment is absent")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target / "src")
    subprocess.run(
        [str(python), "-m", "pytest", "-q", "tests/test_redco_spawn_provenance_v2.py"],
        cwd=target,
        check=True,
        capture_output=True,
        env=environment,
    )
