from __future__ import annotations

from pathlib import Path

import pytest

from redco.analysis.stage_d_three_arm_prime import (
    materialize_prime_objective_binding,
)


def test_pinned_prime_materializes_actual_branch_objective(tmp_path: Path) -> None:
    pytest.importorskip("prime_rl")
    source = Path("configs/stage-d/stage-d-e2-trainer-v1.toml").read_bytes()
    config = tmp_path / "trainer.toml"
    config.write_bytes(source)
    binding, parsed, loss_fn = materialize_prime_objective_binding(
        arm="local",
        evidence_class="fixture-only",
        effective_argv=("@", str(config)),
        trainer_toml_path=config,
        trainer_toml_bytes=config.read_bytes(),
    )
    assert binding.arm == "local"
    assert parsed.loss.import_path == (
        "prime_rl.trainer.rl.redco_loss.clean_decision_loss"
    )
    assert callable(loss_fn)


def test_pinned_prime_rejects_cli_override_shape(tmp_path: Path) -> None:
    pytest.importorskip("prime_rl")
    config = tmp_path / "trainer.toml"
    config.write_bytes(b"max_steps = 1\n")
    with pytest.raises(ValueError, match="one exact TOML"):
        materialize_prime_objective_binding(
            arm="local",
            evidence_class="fixture-only",
            effective_argv=("@", str(config), "--max_steps", "2"),
            trainer_toml_path=config,
            trainer_toml_bytes=config.read_bytes(),
        )
