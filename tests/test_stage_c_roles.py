import pytest

from redco.integrations.stage_c_roles import stage_c_branch_roles


def test_stage_c_branch_roles_supports_power_analysis_design() -> None:
    roles = stage_c_branch_roles(11)

    assert roles[0] == "original"
    assert roles[-1] == "alternative_10"
    assert len(roles) == 11
    assert len(set(roles)) == 11


@pytest.mark.parametrize("branch_group_size", [1, 12])
def test_stage_c_branch_roles_rejects_unsupported_size(
    branch_group_size: int,
) -> None:
    with pytest.raises(ValueError):
        stage_c_branch_roles(branch_group_size)
