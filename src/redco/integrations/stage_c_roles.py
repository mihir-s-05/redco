"""Role naming for finite Stage-C counterfactual branch groups."""

from __future__ import annotations

MAX_STAGE_C_BRANCHES = 11


def stage_c_branch_roles(branch_group_size: int) -> tuple[str, ...]:
    """Return the original role followed by numbered alternative roles."""
    if not 2 <= branch_group_size <= MAX_STAGE_C_BRANCHES:
        raise ValueError(
            f"branch_group_size must be between 2 and {MAX_STAGE_C_BRANCHES}"
        )
    return (
        "original",
        *(f"alternative_{index}" for index in range(1, branch_group_size)),
    )
