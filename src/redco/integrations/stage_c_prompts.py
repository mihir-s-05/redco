"""Shared prompt contracts for the live Stage-C credit environment."""

from __future__ import annotations

from collections.abc import Sequence

ROUTES = ("alpha", "beta", "gamma", "delta")


def stage_c_root_prompt() -> str:
    """Return the exact root-decision prompt used by training and SFT."""
    return (
        "Choose a routing label for a later sub-call. The route adds a "
        "deterministic background contribution to the final reward, "
        "while the sub-call action supplies the targeted contribution. "
        "Your final reply must end with exactly "
        "<route>VALUE</route>, replacing VALUE with one of: "
        f"{', '.join(ROUTES)}."
    )


def stage_c_branch_prompt(
    context_reply: str | None,
    actions: Sequence[str],
) -> str:
    """Return the exact target-decision prompt used by training and SFT."""
    return (
        "You are the trainable depth-one sub-call in a deterministic "
        "credit-assignment probe. The already-sampled root context is quoted "
        f"below; do not modify it:\n<context>{context_reply}</context>\n"
        "Choose one allowed action. Reply with exactly one digit and no other "
        f"text. Allowed digits: {', '.join(actions)}. The decoder samples "
        "one token from the model's complete vocabulary; any other token is "
        "retained as an invalid action and receives the failure reward."
    )
