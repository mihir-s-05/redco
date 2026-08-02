from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redco_evidence_selection_v2.source_env import StageDSourceEnv
    from redco_evidence_selection_v2.taskset import EvidenceSelectionTaskset


def __getattr__(name: str) -> Any:
    if name == "EvidenceSelectionTaskset":
        from redco_evidence_selection_v2.taskset import EvidenceSelectionTaskset

        return EvidenceSelectionTaskset
    if name == "StageDSourceEnv":
        from redco_evidence_selection_v2.source_env import StageDSourceEnv

        return StageDSourceEnv
    raise AttributeError(name)


__all__ = ["EvidenceSelectionTaskset", "StageDSourceEnv"]
