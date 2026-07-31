from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from redco_evidence_selection_v2.taskset import EvidenceSelectionTaskset


def __getattr__(name: str) -> Any:
    if name != "EvidenceSelectionTaskset":
        raise AttributeError(name)
    from redco_evidence_selection_v2.taskset import EvidenceSelectionTaskset

    return EvidenceSelectionTaskset


__all__ = ["EvidenceSelectionTaskset"]
