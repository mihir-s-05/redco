from __future__ import annotations

from redco.analysis.stage_d_child_consumers import classify_child_consumption


def _classify(
    child: str,
    parent: str,
    other: str,
    duplicates: tuple[int, ...] = (1,),
) -> str:
    return str(
        classify_child_consumption(
            child_text=child,
            child_call_index=duplicates[0],
            duplicate_call_indices=duplicates,
            parent_tool_content=parent,
            other_tool_contents=(other,),
        )["classification"]
    )


def test_classifies_exact_escaped_hidden_and_duplicate_surfaces() -> None:
    assert _classify("alpha", "['alpha']", "") == "exact_surface"
    assert _classify("a\nb", "['a\\nb']", "") == "escaped_surface"
    assert _classify("hidden", "[]", "") == "no_serialized_surface_observed"
    assert (
        _classify("same", "[]", "['same']", (1, 4))
        == "duplicate_alias_elsewhere"
    )
    assert (
        _classify("same", "['same']", "", (1, 4))
        == "exact_surface_ambiguous_duplicate"
    )


def test_diagnostic_does_not_confuse_exact_text_with_escaped_form() -> None:
    report = classify_child_consumption(
        child_text="single line",
        child_call_index=2,
        duplicate_call_indices=(2,),
        parent_tool_content="single line",
        other_tool_contents=(),
    )
    assert report["parent_tool_exact_count"] == 1
    assert report["parent_tool_escaped_repr_count"] == 0
