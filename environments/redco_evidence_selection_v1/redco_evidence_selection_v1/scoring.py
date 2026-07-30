from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceParse:
    spans: tuple[str, ...]
    parseable: bool


def parse_evidence(text: str) -> EvidenceParse:
    """Parse a final answer without accepting executable Python."""

    try:
        value = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError):
        return EvidenceParse((), False)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) for item in value
    ):
        return EvidenceParse((), False)
    return EvidenceParse(tuple(item for item in value if item), True)


def _all_exact_intervals(text: str, spans: Iterable[str]) -> tuple[tuple[int, int], ...]:
    intervals: list[tuple[int, int]] = []
    for span in spans:
        start = 0
        while span and (index := text.find(span, start)) >= 0:
            intervals.append((index, index + len(span)))
            start = index + 1
    return tuple(intervals)


def _merge(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def _union_size(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge(intervals))


def _intersection_size(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> int:
    a = _merge(left)
    b = _merge(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        total += max(0, hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def score_exact_spans(
    paper: str, predicted: Iterable[str], reference: Iterable[str]
) -> dict[str, float]:
    predicted_tuple = tuple(predicted)
    reference_tuple = tuple(reference)
    predicted_intervals = _all_exact_intervals(paper, predicted_tuple)
    reference_intervals = _all_exact_intervals(paper, reference_tuple)
    covered = _intersection_size(predicted_intervals, reference_intervals)
    retrieved = _union_size(predicted_intervals)
    relevant = _union_size(reference_intervals)
    precision = covered / retrieved if retrieved else 0.0
    recall = covered / relevant if relevant else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    exact = sum(span in paper for span in predicted_tuple)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_substring_fraction": (
            exact / len(predicted_tuple) if predicted_tuple else 0.0
        ),
    }
