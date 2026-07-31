from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceParse:
    spans: tuple[str, ...]
    parseable: bool


def parse_evidence(text: str) -> EvidenceParse:
    """Parse only a literal Python list of strings, never executable Python."""

    try:
        value = ast.literal_eval(text.strip())
    except (SyntaxError, ValueError):
        return EvidenceParse((), False)
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        return EvidenceParse((), False)
    return EvidenceParse(tuple(value), True)


def _all_exact_intervals(
    text: str, spans: Iterable[str]
) -> tuple[tuple[int, int], ...]:
    intervals: list[tuple[int, int]] = []
    for span in spans:
        start = 0
        while span and (index := text.find(span, start)) >= 0:
            intervals.append((index, index + len(span)))
            start = index + 1
    return tuple(intervals)


def _merge(
    intervals: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
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
    exact_count = sum(bool(span) and span in paper for span in predicted_tuple)
    valid_prediction = bool(predicted_tuple) and exact_count == len(
        predicted_tuple
    )
    valid_reference = bool(reference_tuple) and all(
        span and span in paper for span in reference_tuple
    )
    common = {
        "exact_substring_fraction": (
            exact_count / len(predicted_tuple) if predicted_tuple else 0.0
        ),
        "all_predicted_spans_verbatim": float(valid_prediction),
        "valid_reference": float(valid_reference),
        "predicted_characters": float(sum(map(len, predicted_tuple))),
        "predicted_span_count": float(len(predicted_tuple)),
    }
    if not valid_prediction or not valid_reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, **common}

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
    return {"precision": precision, "recall": recall, "f1": f1, **common}


def score_evidence_reply(
    paper: str, reply: str, reference: Iterable[str]
) -> dict[str, float]:
    parsed = parse_evidence(reply)
    score = score_exact_spans(paper, parsed.spans, reference)
    score["parseable"] = float(parsed.parseable)
    if not parsed.parseable:
        score["precision"] = 0.0
        score["recall"] = 0.0
        score["f1"] = 0.0
        score["all_predicted_spans_verbatim"] = 0.0
    return score
