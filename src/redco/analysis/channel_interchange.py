"""Deterministic kill test for declared-versus-ambient channel interchange."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json


class ProbeKind(StrEnum):
    ARTIFACT_ONLY = "artifact_only"
    CONTEXT_ONLY = "context_only"
    REDUNDANT = "redundant"
    SYNERGISTIC = "synergistic"


class Representation(StrEnum):
    CANONICAL = "canonical"
    EQUIVALENT = "equivalent"


@dataclass(frozen=True, slots=True)
class InterchangeEffects:
    """Four-cell factorial effects for one declared replay protocol."""

    r00: float
    r10: float
    r01: float
    r11: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.cells):
            raise ValueError("interchange rewards must be finite")

    @property
    def cells(self) -> tuple[float, float, float, float]:
        return self.r00, self.r10, self.r01, self.r11

    @property
    def total(self) -> float:
        return self.r11 - self.r00

    @property
    def artifact_baseline_effect(self) -> float:
        return self.r10 - self.r00

    @property
    def context_baseline_effect(self) -> float:
        return self.r01 - self.r00

    @property
    def interaction(self) -> float:
        return self.r11 - self.r10 - self.r01 + self.r00

    @property
    def artifact_allocation(self) -> float:
        return 0.5 * ((self.r10 - self.r00) + (self.r11 - self.r01))

    @property
    def context_allocation(self) -> float:
        return 0.5 * ((self.r01 - self.r00) + (self.r11 - self.r10))

    @property
    def conservation_error(self) -> float:
        return self.artifact_allocation + self.context_allocation - self.total

    def as_dict(self) -> dict[str, float]:
        return {
            "artifact_allocation": self.artifact_allocation,
            "artifact_baseline_effect": self.artifact_baseline_effect,
            "conservation_error": self.conservation_error,
            "context_allocation": self.context_allocation,
            "context_baseline_effect": self.context_baseline_effect,
            "interaction": self.interaction,
            "r00": self.r00,
            "r01": self.r01,
            "r10": self.r10,
            "r11": self.r11,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class ChannelExecution:
    artifact_read: bool
    artifact_sha256: str
    artifact_signal: bool
    ambient_included: bool
    ambient_sha256: str
    ambient_signal: bool
    prompt_geometry: dict[str, object]
    reward: float

    def as_dict(self) -> dict[str, object]:
        return {
            "ambient_included": self.ambient_included,
            "ambient_sha256": self.ambient_sha256,
            "ambient_signal": self.ambient_signal,
            "artifact_read": self.artifact_read,
            "artifact_sha256": self.artifact_sha256,
            "artifact_signal": self.artifact_signal,
            "prompt_geometry": self.prompt_geometry,
            "reward": self.reward,
        }


_AMBIENT_SIGNAL = re.compile(r"^\s*ambient_signal\s*=\s*(true|false)\s*$")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_bytes(signal: bool, representation: Representation) -> bytes:
    value = {"schema_version": 1, "signal": signal}
    if representation is Representation.CANONICAL:
        return canonical_json(value)
    if representation is Representation.EQUIVALENT:
        return json.dumps(
            {"signal": signal, "schema_version": 1},
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
    raise ValueError(f"unsupported representation: {representation}")


def _ambient_text(signal: bool, representation: Representation) -> str:
    word = "true" if signal else "false"
    if representation is Representation.CANONICAL:
        return f"ambient_signal={word}"
    if representation is Representation.EQUIVALENT:
        return f"  ambient_signal = {word}\n"
    raise ValueError(f"unsupported representation: {representation}")


def _read_declared_artifact(raw: bytes) -> bool:
    try:
        value: Any = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("declared artifact is not UTF-8 JSON") from error
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "signal"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["signal"]) is not bool
    ):
        raise ValueError("declared artifact has the wrong schema")
    return value["signal"]


def _observe_ambient_context(text: str) -> bool:
    match = _AMBIENT_SIGNAL.fullmatch(text)
    if match is None:
        raise ValueError("ambient context has the wrong contract")
    return match.group(1) == "true"


def _reward(kind: ProbeKind, *, artifact_signal: bool, ambient_signal: bool) -> float:
    if kind is ProbeKind.ARTIFACT_ONLY:
        success = artifact_signal
    elif kind is ProbeKind.CONTEXT_ONLY:
        success = ambient_signal
    elif kind is ProbeKind.REDUNDANT:
        success = artifact_signal or ambient_signal
    elif kind is ProbeKind.SYNERGISTIC:
        success = artifact_signal and ambient_signal
    else:
        raise ValueError(f"unsupported probe: {kind}")
    return float(success)


def execute_cell(
    kind: ProbeKind,
    *,
    artifact_original: bool,
    ambient_original: bool,
    representation: Representation,
) -> ChannelExecution:
    """Execute one mixed state with an explicit artifact read and ambient view."""
    artifact_raw = _artifact_bytes(artifact_original, representation)
    ambient_text = _ambient_text(ambient_original, representation)
    reference_artifact = _artifact_bytes(True, representation)
    reference_ambient = _ambient_text(True, representation).encode("utf-8")
    rendered_bytes = len(artifact_raw) + len(ambient_text.encode("utf-8"))
    reference_bytes = len(reference_artifact) + len(reference_ambient)
    artifact_signal = _read_declared_artifact(artifact_raw)
    ambient_signal = _observe_ambient_context(ambient_text)
    return ChannelExecution(
        artifact_read=True,
        artifact_sha256=_sha256(artifact_raw),
        artifact_signal=artifact_signal,
        ambient_included=True,
        ambient_sha256=_sha256(ambient_text.encode("utf-8")),
        ambient_signal=ambient_signal,
        prompt_geometry={
            "contract_equivalent": True,
            "length_matched": rendered_bytes == reference_bytes,
            "newly_truncated_span_ids": [],
            "original_reference_utf8_bytes": reference_bytes,
            "rendered_utf8_bytes": rendered_bytes,
            "tokenizer_used": False,
            "unrelated_tokens_displaced": None,
            "utf8_byte_delta": rendered_bytes - reference_bytes,
        },
        reward=_reward(
            kind,
            artifact_signal=artifact_signal,
            ambient_signal=ambient_signal,
        ),
    )


def evaluate_probe(
    kind: ProbeKind,
    representation: Representation,
) -> tuple[InterchangeEffects, dict[str, ChannelExecution]]:
    cells = {
        "r00": execute_cell(
            kind,
            artifact_original=False,
            ambient_original=False,
            representation=representation,
        ),
        "r10": execute_cell(
            kind,
            artifact_original=True,
            ambient_original=False,
            representation=representation,
        ),
        "r01": execute_cell(
            kind,
            artifact_original=False,
            ambient_original=True,
            representation=representation,
        ),
        "r11": execute_cell(
            kind,
            artifact_original=True,
            ambient_original=True,
            representation=representation,
        ),
    }
    effects = InterchangeEffects(
        r00=cells["r00"].reward,
        r10=cells["r10"].reward,
        r01=cells["r01"].reward,
        r11=cells["r11"].reward,
    )
    return effects, cells


_EXPECTED = {
    ProbeKind.ARTIFACT_ONLY: InterchangeEffects(0.0, 1.0, 0.0, 1.0),
    ProbeKind.CONTEXT_ONLY: InterchangeEffects(0.0, 0.0, 1.0, 1.0),
    ProbeKind.REDUNDANT: InterchangeEffects(0.0, 1.0, 1.0, 1.0),
    ProbeKind.SYNERGISTIC: InterchangeEffects(0.0, 0.0, 0.0, 1.0),
}


def build_report() -> dict[str, object]:
    records: list[dict[str, object]] = []
    all_planted_effects_recovered = True
    all_representations_stable = True
    all_channels_observed = True
    all_conservation_exact = True

    for kind in ProbeKind:
        by_representation: dict[Representation, InterchangeEffects] = {}
        representation_records: list[dict[str, object]] = []
        for representation in Representation:
            effects, cells = evaluate_probe(kind, representation)
            by_representation[representation] = effects
            representation_records.append(
                {
                    "cells": {
                        name: execution.as_dict() for name, execution in sorted(cells.items())
                    },
                    "effects": effects.as_dict(),
                    "representation": representation.value,
                }
            )
            all_channels_observed &= all(
                execution.artifact_read and execution.ambient_included
                for execution in cells.values()
            )
            all_conservation_exact &= effects.conservation_error == 0.0

        canonical = by_representation[Representation.CANONICAL]
        equivalent = by_representation[Representation.EQUIVALENT]
        recovered = canonical == _EXPECTED[kind]
        stable = canonical == equivalent
        all_planted_effects_recovered &= recovered
        all_representations_stable &= stable
        records.append(
            {
                "planted_effect_recovered": recovered,
                "probe": kind.value,
                "representations": representation_records,
                "semantics_preserving_representation_stable": stable,
            }
        )

    gates = {
        "all_channels_observed": all_channels_observed,
        "conservation_exact": all_conservation_exact,
        "planted_effects_recovered": all_planted_effects_recovered,
        "semantics_preserving_representation_stable": all_representations_stable,
    }
    payload: dict[str, object] = {
        "channel_contract": {
            "ambient": "automatically visible observation under an explicit text contract",
            "declared": (
                "schema-validated provenance-bearing value visible only after an explicit read"
            ),
        },
        "gates": gates,
        "interpretation": (
            "The deterministic typed-interchange measurement contract is coherent on "
            "planted cases. This is a systems/formula gate, not evidence that an LLM "
            "learns robust routing or that mixed states are valid in a real workflow."
        ),
        "passed": all(gates.values()),
        "probes": records,
        "training_performed": False,
    }
    return {
        "payload": payload,
        "payload_sha256": _sha256(canonical_json(payload)),
        "schema_version": 1,
    }


def compact_result(
    report: dict[str, object],
    *,
    source_path: str,
    source_raw_sha256: str,
) -> dict[str, object]:
    payload = report["payload"]
    if type(payload) is not dict or type(payload.get("probes")) is not list:
        raise ValueError("full report has the wrong schema")
    probes: list[dict[str, object]] = []
    for record in payload["probes"]:
        if type(record) is not dict or type(record.get("representations")) is not list:
            raise ValueError("full report probe has the wrong schema")
        canonical = next(
            (
                item
                for item in record["representations"]
                if type(item) is dict and item.get("representation") == "canonical"
            ),
            None,
        )
        if canonical is None:
            raise ValueError("full report lacks canonical probe effects")
        probes.append(
            {
                "effects": canonical["effects"],
                "probe": record["probe"],
                "representation_stable": record["semantics_preserving_representation_stable"],
            }
        )
    compact_payload: dict[str, object] = {
        "gates": payload["gates"],
        "interpretation": payload["interpretation"],
        "passed": payload["passed"],
        "probes": probes,
        "source_report_path": source_path,
        "source_report_raw_sha256": source_raw_sha256,
        "training_performed": False,
    }
    return {
        "payload": compact_payload,
        "payload_sha256": _sha256(canonical_json(compact_payload)),
        "schema_version": 1,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.check and (args.output is not None or args.compact_output is not None):
        raise ValueError("--check cannot write outputs")
    if args.compact_output is not None and args.output is None:
        raise ValueError("--compact-output requires --output")
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
        if args.compact_output is not None:
            compact = compact_result(
                report,
                source_path=args.output.as_posix(),
                source_raw_sha256=_sha256(rendered.encode("utf-8")),
            )
            compact_rendered = (
                json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            )
            args.compact_output.parent.mkdir(parents=True, exist_ok=True)
            with args.compact_output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(compact_rendered)
    else:
        print(rendered, end="")
    payload = report["payload"]
    if type(payload) is not dict:
        raise AssertionError("internal report payload is not an object")
    return 0 if payload["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
