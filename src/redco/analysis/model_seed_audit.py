"""Audit same-prompt/same-seed sampling reproducibility on an OAI endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from redco.contracts import canonical_json


def request_completion(
    base_url: str,
    *,
    model: str,
    prompt: str,
    seed: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_completion_tokens": 48,
        "seed": seed,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=canonical_json(payload),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result: dict[str, Any] = json.load(response)
    content = result["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise TypeError("completion content must be a string")
    return content


def run_audit(
    base_url: str,
    *,
    model: str,
    prompt: str,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("repeats must be at least two")
    same_seed = [
        request_completion(base_url, model=model, prompt=prompt, seed=seed)
        for _ in range(repeats)
    ]
    alternate = [
        request_completion(base_url, model=model, prompt=prompt, seed=seed + offset)
        for offset in range(1, repeats + 1)
    ]
    same_hashes = [_hash_text(text) for text in same_seed]
    alternate_hashes = [_hash_text(text) for text in alternate]
    exact_reproduction = len(set(same_hashes)) == 1
    different_seeds_show_variation = len(set([*same_hashes, *alternate_hashes])) > 1
    return {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "prompt_sha256": _hash_text(prompt),
        "seed": seed,
        "repeats": repeats,
        "same_seed_response_sha256": same_hashes,
        "alternate_seed_response_sha256": alternate_hashes,
        "same_prompt_same_seed_exact": exact_reproduction,
        "different_seeds_show_variation": different_seeds_show_variation,
        "passed": exact_reproduction and different_seeds_show_variation,
    }


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="Return one invented six-letter word, and nothing else.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_audit(
        args.base_url,
        model=args.model,
        prompt=args.prompt,
        seed=args.seed,
        repeats=args.repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
