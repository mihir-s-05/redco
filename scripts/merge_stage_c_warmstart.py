"""Merge the selected Stage-C SFT adapter into a temporary local base model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    tuned = PeftModel.from_pretrained(base, args.adapter)
    merged = tuned.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.output)
    files = {
        str(path.relative_to(args.output).as_posix()): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(args.output.rglob("*"))
        if path.is_file()
    }
    payload = {
        "schema_version": 1,
        "operation": "merge-stage-c-warmstart",
        "base_model": args.model,
        "adapter": str(args.adapter.as_posix()),
        "adapter_model_sha256": _sha256(
            args.adapter / "adapter_model.safetensors"
        ),
        "output": str(args.output.as_posix()),
        "files": files,
        "local_retention": (
            "The merged 4B base is ephemeral and reproducible; retain only the "
            "selected LoRA adapter and this manifest locally."
        ),
    }
    signed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
