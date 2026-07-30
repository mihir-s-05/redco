"""Freeze a step-correct, content-identical Stage-C8 batch reuse contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import msgspec
import torch
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.utils import build_bin_cost
from prime_rl.transport import MicroBatch, TrainingBatch


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_contract(micro_batch: MicroBatch) -> dict[str, Any]:
    tensors = {
        "input_ids": torch.tensor(micro_batch.input_ids, dtype=torch.long).unsqueeze(0),
        "position_ids": torch.tensor(
            micro_batch.position_ids,
            dtype=torch.long,
        ).unsqueeze(0),
        "advantages": torch.tensor(
            micro_batch.advantages,
            dtype=torch.float,
        ).unsqueeze(0),
        "inference_logprobs": torch.tensor(
            micro_batch.inference_logprobs,
            dtype=torch.float,
        ).unsqueeze(0),
        "loss_mask": torch.tensor(
            micro_batch.loss_mask,
            dtype=torch.bool,
        ).unsqueeze(0),
        "temperatures": torch.tensor(
            micro_batch.temperatures,
            dtype=torch.float,
        ).unsqueeze(0),
        "lora_num_tokens": torch.tensor(
            micro_batch.lora_num_tokens,
            dtype=torch.int32,
        ),
        "seq_lens": torch.tensor(micro_batch.seq_lens, dtype=torch.long),
    }
    return {
        name: {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
        }
        for name, tensor in tensors.items()
    }


def _validate_examples(batch: TrainingBatch) -> None:
    for sample in batch.examples:
        length = len(sample.token_ids)
        required = (
            sample.mask,
            sample.logprobs,
            sample.temperatures,
            sample.advantages,
        )
        if any(stream is None or len(stream) != length for stream in required):
            raise ValueError("required training streams are not token-aligned")
        for stream in (
            sample.rl_weights,
            sample.ce_weights,
            sample.ref_kl_weights,
            sample.ref_logprobs,
            sample.mm_token_type_ids,
        ):
            if stream is not None and len(stream) != length:
                raise ValueError("optional training stream is not token-aligned")


def build_contract(
    source: Path,
    step_1_output: Path,
    step_2_output: Path,
    audit_output: Path,
) -> dict[str, Any]:
    source_payload = source.read_bytes()
    batch = msgspec.msgpack.decode(source_payload, type=TrainingBatch)
    if batch.step != 1:
        raise ValueError(f"expected source batch step 1, found {batch.step}")
    _validate_examples(batch)

    examples_payload = msgspec.msgpack.encode(batch.examples)
    examples_sha256 = _sha256(examples_payload)
    encoded_by_step: dict[int, bytes] = {}
    for step, output in ((1, step_1_output), (2, step_2_output)):
        payload = msgspec.msgpack.encode(
            TrainingBatch(examples=batch.examples, step=step),
        )
        decoded = msgspec.msgpack.decode(payload, type=TrainingBatch)
        if decoded.step != step:
            raise ValueError("re-encoded transport step does not match path")
        if _sha256(msgspec.msgpack.encode(decoded.examples)) != examples_sha256:
            raise ValueError("example stream changed while normalizing step")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
        encoded_by_step[step] = payload

    packed_by_worker = prepare_batch(
        rollouts=batch.examples,
        seq_len=512,
        num_train_workers=1,
        idxs=[0] * len(batch.examples),
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )
    packed = [micro for worker in packed_by_worker for micro in worker]
    if not packed:
        raise ValueError("frozen examples produced no packed microbatches")

    packed_rows = []
    for index, micro_batch in enumerate(packed):
        token_count = len(micro_batch.input_ids)
        if micro_batch.lora_num_tokens != [token_count]:
            raise ValueError("single-adapter token count does not cover batch")
        packed_rows.append(
            {
                "index": index,
                "token_count": token_count,
                "sequence_count": len(micro_batch.sequence_lengths),
                "sequence_lengths": micro_batch.sequence_lengths,
                "lora_num_tokens": micro_batch.lora_num_tokens,
                "tensor_contract": _tensor_contract(micro_batch),
            }
        )

    sample_lengths = [len(sample.token_ids) for sample in batch.examples]
    result = {
        "schema_version": 1,
        "source": {
            "path": str(source).replace("\\", "/"),
            "sha256": _sha256(source_payload),
            "bytes": len(source_payload),
            "transport_step": batch.step,
        },
        "frozen_examples": {
            "msgpack_sha256": examples_sha256,
            "count": len(batch.examples),
            "total_tokens": sum(sample_lengths),
            "total_trainable_tokens": sum(
                sum(sample.mask) for sample in batch.examples
            ),
            "minimum_tokens": min(sample_lengths),
            "maximum_tokens": max(sample_lengths),
            "environment_counts": dict(
                sorted(Counter(sample.env_name for sample in batch.examples).items())
            ),
        },
        "normalized_batches": {
            str(step): {
                "path": str(output).replace("\\", "/"),
                "sha256": _sha256(encoded_by_step[step]),
                "bytes": len(encoded_by_step[step]),
                "transport_step": step,
                "examples_msgpack_sha256": examples_sha256,
            }
            for step, output in ((1, step_1_output), (2, step_2_output))
        },
        "packing": {
            "seq_len": 512,
            "num_train_workers": 1,
            "num_loras": 1,
            "microbatch_count": len(packed_rows),
            "packed_msgpack_sha256": _sha256(msgspec.msgpack.encode(packed)),
            "microbatches": packed_rows,
        },
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--step-1-output", type=Path, required=True)
    parser.add_argument("--step-2-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    result = build_contract(
        args.source,
        args.step_1_output,
        args.step_2_output,
        args.audit_output,
    )
    print(json.dumps(result["frozen_examples"], sort_keys=True))


if __name__ == "__main__":
    main()
