from pathlib import Path


def test_canonical_scorer_excludes_unstable_runtime_features() -> None:
    text = Path(
        "scripts/score_stage_c6_canonical_transformers.py"
    ).read_text(encoding="utf-8")
    assert "attn_implementation=\"eager\"" in text
    assert "torch.use_deterministic_algorithms(True)" in text
    assert "allow_tf32 = False" in text
    assert "batch_size" in text
    assert "from vllm" not in text
    assert "LoRARequest" not in text
    assert "SamplingParams" not in text
