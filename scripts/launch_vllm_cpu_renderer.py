"""Launch vLLM's renderer when a CUDA wheel runs on a GPU-less host."""

from __future__ import annotations

import sys

import vllm.platforms
from vllm.model_executor.models.registry import _ModelRegistry
from vllm.platforms.cpu import CpuPlatform


def _is_text_generation_model(*_args: object, **_kwargs: object) -> bool:
    return True


def _is_pooling_model(*_args: object, **_kwargs: object) -> bool:
    return False


def main() -> None:
    # vLLM 0.24 builds every top-level CLI parser before dispatching
    # ``launch render``. A CUDA wheel on a GPU-less host therefore fails
    # device inference even though the renderer itself performs no inference.
    vllm.platforms._current_platform = CpuPlatform()
    # ModelConfig also spawns a subprocess to import the model implementation
    # solely to classify its runner. The renderer never loads that
    # implementation, and this diagnostic is pinned to a causal LM.
    _ModelRegistry.is_text_generation_model = _is_text_generation_model
    _ModelRegistry.is_pooling_model = _is_pooling_model
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = ["vllm", *sys.argv[1:]]
    vllm_main()


if __name__ == "__main__":
    main()
