from __future__ import annotations

import math
from types import SimpleNamespace

from redco.analysis.stage_c3_power import ROUTES
from redco.analysis.stage_c5_constrained import ROUTE_CHOICES
from redco.analysis.stage_c6_v3_interface import verify_interface


def _root_scores() -> dict[str, object]:
    return {
        "temperature_2": {
            "token_details": {
                route: [
                    {
                        "token_id": token_id,
                        "temperature_2_logprob": math.log(probability),
                    }
                ]
                for route, token_id, probability in zip(
                    ROUTES,
                    (11, 22, 33, 44),
                    (0.1, 0.2, 0.3, 0.4),
                    strict=True,
                )
            }
        }
    }


def test_exact_interface_uses_live_behavior_not_static_reference() -> None:
    behavior = -0.7
    trainer = -0.8
    trace = {
        "agent": {
            "name": "context",
            "sampling": {
                "extra_body": {
                    "structured_outputs": {"choice": list(ROUTE_CHOICES)}
                }
            },
        },
        "nodes": [
            {
                "sampled": True,
                "token_ids": [44],
                "is_content": [True],
                "logprobs": [behavior],
            }
        ],
    }
    batch = SimpleNamespace(
        examples=[
            SimpleNamespace(
                token_ids=[44], mask=[True], logprobs=[behavior]
            )
        ]
    )
    token_export = {
        "token_ids": [44],
        "loss_mask": [True],
        "inference_logprobs": [behavior],
        "trainer_logprobs": [trainer],
        "log_importance_ratio": [trainer - behavior],
    }

    result = verify_interface(
        traces=[trace],
        batch=batch,
        token_exports=[token_export],
        root_scores=_root_scores(),
        expected_context_traces=1,
    )

    assert result["status"] == "passed"
    assert result["maximum_static_reference_error_nats"] > 0.2
    assert all(result["checks"].values())


def test_exact_interface_rejects_untransported_behavior_logprob() -> None:
    behavior = -0.7
    trace = {
        "agent": {
            "name": "context",
            "sampling": {
                "extra_body": {
                    "structured_outputs": {"choice": list(ROUTE_CHOICES)}
                }
            },
        },
        "nodes": [
            {
                "sampled": True,
                "token_ids": [44],
                "is_content": [True],
                "logprobs": [behavior],
            }
        ],
    }
    batch = SimpleNamespace(
        examples=[
            SimpleNamespace(token_ids=[44], mask=[True], logprobs=[-1.1])
        ]
    )
    token_export = {
        "token_ids": [44],
        "loss_mask": [True],
        "inference_logprobs": [-1.1],
        "trainer_logprobs": [-0.8],
        "log_importance_ratio": [0.3],
    }

    result = verify_interface(
        traces=[trace],
        batch=batch,
        token_exports=[token_export],
        root_scores=_root_scores(),
        expected_context_traces=1,
    )

    assert result["status"] == "failed"
    assert not result["checks"]["packed_behavior_logprobs_match_traces"]
