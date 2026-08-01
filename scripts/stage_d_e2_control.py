"""Command-line controller for the bounded Stage-D E2 live update."""

from __future__ import annotations

import argparse
from pathlib import Path

from redco.analysis.stage_d_e2_live import (
    base_snapshot_manifest,
    prepare_live_inputs,
    validate_prime_configs,
    verify_terminal_run,
)
from redco.analysis.stage_d_live_update import authorize_live_update, complete_live_update
from redco.contracts import canonical_json


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("base-manifest")
    manifest.add_argument("--model-root", type=Path, required=True)
    manifest.add_argument("--revision", required=True)
    manifest.add_argument("--output", type=Path, required=True)

    parse_configs = subparsers.add_parser("parse-configs")
    parse_configs.add_argument("--trainer-config", type=Path, required=True)
    parse_configs.add_argument("--control-config", type=Path, required=True)
    parse_configs.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--sealed-batch", type=Path, required=True)
    prepare.add_argument("--golden-manifest", type=Path, required=True)
    prepare.add_argument("--trainer-config", type=Path, required=True)
    prepare.add_argument("--control-config", type=Path, required=True)
    prepare.add_argument("--base-manifest", type=Path, required=True)
    prepare.add_argument("--rollout", type=Path, required=True)
    prepare.add_argument("--binding", type=Path, required=True)
    prepare.add_argument("--preflight", type=Path, required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--binding", type=Path, required=True)
    authorize.add_argument("--prestate", type=Path, required=True)
    authorize.add_argument("--ledger-root", type=Path, required=True)
    authorize.add_argument("--consumer-id", required=True)
    authorize.add_argument("--output", type=Path, required=True)

    complete = subparsers.add_parser("complete")
    complete.add_argument("--binding", type=Path, required=True)
    complete.add_argument("--prestate", type=Path, required=True)
    complete.add_argument("--authorization", type=Path, required=True)
    complete.add_argument("--poststep", type=Path, required=True)
    complete.add_argument("--ledger-root", type=Path, required=True)
    complete.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--sealed-batch", type=Path, required=True)
    verify.add_argument("--binding", type=Path, required=True)
    verify.add_argument("--prestate", type=Path, required=True)
    verify.add_argument("--poststep", type=Path, required=True)
    verify.add_argument("--ledger-root", type=Path, required=True)
    verify.add_argument("--metrics", type=Path, required=True)
    verify.add_argument("--token-export", type=Path, required=True)
    verify.add_argument("--adapter", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "base-manifest":
        args.output.write_bytes(base_snapshot_manifest(args.model_root, revision=args.revision))
    elif args.command == "parse-configs":
        args.output.write_bytes(
            canonical_json(validate_prime_configs(args.trainer_config, args.control_config))
        )
    elif args.command == "prepare":
        prepare_live_inputs(
            sealed_batch_path=args.sealed_batch,
            golden_manifest_path=args.golden_manifest,
            trainer_config_path=args.trainer_config,
            control_config_path=args.control_config,
            base_manifest_path=args.base_manifest,
            rollout_path=args.rollout,
            binding_path=args.binding,
            preflight_path=args.preflight,
        )
    elif args.command == "authorize":
        args.output.write_bytes(
            authorize_live_update(
                binding_bytes=args.binding.read_bytes(),
                prestate_bytes=args.prestate.read_bytes(),
                ledger_root=args.ledger_root,
                consumer_id=args.consumer_id,
            )
        )
    elif args.command == "complete":
        completion = complete_live_update(
            binding_bytes=args.binding.read_bytes(),
            prestate_bytes=args.prestate.read_bytes(),
            authorization_bytes=args.authorization.read_bytes(),
            poststep_bytes=args.poststep.read_bytes(),
            ledger_root=args.ledger_root,
        )
        args.output.write_bytes(
            canonical_json(
                {
                    "ledger_id": completion.ledger_id,
                    "completion_sha256": completion.completion_sha256,
                    "post_model_sha256": completion.post_model_sha256,
                    "post_optimizer_sha256": completion.post_optimizer_sha256,
                    "step_evidence_sha256": completion.step_evidence_sha256,
                }
            )
        )
    elif args.command == "verify":
        result = verify_terminal_run(
            sealed_batch_path=args.sealed_batch,
            binding_path=args.binding,
            prestate_path=args.prestate,
            poststep_path=args.poststep,
            ledger_root=args.ledger_root,
            metrics_path=args.metrics,
            token_export_path=args.token_export,
            adapter_path=args.adapter,
        )
        args.output.write_bytes(canonical_json(result))


if __name__ == "__main__":
    main()
