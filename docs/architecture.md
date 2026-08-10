# Redco architecture

Redco combines a reusable Python core with versioned scientific protocols and
evidence. Ownership should be obvious without erasing frozen scientific or
security contracts. See [provenance](provenance.md) before moving historical
surfaces and [development](development.md) for verification.

## Dependency direction

Dependencies point inward:

```text
scripts and launch adapters
        |
        v
stage orchestration and publication
        |
        v
pure contracts, selectors, codecs, and validators
        |
        v
redco.algo / redco.env / redco.contracts
```

Lower layers must not import stage builders, reports, launch adapters, or
provider integrations. Importing analysis code must not contact Prime, load a
model, provision hardware, launch a provider, or publish an artifact.

## Ownership map

| Area | Owner | Responsibility |
|---|---|---|
| Shared contracts | `src/redco/contracts.py` | Dependency-free canonical JSON and cross-stage values |
| Byte integrity | `src/redco/integrity.py` | Raw SHA-256 and strict lowercase digest validation; no authority policy |
| Algorithmic core | `src/redco/algo/` | Branching and training calculations without publication effects |
| Replay environment | `src/redco/env/` | Artifacts, tracing, replay, cache, and task behavior |
| Integrations | `src/redco/integrations/` | Process, write-once, signature, and external provenance boundaries |
| Stage analysis | `src/redco/analysis/` | Versioned contracts, audits, state machines, and orchestration |
| Commands | `scripts/` | Stable commands and real runtime setup; reusable behavior belongs in `src` |
| Inputs | `configs/`, `datasets/`, `patches/`, `environments/` | Intent, frozen inputs, deployment deltas, and environment contracts |
| Evidence | `reports/`, `runs/` | Audits, manifests, receipts, ledgers, and terminal evidence |

A version suffix normally denotes a distinct frozen contract, not a clone to
collapse. Similar names are not evidence of identical semantics.

`integrations.signed_subprocess.sign_payload()` is historically named but uses
an unkeyed SHA-256 checksum. It proves self-integrity, not signer authority.
Only a verified key-backed signature plus trusted-key policy grants authority.

## Stage-D trust boundaries

- Scientific identity and selection law belong to source contracts, spawn
  provenance, scientific branch groups, and exact-action owners.
- `stage_d_receipt_ledger` owns the durable scientific receipt chain; its small
  contracts and read-only validator remain separate.
- `stage_d_evaluation_*` owns evaluation transport, actuation, evidence, and
  state. Immutable public models live in `stage_d_evaluation_state`; complete
  record validation and transitions live only in `stage_d_evaluation_reducer`;
  content-addressed storage lives in `stage_d_evaluation_codec`.
- `stage_d_update_ledger` is a separate trainer-authorization state machine.
  `stage_d_ledger` is analytical usage/cost aggregation, not authority.
- V13 source authentication, one-attempt selection, publication, readiness,
  launch, and lifecycle remain separate because their mutability and authority
  differ.

These domains may share byte primitives only when behavior is identical. Do not
hide schemas, locks, transitions, evidence rules, or fail-closed decisions behind
a generic ledger or protocol API.

## Refactor rules

1. One behavior owner; retain an explicit versioned copy only when historical
   semantics differ.
2. Parse and validate completely before mutation, publication, or external
   effects.
3. A builder and check-only mode share one payload owner. Publication is atomic,
   rejects aliases to immutable inputs, and publishes the manifest last when the
   protocol requires it.
4. Scripts retain real operational setup or a stable compatibility command;
   identity wrappers and reusable Python logic do not multiply by default.
5. Represent identifiers, dispositions, capabilities, receipts, and states with
   precise types rather than magic-string protocols.
6. Trust tests cover tampering, aliases, rollback, malformed schemas, and
   fail-closed paths with independent oracles.

Consolidate mechanics only when canonical bytes, domain/version, authority,
locking/recovery, publication order, and scientific cohort/seed/threshold laws
are identical. [Provenance](provenance.md) owns frozen-byte and successor rules.
