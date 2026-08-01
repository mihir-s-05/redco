# Stage D E1 Prime training-bridge report

Status: **PASS** (CPU integration contract; no scientific arm and no 4B model update).

## Result

Verified C1 branch-group artifacts now compile into a canonical, sealed batch for the pinned Prime-RL clean decision loss. The compiler reverifies the raw artifact bytes, complete target rosters, source/group/action-slot coverage, per-prompt behavior laws, a shared prompt-independent policy family, exact prompt/action masks, sampled log probabilities, scalar advantages, decision weights, and decision normalizers. Flat groups remain present with zero advantages.

The Prime adapter serializes an actual `TrainingBatch` with msgpack, decodes it, runs the pinned `prepare_batch`, reconstructs every packed sequence, and requires the packed token/mask/logprob/temperature/advantage/weight/environment/normalizer multiset to equal the sealed bridge records. The actual pinned `clean_decision_loss` equals an independent formula. The optimizer audit consumes only this rederived packed object.

The separate update ledger binds the producer seal, C1 batch identity, bridge bytes, Prime payload bytes, runtime, trainer configuration, and expected pre-update state. Its canonical path verifies the binding, durably authorizes, executes exactly one Prime loss/backward/AdamW step, records pre/post model and optimizer hashes, completes, and seals. Exceptions and a real `os._exit` hard death leave the attempt consumed; read-only inspection remains possible and retry is rejected.

## Prime evidence

- Pod: `9a298ff400e848bc86c182a54a06399c` (`CPU_NODE x1`, DataCrunch FI, non-spot)
- Rate: `$0.0279/hour`
- Pinned Prime-RL commit: `3b22dd951cad1036d1fe8dd0a0bfc40807a9b360`
- Final live test: `17 passed`
- Training-batch identity: `78effc2dcb5ea51abdee6c0c04947088ad09372ab4716bc4311c8efb80a64e38`
- Bridge payload: `f4c15057513e80cc4cba133cc755fa7b7bc7eea7faaec9d48661e61c01118ac3`
- Prime msgpack payload: `35691843105fb90a7c2fa5d0b7a45f7e55be7c77b9f63c4587cfa17553696903`
- Records / distinct behavior laws / normalizer: `8 / 2 / 2.0`
- Prime loss / independent loss: `0.0 / 0.0`
- Gradient L2: `0.24650332429581734`
- Pre/post model: `e68e9a4c2f7f88779ffb18947db30867bbc20bbab3c2edf34b82c40f7a76a623` / `eb57ac1285d2f4d390984b65722068702cdc82c02c5b10a354720c9f7a4c84ba`
- Pre/post optimizer: `fa3753f15873d4c8f707cb32dd3994829f6509be31278408077f8c7dd89971e8` / `ccb41bb0b74b6e7d36bc7b9556d0d9d46fbc4809bda5aaae8847c97bf03ccbbc`
- Step evidence: `26d60a816f8747c508be456852ee7589607189a363ec44563c9e9f39e9e524c9`
- Terminal ledger status: `complete`

The live test uses deterministic synthetic C1 fixtures and placeholder frozen binding values for the producer/runtime/config manifests. Before a scientific update, the deployment preflight must observe and freeze the actual runtime/config hashes. For this audit, the observed Prime commit and patched source hashes were recorded explicitly; local and pod bridge/test files were byte-identical.

## Validation and review

- Local Ruff formatting/check: pass
- Local strict mypy: pass
- Local focused E1/C1/D1/D1b suite: pass; only expected Prime-only skips
- Fresh pod run after the final code changes: 17/17 pass
- Independent Sol xhigh thermo-nuclear code review: **GO**, no P0/P1 findings
- Persistent Prime disks: zero

## Billing

- Wallet before provisioning: `$44.9264`
- Final balance after termination: `$44.8769`
- Final billed cost: `$0.0495`
