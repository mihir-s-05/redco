# Stage D E2 4B bridge: terminal report

Date: 2026-08-01

## Outcome

The frozen engineering bridge passed. One sealed eight-record ReDCO batch
traversed the pinned Prime serialization and packing path, a 4B Qwen LoRA
forward/backward pass, exactly one locally authorized AdamW update, token
export, and adapter-only checkpointing.

This result is intentionally engineering-only. It does not establish on-policy
estimator validity, task learning, scientific Stage D performance, or
superiority to a baseline.

## Frozen execution

- Source commit: `e820b10f9da3d48ebf349bbd13e0c58a5d1f7e1c`
- Hardware amendment commit: `eef60da4a9873ef5b34fc99ad184beee251da172`
- Protocol SHA-256: `4b3ff3aaca5354a19363a3e54c29d87ca3e3db7522c5a9d16d62f1db6a7bc767`
- Model: `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`
- Prime commit: `3b22dd951cad1036d1fe8dd0a0bfc40807a9b360`
- Training-batch identity: `e6edc0ed732d77f3f28412635cf4bcb69113c38b63c1325154c34b67ee58c4c1`
- Hardware: one non-spot A6000 48 GB, resource `7574d6`, at $0.54/hour
- Pod: `2470254da12f4674879f342da27f33a9`
- Package manager: `uv` only; no persistent storage was created

The trainer reached the frozen pre-authorization boundary after config and
batch parsing, 4B forward/backward, and finite gradient clipping. The local
single-use ledger then authorized one update. Its post-step receipt was copied
back and the ledger was completed before checkpoint verification. No second
authorization or optimizer attempt occurred.

## Verified evidence

The frozen terminal verifier returned `pass` with:

- optimizer steps: exactly `[1]`
- gradient L2 norm: `0.8235899209976196`
- pre-model state: `4e5bd24f410de9238e83e93818c4b74d54d3f6feb7aa31a74958e66f61aecc9b`
- post-model state: `e6a4ae56c4e35a4deb2efa47cbc24649d13b553ab129b8977078588eb855514e`
- complete local ledger: four chained records
- token export: eight exact-field records matching the sealed batch
- retained adapter SHA-256: `1e737fbfea994f50ade2ef8ae9d5c6a79ad34438117bd6f682d1bb96d6092e17`
- token export SHA-256: `35e4ff2449ae953615a3cf8a0b08d469a49b5eb3907560d2c9fa8ece63d3b36f`
- metrics SHA-256: `e4c9457f83d7c6cc15b0096aff7348ede98762cd2c214fa20be61b8ab40eb813`
- post-step receipt SHA-256: `404da971bc14b5683a68ef3306ff7d2ba0dfdd79f97f80519748ee4964818e7c`
- terminal-verification SHA-256: `7f832adfe81b7a08108bc693215afd1bef1da3e65e7a918751441792037f23d`

Prime reported loss `-0.02170310914516449`, entropy `3.024822950363159`,
mismatch KL `6.063998699188232`, peak GPU memory `17.59 GiB`, and 98.28 seconds
for the single training step. The pass decision uses the frozen exact artifact
contracts, not these descriptive metrics alone.

The compact local evidence bundle is under
`runs/stage-d/e2-4b-live-v1-local`. It is about 66 MB and retains one adapter,
receipts, the ledger, metrics, token export, and logs. The duplicate broadcast,
base-model snapshot, optimizer state, merged model, package environment, and
CUDA/Hugging Face caches were not retained.

## Validation note

The lightweight Windows environment lacked Torch, and a temporary Windows
Torch import did not produce a usable process result. The frozen verifier was
therefore run in the already-pinned Prime environment against the recovered
artifacts and a copied, byte-identical local ledger; its JSON verdict was then
copied into the local evidence bundle. This did not invoke the model or perform
another update. The verdict's referenced adapter, token export, metrics,
receipts, and ledger all remain locally available by the hashes above.

## Billing and teardown

Prime's billing row records `$0.18` for this pod. The wallet moved from
`$44.8769` before provisioning to a displayed `$44.69` after termination.
Teardown was confirmed with zero active pods and zero persistent disks.

## Disposition

The 4B trainer bridge is cleared. The next scientific work must still create
real on-policy Stage D rollouts, pass the frozen eligible-and-informative target
density gate, and compare the shared-initialization stock, branch-global, and
local-credit arms. This smoke supplies no shortcut around those gates; it
removes the outstanding uncertainty that the synthetic ReDCO records could not
survive the real 4B Prime training path.
