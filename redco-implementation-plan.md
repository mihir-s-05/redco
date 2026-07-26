# ReDCO: Assessment and Implementation Plan

**Recursive Dataflow Credit Optimization — from idea to working system**

*Prepared July 2026. This document is self-contained for a new contributor: the
background primer and links to every referenced resource are in Appendix D, the
original (pre-correction) ReDCO spec and the full list of corrections in Appendix A,
data schemas and repo layout in Appendix B, and hyperparameter defaults in
Appendix C. Read §0–§2 first, then Appendix D if any referenced system (RLM,
prime-rl, verifiers, SkyRL) is unfamiliar.*

---

## 0. Verdict and framing

ReDCO has merit, in a specific and narrower form than the original spec. The gap is
real and publicly acknowledged: every existing RLM training recipe broadcasts a
single terminal advantage across the entire recursive rollout (GRPO broadcast in the
harness blog via prime-rl; verbatim parent-advantage inheritance in the
alphaXiv/SkyRL work, whose authors explicitly name finer-grained credit assignment
as the open problem). The 2025–26 agentic-RL literature (StepPO, GiGPO, Turn-PPO,
Tree-GRPO, RTMC, TRACE, TEMPO, CRAFT, C3, Agent Lightning) has established that
finer-than-trajectory credit helps — but none of it exploits the fact that an RLM
rollout is an executable dataflow program whose dependency structure is partially
recoverable.

**The defensible form, stated precisely:**

> **ReDCO-Lite: exact behavior-policy counterfactual action branching at explicit
> RLM action boundaries (root turns and sub-call outputs), all-branch leave-one-out
> training, and dependency-sound dynamic partial replay in a restricted executable
> dataflow environment.**

The division of labor inside that sentence matters for novelty and experimental
design:

- **The statistical estimator is C3-style fixed-state branching** (freeze a behavior
  policy, sample several complete actions from the same state, roll each to
  completion under that policy, form leave-one-out advantages, train on every
  branch). ReDCO does not improve on this estimator.
- **The graph is the causal-slicing and replay system.** Its contribution is making
  those counterfactuals *cheaper* (replay only the true dynamic descendants of an
  intervention) and *applicable to executable recursive agents* (REPL state,
  sub-calls, artifacts), not statistically different.
- **Recursive RLM integration and structural interventions are domain-specific
  extensions**, staged after the core result.

This immediately implies the decisive experiment (§6.1): run identical
counterfactual action samples once with full-suffix environment replay and once
with graph-sliced replay, under the same cached-action reuse semantics. If the
reward distributions disagree, the graph replay is unsound (a bug, not a method).
If they agree and sliced replay is substantially cheaper, that is the graph
contribution — and at matched compute, the savings should buy more branches and
better learning.

**Scope honesty:** the initial method is **turn/sub-call-level ReDCO (Level 1)** —
one macro-action per root REPL turn or child output — not arbitrary
operation-level or fully recursive internal credit. Operation-level credit requires
making operations explicit policy actions (Stage E); recursive internal credit is a
later extension. Claims are worded accordingly throughout.

**Training stack, decided up front:** prime-rl + verifiers (Prime Intellect). This
is the stack the harness blog's own experiments ran on, which makes the incumbent
baseline exactly reproducible and the flagship head-to-head a same-stack, full
fine-tuning comparison; verifiers provides the multi-turn environment machinery
(rollout loop, env servers, judge rubrics, branch-carrying traces) an RLM
environment needs; and prime-rl's documented "bring your own algorithms" extension
point accepts per-token advantage arrays — the exact shape of ReDCO's credit. §4
gives the full integration design.

The plan below is a staged program with explicit decision gates. Each stage produces
a result that is independently publishable or independently kills the project
cheaply. The pivot-out is also defined: if exact prompt provenance shows nearly
every node affects every future prompt (no slicing savings), the correct fallback is
a simpler C3-style turn-level credit method for RLMs — still novel, much simpler —
rather than continuing into the critic and structural phases.

### Implementation checkpoint — 2026-07-25

- The Tier-0 CPU campaign passed 10,000/10,000 deterministic sliced-vs-full
  comparisons, but only for value replacements on a static synthetic command
  topology. This establishes substrate soundness for that scope, not the full
  dynamic-branch replay contract below.
- The campaign's 2.05× aggregate event-work reduction is synthetic
  instrumentation evidence, not a production RAF result. Representative RLM
  traces must still demonstrate the affected-work fraction required by §8.
- The frozen GPU trainer pairs validated stock-noise margin transfer across four
  unseen seeds. They did not execute orchestrator algorithm selection and are not
  a no-op integration test.
- A subsequent CPU producer-equivalence gate executes config dispatch,
  `finalize_group`, advantage/routing stamping, trainer packing, and serialization
  under both `grpo` and `redco_noop`; the resulting trainer bytes are exact-equal.
- Before Stage C, implement and pass end-to-end replay equivalence for
  branch-specific dynamic topology (different turns, calls, and artifacts) and
  measure RAF on representative RLM traces. The current `TopologyDivergence`
  record is not an implementation of that behavior.

---

## 1. The corrected method

### 1.1 The estimand and the branch-group estimator

At a policy decision node v, let S_v be the complete pre-action state, o_v the exact
policy observation (including prompt token IDs), a_v the sampled action, and π_b the
frozen behavior policy. The target quantity is the ordinary local advantage

    A^{π_b}(S_v, a) = Q^{π_b}(S_v, a) − V^{π_b}(S_v).

**Branch groups (all branches train — none are wasted).** For each target node,
sample n = K+1 actions i.i.d. from the *exact* behavior policy:

    a_1 … a_n ~ π_b(· | o_v)        # identical checkpoint, temperature, top-p,
                                    # system prompt, grammar, tool availability;
                                    # only the seed varies

Pre-sample c exogenous seed bundles U_1…U_c and reuse the *same* U_m across all
actions (common-random-number coupling). A bundle is not a sequential RNG stream:
every stochastic event receives a counter-based seed

    seed = PRF(master_seed, rollout_id, target_id, replicate_m,
               event_address, occurrence_index)

where `event_address` is structural (parent node ID, turn index, call-slot index),
not content-derived. Thus an extra call cannot shift every later seed; disappearing
calls consume nothing and new calls receive deterministic fresh keys. Coupling
after topology divergence is a variance-reduction heuristic only — exogeneity of
the keyed seeds, and therefore estimator validity, does not depend on the coupling
remaining strong.

The original rollout records its post-target continuation namespace as U_1; because
the target is committed before a_v, downstream original events can already use
keys containing `(target_id, replicate_1)`. Any U_2…U_c are fixed before branch
rewards are observed. Candidate-action generation uses a separate structural
`action_slot_i` key and each sampled a_i is held fixed across all c continuation
replicates. This makes same-prompt/same-seed recorded-action reuse possible for
replicate 1 without conflating candidate-action randomness with continuation
randomness. For each action and seed bundle, partial-replay to termination under
π_b (§1.3):

    R_{i,m} = R( Exec_{π_b}(S_v, a_i, U_m) ) − λ_phys · cost(branch)
    Q̂_i     = (1/c) Σ_m R_{i,m}
    Â_i     = Q̂_i − (1/(n−1)) Σ_{j≠i} Q̂_j          # leave-one-out

The local policy-gradient sample uses **every** branch, weighted so the branch group
counts as one decision state:

    ĝ_v = (1/n) Σ_i Â_i ∇_θ log π_θ(a_i | o_v)

λ_phys is a *fixed physical* cost penalty (dollars, output tokens, or latency), not
a group-relative normalized penalty — group-relative normalization changes with
branch-group membership and makes branch rewards incomparable. `cost(branch)` is
the **logical deployment cost of the counterfactual workflow as if run fresh**:
cached/reused policy actions are billed at their normal token cost. Actual compute
spent evaluating the branch is a separate experiment ledger and never changes the
reward.

**Selection-conditioning rule.** Stage C commits the target **online, before its
action is sampled**, using only prefix/static features available at that point:
node type, depth, turn index, task metadata, and a replay-cost prediction computed
from those features. Realized descendant-set size is post-action information and
is forbidden. With this online commitment the original action remains an
unconditional sample from π_b(·|o_v) and joins the branch group as a_1. If selection
ever uses the realized output, descendants, terminal reward, or any other
post-action information, exclude the original and sample all n actions fresh.

**Validity conditions (all enforced; violations are logged, not silently absorbed):**
1. *Same sampling distribution*: only the seed varies across branches. Temperature
   or model-choice variations are diagnostics, never baseline branches.
2. *No silent rejection sampling*: an invalid or ill-typed action is **executed and
   receives its true error/failure reward** (the natural REPL semantics — bad code
   raises), or the action space is constrained identically at rollout and branch
   time (grammar/constrained decoding), or a repair step is a deterministic part of
   the environment applied identically in both. Never "resample/skip."
3. *Exact behavior snapshot*: π_b is a **pinned checkpoint served for the duration
   of the branch evaluations** (§4.2 item 6). The "live server, a few steps ahead"
   approximation is an engineering option only after the estimator is validated.
4. *Behavior-policy continuations*: downstream policy actions are regenerated under
   π_b whenever their exact observation changed (§1.3); "fixed continuation" never
   means "hold downstream actions fixed regardless of what they observe."
5. *Nonadaptive allocation*: n and c are fixed before observing any branch reward.
   Adaptive allocation (more replays for uncertain nodes) is a later, explicitly
   biased optimization.

**The honest claim:** the raw LOO estimator is unbiased for A^{π_b} conditional on a
replay-complete state, identical behavior-policy sampling, fixed branch allocation,
and behavior-policy continuations. Everything layered on top — clipping, staleness,
selective targeting, blending with trajectory credit — is a practical, biased
stabilizer and is labeled as such.

**Necessity credit (diagnostics only).** Degenerate interventions (delete/bypass,
identity filter, abstention, monolithic-sub-call replacement, alternative chunk
size) measure

    N_I(v) = R(G) − E[ R(Exec_{π_b}(S_v, degenerate_I, U)) ]

indexed by the intervention I — necessity is not an intrinsic node property; it
depends on the replacement, the repair semantics, the continuation policy, and
redundancy/synergy with other nodes. Uses: analysis plots, pruning candidates,
counterexample generation, critic pretraining data. **Not** a shaping reward
(§2 Stage E explains why: a policy can manufacture necessity by threading a useless
artifact through every later operation, making deletion break the workflow while
adding no capability).

### 1.2 Combining local and trajectory credit (replace or blend — never add)

For every ReDCO arm, trajectory credit uses leave-one-out trajectory advantage

    A_traj,i = R_i − (1/(G−1)) Σ_{j≠i} R_j.

The stock broadcast arm alone retains prime-rl's exact default,
`R_i − mean(R_1…R_G)`, for incumbent reproducibility. The inclusive group mean is
`(G−1)/G` times the LOO estimator in expectation; mixing that stock-scaled signal
with unscaled branch LOO would inflate targeted-node gradients by `G/(G−1)`
(≈14% at G=8). Using RLOO consistently within ReDCO avoids this hidden reweighting
without a magic rescaling constant.

A_traj (trajectory RLOO) and A_cf (branch-group LOO) are **two estimators of the
same policy-gradient contribution at node v**, not two objectives. Adding them
(α·A_traj + β·A_cf) gives targeted nodes an expected gradient scale of ~(α+β)
while untargeted nodes get α. The correct combination is a convex blend:

    A_used(v) = (1 − η_v) · A_traj(v) + η_v · A_cf(v),      0 ≤ η_v ≤ 1

- η_v = 0 for untargeted nodes (they keep ordinary trajectory credit),
- η_v = 1 for a high-confidence real branch group (**the Phase-1 default**: at the
  selected node, branch credit *replaces* trajectory credit; the branch group's
  gradient contribution replaces the original action's trajectory-credit term),
- intermediate η_v only when deliberately trading trajectory-credit cost against
  counterfactual-estimator variance (later, with measured variances).

Both signals stay in **raw reward units**. Node-type bucket normalization and
clipping are optimizer heuristics — available if raw-scale training proves
unstable, ablated, and never described as part of the causal estimator. The same
label applies to weighting child datums by 1/k_g (alphaXiv's depth balancing): k_g
is policy-dependent, so this changes the objective relative to summing all decision
log-likelihood gradients; keep it as the default for comparability with the
incumbent, and ablate it. A targeted child branch group inherits the same outer
`1/k_g` weight as the original child contribution; each of its n records therefore
has weight `1/(k_g·n)`.

### 1.3 The graph: an event-level causal trace, and dependency-sound partial replay

**Node taxonomy** (kept strictly separate):
- **Policy nodes**: LM-generated macro-actions that receive gradients — one per
  root REPL turn, one per child/sub-call output. Each records S_v references, exact
  observation o_v (prompt token IDs), action token span, behavior log-probs,
  checkpoint ID, decoding config.
- **Environment operation nodes**: executions (Python cells, tool calls) used for
  dependency tracking and replay — no gradients.
- **Artifact nodes**: values, **versioned SSA-style** (x⁽⁰⁾, x⁽¹⁾, …) so in-place
  mutation cannot make the event graph cyclic or hide which version a consumer read.
- **Reward/resource nodes**: correctness components, token/latency/cost meters.

**Edge taxonomy** (all six required for sound replay):
1. **Dataflow**: artifact version produced → consumed.
2. **Control-dependence**: branch predicates, exceptions, retries, loop conditions,
   termination — which operations *execute at all* depends on these.
3. **Call**: parent turn → child rollout → returned artifact.
4. **Side-effect / ordering**: mutable objects, files, module globals, caches,
   subprocesses — anything where execution order carries information.
5. **Observation (prompt provenance)**: the **exact prompt token spans** of every
   policy node, mapped back to the artifacts/stdout that produced them. Context
   edges are *actual-inclusion* edges — an artifact printed at turn i is an input to
   turn j only if it actually appears in turn j's rendered prompt (the verifiers RLM
   scaffold does context dropping, so old prints are not automatically present).
   All-to-all "everything printed reaches every later turn" is a conservative
   *fallback* used only where provenance cannot be reconstructed — it collapses the
   graph toward a chain and destroys slicing value, so measure how often it fires.
6. **Resource**: token count, latency, cost, memory — needed when these enter
   reward.

**The replay rule (the load-bearing invariant):**

> A downstream policy output may be reused if and only if its exact pre-action
> observation — including prompt token IDs — **and its event-keyed seed** are
> unchanged. If unchanged, both replay modes reuse the recorded action as part of
> the shared `Exec` semantics. If either changed, the action is regenerated under
> the frozen π_b.

Cached-action reuse is therefore not the sliced mode's special shortcut. Both
`sliced` and `full_suffix` use it, so their comparison isolates dependency slicing
rather than vLLM kernel reproducibility. A separate audit reissues a sample of
same-prompt, same-seed model calls and measures whether vLLM reproduces the recorded
tokens; failures there are sampling-system findings, not slicing failures.

**Incremental partial-replay algorithm** for an intervention at node v:
1. Restore the exact state immediately before v (artifact store + env snapshot);
   inject the branch action a_i.
2. Re-execute affected deterministic environment descendants (dataflow + control +
   side-effect closure), reusing cached sub-call outputs keyed by
   (prompt-hash, model-version, seed) where their keys are unchanged.
3. At every downstream policy node, reconstruct its actual prompt.
4. Prompt hash and event-keyed seed unchanged → reuse the recorded action.
5. Prompt hash or event-keyed seed changed → resample from π_b; from here the
   branch's graph may
   **diverge in topology** (different numbers of turns, calls, artifacts).
6. Continue to termination; evaluate reward and fixed physical cost.

`full_suffix` re-executes every environment event after v while obeying the same
policy-action reuse rule. `sliced` skips events outside the computed dynamic
dependency closure. All injected actions, keyed seeds, cached-action decisions, and
logical cost accounting are otherwise identical.

Because branches diverge, the intervened execution is **not** "the original graph
with one value overwritten." Notation: each branch is its own dynamic execution

    (G_{i,m}, R_{i,m}) = Exec_{π_b}(S_v, a_i, U_m).

**Current implementation boundary (2026-07-25):** Tier-0 replay currently
performs value replacement on a static typed-command topology. It detects and can
record topology divergence, but it does not yet construct and compare independent
branch-specific graphs when an intervention changes turns, calls, or artifacts.
That missing behavior is a blocking Stage-B item, not covered by the 10,000-pair
campaign.

**Cost note:** cached-action reuse in step 4 reduces the cost of *both* replay modes
relative to naive continuation resampling; it is not credited to graph slicing.
The incremental sliced-vs-full saving comes from step 2: sliced replay skips
environment events outside the dependency closure, while full-suffix replay
reexecutes them. The affected-work fraction f̄ and resulting savings are
**empirical quantities gated in Stage B**, not assumptions (§8).

**Correctness oracle:** exact state restoration means immutable,
content-addressed artifacts plus deterministic prefix re-execution from the event
log using the event-keyed seeds. In-memory namespaces, process snapshots, and other
fast restore paths are optimizations and must be continuously checked against this
oracle.

### 1.4 The Phase 0–1 execution substrate: restricted typed dataflow

Sound replay for arbitrary mutable Python (aliasing, `obj.field = x`,
`cache["k"] = v`, `globals()`, iterators, subprocess/file/network state) is
high-risk; AST analysis plus a wrapped namespace will miss cases. The first
experiments therefore run on a **typed, mostly functional command environment**:

    PARTITION(input=artifact_3, chunk_size=4096) -> chunks_4
    CALL(prompt=chunk_7, schema=EvidenceList)    -> evidence_8
    FILTER(input=evidence_8, predicate=...)      -> evidence_9
    AGGREGATE(inputs=[...], method=...)          -> answer_10
    VERIFY(input=answer_10, rubric=...)          -> verdict_11
    FINAL(input=answer_10)

with Python admitted only through a declared-IO escape hatch:

    PURE_PYTHON(declared_inputs=[artifact_3, artifact_5],
                declared_outputs=[artifact_8], code=...)

— immutable or copied inputs, explicit output schema, filesystem/network disabled,
wrapped randomness, content hashes before and after, no arbitrary persistent
objects. This buys a provable replay property (see the soundness statement below)
at the cost of restricting the action space.

**Transfer caveat, stated openly:** the restricted substrate differs from the
free-form Python REPL that the harness blog and alphaXiv trained on. Results in the
restricted substrate validate the *estimator and the replay system*; Stage E
bridges back to the general REPL (strict mode: declared-IO cells, print truncation,
type allowlists) before the Phase-4 head-to-head, and the general-REPL setting is
where comparability with published RLM results lives.

**Replay soundness statement (to test empirically in Stage B, and state as a
theorem for the paper):**

> If every operation is deterministic conditional on its recorded parents and
> exogenous seed, and the event graph includes every dataflow, control,
> observation, side-effect, and resource dependency, then recomputing the dynamic
> descendants of an intervention produces the same terminal result as full replay
> under the same exogenous variables.

Any measured discrepancy between sliced and full replay is a replay bug by
definition.

### 1.5 Intervention targeting and sibling handling

**Targeting (Stage C defaults):** at most **one target node per rollout**, chosen
and committed **online before its action is sampled**, using only prefix/static
features — node type (sub-call outputs first), depth, turn index, task metadata, and
predicted replay cost from those features. Realized descendant-set size, node
output, future topology, and terminal reward are forbidden (§1.1 selection rule).
Record the commitment and feature vector with each branch tuple. If no eligible
sub-call node exists, skip branching and log the skip; do not fall back to a root
turn during Stage C. Multiple targets per rollout come later.

**Sibling graphs are not free counterfactual rewards.** The GRPO group's G rollouts
for the same task contain structurally similar nodes, but a sibling's terminal
reward reflects its own upstream actions, artifacts, prompts, downstream actions,
and randomness:

    R(G_sibling) ≠ R(G_g^{v ← a_sibling})   in general.

(CRAFT's sibling estimator is explicitly a group-level, state-marginal quantity for
this reason — it never claims fixed-state transplantation.) Sibling data is used in
exactly three ways:
1. **Exact state match** (same prompt token IDs, artifact versions/hashes, env and
   REPL snapshot identity, checkpoint + decoding config, tool-cache and external
   state, RNG bundle): the sibling output is a valid alternative *action* — and it
   is still normally inserted and replayed rather than its reward transplanted.
2. **State match, different continuation**: splice the sibling action into the
   current checkpoint and replay under π_b (an ordinary branch whose action
   generation was free).
3. **Structural similarity only**: action proposals, graph-alignment training data,
   retrieval, critic pretraining — never a counterfactual reward.

A "state match" is the strong condition in case 1, not "similar graph position."

### 1.6 Losses: node-aligned objective, separated from the practical clipped loss

A ReDCO decision is a complete code turn or sub-call output (a macro-action). For
on-policy REINFORCE, broadcasting the node advantage over the action's tokens and
summing log-probs is exact, since
log π_θ(a_v|s_v) = Σ_{t∈a_v} log π_θ(y_t|s_v, y_<t). But the practical prime-rl
default is a **token-level** importance-ratio loss with token-level clipping and
token-count normalization; feeding it a constant per-token advantage does *not*
make it a node-level clipped objective. Two losses, as an explicit ablation:

1. **Clean estimator loss** (early, near-on-policy):
   L_v = −Â_v · Σ_{t∈a_v} log π_θ(y_t|·), averaged over decision nodes / branch
   groups, KL regularizer separate. Easiest to reason about; use for the estimator
   validation runs.
2. **Practical clipped node loss**: compare the exact sequence ratio
   ρ_v = exp(Σ_t log π_θ/π_b) (principled, numerically extreme for long actions)
   against the StepPO-style length-normalized geometric mean
   ρ̄_v = exp((1/|a_v|) Σ_t log π_θ/π_b) (stable, changes the surrogate). Registered
   via prime-rl's custom per-sequence loss (§4.2 item 4).

Untargeted nodes keep the trajectory-credit path (default token loss with scalar
advantage) so the incumbent arm is untouched.

**Normalization contract (written before the trainer integration):** the clean
batch loss averages over decision units — one unit per untargeted policy node and
one unit per branch group. Each branch record carries explicit weight
`outer_weight/n` and uses its node-summed log-probability; there is no token-count
normalization in the clean arm. For a targeted child,
`outer_weight = 1/k_g`, so each record has weight `1/(k_g·n)`. Explicit weights
travel in rollout records rather than relying on the trainer's default token
normalization. The practical token and ratio losses are separate ablation arms and
must report their effective weighting.

### 1.7 The critic (Stage F only, with causal hygiene)

**Inputs: pre-action information only.** The critic predicts Q(s_v, a) from the
complete pre-action state S_v (serialized), the candidate action a, and static
structure known before the action. **Original descendants and the original node
output are post-treatment variables** — conditioning on them leaks the original
action's outcome and shortcuts the counterfactual question. The baseline is
V̂_φ(s_v) = E_{a′~π_b}[Q_φ(s_v, a′)], never a value conditioned on the realized
output.

**Training data: real branch tuples only.** Never regress the critic toward its own
predictions (the original spec's step-4 fill-in did exactly this via the p_replay
branch — a self-training bug, removed). Log every real branch tuple from Stage C
onward; before training a parametric critic, test a **retrieval baseline** over
canonicalized state/action signatures.

**Deployment ladder** (each rung gated on held-out replay calibration — ECE-style
uncertainty check plus rank correlation within node types):
1. Replay allocation: choose which nodes get branch groups (pre-action features
   only, so the §1.1 selection rule is preserved).
2. Branch-count allocation decided before observing current rewards.
3. Action proposal (suggesting promising alternatives to include in branch groups).
4. Variance-reduction control variate on real branch estimates.
5. Direct credit substitution — only with cross-fitting or a doubly robust
   correction using known replay propensities and real residuals, and a permanent
   randomized real-replay fraction. A rank correlation of 0.6 alone does not
   justify this rung.

### 1.8 End-to-end training iteration (Stage C, normative pseudocode)

Runs inside a custom prime-rl `Algorithm` subclass (`RedcoAlgorithm`, §4.2 item 3):
the orchestrator drives rollouts through the verifiers RLM environment; ReDCO's
logic lives in `score_group` / `score_rollout` / `assign_advantages`; the trainer
applies the update. Hyperparameters in Appendix C; schemas in Appendix B.

```
for each training iteration:

    freeze exact behavior snapshot θ_t as π_b
    load θ_t into BOTH rollout and branch servers
    # Stage C hard rule: no weight drift and exactly one optimizer step per snapshot

    for each task x (batch B; orchestrator schedules asynchronously):

    # -- 1. Ordinary rollouts (orchestrator + verifiers env; group_size = G) --
        generate G rollouts under π_b
        as each eligible policy node is reached:
            online_target_selector(prefix/static features only)
            if selected, commit v BEFORE sampling a_v and save pre-action state
        for each rollout τ, the env records into the trace:
            exact prompt token IDs at every policy decision
            policy checkpoint ID + behavior logprobs + decoding config
            versioned artifact snapshots; sub-call cache entries
            dataflow / control / call / side-effect / observation / resource edges
            environment + RNG + tool-cache state references
            raw task reward and deployment cost
        R_g = reward(x, τ_g) − λ_phys · deployment_cost(τ_g)

    # -- 2. score_group: trajectory credit for untargeted decisions --
        for every ReDCO arm:
            A_traj_g = R_g − mean(R_j for j != g)       # trajectory RLOO
        stock broadcast arm only:
            A_stock_g = R_g − mean(R_1..G)              # exact prime-rl default

    # -- 3. score_rollout (async): one branch group per rollout --
        for each rollout τ_g:
            v = committed_online_target(τ_g)   # selected before a_v was sampled
            if no eligible target:
                log target_skip
                continue                      # trajectory credit only; no root fallback
            branch actions:
                include original a_v as a_1    # valid: online selection ignored output
                a_2 … a_n ~ π_b(· | o_v)       # identical decoding config;
                                               # seeds differ; n = K+1
                invalid actions execute and receive their true error reward
            seed bundles U_1 … U_c:            # counter-based PRF event keys
                U_1 = recorded post-target continuation namespace
                pre-sample U_2 … U_c before rewards
                share each U_m across actions (CRN)

            for each action a_i, seed U_m:
                (G_{i,m}, R_{i,m}) = PartialReplay_{π_b}(S_v, a_i, U_m)   # §1.3:
                    # restore state before v; inject a_i
                    # re-execute deterministic descendants (cached sub-calls)
                    # downstream policy node:
                    # prompt hash + keyed seed unchanged → reuse recorded action;
                    # either changed → resample from π_b
                    # branch topology may diverge
                R_{i,m} −= λ_phys · logical_deployment_cost_as_if_run_fresh
            Q̂_i = mean_m R_{i,m}
            Â_i = Q̂_i − mean_{j≠i} Q̂_j                       # LOO, all branches

            log all branch tuples (actions, seeds, rewards, selection rule,
                                   sliced-vs-full flag) → analysis + critic data

        # -- 4. assign_advantages: per-token credit --
        for each untargeted policy node t in each τ_g:
            adv[action tokens of t] = A_traj_g                 # η = 0
        for the targeted node v in each τ_g:                   # η = 1: REPLACE
            emit the branch group as training data:
                for each branch action a_i:
                    adv[tokens of a_i] = Â_i
                record_weight[a_i] = outer_weight(v) / n
                # outer_weight = 1 for root; 1/k_g for a child under the
                # incumbent-comparability heuristic
                weight the branch group as ONE decision state
        child rollouts (untargeted): inherit A_traj_g, weighted 1/k_g
            (comparability heuristic — ablate)
        emit full-length per-token advantage list (0.0 off-mask)

    # -- 5. Update (trainer) --
    # estimator-validation runs: clean node loss (§1.6 loss 1, custom loss)
    # production runs: default token loss for untargeted + node-loss ablation arms
    # KL regularizer separate; no entropy games
    take EXACTLY ONE optimizer step on data collected from θ_t
    publish θ_{t+1}; begin the next snapshot cycle
```

Arms and controls: broadcast incumbent = exact stock `grpo` (inclusive group mean,
no step 3); ReDCO inheritance = trajectory RLOO with η ≡ 0 and 1/k_g;
**full-suffix C3 arm** = step 3 with all suffix environment events re-executed
instead of sliced (identical actions, event-keyed seeds, and cached-action reuse —
the decisive comparison); sign-flip = negate Â; placebo = permute Â within branch
groups. All arms are `RedcoConfig` flags on the same algorithm class (§4.3).

---

## 2. Staged implementation

Stages A–F, each with a hard gate. Durations assume one focused person.

### Stage A — Exact incumbent reproduction (2–3 weeks)

- **Pin exact prime-rl and verifiers commits** for the life of the project. Do not
  rely on API or default descriptions from an unpinned `main` (defaults have
  churned: e.g., the current documented default advantage is reward minus group
  mean *without* std normalization; the default RL objective is a token-level
  clipped importance-ratio loss plus KL).
- At pin time, inspect the bring-your-own-algorithms path. If the pinned stack
  supports genuine out-of-tree registration (entry points/import hooks), keep
  ReDCO external. Otherwise maintain a pinned prime-rl fork with one minimal,
  marked patch for the config-union member and registry entry (target: <100 lines);
  document the diff and never use a fragile runtime monkey-patch.
- Run a stock multi-turn example (`uv run rl`) to confirm
  trainer/orchestrator/inference wiring; under the committed Tier-0–2 budget, use
  an official single-GPU config for **Qwen3-1.7B full fine-tuning or Qwen3-4B
  LoRA**, selected according to the pinned commit's known-good examples.
- Build the **no-op `RedcoAlgorithm`** (`score_group` = stock advantage,
  `assign_advantages` = scalar broadcast) and demonstrate statistically identical
  training to stock `grpo`. This stub is the scaffold every arm grows from.
- Port the alphaXiv RLM scaffold + evidence-selection environment into a verifiers
  environment (§4.2 items 1–2) under Tier 0. Under funded Tier 3, reproduce the
  *shape* of their single-paper result (SFT cold start → clear RL lift; their run
  went 0.6 → 0.8 judge score) before using it as Stage D's first real-task
  comparison.

**Gate GA-micro (required before mini Stage C):** no-op algorithm ≡ stock baseline
on an official single-GPU example under pre-registered confidence intervals/seed
counts fixed after the Tier-0 noise-floor measurement. **Gate GA-full (required
before Stage D):** the evidence-selection incumbent recipe is reproduced at 4B
full-FT scale. GA-full is Tier-3 grant-contingent and does not block the $50
replay/estimator go-no-go program.

### Stage B — Replay engine without any training (3–5 weeks)

Build the restricted typed environment (§1.4) with the tracer, versioned artifact
store, snapshot/restore, and the `replay` endpoint (§4.2 item 2). The correctness
oracle is immutable content-addressed artifacts plus deterministic prefix
re-execution from the event log with event-keyed seeds; every faster restore path
is checked against it. Then, against a frozen model, run **thousands of randomized
interventions**, each evaluated twice: full suffix replay and graph-sliced replay,
with identical actions, keyed seeds, and cached-action decisions.

**Gate GB (all required):**
- Zero failures over thousands of interventions under exact bitwise equality of
  terminal artifacts for deterministic branches. For explicitly floating-point
  artifact types, deterministic serialization/canonicalization is part of the
  environment contract rather than a post-hoc tolerance.
- TOST-style equivalence of stochastic reward distributions, with margins and
  sample counts fixed after Tier-0 noise-floor measurement and before training.
- No manually discoverable omitted dependency on a hand-audited sample.
- Exact actual-prompt provenance working (observation edges reflect rendered
  prompts; measure how often the all-to-all fallback fires).
- **Empirical replay amplification factor (RAF)** measured by node type and depth,
  with separate meters for: alternative-action generation tokens, regenerated
  downstream policy tokens, judge calls, CPU time, GPU seconds, wall clock, storage.
  RAF ≈ 1 + M·K·c·f̄ + action-generation fraction; the project's compute plan (§8)
  is recomputed from the measured f̄, not assumed.
- A separate same-prompt/same-seed vLLM reproducibility audit is reported but is
  not counted as sliced-vs-full replay disagreement, because both replay modes use
  cached-action reuse as shared semantics.

Do not proceed to RL while sliced and full replay disagree. If provenance shows
f̄ ≈ 1 (everything affects everything), pivot per §0.

### Stage C — Minimal training result (3–5 weeks)

The smallest configuration that can establish the core scientific result:
- Restricted substrate; deterministic synthetic rewards (credit-probe suite, §5).
- Sub-call-output target nodes only; **one target per rollout**; branch group
  n = 4 (original + 3); c = 1 (deterministic env); pinned π_b; all-branch LOO
  training with replacement (η = 1); no cost penalty in the first stability run;
  no normalization heuristics unless raw scale is unstable.
- Commit targets online before their actions are sampled. A rollout with no
  eligible sub-call target receives trajectory RLOO only; skip and log it, with no
  root-turn fallback. Construct the Stage-C probes so such skips are rare.
- Hard snapshot rule: θ_t is loaded into both rollout and branch servers, collection
  completes without weight drift, and exactly one optimizer step produces θ_{t+1}.
- Arms: broadcast; inheritance; **full-suffix C3 branching**; **graph-sliced ReDCO**
  (identical branches/seeds as the C3 arm); bigger-G GRPO at matched total tokens;
  sign-flip; placebo. Loss ablation: clean node loss vs default token loss vs
  geometric-mean node ratio (§1.6).

**Gate GC** (decisive, on the synthetic suite):
- *Replay equivalence in-loop*: sliced and full-suffix arms pass pre-registered
  equivalence tests for advantages at equal branch counts; learning-curve
  confidence intervals and equivalence margins are fixed before the run.
- *Estimator fidelity*: high sign accuracy and rank correlation of Â against
  enumerated ground-truth Q(s,a) on small finite tasks; **cosine similarity between
  the estimated policy gradient and an exhaustive Monte Carlo gradient** (the most
  informative metric — more than critical-node top-1). Sign accuracy is evaluated
  only where ground-truth |Δ| exceeds the measured noise floor.
- *Learning*: branch-credit arms beat broadcast at matched branch evaluations;
  sliced ReDCO ≥ full-suffix C3 at matched accelerator-hours (this is the graph
  contribution); sign-flip hurts; placebo ≈ baseline.

### Stage D — Real RLM tasks (4–6 weeks)

Evidence selection → OOLONG short-to-long → MRCRv2 anti-shortcut experiment (§5).
Use a pinned local judge with greedy decoding; record judge weights/version, prompt,
decoding configuration, and retry policy as replay state. If an external API judge
is unavoidable, average a fixed k calls chosen before rewards are observed and
treat residual judge noise as exogenous independent noise included in the
continuation budget c.
For learning-efficiency comparisons, **total generated tokens (including branch
generation)** is the primary stopping constraint. For the headline scale
comparison, **wall clock/GPU-hours** is primary. Branch evaluations, judge calls,
the non-primary compute ledger, and storage are reported rather than forced to
equality; Pareto frontiers summarize the tradeoffs. ReDCO's branches consume budget
that baselines may spend on more rollouts.

**Gate GD:** ≥15% fewer training tokens to the incumbent's final reward on ≥2
tasks, or higher long/OOD eval at matched budget; the MRCRv2 experiment reported
either way. This is the **minimum continuation floor**. The ≥1.5× efficiency result
on ≥3 tasks in §6.2 is the aspirational paper headline, not the gate.

### Stage E — Operation structure, general REPL, and explicit objectives (4–8 weeks)

Only after GC/GD:
1. **Structured operations as policy actions**: make PARTITION/FILTER/VERIFY/
   chunk-size/NO_OP/IDENTITY/BYPASS explicit, policy-supported choices. Structural
   alternatives then become ordinary policy-sampled branches — the sound way to
   train graph structure (necessity stays diagnostic; **no necessity bonus**: the
   artificial-necessity exploit in §1.1 makes a positive bonus unsafe, and the
   ordinary global cost penalty already discourages useless computation).
2. **Explicit context-cost objective**: to train context offloading, add a
   transparent resource term −γ·(task-derived tokens included in root prompts),
   with provenance measured by the observation edges; study the success vs
   context-cost Pareto frontier. (Offloading pressure does *not* follow
   automatically from the ReDCO objective — printing into context raises replay
   cost and variance but not necessarily a negative expected gradient. The
   emergent-pressure idea is retained only as an exploratory measurement:
   context-leakage rate over training, ReDCO vs broadcast.)
3. **General-REPL bridge**: strict mode (declared-IO cells, print truncation, type
   allowlists) on the real RLM scaffold; re-run GB's replay-equivalence testing
   there; multiple targets per rollout; sibling splice-and-replay (§1.5 cases 1–2).
4. Level-1 → recursive: branch groups at decision nodes *inside* child rollouts.

**Gate GE:** at least one structural/objective result that improves long/OOD eval,
not just train reward; replay equivalence holds in strict-mode general REPL.

### Stage F — Critic (4–8 weeks; optional for the first paper)

Per §1.7: retrieval baseline first; pre-action inputs only; train on accumulated
real branch tuples (≥50k before starting); held-out tasks *and* graph topologies;
deploy up the ladder (allocation → proposal → control variate → cross-fitted
substitution) with a permanent randomized real-replay fraction.

**Gate GF:** critic-assisted ReDCO ≥ replay-only ReDCO at matched compute. If the
critic never calibrates, ship ReDCO as a replay-only method — Stages C–E stand
alone.

### Phase 4 / scale-up (open-ended, after GD)

Qwen3-30B-A3B-Instruct-2507 with the harness blog's published prime-rl configs
verbatim (same model, same envs, same stack — a config change); the blog's
length/domain-transfer experiments re-run under ReDCO training; free-mode vs
strict-mode comparison; BrowseComp-Plus (tool responses cached at rollout, tool
nodes frozen under intervention). Budget permitting, a ~100B-class MoE
(Qwen3.5-122B-A10B or GLM-4.7-Air-class; GLM-5.2 at 744B and Kimi K3 at 2.8T are
teachers/judges only).

---

## 3. Models

| Role | Recommendation | Why |
|---|---|---|
| Tier 0–1 replay testing | **Qwen3-0.6B/1.7B-class**, quantized OK, served locally or on one community GPU | Replay equivalence needs a stochastic policy, not a smart one; free-to-$10 |
| Tier 2 mini Stage C | **Qwen3-1.7B (full FT) or Qwen3-4B-Instruct-2507 (LoRA)** on one H100 spot | Official prime-rl single-GPU example configs exist for both |
| Stage A–F policy (funded) | **Qwen3-4B-Instruct-2507-class** (whatever has an official pinned prime-rl example config; alphaXiv's recipe used the same scale) | Known-good full-FT on one node; cheap iteration; needs cold-start SFT |
| Scale-up policy | **Qwen3-30B-A3B-Instruct-2507** | Exact model of the harness blog → clean head-to-head vs GRPO broadcast on published envs and configs; MoE with 3B active trainable on 1–2 nodes |
| Stretch policy | Qwen3.5-122B-A10B or GLM-4.7-Air-class | Only if multi-node budget materializes |
| SFT teacher | GLM-5.2, Qwen3.5-397B-A17B, or Kimi K3 API | Frontier-open quality for teacher trajectories; never trained |
| Stage-D reward judge | Pinned open-weight model served locally, greedy decoding | Replayable version/config/retry semantics; external fixed-k judging only if unavoidable |
| Sub-call model | Same policy (shared root/child weights, alphaXiv-style) | One policy, one set of gradients; sub-call branch groups then improve the child directly |
| Critic (Stage F) | Retrieval baseline first; then Qwen ≤4B on serialized pre-action state | §1.7 ladder |

Verify each model choice against the pinned prime-rl commit's example configs
before assuming support. License note: Qwen (Apache-2.0) and GLM (MIT) are both
fine for research artifacts.

---

## 4. Training stack: prime-rl + verifiers

Why this stack: (1) verifiers provides, prebuilt, exactly the environment machinery
an RLM needs — multi-turn rollout loop, isolated env servers, judge rubrics, and
typed traces that natively carry the branch structure of child rollouts; (2)
prime-rl's "bring your own algorithms" extension point accepts full-length per-token
advantage arrays and custom per-sequence losses — precisely ReDCO's credit and loss
shapes; (3) the harness blog's published runs are prime-rl runs, so the incumbent
baseline and the flagship comparison are same-stack, full-fine-tuning, exactly
reproducible; (4) co-located vLLM inference keeps deep recursive rollouts and branch
fan-out fast and supports the pinned-π_b server the estimator requires.

### 4.1 Architecture and launch

An RL run is three cooperating processes, launched together on a single GPU/node
with one command or separately for multi-node:

```bash
uv run rl \
    --trainer      @ configs/redco/train.toml \
    --orchestrator @ configs/redco/orch.toml \
    --inference    @ configs/redco/infer.toml
```

- **Inference** — vLLM server holding the current policy. Always launch via
  `uv run inference` (not `vllm serve`): it adds `/update_weights`,
  `/load_lora_adapter`, `/init_broadcaster`, and exposes OpenAI-compatible routes —
  which is what the environment layer's sampling client targets.
- **Orchestrator** — lightweight CPU process that drives rollouts through
  `verifiers` environments (each env runs as a `vf.EnvServer` sidecar subprocess),
  computes advantages, packs batches, and relays weights trainer → inference. Key
  config: `orchestrator.batch_size`, `orchestrator.group_size` (= G),
  `orchestrator.max_off_policy_steps`, `[[orchestrator.train.env]]` (multi-env
  mixes weighted by `ratio`).
- **Trainer** — FSDP2, any HF model, full fine-tuning; SLURM/Kubernetes for
  multi-node; SFT via `uv run sft` (the cold-start phase).

Tiers 0–2 use synchronous single-GPU alternation: both inference roles load θ_t,
collection finishes, exactly one optimizer step runs, and θ_{t+1} is published.
The asynchronous one-step-ahead pipeline is a later practical-loss optimization,
not part of the Stage-C clean-estimator protocol.

Pin exact commits of both repos (Stage A); all statements below were verified
against the July 2026 docs and must be re-verified against the pinned commit.

### 4.2 Where each ReDCO component lives

1. **The RLM environment → a verifiers environment.** Target the v1 API
   (`verifiers.v1`: taskset / harness / runtime; the RLM scaffold is a *harness* in
   their vocabulary, and Prime Intellect ships an experimental `RLMEnv` to start
   from). The v0 `MultiTurnEnv` API is the fallback: REPL step in
   `env_response(messages, state)`, per-rollout init in `setup_state(state)`,
   termination on `FINAL`/`FINAL_VAR` via `@vf.stop`, teardown via `@vf.cleanup`.
   Rewards are `Rubric` functions (`JudgeRubric` for the LLM-judge recipe). The v1
   `Trace` carries branch structure natively — child rollouts ride along. Store the
   event-graph JSON, artifact version refs, prompt-provenance spans, and sub-call
   cache keys in the rollout state/trace metadata. Note the scaffold's prompt
   builder does context dropping — observation edges must be read from the actual
   rendered prompts (exact token IDs sent to inference), which prime-rl's token-in
   multi-turn machinery exposes; never approximate from reconstructed message text.
2. **Replay and interventions → env-server endpoints.**
   `replay_restore(rollout_ref, before_node)` and
   `replay(rollout_ref, node_id, action, seed_bundle, mode)` run the §1.3 machinery
   (restore, inject, deterministic re-execution with cached sub-calls, prompt-hash
   checks, π_b regeneration where needed) and return the branch's terminal reward,
   cost meters, and divergence markers. `mode ∈ {sliced, full_suffix}` — both modes
   ship, because their agreement is the decisive experiment. The environment and
   replay implementation require no prime-rl core changes.
3. **ReDCO credit → a custom `Algorithm` subclass.** The documented "bring your own
   algorithms" extension point:
   - At pin time, prefer genuine out-of-tree registration if the selected commit
     provides it. Otherwise use the minimal pinned-fork patch from Stage A:
     subclass `Algorithm`, add a typed `RedcoConfig` to
     `prime_rl.configs.algorithm`'s discriminated union, register
     `"redco": RedcoAlgorithm` in `ALGORITHM_CLASSES`, and select with
     `[orchestrator.algo] type = "redco"` plus kwargs (n, c, η policy, target
     selection rule, replay mode, arm flags).
   - **`score_group`**: trajectory RLOO for every ReDCO arm. The exact stock
     broadcast flag alone uses prime-rl's inclusive group mean.
   - **Environment/harness during rollout**: commit at most one target online
     before its action is sampled, using prefix/static metadata only; save the
     pre-action state reference. Log and skip rollouts with no eligible node.
   - **`score_rollout`** (async; may make model calls): evaluate the already
     committed target, sample alternatives against the pinned π_b server, issue
     `replay` calls, compute LOO, and log branch tuples.
   - **`assign_advantages`**: full-length per-token list (`0.0` off-mask) for
     untargeted nodes; for the targeted node, the branch group enters the batch as
     additional samples (each branch action with its Â_i), replacing the original
     action's trajectory-credit contribution (η = 1).
4. **Loss** — untargeted nodes keep the default token-level importance-sampling
   loss with KL. The clean node loss and the two node-level ratio variants (§1.6)
   are registered via `[trainer.loss] type = "custom"` with
   `import_path = "redco.algo.losses...."`; the custom function receives
   `LossInputs` (`trainer_logprobs`, `inference_logprobs`, per-token `advantages`,
   `loss_mask`) per sequence and returns `LossOutputs`.
   The clean loss additionally consumes explicit decision-unit weights: one unit
   per untargeted node and one per branch group, with branch-record weight
   `outer_weight/n`; it never divides the node-summed log-prob by action length.
5. **Cost penalties, two distinct places** — (a) *deployment cost* inside task
   reward: a **fixed physical** penalty (tokens/latency/dollars), identical across
    arms and branch groups; do not use the built-in group-relative length penalty
    for branch rewards, because its normalization changes with group membership.
    This is logical as-if-fresh workflow cost, including normally billed cost for
    reused actions.
    (b) *Training/experiment cost*: never in reward; tracked in the experiment
    ledgers (§2 Stage D), using actual compute spent.
6. **The pinned behavior snapshot π_b** — run a second inference role for branch
   sampling and π_b continuations (the auxiliary frozen-server pattern
   prime-rl's `opd` algorithm uses for its teacher: a `name` + `base_url` entry you
   serve yourself). In Stage C, both rollout and branch roles load the same θ_t;
   collection is synchronous and followed by exactly one update. The clean
   estimator claim depends on it. The "live server, bounded staleness via
   `max_off_policy_steps`" approximation is a later engineering option, introduced
   only with measurement (watch `mismatch_kl/all/mean`). On one node, the pinned
   server for a 4B model shares GPUs with the policy server; budget memory
   accordingly.
7. **The critic (Stage F)** — retrieval service first; then a small SFT/regression
   job (`uv run sft` or a plain trainer job) on the branch-tuple dataset, served as
   another auxiliary model server.
8. **Async pipeline discipline (practical-loss arms only)** — the orchestrator runs one step ahead of the
   trainer, and `score_rollout` branch evaluation adds latency to rollout
   finalization. Keep branch time per group below trainer step time or the pipeline
   stalls; levers: one target per rollout (Stage C default), cheap-node targeting,
   parallel replay calls, a widened env-worker pool.

### 4.3 Arms, budget accounting, and hardware

**Arms as `RedcoConfig` flags** on one algorithm class — same environment, same
loss defaults, same fixed cost penalty: exact stock broadcast (inclusive group
mean, no-op path); ReDCO inheritance (trajectory RLOO, η ≡ 0, 1/k_g); full-suffix
C3 branching (`mode = full_suffix`);
graph-sliced ReDCO (`mode = sliced`); bigger-G GRPO; sign-flip; placebo; loss
variants. One TOML per arm in `configs/`.

**Budget accounting:** ledgers per arm — branch evaluations, total generated
tokens, judge calls, GPU-hours, wall clock, and storage — collected from the
inference servers' metrics and the env servers' meters. Every experiment names one
primary stopping ledger before launch: generated tokens for learning efficiency,
wall clock/GPU-hours for the headline matched-compute result. Other ledgers are
reported, not forced to equality; the paper includes Pareto frontiers.

**Hardware:** starts at a single GPU and scales with funding (see §8 budget
tiers). Micro-scale (Tiers 1–2): one community 4090/A100 for Stage B inference,
one H100 spot for Stage A wiring and mini Stage C at 1.7B–4B with LoRA and
synchronous branch-generation/training alternation. Funded (Tier 3): one 8×H100/
H200 node covers Stages A–F at 4B full-FT (alphaXiv precedent: one node, batch
16 × group 8, up to 512 concurrent rollouts — plus the pinned π_b server) and
Qwen3-30B-A3B at scale-up (harness blog precedent: 150-step runs, batch 64,
4 rollouts on 8×H100 nodes; adopt their published configs verbatim).

SkyRL (the alphaXiv release) is a read-only reference for the RLM scaffold
semantics and their evidence-selection env — port its environment logic into the
verifiers environment rather than adopting a second training stack.

---

## 5. Task ladder and synthetic suite

Ordered by diagnostic value per dollar. Design principle: start where ground truth
is *known by construction*, so estimator fidelity is measured before any end-to-end
claim.

1. **Synthetic credit-probe suite (build first; ~1–2 weeks).** Small enough to
   enumerate all actions and estimate high-budget ground-truth Q(s,a):
   - *Planted needle*: context split into k chunks, one contains the answer; the
     sub-call on that chunk is the critical node.
   - *Poisoned filter*: a predicate either keeps or destroys the needle.
   - *Decomposition-mandatory*: context exceeds any single sub-call's window.
   - *Redundancy*: either of two nodes suffices (each node's marginal effect ≈ 0
     despite joint necessity — LOO must reflect this).
   - *Synergy*: two individually weak nodes matter only together.
   - *Spurious correlation*: a node correlates with success, no causal effect.
   - *Luck*: a stochastic node looks decisive without changing reward probability.
   - *Context leakage*: an artifact matters only through a printed prompt span
     (tests observation edges).
   - *Mutable aliasing* (strict-mode general REPL only): a node mutates an object
     without rebinding (tests side-effect edges).
   - *Control flow*: an exception/predicate changes which later nodes execute.
   - *Dynamic topology*: an alternative action spawns a different number of calls.
   - *Cost-only effect*: output unused, but computation affects reward via cost.

   **Metrics**: advantage MSE, sign accuracy, rank correlation, critical-node
   top-k, **policy-gradient cosine vs exhaustive Monte Carlo gradient**, replay
   equivalence (sliced vs full), replay amplification factor.
2. **Evidence selection (alphaXiv env, public code).** Known-good RL recipe, pinned
   local greedy rubric judge, real sub-RLM calls. First real task.
3. **OOLONG [trec-coarse] short→long split.** Harness blog suite; rich
   multi-sub-call graphs; length-generalization eval built in.
4. **MRCRv2 (2-needle → 8-needle/2M).** Where the harness blog documents the
   degenerate single-sub-call shortcut — the Stage-D/E flagship testbed.
5. **GraphWalks, Ada-LEval, OOLONG-Pairs.** Round out the harness blog suite for
   the scale-up head-to-head.
6. **BrowseComp-Plus.** Real retrieval tools; graphs with non-replayable external
   calls (cache tool responses at rollout; tool nodes frozen under intervention).

Avoid at first: SWE-bench-style coding agents (environment state too entangled),
anything requiring OCR/network at replay time, and judge-only rewards until the
pinned local judge's variance is characterized. If an external API judge is ever
unavoidable, average a fixed k calls and treat residual noise as independent
exogenous variance; judge calls remain a budget ledger of their own.

---

## 6. Experiments, metrics, and end goals

### 6.1 The factorial design (isolates each contribution)

| Question | Required comparison |
|---|---|
| Does local counterfactual credit help? | Exact-stock broadcast vs full-suffix C3-style branching; primary ledger = total generated tokens |
| Does the dataflow graph preserve correctness? | Full-suffix branch reward vs graph-sliced branch reward, identical actions and seeds |
| Does the graph save compute? | Full-suffix C3 vs graph-sliced ReDCO at identical branch count |
| Do the savings improve learning? | Full-suffix C3 vs sliced ReDCO; primary ledger = generated tokens at micro scale, GPU-hours/wall clock for the scale headline |
| Does branch reuse matter? | Original-action-only update vs all-branch LOO update |
| Does node-aligned optimization matter? | Default token loss vs node-level objective (both ratio variants) |
| Does explicit structure shaping help? | Diagnostics-only vs explicit structural policy actions (Stage E, after Stage C succeeds) |

The strongest single validation:

> For exactly the same intervention, action samples, and exogenous seeds, full
> replay and sliced replay produce the same terminal reward and materially
> identical graph-visible artifacts. Any discrepancy is a replay bug.

### 6.2 Primary metrics

1. **Replay equivalence** (Stage B/C gate; §6.1 row 2).
2. **Compute savings**: measured RAF and per-branch cost, sliced vs full-suffix, by
   node type/depth.
3. **Sample efficiency under a predeclared primary ledger**: total generated tokens
   for learning-efficiency comparisons; GPU-hours/wall clock for the headline
   scale comparison. Branch evaluations, judge calls, storage, and non-primary
   compute ledgers are reported alongside Pareto frontiers. Aspirational paper
   target: ≥1.5× on ≥3 tasks; GD's ≥15% on ≥2 tasks is the continuation floor.
4. **Long/OOD eval lift** at matched budget (harness blog's generalization metric),
   with the gap widening on degenerate-strategy tasks (MRCRv2).
5. **Inference-cost reduction** per solved eval task after training.

### 6.3 Secondary metrics

6. Estimator fidelity on synthetics: gradient cosine, advantage MSE, sign accuracy,
   rank correlation (§5).
7. Sign-flip and placebo controls behave as predicted.
8. Gradient variance vs broadcast.
9. Context-leakage rate and decomposition-rate trajectories (exploratory
   measurement; the offloading *objective* is Stage E's explicit −γ term).
10. Critic calibration (Stage F): ECE, rank correlation, ladder rung achieved,
    randomized real-replay fraction maintained.

### 6.4 End-goal claims (worded to survive review)

- *Minimum viable (Stage C–D):* behavior-policy counterfactual branching at RLM
  turn/sub-call boundaries, with graph-sliced partial replay, matches full-suffix
  branching's credit at a measured fraction of its cost, and beats
  trajectory-broadcast RL on matched budgets — **the first graph-sound
  counterfactual credit method for recursive executable agents.**
- *Strong (Stage E):* explicit structural policy actions trained with branch credit
  steer RLMs away from degenerate, non-generalizing strategies without hand-written
  decomposition hints; an explicit context-cost objective yields a favorable
  success/offloading Pareto frontier.
- *Full (Stage F + scale):* a causally-hygienic critic allocates replay budget and
  reduces variance at scale, with replay-anchored calibration.

Do **not** claim: "first fine-grained credit assignment for agent workflows"
(false — C3/StepPO/RTMC/TRACE/TEMPO/Agent Lightning), automatic context-offloading
pressure (unproven), operation-level or fully recursive credit (Level 1 only, until
Stage E), or unbiasedness beyond the §1.1 conditions.

### 6.5 Kill criteria

- GB unmet after 6 weeks of replay-engine work → the replay substrate is not viable;
  pivot to full-suffix C3-style turn-level credit for RLMs (simpler paper).
- Provenance shows f̄ ≈ 1 (no slicing savings) → same pivot.
- Bigger-G GRPO matches branch credit everywhere under the same predeclared primary
  budget ledger → the signal isn't worth targeting; publish the negative result.
- Gradient cosine on synthetics ≈ 0 → estimator broken; stop and debug.

---

## 7. Risk register

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | Replay unsound for general Python (aliasing, side effects, hidden state) | High | Restricted typed substrate for Stages B–D (§1.4); declared-IO PURE_PYTHON; strict mode for the general-REPL bridge; GB equivalence gate |
| 2 | Prompt-provenance edges unavailable → all-to-all fallback collapses the graph | Medium-high | Read exact rendered token IDs from the inference path; measure fallback rate in GB; pivot per §6.5 if f̄ ≈ 1 |
| 3 | Sub-LM nondeterminism breaks replay parity | High | Sub-call cache keyed by (prompt-hash, model-version, event seed); counter-based PRF keys; both replay modes share cached-action semantics; separate vLLM reproducibility audit |
| 4 | Branch rewards too noisy | High | n = 4 branches, c ≥ 2 when stochastic, shared seeds; CIs reported; fixed allocation (no adaptive n/c in Stage C) |
| 5 | Branch budget starves rollout budget | Medium-high | Predeclare generated tokens or GPU-hours/wall clock as the primary ledger; report the others; one target per rollout; bigger-G null arm |
| 6 | Biased advantage from mixing necessity into branch credit | Fixed by design | Two estimands (§1.1); necessity diagnostics-only; no shaping bonus (artificial-necessity exploit) |
| 7 | Selection conditioning breaks the branch-group distribution | Medium | Commit online before sampling the target action using prefix/static features; otherwise exclude original and sample all n fresh (§1.1, §1.5) |
| 8 | Invalid-action handling distorts the action distribution | Medium | Execute-and-score error semantics (or identical constrained decoding at rollout and branch time); never resample/skip |
| 9 | Additive global+local double-weighting | Fixed by design | η-blend with η=1 replacement at targeted nodes (§1.2) |
| 10 | Token-level clipped loss/normalization ≠ node-level objective | Medium | Explicit decision-unit weights and node-summed log-probs in the clean arm; sequence-ratio vs geometric-mean ablation (§1.6) |
| 11 | Entropy collapse from SFT or sharp local credit | Medium | Tiny SFT set (alphaXiv lesson); entropy monitoring; raw-unit advantages; heuristics only if unstable |
| 12 | Reward hacking or nondeterminism from rubric judges | Medium | Pinned local greedy judge with full replay-state metadata; spot audits; fixed-k API averaging only if unavoidable; judge-call ledger |
| 13 | Policy games structure credit (artificial necessity, fake dependencies) | Medium (Stage E) | No necessity bonus; structure trained only via explicit policy actions with branch credit; graph audits |
| 14 | Race conditions with parallel child rollouts | Known (alphaXiv hit it) | Their fixes are public; stress-test in Stage B |
| 15 | Two inference roles create memory pressure on one GPU | Medium | Tiers 0–2 synchronously alternate collection and one-step training with both roles on θ_t; funded 4B runs share GPUs or use a second role/server |
| 16 | Async pipeline stalls from branch latency in practical-loss runs | Medium | Stage C clean arm is synchronous; later async arms use one target, parallel replay, widened env-worker pool, and step-ahead monitoring |
| 17 | MoE instability at 30B-A3B scale-up | Medium | Harness blog configs verbatim |
| 18 | Degenerate strategy converges before credit signal acts | Medium | Branch credit from step 0; MRCRv2 experiment targets this |
| 19 | Tracer misses dependencies in strict-mode general REPL | Medium | Runtime instrumentation backstop; hand audits; aliasing/control-flow probes in the synthetic suite; GB re-run at Stage E |
| 20 | External tools (BrowseComp) not replayable | Certain (scale-up) | Cache tool responses at rollout; tool nodes frozen under intervention |
| 21 | RAF far worse than hoped → compute plan invalid | Medium | GB measures RAF before any training; §8 recomputed from measured f̄; pivot path defined |
| 22 | verifiers/prime-rl API churn (v1 is a preview; defaults evolve) | Medium | Pin commits for the project's life (Stage A); isolate env code behind Appendix-B interfaces |
| 23 | Restricted-substrate results fail to transfer to the general REPL | Medium | Stated as an explicit limitation; Stage E bridge with re-gated replay equivalence; the scale-up head-to-head runs on the real scaffold |
| 24 | Critic post-treatment leakage or self-training | Fixed by design | Pre-action inputs only; real branch tuples only; retrieval baseline; ladder with permanent real-replay fraction (§1.7) |

---

## 8. Compute, budget tiers, and timeline

The budget is tiered, and the tiers align with the stage gates: each tier's results
are the evidence that justifies (or kills) spending the next one. Reference prices
on Prime Intellect (July 2026): H100 spot ≈ $0.94/hr, H100 on-demand ≈ $2.43/hr,
community 4090s ≈ $0.30/hr, A100s ≈ $0.87–1/hr (check live with
`prime-intellect gpu-availability`; per-second billing; community cloud is the
cheap tier; always spot + aggressive checkpointing).

**Replay overhead is measured, not assumed.** The replay amplification factor is
RAF ≈ 1 + M·K·c·f̄ + (action-generation fraction), where f̄ is the average fraction
of original LM/environment work a branch actually re-does. With M=1, n=4, c=1, a
30–50% overhead requires f̄ ≈ 0.1–0.17; with c=2, f̄ ≈ 0.05–0.08. Stage B's GB gate
measures this before any training money is spent.

### Tier 0 — $0 (weeks of the highest-value work)

Everything in Stages A–B that is engineering, not compute: pin the repos; build the
typed environment, event tracer, versioned artifact store, both replay modes, and
the determinism harness; build the synthetic credit probes with enumerable
ground-truth Q(s,a). For Stage B's frozen policy, replay-equivalence testing does
not need a *smart* model — it needs a *stochastic* one. A small quantized model
(Qwen 0.6B–1.7B class) served locally on your own machine (llama.cpp/ollama, or
vLLM if you have any GPU) is sufficient to run randomized interventions and verify
sliced ≡ full-suffix replay end to end.

### Tier 1 — ~$10–15 (Stage B at speed + RAF numbers)

One community 4090 (~$0.30/hr) or A100 spot running vLLM with a 1.7B-class model:
thousands of randomized interventions, sliced-vs-full equivalence at scale, RAF and
provenance-fallback measurement by node type and depth. 30–40 GPU-hours ≈ $10–12.
**This produces the project's go/no-go evidence for about ten dollars.**

### Tier 2 — ~$25–40 (Stage A wiring + mini Stage C)

prime-rl has official single-GPU examples (their Alphabet Sort recipe trains
Qwen3-4B-Instruct-2507 via LoRA on one H100 in ~1 hour; Wordle trains Qwen3-1.7B on
2–4 GPUs in hours). So on **one H100 spot** (~$0.94/hr):
- stock example + no-op `RedcoAlgorithm` equivalence run (~2–4 hrs);
- mini Stage C on the synthetic suite with a 1.7B–4B model, **arms trimmed to the
  three decisive ones** — broadcast, full-suffix C3 branching, sliced ReDCO with
  identical branches/seeds — at ~3–5 hrs per arm.
Total ≈ 15–25 H100-spot-hours ≈ $15–25. Micro-scale compromises, all acceptable
here and noted as such: LoRA instead of full FT (internal comparisons stay clean —
all arms share the identical LoRA config; full-FT comparability only matters at the
flagship scale-up), synchronous alternation of branch-generation and training
phases instead of the async pipeline (removes the pinned-π_b memory pressure on one
GPU: generate all branches against frozen weights, then train), and gate GC scored
on gradient-cosine/sign-accuracy fidelity rather than long learning curves.

**A $50 preload covers Tier 0 + Tier 1 + most of Tier 2.** That is enough to know
whether ReDCO works.

### Tier 3 — funded (Stages D–F and scale-up; ~$15–35k or granted compute)

The full-scale table below applies only after GB/GC pass — at which point the
replay-equivalence result, measured RAF, and micro-scale learning curves are
precisely the preliminary-results section of a compute-grant application. Concrete
avenues, in order of fit: **Laude Institute Slingshots** (funded the harness blog's
8×H100 nodes — exactly this research area), Prime Intellect research
credits/Environments Hub visibility (publishing the RLM ReDCO environment on the
Hub is free marketing to the team whose stack this runs on), Modal/HuggingFace/
Google TRC academic and OSS credit programs.

| Stage | Duration | Compute (funded) | Notes |
|---|---|---|---|
| A | 2–3 wk | Tier 2 covers it | Pinning, no-op algorithm, incumbent reproduction |
| B | 3–5 wk | Tier 0–1 covers it | Replay engine + equivalence + RAF |
| C | 3–5 wk | Tier 2 (mini) → ~1–2 node-weeks (full) | Full arm matrix + real learning curves |
| D | 4–6 wk | ~2–3 node-weeks | Three real tasks; generated-token primary ledger, all others reported |
| E | 4–8 wk | ~2 node-weeks | Structured ops, context objective, strict-mode REPL |
| F | 4–8 wk | ~1–2 node-weeks | Critic (optional for first paper) |
| Scale-up | 8+ wk | 4–8 node-weeks | 30B-A3B, harness blog configs |

Total to the strong claim: roughly 5–7 months of one focused person; under $50
out-of-pocket to the go/no-go gate, then ~$15–35k of node rental *or granted
compute* for the funded tier.

---

## 9. Related-work positioning (write this into the paper from day one)

The safest novelty statement:

> **Graph-sound behavior-policy counterfactual branching for recursive executable
> agents, using dynamic dependency slicing to reduce replay cost and reusing every
> branch for node-aligned policy optimization.**

- **C3 (arXiv 2603.06859)** — the closest method and the statistical backbone:
  frozen behavior policy, fixed-state action branching, behavior-policy
  continuations, LOO baselines, training on all sampled action–advantage pairs,
  unbiasedness at the snapshot. ReDCO adopts this estimator; its delta is the
  executable-graph setting (REPL state, sub-calls, artifacts), dependency-sliced
  replay in place of full-suffix replay, and the recursive/structural extensions.
  C3's exactness argument relies on the transcript fully representing state — an
  assumption RLMs violate, which is precisely what the event-graph machinery
  addresses. Adopt their proof technique and common-random-numbers appendix.
- **StepPO (arXiv 2604.18401)** — step-level MDP alignment and the
  length-normalized (geometric-mean) step importance ratio; the direct reference
  for §1.6's node-level loss ablation.
- **RTMC (arXiv 2604.11037)** — rollout trees via state/action signatures,
  critic-free step values; a grouping-based (non-interventional) neighbor.
- **TRACE (arXiv 2607.13988)** — turn-level rewards from reference-model state
  values and temporal differences; concurrent turn-level credit work to cite.
- **TEMPO (arXiv 2509.18314)** — prefix trees with branch-gated value corrections.
- **GiGPO / Turn-PPO / Tree-GRPO** — grouping/tree-based step or turn credit for
  linear agent loops; GiGPO-style anchor grouping is a baseline arm.
- **CRAFT (arXiv 2606.29476)** — sibling-rollout reuse, explicitly framed as a
  group-level state-marginal estimator; the reason §1.5 does *not* transplant
  sibling rewards.
- **Agent Lightning (arXiv 2508.03680)** — general agent-execution-to-MDP framing
  with hierarchical credit; adjacent infrastructure framing to acknowledge.
- **Trace/OPTO (arXiv 2406.16218) and LLM-AutoDiff** — execution-trace /
  computation-graph optimization of multi-component workflows. They optimize
  general workflows via trace feedback rather than learning a policy through
  behavior-policy counterfactuals, but they own part of the "graph of an LLM
  workflow" idea — cite to avoid overclaiming the graph concept itself.
- **Counterfactual importance weighting & Critical Token Fine-Tuning** —
  token-level necessity via masking/forward passes; complementary granularity;
  their sign-flip validation protocol is reused.
- **Failure attribution (Who&When, arXiv 2505.00212; causal MAS debugging,
  arXiv 2509.08682)** — the diagnostic side of necessity credit; their low
  step-level accuracy without replay access underscores what executable replay
  provides.
- **Reinforcing RLMs (alphaXiv/SkyRL)** — the incumbent recipe and the explicit
  statement of the gap; environment code seeds the Stage-A port; their recipe is
  the baseline arm.
- **RLM paper + harness blog (Zhang et al.)** — the substrate and the
  generalization phenomenon; the scale-up replicates their transfer experiments on
  the same prime-rl stack.

---

## 10. Immediate next actions (first two weeks — all Tier 0, $0)

1. Pin prime-rl and verifiers commits locally (`uv sync`; no GPU needed to read
    code and write against the pinned APIs). Read
    `docs/bring-your-own-algorithms.md` and the Algorithms doc *at the pinned
    commit*; choose out-of-tree registration if genuinely supported, otherwise
    create and document the minimal pinned-fork patch. Write (don't yet run) the
    no-op `RedcoAlgorithm`.
2. Write the executable contracts before implementation: online pre-action target
   commitment; counter-based event seed addresses; decision-unit loss weights;
   logical deployment cost vs actual evaluation cost; synchronous one-step snapshot
   lifecycle; and the post-noise-floor gate-registration template.
3. Start the restricted typed environment (§1.4) and the event tracer (offline, no
    training needed): typed commands, versioned artifacts, prompt-provenance
    recording. Visualize ten traces by hand.
4. Build the first synthetic credit probes (planted needle, redundancy, spurious
    correlation) with enumerable ground-truth Q(s,a).
5. Implement deterministic prefix re-execution as the correctness oracle, then the
   sub-call cache and the two replay modes
    (`sliced`, `full_suffix`); begin Stage B's randomized-intervention equivalence
   testing against a frozen local endpoint (small quantized model via
   llama.cpp/ollama, or vLLM on any local GPU); start measuring RAF and the
   provenance-fallback rate.
6. When local iteration gets slow, spend Tier 1 (~$10): one community 4090 or A100
   spot on Prime Intellect serving a 1.7B model with vLLM, and run the Stage B
   equivalence + RAF campaign at scale.
7. Only after the GB gate passes, spend Tier 2 (~$25): single H100 spot, no-op
   algorithm equivalence run, then the three-arm mini Stage C (§8 Tier 2).
   Porting the alphaXiv evidence-selection environment into verifiers (§4.2
   item 1) stays Tier 0 — write it now, run it funded.

---

## Appendix A. The original ReDCO spec and the full correction history

**Original idea (condensed).** ReDCO (Recursive Dataflow Credit Optimization)
represents an agent rollout as a directed graph of operation nodes (filter,
partition, code execution, recursive LM call, verify, aggregate) and artifact nodes
(context slices, variables, sub-call outputs), with edges recording produce/consume
relations. After computing a global task reward, ReDCO estimates each node's
marginal contribution by intervening on it — replacing a sub-call output with
another sample, an abstention, or a different decoding config; swapping a filter
for identity; removing a verifier; changing a chunk size — and replaying only the
affected descendants from cached artifacts:

    Δ_v = R(G) − E_ṽ[R(G^{v←ṽ})]

Per-node advantages combine global and local signal, A_v = α·A_global +
β·Normalize(Δ_v) − λ·C_v, and weight the log-probability of node-generating
actions. A graph-structured critic Q_φ(task, graph, node, intervention) →
(reward, uncertainty) is trained on real replays and progressively substitutes for
them; a priority rule concentrates real interventions on important,
poorly-estimated nodes. Credit is split into structural (should this node/edge
exist) and execution (was this node's output/config good).

**Corrections applied in this plan, and why:**

| # | Original / earlier draft | Changed to | Reason |
|---|---|---|---|
| 1 | One Δ_v from any type-valid alternative feeds the advantage | Two estimands: branch-group LOO credit (policy-resampled only → advantage) and N_I(v) (degenerate interventions → diagnostics only) | Necessity ≠ relative action quality; mixing them biases the gradient (§1.1) |
| 2 | Alternatives may vary seed *or temperature* | Only the seed varies; identical decoding config | Different temperatures are not samples from π_b (§1.1 condition 1) |
| 3 | Invalid alternatives: "validate type, else resample/skip" | Execute invalid actions and score their true error reward (or constrain decoding identically everywhere) | Silent rejection conditions the action distribution on validity (§1.1 condition 2) |
| 4 | Alternatives used only as a baseline for the original action | All n = K+1 branches train, with symmetric LOO advantages, branch group weighted as one decision state | C3 trains on every pair; discarding branches wastes the replay budget (§1.1) |
| 5 | Targets selectable by predicted effect/uncertainty or realized replay cost on the completed rollout | Commit online before the action using prefix/static features; if post-action data is ever used, exclude the original and sample all n fresh | Output/descendant-conditioned selection breaks the unconditional-sample property (§1.1) |
| 6 | Add global+local credit; later, mix stock inclusive-mean trajectory credit with branch LOO | Convex replacement/blend, with trajectory RLOO in every ReDCO arm and exact stock scaling isolated to the incumbent arm | Addition double-weights targets; inclusive-mean GRPO is `(G−1)/G`-scaled relative to LOO (§1.2) |
| 7 | Node-type z-scoring and clipping as part of the method | Raw reward units; normalization/clipping labeled optimizer heuristics, ablated | Keeps the causal estimator interpretable (§1.2) |
| 8 | 1/k_g child weighting described as correct; targeted replacement weight unspecified | Kept as comparability default, labeled a policy-dependent heuristic, ablated; targeted child branch group inherits the same outer 1/k_g | k_g is policy-dependent, but applying it inconsistently would additionally reweight selected children (§1.2) |
| 9 | Dataflow edges only; later dataflow+context+call | Six edge types: dataflow, control, call, side-effect/ordering, observation (prompt provenance), resource; SSA-versioned artifacts; event-DAG semantics | Sound descendant replay needs control/side-effect/observation closure (§1.3) |
| 10 | Context edges = "anything printed reaches every later turn" | Exact prompt-token provenance (actual inclusion after context dropping); all-to-all only as measured fallback | All-to-all collapses the graph and destroys slicing value (§1.3 edge 5) |
| 11 | Replay = "re-execute recorded code; regenerate when context edges exist" | Event-keyed PRF seeds; both modes reuse a downstream action iff observation and seed are unchanged; otherwise resample from π_b; branch topology may diverge | Sequential RNG streams break under topology changes; shared reuse semantics isolate slicing correctness from sampler determinism (§1.3) |
| 12 | Arbitrary Python REPL from day one | Restricted typed dataflow substrate for Stages B–D; declared-IO PURE_PYTHON; general REPL bridged in Stage E | Mutable Python defeats AST+namespace tracing; the replay theorem needs the restriction (§1.4) |
| 13 | Sibling outcomes recorded as free counterfactual rewards | Sibling three-case rule: exact-state actions (still replayed), splice-and-replay, or proposals/critic-data only; **no reward transplantation** | R(G_sibling) ≠ R(G^{v←a_sibling}); CRAFT's estimator is state-marginal, not fixed-state (§1.5) |
| 14 | λ·cost subtracted globally and per node; later, group-relative length penalty | Logical as-if-fresh deployment cost in reward once; actual replay/training cost only in ledgers | Replay caching must not change the reward; group-relative normalization is incoherent (§1.1, §4.2 item 5) |
| 15 | Live-server π_b approximation as the default | Stage C synchronously loads θ_t into rollout and branch roles, collects without drift, and takes exactly one optimizer step | The clean unbiasedness claim requires an exact snapshot and near-on-policy update (§1.8, §4.2 item 6) |
| 16 | Constant per-token advantage into the default token loss, unexamined | Clean decision-unit loss with explicit `outer_weight/n`, no token-count normalization; ratio losses are ablations | Token-level clipping/normalization ≠ macro-action objective (§1.6, StepPO) |
| 17 | Critic sees original graph incl. descendants; p_replay branch regresses critic toward its own predictions; rank-corr 0.6 gates substitution | Pre-action state + candidate action only; real branch tuples only; retrieval baseline; allocation-first ladder; substitution only with cross-fitting/DR | Post-treatment leakage; self-training bug; premature substitution (§1.7) |
| 18 | Phase-2 necessity bonus for decomposition nodes; claimed emergent context-offloading pressure | No necessity bonus (artificial-necessity exploit); structure trained via explicit policy actions; offloading via explicit −γ context-cost term + Pareto study | High necessity is manufacturable; the offloading pressure does not follow from the objective (§1.1, §2 Stage E) |
| 19 | 30–50% replay overhead assumed | Empirical RAF gate in Stage B; compute plan recomputed from measured f̄ | The assumption implied f̄ ≈ 3–6%, unverified (§8) |
| 20 | Claim: "first fine-grained credit method for recursive harnesses"; operation-level, recursive | Narrowed: graph-sound counterfactual branching for recursive executable agents; Level-1 (turn/sub-call) scope; full-suffix C3 as the decisive baseline | C3/StepPO/RTMC/TRACE/TEMPO/Agent Lightning exist; the graph's contribution is replay efficiency and must be isolated (§0, §6.1, §9) |
| 21 | Match branch evaluations, tokens, judge calls, GPU-hours, and wall clock simultaneously | Predeclare one primary stopping ledger per comparison; report the others and Pareto frontiers | The five constraints are generally impossible to equalize, and broadcast has zero branch evaluations (§4.3, §6) |
| 22 | Qualitative gates such as "statistically identical" | Gate forms fixed now; numerical margins and N fixed after Tier-0 noise-floor measurement and before training | Equivalence requires predeclared tolerances without inventing arbitrary pre-noise numbers (§2) |

## Appendix B. Repo layout, interfaces, and data schemas

Suggested structure (Python, uv-managed). The `env/` package is a verifiers
environment (installable, EnvServer-hostable); the `algo/` package is the prime-rl
extension:

```
redco/
  env/
    rlm_env.py          # verifiers environment: typed-command harness (Stage B–D)
                        # and strict-mode REPL scaffold (Stage E+); rlm_query(_batched),
                        # FINAL/FINAL_VAR, per-turn prompt re-appending;
                        # replay_restore / replay endpoints (sliced | full_suffix)
    commands.py         # PARTITION/CALL/FILTER/AGGREGATE/VERIFY/FINAL/PURE_PYTHON
    repl.py             # sandboxed execution worker (subprocess), declared-IO cells
    tracer.py           # event-DAG builder: six edge types, SSA artifact versioning,
                        # prompt-token provenance recording
    artifacts.py        # content-addressed versioned artifact store, sub-call cache
    replay.py           # restore, prompt-hash checks, dynamic-slice execution,
                        # full-suffix mode, determinism harness, RAF meters
    rubrics.py          # reward functions incl. JudgeRubric wiring; fixed physical
                        # cost penalty
    tasks/              # credit_probes/, evidence_selection/, oolong/, mrcr/, ...
  algo/
    redco_algorithm.py  # prime-rl Algorithm subclass: score_group, score_rollout,
                        # assign_advantages (§1.8 / §4.2 item 3)
    config.py           # RedcoConfig (n, c, η policy, selection rule, replay mode,
                        # arm flags)
    branching.py        # branch-group construction, LOO, keyed CRN bundles,
                        # online target commitments (prefix/static features only)
    losses.py           # clean node loss; sequence-ratio and geometric-mean
                        # clipped node losses (custom [trainer.loss])
    critic/             # Stage F: retrieval baseline, then parametric
  sampling.py           # thin OpenAI-compatible client; `checkpoint` arg selects
                        # live policy server vs pinned π_b server
  configs/              # train.toml / orch.toml / infer.toml per arm; pinned-commit
                        # lockfile; every run reproducible from config
  analysis/             # replay-equivalence reports, RAF dashboards, gradient-cosine
                        # eval, leakage rate, graph visualizer
```

**Policy-node record (per decision, in the verifiers trace/state):** `node_id`,
`kind` ∈ {root_turn, subcall_output}, `prompt_token_ids_hash` (+ provenance spans:
list of (artifact_id, version, token_span)), `action_token_span`,
`behavior_logprobs_ref`, `checkpoint_id`, `decoding_config_hash`,
`state_snapshot_ref`, `event_seed_key`, `target_commitment` (eligible/committed/
skipped, prefix feature vector, commitment timestamp before action sampling).

**Event-graph record:** `nodes[]` (policy | operation | artifact | reward_resource;
artifacts carry `version`), `edges[]` (`src`, `dst`,
`kind` ∈ {dataflow, control, call, side_effect, observation, resource}, `via`).

**Branch tuple (analysis + critic dataset, JSONL):** `task_id`, `rollout_id`,
`node_id`, `selection_rule`, `action_source` ∈ {original, fresh, sibling_splice},
`action_ref`, `seed_bundle_id` (+ PRF master/key schema), `replay_mode` ∈
{sliced, full_suffix}, `reward`, `logical_deployment_cost`,
`actual_eval_cost_meters` (tokens, judge calls, CPU, GPU-s, wall clock, storage),
`decision_unit_weight`,
`divergence_markers` (first changed prompt hash, topology delta),
`checkpoint_id`. Never contains predicted rewards.

**Sampling interface:** `sample(prompt_token_ids, *, event_seed, max_tokens, checkpoint)
→ tokens, logprobs` — decoding config comes from the pinned run config, not
per-call arguments (enforces §1.1 condition 1); `checkpoint` selects the live
policy vs the pinned π_b server. `event_seed` is derived from the structural PRF
key specified in §1.1, never from a mutable sequential stream or prompt content.

## Appendix C. Hyperparameter defaults (Stage C starting points)

| Param | Default | Notes |
|---|---|---|
| Model | Tier 2: Qwen3-1.7B full-FT or Qwen3-4B LoRA; funded: Qwen3-4B full-FT | Choose only from official configs at the pinned commit |
| Batch B × group G | 16 × 8 (`orchestrator.batch_size` × `group_size`) | alphaXiv used 16 × 8 |
| Targets per rollout | 1 | Commit online before action; sub-call outputs only; no eligible target → skip and log |
| Branch group n / continuations c | 4 (original + 3) / 1 deterministic, 2 stochastic | Fixed before rewards; counter-based event-keyed CRN bundles shared across branches |
| η at targeted nodes | 1 (replacement) | Blended η only later, with measured variances |
| Trajectory advantage | ReDCO arms: RLOO; stock arm: inclusive group mean | Keeps local and trajectory LOO scales aligned while preserving exact incumbent |
| λ_phys (deployment cost) | 0 first; then fixed physical rate on logical as-if-fresh cost | Identical across arms/groups; actual replay cost stays in ledgers |
| Loss | Clean decision-unit loss with explicit weights; token/ratio variants as ablations | No token-count normalization in clean arm (§1.6) |
| π_b serving/update | θ_t in rollout + branch roles; collect; exactly one update | Live/asynchronous approximation only later, with measurement |
| Normalization / clipping heuristics | Off | Enable only if raw-scale training is unstable; ablate |
| 1/k_g child weighting | On (incumbent comparability) | Labeled heuristic; ablate |
| SFT cold start | ≤50 filtered teacher trajectories via `uv run sft` | More → entropy collapse (alphaXiv) |
| Steps / eval cadence | 150–300 / every 10 | Matches harness blog protocol |

## Appendix D. Primer and resource links (for a contributor starting cold)

**What an RLM is, in one paragraph.** A Recursive Language Model stores the (long)
task context as a variable in a persistent execution environment instead of in the
LM's prompt. The root LM sees only the query plus environment feedback; each turn
it emits an action (typed command or code), the environment executes it, and output
comes back as the next observation. Built-ins let it spawn child (R)LM calls
(`rlm_query`, `rlm_query_batched`) whose outputs land in environment variables, and
finish via `FINAL(answer)` / `FINAL_VAR(name)`. Two scaffold facts that matter for
training: the user prompt is re-appended each turn (turns do not share prefixes, so
each turn is its own training datum), and the prompt builder does context dropping
(old printed output is not automatically in later prompts — which is why
observation edges must come from actual rendered prompts).

**Project resources:**
- RLM paper: arxiv.org/abs/2512.24601 · code: github.com/alexzhang13/rlm
- RLM intro blog: alexzhang13.github.io/blog/2025/rlm/
- Harness / compositional-generalization blog (motivating result + scale-up envs,
  protocol, and prime-rl precedent): alexzhang13.github.io/blog/2026/harness/
- prime-rl (training stack): github.com/primeintellect-ai/prime-rl · docs:
  docs.primeintellect.ai/prime-rl (see: overview, algorithms, entrypoints, and
  `docs/bring-your-own-algorithms.md` and `docs/algorithms.md` in the repo — the
  custom `Algorithm` / custom-loss extension points ReDCO uses; note defaults
  evolve, pin commits)
- verifiers (environments): github.com/primeintellect-ai/verifiers · docs:
  docs.primeintellect.ai/verifiers (v1 = taskset/harness/runtime + `Trace`;
  v0 `MultiTurnEnv` under Legacy) · verifiers v1 announcement:
  primeintellect.ai/blog/verifiers-v1 · Environments Hub:
  app.primeintellect.ai/dashboard/environments
- Prime Intellect RLM writeup + experimental verifiers RLMEnv:
  primeintellect.ai/blog/rlm
- Reinforcing RLMs (incumbent recipe; env code to port):
  alphaxiv.org/blog/reinforcement-learning-for-rlms · SkyRL (read-only reference):
  github.com/NovaSky-AI/SkyRL
- C3 — the statistical backbone (adopt proof + CRN appendix):
  arxiv.org/abs/2603.06859
- StepPO (node-level ratio reference): arxiv.org/abs/2604.18401 ·
  RTMC: arxiv.org/abs/2604.11037 · TRACE: arxiv.org/abs/2607.13988 ·
  TEMPO: arxiv.org/abs/2509.18314 · Agent Lightning: arxiv.org/abs/2508.03680 ·
  CRAFT: arxiv.org/abs/2606.29476 · Trace/OPTO: arxiv.org/abs/2406.16218
- Counterfactual importance weighting: openreview.net (search "Counterfactual
  Credit Assignment for Policy Optimization", GSM8K masking paper)
- Failure attribution context: Who&When (arxiv.org/abs/2505.00212), causal MAS
  debugging (arxiv.org/abs/2509.08682)

**Benchmarks/data:** OOLONG, MRCRv2, GraphWalks, LongBenchPro, Ada-LEval,
BrowseComp-Plus are all named with splits in the harness blog; the alphaXiv
evidence-selection dataset generator is described in their post (synthetic
generation over arXiv paper groups) with code in their release. The synthetic
credit-probe suite (§5 item 1) is built in-house — see task specs there.
