"""Single-worker Stage-D source collection environment."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import model_validator

from redco.analysis.stage_d_collection import (
    derive_scientific_group_id,
    derive_source_episode_seed_and_salt,
)
from redco.analysis.stage_d_live_observer import (
    StageDForwardDirectiveObserver,
    StageDObserverIdentity,
    StageDObserverProtocol,
    StageDPreparedCallObserver,
    require_zero_retry_configuration,
)
from redco.analysis.stage_d_receipt_ledger import (
    GenesisBinding,
    StageDReceiptLedger,
)
from redco.analysis.stage_d_runtime_isolation import (
    StageDIsolatedRuntimeContract,
    build_pre_action_runtime_snapshot,
    run_isolated_runtime_preflight,
)
from redco.analysis.stage_d_source_artifacts import StageDSourceArtifactStore
from redco.analysis.stage_d_source_contracts import SourceRollout
from redco.analysis.stage_d_source_producer import StageDSourceRolloutProducer
from redco.analysis.stage_d_spawn_provenance import PolicyEventAddress
from redco.contracts import canonical_json
from redco_evidence_selection_v2.scoring import score_evidence_reply
from redco_evidence_selection_v2.taskset import (
    CONTEXT_PATH,
    WORKDIR,
    EvidenceSelectionConfig,
    EvidenceSelectionData,
    EvidenceSelectionTaskset,
    prepare_isolated_workspace,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class StageDSourceData(EvidenceSelectionData):
    scientific_group_id: str
    rollout_slot: int


class StageDSourceTask(vf.Task[StageDSourceData]):
    NEEDS_CONTAINER = False

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del trace
        if self.data.network_allow == []:
            await prepare_isolated_workspace(runtime, self.data.paper.encode("utf-8"))
        else:
            await runtime.run(["mkdir", "-p", WORKDIR], {})
            await runtime.write(CONTEXT_PATH, self.data.paper.encode("utf-8"))

    async def pre_generation(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if self.data.network_allow != []:
            return
        if self.tool_servers():
            raise RuntimeError("Stage-D task exposes colocated model-callable servers")
        report = await run_isolated_runtime_preflight(runtime)
        trace.info["stage_d_isolated_runtime_preflight"] = json.loads(report)

    def _score(self, trace: vf.Trace) -> dict[str, float]:
        return score_evidence_reply(
            self.data.paper,
            trace.last_reply,
            self.data.reference_evidence,
        )

    @vf.reward
    async def exact_span_f1(self, trace: vf.Trace) -> float:
        return self._score(trace)["f1"]

    @vf.metric
    async def evidence_precision(self, trace: vf.Trace) -> float:
        return self._score(trace)["precision"]

    @vf.metric
    async def evidence_recall(self, trace: vf.Trace) -> float:
        return self._score(trace)["recall"]

    @vf.metric
    async def evidence_parseable(self, trace: vf.Trace) -> float:
        return self._score(trace)["parseable"]

    @vf.metric
    async def all_predicted_spans_verbatim(self, trace: vf.Trace) -> float:
        return self._score(trace)["all_predicted_spans_verbatim"]

    @vf.metric
    async def predicted_characters(self, trace: vf.Trace) -> float:
        return self._score(trace)["predicted_characters"]

    @vf.metric
    async def predicted_span_count(self, trace: vf.Trace) -> float:
        return self._score(trace)["predicted_span_count"]

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        score = self._score(trace)
        trace.info["evidence_selection"] = {
            "example_id": self.data.example_id,
            "paper_id": self.data.paper_id,
            "split": self.data.split,
            "answer_type": self.data.answer_type,
            "snapshot_sha256": self.data.snapshot_sha256,
            "score": score,
        }
        trace.info["checkpoint_id"] = self.data.policy_checkpoint_id


class StageDSourceTasksetConfig(EvidenceSelectionConfig):
    scientific_group_namespace: str
    rollouts_per_task: int

    @model_validator(mode="after")
    def validate_source_roster(self) -> StageDSourceTasksetConfig:
        if self.rollouts_per_task < 1:
            raise ValueError("Stage-D source groups require at least one frozen rollout")
        return self


class StageDSourceTaskset(vf.Taskset[StageDSourceTask, StageDSourceTasksetConfig]):
    """Wrap the frozen evidence taskset with campaign-scoped group identities."""

    def load(self) -> list[StageDSourceTask]:
        if not self.config.scientific_group_namespace:
            raise ValueError("Stage-D source taskset requires a group namespace")
        base_tasks = EvidenceSelectionTaskset(self.config).load()
        tasks: list[StageDSourceTask] = []
        for task in base_tasks:
            group_id = derive_scientific_group_id(
                namespace=self.config.scientific_group_namespace,
                example_id=task.data.example_id,
            )
            for rollout_slot in range(self.config.rollouts_per_task):
                data = StageDSourceData.model_validate(
                    {
                        **task.data.model_dump(mode="json"),
                        "scientific_group_id": group_id,
                        "rollout_slot": rollout_slot,
                    }
                )
                tasks.append(StageDSourceTask(data, self.config.task))
        return tasks


class StageDSourceEnvConfig(vf.SingleAgentEnvConfig):
    """Frozen paths and identities required for exact source collection."""

    taskset: StageDSourceTasksetConfig
    ledger_path: Path
    artifact_path: Path
    master_seed: str
    preregistration_sha256: str
    source_sha256: str
    runtime_sha256: str
    config_sha256: str
    protocol_manifest_sha256: str | None = None
    support_rules_sha256: str
    checkpoint_id: str
    base_model_manifest_path: Path
    base_model_manifest_sha256: str
    adapter_manifest_path: Path | None = None
    adapter_manifest_sha256: str | None = None
    tokenizer_manifest_path: Path
    tokenizer_manifest_sha256: str
    renderer_manifest_path: Path
    renderer_manifest_sha256: str
    sampler_conformance_manifest_path: Path
    sampler_conformance_manifest_sha256: str
    resolved_agent_sampling_law_sha256: str
    resolved_train_client_sha256: str
    branch_count: int
    continuation_replicates: int
    failure_reward: float
    frozen_workspace_manifest_path: Path | None = None
    frozen_workspace_manifest_sha256: str | None = None
    root_policy_turn_count: int = 2
    maximum_observed_root_policy_turn_count: int = 4
    child_parent_lineage: str = "root"
    child_parent_session_call_ordinal: int = 0
    child_parent_turn: int = 0
    parent_tool_call_slot: int = 0
    max_concurrent: int | None = 1

    @model_validator(mode="before")
    @classmethod
    def _resolve_taskset(cls, data: Any) -> Any:
        """Keep the source-specific taskset type inside this explicit env profile."""
        return data

    @model_validator(mode="after")
    def validate_source_collection(self) -> StageDSourceEnvConfig:
        if self.max_concurrent != 1:
            raise ValueError("Stage-D source collection requires max_concurrent=1")
        if self.maximum_observed_root_policy_turn_count < self.root_policy_turn_count:
            raise ValueError("Stage-D observed root-call ceiling is below the target topology")
        if self.retries.max_retries != 0 or self.agent.retries.max_retries != 0:
            raise ValueError("Stage-D source collection forbids episode and agent retries")
        if self.agent.sampling is not None:
            raise ValueError(
                "Stage-D source agent sampling must inherit the episode-addressed context"
            )
        if not self.master_seed or not self.checkpoint_id:
            raise ValueError("Stage-D source collection identities must be nonempty")
        for name in (
            "preregistration_sha256",
            "source_sha256",
            "runtime_sha256",
            "config_sha256",
            "base_model_manifest_sha256",
            "tokenizer_manifest_sha256",
            "renderer_manifest_sha256",
            "sampler_conformance_manifest_sha256",
            "resolved_agent_sampling_law_sha256",
            "resolved_train_client_sha256",
            "support_rules_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if (self.adapter_manifest_path is None) != (self.adapter_manifest_sha256 is None):
            raise ValueError("adapter manifest path and hash must be supplied together")
        if self.adapter_manifest_sha256 is not None and len(self.adapter_manifest_sha256) != 64:
            raise ValueError("adapter manifest hash must be SHA-256")
        if self.protocol_manifest_sha256 is not None and len(
            self.protocol_manifest_sha256
        ) != 64:
            raise ValueError("protocol manifest hash must be SHA-256")
        isolated = self.taskset.isolated_runtime_image is not None
        if isolated != (self.frozen_workspace_manifest_path is not None) or isolated != (
            self.frozen_workspace_manifest_sha256 is not None
        ):
            raise ValueError(
                "isolated source runtime requires one frozen workspace manifest"
            )
        if self.frozen_workspace_manifest_sha256 is not None and len(
            self.frozen_workspace_manifest_sha256
        ) != 64:
            raise ValueError("workspace manifest hash must be SHA-256")
        return self


class StageDSourceEnv(vf.Env[StageDSourceEnvConfig]):
    """Collect one exact source per successful episode without training it."""

    def __init__(self, config: StageDSourceEnvConfig) -> None:
        super().__init__(config)
        self.taskset = StageDSourceTaskset(config.taskset)
        self._task_cls = StageDSourceTask
        self._run_lock = asyncio.Lock()
        self._ledger: StageDReceiptLedger | None = None
        self._artifacts: StageDSourceArtifactStore | None = None
        self._producers: dict[str, StageDSourceRolloutProducer] = {}
        self._terminal_traces: set[str] = set()
        self._completed_sources: dict[str, SourceRollout] = {}
        self._failed = False
        self._identity: StageDObserverIdentity | None = None
        self._worker_lease_fd: int | None = None
        self._worker_lease_path = self.config.ledger_path.with_name(
            f"{self.config.ledger_path.name}.single-worker.lock"
        )

    async def setup(self, agents: vf.Agents) -> None:
        agents.agent.trainable = False

    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        await agents.agent.run(task)

    async def start(self) -> None:
        self._acquire_worker_lease()
        try:
            self._start_after_worker_lease()
        except BaseException:
            if self._ledger is not None:
                self._ledger.close()
                self._ledger = None
            self._artifacts = None
            self._identity = None
            self._release_worker_lease()
            raise

    def _start_after_worker_lease(self) -> None:
        if self.config.protocol_manifest_sha256 is None:
            raise ValueError("Stage-D source collection lacks its protocol manifest")
        binding = GenesisBinding(
            preregistration_sha256=self.config.preregistration_sha256,
            source_sha256=self.config.source_sha256,
            runtime_sha256=self.config.runtime_sha256,
            config_sha256=self.config.config_sha256,
            protocol_manifest_sha256=self.config.protocol_manifest_sha256,
            master_seed_sha256=_sha256(self.config.master_seed.encode("utf-8")),
            support_rules_sha256=self.config.support_rules_sha256,
        )
        if self.config.ledger_path.exists():
            self._ledger = StageDReceiptLedger(
                self.config.ledger_path,
                master_seed=self.config.master_seed,
            )
            if self._ledger.genesis_binding != binding:
                raise ValueError("existing Stage-D ledger has a different genesis binding")
            if self._ledger.record_count != 1 or self._ledger.completed_source_sha256s:
                raise RuntimeError(
                    "Stage-D source collection cannot restart after any observed call"
                )
        else:
            self._ledger = StageDReceiptLedger.create(
                self.config.ledger_path,
                binding=binding,
                master_seed=self.config.master_seed,
            )
        self._artifacts = StageDSourceArtifactStore(self.config.artifact_path)
        self._artifacts.assert_pristine()
        self._identity = StageDObserverIdentity(
            checkpoint_id=self.config.checkpoint_id,
            base_model_manifest=self._manifest(
                self.config.base_model_manifest_path,
                self.config.base_model_manifest_sha256,
                "base model",
            ),
            adapter_manifest=(
                None
                if self.config.adapter_manifest_path is None
                else self._manifest(
                    self.config.adapter_manifest_path,
                    str(self.config.adapter_manifest_sha256),
                    "adapter",
                )
            ),
            tokenizer_manifest=self._manifest(
                self.config.tokenizer_manifest_path,
                self.config.tokenizer_manifest_sha256,
                "tokenizer",
            ),
            renderer_manifest=self._manifest(
                self.config.renderer_manifest_path,
                self.config.renderer_manifest_sha256,
                "renderer",
            ),
            sampler_conformance_manifest=self._manifest(
                self.config.sampler_conformance_manifest_path,
                self.config.sampler_conformance_manifest_sha256,
                "sampler conformance",
            ),
            eos_token_id=self._eos_token_id(),
        )

    async def stop(self) -> None:
        try:
            if set(self._producers) != self._terminal_traces:
                self._failed = True
            if self._artifacts is not None and not self._failed:
                self._artifacts.assert_no_pending()
        finally:
            if self._ledger is not None:
                self._ledger.close()
                self._ledger = None
            self._release_worker_lease()

    async def run_episode(
        self,
        task: vf.Task,
        ctx: vf.ModelContext,
        **kwargs: Any,
    ) -> vf.Episode:
        async with self._run_lock:
            if self._failed:
                raise RuntimeError("Stage-D source environment is terminal after a failure")
            producers_before = set(self._producers)
            episode: vf.Episode | None = None
            producer: StageDSourceRolloutProducer | None = None
            try:
                if not isinstance(task.data, StageDSourceData):
                    raise TypeError("Stage-D source episode requires source task data")
                episode_ctx = vf.ModelContext(
                    model=ctx.model,
                    client=ctx.client,
                    sampling=_episode_sampling(
                        ctx.sampling,
                        master_seed=self.config.master_seed,
                        scientific_group_id=task.data.scientific_group_id,
                        rollout_slot=task.data.rollout_slot,
                    ),
                )
                episode = await super().run_episode(task, episode_ctx, **kwargs)
                if not episode.ok or len(episode.traces) != 1:
                    raise RuntimeError("source collection episode did not complete exactly once")
                trace = episode.traces[0]
                producer = self._producers.get(trace.id)
                if producer is None or trace.id in self._terminal_traces:
                    raise RuntimeError("source collection trace registry is inconsistent")
                raw_episode = canonical_json(episode.model_dump(mode="json"))
                if self._artifacts is None:
                    raise RuntimeError("source artifact store is not started")

                def prepare(value: bytes) -> None:
                    assert self._artifacts is not None
                    self._artifacts.prepare(value)

                source = producer.finalize_episode(
                    raw_episode,
                    prepare_source_rollout=prepare,
                )
                self._artifacts.commit(source)
                self._terminal_traces.add(trace.id)
                self._completed_sources[trace.id] = source
                return episode
            except asyncio.CancelledError as error:
                self._failed = True
                self._abort_interrupted_episode(
                    producer,
                    producers_before=producers_before,
                    error=error,
                )
                raise
            except Exception as error:
                self._failed = True
                self._abort_interrupted_episode(
                    producer,
                    producers_before=producers_before,
                    error=error,
                )
                if episode is None:
                    raise
                episode.ok = False
                episode.errors.append(
                    vf.Error(
                        type=type(error).__qualname__,
                        message=str(error),
                        traceback=traceback.format_exc(),
                    )
                )
                return episode

    def _abort_interrupted_episode(
        self,
        producer: StageDSourceRolloutProducer | None,
        *,
        producers_before: set[str],
        error: BaseException,
    ) -> None:
        resolved = producer
        if resolved is None:
            new_ids = set(self._producers) - producers_before
            if len(new_ids) == 1:
                resolved = self._producers[new_ids.pop()]
            elif new_ids:
                raise RuntimeError("interrupted source episode registered multiple producers")
        if resolved is not None:
            resolved.abort_finalization(error)

    def prepared_call_observer(
        self,
        task: vf.Task,
        trace: vf.Trace,
        agent_config: vf.AgentConfig,
        client: vf.Client,
    ) -> StageDForwardDirectiveObserver:
        from verifiers.v1.clients.train import TrainClient

        if self._failed or self._ledger is None or self._identity is None:
            raise RuntimeError("Stage-D source environment is not writable")
        if trace.id in self._producers or trace.id in self._terminal_traces:
            raise ValueError("source collection trace was registered more than once")
        if not isinstance(task.data, StageDSourceData):
            raise TypeError("Stage-D source environment requires evidence-selection data")
        if not task.data.scientific_group_id:
            raise ValueError("source task lacks its scientific group ID")
        if not isinstance(client, TrainClient):
            raise TypeError("Stage-D source collection requires the resolved TrainClient")
        require_zero_retry_configuration(
            agent_max_retries=agent_config.retries.max_retries,
            client_max_retries=client.openai.max_retries,
        )
        if agent_config.model != self.config.checkpoint_id:
            raise ValueError("resolved agent checkpoint differs from the frozen checkpoint")
        if agent_config.sampling is None:
            raise ValueError("resolved agent lacks a sampling configuration")
        sampling_law_sha256 = _resolved_agent_sampling_law_sha256(agent_config.sampling)
        if sampling_law_sha256 != self.config.resolved_agent_sampling_law_sha256:
            raise ValueError("resolved agent sampling law differs from its frozen manifest")
        expected_seed, expected_salt = _episode_seed_and_salt(
            master_seed=self.config.master_seed,
            scientific_group_id=task.data.scientific_group_id,
            rollout_slot=task.data.rollout_slot,
        )
        sampling_payload = agent_config.sampling.model_dump(mode="json", exclude_none=False)
        extra_body = sampling_payload.get("extra_body")
        if (
            sampling_payload.get("seed") != expected_seed
            or not isinstance(extra_body, dict)
            or extra_body.get("cache_salt") != expected_salt
        ):
            raise ValueError("resolved agent seed differs from its episode address")
        if _resolved_train_client_sha256(client) != self.config.resolved_train_client_sha256:
            raise ValueError("resolved TrainClient differs from its frozen manifest")
        parent = PolicyEventAddress(
            0,
            self.config.child_parent_lineage,
            self.config.child_parent_session_call_ordinal,
            self.config.child_parent_turn,
        )
        producer = StageDSourceRolloutProducer(
            ledger=self._ledger,
            group_id=task.data.scientific_group_id,
            rollout_id=trace.id,
            child_parent_event=parent,
            child_parent_tool_call_slot=self.config.parent_tool_call_slot,
            root_policy_turn_count=self.config.root_policy_turn_count,
            base_model_manifest_sha256=self.config.base_model_manifest_sha256,
        )
        self._producers[trace.id] = producer
        return StageDForwardDirectiveObserver(
            StageDPreparedCallObserver(
                producer=producer,
                trace_id=trace.id,
                identity=self._identity,
                protocol=StageDObserverProtocol(
                    branch_count=self.config.branch_count,
                    continuation_replicates=self.config.continuation_replicates,
                    failure_reward=self.config.failure_reward,
                    root_policy_turn_count=self.config.root_policy_turn_count,
                    maximum_observed_root_policy_turn_count=(
                        self.config.maximum_observed_root_policy_turn_count
                    ),
                    child_parent_event=parent,
                    parent_tool_call_slot=self.config.parent_tool_call_slot,
                ),
                runtime_snapshot=self._runtime_snapshot(task, agent_config),
                encode_action=lambda request, message, prompt_token_ids: (
                    client.encode_assistant_action(
                        request,
                        message,
                        model=str(agent_config.model),
                        prompt_token_ids=prompt_token_ids,
                    )
                ),
            ),
            pre_forward_guard=lambda: self._require_runtime_preflight(trace),
        )

    def _require_runtime_preflight(self, trace: vf.Trace) -> None:
        if self.config.taskset.isolated_runtime_image is None:
            return
        preflight = trace.info.get("stage_d_isolated_runtime_preflight")
        if not isinstance(preflight, dict) or preflight.get("domain") != (
            "redco-stage-d-isolated-runtime-preflight-v1"
        ):
            raise RuntimeError("Stage-D source call bypassed runtime preflight")

    def _runtime_snapshot(self, task: vf.Task, agent_config: Any) -> bytes:
        image = self.config.taskset.isolated_runtime_image
        if image is None:
            return canonical_json(
                {
                    "schema_version": 1,
                    "domain": "redco-stage-d-legacy-runtime-binding-v1",
                    "runtime_sha256": self.config.runtime_sha256,
                }
            )
        if not isinstance(task.data, StageDSourceData):
            raise TypeError("isolated runtime snapshot requires source task data")
        manifest_path = self.config.frozen_workspace_manifest_path
        manifest_sha256 = self.config.frozen_workspace_manifest_sha256
        assert manifest_path is not None and manifest_sha256 is not None
        manifest_bytes = self._manifest(
            manifest_path,
            manifest_sha256,
            "frozen workspace",
        )
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("frozen workspace manifest must be a JSON object")
        runtime = agent_config.harness.runtime.model_dump(
            mode="json",
            exclude_none=False,
        )
        return build_pre_action_runtime_snapshot(
            contract=StageDIsolatedRuntimeContract(image),
            runtime_config=runtime,
            task_data=task.data.model_dump(mode="json", exclude_none=False),
            task_config=task.config.model_dump(mode="json", exclude_none=False),
            paper=task.data.paper.encode("utf-8"),
            frozen_workspace_manifest=manifest,
        )

    def verified_completed_sources(self) -> tuple[SourceRollout, ...]:
        """Return the exact in-memory sources only after a clean complete run."""
        if self._failed:
            raise RuntimeError("Stage-D source environment ended in failure")
        expected = set(self._producers)
        if (
            not expected
            or expected != self._terminal_traces
            or expected != set(self._completed_sources)
        ):
            raise RuntimeError("Stage-D source collection is not terminally complete")
        return tuple(
            self._completed_sources[rollout_id]
            for rollout_id in sorted(self._completed_sources)
        )

    def _manifest(self, path: Path, expected: str, name: str) -> bytes:
        value = path.read_bytes()
        if not value or _sha256(value) != expected:
            raise ValueError(f"{name} manifest bytes differ from the frozen hash")
        return value

    def _acquire_worker_lease(self) -> None:
        if self._worker_lease_fd is not None:
            raise RuntimeError("Stage-D source worker lease is already held")
        self._worker_lease_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self._worker_lease_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )

        except FileExistsError as error:
            raise RuntimeError(
                "Stage-D source collection requires one static worker; lease exists"
            ) from error
        try:
            os.write(descriptor, canonical_json({"pid": os.getpid()}))
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            self._worker_lease_path.unlink(missing_ok=True)
            raise
        self._worker_lease_fd = descriptor

    def _release_worker_lease(self) -> None:
        descriptor = self._worker_lease_fd
        if descriptor is None:
            return
        self._worker_lease_fd = None
        os.close(descriptor)
        self._worker_lease_path.unlink(missing_ok=True)

    def _eos_token_id(self) -> int:
        payload = json.loads(self.config.tokenizer_manifest_path.read_bytes())
        value = payload.get("eos_token_id") if isinstance(payload, dict) else None
        if type(value) is not int or value < 0:
            raise ValueError("tokenizer manifest lacks an exact EOS token ID")
        return value


def _resolved_train_client_sha256(client: Any) -> str:
    config = client.config
    payload = {
        "schema_version": 1,
        "domain": "redco-stage-d-resolved-train-client-v1",
        "base_url": str(client.openai.base_url).rstrip("/"),
        "max_retries": client.openai.max_retries,
        "pool_size": client.pool_size,
        "renderer_model_name": client.renderer_model_name,
        "default_headers": dict(client.default_headers),
        "api_key_var": client.api_key_var,
        "renderer_config": (None if config is None else config.model_dump(mode="json")),
    }
    return _sha256(canonical_json(payload))


def _episode_seed_and_salt(
    *,
    master_seed: str,
    scientific_group_id: str,
    rollout_slot: int,
) -> tuple[int, str]:
    return derive_source_episode_seed_and_salt(
        master_seed=master_seed,
        scientific_group_id=scientific_group_id,
        rollout_slot=rollout_slot,
    )


def _episode_sampling(
    sampling: vf.Sampling,
    *,
    master_seed: str,
    scientific_group_id: str,
    rollout_slot: int,
) -> vf.Sampling:
    seed, cache_salt = _episode_seed_and_salt(
        master_seed=master_seed,
        scientific_group_id=scientific_group_id,
        rollout_slot=rollout_slot,
    )
    payload = sampling.model_dump(mode="json", exclude_none=False)
    extra_body = payload.get("extra_body")
    if extra_body is None:
        extra_body = {}
    if not isinstance(extra_body, dict) or set(extra_body) - {"cache_salt"}:
        raise ValueError("source sampling extra_body may contain only cache_salt")
    return sampling.model_copy(
        update={"seed": seed, "extra_body": {"cache_salt": cache_salt}},
        deep=True,
    )


def _resolved_agent_sampling_law_sha256(sampling: vf.Sampling) -> str:
    payload = sampling.model_dump(mode="json", exclude_none=False)
    payload.pop("seed", None)
    extra_body = payload.pop("extra_body", None)
    if extra_body is not None and (
        not isinstance(extra_body, dict) or set(extra_body) - {"cache_salt"}
    ):
        raise ValueError("source sampling extra_body may contain only cache_salt")
    return _sha256(
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-source-sampling-law-v1",
                "sampling": payload,
            }
        )
    )


__all__ = [
    "StageDSourceEnv",
    "StageDSourceTaskset",
]
