"""Exercise Stage D event replay through the production Verifiers path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from redco.analysis.rlm_episode_replay import (
    CounterfactualCompletionRouter,
    HTTPCompletionGenerator,
    ScriptedCompletionRouter,
    ScriptedModelServer,
    inject_committed_child_answer,
    trace_to_scripted_events,
)
from redco.analysis.stage_d_all_child_support_v2 import (
    precommit_all_depth_one_policy_targets,
    verify_canonical_precommit_v2,
)
from redco.integrations.signed_subprocess import atomic_write_json, sign_payload
from redco.integrations.verifiers_trace import audit_trace_file
from scripts.build_stage_d_scaffold_sft_v2 import scaffold_code

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RLM_COMMIT = "56218f33796ecbe465445bc43948886354fde196"
QUESTION = "What changed in latency, and what happened to accuracy?"
LATENCY = "The intervention reduced latency from 420 ms to 260 ms."
ACCURACY = "Exact-answer accuracy remained 81.4 percent."
SANDBOX_SENTINEL = ".redco-stage-d-production-replay-sentinel"


class _TokenServer(ThreadingHTTPServer):
    tokenizer: Any
    model: str
    lock: threading.Lock
    root_turns: int
    requests: list[dict[str, Any]]


class _TokenHandler(BaseHTTPRequestHandler):
    server: _TokenServer

    def do_GET(self) -> None:
        if self.path.rstrip("/") != "/v1/models":
            self.send_error(404)
            return
        self._reply(
            {
                "object": "list",
                "data": [
                    {
                        "id": self.server.model,
                        "object": "model",
                        "max_model_len": 32768,
                    }
                ],
            }
        )

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            self._chat_completion()
            return
        if self.path != "/inference/v1/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            self.send_error(400)
            return
        token_ids = body.get("token_ids")
        if not isinstance(token_ids, list):
            self.send_error(400)
            return
        prompt = self.server.tokenizer.decode(token_ids, skip_special_tokens=False)
        # The root carries the frozen scaffold on every turn; recursive prompts
        # contain the excerpt request but never inherit that root-only marker.
        is_child = "SHARED MIDPOINT-SHARD SCAFFOLD INTERVENTION V4" not in prompt
        with self.server.lock:
            if is_child:
                response_text = "duplicate child answer\nwith two lines"
                classification = "child"
            else:
                root_turn = self.server.root_turns
                self.server.root_turns += 1
                classification = f"root-{root_turn}"
                if root_turn == 0:
                    response_text = _tool_call(scaffold_code(QUESTION))
                elif root_turn == 1:
                    response_text = _tool_call(
                        "value = '|'.join(child_answers)\n"
                        "open('/workspace/replay_result.txt', 'w', "
                        "encoding='utf-8').write(value)\n"
                        "print(value)"
                    )
                elif root_turn == 2:
                    response_text = repr([LATENCY, ACCURACY])
                else:
                    raise RuntimeError("unexpected extra root model call")
            self.server.requests.append(
                {
                    "classification": classification,
                    "request_sha256": hashlib.sha256(
                        json.dumps(body, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
        if is_child and "FIRST_HALF_MARKER" in prompt:
            time.sleep(0.08)
        completion_ids = self.server.tokenizer.encode(response_text, add_special_tokens=False)
        self._reply(
            {
                "request_id": f"fixture-{len(self.server.requests)}",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": completion_ids,
                        "logprobs": {
                            "content": [
                                {"token": f"token_id:{token_id}", "logprob": -0.1}
                                for token_id in completion_ids
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    def _chat_completion(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            self.send_error(400)
            return
        messages = body["messages"]
        tool_messages = [
            message.get("content")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        last_tool = tool_messages[-1] if tool_messages else ""
        if not isinstance(last_tool, str):
            self.send_error(400)
            return
        if "|" not in last_tool:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "ipython",
                            "arguments": json.dumps(
                                {
                                    "code": (
                                        "value = '|'.join(child_answers)\n"
                                        "open('/workspace/replay_result.txt', 'w', "
                                        "encoding='utf-8').write(value)\n"
                                        "print(value)"
                                    )
                                },
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            final_spans = [LATENCY] if "changed" in last_tool else [LATENCY, ACCURACY]
            message = {"role": "assistant", "content": repr(final_spans)}
            finish_reason = "stop"
        self._reply(
            {
                "id": "counterfactual-regression",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", MODEL),
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    def _reply(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _: str, *args: object) -> None:
        del args


def _tool_call(code: str) -> str:
    payload = json.dumps(
        {"name": "ipython", "arguments": {"code": code}},
        separators=(",", ":"),
    )
    return f"<tool_call>\n{payload}\n</tool_call>"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token_sha256(token_ids: tuple[int, ...] | list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def _blocked_network_check() -> dict[str, Any]:
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=0.25)
    except OSError as error:
        return {"blocked": True, "error_type": type(error).__name__}
    raise RuntimeError("non-loopback network is reachable")


def _validated_task_workspace(work: Path) -> tuple[Path, Path]:
    declared = os.environ.get("REDCO_STAGE_D_PRODUCTION_SANDBOX_ROOT")
    if not declared:
        raise RuntimeError("production replay sandbox root is not declared")
    sandbox_root = Path(declared).resolve(strict=True)
    if not work.resolve().is_relative_to(sandbox_root):
        raise RuntimeError("production work directory is outside the sandbox root")
    task_workspace = Path("/workspace").resolve(strict=True)
    if task_workspace != Path("/workspace"):
        raise RuntimeError("production task workspace is not the canonical mount")
    sentinel = task_workspace / SANDBOX_SENTINEL
    if not sentinel.is_file():
        raise RuntimeError("production task workspace sentinel is absent")
    if sentinel.read_text(encoding="utf-8").strip() != str(sandbox_root):
        raise RuntimeError("production task workspace sentinel does not match")
    return task_workspace, sentinel


def _run_rlm(
    *,
    python: Path,
    workspace: Path,
    router: Any,
    prompt: str,
    model: str,
    paper_text: str,
    task_workspace: Path,
    sentinel: Path,
) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=False)
    for path in task_workspace.iterdir():
        if path == sentinel:
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    (task_workspace / "evidence_context.txt").write_text(paper_text, encoding="utf-8")
    initial_manifest = {
        SANDBOX_SENTINEL: _sha256(sentinel),
        "evidence_context.txt": _sha256(task_workspace / "evidence_context.txt"),
    }
    result_path = task_workspace / "replay_result.txt"
    with ScriptedModelServer(router) as server:
        env = {
            **os.environ,
            "RLM_API_KEY": "scripted-local-only",
            "RLM_BASE_URL": server.base_url,
            "RLM_MODEL": model,
            "RLM_DEPTH": "0",
            "RLM_MAX_DEPTH": "1",
            "RLM_SDK_MAX_RETRIES": "0",
            "RLM_MAX_TOKENS": "8192",
            "RLM_HOME": str(workspace / ".rlm"),
        }
        program = (
            "import asyncio,json; from rlm.api import run; "
            f"r=asyncio.run(run({prompt!r}, "
            f"cwd={str(task_workspace)!r})); "
            "print(json.dumps({'answer':r.answer,'turns':r.turns}))"
        )
        completed = subprocess.run(
            [str(python), "-c", program],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"RLM replay failed with {completed.returncode}: "
            f"{completed.stderr[-4000:]} router_audit="
            f"{json.dumps(router.audit(), sort_keys=True)[-8000:]}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    result: dict[str, Any] = json.loads(lines[-1])
    result["terminal_text"] = result_path.read_text(encoding="utf-8")
    result["initial_workspace_manifest"] = initial_manifest
    result["router_audit"] = router.audit()
    return result


def run(args: argparse.Namespace) -> None:
    network = _blocked_network_check()
    repo = args.repo.resolve()
    work = args.work.resolve()
    if work.exists():
        raise FileExistsError(work)
    work.mkdir(parents=True)
    task_workspace, sentinel = _validated_task_workspace(work)
    dataset = work / "fixture.jsonl"
    midpoint_padding = "A" * 128
    paper = f"FIRST_HALF_MARKER {LATENCY} {midpoint_padding} SECOND_HALF_MARKER {ACCURACY}"
    row = {
        "example_id": "event-replay-regression-000",
        "paper_id": "event-replay-regression",
        "title": "Event replay regression",
        "question": QUESTION,
        "paper": paper,
        "reference_evidence": [LATENCY, ACCURACY],
        "answer_type": "list",
        "split": "event_replay_regression",
    }
    dataset.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    scaffold = repo / "configs/stage-d/stage-d0-scaffold-fewshot-v4.txt"
    output_dir = work / "production"

    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    server = _TokenServer(("127.0.0.1", 0), _TokenHandler)
    server.tokenizer = tokenizer
    server.model = MODEL
    server.lock = threading.Lock()
    server.root_turns = 0
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command = [
            str(args.production_python),
            "-m",
            "redco_evidence_selection_v2.run_event_replay_regression_v1",
            "--model",
            MODEL,
            "--renderer-model-name",
            MODEL,
            "--base-url",
            f"http://127.0.0.1:{server.server_address[1]}/v1",
            "--dataset",
            str(dataset),
            "--dataset-sha256",
            _sha256(dataset),
            "--split",
            "event_replay_regression",
            "--prompt-profile",
            "fewshot_fixture_v4",
            "--scaffold-prompt",
            str(scaffold),
            "--scaffold-prompt-sha256",
            _sha256(scaffold),
            "--output-dir",
            str(output_dir),
            "--num-tasks",
            "1",
            "--replicates",
            "1",
            "--max-completion-tokens",
            "768",
            "--max-total-tokens",
            "8192",
            "--rlm-version",
            RLM_COMMIT,
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "VLLM_API_KEY": "local-regression",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    (work / "production.stdout").write_text(completed.stdout, encoding="utf-8")
    (work / "production.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"production CLI failed with {completed.returncode}: {completed.stderr[-2000:]}"
        )
    trace = output_dir / "traces.jsonl"
    audit = audit_trace_file(trace)
    precommit = precommit_all_depth_one_policy_targets(trace)
    verify_canonical_precommit_v2(trace, precommit)
    atomic_write_json(work / "precommit.json", precommit)
    events = trace_to_scripted_events(
        trace,
        expected_sha256=_sha256(trace),
        signed_precommit=precommit,
        engineering_transport_path_normalization=True,
    )
    _, target_event, committed_candidate = inject_committed_child_answer(
        events,
        signed_precommit=precommit,
        candidate_rank=0,
        answer="changed",
    )
    trace_row = json.loads(trace.read_text(encoding="utf-8"))
    native_trace = trace_row["traces"][0]
    replay_prompt = native_trace["task"]["data"]["prompt"]
    replay_model = audit.calls[0].checkpoint_id
    original = _run_rlm(
        python=args.rlm_python,
        workspace=work / "replay-original",
        router=ScriptedCompletionRouter(events),
        prompt=replay_prompt,
        model=replay_model,
        paper_text=paper,
        task_workspace=task_workspace,
        sentinel=sentinel,
    )
    counterfactual_server = _TokenServer(("127.0.0.1", 0), _TokenHandler)
    counterfactual_server.tokenizer = tokenizer
    counterfactual_server.model = MODEL
    counterfactual_server.lock = threading.Lock()
    counterfactual_server.root_turns = 0
    counterfactual_server.requests = []
    counterfactual_thread = threading.Thread(
        target=counterfactual_server.serve_forever, daemon=True
    )
    counterfactual_thread.start()
    generator = HTTPCompletionGenerator(
        base_url=(f"http://127.0.0.1:{counterfactual_server.server_address[1]}"),
        api_key="local-regression",
        timeout_seconds=30,
        temperature=0.7,
        max_tokens=768,
    )
    branch_router = CounterfactualCompletionRouter(
        events,
        target=target_event,
        candidate_message={"role": "assistant", "content": "changed"},
        candidate_finish_reason="stop",
        candidate_prompt_tokens=1,
        candidate_completion_tokens=1,
        master_seed="stage-d-production-replay-regression-v1",
        trace_id=str(native_trace["id"]),
        target_id=str(committed_candidate["pre_action_rank_sha256"]),
        generator=generator,
    )
    try:
        branch = _run_rlm(
            python=args.rlm_python,
            workspace=work / "replay-branch",
            router=branch_router,
            prompt=replay_prompt,
            model=replay_model,
            paper_text=paper,
            task_workspace=task_workspace,
            sentinel=sentinel,
        )
    finally:
        counterfactual_server.shutdown()
        counterfactual_thread.join(timeout=5)
        counterfactual_server.server_close()
    child_calls = [call for call in audit.calls if call.agent_depth == 1]
    from redco_evidence_selection_v2.scoring import (  # type: ignore[import-not-found]
        score_evidence_reply,
    )

    original_score = score_evidence_reply(paper, original["answer"], (LATENCY, ACCURACY))
    branch_score = score_evidence_reply(paper, branch["answer"], (LATENCY, ACCURACY))
    expected_branch_terminal = (
        "changed|duplicate child answer\nwith two lines"
        if target_event.invocation_id == "midpoint-shard-0"
        else "duplicate child answer\nwith two lines|changed"
    )
    payload = sign_payload(
        {
            "schema_version": 1,
            "analysis": "stage-d-production-event-replay-regression",
            "claim_scope": {
                "engineering_only": True,
                "establishes": (
                    "A signed child intervention propagates through fresh "
                    "downstream RLM calls and changes the actual terminal score."
                ),
                "does_not_establish": [
                    "scientific LOO validity",
                    "exact rendered-token request identity",
                    "behavior-policy action/logprob capture",
                    "training-record integration",
                ],
                "transport_normalization": [
                    "documented RLM working-directory line",
                    "documented RLM conversation-log line",
                    "tool-call assistant null versus empty content",
                    "Verifiers-only tool-result name metadata",
                ],
            },
            "network": network,
            "source_trace_sha256": _sha256(trace),
            "source_hashes": {
                path: _sha256(repo / path)
                for path in (
                    "uv.lock",
                    "configs/stage-d/stage-d1-event-replay-successor-design-v1.json",
                    "configs/stage-d/stage-d0-scaffold-fewshot-v4.txt",
                    "patches/rlm-event-replay-provenance.patch",
                    "patches/verifiers-rlm-event-replay-provenance.patch",
                    "environments/redco_evidence_selection_v2/redco_evidence_selection_v2/run_event_replay_regression_v1.py",
                    "environments/redco_evidence_selection_v2/redco_evidence_selection_v2/run_feasibility.py",
                    "environments/redco_evidence_selection_v2/redco_evidence_selection_v2/taskset.py",
                    "scripts/run_stage_d_production_replay_regression_v1.py",
                    "scripts/run_stage_d_production_replay_wsl_v1.sh",
                    "src/redco/analysis/rlm_episode_replay.py",
                    "src/redco/analysis/stage_d_all_child_support_v2.py",
                    "src/redco/integrations/verifiers_trace.py",
                    ".redco/vendor/rlm-56218f3/src/rlm/api.py",
                    ".redco/vendor/rlm-56218f3/src/rlm/client.py",
                    ".redco/vendor/rlm-56218f3/src/rlm/engine.py",
                    ".redco/vendor/rlm-56218f3/src/rlm/session.py",
                    "external/prime-rl/deps/verifiers/verifiers/v1/interception/server.py",
                    "external/prime-rl/deps/verifiers/verifiers/v1/trace.py",
                )
            },
            "runtime_commits": {
                "rlm": RLM_COMMIT,
                "verifiers": "b13ba60da63cea91389e7575766b7270d0d11fc5",
                "prime_rl": "3b22dd951cad1036d1fe8dd0a0bfc40807a9b360",
            },
            "production_command": command,
            "production_requests": server.requests,
            "trace_audit": {
                "model_call_count": audit.model_call_count,
                "linked_call_count": audit.linked_call_count,
                "exact_key_complete_count": audit.exact_key_complete_count,
                "failed_model_call_count": audit.failed_model_call_count,
                "ready_for_exact_key_replay": audit.ready_for_exact_key_replay,
                "calls": [
                    {
                        "call_index": call.call_index,
                        "agent_depth": call.agent_depth,
                        "turn_index": call.turn_index,
                        "call_kind": call.call_kind,
                        "parent_turn_index": call.parent_turn_index,
                        "parent_tool_call_id": call.parent_tool_call_id,
                        "invocation_id": call.invocation_id,
                        "prompt_sha256": call.prompt_sha256,
                        "action_sha256": _token_sha256(call.action_token_ids),
                    }
                    for call in audit.calls
                ],
            },
            "precommit_signed_payload_sha256": precommit["signed_payload_sha256"],
            "target_address": target_event.key(),
            "committed_candidate": {
                key: value
                for key, value in committed_candidate.items()
                if key != "prompt_token_ids"
            }
            | {"prompt_sha256": _token_sha256(committed_candidate["prompt_token_ids"])},
            "original": original,
            "branch": branch,
            "scores": {"original": original_score, "branch": branch_score},
            "passes": (
                audit.model_call_count == 5
                and len(child_calls) == 2
                and all(
                    call.parent_tool_call_id is not None and call.invocation_id is not None
                    for call in child_calls
                )
                and {call.invocation_id for call in child_calls}
                == {"midpoint-shard-0", "midpoint-shard-1"}
                and original["terminal_text"]
                == "duplicate child answer\nwith two lines|duplicate child answer\nwith two lines"
                and branch["terminal_text"] == expected_branch_terminal
                and original["initial_workspace_manifest"] == branch["initial_workspace_manifest"]
                and original["router_audit"]["complete"]
                and branch["router_audit"]["valid_counterfactual"]
                and branch["router_audit"]["resampled_policy_calls"] == 2
                and original_score["f1"] == 1.0
                and branch_score["f1"] < original_score["f1"]
            ),
        }
    )
    atomic_write_json(args.output, payload)
    if not payload["passes"]:
        raise RuntimeError("production event-replay regression failed")
    if not args.retain_work:
        for path in (work / "replay-original", work / "replay-branch"):
            shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-python", type=Path, required=True)
    parser.add_argument("--rlm-python", type=Path, required=True)
    parser.add_argument("--retain-work", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
