from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from redco.analysis.stage_d_checkpoint_evidence import (
    CheckpointMember,
    StageDCheckpointManifest,
)
from redco.analysis.stage_d_live_update import adapter_file_state_sha256
from redco.analysis.stage_d_reload_supervisor import (
    ReloadWorkerCompletion,
    ReloadWorkerResult,
    StageDReloadProbe,
    _exclusive_write,
    _kill_process_tree,
    run_fresh_reload_pair,
)
from redco.contracts import canonical_json


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(path: Path, value: bytes) -> Path:
    path.write_bytes(value)
    return path


@pytest.mark.skipif(os.name == "nt", reason="reload identity is a pinned Linux contract")
def test_supervisor_owns_two_real_reload_processes(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "STABLE").write_bytes(b"")
    (checkpoint_root / "adapter_config.json").write_bytes(b"{}")
    (checkpoint_root / "adapter_model.safetensors").write_bytes(b"fixture-adapter")
    base_manifest = _write(tmp_path / "base.json", b"base-manifest")
    base_sha = _sha256(base_manifest.read_bytes())
    post_sha = _sha256(b"fixture-loaded-state")
    manifest = StageDCheckpointManifest(
        "stock",
        1,
        base_sha,
        post_sha,
        tuple(
            CheckpointMember(path.name, path.stat().st_size, _sha256(path.read_bytes()))
            for path in sorted(checkpoint_root.iterdir())
        ),
    )
    manifest_path = _write(tmp_path / "checkpoint.json", manifest.to_bytes())
    tokenizer = _write(tmp_path / "tokenizer.json", b"tokenizer")
    renderer = _write(tmp_path / "renderer.json", b"renderer")
    runtime = _write(tmp_path / "runtime.json", b"runtime")
    probe = StageDReloadProbe(
        prompt_token_ids=(11, 22, 33),
        max_new_tokens=2,
        tokenizer_manifest_sha256=_sha256(tokenizer.read_bytes()),
        renderer_manifest_sha256=_sha256(renderer.read_bytes()),
        runtime_manifest_sha256=_sha256(runtime.read_bytes()),
    )
    probe_path = _write(tmp_path / "probe.json", probe.to_bytes())
    base_root = tmp_path / "model"
    base_root.mkdir()
    evidence, outputs, result_bytes = run_fresh_reload_pair(
        arm="stock",
        checkpoint_root=checkpoint_root,
        checkpoint_manifest_path=manifest_path,
        reload_probe_path=probe_path,
        base_model_root=base_root,
        base_model_manifest_path=base_manifest,
        tokenizer_manifest_path=tokenizer,
        renderer_manifest_path=renderer,
        runtime_manifest_path=runtime,
        evidence_root=tmp_path / "evidence",
        timeout_seconds=30,
        backend="test-fixture",
        allow_test_backend=True,
    )
    assert outputs[0] == outputs[1]
    assert len(set(evidence.process_identities)) == 2
    results = tuple(
        ReloadWorkerCompletion.from_bytes(
            (tmp_path / "evidence" / f"reload-{ordinal}.completion.json").read_bytes()
        ).result
        for ordinal in (1, 2)
    )
    assert results[0].pid != results[1].pid
    assert tuple(result.identity for result in results) == evidence.process_identities
    assert tuple(result.to_bytes() for result in results) == result_bytes
    result_mtimes = tuple(
        (tmp_path / "evidence" / f"reload-{ordinal}.completion.json").stat().st_mtime_ns
        for ordinal in (1, 2)
    )
    resumed = run_fresh_reload_pair(
        arm="stock",
        checkpoint_root=checkpoint_root,
        checkpoint_manifest_path=manifest_path,
        reload_probe_path=probe_path,
        base_model_root=base_root,
        base_model_manifest_path=base_manifest,
        tokenizer_manifest_path=tokenizer,
        renderer_manifest_path=renderer,
        runtime_manifest_path=runtime,
        evidence_root=tmp_path / "evidence",
        timeout_seconds=30,
        backend="test-fixture",
        allow_test_backend=True,
    )
    assert resumed == (evidence, outputs, result_bytes)
    assert result_mtimes == tuple(
        (tmp_path / "evidence" / f"reload-{ordinal}.completion.json").stat().st_mtime_ns
        for ordinal in (1, 2)
    )


@pytest.mark.skipif(os.name == "nt", reason="reload identity is a pinned Linux contract")
def test_two_supervisors_race_to_the_same_single_worker_pair(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    members = {
        "STABLE": b"",
        "adapter_config.json": b"{}",
        "adapter_model.safetensors": b"fixture-adapter",
    }
    for name, value in members.items():
        (checkpoint_root / name).write_bytes(value)
    base_manifest = _write(tmp_path / "base.json", b"base-manifest")
    base_sha = _sha256(base_manifest.read_bytes())
    manifest = StageDCheckpointManifest(
        "stock",
        1,
        base_sha,
        _sha256(b"fixture-loaded-state"),
        tuple(
            CheckpointMember(name, len(value), _sha256(value))
            for name, value in sorted(members.items())
        ),
    )
    manifest_path = _write(tmp_path / "checkpoint.json", manifest.to_bytes())
    tokenizer = _write(tmp_path / "tokenizer.json", b"tokenizer")
    renderer = _write(tmp_path / "renderer.json", b"renderer")
    runtime = _write(tmp_path / "runtime.json", b"runtime")
    probe_path = _write(
        tmp_path / "probe.json",
        StageDReloadProbe(
            prompt_token_ids=(11, 22, 33),
            max_new_tokens=2,
            tokenizer_manifest_sha256=_sha256(tokenizer.read_bytes()),
            renderer_manifest_sha256=_sha256(renderer.read_bytes()),
            runtime_manifest_sha256=_sha256(runtime.read_bytes()),
        ).to_bytes(),
    )
    arguments = {
        "arm": "stock",
        "checkpoint_root": checkpoint_root,
        "checkpoint_manifest_path": manifest_path,
        "reload_probe_path": probe_path,
        "base_model_root": tmp_path / "model",
        "base_model_manifest_path": base_manifest,
        "tokenizer_manifest_path": tokenizer,
        "renderer_manifest_path": renderer,
        "runtime_manifest_path": runtime,
        "evidence_root": tmp_path / "evidence",
        "timeout_seconds": 30,
        "backend": "test-fixture",
        "allow_test_backend": True,
    }
    arguments["base_model_root"].mkdir()  # type: ignore[union-attr]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_fresh_reload_pair, **arguments) for _ in range(2)]
    first, second = (future.result() for future in futures)
    assert first == second
    completions = tuple((tmp_path / "evidence").glob("*.completion.json"))
    assert len(completions) == 2


@pytest.mark.skipif(os.name == "nt", reason="PEFT reload is a pinned Linux contract")
def test_real_transformers_peft_reload_hashes_the_loaded_adapter(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    base_root = tmp_path / "tiny-base"
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=64,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.save_pretrained(base_root, safe_serialization=True)
    files = []
    for path in sorted(base_root.rglob("*")):
        if path.is_file():
            value = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(base_root).as_posix(),
                    "size": len(value),
                    "sha256": _sha256(value),
                }
            )
    base_manifest = _write(
        tmp_path / "base.json",
        canonical_json(
            {
                "schema_version": 1,
                "domain": "redco-stage-d-e2-base-snapshot-v1",
                "repo_id": "test/tiny-gpt2",
                "revision": "0" * 40,
                "files": files,
            }
        ),
    )
    base_sha = _sha256(base_manifest.read_bytes())
    adapter_model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=("c_attn",),
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    with torch.no_grad():
        for name, parameter in adapter_model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.125)
    checkpoint_root = tmp_path / "checkpoint"
    adapter_model.save_pretrained(checkpoint_root, safe_serialization=True)
    if (checkpoint_root / "README.md").exists():
        (checkpoint_root / "README.md").unlink()
    (checkpoint_root / "STABLE").write_bytes(b"")
    post_sha = adapter_file_state_sha256(
        checkpoint_root / "adapter_model.safetensors",
        base_snapshot_manifest_sha256=base_sha,
    )
    checkpoint_manifest = StageDCheckpointManifest.build(
        arm="stock",
        trainer_step=1,
        checkpoint_root=checkpoint_root,
        base_model_manifest_sha256=base_sha,
        observed_post_model_sha256=post_sha,
    )
    checkpoint_manifest_path = _write(tmp_path / "checkpoint.json", checkpoint_manifest.to_bytes())
    tokenizer = _write(tmp_path / "tokenizer.json", b"tokenizer")
    renderer = _write(tmp_path / "renderer.json", b"renderer")
    runtime = _write(tmp_path / "runtime.json", b"runtime")
    probe = StageDReloadProbe(
        prompt_token_ids=(1, 7, 9),
        max_new_tokens=2,
        tokenizer_manifest_sha256=_sha256(tokenizer.read_bytes()),
        renderer_manifest_sha256=_sha256(renderer.read_bytes()),
        runtime_manifest_sha256=_sha256(runtime.read_bytes()),
    )
    probe_path = _write(tmp_path / "probe.json", probe.to_bytes())
    evidence, outputs, results = run_fresh_reload_pair(
        arm="stock",
        checkpoint_root=checkpoint_root,
        checkpoint_manifest_path=checkpoint_manifest_path,
        reload_probe_path=probe_path,
        base_model_root=base_root,
        base_model_manifest_path=base_manifest,
        tokenizer_manifest_path=tokenizer,
        renderer_manifest_path=renderer,
        runtime_manifest_path=runtime,
        evidence_root=tmp_path / "evidence",
        timeout_seconds=60,
    )
    assert outputs[0] == outputs[1]
    assert all(
        ReloadWorkerResult.from_bytes(value).loaded_model_sha256 == post_sha for value in results
    )
    evidence.verify_process_result_bytes(results)


def test_fixture_reload_is_forbidden_without_explicit_test_authorization(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="forbidden in deployment"):
        run_fresh_reload_pair(
            arm="stock",
            checkpoint_root=tmp_path,
            checkpoint_manifest_path=tmp_path / "missing",
            reload_probe_path=tmp_path / "missing",
            base_model_root=tmp_path,
            base_model_manifest_path=tmp_path / "missing",
            tokenizer_manifest_path=tmp_path / "missing",
            renderer_manifest_path=tmp_path / "missing",
            runtime_manifest_path=tmp_path / "missing",
            evidence_root=tmp_path / "evidence",
            timeout_seconds=1,
            backend="test-fixture",
        )


@pytest.mark.parametrize(
    ("fault_stage", "final_exists"),
    (("after-reload-pending-fsync", False), ("after-reload-link", True)),
)
def test_reload_atomic_write_survives_hard_process_exit(
    tmp_path: Path, fault_stage: str, final_exists: bool
) -> None:
    path = tmp_path / "completion.json"
    value = canonical_json({"completion": "exact"})
    code = r"""
import os
import sys
from pathlib import Path
from redco.analysis.stage_d_reload_supervisor import _exclusive_write

def crash(stage, _path):
    if stage == sys.argv[3]:
        os._exit(97)

_exclusive_write(Path(sys.argv[1]), bytes.fromhex(sys.argv[2]), fault_hook=crash)
"""
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), value.hex(), fault_stage],
        check=False,
        capture_output=True,
        env=environment,
    )
    assert result.returncode == 97
    assert path.exists() is final_exists
    _exclusive_write(path, value)
    assert path.read_bytes() == value


@pytest.mark.skipif(os.name == "nt", reason="process-group kill is a pinned Linux contract")
def test_reload_timeout_kills_the_entire_worker_process_group() -> None:
    process = subprocess.Popen(
        ["bash", "-c", "sleep 60 & echo $!; wait"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    child_pid = int(process.stdout.readline().decode("ascii").strip())
    assert Path(f"/proc/{child_pid}").exists()
    _kill_process_tree(process)
    deadline = time.monotonic() + 5
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert process.returncode is not None
    assert not Path(f"/proc/{child_pid}").exists()


@pytest.mark.skipif(os.name == "nt", reason="process-group kill is a pinned Linux contract")
def test_reload_cleanup_kills_descendants_after_session_leader_exits() -> None:
    process = subprocess.Popen(
        ["bash", "-c", "sleep 60 & echo $!"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    child_pid = int(process.stdout.readline().decode("ascii").strip())
    process.wait(timeout=5)
    assert Path(f"/proc/{child_pid}").exists()
    _kill_process_tree(process)
    deadline = time.monotonic() + 5
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()
