"""Offline, hash-bound Linux pod bootstrap for the v13 support attempt.

The bootstrap is installed and invoked by the local orchestrator only after a
raw Prime observation has passed.  It is never run during CPU verification and
has no network fallback or scientific parameter overrides.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redco.analysis.stage_d_v13_draft import sha256_bytes
from redco.analysis.stage_d_v13_launch_observations import (
    capture_pod_runtime_observation,
)

ROOT = Path(__file__).resolve().parents[1]
VLLM_BASE_URL = "http://127.0.0.1:8000"
UV_RELATIVE = ".runtime/stage-d/uv"
UV_CACHE_RELATIVE = ".runtime/stage-d/uv-cache"
VLLM_MODEL = "/workspace/models/stage-d1-merged"
VLLM_PORT = 8000
TERMINATION_SECONDS = 2.0
MERGED_MODEL_MANIFEST = (
    "runs/stage-d/stage-d1-support-v13-launch/runtime/merged-model-manifest.json"
)


MERGE_SCRIPT_SHA256 = "06046f345d0ac29e0919d43fce50c2a6ae20a29d03be215785323413e28d0416"
LORA_HELPER_SHA256 = "0647f6ed77e11757fe25279e5f29e560ddd3bd8a4c1fb8db26b944553d52d846"
ADAPTER_CONFIG_SHA256 = "623e0f9d88bb63de8766bf07063f09de40786780b5a7d794e073b6e57e18065c"


@dataclass(frozen=True, slots=True)
class AssetBinding:
    local_source: Path
    remote_destination: str
    sha256: str


def _remote_destination_path(root: Path, destination: str) -> Path:
    if destination.startswith("/workspace/redco/"):
        return root / destination.removeprefix("/workspace/redco/")
    if destination.startswith("/workspace/"):
        return Path(destination)
    raise RuntimeError("launch asset destination is not a fixed Linux path")


def required_asset_mappings(
    root: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, AssetBinding]:
    """Resolve the signed asset contract for the local orchestrator.

    The operator supplies ``artifact_root`` on the local platform.  Remote
    bootstrap uses :func:`required_remote_asset_mappings` instead and never
    interprets a Linux destination as a local source.
    """

    from redco.analysis import stage_d_v13_support_launch as launch

    if artifact_root is None:
        artifact_root = root / ".runtime/stage-d/artifact-store"
    contract = launch.asset_binding_contract(root)
    return {
        name: AssetBinding(
            launch.resolve_local_asset_locator(root, artifact_root, value["local_locator"]),
            value["remote_destination"],
            value["sha256"],
        )
        for name, value in contract.items()
    }


def required_remote_asset_mappings(root: Path) -> dict[str, AssetBinding]:
    from redco.analysis import stage_d_v13_support_launch as launch

    return {
        name: AssetBinding(
            _remote_destination_path(root, value["remote_destination"]),
            value["remote_destination"],
            value["sha256"],
        )
        for name, value in launch.asset_binding_contract(root).items()
    }


def required_asset_paths(root: Path) -> dict[str, tuple[Path, str]]:
    """Compatibility projection for runtime observation hashing."""

    return {
        name: (binding.local_source, binding.sha256)
        for name, binding in required_remote_asset_mappings(root).items()
    }


def _runtime_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(root / UV_CACHE_RELATIVE)
    env["UV_PROJECT_ENVIRONMENT"] = str(root / ".runtime/stage-d/venv")
    env["PYTHONPATH"] = str(root / "src")
    return env


def _run(root: Path, *args: str) -> None:
    uv = root / UV_RELATIVE
    if uv.is_symlink() or not uv.is_file():
        raise RuntimeError("pinned offline uv binary is missing")
    subprocess.run(
        [str(uv), "run", "--offline", "--no-project", "python", *args],
        cwd=root,
        check=True,
        env=_runtime_env(root),
    )


def _verify_runtime_versions(root: Path) -> bytes:
    uv = root / UV_RELATIVE
    code = (
        "import datasets, json, pyarrow, sys; "
        "assert sys.version_info[:3] == (3, 12, 3); "
        "assert datasets.__version__ == '5.0.0'; "
        "assert pyarrow.__version__ == '25.0.0'; "
        "print(json.dumps("
        "{'python':'3.12.3','datasets':'5.0.0','pyarrow':'25.0.0'}, "
        "separators=(',', ':')))"
    )
    result = subprocess.run(
        [str(uv), "run", "--offline", "--no-project", "python", "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        env=_runtime_env(root),
    )
    return bytes(result.stdout).strip()


def _verify_assets(root: Path) -> None:
    for name, binding in sorted(required_remote_asset_mappings(root).items()):
        source = binding.local_source
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"required launch asset is missing: {name}")
        if sha256_bytes(source.read_bytes()) != binding.sha256:
            raise RuntimeError(f"required launch asset hash differs: {name}")
    adapter_config = root / ".runtime/stage-d1/adapter/adapter_config.json"
    if (
        adapter_config.is_symlink()
        or not adapter_config.is_file()
        or sha256_bytes(adapter_config.read_bytes()) != ADAPTER_CONFIG_SHA256
    ):
        raise RuntimeError("required adapter configuration is missing or changed")


def _prepare_offline_runtime(root: Path) -> None:
    mappings = required_remote_asset_mappings(root)
    archives = (
        ("offline:checkout_archive_path", root / ".runtime/stage-d/rlm-extracted"),
        ("offline:checkout_cache_archive_path", root / UV_CACHE_RELATIVE),
        ("adapter:archive", root / ".runtime/stage-d1/adapter"),
    )
    for name, extract in archives:
        binding = mappings[name]
        archive = binding.local_source
        if archive.is_symlink() or not archive.is_file():
            raise RuntimeError(f"required offline archive is missing: {name}")
        if extract.exists() and extract.is_symlink():
            raise RuntimeError(f"archive extraction path is unsafe: {name}")
        if extract.exists():
            raise RuntimeError(f"archive extraction path is already populated: {name}")
        with tarfile.open(archive, mode="r:*") as handle:
            members = handle.getmembers()
            for member in members:
                normalized = PurePosixPath(member.name)
                if (
                    normalized.is_absolute()
                    or any(part in {"", ".", ".."} for part in normalized.parts)
                    or member.issym()
                    or member.islnk()
                ):
                    raise RuntimeError(f"offline archive has an unsafe member: {name}")
            staging = extract.with_name(extract.name + ".staging")
            if staging.exists() or staging.is_symlink():
                raise RuntimeError(f"offline archive staging path already exists: {name}")
            staging.mkdir(parents=True)
            try:
                handle.extractall(staging)
                os.replace(staging, extract)
            finally:
                if staging.exists() or staging.is_symlink():
                    shutil.rmtree(staging)
    _verify_uv_cache_directory(root)


def _verify_uv_cache_directory(root: Path) -> None:
    """Use the extracted cache itself for an offline uv probe."""

    uv = root / UV_RELATIVE
    if uv.is_symlink() or not uv.is_file():
        raise RuntimeError("pinned offline uv binary is missing")
    result = subprocess.run(
        [str(uv), "cache", "dir"],
        cwd=root,
        check=True,
        capture_output=True,
        env=_runtime_env(root),
    )
    reported = Path(result.stdout.decode("utf-8").strip()).resolve()
    expected = (root / UV_CACHE_RELATIVE).resolve()
    if reported != expected:
        raise RuntimeError("uv cache directory is not the authenticated extracted root")
    probe = subprocess.run(
        [
            str(uv),
            "run",
            "--offline",
            "--no-project",
            "python",
            "-c",
            "import sys; print(sys.version_info[:3])",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        env=_runtime_env(root),
    )
    if not probe.stdout.strip():
        raise RuntimeError("offline uv cache probe produced no interpreter evidence")


def _materialize_merged_model(root: Path) -> None:
    output = Path(VLLM_MODEL)
    if output.exists() or output.is_symlink():
        raise RuntimeError("merged model output already exists")
    _run(
        root,
        "scripts/merge_stage_c_warmstart.py",
        "--model",
        "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
        "--adapter",
        "/workspace/redco/.runtime/stage-d1/adapter",
        "--output",
        VLLM_MODEL,
        "--manifest",
        f"/workspace/redco/{MERGED_MODEL_MANIFEST}",
    )
    manifest = root / MERGED_MODEL_MANIFEST
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeError("merged model manifest was not produced")


def _start_vllm(root: Path) -> subprocess.Popen[bytes]:
    uv = root / UV_RELATIVE
    return subprocess.Popen(
        [
            str(uv),
            "run",
            "--offline",
            "--no-project",
            "vllm",
            "serve",
            VLLM_MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(VLLM_PORT),
            "--disable-log-requests",
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_runtime_env(root),
    )


def _wait_for_vllm(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("owned vLLM process exited before health")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError("owned vLLM process did not become healthy")


def _stop_vllm(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=TERMINATION_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def bootstrap(
    root: Path,
    *,
    observation: Path,
    pod_observation: Path,
    capability: Path,
    capability_signature: Path,
    execute: bool,
) -> None:
    if not execute:
        raise ValueError("remote bootstrap requires explicit execution mode")
    from redco.analysis import stage_d_v13_support_launch as launch

    if observation != root / launch.LAUNCH_PRIME_OBSERVATION_RELATIVE:
        raise ValueError("remote bootstrap requires the fixed Prime observation path")
    if pod_observation != root / launch.LAUNCH_POD_OBSERVATION_RELATIVE:
        raise ValueError("remote bootstrap requires the fixed pod observation path")
    if capability != root / launch.LAUNCH_HANDOFF_RELATIVE:
        raise ValueError("remote bootstrap requires the fixed execute handoff path")
    if capability_signature != root / launch.LAUNCH_HANDOFF_SIGNATURE_RELATIVE:
        raise ValueError("remote bootstrap requires the fixed execute handoff signature path")

    launch.verify_launch_bundle(root, require_post_commit=True)
    from redco.analysis.stage_d_v13_support_launch_runtime import (
        authorize_handoff_before_runtime,
    )

    authorize_handoff_before_runtime(
        root,
        observation=observation,
        capability=capability,
        signature=capability_signature,
    )
    _prepare_offline_runtime(root)
    _verify_assets(root)
    runtime_probe = _verify_runtime_versions(root)
    _materialize_merged_model(root)
    vllm = _start_vllm(root)
    try:
        _wait_for_vllm(vllm)
        value = capture_pod_runtime_observation(
            base_url=VLLM_BASE_URL,
            asset_paths=required_asset_paths(root),
            runtime_probe=runtime_probe,
        )
        pod_observation.parent.mkdir(parents=True, exist_ok=True)
        if pod_observation.exists() or pod_observation.is_symlink():
            raise RuntimeError("pod observation already exists")
        temporary = pod_observation.with_name(f".{pod_observation.name}.tmp")
        if temporary.exists() or temporary.is_symlink():
            raise RuntimeError("pod observation temporary path already exists")
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, pod_observation)
        try:
            descriptor = os.open(pod_observation.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
        _run(
            root,
            "scripts/run_stage_d_v13_support.py",
            "--verify-only",
            "--repository",
            str(root),
        )
        _run(
            root,
            "scripts/run_stage_d_v13_support.py",
            "--preflight-only",
            "--repository",
            str(root),
            "--preflight-observation",
            str(observation),
            "--pod-runtime-observation",
            str(pod_observation),
            "--capability",
            str(capability),
            "--capability-signature",
            str(capability_signature),
        )
        from redco.analysis.stage_d_v13_support_launch_runtime import execute_support_once

        execute_support_once(
            root,
            preflight_observation=observation,
            pod_runtime_observation=pod_observation,
            capability=capability,
            capability_signature=capability_signature,
        )
    finally:
        _stop_vllm(vllm)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--pod-observation", type=Path, required=True)
    parser.add_argument("--capability", type=Path, required=True)
    parser.add_argument("--capability-signature", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    bootstrap(
        args.repository.resolve(),
        observation=args.observation.resolve(),
        pod_observation=args.pod_observation.resolve(),
        capability=args.capability.resolve(),
        capability_signature=args.capability_signature.resolve(),
        execute=args.execute,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
