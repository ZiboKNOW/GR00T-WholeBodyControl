#!/usr/bin/env python3
"""
One-click sim2sim VLA recorder for the suitcase no-hand checkpoint (29-DOF G1).

Uses the unitree_ros ``g1_29dof`` rubber-hand suitcase scene with ego_view camera (640x480),
aligned with Isaac ``render_vla`` / ``unitree_g1_sonic_no_hand_wo_wrist``.

Starts all six stack components as background processes, waits for readiness,
sends keyboard commands automatically (k -> i -> p -> c -> ... -> s), then
shuts everything down and prints the recorded video paths.

Usage (from GR00T-WholeBodyControl repo root):

    python gear_sonic/scripts/launch_sim2sim_record.py

    python gear_sonic/scripts/launch_sim2sim_record.py \\
        --record-seconds 20 \\
        --model-path /home/ubuntu/DATA4/zzb/chekpoint_suitcase_final/checkpoint-25000

Prerequisites:
    - .venv_sim, .venv_inference, .venv_data_collection
    - gear_sonic_deploy built (deploy.sh)
    - Isaac-GR00T with `uv` (for PolicyServer)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ISAAC_GR00T_ROOT = REPO_ROOT.parent / "Isaac-GR00T"
ISAAC_GR00T_PYTHON = ISAAC_GR00T_ROOT / ".venv" / "bin" / "python"
HF_CACHE_ROOT = REPO_ROOT.parent / "hf_cache"
DEFAULT_KEYBOARD_PORT = 5580

# Broken http_proxy (e.g. 127.0.0.1:7890 with no daemon) breaks uv/GitHub and HF Hub.
_PROXY_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "no_proxy",
    "NO_PROXY",
)


def _bootstrap_venv() -> None:
    try:
        import tyro  # noqa: F401
        import zmq  # noqa: F401
        return
    except ImportError:
        pass

    venv_python = REPO_ROOT / ".venv_inference" / "bin" / "python"
    if not venv_python.exists():
        print(
            "ERROR: tyro/zmq not available and .venv_inference not found.\n"
            "  Run: bash install_scripts/install_inference.sh"
        )
        sys.exit(1)
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_bootstrap_venv()

import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.utils.data_collection.zmq_state_subscriber import (
    ZMQStateSubscriber,
    poll_robot_config_zmq,
)


@dataclass
class Sim2SimRecordConfig:
    """CLI for automated sim2sim recording."""

    model_path: str = "/home/ubuntu/DATA4/zzb/chekpoint_suitcase_final/checkpoint-25000"
    """Path to the finetuned Isaac-GR00T checkpoint."""

    embodiment_tag: str = "unitree_g1_sonic_no_hand_wo_wrist"
    """Embodiment tag (must match the checkpoint; ego_view only)."""

    prompt: str = "Pick up the suitcase in front of you and move it."
    """Language prompt for VLA inference and the dataset metadata."""

    record_seconds: float = 20.0
    """How long to record after pressing 'c' (seconds)."""

    gpu_id: int = 10
    """CUDA device for PolicyServer."""

    policy_host: str = "localhost"
    policy_port: int = 5550
    camera_host: str = "localhost"
    camera_port: int = 5555

    action_publish_rate: int = 50
    action_horizon: int = 40
    data_collection_frequency: int = 50

    root_output_dir: str = "/home/ubuntu/DATA4/zzb/HDMI/vla_sim_videos"
    """Directory where the LeRobot-style dataset (and mp4s) are written."""

    dataset_name: str = ""
    """Output folder name under root_output_dir (auto-generated if empty)."""

    record_wrist_cameras: bool = False
    """Also record left/right wrist camera mp4s (VLA uses ego_view only)."""

    skip_policy_server: bool = False
    """Skip launching PolicyServer (use if one is already running on policy_port)."""

    keep_processes: bool = False
    """Leave background processes running after the script finishes."""

    startup_timeout: float = 300.0
    """Max seconds to wait for PolicyServer + camera + deploy config."""

    sim_warmup_seconds: float = 2.0
    """Seconds after MuJoCo starts before k/i/p (sim2sim.md: press keys once sim is up)."""

    save_timeout: float = 60.0
    """Max seconds to wait for mp4 finalize after pressing 's'."""


def _resolve_cuda_visible_devices(config_gpu_id: int) -> str:
    return os.environ.get("CUDA_VISIBLE_DEVICES", str(config_gpu_id))


class _StreamProbes:
    """Reuse ZMQ subscribers across readiness polls (new SUB sockets miss PUB frames)."""

    def __init__(self) -> None:
        self.camera: ComposedCameraClientSensor | None = None
        self.state: ZMQStateSubscriber | None = None

    def camera_ready(self, host: str, port: int) -> bool:
        try:
            if self.camera is None:
                self.camera = ComposedCameraClientSensor(server_ip=host, port=port)
                time.sleep(1.0)
            msg = self.camera.read(blocking=False)
            return msg is not None and "ego_view" in msg.get("images", {})
        except Exception:
            return False

    def proprio_ready(self, host: str, port: int) -> bool:
        try:
            if self.state is None:
                self.state = ZMQStateSubscriber(host=host, port=port)
            msg = self.state.get_msg(clear=False)
            return msg is not None and "body_q" in msg
        except Exception:
            return False

    def close(self) -> None:
        if self.camera is not None:
            self.camera.close()
            self.camera = None


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in _PROXY_ENV_KEYS:
        env.pop(key, None)
    if HF_CACHE_ROOT.is_dir():
        env["HF_HOME"] = str(HF_CACHE_ROOT)
        env["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")
    # Isaac-GR00T patches: prefer local HF cache, skip spurious Hub calls (Cosmos/Qwen3).
    env["GROOT_HF_LOCAL_FIRST"] = "1"
    env["GROOT_PATCH_MISTRAL"] = "1"
    return env


def _shell_preamble() -> str:
    """Shell prefix: drop login-shell proxy aliases and use local HF cache."""
    parts = [
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy",
        "export GROOT_HF_LOCAL_FIRST=1",
        "export GROOT_PATCH_MISTRAL=1",
    ]
    if HF_CACHE_ROOT.is_dir():
        parts.append(f"export HF_HOME={HF_CACHE_ROOT}")
        parts.append(f"export HUGGINGFACE_HUB_CACHE={HF_CACHE_ROOT}/hub")
    return "; ".join(parts) + "; "


def _cleanup_stale_processes(ports: list[int]) -> None:
    """Kill leftover sim2sim processes that block ZMQ ports."""
    patterns = [
        "run_gr00t_server.py",
        "run_sim_loop.py",
        "run_vla_inference.py",
        "run_data_exporter.py",
        "g1_deploy_onnx_ref",
        "keyboard_publisher.py",
    ]
    for pattern in patterns:
        subprocess.run(["pkill", "-f", pattern], check=False)

    for port in ports:
        subprocess.run(["fuser", "-k", "-n", "tcp", str(port)], check=False)

    time.sleep(1.0)


class _ProcGroup:
    def __init__(self, name: str, cmd: str, log_path: Path, cwd: Path | None = None):
        self.name = name
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_path, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=str(cwd or REPO_ROOT),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_subprocess_env(),
        )

    def poll(self) -> int | None:
        return self._proc.poll()

    def terminate(self, grace_sec: float = 5.0) -> None:
        if self._proc.poll() is not None:
            self._close_log()
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            self._close_log()
            return
        deadline = time.monotonic() + grace_sec
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                break
            time.sleep(0.2)
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._close_log()

    def _close_log(self) -> None:
        if not self._log_file.closed:
            self._log_file.close()


class KeyboardAutomation:
    """Minimal ZMQ keyboard publisher (replaces interactive keyboard_publisher.py)."""

    def __init__(self, port: int = DEFAULT_KEYBOARD_PORT):
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://*:{port}")
        time.sleep(0.5)

    def send(self, key: str) -> None:
        self._pub.send_string(key)
        print(f"[keyboard] sent '{key}'")

    def close(self) -> None:
        self._pub.close()
        self._ctx.term()


def _wait_until(
    predicate,
    timeout: float,
    desc: str,
    poll: float = 0.5,
    processes: list["_ProcGroup"] | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if processes:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"{proc.name} exited early (code {proc.poll()}). "
                        f"See log: {proc.log_path}"
                    )
        if predicate():
            print(f"[ready] {desc}")
            return
        time.sleep(poll)
    raise TimeoutError(f"Timed out waiting for: {desc}")


def _policy_server_ready(host: str, port: int) -> bool:
    try:
        from gr00t.policy.server_client import PolicyClient

        client = PolicyClient(host=host, port=port, timeout_ms=2000)
        return bool(client.ping())
    except Exception:
        return False


def _robot_config_ready(host: str, port: int) -> bool:
    try:
        poll_robot_config_zmq(host, port, timeout_sec=0.5)
        return True
    except TimeoutError:
        return False


def _video_paths(dataset_root: Path) -> list[Path]:
    videos_dir = dataset_root / "videos"
    if not videos_dir.exists():
        return []
    return sorted(videos_dir.rglob("*.mp4"))


def _mp4_playable(path: Path) -> bool:
    """Return True when ffprobe can read the file (moov atom written)."""
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    result = subprocess.run(
        ["ffprobe", "-v", "error", str(path)],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _videos_finalized(dataset_root: Path, min_videos: int = 1) -> bool:
    videos = _video_paths(dataset_root)
    if len(videos) < min_videos:
        return False
    return all(_mp4_playable(path) for path in videos)


def _episode_saved(dataset_root: Path) -> bool:
    """True when mp4 moov atoms are written and parquet exists."""
    if not _videos_finalized(dataset_root):
        return False
    parquet_files = list((dataset_root / "data").rglob("*.parquet"))
    return len(parquet_files) > 0


def _mujoco_log_text(log_path: Path) -> str:
    if not log_path.is_file():
        return ""
    return log_path.read_text(errors="replace")


def _release_elastic_band(
    keyboard: KeyboardAutomation,
    mujoco_proc: "_ProcGroup",
    log_path: Path,
    timeout: float = 15.0,
) -> None:
    """Spam k until sim logs ElasticBand released (ZMQ slow-joiner safe)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mujoco_proc.poll() is not None:
            raise RuntimeError(
                f"{mujoco_proc.name} exited early (code {mujoco_proc.poll()}). "
                f"See log: {mujoco_proc.log_path}"
            )
        log_text = _mujoco_log_text(log_path)
        if "ElasticBand released" in log_text:
            print("[ready] elastic band released")
            return
        if "ZMQKeyboardSubscriber" in log_text:
            keyboard.send("k")
        time.sleep(0.15)
    raise RuntimeError(
        f"Timed out waiting for elastic band release. See log: {log_path}"
    )


def main(config: Sim2SimRecordConfig) -> None:
    if not config.dataset_name:
        config.dataset_name = f"vla_sim2sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    dataset_root = Path(config.root_output_dir) / config.dataset_name
    # Keep launch logs outside dataset_root — an empty/partial folder makes
    # run_data_exporter think it should resume from HuggingFace.
    log_dir = Path(config.root_output_dir) / "_launch_logs" / config.dataset_name
    processes: list[_ProcGroup] = []
    keyboard: KeyboardAutomation | None = None
    probes = _StreamProbes()

    print("=" * 60)
    print("  Sim2Sim one-click recorder")
    print("=" * 60)
    print(f"  Checkpoint:      {config.model_path}")
    print(f"  Embodiment:      {config.embodiment_tag}")
    print(f"  Prompt:          {config.prompt}")
    print(f"  Record:          {config.record_seconds:.1f}s")
    print(f"  Output:          {dataset_root}")
    print(f"  Launch logs:     {log_dir}")
    print("=" * 60)

    try:
        _cleanup_stale_processes(
            ports=[config.policy_port, config.camera_port, DEFAULT_KEYBOARD_PORT, 5556, 5557]
        )
        keyboard = KeyboardAutomation()

        shell = _shell_preamble()

        if not config.skip_policy_server:
            if not ISAAC_GR00T_ROOT.is_dir():
                raise FileNotFoundError(f"Isaac-GR00T not found at {ISAAC_GR00T_ROOT}")
            if not ISAAC_GR00T_PYTHON.is_file():
                raise FileNotFoundError(
                    f"Isaac-GR00T venv not found at {ISAAC_GR00T_PYTHON}. "
                    "Create it in Isaac-GR00T (do not use `uv run`, which may "
                    "re-fetch flash-attn over the network)."
                )
            policy_cmd = (
                f"{shell}"
                f"cd {ISAAC_GR00T_ROOT} && "
                f"export CUDA_VISIBLE_DEVICES={_resolve_cuda_visible_devices(config.gpu_id)} && "
                f"{ISAAC_GR00T_PYTHON} gr00t/eval/run_gr00t_server.py "
                f"--model-path {config.model_path} "
                f"--embodiment-tag {config.embodiment_tag} "
                f"--device cuda:0 "
                f"--port {config.policy_port}"
            )
            processes.append(
                _ProcGroup("policy_server", policy_cmd, log_dir / "policy_server.log")
            )
            print("[start] PolicyServer")
        else:
            print("[skip] PolicyServer (already running)")

        sim_cmd = (
            f"{shell}"
            f"cd {REPO_ROOT} && "
            f"source .venv_sim/bin/activate && "
            f"export MUJOCO_GL=egl && "
            f"python gear_sonic/scripts/run_sim_loop.py "
            f"--wbc-version nohand_suitcase --no-with-hands "
            f"--enable-offscreen --enable-image-publish --no-enable-onscreen "
            f"--ego-view-only "
            f"--camera-port {config.camera_port}"
        )

        deploy_cmd = (
            f"{shell}"
            f"cd {REPO_ROOT / 'gear_sonic_deploy'} && "
            f"source scripts/setup_env.sh >/dev/null 2>&1 && "
            f"printf '\\n' | ./deploy.sh --input-type zmq_manager sim"
        )

        inference_cmd = (
            f"{shell}"
            f"cd {REPO_ROOT} && "
            f"source .venv_inference/bin/activate && "
            f"python gear_sonic/scripts/run_vla_inference.py "
            f"--host {config.policy_host} "
            f"--port {config.policy_port} "
            f"--embodiment-tag {config.embodiment_tag} "
            f"--prompt '{config.prompt}' "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port} "
            f"--action-publish-rate {config.action_publish_rate} "
            f"--action-horizon {config.action_horizon}"
        )

        exporter_cmd = (
            f"{shell}"
            f"cd {REPO_ROOT} && "
            f"source .venv_data_collection/bin/activate && "
            f"python gear_sonic/scripts/run_data_exporter.py "
            f"--task-prompt '{config.prompt}' "
            f"--dataset-name '{config.dataset_name}' "
            f"--root-output-dir '{config.root_output_dir}' "
            f"--data-collection-frequency {config.data_collection_frequency} "
            f"--camera-host {config.camera_host} "
            f"--camera-port {config.camera_port} "
            f"--no-text-to-speech"
        )
        if config.record_wrist_cameras:
            exporter_cmd += " --record-wrist-cameras"
        else:
            exporter_cmd += " --no-record-wrist-cameras"

        processes.append(
            _ProcGroup("data_exporter", exporter_cmd, log_dir / "data_exporter.log")
        )
        print("[start] Data exporter")

        if not config.skip_policy_server:
            _wait_until(
                lambda: _policy_server_ready(config.policy_host, config.policy_port),
                config.startup_timeout,
                f"PolicyServer on {config.policy_host}:{config.policy_port}",
                processes=processes,
            )
        else:
            _wait_until(
                lambda: _policy_server_ready(config.policy_host, config.policy_port),
                30.0,
                f"existing PolicyServer on {config.policy_host}:{config.policy_port}",
                processes=processes,
            )

        # sim2sim.md: PolicyServer -> MuJoCo -> deploy. Release elastic band with k
        # before deploy/VLA start (training scene + band diverges ~0.6s without k).
        mujoco_log = log_dir / "mujoco_sim.log"
        mujoco_proc = _ProcGroup("mujoco_sim", sim_cmd, mujoco_log)
        processes.append(mujoco_proc)
        print("[start] MuJoCo sim")
        print("[run] release elastic band (k) — sim only, VLA not started yet")
        _release_elastic_band(keyboard, mujoco_proc, mujoco_log)

        processes.append(
            _ProcGroup(
                "cpp_deploy",
                deploy_cmd,
                log_dir / "cpp_deploy.log",
                cwd=REPO_ROOT / "gear_sonic_deploy",
            )
        )
        print("[start] C++ deploy")

        processes.append(
            _ProcGroup("vla_inference", inference_cmd, log_dir / "vla_inference.log")
        )
        print("[start] VLA inference")

        _wait_until(
            lambda: _robot_config_ready(config.policy_host, 5557),
            config.startup_timeout,
            "C++ deploy robot_config ZMQ",
            processes=processes,
        )

        _wait_until(
            lambda: probes.camera_ready(config.camera_host, config.camera_port),
            config.startup_timeout,
            f"camera stream on {config.camera_host}:{config.camera_port} (sim stable)",
            processes=processes,
        )

        print("[run] control sequence: k -> i -> c -> p (record before policy)")
        keyboard.send("k")
        time.sleep(2.0)
        keyboard.send("i")
        time.sleep(2.0)
        keyboard.send("c")
        time.sleep(2.0)
        keyboard.send("p")
        time.sleep(2.0)

        _wait_until(
            lambda: probes.proprio_ready(config.policy_host, 5557),
            config.startup_timeout,
            "robot proprio state on ZMQ :5557 (after k/i/c/p)",
            processes=processes,
        )

        print(f"[run] recording for {config.record_seconds:.1f}s ...")
        time.sleep(config.record_seconds)
        keyboard.send("s")
        print("[run] waiting for data_exporter to finalize mp4 (moov atom) ...")
        _wait_until(
            lambda: _episode_saved(dataset_root),
            config.save_timeout,
            "saved episode (playable mp4 + parquet)",
            processes=[p for p in processes if p.name != "data_exporter"],
        )
        time.sleep(1.0)

        videos = _video_paths(dataset_root)
        print()
        print("=" * 60)
        print("  Recording finished")
        print("=" * 60)
        print(f"  Dataset: {dataset_root}")
        if videos:
            print("  Videos:")
            for path in videos:
                print(f"    {path}")
        else:
            print("  WARNING: no mp4 files found yet — check logs in:")
            print(f"    {log_dir}")
        print("=" * 60)

    finally:
        probes.close()
        if keyboard is not None:
            keyboard.close()
        if not config.keep_processes:
            print("[cleanup] stopping background processes ...")
            for proc in reversed(processes):
                print(f"  stopping {proc.name}")
                grace = 20.0 if proc.name == "data_exporter" else 5.0
                proc.terminate(grace_sec=grace)
        else:
            print("[keep] background processes left running")
            for proc in processes:
                print(f"  {proc.name}: log -> {proc.log_path}")


if __name__ == "__main__":
    tyro.cli(main)
