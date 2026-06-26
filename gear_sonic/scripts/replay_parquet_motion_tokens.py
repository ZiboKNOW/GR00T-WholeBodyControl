#!/usr/bin/env python3
"""Replay LeRobot motion tokens through the native WholeBodyControl decoder.

This script replaces ``run_vla_inference.py`` when the action source is an
existing parquet episode. It publishes protocol-v4 ``token_state`` messages to
the C++ deploy process; decoding, MuJoCo control, camera rendering, and video
recording remain in GR00T-WholeBodyControl.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import tyro
import zmq

from gear_sonic.scripts.run_vla_inference import (
    LATENT_INITIAL_MOTION_TOKEN,
    pack_latent_action_message,
)
from gear_sonic.utils.data_collection.keyboard_subscriber import DEFAULT_ZMQ_KEYBOARD_PORT
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message


@dataclass
class ReplayConfig:
    parquet: Path
    """Path to data/chunk-XXX/episode_XXXXXX.parquet."""

    action_zmq_host: str = "localhost"
    action_zmq_port: int = 5556
    """ZMQ endpoint consumed by gear_sonic_deploy."""

    keyboard_zmq_host: str = "*"
    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """Optional keyboard PUB endpoint for run_data_exporter recording keys."""

    rate_hz: float = 50.0
    start_frame: int = 0
    max_frames: int = -1
    loop: bool = False
    quantize_motion_token_on_publish: bool = False

    start_control: bool = True
    """Send deploy command k-equivalent before replay."""

    initial_pose_seconds: float = 2.0
    """Publish latent initial pose before replay; <=0 disables it."""

    record: bool = False
    """Send c before replay and s after replay to run_data_exporter."""

    record_warmup_seconds: float = 0.5
    stop_control_after_replay: bool = False
    print_every: int = 50


def _load_motion_tokens(path: Path) -> np.ndarray:
    df = pd.read_parquet(path)
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index", kind="stable")
    if "action.motion_token" not in df.columns:
        raise KeyError(
            f"{path} does not contain action.motion_token. "
            f"Available columns: {list(df.columns)}"
        )
    tokens = np.asarray(df["action.motion_token"].to_list(), dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != 64:
        raise ValueError(f"Expected tokens with shape [T, 64], got {tokens.shape}")
    return tokens


def _bind_pub(ctx: zmq.Context, endpoint: str) -> zmq.Socket:
    sock = ctx.socket(zmq.PUB)
    sock.bind(endpoint)
    # Give SUB sockets a moment to connect; otherwise the first PUB messages can vanish.
    time.sleep(0.5)
    return sock


def _send_keyboard(keyboard_sock: zmq.Socket | None, key: str) -> None:
    if keyboard_sock is None:
        return
    keyboard_sock.send_string(key)
    print(f"[replay] keyboard '{key}'")


def main(config: ReplayConfig) -> None:
    parquet = config.parquet.expanduser().resolve()
    if not parquet.is_file():
        raise FileNotFoundError(parquet)

    tokens = _load_motion_tokens(parquet)
    start = max(0, int(config.start_frame))
    end = tokens.shape[0] if config.max_frames < 0 else min(tokens.shape[0], start + config.max_frames)
    tokens = tokens[start:end]
    if tokens.size == 0:
        raise ValueError(f"No frames selected from {parquet}")

    ctx = zmq.Context()
    action_endpoint = f"tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    action_sock = _bind_pub(ctx, action_endpoint)
    keyboard_sock = None
    if config.record:
        keyboard_endpoint = f"tcp://{config.keyboard_zmq_host}:{config.keyboard_zmq_port}"
        keyboard_sock = _bind_pub(ctx, keyboard_endpoint)
    period = 1.0 / float(config.rate_hz)

    print(f"[replay] parquet: {parquet}")
    print(f"[replay] action endpoint: {action_endpoint}")
    print(f"[replay] frames: {tokens.shape[0]}, rate: {config.rate_hz:.2f} Hz")

    try:
        if config.start_control:
            action_sock.send(build_command_message(start=True, stop=False, planner=True))
            print("[replay] sent deploy start command (planner mode)")
            time.sleep(0.2)

        if config.initial_pose_seconds > 0:
            init_frames = max(1, int(round(config.initial_pose_seconds * config.rate_hz)))
            init_msg = pack_latent_action_message(
                LATENT_INITIAL_MOTION_TOKEN,
                np.array([0], dtype=np.int64),
                quantize=config.quantize_motion_token_on_publish,
            )
            for _ in range(init_frames):
                action_sock.send(init_msg)
                time.sleep(period)
            action_sock.send(build_command_message(start=True, stop=False, planner=False))
            print("[replay] sent latent initial pose and switched to pose mode")

        if config.record:
            _send_keyboard(keyboard_sock, "c")
            time.sleep(max(0.0, float(config.record_warmup_seconds)))

        frame_counter = 0
        while True:
            for token in tokens:
                t0 = time.monotonic()
                msg = pack_latent_action_message(
                    token,
                    np.array([frame_counter], dtype=np.int64),
                    quantize=config.quantize_motion_token_on_publish,
                )
                action_sock.send(msg)
                if config.print_every > 0 and frame_counter % config.print_every == 0:
                    print(f"[replay] sent frame {frame_counter}")
                frame_counter += 1
                remaining = period - (time.monotonic() - t0)
                if remaining > 0:
                    time.sleep(remaining)
            if not config.loop:
                break

        if config.record:
            _send_keyboard(keyboard_sock, "s")

        if config.stop_control_after_replay:
            action_sock.send(build_command_message(start=False, stop=True, planner=False))
            print("[replay] sent deploy stop command")
    finally:
        action_sock.close()
        if keyboard_sock is not None:
            keyboard_sock.close()
        ctx.term()


if __name__ == "__main__":
    main(tyro.cli(ReplayConfig))
