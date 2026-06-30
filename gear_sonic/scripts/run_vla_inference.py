"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + optional hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> send initial pose and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose (hand-enabled policies only)
  ]  -> toggle right hand open/closed for initial pose (hand-enabled policies only)
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any

import numpy as np
import tyro
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)

INSPIRE_HAND_DOF = 6
INSPIRE_OPEN_HAND = np.array([-0.1, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
INSPIRE_CLOSED_HAND = np.array([1.3, 0.6, 1.7, 1.7, 1.7, 1.7], dtype=np.float32)

DEFAULT_EMBODIMENT_TAG = "unitree_g1_sonic_no_hand_wo_wrist"
DEFAULT_MOTION_TOKEN_REPLAY_PATH = (
    "/home/ubuntu/DATA4/zzb/HDMI/groot_data/move_suitcase_0628/data/chunk-000/"
    "episode_000000.parquet"
)
MOTION_TOKEN_MODE_VLA = "vla"
MOTION_TOKEN_MODE_PARQUET_REPLAY = "parquet_replay"
MOTION_TOKEN_DIM = 64

SONIC_BODY_STATE_SLICES = {
    "left_leg": slice(0, 6),
    "right_leg": slice(6, 12),
    "waist": slice(12, 15),
    "left_arm": slice(15, 22),
    "right_arm": slice(22, 29),
}

SONIC_HAND_STATE_KEYS = {"left_hand", "right_hand"}
SONIC_HAND_ACTION_KEYS = {"left_hand_joints", "right_hand_joints"}

SONIC_STATE_KEYS_WITH_HANDS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "projected_gravity",
]
SONIC_STATE_KEYS_NO_HANDS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "projected_gravity",
]
SONIC_ACTION_KEYS_WITH_HANDS = [
    "motion_token",
    "left_hand_joints",
    "right_hand_joints",
]
SONIC_ACTION_KEYS_NO_HANDS = ["motion_token"]


def _validate_hand_action(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != INSPIRE_HAND_DOF:
        raise ValueError(
            f"{name} must have last dimension {INSPIRE_HAND_DOF}, got {value.shape}"
        )
    return value


def _normalize_embodiment_tag(tag: str) -> str:
    return tag.strip().lower()


def _keys_from_modality(modality_config: Any, modality: str) -> list[str]:
    config = modality_config.get(modality)
    if config is None:
        return []
    if hasattr(config, "modality_keys"):
        return list(config.modality_keys)
    if isinstance(config, dict):
        return list(config.get("modality_keys", []))
    raise TypeError(f"Unsupported {modality} modality config type: {type(config)}")


def _fallback_modality_keys(embodiment_tag: str) -> dict[str, list[str]]:
    tag = _normalize_embodiment_tag(embodiment_tag)
    has_hands = "no_hand" not in tag
    has_wrist_cameras = tag in {
        "unitree_g1_sonic_no_hand",
        "g1_sonic_inspire_wrist",
    }

    video_keys = ["ego_view"]
    if has_wrist_cameras:
        video_keys.extend(["left_wrist", "right_wrist"])

    return {
        "video": video_keys,
        "state": SONIC_STATE_KEYS_WITH_HANDS if has_hands else SONIC_STATE_KEYS_NO_HANDS,
        "action": SONIC_ACTION_KEYS_WITH_HANDS if has_hands else SONIC_ACTION_KEYS_NO_HANDS,
    }


def _resolve_modality_keys(policy, embodiment_tag: str) -> dict[str, list[str]]:
    try:
        modality_config = policy.get_modality_config()
        keys = {
            "video": _keys_from_modality(modality_config, "video"),
            "state": _keys_from_modality(modality_config, "state"),
            "action": _keys_from_modality(modality_config, "action"),
        }
        if all(keys.values()):
            return keys
        print(
            "[Warning] PolicyServer returned incomplete modality config; "
            "falling back to embodiment-tag defaults."
        )
    except Exception as e:
        print(
            f"[Warning] Could not query PolicyServer modality config ({e}); "
            "falling back to embodiment-tag defaults."
        )
    return _fallback_modality_keys(embodiment_tag)


def _get_optional_action_field(action_dict: dict, key: str):
    value = action_dict.get(key)
    if value is not None:
        return value
    return action_dict.get(f"action.{key}")


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host."""

    camera_port: int = 5555
    """Camera server port."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = DEFAULT_EMBODIMENT_TAG
    """Embodiment tag for policy inference."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Motion token source
    motion_token_mode: str = MOTION_TOKEN_MODE_VLA
    """Motion token source: 'vla' or 'parquet_replay'."""

    motion_token_replay_path: str = DEFAULT_MOTION_TOKEN_REPLAY_PATH
    """Parquet file to replay action.motion_token from in parquet_replay mode."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[6]`` or ``[1, 6]``, optional.
        right_hand_joints: Shape ``[6]`` or ``[1, 6]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = _validate_hand_action("left_hand_joints", left_hand_joints)
        if left_hand_joints.ndim == 1:
            left_hand_joints = left_hand_joints.reshape(1, INSPIRE_HAND_DOF)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = _validate_hand_action("right_hand_joints", right_hand_joints)
        if right_hand_joints.ndim == 1:
            right_hand_joints = right_hand_joints.reshape(1, INSPIRE_HAND_DOF)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


def _load_replay_motion_tokens(path: str) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(path)
    key = "action.motion_token"
    if key not in df.columns:
        raise KeyError(
            f"Replay parquet must contain column '{key}'. Available columns: {list(df.columns)}"
        )

    tokens = np.stack(df[key].to_numpy()).astype(np.float32)
    if tokens.ndim != 2 or tokens.shape[-1] != MOTION_TOKEN_DIM:
        raise ValueError(
            f"Replay motion tokens must have shape [N, {MOTION_TOKEN_DIM}], got {tokens.shape}"
        )
    return tokens


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    video_keys: list[str],
    state_keys: list[str],
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    body_q = np.asarray(state_msg["body_q"], dtype=np.float32)
    if body_q.shape[-1] != 29:
        raise ValueError(f"body_q must have shape [29], got {body_q.shape}")

    images = camera_msg["images"]
    video = {}
    missing_video_keys = [key for key in video_keys if key not in images]
    if missing_video_keys:
        if log_errors:
            print(
                "[DEBUG] prepare_observation: waiting for required camera keys "
                f"{missing_video_keys}; available: {list(images.keys())}",
                flush=True,
            )
        return None
    for key in video_keys:
        video[key] = images[key][np.newaxis, np.newaxis]

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }
    for key in state_keys:
        if key in SONIC_BODY_STATE_SLICES:
            observation["state"][key] = body_q[SONIC_BODY_STATE_SLICES[key]][
                np.newaxis, np.newaxis
            ]

    if "left_hand" in state_keys:
        if "left_hand_q" not in state_msg:
            raise KeyError("Policy requires state.left_hand but state_msg lacks left_hand_q")
        left_hand_q = np.asarray(state_msg["left_hand_q"], dtype=np.float32)
        if left_hand_q.shape[-1] != INSPIRE_HAND_DOF:
            raise ValueError(
                f"left_hand_q must have shape [{INSPIRE_HAND_DOF}], got {left_hand_q.shape}"
            )
        observation["state"]["left_hand"] = left_hand_q[np.newaxis, np.newaxis]

    if "right_hand" in state_keys:
        if "right_hand_q" not in state_msg:
            raise KeyError("Policy requires state.right_hand but state_msg lacks right_hand_q")
        right_hand_q = np.asarray(state_msg["right_hand_q"], dtype=np.float32)
        if right_hand_q.shape[-1] != INSPIRE_HAND_DOF:
            raise ValueError(
                f"right_hand_q must have shape [{INSPIRE_HAND_DOF}], got {right_hand_q.shape}"
            )
        observation["state"]["right_hand"] = right_hand_q[np.newaxis, np.newaxis]

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    missing_state_keys = [key for key in state_keys if key not in observation["state"]]
    if missing_state_keys:
        supported_state_keys = (
            set(SONIC_BODY_STATE_SLICES) | SONIC_HAND_STATE_KEYS | {"projected_gravity"}
        )
        raise KeyError(
            f"Cannot build required policy state keys {missing_state_keys}. "
            f"Supported Sonic state keys: {sorted(supported_state_keys)}"
        )

    return observation


def run_policy_inference_and_process(policy, observation, robot_model, action_keys: list[str]):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_token = _get_optional_action_field(action, "motion_token")
        if motion_token is None:
            raise KeyError(
                f"Policy action did not include motion_token. Available keys: {list(action.keys())}"
            )
        if np.abs(np.asarray(motion_token)).max() > 1.25:
            print(
                f"[Warning] action['motion_token'] max "
                f"({np.abs(np.asarray(motion_token)).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        for key in action_keys:
            if _get_optional_action_field(action, key) is None:
                raise KeyError(
                    f"Policy action missing required field '{key}'. "
                    f"Available keys: {list(action.keys())}"
                )

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config: InferenceConfig):
    pause_loop = True

    motion_token_mode = config.motion_token_mode.strip().lower()
    valid_motion_token_modes = {
        MOTION_TOKEN_MODE_VLA,
        MOTION_TOKEN_MODE_PARQUET_REPLAY,
    }
    if motion_token_mode not in valid_motion_token_modes:
        raise ValueError(
            f"motion_token_mode must be one of {sorted(valid_motion_token_modes)}, "
            f"got {config.motion_token_mode!r}"
        )

    replay_motion_tokens = None
    replay_token_index = 0
    if motion_token_mode == MOTION_TOKEN_MODE_PARQUET_REPLAY:
        replay_motion_tokens = _load_replay_motion_tokens(config.motion_token_replay_path)
        print_green(
            f"Loaded {len(replay_motion_tokens)} replay motion tokens from "
            f"{config.motion_token_replay_path}"
        )

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # Isaac-GR00T PolicyClient
    from gr00t.policy.server_client import PolicyClient

    n1_policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    camera_subscriber = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    modality_keys = _resolve_modality_keys(n1_policy, config.embodiment_tag)
    video_keys = modality_keys["video"]
    state_keys = modality_keys["state"]
    action_keys = modality_keys["action"]
    hand_action_enabled = bool(SONIC_HAND_ACTION_KEYS & set(action_keys))
    print_green(f"Using embodiment tag: {config.embodiment_tag}")
    print_green(f"Policy video keys: {video_keys}")
    print_green(f"Policy state keys: {state_keys}")
    print_green(f"Policy action keys: {action_keys}")

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        print("Moving to initial pose")
        left_hand = None
        right_hand = None
        if "left_hand_joints" in action_keys:
            left_hand = (
                INSPIRE_CLOSED_HAND.copy()
                if initial_pose_left_hand_closed
                else INSPIRE_OPEN_HAND.copy()
            )
        if "right_hand_joints" in action_keys:
            right_hand = (
                INSPIRE_CLOSED_HAND.copy()
                if initial_pose_right_hand_closed
                else INSPIRE_OPEN_HAND.copy()
            )
        zmq_message = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    # Async inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0

    PROMPT_MSG_PREFIX = "prompt:"

    def next_replay_motion_token() -> np.ndarray:
        nonlocal replay_token_index
        if replay_motion_tokens is None:
            raise RuntimeError("Replay motion tokens are not loaded.")
        if replay_token_index >= len(replay_motion_tokens):
            return np.zeros(MOTION_TOKEN_DIM, dtype=np.float32)
        motion_token = replay_motion_tokens[replay_token_index]
        replay_token_index += 1
        return motion_token

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter, replay_token_index

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            print("Moving to initial pose")
            zmq_frame_counter = 0
            print("Reset ZMQ frame counter")
            publish_initial_pose()
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk")
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                print("Policy loop resumed")
                if motion_token_mode == MOTION_TOKEN_MODE_PARQUET_REPLAY:
                    replay_token_index = 0
                    print_green("Reset replay motion token index")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            if not hand_action_enabled:
                print("Initial pose hand toggles are disabled for this no-hand policy.")
                return
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            if not hand_action_enabled:
                print("Initial pose hand toggles are disabled for this no-hand policy.")
                return
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                video_keys=video_keys,
                state_keys=state_keys,
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
                action_keys=action_keys,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                action_chunk_index = calculate_latency_compensated_index(
                    inference_delay, config.action_publish_rate, config.action_horizon
                )
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", '
                    f"latency: {inference_delay:.3f}s)"
                )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            should_start = should_trigger_new_inference(
                cached_chunk_exists=(cached_action_chunk is not None),
                inference_thread_running=worker_is_busy,
                time_since_last_inference=(time.monotonic() - last_inference_time),
                inference_interval=inference_interval,
            )

            if should_start:
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    hand_actions = {}
                    for hand_key in sorted(SONIC_HAND_ACTION_KEYS & set(action_keys)):
                        hand_actions[hand_key] = _validate_hand_action(
                            hand_key,
                            np.asarray(
                                get_action_field(processed_action, hand_key),
                                dtype=np.float32,
                            ),
                        )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]
                    for hand_key, hand_value in list(hand_actions.items()):
                        if hand_value.ndim == 3:
                            hand_actions[hand_key] = hand_value[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = min(action_chunk_index, horizon - 1)

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]
                    for hand_key, hand_value in list(hand_actions.items()):
                        if hand_value.ndim == 2:
                            hand_actions[hand_key] = hand_value[current_idx]

                    if motion_token_mode == MOTION_TOKEN_MODE_PARQUET_REPLAY:
                        motion_token = next_replay_motion_token()

                    # DEBUG: force published motion token to zero.
                    # motion_token = np.zeros_like(motion_token, dtype=np.float32)                
                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=hand_actions.get("left_hand_joints"),
                        right_hand_joints=hand_actions.get("right_hand_joints"),
                    )
                    zmq_socket.send(zmq_message)
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - "
                            f"frame: {frame_index[0]}, "
                            f"token shape: {motion_token.shape}"
                        )

                action_chunk_index = min(action_chunk_index + 1, config.action_horizon - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
