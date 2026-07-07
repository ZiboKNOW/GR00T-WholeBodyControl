"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs an Isaac-GR00T VLA policy against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ/TCP via ComposedCameraClientSensor
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Async timing follows HDMI ``render_vla_asyn.py`` sim action-slot scheduling.
No camera delay or latency compensation is enabled by default.

Uses the Isaac-GR00T PolicyClient (ZMQ REQ/REP) to communicate with a
running PolicyServer.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> blend smoothly to initial pose (or snap if no prior token) and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import time

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
from gear_sonic.utils.inference.sim2sim_async_timing import (
    Sim2SimAsyncTimingConfig,
    Sim2SimAsyncVlaLoop,
    Sim2SimCameraTimeline,
)
from gear_sonic.utils.inference.vla_utils import (
    build_vla_state_from_configuration,
    concat_action,
    embodiment_uses_wrist_cameras,
    is_no_hand_embodiment,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)

INSPIRE_HAND_DOF = 6
INSPIRE_OPEN_HAND = np.array([-0.1, -0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
INSPIRE_CLOSED_HAND = np.array([1.3, 0.6, 1.7, 1.7, 1.7, 1.7], dtype=np.float32)
MOTION_TOKEN_QUANTIZE_STEP = 0.0625


def quantize_motion_token(motion_token: np.ndarray) -> np.ndarray:
    """Snap motion tokens to the SONIC FSQ grid used during HDMI training."""
    token = np.asarray(motion_token, dtype=np.float32)
    return np.round(token / MOTION_TOKEN_QUANTIZE_STEP) * MOTION_TOKEN_QUANTIZE_STEP


def _validate_hand_action(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != INSPIRE_HAND_DOF:
        raise ValueError(
            f"{name} must have last dimension {INSPIRE_HAND_DOF}, got {value.shape}"
        )
    return value


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    host: str = "localhost"
    port: int = 5550

    action_publish_rate: int = 50
    """Logical sim action slot rate (Hz); matches render ``vla.action_publish_rate_hz``."""

    action_horizon: int = 40

    inference_rate_hz: float = 2.5
    """VLA forward-pass rate (Hz); matches render ``vla.inference_rate_hz``."""

    rate: float | None = None
    """Deprecated alias for ``inference_rate_hz`` (e.g. 5.0 means 5 Hz)."""

    execution_horizon: int | None = None
    action_step: int = 0

    camera_delay_ms: float = 0.0
    """Extra camera pipeline delay (ms). Default 0: use latest frame, no injected lag."""

    camera_delay_history_steps: int | None = None

    async_timing_mode: str = "sim"
    latency_compensation: bool = False
    """Skip stale chunk indices after inference; set false to avoid pending release waits."""
    latency_compensation_source: str = "sim"
    wait_for_first_action: bool = False
    first_action_timeout_s: float = 0.0
    log_async_chunks: bool = True
    log_pending_action_slots: bool = False

    quantize_motion_token_on_publish: bool = False
    """Match render ``disable_decoder_quantization=true`` when False."""

    camera_host: str = "localhost"
    camera_port: int = 5555

    state_zmq_host: str = "localhost"
    state_zmq_port: int = 5557

    action_zmq_host: str = "localhost"
    action_zmq_port: int = 5556

    keyboard_zmq_host: str = "localhost"
    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT

    embodiment_tag: str = "unitree_g1_sonic_no_hand_wo_wrist"
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    initial_pose_blend_duration: float = 1.0
    """Duration (seconds) for smooth interpolation to initial pose. The robot
    blends from its current motion token to the initial pose token over this
    period. Set to 0 to snap instantly (no blend)."""

    verbose_timing: bool = False

    def resolved_inference_rate_hz(self) -> float:
        if self.rate is not None:
            return float(self.rate)
        return float(self.inference_rate_hz)

    def timing_config(self) -> Sim2SimAsyncTimingConfig:
        return Sim2SimAsyncTimingConfig(
            action_publish_rate_hz=float(self.action_publish_rate),
            inference_rate_hz=self.resolved_inference_rate_hz(),
            action_horizon=int(self.action_horizon),
            action_step=int(self.action_step),
            execution_horizon=self.execution_horizon,
            camera_delay_ms=float(self.camera_delay_ms),
            camera_delay_history_steps=self.camera_delay_history_steps,
            async_timing_mode=str(self.async_timing_mode),
            latency_compensation=bool(self.latency_compensation),
            latency_compensation_source=str(self.latency_compensation_source),
            wait_for_first_action=bool(self.wait_for_first_action),
            first_action_timeout_s=float(self.first_action_timeout_s),
            log_async_chunks=bool(self.log_async_chunks),
            log_pending_action_slots=bool(self.log_pending_action_slots),
        )


def print_green(x):
    print(f"\033[92m{x}\033[0m")


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
    *,
    quantize: bool = True,
) -> bytes:
    motion_token = np.asarray(motion_token, dtype=np.float32)
    if quantize:
        motion_token = quantize_motion_token(motion_token)
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


def _read_camera_images(
    camera_subscriber,
) -> tuple[dict[str, np.ndarray] | None, float | None]:
    """Return the latest camera frames and their MuJoCo sim timestamp.

    The image publisher stamps every frame with ``mj_data.time`` under
    ``timestamps``; that real sim time is the master clock for the action
    timeline, so it must be propagated rather than discarded.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        return None, None
    images = camera_msg.get("images", {})
    if not images:
        return None, None

    timestamps = camera_msg.get("timestamps", {}) or {}
    frame_sim_time_s: float | None = None
    if "ego_view" in timestamps:
        frame_sim_time_s = float(timestamps["ego_view"])
    elif timestamps:
        frame_sim_time_s = float(next(iter(timestamps.values())))

    out = {
        str(key): np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
        for key, img in images.items()
    }
    return out, frame_sim_time_s


def prepare_observation_from_buffers(
    images: dict[str, np.ndarray],
    state_msg: dict,
    robot_model,
    language_prompt: str,
    embodiment_tag: str,
    *,
    sim_time_s: float,
    no_hand: bool = False,
) -> dict:
    if "ego_view" not in images:
        raise KeyError("Camera images must include 'ego_view'")

    body_q = np.asarray(state_msg["body_q"], dtype=np.float32)
    if body_q.shape[-1] != 29:
        raise ValueError(f"body_q must have shape [29], got {body_q.shape}")

    video = {"ego_view": images["ego_view"][np.newaxis, np.newaxis]}
    if embodiment_uses_wrist_cameras(embodiment_tag):
        for wrist_key in ("left_wrist", "right_wrist"):
            if wrist_key not in images:
                raise ValueError(
                    f"{embodiment_tag} requires camera '{wrist_key}' "
                    "(enable wrist cameras in run_sim_loop.py or use "
                    "unitree_g1_sonic_no_hand_wo_wrist for ego_view only)"
                )
            video[wrist_key] = images[wrist_key][np.newaxis, np.newaxis]

    if not no_hand:
        left_hand_q = np.asarray(state_msg["left_hand_q"], dtype=np.float32)
        right_hand_q = np.asarray(state_msg["right_hand_q"], dtype=np.float32)
        whole_q = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_q,
            left_hand_actuated_joint_values=left_hand_q,
            right_hand_actuated_joint_values=right_hand_q,
        )
    else:
        whole_q = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body_q,
        )

    observation = {
        "video": video,
        "state": build_vla_state_from_configuration(
            robot_model,
            whole_q,
            include_hands=not no_hand,
        ),
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "timestamps": np.asarray(sim_time_s, dtype=np.float64),
    }
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]
    return observation


def build_inference_observation(
    camera_timeline: Sim2SimCameraTimeline,
    state_subscriber: ZMQStateSubscriber,
    robot_model,
    language_prompt: str,
    embodiment_tag: str,
    *,
    no_hand: bool,
) -> dict | None:
    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        return None
    try:
        images = camera_timeline.delayed_images()
    except RuntimeError:
        return None
    camera_meta = camera_timeline.last_delay_metadata
    return prepare_observation_from_buffers(
        images=images,
        state_msg=state_msg,
        robot_model=robot_model,
        language_prompt=language_prompt,
        embodiment_tag=embodiment_tag,
        sim_time_s=float(camera_meta.get("selected_time_s", 0.0)),
        no_hand=no_hand,
    )


def run_policy_inference_and_process(policy, observation, robot_model):
    try:
        action, _info = policy.get_action(observation)
        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        return concat_action(robot_model, action)
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _extract_hand_targets(processed_action: dict, index: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    left = processed_action.get("left_hand_joints")
    if left is None:
        left = processed_action.get("action.left_hand_joints")
    right = processed_action.get("right_hand_joints")
    if right is None:
        right = processed_action.get("action.right_hand_joints")
    if left is None and right is None:
        return None, None

    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.ndim == 3:
        left = left[0]
    if right.ndim == 3:
        right = right[0]
    if left.ndim == 2:
        left = left[min(index, left.shape[0] - 1)]
    if right.ndim == 2:
        right = right[min(index, right.shape[0] - 1)]
    return (
        _validate_hand_action("left_hand_joints", left),
        _validate_hand_action("right_hand_joints", right),
    )


def main(config: InferenceConfig):
    pause_loop = True
    no_hand = is_no_hand_embodiment(config.embodiment_tag)
    timing_cfg = config.timing_config()
    camera_timeline = Sim2SimCameraTimeline(timing_cfg)
    async_loop = Sim2SimAsyncVlaLoop(timing_cfg, camera_timeline)

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    from gr00t.policy.server_client import PolicyClient

    policy = PolicyClient(host=config.host, port=config.port)

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if policy.ping():
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
    print_green(f"Using embodiment tag: {config.embodiment_tag}")
    print_green(
        "[sim2sim][vla] async timing: "
        f"mode={timing_cfg.async_timing_mode}, "
        f"action_publish_rate={timing_cfg.action_publish_rate_hz:.3f}Hz, "
        f"inference_rate={timing_cfg.inference_rate_hz:.3f}Hz, "
        f"inference_interval_actions={timing_cfg.inference_interval_actions}, "
        f"execution_horizon={timing_cfg.execution_horizon}, "
        f"camera_delay={timing_cfg.camera_delay_s * 1000.0:.1f}ms, "
        f"latency_compensation={timing_cfg.latency_compensation}, "
        f"quantize_on_publish={config.quantize_motion_token_on_publish}"
    )
    if no_hand:
        print_green(
            "No-hand mode: VLA publishes motion_token only; "
            "sim hands stay at C++ deploy default pose."
        )

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )
    telemetry = Telemetry(window_size=100)
    loop_period = 1.0 / float(config.action_publish_rate)

    cpp_loop_running = False
    cpp_mode = "OFF"
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False
    zmq_frame_counter = 0
    language_prompt_ref: list[str] = [config.prompt]
    latest_camera_images: dict[str, np.ndarray] | None = None
    latest_frame_sim_time_s: float | None = None
    missing_sim_time_warned = False
    last_sent_motion_token: np.ndarray | None = None

    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    async_loop.start_worker(
        lambda observation: run_policy_inference_and_process(policy, observation, robot_model)
    )

    def publish_initial_pose():
        nonlocal last_sent_motion_token
        print("Moving to initial pose")
        left_hand = (
            INSPIRE_CLOSED_HAND.copy()
            if initial_pose_left_hand_closed
            else INSPIRE_OPEN_HAND.copy()
        )
        right_hand = (
            INSPIRE_CLOSED_HAND.copy()
            if initial_pose_right_hand_closed
            else INSPIRE_OPEN_HAND.copy()
        )
        if no_hand:
            zmq_message = pack_latent_action_message(
                motion_token=LATENT_INITIAL_MOTION_TOKEN,
                frame_index=np.array([0], dtype=np.int64),
                quantize=config.quantize_motion_token_on_publish,
            )
        else:
            zmq_message = pack_latent_action_message(
                motion_token=LATENT_INITIAL_MOTION_TOKEN,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
                quantize=config.quantize_motion_token_on_publish,
            )
        zmq_socket.send(zmq_message)
        last_sent_motion_token = LATENT_INITIAL_MOTION_TOKEN.copy()
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def blend_to_initial_pose(duration_s: float) -> bool:
        """Smoothly interpolate from the last sent motion token to the initial pose.

        Linearly blends over ``duration_s`` seconds at the action publish rate,
        sending intermediate tokens each loop iteration. Returns True if blend
        was performed, False if skipped (no previous token available).
        """
        nonlocal last_sent_motion_token
        if last_sent_motion_token is None:
            print("No previous motion token — snapping to initial pose instead.")
            publish_initial_pose()
            return False

        start_token = last_sent_motion_token.copy()
        target_token = LATENT_INITIAL_MOTION_TOKEN.copy()
        num_steps = max(1, round(config.action_publish_rate * duration_s))
        step_period = 1.0 / config.action_publish_rate

        left_hand = (
            INSPIRE_CLOSED_HAND.copy()
            if initial_pose_left_hand_closed
            else INSPIRE_OPEN_HAND.copy()
        )
        right_hand = (
            INSPIRE_CLOSED_HAND.copy()
            if initial_pose_right_hand_closed
            else INSPIRE_OPEN_HAND.copy()
        )

        print(
            f"Blending to initial pose over {duration_s:.2f}s "
            f"({num_steps} steps at {config.action_publish_rate} Hz)"
        )

        for step in range(num_steps):
            t_step_start = time.monotonic()
            alpha = (step + 1) / num_steps
            blended_token = ((1.0 - alpha) * start_token + alpha * target_token).astype(
                np.float32
            )
            if no_hand:
                zmq_message = pack_latent_action_message(
                    motion_token=blended_token,
                    frame_index=np.array([0], dtype=np.int64),
                    quantize=config.quantize_motion_token_on_publish,
                )
            else:
                zmq_message = pack_latent_action_message(
                    motion_token=blended_token,
                    frame_index=np.array([0], dtype=np.int64),
                    left_hand_joints=left_hand,
                    right_hand_joints=right_hand,
                    quantize=config.quantize_motion_token_on_publish,
                )
            zmq_socket.send(zmq_message)
            last_sent_motion_token = blended_token.copy()

            elapsed = time.monotonic() - t_step_start
            remaining = step_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print_green("Initial pose blend complete.")
        return True

    def send_cpp_control_command(start: bool, planner: bool = False):
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            cpp_loop_running = start
            cpp_mode = "PLANNER" if (start and planner) else ("POSE" if start else "OFF")
            print_green(
                f"Sent ZMQ command: {'start' if start else 'stop'} control loop "
                f"({'planner' if planner else 'pose'} mode)"
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to send control command: {e}")
            return False

    def reset_async_state(clear_last_token: bool = False):
        nonlocal zmq_frame_counter, latest_camera_images, latest_frame_sim_time_s
        nonlocal last_sent_motion_token
        async_loop.reset()
        zmq_frame_counter = 0
        latest_camera_images = None
        latest_frame_sim_time_s = None
        if clear_last_token:
            last_sent_motion_token = None

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal zmq_frame_counter, last_sent_motion_token

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX) :]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            if cpp_loop_running and cpp_mode == "PLANNER":
                send_cpp_control_command(start=True, planner=False)
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")

            pause_loop = True
            if config.initial_pose_blend_duration > 0 and last_sent_motion_token is not None:
                blend_to_initial_pose(config.initial_pose_blend_duration)
            else:
                publish_initial_pose()

            reset_async_state()
            print("Cleared async timing state and reset frame counter")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if not pause_loop:
                reset_async_state()
                print(
                    "Cleared async timing state — wait for a fresh inference "
                    "(press c before p when recording so suitcase stays upright)"
                )
        elif key == "k":
            if cpp_loop_running:
                send_cpp_control_command(start=False, planner=(cpp_mode == "PLANNER"))
            else:
                send_cpp_control_command(start=True, planner=True)
                if pause_loop:
                    print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            if pause_loop:
                time.sleep(0.05)
                continue

            camera_images, frame_sim_time_s = _read_camera_images(camera_subscriber)
            if camera_images is not None:
                latest_camera_images = camera_images
                if frame_sim_time_s is not None:
                    latest_frame_sim_time_s = frame_sim_time_s
                elif not missing_sim_time_warned:
                    print(
                        "[Warning] Camera frames carry no sim timestamp; falling back to "
                        "wall-clock pacing. Action timeline may drift from MuJoCo physics."
                    )
                    missing_sim_time_warned = True

            if latest_camera_images is not None:
                # Real MuJoCo sim time is the master clock. If frames lack a
                # timestamp (older publisher), degrade gracefully to wall clock.
                clock_s = (
                    latest_frame_sim_time_s
                    if latest_frame_sim_time_s is not None
                    else time.monotonic()
                )
                camera_timeline.tick(latest_camera_images, clock_s)

            # Advance the action timeline by however many action_dt slots of real
            # sim time have elapsed — not by wall-clock loop iterations. When the
            # simulator runs slower than real time this naturally publishes fewer
            # slots, keeping the latent trajectory in lockstep with physics.
            slots_due = 0
            if camera_timeline.has_frames():
                slots_due = async_loop.sim_publish_slots_due(camera_timeline.sim_time_s)

            for _ in range(slots_due):
                async_loop.consume_async_result(
                    allow_initial_immediate_activation=(
                        async_loop.published_action_count == 0
                        and not async_loop.has_action_chunk()
                    )
                )
                async_loop.activate_pending_action_chunk_if_ready()

                if async_loop.should_start_async_inference():
                    observation = build_inference_observation(
                        camera_timeline=camera_timeline,
                        state_subscriber=state_subscriber,
                        robot_model=robot_model,
                        language_prompt=language_prompt_ref[0],
                        embodiment_tag=config.embodiment_tag,
                        no_hand=no_hand,
                    )
                    if observation is not None:
                        async_loop.start_async_inference(observation)

                with telemetry.timer("total_loop"):
                    if not async_loop.on_action_slot():
                        continue

                    motion_token = async_loop.current_motion_token()
                    if motion_token is not None:
                        chunk = async_loop.active_action_chunk()
                        assert chunk is not None
                        action_index = async_loop.current_action_index()

                        frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                        zmq_frame_counter += 1

                        if no_hand:
                            zmq_message = pack_latent_action_message(
                                motion_token,
                                frame_index,
                                quantize=config.quantize_motion_token_on_publish,
                            )
                        else:
                            left_hand, right_hand = _extract_hand_targets(chunk, action_index)
                            zmq_message = pack_latent_action_message(
                                motion_token,
                                frame_index,
                                left_hand_joints=left_hand,
                                right_hand_joints=right_hand,
                                quantize=config.quantize_motion_token_on_publish,
                            )
                        zmq_socket.send(zmq_message)
                        last_sent_motion_token = motion_token.copy()
                        if zmq_frame_counter % 50 == 0:
                            print_green(
                                f"ZMQ: Sent latent action - frame: {frame_index[0]}, "
                                f"sim_time={camera_timeline.sim_time_s:.3f}s, "
                                f"action_index={action_index}, "
                                f"token shape: {motion_token.shape}"
                            )
                        async_loop.advance_after_publish()
                    else:
                        async_loop.advance_empty_action_slot()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")
    finally:
        async_loop.close()
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
