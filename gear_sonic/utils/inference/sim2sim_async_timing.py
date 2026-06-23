"""Sim2sim async VLA timing aligned with HDMI ``render_vla_asyn.py``.

Uses a discrete simulation action timeline (Hz = action publish rate) with:
  - camera history + configurable delay (default 35 ms)
  - async inference worker with pre-snapshotted observations
  - latency compensation in sim-time action slots (policy infer duration)
  - pending action chunks released after compensated action slots elapse
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

from gear_sonic.utils.inference.vla_utils import calculate_latency_compensated_index


@dataclass
class Sim2SimAsyncTimingConfig:
    """Timing knobs mirroring ``render_vla.yaml`` / ``render_vla_asyn.py`` defaults."""

    action_publish_rate_hz: float = 50.0
    """Logical sim action slot rate (matches C++ deploy control_dt=0.02)."""

    inference_rate_hz: float = 2.5
    """How often to trigger a new VLA forward pass."""

    action_horizon: int = 40
    action_step: int = 0
    execution_horizon: int | None = None

    camera_delay_ms: float = 0.0
    camera_delay_history_steps: int | None = None

    async_timing_mode: str = "sim"
    latency_compensation: bool = False
    latency_compensation_source: str = "sim"

    log_async_chunks: bool = True
    log_pending_action_slots: bool = False
    wait_for_first_action: bool = False
    first_action_timeout_s: float = 0.0

    async_queue_size: int = 1
    async_join_timeout_s: float = 2.0

    def __post_init__(self) -> None:
        self.async_timing_mode = str(self.async_timing_mode).strip().lower()
        if self.async_timing_mode not in ("sim", "wall"):
            raise ValueError(
                f"async_timing_mode must be 'sim' or 'wall', got {self.async_timing_mode!r}"
            )
        self.latency_compensation_source = str(self.latency_compensation_source).strip().lower()
        if self.latency_compensation_source not in ("sim", "infer", "wall"):
            raise ValueError(
                "latency_compensation_source must be 'sim', 'infer', or 'wall', "
                f"got {self.latency_compensation_source!r}"
            )
        if self.async_timing_mode == "sim" and self.latency_compensation_source == "wall":
            self.latency_compensation_source = "sim"

        if self.action_publish_rate_hz <= 0.0:
            raise ValueError(f"action_publish_rate_hz must be positive, got {self.action_publish_rate_hz}")
        if self.inference_rate_hz <= 0.0:
            raise ValueError(f"inference_rate_hz must be positive, got {self.inference_rate_hz}")

        self.action_dt_s = 1.0 / float(self.action_publish_rate_hz)
        self.action_timing_rate_hz = (
            float(self.action_publish_rate_hz)
            if self.async_timing_mode == "wall"
            else float(self.action_publish_rate_hz)
        )
        self.action_publish_period_s = 1.0 / float(self.action_timing_rate_hz)
        self.inference_interval_s = 1.0 / float(self.inference_rate_hz)
        self.inference_interval_actions = max(
            1,
            int(round(self.action_timing_rate_hz / float(self.inference_rate_hz))),
        )

        max_execution_horizon = max(1, int(self.action_horizon) - int(self.action_step))
        if self.execution_horizon is None:
            self.execution_horizon = min(self.inference_interval_actions, max_execution_horizon)
        else:
            self.execution_horizon = min(max(1, int(self.execution_horizon)), max_execution_horizon)

        self.camera_delay_s = max(0.0, float(self.camera_delay_ms) / 1000.0)
        if self.camera_delay_history_steps is None:
            self.camera_history_maxlen = max(
                2,
                int(np.ceil((self.camera_delay_s + self.action_dt_s) / self.action_dt_s)) + 4,
            )
        else:
            self.camera_history_maxlen = max(2, int(self.camera_delay_history_steps))


class Sim2SimCameraTimeline:
    """Discrete sim-time camera buffer with render-style delay selection."""

    def __init__(self, config: Sim2SimAsyncTimingConfig) -> None:
        self.config = config
        self._sim_time_s = 0.0
        self._latest_camera_time_s = 0.0
        self._camera_tick_count = 0
        # Baseline MuJoCo sim time of the first frame; sim time is tracked relative
        # to this origin so the action timeline runs on real physics time.
        self._origin_sim_time_s: float | None = None
        self._history: list[tuple[float, dict[str, np.ndarray]]] = []
        self.last_delay_metadata: dict[str, float | int] = {
            "requested_delay_s": float(config.camera_delay_s),
            "actual_delay_s": 0.0,
            "target_time_s": 0.0,
            "selected_time_s": 0.0,
            "latest_time_s": 0.0,
            "history_len": 0,
            "tick_count": 0,
        }

    def reset(self) -> None:
        self._sim_time_s = 0.0
        self._latest_camera_time_s = 0.0
        self._camera_tick_count = 0
        self._origin_sim_time_s = None
        self._history.clear()

    def tick(self, images: dict[str, np.ndarray], frame_sim_time_s: float) -> bool:
        """Record a camera frame stamped with the real MuJoCo sim time.

        Sim time is the authoritative clock (carried per-frame from
        ``mj_data.time``). Frames that do not advance sim time (duplicates /
        stale reuse while physics is mid-step) are dropped so the action
        timeline only progresses when physics actually progresses.

        Returns True when a new frame was recorded.
        """
        frame_sim_time_s = float(frame_sim_time_s)
        if self._origin_sim_time_s is None:
            self._origin_sim_time_s = frame_sim_time_s
        rel_time = frame_sim_time_s - self._origin_sim_time_s

        if self._history and rel_time <= self._latest_camera_time_s + 1e-9:
            return False

        copied = {
            str(key): np.ascontiguousarray(np.asarray(img, dtype=np.uint8)).copy()
            for key, img in images.items()
        }
        self._history.append((rel_time, copied))
        if len(self._history) > self.config.camera_history_maxlen:
            del self._history[: len(self._history) - self.config.camera_history_maxlen]
        self._latest_camera_time_s = rel_time
        self._sim_time_s = rel_time
        self._camera_tick_count += 1
        return True

    def has_frames(self) -> bool:
        return bool(self._history)

    def delayed_images(self) -> dict[str, np.ndarray]:
        if not self._history:
            raise RuntimeError("Camera timeline is empty; call tick() before requesting delayed images.")

        target_time = self._latest_camera_time_s - self.config.camera_delay_s
        selected_timestamp, selected_images = self._history[0]
        for timestamp, images in reversed(self._history):
            if timestamp <= target_time + 1e-9:
                selected_timestamp = timestamp
                selected_images = images
                break

        self.last_delay_metadata = {
            "requested_delay_s": float(self.config.camera_delay_s),
            "actual_delay_s": float(max(0.0, self._latest_camera_time_s - selected_timestamp)),
            "target_time_s": float(target_time),
            "selected_time_s": float(selected_timestamp),
            "latest_time_s": float(self._latest_camera_time_s),
            "history_len": int(len(self._history)),
            "tick_count": int(self._camera_tick_count),
        }
        return {
            str(key): np.ascontiguousarray(np.asarray(img, dtype=np.uint8)).copy()
            for key, img in selected_images.items()
        }

    @property
    def sim_time_s(self) -> float:
        return float(self._sim_time_s)

    @property
    def published_action_count(self) -> int:
        return int(self._camera_tick_count)


def motion_token_horizon(processed_action: dict[str, Any]) -> int:
    motion = _action_field_np(processed_action, "motion_token")
    if motion.ndim == 3:
        return int(motion.shape[1])
    if motion.ndim == 2:
        return int(motion.shape[0])
    return 1


def extract_motion_token(processed_action: dict[str, Any], index: int) -> np.ndarray:
    motion = np.asarray(_action_field_np(processed_action, "motion_token"), dtype=np.float32)
    if motion.ndim == 3:
        motion = motion[0]
    if motion.ndim == 2:
        index = min(max(int(index), 0), motion.shape[0] - 1)
        return np.asarray(motion[index], dtype=np.float32)
    return np.asarray(motion, dtype=np.float32).reshape(-1)


def _action_field_np(action_dict: dict[str, Any], key: str) -> np.ndarray:
    value = action_dict.get(key)
    if value is None:
        value = action_dict.get(f"action.{key}")
    if value is None:
        raise KeyError(f"Action dict missing {key!r} / action.{key!r}")
    return np.asarray(value)


class Sim2SimAsyncVlaLoop:
    """Async VLA action scheduler for sim2sim (render_vla_asyn semantics)."""

    def __init__(self, config: Sim2SimAsyncTimingConfig, camera_timeline: Sim2SimCameraTimeline) -> None:
        self.config = config
        self.camera_timeline = camera_timeline

        self._action_chunk: dict[str, Any] | None = None
        self._chunk_index = 0
        self._pending_action_chunk: dict[str, Any] | None = None
        self._pending_chunk_index = 0
        self._pending_release_publish_count = 0
        self._pending_chunk_id: int | None = None
        self._active_chunk_id: int | None = None

        self._published_action_count = 0
        self._publish_countdown = 0
        self._last_inference_request_publish_count = 0
        self._last_inference_result_publish_count = 0
        self._last_inference_result_time = 0.0
        self._next_action_publish_wall_time: float | None = None
        self._next_action_publish_sim_time: float | None = None

        self._inference_epoch = 0
        self._inference_seq = 0
        self._policy_step = 0

        self._inference_queue: queue.Queue | None = None
        self._result_queue: queue.Queue | None = None
        self._inference_busy_event: threading.Event | None = None
        self._inference_pending_event: threading.Event | None = None
        self._stop_event: threading.Event | None = None
        self._worker_thread: threading.Thread | None = None

    def start_worker(self, inference_fn: Callable[[dict[str, Any]], dict[str, Any] | None]) -> None:
        queue_size = max(1, int(self.config.async_queue_size))
        self._inference_queue = queue.Queue(maxsize=queue_size)
        self._result_queue = queue.Queue(maxsize=1)
        self._inference_busy_event = threading.Event()
        self._inference_pending_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(inference_fn,),
            name="sim2sim-vla-inference-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._inference_queue is not None:
            try:
                self._inference_queue.put_nowait(None)
            except queue.Full:
                pass
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=float(self.config.async_join_timeout_s))

    def reset(self) -> None:
        self.camera_timeline.reset()
        self._action_chunk = None
        self._chunk_index = 0
        self._pending_action_chunk = None
        self._pending_chunk_index = 0
        self._pending_release_publish_count = 0
        self._pending_chunk_id = None
        self._active_chunk_id = None
        self._published_action_count = 0
        self._publish_countdown = 0
        self._last_inference_request_publish_count = 0
        self._last_inference_result_publish_count = 0
        self._last_inference_result_time = 0.0
        self._next_action_publish_wall_time = None
        self._next_action_publish_sim_time = None
        self._inference_epoch += 1
        self._policy_step = 0
        if self._result_queue is not None:
            while True:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    break

    def build_request_metadata(self) -> dict[str, float | int]:
        camera_meta = dict(self.camera_timeline.last_delay_metadata)
        self._inference_seq += 1
        return {
            "inference_id": int(self._inference_seq),
            "epoch": int(self._inference_epoch),
            "request_policy_step": int(self._policy_step),
            "request_publish_count": int(self._published_action_count),
            "request_action_sim_time_s": (
                float(self._published_action_count) / float(self.config.action_timing_rate_hz)
            ),
            "request_sim_time_s": float(camera_meta.get("latest_time_s", 0.0)),
            "request_wall_time_s": float(time.monotonic()),
            "camera_delay_requested_s": float(camera_meta.get("requested_delay_s", 0.0)),
            "camera_delay_actual_s": float(camera_meta.get("actual_delay_s", 0.0)),
            "camera_target_time_s": float(camera_meta.get("target_time_s", 0.0)),
            "camera_selected_time_s": float(camera_meta.get("selected_time_s", 0.0)),
            "camera_latest_time_s": float(camera_meta.get("latest_time_s", 0.0)),
            "camera_history_len": int(camera_meta.get("history_len", 0)),
            "camera_tick_count": int(camera_meta.get("tick_count", 0)),
        }

    def start_async_inference(self, observation: dict[str, Any]) -> None:
        if self._inference_queue is None or self._inference_pending_event is None:
            return
        metadata = self.build_request_metadata()
        item = (int(self._inference_epoch), int(metadata["inference_id"]), observation, metadata)
        try:
            self._inference_pending_event.set()
            self._inference_queue.put_nowait(item)
            self._last_inference_request_publish_count = int(self._published_action_count)
        except queue.Full:
            self._inference_pending_event.clear()

    def should_start_async_inference(self) -> bool:
        if self._inference_queue is None:
            return False
        if self._inference_pending_event is not None and self._inference_pending_event.is_set():
            return False
        if self._inference_busy_event is not None and self._inference_busy_event.is_set():
            return False
        if self._pending_action_chunk is not None:
            return False
        if self._action_chunk is None:
            return True
        if self.config.async_timing_mode == "sim":
            elapsed_actions = int(self._published_action_count) - int(self._last_inference_request_publish_count)
            return elapsed_actions >= self.config.inference_interval_actions
        return (time.monotonic() - float(self._last_inference_result_time)) >= self.config.inference_interval_s

    def activate_pending_action_chunk_if_ready(self) -> bool:
        if self._pending_action_chunk is None:
            return False
        if int(self._published_action_count) < int(self._pending_release_publish_count):
            return False
        self._action_chunk = self._pending_action_chunk
        self._chunk_index = int(self._pending_chunk_index)
        self._active_chunk_id = self._pending_chunk_id
        self._last_inference_result_publish_count = int(self._published_action_count)
        self._pending_action_chunk = None
        self._pending_chunk_index = 0
        self._pending_release_publish_count = 0
        self._pending_chunk_id = None
        return True

    def consume_async_result(self, *, allow_initial_immediate_activation: bool = False) -> bool:
        if self._result_queue is None:
            return False

        latest = None
        while True:
            try:
                latest = self._result_queue.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return False

        (
            epoch,
            inference_id,
            inference_start_time,
            inference_end_time,
            processed_action,
            error,
            request_meta,
        ) = latest
        if int(epoch) != int(self._inference_epoch):
            return False
        if error is not None:
            raise RuntimeError("Async VLA inference failed.") from error
        if processed_action is None:
            return False

        result_consume_time = time.monotonic()
        wall_latency_s = result_consume_time - float(inference_start_time)
        policy_infer_time_s = float(request_meta.get("policy_infer_time_s", wall_latency_s))
        horizon = motion_token_horizon(processed_action)
        request_publish_count = int(request_meta.get("request_publish_count", 0))
        result_publish_count = int(self._published_action_count)
        elapsed_actions = max(0, result_publish_count - request_publish_count)

        if self.config.latency_compensation:
            if self.config.async_timing_mode == "wall" and self.config.latency_compensation_source == "wall":
                compensation_latency_s = float(wall_latency_s)
                compensation_time_axis = "wall"
            else:
                compensation_latency_s = float(policy_infer_time_s)
                compensation_time_axis = "sim"
            compensation_elapsed_actions = max(
                0,
                int(np.round(float(compensation_latency_s) * float(self.config.action_timing_rate_hz))),
            )
        else:
            compensation_elapsed_actions = 0
            compensation_latency_s = 0.0
            compensation_time_axis = "disabled"

        compensated_index = min(int(compensation_elapsed_actions), max(0, horizon - 1))
        start_index = min(max(compensated_index, int(elapsed_actions)), max(0, horizon - 1))
        release_publish_count = request_publish_count + int(compensation_elapsed_actions)
        pending_release_actions = max(0, release_publish_count - result_publish_count)
        initial_wait_activation = (
            bool(allow_initial_immediate_activation)
            and result_publish_count == 0
            and self._action_chunk is None
        )
        effective_pending_release_actions = 0 if initial_wait_activation else pending_release_actions

        if self.config.async_timing_mode == "sim" and effective_pending_release_actions > 0:
            self._pending_action_chunk = processed_action
            self._pending_chunk_index = int(start_index)
            self._pending_release_publish_count = int(release_publish_count)
            self._pending_chunk_id = int(inference_id)
        else:
            self._action_chunk = processed_action
            self._chunk_index = int(start_index)
            self._active_chunk_id = int(inference_id)
            self._last_inference_result_publish_count = int(result_publish_count)

        self._last_inference_result_time = time.monotonic()
        self._next_action_publish_wall_time = None

        if self.config.log_async_chunks:
            print(
                "[sim2sim][vla] new async action chunk "
                f"id={int(inference_id)}, "
                f"infer={policy_infer_time_s * 1000.0:.1f}ms, "
                f"wall_latency={wall_latency_s * 1000.0:.1f}ms, "
                f"sim_comp={compensation_latency_s * 1000.0:.1f}ms/{compensation_time_axis}, "
                f"camera_delay={float(request_meta.get('camera_delay_actual_s', 0.0)) * 1000.0:.1f}ms, "
                f"skip={start_index}, comp_actions={compensation_elapsed_actions}, "
                f"actual_elapsed_actions={elapsed_actions}, "
                f"pending_release_actions={effective_pending_release_actions}, "
                f"initial_wait={int(initial_wait_activation)}, "
                f"release_count={release_publish_count}, horizon={horizon}",
                flush=True,
            )
        return True

    def await_initial_action_chunk(self) -> None:
        if not self.config.wait_for_first_action:
            return
        deadline = None if self.config.first_action_timeout_s <= 0.0 else (
            time.monotonic() + float(self.config.first_action_timeout_s)
        )
        while self._action_chunk is None and self._pending_action_chunk is None:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    "Timed out waiting for the first async VLA action chunk before robot control."
                )
            if self.should_start_async_inference():
                raise RuntimeError(
                    "await_initial_action_chunk() requires the caller to enqueue inference before waiting."
                )
            self.consume_async_result(allow_initial_immediate_activation=True)
            self.activate_pending_action_chunk_if_ready()
            time.sleep(0.001)

    def has_action_chunk(self) -> bool:
        return self._action_chunk is not None

    @property
    def published_action_count(self) -> int:
        return int(self._published_action_count)

    def current_motion_token(self) -> np.ndarray | None:
        if self._action_chunk is None:
            return None
        horizon = motion_token_horizon(self._action_chunk)
        action_index = min(int(self.config.action_step) + int(self._chunk_index), horizon - 1)
        return extract_motion_token(self._action_chunk, action_index)

    def current_action_index(self) -> int:
        if self._action_chunk is None:
            return 0
        horizon = motion_token_horizon(self._action_chunk)
        return min(int(self.config.action_step) + int(self._chunk_index), horizon - 1)

    def active_action_chunk(self) -> dict[str, Any] | None:
        return self._action_chunk

    def advance_after_publish(self) -> None:
        if self._action_chunk is not None:
            horizon = motion_token_horizon(self._action_chunk)
            self._chunk_index = min(self._chunk_index + 1, horizon - 1)
        self._published_action_count += 1
        self._policy_step += 1
        if self.config.async_timing_mode != "wall":
            publish_stride = max(
                1,
                int(round(
                    float(self.config.action_timing_rate_hz)
                    / float(self.config.action_publish_rate_hz)
                )),
            )
            self._publish_countdown = publish_stride - 1

    def advance_empty_action_slot(self) -> None:
        """Advance sim action-slot timing without publishing a motion token.

        Matches render ``_publish_one_action_slot`` zero-action fallback so pending
        chunks can reach their release count while waiting for VLA inference.
        """
        self._published_action_count += 1
        self._policy_step += 1
        if self.config.async_timing_mode != "wall":
            publish_stride = max(
                1,
                int(round(
                    float(self.config.action_timing_rate_hz)
                    / float(self.config.action_publish_rate_hz)
                )),
            )
            self._publish_countdown = publish_stride - 1

    def on_action_slot(self) -> bool:
        """Returns True when a new action should be published this sim slot."""
        if self.config.async_timing_mode == "wall":
            return self._wall_publish_slots_due() > 0
        should_publish = self._action_chunk is None or self._publish_countdown <= 0
        if should_publish:
            self._publish_countdown = 0
            return True
        self._publish_countdown -= 1
        return False

    def sim_publish_slots_due(self, sim_time_s: float) -> int:
        """Number of action slots due given the current real sim time.

        Paces action publication to ``action_dt_s`` of MuJoCo sim time (not
        wall-clock), so the latent trajectory advances in lockstep with physics
        even when the simulator runs slower than real time. The first call
        publishes one slot immediately to kick off control.
        """
        sim_time_s = float(sim_time_s)
        if self._next_action_publish_sim_time is None:
            self._next_action_publish_sim_time = sim_time_s + self.config.action_dt_s
            return 1
        if sim_time_s + 1e-9 < self._next_action_publish_sim_time:
            return 0
        elapsed = sim_time_s - self._next_action_publish_sim_time
        slots = int(np.floor(elapsed / self.config.action_dt_s)) + 1
        slots = max(1, slots)
        self._next_action_publish_sim_time += float(slots) * self.config.action_dt_s
        return slots

    def _wall_publish_slots_due(self) -> int:
        now = time.monotonic()
        if self._next_action_publish_wall_time is None:
            self._next_action_publish_wall_time = now + self.config.action_publish_period_s
            return 1
        if now < self._next_action_publish_wall_time:
            return 0
        elapsed = now - self._next_action_publish_wall_time
        slots = int(np.floor(elapsed / self.config.action_publish_period_s)) + 1
        slots = max(1, slots)
        self._next_action_publish_wall_time += float(slots) * self.config.action_publish_period_s
        return slots

    def _worker_loop(self, inference_fn: Callable[[dict[str, Any]], dict[str, Any] | None]) -> None:
        assert self._inference_queue is not None
        assert self._inference_busy_event is not None
        assert self._inference_pending_event is not None
        assert self._stop_event is not None
        assert self._result_queue is not None

        while not self._stop_event.is_set():
            try:
                try:
                    item = self._inference_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    continue

                epoch, inference_id, observation, request_meta = item
                self._inference_busy_event.set()
                inference_start_time = time.monotonic()
                try:
                    processed_action = inference_fn(observation)
                    inference_end_time = time.monotonic()
                    request_meta = dict(request_meta)
                    request_meta.update(
                        {
                            "inference_start_wall_time_s": float(inference_start_time),
                            "inference_end_wall_time_s": float(inference_end_time),
                            "policy_infer_time_s": float(inference_end_time - inference_start_time),
                        }
                    )
                    self._put_latest_result(
                        (
                            epoch,
                            inference_id,
                            inference_start_time,
                            inference_end_time,
                            processed_action,
                            None,
                            request_meta,
                        )
                    )
                except Exception as exc:
                    inference_end_time = time.monotonic()
                    request_meta = dict(request_meta)
                    request_meta.update(
                        {
                            "inference_start_wall_time_s": float(inference_start_time),
                            "inference_end_wall_time_s": float(inference_end_time),
                            "policy_infer_time_s": float(inference_end_time - inference_start_time),
                        }
                    )
                    self._put_latest_result(
                        (
                            epoch,
                            inference_id,
                            inference_start_time,
                            inference_end_time,
                            None,
                            exc,
                            request_meta,
                        )
                    )
                finally:
                    self._inference_busy_event.clear()
                    self._inference_pending_event.clear()
            except Exception as exc:
                self._inference_pending_event.clear()
                print(f"[sim2sim][vla] async inference worker error: {exc}", flush=True)

    def _put_latest_result(self, result) -> None:
        assert self._result_queue is not None
        while True:
            try:
                self._result_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._result_queue.put_nowait(result)
        except queue.Full:
            pass


def legacy_wall_latency_index(inference_delay_s: float, control_freq_hz: float, action_horizon: int) -> int:
  return calculate_latency_compensated_index(inference_delay_s, control_freq_hz, action_horizon)
