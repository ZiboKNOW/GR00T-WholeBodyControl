"""Stereolabs ZED camera driver.

Requires the system ZED SDK and its Python API.  After activating the
camera server venv, install the API with::

    python /usr/local/zed/get_python_api.py

This driver is optimized for VLA RGB input: by default it opens the ZED in
HD720 at 60 FPS, disables depth computation, reads the rectified left image,
and center-crops/resizes it to 640x480 to match the existing OAK/RealSense
camera output shape.
"""

from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment]

from gear_sonic.camera.sensor import Sensor
from gear_sonic.camera.sensor_server import CameraMountPosition, ImageMessageSchema


@dataclass
class ZEDCameraConfig:
    """Configuration for a Stereolabs ZED camera."""

    resolution: str = "HD720"
    camera_fps: int = 60
    depth_mode: str = "NONE"
    view: str = "LEFT"
    output_width: int = 640
    output_height: int = 480
    resize_mode: str = "crop"


def _load_zed_sdk():
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise RuntimeError(
            "pyzed.sl is not installed. Install the ZED SDK first, then run "
            "`python /usr/local/zed/get_python_api.py` inside .venv_camera."
        ) from exc
    return sl


def _get_enum_value(enum_cls, name: str, label: str):
    normalized = name.strip().upper()
    if hasattr(enum_cls, normalized):
        return getattr(enum_cls, normalized)
    valid = sorted(attr for attr in dir(enum_cls) if attr.isupper() and not attr.startswith("_"))
    raise ValueError(f"Unsupported ZED {label}: {name!r}. Valid values include: {valid}")


def _resize_image(image: np.ndarray, width: int, height: int, mode: str) -> np.ndarray:
    if width <= 0 or height <= 0:
        return image

    src_h, src_w = image.shape[:2]
    if (src_w, src_h) == (width, height):
        return image

    mode = mode.strip().lower()
    if mode == "none":
        return image

    if mode == "resize":
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    if mode == "crop":
        target_aspect = width / height
        src_aspect = src_w / src_h
        cropped = image
        if src_aspect > target_aspect:
            crop_w = int(round(src_h * target_aspect))
            x0 = max((src_w - crop_w) // 2, 0)
            cropped = image[:, x0 : x0 + crop_w]
        elif src_aspect < target_aspect:
            crop_h = int(round(src_w / target_aspect))
            y0 = max((src_h - crop_h) // 2, 0)
            cropped = image[y0 : y0 + crop_h, :]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)

    if mode == "letterbox":
        scale = min(width / src_w, height / src_h)
        new_w = max(int(round(src_w * scale)), 1)
        new_h = max(int(round(src_h * scale)), 1)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
        x0 = (width - new_w) // 2
        y0 = (height - new_h) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return canvas

    raise ValueError("ZED resize_mode must be one of: none, resize, crop, letterbox")


class ZEDCameraSensor(Sensor):
    """Sensor for Stereolabs ZED cameras using the ZED SDK."""

    def __init__(
        self,
        config: ZEDCameraConfig = ZEDCameraConfig(),
        mount_position: str = CameraMountPosition.EGO_VIEW.value,
        device_id: str | None = None,
    ):
        self.config = config
        self.mount_position = mount_position
        self.sl = _load_zed_sdk()

        self.zed = self.sl.Camera()
        init_params = self.sl.InitParameters()
        init_params.camera_resolution = _get_enum_value(
            self.sl.RESOLUTION, config.resolution, "resolution"
        )
        init_params.camera_fps = config.camera_fps
        init_params.depth_mode = _get_enum_value(
            self.sl.DEPTH_MODE, config.depth_mode, "depth_mode"
        )
        self._apply_device_id(init_params, device_id)

        err = self.zed.open(init_params)
        if err != self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        self.runtime_params = self.sl.RuntimeParameters()
        self.image_mat = self.sl.Mat()
        self.view = _get_enum_value(self.sl.VIEW, config.view, "view")

        info = self.zed.get_camera_information()
        camera_config = getattr(info, "camera_configuration", None)
        if camera_config is not None:
            resolution = camera_config.resolution
            source_fps = getattr(camera_config, "fps", config.camera_fps)
        else:
            resolution = info.camera_resolution
            source_fps = config.camera_fps
        self.source_width = int(resolution.width)
        self.source_height = int(resolution.height)

        if config.resize_mode.strip().lower() == "none" or (
            config.output_width <= 0 or config.output_height <= 0
        ):
            self.output_width = self.source_width
            self.output_height = self.source_height
        else:
            self.output_width = config.output_width
            self.output_height = config.output_height

        print(f"[{mount_position}] ZED camera opened")
        print(f"  Model: {getattr(info, 'camera_model', 'unknown')}")
        print(f"  Serial number: {getattr(info, 'serial_number', 'unknown')}")
        print(f"  Source: {self.source_width}x{self.source_height} @ {source_fps} FPS")
        print(
            f"  Output: {self.output_width}x{self.output_height} "
            f"(resize_mode={config.resize_mode})"
        )
        print(f"  View: {config.view}, depth_mode={config.depth_mode}")

        print(f"[{mount_position}] Warming up ZED camera...")
        for _ in range(10):
            if self.zed.grab(self.runtime_params) == self.sl.ERROR_CODE.SUCCESS:
                break
            time.sleep(0.1)

    def _apply_device_id(self, init_params: Any, device_id: str | None):
        if not device_id:
            return

        value = device_id.strip()
        if value.startswith("serial:"):
            serial = int(value.split(":", 1)[1])
            init_params.set_from_serial_number(serial)
            print(f"Selecting ZED by serial number: {serial}")
        elif value.startswith("id:"):
            camera_id = int(value.split(":", 1)[1])
            init_params.set_from_camera_id(camera_id)
            print(f"Selecting ZED by camera id: {camera_id}")
        elif value.isdigit():
            camera_id = int(value)
            init_params.set_from_camera_id(camera_id)
            print(f"Selecting ZED by camera id: {camera_id}")
        else:
            raise ValueError(
                "ZED device_id must be empty, an integer camera id, "
                "`id:<n>`, or `serial:<serial_number>`"
            )

    def read(self) -> dict[str, Any] | None:
        grab_status = self.zed.grab(self.runtime_params)
        if grab_status != self.sl.ERROR_CODE.SUCCESS:
            print(f"[{self.mount_position}] ZED grab failed: {grab_status}")
            return None

        retrieve_status = self.zed.retrieve_image(self.image_mat, self.view)
        if retrieve_status is not None and retrieve_status != self.sl.ERROR_CODE.SUCCESS:
            print(f"[{self.mount_position}] ZED retrieve_image failed: {retrieve_status}")
            return None

        frame = self.image_mat.get_data()
        if frame is None or frame.size == 0:
            print(f"[{self.mount_position}] ZED returned an empty frame")
            return None

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print(f"[{self.mount_position}] Unexpected ZED frame shape: {frame.shape}")
            return None

        frame_rgb = _resize_image(
            frame_rgb,
            self.config.output_width,
            self.config.output_height,
            self.config.resize_mode,
        )

        return {
            "timestamps": {self.mount_position: time.time()},
            "images": {self.mount_position: frame_rgb},
        }

    def serialize(self, data: dict[str, Any]) -> dict[str, Any]:
        serialized_msg = ImageMessageSchema(timestamps=data["timestamps"], images=data["images"])
        return serialized_msg.serialize()

    def observation_space(self):
        if gym is None:
            return None
        return gym.spaces.Dict(
            {
                "color_image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.output_height, self.output_width, 3),
                    dtype=np.uint8,
                ),
            }
        )

    def close(self):
        if getattr(self, "zed", None) is not None:
            self.zed.close()
            self.zed = None
