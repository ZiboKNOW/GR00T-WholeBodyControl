#!/usr/bin/env python3
"""Build MuJoCo sim scene from whole_body_tracking training URDF.

Exports g1_29dof.urdf via MuJoCo, then post-processes for sim2sim:
  - floating_base on pelvis (sim2sim spawn height z=0.793, matches scene_suitcase_43dof)
  - sensor frames parsed from URDF fixed joints (imu / d435 / mid360)
  - head_camera mounted on d435_link at URDF origin (no extra offset)
  - ImplicitActuator armature defaults from whole_body_tracking g1.py
  - actuator motors with training effort_limit_sim overrides
  - rubber_hand visual + collision from left/right_rubber_hand.STL (no eef_box)
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_URDF = (
    REPO_ROOT.parent
    / "whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/g1_description/g1_29dof.urdf"
)
TRAINING_MESHES = TRAINING_URDF.parent / "meshes"

OUT_DIR = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_nohand"
G1_DESC_DIR = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_description"
OUT_MESHES = G1_DESC_DIR / "meshes"
ROBOT_XML = G1_DESC_DIR / "g1_29dof_inspire_nohand_rubberhand.xml"
OUT_SCENE_XML = OUT_DIR / "g1_29dof_rubberhand_suitcase_sim.xml"

FLOATING_BASE_BLOCK = """  <link name="world"></link>
  <joint name="floating_base_joint" type="floating">
    <parent link="world"/>
    <child link="pelvis"/>
  </joint>"""

# whole_body_tracking/robots/g1.py effort_limit_sim (overrides URDF <limit effort=...>)
TRAINING_EFFORT_LIMITS: dict[str, float] = {
    "*_hip_yaw_joint": 88.0,
    "*_hip_roll_joint": 139.0,
    "*_hip_pitch_joint": 88.0,
    "*_knee_joint": 139.0,
    "*_ankle_pitch_joint": 50.0,
    "*_ankle_roll_joint": 50.0,
    "waist_yaw_joint": 88.0,
    "waist_roll_joint": 50.0,
    "waist_pitch_joint": 50.0,
    "*_shoulder_pitch_joint": 25.0,
    "*_shoulder_roll_joint": 25.0,
    "*_shoulder_yaw_joint": 25.0,
    "*_elbow_joint": 25.0,
    "*_wrist_roll_joint": 25.0,
    "*_wrist_pitch_joint": 5.0,
    "*_wrist_yaw_joint": 5.0,
}

# Isaac Sim / HDMI render_vla ego_view: d435 mount + front_cam offset on torso_link.
# (Differs from training URDF d435 z=0.41987; sim2sim uses this mount for correct ego_view.)
ISAAC_D435_LINK_POS = (0.0576235, 0.01753, 0.41987)
ISAAC_D435_LINK_QUAT = (0.91496, 0.0, 0.403545, 0.0)
ISAAC_HEAD_CAMERA_POS = (0.01, 0.0, 0.0)
ISAAC_HEAD_CAMERA_QUAT = (0.5, 0.5, -0.5, -0.5)

# MuJoCo head_camera intrinsics (Isaac tiled_camera / PinholeCameraCfg).
HEAD_CAMERA_WIDTH = 640
HEAD_CAMERA_HEIGHT = 480
HEAD_CAMERA_FX = 487.8023681640625
HEAD_CAMERA_FY = 487.8023681640625
HEAD_CAMERA_CX = 325.7099609375
HEAD_CAMERA_CY = 233.83817545572917
HEAD_CAMERA_PRINCIPALPIXEL_OFFSET = (
    HEAD_CAMERA_CX - HEAD_CAMERA_WIDTH / 2,
    HEAD_CAMERA_CY - HEAD_CAMERA_HEIGHT / 2,
)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
ARMATURE_5020_ANKLE_WAIST = 2.0 * ARMATURE_5020

JOINT_DEFAULTS_XML = f"""  <default>
    <!-- Passive joint dynamics from whole_body_tracking/robots/g1.py ImplicitActuator armature. -->
    <default class="hip_pitch_motor">
      <joint damping="0" armature="{ARMATURE_7520_14}" frictionloss="0"/>
    </default>
    <default class="hip_roll_motor">
      <joint damping="0" armature="{ARMATURE_7520_22}" frictionloss="0"/>
    </default>
    <default class="hip_yaw_motor">
      <joint damping="0" armature="{ARMATURE_7520_14}" frictionloss="0"/>
    </default>
    <default class="knee_motor">
      <joint damping="0" armature="{ARMATURE_7520_22}" frictionloss="0"/>
    </default>
    <default class="ankle_motor">
      <joint damping="0" armature="{ARMATURE_5020_ANKLE_WAIST}" frictionloss="0"/>
    </default>
    <default class="waist_yaw_motor">
      <joint damping="0" armature="{ARMATURE_7520_14}" frictionloss="0"/>
    </default>
    <default class="waist_motor">
      <joint damping="0" armature="{ARMATURE_5020_ANKLE_WAIST}" frictionloss="0"/>
    </default>
    <default class="arm_motor">
      <joint damping="0" armature="{ARMATURE_5020}" frictionloss="0"/>
    </default>
    <default class="wrist_motor">
      <joint damping="0" armature="{ARMATURE_4010}" frictionloss="0"/>
    </default>
  </default>"""

JOINT_CLASS_BY_NAME: dict[str, str] = {
    "left_hip_pitch_joint": "hip_pitch_motor",
    "right_hip_pitch_joint": "hip_pitch_motor",
    "left_hip_roll_joint": "hip_roll_motor",
    "right_hip_roll_joint": "hip_roll_motor",
    "left_hip_yaw_joint": "hip_yaw_motor",
    "right_hip_yaw_joint": "hip_yaw_motor",
    "left_knee_joint": "knee_motor",
    "right_knee_joint": "knee_motor",
    "left_ankle_pitch_joint": "ankle_motor",
    "right_ankle_pitch_joint": "ankle_motor",
    "left_ankle_roll_joint": "ankle_motor",
    "right_ankle_roll_joint": "ankle_motor",
    "waist_yaw_joint": "waist_yaw_motor",
    "waist_roll_joint": "waist_motor",
    "waist_pitch_joint": "waist_motor",
    "left_shoulder_pitch_joint": "arm_motor",
    "right_shoulder_pitch_joint": "arm_motor",
    "left_shoulder_roll_joint": "arm_motor",
    "right_shoulder_roll_joint": "arm_motor",
    "left_shoulder_yaw_joint": "arm_motor",
    "right_shoulder_yaw_joint": "arm_motor",
    "left_elbow_joint": "arm_motor",
    "right_elbow_joint": "arm_motor",
    "left_wrist_roll_joint": "arm_motor",
    "right_wrist_roll_joint": "arm_motor",
    "left_wrist_pitch_joint": "wrist_motor",
    "right_wrist_pitch_joint": "wrist_motor",
    "left_wrist_yaw_joint": "wrist_motor",
    "right_wrist_yaw_joint": "wrist_motor",
}

MOTOR_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

SCENE_WRAPPER = """<mujoco model="g1_29dof_rubberhand_suitcase_sim">
  <include file="../g1_description/g1_29dof_inspire_nohand_rubberhand.xml"/>
  <include file="objects/suitcase/suitcase-simplified_training.xml"/>

  <statistic center="0 0 0.5" extent="2.0"/>

  <visual>
    <!-- Match Isaac Sim viewport: light-blue sky + blue checker ground with white grid. -->
    <headlight diffuse="0 0 0" ambient="0.36 0.40 0.48" specular="0 0 0"/>
    <rgba haze="0.72 0.80 0.95 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.72 0.80 0.95" rgb2="0.72 0.80 0.95" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.95 0.95 0.95" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="8 8" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light name="key_light" pos="1 0 3.5" dir="0 0 -1" directional="true" diffuse="1.0 0.95 0.86" ambient="0 0 0" specular="0 0 0"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="1.0 0.005 0.0001"/>
    <camera name="global_view" pos="2.910 -5.040 3.860" xyaxes="0.866 0.500 0.000 -0.250 0.433 0.866"/>
  </worldbody>

  <default>
    <geom friction="1.0"/>
  </default>
</mujoco>
"""


@dataclass(frozen=True)
class UrdfMount:
    child_link: str
    parent_link: str
    pos: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


def _parse_float_triplet(text: str) -> tuple[float, float, float]:
    parts = [float(x) for x in text.split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 floats, got {text!r}")
    return parts[0], parts[1], parts[2]


def _rpy_to_quat_wxyz(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    return (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{v:.10g}" for v in values)


def _training_effort(joint_name: str) -> float | None:
    for pattern, effort in TRAINING_EFFORT_LIMITS.items():
        if fnmatch.fnmatch(joint_name, pattern):
            return effort
    return None


def _prepare_urdf_text(urdf_text: str, mesh_dir: Path) -> str:
    text = urdf_text.replace('filename="meshes/', 'filename="')
    text = text.replace(
        '  <!-- <link name="world"></link>\n'
        '  <joint name="floating_base_joint" type="floating">\n'
        '    <parent link="world"/>\n'
        '    <child link="pelvis"/>\n'
        '  </joint> -->',
        FLOATING_BASE_BLOCK,
    )
    return re.sub(
        r'<compiler meshdir="[^"]*"',
        f'<compiler meshdir="{mesh_dir}" autolimits="true"',
        text,
        count=1,
    )


def _parse_urdf_sensor_mounts(urdf_path: Path) -> dict[str, UrdfMount]:
    root = ET.parse(urdf_path).getroot()
    mounts: dict[str, UrdfMount] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        child = joint.find("child")
        parent = joint.find("parent")
        if child is None or parent is None:
            continue
        child_name = child.get("link")
        parent_name = parent.get("link")
        if child_name not in {"imu_in_torso", "imu_in_pelvis", "d435_link", "mid360_link"}:
            continue
        origin = joint.find("origin")
        xyz = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
        if origin is not None:
            if origin.get("xyz"):
                xyz = _parse_float_triplet(origin.get("xyz", "0 0 0"))
            if origin.get("rpy"):
                rpy = _parse_float_triplet(origin.get("rpy", "0 0 0"))
        mounts[child_name] = UrdfMount(
            child_link=child_name,
            parent_link=parent_name,
            pos=xyz,
            quat_wxyz=_rpy_to_quat_wxyz(rpy),
        )
    required = {"imu_in_torso", "imu_in_pelvis", "d435_link", "mid360_link"}
    missing = required - mounts.keys()
    if missing:
        raise ValueError(f"URDF missing sensor mounts: {sorted(missing)}")
    return mounts


def _export_mjcf_from_urdf(prepared_urdf: Path) -> str:
    model = mujoco.MjModel.from_xml_path(str(prepared_urdf))
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as handle:
        temp_xml = Path(handle.name)
    try:
        mujoco.mj_saveLastXML(str(temp_xml), model)
        return temp_xml.read_text()
    finally:
        temp_xml.unlink(missing_ok=True)


def _normalize_exported_xml(xml: str) -> str:
    xml = xml.replace(
        '<mujoco model="g1_29dof">',
        '<mujoco model="g1_29dof_inspire_nohand_rubberhand">',
        1,
    )
    xml = re.sub(
        r'<compiler([^>]*?)meshdir="[^"]*"',
        r'<compiler\1meshdir="meshes/"',
        xml,
        count=1,
    )
    xml = re.sub(
        r' content_type="model/stl"',
        "",
        xml,
    )
    return xml


def _inject_joint_defaults(xml: str) -> str:
    if "<default>" in xml.split("<asset>", 1)[0]:
        xml = re.sub(r"  <default>.*?</default>\s*(?=<asset>)", JOINT_DEFAULTS_XML + "\n\n", xml, count=1, flags=re.DOTALL)
    else:
        xml = xml.replace("<asset>", JOINT_DEFAULTS_XML + "\n\n  <asset>", 1)
    return xml


def _apply_joint_classes(xml: str) -> str:
    for joint_name, joint_class in JOINT_CLASS_BY_NAME.items():
        xml = re.sub(
            rf'(<joint name="{re.escape(joint_name)}"(?![^>]*\bclass=)[^>]*)(/>)',
            rf'\1 class="{joint_class}" \2',
            xml,
            count=1,
        )
    return xml


def _apply_training_effort_limits(xml: str) -> str:
    for joint_name in MOTOR_JOINTS:
        effort = _training_effort(joint_name)
        if effort is None:
            continue
        xml = re.sub(
            rf'(<joint name="{re.escape(joint_name)}"[^>]*actuatorfrcrange=")-?[0-9.]+ -?[0-9.]+(")',
            rf"\1-{effort:g} {effort:g}\2",
            xml,
            count=1,
        )
    return xml


def _set_pelvis_spawn_height(xml: str, height_m: float = 0.793) -> str:
    return re.sub(
        r'<body name="pelvis"([^>]*)>',
        rf'<body name="pelvis"\1 pos="0 0 {height_m:g}">',
        xml,
        count=1,
    )


def _inject_torso_sensors(xml: str, mounts: dict[str, UrdfMount]) -> str:
    imu = mounts["imu_in_torso"]
    mid360 = mounts["mid360_link"]
    head_camera_xml = (
        f'<camera name="head_camera" pos="{_format_vec(ISAAC_HEAD_CAMERA_POS)}" '
        f'quat="{_format_vec(ISAAC_HEAD_CAMERA_QUAT)}" '
        f'focalpixel="{HEAD_CAMERA_FX} {HEAD_CAMERA_FY}" '
        f'principalpixel="{HEAD_CAMERA_PRINCIPALPIXEL_OFFSET[0]} {HEAD_CAMERA_PRINCIPALPIXEL_OFFSET[1]}" '
        f'resolution="{HEAD_CAMERA_WIDTH} {HEAD_CAMERA_HEIGHT}" sensorsize="0.01 0.01"/>'
    )
    sensor_block = (
        f'\n          <body name="imu_in_torso" pos="{_format_vec(imu.pos)}"/>'
        f'\n          <body name="d435_link" pos="{_format_vec(ISAAC_D435_LINK_POS)}" '
        f'quat="{_format_vec(ISAAC_D435_LINK_QUAT)}">'
        f"\n            {head_camera_xml}"
        f"\n          </body>"
        f'\n          <body name="mid360_link" pos="{_format_vec(mid360.pos)}" quat="{_format_vec(mid360.quat_wxyz)}"/>'
    )
    marker = '<body name="left_shoulder_pitch_link"'
    if marker not in xml:
        raise ValueError("Could not find torso insertion point for sensor bodies")
    if 'name="d435_link"' in xml:
        raise ValueError("d435_link already present in exported MJCF")
    return xml.replace(marker, sensor_block + "\n          " + marker, 1)


def _inject_pelvis_imu_site(xml: str, mounts: dict[str, UrdfMount]) -> str:
    imu = mounts["imu_in_pelvis"]
    site = f'<site name="imu_in_pelvis" size="0.01" pos="{_format_vec(imu.pos)}"/>'
    if 'name="imu_in_pelvis"' in xml:
        return xml
    marker = '<body name="left_hip_pitch_link"'
    return xml.replace(marker, site + "\n      " + marker, 1)


def _build_actuator_section() -> str:
    lines = ["  <actuator>"]
    for joint_name in MOTOR_JOINTS:
        effort = _training_effort(joint_name)
        if effort is None:
            raise ValueError(f"No training effort for joint {joint_name}")
        motor_name = joint_name.removesuffix("_joint")
        lines.append(
            f'    <motor name="{motor_name}" joint="{joint_name}" ctrlrange="-{effort:g} {effort:g}"/>'
        )
    lines.append("  </actuator>")
    return "\n".join(lines)


def _inject_actuators(xml: str) -> str:
    if "<actuator>" in xml:
        xml = re.sub(r"  <actuator>.*?</actuator>\s*", "", xml, count=1, flags=re.DOTALL)
    return xml.replace("</mujoco>", _build_actuator_section() + "\n</mujoco>", 1)


def build_robot_xml() -> str:
    if not TRAINING_URDF.is_file():
        raise FileNotFoundError(f"Missing training URDF: {TRAINING_URDF}")
    mounts = _parse_urdf_sensor_mounts(TRAINING_URDF)
    prepared = _prepare_urdf_text(TRAINING_URDF.read_text(), TRAINING_MESHES)
    with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False) as handle:
        prepared_path = Path(handle.name)
    try:
        prepared_path.write_text(prepared)
        xml = _export_mjcf_from_urdf(prepared_path)
    finally:
        prepared_path.unlink(missing_ok=True)

    xml = _normalize_exported_xml(xml)
    xml = _inject_joint_defaults(xml)
    xml = _apply_joint_classes(xml)
    xml = _apply_training_effort_limits(xml)
    xml = _set_pelvis_spawn_height(xml)
    xml = _inject_pelvis_imu_site(xml, mounts)
    xml = _inject_torso_sensors(xml, mounts)
    xml = _inject_actuators(xml)
    return xml


def sync_training_meshes() -> None:
    if not TRAINING_MESHES.is_dir():
        raise FileNotFoundError(f"Missing training meshes: {TRAINING_MESHES}")
    OUT_MESHES.mkdir(parents=True, exist_ok=True)
    needed = {path.name for path in TRAINING_MESHES.glob("*.STL")}
    needed.update(path.name for path in TRAINING_MESHES.glob("*.stl"))
    for name in sorted(needed):
        src = TRAINING_MESHES / name
        if not src.is_file():
            alt = TRAINING_MESHES / name.lower()
            alt2 = TRAINING_MESHES / name.upper()
            src = alt if alt.is_file() else alt2
        if not src.is_file():
            continue
        shutil.copy2(src, OUT_MESHES / name)


def _verify_built_model(scene_xml: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    d435_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "d435_link")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
    if d435_body < 0 or camera_id < 0:
        raise RuntimeError("Built MJCF is missing d435_link or head_camera")
    if model.nu != len(MOTOR_JOINTS):
        raise RuntimeError(f"Expected {len(MOTOR_JOINTS)} actuators, got {model.nu}")

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    d435_pos = data.xpos[d435_body].copy()
    torso_pos = data.xpos[torso_id].copy()
    torso_mat = data.xmat[torso_id].reshape(3, 3)
    rel = torso_mat.T @ (d435_pos - torso_pos)
    expected = ISAAC_D435_LINK_POS
    if max(abs(rel[i] - expected[i]) for i in range(3)) > 1e-4:
        raise RuntimeError(
            f"d435_link mount mismatch vs Isaac sim2sim: got {rel.tolist()}, expected {list(expected)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate-robot",
        action="store_true",
        help=(
            "Re-export robot MJCF from training URDF. Default keeps the hand-crafted "
            "rubber_hand STL visual/collision geoms in g1_29dof_inspire_nohand_rubberhand.xml."
        ),
    )
    args = parser.parse_args()

    sync_training_meshes()
    G1_DESC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.regenerate_robot:
        ROBOT_XML.write_text(build_robot_xml())
        print(f"Wrote robot MJCF from training URDF: {ROBOT_XML}")
    elif not ROBOT_XML.is_file():
        ROBOT_XML.write_text(build_robot_xml())
        print(f"Wrote robot MJCF from training URDF: {ROBOT_XML}")
    else:
        print(f"Kept existing robot MJCF (rubber_hand STL mesh): {ROBOT_XML}")
    OUT_SCENE_XML.write_text(SCENE_WRAPPER)
    _verify_built_model(OUT_SCENE_XML)
    print(f"Wrote scene: {OUT_SCENE_XML}")
    print(f"Training URDF: {TRAINING_URDF}")
    print(f"Meshes synced to: {OUT_MESHES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
