#!/usr/bin/env python3
"""Build MuJoCo sim scene for unitree_g1_sonic_no_hand sim2sim.

Aligns with git scene_suitcase_43dof (Inspire DFQ body + damping + scene wrapper),
except:
  - Inspire hands removed; rubber_hand visual + eef_box wrist contacts added
  - training suitcase (suitcase-simplified_training.xml) instead of suitcase_scene.xml
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INSPIRE_ROBOT_XML = (
    REPO_ROOT
    / "gear_sonic/data/robot_model/model_data/g1/g1_description"
    / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.xml"
)
INSPIRE_MESHES = INSPIRE_ROBOT_XML.parent / "meshes"
HDMI_RUBBER_MESHES = Path(
    "/home/ubuntu/DATA4/zzb/HDMI/active_adaptation/assets_mjcf/g1_29dof_nohand/meshes"
)

OUT_DIR = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_nohand"
G1_DESC_DIR = REPO_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_description"
ROBOT_XML = G1_DESC_DIR / "g1_29dof_inspire_nohand_rubberhand.xml"
OUT_SCENE_XML = OUT_DIR / "g1_29dof_rubberhand_suitcase_sim.xml"

INSPIRE_HAND_MESHES = (
    ["L_hand_base_link"] + [f"Link{i:02d}_L" for i in range(11, 23)]
    + ["R_hand_base_link"] + [f"Link{i:02d}_R" for i in range(11, 23)]
)

# Collision-only L-shaped wrist contacts (invisible + thinner so wrist cameras stay clean).
_EEF_BOX_A = (
    'type="box" size="0.02 0.02 0.001" pos="0.13 0 0" '
    'rgba="0 0 0 0" contype="1" conaffinity="1" group="3"'
)
_EEF_BOX_B = (
    'type="box" size="0.001 0.02 0.02" pos="0.10 0 -0.03" '
    'rgba="0 0 0 0" contype="1" conaffinity="1" group="3"'
)

LEFT_WRIST_EXTRAS = f"""
                          <geom pos="0.0415 0.003 0" quat="1 0 0 0" type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.7 0.7 0.7 1" mesh="left_rubber_hand"/>
                          <geom name="left_eef_box_a" {_EEF_BOX_A}/>
                          <geom name="left_eef_box_b" {_EEF_BOX_B}/>"""

RIGHT_WRIST_EXTRAS = f"""
                          <geom pos="0.0415 -0.003 0" quat="1 0 0 0" type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.7 0.7 0.7 1" mesh="right_rubber_hand"/>
                          <geom name="right_eef_box_a" {_EEF_BOX_A}/>
                          <geom name="right_eef_box_b" {_EEF_BOX_B}/>"""

SCENE_WRAPPER = """<mujoco model="g1_29dof_rubberhand_suitcase_sim">
  <include file="../g1_description/g1_29dof_inspire_nohand_rubberhand.xml"/>
  <include file="objects/suitcase/suitcase-simplified_training.xml"/>

  <statistic center="0 0 0.5" extent="2.0"/>

  <visual>
    <headlight diffuse="0 0 0" ambient="0.36 0.40 0.48" specular="0 0 0"/>
    <rgba haze="0.72 0.80 0.95 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="flat" rgb1="0.72 0.80 0.95" rgb2="0.72 0.80 0.95" width="512" height="3072"/>
    <material name="groundplane" rgba="0.03 0.07 0.16 1" reflectance="0.0"/>
  </asset>

  <worldbody>
    <light name="key_light" pos="0 0 3" dir="0 0 -1" directional="true" diffuse="1.0 0.95 0.86" ambient="0 0 0" specular="0 0 0"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <camera name="global_view" pos="2.910 -5.040 3.860" xyaxes="0.866 0.500 0.000 -0.250 0.433 0.866"/>
  </worldbody>

  <default>
    <geom friction="1.0"/>
  </default>
</mujoco>
"""


def _remove_body_subtree(xml: str, body_name: str) -> str:
    token = f'<body name="{body_name}"'
    start = xml.find(token)
    if start < 0:
        return xml
    depth = 0
    idx = start
    while idx < len(xml):
        if xml.startswith("<body", idx):
            depth += 1
        elif xml.startswith("</body>", idx):
            depth -= 1
            if depth == 0:
                end = idx + len("</body>")
                return xml[:start] + xml[end:]
        idx += 1
    raise ValueError(f"Unbalanced body subtree while removing {body_name}")


def _strip_inspire_hands(xml: str) -> str:
    for mesh_name in INSPIRE_HAND_MESHES:
        xml = re.sub(
            rf'\s*<mesh name="{re.escape(mesh_name)}"[^>]*/>\s*',
            "\n",
            xml,
        )
    xml = re.sub(
        r'\s*<default class="inspire_hand_joint">.*?</default>\s*',
        "\n",
        xml,
        flags=re.DOTALL,
    )
    xml = _remove_body_subtree(xml, "L_hand_base_link")
    xml = _remove_body_subtree(xml, "R_hand_base_link")
    xml = re.sub(
        r'\s*<motor name="[LR]_[^"]+" joint="[LR]_[^"]+"[^>]*/>\s*',
        "\n",
        xml,
    )
    return xml


def _add_rubber_hands(xml: str) -> str:
    if "left_rubber_hand" not in xml:
        xml = xml.replace(
            '<mesh name="right_wrist_yaw_link"',
            '<mesh name="left_rubber_hand" file="left_rubber_hand.STL"/>\n'
            '    <mesh name="right_rubber_hand" file="right_rubber_hand.STL"/>\n'
            '    <mesh name="right_wrist_yaw_link"',
            1,
        )
    if "left_eef_box_a" not in xml:
        xml = xml.replace(
            '<camera name="left_wrist_camera"',
            LEFT_WRIST_EXTRAS + '\n                          <camera name="left_wrist_camera"',
            1,
        )
    if "right_eef_box_a" not in xml:
        xml = xml.replace(
            '<camera name="right_wrist_camera"',
            RIGHT_WRIST_EXTRAS + '\n                          <camera name="right_wrist_camera"',
            1,
        )
    return xml


def build_robot_xml() -> str:
    if not INSPIRE_ROBOT_XML.is_file():
        raise FileNotFoundError(f"Missing Inspire robot MJCF: {INSPIRE_ROBOT_XML}")
    xml = INSPIRE_ROBOT_XML.read_text()
    xml = xml.replace(
        '<mujoco model="g1_29dof_rev_1_0_with_inspire_hand_DFQ">',
        '<mujoco model="g1_29dof_inspire_nohand_rubberhand">',
    )
    xml = _strip_inspire_hands(xml)
    xml = _add_rubber_hands(xml)
    return xml


def ensure_rubber_meshes() -> None:
    if not HDMI_RUBBER_MESHES.is_dir():
        raise FileNotFoundError(f"Missing HDMI rubber-hand meshes: {HDMI_RUBBER_MESHES}")
    if not INSPIRE_MESHES.is_dir():
        raise FileNotFoundError(f"Missing Inspire meshes: {INSPIRE_MESHES}")
    for name in ("left_rubber_hand.STL", "right_rubber_hand.STL"):
        src = HDMI_RUBBER_MESHES / name
        dst = INSPIRE_MESHES / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing {src}")
        shutil.copy2(src, dst)


def main() -> int:
    ensure_rubber_meshes()
    G1_DESC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ROBOT_XML.write_text(build_robot_xml())
    OUT_SCENE_XML.write_text(SCENE_WRAPPER)
    print(f"Wrote robot: {ROBOT_XML}")
    print(f"Wrote scene: {OUT_SCENE_XML}")
    print(f"meshes: {INSPIRE_MESHES} (+ rubber_hand STL)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
