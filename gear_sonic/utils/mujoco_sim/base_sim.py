"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import os
import pathlib
from pathlib import Path
import pickle
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# omomo sub1_suitcase_011 frame 0: suitcase link origin in pelvis-yaw frame when the
# robot stands at the origin facing +x (pelvis z=0.793). Geom offset (-0.1,0,0.2) keeps
# the 0.2x0.3x0.4 m box upright with its 20x30 cm face on the floor.
SUITCASE_OFFSET_PELVIS_YAW = np.array([0.532087, -0.003498, 0.0])
SUITCASE_SPAWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
INSPIRE_PASSIVE_HAND_MIMIC = {
    "thumb_intermediate": (1, 1.6),
    "thumb_distal": (1, 2.4),
    "index_intermediate": (2, 1.0),
    "middle_intermediate": (3, 1.0),
    "ring_intermediate": (4, 1.0),
    "pinky_intermediate": (5, 1.0),
}


def _rotation_from_mujoco_quat(quat_wxyz: np.ndarray) -> Rotation:
    """Convert MuJoCo wxyz quaternion to scipy Rotation, tolerating invalid states."""
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-8:
        return Rotation.identity()
    return Rotation.from_quat(quat_xyzw / norm)


EGO_VIEW_CAMERA_CONFIGS = {
    "ego_view": {"height": 480, "width": 640, "mjcf_name": "head_camera"},
}

EGO_AND_GLOBAL_CAMERA_CONFIGS = {
    **EGO_VIEW_CAMERA_CONFIGS,
    "global_view": {"height": 480, "width": 640, "mjcf_name": "global_view"},
}

FULL_SIM_CAMERA_CONFIGS = {
    **EGO_AND_GLOBAL_CAMERA_CONFIGS,
    "left_wrist": {"height": 480, "width": 640, "mjcf_name": "left_wrist_camera"},
    "right_wrist": {"height": 480, "width": 640, "mjcf_name": "right_wrist_camera"},
}


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs

        if not camera_configs and offscreen and enable_image_publish:
            if config.get("EGO_VIEW_ONLY", False):
                self.camera_configs = dict(EGO_AND_GLOBAL_CAMERA_CONFIGS)
            else:
                self.camera_configs = dict(FULL_SIM_CAMERA_CONFIGS)

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen
        self.elastic_band = None

        self.init_scene()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand(
                use_angular=self.config.get("ELASTIC_BAND_USE_ANGULAR", True)
            )
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        left_hand_prefix = self.config.get("LEFT_HAND_JOINT_PREFIX", "left_hand")
        right_hand_prefix = self.config.get("RIGHT_HAND_JOINT_PREFIX", "right_hand")

        def _is_independent_hand_joint(joint_name: str, prefix: str) -> bool:
            if joint_name.startswith(prefix):
                # Inspire MJCF lists mimic joints as separate DoF; only count actuated primaries.
                return "intermediate" not in joint_name and "distal" not in joint_name
            return prefix in joint_name

        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if name == "floating_base_joint":
                continue
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif _is_independent_hand_joint(name, left_hand_prefix):
                self.left_hand_index.append(i)
            elif _is_independent_hand_joint(name, right_hand_prefix):
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)

        if len(self.torque_limit) != self.mj_model.nu:
            limits = np.ones(self.mj_model.nu)
            for i in range(self.mj_model.nu):
                if self.mj_model.actuator_forcelimited[i]:
                    limits[i] = max(
                        abs(self.mj_model.actuator_forcerange[i, 0]),
                        abs(self.mj_model.actuator_forcerange[i, 1]),
                    )
                elif i < len(self.torque_limit):
                    limits[i] = self.torque_limit[i]
            self.torque_limit = limits
        self.torques = np.zeros(self.mj_model.nu)
        self.passive_hand_mimic_actuators = self._collect_passive_hand_mimic_actuators()
        self._apply_spawn_pose()

    def _compute_suitcase_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Map pelvis-yaw horizontal offset to world frame; freejoint z stays on the floor."""
        pelvis_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_pos = self.mj_data.xpos[pelvis_id]
        pelvis_mat = self.mj_data.xmat[pelvis_id].reshape(3, 3)
        yaw = np.arctan2(pelvis_mat[1, 0], pelvis_mat[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        offset_xy = np.array(
            [
                c * SUITCASE_OFFSET_PELVIS_YAW[0] - s * SUITCASE_OFFSET_PELVIS_YAW[1],
                s * SUITCASE_OFFSET_PELVIS_YAW[0] + c * SUITCASE_OFFSET_PELVIS_YAW[1],
                0.0,
            ]
        )
        world_pos = pelvis_pos + offset_xy
        world_pos[2] = 0.0
        return world_pos, SUITCASE_SPAWN_QUAT.copy()

    def _apply_spawn_pose(self, reset_object: bool = True):
        """Apply training-aligned default joint poses; optionally reset suitcase spawn."""
        default_angles = self.robot.DEFAULT_DOF_ANGLES
        for i, joint_id in enumerate(self.body_joint_index):
            qadr = int(self.mj_model.jnt_qposadr[joint_id])
            self.mj_data.qpos[qadr] = default_angles[i]

        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        if reset_object:
            suitcase_joint = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "suitcase_root"
            )
            if suitcase_joint >= 0:
                qadr = int(self.mj_model.jnt_qposadr[suitcase_joint])
                spawn_pos, spawn_quat = self._compute_suitcase_world_pose()
                # freejoint tracks the motion body origin; geom offset (-0.1,0,0.2) places the
                # box upright (20x30 cm face on floor, 40 cm tall) without tilt or penetration.
                self.mj_data.qpos[qadr : qadr + 3] = spawn_pos
                self.mj_data.qpos[qadr + 3 : qadr + 7] = spawn_quat
                mujoco.mj_forward(self.mj_model, self.mj_data)

    def _get_suitcase_state(self):
        suitcase_joint = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "suitcase_root"
        )
        if suitcase_joint < 0:
            return None, None
        qadr = int(self.mj_model.jnt_qposadr[suitcase_joint])
        vadr = int(self.mj_model.jnt_dofadr[suitcase_joint])
        return (
            self.mj_data.qpos[qadr : qadr + 7].copy(),
            self.mj_data.qvel[vadr : vadr + 6].copy(),
        )

    def _set_suitcase_state(self, qpos, qvel):
        suitcase_joint = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "suitcase_root"
        )
        if suitcase_joint < 0 or qpos is None or qvel is None:
            return
        qadr = int(self.mj_model.jnt_qposadr[suitcase_joint])
        vadr = int(self.mj_model.jnt_dofadr[suitcase_joint])
        self.mj_data.qpos[qadr : qadr + 7] = qpos
        self.mj_data.qvel[vadr : vadr + 6] = qvel

    def _reset_robot_pose(self):
        """Reset only the floating base and body joints; preserve free objects."""
        default_angles = self.robot.DEFAULT_DOF_ANGLES
        if self.use_floating_root_link:
            self.mj_data.qpos[0:3] = self.mj_model.qpos0[0:3]
            self.mj_data.qpos[3:7] = self.mj_model.qpos0[3:7]
            self.mj_data.qvel[0:6] = 0.0
        for i, joint_id in enumerate(self.body_joint_index):
            qadr = int(self.mj_model.jnt_qposadr[joint_id])
            vadr = int(self.mj_model.jnt_dofadr[joint_id])
            self.mj_data.qpos[qadr] = default_angles[i]
            self.mj_data.qvel[vadr] = 0.0
        self.mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _collect_passive_hand_mimic_actuators(self):
        mimic_actuators = []
        for i in range(self.mj_model.nu):
            name = self.mj_model.actuator(i).name
            is_left = name.startswith("L_")
            if not (is_left or name.startswith("R_")):
                continue
            for pattern, (src_idx, multiplier) in INSPIRE_PASSIVE_HAND_MIMIC.items():
                if pattern in name:
                    mimic_actuators.append((i, is_left, src_idx, multiplier))
                    break
        return mimic_actuators

    def _apply_passive_hand_joint_mimic(self):
        for act_i, is_left, src_idx, multiplier in self.passive_hand_mimic_actuators:
            if self.unitree_bridge is not None:
                received_name = "left_hand_cmd_received" if is_left else "right_hand_cmd_received"
                if not getattr(self.unitree_bridge, received_name, False):
                    continue
            source_joints = self.left_hand_index if is_left else self.right_hand_index
            if src_idx >= len(source_joints):
                continue
            src_joint_id = int(source_joints[src_idx])
            src_qadr = self.mj_model.jnt_qposadr[src_joint_id]
            src_vadr = self.mj_model.jnt_dofadr[src_joint_id]

            joint_id = self.mj_model.actuator_trnid[act_i, 0]
            qadr = self.mj_model.jnt_qposadr[joint_id]
            vadr = self.mj_model.jnt_dofadr[joint_id]
            q = self.mj_data.qpos[src_qadr] * multiplier
            if self.mj_model.jnt_limited[joint_id]:
                q = np.clip(
                    q,
                    self.mj_model.jnt_range[joint_id, 0],
                    self.mj_model.jnt_range[joint_id, 1],
                )
            self.mj_data.qpos[qadr] = q
            self.mj_data.qvel[vadr] = self.mj_data.qvel[src_vadr] * multiplier
            self.torques[act_i] = 0.0

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if (
            self.unitree_bridge is not None
            and self.unitree_bridge.low_cmd
            and getattr(self.unitree_bridge, "low_cmd_received", False)
        ):
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        root_quat_wxyz = self.mj_data.body("torso_link").xquat.copy()
        head_pos = root_pos + _rotation_from_mujoco_quat(root_quat_wxyz).apply(
            np.array([0.0, 0.0, -0.044])
        )
        root_quat_xyzw = root_quat_wxyz[[1, 2, 3, 0]]
        if np.linalg.norm(root_quat_xyzw) < 1e-8:
            root_quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        return np.concatenate((head_pos, root_quat_xyzw))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None:
            num_hand_ctrl = min(self.unitree_bridge.num_hand_motor, self.num_hand_dof)
            if getattr(self.unitree_bridge, "left_hand_cmd_received", False):
                for i in range(num_hand_ctrl):
                    left_hand_torques[i] = (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                        )
                    )
            if getattr(self.unitree_bridge, "right_hand_cmd_received", False):
                for i in range(num_hand_ctrl):
                    right_hand_torques[i] = (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                        )
                    )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            num_hand_ctrl = min(
                self.unitree_bridge.num_hand_motor,
                self.num_hand_dof,
            )
            for i in range(num_hand_ctrl):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            obs["floating_base_pose"] = np.zeros(7)
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + 7 - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + 6 - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + 6 - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]
        if self.passive_hand_mimic_actuators:
            self._apply_passive_hand_joint_mimic()

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)
        if self.passive_hand_mimic_actuators:
            self._apply_passive_hand_joint_mimic()
        if self.num_hand_dof > 0 or self.passive_hand_mimic_actuators:
            mujoco.mj_forward(self.mj_model, self.mj_data)

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            elif "mjcf_name" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["mjcf_name"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "k":
            self.release_elastic_band()
        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def release_elastic_band(self):
        if self.elastic_band:
            self.elastic_band.release()

    def check_fall(self):
        if self.elastic_band and self.elastic_band.enable:
            return

        if not np.isfinite(self.mj_data.qpos[2]):
            print("Warning: Sim state invalid (non-finite height), resetting")
            self.reset()
            return

        self.fall = False
        if self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            # Match git sim2sim: deploy InitControl ramps pose while elastic band holds.
            # Auto-reset during active lowcmd fights PD and explodes the sim.
            deploy_active = self.unitree_bridge is not None and getattr(
                self.unitree_bridge, "low_cmd_received", False
            )
            if not deploy_active:
                # Keep suitcase/object pose from physics; only stand the robot back up.
                self.reset(reset_object=False)

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self, reset_object: bool = True):
        if reset_object:
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            self._apply_spawn_pose(reset_object=True)
            return

        suitcase_qpos, suitcase_qvel = self._get_suitcase_state()
        self._reset_robot_pose()
        self._set_suitcase_state(suitcase_qpos, suitcase_qvel)
        mujoco.mj_forward(self.mj_model, self.mj_data)


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True
        self.keyboard_listener = None

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        if self.config.get("ENABLE_ELASTIC_BAND", False):
            try:
                from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber

                self.keyboard_listener = ZMQKeyboardSubscriber()
            except Exception as e:
                print(f"Warning: ElasticBand keyboard subscriber disabled: {e}")

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        pass

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def poll_keyboard_commands(self):
        if self.keyboard_listener is None:
            return
        key = self.keyboard_listener.read_msg()
        if key == "k":
            self.sim_env.release_elastic_band()

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.poll_keyboard_commands()
                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if self.keyboard_listener is not None:
                self.keyboard_listener.close()
                self.keyboard_listener = None
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
