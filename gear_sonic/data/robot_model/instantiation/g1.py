"""Factory function to instantiate a configured G1 RobotModel from URDF."""

import os
from pathlib import Path
from typing import Literal

from gear_sonic.data.robot_model.robot_model import RobotModel
from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
    ElbowPose,
    G1SupplementalInfo,
    HandType,
    WaistLocation,
)

_INSPIRE_URDF_NAME = "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"


def _resolve_inspire_urdf_paths() -> tuple[Path, Path]:
    """Resolve Inspire DFQ URDF and mesh directory."""
    env_root = os.environ.get("UNITREE_G1_DESCRIPTION")
    if env_root:
        asset_path = Path(env_root)
        urdf_path = asset_path / _INSPIRE_URDF_NAME
        if urdf_path.is_file():
            return urdf_path, asset_path

    workspace_root = Path(__file__).resolve().parents[5]
    unitree_g1_description = workspace_root / "unitree_ros" / "robots" / "g1_description"
    urdf_path = unitree_g1_description / _INSPIRE_URDF_NAME
    if urdf_path.is_file():
        return urdf_path, unitree_g1_description

    raise FileNotFoundError(
        "Inspire DFQ URDF not found. Expected at "
        f"{unitree_g1_description / _INSPIRE_URDF_NAME} "
        "or set UNITREE_G1_DESCRIPTION to the g1_description directory."
    )


def instantiate_g1_robot_model(
    waist_location: Literal["lower_body", "upper_body", "lower_and_upper_body"] = "lower_body",
    high_elbow_pose: bool = False,
    hand_type: Literal["g1_three_finger", "inspire_dfq"] = "g1_three_finger",
):
    """
    Instantiate a G1 robot model with configurable waist location and pose.

    Args:
        waist_location: Whether to put waist in "lower_body" (default G1 behavior),
                        "upper_body" (waist controlled with arms/manipulation via IK),
                        or "lower_and_upper_body" (waist reference from arms/manipulation
                        via IK then passed to lower body policy)
        high_elbow_pose: Whether to use high elbow pose configuration for default joint positions
        hand_type: Hand model variant. ``inspire_dfq`` loads the 6-DOF Inspire hand URDF used
                   by sim, deploy, and Sonic VLA training.

    Returns:
        RobotModel: Configured G1 robot model
    """
    model_data_dir = Path(__file__).resolve().parent.parent / "model_data" / "g1"
    assert hand_type in [
        "g1_three_finger",
        "inspire_dfq",
    ], f"Invalid hand_type: {hand_type}. Must be 'g1_three_finger' or 'inspire_dfq'"

    if hand_type == "inspire_dfq":
        urdf_path, asset_path = _resolve_inspire_urdf_paths()
        robot_model_config = {
            "asset_path": str(asset_path),
            "urdf_path": str(urdf_path),
        }
    else:
        robot_model_config = {
            "asset_path": str(model_data_dir),
            "urdf_path": str(model_data_dir / "g1_29dof_with_hand.urdf"),
        }

    assert waist_location in [
        "lower_body",
        "upper_body",
        "lower_and_upper_body",
    ], f"Invalid waist_location: {waist_location}. Must be 'lower_body' or 'upper_body' or 'lower_and_upper_body'"

    waist_location_enum = {
        "lower_body": WaistLocation.LOWER_BODY,
        "upper_body": WaistLocation.UPPER_BODY,
        "lower_and_upper_body": WaistLocation.LOWER_AND_UPPER_BODY,
    }[waist_location]

    elbow_pose_enum = ElbowPose.HIGH if high_elbow_pose else ElbowPose.LOW
    hand_type_enum = {
        "g1_three_finger": HandType.G1_THREE_FINGER,
        "inspire_dfq": HandType.INSPIRE_DFQ,
    }[hand_type]

    robot_model_supplemental_info = G1SupplementalInfo(
        waist_location=waist_location_enum,
        elbow_pose=elbow_pose_enum,
        hand_type=hand_type_enum,
    )

    robot_model = RobotModel(
        robot_model_config["urdf_path"],
        robot_model_config["asset_path"],
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model
