# Derived from Open-X-Humanoid/xGMR (Apache-2.0) and GMR (MIT).
# Modifications Copyright (c) 2026, Loong: BVH adapters and SciPy transforms.
# See Dex-GMR/LICENSE and THIRD_PARTY_NOTICES.md at the repository root.
import numpy as np
from scipy.spatial.transform import Rotation as R

from .bvh import read_bvh, world_transforms


def _canonicalize_bone_name(bone_name: str) -> str:
    """Map dataset-specific bone names to the canonical naming used by the IK configs."""
    # Strip common prefixes introduced by certain BVH exports (e.g., Skeleton7_)
    if bone_name.startswith("Skeleton7_"):
        bone_name = bone_name.replace("Skeleton7_", "", 1)

    # Normalise joints where the exporter appends suffixes
    if bone_name == "LeftToeBase":
        return "LeftToe"
    if bone_name == "RightToeBase":
        return "RightToe"

    return bone_name

def arrange_realtime_data(quats, pos, parents, bones):
    global_data = world_transforms(quats, pos, parents)

    # BVH files are typically Y-up. Rotate into the robot's Z-up frame.
    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    frame_rotation = R.from_matrix(rotation_matrix)

    frame = {}
    for i, bone in enumerate(bones):
        orientation = (frame_rotation * R.from_quat(global_data[0][0, i], scalar_first=True)).as_quat(scalar_first=True)
        position = global_data[1][0, i] @ rotation_matrix.T / 100  # cm to m
        canonical_bone = _canonicalize_bone_name(bone)
        frame[canonical_bone] = (position, orientation)

    # Add modified foot pose
    frame["LeftFootMod"] = (frame["LeftFoot"][0], frame["LeftFoot"][1])
    frame["RightFootMod"] = (frame["RightFoot"][0], frame["RightFoot"][1])
    
    return frame

def load_soma_file(bvh_file):
    """
    Load a SOMA-format BVH file (mixed per-joint channel counts).
    Bone naming differs from lafan1: LeftLeg=upper leg, LeftShin=lower leg.
    """
    data = read_bvh(bvh_file)
    global_data = world_transforms(data.quats, data.pos, data.parents)

    # BVH files are Y-up. Rotate into the robot's Z-up frame.
    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    frame_rotation = R.from_matrix(rotation_matrix)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = (frame_rotation * R.from_quat(global_data[0][frame, i], scalar_first=True)).as_quat(scalar_first=True)
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            canonical_bone = _canonicalize_bone_name(bone)
            result[canonical_bone] = (position, orientation)

        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftFoot"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightFoot"][1])

        frames.append(result)

    human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    human_height = 1.75

    actual_fps = 1.0 / data.frame_time

    return frames, human_height,actual_fps


def load_lafan1_file(bvh_file):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    data = read_bvh(bvh_file)
    global_data = world_transforms(data.quats, data.pos, data.parents)

    # BVH files are typically Y-up. Rotate into the robot's Z-up frame.
    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])# @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    frame_rotation = R.from_matrix(rotation_matrix)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = (frame_rotation * R.from_quat(global_data[0][frame, i], scalar_first=True)).as_quat(scalar_first=True)
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            canonical_bone = _canonicalize_bone_name(bone)
            result[canonical_bone] = (position, orientation)

        # Add modified foot pose
        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftFoot"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightFoot"][1])
        
        frames.append(result)
    
    human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    actual_fps = 1.0 / data.frame_time

    return frames, human_height,actual_fps
