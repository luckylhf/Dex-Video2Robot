import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh, read_bvh_soma


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
    global_data = utils.quat_fk(quats, pos, parents)

    # BVH files are typically Y-up. Rotate into the robot's Z-up frame.
    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frame = {}
    for i, bone in enumerate(bones):
        orientation = utils.quat_mul(rotation_quat, global_data[0][0, i])
        position = global_data[1][0, i] @ rotation_matrix.T / 100  # cm to m
        canonical_bone = _canonicalize_bone_name(bone)
        frame[canonical_bone] = (position, orientation)

    # Add modified foot pose
    frame["LeftFootMod"] = (frame["LeftFoot"][0], frame["LeftFoot"][1])
    frame["RightFootMod"] = (frame["RightFoot"][0], frame["RightFoot"][1])
    
    return frame

def _parse_bvh_frametime(bvh_file):
    import re
    with open(bvh_file, "r") as f:
        for line in f:
            m = re.match(r"\s*Frame Time:\s+([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None

def load_soma_file(bvh_file):
    """
    Load a SOMA-format BVH file (mixed per-joint channel counts).
    Bone naming differs from lafan1: LeftLeg=upper leg, LeftShin=lower leg.
    """
    data = read_bvh_soma(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # BVH files are Y-up. Rotate into the robot's Z-up frame.
    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            canonical_bone = _canonicalize_bone_name(bone)
            result[canonical_bone] = (position, orientation)

        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftFoot"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightFoot"][1])

        frames.append(result)

    human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    human_height = 1.75

    frametime = _parse_bvh_frametime(bvh_file)
    actual_fps = (1.0 / frametime) if frametime else None

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
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # BVH files are typically Y-up. Rotate into the robot's Z-up frame.
    rotation_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])# @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
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

    frametime = _parse_bvh_frametime(bvh_file)
    actual_fps = (1.0 / frametime) if frametime else None

    return frames, human_height,actual_fps
