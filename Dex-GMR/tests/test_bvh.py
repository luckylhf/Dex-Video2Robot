"""BVH format, transform and pipeline adapter regressions using synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from general_motion_retargeting.utils.bvh import read_bvh, world_transforms
from general_motion_retargeting.utils.lafan1 import load_lafan1_file, load_soma_file


def skeleton(mixed=False):
    foot_channels = "6 Zposition Xposition Yposition Xrotation Zrotation Yrotation" if mixed else "3 Zrotation Xrotation Yrotation"
    hierarchy = f"""HIERARCHY
ROOT Hips {{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT LeftFoot {{
  OFFSET 1 -2 0
  CHANNELS {foot_channels}
  End Site {{ OFFSET 0 -1 0 }}
 }}
 JOINT RightFoot {{ OFFSET -1 -2 0 CHANNELS 3 Yrotation Zrotation Xrotation }}
 JOINT Head {{ OFFSET 0 2 0 CHANNELS 3 Zrotation Xrotation Yrotation }}
}}
MOTION
Frames: 2
Frame Time: 0.02
"""
    foot_values = "5 3 4 0 0 0" if mixed else "0 0 0"
    row = f"10 20 30 90 0 0 {foot_values} 0 0 0 0 0 0"
    return hierarchy + row + "\n" + row + "\n"


def write_motion(tmp_path, text):
    path = tmp_path / "motion.bvh"
    path.write_text(text)
    return path


@pytest.mark.parametrize("mixed", [False, True])
def test_channel_layout_and_analytic_world_positions(tmp_path, mixed):
    path = write_motion(tmp_path, skeleton(mixed))
    motion = read_bvh(path)
    assert motion.bones == ["Hips", "LeftFoot", "RightFoot", "Head"]
    np.testing.assert_array_equal(motion.parents, [-1, 0, 0, 0])
    quats, positions = world_transforms(motion.quats, motion.pos, motion.parents)
    # Root rotates local X onto world Y; child translation is local to root.
    np.testing.assert_allclose(positions[0, 1], [6, 23, 35] if mixed else [12, 21, 30])
    np.testing.assert_allclose(positions[0, 2], [12, 19, 30])
    np.testing.assert_allclose(positions[0, 3], [8, 20, 30])
    np.testing.assert_allclose(Rotation.from_quat(quats[0, 0], scalar_first=True).apply([1, 0, 0]),
                               [0, 1, 0], atol=1e-14)
    loader = load_soma_file if mixed else load_lafan1_file
    frames, height, fps = loader(path)
    assert (len(frames), height, fps) == (2, 1.75, 50)
    np.testing.assert_allclose(frames[0]["Head"][0], [0.30, 0.08, 0.20])
    np.testing.assert_allclose(frames[0]["LeftFootMod"][0], positions[0, 1, [2, 0, 1]] / 100)


def test_joint_euler_orders_and_quaternion_continuity(tmp_path):
    text = skeleton().replace("10 20 30 90 0 0 0 0 0 0 0 0 0 0 0",
                              "10 20 30 170 30 20 10 40 -20 20 30 40 0 0 0", 1)
    text = text.replace("10 20 30 90 0 0", "10 20 30 190 30 20")
    motion = read_bvh(write_motion(tmp_path, text))
    # Matrix products provide an oracle independent of the parser's Euler call.
    def axis_rotation(axis, angle):
        vector = np.zeros(3)
        vector["XYZ".index(axis)] = np.deg2rad(angle)
        return Rotation.from_rotvec(vector).as_matrix()
    expected = axis_rotation("Y", 20) @ axis_rotation("Z", 30) @ axis_rotation("X", 40)
    actual = Rotation.from_quat(motion.quats[0, 2], scalar_first=True).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert np.all(np.sum(motion.quats[1:] * motion.quats[:-1], axis=-1) >= 0)


@pytest.mark.parametrize("old,new", [
    ("Frames: 2", "Frames: 3"),
    ("Frame Time: 0.02", "Frame Time: 0"),
    ("Frame Time: 0.02", "Frame Time: nan"),
    ("10 20 30 90", "nan 20 30 90"),
    ("10 20 30 90", "20 30 90"),
    ("Xposition Yposition Zposition", "Xposition Xposition Zposition"),
    ("JOINT Head", "JOINT Hips"),
    ("End Site { OFFSET 0 -1 0 }", "End Site { OFFSET 0 -1 0"),
])
def test_malformed_motion_fails_clearly(tmp_path, old, new):
    with pytest.raises(ValueError):
        read_bvh(write_motion(tmp_path, skeleton().replace(old, new)))


def test_forward_kinematics_accumulates_nested_rotations():
    quats = Rotation.from_euler("Z", [[90], [90], [0]], degrees=True).as_quat(scalar_first=True)[None]
    positions = np.array([[[1, 2, 3], [1, 0, 0], [1, 0, 0]]], dtype=float)
    _, world = world_transforms(quats, positions, [-1, 0, 1])
    np.testing.assert_allclose(world[0], [[1, 2, 3], [1, 3, 3], [0, 3, 3]], atol=1e-14)
