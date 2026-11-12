# Copyright (c) 2026, Loong
# SPDX-License-Identifier: BSD-3-Clause
"""BVH hierarchy/channel reader using NumPy and SciPy rotations.

Format reference: https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html
Translation channels use the absolute local positions used by this pipeline's
LAFAN/SOMA inputs; joints without translation channels use their OFFSET.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class BvhMotion:
    bones: list[str]
    parents: np.ndarray
    offsets: np.ndarray
    pos: np.ndarray
    quats: np.ndarray  # wxyz, local to parent
    frame_time: float


def read_bvh(path):
    """Read one skeleton with per-joint position/rotation channel layouts.

    Supports the 3/6-channel layouts used by LAFAN1 and SOMA, including
    different Euler orders per joint. End Site markers are not added as joints.
    """
    source = Path(path).read_text(encoding="utf-8-sig")
    sections = re.split(r"\bMOTION\b", source, maxsplit=1)
    if len(sections) != 2:
        raise ValueError("BVH is missing MOTION")
    tokens = iter(re.findall(r"[{}]|[^\s{}]+", sections[0]))
    names, parents, offsets, layouts = [], [], [], []

    def take(expected=None):
        value = next(tokens, None)
        if value is None or (expected is not None and value != expected):
            raise ValueError(f"BVH hierarchy: expected {expected or 'a token'}, got {value!r}")
        return value

    def joint(parent):
        name = take()
        if name in names:
            raise ValueError(f"Duplicate BVH joint: {name}")
        index = len(names)
        names.append(name)
        parents.append(parent)
        offsets.append(None)
        layouts.append(None)
        take("{")
        while (keyword := take()) != "}":
            if keyword == "OFFSET":
                if offsets[index] is not None:
                    raise ValueError(f"Duplicate OFFSET for {name}")
                offsets[index] = [float(take()) for _ in range(3)]
            elif keyword == "CHANNELS":
                if layouts[index] is not None:
                    raise ValueError(f"Duplicate CHANNELS for {name}")
                count = int(take())
                if count not in (3, 6):
                    raise ValueError(f"Unsupported BVH channel count for {name}: {count}")
                channels = [take() for _ in range(count)]
                valid = {axis + kind for axis in "XYZ" for kind in ("position", "rotation")}
                if len(set(channels)) != count or any(c not in valid for c in channels):
                    raise ValueError(f"Invalid BVH channels for {name}: {channels}")
                rotations = [c[0] for c in channels if c.endswith("rotation")]
                if set(rotations) != set("XYZ") or len(rotations) != 3:
                    raise ValueError(f"Expected three rotation channels for {name}")
                layouts[index] = channels
            elif keyword == "JOINT":
                joint(index)
            elif keyword == "End":
                take("Site")
                take("{")
                take("OFFSET")
                for _ in range(3):
                    float(take())
                take("}")
            else:
                raise ValueError(f"Unsupported BVH hierarchy token: {keyword}")
        if offsets[index] is None or layouts[index] is None:
            raise ValueError(f"Missing OFFSET or CHANNELS for {name}")

    take("HIERARCHY")
    take("ROOT")
    joint(-1)
    if next(tokens, None) is not None:
        raise ValueError("Expected one BVH skeleton")

    header = re.match(
        r"\s*Frames\s*:\s*(\d+)\s+Frame\s+Time\s*:\s*(\S+)\s*", sections[1]
    )
    if header is None:
        raise ValueError("BVH is missing Frames or Frame Time")
    frame_count = int(header[1])
    frame_time = float(header[2])
    if frame_count < 1 or not np.isfinite(frame_time) or frame_time <= 0:
        raise ValueError("BVH frame count and frame time must be positive")
    # Parse by row so a truncated frame cannot consume channels from its neighbour.
    rows = [line.split() for line in sections[1][header.end():].splitlines() if line.strip()]
    width = sum(map(len, layouts))
    if len(rows) != frame_count or any(len(row) != width for row in rows):
        raise ValueError(f"Expected {frame_count} BVH frames with {width} channels each")
    values = np.asarray(rows, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(offsets).all():
        raise ValueError("BVH contains non-finite values")
    positions = np.broadcast_to(offsets, (frame_count, len(names), 3)).copy()
    quaternions = np.empty((frame_count, len(names), 4), dtype=np.float64)
    cursor = 0
    for index, channels in enumerate(layouts):
        angle_columns, axes = [], []
        for column, channel in enumerate(channels, cursor):
            if channel.endswith("position"):
                positions[:, index, "XYZ".index(channel[0])] = values[:, column]
            else:
                axes.append(channel[0])
                angle_columns.append(column)
        rotations = Rotation.from_euler("".join(axes), values[:, angle_columns], degrees=True)
        q = rotations.as_quat(scalar_first=True)
        flips = np.sum(q[1:] * q[:-1], axis=-1) < 0
        signs = np.concatenate(([1], np.cumprod(np.where(flips, -1, 1))))
        quaternions[:, index] = q * signs[:, None]
        cursor += len(channels)
    return BvhMotion(names, np.asarray(parents), offsets, positions, quaternions, frame_time)


def world_transforms(local_quats, local_positions, parents):
    """Accumulate local transforms in parent-first order; quaternions are wxyz."""
    local_quats = np.asarray(local_quats)
    local_positions = np.asarray(local_positions)
    if local_positions.ndim != 3 or local_quats.shape != (*local_positions.shape[:2], 4):
        raise ValueError("Expected local positions (frames,joints,3) and quaternions (frames,joints,4)")
    if local_positions.shape[-1] != 3 or len(parents) != local_positions.shape[1]:
        raise ValueError("BVH transform dimensions differ")
    positions = np.empty_like(local_positions, dtype=np.float64)
    quats = np.empty_like(local_quats, dtype=np.float64)
    rotations = []
    for index, parent in enumerate(parents):
        rotation = Rotation.from_quat(local_quats[:, index], scalar_first=True)
        if parent == -1:
            positions[:, index] = local_positions[:, index]
        elif 0 <= parent < index:
            positions[:, index] = positions[:, parent] + rotations[parent].apply(local_positions[:, index])
            rotation = rotations[parent] * rotation
        else:
            raise ValueError("BVH parents must precede their children")
        rotations.append(rotation)
        quats[:, index] = rotation.as_quat(scalar_first=True)
    return quats, positions
