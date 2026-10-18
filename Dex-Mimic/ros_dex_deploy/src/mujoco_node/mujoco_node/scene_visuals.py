"""Shared viewer scenery, compiled in memory without changing the robot asset."""
from pathlib import Path

import mujoco
import numpy as np


def load_model_with_scene(model_path):
    """Give the existing plane an infinite blue-grey grid and a soft horizon.

    Plane sizes only control rendering in MuJoCo. Its collision surface remains
    infinite; friction, contact parameters, robot and motor settings are retained.
    """
    spec = mujoco.MjSpec.from_file(str(Path(model_path).resolve()))
    floor = spec.geom("floor")
    if floor is None or floor.type != mujoco.mjtGeom.mjGEOM_PLANE:
        return spec.compile()

    pixels = np.full((256, 256, 3), (86, 109, 123), dtype=np.uint8)
    pixels[:2, :, :] = (153, 174, 184)
    pixels[:, :2, :] = (153, 174, 184)
    spec.add_texture(name="dex_viewer_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     width=256, height=256, nchannel=3, data=pixels.ravel())
    material = spec.add_material(name="dex_viewer_floor", texuniform=True,
                                 texrepeat=[1, 1], rgba=[1, 1, 1, 1],
                                 specular=.1, shininess=.1, reflectance=0.)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "dex_viewer_grid"
    floor.material = material.name
    floor.rgba = [1, 1, 1, 1]
    floor.size = [0, 0, .1]
    if not any(t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for t in spec.textures):
        spec.add_texture(name="dex_viewer_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[.38, .51, .63], rgb2=[.78, .84, .88],
                         width=512, height=512)
    spec.visual.rgba.haze = [.72, .79, .83, 1]
    spec.visual.map.haze = .3
    spec.visual.map.zfar = 100
    spec.visual.headlight.ambient = [.4, .4, .4]
    spec.visual.headlight.diffuse = [.7, .7, .7]
    return spec.compile()
