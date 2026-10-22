"""Shared viewer scenery, compiled in memory without changing the robot asset."""
from pathlib import Path

import mujoco


def load_model_with_scene(model_path):
    """Match the xSIM reference floor / haze / lighting, keep our gradient skybox.

    Reference: 源头参考/xSIM_MUJOCO/resources/tiangong3/urdf/tiangong3.xml
      <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4"
               rgb2=".1 0.15 0.2" width="512" height="512"
               mark="cross" markrgb=".8 .8 .8"/>
      <material name="matplane" reflectance="0." texture="texplane"
                texrepeat="1 1" texuniform="true"/>
      <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1"
             pos="0 0 5.0" dir="0 0 -1" castshadow="false"/>
      <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2"
             pos="0 0 4" dir="0 0 -1"/>
    Haze / headlight stay at MuJoCo defaults, exactly like the reference (it has no
    <visual> block), so nothing here overrides them any more.

    Plane sizes only control rendering in MuJoCo. Its collision surface remains
    infinite; friction, contact parameters, robot and motor settings are retained.
    """
    spec = mujoco.MjSpec.from_file(str(Path(model_path).resolve()))
    floor = spec.geom("floor")
    if floor is None or floor.type != mujoco.mjtGeom.mjGEOM_PLANE:
        return spec.compile()

    spec.add_texture(name="dex_viewer_checker", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_CROSS,
                     rgb1=[.2, .3, .4], rgb2=[.1, .15, .2],
                     markrgb=[.8, .8, .8], width=512, height=512)
    material = spec.add_material(name="dex_viewer_floor", texuniform=True,
                                 texrepeat=[1, 1], rgba=[1, 1, 1, 1],
                                 specular=.5, shininess=.5, reflectance=0.)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "dex_viewer_checker"
    floor.material = material.name
    floor.rgba = [1, 1, 1, 1]
    floor.size = [0, 0, .1]

    # URDF 转换把碰撞体放在 group 1（见 dex_29.xml 的 collision 类），而 MuJoCo 默认渲染
    # group 0/1/2，于是 51 个凸包会叠在 39 个视觉网格外面，窗口里出现深色多边形补丁。
    # 改到 group 3 只影响渲染分组，几何/接触/摩擦不变；调试时勾选 group 3 仍可查看。
    for geom in spec.geoms:
        if geom.group == 1:
            geom.group = 3

    world = spec.worldbody
    world.add_light(name="dex_fill_light", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[.4, .4, .4], specular=[.1, .1, .1],
                    pos=[0, 0, 5], dir=[0, 0, -1], castshadow=False)
    world.add_light(name="dex_key_light", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                    diffuse=[.6, .6, .6], specular=[.2, .2, .2],
                    pos=[0, 0, 4], dir=[0, 0, -1])

    if not any(t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for t in spec.textures):
        spec.add_texture(name="dex_viewer_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[.38, .51, .63], rgb2=[.78, .84, .88],
                         width=512, height=512)
    return spec.compile()
