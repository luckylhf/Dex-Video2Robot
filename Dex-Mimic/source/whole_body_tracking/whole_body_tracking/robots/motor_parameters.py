"""Optional candidate motor parameters; no body/geometry or asset edits."""
import hashlib
import json
import re
from pathlib import Path

PROFILE_FILE = Path(__file__).with_name("tg3_motor_profile.json")


def load_profile(name):
    if name not in ("none", "tg3-armature", "tg3-full"):
        raise ValueError(f"Unknown motor profile: {name}")
    data = json.loads(PROFILE_FILE.read_text())
    data["name"] = name
    data["sha256"] = hashlib.sha256(PROFILE_FILE.read_bytes()).hexdigest()
    return data


def armature_for_joint(profile, name):
    if profile["name"] == "none":
        return 0.0
    matches = [v for pattern, v in profile["armature_kg_m2"].items() if re.fullmatch(pattern, name)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one armature mapping for {name}, got {matches}")
    return matches[0]


def apply_isaac_profile(actuators, profile):
    if profile["name"] == "tg3-full":
        raise ValueError("MuJoCo frictionloss (Nm) is not PhysX joint friction (unitless). Use tg3-armature for cross-engine validation.")
    if profile["name"] == "none":
        return
    manifest = json.loads((PROFILE_FILE.parents[1] / "assets/dex_evt/model.json").read_text())
    for cfg in actuators.values():
        cfg.armature = {name: armature_for_joint(profile, name) for name in manifest["joint_names"]
                        if any(re.fullmatch(expr, name) for expr in cfg.joint_names_expr)}
        if not cfg.armature:
            raise ValueError(f"No armature mapping for actuator {cfg.joint_names_expr}")
        cfg.friction = 0.0
        cfg.dynamic_friction = 0.0
        cfg.viscous_friction = 0.0


def apply_mujoco_profile(model, profile):
    import mujoco
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        dof = model.jnt_dofadr[joint]
        model.dof_armature[dof] = armature_for_joint(profile, name)
        model.dof_damping[dof] = profile["mujoco_passive_damping"] if profile["name"] == "tg3-full" else 0.0
        model.dof_frictionloss[dof] = profile["mujoco_frictionloss_Nm"] if profile["name"] == "tg3-full" else 0.0
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG

