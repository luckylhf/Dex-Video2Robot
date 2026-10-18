"""Portable policy identity and simulation dynamics; no ROS or hardware imports."""
import hashlib
import json
from pathlib import Path

import numpy as np

CONTRACT_VERSION = "2"
TRAINING_CONTROL_DT = 0.02
DEPLOYMENT_CONTROL_DT = 0.01


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(model_dir):
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "model.json").read_text())
    if file_sha256(model_dir / "mjcf/dex_29.xml") != manifest["model_sha256"]:
        raise ValueError("MJCF has changed: rebuild the model and regenerate matching motion/policy data")
    if file_sha256(model_dir / "urdf/tiangong3_source.urdf") != manifest["source_urdf_sha256"]:
        raise ValueError("Source URDF has changed: rebuild the robot assets")
    if file_sha256(model_dir / "urdf/dex_29.urdf") != manifest.get("urdf_sha256"):
        raise ValueError("Working URDF has changed: rebuild the robot assets")
    return manifest


def asset_sha256(model_dir):
    """Include both URDFs and meshes; ignore generated USD caches and viewer code."""
    root = Path(model_dir)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".xml", ".urdf", ".stl"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def dynamics_from_metadata(meta):
    if meta.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Policy lacks the release dynamics contract; export it again with the current pipeline")
    names = meta.get("joint_names", "").split(",")
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Policy must name exactly 29 unique joints")
    arrays = {}
    for field in ("joint_armature", "joint_passive_damping", "joint_frictionloss"):
        values = np.fromstring(meta.get(field, ""), sep=",")
        if values.shape != (29,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid dynamics field: {field}")
        arrays[field] = values
    if meta.get("mujoco_integrator") != "implicitfast" or meta.get("mujoco_solver") != "CG":
        raise ValueError("Unsupported MuJoCo integrator/solver in policy")
    if not np.isclose(float(meta.get("control_dt", 0)), TRAINING_CONTROL_DT):
        raise ValueError("Expected training control_dt=0.02 s; this is separate from deployment inference timing")
    return names, arrays


def validate_deployment_timing(meta, physical_dt, decimation=1):
    """Preserve training provenance while enforcing the tested hardware minimum."""
    if not np.isclose(float(meta.get("control_dt", 0)), TRAINING_CONTROL_DT):
        raise ValueError("Expected training control_dt=0.02 s")
    physical_dt, decimation = float(physical_dt), float(decimation)
    if (not np.isfinite(physical_dt) or physical_dt <= 0
            or not np.isfinite(decimation) or decimation < 1 or not decimation.is_integer()):
        raise ValueError("Invalid deployment physical_dt or decimation")
    period = physical_dt * decimation
    if period > DEPLOYMENT_CONTROL_DT + 1e-9:
        raise ValueError("Dex deployment requires at least 100 Hz fresh policy inference; repeating targets is insufficient")
    return period


def validate_identity(meta, model_dir):
    manifest = verify_model(model_dir)
    if meta.get("model_sha256") != manifest["model_sha256"]:
        raise ValueError("Policy and robot model differ")
    if meta.get("asset_sha256") != asset_sha256(model_dir):
        raise ValueError("Policy URDF/meshes differ from current assets; re-export after checking the model")
    dynamics_from_metadata(meta)
    return manifest


def apply_policy_dynamics(model, meta):
    import mujoco
    names, arrays = dynamics_from_metadata(meta)
    for index, name in enumerate(names):
        dof = int(model.jnt_dofadr[model.joint(name).id])
        model.dof_armature[dof] = arrays["joint_armature"][index]
        model.dof_damping[dof] = arrays["joint_passive_damping"][index]
        model.dof_frictionloss[dof] = arrays["joint_frictionloss"][index]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG


def resolve_mimic_config(default_path, explicit_path=None):
    """One selected bundle for both FSMs and launch-time validation."""
    import os
    import yaml
    path = Path(explicit_path or os.environ.get("MIMIC_CONFIG_PATH") or default_path).resolve()
    config = yaml.safe_load(path.read_text())
    bundle = os.environ.get("MIMIC_BUNDLE_DIR")
    if bundle:
        bundle = Path(bundle).expanduser().resolve()
        config["onnx_path"] = str(bundle / "policy_merged.onnx")
        config["npz_dir"] = str(bundle / "npz")
    return config
