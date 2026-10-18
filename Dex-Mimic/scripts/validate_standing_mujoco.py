"""PD response and free-standing policy validation, optionally with a GUI.

Candidate settings are runtime-only; source URDF/MJCF and deploy bundles are untouched.
"""
import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/whole_body_tracking/whole_body_tracking/robots"))
from motor_parameters import apply_mujoco_profile, load_profile
from sim2sim_dex import MujocoSim2Sim, NpzMotionStream, OnnxMotionPolicy, PolicyMetadata
from mujoco_node.policy_contract import TRAINING_CONTROL_DT

MODEL = ROOT / "source/whole_body_tracking/whole_body_tracking/assets/dex_evt/mjcf/dex_29.xml"
POLICY = ROOT / "ros_dex_deploy/src/rl_control/policy/mimic/out/policy_merged.onnx"


class Policy(OnnxMotionPolicy):
    def __init__(self, path):
        self.metadata = PolicyMetadata.from_onnx(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        self.input_obs_name, self.input_step_name = inputs[0].name, inputs[1].name
        self.output_names = [x.name for x in self.session.get_outputs()]


def configure_pd(model, meta, mode):
    joints = np.array([model.joint(n).id for n in meta.joint_names])
    qids, vids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    aids = np.array([next(i for i in range(model.nu) if model.actuator_trnid[i, 0] == j) for j in joints])
    model.actuator_gaintype[aids] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[aids] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[aids] = 0
    model.actuator_biasprm[aids] = 0
    if mode == "position":
        model.actuator_gainprm[aids, 0] = meta.joint_stiffness
        model.actuator_biasprm[aids, 1] = -meta.joint_stiffness
        model.actuator_biasprm[aids, 2] = -meta.joint_damping
        model.actuator_ctrlrange[aids] = np.stack([meta.joint_pos_lower, meta.joint_pos_upper], -1)
    else:
        model.actuator_gainprm[aids, 0] = 1
        model.actuator_ctrlrange[aids] = np.stack([-meta.joint_effort_limit, meta.joint_effort_limit], -1)
    model.actuator_forcelimited[aids] = 1
    model.actuator_forcerange[aids] = np.stack([-meta.joint_effort_limit, meta.joint_effort_limit], -1)
    return qids, vids, aids


def advance(model, data, meta, ids, target, mode, steps=1):
    qi, vi, ai = ids
    target = np.clip(target, meta.joint_pos_lower, meta.joint_pos_upper)
    for _ in range(steps):
        data.ctrl[ai] = target if mode == "position" else (
            meta.joint_stiffness * (target - data.qpos[qi]) - meta.joint_damping * data.qvel[vi])
        mujoco.mj_step(model, data)


def pd_response(meta, profile, mode, dt):
    tree = ET.parse(MODEL)
    root = tree.getroot()
    pelvis = root.find(".//body[@name='pelvis']")
    for joint in list(pelvis.findall("joint")) + list(pelvis.findall("freejoint")):
        if joint.tag == "freejoint" or joint.get("type") == "free":
            pelvis.remove(joint)
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", str((MODEL.parent / mesh.get("file")).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    apply_mujoco_profile(model, profile)
    model.opt.timestep = dt
    model.opt.gravity[:] = 0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    ids = configure_pd(model, meta, mode)
    target = meta.default_joint_pos.astype(np.float64)
    rows = []
    for col, name in enumerate(meta.joint_names):
        data = mujoco.MjData(model)
        data.qpos[ids[0]] = target
        data.qpos[ids[0][col]] += .03
        mujoco.mj_forward(model, data)
        peak_speed = 0.
        tail = []
        for i in range(round(3. / dt)):
            advance(model, data, meta, ids, target, mode)
            peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel[ids[1]]))))
            if i * dt > 2.:
                tail.append(float(np.max(np.abs(data.qpos[ids[0]] - target))))
        error = max(tail)
        rows.append(dict(joint=name, peak_speed_rad_s=peak_speed, tail_max_error_rad=error,
                         passed=bool(error < .005 and peak_speed < 10 and np.isfinite(data.qpos).all())))
    return dict(kind="fixed_base_no_gravity_no_contact_PD_response", profile=profile["name"],
                pd_mode=mode, dt=dt, pulse_rad=.03, all_passed=all(r["passed"] for r in rows), joints=rows)


def stand(policy, profile, args):
    stream = NpzMotionStream(str(ROOT / "dataset/standing/standing.npz"), policy.metadata.joint_names, policy.metadata.body_names)
    sim = MujocoSim2Sim(MODEL, policy, args.control_dt, args.sim_dt, True, None, motion_stream=stream)
    apply_mujoco_profile(sim.model, profile)
    ids = configure_pd(sim.model, policy.metadata, args.pd_mode)
    mujoco.mj_forward(sim.model, sim.data)
    ref = sim._reference_output(0)
    initial = sim.data.qpos[:3].copy()
    rows = []
    viewer = None
    if args.gui:
        from mujoco import viewer as viewer_module
        viewer = viewer_module.launch_passive(sim.model, sim.data)
        viewer.cam.distance = 3.5
        viewer.cam.lookat[:] = [0, 0, .9]
        viewer.cam.azimuth, viewer.cam.elevation = 90, -15
    renderer = mujoco.Renderer(sim.model, height=480, width=640) if args.capture else None
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .9]
    camera.distance, camera.azimuth, camera.elevation = 3.5, 90, -15
    fall = None
    try:
        for i in range(round(args.seconds / args.control_dt)):
            wall = time.monotonic()
            step = 0 if args.phase == "fixed" else int(i * args.control_dt * stream.fps + 1e-8) % (stream.T - 1)
            obs = sim.build_observation(step, ref, ref, tracking_weight=0.)
            action = policy.infer(obs, step)["actions"]
            target = ref["joint_pos"] + action * policy.metadata.action_scale
            advance(sim.model, sim.data, policy.metadata, ids, target, args.pd_mode, round(args.control_dt / args.sim_dt))
            sim.last_action = action
            q = sim.data.qpos[3:7]
            tilt = float(np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1))))
            row = dict(t=(i+1)*args.control_dt, z=float(sim.data.qpos[2]), tilt_deg=tilt,
                       xy_drift=float(np.linalg.norm(sim.data.qpos[:2]-initial[:2])),
                       joint_pos=sim.data.qpos[ids[0]].tolist(), joint_vel=sim.data.qvel[ids[1]].tolist())
            rows.append(row)
            if i == 0 or (i+1) % 250 == 0:
                print(json.dumps({k:v for k,v in row.items() if k not in ["joint_pos", "joint_vel"]}), flush=True)
            if renderer is not None and (i+1 == round(1./args.control_dt) or (i+1) == round(args.seconds/args.control_dt)):
                camera.lookat[:2] = sim.data.qpos[:2]
                renderer.update_scene(sim.data, camera=camera)
                from PIL import Image
                Image.fromarray(renderer.render()).save(args.output.with_name(args.output.stem+f"_{i+1:04d}.png"))
            if viewer is not None:
                viewer.cam.lookat[:2] = sim.data.qpos[:2]
                viewer.sync()
                if not viewer.is_running():
                    break
                time.sleep(max(0., args.control_dt-(time.monotonic()-wall)))
            if row["z"] < .55 or tilt > 60 or not np.isfinite(sim.data.qpos).all():
                fall = row["t"]
                break
    finally:
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
    return dict(kind="free_base_policy_standing", profile=profile, pd_mode=args.pd_mode, phase=args.phase,
                dt=args.sim_dt, control_dt=args.control_dt, training_control_dt=.02,
                model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(), requested_seconds=args.seconds,
                elapsed_seconds=rows[-1]["t"], fall_time=fall, max_drift_m=max(r["xy_drift"] for r in rows),
                max_tilt_deg=max(r["tilt_deg"] for r in rows), rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["none", "tg3-armature", "tg3-full"], default="tg3-armature")
    p.add_argument("--policy", type=Path, default=POLICY)
    p.add_argument("--pd-mode", choices=["position", "external"], default="external")
    p.add_argument("--sim-dt", type=float, default=.001)
    p.add_argument("--control-dt", type=float, default=TRAINING_CONTROL_DT,
                   help="Standalone inference period; default 0.02 s (50 Hz); use 0.01 only for a 100 Hz diagnostic")
    p.add_argument("--phase", choices=["fixed", "sweep"], default="fixed")
    p.add_argument("--seconds", type=float, default=60.)
    p.add_argument("--pd-response", action="store_true")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--capture", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (not np.isfinite(args.sim_dt) or not np.isfinite(args.control_dt)
            or args.sim_dt <= 0 or args.control_dt <= 0
            or not np.isclose(args.control_dt/args.sim_dt, round(args.control_dt/args.sim_dt))):
        p.error("physics dt must divide the inference period exactly")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile = load_profile(args.profile)
    policy = Policy(args.policy)
    result = pd_response(policy.metadata, profile, args.pd_mode, args.sim_dt) if args.pd_response else stand(policy, profile, args)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ["rows", "joints"]}), flush=True)


if __name__ == "__main__":
    main()
