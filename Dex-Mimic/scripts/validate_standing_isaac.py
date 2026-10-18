"""Single-robot standing validation with live physics, runtime motor audit and GUI.

No automatic reset, motion replay teleportation, policy export or bundle replacement.
Randomization/delay/noise are disabled only in this diagnostic entry point.
"""
import argparse
import faulthandler
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", choices=["none", "tg3-armature"], default="tg3-armature")
parser.add_argument("--seconds", type=float, default=60.)
parser.add_argument("--phase", choices=["fixed", "sweep"], default="fixed")
parser.add_argument("--checkpoint", type=Path, default=ROOT / "policy/dex/standing_baseline.pt")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sim-dt", type=float, default=.001)
parser.add_argument("--realtime", action="store_true")
parser.add_argument("--render-every", type=int, default=10, help="Render every N policy steps; physics/policy rates stay unchanged")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
os.environ["DEX_MOTOR_PROFILE"] = args.profile
faulthandler.dump_traceback_later(120, repeat=True)
app_launcher = AppLauncher(args, width=1280, height=720, window_width=1280, window_height=720)
app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import whole_body_tracking.tasks
from isaaclab_tasks.utils import parse_env_cfg
from whole_body_tracking.robots.motor_parameters import load_profile, armature_for_joint
import isaaclab.sim as sim_utils
torch.set_num_threads(1)


def capture(path, simulator):
    try:
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
        viewport = get_active_viewport()
        if viewport is not None:
            capture_viewport_to_file(viewport, str(path))
            for _ in range(10):
                simulator.render()
    except Exception as exc:
        print(f"[capture] {type(exc).__name__}: {exc}", flush=True)


def main():
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.sim_dt <= 0 or not np.isclose(.02/args.sim_dt, round(.02/args.sim_dt)):
        raise ValueError("physics dt must divide the 0.02 s policy period")
    cfg = parse_env_cfg("Tracking-Flat-DexEVT-Simple-v0", device=args.device, num_envs=1)
    cfg.commands.motion.motion_file = str(ROOT / "dataset/standing/standing.npz")
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.25, .25, .25))
    cfg.sim.dt, cfg.decimation = args.sim_dt, round(.02/args.sim_dt)
    if args.render_every < 1:
        raise ValueError("render-every must be positive")
    cfg.sim.render_interval = cfg.decimation * args.render_every
    cfg.episode_length_s = args.seconds + 10
    cfg.observations.policy.enable_corruption = False
    # These diagnostics measure uninterrupted survival; a reset must never mask a fall.
    for name, value in list(vars(cfg.terminations).items()):
        if not name.startswith("_") and value is not None:
            setattr(cfg.terminations, name, None)
    for name, value in list(vars(cfg.events).items()):
        if not name.startswith("_") and value is not None:
            setattr(cfg.events, name, None)
    for actuator in cfg.scene.robot.actuators.values():
        actuator.min_delay = actuator.max_delay = 0
    # Inherit the task's asset_root camera; shared with normal RSL-RL play.
    print("[validation] Creating deterministic environment", flush=True)
    env = gym.make("Tracking-Flat-DexEVT-Simple-v0", cfg=cfg)
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_term("motion")
    env.reset()
    materials = robot.root_physx_view.get_material_properties()
    materials[..., :2] = 1.0
    materials[..., 2] = 0.0
    robot.root_physx_view.set_material_properties(materials, indices=torch.arange(1, dtype=torch.int32))
    names = robot.joint_names
    profile = load_profile(args.profile)
    expected = np.array([armature_for_joint(profile, n) for n in names])
    actual = robot.data.joint_armature[0].cpu().numpy()
    if not np.allclose(actual, expected, atol=1e-6):
        raise RuntimeError(f"Armature did not reach PhysX: actual={actual}, expected={expected}")
    # Initialize once, in robot name order; no further state writes during rollout.
    root = robot.data.default_root_state.clone()
    root[:, :3] = command.motion.body_pos_w[0, 0] + env.unwrapped.scene.env_origins
    root[:, 3:7] = command.motion.body_quat_w[0, 0]
    root[:, 7:] = 0
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(command.motion.joint_pos[0:1].clone(), torch.zeros_like(command.motion.joint_pos[0:1]))
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    env.unwrapped.scene.update(args.sim_dt)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    sd = checkpoint["model_state_dict"]
    layers = []
    for index in [0, 2, 4, 6]:
        weight = sd[f"actor.{index}.weight"]
        layer = torch.nn.Linear(weight.shape[1], weight.shape[0], device=args.device)
        layer.weight.data.copy_(weight)
        layer.bias.data.copy_(sd[f"actor.{index}.bias"])
        layers.append(layer)
        if index != 6:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).eval()
    norm = checkpoint["obs_norm_state_dict"]
    mean, std = norm["_mean"], norm["_std"]
    initial = robot.data.root_pos_w[0].cpu().numpy().copy()
    rows = []
    report = dict(profile=profile, checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  dt=args.sim_dt, control_dt=.02, phase=args.phase, gui=not args.headless,
                  randomization=False, auto_reset=False, joint_names=names, body_names=robot.body_names,
                  actual_armature=actual.tolist(), kp=robot.data.joint_stiffness[0].cpu().tolist(),
                  kd=robot.data.joint_damping[0].cpu().tolist(), requested_seconds=args.seconds,
                  actual_joint_friction=robot.data.joint_friction_coeff[0].cpu().tolist(),
                  body_masses=robot.root_physx_view.get_masses()[0].cpu().tolist(),
                  nominal_contact_friction=1.0,
                  rows=rows, fall_time=None)
    try:
        from pxr import Usd, UsdPhysics
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        report["foot_collision_prims"] = [{"path": str(prim.GetPath()), "type": prim.GetTypeName()}
                                         for prim in stage.Traverse(Usd.TraverseInstanceProxies())
                                         if "ankle_roll" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI)]
    except Exception as exc:
        report["geometry_audit_error"] = str(exc)
    print("[runtime] " + json.dumps({k:v for k,v in report.items() if k not in ["rows"]}), flush=True)
    faulthandler.cancel_dump_traceback_later()
    try:
        for i in range(round(args.seconds/.02)):
            if not app.is_running():
                break
            wall = time.monotonic()
            # Keep static reference alive beyond its 5 s training clip without resetting the robot.
            command.time_steps.fill_(0 if args.phase == "fixed" else i % (command.motion.time_step_total-2))
            obs = env.unwrapped.observation_manager.compute_group("policy")
            if args.phase == "fixed":
                obs[:, 154:156] = torch.tensor([0., 1.], device=obs.device)
            with torch.inference_mode():
                action = actor((obs-mean)/(std+.01))
                env.step(action)
            pos = robot.data.root_pos_w[0].cpu().numpy()
            q = robot.data.root_quat_w[0].cpu().numpy()
            tilt = float(np.degrees(np.arccos(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1))))
            row = dict(t=(i+1)*.02, z=float(pos[2]), tilt_deg=tilt,
                       xy_drift=float(np.linalg.norm(pos[:2]-initial[:2])),
                       joint_pos=robot.data.joint_pos[0].cpu().tolist(),
                       joint_vel=robot.data.joint_vel[0].cpu().tolist())
            rows.append(row)
            if i == 0 or (i+1) % 50 == 0:
                print(json.dumps({k:v for k,v in row.items() if k not in ["joint_pos", "joint_vel"]}), flush=True)
            if not args.headless and (i == 49 or (i+1) == round(args.seconds/.02)):
                capture(args.output.with_name(args.output.stem+f"_{i+1:04d}.png"), env.unwrapped.sim)
            if row["z"] < .55 or tilt > 60 or not np.isfinite(pos).all():
                report["fall_time"] = row["t"]
                break
            if args.realtime:
                time.sleep(max(0., .02-(time.monotonic()-wall)))
    finally:
        report["elapsed_seconds"] = rows[-1]["t"] if rows else 0
        report["max_drift_m"] = max((r["xy_drift"] for r in rows), default=0.)
        report["max_tilt_deg"] = max((r["tilt_deg"] for r in rows), default=0.)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print("[summary] " + json.dumps({k: report[k] for k in ["elapsed_seconds", "fall_time", "max_drift_m", "max_tilt_deg"]}), flush=True)
        env.close()


try:
    main()
finally:
    faulthandler.cancel_dump_traceback_later()
    app.close()
