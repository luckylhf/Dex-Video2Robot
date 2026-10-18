"""Dex-GMR named PKL → latest Dex FK NPZ, with optional standing transitions."""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_to_npz_mujoco import (ROBOT_BODY_NAMES, ROBOT_JOINT_NAMES, ROBOT_MJCF,
    add_transition_to_npz, recompute_velocities_raw, report_velocity_anomalies,
    run_fk, _recompute_body_velocities_file)

def _slerp_batch(quats: np.ndarray, idx0: np.ndarray, idx1: np.ndarray,
                 blend: np.ndarray) -> np.ndarray:
    q0 = quats[idx0]
    q1 = quats[idx1]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(0.0, 1.0)
    linear = q0 + blend[:, None] * (q1 - q0)
    linear /= np.linalg.norm(linear, axis=-1, keepdims=True).clip(1e-8)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * blend[:, None]
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0.clip(1e-8)
    s1 = np.sin(theta) / sin_theta_0.clip(1e-8)
    slerp = s0 * q0 + s1 * q1
    slerp /= np.linalg.norm(slerp, axis=-1, keepdims=True).clip(1e-8)
    use_linear = (dot > 0.9995).squeeze(-1)
    return np.where(use_linear[:, None], linear, slerp).astype(np.float32)


def load_pkl(pkl_path: str, output_fps: int, robot: str) -> dict[str, np.ndarray]:
    if robot != "dex_evt":
        raise ValueError("This package supports Dex only")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    manifest = json.loads((Path(ROBOT_MJCF[robot]).parents[1] / "model.json").read_text())
    if data.get("model_sha256") != manifest["model_sha256"]:
        raise ValueError("PKL model mismatch: regenerate with the matching Dex-GMR model")
    names = list(data.get("joint_names", []))
    target = ROBOT_JOINT_NAMES[robot]
    if len(names) != len(set(names)) or set(names) != set(target):
        raise ValueError("PKL must name exactly the 29 unique Dex joints")
    if data.get("quat_order") != "xyzw":
        raise ValueError("PKL root quaternion order must explicitly be xyzw")
    fps = float(data["fps"])
    pos = np.asarray(data["root_pos"], dtype=float)
    quat = np.asarray(data["root_rot"], dtype=float)
    dof = np.asarray(data["dof_pos"], dtype=float)
    if not np.isfinite(fps) or fps <= 0 or output_fps <= 0:
        raise ValueError("fps must be positive")
    n = len(pos)
    if n < 2 or pos.shape != (n, 3) or quat.shape != (n, 4) or dof.shape != (n, 29):
        raise ValueError("PKL requires >=2 frames with root_pos(T,3), root_rot(T,4), dof_pos(T,29)")
    if not all(np.isfinite(a).all() for a in [pos, quat, dof]):
        raise ValueError("PKL contains NaN/Inf")
    if not np.allclose(np.linalg.norm(quat, axis=1), 1.0, atol=1e-3):
        raise ValueError("PKL xyzw quaternions must be normalized")
    dof = dof[:, [names.index(name) for name in target]]
    duration = (n - 1) / fps
    times = np.arange(int(np.floor(duration * output_fps + 1e-8)) + 1) / output_fps
    phase = times * fps
    idx0 = np.minimum(np.floor(phase).astype(int), n - 1)
    idx1 = np.minimum(idx0 + 1, n - 1)
    blend = np.clip(phase - idx0, 0, 1)
    return {
        "base_pos": ((1-blend[:, None])*pos[idx0] + blend[:, None]*pos[idx1]).astype(np.float32),
        "base_quat": _slerp_batch(quat[:, [3,0,1,2]], idx0, idx1, blend),
        "joint_pos": ((1-blend[:, None])*dof[idx0] + blend[:, None]*dof[idx1]).astype(np.float32),
    }


def convert_single(
    input_file: str,
    output_name: str,
    output_fps: int,
    robot: str,
    mjcf_path: str,
) -> None:
    joint_names = ROBOT_JOINT_NAMES[robot]
    body_names  = ROBOT_BODY_NAMES[robot]

    arrays = load_pkl(input_file, output_fps, robot)

    body_pos_w, body_quat_w = run_fk(
        base_pos=arrays["base_pos"],
        base_quat=arrays["base_quat"],
        joint_pos=arrays["joint_pos"],
        joint_names=joint_names,
        body_names=body_names,
        mjcf_path=mjcf_path,
    )
    joint_vel, body_lin_vel_w, body_ang_vel_w = recompute_velocities_raw(
        joint_pos=arrays["joint_pos"],
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        fps=float(output_fps),
    )
    report_velocity_anomalies(joint_vel, body_lin_vel_w, body_ang_vel_w, joint_names, output_fps, input_file)

    out_path = output_name if output_name.endswith(".npz") else output_name + ".npz"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(mjcf_path).parents[1] / "model.json").read_text())
    np.savez(
        out_path,
        model_sha256=np.array(manifest["model_sha256"]),
        quat_order=np.array("wxyz"),
        fps=np.array(output_fps),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
        joint_pos=arrays["joint_pos"],
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"[pkl_to_npz] saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PKL → NPZ pipeline: convert, add transition, recompute velocities."
    )

    parser.add_argument("--input_file",  default=None)
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--batch_dir",   default=None)
    parser.add_argument("--raw_output_dir", default=None)
    parser.add_argument("--output_dir",  default=None)

    parser.add_argument("--robot",      default="dex_evt", choices=list(ROBOT_MJCF.keys()), help="Robot type (this package ships dex_evt only)")
    parser.add_argument("--mjcf",       default=None)
    parser.add_argument("--output_fps", type=int, default=50)

    parser.add_argument("--force", action="store_true")

    parser.add_argument("--add_transition",      action="store_true")
    parser.add_argument("--hold_duration",        type=float, default=1.0)
    parser.add_argument("--trans_duration",       type=float, default=1.0)
    parser.add_argument("--tail_trans_duration",  type=float, default=2.5)
    parser.add_argument("--tail_hold_duration",   type=float, default=1.0)
    parser.add_argument("--stand_height",         type=float, default=0.87)
    parser.add_argument("--head_skip_frames",     type=int,   default=-1)
    parser.add_argument("--tail_skip_frames",     type=int,   default=-1)
    parser.add_argument("--skip_threshold",       type=float, default=1.5)

    parser.add_argument("--recompute_velocities", action="store_true")

    args = parser.parse_args()
    mjcf_path = args.mjcf or ROBOT_MJCF[args.robot]

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch_dir is not None:
        batch_dir = Path(args.batch_dir)
        pkl_files = sorted(batch_dir.glob("*.pkl"))
        if not pkl_files:
            print(f"[batch] No PKL files found in {batch_dir}")
            sys.exit(1)

        if args.add_transition:
            # Two-phase: raw → transition
            root = batch_dir.parent
            raw_dir = Path(args.raw_output_dir) if args.raw_output_dir else root / "npz_dex_raw"
            out_dir = Path(args.output_dir)     if args.output_dir     else root / "npz_dex"
            raw_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"=== Phase 1: PKL → NPZ (+ transition) ===")
            for pkl in pkl_files:
                name = pkl.stem
                raw_npz        = raw_dir / f"{name}.npz"
                transition_npz = out_dir  / f"{name}_with_transition.npz"

                if not raw_npz.exists() or args.force:
                    print(f"[CONVERT] {name}")
                    convert_single(str(pkl), str(raw_npz.with_suffix("")),
                                   args.output_fps, args.robot, mjcf_path)
                else:
                    print(f"[SKIP] {name} raw npz already exists")

                if transition_npz.exists() and not args.force:
                    print(f"[SKIP] {name}_with_transition.npz already exists")
                    continue
                print(f"[TRANSITION] {name}")
                add_transition_to_npz(
                    raw_npz=str(raw_npz),
                    output_npz=str(transition_npz),
                    hold_duration=args.hold_duration,
                    trans_duration=args.trans_duration,
                    tail_trans_duration=args.tail_trans_duration,
                    tail_hold_duration=args.tail_hold_duration,
                    stand_height=args.stand_height,
                    head_skip_frames=args.head_skip_frames,
                    tail_skip_frames=args.tail_skip_frames,
                    skip_threshold=args.skip_threshold,
                    mjcf_path=mjcf_path,
                )
            print(f"\n=== Done ===")

            if args.recompute_velocities:
                print(f"\n=== Phase 2: Recompute body velocities in {out_dir} ===")
                for p in sorted(out_dir.glob("*.npz")):
                    _recompute_body_velocities_file(p)
        else:
            # Direct mode: PKL → NPZ to output_dir
            out_dir = Path(args.output_dir) if args.output_dir else batch_dir.parent / "npz_dex"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"=== PKL → NPZ (direct, no transition) ===")
            print(f"  Output: {out_dir}")
            for pkl in pkl_files:
                name = pkl.stem
                out_npz = out_dir / f"{name}.npz"
                if out_npz.exists() and not args.force:
                    print(f"[SKIP] {name}.npz already exists")
                    continue
                print(f"[CONVERT] {name}")
                convert_single(str(pkl), str(out_npz.with_suffix("")),
                               args.output_fps, args.robot, mjcf_path)
            print(f"\n=== Done ===")

            if args.recompute_velocities:
                print(f"\n=== Recompute body velocities in {out_dir} ===")
                for p in sorted(out_dir.glob("*.npz")):
                    _recompute_body_velocities_file(p)
        return

    # ── Single-file mode ──────────────────────────────────────────────────────
    if args.input_file is None or args.output_name is None:
        parser.error("Provide --input_file and --output_name, or --batch_dir.")

    convert_single(args.input_file, args.output_name,
                   args.output_fps, args.robot, mjcf_path)

    if args.add_transition:
        raw_npz = args.output_name if args.output_name.endswith(".npz") else args.output_name + ".npz"
        base = args.output_name[:-4] if args.output_name.endswith(".npz") else args.output_name
        transition_path = str(Path(args.output_dir) / (Path(base).stem + "_with_transition.npz")) if args.output_dir else base + "_with_transition.npz"
        Path(transition_path).parent.mkdir(parents=True, exist_ok=True)
        add_transition_to_npz(
            raw_npz=raw_npz,
            output_npz=transition_path,
            hold_duration=args.hold_duration,
            trans_duration=args.trans_duration,
            tail_trans_duration=args.tail_trans_duration,
            tail_hold_duration=args.tail_hold_duration,
            stand_height=args.stand_height,
            head_skip_frames=args.head_skip_frames,
            tail_skip_frames=args.tail_skip_frames,
            skip_threshold=args.skip_threshold,
            mjcf_path=mjcf_path,
        )

        if args.recompute_velocities:
            _recompute_body_velocities_file(Path(transition_path))


if __name__ == "__main__":
    main()
