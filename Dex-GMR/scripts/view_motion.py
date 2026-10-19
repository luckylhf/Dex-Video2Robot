"""查看已导出的 Dex PKL，属于运动学显示。"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion', type=Path)
    parser.add_argument('--loop', action='store_true', help='Repeat until the viewer window is closed')
    args = parser.parse_args()
    from general_motion_retargeting import RobotMotionViewer
    from general_motion_retargeting.data_loader import load_robot_motion
    motion, fps, pos, quat, joints, _, _ = load_robot_motion(args.motion)
    if motion['model_sha256'] != json.loads((ROOT / 'assets/dex/model.json').read_text())['model_sha256']:
        raise ValueError('Motion belongs to a different robot model')
    viewer = RobotMotionViewer(robot_type='dex_evt', motion_fps=fps)
    try:
        while viewer.viewer.is_running():
            for p, q, j in zip(pos, quat, joints):
                if not viewer.viewer.is_running():
                    break
                viewer.step(root_pos=p, root_rot=q, dof_pos=j, rate_limit=True)
            if not args.loop:
                break
    finally:
        viewer.close()
