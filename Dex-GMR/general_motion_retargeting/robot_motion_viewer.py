import os
import time
import mujoco as mj
import mujoco.viewer as mjv
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print


def draw_frame(
    pos,
    mat,
    v,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    for i in range(3):
        geom = v.user_scn.geoms[v.user_scn.ngeom]
        mj.mjv_initGeom(
            geom,
            type=mj.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=pos + pos_offset,
            mat=mat.flatten(),
            rgba=rgba_list[i],
        )
        if joint_name is not None:
            geom.label = joint_name  # 这里赋名字
        fix = orientation_correction.as_matrix()
        mj.mjv_connector(
            v.user_scn.geoms[v.user_scn.ngeom],
            type=mj.mjtGeom.mjGEOM_ARROW,
            width=0.005,
            from_=pos + pos_offset,
            to=pos + pos_offset + size * (mat @ fix)[:, i],
        )
        v.user_scn.ngeom += 1

class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                headless=False,
                # video recording
                record_video=False,
                video_path=None,
                video_width=640,
                video_height=480):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video
        self.headless = headless

        if not headless:
            self.viewer = mjv.launch_passive(
                model=self.model,
                data=self.data,
                show_left_ui=True,
                show_right_ui=True
                )
            self.viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = transparent_robot
        else:
            self.viewer = None
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path
            video_dir = os.path.dirname(self.video_path)
            
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
            
            # Initialize renderer for video recording
            self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        
        # Auto-discover end-effector bodies by keyword matching
        def _find_end_body(model, keyword, side):
            """Find the last body matching keyword+side in the kinematic chain."""
            for i in range(model.nbody):
                bname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or ''
                bl = bname.lower()
                if keyword not in bl:
                    continue
                if side == 'left':
                    if any(p in bl for p in ('_l_', '_l_link', 'left')) or bl.startswith('l_'):
                        return bname
                elif side == 'right':
                    if any(p in bl for p in ('_r_', '_r_link', 'right')) or bl.startswith('r_'):
                        return bname
            return None

        _auto_map = {
            'left_hand_end':  _find_end_body(self.model, 'wrist', 'left'),
            'right_hand_end': _find_end_body(self.model, 'wrist', 'right'),
            'left_foot_end':  _find_end_body(self.model, 'ankle', 'left'),
            'right_foot_end': _find_end_body(self.model, 'ankle', 'right'),
        }
        self.end_site_names = {k: [v] for k, v in _auto_map.items() if v is not None}

        for key, value in self.end_site_names.items():
            try:
                sid = self.model.site(key).id
            except Exception:
                sid = -1
            self.end_site_names[key].append(sid)
            self.end_site_names[key].append(self.model.body(value[0]).id)

        self.pre_dof_pos = None
        # Build ordered list of non-free joint qpos addresses for get_info
        free_joint_ids = {
            j for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
        }
        self._dof_joint_names = []
        self._dof_qpos_adrs = []
        for j in range(self.model.njnt):
            if j in free_joint_ids:
                continue
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
            if name is None:
                continue
            self._dof_joint_names.append(name)
            self._dof_qpos_adrs.append(int(self.model.jnt_qposadr[j]))
        
    def get_info(self):
        qpos = self.data.qpos.copy()
        dof_pos = np.array([qpos[adr] for adr in self._dof_qpos_adrs], dtype=np.float64)

        if self.pre_dof_pos is None:
            dof_vel = np.zeros_like(dof_pos, dtype=np.float64)
        else:
            dof_vel = (dof_pos - self.pre_dof_pos) / (1.0 / self.motion_fps)

        self.pre_dof_pos = dof_pos.copy()

        # Get end-effector positions in root-local frame
        root_body_name = self.robot_base
        root_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, root_body_name)
        end_pos = []
        poss_world = []
        for name, info in self.end_site_names.items():
            if info[1] == -1:
                continue
            pos_world = np.array(self.data.site_xpos[info[1]])
            root_pos = np.array(self.data.xpos[root_id])
            root_xmat = np.array(self.data.xmat[root_id]).reshape(3, 3)
            pos_local = root_xmat.T.dot(pos_world - root_pos)
            poss_world.append(pos_world)
            end_pos += pos_local.tolist()

        self.poss_world = poss_world
        self.end_pos = end_pos
        dof_pos_full = np.append(qpos[:7], dof_pos)
        return dof_pos_full, dof_vel, np.array(end_pos)
    
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        by default visualize robot motion.
        also support visualize human motion by providing human_motion_data, to compare with robot motion.
        
        human_motion_data is a dict of {"human body name": (3d global translation, 3d global rotation)}.

        if rate_limit is True, the motion will be visualized at the same rate as the motion data.
        else, the motion will be visualized as fast as possible.
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot # quat need to be scalar first! for mujoco
        self.data.qpos[7:] = dof_pos

        mj.mj_forward(self.model, self.data)

        if self.headless:
            if rate_limit is True:
                self.rate_limiter.sleep()
            return

        if follow_camera:
            self.viewer.cam.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.viewer.cam.distance = self.viewer_cam_distance

        if human_motion_data is not None:
            # Clean custom geometry
            self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            for human_body_name, (pos, rot) in human_motion_data.items():
                draw_frame(
                    pos,
                    R.from_quat(rot, scalar_first=True).as_matrix(),
                    self.viewer,
                    human_point_scale,
                    pos_offset=human_pos_offset,
                    joint_name=human_body_name if show_human_body_name else None
                    )

        self.viewer.sync()
        if rate_limit is True:
            self.rate_limiter.sleep()

        if self.record_video:
            # Use renderer for proper offscreen rendering
            self.renderer.update_scene(self.data, camera=self.viewer.cam)
            img = self.renderer.render()
            self.mp4_writer.append_data(img)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            time.sleep(0.5)
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
