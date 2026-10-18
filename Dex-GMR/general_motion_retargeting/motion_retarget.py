
import mink
import mujoco as mj
import numpy as np
import json
import time
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT, ROBOT_URDF_DICT
from .collision_config import COLLISION_CONFIG
from rich import print
from general_motion_retargeting.ground_avoidance_task import GroundAvoidanceTask
from general_motion_retargeting.geom_distance_limit import GeomDistanceLimit, FootGroundContactLimit, ObstacleContactLimit
from general_motion_retargeting.foot_contact_task import FootContactTask

VALOCITY_MAX = 0.7

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=False,
        use_velocity_limit: bool=False,
        use_collision_avoidance: bool=True,
        motion_fps: int=30
    ) -> None:
        self.verbose = verbose
        self.ground_offset = None
        self.lowest_pos = np.inf
        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        # Ordered joint names matching qpos[7:] (dof_pos), excluding free joint.
        # This list is used by downstream consumers (e.g. pkl_to_npz) to map
        # dof_pos columns → joint names for the saved PKL.
        self.robot_joint_names = []
        joint_info = []
        for jnt_id in range(self.model.njnt):
            if self.model.jnt_type[jnt_id] == mj.mjtJoint.mjJNT_FREE:
                continue
            jnt_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
            qposadr = self.model.jnt_qposadr[jnt_id]
            joint_info.append((qposadr, jnt_name))
        joint_info.sort()
        self.robot_joint_names = [name for _, name in joint_info]
        
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")
        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        # for key in ik_config["human_scale_table"].keys():0    
        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping
        self.solve_error_arr = []

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.task_to_human_body1 = {}  # reverse mapping: task → human body name
        self.task_to_human_body2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}
        self.motion_fps = motion_fps
        self.retarget_dt = 1.0 / motion_fps
        self.use_velocity_limit = use_velocity_limit

        self.task_errors1 = {}
        self.task_errors2 = {}
        self.flag = False
        # self.ik_limits = []
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        self.velocity_limits = self.load_velocity_limits(tgt_robot)
        if use_velocity_limit:
            if self.velocity_limits:
                self.ik_limits.append(mink.VelocityLimit(self.model, self.velocity_limits))
            elif self.verbose:
                print("[warn] no valid joint velocity limits found, skip velocity limiter")
            
        if use_collision_avoidance:
            col_cfg = COLLISION_CONFIG.get(tgt_robot, {})
            col1_pairs = col_cfg.get("collision_avoidance_pairs1", [])
            col2_pairs = col_cfg.get("collision_avoidance_pairs2", [])
            foot_ground_geoms = col_cfg.get("foot_ground_geoms", [])
            col_params = col_cfg.get("collision_params", {})

            configured_geoms = set(foot_ground_geoms)
            for a, b in col1_pairs + col2_pairs:
                configured_geoms.update(a + b)
            missing_geoms = sorted(n for n in configured_geoms
                                   if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, n) < 0)
            if missing_geoms:
                raise ValueError(f"Collision configuration references missing geometry: {missing_geoms}")

            if col1_pairs:
                valid_col1 = [(a, b) for a, b in col1_pairs
                              if all(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, n) >= 0 for n in a + b)]
                if valid_col1:
                    self.col_limit1 = GeomDistanceLimit(
                        model=self.model,
                        geom_pairs=valid_col1,
                        gain=col_params.get("gain", 0.1),
                        minimum_distance_from_collisions=col_params.get("minimum_distance_from_collisions", 0.02),
                        collision_detection_distance=col_params.get("collision_detection_distance", 0.14),
                        bound_relaxation=col_params.get("bound_relaxation", 0.005),
                    )
                    self.ik_limits.append(self.col_limit1)

            if col2_pairs:
                valid_col2 = [(a, b) for a, b in col2_pairs
                              if all(mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, n) >= 0 for n in a + b)]
                if valid_col2:
                    self.col_limit2 = GeomDistanceLimit(
                        model=self.model,
                        geom_pairs=valid_col2,
                        gain=col_params.get("gain", 0.08),
                        minimum_distance_from_collisions=col_params.get("minimum_distance_from_collisions", 0.075),
                        collision_detection_distance=col_params.get("collision_detection_distance", 0.17),
                        bound_relaxation=col_params.get("bound_relaxation", 0.005),
                    )
                    self.ik_limits.append(self.col_limit2)

            ground_geom = None
            for candidate in ("ground", "floor"):
                if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, candidate) >= 0:
                    ground_geom = candidate
                    break
            valid_foot_geoms = [
                name for name in foot_ground_geoms
                if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, name) >= 0
            ]
            if ground_geom and valid_foot_geoms:
                self.foot_ground_limit = FootGroundContactLimit(
                    model=self.model,
                    foot_geom_names=valid_foot_geoms,
                    min_height=0.0
                )
                self.ik_limits.append(self.foot_ground_limit)

        # 肩关节 PostureTask：防止手臂举起后漂移无法放下
        self.shoulder_posture_task = None
        posture_costs = COLLISION_CONFIG.get(tgt_robot, {}).get("shoulder_posture_costs", {})
        if posture_costs:
            cost_array = np.zeros(self.model.nv)
            for jname, cost in posture_costs.items():
                jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
                if jid >= 0:
                    cost_array[self.model.jnt_dofadr[jid]] = cost
            if cost_array.any():
                self.shoulder_posture_task = mink.PostureTask(self.model, cost=cost_array)
                self.shoulder_posture_task.set_target(self.model.qpos0.copy())

        # ── FootContactTask：防止脚部着地后水平滑动 ──────────────────────
        self.foot_contact_tasks: list[FootContactTask] = []
        self._foot_contact_human_body: dict[FootContactTask, str] = {}  # task → human body name
        self._foot_contact_cfg = ik_config.get("foot_contact", {})
        if self._foot_contact_cfg.get("enabled", False):
            ft_cfg = self._foot_contact_cfg
            foot_task_map: dict[str, str] = ft_cfg.get("foot_tasks", {})
            for robot_body, human_body in foot_task_map.items():
                if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, robot_body) < 0:
                    if verbose:
                        print(f"[warn] FootContactTask: body '{robot_body}' 不存在，跳过")
                    continue
                task = FootContactTask(
                    model=self.model,
                    body_name=robot_body,
                    ground_height=ft_cfg.get("ground_height", 0.0),
                    cost=ft_cfg.get("cost", 500.0),
                    gain=ft_cfg.get("gain", 0.95),
                    stance_threshold=ft_cfg.get("stance_threshold", 0.04),
                    hysteresis=ft_cfg.get("hysteresis", 0.02),
                )
                self.foot_contact_tasks.append(task)
                self._foot_contact_human_body[task] = human_body
                if verbose:
                    print(f"[GMR] FootContactTask: {robot_body} ← {human_body} "
                          f"cost={ft_cfg.get('cost', 500)}, "
                          f"ground={ft_cfg.get('ground_height', 0.0):.3f}")

        self.setup_retarget_configuration()

        # Auto-detect foot collision geoms for root-Z grounding compensation.
        # Falls back gracefully if no plausible foot geom is found.
        self._foot_geom_ids = self._identify_foot_geoms()
        self._foot_z_correction = None

    def _identify_foot_geoms(self):
        """Find collision geoms in 'ankle'/'foot' subtrees that participate in contact.

        Returns a list of geom ids. Geoms with both contype=0 and conaffinity=0
        (visualization-only) are skipped because they don't represent real contact.
        Falls back to all collidable geoms if no foot/ankle subtree is found.
        """
        foot_body_ids = set()
        for b in range(self.model.nbody):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, b) or ""
            lname = name.lower()
            if "ankle" in lname or "foot" in lname:
                foot_body_ids.add(b)

        def _is_collidable(g):
            return int(self.model.geom_contype[g]) != 0 or int(self.model.geom_conaffinity[g]) != 0

        geom_ids = [
            g for g in range(self.model.ngeom)
            if int(self.model.geom_bodyid[g]) in foot_body_ids and _is_collidable(g)
        ]
        if geom_ids:
            return geom_ids

        # Fallback: any collidable geom in the model
        return [g for g in range(self.model.ngeom) if _is_collidable(g)]

    def _compute_foot_min_z(self):
        """World-frame minimum Z over identified foot geoms.

        For BOX, evaluates the 8 corners in world frame (exact).
        For SPHERE, uses center - radius (exact).
        For CYLINDER, evaluates the axial and radial world-Z extents.
        For CAPSULE, evaluates both axial endpoints with radius.
        For MESH, evaluates transformed vertices; other shapes use geom_rbound.
        """
        if not self._foot_geom_ids:
            return 0.0

        BOX = int(mj.mjtGeom.mjGEOM_BOX)
        SPHERE = int(mj.mjtGeom.mjGEOM_SPHERE)
        CAPSULE = int(mj.mjtGeom.mjGEOM_CAPSULE)
        CYLINDER = int(mj.mjtGeom.mjGEOM_CYLINDER)
        MESH = int(mj.mjtGeom.mjGEOM_MESH)

        data = self.configuration.data
        min_z = np.inf
        for g in self._foot_geom_ids:
            gtype = int(self.model.geom_type[g])
            xpos = np.asarray(data.geom_xpos[g])
            xmat = np.asarray(data.geom_xmat[g]).reshape(3, 3)
            size = np.asarray(self.model.geom_size[g])

            if gtype == MESH:
                mid = int(self.model.geom_dataid[g])
                start = int(self.model.mesh_vertadr[mid])
                count = int(self.model.mesh_vertnum[mid])
                vertices = self.model.mesh_vert[start:start + count]
                z = float((vertices @ xmat.T + xpos)[:, 2].min())
            elif gtype == BOX:
                sx, sy, sz = size[0], size[1], size[2]
                # 8 corners in local frame
                signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
                corners_local = signs * np.array([sx, sy, sz])
                corners_world = corners_local @ xmat.T + xpos
                z = float(corners_world[:, 2].min())
            elif gtype == SPHERE:
                z = float(xpos[2] - size[0])
            elif gtype == CYLINDER:
                radius, half_len = size[0], size[1]
                axis_z = xmat[2, 2]
                z = float(xpos[2] - half_len * abs(axis_z) - radius * np.sqrt(max(0., 1 - axis_z**2)))
            elif gtype == CAPSULE:
                radius, half_len = size[0], size[1]
                axis = xmat[:, 2]
                e1 = xpos + axis * half_len
                e2 = xpos - axis * half_len
                z = float(min(e1[2], e2[2]) - radius)
            else:
                z = float(xpos[2] - self.model.geom_rbound[g])

            if z < min_z:
                min_z = z
        return float(min_z)

    def reset(self):
        """Reset transient state so loops start clean.

        Without this, repeating the same motion sequence inherits the previous
        run's IK state — joints with redundant solutions (wrist, elbow) drift
        each loop because the IK warm-starts from the prior pose.
        """
        self.configuration.update(q=self.model.qpos0)
        self.ground_offset = None
        self.lowest_pos = np.inf
        self._foot_z_correction = None
        self.solve_error_arr = []
        for task in self.foot_contact_tasks:
            task.reset()

    def load_velocity_limits(self, tgt_robot: str):
        velocity_limits = {}

        urdf_path = ROBOT_URDF_DICT.get(tgt_robot)
        if urdf_path is not None:
            try:
                root = ET.parse(urdf_path).getroot()
                for joint in root.findall("joint"):
                    joint_type = joint.get("type")
                    if joint_type not in {"revolute", "prismatic"}:
                        continue
                    joint_name = joint.get("name")
                    limit = joint.find("limit")
                    if joint_name is None or limit is None:
                        continue
                    velocity = limit.get("velocity")
                    if velocity is None:
                        continue
                    if mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name) == -1:
                        continue
                    velocity_limits[joint_name] = float(velocity) * VALOCITY_MAX
            except (ET.ParseError, OSError, ValueError) as exc:
                if self.verbose:
                    print(f"[warn] failed to load URDF velocity limits from {urdf_path}: {exc}")

        if not velocity_limits:
            # Fallback: derive velocity limits from MuJoCo model.
            # Use actuator ctrlrange (force/torque range) as a proxy:
            # larger torque capacity -> higher-speed joint.
            free_joints = {
                mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
                for j in range(self.model.njnt)
                if self.model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
            }
            for i in range(self.model.nu):
                jnt_id = int(self.model.actuator_trnid[i, 0])
                jnt_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
                if jnt_name is None or jnt_name in free_joints:
                    continue
                ctrl = self.model.actuator_ctrlrange[i]
                max_ctrl = max(abs(float(ctrl[0])), abs(float(ctrl[1])))
                if max_ctrl >= 150:
                    vel = 12.0
                elif max_ctrl >= 80:
                    vel = 10.0
                elif max_ctrl >= 40:
                    vel = 7.0
                else:
                    vel = 5.0
                velocity_limits[jnt_name] = vel * VALOCITY_MAX

        return velocity_limits
    
    def get_ik_dt(self, active_stage_count: int):
        if not self.use_velocity_limit:
            return self.configuration.model.opt.timestep
        active_stage_count = max(active_stage_count, 1)
        return self.retarget_dt / active_stage_count     
    
    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.task_to_human_body1[task] = body_name
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []

        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.task_to_human_body2[task] = body_name
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

        # 踝关节：将 orientation cost 从标量改为 per-axis，roll/pitch 与 yaw 同级。
        # 配合 update_targets 中的脚底水平目标，强制求解器优先保持脚掌贴地。
        for _stage_tasks in (self.tasks1, self.tasks2):
            for _t in _stage_tasks:
                if "ankle" in _t.frame_name.lower() or "foot" in _t.frame_name.lower():
                    _yaw = float(_t.cost[5]) if _t.cost.shape == (6,) else float(_t.cost)
                    if _yaw > 0:
                        _t.set_orientation_cost(np.array([_yaw, _yaw, _yaw]))

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data_stage1 = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data_stage2 = self.offset_human_data(human_data, self.pos_offsets2, self.rot_offsets2)

        if offset_to_ground:
            human_data_stage1 = self.offset_human_data_to_ground(human_data_stage1)
            human_data_stage2 = self.offset_human_data_to_ground(human_data_stage2)
        else:
            human_data_stage1 = self.apply_ground_offset(human_data_stage1)
            human_data_stage2 = self.apply_ground_offset(human_data_stage2)

        self.scaled_human_data_stage1 = human_data_stage1
        self.scaled_human_data_stage2 = human_data_stage2
        if self.use_ik_match_table2:
            self.scaled_human_data = human_data_stage2
        else:
            self.scaled_human_data = human_data_stage1

        if self.use_ik_match_table1:
            for task in self.tasks1:
                body_name = self.task_to_human_body1[task]
                if body_name not in human_data_stage1:
                    continue
                pos, rot = human_data_stage1[body_name]
                # 踝关节脚底水平：仅跟踪 yaw（脚朝向），roll/pitch 强制置零
                if "ankle" in task.frame_name.lower():
                    r = R.from_quat(rot[[1, 2, 3, 0]])  # scalar-first [w,x,y,z] → xyzw
                    yaw, pitch, roll = r.as_euler('zyx')
                    r_flat = R.from_euler('zyx', [yaw, 0.0, 0.0])
                    rot = r_flat.as_quat(scalar_first=True)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        if self.use_ik_match_table2:
            for task in self.tasks2:
                body_name = self.task_to_human_body2[task]
                if body_name not in human_data_stage2:
                    continue
                pos, rot = human_data_stage2[body_name]
                # 踝关节脚底水平：仅跟踪 yaw（脚朝向），roll/pitch 强制置零
                if "ankle" in task.frame_name.lower():
                    r = R.from_quat(rot[[1, 2, 3, 0]])  # scalar-first [w,x,y,z] → xyzw
                    yaw, pitch, roll = r.as_euler('zyx')
                    r_flat = R.from_euler('zyx', [yaw, 0.0, 0.0])
                    rot = r_flat.as_quat(scalar_first=True)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
    def _update_foot_contact_state(self, human_data_robot_frame: dict):
        """根据动捕脚部高度更新 stance/swing 状态和接触点。

        对每个 FootContactTask：
        1. 从动捕数据中获取对应人脚的世界 Z 坐标
        2. 更新 stance/swing 状态（带迟滞）
        3. 进入 stance 时记录动捕脚 XY 为接触点

        Args:
            human_data_robot_frame: 已经过 scale+offset 处理的动捕数据
                                   （位于机器人世界坐标系）。
        """
        if not self.foot_contact_tasks:
            return

        for task in self.foot_contact_tasks:
            human_body = self._foot_contact_human_body.get(task)
            if human_body is None or human_body not in human_data_robot_frame:
                continue

            foot_pos, _foot_rot = human_data_robot_frame[human_body]
            foot_z = float(foot_pos[2])
            foot_xy = np.asarray(foot_pos[:2], dtype=float)

            was_in_stance = task.in_stance
            task.update_stance_state(foot_z)

            # Entering stance: lock the robot foot at its current world XY.
            # Using the human target XY here makes the constraint snap the
            # robot foot sideways whenever tracking has a residual offset,
            # which creates exactly the slide this task is meant to prevent.
            if not was_in_stance and task.in_stance:
                robot_foot_xy = self.configuration.data.xpos[task.body_id][:2]
                task.set_contact_point(robot_foot_xy)

    def retarget(self, human_data, offset_to_ground=False,framesIdx=-1):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        # 更新脚部接触状态：检测 stance/swing 并设置接触点
        self._update_foot_contact_state(self.scaled_human_data)

        active_stage_count = int(bool(self.use_ik_match_table1 and self.tasks1)) + int(bool(self.use_ik_match_table2 and self.tasks2))
        ik_dt = self.get_ik_dt(active_stage_count)
        for enabled, tasks in [(self.use_ik_match_table1, self.tasks1),
                               (self.use_ik_match_table2, self.tasks2 + ([self.shoulder_posture_task] if self.shoulder_posture_task else []))]:
            if enabled:
                vel = mink.solve_ik(self.configuration, tasks + self.foot_contact_tasks,
                                    ik_dt, self.solver, self.damping, limits=self.ik_limits)
                self.configuration.integrate_inplace(vel, ik_dt)

        qpos = self.configuration.data.qpos.copy()

        # Root-Z grounding compensation: works across robots without per-bot tuning.
        # During warm-up (first 50 frames) recompute and freeze afterwards, matching
        # the existing ground_offset semantics.
        foot_min_z = self._compute_foot_min_z()
        if self._foot_z_correction is None or 0 <= framesIdx < 50:
            self._foot_z_correction = foot_min_z
        qpos[2] -= self._foot_z_correction
        return qpos

    def print_solve_error(self,printAll=False):
        errorLen = len(self.solve_error_arr)
        if errorLen > 8 or (errorLen > 0 and printAll):
            print("[GMR] solve_frame_error",", ".join(self.solve_error_arr))
            self.solve_error_arr = []

    def error1(self):
        all_errors = []
        for task in self.tasks1:
            err = task.compute_error(self.configuration)
            # 如果这个 task 是 GroundAvoidanceTask，就打印
            # if isinstance(task, GroundAvoidanceTask):
            #     print(f"[Ground Task] {task.body_name} error: {err}")
            all_errors.append(err)
        
        all_errors = np.concatenate(all_errors)
        return np.linalg.norm(all_errors)
    
    def error2(self):
        all_errors = []
        for task in self.tasks2:
            err = task.compute_error(self.configuration)
            # 如果这个 task 是 GroundAvoidanceTask，就打印
            # if isinstance(task, GroundAvoidanceTask):
            #     print(f"[Ground Task] {task.body_name} error: {err}")
            all_errors.append(err)
        
        all_errors = np.concatenate(all_errors)
        return np.linalg.norm(all_errors)


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        # scale root
        # The root scale is a morphology/height calibration, not a trajectory
        # scale. Scaling root X/Y lets the optimizer collapse global motion
        # toward the origin and then score against that same collapsed target.
        # Preserve locomotion in world X/Y and apply the calibration only to Z.
        scaled_root_pos = np.asarray(root_pos, dtype=float).copy()
        scaled_root_pos[2] *= human_scale_table[human_root_name]
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            if body_name not in rot_offsets:
                continue
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat

            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)

            offset_human_data[body_name][0] = pos + global_pos_offset

        return offset_human_data
    
    
    # 未优化，躺 走 跳逻辑不同，复杂动作不建议使用
    def offset_human_data_to_ground(self, human_data):
        offset_human_data = {}
        for body_name in human_data.keys():
            # only consider the foot/Foot
            # if "Foot" not in body_name and "foot" not in body_name:
                # print("has not found foot")
                # continue
            pos, quat = human_data[body_name]
            # print("has found foot")
            if pos[2] < self.lowest_pos:
                self.lowest_pos = pos[2]
                # print("lowest_pos has changed")
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, self.lowest_pos]) - np.array([0, 0, -0.0])
        return offset_human_data



    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        cur_lowest_pos = np.inf
        lowest_body_name = ""
        for body_name, (pos, quat) in human_data.items():
            if pos[2] < cur_lowest_pos:
                cur_lowest_pos = pos[2]
                lowest_body_name = body_name
        if self.ground_offset is None:
            self.cur_lowest_pos = cur_lowest_pos
            self.ground_offset = cur_lowest_pos - 0.0

        for body_name in human_data.keys():
            if "Foot" not in lowest_body_name and "foot" not in lowest_body_name and "Toe" not in lowest_body_name:
                pos, quat = human_data[body_name]
                human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
            else:
                pos, quat = human_data[body_name]
                human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])          

        return human_data
