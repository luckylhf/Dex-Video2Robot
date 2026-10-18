import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

DEX_EVT_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ASSET_DIR}/dex_evt/urdf/dex_29.urdf",
        fix_base=False,
        force_usd_conversion=True,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 早期训练关闭，仿 Pro 策略，避免策略随机探索阶段误杀
            solver_position_iteration_count=4,  # 8→4, 降低物理求解迭代防止极端位姿卡死
            solver_velocity_iteration_count=2,  # 4→2
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.97),
        joint_pos={
            # Legs - left: 对齐 NPZ 首帧 (~0 rad)，微小屈膝防锁死
            "hip_yaw_l_joint": 0.0,
            "hip_roll_l_joint": 0.0,
            "hip_pitch_l_joint": -0.05,
            "knee_pitch_l_joint": 0.05,
            "ankle_pitch_l_joint": -0.05,
            "ankle_roll_l_joint": 0.0,
            # Legs - right
            "hip_yaw_r_joint": 0.0,
            "hip_roll_r_joint": 0.0,
            "hip_pitch_r_joint": -0.05,
            "knee_pitch_r_joint": 0.05,
            "ankle_pitch_r_joint": -0.05,
            "ankle_roll_r_joint": 0.0,
            # Arms - left：接近伸直，留安全裕度防过伸
            "shoulder_pitch_l_joint": 0.0,
            "shoulder_roll_l_joint": 0.1,
            "shoulder_yaw_l_joint": 0.0,
            "elbow_pitch_l_joint": -0.1,
            "elbow_yaw_l_joint": 0.0,
            "wrist_pitch_l_joint": 0.0,
            "wrist_roll_l_joint": 0.0,
            # Arms - right
            "shoulder_pitch_r_joint": 0.0,
            "shoulder_roll_r_joint": -0.1,
            "shoulder_yaw_r_joint": 0.0,
            "elbow_pitch_r_joint": -0.1,
            "elbow_yaw_r_joint": 0.0,
            "wrist_pitch_r_joint": 0.0,
            "wrist_roll_r_joint": 0.0,
            # Waist：中立
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_yaw_.*_joint",
                "hip_roll_.*_joint",
                "hip_pitch_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_yaw_.*_joint": 142.0,
                "hip_roll_.*_joint": 200.0,
                "hip_pitch_.*_joint": 200.0,
                "knee_pitch_.*_joint": 330.0,
            },
            velocity_limit_sim={
                "hip_yaw_.*_joint": 13.823,
                "hip_roll_.*_joint": 12.566,
                "hip_pitch_.*_joint": 12.566,
                "knee_pitch_.*_joint": 11.106,
            },
            stiffness={
                "hip_yaw_.*_joint": 350.0,    # 200→350, 接近 Pro 500
                "hip_roll_.*_joint": 600.0,   # 450→600, 接近 Pro 700
                "hip_pitch_.*_joint": 600.0,  # 450→600
                "knee_pitch_.*_joint": 650.0, # 500→650, 接近 Pro 700
            },
            damping={
                "hip_yaw_.*_joint": 10.0,     # 6→10, 维持阻尼比
                "hip_roll_.*_joint": 15.0,    # 12→15
                "hip_pitch_.*_joint": 15.0,   # 12→15
                "knee_pitch_.*_joint": 14.0,  # 12→14
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["ankle_pitch_.*_joint", "ankle_roll_.*_joint"],
            effort_limit_sim={
                "ankle_pitch_.*_joint": 100.0,
                "ankle_roll_.*_joint": 50.0,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 14.6,
                "ankle_roll_.*_joint": 14.6,
            },
            stiffness={
                "ankle_pitch_.*_joint": 60.0,   # 30→60, 翻倍提升 PD 推力
                "ankle_roll_.*_joint": 35.0,    # 16.8→35, 翻倍
            },
            damping={
                "ankle_pitch_.*_joint": 7.0,    # 5.0→7.0, 维持阻尼比
                "ankle_roll_.*_joint": 3.0,     # 2.0→3.0
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_pitch_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 90.0,
                "shoulder_roll_.*_joint": 60.0,
                "shoulder_yaw_.*_joint": 36.0,
                "elbow_pitch_.*_joint": 60.0,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 15.184,
                "shoulder_roll_.*_joint": 14.137,
                "shoulder_yaw_.*_joint": 9.739,
                "elbow_pitch_.*_joint": 14.137,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 150.0,
                "shoulder_roll_.*_joint": 50.0,
                "shoulder_yaw_.*_joint": 50.0,
                "elbow_pitch_.*_joint": 150.0,
            },
            damping={
                "shoulder_pitch_.*_joint": 5.0,
                "shoulder_roll_.*_joint": 2.5,
                "shoulder_yaw_.*_joint": 2.5,
                "elbow_pitch_.*_joint": 5.0,
            },
        ),
        "arms_distal": ImplicitActuatorCfg(
            joint_names_expr=[
                "elbow_yaw_.*_joint",
                "wrist_pitch_.*_joint",
                "wrist_roll_.*_joint",
            ],
            effort_limit_sim={
                "elbow_yaw_.*_joint": 36.0,
                "wrist_pitch_.*_joint": 36.0,
                "wrist_roll_.*_joint": 36.0,
            },
            velocity_limit_sim={
                "elbow_yaw_.*_joint": 9.739,
                "wrist_pitch_.*_joint": 9.739,
                "wrist_roll_.*_joint": 9.739,
            },
            stiffness={
                "elbow_yaw_.*_joint": 500.0,
                "wrist_pitch_.*_joint": 200.0,
                "wrist_roll_.*_joint": 200.0,
            },
            damping={
                "elbow_yaw_.*_joint": 15.0,   # 5→15: action_scale=0, PD直接跟踪, ζ: 0.112→0.335
                "wrist_pitch_.*_joint": 8.0,  # 2→8:  同上, ζ: 0.071→0.283
                "wrist_roll_.*_joint": 8.0,   # 2→8:  同上
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            effort_limit_sim={
                "waist_yaw_joint": 91.0,
                "waist_roll_joint": 91.0,
                "waist_pitch_joint": 91.0,
            },
            velocity_limit_sim={
                "waist_yaw_joint": 9.2153,
                "waist_roll_joint": 9.2153,
                "waist_pitch_joint": 9.2153,
            },
            stiffness={
                "waist_yaw_joint": 400.0,
                "waist_roll_joint": 400.0,
                "waist_pitch_joint": 400.0,
            },
            damping={
                "waist_yaw_joint": 10.0,   # 5→10: 腰偏航振荡通过腿长放大为脚坐标偏差, ζ: 0.125→0.250
                "waist_roll_joint": 10.0,
                "waist_pitch_joint": 10.0,
            },
        ),
    },
)

# Latest source URDF limits, selected by actual joint name rather than old group constants.
import json
import re
from pathlib import Path
_limits = json.loads((Path(ASSET_DIR) / "dex_evt/model.json").read_text())["joint_limits"]
for actuator in DEX_EVT_CFG.actuators.values():
    selected = {name: limit for name, limit in _limits.items()
                if any(re.fullmatch(pattern, name) for pattern in actuator.joint_names_expr)}
    actuator.effort_limit_sim = {name: float(limit["effort"]) for name, limit in selected.items()}
    actuator.velocity_limit_sim = {name: float(limit["velocity"]) for name, limit in selected.items()}

# Release default: the validated armature candidate. Geometry identity stays unchanged.
import os
if os.environ.get("DEX_MOTOR_PROFILE", "tg3-armature"):
    from .motor_parameters import apply_isaac_profile, load_profile
    apply_isaac_profile(DEX_EVT_CFG.actuators, load_profile(os.environ.get("DEX_MOTOR_PROFILE", "tg3-armature")))

DEX_EVT_ACTION_SCALE = {}
for actuator in DEX_EVT_CFG.actuators.values():
    for pattern, stiffness in actuator.stiffness.items():
        efforts = [value for name, value in actuator.effort_limit_sim.items() if re.fullmatch(pattern, name)]
        DEX_EVT_ACTION_SCALE[pattern] = 0.25 * min(efforts) / stiffness
