# 模型、数据与动力学约定

本体模型基于 Dex URDF。工作模型将头部 yaw/pitch 固定在 URDF 零位，保留全部 link 质量、质心和惯性；身体 29 轴，不含手指控制。

## 三份模型的用途

| 位置 | 使用方 |
| --- | --- |
| `Dex-GMR/assets/dex/` | 人体动作 IK、PKL 查看 |
| `Dex-Mimic/source/whole_body_tracking/whole_body_tracking/assets/dex_evt/` | Isaac 资产导入、数据 FK |
| `Dex-Mimic/ros_dex_deploy/src/mujoco_node/resources/dex/` | 可独立拷走的 ROS2 部署包 |

三份资产需要一致。部署包保留自己的资产副本，使其不依赖训练工作站目录。`tools/check_release.py` 检查模型标识；策略契约进一步检查 URDF、MJCF 与 STL 的整体哈希。

`model.json` 记录 MJCF、原始 URDF 与工作 URDF 的 SHA256。更改模型后应重新生成对应动作、检查训练配置并重新导出策略，不要手动修改旧数据中的模型标识。

## 39 body 与 30 body

模型有 39 个机器人 body，其中含传感器、TCP 等固定连接件；策略只选其中 30 个 body 构造跟踪任务。两者都对应 29 个活动关节。NPZ 保存完整模型 body，训练按名称取子集，避免用固定索引猜顺序。MuJoCo 自带的 world body 不计入这 39 个。

| 接口 | 约定 |
| --- | --- |
| GMR PKL 根四元数 | xyzw |
| NPZ 四元数 | wxyz |
| NPZ 帧率 | 50 fps |
| Isaac 训练/播放策略周期 | 0.02 s，50 Hz |
| ROS 主循环与部署推理 | 0.01 s，100 Hz，策略 decimation=1 |
| 独立 MuJoCo 默认推理 | 0.02 s，50 Hz |
| 独立 MuJoCo 物理周期 | sim2sim_dex.sh 传入 0.0025 s；Python 直接入口默认 0.005 s |
| ROS MuJoCo 物理周期 | 0.001 s |
| 双头 ONNX | obs(1,216)、time_step(1,1) → actions(1,29) |
| 头选择 | 最后两维 `[1,0]` 为站立，`[0,1]` 为动作 |
| 脚底 | 每脚 7 个 Cylinder；Isaac 不将其替换为 Capsule |

## 为什么部署至少需要 100 Hz

工程师真机实测表明：当前机器人的电机响应、Kp/Kd 和控制链路要求 **ROS 真机部署**闭环至少以 100 Hz 更新，降低到 50 Hz 无法保持平衡。默认配置为每 10 ms 获取最新状态、重新运行 Mimic 网络并发送目标，`physical_dt=0.01`、`decimation=1`。训练和独立 MuJoCo 保持 50 Hz。100 Hz 的要求适用于当前机器人的 ROS 控制链路，不是所有机器人通用的固定频率门槛。

PD 力矩可概括为 `τ = Kp × (目标位置 − 实际位置) + Kd × (目标速度 − 实际速度) + 前馈力矩`。Kp 提供位置纠正，Kd 提供阻尼；稳定性取决于增益、电机响应、采样周期和端到端延迟的组合。降低反馈更新频率会增加目标保持时间和纠正延迟，在当前增益与电机特性下可能导致纠正滞后、振荡和失去平衡。采样与延迟影响稳定裕度的原理可参考 [MathWorks 控制采样说明](https://www.mathworks.com/help/slcontrol/ug/modeling-computational-delays-and-sampling-effects.html)。

必须区分以下频率：

- **网络推理 100 Hz**：每 10 ms 使用新观测计算平衡修正。仅以 100 Hz 重发上一次 50 Hz 推理的目标，不等价于此配置。
- **ROS 指令发送 100 Hz**：主循环每 10 ms 发送位置、速度、力矩及 Kp/Kd。电机驱动器内部 PD 的真实执行频率由底层固件决定，不能从 Python 发送频率推定。
- **参考动作 50 fps**：每帧持续 20 ms。独立 MuJoCo 每帧推理一次；ROS 的 100 Hz 推理每帧使用两次，两次均刷新观测并重新推理。动作时长、速度、相位和前瞻帧含义保持原定义。
- **独立 MuJoCo 50 Hz**：`scripts/sim2sim_dex.py` 的 `control_dt=0.02`，用于离线仿真评估；ROS 真机控制链路单独使用 100 Hz。
- **Isaac 训练/播放 50 Hz**：现有 checkpoint 的训练时间尺度为 20 ms，保存在权重和 ONNX 中。训练周期和 ROS 部署周期分别记录，不能通过修改元数据将二者混淆。

ONNX 的 `control_dt=0.02` 和 PT 内的控制周期记录训练配置。ROS 部署 `bundle.json` 单独记录 `training_control_dt=0.02` 与 `policy_control_dt=0.01`；后者只约束 ROS，不覆盖独立 MuJoCo 的周期。启动校验分别检查训练来源和 ROS 部署周期，拒绝把 ROS Mimic 配置降为 50 Hz。独立 MuJoCo 默认保持 50 Hz，只有显式传入 `--control-dt 0.01` 时才进行额外的 100 Hz 诊断。

## 电机候选基线

默认 `tg3-armature` 的公开参数来源是 [xSIM_MUJOCO 天工3模型](https://github.com/Open-X-Humanoid/xSIM_MUJOCO/blob/942469c660ec93e6d7ae9e78026db81cfa623591/resources/tiangong3/urdf/tiangong3.xml)。这是公开仿真参数候选，不是本机电机辨识报告。

| 关节组 | armature（kg·m²） |
| --- | ---: |
| 髋 pitch / roll | 0.24 |
| 髋 yaw | 0.18 |
| 膝 | 0.37 |
| 踝 | 0.032 |
| 腰 | 0.17 |
| 肩 / 肘 | 0.10 |
| 腕 | 0.0236 |

发布主流程保留当前 Kp/Kd 和硬件限位，关节被动 damping 与 frictionloss 为零；MuJoCo 使用 `implicitfast` + `CG`。上游完整 MJCF 另含 damping=1、frictionloss=0.1 Nm，诊断脚本可用 `tg3-full` 做 MuJoCo 消融；不直接把它当作 PhysX 的同值摩擦系数。

训练侧配置在 `robots/tg3_motor_profile.json` 和 `robots/motor_parameters.py`。ONNX 导出器记录 nominal armature、Kp/Kd、力矩与位置边界，MuJoCo 从选中的 ONNX 读取；ROS 部署目录可独立运行。

Isaac 使用隐式 PD，ROS MuJoCo 使用外部 PD→力矩；参数相符仍不代表两个求解器完全等价，需分别运行验证。

## 显示设置

Isaac 默认跟随第 0 个环境的 `robot` 根节点，相对 eye=`(3,3,0.7)`、lookat=`(0,0,-0.1)`，在任务 `tracking_env_cfg.py` 中维护。

MuJoCo 的 `scene_visuals.py` 在内存编译时应用无限蓝灰网格和天空渐变，原 MJCF 不变。plane 的显示尺寸不改变其无限碰撞平面；地面摩擦和脚底接触参数保持原值。
