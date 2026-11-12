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

真机实测表明：当前机器人的电机响应、Kp/Kd 和控制链路要求 **ROS 真机部署**闭环至少以 100 Hz 更新，降低到 50 Hz 无法保持平衡。默认配置为每 10 ms 获取最新状态、重新运行 Mimic 网络并发送目标，`physical_dt=0.01`、`decimation=1`。训练和独立 MuJoCo 保持 50 Hz。100 Hz 的要求适用于当前机器人的 ROS 控制链路，不是所有机器人通用的固定频率门槛。

按以下周期配置各阶段：

- **ROS 推理与发送 100 Hz**：每 10 ms 获取新观测、重新推理并发送目标。电机驱动器内部 PD 周期由固件决定。
- **参考动作 50 fps**：每帧 20 ms，ROS 每帧使用两次，动作时长和前瞻帧定义保持不变。
- **Isaac 与独立 MuJoCo 50 Hz**：策略周期为 0.02 s；独立 MuJoCo 可用 `--control-dt 0.01` 做额外诊断。

ONNX 的 `control_dt=0.02` 记录训练周期；ROS `bundle.json` 用 `training_control_dt=0.02`、`policy_control_dt=0.01` 分别记录训练与部署周期。保持这些字段与实际运行配置一致。

## 电机参数

默认 `tg3-armature` 的公开参数来源是 [xSIM_MUJOCO 天工3模型](https://github.com/Open-X-Humanoid/xSIM_MUJOCO/blob/942469c660ec93e6d7ae9e78026db81cfa623591/resources/tiangong3/urdf/tiangong3.xml)。下表为当前默认仿真值；更换电机或本体后应重新检查。

| 关节组 | armature（kg·m²） |
| --- | ---: |
| 髋 pitch / roll | 0.24 |
| 髋 yaw | 0.18 |
| 膝 | 0.37 |
| 踝 | 0.032 |
| 腰 | 0.17 |
| 肩 / 肘 | 0.10 |
| 腕 | 0.0236 |

默认使用当前 Kp/Kd 和硬件限位，关节被动 damping 与 frictionloss 为零；MuJoCo 使用 `implicitfast` + `CG`。上游完整 MJCF 另含 damping=1、frictionloss=0.1 Nm，诊断脚本可用 `tg3-full` 做 MuJoCo 消融；不直接把它当作 PhysX 的同值摩擦系数。

训练侧配置在 `robots/tg3_motor_profile.json` 和 `robots/motor_parameters.py`。ONNX 导出器记录 nominal armature、Kp/Kd、力矩与位置边界，MuJoCo 从选中的 ONNX 读取；ROS 部署目录可独立运行。

Isaac 使用隐式 PD，ROS MuJoCo 使用外部 PD→力矩；参数相符仍不代表两个求解器完全等价，需分别运行验证。

## 显示设置

Isaac 默认跟随第 0 个环境的 `robot` 根节点，相对 eye=`(3,3,0.7)`、lookat=`(0,0,-0.1)`，在任务 `tracking_env_cfg.py` 中维护。

MuJoCo 的 `scene_visuals.py` 在内存编译时应用无限蓝灰网格和天空渐变，原 MJCF 不变。plane 的显示尺寸不改变其无限碰撞平面；地面摩擦和脚底接触参数保持原值。
