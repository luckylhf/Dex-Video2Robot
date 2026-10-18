# Dex：视频动作到人形机器人控制

面向天工 Dex 的工程工具链：**MP4 → 人体动作 → 机器人参考动作 → 模仿学习 → Isaac / MuJoCo 仿真 → ROS2 部署**。

本项目由 Open-X-Humanoid 的 [xGMR](https://github.com/Open-X-Humanoid/xGMR)、[xMimic](https://github.com/Open-X-Humanoid/xMimic) 和 [xSIM_MUJOCO](https://github.com/Open-X-Humanoid/xSIM_MUJOCO) 派生整理，使用修正后的本体 URDF，统一模型、数据接口与各阶段运行入口。

## 两个包的分工

| 包 | 负责什么 | 输入 → 输出 |
| --- | --- | --- |
| [Dex-GMR](Dex-GMR/README.md) | 视频处理、人体运动估计、逆运动学重定向 | MP4 / 人体动作 → Dex PKL → 训练 NPZ |
| [Dex-Mimic](Dex-Mimic/README.md) | 参考动作检查、策略训练、播放、导出、打包与部署 | NPZ + 可选预制权重 → PT → ONNX → ROS2 控制 |

```text
MP4 ── GVHMR ── 人体运动参数 ── Dex IK ── PKL
                                          │
                                    重采样 / FK / 过渡
                                          │
                                         NPZ
                                          │
                             Isaac Lab：站立头 / 动作头训练
                                          │
                                 Isaac play + ONNX 导出
                                          │
                            MuJoCo ── ROS2 MuJoCo ── 真机
```

PKL/NPZ 回放用于检查参考动作；运行策略则由神经网络和物理仿真共同产生机器人运动。两者应分别检查。

## 本项目整理了什么

- 统一当前 Dex 的 URDF、MJCF、关节名、限位和脚底碰撞，保留当前本体质量与惯性。
- 身体为 **29 个活动关节**，头部两轴固定，保留头部质量；左右肘 yaw、腕 pitch/roll 六轴保持活动，不控制手指。
- 区分 **39 个机器人模型 body** 和 **30 个策略 body 子集**，统一动作数据与观测映射。
- 默认采用 `tg3-armature` 电机配置；导出的 ONNX 携带逐轴 armature，MuJoCo 自动读取。
- 校验模型、URDF、网格、策略和动作文件的版本一致性；自定义部署包由一个 `MIMIC_BUNDLE_DIR` 同时控制检查器、FSM 与仿真节点。
- 提供中文视频名处理、分阶段续跑、预制权重初始化、自动跟随镜头、无限网格地面和 ROS2 Teleop GUI。

电机参数来源与数值含义见 [模型与动力学说明](docs/model_and_dynamics.md)。

**各环节分别保持工程师原配置：Isaac 训练/播放 50 Hz、独立 MuJoCo 50 Hz、ROS 部署主循环与推理 100 Hz。** 真机的 100 Hz 要求适用于 ROS 部署。参考动作为 50 fps，独立 MuJoCo 每帧推理一次，ROS 每帧推理两次，播放速度均不变。具体区别见 [部署频率说明](docs/model_and_dynamics.md#为什么部署至少需要-100-hz)。

## 随包权重

`Dex-Mimic/policy/dex/` 中保留：

| 文件 | 用途 |
| --- | --- |
| `standing_model_2000.pt` | 工程师提供的预制站立权重，用于微调初始化 |
| `wbt_model_46400.pt` | 工程师提供的预制动作跟踪权重，用于加快新动作训练 |
| `standing_baseline.pt` | 适配当前本体的站立基线，用于站立播放和默认演示 |

**团队经验：加载预制动作权重后，约 3000 次迭代可开始看到效果。** 这是训练起点的经验值，不代表任意动作在 3000 次迭代后都已收敛。目标舞蹈仍需检查跟踪误差、跌倒和起止切换。

`LOAD_CKPT` 显式加载权重和兼容的归一化状态，重新开始优化；`RESUME=true` 恢复已有训练。两者不能同时使用。训练脚本不会暗中加载预制权重。

各权重的用途、站立基线 JSON 的作用及新训练权重的保存方式见 [权重目录说明](Dex-Mimic/policy/dex/README.md)。

## 从哪里开始

在项目根目录设置路径，后续示例都使用它：

```bash
export DEX_ROOT="$(pwd)"
```

1. 按 [Dex-GMR 安装与视频流程](Dex-GMR/README.md) 准备 GMR、GVHMR 环境及人体资源，生成动作。
2. 按 [Dex-Mimic 训练与导出](Dex-Mimic/README.md) 加载预制权重训练，选择实际 checkpoint 播放、导出。
3. 按 [ROS2 仿真与部署](Dex-Mimic/ros_dex_deploy/README.md) 检查默认站立演示，或加载自己打包的新动作。

IK 出现 `NoSolutionFound` 时，见 [IK 无解处理流程](Dex-GMR/README.md#ik-no-solution)。可先显式关闭 IK 避碰、保留速度约束，复用已生成的人体预测重做动作；必须通过 NPZ 后置检查后才继续训练。默认配置不会自动关闭约束。

各阶段环境独立：

| 阶段 | 环境 |
| --- | --- |
| GVHMR 人体预测 | 上游 GVHMR 的独立 GPU 环境，通过 `GVHMR_PYTHON` 调用 |
| IK、PKL/NPZ、离线 MuJoCo | GMR / 离线 Python 环境 |
| 训练、Isaac 播放及导出 | Isaac Lab 2.2.0、Isaac Sim 5.1 SDK、Python 3.11、Torch 2.7.0、RSL-RL 2.3.3 |
| ROS2 MuJoCo / 真机 | Ubuntu 22.04 + Humble + Python 3.10，或 Ubuntu 24.04 + Jazzy + Python 3.12，使用部署目录的 venv |

已配置 ROS2 环境时，默认站立演示的入口为：

```bash
cd "$DEX_ROOT/Dex-Mimic/ros_dex_deploy"
bash run_sim.sh
```

默认包只包含站立参考，两个头均来自站立基线，**按 B 不会播放舞蹈**。新动作必须先训练、导出、打包，再显式选择自己的包。

GUI 顺序：F=`down` → 勾 D 后取消 → 用 DOWN 降至脚底贴地 → 保持 F=`down`，**勾 A 约 1 s 再取消（长按触发）**。日志出现进入 `MIMIC` 才表示站立策略正在运行；`MIMICDEFAULT` 是准备阶段的 PD 姿态保持。C 停止。

## 目录与发行边界

```text
README.md                       # 总览
docs/                           # 模型、动力学、来源说明与免责声明
tools/                          # 只读发行检查及打包
Dex-GMR/                        # 视频与动作重定向
Dex-Mimic/                      # 训练、仿真与部署
```

发行包包含两个功能包、机器人资产、预制权重、默认站立演示及回归测试源码。用户视频和中间数据默认不包含在内，可通过 `--include-data` 选择携带。

```bash
cd "$DEX_ROOT"
python tools/check_release.py
python tools/release.py --output /tmp/dex-pipeline-release.tar.gz
# 需要同时携带本地 GMR 数据时追加 --include-data
```

打包不包含 venv、Git 元数据、ROS 构建目录或训练日志，同时生成文件 SHA256 清单。第三方人体模型参数不随源码发行，请按 GMR 安装说明准备。

## 使用范围

默认演示为站立策略，持续运行存在水平漂移。新动作需使用自己的参考数据训练，并根据跟踪误差、跌倒情况及起止切换效果选择 checkpoint。参考动作回放成功不代表物理策略已经收敛。

## 免责声明

使用本仓库前请完整阅读 [免责声明](docs/DISCLAIMER.md)（同一文档的 PDF 版本：[机器人运控框架免责声明.pdf](docs/机器人运控框架免责声明.pdf)）。人形机器人的训练、仿真与部署属于高风险作业，须由受训人员在符合 ISO 10218、ISO/TS 15066 的环境下操作，并配备急停与物理防护。软件按「现状」提供，不附带任何明示或暗示的保证。

## 来源与许可

感谢 [xGMR](https://github.com/Open-X-Humanoid/xGMR)、[xMimic](https://github.com/Open-X-Humanoid/xMimic)、[xSIM_MUJOCO](https://github.com/Open-X-Humanoid/xSIM_MUJOCO)、[GVHMR](https://github.com/zju3dv/GVHMR)、Isaac Lab、MuJoCo 与 Mink。组件沿用各自的作者声明与许可，见 [Dex-GMR/LICENSE](Dex-GMR/LICENSE)、[Dex-Mimic/LICENCE](Dex-Mimic/LICENCE) 及各组件目录。人体模型、机器人模型和第三方权重的来源说明随对应资源保留。
