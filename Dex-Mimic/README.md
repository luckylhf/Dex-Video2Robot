# Dex-Mimic：训练、导出与仿真

[返回总览](../README.md) · [视频与 IK](../Dex-GMR/README.md) · [ROS2 部署](ros_dex_deploy/README.md)

本包接收当前 Dex 的参考动作 NPZ，使用 Isaac Lab / RSL-RL 训练站立与动作头，导出双头 ONNX，并打包给独立 MuJoCo 或 ROS2 控制链路。

## 1. 环境

### 离线数据与 MuJoCo

```bash
conda create -n dex_offline python=3.11 -y
conda activate dex_offline
cd "$DEX_ROOT/Dex-Mimic"
python -m pip install -r requirements-offline.txt
```

该环境能转换/检查数据、打包 ONNX 和运行 MuJoCo，不包含 Isaac 训练能力。

### Isaac 训练、播放与导出

使用以下配套版本：Python 3.11、Isaac Lab 2.2.0、Isaac Sim 5.1 独立 SDK、Torch 2.7.0、RSL-RL 2.3.3。先按 [Isaac Lab 2.2 安装说明](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/index.html) 配好对应的驱动、SDK 与环境；本项目放在 Isaac Lab 目录外。

在 Isaac 的 Python 3.11 环境安装本包：

```bash
cd "$DEX_ROOT/Dex-Mimic"
python -m pip install -e source/whole_body_tracking
```

所有训练/播放入口通过 `scripts/isaac_python.sh` 选择解释器：

- pip 安装的 Isaac 环境：激活环境后直接运行，无需 SDK 变量。
- 独立 Isaac Sim SDK：设置下面两个变量；脚本自动加载 SDK 路径并优先使用该解释器的标准库。
- 使用 Isaac Lab 自带启动器：可设置 `ISAACLAB_ROOT=/path/to/IsaacLab`，通过其 `isaaclab.sh -p` 执行；不要同时设置两种根路径。

```bash
export ISAAC_SIM_ROOT=/path/to/isaac-sim
export PYTHONEXE=/path/to/isaac/python3.11
```

SDK 路径应包含 `python.sh`。使用其他 Isaac Lab 版本时，需要适配相应的配置和 RSL 接口。

## 2. 准备并检查参考动作

视频入口已生成 NPZ 时直接检查；已有 PKL 目录时用 `prepare.sh` 转换：

```bash
cd "$DEX_ROOT/Dex-Mimic"
# 可选：从 PKL 批量生成 raw 和带过渡的动作
bash prepare.sh "$DEX_ROOT/Dex-GMR/data/outputs"
python scripts/check_motion.py dataset/npz_dex/dance_with_transition.npz
bash replay.sh --npz-dir dataset/npz_dex
```

带过渡段的可调参数都在 `scripts/pkl_to_npz.py` / `scripts/csv_to_npz_mujoco.py`（默认值）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--hold_duration` | `1.0` s | 开头站立保持（先站着） |
| `--trans_duration` | `1.0` s | 开头站立 → 动作起始的平滑过渡（五次曲线） |
| `--tail_trans_duration` | `2.5` s | 动作末尾 → 站立的平滑过渡 |
| `--tail_hold_duration` | `1.0` s | 结尾站立保持（收尾再站着） |
| `--stand_height` | `0.87` m | 站立姿态的根高（pelvis 高度） |
| `--head_skip_frames` | `-1` | 开头裁帧数；`-1` 表示按速度阈值自动裁 |
| `--tail_skip_frames` | `-1` | 结尾裁帧数；`-1` 表示按速度阈值自动裁 |
| `--skip_threshold` | `1.5` rad/s | 自动裁帧的关节速度阈值（低于它视为"还没开始动"） |

> 默认的 1+1+2.5+1.0 s 会让带过渡 NPZ 比 raw NPZ 多 5.5 s（共 275 帧 @50 fps）。调短会让训练数据更紧凑但可能影响起止稳定性。

示例：从单个 PKL 重转，缩短过渡段：

```bash
cd "$DEX_ROOT/Dex-Mimic"
python scripts/pkl_to_npz.py \
  --input_file "$DEX_ROOT/Dex-GMR/data/outputs/dance.pkl" \
  --robot dex_evt \
  --output_name dataset/npz_dex_raw/dance \
  --output_dir dataset/npz_dex \
  --add_transition --recompute_velocities \
  --hold_duration 0.5 --trans_duration 1.0 \
  --tail_trans_duration 2.0 --tail_hold_duration 0.5
```

`replay.sh` 只回放参考轨迹，不运行策略。训练前检查动作方向、关节范围、脚底、轨迹漂移和起止过渡。新动作数据通过 GMR 生成。

如果 GMR 阶段报 `NoSolutionFound`，按 [IK 无解处理流程](../Dex-GMR/README.md#ik-no-solution) 排查：优先显式使用 `--no-collision-avoidance` 保留速度约束，复用人体预测重新生成动作，再执行 `check_motion.py`。只有检查成功退出后才训练；生成了 NPZ 文件不代表已通过检查。该检查覆盖关节位置、速度与 FK 等数据约定，不检查自碰撞，仍需回放参考并验证物理策略。

随包的 `dataset/standing/standing.npz` 是当前本体的静态站立参考，需要重建时执行 `python scripts/make_standing_motion.py`。

## 3. 训练

切换到 Isaac 环境。默认采用 `tg3-armature` 电机配置。

各环节分别沿用工程师原配置：**Isaac 训练/播放 50 Hz、独立 MuJoCo 50 Hz、ROS 部署推理 100 Hz**。参考 NPZ 为 50 fps；独立 MuJoCo 每帧推理一次，ROS 每帧推理两次，均按正常时长播放。真机至少 100 Hz 的实测要求适用于 ROS 部署，不扩展到训练和独立 MuJoCo。原因与周期记录方式见 [动力学说明](../docs/model_and_dynamics.md#为什么部署至少需要-100-hz)。

动作初始化示例：

```bash
cd "$DEX_ROOT/Dex-Mimic"
RESUME=false LOAD_CKPT=policy/dex/wbt_model_46400.pt \
  NUM_ENVS=1024 MAX_ITER=3000 \
  bash train_dex.sh medium dataset/npz_dex/dance_with_transition.npz 1
```

先验证环境时可用 `NUM_ENVS=64 MAX_ITER=2` 执行相同入口。两轮只证明能完成 PPO 更新，不证明动作质量；约 3000 次迭代初见成效是团队经验，目标动作还需评估。

| 模式 | 默认总环境数 | 默认迭代数 |
| --- | ---: | ---: |
| smoke | 256 | 500 |
| medium | 1024 | 5000 |
| full | 12000 | 50000 |

`NUM_ENVS`、`MAX_ITER` 覆盖模式预设；按显存选择环境数。最后一个参数是 GPU 数，多卡时总环境数须能被 GPU 数整除。日志和 checkpoint 位于 `logs/rsl_rl/dex_evt_fix/`，终端会打印实际目录；`LOG_ROOT` 可指定其他输出根目录。

站立头可单独微调：

```bash
RESUME=false LOAD_CKPT=policy/dex/standing_model_2000.pt \
  RUN_PREFIX=dex_standing NUM_ENVS=1024 MAX_ITER=3000 \
  bash train_dex.sh medium dataset/standing/standing.npz 1
```

也可以先使用随包的 `policy/dex/standing_baseline.pt` 进行站立播放。两份预制初始化权重、当前基线及校验 JSON 的区别见 [权重说明](policy/dex/README.md)。

恢复本项目中断的训练：

```bash
RESUME=true bash train_dex.sh medium dataset/npz_dex/dance_with_transition.npz \
  1 ACTUAL_RUN_NAME model_1000.pt
```

`LOAD_CKPT` 与 `RESUME=true` 互斥。观察训练曲线可运行 `bash log.sh`。

## 4. 播放与导出

将下面的 `ACTUAL_RUN_NAME/model_N.pt` 替换为训练生成的 checkpoint，并选择匹配的 NPZ：

```bash
export MOTION_FILE=dataset/npz_dex/dance_with_transition.npz
export CHECKPOINT_PATH=logs/rsl_rl/dex_evt_fix/ACTUAL_RUN_NAME/model_N.pt
export STANDING_CHECKPOINT=policy/dex/standing_baseline.pt
bash play_dex.sh
```

相机默认跟随机器人全身。该入口同时导出策略；仅需无界面导出时，保留上述变量并执行 `bash export.sh`。

导出产物在动作 checkpoint 同目录的 `exported/` 下：`policy.onnx` 为单头诊断，`policy_merged.onnx` 为 ROS 使用的站立/动作双头。各头保留自己的归一化。

ONNX 同时携带模型、URDF/网格整体哈希、关节顺序、逐轴 armature、PD、动作缩放、限位和控制周期。旧版缺少发布契约的 ONNX 会被拒绝，需要用当前导出器重新导出。

新 checkpoint 会记录模型和执行器配置，恢复/播放时要求匹配。没有来源记录的旧 PT 只允许随包、已核验的 `standing_baseline.pt` 直接播放；其他旧权重请通过 `LOAD_CKPT` 初始化并训练后再导出，不能只换模型标识冒充适配。

## 5. 打包并选择新动作

切换到离线环境，显式选择新目录，避免覆盖随包站立演示：

```bash
cd "$DEX_ROOT/Dex-Mimic"
python scripts/prepare_deploy.py \
  /path/to/tracking_run/exported/policy_merged.onnx \
  dataset/npz_dex/dance_with_transition.npz \
  --output "$DEX_ROOT/Dex-Mimic/ros_dex_deploy/bundles/dance"
export MIMIC_BUNDLE_DIR="$DEX_ROOT/Dex-Mimic/ros_dex_deploy/bundles/dance"
python ros_dex_deploy/check_bundle.py
```

一个包包括 `policy_merged.onnx`、`bundle.json`、`npz/`。多动作策略可将第二个参数改为只包含本策略所需动作的目录。已有包不自动覆盖，应使用新目录。

独立 MuJoCo：

```bash
POLICY="$MIMIC_BUNDLE_DIR/policy_merged.onnx" \
  NPZ_DIR="$MIMIC_BUNDLE_DIR/npz" bash sim2sim_dex.sh
# B 开始或切换动作；默认先站立
```

完整 ROS2 链路的环境、启动、长按触发及真机步骤见 [部署说明](ros_dex_deploy/README.md)。`MIMIC_BUNDLE_DIR` 同时用于启动校验、两个 mimic FSM 和 MuJoCo 动力学，避免检查一个包却运行另一个包。

## 6. 直接查看随包站立基线

不需要训练新动作：

```bash
# Isaac 环境
bash scripts/run_standing_isaac.sh --seconds 60 --sim-dt 0.005 \
  --realtime --render-every 1 --output /tmp/dex_isaac_standing.json

# 离线 MuJoCo 环境；--gui 会按实际时间显示
python scripts/validate_standing_mujoco.py --gui --seconds 60 \
  --sim-dt 0.005 --output /tmp/dex_mujoco_standing.json
```

这两个入口固定选择站立头并且不自动复位。诊断数据写入指定输出，不混进训练数据。评估站立时同时观察倾角、跌倒和水平漂移。
