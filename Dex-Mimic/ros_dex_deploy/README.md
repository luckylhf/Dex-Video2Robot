# Dex ROS2 仿真与部署

[项目总览](../../README.md) · [训练、导出与打包](../README.md)

本目录可独立拷到工控机，包含硬件 ID 映射、机器人接口、控制主循环、真机 launch，以及模型/策略校验、部署包选择和 MuJoCo 仿真。模型使用 Dex 的 29 轴工作版本。

## 1. 安装

环境入口：Ubuntu 22.04 / ROS2 Humble / Python 3.10，或 Ubuntu 24.04 / ROS2 Jazzy / Python 3.12。系统、ROS 与 Python 版本需要配套。

先安装 ROS2 desktop、对应 Python 的 venv、Tk 与系统编译工具。以 22.04 为例：

```bash
cd "$DEX_ROOT/Dex-Mimic/ros_dex_deploy"
sudo apt install python3.10-venv python3-tk build-essential
sudo apt install ./src/rl_control/lib/22/ros-humble-bodyctrl-msgs_*.deb
source /opt/ros/humble/setup.bash
ros2 interface list | grep bodyctrl
bash init_env.sh
```

24.04 改用 Python 3.12 / Jazzy，并安装 `lib/24/` 的配套 deb。`src/rl_control/lib/` 中的 deb 与 `sptlib_python` wheel 是部署依赖，需要一同携带。

初始化脚本创建带 `--system-site-packages` 的 `.venv`，安装对应 requirements 和 wheel，然后检查导入；已有同版本 venv 会复用。系统缺少 ensurepip 时也支持已安装的 `virtualenv`。下载源可通过标准 `PIP_INDEX_URL` 设置；离线安装可设置 `PIP_NO_INDEX=1` 和 `PIP_FIND_LINKS=/path/to/wheels`，事先准备同系统、同 Python 版本的完整 wheel。

手工验证时先激活 venv，再加载 ROS：

```bash
source .venv/bin/activate
source /opt/ros/humble/setup.bash
python -c "import rclpy, bodyctrl_msgs.msg, mujoco, onnxruntime, sptlib_python; print('ROS imports OK')"
python check_bundle.py
```

## 2. 默认站立演示

```bash
bash run_sim.sh
```

脚本检查当前策略包，执行 `colcon build --symlink-install`，打开 MuJoCo、Teleop GUI 和控制节点。默认 ROS_DOMAIN_ID=42，可显式覆盖。窗口需要 DISPLAY。

默认 `src/rl_control/policy/mimic/out/` 是当前站立基线的演示包，仅含 `standing.npz`，两个头均来自站立基线。它用于验证链路；B 不会播放舞蹈。

MuJoCo 地面是无限网格。仿真从准备状态开始：未启动物理时只更新运动学，窗口不动不代表策略已经站住。

| 步骤 | Teleop GUI 操作 | 日志 / 行为 |
| --- | --- | --- |
| 1 | F 选 `down`，勾 D 后取消 | 启动物理并进入 `MIMICDEFAULT`，弹力带仍连接 |
| 2 | 使用 DOWN 逐步下降至脚底贴地 | 不从高处直接释放 |
| 3 | 保持 F=`down`，勾 A 约 1 s 再取消（长按触发） | 释放弹力带，状态切换为 `MIMIC` |
| 4 | B 勾选后取消 | 新动作包中开始/切换动作；站立演示包仍为站立参考 |
| 5 | C | 停止 |

复位：勾 C 后取消 → F 回 `mid` → 勾 D 后取消 → 从第 1 步重来。`MIMICDEFAULT` 只做 PD 准备姿态，`MIMIC` 才运行站立/跟踪网络。短点 A 可能只释放弹力带而未达到长按阈值。

## 3. 切换到自己的动作包

先按 [训练说明](../README.md) 完成训练、导出和 `prepare_deploy.py`。在部署机器上选择该目录的绝对路径：

```bash
export MIMIC_BUNDLE_DIR=/absolute/path/to/ros_dex_deploy/bundles/dance
python check_bundle.py
bash run_sim.sh
```

`MIMIC_BUNDLE_DIR` 同时作用于校验器、MIMIC/MIMICDEFAULT 和 MuJoCo，重启后生效；取消该变量可回到随包默认站立演示。自定义 `MIMIC_CONFIG_PATH` 可调整增益/阶段配置；启用包选择变量时，ONNX 与 NPZ 路径仍统一来自该包。

MuJoCo 从选中 ONNX 读取 armature、被动阻尼、摩擦与求解器配置，使用 1 ms 物理步长；ROS 主循环和 Mimic 网络推理均为 **100 Hz**，`physical_dt=0.01`、`decimation=1`。

工程师真机实测要求：当前电机特性和 Kp/Kd 配置下，部署闭环至少需要 100 Hz，50 Hz 无法保持平衡。因此每 10 ms 必须使用新观测重新推理，不能只把旧目标以 100 Hz 重发。参考动作仍为 50 fps，每帧使用两次，播放速度不变；Isaac 训练仍为 50 Hz。电机内部 PD 频率与 ROS 发送、网络推理频率是不同概念。详细依据及周期字段含义见 [为什么部署至少需要 100 Hz](../../docs/model_and_dynamics.md#为什么部署至少需要-100-hz)。

旧 `bundle.json` 若仍记录 `policy_control_dt=0.02`，请按训练说明用原 ONNX 和匹配 NPZ 向新目录重新执行 `prepare_deploy.py`；无需为修改部署周期改写权重的训练元数据。

## 4. 真机部署

把整个 `ros_dex_deploy/`、已准备的动作包和 `src/rl_control/lib/` 拷到工控机，按同样顺序完成依赖初始化。核对 `src/rl_control/config/dex_config_real.yaml`：`sim: false`、`control_tool: joystick`，以及所选策略使用的腰部/手臂控制状态。

在工控机上启动：

```bash
sudo -i
cd /path/to/ros_dex_deploy
export MIMIC_BUNDLE_DIR=/absolute/path/to/selected_bundle
# 默认 /home/ubuntu/xos/setup.bash；不同安装位置需显式设置
export XOS_SETUP=/path/to/xos/setup.bash
bash run_real.sh
```

`sudo -i` 会改变环境，包路径和 `XOS_SETUP` 在该终端重新设置。脚本加载 XOS、编译工作区并启动 body_control / joystick / rl_control 链路。操作顺序：F 下拨 → D 准备 → 长按 A 进入 MIMIC → B 播放；C 停止。

