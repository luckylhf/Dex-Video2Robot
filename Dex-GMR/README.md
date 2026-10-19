# Dex-GMR：视频与动作重定向

[返回总览](../README.md) · [训练与导出](../Dex-Mimic/README.md)

本包把 MP4 中的人体运动映射为当前 Dex 的 29 轴参考动作。视频人体估计由独立 GVHMR 环境执行；本包负责视频调度、SMPL-X 读取、Mink IK 和 PKL 输出，再调用同级 Dex-Mimic 生成 NPZ。

## 1. 安装 GMR 环境

两个包保持同级。以下假定已设置 `DEX_ROOT` 为项目根目录：

```bash
conda create -n dex_gmr python=3.11 -y
conda activate dex_gmr
cd "$DEX_ROOT/Dex-GMR"
python -m pip install -e .
sudo apt install ffmpeg
python -c "import mujoco, mink, smplx, torch; print('GMR imports OK')"
```

依赖由 `setup.py` 声明，包含 MuJoCo 3.10.0、Mink 1.3.0、SMPL-X 0.1.28 和 DAQP。使用 editable 安装并保留源码目录，模型与网格从源码路径加载。

## 2. 准备 GVHMR 与人体资源

按 [GVHMR 官方安装说明](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md) 在单独环境安装 GVHMR，不与 GMR 或 Isaac 的 Torch 环境混用。配套环境使用 Python 3.10、Torch 2.3.0+cu121；PyTorch3D 轮子须与其配套。

GMR 默认读取 `assets/body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz`。发行源码包不捆绑第三方人体模型参数，请按 [SMPL-X 官方入口](https://smpl-x.is.tue.mpg.de/) 获取后放入该目录。也可设置 `SMPLX_MODEL_DIR`，指向包含 `smplx/` 的父目录；CLI `--body-models` 的优先级更高。

```bash
mkdir -p assets/body_models/smplx
# 将获得的三份 SMPLX_*.npz 放入这里
```

放置后先做一次自检，确认文件齐全且未被截断：

```bash
python scripts/check_body_models.py              # 检查 NEUTRAL/MALE/FEMALE
python scripts/check_body_models.py --gender neutral
```

- 文件缺失时脚本直接以退出码 2 结束，并打印下载地址与目标路径；`retarget_motion.py` 也会在加载前调用同一检查，不会抛出 smplx 的底层异常。
- `assets/body_models/smplx/SHA256SUMS` 记录了本包适配版本的哈希；哈希不一致只告警（可能是 SMPL-X 版本不同），加 `--strict` 可视为失败。
- 这三份参数受 MPI 许可限制，`assets/body_models/**` 下的 `*.npz`/`*.pkl` 已在 `.gitignore` 中排除，不要提交或随包分发。

GVHMR 自身需要对应资源，不能只准备 GMR 目录：

```text
GVHMR/inputs/checkpoints/
├── body_models/
│   ├── smplx/SMPLX_{GENDER}.npz
│   └── smpl/SMPL_{GENDER}.pkl
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

按人体模型与权重各自来源准备这些文件。上游 demo 会生成渲染视频，因此 SMPL 渲染资源也需要准备。GMR 随包的人体资源位置与 GVHMR 的位置是两个独立配置。

回到 GMR 环境设置外部解释器：

```bash
export GVHMR_ROOT=/path/to/GVHMR
export GVHMR_PYTHON=/path/to/gvhmr/env/bin/python
export PYTHONNOUSERSITE=1
"$GVHMR_PYTHON" -c "import torch, hmr4d; print('CUDA:', torch.cuda.is_available())"
test -f "$GVHMR_ROOT/tools/demo/demo.py"
```

GVHMR 推理需要 NVIDIA GPU。脚本自动设置 `PYTHONNOUSERSITE=1`，防止用户目录里的 NumPy 覆盖独立环境。

## 3. MP4 → PKL → NPZ

在 GMR 环境运行。日常转换使用恢复模式 `--ik-recovery table-retry`；各模式和约束见第 5 节。把视频路径替换为自己的 MP4；固定相机时才加 `--static-camera`：

```bash
cd "$DEX_ROOT/Dex-GMR"
bash run_video.sh /path/to/dance.mp4 \
  --ik-recovery table-retry \
  --static-camera \
  --work-dir /tmp/dex_video_job \
  --output "$DEX_ROOT/Dex-GMR/data/outputs/dance.pkl"
```

工作目录必须是新的 ASCII 路径。中文视频名可以保留，传给 Hydra 的规范化文件名固定为 `source_30fps.mp4`。不指定工作目录时，脚本在 `data/work/` 下创建 ASCII 名称的独立任务目录。

主要输出：

```text
Dex-GMR/data/outputs/dance.pkl
Dex-GMR/data/outputs/dance.ik.json
Dex-Mimic/dataset/npz_dex_raw/dance.npz
Dex-Mimic/dataset/npz_dex/dance_with_transition.npz
```

默认调用当前 GMR 解释器执行 NPZ 转换。如离线转换使用另一个已配好环境，设置 `MIMIC_PYTHON=/path/to/offline/python`。`--mimic-dir` 可指定另一个配套 Mimic 目录，`--mimic-output-dir` 可把生成数据放到外部目录。

### 从失败阶段继续

使用同一个视频、输出和工作目录，保留原参数并追加 `--resume`：

```bash
bash run_video.sh /path/to/dance.mp4 \
  --ik-recovery table-retry \
  --static-camera --work-dir /tmp/dex_video_job \
  --output "$DEX_ROOT/Dex-GMR/data/outputs/dance.pkl" --resume
```

脚本核对视频哈希、模型与 IK 配置，跳过已生成的规范视频、人体预测和 PKL。GVHMR 已保存人体预测而仅渲染失败时，会提示并继续 IK，不必整段重算。未指定 `--resume` 时不会覆盖已有 PKL。

## 4. 已有动作参数与查看

已有 GVHMR `.pt`、SMPL-X `.npz` 或 BVH 时，可直接做 IK：

```bash
python scripts/retarget_motion.py /path/to/hmr4d_results.pt \
  --ik-recovery table-retry \
  --output data/outputs/dance.pkl
python scripts/view_motion.py data/outputs/dance.pkl
```

BVH 通过 `--bvh-format bvh_lafan1` 或 `bvh_soma` 指定约定。PKL 查看是运动学显示，不代表物理控制策略已通过。

如果视频人物过小、遮挡、机位移动或轨迹漂移明显，先裁段、改善素材或检查人体轨迹。IK 出现 `NoSolutionFound` 时，按下面的流程排查。

<a id="ik-no-solution"></a>

## 5. IK：模式、约束与默认用法

### 默认用哪个？

**日常转换用恢复模式：`--ik-recovery table-retry`。** 第 3、4 节的命令已经带上它。先保留全部约束，某个阶段无解时才放宽重试。

**如果命令里不写 `--ik-recovery`，代码默认是 `strict`。** 因此复制命令或续跑时，请保留 `--ik-recovery table-retry`。

### 有哪些模式？

程序有两种恢复模式：`table-retry` 和 `strict`。另外两个关闭开关用于选择约束，常见组合如下：

| 用法 | 参数 | 正常求解时的硬约束 | 无解后怎么处理 |
| --- | --- | --- | --- |
| **恢复模式（日常使用）** | `--ik-recovery table-retry` | 关节位置、速度、避碰、地面 | 当前 table 重试一次，只保留关节位置 |
| 严格模式 | `--ik-recovery strict` | 关节位置、速度、避碰、地面 | 报错退出 |
| 严格模式，关闭避碰 | `--ik-recovery strict --no-collision-avoidance` | 关节位置、速度 | 报错退出 |
| 严格模式，关闭速度限制 | `--ik-recovery strict --no-velocity-limit` | 关节位置、避碰、地面 | 报错退出 |
| 严格模式，同时关闭两者 | `--ik-recovery strict --no-collision-avoidance --no-velocity-limit` | 仅关节位置 | 报错退出；用于诊断 |

两个关闭开关对**整段动作**生效，也可以与恢复模式组合。例如 `--ik-recovery table-retry --no-collision-avoidance`：正常求解保留位置和速度，无解重试只保留位置。**需要始终保留速度约束时，用 `strict`，且不加 `--no-velocity-limit`。**

### 每种约束管到哪里？

| 约束 | 限制范围 | 什么时候会关闭？ |
| --- | --- | --- |
| 关节位置 | 模型中有关节限位的轴，限制在各自角度范围内；输出 PKL 前再按 URDF 限位检查 | 所有上述模式都保留，无关闭开关 |
| 关节速度 | 当前 Dex 按 URDF 各轴最大速度的 **70%** 限制 IK 关节速度；不限制根节点整体平移、转动速度 | 加 `--no-velocity-limit` 时整段关闭；恢复模式的重试阶段也关闭 |
| 自碰撞 | 只约束 [避碰配置](general_motion_retargeting/collision_config.py) 中列出的几何对，不覆盖任意身体部位之间的碰撞 | 加 `--no-collision-avoidance` 时整段关闭；恢复模式的重试阶段也关闭 |
| 地面 | 配置中脚部几何体靠近地面时限制其向下运动；当前地面高度为 0，余量为 7.5 mm | 与避碰一起关闭，规则同上一行 |

人体姿态匹配、肩部姿态和启用的脚部接触任务是**尽量满足的目标**，不是必须精确达到的硬约束；恢复重试仍保留这些目标。关闭地面硬约束后，原有落地补偿仍执行。这里的开关只影响 IK，后续物理仿真仍使用原模型的碰撞设置。

### 恢复模式具体怎么走？

每帧按顺序执行 table1、table2（两组姿态匹配任务）：

1. 当前 table 先按所选约束正常求解。
2. 仅当求解器报 `NoSolutionFound` 时重试一次：阻尼由 0.5 增至 25，硬约束只保留关节位置。
3. 正常解或重试解成功后，都立即积分更新姿态；当前 table 两次都失败，则恢复该 table 开始时的状态，继续下一个 table。
4. 一个 table 成功就保留其结果；两个都失败则保持本帧开始时的姿态。该帧仍保留，不删帧、不缩短动作时间。

其他程序错误、NaN 或 Inf 直接报错退出，不进入恢复。

PKL 旁边的 `动作名.ik.json` 记录恢复位置和原因。帧状态含义：

| 状态 | 含义 |
| --- | --- |
| `normal` | 所有启用的 table 都正常成功 |
| `recovered` | 所有启用的 table 都成功，至少一个用了重试解 |
| `partial` | 部分 table 失败，保留成功部分的结果 |
| `held` | 所有启用的 table 都失败，保持本帧开始姿态 |

### 已有预测怎么重跑？

不用重新跑 GVHMR，直接复用 `hmr4d_results.pt`，选择一个新的输出名：

```bash
cd "$DEX_ROOT/Dex-GMR"
python scripts/retarget_motion.py \
  /tmp/dex_video_job/human/source_30fps/hmr4d_results.pt \
  --body-models "${SMPLX_MODEL_DIR:-$DEX_ROOT/Dex-GMR/assets/body_models}" \
  --ik-recovery table-retry \
  --output data/outputs/dance_retry.pkl
```

要做约束对照，把上面的模式参数换成表中对应组合，同时更换输出名。`NoSolutionFound` 只说明这次求解没找到解，不能仅凭异常判断是哪条约束冲突；对照时使用同一份人体预测。

**更换模式或关闭开关后，不要在旧任务上直接加 `--resume`。** 复用人体预测时按上面重跑 IK；从 MP4 重跑时换新的 `--work-dir` 和输出名。只有参数一致的原任务才能续跑。

### 解出来后，还检查什么？

**IK 完成不等于动作合格。** 恢复阶段没有速度、避碰和地面硬约束，必须检查最终带过渡的 NPZ：

```bash
cd "$DEX_ROOT/Dex-Mimic"
"${MIMIC_PYTHON:-python}" scripts/pkl_to_npz.py \
  --input_file "$DEX_ROOT/Dex-GMR/data/outputs/dance_retry.pkl" \
  --robot dex_evt \
  --output_name dataset/npz_dex_raw/dance_retry \
  --output_dir dataset/npz_dex \
  --add_transition --recompute_velocities && \
"${MIMIC_PYTHON:-python}" scripts/check_motion.py \
  dataset/npz_dex/dance_retry_with_transition.npz
```

`run_video.sh` 会自动执行转换和后置检查；直接调用 `retarget_motion.py` 时，需手动执行上面的命令。

- **检查通过才训练。** NPZ 文件存在不代表检查通过；训练时显式选择通过检查的文件。
- 检查器校验模型、名称、帧率、数值、四元数、URDF 关节位置与速度，以及抽样 FK。最终速度按 **URDF 限位**检查，不是 IK 使用的 70%。
- 检查器**不检查自碰撞、平衡或跟踪效果**。回放时看身体穿插、脚底、漂移，以及 `partial` / `held` 帧；策略效果再到 Isaac / MuJoCo 中验证。
- 位置或速度超限时，修正动作、降速或裁段后重新生成，不修改 URDF 或检查器阈值来过关。

模型与 39/30 body、四元数等接口约定见 [模型说明](../docs/model_and_dynamics.md)。
