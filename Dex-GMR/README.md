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

在 GMR 环境运行。把视频路径替换为自己的 MP4；固定相机时才加 `--static-camera`：

```bash
cd "$DEX_ROOT/Dex-GMR"
bash run_video.sh /path/to/dance.mp4 \
  --static-camera \
  --work-dir /tmp/dex_video_job \
  --output "$DEX_ROOT/Dex-GMR/data/outputs/dance.pkl"
```

工作目录必须是新的 ASCII 路径。中文视频名可以保留，传给 Hydra 的规范化文件名固定为 `source_30fps.mp4`。不指定工作目录时，脚本在 `data/work/` 下创建 ASCII 名称的独立任务目录。

主要输出：

```text
Dex-GMR/data/outputs/dance.pkl
Dex-Mimic/dataset/npz_dex_raw/dance.npz
Dex-Mimic/dataset/npz_dex/dance_with_transition.npz
```

默认调用当前 GMR 解释器执行 NPZ 转换。如离线转换使用另一个已配好环境，设置 `MIMIC_PYTHON=/path/to/offline/python`。`--mimic-dir` 可指定另一个配套 Mimic 目录，`--mimic-output-dir` 可把生成数据放到外部目录。

### 从失败阶段继续

使用同一个视频、输出和工作目录，保留原参数并追加 `--resume`：

```bash
bash run_video.sh /path/to/dance.mp4 \
  --static-camera --work-dir /tmp/dex_video_job \
  --output "$DEX_ROOT/Dex-GMR/data/outputs/dance.pkl" --resume
```

脚本核对视频哈希、模型与 IK 配置，跳过已生成的规范视频、人体预测和 PKL。GVHMR 已保存人体预测而仅渲染失败时，会提示并继续 IK，不必整段重算。未指定 `--resume` 时不会覆盖已有 PKL。

## 4. 已有动作参数与查看

已有 GVHMR `.pt`、SMPL-X `.npz` 或 BVH 时，可直接做 IK：

```bash
python scripts/retarget_motion.py /path/to/hmr4d_results.pt \
  --output data/outputs/dance.pkl
python scripts/view_motion.py data/outputs/dance.pkl
```

BVH 通过 `--bvh-format bvh_lafan1` 或 `bvh_soma` 指定约定。PKL 查看是运动学显示，不代表物理控制策略已通过。

如果视频人物过小、遮挡、机位移动或轨迹漂移明显，先裁段、改善素材或检查人体轨迹。IK 出现 `NoSolutionFound` 时，按下面的流程排查。

<a id="ik-no-solution"></a>

## 5. 常见问题：IK 无解（NoSolutionFound）

出现 `NoSolutionFound: QP solver daqp failed to find a solution`，表示当前 IK 求解未找到可行解。单凭异常不能确定是动作异常、约束冲突还是数值问题。先检查失败位置附近的人体预测、机器人模型和 IK 配置；保留日志及已生成的 `hmr4d_results.pt`，不必重新跑 GVHMR。

**建议先显式关闭 IK 避碰，保留关节位置和速度约束，再执行动作数据检查。** 两个关闭开关 `--no-collision-avoidance`、`--no-velocity-limit` 默认均不启用；默认求解失败会直接退出，不会自动关约束或静默降级。

### 已有 GVHMR 预测：直接从 IK 重跑

以下接续第 3 节的 `/tmp/dex_video_job`，写入一个新的 PKL，保留原始失败任务和产物：

```bash
cd "$DEX_ROOT/Dex-GMR"
python scripts/retarget_motion.py \
  /tmp/dex_video_job/human/source_30fps/hmr4d_results.pt \
  --body-models "${SMPLX_MODEL_DIR:-$DEX_ROOT/Dex-GMR/assets/body_models}" \
  --no-collision-avoidance \
  --output data/outputs/dance_no_collision.pkl
```

然后转换并检查新动作。离线解释器沿用前面配置的 `MIMIC_PYTHON`，未设置时使用当前 `python`：

```bash
cd "$DEX_ROOT/Dex-Mimic"
"${MIMIC_PYTHON:-python}" scripts/pkl_to_npz.py \
  --input_file "$DEX_ROOT/Dex-GMR/data/outputs/dance_no_collision.pkl" \
  --robot dex_evt \
  --output_name dataset/npz_dex_raw/dance_no_collision \
  --output_dir dataset/npz_dex \
  --add_transition --recompute_velocities && \
"${MIMIC_PYTHON:-python}" scripts/check_motion.py \
  dataset/npz_dex/dance_no_collision_with_transition.npz
```

**只有 `check_motion.py` 成功退出后，才继续训练。** 转换器可能已写出 NPZ，而后置检查才发现超限；文件存在不代表检查通过。训练时显式选择 `dataset/npz_dex/dance_no_collision_with_transition.npz`，避免把未通过检查的诊断动作混入训练目录。

`check_motion.py` 校验模型、名称、帧率、数值、四元数、URDF 关节位置与速度，以及抽样 FK；它不检查自碰撞。回放参考动作时仍需检查身体穿插、脚底和轨迹漂移，再分别验证 Isaac / MuJoCo 策略表现。

该开关关闭整个 IK 避碰分支，包括几何避碰和地面几何高度约束。关节位置、速度约束仍保留；落地补偿和 NPZ 过渡处理仍执行。后续物理仿真沿用原模型的碰撞设置。

### 从 MP4 重新启动：使用新的任务目录

也可以把显式开关交给视频入口，由它自动完成 IK、转换与后置检查。下面是复用预测方案的替代入口；工作目录和输出名均使用新路径，固定相机素材才额外添加 `--static-camera`：

```bash
cd "$DEX_ROOT/Dex-GMR"
bash run_video.sh /path/to/dance.mp4 \
  --no-collision-avoidance \
  --work-dir /tmp/dex_video_job_no_collision \
  --output "$DEX_ROOT/Dex-GMR/data/outputs/dance_no_collision_video.pkl"
```

**改变 IK 开关后，不能在原任务上直接追加 `--resume`。** 续跑会核对 IK 配置并拒绝不一致的参数；需要复用人体预测时，使用上面的 `retarget_motion.py` 入口并选择新输出。新任务后续若中断，才使用它自己的相同参数、工作目录和输出加 `--resume`，不要手工修改 `pipeline.json` 绕过核对。

### 进一步排查

如果关闭避碰仍无解，可将 `--no-collision-avoidance` **替换为** `--no-velocity-limit`，用独立的新输出做速度约束对照；关闭速度约束更容易得到超限动作，仍须执行同样的 NPZ 后置检查。若只有同时关闭两者才能求解，先将结果作为诊断数据，检查人体轨迹、模型与约束配置，并裁段、降速或修正预测，再重新检查。

关节位置或速度超限时，应修正参考动作并重新生成，不要通过修改 URDF 限位或放宽检查器阈值让数据过关。参考动作检查通过后，还需训练策略并检查物理仿真中的跟踪效果、平衡和起止切换。

模型与 39/30 body、四元数等接口约定见 [模型说明](../docs/model_and_dynamics.md)。
