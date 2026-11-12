# SMPL-X 人体资源

从 [SMPL-X 官网](https://smpl-x.is.tue.mpg.de/)注册并下载 v1.1 模型，将三份文件放入：

```text
smplx/SMPLX_NEUTRAL.npz
smplx/SMPLX_MALE.npz
smplx/SMPLX_FEMALE.npz
```

GMR 默认读取本目录。使用外部目录时，设置 `SMPLX_MODEL_DIR` 或 `--body-models`，路径指向包含 `smplx/` 的父目录；命令行参数优先。

在 `Dex-GMR/` 下运行：

```bash
python scripts/check_body_models.py
python scripts/check_body_models.py --gender neutral
```

脚本检查文件完整性并对照 [SHA256SUMS](smplx/SHA256SUMS)。缺失文件返回退出码 2；版本哈希不一致时告警，需要严格匹配时追加 `--strict`。

模型参数按 [MPI 官方许可](https://smpl-x.is.tue.mpg.de/modellicense.html)获取和使用，商业用途需另行授权，未经许可不得再分发。源码包仅携带本说明与哈希清单。

GVHMR 的 `inputs/checkpoints/body_models/` 需要单独准备，见 [GMR 安装说明](../../README.md#2-准备-gvhmr-与人体资源)。
