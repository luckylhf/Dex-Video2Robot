# SMPL-X 人体资源

需要准备 SMPL-X NEUTRAL、MALE、FEMALE 三份 NPZ。**发行源码包只携带本说明与 `smplx/SHA256SUMS`**；参数文件请按原始来源获取并保留相应许可。

GMR 默认读取此目录；`--body-models` 优先，其次是 `SMPLX_MODEL_DIR`。路径指向包含 `smplx/` 的父目录。GVHMR 自身的模型、推理权重及环境需另行准备，见 [GMR 安装说明](../../README.md)。

## 获取

1. 在 <https://smpl-x.is.tue.mpg.de/> 注册并下载 SMPL-X v1.1 模型（非商业科研用途免费，商用需另购授权）。
2. 解压后把 `SMPLX_NEUTRAL.npz`、`SMPLX_MALE.npz`、`SMPLX_FEMALE.npz` 放到本目录的 `smplx/` 下。

## 自检

```bash
python scripts/check_body_models.py
```

- 缺文件：退出码 2，打印官方下载地址与目标路径。
- 哈希不一致：默认只告警（可能是 SMPL-X 版本不同），`--strict` 下视为失败。

## 哈希清单（本包适配版本）

| 文件 | sha256 |
| --- | --- |
| `SMPLX_NEUTRAL.npz` | `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992` |
| `SMPLX_MALE.npz` | `ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5` |
| `SMPLX_FEMALE.npz` | `b2a3686c9d6d218ff6822fba411c607a3c8125a70af340f384ce68bebecabe0e` |

机器可读版本见 [`smplx/SHA256SUMS`](smplx/SHA256SUMS)，可直接 `sha256sum -c`。

## 许可与分发

SMPL-X 参数受 Max Planck Institute 的许可限制：仅限非商业科研用途，且**不允许再分发**模型文件。因此：

- `.gitignore` 已排除 `assets/body_models/**/*.npz` 与 `*.pkl`；
- 打包交付时同样要排除这些文件，只保留本说明与 `SHA256SUMS`；
- GVHMR 目录下的 `inputs/checkpoints/body_models/`（含 `smpl/SMPL_*.pkl`）适用同样的限制。
