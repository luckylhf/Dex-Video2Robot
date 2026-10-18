# 训练参考数据

- `standing/standing.npz`：当前 39 body / 29 轴 / 50 fps 的静态站立参考，随包保留。
- `npz_dex_raw/`：由 PKL 转换的原始参考，使用时生成。
- `npz_dex/`：带起止站立过渡的动作参考，训练时显式选择。

新动作通过 GMR 输出转换生成，训练前运行 `scripts/check_motion.py`，具体步骤见 [训练说明](../README.md)。
