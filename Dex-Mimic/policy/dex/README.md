# 权重选择与保存

| 文件 | 推荐用途 |
| --- | --- |
| `wbt_model_46400.pt` | 用 `LOAD_CKPT` 初始化新动作训练 |
| `standing_baseline.pt` | 直接播放站立、默认部署演示，或初始化站立微调；持续运行存在水平漂移 |
| `standing_baseline.json` | 与站立基线配套的文件、模型和电机配置校验 |
| `standing_2000.pt` | 从站立基线在 4096 环境下训练约 2000 次迭代的站立参考，用于效果对比 |

新动作先从 `wbt_model_46400.pt` 初始化，训练和导出命令见 [Dex-Mimic](../../README.md)。约 3000 次迭代后检查跟踪效果，再按需要继续训练。

动作效果稳定后，可用 `standing_baseline.pt` 和 `dataset/standing/standing.npz` 单独微调站立头。检查站立稳定性与漂移后，与动作头合并导出，并检查两者切换。

## 加载与续训

- `LOAD_CKPT`：加载权重和兼容的归一化状态，重新开始优化。
- `RESUME=true`：恢复本项目中断的训练，与 `LOAD_CKPT` 互斥。
- 直接播放站立基线时保留配套 JSON。它检查 PT 哈希、模型资产和执行器配置；Isaac 周期为 0.02 s，ROS 部署推理周期为 0.01 s。

其他缺少内置配置记录的旧 PT，请通过 `LOAD_CKPT` 训练后再导出。更换机器人模型或控制参数时，重新检查兼容性。

## 保存新权重

训练产生的 PT 内嵌 `infos.dex_contract`，与同次训练配置配套使用，无需再生成站立基线式的外置 JSON。保留训练 checkpoint 和参考动作，再用当前导出器生成 ONNX 与部署包。

要发布新动作，在相应目录记录参考动作、checkpoint、训练迭代数和使用方法。部署包的 `bundle.json` 由 `prepare_deploy.py` 生成，用于检查 ONNX、模型和 NPZ 是否配套。
