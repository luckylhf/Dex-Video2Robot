# 随包站立演示

本目录是当前本体的站立演示包：两个 ONNX 头均来自 `policy/dex/standing_baseline.pt`，只包含 `standing.npz`。运行自己的动作需另行训练并生成部署包。

ROS 部署推理为 100 Hz，参考动作为 50 fps。`bundle.json` 的 `policy_control_dt=0.01` 表示 ROS 推理周期；`training_control_dt=0.02` 保留 ONNX 对应的训练周期。独立 MuJoCo 默认使用 50 Hz。

新动作请完成训练和导出后，使用 `prepare_deploy.py --output` 创建新包，再通过 `MIMIC_BUNDLE_DIR` 选择，见 [部署说明](../../../../../README.md)。
