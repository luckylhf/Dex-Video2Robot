# MuJoCo ROS2 节点

加载当前 Dex 模型，读取所选 ONNX 的动力学参数，执行外部 PD→力矩，并发布关节和 IMU 状态。模型注册名 evt2 沿用兼容标识，实际指向 resources/dex/mjcf/dex_29.xml。

安装、完整启动和操作顺序统一见 [ROS2 部署说明](../../README.md)。
