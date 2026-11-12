# 第三方来源与许可

本项目中由 Loong 原创且有权授权的部分采用 [BSD-3-Clause](LICENSE)。第三方内容保留原作者的版权和许可；顶层 LICENSE 不替代下表中的许可。

## 随源码保留的代码

| 来源 | 本项目中的使用范围 | 许可文本 |
| --- | --- | --- |
| [GMR](https://github.com/YanjieZe/GMR)，Copyright 2025 Yanjieze | `Dex-GMR/` 中继承的动作读取、重定向与查看代码 | MIT，见 [Dex-GMR/LICENSE](Dex-GMR/LICENSE) |
| [xGMR](https://github.com/Open-X-Humanoid/xGMR)，Open-X-Humanoid | `Dex-GMR/` 中基于 xGMR 的 Dex 适配；本地修改包括模型映射、IK 约束、数据读取与查看 | Apache-2.0，见 [Dex-GMR/LICENSE](Dex-GMR/LICENSE) |
| [xMimic](https://github.com/Open-X-Humanoid/xMimic) | `Dex-Mimic/` 中继承的动作跟踪、训练和导出代码；本地增加 Dex 适配与部署入口 | MIT，保留其 Isaac Lab 作者声明，见 [Dex-Mimic/LICENCE](Dex-Mimic/LICENCE) |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | `Dex-Mimic/source/whole_body_tracking/whole_body_tracking/utils/exporter.py` 中带有 BSD 标记的代码 | [BSD-3-Clause](licenses/IsaacLab-BSD-3-Clause.txt)；该文件保留原版权头 |
| [xSIM_MUJOCO](https://github.com/Open-X-Humanoid/xSIM_MUJOCO) | ROS MuJoCo 仿真代码与引用的电机配置 | [Apache-2.0](licenses/Apache-2.0.txt)，另见 [节点 LICENSE](Dex-Mimic/ros_dex_deploy/src/mujoco_node/LICENSE) |
| Open X-Humanoid / Deploy_Tienkung、TU Munich broccoli | `Dex-Mimic/ros_dex_deploy/src/rl_control/` 中的控制接口及相关代码 | BSD-3-Clause / MIT，见 [控制节点 LICENSE](Dex-Mimic/ros_dex_deploy/src/rl_control/LICENSE.txt) |

分发这些代码时保留适用的完整许可与作者声明；修改 Apache-2.0 来源文件时注明修改。新写的 BVH 解析器 `Dex-GMR/general_motion_retargeting/utils/bvh.py` 使用 NumPy / SciPy 实现，按文件中的 BSD-3-Clause 标记授权。

## 外部工具与数据

视频动作推荐流程为 **GVHMR 人体预测 → SMPL-X 人体重建 → GMR 或 Dex-Retarget 机器人重定向**。安装步骤见 [Dex-GMR](Dex-GMR/README.md)。这些外部资源的使用权需按原许可取得。

- **GVHMR**：Zhejiang University 3D Vision Group。软件许可允许教育、研究和非营利用途；基于其工作的修改须开源并禁止商用。商业使用请联系原作者。完整条款见 [官方 LICENSE](https://github.com/zju3dv/GVHMR/blob/main/LICENSE)，本地保留[许可副本](licenses/GVHMR.txt)。本项目通过外部环境调用 GVHMR，预测文件适配器承接 xGMR 的读取接口。
- **SMPL-X / SMPL**：Max Planck Institute for Intelligent Systems 及合作者提供的人体模型。按[官方模型许可](https://smpl-x.is.tue.mpg.de/modellicense.html)获取和使用，商业用途需另行授权；模型参数不随源码分发。准备方法见[人体资源说明](Dex-GMR/assets/body_models/README.md)。
- **LAFAN1**：Ubisoft La Forge。原始动作数据采用 [CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt)。本项目支持其 BVH 格式，不分发原始动作数据或 Ubisoft 的 BVH 解析代码。使用数据时须遵守非商业、署名及禁止分享改编材料等条件；额外权限由权利人授予。
- **运行依赖**：MuJoCo、Mink、NumPy、SciPy、PyTorch、SMPL-X Python 包、Isaac Lab、RSL-RL、ROS2 和 ONNX Runtime 等由各自的软件发行包提供，适用各自版本的许可。GVHMR 的预训练权重及依赖也分别遵循其来源条款。

使用本仓库的 BSD 代码不等于取得上述外部工具、模型和数据的商业授权。使用视频流程前，请按实际选用的资源确认适用许可。

## 论文引用

使用 GVHMR 的研究工作请引用：

```bibtex
@inproceedings{shen2024gvhmr,
  title={World-Grounded Human Motion Recovery via Gravity-View Coordinates},
  author={Shen, Zehong and Pi, Huaijin and Xia, Yan and Cen, Zhi and Peng, Sida and Hu, Zechen and Bao, Hujun and Hu, Ruizhen and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia Conference Proceedings},
  year={2024}
}
```

使用 SMPL-X 的出版物请引用：

```bibtex
@inproceedings{SMPL-X:2019,
  title={Expressive Body Capture: 3D Hands, Face, and Body from a Single Image},
  author={Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle={Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

使用 LAFAN1 时，上游建议引用：

```bibtex
@article{harvey2020robust,
  author={Felix G. Harvey and Mike Yurick and Derek Nowrouzezahrai and Christopher Pal},
  title={Robust Motion In-Betweening},
  journal={ACM Transactions on Graphics},
  volume={39},
  number={4},
  year={2020}
}
```
