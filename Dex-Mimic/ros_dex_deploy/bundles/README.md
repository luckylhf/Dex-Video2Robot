# 选择部署包

每个子目录包含 ONNX、参考动作和校验清单。设置 `MIMIC_BUNDLE_DIR` 后，校验器、控制器与 MuJoCo 使用同一个包，重启节点后生效。

## 随包动作

| 目录 | 参考动作 | 动作头 | 站立头 |
| --- | --- | --- | --- |
| `gbtc_s2000` | `gbtc_with_transition.npz`，947 帧 / 18.9 s / 50 fps | GBTC 动作训练约 2000 次迭代 | `policy/dex/standing_2000.pt` |

在项目根目录设置 `DEX_ROOT` 后运行：

```bash
cd "$DEX_ROOT/Dex-Mimic/ros_dex_deploy"
export MIMIC_BUNDLE_DIR="$PWD/bundles/gbtc_s2000"
python check_bundle.py
ROS_DOMAIN_ID=77 bash run_sim.sh
```

独立 MuJoCo（无需 ROS）使用同一动作包：

```bash
cd "$DEX_ROOT/Dex-Mimic"
POLICY="$MIMIC_BUNDLE_DIR/policy_merged.onnx" \
  NPZ_DIR="$MIMIC_BUNDLE_DIR/npz" bash sim2sim_dex.sh
```

取消 `MIMIC_BUNDLE_DIR` 可回到默认站立演示。操作顺序见[部署说明](../README.md)。

## 创建自己的包

先完成[训练与导出](../../README.md)，再用 `scripts/prepare_deploy.py` 输出到一个新目录：

```text
dance/
├── policy_merged.onnx    # 双头策略、归一化和配置
├── npz/                  # 该策略使用的参考动作
└── bundle.json           # 模型、ONNX 和 NPZ 的校验清单
```

包与导出时的 URDF、MJCF 和网格绑定。更换模型或网格后，检查控制配置并重新导出；影响动力学的修改还需重新训练或微调。校验失败时使用匹配的资产和策略，避免手工改哈希跳过检查。

复现训练需要保留原始 checkpoint、参考动作与训练配置；部署包用于运行策略，不包含完整训练日志。
