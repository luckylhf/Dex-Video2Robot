# 部署包（bundles）

每个子目录是一个自包含的部署包，由 `Dex-Mimic/scripts/prepare_deploy.py` 生成，
`MIMIC_BUNDLE_DIR` 指向哪个目录，校验器、FSM 和 MuJoCo 节点就用哪一个。

## 现有包

| 目录 | 参考动作 | 动作头 | 站立头 |
| --- | --- | --- | --- |
| `gbtc_s2000` | `gbtc_with_transition.npz`（947 帧 / 18.9 s / 50 fps） | `1021_005958_dex_evt_repair_balance_sim2sim_medium/model_2000.pt`（约 2000 迭代） | `policy/dex/standing_2000.pt`（由 `policy/dex/standing_baseline.pt` 微调 2001 迭代） |

## 用法

```bash
cd Dex-Mimic/ros_dex_deploy
MIMIC_BUNDLE_DIR=$PWD/bundles/gbtc_s2000 python check_bundle.py   # 先校验
ROS_DOMAIN_ID=77 MIMIC_BUNDLE_DIR=$PWD/bundles/gbtc_s2000 bash run_sim.sh
# 独立 MuJoCo（无 ROS）：
cd Dex-Mimic && POLICY=$PWD/ros_dex_deploy/bundles/gbtc_s2000/policy_merged.onnx \
  NPZ_DIR=$PWD/ros_dex_deploy/bundles/gbtc_s2000/npz bash sim2sim_dex.sh
```

## 结构

```
gbtc_s2000/
├── policy_merged.onnx    双头 actor + 观测归一化 + 元数据契约
├── npz/                  参考动作（可多文件）
└── bundle.json           清单与校验（model/onnx/npz 哈希、contract_version、purpose）
```

## 与模型资产绑定

`policy_merged.onnx` 的元数据记录了打包时的 `asset_sha256`
（`source/whole_body_tracking/.../assets/dex_evt` 下全部 `.xml/.urdf/.stl` 的指纹）。
**替换任何网格后，旧包会直接校验失败**：

```
ValueError: Policy URDF/meshes differ from current assets; re-export after checking the model
```

此时需要重新导出。若只是网格外观变化、策略权重不需要变，
可以只把 checkpoint 的 `infos.dex_contract` 更新为新资产指纹后重新导出
（本目录下的 `gbtc_s2000` 就是这样重锚的，权重与训练产物一致）。

## 不在仓库里的东西

训练日志（`Dex-Mimic/logs/`）、数据集（`Dex-Mimic/dataset/`）不出现在版本库里；
需要复现某个包时，用上面「动作头 / 站立头」两列指向的 checkpoint 重新执行
`export.sh` 与 `prepare_deploy.py`。
