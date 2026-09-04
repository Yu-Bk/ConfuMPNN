# per-epoch 前向 val-loss 曲线 — train vs val（v12.2 / v12.3 / v14 配体）

> 方法（2026-09-04 用户裁定）：**前向全量 val-loss 回放**（`code/tests/val_loss_curve.py`，no-grad、不生成、
> 全量验证集每点、self-arm 训练同口径）。指标双出：**平均电荷偏差 cd（结果）+ 总 loss（过程，train 同口径）**。
> 验证集：蛋白 = CATH 15% hold-out 1176+supp23（对齐 1104 域解析通过）；配体 = 外部 805。
> 产物：`output/val_loss_curve_{v12_2,v12_3,v14_ligand}.json`；合并作图数据 `output/val_loss_curve_trainval_plot.json`。

## 一、曲线总览（train 全臂 vs val self-arm；train cd_self 与 val cd 同口径可比）

### v12_2 蛋白（30ep，MoMPNN）
| ep | train total | train cd_self | val total | val cd | val rec |
|---|---|---|---|---|---|
| 1 | 4.962 | 3.460 | 5.616 | 4.429 | 0.422 |
| 16 | 4.291 | 2.355 | 4.492 | 2.606 | 0.407 |
| 30 | 4.199 | 2.233 | 4.371 | 2.395 | 0.401 |

### v12_3 蛋白（40ep，MoMPNN）
| ep | train total | train cd_self | val total | val cd | val rec |
|---|---|---|---|---|---|
| 1 | 4.924 | 3.443 | 5.265 | 3.871 | 0.420 |
| 21 | 4.231 | 2.311 | 4.189 | 2.257 | 0.404 |
| 40 | 4.095 | 2.118 | 4.173 | 2.151 | 0.399 |

### v14 配体（50ep，LigandMPNN RNA/DNA+A1global）
| ep | train total | train cd_self | val total | val cd | val rec |
|---|---|---|---|---|---|
| 1 | 6.217 | 5.332 | 5.142 | 4.046 | 0.519 |
| 26 | 4.635 | 3.051 | 4.125 | 2.465 | 0.482 |
| 50 | 4.389 | 2.758 | 3.971 | 2.180 | 0.472 |

## 二、判定：是否"充分拟合 + 无过拟合"
- **无过拟合 ✅**：三版 val total/cd 全程随 train 下降并趋平台，末段**无回升**；v12_3 val 末 ≈ train（2.15 vs 2.12）、v14 val 末 < train（2.18 vs 2.76）。
- **拟合程度**：目标电荷偏差 cd_self 已压到 **2.1-2.4（蛋白）/ 2.2（配体）**并接近平台，但**非零残差**——~2 个净电荷的平均 |期望−目标| 是当前监督下（含删减捷径副作用）的模型极限，非过拟合所致；真实功能命中由**测试集**定（v14 clean H2 45/50=90%）。
- val rec（native 回收）稳定 ~0.40-0.42（蛋白）/0.47-0.52（配体），未随训练塌缩。
- 说明：v12_2/v12_3 末段仍微降（30/40ep 未完全压平），v14 末段已较平（50ep）；三版**非过拟合，属近平台/欠一点收敛**，加大轮次收益有限（与既有"响应增益不是轮次问题"结论一致）。

## 三、数据位置（供作图）
- 逐点曲线：`output/val_loss_curve_{v12_2,v12_3,v14_ligand}.json`（meta 含 lambda/配置）
- train-vs-val 对齐：`output/val_loss_curve_trainval_plot.json`（train_total/ce/cd_self/charge + val_total/ce/cd/rec）
- train 侧源：`log/v12_2_train_mompnn.log`、`log/v12_3_train_mompnn.log`、`log/v14_ligand_train_stdout.log`
- 口径细节：`code/tests/val_replay_configs.md`、`session/2026-09-04_val_loss_curve_build.md`
