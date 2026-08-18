# v9 训练报告 — 在 LigandMPNN backbone 上微调条件编码器

> 日期：2026-08-18
> 背景：迁移检验定位「LigandMPNN 配体模式电荷失效」（`2026-08-18_seq_sanity_and_transfer.md`）——根因 = 条件编码器只在 MoMPNN backbone 上训练，LigandMPNN 的 h_V 特征分布不匹配。v9 = 用配体复合物数据在 LigandMPNN backbone 上重训同一 `ConditionEncoder`（74880 参数）。
> 计划：`index/PROJECT_V9_LIGAND_PLAN.md`

---

## 一、结论先行

**v9 成功根治「LigandMPNN 配体模式电荷失效」。** 三个配体复合物验证蛋白电荷控制全部达标（dev ≤ 2.0），核心问题 1MBN 从 **dev 14.05 降到 1.55**。

| 蛋白 | target@7.4 | v7 dev | **v9 dev** | v9 recovery | 判定 |
|------|-----------|--------|-----------|------------|------|
| **1MBN**（肌红蛋白+血红素）| +2 | **14.05** ❌ | **1.55** | 0.30 | ✅ **达标** |
| **4DFR**（还原酶+MTX）| −9 | 2.71 | **1.45** | 0.44 | ✅ 达标 |
| **1FQG**（蛋白酶+PNM）| −6 | 0.97（巧合假阳性）| **1.07** | 0.45 | ✅ 达标 |

> **1FQG 说明**：v7 下 dev 0.97 是「负电 target 接近 LigandMPNN 默认偏好」的巧合。v9 下 1FQG 仍需 +0.9 调整（mean −4.93 vs target −6），与 v7 表面"达标"实质不同——v9 是真实的条件控制能力。

## 二、训练数据（用户确认的最终方案）

| 类别 | 数量 | 说明 |
|------|------|------|
| 小分子配体 | 4389 | 药物/辅因子/代谢物 |
| 金属离子配体 | 568 | Zn/Mg/Ca/Fe 配位 |
| 核酸配体（RNA+DNA 合并）| 264 | 核苷酸 |
| **合计** | **4972 复合物 × 8 pH = 39,776 样本** | L≤500，跳过含 X 序列 |

**用户决策**（2026-08-18）：跳过含 X 残基序列；L≤500（v9 覆盖更大蛋白，与 v7 L≤300 分工）；多结合水不单独分类（LigandMPNN 特征化排除水）；DNA 并入 RNA（核酸类）。

## 三、训练过程

- **脚本**：`train_finetune.py --ligand`（改造：load_backbone 自动检测 LigandMPNN 权重 + featurize 切配体模式 + build_domain 透传配体键 + dompdb 支持真实后缀文件）
- **配置**：`λ_c=0.5 λ_kl=0.05 λ_keep=0.5 perturb_prob=0.3 perturb_scale=4.0 placeholder_prob=0.15 charge_temp=0.5`（沿用 v7，**不调 λ_keep**）
- **硬件**：cuda:3，30 epoch，**111.5 min**
- **损失收敛**：

| epoch | total | ce | charge | kl | keep |
|-------|-------|----|----|----|------|
| 1 | 4.30 | 1.43 | 5.08 | 0.11 | 0.66 |
| 15 | 3.55 | 1.55 | 3.17 | 0.21 | 0.81 |
| 30 | 3.33 | 1.55 | **2.72** | 0.22 | 0.81 |

- charge loss 5.08→2.72（电荷学习收敛），keep 稳定 0.81（序列保持），ce 收敛——训练健康

## 四、验证方法

- **wrapper**：`code/tests/ligand_v9/validate_v9.sh`（训练结束后自动运行）
- **验证蛋白**：1MBN/4DFR_chainA/1FQG，`--mode ligand`（LigandMPNN 权重 + 配体上下文），n=20，pH 7.4
- **对比**：v7 编码器（`output/finetune_v7/finetune_epoch030.pt`）同条件

## 五、数据与脚本分离（MoMPNN / LigandMPNN 不共用部分）

| 资源 | 位置 | 说明 |
|------|------|------|
| v9 数据 | `data/ligand_train/{small_mol,metal,rna,dna}/` + `all_pdb/` | 配体复合物（MoMPNN 用 `data/cath/`）|
| v9 标签 | `data/ligand_train/labels.npz` | 4972 复合物 |
| v9 数据脚本 | `code/tests/ligand_v9/fetch_ligand_pdbs.py` | RCSB 获取+分类 |
| v9 标签脚本 | `code/tests/ligand_v9/build_ligand_labels.py` | 标签构建 |
| v9 验证 | `code/tests/ligand_v9/validate_v9.sh` | 验证 wrapper |
| v9 checkpoint | `output/finetune_ligand_v9/` | 30 epoch 全量 |
| v9 验证结果 | `output/transfer_v9/` + `transfer_v9_stats.json` | 验证数据 |

**共用保留**：`transfer_validation.py`/`transfer_stats.py`（两模式）、`train_finetune.py`（`--ligand` 开关）。

## 六、限制与下一步

- ✅ **配体模式电荷控制修复**（本报告核心）
- ⏳ **待验证**：① ESMFold 折叠回折（TM，配体模式是否保持折叠）；② 极端正电 target 是否改善；③ 电荷可靠范围更新（现在 L≤500）
- 📌 与 v7 分工：**无配体/小蛋白用 v7（MoMPNN，L≤300），配体/大蛋白用 v9（LigandMPNN，L≤500）**

## 七、产物

- 训练日志：`log/v9_train.log`；checkpoint：`output/finetune_ligand_v9/finetune_epoch030.pt`
- 验证结果：`output/transfer_v9_stats.json`
- 计划：`index/PROJECT_V9_LIGAND_PLAN.md`
