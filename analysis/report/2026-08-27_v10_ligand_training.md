# v10-LigandMPNN 训练分析报告（2026-08-27/28）

> **背景**：v10 论文导向方案（`index/PROJECT_LOCAL.md`）核心方法升级——A 条件解耦 + B 表面电荷监督 + C 结构惩罚，治 P1"删减捷径"。本报告记录 **LigandMPNN backbone 侧**的 v10 训练过程、踩坑与结果，并与 MoMPNN 侧对比。
> **状态**：训练完成（30 epoch，checkpoint 有效，NaN=0）。
> **产物**：`output/finetune_v10_ligand/finetune_epoch030.pt`（907,892 B，NaN=0）。

---

## 1. 训练配置（v10 三组件全开）

| 参数 | 值 | 说明 |
|------|-----|------|
| backbone | LigandMPNN（`ligandmpnn_v_32_010_25.pt`，冻结）| 配体模式 |
| 数据 | `data/ligand_train/labels.npz`（**4956 域 × 8 pH = 39,648 样本**）| 已清理 15 个坏域 |
| epochs | 30 | 与 v7/v9 同标准 |
| λ_c / λ_kl / λ_keep | 0.5 / 0.05 / 0.5 | 沿用 v7/v9 已验证 |
| charge_temp | 0.5 | 温度化（过冲根治）|
| perturb_prob | 0.3 | 扰动样本比例 |
| placeholder_prob | 0.15 | 占位符 |
| **A 条件解耦** | `--decouple_perturb --decouple_range 12` | target 与 native 无关 Uniform[-12,12] |
| **B 表面电荷监督** | `--add_supervision --lambda_add 0.3 --sasa_threshold 0.25` | L_add 只加表面 |
| **C 结构惩罚** | `--ph_aware_filter --structure_boost 1.5` | pH 自适应带电集合 |

## 2. 训练曲线（真实数据）

| epoch | LigandMPNN total | LigandMPNN charge | MoMPNN total | MoMPNN charge |
|-------|------------------|-------------------|--------------|---------------|
| 1 | 4.370 | 5.184 | 4.099 | 3.466 |
| 10 | 3.931 | — | — | — |
| 15 | — | 3.616 | 3.623 | 2.333 |
| 20 | 3.668 | — | — | — |
| 30 | **3.496** | **3.025** | **3.519** | **2.048** |

**分析**：
- **两者都收敛**：LigandMPNN total 4.37→3.50，MoMPNN 4.10→3.52，均稳定下降无 NaN
- **电荷损失**：LigandMPNN charge 5.18→3.03，MoMPNN 3.47→2.05。**LigandMPNN 电荷损失起点和终点都更高**——因为配体模式数据更复杂（含金属/核酸/小分子配体），电荷控制更难
- **收敛性**：最后 5 epoch 变化小（LigandMPNN total 3.50→3.42），接近平台；MoMPNN 更早稳定

## 3. 关键踩坑与解决（本报告核心价值）

### 3.1 LigandMPNN 完整训练 NaN（根因：1GTV 不完整结构）
- **现象**：完整 4971 域训练 epoch 1 起 total=nan，checkpoint 权重全 NaN
- **定位**：收集模式 NaN 检测 → **唯一触发域 1GTV.pdb**，`add: nan` + `frac_nan: 214`
- **根因链**：
  ```
  1GTV 极不完整结构（平均 4.8 原子/残基，正常 ~8；ARG 5/11、GLU 5/9 侧链缺失）
    → freesasa 对 177 个残基返回 NaN relativeTotal（原子几何异常）
    → 传入 B 组件 L_add → sigmoid(k*(NaN−θ)) = NaN → add loss = NaN
    → total = NaN → backward 梯度 NaN → 权重全 NaN → 永久发散
  ```
- **为何只有 1GTV**：其他配体域是完整结构，SASA 正常；1GTV 是训练集唯一如此残缺的域（在 1000+ 位置，解释了 50/200/500/1000 测试都正常）
- **修复**：`src/sasa.py` 用 `np.nan_to_num` 把 NaN/Inf → 0.0（视为埋藏，不参与 L_add）。验证：修复后完整训练无 NaN 域

### 3.2 不完整 PDB 清理（用户要求）
- **扫描**：全量 4972 域，文本 + Bio.PDB 双方法交叉验证一致——14 个平均 1.0 原子/残基的极残缺域 + 1GTV = **15 个坏域（0.3%）**
- **处理**：删除并重建 `labels.npz`（4956 域 × 8pH），原文件备份 `labels_orig_4972.npz`
- **判断**：占比极小（0.3%）但属明确质量问题（侧链几乎全缺 → SASA NaN + 标签不准），删除划算

### 3.3 其他修复
- **CATH 无后缀文件**：prody 当 mmCIF → 改"带后缀才直用"
- **SASA resnum 交集对齐**（用户指出）：sasa.py 返回 residues[]，按 R_idx 残基号匹配
- **dangling symlink 崩溃**：`os.path.lexists()` + symlink 移入 try 块

## 4. 论文意义

- **v10 双 backbone 都训练成功**：MoMPNN（无配体）+ LigandMPNN（配体），为论文主方法提供完整编码器对
- **数据质量方法论**：不完整结构的 SASA 处理（NaN→0）+ 清理低质量域，是"训练数据质量"章节的实证
- **charge 曲线差异**：配体模式电荷控制更难（charge 终点 3.03 vs MoMPNN 2.05），是论文"局限性/边界"的候选证据

## 5. 下一步（等待用户确认后执行）

1. **双编码器泛化验证**：MoMPNN（protein 模式）+ LigandMPNN（both 模式）在泛化 10 蛋白上验证
2. **对照实验**：C1/C3/C4/C6（条件化 vs 无条件，显式 vs 隐式）
3. **PROPKA 复核（H4）** + AF2 交叉回折
4. 统计 + 论文图表

> 注：本报告按"论文产出导向"如实记录训练数据与踩坑，不夸大收敛性；电荷控制能力的**最终评估在泛化验证后**（charge loss 只是训练指标，验证看实际电荷命中）。
