# 2026-09-05 Task A — 蛋白模式版本史 + 蛋白 vs 配体删减差异分析（session 过程记录）

> 任务：梳理蛋白(MoMPNN)模式版本更迭史，用真实数据解释"为何蛋白模式删减较轻/表面可控、配体模式即便 A1-global 仍删减"。
> 纯 CPU 文档/数据分析，零新采样，不 commit/push。
> 产出：`analysis/report/2026-09-05_taskA_timeline.md`（时间线）、`analysis/report/2026-09-05_protein_history_vs_ligand_deletion.md`（主报告）、本 session、`output/protein_vs_ligand_zone_deletion.json`（对照数据）、`code/tests/ligand_v9/protein_vs_ligand_zone_deletion.py`（复现脚本）。

## 1. 读了哪些权威材料
- 蛋白模式：`index/PROJECT_LOCAL_V12_2.md`、`index/PROJECT_LOCAL.md`、`session/2026-08-29_v11_ablation.md`、`analysis/report/2026-08-28_v10_diag_response.md`、`2026-08-31_v12_2_{diag,summary}.md`、`2026-09-03_v12_3_vs_v12_2_final.md`、`2026-08-18_model_charge_limits.md`（P1 起源）
- 配体模式：`analysis/report/2026-09-01_v12_2_ligand_comp_analysis.md`、`2026-09-02_v13_ligand_validation.md`、`2026-09-04_v14_clean_validation.md`、`2026-09-04_v14_deletion_location.md`、`2026-09-04_v14_fixbinding.md`、`2026-09-04_v14_largen_search.md`、`2026-09-04_v13_in10_validation.md`
- 代码：`code/src/v12_losses.py`（surface_composition/gravy/charge_target/pocket_count 全部四个损失）、`code/train_finetune.py`（分区 mask、监督装配、pocket_mode keep/global 逻辑）

## 2. 验证前提的关键动作
用户前提："蛋白模式 v12.2 能较好控制表面电荷、删减较轻（1A65 健康 1.20/1.27）；配体模式 v14 更严 A1-global 仍删 0.43-0.69×"。

核实方式：写 `code/tests/ligand_v9/protein_vs_ligand_zone_deletion.py`，**复用 v14 删除定位分析的同一套几何代码**（parse_PDB + freesasa 全结构含配体、pocket=Cα-配体≤8Å、surface=frac_sasa≥0.25、core=其余、DEKR 带电），对蛋白模式 small-batch（`generalization_v12_2_calib_small/protein`）的 native 臂 n=30 生成、配体 v14-clean n=50 生成逐蛋白算三区保留率。
- 复算结果与 `2026-09-03_v12_3_vs_v12_2_final.md §五` 蛋白表逐项吻合（1AZM 0.48/0.50、1AS2 0.61/0.59、2FEO 0.75/0.75、5CQH 0.67/0.54、1CGE 0.84/0.62、1A65 1.20/1.27、1BJ4 0.63/0.59）。
- 产出 JSON：`output/protein_vs_ligand_zone_deletion.json`。

## 3. 关键数据结论（详见主报告）
1. 蛋白 v12.2 native 臂**并非健康**：10 蛋白中 8 个 < native（全链 CHG 保留 0.49-0.77），1A65 是"过度添加 1.23×"而非"健康 1.0×"，1C6O 才接近 1.0（1.07）。删除较配体轻但真实存在。
2. 蛋白删除空间均匀（pocket≈surface≈core），表面删除被 floor(0.5) 兜住（surface 保留多 0.5-0.93）；个别蛋白（1AZM 0.49）已到配体档。
3. 配体 v14 全部 10 蛋白删除（0.43-0.69），**删除越靠配体/越埋藏越重**（低 frac_sasa AUC 0.83、d_lig AUC 0.65）；核酸结合界面携带大量带电残基被定向删。
4. fixbinding 实验：口袋 100% 保留后，surface/core 删除不动（0.60/0.37）→ 删除主体是**全局默认分布**，不是口袋逃逸独有 → A1 扩大覆盖不能根治。
5. largen：n=200 下"电荷+不删+无聚集"三达标序列存在但仅 5.2%（删除通过率 17.5% ≪ 电荷 32.1%）→ 模型默认分布强偏删除。

## 4. 写文档
- 时间线 → `analysis/report/2026-09-05_taskA_timeline.md`
- 主报告 → `analysis/report/2026-09-05_protein_history_vs_ligand_deletion.md`
- 本 session

## 5. 结论摘要（交付）
- 前提**部分修正**：蛋白 v12.2 表面可控为真（多数表面保留 0.6-0.93），但"整体不删/健康"为误读（8/10 native 臂 0.49-0.77，仅 1C6O≈1.0；1A65 1.2/1.27 是过度添加特例）。
- 核心机制：删减捷径（成对删 D+K 保净电荷）是两模式共有；蛋白模式被"表面 floor+无配体疏水先验+埋藏带电池小"挡住 → 均匀、较轻；配体模式因"深口袋 frac_sasa 盲区 × 配体疏水先验 × 微调放大 × 结合面=带电富矿" → 定向且更深；A1-global 软约束权重小且归一化把口袋深删平均掉 → 未根治。
