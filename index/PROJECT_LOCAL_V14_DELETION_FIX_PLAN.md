# v14 组成删减定位 & 结合位点保护 — 三任务计划（2026-09-04 用户批准）

> 目标：诊断 v14 系统性删减带电残基（0.43-0.69×）的**位置与特征**，并测试**局部保护（fix 结合残基）能否改善**。
> 理论锚：2025 MPNN-bias 工作"global biases cannot address local sequence features critical for binding/developability"
> （见 `literature/note_2025_global_bias_local_features.md`）——全局抑制表面电荷会连带破坏氢键/盐桥等局部特征。
> 前提结论：v13 产物只覆盖共享 5 单体(1AS2/2FEO/5CQH/1CGE/1BJ4)+边界1A65，**RNA/DNA 新成员无 v13 基线 → v14-vs-v13 对比天然不完整**。

## 公共输入（三任务共用）
- 测试集 in-10：`data/validation_pdbs/validation_manifest_v14_in.json`
- 权重/编码器：`output/finetune_ligand_v14_rna/finetune_epoch050.pt`；骨架 `LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`（atom25）
- 校准：`output/charge_calibration_v14_ligand_clean.json`（global slope 1.492/inter −1.260；per_protein 18）
- v14 clean 采样输出：`output/generalization_ligand_v14_clean/ligand/<pdb>/pH7.4/arm_{native,n2,p2,n8,p8}/seqs.fa`（unfix 基线）
- v13 采样输出：`output/generalization_ligand_v13/ligand/<pdb>/...`（仅 5 共享）
- 参考结构 ref：`output/generalization_ligand_v14_clean/ref/<pdb>_ref.pdb`；原始 `data/validation_pdbs/<pdb>.pdb`（配体 HETATM）

## 任务 1 删减定位分析（CPU，诊断，零新采样）
**问**：v14 删带电残基发生在哪（pocket/surface/core）？重删 vs 轻删区有什么简单特征？RNA/DNA 如何？v14-vs-v13 是否完整？
**法**：每残基算 距配体距离（Cα≤8Å=pocket）/ frac_sasa（≥0.25=surface 否则 core）/ SASA 分位 / 次结构(可选) →
  对 native-arm 生成序列 vs native，按区与 SASA 箱统计 D/E、K/R 保留率 → 定位删减热点与分界特征；
  RNA/DNA（21KL_A/5O60_E/3MXB_A/9DWG_L）单列；v13 在共享 5 同法算 → v14-vs-v13 区位对照（RNA/DNA 标注无基线）。
**GPU**：CPU。产物：`output/v14_deletion_location*.json` + `analysis/report/2026-09-04_v14_deletion_location.md`。

## 任务 2 结合残基 fix 重生成（GPU6，代码小改 + 采样）
**问**：把配体结合残基（Cα≤8Å，复用 pocket 口径）**固定 native** 后再生成，删减降多少？电荷命中/聚集怎么变？
**法**：新增采样驱动（尽量独立脚本、不改公共 sample 核心除非必要）支持"固定位置强制 native"的自回归约束 →
  in-10 全 10 蛋白 × 5 臂 × n40，per-protein 校准；**只测** 电荷 dev + 分区 D/E/K/R 删减（结合区必 native，验证约束生效）+ H3 电荷聚集；
  **不做 ESMFold/Tm/Sol**。对照 unfix(v14_clean)：删减总量/分布、H2 命中率、H3 变化。
**注意**：fix 减少可调自由度 → H2 可能变差；结果二义：fix 后仍重删→删减在非结合区也发生(纯 global 问题)；
  fix 后删减受抑→局部保护有效(支撑 binding-aware 监督)。产物 `output/fixbinding_v14/` + report。

## 任务 3 大样本搜索（GPU4，采样）
**问**：n 放大后是否存在"电荷达标 + 不重删 + 无电荷聚集"三者同时满足的序列？（判：模型从不生成 vs 稀有可采样救回）
**法**：in-10 全 10 × 5 臂 × **n200**；逐序列算 电荷 dev(校准后 target)、删减(相对 native)、H3 聚集违规
  → 统计"三达标"存在率 vs n、Pareto 前沿；逐蛋白/臂报告。
**产物** `output/largen_v14/` + report。（GPU4 与他人共享 → 变慢可接受，若过慢降 n 并注明）

## GPU 分配 & 并行
- Task1 CPU 子代理 A；Task2 GPU6 子代理 B；Task3 GPU4 子代理 C（用户批准 GPU6+GPU4 并行）。
- 子代理各自独立目录/session doc，**不 git push**（主会话统一归档）；不碰其它任务产物。

## 归档
每任务完成 → 主会话汇总报告 → git commit+push → 记入记忆。
