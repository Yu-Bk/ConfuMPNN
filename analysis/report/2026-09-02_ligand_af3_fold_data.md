# 配体模式生成序列的 AF3 二聚体/配体验证数据清单（2026-09-02）

> **目的**：v13 配体泛化验证里，3 个同源二聚体蛋白（1C6O/1AXW/1AG0）因 ESMFold 只折单链
> 无法评估"生成序列折叠后配体结合位点 / 二聚体界面是否保持"。改用 **AF3（蛋白+配体复合物
> 建模）手动验证**。本文件给出数据路径 + AF3 输入准备。
>
> 数据源：v13 配体泛化验证产物 `output/generalization_ligand_v13/`

## 一、产物路径（请完整保存 output/generalization_ligand_v13/）

```
output/generalization_ligand_v13/
├── ligand/<PDB>/pH7.4/arm_{native,n2,p2,n8,p8}/
│   ├── seqs.fa                     ← 生成序列（seed_* 50 条 + native 1 条，header 含 target/charge）
│   ├── folds/<name>.pdb            ← ESMFold 单链整链回折结构（51 个：1 native + 50 seed）
│   ├── tm.csv                      ← TM-score（对 ref 骨架；双链蛋白此值 ~0.50 无意义）
│   └── plddt.csv                   ← ESMFold pLDDT（置信度）
└── ref/<PDB>_ref.pdb               ← N/CA/C 骨架参考（resSeq 已连续化）
```

native 原结构（含配体 HETATM）：
```
data/validation_pdbs/{1C6O,1AXW,1AG0}.pdb
```

## 二、双链蛋白拆分信息（AF3 需按链建模）

| PDB | 类别 | native PDB 链（各链 CA 数）| parse_PDB 合并后 L | 配体 |
|-----|------|---------------------------|-------------------|------|
| 1C6O | small_mol | A=89, B=89 | 177 | 血红素 HEM（HETATM 182 行）|
| 1AXW | rna | A=265, B=265 | 528 | 核苷酸/RNA 片段（HETATM 300 行）|
| 1AG0 | metal | A=128, B=128 | 256 | 金属/配体（HETATM 91 行）|

⚠️ parse_PDB 把 A+B 两链按序**串联成一条序列**（1C6O native 前 89 残基 = 原 A 链序列，已验证一致）。
AF3 验证时**不要把 177/528/256 当单链**输入，应按 native PDB 的两链拆分。

## 三、建议验证目标蛋白 × 臂（重点）

删减捷径发生在**深部/口袋带电残基**，负电荷要求臂（n2/n8）删减最狠 → 优先验证：

| 蛋白 | 重点臂 | 理由 |
|------|--------|------|
| 1C6O | native, n2, n8 | 组成达标(1.04×)但 n8 有删减信号；TmΔu−3~−4 |
| 1AXW | native, n2, n8 | 组成 0.68×（删 32%）|
| 1AG0 | native, n2, n8 | 组成 0.69×（删 31%）|

对每个蛋白 × 重点臂，建议从 `arm_*/seqs.fa` 取 seed_ 序列（native 臂取接近 native 电荷的几条 + n8 臂取最深负电的几条）。

## 四、AF3 输入准备建议

1. **序列**：取 `seqs.fa` 的 seed_ 行；对双链蛋白按 §二 拆分位置切成两条链输入
2. **模板/配体**：native PDB（含 HETATM 配体）作参考；AF3 可带配体 SMILES/原子坐标
3. **判读指标**：
   - 配体原子 − 蛋白重原子最短距离（clash 检测 <2.5Å 冲突）
   - 结合位点残基（native 里 <8Å 的）在生成结构中是否仍近配体
   - 两单体在生成结构里的相对方位 vs native 二聚体（TM-score/界面 RMSD）
   - 与 ESMFold 单链折叠的对比（AF3 有配体上下文，应更能保口袋）
4. ⚠️ **局限提醒**：即使 AF3 显示配体放不进/界面变，也不代表"序列错误"本身——
   需区分"电荷调节造成的可接受构象变化"与"删减导致的结合位点破坏"。

## 五、相关产物（可追溯）

- 组成 `output/v13_ligand_comp.json`（全蛋白 D/E+K/R 计数 vs native）
- 校准表 `output/charge_calibration_v13_ligand.json`
- 统计 `output/v13_ligand_gen_stats.json`（H2/H1）
- PROPKA `output/propka_v13_ligand/`（物理电荷复核）
- 报告 `analysis/report/2026-09-02_v13_ligand_validation.md`
