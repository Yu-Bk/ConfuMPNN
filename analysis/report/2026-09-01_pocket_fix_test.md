# 实验报告 — 口袋 fix 实测：配体口袋删减是否缓解（2026-09-01）

> **状态**：实测完成。**fix 100% 保住深部带电残基，口袋删减大幅缓解（2FEO -77%→-12%、1AXW -59%→-26%）；但删减全局性病态，fix 只堵深部出口，其余区域删减照旧**（全序列仍 0.66-0.68×）；2FEO 电荷 dev 恶化（1.4→4.0，校准失配）。结论：**fix=保深部功能残基的有效手段，非根治全局删减的手段**。
> **关联**：口袋工具 `code/tools/pocket_protect/define_pocket.py`（d4c239b）；机制报告 `2026-09-01_v12_2_ligand_comp_analysis.md`。

---

## 一、目的

实测"加口袋 fix"在生成流程中的效果：对比同一蛋白 native 臂（target=native 电荷）**无 fix** vs **有 fix**（fix 深部带电残基）的组成/口袋删减/电荷。回答：① 删减捷径是否被 fix 缓解？② 未覆盖残基是否仍被修改？

## 二、方法

- **生成**：`validate_generalization.py`（新加可选 `--fixed_residues`，默认 None 不影响 mompnn）+ 配体 v12.2 编码器 + 配体校准表，native 臂 n30，seed_base 2000，pH7.4
- **fix 列表**：`define_pocket.py` 输出的 `pocket_fix.txt`（深部带电 frac_sasa<0.25 + D/E/K/R），2FEO/1AXW 各 10 个、1C6O 4 个
- **对比**：`compare_fix_effect.py`（`code/tools/pocket_protect/`）——全序列/口袋/深部 fix 位点带电残基数倍率 + 电荷 dev + recovery
- 蛋白：2FEO（删减最重）、1AXW（删减最重，RNA 结合）、1C6O（组成健康对照）
- 产物：`output/pocket_fix_test/v12_2/` + `compare_fix_effect.json`

## 三、结果

| 蛋白 | 口径 | 全序列倍率 | 口袋删减 | charge dev | rec_pkt | 深部fix位点(native→gen) |
|------|------|:---:|:---:|:---:|:---:|:---:|
| 1C6O | 无fix | 1.01 | +57%增 | 2.1 | 0.569 | 4→2.0（0.5×）|
| 1C6O | **有fix** | 1.13 | +81%增 | **0.1** | **0.602** | 4→4.0（**1.0×**）|
| 1AXW | 无fix | 0.59 | −59%删 | 1.7 | 0.441 | 10→2.0（0.2×）|
| 1AXW | **有fix** | 0.66 | **−26%删** | **0.7** | **0.516** | 10→10.0（**1.0×**）|
| 2FEO | 无fix | 0.54 | −77%删 | 1.4 | 0.337 | 10→1.4（0.14×）|
| 2FEO | **有fix** | 0.68 | **−12%删** | ⚠️4.0 | **0.551** | 10→10.0（**1.0×**）|

## 四、结论

1. **fix 100% 保住深部带电残基**（三个蛋白深部位点全 1.0×）——直接目标完全达成
2. **口袋删减大幅缓解**：2FEO −77%→−12%（改善 65pp）、1AXW −59%→−26%（改善 33pp）；rec_pkt 提升（2FEO 0.337→0.551、1AXW 0.441→0.516）
3. **⛔ 更正：不是"删减转移"，是"删减全局性病态，fix 只堵一个出口"**（分区域分析 `analyze_deletion_transfer.py`）：
   | 2FEO | native带 | 无fix | 倍率 | **有fix** | 倍率 |
   |------|:---:|:---:|:---:|:---:|:---:|
   | 深部fix | 10 | 1.4 | 0.14 | 10.0 | **1.00** ✅ |
   | 口袋表面带 | 2 | 0.4 | 0.20 | 0.4 | 0.18 |
   | 口袋外表面 | 39 | 27.2 | 0.70 | 26.6 | 0.68 |
   | 口袋外深部 | 5 | 0.5 | 0.10 | 0.6 | 0.13 |
   **fix 前后，除深部 fix 位点外，其余区域删减倍率几乎不变**（口袋外表面 0.70→0.68、口袋外深部 0.10→0.13）——删减**本来就全局分布**（深部最狠 0.1-0.2×、表面 0.65-0.74×），fix 只保住深部 fix 出口（全序列增加全部来自此，+8.6），其余区域删减照旧。**表面并非"失去控制"——表面删减是 v12 配体模式固有的，fix 前后不变**。根治全局删减需训练侧组成监督（堵监督逃逸）
4. **电荷副作用（2FEO）**：dev 1.4→4.0 恶化——fix 缩窄可调残基空间 + **校准表（无fix拟合）对 fix 后响应失配**（2FEO 本为高方差蛋白）；1C6O/1AXW 电荷反而改善（2.1→0.1、1.7→0.7）。⚠️ 实际使用若 fix 口袋，建议**重新小样本标定**
5. 1C6O（健康对照）口袋带电**增加**（+57%增）→ 其配体先验是"口袋加电"非"删电"，与 2FEO/1AXW（删减）相反——删减方向与配体类型相关

## 五、工具完善点（用户问"是否可完善"）

1. ✅ **已修（level 判定缺陷）**：深部带电 + 强接触残基（2FEO A18/A132）曾因 level 优先"人工fix(强接触)"而不在默认 `pocket_fix.txt` → 漏保护。改为**深部带电优先**（无论是否强接触都进建议 fix），2FEO 深部 fix 8→10
2. **新增认识（实测暴露）**：
   - 删减**全局性存在**（非 fix 后转移）——深部 fix 外其余区域 fix 前后倍率不变 → 工具提示可加"只 fix 深部带电时，其余区域删减照旧"
   - fix 后 **2FEO 校准失配** → 使用建议：fix 口袋后现场小样本标定
   - 对比脚本"删减列"显示修正（>1 时标"+%增"）
   - 新增 `analyze_deletion_transfer.py`（分区域删减分析）
3. **validate_generalization.py 新增可选 `--fixed_residues`**（默认 None，不影响 mompnn 流程）——支持配体 fix 对比实验

## 六、复现

```bash
# 1. 生成带 fix 的 native 臂（per-protein pocket_fix.txt）
python code/tests/ligand_v9/validate_generalization.py \
  --manifest data/validation_pdbs/validation_manifest.json \
  --out_dir output/pocket_fix_test/v12_2 --mode ligand \
  --cond_encoder output/finetune_ligand_v12_2/finetune_epoch030.pt \
  --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --arms native --n 30 --device cuda:5 \
  --calibrate auto --calibration_file output/charge_calibration_v12_2_ligand.json \
  --fixed_residues "$(cat output/pocket_protect/2FEO/pocket_fix.txt)" \
  --start 4 --end 5
# 2. 对比
python code/tools/pocket_protect/compare_fix_effect.py --names 2FEO,1AXW,1C6O
```
