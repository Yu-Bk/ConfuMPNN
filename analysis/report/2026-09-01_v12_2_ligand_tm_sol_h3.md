# v12.2 配体模式物化验证：Tm/Sol + H3 电荷合法性（2026-09-01）

> **结论（先说）**：配体模式物化验证**未通过**，且与删减捷径物理一致——
> **Tm 9/50 臂明显恶化（全部集中在负电臂 n2/n8，ΔTm −5~−9℃）**，对照 mompnn 蛋白线 0/50；
> **H3 电荷合法性 46/50（mompnn 48/50）**，共同失败点为 n8 极端负电臂。
> → 按决策框架（`session/2026-09-01_v12_2_ligand_tm_sol_and_h3.md §五`）走**路径 B：A1+A2 重训治删减**。
> **关联**：根因 `2026-09-01_v12_2_ligand_comp_analysis.md`；设计 `PROJECT_LOCAL_V12_2.md §7`；泛化 `2026-09-01_v12_2_ligand_validation.md`。

## 一、验证目标与流程

验证"配体微调引入的删减捷径"（负电臂删 D/E/K/R 调净电荷）的**物理后果**，分三块：

| 判据 | 内容 | 结果 |
|------|------|------|
| **S2（Tm/Sol）** | 各臂 vs 无条件基线：ΔTm<−5 或 Δ%sol<−10 视为明显恶化 | ❌ **Tm 9/50**；Sol 0/50 |
| **H3** | 条件臂结构过滤器违规率 ≤ max(native_ref, uncond)+5pp | ⚠️ **46/50**（ligand）/ 48/50（mompnn）|

**基线设计**（隔离"电荷条件化代价"）：
- 无条件基线 = 同模型同骨架、net_charge=训练均值 1.4243（配体上下文）
- native_ref = 同骨架 native 序列

**产物**：`output/tm_sol_ligand_v12_2/`（Tm csv 70 + sol txt 70）；汇总 `tm_sol_summary.json`；H3 `output/h3_ligand.json` / `output/h3_protein.json`。

## 二、Tm 结果：负电臂系统性热稳恶化

**S2 明显恶化 9/50 臂，8/9 为负电臂（n2/n8），正电臂 0 个。**

| 蛋白 | 恶化臂 | ΔTm（vs uncond）| 无条件基线 |
|------|--------|----------------|-----------|
| 5CQH | **n8 / n2 / native** | **−9.02 / −7.35 / −5.66** | 61.3 |
| 2FEO | n8 / n2 | −7.64 / −5.85 | 57.1 |
| 1AG0 | n8 | −6.68 | 63.0 |
| 1AS2 | n8 / n2 | −5.64 / −5.11 | 53.3 |
| 1C6O | n8 | −5.53 | 65.3 |

**对照**：mompnn 蛋白模式 v12.2（同 v12.2 权重、无配体上下文）**S2 0/50 无恶化**（`2026-08-31_v12_2_tm_sol.md`）。
配体模式的热稳恶化是**微调引入**（原始 LigandMPNN 无系统性删减，`comp_analysis.md` 第 4 层证据），
且**定向在负电臂** = 删减捷径发生处 → 物化证据与组成分析互相印证。

**机理解释**：负电臂靠删带电残基实现净电荷（删减捷径）。删除表面带电残基（D/E/K/R）破坏盐桥与表面极性
相互作用 → 热稳定性下降；净电荷要求越极端（n8），删除越多，Tm 掉得越多（5CQH n8 −9.0℃ 最极端）。

## 三、Sol 结果：无恶化

- 无任何臂 Δ%sol < −10。
- 反而多数负电臂 sol 大涨：1C6O n8 +21.8、5CQH n8 +15.8、2FEO n8 +16.4、1CGE n8 +16.5、1AG0 n8 +12.5。
  （删带电残基 → 组成变化，protein-sol 的 FCR 模型按组成预测，sol 上升不解读为"物理更好"，
  与 Tm 恶化同源的删减信号。）

## 四、H3 电荷合法性：46/50（ligand）/ 48/50（mompnn）

**方法**：structure_aware_filter 4 规则（charge_cluster R1 / salt_bridge R2 / core_charge R3 /
same_sign_cluster R4）**全量事后统计**（compute_bias 只统计未解码位置，不适用完整序列）。
带电集合 = pH7.4 强电荷 (K,R)/(D,E)。基线 = max(native_ref, uncond) + 5pp。

| 线 | 通过 | 失败臂 |
|----|------|--------|
| **mompnn 蛋白** | 48/50 | 1C6O/n8（0.297 vs 0.242）、1A65/n8（0.217 vs 0.207）|
| **ligand 配体** | 46/50 | 1C6O/n8（0.253 vs 0.242）、1A65/native（0.207 vs 0.207）、1A65/n2（0.209 vs 0.207）、1A65/n8（0.220 vs 0.207）|

**共同模式**：
- 失败臂集中在 **n8（极端负电）**——删减捷径的删电荷重排 → **R4 same_sign_cluster 大幅升高**
  （1C6O n8 R4=52.5 vs 其他臂 22-40；1A65 n8 R4=109 vs native 96.9）。
- 1A65 配体线基线过紧：uncond 违规率仅 0.076（长蛋白天然电荷稀疏），native_ref 0.157，
  基线 0.207——配体模式下 1A65 所有臂的电荷布局都比 native 密集，native 臂 0.2070 仅差 0.0003。
- R1（charge_cluster 10Å 同号≥阈值）在 n8 也系统性升高：1C6O n8 R1=6.1（mompnn）、
  1C6O n8 R1=1.6（ligand）、1A65 n8 R1=2.2-3.6 → 删减后同号残基更易聚集。

**H3 与 Tm 同源**：删减 → 电荷重排 → 同号聚集（H3 R1/R4 升）+ 盐桥/极性损失（Tm 降）。

## 五、结论与决策

1. **配体 Tm/Sol 物化验证未通过**（负电臂 Tm −5~−9℃ 系统性下降），**H3 也未全绿**（共同 n8 失败）。
2. 与删减捷径（监督逃逸 × 配体疏水先验 × 微调放大）的预测完全一致：**物理代价 = 极端负电臂
   热稳定性和电荷布局同时恶化**，且 mompnn 蛋白线（0/50 Tm）证明非模型/骨架固有。
3. **决策 → 路径 B**：执行 `PROJECT_LOCAL_V12_2.md §7` A1+A2 重训（~16h）：
   - 三块互斥分区（core 锁死 / pocket 温和改 / surface 温和改）解矛盾 bug；
   - A1 双向计数 `relu(N_p×0.7−gen)+relu(gen−N_p×1.3)`（ceil 1.3 防成对加）；
   - 净电荷锚 `target − q_core`；keep/free 开关（默认 keep 保护配体结合）；
   - 重训后全链复验：组成健康 → 响应 slope → 泛化 H1/H2 → **H3 + Tm/Sol 复测**。

**快速定稿路径 A 不适用**：Tm/Sol 已实测恶化，不能在论文中声称配体模式电荷控制"物理无害"。

## 六、复现命令

```bash
# Tm（confumpnn-temberture 环境）
python code/tests/temberture_score.py --input-dir output/tm_sol_ligand_v12_2/seqs
python code/tests/temberture_score.py --input-dir output/tm_sol_ligand_v12_2/uncond
# protein-sol（串行）
for fa in $(find output/generalization_ligand_v12_2/ligand -name seqs.fa); do
  python3 protein_sol_mcp/scripts/protein_sol_predict.py "$fa"
done
# 汇总（confumpnn 环境，修复 native_ref 行名读取 seed_only=False）
PYTHONPATH=code python code/tests/ligand_v9/v12_2_ligand_tm_sol_summarize.py
# H3 双线（修复：ref 骨架 resnum 全归一化为 4，不能 parse_PDB，按行序提取 CA）
PYTHONPATH=code python code/tests/h3_charge_legality.py \
  --gen-root output/generalization_ligand_v12_2/ligand --ref-root output/generalization_ligand_v12_2/ref \
  --native-root output/tm_sol_ligand_v12_2/ref_native --uncond-root output/tm_sol_ligand_v12_2/uncond \
  --pH 7.4 --out output/h3_ligand.json
```

## 七、脚本修复记录（2026-09-01）

| 脚本 | 修复 | 原因 |
|------|------|------|
| `v12_2_ligand_tm_sol_summarize.py` | `read_tm_csv(seed_only=False)` 读 native_ref | native_ref csv 行名 `<PDB>_native L=<L>` 非 `seed_` 开头，被排除 → ref_tm=None → 打印格式化崩溃 |
| `h3_charge_legality.py` | ① 加 `LigandMPNN` 到 sys.path；② 弃用 parse_PDB 改按行序提 CA | ① `data_utils.py` 在 LigandMPNN/ 下；② ref 骨架文件 resnum 全归一化为 4 → CA_dict 塌缩为 1 残基 → 越界 |
