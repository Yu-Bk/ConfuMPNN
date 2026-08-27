# v9 泛化验证计划 — 未见蛋白的电荷控制 / 折叠 / 配体依赖全面检验

> 日期：2026-08-19
> 状态：**计划定稿（当时）**，执行中（v10 演进见 PROJECT_LOCAL.md）
> 背景：v9 训练成功（`2026-08-18_v9_ligand_training.md`），3 个验证蛋白（1MBN/4DFR/1FQG）电荷控制达标。
> 本计划扩展验证：① 泛化能力（10 个**未见蛋白**，覆盖 5 类配体）；② 极端电荷泛化（±8）；
> ③ 配体上下文贡献（有配体 vs 无配体消融）；④ 折叠与合理性（ESMFold + TM + 口袋保持）。
> 关联：`2026-08-18_model_charge_limits.md`（v7 电荷可靠范围，对比基准）、`index/DESIGN_CRITERIA.md` v2（判据）。

---

## 一、目标与验收标准

### 1.1 目标
回答四个问题：
1. **泛化**：v9 编码器在**训练/验证范围之外**的 10 个蛋白上，电荷控制（H2）是否保持？
2. **极端电荷**：target = native ± 8 时（v7 已知极限），v9 表现如何？负电外推是否保持、正电过冲是否改善？
3. **折叠**：生成序列能否折回原骨架（H1，ESMFold + TM）？
4. **配体依赖**：同一 backbone + 编码器，去掉配体原子上下文后效果下降多少？

### 1.2 验收标准（对齐 DESIGN_CRITERIA v2）

| 判据 | 阈值 | 评估方式 |
|------|------|---------|
| **H2 电荷命中** | dev ≤ 2.0（n=30 均值） | net_charge(seq, 7.4) vs target |
| **H1 折叠** | TM 中位 ≥ 0.70，失败率(TM<0.5) ≤ 10% | ESMFold 回折 → US-align vs native 骨架 |
| **合理性** | recovery、口袋 recovery、GRAVY、pLDDT | 报告（软监控）|
| **配体消融** | ligand vs protein 模式 H2 dev / H1 TM 对比 | 报告下降幅度 |

---

## 二、蛋白选择（10 个，防泄漏）

从 RCSB 单链蛋白 + 非聚合物配体池（分辨率≤3.0）随机采样 800 个 → 本地 HETATM 分类 →
**排除训练集 4972 复合物 + 验证蛋白 1mbn/4dfr/1fqg/5hvx/3t0f** → 逐类挑选。

> DNA 配体（游离脱氧核苷酸 DA/DT/DG/DC）在 RCSB 中极稀缺，专门用
> `rcsb_nonpolymer_entity_instance_container_identifiers.comp_id` 搜索（全网仅 2 个单链蛋白）。

| # | PDB | 类别 | L | native 电荷@7.4 | 配体 | 说明 |
|---|-----|------|-----|------|------|------|
| 1 | 1C6O | 小分子 | 177 | −14.31 | HEM 血红素 | 细胞色素类 |
| 2 | 1AZM | 小分子 | 258 | −1.71 | AZM 乙酰唑胺 | 碳酸酐酶-药物 |
| 3 | 1AS2 | RNA | 312 | −2.69 | GDP | G 蛋白类 |
| 4 | 1AXW | RNA | 528 | −18.30 | UMP | 尿苷酸酶，兼具长序列 |
| 5 | 2FEO | DNA | 221 | −6.88 | DC 脱氧胞苷 | DNA 核苷酸配体 |
| 6 | 5CQH | DNA | 183 | −5.53 | DC 脱氧胞苷 | DNA 核苷酸配体 |
| 7 | 1CGE | 金属 | 162 | −11.66 | CA + ZN | 金属配位 |
| 8 | 1AG0 | 金属 | 256 | −8.16 | CU | 氧化还原酶 |
| 9 | 1A65 | 长序列 | 504 | −26.85 | NAG 糖 | L 略超训练上限(500) |
| 10 | 1BJ4 | 长序列 | 470 | +0.42 | PLP 吡哆醛磷酸 | 转氨酶 |

- 全部经 `parse_PDB`（训练同款）解析成功、序列无 X、不在训练/验证集。
- 电荷覆盖 −26.9 ~ +0.4（天然负电为主），长度覆盖 162–528。

---

## 三、电荷臂设计（每蛋白 5 臂，pH 固定 7.4）

| 臂 | target | 类型 | 依据 |
|----|--------|------|------|
| a0 native | round(native) | 原生保持 | baseline |
| a1 n2 | native − 2 | 温和负电 | v7 可靠区 |
| a2 p2 | native + 2 | 温和正电 | v7 可靠区 |
| a3 n8 | native − 8 | **极端负电** | v7 可靠区下界（95% 命中）→ 测 v9 保持 |
| a4 p8 | native + 8 | **极端正电** | v7 危险区（40% 命中）→ 测 v9 是否改善 |

> **极端选 ±8 理由**（用户授权"根据经验定"）：`2026-08-18_model_charge_limits.md` 已用
> [native−8, native+8] 系统测过 v7，**±8 是同一把标尺**，v9 结果可直接对比；正电 +8 失败
> 也是"泛化边界"的有效信息。

---

## 四、配体消融设计

- **主实验（ligand 模式）**：`use_atom_context=True`（配体原子进上下文），全 5 臂 × n=30
- **消融（protein 模式）**：`use_atom_context=False`（忽略配体原子），跑代表性 3 臂（native/n8/p8）× n=30
- **同一 backbone（LigandMPNN 权重）+ 同一 v9 编码器**，只切换配体原子上下文开关
- 对比指标：H2 dev、H1 TM、recovery → 量化"配体信息"的贡献

## 五、评估维度

| 维度 | 指标 | 工具 |
|------|------|------|
| H2 电荷 | \|mean charge − target\|，dev ≤ 2.0 | net_charge |
| H1 折叠 | TM-score 中位 ≥0.70、失败率 ≤10% | ESMFold(esmfold_score.py) + US-align(tm_score.py) |
| 序列保持 | native recovery、口袋 recovery（配体原子 <8Å 残基） | 验证脚本内计算 |
| 理化 | GRAVY 疏水性、pLDDT | 验证脚本 + ESMFold |

---

## 六、脚本与路径（按 index/FILE_MANAGEMENT.md）

```
code/tests/ligand_v9/
  ├── pick_validation_pdbs.py       # 候选选择（RCSB + 分类 + 防泄漏）
  └── validate_generalization.py    # 泛化验证采样（多蛋白×多臂×双模式）
data/validation_pdbs/
  ├── {pdb}.pdb|cif                 # 候选结构（git 忽略）
  ├── candidates.json               # 候选分类清单
  └── validation_manifest.json      # 最终 10 蛋白清单
output/generalization_v9/
  ├── ref/{pdb}_ref.pdb             # 参考骨架（N,CA,C 提取，供 TM）
  ├── {ligand|protein}/{pdb}/pH7.4/arm{tag}/seqs.fa   # 生成序列
  ├── {ligand|protein}/{pdb}/validation.json          # 每蛋白统计
  └── .../plddt.csv + folds/        # ESMFold 回折产物
index/PROJECT_V9_GENERALIZATION_PLAN.md      # 本计划
analysis/report/2026-08-19_v9_generalization_validation.md  # 验证报告
```

## 七、执行步骤与计算量

| 步骤 | 内容 | 预估 |
|------|------|------|
| 1 | 候选选择 + 10 蛋白清单 | ✅ 完成 |
| 2 | 采样：10×(5+3) 臂 × 30 = 2400 序列（cuda:3） | ~2-3 h（后台）|
| 3 | ESMFold 回折 2400 条（confumpnn-esmfold） | ~4-8 h（后台）|
| 4 | US-align TM-score + 汇总统计 | ~30 min |
| 5 | 配体消融对比 + 验证报告 | ~1 h |

## 八、风险与缓解

| 风险 | 缓解 |
|------|------|
| 极端正电(+8)大量失败 | 预期内——这正是"泛化边界"，报告如实呈现，与 v7 对比 |
| 大蛋白(L>500)折叠耗时 | 后台分批跑；1AXW/1A65 较长，ESMFold 时间上浮 |
| 配体原子被忽略后特征错位 | protein 模式=同一 backbone 无配体特征，正是要测的下降 |
| 口袋 recovery 定义偏差 | 用配体原子<8Å 定义，与训练特征化口径一致 |

## 九、对比基准

- **v7 电荷可靠区**（`2026-08-18_model_charge_limits.md`）：native−8 ~ native+2 可靠(91-100%)，
  native+3~+5 警告(83%)，>native+5 危险(40%)——v9 目标：配体模式下保持负电、改善正电
- **v7 折叠**：1UBQ/2LZM/1BC8 各臂 TM 中位 0.95+（v7 在 MoMPNN 小蛋白折叠良好）
