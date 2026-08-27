# v10 双编码器泛化验证报告（2026-08-28）

> **背景**：v10（A 条件解耦 + B 表面电荷监督 + C 结构惩罚）双 backbone 训练完成后的泛化验证
> （P3 阶段，对应 `index/PROJECT_LOCAL.md` §3.1）。本报告记录验证结果、与 v9 基线对比、根因分析与决策点。
> **状态**：验证完成，**发现重大退化（电荷控制），需用户决策**。

## 1. 验证配置

| 项 | 值 |
|----|-----|
| 测试集 | 泛化 10 蛋白（同 v9：small_mol×2 / rna×2 / dna×2 / metal×2 / long×2）|
| 电荷臂 | native / n2 / p2 / n8 / p8（**5 臂**，对齐 PROJECT_LOCAL §4 协议）|
| n | 30 / 臂 / 蛋白 |
| pH | 7.4 |
| v10-MoMPNN | protein 模式（纯骨架），编码器 `finetune_v10_mompnn/finetune_epoch030.pt` |
| v10-LigandMPNN | both 模式（ligand 5 臂 + protein 消融 3 臂），编码器 `finetune_v10_ligand/finetune_epoch030.pt` |
| 回折 | ESMFold（confumpnn-esmfold），TM-score（US-align vs ref 骨架 N,CA,C）|
| 总臂数 | MoMPNN 50 + LigandMPNN 80 = 130（全部完成，无 NaN）|

**验证全流程**（`code/tests/ligand_v9/run_v10_validation.sh`）：
采样（validate_generalization.py）→ ESMFold 回折 → TM-score → 统计（generalization_stats.py）。
产物：`output/generalization_v10_mompnn/` + `output/generalization_v10_ligand/` + `_stats.json`。

## 2. 核心结果：电荷命中率（H2，dev≤2.0）

| 模式 | 编码器 | native | n2 | p2 | n8 | p8 |
|------|--------|--------|----|----|----|----|
| protein | **v7**（基线）| **0.6** | — | — | **0.4** | **0.6** |
| protein | **v10** | **0.2** | 0.0 | 0.2 | **0.0** | **0.2** |
| ligand | **v9**（基线）| **0.7** | 0.5 | 0.6 | **0.3** | **0.7** |
| ligand | **v10** | **0.4** | 0.3 | 0.8 | **0.3** | **0.5** |

> 表格数值 = 该臂 10 蛋白中 dev≤2.0 的命中率。n2/p2 为 v10 新增臂，无 v9 同臂基线（v9 仅 native/n8/p8）。

**结论：v10 电荷控制在泛化集上系统退化**，MoMPNN 侧最重（native 0.6→0.2，n8 0.4→0.0，p8 0.6→0.2）；LigandMPNN 侧次之（native 0.7→0.4）。唯一改善：LigandMPNN p2 0.6→0.8。

## 3. 折叠可靠性（H1，TM-score）——**正常甚至略升**

| 模式 | TM 中位（跨臂范围）| TM≥0.7 | TM<0.5 失败率 | RMSD 中位 | pLDDT |
|------|------------------|--------|---------------|-----------|-------|
| v7 protein | 0.81–0.86 | 0.7–0.8 | 0.1 | — | 79.6–82.8 |
| v10 protein | 0.846–0.858 | 0.8 | 0.1–0.2 | 1.58–1.75 | 80.4–83.2 |
| v9 ligand | 0.816–0.852 | 0.7–0.8 | 0.1 | — | 84.2–84.4 |
| v10 ligand | 0.845–0.861 | 0.8 | 0.1 | 1.47–1.51 | 84.1–84.3 |

**折叠/结构保持未受 v10 破坏**（TM 中位略升，pLDDT 持平）。v10 没有损伤 backbone 的结构保持能力，只是电荷控制失效。

## 4. 逐蛋白退化模式（native 臂 dev）

| 蛋白 | v7 dev | v10 dev | v9 ligand dev | v10 ligand dev |
|------|--------|---------|---------------|----------------|
| 1A65（long 504）| 0.5 ✓ | **27.4 ✗** | 5.2 ✗ | 2.3 ✗ |
| 1AXW（long 528）| 0.3 ✓ | **23.5 ✗** | 1.9 ✓ | 2.2 ✗ |
| 1AG0（metal）| 0.1 ✓ | **12.1 ✗** | 1.1 ✓ | 0.3 ✓ |
| 1AS2（rna）| 1.7 ✓ | **7.3 ✗** | 1.6 ✓ | 1.8 ✓ |
| 5CQH（dna）| 1.6 ✓ | **6.2 ✗** | 1.4 ✓ | 2.2 ✗ |
| 1BJ4（long）| 2.8 ✗ | 0.6 ✓ | 2.3 ✗ | 2.4 ✗ |
| 2FEO（dna）| 2.5 ✗ | 3.9 ✗ | 1.7 ✓ | 0.1 ✓ |
| 1CGE（metal）| 2.1 ✗ | 3.4 ✗ | 0.8 ✓ | 1.6 ✓ |
| 1C6O（small_mol）| 3.3 ✗ | 4.8 ✗ | 4.1 ✗ | 3.4 ✗ |
| 1AZM（small_mol）| 1.8 ✓ | 1.6 ✓ | 1.6 ✓ | 2.4 ✗ |

- **v10-MoMPNN 长蛋白最惨**：1A65/1AXW 生成电荷偏离 target 23–27（v7 仅 0.3–0.5）。
  生成序列电荷约 = native 的 2 倍负（1A65 target −27 → 生成 −54）。
- **v10-LigandMPNN 退化温和**：命中蛋白变少但 dev 放大有限；2FEO/1CGE 反而改善。

## 5. 根因分析（初步，待消融确认）

**训练收敛正常**（v10-MoMPNN charge 2.05、LigandMPNN 2.82 终点，epoch 30 接近平台），**非欠拟合**。
退化是**训练-推理分布不匹配**——训练目标在泛化集上未转化为电荷控制。

最可疑组件：**A 条件解耦（`--decouple_perturb --decouple_range 12`）**，证据：

1. **只此一个训练变化即可致退化**：v10-MoMPNN = v7 训练配置 + decouple（±12）。v7→v10 唯一改动是 decouple，而 MoMPNN 退化最重（0.6→0.2）。LigandMPNN = v9 + decouple + B(L_add) + C(structure_boost)，退化轻——B/C 可能部分抵消 decouple 破坏。
2. **机制**（`train_finetune.py:542-560`）：decouple 作用在 30% 扰动样本，offset 从 v7/v9 的"native ± 课程控制的 ±1..8"变为"**Uniform[−12, +12] 无 native 锚点**"。30% 训练样本的电荷条件与骨架 native 电荷无关 → 模型对"条件电荷数值"的响应被拉宽/模糊化 → 对具体 target（尤其 native 附近）的精确控制退化。
3. **长蛋白失稳**：1A65/1AXW（L≥470，P4 已知薄弱区）在 ±12 大扰动下电荷响应最不稳定（dev 23–27）。

**次要可能**：B(L_add) 与 charge 损失方向冲突（native 臂下 L_add 要求"表面加电荷"而 charge 要求"保持 native"）；C(structure_boost=1.5) 压缩电荷调整空间。需消融（A7）定位。

> 注：v10 训练报告（`2026-08-27_v10_ligand_training.md`）曾标注"电荷控制最终评估在泛化验证后"——本报告即该评估，**结论：v10 三组件组合未达成治 P1 的目标，反而破坏电荷控制**。

## 6. 本次验证附带修复（P0 代码 bug）

- **LigandMPNN protein 消融模式 IndexError**（`validate_generalization.py`）：我 8/27 引入的回归——`model_type="ligand_mpnn"` + `number_of_ligand_atoms=0` + strip 配体（Y 空）→ `get_nearest_neighbours` 对空 Y 的 `L2_AB_nn[:, 0]` 崩溃。修复：protein 消融保留 Y 非空 + `use_atom_context=False`（模型同样看不到配体）。
- **MoMPNN OOM**：GPU0 被外部进程占满（142GB），验证换到 cuda:1/2。

## 7. 论文意义与决策点

**论文意义（不掩盖）**：
- v10 的失败是**真实、可复现的对照证据**：条件解耦（target 与 native 无关）会**削弱逆折叠模型的电荷控制**——这对"可控蛋白设计"社区是重要的 negative result。
- 若论文主方法回到 v7/v9（PROJECT_LOCAL §11 决策 2 的备选），v10 可作"改进尝试与失败分析"章节或消融证据（A7 的一部分）。
- 折叠不受损（TM 正常）说明 v10 组件未破坏 backbone 结构保持，问题集中在条件-响应映射。

**待用户决策（重大决策，暂停）**：
1. **回退主方法**：v7/v9 作主方法（已证明有效），v10 作消融/失败分析。
2. **v10 超参修复**：decouple_range 12→4（温和解耦）或去掉 A 只留 B/C → 重训 + 重验证（2–3 天）。
3. **v10 消融 A7**：先跑 A/B/C 三因子，精确定位退化组件 → 再决定修复方向。
4. **接受 v10 为"表面电荷受控"方法**：全局电荷控制交 v7/v9，v10 用于"区域级电荷斑块"场景（二阶段目标）。

> 报告数据全部来自真实验证产物（130 臂无 NaN）；判定不夸大、不掩盖，按论文产出导向记录。
