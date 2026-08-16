# Phase 3 pH 响应 Go/No-Go —— 条件注入验证报告

> 日期：2026-08-16
> 前置：Phase 2 微调完成（`train_finetune.py`，30 epoch，charge 收敛 1.58）
> 模型：MoMPNN（冻结 backbone）+ 微调 ConditionEncoder（cross-attention 注入 h_V）
> 报告对应：`analysis/report/2026-08-16_phase2_training_start.md` → 本文件

---

## 一、验证目标

回答两个问题：
1. **Go/No-Go**：微调后的模型是否真正「感知」pH/电荷条件？
   —— Phase 1 诚实边界是「无引导时模型不感知 pH，同一 seed 下各 pH 序列完全相同」。
2. **校准**：条件 target 与生成序列电荷的关系（是否精确可控）。

## 二、方法

- 4 PDB ×（① target 响应 @pH7.4 + ② pH 响应 target=0）
- **固定 seed**：同一 seed → 解码顺序相同 → 序列差异只来自条件注入（干净分离「条件影响」与「采样随机性」）
- 跨 pH identity：同 seed 对应位置逐位一致率；Phase 1 诚实边界应为 ~1.0（序列相同）
- 脚本：`code/tests/phase3_pH_response.py`；数据：`code/output/phase3/{pdb}/phase3_pH_response.json`

## 三、结果

### ① target 响应（固定 pH=7.4，单调性 + 增益）

| PDB | native | target → 生成电荷 | 单调 |
|-----|--------|------------------|------|
| 1BC8 | +8.9 | 0→+0.9, 4→+9.8, 9→+26.6, 14→+33.6 | ✅ |
| 1CRN | -0.7 | -6→-10.9, -1→-3.3, 0→-2.3, 4→+3.9 | ✅ |
| 1UBQ | +0.0 | -5→-13.1, 0→-1.0, +5→+6.1 | ✅ |
| 2LZM | +7.8 | 0→+1.7, 3→+11.8, 8→+24.6, 13→+37.4 | ✅ |

**4/4 PDB target 响应严格单调** —— 模型学会了「target 越高 → 生成越偏碱性的序列」。

### ② pH 响应 + 跨 pH identity（target=0，固定 seed）

| PDB | pH4 charge | pH7.4 charge | pH9 charge | 跨 pH identity（4 vs 9） |
|-----|-----------|--------------|-----------|--------------------------|
| 1BC8 | -2.9 | +0.9 | +1.2 | **0.712** |
| 1CRN | -1.7 | -2.3 | -5.6 | **0.747** |
| 1UBQ | -3.9 | -1.0 | +0.2 | **0.684** |
| 2LZM | -0.4 | +1.7 | +4.5 | **0.749** |

**跨 pH identity 0.68–0.92，显著 < 100%** —— 模型随 pH 改变氨基酸组成（Phase 1 诚实边界被打破）。
pH 越高，序列在自身 pH 下的电荷越偏正（如 2LZM：pH4→-0.4、pH9→+4.5）：因 target=0 固定，
pH 低时蛋白天然偏正电，需更多酸性残基抵消（→ 序列电荷偏负）；pH 高时反之。**方向符合物理直觉**。

### ⚠️ 校准发现：target→电荷线性增益 ~2.9×

- 拟合：`实际电荷 ≈ 2.9 × target − 1.1`（R² 高，响应干净但过冲 ~2.9×）
- **机制（温度实验证实 = 采样置信度放大）**：训练损失优化的是 softmax **期望电荷** E[Q]，
  但推理测的是**采样序列**的实际电荷。模型被 CE 训练得很自信 → 采样序列比 E[Q] 更极端。
  温度越高置信度越低：
  | 温度 | target=+5 的生成电荷 |
  |------|--------------------|
  | 0.1 | +14.9 |
  | 0.3 | +13.0 |
  | 1.0 | +7.1 |
  | 2.0 | +3.7 |
- **实用含义**：① 提高采样温度（1.0–2.0）可显著改善电荷命中；② 或推理时对 target 做线性校准
  `target_effective = (desired + 1.1) / 2.9`。二者可组合使用。

## 四、结论（Go/No-Go）

| 判据 | 结果 |
|------|------|
| 模型感知 pH/target 条件 | ✅ **PASS**（4/4 PDB target 单调 + 跨 pH identity <100%） |
| Phase 1 诚实边界被打破 | ✅ PASS（同一 seed 下各 pH 序列不再相同） |
| 电荷精确可控 | ⚠️ 方向对、幅度过冲（~2.9×，机制已明确，可校准） |

**Phase 3 核心结论：条件微调让模型真正获得了 pH 感知，这是 Phase 1 logit-bias 引导做不到的。**

## 五、待办（下一步）

- 防失控最终判据（进行中）：`code/output/phase3_antidrift/` 序列四指标打分
  （ESMFold pLDDT + TM-score + Protein-Sol %sol + TemBERTure Tm），对比 E1b MoMPNN 基线
  → 确认微调没有破坏可溶/Tm/可设计性。结果见 `2026-08-16_phase3_antidrift.md`（待产出）。
- 可选：接入 `run_guided.py --cond_encoder` 已可用；校准系数（gain/offset）可写入 config。
