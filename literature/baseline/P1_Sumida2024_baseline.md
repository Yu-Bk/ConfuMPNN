# P1 — Sumida 2024 全景流程管线

**论文**: Improving Protein Expression, Stability, and Function with ProteinMPNN
**作者**: Kiera H. Sumida, David Baker 等（IPD, UW）
**出处**: J. Am. Chem. Soc. 2024, 146, 2054−2061 | DOI: 10.1021/jacs.3c10941

## 核心设计思想

天然蛋白在高表达、可溶性、热稳定性方面普遍不佳（进化优先优化功能而非稳定性）。本文提出一条**「零训练」的通用管线**：不修改 ProteinMPNN 权重，而是靠**「固定功能位点 + 结构信息过滤」**来保留功能、同时用 ProteinMPNN 重设计序列以提升物理性质。本质是把逆折叠模型当成「定向进化的替代品」，以大幅减少实验筛选量。

## 完整流程（端到端）

```
输入：天然蛋白的 3D 结构（晶体结构）
            │
  ① 定义「功能保留区」（fix positions）
            │
  ② （可选）骨架重塑：RoseTTAFold joint inpainting 重塑低保守 loop 区
            │
  ③ ProteinMPNN 序列设计（生成 N 条序列）
            │
  ④ AlphaFold2 单序列预测（无 MSA）→ 结构预测
            │
  ⑤ 过滤：pLDDT + Cα RMSD
            │
  ⑥ 湿实验：E. coli 表达 → IMAC/SEC 纯化 → 活性/稳定性表征
            │
输出：高表达、高稳定、功能保留（甚至提升）的设计变体
```

## 各模块原理

### ① 功能保留区定义（fix positions）

ProteinMPNN 是**纯结构驱动**的，无法感知功能信息。为保留功能，必须显式固定某些位点的氨基酸身份：

- **第一壳层功能位点**：定义在配体结合晶体结构中「距离底物/配体 7 Å 以内」的残基。这些残基的身份被固定（不被重设计），以保护催化机构与底物结合位点。
- （TEV 额外）**进化保守位点**：用序列比对（UniRef30）识别家族内高度保守的残基并固定。因为**远离活性位点的残基也可能通过变构/全局构象影响功能**（引用 Halabi et al. protein sectors）。论文试验了固定活性位点 + 前 30% / 50% / 70% 最保守残基四种设定。

### ② 骨架重塑（可选，仅 myoglobin）

- 目标是进一步稳定结构。选择**低保守的 loop 区**（globin 折叠的端部、围绕 heme 口袋的两个 loop、EF 螺旋间区）。
- 用 **RoseTTAFold joint inpainting** 重新生成这些区域的骨架。
- 在重塑后的骨架上再做 ProteinMPNN 序列设计（heme 结合位点仍固定）。

### ③ ProteinMPNN 序列设计

- 输入 backbone + fixed positions 信息，输出若干条「预测能折叠回该结构」的序列。
- myoglobin：生成 60 条；TEV：生成 144 条。

### ④ AlphaFold2 预测

- 用 **单序列预测（无 MSA）** 评估设计序列能否折叠回目标结构（independent of 进化信息，更严格）。

### ⑤ 过滤标准

过滤指标（在结构预测指标上）：
- myoglobin：`pLDDT > 85.0` 且 `Cα RMSD < 1.0 Å`（对照组：原生序列单序列预测 pLDDT=50.6、Cα RMSD=7.5 Å，说明原生序列其实折叠预测差）。
- TEV：`pLDDT > 87.5`（原生 TEV pLDDT=90）。

### ⑥ 湿实验表征

- 表达：E. coli → IMAC（固定金属亲和层析）+ SEC（尺寸排阻）纯化，测可溶产量。
- 稳定性：CD（圆二色）测 Tm；benchtop 稳定性测 30°C 孵育时间曲线。
- 功能：heme 光谱（Soret 峰/Q 带）验证 heme 结合；TEV 用香豆素肽底物测 kcat/Km，融合蛋白底物（MBP-TEVcs-FKBP-EGFP）测标签切除活性。

## 关键结果（baseline 的性能锚点）

- myoglobin：20 个设计全表达、全单体；13/20 产率高于原生（最高 4.1 倍）；8/8 测 Tm 的设计 Tm 均升高，6 个 95°C 仍折叠（原生 80°C 熔解）。
- TEV：144 设计中 134 可溶表达、129 高于原生表达（原生 1 mg/L，设计平均 20.1 mg/L）；最好的设计 kcat/Km 提升最高 26 倍；Tm 最高 84°C（比原生高 40°C）。

> **为什么这属于 baseline 而非 innovation**：本管线不引入新的训练方法或新的模型改动，而是「ProteinMPNN 直接应用 + 位点固定 + AF2 过滤」这一基础范式。它正是 P2/P3/P4 要在此之上做「对齐微调」的起点。