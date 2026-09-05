# 对比实验 exp7 — 跨 pH(5/9) 的 pH 敏感性（计划 2026-09-06，GPU 空后执行）

> 归属：`compare/`（条件敏感性对照）。状态：**排队**（GPU2/6 被 exp1 bundle 占用，结束即跑）。
> 目的：证 v12.2/v14 **对 pH 本身敏感**（不只是对电荷），并为不同 pH 环境的设计提供边界。

## 两类（指令明确）
- **class A｜换 pH 看臂**：测试集蛋白在 pH=5 与 pH=9（+7.4 对照）下，各做 电荷臂 native/n2/p2/n8/p8 采样，量 dev/H2（"pH 变了 → 同样 target 序列能否仍被控住"）。
- **class B｜固定 target 换 pH**：固定同一电荷臂 target（如 native 或 0），把 pH 5→7.4→9，看生成序列的**实际净电荷是否随 pH 而变**（真 pH 敏感：同一条件向量的 pH 分量驱动不同序列/不同实测电荷）。若序列几乎不变 → pH 不敏感，重要负结果。

## 范围
- 蛋白（v12.2/MoMPNN）与配体（v14）两模式；蛋白各 3-4 个代表（碱/酸/中性 native pI），配体 in-10 各 3-4 个（含 RNA 结合）。
- **只算电荷类指标**（dev/达标/序列差异 identity），**不走全理化验证**；但对**极端 pH 代表蛋白**抽查 H1(ESMFold TM)、H3、H4(PROPKA) 确保没被破坏。
- n≈50-100/臂；校准按各模式（表内 per-protein，表外 global/小样本）。

## 输出
`output/exp_pH_{prot,lig}/` + `analysis/report/2026-09-06_pH_response_{prot,lig}.md`；接 exp5 配对（跨 pH 配对可选）。
