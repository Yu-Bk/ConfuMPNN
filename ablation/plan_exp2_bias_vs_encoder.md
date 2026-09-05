# 消融 exp2 — condition_embedding 去留：bias-only vs 条件编码器（计划 2026-09-06）

> 归属：`ablation/`（受控消融）。状态：**执行中**（并入蛋白/配体 bundle：子代理 aeb37ce / ab59fdc）。
> 目的：隔离"学到的 ConditionEncoder"相对"简单电荷引导 logit bias（guided_sampler/charge_lookahead）"的增益，呼应 2025"全局 bias 管不住局部"论点。

## 1 实验
对 exp1 相同的测试蛋白（蛋白/配体模式各 3，含配体 RNA 结合"好"样本），在 native/n2/p2（可扩 n8/p8）臂：
- **route B**：encoder 条件注入（当前模型，对照）；
- **route C**：去掉 encoder → 仅引导 bias（guided_sampler 静态/动态 + charge_lookahead 电荷前瞻 bias）同 seed n≥200；
- 量 per-arm dev / H2 / **native 回收率** → B−C 增益 = encoder 相对纯 bias 的控制与保序增益。

## 2 边界
- 先确认 guided_sampler 对 MoMPNN 与 ligand_mpnn+配体上下文可跑通；不行用最小 bias 注入实现并说明。
- 若 C 某臂命中很差 → 记录并给原因（bias 是"推理侧每步修正"、encoder 是"训练学到的整体条件"的本质差异）。

## 3 输出
并入 `output/exp_control_{prot,lig}/`；报告含 exp2 小节；`ablation/report/2026-09-06_exp2_{prot,lig}_bias_vs_encoder.md`。

## 4 后处理
数据到位 → exp5 Wilcoxon（B vs C 与 B vs A 都配对检验）。
