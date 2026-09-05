# 对比实验 exp1 — 裸 backbone 无条件 vs ConfuMPNN 条件生成（计划 2026-09-06）

> 归属：`compare/`（版本/方法对比）。目标蛋白×route 见 §2。状态：**执行中**（蛋白 bundle GPU2 / 配体 bundle GPU6，子代理 aeb37ce 与 ab59fdc）。
> 关联图：`figure/plan_01.md` Fig7（baseline 对照）。

## 1 科学问题
把"条件电荷控制到底比**裸 backbone（MoMPNN/LigandMPNN 无条件重设计）**多控制了多少"量化成可上图对照的净增益（%达标 vs 天然电荷分布占比）。

## 2 设计（严格受控）
- **两种模式 × 两 backbone**：蛋白模式 = MoMPNN 裸 vs +v12.2 编码器；配体模式 = LigandMPNN 裸 vs +v14 编码器（atom25）。
- 每模式 **3 个测试蛋白（好/中/坏）**：蛋白按既有 H2/recovery 选；配体"好"用 **RNA 结合蛋白 5O60_E**。
- **route A 裸无条件**：backbone 直接采样 **n=1000**/蛋白重设计 → 逐条 pH7.4 净电荷，统计落在 ≈native(±1)、≈native−8/−2/+2/+8(±1 容差，对应 n8/n2/p2/p8 区) 的占比 = 天然基线。
- **route B 条件生成**：5 臂 native/n2/p2/n8/p8 各 n=1000 → per-arm |dev|≤2 达标率 + per-sequence 命中效率 + mean dev。
- **同 seed 体系**（同蛋白同臂 A/B 用同一批 seed 偏移）。
- **增益** = B 各臂达标率 − A 对应电荷区占比。
- 某臂 B 很差 → 附加小样本现场标定组；否则只用直接生成。
- 蛋白/配体各 6 组（3 蛋白×A/B），配体含 RNA 结合（"好"）。

## 3 指标与输出
mean 口径 H2 + per-sequence 命中效率 + dev；数据 `output/exp_control_{prot,lig}/`；报告 `analysis/report/2026-09-06_{prot,lig}_barebackbone_control.md`；session `session/2026-09-06_exp_{prot,lig}_control.md`。

## 4 后处理
完成后 → **exp5 Wilcoxon 配对检验**（见 `compare/plan_exp5_wilcoxon.md`）以 per-protein 配对。
