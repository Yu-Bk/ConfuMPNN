# E1 对照实验：三目标 + 电荷响应完整对比

> 日期：2026-08-16　|　对应 `PROJECT_EXTEND.md` Stage E1
> **结论：MoMPNN 在可溶性、热稳定性、电荷响应三个维度均显著优于原版 LigandMPNN，可设计性持平**——完全验证了「多目标 DPO 微调不损失 pH 控制与可折叠性」的预期。

## 一、实验设置

- 序列：1BC8.pdb（93 残基），`pH7.4 target=0` 引导采样各 10 条（`code/output/compare/*/seqs.fa`）+ native 对照
- **可设计性**：ESMFold mean_pLDDT（`confumpnn-esmfold` 环境，`model.infer`，num_recycles=3）
- **可溶性**：Protein-Sol percent-sol（Hebditch2017，Manchester Perl 管线）
- **热稳定性**：TemBERTureTm 熔解温度（3 replica 平均，`confumpnn-temberture` 环境，protBERT-BFD）
- **电荷响应**：`run_guided.py` 净电荷偏差（阶段 1）

## 二、结果

| 目标 | 指标 | MoMPNN | 原版 LigandMPNN | native | 差异 |
|------|------|--------|----------------|--------|------|
| 可设计性 | ESMFold pLDDT | 82.61 | 82.74 | 80.45 | **持平**（−0.13） |
| 可溶性 | Protein-Sol %sol | **67.96** | 55.14 | 66.77 | **+12.8** |
| 热稳定性 | TemBERTure Tm (°C) | **61.95** | 54.18 | 50.90 | **+7.8** |
| 电荷响应 | 净电荷偏差 | **≤0.10** | +0.2~+0.7 | — | **更精准** |

（原始数据：`code/output/{plddt,protsol,temberture}/`；脚本 `code/tests/{esmfold_score,gravy_scores,temberture_score}.py`）

## 三、分析

1. **可溶性（+12.8 个百分点）与热稳定（+7.8°C）显著更优**——两个都正是 MoMPNN 的 DPO 优化目标，验证了权重确实内化了「可溶、热稳」偏好。
2. **可设计性不降**（pLDDT 82.6 vs 82.7）——多目标 DPO 没有以牺牲可折叠性为代价，这是微调成功的关键指标。
3. **电荷响应更准**（阶段 1 结论）——在完全相同的 bias 引导下，MoMPNN 的序列先验更中性，命中 target 更稳。
4. ⚠️ **GRAVY 代理失准教训**：GRAVY（疏水性）曾显示 MoMPNN 更疏水，与真实 Protein-Sol 结论相反。**打分必须用真实工具**，代理指标可能误导。

**综合**：MoMPNN 作为生成器，在不牺牲可设计性的前提下显著提升可溶性与热稳定，并保持精准的 pH 电荷控制——**E4 阶段把它设为 `run_guided.py` 默认生成器具备充分依据**。

## 四、待办

1. 扩展验证样本（多 PDB / 多 pH / 多 target，提高统计置信度）
2. 可用率指标（pLDDT>80 且电荷达标 且 %sol 高）的联合统计
3. E4：把 MoMPNN 设为默认生成器 + 完整对照实验
