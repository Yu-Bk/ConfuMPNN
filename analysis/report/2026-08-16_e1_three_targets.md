# E1 对照实验（第二阶段）：三目标打分（可设计性 + 可溶性已完成）

> 日期：2026-08-16　|　对应 `PROJECT_EXTEND.md` Stage E1「用 ESMFold pLDDT / Protein-Sol / TemBERTure 打分验证三目标」
> 状态：**可设计性 + 可溶性已完成**（ESMFold pLDDT + Protein-Sol）；**热稳（TemBERTure）权重下载中**（待补）。

## 一、数据来源与打分条件

- 序列：阶段 1 生成的 `pH7.4 target=0` 各 10 条（`code/output/compare/{mompnn,ligand}_pH7.4_t0/seqs.fa`）+ native 对照
- 可设计性：**ESMFold**（fair-esm 2.0.0 `esmfold_v1`，3.53B 参数，`confumpnn-esmfold` 环境，`model.infer` 的 `mean_plddt`，num_recycles=3）
- 可溶性：**Protein-Sol**（Hebditch2017 算法，Manchester Perl 管线 `protein_sol_mcp`，输出 percent-sol，越高越可溶）
- 可溶性代理（对比用）：GRAVY（Kyte-Doolittle 疏水性）
- 原始数据：`code/output/plddt/`、`code/output/protsol/`；脚本 `code/tests/{esmfold_score.py, gravy_scores.py}`

## 二、结果

| 组 | mean pLDDT（可设计性） | **Protein-Sol %sol**（可溶性） | mean GRAVY（代理） | n |
|----|----|----|----|----|
| **MoMPNN** | **82.61** | **67.96** | −0.327 | 10 |
| 原版 LigandMPNN | 82.74 | 55.14 | −0.453 | 10 |
| native | 80.45 | 66.77 | −0.434 | 1 |

（结合阶段 1：MoMPNN 电荷偏差 ≤0.10，原版 +0.2~+0.7）

## 三、分析

1. **可设计性（pLDDT）：两模型持平**（82.61 vs 82.74，差 0.13）。MoMPNN 的多目标 DPO 训练**未牺牲序列可折叠性**，且两者都显著高于 native（80.45）。
2. **可溶性（Protein-Sol %sol）：MoMPNN 显著更优**（**67.96 vs 55.14，+12.8 个百分点**），甚至略高于 native（66.77）。**完全符合其 Protein-Sol 优化目标**。
   - ⚠️ **GRAVY 代理被证伪**：GRAVY 显示 MoMPNN 略疏水（−0.33 vs −0.45），但真实 Protein-Sol 相反——证实 GRAVY（单特征疏水性）不能代表 Protein-Sol（35 特征）。教训：**打分必须用真实工具，不用 GRAVY 这类代理下结论**。
3. **电荷响应（阶段 1 结论）**：MoMPNN 显著更精准。

**当前三目标小结**：MoMPNN 在「可溶性」显著更优、「电荷响应」更准、「可设计性」不降（持平）——即**换 MoMPNN 作生成器，在不损失可设计性的前提下，序列更可溶、电荷控制更好**。

## 四、待办

1. **热稳（TemBERTure）**：`ibmm-unibe-ch/TemBERTure`（protBERT 权重下载中，需单独环境）——完成全部三目标
2. 扩展打分样本（多 pH / 多 target）与可用率指标（pLDDT>80 且电荷达标）
