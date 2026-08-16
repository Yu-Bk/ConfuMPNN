# E1 对照实验（第二阶段）：三目标打分初步

> 日期：2026-08-16　|　对应 `PROJECT_EXTEND.md` Stage E1「用 ESMFold pLDDT / Protein-Sol / TemBERTure 打分验证三目标」
> 状态：**初步**——已完成可设计性（ESMFold pLDDT）与可溶性代理（GRAVY）；热稳（TemBERTure）与真实可溶性（Protein-Sol）待装。

## 一、数据来源与打分条件

- 序列：阶段 1 生成的 `pH7.4 target=0` 各 10 条（`code/output/compare/{mompnn,ligand}_pH7.4_t0/seqs.fa`）+ native 对照
- 可设计性：**ESMFold**（fair-esm 2.0.0 `esmfold_v1`，3.53B 参数，`confumpnn-esmfold` 环境，`model.infer` 的 `mean_plddt`，num_recycles=3）
- 可溶性代理：**GRAVY**（Kyte-Doolittle 疏水性，Biopython，越负越亲水）
- 原始数据：`code/output/plddt/{plddt_scores.csv, three_targets.csv, score_input.fa}`；脚本 `code/tests/{esmfold_score.py, gravy_scores.py}`

## 二、结果

| 组 | mean pLDDT（可设计性） | mean GRAVY（可溶性代理） | n |
|----|----|----|----|
| **MoMPNN** | **82.61** | −0.327 | 10 |
| 原版 LigandMPNN | 82.74 | −0.453 | 10 |
| native | 80.45 | −0.434 | 1 |

（结合阶段 1：MoMPNN 电荷偏差 ≤0.10，原版 +0.2~+0.7）

## 三、分析

1. **可设计性（pLDDT）：两模型持平**（82.61 vs 82.74，差 0.13）。MoMPNN 的多目标 DPO 训练**未牺牲序列可折叠性**，且两者都显著高于 native（80.45）——在 pH 电荷引导下均产出高 pLDDT 序列。
2. **可溶性代理（GRAVY）：MoMPNN 略疏水**（−0.327 vs −0.453）。**与「Protein-Sol 可溶优化」的预期相反**，需要谨慎解读：
   - GRAVY 只是疏水性单特征，Protein-Sol 用 **35 个特征**（含组成/电荷/长度等），不能直接等同；
   - 样本量小（10 条），且电荷引导（target=0）本身会改变残基组成；
   - 结论：**此差异不足以断言 MoMPNN 可溶性更差**，必须用真实 Protein-Sol 打分核实。
3. **电荷响应（阶段 1 结论）**：MoMPNN 显著更精准（这是本对照里最明确的优势）。

## 四、下一步

1. **安装真实打分器**（待用户确认）：TemBERTure（热稳，`ibmm-unibe-ch/TemBERTure`，基于 protBERT，需单独环境 + ~1GB 权重）；Protein-Sol（可溶，Hebditch2017 算法，需 Perl 管线）
2. 扩展打分样本（多 pH / 多 target，不只 target=0）
3. 汇总完整三目标对比 + 可用率（pLDDT>80 且电荷达标）指标
