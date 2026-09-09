# 会话记录：L11 设计与 design_bench 通用化（2026-09-09）

## 概要
在 ConfuMPNN 内新增 **L11design** 工作区（目标=增强 L11/uL11–RNA 结合，选 uL11 湿实验）。所有设计**固定链 I 位点 3,5,9,34,35,89,124,131,134,135(native)**。

## 关键决策
1. 模式绑定输入（用户 09-09 更正）：蛋白模式只用 `L11.pdb`、配体模式只用 `L11_RNA.pdb`；Exp1 由 8 组改 4 组×100。
2. Exp1/2 用 **global 校准**（L11 表外）→ 电荷过冲 +2~+5；结论=表外高 pI 蛋白需**现场小样本标定**。
3. **方法澄清**：现场标定=在 L11 骨架新采探针批（native±[8,4,0,4,8]×n10=50）拟合自身 slope；**不是**用"设计得好"的序列拟合。
4. Exp3(=重做 Exp2 全量, Bsmall)、Exp4(=重做 Exp1 全量, Bsmall, 每 pH 各拟合) 执行中（采样已齐，折叠/报告续跑）。
5. **design_bench 通用化**：`code/tools/design_bench/`（tests=打分手稿仓库、tools=可复用工具）：`per_protein_small_cal.py`、`auto_calib_guided.py`(--autofit 一键现场标定, opt-in 不改模型)、`design_group_runner.py`、`fold_scheduler.py`、`score_aggregator.py`+README；CPU 冒烟通过（1BC8 slope0.919→校准后 dev−0.14）；修掉 run_guided 不支持 `--num_ligand_atoms` 透传 bug。用法写入 README/WORKFLOW。
6. 文档/索引/部署核校已推送；删旧 v14-clean 检查点 cron；L11 自动检查点 `4bf73509`(每2h, 仅归档完成块)。

## Exp1/Exp2 结果要点（详 report_exp1/2.md）
- Exp1: 蛋白模式骨架保真>配体(TM0.65vs0.61)、配体 %sol~95>蛋白~79、Tm 无恶化、固定位 400/400 零错配、电荷两模式过冲(pH8 更甚)。
- Exp2: 电荷随 target +6.87→+12 单调(方向可控)但过冲 dev+2~+5、|dev|≤2 仅18-29%、固定位 2400/2400 零错配、pLDDT54-56/TM0.60-0.67/Tm 无恶化、配体 %sol~95。

## 产物
L11design/{PLAN,input,test(report/session/脚本),output,data,backup}；design_bench/；README/WORKFLOW 一键标定段。
（待补：Exp3/4 报告与对比、design_bench 若要用可再推广。）
