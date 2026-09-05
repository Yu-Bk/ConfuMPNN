# ConfuMPNN — 文件管理规范（2026-09-05 修订）

> **来源**：项目根 `logical_chain.md`（唯一规则原版）。本文件为 index 引用副本，两者须同步。
> 归档/备份/提交规则见 `docs/MIGRATION_GIT_POLICY.md`；所有文档定位见 [[DOCUMENT_INDEX.md]]。
> ⚠️ 项目演进后实际目录体系已从"code/ 内 input|output|log"迁到**顶层**，本文件按实际修订。

## 目录总览（当前事实标准）
```
ConfuMPNN/
├── code/            # 本项目代码：src/ configs/ tests/(含 ligand_v9/配体线) input/(示例) run_guided.py train_finetune.py
│                     # code/output、code/log = v1–v6/早期遗留，保留勿新增
├── output/          # 训练与采样产物；顶层 *.json=论文关键数字(入库)；finetune_*/generalization_*/tm_sol_*/propka_* 等重型子目录 gitignore(→Release/NAS)
├── log/             # 训练/验证/诊断 stdout（入库）
├── data/            # 输入大数据(CATH/ligand_train/validation_pdbs/ribosome_7k00/labels npz)；gitignore，仅 README/SHA256/validation manifest 入库(→NAS)
├── analysis/        # 实验分析：report/=最新权威(被替代移 archieved/)；accident/{report,root}=意外+根因；ablation/ 旧消融
├── ablation/        # ★受控消融(09-05 起顶层)：plan/data/runs/report/figure
├── compare/         # ★版本/方法对比实验登记(09-05 起)
├── figure/          # ★论文全部图的计划 plan_01.md 与成品图
├── index/           # 文档定位索引 + 计划/决策/判据 + FILE_MANAGEMENT 引用副本 + v10_repair/
├── session/         # Claude Code 会话概览/决策日志
├── literature/      # baseline/ innovation/ pattern/ tools/ phenomena/
├── source/          # 论文源码链接登记（实体 clone 在顶层）
├── docs/            # 技术/配置/使用/新机部署 + MIGRATION_GIT_POLICY
├── weights_release/ # 已确认最终编码器暂存（Release 中转）
└── 根文档           # README / WORKFLOW_GUIDE / logical_chain / CLAUDE / LICENSE
外部源码(gitignore)：LigandMPNN/ MoMPNN/ foundry/ protein_sol_mcp/ TemBERTure/
```

## 规则详解
1. **`code/`**：实验模块代码 + 流程代码。不同模块/阶段建子目录；`tests/` 是"验证/打分手稿仓库"（真单测入口 `test_all.py`）。示例 PDB 放 `code/input`。
2. **`output/`（训练/验证产物主目录）**：训练 ckpt→`finetune_*/`（gitignore）；采样/折叠→`generalization_*/`、`tm_sol_*/`、`propka_*/`（gitignore）；**论文关键数字一律写成 `output/` 顶层 `*.json`**（自动入库）。
3. **`log/`**：训练/验证/诊断日志（入库），不复用 `code/log`（早期遗留）。
4. **`analysis/`**：`report/` 只放**最新权威报告**；被证伪/过时主体移 `archieved/`（注明原因）；意外实验+根因入 `accident/{report,root}`；真实消融放顶层 `ablation/`（`analysis/ablation` 留空即可）。
5. **`ablation/`**：受控消融（plan/data/runs/report/figure），勿与 `analysis/ablation` 混。
6. **`compare/`**：版本/方法对比（v13-vs-v14 等）方案与结果。
7. **`figure/`**：论文图唯一计划 `plan_01.md` + 全部成品图（每张图按编号登记数据源）。
8. **`index/`**：所有文档定位(`DOCUMENT_INDEX.md`)、关键决策/方向、宏观与论文规划(`PROJECT_*.md`)、判据(`DESIGN_CRITERIA.md`)、结果清单(`RESULTS_MANIFEST.md`)。
9. **`literature/ session/ source/ docs/`**：同 logical_chain 定义；`docs/` 里 `MIGRATION_GIT_POLICY.md` 为 git/Release/NAS 唯一规则。
10. **`data/` 大文件 & 权重 & 重型 output**：不入 git；提交=复现脚本 + 论文 JSON；备份见 `MIGRATION_GIT_POLICY.md`。

## 实验工作流（每次实验）
1. 开跑前在 `code/`（或 `ablation/`、`compare/`）定位脚本目录，明确归属模块。
2. 运行：脚本输出写 `output/` 对应子目录（顶层的写顶层 json），日志写 `log/`。
3. 跑完：**立即**在 `analysis/report/`（或 `ablation/report/`、`compare/`）写报告（结论+证据）；论文关键数字 json 放 `output/` 顶层。
4. 被证伪/过时：从 `report/` 移 `archieved/` 并注明。
5. 意外现象：记入 `analysis/accident/`（report+root+因果链）。
6. 读论文：按 literature 五维做笔记；源码登记 `source/`。
7. 重要对话：在 `session/` 写概要与决策。
8. 决策/方向/规划：更新 `index/` 对应文档。
9. **新增/移动/删除任何文档 → 同步 `index/DOCUMENT_INDEX.md` 与本规则。**
