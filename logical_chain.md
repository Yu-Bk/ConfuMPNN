# ConfuMPNN 工程文件分类规则（2026-09-05 修订版）

> 唯一分类规则。历史版：初版要求 code/ 内 input|output|log 三件套，项目演进后实际已迁移到
> 顶层目录体系（见下）。本文件与 `index/FILE_MANAGEMENT.md` 保持一致（后者为准引用副本）。
> 归档/备份/提交唯一规则见 `docs/MIGRATION_GIT_POLICY.md`；git 忽略见 `.gitignore`。

## 顶层目录（当前事实标准）
| 目录 | 放什么 |
|---|---|
| `code/` | 本项目全部代码：`src/`(核心模块)、`configs/`、`tests/`(验证/打分手稿仓库；`ligand_v9/` 配体线全套)、`input/`(示例 PDB)、`train_finetune.py`、`run_guided.py`。`code/output`、`code/log` 为 **v1–v6/早期遗留**，保留勿新增 |
| `output/` | 训练与采样/验证产物。**顶层 `*.json` = 论文关键数字（入库）**；`finetune_*/`(权重，gitignore→Release/NAS)、`generalization_*/`、`tm_sol_*/`、`propka_*/` 等重型子目录整体 gitignore（→NAS/网盘） |
| `log/` | 训练/验证/诊断 stdout 全量（入库） |
| `data/` | 输入大数据（CATH、ligand_train、validation_pdbs、ribosome_7k00、各 labels npz）。整体 gitignore，仅 `README.md`/`SHA256SUMS.txt`/validation manifest json 入库（→NAS 备份） |
| `analysis/` | 实验分析。`report/`=最新权威报告（**只留最新，被替代/证伪的移入 `archieved/`**）；`accident/{report,root}`=意外实验与根因；`ablation/`=旧消融（现消融走顶层 `ablation/`） |
| `ablation/` | **受控消融实验**（2026-09-05 起顶层，勿与 `analysis/ablation` 混）：`plan.md / data / runs / report / figure` |
| `compare/` | **版本/方法间对比实验登记**（2026-09-05 起）：README + 对比结果 |
| `figure/` | **论文所有图的计划 `plan_01.md` 与成品图** |
| `index/` | 项目文档定位索引（`DOCUMENT_INDEX.md`）、文件规则（本规则引用副本 `FILE_MANAGEMENT.md`）、宏观计划/决策/判据（`PROJECT_*.md`、`DESIGN_CRITERIA.md`、`RESULTS_MANIFEST.md`）、修复包 `v10_repair/` |
| `session/` | Claude Code 会话概览/决策日志 |
| `literature/` | 论文笔记：`baseline/ innovation/ pattern/ tools/ phenomena/` |
| `source/` | 论文开源源码链接登记（实体 clone 在顶层，见下） |
| `docs/` | 技术/配置/使用/新机部署文档 + `MIGRATION_GIT_POLICY.md` |
| `weights_release/` | 已确认最终编码器暂存（Release 上传中转，含 SHA256+README） |
| 外部源码（gitignore） | `LigandMPNN/ MoMPNN/ foundry/ protein_sol_mcp/ TemBERTure/` |
| 根文档 | `README.md`、`WORKFLOW_GUIDE.md`、`logical_chain.md`、`CLAUDE.md`、`LICENSE` |

## 细则
1. **新增任何文档 → 同步 `index/DOCUMENT_INDEX.md`；新增/移动/删除文件 → 同步本规则与索引。**
2. **结果/报告归属不定时**：结论性 markdown → `analysis/report/`；计划/决策/判据 → `index/`；会话过程 → `session/`；拿不准先 `index/` 并登记。
3. **大文件去向**：权重 `*.pt/*.ckpt`、`data/`、output 重型子目录、采样序列等 → 不入 git，按 `MIGRATION_GIT_POLICY.md` 走 Release/NAS/网盘；git 里保留**复现脚本**与**论文关键 JSON**。
4. 意外/证伪实验与"被替代结论"：在 `analysis/report/` 保留指向即可，过时主体移 `analysis/archieved/` 或注明。
