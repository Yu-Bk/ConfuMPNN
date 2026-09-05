# ConfuMPNN 全仓盘点 + 缺口/漏洞审计

> 日期：2026-09-05（只读盘点，未改动任何文件、未 git、未动数据）
> 范围：仓库 `/data/nfs/IC/baokun_yu/ConfuMPNN`（约 19G，其中 data 13G / output 5.3G / 外部源码 ~0.9G）
> 说明：所有路径均相对仓库根目录；大文件（PDB、ckpt、采样产物）只做目录级盘点，未读取内容。

---

## 一、清单盘点

### 1.1 顶层一览（大小排序）

| 顶层项 | 大小 | 内容性质 |
|---|---|---|
| `data/` | 13G | 训练/验证数据集（gitignore，仅 README/SHA256/validation manifest 入库） |
| `output/` | 5.3G | 训练权重 + 泛化采样 + 验证产物（子目录整体 ignore，仅顶层/部分结果 JSON 入库） |
| `TemBERTure/` | 518M | 外部源码（Tm 预测，含自有 data/） |
| `code/` | 480M | 本项目代码 + `code/output`(早期实验) + `code/log`(早期日志) |
| `foundry/` | 142M | 外部源码（PyRosetta/Foundry 系） |
| `ablation/` | 140M | 09-05 新增受控消融落地目录（plan/runs/data/report/figure） |
| `LigandMPNN/` | 137M | 外部源码（逆折叠 backbone，含 model_params 权重） |
| `MoMPNN/` | 99M | 外部源码（含 mompnn_paper_checkpoints） |
| `log/` | 25M | 训练/验证/诊断 stdout（344 个，全部 tracked） |
| `protein_sol_mcp/` | 3M | 外部 MCP 工具（Protein-Sol %sol） |
| `weights_release/` | 1.2M | v7/v9 编码器 + SHA256 + README（Release 暂存） |
| `analysis/` | 664K | 报告/证据/归档/事故目录 |
| `session/` | 368K | Claude Code 会话记录（37 个 md） |
| `index/` | 312K | 计划/决策/判据/文档索引 |
| `literature/` | 116K | 论文笔记（5 子目录） |
| `docs/` | 44K | TECH/CONFIG/USAGE/SETUP/MIGRATION |
| `source/` | 12K | 论文源码链接记录（README 单文件） |
| `figure/` | 12K | 论文图目录计划（`plan_01.md`，尚无成品图） |
| `compare/` | 8K | 对比实验登记（README 单文件，09-05 起） |
| 根文档 | — | README.md / WORKFLOW_GUIDE.md / logical_chain.md / CLAUDE.md / LICENSE |

### 1.2 `code/`（核心代码 vs 测试/工具）

- **核心训练/推理入口**（2 个顶层 .py）：`code/train_finetune.py`（57K）、`code/run_guided.py`（22K）。
- **核心模块** `code/src/`：13 个 .py（`pka / differentiable_charge / isoelectric_point / condition_embedding / conditioned_sampler / guided_sampler / structure_aware_filter / charge_lookahead / sasa / losses / v10_losses / v12_losses / __init__`）。
- **配置** `code/configs/`：5 个 yaml（`condition_defaults.yaml` + v3/v4/v5 备份 + `filter_presets.yaml`）。
- **`code/tests/`**：75 项（59 .py + 13 .sh + 1 .md + `ligand_v9/` 子目录）。**本质是"验证/打分手稿仓库"，不是纯单元测试**；真正可跑的单元测试入口是 `code/tests/test_all.py`。
  - `ligand_v9/` 子目录 33 项（24 .py + 8 .sh + 1 .md）：v9/v12/v13/v14 配体模式全套 build/validation/analyze 脚本（`build_calibration_ligand.py`、`build_rna_v14_labels.py`、`largen_search_v14.py`、`deletion_location_analysis.py`、`run_v13_in10_chain.sh`、`run_v14_ligand_validation.sh` 等）。
- **`code/tools/`**：`pocket_protect/`（`define_pocket.py` / `analyze_deletion_transfer.py` / `compare_fix_effect.py`）。
- `code/input/`（示例 PDB）、`code/output/`（**早期 v1–v6/phase3 实验产物，已过时保留**）、`code/log/`（早期日志，194 个 tracked）。

### 1.3 `analysis/`

- `analysis/report/`：**59 份报告**（全部 markdown，见 1.4 全表）。已含 archieved/accident/ablation 子目录：
  - `analysis/archieved/`：仅 `README.md`，**实际没有归档任何文档**。
  - `analysis/accident/`：`report/` 与 `root/` **均为空目录（0 文件）**。
  - `analysis/ablation/`：**空目录**（真实消融在顶层 `ablation/`，见 1.9，标准已偏离）。
- `analysis/evidence/`：5 个 JSON（ph_scan/transfer/generalization_v9 统计）。
- 根目录 2 个 md：`2026-08-16_mompnn_avail.md`、`artifacts_manifest.md`。

### 1.4 `analysis/report/` 全部报告（日期 | 标题）

> "★"＝当前最贴近"权威/最新"候选。同一天多份按内容链划分。

**08-16**：E1 pH 响应 / E1 三目标 / E1 扩展验证 / E4 默认 MoMPNN / Phase1 收尾 / Phase2 训练启动 / Phase3 pH 响应 / Phase3 防失控 / Phase3 n20 防失控 / Phase3 温度化根治
**08-17**：Phase3 S1 fix / Phase3 v2 复验 / Phase3 v3 占位符修复 / Phase3 v4 分层数据 / Phase3 v5 cap2 / Phase3 v7 课程学习
**08-18**：Phase3 v6 三类平衡 / v9 训练报告 / 模型电荷限制（使用指南）/ 配体结合能力 / 序列合理性+迁移 / 多 pH 温和区验证
**08-19**：v9 泛化验证报告
**08-27/28**：v10-LigandMPNN 训练 / ★v10 响应诊断（推翻外推假说）/ v10 泛化验证
**08-29**：★v11b/c 四版消融对比 / v12 训练（治删减成功但过度添加）/ ★电荷校准零重训修复验证
**08-30**：★v12.1 闭环+泛化验证（折叠 8/10、负向可靠）
**08-31**：v12.2 训练 / v12.2 diag / v12.2 Tm-Sol / ★v12.2 阶段性总结（完整验证链 + 校准三口径）
**09-01**：v12.2 配体迁移 diag / ★v12.2 配体删减机制解析 / 口袋 fix 实测 / v12.2 配体验证链 / v12.2 配体 Tm-Sol+H3
**09-02**：★v13（A1+A2）配体复验 / v14 RNA 数据+A1 全局化 / v12.3 diag / v12.3 响应弯曲 / AF3 折叠数据清单
**09-03**：v12.3 vs v12.2 蛋白最终对比 / ★验证重构+判据修正主报告 / long_neg_charge 归档（长蛋白限制）/ paper_gap1 v12.2 长蛋白
**09-04**：★v14 配体干净全链权威验证 / ★v13-in10 vs v14-clean 权威对照 / v14 删减定位(Task1) / v14 fixbinding(Task2) / v14 largen(Task3) / val_loss 曲线 / 论文子结论草稿
**09-05**：★7K00 核糖体可设计性测试 / ★TaskA 蛋白模式时间线 / ★蛋白 vs 配体删减机制 / ★TaskB 配体版本更迭 v13-vs-v14

**权威/最新候选**：`2026-09-04_v14_clean_validation.md`（v14 配体权威验证）、`2026-09-04_v13_in10_validation.md`（v13-vs-v14 同协议对照）、`2026-08-31_v12_2_summary.md`（蛋白 v12.2 完整链）、`2026-09-03_validation_standards.md`（判据）、`2026-09-05_{taskA_timeline,ligand_history_v13_v14,protein_history_vs_ligand_deletion}.md`（版本史）、`ablation/report/2026-09-05_ablation_{prot,lig}.md`（消融，注意在顶层 ablation/ 而非 analysis/）。

### 1.5 `index/`（16 项）

计划/方案类：`PROJECT_PLAN.md`(v1)、`PROJECT_EXTEND.md`(v2)、`PROJECT_LOCAL.md`(v3 论文导向主方案)、`PROJECT_LOCAL_P1_PLAN.md`、`PROJECT_LOCAL_V12_2.md`(当前蛋白最优交付)、`PROJECT_LOCAL_V14_DELETION_FIX_PLAN.md`、`PROJECT_LOCAL_V14_FINAL_EXPERIMENTS.md`(09-05 收尾)、`PROJECT_V9_GENERALIZATION_PLAN.md`、`PROJECT_V9_LIGAND_PLAN.md`、`PROJECT_SUPPLEMENT_H3_REVIEW.md`。
判据/索引类：`DESIGN_CRITERIA.md`(判据 v2)、`RESULTS_MANIFEST.md`(09-01 结果文件索引)、`FILE_MANAGEMENT.md`(分类唯一规则)、`DOCUMENT_INDEX.md`(文档定位索引)。
修复包：`v10_repair/`(7 项，含 `_adapters.py` 等)。
决策记录分散在 `PROJECT_LOCAL*.md` 与 `session/2026-09-04_decision_log.md`。

### 1.6 `literature/` / `session/` / `docs/` / `source/`

- `literature/`：README + 5 子目录。`baseline/`(P1–P4 各 1)、`innovation/`(P1–P4 + cross-paper)、`pattern/`(README 内含规律+工作量表)、`tools/`(README 内含工具表)、`phenomena/`(README)。根目录另有一份 `note_2025_global_bias_local_features.md`。
- `session/`：37 份 md。最近 10 份（新→旧）：`2026-09-05_ablation_run`、`2026-09-05_7k00_ribosome`、`2026-09-05_taskA_protein_history`、`2026-09-05_taskB_ligand_history`、`2026-09-04_v13_in10_chain`、`2026-09-04_task3_largen`、`2026-09-04_task2_fixbinding`、`2026-09-04_task1_deletion_location`、`2026-09-04_v14_clean_chain_autolog`、`2026-09-04_decision_log`。
- `docs/`：`CONFIG / TECH / USAGE / SETUP_NEW_MACHINE`（均停在 v9 节点 08-19）+ `MIGRATION_GIT_POLICY.md`(09-04 定稿，当前 git/归档唯一规则)。
- `source/`：README（P1–P4 源码链接与核实记录），无实体子目录（实体 clone 在顶层 MoMPNN/LigandMPNN 等）。

### 1.7 `data/`（各级子目录与作用，未读 PDB）

| 子目录 | 大小/文件数 | 作用 |
|---|---|---|
| `cath/` | 6.4G | 蛋白训练集：`S40/dompdb`(3.0G, 34,673 域)、`ext_basic_pdb`(2.3G)/`ext_basic_dompdb`、`ext_deepneg_raw`、各版本 labels npz |
| `ligand_train/` | 5.4G | 配体训练集：`all_pdb`(符号链接池)、`rna`/`rna_pdbs`/`rna_pdbs_ext`/`dna`/`small_mol`/`metal`/`water`/`all_pdb_pdb`、`v14_valset_pdb`/`v14_ext_valset_pdb`、`rna_complex_raw` 等原始下载；各版本 labels npz + manifest json |
| `ribosome_7k00/` | 18M | 7K00 核糖体 RNA 蛋白测试集（2 文件，labels/manifest） |
| `validation_pdbs/` | 456M | v9 泛化验证 10 蛋白 + noplig 版 + 候选缓存（871 文件） |
| `ligand_test/` | 3.2M | 迁移检验 5 复合物 |
| `transfer_test/` | 896K | 迁移测试 5 蛋白 |
| `validation_candidates/` | 23M | 选蛋白候选池 |
| 顶层文件 | — | `README.md`、`SHA256SUMS.txt` |

labels 版本全量（25 个 npz）：cath 侧 `labels_balanced(_v5/_v6/_v7).npz`、`labels_v12_2_train.npz`、`labels_v12_3_train.npz`、`labels_v12_3_valsupp(_a/_b).npz`、`labels_holdout_train.npz`；ligand 侧 `labels.npz`、`labels_orig_4972.npz`、`labels_rna_v14(_sup/_sup2).npz`、`labels_v14_merged/final.npz`、`labels_v14_ext_valset.npz`、`labels_v14_valset_805.npz`、各 smoke 版；另有 `ablation/data/labels_ablate_{prot,lig}.npz`。

### 1.8 `output/`（178 顶层项；下按内容类型分组）

- **顶层结果 JSON（89 个，论文关键数字，全部 tracked）**：`charge_calibration_*.json`(v7→v14 全)、`*_diag_response.json`、`v14_ligand_{diag,comp,gen_stats,deletion,fixbinding}_*.json`、`generalization_*_stats.json`、`h3_*.json`、`holdout_eval_v12_2.json`、`largen_v14_summary.json`、`tm_sol_*/tm_sol_summary.json`、`val_loss_curve_*.json`、`protein_vs_ligand_zone_deletion.json`、`_timing_v12_3.json` 等。
- **`finetune_*`（23 个训练目录）**：每目录含 `finetune_epochNNN.pt`（1–50 个）+ `condition_encoder_last.pt`；epoch 数见 1.11 表。
- **`generalization_*`（~30 个，共约 4G）**：每蛋白 × 电荷臂采样序列 + ESMFold 回折结构 + 打分（重型产物，gitignore）。
- **`tm_sol_*`（8 个，~4M 各）**：Tm/Sol 汇总 JSON。
- **`propka_*`（7 个）**：PROPKA 物理复核 per-residue JSON。
- **`ph_scan*` / `transfer*` / `largen_v14` / `fixbinding_v14` / `pocket_protect` / `pocket_fix_test` / `ribosome_7k00` / `paper_gap1_v122_long` / `ligand_*/` 小目录**：专项验证中间产物。

### 1.9 `ablation/`（顶层，09-05 新增；与 logical_chain 的 `analysis/ablation` 偏离）

- `plan.md`：收口版消融计划；`data/`：`build_ablate_subsets.py` + `subsets_balance.json` + 2 个 labels npz（**未 tracked**）；`runs/prot`（5 版：run_FULL/noph/nokeep/notarget/nov12comp × 10 epoch，5 个 probe json）、`runs/lig`（6 版：run_FULL/noph/nokeep/notarget/nov12comp/noA1 × 16 epoch，6 个 probe json）；`report/`：`2026-09-05_ablation_{prot,lig}.md` + `ablation_summary_{prot,lig}.json`；`figure/`：`ablation_{prot,lig}_figdata.json`。
- `compare/`：README 登记 v13-vs-v14 等 5 项对比指针。

### 1.10 外部源码简述

- `LigandMPNN/`：Dauparas 逆折叠 backbone（含 `model_params/ligandmpnn_v_32_010_25.pt`、openfold）。
- `MoMPNN/`：ICLR2026 ProtAlign 官方仓库（仅权重+inference，含 `mompnn_paper_checkpoints`）。
- `foundry/`：Foundry 系工具源码。
- `protein_sol_mcp/`：Protein-Sol MCP（%sol 打分）。
- `TemBERTure/`：Tm 预测工具（含自带 data，体积最大 518M）。

### 1.11 版本权重清单（`output/finetune_*` 最终件与 epoch）

| 目录 | 训练集语义 | epochs | 最终件 |
|---|---|---|---|
| finetune_v7 | MoMPNN/蛋白 | 30 | `condition_encoder_last.pt` |
| finetune_v10_mompnn / v11a_boff / v11b_afix / v11c_fullfix / v12 / v12_1 / v12_2 | 蛋白 | 30 | `condition_encoder_last.pt` |
| finetune_v12_3 | 蛋白+长蛋白扩充 | 40 | `condition_encoder_last.pt` |
| finetune_v10_ligand / finetune_ligand_v9 / v12_2 / v13 | 配体 | 30 | `condition_encoder_last.pt` |
| **finetune_ligand_v14_rna** | **配体 v14（RNA/A1 全局）** | **50** | **`finetune_epoch050.pt`（run_v14_ligand_validation.sh 引用）** |
| finetune_ligand_v14_dryrun[1-3] | 配体 dryrun | 1 | 仅 dryrun |
| *_smoke / *_test / *_dryrun | 冒烟 | 1 | 忽略 |

`weights_release/`：`condition_encoder_v7_last.pt` + `condition_encoder_v9_epoch030.pt` + `SHA256SUMS.txt` + README（Release preview1.0.0 暂存，**未更新到 v12.2/v14**）。

### 1.12 根文档定位与过时判断

| 文档 | 定位 | 覆盖度/过时判断 |
|---|---|---|
| `README.md` | 项目入口 | **过时**：标题区"当前进度"停在 v10 演进（08-27）；快速上手指向 `output/finetune_v7/...`（相对 `code/` 无效，正确是 `../output/...`）；未见 v12.2/v13/v14/ablation/ribosome |
| `WORKFLOW_GUIDE.md` | 权威新人指南 | **部分过时**：顶部"版本 v9 版（v10 演进中）"，§9 状态停 08-27；但 §7.6 已补到 v12.2（09-01）。缺 v12.3/v13/v14/消融/核糖体 |
| `logical_chain.md` | 分类唯一规则（根目录原始版） | 内容与 `index/FILE_MANAGEMENT.md` 一致，但**两者都未反映实际演化**（顶层 output/log/ablation/compare、code/output 事实废弃） |
| `CLAUDE.md` | 项目说明 | **过时**：当前状态标注 2026-08-31 v12.2 = 当前最优；缺 09-01~09-05 的 v13/v14 配体删减局限、ablation、核糖体 |
| `index/FILE_MANAGEMENT.md` | 文件分类唯一规则 | 见上（未反映新目录约定），内容还写 `LigandMPNN/、foundry/`，漏 MoMPNN/protein_sol_mcp/TemBERTure 已在 .gitignore |
| `index/DOCUMENT_INDEX.md` | 文档定位索引 | **严重过时**：最后更新 08-28；未登记 09-01~09-05 的约 20+ 份报告与 10+ session、PROJECT_LOCAL_V12_2/V14、顶层 ablation/compare、新增 docs/MIGRATION_GIT_POLICY 等；"report 23 份"等目录状态停留在 v9 节点 |
| `data/README.md` + `SHA256SUMS.txt` | 数据说明/校验 | **过时**：停在 v9 节点（08-19）；未覆盖 v12_2/v12_3/v14 labels、ribosome_7k00、v14_valset、rna_pdbs 扩充等新增数据；规模 8G→13G |
| `docs/{CONFIG,TECH,USAGE,SETUP_NEW_MACHINE}` | 技术/使用/新机配置 | **过时（v9 节点）**；仅 `docs/MIGRATION_GIT_POLICY.md`（09-04）为新 |
| `figure/plan_01.md` | 论文图目录唯一计划 | 新（09-05）；`figure/` 内**尚无成品图**，仅计划 |

### 1.13 未 git 跟踪 / 未入库清单

- `git status --porcelain` 仅 **2 个未跟踪文件**：`ablation/data/labels_ablate_lig.npz`、`ablation/data/labels_ablate_prot.npz`（非 .gitignore 命中，纯未 add）。
- `.gitignore` 忽略的一级模式：`LigandMPNN/` `foundry/` `MoMPNN/` `protein_sol_mcp/` `TemBERTure/`（克隆源码）；`__pycache__/ *.py[cod] *.egg-info/ .ipynb_checkpoints/`；`.venv/ env/`；`*.pt *.ckpt *.safetensors`（权重）；`code/output/`；`output/*`（**子目录整体忽略**）但放行 `!output/*.json`（顶层结果 JSON）；`results/`；`data/*`（放行 `!data/README.md !data/SHA256SUMS.txt !data/validation_pdbs/*.json`）；`.DS_Store .idea/ .vscode/`。`log/`、`code/log/` **不再忽略**（训练/验证日志入库，344 + 194 个 tracked）。
- **结论：论文关键数字 JSON 全部入库**。抽查结论：output 顶层 `*_clean / *_in10 / *ablation* / *fixbinding* / *largen* / val_loss_curve* / *deletion*` 相关 JSON 均 **tracked**（磁盘上 89 个顶层 JSON 全部 tracked；`git ls-files output/*.json` 共 188 条，含深层的 ribosome/propka/largen 等子目录汇总 JSON，亦已入库）。
- 未入库属设计（迁移策略见 `docs/MIGRATION_GIT_POLICY.md`）：`output/*/` 采样/折叠重型产物、`*.pt` 权重（走 Release/NAS）、`data/` 大文件（走 NAS）。
- `git ls-files` 全仓 **1097 文件**，.git 18M。

---

## 二、缺口/漏洞审计表

| # | 项 | 判定 | 证据与说明 |
|---|---|---|---|
| D1 | `data/SHA256SUMS.txt` / `data/README.md` 覆盖大文件 | **有，但过时** | 两者存在且覆盖 CATH/ligand_train/验证集 PDB；但停在 v9 节点：`labels_v12_2_train / labels_v12_3_train / labels_v14_{final,merged} / labels_rna_v14* / labels_v14_valset_805 / labels_v14_ext_valset` 及 ribosome_7k00、v14_valset_pdb、rna_pdbs_ext 等新增**均未登记校验和**；README 规模"约 8G"与实际 13G 不符 |
| D2 | labels 各版本可复现脚本 | **有** | v7 `build_labels_v2.py`、v9 `build_ligand_labels.py`、v12_3 `build_v12_3_augment.py`(+valsupp `build_valset_supp_a.py`)、v14 `build_rna_v14_labels.py` / `assemble_v14_805.py` / `select_v14_805.py` / `build_v14_ext_valset.py`、ablation `ablation/data/build_ablate_subsets.py`；但 SHA256 未覆盖 → 完全复现仍依赖 NAS 备份 |
| C1 | 训练/采样/回放/分析主脚本齐全 | **基本齐全** | `train_finetune.py`、`run_guided.py`、`src/*`、`tests/ligand_v9/*` 均在；v13/v14 链由 `run_v13_in10_chain.sh` / `run_v14_ligand_validation.sh` / val_loss 回放驱动。**缺口**：脚本间存在不存在的相对路径引用（见 C2），且多个 .sh 是"驱动链"而非可独立复跑 |
| C2 | 脚本引用路径与仓库实际一致性 | **发现 3 类失效/陈旧引用** | ① `validate_generalization.py` 默认 `--calibration_file output/charge_calibration.json`，**该文件不存在**（旧表；全仓 find 无此文件）；`run_guided.py` 默认已改 `output/charge_calibration_v12_2.json`（存在），但其 help 字符串仍写旧名。② README §三示例在 `code/` 下用 `output/finetune_v7/condition_encoder_last.pt` → 实际为 `code/output/finetune_v7`（**不存在**，正确为 `../output/...`）；③ `code/` 内 71 处硬编码 `/data/nfs/IC/baokun_yu/ConfuMPNN/...`（build_labels*.py 默认参数、多支 summarizer `ROOT=`、phase3 系列脚本），换机不可移植 |
| W1 | README 覆盖"如何跑/复现/文件结构/版本" | **有但过时** | 章节齐全，但版本状态/命令路径停在 v7-v10；无 v12.2 校准自动启用、v13/v14 局限、ablation 用法 |
| W2 | WORKFLOW_GUIDE 对照 CLAUDE.md"当前状态" | **过时** | WORKFLOW 头部 v9/§9 停 08-27（CLAUDE.md 08-31）；CLAUDE.md 本身亦过时（见 1.12） |
| W3 | `index/DOCUMENT_INDEX` 是否列全 analysis/report 与 session 最新件 | **否，严重过时** | 停 08-28；09-01~09-05 约 25+ 报告/session、PROJECT_LOCAL_V12_2/V14 系列、ablation/、compare/、docs/MIGRATION_GIT_POLICY 均未登记；违背其自述"新增任何文档必须同步" |
| V1 | finetune_* 各版本 ckpt / condition_encoder 齐全 | **齐全（本地）** | 23 个 finetune 目录均有 `condition_encoder_last.pt` + epoch 件（见 1.11）；当前配体引用件 `finetune_ligand_v14_rna/finetune_epoch050.pt` 存在；蛋白侧 v12.2/v12.3 存在 |
| V2 | `weights_release/` 内容 | **过时（仅 v7/v9）** | 08-19 后未更新；当前最优蛋白 v12.2/v12.3 与配体 v14 编码器只在 `output/finetune_*`（未走 Release/NAS）；按 MIGRATION 策略"确认最终件后才上传"，当前阶段可接受但属**待补** |
| M1 | 测试集/验证集是否混用 | **未见混用（依据文档）** | data/README 划分 v7→validation_pdbs、v9→ligand_test/transfer_test 均 --exclude 排除；v12.2 另做 hold-out + 无泄露(noleak) 口径（`generalization_v12_2_calib_noleak*`） |
| M2 | 校准是否跨集泄漏 | **未见明显泄漏，但口径需注意** | 校准三口径（per-protein 表内 / 小样本现场标定表外 / global 40-44%）已在 08-31 定稿；`build_calibration_small.py` 标定用的是表外蛋白自身的 50 条现场采样（"现场标定"属正常使用，非泄漏）；global 40-44% 被明示为固有上限 |
| M3 | A1/删减跨版本口径一致 | **口径不完全一致（已被项目自身指出并补做对照）** | v13(A1 keep+pocket) 与 v14(A1 global) 在**数据/轮次/验证集(in10 vs clean)** 都不同；TaskB 报告与 compare/README 明确"组成 v13 每蛋白删得更轻、v14 更深"，但归因需受控消融——09-05 已跑 ablation lig 的 noA1/keep 对照补此缺口（结果见 `ablation/report/2026-09-05_ablation_lig.md`）。v13-in10 vs v14-clean 的 "同协议"指流程而非同一验证集，横向对比时应注明 |
| M4 | figure/plan_01 中标注"待做"缺口的当前状态 | **部分已补、部分仍待做** | Fig22/23 RNA 核糖体：已做 **7K00 核糖体蛋白测试**（09-05，46 蛋白 native+n2+p2）；但 plan 中"RNA 结合核蛋白更广目标集"仍**待确认方案**。Fig25 消融：09-05 蛋白族+配体族已跑完（受控减预算）；但 plan 中额外消融点（λ_target 锚必要性、A1 keep vs global 受控同数据）已由 ablation run 覆盖 noph/nokeep/nov12comp/noA1。Fig26（校准+fix+大样本组合补救对比）、Fig27（AF3 取样对比）：**未做**。Fig24 湿实验：无数据。另外 `figure/` 下**没有任何成品图**（仅 plan + ablation figdata json），Fig1-27 全部待绘图 |
| M5 | 数据 manifest 入库情况 | **基本合规** | 10 蛋白 validation manifest、7K00 manifest/protein_labels/summary JSON 均已 `git add -f` 入库（git log 606e396） |

---

## 三、与 logical_chain.md 的分类差异

logical_chain.md 原始约定（= index/FILE_MANAGEMENT.md）要求 vs 仓库实际：

| 项 | logical_chain 要求 | 仓库实际 | 判定 |
|---|---|---|---|
| 输出目录 | `code/output` 放输出结果 | v7 起全部训练产物在**顶层 `output/`**；`code/output` 只剩 v1–v6/phase3 早期产物（过时保留） | **需移动/清理 + 建议更新标准** |
| 日志目录 | `code/log` 放测试脚本日志 | v7 起在**顶层 `log/`**（344 全量 tracked）+ `code/log`（早期 194）双份并存 | **需移动 + 建议更新标准** |
| 输入数据 | `code/input` 放输入 PDB | 示例/少量在 `code/input`；大规模训练/验证数据在**顶层 `data/`**（gitignore） | **已偏离，建议更新标准** |
| 分析报告 | `analysis/report` 最新报告 | 一致（59 份全在此） | **已一致** |
| 归档 | `analysis/archieved` 存过时方案 | 目录空（仅 README），过时文档仍留在 report/根目录未归档 | **需归档（待执行）** |
| 意外实验 | `analysis/accident/{report,root}` | 目录存在但**空**；实际意外分析写在 `analysis/report/*_diag/*_analysis.md` | **需清理/明确去向（或用起来）** |
| 消融 | `analysis/ablation/<实验>` | 09-05 起真实消融在**顶层 `ablation/`**（plan/runs/report/figure/data），`analysis/ablation` 空 | **需新建规则/更新标准** |
| 对比实验 | （未定义） | 新增顶层 `compare/`（09-05） | **建议补进标准** |
| 文档索引 | `index/DOCUMENT_INDEX` 列全部文档 | 存在但停 08-28，未登记最新 25+ 文档 | **需更新** |
| 决策记录 | `index/` 放关键决策 | 分散在 PROJECT_LOCAL_V12_2/V14 与 `session/2026-09-04_decision_log.md` | **部分一致，建议收敛** |
| literature / session / source | 5 子目录 / session 概览 / 源码链接 | 一致（source 仅 README，符合"登记"语义） | **已一致** |
| 外部克隆源码 | （未定义放法） | 顶层 `LigandMPNN/ MoMPNN/ foundry/ protein_sol_mcp/ TemBERTure/`，gitignore | **建议补进 FILE_MANAGEMENT（现 README 已说明）** |

**总体**：真实落盘约定已经从 logical_chain 的"code/ 内三件套（input/output/log）"迁移到"顶层 data/output/log + analysis/report + 顶层 ablation/compare"的新体系；`logical_chain.md` 与 `index/FILE_MANAGEMENT.md` 仍是旧文字 → **三处需同步更新**。

---

## 四、结论与建议

### 4.1 最关键缺口（Top 5）

1. **`index/DOCUMENT_INDEX.md` 严重过时（停 08-28）**，未登记 09-01~09-05 约 25+ 份报告/session 与 PROJECT_LOCAL_V12_2/V14、ablation/、compare/、docs/MIGRATION_GIT_POLICY——违背它自己定的"新增文档必须同步"规则，是文档体系里最影响检索的一环。
2. **根文档全线停在 v7–v12.2（README / CLAUDE.md / WORKFLOW_GUIDE 头部 / docs/TECH/CONFIG/USAGE/SETUP）**，未反映 v13/v14 配体删减局限、09-05 消融与核糖体结果、顶层目录新约定；README 还有指向不存在路径的示例命令。
3. **`data/SHA256SUMS.txt` + `data/README.md` 未覆盖 v12.3/v14 全部 labels 与新增数据目录**——论文关键数据（`labels_v14_final.npz` 等 20M 级文件）无校验和，NAS 备份唯一性未固化，迁移有静默损坏风险。
4. **`validate_generalization.py` 默认 `--calibration_file` 指向不存在的 `output/charge_calibration.json`**（run_guided 已改 v12.2，validate 未改）；代码内另有 71 处硬编码 `/data/nfs/...` 绝对路径，换机不可移植。
5. **`weights_release/` 停在 v7/v9**，当前最优（蛋白 v12.2/v12.3、配体 v14）最终编码器只存在于本地 `output/finetune_*`，尚未按 `MIGRATION_GIT_POLICY` 上传 Release/NAS（迁移单点风险）。

### 4.2 次要缺口/待办

- 2 个未 tracked 文件：`ablation/data/labels_ablate_{prot,lig}.npz`（应随 ablation 一起入库或登记进备份；重建脚本 `build_ablate_subsets.py` 已入库）。
- `analysis/archieved` / `analysis/accident/*` 空置：多份"被证伪/过时"报告（如 08-28 v10 系列中已被修正结论的报告）未按规则归档。
- `figure/` 无成品图、无绘图脚本；plan_01 图源 JSON 已齐（output 顶层），缺"画图动作"。
- README/workflow 对 `code/tests` 的定位（手稿仓库 vs 单元测试）没有说明；`test_all.py` 才是真单测。
- 论文数字虽然都入库，但**部分"结论性 JSON"存在于 output 深层目录靠 `git add -f` 强入**（如 ribosome/propka/largen），后续新人难以区分"哪些 JSON 权威"→ 建议以 `index/RESULTS_MANIFEST.md`（09-01）为准定期刷新。

### 4.3 建议整理动作清单（按优先级）

1. 更新 `index/DOCUMENT_INDEX.md`（补 09-01~09-05 全部文档；把 ablation/compare 纳入树）→ 顺带刷新 `index/RESULTS_MANIFEST.md`。
2. 同步修订 `logical_chain.md` 与 `index/FILE_MANAGEMENT.md` 的分类规则：顶层 `output/ log/ ablation/ compare/ data/` + `code/input` 仅示例；明确 `code/output`/`code/log` 为历史遗留（可整体移入 `analysis/archieved/` 或删除）。
3. 更新根文档状态：`CLAUDE.md` / `README.md` / `WORKFLOW_GUIDE.md` 头部版本与 §9 状态 → v13/v14/消融/核糖体；修正 README 里失效的示例路径（`../output/...`）。
4. `data/README.md` + `SHA256SUMS.txt` 增补 v12.3/v14/ribosome/valset 各 npz 与新增目录；确认 `labels_ablate_*.npz` 入库或纳入备份清单。
5. 代码小修：`validate_generalization.py` 默认校准表改为 `charge_calibration_v12_2.json`（或删除默认、强制显式传）；清理/参数化 71 处硬编码绝对路径（至少 build_labels*.py 默认值与 summarizer ROOT）。
6. `weights_release/`（或 Release）补当前最终件 v12.2/v12.3/v14 编码器，更新 SHA256 与 README；按 MIGRATION_GIT_POLICY 登记。
7. 归档动作：把已被替代/证伪的报告移入 `analysis/archieved/`（注明原因），使 `analysis/report/` 保持"最新权威"语义。

---

*审计方式：只读 ls/find/du/git/grep + 读取小体积 md/yaml；未读取任何 PDB/大 npz/ckpt 内容。仓库实际 git 仓库（main, HEAD=f0b1856, origin/main 同步），与任务环境备注"非 git 仓库"不符，已在盘点中按实际处理。*
