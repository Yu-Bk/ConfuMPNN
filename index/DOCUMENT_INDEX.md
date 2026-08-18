# ConfuMPNN — 文档索引

> 本文件是项目**所有文档的定位索引**（规则见 `FILE_MANAGEMENT.md` 第 3 条）。
> **⚠️ 新增 / 移动 / 删除任何文档后，必须同步更新本文件。**

## 当前项目文档

| 文档 | 位置 | 说明 | 最后更新 |
|------|------|------|---------|
| 项目计划（第一版） | `index/PROJECT_PLAN.md` | 完整技术计划（文献调研 / 分阶段实施 / 风险表 / 决策记录 / 输出清单） | 2026-08-16 |
| 项目拓展（第二版） | `index/PROJECT_EXTEND.md` | 多目标可开发性微调（可设计/热稳定/可溶）；MoMPNN 接入 → 自微调 → 集成回主线 | 2026-08-16 |
| 项目说明 | `CLAUDE.md`（项目根） | Claude Code 项目级说明：环境 / 常用命令 / 文件结构 / 下一步 | 2026-08-15 |
| 文件管理规范 | `index/FILE_MANAGEMENT.md` | 文件分类存放的唯一规则 | 2026-08-15 |
| 文档索引 | `index/DOCUMENT_INDEX.md` | 本文件 | 2026-08-15 |
| README | `README.md` | 完整入门：项目简介 / 从零搭建环境 / 快速上手 / 使用指南 / 输出解读 / 文档导航 | 2026-08-16 |
| **完整流程说明** | `WORKFLOW_GUIDE.md`（根目录） | **全项目整合汇报版**：构建计划与思路演变 / 核心公式参数 / 模块架构 / 数据流动 / 每轮训练结论 / 困难与解决（含文献）/ 复盘 / 下一步 | 2026-08-18 |
| MoMPNN 可用性调研 | `analysis/2026-08-16_mompnn_avail.md` | Stage E0 结论：权重=纯 backbone ProteinMPNN，`strict=True` 可加载 + 1BC8 前向跑通 | 2026-08-16 |
| E1 pH 响应对比 | `analysis/report/2026-08-16_e1_pH_response.md` | MoMPNN 电荷命中偏差≤0.10 vs 原版 +0.2~0.7 | 2026-08-16 |
| E1 三目标 | `analysis/report/2026-08-16_e1_three_targets.md` | 完整对比：可溶+12.8、热稳+7.8°C、电荷更准、pLDDT 持平 | 2026-08-16 |
| E1 验证扩展设计 | `session/2026-08-16_e1_validation_design.md` | 4 PDB×3pH×3target 混合设计；TM-score 主证据；位点固定；阈值防过拟合；CATH 4.2 数据 | 2026-08-16 |
| E1 扩展验证结果 | `analysis/report/2026-08-16_e1_extended.md` | 电荷 24/24 单调；MoMPNN 16/16 全优（4 指标×4 PDB）；可用率互有胜负 | 2026-08-16 |
| Phase 1 收尾 | `analysis/report/2026-08-16_phase1_examples.md` | 结构过滤器 99 分位阈值（CATH S40 统计）+ 示例蛋白对比；诚实边界：无引导时模型不感知 pH | 2026-08-16 |
| 技术文档 | `docs/TECH.md` | 架构 / 算法原理（HH 电荷、电荷前瞻、softmax 教训）/ 设计决策 / 验证摘要 | 2026-08-16 |
| 配置文档 | `docs/CONFIG.md` | filter_presets / condition_defaults / 命令行参数 / 环境 | 2026-08-16 |
| 使用说明 | `docs/USAGE.md` | 上手 / 5 种场景 / 输出解读 / FAQ / 批处理参考 | 2026-08-16 |
| E4 默认生成器 | `analysis/report/2026-08-16_e4_default_mompnn.md` | MoMPNN 设为 run_guided.py 默认生成器；实现/验证/文档同步 | 2026-08-16 |
| Phase 2 训练启动 | `analysis/report/2026-08-16_phase2_training_start.md` | 微调目标三层 / 冻结 backbone+KL 锚定防失控 / 混合目标 / 启动与查询 | 2026-08-16 |
| Phase 3 pH 响应 | `analysis/report/2026-08-16_phase3_pH_response.md` | 条件注入 Go/No-Go 4/4 PDB 通过（target 单调+跨 pH identity<100%）；校准增益 ~2.9× 机制 | 2026-08-16 |
| Phase 3 防失控 | `analysis/report/2026-08-16_phase3_antidrift.md` | 条件注入 vs E1b 基线四指标对比：pLDDT 掉是过冲所致（校准后 1BC8 82.3≈基线 82.8）；%sol/Tm 在噪声内，PASS | 2026-08-16 |
| Phase 3 温度化根治 | `analysis/report/2026-08-16_phase3_charge_temp.md` | 电荷损失温度化（charge_temp=0.5）：增益 2.57→1.04，无需推理侧校准；pH 感知保留 | 2026-08-16 |
| **判断标准 v2** | `index/DESIGN_CRITERIA.md` | 条件设计 PASS/FAIL 判据（v1→v2：S1 硬判降级为相似性软区间 0.4–0.7+防坍塌；对齐两真实目标；新增占位符语义 S3/位点固定 S4）| 2026-08-17 |
| **验证计划 v2** | `session/2026-08-17_validation_plan_v2.md` | 修正 S1/seq-keep 跑偏，对齐目标 1（天然骨架→全新序列+pI≈天然）与目标 2（人工骨架→全新序列+占位符）；identity 软约束三层方案（文献 P2 RCL/P3 坍塌）；扰动 ±1~8 + 占位符样本 | 2026-08-17 |
| Phase 3 防失控 n=20 | `analysis/report/2026-08-16_phase3_n20_antidrift.md` | n=20 对称配对检验推翻 n=5 假阴性（23/32 显著）；机制=条件注入 >50% 位点非保守替换；按判断标准 v1：H1✅/H2⚠️/S1❌ | 2026-08-16 |
| **Phase 3 S1 训练修正** | `analysis/report/2026-08-17_phase3_s1_fix.md` | 治 S1 部分成功：原生标签 70% + seq-keep 正则；H1 全达标优于上轮（折叠失败 0%、pLDDT 大幅修复）、H2 3/4 PDB、S1 0.45→0.67 未达 0.7、%sol 仍降（设计权衡） | 2026-08-17 |
| **Phase 3 v2 复验（第十五轮）** | `analysis/report/2026-08-17_phase3_v2_validation.md` | 对齐两目标：目标1（天然骨架+位点固定+pI≈天然）基本成功（H1 12/12、S4 100%、无坍塌）；目标2 负电 target 4/4 精确命中+折叠良好、**正电过冲（1BC8/2LZM）**、**占位符臂折叠全失败（S3 根因=训练偏负+无条件负漂移）** | 2026-08-17 |
| **Phase 3 v3 占位符修复（第十七轮）** | `analysis/report/2026-08-17_phase3_v3_placeholder_fix.md` | **占位符折叠完全修复**（均值占位+占位样本施加电荷损失 → t2_ph TM 0.89–0.97、全部 20 臂 H1 通过）；负电 4/4 保持；正电温和化后 1CRN/1UBQ 命中（1BC8/2LZM 仍过冲）；H3 正电违规 3/4→1/4 | 2026-08-17 |
| **Phase 3 v4 分层数据+逆加权（第十八轮）** | `analysis/report/2026-08-17_phase3_v4_balanced_data.md` | **1BC8 全 6 臂命中**（极端正电+17 dev 0.37）、2LZM 目标1 修复（0.35）、新正电验证 1b24A01 泛化 4/5；**但 1UBQ 退化（逆加权 cap=5 过头，牺牲中性蛋白）**——需降 cap 平衡 | 2026-08-17 |
| **Phase 3 v5 cap=2+扩大数据（第十九轮）** | `analysis/report/2026-08-17_phase3_v5_cap2_analysis.md` | **H1 折叠 30/30 全通过**、H2 16/25（v4 15/25）、S4 20/20；cap 5→2 修复 1CRN/2LZM 温和正电；**根因发现：分层「每箱 300」砍掉中性骨架多样性 97%（1UBQ 中性泛化失败的元凶）+ 箱8 高正电仅 76 域**——v6 改分层+过采样稀有类 | 2026-08-17 |
| **Phase 3 v6 三类平衡（第二十轮）** | `analysis/report/2026-08-18_phase3_v6_class_balance.md` | **1UBQ 大幅恢复（1/5→4/5，中性多样性 2500 验证根因）+ 极端正电普遍改善**（1CRN 2.98→0.78、2LZM 6.71→2.36）、H2 19/25、目标1 形态 t1_cond 5/5 全达标；⚠️ 折叠退化（H1 25/30，1CRN/1b24A01 部分臂失败率超标）+ 1BC8 正电退化（+15~+20 极端 target 仅 76 域仍不足） | 2026-08-18 |
| **Phase 3 v7 外部碱性+课程学习（第二十一轮）** | `analysis/report/2026-08-18_phase3_v7_curriculum.md` | **H2 20/25（+1）**：2LZM 极端正电根治（6.71→1.07）、温和正电全面达标、1UBQ 恢复保持 4/5；⚠️ 极端正电"两极化"（1BC8 5.45/1UBQ 2.54/1b24A01 4.58 小中性骨架仍过冲，**根因=碱性 target 与骨架类型未解耦**）+ **1CRN 折叠大幅退化（4/6→1/6，H1 22/30）** + 1b24A01 H2 退化（2/5）；外部碱性 678 域 + 课程学习（perturb 2.0→8.0）| 2026-08-18 |
| **模型电荷限制（使用指南）** | `analysis/report/2026-08-18_model_charge_limits.md` | 三区间：✅ 可靠 [native−8, native+2]（命中率 91-100%）、⚠️ 警告 (native+2, +5]（83%）、🔴 危险 >native+5（40%，小骨架过冲）；**绝大多数蛋白（目标≈天然）可靠**；负电外推比正电稳；补充蛋白规模限制（L≤300 单体可靠，大/多链失配）| 2026-08-18 |
| **配体结合能力验证** | `analysis/report/2026-08-18_ligand_binding_capacity.md` | **配体模式+条件化可用**（LigandMPNN 权重+编码器兼容，1FQG 电荷命中 dev 0.97）；⚠️ 条件化降低结合位点 recovery（0.62→0.41、0.71→0.58）；**缓解=固定结合位点残基**（已验证：口袋 100% 保持+电荷命中）；新增限制：大蛋白（L>400）/多链电荷失配（3T0F dev 21）| 2026-08-18 |
| **训练后验证：多 pH 复现天然（温和区）** | `analysis/report/2026-08-18_model_validation_phscan.md` | 3 蛋白（碱/酸/中性）× pH 5/7.4/9 × n=50，target=native 电荷：**折叠成功率 92-100%、失败率 0%（9/9 点全过）**；recovery 0.38-0.44（无条件基线 0.47-0.52，条件化低 ~0.1）；电荷 7/9 达标（1FQG L=263 大蛋白过冲=规模限制）；pH 5-9 温和区可靠折回原结构 | 2026-08-18 |

## Phase 1 代码模块（`code/`）

| 文档 | 位置 | 说明 |
|------|------|------|
| pKa 常量表 | `code/src/pka.py` | 侧链/末端 pKa、AA 索引、带电类型 |
| 可微电荷计算 | `code/src/differentiable_charge.py` | HH 方程平滑近似：`net_charge`（字符串）/ `net_charge_from_logits`（可微） |
| pI 查找器 | `code/src/isoelectric_point.py` | `find_pI` 二分搜索（验证用） |
| 结构感知过滤器 | `code/src/structure_aware_filter.py` | 4 条规则 → [L,21] bias；`load_preset` 读 YAML |
| 条件编码器 | `code/src/condition_embedding.py` | Soft Prompt MLP + mask-aware 条件向量（Phase 2） |
| 复合损失 | `code/src/losses.py` | CE + 电荷偏差 + 结构惩罚 + DPO + margin + `sequence_keep_loss`（序列保持正则，治 S1，Phase 2/3） |
| 引导采样器 | `code/src/guided_sampler.py` | 静态/动态 bias 解码，包装 LigandMPNN |
| 动态电荷前瞻 | `code/src/charge_lookahead.py` | 每步电荷 lookahead → 逐候选 bias；修复 target 被 softmax 抵消 bug |
| 一键运行入口 | `code/run_guided.py` | `--pdb --pH --target_charge --preset [--model_type]`，自动检测 LigandMPNN/MoMPNN 权重，输出 fasta+json+统计；**`--fixed_residues`** 位点固定（第十五轮新增，复用 LigandMPNN chain_mask 原生机制） |
| 过滤器预设 | `code/configs/filter_presets.yaml` | default / nucleic_acid_binding / membrane / acidic |
| 条件默认配置 | `code/configs/condition_defaults.yaml` | 条件向量/标准化/编码器参数（Phase 2） |
| 单元测试 | `code/tests/test_all.py` | 36 项，全通过 |
| 冒烟测试 | `code/tests/smoke_guided.py` | 真实 LigandMPNN + 1BC8.pdb，通过 |
| MoMPNN 兼容性测试 | `code/tests/mompnn_compat_test.py` | 8 权重 × 2 模式 load_state_dict（protein_mpnn 全 PASS） |
| MoMPNN 前向验证 | `code/tests/mompnn_forward_test.py` | MoMPNN 权重 + 1BC8.pdb 采样，seq_rec≈0.45 |
| E1b 扩展采样 | `code/tests/e1_extended.sh` | 4 PDB × (基线+9 条件) × 2 模型采样 |
| E1b 扩展打分 | `code/tests/e1_ext_score.sh` | ESMFold/TM/Protein-Sol/TemBERTure 批量驱动 |
| ESMFold 回折打分 | `code/tests/esmfold_score.py` | pLDDT + 回折存 PDB（--input-dir 批量） |
| TM-score 自洽 | `code/tests/tm_score.py` | US-align 批量：回折结构 vs 原骨架 |
| TemBERTure 打分 | `code/tests/temberture_score.py` | 3 replica 平均 Tm（--dirs-file 并行分组） |
| E1b 汇总/分析 | `code/tests/e1_ext_{summarize,analyze}.py` | 336 样本汇总 + 电荷/对比/可用率/留一分析 |
| 阈值统计 | `code/tests/threshold_stats.py` | CATH S40 采样统计 4 规则 99 分位阈值 |
| 示例蛋白对比 | `code/tests/examples_{compare,summarize}.sh/.py` | 4 蛋白 × 4 预设 + 3 pH 生成对比 |
| 分段并行下载 | `code/tests/parallel_download.py` | Range 分段下载大文件（绕过单连接限速） |
| 标签构建 | `code/tests/build_labels.py` | CATH S40 → 多 pH 条件标签（坐标+序列+pH+净电荷），算 μ/σ 写 condition_defaults.yaml |
| Phase 2 微调训练 | `code/train_finetune.py` | 冻结 MoMPNN + ConditionEncoder（cross-attention 注入 h_V）+ 复合损失 CE+电荷+KL 锚定+SeqKeep；混合目标（`--perturb_prob` 0.3=原生 70%，`--perturb_scale` 扰动幅度，第十五轮 4→8 治过冲）；**`--placeholder_prob`** 占位符样本（第十五轮新增，两种占位语义，目标 2）；每 epoch checkpoint+进度 |
| 条件注入采样 | `code/src/conditioned_sampler.py` | `inject_prompt`（cross-attention，训练/推理共用）+ `conditioned_sample` |
| Phase 3 pH 响应实验 | `code/tests/phase3_pH_response.py` | 4 PDB × target 响应 + pH 响应 + 跨 pH identity（固定 seed 分离条件影响） |
| Phase 3 防失控打分 | `code/tests/phase3_antidrift_score.sh` | 四指标打分管线（ESMFold/TM/%sol/TemBERTure）+ `phase3_score_status.sh` 进度查询 |
| n=20 扩样本采样 | `code/tests/phase3_antidrift_extend.py` | 对称配对采样（基线 vs 条件注入，双场景 A/B，固定 seed 协议），320 条 |
| n=20 四指标打分 | `code/tests/phase3_antidrift_n20_score.sh` | 递归扫描 16 臂的四指标打分管线 |
| n=20 配对统计 | `code/tests/phase3_antidrift_n20_stats.py` | 配对 t+Wilcoxon+BH-FDR；按判断标准输出 H1/H2/S1 判定 |

## 目录填充状态

| 目录 | 用途 | 当前内容 | 状态 |
|------|------|---------|------|
| `code/` | 实验模块代码 | `src/`（7 模块）+ `configs/`（2 yaml）+ `tests/`（2 脚本）+ `input/`（1BC8.pdb）+ `output/`（smoke 结果）+ `log/`（测试日志） | ✅ Phase 1 模块就绪 |
| `analysis/` | 实验结果分析 | `2026-08-16_mompnn_avail.md`（Stage E0 调研） | ✅ 已有首份报告 |
| `literature/` | 论文笔记 | 5 子目录骨架（baseline/innovation/pattern/tools/phenomena），均空 | ⬜ 待论文笔记导入 |
| `session/` | 会话记录 | `2026-08-15_phase1_modules.md`、`2026-08-16_charge_lookahead_fix.md`、`2026-08-16_PROJECT_STATUS.md`、`2026-08-16_e1_validation_design.md`、`2026-08-17_s1_training_fix.md` | ✅ 快照已含 E1 完成 + 验证设计 + 第十四轮 S1 修正 |
| `source/` | 论文源码/链接 | （空） | ⬜ 待填充 |

## 约定

- 文档归属不定时，先按 `FILE_MANAGEMENT.md` 的目录规则判断，拿不准就放 `index/` 并在本表登记。
- 论文笔记导入时，按 `literature/` 五个维度（baseline/innovation/pattern/tools/phenomena）分类，并在此登记。
