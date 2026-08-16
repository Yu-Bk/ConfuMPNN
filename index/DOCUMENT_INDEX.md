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

## Phase 1 代码模块（`code/`）

| 文档 | 位置 | 说明 |
|------|------|------|
| pKa 常量表 | `code/src/pka.py` | 侧链/末端 pKa、AA 索引、带电类型 |
| 可微电荷计算 | `code/src/differentiable_charge.py` | HH 方程平滑近似：`net_charge`（字符串）/ `net_charge_from_logits`（可微） |
| pI 查找器 | `code/src/isoelectric_point.py` | `find_pI` 二分搜索（验证用） |
| 结构感知过滤器 | `code/src/structure_aware_filter.py` | 4 条规则 → [L,21] bias；`load_preset` 读 YAML |
| 条件编码器 | `code/src/condition_embedding.py` | Soft Prompt MLP + mask-aware 条件向量（Phase 2） |
| 复合损失 | `code/src/losses.py` | CE + 电荷偏差 + 结构惩罚 + DPO + margin（Phase 2） |
| 引导采样器 | `code/src/guided_sampler.py` | 静态/动态 bias 解码，包装 LigandMPNN |
| 动态电荷前瞻 | `code/src/charge_lookahead.py` | 每步电荷 lookahead → 逐候选 bias；修复 target 被 softmax 抵消 bug |
| 一键运行入口 | `code/run_guided.py` | `--pdb --pH --target_charge --preset [--model_type]`，自动检测 LigandMPNN/MoMPNN 权重，输出 fasta+json+统计 |
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
| Phase 2 微调训练 | `code/train_finetune.py` | 冻结 MoMPNN + ConditionEncoder（cross-attention 注入 h_V）+ 复合损失 CE+电荷+KL 锚定；混合目标；每 epoch checkpoint+进度 |
| 条件注入采样 | `code/src/conditioned_sampler.py` | `inject_prompt`（cross-attention，训练/推理共用）+ `conditioned_sample` |
| Phase 3 pH 响应实验 | `code/tests/phase3_pH_response.py` | 4 PDB × target 响应 + pH 响应 + 跨 pH identity（固定 seed 分离条件影响） |
| Phase 3 防失控打分 | `code/tests/phase3_antidrift_score.sh` | 四指标打分管线（ESMFold/TM/%sol/TemBERTure）+ `phase3_score_status.sh` 进度查询 |

## 目录填充状态

| 目录 | 用途 | 当前内容 | 状态 |
|------|------|---------|------|
| `code/` | 实验模块代码 | `src/`（7 模块）+ `configs/`（2 yaml）+ `tests/`（2 脚本）+ `input/`（1BC8.pdb）+ `output/`（smoke 结果）+ `log/`（测试日志） | ✅ Phase 1 模块就绪 |
| `analysis/` | 实验结果分析 | `2026-08-16_mompnn_avail.md`（Stage E0 调研） | ✅ 已有首份报告 |
| `literature/` | 论文笔记 | 5 子目录骨架（baseline/innovation/pattern/tools/phenomena），均空 | ⬜ 待论文笔记导入 |
| `session/` | 会话记录 | `2026-08-15_phase1_modules.md`、`2026-08-16_charge_lookahead_fix.md`、`2026-08-16_PROJECT_STATUS.md`、`2026-08-16_e1_validation_design.md` | ✅ 快照已含 E1 完成 + 验证设计 |
| `source/` | 论文源码/链接 | （空） | ⬜ 待填充 |

## 约定

- 文档归属不定时，先按 `FILE_MANAGEMENT.md` 的目录规则判断，拿不准就放 `index/` 并在本表登记。
- 论文笔记导入时，按 `literature/` 五个维度（baseline/innovation/pattern/tools/phenomena）分类，并在此登记。
