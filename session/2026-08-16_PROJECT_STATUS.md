# 项目进展快照 — ConfuMPNN

> 汇总性进展记录（最后一次更新 2026-08-16）。明天继续时从本文件开始恢复上下文。
> 细粒度会话记录：`session/2026-08-15_phase1_modules.md`、`session/2026-08-16_charge_lookahead_fix.md`、`session/2026-08-16_e1_validation_design.md`

---

## 今日总览（2026-08-16 一天成果）

| 里程碑 | 内容 | 提交 |
|--------|------|------|
| E1 对照实验 | MoMPNN 三目标显著优：可溶+12.8、热稳+7.8°C、电荷更准、pLDDT 持平 | `c644a6b` |
| E1 验证设计 | 混合实验设计 + TM-score 主证据 + CATH 数据选型（`e1_validation_design.md`） | `bfb785a` |
| E1b 扩展验证 | 4 PDB×3pH×3target=336 样本：电荷 24/24 单调、**MoMPNN 16/16 全优** | `f38bec6` |
| Phase 1 收尾 | 结构过滤器 99 分位阈值（CATH S40 统计）+ 示例蛋白对比 + 诚实边界 | `1df4b2f` |
| 文档交付 | `docs/TECH.md`、`CONFIG.md`、`USAGE.md` 三份详细文档 | `2bf144d` |
| E4 集成 | MoMPNN 设为 `run_guided.py` 默认生成器 | `900eab7` |
| README 重写 | 完整入门文档（从零搭建→上手→使用指南→FAQ） | `531bd92` |
| 微调训练启动 | `train_finetune.py` 就绪+冒烟通过+后台启动（冻结 MoMPNN+ConditionEncoder+KL 锚定防失控） | `1477a79` |
| 微调训练完成 | 30 epoch×999 域=14.7min；charge 5.16→1.58、ce 稳定 1.86、kl 稳定 0.15 | — |
| Phase 3 pH 响应 | 条件注入接入 run_guided.py + Go/No-Go 4/4 PDB 通过（target 单调 + 跨 pH identity<100%） | `4fccc1c` |
| Phase 3 防失控（n=5） | 条件注入 vs E1b 基线四指标：pLDDT 掉是过冲所致（校准后恢复）；%sol/Tm 噪声内 → PASS | `c2de909` |
| 训练侧根治过冲 | 电荷损失温度化（charge_temp=0.5）：增益 2.57→1.04，新编码器默认关校准 | `b91ab93` |
| **第十三轮 n=20 扩样本** | **推翻 n=5 假阴性**：32 组配对检验 23 组显著（%sol/Tm 真实下降）；机制=条件注入 >50% 位点非保守替换 | `220ab3b` |
| **判断标准 v1** | `index/DESIGN_CRITERIA.md`（用户要求先立标准再训练）：H1 TM≥0.70 / H2 电荷±2 / H3 聚集合法；S1 注入选择性 / S2 可开发性权衡 | `220ab3b` |
| **第十四轮 S1 训练修正** | **治 S1 部分成功**：原生标签 50%→70% + seq-keep 正则；H1 全达标优于上轮（折叠失败 0%、pLDDT 大幅修复）、H2 3/4 PDB、S1 0.45→0.67 未达 0.7、%sol 仍降（设计权衡） | `待提交` |
| **第十五轮 对齐两真实目标** | **方向修正**：S1/seq-keep 跑偏作废（目标=全新序列）；2LZM 过冲根因=base 电荷漂移−6.32 超训练扰动范围 + 加电开关多；判断标准 v1→v2（S1 降级软区间+防坍塌）；实现 `--fixed_residues`（冒烟 100% 保持）+ `--placeholder_prob` 占位符样本 + 扰动 ±1~8；重训中 `output/finetune_v2/` | `待提交` |
| **第十六轮 v2 复验（n=20）** | **目标1 基本成功**：H1 折叠 12/12、S4 位点固定 100%、无坍塌；**目标2 分叉**：负电 target 4/4 精确命中+折叠良好、**正电过冲（1BC8/2LZM）**、**占位符臂折叠全失败**（根因=训练偏负+无条件负漂移，D+E 占比 18–36%）| `待提交` |
| **第十七轮 占位符修复（n=20）** | **占位符折叠完全修复**：均值占位（has_charge=1+值=训练均值）+占位样本施加电荷损失 → t2_ph TM 0.89–0.97、**全部 20 臂 H1 通过**；负电 4/4 保持、正电温和化后 1CRN/1UBQ 命中（1BC8/2LZM 仍过冲）、H3 正电违规 3/4→1/4 | `待提交` |

**今日资产**：`data/cath/`（CATH S40 34,653 结构域坐标+序列，818MB，git 不跟踪）；打分工具链（ESMFold 回折+TM-score、Protein-Sol、TemBERTure）全通。

---

## 一、项目一句话

把**工作环境 pH（及净电荷/局部电荷）作为条件约束**，整合进 LigandMPNN 结构逆折叠模型，生成「符合 pH 电荷约束」的蛋白序列。

两级计划：
- **第一版** `index/PROJECT_PLAN.md` — pH 电荷条件生成主线（Phase 0–4）
- **第二版拓展** `index/PROJECT_EXTEND.md` — 多目标可开发性微调（可设计/热稳/可溶），优先用开源 MoMPNN，把更好模型放回主线管线

---

## 二、当前阶段：Phase 1 已全部完成 ✅

**里程碑已达成**：不改模型代码，纯 logit bias 采样策略实现 pH 感知电荷约束生成。

### 已完成模块（`code/src/`，全部通过测试）
| 模块 | 作用 |
|------|------|
| `pka.py` | 侧链/末端 pKa 表、AA 索引、带电类型 |
| `differentiable_charge.py` | sigmoid 平滑 HH 方程，`net_charge`（字符串）/`net_charge_from_logits`（可微） |
| `isoelectric_point.py` | `find_pI` 二分搜索（验证用） |
| `structure_aware_filter.py` | 4 条结构规则 → [L,21] bias，YAML 预设 |
| `condition_embedding.py` | Soft Prompt MLP + mask-aware 条件向量 [7]（Phase 2 用） |
| `losses.py` | 复合损失 CE+电荷偏差+结构惩罚+DPO+margin（Phase 2 用） |
| `guided_sampler.py` | 静态/动态 bias 解码，包装 LigandMPNN |
| `charge_lookahead.py` | **动态电荷前瞻**：每步 bias = strength·(target−Q_current)·q_k |

### 一键入口与测试
- `code/run_guided.py` — 完整管线：PDB→filter→引导采样→电荷/pI 统计→fasta+json
- `code/tests/test_all.py` — **36 项全通过**
- `code/tests/smoke_guided.py` — 真实 LigandMPNN + 1BC8.pdb 冒烟通过

### 本次会话关键成果（2026-08-16）
1. **修复 charge_lookahead target 失效 bug**（提交 `6d76da4`）：
   - 根因：`bias=-strength·(Q_k−target)` 中 target 落在常数项，被 softmax 平移不变性抵消
   - 修复：`bias_k = strength·(target−Q_current)·q_k`，target 进入交叉项
   - **验证**：1BC8 pH7.4，target=+8/0/−8 → 平均净电荷 **+8.06 / +0.23 / −7.96**，精准命中；叠加结构过滤器、弱强度均正常
2. **第二版拓展计划** `index/PROJECT_EXTEND.md`（提交 `cac283f`）：
   - 路线 A：直接用开源 **MoMPNN**（ProtAlign ICLR 2026，GitHub: Qivon7/MoMPNN，多目标 DPO：可设计+溶解+热稳）
   - 路线 B：按 ProtAlign 方法自微调；路线 C：兜底自研
   - 第一版 `PROJECT_PLAN.md` 三处加指针，两版形成整体

---

## 三、进展与待办

### 本轮已完成（2026-08-16 第二轮）
1. **Stage E0 ✅**：clone MoMPNN（`ConfuMPNN/MoMPNN/`，仅权重无代码）。结论：权重 = **纯 backbone ProteinMPNN**，`load_state_dict(strict=True)` 8/8 通过，1BC8 前向跑通（seq_rec≈0.45）。报告：`analysis/2026-08-16_mompnn_avail.md`
2. **Stage E1 接入 ✅**：`run_guided.py` 加 `--model_type auto`（按 ckpt 有无 `atom_context_num` 自动检测）。MoMPNN 权重跑通 pH 引导（target=0 → 平均 −0.01±0.80）；原版 LigandMPNN 回归通过
3. **仓库清理 ✅**：`MoMPNN/` 已 gitignore；冗余 remote `new` 删除

### 第三轮（2026-08-16）— E1 验证扩展设计（已归档）
E1 对照实验**全部完成并 push**（提交 `c644a6b`，三目标结果：可溶 +12.8、热稳 +7.8°C、电荷更准、pLDDT 持平）。随后完成**验证扩展的方法论设计**，归档 `session/2026-08-16_e1_validation_design.md`，并写入 `PROJECT_EXTEND.md`（Stage E1b + 决策 6–8）。要点：
- ⚠️ **1BC8 身份修正**：SAP-1 ETS 转录因子 DNA 结合域（93aa，winged HTH），非普通球状蛋白
- **PDB 代表性矩阵**：1BC8（核酸结合）/ 1CRN（极小疏水）/ 1UBQ（典型可溶球状）/ 2LZM（全 α 较大）
- **混合设计**：方案 A（同骨架变条件→机制）+ 方案 B（各蛋白生理条件 + 回折 TM-score 对比原结构→泛化）
- **主证据升级**：回折 TM-score（ESMFold 存结构 → us-align）为主，pLDDT 仅辅助
- **位点固定对照臂**：需给 `run_guided.py` 加 `--fixed_residues` + `pka.py` 电荷空间预检
- **阈值防过拟合**：先验设定 + 留一蛋白检查，不做阈值搜索
- **Phase 2 数据**：CATH 4.2 S40（ESM-IF 同源分离划分），条件标签自算

### 第四轮（2026-08-16）— E1b 验证扩展执行完成 ✅
按 `session/2026-08-16_e1_validation_design.md` 全量执行并产出报告 `analysis/report/2026-08-16_e1_extended.md`：
- 采样 336 样本（4 PDB × 2 模型 × 10 条件目录）；打分四指标全齐（ESMFold 回折+TM-score、Protein-Sol、TemBERTure）
- **电荷响应 24/24 单调**（4 PDB × 2 模型 × 3 pH，target 梯度全过）
- **MoMPNN 16/16 全优**（pLDDT/TM/%sol/Tm × 4 PDB，留一蛋白符号一致）
- 联合可用率互有胜负（卡点：电荷精确命中 ±0.3 + pH4 物理极限）
- 工具链沉淀：`tm_score.py`（US-align）、`esmfold_score.py`（回折存结构+批量）、`temberture_score.py`（并行分组）、`e1_ext_*.{sh,py}`
- ⚠️ 性能教训：TemBERTure 必须用**直接 env python + OMP_NUM_THREADS=4** 并行（16 进程×4 线程），`conda run` 并发会因 conda 锁卡死、无线程限制会因 CPU 争抢慢 5 倍
- ⚠️ 1CRN 特例：crambin 二硫键蛋白 ESMFold 置信度低（native pLDDT 48.7），该 PDB 的 pLDDT/TM 解读需谨慎

### 第五轮（2026-08-16）— 计划1 Phase 1 收尾完成 ✅（最初版本可交付）
1. **结构过滤器 99 分位阈值**（CATH S40 34,653 结构域采样 1,000 个 / 151,519 残基位）：salt_bridge 5→**4**、core_charge 4→**6**、规则 4 改**局部密度口径**（原连通分量口径 p50=17 全量误触发）；36/36 测试通过；写入 `filter_presets.yaml` + `structure_aware_filter.py`
2. **示例蛋白对比**（4 蛋白 × 4 预设 + 3 pH）：电荷引导精确命中 target（预设无关，验证正交设计）；**诚实边界**：无引导时模型不感知 pH（同一蛋白各 pH 序列完全相同，电荷差异来自 net_charge 物理计算）
3. **最初版本交付**：README 加使用说明 + `run_guided.py` 一键生成；`data/cath/`（S40 818MB 已下载，git 不跟踪）
4. 报告：`analysis/report/2026-08-16_phase1_examples.md`
5. ⚠️ 下载技巧：单连接被限速时用 `parallel_download.py` Range 8 段并行（818MB 从 ~30min+ 降到 ~10min）
6. **详细文档交付**：`docs/TECH.md`（技术）、`docs/CONFIG.md`（配置）、`docs/USAGE.md`（使用），README 加文档导航，DOCUMENT_INDEX 登记

### 第六轮（2026-08-16）— E4 完成 ✅（MoMPNN 设为默认生成器）
- `run_guided.py` 默认权重改为 MoMPNN（`mompnn_temberture_tm_esm_6_4_4_b01.ckpt`），`--weights` 保留可覆盖（回退原版 LigandMPNN 含配体上下文）
- 默认命令冒烟：target=0 → −0.01±0.80，序列确认 MoMPNN
- 文档同步（README/USAGE/CONFIG）；报告 `analysis/report/2026-08-16_e4_default_mompnn.md`
- ⚠️ 使用影响：默认纯 backbone（无配体上下文），需配体任务显式指定原版权重

### 第七轮（2026-08-16）— Phase 2 训练数据标签构建完成 ✅
- CATH S40 999 结构域 × 8 pH = **7,992 样本**：`data/cath/labels.npz`（coords/seqs/pH/charge/pI）
- 条件向量 μ/σ 已统计并写入 `condition_defaults.yaml`（μ=[6.97,1,−1.34,...], σ=[1.73,0,7.78,...]）；ConditionEncoder 带 μ/σ 前向验证通过
- 标签构建方案（self-consistent）：条件电荷 = native 序列在该 pH 下的净电荷，使 CE 与 charge_deviation 损失一致不冲突；推理时给任意 (pH,target) 外推
- 脚本 `code/tests/build_labels.py`；README 补"CATH 训练数据下载（git 不跟踪）+ 选装打分工具"节

### 第八轮（2026-08-16）— Phase 2 条件微调训练启动 ✅
- **微调目标（三层）**：架构=冻结 MoMPNN 只训 ConditionEncoder（~75K 参数）；直接=学会「(pH, target_charge)→氨基酸分布」映射（冒烟 charge 4.71→3.67 下降）；最终=推理外推到未见过的 (pH,target) 组合
- **脚本 `code/train_finetune.py`**：冻结 MoMPNN backbone + ConditionEncoder（cross-attention 注入 h_V，等价 soft prompt 但无需改 E_idx/mask）+ teacher-forced 并行解码 + 复合损失
- **损失 `CE + λ_c·charge_deviation + λ_kl·KL锚定`**：KL 锚定（新增）约束条件化输出不偏离 backbone 无条件分布太远 → **防失控**（防止破坏 MoMPNN 的可溶/Tm/可设计性）
- **混合目标**：50% 自洽（target=native 电荷，锚定结构）+ 50% 扰动（±Uniform[1,4]，制造 CE 与电荷冲突 → 教模型电荷偏移；纯自洽的隐患：CE 与电荷同时被重建 native 满足，模型学不到偏移）
- **⚠️ 关键教训/修正**：
  - prody `parsePDB` 按文件名后缀判格式，CATH 无后缀文件被当 mmCIF → 用 `.pdb` 符号链接目录解决（`data/cath/S40/dompdb_pdb/`，git 不跟踪）
  - 对计划 4.5「token 拼 decoder 前缀」的实现修正：前缀需重排 E_idx 易错，改用 cross-attention 注入 h_V
  - 对计划「全量微调」的保守偏离：默认生成器已是 MoMPNN（多目标 DPO 权重），全量微调有破坏其价值的风险 → 先冻结 backbone 只训编码器，不够再逐层放开
- 冒烟：3 域 1 epoch ✅、5 域 3 epoch ✅（ce 稳定 1.58，charge 4.71→3.67）
- **后台启动**：`nohup setsid ... python code/train_finetune.py --device cuda:1 --epochs 30`；进度 `bash code/tests/train_status.sh`；每 epoch 存 checkpoint + `log/train_progress.json`
- 报告：`analysis/report/2026-08-16_phase2_training_start.md`

### 第九轮（2026-08-16）— Phase 3 条件注入验证进行中 ✅/⏳
- **接入 `run_guided.py`**（提交 `4fccc1c`）：`--cond_encoder` 加载微调编码器，cross-attention 注入 h_V（与训练同机制）；`conditioned/baseline` 两模式。新模块 `src/conditioned_sampler.py`；`guided_sampler.guided_sample` 支持预编码 h_V
- **pH 响应 Go/No-Go 通过（4/4 PDB）**，报告 `analysis/report/2026-08-16_phase3_pH_response.md`：
  - target 响应严格单调（1BC8: 0→+0.9, 9→+26.6；2LZM: 0→+1.7, 13→+37.4 等）
  - 跨 pH identity 0.68–0.92（<100%）——**Phase 1 诚实边界（同 seed 各 pH 序列相同）被打破**
  - ⚠️ **校准发现**：target→电荷线性增益 ~2.9×（`实际≈2.9·target−1.1`）；机制=采样置信度放大（训练优化 softmax 期望电荷，推理测采样序列电荷；温度实验证实：temp 0.3→+13.0、1.0→+7.1、2.0→+3.7@target=+5）。实用缓解：温度 1.0–2.0 或线性校准
- **防失控判据 PASS**（报告 `analysis/report/2026-08-16_phase3_antidrift.md`）：
  - **机制证实**：pLDDT 掉主因是电荷过冲，非条件化本身——1BC8 过冲版 77.4 → 校准版 82.3 → 基线 82.8（校准后恢复≈基线）
  - **%sol/Tm 在采样噪声内**（1BC8 %sol std=7.9、Tm std=6.2，差异 <1σ；2LZM 甚至 +2.4/+2.5）
  - **TM-score 结构保持**（0.84-0.98）；唯一待修=电荷校准过冲 ~2.9×（推理侧线性校准已验证有效，或训练侧改）

### 第十轮（2026-08-16）— 电荷校准落地 ✅
- `condition_defaults.yaml` 新增 `charge_calibration`（gain=2.57, offset=0.16；4 PDB 15 点合并拟合 R²=0.946，比单用 1BC8 的 2.9 更鲁棒）
- `run_guided.py --cond_encoder` 默认线性校准 `target_eff=(desired-offset)/gain`；`--no_calibration` 关闭；summary 记录 `calibrated` 字段
- 验证 1BC8：target 8.9/5/-5 → +8.76/+5.62/-7.13（校准前 +25.6/+13.0/-15.6，误差收敛 <1）；残余在采样噪声内
- 提交 `0be534b`；`docs/CONFIG.md` 补电荷校准节

### 第十二轮（2026-08-16）— 训练侧根治过冲 ✅（推荐方案）
- `net_charge_from_logits`/`charge_deviation_loss` 加 `temperature` 参数；`train_finetune.py --charge_temp 0.5` 重训（14.8min，`output/finetune_t05/`）
- **效果：增益从 2.57 收敛到 ~1.04**（1BC8 未校准：target 8.9/5/0/-5 → +9.75/+4.74/-0.14/-4.89，误差 ≤0.85）
- **pH 响应保留**：4/4 PDB target 单调 + 跨 pH identity 0.78-0.92（<100%）
- **新编码器默认关校准**（`enabled: false`）——全局校准会过校正（per-PDB 增益 1.04~1.7，2LZM ~1.7 仍略过冲）
- 教训：推理侧校准是补丁（对旧编码器有效），训练侧温度化才是根治（增益→1，且自动对齐推理采样分布）

### 第十三轮（2026-08-16）— n=20 扩样本推翻 n=5 假阴性 + 判断标准 v1 ✅
- **扩样本动机**：n=5 的"在噪声内"是**统计功效不足的假阴性**。防过拟合第一道防线=**泄漏检查**：1BC8/1CRN/1UBQ/2LZM 均不在训练域列表（labels.npz 999 域；dompdb 目录里的 1crnA00 已下载但未抽样进训练集）
- **对称配对协议**（新脚本 `phase3_antidrift_extend.py`）：同一 seed → 同一 randn → 同一解码顺序；基线（MoMPNN 无注入）vs 条件（finetune_t05）唯一差异=条件；双场景（A 温和 pH7.4/target=原生；B 压力 pH4.0/target=原生+5）；n=20 × 4 臂 × 4 PDB = **320 条**
- **四指标打分**（`phase3_antidrift_n20_score.sh`，递归扫描 16 个 arm）；**配对统计**（`phase3_antidrift_n20_stats.py`）：32 组配对 t + Wilcoxon，BH-FDR 校正
- **🚨 发现**：条件注入显著降低 **%sol（-4~-24 分）**、**Tm（-2.5~-7.1°C）**、部分 pLDDT（1BC8 -5.2、2LZM -4.4）；TM-score 仅轻微下降（-0.007~-0.074）；32 组中 **23 组显著**——**n=5 防失控 PASS 是假阴性**
- **机制**（序列级证据）：条件注入改变 **>50% 位点**，非保守替换（`R→I`、`K→P`、`E→T`、`D→N`）；电荷目标命中（增益≈1）但以牺牲可设计性为代价；A 场景（target=原生）identity 仅 0.45-0.59 = **无需求时也重写**（注入选择性失败）
- **用户概念纠正**：逆折叠本质=**骨架固定、序列可重写**，序列大改是设计行为非破坏；"改了 pI 还要求序列相似"才不合理；%sol/Tm 下降是**设计权衡**非失控
- **判断标准 v1**（`index/DESIGN_CRITERIA.md`，用户要求"先立标准再训练"）：
  - 硬约束：H1 结构自洽（回折 TM 中位数≥0.70=整体骨架相似，失败率 TM<0.5 ≤10%）/ H2 电荷命中（|平均实际−target|≤2.0）/ H3 电荷聚集合法（结构过滤器违规率≤基线+5pp）
  - 软判据：S1 注入选择性（A 场景 identity≥0.7）/ S2 可开发性（pLDDT/%sol/Tm 报告绝对值，不判 FAIL）
  - 参照：R1 天然蛋白对（CATH 同 superfamily 找"骨架相似 pI 不同"蛋白对，验证目标可达）
- **现有编码器按新标准**：H1 ✅（放宽后全 PDB 过）、H2 ⚠️（2LZM 过冲 target 8→+13.24）、**S1 ❌**
- **明天训练修正方向**（见"下一步"）

### 第十四轮（2026-08-17）— 训练修正治 S1 注入选择性 ⏳（部分成功）
- **训练修正设计**（session `2026-08-17_s1_training_fix.md`）：原生标签比例 50%→70%（`--perturb_prob 0.3`）+ 新增**序列保持正则**（`--lambda_keep 0.5`，`losses.py` 新增 `sequence_keep_loss`）：以无条件 argmax 序列为锚，对**自洽样本**做 `CE(logits_cond, anchor)`——S1 判据（A 场景 identity≥0.7）的训练侧直接对应，比 KL 更直接（管住 argmax 翻盘）
- **训练**（`output/finetune_s1/`，30 epoch 16.5min）：ce 1.856 稳定、charge 1.969、**keep 0.843 全程稳定**
- **复验**（`output/phase3_antidrift_s1_n20/`，320 条；TemBERTure SSL 错误→`HF_HUB_OFFLINE=1` 离线重跑修复）：
  - **H1 全达标且优于上轮**：条件臂 TM 中位 0.862-0.972 ≥0.70；**折叠失败率 0%**（上轮 2 条失败）；pLDDT 掉落大幅修复（2LZM −4.4→−1.1、1BC8 不再显著）
  - **H2 6/8 达标**（1BC8/1CRN/1UBQ 全过；2LZM 仍过冲 A +11.09/B +19.45 vs target 8/13，略改善）
  - **S1 提升未达标**：A 场景 identity 0.52-0.67（上轮 0.45-0.59；1CRN 0.668 逼近 0.7）
  - **%sol 仍 8/8 显著降**（−3.8~−21.2，1CRN/1UBQ 最重）——A 场景残余重写代价 + B 场景电荷目标张力（设计权衡）
  - Tm 半数恢复（1BC8/2LZM PASS）；显著组 23→18
- 报告：`analysis/report/2026-08-17_phase3_s1_fix.md`

### 第十五轮（2026-08-17）— 对齐两真实目标：方向修正 + 治 H2 ⏳（训练中）
- **方向修正（用户纠正，重要）**：S1（identity≥0.7）与目标"全新序列"矛盾 → **作废**，seq-keep 不再加压。
  两个真实目标：① 天然骨架（RF3 relax 微调，非本项目）→ **全新序列** + 理化性质≈天然；② 人工设计骨架（靶点口袋，RF3 生成）→ **全新序列** + 简单理化预期（**部分条件用占位符不控制**）。
  ConfuMPNN 只负责：限制条件下生成序列 + 确保折叠回骨架；位点固定=人工/ligandMPNN 的事
- **2LZM 过冲根因（2026-08-17 数据实测）**：❌排除"柔性链"（B-factor 中位 16.7，T4L 最刚性）；❌排除"超长"
  （164 在训练分布 60-75 分位）；✅**真因 = base 电荷漂移 × 加电开关多 × 正电富集骨架**：
  MoMPNN 不感知电荷 → 4 验证蛋白 base 电荷全偏离原生（1BC8 −9.66 / 2LZM −6.32 / 1UBQ −3.46 / 1CRN −0.18）；
  2LZM 需补 +6.32 **超训练扰动范围 ±1~4**；带电残基 45 个（27%）最多 → 加电位置多累积过冲（seed 波动 [+2.1,+17.1]）；
  1CRN 反例：仅 4 带电残基，B 场景欠调够不着 target
- **判断标准 v1→v2**（`index/DESIGN_CRITERIA.md`）：S1 降级为 **S1\* 相似性软区间 0.4–0.7 + 防坍塌监控**
  （文献：P2 ResiDPO 的 RCL = 选择性保序列；P3 Weighted-score DPO = 坍塌教训）；新增 **S3 占位符语义**、**S4 位点固定行为**
- **代码实现（已冒烟 ✅）**：
  - `run_guided.py --fixed_residues`（复用 LigandMPNN chain_mask 原生机制，guided_sampler 已支持；冒烟 5 位置 × 4 序列 100% 保持）
  - `train_finetune.py --placeholder_prob 0.15`（占位符样本，两种语义各半：flag=0+值0 / flag=1+值=均值，跳过电荷损失）
  - 扰动幅度 `--perturb_scale 4 → 8`（参数已存在，治 2LZM 过冲/1CRN 欠调）
- **训练（第十五轮，后台启动，`output/finetune_v2/`）**：30 epoch，perturb_scale=8 + placeholder_prob=0.15 + λ_keep=0.5 + charge_temp=0.5
- 计划：`session/2026-08-17_validation_plan_v2.md`

### 第十六轮（2026-08-17）— v2 复验（n=20，对齐两真实目标）✅ 完成
- **复验报告**：`analysis/report/2026-08-17_phase3_v2_validation.md`（判定 JSON：`output/finetune_v2_validate/v2_judgment.json`）
- **目标 1 形态（天然骨架 + 位点固定 + pI≈天然）基本成功**：
  - H1 折叠 12/12（TM 中位 0.86–0.98，失败率 0%）
  - **S4 位点固定 100% 保持**（4 PDB × 4 位点 × 20 序列全为指定氨基酸）
  - S1\* identity 0.55–0.75 健康、防坍塌正常；S2 pLDDT 75–90 / Tm 56–71°C 健康
- **目标 2 形态分叉**：
  - ✅ **负电 target（从零设 pI）4/4 精确命中**（dev≤0.72）+ 折叠良好
  - ❌ **正电 target 过冲**：1BC8/2LZM（target +13/+14 → dev 5.8–8.7）——训练数据平均电荷 −1.34 偏负，正电外推不够
  - ❌ **占位符臂（has_charge=0）折叠全失败**：TM 中位 0.21–0.34、失败率 100%、pLDDT 33–58；
    **根因链条**：占位样本跳过电荷损失 → 模型学"维度不控制" → MoMPNN 无条件负漂移（−3.5~−9.7）→ 电荷极端负极化（−8~−16，D+E 占比 18–36%）→ 破坏折叠
- **H3 电荷聚集**：正电臂 3/4 违规（与过冲耦合）；负电臂 2/4 小幅违规
- **代码**：`tests/phase3_v2_validation.py`（采样）、`tests/phase3_v2_score.sh`（打分）、`tests/phase3_v2_stats.py`（v2 判定）

### 第十七轮（2026-08-17）— 占位符折叠修复（均值占位）✅ 完成
- **训练修正**：占位符语义统一**均值占位**（has_charge=1 + 值=训练均值 −1.34）+ **占位样本施加电荷损失**（target=均值）
  ——对应第十六轮占位符折叠失败根因（占位样本跳过电荷损失 + seq-keep 锚定无条件 argmax 负漂移基线）
- **结果（n=20，报告 `analysis/report/2026-08-17_phase3_v3_placeholder_fix.md`）**：
  - ✅ **占位符折叠完全修复**：t2_ph TM 0.89–0.97、失败率 0–5%（十六轮 0.21–0.34/100%）；电荷全部落均值附近
    （−0.3~−1.6，十六轮 −8~−16）；多样性恢复（pairID 0.61–0.69，十六轮 0.13）
  - ✅ **全部 20 臂 H1 通过**（TM 0.88–0.98，失败率 ≤5%）
  - ✅ 负电 target 4/4 命中（dev≤1.82）；正电温和化（+3）后 1CRN/1UBQ 命中（0.48/0.41）
  - ⚠️ **1BC8/2LZM（正电富集蛋白 native+8/+9）仍正电过冲**（dev 4.6/7.4）——训练分布偏负，高正电外推不足
  - ✅ H3 正电违规 3/4→1/4；S4 位点固定 100%；S1\* 无坍塌
- **结论**：目标 1 ✅ 成立；目标 2 核心能力 ✅（负电/中性 4/4 + 占位符可折叠），正电富集蛋白是唯一残余
- **代码**：`train_finetune.py`（均值占位）、`phase3_v2_validation.py`（t2_pos+3/t2_ph 均值）、`phase3_v2_stats.py`（t2_ph 不判 H2）

### 下一步（2026-08-18，从第十七轮继续）
1. **正电过冲（1BC8/2LZM 正电富集蛋白）**（可选，若需根治）：训练域补正电富集蛋白 / 训练扰动偏向正电 / 推理侧正电 target 单独校准 → 重训 + 复验
2. **R1 天然蛋白对参照**（可选）：CATH 同 superfamily 找"骨架相似 pI 不同"蛋白对
3. **真实骨架泛化**：用户提供 RF3 relax/人工骨架后，复用同一流程跑通（考验泛化，无需重训）

---

## 四、运行速查

```bash
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn          # Python 3.11, torch 2.2.1+cu121
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
python tests/test_all.py          # 36 项单元测试
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --num_samples 5 --strength 0.5 --out_dir output/guided_1BC8_pH7.4/target_0
```

环境细节见 `CLAUDE.md` 与 memory `confumpnn-env-setup.md`。
ESMFold 回折在 `confumpnn-esmfold` 环境（conda, Python 3.10, torch 2.6.0+cu124，openfold 依赖需确认）。

---

## 五、Git 状态

- 分支 `main`，远程 `origin` = git@github.com:Yu-Bk/ConfuMPNN.git
- 最近提交：`b51cdaa`（第十四轮：训练修正治 S1 部分成功）← `220ab3b`（第十三轮：n=20 扩样本推翻假阴性 + 判断标准 v1）← `b91ab93`（温度化根治过冲）← `d9b67ff`（电荷损失温度化）← `212d392`（文档）← `0be534b`（校准落地）← `c2de909`（n=5 防失控 PASS）← `d75fb96`（打分工具修复）← `fcc9845`（pH 响应报告）← `4fccc1c`（Phase 3 注入）← `177d902`（train_status 修复）← `1477a79`（微调启动）
- ⚠️ 待提交（第十五~十七轮）：`code/run_guided.py`（--fixed_residues）、`code/train_finetune.py`（--placeholder_prob + 均值占位）、`index/DESIGN_CRITERIA.md`（v2）、`session/2026-08-17_validation_plan_v2.md`（含第十七轮 §8）、`analysis/report/2026-08-17_phase3_v2_validation.md`（十六轮）、`analysis/report/2026-08-17_phase3_v3_placeholder_fix.md`（十七轮）、`code/tests/phase3_v2_{validation,score,stats}.py`（复验工具）、本文件（第十五~十七轮）、`index/DOCUMENT_INDEX.md`
- `LigandMPNN/`、`foundry/`、`MoMPNN/` 为 clone 源码不跟踪；`data/`、`code/output/`、`code/log/`、`*.pt`、`*.ckpt` 已 gitignore
