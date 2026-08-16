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

### 下一步（按优先级）
1. **第一版 Phase 2 / 第二版 E2 条件微调**：训练数据 CATH S40 已就绪（坐标+序列）；计算条件向量 μ/σ → `condition_defaults.yaml`；编写微调脚本（backbone + ConditionEncoder + 复合 loss）；GPU 训练 → **模型 pH 感知的正解**（Phase 1 诚实边界指出模型自身无 pH 先验）
2. 可选：`--fixed_residues` 位点固定对照臂

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

- 分支 `main`，远程 `origin` = git@github.com:Yu-Bk/ConfuMPNN.git（E1 对照已 push 至 `c644a6b`；**本次方法论归档待提交**）
- 最近提交：`c644a6b`（E1 三目标对照完成）← `e08c4c0`（E0 调研）← 此前（E1 接入 / gitignore MoMPNN / Phase 1 快照）
- `LigandMPNN/`、`foundry/`、`MoMPNN/` 为 clone 源码不跟踪；`code/output/`、`code/log/`、`*.pt` 已 gitignore
