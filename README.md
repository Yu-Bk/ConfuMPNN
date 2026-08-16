# ConfuMPNN

把**工作环境 pH（及净电荷）作为条件约束**，整合进 LigandMPNN 结构逆折叠模型，生成「符合 pH 电荷约束」的蛋白序列。核心创新：在显式建模配体原子上下文的结构条件逆折叠模型上，首次加入 pH 感知的电荷条件控制。

> 当前进度：**Phase 1（Level 1 引导采样）已完成并交付**，`run_guided.py` 一键生成可用。详细计划见 `index/PROJECT_PLAN.md`（第一版）与 `index/PROJECT_EXTEND.md`（第二版拓展）。

## 快速开始（Phase 1 一键生成）

环境：`conda activate confumpnn`（Python 3.11, torch 2.2.1+cu121）。

```bash
cd code
# 一键生成：指定 PDB + pH，可选目标净电荷与结构过滤器预设
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 --num_samples 10

# 用 MoMPNN 权重（多目标 DPO 微调版，可溶/热稳显著更优）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --weights ../MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt

# 不同结构过滤场景（default / nucleic_acid_binding / membrane / acidic）
python run_guided.py --pdb input/2LZM.pdb --pH 5.5 --preset acidic --target_charge 0
```

输出：`code/output/guided_<pdb>_pH<pH>/` 下 `seqs.fa`（含每序列净电荷/pI）+ `summary.json`。

**核心机制**（两条正交约束叠加，`code/src/`）：
- **动态电荷前瞻** `charge_lookahead.py`：解码每步把净电荷拉向 target（`bias_k = strength·(target−Q_current)·q_k`）
- **结构感知过滤器** `structure_aware_filter.py`：4 条空间规则（99 分位阈值），抑制电荷异常聚集
- 两者通过 `guided_sampler.py` 在采样时逐步施加

**验证报告**：`analysis/report/2026-08-16_phase1_examples.md`（阈值统计 + 示例蛋白）；`analysis/report/2026-08-16_e1_three_targets.md` 与 `2026-08-16_e1_extended.md`（MoMPNN 对比）。

## 文件结构

项目的文件分类存放遵循 [index/FILE_MANAGEMENT.md](index/FILE_MANAGEMENT.md)：

- `code/` — 实验模块代码（含 `input/`、`output/`、`log/` 子目录）
- `analysis/` — 实验分析报告（含 `report/`、`archieved/`、`accident/`、`ablation/` 子目录）
- `index/` — 项目规划、文件管理规范、文档索引（`PROJECT_PLAN.md` 等）
- `literature/` — 论文笔记（含 `baseline/`、`innovation/`、`pattern/`、`tools/`、`phenomena/` 子目录）
- `session/` — Claude Code 会话记录
- `source/` — 论文源码或链接
- `data/` — 外部数据集（CATH S40 结构域，818MB，git 不跟踪）

详细技术计划见 [index/PROJECT_PLAN.md](index/PROJECT_PLAN.md)。
