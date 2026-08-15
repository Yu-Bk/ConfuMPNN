# ConfuMPNN — 项目说明

## 项目概述
将蛋白质在特定 pH 环境下的理化性质（净电荷、局部电荷分布）作为条件约束，整合到基于结构的蛋白序列生成流程（LigandMPNN 逆折叠）中。核心创新：**在 LigandMPNN 这类显式建模配体原子上下文的结构条件逆折叠模型上，首次加入 pH 感知的电荷条件控制**。

详细技术计划见 `PROJECT_PLAN.md`（完整中文计划，含文献调研、分阶段实施、风险表）。

## 运行环境（conda，位于 ~/miniconda3/envs/）

| 环境 | Python | torch | 用途 | 状态 |
|------|--------|-------|------|------|
| `confumpnn` | 3.11 | 2.2.1+cu121 | **LigandMPNN 推理/开发** | ✅ 已验证跑通 |
| `confumpnn-esmfold` | 3.10 | 2.6.0+cu124 | **ESMFold 回折验证** | ⚠️ openfold 依赖需确认 |

### 已确认的环境要点
- `confumpnn` 环境**不需要 dgl、torchvision、torchaudio**。之前 torchvision 0.21.0 / torchaudio 2.6.0 与 torch 2.2.1 不匹配导致 import 崩溃，已卸载。若装新包注意不要引入它们。
- LigandMPNN 需要：torch 2.2.1、biopython 1.79、numpy 1.23.5、scipy 1.12.0、prody 2.4.1、networkx、`dm-tree`（import 名是 `tree`）。propka 3.5.1 也已装。
- `confumpnn-esmfold` 用 fair-esm 2.0.0 的 `esm.pretrained.esmfold_v1()`（自带 ESMFold），需要 openfold、fairscale、einops、pytorch_lightning。**nvcc 系统版是 CUDA 11.5，与 torch cu124 不匹配**——编译 openfold 必须在 env 里装 conda cuda-toolkit 并用 `pip install --no-build-isolation openfold`。

## 常用命令

### LigandMPNN 推理（在 ConfuMPNN/LigandMPNN 目录下）
```bash
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn

# 默认加载 ProteinMPNN 权重，若用配体上下文要显式指定 LigandMPNN 权重：
python run.py --seed 111 --pdb_path ./inputs/1BC8.pdb \
  --out_folder ./outputs/default \
  --model_weights_path ./model_params/ligandmpnn_v_32_010_25.pt
```
- 模型权重在 `ConfuMPNN/LigandMPNN/model_params/`（已完整下载 15 个 .pt 文件）
- 示例输入 PDB 在 `inputs/`（1BC8.pdb 等）

## 文件结构
- **文件管理规范**：所有文件分类存放遵循 `index/FILE_MANAGEMENT.md`（实验进行时的文件管理规则），文档定位见 `index/DOCUMENT_INDEX.md`
- `index/PROJECT_PLAN.md` — 完整项目计划（中文）
- `LigandMPNN/` — LigandMPNN 官方源码 clone（含 openfold/ 子模块；源码 clone，未跟踪，不提交）
- `foundry/` — RosettaCommons 蛋白设计工具库 clone（含 RF3/ProteinMPNN，备选验证方案；源码 clone，未跟踪，不提交）
- `code/`、`analysis/`、`literature/`、`session/`、`source/` — 按 `index/FILE_MANAGEMENT.md` 分类存放代码 / 实验分析 / 论文笔记 / 会话记录 / 论文源码

## 下一步（Phase 1：Level 1 引导采样）
按 `PROJECT_PLAN.md` 第五部分实施，待创建模块：
- `differentiable_charge.py` — 可微 pH 感知净电荷计算（Henderson-Hasselbalch + pKa 表）
- `isoelectric_point.py` — pI 二分搜索
- `structure_aware_filter.py` — 结构感知过滤器（5 条规则，logit bias 注入）
- `guided_sampler.py` — 引导采样 wrapper
- `configs/filter_presets.yaml` — 过滤器场景预设

## Git 说明
- GitHub 远程：`origin` = git@github.com:Yu-Bk/ConfuMPNN.git（另有冗余 remote `new`，同为 https，可删）
- `LigandMPNN/`、`foundry/` 是克隆源码，**未跟踪**，不应提交（建议加入 .gitignore）
