# ConfuMPNN — 项目说明

## 项目概述
将蛋白质在特定 pH 环境下的理化性质（净电荷、局部电荷分布）作为条件约束，整合到基于结构的蛋白序列生成流程（LigandMPNN 逆折叠）中。核心创新：**在显式建模配体原子上下文的结构条件逆折叠模型上，首次加入 pH 感知的电荷条件控制**。

**当前状态（2026-08-27）**：
- **模型为迭代演进中**：v7（MoMPNN backbone，无配体/小蛋白）+ v9（LigandMPNN backbone，配体/大蛋白）是**阶段性成果**（2026-08-19 曾暂停训练），**不是终版**——v10（A 条件解耦 + B 表面电荷监督 + C 结构惩罚）正在设计中，见 `index/PROJECT_LOCAL.md`（v3 论文导向方案）与 `index/PROJECT_LOCAL_P1_PLAN.md`（P1 对照计划）
- v7/v9 使用边界（当前可用版本）：`analysis/report/2026-08-18_model_charge_limits.md` §8（v9 配体模式：正电到 +8、负电保守到 −5、长序列需检查）
- 权威指南：`WORKFLOW_GUIDE.md`（框架/数据流/参数/损失/为什么）；新机配置 `docs/SETUP_NEW_MACHINE.md`；数据组织 `data/README.md`

## 运行环境（conda，位于 ~/miniconda3/envs/）

| 环境 | Python | torch | 用途 | 状态 |
|------|--------|-------|------|------|
| `confumpnn` | 3.11 | 2.2.1+cu121 | **训练/推理/采样** | ✅ 已验证跑通 |
| `confumpnn-esmfold` | 3.10 | 2.6.0+cu124 | **ESMFold 回折验证** | ✅ 可用 |

### 已确认的环境要点
- `confumpnn` 环境**不需要 dgl、torchvision、torchaudio**。之前 torchvision 0.21.0 / torchaudio 2.6.0 与 torch 2.2.1 不匹配导致 import 崩溃，已卸载。若装新包注意不要引入它们。
- LigandMPNN 需要：torch 2.2.1、biopython 1.79、numpy 1.23.5、scipy 1.12.0、prody 2.4.1、networkx、`dm-tree`（import 名是 `tree`）。propka 3.5.1 也已装。
- `confumpnn-esmfold` 用 fair-esm 2.0.0 的 `esm.pretrained.esmfold_v1()`（自带 ESMFold），需要 openfold、fairscale、einops、pytorch_lightning。**nvcc 系统版是 CUDA 11.5，与 torch cu124 不匹配**——编译 openfold 必须在 env 里装 conda cuda-toolkit 并用 `pip install --no-build-isolation openfold`。

## 常用命令

### 条件采样（主线，v7/v9 双编码器）

```bash
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code

# v7（无配体/小蛋白）：MoMPNN 权重 + v7 编码器
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../output/finetune_v7/condition_encoder_last.pt

# v9（配体模式）：LigandMPNN 权重 + v9 编码器
python run_guided.py --pdb ../data/validation_pdbs/1AZM.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../output/finetune_ligand_v9/finetune_epoch030.pt \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
```

### 训练（如需重新微调）

```bash
# v7
python train_finetune.py --device cuda:0 --epochs 30 \
  --labels ../data/cath/labels_balanced_v7.npz --dompdb ../data/cath/S40/dompdb \
  --curriculum --perturb_scale 2.0 --curriculum_scale_max 8.0 \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --out_dir ../output/finetune_v7

# v9（--ligand 开关）
python train_finetune.py --device cuda:0 --epochs 30 --ligand \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --labels ../data/ligand_train/labels.npz --dompdb ../data/ligand_train/all_pdb \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --out_dir ../output/finetune_ligand_v9
```

## 文件结构
- **文件管理规范**：所有文件分类存放遵循 `index/FILE_MANAGEMENT.md`，文档定位见 `index/DOCUMENT_INDEX.md`
- **权威指南**：`WORKFLOW_GUIDE.md`（根目录）——框架/数据流/参数/损失/为什么，面向计算机新人
- **新机配置**：`docs/SETUP_NEW_MACHINE.md`（权重下载/环境/数据重建/验证）
- **数据组织**：`data/README.md`（数据划分/重建命令/SHA256 清单）
- **计划/判据**：`index/PROJECT_PLAN.md`、`index/PROJECT_EXTEND.md`、`index/DESIGN_CRITERIA.md`
- **实验报告**：`analysis/report/`（E1 → v9 泛化验证完整链）
- `LigandMPNN/`、`MoMPNN/`、`foundry/` — 克隆的外部源码（含权重），未跟踪，不提交
- `code/`、`analysis/`、`literature/`、`session/`、`source/` — 按 `index/FILE_MANAGEMENT.md` 分类存放

## Git 说明
- GitHub 远程：`origin` = git@github.com:Yu-Bk/ConfuMPNN.git（另有冗余 remote `new`，同为 https，可删）
- `LigandMPNN/`、`MoMPNN/`、`foundry/` 是克隆源码，**未跟踪**，不应提交（已在 .gitignore）
- **模型权重（*.pt/*.ckpt）与数据（data/）均在 .gitignore**——v7/v9 自训编码器从 GitHub Releases 下载（`gh release download preview1.0.0`），数据从组内 NAS 恢复或重建脚本重跑
