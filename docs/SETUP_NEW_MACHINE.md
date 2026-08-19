# ConfuMPNN 新机器配置指南（从零复现）

> 目标：在一台全新 Linux + NVIDIA GPU 机器上，完整复现 ConfuMPNN（含 v7/v9 电荷微调），并验证能生成电荷受控的序列。
> 前置阅读：`WORKFLOW_GUIDE.md`（理解框架）；本文只讲"怎么搭起来"。
> 更新时间：2026-08-19（v9 定稿）。

---

## 0. 需要什么

| 项 | 要求 |
|----|------|
| 操作系统 | Linux（本文档按 Ubuntu 系）|
| GPU | NVIDIA + CUDA 驱动（推理也可 CPU，但很慢）|
| 软件 | Git、conda（Miniconda 推荐）、GitHub CLI（`gh`，用于下载自训编码器）|

**总览：整个项目需要 4 类权重 + 可选的数据**：

| # | 权重/数据 | 来源 | 大小 | 放哪 |
|---|----------|------|------|------|
| 1 | LigandMPNN 权重（15 个 .pt）| `git clone` 自带 | ~120MB | `LigandMPNN/model_params/` |
| 2 | MoMPNN 权重（8 个 .ckpt）| `git clone` 自带 | ~52MB | `MoMPNN/mompnn_paper_checkpoints/` |
| 3 | **v7/v9 自训编码器** | **GitHub Releases 下载** | ~1.2MB | `code/weights/` |
| 4 | ESMFold 权重 | 首次运行自动下载 | ~数 GB | `~/.cache/`（自动）|
| 5 | 训练/验证数据 | NAS 恢复 或 重建脚本 | 8GB | `data/` |

> 🔑 **关键理解**：`ConfuMPNN` 仓库（git clone 得到）里**没有**权重文件（`.pt`/`.ckpt` 被 `.gitignore` 排除）。其中 1、2 号权重随"外部源码 clone"一起拿到；**3 号（v7/v9 自训编码器）是项目自己的训练产物，必须从 GitHub Releases 单独下载**。

---

## 1. 克隆仓库 + 外部源码（含 1、2 号权重）

```bash
# 1) 项目仓库（含全部代码/文档/脚本）
git clone git@github.com:Yu-Bk/ConfuMPNN.git
cd ConfuMPNN

# 2) LigandMPNN 官方源码（自带 model_params/ 的 15 个权重）
git clone https://github.com/dauparas/LigandMPNN.git

# 3) MoMPNN 官方权重（ProtAlign 多目标 DPO 微调版）
git clone https://github.com/Qivon7/MoMPNN.git
```

**验证**（确认权重文件在位）：
```bash
ls LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt   # 配体模式 backbone
ls MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt  # 无配体 backbone
```

---

## 2. 下载 v7/v9 自训编码器（3 号权重，GitHub Releases）

两个编码器是 ConfuMPNN 微调训练的核心交付物：
- **v7**（`condition_encoder_v7_last.pt`，296K）：MoMPNN backbone，无配体/小蛋白
- **v9**（`condition_encoder_v9_epoch030.pt`，887K）：LigandMPNN backbone，配体模式

```bash
# 确保 gh 已登录（gh auth login）
gh auth status

# 查看 release 附件
gh release view v1.0.0

# 下载两个编码器权重到 code/weights/
mkdir -p code/weights
gh release download v1.0.0 --pattern "condition_encoder*.pt" -D code/weights/

# 校验完整性（SHA256）
cd code/weights
sha256sum -c SHA256SUMS.txt
```

**如果不用 gh**（无 GitHub CLI），也可直接浏览器访问仓库页面的 Releases 下载。

---

## 3. 创建 conda 环境

### 3.1 主环境 `confumpnn`（训练/推理/采样）

```bash
conda create -n confumpnn python=3.11 -y
conda activate confumpnn
conda install pytorch==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install biopython==1.79 numpy==1.23.5 scipy==1.12.0 prody==2.4.1 networkx dm-tree propka==3.5.1
```

> ⚠️ **不要装** torchvision / torchaudio / dgl——LigandMPNN 不需要，且曾与 torch 版本不匹配导致 import 崩溃（历史教训）。

### 3.2 打分环境 `confumpnn-esmfold`（ESMFold 回折验证，选装但推荐）

```bash
conda create -n confumpnn-esmfold python=3.10 -y
conda activate confumpnn-esmfold
conda install pytorch==2.6.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c nvidia cuda-toolkit -y      # 编译 openfold 需要本地 nvcc
pip install fair-esm==2.0.0 openfold==2.0.0 fairscale einops pytorch_lightning
```

> ⚠️ 系统级 nvcc 通常是 CUDA 11.5，与 torch cu124 不匹配。必须在 env 内装 cuda-toolkit，并用 `pip install --no-build-isolation openfold` 编译。

---

## 4. 验证安装（不需要数据即可跑）

```bash
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
conda activate confumpnn

# 4.1 单元测试（36 项，全部通过）
python tests/test_all.py

# 4.2 冒烟：v7 编码器加载 + 电荷控制（1BC8，target 0）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../code/weights/condition_encoder_v7_last.pt \
  --num_samples 3
```

**预期**：终端输出每条序列的 charge ≈ 0（|dev| ≤ 2.0 为达标）、pI 合理；`output/guided_1BC8_pH7.4/seqs.fa` 生成。

```bash
# 4.3 冒烟：v9 编码器 + 配体模式（用任意含配体的 PDB，如 data/validation_pdbs/1AZM.pdb）
python run_guided.py --pdb ../data/validation_pdbs/1AZM.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../code/weights/condition_encoder_v9_epoch030.pt \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --num_samples 3
```

> 若 `data/validation_pdbs/` 还没恢复，4.3 可跳过（它需要配体 PDB）。仅验证 v7 即可确认安装成功。

---

## 5. 数据重建（可选：需要重新训练或完整验证时）

**推理生成不需要训练数据。** 只有两种情况需要数据：
- **重新训练/微调编码器** → 需要训练数据（CATH + 配体）
- **做完整验证**（ESMFold 回折判定）→ 需要验证集 PDB

### 5.1 从组内 NAS 恢复（最快，推荐）

本项目 8GB 数据已打包备份在组内 NAS（路径见 `data/README.md` 或与项目负责人确认）。恢复：

```bash
# 下载 tar 包到项目旁，解压到 data/
tar -xzf confumpnn_data_v1.tar.gz -C /data/nfs/IC/baokun_yu/ConfuMPNN/
sha256sum -c data/SHA256SUMS.txt    # 校验
```

### 5.2 重建训练数据（无备份时）

| 数据集 | 重建命令 | 产物 |
|--------|---------|------|
| CATH S40（v7 训练）| 见下方 | `data/cath/S40/dompdb` |
| v7 标签 | `python code/tests/build_labels_v2.py --class_balance --per_class 2500 ...` | `data/cath/labels_balanced_v7.npz` |
| 配体复合物（v9 训练）| `python code/tests/fetch_ligand_pdbs.py --sampled 15000 ...` | `data/ligand_train/{small_mol,metal,rna,dna}` |
| v9 标签 | `python code/tests/build_ligand_labels.py --dompdb data/ligand_train/all_pdb --out data/ligand_train/labels.npz` | `data/ligand_train/labels.npz` |

CATH S40 下载（818MB）：
```bash
mkdir -p data/cath && cd data/cath
curl -O https://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.list
curl -O https://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.fa
python ../../code/tests/parallel_download.py \
  https://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz \
  cath-dataset-nonredundant-S40.pdb.tgz 8
mkdir -p S40 && tar -xzf cath-dataset-nonredundant-S40.pdb.tgz -C S40
```

### 5.3 验证集 PDB（小，10 个蛋白）

清单在 `data/validation_pdbs/validation_manifest.json`（含 PDB ID + 路径）。可从 RCSB 逐个下载，或直接从 NAS 恢复（推荐）。

---

## 6. 重新训练（如需微调自己的编码器）

```bash
# v7（MoMPNN，无配体）
python train_finetune.py --device cuda:0 --epochs 30 \
  --labels ../data/cath/labels_balanced_v7.npz --dompdb ../data/cath/S40/dompdb \
  --curriculum --perturb_scale 2.0 --curriculum_scale_max 8.0 \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --out_dir ../output/finetune_v7

# v9（LigandMPNN 配体模式）
python train_finetune.py --device cuda:0 --epochs 30 --ligand \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --labels ../data/ligand_train/labels.npz --dompdb ../data/ligand_train/all_pdb \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --out_dir ../output/finetune_ligand_v9
```

> 每个参数的含义和为什么，见 `WORKFLOW_GUIDE.md` §5（损失）和 §6（参数）。

---

## 7. 完整验证管线（ESMFold 回折 + TM-score）

```bash
# 1) ESMFold 回折（confumpnn-esmfold 环境，首次运行自动下载 ESMFold 权重）
conda activate confumpnn-esmfold
python code/tests/esmfold_score.py --fasta <seqs.fa> --out output/fold --device cuda:0

# 2) US-align TM-score（对照参考骨架）
conda activate confumpnn
python code/tests/tm_score.py <ref.pdb> <pred.pdb>

# 3) 判定：H1 TM≥0.70、H2 |电荷-target|≤2.0（见 index/DESIGN_CRITERIA.md）
```

---

## 8. 常见问题

| 问题 | 解决 |
|------|------|
| `import tree` 报错 | 装 `dm-tree`（`pip install dm-tree`）|
| `import dgl` 报错 | 主环境**不要**装 dgl |
| MoMPNN 权重加载报错 | `--model_type auto` 会自动识别；确认权重路径正确 |
| `gh release download` 找不到附件 | 先 `gh auth login`；确认 release 名 `v1.0.0` |
| ESMFold 编译 openfold 失败 | 见 §3.2 的 nvcc/cuda-toolkit 要点 |
| 没 GPU | 自动回退 CPU，能跑但很慢 |

---

## 9. 配置核对清单

- [ ] `git clone` 三个仓库（ConfuMPNN / LigandMPNN / MoMPNN）
- [ ] v7/v9 编码器下载到 `code/weights/` 且 SHA256 校验通过
- [ ] conda 环境 `confumpnn` 创建 + 依赖装齐
- [ ] `python tests/test_all.py` 36/36 通过
- [ ] v7 冒烟生成电荷 ≈ target
- [ ] （可选）`confumpnn-esmfold` 环境 + ESMFold 回折验证
- [ ] （需要训练时）数据恢复/重建完成
