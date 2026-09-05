# ConfuMPNN 数据组织说明

> 本文件说明 `data/` 目录的结构、**训练/验证划分逻辑**、重建命令与备份方式。
> 更新时间：2026-09-05（补录 v12.3/v14/7K00/ablation 数据与 SHA256）。
> 数据目录整体被 `.gitignore` 忽略（git 不跟踪），需要单独备份/恢复。大文件 SHA 见同目录 `SHA256SUMS.txt`（含 2026-09-05 补录段）。

---

## 0. 一句话

`data/` 分两类：**训练集**（模型学习用）和**验证集**（泛化检验用，已从训练集排除防泄漏）。当前约 **13GB**，含蛋白 CATH/长蛋白补充、配体复合物（小分子/金属/核苷酸/RNA-DNA 414）、验证集（in-10/805/7K00 核糖体）与各版本 labels npz（详见 `analysis/report/2026-09-05_repo_audit.md` §1.7）。

---

## 1. 目录总览

```
data/
├── cath/                      # 【训练集 - v7】CATH 结构域
│   ├── S40/dompdb/            #   CATH 4.4 S40 非冗余结构域（34,653 个，无扩展名）
│   ├── S40/dompdb_pdb/        #   dompdb 的 .pdb 符号链接（训练脚本自动创建）
│   ├── labels.npz             #   早期标签（999 域 × 8 pH，v2 用）
│   ├── labels_balanced_v5.npz #   分层采样（2,176 域）
│   ├── labels_balanced_v6.npz #   三类平衡（7,208 域）
│   ├── labels_balanced_v7.npz #   ★ v7 最终训练标签（7,886 域 = CATH + 外部碱性）
│   ├── ext_basic_dompdb/      #   外部补充碱性域（781 个，补 CATH 碱性不足）
│   └── ext_basic_pdb/         #   其 .pdb 副本
├── ligand_train/              # 【训练集 - v9】配体复合物
│   ├── small_mol/             #   小分子配体复合物（4,155 个）
│   ├── metal/                 #   金属配体（567 个）
│   ├── rna/                   #   RNA 核苷酸配体（244 个）
│   ├── dna/                   #   DNA 核苷酸配体（6 个，并入 rna 语义但分开存）
│   ├── all_pdb/               #   五类合并（符号链接，训练 --dompdb 用）
│   └── labels.npz             #   ★ v9 最终训练标签（4,972 复合物 × 8 pH）
├── validation_pdbs/           # 【验证集】v9 泛化验证（10 个未见蛋白）
│   ├── validation_manifest.json   #   ★ 蛋白清单（10 个 + 类别 + 配体注释）
│   ├── {PDB}.pdb              #   10 个最终验证蛋白（1C6O/1AZM/1AS2/1AXW/2FEO/5CQH/1CGE/1AG0/1A65/1BJ4）
│   ├── {PDB}_noplig.pdb       #   去配体版（配体消融实验用）
│   └── （其余 672 个 .pdb/.cif = 选蛋白时的候选缓存，**非必要**，可删可留）
├── ligand_test/               # 【验证集】迁移检验（5 个：1FQG/2XUO/3T0F/4DFR/5HVX）
├── transfer_test/             # 【验证集】迁移测试（5 个：1LYZ/1MBN/1TIM/4DFR_chainA）
└── SHA256SUMS.txt             # ★ 关键文件校验清单
```

---

## 2. 划分逻辑（防泄漏）

```
训练集（模型见过）              验证集（模型没见过，全部 --exclude 排除）
─────────────────────         ─────────────────────────────────────────
data/cath/         → v7        data/validation_pdbs/  → v9 泛化验证（10 未见蛋白）
data/ligand_train/ → v9        data/ligand_test/      → 编码器迁移检验（5）
                               data/transfer_test/    → 迁移测试（5）
```

**为什么这样划分**：
- 训练数据与验证数据**无重叠**（验证蛋白曾通过 `--exclude` 从训练集排除，拦截过 1b24A01 进训练集）。
- 验证集覆盖配体五类 + 长序列（小分子/DNA/RNA/金属/长蛋白），检验"未见蛋白"泛化。

---

## 3. 关键文件用途

| 文件 | 用途 | 谁在用 |
|------|------|--------|
| `data/cath/labels_balanced_v7.npz` | v7 训练标签（7,886 域 × 8pH）| `train_finetune.py --labels` |
| `data/cath/S40/dompdb` | v7 训练结构域 | `train_finetune.py --dompdb` |
| `data/ligand_train/labels.npz` | v9 训练标签（4,972 × 8pH）| `train_finetune.py --labels`（配体）|
| `data/ligand_train/all_pdb` | v9 训练结构 | `train_finetune.py --dompdb`（配体）|
| `data/validation_pdbs/validation_manifest.json` | v9 验证蛋白清单 | `validate_generalization.py --manifest` |

---

## 4. 校验（SHA256）

`data/SHA256SUMS.txt` 记录关键文件（训练标签、manifest、验证/测试 PDB）的校验和。验证：

```bash
cd /data/nfs/IC/baokun_yu/ConfuMPNN
sha256sum -c data/SHA256SUMS.txt     # 在 ConfuMPNN 根目录执行（路径按文件内相对根目录）
```

> 34,653 个 CATH 域和 4,972 个配体不做逐个校验（文件量太大）；完整性由 **打包 tar.gz 的 SHA256** 保证（见 §6）。

---

## 5. 重建命令（无备份时）

| 数据集 | 命令 | 耗时 |
|--------|------|------|
| CATH S40 | `curl -O https://download.cathdb.info/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz` + 解压到 `S40/` | ~30 min |
| v7 标签 | `python code/tests/build_labels_v2.py --class_balance --per_class 2500 ...` | ~20 min |
| 配体复合物 | `python code/tests/fetch_ligand_pdbs.py --sampled 15000 --out data/ligand_train --targets rna:1000,dna:1000,small_mol:3000,metal:2500` | ~1-2 h |
| v9 标签 | `python code/tests/build_ligand_labels.py --dompdb data/ligand_train/all_pdb --out data/ligand_train/labels.npz` | ~20 min |
| 验证集 | 从 NAS 恢复（推荐）或按 `validation_manifest.json` 从 RCSB 逐个下载 | 几分钟 |

> **注意**：重建出的标签与原始标签在随机采样上**不完全一致**（pH 采样/域选择有随机性），训练结果会有细微差异但趋势一致。**若要完全复现原始训练，请用 NAS 备份恢复。**

---

## 6. 备份与恢复（NAS）

### 备份（本机 → 组内 NAS）

```bash
# 打包脚本：data/cath + ligand_train + validation_pdbs + ligand_test + transfer_test
bash code/tests/backup_data.sh <输出目录，如 /data/nfs/IC/baokun_yu/ConfuMPNN_backup/>
# 产物：confumpnn_data_v1.tar.gz + SHA256SUMS.txt（tar 包校验和）
# 然后手动上传 tar.gz 到组内 NAS 共享路径（如 /data/nfs/.../ConfuMPNN_data/）
```

### 恢复（NAS → 新机器）

```bash
tar -xzf confumpnn_data_v1.tar.gz -C /data/nfs/IC/baokun_yu/ConfuMPNN/
cd /data/nfs/IC/baokun_yu/ConfuMPNN
sha256sum -c confumpnn_data_v1.tar.gz.sha256   # 校验 tar 包
sha256sum -c data/SHA256SUMS.txt               # 校验解压后的关键文件
```

> **为什么数据必须备份**：data/ 在 `.gitignore`（git 不跟踪），clone 仓库不会带来任何数据。CATH 下载和配体重拉都很耗时，NAS 备份是唯一"完全复现"的途径。
