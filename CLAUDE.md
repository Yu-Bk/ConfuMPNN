# ConfuMPNN — 项目说明

## 项目概述
将蛋白质在特定 pH 环境下的理化性质（净电荷、局部电荷分布）作为条件约束，整合到基于结构的蛋白序列生成流程（LigandMPNN 逆折叠）中。核心创新：**在显式建模配体原子上下文的结构条件逆折叠模型上，首次加入 pH 感知的电荷条件控制**。

**当前状态（2026-09-05）**：状态/版本史/报告全表以 **`index/DOCUMENT_INDEX.md`** 与 `README.md` 为准；模型版本史见 `analysis/report/2026-09-05_{protein_history_vs_ligand_deletion,ligand_history_v13_v14}.md`。
- **蛋白模式当前最优 = v12.2**（MoMPNN backbone，无配体/小蛋白；v12.1 + λ_target 表面电荷锚）：**完整验证链达标**（slope 1.00、泛化 per-protein H2 72%、小样本 74%、S2 0/50、无过拟合）。训练 `output/finetune_v12_2/finetune_epoch030.pt`。v12.3 = 长蛋白/深负外推增强（覆盖内略退，按需选用）。
- **配体模式当前最优 = v14**（LigandMPNN RNA/DNA 扩充 414 + A1 全局化，`output/finetune_ligand_v14_rna/finetune_epoch050.pt`）：clean 测试链校准后 **H2 45/50(90%)、H1·H3 50/50、S2 0/50**；**⚠️ 已知局限=组成删减 0.43-0.69×（未愈，机制+配方见 `analysis/report/2026-09-04_paper_subconclusions.md`）**。v13（A1 口袋 keep）未达标被取代。
- **校准三口径（使用前必读，2026-08-31 定稿）**：① per-protein（17 蛋白校准表 `charge_calibration_v12_2.json` 内）→ 72%；② **表外蛋白先小样本现场标定**（`build_calibration_small.py`，**默认 n_per=10 采 50 条**拟合自身 slope，带 LOOCV 稳定性校验）→ 74%；③ 不标定用 global → 40-44%（固有上限）。**校准自动启用**：`run_guided.py` 默认校准表已改为 `charge_calibration_v12_2.json`，`--calibrate auto`（默认）表内 per-protein、表外回退 global；无需手动 `--calibration_file`。**⚠️ 小样本 n_per 不要加大到 20**（n20 对高方差蛋白 1BJ4/2FEO 反而退化——更接近诊断真值但校准反推落到响应弯曲段→命中率降；n10 的"过度拟合"碰巧补偿弯曲）。**响应弯曲蛋白**（长蛋白极端负电区，LOOCV 大）→ 自动标记 unreliable 回退 global，不是加大 n_per
- **v9 迁移待定**（用户暂缓）：LigandMPNN 配体模式重训（`--ligand --v12_supervision` + λ_target 配体适配 + 配体校准表）未启动。v9（配体模式）仍用旧边界：正电到 +8、负电保守到 −5、长序列需检查
- 验证链报告：`analysis/report/2026-08-31_v12_2_{training,diag,tm_sol,summary}.md`；计划 `index/PROJECT_LOCAL_V12_2.md`（§6 自动执行流程 + §6E 无泄露补跑）
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
