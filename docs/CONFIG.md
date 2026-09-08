# ConfuMPNN 配置文档（参考）

> **权威完整指南**：`WORKFLOW_GUIDE.md`（根目录 §6 参数全表）。本文档为配置速查。
> 更新：2026-09-06 —— **当前交付 = v12.2（蛋白/MoMPNN）、v14（配体/LigandMPNN RNA-DNA）**；v7/v9 = 历史。
> ⚠️ §二"校准 enabled=false"为 v9 时代旧况：**v12.2/v14 使用侧默认启用校准**（`run_guided.py --calibrate auto`，默认表 `charge_calibration_v12_2.json`）。

---

## 一、结构过滤器预设（`code/configs/filter_presets.yaml`）

4 条空间规则（阈值来自 CATH S40 统计的 99 分位）：

| 规则键 | 检测内容 | 阈值 |
|--------|---------|------|
| `charge_cluster` | 10Å 内同号强电荷（K/R 或 D/E）| 6 |
| `salt_bridge` | 10Å 内正负电荷对 | 4 |
| `core_charge` | 核心埋藏位置 8Å 内带电残基 | 6 |
| `same_sign_cluster` | 8Å 邻域同号电荷 | 4 |

4 个预设（`--preset`）：`default` / `nucleic_acid_binding`（正电更宽容）/ `membrane`（核心严格禁电）/ `acidic`。

---

## 二、条件默认配置（`code/configs/condition_defaults.yaml`）

条件向量 `[7]`（mask-aware）：`[pH, has_charge_flag, charge_val, has_pos_flag, pos_val, has_neg_flag, neg_val]`

| 键 | 含义 | 当前值 |
|----|------|--------|
| `cond_dim` | 条件向量维度 | 7 |
| `pH_min` / `pH_max` | 训练 pH 采样范围 | 4.0 / 10.0 |
| `normalization.mean` / `.std` | 每维标准化常量（训练集统计）| pH 均值 6.9982、电荷均值 1.4243（**已填，非 null**）|
| `encoder.hidden_dim` / `token_dim` / `n_tokens` | 编码器结构 | 64 / 128 / 4 |
| `charge_calibration.gain` / `.offset` / `.enabled` | 电荷校准 | **1.289 / 0.74 / false** |

⚠️ **校准现状（重要）**：`enabled: false`——过冲已由训练侧 `charge_temp=0.5` 根治（v9 起），推理侧线性校准不再需要。历史：早期过冲 ~2.9× 曾用推理侧校准（gain=2.57）补偿。

---

## 三、训练参数（`train_finetune.py`）

### 3.1 通用（蛋白/配体共用）

| 参数 | 默认 | 含义 |
|------|------|------|
| `--weights` | MoMPNN 权重 | backbone 权重 |
| `--ligand` | 关 | 配体模式：LigandMPNN 权重 + 配体上下文（atom_context=25）|
| `--lr` | 1e-3 | 学习率 |
| `--epochs` | 30 | 训练轮数（v14 用 50）|
| `--lambda_c` | 0.5 | 电荷损失权重 |
| `--lambda_kl` | 0.05 | KL 锚权重 |
| `--lambda_keep` | 0.5 | 序列保持权重（用户固定）|
| `--perturb_prob` | 0.3 | 扰动样本比例（70/30 混合目标）|
| `--perturb_scale` | 4.0 | 扰动幅度上限 |
| `--placeholder_prob` | 0.15 | 占位符样本比例 |
| `--charge_temp` | 0.5 | 电荷损失温度（根治过冲）|
| `--max_domains` | 0 | 冒烟测试用（前 N 域）|

### 3.1b v12 系监督参数（**当前配方 v12.2/v12.3/v14 必用**；v7/v9 无）

| 参数 | v12.2/v12.3 蛋白值 | v14 配体值 | 含义 |
|------|---|---|---|
| `--v12_supervision` | on | on | 组成双计数 + 表面 GRAVY 监督（防删减）|
| `--frac_floor` | 0.5 | 0.5 | 表面 D/E/K/R 保留下限 |
| `--gravy_margin` | 0.4 | 0.4 | 表面疏水 margin |
| `--lambda_v12` | 0.2 | 0.2 | 组成/GRAVY 权重 |
| `--lambda_target` | 0.2 | 0.2 | 表面电荷锚（target−核心电荷）|
| `--sasa_threshold` | 0.25 | 0.25 | 表面定义 frac_sasa 阈值 |
| `--ph_aware_filter` / `--structure_boost` | on / 1.5 | on / 1.5 | pH 自适应结构惩罚 |
| `--decouple_perturb` + `--decouple_range 12.0` | v12.3 用 | — | 蛋白：扰动解耦 |
| `--decouple_absolute` | — | v14 用 | 配体：绝对解耦 |
| `--pocket_mode` | —(蛋白无配体=不启) | `global`（floor0.8/ceil1.3/λ_pocket 0.3）| A1 双向计数（护带电残基总数）|

### 3.2 当前实际训练命令（v12.2 蛋白 / v14 配体）

```bash
# v12.2 蛋白（MoMPNN；更全命令见 log/v12_2_train_mompnn.log）
python code/train_finetune.py --device cuda:0 --epochs 30 \
  --labels data/cath/labels_v12_2_train.npz --dompdb data/cath/S40/dompdb \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 \
  --sasa_threshold 0.25 --ph_aware_filter --structure_boost 1.5 \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --out_dir output/finetune_v12_2

# v14 配体（LigandMPNN RNA/DNA+A1-global；atom25）
python code/train_finetune.py --device cuda:0 --epochs 50 --ligand \
  --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --labels data/ligand_train/labels_v14_final.npz --dompdb data/ligand_train/all_pdb \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 \
  --sasa_threshold 0.25 --ph_aware_filter --structure_boost 1.5 \
  --pocket_mode global --pocket_floor 0.8 --pocket_ceil 1.3 --lambda_pocket 0.3 \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --out_dir output/finetune_ligand_v14_rna
```

（历史 v7/v9 训练命令保留于 SETUP_NEW_MACHINE.md §6 或 git 历史；不再作当前推荐。）

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

---

## 四、采样参数（`run_guided.py`）

| 参数 | 默认 | 含义 |
|------|------|------|
| `--pdb` | 必填 | 输入 PDB |
| `--pH` | 必填 | 工作环境 pH |
| `--target_charge` | None | 目标净电荷（None=只结构过滤）|
| `--cond_encoder` | None | **自训编码器权重**（蛋白 v12.2 / 配体 v14；给了走条件注入）|
| `--cond_mode` | conditioned | conditioned=注入 / baseline=加载不注入（对照）|
| `--weights` | MoMPNN | backbone 权重（配体模式用 LigandMPNN 权重）|
| `--temperature` | 0.3 | 采样温度 |
| `--num_samples` | 10 | 候选序列数 |
| `--fixed_residues` | None | 固定残基（如 `'A12 C15'`）|
| `--preset` | default | 结构过滤器预设 |
| `--seed` | 111 | 随机种子 |
| `--out_dir` | 自动 | 输出目录 |
| `--calibrate` | auto | **off/auto/global**：auto=表内 per-protein、表外回退 global |
| `--calibration_file` | 表 | 校准表（蛋白 `charge_calibration_v12_2.json`；配体用对应 `charge_calibration_v14_ligand_clean.json`）|
| `--num_ligand_atoms` | 25 | 配体原子上下文（配体模式；权重为 25，勿改 16）|

**权重自动检测**：ckpt 含 `atom_context_num`(>0) → ligand_mpnn；否则 protein_mpnn（MoMPNN）。
> 配体模式采样/验证请显式给 `--calibration_file`（见各 clean 链命令，勿用默认蛋白表）。

---

## 五、环境配置

| 项 | 值 | 说明 |
|----|-----|------|
| 主环境 | `confumpnn`（Python 3.11, torch 2.2.1+cu121）| 训练/推理/采样 |
| ESMFold | `confumpnn-esmfold`（torch 2.6.0+cu124, fair-esm 2.0.0）| 回折验证（TM-score）|
| TemBERTure | `confumpnn-temberture`（torch CPU）| 热稳定打分 |
| Protein-Sol | 系统 python + Perl | 可溶打分 |

> ⚠️ 主环境不要装 torchvision/torchaudio/dgl（曾致 import 崩溃）。新机配置见 `docs/SETUP_NEW_MACHINE.md`。
