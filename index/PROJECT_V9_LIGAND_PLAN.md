# v9 训练计划 — 在 LigandMPNN backbone 上微调条件编码器（五类配体数据）

> 日期：2026-08-18
> 状态：**计划定稿，待执行**（数据清洗/训练在计划后开始）
> 关联：`session/2026-08-18_v9_ligand_plan.md`（早期草稿，以本文件为准）
> 背景链：迁移检验定位「LigandMPNN 配体模式电荷失效」（`analysis/report/2026-08-18_seq_sanity_and_transfer.md`）——根因 = 条件编码器只在 MoMPNN backbone 上训练，LigandMPNN 的 h_V 特征分布不匹配。v9 = 用配体复合物数据在 LigandMPNN backbone 上重训同一 `ConditionEncoder`。

---

## 一、目标与验收标准

### 1.1 目标
训练一个在 **LigandMPNN backbone + 配体原子上下文** 下电荷控制可靠的 `ConditionEncoder`（74880 参数，架构不变），替换 v7 编码器用于配体场景。

### 1.2 验收标准（对齐 `index/DESIGN_CRITERIA.md` v2）

| 判据 | 阈值 | 当前值 | 依据 |
|------|------|--------|------|
| H2 电荷命中 | dev ≤ 2.0 | 1MBN pH7.4 dev **14.05** ❌ | 迁移检验，v9 核心修复目标 |
| H1 结构自洽 | TM ≥ 0.70，失败率 ≤ 10% | 1MBN/4DFR 已 100% 折回 | 配体模式不破坏折叠 |
| 迁移复验 | 1MBN / 4DFR / 1FQG | dev 14.05 / 2.71 / 0.97(巧合) | 三个蛋白全部复验 |

> **注意 1FQG**：v7 下 dev 0.97 是「负电 target 接近 LigandMPNN 默认偏好」的巧合假阳性，**必须**用 1MBN（+2 需大幅正电化）才能区分真假修复。

## 二、数据设计（五类配体）

### 2.1 类别与规模（用户要求：比 MoMPNN 7886 域稍多）

总量目标 **~9000 复合物**。五类分目录、分别取数、**合并训练**（一个编码器见过所有配体类型）。

**实测数据现实（2026-08-18，用户已确认以下决策）**：
- 单链蛋白+有配体+高分辨池 113,973 个，本地按 HETATM 分类：**小分子 4389 + 金属 568 + RNA 258 + DNA 6**（L≤500）
- **RNA/DNA 配体天然稀缺**（结合蛋白多为多链/含核酸链）→ **DNA 并入 RNA 合成核酸类**（264 个，同为核苷酸配体）
- **多结合水不算配体**（LigandMPNN parse_PDB 显式排除水）→ **不单独分类** ✓
- **跳过含 X 残基的序列**（非标准残基，标签构建已实现）
- **L≤500**（用户确认）：v9 覆盖更大蛋白，与 v7（L≤300）分工 → **v7 小序列专精，v9 大序列覆盖**

**最终数据方案**：
| 类别 | 实际数量 | 说明 |
|------|---------|------|
| 小分子 | 4389 | 主流（药物/辅因子/代谢物）|
| 金属 | 568 | 金属配位 |
| 核酸（RNA+DNA）| 264 | RNA/DNA 核苷酸，合并 |
| **合计** | **~5200** | 全部 L≤500，×8 pH = 41,600 样本 |

### 2.2 候选池过滤条件（RCSB 搜索 API，已验证可用）

- 分辨率 ≤ 2.5 Å
- 单链蛋白：`polymer_entity_count=1`、`polymer_entity_count_protein=1`、`polymer_entity_count_nucleic_acid=0`
- 有非聚合物配体：`nonpolymer_entity_count ≥ 1`
- **候选池规模：113,995 个**（实测，远超需求）
- 长度 L ≤ 300（对齐条件化可靠范围；可在本地过滤）

### 2.3 五类配体的分类依据

RCSB chemcomp 类型（本地用 data API 抽查确认）+ PDB HETATM 残基名。**以本地 HETATM 残基名分类为准**（与训练脚本 `parse_PDB` 处理逻辑一致，不依赖 RCSB 搜索属性名的易变细节）。

## 三、目录与路径（按 `index/FILE_MANAGEMENT.md`）

```
ConfuMPNN/
├── code/tests/
│   ├── fetch_ligand_pdbs.py      # RCSB 搜索 API → 下载 → 本地五类分类
│   ├── build_ligand_labels.py    # 配体复合物 → labels.npz
│   └── classify_ligand.py        # HETATM 残基名 → 五类（fetch 复用）
├── data/ligand_train/
│   ├── rna/  dna/  small_mol/  metal/  water/   # 五类 PDB（分类存放）
│   ├── all_pdb/                                  # 五类合并（symlink，供训练 --dompdb）
│   ├── candidates.json          # 候选 ID 清单（含过滤元数据）
│   └── labels.npz               # 训练标签（domain_ids/seqs/coords/pH/charge/pI）
├── output/finetune_ligand_v9/   # 训练 checkpoint（每 epoch 一个）
├── log/v9_train.log             # 训练日志（train_finetune.py --log_file）
└── analysis/report/2026-08-18_v9_ligand_training.md   # 训练后报告
```

> 数据放 `data/`（git 忽略，规则 1 允许输入数据放 `code/input` 或项目数据区；本项目数据惯例在 `data/`）。脚本 `code/tests/`。报告 `analysis/report/`。计划 `index/`。

## 四、训练改造（最小改动）

### 4.1 `code/train_finetune.py` 加 `--ligand` 开关（2 处）

1. **`load_backbone()`**：`--ligand` 时用 `LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`（已在仓库，10.5MB），
   `atom_context_num=16, model_type="ligand_mpnn", ligand_mpnn_use_side_chain_context=0`（复用 `run_guided.py:104-131` 自动检测逻辑）。
2. **数据加载循环**（当前第 365 行硬编码）：`--ligand` 时 featurize 改
   `use_atom_context=True, number_of_ligand_atoms=16, model_type="ligand_mpnn"`。

**其余不动**：`parse_PDB` 已支持配体原子（Y/Y_t/Y_m，`LigandMPNN/data_utils.py:834-858`）；
`build_domain` / `decoder_forward` / 损失函数（CE+电荷+KL+keep）全与配体无关。

### 4.2 训练命令（冒烟 → 正式）

```bash
# 冒烟（50 域，验证 ligand 前向）
PYTHONPATH=/data/nfs/IC/baokun_yu/ConfuMPNN/code \
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python \
  /data/nfs/IC/baokun_yu/ConfuMPNN/code/train_finetune.py \
    --weights /data/nfs/IC/baokun_yu/ConfuMPNN/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
    --ligand \
    --labels /data/nfs/IC/baokun_yu/ConfuMPNN/data/ligand_train/labels.npz \
    --dompdb /data/nfs/IC/baokun_yu/ConfuMPNN/data/ligand_train/all_pdb \
    --out_dir /data/nfs/IC/baokun_yu/ConfuMPNN/output/finetune_ligand_v9 \
    --device cuda:3 --epochs 30 --max_domains 50
# 正式（去掉 --max_domains；nohup 后台）
```

### 4.3 训练超参（沿用 v7 已验证配置，不引入新变量）

`λ_c=0.5 λ_kl=0.05 λ_keep=0.5 perturb_prob=0.3 perturb_scale=8(课程) charge_temp=0.5`
——**不调 λ_keep**（用户明确：λ_keep 保持 0.5）。

## 五、执行步骤

| 步骤 | 内容 | 预计耗时 |
|------|------|---------|
| 1 | RCSB 搜索 API 拉候选 ID（113,995 池）→ 随机采样 ~12000 | ~5 min |
| 2 | 批量下载 PDB（并行 curl）→ 本地 HETATM 分类五类 → 各取目标数 | ~1-2 h |
| 3 | `build_ligand_labels.py` 生成 labels.npz（8 pH 连续采样） | ~20 min |
| 4 | `train_finetune.py` 加 `--ligand` + 冒烟（50 域） | ~10 min |
| 5 | 后台正式训练 30 epoch（预估 ~60-90 min，cuda:3） | ~1.5 h |
| 6 | 迁移复验（1MBN/4DFR/1FQG）+ 报告 | ~30 min |

## 六、风险与缓解

| 风险 | 缓解 |
|------|------|
| 配体复合物 parse 失败率高（非标准残基/修饰） | 分类时跳过坏域（现有坏域跳过机制）；每类冗余采样 ~20% |
| 类别重叠/重复 | 「主配体类型」去重（§2.1），一个结构只进一类 |
| 某类候选不足（如 RNA 配体少） | 降低该目标数、从小分子/金属池补（弹性调配） |
| ligand 模式训练显存/时间超预期 | 冻结 backbone + 只训 74880 参数；冒烟先验证 |
| 训练不收敛（配体特征干扰） | 先看 v7 同款损失曲线（ce/charge/keep）；不收敛再降 lr |

## 七、预期收益

- **配体模式电荷控制恢复**：1MBN dev 14.05 → ≤2（核心）
- **不破坏**：MoMPNN 无配体模式（该模式仍用 v7 编码器，两条线并存）
- **解锁**：用户配体场景（固定口袋 + 电荷控制 真正可用，非巧合）

## 八、决策记录

1. 数据按五类配体**分别取数、合并训练**一个编码器（用户要求"分别取数据" = 各类覆盖全，非五模型）。
2. 配体类型分类**以本地 HETATM 残基名判定**（与训练特征化一致，避开 RCSB 搜索属性名易变性）。
3. 数据量 ~9000 > MoMPNN 7886（用户要求"稍多"）。
4. **不调 λ_keep**（用户明确）；训练超参沿用 v7 已验证配置。
5. v9 编码器与 v7 并存：无配体用 v7，配体用 v9，`run_guided.py --cond_encoder` 可切换。
6. **v9 定稿（2026-08-19 用户确认）**：泛化验证完成（10 未见蛋白 × 5 电荷臂 × n30，见 `analysis/report/2026-08-19_v9_generalization_validation.md`）——折叠泛化可靠（7 健康蛋白 35 臂 TM≥0.7）、温和区与极端正电+8 可靠、极端负电−8 弱（欠冲，40%）。**停止训练，不再规划新轮次**；使用按 `2026-08-18_model_charge_limits.md` §8 边界（配体模式：正电可用到 +8、负电保守到 −5、长序列 L≥470 需检查）。电荷控制根因 = 模型无差别减少带电残基总数（收敛低电荷密度），靠不对称删减（负电多删 K/R、正电多删 D/E）逼近 target。
