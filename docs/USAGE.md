# ConfuMPNN 使用说明（参考）

> **权威完整指南**：`WORKFLOW_GUIDE.md`（根目录，含命令速查 §7）。本文档为使用速查。
> 更新至 v9 定稿（2026-08-19）。

---

## 一、环境准备

```bash
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
```

---

## 二、快速开始（条件采样，主线）

### v7（无配体/小蛋白，MoMPNN backbone）

```bash
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../output/finetune_v7/condition_encoder_last.pt \
  --num_samples 10
```

### v9（配体模式，LigandMPNN backbone）

```bash
python run_guided.py --pdb ../data/validation_pdbs/1AZM.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../output/finetune_ligand_v9/finetune_epoch030.pt \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --num_samples 10
```

输出（终端）：每条的 charge/pI + native 对照 + 均值统计；结果写 `output/guided_<pdb>_pH<pH>/`（`seqs.fa` + `summary.json`）。

---

## 三、典型场景

### 场景 1：指定目标净电荷

```bash
# 负电设计（v7 强项，可靠区 native−3~−8）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge -8 \
  --cond_encoder ../output/finetune_v7/condition_encoder_last.pt

# 正电设计（v9 强项，可到 native+8）
python run_guided.py --pdb ../data/validation_pdbs/1AZM.pdb --pH 7.4 --target_charge 6 \
  --cond_encoder ../output/finetune_ligand_v9/finetune_epoch030.pt \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
```

### 场景 2：固定结合位点（保留配体口袋）

```bash
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../output/finetune_v7/condition_encoder_last.pt \
  --fixed_residues "A12 C15"       # 这些位氨基酸不变，其余由模型设计
```

### 场景 3：引导采样路线（Level 1，对照/快速原型）

```bash
# 不加 --cond_encoder → 走 Phase 1 引导采样（电荷 lookahead + 结构过滤）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 --preset default
```

⚠️ 此路线模型自身不感知 pH（诚实边界），电荷引导由解码时 bias 完成。

### 场景 4：批量验证（v9 泛化管线）

```bash
PYTHONPATH=code python code/tests/ligand_v9/validate_generalization.py \
  --manifest data/validation_pdbs/validation_manifest.json \
  --out_dir output/generalization_v9 --mode both \
  --cond_encoder output/finetune_ligand_v9/finetune_epoch030.pt \
  --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --n 30 --device cuda:0 --pH 7.4
```

---

## 四、输出解读

**seqs.fa**：每条 `>sample_N pH=... charge=... pI=...` + 序列；末尾 native 对照。
- `charge`：该序列在指定 pH 的净电荷（HH 平滑计算）
- `pI`：等电点（二分搜索）

**summary.json**：运行参数 + 每序列 seq/charge/pI + 均值/标准差。

---

## 五、下游验证（可选）

| 目标 | 工具 | 环境 | 脚本 |
|------|------|------|------|
| 可设计性（能否折叠回骨架）| ESMFold pLDDT + US-align TM-score | `confumpnn-esmfold` + `confumpnn` | `esmfold_score.py` + `tm_score.py` |
| 可溶性 | Protein-Sol %sol | 系统 python + Perl | `protein_sol_mcp/scripts/protein_sol_predict.py` |
| 热稳定性 | TemBERTure Tm | `confumpnn-temberture` | `temberture_score.py` |

判定标准见 `index/DESIGN_CRITERIA.md`（v2：H1 TM≥0.70、H2 dev≤2.0）。

---

## 六、FAQ

**Q1：v7 还是 v9？**
无配体/小蛋白（L≤300）用 v7；配体口袋/大蛋白用 v9。注意电荷边界不同（v7 负电强、v9 正电强，见 `WORKFLOW_GUIDE.md` §8 或电荷限制报告）。

**Q2：target 没达到？**
检查是否在电荷边界内（§8 速查表）；极端负电用 v7、极端正电用 v9；长序列（L≥470）需检查结果。

**Q3：生成序列和 native 差别大？**
逆折叠是"换序列保骨架"，恢复率 ~0.5 正常。若需保留关键位点，用 `--fixed_residues`。

**Q4：只改 pH 序列没变？**
模型对 pH 的响应体现在"同 pH 不同电荷 → 不同序列"；只改 pH 不改 target 时温和区影响小。要主动控制电荷必须设 `--target_charge`。

**Q5：如何复现？**
固定 `--seed` + 相同参数 → 相同序列。

**Q6：编码器从哪来？**
v7/v9 编码器不在仓库（git 忽略权重），从 GitHub Releases 下载（`gh release download v1.0.0`，见 `README.md` §二 步骤 3）。
