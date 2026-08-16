# ConfuMPNN 使用说明

> 面向初次使用者的**完整上手指南**：环境 → 命令 → 示例 → 输出解读 → 常见问题。
> 技术原理见 `docs/TECH.md`，配置项见 `docs/CONFIG.md`。

---

## 一、环境准备

**唯一必需环境**：`confumpnn`（Python 3.11, torch 2.2.1+cu121）。激活：

```bash
source /home/baokun_yu/miniconda3/etc/profile.d/conda.sh
conda activate confumpnn
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
```

其他环境（ESMFold / TemBERTure / Protein-Sol）仅在做**下游打分验证**时用，生成本身不需要。

---

## 二、快速开始（最小示例）

生成 10 条满足「pH 7.4、净电荷≈0」的候选序列：

```bash
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 --num_samples 10
```

> 默认生成器 = **MoMPNN**（`mompnn_temberture_tm_esm_6_4_4_b01.ckpt`，多目标 DPO 微调版，E1b 验证四指标×4 PDB 全优）。回退原版 LigandMPNN 见场景 4。

输出（终端）：
```
[1] 加载模型: mompnn_temberture_tm_esm_6_4_4_b01.ckpt  (device=cuda)
[2] 读取 PDB: input/1BC8.pdb
    蛋白长度 93，native: MDSAITLWQFLLQLLQKPQNKHMICWTSNDGQFKLLQAEEVARLWGIRKN...
[3] 引导设置: pH=7.4, target_charge=0.0, preset=default, strength=0.5
[4] 引导采样 10 条候选序列...
    [ 1] charge= +1.06  pI= 9.14  AASPISLHQFLLQLLSNPAYSSIIAWVSSSGEFQLLDPEAV...
    ...
[5] native   : charge= +8.90  pI=10.22  MDSAITLWQFLL...
    平均净电荷 = +0.23 ± 0.80  (目标 0.0)
[6] 输出已保存: output/guided_1BC8_pH7.4/seqs.fa
完成 ✅
```

**结果文件**（`output/guided_1BC8_pH7.4/`）：
- `seqs.fa`：全部候选序列 + native 对照（每条含 pH/净电荷/pI）
- `summary.json`：结构化结果（参数 + 每序列的 seq/charge/pI）

---

## 三、典型使用场景

### 场景 1：指定 pH，不指定电荷（只做结构过滤）

```bash
python run_guided.py --pdb input/1UBQ.pdb --pH 5.5
```
⚠️ **注意**：此时生成的序列**与 pH 无关**（模型不感知 pH，`docs/TECH.md` §6 的诚实边界）。净电荷随 pH 的变化只是同一序列的物理电荷计算。要让序列真正响应 pH，必须用 `--target_charge` 引导或后续 Phase 2 微调。

### 场景 2：指定目标净电荷（电荷引导）

```bash
# 需要负电序列（如酸性环境工作蛋白）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge -8 --num_samples 10

# 需要正电序列（如 DNA 结合面）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge +8 --num_samples 10
```
target 命中验证：E1 中 target=+8/0/−8 → 平均净电荷 **+8.06 / +0.23 / −7.96**。

### 场景 3：不同结构过滤场景（`--preset`）

```bash
# 核酸结合蛋白（正电聚集更宽容）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --preset nucleic_acid_binding --target_charge 0

# 膜蛋白（疏水核心严格禁带电）
python run_guided.py --pdb input/2LZM.pdb --pH 7.4 --preset membrane --target_charge 0

# 酸性环境（溶酶体 pH≈5）
python run_guided.py --pdb input/2LZM.pdb --pH 5.5 --preset acidic --target_charge 0
```

### 场景 4：回退原版 LigandMPNN（含配体上下文）

MoMPNN 是纯 backbone（无配体上下文）。若你的任务需要**配体/核酸结合位点上下文**（LigandMPNN 的招牌能力），显式指定原版权重：

```bash
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
```
`--model_type auto` 会自动识别该权重为 ligand_mpnn（配体上下文）。

### 场景 5：可复现实验（固定种子 + 指定输出目录）

```bash
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --seed 111 --num_samples 20 --out_dir output/my_run
```

---

## 四、输出解读

### seqs.fa 格式

```
>sample_1 pH=7.4 charge=+1.06 pI=9.14
AASPISLHQFLLQLLSNPAYSSIIAWVSSSGEFQLLDPEAVAKLWGERKGKPSMNWGNLQ
>sample_2 ...
...
>native charge=+8.90 pI=10.22
MDSAITLWQFLLQLLQKPQNKHMICWTSNDGQFKLLQAEEVARLWGIRKNKPNMNYDKLS...
```

- `charge`：该序列在指定 pH 下的净电荷（HH 平滑计算，`src/differentiable_charge.py`）
- `pI`：该序列的等电点（二分搜索）
- 最后一条 `native`：输入 PDB 的原始序列（对照）

### summary.json 格式

```json
{
  "pdb": "input/1BC8.pdb", "pH": 7.4, "target_charge": 0.0,
  "preset": "default", "temperature": 0.3, "strength": 0.5, "seed": 111,
  "native_charge": 8.90, "native_pI": 10.22,
  "mean_charge": 0.23, "std_charge": 0.80,
  "sequences": [{"seq": "...", "charge": 1.06, "pI": 9.14}, ...]
}
```

---

## 五、下游验证（可选）

生成序列可用三个工具打分（各在不同环境）：

| 目标 | 工具 | 环境 | 脚本 |
|------|------|------|------|
| 可设计性（能否折叠回骨架） | ESMFold pLDDT + TM-score | `confumpnn-esmfold` + `confumpnn` | `esmfold_score.py` + `tm_score.py` |
| 可溶性 | Protein-Sol %sol | 系统 python + Perl | `protein_sol_mcp/scripts/protein_sol_predict.py` |
| 热稳定性 | TemBERTure Tm | `confumpnn-temberture` | `temberture_score.py` |

批量打分参考 `code/tests/e1_ext_score.sh`（含并行与线程调优经验）。

---

## 六、常见问题（FAQ）

**Q1：生成序列为什么和 native 差别很大？**
逆折叠（inverse folding）是"给定骨架换序列"，设计性任务，序列恢复率 ~0.5 属正常。native 仅作对照。

**Q2：指定 target 但实际电荷没到？**
可能原因：① `--strength` 太小（增大到 0.5–1.0）；② target 超出物理可能（如 pH 4 下要求强负电，酸性环境净电荷下限受限，见 E1b 报告）；③ 结构过滤器在抑制带电残基（换更宽松预设）。

**Q3：MoMPNN 权重加载报错？**
确认 `--model_type auto`（默认）能自动识别。若显式指定，MoMPNN 用 `protein_mpnn`。

**Q4：多链 PDB 会怎样？**
自动取蛋白链（LigandMPNN `parse_PDB` 只收集标准氨基酸链）。DNA/配体链被忽略；TM-score 回折比对需纯蛋白链参考（如 1BC8 → `input/1BC8_chainC.pdb`）。

**Q5：如何复现结果？**
固定 `--seed`；同一 seed + 相同参数 → 相同序列。

**Q6：CUDA 不可用时？**
自动回退 CPU（`torch.cuda.is_available()` 检测），但会很慢，建议 GPU。

---

## 七、一键批处理参考

- 多蛋白 × 多条件：`code/tests/examples_compare.sh`（示例蛋白对比）
- 多 PDB × 多 pH × 多 target：`code/tests/e1_extended.sh`（E1b 扩展）
