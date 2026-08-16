# ConfuMPNN — pH 感知的蛋白序列生成

给定一个蛋白的**骨架结构**（PDB）和一个**工作环境 pH**（可选目标净电荷），生成一段**在该 pH 下净电荷符合目标、且空间上电荷分布合理**的蛋白序列。

> 核心创新：在结构逆折叠模型（LigandMPNN）上首次加入 pH 感知的电荷条件控制。
> 当前进度：**Phase 1（Level 1 引导采样）已完成交付**，默认生成器为 MoMPNN。

---

## 📑 目录

1. [项目简介](#一项目简介)
2. [从零开始（环境搭建）](#二从零开始环境搭建)
3. [快速上手（一键生成）](#三快速上手一键生成)
4. [使用指南（进阶）](#四使用指南进阶)
5. [输出解读](#五输出解读)
6. [文档与报告](#六文档与报告)
7. [项目结构](#七项目结构)

---

## 一、项目简介

**要解决的问题**：蛋白质工程中，常希望"骨架不变、换序列"——比如调整等电点、表面电荷分布，让蛋白更可溶、更稳定、或适应特定 pH 环境（溶酶体 pH≈5、血液 pH≈7.4 等）。但盲目改序列会破坏折叠。

**为什么 pH 重要**：氨基酸侧链有可电离基团（Asp/Glu 的 -COOH、Lys/Arg 的 -NH3+、His 的咪唑基）。蛋白质的**净电荷随 pH 变化**（pH 低 → 质子多 → 净电荷偏正；pH 高 → 偏负）。净电荷影响溶解度、聚集、与配体结合。

**我们的方法**（Level 1，不改模型）：在 LigandMPNN 解码时注入两条正交的 logit bias——
- **动态电荷前瞻**（`charge_lookahead.py`）：每一步把整条序列的净电荷拉向目标值
- **结构感知过滤器**（`structure_aware_filter.py`）：4 条空间规则，抑制电荷异常聚集（同号扎堆、盐桥过密、电荷渗入疏水核心）

理解原理请看 [docs/TECH.md](docs/TECH.md)。

---

## 二、从零开始（环境搭建）

> 本机已配好环境 `confumpnn`（见 [docs/CONFIG.md](docs/CONFIG.md) 环境节）。以下是从零复现步骤，供其他机器使用。

### 前置

- Git、conda（Miniconda 推荐）
- GPU（NVIDIA + CUDA；无 GPU 也能跑但很慢）

### 步骤 1：克隆仓库

```bash
git clone git@github.com:Yu-Bk/ConfuMPNN.git
cd ConfuMPNN
```

### 步骤 2：克隆依赖源码与模型权重

```bash
# LigandMPNN（逆折叠模型源码，含权重 model_params/）
git clone https://github.com/dauparas/LigandMPNN.git
# MoMPNN（ProtAlign 多目标 DPO 微调权重，默认生成器）
git clone https://github.com/Qivon7/MoMPNN.git
```

> 这两个目录是外部源码/权重，git 不跟踪（已在 `.gitignore`）。MoMPNN 仓库只含权重（`MoMPNN/mompnn_paper_checkpoints/`）。

### 步骤 3：创建 conda 环境

```bash
conda create -n confumpnn python=3.11
conda activate confumpnn
conda install pytorch==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install biopython==1.79 numpy==1.23.5 scipy==1.12.0 prody==2.4.1 networkx dm-tree propka==3.5.1
```

> ⚠️ 不要装 torchvision/torchaudio/dgl——LigandMPNN 不需要，且曾与 torch 版本不匹配导致 import 崩溃。

### 步骤 4：验证

```bash
cd code
python tests/test_all.py        # 36 项单元测试，应全部通过
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 --num_samples 3
```

---

## 三、快速上手（一键生成）

```bash
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
conda activate confumpnn

# 最小用法：指定 PDB + pH，生成 10 条满足「pH 7.4、净电荷≈0」的序列
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 --num_samples 10
```

**预期输出**（终端）：
```
[1] 加载模型: mompnn_temberture_tm_esm_6_4_4_b01.ckpt  (device=cuda)
[2] 读取 PDB: input/1BC8.pdb
    蛋白长度 93，native: MDSAITLWQFLLQLLQKPQNKHMICWTSNDGQFKLLQAEEVARLWGIRKN...
[3] 引导设置: pH=7.4, target_charge=0.0, preset=default, strength=0.5
[4] 引导采样 10 条候选序列...
    [ 1] charge= -0.98  pI= 5.79  MKSKISLYEFLYYLLSKPEYNSIIRWTSNNGEFELIDPEAV...
    ...
[5] native   : charge= +8.90  pI=10.22  MDSAITLWQFLL...
    平均净电荷 = -0.01 ± 0.80  (目标 0.0)
[6] 输出已保存: output/guided_1BC8_pH7.4/seqs.fa
完成 ✅
```

**结果文件**（`code/output/guided_1BC8_pH7.4/`）：
- `seqs.fa`：全部候选序列 + native 对照（每条含 pH/净电荷/pI）
- `summary.json`：结构化结果（参数 + 每序列的 seq/charge/pI）

> 默认生成器 = **MoMPNN**（多目标 DPO 微调版，可溶/热稳更优）。原版 LigandMPNN（含配体上下文）用 `--weights` 回退，见下一节。

---

## 四、使用指南（进阶）

| 需求 | 命令 | 说明 |
|------|------|------|
| 指定目标净电荷 | `--target_charge -8` | 负值=偏酸/负电序列，正值=偏正电（如 DNA 结合面） |
| 指定 pH | `--pH 5.5` | 与 target 配合；⚠️ 只改 pH 不改 target 时，序列不变（模型不感知 pH，见 FAQ） |
| 结构过滤预设 | `--preset nucleic_acid_binding` | 4 种：`default` / `nucleic_acid_binding` / `membrane` / `acidic` |
| 用原版 LigandMPNN | `--weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt` | 配体/核酸结合上下文场景 |
| 调整引导强度 | `--strength 0.8` | 电荷引导强度（建议 0.2–0.5 起步，过大破坏可折叠性） |
| 可复现实验 | `--seed 111 --out_dir output/my_run` | 固定种子 + 指定输出目录 |
| 生成更多 | `--num_samples 20` | 候选序列数 |

**示例组合**：
```bash
# 酸性环境（溶酶体 pH≈5）+ 偏负电 + 酸性预设
python run_guided.py --pdb input/2LZM.pdb --pH 5.5 --target_charge -4 --preset acidic

# 核酸结合蛋白：正电聚集更宽容
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 --preset nucleic_acid_binding

# 回退原版 LigandMPNN（配体上下文）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
```

**批量/下游验证**：多蛋白批处理参考 `code/tests/examples_compare.sh` 与 `code/tests/e1_extended.sh`；生成序列可再用 ESMFold（pLDDT/TM-score）、Protein-Sol（%sol）、TemBERTure（Tm）打分（见 [docs/USAGE.md](docs/USAGE.md) 第五节）。完整用法与 FAQ 见 [docs/USAGE.md](docs/USAGE.md)。

---

## 五、输出解读

**seqs.fa**：
```
>sample_1 pH=7.4 charge=+1.06 pI=9.14
AASPISLHQFLLQLLSNPAYSSIIAWVSSSGEFQLLDPEAVAKLWGERKGKPSMNWGNLQ
...
>native charge=+8.90 pI=10.22
MDSAITLWQFLLQLLQKPQNKHMICWTSNDGQFKLLQAEEVARLWGIRKNKPNMNYDKLS...
```
- `charge`：该序列在指定 pH 下的净电荷（Henderson-Hasselbalch 平滑计算）
- `pI`：该序列的等电点
- 最后一条 `native`：输入 PDB 的原始序列（对照）

**summary.json**：结构化副本，含运行参数 + 每序列 seq/charge/pI + 统计均值。

**FAQ（简版）**：
- 生成序列和 native 差别大？→ 逆折叠是"换序列保骨架"任务，序列恢复率 ~0.5 正常。
- target 没达到？→ 增大 `--strength`；或 target 超出物理极限（如 pH 4 要强负电）；或过滤器在抑制带电残基（换宽松预设）。
- 只改 pH 序列没变？→ 模型自身不感知 pH（Level 1 诚实边界），必须用 `--target_charge` 引导；真正的模型 pH 感知靠 Phase 2 条件微调。

---

## 六、文档与报告

**文档**（`docs/`）：
- [docs/TECH.md](docs/TECH.md) — 技术文档：架构 / 算法原理 / 设计决策 / 验证摘要
- [docs/CONFIG.md](docs/CONFIG.md) — 配置文档：YAML / 命令行参数 / 环境
- [docs/USAGE.md](docs/USAGE.md) — 使用说明：完整场景 / 输出解读 / FAQ / 批处理

**计划**：`index/PROJECT_PLAN.md`（第一版）与 `index/PROJECT_EXTEND.md`（第二版拓展）；文档索引 `index/DOCUMENT_INDEX.md`。

**实验报告**（`analysis/report/`）：
- `2026-08-16_phase1_examples.md` — 结构过滤器阈值统计 + 示例蛋白对比
- `2026-08-16_e1_three_targets.md` / `2026-08-16_e1_extended.md` — MoMPNN vs 原版对照（三目标 / 4 PDB 扩展）
- `2026-08-16_e4_default_mompnn.md` — 默认生成器切换

---

## 七、项目结构

文件分类存放遵循 [index/FILE_MANAGEMENT.md](index/FILE_MANAGEMENT.md)：

- `code/` — 实验模块代码（`src/` 核心模块 + `configs/` 配置 + `tests/` 测试 + `input/`、`output/`、`log/`）
- `docs/` — 技术 / 配置 / 使用说明文档
- `analysis/` — 实验报告（`report/` 等）
- `index/` — 项目规划、文件管理规范、文档索引
- `literature/` — 论文笔记（五维度分类）
- `session/` — 会话记录
- `data/` — 外部数据集（CATH S40 结构域，git 不跟踪）
- `LigandMPNN/`、`MoMPNN/` — 外部源码/权重（git 不跟踪）
