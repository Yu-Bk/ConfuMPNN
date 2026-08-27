# ConfuMPNN — pH 感知的蛋白序列生成

给定一个蛋白的**骨架结构**（PDB）和一个**工作环境 pH**（可选目标净电荷），生成一段**在该 pH 下净电荷符合目标、能折叠回原骨架、且空间上电荷分布合理**的蛋白序列。

> 核心创新：在结构逆折叠模型（LigandMPNN）上首次加入 **pH 感知的电荷条件控制**。
> 当前进度：**v7/v9 为阶段性成果（2026-08-19 暂停训练），v10 演进中（2026-08-27）**——两个条件编码器（v7 MoMPNN / v9 LigandMPNN）当前可用，v10 方案（A/B/C）见 [v3 论文导向方案](index/PROJECT_LOCAL.md)。当前版本使用边界见 [电荷限制指南](analysis/report/2026-08-18_model_charge_limits.md)。

---

## 📑 目录

1. [项目简介](#一项目简介)
2. [从零开始（新机器配置）](#二从零开始新机器配置)
3. [快速上手（条件采样）](#三快速上手条件采样)
4. [使用指南（进阶）](#四使用指南进阶)
5. [输出解读](#五输出解读)
6. [文档与报告](#六文档与报告)
7. [项目结构](#七项目结构)

---

## 一、项目简介

**要解决的问题**：蛋白质工程中，常希望"骨架不变、换序列"——比如调整等电点、表面电荷分布，让蛋白更可溶、更稳定、或适应特定 pH 环境（溶酶体 pH≈5、血液 pH≈7.4 等）。但盲目改序列会破坏折叠。

**为什么 pH 重要**：氨基酸侧链有可电离基团（Asp/Glu 的 -COOH、Lys/Arg 的 -NH3+、His 的咪唑基）。蛋白质的**净电荷随 pH 变化**（pH 低 → 质子多 → 净电荷偏正；pH 高 → 偏负）。净电荷影响溶解度、聚集、与配体结合。

**我们的方法（主线：条件微调，模型 pH 感知）**：训练一个小型 **ConditionEncoder**（0.08M 参数），把 (pH, 目标净电荷) 编码成 soft prompt，通过 cross-attention 注入冻结的 backbone（MoMPNN 无配体 / LigandMPNN 配体模式），让模型**自身**学会按条件生成序列。备用的**引导采样路线**（不改模型，解码时注入电荷前瞻 logit bias）用于对照与快速原型。

**两个当前可用编码器**（v10 演进中）：
- **v7**（MoMPNN backbone，无配体/小蛋白）：负电可靠，正电弱
- **v9**（LigandMPNN backbone，配体/大蛋白）：正电可靠，负电弱

理解原理请看 [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)（面向新人的完整指南，含框架/数据流/参数/损失/为什么）。

---

## 二、从零开始（新机器配置）

> 本机已配好环境（`confumpnn` / `confumpnn-esmfold`）。从零复现请跟随 **`docs/SETUP_NEW_MACHINE.md`**（新机器配置指南：权重下载、环境、数据重建、验证）。
> 以下为概要。

### 前置

- Git、conda（Miniconda 推荐）、GPU（NVIDIA + CUDA）

### 步骤 1：克隆仓库

```bash
git clone git@github.com:Yu-Bk/ConfuMPNN.git
cd ConfuMPNN
```

### 步骤 2：克隆依赖源码（含官方权重）

```bash
git clone https://github.com/dauparas/LigandMPNN.git   # 逆折叠模型，自带 model_params/ 权重
git clone https://github.com/Qivon7/MoMPNN.git          # 多目标 DPO 微调权重（默认生成器）
```

### 步骤 3：下载自训编码器权重（v7/v9，GitHub Releases）

v7/v9 编码器是本项目微调产物，不在仓库内，从 GitHub Releases 下载：

```bash
# 查看已发布的 release 附件
gh release list
gh release view preview1.0.0

# 下载两个编码器权重（v7 296K + v9 887K）
gh release download preview1.0.0 --pattern "condition_encoder*.pt" -D code/weights/
sha256sum -c code/weights/SHA256SUMS.txt      # 校验完整性
```

放到 `code/weights/` 后，采样命令的 `--cond_encoder` 指向它们（见第三节）。

### 步骤 4：创建 conda 环境

```bash
conda create -n confumpnn python=3.11
conda activate confumpnn
conda install pytorch==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install biopython==1.79 numpy==1.23.5 scipy==1.12.0 prody==2.4.1 networkx dm-tree propka==3.5.1
```

> ⚠️ 不要装 torchvision/torchaudio/dgl——LigandMPNN 不需要，且曾与 torch 版本不匹配导致 import 崩溃。
> ESMFold 回折需要单独的 `confumpnn-esmfold` 环境（详见 `docs/SETUP_NEW_MACHINE.md` 或 `docs/CONFIG.md`）。

### 步骤 5：验证

```bash
cd code
python tests/test_all.py        # 单元测试，应全部通过
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder weights/condition_encoder_v7_last.pt --num_samples 3
```

### 步骤 6（可选）：训练数据

**推理生成不需要训练数据**。若要重新训练/微调编码器，数据可重新获取（CATH 818MB 下载 + RCSB 配体重拉，脚本在 `code/tests/`）或从组内 NAS 恢复，详见 `data/README.md`。

### 步骤 7（选装）：下游打分工具

生成后的序列可再打分（可设计性/可溶/热稳），均为**选装**：
- **ESMFold**（pLDDT + 回折 TM-score）——`confumpnn-esmfold` 环境
- **Protein-Sol**（%sol）、**TemBERTure**（Tm）

---

## 三、快速上手（条件采样）

```bash
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
conda activate confumpnn

# 无配体/小蛋白 → 用 v7 编码器（MoMPNN backbone）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder output/finetune_v7/condition_encoder_last.pt \
  --num_samples 10

# 配体模式 → 用 v9 编码器（LigandMPNN backbone + 配体原子上下文）
python run_guided.py --pdb ../data/validation_pdbs/1AZM.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder output/finetune_ligand_v9/finetune_epoch030.pt \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --num_samples 10
```

**预期输出**（终端）：
```
[1] 加载模型: ligandmpnn_v_32_010_25.pt  (device=cuda)
[2] 读取 PDB: ../data/validation_pdbs/1AZM.pdb
    蛋白长度 258，native: MSTPQGRLYLFFSTCPS...
[3] 条件注入模式: cond_mode=conditioned, cond_vec=[7.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    电荷校准: （config 中 charge_calibration.enabled=false，未校准）
[4] 条件注入采样 10 条候选序列...
    [ 1] charge= +0.35  pI= 6.90  MSTPQGRLYLFFSTCPELYYF...
    ...
[5] native   : charge= -1.71  pI= 6.55  MSTPQGRLYLFFSTCPELYYF...
    平均净电荷 = +0.35 ± 0.40  (目标 0.0)
[6] 输出已保存: output/guided_1AZM_pH7.4/seqs.fa
完成 ✅
```

**结果文件**：`code/output/guided_<pdb>_pH<pH>/` 下 `seqs.fa`（候选序列 + native 对照）+ `summary.json`（结构化结果）。

> 💡 **选 v7 还是 v9**：无配体 / 小蛋白（L≤300）用 v7；配体口袋 / 大蛋白用 v9。电荷边界（v7 负电强、v9 正电强）见第四节速查。

---

## 四、使用指南（进阶）

### 4.1 条件采样参数

| 参数 | 说明 |
|------|------|
| `--pH` | 工作环境 pH（与 target 配合）|
| `--target_charge` | 目标净电荷（None=不引导电荷，只结构过滤）|
| `--cond_encoder` | v7/v9 编码器权重路径（给了才走条件注入）|
| `--weights` | backbone 权重（默认 MoMPNN；配体模式用 LigandMPNN 权重）|
| `--num_samples` | 候选序列数 |
| `--temperature` | 采样温度（默认 0.3）|
| `--fixed_residues` | 固定残基（如 `'A12 C15'`），保留结合位点 |
| `--seed` | 固定随机种子，可复现 |

### 4.2 电荷边界速查（完整版见 `analysis/report/2026-08-18_model_charge_limits.md`）

| 条件 | v7（MoMPNN） | v9（配体模式） |
|------|-------------|---------------|
| 温和 native±2 | 91–100% ✅ | 87% ✅ |
| 极端负电 native−8 | **95% ✅** | **40% ⚠️ 欠冲** |
| 极端正电 native+8 | **40% ❌ 过冲** | **100% ✅** |

**使用规则（v9 配体模式）**：正电可用到 native+8；负电保守到 native−5；长序列（L≥470）需检查。

### 4.3 其他命令

```bash
# 固定结合位点（保留配体口袋）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder output/finetune_v7/condition_encoder_last.pt \
  --fixed_residues "A12 C15"

# 引导采样路线（Phase 1，不改模型，对照用）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 --preset default
```

---

## 五、输出解读

**seqs.fa**：
```
>sample_1 pH=7.4 charge=+0.35 pI=6.90
MSTPQGRLYLFFSTCPELYYF...
...
>native charge=-1.71 pI=6.55
MSTPQGRLYLFFSTCPELYYF...
```
- `charge`：该序列在指定 pH 下的净电荷（Henderson-Hasselbalch 平滑计算）
- `pI`：该序列的等电点
- 最后一条 `native`：输入 PDB 的原始序列（对照）

**summary.json**：结构化副本（运行参数 + 每序列 seq/charge/pI + 统计均值）。

**FAQ（简版）**：
- 生成序列和 native 差别大？→ 逆折叠是"换序列保骨架"，序列恢复率 ~0.5 正常。
- target 没达到？→ 检查是否在电荷边界内（§4.2）；或 target 超出物理极限；或固定了太多残基。
- 只改 pH 序列没变？→ 必须用 `--target_charge` 配合（模型对 pH 的响应体现在"同 pH 不同电荷 → 不同序列"；纯 pH 单独变化在温和区影响小）。

---

## 六、文档与报告

**权威指南**：`WORKFLOW_GUIDE.md`（根目录）——框架/数据流/参数/损失/为什么，面向计算机新人。

**新机配置**：`docs/SETUP_NEW_MACHINE.md`（权重下载、环境、数据重建、验证）

**数据组织**：`data/README.md`（数据划分、重建命令、SHA256 清单）

**其他文档**（`docs/`）：
- `docs/TECH.md` — 技术原理与公式
- `docs/CONFIG.md` — 配置与参数
- `docs/USAGE.md` — 使用场景与 FAQ

**计划/判据**（`index/`）：`PROJECT_PLAN.md`、`PROJECT_EXTEND.md`、`DESIGN_CRITERIA.md`（判断标准 v2）、`DOCUMENT_INDEX.md`（文档索引）

**实验报告**（`analysis/report/`）：从 E1 到 v9 泛化验证的完整报告链。最新：
- `2026-08-19_v9_generalization_validation.md` — v9 泛化验证（10 未见蛋白）
- `2026-08-18_model_charge_limits.md` — 电荷边界 + 根因分析（使用前必读）
- `2026-08-18_v9_ligand_training.md` — v9 训练

---

## 七、项目结构

文件分类存放遵循 `index/FILE_MANAGEMENT.md`：

- `code/` — 代码（`src/` 核心模块 + `configs/` 配置 + `tests/` 测试 + `run_guided.py`/`train_finetune.py`）
- `docs/` — 技术/配置/使用/新机配置文档
- `analysis/` — 实验报告（`report/`、`archieved/`、`accident/`、`ablation/`）
- `index/` — 规划、文件管理规范、文档索引、判断标准
- `data/` — 外部数据集（CATH/配体/验证集，git 不跟踪，见 `data/README.md`）
- `output/` — 训练产物（v7/v9 当前编码器 + 验证结果，git 不跟踪）
- `LigandMPNN/`、`MoMPNN/` — 外部源码/权重（git 不跟踪）
