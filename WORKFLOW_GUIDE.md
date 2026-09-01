# ConfuMPNN 完整使用指南（从零理解 pH 感知的蛋白序列生成）

> 版本：v9 版（2026-08-19，v10 演进中）　适用对象：**计算机新人**（也适合生物背景读者）
> 定位：本指南是**全项目唯一权威入口**——讲清楚「整个框架长什么样、数据怎么流动、每个参数/函数/损失是什么、**为什么**要用它们」。
> 配套文档：新机配置 → `docs/SETUP_NEW_MACHINE.md`；数据组织 → `data/README.md`；判断标准 → `index/DESIGN_CRITERIA.md`；电荷边界 → `analysis/report/2026-08-18_model_charge_limits.md`。

---

## 0. 项目一句话 + 快速导航

**一句话**：给定一个蛋白的**骨架结构**（PDB 文件）和**工作环境 pH**（可选：目标净电荷），ConfuMPNN 生成一段「在该 pH 下净电荷符合目标、能折叠回原骨架、且电荷空间分布合理」的蛋白序列。

**核心创新**：现有的"按性质生成蛋白序列"工具（LaMBO-2、AntiBARTy 等）都是纯序列模型或抗体专用；在**显式建模配体原子上下文**的结构逆折叠模型（LigandMPNN）上做 pH/电荷的显式条件控制，本项目是第一例。

### 0.1 本指南怎么读（新手导航）

| 你处于哪个阶段 | 建议阅读路径 |
|---------------|-------------|
| 完全新手，想先懂原理 | §1 背景知识 → §2 整体框架 → §3 数据流 |
| 想用现成模型生成序列 | §7 命令速查 + §8 电荷边界（最重要） |
| 想重新训练/微调模型 | §3.2-3.4 + §4 核心模块 + §5 损失 + §6 参数 |
| 想在新机器复现整个项目 | §10 → 跳到 `docs/SETUP_NEW_MACHINE.md` |
| 遇到不理解的概念 | 随时翻 §11 术语表 |

> 💡 **新手提示**：本文档会反复用「类比」解释计算机概念。遇到不懂的术语，先看 §11 术语表；如果还是不懂，把它当作一个"黑盒子"先记住输入输出，等看到它在流程图里的位置就自然明白了。

---

## 1. 背景知识（生物 + 计算机双视角）

### 1.1 逆折叠：换序列、保骨架

**正向折叠**（fold）：给定氨基酸序列 → 预测三维结构（如 AlphaFold/ESMFold 做的事）。

**逆折叠**（inverse folding）：反过来——给定三维骨架（坐标），设计一段能折叠成这个形状的序列。

> **类比**：骨架是"模具"，序列是"浇筑进去的材料"。模具固定，材料可以换，但换的材料必须能填满这个模具并保持形状。

逆折叠是蛋白工程的常用手段：你想改某个性质（等电点、表面电荷、溶解度），但又不想破坏折叠结构，就固定骨架、重写序列。**序列改变是设计行为，不是破坏**。

### 1.2 pH 与蛋白质净电荷

氨基酸侧链有**可电离基团**（能带电荷的化学基团）：
- **酸性残基** Asp(D)/Glu(E)：侧链 -COOH，pH 高时失去质子（去质子化）→ 带 **-1 负电**
- **碱性残基** Lys(K)/Arg(R)/His(H)：侧链 -NH₃⁺/胍基/咪唑基，pH 低时获得质子（质子化）→ 带 **+1 正电**

蛋白质在某个 pH 下的**净电荷** = 所有带电残基电荷的代数和（加上主链 N/C 端的电荷）。

**pH 对净电荷的影响**（这是本项目一切的基础）：
- pH 越低（酸性环境，质子多）→ 酸性基团被质子化、带负电能力下降 → **净电荷偏正**
- pH 越高（碱性环境）→ 相反 → **净电荷偏负**

这条关系由 **Henderson-Hasselbalch（HH）方程**描述。本项目用它的平滑近似（§4.2）。

### 1.3 等电点 pI

**等电点 pI** = 让分子净电荷恰好为 0 的那个 pH。它由氨基酸组成**唯一确定**（可以用 §4.3 的二分搜索算出来）。

> 💡 本项目的**设计哲学**：pI 不作为模型输入，而是生成序列的**推导属性**——模型只负责让净电荷贴近目标，pI 是结果的自然产物，用来做验证检查。

### 1.4 神经网络的"语言"：logits、softmax、概率

这是理解本项目所有代码的关键。ProteinMPNN/LigandMPNN 的解码器在生成序列时，每个位置会输出一个 **logits 向量**（21 个数字，对应 20 种氨基酸 + 1 个未知 X）。

```
logits = [3.2, 1.1, -0.5, ...]   # 21 个"打分"，越大表示模型越倾向该氨基酸
```

**logits 本身不是概率**（可以是负数、加起来不是 1）。要变成"每种氨基酸被选中的概率"，需要 **softmax**：

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

把 21 个打分变成 21 个概率（都在 0~1 之间，加起来 = 1）。概率最高的那个氨基酸就是模型最可能选的。

> **类比**：logits 是评委给 21 个候选选手的打分；softmax 是把分数换算成"夺冠概率"。打分差距越大，冠军概率越悬殊。

**logit bias**（本项目 Phase 1 用的技巧）：在采样前，往 logits 上加一个偏置向量 `bias`，人为抬高/压低某些氨基酸的概率：

```
probs = softmax((logits + bias) / temperature)
```

比如想让序列更负电，就给酸性残基 D/E 加正的 bias（提高概率）、给碱性残基加负的 bias。

> ⚠️ **softmax 平移不变性**（本项目踩过的坑，§4.8）：`softmax(x+c) = softmax(x)`——如果 bias 里有一项跟候选氨基酸无关的常数，它会被 softmax 完全抵消，等于没加。所以 bias 必须写成"跟候选相关"的形式。

### 1.5 损失函数：模型怎么"学习"

**损失函数（loss）** 是一个数字，衡量"模型当前输出和期望输出差多远"。模型训练就是不断调整参数让这个数字变小（梯度下降）。

> **类比**：损失函数是"考试评分标准"。模型是学生，训练是反复做题（看到骨架），按评分标准（损失）知道自己哪里错，修正自己的解题思路（参数），直到每次考试都接近满分（损失很小）。

**梯度**：损失对模型参数求偏导，告诉模型"每个参数该往哪个方向调能降低损失"。**梯度下降**就是沿梯度反方向更新参数。

**过冲（overshoot）**：训练想让净电荷到达 target，但模型调过头了，生成的序列电荷远超目标。这是本项目反复解决的核心问题（§5.3、§6.4）。

**温度（temperature）**：softmax 里的除数。温度 < 1 让概率分布"更锐利"（更倾向最高分的氨基酸）；温度 > 1 让分布"更平滑"（更随机）。采样时常用低温（如 0.3）让生成更确定。

### 1.6 本章小结

- 逆折叠 = 固定骨架、重写序列。
- 净电荷随 pH 变化（HH 方程），是模型要控制的"性质"。
- 模型输出 logits → softmax → 概率；损失函数驱动训练；温度控制采样锐利度。

---

## 2. 整体框架：两条技术路线

### 2.1 总览图

```
┌──────────────────────────────────────────────────────────────────────┐
│                          run_guided.py（主入口）                       │
│   输入：--pdb 骨架 + --pH 工作pH + [--target_charge 目标净电荷]         │
└──────┬───────────────────────────────────────────────────────────────┘
       │
       ▼ 解析 PDB → 提取骨架特征
   parse_PDB（LigandMPNN/data_utils.py）→ featurize → 特征张量 {X, E_idx, S}
       │
       ├───────────────【路线一：Phase 1 引导采样（不改模型）】───────────┐
       │                ┌──────────────────────────┐                    │
       │  解码每个位置时 → │ 动态电荷前瞻 bias（电荷lookahead）│  → logits    │
       │                │ + 结构感知过滤器（空间规则）   │      + bias     │
       │                └──────────────────────────┘      → softmax → 采样 │
       │                                                                  │
       └───────────────【路线二：Phase 3 条件注入（微调模型，主线）】───────┘
                        ┌──────────────────────────────┐
        条件向量[7] →    │ ConditionEncoder（唯一训练对象） │ → [4,128] soft prompt
        (pH, charge)    └──────────────────────────────┘       │
                                                                ▼
                     backbone encode 得到 h_V → cross-attention 注入 → 解码
                     （MoMPNN：纯骨架　/　LigandMPNN：+配体原子上下文）
       │
       ▼ 输出
   seqs.fa（候选序列）+ summary.json（每条的电荷/pI）
       │
       ▼ 验证管线（可选但推荐）
   ESMFold 回折 → US-align TM-score → DESIGN_CRITERIA v2 判定（H1/H2/H3）
```

### 2.2 路线一：引导采样（Phase 1，不改模型）

**思想**：模型本身不感知 pH，我们通过**解码时注入 logit bias** 来"推"它生成符合电荷目标的序列。两条正交约束：
1. **动态电荷前瞻**（`charge_lookahead.py`）：每一步估算"放某个氨基酸会把整条序列净电荷推向哪"，据此给候选氨基酸加减 bias（管**净电荷总量**）。
2. **结构感知过滤器**（`structure_aware_filter.py`）：4 条空间规则抑制"电荷异常聚集"（同号扎堆、盐桥过密、电荷渗入疏水核心），给违规位置加负 bias（管**电荷空间分布**）。

**诚实边界**（本项目反复强调）：路线一不改模型，所以**模型自身对 pH 无感知**——同一蛋白、只改 pH 不改 target，生成的序列完全相同。真正的"模型 pH 感知"要靠路线二。

> 路线一现在主要用于对照实验和快速原型，正式使用推荐路线二。

### 2.3 路线二：条件注入（Phase 3，本项目主线）

**思想**：训练一个小网络 **ConditionEncoder**，把 (pH, 目标净电荷) 编码成一组 **soft prompt**（软提示，4 个向量），通过 **cross-attention** 注入到 backbone 的编码特征 h_V 里，让 backbone **本身学会**按条件生成序列。

**核心组件**（§4.4 / §4.5 详述）：
```
条件向量 c[7] ──> ConditionEncoder（MLP）──> soft prompt tokens[4,128]
                                                │
   h_V（backbone 编码的骨架特征 [L,128]） ──────┤  cross-attention 注入
                                                ▼
                      h_V ← h_V + softmax(h_V·promptᵀ/√d)·prompt
```

**为什么用 soft prompt 而不是别的**：
- 不用离散 token（MolGPT 式）：pH 是连续值（4.0–10.0 可以要 7.43），离散 token 精度损失大。
- 不用字面前缀拼接：decoder 的邻接索引依赖固定长度，前缀要重排索引易错；cross-attention 注入不需要改解码器。
- 只训 ConditionEncoder（74,880 参数 ≈ 0.08M），**冻结 backbone**（MoMPNN 的权重是多目标 DPO 优化过的，全量微调有破坏风险）。

### 2.4 v7 与 v9：双编码器分工

| | **v7 编码器** | **v9 编码器** |
|---|---|---|
| 权重文件 | `output/finetune_v7/condition_encoder_last.pt` | `output/finetune_ligand_v9/finetune_epoch030.pt` |
| backbone | **MoMPNN**（纯骨架 ProteinMPNN） | **LigandMPNN**（含配体原子上下文） |
| 训练数据 | CATH 结构域（`data/cath/labels_balanced_v7.npz`，7,886 域） | 配体复合物（`data/ligand_train/labels.npz`，4,972 × 8pH） |
| 适用场景 | **无配体 / 小蛋白**（单体，L≤~300） | **有配体 / 大蛋白**（配体口袋、L 可达 500） |
| 电荷边界 | 负电强、正电弱（+8 过冲） | 正电强、负电弱（−8 欠冲）→ §8 |
| 使用 | `--cond_encoder output/finetune_v7/...` + MoMPNN 权重 | `--cond_encoder output/finetune_ligand_v9/...` + LigandMPNN 权重 |

> **为什么两个编码器**：v7 只在 MoMPNN backbone 上训练过，直接用到 LigandMPNN 配体模式时，因为 LigandMPNN 的 h_V 特征分布不同，电荷控制失效（1MBN dev 14.05）。v9 就是专门在 LigandMPNN backbone 上重训同一架构的 ConditionEncoder，恢复配体模式的电荷控制。

---

## 3. 数据流动（端到端）

### 3.1 总流程图

```
【训练侧】
CATH S40 结构域 / RCSB 配体复合物
   │
   ▼  build_labels_v2.py / build_ligand_labels.py
labels.npz  = { domain_ids, seqs, coords, pH[8], charge[8], pI[8] }
   │           每个结构域带 8 个 (pH, 电荷) 条件标签
   ▼
train_finetune.py ──> 冻结 backbone  +  训练 ConditionEncoder
   │
   ▼
condition_encoder_last.pt（v7）/ finetune_epoch030.pt（v9）── 当前交付权重（v10 演进中）

【推理侧】
用户 PDB（骨架 ± 配体）
   │
   ▼  parse_PDB → featurize
特征张量 { X[L,4,3], E_idx, S, [Y 配体原子] }
   │
   ▼  run_guided.py / validate_generalization.py
   （指定 --cond_encoder = v7 或 v9 编码器 + 对应 backbone 权重）
   │
   ▼
seqs.fa（N 条候选序列）
   │
   ▼  esmfold_score.py（ESMFold 回折）→ tm_score.py（US-align TM-score）
DESIGN_CRITERIA v2 判定：H1 折叠 / H2 电荷 / H3 电荷分布
```

### 3.2 训练数据从哪来

| 数据集 | 来源 | 用途 | 位置 |
|--------|------|------|------|
| CATH S40 结构域 | CATH 4.4.0 非冗余数据集（818MB，34,653 域） | **v7 训练**（+外部碱性补充） | `data/cath/S40/dompdb` |
| 外部碱性域 | RCSB 额外下载（781 个） | 补 CATH 碱性不足 | `data/cath/ext_basic_*` |
| 配体复合物 | RCSB 搜索 API（4,972 个：小分子 4,155 + 金属 567 + RNA 244 + DNA 6） | **v9 训练** | `data/ligand_train/{small_mol,metal,rna,dna}` |
| 验证蛋白（未见） | RCSB 选取 10 个（含小分子/DNA/RNA/金属/长序列） | v9 泛化验证（**训练集排除**） | `data/validation_pdbs/` |
| 迁移测试 | 1MBN/4DFR/1FQG/5HVX/3T0F | 编码器迁移检验 | `data/ligand_test/` |

> ⚠️ **防泄漏**：所有验证蛋白都通过 `--exclude` 从训练集排除（曾拦截 1b24A01 进入训练集）。

### 3.3 标签怎么构建（labels.npz）

每个结构域/复合物，生成 **8 个 (pH, 电荷) 训练样本**：
```
对 8 个 pH 值（uniform(4.0, 10.0) 随机采样）：
    charge = net_charge(该域 native 序列, 该 pH)   # HH 方程计算净电荷
    pI     = find_pI(该域 native 序列)              # 二分搜索
```
存为 `labels.npz`：`domain_ids / seqs / coords / pH / charge / pI`。

**为什么要 8 个 pH**：让模型见过同一骨架在不同 pH 下的电荷需求，学会"pH 改变 → 净电荷该变多少"的响应。

**目标电荷策略（关键设计）**：训练时 target 不能恒等于 native 电荷，否则模型只要"重建 native 序列"就能同时满足 CE 和电荷损失，学不会电荷偏移能力。所以用**混合目标**（§5.7）：
- **70% 自洽样本**：target = native 电荷（锚定结构）
- **30% 扰动样本**：target = native 电荷 ± Uniform[1, perturb_scale]（制造"target 偏离 native 时如何偏移"的学习信号）

### 3.4 训练时数据怎么用（train_finetune.py 的预解析）

1. **解析**：`parse_PDB` 读每个域（CATH 文件无扩展名 → 创建 `.pdb` 符号链接让 prody 识别）。
2. **特征化**：`featurize` 得到 X（坐标）、E_idx（邻接）、S（序列）。v9 配体模式还要配体原子 Y。
3. **一次性 encode**：backbone 冻结 → 每个域的 `h_V, h_E, E_idx` 只算一次，全 epoch 复用（省算力）。
4. **无条件 logits 预计算**：不给条件注入的 backbone 输出（KL 锚 + seq-keep 锚的参考，§5）。
5. **每个 epoch 迭代**：取 8 个 pH 条件 → 按 70/30 决定 target → 构造条件向量 → 注入 → 解码 → 算 4 项损失 → 更新 ConditionEncoder。

> **批内设计**：一个域的 8 个 pH 样本组成一个 batch（结构共享、条件不同），充分利用冻结的 encode 结果。

### 3.5 推理时数据怎么流（run_guided.py）

```
--pdb → parse_PDB → protein_dict（X, S, R_idx, [Y]）
   → chain_mask（1=设计 / 0=固定）
   → featurize（use_atom_context 取决于模型类型）
   → make_condition_vector(pH, net_charge=target)   # 7 维条件向量
   → ConditionEncoder(条件向量) → prompt[4,128]
   → backbone.encode → h_V → inject_prompt(h_V, prompt)   # 注入
   → 解码循环（guided_sampler）→ 一条序列
   → net_charge(seq, pH) + find_pI(seq) 输出统计
```

### 3.6 配体模式的数据差异

| | 无配体（v7） | 配体模式（v9） |
|---|---|---|
| 特征化 | `use_atom_context=False` | `use_atom_context=True, number_of_ligand_atoms=16` |
| parse_PDB 输出 | 蛋白原子 X | + 配体原子 **Y/Y_t/Y_m** |
| backbone | ProteinMPNN（MoMPNN 权重） | ProteinMPNN 实例 + LigandMPNN 权重（含配体上下文层） |
| 口袋残基 | 无 | 距配体原子 < 8Å 的残基（`pocket_residues`） |
| 消融对照 | — | `strip_ligands` 去掉 HETATM 行 → 同一模型、无配体原子 |

> **配体消融实验的发现**：配体上下文对电荷控制**无系统性增益**，大蛋白（L=504）上反而有害（注意力被配体原子稀释）。若配体模式电荷控制不佳，可试无配体模式或固定结合位点（§7.5）。

---

## 4. 核心模块逐一详解

所有模块在 `code/src/`（部分直接复用 `LigandMPNN/data_utils.py` 的函数）。

### 4.1 `pka.py` — 氨基酸 pKa 表

**功能**：定义各氨基酸侧链的 pKa、带电类型、字母表。

```python
PKA_SIDECHAIN = {"D": 3.9, "E": 4.3, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}
AAS = "ACDEFGHIKLMNPQRSTVWY"      # 标准 20 氨基酸，顺序与 LigandMPNN 一致
ACIDIC = ("D","E","C","Y")        # 去质子化带负电
BASIC  = ("K","R","H")            # 质子化带正电
```

**为什么**：净电荷计算需要知道每个氨基酸的 pKa（HH 方程的参数）。这是"游离氨基酸 pKa"近似（结合层面），不做微环境修正。

### 4.2 `differentiable_charge.py` — 可微净电荷计算（全项目地基）

**功能**：算一条序列/一组 logits 在某个 pH 下的净电荷，且**处处可微**（能反向传播梯度）。

**核心公式**（HH 方程的 sigmoid 平滑近似）：
```
去质子化分数 = σ(ln10 · (pH − pKa))          # σ = sigmoid
酸性残基（D/E/C/Y）：电荷 = −σ(ln10·(pH−pKa))   # pH 高 → 去质子化 → 带 −1
碱性残基（K/R/H）：   电荷 = +σ(ln10·(pKa−pH))   # pH 低 → 质子化 → 带 +1
N 端 α-NH₃⁺（pKa≈9.7）：+σ(ln10·(9.7−pH))
C 端 α-COOH（pKa≈2.3）：−σ(ln10·(pH−2.3))
```

**为什么用 sigmoid 而非硬阈值**：sigmoid 平滑可导；硬阈值（"pH>pKa 就带 -1"）在阈值处不可导，无法用于梯度训练。

**两个接口**：
```python
net_charge(seq, pH)                    # 字符串序列 → 净电荷 float（验证用）
net_charge_from_logits(logits, pH, temperature)  # logits → 期望净电荷（训练用，可微）
```

`net_charge_from_logits` 的机制：softmax 概率 × 20 种氨基酸电荷 → 加权平均 → 求和。**temperature 参数**（`charge_temp`）：<1 时概率分布锐化，让"训练优化的期望电荷"更接近"推理时实际采样序列的电荷"，缩小训练/推理不一致导致的过冲（§5.3）。

### 4.3 `isoelectric_point.py` — pI 二分搜索

**功能**：`find_pI(seq)` 在 [0, 14] 二分搜索使净电荷为 0 的 pH。

**为什么用二分**：净电荷是 pH 的**单调递减函数**（pH 越高越负），单调函数可用二分快速求根。pI 是序列的推导属性，只用于验证检查，不输入模型。

### 4.4 `condition_embedding.py` — 条件向量 + 条件编码器（项目的"大脑"）

**条件向量（mask-aware，7 维）**——`make_condition_vector(pH, net_charge, ...)`：
```
[ pH,  has_charge_flag,  charge_val,  has_pos_limit_flag,  pos_limit_val,  has_neg_limit_flag,  neg_limit_val ]
  0    1                  2           3                    4               5                    6
```

**为什么有 flag 位**：`has_X_flag`（0/1）告诉网络"哪个值是真实条件、哪个是占位符"。避免 0 值歧义——"没指定电荷"（0 是占位符）≠"目标电荷就是 0"。这是 mask-aware 设计。

**ConditionEncoder 网络结构**：
```
Linear(7→64) → GELU → Linear(64→128) → GELU → Linear(128→4×128) → reshape [4, 128]
```
- 输出 **4 个 soft prompt token**，每个 128 维（与 backbone 特征 h_V 同维，才能做 cross-attention）。
- **为什么 GELU**：主流激活函数之一，平滑可导，比 ReLU 在 0 附近更光滑。
- **标准化**：条件向量里 pH（4–10）和净电荷（−20~+20）量纲差很大，直接输入会梯度不稳定。训练前从训练集算每维 μ/σ，存进 `condition_defaults.yaml`，推理时复用。

**为什么连续向量而非离散 token**：pH 是连续值，MolGPT 式离散 control token 精度损失大；NExT-Mol 式连续 soft prompt 无损。

### 4.5 `conditioned_sampler.py` — soft prompt 注入

**功能**：把条件编码器的输出注入 backbone 编码特征 h_V。

```python
def inject_prompt(h_V, prompt_tokens):
    attn = softmax(h_V @ prompt_tokens.T / √d)   # 每个结构节点对 4 个 prompt 的注意力权重
    return h_V + attn @ prompt_tokens            # 加权求和，加回 h_V
```

**为什么用 cross-attention 注入**：每个结构节点（氨基酸位置）能"按需读取"它最关心的条件信息。等价于 soft prompt，但**无需改动解码器**（字面前缀需要重排邻接索引，易错）。

**为什么 4 个 token**：soft prompt token 数是个超参数；4 个是折中——太少表达力不足，太多与序列特征混淆。

### 4.6 `guided_sampler.py` — 解码循环（推理的心脏）

**功能**：`guided_sample` / `GuidedSampler` 包装 LigandMPNN 解码器，逐位置生成序列。

关键机制：
```python
probs = softmax((logits + bias_t) / temperature)     # bias 直接加在 logits 上
S_t = multinomial(probs_sample)                       # 按概率采样一个氨基酸
```

**解码顺序**：由 `randn`（随机数）决定（`argsort(|randn|)`），不同 seed 得到不同解码顺序 → 不同序列。

**固定残基机制**（`--fixed_residues`）：`chain_mask=0` 的位置在解码时强制保持原氨基酸：
```python
S_t = S_t · chain_mask_t + S_true · (1 − chain_mask_t)
```
设计这些位置 = 保持原样。**为什么有用**：配体结合位点残基通常要保留（结合功能关键），只设计其余位置。

### 4.7 `structure_aware_filter.py` — 结构感知过滤器

**功能**：4 条空间规则，解码时给"电荷异常聚集"位置加负 bias（抑制）。

| 规则 | 检测内容 | 阈值（天然蛋白 99 分位） |
|------|---------|------------------------|
| 1 空间电荷聚集 | 10Å 内同号强电荷（K/R 或 D/E）≥ 6 | 6 |
| 2 盐桥过密 | 10Å 内正负电荷对 ≥ 4 | 4 |
| 3 核心电荷渗入 | 埋藏(burial>0.8)且 8Å 内带电 ≥ 6 | 6 |
| 4 同号电荷聚类 | 8Å 邻域同号电荷 ≥ 4 | 4 |

**阈值为什么用 99 分位**：统计 CATH 34,653 域的 151,519 个残基位，取"超过 99% 天然蛋白"的值为异常线——过滤器的目标是抑制"违反天然规律"的极端聚集，不是干预正常电荷。

**为什么用 bias（软抑制）而非硬禁止**：硬禁止可能把模型逼进死胡同；软 bias 只是降低概率，模型仍有回旋余地。

### 4.8 `charge_lookahead.py` — 动态电荷前瞻（路线一的核心）

**功能**：解码每一步，估算"放 20 种氨基酸各会把整条序列净电荷推向哪"，转成逐候选 bias。

**原理（电荷可加性）**：
```
Q_k = Q_fixed（已解码电荷和）+ q(aa_k, pH)（候选电荷）+ Q_expect_others（未解码位用平均电荷近似）+ Q_termini（端基）
```

**⚠️ 关键 bug 教训（softmax 平移不变性）**：
朴素写法 `bias_k = (Q_k − target)` 会失效——因为 `Q_k − target` 里不依赖候选 k 的项（Q_fixed、Q_expect_others、target）在 softmax 里被常数平移抵消。**正确写法**让 target 进入依赖候选的交叉项：
```
bias_k = strength · (target_charge − Q_current) · q(aa_k, pH)
```
- `Q_current < target`（欠正电）→ 正电候选得正 bias 被促进
- 随解码推进 Q_current 渐准 → 引导收敛

**为什么需要它**（路线一）：模型本身不感知 pH，只有靠解码时的逐步引导才能把电荷推向目标。

### 4.9 `run_guided.py` — 主入口

把整条推理管线串起来（§3.5），也是 **v7/v9 编码器切换的地方**：
- `--cond_encoder` 指定用哪个编码器（v7 或 v9）
- `--weights` 指定 backbone（MoMPNN 默认 / LigandMPNN 配体模式）
- 支持两条路线（§2）：给 `--cond_encoder` 走路线二，不给走路线一

---

## 5. 损失函数：模型在优化什么

### 5.1 总公式

训练时，ConditionEncoder 的目标是让下面这个总损失最小：

```
L = CE + λ_c·charge_deviation + λ_kl·KL_anchor + λ_keep·seq_keep
     ↑       ↑                     ↑              ↑
    重建序列  电荷偏差              防失控          保持序列
    λ=1      λ_c=0.5              λ_kl=0.05       λ_keep=0.5
```

四个损失是**互相配合**的，缺一不可（下面逐个讲）。

### 5.2 CE 交叉熵 — 重建 native 序列（保结构匹配）

```python
cross_entropy_loss(logits, target_seq, mask)   # 标准自回归交叉熵
```

**公式直觉**：对每个位置，看模型给"正确答案（native 氨基酸）"的概率，取负对数。答案概率越高，损失越低。

**为什么**：native 序列是"能折叠回该骨架"的已知答案。用 CE 锚定它，防止条件注入后序列偏离骨架可折叠范围太远。**它是结构匹配度的锚**。

### 5.3 电荷偏差损失 — 让净电荷贴近目标

```python
charge_deviation_loss(logits, pH, target_charge, temperature) = |期望净电荷 − target|
```

**为什么加 λ_c=0.5**：这是项目的"主任务"（控制电荷），权重不能太低。

**temperature 参数（charge_temp=0.5）—— 训练侧根治过冲的关键**：
- **问题**：训练时模型优化的是"softmax 期望电荷 E[Q]"（对所有氨基酸的电荷取期望）；但推理时实际采样用的是"argmax 附近的一条具体序列"的电荷。因为 CE 训练让模型很自信（概率分布尖锐），期望和采样会差很多——实测推理电荷比期望放大 ~2.9 倍（**过冲**）。
- **解法**：`temperature < 1` 让训练时的概率分布也锐化，使"训练优化的量"≈"推理采样的量"。加温度后过冲从 2.57× 降到 ~1.04×。

### 5.4 KL 锚定 — 防条件注入失控

```python
kl_anchor_loss(logits, logits_ref) = KL(p_ref ‖ p_cond)   # 条件化分布 vs 无条件分布
```

**公式直觉**：KL 散度衡量两个概率分布的差异。这里比较"注入条件后的输出分布"和"backbone 无条件输出分布"。

**为什么**：冻结的 backbone（MoMPNN）自带优秀的可溶/热稳性质。KL 锚约束条件注入后的输出**不偏离 backbone 太远**——只允许在电荷约束要求的方向上变化，防止微调破坏 backbone 已优化好的性质。

**为什么 λ_kl 只有 0.05**：它只是"安全带"，不能压过主任务（电荷控制）。

### 5.5 序列保持损失 — 直接管住"argmax 翻盘"

```python
sequence_keep_loss(logits, anchor_seq, mask)   # 对"无条件 argmax 序列"做 CE
```

**为什么需要它**（比 KL 更直接）：
- KL 管的是**分布距离**：概率 0.30→0.29 时 KL 已经很小，但 argmax 可能翻盘（比如 K 变成 R），序列级 identity 照样下降。
- seq_keep 直接以"无条件 argmax 序列"为目标做 CE：没有改电荷需求时，条件输出逐位逼近无条件输出。**它管的是序列本身，不是分布**。

**施加时机**：只在**自洽样本**（target=native）上施加。扰动样本 target≠native 时电荷偏移是期望行为，不受约束。

### 5.6 权重怎么定

| 权重 | 值 | 理由 |
|------|-----|------|
| λ_c | 0.5 | 电荷控制是主任务 |
| λ_kl | 0.05 | 防失控安全带，不宜压主任务 |
| λ_keep | 0.5 | 保序列/结构，与主任务平衡（用户明确不调） |

**不是凭空定的**：这些是 v3–v7 多轮迭代调出来的（历史：无 seq_keep → S1 注入选择性差；λ_kl 太大 → 电荷控制弱；λ_c 太大 → 折叠退化）。

### 5.7 训练数据里的"混合目标"

训练样本按概率分为三类，各自训练目的不同：

| 类型 | 比例 | target | 训练目的 |
|------|------|--------|---------|
| **自洽样本** | 70% | native 电荷 | 锚定结构（CE）+ 无需求时不重写（seq_keep）|
| **扰动样本** | 30% | native ± Uniform[1, scale] | 制造"target 偏离时如何偏移"的信号 |
| **占位样本** | 从自洽中抽 15% | 训练均值（"电荷不控制"语义）| 让"未指定电荷"落在温和可折叠默认 |

**为什么扰动**：如果 target 恒等于 native，CE 和电荷损失同时被"重建 native"满足，模型学不到电荷偏移能力。扰动制造二者冲突 → 教模型"怎么按 target 偏移氨基酸分布"。

**为什么占位符（均值占位）**：支持"部分条件不控制"（目标 2 场景）。历史教训：占位符样本若跳过电荷损失 → 模型把"不控制"学成"放任漂移"→ 推理时电荷极端负极化 → 折叠全失败。**均值占位 + 施加电荷损失（target=均值）**后完全修复。

**课程学习（v7 用）**：`perturb_scale` 随 epoch 从 2.0 渐进到 8.0——先学温和偏移，再学极端外推（"先简单后难"）。

---

## 6. 参数全表

### 6.1 训练参数（train_finetune.py）

| 参数 | 默认 | 含义 | 为什么选这个值 |
|------|------|------|---------------|
| `--weights` | MoMPNN 权重 | backbone | 默认生成器（E4 决策：MoMPNN 可溶/热稳更优）|
| `--ligand` | 关 | v9：用 LigandMPNN 权重 + 配体上下文 | 配体模式训练开关 |
| `--lr` | 1e-3 | 学习率 | 编码器很小（0.08M），常用值 |
| `--epochs` | 30 | 训练轮数 | 收敛稳定值 |
| `--lambda_c` | 0.5 | 电荷损失权重 | §5.6 |
| `--lambda_kl` | 0.05 | KL 锚权重 | §5.6 |
| `--lambda_keep` | 0.5 | 序列保持权重 | 用户明确固定 0.5 |
| `--perturb_prob` | 0.3 | 扰动样本比例 | 70/30 混合目标（历史 0.5→0.3）|
| `--perturb_scale` | 4.0 | 扰动幅度上限 | v7 用课程学习 2.0→8.0 |
| `--curriculum` | 关 | 课程学习 | v7 开：先温和后极端 |
| `--placeholder_prob` | 0.15 | 占位符比例 | §5.7 |
| `--charge_temp` | 0.5 | 电荷损失温度 | §5.3 根治过冲 |
| `--loss_reweight` | 关 | 逆密度加权 | v4+：治高正电 target 过冲（§6.3 附）|
| `--max_domains` | 0 | 用前 N 域 | 冒烟测试用 |

### 6.2 采样参数（run_guided.py / validate_generalization.py）

| 参数 | 默认 | 含义 | 为什么 |
|------|------|------|--------|
| `--pdb` | 必填 | 输入 PDB | — |
| `--pH` | 必填 | 工作环境 pH | 目标条件之一 |
| `--target_charge` | None | 目标净电荷 | 不给=只结构过滤不引导电荷 |
| `--cond_encoder` | None | v7/v9 编码器权重 | 给了才走条件注入路线 |
| `--cond_mode` | conditioned | conditioned=注入 / baseline=加载但不注入 | baseline=诚实边界对照 |
| `--weights` | MoMPNN | backbone 权重 | 配体模式用 LigandMPNN 权重 |
| `--temperature` | 0.3 | 采样温度 | 低温保证质量 |
| `--num_samples` | 10 | 候选序列数 | 验证用 30 |
| `--fixed_residues` | None | 固定残基（如 'A12 C15'） | 保留结合位点 |
| `--out_dir` | 自动 | 输出目录 | — |

### 6.3 配置 yaml（condition_defaults.yaml）

```yaml
condition_defaults:
  cond_dim: 7                    # 条件向量维度
  normalization:
    mean: [6.9982, 1.0, 1.4243, 0, 0, 0, 0]   # pH 均值≈7，电荷均值≈1.42
    std:  [1.7299, 0.0, 9.5259, 0, 0, 0, 0]
  encoder:
    hidden_dim: 64   token_dim: 128   n_tokens: 4
  charge_calibration:
    gain: 1.289   offset: 0.74   enabled: false
```

**normalization 从哪来**：训练集所有条件向量的 μ/σ（pH、电荷两个维度实际有值）。

### 6.4 电荷校准的前世今生（重要现状说明）

**历史**：Phase 3 早期发现过冲 ~2.9 倍 → 用**推理侧线性校准** `target_eff = (desired − offset) / gain` 补偿（早期 gain=2.57）。

**现状**：v9 起改用 **训练侧温度化**（`charge_temp=0.5`，§5.3）根治过冲 → 推理侧校准已**不再需要**，`enabled: false`。

> ⚠️ 注意：如果你看到 `run_guided.py` 的文档字符串还写旧值 gain=2.57/默认开，那是历史残留文字；实际代码读 yaml，行为是"默认不校准"。

---

## 7. 命令速查

所有命令在 `code/` 目录下、`conda activate confumpnn` 环境执行。

### 7.1 训练 v7（MoMPNN，无配体）

```bash
python train_finetune.py --device cuda:0 --epochs 30 \
  --labels ../data/cath/labels_balanced_v7.npz \
  --dompdb ../data/cath/S40/dompdb \
  --curriculum --perturb_scale 2.0 --curriculum_scale_max 8.0 \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --out_dir ../output/finetune_v7
```

### 7.2 训练 v9（LigandMPNN 配体模式）

```bash
python train_finetune.py --device cuda:0 --epochs 30 --ligand \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --labels ../data/ligand_train/labels.npz \
  --dompdb ../data/ligand_train/all_pdb \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --out_dir ../output/finetune_ligand_v9
```

### 7.3 条件采样（推理，指定 v7 或 v9 编码器）

```bash
# v7：无配体/小蛋白，MoMPNN backbone
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../output/finetune_v7/condition_encoder_last.pt \
  --num_samples 10

# v9：配体模式，LigandMPNN backbone + 配体上下文
python run_guided.py --pdb ../data/validation_pdbs/1AZM.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../output/finetune_ligand_v9/finetune_epoch030.pt \
  --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --num_samples 10
```

### 7.4 配体消融（同一模型，去配体）

```python
# validate_generalization.py 的 strip_ligands 已实现：
# 去掉 PDB 的 HETATM 行 → 重新 parse → 配体原子上下文为空
# 完整批处理见：
PYTHONPATH=code python code/tests/ligand_v9/validate_generalization.py \
  --manifest data/validation_pdbs/validation_manifest.json \
  --out_dir output/generalization_v9 --mode both \
  --cond_encoder output/finetune_ligand_v9/finetune_epoch030.pt \
  --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --n 30 --device cuda:0 --pH 7.4
```

### 7.5 固定结合位点

```bash
# 固定 A 链 12 和 C 链 15 位残基，其余位置由模型设计
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder ../output/finetune_v7/condition_encoder_last.pt \
  --fixed_residues "A12 C15"
```

### 7.6 完整验证打分管线（v12.2 现状，2026-09-01）

```bash
# 1. ESMFold 回折（confumpnn-esmfold 环境）
python code/tests/esmfold_score.py --input-dir <gen_root> --device cuda:4
# 2. US-align TM-score（对照参考骨架 ref/<pdb>_ref.pdb）
python code/tests/tm_score.py --folds <arm_dir>/folds --ref <ref.pdb> --out <arm_dir>/tm.csv
# 3. 判定（H1/H2/H3/H4）→ 见 index/DESIGN_CRITERIA.md
# 4. H3 电荷聚集合法性（2026-09-01 采纳，复用 structure_aware_filter 4 规则事后统计）
python code/tests/h3_charge_legality.py   # 条件臂 vs native_ref vs 无条件基线，违规率 ≤ 基线+5pp
# 5. H4 PROPKA 物理复核
python code/tests/propka_charge_check.py --pdb <folds> --pH 7.4 --target <q> --out <json>
```

**完整验证链（v12.2）**：训练 → 17 蛋白响应诊断（slope 校准后 ∈[0.9,1.15]）→ 组成分析（D/K vs native，防删减/过度添加）→ hold-out 评估（未见域 H2）→ 泛化验证（ESMFold H1 + 电荷 H2 + PROPKA H4）→ 无泄露 big-global 补跑 → 小样本现场标定 → Tm/Sol（S2 不恶化）→ **H3 电荷聚集合法性（新增）** → v9 配体迁移。**关键坑**：① 无条件基线必须用训练均值占位 `net_charge=1.4243`（`None` → poly-G 退化）；② USalign 需 `export PATH=.../confumpnn/bin`；③ ligand 泛化目录 `ligand/<pdb>/...` 蛋白名在 cut -f2。

---

## 8. 电荷边界使用指南（最重要的一章）

> 完整分析见 `analysis/report/2026-08-18_model_charge_limits.md`。这里是速查。

### 8.1 v7 vs v9 速查表（电荷命中率）

设 `target` 为输入电荷目标，`native` 为目标蛋白天然净电荷（pH 7.4 下）：

| 条件 | **v7（MoMPNN，无配体/小蛋白）** | **v9（配体模式，中短蛋白 L≤312）** |
|------|-------------------------------|-----------------------------------|
| 温和 native±2 | **91–100% ✅** | **87% ✅** |
| **极端负电 native−8** | **95% ✅ 可靠** | **40% ⚠️ 欠冲（弱项）** |
| **极端正电 native+8** | **40% ❌ 过冲** | **100% ✅ 强项** |
| 长序列 L≥470 | 未充分测试 | ❌ 全臂欠达标（折叠仍好）|

**v9 反转了 v7 的正负电不对称**。两条线要分开看。

### 8.2 v9 使用规则（配体模式）

- ✅ 温和区 native±2：可靠（87%）
- ✅ **正电设计可用到 native+8**（v9 强项）
- ⚠️ **负电设计保守到 native−5**（−8 欠冲）
- ⚠️ 长序列（L≥470）/ 血红素类：电荷控制需检查

### 8.3 为什么电荷控制会出问题（根因）

**关键认知（2026-08-19 修正）**：模型并非"精确删掉对侧电荷"——它**同时减少 D/E 和 K/R 两边**（整体减少带电残基总数，收敛到低电荷密度序列），净电荷方向由"哪边删得多"决定（负电化多删 K/R、正电化多删 D/E）。

**证据**（泛化验证 n=30 均值）：1BJ4 native 有 105 个带电残基（52D + 53R），生成序列只保留 ~18 个带电残基——**天然蛋白靠表面电荷"斑块"分布**（净电荷小 ≠ 带电残基少），删空一边会丢失电荷斑块 + 表面疏水化（GRAVY 上升）。

**机制链条**：
1. 统一策略 = **减少带电残基总数**，靠不对称删减调净电荷。
2. **删的代价**：表面疏水化 + 电荷斑块丢失 → 折叠仍好但溶解性/静电功能受损。
3. **为什么倾向"删"而非"加"**：删带电残基 = 表面换疏水/极性残基，结构上容易；加 D/E 需要表面可容纳位点。模型宁可欠冲也不生成可能不折叠的高电荷序列。
4. **seq-keep 正则强化"最小改动"**：约束序列接近 backbone 输出 → 倾向保守删减。

**启示**：正电设计是 v9 强项、v7 弱项；负电设计是 v7 强项、v9 弱项。按 8.1/8.2 边界使用，并对结果做电荷复算。

---

## 9. 项目当前状态与迭代史

### 9.1 阶段状态（2026-08-19 暂停训练；2026-08-27 进入 v10 演进）

- **当前可用**：v7 编码器（MoMPNN）+ v9 编码器（LigandMPNN），双编码器按场景分工。**这是阶段性成果，不是终版**——已知问题（删减捷径/电荷斑块丢失/极端边界）见 v3 方案 `index/PROJECT_LOCAL.md`，v10（A 条件解耦 + B 表面电荷监督 + C 结构惩罚）正在设计。
- **验证完成（v9 节点）**：v9 泛化验证（10 未见蛋白 × 5 电荷臂 × n30 × ligand/protein 双模式）→ `analysis/report/2026-08-19_v9_generalization_validation.md`。
- **使用边界**：§8。

### 9.2 迭代史时间线（浓缩）

```
Phase 0-1  环境 + 引导采样（路线一：电荷 lookahead + 结构过滤器）
E0-E4      MoMPNN 接入（多目标 DPO 微调版，设为默认生成器）
Phase 2    条件微调（ConditionEncoder + 4 项损失）
Phase 3    条件注入验证 → 发现过冲 → 校准 → 训练侧温度化根治
v2-v6      数据迭代（999域 → 分层 → 三类平衡 7,208 域）
v7         外部碱性数据 + 课程学习（H2 20/25，极端正电根治）
v9         配体模式重训（LigandMPNN backbone，电荷控制恢复）
v9 暂停    泛化验证完成 → 阶段节点（2026-08-19）
v10        演进中（A 条件解耦 + B 表面电荷监督 + C 结构惩罚，见 PROJECT_LOCAL）
```

**关键迭代教训**（详见 WORKFLOW_GUIDE 早期版 §7 / 各报告）：
- softmax 平移不变性使朴素 bias 失效 → 改交叉项写法（§4.8）
- 过冲 ~2.9 倍 → 训练侧温度化根治（§5.3）
- 分层采样砍中性多样性 97% → 三类平衡采样（v6）
- 占位符无监督 → 负漂移 → 折叠失败 → 均值占位修复（§5.7）
- 逆密度加权 cap 过大 → 中性退化 → cap=2

### 9.3 关键决策记录

1. **用现成 MoMPNN 而非自训**：用户明确偏好，多目标 DPO 权重直接受益。
2. **冻结 backbone 只训编码器**：防破坏 DPO 权重。
3. **主证据用 TM-score**（ESMFold 回折 → US-align），pLDDT 仅辅助。
4. **连续 soft prompt 而非离散 token**：pH 连续值精度无损。
5. **自回归而非扩散**：pI/电荷闭式可算，自回归精确 lookahead 更简单。
6. **验证主证据用 TM-score**：pLDDT 是模型自我置信度，可被先验欺骗。

---

## 10. 新机器配置 + 数据组织

### 10.1 新机器配置

完整的从零配置指南（克隆仓库、conda 环境、**下载 v7/v9 编码器权重**、重建数据、验证）见 **`docs/SETUP_NEW_MACHINE.md`**。

要点预告：
- 代码：`git clone git@github.com:Yu-Bk/ConfuMPNN.git`（含全部文档 + 脚本 + 示例 PDB）
- 外部源码：`git clone` LigandMPNN + MoMPNN（含权重）
- **自训编码器（v7/v9）**：从 GitHub Releases 下载（`gh release download`）
- 数据：CATH / RCSB 重新获取（脚本可重跑）或从组内 NAS 恢复

### 10.2 数据组织

数据集的划分、重建命令、SHA256 校验清单见 **`data/README.md`**。

划分逻辑（防泄漏）：
- **训练集**：`data/cath`（v7）、`data/ligand_train`（v9）
- **验证集**：`data/validation_pdbs`（10 未见蛋白）、`data/ligand_test`（5）、`data/transfer_test`（5）——全部从训练集排除

---

## 11. 术语表

| 术语 | 含义（类比） |
|------|-------------|
| **逆折叠** | 给定骨架设计序列（换材料、保模具） |
| **logits** | 模型对每个候选氨基酸的原始打分（可正可负） |
| **softmax** | 把打分变成概率（0~1，和为 1） |
| **temperature** | softmax 除数，<1 让概率更锐利（更确定） |
| **logit bias** | 采样前加到 logits 上的偏置，抬高/压低某些氨基酸 |
| **交叉熵 CE** | 损失函数：模型给正确答案的概率越接近 1 越好 |
| **KL 散度** | 两个概率分布的差异度 |
| **梯度下降** | 沿损失下降方向调参数的学习过程 |
| **过冲** | 模型把电荷调过头，远超目标 |
| **soft prompt** | 一组可学习的向量，作为"条件提示"注入模型 |
| **cross-attention** | 让每个位置按需读取提示信息的机制 |
| **ConditionEncoder** | 本项目训练的小网络（0.08M 参数），条件→soft prompt |
| **h_V** | backbone 编码出的结构特征（每个氨基酸位置一个向量） |
| **chain_mask** | 0/1 掩码，0=该位置固定不动 |
| **净电荷** | 序列在特定 pH 下所有带电残基的代数和 |
| **pI** | 净电荷=0 时的 pH（序列推导属性） |
| **HH 方程** | pH 与带电比例的关系式 |
| **GRAVY** | 序列疏水性指标（越正越疏水） |
| **TM-score** | 两结构相似度（>0.7 = 相同拓扑）|
| **ESMFold** | 从序列预测结构的深度学习模型（验证用）|
| **backbone** | 承载主体能力的预训练模型（MoMPNN/LigandMPNN）|
| **符号链接 symlink** | 指向另一文件的快捷方式（不复制内容）|

---

*本文档为 ConfuMPNN 项目唯一权威使用指南，随项目状态持续更新。最后更新：2026-08-27（v10 演进中）。*
