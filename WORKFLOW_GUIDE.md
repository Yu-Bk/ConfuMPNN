# ConfuMPNN 完整工作流程说明（工作汇报版）

> 日期：2026-08-18　作者：ConfuMPNN 项目
> 定位：整合全项目技术文档（`index/PROJECT_PLAN.md`、`index/PROJECT_EXTEND.md`、`index/DESIGN_CRITERIA.md`、`docs/TECH.md`、`docs/CONFIG.md`、各轮实验报告）为**一份完整、直观、可汇报**的流程说明。
> 内容覆盖：构建计划与思路演变 → 模块架构 → 核心公式与参数 → 数据流动 → 每轮训练结论 → 困难与解决（含文献参考）→ 复盘 → 下一步。

---

## 0. 项目一句话概括

**ConfuMPNN**：在结构条件逆折叠模型（LigandMPNN）上，首次加入 **pH 感知的电荷条件控制**——用户指定工作环境 pH（和可选的目标净电荷），模型生成**满足该 pH 电荷约束、能折叠回给定骨架、且电荷空间分布合理**的蛋白质序列。

**核心创新**：现有性质条件化工作（LaMBO-2、AntiBARTy 等）都是纯序列模型或抗体专用；在**显式建模配体原子上下文**的结构逆折叠模型上做 pH/电荷显式条件控制，本项目是第一例。

---

## 1. 研究问题与设计哲学

### 1.1 要解决的问题

给定蛋白骨架（PDB 结构）+ 工作 pH（可选：目标净电荷），生成：
- **该 pH 下净电荷符合目标**（H2 判据）
- **能折叠回原骨架**（H1 判据，TM-score≥0.70）
- **电荷空间分布合理**（H3 判据，不违反生物物理电荷规律）

### 1.2 为什么 pH 重要（生物学背景）

氨基酸侧链有可电离基团（Asp/Glu 的 -COOH、Lys/Arg 的 -NH₃⁺/胍基、His 的咪唑基）。**蛋白质净电荷随 pH 变化**：
- pH 越低 → 质子越多 → 酸性基团被质子化（带负电能力下降）→ 净电荷偏正
- pH 越高 → 相反 → 净电荷偏负

该关系由 **Henderson-Hasselbalch（HH）方程**描述（见 §2.1）。

### 1.3 设计哲学（关键认知）

1. **逆折叠 = 骨架固定、序列可重写**。序列改变是**设计行为**，不是破坏。
2. **"改 pI 还要求序列相似"自相矛盾**：pI 由序列决定，改 pI 必须改序列。
3. 可溶性/热稳定等新序列固有属性的下降是**设计权衡**，不是失控——除非破坏源自"折叠不回骨架"。
4. **pI 不作为直接输入**，而是生成序列的推导属性（由氨基酸组成 + pKa 表唯一确定），在验证阶段用作一致性检查。
5. 两个真实目标都要求**全新序列**：① 天然骨架（RF3 relax 微调后）→ 全新序列 + 理化性质≈天然；② 人工设计骨架 → 全新序列 + 简单理化预期（部分条件用占位符不控制）。

---

## 2. 核心科学原理与公式

### 2.1 可微电荷计算（`differentiable_charge.py`）

HH 方程用 **sigmoid 平滑近似**（让电荷对 pH 处处可微，供梯度反向传播）：

```
去质子化分数 = σ(ln10 · (pH − pKa))          # σ = sigmoid
酸性残基 D/E/C/Y：电荷 = −σ(ln10·(pH−pKa))    # pH 高 → 去质子化 → 带 −1
碱性残基 K/R/H  ：电荷 = +σ(ln10·(pKa−pH))    # pH 低 → 质子化 → 带 +1
N 端 α-NH₃⁺（pKa≈9.7）：+σ(ln10·(9.7−pH))
C 端 α-COOH（pKa≈2.3）：−σ(ln10·(pH−2.3))
```

游离 pKa 表（`pka.py`）：Asp 3.9、Glu 4.3、His 6.0、Cys 8.3、Tyr 10.1、Lys 10.5、Arg 12.5、N 端 ~9.7、C 端 ~2.3。

两个接口：
- `net_charge(seq, pH)`：字符串序列 → 净电荷 float（验证用）
- `net_charge_from_logits(logits, pH)`：解码器 logits → 期望净电荷（**可微**，softmax 概率对 20 种氨基酸电荷加权平均）

### 2.2 电荷可加性 → 逐位前瞻（`charge_lookahead.py`）

净电荷是**可加和**的。解码第 t 位时可估算"放某氨基酸后整条序列净电荷"：

```
Q_k = Q_fixed（已解码电荷和）+ q(aa_k, pH)（候选侧链电荷）
      + Q_expect_others（未解码位用 20 种 AA 平均电荷近似）+ Q_termini（端基常数）
```

**关键 bug 教训（softmax 平移不变性）**：bias 不能写成 `(Q_k − target)`——softmax 对常数平移不变（softmax(x+c)=softmax(x)），不依赖候选 k 的项被抵消，target 进不了分布。**正确写法让 target 进入依赖候选的交叉项**：

```
bias_k = strength · (target_charge − Q_current) · q(aa_k, pH)
```

- `Q_current < target`（欠正电）→ `(target−Q_current) > 0` → 正电候选得正 bias 被促进
- 随解码推进 `Q_current` 渐准，引导收敛（E1 验证：1BC8 target=+8/−8 → +8.06/−7.96）

### 2.3 结构感知过滤器（`structure_aware_filter.py`）

4 条空间规则，解码时实时对异常电荷聚集施加**负 bias**（抑制，非硬过滤）：

| 规则 | 检测内容 | 99 分位阈值 |
|------|---------|------------|
| 1. 空间电荷聚集 | 10Å 内同号强电荷（K/R 或 D/E）≥6 | 6 |
| 2. 盐桥过密 | 10Å 内正负电荷对 ≥4 | 4 |
| 3. 核心电荷渗入 | 埋藏(burial>0.8)且 8Å 内带电 ≥6 | 6 |
| 4. 同号电荷聚类 | 8Å 邻域同号电荷 ≥4 | 4 |

**阈值来源**：CATH 4.4 S40（34,653 域）采样 1,000 个、统计 151,519 残基位取 **99 分位**（超过 99% 天然蛋白 = 异常）。⚠️ 规则 4 原设计"8Å 连通图同号聚类"会把全蛋白连成一个大分量导致全量误触发 → 改为**每残基 8Å 邻域同号电荷数**（局部密度口径）。

### 2.4 条件向量设计（mask-aware，`condition_embedding.py`）

```
[pH, has_charge_flag, charge_val, has_pos_limit_flag, pos_limit_val, has_neg_limit_flag, neg_limit_val]  shape [7]
```

`has_X_flag`（0/1）告诉网络哪些值是真条件、哪些是占位符，**避免 0 值歧义**（"没指定"≠"目标就是 0"）。

### 2.5 条件编码器（Soft Prompt，NExT-Mol 风格）

```
条件向量 c [7]
  → Linear(7→64) → GELU → Linear(64→128) → GELU → Linear(128→4×128)
  → reshape [4, 128]    # 4 个 soft prompt token，拼到解码前缀（cross-attention 注入 h_V）
```

**为什么连续向量而非离散 token**：pH 要精确到 0.1（4.0–10.0 有 60 档），MolGPT 式离散 control token 精度大损；NExT-Mol 式连续 soft prompt 无精度损失。

### 2.6 复合损失（`losses.py`）

```
L = CE + λ_c·charge_deviation + λ_kl·KL_anchor + λ_keep·seq_keep
```

- **CE**：标准自回归交叉熵（锚 native 序列，保结构匹配）
- **charge_deviation**：|期望净电荷 − target|，温度化（`charge_temp=0.5`，让训练优化的分布≈推理采样分布，根治 ~2.9× 电荷过冲）
- **KL_anchor**：条件化输出 vs 无条件 backbone 输出的分布距离（防条件注入失控）
- **seq_keep**（`sequence_keep_loss`）：以**无条件 argmax 序列**为锚做 CE——无改 pI 需求时（target=native）不扰动，有需求时才改写。比 KL 更直接（KL 管分布距离，管不住 argmax 翻盘）

实际训练权重：λ_c=0.5、λ_kl=0.05、λ_keep=0.5。

### 2.7 逆密度加权（imbalanced regression 标准解法，v4+）

```python
w = k / (density_norm[bucket(target)] + eps)    # k=1.0, eps=1e-3
cd_loss *= min(w, cap)                          # cap=2（v5/v6）
```

- **原理**：训练集电荷分布偏负（均值约 −2.8），高正电 target 是**稀疏区**，常规回归会欠拟合稀有 target → 高正电外推过冲。按 target 密度逆加权，稀有 target 权重更大。
- **cap 是权重上限**：防止过度放大稀有样本、牺牲高频样本（v4 用 cap=5 过头 → 1UBQ 退化；v5/v6 降 cap=2）。
- **文献依据**：imbalanced regression 的标准解法（US11720818B2 / arXiv 2506.01486）。

### 2.8 占位符语义（目标 2）

训练注入 `placeholder_prob=0.15` 的占位样本：电荷条件置为**均值占位**（has_charge=1 + 值=训练均值），**并施加电荷损失**（target=均值）。让"电荷不控制"落在温和可折叠默认，而非放任漂移。
（v3 教训：占位样本跳过电荷损失 → 模型学"维度不控制" → 无条件负漂移 → 折叠全失败。均值占位 + 施加损失后完全修复。）

---

## 3. 系统架构（模块流程图）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            run_guided.py（主入口）                        │
│    --pdb --pH --target_charge --preset --cond_encoder --fixed_residues    │
└───────┬─────────────────────────────────────────────────────────────────┘
        │
   ┌────▼─────┐   ┌───────────────┐   ┌──────────────────┐   ┌────────────┐
   │ 输入 PDB  │──▶│  featurize     │──▶│  骨架/特征张量     │──▶│  解码器     │
   │ parse_PDB│   │ (LigandMPNN)  │   │  (X,E_idx,S)     │   │ProteinMPNN │
   └──────────┘   └───────────────┘   └────────┬─────────┘   │ (MoMPNN)   │
                                               │             └─────▲──────┘
        ┌──────────────────────────────────────┼──────────────────┼──────┐
        │  条件注入（Level 2，Phase 3 起默认）    │                  │      │
        │  ConditionEncoder(cond_vec[7]) ─────▶ 4×soft prompt     │      │
        │  经 cross-attention 注入 h_V          ───────────────▶  │      │
        └──────────────────────────────────────┘                  │      │
   ┌───────────────────────────────────────────────────────────────┘      │
   │ 引导采样（guided_sampler.py）：bias = 电荷 lookahead + 结构过滤        │
   │ probs = softmax((logits + bias) / temperature)                       │
   └──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   候选序列 (n=20) + charge_stats.json
                              │
   ┌──────────────────────────▼────────────────────────────┐
   │ 打分管线（验证）：ESMFold 回折 → us-align TM-score      │
   │   → Protein-Sol %sol → TemBERTure Tm                  │
   └──────────────────────────┬────────────────────────────┘
                              ▼
              DESIGN_CRITERIA v2 判定（H1/H2/H3/S1*-S4）
```

**模块清单**（`code/src/`）：

| 模块 | 功能 | 关键接口 |
|------|------|---------|
| `pka.py` | 游离 pKa 表 + 带电类型 | `PKA_SIDECHAIN`、`AA_TO_IDX` |
| `differentiable_charge.py` | HH 平滑电荷计算 | `net_charge(seq,pH)`、`net_charge_from_logits(logits,pH)` |
| `isoelectric_point.py` | pI 二分搜索（验证） | `find_pI(seq)` |
| `structure_aware_filter.py` | 4 条空间规则 → [L,21] bias | `compute_bias()`、`load_preset()` |
| `charge_lookahead.py` | 逐位电荷前瞻 bias | `bias_at()`、`make_dynamic_callback()` |
| `guided_sampler.py` | 引导解码循环 | `GuidedSampler.sample()` |
| `condition_embedding.py` | Soft Prompt 条件编码器 | `ConditionEncoder(c)`、`make_condition_vector()` |
| `conditioned_sampler.py` | cross-attention 条件注入采样 | `conditioned_sample()` |
| `losses.py` | 复合损失 | `composite_loss()`、`sequence_keep_loss()` |
| `train_finetune.py` | 微调训练 | 混合目标 + 占位符 + 逆加权 |

---

## 4. 数据流动（端到端）

### 4.1 数据来源与标签构建

```
CATH 4.4 S40（34,653 结构域，非冗余，818MB）
   │  parse_domain 提取 Cα 坐标 + 序列（≥20 残基）
   ▼
每个域 → 多 pH 标签（n_pH=8）：pH = uniform(4.0, 10.0) 连续采样
    → 每 pH 算 net_charge(seq, pH) + find_pI(seq)
   ▼
labels.npz：domain_ids / seqs / coords / pH / charge / pI
   ▼
计算条件向量 μ/σ（7 维）→ 写入 condition_defaults.yaml（normalization）
```

### 4.2 数据采样策略演变（关键迭代）

| 版本 | 采样策略 | 域数 | 电荷分布 | 问题 |
|------|---------|------|---------|------|
| Phase 2 | 随机 999 域 | 999 | 偏负（native>+10 仅 1.3%）| 高正电 target 分布外 → 过冲 |
| v4 | 8 箱分层 ×100 | 776 | 均匀（正电 32.8%）| 1UBQ 退化（逆加权 cap5 过头）|
| v5 | 8 箱分层 ×300 | 2,176（箱8 仅 76）| 均匀 | 1UBQ 仍失败；**根因=分层砍中性多样性 97%** |
| **v6** | **三类平衡** | **7,208** | acid 2,500 / neutral 2,500 / basic 2,208 | 1UBQ 恢复，但极端正电(+15~+20)仅 76 域仍不足 |

**v6 三类平衡**（`build_labels_v2.py --class_balance --per_class 2500`）：按 native 电荷@7.4 分三类——acid(<−5)、neutral(−5~+5)、basic(>+5)。acid/neutral 各抽 2,500，**basic 全保留**（2,208）。既保证三类数量相近，又保住中性骨架多样性（中性域从 600→2,500）和碱性多样性最大化。

### 4.3 验证蛋白选择

| PDB | native 电荷 | 长度 | 蛋白 | 代表性 |
|-----|------------|------|------|--------|
| 1BC8 | +8.90 | 93aa | SAP-1 ETS 转录因子 DNA 结合域 | 正电富集、核酸结合 |
| 1CRN | −0.68 | 46aa | Crambin（种子）| 极小、疏水、native 难折叠 |
| 1UBQ | +0.03 | 76aa | Ubiquitin（泛素）| 典型中性可溶球状 |
| 2LZM | +7.80 | 164aa | T4 溶菌酶 | 较大、正电、经典工程模型 |
| 1b24A01 | +8.01 | 95aa | 正电验证蛋白（v4 新增）| 域外泛化验证 |

**泄漏检查**：全部验证 PDB 从训练集排除（v6 加 `--exclude`，曾拦下 1b24A01 进入训练集）。
**可折叠性预筛**：1a87A02 已弃用（native ESMFold pLDDT 49.8 本身难折叠，不适合判据）。

---

## 5. 训练与验证方法论

### 5.1 训练目标与参数（v5/v6 稳定版）

```bash
train_finetune.py --device cuda:1 --epochs 30 \
  --perturb_prob 0.3 --perturb_scale 8 --placeholder_prob 0.15 \
  --lambda_keep 0.5 --charge_temp 0.5 \
  --loss_reweight 1 --reweight_k 1.0 --reweight_eps 1e-3 --reweight_cap 2 \
  --labels <labels_balanced_v6.npz> --out_dir output/finetune_v6
```

| 参数 | 值 | 作用 |
|------|-----|------|
| `perturb_prob` | 0.3 | 70% 样本 target=native（自洽），30% 扰动电荷（制造偏移学习信号）|
| `perturb_scale` | 8 | 扰动幅度 ±1~8（v2 从 4 加大到 8，让模型见过"补 6-8 单位电荷"需求）|
| `placeholder_prob` | 0.15 | 占位符样本比例（均值占位语义）|
| `charge_temp` | 0.5 | 电荷损失温度化（根治 ~2.9× 过冲）|
| `lambda_keep` | 0.5 | seq-keep 正则权重（保结构、防无需求时重写）|
| `loss_reweight` + cap=2 | 开 | 逆密度加权（治高正电外推过冲）|

**模型结构**：MoMPNN backbone（ProteinMPNN 纯 backbone，1.66M 参数，**冻结**）+ ConditionEncoder（74,880 参数，唯一训练对象）。

### 5.2 判断标准 v2（`index/DESIGN_CRITERIA.md`）

**硬约束（任一不满足 → FAIL）**：

| # | 判据 | 阈值 |
|---|------|------|
| H1 | 结构自洽（ESMFold 回折 TM-score）| TM 中位 ≥0.70；失败率（TM<0.5）≤10% |
| H2 | 电荷目标命中 | \|平均实际 − target\| ≤ 2.0 |
| H3 | 电荷聚集合法 | 违规率 ≤ 基线 +5pp |

**软判据（报告，不单独判 FAIL）**：
- **S1\***：identity 落 0.4–0.7 健康 + 防坍塌监控（pairID<0.8、位置熵不骤降）。v1→v2 从"硬判 ≥0.7"降级，因两个真实目标都要全新序列，序列重写是期望行为。
- **S2**：pLDDT / %sol / Tm 绝对值报告（设计权衡）。
- **S3**：占位符语义（t2_ph 臂折叠达标 = 占位不破坏控制）。
- **S4**：位点固定 100% 保持。

**判定依据**：TM-score>0.7 = 相同拓扑（Zhang & Skolnick 2004）；阈值先验设定 + 留一蛋白检查，不做阈值搜索（防过拟合）。

### 5.3 复验方案（对齐两真实目标）

```
5 PDB × 6 臂 × n=20（固定 seed 协议，防挑 seed）：
  t1_cond      : target=round(native) + 固定 4 个疏水核心位点  → 目标 1 形态（判 H1/H2/S4）
  t1_base      : target=round(native)，不固定                  → S1* 参照
  t2_pos       : target=native+3（温和正电）                    → 目标 2 从零设 pI
  t2_pos_extreme: target=native+8（极端正电）                   → 高正电外推测试
  t2_neg       : target=native−5（负电）                        → 目标 2 负电
  t2_ph        : target=均值占位                                → S3 占位符语义（不判 H2）

打分管线：ESMFold 回折 → us-align TM-score → Protein-Sol → TemBERTure（HF_HUB_OFFLINE=1）
判定：phase3_v2_stats.py 输出 v2_judgment.json
```

---

## 6. 迭代历程与每次训练结论（分支节点）

### 6.1 时间线总览

```
Phase 0-1（环境+引导采样）→ E0-E4（MoMPNN 接入）→ Phase 2（条件微调）
→ Phase 3（条件注入验证+校准）→ v2-v6（通用模型数据迭代，5 轮）
```

### 6.2 各阶段详情

**① Phase 1（Level 1 引导采样，不改模型）**
- 实现可微电荷、pI 查找、结构过滤器、电荷前瞻、引导采样器
- **36/36 单元测试通过**；电荷引导精确命中 target
- 诚实边界：**无引导时模型不感知 pH**（同一蛋白各 pH 序列相同）→ 必须靠引导或微调

**② E0：MoMPNN 可用性调研**（文献：ProtAlign, ICLR 2026）
- MoMPNN = 多目标 DPO 微调的 ProteinMPNN 变体，优化"可设计/热稳/可溶"三目标
- **结论：权重=纯 backbone ProteinMPNN，`strict=True` 8/8 可加载** ✅

**③ E1：三目标对照（MoMPNN vs 原版 LigandMPNN）**
- 可溶 **+12.8**（Protein-Sol %sol）、热稳 **+7.8°C**（TemBERTure）、电荷更准、pLDDT 持平
- ⚠️ 教训：GRAVY 疏水性代理曾误判 MoMPNN 更疏水，真实 Protein-Sol 相反——**打分必须用真实工具**

**④ E1b：4 PDB × 3pH × 3target 扩展验证**
- 电荷响应 **24/24 单调**；MoMPNN **16/16 全优**（4 指标 × 4 PDB）——E4 设默认生成器有强依据

**⑤ E4：MoMPNN 设为默认生成器**

**⑥ Phase 2：条件微调（Level 2）**
- CATH 999 域 × 8 pH = 7,992 标签；ConditionEncoder soft prompt 注入
- 防失控设计：冻结 backbone + KL 锚定，事后 E1b 验证 %sol/Tm 不掉
- 混合目标：50% 自洽 + 50% 扰动（±Uniform[1,4]）——纯自洽会让 CE+电荷同时被重建 native 满足，学不到电荷偏移

**⑦ Phase 3：条件注入验证 + 电荷校准**
- **pH 响应 Go/No-Go 4/4 PDB 通过**：target 单调、跨 pH identity<100% → 模型真正感知 pH
- **发现 ~2.9× 电荷过冲**：训练优化 softmax 期望电荷 E[Q]，推理测采样序列电荷，CE 训练使模型自信 → 采样更极端
- 解法演进：推理侧线性校准（gain=2.57）→ **训练侧温度化根治**（charge_temp=0.5，增益 2.57→1.04）✅
- **n=20 扩样本推翻 n=5 假阴性**：32 组配对检验 23 组显著，条件注入显著降 %sol/Tm——这是**设计权衡**（序列重写固有代价），非失控
- **判断标准 v1→v2**：S1 硬判降级为相似性软区间（两目标要全新序列）

**⑧ 第十四轮：seq-keep 正则（治 S1）**
- 原生标签 50%→70%（perturb_prob 0.3）+ `sequence_keep_loss`（λ=0.5）
- 结果：H1 全达标、折叠失败 0%、pLDDT 掉落大幅修复；S1 identity 0.45→0.67 仍不足 0.7——但**方向修正后 S1 判据作废**（见下）

**⑨ 第十五轮：方向修正（对齐两真实目标）**
- 用户纠正：目标要**全新序列**，identity≥0.7 与"全新序列"矛盾 → **S1 作废**，seq-keep 不再加压
- 实现 `--fixed_residues`（位点固定，复用 chain_mask 原生机制）+ `--placeholder_prob 0.15`（占位符样本）+ `perturb_scale 4→8`
- 制定判断标准 v2 + 验证计划 v2

**⑩ 第十六轮：v2 复验（初步结果）**
- 目标 1 形态基本成功（H1 12/12、S4 100%、无坍塌）
- 负电 target 4/4 精确命中；**正电过冲**（1BC8/2LZM）；**占位符折叠全失败**（S3 根因：训练偏负 + 无条件负漂移）

**⑪ 第十七轮：占位符修复（均值占位）**
- 修复：占位统一均值占位（has_charge=1 + 值=均值）+ 占位样本施加电荷损失
- 结果：**占位符折叠完全修复**（t2_ph TM 0.89-0.97）、全部 20 臂 H1 通过、负电 4/4 保持

**⑫ 第十八轮：v4 通用模型（分层采样 + 逆加权）**
- 用户需求：通用模型根治正电过冲。文献支撑：分层平衡采样 + 逆密度加权（imbalanced regression）
- 数据：8 箱分层 776 域；训练：cap=5
- 结果：**1BC8 全 6 臂命中**（极端+17 dev 0.37）、2LZM 修复、1b24A01 泛化 4/5；**但 1UBQ 退化 0/5**（逆加权过头牺牲中性）

**⑬ 第十九轮：v5（cap=2 + 扩大数据）**
- 数据 2,176 域（每箱 300）；训练 cap=2
- 结果：H1 折叠 **30/30 全过**、H2 16/25；**1UBQ 仍失败**
- **🚨 根因发现**：分层"每箱 300"砍掉中性骨架多样性 97%（1UBQ 中性泛化失败元凶）+ 箱8 高正电仅 76 域

**⑭ 第二十轮：v6（三类平衡采样）**
- 数据 7,208 域（acid 2500 / neutral 2500 / basic 2208 全保留）
- 结果（当前最优）：
  - **1UBQ 大幅恢复 1/5→4/5**（证实根因判断）
  - **极端正电普遍改善**（1CRN 2.98→0.78、2LZM 6.71→2.36）
  - **H2 19/25**（v5 16/25）；**目标 1 形态 t1_cond 5/5 全达标**
  - ⚠️ 折叠退化（H1 25/30）：1CRN/1b24A01 部分臂失败率超标；1BC8 正电退化（极端 target 样本仍不足）

---

## 7. 遇到的困难与解决（分支节点，含文献参考）

| # | 困难 | 根因 | 解决/参考 |
|---|------|------|----------|
| 1 | softmax 平移不变性使 bias 失效 | bias 含不依赖候选的常数项被抵消 | 改 `bias_k = strength·(target−Q)·q_k`（进入交叉项）|
| 2 | 结构过滤器规则 4 全量误触发 | 连通图口径随蛋白大小暴涨 | 改局部密度口径（8Å 邻域同号数），p99=4 |
| 3 | 电荷过冲 ~2.9× | 训练优化 E[Q] vs 推理测采样电荷，CE 自信放大 | 训练侧温度化 `charge_temp=0.5`（增益→1.04）|
| 4 | 条件注入显著降 %sol/Tm | 序列重写固有代价（设计权衡）| 判断标准：S2 软判据，不判 FAIL |
| 5 | 正电 target 过冲 | 训练数据偏负，高正电分布外 | ①分层采样补覆盖 ②逆密度加权（imbalanced regression 文献 US11720818B2/arXiv 2506.01486）|
| 6 | 1UBQ 中性泛化失败 | 分层"每箱 300"砍中性骨架多样性 97% | 三类平衡采样（v6），中性域 600→2,500 |
| 7 | 占位符折叠全失败 | 占位样本跳过电荷损失→无条件负漂移 | 均值占位 + 占位样本施加电荷损失（v3 修复）|
| 8 | 逆加权 cap=5 过头（1UBQ 退化）| naive 加权牺牲高频样本 | 降 cap 5→2（v5/v6）|
| 9 | 极端正电(+15~+20)仍不足 | CATH 碱性域仅 6.3%、极端箱仅 76 域 | **结构性瓶颈**：需外部补碱性 PDB 或课程学习 |
| 10 | 折叠退化（v6 H1 25/30）| 电荷驱动的过激替换对弱折叠/域外蛋白保真下降 | 待评估 λ_keep/扰动调整（v7）|
| 11 | 数据泄漏（1b24A01 进训练集）| 三类平衡"碱性全保留"收进验证蛋白 | `--exclude` 排除 + 泄漏检查（前期检查流程）|

**主要文献支撑**：
- **ProtAlign/MoMPNN**（ICLR 2026）：多目标半在线 DPO + 自适应 margin；多目标不加 margin 会退化的消融证据（支撑本项目的条件边界思想）
- **P2 ResiDPO / EnhancedMPNN**（arXiv 2506.00297）：RCL 选择性保序列；pLDDT 作可设计性代理
- **NExT-Mol**：连续 soft prompt 条件注入（vs MolGPT 离散 token 反例）
- **Zhang & Skolnick 2004**：TM-score>0.7 = 相同拓扑（H1 阈值依据）
- **imbalanced regression**（US11720818B2 / arXiv 2506.01486）：逆密度加权
- **P3 Weighted-score DPO**：坍塌教训（防坍塌监控依据）

---

## 8. 关键决策记录

1. **为什么 LigandMPNN 而非 AF3/RF3**：AF3/RF3 的序列是输入不是输出；RF3 定位为验证工具。
2. **为什么自回归而非扩散**：pI/电荷闭式可算，自回归精确 lookahead 比扩散 guidance 更简单直接。
3. **为什么 NExT-Mol 式连续 soft prompt**：pH 连续值需连续向量编码（精度无损）。
4. **为什么不改 vocabulary**：LigandMPNN/RF3 都不区分质子化态 His；电荷约束靠条件嵌入 + bias 实现。
5. **优先用文献现成模型**（MoMPNN）而非自训练（用户明确偏好），条件嵌入 context 不变。
6. **冻结 backbone 只训编码器**：MoMPNN 价值在 DPO 权重，全量微调有破坏风险。
7. **验证主证据用 TM-score**（ESMFold 回折 → us-align），pLDDT 仅辅助（模型自我置信度可被先验欺骗）。
8. **阈值先验设定 + 留一蛋白检查**，不做阈值搜索（防过拟合）。
9. **S1 判据作废**（两目标要全新序列），identity 降级为软区间 + 防坍塌监控。
10. **三类平衡采样**（acid/neutral/basic 相近 + basic 全保留）替代"每箱等量"。

---

## 9. 复盘（项目层面）

### 9.1 方法论上的成功
- **"先立标准再训练"**：判断标准 v1→v2 的演进（S1 作废）避免了在错误目标上浪费迭代。
- **"诚实边界"文化**：无引导时模型不感知 pH、n=5 假阴性被 n=20 推翻、1a87A02 弱验证蛋白弃用——每次都如实报告负面结果。
- **打分用真实工具**：GRAVY 代理误判教训后，坚持 ESMFold/Protein-Sol/TemBERTure 实测。

### 9.2 数据驱动的根因排查（v5→v6 的典范）
1. v5 发现 1UBQ 失败但归因错误（曾以为是 cap）→ 查数据箱容量才发现中性多样性被砍 97%
2. 用"v3 全量数据命中 / v4-v5 分层子集失败"的对照**证明**根因是数据多样性而非训练策略
3. v6 三类平衡后 1UBQ 恢复 4/5 → **假设验证闭环**完成

### 9.3 尚存的短板
- **极端正电（+15~+20）仍是结构性瓶颈**：CATH 内仅 76 域，逆加权 cap=2 无法完全弥补
- **折叠保真 vs 电荷精度**的权衡仍敏感：v6 电荷改善的同时 1CRN/1b24A01 折叠略退化
- 验证蛋白偏少（5 个）、多为正电/中性，负电极端蛋白覆盖不足

### 9.4 工程与协作复盘
- 长任务启动前的**前期检查清单**（计划/数据/环境/dry-run/泄漏）已固化为工作准则，成功拦截 1 次数据泄漏、多个训练 bug
- 自动化收尾流程（后台监控 → 自动复验 → 自动存档 push）保证了无人值守时的完整闭环

---

## 10. 下一步计划（v7 候选）

| 优先级 | 方向 | 说明 |
|--------|------|------|
| 1 | **外部补充碱性 PDB** | 下载 native>+10 蛋白，预筛 ESMFold 可折叠性 + 泄漏检查 + 建标签，加入三类平衡数据（治极端正电根因）|
| 2 | **极端 target 课程学习** | 先中性/温和充分训练，再引入极端 target（用户提过"先学简单再学难"）|
| 3 | **折叠退化排查** | 评估 λ_keep/扰动幅度调整，保护 1CRN/1b24A01 类弱折叠/域外蛋白 |
| 4 | R1 天然蛋白对参照 | CATH 同 superfamily 找"骨架相似 pI 不同"蛋白对（验证目标可达性）|
| 5 | 真实骨架泛化 | 用户提供 RF3 relax/人工骨架后复用同一流程（考验最终泛化）|

---

## 附：快速查阅索引

| 主题 | 文档 |
|------|------|
| 第一版技术计划 | `index/PROJECT_PLAN.md` |
| 第二版拓展计划（MoMPNN 接入）| `index/PROJECT_EXTEND.md` |
| 判断标准 v2 | `index/DESIGN_CRITERIA.md` |
| 技术原理/公式 | `docs/TECH.md` |
| 配置/参数 | `docs/CONFIG.md` |
| 各轮实验报告 | `analysis/report/`（E1 / Phase1-3 / v4 / v5 / v6）|
| 完整项目状态 | memory `confumpnn-project-status.md` + `session/2026-08-16_PROJECT_STATUS.md` |
| 文档索引 | `index/DOCUMENT_INDEX.md` |

*本文档为全项目技术脉络的整合说明，随项目进展持续更新。*
