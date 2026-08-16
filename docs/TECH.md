# ConfuMPNN 技术文档

> 对应 `index/PROJECT_PLAN.md`（第一版）Phase 1，Level 1 引导采样。
> 本文档解释系统**怎么工作、为什么这么设计**，面向有生物学背景的读者。

---

## 一、问题与思路

**要解决的问题**：给定一个蛋白的骨架结构（PDB），以及一个**工作环境 pH**（可选：目标净电荷），生成一段**在该 pH 下净电荷符合目标、且空间上电荷分布合理**的蛋白序列。

**为什么 pH 重要（生物背景）**：氨基酸侧链有可电离基团（酸性 Asp/Glu 的 -COOH、碱性 Lys/Arg 的 -NH3+/胍基、His 的咪唑基等）。蛋白质的**净电荷随环境 pH 变化**——pH 越低，质子越多，酸性基团被质子化（带负电能力下降），碱性基团保持正电，净电荷偏正；pH 越高则相反。这个"净电荷随 pH 变化"的关系由 **Henderson-Hasselbalch（HH）方程**描述：

```
去质子化分数 = 1 / (1 + 10^(pH − pKa))
```

pKa 是每种基团的固有常数（如 Asp≈3.9、Glu≈4.3、His≈6.0、Lys≈10.5）。工程上常用"pI（等电点）"描述蛋白在某个 pH 下净电荷为零。

**核心设计决策**：**不改模型**（LigandMPNN），通过**解码时的 logit bias 引导**让生成的序列满足约束。这是 Level 1（引导采样）；Level 2（条件微调，改模型）见第七节。

---

## 二、系统架构

```
                ┌─────────────────────────────────────────────────────┐
                │                     run_guided.py                    │
                │                       （主入口）                       │
                └────────────────────────┬────────────────────────────┘
                                         │
        ┌────────────────────────────────┼───────────────────────────────┐
        │                                │                                │
   ┌────▼─────┐                   ┌──────▼───────┐                ┌──────▼───────┐
   │ LigandMPNN│                  │ ChargeLookahead│              │StructureAware│
   │  (解码器)  │<── bias 注入────  │ 动态电荷前瞻    │              │   Filter     │
   │           │                  │ （管总量）      │              │ （管空间分布）  │
   └────▲─────┘                   └───────────────┘              └──────────────┘
        │                                  │                              │
   ┌────┴─────┐                     ┌──────▼───────┐               ┌─────▼──────┐
   │ featurize│                     │ differentia- │               │   pka.py    │
   │ parse_PDB│                     │ ble_charge   │               │  pKa 表     │
   └──────────┘                     └──────────────┘               └────────────┘
```

**两条正交约束**（可叠加，`guided_sampler.py` 的 `make_dynamic_callback` 合并）：
1. **电荷 lookahead 管"总量"**：每一步看"整条序列还差多少电荷到 target"，去推动当前位点的候选氨基酸。
2. **结构感知过滤器管"空间分布"**：检查已解码残基是否在空间上形成异常的电荷聚集（如同号电荷扎堆、盐桥过密、电荷渗入疏水核心），有则抑制相关位点继续放带电残基。

---

## 三、核心原理

### 3.1 可微电荷计算（`differentiable_charge.py`）

HH 方程用 **sigmoid 平滑近似**（让电荷对 pH 处处可微，Level 2 训练需要梯度）：

```
去质子化分数 = σ(ln10 · (pH − pKa))        # σ 是 sigmoid
酸性残基 D/E/C/Y：电荷 = −σ(ln10·(pH−pKa))   # pH 高 → 去质子化 → 带 −1
碱性残基 K/R/H  ：电荷 = +σ(ln10·(pKa−pH))   # pH 低 → 质子化 → 带 +1
N 端 α-NH3+（pKa≈9.7）：+σ(ln10·(9.7−pH))
C 端 α-COOH（pKa≈2.3）：−σ(ln10·(pH−2.3))
```

提供两个接口：
- `net_charge(seq, pH)`：**字符串序列** → 净电荷 float（Phase 1 验证用）
- `net_charge_from_logits(logits, pH)`：**解码器 logits** → 期望净电荷（可微，Phase 2 训练用，softmax 概率对 20 种氨基酸电荷加权平均）

### 3.2 电荷可加性 → 逐位前瞻（`charge_lookahead.py`）

净电荷是**可加和**的（各残基贡献相加）。因此在解码第 t 位时，可以估算"如果放某种氨基酸，整条序列净电荷会到多少"：

```
Q_k = Q_fixed（已解码残基的电荷和）
    + q(aa_k, pH)（候选氨基酸 k 的侧链电荷）
    + Q_expect_others（其余未解码位用 20 种 AA 平均电荷近似）
    + Q_termini（N/C 端常数）
```

**关键 bug 教训（softmax 平移不变性）**：bias 不能写成 `(Q_k − target)`。因为 softmax 对常数平移不变（softmax(x+c)=softmax(x)），不依赖候选 k 的项会被抵消，target 根本进不了分布。正确写法让 target 进入**依赖候选的交叉项**：

```
bias_k = strength · (target_charge − Q_current) · q(aa_k, pH)
```

- `Q_current < target`（还欠正电荷）→ `(target−Q_current) > 0`，正电候选（q_k>0）得正 bias 被促进、负电候选被抑制
- 随解码推进 `Q_current` 越来越准，引导渐近收敛到 target（E1 验证：1BC8 target=+8/−8 → 平均净电荷 +8.06/−7.96）

### 3.3 结构感知过滤器（`structure_aware_filter.py`）

4 条空间规则，在解码时**实时**检查已解码残基，对异常聚集施加负 bias（抑制，不是硬过滤）：

| 规则 | 检测内容 | 阈值（99 分位） |
|------|---------|----------------|
| 1. 空间电荷聚集 | 某位置 10Å 邻域内已解码同号强电荷（K/R 或 D/E）≥6 | 6 |
| 2. 盐桥过密 | 10Å 内正负电荷对（min(正,负)）≥4 | 4 |
| 3. 核心电荷渗入 | 埋在核心（burial>0.8）且 8Å 内带电残基 ≥6 | 6 |
| 4. 同号电荷聚类 | 某位置 8Å 邻域内同号电荷 ≥4 | 4 |

**阈值来源**：从 CATH 4.4.0 非冗余 S40 数据集（34,653 结构域）随机采样 1,000 个、统计 151,519 个残基位，取各特征分布的 **99 分位**（超过 99% 天然蛋白 = 异常聚集）。统计脚本 `code/tests/threshold_stats.py`，原始分布 `code/output/threshold_stats.csv`。

⚠️ **统计口径修正（重要）**：规则 4 原设计是"8Å 连通图同号聚类"，但连通图会把整个蛋白连成一个大分量，同号电荷总数随蛋白大小暴涨（采样显示 p50 就 17）→ 规则几乎全量误触发。已改为**每残基 8Å 邻域同号电荷数**（局部密度），p99=4 合理。

### 3.4 引导采样（`guided_sampler.py`）

复刻 LigandMPNN 的解码循环（batch=1），每解码一个残基调用 `bias_callback(S_cur, t)` 实时计算该位置 bias：

```
probs = softmax((logits + bias_t) / temperature)
```

`bias_callback` 由 `make_dynamic_callback` 构造，合并电荷 lookahead 与结构过滤器的 bias（两者正交相加）。

---

## 四、模块清单（`code/src/`）

| 模块 | 功能 | 关键接口 |
|------|------|---------|
| `pka.py` | 游离 pKa 表 + 带电类型常量 | `PKA_SIDECHAIN`、`AA_TO_IDX`、`STRONG_POSITIVE/NEGATIVE` |
| `differentiable_charge.py` | HH 平滑电荷计算 | `net_charge(seq,pH)`、`net_charge_from_logits(logits,pH)` |
| `isoelectric_point.py` | pI 二分搜索（验证用） | `find_pI(seq)` |
| `structure_aware_filter.py` | 4 条空间规则 → [L,21] bias | `StructureAwareFilter.compute_bias(seq_int)`、`load_preset()` |
| `charge_lookahead.py` | 逐位电荷前瞻 bias | `ChargeLookahead.bias_at()`、`make_dynamic_callback()` |
| `guided_sampler.py` | 引导解码循环 | `GuidedSampler.sample(feature_dict, bias_callback)` |
| `condition_embedding.py` | Soft Prompt 条件编码器（**Phase 2**） | `ConditionEncoder(c)`、`make_condition_vector()` |
| `losses.py` | 复合损失（**Phase 2**） | `composite_loss()` |

---

## 五、关键设计决策（含取舍）

1. **不改模型，用 bias 引导**（Level 1）：改动小、可解释、调试容易；代价是模型自身没有 pH 先验（见验证边界），需要显式引导或 Level 2 微调。
2. **bias 必须进交叉项**（softmax 平移不变性）：见 3.2，这是本项目踩过并修复的核心 bug。
3. **两条约束正交分离**：电荷管总量、结构过滤管空间，可独立调强度（`--strength`）与预设。
4. **阈值用统计 99 分位**而非拍脑袋：以天然蛋白库（CATH）的真实分布为基准，避免人工拍阈值过严/过松。
5. **自由能 vs 引导强度**：`--strength` 过大破坏模型先验（可折叠性），建议从 0.2–0.5 起步（PROJECT_PLAN 4.1）。

---

## 六、验证结果摘要

| 验证 | 结果 |
|------|------|
| 单元测试 `test_all.py` | **36/36 通过** |
| 冒烟测试（1BC8 真实采样） | seq_rec≈0.49，电荷引导正常 |
| 电荷命中（charge_lookahead） | 1BC8 target +8/0/−8 → +8.06/+0.23/−7.96 |
| E1 三目标对照（MoMPNN vs 原版） | 可溶 +12.8、热稳 +7.8°C、电荷更准、pLDDT 持平 |
| E1b 扩展（4 PDB × 3pH × 3target） | 电荷响应 24/24 单调；MoMPNN 16/16 全优 |
| 结构过滤器阈值 | CATH S40 统计校准，36 测试通过 |
| **诚实边界** | **无引导时模型不感知 pH**（同一蛋白各 pH 序列相同），电荷差异纯来自物理计算——模型 pH 感知必须靠引导或微调 |

---

## 七、Phase 2 展望（条件微调，Level 2）

Level 1 的边界（模型无 pH 先验）正是 Phase 2 要解决的。已就绪的模块：
- `condition_embedding.py`：条件向量 `[7]`（pH + 可选净电荷/局部电荷上限，mask-aware）→ 4 个 soft prompt token 拼到解码前缀
- `losses.py`：复合损失 `CE + λ_c·电荷偏差 + λ_l·结构惩罚 + λ_dpo·DPO`
- 数据：`data/cath/`（S40 坐标+序列，可加多 pH 标签）

训练前待办：条件向量 μ/σ 标准化 → 写入 `condition_defaults.yaml`；编写微调脚本。
