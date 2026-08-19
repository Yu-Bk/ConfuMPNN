# ConfuMPNN 技术原理（参考）

> **权威完整指南**：`WORKFLOW_GUIDE.md`（根目录）。本文档保留技术公式与机制，供快速查阅。
> 更新至 v9 定稿（2026-08-19）。

---

## 一、问题与思路

给定蛋白骨架（PDB）+ 工作 pH（可选：目标净电荷），生成**净电荷符合目标、能折叠回骨架、电荷分布合理**的序列。

净电荷随 pH 变化（HH 方程）：pH 低 → 净电荷偏正；pH 高 → 偏负。

两条技术路线：
1. **引导采样**（Level 1，不改模型）：解码时注入电荷前瞻 + 结构过滤 bias。
2. **条件注入**（Level 2，主线，模型 pH 感知）：ConditionEncoder 生成 soft prompt → cross-attention 注入 backbone。

---

## 二、可微电荷计算（`differentiable_charge.py`）

HH 方程 sigmoid 平滑近似（处处可微，供梯度）：

```
去质子化分数 = σ(ln10 · (pH − pKa))
酸性 D/E/C/Y：电荷 = −σ(ln10·(pH−pKa))
碱性 K/R/H  ：电荷 = +σ(ln10·(pKa−pH))
N 端（pKa≈9.7）：+σ(ln10·(9.7−pH))
C 端（pKa≈2.3）：−σ(ln10·(pH−2.3))
```

两个接口：
- `net_charge(seq, pH)`：字符串序列 → 净电荷（验证用）
- `net_charge_from_logits(logits, pH, temperature)`：logits → 期望净电荷（可微，训练用）

`temperature < 1` 让训练优化的期望电荷 ≈ 推理采样电荷（**根治过冲**，见 §四）。

---

## 三、电荷前瞻 bias（Level 1，`charge_lookahead.py`）

净电荷可加，解码第 t 位可估算候选影响：

```
Q_k = Q_fixed + q(aa_k, pH) + Q_expect_others + Q_termini
```

**⚠️ softmax 平移不变性**：bias 不能写 `(Q_k − target)`（常数项被 softmax 抵消）。正确写法让 target 进交叉项：

```
bias_k = strength · (target_charge − Q_current) · q(aa_k, pH)
```

**结构感知过滤器**（`structure_aware_filter.py`）：4 条空间规则（电荷聚集/盐桥/核心渗入/同号聚类）对异常聚集加负 bias。阈值取 CATH 34,653 域统计的 99 分位。

---

## 四、条件注入机制（Level 2，主线）

### 4.1 条件向量（mask-aware，7 维）

```
[pH, has_charge_flag, charge_val, has_pos_limit_flag, pos_limit_val, has_neg_limit_flag, neg_limit_val]
```

`has_X_flag`（0/1）区分"真实条件"与"占位符"，避免 0 值歧义。

### 4.2 ConditionEncoder

```
Linear(7→64) → GELU → Linear(64→128) → GELU → Linear(128→4×128) → reshape [4, 128]
```

74,880 参数。**唯一训练对象**（backbone 冻结）。输入先按训练集 μ/σ 标准化。

### 4.3 Cross-attention 注入（`conditioned_sampler.py`）

```
attn = softmax(h_V · promptᵀ / √d)         # [L, 4] 每节点对 4 个 prompt 的注意力
h_V ← h_V + attn · prompt                  # 加权求和加回
```

等价 soft prompt，无需改解码器。训练（`train_finetune.py`）与推理（`run_guided.py`/`validate_generalization.py`）用同一 `inject_prompt`。

### 4.4 复合损失

```
L = CE + λ_c·charge_deviation + λ_kl·KL_anchor + λ_keep·seq_keep
     λ_c=0.5  λ_kl=0.05  λ_keep=0.5
```

| 项 | 作用 | 解决什么问题 |
|----|------|-------------|
| CE 交叉熵 | 重建 native 序列 | 保结构匹配（能折叠回骨架）|
| charge_deviation | \|期望净电荷 − target\|（温度化 charge_temp=0.5）| 主任务：电荷控制；温度化根治 ~2.9× 过冲 |
| KL_anchor | 条件化‖无条件分布距离 | 防条件注入失控（保 backbone 已优化的可溶/热稳）|
| seq_keep | 对无条件 argmax 锚做 CE | 无需求时不重写序列（管 argmax 翻盘，比 KL 更直接）|

### 4.5 训练数据策略

- **混合目标**：70% 自洽（target=native）+ 30% 扰动（native ± Uniform[1, scale]）——扰动制造"电荷偏移"学习信号
- **占位符**（15%）：均值占位（"电荷不控制"语义），施加载荷损失防负漂移
- **课程学习**（v7）：perturb_scale 随 epoch 2.0→8.0（先温和后极端）
- **逆密度加权**（v4+）：稀有 target（高正电）权重更大，cap=2 防过校正

---

## 五、配体模式差异（v9）

| | 无配体（v7） | 配体模式（v9） |
|---|---|---|
| 特征化 | `use_atom_context=False` | `use_atom_context=True, number_of_ligand_atoms=16` |
| 配体原子 | 无 | `parse_PDB` 输出 Y/Y_t/Y_m |
| backbone | MoMPNN（纯骨架）| LigandMPNN 权重（配体上下文层）|
| 消融 | — | `strip_ligands` 去 HETATM → 同一模型无配体 |

泛化验证发现（`2026-08-19_v9_generalization_validation.md`）：配体上下文对电荷控制无系统性增益；大蛋白（L≥470）上可能有害（注意力稀释）。

---

## 六、验证判据（`index/DESIGN_CRITERIA.md` v2）

| 判据 | 阈值 |
|------|------|
| H1 折叠自洽 | ESMFold 回折 TM-score 中位 ≥0.70；失败率（TM<0.5）≤10% |
| H2 电荷命中 | \|平均实际电荷 − target\| ≤ 2.0 |
| H3 电荷分布 | 聚集违规率 ≤ 基线 +5pp |

软判据：S1* 相似性 0.4–0.7（防坍塌）、S2 可溶/热稳报告、S3 占位符语义、S4 位点固定 100%。
