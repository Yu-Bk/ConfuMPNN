# 会话记录：charge_lookahead target 失效 bug 修复

日期：2026-08-16
涉及模块：`code/src/charge_lookahead.py`、`code/tests/test_all.py`、`code/run_guided.py`
状态：✅ 已修复并验证

## 背景

Phase 1 引导采样对照实验中，`--target_charge 8 / 0 / -8` 生成的序列**完全相同**
（净电荷都在 +8 附近，逐字符一致）。target 参数对生成毫无影响。

## 定位过程（排除法）

| 步骤 | 实验 | 结论 |
|------|------|------|
| 1 | 直接调用 `bias_at`，比较 target=8 / -8 的 bias[K] | bias 值确实不同（+4.18 / -3.82），非模块返回值问题 |
| 2 | spy 回调确认在 `guided_sample` 内部被调用 | 回调链路正常 |
| 3 | 极端测试 bias[K]=+100 → 93 个位置全 K | bias 应用链路正常（能显著影响采样） |
| 4 | 测 logits 尺度（absmax≈3.6, std≈1.3） | ±4 的 bias 应显著影响 softmax，排除幅度不足 |
| 5 | bias_scan：bias[K]=0/2/5/100 → K 数 16/34/74/93 | bias 幅度确实影响采样 |

五步排除后只剩数学形式本身的问题。

## 根因

原公式 `bias_k = -strength·(Q_k − target)`：

```
bias_k = -strength·(fixed + q_k + expect_others + termini − target)
       = -strength·q_k  −  strength·(fixed + expect_others + termini − target)
                              └────────────── 不依赖候选 k 的常数 ──────────────┘
```

第二项括号内全是**不依赖候选 k** 的常数。由 softmax 常数平移不变性
`softmax(x + c) = softmax(x)`，该项在采样时被完全抵消。残留的 `-strength·q_k`
与 target **无关** → target 对分布零影响。

> 直觉类比：往所有候选的分数上都加同一个数（比如都 +5），相对排序不变，
> softmax 概率也不变。target 恰好只存在于"所有候选共同的部分"里。

## 修复

改用**当前净电荷驱动**形式：

```
Q_current = fixed + expect_others + termini          # 不含候选位
bias_k    = strength · (target − Q_current) · q_k     # q_k = 候选侧链电荷
```

- `(target − Q_current)`：当前还欠多少目标电荷（标量，随解码推进变化）
- 乘上 `q_k`（依赖候选 k）→ target 通过 `target·q_k` 交叉项进入分布
- 渐近收敛：越接近 target，驱动越小，不会过冲

语义验证：全 K 序列（当前偏正）：
- target=+8 → (8−Q)<0? 否，还欠正电荷 → K 被促进（bias[K]>0）✓
- target=0  → 正电过多 → K 被抑制 ✓
- target=−8 → 更抑制 K ✓

## 验证结果

单元测试：36/36 通过（新增 3 项 target 敏感/termini/回调测试）。

真实模型（1BC8, pH 7.4, LigandMPNN, strength=0.5, 5 条）：

| target | 平均净电荷 | pI 范围 | 结论 |
|--------|-----------|---------|------|
| +8.0 | **+8.06 ± 0.92** | 10.1–10.4 | ✓ 精准命中 |
| 0.0 | **+0.23 ± 0.80** | 5.7–9.1 | ✓ 拉向中性 |
| −8.0 | **−7.96 ± 0.95** | 4.2–4.5 | ✓ 精准命中 |

额外验证：
- 电荷引导 + 结构感知过滤器叠加：target=−2 → −1.94 ✓
- 弱强度 strength=0.1，target=−8 → −1.98（介于 native +8 与目标之间）✓ strength 连续可调

## 教训

1. **softmax 常数平移不变性是 logit bias 设计的第一法则**：任何 bias 若包含
   不依赖候选索引的常数项，都会被抵消。写 bias 公式时应直接写"只含依赖候选
   的项"，或做平移不变性检查。
2. 类似的坑也存在于未来的条件嵌入/损失函数设计里——凡是在"所有候选共享"
   的分量上放条件的，都要警惕。
3. 对照实验发现"结果相同"时，先怀疑"参数进了常数项"而非"代码没被调用"。

## 相关文件

- `code/src/charge_lookahead.py` — 修复点
- `code/tests/test_all.py` — 新增 `test_lookahead_target_sensitivity` 等 3 项
- `code/output/guided_1BC8_pH7.4/target_{8,0,-8}/` — 对照结果
- 排查脚本在会话临时目录（不保留）
