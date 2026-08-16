# Phase 2 条件微调 —— 训练启动报告

> 日期：2026-08-16
> 状态：脚本就绪 + 冒烟通过 + 后台启动
> 对应计划：第一版 `PROJECT_PLAN.md` Phase 2 / 第二版 `PROJECT_EXTEND.md` E2（衔接）

---

## 一、微调目标（三层）

| 层级 | 目标 | 可衡量指标 |
|------|------|-----------|
| **架构** | 冻结 MoMPNN backbone，只训练 `ConditionEncoder`（Soft Prompt MLP，~75K 参数） | backbone 参数不更新；可训练参数 74,880 |
| **直接** | 条件编码器学会「(pH, target_charge) → 氨基酸分布」映射 | charge 损失下降（冒烟 4.71→3.67 ✅）；验证集电荷偏差 ≤ 阈值 |
| **最终** | 推理时给训练未见过的 (pH, target) 也能外推（pH 感知完整证据） | 微调后重跑 E1b：跨蛋白/跨 pH 电荷单调性成立 |

**这是 Phase 1 诚实边界（无引导时模型不感知 pH）的正解**：把 pH 感知从「采样时临时注入」变成「模型学到的先验」。

---

## 二、关键设计决策

### 1. 冻结 backbone（对计划「全量微调」的保守偏离）
- 计划原文：`A100 上微调 LigandMPNN（全量，262 万参数）`——但那是针对**原版** LigandMPNN（无多目标先验）。
- 现状：默认生成器是 **MoMPNN**（E4，ProtAlign 多目标 DPO 权重，可溶/Tm/可设计性已优化）。**全量微调有破坏这些权重的风险**。
- 决策：**冻结整个 backbone，只训 ConditionEncoder**。可溶/Tm/可设计性存在冻结权重里，结构上保留；条件注入只是 h_V 上一个小的加性信号（prompt=0 时输出 = MoMPNN 原样）。
- 升级路径：若冻结+prompt 达不到 pH 响应 → 逐层放开 decoder 最后几层 → 全量。分阶段、可回退。

### 2. soft prompt 注入：cross-attention（对计划「字面前缀」的实现修正）
- 计划写「4 个 token 拼到 decoder 输入前缀」，但 decoder 的 E_idx / order mask 依赖固定 L 个位置，字面前缀需重排 E_idx，易错。
- 实际实现：`h_V += softmax(h_V·prompt^T/√d)·prompt`——每个结构节点按需读取 4 个条件 token，等价 soft prompt，无需改动解码器。

### 3. 损失：CE + λ_c·charge_deviation + **λ_kl·KL 锚定**（防失控）
```
L = CE + λ_c·charge_deviation + λ_kl·KL(条件化 ‖ 无条件)
```
- **CE**（权重最高）：重建 native 序列，结构匹配度锚（计划 6.1 风险表）。
- **charge_deviation**：期望净电荷 vs 目标电荷，可微（`differentiable_charge.py`）。
- **KL 锚定（新增，回应「微调是否可能失控」）**：约束条件注入后的输出分布不偏离 backbone 无条件输出太远，**只允许在电荷约束要求的反向上变化** → 防止微调破坏 MoMPNN 的可溶/Tm/可设计性。无条件 logits 冻结，每域算一次缓存，近零额外成本。

### 4. 目标电荷策略：自洽 + 扰动混合（回应「学不到电荷偏移」）
- 纯自洽（target=native 电荷）的隐患：CE 与电荷损失同时被「重建 native」满足 → 模型学不到电荷偏移，条件向量退化为无效输入。
- 混合目标：每样本 **50% 自洽目标**（锚定结构）+ **50% 扰动目标**（native ± Uniform[1,4]，制造 CE 与电荷的冲突，教模型「target 偏离 native 时如何偏移氨基酸分布」）。
- 冒烟证据：charge 4.71→3.67 下降，说明编码器确实在学习电荷偏移。

---

## 三、数据与参数

| 项 | 值 |
|----|----|
| 数据 | `data/cath/labels.npz`：999 结构域 × 8 pH = 7,992 样本（域主序） |
| 每域批 | 共享结构，仅条件向量不同 → 批内 B=8（一次 encode 复用） |
| 预解析 | parse_PDB（CATH 无后缀 → `.pdb` 符号链接解决 prody 格式误判）+ encode + 无条件 logits，每域一次缓存 |
| 学习率 | Adam 1e-3（仅编码器参数） |
| λ_c / λ_kl | 0.5 / 0.05 |
| perturb_prob / scale | 0.5 / 4.0 |
| 设备 | cuda:1（当时最空闲：141GB 空闲） |

---

## 四、启动与查询

```bash
# 后台启动（端口重置/退出终端不中断）
nohup setsid /home/baokun_yu/miniconda3/envs/confumpnn/bin/python \
  code/train_finetune.py --device cuda:1 --epochs 30 \
  > code/log/train.log 2>&1 &

# 进度查询
bash code/tests/train_status.sh
```

时间估算：预解析 ~10min + 30 epoch × ~1min ≈ **45 min 左右**（冒烟实测 15 step/秒级，999 域/epoch ≈ 67s）。可中断续训（每 epoch 存 checkpoint）。

---

## 五、防失控三道防线（总结）

1. **backbone 冻结**：可溶/Tm/可设计性存在 MoMPNN 权重里，结构上保留；
2. **KL 锚定正则**：机制性约束条件化输出不偏离 backbone 分布太远；
3. **事后 E1b 验证**（最终判据）：微调完成后立即重跑 ESMFold pLDDT + TM-score + Protein-Sol %sol + TemBERTure Tm，前后对比。若 %sol/Tm 掉 → 调大 λ_kl、调小 λ_c，或按计划加 DPO_aux/自适应 margin。

> 注：溶解性/Tm 不属于本阶段损失的直接范围——它们归第二版多目标微调（E2）。本阶段的职责是**在不破坏它们的前提下加上 pH 感知**。
