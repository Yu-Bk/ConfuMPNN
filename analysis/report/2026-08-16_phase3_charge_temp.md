# 训练侧根治电荷过冲 —— 温度化电荷损失

> 日期：2026-08-16
> 背景：Phase 3 发现条件注入的 target→实际电荷有 ~2.57× 线性过冲（`2026-08-16_phase3_pH_response.md`）；推理侧校准（`0be534b`）是补丁。本报告验证**训练侧根治**。

---

## 一、机制回顾

训练时 `charge_deviation_loss` 优化的是 **softmax 期望电荷** `E[Q] = softmax(logits)·Q`（等价 τ=1.0）。但推理采样用 τ=0.3 的**锐化分布**——模型被 CE 训练得很自信，采样序列的电荷比 E[Q] 更极端 → 过冲。

## 二、修法

**温度化电荷损失**：训练算电荷时用 `softmax(logits/τ)`，让训练优化的分布 = 推理采样分布。
- `net_charge_from_logits` / `charge_deviation_loss` 加 `temperature` 参数
- `train_finetune.py --charge_temp 0.5`（默认 0.5；τ<1 时梯度集中在最可能的残基，正对采样行为）
- 训练 30 epoch × 999 域 = 14.8 min（`output/finetune_t05/`）

## 三、效果

### 电荷增益收敛（1BC8 未校准实测）

| target | 旧编码器（τ=1.0 训练） | **新编码器（τ=0.5 训练）** | 误差 |
|--------|----------------------|--------------------------|------|
| 8.9 | +25.6 | **+9.75** | 0.85 |
| +5 | +13.0 | **+4.74** | 0.26 |
| 0 | -1.1 | **-0.14** | 0.14 |
| -5 | -15.6 | **-4.89** | 0.11 |

**增益 2.57 → 1.04，offset 0.16 → 0.05** —— 无需任何推理侧校准，电荷直接精准命中。

### pH 感知保留（Go/No-Go 复验 4/4 PDB）
- target 响应单调 ✅（1BC8: 0→+1.5, 9→+10.1；1UBQ: -5→-5.5, 5→+4.9）
- 跨 pH identity 0.78–0.92 < 100% ✅（模型仍感知 pH）

## 四、结论与决策

- **温度化是根治**：把训练目标对齐推理采样分布，过冲从根上消除，且不需要推理侧校准
- **默认关闭推理侧校准**（`charge_calibration.enabled=false`）：新编码器 gain≈1，全局校准（被 2LZM 拉高到 1.289）会过校正 1BC8（target 8.9 → +6.25，反而不如不校准）
- ⚠️ **残余 per-PDB 增益差异**（1.04~1.7）：2LZM 仍略过冲（target 13 → +21.7，旧编码器 +37）。若需精确到每蛋白，可 per-PDB 拟合或继续调 `--charge_temp`

## 五、推荐使用

```bash
# 条件注入生成（默认用 finetune_t05 编码器 + 校准关闭）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 5 \
  --cond_encoder output/finetune_t05/condition_encoder_last.pt
```

## 六、文件

- 改动：`code/src/differentiable_charge.py`、`code/src/losses.py`、`code/train_finetune.py`、`code/configs/condition_defaults.yaml`
- 训练：`output/finetune_t05/`（新编码器）；`output/finetune/`（旧，τ=1.0）
- 验证：`output/phase3_t05/{pdb}/phase3_pH_response.json`
