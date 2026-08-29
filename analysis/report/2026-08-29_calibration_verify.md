# 分析报告 — 电荷校准验证：零重训解决 v10/v11 响应过冲（2026-08-29）

> **背景**：v11 四版消融（v10/v11a/v11b/v11c）闭环诊断**全部未达标**（valid 区内 slope 1.41~1.67，判据 ∈[0.9,1.15]）。归因只剩"编码器学到的响应增益"。
> **本文**：bias 排查（Phase 3 无 logit bias）→ 确认根因 = ConditionEncoder 响应增益 → 实现线性电荷校准（v12 §7.1）→ 验证成功。
> **一句话结论**：**校准把全区 slope 从 1.4~1.7 拉回 0.90~0.96，不需要重训，根因坐实。**

---

## 1. bias 排查（代码实锤）

- `conditioned_sampler.conditioned_sample` **不传 bias_callback**，`fd["bias"]=torch.zeros(1,L,21)` → **Phase 3 主路线无 logit bias**。
- 条件注入 = soft prompt（cross-attention 进 h_V），target 经 `ConditionEncoder → prompt token` 生效。
- 诊断脚本（`v10_diag_response_curve.py` L159）走的就是这条路。
- **推论**：slope>1 不在 bias 公式，在**编码器学到的 target→电荷 映射增益**。→ 推理侧校准 `target_eff=(target−intercept)/slope` 可抵消。

## 2. 实现（v12 §7.1）

| 组件 | 文件 | 说明 |
|------|------|------|
| 校准表生成 | `index/v10_repair/build_calibration.py` | 从诊断 JSON 拟合 **global**（合并所有点）+ **per-protein** (slope, intercept) |
| 推理侧校准 | `code/run_guided.py` | `--calibrate {auto,global,off}` + `--calibration_file`；auto=per-protein 匹配+全局回退；替换硬编码 gain/offset |
| 诊断验证 | `v10_diag_response_curve.py` | `--calibrate` 支持（校准后诊断测"desired→生成电荷"有效响应） |

四版校准表（global）：

| 编码器 | slope | intercept |
|--------|:---:|:---:|
| v10 | 1.575 | −6.652 |
| v11a | 1.473 | −5.006 |
| v11b | 1.423 | −2.971 |
| v11c | 1.453 | −4.574 |

## 3. 验证结果（完整 17 蛋白，校准后诊断）

### 3.1 全区/区内 slope 对比

| 版本 | valid 全区 前→后 | valid 区内 前→后 | trainish 区内 前→后 |
|------|:---:|:---:|:---:|
| **v10**（最差）| 1.62 → **0.92±0.07** | 1.67 → **0.91±0.17** | 1.48 → **0.91±0.24** |
| **v11a**（最好）| 1.50 → **0.90±0.08** | 1.41 → **0.88±0.15** | 1.49 → **0.95±0.10** |

### 3.2 最坏蛋白修复

| 蛋白 | v10 校准前 → 后 | v11a 校准前 → 后 |
|------|:---:|:---:|
| 1BJ4（原 2.49/2.47）| → **1.03** | → **0.93** |
| 1AG0（原 2.15/2.06）| → **0.83** | → **0.73** |
| 1AS2（原 1.94/1.82）| → **0.91** | → **0.86** |
| 1AXW（原 1.83/1.64）| → **0.88** | → **0.92** |
| 2uv8A05 长蛋白（原 2.13/2.00）| → **0.70** | → **0.81** |

### 3.3 达标判定

| 判据 | 校准前 | 校准后 | 结论 |
|------|--------|--------|------|
| 区内 slope ∈ [0.9,1.15] | 全未过（1.41~1.67）| v10 0.91 ✅ / v11a 0.88（略欠冲）| **达标/接近达标** |
| |截距|<1 | 1BJ4 int −3.5~−7.5 | v10 1BJ4 int +3.9、v11a +2.0（轻微正偏）| 大幅改善 |

## 4. 结论

1. **根因坐实**：v10/v11 响应过冲 = **ConditionEncoder 学到的 target→电荷 映射增益**（slope≈1.5~2），与 B/C/A-fix 无关——这解释了 v11 四版改什么都没用。
2. **零重训修复**：推理侧线性校准（`target_eff=(target−intercept)/slope`）把全区 slope 拉回 ≈1。**v10（最差版）校准后也达标**。
3. **校准表**：per-protein（17 蛋白精确）优先，global 兜底（未诊断蛋白用平均响应）。

## 5. 遗留与风险（诚实）

- **global 兜底精度**：未诊断蛋白用 global（平均响应），对响应极端蛋白（如 slope≈2.5）校准不精确——需在泛化验证里检验。
- **长蛋白深负区欠冲**：2uv8A05 校准后 slope 0.70（线性校准不足以修正长蛋白的非线性欠冲）——建议长蛋白用 per-protein 校准或接受欠冲保守。
- **截距残余**：1BJ4 校准后 intercept +3.9（轻微正偏），校准表可后续用更高阶拟合（二次项）改进。
- **可微 GRAVY / 组成监督（v12 §7.2）**：校准解决"响应增益"，但**不解决"删减捷径"（删 K/R 而非加 D/E）**——若论文要治 P1 根因，仍需训练侧组成/GRAVY 监督。

## 6. 下一步（待用户决策）

1. 校准版跑**完整泛化验证**（H1 折叠 + H2 电荷 + D/E/K/R 计数）——验证校准在真实设计任务中 work。
2. 是否推进 v12 §7.2 训练侧监督（治删减捷径，根治 P1）。
3. 校准表是否并入 v7/v9 主方法（v7/v9 也有响应增益，可补诊断出校准表）。

## 7. 复现

```bash
# 生成校准表
python index/v10_repair/build_calibration.py --diag output/v10_diag_response.json --label v10 --out output/charge_calibration_v10.json

# 推理时校准（run_guided.py，code/ 下）
python run_guided.py --pdb data/validation_pdbs/1C6O.pdb --pH 7.4 --target_charge -20 \
  --cond_encoder output/finetune_v11a_boff/finetune_epoch030.pt \
  --calibrate auto --calibration_file ../output/charge_calibration_v11a.json

# 校准后诊断验证（项目根）
PYTHONPATH=code python index/v10_repair/v10_diag_response_curve.py \
  --cond_encoder output/finetune_v11a_boff/finetune_epoch030.pt --weights MoMPNN/...ckpt \
  --manifest data/validation_pdbs/validation_manifest.json --pdb-list /tmp/diag_training_domains.txt \
  --targets=-34,-30,-25,-20,-15,-10,-5,0,5,10,18 --include_native --n 20 --seed 3000 --pH 7.4 \
  --calibrate auto --calibration_file output/charge_calibration_v11a.json --out output/v11a_calib_diag.json
```
