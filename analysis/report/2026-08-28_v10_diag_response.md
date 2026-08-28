# v10-MoMPNN 响应曲线诊断报告（2026-08-28）

> **背景**：v10 泛化验证（`2026-08-28_v10_validation.md`）发现电荷控制系统退化（MoMPNN native 命中 0.6→0.2，n8 0.4→0.0）。
> `index/v10_repair/` 提出"**目标域外推**"假说：训练 target 值域覆盖不足 → 编码器在验证深负靶区（−19~−35）外推，增益≠1。
> 本诊断用**现有 v10 checkpoint（不重训）**扫描 target 网格，拟合"target → 生成电荷"响应曲线，**坐实或证伪该假说**。
> **状态**：✅ 诊断完成（17/17 蛋白，无 NaN）。**结论：外推假说未坐实——训练覆盖区内响应增益已 >1，B/C 与 decouple 的叠加才是主因。**

## 1. 方法

| 项 | 设置 |
|----|------|
| 脚本 | `index/v10_repair/v10_diag_response_curve.py`（MoMPNN 侧，v10 checkpoint）|
| 编码器 | `output/finetune_v10_mompnn/finetune_epoch030.pt` |
| 蛋白 | 训练域 7（CATH S40 抽样，含 1 深负域 native<−15）+ 验证集 10（manifest）|
| target 网格 | −34, −30, −25, −20, −15, −10, −5, 0, 5, 10, 18 + 各蛋白 native 点 |
| n / target | 20；temperature 0.3；pH 7.4；seed_base 3000；device cuda:2 |
| 结果 | `output/v10_diag_response.json` |
| 分段分析 | `index/v10_repair/analyze_diag.py`：区内（target∈[native−12, native+12]，训练覆盖域）vs 区外（负向/正向）|

**判定逻辑（对齐 `v10_repair/README.md` 判据表）**：
- 区内 slope≈1 且区外≈2 → 外推假说坐实 → v11 改 target 覆盖即可
- 区内 slope≈1.5~2 → 模型响应整体坏（B/C 叠加）→ 需拆组件重训

## 2. 结果

### 2.1 全区斜率（脚本直接输出）

| group | n | slope 均值 | intercept 均值 |
|-------|---|-----------|---------------|
| trainish | 7 | **1.50 ± 0.33** | −7.3 |
| valid | 10 | **1.62 ± 0.45** | −5.9 |

> 全区 slope 已 >1，且**训练域（1.50）与验证域（1.62）接近**——这是第一个异常信号：若为纯外推，训练域应≈1。

### 2.2 分段斜率（analyze_diag.py）

| 蛋白 | group | L | native | slope_all | **slope_in(区内)** | slope_negout | slope_posout |
|------|-------|---|--------|-----------|-------------------|--------------|--------------|
| 7pujA01 | trainish | 283 | −12.0 | 1.50 | **1.87** | 1.16 | 1.10 |
| 1d5yB03 | trainish | 168 | −11.2 | 1.59 | **1.96** | 0.95 | 0.94 |
| 6jixA02 | trainish | 270 | −10.6 | 1.21 | **1.45** | 1.26 | 0.73 |
| 1l3eB00 | trainish | 101 | +8.3 | 1.68 | 1.15 | **1.89** | — |
| 2d3yA00 | trainish | 219 | +8.8 | 1.06 | **0.69** | 1.01 | — |
| 8etcb01 | trainish | 146 | +14.1 | 1.35 | 0.87 | 1.29 | — |
| 2uv8A05 | trainish | 617 | −16.0 | 2.13 | **2.38** | **2.72** | 0.76 |
| 1C6O | valid | 177 | −14.3 | 1.28 | **0.98** | 1.90 | 1.64 |
| 1AZM | valid | 258 | −1.7 | 1.30 | 1.06 | 1.69 | — |
| 1AS2 | valid | 312 | −2.7 | 1.94 | **2.11** | 2.00 | 1.17 |
| 1AXW | valid | 528 | −18.3 | 1.83 | **2.11** | — | 1.20 |
| 2FEO | valid | 221 | −6.9 | 1.37 | 1.26 | 1.76 | 0.72 |
| 5CQH | valid | 183 | −5.5 | 1.22 | 1.38 | 1.18 | 0.86 |
| 1CGE | valid | 162 | −11.7 | 1.02 | **1.12** | 0.93 | 0.88 |
| 1AG0 | valid | 256 | −8.2 | 2.16 | **2.09** | **2.84** | 1.62 |
| 1A65 | valid | 504 | −26.9 | 1.64 | **2.16** | — | 1.07 |
| 1BJ4 | valid | 470 | +0.4 | 2.49 | **2.46** | **2.77** | — |

**区内 slope 汇总**：trainish **1.48 ± 0.57**，valid **1.67 ± 0.53**。

### 2.3 逐 target 完整曲线（代表性蛋白，mean_charge）

```
1BJ4 (valid, L=470, native=+0.4)
  target: -34   -30   -25   -20   -15   -10    -5     0     5    10    18    0
  mean  : -96.6 -85.2 -70.4 -57.5 -44.0 -31.7 -16.7  -1.1  10.8  15.9  23.8 -1.1
  dev   :  62.6  55.2  45.4  37.5  29.1  21.7  11.7   1.1   5.8   5.9   5.8  1.1

7pujA01 (trainish, L=283, native=-12.0)
  target: -34   -30   -25   -20   -15   -10    -5     0     5    10    18   -12
  mean  : -57.4 -52.8 -47.0 -39.5 -30.2 -19.6 -11.8  -4.1   2.2   6.9  16.0 -24.1
  dev   :  23.4  22.8  22.0  19.5  15.2   9.6   6.8   4.1   2.8   3.1   2.0  12.1

2d3yA00 (trainish, L=219, native=+8.9)
  target: -34   -30   -25   -20   -15   -10    -5     0     5    10    18     9
  mean  : -36.7 -32.5 -28.3 -23.9 -20.0 -14.2  -5.8   1.7   6.1   9.3  14.2   8.7
  dev   :   2.7   2.5   3.3   3.9   5.0   4.2   0.8   1.7   1.1   0.7   3.8   0.3

1C6O (valid, L=177, native=-14.3)
  target: -34   -30   -25   -20   -15   -10    -5     0     5    10    18   -14
  mean  : -46.0 -38.4 -29.4 -24.0 -19.2 -14.7  -9.5  -3.1   3.9  12.2  26.3 -18.0
  dev   :  12.0   8.4   4.4   4.0   4.2   4.7   4.5   3.1   1.1   2.2   8.3   4.0

1A65 (valid, L=504, native=-26.9)
  target: -34   -30   -25   -20   -15   -10    -5     0     5    10    18   -27
  mean  : -67.5 -60.7 -48.2 -37.6 -27.2 -16.7 -10.6  -2.1   4.5   6.8  12.8 -52.6
  dev   :  33.5  30.7  23.2  17.6  12.2   6.7   5.6   2.1   0.5   3.2   5.2  25.6
```

## 3. 关键发现

1. **训练覆盖区内 slope 已 >1**（trainish 1.48、valid 1.67）——**外推假说未坐实**。
   按 README 判据表，这更接近第二行"训练域也 slope≈2 → 模型响应整体坏（B/C 叠加）"，虽然绝对值是 1.5 而非 2。

2. **正负不对称**：绝大多数蛋白 target≥0 时响应准确（dev 0.3~5.8）；target<0 时系统性过冲（生成比 target 更负）。唯一的正侧异常是 1BJ4（+5→+10.8、+18→+23.8，正过冲 ~6）。**模型对"更负"的要求响应过猛、对"更正"的要求响应平缓**。

3. **深负靶区放大最狠**：target≤−20 时 dev 普遍 20~60（1BJ4 −34→−96.6，2uv8A05 −34→−95.4，1AG0 −34→−87.4）。slope_negout 最高 2.84（1AG0）。

4. **个别蛋白近乎完美**：**2d3yA00**（trainish，native+8.9）全 target dev<5、区内 slope 0.69；**1C6O**（valid）区内 slope 0.98。证明**模型本身没有坏**——响应畸变是"目标符号相关 + 蛋白相关"，不是全局失效。

5. **native 点（自洽样本）也过冲**：7pujA01 target=native=−12 → 生成 −24.1（dev 12.1）；1A65 native=−27 → −52.6（dev 25.6）。**自洽样本本应在训练分布内，却 dev 12~26**——这是"训练收敛（charge 2.05）却在泛化集过冲"的核心悖论，指向训练-推理分布差异而非欠拟合。

## 4. 根因判定（更新 README 假说）

| README 判据 | 诊断实测 | 判定 |
|---|---|---|
| 训练域 slope≈1，验证域≈2 → 外推坐实 | 训练域 1.50，验证域 1.62（**区内也 1.48/1.67**）| ❌ **未坐实** |
| 训练域也 slope≈2 → 模型响应整体坏 | 训练域区内 1.48（部分蛋白 2.1~2.4）| ⚠️ **方向正确，但幅度未到 2** |

**修正结论**：
- **不是纯外推**。训练覆盖区（native±12）内响应增益已 >1，外推只负责把斜率再放大（slope_negout 普遍 >1.6）。
- **负向过冲是主问题**，正向基本正常 → 不是"decouple 把响应整体拉平"（那会正负都坏），更像 **B（L_add 表面添加监督）在负 target 下与模型既有"删减捷径"叠加"双算"**：模型本就会删带电残基逼近 target，L_add 又要求"表面加 D/E"，两边同向加负 → 超线性过冲。
- **decouple（A）的角色待定**：它把 30% 样本 target 与 native 解耦，可能使模型对"保持 native"先验变弱（native 点也过冲），但正向仍准说明响应映射未全面崩塌。
- **长蛋白/深负 native 最差**（1A65/1AXW/2uv8A05）与泛化验证结论一致（P4 已知薄弱区）。

## 5. v11 修复方向建议（决策点前移）

1. **仅改 target 覆盖（A-fix 绝对 target）不够**——区内已坏，必须同时降 B。
   `train_finetune_v11_patch.md` 的三件套方向对，但**预期管理要改**：这不是"覆盖不足→改采样"，而是"**负向响应增益失控→B/C 校准**"。
2. **建议补一个消融顺序**（在 v11 前先做，成本 = 冒烟级）：
   - ① 只开 A-fix（绝对 target）重训 30ep，看负向 slope 是否回 1 → 区分 decouple 本身 vs B/C；
   - ② 再加 B/C-fix → 若还 >1.3，则 B/C 是主因，`lambda_add` 需更低（0.05~0.1）或去掉 B；
   - ③ 若 ① 就回 1 → decouple 才是元凶，可考虑"绝对 target + 保留 native 锚点样本"的折中。
3. **论文叙事**：若 v11 仍无法修复负向超线性，v10 作为**可复现的失败/机制案例**（条件逆折叠对"更负"要求超线性过冲——增益≠1 标定失调 + B/C 叠加），主方法回退 v7/v9。

## 6. 附带修正：验证报告 §5 的归因错误

- `2026-08-28_v10_validation.md` §5 曾写"v10-MoMPNN = v7 训练配置 + decouple，**只此一个训练变化即可致退化**"。
- **实测**：`run_v10_pipeline.sh:80-91` 显示 v10-MoMPNN 训练命令是 **A+B+C 三组件全开**（`--decouple_perturb --decouple_range 12` + `--add_supervision --lambda_add 0.3 --sasa_threshold 0.25` + `--ph_aware_filter --structure_boost 1.5`）。
- 因此"只改 decouple"的归因**不成立**，需按本报告的"B/C 叠加 + 覆盖不足"修正。`v10_repair/README.md` 对此质疑正确，日志已定论（`log/v10_train_mompnn.log` + `run_v10_pipeline.sh`，均已 git 跟踪）。

## 7. 分析过程记录（推理链）

**Step 0 出发问题**：v10 泛化验证显示电荷控制退化，`v10_repair` 提出"target 值域覆盖不足→外推"假说，判据为"训练域 slope≈1、验证域≈2 则坐实"。

**Step 1 判断脚本可用性（冒烟）**：先用 1C6O × targets [−20,−10,0,10] × n=3 冒烟 → slope=1.06、r2=0.95（在训练覆盖区内正常）。确认脚本能加载 v10 checkpoint、采样流程无误。

**Step 2 修路径 bug**：脚本原假设位于 `code/tests/`，`parent.parent` 定位不到项目根 → 改为 `_PROJECT_DIR` 向上搜索，适配 `index/v10_repair/` 位置。

**Step 3 正式诊断**：17 蛋白 × 11 targets(+native) × n=20 = ~4200 次采样，38 分钟跑完。

**Step 4 第一信号（中途）**：训练域蛋白 7pujA01/1d5yB03 全区 slope 已 1.5~1.6 且截距 −7~−7.5 → 与"训练域≈1"矛盾，暂缓下结论。

**Step 5 全区汇总**：trainish 1.50±0.33、valid 1.62±0.45 → 两者接近，纯外推假说存疑。

**Step 6 分段拟合（关键）**：用 analyze_diag.py 按"区内 native±12 vs 区外"分段 → 区内 slope trainish 1.48、valid 1.67，均 >1 → **训练覆盖区内响应已失控**，外推只放大不主导。

**Step 7 逐曲线人工比对（定结论）**：
- 正负不对称：target≥0 时 dev 普遍 <6（正常），target<0 时 dev 10~60（过冲）→ 排除"decouple 整体破坏响应"（那样会正负同坏）；
- 2d3yA00 全区完美、1C6O 区内 slope≈1 → 排除"模型全局坏"；
- native 自洽点过冲（7pujA01 dev 12、1A65 dev 26）→ 训练-推理分布差异，非欠拟合（训练 charge 2.05 收敛正常）。

**Step 8 归因**：负向超线性 + 正向正常 → 指向 B（L_add 表面添加监督）与模型"删减捷径"同向叠加"双算"；decouple 弱化 native 锚点使自洽样本也过冲；长蛋白/深负 native 是放大器。修正验证报告 §5 的"只改 decouple"错误（实测三组件全开）。

**Step 9 修复决策前移**：仅 A-fix 不够，需 B 降权 + 消融顺序区分 A vs B/C。

## 8. 复现

```bash
# 诊断（MoMPNN 侧，v10 checkpoint）
PYTHONPATH=code python index/v10_repair/v10_diag_response_curve.py \
  --cond_encoder output/finetune_v10_mompnn/finetune_epoch030.pt \
  --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
  --backbone auto \
  --manifest data/validation_pdbs/validation_manifest.json \
  --pdb-list /tmp/diag_train.txt \
  --targets=-34,-30,-25,-20,-15,-10,-5,0,5,10,18 \
  --include_native --n 20 --seed_base 3000 --pH 7.4 --device cuda:2 \
  --out output/v10_diag_response.json

# 分段分析
python index/v10_repair/analyze_diag.py output/v10_diag_response.json
```

> 分析过程记录：训练域清单由 `/tmp/prep_diag_pdblist.py` 生成（CATH 抽样→parse→native 电荷→选 6 中小域 + 1 深负域，seed=42）。诊断脚本路径自适配修正见 git diff（`_PROJECT_DIR` 向上搜索）。
