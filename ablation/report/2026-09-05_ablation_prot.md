# 2026-09-05 蛋白族受控减预算消融报告（v12.2 配方 / MoMPNN）

> 母计划 `index/PROJECT_LOCAL_V14_FINAL_EXPERIMENTS.md` §1；落地计划 `ablation/plan.md`。
> 目的=**模块相对贡献/排序**，非绝对 SOTA；绝对数值不与正式版比。
> 全部产物在 `ablation/`（用户 2026-09-05 指定），未 git。

## 1. 设定
- 数据：`labels_v12_3_train.npz`(6580 域) 分层抽 25% → `ablation/data/labels_ablate_prot.npz`（1659 域；L mean 181.8→184.9、Q mean 0.06→0.15，分布近全集）。dompdb=`data/cath/S40/dompdb`。
- 训练：MoMPNN backbone 冻结、只训 ConditionEncoder；epochs=10（正式 30 的 1/3）；seed=42；
  超参 = v12.2 全套（λ_c0.5 λ_kl0.05 λ_keep0.5、charge_temp0.5、perturb0.3/placeholder0.15、
  decouple_perturb range12、v12: frac_floor0.5/gravy0.4/λ_v120.2/λ_target0.2/sasa0.25、
  ph_aware boost1.5、pocket keep λ_pocket=0）。
- run 矩阵：FULL / −v12组成(去 `--v12_supervision`) / −λ_target(去 `--lambda_target`) /
  −ph_filter(去 `--ph_aware_filter`) / −seq_keep(`--lambda_keep 0`)；每 run 仅一处差异。
- 5 runs 全部正常完成，无 NaN。各 run 耗时 21-30min（10 epochs）。

## 2. 前向 val-loss（最终 epoch=10，统一 v12.2 FULL 口径回放，蛋白 val=1104/1199 域可解析）
> 同口径回放 → total/ce/cd 横向可比。`v12_ct` = surface_charge_target 损失（FULL 口径施加）。

| run | ce | cd | total | v12_ct | Δcd vs FULL | Δtotal vs FULL |
|---|---|---|---|---|---|---|
| **FULL** | 1.865 | 2.683 | 4.410 | 2.799 | — | — |
| −v12组成 | 1.867 | 2.727 | 4.553 | 3.271 | **+1.6%** | +3.3% |
| −λ_target | 1.847 | 2.829 | 4.490 | 3.102 | **+5.4%** | +1.8% |
| −ph_filter | 1.866 | 2.688 | 4.376 | 2.620 | +0.2% | −0.8% |
| −seq_keep | 1.927 | 2.466 | 4.438 | 2.681 | −8.1% | +0.6% |

- val-loss 相对差整体很小（|Δtotal|≤3.3%）→ 见 §5 under-train 检查。

## 3. 生成式抽查（raw、无校准；n30/臂；native/n2/p2；H2 = |mean_q−target|≤2）
> raw 响应不加校准，保留模块间差异；1A65 是长负电蛋白（记忆已知校准敏感），绝对 dev 大，看相对。

| run | 1BC8 H2 | 1BC8 dev(nat/n2/p2) | 1A65 H2 | 1A65 dev(nat/n2/p2) | native 保留率(1BC8/1A65) |
|---|---|---|---|---|---|
| **FULL** | 3/3 | 0.46/0.12/0.99 | 0/3 | 5.49/4.77/6.40 | 1.02 / 0.96 |
| −v12组成 | 1/3 | 2.36/1.40/2.79 | 0/3 | **17.6/18.3/16.5** | 1.03 / 1.03 |
| −λ_target | 1/3 | 2.67/1.78/3.10 | 0/3 | 7.21/7.03/8.02 | 1.18 / 1.13 |
| −ph_filter | 3/3 | 1.12/0.39/1.49 | 0/3 | 4.48/3.70/4.94 | 1.05 / 0.91 |
| −seq_keep | 3/3 | 0.54/0.20/0.26 | 0/3 | 8.93/6.25/10.36 | 0.84 / 0.94 |

1A65 native 参考 q=−26.85（带电 79）；1BC8 native q=+8.90（带电 21）。

## 4. 模块相对贡献排序（蛋白族结论）
1. **v12 组成监督（−v12组成）贡献最大**：去掉后 val `v12_ct` +16.9%、total +3.3%；生成端 1A65 崩溃
   （native dev 5.5→17.6、mean_q −27→−44.6 严重负电过冲），1BC8 H2 3/3→1/3。
   → 是防"电荷漂移/大幅过冲"的主力（代码装配中 v12 块还承载 λ_target，见 §6 备注）。
2. **λ_target（surface charge target，−λ_target）次之**：val cd +5.4%（各 off 中 cd 影响最明显）、
   v12_ct +10.8%；probe 1BC8 native/p2 过冲丢 H2、带电保留率升至 1.18/1.13（过度添加带电残基）。
   → 负责"锚定表面净电荷"，去掉则正电臂过冲 + 过多引入带电残基。
3. **seq_keep（−seq_keep）**：H2 数值反而改善（val cd −8.1%、1BC8 dev 更小），但这是"删减捷径"代价
   ——native 臂带电保留率 1BC8 1.02→**0.84**、生成带电残基数 21→17.7，序列系统性删带电残基；
   kl +62%（条件分布偏离无条件锚）。→ 模块作用是保序列稳健/组成保守（真实设计价值），非直接控电荷命中。
4. **ph_filter（−ph_filter）贡献最小**：val-loss 各字段几乎不变（cd +0.2%、total −0.8%、v12_ct −6%），
   probe 1BC8 仍 3/3；1A65 dev 略降但保留率 0.91（删减略增）。→ 该模块是低权重结构辅助惩罚
   （self 臂 0.05），影响方向偏 H3 聚集/结构真实，不在本抽查的 H2/保留率口径内。

总体：**组成(v12)>λ_target>seq_keep(组成稳健)>ph_filter(结构辅助)**；
v12 组成与 λ_target 控制电荷轨道/幅度；seq_keep 防删减捷径；ph_filter 为辅助。

## 5. under-train 压平检查
- 证据：10 epochs（正式 1/3）FULL 最终 cd self=2.74 vs 正式 v12.2(30ep) cd self≈2.2 → 未完全收敛；
  val-loss 相对差很小（|Δcd|≤5.4%、|Δtotal|≤3.3%）→ 存在**部分压平**，可能低估模块真实贡献。
- 但生成端差异未被压平：−v12组成 在 1A65 的 dev 5.5→17.6、−λ_target 在 1BC8 的 native H2 丢失 +
  过添加 retention 1.18，均清晰可辨 → **排序结论可信**；充分训练后模块贡献预计放大而非反转。
- val-loss 自身极小差异部分因为 v12_ct/组成项在"native@pH self 臂"口径本身量级小；生成端才是主要判别面。

## 6. 备注 / 口径 caveat
- 代码装配中 `λ_target`(v12_ct) 挂在 `--v12_supervision` 块内 → "−v12组成" run 实际同时关闭 v12_comp/gravy
  与 v12_ct（三 loss 同模块）；排序解读时把 v12 组成+λ_target 视为同一模块族（§4 第 1 名），λ_target 单开列第 2。
- val n_dom=1104/1199（95 域 PDB 缺失或 SASA 失败跳过，各 run 相同子集→可比）。
- 生成抽查 raw（无校准）；正式版应用需按各版本校准表，但消融只看相对。

## 产物
- run ckpt/log/val：`ablation/runs/prot/run_{FULL,nov12comp,notarget,noph,nokeep}/`
- probe：`ablation/runs/prot/probe_{1BC8,1A65}_run_*.json`
- 汇总 JSON：`ablation/report/ablation_summary_prot.json`；图数据 `ablation/figure/ablation_prot_figdata.json`
- 训练 driver 日志：`ablation/runs/run_prot_ablation.sh` + `run_prot_driver.log`；eval：`ablation/runs/eval_val_loss.sh`
