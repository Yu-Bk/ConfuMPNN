# 2026-09-05 配体族受控减预算消融报告（v14 配方 / LigandMPNN atom25）

> 母计划 `index/PROJECT_LOCAL_V14_FINAL_EXPERIMENTS.md` §1；落地计划 `ablation/plan.md`。
> 目的=模块相对贡献/排序，非绝对 SOTA；绝对数值不与正式版比。
> 全部产物在 `ablation/`，未 git push（主会话统一归档）。

## 1. 设定
- 数据：`labels_v14_final.npz`(5371 域) 分层抽 25%（保 RNA/DNA 7.8%）→
  `ablation/data/labels_ablate_lig.npz`（1364 域；L 281.0→282.1、Q −1.19→−1.16、RNA/DNA 420(7.82%)→114(8.36%)）。
  dompdb=`data/ligand_train/all_pdb`。
- 训练：LigandMPNN `ligandmpnn_v_32_010_25.pt` backbone 冻结、只训 ConditionEncoder；epochs=16（正式 50 的 ~1/3）；
  seed=42；超参=v14 全套（λ_c0.5 λ_kl0.05 λ_keep0.5、charge_temp0.5、perturb0.3/placeholder0.15、
  decouple_absolute lo−35 hi20、v12: frac0.5/gravy0.4/λ_v120.2/λ_target0.2/sasa0.25、ph_aware boost1.5、
  A1 pocket_mode global floor0.8/ceil1.3/λ_pocket0.3、num_ligand_atoms=25）。
- run 矩阵：FULL / −v12组成(去 `--v12_supervision`) / −λ_target / −A1(`--lambda_pocket 0`，保留 global 分区) /
  −ph_filter / −seq_keep(`--lambda_keep 0`)；每 run 仅一处差异。
- 6 runs 全部正常，无 NaN。耗时 32-49min/run。

## 2. 前向 val-loss（最终 epoch=16，统一 v14 FULL 口径回放，配体 val=805/805 域）
> 同口径回放 → total/ce/cd 横向可比。关键项：`v12_comp`(表面组成双计数)、`v12_ct`(表面电荷锚)、`pocket`(A1 双向计数)。

| run | ce | cd | total | v12_comp | v12_ct | pocket | Δcd | Δtotal |
|---|---|---|---|---|---|---|---|---|
| **FULL** | 1.628 | 3.157 | 4.793 | 0.036 | 4.330 | 0.235 | — | — |
| −v12组成 | 1.663 | 2.753 | **5.506** | **3.219** | 4.682 | **0.622** | −13% | **+15%** |
| −λ_target | 1.577 | 3.177 | 4.856 | 0.030 | **5.146** | 0.212 | +0.6% | +1.3% |
| −A1 | 1.636 | 3.023 | 4.722 | 0.062 | 4.162 | 0.262 | −4% | −1.5% |
| −ph_filter | 1.625 | 3.132 | 4.737 | 0.034 | 4.152 | 0.233 | −0.8% | −1.2% |
| −seq_keep | **1.722** | 2.774 | 4.686 | 0.030 | 3.842 | 0.233 | −12% | −2.2% |

- −v12组成 的 **v12_comp 0.036→3.219（~88×）**、pocket 0.235→0.622 → 去掉组成监督后生成序列
  表面带电组成系统性偏离 native、删减进 pocket 区（这正是 v12 模块要治的删减捷径）。
- −λ_target 的 v12_ct 4.330→5.146（+19%）→ 表面电荷锚失效。
- −seq_keep 的 ce +5.8%、kl +43.6%（无条件锚丢失 → 条件分布漂移大）；cd −12%（“更易命中”但以删减/漂移为代价）。
- −A1 / −ph_filter 对 total 影响很小（≤±1.5%）。

## 3. 生成式抽查（5O60_E RNA 结合蛋白；raw 无校准；n30/臂；native/n2/p2）
> 5O60_E native q≈+10.9（target +11 为正电 RNA 界面蛋白）。raw 下所有 run 都大幅过冲
> （mean_q≈+20~24 vs target +9~+13，dev≈6-14）——这是配体 raw 响应增益问题（正式版需校准），
> 绝对 H2 全 False；模块影响看**相对** retention 与 mean 差异。

| run | native mean_q | native dev | H2 | native 保留率(D/E/K/R) |
|---|---|---|---|---|
| **FULL** | +22.97 | 11.97 | 0/3 | 0.787 |
| −v12组成 | +20.05 | 9.05 | 0/3 | **0.503** |
| −λ_target | +22.88 | 11.88 | 0/3 | 0.800 |
| −A1 | +23.35 | 12.35 | 0/3 | 0.772 |
| −ph_filter | +23.41 | 12.41 | 0/3 | 0.792 |
| −seq_keep | +24.23 | 13.23 | 0/3 | 0.794 |

- **−v12组成 保留率 0.787→0.503**：去掉组成监督后生成序列删掉一半带电残基（RNA 正电界面上删减捷径
  复现——与 v12 配体历史问题一致，证实 v12 组成监督是当前防删减的核心）。

## 4. 模块相对贡献排序（配体族结论）
1. **v12 组成监督（−v12组成）贡献最大**：val total +15%、v12_comp +88×、pocket +1.6×；
   probe 5O60_E retention 0.79→0.50（删减一半带电残基）。
   → 是防“表面带电组成偏离 / 删减捷径”的主力模块。
2. **λ_target（−λ_target）**：val v12_ct +19%、cd +0.6%；probe retention 0.80（略偏高=略过添加）。
   → 负责锚定表面净电荷，去掉则 v12_ct 恶化（响应过冲倾向略增）。
3. **seq_keep（−seq_keep）**：val ce +5.8%、kl +44%（条件序列漂移大），cd −12%（因删减捷径“更易命中”）；
   5O60_E retention 0.794（与 FULL 相当，不同于蛋白族——RNA 蛋白上 keep 影响主要在序列稳健/漂移）。
   → 作用是保序列稳健/防无条件漂移；单纯看 cd 会误以为它有害（去掉 cd 降），需结合 ce/kl/保留率。
4. **A1 pocket（−A1）**：val pocket +11%、cd −4%、total −1.5%；probe retention 0.772（略低于 FULL 0.787）。
   → 保 pocket/全局带电计数有方向正确的温和作用，量级小于 v12/λ_target。
5. **ph_filter（−ph_filter）**：各字段几乎不变（self 口径 struct=0），probe 与 FULL 相当。
   → 最小；该模块定位是结构辅助惩罚，不在电荷/组成的本抽查主判面。

排序：**v12组成 > λ_target > seq_keep(稳健) > A1(pocket) > ph_filter(结构辅助)**；
与蛋白族一致的部分 = v12组成+λ_target 是电荷轨道核心；配体族额外确认 A1 有温和正贡献。

## 5. under-train 压平检查
- 16 epochs（正式 50 的 1/3），配体 FULL val cd=3.16 vs 正式版收敛后更低 → 未完全收敛。
- val total 相对差除 −v12组成(+15%)外均 ≤±2.2% → **部分压平**，对 λ_target/seq_keep/A1/ph 的估计可能偏保守。
- 但核心效应未被压平：−v12组成 v12_comp +88×、probe retention 0.50；−λ_target v12_ct +19% → **排序可信**。
- probe raw 全过冲（响应增益）是配体校准问题，不影响模块相对比较；正式应用按各版本校准表。

## 6. 备注 / caveat
- v14 FULL 口径回放中 λ_keep=0.5：−seq_keep run 训时 λ_keep=0，val 时 keep 字段仍按 0.5 计入 total →
  “keep loss 高”反映其条件序列偏离 uncond 锚（对消融是有效判别，非误差）。
- 5O60_E 为强正电 RNA 结合蛋白，raw 大过冲放大了绝对 dev；建议配体族生成结论以 retention 相对差为主。
- val n_dom=805/805 全部可解析。

## 产物
- run ckpt/log/val：`ablation/runs/lig/run_{FULL,nov12comp,notarget,noA1,noph,nokeep}/`
- probe：`ablation/runs/lig/probe_5O60E_run_*.json`
- 汇总：`ablation/report/ablation_summary_lig.json`；图数据 `ablation/figure/ablation_lig_figdata.json`
- driver：`ablation/runs/run_lig_ablation.sh` + `run_lig_driver.log`
