# Task2 — 结合残基 fix 后重生成（v14 配体）过程日志（2026-09-04）

## 科学问题
v14（配体 RNA/DNA + A1 global，finetune_epoch050.pt）条件生成存在"删带电残基捷径"
（删减 0.43-0.69×，删减偏口袋）。本任务把**配体结合残基固定为 native** 后再条件生成，
看捷径减弱多少、电荷命中(H2)/电荷聚集(H3)代价。

## 设计
- binding 残基 = Cα 距配体重原子（HETATM/核酸，非水）≤ 8Å（pocket 口径，与 v12/v14 一致），逐蛋白算。
- fix 机制：protein_dict["chain_mask"][binding]=0 → guided_sampler 自回归把固定位强制 native（原生 ProteinMPNN 机制，无需改共享采样核心）。
- 采样：in-10 manifest × 5 臂(native/n2/p2/n8/p8) × n=40，seed_base=2000，per-protein 校准（charge_calibration_v14_ligand_clean.json），表外用 global（本批 10 蛋白全在表内）。
- 指标：mean net charge & dev→target、H2 命中(|dev|≤2.0)、D/E 与 K/R 计数（总量+3 区 pocket/surface/core）、H3（structure_aware_filter 4 规则）。
- 对照：unfix 基线 = output/generalization_ligand_v14_clean/ligand（同蛋白同臂，n=50）。

## 产物
- code/tests/ligand_v9/sample_fixbinding.py（新，Task2 采样）
- code/tests/ligand_v9/analyze_fixbinding.py（新，Task2 汇总对比）
- output/fixbinding_v14/（seqs.fa + validation.json + fixed/*.json）
- analysis/report/2026-09-04_v14_fixbinding.md（表+结论）

## 时间线
- (fill)

## 采样进度（续）
- 全 10 蛋白 × 5 臂 × n=40 完成（output/fixbinding_v14/ligand/<pdb>/pH7.4/arm_*/seqs.fa，
  10×validation.json，fixed/ 10×*_fixed.json）。fixed_mismatch_total=0（固定位 100% native）。
- 采样器：sample_fixbinding.py（新脚本，chain_mask=0 机制，未改共享采样核心 conditioned_sampler/guided_sampler）。
- H3：h3_charge_legality --gen-root fixbinding_v14 → 50/50 PASS（output/h3_ligand_v14_fixbinding.json）。

## 关键结果（fix vs unfix v14_clean 同蛋白同臂）
- 电荷 H2 命中：unfix 45/50(90%) → fix 18/50(36%)。mean |dev| per arm：unfix ~1.0-1.1 → fix ~3.7-5.1。
  达标蛋白只剩 6D2O/1AS2/1BJ4（fix 比例小，且非正电 native）；2FEO 两版都 0/5。
  新失败集中：正电 native 的核酸/金属结合（5O60_E/21KL_A/3MXB_A/9DWG_L/5CQH）+ 1CGE。
- 组成（native 臂 D/E+K/R 倍率，n40 fix vs n50 unfix）：
  - 总量：0.53 → 0.675（删减缓解部分）
  - 口袋：0.461 → 1.00（fix 构造性恢复）
  - 口袋外表面：0.601 → 0.607（未转移）
  - 核心：0.388 → 0.372（未转移）
  → fix 只堵"结合区删减"出口；全局删减（表面/核心）不变 → 删减主因是全局"删带电残基总数"倾向。
- H3 聚集违规率：fix 全 50 臂高于 unfix（mean viol 0.139→0.172，Δ+0.033pp，1CGE p8 最大 +0.072），
  但全部仍在 native_ref/uncond 基线 +5pp 阈值内 → 50/50 PASS。

## 结论
1. fix 有效抑制结合区删减（口袋带电残基保留 100%），但删减不转移到非结合区——表面/核心倍率 fix 前后不变。
2. 电荷命中代价大（90→36%）：固定位锁住电荷调制自由度 + unfix 校准表对 fix 响应失配，
   正电 native（核酸结合）蛋白过冲最重。
3. H3 略升（+3.3pp）但仍在阈值内；固定带电残基迫使模型在非结合区更密地放电荷。
4. 与 v12.2 pocket_fix 结论一致：fix=堵局部出口，非根治全局删减；且 fix 后需重新小样本标定才能谈 H2。
