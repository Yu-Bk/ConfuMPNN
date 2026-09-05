# 2026-09-05 受控减预算消融 — 运行记录

> 母计划：`index/PROJECT_LOCAL_V14_FINAL_EXPERIMENTS.md` §1；落地计划副本 `ablation/plan.md`。
> 协调者 2026-09-05 指定：产物收 `ablation/`（不写 output/ablate_* 与 analysis/report）；不 git。

## run 矩阵（同族同子集/同 seed42/同基座，只一处差异）
- 蛋白族 5（v12.2 配方 MoMPNN，epochs10）：run_FULL / run_nov12comp / run_notarget / run_noph / run_nokeep
- 配体族 6（v14 配方 LigandMPNN atom25，epochs16）：run_FULL / run_nov12comp / run_notarget / run_noA1 / run_noph / run_nokeep

## 数据子集（完成）
- `ablation/data/labels_ablate_prot.npz`：6580→1659 域（25.2%）；L mean 181.8→184.9，Q mean 0.06→0.15（后验近全集）
- `ablation/data/labels_ablate_lig.npz`：5371→1364 域（25.4%）；L 281.0→282.1，Q −1.19→−1.16；
  RNA/DNA 420(7.82%)→114(8.36%)（分层 ceil 略偏高，可接受）
- 脚本 `ablation/data/build_ablate_subsets.py`；平衡报告 `ablation/data/subsets_balance.json`

## dry-run（完成）
- prot FULL & lig FULL（各 6 域 1 epoch）通过；flag 识别正确（[v12] λ0.2/λt0.2、[A1 keep]λ0.0 prot /
  [A1 global]λ0.3 lig）。配体 dry-run 预解析 6 域 ~29s。
- probe 脚本 `ablation/runs/gen_probe.py` 用 v12.2 encoder 在 1BC8 验证通过（native/n2/p2 H2 全过）。

## 训练启动
- 02:57 prot driver `ablation/runs/run_prot_ablation.sh` 后台（GPU6），顺序 FULL→nov12comp→notarget→noph→nokeep。
- 蛋白族完成后启 `ablation/runs/run_lig_ablation.sh`（16ep × 6）。
- val-loss eval 脚本：`ablation/runs/eval_val_loss.sh`（prot tag v12_2 / lig tag v14_ligand，最终 epoch）。

## 评估约定
- val-loss：同族统一 FULL 配方口径回放（`val_loss_curve.py`），保证 total/ce/cd 横向可比；
  prot 蛋白 val=1176+23=1199；lig val=805。
- 生成抽查：小蛋白 1BC8 + 长蛋白 1A65（蛋白族）；5O60_E RNA（配体族）；native/n2/p2 n30，H2(dev≤2)
  + native 臂带电保留率；raw（无校准）。

## 蛋白族 — 完成（05:24 训练 / 05:39 val eval / ~05:50 probe）
- 5 runs 正常无 NaN；epochs10。val-loss eval（n_dom 1104/1199）见报告：
  - FULL cd=2.683 total=4.410（基线）
  - −v12组成 cd +1.6% total +3.3% v12_ct +16.9%；probe 1A65 native dev 5.5→17.6 崩溃 → **贡献最大**
  - −λ_target cd +5.4% total +1.8% v12_ct +10.8%；1BC8 native/p2 过冲 H2 丢、retention 1.18 过添加 → **次大**
  - −seq_keep cd −8.1% 但 retention 1BC8 0.84（删减捷径）、kl +62% → 保组成稳健
  - −ph_filter cd +0.2% total −0.8% → **最小**（低权重结构辅助）
  - 排序：v12组成>λ_target>seq_keep>ph_filter；under-train 部分压平 val-loss 差异但生成端差异未被压平，排序可信。
- 报告：`ablation/report/2026-09-05_ablation_prot.md`；图数据 `ablation/figure/ablation_prot_figdata.json`。

## 配体族 — 进行中
- 05:26 lig driver 启动（GPU6），run_FULL parse ~10min（1364 域），epoch ~3min → 16ep≈48min/run，
  6 runs 预计 ~5.8h → 约 11:20 完成。之后配体 val-loss eval + probe（5O60_E RNA）。

## 配体族 — 完成（10:11 训练 / 10:32 val eval / 10:40 probe）
- 6 runs 正常无 NaN；epochs16；val-loss eval n_dom=805/805。关键：
  - FULL cd=3.157 total=4.793（基线，v12_comp0.036 v12_ct4.33 pocket0.235）
  - −v12组成：v12_comp 0.036→3.22（+88×）、pocket 0.235→0.62、total +15%；probe 5O60_E retention 0.79→0.50
    （删减一半带电残基）→ **贡献最大**
  - −λ_target：v12_ct 4.33→5.15(+19%)、cd +0.6%；retention 0.80 → **次大**
  - −seq_keep：ce +5.8%、kl +44%（漂移）、cd −12%（删减捷径致命中易）；retention 0.794 → 保稳健
  - −A1：pocket +11%、cd −4%、total −1.5%；retention 0.772 → 温和正贡献
  - −ph_filter：几乎不变 → **最小**
  - 排序：v12组成 > λ_target > seq_keep > A1(pocket) > ph_filter
- under-train：16ep 未完全收敛；除 −v12组成(total+15%/comp88×)外差异偏小（部分压平），但核心效应未被压平 → 排序可信。
- probe raw 全过冲（配体响应增益，正式需校准），模块比较以 retention/组成为主。
- 报告：`ablation/report/2026-09-05_ablation_lig.md`；图数据 `ablation/figure/ablation_lig_figdata.json`。
- 两族全部产物在 `ablation/{plan,data,runs,report,figure}`；session 本文件。
