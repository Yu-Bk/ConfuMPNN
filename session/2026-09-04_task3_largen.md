# Task3 大样本"三达标"搜索 — 过程记录（2026-09-04）

科学问题：把每臂采样数放大到 n=200，是否存在**同时满足**
① 电荷达标(|净电荷−target|≤2) ② 不重删带电残基(生成 D/E+K/R ≥ 0.7×native)
③ 无电荷聚集(H3 合法) 的序列？——判定 v14 删减/电荷失败是"从不生成合格序列"
还是"稀有事件、多采样可救"。

## 输入（Task3 说明）
- manifest in-10: `data/validation_pdbs/validation_manifest_v14_in.json`
- 编码器 `output/finetune_ligand_v14_rna/finetune_epoch050.pt`；骨架 `LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`（atom25）
- 校准 `output/charge_calibration_v14_ligand_clean.json`（per_protein 含全部 in-10；
  global slope=1.4923 / intercept=-1.260）
- ref 骨架 `output/generalization_ligand_v14_clean/ref/<pdb>_ref.pdb`
- 环境 confumpnn；GPU=cuda:4（共享）

## 口径
- 与 validate_generalization.py 对齐：target = int(round(native_charge)) + Δ
  (native/n2/p2/n8/p8 → 0/−2/+2/−8/+8)；per-protein 校准 tgt_eff=(tgt−b)/a；
  temperature=0.3，seed=2000+k；ligand 模式（use_atom_context=True, atom25）。
- 三指标每序列：
  - dev = |net_charge(seq,pH7.4) − target|，达标 ≤ 2
  - del_ratio = (D/E+K/R)/(native D/E+K/R)，达标 ≥ 0.7
  - H3 = 结构感知 4 规则 full 违规率 ≤ native_ref 违规率 + 0.05（H3 判据 per-seq 化；
    额外记 local=R1+R2+R3，R4 因 Cα 8Å 全链连通而退化为电荷含量，本地规则才是"聚集"信号）
- 脚本：`code/tests/ligand_v9/largen_search_v14.py`
- 输出：`output/largen_v14/<pdb>_arm_<arm>/seqs.fa + stats.json`；汇总 `summary.json`
- 断点续跑：每 arm 读已有 seqs.fa 补采缺失 seed；每 10 条原子化快照。

## 进度
- 2026-09-04 pilot（5CQH/1CGE, n=30, cuda:4）：脚本一次通过。
  pilot 到 5CQH n2/native/p2 后被父会话中断（exit 137，非脚本问题）。
  5CQH n=30 结果：native/n2/p2 三达标均 0（pass C=8/11/15, D=0/2/0, H=30/30/30）
  → 删减(判据②)是主因，电荷次之，H3 不卡（n=30 样本）。
- 全量启动：in-10 × 5 臂 × n=200，cuda:4，nohup 后台（log/largen_v14_search.stdout）。

## 中途观察（采样中）
- 首个完整 n=200 蛋白 5CQH：**n8 臂存在三达标序列 24/200 (12%)**（pass C=57 D=105 H=200，
  del_mean=0.719 → 多采样可救该臂）。native/n2/p2 主因预计删减（n=30 pilot 时 D=0）。
- 截至 10k 采样约半（~4990/10000）：6D2O/1AS2/2FEO/5CQH/1CGE 已采完或接近完。

## 完成状态
- 全量 10 蛋白 × 5 臂 × n=200 = **10000 条采样完成**（2026-09-04 20:xx，GPU4 共享）。
  产物 `output/largen_v14/<pdb>_arm_<arm>/{seqs.fa,stats.json}` + `<pdb>_summary.json`
  + `summary.json`；汇总 `output/largen_v14_summary.json`。
- 汇总脚本 `code/tests/ligand_v9/summarize_largen_v14.py`；报告 `analysis/report/2026-09-04_v14_largen_search.md`。

## 结论
- 存在性：**30/50 臂 (60%) 至少 1 条三达标；9/10 蛋白（唯一全零 = 1BJ4, L=470）**。
  按臂方向：native 5 / n2 7 / p2 5 / n8 7 / p8 6（各 10）。
- 整体 10000 条逐序列通过率：电荷 32.1% / 删减 **17.5%** / H3 99.9% / **三达标 5.2%**。
- 存在率 vs n：跨臂累计三达标序列数 10→28、25→62、50→127、100→270、200→523；
  “至少已有三达标”臂数 10→15、25→18、50→24、100→27、200→30。
  n=50→200 使臂 24→30、条数 127→523（约 4×），但 **20 臂到 n=200 仍为零**（删减系统性，
  尤其 1BJ4 全 5 臂、1AS2/2FEO/5CQH 多数臂）。
- 主因：删减（判据②）≫ 电荷散布（判据①）≫ H3（判据③，~100% 通过，不构成限制）。
- 科学问题判定：**v14 既“会生成”合格序列（非从不），但也非单纯“稀有事件多采样可救”——
  是混合**：60% 臂靠 n=200 可捞到；40% 臂本质不生成（删减捷径）。
