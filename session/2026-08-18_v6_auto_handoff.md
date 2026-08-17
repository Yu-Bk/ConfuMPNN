# v6 训练自动完成交接快照（2026-08-18 凌晨）

> 用途：v6 训练后台运行，用户休息。此文件供上下文压缩/会话恢复后无缝续接。
> 训练进程 PID 3184938（后台监控 bag1gq9s2 会在训练完成时通知）。

## 1. v6 训练配置（已启动，epoch 11/30 时快照）
```
train_finetune.py --device cuda:1 \
  --labels /data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/labels_balanced_v6.npz \
  --epochs 30 \
  --perturb_prob 0.3 --perturb_scale 8 --placeholder_prob 0.15 --lambda_keep 0.5 --charge_temp 0.5 \
  --loss_reweight 1 --reweight_k 1.0 --reweight_eps 1e-3 --reweight_cap 2 \
  --out_dir output/finetune_v6 --log_file log/train_v6.log
```
- 数据：7,208 域（acid 2500 / neutral 2500 / basic 2208 全保留），泄漏检查 5/5 通过
- μ/σ：charge mean +0.509、std 9.2157（已写入 condition_defaults.yaml，v5 备份 condition_defaults_v5_backup.yaml）
- 预计完成 ≈ 02:10（单 epoch ~6.2min × 30）
- checkpoint：`output/finetune_v6/finetune_epoch030.pt`

## 2. v6 数据重建（已完成）
- 改 `tests/build_labels_v2.py`：新增 `--class_balance --per_class`（三类等量）+ `--exclude`（泄漏保护）
- 命令：`--class_balance --per_class 2500 --exclude 1b24,1bc8,1crn,1ubq,2lzm`
- 产物：`data/cath/labels_balanced_v6.npz`

## 3. 训练 bug 修复（train_finetune.py 两处）
- 预解析加 try/except 跳过坏域（`1c77B00`、`2qe7G01` prody 解析 None）
- 训练循环用 `n_dom_eff = len(domains)`（跳过坏域后 7206 域，原 n_dom=7208 会越界）

## 4. 训练完成后自动流程（顺序执行）
```bash
cd /data/nfs/IC/baokun_yu/ConfuMPNN/code
PY_CONF=/home/baokun_yu/miniconda3/envs/confumpnn/bin/python

# (1) 采样：5 PDB × 6 臂 × n=20
PYTHONPATH=. $PY_CONF tests/phase3_v2_validation.py \
  --cond_encoder output/finetune_v6/finetune_epoch030.pt \
  --out_dir output/finetune_v6_validate --n 20 --device cuda:1

# (2) 打分：ESMFold/TM/Protein-Sol/TemBERTure（GPU 1）
bash /data/nfs/IC/baokun_yu/ConfuMPNN/code/tests/phase3_v2_score.sh \
  /data/nfs/IC/baokun_yu/ConfuMPNN/code/output/finetune_v6_validate 1

# (3) 判定：DESIGN_CRITERIA v2
PYTHONPATH=. $PY_CONF tests/phase3_v2_stats.py --root output/finetune_v6_validate
```
⚠️ 后台 shell 不加载 conda，一律用绝对路径 $PY_CONF。

## 5. 复验重点（判定时逐条核对）
1. **1UBQ 恢复**：v5 仅 1/5（t2_neg），v6 目标 ≥3/5。中性骨架多样性 600→2500 后应改善。
2. **1BC8 全命中保留**：极端正电 +17 dev 应 <2.0。
3. **1b24A01 泛化**：已从训练集排除（泄漏修复），可作泛化验证（v4 4/5、v5 3/5）。
4. **极端正电过冲**（v5: 2LZM dev 6.71、1b24A01 dev 6.10、1CRN 2.98）：碱性全保留后应改善。
5. H3 1CRN t2_neg 违规率略超（v5 0.172 vs base 0.120）——关注是否改善。

## 6. 判定标准 v2（简）
- H1 折叠：TM 中位≥0.70 且失败率≤10%
- H2 电荷：|mean−target|≤2.0（t2_ph 占位臂不判）
- S4 固定：t1_cond 固定位点 100% 保持
- H3：条件臂违规率 ≤ 基线+5pp
- S1*：pairID 0.4-0.7 无坍塌

## 7. 报告存档 + push
- 写 `analysis/report/2026-08-18_phase3_v6_class_balance.md`（结构仿 v5 报告）
- 更新 `index/DOCUMENT_INDEX.md`（v6 行）、memory `confumpnn-project-status.md` + `MEMORY.md`
- `git add` 报告/脚本/配置 → commit → `git push`（output/ data/ 被 .gitignore 不提交）
- 训练 bug 修复的 train_finetune.py + build_labels_v2.py 改动要提交

## 8. 下一步计划（v7 候选）
- 若 1UBQ 恢复 + 正电改善保留 → v6 平衡点成立，可作为通用模型基线
- 若极端正电仍过冲 → 考虑：碱性域外部补充 / 极端 target 课程学习（用户提过"先学简单再学难"）
- R1 天然蛋白对参照（可选）
- 真实骨架泛化：用户提供 RF3 relax/人工骨架后复用同一流程
