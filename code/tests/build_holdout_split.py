"""从训练集 labels_balanced_v7.npz 按电荷分层划 15% 做 hold-out（补正电尾），输出 85% 训练子集。

背景（index/PROJECT_LOCAL_V12_2.md §2.1 + 用户决策 2026-08-30）：
- 剩余域 hold-out（labels_holdout.npz）天然无高正电（charge7 ≤ ~+8），覆盖不了训练集正电尾
  （charge7>+8 共 1,211 域，全来自外部碱性域）
- 用户选"两者都做"：本脚本从训练集 7,886 域按电荷分层划 15%（~1,183）做 hold-out，
  **分布完全匹配训练集（含正电尾）**；剩余 85%（~6,703）作为 v12.2 训练子集。
- 注意：该 hold-out 对 v12/v12.1 是"见过域"（它们训了全部 7,886），**仅对 v12.2 是未见的**
  （v12.2 只训 85%）。

用法（code/ 下）：
  PYTHONPATH=. python tests/build_holdout_split.py
输出：
  ../data/cath/labels_holdout_train.npz（15% hold-out，标签同构 8pH）
  ../data/cath/labels_v12_2_train.npz（85% v12.2 训练子集）
打印：分层划 15% 统计 + 分布对比（hold-out vs train85 vs 完整）
"""
import argparse
import random

import numpy as np

AP = argparse.ArgumentParser()
AP.add_argument("--labels", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/labels_balanced_v7.npz")
AP.add_argument("--frac", type=float, default=0.15, help="hold-out 比例")
AP.add_argument("--n_bins", type=int, default=8, help="按 charge@7.4 分箱数（每箱按 frac 划）")
AP.add_argument("--seed", type=int, default=42)
AP.add_argument("--holdout_out", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/labels_holdout_train.npz")
AP.add_argument("--train_out", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/labels_v12_2_train.npz")
args = AP.parse_args()

from src.differentiable_charge import net_charge

d = np.load(args.labels, allow_pickle=True)
n_dom = len(d["domain_ids"])
n_pH = d["pH"].shape[0] // n_dom
print(f"训练集 {n_dom} 域 × {n_pH} pH，划 {args.frac:.0%} hold-out", flush=True)

# 每域 charge@7.4（分箱特征，重算）
c7 = np.array([net_charge(s, 7.4) for s in d["seqs"]])

# 分层：8 分位等频箱，每箱按 frac 划（每箱至少 1 个）
qs = np.quantile(c7, np.linspace(0, 1, args.n_bins + 1))
rng = random.Random(args.seed)
sel, bin_stat = [], []
for b in range(args.n_bins):
    lo, hi = qs[b], qs[b + 1]
    if b == args.n_bins - 1:
        idx = [i for i, c in enumerate(c7) if lo <= c <= hi]
    else:
        idx = [i for i, c in enumerate(c7) if lo <= c < hi]
    rng.shuffle(idx)
    take = idx[:max(1, int(len(idx) * args.frac))]
    sel.extend(take)
    bin_stat.append((lo, hi, len(idx), len(take)))
sel = sorted(sel)
keep = [i for i in range(n_dom) if i not in set(sel)]
print("\n分箱划 15% 表（[区间]: 箱内 → 抽 hold-out）：")
for lo, hi, total, take in bin_stat:
    print(f"  [{lo:+6.2f},{hi:+6.2f}): {total:5d} → {take:4d}")
print(f"hold-out {len(sel)} 域（{len(sel)/n_dom:.1%}），train85 {len(keep)} 域", flush=True)

# 分布对比
for name, idx in [("完整", range(n_dom)), ("hold-out", sel), ("train85", keep)]:
    c = c7[list(idx)]
    print(f"  {name:8s}: mean={c.mean():6.2f} std={c.std():6.2f} range=[{c.min():6.1f},{c.max():6.1f}] n={len(c)}")

# 切标签（按域索引，pH/charge/pI 每域 n_pH 个）
def slice_npz(idx, out):
    idx = list(idx)
    np.savez(out,
             domain_ids=d["domain_ids"][idx],
             seqs=d["seqs"][idx],
             coords=d["coords"][idx],
             pH=np.concatenate([d["pH"][i * n_pH:(i + 1) * n_pH] for i in idx]),
             charge=np.concatenate([d["charge"][i * n_pH:(i + 1) * n_pH] for i in idx]),
             pI=np.concatenate([d["pI"][i * n_pH:(i + 1) * n_pH] for i in idx]))
    print(f"已写 {out}（{len(idx) * n_pH} 样本）", flush=True)

slice_npz(sel, args.holdout_out)
slice_npz(keep, args.train_out)
