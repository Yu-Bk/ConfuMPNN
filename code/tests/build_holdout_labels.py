"""构建 hold-out 15% 验证集标签（分层抽样剩余 CATH 域，均值/方差匹配训练集）。

背景（index/PROJECT_LOCAL_V12_2.md §2.1）：
- 训练集 labels_balanced_v7.npz = 7,886 域（分层平衡，均值 1.42 / std 9.44，8 pH）
- CATH S40 dompdb 共 34,653 域 → 剩余 27,445 域 = "有结构答案但从未训练"的天然 hold-out
- 目的：建立同分布（in-distribution）hold-out 验证集，评估模型在未见训练域上的
  native 电荷命中（H2）、recovery、可选折叠——与泛化 OOD 测试互补。

抽样：按 native 电荷@pH7.4 分箱（训练集 8 分位等频箱 → 箱内占比与训练一致），
  每箱抽 n_total/n_bins 个（不足全取）→ 整体 mean/std 自动匹配训练集。
标签：与训练 labels_balanced_v7 同构（每域 8 个随机 pH ∈ [4,10] 的 net_charge + pI）。
  物理计算（net_charge/find_pI）不需要训练，故 hold-out 标签可提前构建。

用法（code/ 下）：
  PYTHONPATH=. python tests/build_holdout_labels.py \
      --n_total 1500 --seed 42 --out ../data/cath/labels_holdout.npz
输出：
  ../data/cath/labels_holdout.npz（domain_ids/seqs/coords/pH/charge/pI）
  打印：剩余域统计 + 分箱抽样表 + 分布对比（mean/std/范围，vs 训练集）
"""
import argparse
import glob
import os
import random

import numpy as np

RESTYPE3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}

# 验证蛋白 PDB 前缀（10 泛化 + 5 早期验证），hold-out 必须排除防泄漏
EXCLUDE_PFX = [
    "1c6o", "1azm", "1as2", "1axw", "2feo", "5cqh", "1cge", "1ag0", "1a65", "1bj4",
    "1mbn", "4dfr", "1fqg", "5hvx", "3t0f",
]


def parse_domain(path):
    """从 CATH domain 文件提取 Cα 坐标 [L,3] + 序列 [L]。返回 (coords, seq) 或 (None,None)。"""
    coords, resnames = [], []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            r = line[17:20].strip()
            if r in RESTYPE3TO1:
                resnames.append(RESTYPE3TO1[r])
    if len(resnames) < 20:
        return None, None
    return np.array(coords, dtype=np.float32), "".join(resnames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dompdb", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/S40/dompdb")
    ap.add_argument("--train_labels",
                    default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/labels_balanced_v7.npz")
    ap.add_argument("--n_total", type=int, default=1500, help="hold-out 目标域数（15% 量级）")
    ap.add_argument("--n_bins", type=int, default=8, help="按 native charge@7.4 分箱数")
    ap.add_argument("--n_pH", type=int, default=8, help="每域标签 pH 数（与训练同构）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/labels_holdout.npz")
    ap.add_argument("--limit", type=int, default=0,
                    help="dry-run：只处理前 N 个剩余域（验证逻辑，默认 0=全部）")
    ap.add_argument("--dry_run", action="store_true",
                    help="全量解析 + 抽样 + 分布对比，但不构建标签（启动前检查用）")
    args = ap.parse_args()

    from src.differentiable_charge import net_charge
    from src.isoelectric_point import find_pI

    # ---- 1. 训练域 + 其 charge7 分布（用 labels 里的序列，不需解析 PDB）----
    train = np.load(args.train_labels, allow_pickle=True)
    train_ids = set(train["domain_ids"])
    train_charge7 = np.array([net_charge(seq, 7.4) for seq in train["seqs"]])
    print(f"[1] 训练域 {len(train_ids)}，charge@7.4 mean={train_charge7.mean():.2f} "
          f"std={train_charge7.std():.2f} range=[{train_charge7.min():.1f},{train_charge7.max():.1f}]",
          flush=True)

    # ---- 2. 剩余域解析（排除训练域 + 验证蛋白）----
    files = sorted(p for p in glob.glob(os.path.join(args.dompdb, "*")) if os.path.isfile(p))
    rem_parsed = []  # (path, coords, seq, charge7)
    for i, p in enumerate(files):
        base = os.path.basename(p).lower()
        if base in train_ids or any(base.startswith(pfx) for pfx in EXCLUDE_PFX):
            continue
        coords, seq = parse_domain(p)
        if coords is None:
            continue
        rem_parsed.append((p, coords, seq, net_charge(seq, 7.4)))
        if args.limit and len(rem_parsed) >= args.limit:
            break
        if (i + 1) % 5000 == 0:
            print(f"  解析 {i+1}/{len(files)}（有效 {len(rem_parsed)}）", flush=True)
    print(f"[2] 剩余候选域 {len(rem_parsed)}（已排除训练 {len(train_ids)} + 验证蛋白）", flush=True)
    if args.limit:
        print(f"    dry-run：仅处理 {len(rem_parsed)} 个", flush=True)

    # ---- 3. 分层抽样：训练集 8 分位等频箱 → 每箱抽 ceil(n_total/n_bins) ----
    qs = np.quantile(train_charge7, np.linspace(0, 1, args.n_bins + 1))
    per_bin = -(-args.n_total // args.n_bins)  # ceil
    rng = random.Random(args.seed)
    selected = []
    bin_stat = []
    charge7_sel = []
    for b in range(args.n_bins):
        lo, hi = qs[b], qs[b + 1]
        if b == args.n_bins - 1:
            idx = [i for i, (_, _, _, c) in enumerate(rem_parsed) if lo <= c <= hi]
        else:
            idx = [i for i, (_, _, _, c) in enumerate(rem_parsed) if lo <= c < hi]
        rng.shuffle(idx)
        take = idx[:per_bin]
        selected.extend(take)
        bin_stat.append((lo, hi, len(idx), len(take)))
        charge7_sel += [rem_parsed[i][3] for i in take]
    selected = sorted(selected)
    charge7_sel = np.array(charge7_sel)
    print("\n[3] 分箱抽样表（[区间]: 箱内候选 → 抽取）：")
    for lo, hi, total, take in bin_stat:
        print(f"  [{lo:+6.2f},{hi:+6.2f}): {total:5d} → {take:4d}")
    print(f"  合计 {len(selected)} 域（目标 {args.n_total}）", flush=True)

    # ---- 4. 分布对比（均值/方差匹配检查）----
    print("\n[4] 分布对比（native charge@7.4）：")
    for name, arr in [("训练集", train_charge7), ("hold-out", charge7_sel)]:
        print(f"  {name:8s} mean={arr.mean():6.2f} std={arr.std():6.2f} "
              f"range=[{arr.min():6.1f},{arr.max():6.1f}] n={len(arr)}")
    print("  → mean 差 = %.2f，std 差 = %.2f（目标：同数量级，mean 差<0.5）"
          % (charge7_sel.mean() - train_charge7.mean(),
             charge7_sel.std() - train_charge7.std()), flush=True)

    # ---- 5. 构建标签（与训练同构：每域 8 个随机 pH）----
    if args.limit or args.dry_run:
        print("\n[dry-run] 跳过标签构建（仅验证抽样逻辑）", flush=True)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    domains, seqs, coords_all, pHs, charges, pIs = [], [], [], [], [], []
    for i, (p, coords_i, seq, _) in enumerate([rem_parsed[j] for j in selected]):
        pH_i = np.random.uniform(4.0, 10.0, args.n_pH)
        charge_i = np.array([net_charge(seq, ph) for ph in pH_i], dtype=np.float32)
        pI = find_pI(seq)
        domains.append(os.path.basename(p))
        seqs.append(seq)
        coords_all.append(coords_i)
        pHs.append(pH_i)
        charges.append(charge_i)
        pIs.append(np.full(args.n_pH, pI, dtype=np.float32))
        if (i + 1) % 300 == 0:
            print(f"  标签 {i+1}/{len(selected)}", flush=True)

    np.savez(args.out, domain_ids=np.array(domains), seqs=np.array(seqs, dtype=object),
             coords=np.array(coords_all, dtype=object), pH=np.concatenate(pHs),
             charge=np.concatenate(charges), pI=np.concatenate(pIs))
    print(f"[5] 已写 {args.out}（{len(selected) * args.n_pH} 样本）", flush=True)


if __name__ == "__main__":
    main()
