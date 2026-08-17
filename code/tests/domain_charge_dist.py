"""统计 CATH S40 全部 34,653 域在 pH7.4 的 native 电荷分布（第十八轮用）。

用途：确认"分层平衡采样"的可行性——按 native 电荷分箱，看每箱有多少域，
决定每箱抽多少能凑出电荷分布均匀的训练集。

用法（code/ 下，confumpnn 环境）：
    python tests/domain_charge_dist.py [--dompdb 路径] [--out npy路径]
输出：
    {out}/domain_charge7.npy  全部域的 pH7.4 净电荷数组
    打印分箱直方图 + 每箱可抽数量
"""
import argparse
import glob
import os

import numpy as np

from src.differentiable_charge import net_charge

R3 = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
      'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
      'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
      'TRP': 'W', 'TYR': 'Y'}


def parse_seq(path):
    seq = []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            r = line[17:20].strip()
            if r in R3:
                seq.append(R3[r])
    return "".join(seq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dompdb", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/S40/dompdb")
    ap.add_argument("--out", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/domain_charge7.npy")
    ap.add_argument("--pH", type=float, default=7.4)
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.dompdb, "*"))
    print(f"总域数: {len(files)}", flush=True)
    charges, ok = [], 0
    for i, p in enumerate(files):
        s = parse_seq(p)
        if len(s) < 20:
            continue
        charges.append(net_charge(s, args.pH))
        ok += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(files)}...", flush=True)
    ch = np.array(charges)
    np.save(args.out, ch)
    print(f"有效 {ok}, charge@{args.pH}: mean={ch.mean():.2f} min={ch.min():.1f} max={ch.max():.1f}", flush=True)

    bins = list(range(-20, 25, 5))
    hist, edges = np.histogram(ch, bins=bins)
    for i in range(len(hist)):
        print(f"  [{edges[i]:+4.0f},{edges[i+1]:+4.0f}): {hist[i]:5d} 域")
    for per in [50, 80, 100, 120]:
        n_box = sum(1 for h in hist if h >= per)
        print(f"  每箱抽 {per}: {n_box} 箱有数据 → 约 {n_box * per} 域")


if __name__ == "__main__":
    main()
