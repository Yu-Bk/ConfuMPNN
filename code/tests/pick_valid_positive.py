"""第十八轮：挑选新的正电验证蛋白（测通用模型泛化）。

从 CATH S40 全部 34,653 域中，选 native 电荷@7.4 ∈ [8, 12]、不在训练集
（labels_balanced.npz 的 776 域）的结构域，复制为 code/input/ 下的标准 PDB
（加 .pdb 后缀），作为复验新增的正电验证蛋白。

用法（code/ 下，confumpnn 环境）：
    PYTHONPATH=. python tests/pick_valid_positive.py --n 2 --lo 8 --hi 12
输出：
    打印候选域，复制 PDB 到 code/input/<did>.pdb，并打印 PDB 清单
"""
import argparse
import glob
import os
import shutil

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
    ap.add_argument("--labels", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/labels_balanced.npz")
    ap.add_argument("--outdir", default="/data/nfs/IC/baokun_yu/ConfuMPNN/code/input")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--lo", type=float, default=8.0)
    ap.add_argument("--hi", type=float, default=12.0)
    args = ap.parse_args()

    train_doms = set(np.load(args.labels, allow_pickle=True)["domain_ids"])
    print(f"训练域（排除）: {len(train_doms)}", flush=True)

    candidates = []
    files = glob.glob(os.path.join(args.dompdb, "*"))
    for i, p in enumerate(files):
        did = os.path.basename(p)
        if did in train_doms:
            continue
        seq = parse_seq(p)
        if len(seq) < 20:
            continue
        c7 = net_charge(seq, 7.4)
        if args.lo <= c7 <= args.hi and 50 <= len(seq) <= 220:
            candidates.append((did, c7, len(seq)))
        if (i + 1) % 5000 == 0:
            print(f"  扫描 {i+1}/{len(files)}，候选 {len(candidates)}", flush=True)
    print(f"候选域（charge {args.lo}~{args.hi}, 长度 50-220）: {len(candidates)}", flush=True)

    # 按电荷离 lo 近的优先（正电但不过激），取前 n 个
    candidates.sort(key=lambda t: (abs(t[1] - args.lo), -t[2]))
    picked = candidates[:args.n]
    os.makedirs(args.outdir, exist_ok=True)
    print("\n选中的验证蛋白:")
    for did, c7, L in picked:
        src = os.path.join(args.dompdb, did)
        dst = os.path.join(args.outdir, f"{did}.pdb")
        shutil.copy(src, dst)
        print(f"  {did}: charge@7.4={c7:+.1f}, L={L} → {dst}")

    with open(os.path.join(args.outdir, "positive_valid_pdbs.txt"), "w") as f:
        for did, c7, L in picked:
            f.write(f"{did} charge={c7:+.1f} L={L}\n")
    print(f"\n清单 → {os.path.join(args.outdir, 'positive_valid_pdbs.txt')}")


if __name__ == "__main__":
    main()
