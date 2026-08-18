"""v7 步骤1：对碱性域候选按电荷区间分层抽样。

原则（用户要求）：分层 + 极端不主导。相对自然分布适度提高极端正电占比
（治极端样本不足），但极端（+15+）占比控制在 ~18%，不主导训练。
抽样后交序列聚类去重 + 下载 PDB。

用法（code/ 下）：
  PYTHONPATH=. python tests/sample_basic.py \
      --in ../data/cath/candidates_basic_v7.txt \
      --out ../data/cath/sample_basic_v7.txt \
      --seed 42
"""
import argparse
import random

# 每电荷区间的目标抽取数（分层目标）
# 依据：候选分布（[5,10)=27901, [10,15)=6898, [15,20)=3577, [20,25)=715,
#       [25,30)=603, [30,35)=78, [35,40)=62），总量目标 ~3500，极端 ~18%
BIN_TARGETS = [
    (5.0, 8.0, 700),    # 温和碱性
    (8.0, 10.0, 1100),  # 温和碱性
    (10.0, 12.0, 800),  # 中等正电
    (12.0, 15.0, 300),  # 中等正电
    (15.0, 18.0, 300),  # 强正电
    (18.0, 20.0, 200),  # 强正电
    (20.0, 25.0, 70),   # 极端
    (25.0, 30.0, 30),   # 极端
    (30.0, 35.0, 10),   # 极端
    (35.0, 40.0, 5),    # 极端
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []  # (did, pdb, chain, L, q)
    with open(args.inp) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            rows.append((parts[0], parts[1], parts[2], int(parts[3]), float(parts[4])))

    rng = random.Random(args.seed)
    selected = []
    stat = []
    for lo, hi, target in BIN_TARGETS:
        pool = [r for r in rows if lo <= r[4] < hi]
        rng.shuffle(pool)
        take = pool[:target]  # 不足全取
        selected.extend(take)
        stat.append((lo, hi, len(pool), len(take)))
    selected.sort(key=lambda r: -r[4])

    print("分层抽样统计（[区间]: 候选总数 → 抽取数）:")
    for lo, hi, pool_n, take_n in stat:
        short = " ⚠️候选不足" if take_n < pool_n else ""
        print(f"  [{lo:+4.1f}, {hi:+4.1f}): {pool_n:6d} → {take_n:4d}{short}")
    n_extreme = sum(1 for r in selected if r[4] >= 15.0)
    print(f"  合计 {len(selected)} 域；极端(≥+15) {n_extreme} ({n_extreme/len(selected)*100:.1f}%)")

    with open(args.out, "w") as f:
        f.write("# DomainID  PDB  chain  length  charge7\n")
        for did, pdb, chain, L, q in selected:
            f.write(f"{did}\t{pdb}\t{chain}\t{L}\t{q:.4f}\n")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
