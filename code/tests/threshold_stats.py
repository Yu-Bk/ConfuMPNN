"""结构过滤器 99 分位阈值统计：从 PDB 采样 N 条蛋白链，统计 4 条规则的触发计数分布。

用法：python threshold_stats.py --pdb-dir <CATH S40 解压目录> --list <S40.list> --n 1000 --seed 42
输出：code/output/threshold_stats.csv + 打印各规则的 50/90/95/99 分位
说明：规则对应 structure_aware_filter.py 的 4 条（同号聚集/盐桥/核心渗入/同号聚类）。
  用 native 序列 + 完整结构计算"每个残基邻域内同号/带电残基数"的分布，
  取 99 分位作为默认阈值（超过 99% 天然蛋白 = 异常聚集）。
"""
import argparse
import csv
import glob
import os
import random

import numpy as np

from src.structure_aware_filter import StructureAwareFilter


def parse_pdb_ca(path):
    """从 PDB 提取 Cα 坐标 + 序列。返回 (coords [L,3], seq1, seqint?)。
    只处理第一条链（CATH domain 文件通常单链）。"""
    coords, resnames = [], []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            resnames.append(line[17:20].strip())
    if not coords:
        return None, None
    return np.array(coords), resnames


RESTYPE3 = {
    "ALA": 0, "CYS": 1, "ASP": 2, "GLU": 3, "PHE": 4, "GLY": 5, "HIS": 6, "ILE": 7,
    "LYS": 8, "LEU": 9, "MET": 10, "ASN": 11, "PRO": 12, "GLN": 13, "ARG": 14,
    "SER": 15, "THR": 16, "VAL": 17, "TRP": 18, "TYR": 19,
}


def seq_to_int(resnames):
    """3 字母 → [L] int（20=X）。未知残基视为 20。"""
    return np.array([RESTYPE3.get(r, 20) for r in resnames], dtype=np.int64)


def compute_features(filt, seq_int):
    """对单结构计算 4 条规则的"计数分布"（native 完整序列，非逐步）。"""
    pos = np.zeros(filt.L, dtype=bool)
    neg = np.zeros(filt.L, dtype=bool)
    for a in ["K", "R"]:
        pos |= (seq_int == {"K": 8, "R": 14}[a])
    for a in ["D", "E"]:
        neg |= (seq_int == {"D": 2, "E": 3}[a])
    charged = pos | neg

    dist = filt._dist
    # 规则1: 10Å 内同号电荷数（正/负分别）
    nb10 = dist <= 10.0
    pos_cnt = (nb10 & pos[None, :]).sum(axis=1)
    neg_cnt = (nb10 & neg[None, :]).sum(axis=1)
    same_sign = np.maximum(pos_cnt, neg_cnt)  # 每个位置的最大同号聚集

    # 规则2: 盐桥 = min(pos,neg) 10Å 内
    salt = np.minimum(pos_cnt, neg_cnt)

    # 规则3: burial（10Å 内 Cα 数归一化）+ 8Å 内带电数
    burial = nb10.sum(axis=1)
    burial_norm = burial / burial.max() if burial.max() > 0 else burial
    nb8 = dist <= 8.0
    charged_cnt = (nb8 & charged[None, :]).sum(axis=1)
    core_charge = charged_cnt[burial_norm > 0.8]  # 仅埋藏位置

    # 规则4: 每残基 8Å 邻域内同号电荷数（局部密度，避免"连通分量全局"误触发）
    nb8_pos = (nb8 & pos[None, :]).sum(axis=1)
    nb8_neg = (nb8 & neg[None, :]).sum(axis=1)
    cluster_same = np.maximum(nb8_pos, nb8_neg)

    return {
        "same_sign_10A": same_sign,
        "salt_bridge_10A": salt,
        "core_charge_8A": core_charge if len(core_charge) else np.array([0]),
        "same_sign_cluster_8A": cluster_same,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/S40/dompdb")
    ap.add_argument("--n", type=int, default=1000, help="采样结构域数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # CATH domain 文件无扩展名，glob 匹配所有文件并过滤非文件
    pdbs = [p for p in glob.glob(os.path.join(args.pdb_dir, "*")) if os.path.isfile(p)]
    if not pdbs:
        raise SystemExit(f"!! {args.pdb_dir} 下没有 PDB")
    random.seed(args.seed)
    sample = random.sample(pdbs, min(args.n, len(pdbs)))
    print(f"采样 {len(sample)} 个结构域（共 {len(pdbs)}）", flush=True)

    feats = {k: [] for k in ["same_sign_10A", "salt_bridge_10A", "core_charge_8A", "same_sign_cluster_8A"]}
    n_ok = 0
    for i, p in enumerate(sample):
        coords, resnames = parse_pdb_ca(p)
        if coords is None or len(coords) < 20:
            continue
        seq_int = seq_to_int(resnames)
        filt = StructureAwareFilter(coords)
        f = compute_features(filt, seq_int)
        for k, arr in f.items():
            if isinstance(arr, np.ndarray):
                feats[k].extend(arr.tolist())
            else:
                feats[k].append(arr)  # 标量（如连通图同号聚类）
        n_ok += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sample)} 处理中，有效 {n_ok}", flush=True)
    print(f"有效结构 {n_ok}", flush=True)

    # 输出 99 分位
    rows = []
    print("\n规则               n     50%     90%     95%     99%")
    for k, arr in feats.items():
        a = np.array(arr, dtype=float)
        if a.size == 0:
            continue
        pcts = np.percentile(a, [50, 90, 95, 99])
        rows.append((k, a.size, *[round(x, 2) for x in pcts]))
        print(f"{k:18s} {a.size:6d}  {pcts[0]:6.1f}  {pcts[1]:6.1f}  {pcts[2]:6.1f}  {pcts[3]:6.1f}")

    out = "/data/nfs/IC/baokun_yu/ConfuMPNN/code/output/threshold_stats.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "n", "p50", "p90", "p95", "p99"])
        w.writerows(rows)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
