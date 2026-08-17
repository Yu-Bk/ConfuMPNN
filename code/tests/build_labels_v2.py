"""构建第十八轮分层平衡标签数据集（替换随机抽样）。

与 build_labels.py 的区别：**按 native 电荷分层等量抽样**，替换原随机抽样。
使训练集电荷分布均匀覆盖 -20~+20（正电域与负电域数量相当），根治"训练偏负
→ 高正电 target 外推过冲"（1BC8/2LZM 正电富集蛋白过冲的根因）。

背景（第十八轮计划 session/2026-08-17_plan_v3_general_model.md）：
- 实测 CATH S40 34,612 域电荷@7.4：[-20,-15) 634 / [-15,-10) 2333 / [-10,-5) 7379 /
  [-5,0) 13504 / [0,+5) 8254 / [+5,+10) 1801 / [+10,+15) 308 / [+15,+20) 76
- 原 999 域随机抽样：native>+10 仅 1.3% → 高正电 target 分布外 → 过冲
- 分层采样：8 箱每箱等量抽取，正电域数量与负电相当

用法（code/ 下）：
  PYTHONPATH=. python tests/build_labels_v2.py --stratify --n_bins 8 --per_bin 100 \
      --out data/cath/labels_balanced.npz
输出：
  data/cath/labels_balanced.npz（domain_ids/seqs/coords/pH/charge/pI）
  打印分箱抽样统计 + 更新 condition_defaults.yaml 的 μ/σ（重算）
"""
import argparse
import glob
import os
import random

import numpy as np
import yaml

from src.differentiable_charge import net_charge
from src.isoelectric_point import find_pI

RESTYPE3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}


def parse_domain(path):
    """从 CATH domain 文件提取 Cα 坐标 [L,3] + 序列 [L]。返回 (coords, seq)。"""
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


def build_condition_vector(pH, charge):
    return np.array([pH, 1.0, charge, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def stratify_sample(files, charge7, n_bins, per_bin, lo=-20.0, hi=20.0, seed=42):
    """按 native 电荷分层等量抽样。

    把 [lo, hi] 均匀分 n_bins 箱，每箱抽 per_bin 个（不足全取）。
    返回选中域索引列表。
    """
    rng = random.Random(seed)
    edges = np.linspace(lo, hi, n_bins + 1)
    selected = []
    stat = []
    for b in range(n_bins):
        lo_b, hi_b = edges[b], edges[b + 1]
        # 边界处理：最后一箱含右端点
        if b == n_bins - 1:
            idx = [i for i, c in enumerate(charge7) if lo_b <= c <= hi_b]
        else:
            idx = [i for i, c in enumerate(charge7) if lo_b <= c < hi_b]
        rng.shuffle(idx)
        take = idx[:per_bin]
        selected.extend(take)
        stat.append((lo_b, hi_b, len(idx), len(take)))
    print("\n分层抽样统计（[区间]: 箱内总数 → 抽取数）:")
    for lo_b, hi_b, total, take in stat:
        print(f"  [{lo_b:+4.0f}, {hi_b:+4.0f}): {total:6d} → {take:4d}")
    print(f"  合计 {len(selected)} 域", flush=True)
    return sorted(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dompdb", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/S40/dompdb")
    ap.add_argument("--n_pH", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/labels_balanced.npz")
    ap.add_argument("--cfg", default="/data/nfs/IC/baokun_yu/ConfuMPNN/code/configs/condition_defaults.yaml")
    ap.add_argument("--stratify", action="store_true", help="分层等量抽样（第十八轮，默认开）")
    ap.add_argument("--n_bins", type=int, default=8, help="电荷分箱数")
    ap.add_argument("--per_bin", type=int, default=100, help="每箱抽取数")
    args = ap.parse_args()

    files = [p for p in glob.glob(os.path.join(args.dompdb, "*")) if os.path.isfile(p)]
    print(f"候选域 {len(files)}", flush=True)

    # 第一遍：解析全部域，算 native 电荷@7.4
    parsed = []  # (path, coords, seq, charge7)
    charge7s = []
    for i, p in enumerate(files):
        coords, seq = parse_domain(p)
        if coords is None:
            continue
        c7 = net_charge(seq, 7.4)
        parsed.append((p, coords, seq))
        charge7s.append(c7)
        if (i + 1) % 5000 == 0:
            print(f"  解析 {i+1}/{len(files)}", flush=True)
    print(f"有效域 {len(parsed)}, charge@7.4 mean={np.mean(charge7s):.2f}", flush=True)

    # 分层抽样
    if args.stratify:
        sel_idx = stratify_sample(files, charge7s, args.n_bins, args.per_bin, seed=args.seed)
    else:
        random.seed(args.seed)
        sel_idx = sorted(random.sample(range(len(parsed)), min(1000, len(parsed))))

    sample = [parsed[i] for i in sel_idx]
    print(f"选中 {len(sample)} 域", flush=True)

    # 第二遍：为选中域构建标签
    random.seed(args.seed)
    np.random.seed(args.seed)
    domains, seqs, coords_all, pHs, charges, pIs = [], [], [], [], [], []
    for i, (p, coords_i, seq) in enumerate(sample):
        pH_i = np.random.uniform(4.0, 10.0, args.n_pH)
        charge_i = np.array([net_charge(seq, ph) for ph in pH_i], dtype=np.float32)
        pI = find_pI(seq)
        domains.append(os.path.basename(p))
        seqs.append(seq)
        coords_all.append(coords_i)
        pHs.append(pH_i)
        charges.append(charge_i)
        pIs.append(np.full(args.n_pH, pI, dtype=np.float32))
        if (i + 1) % 200 == 0:
            print(f"  标签 {i+1}/{len(sample)}", flush=True)

    np.savez(args.out, domain_ids=np.array(domains), seqs=np.array(seqs, dtype=object),
             coords=np.array(coords_all, dtype=object), pH=np.concatenate(pHs),
             charge=np.concatenate(charges), pI=np.concatenate(pIs))
    print(f"已写 {args.out}（{len(sample) * args.n_pH} 样本）", flush=True)

    # 统计条件向量 μ/σ（7 维）并写入 condition_defaults.yaml
    vecs = []
    for pi, ci in zip(pHs, charges):
        for ph, c in zip(pi, ci):
            vecs.append(build_condition_vector(ph, c))
    vecs = np.stack(vecs)
    mean, std = vecs.mean(axis=0), vecs.std(axis=0)
    print("\n条件向量 μ:", np.round(mean, 4))
    print("条件向量 σ:", np.round(std, 4))

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg["condition_defaults"]["normalization"]["mean"] = [round(float(x), 4) for x in mean]
    cfg["condition_defaults"]["normalization"]["std"] = [round(float(x), 4) for x in std]
    with open(args.cfg, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"μ/σ 已写入 {args.cfg}")


if __name__ == "__main__":
    main()
