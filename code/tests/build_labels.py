"""构建 Phase 2 微调的条件标签数据集。

训练样本 = (骨架坐标, native 序列, 条件向量[7])
条件向量 = [pH, has_charge=1, charge, 0, 0, 0, 0]（mask-aware）
其中 charge = native 序列在该 pH 下的净电荷（Henderson-Hasselbalch 平滑计算）。

**self-consistent 设计**：条件电荷用 native 序列自身在 pH 下的净电荷，
使 CE(重建 native) 与 charge_deviation(期望电荷≈条件) 两个损失一致不冲突；
推理时给任意 (pH, target_charge) 外推。

用法（code/ 下）：
  conda activate confumpnn
  PYTHONPATH=. python tests/build_labels.py --n 1000 --n_pH 8 --seed 42
输出：
  data/cath/labels.npz   （coords/seqs/pH/charges/pIs/domain_ids）
  打印条件向量 7 维 μ/σ，并写入 code/configs/condition_defaults.yaml
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
    """mask-aware 条件向量 [7]：pH, has_charge, charge, 0,0,0,0。"""
    return np.array([pH, 1.0, charge, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dompdb", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/S40/dompdb")
    ap.add_argument("--n", type=int, default=1000, help="采样的结构域数")
    ap.add_argument("--n_pH", type=int, default=8, help="每个结构域的 pH 采样数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath/labels.npz")
    ap.add_argument("--cfg", default="/data/nfs/IC/baokun_yu/ConfuMPNN/code/configs/condition_defaults.yaml")
    args = ap.parse_args()

    pdbs = [p for p in glob.glob(os.path.join(args.dompdb, "*")) if os.path.isfile(p)]
    random.seed(args.seed)
    np.random.seed(args.seed)
    sample = random.sample(pdbs, min(args.n, len(pdbs)))
    print(f"采样 {len(sample)} 个结构域（共 {len(pdbs)}），每域 {args.n_pH} 个 pH", flush=True)

    domains, seqs, coords, pHs, charges, pIs = [], [], [], [], [], []
    n_ok = 0
    for i, p in enumerate(sample):
        coords_i, seq = parse_domain(p)
        if coords_i is None:
            continue
        did = os.path.basename(p)
        # 每骨架采样 n_pH 个连续 pH（uniform[4,10]）
        pH_i = np.random.uniform(4.0, 10.0, args.n_pH)
        charge_i = np.array([net_charge(seq, ph) for ph in pH_i], dtype=np.float32)
        pI = find_pI(seq)
        domains.append(did); seqs.append(seq); coords.append(coords_i)
        pHs.append(pH_i); charges.append(charge_i); pIs.append(np.full(args.n_pH, pI, dtype=np.float32))
        n_ok += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sample)} 处理中，有效 {n_ok}", flush=True)
    print(f"有效结构域 {n_ok}", flush=True)

    np.savez(args.out, domain_ids=np.array(domains), seqs=np.array(seqs, dtype=object),
             coords=np.array(coords, dtype=object), pH=np.concatenate(pHs),
             charge=np.concatenate(charges), pI=np.concatenate(pIs))
    print(f"已写 {args.out}（{n_ok * args.n_pH} 个样本）", flush=True)

    # ---- 统计条件向量 μ/σ（7 维）----
    vecs = []
    for pi, ci in zip(pHs, charges):
        for ph, c in zip(pi, ci):
            vecs.append(build_condition_vector(ph, c))
    vecs = np.stack(vecs)
    mean = vecs.mean(axis=0)
    std = vecs.std(axis=0)
    print("\n条件向量 μ:", np.round(mean, 4))
    print("条件向量 σ:", np.round(std, 4))

    # 写入 condition_defaults.yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg["condition_defaults"]["normalization"]["mean"] = [round(float(x), 4) for x in mean]
    cfg["condition_defaults"]["normalization"]["std"] = [round(float(x), 4) for x in std]
    with open(args.cfg, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"μ/σ 已写入 {args.cfg}")


if __name__ == "__main__":
    main()
