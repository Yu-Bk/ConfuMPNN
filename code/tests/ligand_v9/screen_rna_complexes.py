"""RNA/DNA 结合蛋白复合物筛选：prody 数值距离，找"可用单蛋白链"样本。

判定某蛋白链可用：
  - 链上 CA 数 L ∈ [LMIN, LMAX]（默认 80-500，宁精勿滥）
  - 距该链任意重原子 15Å 内存在 >= MIN_NUC_ATOMS 个核酸原子（标准+常见修饰核苷酸）
输出 JSON：每 PDB → 可用链列表 {chain, L, n_nuc_15, n_nuc_8, n_Y_atoms_15}

用法：
  python screen_rna_complexes.py --raw data/ligand_train/rna_complex_raw \
      --out data/ligand_train/rna_complex_raw/_screen.json
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from prody import parsePDB

NUC = {
    "DA", "DC", "DG", "DT", "DU", "DI", "DTN", "DAN", "DCN", "DGN",
    "A", "C", "G", "U", "T", "I",
    # 常见修饰核苷酸（RNA）
    "1MA", "5MC", "5MU", "7MG", "M2G", "PSU", "OMC", "OMG", "OMU",
    "H2U", "2MG", "YYG", "4SU", "5BU", "2MU", "MDO", "1MG", "2OM", "3MC",
}
LMIN, LMAX = 80, 500
MIN_NUC_15 = 20   # 15A 内核酸原子下限（确保真实结合界面）
CHUNK = 200        # 距离计算的蛋白原子分块


def atoms_near(chain_coords, other_coords, radius):
    """other_coords 中落在 chain_coords 任一点 radius A 内的原子索引。分块避免内存爆炸。"""
    hit = np.zeros(len(other_coords), dtype=bool)
    for i in range(0, len(chain_coords), CHUNK):
        block = chain_coords[i:i + CHUNK]
        # cdist block x other
        d2 = ((block[:, None, :] - other_coords[None, :, :]) ** 2).sum(-1)
        hit |= (d2 < radius * radius).any(axis=0)
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/ligand_train/rna_complex_raw")
    ap.add_argument("--out", default="data/ligand_train/rna_complex_raw/_screen.json")
    ap.add_argument("--min-nuc-15", type=int, default=MIN_NUC_15)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw, "*.pdb")))
    results = {}
    t0 = time.time()
    for fi, p in enumerate(files):
        pid = os.path.basename(p)[:-4]
        try:
            atoms = parsePDB(p)
        except Exception as e:
            results[pid] = {"err": f"parsePDB: {e}"}
            continue
        prot = atoms.select("protein")
        ca = prot.select("name CA") if prot is not None else None
        if ca is None or ca.numAtoms() == 0:
            results[pid] = {"err": "no protein CA"}
            continue
        # 核酸原子（含修饰）: 非蛋白非水 且 resname ∈ NUC
        other = atoms.select("not protein and not water")
        nuc_mask = np.array([r in NUC for r in other.getResnames()])
        nuc = other[nuc_mask] if nuc_mask.any() else None
        # 全配体原子（非蛋白非水，作 Y 大小估计，排除纯离子？此处不排除）
        allY = other
        n_ca_total = ca.numAtoms()
        chains_seen = set(ca.getChids())
        chain_rows = []
        for ch in sorted(chains_seen):
            ca_ch = ca.select("chain " + ch)
            L = ca_ch.numAtoms()
            if not (LMIN <= L <= LMAX):
                continue
            heavy = prot.select(f"chain {ch} and not hydrogen")
            if heavy is None or heavy.numAtoms() == 0:
                continue
            hcoords = heavy.getCoords()
            n_nuc15 = 0
            if nuc is not None and nuc.numAtoms() > 0:
                n_nuc15 = int(atoms_near(hcoords, nuc.getCoords(), 15.0).sum())
            if n_nuc15 < args.min_nuc_15:
                continue
            # 8A 内核酸原子（强接触）
            n_nuc8 = int(atoms_near(hcoords, nuc.getCoords(), 8.0).sum()) if (nuc is not None and nuc.numAtoms() > 0) else 0
            # Y 原子估计（15A 内非蛋白非水，含小分子/离子）
            n_Y15 = int(atoms_near(hcoords, allY.getCoords(), 15.0).sum())
            # 残基完整度（该链 N/CA/C/O mask）稍后 parse_PDB 验，这里记录
            chain_rows.append({
                "chain": ch, "L": L, "n_nuc15": n_nuc15, "n_nuc8": n_nuc8,
                "n_Y15": n_Y15, "n_Y_nuc_res": len(set(nuc.getResnums()[atoms_near(hcoords, nuc.getCoords(), 15.0)])),
            })
        if chain_rows:
            results[pid] = {"ok": True, "file": os.path.basename(p),
                            "size_mb": round(os.path.getsize(p) / 1e6, 1),
                            "chains": chain_rows}
        else:
            results[pid] = {"ok": False, "reason": "no usable chain"}
        if (fi + 1) % 150 == 0:
            print(f"  {fi + 1}/{len(files)}  elapsed {time.time() - t0:.0f}s", flush=True)

    json.dump(results, open(args.out, "w"), indent=1)
    n_ok = sum(1 for v in results.values() if v.get("ok"))
    n_chain = sum(len(v["chains"]) for v in results.values() if v.get("ok"))
    print(f"done {len(files)} files -> {n_ok} PDB with usable chain, {n_chain} chain candidates")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
