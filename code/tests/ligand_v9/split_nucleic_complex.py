"""核糖体/蛋白-核酸复合物拆链：每条可用蛋白链 → 独立单链 PDB + 周围 15Å 核酸配体。

背景（v14 RNA/DNA 结合蛋白数据扩充，2026-09-02）：
  - parse_PDB（data_utils）用 prody select("protein") 只认标准 AA → RNA/DNA 链自动落入配体 Y；
  - 训练要求单蛋白链（多蛋白链会全并进 X）→ 每个复合物拆成多条"单链"样本；
  - 配体保留 = 该蛋白链任一重原子 cutoff Å 内的非蛋白、非水原子（RNA/DNA/小分子/配位离子）。

输出文件名：<pdbid>_<chain>.pdb（protein 链 id 重标 A；配体原子重标 Z）。
QC：写完后用 data_utils.parse_PDB 验证 X 长=预期、N/CA/C/O mask 完整、Y 配体原子数>0。

用法：
  python split_nucleic_complex.py --complex <pdb|cif> --out <dir> [--auto] [--chains A B]
      --cutoff 15.0 --min-len 50 --max-len 500 --min-other 20 --min-mask-ratio 0.9
"""
import argparse
import os
import sys
import time

import numpy as np
from prody import parsePDB, writePDB

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_LIG = os.path.normpath(os.path.join(_CODE_DIR, "../../..", "LigandMPNN"))
for p in (_CODE_DIR, _LIG):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import parse_PDB  # noqa: E402


def near_mask(query_coords, ref_coords, radius, chunk=200):
    """query_coords 中距 ref_coords 任一点 radius 内的索引（bool）。分块防内存爆炸。"""
    hit = np.zeros(len(query_coords), dtype=bool)
    for i in range(0, len(ref_coords), chunk):
        d2 = ((ref_coords[i:i + chunk][:, None, :] - query_coords[None, :, :]) ** 2).sum(-1)
        hit |= (d2 < radius * radius).any(axis=0)
    return hit


def split_chain(atoms, chain, out_path, cutoff, min_len, max_len, min_other, min_mask_ratio):
    """拆指定蛋白链 → 写 PDB，QC 返回 dict 或 raise。"""
    prot = atoms.select("protein")
    ca = prot.select("name CA")
    ca_ch = ca.select(f"chain {chain}")
    L = ca_ch.numAtoms()
    if not (min_len <= L <= max_len):
        raise ValueError(f"chain {chain} L={L} 超范围")
    target = prot.select(f"chain {chain}")
    if target is None or target.numAtoms() == 0:
        raise ValueError(f"chain {chain} 无蛋白原子")
    other = atoms.select("not protein and not water")
    if other is None or other.numAtoms() == 0:
        raise ValueError(f"chain {chain} 无配体原子")
    heavy = target.select("not hydrogen")
    keep = near_mask(other.getCoords(), heavy.getCoords(), cutoff)
    n_other = int(keep.sum())
    if n_other < min_other:
        raise ValueError(f"chain {chain} 配体原子 {n_other} < {min_other}")
    sel = other[keep]
    comb = (target + sel).toAtomGroup()
    nT = target.numAtoms()
    comb.setChids(np.array(["A"] * nT + ["Z"] * n_other))
    comb.setOccupancies(np.ones(comb.numAtoms()))
    comb.setBetas(np.zeros(comb.numAtoms()))
    # 保留原 resnum（不重编号）。若源结构同链内有重复 resnum/插入码导致 parse 塌缩，
    # 下方 QC（L 不匹配）会捕获并跳过该链。
    writePDB(out_path, comb)
    # ---- QC：LigandMPNN parse_PDB ----
    d, _, _, _, _ = parse_PDB(out_path, device="cpu", parse_all_atoms=False)
    Lp = d["X"].shape[0]
    mask_ratio = float(d["mask"].float().mean())
    nY = d["Y"].shape[0]
    if Lp != L:
        os.remove(out_path)
        raise ValueError(f"chain {chain} parse L={Lp} != 预期 {L}")
    if nY < min_other * 0.5:
        os.remove(out_path)
        raise ValueError(f"chain {chain} parse Y={nY} 过少")
    if mask_ratio < min_mask_ratio:
        os.remove(out_path)
        raise ValueError(f"chain {chain} mask {mask_ratio:.2f} < {min_mask_ratio}")
    return {"chain": chain, "L": L, "mask_ratio": round(mask_ratio, 3), "nY": nY}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--auto", action="store_true", help="自动拆全部符合长度的蛋白链")
    ap.add_argument("--chains", default="", help="逗号分隔指定链（优先级高于 --auto）")
    ap.add_argument("--cutoff", type=float, default=15.0)
    ap.add_argument("--min-len", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=500)
    ap.add_argument("--min-other", type=int, default=20)
    ap.add_argument("--min-mask-ratio", type=float, default=0.9)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pid = os.path.basename(args.complex).split(".")[0]
    atoms = parsePDB(args.complex)
    prot = atoms.select("protein")
    ca = prot.select("name CA") if prot is not None else None
    if ca is None:
        print("no protein", flush=True)
        return
    all_chains = sorted(set(ca.getChids()))
    if args.chains:
        targets = [c.strip() for c in args.chains.split(",") if c.strip()]
    elif args.auto:
        targets = all_chains
    else:
        print("need --auto or --chains", flush=True)
        return

    ok, fail = [], []
    t0 = time.time()
    for ch in targets:
        out_path = os.path.join(args.out, f"{pid}_{ch}.pdb")
        if os.path.exists(out_path):
            ok.append({"chain": ch, "skipped": True})
            continue
        try:
            r = split_chain(atoms, ch, out_path, args.cutoff,
                            args.min_len, args.max_len, args.min_other, args.min_mask_ratio)
            ok.append(r)
            print(f"  ok {pid}_{ch} L={r['L']} nY={r['nY']}", flush=True)
        except Exception as e:
            fail.append({"chain": ch, "err": str(e)})
            print(f"  fail {pid}_{ch}: {e}", flush=True)
    print(f"done {pid}: ok {len(ok)} fail {len(fail)} elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
