"""7K00（E. coli MRE600 70S, 1.98A EM）拆链：核糖体蛋白单链 + 邻近核酸（rRNA/tRNA/mRNA）配体上下文。

独立脚本，仿 code/tests/ligand_v9/split_nucleic_complex.py 的同款策略，但：
  - 配体只保留核酸（RNA/DNA 聚合物 ATOM，排除离子/抗生素/水）→ 仿 data/validation_pdbs/5O60_E.pdb；
  - 蛋白链按实体身份命名（7K00_L2 等）；
  - 长度过滤 min_len=50 / max_len=500（复用训练集策略，丢弃部分模型 L9 与超短 L34/L36）。

输出：output/ribosome_7k00/pdbs/7K00_<name>.pdb（蛋白链重标 A，核酸配体重标 Z）+ summary JSON。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from prody import parsePDB, writePDB

_CONFU = "/data/nfs/IC/baokun_yu/ConfuMPNN"
_LIG = os.path.join(_CONFU, "LigandMPNN")
for p in (_CONFU, _LIG):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import parse_PDB  # noqa: E402

# auth chain -> E. coli 核糖体蛋白名（依据 7K00 _entity 表）
CHAIN_NAME = {
    # 30S
    "B": "S2", "C": "S3", "D": "S4", "E": "S5", "F": "S6", "G": "S7",
    "H": "S8", "I": "S9", "J": "S10", "K": "S11", "L": "S12", "M": "S13",
    "N": "S14", "O": "S15", "P": "S16", "Q": "S17", "R": "S18", "S": "S19",
    "T": "S20", "U": "S21",
    # 50S
    "c": "L2", "d": "L3", "e": "L4", "f": "L5", "g": "L6", "h": "L9",
    "i": "L13", "j": "L14", "k": "L15", "l": "L16", "m": "L17", "n": "L18",
    "o": "L19", "p": "L20", "q": "L21", "r": "L22", "s": "L23", "t": "L24",
    "u": "L25", "v": "L27", "w": "L28", "x": "L29", "y": "L30", "z": "L32",
    "0": "L33", "1": "L34", "2": "L35", "3": "L36", "4": "L31",
}
# 长度过滤（训练集同款 min 50 / max 500）
MIN_LEN = 50
MAX_LEN = 500
MIN_OTHER = 20
MIN_MASK = 0.9
CUTOFF = 15.0


def near_mask(query_coords, ref_coords, radius, chunk=200):
    hit = np.zeros(len(query_coords), dtype=bool)
    for i in range(0, len(ref_coords), chunk):
        d2 = ((ref_coords[i:i + chunk][:, None, :] - query_coords[None, :, :]) ** 2).sum(-1)
        hit |= (d2 < radius * radius).any(axis=0)
    return hit


def split_one(atoms, chain, out_path):
    name = CHAIN_NAME[chain]
    prot = atoms.select("protein")
    ca_ch = prot.select(f"chain {chain} and name CA")
    L = ca_ch.numAtoms()
    if not (MIN_LEN <= L <= MAX_LEN):
        raise ValueError(f"L={L} 超范围 [{MIN_LEN},{MAX_LEN}]")
    target = prot.select(f"chain {chain}")
    if target is None or target.numAtoms() == 0:
        raise ValueError("无蛋白原子")
    # 配体 = 核酸聚合物（排除蛋白/水/离子/抗生素），15A 内
    other = atoms.select("nucleic")
    if other is None or other.numAtoms() == 0:
        raise ValueError("无核酸配体原子")
    heavy = target.select("not hydrogen")
    keep = near_mask(other.getCoords(), heavy.getCoords(), CUTOFF)
    n_other = int(keep.sum())
    if n_other < MIN_OTHER:
        raise ValueError(f"核酸配体原子 {n_other} < {MIN_OTHER}")
    sel = other[keep]
    comb = (target + sel).toAtomGroup()
    nT = target.numAtoms()
    comb.setChids(np.array(["A"] * nT + ["Z"] * n_other))
    comb.setOccupancies(np.ones(comb.numAtoms()))
    comb.setBetas(np.zeros(comb.numAtoms()))
    writePDB(out_path, comb)
    # QC
    d, _, _, _, _ = parse_PDB(out_path, device="cpu", parse_all_atoms=False)
    Lp = d["X"].shape[0]
    mask_ratio = float(d["mask"].float().mean())
    nY = d["Y"].shape[0]
    if Lp != L:
        os.remove(out_path)
        raise ValueError(f"parse L={Lp} != 预期 {L}")
    if nY < MIN_OTHER * 0.5:
        os.remove(out_path)
        raise ValueError(f"parse Y={nY} 过少")
    if mask_ratio < MIN_MASK:
        os.remove(out_path)
        raise ValueError(f"mask {mask_ratio:.2f} < {MIN_MASK}")
    # 该链 native 序列（parse_PDB 顺序）
    S = d["S"].cpu().numpy().reshape(-1)
    from data_utils import restype_int_to_str
    seq = "".join(restype_int_to_str[int(a)] for a in S)
    return {"chain": chain, "name": name, "L": L, "nY": nY,
            "mask_ratio": round(mask_ratio, 3), "seq": seq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", default="data/ribosome_7k00/7K00.cif")
    ap.add_argument("--out", default="output/ribosome_7k00/pdbs")
    ap.add_argument("--summary", default="output/ribosome_7k00/split_summary.json")
    ap.add_argument("--only", default="", help="逗号分隔只拆指定 name（调试）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    atoms = parsePDB(args.complex)
    prot = atoms.select("protein")
    ca = prot.select("name CA")
    present = set(ca.getChids())
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    ok, fail = [], []
    t0 = time.time()
    # 按蛋白名排序
    items = sorted([(CHAIN_NAME[ch], ch) for ch in present if ch in CHAIN_NAME])
    for name, ch in items:
        if only and name not in only:
            continue
        out_path = os.path.join(args.out, f"7K00_{name}.pdb")
        if os.path.exists(out_path):
            ok.append({"name": name, "chain": ch, "skipped": True})
            continue
        try:
            r = split_one(atoms, ch, out_path)
            r["pdb"] = os.path.basename(out_path)
            ok.append(r)
            print(f"  ok 7K00_{name} chain={ch} L={r['L']} nY={r['nY']} mask={r['mask_ratio']}", flush=True)
        except Exception as e:
            fail.append({"name": name, "chain": ch, "err": str(e)})
            print(f"  fail 7K00_{name} chain={ch}: {e}", flush=True)
    out = {"complex": args.complex, "cutoff": CUTOFF, "min_len": MIN_LEN,
           "max_len": MAX_LEN, "ok": ok, "fail": fail,
           "elapsed_s": round(time.time() - t0, 1)}
    with open(args.summary, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"done: ok {len(ok)} fail {len(fail)} elapsed {time.time()-t0:.0f}s -> {args.summary}", flush=True)


if __name__ == "__main__":
    main()
