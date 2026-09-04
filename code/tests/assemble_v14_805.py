"""阶段 2：解析 ext_smallmol_raw + rna_pdbs_ext → 大 pass 池 JSON（轨 B，2026-09-04）。

用途：把所有本地已下载/拆链的候选统一 parse QC、按训练语义分类、序列去重、coverage 判定，
输出一个"可入池"大表 JSON，供后续配额抽样脚本选满 805。

QC/分类口径（与建训练集一致）：
  - 单蛋白链（prody 判定，>1 链剔除）；parse_PDB L∈[50,500]；Y(非蛋白配体)原子>0；标准 20aa 序列。
  - 类型（ext_smallmol_raw 内，按 HET 配体）：
      nucleotide（核苷酸辅因子：RNA/DNA 单核苷酸 + NAD/NADP/FAD/FMN 等，含其磷酸化形式）
      metal（仅金属离子非蛋白配体）
      small_mol（其余有机小分子）
    注：rna_pdbs_ext 全部归 RNA/DNA（真核酸结合，本地拆链）。
  - 去重：seq 精确不在 labels_v14_final(5371) 与 in-10 测试 + 1A65 序列内；池内按 code 去重。
  - coverage：相对 labels_v14_final，用 coverage_check 口径（|ΔL|≤max(0.15L,40) & |Δq|≤4；in≥100/boundary30-99/out<30）。

用法（项目根，CPU，可并行）：
  PYTHONPATH=code ~/miniconda3/envs/confumpnn/bin/python code/tests/assemble_v14_805.py --raw data/ligand_train/ext_smallmol_raw --out /tmp/v14_pass_pool.json [--nproc 8]
输出：/tmp/v14_pass_pool.json（list[dict]）
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (_CODE_DIR, os.path.join(_CODE_DIR, "..", "..", "LigandMPNN")):
    if p not in sys.path:
        sys.path.insert(0, p)
from data_utils import parse_PDB  # noqa: E402
from prody import parsePDB  # noqa: E402
from run_guided import seq_to_string  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402

NUC = {  # 核苷酸辅因子（含核糖/脱氧单核苷酸 + 常见氧化型）
    "AMP","ADP","ATP","GMP","GDP","GTP","CMP","CDP","CTP","UMP","UDP","UTP",
    "TMP","TDP","TTP","IMP","IDP","ITP","XMP","XDP","XTP","GNP","ANP","ACP",
    "NAD","NAI","NADP","NDP","NAP","FAD","FMN","FMD","FMN","NMN","R5P","SAH",
    "SAM","COA","ACO","BCR","1MA","1MG","2MG","5MC","7MG","M2G","OMC","OMG",
    "PSU","H2U","5BU","2BU","4OC","URA","ADE","GUA","CYT","RGU","RCY",
    "DA","DC","DG","DT","DU","DI","DAN","DCN","DGN","DTN","A","C","G","T","U",
}
METAL = {"ZN","MG","CA","FE","CU","MN","CO","NI","NA","K","CD","HG","PB","MO",
         "W","LI","SR","BA","AL","AG","AU","PT","V","CR","SE","RB","CS","GA",
         "GE","SB","BI","SN","CE","PR","ND","SM","EU","GD","TB","DY","HO","ER",
         "TM","YB","LU","RU","RH","PD","OS","IR","FE2","CU1","MN3","MN2","ZN2"}
HOH = {"HOH","WAT","DOD"}


def parse_one(path):
    code = os.path.basename(path)[:-4]
    try:
        atoms = parsePDB(path)
        ca = atoms.select("protein and name CA")
        if ca is None:
            return {"id": code, "err": "no protein"}
        ch = set(ca.getChids())
        if len(ch) != 1:
            return {"id": code, "err": "n_chain=%d" % len(ch)}
        pd, *_ = parse_PDB(path, device="cpu", parse_all_atoms=False)
        L = int(pd["X"].shape[0])
        seq = seq_to_string(pd["S"].reshape(-1).cpu().numpy())
        if not (50 <= L <= 500):
            return {"id": code, "err": "L=%d" % L}
        Y = pd.get("Y")
        ny = Y.numel() // 3 if Y is not None else 0
        if ny <= 0:
            return {"id": code, "err": "noY"}
        het = set()
        for line in open(path):
            if line.startswith("HETATM"):
                r = line[17:20].strip()
                if r and r not in HOH:
                    het.add(r)
        return {"id": code, "L": int(L), "seq": seq, "ny": int(ny),
                "het": sorted(het)[:12], "q": round(float(net_charge(seq, 7.4)), 2)}
    except Exception as e:
        return {"id": code, "err": str(e)[:80]}


def classify(het):
    nuc = [r for r in het if r in NUC]
    met = [r for r in het if r in METAL]
    org = [r for r in het if r not in NUC and r not in METAL]
    if nuc:
        return "nucleotide"
    if met and not org:
        return "metal"
    return "small_mol"


def worker(path):
    r = parse_one(path)
    if "err" not in r:
        r["type"] = classify(r["het"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="/tmp/v14_pass_pool.json")
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    files = [os.path.join(a.raw, f) for f in os.listdir(a.raw) if f.endswith(".pdb")]
    print("files to parse:", len(files), flush=True)
    with Pool(a.nproc) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(worker, files, chunksize=20)):
            res.append(r)
            if (i + 1) % 400 == 0:
                print("  parsed", i + 1, flush=True)
    ok = [r for r in res if "err" not in r]
    print("parsed ok:", len(ok), "/", len(res), flush=True)
    # coverage vs train
    d = np.load("data/ligand_train/labels_v14_final.npz", allow_pickle=True)
    seqs = d["seqs"]
    Ltr = np.array([len(s) for s in seqs])
    qtr = np.array([net_charge(s, 7.4) for s in seqs])
    tr_seqs = set(str(s) for s in seqs)
    # test seqs (in-10 + 1A65 + 2E9R archived)
    def fchain(p):
        try:
            pd, *_ = parse_PDB(p, device="cpu", parse_all_atoms=False)
            return seq_to_string(pd["S"].reshape(-1).cpu().numpy())
        except Exception:
            return None
    test_seqs = set()
    for mf in ["6D2O","1AS2","2FEO","5CQH","1CGE","1BJ4","21KL_A","5O60_E",
               "3MXB_A","9DWG_L","1A65","2E9R_X"]:
        s = fchain("data/validation_pdbs/" + mf + ".pdb")
        if s:
            test_seqs.add(s)
    out = []
    seen_seq = set()
    for r in ok:
        if r["seq"] in tr_seqs or r["seq"] in test_seqs:
            continue
        if r["seq"] in seen_seq:   # 池内 code 间去重（同序列只留首个）
            continue
        seen_seq.add(r["seq"])
        Lp, qp = r["L"], r["q"]
        tol = max(0.15 * Lp, 40)
        nc = int(sum((Ltr >= max(20, Lp - tol)) & (Ltr <= Lp + tol) & (qtr >= qp - 4) & (qtr <= qp + 4)))
        cov = "in" if nc >= 100 else ("boundary" if nc >= 30 else "out")
        r["n_close"] = nc
        r["coverage"] = cov
        out.append(r)
    json.dump(out, open(a.out, "w"))
    from collections import Counter
    print("final pass pool:", len(out))
    print("by type:", dict(Counter(r["type"] for r in out)))
    print("by coverage:", dict(Counter(r["coverage"] for r in out)))


if __name__ == "__main__":
    main()
