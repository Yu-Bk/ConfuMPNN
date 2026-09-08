#!/usr/bin/env python
"""L11 Exp2：从 run_guided 的 summary.json 生成干净 seqs.fa（data/exp2/<dir>/seqs.fa）。

- 每组一条干净 fasta：header = sample_i, sequence 与 summary 完全一致。
- 另生成 data/exp2/_native/seqs.fa（native 对照，仅一次；供 ESMFold/Tm/Sol 用）。
用法：
    python L11design/test/exp2_build_fastas.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/data/nfs/IC/baokun_yu/ConfuMPNN/code")
sys.path.insert(0, "/data/nfs/IC/baokun_yu/ConfuMPNN/LigandMPNN")
from data_utils import parse_PDB, restype_int_to_str  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402

L11 = Path("/data/nfs/IC/baokun_yu/ConfuMPNN/L11design")
FIXED = [3, 5, 9, 34, 35, 89, 124, 131, 134, 135]

MODES = ["protein", "ligand"]
TS = ["6.87", "8", "10", "12"]


def native_seq():
    protein_dict, _, _, _, _ = parse_PDB(str(L11 / "input/L11.pdb"))
    S = protein_dict["S"]
    return "".join(restype_int_to_str[i] for i in S.tolist())


def write_fasta(path, records):
    """records: list of (name, seq, charge, pI). fasta header 只用 name（干净，便于跨 csv 合并）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name, seq, charge, pI in records:
            f.write(f">{name}\n{seq}\n")


def main():
    nseq = native_seq()
    # native 对照 fasta（只有一次）
    write_fasta(L11 / "data/exp2/_native/seqs.fa",
                [("native", nseq, net_charge(nseq, 7.4), 10.10)])
    n_written = 0
    for mode in MODES:
        for t in TS:
            d = L11 / "output/exp2" / f"{mode}_q{t}"
            sj = d / "summary.json"
            if not sj.is_file():
                print(f"skip {mode}_q{t} (无 summary.json)")
                continue
            s = json.load(open(sj))
            recs = []
            for i, x in enumerate(s["sequences"], 1):
                recs.append((f"sample_{i}", x["seq"], x["charge"], x["pI"]))
            if len(recs) == 0:
                print(f"skip {mode}_q{t} (0 序列)")
                continue
            outfa = L11 / "data/exp2" / f"{mode}_q{t}" / "seqs.fa"
            write_fasta(outfa, recs)
            print(f"{mode}_q{t}: {len(recs)} 条 → {outfa}")
            n_written += len(recs)
    print(f"共写 {n_written} 条样本序列 fasta")


if __name__ == "__main__":
    main()
