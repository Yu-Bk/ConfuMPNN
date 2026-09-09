#!/usr/bin/env python
"""L11 Exp3/Exp4 postprocess：输出目录 summary → data 干净 fasta + 固定位校验 + 电荷重算。

Exp3（重做 Exp2，布局仿 exp2）：
    data/exp3/<mode>_q<t>/seqs.fa           仅 sample_i 头（干净）
    data/exp3/_native/seqs.fa               native 一条（对照）
Exp4（重做 Exp1，布局仿 exp1）：
    output/exp4/<grp>/seqs_clean.fa + meta.json
    data/exp4/<grp>/seqs.fa                 sample_i + native 行（末尾）

用法（confumpnn 环境）：python L11design/test/exp34_build_fastas.py [--exp exp3|exp4]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "LigandMPNN"))

from data_utils import parse_PDB, restype_int_to_str  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402

L11 = ROOT / "L11design"
FIXED_RESNUMS = [3, 5, 9, 34, 35, 89, 124, 131, 134, 135]


def native_seq():
    d, _, _, _, _ = parse_PDB(str(L11 / "input/L11.pdb"))
    S = d["S"].reshape(-1).cpu().numpy()
    return "".join(restype_int_to_str[i] for i in S)


def write_fasta(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for name, seq in records:
            f.write(f">{name}\n{seq}\n")


def build_exp3(cfg):
    nat = native_seq()
    write_fasta(L11 / "data/exp3/_native/seqs.fa", [("native", nat)])
    groups = cfg["groups"]
    n_written = 0
    for grp, g in groups.items():
        pH = float(g.get("pH", cfg.get("pH", 7.4)))
        outd = L11 / "output/exp3" / g["dir"]
        sj = outd / "summary.json"
        if not sj.is_file():
            print(f"  skip {grp}（无 summary.json）")
            continue
        s = json.load(open(sj))
        recs = []
        n_mis = 0
        for i, x in enumerate(s["sequences"], 1):
            seq = x["seq"]
            # 固定位校验（resnum r = index r-1，L11 链 I 1..141 连续）
            mis = [r for r in FIXED_RESNUMS if seq[r - 1] != nat[r - 1]]
            if mis:
                n_mis += 1
            recs.append((f"sample_{i}", seq))
        if not recs:
            print(f"  skip {grp}（0 序列）")
            continue
        outfa = L11 / "data/exp3" / g["dir"] / "seqs.fa"
        write_fasta(outfa, recs)
        n_written += len(recs)
        print(f"[exp3] {grp}: {len(recs)} 条 → {outfa}  fixed_mismatch={n_mis}")
    print(f"exp3 共写 {n_written} 条样本 fasta")


def build_exp4(cfg):
    nat = native_seq()
    for grp, g in cfg["groups"].items():
        pH = float(g["pH"])
        outd = L11 / "output/exp4" / g["dir"]
        dird = L11 / "data/exp4" / g["dir"]
        sj = outd / "summary.json"
        if not sj.is_file():
            print(f"  skip {grp}（无 summary.json）")
            continue
        s = json.load(open(sj))
        clean_lines = []
        rows = []
        n_mis = 0
        for i, x in enumerate(s["sequences"], 1):
            seq = x["seq"]
            mis = [r for r in FIXED_RESNUMS if seq[r - 1] != nat[r - 1]]
            if mis:
                n_mis += 1
            nm = f"sample_{i}"
            clean_lines.append(f">{nm}\n{seq}\n")
            rows.append({"name": nm, "seq": seq,
                         "charge": round(float(net_charge(seq, pH)), 3),
                         "pI": x["pI"],
                         "fixed_ok": not mis,
                         "fixed": {r: seq[r - 1] for r in FIXED_RESNUMS}})
        q_nat = float(net_charge(nat, pH))
        clean_lines.append(f">native\n{nat}\n")
        rows.append({"name": "native", "seq": nat,
                     "charge": round(q_nat, 3), "pI": None,
                     "fixed_ok": True,
                     "fixed": {r: nat[r - 1] for r in FIXED_RESNUMS}})
        (outd / "seqs_clean.fa").write_text("".join(clean_lines))
        (outd / "meta.json").write_text(json.dumps({
            "group": grp, "mode": g["mode"], "pH": pH, "target": float(g["target"]),
            "n_samples": len(s["sequences"]),
            "native_charge": round(q_nat, 3),
            "n_fixed_mismatch": n_mis,
            "rows": rows,
        }, indent=2))
        dird.mkdir(parents=True, exist_ok=True)
        shutil.copy(outd / "seqs_clean.fa", dird / "seqs.fa")
        print(f"[exp4] {grp}: {len(rows)-1} 条 sample + native → {dird / 'seqs.fa'}  "
              f"fixed_mismatch={n_mis}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", "exp3", "exp4"])
    args = ap.parse_args()
    if args.exp in ("all", "exp3"):
        build_exp3(json.load(open(L11 / "test/exp3_groups.json")))
    if args.exp in ("all", "exp4"):
        build_exp4(json.load(open(L11 / "test/exp4_groups.json")))
    print("BUILD FASTAS DONE")


if __name__ == "__main__":
    main()
