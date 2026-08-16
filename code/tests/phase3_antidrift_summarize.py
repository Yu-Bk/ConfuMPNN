"""Phase 3 防失控判据汇总：条件注入序列 vs E1b MoMPNN 基线（pLDDT/TM/%sol/Tm）。

数据源：
  条件注入（Phase 3）：code/output/phase3_antidrift/{pdb}/{plddt,tm,seqs.fa-protein_sol,seqs.fa.tm}.csv
  基线（E1b）：code/output/e1_ext/summary_cond.csv 的 mompnn,baseline 行
用法（code/ 下）：
  python tests/phase3_antidrift_summarize.py
输出：CSV（code/output/phase3_antidrift/comparison.csv）+ 打印表格
"""
import csv
import glob
import os
import sys
from pathlib import Path

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
ANTI = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "code/output/phase3_antidrift")

# E1b MoMPNN 基线（summary_cond.csv mompnn,baseline 行）
BASELINE = {
    "1BC8": dict(plddt=82.824, tm=0.915, sol=81.424, temberture=64.818),
    "1CRN": dict(plddt=89.140, tm=0.905, sol=84.106, temberture=59.114),
    "1UBQ": dict(plddt=89.356, tm=0.962, sol=85.987, temberture=68.612),
    "2LZM": dict(plddt=88.466, tm=0.971, sol=80.145, temberture=67.708),
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def mean(xs):
    xs = [float(x) for x in xs if x]
    return sum(xs) / len(xs) if xs else float("nan")


def per_pdb(pdb):
    d = ANTI / pdb
    plddt = [r["mean_plddt"] for r in read_csv(d / "plddt.csv")
             if not r["name"].startswith("native")]
    tm = [r["tm_score"] for r in read_csv(d / "tm.csv")
          if not r["name"].startswith("native")]
    sol = [r["percent-sol"] for r in read_csv(d / "seqs.fa-protein_sol.csv")
           if not r["ID"].startswith("native")]
    temp = [r["mean_tm"] for r in read_csv(d / "seqs.fa.tm.csv")
            if not r["name"].startswith("native")]
    return {
        "plddt": (mean(plddt), len(plddt)),
        "tm": (mean(tm), len(tm)),
        "sol": (mean(sol), len(sol)),
        "temberture": (mean(temp), len(temp)),
    }


def main():
    rows = []
    for pdb in ["1BC8", "1CRN", "1UBQ", "2LZM"]:
        got = per_pdb(pdb)
        base = BASELINE[pdb]
        row = {"pdb": pdb}
        verdicts = []
        for metric in ["plddt", "tm", "sol", "temberture"]:
            v, n = got[metric]
            b = base[metric]
            diff = v - b
            # 防失控判据：pLDDT/TM 掉 >1.0 或 %sol/Tm 掉 >2.0 → 警惕
            thresh = {"plddt": 1.0, "tm": 0.01, "sol": 2.0, "temberture": 2.0}[metric]
            bad = (metric in ("plddt", "tm")) and diff < -thresh
            bad = bad or (metric in ("sol", "temberture") and diff < -thresh)
            verdicts.append("DANGER" if bad else ("ok" if diff >= 0 else "mild"))
            row[f"{metric}_cond"] = round(v, 3)
            row[f"{metric}_base"] = b
            row[f"{metric}_diff"] = round(diff, 3)
            row[f"{metric}_n"] = n
        row["verdict"] = "DANGER" if "DANGER" in verdicts else "PASS"
        rows.append(row)

    out = ANTI / "comparison.csv"
    fields = ["pdb", "plddt_cond", "plddt_base", "plddt_diff", "plddt_n",
              "tm_cond", "tm_base", "tm_diff", "tm_n",
              "sol_cond", "sol_base", "sol_diff", "sol_n",
              "temberture_cond", "temberture_base", "temberture_diff", "temberture_n",
              "verdict"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 打印
    print(f"{'PDB':5} {'pLDDT(cond/base/diff)':26} {'TM(cond/base/diff)':26} "
          f"{'%sol(cond/base/diff)':26} {'Tm(cond/base/diff)':26}  判定")
    for r in rows:
        print(f"{r['pdb']:5} {r['plddt_cond']:7.2f}/{r['plddt_base']:5.2f}/{r['plddt_diff']:+6.2f}"
              f"  {r['tm_cond']:.3f}/{r['tm_base']:.3f}/{r['tm_diff']:+.3f}"
              f"  {r['sol_cond']:6.1f}/{r['sol_base']:5.1f}/{r['sol_diff']:+5.1f}"
              f"  {r['temberture_cond']:6.1f}/{r['temberture_base']:5.1f}/{r['temberture_diff']:+5.1f}"
              f"  {r['verdict']}")
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
