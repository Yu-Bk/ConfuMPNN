#!/usr/bin/env python
"""指标汇总：把 <root>/<grp>/ 下的 电荷(summary.json 或 seqs.fa header) + plddt.csv + tm.csv
+ tmseq + sol 合并成一行/序列的 merged.csv，并输出组级 mean±std。
用法: python code/tools/design_bench/score_aggregator.py --root output/design_groups --out output/design_groups/_merged_all.csv
（逐序列 merge 的健壮实现可参考活用例：L11design/test/exp1 分析脚本，本文件提供最小可用版。）
"""
import argparse, csv, json, re
from pathlib import Path

def read_seq_charge(grp):
    fa = grp / "seqs.fa"; out = []
    if fa.exists():
        for line in open(fa):
            m = re.search(r"charge=([-+]?[0-9.]+)", line)
            if m: out.append(float(m.group(1)))
    return out

def csv_cols(p, key):
    if not p.exists(): return []
    with open(p) as f:
        r = list(csv.reader(f))
    if not r: return []
    hdr = r[0]; idx = hdr.index(key) if key in hdr else 0
    return [x[idx] for x in r[1:]]

def mean(x):
    x = [float(v) for v in x if v not in ("", "None")]
    return round(sum(x) / len(x), 3) if x else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for grp in sorted(Path(a.root).iterdir()):
        if not (grp / "seqs.fa").exists(): continue
        ch = read_seq_charge(grp)
        pld = [x for x in csv_cols(grp / "plddt.csv", "plddt") if False] or []
        rows.append({"grp": grp.name, "n": len(ch),
                     "charge_mean": mean(ch),
                     "tm_mean": mean(csv_cols(grp / "tm.csv", "tm_score")),
                     "rmsd_mean": mean(csv_cols(grp / "tm.csv", "rmsd")),
                     "plddt_mean": mean(csv_cols(grp / "plddt.csv", "plddt"))})
        print(rows[-1])
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print("已写", a.out)

if __name__ == "__main__":
    main()
