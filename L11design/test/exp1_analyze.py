"""Exp1 analysis: merge per-group metrics -> data/exp1/<grp>/merged.csv; stats -> summary_stats.json;
print 8 stats incl mode effect (protein vs ligand at same condition) and pH effect (within mode).

Run in confumpnn env (has scipy).
Usage: <confumpnn>/bin/python L11design/test/exp1_analyze.py
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
OUTROOT = ROOT / "L11design/output/exp1"
DATAROOT = ROOT / "L11design/data/exp1"
MAN = json.load(open(ROOT / "L11design/test/exp1_groups.json"))
NATIVE = json.load(open(ROOT / "L11design/data/exp1/native.json"))
NATIVE_SEQ = NATIVE["L11design/input/L11.pdb"]["seq"]
GROUPS = list(MAN["groups"].keys())
NAT_LEN = len(NATIVE_SEQ)
SOL_RE = re.compile(r"SEQUENCE PREDICTIONS,>\S+?,([-\d.]+),")

PAIRS_MODE = [("prot_C1", "lig_C1"), ("prot_C2", "lig_C2")]
PAIRS_PH = [("prot_C1", "prot_C2"), ("lig_C1", "lig_C2")]
METRICS = ["charge", "plddt", "tm_score", "rmsd", "mean_tm", "percent_sol", "recovery"]


def read_plddt(csvp):
    d = {}
    if not csvp.exists():
        return d
    for line in open(csvp):
        parts = line.rstrip("\n").split(",")
        if len(parts) >= 3 and parts[0] != "name":
            try:
                d[parts[0]] = float(parts[2])
            except ValueError:
                pass
    return d


def read_tm(csvp):
    d = {}
    if not csvp.exists():
        return d
    for line in open(csvp):
        parts = line.rstrip("\n").split(",")
        if len(parts) >= 3 and parts[0] != "name":
            try:
                d[parts[0]] = (float(parts[1]) if parts[1] else None,
                               float(parts[2]) if parts[2] else None)
            except ValueError:
                pass
    return d


def read_tm_temberture(csvp):
    d = {}
    if not csvp.exists():
        return d
    for line in open(csvp):
        parts = line.rstrip("\n").split(",")
        if len(parts) >= 3 and parts[0] != "name":
            try:
                d[parts[0]] = float(parts[2])
            except ValueError:
                pass
    return d


def read_sol(txtp):
    d = {}
    if not txtp.exists():
        return d
    for line in open(txtp):
        m = SOL_RE.search(line)
        if m:
            name = line.split(",")[1].lstrip(">").strip()
            d[name] = float(m.group(1))
    return d


def identity(seq):
    return sum(1 for a, b in zip(seq, NATIVE_SEQ) if a == b) / NAT_LEN * 100.0


def load_group(grp):
    cfg = MAN["groups"][grp]
    d = DATAROOT / grp
    meta = json.load(open(OUTROOT / grp / "meta.json"))
    plddt = read_plddt(d / "plddt.csv")
    tm = read_tm(d / "tm.csv")
    tmt = read_tm_temberture(d / "seqs.fa.tm.csv")
    sol = read_sol(Path(str(d / "seqs.fa") + "-protein_sol_prediction.txt"))
    rows = []
    for m in meta["rows"]:
        nm = m["name"]
        if nm == "native":
            continue
        r = {
            "name": nm,
            "mode": cfg["mode"],
            "grp": grp,
            "charge": m["charge"],
            "pI": m.get("pI"),
            "fixed_ok": m["fixed_ok"],
            "recovery": round(identity(m["seq"]), 2),
            "plddt": plddt.get(nm),
            "tm_score": tm.get(nm, (None, None))[0],
            "rmsd": tm.get(nm, (None, None))[1],
            "mean_tm": tmt.get(nm),
            "percent_sol": sol.get(nm),
        }
        rows.append(r)
    return rows


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.array(vals, dtype=float)
    return {"mean": round(float(a.mean()), 3), "std": round(float(a.std(ddof=0)), 3),
            "n": len(a), "median": round(float(np.median(a)), 3)}


def fmt(v):
    if v is None:
        return "-"
    return f"{v['mean']:.2f}±{v['std']:.2f}"


def load_raw_charge(grp):
    """Raw (no-calibration) diagnostic run charge distribution from rawdiag summary."""
    p = OUTROOT / "rawdiag" / grp / "summary.json"
    if not p.exists():
        return None
    s = json.load(open(p))
    ch = [x["charge"] for x in s["sequences"]]
    a = np.array(ch, dtype=float)
    return {"mean": round(float(a.mean()), 3), "std": round(float(a.std(ddof=0)), 3),
            "n": len(a), "min": round(float(a.min()), 2), "max": round(float(a.max()), 2)}


def main():
    group_rows = {}
    for grp in GROUPS:
        group_rows[grp] = load_group(grp)
        # write merged.csv
        with open(DATAROOT / grp / "merged.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "mode", "grp", "charge", "pI",
                                              "fixed_ok", "recovery", "plddt", "tm_score",
                                              "rmsd", "mean_tm", "percent_sol"])
            w.writeheader()
            for r in group_rows[grp]:
                w.writerow(r)
        print(f"[{grp}] rows={len(group_rows[grp])}")

    summary = {"groups": {}}
    print("\n=== mean±std (samples only) ===")
    hdr = f"{'grp':9s} " + " ".join(f"{m:>11s}" for m in METRICS) + f"{'rawChg':>11s}"
    print(hdr)
    for grp in GROUPS:
        rs = group_rows[grp]
        st = {}
        for m in METRICS:
            key = {"charge": "charge", "plddt": "plddt", "tm_score": "tm_score",
                   "rmsd": "rmsd", "mean_tm": "mean_tm",
                   "percent_sol": "percent_sol", "recovery": "recovery"}[m]
            st[m] = stats([r[key] for r in rs])
        rc = load_raw_charge(grp)
        st["raw_charge"] = rc
        summary["groups"][grp] = st
        rcs = f"{rc['mean']:.2f}±{rc['std']:.2f}" if rc else "-"
        line = f"{grp:9s} " + " ".join(f"{fmt(st[m]):>11s}" for m in METRICS) + f"{rcs:>11s}"
        print(line)

    # native reference metrics (from whichever group's native row, same sequence)
    native_metrics = {}
    # pLDDT/TM/Tm/Sol for native come from the native row inside each group's files; gather from prot_C1
    grp = "prot_C1"
    meta = json.load(open(OUTROOT / grp / "meta.json"))
    # native row present in merged? we skipped it; fetch directly
    d = DATAROOT / grp
    plddt = read_plddt(d / "plddt.csv"); tm = read_tm(d / "tm.csv")
    tmt = read_tm_temberture(d / "seqs.fa.tm.csv"); sol = read_sol(Path(str(d / "seqs.fa") + "-protein_sol_prediction.txt"))
    native_metrics = {
        "charge": meta["rows"][-1]["charge"],
        "plddt": plddt.get("native"), "tm_score": tm.get("native", (None, None))[0],
        "rmsd": tm.get("native", (None, None))[1],
        "mean_tm": tmt.get("native"), "percent_sol": sol.get("native"),
    }
    summary["native"] = native_metrics
    print("\n=== native reference (L11 chain I) ===")
    print(native_metrics)

    # Wilcoxon pairwise
    print("\n=== Pairwise Mann-Whitney U (two-sided) ===")
    pw = {}
    for label, (a, b) in [("mode_C1", PAIRS_MODE[0]), ("mode_C2", PAIRS_MODE[1]),
                          ("pH_prot", PAIRS_PH[0]), ("pH_lig", PAIRS_PH[1])]:
        pw[label] = {"a": a, "b": b, "metrics": {}}
        ra, rb = group_rows[a], group_rows[b]
        for m in METRICS:
            va = [r[m] for r in ra if r[m] is not None]
            vb = [r[m] for r in rb if r[m] is not None]
            if len(va) >= 3 and len(vb) >= 3:
                u, p = mannwhitneyu(va, vb, alternative="two-sided")
                pw[label]["metrics"][m] = {"mean_a": float(np.mean(va)), "mean_b": float(np.mean(vb)),
                                           "p": float(p)}
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                print(f"  {label:9s} {m:>11s} mean {np.mean(va):7.2f} vs {np.mean(vb):7.2f}  U={u:8.1f} p={p:.4f} {sig}")
            else:
                print(f"  {label:9s} {m:>11s} n too small ({len(va)},{len(vb)})")
    summary["pairwise"] = pw

    with open(DATAROOT / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nwrote data/exp1/summary_stats.json")


if __name__ == "__main__":
    main()
