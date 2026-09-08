"""Exp1 postprocess: clean fasta headers, verify fixed positions, recompute net_charge,
copy clean fasta into data/exp1/<grp> for ESMFold/Tm/Sol scoring.

Usage: <env-confumpnn>/bin/python L11design/test/exp1_postprocess.py [grp ... | all]
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "LigandMPNN"))

from src.differentiable_charge import net_charge  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
OUTROOT = ROOT / "L11design/output/exp1"
DATAROOT = ROOT / "L11design/data/exp1"
MAN = json.load(open(ROOT / "L11design/test/exp1_groups.json"))
NATIVE = json.load(open(ROOT / "L11design/data/exp1/native.json"))
NATIVE_SEQ = NATIVE["L11design/input/L11.pdb"]["seq"]
FIXED = {int(k): v for k, v in NATIVE["L11design/input/L11.pdb"]["fixed"].items()}

GROUPS = list(MAN["groups"].keys())


def read_raw_fasta(fa):
    """run_guided output: header lines have metadata; seq follows."""
    names, seqs = [], []
    cur = None
    buf = []
    for line in open(fa):
        line = line.strip()
        if line.startswith(">"):
            if cur is not None:
                seqs.append("".join(buf))
            cur = line[1:].split()[0]  # first token = sample_N or native
            names.append(cur)
            buf = []
        elif line:
            buf.append(line)
    if cur is not None:
        seqs.append("".join(buf))
    return names, seqs


def process(grp):
    cfg = MAN["groups"][grp]
    pH = float(cfg["pH"])
    target = float(cfg["target"])
    gdir = OUTROOT / grp
    fa = gdir / "seqs.fa"
    summ = json.load(open(gdir / "summary.json"))
    names, seqs = read_raw_fasta(fa)
    assert len(seqs) == len(summ["sequences"]) + 1, f"{grp}: fasta count mismatch"
    meta_rows = []
    clean_lines = []
    n_sample = 0
    for nm, seq in zip(names, seqs):
        if nm == "native":
            continue
        n_sample += 1
        # verify fixed positions
        fixed_ok = all(seq[r - 1] == aa for r, aa in FIXED.items())
        fixed_dict = {r: seq[r - 1] for r in FIXED}
        q = net_charge(seq, pH)
        clean_lines.append(f">{nm}\n{seq}\n")
        meta_rows.append({
            "name": nm, "seq": seq,
            "charge": round(q, 3),
            "pI": summ["sequences"][n_sample - 1]["pI"],
            "fixed_ok": bool(fixed_ok),
            "fixed": fixed_dict,
        })
    # append native row
    q_nat = net_charge(NATIVE_SEQ, pH)
    clean_lines.append(f">native\n{NATIVE_SEQ}\n")
    meta_rows.append({"name": "native", "seq": NATIVE_SEQ,
                      "charge": round(q_nat, 3),
                      "fixed_ok": True,
                      "fixed": {r: NATIVE_SEQ[r - 1] for r in FIXED}})
    out_clean = gdir / "seqs_clean.fa"
    out_clean.write_text("".join(clean_lines))
    (gdir / "meta.json").write_text(json.dumps({
        "group": grp, "mode": cfg["mode"], "pH": pH, "target": target,
        "n_samples": n_sample, "native_charge": round(q_nat, 3),
        "n_fixed_mismatch": sum(1 for m in meta_rows if not m["fixed_ok"]),
        "rows": meta_rows,
    }, indent=2))
    # copy clean fasta into data/exp1/<grp>/
    ddir = DATAROOT / grp
    ddir.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_clean, ddir / "seqs.fa")
    mismatch = sum(1 for m in meta_rows if m["name"] != "native" and not m["fixed_ok"])
    print(f"[{grp}] n_sample={n_sample} fixed_mismatch={mismatch}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    for grp in GROUPS:
        if "all" in targets or grp in targets:
            process(grp)
    print("POSTPROCESS DONE")
