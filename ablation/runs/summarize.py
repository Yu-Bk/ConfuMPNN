"""Summarize ablation val-loss + probe results into compact relative-change tables.

Reads, per family:
  - ablation/runs/{prot,lig}/run_<tag>/val_loss.json   (val_loss_curve output, final epoch)
  - ablation/runs/{prot,lig}/run_<tag>/probe_*.json     (gen_probe output per representative)

Output: JSON with per-run loss fields (absolute + rel-change vs FULL) and probe H2/retention.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")

# final epoch per family
FINAL = {"prot": "10", "lig": "16"}
RUN_ORDER = {
    "prot": ["run_FULL", "run_nov12comp", "run_notarget", "run_noph", "run_nokeep"],
    "lig": ["run_FULL", "run_nov12comp", "run_notarget", "run_noA1", "run_noph", "run_nokeep"],
}
LOSS_KEYS = ["ce", "cd", "kl", "keep", "v12_comp", "v12_gravy", "v12_ct",
             "struct", "pocket", "total"]


def load_val_loss(run_dir, ep):
    f = run_dir / "val_loss.json"
    if not f.is_file():
        return None
    d = json.load(open(f))
    if str(ep) not in d.get("epochs", {}):
        return None
    return d["epochs"][str(ep)]


def rel(x, base):
    if base is None:
        return None
    if abs(base) < 1e-9:
        return None
    return (x - base) / abs(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", choices=["prot", "lig"], required=True)
    args = ap.parse_args()

    base_dir = ROOT / "ablation" / "runs" / args.fam
    ep = FINAL[args.fam]
    out = {"family": args.fam, "final_epoch": ep, "runs": {}}

    full = None
    for tag in RUN_ORDER[args.fam]:
        run_dir = base_dir / tag
        s = load_val_loss(run_dir, ep)
        if s is None:
            out["runs"][tag] = {"error": "no val_loss.json"}
            continue
        rec = {"abs": {k: round(s.get(k, float("nan")), 4) for k in LOSS_KEYS},
               "n_dom": s.get("n_dom"), "n_arm": s.get("n_arm")}
        out["runs"][tag] = {"loss": rec}
        if tag == "run_FULL":
            full = rec["abs"]
    if full:
        for tag in RUN_ORDER[args.fam]:
            rec = out["runs"][tag].get("loss")
            if not rec:
                continue
            rec["rel_vs_full"] = {
                k: (None if full.get(k) in (None, float("nan")) or abs(full.get(k, 0)) < 1e-9
                    else round(rel(rec["abs"].get(k), full.get(k)), 4))
                for k in LOSS_KEYS
            }
    # probes
    probes = {}
    for p in sorted(base_dir.glob("run_*/probe_*.json")):
        key = p.parent.name + "/" + p.stem
        try:
            probes[key] = json.load(open(p))
        except Exception:
            pass
    out["probes"] = probes

    op = ROOT / "ablation" / "report"
    op.mkdir(exist_ok=True)
    outfile = op / f"ablation_summary_{args.fam}.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {outfile}")

    # quick printed table
    print(f"\n=== {args.fam} val-loss (final epoch {ep}) — abs [rel vs FULL] ===")
    hdr = "run".ljust(13) + " " + " ".join(k.rjust(8) for k in
           ["ce", "cd", "total"])
    print(hdr)
    for tag in RUN_ORDER[args.fam]:
        rec = out["runs"][tag].get("loss")
        if not rec:
            continue
        row = tag.ljust(13)
        for k in ["ce", "cd", "total"]:
            a = rec["abs"].get(k)
            rv = rec["rel_vs_full"].get(k) if rec.get("rel_vs_full") else None
            cell = f"{a:.3f}" if a is not None else "  nan"
            if rv is not None:
                cell += f"[{rv:+.2f}]"
            row += cell.rjust(8 + 6)
        print(row)


if __name__ == "__main__":
    main()
