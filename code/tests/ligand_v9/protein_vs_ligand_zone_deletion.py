"""Task A — protein-mode vs ligand-mode native-arm charged-residue deletion, zone-level.

CPU, zero new sampling. Reuses structure code path of deletion_location_analysis.py
(pocket = Ca-ligand(Y, incl. nucleic acid)<=8A; surface = frac_sasa>=0.25; core = rest;
charged = DEKR; retention = mean gen count / native count).

Reads native sequence from the *reference PDB parse* (protein-mode seqs.fa has no
">native" line) so protein and ligand are compared under identical geometry code.

Usage:
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python \
    code/tests/ligand_v9/protein_vs_ligand_zone_deletion.py

Output: output/protein_vs_ligand_zone_deletion.json (+ printed table)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))
sys.path.insert(0, str(_PROJECT_DIR / "code"))
sys.path.insert(0, str(_PROJECT_DIR / "code" / "tests" / "ligand_v9"))

from deletion_location_analysis import (  # noqa: E402
    read_fa, structure_features, ACIDIC, BASIC, CHARGED,
)

PDB_DIR = _PROJECT_DIR / "data" / "validation_pdbs"
PROTEIN_ROOT = _PROJECT_DIR / "output" / "generalization_v12_2_calib_small" / "protein"
LIGAND_V14_ROOT = _PROJECT_DIR / "output" / "generalization_ligand_v14_clean" / "ligand"
LIGAND_V13_ROOT = _PROJECT_DIR / "output" / "generalization_ligand_v13_in10" / "ligand"
LIGAND_V122_ROOT = _PROJECT_DIR / "output" / "generalization_ligand_v12_2" / "ligand"
OUT = _PROJECT_DIR / "output" / "protein_vs_ligand_zone_deletion.json"


def zone_retention(gen_seqs, native, st):
    """Per-zone + all-sequence DE/KR/DEKR retention.

    gen_seqs: list of same-length generated sequences (native arm).
    native: reference (backbone-consistent) sequence.
    st: dict from structure_features (pocket bool, frac_sasa array).
    Returns dict keyed by zone with counts & retention.
    """
    L = len(native)
    frac = np.nan_to_num(st["frac_sasa"], nan=0.0)
    pocket = st["pocket"]
    surface = frac >= 0.25
    zone = np.where(pocket, "pocket", np.where(surface, "surface", "core"))
    idx = np.arange(L)

    # gen per-residue frequency over the generated set
    de_mat = np.array([[s[i] in ACIDIC for i in range(L)] for s in gen_seqs], dtype=np.float64)
    kr_mat = np.array([[s[i] in BASIC for i in range(L)] for s in gen_seqs], dtype=np.float64)
    chg_mat = np.array([[s[i] in CHARGED for i in range(L)] for s in gen_seqs], dtype=np.float64)
    f_de = de_mat.mean(axis=0)
    f_kr = kr_mat.mean(axis=0)
    f_chg = chg_mat.mean(axis=0)

    def _count(sub):
        sub = np.asarray(sub)
        nat_de = int(sum(1 for i in idx[sub] if native[i] in ACIDIC))
        nat_kr = int(sum(1 for i in idx[sub] if native[i] in BASIC))
        nat_ch = nat_de + nat_kr
        g_de = float(f_de[sub].sum())
        g_kr = float(f_kr[sub].sum())
        g_ch = float(f_chg[sub].sum())
        return {
            "n_res": int(sub.sum()),
            "nat_DE": nat_de, "nat_KR": nat_kr, "nat_CHG": nat_ch,
            "gen_DE": round(g_de, 3), "gen_KR": round(g_kr, 3), "gen_CHG": round(g_ch, 3),
            "ret_DE": (round(g_de / nat_de, 3) if nat_de else None),
            "ret_KR": (round(g_kr / nat_kr, 3) if nat_kr else None),
            "ret_CHG": (round(g_ch / nat_ch, 3) if nat_ch else None),
        }

    out = {
        "pocket": _count(zone == "pocket"),
        "surface": _count(zone == "surface"),
        "core": _count(zone == "core"),
        "all": _count(np.ones(L, dtype=bool)),
        "n_gen": len(gen_seqs),
    }
    return out


def load_native_from_pdb(pdb):
    st = structure_features(PDB_DIR / f"{pdb}.pdb")
    return st, st["seq"]


def gen_from_fa(fa):
    """Read generated seqs only (skip any >native line if present)."""
    if not Path(fa).exists():
        return None
    gen, native, _ = read_fa(fa)
    return gen


def analyze_root(pdb, root, native_src="pdb"):
    fa = Path(root) / pdb / "pH7.4" / "arm_native" / "seqs.fa"
    if not Path(fa).exists():
        return {"pdb": pdb, "error": f"missing {fa}"}
    st = structure_features(PDB_DIR / f"{pdb}.pdb")
    if native_src == "pdb":
        native = st["seq"]
    else:
        gen, native, _ = read_fa(fa)
    gen = gen_from_fa(fa)
    if not gen:
        return {"pdb": pdb, "error": "no gen seqs"}
    gen = [s for s in gen if len(s) == len(native)]
    if len(gen) == 0:
        return {"pdb": pdb, "error": "no length-matched gen seqs"}
    return {"pdb": pdb, "L": len(native), **zone_retention(gen, native, st)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=str(OUT))
    args = ap.parse_args()

    # protein-mode (v12.2 MoMPNN, small batch) proteins present on disk
    prot_pdbs = sorted(p.name for p in Path(PROTEIN_ROOT).iterdir() if p.is_dir())
    ligand14_pdbs = ["6D2O", "1AS2", "2FEO", "5CQH", "1CGE", "1BJ4",
                     "21KL_A", "5O60_E", "3MXB_A", "9DWG_L"]

    print("==== protein mode (v12.2 MoMPNN small-batch) — native arm, zones ====")
    prot = {}
    for pdb in prot_pdbs:
        r = analyze_root(pdb, PROTEIN_ROOT, native_src="pdb")
        prot[pdb] = r
        if "error" in r:
            print(f"  !! {pdb}: {r['error']}")
            continue
        a = r["all"]; pk = r["pocket"]; sf = r["surface"]; co = r["core"]
        print(f"  {pdb:6s} L={r['L']:4d} n={r['n_gen']:3d} "
              f"| all CHG ret={a['ret_CHG']} (DE {a['ret_DE']}/KR {a['ret_KR']}) "
              f"| pocket {pk['ret_CHG']} surf {sf['ret_CHG']} core {co['ret_CHG']} "
              f"(natCHG pocket {pk['nat_CHG']}/surf {sf['nat_CHG']}/core {co['nat_CHG']})")

    print("\n==== ligand v14 clean — native arm, zones (recompute, sanity) ====")
    lig14 = {}
    for pdb in ligand14_pdbs:
        if not (Path(LIGAND_V14_ROOT) / pdb / "pH7.4" / "arm_native" / "seqs.fa").exists():
            continue
        r = analyze_root(pdb, LIGAND_V14_ROOT, native_src="fasta")
        lig14[pdb] = r
        if "error" in r:
            print(f"  !! {pdb}: {r['error']}")
            continue
        a = r["all"]; pk = r["pocket"]; sf = r["surface"]; co = r["core"]
        print(f"  {pdb:6s} L={r['L']:4d} n={r['n_gen']:3d} "
              f"| all CHG ret={a['ret_CHG']} (DE {a['ret_DE']}/KR {a['ret_KR']}) "
              f"| pocket {pk['ret_CHG']} surf {sf['ret_CHG']} core {co['ret_CHG']} "
              f"(natCHG pocket {pk['nat_CHG']}/surf {sf['nat_CHG']}/core {co['nat_CHG']})")

    print("\n==== ligand v13 in-10 — native arm, zones (recompute, sanity) ====")
    lig13 = {}
    for pdb in ["1AS2", "2FEO", "5CQH", "1CGE", "1BJ4", "21KL_A", "5O60_E", "3MXB_A", "9DWG_L", "6D2O"]:
        if not (Path(LIGAND_V13_ROOT) / pdb / "pH7.4" / "arm_native" / "seqs.fa").exists():
            continue
        r = analyze_root(pdb, LIGAND_V13_ROOT, native_src="fasta")
        lig13[pdb] = r
        if "error" in r:
            print(f"  !! {pdb}: {r['error']}")
            continue
        a = r["all"]; pk = r["pocket"]; sf = r["surface"]; co = r["core"]
        print(f"  {pdb:6s} L={r['L']:4d} n={r['n_gen']:3d} "
              f"| all CHG ret={a['ret_CHG']} (DE {a['ret_DE']}/KR {a['ret_KR']}) "
              f"| pocket {pk['ret_CHG']} surf {sf['ret_CHG']} core {co['ret_CHG']} "
              f"(natCHG pocket {pk['nat_CHG']}/surf {sf['nat_CHG']}/core {co['nat_CHG']})")

    print("\n==== ligand v12.2 (first ligand migration) — native arm, zones ====")
    lig122 = {}
    for pdb in ["1C6O", "1AZM", "1AS2", "1AXW", "2FEO", "5CQH", "1CGE", "1AG0", "1A65", "1BJ4"]:
        if not (Path(LIGAND_V122_ROOT) / pdb / "pH7.4" / "arm_native" / "seqs.fa").exists():
            continue
        r = analyze_root(pdb, LIGAND_V122_ROOT, native_src="fasta")
        lig122[pdb] = r
        if "error" in r:
            print(f"  !! {pdb}: {r['error']}")
            continue
        a = r["all"]; pk = r["pocket"]; sf = r["surface"]; co = r["core"]
        print(f"  {pdb:6s} L={r['L']:4d} n={r['n_gen']:3d} "
              f"| all CHG ret={a['ret_CHG']} (DE {a['ret_DE']}/KR {a['ret_KR']}) "
              f"| pocket {pk['ret_CHG']} surf {sf['ret_CHG']} core {co['ret_CHG']} "
              f"(natCHG pocket {pk['nat_CHG']}/surf {sf['nat_CHG']}/core {co['nat_CHG']})")

    json.dump({"protein_v122": prot, "ligand_v14": lig14,
               "ligand_v13": lig13, "ligand_v122": lig122},
              open(args.out_json, "w"), indent=1, ensure_ascii=False)
    print(f"\nJSON -> {args.out_json}")


if __name__ == "__main__":
    main()
