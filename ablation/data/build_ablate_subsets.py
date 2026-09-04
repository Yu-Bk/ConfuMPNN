"""Build stratified 25% training subsets for controlled ablation (prot + lig).

Prot source: data/cath/labels_v12_3_train.npz  (6580 dom, protein MoMPNN v12.2 recipe)
Lig  source: data/ligand_train/labels_v14_final.npz (5371 dom, LigandMPNN v14 recipe)

Stratification:
  - ligand: hard stratify RNA/DNA flag (must preserve ~7.8%); all: L-bin x charge-quantile-bin
  - protein: L-bin x charge-quantile-bin (long L tails preserved via dedicated L bin)
Output npz keeps original domain order & schema (domain_ids/seqs/coords/pH/charge/pI).
Also writes a post-hoc balance report (JSON + printed) comparing subset vs full.
"""
import json
import os
from collections import Counter

import numpy as np

ROOT = "/data/nfs/IC/baokun_yu/ConfuMPNN"
FRAC = 0.25
SEED = 42

PROT_SRC = f"{ROOT}/data/cath/labels_v12_3_train.npz"
PROT_DOMPDB = f"{ROOT}/data/cath/S40/dompdb"
PROT_OUT = f"{ROOT}/ablation/data/labels_ablate_prot.npz"
LIG_SRC = f"{ROOT}/data/ligand_train/labels_v14_final.npz"
LIG_DOMPDB = f"{ROOT}/data/ligand_train/all_pdb"
LIG_OUT = f"{ROOT}/ablation/data/labels_ablate_lig.npz"

# RNA/DNA candidate dirs under data/ligand_train (used to tag each lig domain)
LIG_RNA_DNA_DIRS = ["rna_pdbs", "rna_pdbs_ext", "rna_complex_raw", "dna"]


def load(src):
    d = np.load(src, allow_pickle=True)
    ids = [str(x) for x in d["domain_ids"]]
    n = len(ids)
    return {
        "domain_ids": np.array(ids),
        "seqs": d["seqs"],
        "coords": d["coords"],
        "pH": d["pH"].reshape(n, 8),
        "charge": d["charge"].reshape(n, 8),
        "pI": d["pI"].reshape(n, 8),
    }


def dom_L(seqs):
    out = np.zeros(len(seqs), dtype=int)
    for i, s in enumerate(seqs):
        out[i] = len(s)
    return out


def quantile_bins(vals, qs=(0, 0.25, 0.5, 0.75, 1.0)):
    edges = np.quantile(vals, qs)
    edges = list(np.unique(edges))
    return edges


def bin_idx(vals, edges):
    # rightmost edge included
    idx = np.searchsorted(edges, vals, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx


def lig_rna_dna_set():
    s = set()
    for sub in LIG_RNA_DNA_DIRS:
        p = os.path.join(f"{ROOT}/data/ligand_train", sub)
        if os.path.isdir(p):
            s |= {f for f in os.listdir(p) if f.endswith(".pdb")}
    # rna label npz extra safety
    for f in ["labels_rna_v14.npz", "labels_rna_v14_sup.npz", "labels_rna_v14_sup2.npz"]:
        fp = os.path.join(f"{ROOT}/data/ligand_train", f)
        if os.path.isfile(fp):
            dd = np.load(fp, allow_pickle=True)
            s |= {str(x) for x in dd["domain_ids"]}
    return s


def stratify(data, frac, seed, rna_ids=None):
    n = len(data["domain_ids"])
    L = dom_L(data["seqs"])
    q_mean = data["charge"].mean(axis=1)          # domain mean net charge across 8 arms
    rng = np.random.RandomState(seed)

    if rna_ids is not None:
        is_rna = np.array([x in rna_ids for x in data["domain_ids"]], dtype=bool)
    else:
        is_rna = None

    # L bins: shared for both; long-tail preserved by extra bins
    L_edges = [0, 60, 100, 150, 220, 320, 500, 1_000_000]
    if L.max() > 1000:
        L_edges = [0, 60, 100, 150, 220, 320, 500, 800, 1_000_000]
    l_bin = bin_idx(L, L_edges)
    # charge bins: 5 quantile bins of mean charge
    c_edges = quantile_bins(q_mean)
    c_bin = bin_idx(q_mean, c_edges)

    layers = {}
    for i in range(n):
        if is_rna is not None:
            key = (int(is_rna[i]), int(l_bin[i]), int(c_bin[i]))
        else:
            key = (int(l_bin[i]), int(c_bin[i]))
        layers.setdefault(key, []).append(i)

    picked = []
    for key, idxs in layers.items():
        idxs = np.array(sorted(idxs))
        k = int(np.ceil(len(idxs) * frac))
        k = min(k, len(idxs))
        if k >= len(idxs):
            pick = idxs
        else:
            pick = rng.choice(idxs, size=k, replace=False)
        picked.append(pick)
    picked = np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)
    return picked, L, q_mean, is_rna, L_edges, c_edges, l_bin, c_bin


def save_subset(data, idx, out):
    np.savez_compressed(
        out,
        domain_ids=data["domain_ids"][idx],
        seqs=data["seqs"][idx],
        coords=data["coords"][idx],
        pH=data["pH"][idx].reshape(-1),
        charge=data["charge"][idx].reshape(-1),
        pI=data["pI"][idx].reshape(-1),
    )
    print(f"wrote {out}  n_dom={len(idx)}")


def report(name, data, idx, L, q_mean, is_rna=None, rna_ids=None):
    Ls, Qs = L, q_mean
    def _p(vals):
        return [round(float(x), 1) for x in np.percentile(vals, [25, 50, 75])]
    full = {"n": int(len(Ls)), "L_mean": float(Ls.mean()), "L_p": _p(Ls),
            "Q_mean": float(Qs.mean()), "Q_p": _p(Qs)}
    sub = {"n": int(len(idx)), "L_mean": float(Ls[idx].mean()),
           "L_p": _p(Ls[idx]),
           "Q_mean": float(Qs[idx].mean()),
           "Q_p": _p(Qs[idx])}
    frac_real = len(idx) / len(Ls)
    rep = {"name": name, "full": full, "subset": sub, "subset_frac": round(frac_real, 4)}
    if is_rna is not None:
        rep["full_rna_dna"] = int(is_rna.sum())
        rep["subset_rna_dna"] = int(is_rna[idx].sum())
        rep["subset_rna_frac"] = round(float(is_rna[idx].mean()), 4)
    print("=" * 70)
    print(f"[{name}] subset/frac = {len(idx)}/{len(Ls)} = {frac_real:.4f}")
    print(f"  L   full mean {full['L_mean']:.1f} p {full['L_p']}  |  sub mean {sub['L_mean']:.1f} p {sub['L_p']}")
    print(f"  Q   full mean {full['Q_mean']:.2f} p {full['Q_p']}  |  sub mean {sub['Q_mean']:.2f} p {sub['Q_p']}")
    if is_rna is not None:
        print(f"  RNA/DNA full {rep['full_rna_dna']} ({rep['full_rna_dna']/len(Ls):.4f}) | "
              f"sub {rep['subset_rna_dna']} ({rep['subset_rna_frac']:.4f})")
    return rep


def main():
    os.makedirs(f"{ROOT}/ablation/data", exist_ok=True)
    summary = {}

    # ---- protein ----
    pd = load(PROT_SRC)
    p_idx, L, Q, _, _, _, _, _ = stratify(pd, FRAC, SEED, rna_ids=None)
    save_subset(pd, p_idx, PROT_OUT)
    summary["prot"] = report("PROT", pd, p_idx, L, Q)

    # ---- ligand ----
    rna_ids = lig_rna_dna_set()
    ld = load(LIG_SRC)
    l_idx, L, Q, is_rna, _, _, _, _ = stratify(ld, FRAC, SEED, rna_ids=rna_ids)
    save_subset(ld, l_idx, LIG_OUT)
    summary["lig"] = report("LIG", ld, l_idx, L, Q, is_rna=is_rna, rna_ids=rna_ids)

    with open(f"{ROOT}/ablation/data/subsets_balance.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nsummary written ablation/data/subsets_balance.json")


if __name__ == "__main__":
    main()
