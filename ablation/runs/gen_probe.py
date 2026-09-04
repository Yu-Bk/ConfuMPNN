"""Ablation generation probe: per-run encoder, sample n seqs on arms {native,n2,p2}.

Measures (lightweight, for module-contribution ordering only):
  - H2: |mean_generated_charge(pH) - target| <= 2.0  (n per arm)
  - native-arm charged-residue retention: mean over samples of
    (D+E+K+R count in gen seq) / (D+E+K+R count in native seq)  [deletion proxy]
No calibration is applied (raw ConditionEncoder response), since calibration would
compress per-module differences we intend to rank.

Usage:
  python gen_probe.py --mode protein --pdb code/input/1BC8_chainC.pdb \
      --encoder ablation/runs/prot/run_FULL/condition_encoder_last.pt \
      --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
      --out ablation/runs/prot/probe_FULL.json --n 30 --device cuda:6
  python gen_probe.py --mode ligand --pdb <rna.pdb> ... (uses atom25 ligand context)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_DIR = next(p for p in Path(__file__).resolve().parents
                    if (p / "code").is_dir() and (p / "LigandMPNN").is_dir())
_CODE_DIR = _PROJECT_DIR / "code"
for _p in [str(_CODE_DIR), str(_PROJECT_DIR / "LigandMPNN")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import load_condition_encoder, load_model, seq_to_string  # noqa: E402

AA_CHARGED = set("DEKR")


def build_feature_dict(pdb_path, device, ligand):
    protein_dict, *_ = parse_PDB(str(pdb_path), device="cpu", parse_all_atoms=False)
    L = protein_dict["X"].shape[0]
    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
    n_lig = 25 if ligand else 0
    feature_dict = featurize(
        protein_dict,
        use_atom_context=ligand,
        number_of_ligand_atoms=n_lig,
        model_type=("ligand_mpnn" if ligand else "protein_mpnn"),
    )
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = 0.3
    L = feature_dict["X"].shape[1]
    feature_dict["bias"] = torch.zeros(1, L, 21)
    return feature_dict


def sample_one(model, enc, fd, cond_vec, device, seed):
    torch.manual_seed(seed)
    L = fd["X"].shape[1]
    fd["randn"] = torch.randn(1, L)
    out = conditioned_sample(model, enc, fd, cond_vec, device=device)
    return seq_to_string(out["S"][0].cpu().numpy())


def charged_counts(seq):
    return sum(1 for aa in seq if aa in AA_CHARGED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["protein", "ligand"], required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--encoder", required=True, help="ConditionEncoder ckpt")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed_base", type=int, default=111)
    ap.add_argument("--device", default="cuda:6")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = args.weights
    if weights is None:
        if args.mode == "protein":
            weights = str(_PROJECT_DIR / "MoMPNN/mompnn_paper_checkpoints"
                          / "mompnn_temberture_tm_esm_6_4_4_b01.ckpt")
        else:
            weights = str(_PROJECT_DIR / "LigandMPNN/model_params"
                          / "ligandmpnn_v_32_010_25.pt")

    model_type = "ligand_mpnn" if args.mode == "ligand" else "protein_mpnn"
    model = load_model(weights, device, model_type=model_type)
    enc = load_condition_encoder(args.encoder, device)

    fd0 = build_feature_dict(args.pdb, device, args.mode == "ligand")
    L = fd0["X"].shape[1]
    native = seq_to_string(fd0["S"][0].cpu().numpy())
    native_charge = float(net_charge(native, args.pH))
    t_nat = int(round(native_charge))
    arms = {"native": float(t_nat), "n2": float(t_nat - 2), "p2": float(t_nat + 2)}
    native_charged = charged_counts(native)

    print(f"[probe] mode={args.mode} L={L} native@{args.pH}={native_charge:+.2f} "
          f"charged_native={native_charged} n={args.n} arms={arms}", flush=True)

    results = {"mode": args.mode, "pdb": args.pdb, "L": L,
               "native_charge": round(native_charge, 2), "n_charged_native": native_charged,
               "arms": {}}
    for arm, tgt in arms.items():
        fd = build_feature_dict(args.pdb, device, args.mode == "ligand")
        charges, seqs = [], []
        for k in range(args.n):
            seed = args.seed_base + k
            cond_vec = make_condition_vector(args.pH, net_charge=tgt)
            seq = sample_one(model, enc, fd, cond_vec, device, seed)
            q = float(net_charge(seq, args.pH))
            charges.append(q)
            seqs.append(seq)
        mean_q = float(np.mean(charges))
        std_q = float(np.std(charges))
        dev = abs(mean_q - tgt)
        h2 = dev <= 2.0
        # retention on native arm only
        ret = None
        if arm == "native" and native_charged > 0:
            ret = float(np.mean([charged_counts(s) / native_charged for s in seqs]))
        results["arms"][arm] = {
            "target": tgt, "mean_charge": round(mean_q, 2),
            "std_charge": round(std_q, 2), "dev": round(dev, 2),
            "h2": bool(h2), "retention_native_arm": ret,
            "mean_charged_count": round(float(np.mean([charged_counts(s) for s in seqs])), 2),
        }
        print(f"  {arm:6s} target={tgt:+.1f} mean={mean_q:+6.2f}±{std_q:.2f} "
              f"dev={dev:.2f} h2={h2} ret={ret}", flush=True)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
