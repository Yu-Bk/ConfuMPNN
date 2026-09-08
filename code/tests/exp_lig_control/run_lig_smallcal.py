"""配体模式现场小样本标定附加组（v2 5O60_E 之类某臂 B 差时）。

对指定蛋白：
  1) 现场标定：direct 注入 5 个 offset target（t_native+[-8,-4,0,4,8]）× n_per=10 → 50 条
     拟合生成电荷 ~ slope*target+inter（LOOCV 校验）。
  2) 用该 per-protein 校准重采样指定臂（默认 native/p2，即 route B 里 dev>2 的臂）n=1000/臂，
     与 route B 同 seed（seed_base+k）。

用法：
  python code/tests/exp_lig_control/run_lig_smallcal.py --device cuda:2 \
      --proteins 5O60_E --arms native,p2 --out_dir output/exp_control_lig_v2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parents[2]
_ROOT = _CODE_DIR.parent
for p in [str(_CODE_DIR), str(_ROOT / "LigandMPNN")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import parse_PDB  # noqa: E402
from run_guided import load_model, load_condition_encoder, seq_to_string  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402

PROTEINS = {
    "5O60_E": "data/validation_pdbs/5O60_E.pdb",
    "1CGE": "data/validation_pdbs/1CGE.pdb",
    "1BJ4": "data/validation_pdbs/1BJ4.pdb",
}
DEFAULT_W = str(_ROOT / "LigandMPNN" / "model_params" / "ligandmpnn_v_32_010_25.pt")
DEFAULT_ENC = str(_ROOT / "output" / "finetune_ligand_v14_rna" / "finetune_epoch050.pt")
OFFSETS = [-8, -4, 0, 4, 8]
pH = 7.4
TEMP = 0.3
N_LIG = 25


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 1e-9 else float("nan")
    b = my - a * mx
    return a, b


def prep(pdb_path):
    protein_dict, *_ = parse_PDB(pdb_path, device="cpu")
    L = int(protein_dict["X"].shape[0])
    native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
    q_nat = float(net_charge(native, pH))
    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
    from data_utils import featurize
    fd = featurize(protein_dict, cutoff_for_score=8.0, use_atom_context=True,
                   number_of_ligand_atoms=N_LIG, model_type="ligand_mpnn")
    fd["batch_size"] = 1
    fd["temperature"] = TEMP
    fd["bias"] = torch.zeros(1, L, 21)
    return fd, L, native, q_nat


def sample_cond(model, enc, fd, tgt_eff, seed, device):
    torch.manual_seed(seed)
    fd["randn"] = torch.randn(1, fd["X"].shape[1])
    cv = make_condition_vector(pH, net_charge=float(tgt_eff)).to(device)
    out = conditioned_sample(model, enc, fd, cv, device)
    return seq_to_string(out["S"][0].cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--weights", default=DEFAULT_W)
    ap.add_argument("--cond_encoder", default=DEFAULT_ENC)
    ap.add_argument("--proteins", default="5O60_E")
    ap.add_argument("--arms", default="native,p2")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_per_calib", type=int, default=10)
    ap.add_argument("--out_dir", default=str(_ROOT / "output" / "exp_control_lig_v2"))
    args = ap.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.out_dir)
    model = load_model(args.weights, device, model_type="auto")
    enc = load_condition_encoder(args.cond_encoder, device)
    pdbs = [p.strip() for p in args.proteins.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    ARM_DQ = {"native": 0, "n2": -2, "p2": +2, "n8": -8, "p8": +8}

    for pdb in pdbs:
        pdb_path = _ROOT / PROTEINS[pdb]
        fd, L, native, q_nat = prep(str(pdb_path))
        t_nat = int(round(q_nat))
        print(f"\n===== {pdb} L={L} native_q={q_nat:+.2f} round={t_nat} =====", flush=True)
        pout = out_root / pdb / "routeB_small"
        pout.mkdir(parents=True, exist_ok=True)

        # --- on-site calibration (direct injection, 50 seq) ---
        xs, ys, calib_rows = [], [], []
        for off in OFFSETS:
            tgt = t_nat + off
            qs = []
            for k in range(args.n_per_calib):
                sd = 3000 + 50000 + k + off  # 与蛋白 driver 类似，另起 seed 段避免碰撞
                s = sample_cond(model, enc, fd, float(tgt), sd, device)
                q = float(net_charge(s, pH))
                qs.append(q)
                calib_rows.append({"offset": off, "target": tgt, "seed": sd,
                                   "charge": round(q, 3)})
            xs.append(tgt); ys.append(float(np.mean(qs)))
        slope, inter = linfit(xs, ys)
        errs = []
        for i in range(len(xs)):
            xr = xs[:i] + xs[i+1:]; yr = ys[:i] + ys[i+1:]
            a, b = linfit(xr, yr)
            errs.append(abs((a*xs[i]+b) - ys[i]))
        loocv = float(np.mean(errs))
        sc = {"pdb": pdb, "slope": round(slope, 4), "intercept": round(inter, 4),
              "n_calib": len(xs)*args.n_per_calib, "native_q": t_nat,
              "loocv": round(loocv, 2),
              "mean_charge_by_target": {str(t): round(m, 3) for t, m in zip(xs, ys)},
              "calib_rows": calib_rows}
        with open(pout / f"{pdb}_smallcal.json", "w") as f:
            json.dump(sc, f, indent=2)
        print(f"  smallcal: slope={slope:.3f} inter={inter:.2f} LOOCV={loocv:.2f}", flush=True)

        # --- resample selected arms ---
        for arm in arms:
            dq = ARM_DQ[arm]
            tgt = t_nat + dq
            tgt_eff = (tgt - inter) / slope
            arm_dir = pout / f"arm_{arm}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            seqs, charges, devs, recs = [], [], [], []
            t0 = time.time()
            for k in range(args.n):
                sd = 3000 + k
                s = sample_cond(model, enc, fd, tgt_eff, sd, device)
                q = float(net_charge(s, pH))
                seqs.append(s); charges.append(q)
                devs.append(q - tgt)
                recs.append(sum(a == b for a, b in zip(s, native)) / len(native))
                if (k + 1) % 200 == 0:
                    print(f"    [{arm}] {k+1}/{args.n} ...", flush=True)
            res = {"seed_base": 3000, "n": args.n, "arm": arm, "target": tgt,
                   "target_inject": round(tgt_eff, 3), "calibrate": True,
                   "smallcal_slope": slope, "smallcal_inter": inter,
                   "charges": charges, "devs": devs, "recs": recs, "seqs": seqs}
            with open(arm_dir / "sequences.json", "w") as f:
                json.dump(res, f, indent=2)
            with open(arm_dir / "seqs.fa", "w") as f:
                for i, (s, q) in enumerate(zip(seqs, charges)):
                    f.write(f">seed{i} target={tgt:+d} charge={q:+.2f}\n{s}\n")
                f.write(f">native charge={q_nat:+.2f}\n{native}\n")
            md = float(np.mean(charges)) - tgt
            hit = float(np.mean(np.abs(np.array(charges) - tgt) <= 2.0))
            print(f"    [{arm}] n={args.n} mean={np.mean(charges):+.2f} dev={md:+.2f} "
                  f"hit≤2={hit:.3f} rec={np.mean(recs):.3f} ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== done ===", flush=True)


if __name__ == "__main__":
    main()
