#!/usr/bin/env python
"""L11 现场小样本标定探针（单蛋白 L11，支持蛋白/配体模式与 pH7.4/8.0）。

方法（仿 code/tests/build_calibration_small.py，2026-08-31 定稿协议）：
  在 L11 骨架上（含固定位 I3 I5 I9 I34 I35 I89 I124 I131 I134 I135，与真实实验一致）
  采 native_q(round) ± [8,4,0,4,8] 5 档 × --n_per(=10) = 50 条，
  拟合 target→生成电荷均值 的直线 slope/intercept；LOOCV 标记 unreliable。

与 build_calibration_small 的两点差异（都符合 L11 实验语境）：
  1. 单蛋白（不读 manifest/10 蛋白）；
  2. 探针批次同样固定 10 个界面位（保证拟合的响应与真实采样条件一致）。

用法（confumpnn 环境，任意 cwd）：
  python L11design/test/exp3_calibrate_probe.py --mode protein --pH 7.4 \
         --out L11design/output/exp3/cal/L11_small_prot_pH74.json [--device cpu]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "LigandMPNN"))

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import load_model, load_condition_encoder, seq_to_string  # noqa: E402

FIXED_STR = "I3 I5 I9 I34 I35 I89 I124 I131 I134 I135"
OFFSETS = [-8, -4, 0, 4, 8]

MODES = {
    "protein": dict(
        pdb=ROOT / "L11design/input/L11.pdb",
        weights=ROOT / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt",
        cond_encoder=ROOT / "output/finetune_v12_2/finetune_epoch030.pt",
        fallback_cal=ROOT / "output/charge_calibration_v12_2.json",
        per_key="L11",
        label="v12_2_protein",
    ),
    "ligand": dict(
        pdb=ROOT / "L11design/input/L11_RNA.pdb",
        weights=ROOT / "LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt",
        cond_encoder=ROOT / "output/finetune_ligand_v14_rna/finetune_epoch050.pt",
        fallback_cal=ROOT / "output/charge_calibration_v14_ligand_clean.json",
        per_key="L11_RNA",
        label="v14_ligand",
    ),
}


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 1e-9 else float("nan")
    b = my - a * mx
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["protein", "ligand"])
    ap.add_argument("--pH", type=float, required=True)
    ap.add_argument("--n_per", type=int, default=10)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--consistency_thresh", type=float, default=3.0)
    args = ap.parse_args()

    cfg = MODES[args.mode]
    device = torch.device(args.device)
    pH = args.pH
    print(f"[L11 probe] mode={args.mode} pH={pH} n_per={args.n_per} device={args.device}", flush=True)

    enc = load_condition_encoder(str(cfg["cond_encoder"]), device)
    model = load_model(str(cfg["weights"]), device, model_type="auto")
    mt = model.model_type
    use_atom_context = (mt == "ligand_mpnn")
    n_lig_atoms = 25 if use_atom_context else 0
    print(f"    模型类型 = {mt}  use_atom_context={use_atom_context} n_lig_atoms={n_lig_atoms}", flush=True)

    protein_dict, _, _, icodes, _ = parse_PDB(str(cfg["pdb"]))
    L = protein_dict["X"].shape[0]
    chain_letters = list(protein_dict["chain_letters"])
    R_idx = list(protein_dict["R_idx"].cpu().numpy())
    encoded = [str(chain_letters[i]) + str(R_idx[i]) + icodes[i] for i in range(L)]
    fixed_set = set(FIXED_STR.split())

    protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
    fixed_i = []
    for i, name in enumerate(encoded):
        if name in fixed_set:
            protein_dict["chain_mask"][i] = 0
            fixed_i.append(i)
    print(f"    固定位 {len(fixed_i)} 个: {[encoded[i] for i in fixed_i]}", flush=True)
    assert len(fixed_i) == 10, f"固定位应 10 个，实际 {len(fixed_i)}"

    fd = featurize(protein_dict, cutoff_for_score=8.0, use_atom_context=use_atom_context,
                   number_of_ligand_atoms=n_lig_atoms, model_type=mt)
    fd["batch_size"] = 1
    fd["temperature"] = args.temperature
    fd["bias"] = torch.zeros(1, L, 21)

    native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
    native_q = round(float(net_charge(native, pH)))
    print(f"    native len={L}  native_q@pH{pH} = {net_charge(native,pH):+.3f} → round {native_q:+d}", flush=True)

    xs, ys, raw = [], [], []
    for off in OFFSETS:
        tgt = native_q + off
        charges = []
        for k in range(args.n_per):
            torch.manual_seed(args.seed + off + k)
            fd["randn"] = torch.randn(1, L)
            cond_vec = make_condition_vector(pH, net_charge=tgt)
            out = conditioned_sample(model, enc, fd, cond_vec, device)
            seq = seq_to_string(out["S"][0].cpu().numpy())
            # 校验固定位
            bad = [encoded[i] for i in fixed_i if seq[i] != native[i]]
            if bad:
                raise RuntimeError(f"固定位错配 {bad} @ tgt {tgt}")
            charges.append(float(net_charge(seq, pH)))
        m = float(np.mean(charges))
        xs.append(tgt)
        ys.append(m)
        raw.append({"target": tgt, "charges": [round(c, 3) for c in charges],
                    "mean": round(m, 3), "std": round(float(np.std(charges)), 3)})
        print(f"    target {tgt:+d} → 生成电荷均值 {m:+.2f} ± {np.std(charges):.2f}  "
              f"({[round(c,1) for c in charges]})", flush=True)

    slope, inter = linfit(xs, ys)
    # LOOCV
    loocv_errs = []
    for i in range(len(xs)):
        xr = xs[:i] + xs[i + 1:]
        yr = ys[:i] + ys[i + 1:]
        a, b = linfit(xr, yr)
        loocv_errs.append(abs((a * xs[i] + b) - ys[i]))
    loocv = float(sum(loocv_errs) / len(loocv_errs))
    unreliable = loocv > args.consistency_thresh

    # global 兜底 = 原表 global（表外蛋白回退口径 = Exp1/Exp2 用的 global）
    fc = json.load(open(cfg["fallback_cal"]))
    g = fc.get("global", {})
    if not g:
        # 退化：用小样本拟合自己兜底
        g = {"slope": slope, "intercept": inter}
    out = {
        "label": f"L11_small_{args.mode}_pH{args.pH}",
        "note": "L11 现场小样本标定（探针含固定位）。生成电荷≈slope*target+intercept；"
                "校准 target_eff=(desired−intercept)/slope。global=原表 global 兜底（Exp1/2 口径）。",
        "mode": args.mode, "pH": pH, "protein": "L11",
        "global": {"slope": round(float(g["slope"]), 4), "intercept": round(float(g["intercept"]), 4),
                   "source": cfg["fallback_cal"].name},
        "per_protein": {
            cfg["per_key"]: {
                "slope": round(slope, 4), "intercept": round(inter, 4),
                "n_calib": len(xs) * args.n_per, "native_q": native_q,
                "loocv": round(loocv, 2), "unreliable": unreliable,
            }
        },
        "fit_diag": {"targets": xs, "mean_charges": [round(y, 3) for y in ys],
                     "slope_raw": slope, "intercept_raw": inter, "loocv": round(loocv, 2),
                     "points": raw},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    tag = " ⚠️ unreliable(LOOCV大)" if unreliable else ""
    print(f"\nL11 {args.mode} pH{pH}: slope={slope:.4f} intercept={inter:.4f} "
          f"LOOCV={loocv:.2f}{tag} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
