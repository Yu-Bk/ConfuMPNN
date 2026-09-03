"""配体模式（v14）校准表构建：big（纯训练域 global，无泄露）/ small（验证蛋白现场标定）。

口径（2026-09-03 主报告 §1.2）：达标只报 big-global（纯训练域拟合）与小样本现场标定；
per-protein 表内（诊断网格）剔除出达标讨论。global 校准表不得混入 valid 点。

用法：
  # big：从 labels_v14_final 训练域分层抽样 60 域 × 5 offset(native±[8,4,0,4,8]) × n10 → global
  PYTHONPATH=code python code/tests/ligand_v9/build_calibration_ligand.py --mode big \
     --enc output/finetune_ligand_v14_rna/finetune_epoch050.pt \
     --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
     --device cuda:5 --n_dom 60 --out output/charge_calibration_v14_big.json

  # small：对 manifest(in 10) 每蛋白 5 offset × n10 → per_protein；global 兜底用 big 表
  PYTHONPATH=code python code/tests/ligand_v9/build_calibration_ligand.py --mode small \
     --manifest data/validation_pdbs/validation_manifest_v14_in.json \
     --big_cal output/charge_calibration_v14_big.json \
     --enc output/finetune_ligand_v14_rna/finetune_epoch050.pt \
     --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
     --device cuda:5 --out output/charge_calibration_v14_small.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_DIR = next(p for p in Path(__file__).resolve().parents
                    if (p / "code").is_dir() and (p / "LigandMPNN").is_dir())
_CODE_DIR = _PROJECT_DIR / "code"
for p in (str(_CODE_DIR), str(_PROJECT_DIR / "LigandMPNN")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import load_model, load_condition_encoder, seq_to_string  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--mode", choices=["big", "small"], required=True)
AP.add_argument("--n_dom", type=int, default=60, help="big：训练域抽样数")
AP.add_argument("--n_per", type=int, default=10, help="每 offset 采样数")
AP.add_argument("--seed", type=int, default=777)
AP.add_argument("--temperature", type=float, default=0.3)
AP.add_argument("--device", default="cuda:5")
AP.add_argument("--labels", default=str(_PROJECT_DIR / "data/ligand_train/labels_v14_final.npz"))
AP.add_argument("--dompdb", default=str(_PROJECT_DIR / "data/ligand_train/all_pdb"))
AP.add_argument("--manifest", default=str(_PROJECT_DIR / "data/validation_pdbs/validation_manifest_v14_in.json"))
AP.add_argument("--enc", default=str(_PROJECT_DIR / "output/finetune_ligand_v14_rna/finetune_epoch050.pt"))
AP.add_argument("--weights", default=str(_PROJECT_DIR / "LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"))
AP.add_argument("--big_cal", default=str(_PROJECT_DIR / "output/charge_calibration_v14_big.json"))
AP.add_argument("--out", required=True)
ARGS = AP.parse_args()

OFFSETS = [-8, -4, 0, 4, 8]


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 1e-9 else float("nan")
    b = my - a * mx
    return a, b


def featurize_ligand(protein_dict):
    protein_dict["chain_mask"] = torch.ones(protein_dict["X"].shape[0], dtype=torch.int32)
    fd = featurize(protein_dict, cutoff_for_score=8.0, model_type="ligand_mpnn",
                   use_atom_context=True, number_of_ligand_atoms=25)
    fd["batch_size"] = 1
    fd["temperature"] = ARGS.temperature
    fd["bias"] = torch.zeros(1, fd["X"].shape[1], 21)
    return fd


def sample_mean(model, enc, fd, device, tgt, n):
    charges = []
    for k in range(n):
        torch.manual_seed(ARGS.seed + int(tgt * 7) + k)
        fd["randn"] = torch.randn(1, fd["X"].shape[1])
        cond_vec = make_condition_vector(7.4, net_charge=tgt)
        out = conditioned_sample(model, enc, fd, cond_vec, device)
        seq = seq_to_string(out["S"][0].cpu().numpy())
        charges.append(float(net_charge(seq, 7.4)))
    return float(np.mean(charges))


def main():
    device = torch.device(ARGS.device)
    enc = load_condition_encoder(ARGS.enc, device)
    model = load_model(ARGS.weights, device, model_type="auto")
    print(f"backbone={model.model_type} enc={Path(ARGS.enc).name}", flush=True)

    if ARGS.mode == "big":
        d = np.load(ARGS.labels, allow_pickle=True)
        c7 = np.array([float(net_charge(str(s), 7.4)) for s in d["seqs"]])
        # 分层：按电荷分箱，每箱均取，覆盖全范围
        qs = np.quantile(c7, np.linspace(0, 1, 11))
        rng = random.Random(ARGS.seed)
        per_bin = ARGS.n_dom // 10
        sel = []
        for b in range(10):
            lo, hi = qs[b], qs[b + 1]
            idx = [i for i, c in enumerate(c7) if lo <= c < hi or (b == 9 and lo <= c <= hi)]
            take = min(per_bin, len(idx))
            sel.extend(rng.sample(idx, take))
        sel = sorted(sel[:ARGS.n_dom])
        print(f"训练域抽样 {len(sel)}（charge@7.4 范围 [{c7[sel].min():.1f},{c7[sel].max():.1f}]）", flush=True)
        points, per_domain = [], {}
        for i in sel:
            did = str(d["domain_ids"][i])
            pdb_path = Path(ARGS.dompdb) / did
            try:
                protein_dict, *_ = parse_PDB(str(pdb_path), device="cpu", parse_all_atoms=False)
            except Exception as e:
                print(f"  !! {did} parse 失败: {e}", flush=True)
                continue
            native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
            nq = round(float(net_charge(native, 7.4)))
            fd = featurize_ligand(protein_dict)
            arms = {}
            for off in OFFSETS:
                tgt = nq + off
                m = sample_mean(model, enc, fd, device, tgt, ARGS.n_per)
                points.append((tgt, m))
                arms[off] = round(m, 2)
            per_domain[did] = {"L": fd["X"].shape[1], "native_q": nq, "arms": arms}
            print(f"{did} L={fd['X'].shape[1]} nq={nq:+d}: " +
                  " ".join(f"{o:+d}→{arms[o]:+.1f}" for o in OFFSETS), flush=True)
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        slope, inter = linfit(xs, ys)
        out = {"global": {"slope": round(slope, 4), "intercept": round(inter, 4),
                          "n_point": len(points), "n_domains": len(per_domain)},
               "per_protein": per_domain}
        json.dump(out, open(ARGS.out, "w"), indent=2)
        print(f"已写 {ARGS.out}: global slope={slope:.3f} inter={inter:.3f} "
              f"n_point={len(points)}（{len(per_domain)} 域，纯训练域，无 valid 混入）", flush=True)

    else:  # small
        big = json.load(open(ARGS.big_cal))
        man = json.load(open(ARGS.manifest))
        per = {}
        for it in man["items"]:
            name, path = it["pdb"], str(_PROJECT_DIR / it["path"])
            try:
                protein_dict, *_ = parse_PDB(path, device="cpu", parse_all_atoms=False)
            except Exception as e:
                print(f"  !! {name} parse 失败: {e}", flush=True)
                continue
            native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
            nq = round(float(net_charge(native, 7.4)))
            fd = featurize_ligand(protein_dict)
            xs, ys = [], []
            for off in OFFSETS:
                tgt = nq + off
                m = sample_mean(model, enc, fd, device, tgt, ARGS.n_per)
                xs.append(tgt); ys.append(m)
            slope, inter = linfit(xs, ys)
            # LOOCV 稳定性
            loocv_errs = []
            for i in range(len(xs)):
                xr = xs[:i] + xs[i + 1:]; yr = ys[:i] + ys[i + 1:]
                a, b = linfit(xr, yr)
                loocv_errs.append(abs((a * xs[i] + b) - ys[i]))
            loocv = float(sum(loocv_errs) / len(loocv_errs))
            unreliable = loocv > 3.0
            per[name] = {"slope": round(slope, 4), "intercept": round(inter, 4),
                         "n_calib": len(xs) * ARGS.n_per, "native_q": nq,
                         "loocv": round(loocv, 2), "unreliable": unreliable}
            tag = " ⚠️LOOCV大" if unreliable else ""
            print(f"{name:8s} L={fd['X'].shape[1]} nq={nq:+d} slope={slope:.3f} int={inter:.2f} "
                  f"LOOCV={loocv:.2f}{tag} | " + " ".join(f"{t:+d}→{m:+.1f}" for t, m in zip(xs, ys)), flush=True)
        out = {"global": big["global"], "per_protein": per}
        json.dump(out, open(ARGS.out, "w"), indent=2)
        n_bad = sum(1 for v in per.values() if v.get("unreliable"))
        print(f"已写 {ARGS.out}: {len(per)} 个 per_protein（现场标定），{n_bad} unreliable", flush=True)


if __name__ == "__main__":
    main()
