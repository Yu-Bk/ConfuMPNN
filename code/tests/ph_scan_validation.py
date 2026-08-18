"""多 pH 温和区复现天然蛋白验证：在多个 pH 下 target=native 电荷，测能否复现天然。

背景：用户要求系统验证——"以 PDB 骨架为输入，在 pH=5~10 下 target≈天然电荷，
能否设计出合适序列并折回原结构，成功率多少"。本脚本覆盖 **温和区**
（target=round(native charge@pH)），与已有的极端区数据（v5-v7 三区间）互补。

输出结构（供 esmfold_score.py 递归回折）：
  {out_dir}/{pdb}/pH{ph}/seqs.fa       每 pH n 条序列
  {out_dir}/{pdb}/pH{ph}/charge_stats.json

用法（code/ 下）：
  PYTHONPATH=. python tests/ph_scan_validation.py \
      --pdb ../code/input/1UBQ.pdb --pH_list "5,7.4,9" --n 50 \
      --cond_encoder ../output/finetune_v7/finetune_epoch030.pt \
      --out_dir ../output/ph_scan --device cuda:3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR.parent / "LigandMPNN"))

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import load_model, load_condition_encoder, seq_to_string  # noqa: E402

DEFAULT_BACKBONE = (_CODE_DIR.parent / "MoMPNN" / "mompnn_paper_checkpoints"
                    / "mompnn_temberture_tm_esm_6_4_4_b01.ckpt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--pH_list", default="5,7.4,9")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--weights", default=None, help="backbone（默认 MoMPNN）")
    ap.add_argument("--cond_encoder", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed_base", type=int, default=111)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--uncond", action="store_true",
                    help="无条件模式（不注入条件）：作 recovery 基线对照")
    args = ap.parse_args()

    device = torch.device(args.device)
    weights = Path(args.weights) if args.weights else DEFAULT_BACKBONE
    model = load_model(str(weights), device, model_type="auto")
    enc = load_condition_encoder(args.cond_encoder, device)

    protein_dict, _, _, icodes, _ = parse_PDB(args.pdb)
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32)
    fd = featurize(protein_dict, cutoff_for_score=8.0, use_atom_context=False,
                   number_of_ligand_atoms=0, model_type="protein_mpnn")
    L = fd["X"].shape[1]
    fd["batch_size"] = 1
    fd["temperature"] = 0.3
    fd["bias"] = torch.zeros(1, L, 21)
    native = seq_to_string(fd["S"][0].cpu().numpy())

    out_root = Path(args.out_dir) / Path(args.pdb).stem
    print(f"[load] {Path(args.pdb).name} L={L} "
          f"native={native[:30]}...", flush=True)

    summary = {"pdb": Path(args.pdb).name, "L": L, "n": args.n,
               "native": native, "pH_arms": {}}

    for ph_s in args.pH_list.split(","):
        pH = float(ph_s)
        q_nat = float(net_charge(native, pH))
        tgt = round(q_nat)
        charges, seqs = [], []
        for k in range(args.n):
            torch.manual_seed(args.seed_base + k)
            fd["randn"] = torch.randn(1, L)
            if args.uncond:
                # 无条件基线：不注入条件，同 seed 同解码顺序，唯一差异=无条件
                out = conditioned_sample(model, None, fd, None, device)
            else:
                cond_vec = make_condition_vector(pH, net_charge=tgt)
                out = conditioned_sample(model, enc, fd, cond_vec, device)
            seq = seq_to_string(out["S"][0].cpu().numpy())
            seqs.append(seq)
            charges.append(float(net_charge(seq, pH)))
        mean_c = float(np.mean(charges))
        dev = abs(mean_c - tgt) if not args.uncond else None
        recs = [sum(a == b for a, b in zip(s, native)) / L for s in seqs]
        arm_dir = out_root / f"pH{ph_s}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        fa = arm_dir / "seqs.fa"
        with open(fa, "w") as f:
            for i, (s, c) in enumerate(zip(seqs, charges)):
                tag = "uncond" if args.uncond else f"target={tgt:+.0f}"
                f.write(f">seed_{args.seed_base+i} pH={pH} {tag} "
                        f"charge={c:+.2f}\n{s}\n")
            f.write(f">native pH={pH} charge={q_nat:+.2f}\n{native}\n")
        summary["pH_arms"][ph_s] = {
            "mode": "uncond" if args.uncond else "cond_native",
            "target": None if args.uncond else tgt,
            "native_charge": round(q_nat, 2),
            "mean_charge": round(mean_c, 2),
            "std_charge": round(float(np.std(charges)), 2),
            "dev": round(dev, 2) if dev is not None else None,
            "recovery": round(float(np.mean(recs)), 3),
        }
        print(f"  pH={pH:>4} {'uncond':6s} native={q_nat:+6.2f} "
              f"mean={mean_c:+6.2f} recovery={np.mean(recs):.3f}"
              if args.uncond else
              f"  pH={pH:>4} cond   native={q_nat:+6.2f} target={tgt:>3} "
              f"mean={mean_c:+6.2f} dev={dev:.2f} recovery={np.mean(recs):.3f}",
              flush=True)

    with open(out_root / "ph_scan.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"已写 {out_root}/ph_scan.json")


if __name__ == "__main__":
    main()
