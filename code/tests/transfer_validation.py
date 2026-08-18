"""迁移应用能力检验：模型在**没见过的新蛋白**上能否正常工作（小规模采样）。

背景：用户要求"多取几个其他的有配体没有配体的蛋白 PDB 进行检验，每个蛋白
的采样条件和生成序列数量可以大大降低，检验模型在其他蛋白中的迁移应用能力"。

设计（相比 ph_scan_validation 的 3 蛋白×3pH×n50，本脚本刻意小规模）：
  每个蛋白：pH_list 默认 7.4（+可选一个梯度 pH），n 默认 10。
  模式：protein=无配体（protein_mpnn 特征），ligand=有配体（ligand_mpnn 特征，
       权重用原版 LigandMPNN，配体原子进上下文）。
  target = round(native charge @ pH)（温和区复现天然，验证迁移时"保持电荷"能否做到）。

输出（与 ph_scan 同构，供 esmfold_score.py 批量回折 + tm_score.py）：
  {out_dir}/{pdb}/pH{ph}/seqs.fa
  {out_dir}/{pdb}/transfer.json

用法（code/ 下，confumpnn 环境）：
  # 无配体蛋白
  PYTHONPATH=. python tests/transfer_validation.py \
      --pdb ../data/transfer_test/1LYZ.pdb --mode protein \
      --cond_encoder ../output/finetune_v7/finetune_epoch030.pt \
      --out_dir ../output/transfer --n 10 --device cuda:3
  # 有配体蛋白
  PYTHONPATH=. python tests/transfer_validation.py \
      --pdb ../data/transfer_test/4DFR_chainA.pdb --mode ligand \
      --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
      --cond_encoder ../output/finetune_v7/finetune_epoch030.pt \
      --out_dir ../output/transfer --n 10 --device cuda:3
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
    ap.add_argument("--mode", choices=["protein", "ligand"], default="protein")
    ap.add_argument("--weights", default=None,
                    help="backbone（protein 模式默认 MoMPNN；ligand 模式必填原版 LigandMPNN）")
    ap.add_argument("--cond_encoder", required=True)
    ap.add_argument("--pH_list", default="7.4")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed_base", type=int, default=101)
    ap.add_argument("--uncond", action="store_true",
                    help="无条件模式（不注入条件）：作 backbone 基线电荷偏好对照")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--num_ligand_atoms", type=int, default=16)
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.mode == "protein":
        weights = Path(args.weights) if args.weights else DEFAULT_BACKBONE
        model = load_model(str(weights), device, model_type="auto")
        feats = dict(model_type="protein_mpnn", use_atom_context=False,
                     number_of_ligand_atoms=0)
    else:
        if not args.weights:
            raise SystemExit("ligand 模式需要 --weights（原版 LigandMPNN 权重）")
        model = load_model(args.weights, device, model_type="ligand_mpnn")
        feats = dict(model_type="ligand_mpnn", use_atom_context=True,
                     number_of_ligand_atoms=args.num_ligand_atoms)
    enc = load_condition_encoder(args.cond_encoder, device)

    protein_dict, _, _, icodes, _ = parse_PDB(args.pdb)
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32)
    fd = featurize(protein_dict, cutoff_for_score=8.0, **feats)
    L = fd["X"].shape[1]
    fd["batch_size"] = 1
    fd["temperature"] = 0.3
    fd["bias"] = torch.zeros(1, L, 21)
    native = seq_to_string(fd["S"][0].cpu().numpy())

    out_root = Path(args.out_dir) / Path(args.pdb).stem
    print(f"[load] {Path(args.pdb).name} mode={args.mode} L={L} "
          f"native={native[:30]}...", flush=True)

    summary = {"pdb": Path(args.pdb).name, "mode": args.mode, "L": L,
               "n": args.n, "native": native, "pH_arms": {}}

    for ph_s in args.pH_list.split(","):
        pH = float(ph_s)
        q_nat = float(net_charge(native, pH))
        tgt = round(q_nat)
        charges, seqs = [], []
        for k in range(args.n):
            torch.manual_seed(args.seed_base + k)
            fd["randn"] = torch.randn(1, L)
            if args.uncond:
                out = conditioned_sample(model, None, fd, None, device)
            else:
                cond_vec = make_condition_vector(pH, net_charge=tgt)
                out = conditioned_sample(model, enc, fd, cond_vec, device)
            seq = seq_to_string(out["S"][0].cpu().numpy())
            seqs.append(seq)
            charges.append(float(net_charge(seq, pH)))
        mean_c = float(np.mean(charges))
        dev = abs(mean_c - tgt)
        recs = [sum(a == b for a, b in zip(s, native)) / L for s in seqs]
        arm_dir = out_root / f"pH{ph_s}"
        arm_dir.mkdir(parents=True, exist_ok=True)
        fa = arm_dir / "seqs.fa"
        with open(fa, "w") as f:
            for i, (s, c) in enumerate(zip(seqs, charges)):
                f.write(f">seed_{args.seed_base+i} pH={pH} target={tgt:+.0f} "
                        f"charge={c:+.2f}\n{s}\n")
            f.write(f">native pH={pH} charge={q_nat:+.2f}\n{native}\n")
        summary["pH_arms"][ph_s] = {
            "target": tgt, "native_charge": round(q_nat, 2),
            "mean_charge": round(mean_c, 2),
            "std_charge": round(float(np.std(charges)), 2),
            "dev": round(dev, 2), "recovery": round(float(np.mean(recs)), 3),
        }
        print(f"  pH={ph_s:>4} native={q_nat:+6.2f} target={tgt:>3} "
              f"mean={mean_c:+6.2f} dev={dev:.2f} recovery={np.mean(recs):.3f}",
              flush=True)

    with open(out_root / "transfer.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"已写 {out_root}/transfer.json")


if __name__ == "__main__":
    main()
