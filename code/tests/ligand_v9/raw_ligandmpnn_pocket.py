"""原始 LigandMPNN（未微调）口袋带电残基倾向测试。

背景（2026-09-01 配体迁移删减机制）：
  要区分"疏水口袋倾向"是 LigandMPNN 模型固有先验还是 v12 微调放大。本脚本加载
  原始 LigandMPNN 权重（ligandmpnn_v_32_010_25.pt），以**无条件模式**
  （conditioned_sample enc=None，不注入电荷条件）对带配体蛋白生成序列，统计
  口袋（配体 8Å 内）与全序列的带电残基数 vs native。

结论（2026-09-01 实测）：原始 LigandMPNN 口袋带电残基 0.78-0.90（温和删 10-22%），
  全序列 0.87-1.30（不删甚至增加）→ 系统性删减（全局 0.53-0.65 + 口袋 0.23-0.54）
  是 v12 微调放大/引入，非模型固有通病。

用法（项目根）：
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/ligand_v9/raw_ligandmpnn_pocket.py \
      --device cuda:4 --n 20
输出：逐蛋白表（native 口袋带电 / 原始 LigandMPNN 口袋 / 全序列）
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))
sys.path.insert(0, str(_PROJECT_DIR / "code"))
from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from run_guided import load_model, seq_to_string  # noqa: E402

CHARGED = "DEKR"
PROTS = ["2FEO", "1AS2", "1AXW", "1CGE", "1BJ4", "1C6O"]


def pocket_residues(protein_dict, cutoff=8.0):
    Y = protein_dict.get("Y")
    X = protein_dict["X"]
    if Y is None or Y.numel() == 0:
        return None
    Yc = Y.reshape(-1, 3).cpu().numpy()
    CA = X[:, 1, :].cpu().numpy()
    if len(Yc) == 0:
        return None
    d = np.linalg.norm(CA[:, None, :] - Yc[None, :, :], axis=-1)
    return np.where(d.min(axis=1) < cutoff)[0]


def main():
    ap = argparse.ArgumentParser(description="原始 LigandMPNN 口袋带电残基测试")
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--prots", default=",".join(PROTS))
    args = ap.parse_args()

    device = torch.device(args.device)
    prots = [p for p in args.prots.split(",") if p.strip()]
    print("加载原始 LigandMPNN 权重（未微调，无条件生成）...", flush=True)
    model = load_model(str(_PROJECT_DIR / "LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"),
                       device, model_type="ligand_mpnn")

    print(f"{'蛋白':6s} {'native口袋带':>8s} {'原始LigandMPNN口袋':>13s} {'口袋倍率':>6s} "
          f"{'native全带':>7s} {'原始全带':>7s} {'全倍率':>6s}")
    for name in prots:
        protein_dict, *_ = parse_PDB(f"data/validation_pdbs/{name}.pdb")
        pocket = pocket_residues(protein_dict)
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
        fd = featurize(protein_dict, cutoff_for_score=8.0, model_type="ligand_mpnn",
                       use_atom_context=True, number_of_ligand_atoms=25)
        fd["batch_size"] = 1
        fd["temperature"] = args.temperature
        fd["bias"] = torch.zeros(1, L, 21)
        cond_vec = make_condition_vector(7.4, net_charge=0.0)  # 占位（enc=None 不注入）
        seqs = []
        for k in range(args.n):
            torch.manual_seed(args.seed + k)
            fd["randn"] = torch.randn(1, L)
            out = conditioned_sample(model, None, fd, cond_vec, device)
            seqs.append(seq_to_string(out["S"][0].cpu().numpy()))

        def ch(seq, idx=None):
            if idx is not None:
                return sum(1 for i in idx if seq[i] in CHARGED)
            return sum(1 for a in seq if a in CHARGED)

        nat_p = ch(native, pocket)
        nat_all = ch(native)
        g_p = np.mean([ch(s, pocket) for s in seqs])
        g_all = np.mean([ch(s) for s in seqs])
        rp = g_p / nat_p if nat_p > 0 else float("nan")
        ra = g_all / nat_all if nat_all > 0 else float("nan")
        print(f"{name:6s} {nat_p:8d} {g_p:13.1f} {rp:6.2f} {nat_all:7d} {g_all:7.1f} {ra:6.2f}",
              flush=True)


if __name__ == "__main__":
    main()
