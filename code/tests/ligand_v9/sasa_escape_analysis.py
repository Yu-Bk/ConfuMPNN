"""SASA 监督逃逸分析：深部口袋（frac_sasa<0.25）的带电残基是否被删减。

背景（2026-09-01 配体迁移删减机制）：
  v12 的 surface_composition/gravy/charge_target 三个损失都只在表面残基
  （frac_sasa >= surface_threshold，默认 0.25）上计算。配体结合口袋是蛋白表面
  的深凹陷，60-75% 口袋残基 frac_sasa < 0.25 → 被划入"核心"（非表面）→ 不受
  组成监督。本脚本量化：口袋残基的 frac_sasa 分布 + 深部口袋带电残基的删减
  倍率，验证"监督看不见的地方删得最狠"。

用法（项目根）：
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/ligand_v9/sasa_escape_analysis.py
输出：逐蛋白表（口袋非表面占比 / 深部带电删减 / 全序列删减）
"""
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))
sys.path.insert(0, str(_PROJECT_DIR / "code"))
sys.path.insert(0, str(_PROJECT_DIR))
from src.sasa import fractional_sasa  # noqa: E402

CHARGED = "DEKR"
# 删减最严重的配体蛋白（2026-09-01 实测），可改 --prots
PROTS = ["2FEO", "1AS2", "1AXW", "5CQH", "1CGE"]
GEN_ROOT = _PROJECT_DIR / "output/generalization_ligand_v12_2" / "ligand"


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


def read_gen(fa):
    lines = open(fa).read().splitlines()
    seqs, cur = [], None
    for line in lines:
        if line.startswith(">"):
            if cur:
                seqs.append(cur)
            cur = (line[1:], "")
        elif cur:
            cur = (cur[0], cur[1] + line)
    if cur:
        seqs.append(cur)
    gen = [s for n, s in seqs if not n.startswith("native")]
    nat = [s for n, s in seqs if n.startswith("native")]
    return gen, (nat[0] if nat else None)


def charged_in(seq, idx):
    return sum(1 for i in idx if seq[i] in CHARGED)


def main():
    print(f"{'蛋白':6s} {'口袋res':>5s} {'口袋非表面占比':>8s} {'非口袋非表面':>8s} "
          f"{'口袋带电(nat/gen)':>14s} {'其中深部(nat/gen)':>13s} {'深部倍率':>6s}")
    for name in PROTS:
        pdb_path = f"data/validation_pdbs/{name}.pdb"
        from data_utils import parse_PDB
        protein_dict, *_ = parse_PDB(pdb_path)
        pocket = pocket_residues(protein_dict)
        s = fractional_sasa(pdb_path, align_to_full=True)
        frac = np.array(s["frac_sasa"])
        L = len(frac)
        all_idx = np.arange(L)
        nonpkt = np.setdiff1d(all_idx, pocket)
        pkt_ns = np.mean(frac[pocket] < 0.25)
        nonpkt_ns = np.mean(frac[nonpkt] < 0.25)
        native = s["seq"]
        fa = GEN_ROOT / name / "pH7.4" / "arm_native" / "seqs.fa"
        if not fa.exists():
            print(f"  !! {name} 无 {fa}")
            continue
        gen, _ = read_gen(fa)
        deep_pkt = pocket[frac[pocket] < 0.25]
        nat_pkt_ch = charged_in(native, pocket)
        nat_deep_ch = charged_in(native, deep_pkt)
        g_pkt_ch = np.mean([charged_in(seq, pocket) for seq in gen])
        g_deep_ch = np.mean([charged_in(seq, deep_pkt) for seq in gen])
        r_deep = g_deep_ch / nat_deep_ch if nat_deep_ch > 0 else float("nan")
        print(f"{name:6s} {len(pocket):5d} {pkt_ns:8.2f} {nonpkt_ns:8.2f} "
              f"{nat_pkt_ch:3d}/{g_pkt_ch:5.1f} {nat_deep_ch:3d}/{g_deep_ch:5.1f} {r_deep:6.2f}")


if __name__ == "__main__":
    main()
