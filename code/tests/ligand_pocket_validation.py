"""配体结合位点保持验证：原始 LigandMPNN vs 条件化（v7 编码器）在配体复合物上。

背景：用户要求"维持 LigandMPNN 的配体结合能力"。本脚本验证条件化改造
（冻结 backbone + cross-attention 旁路注入 pH/电荷条件）是否破坏了模型在
配体结合位点（配体 5A 内残基）的设计能力。

三组对照（同一 LigandMPNN backbone，同 seed 解码顺序）：
  uncond       : 无条件采样（原始 LigandMPNN，结合位点 recovery 基线）
  cond_native  : 条件化 + target=天然电荷@pH  → 条件化是否降低结合位点 recovery
  cond_neg5    : 条件化 + target=天然-5       → 电荷改写是否破坏结合位点

输出：每组全局 recovery / 结合位点 recovery / 电荷命中 / 结合位点定义。

用法（code/ 下）：
  PYTHONPATH=. python tests/ligand_pocket_validation.py \
      --pdb ../data/ligand_test/1FQG.pdb \
      --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
      --cond_encoder ../output/finetune_v7/finetune_epoch030.pt \
      --out_dir ../output/ligand_pocket_1FQG --n 20 --device cuda:3
"""
import argparse
import json
import os
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

# 排除的 HETATM（水/单原子离子/常见结晶溶剂）
EXCLUDE_HET = {
    "HOH", "WAT", "DOD", "NA", "CL", "CA", "ZN", "MG", "MN", "FE", "CO",
    "CU", "K", "EDO", "ACT", "GOL", "DMS", "SO4", "PO4", "CIT", "TRS",
    "PEG", "1PE", "EPE", "FMT", "GLC", "BME",
}


def ligand_atoms(pdb_path):
    """从 PDB 提取非溶剂配体原子坐标。返回 [(resname, xyz)]。"""
    atoms = []
    for line in open(pdb_path):
        if line.startswith("HETATM") and len(line) >= 54:
            res = line[17:20].strip()
            if res in EXCLUDE_HET:
                continue
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            atoms.append((res, xyz))
    return atoms


def pocket_residues(pd, cutoff=5.0):
    """配体原子 cutoff 埃内残基索引（按 Cα 距离）。

    pd: parse_PDB 输出的 protein_dict。
       X = [L, 4, 3] 主链原子 N/CA/C/O，Cα = X[:, 1, :]
       Y = [N_lig, 3] 配体原子坐标（非水 HETATM，parse_PDB 已分离水）
    """
    CA = pd["X"].cpu().numpy()[:, 1, :]  # [L, 3] Cα
    Y = pd["Y"].cpu().numpy()            # [N_lig, 3]
    if len(Y) == 0 or not np.any(Y):
        return []
    idx = []
    for i in range(len(CA)):
        d = float(np.min(np.linalg.norm(CA[i] - Y, axis=1)))
        if d <= cutoff:
            idx.append(i)
    return idx


def recovery(seq, native, idx):
    """seq/native 在 idx 位置逐位一致比例（idx 为空返回 None）。"""
    if not idx:
        return None
    return sum(seq[i] == native[i] for i in idx) / len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--weights", required=True, help="原版 LigandMPNN 权重（含配体上下文）")
    ap.add_argument("--cond_encoder", default=None, help="条件编码器（None = 无条件）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed_base", type=int, default=111)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--cutoff", type=float, default=5.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.weights, device, model_type="ligand_mpnn")
    enc = (load_condition_encoder(args.cond_encoder, device)
           if args.cond_encoder else None)

    protein_dict, _, _, icodes, _ = parse_PDB(args.pdb)
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32)
    fd = featurize(protein_dict, cutoff_for_score=8.0, use_atom_context=True,
                   number_of_ligand_atoms=16, model_type="ligand_mpnn")
    L = fd["X"].shape[1]
    fd["batch_size"] = 1
    fd["temperature"] = 0.3
    fd["bias"] = torch.zeros(1, L, 21)

    native = seq_to_string(fd["S"][0].cpu().numpy())
    native_charge = float(net_charge(native, args.pH))
    pocket = pocket_residues(protein_dict, args.cutoff)
    n_pocket = len(pocket)
    print(f"[load] L={L} native_charge@7.4={native_charge:+.2f} "
          f"结合位点({args.cutoff}A内) {n_pocket} 残基", flush=True)
    print(f"  配体: {[r for r in set(r for r,_ in ligand_atoms(args.pdb))]}", flush=True)

    scenarios = {
        "uncond": (None, None, "无条件（原始 LigandMPNN）"),
        "cond_native": (enc, native_charge, "条件化 target=native"),
        "cond_neg5": (enc, native_charge - 5, "条件化 target=native-5"),
    }

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"pdb": os.path.basename(args.pdb), "L": L,
               "native_charge": native_charge, "pocket_n": n_pocket,
               "cutoff": args.cutoff, "arms": {}}

    for name, (encoder, tgt, desc) in scenarios.items():
        charges, seqs, rec_all, rec_pocket = [], [], [], []
        for k in range(args.n):
            torch.manual_seed(args.seed_base + k)
            fd["randn"] = torch.randn(1, L)
            if encoder is None:
                out = conditioned_sample(model, None, fd, None, device)
            else:
                cond_vec = make_condition_vector(args.pH, net_charge=tgt)
                out = conditioned_sample(model, encoder, fd, cond_vec, device)
            seq = seq_to_string(out["S"][0].cpu().numpy())
            seqs.append(seq)
            charges.append(float(net_charge(seq, args.pH)))
            rec_all.append(sum(a == b for a, b in zip(seq, native)) / L)
            rec_pocket.append(recovery(seq, native, pocket))
        mean_c = float(np.mean(charges))
        dev = abs(mean_c - tgt) if tgt is not None else None
        summary["arms"][name] = {
            "desc": desc, "target": tgt,
            "mean_charge": round(mean_c, 2), "std_charge": round(float(np.std(charges)), 2),
            "dev": round(dev, 2) if dev is not None else None,
            "global_recovery": round(float(np.mean(rec_all)), 3),
            "pocket_recovery": (round(float(np.mean(rec_pocket)), 3)
                                if n_pocket else None),
            "pocket_recovery_std": (round(float(np.std(rec_pocket)), 3)
                                    if n_pocket else None),
        }
        print(f"  {name:12s} {desc:<28s} charge={mean_c:+5.2f}±{np.std(charges):.2f} "
              f"globRec={np.mean(rec_all):.3f} pocketRec={np.mean(rec_pocket):.3f}"
              if rec_pocket else
              f"  {name:12s} {desc:<28s} charge={mean_c:+5.2f}  globRec={np.mean(rec_all):.3f}",
              flush=True)
        # 写序列
        fa = out_root / f"{name}_seqs.fa"
        with open(fa, "w") as f:
            for i, (s, c) in enumerate(zip(seqs, charges)):
                f.write(f">sample_{i+1} charge={c:+.2f}\n{s}\n")

    with open(out_root / "pocket_validation.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n已写 {out_root / 'pocket_validation.json'}")


if __name__ == "__main__":
    main()
