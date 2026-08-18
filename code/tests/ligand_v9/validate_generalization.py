"""v9 泛化验证：多蛋白 × 多电荷臂 × ligand/protein 双模式采样 + 综合评估。

背景（index/PROJECT_V9_GENERALIZATION_PLAN.md）：验证 v9 编码器在**未见蛋白**上的
泛化能力——电荷控制（H2）、折叠（H1 由 esmfold/tm_score 后处理）、序列合理性，
以及配体上下文对效果的贡献（ligand vs protein 模式对比）。

设计：
  - manifest：JSON 蛋白清单 [{pdb, path, cat}]（pick_validation_pdbs.py 选择，防泄漏）
  - 每个蛋白 5 个电荷臂（pH 固定 7.4，target = native + Δ）：
      arm0 native, arm1 -2, arm2 +2, arm3 -8, arm4 +8
  - 每臂 n=30 序列（seed_base 固定，可复现）
  - 模式：ligand（有配体原子上下文，主实验）/ protein（无配体，消融对照）
  - 评估每序列：实际净电荷、native recovery、配体口袋 recovery、GRAVY
  - 输出每臂 seqs.fa（供 esmfold_score.py 批量回折）+ validation.json

输出结构：
  {out_dir}/{mode}/{pdb}/pH7.4/arm{N}_{tag}/seqs.fa
  {out_dir}/{mode}/{pdb}/validation.json

用法（code/ 下，confumpnn 环境）：
  PYTHONPATH=code python code/tests/ligand_v9/validate_generalization.py \
      --manifest data/validation_pdbs/validation_manifest.json \
      --out_dir output/generalization_v9 --mode both \
      --cond_encoder output/finetune_ligand_v9/finetune_epoch030.pt \
      --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
      --n 30 --device cuda:3 --pH 7.4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parents[2]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR.parent / "LigandMPNN"))

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import load_model, load_condition_encoder, seq_to_string  # noqa: E402

# Kyte-Doolittle 疏水性表（GRAVY）
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
      "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
      "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
      "Y": -1.3, "V": 4.2, "X": 0.0}

# 电荷臂定义（Δ 相对 native 电荷）
ARMS = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]


def gravy(seq):
    return np.mean([KD.get(a, 0.0) for a in seq])


def pocket_residues(protein_dict, cutoff=8.0):
    """配体口袋 = 与配体原子（Y）距离 < cutoff Å 的蛋白残基索引（Cα 计算）。"""
    Y = protein_dict.get("Y")
    X = protein_dict["X"]
    if Y is None or Y.numel() == 0:
        return None
    Yc = Y.reshape(-1, 3).cpu().numpy()          # [n_lig, 3]
    CA = X[:, 1, :].cpu().numpy()                # [L, 3]
    if len(Yc) == 0:
        return None
    d = np.linalg.norm(CA[:, None, :] - Yc[None, :, :], axis=-1)  # [L, n_lig]
    return np.where(d.min(axis=1) < cutoff)[0]


def write_ref_skeleton(protein_dict, path):
    """从 parse_PDB 的 X 坐标写纯蛋白链骨架 PDB（供 TM-score 参考）。"""
    X = protein_dict["X"].cpu().numpy()
    from data_utils import restype_int_to_str
    S = protein_dict["S"].cpu().numpy().reshape(-1)
    lines = []
    atom_i = 1
    for i, aa in enumerate(S):
        res3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
                "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
                "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
                "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
                "X": "UNK"}.get(restype_int_to_str[int(aa)], "UNK")
        for j, name in enumerate(("N", "CA", "C")):
            x = X[i, j]
            lines.append(
                f"ATOM  {atom_i:5d} {name:^4s} {res3:>3s} A{4:4d}    "
                f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}  1.00 20.00          "
                f"{name[0]:>2s}")
            atom_i += 1
    with open(path, "w") as f:
        f.write("REMARK  reference skeleton from parse_PDB (N,CA,C only)\n")
        f.write("\n".join(lines) + "\nEND\n")


def strip_ligands(src, dst):
    """去掉 HETATM（配体/水/离子），保留蛋白 ATOM + 必要记录 → 无配体 PDB。

    protein（消融）模式用：同一 LigandMPNN 模型 + 去配体 PDB 重新特征化，
    use_atom_context=True 时 Y 自动为空 → 模型只凭蛋白骨架生成。
    """
    keep_pfx = ("ATOM", "TER", "END", "MODEL", "HEADER", "TITLE", "CRYST1",
                "SCALE", "REMARK", "COMPND", "SOURCE", "EXPDTA", "AUTHOR")
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            if line.startswith(keep_pfx):
                g.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", default="output/generalization_v9")
    ap.add_argument("--mode", choices=["ligand", "protein", "both"], default="both")
    ap.add_argument("--cond_encoder", required=True)
    ap.add_argument("--weights", default=None,
                    help="ligand 模式必填（LigandMPNN 权重）")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--device", default="cuda:3")
    ap.add_argument("--num_ligand_atoms", type=int, default=16)
    ap.add_argument("--arms", default="native,n2,p2,n8,p8")
    ap.add_argument("--protein_arms", default="native,n8,p8",
                    help="protein（消融）模式跑的臂子集，默认 3 臂")
    ap.add_argument("--start", type=int, default=0,
                    help="从 manifest 第几个蛋白开始（断点续跑）")
    ap.add_argument("--end", type=int, default=None,
                    help="到第几个蛋白结束（None=全部）")
    args = ap.parse_args()

    device = torch.device(args.device)
    arm_map = dict(ARMS)
    sel_arms = [a for a in args.arms.split(",") if a in arm_map]
    prot_arms = [a for a in args.protein_arms.split(",") if a in arm_map]

    # 加载 backbone + 编码器（ligand 与 protein 消融共用同一 LigandMPNN 模型）
    enc = load_condition_encoder(args.cond_encoder, device)
    if not args.weights:
        raise SystemExit("需要 --weights（LigandMPNN 权重）")
    model = load_model(args.weights, device, model_type="ligand_mpnn")

    manifest = json.load(open(args.manifest))
    items = manifest["items"][args.start: args.end]
    print(f"处理 {len(items)} 个蛋白（{args.start}-{args.end or len(manifest['items'])}），"
          f"臂 ligand={sel_arms} protein={prot_arms}，n={args.n}，pH={args.pH}",
          flush=True)

    out_root = Path(args.out_dir)
    for it in items:
        pdb = it["pdb"]
        pdb_path = Path(it["path"])
        protein_dict, *_ = parse_PDB(str(pdb_path))
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        q_nat = float(net_charge(native, args.pH))
        pocket = pocket_residues(protein_dict)
        print(f"\n=== {pdb} cat={it.get('cat')} L={L} native={native[:25]}... "
              f"charge@{args.pH}={q_nat:+.2f} 口袋残基={len(pocket) if pocket is not None else 'N/A'}"
              , flush=True)

        # 去配体版本（protein 消融模式用，同一模型 + 无配体原子上下文）
        noplig_path = pdb_path.with_name(f"{pdb}_noplig{pdb_path.suffix}")
        strip_ligands(str(pdb_path), str(noplig_path))

        # 参考骨架（TM-score 用）
        ref_dir = out_root / "ref"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_pdb = ref_dir / f"{pdb}_ref.pdb"
        write_ref_skeleton(protein_dict, ref_pdb)

        modes = {"ligand": ["ligand"], "protein": ["protein"],
                 "both": ["ligand", "protein"]}[args.mode]
        for mode in modes:
            arms = sel_arms if mode == "ligand" else prot_arms
            # protein 消融：重新 parse 去配体 PDB（Y 为空），ligand 特征化不变
            if mode == "protein":
                protein_dict, *_ = parse_PDB(str(noplig_path))
                if protein_dict["X"].shape[0] != L:
                    raise SystemExit(f"{pdb} protein 模式 L 不一致")
            feats = dict(model_type="ligand_mpnn", use_atom_context=True,
                         number_of_ligand_atoms=args.num_ligand_atoms)
            pocket_cur = None if mode == "protein" else pocket
            protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
            fd = featurize(protein_dict, cutoff_for_score=8.0, **feats)
            fd["batch_size"] = 1
            fd["temperature"] = 0.3
            fd["bias"] = torch.zeros(1, L, 21)
            fd["randn"] = torch.randn(1, L)

            summary = {"pdb": pdb, "cat": it.get("cat"), "L": L, "mode": mode,
                       "native": native, "native_charge": round(q_nat, 2),
                       "arms": {}}
            for arm in arms:
                dq = arm_map[arm]
                tgt = int(round(q_nat)) + dq
                charges, recs, pkt_recs, gravs, seqs = [], [], [], [], []
                for k in range(args.n):
                    torch.manual_seed(args.seed_base + k)
                    fd["randn"] = torch.randn(1, L)
                    cond_vec = make_condition_vector(args.pH, net_charge=tgt)
                    out = conditioned_sample(model, enc, fd, cond_vec, device)
                    seq = seq_to_string(out["S"][0].cpu().numpy())
                    seqs.append(seq)
                    charges.append(float(net_charge(seq, args.pH)))
                    recs.append(sum(a == b for a, b in zip(seq, native)) / L)
                    if pocket_cur is not None and len(pocket_cur):
                        pkt_recs.append(sum(seq[i] == native[i] for i in pocket_cur) / len(pocket_cur))
                    gravs.append(gravy(seq))
                mean_c = float(np.mean(charges))
                dev = abs(mean_c - tgt)
                arm_dir = out_root / mode / pdb / f"pH{args.pH}" / f"arm_{arm}"
                arm_dir.mkdir(parents=True, exist_ok=True)
                fa = arm_dir / "seqs.fa"
                with open(fa, "w") as f:
                    for i, (s, c) in enumerate(zip(seqs, charges)):
                        f.write(f">seed_{args.seed_base+i} arm={arm} target={tgt:+.0f} "
                                f"charge={c:+.2f}\n{s}\n")
                    f.write(f">native charge={q_nat:+.2f}\n{native}\n")
                summary["arms"][arm] = {
                    "target": tgt, "mean_charge": round(mean_c, 2),
                    "std_charge": round(float(np.std(charges)), 2),
                    "dev": round(dev, 2),
                    "recovery": round(float(np.mean(recs)), 3),
                    "pocket_recovery": (round(float(np.mean(pkt_recs)), 3)
                                        if pkt_recs else None),
                    "gravy_mean": round(float(np.mean(gravs)), 3),
                    "n_generated": len(seqs),
                }
                print(f"  [{mode}/{arm}] target={tgt:>4} mean={mean_c:+6.2f} "
                      f"dev={dev:.2f} rec={np.mean(recs):.3f} "
                      f"pkt={np.mean(pkt_recs) if pkt_recs else float('nan'):.3f}"
                      f" gravy={np.mean(gravs):.3f}", flush=True)
            with open(out_root / mode / pdb / "validation.json", "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  已写 {pdb} 结果", flush=True)

    print("\n=== 全部完成 ===", flush=True)


if __name__ == "__main__":
    main()
