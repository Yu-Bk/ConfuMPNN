"""配体模式 vs mompnn：口袋（配体 8Å 内）与非口袋的带电残基删减对比。

背景（2026-09-01 配体迁移组成分析）：
  配体模式 native 臂 8/10 蛋白带电残基总数系统性删减（0.53-0.65×），且删减
  偏向配体口袋（配体相互作用位点）。mompnn 删减全局均匀。本脚本量化口袋 vs
  非口袋的删减差异，支持对比配体模式与 mompnn 两个生成目录。

口径：
  - 口袋 = 配体原子 8Å 内残基（Cα 距离，与 validate_generalization.pocket_residues 一致）
  - 带电残基 = D/E + K/R（与 compare_comp_v12_2 / compare_comp_ligand 口径一致）
  - 生成序列取 target=native 臂（arm_native）seqs.fa，native 从 fasta 的 >native 行读
  - 倍率 = 生成均值 / native 计数

用法（项目根）：
  # 配体模式
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/ligand_v9/pocket_comp_compare.py \
      --gen-root output/generalization_ligand_v12_2/ligand
  # mompnn 对照
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/ligand_v9/pocket_comp_compare.py \
      --gen-root output/generalization_v12_2_calib/protein
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))
sys.path.insert(0, str(_PROJECT_DIR / "code"))
from data_utils import parse_PDB  # noqa: E402

CHARGED = "DEKR"


def pocket_residues(protein_dict, cutoff=8.0):
    """配体口袋 = 与配体原子（Y）距离 < cutoff Å 的残基索引（Cα 计算）。"""
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


def read_seqfa(fa):
    """读 seqs.fa：返回 (生成序列列表, native 序列)。"""
    lines = open(fa).read().splitlines()
    seqs, cur = [], None
    for line in lines:
        if line.startswith(">"):
            if cur is not None:
                seqs.append(cur)
            cur = (line[1:], "")
        elif cur is not None:
            cur = (cur[0], cur[1] + line)
    if cur is not None:
        seqs.append(cur)
    gen = [s for n, s in seqs if not n.startswith("native")]
    nat = [s for n, s in seqs if n.startswith("native")]
    return gen, (nat[0] if nat else None)


def charged_in(seq, idx=None):
    if idx is not None:
        return sum(1 for i in idx if seq[i] in CHARGED)
    return sum(1 for a in seq if a in CHARGED)


def main():
    ap = argparse.ArgumentParser(description="口袋 vs 非口袋带电残基删减对比")
    ap.add_argument("--gen-root", required=True,
                    help="生成序列根目录（ligand 或 protein 模式，含 <pdb>/pH7.4/arm_native/seqs.fa）")
    ap.add_argument("--manifest", default=str(_PROJECT_DIR / "data/validation_pdbs/validation_manifest.json"))
    ap.add_argument("--pdb-dir", default=str(_PROJECT_DIR / "data/validation_pdbs"))
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    gen_root = Path(args.gen_root)
    print(f"{'蛋白':7s} {'口袋res':>6s} {'native口袋带':>8s} {'gen口袋带':>8s} {'口袋倍率':>6s} "
          f"{'native全带':>7s} {'gen全带':>6s} {'全倍率':>6s} {'非口袋倍率':>8s}")
    for it in man["items"]:
        name = it["pdb"]
        protein_dict, *_ = parse_PDB(f"{args.pdb_dir}/{name}.pdb")
        pocket = pocket_residues(protein_dict)
        fa = gen_root / name / "pH7.4" / "arm_native" / "seqs.fa"
        if not fa.exists():
            print(f"  !! {name} 无 {fa}")
            continue
        gen, native = read_seqfa(fa)
        if not gen or not native:
            print(f"  !! {name} 序列读取失败")
            continue
        L = len(native)
        all_idx = np.arange(L)
        nonpkt = np.setdiff1d(all_idx, pocket) if pocket is not None else all_idx
        nat_p = charged_in(native, pocket) if pocket is not None else 0
        nat_all = charged_in(native)
        nat_np = nat_all - nat_p
        g_p = np.mean([charged_in(s, pocket) for s in gen]) if pocket is not None else 0.0
        g_all = np.mean([charged_in(s) for s in gen])
        g_np = np.mean([charged_in(s, nonpkt) for s in gen])
        r_p = g_p / nat_p if nat_p > 0 else float("nan")
        r_a = g_all / nat_all if nat_all > 0 else float("nan")
        r_np = g_np / nat_np if nat_np > 0 else float("nan")
        print(f"{name:7s} {len(pocket) if pocket is not None else 0:6d} {nat_p:8d} {g_p:8.1f} "
              f"{r_p:6.2f} {nat_all:7d} {g_all:6.1f} {r_a:6.2f} {r_np:8.2f}")


if __name__ == "__main__":
    main()
