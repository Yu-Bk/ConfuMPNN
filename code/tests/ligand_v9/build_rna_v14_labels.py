"""RNA/DNA 结合蛋白拆链样本 → labels_rna_v14.npz（v14 配体数据扩充）。

对 data/ligand_train/rna_pdbs/*.pdb（拆链单蛋白+RNA/DNA 配体）逐域：
  - parse_PDB → seq / Cα coords / L
  - 序列去重（同源核糖体蛋白跨结构重复，histone 等）→ 每个唯一序列保留一条
  - 8 pH 扰动（uniform 4-10，同 build_ligand_labels）+ net_charge + pI
  - 排除与现有 labels.npz 完全相同的序列（防与旧小分子集重复）

输出：
  data/ligand_train/labels_rna_v14.npz（domain_ids/seqs/coords/pH/charge/pI，N×8）
  data/ligand_train/rna_v14_manifest.json（含每域来源/L/去重信息）

用法（confumpnn 环境）：
  PYTHONPATH=code:code/tests python code/tests/ligand_v9/build_rna_v14_labels.py \
      --dompdb data/ligand_train/rna_pdbs --out data/ligand_train/labels_rna_v14.npz
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[2]
for p in (str(_CODE_DIR), str(_CODE_DIR.parent / "LigandMPNN"), str(_CODE_DIR / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import parse_PDB  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.isoelectric_point import find_pI  # noqa: E402


def seq_from_S(S_int):
    from data_utils import restype_int_to_str
    return "".join(restype_int_to_str[int(x)] for x in S_int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dompdb", default="data/ligand_train/rna_pdbs")
    ap.add_argument("--out", default="data/ligand_train/labels_rna_v14.npz")
    ap.add_argument("--manifest", default="data/ligand_train/rna_v14_manifest.json")
    ap.add_argument("--n_pH", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=500)
    ap.add_argument("--min_len", type=int, default=20)
    ap.add_argument("--old_labels", default="data/ligand_train/labels.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dompdb, "*.pdb")))
    print(f"候选拆链样本 {len(files)}", flush=True)

    # 现有训练集序列（防重复）
    old_seqs = set()
    if os.path.exists(args.old_labels):
        old = np.load(args.old_labels, allow_pickle=True)
        old_seqs = {str(s) for s in old["seqs"]}
        print(f"现有训练集序列 {len(old_seqs)}（用于去重）", flush=True)

    rng = np.random.RandomState(args.seed)
    seen_seq = {}
    manifest = []
    n_ok, n_skip = 0, 0
    for i, p in enumerate(files):
        pid = os.path.basename(p)
        try:
            protein_dict, *_ = parse_PDB(p, device="cpu", parse_all_atoms=False)
            L = protein_dict["X"].shape[0]
            if not (args.min_len <= L <= args.max_len):
                raise ValueError(f"L={L} 超范围")
            S = protein_dict["S"].reshape(-1).cpu().numpy()
            seq = seq_from_S(S)
            if "X" in seq or len(seq) < args.min_len:
                raise ValueError("序列含 X 或过短")
            if int(protein_dict["mask"].sum()) < 0.9 * L:
                raise ValueError(f"mask 不完整 {int(protein_dict['mask'].sum())}/{L}")
            if seq in old_seqs:
                raise ValueError("与现有训练集序列重复")
            coords = protein_dict["X"][:, 1, :].cpu().numpy()  # [L,3] Cα
            # 序列去重（同源核糖体蛋白跨结构重复 / 核小体组蛋白多拷贝）
            if seq in seen_seq:
                seen_seq[seq]["dup_of"].append(pid)
                manifest.append({"domain": pid, "L": L, "dup": True, "dup_of": seen_seq[seq]["first"]})
                n_skip += 1
                continue
            seen_seq[seq] = {"first": pid, "L": L, "dup_of": []}
            pH_i = rng.uniform(4.0, 10.0, args.n_pH)
            charge_i = np.array([net_charge(seq, ph) for ph in pH_i], dtype=np.float32)
            pI = find_pI(seq)
            manifest.append({"domain": pid, "L": L, "dup": False,
                             "seq_len": len(seq),
                             "n_neg": seq.count("D") + seq.count("E"),
                             "n_pos": seq.count("K") + seq.count("R"),
                             "pI": float(pI),
                             "charge_at74": float(net_charge(seq, 7.4))})
            # 追加到全局数组
            _acc["domains"].append(pid)
            _acc["seqs"].append(seq)
            _acc["coords"].append(coords)
            _acc["pHs"].append(pH_i)
            _acc["charges"].append(charge_i)
            _acc["pIs"].append(np.full(args.n_pH, pI, dtype=np.float32))
            n_ok += 1
        except Exception as e:
            n_skip += 1
            manifest.append({"domain": pid, "dup": False, "err": str(e)[:120]})
            if n_skip <= 8:
                print(f"  ⚠️ 跳过 {pid}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"  标签 {n_ok}/{i+1}（跳过 {n_skip}）", flush=True)

    np.savez(args.out, domain_ids=np.array(_acc["domains"]),
             seqs=np.array(_acc["seqs"], dtype=object),
             coords=np.array(_acc["coords"], dtype=object),
             pH=np.concatenate(_acc["pHs"]), charge=np.concatenate(_acc["charges"]),
             pI=np.concatenate(_acc["pIs"]))
    json.dump(manifest, open(args.manifest, "w"), indent=1)
    print(f"已写 {args.out}（{n_ok} 域 × {args.n_pH} pH = {n_ok*args.n_pH} 样本，跳过 {n_skip}）",
          flush=True)


# 全局累积（避免函数内闭包复杂化）
_acc = {"domains": [], "seqs": [], "coords": [], "pHs": [], "charges": [], "pIs": []}


if __name__ == "__main__":
    main()
