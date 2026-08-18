"""v9 配体训练标签构建：五类配体复合物 → labels.npz（对齐 build_labels_v2 格式）。

背景（index/PROJECT_V9_LIGAND_PLAN.md）：v9 在 LigandMPNN backbone 上重训条件编码器，
需要配体复合物的 (pH, net_charge) 标签。电荷是序列的物理属性，与 backbone 无关，
复用 build_labels_v2 的逻辑（parse → net_charge@8 pH → find_pI）。

与 CATH 版的差异：
  - domain_ids = **完整文件名（含 .pdb/.cif 后缀）**，训练脚本 basename 匹配 all_pdb/
  - 从配体复合物提取蛋白序列（配体原子不进入序列，parse_PDB 的 S 只含 protein）
  - 用 parse_PDB（prody）提取 Cα 坐标 + 序列（与训练脚本一致，确保标签-特征对齐）

用法（confumpnn 环境，需 LigandMPNN/data_utils）：
  PYTHONPATH=code:code/tests python code/tests/build_ligand_labels.py \
      --dompdb data/ligand_train/all_pdb --out data/ligand_train/labels.npz
输出：
  data/ligand_train/labels.npz（domain_ids/seqs/coords/pH/charge/pI）
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

# 本脚本在 code/tests/ligand_v9/ → parents[2] = code/
_CODE_DIR = Path(__file__).resolve().parents[2]
for p in (str(_CODE_DIR), str(_CODE_DIR.parent / "LigandMPNN"), str(_CODE_DIR / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import parse_PDB  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.isoelectric_point import find_pI  # noqa: E402

RESTYPE3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}


def seq_from_S(S_int):
    """parse_PDB 返回的 S（restype_int_to_str 序号）→ 1-letter 序列。

    data_utils.restype_int_to_str：0→A ... 19→Y, 20→X。S 序号直接映射。
    """
    from data_utils import restype_int_to_str
    return "".join(restype_int_to_str[int(x)] for x in S_int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dompdb", default="data/ligand_train/all_pdb")
    ap.add_argument("--out", default="data/ligand_train/labels.npz")
    ap.add_argument("--n_pH", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=500,
                    help="最大残基数（分类已滤 L≤500）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", default="1mbn,4dfr,1fqg,5hvx,3t0f",
                    help="逗号分隔 PDB 前缀排除（验证蛋白防泄漏）")
    args = ap.parse_args()
    exclude_pfx = [s.strip().lower() for s in args.exclude.split(",") if s.strip()]

    files = [p for p in glob.glob(os.path.join(args.dompdb, "*"))
             if os.path.isfile(p) and p.endswith((".pdb", ".cif"))]
    files = [f for f in files
             if not any(os.path.basename(f).lower().startswith(x) for x in exclude_pfx)]
    print(f"候选配体复合物 {len(files)}（排除验证蛋白后）", flush=True)

    random = np.random.RandomState(args.seed)
    domains, seqs, coords_all, pHs, charges, pIs = [], [], [], [], [], []
    n_ok, n_skip = 0, 0
    for i, p in enumerate(files):
        try:
            protein_dict, *_ = parse_PDB(p, device="cpu", parse_all_atoms=False)
            L = protein_dict["X"].shape[0]
            if L <= 0 or L > args.max_len:
                raise ValueError(f"L={L} 超出范围")
            S = protein_dict["S"].reshape(-1).cpu().numpy()
            seq = seq_from_S(S)
            if "X" in seq or len(seq) < 20:
                raise ValueError("序列含 X 或过短")
            coords = protein_dict["X"][:, 1, :].cpu().numpy()  # [L,3] Cα
            pH_i = random.uniform(4.0, 10.0, args.n_pH)
            charge_i = np.array([net_charge(seq, ph) for ph in pH_i], dtype=np.float32)
            pI = find_pI(seq)
            domains.append(os.path.basename(p))
            seqs.append(seq)
            coords_all.append(coords)
            pHs.append(pH_i)
            charges.append(charge_i)
            pIs.append(np.full(args.n_pH, pI, dtype=np.float32))
            n_ok += 1
        except Exception as e:
            n_skip += 1
            if n_skip <= 5:
                print(f"  ⚠️ 跳过 {os.path.basename(p)}: {e}", flush=True)
        if (i + 1) % 500 == 0:
            print(f"  标签 {n_ok}/{i+1}（跳过 {n_skip}）", flush=True)

    np.savez(args.out, domain_ids=np.array(domains), seqs=np.array(seqs, dtype=object),
             coords=np.array(coords_all, dtype=object), pH=np.concatenate(pHs),
             charge=np.concatenate(charges), pI=np.concatenate(pIs))
    print(f"已写 {args.out}（{n_ok} 复合物 × {args.n_pH} pH = {n_ok*args.n_pH} 样本，"
          f"跳过 {n_skip}）", flush=True)


if __name__ == "__main__":
    main()
