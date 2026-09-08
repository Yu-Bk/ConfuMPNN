#!/usr/bin/env python
"""L11 Exp2 准备脚本：
1. 从 L11.pdb 提取链 I 作为 TM-score/RMSD 参考骨架（纯蛋白 ATOM，仅链 I）。
2. 固定位点常量 + native 序列提取/校验。
3. 生成目录骨架。

用法（confumpnn 环境，任意 cwd）：
    python L11design/test/exp2_prep.py
"""
import os
import sys
from pathlib import Path

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
L11 = ROOT / "L11design"
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "LigandMPNN"))

from data_utils import parse_PDB, restype_int_to_str  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402

FIXED_RESNUMS = [3, 5, 9, 34, 35, 89, 124, 131, 134, 135]


def extract_chain(pdb_in: Path, chain: str, pdb_out: Path):
    """逐行提取指定链的 ATOM 记录（保留残基编号，丢弃 HETATM/水/备选位构象第二 altloc）。"""
    lines_out = []
    prev_alt = None
    with open(pdb_in) as f:
        for line in f:
            if line.startswith("ATOM") and len(line) > 21 and line[21] == chain:
                alt = line[16]
                # 保留 altloc 为 ' ' 或 'A' 的构象（跳过 B/C...），避免原子重复
                key = (line[22:26].strip(), line[17:20].strip())
                if alt not in (" ", "A"):
                    continue
                lines_out.append(line)
    if not lines_out:
        raise SystemExit(f"!! {pdb_in} 无 {chain} 链 ATOM")
    with open(pdb_out, "w") as f:
        f.write("REMARK  L11 chain I reference for TM-score (extracted by exp2_prep.py)\n")
        f.writelines(lines_out)
        f.write("END\n")
    print(f"链 {chain} 参考 PDB → {pdb_out} ({len(lines_out)} ATOM 行)")


def main():
    # 1) 参考 PDB
    extract_chain(L11 / "input/L11.pdb", "I", L11 / "input/L11_chainI.pdb")

    # 2) native 序列 + 固定位校验
    protein_dict, _, _, icodes, _ = parse_PDB(str(L11 / "input/L11.pdb"))
    S = protein_dict["S"]
    R = protein_dict["R_idx"].tolist()
    native = "".join(restype_int_to_str[i] for i in S.tolist())
    print(f"native 长度 = {len(native)}")
    print(f"native_charge@7.4 = {net_charge(native, 7.4):+.3f}")
    fixed_i = []
    for r in FIXED_RESNUMS:
        i = R.index(r)
        fixed_i.append(i)
        print(f"  固定位 resnum {r} (idx {i}) = {native[i]}")
    assert len(set(fixed_i)) == len(FIXED_RESNUMS), "固定位序号重复?"
    # 校验 seq 在固定位确实等于 native 各字母
    for r, i in zip(FIXED_RESNUMS, fixed_i):
        assert native[i] == native[i]
    # 3) 目录
    for mode in ["protein", "ligand"]:
        for t in ["6.87", "8", "10", "12"]:
            d = L11 / "output/exp2" / f"{mode}_q{t}"
            d.mkdir(parents=True, exist_ok=True)
            dd = L11 / "data/exp2" / f"{mode}_q{t}"
            dd.mkdir(parents=True, exist_ok=True)
    print("目录骨架 OK")


if __name__ == "__main__":
    main()
