"""逐残基 fractional SASA 计算（v3 方案 D3 / D10 / B 表面电荷监督依赖）。

用途：
  1. **暴露度指标（D3）**：fractional SASA = 残基 SASA / 参考 SASA（Gly-X-Gly 三肽
     内建参考，freesasa 的 relativeTotal）。这是标准暴露度口径，替代早期 10Å 邻居数近似。
  2. **v10 B 表面添加电荷监督（L_add）**：表面资格硬门槛（fracSASA ≥ θ 才计入 L_add），
     埋藏位 fracSASA≈0 → 权重≈0。
  3. **D10 SASA 旁路注入 h_V**（二阶段，不动 backbone）：逐残基 [L] 特征 → 小投影层。

实现：
  - Bio.PDB 解析结构 → freesasa.structureFromBioPDB（重原子、VdW 半径）→ calc → residueAreas
  - 每残基取 relativeTotal（freesasa 内建参考分母，即 Gly-X-Gly 扩展值）
  - 只统计标准氨基酸残基（跳过核酸/配体/水）

依赖：freesasa（pip install freesasa）、biopython。都在 confumpnn 环境。
"""

from pathlib import Path

import numpy as np

# 标准 20 氨基酸（判断哪些 ATOM 是蛋白残基，跳过核酸/配体）
_AA20 = set("ACDEFGHIKLMNPQRSTVWY")
# 蛋白链残基的 3 字母 → 1 字母（freesasa residueType 是 3 字母名）
_RES3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def fractional_sasa(pdb_path, surface_threshold=0.25):
    """计算 PDB 的逐残基 fractional SASA。

    参数:
        pdb_path: PDB 文件路径（蛋白结构，可含配体/核酸，会跳过）
        surface_threshold: 表面资格阈值 θ；fracSASA ≥ θ 视为"表面位点"
            （v10 B 的 L_add 表面硬门槛，θ 由配置定，默认 0.25）

    返回:
        dict:
            seq: 蛋白序列（单字母，只含标准氨基酸，沿链顺序）
            frac_sasa: [L] float 数组，逐残基 fractional SASA
            surface_mask: [L] bool，fracSASA ≥ θ 的表面位点掩码
            is_surface: 表面位点数量
    抛出:
        RuntimeError: 无蛋白残基 或 freesasa 不可用
    """
    try:
        import freesasa
    except ImportError:
        raise RuntimeError("需要 freesasa（pip install freesasa），confumpnn 环境已装")

    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))

    seq = []
    fracs = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res3 = residue.get_resname().strip()
                aa = _RES3TO1.get(res3)
                if aa is None:
                    continue  # 跳过核酸/配体/水
                if not residue.has_id("CA"):
                    continue  # 无 Cα（如异常残基），跳过
                seq.append(aa)
                # 占位：fractional SASA 稍后从 freesasa 按 (chain, resid) 填
                fracs.append(None)

    if not seq:
        raise RuntimeError(f"{pdb_path} 无蛋白残基")

    # freesasa 全结构计算 → 逐残基 relativeTotal
    try:
        fs = freesasa.structureFromBioPDB(structure)
        result = freesasa.calc(fs)
        residue_areas = result.residueAreas()  # {chain: {resnum: ResidueArea}}
    except Exception as e:
        raise RuntimeError(f"freesasa 计算失败: {e}")

    # 第二次遍历，把 freesasa 的 relativeTotal 填进 fracs（按链+残基号匹配）
    idx = 0
    for model in structure:
        for chain in model:
            chain_id = chain.id
            chain_areas = residue_areas.get(chain_id, {})
            for residue in chain:
                res3 = residue.get_resname().strip()
                aa = _RES3TO1.get(res3)
                if aa is None or not residue.has_id("CA"):
                    continue
                resid = residue.id[1]  # 残基号
                ra = chain_areas.get(str(resid))
                if ra is not None:
                    fracs[idx] = ra.relativeTotal
                else:
                    fracs[idx] = 0.0  # 兜底：未匹配则视为埋藏
                idx += 1

    if any(f is None for f in fracs):
        raise RuntimeError(f"{pdb_path} 部分残基未匹配 freesasa 结果")

    frac_arr = np.clip(np.array(fracs, dtype=np.float64), 0.0, None)
    surface_mask = frac_arr >= surface_threshold
    return {
        "seq": "".join(seq),
        "frac_sasa": frac_arr,
        "surface_mask": surface_mask,
        "is_surface": int(surface_mask.sum()),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="逐残基 fractional SASA")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="输出 npz 路径（可选）")
    args = ap.parse_args()

    r = fractional_sasa(args.pdb, surface_threshold=args.threshold)
    print(f"序列长度 {len(r['seq'])}，表面位点 {r['is_surface']} / {len(r['seq'])}")
    print(f"fracSASA: min={r['frac_sasa'].min():.3f} max={r['frac_sasa'].max():.3f} "
          f"mean={r['frac_sasa'].mean():.3f}")
    # 表面位点示例
    surf = [(i, aa) for i, (aa, m) in enumerate(zip(r["seq"], r["surface_mask"])) if m][:10]
    print(f"表面位点示例（前 10）: {surf}")
    if args.out:
        np.savez(args.out, seq=np.array(list(r["seq"]), dtype="U1"),
                 frac_sasa=r["frac_sasa"], surface_mask=r["surface_mask"])
        print(f"已写 {args.out}")
