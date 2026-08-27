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


def fractional_sasa(pdb_path, surface_threshold=0.25, align_to_full=True):
    """计算 PDB 的逐残基 fractional SASA。

    参数:
        pdb_path: PDB 文件路径（蛋白结构，可含配体/核酸，会跳过）
        surface_threshold: 表面资格阈值 θ；fracSASA ≥ θ 视为"表面位点"
            （v10 B 的 L_add 表面硬门槛，θ 由配置定，默认 0.25）
        align_to_full: 是否**保留非标准残基位置**（X 处 frac 补 0.0）。
            True 时返回长度 = parse_PDB 的 L（含 X/非标准残基，X 对齐为 0），
            供 train_finetune 直接按位置对齐（L_add 需要全残基索引）。
            False 时只返回标准氨基酸（长度 = 有效残基数，freesasa 原生口径）。

    返回:
        dict:
            seq: 蛋白序列（单字母；align_to_full=True 时 X/非标准残基用 '-' 占位）
            frac_sasa: [L] float 数组，逐残基 fractional SASA（X 位置为 0.0）
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

    # 收集 (链, 残基号, aa, 是标准氨基酸) 顺序列表
    residues = []  # 按 PDB 顺序：{chain, resid, aa, is_aa}
    for model in structure:
        for chain in model:
            for residue in chain:
                res3 = residue.get_resname().strip()
                aa = _RES3TO1.get(res3)
                residues.append({
                    "chain": chain.id,
                    "resid": residue.id[1],
                    "aa": aa,
                    "is_aa": aa is not None and residue.has_id("CA"),
                })

    if not any(r["is_aa"] for r in residues):
        raise RuntimeError(f"{pdb_path} 无蛋白残基")

    # freesasa 全结构计算 → 逐残基 relativeTotal（按链+残基号索引）
    try:
        fs = freesasa.structureFromBioPDB(structure)
        result = freesasa.calc(fs)
        residue_areas = result.residueAreas()  # {chain: {resnum: ResidueArea}}
    except Exception as e:
        raise RuntimeError(f"freesasa 计算失败: {e}")

    # 逐残基填 frac
    seq_chars, fracs, resids = [], [], []
    for r in residues:
        if not r["is_aa"]:
            seq_chars.append("-" if align_to_full else None)  # 非标准/无CA
            fracs.append(0.0 if align_to_full else None)      # X 位置 frac=0
            resids.append(None if align_to_full else None)
            continue
        chain_areas = residue_areas.get(r["chain"], {})
        ra = chain_areas.get(str(r["resid"]))
        frac = ra.relativeTotal if ra is not None else 0.0
        seq_chars.append(r["aa"])
        fracs.append(frac)
        resids.append(r["resid"])   # 残基号（供 resnum 交集对齐）

    if align_to_full:
        # 全残基口径：长度 = parse_PDB L（含 X/非标准残基，X 处 frac=0）
        seq = "".join(c if c is not None else "X" for c in seq_chars)
        frac_arr = np.clip(np.array(fracs, dtype=np.float64), 0.0, None)
        resid_arr = np.array([r if r is not None else -1 for r in resids], dtype=np.int64)
    else:
        # 只保留标准氨基酸（freesasa 原生口径），同时返回残基号列表
        keep = [(c, f, rid) for c, f, rid in zip(seq_chars, fracs, resids) if c is not None]
        seq = "".join(c for c, _, _ in keep)
        frac_arr = np.clip(np.array([f for _, f, _ in keep], dtype=np.float64), 0.0, None)
        resid_arr = np.array([rid for _, _, rid in keep], dtype=np.int64)

    # ⚠️ 防 NaN 污染（v10 踩坑）：freesasa 对某些异常几何残基返回 NaN relativeTotal
    # （如 1GTV.pdb 有 177 个 NaN）。NaN 传入 surface_add_charge_loss 会污染整个训练。
    # 处理：NaN/Inf → 0.0（视为"无法计算 → 当埋藏，不参与 L_add"）。
    frac_arr = np.nan_to_num(frac_arr, nan=0.0, posinf=0.0, neginf=0.0)

    surface_mask = frac_arr >= surface_threshold
    return {
        "seq": seq,
        "frac_sasa": frac_arr,
        "residues": resid_arr,       # 与 frac_sasa 同长的残基号（resnum 对齐用）
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
