"""口袋范围定义工具：带配体蛋白的配体结合口袋分类与删减风险预警。

背景（2026-09-01 配体模式删减捷径）：
  v12 三个组成监督损失只覆盖表面残基（frac_sasa>=0.25）。配体结合口袋是表面
  深凹陷，60-75% 口袋残基 frac_sasa<0.25 -> 划入"核心"（不受监督），其中带电
  残基（D/E/K/R）会走"成对删"捷径被无差别删除（实测 2FEO 深部 10->1.5）。
  本工具在**设计前**定义口袋范围并输出删减风险预警，供人工决定是否 fix：

  1. 口袋定义：配体原子 8Å 内残基（Cα 距离，与验证脚本口径一致）
  2. 每个口袋残基分类：最近配体距离 / frac_sasa / 表面或深部 / 是否带电 / 保护建议
  3. 输出建议 fix 列表（深部带电残基，可直接传给 run_guided --fixed_residues）
  4. 输出强接触残基数据列（骨架原子 <4.5Å，近似），供人工结构分析参考

边界（用户定，2026-09-01）：
  - 不做"强相互作用"结构分析（那是设计前的人工结构调研/PLIP 工作），
    本工具只输出数据列辅助人工判断。
  - 不修改任何训练脚本/模型；不退役 q_core；核心残基不额外自动 fix。
  - 深部/表面分类基于 freesasa（含侧链原子），准确；接触距离基于骨架原子
    （N/CA/C/O），未含侧链原子，是近似值，仅作参考列。

用法（项目根）：
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python \
      code/tools/pocket_protect/define_pocket.py --pdb data/validation_pdbs/2FEO.pdb

输出：output/pocket_protect/<pdb名>/
  - pocket_table.txt         逐残基分类表（人读）
  - pocket_table.json        同内容机器可读
  - pocket_fix.txt           建议 fix 残基（默认=深部带电；--include-contact 追加强接触）
  - contact_residues.txt     强接触残基（数据列，人工结构分析参考）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))
sys.path.insert(0, str(_PROJECT_DIR / "code"))
sys.path.insert(0, str(_PROJECT_DIR))
from data_utils import parse_PDB  # noqa: E402
from src.sasa import fractional_sasa  # noqa: E402

CHARGED = "DEKR"


def pocket_distances(protein_dict, cutoff):
    """口袋 = 与配体原子（Y）Cα 距离 < cutoff Å 的残基索引 + 逐残基最近距离。"""
    Y = protein_dict.get("Y")
    X = protein_dict["X"]
    if Y is None or Y.numel() == 0:
        return None, None
    Yc = Y.reshape(-1, 3).cpu().numpy()
    CA = X[:, 1, :].cpu().numpy()
    d = np.linalg.norm(CA[:, None, :] - Yc[None, :, :], axis=-1)  # (L, n_lig)
    dmin = d.min(axis=1)
    idx = np.where(dmin < cutoff)[0]
    return idx, dmin


def contact_distances(protein_dict, cutoff):
    """强接触 = 残基骨架原子（N/CA/C/O）到配体 < cutoff Å（近似，无侧链）。"""
    Y = protein_dict.get("Y")
    X = protein_dict["X"]
    if Y is None or Y.numel() == 0:
        return None, None
    Yc = Y.reshape(-1, 3).cpu().numpy()
    B = X.reshape(-1, 3).cpu().numpy()          # (L*4, 3)
    d = np.linalg.norm(B[:, None, :] - Yc[None, :, :], axis=-1)  # (L*4, n_lig)
    dmin = d.reshape(X.shape[0], -1).min(axis=1)  # 每残基最近骨架原子-配体距离
    idx = np.where(dmin < cutoff)[0]
    return idx, dmin


def main():
    ap = argparse.ArgumentParser(description="口袋范围定义：配体结合口袋分类与删减风险预警")
    ap.add_argument("--pdb", required=True, help="带配体原子的 PDB 路径")
    ap.add_argument("--pocket-cutoff", type=float, default=8.0,
                    help="口袋范围（Cα-配体 距离 Å，默认 8，与验证脚本口径一致）")
    ap.add_argument("--contact-cutoff", type=float, default=4.5,
                    help="强接触（骨架原子-配体 距离 Å，默认 4.5，近似参考列）")
    ap.add_argument("--sasa-threshold", type=float, default=0.25,
                    help="表面/深部阈值（frac_sasa，默认 0.25，与 v12 损失一致）")
    ap.add_argument("--include-contact", action="store_true",
                    help="把强接触残基也加入 pocket_fix.txt（默认只含深部带电）")
    ap.add_argument("--outdir", default=None,
                    help="输出目录（默认 output/pocket_protect/<pdb名>/）")
    args = ap.parse_args()

    pdb_path = Path(args.pdb)
    name = pdb_path.stem
    outdir = Path(args.outdir) if args.outdir else _PROJECT_DIR / "output/pocket_protect" / name
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 读结构
    protein_dict, _, _, icodes, _ = parse_PDB(str(pdb_path))
    X = protein_dict["X"]
    L = X.shape[0]
    R_idx = list(protein_dict["R_idx"].cpu().numpy())
    chain_letters = list(protein_dict["chain_letters"])
    resnames = [str(chain_letters[i]) + str(R_idx[i]) + icodes[i] for i in range(L)]

    # 序列：S 是字母索引，转一字母（与 seq_to_string 同义，本地实现避免 import 重依赖）
    from run_guided import seq_to_string  # noqa: F401  # 复用已验证的转换
    seq = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())

    if protein_dict.get("Y") is None or protein_dict["Y"].numel() == 0:
        print(f"!! {name}: 无配体原子（HETATM）→ 无法定义口袋。配体模式才需要本工具。")
        sys.exit(1)

    # 2. SASA（表面/深部）
    # align_to_full=True 会含配体/非标准残基（2FEO: 265 vs 蛋白 221），
    # 故用 False（只标准氨基酸），长度应 = parse_PDB L；不匹配时按残基号对齐兜底。
    s = fractional_sasa(str(pdb_path), align_to_full=False)
    frac = np.array(s["frac_sasa"], dtype=np.float64)
    if len(frac) != L:
        # 兜底：按 PDB 残基号（resid）交集对齐。
        # parse_PDB 可能跳过 sasa 视为标准的残基（如 1AS2 的 236-238 原子不全）：
        # 丢弃 sasa 里 parse 不存在的残基，保留者顺序应与 parse_PDB 一致（都是 PDB 文件顺序）。
        resid_sasa = np.array(s["residues"], dtype=np.int64)
        map_r = {int(r): i for i, r in enumerate(R_idx)}
        keep = [int(r) in map_r for r in resid_sasa]
        frac_keep = frac[keep]
        resid_keep = resid_sasa[keep]
        idx_map = [map_r[int(r)] for r in resid_keep]
        ordered = len(idx_map) == L and np.all(np.diff(idx_map) > 0)
        if ordered:
            frac_aligned = np.zeros(L, dtype=np.float64)
            frac_aligned[np.array(idx_map)] = frac_keep
            frac = frac_aligned
            print(f"  (resid 对齐: 丢弃 sasa 多出 {len(resid_sasa) - L} 个非 parse 残基 -> L {L})",
                  flush=True)
        else:
            print(f"!! {name}: frac_sasa 长度 {len(frac)} != 序列长度 {L}，且 resid 对齐失败"
                  f"（keep={sum(keep)}, ordered={ordered}）。")
            sys.exit(1)

    # 3. 口袋 / 接触
    pocket, d_ca = pocket_distances(protein_dict, args.pocket_cutoff)
    contact, d_bb = contact_distances(protein_dict, args.contact_cutoff)
    contact_set = set(contact.tolist())

    # 4. 逐残基分类
    rows = []
    n_deep_charged = n_surf_charged = n_contact = 0
    for i in pocket:
        aa = seq[i]
        frac_i = frac[i]
        is_surf = frac_i >= args.sasa_threshold
        charged = aa in CHARGED
        in_contact = int(i) in contact_set
        if in_contact:
            level = "人工fix(强接触)"
            n_contact += 1
        elif (not is_surf) and charged:
            level = "建议fix(深部带电)"
            n_deep_charged += 1
        elif is_surf and charged:
            level = "可选fix(表面带电)"
            n_surf_charged += 1
        else:
            level = "无需"
        rows.append({
            "idx": int(i),
            "resname": resnames[i],
            "aa": aa,
            "dist_ca": round(float(d_ca[i]), 2),
            "dist_contact": round(float(d_bb[i]), 2) if int(i) in contact_set else None,
            "frac_sasa": round(float(frac_i), 3),
            "zone": "深部" if not is_surf else "表面",
            "charged": charged,
            "level": level,
        })

    # fix 列表：深部带电（默认），可选追加强接触
    fix = [r["resname"] for r in rows if r["level"] == "建议fix(深部带电)"]
    if args.include_contact:
        fix += [r["resname"] for r in rows if r["level"] == "人工fix(强接触)"]
    fix = sorted(set(fix))
    contact_list = sorted(set(resnames[i] for i in contact))

    # 5. 输出
    rows.sort(key=lambda r: r["idx"])
    print(f"\n===== 口袋范围定义: {name} (L={L}) =====", flush=True)
    print(f"口袋残基(≤{args.pocket_cutoff}Å Cα): {len(pocket)}  "
          f"强接触(≤{args.contact_cutoff}Å 骨架,近似): {n_contact}  "
          f"深部带电(建议fix): {n_deep_charged}  表面带电(可选): {n_surf_charged}", flush=True)
    print(f"\n{'残基':8s} {'AA':>3s} {'Cα-配体Å':>8s} {'骨架接触Å':>8s} {'fracSASA':>8s} "
          f"{'区':>4s} {'带电':>3s} {'建议'}", flush=True)
    print("-" * 62, flush=True)
    for r in rows:
        dc = f"{r['dist_contact']:.2f}" if r["dist_contact"] is not None else "   -"
        print(f"{r['resname']:8s} {r['aa']:>3s} {r['dist_ca']:8.2f} {dc:>8s} {r['frac_sasa']:8.3f} "
              f"{r['zone']:>4s} {'✓' if r['charged'] else ' ':>3s} {r['level']}", flush=True)

    # 6. 写文件
    with open(outdir / "pocket_table.txt", "w") as f:
        f.write(f"# 口袋范围定义 {name} (L={L})\n")
        f.write(f"# 口径: 口袋≤{args.pocket_cutoff}Å Cα; 强接触≤{args.contact_cutoff}Å 骨架原子(近似); "
                f"表面≥frac_sasa {args.sasa_threshold}\n")
        f.write(f"{'残基':8s} {'AA':>3s} {'Cα-配体Å':>8s} {'骨架接触Å':>8s} {'fracSASA':>8s} "
                f"{'区':>4s} {'带电':>3s} {'建议'}\n")
        for r in rows:
            dc = f"{r['dist_contact']:.2f}" if r["dist_contact"] is not None else "   -"
            f.write(f"{r['resname']:8s} {r['aa']:>3s} {r['dist_ca']:8.2f} {dc:>8s} "
                    f"{r['frac_sasa']:8.3f} {r['zone']:>4s} "
                    f"{'✓' if r['charged'] else ' ':>3s} {r['level']}\n")

    with open(outdir / "pocket_table.json", "w") as f:
        json.dump({
            "pdb": str(pdb_path), "name": name, "L": L,
            "pocket_cutoff": args.pocket_cutoff, "contact_cutoff": args.contact_cutoff,
            "sasa_threshold": args.sasa_threshold,
            "n_pocket": int(len(pocket)), "n_contact": n_contact,
            "n_deep_charged": n_deep_charged, "n_surf_charged": n_surf_charged,
            "pocket": rows,
            "fix": fix,
            "contact": contact_list,
        }, f, indent=2)

    with open(outdir / "pocket_fix.txt", "w") as f:
        f.write("\n".join(fix) + ("\n" if fix else ""))

    with open(outdir / "contact_residues.txt", "w") as f:
        f.write("# 强接触残基（骨架原子近似，供人工结构分析/PLIP 交叉验证）\n")
        f.write("\n".join(contact_list) + ("\n" if contact_list else ""))

    print(f"\n已写 {outdir}/", flush=True)
    print(f"  pocket_fix.txt = {len(fix)} 个建议 fix 残基（可直接 --fixed_residues \"$(cat 文件)\"）",
          flush=True)


if __name__ == "__main__":
    main()
