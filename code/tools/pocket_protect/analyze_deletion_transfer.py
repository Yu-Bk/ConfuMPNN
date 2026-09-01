"""删减区域分析：fix 前后各区域带电残基删减对比（回答"删减转移在哪"）。

背景（2026-09-01 口袋 fix 实测）：fix 深部带电后全序列组成仍 0.66-0.68×，
需要弄清删减在哪些区域。本脚本分 5 区域统计 fix 前后带电残基数：
  深部fix      = 口袋内 frac_sasa<0.25 且带电（建议 fix 位点）
  口袋表面带   = 口袋内 frac_sasa≥0.25 且带电（可选 fix）
  口袋其他     = 口袋内其余残基
  口袋外表面   = 口袋外 frac_sasa≥0.25
  口袋外深部   = 口袋外 frac_sasa<0.25

实测结论（2026-09-01）：**删减是全局性病态，非"fix 后转移"**——除深部 fix
位点外，其余区域 fix 前后删减倍率几乎不变（2FEO 口袋外表面 0.70→0.68、
口袋外深部 0.10→0.13）；全序列组成的增加全部来自深部 fix 位点保护。fix 只
堵住深部 fix 出口，其余区域删减照旧 → 根治全局删减需训练侧组成监督。

用法（项目根）：
  /home/baokun_yu/miniconda3/envs/confumpnn/bin/python \
      code/tools/pocket_protect/analyze_deletion_transfer.py --names 2FEO,1AXW,1C6O
输出：终端表 + output/pocket_fix_test/deletion_transfer.json
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
sys.path.insert(0, str(_PROJECT_DIR / "code/tests/ligand_v9"))
from data_utils import parse_PDB  # noqa: E402
from src.sasa import fractional_sasa  # noqa: E402
from pocket_comp_compare import read_seqfa  # noqa: E402

CHARGED = "DEKR"
ZONES = ["深部fix", "口袋表面带", "口袋其他", "口袋外表面", "口袋外深部"]


def zone_of(i, lev, frac, thresh):
    if i in lev and lev[i] == "建议fix(深部带电)":
        return "深部fix"
    if i in lev and lev[i] == "可选fix(表面带电)":
        return "口袋表面带"
    if i in lev:
        return "口袋其他"
    return "口袋外表面" if frac[i] >= thresh else "口袋外深部"


def main():
    ap = argparse.ArgumentParser(description="删减区域分析（fix 前后）")
    ap.add_argument("--names", default="2FEO,1AXW,1C6O")
    ap.add_argument("--no-fix-root", default=str(_PROJECT_DIR / "output/generalization_ligand_v12_2/ligand"))
    ap.add_argument("--fix-root", default=str(_PROJECT_DIR / "output/pocket_fix_test/v12_2/ligand"))
    ap.add_argument("--pdb-dir", default=str(_PROJECT_DIR / "data/validation_pdbs"))
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--sasa-threshold", type=float, default=0.25)
    args = ap.parse_args()

    names = [n for n in args.names.split(",") if n.strip()]
    results = {}
    for name in names:
        lev = {r["idx"]: r["level"] for r in
               json.load(open(_PROJECT_DIR / "output/pocket_protect" / name / "pocket_table.json"))["pocket"]}
        pdb = Path(args.pdb_dir) / f"{name}.pdb"
        d, _, _, _, _ = parse_PDB(str(pdb))
        L = d["X"].shape[0]
        R_idx = list(d["R_idx"].cpu().numpy())
        s = fractional_sasa(str(pdb), align_to_full=False)
        map_r = {int(r): i for i, r in enumerate(R_idx)}
        frac = np.zeros(L)
        for f, r in zip(s["frac_sasa"], s["residues"]):
            if int(r) in map_r:
                frac[map_r[int(r)]] = f
        g0, n0 = read_seqfa(str(Path(args.no_fix_root) / name / f"pH{args.pH}" / "arm_native" / "seqs.fa"))
        g1, n1 = read_seqfa(str(Path(args.fix_root) / name / f"pH{args.pH}" / "arm_native" / "seqs.fa"))
        print(f"\n=== {name} (L={L}) ===")
        print(f"{'区域':10s} {'native带':>6s} {'无fix':>6s} {'倍率':>5s} | {'有fix':>6s} {'倍率':>5s}")
        results[name] = {}
        for z in ZONES:
            idx = [i for i in range(L) if zone_of(i, lev, frac, args.sasa_threshold) == z]
            if not idx:
                continue
            nat = sum(1 for i in idx if n0[i] in CHARGED)
            c0 = np.mean([sum(1 for i in idx if x[i] in CHARGED) for x in g0])
            c1 = np.mean([sum(1 for i in idx if x[i] in CHARGED) for x in g1])
            r0 = c0 / nat if nat else 0
            r1 = c1 / nat if nat else 0
            print(f"{z:10s} {nat:6d} {c0:6.1f} {r0:5.2f} | {c1:6.1f} {r1:5.2f}")
            results[name][z] = {"native": nat, "no_fix": round(float(c0), 1),
                                "no_fix_ratio": round(float(r0), 2),
                                "with_fix": round(float(c1), 1),
                                "with_fix_ratio": round(float(r1), 2)}

    out = _PROJECT_DIR / "output/pocket_fix_test"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "deletion_transfer.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n已写 {out}/deletion_transfer.json", flush=True)


if __name__ == "__main__":
    main()
