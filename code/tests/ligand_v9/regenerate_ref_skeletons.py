"""重新生成 ref 骨架 PDB（write_ref_skeleton 修复真实残基号后，一次性重生成）。

背景：write_ref_skeleton 早期把残基号硬编码为 `A{4:4d}`（恒 4），2026-09-01 修正为
parse_PDB 的 R_idx 真实值。坐标不变（同一 parse_PDB X），仅残基号 4 → 真实值。
本脚本对两线 ref 目录重跑一遍，保持产物一致。TM-score（只比坐标）/H3（按行序提 CA）
均不受残基号影响。

用法（项目根，confumpnn 环境）：
  PYTHONPATH=code python code/tests/ligand_v9/regenerate_ref_skeletons.py
"""
import json
import sys
from pathlib import Path

_PROJ = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(_PROJ / "LigandMPNN"))
sys.path.insert(0, str(_PROJ / "code"))
sys.path.insert(0, str(_PROJ / "code" / "tests" / "ligand_v9"))

from data_utils import parse_PDB  # noqa: E402
from validate_generalization import write_ref_skeleton  # noqa: E402

# 两线 ref 目录（与现有泛化验证产物一致）
TARGETS = [
    _PROJ / "output" / "generalization_v12_2_calib_small" / "ref",
    _PROJ / "output" / "generalization_ligand_v12_2" / "ref",
]

manifest = json.load(open(_PROJ / "data" / "validation_pdbs" / "validation_manifest.json"))
items = {it["pdb"]: Path(it["path"]) for it in manifest["items"]}

for ref_dir in TARGETS:
    for pdb in sorted(items):
        out = ref_dir / f"{pdb}_ref.pdb"
        if not out.exists():
            print(f"  skip {pdb}（{out} 不存在）")
            continue
        protein_dict, _, _, icodes, _ = parse_PDB(str(items[pdb]))
        write_ref_skeleton(protein_dict, out, icodes=icodes)
        # 校验：残基号（含 icode）唯一且 = CA 数（全 4 旧版会塌缩为 1）
        resids = sorted({l[22:27].strip() for l in open(out) if l.startswith("ATOM")})
        n_ca = sum(1 for l in open(out) if l.startswith("ATOM") and l[12:16].strip() == "CA")
        flag = "✅" if len(resids) == n_ca else "⚠️"
        print(f"  {flag} {pdb}: {len(resids)} 唯一残基号 / {n_ca} CA"
              f"（{resids[:3]}...{resids[-2:]}）")

print("完成。")
