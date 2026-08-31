"""v12.2 Tm/Sol 准备：用 parse_PDB 提取 native 序列（与 validate_generalization.py 完全一致）。

采样脚本用 parse_PDB(原始PDB) 得到 native 序列（多链自动拼接），Tm/Sol 的 native 基线
必须用同一个序列空间。从原始 PDB（data/validation_pdbs/<NAME>.pdb）解析，
seq_to_string 转字符串，写 native.fa。

用法（项目根）：
  PYTHONPATH=code python code/tests/v12_2_extract_native.py
"""
import json
import sys
from pathlib import Path

_PROJECT_DIR = next(p for p in Path(__file__).resolve().parents
                    if (p / "code").is_dir() and (p / "LigandMPNN").is_dir())
_CODE_DIR = _PROJECT_DIR / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))

from data_utils import parse_PDB  # noqa: E402
from run_guided import seq_to_string  # noqa: E402

MANIFEST = _PROJECT_DIR / "data/validation_pdbs/validation_manifest.json"
OUT_DIR = _PROJECT_DIR / "output/tm_sol_v12_2/ref_native"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    man = json.load(open(MANIFEST))
    for it in man["items"]:
        name, path = it["pdb"], _PROJECT_DIR / it["path"]
        try:
            protein_dict, *_ = parse_PDB(str(path))
        except Exception as e:
            print(f"  !! {name} parse 失败: {e}")
            continue
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        out = OUT_DIR / f"{name}_native.fa"
        with open(out, "w") as f:
            f.write(f">{name}_native L={L}\n{native}\n")
        print(f"{name:6s} L={L} -> {out}")


if __name__ == "__main__":
    main()
