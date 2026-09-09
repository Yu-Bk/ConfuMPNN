#!/usr/bin/env python
"""批量条件设计组运行器：读组表(json) → 每组用 auto_calib_guided 采样（可选 autofit 标定）。
组表 json 例:
[
 {"name":"prot_q8","mode":"protein","enc":"output/finetune_v12_2/finetune_epoch030.pt",
  "weights":"MoMPNN/.../mompnn_...b01.ckpt","pdb":"input/L11.pdb","pH":7.4,"target":8.0,
  "native_q":6.87,"n":300,"fixed":"I3 I5 ...","num_ligand_atoms":0}
]
用法: python code/tools/design_bench/design_group_runner.py --groups groups.json --outroot output/mygroups --autofit
产物: <outroot>/<name>/seqs.fa + summary.json。参考活用例：L11design/test/exp2_*.sh。"""
import argparse, json, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--outroot", default="output/design_groups")
    ap.add_argument("--autofit", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    groups = json.load(open(a.groups))
    Path(a.outroot).mkdir(parents=True, exist_ok=True)
    for g in groups:
        outdir = Path(a.outroot) / g["name"]
        cmd = [a.python, "code/tools/design_bench/auto_calib_guided.py"]
        if a.autofit:
            cmd += ["--autofit"]
        cmd += ["--enc", g["enc"], "--weights", g["weights"], "--pdb", g["pdb"],
                "--pH", str(g["pH"]), "--target_charge", str(g["target"]),
                "--native_q", str(g.get("native_q", 0.0)),
                "--num_samples", str(g.get("n", 100)),
                "--calib_cache", g.get("calib_cache", f"output/charge_calibration_{Path(g['pdb']).stem}.small.json")]
        if g.get("fixed"):
            cmd += ["--fixed_residues", g["fixed"]]
        cmd += ["--out_dir", str(outdir)]
        print(">>", " ".join(cmd)); subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
