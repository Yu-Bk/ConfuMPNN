"""用 US-align 批量计算「回折结构 vs 参考骨架」的 TM-score（结构保持度）。

自洽性检验：生成序列经 ESMFold 回折的结构与原始 PDB 骨架比对，
TM-score 高 = 序列能折叠回原骨架（客观结构保持），比 pLDDT 更有说服力。

用法（confumpnn 环境）：
  python tm_score.py --folds <ESMFold 输出目录> --ref <参考骨架.pdb> --out tm.csv
"""
import argparse
import csv
import glob
import os
import re
import subprocess


def _find_usalign():
    """定位 USalign：优先 PATH，回退 confumpnn 环境 bin（直接调 env python 时 PATH 无它）。"""
    import shutil
    exe = shutil.which("USalign")
    if exe:
        return exe
    cand = os.path.expanduser(
        "~/miniconda3/envs/confumpnn/bin/USalign"
    )
    return cand if os.path.isfile(cand) else "USalign"


def run_usalign(q, ref):
    """调用 USalign，返回 (tmscore, rmsd)。TM-score 取 normalized by reference(Structure_2)。"""
    usalign = _find_usalign()
    p = subprocess.run(
        [usalign, q, ref], capture_output=True, text=True, timeout=120
    )
    text = p.stdout
    tm = None
    rmsd = None
    for line in text.splitlines():
        m = re.search(r"TM-score=\s*([0-9.]+)\s*\(normalized by length of Structure_2", line)
        if m:
            tm = float(m.group(1))
        m2 = re.search(r"Aligned length=\s*\d+,\s*RMSD=\s*([0-9.]+)", line)
        if m2:
            rmsd = float(m2.group(1))
    return tm, rmsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, help="ESMFold 回折 PDB 目录（--outdir 输出）")
    ap.add_argument("--ref", required=True, help="参考骨架 PDB（纯蛋白链，如 1BC8_chainC.pdb）")
    ap.add_argument("--out", required=True, help="输出 CSV: name,tm_score,rmsd")
    args = ap.parse_args()

    pdbs = sorted(glob.glob(os.path.join(args.folds, "*.pdb")))
    if not pdbs:
        print(f"!! {args.folds} 下没有 PDB")
        return
    rows = []
    for q in pdbs:
        name = os.path.splitext(os.path.basename(q))[0]
        tm, rmsd = run_usalign(q, args.ref)
        rows.append((name, round(tm, 4) if tm else None, round(rmsd, 2) if rmsd else None))
        print(f"  {name:<40} TM-score={tm if tm else 'NA'}  RMSD={rmsd if rmsd else 'NA'}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "tm_score", "rmsd"])
        w.writerows(rows)
    print(f"已写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
