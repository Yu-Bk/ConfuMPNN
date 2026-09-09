#!/usr/bin/env python
"""打分/折叠调度器：对每个含 seqs.fa 的组目录批量 ESMFold 回折 + TM/RMSD vs 参考骨架 + Tm + Sol。
用法（仓库根）:
  python code/tools/design_bench/fold_scheduler.py --root output/design_groups \
    --ref data/validation_pdbs/1BC8.pdb --outcsv output/design_groups/_metrics.csv \
    --esm_env confumpnn-esmfold --tm_env confumpnn --temberture_env confumpnn-temberture
每个 <root>/<grp>/seqs.fa → <grp>/folds(pdb) + plddt.csv + tm.csv + *.tm.csv + *protein_sol*.txt。
复用 code/tests/{esmfold_score,tm_score,temberture_score}.py 与 protein_sol_mcp；本脚本只做目录遍历/分批/汇总（resume：已有 folds 跳过）。
参考活用例：L11design/test/exp1_fold*.sh。"""
import argparse, os, subprocess, sys, csv, re
from pathlib import Path

def find_seqs(root):
    return sorted(Path(root).glob("*/seqs.fa"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref", required=True, help="参考骨架 PDB（native 链）")
    ap.add_argument("--outcsv", default=None)
    ap.add_argument("--esm_env", default="confumpnn-esmfold")
    ap.add_argument("--tm_env", default="confumpnn")
    ap.add_argument("--temberture_env", default="confumpnn-temberture")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    env = {k: os.environ[k] for k in os.environ}
    def py(e): return f"{os.path.expanduser('~')}/miniconda3/envs/{e}/bin/python"
    rows = []
    for fa in find_seqs(a.root):
        grp = fa.parent
        folddir = grp / "folds"; folddir.mkdir(exist_ok=True)
        pld = grp / "plddt.csv"; tm = grp / "tm.csv"
        if not pld.exists() and not a.dry:
            subprocess.run([py(a.esm_env), "code/tests/esmfold_score.py",
                            "--fasta", str(fa), "--out", str(pld), "--outdir", str(folddir),
                            "--device", a.device], env=env)
        if not tm.exists() and not a.dry:
            subprocess.run([py(a.tm_env), "code/tests/tm_score.py",
                            "--folds", str(folddir), "--ref", a.ref, "--out", str(tm)], env=env)
        # Tm
        subprocess.run([py(a.temberture_env), "code/tests/temberture_score.py",
                        "--fasta", str(fa), "--out", str(grp / "seqs.fa.tm.csv")], env=env, capture_output=True)
        # Sol
        subprocess.run(["python3", "protein_sol_mcp/scripts/protein_sol_predict.py", str(fa)], capture_output=True)
        rows.append((grp.name, fa, pld, tm))
        print(f"[fold] {grp.name}: {'ok' if (pld.exists() and tm.exists()) else 'pending'}")
    if a.outcsv and not a.dry:
        with open(a.outcsv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["grp", "seqs_fa", "plddt", "tm"])
            w.writerows(rows)
        print("已写", a.outcsv)

if __name__ == "__main__":
    main()
