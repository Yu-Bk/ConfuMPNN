"""多 pH 温和区验证统计：合并电荷/恢复率/TM 折叠(RMSD)/pLDDT，输出汇总表。

v3 D6：RMSD 作为 H1b 辅助指标（US-align 输出，与 TM 联报，按域报告）。

输入（由 ph_scan_validation.py 采样 + esmfold_score.py + tm_score.py 生成）：
  --root/{pdb}/pH{ph}/ph_scan.json   条件化（target=native）电荷+recovery
  --uncond_root/{pdb}/ph_scan.json  无条件基线 recovery
  --root/{pdb}/pH{ph}/tm.csv        USalign TM-score（回折 vs 骨架）
  --root/{pdb}/pH{ph}/plddt.csv     ESMFold pLDDT

输出：--out json + 打印汇总表。

用法（code/ 下）：
  PYTHONPATH=. python tests/ph_scan_stats.py \
      --root ../output/ph_scan --uncond_root ../output/ph_scan_uncond \
      --out ../output/ph_scan_stats.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parents[1]


def read_csv_col(path, col, cast=float):
    vals = []
    if not path.exists():
        return vals
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            if col in row:
                try:
                    vals.append(cast(row[col]))
                except (ValueError, TypeError):
                    pass
    return vals


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def frac(xs, lo=None, hi=None):
    if not xs:
        return None
    n = len(xs)
    if lo is not None:
        return sum(1 for x in xs if x >= lo) / n
    return sum(1 for x in xs if x < hi) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--uncond_root", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    summary = {}
    print(f"{'蛋白':<12}{'pH':>5}{'target':>7}{'dev':>6}{'recovery':>9}"
          f"{'rec_基线':>8}{'TM中位':>8}{'TM>0.7':>8}{'TM<0.5':>8}{'RMSD':>7}{'pLDDT':>7}")
    for json_path in sorted(root.glob("*/ph_scan.json")):
        pdb = json_path.parent.name
        d = json.load(open(json_path))
        summary[pdb] = {"L": d["L"], "native": d["native"], "pH_arms": {}}
        for ph_s, arm in d["pH_arms"].items():
            arm_dir = json_path.parent / f"pH{ph_s}"
            tm = read_csv_col(arm_dir / "tm.csv", "tm_score")
            rmsd = read_csv_col(arm_dir / "tm.csv", "rmsd")  # v3 D6：RMSD 辅助指标
            plddt = read_csv_col(arm_dir / "plddt.csv", "mean_plddt")
            # 无条件 recovery
            rec_base = None
            if args.uncond_root:
                uj = Path(args.uncond_root) / pdb / "ph_scan.json"
                if uj.exists():
                    ud = json.load(open(uj))
                    rec_base = ud["pH_arms"].get(ph_s, {}).get("recovery")
            summary[pdb]["pH_arms"][ph_s] = {
                **arm,
                "tm_median": round(median(tm), 3) if tm else None,
                "tm_ge070": round(frac(tm, lo=0.7), 3) if tm else None,
                "tm_lt050": round(frac(tm, hi=0.5), 3) if tm else None,
                "rmsd_median": round(median(rmsd), 2) if rmsd else None,
                "plddt_median": round(median(plddt), 1) if plddt else None,
                "recovery_baseline": round(rec_base, 3) if rec_base else None,
            }
            print(f"{pdb:<12}{ph_s:>5}{str(arm['target']):>7}"
                  f"{str(arm['dev']):>6}{arm['recovery']:>9}"
                  f"{str(round(rec_base, 3) if rec_base else '-'):>8}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['tm_median']):>8}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['tm_ge070']):>8}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['tm_lt050']):>8}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['rmsd_median']):>7}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['plddt_median']):>7}",
                  flush=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n已写 {args.out}")


if __name__ == "__main__":
    main()
