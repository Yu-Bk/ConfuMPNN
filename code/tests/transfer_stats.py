"""迁移检验统计汇总：合并电荷/恢复率/TM 折叠(RMSD)/pLDDT，输出汇总表。

v3 D6：RMSD 作为 H1b 辅助指标（US-align 输出，与 TM 联报，按域报告）。

输入（transfer_validation.py 采样 + esmfold_score.py + tm_score.py 生成）：
  --root/{pdb}/transfer.json            迁移主实验（条件化，各 pH）
  --uncond_root/{pdb}/transfer.json     无条件基线（MoMPNN，backbone 电荷偏好）
  --mompnn_root/{pdb}/transfer.json     MoMPNN-cond 对照（有配体蛋白忽略配体）
  --root/{pdb}/pH{ph}/tm.csv            USalign TM-score

输出：--out json + 打印汇总表。

用法（code/ 下，confumpnn 环境）：
  PYTHONPATH=. python tests/transfer_stats.py \
      --root ../output/transfer --uncond_root ../output/transfer_uncond \
      --mompnn_root ../output/transfer_mompnn \
      --out ../output/transfer_stats.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[1]


def read_csv_col(path, col, cast=float):
    vals = []
    if not path.exists():
        return vals
    with open(path) as f:
        for row in csv.DictReader(f):
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
    ap.add_argument("--mompnn_root", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    summary = {}
    print(f"{'蛋白':<14}{'pH':>5}{'target':>7}{'mean':>7}{'dev':>6}{'recovery':>9}"
          f"{'uncond':>8}{'mompnnC':>8}{'TM中位':>8}{'TM>0.7':>8}{'TM<0.5':>8}{'RMSD':>7}{'pLDDT':>7}",
          flush=True)

    for tj in sorted(root.glob("*/transfer.json")):
        pdb = tj.parent.name
        d = json.load(open(tj))
        summary[pdb] = {"L": d["L"], "mode": d["mode"], "native": d["native"],
                        "pH_arms": {}}
        for ph_s, arm in d["pH_arms"].items():
            arm_dir = tj.parent / f"pH{ph_s}"
            tm = read_csv_col(arm_dir / "tm.csv", "tm_score")
            rmsd = read_csv_col(arm_dir / "tm.csv", "rmsd")  # v3 D6：RMSD 辅助指标
            plddt = read_csv_col(arm_dir / "plddt.csv", "mean_plddt")
            # 无条件基线（同蛋白 pH7.4）
            uncond = None
            if args.uncond_root:
                uj = Path(args.uncond_root) / pdb / "transfer.json"
                if uj.exists():
                    u = json.load(open(uj))
                    ua = u["pH_arms"].get("7.4", {})
                    if ph_s in u["pH_arms"]:
                        ua = u["pH_arms"][ph_s]
                    uncond = ua.get("mean_charge")
            # MoMPNN-cond 对照
            mompnn = None
            if args.mompnn_root:
                mj = Path(args.mompnn_root) / pdb / "transfer.json"
                if mj.exists():
                    m = json.load(open(mj))
                    ma = m["pH_arms"].get(ph_s, {})
                    mompnn = ma.get("mean_charge")
            summary[pdb]["pH_arms"][ph_s] = {
                **arm,
                "uncond_mean_charge": (round(uncond, 2) if uncond is not None else None),
                "mompnn_cond_mean_charge": (round(mompnn, 2) if mompnn is not None else None),
                "tm_median": round(median(tm), 3) if tm else None,
                "tm_ge070": round(frac(tm, lo=0.7), 3) if tm else None,
                "tm_lt050": round(frac(tm, hi=0.5), 3) if tm else None,
                "rmsd_median": round(median(rmsd), 2) if rmsd else None,
                "plddt_median": round(median(plddt), 1) if plddt else None,
            }
            print(f"{pdb:<14}{ph_s:>5}{str(arm['target']):>7}"
                  f"{arm['mean_charge']:>7}{arm['dev']:>6}{arm['recovery']:>9}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['uncond_mean_charge']):>8}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['mompnn_cond_mean_charge']):>8}"
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
