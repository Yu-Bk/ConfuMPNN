"""v12.2 4 长蛋白两口径 H2 汇总（paper gap1 v12.2 基线）。

读取 output/paper_gap1_v122_long/generalization_{small,bigglobal}/protein/<pdb>/validation.json
的 arms dev → H2（dev<=2.0 命中），并与 v12.3 同口径权威值对照。

用法（项目根，confumpnn 环境）：
  PYTHONPATH=code python code/tests/ligand_v9/summarize_v12_2_long_gap1.py
"""
import json
from pathlib import Path

ROOT = Path("output/paper_gap1_v122_long")
PROTS = ["1A65", "1BJ4", "13BB", "1CDG"]
ARM_ORDER = ["native", "n2", "p2", "n8", "p8"]
H2_DEV = 2.0

# v12.3 权威对照值（主 session 已核验）
V123 = {
    "small": {"1A65": 2, "1BJ4": 5, "13BB": 1, "1CDG": 4},
    "bigglobal": {"1A65": 1, "1BJ4": 1, "13BB": 4, "1CDG": 3},
}

CALDIR = {"small": "generalization_small", "bigglobal": "generalization_bigglobal"}

def load_arms(caliber, pdb):
    vj = ROOT / CALDIR[caliber] / "protein" / pdb / "validation.json"
    if not vj.exists():
        return None
    d = json.load(open(vj))
    arms = {}
    for a in ARM_ORDER:
        if a in d["arms"]:
            x = d["arms"][a]
            arms[a] = {"target": x["target"], "mean": x["mean_charge"], "dev": x["dev"]}
    return d, arms

def count_h2(arms):
    return sum(1 for a in ARM_ORDER if a in arms and arms[a]["dev"] <= H2_DEV)

def fmt_arms(arms):
    return "  ".join(
        f"{a}:dev{arms[a]['dev']:.2f}{'*' if arms[a]['dev']<=H2_DEV else ''}"
        for a in ARM_ORDER if a in arms)

def main():
    out_lines = []
    out_lines.append("# v12.2 4 长蛋白两口径 H2 汇总（paper_gap1，2026-09-03）")
    out_lines.append("")
    out_lines.append("判据：H2 = |mean_charge - target| <= 2.0 命中；每臂 n30；native/n2/p2/n8/p8。")
    out_lines.append("")
    for caliber in ["small", "bigglobal"]:
        out_lines.append(f"## 口径：{caliber}")
        header = f"{'pdb':6s} {'native_q':>8s}  {'native':>7s} {'n2':>7s} {'p2':>7s} {'n8':>7s} {'p8':>7s} | H2  | v12.3 {caliber}"
        out_lines.append(header)
        out_lines.append("-" * len(header))
        n_ok = 0
        for pdb in PROTS:
            d, arms = load_arms(caliber, pdb)
            if arms is None:
                out_lines.append(f"{pdb:6s}  (缺 validation.json)")
                continue
            h2 = count_h2(arms)
            n_ok += h2
            row = (f"{pdb:6s} {d['native_charge']:>8.2f}  " +
                   "  ".join(f"{arms[a]['dev']:6.2f}{'*' if arms[a]['dev']<=H2_DEV else ' '}"
                             for a in ARM_ORDER if a in arms))
            out_lines.append(f"{row} | {h2}/5  | {V123[caliber].get(pdb, '?')}/5")
        out_lines.append(f"\nv12.2 合计 H2：{n_ok}/{len(PROTS)*5}")
        out_lines.append("")
    report = "\n".join(out_lines)
    print(report)
    (ROOT / "summary_h2_table.txt").write_text(report)
    print(f"\n已写 {ROOT / 'summary_h2_table.txt'}")

if __name__ == "__main__":
    main()
