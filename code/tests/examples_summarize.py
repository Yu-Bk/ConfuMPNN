"""任务28 汇总：示例蛋白 × pH/预设 生成对比表。
输入：code/output/examples/{pdb}_{preset}_pH{pH}/ 与 {pdb}_default_pH{pH}/ 的 summary.json
输出：打印对比表 + 写 code/output/examples/summary_table.csv
"""
import csv
import glob
import json
import os

ROOT = "/data/nfs/IC/baokun_yu/ConfuMPNN/code/output/examples"


def parse_name(name):
    """name → (pdb, preset, pH)。如 1BC8_membrane_pH7.4 / 1UBQ_default_pH5.5"""
    parts = name.rsplit("_", 2)
    pdb, preset = parts[0], parts[1]
    ph = parts[2][2:]
    return pdb, preset, ph


def main():
    rows = []
    print(f"{'蛋白':<6}{'预设':<22}{'pH':<6}{'平均电荷':<10}{'±std':<8}{'native电荷':<10}{'样本':<6}")
    for d in sorted(glob.glob(os.path.join(ROOT, "*/"))):
        name = os.path.basename(d.rstrip("/"))
        sj_path = os.path.join(d, "summary.json")
        if not os.path.exists(sj_path):
            continue
        sj = json.load(open(sj_path))
        pdb, preset, ph = parse_name(name)
        rows.append([pdb, preset, ph, sj["mean_charge"], sj["std_charge"],
                     sj["native_charge"], sj["num_samples"]])
        print(f"{pdb:<6}{preset:<22}{ph:<6}{sj['mean_charge']:+8.2f}{sj['std_charge']:8.2f}"
              f"{sj['native_charge']:+10.2f}{sj['num_samples']:<6}")
    with open(os.path.join(ROOT, "summary_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb", "preset", "pH", "mean_charge", "std_charge", "native_charge", "n"])
        w.writerows(rows)
    print(f"\n已写 {ROOT}/summary_table.csv")


if __name__ == "__main__":
    main()
