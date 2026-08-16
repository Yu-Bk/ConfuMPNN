"""E1b 扩展分析：电荷响应 + 三目标对比 + 联合可用率 + 留一蛋白检查。
输入：code/output/e1_ext/summary_all.csv（逐样本）、summary_cond.csv（按条件）
输出：analysis 目录下多个 csv + 打印关键结论
"""
import csv
import os

ROOT = "/data/nfs/IC/baokun_yu/ConfuMPNN"
ALL = os.path.join(ROOT, "code/output/e1_ext/summary_all.csv")
COND = os.path.join(ROOT, "code/output/e1_ext/summary_cond.csv")
OUT = os.path.join(ROOT, "code/output/e1_ext")


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_cond(name):
    """从条件名解析 (pH, target)。baseline 返回 (7.4, None)。"""
    if name == "baseline":
        return 7.4, None
    m = name.split("_")
    return float(m[0][2:]), float(m[1][1:])


def main():
    all_rows = load(ALL)
    conds = load(COND)

    # native 参考（每 PDB 取一条 native 的 %sol / pLDDT / Tm / TM-score）
    native = {}
    for r in all_rows:
        if r["is_native"] == "Y":
            p = r["pdb"]
            native.setdefault(p, {})
            for k in ["protsol", "plddt", "temberture_tm", "tm_score"]:
                if r[k] and k not in native[p]:
                    native[p][k] = float(r[k])

    # ---------- 1. 电荷响应矩阵 ----------
    print("=" * 70)
    print("【1】电荷响应：target 单调性 + pH 梯度（机制验证）")
    print("=" * 70)
    charge_out = []
    for r in conds:
        ph, tg = parse_cond(r["condition"])
        charge_out.append((r["pdb"], r["model"], r["condition"], ph,
                           tg if tg is not None else "", r["charge_mean"], r["charge_dev_mean"]))
    charge_out.sort()
    with open(os.path.join(OUT, "charge_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb", "model", "condition", "pH", "target", "charge_mean", "dev_mean"])
        w.writerows(charge_out)
    # 单调性检查（同 pdb/model/pH 下 target -5/0/+5 是否单调递增）
    print("monotonic(同pH,target升→charge升)?  pdb model pH 实际charge序列")
    for pdb in ["1BC8", "1CRN", "1UBQ", "2LZM"]:
        for model in ["ligand", "mompnn"]:
            for ph in [4.0, 7.4, 9.0]:
                got = {}
                for r in conds:
                    if r["pdb"] == pdb and r["model"] == model:
                        p2, tg = parse_cond(r["condition"])
                        if p2 == ph and tg is not None:
                            got[tg] = float(r["charge_mean"])
                if len(got) == 3:
                    vals = [got[t] for t in [-5, 0, 5]]
                    mono = vals[0] < vals[1] < vals[2]
                    print(f"  {pdb} {model:6s} pH{ph}: {[round(v,1) for v in vals]}  {'✓单调' if mono else '✗不单调'}")

    # ---------- 2. 三目标对比（按 PDB 汇总，全部条件样本） ----------
    print()
    print("=" * 70)
    print("【2】三目标对比：MoMPNN vs Ligand（按 PDB 汇总，条件样本均值）")
    print("=" * 70)
    metrics = [("plddt", "pLDDT"), ("tm_score", "TM-score"), ("protsol", "%sol"), ("temberture_tm", "Tm(°C)")]
    compare = []
    for pdb in ["1BC8", "1CRN", "1UBQ", "2LZM"]:
        acc = {m: {"ligand": [], "mompnn": []} for m, _ in metrics}
        for r in all_rows:
            if r["is_native"] == "Y" or r["pdb"] != pdb:
                continue
            for m, _ in metrics:
                if r[m]:
                    acc[m][r["model"]].append(float(r[m]))
        print(f"--- {pdb}  native %sol={native[pdb].get('protsol',0):.1f}  Tm={native[pdb].get('temberture_tm',0):.1f}  pLDDT={native[pdb].get('plddt',0):.1f}  TM={native[pdb].get('tm_score',0):.3f}")
        for m, label in metrics:
            li = acc[m]["ligand"]; mo = acc[m]["mompnn"]
            if li and mo:
                lv = sum(li) / len(li); mv = sum(mo) / len(mo)
                compare.append((pdb, label, lv, mv, mv - lv))
                print(f"  {label:8s} Ligand={lv:7.2f}  MoMPNN={mv:7.2f}  差={mv-lv:+6.2f}")
    with open(os.path.join(OUT, "compare_by_pdb.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb", "metric", "ligand", "mompnn", "diff"])
        w.writerows(compare)

    # ---------- 3. 联合可用率 ----------
    print()
    print("=" * 70)
    print("【3】联合可用率：pLDDT>80 且 |电荷偏差|≤0.3 且 %sol≥native（样本级）")
    print("=" * 70)
    for pdb in ["1BC8", "1CRN", "1UBQ", "2LZM"]:
        nat_sol = native[pdb].get("protsol", 0)
        for model in ["ligand", "mompnn"]:
            ok = total = 0
            for r in all_rows:
                if r["is_native"] == "Y" or r["pdb"] != pdb or r["model"] != model:
                    continue
                total += 1
                # 偏差：target 来自条件名
                tg = parse_cond(r["condition"])[1]
                if tg is None:
                    continue
                dev = abs(float(r["charge"]) - tg)
                ok3 = (float(r["plddt"]) > 80 and dev <= 0.3 and float(r["protsol"]) >= nat_sol)
                if ok3:
                    ok += 1
            print(f"  {pdb} {model:6s}: {ok}/{total} = {100*ok/total:.0f}%")

    # ---------- 4. 留一蛋白检查（每 PDB 三目标符号一致性） ----------
    print()
    print("=" * 70)
    print("【4】留一蛋白检查：MoMPNN 优势是否在所有 PDB 一致（符号检查）")
    print("=" * 70)
    signs = {}
    for pdb, metric, lv, mv, diff in compare:
        signs.setdefault(metric, []).append((pdb, diff))
    for metric, lst in signs.items():
        pos = sum(1 for _, d in lst if d > 0)
        print(f"  {metric:8s}: MoMPNN 占优 {pos}/{len(lst)} 个 PDB -> {'✓一致' if pos == len(lst) else ('✗ 不一致' if pos == 0 else '~部分一致')}")


if __name__ == "__main__":
    main()
