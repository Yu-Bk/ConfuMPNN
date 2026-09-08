#!/usr/bin/env python
"""L11 Exp2：聚合打分数据 → 每组表格 + 报告 report_exp2.md。

读：
    output/exp2/<mode>_q<t>/summary.json          → seq/charge/pI
    data/exp2/<mode>_q<t>/plddt.csv               → mean_plddt
    data/exp2/<mode>_q<t>/tm.csv                  → tm_score, rmsd
    data/exp2/<mode>_q<t>/seqs.fa.tm.csv          → mean_tm (TemBERTure)
    data/exp2/<mode>_q<t>/seqs.fa-protein_sol.csv → percent-sol, scaled-sol
    data/exp2/_native/...                          → native 对照
固定位 (I3 I5 I9 I34 I35 I89 I124 I131 I134 I135) 逐条校验 = native。

输出：L11design/test/report_exp2.md；每组 data/exp2/<dir>/metrics.csv；
      data/exp2/_per_group_summary.csv
用法（confumpnn 环境）：python L11design/test/exp2_analyze.py [--report PATH]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/data/nfs/IC/baokun_yu/ConfuMPNN/code")
sys.path.insert(0, "/data/nfs/IC/baokun_yu/ConfuMPNN/LigandMPNN")
from data_utils import parse_PDB, restype_int_to_str  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402

L11 = Path("/data/nfs/IC/baokun_yu/ConfuMPNN/L11design")
FIXED_RESNUMS = [3, 5, 9, 34, 35, 89, 124, 131, 134, 135]
FIXED_STR = "I3 I5 I9 I34 I35 I89 I124 I131 I134 I135"
TARGETS = [(6.87, "6.87"), (8.0, "8"), (10.0, "10"), (12.0, "12")]
MODES = ["protein", "ligand"]
TH_DEV = 2.0
TH_TM = 0.7


def native_seq_and_fixed():
    protein_dict, _, _, _, _ = parse_PDB(str(L11 / "input/L11.pdb"))
    S = protein_dict["S"]
    R = protein_dict["R_idx"].tolist()
    seq = "".join(restype_int_to_str[i] for i in S.tolist())
    fixed_i = [R.index(r) for r in FIXED_RESNUMS]
    return seq, fixed_i


def read_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def sample_name_to_idx(name):
    return int(name.split()[0].replace("sample_", ""))


def compute_recovery(seq, native):
    return sum(a == b for a, b in zip(seq, native)) / len(native)


def load_group(mode, tstr, tf, native, fixed_i):
    """返回 per-sample dict 列表。缺失的项置 None。"""
    outd = L11 / "output/exp2" / f"{mode}_q{tstr}"
    dird = L11 / "data/exp2" / f"{mode}_q{tstr}"
    sj = outd / "summary.json"
    if not sj.is_file():
        return None, f"缺 {sj}"
    s = json.load(open(sj))
    seqs = s["sequences"]

    plddt = {sample_name_to_idx(r["name"]): float(r["mean_plddt"]) for r in read_csv(dird / "plddt.csv")} \
        if (dird / "plddt.csv").is_file() else {}
    tm = {}
    rmsd = {}
    if (dird / "tm.csv").is_file():
        for r in read_csv(dird / "tm.csv"):
            i = int(r["name"].split("_")[-1])
            tm[i] = float(r["tm_score"]) if r["tm_score"] != "None" and r["tm_score"] else None
            rmsd[i] = float(r["rmsd"]) if r["rmsd"] != "None" and r["rmsd"] else None
    tmc = {}
    if (dird / "seqs.fa.tm.csv").is_file():
        for r in read_csv(dird / "seqs.fa.tm.csv"):
            i = int(r["name"].split("_")[-1])
            tmc[i] = float(r["mean_tm"]) if r["mean_tm"] != "None" else None
    sol = {}
    ps = dird / "seqs.fa-protein_sol.csv"
    if ps.is_file():
        for r in read_csv(ps):
            nm = r["ID"].split()[0]
            if nm.startswith("sample_"):
                i = int(nm.replace("sample_", ""))
                sol[i] = float(r["percent-sol"]) if r["percent-sol"] else None

    recs = []
    for i, x in enumerate(seqs, 1):
        seq = x["seq"]
        ch = x["charge"]
        if x["pI"] is None:
            pi = None
        else:
            pi = x["pI"]
        recs.append({
            "i": i, "seq": seq, "charge": ch, "pI": pi,
            "dev": ch - tf, "ok": abs(ch - tf) <= TH_DEV,
            "recovery": compute_recovery(seq, native),
            "nonfixed_rec": (sum(seq[j] == native[j] for j in range(len(native)) if j not in fixed_i)
                             / (len(native) - len(fixed_i))),
            "fixed_ok": all(seq[j] == native[j] for j in fixed_i),
            "fixed_n": sum(seq[j] == native[j] for j in fixed_i),
            "plddt": plddt.get(i), "tm": tm.get(i), "rmsd": rmsd.get(i),
            "tm_bert": tmc.get(i), "sol": sol.get(i),
        })
    return recs, None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _f(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def summarize(recs):
    out = {}
    n = len(recs)
    out["n"] = n
    out["charge_mean"] = _mean([r["charge"] for r in recs])
    out["charge_std"] = _std([r["charge"] for r in recs])
    out["dev_mean"] = _mean([r["dev"] for r in recs])
    out["ok_rate"] = sum(r["ok"] for r in recs) / n
    out["recovery"] = _mean([r["recovery"] for r in recs])
    out["nonfixed_rec"] = _mean([r["nonfixed_rec"] for r in recs])
    out["fixed_ok_rate"] = sum(r["fixed_ok"] for r in recs) / n
    out["plddt"] = _mean([r["plddt"] for r in recs])
    out["tm"] = _mean([r["tm"] for r in recs])
    out["rmsd"] = _mean([r["rmsd"] for r in recs])
    out["tm_bert"] = _mean([r["tm_bert"] for r in recs])
    out["sol"] = _mean([r["sol"] for r in recs])
    out["tm_ge70"] = sum(1 for r in recs if r["tm"] is not None and r["tm"] >= TH_TM) / n
    return out


def native_metrics(native):
    """从 _native 目录读取 native 对照指标。"""
    nd = L11 / "data/exp2/_native"
    d = {"plddt": None, "tm": None, "rmsd": None, "tm_bert": None, "sol": None}
    if (nd / "plddt.csv").is_file():
        for r in read_csv(nd / "plddt.csv"):
            if r["name"].startswith("native"):
                d["plddt"] = float(r["mean_plddt"])
    if (nd / "tm.csv").is_file():
        for r in read_csv(nd / "tm.csv"):
            if r["name"].startswith("native"):
                d["tm"] = float(r["tm_score"]) if r["tm_score"] != "None" else None
                d["rmsd"] = float(r["rmsd"]) if r["rmsd"] != "None" else None
    if (nd / "seqs.fa.tm.csv").is_file():
        for r in read_csv(nd / "seqs.fa.tm.csv"):
            if r["name"].startswith("native"):
                d["tm_bert"] = float(r["mean_tm"])
    ps = nd / "seqs.fa-protein_sol.csv"
    if ps.is_file():
        for r in read_csv(ps):
            if r["ID"].split()[0].startswith("native"):
                d["sol"] = float(r["percent-sol"])
    return d


def trend(rows, key):
    vals = [r[key] for r in rows]
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(L11 / "test/report_exp2.md"))
    args = ap.parse_args()

    native, fixed_i = native_seq_and_fixed()
    print(f"native len={len(native)} fixed={len(fixed_i)}")

    summary_rows = []  # 供 CSV
    group_stats = {}   # (mode,tstr) -> summarize dict
    issues = []
    per_group_recs = {}

    for mode in MODES:
        for tf, tstr in TARGETS:
            recs, err = load_group(mode, tstr, tf, native, fixed_i)
            if err:
                issues.append(f"{mode}_q{tstr}: {err}")
                continue
            per_group_recs[(mode, tstr)] = recs
            st = summarize(recs)
            group_stats[(mode, tstr)] = st
            # 写每组 metrics.csv
            mdir = L11 / "data/exp2" / f"{mode}_q{tstr}"
            mdir.mkdir(parents=True, exist_ok=True)
            with open(mdir / "metrics.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sample", "charge", "dev", "ok", "recovery", "nonfixed_recovery",
                            "fixed_n", "plddt", "tm", "rmsd", "tm_bert", "sol"])
                for r in recs:
                    w.writerow([r["i"], _f(r["charge"], 2), _f(r["dev"], 2), int(r["ok"]),
                                _f(r["recovery"], 4), _f(r["nonfixed_rec"], 4), r["fixed_n"],
                                _f(r["plddt"], 2), _f(r["tm"], 3), _f(r["rmsd"], 2),
                                _f(r["tm_bert"], 2), _f(r["sol"], 2)])
            # summary rows csv
            summary_rows.append([mode, tstr, st["n"], _f(st["charge_mean"], 3), _f(st["charge_std"], 3),
                                 _f(st["dev_mean"], 3), _f(st["ok_rate"], 4), _f(st["recovery"], 4),
                                 _f(st["nonfixed_rec"], 4), _f(st["fixed_ok_rate"], 4), _f(st["plddt"], 3),
                                 _f(st["tm"], 3), _f(st["rmsd"], 2), _f(st["tm_bert"], 2), _f(st["sol"], 2),
                                 _f(st["tm_ge70"], 4)])
            print(f"aggregated {mode}_q{tstr}: ok={st['ok_rate']:.2f} rec={st['recovery']:.3f} "
                  f"plddt={_f(st['plddt'])} tm={_f(st['tm'])} tmbert={_f(st['tm_bert'])} sol={_f(st['sol'])}")

    # write per-group summary csv
    with open(L11 / "data/exp2/_per_group_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "target", "n", "charge_mean", "charge_std", "dev_mean", "ok_rate",
                    "recovery", "nonfixed_recovery", "fixed_ok_rate", "plddt", "tm", "rmsd",
                    "tm_bert", "sol", "tm_ge70_frac"])
        w.writerows(summary_rows)

    nm = native_metrics(native)
    # ---------- 生成报告 ----------
    P = []
    A = P.append
    A("# L11 Exp2 多目标电荷可设计性（pH 7.4）\n")
    A("> 日期：2026-09-09；模式 × target{+6.87(native), +8, +10, +12} × n=300；固定链 I 残基 "
      f"`{FIXED_STR}`。")
    A(f"> 模型：蛋白 MoMPNN+v12.2（global 校准 slope 1.579/intercept −3.139）；"
      f"配体 LigandMPNN(atom25)+v14（global 校准 slope 1.492/intercept −1.260）。")
    A(f"> native 序列 {len(native)} aa，net_charge@7.4 = {net_charge(native, 7.4):+.2f}（计划 +6.87）。\n")

    A("## 0 固定位校验（逐条）")
    A("| 位点 | native | 说明 |")
    A("|---|---|---|")
    aa_map = {i: native[i] for i in fixed_i}
    for r, i in zip(FIXED_RESNUMS, fixed_i):
        A(f"| I{r} | {aa_map[i]} | 解码强制保持 |")
    A("")

    # 每模式表
    for mode in MODES:
        A(f"\n## {mode} 模式\n")
        A(f"| target | 电荷均值 | 电荷std | dev均值 | 达标率(\|dev\|≤2) | 回收 | 非固定回收 | "
          f"固定位全对率 | pLDDT | TM | RMSD | Tm(°C) | Sol(%) | TM≥0.7占比 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for tf, tstr in TARGETS:
            if (mode, tstr) not in group_stats:
                A(f"| {tf:+.2f} | — | | | | | | | | | | | |")
                continue
            st = group_stats[(mode, tstr)]
            A(f"| {tf:+.2f} | {_f(st['charge_mean'],2)} | {_f(st['charge_std'],2)} | "
              f"{_f(st['dev_mean'],2)} | {st['ok_rate']:.0%} | {st['recovery']:.3f} | "
              f"{_f(st['nonfixed_rec'],3)} | {st['fixed_ok_rate']:.0%} | "
              f"{_f(st['plddt'],2)} | {_f(st['tm'],3)} | {_f(st['rmsd'],2)} | "
              f"{_f(st['tm_bert'],1)} | {_f(st['sol'],1)} | {st['tm_ge70']:.0%} |")
        A("")
        A("native 对照：pLDDT={}，TM(vs native 骨架)={}，RMSD={}，Tm={}°C，Sol={}%。".format(
            _f(nm["plddt"], 2), _f(nm["tm"], 3), _f(nm["rmsd"], 2), _f(nm["tm_bert"], 1), _f(nm["sol"], 1)))
        # 趋势
        A("\n**target 6.87→12 电荷-性质趋势（mode={}）**".format(mode))
        A("| target | 电荷均值 | pLDDT | TM | Tm | Sol |")
        A("|---|---|---|---|---|---|")
        for tf, tstr in TARGETS:
            if (mode, tstr) not in group_stats:
                continue
            st = group_stats[(mode, tstr)]
            A(f"| {tf:+.2f} | {_f(st['charge_mean'],2)} | {_f(st['plddt'],2)} | "
              f"{_f(st['tm'],3)} | {_f(st['tm_bert'],1)} | {_f(st['sol'],1)} |")
        # 简单线性拟合电荷 vs target
        xs = [tf for tf, _ in TARGETS if (mode, _) in group_stats and group_stats[(mode, _)]["charge_mean"] is not None]
        ys = [group_stats[(mode, _)]["charge_mean"] for _, _ in TARGETS if (mode, _) in group_stats and group_stats[(mode, _)]["charge_mean"] is not None]
        if len(xs) >= 2 and len(ys) == len(xs):
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            a = my - b * mx
            A(f"\n电荷-响应线性拟合：`生成电荷 ≈ {b:.3f}·target {a:+.2f}`（单位斜率=1 为理想可控）。")
            if group_stats[(mode, "8")] and group_stats[(mode, "12")]:
                dq = group_stats[(mode, "12")]["charge_mean"] - group_stats[(mode, "6.87")]["charge_mean"]
                A(f"target +6.87→+12（Δ=+5.13）实测电荷均值变化 Δ={dq:+.2f}。")

    # 收敛性说明
    A("\n## 综合结论")
    all_done = len(per_group_recs) == len(MODES) * len(TARGETS)
    if not all_done:
        A(f"\n> ⚠️ 数据未齐全：缺失 {issues}。以下结论基于已有组。")
    # 结论按可用数据生成
    for mode in MODES:
        if all((mode, t) in group_stats for _, t in TARGETS):
            sts = [group_stats[(mode, t)] for _, t in TARGETS]
            ok_list = [st["ok_rate"] for st in sts]
            tm_list = [st["tm"] for st in sts if st["tm"] is not None]
            fold_ok = all(t >= TH_TM for t in tm_list)
            A(f"\n### {mode} 模式可设计性判读")
            ok_str = ", ".join(f"{x:.0%}" for x in ok_list)
            A(f"- 达标率(|dev|≤2)：{ok_str}（target 6.87→12）。")
            tm_str = ", ".join(_f(x, 2) for x in [st["tm"] for st in sts])
            pl_str = ", ".join(_f(x, 2) for x in [st["plddt"] for st in sts])
            A(f"- 折叠质量：各组 TM 均值 {tm_str}，pLDDT 均值 {pl_str}。")
            fo_str = ", ".join(f"{st['fixed_ok_rate']:.0%}" for st in sts)
            A("- 固定位：逐条全对率 " + fo_str + "。" if sts else "- 固定位：数据缺。")
            rec_str = ", ".join(f"{st['recovery']:.3f}" for st in sts)
            A(f"- 回收率：{rec_str}。")
    A("")

    report_path = Path(args.report)
    report_path.write_text("\n".join(P))
    print(f"\n报告已写入 {report_path}")


if __name__ == "__main__":
    main()
