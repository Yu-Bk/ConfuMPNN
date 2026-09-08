"""exp1 v2（均衡集）报告生成：并排 v1(旧偏酸/缺中性) + v2(三类均衡)。

读：
  prot v1: output/exp_control_prot/_analysis_summary.json
  prot v2: output/exp_control_prot_v2/_analysis_summary.json
  lig  v1: output/exp_control_lig/summary.json
  lig  v2: output/exp_control_lig_v2/summary.json
写：
  compare/report_2026-09-06_exp1_prot_barebackbone_v2.md
  compare/report_2026-09-06_exp1_lig_barebackbone_v2.md

用法：PYTHONPATH=code python code/tests/make_v2_report.py [--proto_only]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
ARMS = ["native", "n2", "p2", "n8", "p8"]
ARM_LABEL = {"native": "native", "n2": "n2(−2)", "p2": "p2(+2)",
             "n8": "n8(−8)", "p8": "p8(+8)"}
LIG_DQ = {"native": 0, "n2": -2, "p2": +2, "n8": -8, "p8": +8}


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


# ---------------- protein summaries ----------------
def prot_summary(which):
    if which == "v1":
        return load(ROOT / "output/exp_control_prot/_analysis_summary.json")
    return load(ROOT / "output/exp_control_prot_v2/_analysis_summary.json")


# ---------------- ligand summaries ----------------
def lig_summary(which):
    if which == "v1":
        return load(ROOT / "output/exp_control_lig/summary.json")
    return load(ROOT / "output/exp_control_lig_v2/summary.json")


def lig_prot_dict(s):
    return {p["pdb"]: p for p in s["proteins"]} if s else {}


def p_fmt(x, nd=2):
    return f"{x:+.{nd}f}" if isinstance(x, (int, float)) else "na"


def pct(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "na"


# =================================================================
# 蛋白模式
# =================================================================
def table_prot_selection():
    # 覆盖矩阵 + 三类表
    rows = [
        ("| 类别 | 代表 | L | native_q@7.4 | 来源 | 角色注记 |", "|---|---|---|---|---|---|"),
        ("| 酸 | 1CGE | 162 | −11.66 | `data/validation_pdbs/1CGE.pdb`（v12.2 蛋白 generalize 集 in-table）| 金属 CA+ZN，蛋白 featurize 忽略配体；表内 slope 1.198 |", None),
        ("| 中性 | 1BJ4 | 470 | +0.42 | `data/validation_pdbs/1BJ4.pdb`（in-table）| 近零电荷长蛋白；表内 slope 2.488 高增益→难臂备 Bsmall |", None),
        ("| 碱（补入）| 1LYZ | 129 | +7.13 | `data/transfer_test/1LYZ.pdb` | 溶菌酶高 pI 单体；**不在原蛋白已测集**；非表内→route B global 校准 |", None),
    ]
    return rows


def md_routeA_prot(prots, out_root_tag):
    lines = ["### route A：裸 backbone（MoMPNN 无条件 n=1000）净电荷分布 → 各电荷区占比（round 定心）"]
    lines.append(f"数据：`output/{out_root_tag}/<pdb>/route_A/per_seq.jsonl`。")
    lines.append("| 蛋白 | mean_q±std | native±1 | native±2 | n2±2 | p2±2 | n8±2 | p8±2 | mean rec |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for pdb, p in prots.items():
        a = p["routes"]["A"]
        f = a["fracs"]
        lines.append(
            f"| {pdb} | {p_fmt(a['mean_charge'])}±{a['std_charge']:.2f} | "
            f"{pct(f['native_round_tol1'])} | {pct(f['native_round_tol2'])} | "
            f"{pct(f['n2_round_tol2'])} | {pct(f['p2_round_tol2'])} | "
            f"{pct(f['n8_round_tol2'])} | {pct(f['p8_round_tol2'])} | "
            f"{a['mean_recovery']:.3f} |")
    return lines


def md_routeB_prot(prots, out_root_tag):
    lines = ["### route B：条件生成（v12.2 编码器，auto 校准）每臂达标率 + 增益 vs route A"]
    lines.append(f"数据：`output/{out_root_tag}/<pdb>/route_B/arm_<arm>/per_seq.jsonl`。"
                 "H2(mean口径)=mean|q−target|≤2；per-seq 命中=|q−target|≤2 占比；"
                 "增益 = B per-seq 命中 − A 同区±2（round 定心）。")
    lines.append("| 蛋白 | 臂 | target | B mean_q | B dev_of_mean | B seq命中≤2 | A 区±2 | 增益 | B rec |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for pdb, p in prots.items():
        for arm in ARMS:
            b = p["routes"]["B"]["by_arm"].get(arm)
            if not b:
                continue
            A = p["routes"]["A"]["fracs"][f"{arm}_round_tol2"]
            gain = b["per_seq_hit_le2"] - A
            lines.append(
                f"| {pdb} | {ARM_LABEL[arm]} | {b['target']:+d} | {p_fmt(b['mean_charge'])} | "
                f"{b['dev_of_mean']:.2f} | {pct(b['per_seq_hit_le2'])} | {pct(A)} | {p_fmt(gain)} | "
                f"{b['mean_recovery']:.3f} |")
    lines.append("")
    lines.append("### route B 汇总（mean口径 H2 / per-seq 总命中）")
    lines.append("| 蛋白 | B arm命中(H2) | B per-seq 总命中 |")
    lines.append("|---|---|---|")
    for pdb, p in prots.items():
        b = p["routes"]["B"]
        lines.append(f"| {pdb} | {b['n_arms_hit']}/{b['n_arms']} | {pct(b['per_seq_total_hit'])} |")
    return lines


def md_routeC_prot(prots, out_root_tag):
    lines = ["### route C：bias-only（charge-lookahead，strength 0.5，无校准）vs route B（同 seed 配对）"]
    lines.append(f"数据：`output/{out_root_tag}/<pdb>/route_C/arm_<arm>/per_seq.jsonl`。")
    lines.append("| 蛋白 | 臂 | B dev_of_mean | C dev_of_mean | B seq命中≤2 | C seq命中≤2 | B rec | C rec | B−C seq命中 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for pdb, p in prots.items():
        for arm in ARMS:
            b = p["routes"]["B"]["by_arm"].get(arm)
            c = p["routes"]["C"]["by_arm"].get(arm)
            if not b or not c:
                continue
            lines.append(
                f"| {pdb} | {ARM_LABEL[arm]} | {b['dev_of_mean']:.2f} | {c['dev_of_mean']:.2f} | "
                f"{pct(b['per_seq_hit_le2'])} | {pct(c['per_seq_hit_le2'])} | "
                f"{b['mean_recovery']:.3f} | {c['mean_recovery']:.3f} | "
                f"{p_fmt(b['per_seq_hit_le2'] - c['per_seq_hit_le2'])} |")
    return lines


def md_bsmall_prot(prots, out_root_tag):
    has = any("Bsmall" in p.get("routes", {}) for p in prots.values())
    if not has:
        return []
    lines = ["### route Bsmall：现场小样本标定附加组（route B 某臂差时触发）"]
    lines.append(f"数据：`output/{out_root_tag}/<pdb>/route_Bsmall/arm_<arm>/per_seq.jsonl`；"
                 "标定参数 `<pdb>_smallcal.json`。")
    lines.append("| 蛋白 | 臂 | Bsmall dev_of_mean | Bsmall seq命中≤2 | rec | slope/inter |")
    lines.append("|---|---|---|---|---|---|")
    for pdb, p in prots.items():
        if "Bsmall" not in p.get("routes", {}):
            continue
        sc = p["routes"]["Bsmall"]["smallcal"]
        for arm in ARMS:
            a = p["routes"]["Bsmall"]["by_arm"].get(arm)
            if not a:
                continue
            lines.append(
                f"| {pdb} | {ARM_LABEL[arm]} | {a['dev_of_mean']:.2f} | "
                f"{pct(a['per_seq_hit_le2'])} | {a['mean_recovery']:.3f} | "
                f"{sc['slope']:.2f}/{p_fmt(sc['intercept'])} |")
    return lines


def prot_report(v1, v2):
    out = []
    out.append("# 对比实验 exp1（蛋白模式）均衡集 v2 — 裸 backbone 无条件 vs ConfuMPNN 条件生成（2026-09-06）")
    out.append("")
    out.append("> 归属：compare/。v1（旧）= 偏酸/中性覆盖 {1AZM(−1.7), 1AS2(−2.7), 1BJ4(+0.4)}；"
               "v2（本报告主结论）= 三类均衡 {1CGE(−11.7 酸), 1BJ4(+0.4 中性), 1LYZ(+7.1 碱补入)}。"
               "v1 结果并排保留作覆盖对照；**结论只从 v2 下**。"
               "计划 compare/plan_exp1_barebackbone_control.md；session session/2026-09-06_exp1_v2_run.md。")
    out.append("")
    out.append("## 0 选样（native_q@7.4 先定类再跑）")
    for row in table_prot_selection():
        for part in row:
            if part:
                out.append(part)
    out.append("")
    out.append("**覆盖对照**：蛋白已测集（v12.2 generalize 单链单体）负/中性充足、正电类缺失 → "
               "本任务补入 1LYZ（+7.13）。v1 的 1AZM/1AS2 保留在旧报告中作覆盖。")
    out.append("")
    out.append("## 1 route A（裸 backbone 电荷分布 = 天然基线）")
    out.append("")
    out += md_routeA_prot(v2["proteins"], "exp_control_prot_v2")
    out.append("")
    out.append("### 与 v1（偏酸/中性）route A 对照")
    out.append("| 集 | 蛋白 | native_q | A mean_q±std | native±2 | n8±2 | p8±2 |")
    out.append("|---|---|---|---|---|---|---|")
    for tag, s in (("v1", v1), ("v2", v2)):
        for pdb, p in s["proteins"].items():
            a = p["routes"]["A"]
            f = a["fracs"]
            out.append(f"| {tag} | {pdb} | {p_fmt(p['native_charge'])} | "
                       f"{p_fmt(a['mean_charge'])}±{a['std_charge']:.2f} | "
                       f"{pct(f['native_round_tol2'])} | {pct(f['n8_round_tol2'])} | "
                       f"{pct(f['p8_round_tol2'])} |")
    out.append("")
    out.append("## 2 route B（条件生成 + 校准）与增益")
    out.append("")
    out += md_routeB_prot(v2["proteins"], "exp_control_prot_v2")
    out.append("")
    out += md_bsmall_prot(v2["proteins"], "exp_control_prot_v2")
    out.append("")
    out.append("## 3 route C（bias-only 消融）")
    out.append("")
    out += md_routeC_prot(v2["proteins"], "exp_control_prot_v2")
    out.append("")
    out.append("## 4 v1 vs v2 汇总对照（覆盖变化）")
    out.append("")
    out.append("| 集 | 覆盖 | 蛋白 | B H2 | B per-seq | C H2 | C per-seq | 注 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for tag, s, cover, note in (
            ("v1", v1, "偏酸/中性", "旧集无碱；1BJ4 需 Bsmall"),
            ("v2", v2, "酸/中性/碱", "本报告主结论；1LYZ 补碱")):
        for pdb, p in s["proteins"].items():
            b = p["routes"]["B"]
            c = p["routes"].get("C") or {}
            cH = f"{c.get('n_arms_hit')}/{c.get('n_arms')}" if c.get("by_arm") else "--"
            cS = pct(c.get("per_seq_total_hit")) if c.get("by_arm") else "--"
            out.append(f"| {tag} | {cover} | {pdb} | {b['n_arms_hit']}/{b['n_arms']} | "
                       f"{pct(b['per_seq_total_hit'])} | {cH} | {cS} | {note} |")
    out.append("")
    out.append("## 5 数据文件")
    out.append("- 蛋白 v2：`output/exp_control_prot_v2/_analysis_summary.json`、`_report_tables.md`；逐臂 `route_{A,B,Bsmall,C}/arm_*/per_seq.jsonl` + `seqs.fa`")
    out.append("- 蛋白 v1：`output/exp_control_prot/_analysis_summary.json`")
    out.append("- 日志：`log/exp_control_prot_v2_{chain,A,B,C}.log`")
    return "\n".join(out) + "\n"


# =================================================================
# 配体模式
# =================================================================
def lig_routeA_md(ps, tag):
    lines = []
    for pdb, p in ps.items():
        A = p.get("routeA")
        if not A:
            continue
        lines.append(f"| {tag} | {pdb} | {p_fmt(p['native_charge'])} | "
                     f"{p_fmt(A['mean_q'])}±{A['std_q']:.2f} | {A['mean_rec']:.3f} | "
                     f"{pct(A['frac_native_pm1'])} | "
                     f"{pct(A['frac_by_arm_pm1'].get('native'))} | "
                     f"{pct(A['frac_by_arm_pm1'].get('n2'))} | "
                     f"{pct(A['frac_by_arm_pm1'].get('p2'))} | "
                     f"{pct(A['frac_by_arm_pm1'].get('n8'))} | "
                     f"{pct(A['frac_by_arm_pm1'].get('p8'))} |")
    return lines


def lig_routeB_md(ps, tag, whichB):
    """whichB: 'routeB' 或 'routeB_cal'（v2 的 routeB 即校准口径）。"""
    lines = []
    for pdb, p in ps.items():
        B = p.get(whichB) or {}
        for arm in ARMS:
            a = B.get(arm)
            if not a:
                continue
            af1 = (p["routeA"].get("frac_by_arm_pm1") or {}).get(arm)
            lines.append(
                f"| {tag} | {pdb} | {ARM_LABEL[arm]} | {a['target']:+d} | "
                f"{p_fmt(a['mean_q'])} | {a['mean_dev']:+.2f} | "
                f"{pct(a['per_seq_hit_2'])} | {pct(af1)} | {p_fmt(a['per_seq_hit_2'] - af1) if af1 is not None else 'na'} | "
                f"{a['mean_rec']:.3f} | cr={pct(a.get('charged_ratio'), 2)} |")
    return lines


def lig_summary_md(ps):
    lines = []
    for pdb, p in ps.items():
        bb = []
        for rn, hitname in (("routeB", "routeB"), ("routeC", "routeC")):
            B = p.get(rn)
            if B:
                n_hit = sum(1 for a in B.values() if a.get("arm_mean_hit_2"))
                tot = float(sum(a["per_seq_hit_2"] for a in B.values()) / max(1, len(B)))
                bb.append(f"{rn}: {n_hit}/{len(B)} H2, per-seq {tot:.3f}")
        lines.append(f"| {pdb} | native_q={p_fmt(p['native_charge'])} | " + " ; ".join(bb) + " |")
    return lines


def lig_routeBsmall_md(v2s, out_tag):
    """读取 <out_dir>/<pdb>/routeB_small/arm_<arm>/sequences.json（现场小样本校准附加组）。"""
    lines = []
    root = ROOT / "output" / out_tag
    for pdb in sorted(v2s):
        bs_dir = root / pdb / "routeB_small"
        if not (bs_dir / f"{pdb}_smallcal.json").exists():
            continue
        sc = json.load(open(bs_dir / f"{pdb}_smallcal.json"))
        for arm in ARMS:
            adir = bs_dir / f"arm_{arm}"
            if not (adir / "sequences.json").exists():
                continue
            d = json.load(open(adir / "sequences.json"))
            q = np.array(d["charges"], dtype=float)
            tgt = d["target"]
            dev = q.mean() - tgt
            hit = float(np.mean(np.abs(q - tgt) <= 2.0))
            rec = float(np.mean(d["recs"]))
            lines.append(
                f"| {pdb} | {ARM_LABEL[arm]} | {tgt:+d} | {q.mean():+.2f} | {dev:+.2f} | "
                f"{pct(hit)} | {rec:.3f} | {sc['slope']:.2f}/{sc['intercept']:+.2f} |")
    if lines:
        lines.insert(0, "### 附加组 routeB_small：现场小样本标定（route B 某臂 |dev|>2 时触发，n=1000/臂）")
        lines.insert(1, f"数据：`output/{out_tag}/<pdb>/routeB_small/arm_<arm>/sequences.json`；标定 `{pdb}_smallcal.json`。")
        lines.insert(2, "| 蛋白 | 臂 | target | mean_q | mean_dev | seq命中≤2 | rec | slope/inter |")
        lines.insert(3, "|---|---|---|---|---|---|---|---|")
        lines.append("")
    return lines


def lig_report(v1s, v2s):
    v1 = lig_prot_dict(v1s)
    v2 = lig_prot_dict(v2s)
    out = []
    out.append("# 对比实验 exp1（配体模式）均衡集 v2 — 裸 backbone vs ConfuMPNN 条件生成（2026-09-06）")
    out.append("")
    out.append("> 归属：compare/。v1（旧）= {5O60_E(+11.2 好), 1CGE(−11.7 中), 2FEO(−6.9 难)} —— 有正有负但**无 native≈0 中性**；"
               "v2（本报告主结论）= 三类均衡 {1CGE(−11.7 酸), 1BJ4(+0.4 中性), 5O60_E(+11.2 碱)}。"
               "v1 结果并排保留作覆盖对照；**结论只从 v2 下**。"
               "计划 compare/plan_exp1_barebackbone_control.md；session session/2026-09-06_exp1_v2_run.md。")
    out.append("")
    out.append("## 0 选样")
    out.append("| 类别 | 代表 | L | native_q@7.4 | 来源 | 注 |")
    out.append("|---|---|---|---|---|---|")
    out.append("| 酸 | 1CGE | 162 | −11.66 | `data/validation_pdbs/1CGE.pdb`（in-10, CA+ZN）| v14 clean slope 0.983 |")
    out.append("| 中性（补入）| 1BJ4 | 470 | +0.42 | `data/validation_pdbs/1BJ4.pdb`（in-10, PLP）| **v1 配体缺中性**；v14 clean slope 2.179, 泛化 5 臂 dev<1 |")
    out.append("| 碱 | 5O60_E | 209 | +11.18 | `data/validation_pdbs/5O60_E.pdb`（in-10, rRNA）| RNA 结合蛋白；v14 clean slope 1.78 |")
    out.append("")
    out.append("**覆盖对照**：配体 in-10 全集负/正都有、缺 native≈0；1BJ4(+0.4) 为最接近 0 的表内代表。")
    out.append("")
    out.append("## 1 route A（裸 LigandMPNN n=1000 电荷分布）")
    out.append("")
    out.append("| 集 | 蛋白 | native_q | A mean_q±std | rec | native±1 | native臂±1 | n2±1 | p2±1 | n8±1 | p8±1 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    out += lig_routeA_md(v1, "v1")
    out += lig_routeA_md(v2, "v2")
    out.append("")
    out.append("## 2 route B（v2 = per-protein 校准注入；v1 = direct 与 B_cal 两口径）")
    out.append("")
    out.append("表头：A±1 = 裸 backbone 落在该 target±1 的占比（天然基线）。增益 = B per-seq 命中 − A±1。")
    out.append("")
    out.append("### v2 route B（校准，主结论）")
    out.append("| 集 | 蛋白 | 臂 | target | B mean_q | B mean_dev | B seq命中≤2 | A±1 | 增益 | rec |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    out += lig_routeB_md(v2, "v2", "routeB")
    out.append("")
    out.append("### v1 route B_direct 与 B_cal（旧集，覆盖对照）")
    out.append("| 集 | 蛋白 | 臂 | direct mean_dev | direct seq命中≤2 | B_cal mean_dev | B_cal seq命中≤2 |")
    out.append("|---|---|---|---|---|---|---|")
    for pdb, p in v1.items():
        for arm in ARMS:
            bd = p.get("routeB", {}).get(arm)
            bc = p.get("routeB_cal", {}).get(arm)
            if not (bd or bc):
                continue
            d_md = f"{bd['mean_dev']:+.2f}" if bd else "--"
            d_hit = pct(bd["per_seq_hit_2"]) if bd else "--"
            c_md = f"{bc['mean_dev']:+.2f}" if bc else "--"
            c_hit = pct(bc["per_seq_hit_2"]) if bc else "--"
            out.append(f"| v1 | {pdb} | {ARM_LABEL[arm]} | {d_md} | {d_hit} | {c_md} | {c_hit} |")
    out.append("")
    out += lig_routeBsmall_md(v2, "exp_control_lig_v2")
    out.append("### 配体 route 汇总")
    out.append("| 集 | 蛋白 | 汇总 |")
    out.append("|---|---|---|")
    for tag, ps in (("v1", v1), ("v2", v2)):
        for pdb, p in ps.items():
            parts = []
            for rn in ("routeA", "routeB", "routeB_cal", "routeC"):
                if p.get(rn):
                    if rn == "routeA":
                        parts.append(f"A native±1 {pct(p['routeA']['frac_native_pm1'])}")
                    else:
                        h = sum(1 for a in p[rn].values() if a.get("arm_mean_hit_2"))
                        tot = float(sum(a["per_seq_hit_2"] for a in p[rn].values()) / max(1, len(p[rn])))
                        parts.append(f"{rn} {h}/{len(p[rn])} H2 seq{tot:.3f}")
            out.append(f"| {tag} | {pdb} | " + " ; ".join(parts) + " |")
    out.append("")
    out.append("## 3 route C（bias-only，n≥200/臂，strength 0.5，无校准）")
    out.append("")
    out.append("| 集 | 蛋白 | 臂 | C mean_dev | C seq命中≤2 | rec | cr |")
    out.append("|---|---|---|---|---|---|---|")
    for tag, ps in (("v1", v1), ("v2", v2)):
        for pdb, p in ps.items():
            C = p.get("routeC") or {}
            for arm in ARMS:
                a = C.get(arm)
                if not a:
                    continue
                out.append(f"| {tag} | {pdb} | {ARM_LABEL[arm]} | {a['mean_dev']:+.2f} | "
                           f"{pct(a['per_seq_hit_2'])} | {a['mean_rec']:.3f} | "
                           f"{pct(a.get('charged_ratio'), 2)} |")
    out.append("")
    out.append("## 4 数据文件")
    out.append("- 配体 v2：`output/exp_control_lig_v2/summary.json`；逐臂 `{pdb}/route{A,B,C}/arm_*/sequences.json` + `seqs.fa`")
    out.append("- 配体 v1：`output/exp_control_lig/summary.json`")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto_only", action="store_true")
    ap.add_argument("--lig_only", action="store_true")
    args = ap.parse_args()
    if not args.lig_only:
        v1 = prot_summary("v1")
        v2 = prot_summary("v2")
        if not v1:
            print("WARN: prot v1 summary 缺失，跳过 protein v1 对照部分")
        if not v2:
            raise SystemExit("prot v2 summary 缺失：先跑 analyze_control.py --out_dir output/exp_control_prot_v2")
        txt = prot_report(v1, v2)
        outp = ROOT / "compare" / "report_2026-09-06_exp1_prot_barebackbone_v2.md"
        outp.write_text(txt)
        print("wrote", outp)
    if not args.proto_only:
        v1s = lig_summary("v1")
        v2s = lig_summary("v2")
        if not v2s:
            raise SystemExit("lig v2 summary 缺失：先跑 analyze_exp_control_lig.py --in_dir output/exp_control_lig_v2")
        txt = lig_report(v1s, v2s)
        outp = ROOT / "compare" / "report_2026-09-06_exp1_lig_barebackbone_v2.md"
        outp.write_text(txt)
        print("wrote", outp)


if __name__ == "__main__":
    main()
