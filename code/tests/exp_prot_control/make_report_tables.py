"""exp_control_prot — 从 _analysis_summary.json 生成报告 Markdown 表格片段。

用法：先跑 analyze_control.py 生成 _analysis_summary.json，再跑本脚本。
输出：output/exp_control_prot/_report_tables.md（含每个表的数据文件路径注解）
"""
import json
from pathlib import Path

OUT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN/output/exp_control_prot")
ARMS = ["native", "n2", "p2", "n8", "p8"]
ARM_LABEL = {"native": "native", "n2": "n2(−2)", "p2": "p2(+2)",
             "n8": "n8(−8)", "p8": "p8(+8)"}


def data_path(pdb, route, arm=None):
    base = f"output/exp_control_prot/{pdb}/route_{route}"
    if arm is None:
        return f"`{base}/per_seq.jsonl`（汇总 `{base}/summary.json`）"
    return f"`{base}/arm_{arm}/per_seq.jsonl`（逐臂 `{base}/arm_{arm}/arm_summary.json`）"


def main():
    S = json.load(open(OUT / "_analysis_summary.json"))
    lines = []
    prots = S["proteins"]
    # ---- exp1 表 1：蛋白选择依据 ----
    lines.append("### 表1 测试蛋白选择依据（既有 v12.2 per-protein / no-leak H2 与 recovery）")
    lines.append("数据源：`output/generalization_v12_2_calib/protein/<pdb>/validation.json`；"
                 "`analysis/report/2026-08-31_v12_2_diag.md` §七/§八/§十。")
    lines.append("| 蛋白 | L | cat | native_q@7.4 | per-protein H2 | no-leak(global) | recovery | 角色 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    src = {
        "1AZM": ("small_mol", "5/5", "4/5", "0.45", "好"),
        "1AS2": ("rna", "5/5", "2/5", "0.34", "中"),
        "1BJ4": ("long", "0/5", "1/5", "0.41", "难"),
    }
    for pdb in prots:
        m = S["proteins"][pdb]
        cat, php, nl, rec, role = src[pdb]
        lines.append(f"| {pdb} | {m['L']} | {cat} | {m['native_charge']:+.2f} | {php} | {nl} | {rec} | {role} |")
    lines.append("")

    # ---- exp1 route A 表 ----
    lines.append("### 表2 route A：裸 backbone（MoMPNN 无条件 n=1000）净电荷分布 → 各电荷区占比")
    lines.append("数据：`output/exp_control_prot/<pdb>/route_A/per_seq.jsonl`。"
                 "定心 = round(native_q)+Δ（对齐 route B target）；±1/±2 两种容差。")
    lines.append("| 蛋白 | mean_q±std | native±1 | native±2 | n2±1 | n2±2 | p2±1 | p2±2 | n8±1 | n8±2 | p8±1 | p8±2 | mean rec |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pdb in prots:
        a = S["proteins"][pdb]["routes"]["A"]
        f = a["fracs"]
        lines.append(
            f"| {pdb} | {a['mean_charge']:+.2f}±{a['std_charge']:.2f} | "
            f"{f['native_round_tol1']:.4f} | {f['native_round_tol2']:.4f} | "
            f"{f['n2_round_tol1']:.4f} | {f['n2_round_tol2']:.4f} | "
            f"{f['p2_round_tol1']:.4f} | {f['p2_round_tol2']:.4f} | "
            f"{f['n8_round_tol1']:.4f} | {f['n8_round_tol2']:.4f} | "
            f"{f['p8_round_tol1']:.4f} | {f['p8_round_tol2']:.4f} | "
            f"{a['mean_recovery']:.3f} |")
    lines.append("")

    # ---- exp1 route B per-arm 表（含增益）----
    lines.append("### 表3 route B：v12.2 条件生成（per-protein 校准）每臂达标率 + 增益")
    lines.append("数据：`output/exp_control_prot/<pdb>/route_B/arm_<arm>/per_seq.jsonl`。"
                 "H2(mean口径)=该臂 mean|q−target|≤2；per-seq 命中=|q−target|≤2 的序列占比；"
                 "增益 = B per-seq 命中 − A 同区±2 占比（round 定心）。")
    lines.append("| 蛋白 | 臂 | target | B mean_q | B mean_dev | B seq命中≤2 | A 区±2 | 增益 | B rec |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for pdb in prots:
        for arm in ARMS:
            B = S["proteins"][pdb]["routes"]["B"]["by_arm"].get(arm)
            if not B:
                continue
            A = S["proteins"][pdb]["routes"]["A"]["fracs"][f"{arm}_round_tol2"]
            gain = B["per_seq_hit_le2"] - A
            lines.append(
                f"| {pdb} | {ARM_LABEL[arm]} | {B['target']:+d} | {B['mean_charge']:+.2f} | "
                f"{B['dev_of_mean']:.2f} | {B['per_seq_hit_le2']:.3f} | {A:.4f} | {gain:+.3f} | "
                f"{B['mean_recovery']:.3f} |")
    lines.append("")

    # ---- exp1 汇总（均值口径 H2 与 per-seq 总命中）----
    lines.append("### 表4 exp1 汇总：route A/B 各蛋白 mean口径 H2 与 per-seq 总命中")
    lines.append("| 蛋白 | route A native±2 占比 | route B 臂命中(H2) | route B per-seq 总命中 |")
    lines.append("|---|---|---|---|")
    for pdb in prots:
        A = S["proteins"][pdb]["routes"]["A"]["fracs"]["native_round_tol2"]
        B = S["proteins"][pdb]["routes"]["B"]
        lines.append(f"| {pdb} | {A:.4f} | {B['n_arms_hit']}/{B['n_arms']} | "
                     f"{B['per_seq_total_hit']:.3f} |")
    lines.append("")

    # ---- exp2 route B vs C（native/n2/p2，n8/p8 附）----
    lines.append("### 表5 exp2：encoder(route B) vs bias-only(route C) 每臂对比")
    lines.append("数据：route B `output/exp_control_prot/<pdb>/route_B/arm_<arm>/per_seq.jsonl`；"
                 "route C `output/exp_control_prot/<pdb>/route_C/arm_<arm>/per_seq.jsonl`。"
                 "同 seed 配对。C 无校准、直接 target（bias strength=0.5）。")
    lines.append("| 蛋白 | 臂 | B mean_dev | C mean_dev | B seq命中≤2 | C seq命中≤2 | B rec | C rec | B−C seq命中 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for pdb in prots:
        for arm in ARMS:
            B = S["proteins"][pdb]["routes"]["B"]["by_arm"].get(arm)
            C = S["proteins"][pdb]["routes"]["C"]["by_arm"].get(arm)
            if not B or not C:
                continue
            lines.append(
                f"| {pdb} | {ARM_LABEL[arm]} | {B['dev_of_mean']:.2f} | {C['dev_of_mean']:.2f} | "
                f"{B['per_seq_hit_le2']:.3f} | {C['per_seq_hit_le2']:.3f} | "
                f"{B['mean_recovery']:.3f} | {C['mean_recovery']:.3f} | "
                f"{B['per_seq_hit_le2']-C['per_seq_hit_le2']:+.3f} |")
    lines.append("")
    lines.append("### 表6 exp2 汇总：encoder vs bias 各蛋白 mean口径 H2 与 per-seq 总命中")
    lines.append("| 蛋白 | B arm命中(H2) | C arm命中(H2) | B per-seq总命中 | C per-seq总命中 |")
    lines.append("|---|---|---|---|---|")
    for pdb in prots:
        B = S["proteins"][pdb]["routes"]["B"]
        C = S["proteins"][pdb]["routes"]["C"]
        lines.append(f"| {pdb} | {B['n_arms_hit']}/{B['n_arms']} | {C['n_arms_hit']}/{C['n_arms']} | "
                     f"{B['per_seq_total_hit']:.3f} | {C['per_seq_total_hit']:.3f} |")
    lines.append("")

    # ---- Bsmall（若存在）----
    has_bs = any("Bsmall" in S["proteins"][p].get("routes", {}) for p in prots)
    if has_bs:
        lines.append("### 表7 route Bsmall：现场小样本标定附加组（当 route B 某臂命中差）")
        lines.append("数据：`output/exp_control_prot/<pdb>/route_Bsmall/arm_<arm>/per_seq.jsonl`；"
                     "标定参数 `output/exp_control_prot/<pdb>/route_Bsmall/<pdb>_smallcal.json`。")
        lines.append("| 蛋白 | 臂 | Bsmall mean_dev | Bsmall seq命中≤2 | Bsmall rec | slope/inter |")
        lines.append("|---|---|---|---|---|---|")
        for pdb in prots:
            if "Bsmall" not in S["proteins"][pdb]["routes"]:
                continue
            sc = S["proteins"][pdb]["routes"]["Bsmall"]["smallcal"]
            for arm in ARMS:
                a = S["proteins"][pdb]["routes"]["Bsmall"]["by_arm"].get(arm)
                if not a:
                    continue
                lines.append(f"| {pdb} | {ARM_LABEL[arm]} | {a['dev_of_mean']:.2f} | "
                             f"{a['per_seq_hit_le2']:.3f} | {a['mean_recovery']:.3f} | "
                             f"{sc['slope']:.2f}/{sc['intercept']:+.2f} |")
        lines.append("")

    txt = "\n".join(lines)
    out = OUT / "_report_tables.md"
    out.write_text(txt)
    print(f"已写 {out}（{len(lines)} 行）")


if __name__ == "__main__":
    main()
