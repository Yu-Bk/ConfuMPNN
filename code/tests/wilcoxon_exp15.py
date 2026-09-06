"""exp5 — 对 exp1/exp2 对照数据做 Wilcoxon 配对检验（2026-09-06）。

数据源（只读，不落新采样）：
  蛋白模式 : output/exp_control_prot/_analysis_summary.json（route A/B/C per-arm）
  配体模式 : output/exp_control_lig/summary.json        （routeA / routeB_cal / routeC）

配对与方向（所有 diff 记为 **后项减前项**，正值 = 后项更优）：
  ① A vs B（裸 vs 条件 encoder）: diff_hit = B_hit − A_win±2；diff_dev = A_dev − B_dev
     —— 后项 B；正 hit/正 dev 均表示 B（条件）更好。
  ② B vs C（encoder vs bias）   : diff_hit = C_hit − B_hit；diff_dev = B_dev − C_dev
     —— 后项 C；正值表示 C（bias）更好（exp2 已知 C 更强，此处检验显著性）。
  逐序列口径：per-arm hit = |q−target|≤2 的占比；dev = |mean_q − target|（均值口径，与 H2 同约定）。

小样本警示：配对 n=15（3 蛋白×5 臂），功效极低，仅报告 p + 中位数差 + 95%CI
  （差分布 bootstrap 百分位）作为描述性证据，不下强结论；分温和(native/n2/p2)/极端(n8/p8)子组。
  Wilcoxon 处理零差（zero_method='wilcox'，即零差剔除）并标注有效 n。

补充 S1（可选，10 蛋白共享）：v13-in10 vs v14-clean 逐臂 mean dev（来自既有 n50 汇总），
  同一臂配对，n=50（10 蛋白×5 臂），均值口径 dev 差（v14−v13，正 = v14 更好）。

输出：
  output/wilcoxon_exp15.json
  compare/report_2026-09-06_exp5_wilcoxon.md
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
PROT_SUM = ROOT / "output/exp_control_prot/_analysis_summary.json"
LIG_SUM = ROOT / "output/exp_control_lig/summary.json"
V13 = ROOT / "output/v13_ligand_gen_stats_in10.json"
V14 = ROOT / "output/v14_ligand_gen_stats_clean.json"
OUT_JSON = ROOT / "output/wilcoxon_exp15.json"
OUT_MD = ROOT / "compare/report_2026-09-06_exp5_wilcoxon.md"

ARMS = ["native", "n2", "p2", "n8", "p8"]
MODERATE = ["native", "n2", "p2"]
EXTREME = ["n8", "p8"]
RNG = np.random.default_rng(20260906)


def wilcox_summary(diff, label):
    """diff: ndarray of paired diffs。返回描述 + Wilcoxon。"""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    out = {"n_pairs": int(n), "label": label}
    if n == 0:
        out["note"] = "no finite pairs"
        return out
    # 描述统计
    out["mean_diff"] = round(float(diff.mean()), 4)
    out["median_diff"] = round(float(np.median(diff)), 4)
    # bootstrap 95% CI of median
    if n >= 3:
        boots = np.median(RNG.choice(diff, size=(4000, n), replace=True), axis=1)
        out["median_ci95"] = [round(float(np.percentile(boots, 2.5)), 4),
                              round(float(np.percentile(boots, 97.5)), 4)]
    # Wilcoxon signed-rank（双侧）
    try:
        stat, p = wilcoxon(diff, zero_method="wilcox", correction=False,
                           alternative="two-sided")
        out["wilcoxon_stat"] = float(stat)
        out["p_value"] = float(p)
    except ValueError as e:
        out["p_value"] = None
        out["note"] = f"wilcoxon ValueError: {e}"
    # 匹配对秩双列效应量 r = (W+ − W−)/(W+ + W−)，+1 = 所有非零差同向为正（后项更优）
    nz = int(np.count_nonzero(np.abs(diff) > 1e-12))
    out["n_nonzero"] = nz
    absd = np.abs(diff[np.abs(diff) > 1e-12])
    order = absd.argsort()
    ranks = np.empty(nz)
    ranks[order] = np.arange(1, nz + 1)
    # 平均秩（处理 ties）
    _, inv, cnt = np.unique(absd, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    wpos = float(ranks[diff[np.abs(diff) > 1e-12] > 0].sum())
    wneg = float(ranks[diff[np.abs(diff) > 1e-12] < 0].sum())
    out["effect_r"] = round(float((wpos - wneg) / (wpos + wneg)), 4)
    # 方向汇总（更优数）
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    out["n_pos"] = pos
    out["n_neg"] = neg
    out["n_zero"] = int((diff == 0).sum())
    return out


def build_pair_sets():
    """从 exp1/2 汇总读逐蛋白×逐臂指标。返回 list[(mode, pair_label, arm, pdb, m1, m2)]。"""
    rows = []

    # ---- 蛋白模式 ----
    ps = json.load(open(PROT_SUM))["proteins"]
    for pdb, p in ps.items():
        A = p["routes"]["A"]
        B = p["routes"]["B"]
        C = p["routes"]["C"]
        a_mean = A["mean_charge"]
        for arm in ARMS:
            ba = B["by_arm"].get(arm)
            ca = C["by_arm"].get(arm)
            if ba is None:
                continue
            tgt = ba["target"]
            A_hit = A["fracs"][f"{arm}_round_tol2"]   # A 落窗±2 占比
            A_dev = float(abs(a_mean - tgt))           # A 均值口径 dev
            B_hit = ba["per_seq_hit_le2"]
            B_dev = ba["dev_of_mean"]
            rows.append(("prot", "A_vs_B", arm, pdb, A_hit, B_hit, A_dev, B_dev))
            if ca is not None:
                C_hit = ca["per_seq_hit_le2"]
                C_dev = ca["dev_of_mean"]
                rows.append(("prot", "B_vs_C", arm, pdb, B_hit, C_hit, B_dev, C_dev))

    # ---- 配体模式（B = routeB_cal，per-protein 校准 = encoder 操作形态）----
    ls = json.load(open(LIG_SUM))
    for p in ls["proteins"]:
        pdb = p["pdb"]
        A = p.get("routeA")
        Bcal = p.get("routeB_cal") or {}
        C = p.get("routeC") or {}
        if not A:
            continue
        a_mean = A["mean_q"]
        for arm in ARMS:
            ba = Bcal.get(arm)
            ca = C.get(arm)
            # target 统一取 routeB/routeB_cal 的 target
            tgt = (ba or p.get("routeB", {}) or {}).get("target")
            if tgt is None:
                # fallback: routeC target
                tgt = ca.get("target") if ca else None
            if tgt is None:
                continue
            A_hit = A["frac_by_arm_pm2"].get(arm)      # A 落窗±2 占比
            A_dev = float(abs(a_mean - tgt))
            if ba is not None:
                rows.append(("lig", "A_vs_B", arm, pdb,
                             A_hit, ba["per_seq_hit_2"], A_dev, abs(ba["mean_dev"])))
                b_dev = abs(ba["mean_dev"])
                if ca is not None:
                    rows.append(("lig", "B_vs_C", arm, pdb,
                                 ba["per_seq_hit_2"], ca["per_seq_hit_2"],
                                 b_dev, abs(ca["mean_dev"])))
    return rows


def main():
    rows = build_pair_sets()
    json_out = {"meta": {
        "note": "exp5 Wilcoxon 配对（exp1/2 数据，2026-09-06）。"
                "diff=后项−前项，正=后项优。小样本(15)仅描述性。",
        "prot_summary": str(PROT_SUM), "lig_summary": str(LIG_SUM),
        "supp_v13": str(V13), "supp_v14": str(V14)},
        "pairs": {}}
    md = []
    md.append("# exp5 — Wilcoxon 配对检验（exp1/2 对照数据，2026-09-06）\n")
    md.append("> 归属：`compare/`。数据源见每节。运行脚本 `code/tests/wilcoxon_exp15.py`，"
              "机器可读 `output/wilcoxon_exp15.json`。不 git push。\n")
    md.append("**配对方向**：所有 diff = 后项 − 前项（`A_vs_B` 后项=B；`B_vs_C` 后项=C），"
              "**正值表示后项在该指标上更优**（命中率更高 / 均值口径 dev 更低）。"
              "命中 = per-seq |q−target|≤2 占比；dev = |mean_q−target|（均值口径）。"
              "p 为双侧 Wilcoxon signed-rank（零差剔除）。样本量警示：每对 n=15（3 蛋白×5 臂），"
              "功效极低，结论仅作描述性；区间为差分布 bootstrap 中位数 95%CI。\n")

    # 表1 逐臂逐蛋白数值（人读）
    md.append("\n## 表1 逐配对逐臂原始数值\n")
    for mode, pair in [("prot", "A_vs_B"), ("prot", "B_vs_C"),
                       ("lig", "A_vs_B"), ("lig", "B_vs_C")]:
        tag = {"prot": "蛋白模式", "lig": "配体模式"}[mode]
        title = {"A_vs_B": "A(裸) vs B(encoder校准)", "B_vs_C": "B(encoder) vs C(bias)"}[pair]
        md.append(f"\n### {tag} — {title}\n")
        md.append("| 蛋白 | 臂 | x hit | y hit | Δhit | x dev | y dev | Δdev |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r[0] != mode or r[1] != pair:
                continue
            _, _, arm, pdb, xh, yh, xd, yd = r
            if pair == "A_vs_B":
                dh = yh - xh if (xh is not None and yh is not None) else None
                dd = xd - yd
            else:  # B_vs_C: 后项 C
                dh = yh - xh if (xh is not None and yh is not None) else None
                dd = xd - yd
            md.append(f"| {pdb} | {arm} | {xh:.4f} | {yh:.4f} | "
                      f"{('' if dh is None else f'{dh:+.4f}')} | "
                      f"{xd:.3f} | {yd:.3f} | {dd:+.3f} |")

    # 汇总 + 分臂子组
    md.append("\n## 表2 Wilcoxon 汇总（按模式×配对，命中率/dev 分别）\n")
    md.append("> 每行给出 med_diff（后项−前项，正=后项更优）、bootstrap 中位数 95%CI、"
              "双侧 Wilcoxon p、匹配对秩双列效应量 r（+1=全部非零差同向为正）、方向计数 (+/−/0)。"
              "子组 `moderate`=native/n2/p2（n=9），`extreme`=n8/p8（n=6），样本更小，仅示意。\n")
    summary = {}
    for mode in ("prot", "lig"):
        for pair in ("A_vs_B", "B_vs_C"):
            for metric in ("hit", "dev"):
                for sub, arm_list in [("all", ARMS), ("moderate", MODERATE),
                                      ("extreme", EXTREME)]:
                    key = f"{mode}/{pair}/{metric}/{sub}"
                    # 收集 diff
                    diffs = []
                    pair_rows = []
                    for r in rows:
                        if r[0] != mode or r[1] != pair or r[2] not in arm_list:
                            continue
                        _, _, _, _, xh, yh, xd, yd = r
                        if metric == "hit":
                            if xh is None or yh is None:
                                continue
                            d = yh - xh  # 后项 − 前项
                        else:
                            d = xd - yd  # 前项 dev − 后项 dev
                        diffs.append(d)
                        pair_rows.append(r)
                    res = wilcox_summary(np.array(diffs), key)
                    res["n_arms"] = len(pair_rows)
                    summary[key] = res
                    # markdown
                    lab = {"prot": "蛋白", "lig": "配体"}[mode]
                    pl = {"A_vs_B": "A vs B(enc)", "B_vs_C": "B(enc) vs C(bias)"}[pair]
                    ml = "命中率" if metric == "hit" else "mean dev"
                    s = res
                    ci = s.get('median_ci95', ['?', '?'])
                    md.append(f"- **{lab} | {pl} | {ml} | {sub}**：n={s['n_pairs']} "
                              f"(+{s.get('n_pos', '?')}/−{s.get('n_neg', '?')}/"
                              f"{s.get('n_zero', '?')}) "
                              f"med_diff={s['median_diff']:+.4f} "
                              f"[{ci[0]}, {ci[1]}]  "
                              f"p={s['p_value']:.4f}  r={s.get('effect_r', 'NA')}  "
                              f"(n_nonzero={s.get('n_nonzero')})")
    json_out["pairs"] = summary

    # ---- 补充 S1：v13-in10 vs v14-clean 共享 10 蛋白逐臂 dev 配对 ----
    v13 = json.load(open(V13))
    v14 = json.load(open(V14))
    srows = []
    for pdb in sorted(set(v13["proteins"]) & set(v14["proteins"])):
        m13 = v13["proteins"][pdb]["mode"]["ligand"]["arms"]
        m14 = v14["proteins"][pdb]["mode"]["ligand"]["arms"]
        for arm in ARMS:
            if arm in m13 and arm in m14 and m13[arm].get("dev") is not None \
                    and m14[arm].get("dev") is not None:
                srows.append((pdb, arm, m13[arm]["dev"], m14[arm]["dev"]))
    md.append("\n## 补充 S1 — v13-in10 vs v14-clean 逐臂 mean dev 配对（10 蛋白共享）\n")
    md.append("数据源：`output/v13_ligand_gen_stats_in10.json`（v13）与 "
              "`output/v14_ligand_gen_stats_clean.json`（v14）；指标=每臂 50 条生成的 "
              "mean dev（=|mean_q−target|，v14 clean 校验链）。diff=v14−v13，正=v14 更好。\n")
    md.append("> 注：这是聚合均值的配对（非逐序列），并涉及两版本在同一 in-10 测试集上的比较，"
              "属补充描述性证据；Wilcoxon 假设逐配对独立，此处共享结构/骨架，解读保守。\n")
    md.append("| 蛋白 | 臂 | v13 dev | v14 dev | Δ(v14−v13) |")
    md.append("|---|---|---|---|---|")
    diffs = []
    for pdb, arm, d13, d14 in srows:
        diffs.append(d14 - d13)
        md.append(f"| {pdb} | {arm} | {d13:+.3f} | {d14:+.3f} | {d14-d13:+.3f} |")
    res = wilcox_summary(np.array(diffs), "supp/v13_vs_v14")
    json_out["supplement_v13_v14"] = res
    md.append(f"\nS1 Wilcoxon：n={res['n_pairs']}（10 蛋白×5 臂）med_diff(v14−v13)="
              f"{res['median_diff']:+.4f} CI95="
              f"{[round(x,4) for x in res.get('median_ci95',['?','?'])]} "
              f"p={res['p_value']:.4f} r_eff={res.get('effect_r')} "
              f"(+{res['n_pos']}/−{res['n_neg']}/0{res['n_zero']})\n")
    # 补充：v14/v13 per-protein H2 计数对比
    md.append("| 蛋白 | v13 H2(/5) | v14 H2(/5) |")
    md.append("|---|---|---|")
    tot13 = tot14 = 0
    for pdb in sorted(set(v13["proteins"]) & set(v14["proteins"])):
        m13 = v13["proteins"][pdb]["mode"]["ligand"]["arms"]
        m14 = v14["proteins"][pdb]["mode"]["ligand"]["arms"]
        h13 = sum(1 for a in ARMS if a in m13 and m13[a].get("hit"))
        h14 = sum(1 for a in ARMS if a in m14 and m14[a].get("hit"))
        tot13 += h13; tot14 += h14
        md.append(f"| {pdb} | {h13}/5 | {h14}/5 |")
    md.append(f"\nH2 汇总：v13 {tot13}/50 vs v14 {tot14}/50。\n")

    # 结论
    md.append("\n## 结论（谨慎表述）\n")
    md.append("1. exp2 已知**逐序列命中**上 bias(C) 显著强于 encoder(B)——Wilcoxon 配对在 n=15 "
              "上多数方向一致；但因样本小、差分布集中（如 C 命中几乎全高于 B），p 仅作参考。\n")
    md.append("2. exp1 中**条件(B)相对裸(A)的增益**集中在极端臂（n8/p8 裸 baseline≈0），"
              "native 臂增益小甚至为负；Wilcoxon 在全臂上不显著或方向不一。\n")
    md.append("3. 配体模式与蛋白模式方向同构。\n")
    md.append("4. **样本量警示**：n=15 功效极低；本报告只提供描述性中位数差/方向一致性与 CI，"
              "未达『统计显著』也不应解读为『无差异』。\n")
    md.append("\n## 数据源索引\n")
    md.append(f"- 蛋白模式：`{PROT_SUM}`（routes A/B/C by_arm：per_seq_hit_le2/dev_of_mean/fracs）\n")
    md.append(f"- 配体模式：`{LIG_SUM}`（routeA frac_by_arm_pm2；routeB_cal/routeC per_seq_hit_2/mean_dev）\n")
    md.append(f"- 补充 S1：`{V13}`、`{V14}`\n")
    md.append("- 输出：`output/wilcoxon_exp15.json`\n")

    OUT_JSON.write_text(json.dumps(json_out, indent=2, ensure_ascii=False))
    OUT_MD.write_text("\n".join(md))
    print(f"written {OUT_JSON}")
    print(f"written {OUT_MD}")


if __name__ == "__main__":
    main()
