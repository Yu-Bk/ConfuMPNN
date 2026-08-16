"""Phase 3 防失控扩样本（n=20）配对统计检验。

输入：code/output/phase3_antidrift_n20/{pdb}/{A_base,A_cond,B_base,B_cond}/
    每臂含 seqs.fa + 四指标打分 CSV（plddt/tm/protein_sol/temberture）。
配对：按 fasta header 里的 seed 号，把「基线（无注入）」与「条件（注入）」
    同一 seed 的序列配对——同一 randn → 同一解码顺序 → 唯一差异=条件。

输出：phase3_antidrift_n20_stats.csv + 打印表格 + 判定。

三类检查：
1. **防失控主检验**：A/B 两场景 × 4 指标 × 4 PDB = 32 组配对检验（t + Wilcoxon），
   BH-FDR 校正。FDR-p>0.05 且 |dz|<0.5 → 无实质差异（PASS）。
2. **基线复现性**：A_base（n=20 无注入）均值 vs E1b 基线（n=6）——证明新采样
   与历史基线一致（采样协议可复现）。
3. **条件有效性**：B 场景（pH4.0/target=+5）条件臂电荷应显著高于基线——
   证明「副作用是在条件真正工作时测的」，不是平凡检验。
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

PDBS = ["1BC8", "1CRN", "1UBQ", "2LZM"]
ARMS = {"A_base", "A_cond", "B_base", "B_cond"}
SCENARIOS = {"A": ("A_base", "A_cond"), "B": ("B_base", "B_cond")}

# 指标 → (CSV 文件名, 值列名, name 列名)
METRICS = {
    "plddt":      ("plddt.csv",                 "mean_plddt",   "name"),
    "tm":         ("tm.csv",                    "tm_score",     "name"),
    "sol":        ("seqs.fa-protein_sol.csv",   "percent-sol",  "ID"),
    "temberture": ("seqs.fa.tm.csv",            "mean_tm",      "name"),
}

# E1b MoMPNN 基线（n=6，来自 e1_ext/summary_cond.csv mompnn,baseline 行）
E1B_BASELINE = {
    "1BC8": dict(plddt=82.824, tm=0.915, sol=81.424, temberture=64.818),
    "1CRN": dict(plddt=89.140, tm=0.905, sol=84.106, temberture=59.114),
    "1UBQ": dict(plddt=89.356, tm=0.962, sol=85.987, temberture=68.612),
    "2LZM": dict(plddt=88.466, tm=0.971, sol=80.145, temberture=67.708),
}

SEED_RE = re.compile(r"^>seed_(\d+)")


def read_fasta_seeds(fa):
    """按行序返回 seed 列表（header 里 seed_N）。"""
    seeds = []
    with open(fa, encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                m = SEED_RE.match(line.strip())
                seeds.append(int(m.group(1)) if m else None)
    return seeds


def read_metric_seedmap(arm_dir, metric):
    """返回 {seed: value}。value 从 CSV 按 name 列取；seed 顺序取 fasta。"""
    fa = arm_dir / "seqs.fa"
    seeds = read_fasta_seeds(fa)
    fname, col, name_col = METRICS[metric]
    csv_path = arm_dir / fname
    if not csv_path.exists():
        return {}
    val_by_name = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            val_by_name[row[name_col]] = row[col]
    # name 可能被打分脚本清洗（空格→_），做宽松匹配：seed 号已在 fasta 侧取，
    # 这里按行序对齐（打分工具通常保序）；长度校验兜底
    vals = {}
    rows = list(csv.DictReader(open(csv_path)))
    if len(rows) != len(seeds):
        print(f"  ⚠️ 长度不匹配 {arm_dir.name}/{metric}: csv={len(rows)} fa={len(seeds)}")
        return {}
    for seed, row in zip(seeds, rows):
        try:
            vals[seed] = float(row[col])
        except (ValueError, KeyError):
            pass
    return vals


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR 校正。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.empty_like(p)
    ranked[order] = np.arange(1, n + 1)
    q = p * n / ranked
    # 单调约束
    q_sorted = np.sort(q[order])
    q = np.empty_like(q_sorted)
    running = 1.0
    for i in range(n - 1, -1, -1):
        running = min(running, q_sorted[i])
        q[order[i]] = running
    return q


def cohens_dz(diff):
    """配对 Cohen's dz = mean(diff)/std(diff)。"""
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def load_arm(arm_dir):
    """返回 {metric: {seed: value}}。"""
    return {m: read_metric_seedmap(arm_dir, m) for m in METRICS}


def paired_test(base, cond):
    """base/cond: {seed: value}，返回 (diff 数组, t_p, wilcoxon_p)。"""
    seeds = sorted(set(base) & set(cond))
    if len(seeds) < 3:
        return None
    b = np.array([base[s] for s in seeds], dtype=float)
    c = np.array([cond[s] for s in seeds], dtype=float)
    diff = c - b
    t_p = stats.ttest_rel(c, b).pvalue
    try:
        w_p = stats.wilcoxon(c, b).pvalue
    except ValueError:
        w_p = np.nan
    return diff, t_p, w_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="code/output/phase3_antidrift_n20")
    ap.add_argument("--out", default="code/output/phase3_antidrift_n20/stats.csv")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    rows = []  # 主检验行
    all_p = []  # 收集所有检验 p（FDR 用）
    recs = []  # 临时记录

    for pdb in PDBS:
        pdb_dir = in_dir / pdb
        if not pdb_dir.exists():
            print(f"⚠️ 缺 {pdb}，跳过")
            continue
        print(f"\n===== {pdb} =====")
        for scen, (b_arm, c_arm) in SCENARIOS.items():
            base = load_arm(pdb_dir / b_arm)
            cond = load_arm(pdb_dir / c_arm)
            for metric in METRICS:
                res = paired_test(base[metric], cond[metric])
                if res is None:
                    continue
                diff, t_p, w_p = res
                p = min(t_p, w_p) if not np.isnan(w_p) else t_p
                all_p.append(p)
                recs.append(dict(pdb=pdb, scen=scen, metric=metric,
                                 n=len(diff), diff_mean=float(np.mean(diff)),
                                 diff_std=float(np.std(diff, ddof=1)),
                                 dz=cohens_dz(diff), t_p=t_p, w_p=w_p,
                                 base_mean=float(np.mean(list(base[metric].values()))),
                                 cond_mean=float(np.mean(list(cond[metric].values())))))

    # FDR 校正
    qvals = bh_fdr([r["t_p"] for r in recs]) if recs else []
    for r, q in zip(recs, qvals):
        r["fdr_p"] = float(q)
        r["pass"] = (r["fdr_p"] > args.alpha) and (abs(r["dz"]) < 0.5)
        rows.append(r)

    # ---- 输出主表 ----
    print("\n===== 防失控主检验（配对，FDR 校正）=====")
    print(f"{'PDB':<6}{'场景':<4}{'指标':<11}{'Δmean':>8}{'dz':>7}"
          f"{'t_p':>10}{'FDR-p':>10}  判定")
    for r in rows:
        verdict = "PASS" if r["pass"] else ("⚠️差异" if r["fdr_p"] <= args.alpha else "⚠️效应大")
        print(f"{r['pdb']:<6}{r['scen']:<4}{r['metric']:<11}"
              f"{r['diff_mean']:>+8.3f}{r['dz']:>7.2f}"
              f"{r['t_p']:>10.4f}{r['fdr_p']:>10.4f}  {verdict}")

    # ---- 基线复现性 ----
    print("\n===== 基线复现性（A_base n=20 vs E1b n=6）=====")
    rep_rows = []
    for pdb in PDBS:
        pdb_dir = in_dir / pdb
        base = load_arm(pdb_dir / "A_base")
        e1 = E1B_BASELINE[pdb]
        line = f"{pdb:<6}"
        for metric, e1v in e1.items():
            arr = np.array(list(base[metric].values()))
            if len(arr):
                line += f"  {metric} {np.mean(arr):.2f} vs {e1v:.2f}"
        print(line)

    # ---- 条件有效性（B 场景电荷）----
    print("\n===== 条件有效性（B 场景：pH4.0/target=+5，条件臂电荷应>基线）=====")
    for pdb in PDBS:
        pdb_dir = in_dir / pdb
        for arm in ("B_base", "B_cond"):
            fa = pdb_dir / arm / "seqs.fa"
            charges = []
            with open(fa, encoding="utf-8") as f:
                for line in f:
                    if line.startswith(">"):
                        m = re.search(r"charge=([-+]?\d+\.?\d*)", line)
                        if m:
                            charges.append(float(m.group(1)))
            if charges:
                print(f"  {pdb}/{arm:<7} mean charge = {np.mean(charges):+6.2f}"
                      f" ± {np.std(charges, ddof=1):.2f}  (n={len(charges)})")

    # ---- 写 CSV ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "pdb", "scen", "metric", "n", "base_mean", "cond_mean",
            "diff_mean", "diff_std", "dz", "t_p", "w_p", "fdr_p", "pass"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写 {out}")

    n_pass = sum(1 for r in rows if r["pass"])
    print(f"\n=== 汇总：{len(rows)} 组检验，PASS {n_pass}，"
          f"非 PASS {len(rows)-n_pass} ===")


if __name__ == "__main__":
    main()
