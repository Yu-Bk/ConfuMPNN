"""exp1/exp2（配体模式）受控对照：汇总 output/exp_control_lig/ → summary.json。

对每个测试蛋白计算：
  route A（裸无条件）：
      * n, mean/std charge
      * 各电荷窗占比（native±1；native+dq±1 与 ±2，dq=-8/-2/+2/+8）= 天然基线命中率
      * |q - q_native|<=1 占比
  route B（条件编码器直接生成，5 臂 native/n2/p2/n8/p8）：
      * per-arm n, mean_q, mean dev, per-seq |dev|<=2 命中率（命中效率）
      * arm 均值口径 dev<=2 → H2 arm 达标
      * native identity recovery、带电残基总数倍率（组成）
  route B_cal（若存在）：同上（校准附加组）
  route C（bias-only，native/n2/p2）：同上
  gain:
      * B_direct_hit  − A(±1)   （exp1 主增益，spec 口径）
      * B_cal_hit − A(±1)
      * C_hit − A(±1)（exp2 参考）

输出：
  output/exp_control_lig/summary.json  （机器可读，作图引用数据路径）
  output/exp_control_lig/summary_tables.md（人读）
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "LigandMPNN"))

CHARGED = ("D", "E", "K", "R")
POS = ("K", "R")
NEG = ("D", "E")
ARMS_B = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]
# exp2 规范臂 native/n2/p2；扩展 n8/p8 一并纳入分析（报告区分标注）
ARMS_C = [("native", 0), ("n2", -2), ("p2", +2), ("n8", -8), ("p8", +8)]


def charged_counts(seq):
    return sum(1 for a in seq if a in CHARGED)


def comp_pos_neg(seq):
    """返回 (K+R 正电残基数, D+E 负电残基数, 总带电残基数)。"""
    pos = sum(1 for a in seq if a in POS)
    neg = sum(1 for a in seq if a in NEG)
    return pos, neg, pos + neg


def frac_in(charges, lo, hi):
    if len(charges) == 0:
        return float("nan")
    return float(np.mean([1.0 if (lo <= q <= hi) else 0.0 for q in charges]))


def analyze_routeA(adir, native_comp=None):
    d = json.load(open(adir / "sequences.json"))
    q = np.array(d["charges"])
    rec = np.array(d["recs"])
    comp = np.array([comp_pos_neg(s) for s in d["seqs"]])  # [n,3] pos,neg,total
    out = {"n": len(q), "mean_q": float(q.mean()), "std_q": float(q.std()),
           "mean_rec": float(rec.mean()),
           "mean_pos": float(comp[:, 0].mean()), "mean_neg": float(comp[:, 1].mean()),
           "mean_charged": float(comp[:, 2].mean())}
    if native_comp:
        out["pos_ratio"] = round(float(comp[:, 0].mean() / native_comp[0]), 3) if native_comp[0] else None
        out["neg_ratio"] = round(float(comp[:, 1].mean() / native_comp[1]), 3) if native_comp[1] else None
        out["charged_ratio"] = round(float(comp[:, 2].mean() / native_comp[2]), 3) if native_comp[2] else None
    return out


def analyze_arm(adir, target, native_comp):
    d = json.load(open(adir / "sequences.json"))
    q = np.array(d["charges"], dtype=float)
    devs = np.array(d["devs"], dtype=float) if "devs" in d else q - target
    recs = np.array(d["recs"], dtype=float)
    per_seq_hit = float(np.mean(np.abs(devs) <= 2.0))     # per-seq |dev|<=2
    mean_dev = float(q.mean() - target)                   # signed mean dev
    hit_mean = bool(abs(mean_dev) <= 2.0)                 # arm 均值口径达标(H2)
    # 组成：带电残基总数/正电(K+R)/负电(D+E) 倍率（相对 native）
    comp = np.array([comp_pos_neg(s) for s in d["seqs"]])  # [n,3]
    n0, n1, n2 = native_comp
    out = {"n": len(q), "target": target, "mean_q": float(q.mean()),
           "std_q": float(q.std()), "mean_dev": round(mean_dev, 3),
           "per_seq_hit_2": round(per_seq_hit, 4),
           "arm_mean_hit_2": hit_mean,
           "mean_rec": float(recs.mean()),
           "mean_pos": float(comp[:, 0].mean()), "mean_neg": float(comp[:, 1].mean())}
    out["pos_ratio"] = round(float(comp[:, 0].mean() / n0), 3) if n0 else None
    out["neg_ratio"] = round(float(comp[:, 1].mean() / n1), 3) if n1 else None
    out["charged_ratio"] = round(float(comp[:, 2].mean() / n2), 3) if n2 else None
    out["file"] = str((adir / "sequences.json").resolve())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default=str(ROOT / "output" / "exp_control_lig"))
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    summary = {"root_data": str(in_dir.resolve()), "proteins": []}

    # meta 提供 native 信息 fallback
    meta_path = in_dir / "meta.json"
    if meta_path.exists():
        summary["meta"] = json.load(open(meta_path))
    prots = sorted([p for p in in_dir.iterdir() if (p / "native.json").exists()])

    for pdir in prots:
        pdb = pdir.name
        native_info = json.load(open(pdir / "native.json"))
        native_seq = native_info["native"]
        q_nat = native_info["native_charge"]
        native_comp = comp_pos_neg(native_seq)
        tgt_int = int(round(q_nat))
        arm_target = {tag: tgt_int + dq for tag, dq in ARMS_B}

        psum = {"pdb": pdb, "rank": native_info.get("rank"),
                "native_charge": q_nat, "round_target_int": tgt_int, "arms": {}}

        # ---- route A ----
        if (pdir / "routeA" / "sequences.json").exists():
            a = analyze_routeA(pdir / "routeA", native_comp=native_comp)
            qA = np.array(json.load(open(pdir / "routeA" / "sequences.json"))["charges"])
            a["frac_native_pm1"] = round(frac_in(qA, q_nat - 1, q_nat + 1), 4)
            a["frac_by_arm_pm1"] = {tag: round(frac_in(qA, t - 1, t + 1), 4)
                                    for tag, t in arm_target.items()}
            a["frac_by_arm_pm2"] = {tag: round(frac_in(qA, t - 2, t + 2), 4)
                                    for tag, t in arm_target.items()}
            a["file"] = str((pdir / "routeA" / "sequences.json").resolve())
            psum["routeA"] = a
            qA_lo = float(np.percentile(qA, 5)); qA_hi = float(np.percentile(qA, 95))
            psum["routeA"]["q_p5_p95"] = [round(qA_lo, 2), round(qA_hi, 2)]
        else:
            psum["routeA"] = None

        # ---- route B ----
        psum["routeB"] = {}
        for tag, dq in ARMS_B:
            adir = pdir / "routeB" / f"arm_{tag}"
            if (adir / "sequences.json").exists():
                psum["routeB"][tag] = analyze_arm(adir, arm_target[tag], native_comp)
        # ---- route B_cal ----
        psum["routeB_cal"] = {}
        for tag, dq in ARMS_B:
            adir = pdir / "routeB_cal" / f"arm_{tag}"
            if (adir / "sequences.json").exists():
                psum["routeB_cal"][tag] = analyze_arm(adir, arm_target[tag], native_comp)
        # ---- route C ----
        psum["routeC"] = {}
        for tag, dq in ARMS_C:
            adir = pdir / "routeC" / f"arm_{tag}"
            if (adir / "sequences.json").exists():
                psum["routeC"][tag] = analyze_arm(adir, arm_target[tag], native_comp)

        # ---- gains ----
        gains = {}
        A = psum["routeA"] or {}
        for rname in ("routeB", "routeB_cal", "routeC"):
            for tag in psum.get(rname, {}):
                bhit = psum[rname][tag].get("per_seq_hit_2")
                af1 = (A.get("frac_by_arm_pm1") or {}).get(tag)
                af2 = (A.get("frac_by_arm_pm2") or {}).get(tag)
                if bhit is not None:
                    gains.setdefault(rname, {})[tag] = {
                        "B_hit_minus_A_pm1": round(bhit - af1, 4) if af1 is not None else None,
                        "B_hit_minus_A_pm2": round(bhit - af2, 4) if af2 is not None else None,
                        "A_pm1": af1, "A_pm2": af2}
        psum["gains"] = gains
        summary["proteins"].append(psum)

    # ---- 全局聚合（exp1 主增益 = B_direct − A ±1，均值跨蛋白/臂）----
    agg = {"arm_levels": {}}
    alltags = [t for t, _ in ARMS_B]
    for rname in ("routeB", "routeB_cal", "routeC"):
        tags = [t for t, _ in ARMS_B] if rname != "routeC" else [t for t, _ in ARMS_C]
        for tag in tags:
            hits = [p[rname][tag]["per_seq_hit_2"] for p in summary["proteins"]
                    if p.get(rname) and tag in p[rname]]
            afs1 = [p["routeA"]["frac_by_arm_pm1"][tag] for p in summary["proteins"]
                    if p.get("routeA") and p["routeA"].get("frac_by_arm_pm1")]
            afs2 = [p["routeA"]["frac_by_arm_pm2"][tag] for p in summary["proteins"]
                    if p.get("routeA") and p["routeA"].get("frac_by_arm_pm2")]
            agg["arm_levels"][f"{rname}:{tag}"] = {
                "mean_B_hit": round(float(np.mean(hits)), 4) if hits else None,
                "mean_A_pm1": round(float(np.mean(afs1)), 4) if afs1 else None,
                "mean_A_pm2": round(float(np.mean(afs2)), 4) if afs2 else None,
                "gain_B_minus_A_pm1": round(float(np.mean(hits)) - float(np.mean(afs1)), 4)
                                      if hits and afs1 else None,
                "gain_B_minus_A_pm2": round(float(np.mean(hits)) - float(np.mean(afs2)), 4)
                                      if hits and afs2 else None,
                "n_prot": len(hits)}
    summary["aggregate"] = agg

    with open(in_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"wrote {in_dir / 'summary.json'}")
    # 打印紧凑表
    for p in summary["proteins"]:
        print(f"\n== {p['pdb']} native_q={p['native_charge']:+.2f} ==")
        A = p.get("routeA")
        if A:
            print(f"  A: mean={A['mean_q']:+.2f} std={A['std_q']:.2f} "
                  f"rec={A['mean_rec']:.3f} native±1={A['frac_native_pm1']:.3f} "
                  f"p5-p95=[{A['q_p5_p95'][0]},{A['q_p5_p95'][1]}]")
        for rname in ("routeB", "routeB_cal", "routeC"):
            if not p.get(rname):
                continue
            parts = []
            for tag in sorted(p[rname]):
                a = p[rname][tag]
                parts.append(f"{tag}:Δ{a['mean_dev']:+.1f} "
                             f"hit{a['per_seq_hit_2']:.2f} "
                             f"rec{a['mean_rec']:.2f} cr{a['charged_ratio']:.2f}")
            print(f"  {rname}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
