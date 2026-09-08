"""exp_control_prot 分析：读取 output/exp_control_prot 生成聚合统计。

计算：
  - route A：逐蛋白裸 backbone 电荷分布 → native/n2/p2/n8/p8 区占比（±1 与 ±2 两种容差）
  - route B：逐臂 mean dev / mean-口径 H2(mean dev≤2) / per-seq 命中效率(|dev|≤2) / native recovery
  - route C：同上（bias）
  - 增益表：route B per-arm per-seq 命中效率 − route A 对应电荷区占比（±2 容差口径为主，±1 附注）
  - 汇总：蛋白×臂 → 均值口径 H2、per-sequence 命中效率（总）

用法：PYTHONPATH=code python code/tests/exp_prot_control/analyze_control.py [--out_dir ...]
输出：<out_dir>/_analysis_summary.json + 打印表
"""
import argparse
import json
from pathlib import Path
import numpy as np

OUT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN/output/exp_control_prot")
ARMS = ["native", "n2", "p2", "n8", "p8"]
ARM_DQ = {"native": 0, "n2": -2, "p2": +2, "n8": -8, "p8": +8}


def load_rows(p):
    return [json.loads(l) for l in open(p)]


def load_route_rows(pdb_dir, route):
    """route 数据按臂存在 route_dir/arm_<arm>/per_seq.jsonl；合并读入。"""
    route_dir = pdb_dir / f"route_{route}"
    if (route_dir / "per_seq.jsonl").is_file():   # route A 单池文件
        return load_rows(route_dir / "per_seq.jsonl")
    rows = []
    for arm_dir in sorted(route_dir.glob("arm_*/per_seq.jsonl")):
        rows += load_rows(arm_dir)
    return rows


def route_a_zone_fractions(rows, t_native, q_native):
    """route A（bare）落在 5 个电荷区的占比。tol=1 或 2。

    定心有两种口径：
      - q_native 口径（任务字面）：nominal = q_native + dq，native dq=0
      - round 口径（对齐 route B target）：nominal = t_native + dq
    """
    qs = np.array([r["charge"] for r in rows])
    fracs = {}
    for arm in ARMS:
        dq = ARM_DQ[arm]
        for tag, nominal in (("qnat", q_native + dq), ("round", t_native + dq)):
            for tol in (1, 2):
                fracs[f"{arm}_{tag}_tol{tol}"] = float(
                    (np.abs(qs - nominal) <= tol).mean())
    return {"n": len(qs), "mean_charge": float(qs.mean()),
            "std_charge": float(qs.std()),
            "mean_recovery": float(np.mean([r["recovery"] for r in rows])),
            "fracs": fracs}


def route_bc_summary(rows, t_native):
    qs = np.array([r["charge"] for r in rows])
    by_arm = {}
    n_all = 0
    hit_le2_all = 0
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        if not rr:
            continue
        tgt = t_native + ARM_DQ[arm]
        q = np.array([r["charge"] for r in rr])
        dev = np.abs(q - tgt)
        hit = float((dev <= 2.0).mean())
        dev_of_mean = float(abs(q.mean() - tgt))  # 均值口径 dev（既有 H2 约定）
        n_all += len(rr)
        hit_le2_all += int((dev <= 2.0).sum())
        by_arm[arm] = {
            "target": tgt, "n": len(rr),
            "mean_charge": round(float(q.mean()), 3),
            "std_charge": round(float(q.std()), 3),
            "dev_of_mean": round(dev_of_mean, 3),   # 均值口径 dev
            "mean_abs_dev": round(float(dev.mean()), 3),  # 逐序列平均绝对偏离（弥散）
            "arm_H2_hit": bool(dev_of_mean <= 2.0),
            "per_seq_hit_le2": round(hit, 4),
            "mean_recovery": round(float(np.mean([r["recovery"] for r in rr])), 4),
        }
    # 均值口径 H2 与 per-seq 总命中效率（按全部有效臂/序列）
    return {"by_arm": by_arm,
            "n_arms_hit": sum(1 for a in by_arm.values() if a["arm_H2_hit"]),
            "n_arms": len(by_arm),
            "per_seq_total_hit": round(hit_le2_all / n_all, 4) if n_all else None,
            "n_seq": n_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT))
    args = ap.parse_args()
    OUTp = Path(args.out_dir)
    # 发现蛋白：扫描含 meta.json 的子目录（不依赖顶层 meta 的 proteins 列表，
    # 因为最后一次采样运行只会写它的 --proteins 子集）
    proteins = sorted(d.name for d in OUTp.iterdir()
                      if d.is_dir() and (d / "meta.json").is_file())
    meta = {}
    mf = OUTp / "meta.json"
    if mf.is_file():
        meta = json.load(open(mf))
    summary = {"meta": {"pH": meta.get("pH"), "temperature": meta.get("temperature"),
                        "bias_strength": meta.get("bias_strength"),
                        "backbone": meta.get("backbone"),
                        "cond_encoder": meta.get("cond_encoder")},
               "proteins": {}}
    for pdb in proteins:
        m = json.load(open(OUTp /pdb / "meta.json"))
        tn = m["round_target"]
        psum = {"pdb": pdb, "L": m["L"], "native_charge": m["native_charge"],
                "round_target": tn, "routes": {}}
        # route A
        rA = load_rows(OUTp /pdb / "route_A" / "per_seq.jsonl")
        psum["routes"]["A"] = route_a_zone_fractions(rA, tn, m["native_charge"])
        # route B
        rB = load_route_rows(OUTp /pdb, "B")
        psum["routes"]["B"] = route_bc_summary(rB, tn)
        # route Bsmall (若存在)
        bs_p = OUTp /pdb / "route_Bsmall"
        if any(bs_p.glob("arm_*/per_seq.jsonl")):
            rBs = load_route_rows(OUTp /pdb, "Bsmall")
            psum["routes"]["Bsmall"] = route_bc_summary(rBs, tn)
            sc = json.load(open(bs_p / f"{pdb}_smallcal.json"))
            psum["routes"]["Bsmall"]["smallcal"] = {
                "slope": sc["slope"], "intercept": sc["intercept"],
                "loocv": sc["loocv"], "unreliable": sc["unreliable"]}
        # route C
        rC = load_route_rows(OUTp /pdb, "C")
        psum["routes"]["C"] = route_bc_summary(rC, tn)
        summary["proteins"][pdb] = psum

    # ---- 打印 ----
    for pdb in proteins:
        p = summary["proteins"][pdb]
        print(f"\n=== {pdb} L={p['L']} native={p['native_charge']:+.2f} round={p['round_target']}")
        fa = p["routes"]["A"]["fracs"]
        print(" routeA zone frac (qnat 定心): " + " ".join(
            f"{arm}:±1={fa[f'{arm}_qnat_tol1']:.4f}/±2={fa[f'{arm}_qnat_tol2']:.4f}"
            for arm in ARMS))
        print(" routeA zone frac (round 定心): " + " ".join(
            f"{arm}:±1={fa[f'{arm}_round_tol1']:.4f}/±2={fa[f'{arm}_round_tol2']:.4f}"
            for arm in ARMS))
        for route in ("B", "Bsmall", "C"):
            if route not in p["routes"] or not p["routes"][route].get("by_arm"):
                continue
            bc = p["routes"][route]
            psh = bc["per_seq_total_hit"]
            print(f" route{route}: mean口径H2={bc['n_arms_hit']}/{bc['n_arms']}  "
                  f"per-seq总命中={'NA' if psh is None else f'{psh:.3f}'}")
            for arm in ARMS:
                if arm in bc["by_arm"]:
                    a = bc["by_arm"][arm]
                    print(f"    {arm:6s} target={a['target']:+d} mean={a['mean_charge']:+6.2f} "
                          f"std={a['std_charge']:4.2f} dev_of_mean={a['dev_of_mean']:4.2f} "
                          f"(mean_abs_dev={a['mean_abs_dev']:4.2f}) "
                          f"seqhit≤2={a['per_seq_hit_le2']:.3f} rec={a['mean_recovery']:.3f}")
        if "Bsmall" in p["routes"]:
            print("   Bsmall calib:", p["routes"]["Bsmall"]["smallcal"])

    # 汇总表：增益（B per-seq 命中 − A ±2 占比）
    print("\n=== 增益表 (B per-seq hit ≤2  − A 对应区 ±2 占比) ===")
    for pdb in proteins:
        p = summary["proteins"][pdb]
        row = [f"{pdb}"]
        for arm in ARMS:
            if arm in p["routes"]["B"]["by_arm"]:
                Bhit = p["routes"]["B"]["by_arm"][arm]["per_seq_hit_le2"]
            else:
                Bhit = float("nan")
            Af = p["routes"]["A"]["fracs"][f"{arm}_round_tol2"]
            row.append(f"{arm}:{Bhit:.3f}-{Af:.4f}={Bhit-Af:+.3f}")
        print("  " + "  ".join(row))

    out = OUTp /"_analysis_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n已写 {out}")


if __name__ == "__main__":
    main()
