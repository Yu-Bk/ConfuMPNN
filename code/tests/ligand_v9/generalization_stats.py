"""v9 泛化验证汇总统计：合并电荷/H1 折叠/pLDDT/recovery，ligand vs protein 消融对比。

输入（validate_generalization.py 采样 + esmfold_score.py 回折 + tm_score.py 打表生成）：
  {out_dir}/{mode}/{pdb}/validation.json      电荷/recovery/口袋/GRAVY
  {out_dir}/{mode}/{pdb}/pH7.4/arm{tag}/tm.csv    TM-score（US-align vs ref）
  {out_dir}/{mode}/{pdb}/pH7.4/arm{tag}/plddt.csv ESMFold pLDDT

输出：--out json + 汇总表（含 H2 达标判定、H1 TM 判定、ligand-protein 消融对比）。

用法（code/ 下，任意环境，纯 python）：
  python code/tests/ligand_v9/generalization_stats.py \
      --root output/generalization_v9 --manifest data/validation_pdbs/validation_manifest.json \
      --out output/generalization_v9_stats.json
"""
import argparse
import csv
import json
from pathlib import Path


def read_csv_col(path, col, cast=float, exclude_native=False):
    vals = []
    if not path.exists():
        return vals
    with open(path) as f:
        for row in csv.DictReader(f):
            if exclude_native and row.get("name", "").startswith("native"):
                continue
            if col in row:
                try:
                    vals.append(cast(row[col]))
                except (ValueError, TypeError):
                    pass
    return vals


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def frac_ge(xs, t):
    return round(sum(1 for x in xs if x >= t) / len(xs), 3) if xs else None


def frac_lt(xs, t):
    return round(sum(1 for x in xs if x < t) / len(xs), 3) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/generalization_v9")
    ap.add_argument("--manifest", default="data/validation_pdbs/validation_manifest.json")
    ap.add_argument("--out", default="output/generalization_v9_stats.json")
    args = ap.parse_args()

    root = Path(args.root)
    manifest = json.load(open(args.manifest))
    cat_map = {it["pdb"]: it.get("cat", "?") for it in manifest["items"]}

    summary = {"proteins": {}}
    hdr = (f"{'蛋白':<7}{'类':<10}{'L':>4} {'模式':<7}{'臂':<6}{'目标':>4}{'均值':>7}"
           f"{'dev':>6}{'达标':>4} {'rec':>6}{'pkt':>6}{'gravy':>7}"
           f"{'TM中位':>7}{'TM≥0.7':>7}{'TM<0.5':>7}{'pLDDT':>7}")
    print(hdr, flush=True)

    for mode in ("ligand", "protein"):
        mode_dir = root / mode
        if not mode_dir.exists():
            continue
        for vj in sorted(mode_dir.glob("*/validation.json")):
            pdb = vj.parent.name
            d = json.load(open(vj))
            cat = cat_map.get(pdb, d.get("cat", "?"))
            summary["proteins"].setdefault(pdb, {
                "cat": cat, "L": d["L"], "mode": {}, "native_charge": d["native_charge"]})
            summary["proteins"][pdb]["mode"][mode] = {"arms": {}}
            for arm, a in d["arms"].items():
                arm_dir = vj.parent / f"pH7.4" / f"arm_{arm}"
                # 只统计生成序列，排除 native 行（esmfold name 含 'native'）
                tm_gen = read_csv_col(arm_dir / "tm.csv", "tm_score",
                                      exclude_native=True)
                plddt_gen = read_csv_col(arm_dir / "plddt.csv", "mean_plddt",
                                         exclude_native=True)
                tm_med = median(tm_gen)
                rec = summary["proteins"][pdb]["mode"][mode]["arms"][arm] = {
                    "target": a["target"],
                    "mean_charge": a["mean_charge"],
                    "std_charge": a["std_charge"],
                    "dev": a["dev"],
                    "hit": a["dev"] <= 2.0,
                    "recovery": a["recovery"],
                    "pocket_recovery": a["pocket_recovery"],
                    "gravy_mean": a["gravy_mean"],
                    "tm_median": round(tm_med, 3) if tm_med else None,
                    "tm_ge070": frac_ge(tm_gen, 0.7),
                    "tm_lt050": frac_lt(tm_gen, 0.5),
                    "plddt_median": round(median(plddt_gen), 1) if plddt_gen else None,
                }
                hit = "✓" if a["dev"] <= 2.0 else "✗"
                print(f"{pdb:<7}{cat:<10}{d['L']:>4} {mode:<7}{arm:<6}{a['target']:>4}"
                      f"{a['mean_charge']:>7}{a['dev']:>6}{hit:>4}"
                      f"{a['recovery']:>6}{str(a['pocket_recovery']):>6}{a['gravy_mean']:>7}"
                      f"{str(rec['tm_median']):>7}{str(rec['tm_ge070']):>7}"
                      f"{str(rec['tm_lt050']):>7}{str(rec['plddt_median']):>7}", flush=True)

    # 消融对比汇总：ligand vs protein（同蛋白同臂）
    print("\n=== 配体消融对比（ligand vs protein，H2 dev / H1 TM）===", flush=True)
    ablation = {}
    for pdb, p in summary["proteins"].items():
        if "ligand" not in p["mode"] or "protein" not in p["mode"]:
            continue
        ablation[pdb] = {}
        for arm in p["mode"]["protein"]["arms"]:
            if arm in p["mode"]["ligand"]["arms"]:
                l = p["mode"]["ligand"]["arms"][arm]
                pr = p["mode"]["protein"]["arms"][arm]
                ablation[pdb][arm] = {
                    "ligand_dev": l["dev"], "protein_dev": pr["dev"],
                    "dev_delta": round(pr["dev"] - l["dev"], 2),
                    "ligand_tm": l["tm_median"], "protein_tm": pr["tm_median"],
                    "ligand_recovery": l["recovery"], "protein_recovery": pr["recovery"],
                }
                print(f"{pdb:<7} arm={arm:<6} dev: ligand={l['dev']:.2f} protein={pr['dev']:.2f} "
                      f"(Δ{pr['dev']-l['dev']:+.2f})  TM: {l['tm_median']}→{pr['tm_median']}  "
                      f"rec: {l['recovery']}→{pr['recovery']}", flush=True)
    summary["ablation"] = ablation

    # 总命中统计
    tot_hit = tot = 0
    tot_h1 = tot_h1n = 0
    for pdb, p in summary["proteins"].items():
        for mode, m in p["mode"].items():
            for arm, a in m["arms"].items():
                tot += 1
                tot_hit += a["hit"]
                if a["tm_median"] is not None:
                    tot_h1n += 1
                    tot_h1 += (a["tm_median"] >= 0.70)
    summary["totals"] = {
        "arms_tested": tot, "h2_hit": tot_hit,
        "h2_hit_rate": round(tot_hit / tot, 3) if tot else None,
        "h1_tm_arms": tot_h1n, "h1_tm_pass": tot_h1,
        "h1_pass_rate": round(tot_h1 / tot_h1n, 3) if tot_h1n else None,
    }
    print(f"\nH2 电荷达标: {tot_hit}/{tot} = {summary['totals']['h2_hit_rate']}")
    print(f"H1 TM≥0.70: {tot_h1}/{tot_h1n} = {summary['totals']['h1_pass_rate']}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n已写 {args.out}")


if __name__ == "__main__":
    main()
