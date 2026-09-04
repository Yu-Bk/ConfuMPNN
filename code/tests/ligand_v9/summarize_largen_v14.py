"""汇总 largen_v14 大样本三达标搜索结果 → 报告表数据。

读 output/largen_v14/<pdb>_arm_<arm>/stats.json + <pdb>_summary.json，
生成：
  - 每 (蛋白,臂) 三达标存在性 / 比例 / 前缀存在率 (n=10/25/50/100/200)
  - 主因分解（charge/deletion/h3）
  - Pareto 前沿示例（从 stats.json pareto_examples 提取）
输出 summary_largen_v14.json + 可读 .md 片段。

用法（项目根）：
  PYTHONPATH=code python code/tests/ligand_v9/summarize_largen_v14.py \
    --root output/largen_v14 \
    --manifest data/validation_pdbs/validation_manifest_v14_in.json \
    --out output/largen_v14_summary.json
"""
import argparse
import json
from pathlib import Path

ARMS = ["native", "n2", "p2", "n8", "p8"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/largen_v14")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="output/largen_v14_summary.json")
    args = ap.parse_args()

    root = Path(args.root)
    man = json.load(open(args.manifest))
    items = man["items"]

    all_exist = []
    rows = []
    per_arm_exist = {a: [] for a in ARMS}
    for it in items:
        p = it["pdb"]
        psum_path = root / f"{p}_summary.json"
        if not psum_path.exists():
            print(f"!! {p} summary 缺失", flush=True)
            continue
        ps = json.load(open(psum_path))
        for arm in ARMS:
            st = (ps.get("arms") or {}).get(arm)
            if not st:
                print(f"!! {p} arm_{arm} 缺失", flush=True)
                continue
            triple = st.get("triple_pass", 0)
            n = st.get("n", 0)
            exist = triple > 0
            prefix = {k: v for k, v in (st.get("prefix") or {}).items() if v is not None}
            rows.append({
                "pdb": p, "cat": it.get("cat"), "L": ps.get("L"), "arm": arm,
                "target": st.get("target"), "n": n,
                "triple_pass": triple,
                "triple_rate": round(triple / n, 4) if n else None,
                "exists": exist,
                "prefix": prefix,
                "main_cause_if_none": st.get("main_cause_if_none"),
                "pass": {"charge": st.get("charge", {}).get("pass"),
                         "deletion": st.get("deletion", {}).get("pass"),
                         "h3": st.get("h3", {}).get("pass")},
                "del_mean": st.get("deletion", {}).get("ratio_mean"),
                "dev_mean": st.get("charge", {}).get("dev_mean"),
                "native_dk": ps.get("native_dk"),
                "native_charge": ps.get("native_charge"),
                "native_h3_rate": ps.get("native_h3_rate"),
            })
            all_exist.append(exist)
            per_arm_exist[arm].append(exist)

    # 汇总
    total = len(rows)
    n_exist_arm = sum(all_exist)
    # 按蛋白看是否存在至少一个 arm
    by_prot = {}
    for r in rows:
        by_prot.setdefault(r["pdb"], []).append(r)
    n_prot_any = sum(any(r["exists"] for r in rr) for rr in by_prot.values())

    main_cause_counts = {}
    for r in rows:
        if not r["exists"] and r.get("main_cause_if_none"):
            for c in r["main_cause_if_none"]:
                main_cause_counts[c] = main_cause_counts.get(c, 0) + 1

    summary = {
        "total_arms": total,
        "arms_with_triple": n_exist_arm,
        "arms_with_triple_rate": round(n_exist_arm / total, 4) if total else None,
        "proteins_with_any_triple": n_prot_any,
        "per_arm_exist_rate": {a: (round(sum(per_arm_exist[a]) / len(per_arm_exist[a]), 4)
                                    if per_arm_exist[a] else None)
                               for a in ARMS},
        "rows": rows,
        "main_cause_among_no_triple_arms": main_cause_counts,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 控制台表
    print(f"共 {total} 臂；三达标存在 {n_exist_arm} 臂 ({round(n_exist_arm/total*100,1)}%)；"
          f"含至少一臂三达标的蛋白 {n_prot_any}/{len(by_prot)}")
    hdr = f"{'pdb':8s} {'arm':7s} {'triple':>6s} {'rate':>7s} {'p10':>4s} {'p25':>4s} {'p50':>4s} {'p100':>5s} {'p200':>5s}  {'main':>14s}"
    print(hdr)
    for r in rows:
        pr = r["prefix"]
        main = ",".join(r["main_cause_if_none"]) if r["main_cause_if_none"] else "-"
        print(f"{r['pdb']:8s} {r['arm']:7s} {r['triple_pass']:6d} "
              f"{(str(r['triple_rate'])+'%') if r['triple_rate'] is not None else '-':>7s} "
              f"{pr.get('p10','-'):>4} {pr.get('p25','-'):>4} {pr.get('p50','-'):>4} "
              f"{pr.get('p100','-'):>5} {pr.get('p200','-'):>5}  {main:>14s}")
    print("main cause among no-triple arms:", main_cause_counts)
    print(f"已写 {args.out}", flush=True)


if __name__ == "__main__":
    main()
