"""阶段1 结果汇总：读 output/compare/*/summary.json 输出对比表。"""
import glob
import json
import os

base = "/data/nfs/IC/baokun_yu/ConfuMPNN/code/output/compare"
rows = []
for d in sorted(glob.glob(os.path.join(base, "*/summary.json"))):
    with open(d) as f:
        s = json.load(f)
    rows.append({
        "name": os.path.basename(os.path.dirname(d)),
        "pH": s["pH"],
        "target": s["target_charge"],
        "mean": s["mean_charge"],
        "std": s["std_charge"],
        "native": s["native_charge"],
    })

print(f"{'实验':<28} {'pH':>5} {'target':>7} {'mean±std':>14} {'偏差':>8}")
print("-" * 70)
for r in sorted(rows, key=lambda x: (x["name"].split("_")[0], x["pH"], x["target"])):
    model = "MoMPNN  " if r["name"].startswith("mompnn") else "Ligand  "
    bias = r["mean"] - r["target"]
    print(f"{model}{r['name'][:20]:<20} {r['pH']:>5} {r['target']:>7.0f} "
          f"{r['mean']:>+7.2f}±{r['std']:.2f} {bias:>+7.2f}")
