"""E1b 验证扩展：汇总所有打分为逐样本明细 + 按条件汇总。
输入：code/output/e1_ext/{pdb}_{model}/{condition}/ 下
  seqs.fa（header 含 pH/charge）、plddt.csv、tm.csv、seqs.fa-protein_sol.csv、seqs.fa.tm.csv
输出：code/output/e1_ext/summary_all.csv（逐样本）、summary_cond.csv（按 pdb/model/condition 汇总）
用法（任意环境）：python e1_ext_summarize.py
"""
import csv
import glob
import os
import re

ROOT = "/data/nfs/IC/baokun_yu/ConfuMPNN/code/output/e1_ext"
CHARGE_RE = re.compile(r"charge=([+-]?\d+(?:\.\d+)?)")


def read_fasta(path):
    """返回 [(name, seq)]，按文件顺序。"""
    out, name, lines = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(lines)))
            name, lines = line[1:], []
        elif line:
            lines.append(line)
    if name is not None:
        out.append((name, "".join(lines)))
    return out


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:50]


def cond_dir(name):
    """把目录名条件化：baseline / pH4.0_t-5。"""
    return name


def main():
    all_rows = []
    conds = sorted(glob.glob(os.path.join(ROOT, "*/", "*/", "")))
    # 排除 probe 目录
    conds = [c for c in conds if not os.path.basename(c.rstrip("/")).startswith("_probe")]
    print(f"共 {len(conds)} 个条件目录")

    for d in conds:
        parts = os.path.normpath(d).split(os.sep)
        pdb_model, cond = parts[-2], os.path.basename(d.rstrip("/"))
        pdb, model = pdb_model.split("_", 1)
        fa_path = os.path.join(d, "seqs.fa")
        if not os.path.exists(fa_path):
            continue
        seqs = read_fasta(fa_path)

        plddt = {r["name"]: float(r["mean_plddt"]) for r in read_csv(os.path.join(d, "plddt.csv"))}
        tm = {r["name"]: (float(r["tm_score"]) if r["tm_score"] else None)
              for r in read_csv(os.path.join(d, "tm.csv"))}
        sol = {r["ID"]: (float(r["percent-sol"]) if r["percent-sol"] else None)
               for r in read_csv(os.path.join(d, "seqs.fa-protein_sol.csv"))}
        tmb = {r["name"]: float(r["mean_tm"])
               for r in read_csv(os.path.join(d, "seqs.fa.tm.csv"))}

        for name, seq in seqs:
            is_native = name.startswith("native")
            m = CHARGE_RE.search(name)
            charge = float(m.group(1)) if m else None
            target = 0.0 if is_native else None
            safe_name = safe(name)
            row = {
                "pdb": pdb, "model": model, "condition": cond,
                "sample": name, "is_native": "Y" if is_native else "N",
                "charge": charge, "len": len(seq),
                "plddt": plddt.get(name) or plddt.get(safe_name),
                "tm_score": tm.get(safe_name),
                "protsol": sol.get(name) or sol.get(safe_name),
                "temberture_tm": tmb.get(name) or tmb.get(safe_name),
            }
            all_rows.append(row)

    with open(os.path.join(ROOT, "summary_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [])
        w.writeheader()
        w.writerows(all_rows)
    print(f"逐样本明细: {len(all_rows)} 行 -> summary_all.csv")

    # 按 pdb/model/condition 汇总（样本均值，含 native 对照单独列）
    groups = {}
    for r in all_rows:
        if r["is_native"] == "Y":
            continue
        k = (r["pdb"], r["model"], r["condition"])
        groups.setdefault(k, []).append(r)
    with open(os.path.join(ROOT, "summary_cond.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pdb", "model", "condition", "n", "charge_mean", "charge_dev_mean",
                    "plddt_mean", "tm_score_mean", "protsol_mean", "temberture_mean"])
        for (pdb, model, cond), rows in sorted(groups.items()):
            def mean(key):
                vals = [r[key] for r in rows if r[key] is not None]
                return round(sum(vals) / len(vals), 3) if vals else ""
            target = None
            if cond != "baseline":
                m = re.search(r"t(-?\d+)", cond)
                target = float(m.group(1)) if m else 0.0
            devs = [abs(r["charge"] - target) for r in rows if r["charge"] is not None and target is not None]
            w.writerow([pdb, model, cond, len(rows), mean("charge"),
                        round(sum(devs) / len(devs), 3) if devs else "",
                        mean("plddt"), mean("tm_score"), mean("protsol"), mean("temberture_tm")])
    print("按条件汇总 -> summary_cond.csv")


if __name__ == "__main__":
    main()
