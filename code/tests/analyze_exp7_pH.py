"""exp7 分析：从 output/exp_pH_{prot,lig}/ 计算两 class 汇总。

class A（同 pH 5 臂）: 每 (蛋白, pH, 臂) 的 mean_q(q_own), std, dev_of_mean=|mean−target|,
  per-seq 命中 |q−target|≤2, mean_abs_dev, H2(dev_of_mean≤2), recovery；跨 pH 配对 identity。
class B（固定 target 换 pH）: 每 (蛋白, target, pH) 的 mean_q(q_own)/std、per-seq 命中、
  组成 (n_pos/n_neg/n_his/charged)、相对 7.4 的逐序列 identity（同 seed）。

输出：
  output/exp_pH_{mode}/_summary.json   （机器可读）
  output/exp_pH_{mode}/_report_tables.md（人读，作图数据）
报告 compare/report_2026-09-06_exp7_pH_{prot,lig}.md 由主流程从这些表拼装。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(ROOT / "code"))

PHIS = [5.0, 7.4, 9.0]
ARMS = ["native", "n2", "p2", "n8", "p8"]


def ident_arrays(rec_a, rec_b):
    """同 seed 配对逐序列 identity（records 顺序即 seed 顺序）。返回 (mean, list)。"""
    vals = []
    for ra, rb in zip(rec_a, rec_b):
        s1, s2 = ra["seq"], rb["seq"]
        vals.append(sum(c1 == c2 for c1, c2 in zip(s1, s2)) / len(s1))
    return float(np.mean(vals)), vals


def arm_stats(records, target):
    q = np.array([r["q_own"] for r in records])
    dev = np.abs(q - target)
    return {
        "n": len(q),
        "mean_q": round(float(q.mean()), 3),
        "std_q": round(float(q.std()), 3),
        "dev_of_mean": round(float(abs(q.mean() - target)), 3),
        "mean_abs_dev": round(float(dev.mean()), 3),
        "per_seq_hit_le2": round(float((dev <= 2.0).mean()), 4),
        "H2": bool(abs(q.mean() - target) <= 2.0),
        "mean_rec": round(float(np.mean([r["rec"] for r in records])), 4),
        "mean_pos": round(float(np.mean([r["n_pos"] for r in records])), 3),
        "mean_neg": round(float(np.mean([r["n_neg"] for r in records])), 3),
        "mean_his": round(float(np.mean([r["n_his"] for r in records])), 3),
        "mean_charged": round(float(np.mean([r["n_charged"] for r in records])), 3),
    }


def summarize_mode(mode):
    out_root = ROOT / f"output/exp_pH_{mode}"
    summary = {"mode": mode, "proteins": {}}
    pdbs = sorted(d.name for d in out_root.iterdir()
                  if d.is_dir() and (d / "native.json").is_file())
    for pdb in pdbs:
        pdir = out_root / pdb
        native = json.load(open(pdir / "native.json"))
        ps = {"pdb": pdb, "L": native["L"],
              "native_q_by_pH": native["native_q_by_pH"],
              "round_target_74": native["round_target_74"],
              "classA": {}, "classB": {}}
        ca_path = pdir / "classA.json"
        if ca_path.is_file():
            ca = json.load(open(ca_path))
            ps["classA"]["cal_source"] = ca.get("cal_source")
            ps["classA"]["slope"] = ca.get("slope")
            ps["classA"]["intercept"] = ca.get("intercept")
            ps["classA"]["by_pH"] = {}
            for ph in PHIS:
                phs = str(ph)
                if phs not in ca["by_pH"]:
                    continue
                ps["classA"]["by_pH"][phs] = {}
                for arm in ARMS:
                    blk = ca["by_pH"][phs].get(arm)
                    if not blk:
                        continue
                    st = arm_stats(blk["records"], blk["target"])
                    st["target"] = blk["target"]
                    st["target_eff"] = blk.get("target_eff")
                    ps["classA"]["by_pH"][phs][arm] = st
            # 跨 pH identity：同臂下 7.4 vs 5/9
            if "7.4" in ca["by_pH"]:
                ps["classA"]["identity_vs_74"] = {}
                for ph in PHIS:
                    phs = str(ph)
                    if phs == "7.4" or phs not in ca["by_pH"]:
                        continue
                    ps["classA"]["identity_vs_74"][phs] = {}
                    for arm in ARMS:
                        ba = ca["by_pH"]["7.4"].get(arm, {}).get("records")
                        bb = ca["by_pH"][phs].get(arm, {}).get("records")
                        if ba and bb and len(ba) == len(bb):
                            mid, _ = ident_arrays(ba, bb)
                            ps["classA"]["identity_vs_74"][phs][arm] = round(mid, 4)
        cb_path = pdir / "classB.json"
        if cb_path.is_file():
            cb = json.load(open(cb_path))
            for tlab, tinfo in cb["targets"].items():
                tgt = tinfo["target"]
                ps["classB"][tlab] = {"target": tgt, "by_pH": {},
                                      "identity_vs_74": {}}
                base74 = None
                for ph in PHIS:
                    phs = str(ph)
                    blk = tinfo["pH"].get(phs)
                    if not blk:
                        continue
                    st = arm_stats(blk["records"], tgt)
                    ps["classB"][tlab]["by_pH"][phs] = st
                # identity vs 7.4 by target
                if "7.4" in tinfo["pH"]:
                    base74 = tinfo["pH"]["7.4"]["records"]
                for ph in PHIS:
                    phs = str(ph)
                    if phs == "7.4" or phs not in tinfo["pH"]:
                        continue
                    bb = tinfo["pH"][phs]["records"]
                    if base74 and bb and len(base74) == len(bb):
                        mid, _ = ident_arrays(base74, bb)
                        ps["classB"][tlab]["identity_vs_74"][phs] = round(mid, 4)
        summary["proteins"][pdb] = ps
    (out_root / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def write_tables(summary):
    mode = summary["mode"]
    out_root = ROOT / f"output/exp_pH_{mode}"
    md = [f"# exp_pH_{mode} report tables（作图数据）2026-09-06\n"]
    md.append(f"| 蛋白 | L | native_q@pH5 | @7.4 | @9 | round74 |")
    md.append("|---|---|---|---|---|---|")
    for pdb, ps in summary["proteins"].items():
        q = ps["native_q_by_pH"]
        md.append(f"| {pdb} | {ps['L']} | {q['5.0']:+.2f} | {q['7.4']:+.2f} | "
                  f"{q['9.0']:+.2f} | {ps['round_target_74']:+d} |")
    # class A tables
    for pdb, ps in summary["proteins"].items():
        if not ps["classA"]:
            continue
        md.append(f"\n## classA {pdb}\n")
        md.append("| pH | 臂 | target | mean_q | std | dev_of_mean | per-seq≤2 | H2 | "
                  "rec | pos | neg | his |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for ph in PHIS:
            phs = str(ph)
            if phs not in ps["classA"]["by_pH"]:
                continue
            for arm in ARMS:
                st = ps["classA"]["by_pH"][phs].get(arm)
                if not st:
                    continue
                md.append(f"| {ph} | {arm} | {st['target']:+d} | {st['mean_q']:+.2f} | "
                          f"{st['std_q']:.2f} | {st['dev_of_mean']:.2f} | "
                          f"{st['per_seq_hit_le2']:.3f} | {'Y' if st['H2'] else 'N'} | "
                          f"{st['mean_rec']:.3f} | {st['mean_pos']:.1f} | "
                          f"{st['mean_neg']:.1f} | {st['mean_his']:.1f} |")
        # identity across pH (class A, per arm)
        if ps["classA"].get("identity_vs_74"):
            md.append(f"\nclassA {pdb} 逐序列 identity vs pH7.4（同 seed 同臂，跨 pH 序列漂移）：\n")
            md.append("| 臂 | id(pH5 vs 7.4) | id(pH9 vs 7.4) |")
            md.append("|---|---|---|")
            for arm in ARMS:
                i5 = ps["classA"]["identity_vs_74"].get("5.0", {}).get(arm)
                i9 = ps["classA"]["identity_vs_74"].get("9.0", {}).get(arm)
                md.append(f"| {arm} | {'NA' if i5 is None else f'{i5:.3f}'} | "
                          f"{'NA' if i9 is None else f'{i9:.3f}'} |")
    # class B tables
    for pdb, ps in summary["proteins"].items():
        if not ps["classB"]:
            continue
        for tlab, tinfo in ps["classB"].items():
            md.append(f"\n## classB {pdb} {tlab} (target={tinfo['target']:+d})\n")
            md.append("| pH | mean_q(q_own) | std | dev_of_mean | per-seq≤2 | rec | pos | "
                      "neg | his | identity vs 7.4 |")
            md.append("|---|---|---|---|---|---|---|---|---|---|")
            for ph in PHIS:
                phs = str(ph)
                st = tinfo["by_pH"].get(phs)
                if not st:
                    continue
                i = tinfo.get("identity_vs_74", {}).get(phs)
                md.append(f"| {ph} | {st['mean_q']:+.2f} | {st['std_q']:.2f} | "
                          f"{st['dev_of_mean']:.2f} | {st['per_seq_hit_le2']:.3f} | "
                          f"{st['mean_rec']:.3f} | {st['mean_pos']:.1f} | "
                          f"{st['mean_neg']:.1f} | {st['mean_his']:.1f} | "
                          f"{'NA' if i is None else f'{i:.3f}'} |")
    (out_root / "_report_tables.md").write_text("\n".join(md))
    print(f"wrote tables {out_root / '_report_tables.md'}")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["prot", "lig"]:
        s = summarize_mode(m)
        write_tables(s)
        print(f"summarized mode={m}")
