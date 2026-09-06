"""exp7b 分析：从 output/exp_pH2_{prot,lig}/ 计算 pH 细网格 × 三处理汇总。

每个 (mode, pdb, treatment, ph, arm)：
  mean_q(q_own@ph), std_q, dev_of_mean=|mean−target|, per-seq 命中 |q−target|≤2,
  H2(dev_of_mean≤2), mean_rec, mean_pos, mean_neg, mean_his, mean_charged;
  跨 pH 配对 identity vs 7.4（同 treatment/arm 同 seed）。

输出：
  output/exp_pH2_{mode}/_summary.json
  output/exp_pH2_{mode}/_report_tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(ROOT / "code"))

GRID = [5.0, 5.5, 6.0, 6.5, 7.0, 7.4, 8.0, 8.5, 9.0]
ARM_LABELS = ["native", "n2", "p2", "n8", "p8"]
ARM_DQ = {"native": 0, "n2": -2, "p2": 2, "n8": -8, "p8": 8}
TREATS = ["T1", "T2", "T3"]
PH74 = "7.4"


def load_arms(out_root, pdb, treat):
    """返回 {ph_str: {arm: block}}；block=json.load 文件。"""
    d = {}
    tdir = out_root / pdb / treat
    if not tdir.is_dir():
        return d
    for phd in sorted(tdir.iterdir()):
        if not phd.is_dir():
            continue
        ph = phd.name
        d[ph] = {}
        for arm in ARM_LABELS:
            fp = phd / f"{arm}.json"
            if fp.is_file():
                d[ph][arm] = json.load(open(fp))
    return d


def ident_mean(rec_a, rec_b):
    """records 顺序即 seed 顺序。返回平均逐序列一致率。"""
    v = []
    for ra, rb in zip(rec_a, rec_b):
        if ra["seq"] == rb["seq"]:
            v.append(1.0)
        else:
            v.append(sum(c1 == c2 for c1, c2 in zip(ra["seq"], rb["seq"])) / len(ra["seq"]))
    return float(np.mean(v)) if v else None


def arm_stats(block):
    records = block["records"]
    q = np.array([r["q_own"] for r in records])
    target = block["target"]
    dev = np.abs(q - target)
    return {
        "n": len(q),
        "target": target,
        "dq": block.get("dq"),
        "target_eff": block.get("target_eff"),
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
    out_root = ROOT / f"output/exp_pH2_{mode}"
    summary = {"mode": mode, "grid": GRID, "proteins": {}}
    pdbs = sorted(d.name for d in out_root.iterdir()
                  if d.is_dir() and (d / "native.json").is_file())
    for pdb in pdbs:
        native = json.load(open(out_root / pdb / "native.json"))
        ps = {"pdb": pdb, "L": native["L"],
              "native_q_by_pH": native.get("native_q_by_pH"),
              "round_target_by_pH": native.get("round_target_by_pH"),
              "treatments": {}}
        for treat in TREATS:
            arms_by_ph = load_arms(out_root, pdb, treat)
            if not arms_by_ph:
                continue
            t_out = {"by_pH": {}, "identity_vs_74": {}}
            for ph in GRID:
                phs = str(ph)
                if phs not in arms_by_ph:
                    continue
                t_out["by_pH"][phs] = {}
                for arm in ARM_LABELS:
                    blk = arms_by_ph[phs].get(arm)
                    if blk:
                        t_out["by_pH"][phs][arm] = arm_stats(blk)
            # identity vs 7.4: per arm, ph != 7.4 vs 7.4
            if PH74 in arms_by_ph:
                t_out["identity_vs_74"] = {}
                for ph in GRID:
                    phs = str(ph)
                    if phs == PH74 or phs not in arms_by_ph:
                        continue
                    t_out["identity_vs_74"][phs] = {}
                    for arm in ARM_LABELS:
                        a74 = arms_by_ph[PH74].get(arm)
                        ap = arms_by_ph[phs].get(arm)
                        if a74 and ap and len(a74["records"]) == len(ap["records"]):
                            t_out["identity_vs_74"][phs][arm] = round(
                                ident_mean(a74["records"], ap["records"]), 4)
            ps["treatments"][treat] = t_out
        summary["proteins"][pdb] = ps
    (out_root / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def write_tables(summary):
    mode = summary["mode"]
    out_root = ROOT / f"output/exp_pH2_{mode}"
    md = [f"# exp_pH2_{mode} report tables（作图数据）2026-09-06",
          "pH 细网格 × T1(原始)/T2(7.4校准外推)/T3(逐pH现场标定)；n=60/臂。"]
    for pdb, ps in summary["proteins"].items():
        q = ps.get("native_q_by_pH", {})
        qs = " ".join(f"@{ph}:{q.get(str(ph), float('nan')):+.2f}" for ph in [5.0, 7.4, 9.0])
        md.append(f"\n## {pdb} L={ps['L']} native_q {qs}\n")
        for treat in TREATS:
            if treat not in ps["treatments"]:
                continue
            to = ps["treatments"][treat]
            md.append(f"\n### {treat}\n")
            md.append("| pH | 臂 | target | mean_q | std | dev_of_mean | per-seq≤2 | H2 | "
                      "rec | pos | neg | his | charged | id_vs74 |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for ph in GRID:
                phs = str(ph)
                if phs not in to["by_pH"]:
                    continue
                for arm in ARM_LABELS:
                    st = to["by_pH"][phs].get(arm)
                    if not st:
                        continue
                    idv = to.get("identity_vs_74", {}).get(phs, {}).get(arm)
                    md.append(f"| {ph} | {arm} | {st['target']:+d} | {st['mean_q']:+.2f} | "
                              f"{st['std_q']:.2f} | {st['dev_of_mean']:.2f} | "
                              f"{st['per_seq_hit_le2']:.3f} | {'Y' if st['H2'] else 'N'} | "
                              f"{st['mean_rec']:.3f} | {st['mean_pos']:.1f} | "
                              f"{st['mean_neg']:.1f} | {st['mean_his']:.1f} | "
                              f"{st['mean_charged']:.1f} | "
                              f"{'-' if idv is None else f'{idv:.3f}'} |")
    (out_root / "_report_tables.md").write_text("\n".join(md))
    print(f"wrote tables {out_root / '_report_tables.md'}")


if __name__ == "__main__":
    for m in sys.argv[1:] or ["prot", "lig"]:
        s = summarize_mode(m)
        write_tables(s)
        print(f"summarized mode={m}")
