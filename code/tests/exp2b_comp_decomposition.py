"""exp2b — 组成分解（膨胀/删减/替换）分析（2026-09-06，S2，CPU）。

用 output/exp_control_{prot,lig}/ 已有 route B(encoder)/C(bias)/A(裸) 序列，
对每蛋白每臂（B/C）或 pool（A）逐设计序列 vs native（同骨架逐位）做电荷组成分解：
  - Δtotal_charged：由 膨胀(加带电残基 add) / 删减(删带电残基 dele) / 置换(D/E↔K/R 对侧 flip) 贡献
  - 逐位识别：native/design 在 D/E/K/R 出现/消失/换成对侧；同号置换(D↔E,K↔R) 另记 same_sub
  - cr = charged_design/charged_native；Δn_pos(K+R)、Δn_neg(D+E)

route 映射：
  prot: B=route_B(encoder+cal, per-protein 校准), C=route_C(bias-only), A=route_A(裸 pool)
  lig : B=routeB_cal(encoder+cal), C=routeC(bias-only), A=routeA(裸 pool)

输出：
  output/exp2_comp_decomposition.json
  output/exp2_comp_decomposition_tables.md（人读）
用法（confumpnn 环境）：
  PYTHONPATH=code python code/tests/exp2b_comp_decomposition.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
POS = set("KR")
NEG = set("DE")
ARM_LABELS = ["native", "n2", "p2", "n8", "p8"]
ARM_DQ = {"native": 0, "n2": -2, "p2": 2, "n8": -8, "p8": 8}


def cls(a):
    if a in POS:
        return "pos"
    if a in NEG:
        return "neg"
    return None


def decomp_one(native, seq):
    """逐位比较，返回电荷组成事件计数。"""
    ev = {"add_pos": 0, "add_neg": 0, "del_pos": 0, "del_neg": 0,
          "flip_pn": 0, "flip_np": 0, "same_sub": 0, "neutral": 0,
          "n_pos_d": 0, "n_neg_d": 0}
    for a, b in zip(native, seq):
        ca, cb = cls(a), cls(b)
        if ca == "pos":
            ev["n_pos_d"] += 0
        if ca is None and cb is None:
            ev["neutral"] += 1
        elif ca is None and cb == "pos":
            ev["add_pos"] += 1
        elif ca is None and cb == "neg":
            ev["add_neg"] += 1
        elif ca == "pos" and cb is None:
            ev["del_pos"] += 1
        elif ca == "neg" and cb is None:
            ev["del_neg"] += 1
        elif ca == "pos" and cb == "neg":
            ev["flip_pn"] += 1
        elif ca == "neg" and cb == "pos":
            ev["flip_np"] += 1
        elif ca == cb:
            ev["same_sub"] += 1
    # design counts
    ev["n_pos_d"] = seq.count("K") + seq.count("R")
    ev["n_neg_d"] = seq.count("D") + seq.count("E")
    return ev


def summarize(events, nat_pos, nat_neg):
    """事件列表 → 均值聚合。"""
    n = len(events)
    def mean(k):
        return round(float(np.mean([e[k] for e in events])), 3)
    add = mean("add_pos") + mean("add_neg")
    dele = mean("del_pos") + mean("del_neg")
    flip = mean("flip_pn") + mean("flip_np")
    same = mean("same_sub")
    add_pos, add_neg = mean("add_pos"), mean("add_neg")
    del_pos, del_neg = mean("del_pos"), mean("del_neg")
    flip_pn, flip_np = mean("flip_pn"), mean("flip_np")
    d_pos = mean("n_pos_d") - nat_pos
    d_neg = mean("n_neg_d") - nat_neg
    char_nat = nat_pos + nat_neg
    cr = round((mean("n_pos_d") + mean("n_neg_d")) / char_nat, 3) if char_nat else None
    tot_events = add + dele + flip
    return {
        "n": n,
        "cr": cr,
        "charged_native": char_nat,
        "n_pos_native": nat_pos,
        "n_neg_native": nat_neg,
        "add_total": round(add, 3),
        "del_total": round(dele, 3),
        "flip_total": round(flip, 3),
        "same_sub": round(same, 3),
        "add_pos": add_pos, "add_neg": add_neg,
        "del_pos": del_pos, "del_neg": del_neg,
        "flip_pos2neg": flip_pn, "flip_neg2pos": flip_np,
        "d_total_charged": round(add - dele, 3),
        "d_pos": round(d_pos, 3), "d_neg": round(d_neg, 3),
        "frac_add": round(add / tot_events, 3) if tot_events else None,
        "frac_del": round(dele / tot_events, 3) if tot_events else None,
        "frac_flip": round(flip / tot_events, 3) if tot_events else None,
        "mean_recovery": round(float(np.mean([e.get("rec", float("nan")) for e in events])), 3)
        if "rec" in events[0] else None,
    }


def load_native_prot(pdb):
    m = json.load(open(ROOT / f"output/exp_control_prot/{pdb}/meta.json"))
    return m["native"]


def load_native_lig(pdb):
    d = json.load(open(ROOT / f"output/exp_control_lig/{pdb}/native.json"))
    return d["native"]


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_prot_arm(pdb, route, arm):
    """返回 [{seq, rec}]。route: B/C 需 arm；A pool。"""
    base = ROOT / f"output/exp_control_prot/{pdb}/route_{route}"
    if route == "A":
        return [{"seq": r["seq"], "rec": r.get("recovery")} for r in iter_jsonl(base / "per_seq.jsonl")]
    return [{"seq": r["seq"], "rec": r.get("recovery")}
            for r in iter_jsonl(base / f"arm_{arm}" / "per_seq.jsonl")]


def read_lig_route(pdb, route, arm):
    base = ROOT / f"output/exp_control_lig/{pdb}"
    if route == "A":
        d = json.load(open(base / "routeA" / "sequences.json"))
    elif route == "B":
        d = json.load(open(base / f"routeB_cal/arm_{arm}/sequences.json"))
    else:
        d = json.load(open(base / f"routeC/arm_{arm}/sequences.json"))
    out = []
    for i, s in enumerate(d["seqs"]):
        rec = d.get("recs", [None] * len(d["seqs"]))[i] if "recs" in d else None
        out.append({"seq": s, "rec": rec})
    return out


def analyze_mode(mode, out):
    mode_out = {"proteins": {}}
    if mode == "prot":
        pdbs = ["1AZM", "1AS2", "1BJ4"]
        # (输出 route 标签, 磁盘 route 目录, arm；arm=None 表示 pool)
        routes = ([("B", "B", a) for a in ARM_LABELS]
                  + [("C", "C", a) for a in ARM_LABELS]
                  + [("A", "A", None)])
    else:
        pdbs = ["5O60_E", "1CGE", "2FEO"]
        routes = ([("B", "B", a) for a in ARM_LABELS]
                  + [("C", "C", a) for a in ARM_LABELS]
                  + [("A", "A", None)])
    for pdb in pdbs:
        native = load_native_prot(pdb) if mode == "prot" else load_native_lig(pdb)
        nat_pos = sum(native.count(x) for x in "KR")
        nat_neg = sum(native.count(x) for x in "DE")
        p_out = {"L": len(native), "n_pos_native": nat_pos, "n_neg_native": nat_neg,
                 "charged_native": nat_pos + nat_neg, "native_q": None,
                 "routes": {}}
        # native_q@7.4 for reference
        if mode == "prot":
            m = json.load(open(ROOT / f"output/exp_control_prot/{pdb}/meta.json"))
            p_out["native_q"] = m.get("native_charge")
            p_out["round_target"] = m.get("round_target")
        else:
            d = json.load(open(ROOT / f"output/exp_control_lig/{pdb}/native.json"))
            p_out["native_q"] = d.get("native_charge")
        for routelab, disk_route, arm in routes:
            if arm is None:
                seqs = read_prot_arm(pdb, disk_route, None) if mode == "prot" \
                    else read_lig_route(pdb, disk_route, None)
                events = [dict(decomp_one(native, r["seq"]), rec=r["rec"]) for r in seqs]
                p_out["routes"][routelab] = {"pool": summarize(events, nat_pos, nat_neg)}
            else:
                seqs = read_prot_arm(pdb, disk_route, arm) if mode == "prot" \
                    else read_lig_route(pdb, disk_route, arm)
                events = [dict(decomp_one(native, r["seq"]), rec=r["rec"]) for r in seqs]
                if routelab not in p_out["routes"]:
                    p_out["routes"][routelab] = {}
                p_out["routes"][routelab][arm] = summarize(events, nat_pos, nat_neg)
        mode_out["proteins"][pdb] = p_out
    out[mode] = mode_out


def _fmt_rec(x):
    return "-" if x is None else f"{x:.3f}"


def write_md(out):
    md = ["# exp2b 组成分解（膨胀/删减/置换）报告表（作图数据）2026-09-06",
          "", "逐位定义：charged={D,E,K,R}；add=非带电位→带电位；del=带电位→非带电位；",
          "flip=对侧置换(D/E↔K/R)；same_sub=同号置换(D↔E 或 K↔R)。cr=设计带电总数/native 带电总数。",
          ""]
    for mode in ["prot", "lig"]:
        md.append(f"\n## 模式 {mode}\n")
        for pdb, po in out[mode]["proteins"].items():
            md.append(f"### {pdb} L={po['L']} native: pos={po['n_pos_native']} "
                      f"neg={po['n_neg_native']} charged={po['charged_native']} "
                      f"q74={po['native_q']}\n")
            md.append("| route | 臂 | cr | Δtotal | Δpos(KR) | Δneg(DE) | add(pos/neg) | "
                      "del(pos/neg) | flip(P→N/N→P) | same_sub | frac add/del/flip | rec |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for routelab, rblk in po["routes"].items():
                items = list(rblk.items())
                for arm, st in items:
                    arm_s = arm if arm != "pool" else "pool(全)"
                    fs = st.get("frac_add")
                    frac_s = (f"{st['frac_add']:.2f}/{st['frac_del']:.2f}/"
                              f"{st['frac_flip']:.2f}") if fs is not None else "-"
                    md.append(
                        f"| {routelab} | {arm_s} | {st['cr']} | {st['d_total_charged']:+.1f} | "
                        f"{st['d_pos']:+.1f} | {st['d_neg']:+.1f} | "
                        f"{st['add_total']:.1f}(+{st['add_pos']:.1f}/-{st['add_neg']:.1f}) | "
                        f"{st['del_total']:.1f}(+{st['del_pos']:.1f}/-{st['del_neg']:.1f}) | "
                        f"{st['flip_total']:.1f}({st['flip_pos2neg']:.1f}/{st['flip_neg2pos']:.1f}) | "
                        f"{st['same_sub']:.1f} | {frac_s} | "
                        f"{_fmt_rec(st['mean_recovery'])} |")
    (ROOT / "output/exp2_comp_decomposition_tables.md").write_text("\n".join(md))
    print("wrote tables output/exp2_comp_decomposition_tables.md")


if __name__ == "__main__":
    out = {}
    for mode in ["prot", "lig"]:
        analyze_mode(mode, out)
    (ROOT / "output/exp2_comp_decomposition.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote output/exp2_comp_decomposition.json")
    write_md(out)
