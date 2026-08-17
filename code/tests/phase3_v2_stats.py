"""Phase 3 v2 复验统计 + 按 DESIGN_CRITERIA v2 判定。

读 `output/finetune_v2_validate/` 下采样 + 四指标打分管线产物：
    {pdb}/charge_stats.json          采样电荷统计（含防坍塌 diversity）
    {pdb}/{arm}/seqs.fa              序列（S4 固定位点检查 / H3 用）
    {pdb}/{arm}/folds/*.pdb + tm.csv TM-score（H1）
    {pdb}/{arm}/plddt.csv            ESMFold mean_plddt（H1/S2）
    {pdb}/{arm}/seqs.fa.tm.csv       TemBERTure Tm（S2）
    {pdb}/{arm}/seqs.fa.sol.csv      Protein-Sol %sol（S2）

判定（DESIGN_CRITERIA v2）：
    H1: TM 中位≥0.70 且折叠失败(TM<0.5)≤10%
    H2: |mean_charge - target| ≤ 2.0（target=None 的占位臂不判 H2）
    H3: 条件臂(结构化违规率) ≤ 基线臂(t1_base) + 5pp
    S1*: pairwise identity 落 0.4-0.7（<0.8 防坍塌）+ 位置熵报告
    S2: pLDDT/%sol/Tm 报告（不单独判 FAIL）
    S3: t2_ph（占位符）H1 达标 = 占位不破坏折叠（其余条件仍生效）
    S4: t1_cond 固定位点 100% 保持 = 指定氨基酸

用法（code/ 下，confumpnn 环境）：
    PYTHONPATH=. python tests/phase3_v2_stats.py [--root output/finetune_v2_validate]
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR.parent / "LigandMPNN"))

from data_utils import parse_PDB  # noqa: E402
from src.guided_sampler import extract_calpha_coords  # noqa: E402
from src.structure_aware_filter import default_config  # noqa: E402
from src.pka import AA_TO_IDX, STRONG_POSITIVE, STRONG_NEGATIVE  # noqa: E402

PDBS = ["1BC8", "1CRN", "1UBQ", "2LZM", "1b24A01"]
REF_PDB = {
    "1BC8": "input/1BC8_chainC.pdb",
    "1CRN": "input/1CRN.pdb",
    "1UBQ": "input/1UBQ.pdb",
    "2LZM": "input/2LZM.pdb",
    "1b24A01": "input/1b24A01.pdb",  # 正电验证蛋白（native ESMFold 84.8，可折叠）
}
ARMS = ["t1_cond", "t1_base", "t2_pos", "t2_pos_extreme", "t2_neg", "t2_ph"]
PLACEHOLDER_ARMS = {"t2_ph"}  # 占位臂（均值占位语义）：不判 H2，只看折叠
AA1 = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path):
    seqs = []
    name, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(buf)))
                name, buf = line[1:], []
            elif line:
                buf.append(line)
    if name is not None:
        seqs.append((name, "".join(buf)))
    return seqs


def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_charge_stats(root):
    out = {}
    for pdb in PDBS:
        p = root / pdb / "charge_stats.json"
        if p.exists():
            out[pdb] = json.load(open(p))
    return out


def get_arm_charge(stats, pdb, arm):
    return stats[pdb]["arms"].get(arm, {})


def tm_values(root, pdb, arm):
    """返回该臂 TM-score 列表（数值）。"""
    vals = []
    p = root / pdb / arm / "tm.csv"
    for r in read_csv(p):
        try:
            v = float(r["tm_score"])
            if np.isfinite(v):
                vals.append(v)
        except (KeyError, ValueError, TypeError):
            continue
    return vals


def plddt_values(root, pdb, arm):
    vals = []
    for r in read_csv(root / pdb / arm / "plddt.csv"):
        try:
            vals.append(float(r["mean_plddt"]))
        except (KeyError, ValueError, TypeError):
            continue
    return vals


def sol_values(root, pdb, arm):
    vals = []
    for r in read_csv(root / pdb / arm / "seqs.fa-protein_sol.csv"):
        try:
            vals.append(float(r["percent-sol"]))
        except (KeyError, ValueError, TypeError):
            continue
    return vals


def tm_values_temberture(root, pdb, arm):
    vals = []
    for r in read_csv(root / pdb / arm / "seqs.fa.tm.csv"):
        try:
            vals.append(float(r["mean_tm"]))
        except (KeyError, ValueError, TypeError):
            continue
    return vals


# ---- H3 事后结构违规检查 ----
def post_hoc_violations(coords, seq):
    """对完整序列做 4 条结构过滤器规则的事后检查，返回违规位置数。

    与 structure_aware_filter.py 的解码时语义一致，但针对完整序列：
    规则1 10Å 同号电荷≥6 / 规则2 10Å 正负对≥4 / 规则3 核心带电≥6 / 规则4 8Å 同号连通≥4。
    """
    cfg = default_config()
    L = len(seq)
    seq_int = np.array([AA_TO_IDX.get(a, 20) for a in seq], dtype=int)
    pos = np.array([a in STRONG_POSITIVE for a in seq])
    neg = np.array([a in STRONG_NEGATIVE for a in seq])
    charged = pos | neg

    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, 0.0)

    viol = np.zeros(L, dtype=bool)
    # 规则1
    nb = dist <= cfg["charge_cluster"]["radius"]
    pos_count = (nb & pos[None, :]).sum(axis=1)
    neg_count = (nb & neg[None, :]).sum(axis=1)
    viol |= pos & (pos_count >= cfg["charge_cluster"]["threshold"])
    viol |= neg & (neg_count >= cfg["charge_cluster"]["threshold"])
    # 规则2
    pairs = np.minimum(pos_count, neg_count)
    viol |= charged & (pairs >= cfg["salt_bridge"]["threshold"])
    # 规则3
    burial = (dist <= cfg["core_charge"]["burial_radius"]).sum(axis=1)
    burial_ratio = burial / burial.max() if burial.max() > 0 else burial
    charge8 = ((dist <= cfg["core_charge"]["charge_radius"]) & charged[None, :]).sum(axis=1)
    core = (burial_ratio > cfg["core_charge"]["burial_threshold"]) & (
        charge8 >= cfg["core_charge"]["charge_count"])
    viol |= charged & core
    # 规则4
    adj = dist <= cfg["same_sign_cluster"]["radius"]
    n_comp, labels = connected_components(csr_matrix(adj.astype(int)), directed=False)
    for c in range(n_comp):
        members = labels == c
        if int((members & pos).sum()) >= cfg["same_sign_cluster"]["threshold"]:
            viol |= members & pos
        if int((members & neg).sum()) >= cfg["same_sign_cluster"]["threshold"]:
            viol |= members & neg
    return int(viol.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/finetune_v2_validate")
    args = ap.parse_args()
    root = Path(args.root)
    stats = load_charge_stats(root)

    # 预加载 Cα 坐标（H3 用）
    ca = {}
    for pdb in PDBS:
        protein_dict, _, _, _, _ = parse_PDB(
            str(_CODE_DIR / REF_PDB[pdb]))
        ca[pdb] = extract_calpha_coords(protein_dict)

    report = {}
    for pdb in PDBS:
        arms = {}
        s = stats[pdb]
        native = s["native"]
        fixed_ids = s.get("fixed_ids", [])
        # 固定位点的 native 氨基酸（S4 参照）
        fixed_ref = {}
        if fixed_ids:
            resnames = []
            protein_dict, _, _, icodes, _ = parse_PDB(str(_CODE_DIR / REF_PDB[pdb]))
            rr = list(protein_dict["R_idx"].cpu().numpy())
            cl = list(protein_dict["chain_letters"])
            resnames = [str(cl[i]) + str(rr[i]) + icodes[i] for i in range(len(rr))]
            for i, nm in enumerate(resnames):
                if nm in fixed_ids:
                    fixed_ref[nm] = native[i]

        for arm in ARMS:
            a = {"charge": get_arm_charge(stats, pdb, arm)}
            tm = tm_values(root, pdb, arm)
            plddt = plddt_values(root, pdb, arm)
            sol = sol_values(root, pdb, arm)
            tmv = tm_values_temberture(root, pdb, arm)
            a["n"] = len(tm)
            a["TM"] = {"median": round(float(np.median(tm)), 4) if tm else None,
                       "fail_rate": round(float(np.mean([t < 0.5 for t in tm])), 3) if tm else None}
            a["pLDDT"] = round(float(np.mean(plddt)), 1) if plddt else None
            a["%sol"] = round(float(np.mean(sol)), 1) if sol else None
            a["Tm"] = round(float(np.mean(tmv)), 1) if tmv else None
            # S4：t1_cond 固定位点保持率
            if arm == "t1_cond" and fixed_ref:
                seqs = read_fasta(root / pdb / arm / "seqs.fa")
                protein_dict, _, _, icodes, _ = parse_PDB(
                    str(_CODE_DIR / REF_PDB[pdb]))
                rr = list(protein_dict["R_idx"].cpu().numpy())
                cl = list(protein_dict["chain_letters"])
                names = [str(cl[i]) + str(rr[i]) + icodes[i] for i in range(len(rr))]
                keeps = 0
                for nm, ref_aa in fixed_ref.items():
                    posi = names.index(nm)
                    ok = all(s_[posi] == ref_aa for _, s_ in seqs)
                    keeps += int(ok)
                a["S4_fixed_keep"] = f"{keeps}/{len(fixed_ref)}"
            # H3：违规率（基线 t1_base 单独记）
            if arm in ("t1_base", "t2_pos", "t2_neg"):
                seqs = read_fasta(root / pdb / arm / "seqs.fa")
                viol = [post_hoc_violations(ca[pdb], s_) for _, s_ in seqs]
                a["H3_viol_per_seq"] = [round(v / len(seqs[0][1]), 3) for v in viol]
                a["H3_viol_rate"] = round(float(np.mean(a["H3_viol_per_seq"])), 4)
            arms[arm] = a

        report[pdb] = {"native_charge": s["native_charge"],
                       "fixed_ids": fixed_ids, "arms": arms}

    # ---- v2 判定 ----
    print("=" * 96)
    print("ConfuMPNN 第十五轮复验 —— 按 DESIGN_CRITERIA v2 判定")
    print("=" * 96)
    for pdb in PDBS:
        r = report[pdb]
        print(f"\n### {pdb}  native_charge={r['native_charge']:+.2f}  "
              f"固定位点={r['fixed_ids']}")
        for arm in ARMS:
            a = r["arms"][arm]
            tgt = a["charge"].get("target")
            dev = a["charge"].get("mean_abs_dev")
            # 占位臂（t2_ph 均值占位）不判 H2——它语义是"温和默认电荷"，非精确控制
            h2 = (dev is not None and tgt is not None
                  and arm not in PLACEHOLDER_ARMS and dev <= 2.0)
            tm_med, fail = a["TM"]["median"], a["TM"]["fail_rate"]
            h1 = (tm_med is not None and tm_med >= 0.70 and fail is not None
                  and fail <= 0.10)
            mark_h1 = "✅" if h1 else ("❌" if tm_med is not None else "—")
            mark_h2 = ("—" if arm in PLACEHOLDER_ARMS else
                       ("✅" if h2 else ("❌" if dev is not None else "—")))
            print(f"  {arm:9s} | target={tgt if tgt is not None else '占位':>5} "
                  f"dev={dev if dev is not None else '—':>5} "
                  f"TM中位={tm_med} 失败率={fail} pLDDT={a['pLDDT']} "
                  f"%sol={a['%sol']} Tm={a['Tm']} pairID={a['charge']['diversity']['pairwise_identity']} "
                  f"Hpos={a['charge']['diversity']['mean_position_entropy']} "
                  f"S4固定={a.get('S4_fixed_keep','—')} "
                  f"H3viol={a.get('H3_viol_rate','—')}")
            print(f"         H1={mark_h1} H2={mark_h2}")

        # 汇总判定
        base_viol = r["arms"]["t1_base"].get("H3_viol_rate")
        print("  [H3] t1_base 违规率 =", base_viol)
        for arm in ("t2_pos", "t2_neg"):
            v = r["arms"][arm].get("H3_viol_rate")
            if base_viol is not None and v is not None:
                print(f"       {arm} {v} vs base {base_viol} "
                      f"→ {'PASS' if v <= base_viol + 0.05 else 'FAIL'}")

    with open(root / "v2_judgment.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整判定 JSON → {root / 'v2_judgment.json'}")


if __name__ == "__main__":
    main()
