"""Task2 汇总分析：fix-binding（output/fixbinding_v14）vs unfix（generalization_ligand_v14_clean）。

对比口径：
  - 电荷：每臂 mean net charge / dev / H2 命中(|dev|≤2.0)（读 validation.json）。
  - 组成（删减）：D/E 与 K/R 计数，native vs 生成均值，总量 + 3 区
    （pocket/surface/core；pocket=Cα-配体≤8Å，surface=frac_sasa≥0.25 非口袋，
     core=frac_sasa<0.25 非口袋）。与 compare_comp_ligand 的 count_dk 口径一致。
  - H3：单独跑 h3_charge_legality（--fix-root），本脚本只汇总对比 h3 JSON 的 viol_rate。

输出：
  --out JSON（每蛋白每臂 fix/unfix 对比 + 汇总）
  stdout 打印核心表
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
sys.path.insert(0, str(_PROJECT_DIR / "code"))
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))

from data_utils import parse_PDB, restype_int_to_str  # noqa: E402
from src.sasa import fractional_sasa  # noqa: E402

ARMS = ["native", "n2", "p2", "n8", "p8"]
CHARGED = "DEKR"


def native_seq_from_pd(protein_dict):
    S = protein_dict["S"].reshape(-1).cpu().numpy()
    return "".join(restype_int_to_str[int(a)] for a in S)


def count_de_kr(seq, idx=None):
    """返回 (DE 计数, KR 计数) 于全序列或给定索引集 idx（复现 compare_comp count 口径）。"""
    if idx is None:
        return (sum(1 for a in seq if a in "DE"), sum(1 for a in seq if a in "KR"))
    de = sum(1 for i in idx if seq[i] in "DE")
    kr = sum(1 for i in idx if seq[i] in "KR")
    return de, kr


def pocket_residues(protein_dict, cutoff=8.0):
    Y = protein_dict.get("Y")
    X = protein_dict["X"]
    if Y is None or Y.numel() == 0:
        return None
    Yc = Y.reshape(-1, 3).cpu().numpy()
    CA = X[:, 1, :].cpu().numpy()
    d = np.linalg.norm(CA[:, None, :] - Yc[None, :, :], axis=-1)
    return np.where(d.min(axis=1) < cutoff)[0]


def read_seqfa(fa):
    """读 seqs.fa → (生成序列 list, native 序列)。"""
    lines = open(fa).read().splitlines()
    seqs, cur = [], None
    for line in lines:
        if line.startswith(">"):
            if cur is not None:
                seqs.append(cur)
            cur = (line[1:], "")
        elif cur is not None:
            cur = (cur[0], cur[1] + line)
    if cur is not None:
        seqs.append(cur)
    gen = [s for n, s in seqs if not n.startswith("native")]
    nat = [s for n, s in seqs if n.startswith("native")]
    return gen, (nat[0] if nat else None)


def load_arm_summary(vjson_path):
    if not Path(vjson_path).exists():
        return None
    d = json.load(open(vjson_path))
    out = {}
    for arm, a in d.get("arms", {}).items():
        out[arm] = {"target": a["target"], "mean_charge": a["mean_charge"],
                    "std_charge": a["std_charge"], "dev": a["dev"],
                    "hit": bool(a["dev"] <= 2.0)}
    return {"native_charge": d.get("native_charge"), "L": d.get("L"), "arms": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-root", default=str(_PROJECT_DIR / "output/fixbinding_v14" / "ligand"),
                    help="fix-binding 生成根（含 <pdb>/pH7.4/arm_*/seqs.fa）")
    ap.add_argument("--unfix-root",
                    default=str(_PROJECT_DIR / "output/generalization_ligand_v14_clean" / "ligand"))
    ap.add_argument("--fixed-json-root",
                    default=str(_PROJECT_DIR / "output/fixbinding_v14" / "fixed"))
    ap.add_argument("--manifest", default=str(_PROJECT_DIR / "data/validation_pdbs/validation_manifest_v14_in.json"))
    ap.add_argument("--pdb-dir", default=str(_PROJECT_DIR / "data/validation_pdbs"))
    ap.add_argument("--sasa-threshold", type=float, default=0.25)
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--h3-fix", default=str(_PROJECT_DIR / "output/h3_ligand_v14_fixbinding.json"),
                    help="fix 的 H3 JSON（h3_charge_legality 产物）")
    ap.add_argument("--h3-unfix", default=str(_PROJECT_DIR / "output/h3_ligand_v14_clean.json"))
    ap.add_argument("--out", default=str(_PROJECT_DIR / "output/v14_fixbinding_summary.json"))
    args = ap.parse_args()

    fix_root, unfix_root = Path(args.fix_root), Path(args.unfix_root)
    man = json.load(open(args.manifest))
    items = man["items"]

    # H3 加载（若已生成）
    def load_h3(p):
        if Path(p).exists():
            return json.load(open(p))
        return None
    h3_fix, h3_unfix = load_h3(args.h3_fix), load_h3(args.h3_unfix)

    results = {"proteins": {}}
    print(f"{'蛋白':6s} {'L':>4s} {'fix%':>5s} {'臂':>6s} | {'unfix mean/dev':>13s} "
          f"{'fix mean/dev':>13s} | {'native DK':>7s} {'unfix总×':>6s} {'fix总×':>6s} "
          f"{'unfix 口袋×':>8s} {'fix 口袋×':>7s} {'unfix表×':>6s} {'fix表×':>5s} "
          f"{'unfix心×':>6s} {'fix心×':>5s}", flush=True)
    print("-" * 130)

    for it in items:
        pdb = it["pdb"]
        path = it["path"]
        protein_dict, *_ = parse_PDB(path)
        L = protein_dict["X"].shape[0]
        native_s = native_seq_from_pd(protein_dict)
        pocket = pocket_residues(protein_dict, cutoff=args.cutoff)
        if pocket is None:
            pocket = np.array([], dtype=int)

        # frac_sasa（晶体复合物结构）。⚠️ 不能直接用 align_to_full=True 的数组索引
        # parse_PDB 位置：1AS2 等有 occupancy=0 的 altloc 残基被 parse_PDB 丢弃但
        # freesasa 计入 → 数组错位。改按残基号（compact 口径 + resnum 映射）对齐 parse 索引。
        sasa = fractional_sasa(path, align_to_full=False)
        sasa_resid = list(sasa["residues"])
        sasa_frac = np.asarray(sasa["frac_sasa"], dtype=np.float64)
        ridx_parse = [int(r) for r in protein_dict["R_idx"].reshape(-1).cpu().numpy()]
        frac_map = {}
        for r, f in zip(sasa_resid, sasa_frac):
            frac_map.setdefault(int(r), float(f))
        frac = np.array([frac_map.get(r, 0.0) for r in ridx_parse], dtype=np.float64)
        if len(frac) != L:
            print(f"  !! {pdb}: frac len {len(frac)} != L {L}", flush=True)
        # 口袋/表面/核心分区
        nonpocket = np.setdiff1d(np.arange(L), pocket)
        surface = nonpocket[frac[nonpocket] >= args.sasa_threshold]
        core = nonpocket[frac[nonpocket] < args.sasa_threshold]
        zones = {"total": None, "pocket": pocket, "surface": surface, "core": core}

        # fix binding 元数据
        fix_json = Path(args.fixed_json_root) / f"{pdb}_fixed.json"
        n_fix = 0
        if fix_json.exists():
            n_fix = json.load(open(fix_json)).get("n_fixed", len(pocket))
        else:
            n_fix = len(pocket)

        u_sum = load_arm_summary(unfix_root / pdb / "validation.json")
        f_sum = load_arm_summary(fix_root / pdb / "validation.json")
        native_de, native_kr = count_de_kr(native_s)
        results["proteins"][pdb] = {
            "L": L, "n_fixed": int(n_fix),
            "frac_fixed": round(n_fix / L, 4),
            "native_dk": native_de + native_kr,
            "native_de": native_de, "native_kr": native_kr,
            "arms": {}}
        for arm in ARMS:
            # 组成 per arm
            row = {}
            for tag, root in (("unfix", unfix_root), ("fix", fix_root)):
                fa = root / pdb / f"pH7.4" / f"arm_{arm}" / "seqs.fa"
                gen = []
                if fa.exists():
                    gen, native_from_fa = read_seqfa(fa)
                if not gen:
                    row[tag] = None
                    continue
                cnt = {}
                for zname, idx in zones.items():
                    de = np.mean([count_de_kr(s, idx)[0] for s in gen])
                    kr = np.mean([count_de_kr(s, idx)[1] for s in gen])
                    cnt[zname] = {"de": float(de), "kr": float(kr),
                                  "dk": float(de + kr)}
                row[tag] = cnt
            # 比 native 倍率（DK 总数与分区）
            entry = {}
            for zname in ("total", "pocket", "surface", "core"):
                nat_de, nat_kr = count_de_kr(native_s, zones[zname])
                nat_dk = nat_de + nat_kr
                for tag in ("unfix", "fix"):
                    if row[tag] and row[tag][zname] is not None:
                        ratio = (row[tag][zname]["dk"] / nat_dk) if nat_dk > 0 else None
                    else:
                        ratio = None
                    entry[f"{tag}_{zname}_dk_ratio"] = round(ratio, 3) if ratio is not None else None
            for zname in ("total", "pocket", "surface", "core"):
                nat_de, nat_kr = count_de_kr(native_s, zones[zname])
                for tag in ("unfix", "fix"):
                    if row[tag] and row[tag][zname] is not None:
                        entry[f"{tag}_{zname}_de"] = row[tag][zname]["de"]
                        entry[f"{tag}_{zname}_kr"] = row[tag][zname]["kr"]
            # H2
            for tag, s in (("unfix", u_sum), ("fix", f_sum)):
                if s and arm in s["arms"]:
                    a = s["arms"][arm]
                    entry[f"{tag}_target"] = a["target"]
                    entry[f"{tag}_mean"] = a["mean_charge"]
                    entry[f"{tag}_dev"] = a["dev"]
                    entry[f"{tag}_hit"] = a["hit"]
            # H3 viol rate
            for tag, h3 in (("unfix", h3_unfix), ("fix", h3_fix)):
                if h3 and pdb in h3["proteins"] and arm in h3["proteins"][pdb]["arms"]:
                    entry[f"{tag}_h3_viol"] = h3["proteins"][pdb]["arms"][arm]["viol_rate"]
            results["proteins"][pdb]["arms"][arm] = entry

            # 打印
            u_c = row["unfix"]["total"] if row["unfix"] else None
            f_c = row["fix"]["total"] if row["fix"] else None
            nat_dk_total = native_de + native_kr
            ru = (u_c["dk"] / nat_dk_total) if u_c and nat_dk_total else None
            rf = (f_c["dk"] / nat_dk_total) if f_c and nat_dk_total else None
            u_p = row["unfix"]["pocket"] if row["unfix"] else None
            f_p = row["fix"]["pocket"] if row["fix"] else None
            nat_p_dk = sum(count_de_kr(native_s, pocket))
            rup = (u_p["dk"] / nat_p_dk) if u_p and nat_p_dk else None
            rfp = (f_p["dk"] / nat_p_dk) if f_p and nat_p_dk else None
            u_s = row["unfix"]["surface"] if row["unfix"] else None
            f_s = row["fix"]["surface"] if row["fix"] else None
            nat_s_dk = sum(count_de_kr(native_s, surface))
            rus = (u_s["dk"] / nat_s_dk) if u_s and nat_s_dk else None
            rfs = (f_s["dk"] / nat_s_dk) if f_s and nat_s_dk else None
            u_c2 = row["unfix"]["core"] if row["unfix"] else None
            f_c2 = row["fix"]["core"] if row["fix"] else None
            nat_c_dk = sum(count_de_kr(native_s, core))
            ruc = (u_c2["dk"] / nat_c_dk) if u_c2 and nat_c_dk else None
            rfc = (f_c2["dk"] / nat_c_dk) if f_c2 and nat_c_dk else None
            ua = u_sum["arms"].get(arm, {}) if u_sum else {}
            fa2 = f_sum["arms"].get(arm, {}) if f_sum else {}
            print(f"{pdb:6s} {L:4d} {n_fix / L:5.1%} {arm:6s} | "
                  f"{ua.get('mean_charge', float('nan')):>+7.2f}/{ua.get('dev', float('nan')):>5.2f} "
                  f"{fa2.get('mean_charge', float('nan')):>+7.2f}/{fa2.get('dev', float('nan')):>5.2f} | "
                  f"{native_de + native_kr:7d} "
                  f"{ru:6.2f} {rf:6.2f} "
                  f"{rup:8.2f} {rfp:7.2f} "
                  f"{rus:6.2f} {rfs:5.2f} "
                  f"{ruc:6.2f} {rfc:5.2f}", flush=True)
        print("-" * 130, flush=True)

    # 汇总
    tot = {"H2": {}, "native_arm_deletion": {}, "pocket_deletion": {}, "surface_deletion": {}, "core_deletion": {}}
    h2 = {"unfix": {"hit": 0, "tot": 0}, "fix": {"hit": 0, "tot": 0}}
    ratios = {"unfix": {"native_total": [], "native_pocket": [], "native_surface": [], "native_core": []},
              "fix": {"native_total": [], "native_pocket": [], "native_surface": [], "native_core": []}}
    for pdb, p in results["proteins"].items():
        for arm, e in p["arms"].items():
            for tag in ("unfix", "fix"):
                if e.get(f"{tag}_hit") is not None:
                    h2[tag]["tot"] += 1
                    h2[tag]["hit"] += int(e[f"{tag}_hit"])
        # native 臂
        e = p["arms"].get("native", {})
        for tag, key in (("unfix", "native_total"), ("fix", "native_total")):
            if e.get(f"{tag}_total_dk_ratio") is not None:
                ratios[tag][key].append(e[f"{tag}_total_dk_ratio"])
        for tag, key in (("unfix", "native_pocket"), ("fix", "native_pocket")):
            if e.get(f"{tag}_pocket_dk_ratio") is not None:
                ratios[tag][key].append(e[f"{tag}_pocket_dk_ratio"])
        for tag, key in (("unfix", "native_surface"), ("fix", "native_surface")):
            if e.get(f"{tag}_surface_dk_ratio") is not None:
                ratios[tag][key].append(e[f"{tag}_surface_dk_ratio"])
        for tag, key in (("unfix", "native_core"), ("fix", "native_core")):
            if e.get(f"{tag}_core_dk_ratio") is not None:
                ratios[tag][key].append(e[f"{tag}_core_dk_ratio"])

    def avg(xs):
        return round(float(np.mean(xs)), 3) if xs else None

    agg = {
        "H2": {tag: {"hit": h2[tag]["hit"], "tot": h2[tag]["tot"],
                     "rate": round(h2[tag]["hit"] / h2[tag]["tot"], 3) if h2[tag]["tot"] else None}
               for tag in ("unfix", "fix")},
        "native_arm_dk_ratio_mean": {tag: avg(ratios[tag]["native_total"]) for tag in ("unfix", "fix")},
        "native_pocket_dk_ratio_mean": {tag: avg(ratios[tag]["native_pocket"]) for tag in ("unfix", "fix")},
        "native_surface_dk_ratio_mean": {tag: avg(ratios[tag]["native_surface"]) for tag in ("unfix", "fix")},
        "native_core_dk_ratio_mean": {tag: avg(ratios[tag]["native_core"]) for tag in ("unfix", "fix")},
    }
    results["summary"] = agg
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n=== 汇总（native 臂 D/E+K/R 均值倍率；H2 全臂）===")
    print(f"H2  unfix {agg['H2']['unfix']['hit']}/{agg['H2']['unfix']['tot']} "
          f"({agg['H2']['unfix']['rate']}) | fix {agg['H2']['fix']['hit']}/{agg['H2']['fix']['tot']} "
          f"({agg['H2']['fix']['rate']})")
    print(f"DK 总×    unfix={agg['native_arm_dk_ratio_mean']['unfix']} fix={agg['native_arm_dk_ratio_mean']['fix']}")
    print(f"口袋×      unfix={agg['native_pocket_dk_ratio_mean']['unfix']} fix={agg['native_pocket_dk_ratio_mean']['fix']}")
    print(f"表面×      unfix={agg['native_surface_dk_ratio_mean']['unfix']} fix={agg['native_surface_dk_ratio_mean']['fix']}")
    print(f"核心×      unfix={agg['native_core_dk_ratio_mean']['unfix']} fix={agg['native_core_dk_ratio_mean']['fix']}")
    print(f"已写 {args.out}")


if __name__ == "__main__":
    main()
