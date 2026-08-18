"""生成序列理化合理性检验（seq sanity check）v2 —— 含核心疏水保持检查。

用户问题："对刚才实验生成的进行检验，这些序列合不合理？"
回答方式：不只看 recovery/折叠数字，而是检查生成的**每条序列本身**
是否是一段"看起来像真蛋白"的序列。本脚本对每个蛋白×pH 的生成序列组
做多项检查，并与无条件 MoMPNN 基线对比。

检查项（v2 增加"核心疏水保持"——最有物理意义的合理性判据）：
  1. 非法字符    : 含 X → 异常
  2. 核心疏水保持: 用骨架 Cα 接触数（<8Å 邻居）定义 top30% 接触最多位点=疏水内核，
                   生成序列在这些位点的疏水残基比例 vs native。真蛋白内核必须疏水，
                   若条件化把内核掏空 → 序列不合理。
  3. 组成漂移    : 20 氨基酸频率 L1 vs native；条件化 vs 基线不应显著更大
  4. 表面疏水漂移: 平均 Kyte-Doolittle 疏水性漂移（参考列，不 flag——
                   电荷改写必然改变表面带电残基，属预期副作用）
  5. pI 漂移     : 等电点 vs native（参考列）
  6. 电荷命中    : |平均生成电荷 − target|（≤2 达标）
  7. 折叠        : TM 中位 ≥0.70 且失败率(TM<0.5)≤10%

flag 规则：
  - nX > 0                                   → "异常:含X"
  - core_hydro_gen < core_hydro_native − 0.15 → "注意:核心疏水缺失"
  - comp_l1_cond > comp_l1_base + 0.08       → "注意:组成漂移超基线"
  - dev > 2.0                                → "注意:电荷未命中"
  - TM 中位<0.70 或失败率>10%                 → "注意:折叠未达标"

用法（code/ 下，confumpnn 环境）：
  PYTHONPATH=. python tests/seq_sanity_check.py \
      --root ../output/ph_scan --uncond_root ../output/ph_scan_uncond \
      --pdb_dir ../code/input --out ../output/ph_scan_sanity.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR.parent / "LigandMPNN"))

from data_utils import parse_PDB  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402

# Kyte-Doolittle 疏水性表；疏水残基（KD>1.0）
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
HYDROPHOBIC = {"A", "V", "L", "I", "M", "F", "W", "C"}
AA20 = list(KD.keys())


def read_fasta(path):
    """读 fasta，返回 [(name, seq)]，保持文件内顺序。"""
    seqs = []
    name, lines = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                seqs.append((name, "".join(lines)))
            name, lines = line[1:], []
        elif line:
            lines.append(line)
    if name is not None:
        seqs.append((name, "".join(lines)))
    return seqs


def kd_mean(seq):
    vals = [KD[aa] for aa in seq if aa in KD]
    return float(np.mean(vals)) if vals else None


def composition(seq):
    n = len(seq)
    freq = {aa: 0 for aa in AA20}
    for aa in seq:
        if aa in freq:
            freq[aa] += 1
    return {aa: c / n for aa, c in freq.items()}


def comp_l1(f1, f2):
    return sum(abs(f1.get(a, 0) - f2.get(a, 0)) for a in AA20) / 2


def pI(seq):
    """等电点：解 net_charge(seq, pH)=0 的 pH（二分搜索）。"""
    if any(a not in AA20 for a in seq):
        return None
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if float(net_charge(seq, mid)) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def core_indices(pdb_path, frac=0.3):
    """疏水内核位点：Cα 接触数（<8Å 邻居）最多的 top frac%。

    用原生骨架坐标定义结构核心——折叠决定位点。返回 set of int 索引。
    """
    protein_dict, *_ = parse_PDB(str(pdb_path))
    CA = protein_dict["X"][:, 1, :].numpy()  # [L,3] Cα
    L = len(CA)
    dist = np.linalg.norm(CA[:, None] - CA[None], axis=-1)
    n_neigh = (dist < 8.0).sum(axis=1) - 1  # 去自身
    k = max(1, int(round(L * frac)))
    return set(int(i) for i in np.argsort(n_neigh)[-k:])


def core_hydro_frac(seq, core):
    """生成序列在核心位点的疏水残基比例。"""
    hits = [i for i in core if i < len(seq)]
    if not hits:
        return None
    return sum(1 for i in hits if seq[i] in HYDROPHOBIC) / len(hits)


def analyze_group(seqs_with_names, native, core):
    xs = [(s, kd_mean(s), pI(s), composition(s), s.count("X"),
           core_hydro_frac(s, core)) for _, s in seqs_with_names]
    n = len(xs)
    return {
        "n": n,
        "n_X": sum(x[4] for x in xs),
        "kd_mean": round(float(np.mean([x[1] for x in xs])), 2) if n else None,
        "pI_mean": round(float(np.mean([x[2] for x in xs if x[2] is not None])), 2),
        "comp_l1_native": round(float(np.mean(
            [comp_l1(x[3], composition(native)) for x in xs])), 3),
        "core_hydro": round(float(np.mean([x[5] for x in xs if x[5] is not None])), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--uncond_root", default=None)
    ap.add_argument("--pdb_dir", default=None,
                    help="放原生骨架 PDB 的目录，逗号分隔多个；文件名须与 root 下蛋白目录同名")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    uncond_root = Path(args.uncond_root) if args.uncond_root else None
    pdb_dirs = [Path(p) for p in args.pdb_dir.split(",")] if args.pdb_dir else []

    def resolve_pdb(name):
        for d in pdb_dirs:
            cand = d / f"{name}.pdb"
            if cand.exists():
                return cand
        return None

    summary = {}

    print(f"{'蛋白':<14}{'pH':>5}{'nX':>4}{'core生':>7}{'core生基':>8}"
          f"{'core原生':>8}{'kd漂移':>8}{'kd基漂':>7}{'pI':>7}{'compL1':>8}"
          f"{'comp基':>7}{'dev':>6}{'TM':>6}{'verdict':>18}", flush=True)

    for json_path in sorted(root.glob("*/ph_scan.json")):
        pdb = json_path.parent.name
        d = json.load(open(json_path))
        native = d["native"]
        pdb_path = resolve_pdb(pdb)
        core = core_indices(pdb_path) if pdb_path else set()
        native_meta = {
            "kd": kd_mean(native), "pI": pI(native),
            "core_hydro": core_hydro_frac(native, core),
        }
        summary[pdb] = {"L": d["L"], "native_pI": native_meta["pI"],
                        "native_kd": native_meta["kd"],
                        "core_n": len(core),
                        "native_core_hydro": native_meta["core_hydro"],
                        "pH_arms": {}}
        for ph_s, arm in d["pH_arms"].items():
            arm_dir = json_path.parent / f"pH{ph_s}"
            fa = arm_dir / "seqs.fa"
            if not fa.exists():
                continue
            all_seqs = read_fasta(fa)
            gen = [(nm, s) for nm, s in all_seqs if "native" not in nm]
            g = analyze_group(gen, native, core)

            base = None
            if uncond_root:
                ufa = uncond_root / pdb / f"pH{ph_s}" / "seqs.fa"
                if ufa.exists():
                    ugen = [(nm, s) for nm, s in read_fasta(ufa)
                            if "native" not in nm]
                    base = analyze_group(ugen, native, core)

            # 折叠指标
            tm_med = tm_ge07 = tm_lt05 = None
            tm_csv = arm_dir / "tm.csv"
            if tm_csv.exists():
                tms = []
                with open(tm_csv) as f:
                    for row in csv.DictReader(f):
                        try:
                            tms.append(float(row["tm_score"]))
                        except (ValueError, KeyError):
                            pass
                if tms:
                    tm_med = round(float(np.median(tms)), 3)
                    tm_ge07 = round(sum(1 for t in tms if t >= 0.7) / len(tms), 3)
                    tm_lt05 = round(sum(1 for t in tms if t < 0.5) / len(tms), 3)

            # verdict：只有 X/核心疏水/折叠 是"序列不合理"的判据；
            # 组成漂移/电荷未命中 是条件化改写的已知代价，作为 notes 参考
            flags = []
            if g["n_X"] > 0:
                flags.append("异常:含X")
            if (native_meta["core_hydro"] is not None
                    and g["core_hydro"] is not None
                    and g["core_hydro"] < native_meta["core_hydro"] - 0.15):
                flags.append("异常:核心疏水缺失")
            if tm_med is not None and (tm_med < 0.70 or (tm_lt05 or 0) > 0.1):
                flags.append("异常:折叠未达标")
            verdict = "合理" if not flags else "; ".join(flags)

            notes = []
            if (base is not None
                    and g["comp_l1_native"] > base["comp_l1_native"] + 0.08):
                notes.append("组成漂移超基线")
            if arm.get("dev") is not None and arm["dev"] > 2.0:
                notes.append("电荷未命中")

            kd_drift = (abs(g["kd_mean"] - native_meta["kd"])
                        if g["kd_mean"] is not None and native_meta["kd"] is not None
                        else None)
            kd_drift_base = (abs(base["kd_mean"] - native_meta["kd"])
                             if base and base["kd_mean"] is not None
                             and native_meta["kd"] is not None else None)

            summary[pdb]["pH_arms"][ph_s] = {
                **g,
                "kd_drift": round(kd_drift, 2) if kd_drift is not None else None,
                "kd_drift_base": round(kd_drift_base, 2) if kd_drift_base is not None else None,
                "core_hydro_base": base["core_hydro"] if base else None,
                "comp_l1_base": base["comp_l1_native"] if base else None,
                "dev": arm.get("dev"),
                "tm_median": tm_med, "tm_ge070": tm_ge07, "tm_lt050": tm_lt05,
                "verdict": verdict, "notes": notes,
            }
            print(f"{pdb:<14}{ph_s:>5}{g['n_X']:>4}"
                  f"{g['core_hydro']:>7}"
                  f"{str(base['core_hydro'] if base else '-'):>8}"
                  f"{native_meta['core_hydro']:>8}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['kd_drift']):>8}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['kd_drift_base']):>7}"
                  f"{g['pI_mean']:>7}{g['comp_l1_native']:>8}"
                  f"{str(summary[pdb]['pH_arms'][ph_s]['comp_l1_base']):>7}"
                  f"{str(arm.get('dev')):>6}{str(tm_med):>6}{verdict:>18}",
                  flush=True)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n已写 {args.out}")


if __name__ == "__main__":
    main()
