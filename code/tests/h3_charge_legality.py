"""H3 电荷聚集合法性：structure_aware_filter 4 规则事后统计（不干预解码）。

对应 DESIGN_CRITERIA H3：条件臂序列在结构过滤器规则下违规率 ≤ 基线 + 5pp
（证明电荷条件化不产生物理不可能的电荷布局）。

方法：
- 复用 src.structure_aware_filter 的 4 条规则（charge_cluster / salt_bridge /
  core_charge / same_sign_cluster）的**逻辑**，但改为**全量事后统计**：
  compute_bias 只统计"未解码可抑制位置"（seq_int==20），对完整解码序列无意义，
  故本脚本独立实现"所有残基位置是否处于违规布局"的统计。
- 坐标 = 采样骨干 ref PDB 的 Cα（与生成序列长度 L 对齐）。
- 带电集合 = pH_adaptive_charged_aa(pH)（pH 7.4 → 强电荷 K/R/D/E）。
- 逐规则统计"违规位置数"，4 规则并集去重 → 违规率 = 并集位置数 / L。
- 基线① native_ref（同骨架 native 序列）；基线② 无条件基线（net_charge=训练均值）。
- 判据：条件臂违规率 ≤ max(native_ref, 无条件) + 0.05。

用法（项目根，PYTHONPATH=code）：
  # mompnn 线
  python code/tests/h3_charge_legality.py --gen-root output/generalization_v12_2_calib_small/protein \
      --ref-root output/generalization_v12_2_calib_small/ref \
      --native-root output/tm_sol_v12_2/ref_native --uncond-root output/tm_sol_v12_2/uncond \
      --pH 7.4 --out output/h3_protein.json
  # ligand 线
  python code/tests/h3_charge_legality.py --gen-root output/generalization_ligand_v12_2/ligand \
      --ref-root output/generalization_ligand_v12_2/ref \
      --native-root output/tm_sol_ligand_v12_2/ref_native --uncond-root output/tm_sol_ligand_v12_2/uncond \
      --pH 7.4 --out output/h3_ligand.json

输入目录结构（双线一致）：<gen-root>/<PDB>/pH7.4/arm_<arm>/seqs.fa
  seqs.fa：n 条生成（>seed_... 头）+ 末尾 1 条 native 参考行（跳过）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import sys

_PROJECT_DIR = Path(__file__).resolve().parents[1]
_CODE_DIR = _PROJECT_DIR / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from src.structure_aware_filter import default_config, pH_adaptive_charged_aa  # noqa: E402
from src.pka import AA_TO_IDX  # noqa: E402
from data_utils import parse_PDB  # noqa: E402

ARMS = ["native", "n2", "p2", "n8", "p8"]
PDBS = ["1C6O", "1AZM", "1AS2", "1AXW", "2FEO", "5CQH", "1CGE", "1AG0", "1A65", "1BJ4"]


def read_seqs(fa_path, skip_native=True):
    """读 fasta → [seq,...]；skip_native 跳过 name 非 seed_ 开头的参考行。"""
    seqs, name = [], None
    for line in open(fa_path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
        elif line:
            if skip_native and name and not name.startswith("seed_"):
                continue
            seqs.append(line)
    return seqs


def seq_to_int(seq):
    return np.array([AA_TO_IDX[a] for a in seq], dtype=np.int64)


def count_violations(coords, seq_int, pos_aa, neg_aa, cfg):
    """4 规则全量事后统计：返回 (违规位置数, 并集违规 mask, 各规则计数 dict)。

    coords: [L, 3] Cα；seq_int: [L] int。所有位置全解码，统计"处于违规布局"的位置。
    """
    L = coords.shape[0]
    pos = np.zeros(L, dtype=bool)
    neg = np.zeros(L, dtype=bool)
    for a in pos_aa:
        pos |= (seq_int == AA_TO_IDX[a])
    for a in neg_aa:
        neg |= (seq_int == AA_TO_IDX[a])
    charged = pos | neg

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # [L, L]

    viol = np.zeros(L, dtype=bool)
    per_rule = {}

    # ---- R1 charge_cluster：10Å 内同号 ≥ threshold → 该带电残基违规 ----
    c = cfg["charge_cluster"]
    nb10 = dist <= c["radius"]
    pos_count = (nb10 & pos[None, :]).sum(axis=1)
    neg_count = (nb10 & neg[None, :]).sum(axis=1)
    r1 = (pos & (pos_count >= c["threshold"])) | (neg & (neg_count >= c["threshold"]))
    viol |= r1
    per_rule["charge_cluster"] = int(r1.sum())

    # ---- R2 salt_bridge：10Å 内 min(正,负) ≥ threshold → 带电残基违规 ----
    c = cfg["salt_bridge"]
    pairs = np.minimum(pos_count, neg_count)
    r2 = charged & (pairs >= c["threshold"])
    viol |= r2
    per_rule["salt_bridge"] = int(r2.sum())

    # ---- R3 core_charge：埋藏高 + 8Å 内带电 ≥ threshold → 该带电残基违规 ----
    c = cfg["core_charge"]
    nb_bur = dist <= c["burial_radius"]
    burial = nb_bur.sum(axis=1)
    burial_ratio = burial / burial.max() if burial.max() > 0 else burial
    nb_chg = dist <= c["charge_radius"]
    chg8 = (nb_chg & charged[None, :]).sum(axis=1)
    r3 = charged & (burial_ratio > c["burial_threshold"]) & (chg8 >= c["charge_count"])
    viol |= r3
    per_rule["core_charge"] = int(r3.sum())

    # ---- R4 same_sign_cluster：8Å 连通分量内同号 ≥ threshold → 分量内带电残基违规 ----
    c = cfg["same_sign_cluster"]
    adj = csr_matrix((dist <= c["radius"]).astype(int))
    n_comp, labels = connected_components(adj, directed=False)
    r4 = np.zeros(L, dtype=bool)
    for comp in range(n_comp):
        members = labels == comp
        n_pos = int((members & pos).sum())
        n_neg = int((members & neg).sum())
        if n_pos >= c["threshold"]:
            r4 |= members & pos
        if n_neg >= c["threshold"]:
            r4 |= members & neg
    viol |= r4
    per_rule["same_sign_cluster"] = int(r4.sum())

    return int(viol.sum()), viol, per_rule


def ca_coords_from_pdb(pdb_path):
    """ref PDB → Cα 坐标 [L, 3]（LigandMPNN parse_PDB 的 X[:, 1, :]）。"""
    protein_dict, *_ = parse_PDB(str(pdb_path))
    X = np.asarray(protein_dict["X"])  # [L, 4, 3]（N/Cα/C/O）
    return X[:, 1, :].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", required=True, help="泛化生成根（protein 或 ligand）")
    ap.add_argument("--ref-root", required=True, help="ref 骨架根（<pdb>_ref.pdb）")
    ap.add_argument("--native-root", required=True, help="native_ref fasta 目录")
    ap.add_argument("--uncond-root", required=True, help="无条件基线 fasta 目录")
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--out", default="output/h3.json")
    args = ap.parse_args()

    gen_root, ref_root = Path(args.gen_root), Path(args.ref_root)
    native_root, uncond_root = Path(args.native_root), Path(args.uncond_root)
    pos_aa, neg_aa = pH_adaptive_charged_aa(args.pH)
    cfg = default_config()
    print(f"H3 事后统计 | pH={args.pH} 带电集合=({','.join(pos_aa)}),({','.join(neg_aa)})")
    print(f"gen={gen_root}\nref={ref_root}\nuncond={uncond_root}")
    print(f"{'蛋白':6s} {'臂':8s} {'违规率':>8s} {'R1':>4s} {'R2':>4s} {'R3':>4s} {'R4':>4s} | {'native':>7s} {'uncond':>7s} {'基线+5pp':>9s} {'PASS':>5s}")
    print("-" * 88)

    result = {"pH": args.pH, "charged_aa": {"pos": list(pos_aa), "neg": list(neg_aa)},
              "proteins": {}}
    total_pass = total_arms = 0
    for pdb in PDBS:
        ref_pdb = ref_root / f"{pdb}_ref.pdb"
        coords = ca_coords_from_pdb(ref_pdb)
        L = coords.shape[0]

        # 基线① native_ref（同骨架 native 序列）
        native_fa = native_root / f"{pdb}_native.fa"
        native_viol = None
        if native_fa.exists():
            nseq = read_seqs(native_fa, skip_native=False)[0]
            assert len(nseq) == L, f"{pdb} native L={len(nseq)} != ref L={L}"
            nv, _, _ = count_violations(coords, seq_to_int(nseq), pos_aa, neg_aa, cfg)
            native_viol = nv / L

        # 基线② 无条件基线（同模型同骨架，无电荷条件；多序列取均值）
        u_fa = uncond_root / pdb / "seqs.fa"
        uncond_viol = None
        if u_fa.exists():
            uvs = []
            for s in read_seqs(u_fa):
                if len(s) != L:
                    continue
                v, _, _ = count_violations(coords, seq_to_int(s), pos_aa, neg_aa, cfg)
                uvs.append(v / L)
            uncond_viol = float(np.mean(uvs)) if uvs else None

        base = max([x for x in (native_viol, uncond_viol) if x is not None], default=None)
        per = {"L": L, "native_ref": native_viol, "uncond": uncond_viol, "arms": {}}

        for arm in ARMS:
            fa = gen_root / pdb / f"pH{args.pH}" / f"arm_{arm}" / "seqs.fa"
            if not fa.exists():
                print(f"{pdb:6s} {arm:8s} 缺失 {fa}")
                continue
            viols, per_rules = [], []
            n_seq = 0
            for s in read_seqs(fa):
                if len(s) != L:
                    continue
                n_seq += 1
                v, _, pr = count_violations(coords, seq_to_int(s), pos_aa, neg_aa, cfg)
                viols.append(v / L)
                per_rules.append(pr)
            rate = float(np.mean(viols)) if viols else None
            rule_means = {}
            for k in ["charge_cluster", "salt_bridge", "core_charge", "same_sign_cluster"]:
                vs = [pr[k] for pr in per_rules]
                rule_means[k] = round(float(np.mean(vs)), 2) if vs else None
            thresh = round(base + 0.05, 4) if base is not None else None
            passed = (rate <= thresh) if (rate is not None and thresh is not None) else None
            per["arms"][arm] = {
                "viol_rate": round(rate, 4) if rate is not None else None,
                "n_seq": n_seq, "rules": rule_means,
                "baseline": round(base, 4) if base is not None else None,
                "baseline_thresh": thresh, "pass": passed,
            }
            if passed is True:
                total_pass += 1
            total_arms += 1
            tstr = "✅" if passed is True else ("❌" if passed is False else " -")
            print(f"{pdb:6s} {arm:8s} {rate if rate is not None else '-':>8.4f} "
                  f"{rule_means['charge_cluster'] if rule_means['charge_cluster'] is not None else '-':>4} "
                  f"{rule_means['salt_bridge'] if rule_means['salt_bridge'] is not None else '-':>4} "
                  f"{rule_means['core_charge'] if rule_means['core_charge'] is not None else '-':>4} "
                  f"{rule_means['same_sign_cluster'] if rule_means['same_sign_cluster'] is not None else '-':>4} | "
                  f"{native_viol if native_viol is not None else '-':>7.4f} "
                  f"{uncond_viol if uncond_viol is not None else '-':>7.4f} "
                  f"{thresh if thresh is not None else '-':>9.4f} {tstr:>5}")

        result["proteins"][pdb] = per

    result["pass"] = {"n": total_pass, "total": total_arms,
                      "rate": round(total_pass / total_arms, 3) if total_arms else None}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("-" * 88)
    print(f"\n通过 {total_pass}/{total_arms}（{round(total_pass / max(total_arms, 1) * 100)}%）")
    print(f"已写 {args.out}")


if __name__ == "__main__":
    main()
