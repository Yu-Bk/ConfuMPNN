"""v12.2 配体模式 Tm/Sol 汇总（LigandMPNN 专用）。

与 MoMPNN 版 `code/tests/v12_2_tm_sol_summarize.py` **分开保存**，路径各自独立。
输入：
- Tm：output/tm_sol_ligand_v12_2/seqs/<PDB>/arm_*/seqs.fa.tm.csv（temberture_score.py 产物）
  （seqs.fa 是符号链接 → tm.csv 落在链接目录旁，与泛化源解耦）
- Sol：配体泛化目录 ligand/<PDB>/pH7.4/arm_*/seqs.fa-protein_sol_prediction.txt
        + ref_native/<PDB>_native.fa-protein_sol_prediction.txt
- 无条件基线：output/tm_sol_ligand_v12_2/uncond/<PDB>/seqs.fa（配体上下文、net_charge=训练均值）

⚠️ 与 mompnn 版差异：配体泛化 seqs.fa 末尾含 1 条 native 参考行，Tm 均值**排除**它
（只统计 name 以 "seed_" 开头的生成序列），避免 native 污染均值。

输出：
- stdout：逐蛋白×臂 Tm 均值 / Sol 均值 vs native_ref 与无条件基线
- output/tm_sol_ligand_v12_2/tm_sol_summary.json
- 判据 S2：各臂 vs 无条件基线的 Tm Δ、%sol Δ（明显恶化=ΔTm<-5 或 Δ%sol<-10）
"""
import argparse
import csv
import json
import re
from pathlib import Path

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
# 路径参数化（v12.2 与 v13 共用，--xxx-root 指定各自根）：
# 默认 = v12.2（向后兼容）。v13 示例：
#   --gen-root output/generalization_ligand_v13/ligand \
#   --tm-seqs-root output/tm_sol_ligand_v13/seqs \
#   --ref-native-root output/tm_sol_ligand_v13/ref_native \
#   --uncond-root output/tm_sol_ligand_v13/uncond --out output/tm_sol_ligand_v13/tm_sol_summary.json
GEN = _PROJECT_DIR / "output/generalization_ligand_v12_2/ligand"
TM_SEQS = _PROJECT_DIR / "output/tm_sol_ligand_v12_2/seqs"
REF_NATIVE = _PROJECT_DIR / "output/tm_sol_ligand_v12_2/ref_native"
UNCOND = _PROJECT_DIR / "output/tm_sol_ligand_v12_2/uncond"
OUT_JSON = _PROJECT_DIR / "output/tm_sol_ligand_v12_2/tm_sol_summary.json"

ARMS = ["native", "n2", "p2", "n8", "p8"]
PDBS = ["1C6O", "1AZM", "1AS2", "1AXW", "2FEO", "5CQH", "1CGE", "1AG0", "1A65", "1BJ4"]

SOL_RE = re.compile(r"SEQUENCE PREDICTIONS,>\S+?,([-\d.]+),")


def read_tm_csv(path, seed_only=True):
    """返回 [(name, mean_tm), ...]；seed_only=True 只统计 seed_ 开头（排除 native 参考行）。

    native_ref 的 csv 行名是 "<PDB>_native L=<L>"（非 seed_ 开头），须 seed_only=False 全读。
    """
    rows = []
    if not path.exists():
        return rows
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) >= 3 and parts[0] != "name" and (not seed_only or parts[0].startswith("seed_")):
            try:
                rows.append((parts[0], float(parts[2])))
            except ValueError:
                pass
    return rows


def read_sol(path):
    """返回 percent-sol 均值（生成序列 30 条取均值，native 单条）"""
    vals = []
    if not path.exists():
        return None
    for line in open(path):
        m = SOL_RE.search(line)
        if m:
            vals.append(float(m.group(1)))
    return sum(vals) / len(vals) if vals else None


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def main():
    global GEN, TM_SEQS, REF_NATIVE, UNCOND, OUT_JSON
    ap = argparse.ArgumentParser(description="配体 Tm/Sol 汇总（v12.2/v13 共用，路径参数化）")
    ap.add_argument("--gen-root", default=str(GEN), help="泛化验证根（ligand/ 层）")
    ap.add_argument("--tm-seqs-root", default=str(TM_SEQS), help="Tm seqs 根（tm_sol_*_vNN/seqs）")
    ap.add_argument("--ref-native-root", default=str(REF_NATIVE), help="native_ref fasta 目录")
    ap.add_argument("--uncond-root", default=str(UNCOND), help="无条件基线 fasta 目录")
    ap.add_argument("--out", default=str(OUT_JSON), help="汇总 JSON 输出路径")
    args = ap.parse_args()
    GEN = Path(args.gen_root); TM_SEQS = Path(args.tm_seqs_root)
    REF_NATIVE = Path(args.ref_native_root); UNCOND = Path(args.uncond_root)
    OUT_JSON = Path(args.out)

    summary = {"proteins": {}}
    print(f"{'蛋白':6s} {'臂':8s} {'Tm':>7s} {'TmΔu':>6s} {'%sol':>6s} {'solΔu':>7s}")
    print("  （Δu = vs 无条件基线）")
    print("-" * 50)

    for pdb in PDBS:
        per = {"arms": {}, "native_ref": {}}
        # native 参考基线（同序列空间；Tm csv 在 TM_SEQS/<pdb>/arm_native_ref/，Sol txt 在 ref_native/）
        native_fa = REF_NATIVE / f"{pdb}_native.fa"
        tm_seqs_pdb = TM_SEQS / pdb
        native_tm_csv = tm_seqs_pdb / "arm_native_ref" / "seqs.fa.tm.csv"
        native_sol_txt = native_fa.with_name(native_fa.name + "-protein_sol_prediction.txt")
        ref_tm = mean([r[1] for r in read_tm_csv(native_tm_csv, seed_only=False)])
        ref_sol = read_sol(native_sol_txt)
        per["native_ref"] = {"tm": ref_tm, "sol": ref_sol}

        # 无条件基线（同模型同 backbone，配体上下文，无电荷条件）
        u_dir = UNCOND / pdb
        u_tm = mean([r[1] for r in read_tm_csv(u_dir / "seqs.fa.tm.csv")])
        u_sol = read_sol(u_dir / "seqs.fa-protein_sol_prediction.txt")
        per["uncond"] = {"tm": u_tm, "sol": u_sol}

        for arm in ARMS:
            tm = mean([r[1] for r in read_tm_csv(tm_seqs_pdb / f"arm_{arm}" / "seqs.fa.tm.csv")])
            sol = read_sol(GEN / pdb / "pH7.4" / f"arm_{arm}" / "seqs.fa-protein_sol_prediction.txt")
            per["arms"][arm] = {
                "tm": tm, "sol": sol,
                "tm_delta": (round(tm - ref_tm, 2) if tm and ref_tm else None),
                "sol_delta": (round(sol - ref_sol, 2) if sol and ref_sol else None),
                "tm_delta_uncond": (round(tm - u_tm, 2) if tm and u_tm else None),
                "sol_delta_uncond": (round(sol - u_sol, 2) if sol and u_sol else None),
            }
            tm_d = per["arms"][arm]["tm_delta_uncond"]
            sol_d = per["arms"][arm]["sol_delta_uncond"]
            print(f"{pdb:6s} {arm:8s} {tm if tm is not None else '-':>7} "
                  f"{tm_d if tm_d is not None else '-':>6} "
                  f"{sol if sol is not None else '-':>6} "
                  f"{sol_d if sol_d is not None else '-':>7}")

        summary["proteins"][pdb] = per

    print("\n== S2 判据（vs 无条件基线）：明显恶化 = ΔTm < -5 或 Δ%sol < -10 ==")
    bad = 0
    for pdb in PDBS:
        u_tm = summary["proteins"][pdb]["uncond"]["tm"]
        u_sol = summary["proteins"][pdb]["uncond"]["sol"]
        for arm in ARMS:
            a = summary["proteins"][pdb]["arms"][arm]
            if a["tm_delta_uncond"] is not None and a["tm_delta_uncond"] < -5:
                print(f"  ⚠️ {pdb}/{arm}: Tm Δu={a['tm_delta_uncond']}（无条件 {u_tm:.1f}）")
                bad += 1
            if a["sol_delta_uncond"] is not None and a["sol_delta_uncond"] < -10:
                print(f"  ⚠️ {pdb}/{arm}: %sol Δu={a['sol_delta_uncond']}（无条件 {u_sol:.1f}）")
                bad += 1
    print(f"\n明显恶化臂数：{bad}/50")
    print("\n== 参考：native_ref 基线 vs 无条件 ==")
    def f_or_dash(v):
        return f"{v:7.2f}" if isinstance(v, (int, float)) else "     -"
    for pdb in PDBS:
        ref = summary["proteins"][pdb]["native_ref"]
        u = summary["proteins"][pdb]["uncond"]
        print(f"  {pdb:6s} native_ref Tm={f_or_dash(ref['tm'])} "
              f"uncond Tm={f_or_dash(u['tm'])} | "
              f"native_ref %sol={f_or_dash(ref['sol'])} "
              f"uncond %sol={f_or_dash(u['sol'])}")

    summary["s2_worse_arms"] = bad
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n已写 {OUT_JSON}")


if __name__ == "__main__":
    main()
