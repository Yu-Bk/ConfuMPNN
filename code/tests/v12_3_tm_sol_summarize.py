"""v12.2 Tm/Sol 汇总：提取 TemBERTure Tm + Protein-Sol %sol，按蛋白×臂统计。

输入：
- Tm：output/tm_sol_v12_3/seqs/<PDB>/arm_*/seqs.fa.tm.csv（temberture_score.py 产物）
  （seqs.fa 是符号链接 → tm.csv 落在泛化源目录，故需从泛化目录读）
- Sol：泛化目录 protein/<PDB>/pH7.4/arm_*/seqs.fa-protein_sol_prediction.txt
        + ref_native/*_native.fa-protein_sol_prediction.txt
  每行 "SEQUENCE PREDICTIONS,>NAME,percent-sol,scaled-sol,population-sol,pI"

输出：
- stdout：逐蛋白×臂 Tm 均值 / Sol 均值 vs native_ref 基线
- output/tm_sol_v12_3/tm_sol_summary.json
- 判据 S2：各臂 vs native_ref 的 Tm Δ、%sol Δ（明显恶化=ΔTm<-5 或 Δ%sol<-10）
"""
import csv
import json
import re
from pathlib import Path

_PROJECT_DIR = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
GEN = _PROJECT_DIR / "output/generalization_v12_3_calib/protein"
TM_SEQS = _PROJECT_DIR / "output/tm_sol_v12_3/seqs"
REF_NATIVE = _PROJECT_DIR / "output/tm_sol_v12_3/ref_native"
UNCOND = _PROJECT_DIR / "output/tm_sol_v12_3/uncond"
OUT_JSON = _PROJECT_DIR / "output/tm_sol_v12_3/tm_sol_summary.json"

ARMS = ["native", "n2", "p2", "n8", "p8"]
PDBS = ["1AZM", "1AS2", "2FEO", "5CQH", "1CGE", "1A65", "1BJ4", "13BB", "1CDG"]

SOL_RE = re.compile(r"SEQUENCE PREDICTIONS,>\S+?,([-\d.]+),")


def read_tm_csv(path):
    """返回 [(name, mean_tm), ...]"""
    rows = []
    if not path.exists():
        return rows
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) >= 3 and parts[0] != "name":
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


def main():
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
        ref_tm_rows = read_tm_csv(native_tm_csv)
        ref_tm = sum(r[1] for r in ref_tm_rows) / len(ref_tm_rows) if ref_tm_rows else None
        ref_sol = read_sol(native_sol_txt)
        per["native_ref"] = {"tm": ref_tm, "sol": ref_sol}

        # 无条件基线（同模型同 backbone，无电荷条件；Tm csv 在 UNCOND/<pdb>/seqs.fa.tm.csv）
        uncond_dir = UNCOND / pdb
        u_tm_csv = uncond_dir / "seqs.fa.tm.csv"
        u_sol_txt = uncond_dir / "seqs.fa-protein_sol_prediction.txt"
        u_tm_rows = read_tm_csv(u_tm_csv)
        u_tm = sum(r[1] for r in u_tm_rows) / len(u_tm_rows) if u_tm_rows else None
        u_sol = read_sol(u_sol_txt)
        per["uncond"] = {"tm": u_tm, "sol": u_sol}

        for arm in ARMS:
            tm_csv = tm_seqs_pdb / f"arm_{arm}" / "seqs.fa.tm.csv"
            sol_txt = GEN / pdb / "pH7.4" / f"arm_{arm}" / "seqs.fa-protein_sol_prediction.txt"
            tm_rows = read_tm_csv(tm_csv)
            tm = sum(r[1] for r in tm_rows) / len(tm_rows) if tm_rows else None
            sol = read_sol(sol_txt)
            # 双基线：vs native_ref（天然序列）和 vs uncond（逆折叠固有代价）
            per["arms"][arm] = {"tm": tm, "sol": sol,
                                "tm_delta": (round(tm - ref_tm, 2) if tm and ref_tm else None),
                                "sol_delta": (round(sol - ref_sol, 2) if sol and ref_sol else None),
                                "tm_delta_uncond": (round(tm - u_tm, 2) if tm and u_tm else None),
                                "sol_delta_uncond": (round(sol - u_sol, 2) if sol and u_sol else None)}
            tm_d = per["arms"][arm]["tm_delta_uncond"]
            sol_d = per["arms"][arm]["sol_delta_uncond"]
            print(f"{pdb:6s} {arm:8s} {tm if tm is not None else '-':>7} "
                  f"{tm_d if tm_d is not None else '-':>6} "
                  f"{sol if sol is not None else '-':>6} "
                  f"{sol_d if sol_d is not None else '-':>7}")

        summary["proteins"][pdb] = per

    # 汇总：判定明显恶化（S2）——相对无条件基线（电荷条件化的额外代价）
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
    print(f"\n明显恶化臂数：{bad}/45")
    # native_ref 对比也记录（供参考，非判据）
    print("\n== 参考：native_ref 基线 vs 无条件 ==")
    for pdb in PDBS:
        ref = summary["proteins"][pdb]["native_ref"]
        u = summary["proteins"][pdb]["uncond"]
        print(f"  {pdb:6s} native_ref Tm={ref['tm'] if ref['tm'] else '-':>7.2f} "
              f"uncond Tm={u['tm'] if u['tm'] else '-':>7.2f} | "
              f"native_ref %sol={ref['sol'] if ref['sol'] else '-':>7.2f} "
              f"uncond %sol={u['sol'] if u['sol'] else '-':>7.2f}")

    summary["s2_worse_arms"] = bad
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n已写 {OUT_JSON}")


if __name__ == "__main__":
    main()
