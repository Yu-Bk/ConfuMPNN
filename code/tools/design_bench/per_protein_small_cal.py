#!/usr/bin/env python
"""per-protein 现场小样本校准：对单个表外蛋白采探针批 → 拟合自身 target→电荷 slope。
用法示例（仓库根）:
  python code/tools/design_bench/per_protein_small_cal.py \\
    --enc output/finetune_v12_2/finetune_epoch030.pt \\
    --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \\
    --pdb code/input/1BC8.pdb --pH 7.4 --native_q 0.0 \\
    --n_per 10 --out output/cal_1BC8_small.json
原理：native±[8,4,0,4,8] 五档，各采 n_per 条（沿用既往 build_calibration_small 协议，勿加大 n_per 到 20），
线性拟合 (target, mean 电荷)。输出 json 的 per_protein.<pname> 供 run_guided --calibrate auto 使用。
纯新增工具：不改 run_guided/train；只做编排。"""
import argparse, json, re, subprocess, sys, tempfile, os
from pathlib import Path

OFFSETS = [-8, -4, 0, 4, 8]

def parse_charges(fa_dir):
    """从 run_guided 输出 dir 的 seqs.fa 读每序列 charge（header 'charge=+x'）"""
    fa = Path(fa_dir) / "seqs.fa"
    vals = []
    if not fa.exists():
        # 备选：summary.json
        sj = Path(fa_dir) / "summary.json"
        if sj.exists():
            d = json.load(open(sj))
            for s in d.get("seqs", []):
                if "charge" in s:
                    vals.append(float(s["charge"]))
        return vals
    for line in open(fa):
        m = re.search(r"charge=([-+]?[0-9.]+)", line)
        if m:
            vals.append(float(m.group(1)))
    return vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--pdb", required=True); ap.add_argument("--pH", type=float, required=True)
    ap.add_argument("--native_q", type=float, required=True, help="native 序列在该 pH 的净电荷（HH，用户算好传入，如 L11 pH7.4=+6.87）")
    ap.add_argument("--n_per", type=int, default=10)
    ap.add_argument("--fixed_residues", default=None, help='如 "I3 I5 ..."')
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_ligand_atoms", type=int, default=25)
    ap.add_argument("--pname", default=None, help="per_protein 键名（默认 pdb 文件名去扩展）")
    ap.add_argument("--run_guided", default="code/run_guided.py")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry", action="store_true", help="只打印将运行的命令")
    a = ap.parse_args()
    pname = a.pname or Path(a.pdb).stem
    xs, ys = [], []
    with tempfile.TemporaryDirectory() as td:
        for off in OFFSETS:
            tgt = round(a.native_q + off)
            outdir = Path(td) / f"off{off}"
            cmd = [a.python, a.run_guided, "--pdb", a.pdb, "--pH", str(a.pH),
                   "--target_charge", str(tgt), "--cond_encoder", a.enc,
                   "--weights", a.weights, "--num_samples", str(a.n_per),
                   "--seed", str(a.seed), "--num_ligand_atoms", str(a.num_ligand_atoms),
                   "--calibrate", "off", "--out_dir", str(outdir)]
            if a.fixed_residues:
                cmd += ["--fixed_residues", a.fixed_residues]
            if a.dry:
                print(" ".join(cmd)); continue
            subprocess.run(cmd, check=True, capture_output=True)
            ch = parse_charges(outdir)
            if not ch:
                print(f"!! off{off} 无电荷解析，跳过"); continue
            xs.append(float(tgt)); ys.append(float(sum(ch) / len(ch)))
            print(f"  off{off:+d} target{tgt:+d} -> mean {ys[-1]:+.2f} (n={len(ch)})")
    if len(xs) < 3:
        raise SystemExit("探针档不足(<3)，无法拟合")
    # 线性拟合 y=a*x+b → 反函数 target_eff=(t-b)/a
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    a_ = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs) if n > 1 else 1.0
    b_ = my - a_ * mx
    slope, intercept = a_, b_
    res = {"label": "L11_smallcal", "global": {"slope": 1.0, "intercept": 0.0},
           "per_protein": {pname: {"slope": round(slope, 4), "intercept": round(intercept, 4),
                                    "n_calib": n * a.n_per, "native_q": a.native_q, "pH": a.pH}}}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"slope={slope:.4f} intercept={intercept:.4f} 已写 {a.out}")
    print(f"用法: run_guided --calibrate auto --calibration_file {a.out} （表内蛋白用此 per-protein，表外回退 global）")

if __name__ == "__main__":
    main()
