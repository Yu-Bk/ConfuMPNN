#!/usr/bin/env python
"""auto-calib run_guided：把"现场小样本标定"变成一次调用的一步开关（opt-in，不影响模型/现有流程）。
用法示例（仓库根；--autofit 时若缓存无该蛋白会先自动采探针批拟 slope 再正式采样）：
  python code/tools/design_bench/auto_calib_guided.py --autofit \\
    --enc output/finetune_v12_2/finetune_epoch030.pt \\
    --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \\
    --pdb code/input/1BC8.pdb --pH 7.4 --target_charge 0.0 --native_q 0.0 --num_samples 300
  # 配体模式：--weights LigandMPNN/... --num_ligand_atoms 25 --pdb 含配体
设计：①查缓存 per_protein.<pdb>；无且 --autofit → 调 per_protein_small_cal 补表；②以
--calibrate auto --calibration_file 缓存 调用 run_guided.py 做正式采样。默认(无 --autofit 且无缓存)则
等价于 global/不标定并提示。纯新增工具：不改 run_guided/train 任何采样逻辑。"""
import argparse, json, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--autofit", action="store_true", help="缓存无此蛋白时先探针拟合再采样")
    ap.add_argument("--enc", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--pdb", required=True); ap.add_argument("--pH", type=float, required=True)
    ap.add_argument("--target_charge", type=float, required=True)
    ap.add_argument("--native_q", type=float, required=True, help="native 在该 pH 的净电荷（探针拟合用）")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--n_per", type=int, default=10, help="探针每档条数（勿>20）")
    ap.add_argument("--fixed_residues", default=None)
    ap.add_argument("--num_ligand_atoms", type=int, default=None)
    ap.add_argument("--calib_cache", default=None, help="校准缓存 json（默认 output/charge_calibration_<pdb>.small.json）")
    ap.add_argument("--seed", type=int, default=42)
    args, extra = ap.parse_known_args()
    pname = Path(args.pdb).stem
    cache = args.calib_cache or f"output/charge_calibration_{pname}.small.json"
    cache_p = Path(cache)

    def have():
        if not cache_p.exists():
            return False
        d = json.load(open(cache_p))
        return pname in d.get("per_protein", {})

    if not have():
        if not args.autofit:
            print(f"[auto_calib] 无缓存 {cache}（且未 --autofit）→ 不启用小样本标定，按 run_guided 默认(global/off)执行。")
        else:
            print(f"[auto_calib] 缓存无 {pname} → 探针拟合…")
            subprocess.run([sys.executable, "code/tools/design_bench/per_protein_small_cal.py",
                            "--enc", args.enc, "--weights", args.weights, "--pdb", args.pdb,
                            "--pH", str(args.pH), "--native_q", str(args.native_q),
                            "--n_per", str(args.n_per), "--out", cache,
                            "--pname", pname, "--seed", str(args.seed)],
                           check=True)
    # 正式采样（auto：表内 per-protein，表外 global）
    cmd = [sys.executable, "code/run_guided.py", "--pdb", args.pdb, "--pH", str(args.pH),
           "--target_charge", str(args.target_charge), "--cond_encoder", args.enc,
           "--weights", args.weights, "--num_samples", str(args.num_samples),
           "--calibrate", "auto", "--calibration_file", cache, "--seed", str(args.seed)]
    if args.fixed_residues:
        cmd += ["--fixed_residues", args.fixed_residues]
    if args.num_ligand_atoms is not None:
        cmd += ["--num_ligand_atoms", str(args.num_ligand_atoms)]
    cmd += extra
    print("[auto_calib] run:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
