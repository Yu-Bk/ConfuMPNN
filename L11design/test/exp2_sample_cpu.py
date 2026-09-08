#!/usr/bin/env python
"""L11 Exp2：CPU 并行采样（8 组同时，每组 OMP 12 线程，单进程 n=300）。

说明：192 核空闲、GPU 被他人占满 → 采样先走 CPU。run_guided 各进程独立
（每组一个进程，各写 output/exp2/<mode>_q<t>/）。OMP_NUM_THREADS 默认 12
避免线程过量（默认 192 反而 ~13x 更慢）。已完成的组（summary.json≥300）跳过。

用法（confumpnn 环境）：python L11design/test/exp2_sample_cpu.py [--threads 12] [--concurrency 8]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
L11 = ROOT / "L11design"
CODE = ROOT / "code"
CONF = "/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"
PROT_PDB = str(L11 / "input/L11.pdb")
LIG_PDB = str(L11 / "input/L11_RNA.pdb")
PROT_COND = str(ROOT / "output/finetune_v12_2/finetune_epoch030.pt")
LIG_COND = str(ROOT / "output/finetune_ligand_v14_rna/finetune_epoch050.pt")
PROT_W = str(ROOT / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt")
LIG_W = str(ROOT / "LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt")
PROT_CAL = str(ROOT / "output/charge_calibration_v12_2.json")
LIG_CAL = str(ROOT / "output/charge_calibration_v14_ligand_clean.json")
FIXED = "I3 I5 I9 I34 I35 I89 I124 I131 I134 I135"
GROUPS = []
for mode in ["protein", "ligand"]:
    for t, tf in [("6.87", 6.87), ("8", 8.0), ("10", 10.0), ("12", 12.0)]:
        GROUPS.append((mode, t, tf))


def group_dir(mode, t):
    return L11 / "output/exp2" / f"{mode}_q{t}"


def is_done(mode, t, n=300):
    sj = group_dir(mode, t) / "summary.json"
    if not sj.is_file():
        return False
    try:
        return len(json.load(open(sj)).get("sequences", [])) >= n
    except Exception:
        return False


def build_cmd(mode, t, tf, seed):
    out_dir = group_dir(mode, t)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb = PROT_PDB if mode == "protein" else LIG_PDB
    cond = PROT_COND if mode == "protein" else LIG_COND
    cmd = [CONF, "run_guided.py", "--pdb", pdb, "--pH", "7.4",
           "--target_charge", f"{tf}", "--cond_encoder", cond,
           "--fixed_residues", FIXED, "--num_samples", "300",
           "--seed", str(seed), "--out_dir", str(out_dir),
           "--calibrate", "global"]
    if mode == "protein":
        cmd += ["--weights", PROT_W, "--calibration_file", PROT_CAL]
    else:
        cmd += ["--weights", LIG_W, "--calibration_file", LIG_CAL]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    queue = [g for g in GROUPS if args.force or not is_done(*g)]
    print(f"待采样 {len(queue)} 组: {[f'{m}_q{t}' for m, t, _ in queue]}", flush=True)
    if not queue:
        print("全部已完成。")
        return

    log_dir = L11 / "test" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    active = {}  # mode_t -> Popen
    t0 = time.time()
    while queue or active:
        # 清理完成
        for key in list(active):
            p = active[key]
            if p.poll() is not None:
                rc = p.returncode
                mode, t = key.split("_q")[0], key.split("_q")[1]
                ok = rc == 0 and is_done(mode, t)
                print(f"[{time.strftime('%H:%M:%S')}] {'完成' if ok else 'FAIL'} {key} (rc={rc}) "
                      f"elapsed={(time.time()-t0)/60:.1f}min", flush=True)
                del active[key]
        # 启动新任务
        while queue and len(active) < args.concurrency:
            mode, t, tf = queue.pop(0)
            seed = 20260909 + (3 if mode == "ligand" else 1) * 10000 + int(round(tf * 10))
            cmd = build_cmd(mode, t, tf, seed)
            logf = log_dir / f"sample_cpu_{mode}_q{t}.log"
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = str(args.threads)
            env["MKL_NUM_THREADS"] = str(args.threads)
            env["CUDA_VISIBLE_DEVICES"] = ""
            key = f"{mode}_q{t}"
            print(f"[{time.strftime('%H:%M:%S')}] 启动 {key} (CPU, OMP={args.threads})", flush=True)
            with open(logf, "w") as f:
                active[key] = subprocess.Popen(cmd, cwd=str(CODE), env=env, stdout=f,
                                               stderr=subprocess.STDOUT)
        if active:
            time.sleep(5)
    print(f"CPU 采样全部结束，总用时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
