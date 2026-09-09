#!/usr/bin/env python
"""L11 Exp3/Exp4 CPU 并行采样（读 exp3_groups.json / exp4_groups.json，可 resume）。

每组一个进程跑 run_guided.py：
    protein : MoMPNN backbone + v12.2 cond encoder + L11 现场小样本校准表（auto）
    ligand  : LigandMPNN(atom25) backbone + v14 cond encoder + L11 现场小样本校准表（auto）
固定链 I 残基 I3 I5 I9 I34 I35 I89 I124 I131 I134 I135（native）。
写 output/<exp>/<dir>/{seqs.fa,summary.json}。已完成的组（summary.json >= num_samples）跳过。

用法（confumpnn 环境）：
    python L11design/test/exp34_sampling.py --groups-json L11design/test/exp3_groups.json \
        [--num_samples 300] [--threads 12] [--concurrency 8] [--force]
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

PROT_W = str(ROOT / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt")
LIG_W = str(ROOT / "LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt")
PROT_COND = str(ROOT / "output/finetune_v12_2/finetune_epoch030.pt")
LIG_COND = str(ROOT / "output/finetune_ligand_v14_rna/finetune_epoch050.pt")
FIXED = "I3 I5 I9 I34 I35 I89 I124 I131 I134 I135"


def group_done(out_dir, num_samples):
    sj = out_dir / "summary.json"
    if not sj.is_file():
        return False
    try:
        s = json.load(open(sj))
        return len(s.get("sequences", [])) >= num_samples
    except Exception:
        return False


def build_cmd(exp, grp, cfg, num_samples, cal_root):
    g = cfg["groups"][grp]
    mode = g["mode"]
    pH = float(g.get("pH", cfg.get("pH", 7.4)))
    target = float(g["target"])
    out_dir = L11 / "output" / exp / g["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb = (str(L11 / "input/L11.pdb") if mode == "protein"
           else str(L11 / "input/L11_RNA.pdb"))
    # cal_root 在 groups json 中是相对项目根（如 "L11design/output/exp3/cal"）
    cal_file = str(ROOT / cal_root / g["calib"])
    cmd = [CONF, "run_guided.py", "--pdb", pdb, "--pH", f"{pH}",
           "--target_charge", f"{target}", "--cond_encoder",
           (PROT_COND if mode == "protein" else LIG_COND),
           "--fixed_residues", FIXED, "--num_samples", str(num_samples),
           "--seed", str(g["seed"]), "--out_dir", str(out_dir),
           "--calibrate", "auto", "--calibration_file", cal_file]
    cmd += ["--weights", (PROT_W if mode == "protein" else LIG_W)]
    return cmd, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups-json", required=True)
    ap.add_argument("--num_samples", type=int, default=300)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.groups_json))
    exp = cfg["exp"]
    groups = list(cfg["groups"].keys())
    print(f"Exp={exp}  组={groups}  num_samples={args.num_samples}", flush=True)

    queue = [g for g in groups
             if args.force or not group_done(L11 / "output" / exp / cfg["groups"][g]["dir"],
                                             args.num_samples)]
    print(f"待采样 {len(queue)} 组: {queue}", flush=True)
    if not queue:
        print("全部已完成。")
        return

    log_dir = L11 / "test" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    active = {}
    t0 = time.time()
    while queue or active:
        for key in list(active):
            p = active[key]
            if p.poll() is not None:
                rc = p.returncode
                d = cfg["groups"][key]["dir"]
                ok = rc == 0 and group_done(L11 / "output" / exp / d, args.num_samples)
                print(f"[{time.strftime('%H:%M:%S')}] {'完成' if ok else 'FAIL'} {key} "
                      f"(rc={rc}) el={(time.time()-t0)/60:.1f}min", flush=True)
                del active[key]
        while queue and len(active) < args.concurrency:
            grp = queue.pop(0)
            cmd, out_dir = build_cmd(exp, grp, cfg, args.num_samples,
                                     cfg.get("cal_root", "L11design/output/exp3/cal"))
            logf = log_dir / f"sample_{exp}_{grp}.log"
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = str(args.threads)
            env["MKL_NUM_THREADS"] = str(args.threads)
            env["CUDA_VISIBLE_DEVICES"] = ""
            print(f"[{time.strftime('%H:%M:%S')}] 启动 {grp} (CPU, OMP={args.threads}) → {out_dir}", flush=True)
            with open(logf, "w") as f:
                active[grp] = subprocess.Popen(cmd, cwd=str(CODE), env=env, stdout=f,
                                               stderr=subprocess.STDOUT)
        if active:
            time.sleep(5)
    print(f"Exp{exp} CPU 采样结束，总用时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
