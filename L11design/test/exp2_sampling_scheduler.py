#!/usr/bin/env python
"""L11 Exp2 采样调度器（GPU 轮询版，可 resume）。

对 8 组 (mode,target) 调用 run_guided.py 采样 n=300/组：
    protein : MoMPNN backbone + v12.2 cond encoder + v12_2 global 校准
    ligand  : LigandMPNN(atom25) backbone + v14 cond encoder + v14_clean global 校准
每组 300 条，固定链 I 残基 I3 I5 I9 I34 I35 I89 I124 I131 I134 I135（native）。
目标 pH7.4，target_charge ∈ {6.87, 8, 10, 12}。

调度纪律：只使用 GPU util < 12% 且空闲显存 >= min_free_mem 的卡；每卡同时只跑 1 个任务。
已完成的组（output/exp2/<dir>/summary.json 含 >= num_samples 条）自动跳过 → resume。

用法（confumpnn 环境）：
    python L11design/test/exp2_sampling_scheduler.py [--num_samples 300] [--min_free_mem 4000]
                     [--poll 60] [--device-list 0,1,...] [--force]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data/nfs/IC/baokun_yu/ConfuMPNN")
L11 = ROOT / "L11design"
CODE = ROOT / "code"
CONF = "/home/baokun_yu/miniconda3/envs/confumpnn/bin/python"

GROUPS = []  # list of (mode, target_str, target_float)
for mode in ["protein", "ligand"]:
    for t, tf in [("6.87", 6.87), ("8", 8.0), ("10", 10.0), ("12", 12.0)]:
        GROUPS.append((mode, t, tf))

PROT_PDB = str(L11 / "input/L11.pdb")
LIG_PDB = str(L11 / "input/L11_RNA.pdb")
PROT_COND = str(ROOT / "output/finetune_v12_2/finetune_epoch030.pt")
LIG_COND = str(ROOT / "output/finetune_ligand_v14_rna/finetune_epoch050.pt")
PROT_W = str(ROOT / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt")
LIG_W = str(ROOT / "LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt")
PROT_CAL = str(ROOT / "output/charge_calibration_v12_2.json")
LIG_CAL = str(ROOT / "output/charge_calibration_v14_ligand_clean.json")
FIXED = "I3 I5 I9 I34 I35 I89 I124 I131 I134 I135"


def group_dir(mode, t):
    return L11 / "output/exp2" / f"{mode}_q{t}"


def is_done(mode, t, num_samples):
    d = group_dir(mode, t)
    sj = d / "summary.json"
    if not sj.is_file():
        return False
    try:
        s = json.load(open(sj))
    except Exception:
        return False
    return len(s.get("sequences", [])) >= num_samples


def nvidia():
    """返回 {idx: (util%, free_miB)}"""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    g = {}
    for line in out.strip().splitlines():
        idx, util, free = [x.strip() for x in line.split(",")]
        g[int(idx)] = (int(util), int(free))
    return g


def free_gpus(util_max=12, min_free_mib=4000, allow=None):
    g = nvidia()
    cands = []
    for idx, (util, free) in g.items():
        if allow is not None and idx not in allow:
            continue
        if util < util_max and free >= min_free_mib:
            cands.append(idx)
    return cands


def build_cmd(mode, t, tf, device, seed):
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
    return cmd, out_dir


def run_one(mode, t, tf, device, seed, log_dir):
    cmd, out_dir = build_cmd(mode, t, tf, device, seed)
    logf = log_dir / f"sampling_{mode}_q{t}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    print(f"[{time.strftime('%H:%M:%S')}] 启动 {mode}_q{t} on GPU{device} (seed {seed})", flush=True)
    with open(logf, "w") as f:
        p = subprocess.run(cmd, cwd=str(CODE), env=env, stdout=f, stderr=subprocess.STDOUT)
    ok = p.returncode == 0 and is_done(mode, t, 300)
    print(f"[{time.strftime('%H:%M:%S')}] {'完成' if ok else 'FAIL'} {mode}_q{t} (rc={p.returncode}) log={logf}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=300)
    ap.add_argument("--min_free_mem", type=int, default=4000)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--device-list", default=None, help="逗号分隔允许卡，如 0,1,2")
    ap.add_argument("--force", action="store_true", help="重跑已完成组")
    args = ap.parse_args()

    log_dir = L11 / "test" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    allow = None
    if args.device_list:
        allow = [int(x) for x in args.device_list.split(",")]

    queue = [(m, t, tf) for (m, t, tf) in GROUPS
             if args.force or not is_done(m, t, args.num_samples)]
    if not queue:
        print("全部组已完成。")
        return
    print(f"待采样 {len(queue)} 组: {[f'{m}_q{t}' for m, t, _ in queue]}")

    active = {}  # device -> subprocess info not used (run_one synchronous); handled below

    # 调度：用简单抢占 - 每次看空卡跑一个任务（同步阻塞在 run_one 内，但先探测空卡再跑）
    # 为允许多卡并行，这里用轻量并发：逐设备起 Popen，主循环 poll。
    running = {}  # gpu -> (popen, mode, t, tf, logf)
    qidx = 0
    while qidx < len(queue) or running:
        # 清理已完成
        for gpu in list(running):
            p = running[gpu]
            if p["popen"].poll() is not None:
                rc = p["popen"].returncode
                done = rc == 0 and is_done(p["mode"], p["t"], args.num_samples)
                print(f"[{time.strftime('%H:%M:%S')}] GPU{gpu} {'完成' if done else 'FAIL'} "
                      f"{p['mode']}_q{p['t']} (rc={rc})", flush=True)
                del running[gpu]
        # 填新任务
        if running:
            time.sleep(args.poll)
            continue
        gpus = free_gpus(util_max=12, min_free_mib=args.min_free_mem, allow=allow)
        if not gpus:
            time.sleep(args.poll)
            continue
        # 一次只填一个空卡任务（保持简单），但支持一个卡一个任务；有多个空卡时逐个启动
        for gpu in gpus:
            if qidx >= len(queue):
                break
            m, t, tf = queue[qidx]
            seed = 20260909 + (3 if m == "ligand" else 1) * 10000 + int(float(tf) * 10)
            out_dir = group_dir(m, t)
            cmd, _ = build_cmd(m, t, tf, gpu, seed)
            logf = log_dir / f"sampling_{m}_q{t}.log"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with open(logf, "w") as f:
                p = subprocess.Popen(cmd, cwd=str(CODE), env=env, stdout=f, stderr=subprocess.STDOUT)
            running[gpu] = {"popen": p, "mode": m, "t": t, "tf": tf}
            print(f"[{time.strftime('%H:%M:%S')}] 启动 {m}_q{t} on GPU{gpu} (seed {seed})", flush=True)
            qidx += 1
        if running:
            time.sleep(args.poll)
    print("采样调度全部结束。")


if __name__ == "__main__":
    main()
