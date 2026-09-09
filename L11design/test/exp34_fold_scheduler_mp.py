#!/usr/bin/env python
"""L11 Exp3/Exp4：ESMFold 多 GPU 轮询调度器（可 resume，多空卡并行）。

扫描 data/exp3/*/ 与 data/exp4/*/ 下所有含 seqs.fa 的目录（含 data/exp3/_native）：
    已存在 plddt.csv 且行数 >= fasta 条数 → 跳过。
    否则回折 → dir/folds/*.pdb + dir/plddt.csv。
调度：只用 util<--util_max% 且 free>=--min_free_mem 的卡；一个 GPU 上只跑一个任务，
多个空卡时同时起多个任务。轮询等待。

用法（confumpnn-esmfold 环境）：
    python L11design/test/exp34_fold_scheduler_mp.py [--min_free_mem 20000] [--poll 60]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ESM = "/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python"
CODE = Path("/data/nfs/IC/baokun_yu/ConfuMPNN/code")
L11 = Path("/data/nfs/IC/baokun_yu/ConfuMPNN/L11design")
SCRIPT = CODE / "tests/esmfold_score.py"


def nvidia():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    g = {}
    for line in out.strip().splitlines():
        idx, util, free = [x.strip() for x in line.split(",")]
        g[int(idx)] = (int(util), int(free))
    return g


def fasta_count(fa):
    return sum(1 for line in open(fa) if line.startswith(">"))


def need_fold(d):
    fa = d / "seqs.fa"
    if not fa.is_file():
        return None
    csv = d / "plddt.csv"
    n = fasta_count(fa)
    if csv.is_file():
        try:
            done = sum(1 for line in open(csv)) - 1  # minus header
        except Exception:
            done = 0
        if done >= n:
            return None
    return fa


def free_gpus(util_max, min_free_mib):
    g = nvidia()
    return [idx for idx, (util, free) in g.items()
            if util < util_max and free >= min_free_mib]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_free_mem", type=int, default=20000)
    ap.add_argument("--poll", type=int, default=45)
    ap.add_argument("--num_recycles", type=int, default=3)
    ap.add_argument("--util_max", type=int, default=12)
    args = ap.parse_args()

    pending = []
    for base in [L11 / "data/exp3", L11 / "data/exp4"]:
        for d in sorted(base.glob("*/")):
            fa = need_fold(d)
            if fa is not None:
                pending.append((d, fa))
    if not pending:
        print("没有待回折目录。")
        return
    print(f"待回折 {len(pending)} 目录: {[d.name for d, _ in pending]}", flush=True)

    t0 = time.time()
    running = {}  # gpu -> {popen, dir, logf, fa}
    while pending or running:
        # 清理完成
        for gpu in list(running):
            j = running[gpu]
            if j["popen"].poll() is not None:
                rc = j["popen"].returncode
                ok = rc == 0 and need_fold(j["dir"]) is None
                print(f"[{time.strftime('%H:%M:%S')}] GPU{gpu} {'完成' if ok else 'FAIL'} "
                      f"{j['dir'].name} rc={rc} el={(time.time()-t0)/60:.1f}min", flush=True)
                del running[gpu]
        # 填新任务
        if pending:
            gpus = free_gpus(args.util_max, args.min_free_mem)
            for gpu in gpus:
                if gpu in running or not pending:
                    continue
                d, fa = pending.pop(0)
                logf = d / "esmfold.log"
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                cmd = [ESM, str(SCRIPT), "--fasta", str(fa), "--out", str(d / "plddt.csv"),
                       "--outdir", str(d / "folds"), "--device", "cuda",
                       "--num_recycles", str(args.num_recycles)]
                print(f"[{time.strftime('%H:%M:%S')}] 回折 {d.relative_to(L11)} on GPU{gpu} "
                      f"(n={fasta_count(fa)})", flush=True)
                with open(logf, "w") as f:
                    p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
                running[gpu] = {"popen": p, "dir": d, "fa": fa}
        if running:
            time.sleep(args.poll)
        elif pending:
            print(f"[{time.strftime('%H:%M:%S')}] 暂无空闲卡，{args.poll}s 后再试", flush=True)
            time.sleep(args.poll)
    print("回折全部结束。")


if __name__ == "__main__":
    main()
