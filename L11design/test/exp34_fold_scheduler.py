#!/usr/bin/env python
"""L11 Exp3/Exp4：ESMFold 回折 GPU 轮询调度器（可 resume）。

扫描 data/exp3/*/ 与 data/exp4/*/ 下所有含 seqs.fa 的目录（含 data/exp3/_native）：
    已存在 plddt.csv 且行数 == fasta 条数 → 跳过。
    否则回折全部序列 → dir/folds/*.pdb + dir/plddt.csv。
调度纪律：只用 GPU util<12% 且 free mem>=--min_free_mem 的卡，单卡单任务，轮询等待空卡。

用法（confumpnn-esmfold 环境）：
    python L11design/test/exp34_fold_scheduler.py [--min_free_mem 20000] [--poll 60]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_free_mem", type=int, default=20000)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--num_recycles", type=int, default=3)
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
    while pending:
        g = None
        while g is None:
            for idx, (util, free) in nvidia().items():
                if util < 12 and free >= args.min_free_mem:
                    g = idx
                    break
            if g is None:
                print(f"[{time.strftime('%H:%M:%S')}] 暂无空闲卡（需 util<12% & free>={args.min_free_mem}MiB），"
                      f"{args.poll}s 后再试", flush=True)
                time.sleep(args.poll)
        d, fa = pending.pop(0)
        logf = d / "esmfold.log"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(g)
        cmd = [ESM, str(SCRIPT), "--fasta", str(fa), "--out", str(d / "plddt.csv"),
               "--outdir", str(d / "folds"), "--device", "cuda",
               "--num_recycles", str(args.num_recycles)]
        print(f"[{time.strftime('%H:%M:%S')}] 回折 {d.relative_to(L11)} on GPU{g} "
              f"(n={fasta_count(fa)}) el={(time.time()-t0)/60:.1f}min", flush=True)
        with open(logf, "w") as f:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0 and need_fold(d) is None:
            print(f"[{time.strftime('%H:%M:%S')}] 完成 {d.name}", flush=True)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] FAIL {d.name} rc={rc} → 重新入队", flush=True)
            pending.insert(0, (d, fa))
            time.sleep(30)
    print("回折全部结束。")


if __name__ == "__main__":
    main()
