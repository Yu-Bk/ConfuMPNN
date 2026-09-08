#!/usr/bin/env python
"""L11 Exp2：ESMFold CPU 回折回退（GPU 被他人占满时的后备），可 resume。

策略：把 data/exp2/*/seqs.fa 切成 CHUNK 条的块，每个块一个 ESMFold 进程
（写共享 folds/ 目录 + 各自 chunk csv），并发 CONC 个进程，OMP=THREADS/进程。
全部块完成后合并每个目录的 plddt.csv。已完成的块跳过（resume）。

用法（confumpnn-esmfold 环境）：
    python L11design/test/exp2_fold_cpu.py [--chunk 30] [--conc 20] [--threads 8] [--recycles 3]
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

ESM = "/home/baokun_yu/miniconda3/envs/confumpnn-esmfold/bin/python"
CODE = Path("/data/nfs/IC/baokun_yu/ConfuMPNN/code")
L11 = Path("/data/nfs/IC/baokun_yu/ConfuMPNN/L11design")
SCRIPT = CODE / "tests/esmfold_score.py"


def parse_fasta(fa):
    seqs = []
    name, buf = None, []
    for line in open(fa):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None:
                seqs.append((name, "".join(buf)))
            name, buf = line[1:], []
        elif line:
            buf.append(line)
    if name is not None:
        seqs.append((name, "".join(buf)))
    return seqs


def chunk_count(fa):
    return sum(1 for _ in open(fa) if _.startswith(">"))


def split_chunks(d, chunk):
    """把 d/seqs.fa 切成 d/_chunks/chunk_*.fa；返回 list[chunk_fa]。"""
    fa = d / "seqs.fa"
    seqs = parse_fasta(fa)
    cdir = d / "_chunks"
    cdir.mkdir(exist_ok=True)
    out = []
    for k in range(0, len(seqs), chunk):
        cf = cdir / f"chunk_{k // chunk + 1:03d}.fa"
        with open(cf, "w") as f:
            for name, seq in seqs[k:k + chunk]:
                f.write(f">{name}\n{seq}\n")
        out.append(cf)
    return out


def chunk_done(cf):
    """该 chunk 的 plddt csv 是否完整。"""
    csv = cf.with_suffix(".plddt.csv")
    if not csv.is_file():
        return False
    n = chunk_count(cf)
    try:
        done = sum(1 for _ in open(csv)) - 1
    except Exception:
        done = 0
    return done >= n


def run_chunk(cf, outdir, device, threads, recycles):
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [ESM, str(SCRIPT), "--fasta", str(cf), "--out", str(cf.with_suffix(".plddt.csv")),
           "--outdir", str(outdir), "--device", "cpu", "--num_recycles", str(recycles)]
    log = open(cf.with_suffix(".fold.log"), "w")
    p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, log


def merge_dir(d):
    csvs = sorted(d.glob("_chunks/chunk_*.plddt.csv"))
    if not csvs:
        return
    rows = []
    for c in csvs:
        with open(c) as f:
            lines = f.read().splitlines()
        if not lines:
            continue
        if not rows:
            rows.append(lines[0])  # header
        rows.extend(lines[1:])
    with open(d / "plddt.csv", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"  合并 {d.name}: {len(rows)-1} 条 → plddt.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--conc", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--recycles", type=int, default=3)
    args = ap.parse_args()

    dirs = sorted(Path(L11 / "data/exp2").glob("*/"))
    all_chunks = []
    for d in dirs:
        fa = d / "seqs.fa"
        if not fa.is_file():
            continue
        for cf in split_chunks(d, args.chunk):
            if not chunk_done(cf):
                all_chunks.append((cf, d))
    total = sum(chunk_count(cf) for cf, _ in all_chunks)
    print(f"待回折 chunks: {len(all_chunks)}（{total} 条序列），并发 {args.conc}，OMP {args.threads}/进程", flush=True)
    if not all_chunks:
        print("全部 chunks 已完成。")
        return

    active = []  # list of (Popen, log, cf, d)
    idx = 0
    t0 = time.time()
    while idx < len(all_chunks) or active:
        # 清理完成
        for item in list(active):
            p, log, cf, d = item
            if p.poll() is not None:
                log.close()
                ok = chunk_done(cf)
                print(f"[{time.strftime('%H:%M:%S')}] {'ok' if ok else 'FAIL'} {cf.relative_to(L11)} "
                      f"({(time.time()-t0)/60:.1f}min)", flush=True)
                active.remove(item)
        # 填满
        while idx < len(all_chunks) and len(active) < args.conc:
            cf, d = all_chunks[idx]
            p, log = run_chunk(cf, d / "folds", "cpu", args.threads, args.recycles)
            active.append([p, log, cf, d])
            idx += 1
        if active:
            time.sleep(5)

    # 合并
    print("合并各目录 plddt.csv ...", flush=True)
    for d in dirs:
        if (d / "seqs.fa").is_file():
            merge_dir(d)
    print(f"CPU 回折全部结束，总用时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
