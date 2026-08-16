"""TemBERTureTm 热稳打分：3 replica 平均熔解温度。
支持批量：--input-dir 递归扫描所有 seqs.fa（模型只加载一次）。

用法（confumpnn-temberture 环境）：
  python temberture_score.py --input-dir <dir>        # 每个 seqs.fa 旁输出 seqs.fa.tm.csv
  python temberture_score.py --fasta f.fa --out o.csv # 单文件
"""
import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, "/data/nfs/IC/baokun_yu/ConfuMPNN/TemBERTure/temBERTure")
from temBERTure import TemBERTure  # noqa: E402

TM = "/data/nfs/IC/baokun_yu/ConfuMPNN/TemBERTure/temBERTure/temBERTure_TM"


def read_fa(path):
    seqs = []
    name = None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
        elif line:
            seqs.append((name, line))
    return seqs


def score_fasta(replicas, fa_path, out_csv):
    seqs = read_fa(fa_path)
    rows = []
    for name, seq in seqs:
        tms = [float(m.predict(seq)[0]) for m in replicas]
        mean_tm = sum(tms) / len(tms)
        rows.append([name, len(seq), round(mean_tm, 2), tms])
        print(f"  {name[:40]:<42} Tm={mean_tm:.2f}  replicas={[round(t,1) for t in tms]}", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "length", "mean_tm", "tm_replicas"])
        w.writerows(rows)
    print(f"  [{fa_path}] 已写 {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None, help="批量：递归扫描所有 seqs.fa")
    ap.add_argument("--dirs-file", default=None,
                    help="并行分组：读目录列表文件（每行一个含 seqs.fa 的目录）；已有输出则跳过")
    ap.add_argument("--fasta", default=None, help="单文件：输入 FASTA")
    ap.add_argument("--out", default=None, help="单文件：输出 CSV")
    args = ap.parse_args()
    if args.input_dir is None and args.dirs_file is None and args.fasta is None:
        raise SystemExit("必须提供 --input-dir / --dirs-file / --fasta 之一")
    if args.fasta is not None and args.out is None:
        raise SystemExit("单文件模式需要 --out")

    replicas = []
    for r in ["replica1", "replica2", "replica3"]:
        print(f"加载 {r}...", flush=True)
        m = TemBERTure(adapter_path=f"{TM}/{r}/", device="cpu",
                       batch_size=16, task="regression")
        replicas.append(m)

    if args.dirs_file:
        dirs = [l.strip() for l in open(args.dirs_file) if l.strip()]
        print(f"分组模式：{len(dirs)} 个目录", flush=True)
        for d in dirs:
            fa = os.path.join(d, "seqs.fa")
            if not os.path.exists(fa):
                continue
            out_csv = fa + ".tm.csv"
            if os.path.exists(out_csv):
                print(f"  skip {d}（已存在）", flush=True)
                continue
            score_fasta(replicas, fa, out_csv)
    elif args.input_dir:
        fastas = sorted(glob.glob(os.path.join(args.input_dir, "**", "seqs.fa"), recursive=True))
        print(f"批量模式：找到 {len(fastas)} 个 seqs.fa", flush=True)
        for fa in fastas:
            score_fasta(replicas, fa, fa + ".tm.csv")
    else:
        score_fasta(replicas, args.fasta, args.out)


if __name__ == "__main__":
    main()
