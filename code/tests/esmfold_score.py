"""用 ESMFold 对多条序列批量打分，输出平均 pLDDT（可设计性指标）。

用法（confumpnn-esmfold 环境）：
  conda run -n confumpnn-esmfold python esmfold_score.py --fasta seqs.fa --out plddt.csv
"""
import argparse
import csv

import torch

import esm


def parse_fasta(path):
    seqs = []
    name, lines = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    seqs.append((name, "".join(lines)))
                name, lines = line[1:], []
            elif line:
                lines.append(line)
    if name is not None:
        seqs.append((name, "".join(lines)))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True, help="输入 FASTA（多条）")
    ap.add_argument("--out", required=True, help="输出 CSV")
    ap.add_argument("--num_recycles", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = esm.pretrained.esmfold_v1()
    model = model.to(args.device)
    model.eval()
    print(f"ESMFold 加载完成（{args.device}）", flush=True)

    seqs = parse_fasta(args.fasta)
    print(f"共 {len(seqs)} 条序列", flush=True)
    rows = []
    for name, seq in seqs:
        with torch.no_grad():
            out = model.infer(seq, num_recycles=args.num_recycles)
        plddt = float(out["mean_plddt"][0].item())
        rows.append((name, len(seq), round(plddt, 2)))
        print(f"  {name[:30]:<32} L={len(seq):4d}  mean_pLDDT={plddt:6.2f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "length", "mean_plddt"])
        w.writerows(rows)
    print(f"已写入 {args.out}", flush=True)


if __name__ == "__main__":
    main()
