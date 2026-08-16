"""用 ESMFold 对多条序列批量打分，输出平均 pLDDT（可设计性指标）。
可选 --outdir 保存每条序列的回折结构 PDB（供 TM-score 自洽性检验）。

用法（confumpnn-esmfold 环境）：
  conda run -n confumpnn-esmfold python esmfold_score.py --fasta seqs.fa --out plddt.csv [--outdir folds/]
"""
import argparse
import csv
import glob
import os

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


def score_fasta(model, fasta_path, out_csv, outdir, num_recycles, device):
    """对单个 fasta 打分；outdir 非空时保存回折 PDB（模型已加载，可批量调用）。"""
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    seqs = parse_fasta(fasta_path)
    print(f"[{fasta_path}] 共 {len(seqs)} 条序列", flush=True)
    rows = []
    for name, seq in seqs:
        with torch.no_grad():
            out = model.infer(seq, num_recycles=num_recycles)
        plddt = float(out["mean_plddt"][0].item())
        rows.append((name, len(seq), round(plddt, 2)))
        print(f"  {name[:30]:<32} L={len(seq):4d}  mean_pLDDT={plddt:6.2f}", flush=True)
        if outdir:
            # 清洗 name 为合法文件名；同一 fasta 内 name 唯一（sample_i/native）
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:50]
            pdb_str = model.output_to_pdb(out)[0]  # fair-esm: 实例方法返回 List[str]
            with open(os.path.join(outdir, f"{safe}.pdb"), "w") as f:
                f.write(pdb_str)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "length", "mean_plddt"])
        w.writerows(rows)
    print(f"  已写入 {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=None, help="单文件模式：输入 FASTA")
    ap.add_argument("--out", default=None, help="单文件模式：输出 CSV")
    ap.add_argument("--input-dir", default=None,
                    help="批量模式：递归扫描该目录下所有 seqs.fa，各自输出 plddt.csv + folds/")
    ap.add_argument("--num_recycles", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=None, help="单文件模式：保存回折 PDB 的目录（TM-score 用）")
    args = ap.parse_args()

    if args.input_dir is None and args.fasta is None:
        raise SystemExit("必须提供 --fasta 或 --input-dir")
    if args.input_dir is None and args.out is None:
        raise SystemExit("单文件模式需要 --out")

    model = esm.pretrained.esmfold_v1()
    model = model.to(args.device)
    model.eval()
    print(f"ESMFold 加载完成（{args.device}）", flush=True)

    if args.input_dir:
        fastas = sorted(glob.glob(os.path.join(args.input_dir, "**", "seqs.fa"), recursive=True))
        print(f"批量模式：找到 {len(fastas)} 个 seqs.fa", flush=True)
        for fa in fastas:
            d = os.path.dirname(fa)
            score_fasta(model, fa, os.path.join(d, "plddt.csv"), os.path.join(d, "folds"),
                        args.num_recycles, args.device)
    else:
        score_fasta(model, args.fasta, args.out, args.outdir, args.num_recycles, args.device)


if __name__ == "__main__":
    main()
