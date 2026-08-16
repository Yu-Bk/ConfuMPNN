"""GRAVY（疏水性）计算：可溶性代理指标。用 Biopython ProtParam。"""
import collections

from Bio.SeqUtils.ProtParam import ProteinAnalysis


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


seqs = read_fa("/tmp/score_input.fa")
groups = collections.defaultdict(list)
for name, seq in seqs:
    g = ProteinAnalysis(seq).gravy()
    groups[name.split("_")[0]].append(g)
    print(f"{name:<22} GRAVY={g:+.3f}")
print("-" * 40)
for grp, vals in groups.items():
    print(f"{grp:<10} 均值={sum(vals) / len(vals):+.3f}  (n={len(vals)})")
