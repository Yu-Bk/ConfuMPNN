"""v7 步骤1：对碱性域抽样做去重（链级 + 序列级）。

两级去重（用户要求"去重"，防止同源蛋白冗余代表导致过拟合）：
  1. 链级：同一 PDB 多条链只保留电荷最高的一条（同一实验结构的多拷贝）
  2. 序列级：9-mer containment 同源去重（共享 k-mer 覆盖 ≥0.8 视为同源，近似 90% 同一性）

用法（code/ 下）：
  PYTHONPATH=. python tests/dedup_basic.py \
      --sample ../data/cath/sample_basic_v7.txt \
      --full_fa ../data/cath/cath-domain-seqs.fa \
      --out ../data/cath/dedup_basic_v7.txt
"""
import argparse

DEDUP_CONTAINMENT = 0.8  # 同源判定阈值
KMER = 9


def parse_fasta(path, only_ids=None):
    """解析 fasta → {domain_id: seq}（可选只取指定 ID 的序列）。"""
    want = set(only_ids) if only_ids else None
    seqs = {}
    cur = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None and (want is None or cur in want):
                    seqs[cur] = "".join(buf)
                cur = line[1:].split()[0].split("|")[-1].split("/")[0]
                buf = []
            else:
                buf.append(line)
        if cur is not None and (want is None or cur in want):
            seqs[cur] = "".join(buf)
    return seqs


def kmer_set(seq):
    """序列的 k-mer 集合。"""
    if len(seq) < KMER:
        return {seq}
    return {seq[i:i + KMER] for i in range(len(seq) - KMER + 1)}


def containment(sa, sb):
    """A 被 B 覆盖的比例 = |A∩B| / min(|A|,|B|)。高 = 同源。"""
    inter = len(sa & sb)
    denom = min(len(sa), len(sb))
    return inter / denom if denom else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--full_fa", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []  # (did, pdb, chain, L, q)
    with open(args.sample) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            rows.append((p[0], p[1], p[2], int(p[3]), float(p[4])))

    # ---- 1. 链级去重：同一 PDB 保留电荷最高链 ----
    by_pdb = {}
    for r in rows:
        by_pdb.setdefault(r[1], []).append(r)
    chain_kept = []
    for pdb, rs in by_pdb.items():
        if len(rs) == 1:
            chain_kept.extend(rs)
        else:
            rs.sort(key=lambda r: -r[4])  # 电荷降序
            chain_kept.append(rs[0])      # 只留电荷最高链
    print(f"链级去重: {len(rows)} → {len(chain_kept)}（同一 PDB 多链只留电荷最高）")

    # ---- 2. 序列级去重：k-mer containment 贪心 ----
    ids = [r[0] for r in chain_kept]
    seqs = parse_fasta(args.full_fa, only_ids=set(ids))
    missing = [i for i in ids if i not in seqs]
    if missing:
        print(f"⚠️ 缺失序列 {len(missing)} 个，跳过")
    kept = []
    clusters = []  # 记录同源簇（调试用）
    used = set()
    # 按电荷降序处理：优先保留电荷高的域
    chain_kept.sort(key=lambda r: -r[4])
    kmer_cache = {}
    for r in chain_kept:
        did = r[0]
        if did in used or did not in seqs:
            continue
        kset = kmer_cache.setdefault(did, kmer_set(seqs[did]))
        new_cluster = [did]
        used.add(did)
        # 与已保留的比对，若同源则并入其簇
        for rk in kept:
            ks = kmer_cache.setdefault(rk[0], kmer_set(seqs[rk[0]]))
            if containment(kset, ks) >= DEDUP_CONTAINMENT or containment(ks, kset) >= DEDUP_CONTAINMENT:
                new_cluster.append(rk[0])
        # 与当前保留集去重：移除同源的旧保留
        if len(new_cluster) > 1:
            kept = [k for k in kept if k[0] not in new_cluster[1:]]
        kept.append(r)
        clusters.append(new_cluster)

    kept.sort(key=lambda r: -r[4])
    print(f"序列级去重后: {len(kept)} 域")

    with open(args.out, "w") as f:
        f.write("# DomainID  PDB  chain  length  charge7\n")
        for did, pdb, chain, L, q in kept:
            f.write(f"{did}\t{pdb}\t{chain}\t{L}\t{q:.4f}\n")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
