"""v7 步骤1：从 CATH 完整域库筛选高正电（碱性）域候选。

背景：CATH S40 碱性域（pH7.4 净电荷 > +5）仅 2209 个，其中 +10~+15 仅 308、
+15~+20 仅 76 —— 极端正电 target 外推不稳定的根因。CATH 官方只提供 S20/S40
代表集，但 classification-data/ 有**完整域库**（cath-domain-seqs.fa，含冗余成员），
是扩充碱性骨架的同源来源。

方法：
  1. 解析 cath-domain-seqs.fa（所有 CATH 域序列）→ {domain_id: seq}
  2. 用与训练标签一致的电荷口径（HH 方程 + 同一 pKa 表，pH=7.4）算每域净电荷
  3. 一致性校验：对 S40 域用本脚本快速电荷 vs domain_charge7.npy 应吻合
  4. 候选 = 碱性（charge > +5）且 长度 40~250 且 不在 S40 已有碱性域 且 非验证 PDB
  5. 按电荷区间分层统计，输出候选清单

用法（code/ 下）：
  PYTHONPATH=. python tests/screen_basic_domains.py \
      --full_fa data/cath/cath-domain-seqs.fa \
      --s40_fa  data/cath/cath-dataset-nonredundant-S40.fa \
      --out      data/cath/candidates_basic_v7.txt
"""
import argparse
import math
import time

import numpy as np
import torch

from src.pka import (AAS, PKA_C_TERM, PKA_N_TERM, PKA_SIDECHAIN)

LN10 = math.log(10.0)
pH7 = 7.4
# 验证 PDB（泄漏保护，与训练 --exclude 一致）
EXCLUDE_PDB = {"1b24", "1bc8", "1crn", "1ubq", "2lzm"}


def parse_fasta(path):
    """解析 fasta → {domain_id: seq}。"""
    seqs = {}
    cur = None
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                # header 形如 'cath|4_4_0|101mA00/0-153' → 提取 '101mA00'
                cur = line[1:].split()[0].split("|")[-1].split("/")[0]
                buf = []
            else:
                buf.append(line)
        if cur is not None:
            seqs[cur] = "".join(buf)
    return seqs


def _sigmoid(x):
    """纯 Python sigmoid（与 torch.sigmoid 数值等价，避免 float→Tensor 开销）。"""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def build_charge_table(pH):
    """预计算 20 种 AA 在 pH 下的侧链电荷 + N/C 端电荷（标量，与 net_charge 同口径）。"""
    table = {}
    for aa in AAS:
        pKa = PKA_SIDECHAIN.get(aa)
        if pKa is None:
            table[aa] = 0.0
        elif aa in ("D", "E", "C", "Y"):
            table[aa] = -_sigmoid(LN10 * (pH - pKa))
        else:  # K/R/H
            table[aa] = _sigmoid(LN10 * (pKa - pH))
    n_term = _sigmoid(LN10 * (PKA_N_TERM - pH))
    c_term = -_sigmoid(LN10 * (pH - PKA_C_TERM))
    return table, n_term, c_term


def fast_charge(seq, table, n_term, c_term):
    """O(L) 查表求净电荷（与 net_charge 逐残基一致）。"""
    return sum(table.get(a, 0.0) for a in seq) + n_term + c_term


def parse_domain_id(did):
    """DomainID '1oaiA00' → (pdb='1oai', chain='A', num='00')。"""
    pdb = did[:4].lower()
    chain = did[4]
    num = did[5:]
    return pdb, chain, num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_fa", required=True)
    ap.add_argument("--s40_fa", required=True)
    ap.add_argument("--s40_charge7", default="data/cath/domain_charge7.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--basic_lo", type=float, default=5.0)
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=250)
    args = ap.parse_args()

    table, n_term, c_term = build_charge_table(pH7)

    # ---- 一致性校验：S40 快速电荷 vs domain_charge7.npy（数量级对比） ----
    # fast_charge 与 net_charge 公式/ pKa 表完全一致；npy 由 build_labels 用 net_charge 生成。
    # 顺序不可严格对应（npy 是 glob 目录序），故只对比"碱性域数量"作双重保险。
    t0 = time.time()
    s40_seqs = parse_fasta(args.s40_fa)
    s40_ids = list(s40_seqs.keys())
    q_fast = [fast_charge(s40_seqs[i], table, n_term, c_term) for i in s40_ids]
    ref = np.load(args.s40_charge7)
    n_basic_fast = sum(1 for q in q_fast if q > args.basic_lo)
    n_basic_ref = int((ref > args.basic_lo).sum())
    print(f"[校验] S40 {len(q_fast)} 域（npy {len(ref)}）：碱性域快速算 {n_basic_fast} vs npy {n_basic_ref}")
    if abs(n_basic_fast - n_basic_ref) / max(n_basic_ref, 1) > 0.05:
        print("  ⚠️ 碱性域数量偏差 >5%，电荷口径可能不一致！停止。")
        return

    # S40 已有碱性域（避免重复扩充）
    s40_basic = {i for i, q in zip(s40_ids, q_fast) if q > args.basic_lo}
    print(f"S40 碱性域(>+{args.basic_lo}): {len(s40_basic)}")

    # ---- 完整域库筛选 ----
    full = parse_fasta(args.full_fa)
    print(f"完整域库: {len(full)} 域，解析用时 {(time.time()-t0):.1f}s")

    cands = []
    stat = {}
    for did, seq in full.items():
        pdb, chain, _ = parse_domain_id(did)
        L = len(seq)
        if L < args.min_len or L > args.max_len:
            continue
        q = fast_charge(seq, table, n_term, c_term)
        if q <= args.basic_lo:
            continue
        if did in s40_basic:
            continue  # S40 已训练过的碱性域
        if pdb in EXCLUDE_PDB:
            continue  # 泄漏保护
        cands.append((did, pdb, chain, L, q))
        b = min(int((q + 20) // 5), 11)  # 5 一档
        stat[b] = stat.get(b, 0) + 1

    cands.sort(key=lambda x: -x[4])  # 电荷降序
    print(f"\n候选碱性域（S40 之外新增）: {len(cands)}")
    print("按电荷区间分层：")
    for lo in range(-15, 36, 5):
        n = sum(1 for c in cands if lo <= c[4] < lo + 5)
        if n:
            print(f"  [{lo:+3d}, {lo+5:+3d}): {n}")
    print("\nTop 20（电荷最高）:")
    for did, pdb, chain, L, q in cands[:20]:
        print(f"  {did}  {pdb} chain={chain} L={L} charge={q:+.2f}")

    with open(args.out, "w") as f:
        f.write("# DomainID  PDB  chain  length  charge7\n")
        for did, pdb, chain, L, q in cands:
            f.write(f"{did}\t{pdb}\t{chain}\t{L}\t{q:.4f}\n")
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
