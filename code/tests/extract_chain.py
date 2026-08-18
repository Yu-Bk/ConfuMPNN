"""v7 步骤1：生成外部碱性域的**单链骨架文件**（训练用，CATH 风格无后缀）。

训练脚本假设单链输入。RCSB 整链 PDB 可能多链/超长。统一策略：
  1. 提取目标链（DomainID 里的链 ID）的有序残基
  2. 链长 ≤ max_len → 整链即骨架（真实结构，电荷标签用切割序列自洽重算）
  3. 链长 > max_len → 用域序列**滑动窗口定位**域区域，切出 ≤max_len 的域片段
     （域序列来自 cath-domain-seqs.fa；碱性蛋白 K/R 富集易错位，暴力窗口最可靠）

输出：ext_basic_dompdb/{DomainID}（无后缀 PDB 文本，只含蛋白 ATOM，单 MODEL）
      + _extracted.txt（DomainID PDB chain length）

用法（code/ 下）：
  PYTHONPATH=. python tests/extract_chain.py \
      --list ../data/cath/dedup_basic_v7.txt \
      --full_fa ../data/cath/cath-domain-seqs.fa \
      --pdb_dir ../data/cath/ext_basic_pdb \
      --out_dir ../data/cath/ext_basic_dompdb
"""
import argparse
import os

AA3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y", "MSE": "M", "SEC": "C", "PYL": "K",
}


def parse_fasta(path, only_ids=None):
    """解析 fasta → {domain_id: seq}（只取指定 ID）。"""
    want = set(only_ids) if only_ids else None
    seqs = {}
    cur, buf = None, []
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


def read_chain(pdb_path, chain_id):
    """读 PDB 文本，提取目标链的有序残基列表（蛋白标准残基，单 MODEL）。"""
    residues = []
    seen = set()
    n_model = 0
    for line in open(pdb_path):
        if line.startswith("MODEL"):
            n_model += 1
            continue
        if n_model > 1:
            continue
        if line.startswith("ATOM") and len(line) >= 27:
            if line[21] != chain_id:
                continue
            seq1 = AA3TO1.get(line[17:20].strip())
            if seq1 is None:
                continue
            key = (line[22:26].strip(), line[26:27].strip())
            if key in seen:
                residues[-1]["lines"].append(line)
                continue
            seen.add(key)
            residues.append({
                "seq1": seq1,
                "lines": [line],
            })
    return residues


def best_window(chain_seq, dom_seq):
    """滑动窗口找域序列在链中的最佳位置。返回 (start, len) 或 None（identity<0.85）。"""
    n, m = len(dom_seq), len(chain_seq)
    if n == 0 or m < 20:
        return None
    # 窗口长度 = 链与域长度的折中：优先整域长度，链不足则用链长
    w = min(n, m)
    best_start, best_score = 0, -1
    for start in range(m - w + 1):
        score = sum(1 for j in range(w) if chain_seq[start + j] == dom_seq[j])
        if score > best_score:
            best_score = score
            best_start = start
    if best_score / w < 0.85:
        return None
    return best_start, w


def write_domfile(residues, out_path):
    lines = []
    for r in residues:
        lines.extend(r["lines"])
    with open(out_path, "w") as f:
        f.writelines(lines)
        f.write("END\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--full_fa", required=True)
    ap.add_argument("--pdb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=250)
    args = ap.parse_args()

    rows = []
    with open(args.list) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            rows.append((p[0], p[1], p[2]))
    seqs = parse_fasta(args.full_fa, only_ids={r[0] for r in rows})
    os.makedirs(args.out_dir, exist_ok=True)

    n_whole, n_cut, n_fail = 0, 0, 0
    out_rows = []
    for did, pdb, chain in rows:
        pdb_path = os.path.join(args.pdb_dir, f"{pdb}.pdb")
        if not os.path.exists(pdb_path):
            n_fail += 1
            continue
        residues = read_chain(pdb_path, chain)
        L = len(residues)
        if L < args.min_len:
            n_fail += 1
            continue
        out = os.path.join(args.out_dir, did)
        if L <= args.max_len:
            write_domfile(residues, out)   # 整链即骨架
            n_whole += 1
            out_rows.append((did, pdb, chain, L))
            continue
        # 长链：滑动窗口切域区域
        dom_seq = seqs.get(did, "")
        if not dom_seq:
            n_fail += 1
            continue
        chain_seq = "".join(r["seq1"] for r in residues)
        hit = best_window(chain_seq, dom_seq)
        if hit is None:
            n_fail += 1
            continue
        start, w = hit
        L_cut = min(w, args.max_len)
        write_domfile(residues[start:start + L_cut], out)
        n_cut += 1
        out_rows.append((did, pdb, chain, L_cut))

    print(f"整链骨架: {n_whole}  长链切域: {n_cut}  失败: {n_fail}  合计: {n_whole + n_cut}")
    with open(os.path.join(args.out_dir, "_extracted.txt"), "w") as f:
        f.write("# DomainID PDB chain length\n")
        for did, pdb, chain, L in out_rows:
            f.write(f"{did}\t{pdb}\t{chain}\t{L}\n")


if __name__ == "__main__":
    main()
