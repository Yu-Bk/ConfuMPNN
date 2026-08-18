"""v7 步骤1：用域序列在整链中定位 → 切出 CATH 风格域片段文件。

背景：整链提取方案把 312 条长链（>250 残基）过滤掉了——这些是含大域外部分的
蛋白。CATH 域是链的连续片段，域序列（cath-domain-seqs.fa）是链序列的子串。
本脚本用**子串定位**在整链 PDB 里找到域的起止残基，切出域片段 ATOM → 与 S40
训练域（坐标+域序列+域电荷）粒度完全一致，且保留全部候选。

用法（code/ 下）：
  PYTHONPATH=. python tests/extract_domain.py \
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
    """读 PDB 文本，提取目标链的有序残基列表。

    返回 [ {resnum, inscode, resname, seq1, lines}, ... ]（按 ATOM 记录顺序），
    只保留蛋白标准残基；MODEL>1 跳过。
    """
    residues = []
    seen = set()
    n_model = 0
    for line in open(pdb_path):
        if line.startswith("MODEL"):
            n_model += 1
            continue
        if n_model > 1:
            continue
        if line.startswith(("ATOM",)) and len(line) >= 27:
            if line[21] != chain_id:
                continue
            resname = line[17:20].strip()
            seq1 = AA3TO1.get(resname)
            if seq1 is None:
                continue
            key = (line[22:26].strip(), line[26:27].strip())
            if key in seen:
                # 该残基已有记录 → 追加原子行
                residues[-1]["lines"].append(line)
                continue
            seen.add(key)
            residues.append({
                "resnum": line[22:26].strip(),
                "inscode": line[26:27].strip(),
                "resname": resname,
                "seq1": seq1,
                "lines": [line],
            })
    return residues


def best_match(chain_seq, dom_seq):
    """在链序列里定位域序列：暴力滑动窗口找最大匹配。

    碱性蛋白序列富含 K/R，高度重复，中间段定位易错位。滑动窗口逐个位置算
    与域序列的逐位匹配数，取最高分窗口；identity≥0.85 才接受。容忍 N/C 端
    未建模残基差异。返回 (start, length=n) 或 None。
    """
    n = len(dom_seq)
    m = len(chain_seq)
    if n == 0 or m < n:
        return None
    best_start, best_score = 0, -1
    for start in range(m - n + 1):
        score = 0
        for j in range(n):
            if chain_seq[start + j] == dom_seq[j]:
                score += 1
        if score > best_score:
            best_score = score
            best_start = start
    if best_score / n < 0.85:
        return None
    return best_start, n


def cut_domain(pdb_path, chain_id, dom_seq, out_path, min_len=40, max_len=250):
    """从整链切出域片段，写无后缀域文件。返回 (域长度, 成功与否)。"""
    residues = read_chain(pdb_path, chain_id)
    if not residues:
        return 0, False
    chain_seq = "".join(r["seq1"] for r in residues)
    hit = best_match(chain_seq, dom_seq)
    if hit is None:
        return 0, False
    start, length = hit
    if length < min_len or length > max_len:
        return length, False
    end = start + length - 1
    # 假阳性防护：切割出的序列必须与域序列高度一致（≥0.85），否则拒绝
    cut_seq = "".join(r["seq1"] for r in residues[start:end + 1])
    n_match = sum(1 for a, b in zip(cut_seq, dom_seq) if a == b)
    ident = n_match / max(len(dom_seq), len(cut_seq))
    if ident < 0.85:
        return length, False
    lines = []
    for r in residues[start:end + 1]:
        lines.extend(r["lines"])
    with open(out_path, "w") as f:
        f.writelines(lines)
        f.write("END\n")
    return length, True


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
            rows.append((p[0], p[1], p[2], int(p[3]), float(p[4])))

    seqs = parse_fasta(args.full_fa, only_ids={r[0] for r in rows})
    os.makedirs(args.out_dir, exist_ok=True)

    ok, fail, short, toolong = [], 0, 0, 0
    for did, pdb, chain, L_dom, q in rows:
        dom_seq = seqs.get(did, "")
        if not dom_seq:
            fail += 1
            continue
        pdb_path = os.path.join(args.pdb_dir, f"{pdb}.pdb")
        if not os.path.exists(pdb_path):
            fail += 1
            continue
        out = os.path.join(args.out_dir, did)
        L_cut, success = cut_domain(pdb_path, chain, dom_seq, out,
                                    args.min_len, args.max_len)
        if not success:
            if os.path.exists(out):
                os.remove(out)
            if L_cut == 0:
                fail += 1
            elif L_cut < args.min_len:
                short += 1
            else:
                toolong += 1
            continue
        ok.append((did, pdb, chain, L_cut, q))

    print(f"域片段切割成功: {len(ok)}  定位失败: {fail}  过短: {short}  过长: {toolong}")
    if ok:
        Ls = [r[3] for r in ok]
        print(f"域长范围: {min(Ls)}~{max(Ls)}，平均 {sum(Ls)/len(Ls):.0f}")
    with open(os.path.join(args.out_dir, "_extracted.txt"), "w") as f:
        f.write("# DomainID PDB chain length charge7\n")
        for did, pdb, chain, L, q in ok:
            f.write(f"{did}\t{pdb}\t{chain}\t{L}\t{q:.4f}\n")


if __name__ == "__main__":
    main()
