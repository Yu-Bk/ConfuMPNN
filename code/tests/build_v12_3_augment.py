"""构建 v12.3 训练标签 = v12.2 训练集(可解析 S40 部分) + S40 未用长域（L≥400）追加。

背景（v12.3，2026-09-02）：
- v12.2 蛋白模式最优但长蛋白训练量不足（6710 域中 L>400 仅 132 = 2%，验证 1A65/504、
  1BJ4/470 为长度 OOD）。v12.3 目标 = 补入 S40 长域重训，不改 v12.2 超参。
- 采样方案：**保留原 v12.2 训练集中可解析域 + 追加 S40 长域**（最小改动、与 v12.2
  单一变量 = 加长蛋白数据）。
- ⚠️ 2026-09-02 修复：labels_v12_2_train.npz 含 585 个不可解析域（外部碱性域 ext，
  PDB 不在 S40/dompdb → prody parse 失败）。v12.2 训练实际跳过它们只训 6125 域
  （空洞在数据末尾，无错位）。v12.3 若保留它们，append 的新增长域会被空洞推挤导致
  **pH/charge 标签段错位**。故 base 一律剔除不可解析域：
    ① 不在 --dompdb（dangling symlink 源缺失）
    ② --skip_ids 名单（v12.2 log 实际 skip；含 2 个在 dompdb 但仍 parse 失败的 S40 域）
- 排除：① 已入 base 的域；② holdout 域（保 v12.2 hold-out 口径干净）；③ 验证蛋白同 PDB
  code 域（防 H2/H1 泄漏）；④ 非标残基域。
- 标签格式与 v12.2 完全一致（domain_ids/seqs/coords/pH/charge/pI，每域 8 pH Uniform[4,10]，
  charge=net_charge(seq,pH)，pI=find_pI(seq)，coords=Cα[L,3]）。
- ⚠️ 不动 condition_defaults.yaml（μ/σ 沿用 v12.2，保持推理归一化一致）。

用法（code/ 下）：
  PYTHONPATH=. python tests/build_v12_3_augment.py \
      --skip_ids /tmp/v12_2_skip_ids.txt
产物：data/cath/labels_v12_3_train.npz + 打印校验统计。
"""
import argparse
import os

import numpy as np

from src.differentiable_charge import net_charge
from src.isoelectric_point import find_pI

ROOT = "/data/nfs/IC/baokun_yu/ConfuMPNN/data/cath"
RESTYPE3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}
# 验证蛋白 PDB code（现有泛化 10 蛋白 + E 新增验证蛋白），训练候选须排除同 PDB 域
DEFAULT_EXCLUDE_PFX = [
    "1c6o", "1azm", "1as2", "1axw", "2feo", "5cqh", "1cge", "1ag0", "1a65", "1bj4",
    "1cdg", "13bb", "1acc", "1ayl", "1bpm",
]
# 非标准残基域（freesasa 报错/parse 成 X，剔除，2026-09-02 扫描确认）
BAD_DOMAINS = {"1cc1L00", "1gw5B00", "1nthA00", "3opbA02"}


def parse_domain(path):
    """与 build_labels.py 完全一致的 Cα 解析。"""
    coords, resnames = [], []
    for line in open(path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            r = line[17:20].strip()
            if r in RESTYPE3TO1:
                resnames.append(RESTYPE3TO1[r])
    if len(resnames) < 20:
        return None, None
    return np.array(coords, dtype=np.float32), "".join(resnames)


def fasta_lengths(fa_path):
    """CATH S40 fasta → {domain_id: seq_len}（快速粗筛；最终以 parse_domain 复核）。"""
    seqlen, cur, buf = {}, None, []

    def flush():
        nonlocal cur, buf
        if cur is not None:
            seqlen[cur] = sum(len(x) for x in buf)
        buf = []
    for line in open(fa_path):
        if line.startswith(">"):
            flush()
            cur = line.split("|")[2].split("/")[0]
        else:
            buf.append(line.rstrip())
    flush()
    return seqlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=f"{ROOT}/labels_v12_2_train.npz")
    ap.add_argument("--holdout", default=f"{ROOT}/labels_holdout_train.npz")
    ap.add_argument("--dompdb", default=f"{ROOT}/S40/dompdb")
    ap.add_argument("--fa", default=f"{ROOT}/cath-dataset-nonredundant-S40.fa")
    ap.add_argument("--out", default=f"{ROOT}/labels_v12_3_train.npz")
    ap.add_argument("--skip_ids", default="",
                    help="每行一个域 id；从 base 剔除不可解析域（v12.2 log skip 名单，见文件头）")
    ap.add_argument("--min_len", type=int, default=400)
    ap.add_argument("--n_pH", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_pfx", default=",".join(DEFAULT_EXCLUDE_PFX))
    args = ap.parse_args()
    exclude_pfx = [s.strip().lower() for s in args.exclude_pfx.split(",") if s.strip()]

    # ---- base (v12.2 train) 剔除不可解析域 ----
    d = np.load(args.base, allow_pickle=True)
    base_ids = list(d["domain_ids"])
    n_pH0 = d["pH"].shape[0] // len(base_ids)
    assert n_pH0 == args.n_pH
    dompdb_files = set(os.listdir(args.dompdb))
    skip_set = {l.strip() for l in open(args.skip_ids)} if args.skip_ids else set()
    base_keep = [i for i, x in enumerate(base_ids)
                 if x in dompdb_files and x not in skip_set]
    print(f"base v12.2 train: {len(base_ids)} 域 → 剔除不可解析 "
          f"{len(base_ids)-len(base_keep)}（不在 dompdb / skip），保留 {len(base_keep)}", flush=True)
    b_ids = [base_ids[i] for i in base_keep]
    b_seqs = d["seqs"][base_keep]
    b_coords = d["coords"][base_keep]
    b_pH = np.concatenate([d["pH"][i * n_pH0:(i + 1) * n_pH0] for i in base_keep])
    b_charge = np.concatenate([d["charge"][i * n_pH0:(i + 1) * n_pH0] for i in base_keep])
    b_pI = np.concatenate([d["pI"][i * n_pH0:(i + 1) * n_pH0] for i in base_keep])
    used = set(b_ids)
    if args.holdout:
        ho = set(np.load(args.holdout, allow_pickle=True)["domain_ids"])
        used |= ho
        print(f"holdout 排除 {len(ho)} 域", flush=True)

    # ---- candidate long domains via fasta coarse screen ----
    seqlen = fasta_lengths(args.fa)
    coarse = {dd: L for dd, L in seqlen.items() if L >= args.min_len - 5
              and dd not in used and dd not in BAD_DOMAINS
              and not any(dd[:4].lower() == p for p in exclude_pfx)}
    print(f"粗筛候选 (fa L>={args.min_len-5}, 排除已用/验证/非标): {len(coarse)}", flush=True)

    # ---- parse + label 追加 ----
    rng = np.random.RandomState(args.seed)
    a_ids, a_seqs, a_coords, a_pHs, a_charges, a_pIs = [], [], [], [], [], []
    n_parse_fail, n_short, n_ok = 0, 0, 0
    for dd in sorted(coarse, key=lambda x: -coarse[x]):
        p = os.path.join(args.dompdb, dd)
        coords_i, seq = parse_domain(p)
        if coords_i is None or len(seq) < 20:
            n_parse_fail += 1
            continue
        if len(seq) < args.min_len:
            n_short += 1
            continue
        pH_i = rng.uniform(4.0, 10.0, args.n_pH)
        charge_i = np.array([net_charge(seq, ph) for ph in pH_i], dtype=np.float32)
        pI = find_pI(seq)
        a_ids.append(dd)
        a_seqs.append(seq)
        a_coords.append(coords_i)
        a_pHs.append(pH_i)
        a_charges.append(charge_i)
        a_pIs.append(np.full(args.n_pH, pI, dtype=np.float32))
        n_ok += 1
    print(f"新增长域解析成功 {n_ok}（parse失败 {n_parse_fail}，parse<{args.min_len} 被剔 {n_short}）",
          flush=True)

    # ---- merge & save ----
    ids = np.array(b_ids + a_ids)
    seqs = np.concatenate([b_seqs, np.array(a_seqs, dtype=object)])
    coords = np.concatenate([b_coords, np.array(a_coords, dtype=object)])
    pH = np.concatenate([b_pH, np.concatenate(a_pHs).astype(np.float32)])
    charge = np.concatenate([b_charge, np.concatenate(a_charges)])
    pI = np.concatenate([b_pI, np.concatenate(a_pIs)])
    np.savez(args.out, domain_ids=ids, seqs=seqs, coords=coords,
             pH=pH, charge=charge, pI=pI)
    print(f"已写 {args.out}：{len(ids)} 域 × {args.n_pH} pH = {len(pH)} 样本", flush=True)

    # ---- 校验统计 ----
    lens = [len(s) for s in seqs]
    import collections
    l400 = sum(1 for L in lens if L > 400)
    l450 = sum(1 for L in lens if L > 450)
    l500 = sum(1 for L in lens if L > 500)
    print(f"总域 {len(ids)}  长度 mean={np.mean(lens):.1f} median={np.median(lens):.0f} "
          f"max={max(lens)}  L>400={l400}({l400/len(ids)*100:.1f}%)  "
          f"L>450={l450}  L>500={l500}", flush=True)
    cl = {}
    for line in open(f"{ROOT}/cath-domain-list.txt"):
        if line.startswith("#") or not line.strip():
            continue
        pp = line.split()
        if len(pp) >= 2:
            cl[pp[0]] = int(pp[1])
    cc = collections.Counter(cl.get(i, 0) for i in ids)
    print("CATH class 分布:", dict(cc), flush=True)
    print(f"pH range [{pH.min():.2f},{pH.max():.2f}]", flush=True)


if __name__ == "__main__":
    main()
