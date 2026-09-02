"""构建 v12.3 训练标签 = v12.2 训练集(6710 域) + S40 未用长域（L≥400）追加。

背景（v12.3，2026-09-02）：
- v12.2 蛋白模式最优但长蛋白训练量不足（6710 域中 L>400 仅 132 = 2%，验证 1A65/504、
  1BJ4/470 为长度 OOD）。v12.3 目标 = 补入 S40 长域重训，不改 v12.2 超参。
- 采样方案：**保留原 6710 + 追加**（最小改动、与 v12.2 单一变量 = 加长蛋白数据）。
  候选池 = S40 中 L≥400 且 未入 v12.2 train(6710)/holdout(1176) 的域。
  追加后 L>400 占比 ~2% → ~9.5%，CATH class 1/2/3 比例基本保持。
- 排除：① 已在 base/holdout 的域（防重复、防泄漏 hold-out）；② 验证蛋白同 PDB code 的域
  （防 H2/H1 泄漏）；③ parse 失败域。
- 标签格式与 v12.2 完全一致（domain_ids/seqs/coords/pH/charge/pI，每域 8 pH Uniform[4,10]，
  charge=net_charge(seq,pH)，pI=find_pI(seq)，coords=Cα[L,3]）。
- ⚠️ 不动 condition_defaults.yaml（μ/σ 沿用 v12.2，保持推理归一化一致）。

用法（code/ 下）：
  PYTHONPATH=. python tests/build_v12_3_augment.py
产物：data/cath/labels_v12_3_train.npz + 打印校验统计。
"""
import argparse
import os

import numpy as np
import yaml

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
    # 现有泛化 10 蛋白
    "1c6o", "1azm", "1as2", "1axw", "2feo", "5cqh", "1cge", "1ag0", "1a65", "1bj4",
    # E 新增验证蛋白（2026-09-02 验证集重构）
    "1cdg", "13bb", "1acc", "1ayl", "1bpm",
]


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
    ap.add_argument("--min_len", type=int, default=400)
    ap.add_argument("--n_pH", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_pfx", default=",".join(DEFAULT_EXCLUDE_PFX),
                    help="逗号分隔 PDB 前缀，从候选排除（验证蛋白泄漏保护）")
    args = ap.parse_args()
    exclude_pfx = [s.strip().lower() for s in args.exclude_pfx.split(",") if s.strip()]
    # 非标准残基域（freesasa 报错/parse 成 X，从候选剔除，2026-09-02 扫描确认）
    bad_domains = {"1cc1L00", "1gw5B00", "1nthA00", "3opbA02"}

    # ---- base (v12.2 train) ----
    d = np.load(args.base, allow_pickle=True)
    base_ids = list(d["domain_ids"])
    n_pH0 = d["pH"].shape[0] // len(base_ids)
    assert n_pH0 == args.n_pH
    print(f"base v12.2 train: {len(base_ids)} 域 × {n_pH0} pH", flush=True)
    used = set(base_ids)
    # holdout 域也不加（保 v12.2 hold-out 口径干净）
    if args.holdout:
        ho = set(np.load(args.holdout, allow_pickle=True)["domain_ids"])
        used |= ho
        print(f"holdout 排除 {len(ho)} 域", flush=True)

    # ---- candidate long domains via fasta coarse screen ----
    seqlen = fasta_lengths(args.fa)
    coarse = {dd: L for dd, L in seqlen.items() if L >= args.min_len - 5
              and dd not in used and dd not in bad_domains
              and not any(dd[:4].lower() == p for p in exclude_pfx)}
    print(f"粗筛候选 (fa L>={args.min_len-5}, 排除已用/验证/非标): {len(coarse)}", flush=True)

    # ---- parse + label 追加 ----
    rng = np.random.RandomState(args.seed)
    add_ids, add_seqs, add_coords, add_pHs, add_charges, add_pIs = [], [], [], [], [], []
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
        add_ids.append(dd)
        add_seqs.append(seq)
        add_coords.append(coords_i)
        add_pHs.append(pH_i)
        add_charges.append(charge_i)
        add_pIs.append(np.full(args.n_pH, pI, dtype=np.float32))
        n_ok += 1
    print(f"新增长域解析成功 {n_ok}（parse失败 {n_parse_fail}，parse<{args.min_len} 被剔 {n_short}）",
          flush=True)

    # ---- merge & save ----
    ids = np.array(base_ids + add_ids)
    seqs = np.concatenate([d["seqs"], np.array(add_seqs, dtype=object)])
    coords = np.concatenate([d["coords"], np.array(add_coords, dtype=object)])
    pH = np.concatenate([d["pH"], np.concatenate(add_pHs).astype(np.float32)])
    charge = np.concatenate([d["charge"], np.concatenate(add_charges)])
    pI = np.concatenate([d["pI"], np.concatenate(add_pIs)])
    np.savez(args.out, domain_ids=ids, seqs=seqs, coords=coords,
             pH=pH, charge=charge, pI=pI)
    print(f"已写 {args.out}：{len(ids)} 域 × {args.n_pH} pH = {len(pH)} 样本", flush=True)

    # ---- 校验统计 ----
    lens = [len(s) for s in seqs]
    import collections
    hist = collections.Counter(min(L // 100 * 100, 1000) for L in lens)
    l400 = sum(1 for L in lens if L > 400)
    l450 = sum(1 for L in lens if L > 450)
    l500 = sum(1 for L in lens if L > 500)
    print(f"总域 {len(ids)}  长度 mean={np.mean(lens):.1f} median={np.median(lens):.0f} "
          f"max={max(lens)}  L>400={l400}({l400/len(ids)*100:.1f}%)  "
          f"L>450={l450}  L>500={l500}", flush=True)
    print("长度直方图(/100):", dict(sorted(hist.items())), flush=True)
    # CATH class 比例
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
