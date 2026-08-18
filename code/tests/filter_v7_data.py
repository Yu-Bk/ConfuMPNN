"""v7 后置过滤：剔除外部碱性域中实际电荷 <+5 的"假碱性"样本。

背景：外部骨架是整链/切域，其实际序列电荷可能 ≠ CATH 候选域电荷（整链含域外
负电残基）。训练标签基于骨架实际序列（自洽），但需保证加入的确实是碱性样本。
本脚本加载 labels_balanced_v7.npz，剔除外部域中 net_charge(seq, 7.4) < 5 的域
（连同其 8 个 pH 样本），并重算条件向量 μ/σ 写回 yaml。

用法（code/ 下）：
  PYTHONPATH=. python tests/filter_v7_data.py \
      --npz ../data/cath/labels_balanced_v7.npz \
      --ext_dir ../data/cath/ext_basic_dompdb \
      --cfg configs/condition_defaults.yaml
"""
import argparse
import glob
import os

import numpy as np
import yaml

from src.differentiable_charge import net_charge


def build_condition_vector(pH, charge):
    return np.array([pH, 1.0, charge, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ext_dir", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--basic_lo", type=float, default=5.0)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    ids = list(d["domain_ids"]); seqs = list(d["seqs"])
    pHs = d["pH"].astype(np.float32); charges = d["charge"].astype(np.float32)
    pIs = d["pI"].astype(np.float32); coords = d["coords"]
    n_dom = len(ids)
    n_pH = pHs.size // n_dom
    assert n_pH * n_dom == pHs.size

    ext = set(os.path.basename(p) for p in glob.glob(os.path.join(args.ext_dir, "*"))
              if os.path.isfile(p) and not os.path.basename(p).startswith("_"))
    bad = [i for i, did in enumerate(ids)
           if did in ext and net_charge(seqs[i], 7.4) < args.basic_lo]
    print(f"外部域 {sum(1 for i in ids if i in ext)} 个，剔除 charge<{args.basic_lo} 的 {len(bad)} 个")

    keep = [i for i in range(n_dom) if i not in set(bad)]
    keep_mask = np.zeros(pHs.size, dtype=bool)
    for i in keep:
        keep_mask[i * n_pH:(i + 1) * n_pH] = True

    new_ids = np.array([ids[i] for i in keep])
    new_seqs = np.array([seqs[i] for i in keep], dtype=object)
    new_coords = np.array([coords[i] for i in keep], dtype=object)
    new_pH = pHs[keep_mask]; new_charge = charges[keep_mask]; new_pI = pIs[keep_mask]
    np.savez(args.npz, domain_ids=new_ids, seqs=new_seqs, coords=new_coords,
             pH=new_pH, charge=new_charge, pI=new_pI)
    print(f"已重写 {args.npz}：{len(keep)} 域 × {n_pH} pH = {len(new_pH)} 样本")

    # 重算条件向量 μ/σ（域主序，每域 n_pH 连续）
    vecs = []
    for i in range(len(keep)):
        for j in range(n_pH):
            vecs.append(build_condition_vector(
                new_pH[i * n_pH + j], new_charge[i * n_pH + j]))
    vecs = np.stack(vecs)
    mean, std = vecs.mean(axis=0), vecs.std(axis=0)
    print("条件向量 μ:", np.round(mean, 4))
    print("条件向量 σ:", np.round(std, 4))

    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)
    cfg["condition_defaults"]["normalization"]["mean"] = [round(float(x), 4) for x in mean]
    cfg["condition_defaults"]["normalization"]["std"] = [round(float(x), 4) for x in std]
    with open(args.cfg, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"μ/σ 已写入 {args.cfg}")


if __name__ == "__main__":
    main()
