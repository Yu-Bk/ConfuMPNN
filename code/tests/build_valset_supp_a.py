"""构建 v12.3 验证集"长蛋白/深负补充"标签（轨 A，2026-09-04）。

背景（session/2026-09-04_valset_build.md §轨A）：
- 现有 CATH 15% hold-out(1176) 对 v12.2/v12.3 都是真未见，但相对 v12.3 训练(6580)
  缺 L≥400（2.72% vs 8.97%）与深负 q≤-20（0.68% vs 1.96%）代表性。
- 内部 S40 全池(dompdb, 34653)中筛选真未见（不在 v12.2train/v12.3train/holdout/
  测试蛋白 PDB 前缀）且 parse 后 L≥400 的域 12 个 + 深负 boundary 3 个 = 15 个。
- 本脚本把这 15 域按 labels npz 同构格式（domain_ids/seqs/coords(Cα)/pH/charge/pI，
  每域 8 pH Uniform[4,10]）另存 supplement npz。

用法（项目根，CPU）：
  PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/build_valset_supp_a.py \
     --manifest /tmp/supp_manifest_a.json \
     --out data/cath/labels_v12_3_valsupp_a.npz
输出：data/cath/labels_v12_3_valsupp_a.npz（15 域 × 8 pH = 120 样本）
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "LigandMPNN"))
from data_utils import parse_PDB  # noqa: E402
from run_guided import seq_to_string  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.isoelectric_point import find_pI  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--manifest", required=True)
AP.add_argument("--out", default="data/cath/labels_v12_3_valsupp_a.npz")
AP.add_argument("--seed", type=int, default=7)
AP.add_argument("--n_pH", type=int, default=8)
ARGS = AP.parse_args()

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# 候选域 .pdb 来源（与 validate_holdout 同款 parse_PDB 一致）：
#   src=dompdb      → data/cath/S40/dompdb_valsupp/<id>.pdb（本脚本配套生成）
#   src=dompdb_pdb  → data/cath/S40/dompdb_pdb/<id>.pdb（已存在）
DOMDB_VALSUPP = os.path.join(PROJ, "data/cath/S40/dompdb_valsupp")
DOMPDB_PDB = os.path.join(PROJ, "data/cath/S40/dompdb_pdb")


def extract_via_parse_pdb(p):
    """用与验证完全相同的 parse_PDB 提取 (Cα 坐标[L,3], 序列, L)。

    这样 supplement npz 里存的 seq/charge 与 valcurve 运行时 parse_PDB 看到
    的残基集合完全一致，避免 parse_domain(CA行计数) 与 parse_PDB 因残基命名/
    骨架不完整导致的长度不一致 → 电荷 target 错配。
    """
    protein_dict, *_ = parse_PDB(p, device="cpu", parse_all_atoms=False)
    L = int(protein_dict["X"].shape[0])
    seq = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
    xyz = protein_dict["X"].detach().cpu().numpy()  # [L, 3] Cα
    return np.asarray(xyz, dtype=np.float32), seq, L


def main():
    cands = json.load(open(ARGS.manifest))
    rng = np.random.RandomState(ARGS.seed)
    tmpd = tempfile.mkdtemp(prefix="supp_a_")
    domains, seqs, coords, pHs, charges, pIs = [], [], [], [], [], []
    n_ok = 0
    for c in cands:
        did = c["id"]
        if c.get("src", "dompdb") == "dompdb_pdb":
            p = os.path.join(DOMPDB_PDB, did + ".pdb")
        else:
            p = os.path.join(DOMDB_VALSUPP, did + ".pdb")
        if not os.path.exists(p):
            print(f"!! {did} 无 .pdb（{p}），跳过")
            continue
        try:
            coords_i, seq, L = extract_via_parse_pdb(p)
        except Exception as e:
            print(f"!! {did} parse_PDB 失败: {str(e)[:80]}，跳过")
            continue
        q74 = net_charge(seq, 7.4)
        # 质控：L 与 manifest（parse_PDB 口径）应一致
        flag = "OK" if abs(L - c["L"]) <= 2 else f"⚠ L={L} vs manifest {c['L']}"
        print(f"  {did}: parse_PDB L={L} q7.4={q74:.2f} {flag}")
        if L < 20:
            continue
        pH_i = rng.uniform(4.0, 10.0, ARGS.n_pH)
        charge_i = np.array([net_charge(seq, ph) for ph in pH_i], dtype=np.float32)
        pI = find_pI(seq)
        domains.append(did); seqs.append(seq); coords.append(coords_i)
        pHs.append(pH_i); charges.append(charge_i)
        pIs.append(np.full(ARGS.n_pH, pI, dtype=np.float32))
        n_ok += 1
    shutil.rmtree(tmpd, ignore_errors=True)
    np.savez(ARGS.out,
             domain_ids=np.array(domains),
             seqs=np.array(seqs, dtype=object),
             coords=np.array(coords, dtype=object),
             pH=np.concatenate(pHs),
             charge=np.concatenate(charges),
             pI=np.concatenate(pIs))
    print(f"已写 {ARGS.out}（{n_ok} 域 × {ARGS.n_pH} = {n_ok * ARGS.n_pH} 样本）")


if __name__ == "__main__":
    main()
