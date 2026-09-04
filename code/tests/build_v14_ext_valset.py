"""构建 v14 配体"外部未见"验证集标签 npz（轨 B，2026-09-04，闸口前产物）。

候选来源：small_mol/metal/nucleotide ← data/ligand_train/ext_smallmol_raw/（RCSB 新下载，
  已序列去重 vs 5371 训练 + in-10 测试，coverage in/boundary）；RNA/DNA ← data/ligand_train/rna_pdbs_ext/
  （本地拆链 196 未见链中按配额抽 7）。
每个候选 parse_PDB QC（单链/L≤500/Y 配体存在）；本脚本按与训练标签同构格式（domain_ids/
  seqs/coords/pH/charge/pI，每域 8 pH Uniform[4,10]）另存。

用法（项目根，CPU）：
  PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/build_v14_ext_valset.py
输出：data/ligand_train/labels_v14_ext_valset.npz（30 域 × 8 pH = 240 样本）
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, os.path.join(_CODE_DIR, "..", "..", "LigandMPNN"))
from data_utils import parse_PDB  # noqa: E402
from run_guided import seq_to_string  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.isoelectric_point import find_pI  # noqa: E402

MANIFEST = "/tmp/v14_ext_manifest.json"
PDB_DIR = "data/ligand_train/v14_ext_valset_pdb"
OUT = "data/ligand_train/labels_v14_ext_valset.npz"
SEED = 11


def main():
    items = json.load(open(MANIFEST))
    rng = np.random.RandomState(SEED)
    domains, seqs, coords, pHs, charges, pIs = [], [], [], [], [], []
    out = []
    for it in items:
        did = it["id"]
        p = os.path.join(PDB_DIR, did + ".pdb")
        try:
            pd, *_ = parse_PDB(p, device="cpu", parse_all_atoms=False)
            L = int(pd["X"].shape[0])
            seq = seq_to_string(pd["S"].reshape(-1).cpu().numpy())
            Y = pd.get("Y")
            ny = Y.numel() // 3 if Y is not None else 0
            xyz = np.asarray(pd["X"].detach().cpu().numpy(), dtype=np.float32)
            q7 = net_charge(seq, 7.4)
            pH_i = rng.uniform(4.0, 10.0, 8)
            charge_i = np.array([net_charge(seq, ph) for ph in pH_i], dtype=np.float32)
            pI = find_pI(seq)
            print(f"  {did} [{it['type']}] L={L} nY={ny} q7.4={q7:.2f}")
            domains.append(did); seqs.append(seq); coords.append(xyz)
            pHs.append(pH_i); charges.append(charge_i)
            pIs.append(np.full(8, pI, dtype=np.float32))
            out.append({**it, "L": int(L), "q7.4": round(float(q7), 2), "nY": int(ny)})
        except Exception as e:
            print(f"  !! {did} parse 失败跳过: {str(e)[:100]}")
    np.savez(OUT, domain_ids=np.array(domains), seqs=np.array(seqs, dtype=object),
             coords=np.array(coords, dtype=object), pH=np.concatenate(pHs),
             charge=np.concatenate(charges), pI=np.concatenate(pIs))
    json.dump(out, open("/tmp/v14_ext_built.json", "w"), indent=1)
    print(f"已写 {OUT}（{len(domains)} 域 × 8 = {len(domains) * 8} 样本）")


if __name__ == "__main__":
    main()
