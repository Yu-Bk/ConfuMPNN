"""阶段 2：为选定的 805 候选生成 labels npz + 结构入库 + spec json（轨 B，2026-09-04）。

输入：/tmp/v14_805_selected.json（含 id/type/L/q/coverage/n_close）
结构源：data/ligand_train/ext_smallmol_raw/<id>.pdb（small_mol/metal/nucleotide）或
        data/ligand_train/rna_pdbs_ext/<id>.pdb（RNA/DNA）
输出：
  data/ligand_train/labels_v14_valset_805.npz  （与训练同构 domain_ids/seqs/coords(Cα)/pH×8/charge/pI）
  data/ligand_train/v14_valset_pdb/<id>.pdb    （拷贝结构）
  data/ligand_train/v14_valset_805_spec.json   （类型/电荷/长度分布 + coverage + 来源/去重/下载记录）
"""
import json
import os
import shutil
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "LigandMPNN"))
from data_utils import parse_PDB  # noqa: E402
from run_guided import seq_to_string  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.isoelectric_point import find_pI  # noqa: E402

SELECTED = "/tmp/v14_805_selected.json"
RAW_DIR = "data/ligand_train/ext_smallmol_raw"
RNA_DIR = "data/ligand_train/rna_pdbs_ext"
LONG_DIR = "data/ligand_train/ext_longlig_raw"
PDB_OUT = "data/ligand_train/v14_valset_pdb"
NPZ_OUT = "data/ligand_train/labels_v14_valset_805.npz"
SPEC_OUT = "data/ligand_train/v14_valset_805_spec.json"
SEED = 2026


def pdb_path_of(did):
    for base in (RAW_DIR, RNA_DIR, LONG_DIR):
        p = os.path.join(base, did + ".pdb")
        if os.path.exists(p):
            return p
    return None


def build_one(item):
    did = item["id"]
    p = pdb_path_of(did)
    if p is None:
        return None
    try:
        pd, *_ = parse_PDB(p, device="cpu", parse_all_atoms=False)
        L = int(pd["X"].shape[0])
        seq = seq_to_string(pd["S"].reshape(-1).cpu().numpy())
        X = pd["X"].detach().cpu().numpy()
        if X.ndim == 3:
            coords = np.ascontiguousarray(X[:, 1, :], dtype=np.float32)  # Cα
        else:
            coords = np.asarray(X, dtype=np.float32)
        return {"id": did, "type": item["type"], "L": L, "seq": seq,
                "coords": coords, "q7": round(float(net_charge(seq, 7.4)), 2),
                "coverage": item.get("coverage"), "n_close": item.get("n_close"),
                "src": "rna_pdbs_ext" if p.startswith(RNA_DIR) else "rcsb_download"}
    except Exception as e:
        print("  !!", did, "parse fail", str(e)[:80])
        return None


def main():
    items = json.load(open(SELECTED))
    os.makedirs(PDB_OUT, exist_ok=True)
    with Pool(8) as pool:
        res = [r for r in pool.map(build_one, items) if r is not None]
    print("built ok:", len(res), "/", len(items))
    res.sort(key=lambda r: (r["type"], r["L"]))
    # copy structures
    for r in res:
        src = pdb_path_of(r["id"])
        shutil.copy(src, os.path.join(PDB_OUT, r["id"] + ".pdb"))
    # pH/charge/pI (uniform 4-10, fixed seed)
    rng = np.random.RandomState(SEED)
    domains, seqs, coords, pHs, charges, pIs = [], [], [], [], [], []
    for r in res:
        pH_i = rng.uniform(4.0, 10.0, 8)
        charge_i = np.array([net_charge(r["seq"], ph) for ph in pH_i], dtype=np.float32)
        pI = find_pI(r["seq"])
        domains.append(r["id"]); seqs.append(r["seq"]); coords.append(r["coords"])
        pHs.append(pH_i); charges.append(charge_i)
        pIs.append(np.full(8, pI, dtype=np.float32))
    np.savez(NPZ_OUT, domain_ids=np.array(domains), seqs=np.array(seqs, dtype=object),
             coords=np.array(coords, dtype=object), pH=np.concatenate(pHs),
             charge=np.concatenate(charges), pI=np.concatenate(pIs))
    print("wrote", NPZ_OUT, len(domains), "domains")
    # spec
    spec = {
        "title": "v14 外部未见验证集 805（训练 15% 比例）",
        "date": "2026-09-04",
        "n": len(res),
        "quota": dict(Counter(r["type"] for r in res)),
        "coverage": dict(Counter(r["coverage"] for r in res)),
        "L_quantiles": [int(np.percentile([r["L"] for r in res], p)) for p in (0, 25, 50, 75, 100)],
        "q7_quantiles": [round(float(np.percentile([r["q7"] for r in res], p)), 1) for p in (0, 25, 50, 75, 100)],
        "dedup": "精确序列去重 vs labels_v14_final(5371) + in-10 十码 + 1A65(+2E9R_X)；池内按 code 去重",
        "source": "同源候选池(candidates.json+candidates2.json 6302 未用码)下载 4550 个→parse pass 900→去重/coverage 后配额选满 805；RNA/DNA 用本地 rna_pdbs_ext 拆链(208 拆出/196 未见)",
        "reclass": "nucleotide 按训练语义含 NAD/ADP/FAD/FMN/GDP 等核苷酸辅因子重分类；metal=仅金属离子配体",
        "structure_dir": "data/ligand_train/v14_valset_pdb/",
        "labels_npz": "data/ligand_train/labels_v14_valset_805.npz",
        "per_type": {},
    }
    for t in ("small_mol", "metal", "nucleotide", "RNA/DNA"):
        sub = [r for r in res if r["type"] == t]
        spec["per_type"][t] = {
            "n": len(sub),
            "L_med": int(np.median([r["L"] for r in sub])) if sub else None,
            "q_med": round(float(np.median([r["q7"] for r in sub])), 2) if sub else None,
            "q_le_-20": sum(1 for r in sub if r["q7"] <= -20),
            "q_gt_10": sum(1 for r in sub if r["q7"] > 10),
        }
    json.dump(spec, open(SPEC_OUT, "w"), indent=1, ensure_ascii=False)
    print("wrote", SPEC_OUT)
    # 便于复核的轻量清单（不含 coords/seq 大字段）
    slim = [{k: r[k] for k in ("id", "type", "L", "q7", "coverage", "n_close", "src")} for r in res]
    json.dump(slim, open("/tmp/v14_805_built.json", "w"), indent=1)


if __name__ == "__main__":
    main()
