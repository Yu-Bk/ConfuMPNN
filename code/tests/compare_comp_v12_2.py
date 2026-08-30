"""v12.2 组成分析：5 蛋白生成序列 D/E/K/R 计数 vs native（治删减/过度添加）。

复用 v10_diag_response_curve.py 的采样逻辑（同 checkpoint、同 n/temp/seed），
统计每条生成序列的带电残基总数，对比 native——与 v12.1 报告 §1 的口径一致。

用法（项目根）：
  PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/compare_comp_v12_2.py
输出：终端表 + output/v12_2_comp.json
"""
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_DIR = next(p for p in Path(__file__).resolve().parents
                    if (p / "code").is_dir() and (p / "LigandMPNN").is_dir())
_CODE_DIR = _PROJECT_DIR / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_PROJECT_DIR / "LigandMPNN"))

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from run_guided import load_model, load_condition_encoder, seq_to_string  # noqa: E402

# 与 v12.1 组成分析同款 5 蛋白
PROTEINS = [
    ("1C6O",   "data/validation_pdbs/1C6O.pdb"),
    ("1AG0",   "data/validation_pdbs/1AG0.pdb"),
    ("1BJ4",   "data/validation_pdbs/1BJ4.pdb"),
    ("1A65",   "data/validation_pdbs/1A65.pdb"),
    ("7pujA01", "data/cath/S40/dompdb_pdb/7pujA01.pdb"),
]
ENC = str(_PROJECT_DIR / "output/finetune_v12_2/finetune_epoch030.pt")
WTS = str(_PROJECT_DIR / "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt")
N, TEMP, SEED_BASE, PH = 30, 0.3, 3000, 7.4
AA = "ACDEFGHIKLMNPQRSTVWY"


def count_dk(seq):
    """带电残基总数 = D/E + K/R（与 v12.1 报告口径一致）。"""
    de = sum(1 for a in seq if a in "DE")
    kr = sum(1 for a in seq if a in "KR")
    return de + kr, de, kr


def main():
    device = torch.device("cuda:4")
    enc = load_condition_encoder(ENC, device)
    model = load_model(WTS, device, model_type="auto")
    feats = dict(model_type="protein_mpnn", use_atom_context=False, number_of_ligand_atoms=0)

    print(f"{'name':8s} {'L':>4s} {'native_DK':>9s} {'v12.2_DK':>9s} {'倍率':>5s} {'native_DE/KR':>12s} {'v12.2_DE/KR':>12s}", flush=True)
    results = {}
    for name, pdb_path in PROTEINS:
        protein_dict, *_ = parse_PDB(pdb_path)
        L = protein_dict["X"].shape[0]
        native = seq_to_string(protein_dict["S"].reshape(-1).cpu().numpy())
        q_nat = round(float(net_charge(native, PH)))
        nat_dk, nat_de, nat_kr = count_dk(native)

        protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
        fd = featurize(protein_dict, cutoff_for_score=8.0, **feats)
        fd["batch_size"] = 1
        fd["temperature"] = TEMP
        fd["bias"] = torch.zeros(1, L, 21)

        gen_dk, gen_de, gen_kr = [], [], []
        for k in range(N):
            torch.manual_seed(SEED_BASE + k)
            fd["randn"] = torch.randn(1, L)
            cond_vec = make_condition_vector(PH, net_charge=q_nat)
            out = conditioned_sample(model, enc, fd, cond_vec, device)
            seq = seq_to_string(out["S"][0].cpu().numpy())
            dk, de, kr = count_dk(seq)
            gen_dk.append(dk); gen_de.append(de); gen_kr.append(kr)
        m_dk = float(np.mean(gen_dk)); m_de = float(np.mean(gen_de)); m_kr = float(np.mean(gen_kr))
        ratio = m_dk / nat_dk if nat_dk else float("nan")
        print(f"{name:8s} {L:4d} {nat_dk:9d} {m_dk:9.1f} {ratio:5.2f} "
              f"{nat_de}/{nat_kr:>6d} {m_de:7.1f}/{m_kr:6.1f}", flush=True)
        results[name] = {"L": L, "native_dk": nat_dk, "native_de": nat_de, "native_kr": nat_kr,
                         "v12_2_dk": m_dk, "v12_2_de": m_de, "v12_2_kr": m_kr, "ratio": ratio}

    with open(str(_PROJECT_DIR / "output/v12_2_comp.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n已写 output/v12_2_comp.json", flush=True)


if __name__ == "__main__":
    import json  # noqa: E402
    main()
