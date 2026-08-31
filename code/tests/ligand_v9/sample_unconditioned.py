"""v12.2 Tm/Sol 无条件基线采样：对泛化 10 蛋白不加电荷条件生成 n 条序列。

作用：分离"电荷条件化的物理性质代价" vs "逆折叠本身的固有代价"。
- 无条件基线：net_charge=None（make_condition_vector 只填 pH，电荷位全 0）
- 与泛化验证同配置（同 checkpoint/weights/backbone/temperature/seed_base），
  仅 cond_vec 的 net_charge 为 None。

输出：output/tm_sol_v12_2/uncond/<PDB>/seqs.fa（n 条，>seed_XXXX 头）

用法（项目根）：
  PYTHONPATH=code python code/tests/ligand_v9/sample_unconditioned.py \
      --manifest data/validation_pdbs/validation_manifest.json --out_dir output/tm_sol_v12_2/uncond \
      --cond_encoder output/finetune_v12_2/finetune_epoch030.pt \
      --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
      --n 30 --device cuda:5 --pH 7.4
"""
import argparse
import json
import sys
from pathlib import Path

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
from run_guided import load_model, load_condition_encoder, seq_to_string  # noqa: E402

# 无条件占位电荷 = 训练集电荷维度均值（v12.2 checkpoint mean[2]）
CHARGE_MEAN = 1.4243


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cond_encoder", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed_base", type=int, default=2000)
    ap.add_argument("--device", default="cuda:5")
    ap.add_argument("--pH", type=float, default=7.4)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    device = torch.device(args.device)
    man = json.load(open(args.manifest))
    enc = load_condition_encoder(args.cond_encoder, device)
    model = load_model(args.weights, device, model_type="auto")
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for it in man["items"]:
        pdb, path = it["pdb"], str(_PROJECT_DIR / it["path"])
        try:
            protein_dict, *_ = parse_PDB(path)
        except Exception as e:
            print(f"  !! {pdb} parse 失败: {e}")
            continue
        L = protein_dict["X"].shape[0]
        protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
        fd = featurize(protein_dict, cutoff_for_score=8.0,
                       model_type="protein_mpnn", use_atom_context=False,
                       number_of_ligand_atoms=0)
        fd["batch_size"] = 1
        fd["temperature"] = args.temperature
        fd["bias"] = torch.zeros(1, L, 21)

        out_dir = out_root / pdb
        out_dir.mkdir(parents=True, exist_ok=True)
        fa = out_dir / "seqs.fa"
        if fa.exists():
            print(f"  skip {pdb}（已存在 {fa}）")
            continue

        # 无条件占位符 = 训练均值（has_charge=1 + 值=训练均值 1.4243）。
        # ⚠️ 不能用 net_charge=None（has_c=0+值0）——分布外输入导致 poly-G 退化（2026-08-31 实证）。
        cond_vec = make_condition_vector(args.pH, net_charge=CHARGE_MEAN)  # ← 无条件占位
        with open(fa, "w") as f:
            for k in range(args.n):
                torch.manual_seed(args.seed_base + k)
                fd["randn"] = torch.randn(1, L)
                out = conditioned_sample(model, enc, fd, cond_vec, device)
                seq = seq_to_string(out["S"][0].cpu().numpy())
                f.write(f">seed_{args.seed_base + k} uncond\n{seq}\n")
        print(f"{pdb:6s} L={L} -> {fa}（{args.n} 条无条件）", flush=True)


if __name__ == "__main__":
    main()
