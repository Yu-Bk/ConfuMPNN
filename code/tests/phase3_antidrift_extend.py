"""Phase 3 防失控扩样本：对称配对采样（基线 vs 条件注入，双场景）。

对应 PROJECT_EXTEND 的「可开发性保持」验证。把 Phase 3 防失控从 n≈6~9
的不配对对比（E1b 基线均值 vs 条件注入均值）升级为 **n=20 对称配对设计**，
从而有能力区分「无差异」与「差异太小没检测到」（统计功效）。

设计原则（防过拟合 / 防偏置，四道防线）：
1. **无泄漏**：验证 PDB（1BC8/1CRN/1UBQ/2LZM）均不在训练域列表
   （data/cath/labels.npz 的 999 个 domain_ids）→ 测的是泛化而非记忆。
2. **对称配对**：同一 seed → 同一 randn → 同一解码顺序；基线（无条件注入）
   与条件（finetune_t05 注入）的唯一差异 = 条件本身，采样噪声独立。
3. **固定 seed 协议**：seed = 111+k（k=0..n-1），所有 PDB/场景/臂共用，
   temperature=0.3 一致，杜绝「挑 seed」式选择偏差。
4. **双场景**：A=温和（pH 7.4，target=原生电荷，电荷目标不变，只加 pH 感知）；
   B=压力（pH 4.0，target=原生+5，条件真正改变序列才测副作用，非平凡）。

输出目录：
    {OUT}/{pdb}/{A_base,A_cond,B_base,B_cond}/seqs.fa

用法（code/ 下）：
    PYTHONPATH=. python tests/phase3_antidrift_extend.py \
        --out_dir output/phase3_antidrift_n20 \
        --cond_encoder output/finetune_t05/condition_encoder_last.pt
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR.parent / "LigandMPNN"))

from data_utils import featurize, parse_PDB  # noqa: E402
from src.condition_embedding import make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.guided_sampler import guided_sample  # noqa: E402
from run_guided import _DEFAULT_WEIGHTS, load_condition_encoder, load_model, seq_to_string  # noqa: E402

# 验证 PDB 集合（与 E1b / Phase 3 一致，全部经泄漏检查不在训练集）
PDBS = {
    "1BC8": "input/1BC8_chainC.pdb",
    "1CRN": "input/1CRN.pdb",
    "1UBQ": "input/1UBQ.pdb",
    "2LZM": "input/2LZM.pdb",
}


def build_feature_dict(model, pdb_path, device):
    """与 phase3_pH_response.py 完全一致的 feature_dict 构建。"""
    protein_dict, _, _, _, _ = parse_PDB(pdb_path)
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32
    )
    feature_dict = featurize(
        protein_dict, cutoff_for_score=8.0,
        use_atom_context=False, number_of_ligand_atoms=0,
        model_type="protein_mpnn",
    )
    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = 0.3
    feature_dict["bias"] = torch.zeros(1, L, 21)
    return feature_dict


def sample_pair(model, enc, feature_dict, cond_vec, device, seed):
    """固定 seed 采一对：基线（无注入）+ 条件（注入）。

    同一 randn → 同一解码顺序 → 序列差异只来自条件注入（配对成立）。
    """
    torch.manual_seed(seed)
    L = feature_dict["X"].shape[1]
    feature_dict["randn"] = torch.randn(1, L)          # 同一 randn
    fd = {k: v.to(device) if torch.is_tensor(v) else v
          for k, v in feature_dict.items()}
    base_out = guided_sample(model, fd, device=device, encoded=None)
    base_seq = seq_to_string(base_out["S"][0].cpu().numpy())
    cond_out = conditioned_sample(model, enc, feature_dict, cond_vec, device=device)
    cond_seq = seq_to_string(cond_out["S"][0].cpu().numpy())
    return base_seq, cond_seq


def write_fasta(path, header, seq):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f">{header}\n{seq}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", default=str(_CODE_DIR),
                    help="PDB 相对目录（PDBS 值是相对此目录的路径）")
    ap.add_argument("--cond_encoder", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=20, help="每场景每臂样本数")
    ap.add_argument("--seed_base", type=int, default=111)
    ap.add_argument("--ref_pH", type=float, default=7.4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir)
    weights = args.weights if args.weights else str(_DEFAULT_WEIGHTS)
    model = load_model(weights, device, model_type="protein_mpnn")
    enc = load_condition_encoder(args.cond_encoder, device)

    for pdb, rel in PDBS.items():
        pdb_path = os.path.join(args.pdb_dir, rel)
        fd = build_feature_dict(model, pdb_path, device)
        L = fd["X"].shape[1]
        native_charge = net_charge(
            seq_to_string(fd["S"][0].cpu().numpy()), args.ref_pH)
        tA = round(native_charge)          # 场景 A：保持原生电荷
        tB = round(native_charge) + 5      # 场景 B：压力 +5（pH 4.0）
        print(f"\n=== {pdb}  L={L}  native@{args.ref_pH}={native_charge:+.2f}"
              f"  A:pH{args.ref_pH}/t{tA}  B:pH4.0/t{tB} ===", flush=True)

        scenarios = {
            "A_base": (args.ref_pH, float(tA), False),
            "A_cond": (args.ref_pH, float(tA), True),
            "B_base": (4.0, float(tB), False),
            "B_cond": (4.0, float(tB), True),
        }
        # 清空/重建每臂输出文件
        for arm in scenarios:
            fa = out_root / pdb / arm / "seqs.fa"
            fa.parent.mkdir(parents=True, exist_ok=True)
            if fa.exists():
                fa.unlink()

        # 逐 seed 对称采样（先基后条，同一 randn）
        for k in range(args.n):
            seed = args.seed_base + k
            for arm, (pH, tgt, do_cond) in scenarios.items():
                cond_vec = make_condition_vector(pH, net_charge=tgt)
                base_seq, cond_seq = sample_pair(
                    model, enc, fd, cond_vec, device, seed)
                seq = cond_seq if do_cond else base_seq
                charge = net_charge(seq, pH)
                write_fasta(
                    out_root / pdb / arm / "seqs.fa",
                    f"seed_{seed} arm={'cond' if do_cond else 'base'} "
                    f"scenario={'A' if arm.startswith('A') else 'B'} "
                    f"pH={pH} target={tgt:+.0f} charge={charge:+.2f}",
                    seq,
                )
            if (k + 1) % 5 == 0:
                print(f"  {pdb} seed {k+1}/{args.n} done", flush=True)

    print("\n=== 采样完成 ===")


if __name__ == "__main__":
    main()
