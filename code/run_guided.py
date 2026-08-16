"""一键运行引导采样（Phase 1 主入口）。

把整条管线串起来：
    PDB → 骨架/配体上下文 → 结构感知过滤器 + 动态电荷前瞻 → 引导采样
    → 生成 N 条候选序列 → 计算每条净电荷/pI → 输出统计

用法（在 code/ 目录下）：
    conda activate confumpnn
    python run_guided.py --pdb input/1BC8.pdb --pH 7.4 [--target_charge -2.0]
                         [--preset default] [--num_samples 10]
    # 用 MoMPNN 权重（纯 backbone，--model_type 会自动检测为 protein_mpnn）：
    python run_guided.py --pdb input/1BC8.pdb --pH 7.4 \
        --weights ../MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt

日志建议重定向到 code/log/，输出写入 code/output/。
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ---- 路径设置：code/ 与 LigandMPNN/ ----
_CODE_DIR = Path(__file__).resolve().parent
_LIG_DIR = _CODE_DIR.parent / "LigandMPNN"
for p in [str(_CODE_DIR), str(_LIG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import featurize, parse_PDB, restype_int_to_str  # noqa: E402
from model_utils import ProteinMPNN  # noqa: E402
from src.charge_lookahead import make_dynamic_callback  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.guided_sampler import GuidedSampler, extract_calpha_coords  # noqa: E402
from src.isoelectric_point import find_pI  # noqa: E402
from src.structure_aware_filter import StructureAwareFilter, load_preset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="ConfuMPNN Phase 1 引导采样")
    p.add_argument("--pdb", required=True, help="输入 PDB 路径")
    p.add_argument("--pH", type=float, required=True, help="工作环境 pH")
    p.add_argument("--target_charge", type=float, default=None,
                   help="目标净电荷（None=不引导电荷，只做结构过滤）")
    p.add_argument("--preset", default="default",
                   choices=["default", "nucleic_acid_binding", "membrane", "acidic"],
                   help="结构过滤器场景预设")
    p.add_argument("--num_samples", type=int, default=10, help="生成候选序列数")
    p.add_argument("--temperature", type=float, default=0.3, help="采样温度")
    p.add_argument("--strength", type=float, default=0.5, help="电荷引导强度")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--weights", default=None,
                   help="权重路径（默认 ligandmpnn_v_32_010_25.pt；也可指定 MoMPNN 的 .ckpt）")
    p.add_argument("--model_type", default="auto",
                   choices=["auto", "protein_mpnn", "ligand_mpnn"],
                   help="模型类型：auto=按权重自动检测（默认）；protein_mpnn=纯 backbone（如 MoMPNN）；"
                        "ligand_mpnn=配体上下文（原版 LigandMPNN）")
    p.add_argument("--out_dir", default=None,
                   help="输出目录（默认 code/output/guided_<pdb>_pH<pH>）")
    return p.parse_args()


def load_model(weights, device, model_type="auto"):
    checkpoint = torch.load(weights, map_location=device)
    # 自动检测：权重里有 atom_context_num（且 >0）说明是 LigandMPNN 配体权重；
    # 没有则是纯 backbone ProteinMPNN（如 MoMPNN）。
    if model_type == "auto":
        model_type = (
            "ligand_mpnn" if checkpoint.get("atom_context_num", 0) > 0
            else "protein_mpnn"
        )
    atom_context_num = (
        0 if model_type == "protein_mpnn" else int(checkpoint.get("atom_context_num", 16))
    )
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=int(checkpoint["num_edges"]),
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def seq_to_string(S):
    return "".join(restype_int_to_str[i] for i in S)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    weights = Path(args.weights) if args.weights else (
        _LIG_DIR / "model_params" / "ligandmpnn_v_32_010_25.pt"
    )
    print(f"[1] 加载模型: {weights.name}  (device={device})")
    model = load_model(weights, device, model_type=args.model_type)
    mt = model.model_type  # 解析后的实际模型类型
    print(f"    解析 model_type = {mt}")

    # 2. 读 PDB + featurize（按模型类型决定是否用配体上下文）
    print(f"[2] 读取 PDB: {args.pdb}")
    protein_dict, _, _, _, _ = parse_PDB(args.pdb)
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32  # 默认设计全部残基
    )
    use_atom_context = (mt == "ligand_mpnn")
    feature_dict = featurize(
        protein_dict, cutoff_for_score=8.0,
        use_atom_context=use_atom_context,
        number_of_ligand_atoms=(16 if use_atom_context else 0),
        model_type=mt,
    )
    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = args.temperature
    feature_dict["bias"] = torch.zeros(1, L, 21)
    native_seq = seq_to_string(feature_dict["S"][0].cpu().numpy())
    print(f"    蛋白长度 {L}，native: {native_seq[:50]}...")

    # 3. 结构感知过滤器 + 动态电荷前瞻回调
    print(f"[3] 引导设置: pH={args.pH}, target_charge={args.target_charge}, "
          f"preset={args.preset}, strength={args.strength}")
    coords = extract_calpha_coords(protein_dict)
    structure_filter = StructureAwareFilter(coords, config=load_preset(args.preset))
    bias_callback = make_dynamic_callback(
        pH=args.pH, target_charge=args.target_charge,
        structure_filter=structure_filter, strength=args.strength,
    )

    # 4. 引导采样 N 条
    print(f"[4] 引导采样 {args.num_samples} 条候选序列...")
    sampler = GuidedSampler(model, device=device)
    sequences, charges, pIs = [], [], []
    for i in range(args.num_samples):
        feature_dict["randn"] = torch.randn(1, L)
        out = sampler.sample(feature_dict, bias_callback=bias_callback)
        seq = seq_to_string(out["S"][0].cpu().numpy())
        sequences.append(seq)
        charges.append(net_charge(seq, args.pH))
        pIs.append(find_pI(seq))
        print(f"    [{i+1:2d}] charge={charges[-1]:+6.2f}  pI={pIs[-1]:5.2f}  {seq[:60]}")

    # 5. native 对照
    native_charge = net_charge(native_seq, args.pH)
    native_pI = find_pI(native_seq)
    print(f"[5] native   : charge={native_charge:+6.2f}  pI={native_pI:5.2f}  {native_seq[:60]}")

    # 6. 统计 + 输出
    mean_charge = float(np.mean(charges))
    std_charge = float(np.std(charges))
    print(f"    平均净电荷 = {mean_charge:+.2f} ± {std_charge:.2f}  "
          f"(目标 {args.target_charge})")

    out_dir = Path(args.out_dir) if args.out_dir else (
        _CODE_DIR / "output" / f"guided_{Path(args.pdb).stem}_pH{args.pH}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "seqs.fa"
    with open(fasta_path, "w", encoding="utf-8") as f:
        for i, seq in enumerate(sequences):
            f.write(f">sample_{i+1} pH={args.pH} charge={charges[i]:+.2f} pI={pIs[i]:.2f}\n")
            f.write(seq + "\n")
        f.write(f">native charge={native_charge:+.2f} pI={native_pI:.2f}\n")
        f.write(native_seq + "\n")
    summary = {
        "pdb": args.pdb, "pH": args.pH, "target_charge": args.target_charge,
        "preset": args.preset, "temperature": args.temperature,
        "strength": args.strength, "seed": args.seed, "num_samples": args.num_samples,
        "native_charge": native_charge, "native_pI": native_pI,
        "mean_charge": mean_charge, "std_charge": std_charge,
        "sequences": [
            {"seq": s, "charge": c, "pI": p} for s, c, p in zip(sequences, charges, pIs)
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[6] 输出已保存: {fasta_path}")
    print("完成 ✅")


if __name__ == "__main__":
    main()
