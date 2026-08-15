"""冒烟测试：真实 LigandMPNN 模型 + 1BC8.pdb 上的引导采样。

验证 guided_sampler 在真实模型上能跑通（静态 bias + 动态回调接口）。
运行方式（在 code/ 目录下）：
    conda activate confumpnn
    python tests/smoke_guided.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parents[1]
_LIG_DIR = _CODE_DIR.parent / "LigandMPNN"
for p in [str(_CODE_DIR), str(_LIG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import featurize, parse_PDB, restype_int_to_str  # noqa: E402
from model_utils import ProteinMPNN  # noqa: E402
from src.guided_sampler import GuidedSampler, build_static_bias, extract_calpha_coords  # noqa: E402
from src.structure_aware_filter import StructureAwareFilter, load_preset  # noqa: E402


def main():
    torch.manual_seed(111)
    np.random.seed(111)
    device = torch.device("cpu")

    # 1. 加载 LigandMPNN 权重
    weights = _LIG_DIR / "model_params" / "ligandmpnn_v_32_010_25.pt"
    print(f"[1] 加载模型权重: {weights.name}")
    checkpoint = torch.load(weights, map_location=device)
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=int(checkpoint["num_edges"]),
        device=device,
        atom_context_num=int(checkpoint["atom_context_num"]),
        model_type="ligand_mpnn",
        ligand_mpnn_use_side_chain_context=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"    参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 2. 读 PDB 并 featurize
    pdb_path = _CODE_DIR / "input" / "1BC8.pdb"
    print(f"[2] 读取 PDB: {pdb_path.name}")
    protein_dict, _, _, _, _ = parse_PDB(str(pdb_path))
    # 与 run.py 一致：featurize 前需手动给 protein_dict 加 chain_mask（1=设计）
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32
    )
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=True,
        number_of_ligand_atoms=16,
        model_type="ligand_mpnn",
    )
    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = 0.1
    feature_dict["randn"] = torch.randn(1, L)
    feature_dict["bias"] = torch.zeros(1, L, 21)
    native_seq = "".join(restype_int_to_str[i] for i in feature_dict["S"][0].cpu().numpy())
    print(f"    蛋白长度: {L}, native 序列: {native_seq[:40]}...")

    # 3. 结构感知过滤器 → 静态 bias
    print("[3] 结构感知过滤器（default 预设）")
    coords = extract_calpha_coords(protein_dict)
    filt = StructureAwareFilter(coords, config=load_preset("default"))
    static_bias, info = build_static_bias(feature_dict, filt, seq_ref=None)
    feature_dict["bias"] = static_bias
    n_suppressed = int((static_bias < 0).sum())
    print(f"    过滤器规则触发: {info}")
    print(f"    被抑制的 (位置, 氨基酸) 对数: {n_suppressed}")

    # 4a. 引导采样：静态 bias（直接用构建好的 feature_dict）
    print("[4a] 引导采样（静态 bias）")
    sampler = GuidedSampler(model, device=device)
    out1 = sampler.sample(feature_dict)
    seq1 = "".join(restype_int_to_str[i] for i in out1["S"][0].cpu().numpy())
    print(f"    生成序列: {seq1}")

    # 4b. 引导采样：动态回调接口（示例：每步返回零 bias，仅验证接口可跑）
    def zero_bias(S_cur, t):
        return np.zeros(21, dtype=np.float32)

    print("[4b] 引导采样（动态回调接口，zero bias）")
    out2 = sampler.sample(feature_dict, bias_callback=zero_bias)
    seq2 = "".join(restype_int_to_str[i] for i in out2["S"][0].cpu().numpy())
    print(f"    生成序列: {seq2}")

    # 5. 保存输出到 code/output
    out_path = _CODE_DIR / "output" / "smoke_1BC8_guided.txt"
    out_path.write_text(
        f"native: {native_seq}\nstatic: {seq1}\ndynamic: {seq2}\nfilter: {info}\n",
        encoding="utf-8",
    )
    print(f"[5] 结果已保存: {out_path}")
    print("冒烟测试通过 ✅")


if __name__ == "__main__":
    main()
