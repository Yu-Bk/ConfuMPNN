"""Stage E0 前向验证：MoMPNN 权重（纯 backbone）能否实际生成序列。

把 MoMPNN 权重加载进 model_type='protein_mpnn' 的 ProteinMPNN，
用 1BC8.pdb + featurize(use_atom_context=False) 跑一次 GuidedSampler 采样。

用法（confumpnn 环境）：
  conda run -n confumpnn python /tmp/mompnn_forward_test.py
"""

import sys

import torch

sys.path.insert(0, "/data/nfs/IC/baokun_yu/ConfuMPNN/LigandMPNN")
sys.path.insert(0, "/data/nfs/IC/baokun_yu/ConfuMPNN/code")
from data_utils import featurize, parse_PDB, restype_int_to_str  # noqa: E402
from model_utils import ProteinMPNN  # noqa: E402
from src.guided_sampler import GuidedSampler  # noqa: E402

CKPT = "/data/nfs/IC/baokun_yu/ConfuMPNN/MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
PDB = "/data/nfs/IC/baokun_yu/ConfuMPNN/code/input/1BC8.pdb"

# 1. 加载 MoMPNN 权重 → protein_mpnn 模式（strict=True）
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
model = ProteinMPNN(
    node_features=128, edge_features=128, hidden_dim=128,
    num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48,
    device="cpu", atom_context_num=0, model_type="protein_mpnn",
    ligand_mpnn_use_side_chain_context=False,
)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()
print(f"[1] 模型加载完成: {CKPT.split('/')[-1]}  (strict=True)")

# 2. 读 PDB + featurize（纯 backbone，无配体上下文）
protein_dict, _, _, _, _ = parse_PDB(PDB)
protein_dict["chain_mask"] = torch.ones(
    protein_dict["X"].shape[0], dtype=torch.int32
)
feature_dict = featurize(
    protein_dict, cutoff_for_score=8.0, use_atom_context=False,
    number_of_ligand_atoms=0, model_type="protein_mpnn",
)
L = feature_dict["X"].shape[1]
feature_dict["batch_size"] = 1
feature_dict["temperature"] = 0.3
feature_dict["bias"] = torch.zeros(1, L, 21)
native = "".join(restype_int_to_str[j] for j in feature_dict["S"][0].cpu().numpy())
print(f"[2] 1BC8.pdb 长度 {L}，native: {native[:50]}...")

# 3. 前向采样 2 条序列
torch.manual_seed(111)
sampler = GuidedSampler(model, device="cpu")
for i in range(2):
    feature_dict["randn"] = torch.randn(1, L)
    out = sampler.sample(feature_dict)
    S = out["S"][0].cpu().numpy()
    seq = "".join(restype_int_to_str[j] for j in S)
    # 序列恢复率（与 native 一致的比例）
    rec = float((S == feature_dict["S"][0].cpu().numpy()).mean())
    print(f"  [{i+1}] seq_rec={rec:.3f}  {seq}")
print("[3] 前向采样完成 ✅ MoMPNN 权重可正常生成序列")
