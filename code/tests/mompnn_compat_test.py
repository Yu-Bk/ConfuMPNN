"""Stage E0 核心测试：MoMPNN 权重能否 load_state_dict 进 LigandMPNN 的 ProteinMPNN 类。

对 8 个 MoMPNN .ckpt（PyTorch Lightning checkpoint）逐一测试：
  A) model_type='protein_mpnn'（纯 backbone）→ strict=True 能否直接通过
  B) model_type='ligand_mpnn'（配体模式，我们管线现状）→ 统计 missing / unexpected keys

用法（confumpnn 环境）：
  conda run -n confumpnn python /tmp/mompnn_compat_test.py
"""

import glob
import os
import sys

import torch

sys.path.insert(0, "/data/nfs/IC/baokun_yu/ConfuMPNN/LigandMPNN")
from model_utils import ProteinMPNN  # noqa: E402

CKPT_DIR = "/data/nfs/IC/baokun_yu/ConfuMPNN/MoMPNN/mompnn_paper_checkpoints"
FILES = sorted(glob.glob(os.path.join(CKPT_DIR, "*.ckpt")))
print(f"共 {len(FILES)} 个权重文件\n")


def build(model_type, atom_context_num=0, k_neighbors=48):
    return ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device="cpu",
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=False,
    )


for f in FILES:
    name = os.path.basename(f)
    ckpt = torch.load(f, map_location="cpu", weights_only=False)
    top_keys = list(ckpt.keys())
    sd = ckpt["model_state_dict"]  # 与 LigandMPNN 官方格式一致
    num_edges = ckpt.get("num_edges")
    dtypes = {str(v.dtype) for v in sd.values()}

    # ---- A) protein_mpnn 模式：strict=True ----
    m = build("protein_mpnn")
    try:
        m.load_state_dict(sd, strict=True)
        res_a = "PASS(strict=True)"
    except Exception as e:
        # 失败时回退 strict=False 看具体差异
        miss, unexp = m.load_state_dict(sd, strict=False)
        res_a = f"FAIL: missing={len(list(miss))} unexpected={len(list(unexp))} | {str(e)[:50]}"

    # ---- B) ligand_mpnn 模式：strict=False 统计差异 ----
    m2 = build("ligand_mpnn", atom_context_num=16)
    miss, unexp = m2.load_state_dict(sd, strict=False)
    miss, unexp = list(miss), list(unexp)

    print(f"### {name}")
    print(f"  tensors={len(sd)} dtypes={dtypes} num_edges={num_edges} ckpt_top_keys={top_keys[:6]}")
    print(f"  [A] protein_mpnn: {res_a}")
    print(f"  [B] ligand_mpnn : missing={len(miss)} unexpected={len(unexp)}")
    if miss:
        print(f"      missing 示例: {miss[:6]}")
    if unexp:
        print(f"      unexpected 示例: {unexp[:6]}")
    print()
