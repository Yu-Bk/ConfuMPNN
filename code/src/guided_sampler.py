"""引导采样器：包装 LigandMPNN decoder，注入 pH 感知的 logit bias。

对应 PROJECT_PLAN.md 4.1 Step 3 的引导采样（Level 1，**不改模型代码**）。

两种使用模式：

1. **静态 bias**：先一次性算出 bias（结构感知过滤器 + 基础 omit/bias），
   填入 `feature_dict["bias"]`，直接调用 LigandMPNN 原版 `model.sample`。
   适用于不依赖已生成序列的规则。

2. **动态逐步解码**：复刻 LigandMPNN 的解码循环（batch=1，非对称/单链），
   每解码一个残基调用 `bias_callback(S_cur, t)` 实时计算该位置 bias。
   支持"可微净电荷 lookahead"（每一步对候选氨基酸做前瞻）与依赖已生成
   序列的实时过滤（如结构感知过滤器按已解码残基统计电荷聚集）。

关键机制（见 model_utils.py `sample`）：
    probs = softmax((logits + bias_t) / temperature)
即 bias 直接加在 logits 上，shape [B, L, 21]（前 20 个 AA + X）。
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---- 让本模块能独立 import LigandMPNN 的 model_utils ----
_LIGAND_MPNN_DIR = Path(__file__).resolve().parents[2] / "LigandMPNN"
if str(_LIGAND_MPNN_DIR) not in sys.path:
    sys.path.insert(0, str(_LIGAND_MPNN_DIR))

from model_utils import cat_neighbors_nodes  # noqa: E402

from .pka import AA_TO_IDX  # noqa: E402
from .structure_aware_filter import UNDECODED  # noqa: E402


def extract_calpha_coords(protein_dict):
    """从 parse_PDB 的 protein_dict 提取 Cα 坐标 [L, 3]。

    parse_PDB 的 X 字段是 [L, 4, 3]，顺序为 N, CA, C, O。
    """
    X = protein_dict["X"]
    if torch.is_tensor(X):
        X = X.cpu().numpy()
    return X[:, 1, :]  # CA 索引 = 1


def build_static_bias(feature_dict, structure_filter, seq_ref=None, base_bias=None):
    """预计算静态 bias = 基础 bias（omit/偏置）+ 结构过滤 bias。

    参数:
        feature_dict: featurize 后的字典（用其 X 的长度和 base bias）
        structure_filter: StructureAwareFilter 实例
        seq_ref: [L] 参考序列（int）；None 时按"全未解码"计算（仅结构规则，
                 不含电荷聚集，因为序列未知）
        base_bias: [1, L, 21] 基础 bias（如 omit_AA）；None 时用全零

    返回:
        (bias[1, L, 21], info_dict)
    """
    X = feature_dict["X"]
    L = X.shape[1] if X.ndim == 4 else X.shape[0]
    if seq_ref is None:
        seq_ref = np.full(L, UNDECODED, dtype=np.int64)
    fb, info = structure_filter.compute_bias(seq_ref)

    if base_bias is None:
        base_bias = torch.zeros(1, L, 21, dtype=torch.float32)
    return base_bias + fb.unsqueeze(0), info


def guided_sample(model, feature_dict, bias_callback=None, device="cpu", encoded=None):
    """引导采样主函数（batch=1，非对称解码）。

    参数:
        model: LigandMPNN 模型（ProteinMPNN 实例）
        feature_dict: featurize 后的字典，需含 X/Y/S/mask/chain_mask/randn/
                      temperature/bias（bias 提供基础偏置，如 omit）
        bias_callback: callable(S_cur, t) -> [21] 或 None
            S_cur: [L] 当前部分解码序列（int，20=X 未解码）
            t: 当前解码位置
            返回该位置的 bias 增量（shape [21]，加到 logits）
        device: 计算设备
        encoded: (h_V, h_E, E_idx) 预编码结果（可选）。给定则跳过 encode，
            用于条件注入（Phase 3，h_V 已含 soft prompt 信号）。

    返回:
        dict {S, sampling_probs, log_probs, decoding_order}
    """
    model = model.to(device)
    base_bias = feature_dict["bias"].to(device).float()  # [1, L, 21]
    S_true = feature_dict["S"].to(device)
    mask = feature_dict["mask"].to(device)
    chain_mask = feature_dict["chain_mask"].to(device)
    temperature = float(feature_dict["temperature"])
    randn = feature_dict["randn"].to(device)

    B, L = S_true.shape
    assert B == 1, "guided_sample 目前只支持 batch_size=1"

    if encoded is not None:
        h_V, h_E, E_idx = encoded
    else:
        h_V, h_E, E_idx = model.encode(feature_dict)

    chain_mask = mask * chain_mask
    decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))

    # 非对称（单链）解码的注意力掩码
    permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=L).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(L, L, device=device))),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([B, L, 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    B_decoder = 1
    S_true = S_true.repeat(B_decoder, 1)
    h_V = h_V.repeat(B_decoder, 1, 1)
    h_E = h_E.repeat(B_decoder, 1, 1, 1)
    chain_mask = chain_mask.repeat(B_decoder, 1)
    mask = mask.repeat(B_decoder, 1)
    base_bias = base_bias.repeat(B_decoder, 1, 1)

    all_probs = torch.zeros((B_decoder, L, 20), device=device)
    all_log_probs = torch.zeros((B_decoder, L, 21), device=device)
    h_S = torch.zeros_like(h_V)
    S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
    h_V_stack = [h_V] + [
        torch.zeros_like(h_V, device=device) for _ in range(len(model.decoder_layers))
    ]

    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    for t_ in range(L):
        t = decoding_order[:, t_]  # [B_decoder]
        chain_mask_t = torch.gather(chain_mask, 1, t[:, None])[:, 0]
        mask_t = torch.gather(mask, 1, t[:, None])[:, 0]

        # ---- 本位置 bias：动态回调（优先）或基础 bias ----
        if bias_callback is not None:
            S_cur = S[0].cpu().numpy()  # [L] 当前部分序列
            dyn = np.asarray(bias_callback(S_cur, int(t[0])), dtype=np.float32)
            bias_t = (base_bias[0, int(t[0])] + torch.from_numpy(dyn).to(device)).view(1, -1)
        else:
            bias_t = torch.gather(base_bias, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]

        E_idx_t = torch.gather(
            E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
        )
        h_E_t = torch.gather(
            h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1])
        )
        h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
        h_EXV_encoder_t = torch.gather(
            h_EXV_encoder_fw,
            1,
            t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2],
                                          h_EXV_encoder_fw.shape[-1]),
        )
        mask_bw_t = torch.gather(
            mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2],
                                                      mask_bw.shape[-1])
        )

        for l, layer in enumerate(model.decoder_layers):
            h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
            h_V_t = torch.gather(
                h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1])
            )
            h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
            h_V_stack[l + 1].scatter_(
                1,
                t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                layer(h_V_t, h_ESV_t, mask_V=mask_t),
            )

        h_V_t = torch.gather(
            h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1])
        )[:, 0]
        logits = model.W_out(h_V_t)  # [B_decoder, 21]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax((logits + bias_t) / temperature, dim=-1)
        probs_sample = probs[:, :20] / torch.sum(probs[:, :20], dim=-1, keepdim=True)
        S_t = torch.multinomial(probs_sample, 1)[:, 0]

        all_probs.scatter_(
            1,
            t[:, None, None].repeat(1, 1, 20),
            (chain_mask_t[:, None, None] * probs_sample[:, None, :]).float(),
        )
        all_log_probs.scatter_(
            1,
            t[:, None, None].repeat(1, 1, 21),
            (chain_mask_t[:, None, None] * log_probs[:, None, :]).float(),
        )
        S_true_t = torch.gather(S_true, 1, t[:, None])[:, 0]
        S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
        h_S.scatter_(
            1,
            t[:, None, None].repeat(1, 1, h_S.shape[-1]),
            model.W_s(S_t)[:, None, :],
        )
        S.scatter_(1, t[:, None], S_t[:, None])

    return {
        "S": S,
        "sampling_probs": all_probs,
        "log_probs": all_log_probs,
        "decoding_order": decoding_order,
    }


class GuidedSampler:
    """引导采样器（高层封装）。

    用法:
        sampler = GuidedSampler(model)
        out = sampler.sample(feature_dict)                    # 静态 bias
        out = sampler.sample(feature_dict, bias_callback=fn)  # 动态逐步 bias
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, feature_dict, bias_callback=None):
        """运行引导采样。

        feature_dict 需已包含基础 bias 键（"bias"，见 LigandMPNN run.py）。
        若 bias_callback 为 None，则行为等价于原版 model.sample（但走本模块
        解码循环）；传入回调即可实现每步实时 bias。
        """
        fd = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in feature_dict.items()
        }
        return guided_sample(
            self.model, fd, bias_callback=bias_callback, device=self.device
        )


if __name__ == "__main__":
    # 自检：只验证辅助函数（build_static_bias）的维度，不依赖真实模型
    from .structure_aware_filter import StructureAwareFilter

    L = 10
    fake_X = torch.randn(1, L, 4, 3)  # [B, L, 4, 3]
    coords = fake_X[0, :, 1].numpy()
    filt = StructureAwareFilter(coords)
    fd = {"X": fake_X}
    bias, info = build_static_bias(fd, filt, seq_ref=None)
    print("static bias shape:", tuple(bias.shape), "(expect [1, 10, 21])")
    print("filter info:", info)
