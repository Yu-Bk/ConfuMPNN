"""条件注入采样器：用微调后的 ConditionEncoder 把 (pH, target_charge) 注入 backbone。

Phase 3 验证用。与训练脚本 train_finetune.py 使用**同一套 soft prompt 注入机制**
（cross-attention：h_V += softmax(h_V·prompt^T/√d)·prompt），保证训练/推理一致。

对应 PROJECT_PLAN.md 4.5（soft prompt）的实现：字面前缀需重排 E_idx 易错，
改为 cross-attention 注入 encoder 输出 h_V，无需改动解码器。
"""

import math

import torch

from .guided_sampler import guided_sample


def inject_prompt(h_V, prompt_tokens):
    """soft prompt 注入（与 train_finetune.py 完全一致）。

    参数:
        h_V: [B, L, 128] encoder 输出
        prompt_tokens: [B, 4, 128] ConditionEncoder 输出
    返回:
        h_V + softmax(h_V·prompt^T/√d)·prompt     # [B, L, 128]
    """
    B, L, D = h_V.shape
    scale = math.sqrt(D)
    attn = torch.softmax(h_V @ prompt_tokens.transpose(1, 2) / scale, dim=-1)  # [B,L,4]
    return h_V + attn @ prompt_tokens


def conditioned_sample(model, condition_encoder, feature_dict, cond_vec, device="cpu"):
    """条件注入生成（batch=1）。

    流程：
        1. model.encode → h_V, h_E, E_idx
        2. condition_encoder(cond_vec) → prompt tokens [1, 4, 128]
        3. h_V = inject_prompt(h_V, prompt)          # 条件注入
        4. 复用 guided_sample 的解码循环（无 bias_callback = 无 logit bias，
           模型自身 pH 感知 → Phase 3 Go/No-Go 的干净测试）

    参数:
        model: MoMPNN/ProteinMPNN backbone
        condition_encoder: 微调后的 ConditionEncoder（None = 不注入，
            等价 Phase 1「无引导时模型不感知 pH」的诚实边界对照）
        feature_dict: featurize 后的字典（含 bias 等键）
        cond_vec: [7] 条件向量（未归一化；编码器内部按训练 μ/σ 标准化）
        device: 计算设备

    返回:
        guided_sample 的输出 dict {S, sampling_probs, log_probs, decoding_order}
    """
    # 先把 feature_dict 张量移到 device（与 GuidedSampler.sample 一致）
    fd = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in feature_dict.items()
    }
    h_V, h_E, E_idx = model.encode(fd)
    if condition_encoder is not None:
        c = cond_vec.unsqueeze(0).to(device)          # [1, 7]
        prompt = condition_encoder(c)                 # [1, 4, 128]
        h_V = inject_prompt(h_V, prompt)
    return guided_sample(model, fd, device=device, encoded=(h_V, h_E, E_idx))


if __name__ == "__main__":
    # 自检：只验证 inject_prompt 维度，不依赖真实模型
    B, L, D = 1, 10, 128
    h_V = torch.randn(B, L, D)
    prompt = torch.randn(B, 4, D)
    out = inject_prompt(h_V, prompt)
    print("inject_prompt 输出 shape:", tuple(out.shape), "(expect [1, 10, 128])")
