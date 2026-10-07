"""Condition-injection sampler: injects (pH, target_charge) into the backbone using a fine-tuned ConditionEncoder.

Used for Phase 3 validation. Uses the **same soft-prompt injection mechanism** as training
(cross-attention: h_V += softmax(h_V·prompt^T/√d)·prompt), keeping training and inference consistent.

Literal soft-prompt prefixes require reordering E_idx and are error-prone;
instead, inject encoder output h_V through cross-attention without modifying the decoder.
"""

import math

import torch

from .guided_sampler import guided_sample


def inject_prompt(h_V, prompt_tokens):
    """Inject the soft prompt using the same mechanism as training.

    Parameters:
        h_V: [B, L, 128] encoder output
        prompt_tokens: [B, 4, 128] ConditionEncoder output
    Returns:
        h_V + softmax(h_V·prompt^T/√d)·prompt     # [B, L, 128]
    """
    B, L, D = h_V.shape
    scale = math.sqrt(D)
    attn = torch.softmax(h_V @ prompt_tokens.transpose(1, 2) / scale, dim=-1)  # [B,L,4]
    return h_V + attn @ prompt_tokens


def conditioned_sample(model, condition_encoder, feature_dict, cond_vec, device="cpu"):
    """Generate with condition injection (batch=1).

    Workflow:
        1. model.encode → h_V, h_E, E_idx
        2. condition_encoder(cond_vec) → prompt tokens [1, 4, 128]
        3. h_V = inject_prompt(h_V, prompt)          # condition injection
        4. Reuse guided_sample's decoding loop (without bias_callback there is no logit bias;
           the model's own pH awareness → a clean Phase 3 evaluation)

    Parameters:
        model: MoMPNN/ProteinMPNN backbone
        condition_encoder: fine-tuned ConditionEncoder (None = no injection,
            equivalent to the honest Phase 1 boundary control where the model is unaware of pH without guidance)
        feature_dict: dictionary returned by featurize (includes bias and other keys)
        cond_vec: [7] condition vector (unnormalized; the encoder standardizes it internally using training μ/σ)
        device: compute device

    Returns:
        output dict from guided_sample: {S, sampling_probs, log_probs, decoding_order}
    """
    # Move feature_dict tensors to device first (same as GuidedSampler.sample)
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
    # Self-check: validate inject_prompt dimensions only; no real model required
    B, L, D = 1, 10, 128
    h_V = torch.randn(B, L, D)
    prompt = torch.randn(B, 4, D)
    out = inject_prompt(h_V, prompt)
    print("inject_prompt output shape:", tuple(out.shape), "(expect [1, 10, 128])")
