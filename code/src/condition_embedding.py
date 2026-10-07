"""pH-aware condition encoder (soft prompt).

Core design choices:

1. **Condition vector (mask-aware, shape [7])**:
   [pH, has_charge_flag, charge_val, has_pos_limit_flag, pos_limit_val,
    has_neg_limit_flag, neg_limit_val]
   `has_X_flag` tells the network which values are real conditions and which are placeholders (avoiding ambiguity when a value is 0).

2. **Continuous encoding without quantization**:
   pH is continuous, so an MLP maps it to continuous soft-prompt vectors without quantization.

3. **Normalization**: conditions with different scales (pH 4-10 vs. net charge -20~+20) can make gradients unstable
   when fed directly into the MLP. Compute per-dimension μ/σ from the training set before training, store them in config, and reuse them at inference.

Network architecture:
    Linear(7→64) → GELU → Linear(64→128) → GELU → Linear(128→4×128)
    → reshape [4, 128]  (4 soft-prompt tokens prepended to the decoder input)

This module is used for **Phase 2 (condition fine-tuning)**; Phase 1 (guided sampling) does not use it.
"""

import torch
import torch.nn as nn

# Default dimensions for the soft-prompt encoder
DEFAULT_COND_DIM = 7
DEFAULT_HIDDEN = 64
DEFAULT_TOKEN_DIM = 128
DEFAULT_N_TOKENS = 4


def make_condition_vector(
    pH,
    net_charge=None,
    local_pos_limit=None,
    local_neg_limit=None,
    dtype=torch.float32,
):
    """Build a mask-aware condition vector [7].

    Parameters:
        pH: operating pH (required)
        net_charge: target net charge; None = unspecified
        local_pos_limit: maximum number of positive charges within 10Å; None = unspecified
        local_neg_limit: maximum number of negative charges within 10Å; None = unspecified

    Returns:
        shape [7] tensor: [pH, has_c, charge, has_p, pos, has_n, neg]
    """
    vec = torch.zeros(DEFAULT_COND_DIM, dtype=dtype)
    vec[0] = float(pH)
    if net_charge is not None:
        vec[1] = 1.0
        vec[2] = float(net_charge)
    if local_pos_limit is not None:
        vec[3] = 1.0
        vec[4] = float(local_pos_limit)
    if local_neg_limit is not None:
        vec[5] = 1.0
        vec[6] = float(local_neg_limit)
    return vec


class ConditionEncoder(nn.Module):
    """Map a condition vector to continuous soft-prompt tokens.

    Parameters:
        cond_dim: condition-vector dimension (default 7)
        hidden_dim: hidden-layer dimension (default 64)
        token_dim: dimension of each soft-prompt token (default 128)
        n_tokens: number of soft-prompt tokens (default 4)
        mean, std: normalization constants ([cond_dim]); None disables normalization
    """

    def __init__(
        self,
        cond_dim=DEFAULT_COND_DIM,
        hidden_dim=DEFAULT_HIDDEN,
        token_dim=DEFAULT_TOKEN_DIM,
        n_tokens=DEFAULT_N_TOKENS,
        mean=None,
        std=None,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.token_dim = token_dim
        self.n_tokens = n_tokens
        # Register normalization constants (not trainable; loaded from config at inference)
        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32) if mean is not None else None
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32) if std is not None else None
        )

        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, n_tokens * token_dim),
        )

    def normalize(self, c):
        """Normalize using training-set statistics (compute μ/σ before training and store them in config)."""
        if self.mean is not None and self.std is not None:
            return (c - self.mean) / (self.std + 1e-8)
        return c

    def forward(self, c):
        """Map a condition vector to soft-prompt tokens.

        Parameters:
            c: [B, cond_dim] condition vector (each row contains one 7-dimensional sample)
        Returns:
            [B, n_tokens, token_dim] soft prompt tokens
        """
        c = self.normalize(c)
        out = self.net(c)  # [B, n_tokens*token_dim]
        return out.view(-1, self.n_tokens, self.token_dim)


if __name__ == "__main__":
    # Self-check: build a few condition vectors and run a forward pass
    enc = ConditionEncoder()
    v1 = make_condition_vector(pH=7.4)
    v2 = make_condition_vector(pH=5.0, net_charge=0.0, local_pos_limit=8)
    batch = torch.stack([v1, v2])  # [2, 7]
    tokens = enc(batch)
    print("condition vectors:", batch.tolist())
    print("soft prompt tokens shape:", tuple(tokens.shape))
