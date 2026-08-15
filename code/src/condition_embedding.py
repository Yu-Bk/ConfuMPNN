"""pH 感知条件编码器（Soft Prompt，NExT-Mol 风格）。

对应 PROJECT_PLAN.md 4.2 / 4.5 的核心设计决策：

1. **条件向量（mask-aware，shape [7]）**：
   [pH, has_charge_flag, charge_val, has_pos_limit_flag, pos_limit_val,
    has_neg_limit_flag, neg_limit_val]
   `has_X_flag` 告诉网络哪些值是真实条件、哪些是占位符（避免 0 值歧义）。

2. **连续编码，不量化**（为什么不选 MolGPT 见 PROJECT_PLAN 1.1C）：
   pH 是连续值，用 MLP 映射为连续 soft prompt 向量，精度无损。

3. **标准化**：不同量纲的条件（pH 4-10 vs 净电荷 -20~+20）直接输入 MLP
   会导致梯度不稳定。训练前从训练集算每维度的 μ/σ，写入 config，推理时复用。

网络结构：
    Linear(7→64) → GELU → Linear(64→128) → GELU → Linear(128→4×128)
    → reshape [4, 128]  （4 个 soft prompt token，拼到 decoder 输入前缀）

此模块供 **Phase 2（条件微调）** 使用；Phase 1（引导采样）暂时用不到。
"""

import torch
import torch.nn as nn

# 默认维度（PROJECT_PLAN.md 4.5）
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
    """构造 mask-aware 条件向量 [7]。

    参数:
        pH: 工作环境 pH（必填）
        net_charge: 目标净电荷；None=不指定
        local_pos_limit: 10Å 内正电荷数上限；None=不指定
        local_neg_limit: 10Å 内负电荷数上限；None=不指定

    返回:
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
    """条件向量 → 连续 soft prompt tokens。

    参数:
        cond_dim: 条件向量维度（默认 7）
        hidden_dim: 隐藏层维度（默认 64）
        token_dim: 每个 soft prompt token 的维度（默认 128）
        n_tokens: soft prompt token 数量（默认 4）
        mean, std: 标准化常量（[cond_dim]）；None 表示不做标准化
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
        # 注册标准化常量（不参与梯度，推理时从 config 加载）
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
        """按训练集统计标准化（训练前算好 μ/σ 存 config）。"""
        if self.mean is not None and self.std is not None:
            return (c - self.mean) / (self.std + 1e-8)
        return c

    def forward(self, c):
        """条件向量 → soft prompt tokens。

        参数:
            c: [B, cond_dim] 条件向量（每行一个样本的 7 维向量）
        返回:
            [B, n_tokens, token_dim] soft prompt tokens
        """
        c = self.normalize(c)
        out = self.net(c)  # [B, n_tokens*token_dim]
        return out.view(-1, self.n_tokens, self.token_dim)


if __name__ == "__main__":
    # 自检：构造几个条件向量并前向
    enc = ConditionEncoder()
    v1 = make_condition_vector(pH=7.4)
    v2 = make_condition_vector(pH=5.0, net_charge=0.0, local_pos_limit=8)
    batch = torch.stack([v1, v2])  # [2, 7]
    tokens = enc(batch)
    print("condition vectors:", batch.tolist())
    print("soft prompt tokens shape:", tuple(tokens.shape))
