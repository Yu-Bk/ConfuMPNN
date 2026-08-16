"""可微 pH 感知净电荷计算。

基于 Henderson-Hasselbalch 方程的**平滑（可微）近似**：

    去质子化分数 = 1 / (1 + 10^(pH - pKa)) = σ(ln(10)·(pH - pKa))

其中 σ 是 sigmoid 函数。用 sigmoid 替代硬阈值，让电荷对 pH 处处可微，
从而可以反向传播梯度（Level 2 微调训练时作为辅助 loss）。

电荷规则（结合层面，游离 pKa 表，见 pka.py）：
  - 酸性残基 D/E/C/Y：去质子化带 -1，电荷 = -σ(ln10·(pH - pKa))
  - 碱性残基 K/R/H   ：质子化带 +1，电荷 =  +σ(ln10·(pKa - pH))
  - N 端 α-NH3+（pKa≈9.7）：电荷 = +σ(ln10·(9.7 - pH))
  - C 端 α-COOH（pKa≈2.3）：电荷 = -σ(ln10·(pH - 2.3))

用法：
    # 字符串序列（Phase 1 验证用）
    charge = net_charge("ACDEK", pH=7.4)

    # 解码器 logits（Phase 2 训练用，可微）
    charge_batch = net_charge_from_logits(logits, pH=7.4, mask=mask)
"""

import math

import torch

from .pka import (AA_TO_IDX, AAS, ACIDIC, PKA_C_TERM, PKA_N_TERM,
                  PKA_SIDECHAIN)

LN10 = math.log(10.0)


def _to_tensor(value, dtype=torch.float32, device=None):
    """把 python 标量/列表安全转成 torch 张量。"""
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def sidechain_charge(aa, pH):
    """单个氨基酸侧链在 pH 下的（部分）电荷。返回标量张量，可微。

    参数:
        aa: 单个氨基酸字母（大写，如 "D"）
        pH: 标量或张量
    """
    pH = _to_tensor(pH)
    pKa = PKA_SIDECHAIN.get(aa)
    if pKa is None:
        return torch.zeros_like(pH)  # 非电离残基贡献为 0
    if aa in ACIDIC:
        # 酸性：去质子化带 -1，去质子化分数 = σ(ln10·(pH - pKa))
        return -torch.sigmoid(LN10 * (pH - pKa))
    else:
        # 碱性（K/R/H）：质子化带 +1，质子化分数 = σ(ln10·(pKa - pH))
        return torch.sigmoid(LN10 * (pKa - pH))


def sidechain_charge_vector(pH):
    """返回 shape [20] 的张量：20 种标准氨基酸在 pH 下的侧链电荷。

    顺序与 LigandMPNN restype 一致（AAS = "ACDEFGHIKLMNPQRSTVWY"）。
    """
    pH = _to_tensor(pH)
    return torch.stack([sidechain_charge(a, pH) for a in AAS])


def _termini_charge(pH, n_term=True):
    """主链末端的电荷。

    N 端 α-NH3+（pKa=9.7）质子化带 +1；C 端 α-COOH（pKa=2.3）去质子化带 -1。
    返回标量张量。
    """
    pH = _to_tensor(pH)
    if n_term:
        return torch.sigmoid(LN10 * (PKA_N_TERM - pH))
    return -torch.sigmoid(LN10 * (pH - PKA_C_TERM))


def net_charge(seq, pH, include_termini=True):
    """给定**字符串序列**，计算其在 pH 下的净电荷（返回 float）。

    参数:
        seq: 单字母氨基酸序列，如 "ACDEK"（大小写不敏感，跳过 "-"/"X"/无效字符）
        pH: 工作环境 pH（标量）
        include_termini: 是否计入主链 N/C 端电荷（默认 True）
    """
    seq = "".join(a for a in seq.upper() if a in AA_TO_IDX)
    pH = _to_tensor(pH)
    if not seq:
        raise ValueError("输入序列为空或不含有效氨基酸")
    total = torch.zeros_like(pH)
    for aa in seq:
        total = total + sidechain_charge(aa, pH)
    if include_termini:
        total = total + _termini_charge(pH, n_term=True)
        total = total + _termini_charge(pH, n_term=False)
    return total.item()


def net_charge_from_logits(logits, pH, mask=None, include_termini=True, temperature=1.0):
    """从解码器 logits 计算**期望净电荷**（可微，Phase 2 训练用）。

    用 softmax 概率对每个位置的 20 种氨基酸电荷加权平均，再对全长求和。
    由于 N/C 端电荷只依赖 pH 不依赖残基选择，它们对 logits 无梯度，
    作为常数直接累加（物理上任何残基都有 α-NH3+/α-COOH）。

    参数:
        logits: [B, L, 21] 或 [B, L, 20] 的 logits（21 维时只取前 20 个 AA）
        pH: 工作环境 pH（标量）
        mask: [B, L] 有效残基掩码（1=有效，0=忽略），默认全有效
        include_termini: 是否计入主链末端电荷
        temperature: softmax 温度（默认 1.0）。<1 时分布锐化，E[Q] 更接近
            「argmax 序列的电荷」= 推理采样（temperature 0.3）的电荷，
            使训练优化目标与推理一致 → 减小 Phase 3 发现的 ~2.57× 过冲。

    返回:
        [B] 张量：每个样本的期望净电荷
    """
    if logits.shape[-1] == 21:
        logits = logits[..., :20]
    probs = torch.softmax(logits / temperature, dim=-1)          # [B, L, 20]
    B, L, _ = probs.shape
    if mask is None:
        mask = torch.ones(B, L, device=logits.device)
    mask = mask.float()

    Q = sidechain_charge_vector(pH).to(logits.dtype).to(logits.device)  # [20]
    side_charge = torch.einsum("blk,k->bl", probs, Q)    # [B, L] 期望侧链电荷
    total = (side_charge * mask).sum(dim=-1)             # [B]

    if include_termini and L > 0:
        has_residue = (mask.sum(dim=-1) > 0).float()     # [B]
        n_charge = _termini_charge(pH, n_term=True).to(logits.dtype).to(logits.device)
        c_charge = _termini_charge(pH, n_term=False).to(logits.dtype).to(logits.device)
        total = total + has_residue * (n_charge + c_charge)
    return total


if __name__ == "__main__":
    # 快速自检：D/E 酸性，K/R 碱性，在 pH=7.4 下各贡献应为 ±1
    for aa, expect in [("D", -1.0), ("E", -1.0), ("K", 1.0), ("R", 1.0)]:
        got = sidechain_charge(aa, 7.4).item()
        print(f"{aa} @ pH7.4 = {got:+.3f}  (expect ~{expect:+.1f})")

    seq = "ACDEK"
    print(f"net_charge('{seq}', pH=7.4) = {net_charge(seq, 7.4):+.3f}")
