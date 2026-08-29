"""v12 训练侧监督（v12 §7.2）：治"删减捷径"——组成双计数 + GRAVY + 表面电荷密度目标。

教训对应（为什么不是 v10 的 L_add）：
  - v10 L_add 两大缺陷：① 只盯"D/E 存量 ≥ 增量"下限检查，删 K/R 完全不设防；
    ② **AA 索引 bug**——D=3,E=4,K=10,R=15 相对 pka.AAS "ACDEFGHIKLMNPQRSTVWY"
    的真实索引（D=2,E=3,K=8,R=14）**偏了 1**，实际监督的是 E+F / M+S。
  - v7/v9 根因：模型无差别删带电残基总数（收敛低电荷密度），靠不对称删减调净电荷。
  - 本模块三个损失：
    (1) surface_composition_loss：表面 D/E 和 K/R **双计数**都不许低于 native×frac_floor
        → 删 D/E、删 K/R 都被罚 → 只能通过"加目标侧残基"调净电荷（堵死删减捷径）。
    (2) surface_gravy_loss：生成序列表面 GRAVY ≤ native 表面 GRAVY + margin
        → 删任何带电残基（强亲水）换疏水都会让 GRAVY↑ → 被罚（带电残基总数的代理约束）。
    (3) surface_charge_target_loss：表面净电荷 = target − 核心 native 电荷
        （核心锁死，表面承担全部电荷变化 → 逼"加表面电荷"；可微，用 net_charge_from_logits 的 mask）。
"""

import torch
import torch.nn.functional as F

# pka.AAS = "ACDEFGHIKLMNPQRSTVWY"（与 LigandMPNN restype 一致）的 AA 索引
D_IDX, E_IDX, K_IDX, R_IDX = 2, 3, 8, 14

# Kyte-Doolittle 疏水指数，顺序 = AAS（A C D E F G H I K L M N P Q R S T V W Y）
KD = torch.tensor([
    1.8, 2.5, -3.5, -3.5, 2.8,     # A C D E F
    -0.4, -3.2, 4.5, -3.9, 3.8,    # G H I K L
    1.9, -3.5, -1.6, -3.5, -4.5,   # M N P Q R
    -0.8, -0.7, 4.2, -0.9, -1.3,   # S T V W Y
], dtype=torch.float32)


def _probs(logits):
    """softmax 概率 [B, L, 20]。"""
    return F.softmax(logits[..., :20], dim=-1)


def _surface_mask(frac_sasa, threshold, device, logits_dtype):
    """frac_sasa [L] → 表面掩码 [1, L] float（0/1）。骨架固定 → 常数。"""
    m = torch.as_tensor(frac_sasa, dtype=logits_dtype, device=device) >= threshold
    return m.float().unsqueeze(0)


def surface_composition_loss(logits, frac_sasa, native_seq_int,
                             frac_floor=0.8, surface_threshold=0.25):
    """表面组成双计数：D/E 与 K/R 期望计数都不低于 native×frac_floor。

    native_seq_int: [L] native 序列的 AA 整数索引（LigandMPNN restype 序）。
    frac_floor: 下限比例（native 表面计数的多少倍；0.8=允许温和删减，但不许大幅）。
    返回: 标量（两个方向的 relu 之和）。
    """
    probs = _probs(logits)                      # [B, L, 20]
    smask = _surface_mask(frac_sasa, surface_threshold, logits.device, logits.dtype)
    nat = torch.as_tensor(native_seq_int, device=logits.device)  # [L]

    # native 表面负/正电残基（0/1）[L]
    nat_neg = ((nat == D_IDX) | (nat == E_IDX)).float() * smask.squeeze(0)
    nat_pos = ((nat == K_IDX) | (nat == R_IDX)).float() * smask.squeeze(0)
    n_neg = nat_neg.sum().clamp(min=1e-6)
    n_pos = nat_pos.sum().clamp(min=1e-6)

    # 生成期望表面计数（可微）
    gen_neg = ((probs[..., D_IDX] + probs[..., E_IDX]) * smask).sum(-1)   # [B]
    gen_pos = ((probs[..., K_IDX] + probs[..., R_IDX]) * smask).sum(-1)

    loss = (torch.relu(n_neg * frac_floor - gen_neg) +
            torch.relu(n_pos * frac_floor - gen_pos)).mean()
    return loss


def surface_gravy_loss(logits, frac_sasa, native_gravy_surface,
                       margin=0.15, surface_threshold=0.25):
    """GRAVY 监督：生成序列表面 GRAVY ≤ native 表面 GRAVY + margin。

    native_gravy_surface: 标量，native 序列表面残基的 GRAVY 均值（调用方预计算）。
    margin: 允许的表面 GRAVY 上升量（删减捷径的容忍度，消融超参）。
    返回: 标量 relu(表面GRAVY(gen) − native − margin)。
    """
    probs = _probs(logits)                      # [B, L, 20]
    smask = _surface_mask(frac_sasa, surface_threshold, logits.device, logits.dtype)
    kd = KD.to(logits.device)
    gravy = probs @ kd                          # [B, L] 逐残基期望 GRAVY
    surf_gravy = (gravy * smask).sum(-1) / smask.sum(-1).clamp(min=1.0)  # [B]
    loss = torch.relu(surf_gravy - native_gravy_surface - margin).mean()
    return loss


def surface_charge_target_loss(logits, pH, target_surface_charge, frac_sasa,
                               surface_threshold=0.25, temperature=1.0):
    """表面电荷密度目标：表面净电荷 → target_surface_charge。

    用 net_charge_from_logits 的 mask 只统计表面残基电荷（含两端电荷，
    但 mask 下两端贡献按 has_residue 计入，值小可忽略）。
    target_surface_charge = 目标净电荷 − 核心 native 电荷（调用方预计算，核心锁死）。
    返回: 标量 |表面电荷 − 目标|。
    """
    from .differentiable_charge import net_charge_from_logits
    smask = _surface_mask(frac_sasa, surface_threshold, logits.device, logits.dtype)
    B = logits.shape[0]
    q_surf = net_charge_from_logits(logits, pH, mask=smask.expand(B, -1),
                                    temperature=temperature)
    return (q_surf - target_surface_charge).abs().mean()
