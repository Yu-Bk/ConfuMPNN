"""v10 新增可微损失（v3 方案 §3.1：A 条件解耦 + B 表面添加电荷监督 + C 结构惩罚增强）。

背景（v3 治 P1"删减捷径"）：
  v7/v9 根因 = 模型**无差别删减带电残基**（收敛低电荷密度）靠不对称删减逼近 target，
  代价 = 电荷斑块丢失 + 表面疏水化（GRAVY↑）。v10 用三件套对症：
    A 条件解耦（--decouple_perturb）：训练数据层面，任意骨架 × 随机 target，打破
      "骨架类型与 target 电荷强耦合"（碱性骨架只见正电 target → 模型只能外推）。
    B 表面添加电荷监督（--add_supervision）：损失层面，L_add 直接对抗"只删不加"——
      需要更负 → 表面位点增加 D/E（更正要加 K/R），以净电荷目标为上界，只动表面。
    C 结构惩罚增强（--ph_aware_filter）：损失层面，动态加强盐桥/聚集惩罚，
      防"大额添加"造成电荷成簇；pH 自适应带电集合见 P0-5。

本模块提供 B/C 的损失函数（A 在 train_finetune.py 数据循环里实现）。
"""

import torch
import torch.nn.functional as F


def surface_add_charge_loss(logits, frac_sasa, target_surface_charge_delta,
                            surface_threshold=0.25, k=5.0, w_min=0.05):
    """B 表面添加电荷监督：L_add = |Σ_i w_i·p_i(D/E) − target_surface_delta|。

    思路（v3 §3.1 B）：
      - 需要更负（target < 期望净电荷）时，应在**表面位点**增加负电残基 D/E；
        需要更正时增加正电残基 K/R。
      - 用 soft-count：每个表面位点 i 的"负电添加量" = p_i(D) + p_i(E)
        （模型给 D/E 的概率 = 期望的残基计数，可微）。
      - 权重 w_i = σ(k·(fracSASA_i − θ))：表面位（fracSASA 高）权重大、
        埋藏位（≈0）权重≈0 → 强制"只加表面、不加核心"。
      - 以 target_surface_charge_delta 为**上限**（不是无限加）。

    参数:
        logits: [B, L, 21]（取前 20 个 AA）
        frac_sasa: [L] 逐残基 fractional SASA（0~1+，见 src/sasa.py）
        target_surface_charge_delta: 目标表面电荷增量（标量）。
            负 = 需要表面更负（加 D/E）；正 = 需要表面更正（加 K/R）。
            由调用方按"当前电荷 vs target"的方向决定（见 train_finetune）。
        surface_threshold: 表面资格门槛 θ（fracSASA ≥ θ 才计入）
        k: sigmoid 陡度（越大越"硬门槛"，越小越平滑）
        w_min: 埋藏位权重下限（防数值 0）
    返回:
        标量 L_add（可微，驱动 logits 把表面位点的 D/E（或 K/R）概率推向目标）
    """
    probs = F.softmax(logits[..., :20], dim=-1)          # [B, L, 20]
    # D=3, E=4（见 pka.AAS "ACDEFGHIKLMNPQRSTVWY"）
    d_idx, e_idx = 3, 4
    k_idx, r_idx = 10, 15
    neg_count = probs[..., d_idx] + probs[..., e_idx]      # [B, L] 负电残基期望计数
    pos_count = probs[..., k_idx] + probs[..., r_idx]      # [B, L] 正电残基期望计数

    frac = torch.as_tensor(frac_sasa, dtype=logits.dtype, device=logits.device)  # [L]
    w = torch.sigmoid(k * (frac - surface_threshold)) + w_min  # [L]，表面大→埋藏小
    w = w.unsqueeze(0)                                     # [B, L]

    if target_surface_charge_delta < 0:
        # 需要表面更负 → 增加 D/E。
        # 目标：让总负电残基计数 total_neg 尽量接近 |delta|（delta<0，目标是更负）。
        # 用「单侧 hinge」只推"涨"，不拉"跌"：total_neg < target_abs 时惩罚（需要更多 D/E），
        # total_neg ≥ target_abs 时不罚（达到上界，不多加）。以净电荷目标为上界（v3 §3.1 B）。
        target_abs = -target_surface_charge_delta
        total_neg = (neg_count * w).sum(dim=-1)
        loss = torch.relu(target_abs - total_neg).mean()
    else:
        # 需要表面更正 → 增加 K/R。对称：total_pos < delta 时惩罚，≥ 不罚。
        total_pos = (pos_count * w).sum(dim=-1)
        loss = torch.relu(target_surface_charge_delta - total_pos).mean()
    return loss


def ph_aware_structure_penalty(logits, structure_filter, seq_int_cur, pH,
                               mask=None, scale_boost=1.0):
    """C 结构惩罚增强：用 P0-5 的 pH 自适应过滤器 bias 惩罚带电聚集。

    对比 losses.structure_penalty_loss（旧）：
      - 旧版 filter_bias 由调用方在解码时计算，与本函数无耦合；
      - 新版直接调用 StructureAwareFilter.compute_bias(seq_int, pH=pH)——
        内部已实现 pH 自适应带电集合（D4-③），且能拿到逐规则 info 用于动态加强。

    动态加强：scale_boost >1 时，把 bias 放大（对大额添加的扰动样本更狠地压制聚集）。
    返回: (loss 标量, info dict)
    """
    bias, info = structure_filter.compute_bias(seq_int_cur, pH=pH)
    bias = bias * scale_boost
    probs = F.softmax(logits[..., :20], dim=-1)
    fb = bias[..., :20].float().to(logits.device)          # [L, 20]
    penalty = -(probs * fb.unsqueeze(0)).sum(dim=-1)       # [B, L]
    if mask is None:
        return penalty.mean(), info
    denom = mask.float().sum().clamp(min=1.0)
    return (penalty * mask.float()).sum() / denom, info
