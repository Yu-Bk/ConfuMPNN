"""复合损失函数（Phase 2 条件微调用）。

PROJECT_PLAN.md Phase 2 定义的整体损失：

    L = CE_loss + λ_c·charge_deviation + λ_l·structure_penalty + λ_dpo·DPO_aux

各分量说明：
- CE_loss            : 标准自回归交叉熵（保留较高权重，防止结构匹配度下降）
- charge_deviation   : 生成序列净电荷 vs 目标电荷的偏差（可微，见 differentiable_charge）
- structure_penalty  : 结构惩罚——若模型选择被结构感知过滤器抑制的氨基酸则受罚
- DPO_aux            : 偏好对齐辅助（ProtAlign 迁移，λ_dpo 极轻权重 0.01-0.05）

另提供 ProtAlign flexible margin 迁移（PROJECT_PLAN 1.1B / Phase 2）：
多约束冲突检测——对每个候选氨基酸计算"等效 bias" = 对当前约束的收益 −
对其他活跃约束的代价。若为负（冲突），降低该候选的 logit bias 权重。
"""

import torch
import torch.nn.functional as F

from .differentiable_charge import net_charge_from_logits


def cross_entropy_loss(logits, target_seq, mask):
    """标准自回归交叉熵。

    参数:
        logits: [B, L, 21] 解码器 logits
        target_seq: [B, L] 目标序列（整数）
        mask: [B, L] 参与 loss 的位置掩码（通常 = mask × chain_mask）
    返回:
        标量（mask 位置的平均 NLL）
    """
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, target_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]
    denom = mask.float().sum().clamp(min=1.0)
    return (nll * mask.float()).sum() / denom


def charge_deviation_loss(logits, pH, target_charge, mask=None, temperature=1.0):
    """净电荷偏差：|期望净电荷 − 目标电荷|，可微。

    参数:
        temperature: softmax 温度（<1 锐化，让训练优化的分布≈推理采样分布，
            减小 Phase 3 发现的 ~2.57× 电荷过冲；见 net_charge_from_logits）
    """
    charge = net_charge_from_logits(logits, pH=pH, mask=mask, temperature=temperature)
    return torch.abs(charge - target_charge).mean()


def sequence_keep_loss(logits, anchor_seq, mask):
    """序列保持正则（治 S1 注入选择性，Phase 3 第十四轮新增）。

    语义：条件化输出在每个位置应逼近「无条件 argmax 锚序列」——即冻结
    backbone 在无注入下的最可能输出。这是判断标准 v1 里 S1 判据（A 场景
    条件臂 vs 基线 identity ≥ 0.7）的**训练侧直接对应**：无改 pI 需求时
    （target=native）不扰动，有需求时才改写。

    为什么比 KL 锚（kl_anchor_loss）更直接：
        KL 惩罚的是**分布距离**。softmax 分布概率小幅漂移（如 0.30→0.29）
        时 KL 已很小，但 argmax 可能翻盘（K→R），序列级 identity 照样下降。
        本损失直接以无条件 argmax 为目标序列做 CE，惩罚「无条件最可能
        氨基酸在条件分布下失势」——直接对应序列级判据。

    与 CE(native) 的区别：
        CE 锚 native（结构匹配度锚），本损失锚无条件输出（行为保持）。
        二者都施加在自洽样本上，语义互补。

    施加时机：**只在 target=native 的自洽样本上**施加（扰动样本 target≠native
    时电荷偏移是期望行为，不受本正则约束）。由调用方按 per-sample 判断。

    参数:
        logits: [B, L, 21] 条件化输出 logits
        anchor_seq: [B, L] 无条件 argmax 序列（冻结 backbone 预计算，常数）
        mask: [B, L] 参与位置掩码
    返回:
        标量（mask 位置的平均 NLL on anchor）
    """
    return cross_entropy_loss(logits, anchor_seq, mask)


def structure_penalty_loss(logits, filter_bias, mask=None):
    """结构惩罚：抑制模型选择被过滤器压制的氨基酸。

    实现：softmax 概率与 filter bias 的内积取负。filter bias 为负（抑制）
    的位置，若模型给了高概率则惩罚大；正 bias 的位置不受罚。
    参数:
        logits: [B, L, 21]
        filter_bias: [L, 21] 结构感知过滤器的 bias（正 = 促进，负 = 抑制）
        mask: [B, L] 参与计算的位置掩码
    """
    probs = F.softmax(logits[..., :20], dim=-1)          # [B, L, 20]
    fb = filter_bias[..., :20].float()                    # [L, 20]
    penalty = -(probs * fb.unsqueeze(0)).sum(dim=-1)      # [B, L]（bias 负 → 惩罚正）
    if mask is None:
        return penalty.mean()
    denom = mask.float().sum().clamp(min=1.0)
    return (penalty * mask.float()).sum() / denom


def dpo_aux_loss(
    win_logprobs,
    lose_logprobs,
    ref_win_logprobs,
    ref_lose_logprobs,
    beta=0.1,
):
    """DPO 偏好对齐辅助损失（ProtAlign 迁移，Phase 2）。

    输入都是每样本的**对数概率和**（如 log_probs 对设计位置求和）：
        win_logprobs      : 当前模型给胜者对的 logP
        lose_logprobs     : 当前模型给败者对的 logP
        ref_win_logprobs  : 参考模型给胜者对的 logP
        ref_lose_logprobs : 参考模型给败者对的 logP
    返回标量：−log σ(β·[(logP_win−logP_lose)−(logPref_win−logPref_lose)])
    """
    log_ratio = (win_logprobs - lose_logprobs) - (ref_win_logprobs - ref_lose_logprobs)
    return -F.logsigmoid(beta * log_ratio).mean()


def equivalent_margin_bias(primary_benefit, others_cost, w_primary=1.0, w_others=1.0):
    """多约束冲突检测（ProtAlign flexible margin 迁移，Phase 2）。

    对每个候选氨基酸计算"等效 bias"：
        等效 bias = w_primary·(对当前约束的收益) − w_others·(对其他活跃约束的代价)
    返回为负（冲突）时，应降低该候选的 logit bias 权重。

    参数（都是逐候选的向量）:
        primary_benefit: [n_cand] 该候选对当前约束的收益（如把净电荷拉向目标）
        others_cost: [n_cand] 该候选对其他活跃约束的代价（如加剧电荷聚集）
        w_primary, w_others: 权重标量
    返回:
        [n_cand] 等效 bias
    """
    return w_primary * primary_benefit - w_others * others_cost


def composite_loss(
    logits,
    target_seq,
    mask,
    pH=None,
    target_charge=None,
    filter_bias=None,
    dpo_pairs=None,
    lambda_c=0.1,
    lambda_l=0.05,
    lambda_dpo=0.01,
):
    """整体复合损失（Phase 2 训练时调用）。

    参数:
        logits: [B, L, 21]
        target_seq: [B, L]
        mask: [B, L] 设计位置掩码
        pH: 工作 pH（配合 target_charge 计算电荷偏差）
        target_charge: 目标净电荷
        filter_bias: [L, 21] 结构过滤 bias
        dpo_pairs: dict（可选），含 win/lose 对数概率，见 dpo_aux_loss
        lambda_c / lambda_l / lambda_dpo: 各辅助项权重
    返回:
        dict {total, ce, charge, structure, dpo}，其中 total 用于 backward
    """
    ce = cross_entropy_loss(logits, target_seq, mask)
    total = ce
    terms = {"ce": ce.item()}

    if lambda_c and pH is not None and target_charge is not None:
        cd = charge_deviation_loss(logits, pH=pH, target_charge=target_charge, mask=mask)
        total = total + lambda_c * cd
        terms["charge"] = cd.item()

    if lambda_l and filter_bias is not None:
        sp = structure_penalty_loss(logits, filter_bias, mask=mask)
        total = total + lambda_l * sp
        terms["structure"] = sp.item()

    if lambda_dpo and dpo_pairs is not None:
        dp = dpo_aux_loss(**dpo_pairs)
        total = total + lambda_dpo * dp
        terms["dpo"] = dp.item()

    terms["total"] = total.item()
    return total, terms


if __name__ == "__main__":
    # 自检：构造假 logits 跑一遍复合损失
    B, L = 2, 5
    logits = torch.randn(B, L, 21, requires_grad=True)
    target = torch.randint(0, 20, (B, L))
    mask = torch.ones(B, L)
    filter_bias = -torch.ones(L, 21) * 0.5
    loss, terms = composite_loss(
        logits, target, mask, pH=7.4, target_charge=0.0,
        filter_bias=filter_bias, lambda_c=0.1, lambda_l=0.05,
    )
    loss.backward()
    print("loss terms:", terms)
    print("backward OK, logits grad nonzero:", logits.grad.abs().sum().item() > 0)
