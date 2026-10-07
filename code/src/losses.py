"""Composite loss function for Phase 2 condition fine-tuning.

Overall objective:

    L = CE_loss + λ_c·charge_deviation + λ_l·structure_penalty + λ_dpo·DPO_aux

Components:
- CE_loss            : standard autoregressive cross-entropy (kept at a higher weight to preserve structural compatibility)
- charge_deviation   : differentiable deviation between generated and target net charge (see differentiable_charge)
- structure_penalty  : penalizes amino acids that the structure-aware filter suppresses
- DPO_aux            : preference-alignment auxiliary loss (λ_dpo weight 0.01-0.05)

Also provides a flexible-margin constraint-conflict check:
for each candidate amino acid, compute an "equivalent bias" = benefit for the current constraint −
cost to other active constraints. If negative (conflicting), reduce that candidate's logit-bias weight.
"""

import torch
import torch.nn.functional as F

from .differentiable_charge import net_charge_from_logits


def cross_entropy_loss(logits, target_seq, mask):
    """Standard autoregressive cross-entropy.

    Parameters:
        logits: decoder logits [B, L, 21]
        target_seq: target sequence [B, L] (integers)
        mask: position mask [B, L] for the loss (usually = mask × chain_mask)
    Returns:
        Scalar (mean NLL over masked positions)
    """
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, target_seq.unsqueeze(-1)).squeeze(-1)  # [B, L]
    denom = mask.float().sum().clamp(min=1.0)
    return (nll * mask.float()).sum() / denom


def charge_deviation_loss(logits, pH, target_charge, mask=None, temperature=1.0):
    """Net-charge deviation: |expected net charge − target charge|, differentiable.

    Parameters:
        temperature: softmax temperature (<1 sharpens the distribution so the optimized distribution
            better approximates inference sampling; see net_charge_from_logits)
    """
    charge = net_charge_from_logits(logits, pH=pH, mask=mask, temperature=temperature)
    return torch.abs(charge - target_charge).mean()


def sequence_keep_loss(logits, anchor_seq, mask):
    """Sequence-preservation regularizer for selective condition injection.

    Semantics: the conditioned output should approximate the "unconditional argmax anchor sequence" at each position—
    the most likely output of the frozen backbone without injection. This encourages the sequence to remain unchanged
    when no pI change is requested (target=native), while allowing changes when one is requested.

    Why this is more direct than a KL anchor (kl_anchor_loss):
        KL penalizes **distribution distance**. A small shift in softmax probability (e.g., 0.30→0.29)
        can yield a small KL value even when the argmax flips (K→R) and sequence-level identity drops.
        This loss uses the unconditional argmax as the CE target and penalizes the most likely unconditional
        amino acid losing probability under the conditional distribution, directly addressing sequence-level behavior.

    Difference from CE(native):
        CE anchors to native (structural-compatibility anchor); this loss anchors to the unconditional output (behavior preservation).
        Both are applied to self-consistent samples and have complementary roles.

    Apply **only to self-consistent samples where target=native** (for perturbed samples with target≠native,
    a charge shift is expected and is not constrained by this regularizer). The caller decides per sample.

    Parameters:
        logits: conditioned output logits [B, L, 21]
        anchor_seq: unconditional argmax sequence [B, L] (precomputed by the frozen backbone; constant)
        mask: position mask [B, L]
    Returns:
        Scalar (mean NLL on the anchor over masked positions)
    """
    return cross_entropy_loss(logits, anchor_seq, mask)


def structure_penalty_loss(logits, filter_bias, mask=None):
    """Structural penalty: suppresses model selection of amino acids repressed by the filter.

    Implementation: the negative inner product of softmax probabilities and filter bias. At positions where filter bias is negative (suppression),
    high model probability incurs a large penalty; positions with positive bias are not penalized.
    Parameters:
        logits: [B, L, 21]
        filter_bias: structure-aware filter bias [L, 21] (positive = promote, negative = suppress)
        mask: position mask [B, L] for the calculation
    """
    probs = F.softmax(logits[..., :20], dim=-1)          # [B, L, 20]
    fb = filter_bias[..., :20].float()                    # [L, 20]
    penalty = -(probs * fb.unsqueeze(0)).sum(dim=-1)      # [B, L] (negative bias → positive penalty)
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
    """DPO preference-alignment auxiliary loss.

    Inputs are **per-sample sums of log probabilities** (e.g., log_probs summed over designed positions):
        win_logprobs      : logP assigned to the preferred sequence by the current model
        lose_logprobs     : logP assigned to the dispreferred sequence by the current model
        ref_win_logprobs  : logP assigned to the preferred sequence by the reference model
        ref_lose_logprobs : logP assigned to the dispreferred sequence by the reference model
    Returns the scalar: −log σ(β·[(logP_win−logP_lose)−(logPref_win−logPref_lose)])
    """
    log_ratio = (win_logprobs - lose_logprobs) - (ref_win_logprobs - ref_lose_logprobs)
    return -F.logsigmoid(beta * log_ratio).mean()


def equivalent_margin_bias(primary_benefit, others_cost, w_primary=1.0, w_others=1.0):
    """Multi-constraint conflict check using a flexible-margin formulation.

    Compute an "equivalent bias" for each candidate amino acid:
        equivalent bias = w_primary·(benefit for the current constraint) − w_others·(cost to other active constraints)
    If the result is negative (conflicting), reduce that candidate's logit-bias weight.

    Parameters (all vectors are per candidate):
        primary_benefit: [n_cand] candidate's benefit for the current constraint (e.g., moving net charge toward the target)
        others_cost: [n_cand] candidate's cost to other active constraints (e.g., worsening charge clustering)
        w_primary, w_others: scalar weights
    Returns:
        [n_cand] equivalent bias
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
    """Composite loss called during Phase 2 training.

    Parameters:
        logits: [B, L, 21]
        target_seq: [B, L]
        mask: design-position mask [B, L]
        pH: operating pH (used with target_charge to compute charge deviation)
        target_charge: target net charge
        filter_bias: structure-filter bias [L, 21]
        dpo_pairs: optional dict containing preferred/dispreferred log probabilities; see dpo_aux_loss
        lambda_c / lambda_l / lambda_dpo: weights for the auxiliary terms
    Returns:
        dict {total, ce, charge, structure, dpo}, where total is used for backward
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
    # Self-check: create dummy logits and evaluate the composite loss
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
