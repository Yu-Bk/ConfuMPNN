"""Differentiable losses for surface charge-addition supervision and pH-aware structure penalties.

Surface charge-addition supervision (controlled by --add_supervision) penalizes insufficient surface charge changes toward the target.
The pH-aware structural penalty (controlled by --ph_aware_filter) uses structure-filter bias to discourage charge clustering.
This module provides the charge-addition and structure-penalty loss functions.
"""

import torch
import torch.nn.functional as F


def surface_add_charge_loss(logits, frac_sasa, target_surface_charge_delta,
                            surface_threshold=0.25, k=5.0, w_min=0.05):
    """Surface charge-addition supervision: L_add = |Σ_i w_i·p_i(D/E) − target_surface_delta|.

    Design:
      - When a more negative target is needed (target < expected net charge), add negative residues D/E at **surface positions**;
        for a more positive target, add positive residues K/R.
      - Use a soft count: at each surface position i, the "negative-charge addition" is p_i(D) + p_i(E)
        (the model's probabilities for D/E are the expected residue counts and are differentiable).
      - Weight w_i = σ(k·(fracSASA_i − θ)): surface positions (high fracSASA) receive larger weights,
        while buried positions (≈0) receive weights near 0 → enforce "add at the surface only, not in the core."
      - Treat target_surface_charge_delta as an **upper bound** (not an unlimited addition).

    Parameters:
        logits: [B, L, 21] (uses the first 20 AA)
        frac_sasa: per-residue fractional SASA [L] (0~1+; see src/sasa.py)
        target_surface_charge_delta: target surface-charge change (scalar).
            Negative = make the surface more negative (add D/E); positive = make it more positive (add K/R).
            The caller determines the sign from the current charge relative to the target.
        surface_threshold: surface-eligibility threshold θ (included when fracSASA ≥ θ)
        k: sigmoid steepness (larger is a "harder threshold"; smaller is smoother)
        w_min: minimum weight for buried positions (prevents a numerical zero)
    Returns:
        Scalar L_add (differentiably shifts logits to increase D/E or K/R probabilities at surface positions toward the target)
    """
    probs = F.softmax(logits[..., :20], dim=-1)          # [B, L, 20]
    # D=3, E=4 (see pka.AAS "ACDEFGHIKLMNPQRSTVWY")
    d_idx, e_idx = 3, 4
    k_idx, r_idx = 10, 15
    neg_count = probs[..., d_idx] + probs[..., e_idx]      # [B, L] expected count of negatively charged residues
    pos_count = probs[..., k_idx] + probs[..., r_idx]      # [B, L] expected count of positively charged residues

    frac = torch.as_tensor(frac_sasa, dtype=logits.dtype, device=logits.device)  # [L]
    w = torch.sigmoid(k * (frac - surface_threshold)) + w_min  # [L]; larger on the surface, smaller for buried residues
    w = w.unsqueeze(0)                                     # [B, L]

    if target_surface_charge_delta < 0:
        # To make the surface more negative, increase D/E.
        # Target: bring total_neg close to |delta| (delta<0 means the target is more negative).
        # A one-sided hinge pushes counts up without pulling them down: penalize total_neg < target_abs (more D/E needed);
        # do not penalize total_neg ≥ target_abs (upper bound reached; no further addition). The net-charge target is the upper bound.
        target_abs = -target_surface_charge_delta
        total_neg = (neg_count * w).sum(dim=-1)
        loss = torch.relu(target_abs - total_neg).mean()
    else:
        # To make the surface more positive, increase K/R. Symmetrically, penalize total_pos < delta and not values ≥ delta.
        total_pos = (pos_count * w).sum(dim=-1)
        loss = torch.relu(target_surface_charge_delta - total_pos).mean()
    return loss


def ph_aware_structure_penalty(logits, structure_filter, seq_int_cur, pH,
                               mask=None, scale_boost=1.0):
    """Enhanced structural penalty: use pH-adaptive filter bias to penalize charge clustering.

    Compared with losses.structure_penalty_loss:
      - The other implementation receives filter_bias computed by the caller during decoding and is decoupled from this function.
      - This function calls StructureAwareFilter.compute_bias(seq_int, pH=pH) directly;
        the filter uses a pH-adaptive set of charged residues and returns per-rule info for dynamic scaling.

    Dynamic scaling: when scale_boost >1, increase the bias magnitude to suppress clustering more strongly in perturbed samples with large charge additions.
    Returns: (scalar loss, info dict)
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
