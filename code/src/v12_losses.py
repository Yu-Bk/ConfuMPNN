"""Training-time surface-supervision losses for composition, GRAVY, and surface charge density.

The module defines three objectives:
  (1) surface_composition_loss: expected surface D/E and K/R counts each remain ≥ native×frac_floor.
  (2) surface_gravy_loss: generated surface GRAVY ≤ native surface GRAVY + margin.
  (3) surface_charge_target_loss: surface net charge = target − core native charge
      (the core is fixed, so the surface accounts for the charge change; differentiable via the net_charge_from_logits mask).
"""

import torch
import torch.nn.functional as F

# AA indices for pka.AAS = "ACDEFGHIKLMNPQRSTVWY", ordered consistently with LigandMPNN restype
D_IDX, E_IDX, K_IDX, R_IDX = 2, 3, 8, 14

# Kyte-Doolittle hydropathy scale, ordered as AAS (A C D E F G H I K L M N P Q R S T V W Y)
KD = torch.tensor([
    1.8, 2.5, -3.5, -3.5, 2.8,     # A C D E F
    -0.4, -3.2, 4.5, -3.9, 3.8,    # G H I K L
    1.9, -3.5, -1.6, -3.5, -4.5,   # M N P Q R
    -0.8, -0.7, 4.2, -0.9, -1.3,   # S T V W Y
], dtype=torch.float32)


def _probs(logits):
    """Softmax probabilities [B, L, 20]."""
    return F.softmax(logits[..., :20], dim=-1)


def _surface_mask(frac_sasa, threshold, device, logits_dtype):
    """Convert frac_sasa [L] to a float surface mask [1, L] (0/1). The backbone is fixed, so the mask is constant."""
    m = torch.as_tensor(frac_sasa, dtype=logits_dtype, device=device) >= threshold
    return m.float().unsqueeze(0)


def surface_composition_loss(logits, frac_sasa, native_seq_int,
                             frac_floor=0.8, surface_threshold=0.25):
    """Bidirectional surface-composition count constraint: expected D/E and K/R counts must each be ≥ native×frac_floor.

    native_seq_int: AA integer indices [L] for the native sequence (LigandMPNN restype order).
    frac_floor: lower-bound fraction of native surface counts (0.8 allows modest reduction but prevents large reductions).
    Returns: scalar (sum of the two relu terms).
    """
    probs = _probs(logits)                      # [B, L, 20]
    smask = _surface_mask(frac_sasa, surface_threshold, logits.device, logits.dtype)
    nat = torch.as_tensor(native_seq_int, device=logits.device)  # [L]

    # Native surface negative/positive residues (0/1) [L]
    nat_neg = ((nat == D_IDX) | (nat == E_IDX)).float() * smask.squeeze(0)
    nat_pos = ((nat == K_IDX) | (nat == R_IDX)).float() * smask.squeeze(0)
    n_neg = nat_neg.sum().clamp(min=1e-6)
    n_pos = nat_pos.sum().clamp(min=1e-6)

    # Generated expected surface counts (differentiable)
    gen_neg = ((probs[..., D_IDX] + probs[..., E_IDX]) * smask).sum(-1)   # [B]
    gen_pos = ((probs[..., K_IDX] + probs[..., R_IDX]) * smask).sum(-1)

    loss = (torch.relu(n_neg * frac_floor - gen_neg) +
            torch.relu(n_pos * frac_floor - gen_pos)).mean()
    return loss


def surface_gravy_loss(logits, frac_sasa, native_gravy_surface,
                       margin=0.15, surface_threshold=0.25):
    """GRAVY supervision: generated surface GRAVY ≤ native surface GRAVY + margin.

    native_gravy_surface: scalar mean GRAVY of native surface residues (precomputed by the caller).
    margin: allowed increase in surface GRAVY.
    Returns: scalar relu(surface_GRAVY(gen) − native − margin).
    """
    probs = _probs(logits)                      # [B, L, 20]
    smask = _surface_mask(frac_sasa, surface_threshold, logits.device, logits.dtype)
    kd = KD.to(logits.device)
    gravy = probs @ kd                          # [B, L] expected GRAVY per residue
    surf_gravy = (gravy * smask).sum(-1) / smask.sum(-1).clamp(min=1.0)  # [B]
    loss = torch.relu(surf_gravy - native_gravy_surface - margin).mean()
    return loss


def surface_charge_target_loss(logits, pH, target_surface_charge, frac_sasa,
                               surface_threshold=0.25, temperature=1.0, extra_mask=None):
    """Surface-charge-density target: surface net charge → target_surface_charge.

    Use the mask in net_charge_from_logits to count only supervised-residue charges (including terminal charges;
    under the mask, terminal contributions are counted via has_residue and are usually small).
    target_surface_charge = target net charge − core native charge (precomputed by the caller; the core is fixed).
    extra_mask: add pocket residues to the supervised mask (mask = surface ∪ extra_mask).
    With three mutually exclusive partitions, core excludes pocket → total charge remains equal to target (no double counting or drift).
    Returns: scalar |supervised charge − target|.
    """
    from .differentiable_charge import net_charge_from_logits
    smask = _surface_mask(frac_sasa, surface_threshold, logits.device, logits.dtype)  # [1, L]
    if extra_mask is not None:
        extra = torch.as_tensor(extra_mask, dtype=logits.dtype, device=logits.device).float().unsqueeze(0)
        smask = torch.clamp(smask + extra, max=1.0)
    B = logits.shape[0]
    q_surf = net_charge_from_logits(logits, pH, mask=smask.expand(B, -1),
                                    temperature=temperature)
    return (q_surf - target_surface_charge).abs().mean()


def pocket_count_loss(logits, pocket_mask, native_pocket_counts,
                      floor=0.7, ceil=1.3, min_abs_cap=2.0, normalize=False):
    """Bidirectional count bounds (preserve abundance, not positions; ≠fix): expected D/E and K/R counts each lie in [N·floor, N·ceil].

    Prevents paired deletion of K and D from lowering total charged-residue counts while leaving net charge unchanged:
      - floor≈0.7 constrains reductions;
      - ceil≈1.3 constrains paired additions by providing an upper bound as well as a lower bound.
    mask may be pocket (keep mode) or charge_surf_mask=surface∪pocket (global mode via --pocket_mode global).
    The native counts within the mask define the bounds. Keep mode passes raw counts; global mode uses normalize=True
    (divide by native counts to obtain proportions, so region size does not scale with protein length and λ is comparable across domains).
    min_abs_cap: when native count N≈0 in one direction, the ceil side allows an absolute amount (default 2, permitting a few charged residues to meet the target);
    with no native count to protect on the floor side, no penalty is applied (avoids deadlock when N=0).
    Returns: scalar (mean of four relu terms).
    """
    probs = _probs(logits)                             # [B, L, 20]
    pmask = torch.as_tensor(pocket_mask, dtype=logits.dtype, device=logits.device).float()  # [L]
    n_neg = float(native_pocket_counts[0])
    n_pos = float(native_pocket_counts[1])
    gen_neg = ((probs[..., D_IDX] + probs[..., E_IDX]) * pmask).sum(-1)   # [B]
    gen_pos = ((probs[..., K_IDX] + probs[..., R_IDX]) * pmask).sum(-1)

    def _one_direction(n, gen):
        if n > 0.5:
            if normalize:
                floor_loss = torch.relu(floor - gen / n)
                ceil_loss = torch.relu(gen / n - ceil)
            else:
                floor_loss = torch.relu(n * floor - gen)
                ceil_loss = torch.relu(gen - n * ceil)
        else:
            floor_loss = torch.zeros_like(gen)
            if normalize:
                ceil_loss = torch.relu(gen - min_abs_cap) / min_abs_cap
            else:
                ceil_loss = torch.relu(gen - min_abs_cap)
        return floor_loss + ceil_loss

    loss = (_one_direction(n_neg, gen_neg) + _one_direction(n_pos, gen_pos)).mean()
    return loss
