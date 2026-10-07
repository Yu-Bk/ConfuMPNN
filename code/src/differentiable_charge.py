"""Differentiable pH-aware net-charge calculation.

Based on a **smooth (differentiable) approximation** to the Henderson-Hasselbalch equation:

    Deprotonated fraction = 1 / (1 + 10^(pH - pKa)) = σ(ln(10)·(pH - pKa))

Here, σ is the sigmoid function. Replacing a hard threshold with a sigmoid makes charge differentiable with respect to pH,
allowing gradients to be backpropagated (as an auxiliary loss during Level 2 fine-tuning).

Charge rules (using free-state pKa values; see pka.py):
  - Acidic residues D/E/C/Y: deprotonated form carries -1, charge = -σ(ln10·(pH - pKa))
  - Basic residues K/R/H: protonated form carries +1, charge = +σ(ln10·(pKa - pH))
  - N-terminal α-NH3+ (pKa≈9.7): charge = +σ(ln10·(9.7 - pH))
  - C-terminal α-COOH (pKa≈2.3): charge = -σ(ln10·(pH - 2.3))

Usage:
    # String sequence (for Phase 1 validation)
    charge = net_charge("ACDEK", pH=7.4)

    # Decoder logits (differentiable; for Phase 2 training)
    charge_batch = net_charge_from_logits(logits, pH=7.4, mask=mask)
"""

import math

import torch

from .pka import (AA_TO_IDX, AAS, ACIDIC, PKA_C_TERM, PKA_N_TERM,
                  PKA_SIDECHAIN)

LN10 = math.log(10.0)


def _to_tensor(value, dtype=torch.float32, device=None):
    """Safely convert a Python scalar or list to a torch tensor."""
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def sidechain_charge(aa, pH):
    """Compute the (partial) charge of one amino-acid side chain at a given pH.

    Returns a differentiable scalar tensor.

    Parameters:
        aa: single uppercase amino-acid letter (e.g., "D")
        pH: scalar or tensor
    """
    pH = _to_tensor(pH)
    pKa = PKA_SIDECHAIN.get(aa)
    if pKa is None:
        return torch.zeros_like(pH)  # Non-ionizable residues contribute zero
    if aa in ACIDIC:
        # Acidic: the deprotonated form carries -1; fraction = σ(ln10·(pH - pKa))
        return -torch.sigmoid(LN10 * (pH - pKa))
    else:
        # Basic (K/R/H): the protonated form carries +1; fraction = σ(ln10·(pKa - pH))
        return torch.sigmoid(LN10 * (pKa - pH))


def sidechain_charge_vector(pH):
    """Return a tensor of shape [20] with side-chain charges for the 20 standard amino acids at the given pH.

    Order matches the LigandMPNN restype (AAS = "ACDEFGHIKLMNPQRSTVWY").
    """
    pH = _to_tensor(pH)
    return torch.stack([sidechain_charge(a, pH) for a in AAS])


def _termini_charge(pH, n_term=True):
    """Compute the terminal backbone charges.

    The N-terminal α-NH3+ (pKa=9.7) carries +1 when protonated; the C-terminal α-COOH (pKa=2.3) carries -1 when deprotonated.
    Returns a scalar tensor.
    """
    pH = _to_tensor(pH)
    if n_term:
        return torch.sigmoid(LN10 * (PKA_N_TERM - pH))
    return -torch.sigmoid(LN10 * (pH - PKA_C_TERM))


def net_charge(seq, pH, include_termini=True):
    """Compute the net charge of a **string sequence** at a given pH (returns a float).

    Parameters:
        seq: single-letter amino-acid sequence, e.g., "ACDEK" (case-insensitive; skips "-", "X", and invalid characters)
        pH: operating pH (scalar)
        include_termini: whether to include N- and C-terminal backbone charges (default True)
    """
    seq = "".join(a for a in seq.upper() if a in AA_TO_IDX)
    pH = _to_tensor(pH)
    if not seq:
        raise ValueError("Input sequence is empty or contains no valid amino acids")
    total = torch.zeros_like(pH)
    for aa in seq:
        total = total + sidechain_charge(aa, pH)
    if include_termini:
        total = total + _termini_charge(pH, n_term=True)
        total = total + _termini_charge(pH, n_term=False)
    return total.item()


def net_charge_from_logits(logits, pH, mask=None, include_termini=True, temperature=1.0):
    """Compute the **expected net charge** from decoder logits (differentiable; for Phase 2 training).

    Use softmax probabilities to take a weighted average of the 20 amino-acid charges at each position, then sum over the sequence.
    Because N- and C-terminal charges depend on pH but not residue choice, they have no gradient with respect to logits
    and are added directly as constants (every residue has α-NH3+/α-COOH termini).

    Parameters:
        logits: logits of shape [B, L, 21] or [B, L, 20] (for 21 dimensions, only the first 20 AA are used)
        pH: operating pH (scalar)
        mask: valid-residue mask [B, L] (1=valid, 0=ignored); defaults to all valid
        include_termini: whether to include terminal backbone charges
        temperature: softmax temperature (default 1.0). Values <1 sharpen the distribution, bringing E[Q] closer to
            the charge of the argmax sequence, which is sampled at inference (temperature 0.3),
            so the training objective better matches inference sampling.

    Returns:
        Tensor of shape [B]: expected net charge for each sample
    """
    if logits.shape[-1] == 21:
        logits = logits[..., :20]
    probs = torch.softmax(logits / temperature, dim=-1)          # [B, L, 20]
    B, L, _ = probs.shape
    if mask is None:
        mask = torch.ones(B, L, device=logits.device)
    mask = mask.float()

    Q = sidechain_charge_vector(pH).to(logits.dtype).to(logits.device)  # [20]
    side_charge = torch.einsum("blk,k->bl", probs, Q)    # [B, L] expected side-chain charge
    total = (side_charge * mask).sum(dim=-1)             # [B]

    if include_termini and L > 0:
        has_residue = (mask.sum(dim=-1) > 0).float()     # [B]
        n_charge = _termini_charge(pH, n_term=True).to(logits.dtype).to(logits.device)
        c_charge = _termini_charge(pH, n_term=False).to(logits.dtype).to(logits.device)
        total = total + has_residue * (n_charge + c_charge)
    return total


if __name__ == "__main__":
    # Quick self-check: D/E are acidic and K/R are basic; at pH=7.4, each should contribute ±1
    for aa, expect in [("D", -1.0), ("E", -1.0), ("K", 1.0), ("R", 1.0)]:
        got = sidechain_charge(aa, 7.4).item()
        print(f"{aa} @ pH7.4 = {got:+.3f}  (expect ~{expect:+.1f})")

    seq = "ACDEK"
    print(f"net_charge('{seq}', pH=7.4) = {net_charge(seq, 7.4):+.3f}")
