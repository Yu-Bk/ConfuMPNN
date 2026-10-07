"""Find a sequence's isoelectric point (pI) by binary search.

The isoelectric point is the pH at which a molecule's net charge is zero. For
a given sequence, it is determined by the amino acid composition.

Net charge decreases monotonically as pH increases, so binary search can be
used to locate the pI efficiently.
"""

import torch

from .differentiable_charge import net_charge


def find_pI(seq, lo=0.0, hi=14.0, tol=1e-4, max_iter=200):
    """Binary-search the pH in [lo, hi] for which net_charge(seq, pH) == 0.

    Args:
        seq: Amino acid sequence in one-letter code.
        lo, hi: Search interval (defaults to the full pH range [0, 14]).
        tol: Charge tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        The sequence's isoelectric point as a float.
    """
    charge_lo = net_charge(seq, lo, include_termini=True)
    charge_hi = net_charge(seq, hi, include_termini=True)
    if charge_lo * charge_hi > 0:
        # If the charge has the same sign at both bounds, return the interval midpoint.
        return 0.5 * (lo + hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        charge_mid = net_charge(seq, mid, include_termini=True)
        if abs(charge_mid) < tol:
            return mid
        if charge_mid > 0:
            lo = mid   # The pH is too low (net positive); search higher.
        else:
            hi = mid   # The pH is too high (net negative); search lower.
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    # Check that sequences rich in basic residues have higher pI values than acidic sequences.
    for seq in ["RRRRR", "KKEEDD", "ACDEFGHIKLMNPQRSTVWY", "AAAAA"]:
        print(f"pI({seq}) = {find_pI(seq):.2f}")
