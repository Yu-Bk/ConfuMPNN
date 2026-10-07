"""Compute a dynamic charge lookahead and return a logit bias at each step.

The additive charge model estimates the sequence's net charge for each of the
20 standard amino acid candidates at the current position:

        Q_k = Q_fixed + q(aa_k, pH) + Q_expect_others + Q_termini

    - Q_fixed: Sum of side-chain charges at decoded positions.
    - q(aa_k, pH): Candidate side-chain charge at the specified pH.
    - Q_expect_others: Expected charge at other undecoded positions, approximated
      by the mean charge across the 20 amino acids.
    - Q_termini: N- and C-terminal charges, which depend on pH.

The candidate bias is based on the difference between the target and the
estimated current charge. Including the candidate-dependent term is necessary
because adding the same constant to all logits does not change a softmax:

        Q_current = Q_fixed + Q_expect_others + Q_termini   # excluding the candidate position
        bias_k   = strength · (target_charge − Q_current) · q(aa_k, pH)

    If Q_current is below the target, positively charged candidates receive a
    positive bias and negatively charged candidates receive a negative bias;
    the direction reverses when Q_current is above the target.

The strength parameter scales the bias. The charge lookahead can be combined
with the optional structure-aware filter in `make_dynamic_callback`.
"""

import numpy as np

from .differentiable_charge import (_termini_charge,
                                    sidechain_charge_vector)
from .pka import AAS, PKA_SIDECHAIN

# Undecoded marker (20 = X).
UNDECODED = 20


class ChargeLookahead:
    """Compute candidate logit biases from a dynamic charge lookahead.

    Args:
        pH: Working pH, which determines each residue's charge.
        target_charge: Target net charge; None disables charge guidance.
        strength: Scale factor for the bias.
        include_termini: Whether to include N- and C-terminal charges.
    """

    def __init__(self, pH, target_charge=None, strength=0.5, include_termini=True):
        self.pH = float(pH)
        self.target_charge = target_charge
        self.strength = float(strength)
        self.include_termini = include_termini
        # Precompute the side-chain charge vector for the 20 amino acids and its mean.
        self._q_vec = sidechain_charge_vector(self.pH).numpy()
        self._q_mean = float(self._q_vec.mean())

    def _charge_of(self, aa_idx):
        """Return the side-chain charge at position i (aa_idx is an AAS index)."""
        return float(self._q_vec[aa_idx])

    def bias_at(self, seq_int, position, mask=None):
        """Compute the candidate-specific bias at `position`.

        Args:
            seq_int: Current partially decoded integer sequence; 20 denotes X/undecoded.
            position: Position currently being decoded.
            mask: Valid-residue mask; None treats every position as valid.
        Returns:
            A [21] NumPy array: one bias per amino acid, followed by a zero for X.
        """
        if self.target_charge is None:
            return np.zeros(21, dtype=np.float32)

        seq_int = np.asarray(seq_int)
        L = len(seq_int)
        if mask is None:
            mask = np.ones(L, dtype=bool)
        mask = np.asarray(mask, dtype=bool)

        # 1) Fixed charge at decoded positions, excluding the current and invalid positions.
        fixed = 0.0
        n_undecoded_excl = 0  # Number of other undecoded positions.
        for i in range(L):
            if i == position or not mask[i]:
                continue
            if seq_int[i] != UNDECODED:
                fixed += self._charge_of(int(seq_int[i]))
            else:
                n_undecoded_excl += 1

        # 2) Expected charge at other undecoded positions, approximated by the mean side-chain charge.
        expect_others = n_undecoded_excl * self._q_mean

        # 3) Terminal charges, if the sequence is nonempty.
        termini = 0.0
        if self.include_termini and mask.sum() > 0:
            termini = float(_termini_charge(self.pH, n_term=True).item()) + float(
                _termini_charge(self.pH, n_term=False).item()
            )

        # 4) Current net charge, excluding the candidate position.
        Q_current = fixed + expect_others + termini

        # 5) Compute a bias for each of the 20 candidates. The charge term varies
        #    by candidate, so the target-dependent contribution is not canceled
        #    by the softmax invariance to a shared additive constant.
        biases = np.zeros(21, dtype=np.float32)
        biases[:20] = self.strength * (self.target_charge - Q_current) * self._q_vec
        return biases


def make_dynamic_callback(pH, target_charge=None, structure_filter=None,
                          strength=0.5):
    """Build a dynamic bias callback for GuidedSampler.

    Combines the charge lookahead with an optional structure-aware filter.

    The returned function matches the guided_sampler bias_callback signature:
        fn(S_cur, t) -> [21] NumPy array

    Args:
        pH: Working pH.
        target_charge: Target net charge; None disables charge guidance.
        structure_filter: StructureAwareFilter instance, or None to disable filtering.
        strength: Charge-guidance scale factor.
    """
    lookahead = ChargeLookahead(
        pH, target_charge=target_charge, strength=strength
    )

    def callback(S_cur, t):
        bias = lookahead.bias_at(S_cur, t)
        if structure_filter is not None:
            # Pass the working pH so the filter can include weakly charged residues by protonation state.
            fb, _ = structure_filter.compute_bias(S_cur, pH=pH)
            bias = bias + fb[t].numpy()
        return bias

    return callback


if __name__ == "__main__":
    # Check that target charges of 8, 0, and -8 produce distinct candidate biases.
    import numpy as np
    from .pka import AA_TO_IDX

    seq = [AA_TO_IDX["K"]] * 5 + [UNDECODED] * 5  # Five decoded K residues, then five undecoded positions.
    k_idx = AA_TO_IDX["K"]
    d_idx = AA_TO_IDX["D"]
    results = {}
    for tgt in (8.0, 0.0, -8.0):
        la = ChargeLookahead(pH=7.4, target_charge=tgt, strength=0.5)
        b = la.bias_at(np.array(seq), position=5)
        results[tgt] = (b[k_idx], b[d_idx])
        print(f"target={tgt:+5.1f}  bias[K]={b[k_idx]:+.3f}  bias[D]={b[d_idx]:+.3f}")

    # Assertion 1: the K bias must differ for each target charge.
    k_biases = {t: v[0] for t, v in results.items()}
    assert len(set(round(v, 3) for v in k_biases.values())) == 3, \
        "Candidate biases do not vary with the target charge."
    # Assertion 2: a positive target favors K, while a negative target suppresses it.
    assert results[8.0][0] > 0 > results[-8.0][0], "Unexpected bias sign for K across target charges."
    # Assertion 3: for target 0, K and D biases have opposite signs.
    assert results[0.0][0] < 0 < results[0.0][1], "Unexpected bias signs for target_charge=0."
    print("Self-check passed: target charges 8/0/-8 produce distinct biases.")
