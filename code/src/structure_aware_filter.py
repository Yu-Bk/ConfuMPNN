"""Apply structure-aware logit biases during sequence decoding.

When a rule is triggered, a negative bias is applied to candidate amino acids
at affected undecoded positions. The rules are:

1. **Spatial charge crowding:** suppress additional same-sign charged residues
   when the number of decoded same-sign residues within 10 Å reaches the threshold.
2. **Dense salt bridges:** suppress charged candidates when the estimated number
   of positive-negative pairs within 10 Å reaches the threshold.
3. **Core charge penetration:** suppress charged candidates at buried positions
   when the number of nearby charged residues within 8 Å reaches the threshold.
4. **Same-sign spatial clustering:** suppress same-sign charged candidates in a
   connected component within 8 Å when its decoded charge count reaches the threshold.

Coordinates use Cα atoms as approximate residue positions. Distances and other
thresholds can be configured with YAML presets. Charge counts include decoded
residues only (positions whose sequence index is not X).
"""

from pathlib import Path

import numpy as np
import torch

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .pka import AA_TO_IDX, PKA_SIDECHAIN, STRONG_NEGATIVE, STRONG_POSITIVE

# Undecoded marker (consistent with LigandMPNN restype: 20 = X/unknown).
UNDECODED = 20

# Amino acid indices in the [L, 21] bias tensor: 20 standard residues followed by X.
POS_AA_IDX = [AA_TO_IDX[a] for a in STRONG_POSITIVE]  # K, R
NEG_AA_IDX = [AA_TO_IDX[a] for a in STRONG_NEGATIVE]  # D, E
CHARGED_AA_IDX = POS_AA_IDX + NEG_AA_IDX


def pH_adaptive_charged_aa(pH=None):
    """Return the amino acids treated as charged at the working pH.

    Returns single-letter `(pos_aa, neg_aa)` tuples. With `pH=None`, only the
    strongly charged residues K/R/D/E are included. When pH is provided, His is
    included as positive at or below pKa 6.0, while Cys and Tyr are included as
    negative at or above pKa 8.3 and 10.1, respectively. The relevant charged
    fraction reaches 0.5 at pH = pKa.
    """
    pos = list(STRONG_POSITIVE)
    neg = list(STRONG_NEGATIVE)
    if pH is not None:
        if pH <= PKA_SIDECHAIN["H"]:
            pos.append("H")
        if pH >= PKA_SIDECHAIN["C"]:
            neg.append("C")
        if pH >= PKA_SIDECHAIN["Y"]:
            neg.append("Y")
    return tuple(pos), tuple(neg)


def default_config():
    """Return the default rule thresholds and descriptions."""
    return {
        "charge_cluster": {
            "radius": 10.0, "threshold": 6, "strength": -1.0,
            "desc": "Suppress additional same-sign charged residues when at least 6 are within 10 Å.",
        },
        "salt_bridge": {
            "radius": 10.0, "threshold": 4, "strength": -1.0,
            "desc": "Suppress additional charged residues when at least 4 positive-negative pairs are within 10 Å.",
        },
        "core_charge": {
            "burial_radius": 10.0, "charge_radius": 8.0,
            "burial_threshold": 0.8, "charge_count": 6, "strength": -1.0,
            "desc": "Suppress charged residues at buried positions with at least 6 charged residues within 8 Å.",
        },
        "same_sign_cluster": {
            "radius": 8.0, "threshold": 4, "strength": -1.0,
            "desc": "Suppress same-sign charged residues when at least 4 are in an 8 Å neighborhood.",
        },
    }


class StructureAwareFilter:
    """Structure-aware filter for applying logit biases during decoding.

    Args:
        coords: [L, 3] residue Cα coordinates (PyTorch tensor or NumPy array).
        mask: [L] valid-residue mask (1 = valid); defaults to all valid.
        config: Rule-threshold dictionary; defaults to `default_config()`.
    """

def load_preset(preset="default", path=None):
    """Load a rule preset from code/configs/filter_presets.yaml.

    Args:
        preset: Preset name (default / nucleic_acid_binding / membrane / acidic).
        path: YAML file path; None locates code/configs/filter_presets.yaml automatically.
    Returns:
        Rule configuration that can be passed to `StructureAwareFilter(config=...)`.
    """
    import yaml

    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "filter_presets.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    presets = data["presets"]
    if preset not in presets:
        raise KeyError(f"Unknown preset '{preset}'. Available presets: {list(presets)}")
    cfg = presets[preset]
    # Keep only rule dictionaries and discard metadata fields such as desc and ph_hint.
    return {k: v for k, v in cfg.items() if isinstance(v, dict)}


class StructureAwareFilter:
    """Structure-aware filter for applying logit biases during decoding.

    Args:
        coords: [L, 3] residue Cα coordinates (PyTorch tensor or NumPy array).
        mask: [L] valid-residue mask (1 = valid); defaults to all valid.
        config: Rule-threshold dictionary; defaults to `default_config()`.
    """

    def __init__(self, coords, mask=None, config=None):
        coords = np.asarray(coords, dtype=np.float64)
        self.coords = coords
        self.L = coords.shape[0]
        self.mask = mask if mask is not None else np.ones(self.L, dtype=bool)
        self.config = config if config is not None else default_config()
        self._dist = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        """Return the [L, L] Euclidean distance matrix for Cα coordinates."""
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    @staticmethod
    def _decode_charge_flags(seq_int, pos_aa=None, neg_aa=None):
        """Extract positive and negative charge flags from a decoded [L] integer sequence.

        Only decoded positions are marked. `pos_aa` and `neg_aa` default to
        strongly charged K/R and D/E, respectively.
        """
        pos_aa = pos_aa or STRONG_POSITIVE
        neg_aa = neg_aa or STRONG_NEGATIVE
        pos = np.zeros(len(seq_int), dtype=bool)
        neg = np.zeros(len(seq_int), dtype=bool)
        for a in pos_aa:
            pos |= (seq_int == AA_TO_IDX[a])
        for a in neg_aa:
            neg |= (seq_int == AA_TO_IDX[a])
        return pos, neg

    def _masked(self, mat):
        """Keep only valid residues (mask); set invalid positions to False."""
        return mat & self.mask[None, :]

    @staticmethod
    def _apply_bias(bias, mask, col_idxs, strength):
        """Add strength to the specified columns for rows where mask is True.

        `mask` may be a NumPy boolean array; `col_idxs` is a list of column
        indices (such as [K, R]). Write one column at a time to avoid a shape
        mismatch when combining a boolean mask with multi-column advanced indexing.
        """
        m = (
            torch.from_numpy(mask)
            if not torch.is_tensor(mask)
            else mask.to(torch.bool)
        )
        for k in col_idxs:
            bias[m, k] += strength

    def compute_bias(self, seq_int, pH=None):
        """Compute bias increments from the current decoded sequence.

        Args:
            seq_int: [L] integer sequence; 20 = X marks undecoded positions.
            pH: Working pH. When provided, His, Cys, and Tyr are classified by
                their charged fractions (pKa 6.0, 8.3, and 10.1); None includes
                only the strongly charged residues K/R/D/E.

        Returns:
            bias: [L, 21] float tensor; negative values suppress candidates when
                added to logits.
            info: Per-rule trigger counts.
        """
        seq_int = np.asarray(seq_int)
        pos_aa, neg_aa = pH_adaptive_charged_aa(pH)
        pos_idx = [AA_TO_IDX[a] for a in pos_aa]
        neg_idx = [AA_TO_IDX[a] for a in neg_aa]
        charged_idx = pos_idx + neg_idx
        pos, neg = self._decode_charge_flags(seq_int, pos_aa, neg_aa)
        undecoded = (seq_int == UNDECODED) & self.mask  # Only these positions can receive a bias.
        charged = pos | neg

        bias = torch.zeros(self.L, 21)
        info = {}

        # Rule 1: spatial charge crowding (same-sign charges within the configured radius).
        cfg = self.config["charge_cluster"]
        nb = self._masked(self._dist <= cfg["radius"])
        # Count decoded positive and negative charges in each neighborhood, including the position itself.
        pos_count = (nb & pos[None, :]).sum(axis=1)
        neg_count = (nb & neg[None, :]).sum(axis=1)
        over_pos = undecoded & (pos_count >= cfg["threshold"])
        over_neg = undecoded & (neg_count >= cfg["threshold"])
        self._apply_bias(bias, over_pos, pos_idx, cfg["strength"])
        self._apply_bias(bias, over_neg, neg_idx, cfg["strength"])
        info["charge_cluster"] = {
            "pos_over": int(over_pos.sum()), "neg_over": int(over_neg.sum()),
        }

        # Rule 2: dense salt bridges (positive-negative pairs within the configured radius).
        cfg = self.config["salt_bridge"]
        # Approximate formed or potential salt-bridge pairs as min(positive count, negative count).
        pairs = np.minimum(pos_count, neg_count)
        over_bridge = undecoded & (pairs >= cfg["threshold"])
        self._apply_bias(bias, over_bridge, charged_idx, cfg["strength"])
        info["salt_bridge"] = {"over": int(over_bridge.sum())}

        # Rule 3: charge penetration into the core (burial and nearby charged-residue thresholds).
        cfg = self.config["core_charge"]
        nb_burial = self._masked(self._dist <= cfg["burial_radius"])
        burial_count = nb_burial.sum(axis=1)  # Approximate burial by the number of Cα atoms within 10 Å.
        max_burial = burial_count.max()
        burial_ratio = burial_count / max_burial if max_burial > 0 else burial_count
        nb_charge = self._masked(self._dist <= cfg["charge_radius"])
        charge_count = (nb_charge & charged[None, :]).sum(axis=1)
        core = undecoded & (burial_ratio > cfg["burial_threshold"]) & (
            charge_count >= cfg["charge_count"]
        )
        self._apply_bias(bias, core, charged_idx, cfg["strength"])
        info["core_charge"] = {"core": int(core.sum())}

        # Rule 4: same-sign spatial clustering in the connected-component graph.
        cfg = self.config["same_sign_cluster"]
        adj = csr_matrix(
            (self._dist <= cfg["radius"]).astype(int)
        )  # Connectivity is based on structure and does not use the mask.
        n_comp, labels = connected_components(adj, directed=False)
        for c in range(n_comp):
            members = labels == c
            n_pos = int((members & pos).sum())
            n_neg = int((members & neg).sum())
            tgt = members & undecoded
            if n_pos >= cfg["threshold"]:
                self._apply_bias(bias, tgt, pos_idx, cfg["strength"])
            if n_neg >= cfg["threshold"]:
                self._apply_bias(bias, tgt, neg_idx, cfg["strength"])
        info["same_sign_cluster"] = {"n_components": int(n_comp)}

        return bias, info


if __name__ == "__main__":
    # Check that eight K residues on a line trigger rules 1 and 4.
    coords = np.zeros((8, 3))
    coords[:, 0] = np.arange(8, dtype=float)
    seq = np.array([AA_TO_IDX["K"]] * 8)
    filt = StructureAwareFilter(coords)
    b, info = filt.compute_bias(seq)
    print("rules:", info)
    print("bias shape:", b.shape, "| nonzero rows:", int((b != 0).any(dim=1).sum()))
