"""Guided sampler: wraps the LigandMPNN decoder and injects pH-aware logit bias.

Implements guided sampling without modifying the model code.

Two usage modes:

1. **Static bias**: compute bias once (structure-aware filter + base omit/bias),
   populate `feature_dict["bias"]`, and call the original LigandMPNN `model.sample`.
   Suitable for rules that do not depend on the already-generated sequence.

2. **Dynamic stepwise decoding**: reproduces LigandMPNN's decoding loop (batch=1, asymmetric/single-chain),
   calling `bias_callback(S_cur, t)` at each residue to compute that position's bias in real time.
   Supports "differentiable net-charge lookahead" (look ahead over candidate amino acids at each step) and real-time
   filtering based on the generated sequence (e.g., the structure-aware filter counts charge clustering among decoded residues).

Key mechanism (see `sample` in model_utils.py):
    probs = softmax((logits + bias_t) / temperature)
Thus, bias is added directly to the logits, shape [B, L, 21] (first 20 AA + X).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make model_utils from LigandMPNN importable when this module is imported independently
_LIGAND_MPNN_DIR = Path(__file__).resolve().parents[2] / "LigandMPNN"
if str(_LIGAND_MPNN_DIR) not in sys.path:
    sys.path.insert(0, str(_LIGAND_MPNN_DIR))

from model_utils import cat_neighbors_nodes  # noqa: E402

from .pka import AA_TO_IDX  # noqa: E402
from .structure_aware_filter import UNDECODED  # noqa: E402


def extract_calpha_coords(protein_dict):
    """Extract Cα coordinates [L, 3] from protein_dict returned by parse_PDB.

    The X field returned by parse_PDB has shape [L, 4, 3], ordered N, CA, C, O.
    """
    X = protein_dict["X"]
    if torch.is_tensor(X):
        X = X.cpu().numpy()
    return X[:, 1, :]  # CA index = 1


def build_static_bias(feature_dict, structure_filter, seq_ref=None, base_bias=None):
    """Precompute static bias = base bias (omit/bias) + structure-filter bias.

    Parameters:
        feature_dict: dictionary returned by featurize (uses its X length and base bias)
        structure_filter: StructureAwareFilter instance
        seq_ref: reference sequence [L] (int); if None, treat as "all undecoded" (structure rules only,
                 without charge clustering because the sequence is unknown)
        base_bias: base bias [1, L, 21] (e.g., omit_AA); use all zeros if None

    Returns:
        (bias[1, L, 21], info_dict)
    """
    X = feature_dict["X"]
    L = X.shape[1] if X.ndim == 4 else X.shape[0]
    if seq_ref is None:
        seq_ref = np.full(L, UNDECODED, dtype=np.int64)
    fb, info = structure_filter.compute_bias(seq_ref)

    if base_bias is None:
        base_bias = torch.zeros(1, L, 21, dtype=torch.float32)
    return base_bias + fb.unsqueeze(0), info


def guided_sample(model, feature_dict, bias_callback=None, device="cpu", encoded=None):
    """Main guided-sampling function (batch=1, asymmetric decoding).

    Parameters:
        model: LigandMPNN model (ProteinMPNN instance)
        feature_dict: dictionary returned by featurize; must contain X/Y/S/mask/chain_mask/randn/
                      temperature/bias (bias provides base biases such as omit)
        bias_callback: callable(S_cur, t) -> [21] or None
            S_cur: [L] partially decoded sequence (int; 20=X means undecoded)
            t: current decoding position
            Returns the bias increment for this position (shape [21], added to logits)
        device: compute device
        encoded: optional pre-encoded (h_V, h_E, E_idx). If provided, skip encode;
            used for condition injection (Phase 3; h_V already contains the soft-prompt signal).

    Returns:
        dict {S, sampling_probs, log_probs, decoding_order}
    """
    model = model.to(device)
    base_bias = feature_dict["bias"].to(device).float()  # [1, L, 21]
    S_true = feature_dict["S"].to(device)
    mask = feature_dict["mask"].to(device)
    chain_mask = feature_dict["chain_mask"].to(device)
    temperature = float(feature_dict["temperature"])
    randn = feature_dict["randn"].to(device)

    B, L = S_true.shape
    assert B == 1, "guided_sample currently supports batch_size=1 only"

    if encoded is not None:
        h_V, h_E, E_idx = encoded
    else:
        h_V, h_E, E_idx = model.encode(feature_dict)

    chain_mask = mask * chain_mask
    decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))

    # Attention mask for asymmetric (single-chain) decoding
    permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=L).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(L, L, device=device))),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([B, L, 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    B_decoder = 1
    S_true = S_true.repeat(B_decoder, 1)
    h_V = h_V.repeat(B_decoder, 1, 1)
    h_E = h_E.repeat(B_decoder, 1, 1, 1)
    chain_mask = chain_mask.repeat(B_decoder, 1)
    mask = mask.repeat(B_decoder, 1)
    base_bias = base_bias.repeat(B_decoder, 1, 1)

    all_probs = torch.zeros((B_decoder, L, 20), device=device)
    all_log_probs = torch.zeros((B_decoder, L, 21), device=device)
    h_S = torch.zeros_like(h_V)
    S = 20 * torch.ones((B_decoder, L), dtype=torch.int64, device=device)
    h_V_stack = [h_V] + [
        torch.zeros_like(h_V, device=device) for _ in range(len(model.decoder_layers))
    ]

    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    for t_ in range(L):
        t = decoding_order[:, t_]  # [B_decoder]
        chain_mask_t = torch.gather(chain_mask, 1, t[:, None])[:, 0]
        mask_t = torch.gather(mask, 1, t[:, None])[:, 0]

        # Bias for this position: dynamic callback (preferred) or base bias
        if bias_callback is not None:
            S_cur = S[0].cpu().numpy()  # [L] current partially decoded sequence
            dyn = np.asarray(bias_callback(S_cur, int(t[0])), dtype=np.float32)
            bias_t = (base_bias[0, int(t[0])] + torch.from_numpy(dyn).to(device)).view(1, -1)
        else:
            bias_t = torch.gather(base_bias, 1, t[:, None, None].repeat(1, 1, 21))[:, 0, :]

        E_idx_t = torch.gather(
            E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1])
        )
        h_E_t = torch.gather(
            h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1])
        )
        h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
        h_EXV_encoder_t = torch.gather(
            h_EXV_encoder_fw,
            1,
            t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2],
                                          h_EXV_encoder_fw.shape[-1]),
        )
        mask_bw_t = torch.gather(
            mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2],
                                                      mask_bw.shape[-1])
        )

        for l, layer in enumerate(model.decoder_layers):
            h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
            h_V_t = torch.gather(
                h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1])
            )
            h_ESV_t = mask_bw_t * h_ESV_decoder_t + h_EXV_encoder_t
            h_V_stack[l + 1].scatter_(
                1,
                t[:, None, None].repeat(1, 1, h_V.shape[-1]),
                layer(h_V_t, h_ESV_t, mask_V=mask_t),
            )

        h_V_t = torch.gather(
            h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1])
        )[:, 0]
        logits = model.W_out(h_V_t)  # [B_decoder, 21]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = F.softmax((logits + bias_t) / temperature, dim=-1)
        probs_sample = probs[:, :20] / torch.sum(probs[:, :20], dim=-1, keepdim=True)
        S_t = torch.multinomial(probs_sample, 1)[:, 0]

        all_probs.scatter_(
            1,
            t[:, None, None].repeat(1, 1, 20),
            (chain_mask_t[:, None, None] * probs_sample[:, None, :]).float(),
        )
        all_log_probs.scatter_(
            1,
            t[:, None, None].repeat(1, 1, 21),
            (chain_mask_t[:, None, None] * log_probs[:, None, :]).float(),
        )
        S_true_t = torch.gather(S_true, 1, t[:, None])[:, 0]
        S_t = (S_t * chain_mask_t + S_true_t * (1.0 - chain_mask_t)).long()
        h_S.scatter_(
            1,
            t[:, None, None].repeat(1, 1, h_S.shape[-1]),
            model.W_s(S_t)[:, None, :],
        )
        S.scatter_(1, t[:, None], S_t[:, None])

    return {
        "S": S,
        "sampling_probs": all_probs,
        "log_probs": all_log_probs,
        "decoding_order": decoding_order,
    }


class GuidedSampler:
    """Guided sampler (high-level wrapper).

    Usage:
        sampler = GuidedSampler(model)
        out = sampler.sample(feature_dict)                    # Static bias
        out = sampler.sample(feature_dict, bias_callback=fn)  # Dynamic stepwise bias
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, feature_dict, bias_callback=None):
        """Run guided sampling.

        feature_dict must already contain the base bias key ("bias"; see LigandMPNN run.py).
        If bias_callback is None, behavior is equivalent to the original model.sample (but uses this module's
        decoding loop); passing a callback enables a live bias at each step.
        """
        fd = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in feature_dict.items()
        }
        return guided_sample(
            self.model, fd, bias_callback=bias_callback, device=self.device
        )


if __name__ == "__main__":
    # Self-check: validate only the dimensions of build_static_bias; no real model required
    from .structure_aware_filter import StructureAwareFilter

    L = 10
    fake_X = torch.randn(1, L, 4, 3)  # [B, L, 4, 3]
    coords = fake_X[0, :, 1].numpy()
    filt = StructureAwareFilter(coords)
    fd = {"X": fake_X}
    bias, info = build_static_bias(fd, filt, seq_ref=None)
    print("static bias shape:", tuple(bias.shape), "(expect [1, 10, 21])")
    print("filter info:", info)
