"""Conditional fine-tuning trainer for pH-dependent charge guidance.

Architecture
    backbone         = frozen MoMPNN
    ConditionEncoder = Soft Prompt MLP (condition_embedding.py)

Inject the soft prompt through cross-attention, without changing the decoder:
    h_V += softmax(h_V · prompt^T / √d) · prompt          # [B,L,128] + [B,L,128]
Each structure node reads the four condition tokens as needed.

Loss
    L = CE + λ_c·charge_deviation + λ_kl·KL(conditioned ‖ unconditioned) + λ_keep·SeqKeep

    CE                : reconstruct the native sequence
    charge_deviation  : expected net charge vs. target charge (differentiable, differentiable_charge.py)
    KL-anchor         : keep the conditioned output distribution close to the unconditional backbone distribution
    SeqKeep           : on self-consistent samples (target=native), bring conditioned output toward the unconditional argmax sequence
                        **Apply only to self-consistent samples**; charge shifts are expected for perturbed samples (target≠native).

Training targets are mixed:
    · 70%: self-consistent target = net charge of the native sequence at that pH
    · 30%: perturbed target = native charge ± Uniform[1, perturb_scale]
This mix supplies both a native-sequence reconstruction signal and examples with target-dependent charge shifts.
The mixing parameters λ_c / λ_kl / λ_keep / perturb_prob / perturb_scale can be set from the command line.

Usage
    conda activate confumpnn
    python train_finetune.py --device cuda:1 \
      --epochs 30 --out_dir ../output/finetune
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CODE_DIR.parent
_LIG_DIR = _ROOT_DIR / "LigandMPNN"
for p in [str(_CODE_DIR), str(_LIG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_utils import featurize, parse_PDB  # noqa: E402
from model_utils import ProteinMPNN, cat_neighbors_nodes  # noqa: E402

from src.condition_embedding import ConditionEncoder, make_condition_vector  # noqa: E402
from src.conditioned_sampler import inject_prompt  # noqa: E402  (same injection mechanism for training and inference)
from src.differentiable_charge import net_charge  # noqa: E402
from src.losses import (  # noqa: E402
    charge_deviation_loss, cross_entropy_loss, sequence_keep_loss,
)
from src.v10_losses import (  # noqa: E402
    ph_aware_structure_penalty, surface_add_charge_loss,
)
from src.v12_losses import (  # noqa: E402
    surface_composition_loss, surface_gravy_loss, surface_charge_target_loss,
    pocket_count_loss, KD, D_IDX, E_IDX, K_IDX, R_IDX,
)

# Default generator weights (consistent with E4 in run_guided.py)
_DEFAULT_WEIGHTS = (
    _ROOT_DIR / "MoMPNN" / "mompnn_paper_checkpoints"
    / "mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
)
_DEFAULT_LABELS = _ROOT_DIR / "data" / "cath" / "labels.npz"
_DEFAULT_DOMPDB = _ROOT_DIR / "data" / "cath" / "S40" / "dompdb"
_DEFAULT_CFG = _CODE_DIR / "configs" / "condition_defaults.yaml"


def parse_args():
    p = argparse.ArgumentParser(description="ConfuMPNN Phase 2 conditional fine-tuning")
    p.add_argument("--weights", default=str(_DEFAULT_WEIGHTS))
    p.add_argument("--labels", default=str(_DEFAULT_LABELS))
    p.add_argument("--dompdb", default=str(_DEFAULT_DOMPDB))
    p.add_argument("--cfg", default=str(_DEFAULT_CFG))
    p.add_argument("--ligand", action="store_true",
                   help="Ligand mode: use LigandMPNN weights and featurize ligand-atom context")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_c", type=float, default=0.5,
                   help="Charge-deviation loss weight")
    p.add_argument("--lambda_kl", type=float, default=0.05,
                   help="KL-anchor regularization weight (prevents runaway drift of conditioned output from the backbone)")
    p.add_argument("--lambda_keep", type=float, default=0.5,
                   help="Sequence-preservation regularization weight (conditioned output approaches unconditional argmax on "
                        "self-consistent samples; more directly constrains argmax flips than KL)")
    p.add_argument("--perturb_prob", type=float, default=0.3,
                   help="Probability of using a perturbed charge target (creates a charge-shift learning signal)")
    p.add_argument("--perturb_scale", type=float, default=4.0,
                   help="Maximum perturbed-charge offset (±Uniform[1,scale])")
    p.add_argument("--curriculum", action="store_true",
                   help="Curriculum learning: gradually increase perturb_scale from its starting value to "
                        "curriculum_scale_max across epochs—learn modest shifts before extreme extrapolation")
    p.add_argument("--curriculum_scale_max", type=float, default=8.0,
                   help="Maximum perturbation magnitude at the end of curriculum learning")
    p.add_argument("--placeholder_prob", type=float, default=0.15,
                   help="Fraction of placeholder samples (drawn from self-consistent samples): replace the condition charge with a "
                        "'do not control' placeholder so the model learns behavior when charge is unspecified (placeholder semantics for Objective 2). "
                        "Use each placeholder type equally: 1) has_charge=0 + value 0; 2) has_charge=1 + value=training mean. "
                        "These samples skip charge loss (no target)")
    p.add_argument("--charge_temp", type=float, default=0.5,
                   help="Softmax temperature for charge loss (<1 sharpens: the distribution optimized in training ≈ the inference sampling distribution, "
                        "reducing overshoot by ~2.57×; 1.0 = original expected-charge objective)")
    p.add_argument("--loss_reweight", type=int, default=0,
                   help="Inverse-density-weighted charge loss (addresses overshoot when extrapolating to highly positive targets): "
                        "weight charge loss inversely by target density, weight=k/(density_norm+eps); "
                        "rare targets (highly positive) receive larger weights. 1=on, 0=off")
    p.add_argument("--reweight_k", type=float, default=1.0, help="Inverse-weighting scale")
    p.add_argument("--reweight_eps", type=float, default=1e-3, help="Stabilizing term in the inverse-weighting denominator")
    p.add_argument("--reweight_cap", type=float, default=5.0,
                   help="Maximum weight (prevents excessive weights on highly positive samples from harming negative-charge hits; literature warns that naive weighting can overcorrect)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_domains", type=int, default=0,
                   help="Use at most the first N domains (0=all; for smoke tests)")

    # ---- condition-target, surface-charge, and structure components ----
    p.add_argument("--decouple_perturb", action="store_true",
                   help="Condition decoupling: perturbed target is independent of backbone native charge (Uniform[-range,range]), "
                        "breaking the strong coupling between backbone type and target charge (basic backbones see only positive targets and can only extrapolate)")
    p.add_argument("--decouple_range", type=float, default=12.0,
                   help="Range for independent target sampling (±Uniform[-range, range])")
    p.add_argument("--add_supervision", action="store_true",
                   help="Surface charge-addition supervision: L_add directly counters 'deletions without additions'—add D/E to the surface for a more negative target, "
                        "or K/R for a more positive target; modify only the surface (fracSASA≥θ), with target net charge as the upper bound")
    p.add_argument("--lambda_add", type=float, default=0.3,
                   help="L_add weight for surface charge-addition supervision")
    p.add_argument("--sasa_threshold", type=float, default=0.25,
                   help="B's surface eligibility threshold θ (counted in L_add only when fracSASA ≥ θ)")
    p.add_argument("--ph_aware_filter", action="store_true",
                   help="Stronger structure penalty: use the pH-adaptive filter bias during training to suppress charge clustering "
                        "(include His/Cys/Tyr according to protonation state); dynamically strengthen it for large-addition samples (scale_boost)")
    p.add_argument("--structure_boost", type=float, default=1.5,
                   help="C's structure-penalty multiplier for perturbed samples with large additions (scale_boost)")
    p.add_argument("--v12_supervision", action="store_true",
                   help="Training-side supervision (addresses the deletion shortcut): dual surface-composition counts (neither D/E nor K/R may decrease) "
                        "+ surface GRAVY (must not be more hydrophobic than native)—blocks the shortcut of replacing charged residues with hydrophobic ones")
    p.add_argument("--frac_floor", type=float, default=0.8,
                   help="Lower-bound fraction of native surface counts (0.8 allows modest but not large reductions)")
    p.add_argument("--gravy_margin", type=float, default=0.15,
                   help="Permitted increase in surface GRAVY (margin)")
    p.add_argument("--lambda_v12", type=float, default=0.3,
                   help="Combined composition + GRAVY weight (λ_v12·(comp+gravy))")
    p.add_argument("--lambda_target", type=float, default=0.0,
                   help="Surface-charge target weight: anchor surface net charge = target − core native charge "
                        "(λ_target·|q_surf − target_surf|; 0=off; addresses the missing upper bound that allowed unlimited additions)")
    # Three mutually exclusive regions + bidirectional pocket counts for ligand-mode protection
    p.add_argument("--pocket_mode", choices=["keep", "free", "global"], default="keep",
                   help="Ligand protection/counting mode: keep protects only the pocket (bidirectional counts only in the pocket); "
                        "global: count anchor covers all 'moderately editable' residues in surface∪pocket (charge_surf_mask, "
                        "bypasses frac_sasa blind spots and directly anchors total charged-residue count); "
                        "free=no training-side count protection (model's default hydrophobic prior)")
    p.add_argument("--pocket_cutoff", type=float, default=8.0,
                   help="Pocket range (minimum Cα-to-ligand distance < cutoff Å; consistent with define_pocket.py and validation criteria)")
    p.add_argument("--pocket_floor", type=float, default=0.7,
                   help="Lower-bound multiplier: total charged residues in the counted region ≥ native×floor")
    p.add_argument("--pocket_ceil", type=float, default=1.3,
                   help="Upper-bound multiplier: total charged residues in the counted region ≤ native×ceil (prevents paired additions)")
    p.add_argument("--lambda_pocket", type=float, default=0.2,
                   help="Count-loss weight (λ_pocket; use a dry run to balance against other losses)")
    p.add_argument("--decouple_absolute", action="store_true",
                   help="Absolute target sampling, independent of native and sampled directly over [lo,hi]")
    p.add_argument("--decouple_abs_lo", type=float, default=-35.0,
                   help="Lower bound for absolute target sampling (default -35)")
    p.add_argument("--decouple_abs_hi", type=float, default=20.0,
                   help="Upper bound for absolute targets (default 20)")
    p.add_argument("--add_target_scale", type=float, default=1.0,
                   help="Scale the L_add charge delta (default 1.0) to balance surface charge additions against the net-charge target")

    p.add_argument("--out_dir", default=str(_CODE_DIR / "output" / "finetune"))
    p.add_argument("--log_progress", default=str(_CODE_DIR / "log" / "train_progress.json"))
    p.add_argument("--log_file", default=str(_CODE_DIR / "log" / "train.log"))
    return p.parse_args()


def build_density_table(charge_arr, perturb_scale, lo=-40.0, hi=60.0, bw=0.5):
    """Estimate the training-time target-charge density (native charge at each pH + ± perturbation expansion).

    Used for inverse-density weighting (--loss_reweight): weight charge loss inversely by target density,
    so rare targets (highly positive, in the tail of the training distribution) receive larger weights. See inverse-density weighting for imbalanced regression.
    （US11720818B2 / arXiv 2506.01486）。

    Returns (density_norm, bucket): a normalized density table (0~1, with 1 for the densest bin) and a bin-index function.
    """
    n_buckets = int((hi - lo) / bw)
    counts = np.zeros(n_buckets, dtype=np.float64)

    def bucket(c):
        return int(np.clip((c - lo) / bw, 0, n_buckets - 1))

    for c in charge_arr:
        c0, c1 = bucket(c - perturb_scale), bucket(c + perturb_scale)
        counts[c0:c1 + 1] += 1.0   # Perturbations are symmetric and uniform; targets are approximately uniform within the interval
    counts = np.maximum(counts, 1.0)
    density = counts / counts.sum()
    density_norm = density / density.max()   # [0,1]; densest bin=1 → weight is around k in the middle
    return density_norm, bucket


def load_backbone(weights, device, ligand=False):
    """Load the backbone. MoMPNN uses the plain ProteinMPNN backbone; --ligand uses LigandMPNN weights.

    With --ligand, detect the weight type automatically (as in run_guided.py load_model):
    weights containing atom_context_num (>0) → ligand_mpnn (ligand context); otherwise protein_mpnn.
    """
    checkpoint = torch.load(weights, map_location=device)
    if ligand:
        model_type = (
            "ligand_mpnn" if checkpoint.get("atom_context_num", 0) > 0
            else "protein_mpnn"
        )
        atom_context_num = (
            0 if model_type == "protein_mpnn"
            else int(checkpoint.get("atom_context_num", 25))
        )
    else:
        model_type = "protein_mpnn"
        atom_context_num = 0
    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=int(checkpoint["num_edges"]),
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_domain(feature_dict, device, seed):
    """Arrange the outputs of parse_PDB + featurize as tensors required for training.

    Returns a dict (all tensors on device):
        X, S, mask, chain_mask, R_idx, chain_labels, randn
    + Pass through all featurize keys (in ligand mode, Y/Y_t/Y_m/mask_XY are used by backbone.encode).
    randn is fixed by seed (one decoding order per domain, shared across the 8 pH values in a batch).
    """
    L = feature_dict["X"].shape[1]
    fd = {k: v.to(device) if torch.is_tensor(v) else v for k, v in feature_dict.items()}
    rng = np.random.RandomState(seed)
    randn = torch.from_numpy(rng.randn(L).astype(np.float32)).to(device)
    dom = dict(fd)  # Pass through all keys, including ligand context
    dom.update({
        "X": fd["X"], "S": fd["S"], "mask": fd["mask"].float(),
        "chain_mask": fd["chain_mask"].float(),
        "R_idx": fd["R_idx"], "chain_labels": fd["chain_labels"],
        "randn": randn,
    })
    return dom


def decoder_forward(model, h_V, h_E, E_idx, dom, B, device):
    """Teacher-forced parallel decoding (standard ProteinMPNN training forward pass).

    Each position attends only to positions before it in decoding_order (mask_bw); h_S = W_s(S_true).
    Returns logits [B, L, 21].
    """
    S_true = dom["S"].long().repeat(B, 1)
    mask = dom["mask"].repeat(B, 1)
    chain_mask = dom["chain_mask"].repeat(B, 1)
    randn = dom["randn"].repeat(B, 1)
    L = S_true.shape[1]

    decoding_order = torch.argsort((chain_mask + 0.0001) * torch.abs(randn))  # [B,L]
    permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=L).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(L, L, device=device))),
        permutation_matrix_reverse,
        permutation_matrix_reverse,
    )  # [B, L, L]
    mask_attend = torch.gather(order_mask_backward, 2, E_idx.repeat(B, 1, 1)).unsqueeze(-1)
    mask_1D = mask.view(B, L, 1, 1)
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    h_S = model.W_s(S_true)  # [B, L, 128]
    h_ES = cat_neighbors_nodes(h_S, h_E.repeat(B, 1, 1, 1), E_idx.repeat(B, 1, 1))
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E.repeat(B, 1, 1, 1),
                                       E_idx.repeat(B, 1, 1))
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx.repeat(B, 1, 1))
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    for layer in model.decoder_layers:
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx.repeat(B, 1, 1))
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, mask)

    return model.W_out(h_V)  # [B, L, 21]


def kl_anchor_loss(logits, logits_ref, mask):
    """KL(p_ref ‖ p_cond): keep the conditioned output distribution from drifting too far from the unconditional backbone distribution.

    Compute KL at each position, then average over the mask. p_ref is frozen (constant); gradients flow only through the conditioned branch.
    """
    p_ref = F.softmax(logits_ref, dim=-1)
    logp_cond = F.log_softmax(logits, dim=-1)
    kl = (p_ref * (p_ref.log() - logp_cond)).sum(dim=-1)  # [B, L]
    denom = mask.float().sum().clamp(min=1.0)
    return (kl * mask.float()).sum() / denom


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = open(args.log_file, "a")
    def logln(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    logln("=== ConfuMPNN conditional fine-tuning started ===")
    logln(f"device={device}  epochs={args.epochs}  lr={args.lr}  "
          f"λ_c={args.lambda_c}  λ_kl={args.lambda_kl}  λ_keep={args.lambda_keep}  "
          f"perturb_prob={args.perturb_prob}  perturb_scale={args.perturb_scale}  "
          f"placeholder_prob={args.placeholder_prob}  "
          f"charge_temp={args.charge_temp}")
    if args.decouple_absolute:
        logln(f"[Absolute target sampling] enabled: target ∈ Uniform[{args.decouple_abs_lo}, "
              f"{args.decouple_abs_hi}], independent of native (covers the deeply negative validation range)")
    elif args.decouple_perturb:
        logln(f"[Condition decoupling] enabled: target is independent of native, Uniform[-{args.decouple_range},"
              f"{args.decouple_range}]")
    if args.add_supervision:
        logln(f"[Surface charge-addition supervision] enabled: λ_add={args.lambda_add}  SASA θ={args.sasa_threshold}")
    if args.v12_supervision:
        logln(f"[Training-side supervision] enabled: dual composition counts(floor={args.frac_floor}) + GRAVY(margin={args.gravy_margin})"
              f"  λ_v12={args.lambda_v12}  λ_target={args.lambda_target}  SASA θ={args.sasa_threshold}")
    if args.ph_aware_filter:
        logln(f"[pH-adaptive structure penalty] enabled: boost={args.structure_boost}")
    if args.pocket_mode == "global":
        logln(f"[A1 global] Count anchor covers surface∪pocket: floor={args.pocket_floor} "
              f"ceil={args.pocket_ceil} λ_pocket={args.lambda_pocket} cutoff={args.pocket_cutoff}")
    elif args.pocket_mode == "keep":
        logln(f"[A1 keep] Count anchor covers pocket only: floor={args.pocket_floor} "
              f"ceil={args.pocket_ceil} λ_pocket={args.lambda_pocket} cutoff={args.pocket_cutoff}")

    # ---- backbone + condition encoder ----
    backbone = load_backbone(args.weights, device, ligand=args.ligand)
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    n_backbone = sum(p.numel() for p in backbone.parameters())
    logln(f"backbone: MoMPNN frozen ({n_backbone/1e6:.2f}M parameters; not updated)")

    import yaml
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)["condition_defaults"]
    enc = ConditionEncoder(
        cond_dim=cfg["cond_dim"],
        hidden_dim=cfg["encoder"]["hidden_dim"],
        token_dim=cfg["encoder"]["token_dim"],
        n_tokens=cfg["encoder"]["n_tokens"],
        mean=cfg["normalization"]["mean"],
        std=cfg["normalization"]["std"],
    ).to(device)
    n_trainable = sum(p.numel() for p in enc.parameters())
    logln(f"ConditionEncoder is trainable ({n_trainable} parameters; the only component updated)")

    optimizer = torch.optim.Adam(enc.parameters(), lr=args.lr)

    # ---- Data: labels.npz + dompdb ----
    labels = np.load(args.labels, allow_pickle=True)
    domain_ids = labels["domain_ids"]            # [999]
    pH_arr = labels["pH"].astype(np.float32)     # pH values in domain-major order
    charge_arr = labels["charge"].astype(np.float32)  # Charge labels aligned with pH_arr
    n_dom = len(domain_ids)
    if args.max_domains > 0:
        n_dom = min(n_dom, args.max_domains)
    n_pH = pH_arr.size // len(domain_ids)       # Number of pH values per domain (=8)
    assert n_pH * len(domain_ids) == pH_arr.size
    logln(f"Data: {n_dom} domains × {n_pH} pH values = {n_dom*n_pH} samples ({args.labels})")

    # Inverse-density weighting: precompute the training target-density table (--loss_reweight)
    density_norm, density_bucket = None, None
    if args.loss_reweight:
        density_norm, density_bucket = build_density_table(charge_arr, args.perturb_scale)
        logln(f"Inverse-density weighting enabled: k={args.reweight_k} eps={args.reweight_eps} "
              f"cap={args.reweight_cap} (compensates for charge-distribution skew and overshoot on highly positive targets)")

    # Pre-parse each domain → cache feature tensors, encodings, and unconditional logits
    # (the backbone is frozen, so encode and unconditional output need to be computed only once per domain and reused for all epochs)
    # prody parsePDB selects the format by file suffix: without a .pdb suffix, the file is parsed as mmCIF.
    # CATH files have no extension → use a directory of .pdb-suffixed symlinks (under data/; not tracked by git).
    # Ligand mode: domain_ids contain actual filenames (e.g. 1JCG.pdb/1abc.cif),
    # so locate the actual file directly from --dompdb and skip the symlink (the real suffix lets prody parse it).
    dom_cache_dir = Path(args.dompdb).parent / (Path(args.dompdb).name + "_pdb")
    dom_cache_dir.mkdir(exist_ok=True)
    abs_dompdb = os.path.abspath(args.dompdb)

    domains = []
    n_ok, n_skip = 0, 0
    for i, did in enumerate(domain_ids[:n_dom]):
        # Priority: use the actual file when it has a suffix (ligand data .pdb/.cif; prody selects the format by suffix).
        # ⚠️ CATH domain IDs have no suffix (e.g. 3t97A00): direct.exists() may be True, but prody
        # treats a suffixless file as mmCIF and fails. Use the direct path only when a suffix is present; otherwise use a .pdb symlink.
        direct = Path(args.dompdb) / str(did)
        has_suffix = Path(str(did)).suffix in (".pdb", ".cif", ".ent", ".cif.gz", ".pdb.gz")
        # ⚠️ For suffixless CATH domains, use a .pdb symlink; create it inside the try block,
        #    so domains with missing target files (dangling links) are skipped as invalid instead of crashing.
        try:
            if direct.exists() and has_suffix:
                pdb_path = str(direct)
            else:
                link_path = dom_cache_dir / f"{did}.pdb"
                # lexists checks whether the symlink itself exists (avoids FileExistsError from os.symlink when
                # Path.exists()=False for a dangling link); the target is absolute and independent of cwd.
                if not os.path.lexists(link_path):
                    os.symlink(os.path.join(abs_dompdb, str(did)), link_path)
                pdb_path = str(link_path)
            protein_dict, *_ = parse_PDB(pdb_path, device="cpu", parse_all_atoms=False)
            L = protein_dict["X"].shape[0]
            # Single-chain CATH domain: design all residues → chain_mask = all ones
            protein_dict["chain_mask"] = torch.ones(L, dtype=torch.int32)
            feature_dict = featurize(
                protein_dict,
                use_atom_context=args.ligand,
                number_of_ligand_atoms=(25 if args.ligand else 0),
                model_type=("ligand_mpnn" if args.ligand else "protein_mpnn"),
            )
            dom = build_domain(feature_dict, device, seed=args.seed + i)
            # Encode once with the frozen backbone
            h_V, h_E, E_idx = backbone.encode(dom)
            dom["h_V"] = h_V
            dom["h_E"] = h_E
            dom["E_idx"] = E_idx
            # Unconditional logits (without prompt injection)—reference distribution for the KL anchor; once per domain
            with torch.no_grad():
                logits_uncond = decoder_forward(backbone, h_V, h_E, E_idx, dom, 1, device)
            dom["logits_uncond"] = logits_uncond
            # Unconditional argmax sequence (constant SeqKeep anchor): anchor X positions to 0 and exclude them with ce_mask.
            # ⚠️ dom["S"] has batch shape [1,L]; index [0] to get the single chain and avoid broadcasting to [1,L].
            anchor = logits_uncond[0].argmax(-1)                          # [L]
            anchor = torch.where(dom["S"][0] < 20, anchor, torch.zeros_like(anchor))  # [L]
            dom["seq_anchor"] = anchor
            # CE validity mask: exclude nonstandard residues (X with S==20)
            valid = (dom["S"] < 20).float()
            dom["ce_mask"] = dom["mask"] * dom["chain_mask"] * valid
            # Surface charge-addition supervision: precompute fractional SASA per domain (freesasa; the backbone is frozen,
            # so the structure is static and SASA needs to be computed only once per domain). Supplies the "surface-only addition" weights for L_add.
            # ⚠️ freesasa (Bio.PDB) and LigandMPNN parse_PDB may classify residues differently (freesasa
            # may include residues skipped by parse, such as abnormal terminal residues 136-138). **Correct alignment = intersection of residue numbers**:
            #   match sasa residues[] against dom R_idx[] and keep only residue numbers present in both;
            #   ignore residues present only in freesasa and set frac=0 for residues present only in parse, if any.
            #   → Avoid length mismatches; L_add covers all aligned residues (frac=0 at X/nonstandard positions).
            if args.add_supervision or args.v12_supervision:
                try:
                    from src.sasa import fractional_sasa
                    sasa_info = fractional_sasa(pdb_path,
                                                surface_threshold=args.sasa_threshold,
                                                align_to_full=False)  # Return standard amino-acid positions only
                    sasa_frac = sasa_info["frac_sasa"]     # [n_sasa]
                    sasa_resids = sasa_info["residues"]    # [n_sasa] residue numbers
                    # dom["R_idx"] has shape [1, L] (featurize adds a batch dimension); flatten to [L]
                    dom_resids = np.asarray(dom["R_idx"].cpu().numpy()).reshape(-1)
                    # Map the residue-number intersection: SASA residue number → index
                    sasa_map = {int(r): i for i, r in enumerate(sasa_resids)}
                    aligned = np.zeros(L, dtype=np.float64)
                    n_aligned = 0
                    for pos in range(L):
                        rid = int(dom_resids[pos])
                        if rid in sasa_map:
                            aligned[pos] = sasa_frac[sasa_map[rid]]
                            n_aligned += 1
                        # Not in sasa (e.g. parse-only/nonstandard) → keep 0 (buried/not included)
                    dom["frac_sasa"] = aligned
                    if n_aligned < L:
                        logln(f"  ℹ️ {did} residue-number alignment {n_aligned}/{L} ({L-n_aligned} nonstandard/parse-only residues have frac=0)")
                    # Use three mutually exclusive residue regions for ligand-mode charge supervision.
                    # pocket = minimum Cα-to-ligand distance < pocket_cutoff (define_pocket.py criteria, 2026-09-01);
                    # the regions do not overlap: pocket (distance <8Å, regardless of frac) / core (frac<0.25 and not pocket) /
                    #   surface (frac≥0.25 and not pocket).
                    # charge supervision mask = surface ∪ pocket → all generated pocket charges are supervised;
                    # pocket is excluded from core → avoids the contradictory double-counting/total-charge-drift bug (core point of §7.1).
                    # Both keep and global modes require the three mutually exclusive regions; global count loss covers charge_surf_mask.
                    if args.pocket_mode in ("keep", "global"):
                        Y = dom.get("Y")
                        # ⚠️ dom["X"] shape [1, L, 4, 3] (sample, residue, N/CA/C/O, xyz)
                        # Cα = X[0, :, 1] (the CA atom, index 1, for all residues), not X[0, 1]
                        CA = dom["X"][0, :, 1].cpu().numpy()       # [L,3] Cα
                        if Y is not None and Y.numel() > 0:
                            Yc = Y.reshape(-1, 3).cpu().numpy()
                            dmin = np.linalg.norm(
                                CA[:, None, :] - Yc[None, :, :], axis=-1).min(axis=1)  # [L]
                            pocket = (dmin < args.pocket_cutoff).astype(np.float32)
                        else:
                            pocket = np.zeros(L, dtype=np.float32)  # No ligand → no pocket residues
                        surf = aligned >= args.sasa_threshold       # [L] bool
                        core = ((~surf) & (pocket == 0)).astype(np.float32)
                        dom["pocket_mask"] = pocket
                        dom["core_mask"] = core
                        dom["charge_surf_mask"] = np.clip(
                            surf.astype(np.float32) + pocket, 0.0, 1.0)
                        logln(f"  ℹ️ {did} partition: pocket={int(pocket.sum())}/{L}"
                              f" core={int(core.sum())} surface={int(surf.sum())}")
                except Exception as e:
                    dom["frac_sasa"] = None
                    logln(f"  ⚠️ {did} SASA/pocket partition failed: {e} (skipping training-side supervision)")
            # Eight (pH, charge) conditions for this domain (indexed by accepted-domain count so skipped invalid domains do not shift alignment)
            idx0 = n_ok * n_pH
            dom["pH"] = pH_arr[idx0:idx0 + n_pH]
            dom["charge_label"] = charge_arr[idx0:idx0 + n_pH]
            dom["domain_id"] = str(did)   # Used to identify domains with NaNs in the training loop
            domains.append(dom)
            n_ok += 1
        except Exception as e:
            n_skip += 1
            logln(f"  ⚠️ Skipping invalid domain {did}: {e}")
        if (i + 1) % 200 == 0:
            logln(f"  Pre-parse + encode {n_ok}/{n_dom} (skipped {n_skip})")
    if n_skip:
        logln(f"⚠️ Skipped {n_skip} invalid domains in total (prody could not parse them); training on {n_ok} domains")

    # Gradient flow is confined to the encoder: all backbone parameters have requires_grad=False,
    # and only enc.parameters() are passed to the optimizer.
    total_cached = sum(d["h_E"].numel() for d in domains) * 4 / 1e9
    logln(f"Pre-parse complete; cached encode features ~{total_cached:.2f}GB")

    # ---- Training ----
    n_dom_eff = len(domains)  # Number of domains available for training (may be < n_dom after invalid domains are skipped)
    domain_idx = list(range(n_dom_eff))
    n_steps_total = args.epochs * n_dom_eff
    step = 0
    t_start = time.time()
    logln(f"Starting training: {args.epochs} epochs × {n_dom_eff} domains/epoch = {n_steps_total} steps")

    for epoch in range(1, args.epochs + 1):
        # Curriculum learning: linearly increase perturbation magnitude from perturb_scale to curriculum_scale_max over epochs
        if args.curriculum:
            progress = (epoch - 1) / max(args.epochs - 1, 1)   # 0→1
            scale_cur = args.perturb_scale + (
                args.curriculum_scale_max - args.perturb_scale) * progress
            if epoch == 1:
                logln(f"Curriculum learning enabled: perturb_scale {args.perturb_scale} → "
                      f"{args.curriculum_scale_max} ({args.epochs} epochs total)")
        else:
            scale_cur = args.perturb_scale
        random.shuffle(domain_idx)
        epoch_loss, epoch_ce, epoch_cd, epoch_kl, epoch_keep = [], [], [], [], []
        # NaN diagnostics: record each domain with NaNs in h_V/logits/loss, then skip it and continue
        nan_domains = []
        # Charge-loss breakdown: self=self-consistent/placeholder, mild=modest perturbation, extreme=large perturbation (|offset|≥5)
        grp_cd = {"self": [], "mild": [], "extreme": []}
        for di in domain_idx:
            dom = domains[di]
            B = n_pH
            # Condition vector [B, 7]
            pH_b = torch.from_numpy(dom["pH"]).to(device)              # [8]
            charge_b = torch.from_numpy(dom["charge_label"]).to(device)  # [8]
            # Mixed targets: 70% self-consistent (target=native) + 30% perturbed charge (creates a shift-learning signal)
            mask_p = torch.zeros(B, dtype=torch.bool, device=device)
            if args.perturb_prob > 0:
                mask_p = (torch.rand(B, device=device) < args.perturb_prob)
                # With --decouple_perturb, the perturbed target is **independent** of backbone native charge—
                # Sample an independent random target directly from Uniform[-decouple_range, decouple_range],
                # Break the coupling between backbone type and target charge so varied combinations enter the training distribution.
                # When disabled (default), sample native ± Uniform[1, scale] (magnitude controlled by the curriculum).
                if args.decouple_absolute:
                    # Absolute target ∈ Uniform[lo, hi], independent of backbone native charge.
                    # Still define offset = target − native (used by L_add and group monitoring),
                    # and share the subsequent charge_b = charge_b + offset flow.
                    native_b = charge_b.clone()
                    target_abs = (torch.rand(B, device=device)
                                  * (args.decouple_abs_hi - args.decouple_abs_lo)
                                  + args.decouple_abs_lo)
                    offset = torch.where(mask_p, target_abs - native_b,
                                         torch.zeros(B, device=device))
                elif args.decouple_perturb:
                    offset = torch.where(
                        mask_p,
                        (torch.rand(B, device=device) * 2 - 1) * args.decouple_range,
                        torch.zeros(B, device=device),
                    )
                else:
                    offset = torch.where(
                        mask_p,
                        torch.randint(1, int(scale_cur) + 1, (B,), device=device).float()
                        * torch.where(torch.rand(B, device=device) < 0.5, 1.0, -1.0),
                        torch.zeros(B, device=device),
                    )
                charge_b = charge_b + offset
            # Placeholder samples: randomly select self-consistent samples and replace the charge condition with the training mean.
            # Use has_charge=1 with the mean value and keep charge loss enabled at that target, so an unspecified condition
            # maps to a mild default rather than an unsupervised target.
            mask_ph = torch.zeros(B, dtype=torch.bool, device=device)
            if args.placeholder_prob > 0:
                mask_ph = (~mask_p) & (torch.rand(B, device=device) < args.placeholder_prob)
            charge_mean = float(cfg["normalization"]["mean"][2])  # Training mean for the charge dimension
            cond_b = torch.stack([
                make_condition_vector(p, c) if not mask_ph[i] else
                make_condition_vector(p, net_charge=charge_mean)
                for i, (p, c) in enumerate(zip(pH_b.tolist(), charge_b.tolist()))
            ]).to(device)  # [8, 7]

            # Condition injection + decoding
            prompt = enc(cond_b)                 # [8, 4, 128]
            h_V = dom["h_V"].repeat(B, 1, 1)
            h_V = inject_prompt(h_V, prompt)
            # NaN diagnostics: check h_V / logits separately, record all affected domains, and skip
            # (multiple domains may contain NaNs; continue after skipping this step and print the full list at epoch end)
            dname = dom.get("domain_id", "?")
            if not torch.isfinite(h_V).all():
                nan_domains.append({
                    "domain": dname, "step": "h_V",
                    "cond_b_nan": int(torch.isnan(cond_b).sum().item()),
                    "prompt_nan": int(torch.isnan(prompt).sum().item()),
                    "hV_before_nan": int(torch.isnan(dom["h_V"]).sum().item()),
                })
                continue  # Skip this step
            logits = decoder_forward(backbone, h_V, dom["h_E"], dom["E_idx"], dom, B, device)
            if not torch.isfinite(logits).all():
                nan_domains.append({
                    "domain": dname, "step": "logits",
                    "h_V_nan": int(torch.isnan(h_V).sum().item()),
                })
                continue  # Skip this step

            S_true = dom["S"].long().repeat(B, 1)
            ce_mask = dom["ce_mask"].repeat(B, 1)
            ce = cross_entropy_loss(logits, S_true, ce_mask)

            # Charge deviation is computed per sample because pH differs; temperature scaling optimizes charge under the sampling distribution rather than expected charge.
            # Placeholder samples also receive charge loss with target=training mean: the placeholder represents a mild default charge, not an unsupervised condition.
            cd = torch.zeros(B, device=device)
            for i in range(B):
                tgt_i = charge_mean if mask_ph[i] else charge_b[i]
                cd[i] = charge_deviation_loss(
                    logits[i:i+1], pH=pH_b[i], target_charge=tgt_i,
                    mask=ce_mask[i:i+1], temperature=args.charge_temp,
                )
                if args.loss_reweight:
                    w = args.reweight_k / (
                        density_norm[density_bucket(float(tgt_i))] + args.reweight_eps)
                    cd[i] *= min(w, args.reweight_cap)
                # Breakdown monitoring: self=self-consistent/placeholder, mild=modest perturbation, extreme=large perturbation
                if mask_ph[i] or not mask_p[i]:
                    grp_cd["self"].append(cd[i].item())
                elif abs(offset[i].item()) >= 5:
                    grp_cd["extreme"].append(cd[i].item())
                else:
                    grp_cd["mild"].append(cd[i].item())
            cd = cd.mean()

            # KL anchor (conditioned ‖ unconditioned)
            ref = dom["logits_uncond"].repeat(B, 1, 1)
            kl = kl_anchor_loss(logits, ref, ce_mask) if args.lambda_kl > 0 else torch.zeros((), device=device)

            # Sequence-preservation regularization applies only to self-consistent samples: unperturbed → target=native → conditioned output approaches unconditional argmax;
            # for perturbed samples, target≠native and a charge shift is expected, so it is unconstrained.
            keep = torch.zeros(B, device=device)
            if args.lambda_keep > 0:
                anchor = dom["seq_anchor"].unsqueeze(0)  # [1, L]
                for i in range(B):
                    if not mask_p[i].item():
                        keep[i] = sequence_keep_loss(
                            logits[i:i+1], anchor, ce_mask[i:i+1])
            keep = keep.mean()

            # ---- Surface charge-addition supervision (--add_supervision): counter deletions without additions ----
            # Apply only to perturbed samples (self-consistent target=native has no need for charge addition).
            # Required charge increment = offset[i] (direction of perturbation relative to native):
            #   more negative (offset<0) → add D/E at the surface; more positive (offset>0) → add K/R at the surface.
            add = torch.zeros(B, device=device)
            n_add = 0
            if args.add_supervision and dom.get("frac_sasa") is not None:
                for i in range(B):
                    if mask_ph[i].item() or not mask_p[i].item():
                        continue  # Apply only to perturbed samples
                    delta = float(offset[i].item()) * args.add_target_scale
                    if abs(delta) < 1.0:
                        continue  # Do not enable when the required shift is too small
                    add[i] = surface_add_charge_loss(
                        logits[i:i+1], dom["frac_sasa"],
                        target_surface_charge_delta=delta,
                        surface_threshold=args.sasa_threshold,
                    )
                    n_add += 1
            add = add.mean() if n_add else torch.zeros((), device=device)

            # ---- pH-aware structure penalty (--ph_aware_filter): dynamically suppress charge clustering ----
            # Use pH-adaptive filter bias (include His/Cys/Tyr according to protonation state);
            # strengthen the penalty for perturbed samples requiring large additions (scale_boost) to discourage charge clustering.
            struct_pen = torch.zeros((), device=device)
            if args.ph_aware_filter:
                from src.structure_aware_filter import StructureAwareFilter
                coords = dom["X"][0, :, 1].cpu().numpy()  # [L,3] Cα
                filt = StructureAwareFilter(coords)
                seq_int_cur = dom["S"][0].long().cpu().numpy()
                # Per-sample boost (increase only for perturbed samples) + per-sample pH
                sp_vec = torch.zeros(B, device=device)
                for i in range(B):
                    boost_i = args.structure_boost if mask_p[i].item() else 1.0
                    sp_i, _ = ph_aware_structure_penalty(
                        logits[i:i+1], filt, seq_int_cur,
                        pH=float(pH_b[i].item()),
                        mask=ce_mask[i:i+1], scale_boost=boost_i,
                    )
                    sp_vec[i] = sp_i
                struct_pen = sp_vec.mean()

            # ---- Training-side supervision (--v12_supervision): address the deletion shortcut ----
            # Dual composition counts (neither D/E nor K/R may decrease) + surface GRAVY (must not be more hydrophobic than native).
            # The backbone is fixed → surface mask is constant; native sequence is known → precompute native surface metrics.
            v12_comp = torch.zeros((), device=device)
            v12_gravy = torch.zeros((), device=device)
            if args.v12_supervision and dom.get("frac_sasa") is not None:
                frac = dom["frac_sasa"]                       # [L] numpy
                surf = frac >= args.sasa_threshold            # [L] bool
                nat_int = dom["S"][0].long().cpu().numpy()    # [L] native amino-acid indices
                # Count only surface residues that are standard amino acids (index<20); skip X/nonstandard residues (≥20)
                sel = nat_int[surf]
                sel = sel[sel < 20]
                native_grav = float(KD.cpu().numpy()[sel].mean()) if len(sel) > 0 else 0.0
                n_v12 = 0
                for i in range(B):
                    if mask_ph[i].item():
                        continue  # Skip placeholder samples (target=training mean, not a specific charge target)
                    v12_comp = v12_comp + surface_composition_loss(
                        logits[i:i+1], frac, nat_int, frac_floor=args.frac_floor,
                        surface_threshold=args.sasa_threshold)
                    v12_gravy = v12_gravy + surface_gravy_loss(
                        logits[i:i+1], frac, native_grav, margin=args.gravy_margin,
                        surface_threshold=args.sasa_threshold)
                    n_v12 += 1
                if n_v12 > 0:
                    v12_comp = v12_comp / n_v12
                    v12_gravy = v12_gravy / n_v12

            # Surface-charge target (--lambda_target>0): anchor surface net charge = target − core native charge.
            # Keep the core fixed (native non-surface residue charges are constant, with no gradient); the surface carries all charge changes
            # → set an upper bound for K/R additions while retaining room to adjust and avoid undershooting.
            v12_ct = torch.zeros((), device=device)
            if args.lambda_target > 0 and dom.get("frac_sasa") is not None:
                from src.differentiable_charge import net_charge_from_logits
                if args.pocket_mode in ("keep", "global") and dom.get("core_mask") is not None:
                    core_mask = dom["core_mask"]       # Core = non-surface and non-pocket (three mutually exclusive regions)
                    charge_mask = dom["charge_surf_mask"]  # Supervision region = surface ∪ pocket
                else:
                    core_mask = (~surf).astype(np.float32)   # [L] core (non-surface) residues
                    charge_mask = None                       # Default to surface
                nat_onehot = F.one_hot(torch.as_tensor(nat_int).clamp(0, 19).long(),
                                       num_classes=20).float().to(device)   # [L,20] native side-chain one-hot
                n_ct = 0
                for i in range(B):
                    if mask_ph[i].item():
                        continue  # As with composition/GRAVY losses, skip placeholder samples
                    q_core = net_charge_from_logits(
                        nat_onehot.unsqueeze(0), pH=pH_b[i],
                        mask=torch.as_tensor(core_mask, device=device).unsqueeze(0),
                        include_termini=False)                   # Core side-chain charge (terminal charges belong to the supervised region)
                    target_surf = charge_b[i].detach() - q_core  # target − core → the supervised region carries the remainder
                    v12_ct = v12_ct + surface_charge_target_loss(
                        logits[i:i+1], pH=pH_b[i], target_surface_charge=target_surf,
                        frac_sasa=frac, surface_threshold=args.sasa_threshold,
                        temperature=args.charge_temp, extra_mask=charge_mask)
                    n_ct += 1
                if n_ct > 0:
                    v12_ct = v12_ct / n_ct

            # Bidirectional charged-residue count bounds:
            # Set expected D/E and K/R counts in the counted region to [native×floor, native×ceil].
            #   keep mode: count only the pocket (Cα-to-ligand <8Å), protecting the pocket while allowing surface counts to change;
            #   global mode: count charge_surf_mask = surface ∪ pocket (all moderately editable residues)
            #     → include buried pocket residues that frac_sasa would not classify as surface and anchor the total charged-residue count.
            # Preserve total counts, not positions: do not fix individual sites; allow side-chain rearrangement.
            v12_pocket = torch.zeros((), device=device)
            if (args.pocket_mode in ("keep", "global") and args.lambda_pocket > 0
                    and dom.get("pocket_mask") is not None
                    and dom.get("frac_sasa") is not None):
                if args.pocket_mode == "global":
                    region = dom["charge_surf_mask"].astype(bool)   # surface ∪ pocket
                else:
                    region = dom["pocket_mask"].astype(bool)        # Pocket only
                nat_int_p = dom["S"][0].long().cpu().numpy()
                nat_neg = int(((nat_int_p == D_IDX) | (nat_int_p == E_IDX))[region].sum())
                nat_pos = int(((nat_int_p == K_IDX) | (nat_int_p == R_IDX))[region].sum())
                region_f = region.astype(np.float32)
                n_p = 0
                for i in range(B):
                    if mask_ph[i].item():
                        continue  # As with composition/GRAVY losses, skip placeholder samples
                    v12_pocket = v12_pocket + pocket_count_loss(
                        logits[i:i+1], region_f, (nat_neg, nat_pos),
                        floor=args.pocket_floor, ceil=args.pocket_ceil,
                        normalize=(args.pocket_mode == "global"))
                    n_p += 1
                if n_p > 0:
                    v12_pocket = v12_pocket / n_p

            total = ce + args.lambda_c * cd + args.lambda_kl * kl + args.lambda_keep * keep
            if args.add_supervision:
                total = total + args.lambda_add * add
            if args.ph_aware_filter:
                total = total + 0.05 * struct_pen
            if args.v12_supervision:
                total = total + args.lambda_v12 * (v12_comp + v12_gravy)
                if args.lambda_target > 0:
                    total = total + args.lambda_target * v12_ct
            if args.pocket_mode in ("keep", "global") and args.lambda_pocket > 0:
                total = total + args.lambda_pocket * v12_pocket

            # NaN diagnostics: record loss-component NaNs and skip this step
            if not torch.isfinite(total).all():
                dname = dom.get("domain_id", "?")
                def _s(t):
                    try: return f"{t.item():.4f}"
                    except Exception: return "?"
                def _nans(x):
                    try:
                        t = torch.as_tensor(x, dtype=torch.float32)
                        return int(torch.isnan(t).sum().item())
                    except Exception: return -1
                off0 = offset[0].item() if 'offset' in dir() else None
                ph0 = pH_b[0].item() if 'pH_b' in dir() else None
                nan_domains.append({
                    "domain": dname, "step": "loss",
                    "ce": _s(ce), "charge": _s(cd), "kl": _s(kl),
                    "keep": _s(keep), "add": _s(add), "struct": _s(struct_pen),
                    "pocket": _s(v12_pocket), "ct": _s(v12_ct),
                    "frac_nan": _nans(dom.get('frac_sasa')),
                    "offset0": off0, "pH0": ph0,
                })
                continue  # Skip this step (do not run backward)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            optimizer.step()

            epoch_loss.append(total.item()); epoch_ce.append(ce.item())
            epoch_cd.append(cd.item()); epoch_kl.append(kl.item())
            epoch_keep.append(keep.item())
            step += 1

        # ---- Epoch summary + progress file ----
        avg = lambda x: float(np.mean(x))
        grp_str = "  ".join(f"{k}={avg(v):.3f}" for k, v in grp_cd.items() if v)
        msg = (f"epoch {epoch}/{args.epochs}  total={avg(epoch_loss):.4f}  "
               f"ce={avg(epoch_ce):.4f}  charge={avg(epoch_cd):.4f}  "
               f"kl={avg(epoch_kl):.4f}  keep={avg(epoch_keep):.4f}")
        if grp_str:
            msg += f"  [cd {grp_str}]"
        msg += f"  elapsed={((time.time()-t_start)/60):.1f}min"
        logln(msg)
        # Print all domains that triggered NaNs
        if nan_domains:
            logln(f"🚨 Domains with NaNs this epoch: {len(nan_domains)} (skipped; weights not updated)")
            for nd in nan_domains[:20]:
                logln(f"    NaN {nd}")
            if len(nan_domains) > 20:
                logln(f"    ... {len(nan_domains)} total")
            # Write progress metadata to the progress file
            prog_nan = prog.copy()
            prog_nan["nan_domains"] = nan_domains
            with open(args.log_progress, "w") as f:
                json.dump(prog_nan, f, indent=2)
        prog = {
            "epoch": epoch, "total_epochs": args.epochs,
            "loss": avg(epoch_loss), "ce": avg(epoch_ce),
            "charge": avg(epoch_cd), "kl": avg(epoch_kl),
            "keep": avg(epoch_keep),
            "elapsed_min": round((time.time() - t_start) / 60, 1),
        }
        with open(args.log_progress, "w") as f:
            json.dump(prog, f, indent=2)

        # ---- checkpoint ----
        ckpt_path = out_dir / f"finetune_epoch{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "condition_encoder_state": enc.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cond_dim": cfg["cond_dim"],
            "n_tokens": cfg["encoder"]["n_tokens"],
            "token_dim": cfg["encoder"]["token_dim"],
            "mean": cfg["normalization"]["mean"],
            "std": cfg["normalization"]["std"],
            "backbone_weights": args.weights,
            "loss_terms": prog,
            # Provenance fields for perturbation settings and placeholder sampling
            "perturb_prob": args.perturb_prob,
            "perturb_scale": args.perturb_scale,
            "placeholder_prob": args.placeholder_prob,
            "lambda_keep": args.lambda_keep,
            "charge_temp": args.charge_temp,
            "curriculum": args.curriculum,
            "curriculum_scale_max": args.curriculum_scale_max,
            # Provenance fields for condition-target, surface-charge, and structure terms
            "decouple_perturb": args.decouple_perturb,
            "decouple_range": args.decouple_range,
            "add_supervision": args.add_supervision,
            "lambda_add": args.lambda_add,
            "sasa_threshold": args.sasa_threshold,
            "ph_aware_filter": args.ph_aware_filter,
            "structure_boost": args.structure_boost,
            # Ligand-mode region and charged-residue count settings
            "pocket_mode": args.pocket_mode,
            "pocket_cutoff": args.pocket_cutoff,
            "pocket_floor": args.pocket_floor,
            "pocket_ceil": args.pocket_ceil,
            "lambda_pocket": args.lambda_pocket,
        }, ckpt_path)
        # Keep the latest alias for inference-time loading
        torch.save(enc.state_dict(), out_dir / "condition_encoder_last.pt")

    logln(f"Training complete. Total time: {((time.time()-t_start)/60):.1f} min. Checkpoint: {out_dir}/")
    log.close()


if __name__ == "__main__":
    main()
