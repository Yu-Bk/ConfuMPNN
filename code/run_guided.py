"""One-command guided sampling entry point (Phase 1).

Wires together the full pipeline:
    PDB → backbone/ligand context → structure-aware filter + dynamic charge lookahead → guided sampling
    → generate N candidate sequences → calculate each sequence's net charge/pI → print summary statistics

Usage (from the code/ directory):
    conda activate confumpnn
    python run_guided.py --pdb input/1BC8.pdb --pH 7.4 [--target_charge -2.0]
                         [--preset default] [--num_samples 10]
    # Default generator = MoMPNN (multi-objective DPO fine-tuned)
    # Explicitly fall back to the original LigandMPNN (with ligand context):
    python run_guided.py --pdb input/1BC8.pdb --pH 7.4 \
        --weights ../LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt

Condition-injection mode (Phase 3, using a fine-tuned ConditionEncoder; the model learns pH awareness):
    python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
        --cond_encoder output/finetune/condition_encoder_last.pt \
        [--cond_mode conditioned|baseline] [--no_calibration]
    # conditioned: inject the condition vector (default), without logit bias — test the model's learned pH awareness
    # baseline: load the encoder without injection (equivalent to the honest Phase 1 boundary control)
    # Charge calibration (disabled by default): charge_calibration.enabled=false in configs/condition_defaults.yaml.
    #   Training uses charge_temp=0.5; inference-side calibration remains configurable.
    #   For legacy linear calibration, set enabled to true in YAML (target_eff=(desired-offset)/gain);
    #   --no_calibration forces calibration off.

    # pH-only auto-completion: when --target_charge is omitted, it is completed automatically by default
    #   target = the native sequence's net charge at the given pH ("preserve native charge behavior", within the training distribution),
    #   rather than using flag=0 (training always used flag=1, while inference flag=0 was unseen → unpredictable behavior).
    #   --no_auto_target_charge disables auto-completion and restores the old flag=0 semantics.

Recommended: redirect logs to code/log/; output is written to code/output/.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Path setup: code/ and LigandMPNN/
_CODE_DIR = Path(__file__).resolve().parent
_LIG_DIR = _CODE_DIR.parent / "LigandMPNN"
for p in [str(_CODE_DIR), str(_LIG_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Default generator = MoMPNN (multi-objective DPO fine-tuned)
# Use --weights to explicitly select the original LigandMPNN (with ligand context).
_DEFAULT_WEIGHTS = (
    _CODE_DIR.parent / "MoMPNN" / "mompnn_paper_checkpoints"
    / "mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
)

# Charge calibration table: stores a linear fit to the model's charge response.
# (generated charge ≈ slope·target + intercept). At inference, target_eff=(desired−intercept)/slope
# offsets the response gain learned by the encoder.
# Calibration is enabled automatically and supports per-protein entries with a global fallback.
# Proteins without a per-protein entry use the global fit when available.
_DEFAULT_CALIBRATION = _CODE_DIR.parent / "output" / "charge_calibration_v12_2.json"

import yaml  # noqa: E402

from data_utils import featurize, parse_PDB, restype_int_to_str  # noqa: E402
from model_utils import ProteinMPNN  # noqa: E402
from src.charge_lookahead import make_dynamic_callback  # noqa: E402
from src.condition_embedding import ConditionEncoder, make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.guided_sampler import GuidedSampler, extract_calpha_coords  # noqa: E402
from src.isoelectric_point import find_pI  # noqa: E402
from src.structure_aware_filter import StructureAwareFilter, load_preset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="ConfuMPNN Phase 1 guided sampling")
    p.add_argument("--pdb", required=True, help="Input PDB path")
    p.add_argument("--pH", type=float, required=True, help="Operating pH")
    p.add_argument("--target_charge", type=float, default=None,
                   help="Target net charge (None = no charge guidance; structure filtering only)")
    p.add_argument("--preset", default="default",
                   choices=["default", "nucleic_acid_binding", "membrane", "acidic"],
                   help="Structure-filter preset")
    p.add_argument("--num_samples", type=int, default=10, help="Number of candidate sequences to generate")
    p.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    p.add_argument("--strength", type=float, default=0.5, help="Charge-guidance strength")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--weights", default=None,
                   help="Weights path (default: MoMPNN mompnn_temberture_tm_esm_6_4_4_b01.ckpt; "
                        "use ligandmpnn_v_32_010_25.pt to fall back to the original LigandMPNN)")
    p.add_argument("--model_type", default="auto",
                   choices=["auto", "protein_mpnn", "ligand_mpnn"],
                   help="Model type: auto=detect from weights (default); protein_mpnn=backbone-only (e.g., MoMPNN); "
                        "ligand_mpnn=ligand-context model (original LigandMPNN)")
    p.add_argument("--cond_encoder", default=None,
                   help="Path to fine-tuned ConditionEncoder weights (Phase 3 condition-injection mode). "
                        "Supports condition_encoder_last.pt (state dict) or finetune_epochNNN.pt (includes config)")
    p.add_argument("--cond_mode", default="conditioned",
                   choices=["conditioned", "baseline"],
                   help="Condition-injection mode: conditioned=inject the condition vector (default; test model pH awareness); "
                        "baseline=load encoder without injection (equivalent to the honest Phase 1 boundary control)")
    p.add_argument("--no_calibration", action="store_true",
                   help="Disable charge calibration (enabled by default: apply linear calibration using gain/offset from condition_defaults.yaml, "
                        "target_eff=(desired-offset)/gain, to compensate for the configured response gain)")
    p.add_argument("--calibrate", default="auto", choices=["auto", "global", "off"],
                   help="Charge-calibration mode (takes precedence over --no_calibration): auto=read the calibration table, "
                        "match per-protein and fall back globally if unmatched; global=force global calibration; off=disable calibration. "
                        "Default table: output/charge_calibration.json")
    p.add_argument("--calibration_file", default=str(_DEFAULT_CALIBRATION),
                   help="Path to calibration-table JSON (default: charge_calibration_v12_2.json, enabled automatically)")
    p.add_argument("--no_auto_target_charge", action="store_true",
                   help="Disable pH-only auto-completion. By default, if --target_charge is omitted, auto-complete "
                        "target=native_charge@pH (preserve native charge behavior); this switch restores the old flag=0 semantics")
    p.add_argument("--fixed_residues", default=None,
                   help="Fixed-residue list, space-separated (chain letter + residue number, e.g., 'A12 C15'). "
                        "Amino acids at these positions remain unchanged; the model designs all other positions. "
                        "Reuses the native LigandMPNN mechanism (positions with chain_mask=0 are forced to retain their original amino acids during decoding).")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: code/output/guided_<pdb>_pH<pH>)")
    return p.parse_args()


def load_calibration(path, pdb_stem, force_global=False):
    """Load the charge calibration table; return (slope, intercept, mode, label).

    - Prefer per-protein: use that protein's (slope, intercept) from the table (auto mode);
    - Fall back to global: use the linear fit across proteins (forced by --calibrate global);
    - If the table is missing or unreadable: return (None, None, None, None) (the caller falls back to legacy YAML calibration or no calibration).
    """
    import json as _json

    path = Path(path)
    if not path.is_file():
        return None, None, None, None
    try:
        cal = _json.load(open(path))
    except Exception:
        return None, None, None, None
    per = cal.get("per_protein", {})
    g = cal.get("global", {})
    slope = off = None
    mode = None
    if not force_global and pdb_stem in per and per[pdb_stem] is not None:
        pp = per[pdb_stem]
        # Check the per-protein calibration entry before using its slope and intercept.
        if not pp.get("unreliable", False):
            slope, off = pp.get("slope"), pp.get("intercept")
            mode = f"per-protein({pdb_stem})"
        else:
            # Use the global fit when the per-protein entry is flagged; --calibrate global also forces global selection.
            if g:
                slope, off = g.get("slope"), g.get("intercept")
                mode = f"global(fallback: {pdb_stem})"
    elif g:
        slope, off = g.get("slope"), g.get("intercept")
        mode = "global"
    if slope is None or off is None:
        return None, None, None, None
    return float(slope), float(off), mode, cal.get("label", "?")


def load_model(weights, device, model_type="auto"):
    checkpoint = torch.load(weights, map_location=device)
    # Auto-detection: atom_context_num (>0) in the weights indicates LigandMPNN ligand weights;
    # otherwise, they are backbone-only ProteinMPNN weights (e.g., MoMPNN).
    if model_type == "auto":
        model_type = (
            "ligand_mpnn" if checkpoint.get("atom_context_num", 0) > 0
            else "protein_mpnn"
        )
    atom_context_num = (
        0 if model_type == "protein_mpnn" else int(checkpoint.get("atom_context_num", 25))
    )
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


def load_condition_encoder(path, device):
    """Load the fine-tuned ConditionEncoder.

    Supports two checkpoint formats:
        condition_encoder_last.pt : plain state dict (8 keys, including mean/std buffers)
        finetune_epochNNN.pt      : dict (includes condition_encoder_state and config)
    Architecture parameters (hidden_dim/n_tokens/token_dim/μ/σ) are read from the checkpoint when available;
    otherwise, configs/condition_defaults.yaml is used.
    """
    with open(_CODE_DIR / "configs" / "condition_defaults.yaml") as f:
        cfg = yaml.safe_load(f)["condition_defaults"]

    ck = torch.load(path, map_location=device)
    if "condition_encoder_state" in ck:
        state = ck["condition_encoder_state"]
        mean = ck.get("mean", cfg["normalization"]["mean"])
        std = ck.get("std", cfg["normalization"]["std"])
        n_tokens = ck.get("n_tokens", cfg["encoder"]["n_tokens"])
        token_dim = ck.get("token_dim", cfg["encoder"]["token_dim"])
        epoch = ck.get("epoch", None)
    else:
        state = ck
        mean = cfg["normalization"]["mean"]
        std = cfg["normalization"]["std"]
        n_tokens = cfg["encoder"]["n_tokens"]
        token_dim = cfg["encoder"]["token_dim"]
        epoch = "last"

    enc = ConditionEncoder(
        cond_dim=cfg["cond_dim"],
        hidden_dim=cfg["encoder"]["hidden_dim"],
        token_dim=token_dim,
        n_tokens=n_tokens,
        mean=mean,
        std=std,
    )
    enc.load_state_dict(state)
    enc.to(device)
    enc.eval()
    print(f"    Condition encoder loaded: {Path(path).name}  (epoch={epoch}, n_tokens={n_tokens}, "
          f"token_dim={token_dim})")
    return enc


def seq_to_string(S):
    return "".join(restype_int_to_str[i] for i in S)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load model (default generator = MoMPNN; --weights can override it)
    weights = Path(args.weights) if args.weights else _DEFAULT_WEIGHTS
    print(f"[1] Loading model: {weights.name}  (device={device})")
    model = load_model(weights, device, model_type=args.model_type)
    mt = model.model_type  # Resolved model type
    print(f"    Resolved model_type = {mt}")

    # 2. Read PDB + featurize (use ligand context depending on model type)
    print(f"[2] Reading PDB: {args.pdb}")
    protein_dict, _, _, icodes, _ = parse_PDB(args.pdb)
    protein_dict["chain_mask"] = torch.ones(
        protein_dict["X"].shape[0], dtype=torch.int32  # Design all residues by default
    )
    # Fixed positions (native LigandMPNN mechanism: positions with chain_mask=0 are forced to retain their original amino acids during decoding;
    # see S_t = S_t·chain_mask_t + S_true·(1-chain_mask_t) in guided_sampler.py)
    fixed_positions = []
    if args.fixed_residues:
        R_idx = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters = list(protein_dict["chain_letters"])
        encoded = [
            str(chain_letters[i]) + str(R_idx[i]) + icodes[i]
            for i in range(len(R_idx))
        ]
        fixed_set = set(args.fixed_residues.split())
        for i, name in enumerate(encoded):
            if name in fixed_set:
                protein_dict["chain_mask"][i] = 0
                fixed_positions.append(name)
        print(f"    Fixed residues ({len(fixed_positions)}): {fixed_positions}")
    use_atom_context = (mt == "ligand_mpnn")
    feature_dict = featurize(
        protein_dict, cutoff_for_score=8.0,
        use_atom_context=use_atom_context,
        number_of_ligand_atoms=(25 if use_atom_context else 0),
        model_type=mt,
    )
    L = feature_dict["X"].shape[1]
    feature_dict["batch_size"] = 1
    feature_dict["temperature"] = args.temperature
    feature_dict["bias"] = torch.zeros(1, L, 21)
    native_seq = seq_to_string(feature_dict["S"][0].cpu().numpy())
    print(f"    Protein length {L}, native: {native_seq[:50]}...")
    native_charge = net_charge(native_seq, args.pH)  # Computed early for auto-completion and reuse in step 5

    # 2.5 pH-only auto-completion: if target is not explicitly provided, default to target=native_charge@pH,
    #     meaning "design a sequence that preserves native charge behavior at this pH," fully within the training distribution;
    #     --no_auto_target_charge disables this and restores the old flag=0 (unspecified charge) control.
    auto_target = False
    if args.target_charge is None and not args.no_auto_target_charge:
        args.target_charge = native_charge
        auto_target = True
        print(f"    [pH-only auto-completion] target = native_charge@{args.pH} = {native_charge:+.2f}")
    elif args.target_charge is None:
        print("    [pH-only manually disabled] --no_auto_target_charge: condition flag=0; no target charge specified")

    # 3. Mode branch: condition injection (Phase 3) vs guided sampling (Phase 1)
    mode = "phase1_guided"
    cond_encoder = None
    if args.cond_encoder:
        mode = "phase3_conditioned" if args.cond_mode == "conditioned" else "phase3_baseline"
        cond_encoder = load_condition_encoder(args.cond_encoder, device)

        # Charge calibration: target_eff = (desired − intercept) / slope
        # offsets the response gain learned by ConditionEncoder.
        # Prefer the calibration table (per-protein with global fallback);
        # if unavailable, fall back to legacy YAML gain/offset; do not calibrate auto targets (native charge is already within the training distribution).
        target_eff = args.target_charge
        calib_note = "(No target specified; no calibration)"
        if auto_target:
            calib_note = f"(Auto-completed target={native_charge:+.2f}; calibration skipped)"
        elif args.target_charge is not None and not args.no_calibration and args.calibrate != "off":
            pname = Path(args.pdb).stem
            slope, offset, mode, label = load_calibration(
                args.calibration_file, pname, force_global=(args.calibrate == "global"))
            if slope is not None:
                target_eff = (args.target_charge - offset) / slope
                calib_note = (f"target {args.target_charge} → calibrated target_eff "
                              f"{target_eff:.2f} ({label} {mode}: slope={slope:.3f} off={offset:.3f})")
            else:
                # Calibration table unavailable → fall back to legacy YAML linear calibration
                with open(_CODE_DIR / "configs" / "condition_defaults.yaml") as f:
                    cc = yaml.safe_load(f)["condition_defaults"].get("charge_calibration", {})
                gain, offset = cc.get("gain", 2.57), cc.get("offset", 0.16)
                if cc.get("enabled", True):
                    target_eff = (args.target_charge - offset) / gain
                    calib_note = (f"target {args.target_charge} → calibrated target_eff "
                                  f"{target_eff:.2f} (yaml gain={gain}, offset={offset})")
                else:
                    calib_note = f"(Calibration table {args.calibration_file} unavailable and yaml enabled=false; not calibrated)"
        elif args.target_charge is not None:
            calib_note = "(--no_calibration / --calibrate off; not calibrated)"

        cond_vec = make_condition_vector(args.pH, net_charge=target_eff).to(device)
        print(f"[3] Condition-injection mode: cond_mode={args.cond_mode}, "
              f"cond_vec={[round(x, 2) for x in cond_vec.tolist()]}")
        print(f"    Charge calibration: {calib_note}")
        print(f"[4] Sampling {args.num_samples} candidate sequences with condition injection...")
        sequences, charges, pIs = [], [], []
        for i in range(args.num_samples):
            feature_dict["randn"] = torch.randn(1, L)
            # baseline mode: load the encoder without injection → equivalent to the Phase 1 control (no guidance and no pH awareness)
            enc_inject = None if args.cond_mode == "baseline" else cond_encoder
            out = conditioned_sample(
                model, enc_inject, feature_dict, cond_vec, device=device,
            )
            seq = seq_to_string(out["S"][0].cpu().numpy())
            sequences.append(seq)
            charges.append(net_charge(seq, args.pH))
            pIs.append(find_pI(seq))
            print(f"    [{i+1:2d}] charge={charges[-1]:+6.2f}  pI={pIs[-1]:5.2f}  {seq[:60]}")
    else:
        # Phase 1 guided sampling (structure filter + dynamic charge-lookahead logit bias)
        print(f"[3] Guided settings: pH={args.pH}, target_charge={args.target_charge}, "
              f"preset={args.preset}, strength={args.strength}")
        coords = extract_calpha_coords(protein_dict)
        structure_filter = StructureAwareFilter(coords, config=load_preset(args.preset))
        bias_callback = make_dynamic_callback(
            pH=args.pH, target_charge=args.target_charge,
            structure_filter=structure_filter, strength=args.strength,
        )
        print(f"[4] Sampling {args.num_samples} candidate sequences with guidance...")
        sampler = GuidedSampler(model, device=device)
        sequences, charges, pIs = [], [], []
        for i in range(args.num_samples):
            feature_dict["randn"] = torch.randn(1, L)
            out = sampler.sample(feature_dict, bias_callback=bias_callback)
            seq = seq_to_string(out["S"][0].cpu().numpy())
            sequences.append(seq)
            charges.append(net_charge(seq, args.pH))
            pIs.append(find_pI(seq))
            print(f"    [{i+1:2d}] charge={charges[-1]:+6.2f}  pI={pIs[-1]:5.2f}  {seq[:60]}")

    # 5. Native control (native_charge was computed in step 2.5 for reuse in auto-completion)
    native_pI = find_pI(native_seq)
    print(f"[5] native   : charge={native_charge:+6.2f}  pI={native_pI:5.2f}  {native_seq[:60]}")

    # 6. Summary and output
    mean_charge = float(np.mean(charges))
    std_charge = float(np.std(charges))
    print(f"    Mean net charge = {mean_charge:+.2f} ± {std_charge:.2f}  "
          f"(target {args.target_charge})")

    out_dir = Path(args.out_dir) if args.out_dir else (
        _CODE_DIR / "output" / f"guided_{Path(args.pdb).stem}_pH{args.pH}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "seqs.fa"
    with open(fasta_path, "w", encoding="utf-8") as f:
        for i, seq in enumerate(sequences):
            f.write(f">sample_{i+1} pH={args.pH} charge={charges[i]:+.2f} pI={pIs[i]:.2f}\n")
            f.write(seq + "\n")
        f.write(f">native charge={native_charge:+.2f} pI={native_pI:.2f}\n")
        f.write(native_seq + "\n")
    summary = {
        "pdb": args.pdb, "pH": args.pH, "target_charge": args.target_charge,
        "auto_target": auto_target,  # True = target was auto-completed from pH only
        "mode": mode,
        "cond_encoder": str(args.cond_encoder) if args.cond_encoder else None,
        "calibrated": bool(args.cond_encoder and not args.no_calibration),
        "preset": args.preset, "temperature": args.temperature,
        "strength": args.strength, "seed": args.seed, "num_samples": args.num_samples,
        "native_charge": native_charge, "native_pI": native_pI,
        "mean_charge": mean_charge, "std_charge": std_charge,
        "sequences": [
            {"seq": s, "charge": c, "pI": p} for s, c, p in zip(sequences, charges, pIs)
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[6] Output saved: {fasta_path}")
    print("Done ✅")


if __name__ == "__main__":
    main()
