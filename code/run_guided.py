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
    # Charge calibration: no default table is bundled; YAML fallback is disabled in configs/condition_defaults.yaml.
    #   Training uses charge_temp=0.5; inference-side calibration remains configurable.
    #   For legacy linear calibration, set enabled to true in YAML (target_eff=(desired-offset)/gain);
    #   --no_calibration or --calibrate off disables calibration.

Hybrid mode (ConditionEncoder + ChargeLookahead + StructureAwareFilter):
    python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
        --cond_encoder output/finetune/condition_encoder_last.pt --hybrid
    # Hybrid requires --cond_mode conditioned; the baseline mode never injects the encoder condition.
    # ChargeLookahead uses the requested target, while ConditionEncoder receives target_eff after calibration.

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
# When a table is provided, auto prefers a per-protein fit and falls back to the global fit.
# The optional default path may be absent; in that case calibration is skipped unless YAML fallback is enabled.
_DEFAULT_CALIBRATION = _CODE_DIR.parent / "output" / "charge_calibration_v12_2.json"
_MIN_CALIBRATION_SLOPE = 1e-8

import yaml  # noqa: E402

from data_utils import featurize, parse_PDB, restype_int_to_str  # noqa: E402
from model_utils import ProteinMPNN  # noqa: E402
from src.charge_lookahead import make_dynamic_callback  # noqa: E402
from src.condition_embedding import ConditionEncoder, make_condition_vector  # noqa: E402
from src.conditioned_sampler import conditioned_sample  # noqa: E402
from src.differentiable_charge import net_charge  # noqa: E402
from src.guided_sampler import GuidedSampler, extract_calpha_coords  # noqa: E402
from src.isoelectric_point import find_pI  # noqa: E402
from src.structure_aware_filter import (  # noqa: E402
    FILTER_RULE_VERSION, StructureAwareFilter, load_preset,
)


def parse_args(argv=None):
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
    p.add_argument("--hybrid", action="store_true",
                   help="Combine conditioned ConditionEncoder sampling with ChargeLookahead and the structure filter; "
                        "requires --cond_encoder and --cond_mode conditioned")
    p.add_argument("--no_calibration", action="store_true",
                   help="Disable charge calibration; this and --calibrate off both disable table and YAML calibration")
    p.add_argument("--calibrate", default="auto", choices=["auto", "global", "off"],
                   help="Calibration lookup: auto=per-protein then global table fit; global=global table fit only; "
                        "off=disable calibration. --no_calibration also disables calibration. "
                        "If no table fit is available, the YAML fallback applies only when enabled")
    p.add_argument("--calibration_file", default=None,
                   help="Path to a calibration-table JSON; if omitted, use the optional default at "
                        "output/charge_calibration_v12_2.json when present")
    p.add_argument("--no_auto_target_charge", action="store_true",
                   help="Disable pH-only auto-completion. By default, if --target_charge is omitted, auto-complete "
                        "target=native_charge@pH (preserve native charge behavior); this switch restores the old flag=0 semantics")
    p.add_argument("--fixed_residues", default=None,
                   help="Fixed-residue list, space-separated (chain letter + residue number, e.g., 'A12 C15'). "
                        "Amino acids at these positions remain unchanged; the model designs all other positions. "
                        "Reuses the native LigandMPNN mechanism (positions with chain_mask=0 are forced to retain their original amino acids during decoding).")
    p.add_argument("--out_dir", default=None,
                   help="Output directory (default: code/output/guided_<pdb>_pH<pH>)")
    args = p.parse_args(argv)
    if args.hybrid and not args.cond_encoder:
        p.error("--hybrid requires --cond_encoder")
    if args.hybrid and args.cond_mode != "conditioned":
        p.error("--hybrid requires --cond_mode conditioned; the baseline mode does not inject conditions")
    return args


def load_calibration(path, pdb_stem, force_global=False):
    """Load the charge calibration table; return (slope, intercept, mode, label).

    - Prefer per-protein: use that protein's (slope, intercept) from the table (auto mode);
    - Fall back to global: use the linear fit across proteins (forced by --calibrate global);
    - If the table is missing or contains no applicable fit: return (None, None, None, None).
      Malformed tables raise ValueError rather than silently disabling calibration.
    """
    import json as _json

    path = Path(path)
    if not path.is_file():
        return None, None, None, None
    try:
        with path.open(encoding="utf-8") as f:
            cal = _json.load(f)
    except (OSError, _json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read calibration table {path}: {exc}") from exc
    if not isinstance(cal, dict):
        raise ValueError(f"Calibration table root must be a JSON object: {path}")
    if not any(key in cal for key in ("global", "per_protein")):
        raise ValueError(f"Calibration table must contain 'global' or 'per_protein': {path}")
    per = cal.get("per_protein", {})
    g = cal.get("global", {})
    if not isinstance(per, dict) or not isinstance(g, dict):
        raise ValueError("Calibration table 'per_protein' and 'global' entries must be objects")
    global_fit = _validated_fit(g, "global", path) if g else None
    valid_per_protein = {}
    for protein, fit in per.items():
        if fit is None:
            continue
        if not isinstance(fit, dict):
            raise ValueError(f"Per-protein calibration entry must be an object: {protein}")
        if not fit.get("unreliable", False):
            valid_per_protein[protein] = _validated_fit(fit, f"per-protein {protein}", path)
    if global_fit is None and not valid_per_protein:
        raise ValueError(f"Calibration table contains no usable global or per-protein fit: {path}")

    selected = None
    mode = None
    if not force_global and pdb_stem in per and per[pdb_stem] is not None:
        pp = per[pdb_stem]
        # Check the per-protein calibration entry before using its slope and intercept.
        if not pp.get("unreliable", False):
            selected = valid_per_protein[pdb_stem]
            mode = f"per-protein({pdb_stem})"
        else:
            # Use the global fit when the per-protein entry is flagged; --calibrate global also forces global selection.
            if global_fit is not None:
                selected = global_fit
                mode = f"global(fallback: {pdb_stem})"
    elif global_fit is not None:
        selected = global_fit
        mode = "global"
    if selected is None:
        return None, None, None, None
    slope, off = selected
    return slope, off, mode, cal.get("label", "?")


def _validated_fit(entry, name, path):
    if not isinstance(entry, dict) or not {"slope", "intercept"}.issubset(entry):
        raise ValueError(f"{name} calibration entry requires slope and intercept: {path}")
    try:
        slope, intercept = float(entry["slope"]), float(entry["intercept"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Calibration coefficients must be numeric in {path}") from exc
    if (not np.isfinite(slope) or not np.isfinite(intercept)
            or slope <= _MIN_CALIBRATION_SLOPE):
        raise ValueError(f"Calibration slope must be finite and positive and intercept finite in {path}")
    return slope, intercept


def resolve_charge_calibration(
    target_charge,
    pdb_stem,
    *,
    condition_injected,
    auto_target,
    no_calibration,
    calibration_mode,
    calibration_file,
    calibration_file_explicit,
    yaml_settings,
):
    """Resolve the effective target and report which calibration was applied."""
    table_path = Path(calibration_file) if calibration_file is not None else None
    result = {
        "target_eff": target_charge if condition_injected else None,
        "requested": False,
        "applied": False,
        "source": None,
        "fit_mode": None,
        "label": None,
        "slope": None,
        "intercept": None,
        "table_path": str(table_path) if table_path is not None else None,
        "status": "not_applicable",
    }
    if not condition_injected:
        result["status"] = "condition_not_injected"
        return result
    if calibration_file_explicit and (table_path is None or not table_path.is_file()):
        raise FileNotFoundError(f"Explicit calibration table does not exist: {calibration_file}")
    if auto_target:
        result["status"] = "auto_target_skipped"
        return result
    if target_charge is None:
        result["status"] = "target_unspecified"
        return result
    if no_calibration or calibration_mode == "off":
        result["status"] = "disabled"
        return result

    result["requested"] = True
    if table_path is not None and table_path.is_file():
        slope, intercept, fit_mode, label = load_calibration(
            table_path, pdb_stem, force_global=(calibration_mode == "global"))
        if slope is not None:
            result.update({
                "target_eff": (target_charge - intercept) / slope,
                "applied": True,
                "source": "table",
                "fit_mode": fit_mode,
                "label": label,
                "slope": slope,
                "intercept": intercept,
                "status": "table_applied",
            })
            return result
        result["status"] = "table_fit_unavailable"
    else:
        result["status"] = "default_table_missing"

    if yaml_settings.get("enabled", False):
        try:
            gain = float(yaml_settings["gain"])
            offset = float(yaml_settings["offset"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Enabled YAML calibration requires numeric gain and offset") from exc
        if not np.isfinite(gain) or not np.isfinite(offset) or gain <= _MIN_CALIBRATION_SLOPE:
            raise ValueError("Enabled YAML calibration requires finite values and a positive, nonzero gain")
        result.update({
            "target_eff": (target_charge - offset) / gain,
            "applied": True,
            "source": "yaml",
            "fit_mode": "yaml_linear",
            "slope": gain,
            "intercept": offset,
            "status": "yaml_applied",
        })
    else:
        result["status"] = f"{result['status']}_yaml_disabled"
    return result


def build_mode_metadata(sampling_mode, calibration, calibration_mode,
                        filter_applied):
    """Return stable sampling and actual-calibration fields for summary.json."""
    return {
        "mode": sampling_mode,
        "sampling_mode": sampling_mode,
        "calibrated": bool(calibration["applied"]),
        "calibration_requested": bool(calibration["requested"]),
        "calibration_applied": bool(calibration["applied"]),
        "calibration_mode": calibration_mode,
        "calibration_source": calibration["source"],
        "calibration_fit_mode": calibration["fit_mode"],
        "calibration_slope": calibration["slope"],
        "calibration_intercept": calibration["intercept"],
        "calibration_label": calibration["label"],
        "calibration_table": calibration["table_path"],
        "calibration_table_used": (calibration["table_path"]
                                   if calibration["source"] == "table" else None),
        "calibration_status": calibration["status"],
        "target_eff": calibration["target_eff"],
        "filter_applied": bool(filter_applied),
        "filter_rule_version": FILTER_RULE_VERSION if filter_applied else None,
    }


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

    # 3. Mode branch: condition injection, optional hybrid, or Phase 1 guided sampling.
    sampling_mode = "phase1_guided"
    cond_encoder = None
    filter_callback = None
    filter_applied = False
    calibration_path = (
        Path(args.calibration_file).expanduser()
        if args.calibration_file else _DEFAULT_CALIBRATION
    )
    calibration_info = {
        "target_eff": None,
        "requested": False,
        "applied": False,
        "source": None,
        "fit_mode": None,
        "label": None,
        "slope": None,
        "intercept": None,
        "table_path": None,
        "status": "not_applicable",
    }
    if args.cond_encoder:
        sampling_mode = "phase3_conditioned" if args.cond_mode == "conditioned" else "phase3_baseline"
        if args.hybrid:
            sampling_mode = "phase3_hybrid"
        cond_encoder = load_condition_encoder(args.cond_encoder, device)

        with open(_CODE_DIR / "configs" / "condition_defaults.yaml", encoding="utf-8") as f:
            condition_cfg = yaml.safe_load(f)["condition_defaults"]
        calibration_info = resolve_charge_calibration(
            args.target_charge,
            Path(args.pdb).stem,
            condition_injected=(args.cond_mode == "conditioned"),
            auto_target=auto_target,
            no_calibration=args.no_calibration,
            calibration_mode=args.calibrate,
            calibration_file=calibration_path,
            calibration_file_explicit=(args.calibration_file is not None),
            yaml_settings=condition_cfg.get("charge_calibration", {}),
        )
        target_eff = calibration_info["target_eff"]
        if calibration_info["applied"]:
            calib_note = (f"target {args.target_charge} → target_eff {target_eff:.2f} "
                          f"({calibration_info['source']} {calibration_info['fit_mode']})")
        else:
            calib_note = f"calibration not applied ({calibration_info['status']})"

        if args.hybrid:
            coords = extract_calpha_coords(protein_dict)
            structure_filter = StructureAwareFilter(coords, config=load_preset(args.preset))
            filter_callback = make_dynamic_callback(
                pH=args.pH, target_charge=args.target_charge,
                structure_filter=structure_filter, strength=args.strength,
            )
            filter_applied = True

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
                bias_callback=filter_callback,
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
        filter_callback = make_dynamic_callback(
            pH=args.pH, target_charge=args.target_charge,
            structure_filter=structure_filter, strength=args.strength,
        )
        filter_applied = True
        print(f"[4] Sampling {args.num_samples} candidate sequences with guidance...")
        sampler = GuidedSampler(model, device=device)
        sequences, charges, pIs = [], [], []
        for i in range(args.num_samples):
            feature_dict["randn"] = torch.randn(1, L)
            out = sampler.sample(feature_dict, bias_callback=filter_callback)
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
        "cond_encoder": str(args.cond_encoder) if args.cond_encoder else None,
        "preset": args.preset, "temperature": args.temperature,
        "strength": args.strength, "seed": args.seed, "num_samples": args.num_samples,
        "native_charge": native_charge, "native_pI": native_pI,
        "mean_charge": mean_charge, "std_charge": std_charge,
        "sequences": [
            {"seq": s, "charge": c, "pI": p} for s, c, p in zip(sequences, charges, pIs)
        ],
    }
    summary.update(build_mode_metadata(
        sampling_mode, calibration_info, args.calibrate, filter_applied))
    summary["hybrid"] = bool(args.hybrid)
    summary["filter_preset"] = args.preset if filter_applied else None
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[6] Output saved: {fasta_path}")
    print("Done ✅")


if __name__ == "__main__":
    main()
