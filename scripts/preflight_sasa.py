"""Check that one PDB can be mapped to a finite per-residue SASA profile."""

import argparse
from importlib import metadata
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"


def preflight_sasa(pdb_path, threshold=0.25):
    """Run the public SASA helper and verify its residue-area coverage.

    This preflight supports one model and one protein chain with standard
    amino-acid residues, blank insertion codes, and unique residue numbers.
    Other chains/residues (for example RNA, ligands, and water) may remain as
    structural context. It checks SASA mapping only, not training masks/losses.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise ValueError(f"PDB file does not exist: {pdb_path}")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")

    from Bio.PDB import PDBParser
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from src.sasa import _RES3TO1, fractional_sasa

    structure = PDBParser(QUIET=True).get_structure("preflight", str(pdb_path))
    models = list(structure)
    if len(models) != 1:
        raise ValueError(f"expected exactly one PDB model; found {len(models)}")

    standard_residues = []
    all_residues = []
    for chain in models[0]:
        for residue in chain:
            key = (chain.id, residue.id[1])
            all_residues.append(key)
            resname = residue.get_resname().strip()
            if resname in _RES3TO1:
                if not residue.has_id("CA"):
                    raise ValueError(
                        f"standard amino-acid residue {chain.id}:{residue.id[1]} "
                        "has no CA atom"
                    )
                if residue.id[2].strip():
                    raise ValueError(
                        f"insertion codes are unsupported for SASA mapping: "
                        f"{chain.id}:{residue.id[1]}{residue.id[2].strip()}"
                    )
                standard_residues.append((chain.id, residue.id[1], _RES3TO1[resname]))

    protein_chains = {chain for chain, _, _ in standard_residues}
    if len(protein_chains) != 1:
        raise ValueError(
            "expected standard amino-acid residues in exactly one protein chain; "
            f"found {len(protein_chains)}"
        )
    residue_numbers = [resid for _, resid, _ in standard_residues]
    if len(set(residue_numbers)) != len(residue_numbers):
        raise ValueError("duplicate standard amino-acid residue numbers are unsupported")
    if len(all_residues) != len(set(all_residues)):
        raise ValueError("duplicate chain/residue-number keys make SASA mapping ambiguous")
    if not standard_residues:
        raise ValueError("no standard amino-acid residues with CA atoms were found")

    # Reuse the public calculation and separately inspect raw areas because
    # fractional_sasa replaces missing/non-finite values with zero.
    profile = fractional_sasa(pdb_path, surface_threshold=threshold, align_to_full=True)
    selected = [
        (int(resid), aa, float(frac))
        for resid, aa, frac in zip(profile["residues"], profile["seq"], profile["frac_sasa"])
        if int(resid) != -1
    ]
    expected = [(resid, aa) for _, resid, aa in standard_residues]
    if [(resid, aa) for resid, aa, _ in selected] != expected:
        raise ValueError("SASA helper residue order/identity does not match parsed protein residues")

    import freesasa

    raw_structure = freesasa.structureFromBioPDB(structure)
    raw_areas = freesasa.calc(raw_structure).residueAreas()
    protein_chain = next(iter(protein_chains))
    chain_areas = raw_areas.get(protein_chain, {})
    missing, invalid, mismatched = [], [], []
    for resid, _, helper_value in selected:
        area = chain_areas.get(str(resid))
        if area is None:
            missing.append(resid)
            continue
        value = area.relativeTotal
        if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
            invalid.append(resid)
        elif not math.isclose(float(value), helper_value, rel_tol=1e-7, abs_tol=1e-8):
            mismatched.append(resid)
    if missing or invalid or mismatched:
        raise ValueError(
            f"SASA coverage failed: {len(missing)} residue areas missing, "
            f"{len(invalid)} invalid relativeTotal values (non-finite or negative); "
            f"{len(mismatched)} helper/raw values disagree; missing={missing[:10]}, "
            f"invalid={invalid[:10]}, mismatched={mismatched[:10]}"
        )

    surface_count = sum(frac >= threshold for _, _, frac in selected)
    residue_count = len(selected)
    try:
        freesasa_version = metadata.version("freesasa")
    except metadata.PackageNotFoundError:
        freesasa_version = getattr(freesasa, "__version__", "unknown")
    return {
        "pdb": str(pdb_path),
        "freesasa_version": freesasa_version,
        "protein_chain": protein_chain,
        "threshold": threshold,
        "standard_residues": residue_count,
        "mapped_residues": len(selected),
        "missing_residues": missing,
        "invalid_residues": invalid,
        "surface_residues": surface_count,
        "core_residues": residue_count - surface_count,
        "status": "pass",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, help="single-model PDB to check")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--json-out", help="optional path for the JSON report")
    args = parser.parse_args(argv)

    try:
        report = preflight_sasa(args.pdb, args.threshold)
    except Exception as exc:
        print(f"SASA preflight failed: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
