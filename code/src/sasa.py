"""Compute per-residue fractional solvent-accessible surface area (SASA).

The fractional SASA for each residue is obtained from freesasa's
`relativeTotal` value. The calculation uses Bio.PDB to parse the structure and
freesasa to calculate residue areas; only standard amino acid residues are
included, while nucleic acids, ligands, and water are skipped.

Dependencies: freesasa (`pip install freesasa`) and Biopython.
"""

from pathlib import Path

import numpy as np

# The 20 standard amino acids, used to identify protein residues.
_AA20 = set("ACDEFGHIKLMNPQRSTVWY")
# Map three-letter protein residue names used by freesasa to one-letter codes.
_RES3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def fractional_sasa(pdb_path, surface_threshold=0.25, align_to_full=True):
    """Calculate per-residue fractional SASA for a PDB structure.

    Args:
        pdb_path: Path to the protein structure; ligands and nucleic acids are skipped.
        surface_threshold: A residue is classified as a surface site when
            fracSASA is greater than or equal to this threshold (default 0.25).
        align_to_full: If True, preserve positions for non-standard residues
            using X and a fractional SASA of 0.0. The returned arrays then have
            the same length as the parsed residue sequence. If False, return
            standard amino acids only.

    Returns:
        A dictionary containing the sequence, per-residue fractional SASA,
        surface mask, and number of surface sites. With `align_to_full=True`,
        X marks non-standard residue positions and their fractional SASA is 0.0.

    Raises:
        RuntimeError: If no protein residues are found or freesasa is unavailable.
    """
    try:
        import freesasa
    except ImportError:
        raise RuntimeError("freesasa is required (install with pip install freesasa)")

    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))

    # Collect (chain, residue number, amino acid, is-standard-amino-acid) in PDB order.
    residues = []  # Each entry contains: chain, resid, aa, and is_aa.
    for model in structure:
        for chain in model:
            for residue in chain:
                res3 = residue.get_resname().strip()
                aa = _RES3TO1.get(res3)
                residues.append({
                    "chain": chain.id,
                    "resid": residue.id[1],
                    "aa": aa,
                    "is_aa": aa is not None and residue.has_id("CA"),
                })

    if not any(r["is_aa"] for r in residues):
        raise RuntimeError(f"No protein residues found in {pdb_path}")

    # Calculate freesasa values for the full structure and index by chain and residue number.
    try:
        fs = freesasa.structureFromBioPDB(structure)
        result = freesasa.calc(fs)
        residue_areas = result.residueAreas()  # {chain: {resnum: ResidueArea}}
    except Exception as e:
        raise RuntimeError(f"freesasa calculation failed: {e}")

    # Populate fractional SASA values for each residue.
    seq_chars, fracs, resids = [], [], []
    for r in residues:
        if not r["is_aa"]:
            seq_chars.append("-" if align_to_full else None)  # Non-standard residue or missing Cα.
            fracs.append(0.0 if align_to_full else None)      # Fractional SASA is 0 at X positions.
            resids.append(None if align_to_full else None)
            continue
        chain_areas = residue_areas.get(r["chain"], {})
        ra = chain_areas.get(str(r["resid"]))
        frac = ra.relativeTotal if ra is not None else 0.0
        seq_chars.append(r["aa"])
        fracs.append(frac)
        resids.append(r["resid"])   # Residue number, used for residue-number alignment.

    if align_to_full:
        # Preserve all parsed residue positions, using X and 0.0 for non-standard residues.
        seq = "".join(c if c is not None else "X" for c in seq_chars)
        frac_arr = np.clip(np.array(fracs, dtype=np.float64), 0.0, None)
        resid_arr = np.array([r if r is not None else -1 for r in resids], dtype=np.int64)
    else:
        # Keep standard amino acids only and return their residue numbers.
        keep = [(c, f, rid) for c, f, rid in zip(seq_chars, fracs, resids) if c is not None]
        seq = "".join(c for c, _, _ in keep)
        frac_arr = np.clip(np.array([f for _, f, _ in keep], dtype=np.float64), 0.0, None)
        resid_arr = np.array([rid for _, _, rid in keep], dtype=np.int64)

    # Replace invalid relativeTotal values with 0.0.
    frac_arr = np.nan_to_num(frac_arr, nan=0.0, posinf=0.0, neginf=0.0)

    surface_mask = frac_arr >= surface_threshold
    return {
        "seq": seq,
        "frac_sasa": frac_arr,
        "residues": resid_arr,       # Residue numbers aligned with frac_sasa.
        "surface_mask": surface_mask,
        "is_surface": int(surface_mask.sum()),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Per-residue fractional SASA")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="Optional output NPZ path")
    args = ap.parse_args()

    r = fractional_sasa(args.pdb, surface_threshold=args.threshold)
    print(f"Sequence length {len(r['seq'])}; surface sites {r['is_surface']} / {len(r['seq'])}")
    print(f"fracSASA: min={r['frac_sasa'].min():.3f} max={r['frac_sasa'].max():.3f} "
          f"mean={r['frac_sasa'].mean():.3f}")
    # Show a sample of surface sites.
    surf = [(i, aa) for i, (aa, m) in enumerate(zip(r["seq"], r["surface_mask"])) if m][:10]
    print(f"Example surface sites (first 10): {surf}")
    if args.out:
        np.savez(args.out, seq=np.array(list(r["seq"]), dtype="U1"),
                 frac_sasa=r["frac_sasa"], surface_mask=r["surface_mask"])
        print(f"Wrote {args.out}")
