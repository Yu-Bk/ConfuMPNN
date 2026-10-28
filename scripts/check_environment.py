"""Read-only checks for the ConfuMPNN public repository's runtime setup."""

from importlib import metadata, util
import argparse
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE_DISTRIBUTIONS = (
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("PyYAML", "PyYAML"),
    ("ProDy", "ProDy"),
    ("Biopython", "biopython"),
    ("PyTorch", "torch"),
)
LIGANDMPNN_FILES = ("data_utils.py", "model_utils.py")
MOMPNN_CHECKPOINT = Path(
    "MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt"
)


def package_version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def module_available(module):
    try:
        return util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_environment(root=ROOT, sasa_pdb=None, sasa_threshold=0.25, sasa_json=None):
    root = Path(root)
    failures = []
    print(f"Python: {platform.python_version()} (audited environment: 3.11.16)")
    print("Core packages:")
    for label, distribution in CORE_DISTRIBUTIONS:
        version = package_version(distribution)
        if version is None:
            failures.append(label)
            print(f"  [MISSING] {label}")
            if distribution == "torch":
                print("    Install a compatible PyTorch build separately; see the README.")
            else:
                print("    Hint: python -m pip install -r requirements.txt")
        else:
            print(f"  [OK] {label} {version}")

    missing_ligand_files = [
        name for name in LIGANDMPNN_FILES
        if not (root / "LigandMPNN" / name).is_file()
    ]
    if missing_ligand_files:
        failures.append("LigandMPNN source")
        print("LigandMPNN source: [MISSING] " + ", ".join(missing_ligand_files))
        print("  Hint: git clone https://github.com/dauparas/LigandMPNN.git LigandMPNN")
    else:
        print("LigandMPNN source: [OK] data_utils.py and model_utils.py found")

    checkpoint = root / MOMPNN_CHECKPOINT
    if checkpoint.is_file():
        print(f"MoMPNN default checkpoint: [available] {MOMPNN_CHECKPOINT.as_posix()}")
    else:
        print(f"MoMPNN default checkpoint: [optional, missing] {MOMPNN_CHECKPOINT.as_posix()}")
        print("  Hint: provide the upstream checkpoint at this path or pass --weights.")

    freesasa = module_available("freesasa")
    sasa_version = package_version("freesasa") if freesasa else None
    if freesasa:
        suffix = f" {sasa_version}" if sasa_version else ""
        print(f"FreeSASA: [available]{suffix}")
    else:
        print("FreeSASA: [optional, missing]")
        print("  Hint: python -m pip install -r requirements-sasa.txt")

    if module_available("Bio.PDB"):
        print("Bio.PDB: [available]")
    else:
        print("Bio.PDB: [missing] SASA structure parsing is unavailable")

    if sasa_pdb is not None:
        sys.path.insert(0, str(root))
        try:
            from scripts.preflight_sasa import preflight_sasa

            report = preflight_sasa(sasa_pdb, sasa_threshold)
            print("SASA structure preflight: [passed]")
            print(
                f"  Chain {report['protein_chain']}: "
                f"{report['mapped_residues']}/{report['standard_residues']} residues mapped; "
                f"surface={report['surface_residues']}, core={report['core_residues']}"
            )
            if sasa_json:
                import json
                Path(sasa_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(f"  JSON report: {sasa_json}")
        except Exception as exc:
            failures.append("SASA structure preflight")
            print(f"SASA structure preflight: [FAILED] {exc}")

    if failures:
        print("Environment check failed: install the missing core dependencies/source above.")
        return 1
    print("Environment check passed for mandatory dependencies and LigandMPNN source.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sasa-pdb", help="run a real SASA calculation on a PDB")
    parser.add_argument("--sasa-threshold", type=float, default=0.25)
    parser.add_argument("--sasa-json", help="optional JSON report path (requires --sasa-pdb)")
    args = parser.parse_args()
    if args.sasa_json and not args.sasa_pdb:
        parser.error("--sasa-json requires --sasa-pdb")
    raise SystemExit(
        check_environment(
            sasa_pdb=args.sasa_pdb,
            sasa_threshold=args.sasa_threshold,
            sasa_json=args.sasa_json,
        )
    )
