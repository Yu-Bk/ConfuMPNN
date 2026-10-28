"""Fit a charge-calibration JSON table from user-generated observations.

CSV columns: protein,target_charge,observed_charge.  The observed charge must
come from generated sequences produced under one consistent model and sampling
setup; this script does not generate calibration observations.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


MIN_SLOPE = 1e-8
REQUIRED_COLUMNS = ("protein", "target_charge", "observed_charge")


def read_observations(path):
    """Read and validate measured target/observed charge rows from CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Observation CSV does not exist: {path}")

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Observation CSV is empty: {path}")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError("Observation CSV is missing columns: " + ", ".join(missing))
        for line_number, row in enumerate(reader, start=2):
            protein = (row.get("protein") or "").strip()
            if not protein:
                raise ValueError(f"Missing protein identifier at CSV line {line_number}")
            try:
                target = float(row["target_charge"])
                observed = float(row["observed_charge"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid charge value at CSV line {line_number}") from exc
            if not math.isfinite(target) or not math.isfinite(observed):
                raise ValueError(f"Charge values must be finite at CSV line {line_number}")
            rows.append({
                "protein": protein,
                "target_charge": target,
                "observed_charge": observed,
            })

    if not rows:
        raise ValueError(f"Observation CSV contains no data rows: {path}")
    return rows


def fit_linear(rows, label):
    """Fit observed_charge = slope * target_charge + intercept."""
    targets = np.asarray([row["target_charge"] for row in rows], dtype=float)
    observed = np.asarray([row["observed_charge"] for row in rows], dtype=float)
    if len(np.unique(targets)) < 2:
        raise ValueError(f"{label} needs at least two distinct target_charge values")

    centered = targets - targets.mean()
    denominator = float(np.dot(centered, centered))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError(f"{label} has no usable target-charge variation")
    slope = float(np.dot(centered, observed - observed.mean()) / denominator)
    intercept = float(observed.mean() - slope * targets.mean())
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= MIN_SLOPE:
        raise ValueError(f"{label} fit requires a finite positive, nonzero slope")
    residual = observed - (slope * targets + intercept)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return {
        "slope": slope,
        "intercept": intercept,
        "n": int(len(rows)),
        "n_distinct_targets": int(len(np.unique(targets))),
        "rmse": rmse,
    }


def fit_calibration_table(rows, label="user-supplied observations"):
    """Build the schema consumed by code/run_guided.py::load_calibration."""
    if not rows:
        raise ValueError("At least two observations are required")
    global_fit = fit_linear(rows, "Global fit")

    grouped = {}
    for row in rows:
        grouped.setdefault(row["protein"], []).append(row)

    per_protein = {}
    skipped = {}
    for protein, protein_rows in sorted(grouped.items()):
        try:
            per_protein[protein] = fit_linear(protein_rows, f"Protein {protein}")
        except ValueError as exc:
            skipped[protein] = str(exc)

    return {
        "label": label,
        "global": global_fit,
        "per_protein": per_protein,
        "per_protein_skipped": skipped,
        "fit_method": "ordinary_least_squares",
        "input_columns": list(REQUIRED_COLUMNS),
    }


def write_table(table, output, overwrite=False):
    """Write JSON atomically, refusing to replace a file unless requested."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (pass --overwrite to replace): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(table, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with protein,target_charge,observed_charge columns")
    parser.add_argument("--output", required=True, help="Output calibration JSON path")
    parser.add_argument("--label", default="user-supplied observations", help="Human-readable calibration-table label")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output JSON")
    args = parser.parse_args(argv)

    rows = read_observations(args.input)
    table = fit_calibration_table(rows, label=args.label)
    output = write_table(table, args.output, overwrite=args.overwrite)
    print(f"Wrote calibration table: {output}")
    print(f"Global fit: n={table['global']['n']}, slope={table['global']['slope']:.6g}, "
          f"intercept={table['global']['intercept']:.6g}")
    print(f"Per-protein fits: {len(table['per_protein'])}; skipped: {len(table['per_protein_skipped'])}")
    for protein, reason in table["per_protein_skipped"].items():
        print(f"  Skipped {protein}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
