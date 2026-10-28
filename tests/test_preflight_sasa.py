import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import preflight_sasa


class Residue:
    def __init__(self, name, number, insertion=" ", ca=True):
        self.name = name
        self.id = (" ", number, insertion)
        self.ca = ca

    def get_resname(self):
        return self.name

    def has_id(self, atom):
        return atom == "CA" and self.ca


class Chain:
    id = "A"

    def __init__(self, residues):
        self.residues = residues

    def __iter__(self):
        return iter(self.residues)


class Model:
    def __init__(self, chains):
        self.chains = chains

    def __iter__(self):
        return iter(self.chains)


class Structure:
    def __init__(self, models):
        self.models = models

    def __iter__(self):
        return iter(self.models)


def install_parser(structure):
    bio = types.ModuleType("Bio")
    bio.__path__ = []
    bio_pdb = types.ModuleType("Bio.PDB")

    class Parser:
        def __init__(self, **kwargs):
            pass

        def get_structure(self, name, path):
            return structure

    bio_pdb.PDBParser = Parser
    return patch.dict(sys.modules, {"Bio": bio, "Bio.PDB": bio_pdb})


def install_freesasa(areas):
    module = types.ModuleType("freesasa")
    module.__version__ = "test"
    module.structureFromBioPDB = lambda structure: structure

    class Result:
        def residueAreas(self):
            return areas

    module.calc = lambda structure: Result()
    return patch.dict(sys.modules, {"freesasa": module})


def install_fractional_sasa(profile):
    package = types.ModuleType("src")
    package.__path__ = []
    module = types.ModuleType("src.sasa")
    module._RES3TO1 = {"ALA": "A", "GLY": "G"}
    module.fractional_sasa = lambda *args, **kwargs: profile
    return patch.dict(sys.modules, {"src": package, "src.sasa": module})


class PreflightSasaTests(unittest.TestCase):
    def valid_structure(self):
        return Structure([Model([Chain([Residue("ALA", 1), Residue("GLY", 2)])])])

    def valid_profile(self):
        return {
            "seq": "AG",
            "residues": [1, 2],
            "frac_sasa": [0.1, 0.4],
        }

    def test_reports_real_residue_coverage_and_surface_core_counts(self):
        structure = self.valid_structure()
        areas = {"A": {"1": types.SimpleNamespace(relativeTotal=0.1),
                        "2": types.SimpleNamespace(relativeTotal=0.4)}}
        with tempfile.NamedTemporaryFile() as pdb, install_parser(structure), install_freesasa(areas), \
                install_fractional_sasa(self.valid_profile()):
            report = preflight_sasa.preflight_sasa(pdb.name)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["mapped_residues"], 2)
        self.assertEqual(report["surface_residues"], 1)
        self.assertEqual(report["core_residues"], 1)

    def test_missing_raw_area_fails_even_if_helper_would_fill_zero(self):
        structure = self.valid_structure()
        with tempfile.NamedTemporaryFile() as pdb, install_parser(structure), install_freesasa({"A": {"1": types.SimpleNamespace(relativeTotal=0.1)}}), \
                install_fractional_sasa(self.valid_profile()):
            with self.assertRaisesRegex(ValueError, "1 residue areas missing"):
                preflight_sasa.preflight_sasa(pdb.name)

    def test_rejects_insertion_code_and_multiple_models(self):
        insertion = Structure([Model([Chain([Residue("ALA", 1, "A")])])])
        multiple = Structure([Model([Chain([Residue("ALA", 1)])]), Model([Chain([Residue("ALA", 1)])])])
        with tempfile.NamedTemporaryFile() as pdb, install_parser(insertion), install_fractional_sasa(self.valid_profile()):
            with self.assertRaisesRegex(ValueError, "insertion codes"):
                preflight_sasa.preflight_sasa(pdb.name)
        with tempfile.NamedTemporaryFile() as pdb, install_parser(multiple), install_fractional_sasa(self.valid_profile()):
            with self.assertRaisesRegex(ValueError, "exactly one PDB model"):
                preflight_sasa.preflight_sasa(pdb.name)

    def test_rejects_nonfinite_sasa(self):
        structure = self.valid_structure()
        areas = {"A": {"1": types.SimpleNamespace(relativeTotal=float("nan")),
                        "2": types.SimpleNamespace(relativeTotal=0.4)}}
        with tempfile.NamedTemporaryFile() as pdb, install_parser(structure), install_freesasa(areas), \
                install_fractional_sasa(self.valid_profile()):
            with self.assertRaisesRegex(ValueError, "non-finite or negative"):
                preflight_sasa.preflight_sasa(pdb.name)

    def test_rejects_negative_sasa_even_if_helper_would_clip_it(self):
        structure = self.valid_structure()
        areas = {"A": {"1": types.SimpleNamespace(relativeTotal=-0.1),
                        "2": types.SimpleNamespace(relativeTotal=0.4)}}
        with tempfile.NamedTemporaryFile() as pdb, install_parser(structure), install_freesasa(areas), \
                install_fractional_sasa(self.valid_profile()):
            with self.assertRaisesRegex(ValueError, "negative"):
                preflight_sasa.preflight_sasa(pdb.name)

    def test_rejects_helper_values_that_do_not_match_raw_areas(self):
        structure = self.valid_structure()
        areas = {"A": {"1": types.SimpleNamespace(relativeTotal=0.1),
                        "2": types.SimpleNamespace(relativeTotal=0.4)}}
        profile = {"seq": "AG", "residues": [1, 2], "frac_sasa": [0.2, 0.4]}
        with tempfile.NamedTemporaryFile() as pdb, install_parser(structure), install_freesasa(areas), \
                install_fractional_sasa(profile):
            with self.assertRaisesRegex(ValueError, "helper/raw values disagree"):
                preflight_sasa.preflight_sasa(pdb.name)

    def test_rejects_invalid_threshold(self):
        with tempfile.NamedTemporaryFile() as pdb:
            with self.assertRaisesRegex(ValueError, "threshold must be finite"):
                preflight_sasa.preflight_sasa(pdb.name, float("nan"))

    @unittest.skipUnless(importlib.util.find_spec("freesasa"), "FreeSASA is unavailable")
    def test_real_single_chain_pdb(self):
        pdb_text = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.100   1.400   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       1.500   2.450   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.950  -0.800   1.200  1.00 20.00           C
ATOM      6  N   GLY A   2       3.400   1.500   0.000  1.00 20.00           N
ATOM      7  CA  GLY A   2       4.050   2.750   0.000  1.00 20.00           C
ATOM      8  C   GLY A   2       3.400   4.050   0.000  1.00 20.00           C
ATOM      9  O   GLY A   2       2.200   4.000   0.000  1.00 20.00           O
TER
END
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdb = Path(temp_dir) / "mini.pdb"
            json_report = Path(temp_dir) / "mini_sasa.json"
            pdb.write_text(pdb_text, encoding="ascii")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = preflight_sasa.main(
                    ["--pdb", str(pdb), "--json-out", str(json_report)]
                )
            report = json.loads(json_report.read_text(encoding="utf-8"))
        self.assertEqual(status, 0)
        self.assertIn('"status": "pass"', output.getvalue())
        self.assertEqual(report["mapped_residues"], 2)


if __name__ == "__main__":
    unittest.main()
