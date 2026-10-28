import contextlib
import csv
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def fake_torch_module():
    torch_stub = types.ModuleType("torch")
    torch_stub.zeros = lambda *shape, **kwargs: np.zeros(shape, **kwargs)
    torch_stub.from_numpy = np.asarray
    torch_stub.is_tensor = lambda value: False
    return torch_stub


def load_filter_module():
    import src

    path = CODE_DIR / "src" / "structure_aware_filter.py"
    spec = importlib.util.spec_from_file_location("src.structure_aware_filter", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": fake_torch_module(), "src.structure_aware_filter": module}):
        spec.loader.exec_module(module)
    return module


def load_conditioned_sampler_module():
    import src

    guided_stub = types.ModuleType("src.guided_sampler")
    guided_stub.guided_sample = Mock(return_value={"sentinel": True})
    path = CODE_DIR / "src" / "conditioned_sampler.py"
    spec = importlib.util.spec_from_file_location("src.conditioned_sampler", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "torch": fake_torch_module(),
        "src.guided_sampler": guided_stub,
        "src.conditioned_sampler": module,
    }):
        spec.loader.exec_module(module)
    return module


def load_run_guided_module():
    """Load runner helpers with upstream model and torch imports stubbed."""
    import src

    torch_stub = fake_torch_module()
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = Mock()
    data_utils_stub = types.ModuleType("data_utils")
    data_utils_stub.featurize = Mock()
    data_utils_stub.parse_PDB = Mock()
    data_utils_stub.restype_int_to_str = []
    model_utils_stub = types.ModuleType("model_utils")
    model_utils_stub.ProteinMPNN = object

    def module_stub(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    stubs = {
        "torch": torch_stub,
        "yaml": yaml_stub,
        "data_utils": data_utils_stub,
        "model_utils": model_utils_stub,
        "src.charge_lookahead": module_stub("src.charge_lookahead", make_dynamic_callback=Mock()),
        "src.condition_embedding": module_stub(
            "src.condition_embedding", ConditionEncoder=object, make_condition_vector=Mock()),
        "src.conditioned_sampler": module_stub("src.conditioned_sampler", conditioned_sample=Mock()),
        "src.differentiable_charge": module_stub("src.differentiable_charge", net_charge=Mock()),
        "src.guided_sampler": module_stub(
            "src.guided_sampler", GuidedSampler=object, extract_calpha_coords=Mock()),
        "src.isoelectric_point": module_stub("src.isoelectric_point", find_pI=Mock()),
        "src.structure_aware_filter": module_stub(
            "src.structure_aware_filter", FILTER_RULE_VERSION="test_local_density",
            StructureAwareFilter=object, load_preset=Mock()),
    }
    path = CODE_DIR / "run_guided.py"
    spec = importlib.util.spec_from_file_location("run_guided_test_module", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class FilterAndHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filter_module = load_filter_module()
        cls.conditioned_sampler = load_conditioned_sampler_module()

    def test_same_sign_rule_uses_local_neighborhood_not_chain_component(self):
        filt = self.filter_module.StructureAwareFilter(
            np.column_stack((np.arange(6) * 7.0, np.zeros((6, 2))))
        )
        sequence = np.array([
            self.filter_module.AA_TO_IDX["K"],
            self.filter_module.AA_TO_IDX["K"],
            self.filter_module.AA_TO_IDX["K"],
            self.filter_module.AA_TO_IDX["K"],
            self.filter_module.AA_TO_IDX["K"],
            self.filter_module.UNDECODED,
        ])

        bias, info = filt.compute_bias(sequence)

        self.assertEqual(info["same_sign_cluster"], {"pos_over": 0, "neg_over": 0})
        self.assertEqual(float(bias[5, self.filter_module.AA_TO_IDX["K"]]), 0.0)

    def test_same_sign_neighbor_counts_respect_validity_mask(self):
        mask = np.array([True, True, True, False, False, True])
        filt = self.filter_module.StructureAwareFilter(
            np.column_stack((np.arange(6, dtype=float), np.zeros((6, 2)))), mask=mask
        )
        sequence = np.array([
            self.filter_module.AA_TO_IDX["K"],
            self.filter_module.AA_TO_IDX["K"],
            self.filter_module.AA_TO_IDX["K"],
            self.filter_module.AA_TO_IDX["K"],
            self.filter_module.UNDECODED,
            self.filter_module.UNDECODED,
        ])

        bias, info = filt.compute_bias(sequence)

        self.assertEqual(info["same_sign_cluster"]["pos_over"], 0)
        self.assertEqual(float(bias[5, self.filter_module.AA_TO_IDX["K"]]), 0.0)

    def test_conditioned_sampler_forwards_optional_bias_callback(self):
        callback = object()
        model = Mock()
        model.encode.return_value = ("h_v", "h_e", "e_idx")
        sampler = self.conditioned_sampler
        guided = Mock(return_value={"sample": True})
        sampler.guided_sample = guided

        result = sampler.conditioned_sample(
            model, None, {"feature": 1}, None, device="cpu", bias_callback=callback
        )

        self.assertEqual(result, {"sample": True})
        self.assertIs(guided.call_args.kwargs["bias_callback"], callback)
        self.assertEqual(guided.call_args.kwargs["encoded"], ("h_v", "h_e", "e_idx"))


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_run_guided_module()
        spec = importlib.util.spec_from_file_location(
            "fit_charge_calibration_test_module", ROOT / "scripts" / "fit_charge_calibration.py")
        cls.fitter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fitter)

    def resolve(self, runner, **overrides):
        kwargs = {
            "target_charge": 5.0,
            "pdb_stem": "P1",
            "condition_injected": True,
            "auto_target": False,
            "no_calibration": False,
            "calibration_mode": "auto",
            "calibration_file": None,
            "calibration_file_explicit": False,
            "yaml_settings": {"enabled": False, "gain": 1.0, "offset": 0.0},
        }
        kwargs.update(overrides)
        return runner.resolve_charge_calibration(**kwargs)

    def test_off_missing_and_auto_target_report_not_applied(self):
        runner = self.runner
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.json"
            off = self.resolve(runner, calibration_mode="off", calibration_file=missing)
            self.assertFalse(off["applied"])
            self.assertEqual(off["status"], "disabled")

            absent = self.resolve(runner, calibration_file=missing)
            self.assertFalse(absent["applied"])
            self.assertEqual(absent["target_eff"], 5.0)
            self.assertEqual(absent["status"], "default_table_missing_yaml_disabled")

            auto = self.resolve(runner, auto_target=True, calibration_file=missing)
            self.assertFalse(auto["applied"])
            self.assertEqual(auto["status"], "auto_target_skipped")

    def test_explicit_missing_or_no_fit_table_errors(self):
        runner = self.runner
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                self.resolve(runner, calibration_file=missing, calibration_file_explicit=True)

            empty = Path(temp_dir) / "empty.json"
            empty.write_text(json.dumps({"global": {}, "per_protein": {}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no usable"):
                runner.load_calibration(empty, "P1")

    def test_valid_fit_applies_and_summary_keeps_route_and_fit_metadata(self):
        runner = self.runner
        with tempfile.TemporaryDirectory() as temp_dir:
            table = Path(temp_dir) / "calibration.json"
            table.write_text(json.dumps({
                "label": "test table",
                "global": {"slope": 2.0, "intercept": 1.0},
                "per_protein": {"P1": {"slope": 4.0, "intercept": 1.0}},
            }), encoding="utf-8")
            result = self.resolve(runner, calibration_file=table)
            self.assertTrue(result["applied"])
            self.assertEqual(result["target_eff"], 1.0)
            self.assertEqual(result["fit_mode"], "per-protein(P1)")

            metadata = runner.build_mode_metadata("phase3_conditioned", result, "auto", True)
            self.assertEqual(metadata["mode"], "phase3_conditioned")
            self.assertTrue(metadata["calibrated"])
            self.assertEqual(metadata["calibration_slope"], 4.0)
            self.assertEqual(metadata["calibration_intercept"], 1.0)
            self.assertEqual(metadata["calibration_table_used"], str(table))
            self.assertEqual(metadata["filter_rule_version"], "test_local_density")

    def test_hybrid_cli_requires_conditioned_encoder_mode(self):
        runner = self.runner
        args = ["--pdb", "input.pdb", "--pH", "7.4", "--hybrid"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(args)
        args += ["--cond_encoder", "encoder.pt", "--cond_mode", "baseline"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parse_args(args)

    def test_fit_tool_json_roundtrip_matches_runner_reader(self):
        runner = self.runner
        fitter = self.fitter
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "observations.csv"
            output_path = Path(temp_dir) / "calibration.json"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["protein", "target_charge", "observed_charge"])
                writer.writeheader()
                for target, observed in [(0, 1), (1, 3), (2, 5)]:
                    writer.writerow({"protein": "P1", "target_charge": target, "observed_charge": observed})
                writer.writerow({"protein": "P2", "target_charge": 0, "observed_charge": 0})

            rows = fitter.read_observations(csv_path)
            table = fitter.fit_calibration_table(rows)
            fitter.write_table(table, output_path)
            slope, intercept, mode, _ = runner.load_calibration(output_path, "P1")

            self.assertAlmostEqual(slope, 2.0)
            self.assertAlmostEqual(intercept, 1.0)
            self.assertEqual(mode, "per-protein(P1)")
            self.assertIn("P2", table["per_protein_skipped"])
            with self.assertRaises(FileExistsError):
                fitter.write_table(table, output_path)

    def test_fit_tool_rejects_nonfinite_and_unusable_targets(self):
        fitter = self.fitter
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "bad.csv"
            csv_path.write_text(
                "protein,target_charge,observed_charge\nP1,1,nan\nP1,1,2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite"):
                fitter.read_observations(csv_path)
            with self.assertRaisesRegex(ValueError, "two distinct"):
                fitter.fit_linear([
                    {"target_charge": 1.0, "observed_charge": 2.0},
                    {"target_charge": 1.0, "observed_charge": 3.0},
                ], "test fit")


if __name__ == "__main__":
    unittest.main()
