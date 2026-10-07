import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_environment


class CheckEnvironmentTests(unittest.TestCase):
    def test_missing_optional_checkpoint_and_sasa_do_not_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ligand = root / "LigandMPNN"
            ligand.mkdir()
            for name in check_environment.LIGANDMPNN_FILES:
                (ligand / name).touch()

            def version(distribution):
                return None if distribution == "freesasa" else "1.0"

            output = io.StringIO()
            with patch.object(check_environment, "package_version", side_effect=version), \
                    patch.object(check_environment, "module_available", return_value=False), \
                    contextlib.redirect_stdout(output):
                status = check_environment.check_environment(root)

            self.assertEqual(status, 0)
            self.assertIn("MoMPNN default checkpoint: [optional, missing]", output.getvalue())
            self.assertIn("FreeSASA: [optional, missing]", output.getvalue())
            self.assertIn("Bio.PDB: [missing]", output.getvalue())

    def test_missing_core_dependency_or_ligand_source_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def version(distribution):
                return None if distribution == "numpy" else "1.0"

            output = io.StringIO()
            with patch.object(check_environment, "package_version", side_effect=version), \
                    patch.object(check_environment, "module_available", return_value=False), \
                    contextlib.redirect_stdout(output):
                status = check_environment.check_environment(root)

            self.assertEqual(status, 1)
            self.assertIn("[MISSING] NumPy", output.getvalue())
            self.assertIn("LigandMPNN source: [MISSING]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
