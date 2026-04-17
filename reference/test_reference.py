import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

REFERENCE_PATH = Path(__file__).resolve().parent / "reference.py"
spec = importlib.util.spec_from_file_location("reference_script", REFERENCE_PATH)
reference = importlib.util.module_from_spec(spec)
sys.modules["reference_script"] = reference
assert spec.loader is not None
spec.loader.exec_module(reference)


class ReferenceHarnessTests(unittest.TestCase):
    def test_build_transmon_system_shapes(self):
        H, cops = reference.build_transmon_system(3)
        self.assertEqual(H.shape, (3, 3))
        self.assertGreaterEqual(len(cops), 1)
        self.assertEqual(cops[0].shape, (3, 3))

    def test_matrix_form_support_probe_returns_bool(self):
        supported = reference.matrix_form_supported()
        self.assertIsInstance(supported, bool)

    def test_default_single_step_returns_density_matrix(self):
        H, cops = reference.build_transmon_system(3)
        rho0 = reference.ground_state(3)
        rho1 = reference.propagate_step_qutip(H, cops, rho0, 0.5, solver_mode="default")
        self.assertEqual(rho1.shape, (3, 3))
        self.assertAlmostEqual(float(rho1.tr().real), 1.0, places=6)

    def test_bench_single_default_has_positive_time(self):
        result = reference.bench_single(3, n_reps=2, solver_mode="default")
        self.assertEqual(result["d"], 3)
        self.assertEqual(result["solver_mode"], "default")
        self.assertGreater(result["ns_per_step"], 0.0)

    def test_matrix_form_matches_default_when_supported(self):
        if not reference.matrix_form_supported():
            self.skipTest("Installed QuTiP release does not support matrix_form")

        H, cops = reference.build_transmon_system(3)
        rho0 = reference.ground_state(3)
        rho_default = reference.propagate_step_qutip(
            H, cops, rho0, 0.5, solver_mode="default"
        )
        rho_matrix = reference.propagate_step_qutip(
            H, cops, rho0, 0.5, solver_mode="matrix_form"
        )
        err = np.linalg.norm(rho_default.full() - rho_matrix.full(), "fro")
        self.assertLess(err, 1e-9)


if __name__ == "__main__":
    unittest.main()
