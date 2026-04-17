import sys
import unittest
import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parent / "grape_reference.py"
spec = importlib.util.spec_from_file_location("grape_reference_script", MODULE_PATH)
grape_reference = importlib.util.module_from_spec(spec)
sys.modules["grape_reference_script"] = grape_reference
assert spec.loader is not None
spec.loader.exec_module(grape_reference)


class GrapeReferenceTests(unittest.TestCase):
    def test_build_drive_schedule_is_deterministic(self):
        a = grape_reference.build_drive_schedule(5)
        b = grape_reference.build_drive_schedule(5)
        self.assertEqual(a.shape, (5,))
        self.assertTrue(np.allclose(a, b))

    def test_run_piecewise_mesolve_returns_density_matrix(self):
        rho = grape_reference.run_piecewise_mesolve(
            d=3,
            n_segments=4,
            steps_per_seg=2,
            dt=0.5,
            solver_mode="default",
        )
        self.assertEqual(rho.shape, (3, 3))
        self.assertAlmostEqual(float(rho.tr().real), 1.0, places=6)

    def test_bench_piecewise_mesolve_returns_positive_time(self):
        result = grape_reference.bench_piecewise_mesolve(
            d=3,
            n_segments=4,
            steps_per_seg=2,
            n_trials=1,
            dt=0.5,
            solver_mode="default",
        )
        self.assertEqual(result["d"], 3)
        self.assertEqual(result["solver_mode"], "default")
        self.assertGreater(result["ms_per_trajectory"], 0.0)


if __name__ == "__main__":
    unittest.main()
