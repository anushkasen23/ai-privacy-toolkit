"""Integration test for privacy floor enforcement and halt behavior."""

import unittest

from apt.minimization import default_closed_loop_config, run_closed_loop_minimization


class PrivacyFloorEnforcementTests(unittest.TestCase):
    def test_halts_when_privacy_floor_violated(self) -> None:
        cfg = default_closed_loop_config()
        cfg["runtime"]["epochs"] = 3
        cfg["privacy_floor"]["prs_max"] = 0.15

        result = run_closed_loop_minimization(cfg)

        self.assertTrue(result["halted"])
        self.assertEqual(result["epochs_ran"], 2)
        self.assertTrue(result["history"][-1]["decision"].halt_training)


if __name__ == "__main__":
    unittest.main()
