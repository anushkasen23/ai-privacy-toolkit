"""Integration test for closed-loop minimization invocation sequence."""

import unittest

from apt.minimization.closed_loop_privacy import default_closed_loop_config, run_closed_loop_minimization


class PipelineOrderTests(unittest.TestCase):
    def test_pipeline_step_order(self) -> None:
        cfg = default_closed_loop_config()
        cfg["runtime"]["epochs"] = 1

        result = run_closed_loop_minimization(cfg)
        self.assertEqual(result["epochs_ran"], 1)
        self.assertFalse(result["halted"])
        self.assertEqual(
            result["trace"][0],
            [
                "prepare_dataset_and_model",
                "fit_and_apply_minimizer",
                "diversity_flag_and_mitigate",
                "clip_privatize_and_account",
                "run_mia_and_compute_risk",
                "governor_adjust_or_halt",
                "print_epoch_summary",
            ],
        )


if __name__ == "__main__":
    unittest.main()
