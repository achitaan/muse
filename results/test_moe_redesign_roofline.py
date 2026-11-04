"""Analytical gates only; no GPU correctness or timing claims."""
import unittest

import moe_redesign_roofline as r


class RooflineTests(unittest.TestCase):
    def test_exact_nvfp4_byte_accounting(self):
        # NVFP4: two E2M1 values per byte, one E4M3 scale per 16 values;
        # W13 has 2*H*H values, W2 has H*H values.
        self.assertEqual(r.CODE_BYTES, 24 * 2**20)
        self.assertEqual(r.SCALE_BYTES, 3 * 2**20)
        self.assertEqual(r.EXPERT_BYTES, 27 * 2**20)

    def test_actual_histogram_counter(self):
        # Explicit constructed routes: local experts 0, 1, 31; non-local ids
        # do not read any local weights. Repeats do not reread expert weights.
        result = r.actual_weight_bytes([
            [0, 1, 31, 32, 33, 100, 200, 255],
            [1, 0, 31, 50, 51, 52, 53, 54],
        ])
        self.assertEqual(result["distinct"], 3)
        self.assertEqual(result["routes"], 6)
        self.assertEqual(result["weight_bytes"], 81 * 2**20)
        self.assertEqual(r.actual_weight_bytes([[32, 33, 34, 35, 36, 37, 38, 39]])["weight_bytes"], 0)
        self.assertEqual(r.actual_weight_bytes([[32, 33, 34, 35, 36, 37, 38, 39]], local_start=32)["distinct"], 8)

    def test_nominal_floor_share_is_removed(self):
        report = r.report()
        for s, expected_d in zip(report["shape_estimates"], (11.86, 15.97)):
            # Expected nominal means are obtained by independently inverting
            # the public reference's 0.75-share byte-bound formula.
            self.assertAlmostEqual(s["nominal_mean_distinct"], expected_d)
            self.assertGreater(s["corrected_nominal_hbm_floor_us"], s["max_unpadded_math_us"])
        self.assertFalse(report["actual_job_route_histograms_available"])

    def test_board_weighting_and_build_rejection(self):
        # Exactly two of six equally weighted loads change, so the product
        # of the two shape speedups is raised to 1/6, not to 1/2.
        self.assertAlmostEqual(r.projected_board({16: r.CASES[1][1] / 2})["score"], r.CHECK_BOARD * 2 ** (1 / 6))
        report = r.report()
        self.assertGreater(report["hbm_only_ceiling"]["gain_vs_best"], .03)
        self.assertLess(report["conditional_target"]["gain_vs_best"], .03)
        self.assertFalse(report["shared_ports_capacity_or_concurrency_verified"])
        self.assertIn("NOT PASSED", report["build_gate"])

    def test_shared_traffic_separates_ports(self):
        traffic = r.ss_shared_bytes_per_expert()
        # Expanded A (64 useful rows) and padded B (32/64 useful rows) each
        # write 64 MiB and each reread 64 MiB, per touched expert.
        self.assertEqual(traffic["generic_decoded_writes"], 128 * 2**20)
        self.assertEqual(traffic["tensorcore_weight_reads"], 128 * 2**20)
        self.assertEqual(traffic["generic_reads_and_writes"], 155 * 2**20)
        self.assertEqual(traffic["all_weight_ports_sum"], 310 * 2**20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
