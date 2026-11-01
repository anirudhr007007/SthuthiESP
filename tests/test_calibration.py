import unittest

from tools import calibrate_threshold


class CalibrationMetricTests(unittest.TestCase):
    def test_rates_use_inclusive_false_accept_threshold(self):
        genuine = [0.90, 0.75]
        impostor = [0.20, 0.75]
        far, frr = calibrate_threshold._rates_at(genuine, impostor, 0.80)
        self.assertEqual(far, 0.0)
        self.assertEqual(frr, 0.5)

    def test_threshold_table_covers_the_expected_sweep(self):
        table = calibrate_threshold._threshold_table(
            [0.91, 0.88, 0.86, 0.84],
            [0.25, 0.35, 0.70, 0.78],
        )
        self.assertEqual(len(table), 11)
        self.assertEqual(table[0]["threshold"], 0.4)
        self.assertEqual(table[-1]["threshold"], 0.9)
        for row in table:
            self.assertGreaterEqual(row["false_accept_rate"], 0.0)
            self.assertLessEqual(row["false_accept_rate"], 1.0)
            self.assertGreaterEqual(row["false_reject_rate"], 0.0)
            self.assertLessEqual(row["false_reject_rate"], 1.0)

    def test_empty_genuine_scores_are_all_rejections(self):
        far, frr = calibrate_threshold._rates_at([], [0.2, 0.4], 0.6)
        self.assertEqual(far, 0.0)
        self.assertEqual(frr, 1.0)


if __name__ == "__main__":
    unittest.main()
