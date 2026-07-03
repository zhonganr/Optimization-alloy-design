import unittest

from alloy_pareto.bruteforce import _grid_values


class BruteForceTests(unittest.TestCase):
    def test_grid_values_never_exceed_upper_bound(self):
        values = _grid_values(12.0, 25.0, 8.0)
        self.assertEqual(values, [12.0, 20.0, 25.0])
        self.assertLessEqual(max(values), 25.0)


if __name__ == "__main__":
    unittest.main()
