import unittest

from alloy_pareto.evaluation import AlloyEvaluator, paper2_density_g_cm3


class DataSourceTests(unittest.TestCase):
    def test_paper_mode_uses_paper_search_bounds(self):
        evaluator = AlloyEvaluator(data_source="paper")
        bounds = {bound.name: (bound.lower, bound.upper) for bound in evaluator.space.bounds}
        self.assertEqual(bounds["Al"], (4.0, 8.0))
        self.assertEqual(bounds["Ta"], (5.0, 15.0))

    def test_paper_density_model_is_used(self):
        evaluator = AlloyEvaluator(data_source="paper")
        alloy = evaluator.evaluate({"Al": 6.0, "Co": 4.0, "Cr": 10.0, "Mo": 1.0, "Nb": 1.0, "Ta": 8.0, "Ti": 1.0, "W": 2.0})
        self.assertEqual(alloy.metadata["data_source"], "paper")
        self.assertAlmostEqual(alloy.objectives["density_g_cm3"], paper2_density_g_cm3(alloy.composition))

    def test_invalid_data_source_raises(self):
        with self.assertRaises(ValueError):
            AlloyEvaluator(data_source="unknown")


if __name__ == "__main__":
    unittest.main()
