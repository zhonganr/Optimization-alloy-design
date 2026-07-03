import unittest

from alloy_pareto.evaluation import AlloyEvaluator
from alloy_pareto.models import EvaluatedAlloy
from alloy_pareto.nsga2 import NSGA2Config, run_nsga2
from alloy_pareto.pareto import objective_dominates, pareto_front


def alloy(strength: float, cost: float, penalty: float = 0.0) -> EvaluatedAlloy:
    return EvaluatedAlloy(
        composition={"Ni": 100.0 - cost, "Cr": cost},
        objectives={"strength": strength, "cost": cost},
        objective_min_values=(-strength, cost),
        constraints={"demo": 0.0},
        penalties={"demo": penalty},
    )


class ParetoTests(unittest.TestCase):
    def test_objective_dominance_uses_minimized_values(self):
        strong_and_cheap = alloy(200.0, 10.0)
        weak_and_expensive = alloy(150.0, 20.0)
        self.assertTrue(objective_dominates(strong_and_cheap, weak_and_expensive))

    def test_pareto_front_excludes_dominated_and_infeasible(self):
        front = pareto_front([alloy(200.0, 20.0), alloy(150.0, 10.0), alloy(120.0, 30.0), alloy(250.0, 5.0, 1.0)])
        self.assertEqual(len(front), 2)
        self.assertTrue(all(candidate.feasible for candidate in front))

    def test_nsga2_smoke_run_returns_feasible_front(self):
        result = run_nsga2(AlloyEvaluator(), NSGA2Config(population_size=20, generations=5, seed=3))
        self.assertGreater(len(result), 0)
        self.assertTrue(all(alloy.feasible for alloy in result))


if __name__ == "__main__":
    unittest.main()
