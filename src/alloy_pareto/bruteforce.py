from __future__ import annotations

from itertools import product

from alloy_pareto.evaluation import AlloyEvaluator
from alloy_pareto.models import EvaluatedAlloy
from alloy_pareto.pareto import pareto_front


def _grid_values(lower: float, upper: float, step: float) -> list[float]:
    values: list[float] = []
    current = lower
    while current <= upper + 1e-12:
        values.append(round(min(current, upper), 10))
        current += step
    if values[-1] != upper:
        values.append(upper)
    return values


def brute_force_search(
    evaluator: AlloyEvaluator | None = None,
    step: float = 5.0,
    max_evaluations: int | None = 200_000,
) -> list[EvaluatedAlloy]:
    if step <= 0.0:
        raise ValueError("step must be positive")

    evaluator = evaluator or AlloyEvaluator()
    grids = [_grid_values(bound.lower, bound.upper, step) for bound in evaluator.space.bounds]
    evaluated: list[EvaluatedAlloy] = []

    for evaluation_count, values in enumerate(product(*grids), start=1):
        if max_evaluations is not None and evaluation_count > max_evaluations:
            break
        decision = {bound.name: value for bound, value in zip(evaluator.space.bounds, values)}
        alloy = evaluator.evaluate(decision)
        if alloy.feasible:
            evaluated.append(alloy)

    return pareto_front(evaluated)
