from __future__ import annotations

import random
from dataclasses import dataclass

from alloy_pareto.evaluation import AlloyEvaluator
from alloy_pareto.models import ElementBound, EvaluatedAlloy, SearchSpace
from alloy_pareto.pareto import constrained_dominates, crowding_distance, non_dominated_sort, pareto_front, select_next_population


@dataclass(frozen=True)
class NSGA2Config:
    population_size: int = 100
    generations: int = 150
    crossover_probability: float = 0.7
    mutation_probability: float = 0.1
    sbx_eta: float = 15.0
    seed: int | None = None


def _random_decision(space: SearchSpace, rng: random.Random) -> dict[str, float]:
    values = {bound.name: rng.uniform(bound.lower, bound.upper) for bound in space.bounds}
    return _repair(values, space)


def _repair(values: dict[str, float], space: SearchSpace) -> dict[str, float]:
    repaired = {}
    for bound in space.bounds:
        repaired[bound.name] = min(bound.upper, max(bound.lower, values.get(bound.name, bound.lower)))

    max_alloying = 100.0 - space.balance_minimum
    total = sum(repaired.values())
    if total > max_alloying:
        lower_total = sum(bound.lower for bound in space.bounds)
        scalable_total = sum(repaired[bound.name] - bound.lower for bound in space.bounds)
        target_scalable_total = max(0.0, max_alloying - lower_total)
        scale = 0.0 if scalable_total == 0.0 else target_scalable_total / scalable_total
        for bound in space.bounds:
            repaired[bound.name] = bound.lower + (repaired[bound.name] - bound.lower) * scale
    return repaired


def _sbx_gene(left: float, right: float, bound: ElementBound, eta: float, rng: random.Random) -> tuple[float, float]:
    if rng.random() > 0.5 or abs(left - right) < 1e-12:
        return left, right
    lower, upper = bound.lower, bound.upper
    x1, x2 = sorted((left, right))
    rand = rng.random()

    beta = 1.0 + (2.0 * (x1 - lower) / (x2 - x1))
    alpha = 2.0 - beta ** -(eta + 1.0)
    if rand <= 1.0 / alpha:
        beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
    else:
        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
    child1 = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))

    beta = 1.0 + (2.0 * (upper - x2) / (x2 - x1))
    alpha = 2.0 - beta ** -(eta + 1.0)
    if rand <= 1.0 / alpha:
        beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
    else:
        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
    child2 = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

    child1 = min(upper, max(lower, child1))
    child2 = min(upper, max(lower, child2))
    if rng.random() <= 0.5:
        return child2, child1
    return child1, child2


def _crossover(
    left: dict[str, float],
    right: dict[str, float],
    space: SearchSpace,
    config: NSGA2Config,
    rng: random.Random,
) -> tuple[dict[str, float], dict[str, float]]:
    if rng.random() > config.crossover_probability:
        return dict(left), dict(right)
    child_left: dict[str, float] = {}
    child_right: dict[str, float] = {}
    for bound in space.bounds:
        child_left[bound.name], child_right[bound.name] = _sbx_gene(
            left[bound.name], right[bound.name], bound, config.sbx_eta, rng
        )
    return _repair(child_left, space), _repair(child_right, space)


def _mutate(values: dict[str, float], space: SearchSpace, config: NSGA2Config, rng: random.Random) -> dict[str, float]:
    mutated = dict(values)
    for bound in space.bounds:
        if rng.random() < config.mutation_probability:
            mutated[bound.name] = rng.uniform(bound.lower, bound.upper)
    return _repair(mutated, space)


def _binary_tournament(population: list[EvaluatedAlloy], rng: random.Random) -> EvaluatedAlloy:
    left, right = rng.sample(population, 2)
    if constrained_dominates(left, right):
        return left
    if constrained_dominates(right, left):
        return right
    front = [left, right]
    distances = crowding_distance(front)
    return left if distances[0] >= distances[1] else right


def _make_offspring(
    parents: list[EvaluatedAlloy], evaluator: AlloyEvaluator, config: NSGA2Config, rng: random.Random
) -> list[EvaluatedAlloy]:
    offspring: list[EvaluatedAlloy] = []
    while len(offspring) < config.population_size:
        parent_left = _binary_tournament(parents, rng).composition
        parent_right = _binary_tournament(parents, rng).composition
        left_values = {name: parent_left[name] for name in evaluator.space.decision_elements}
        right_values = {name: parent_right[name] for name in evaluator.space.decision_elements}
        child_left, child_right = _crossover(left_values, right_values, evaluator.space, config, rng)
        child_left = _mutate(child_left, evaluator.space, config, rng)
        child_right = _mutate(child_right, evaluator.space, config, rng)
        offspring.append(evaluator.evaluate(child_left))
        if len(offspring) < config.population_size:
            offspring.append(evaluator.evaluate(child_right))
    return offspring


def run_nsga2(evaluator: AlloyEvaluator | None = None, config: NSGA2Config | None = None) -> list[EvaluatedAlloy]:
    evaluator = evaluator or AlloyEvaluator()
    config = config or NSGA2Config()
    rng = random.Random(config.seed)

    parents = [evaluator.evaluate(_random_decision(evaluator.space, rng)) for _ in range(config.population_size)]
    archive: list[EvaluatedAlloy] = pareto_front(parents)

    for generation in range(config.generations):
        offspring = _make_offspring(parents, evaluator, config, rng)
        parents = select_next_population(parents + offspring, config.population_size)
        archive = pareto_front(archive + pareto_front(parents))
        if generation and generation % 25 == 0 and archive:
            injection_count = min(len(archive), max(1, config.population_size // 10))
            fronts = non_dominated_sort(parents)
            replaceable = [alloy for front in fronts[::-1] for alloy in front]
            injected = rng.sample(archive, injection_count) if len(archive) > injection_count else list(archive)
            parents = injected + replaceable[: config.population_size - len(injected)]

    return pareto_front(archive + parents)
