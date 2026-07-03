from __future__ import annotations

from math import inf

from alloy_pareto.models import EvaluatedAlloy


def objective_dominates(left: EvaluatedAlloy, right: EvaluatedAlloy) -> bool:
    no_worse = all(a <= b for a, b in zip(left.objective_min_values, right.objective_min_values))
    strictly_better = any(a < b for a, b in zip(left.objective_min_values, right.objective_min_values))
    return no_worse and strictly_better


def constrained_dominates(left: EvaluatedAlloy, right: EvaluatedAlloy) -> bool:
    if left.feasible and not right.feasible:
        return True
    if right.feasible and not left.feasible:
        return False
    if left.feasible and right.feasible:
        return objective_dominates(left, right)
    if left.violation_count != right.violation_count:
        return left.violation_count < right.violation_count
    if left.total_penalty != right.total_penalty:
        return left.total_penalty < right.total_penalty
    return objective_dominates(left, right)


def non_dominated_sort(population: list[EvaluatedAlloy]) -> list[list[EvaluatedAlloy]]:
    dominates_map: dict[int, list[int]] = {idx: [] for idx in range(len(population))}
    dominated_count = {idx: 0 for idx in range(len(population))}
    fronts: list[list[int]] = [[]]

    for left_idx, left in enumerate(population):
        for right_idx, right in enumerate(population):
            if left_idx == right_idx:
                continue
            if constrained_dominates(left, right):
                dominates_map[left_idx].append(right_idx)
            elif constrained_dominates(right, left):
                dominated_count[left_idx] += 1
        if dominated_count[left_idx] == 0:
            fronts[0].append(left_idx)

    current = 0
    while fronts[current]:
        next_front: list[int] = []
        for left_idx in fronts[current]:
            for right_idx in dominates_map[left_idx]:
                dominated_count[right_idx] -= 1
                if dominated_count[right_idx] == 0:
                    next_front.append(right_idx)
        current += 1
        fronts.append(next_front)

    return [[population[idx] for idx in front] for front in fronts if front]


def crowding_distance(front: list[EvaluatedAlloy]) -> dict[int, float]:
    if not front:
        return {}
    distances = {idx: 0.0 for idx in range(len(front))}
    objective_count = len(front[0].objective_min_values)

    for objective_idx in range(objective_count):
        ordered = sorted(range(len(front)), key=lambda idx: front[idx].objective_min_values[objective_idx])
        distances[ordered[0]] = inf
        distances[ordered[-1]] = inf
        min_value = front[ordered[0]].objective_min_values[objective_idx]
        max_value = front[ordered[-1]].objective_min_values[objective_idx]
        if max_value == min_value:
            continue
        for position in range(1, len(ordered) - 1):
            previous_value = front[ordered[position - 1]].objective_min_values[objective_idx]
            next_value = front[ordered[position + 1]].objective_min_values[objective_idx]
            distances[ordered[position]] += (next_value - previous_value) / (max_value - min_value)

    return distances


def select_next_population(population: list[EvaluatedAlloy], size: int) -> list[EvaluatedAlloy]:
    selected: list[EvaluatedAlloy] = []
    for front in non_dominated_sort(population):
        remaining = size - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
            continue
        distances = crowding_distance(front)
        ranked = sorted(range(len(front)), key=lambda idx: distances[idx], reverse=True)
        selected.extend(front[idx] for idx in ranked[:remaining])
        break
    return selected


def unique_alloys(population: list[EvaluatedAlloy], digits: int = 8) -> list[EvaluatedAlloy]:
    seen: set[tuple[tuple[str, float], ...]] = set()
    unique: list[EvaluatedAlloy] = []
    for alloy in population:
        signature = tuple(sorted((name, round(value, digits)) for name, value in alloy.composition.items()))
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(alloy)
    return unique


def pareto_front(population: list[EvaluatedAlloy], feasible_only: bool = True) -> list[EvaluatedAlloy]:
    candidates = [alloy for alloy in population if alloy.feasible] if feasible_only else list(population)
    candidates = unique_alloys(candidates)
    if not candidates:
        return []
    return non_dominated_sort(candidates)[0]
