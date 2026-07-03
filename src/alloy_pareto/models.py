from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal


Direction = Literal["min", "max"]


@dataclass(frozen=True)
class ElementBound:
    name: str
    lower: float
    upper: float


@dataclass(frozen=True)
class Objective:
    name: str
    direction: Direction
    evaluate: Callable[[dict[str, float]], float]

    def as_minimization(self, value: float) -> float:
        return value if self.direction == "min" else -value


@dataclass(frozen=True)
class Constraint:
    name: str
    evaluate: Callable[[dict[str, float]], float]
    lower: float | None = None
    upper: float | None = None

    def penalty(self, composition: dict[str, float]) -> float:
        value = self.evaluate(composition)
        below = 0.0 if self.lower is None else max(0.0, self.lower - value)
        above = 0.0 if self.upper is None else max(0.0, value - self.upper)
        return below + above


@dataclass
class EvaluatedAlloy:
    composition: dict[str, float]
    objectives: dict[str, float]
    objective_min_values: tuple[float, ...]
    constraints: dict[str, float]
    penalties: dict[str, float]
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def total_penalty(self) -> float:
        return sum(self.penalties.values())

    @property
    def violation_count(self) -> int:
        return sum(1 for penalty in self.penalties.values() if penalty > 0.0)

    @property
    def feasible(self) -> bool:
        return self.violation_count == 0


@dataclass(frozen=True)
class SearchSpace:
    bounds: tuple[ElementBound, ...]
    balance_element: str = "Ni"
    balance_minimum: float = 45.0

    @property
    def decision_elements(self) -> tuple[str, ...]:
        return tuple(bound.name for bound in self.bounds)
