from __future__ import annotations

from alloy_pareto.models import Constraint, ElementBound, EvaluatedAlloy, Objective, SearchSpace


DataSource = str


ATOMIC_MASS = {
    "Ni": 58.693,
    "Al": 26.982,
    "Ti": 47.867,
    "Cr": 51.996,
    "Co": 58.933,
    "Mo": 95.95,
    "W": 183.84,
    "Nb": 92.906,
    "Ta": 180.948,
}

DENSITY = {
    "Ni": 8.90,
    "Al": 2.70,
    "Ti": 4.51,
    "Cr": 7.19,
    "Co": 8.90,
    "Mo": 10.28,
    "W": 19.25,
    "Nb": 8.57,
    "Ta": 16.69,
}

PRICE_USD_PER_KG = {
    "Ni": 18.0,
    "Al": 2.5,
    "Ti": 12.0,
    "Cr": 4.0,
    "Co": 35.0,
    "Mo": 45.0,
    "W": 40.0,
    "Nb": 55.0,
    "Ta": 180.0,
}

PAPER2_DENSITY_A_I_TIMES_1000 = {
    "Ni": -1.49,
    "Al": 61.83,
    "Co": 3.66,
    "Cr": 1.21,
    "Mo": -18.35,
    "Nb": 53.27,
    "Ta": 10.81,
    "Ti": 48.58,
    "W": 4.57,
}


def default_search_space() -> SearchSpace:
    return SearchSpace(
        bounds=(
            ElementBound("Al", 1.0, 6.0),
            ElementBound("Ti", 0.0, 4.0),
            ElementBound("Cr", 12.0, 25.0),
            ElementBound("Co", 0.0, 20.0),
            ElementBound("Mo", 0.0, 8.0),
            ElementBound("W", 0.0, 8.0),
            ElementBound("Nb", 0.0, 3.0),
            ElementBound("Ta", 0.0, 8.0),
        ),
        balance_element="Ni",
        balance_minimum=45.0,
    )


def paper2_compatible_search_space() -> SearchSpace:
    return SearchSpace(
        bounds=(
            ElementBound("Al", 4.0, 8.0),
            ElementBound("Co", 0.0, 14.0),
            ElementBound("Cr", 4.0, 15.0),
            ElementBound("Mo", 0.0, 3.0),
            ElementBound("Nb", 0.0, 3.0),
            ElementBound("Ta", 5.0, 15.0),
            ElementBound("Ti", 0.0, 3.0),
            ElementBound("W", 0.0, 10.0),
        ),
        balance_element="Ni",
        balance_minimum=45.0,
    )


def complete_with_balance(decision_values: dict[str, float], space: SearchSpace) -> dict[str, float]:
    composition = {name: max(0.0, value) for name, value in decision_values.items()}
    composition[space.balance_element] = 100.0 - sum(composition.values())
    return composition


def atomic_percent(composition: dict[str, float], element: str) -> float:
    moles = {name: wt / ATOMIC_MASS[name] for name, wt in composition.items() if wt > 0.0}
    total_moles = sum(moles.values())
    if total_moles == 0.0:
        return 0.0
    return 100.0 * moles.get(element, 0.0) / total_moles


def cost_usd_per_kg(composition: dict[str, float]) -> float:
    return sum(composition[name] * PRICE_USD_PER_KG[name] / 100.0 for name in composition)


def density_g_cm3(composition: dict[str, float]) -> float:
    specific_volume = sum((wt / 100.0) / DENSITY[name] for name, wt in composition.items())
    return 1.0 / specific_volume


def paper2_density_g_cm3(composition: dict[str, float]) -> float:
    denominator = 0.0
    for name, wt in composition.items():
        if wt <= 0.0:
            continue
        denominator += wt / DENSITY[name]
        denominator += (PAPER2_DENSITY_A_I_TIMES_1000.get(name, 0.0) * 1e-3) * wt
    return 100.0 / denominator


def gamma_prime_proxy_pct(composition: dict[str, float]) -> float:
    al_ti_ta = composition.get("Al", 0.0) + 0.75 * composition.get("Ti", 0.0) + 0.45 * composition.get("Ta", 0.0)
    nb_bonus = 0.25 * composition.get("Nb", 0.0)
    return max(0.0, min(75.0, 7.5 * al_ti_ta + 2.0 * nb_bonus))


def free_chromium_at_pct_proxy(composition: dict[str, float]) -> float:
    carbide_formers = composition.get("Ti", 0.0) + composition.get("Nb", 0.0) + 0.35 * composition.get("Ta", 0.0)
    effective_cr_wt = max(0.0, composition.get("Cr", 0.0) - 0.18 * carbide_formers)
    adjusted = dict(composition)
    adjusted["Cr"] = effective_cr_wt
    return atomic_percent(adjusted, "Cr")


def tcp_risk_proxy(composition: dict[str, float]) -> float:
    return (
        composition.get("Mo", 0.0) / 6.0
        + composition.get("W", 0.0) / 7.5
        + composition.get("Cr", 0.0) / 32.0
        + composition.get("Ta", 0.0) / 20.0
    )


def brittle_temperature_range_proxy_k(composition: dict[str, float]) -> float:
    return (
        35.0
        + 7.0 * composition.get("Nb", 0.0)
        + 4.0 * composition.get("Mo", 0.0)
        + 3.5 * composition.get("W", 0.0)
        + 2.0 * composition.get("Ta", 0.0)
    )


def lattice_misfit_abs_pct_proxy(composition: dict[str, float]) -> float:
    raw_misfit = (
        -0.08
        + 0.018 * composition.get("Al", 0.0)
        + 0.012 * composition.get("Ti", 0.0)
        - 0.006 * composition.get("Cr", 0.0)
        - 0.011 * composition.get("Mo", 0.0)
        - 0.009 * composition.get("W", 0.0)
        + 0.010 * composition.get("Ta", 0.0)
    )
    return abs(raw_misfit)


def creep_strength_proxy_mpa(composition: dict[str, float]) -> float:
    gamma_prime = gamma_prime_proxy_pct(composition)
    solid_solution = (
        5.5 * composition.get("Co", 0.0)
        + 12.0 * composition.get("Mo", 0.0)
        + 10.0 * composition.get("W", 0.0)
        + 9.0 * composition.get("Ta", 0.0)
        + 5.0 * composition.get("Nb", 0.0)
    )
    misfit_penalty = 180.0 * lattice_misfit_abs_pct_proxy(composition)
    tcp_penalty = 90.0 * max(0.0, tcp_risk_proxy(composition) - 1.0)
    gamma_window_penalty = 4.0 * max(0.0, gamma_prime - 55.0) + 3.0 * max(0.0, 25.0 - gamma_prime)
    return 110.0 + 7.0 * gamma_prime + solid_solution - misfit_penalty - tcp_penalty - gamma_window_penalty


class AlloyEvaluator:
    def __init__(
        self,
        space: SearchSpace | None = None,
        objectives: tuple[Objective, ...] | None = None,
        constraints: tuple[Constraint, ...] | None = None,
        data_source: DataSource = "placeholder",
    ) -> None:
        if data_source not in {"placeholder", "paper"}:
            raise ValueError("data_source must be either 'placeholder' or 'paper'")
        self.data_source = data_source
        self.space = space or (paper2_compatible_search_space() if data_source == "paper" else default_search_space())
        density_function = paper2_density_g_cm3 if data_source == "paper" else density_g_cm3
        self.objectives = objectives or (
            Objective("creep_strength_proxy_mpa", "max", creep_strength_proxy_mpa),
            Objective("cost_usd_per_kg", "min", cost_usd_per_kg),
            Objective("density_g_cm3", "min", density_function),
            Objective("misfit_abs_pct", "min", lattice_misfit_abs_pct_proxy),
        )
        self.constraints = constraints or (
            Constraint("ni_balance_wt_pct", lambda c: c.get(self.space.balance_element, 0.0), lower=self.space.balance_minimum),
            Constraint("free_cr_at_pct_proxy", free_chromium_at_pct_proxy, lower=15.0),
            Constraint("gamma_prime_pct_proxy", gamma_prime_proxy_pct, lower=20.0, upper=60.0),
            Constraint("tcp_risk_proxy", tcp_risk_proxy, upper=2.35),
            Constraint("btr_proxy_k", brittle_temperature_range_proxy_k, upper=115.0),
        )

    def evaluate(self, decision_values: dict[str, float]) -> EvaluatedAlloy:
        composition = complete_with_balance(decision_values, self.space)
        objectives = {objective.name: objective.evaluate(composition) for objective in self.objectives}
        objective_min_values = tuple(
            objective.as_minimization(objectives[objective.name]) for objective in self.objectives
        )
        constraint_values = {constraint.name: constraint.evaluate(composition) for constraint in self.constraints}
        penalties = {constraint.name: constraint.penalty(composition) for constraint in self.constraints}
        metadata = {
            "data_source": self.data_source,
            "gamma_prime_pct_proxy": gamma_prime_proxy_pct(composition),
            "free_cr_at_pct_proxy": free_chromium_at_pct_proxy(composition),
            "tcp_risk_proxy": tcp_risk_proxy(composition),
            "btr_proxy_k": brittle_temperature_range_proxy_k(composition),
        }
        return EvaluatedAlloy(composition, objectives, objective_min_values, constraint_values, penalties, metadata)
