# Multi-Criteria Alloy Pareto Prototype

This prototype implements the optimization workflow described in `paper1.pdf` and the exhaustive screening idea from `paper2.pdf`:

- **NSGA-II genetic algorithm** for multi-objective alloy design.
- **Constraint-aware Pareto ranking** where feasible alloys dominate infeasible ones.
- **Brute-force enumeration** for small composition spaces or coarse grids.
- **Swappable surrogate evaluator** so CALPHAD, Thermo-Calc, Gaussian-process, or lab models can replace the included demo heuristics.

The included evaluator is intentionally lightweight and illustrative. It approximates nickel-base superalloy objectives such as cost, density, creep-strength proxy, lattice-misfit proxy, and microstructural constraints. It is not a validated materials model.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
alloy-pareto --mode ga --population 80 --generations 80 --seed 7 --output results/pareto_ga.csv
```

Without installing the package:

```bash
PYTHONPATH=src python -m alloy_pareto --mode ga --population 80 --generations 80 --seed 7
```

Run a coarse brute-force baseline:

```bash
PYTHONPATH=src python -m alloy_pareto --mode brute-force --step 5 --max-evaluations 200000 --output results/pareto_bruteforce.csv
```

Choose the evaluator data source:

```bash
PYTHONPATH=src python -m alloy_pareto --mode ga --data-source placeholder
PYTHONPATH=src python -m alloy_pareto --mode ga --data-source paper
```

`placeholder` uses the original toy formulas and broad prototype bounds. `paper` uses numerical data extracted from the papers where it is directly reusable: Paper2 Table 3 density coefficients and compatible Paper2 Table 4 search bounds. Creep strength, cost, BTR, TCP risk, and corrosion proxies remain placeholders because the PDFs do not include the complete raw training databases or trained Gaussian-process models.

Run tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Model Structure

Decision variables are alloying-element wt% values. Nickel is automatically computed as the balance:

```text
Ni = 100 - sum(alloying elements)
```

Default decision elements are:

```text
Al, Ti, Cr, Co, Mo, W, Nb, Ta
```

Default objectives:

- maximize `creep_strength_proxy_mpa`
- minimize `cost_usd_per_kg`
- minimize `density_g_cm3`
- minimize `misfit_abs_pct`

Default constraints:

- `Ni` must stay above a minimum balance.
- free matrix chromium proxy must exceed a chromia-forming threshold.
- gamma-prime proxy must stay in a target window.
- TCP and BTR/weldability risk proxies must remain below thresholds.

## Replacing Demo Heuristics

The main integration point is `alloy_pareto.evaluation.AlloyEvaluator`. Replace its objective and constraint functions with calls to validated models, for example:

- CALPHAD/Thermo-Calc for phase constitution and stability constraints.
- Gaussian-process or neural-network regressors for creep/yield/oxidation properties.
- Internal cost and supply-risk models.

The optimizer only requires each candidate to produce:

- a composition dictionary;
- objective values;
- constraint penalties.

## Output

The CLI prints a compact Pareto-front summary and can write CSV rows containing composition, objectives, constraints, feasibility, and penalties.
