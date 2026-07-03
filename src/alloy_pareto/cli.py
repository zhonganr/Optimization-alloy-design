from __future__ import annotations

import argparse

from alloy_pareto.bruteforce import brute_force_search
from alloy_pareto.evaluation import AlloyEvaluator
from alloy_pareto.io import format_summary, write_csv
from alloy_pareto.nsga2 import NSGA2Config, run_nsga2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prototype Pareto optimization for nickel-base alloy design")
    parser.add_argument("--mode", choices=("ga", "brute-force"), default="ga")
    parser.add_argument(
        "--data-source",
        choices=("placeholder", "paper"),
        default="placeholder",
        help="Use placeholder formulas or paper-extracted density/search bounds",
    )
    parser.add_argument("--population", type=int, default=100, help="NSGA-II population size")
    parser.add_argument("--generations", type=int, default=150, help="NSGA-II generation count")
    parser.add_argument("--crossover-probability", type=float, default=0.7)
    parser.add_argument("--mutation-probability", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--step", type=float, default=5.0, help="Brute-force grid step in wt%%")
    parser.add_argument("--max-evaluations", type=int, default=200_000, help="Brute-force evaluation cap")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    parser.add_argument("--limit", type=int, default=10, help="Number of Pareto rows printed")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluator = AlloyEvaluator(data_source=args.data_source)

    if args.mode == "ga":
        config = NSGA2Config(
            population_size=args.population,
            generations=args.generations,
            crossover_probability=args.crossover_probability,
            mutation_probability=args.mutation_probability,
            seed=args.seed,
        )
        alloys = run_nsga2(evaluator, config)
    else:
        alloys = brute_force_search(evaluator, step=args.step, max_evaluations=args.max_evaluations)

    alloys = sorted(alloys, key=lambda alloy: alloy.objectives["cost_usd_per_kg"])
    print(format_summary(alloys, limit=args.limit))
    if args.output:
        write_csv(args.output, alloys)
        print(f"Wrote CSV: {args.output}")


if __name__ == "__main__":
    main()
