from __future__ import annotations

import csv
from pathlib import Path

from alloy_pareto.models import EvaluatedAlloy


def alloy_rows(alloys: list[EvaluatedAlloy]) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    for idx, alloy in enumerate(alloys, start=1):
        row: dict[str, float | str | bool] = {"rank": idx, "feasible": alloy.feasible, "total_penalty": alloy.total_penalty}
        row.update({f"wt_{name}": round(value, 6) for name, value in sorted(alloy.composition.items())})
        row.update({name: round(value, 6) for name, value in alloy.objectives.items()})
        row.update({f"constraint_{name}": round(value, 6) for name, value in alloy.constraints.items()})
        row.update({f"penalty_{name}": round(value, 6) for name, value in alloy.penalties.items()})
        rows.append(row)
    return rows


def write_csv(path: str | Path, alloys: list[EvaluatedAlloy]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = alloy_rows(alloys)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_summary(alloys: list[EvaluatedAlloy], limit: int = 10) -> str:
    if not alloys:
        return "No feasible Pareto-optimal alloys found."

    lines = [f"Found {len(alloys)} feasible Pareto-optimal alloys."]
    for idx, alloy in enumerate(alloys[:limit], start=1):
        composition = ", ".join(f"{name}={value:.2f}" for name, value in sorted(alloy.composition.items()))
        objectives = ", ".join(f"{name}={value:.3f}" for name, value in alloy.objectives.items())
        lines.append(f"{idx:>2}. {objectives} | {composition}")
    if len(alloys) > limit:
        lines.append(f"... {len(alloys) - limit} more rows not shown")
    return "\n".join(lines)
