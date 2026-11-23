"""Create paired uncertainty and seed-variability tables from saved evaluation results."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read(path: Path) -> list[dict]:
    """Read saved CSV cells without importing training or simulation code."""
    with path.open() as stream:

        return list(csv.DictReader(stream))


def _write(rows: list[dict], path: Path) -> None:
    """Write a nonempty report table with consistent columns."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _forecast_intervals(rows: list[dict], *, seed: int = 3, resamples: int = 2000) -> list[dict]:
    """Compare LSTM minus MLP error, pairing days and retaining both exploration trajectories."""
    targets = [key.removeprefix("predicted_") for key in rows[0] if key.startswith("predicted_")]
    days = sorted({r["scenario_id"] for r in rows})
    labels = sorted({r["manager"] for r in rows})
    indices = np.random.default_rng(seed).integers(len(days), size=(resamples, len(days)))
    result = []

    for target in targets:
        if not any(r[f"observed_{target}"] != "" for r in rows):
            continue

        aggregates = defaultdict(lambda: np.zeros(3))

        for row in rows:
            observed = row[f"observed_{target}"]

            if observed != "":
                error = float(row[f"predicted_{target}"]) - float(observed)
                aggregates[row["manager"], row["scenario_id"]] += (abs(error), error * error, 1)

        for metric, column in (("mae", 0), ("rmse", 1)):
            estimates, bootstrap = {}, {}

            for label in labels:
                values = np.array([aggregates[label, day] for day in days])
                estimate = values[:, column].sum() / values[:, 2].sum()
                sampled = values[indices].sum(axis=1)
                draws = np.divide(sampled[:, column], sampled[:, 2], out=np.full(resamples, np.nan), where=sampled[:, 2] > 0)

                if metric == "rmse":
                    estimate, draws = np.sqrt(estimate), np.sqrt(draws)

                estimates[label], bootstrap[label] = estimate, draws

            # Average each model's error metric across seeds, not its predictions.
            for mode in ("mean_per_day", "seed-0", "seed-1", "seed-2"):
                mlp = [label for label in labels if label.startswith("mlp-") and (mode == "mean_per_day" or label.endswith(mode))]
                lstm = [label for label in labels if label.startswith("lstm-") and (mode == "mean_per_day" or label.endswith(mode))]
                difference = np.mean([estimates[k] for k in lstm]) - np.mean([estimates[k] for k in mlp])
                draws = np.mean([bootstrap[k] for k in lstm], axis=0) - np.mean([bootstrap[k] for k in mlp], axis=0)
                lower, upper = np.nanquantile(draws, [.025, .975])
                result.append({"target": target, "metric": metric, "seed_mode": "mean_across_seeds" if mode == "mean_per_day" else mode,
                               "comparison": "lstm-minus-mlp", "difference": float(difference),
                               "ci_lower": float(lower), "ci_upper": float(upper), "days": len(days)})

    return result


def _business_intervals(rows: list[dict], *, seed: int = 3, resamples: int = 2000) -> list[dict]:
    """Pair family-average day outcomes, keeping seed replicates inside each day."""
    days = sorted({r["scenario_id"] for r in rows})
    indices = np.random.default_rng(seed).integers(len(days), size=(resamples, len(days)))
    fields = ("profit", "sales", "abandonment", "waste_units", "waste_cost", "preparation_cost", "staff_cost")
    results = []

    for field in fields:
        values = defaultdict(list)

        for row in rows:
            values[row["manager"].split("-seed-")[0], row["scenario_id"]].append(float(row[field]))

        by_family = {family: np.array([np.mean(values[family, day]) for day in days])
                     for family in ("fixed", "adaptive", "mlp", "lstm")}

        for family, baseline in (("mlp", "fixed"), ("mlp", "adaptive"), ("lstm", "fixed"),
                                 ("lstm", "adaptive"), ("lstm", "mlp")):
            differences = by_family[family] - by_family[baseline]
            lower, upper = np.quantile(differences[indices].mean(axis=1), [.025, .975])
            results.append({"manager": family, "baseline": baseline, "metric": field,
                            "mean_daily_difference": float(differences.mean()),
                            "ci_lower": float(lower), "ci_upper": float(upper), "days": len(days)})

    return results


def select_representative_day(day_rows: list[dict], *, reference_manager: str = "adaptive") -> str:
    """Choose the day nearest median adaptive profit, breaking ties by scenario ID."""
    rows = [r for r in day_rows if r["manager"] == reference_manager]
    median = np.median([float(r["profit"]) for r in rows])

    return min(rows, key=lambda r: (abs(float(r["profit"]) - median), r["scenario_id"]))["scenario_id"]


def build_report(results_dir: Path, output_dir: Path) -> list[Path]:
    """Write forecast paired intervals, seed variability, static plots and an interpretation draft."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=False)
    import hashlib
    import platform

    provenance = {"package_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                        "matplotlib": matplotlib.__version__},
                  "bootstrap_seed": 3, "resamples": 2000, "confidence": .95,
                  "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(results_dir.iterdir()) if path.suffix in (".csv", ".json")}}
    (output_dir / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary = _read(results_dir / "summary.csv")
    forecasts = _read(results_dir / "forecast_metrics.csv")
    predictions = [r for r in _read(results_dir / "forecasts.csv") if r["split"] == "test"]
    paired = _read(results_dir / "paired_profit.csv")
    days = _read(results_dir / "days.csv")
    periods = _read(results_dir / "periods.csv")
    protocol = json.loads((results_dir / "protocol.json").read_text())
    business_intervals = _business_intervals(days)
    _write(business_intervals, output_dir / "paired_business.csv")
    forecast_intervals = _forecast_intervals(predictions)
    _write(forecast_intervals, output_dir / "paired_forecasts.csv")
    variability = []

    for source, rows in (("business", summary), ("forecast", [r for r in forecasts if r["split"] == "test"])):
        for family in ("mlp", "lstm"):
            selected = [r for r in rows if r["manager"].startswith(family + "-seed-")]

            for field in selected[0]:
                if not (field.startswith(("mean_", "mae_", "rmse_")) or field in ("served_mean_wait", "violation_rate", "fallback_rate")):
                    continue

                values = [float(r[field]) for r in selected if r[field] != ""]

                if values:
                    variability.append({"source": source, "manager": family, "metric": field, "seeds": len(values),
                                        "seed_mean": float(np.mean(values)), "seed_std": float(np.std(values, ddof=1)),
                                        "seed_min": min(values), "seed_max": max(values)})

    _write(variability, output_dir / "seed_variability.csv")
    text = ["# Three-seed held-out comparison", "",
            f"{protocol['independent_test_days']} independent test days; all six neural checkpoints retained. "
            "Fixed/adaptive settings and neural candidate ranking were frozen on validation before test evaluation.", "",
            "95% percentile bootstrap intervals use 2,000 paired scenario-day resamples. "
            "All policies, seeds, periods and exploratory trajectories belonging to a day stay together. "
            "Intervals quantify day variability conditional on these three trained seeds; seed SD is reported separately. "
            "Forecast comparisons average seed-specific MAE/RMSE, not ensemble predictions. Negative forecast differences favour LSTM.", "",
            "| Manager | Daily profit (HKD) | Day SD | Abandonment/period | Violation rate | Served wait (min) |", "|---|---:|---:|---:|---:|---:|"]

    for row in summary:
        wait = f"{float(row['served_mean_wait']):.3f}" if row["served_mean_wait"] else "undefined"
        text.append(f"| {row['manager']} | {float(row['mean_daily_profit']):.2f} | {float(row['std_daily_profit']):.2f} | "
                    f"{float(row['mean_period_abandonment']):.3f} | {float(row['violation_rate']):.2%} | {wait} |")

    text.extend(["", "| Family | Baseline | Daily profit difference (HKD) | 95% paired CI |", "|---|---|---:|---:|"])

    for row in paired:
        if row["seed_mode"] == "mean_per_day" and row["manager"] in ("mlp", "lstm"):
            text.append(f"| {row['manager']} | {row['baseline']} | {float(row['mean_profit_difference']):.2f} | "
                        f"[{float(row['ci_lower']):.2f}, {float(row['ci_upper']):.2f}] |")

    direct = next(r for r in business_intervals if r["manager"] == "lstm" and r["baseline"] == "mlp" and r["metric"] == "profit")
    text.append(f"| lstm | mlp | {direct['mean_daily_difference']:.2f} | [{direct['ci_lower']:.2f}, {direct['ci_upper']:.2f}] |")
    text.extend(["", "Wait is total served waiting divided by total purchases. Daily abandonment fraction is abandonment / entries; "
                 "the summary explicitly counts valid denominators. Zero-sales periods retain all costs. "
                 "A mean abandonment criterion does not guarantee every period meets the limit of five.", "",
                 "See forecast_metrics.csv for every outcome and seed, paired_forecasts.csv for LSTM–MLP error intervals, "
                 "paired_business.csv for paired daily sales, abandonment, waste and cost differences, "
                 "seed_variability.csv for across-seed SD/range, summary.csv for sales, waste, costs and action frequencies, "
                 "and complexity.csv for measured full-selection CPU latency and training time.", ""])
    text.extend(["", "| Family | Mean daily profit (HKD) | SD across three seed means | Mean violation rate |",
                 "|---|---:|---:|---:|"])

    for family in ("mlp", "lstm"):
        profit = next(r for r in variability if r["manager"] == family and r["metric"] == "mean_daily_profit")
        violations = next(r for r in variability if r["manager"] == family and r["metric"] == "violation_rate")
        text.append(f"| {family} | {profit['seed_mean']:.2f} | {profit['seed_std']:.2f} | {violations['seed_mean']:.2%} |")

    text.extend(["", "| Outcome | MLP MAE ± seed SD | LSTM MAE ± seed SD | LSTM−MLP MAE: 95% paired CI |",
                 "|---|---:|---:|---:|"])

    for comparison in forecast_intervals:
        if comparison["metric"] != "mae" or comparison["seed_mode"] != "mean_across_seeds":
            continue

        field = "mae_" + comparison["target"]
        mlp = next(r for r in variability if r["manager"] == "mlp" and r["metric"] == field)
        lstm = next(r for r in variability if r["manager"] == "lstm" and r["metric"] == field)
        text.append(f"| {comparison['target']} | {mlp['seed_mean']:.4f} ± {mlp['seed_std']:.4f} | "
                    f"{lstm['seed_mean']:.4f} ± {lstm['seed_std']:.4f} | [{comparison['ci_lower']:.4f}, {comparison['ci_upper']:.4f}] |")

    text.extend(["", "| Outcome | MLP RMSE ± seed SD | LSTM RMSE ± seed SD | LSTM−MLP RMSE: 95% paired CI |",
                 "|---|---:|---:|---:|"])

    for comparison in forecast_intervals:
        if comparison["metric"] != "rmse" or comparison["seed_mode"] != "mean_across_seeds":
            continue

        field = "rmse_" + comparison["target"]
        mlp = next(r for r in variability if r["manager"] == "mlp" and r["metric"] == field)
        lstm = next(r for r in variability if r["manager"] == "lstm" and r["metric"] == field)
        text.append(f"| {comparison['target']} | {mlp['seed_mean']:.4f} ± {mlp['seed_std']:.4f} | "
                    f"{lstm['seed_mean']:.4f} ± {lstm['seed_std']:.4f} | [{comparison['ci_lower']:.4f}, {comparison['ci_upper']:.4f}] |")

    text.extend(["", "Sales/abandonment errors are counts per period; wait errors are minutes. "
                 "The intervals are descriptive, with no multiple-comparison correction. "
                 "Three training seeds are insufficient to establish general architectural superiority.", ""])
    (output_dir / "interpretation.md").write_text("\n".join(text))
    labels = [r["manager"] for r in summary]
    fig, ax = plt.subplots(figsize=(10, 5))
    reference = {r["scenario_id"]: float(r["profit"]) for r in days if r["manager"] == "adaptive"}
    differences = [[float(r["profit"]) - reference[r["scenario_id"]] for r in days if r["manager"] == label] for label in labels]
    ax.boxplot(differences, tick_labels=labels)
    ax.axhline(0, color="black", linewidth=.7)
    ax.set_ylabel("Daily profit minus adaptive (HKD)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_profit.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))

    for row in summary:
        x, y = float(row["violation_rate"]), float(row["mean_daily_profit"])
        ax.scatter(x, y, label=row["manager"])

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    ax.set(xlabel="Fraction of periods with abandonment > 5", ylabel="Mean daily profit (HKD)")
    fig.tight_layout()
    fig.savefig(output_dir / "profit_service.png", dpi=150)
    plt.close(fig)
    representative = select_representative_day(days)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    plotted = ["fixed", "adaptive"] + [f"{family}-seed-{protocol['selected_on_validation'][family]['settings']['seed']}" for family in ("mlp", "lstm")]

    for label in plotted:
        rows = sorted((r for r in periods if r["manager"] == label and r["scenario_id"] == representative), key=lambda r: int(r["period_index"]))
        axes[0].plot([int(r["period_index"]) for r in rows], [float(r["profit"]) for r in rows], label=label)
        axes[1].plot([int(r["period_index"]) for r in rows], [float(r["queue_A"]) for r in rows], label=label)

    axes[0].set(title=f"{representative}: nearest median adaptive profit; validation-selected neural seeds", ylabel="Period profit (HKD)")
    axes[1].set(xlabel="Period", ylabel="Opening queue A")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "representative_day.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    targets = [k.removeprefix("predicted_") for k in predictions[0] if k.startswith("predicted_")]

    for i, family in enumerate(("mlp", "lstm")):
        label = f"{family}-seed-{protocol['selected_on_validation'][family]['settings']['seed']}"
        rows = [r for r in predictions if r["manager"] == label]

        for j, fields in enumerate((targets[:3], [targets[3]])):
            actual = [sum(float(r[f"observed_{t}"]) for t in fields) for r in rows]
            predicted = [sum(float(r[f"predicted_{t}"]) for t in fields) for r in rows]
            axes[i, j].hexbin(actual, predicted, gridsize=25, mincnt=1)
            maximum = max(max(actual), max(predicted))
            axes[i, j].plot([0, maximum], [0, maximum], color="red", linewidth=.8)
            axes[i, j].set(title=f"{label}: {'total sales' if j == 0 else 'abandonment'}", xlabel="Observed", ylabel="Predicted")

    fig.tight_layout()
    fig.savefig(output_dir / "forecasts.png", dpi=150)
    plt.close(fig)

    return sorted(output_dir.iterdir())
